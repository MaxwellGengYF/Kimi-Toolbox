"""kimix_native.stream — stream kernels: ANSI strip + line processor.

Native implementations live in ``runtime_py.stream`` (compiled kernels, GIL
released). The pure-Python ``_compat`` functions below mirror the reference
algorithms exactly (src/kimix/tools/common.py):

- ``_compat_filter_output`` — `filter_output` (ANSI regex sub + CRLF normalize)
- ``_compat_dedup_output`` — `_dedup_output` (counter + block modes)

``LineProcessor`` wraps the native class; when the native extension is
disabled it falls back to ``_CompatLineProcessor``, a buffering Python
implementation with identical feed/flush semantics:

- feed() accepts str or bytes; returns completed lines as str (best effort —
  dedup modes buffer and only emit at flush()).
- flush() returns all remaining lines as str; ``"\\n".join(lines)`` is the
  processed output (matching ``filter_output`` / ``_dedup_output`` on LF /
  CRLF / lone-CR input, which is all ``filter_output`` produces).

Line splitting follows Python ``str.splitlines()`` for LF / CRLF / lone CR;
other Unicode line boundaries (U+0085, U+2028, ...) are deliberately not
split (the real pipeline normalizes to LF first).
"""

from __future__ import annotations

try:
    import regex as _re  # faster ANSI scanning; identical semantics on this pattern
except ImportError:  # pragma: no cover - regex is a declared dependency
    import re as _re

from . import _native, use_native

# _ANSI_ESCAPE_RE — exact mirror of tools/common.py (std `re` and the `regex`
# module behave identically on this pattern; verified on a large corpus).
_ANSI_ESCAPE_RE = _re.compile(
    r"\x1B(?:"
    r"\][^\x07\x1B]*(?:\x07|\x1B\\)|"
    r"[P^_][^\x07\x1B]*(?:\x07|\x1B\\)|"
    r"[@-Z\\-_]|"
    r"\[[0-?]*[ -/]*[@-~]"
    r")"
)


# ---------------------------------------------------------------------------
# _compat — exact mirrors of the Python reference implementations
# ---------------------------------------------------------------------------


def _compat_filter_output(text: str) -> str:
    """Mirror of tools/common.py::filter_output."""
    if not isinstance(text, str):
        raise TypeError("filter_output expects a string")
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _compat_dedup_lines(
    lines: list[str], threshold: int = 3, *, max_block_lines: int = 1
) -> list[str]:
    """List-based mirror of tools/common.py::_dedup_output (no string
    round-trip, so trailing empty lines are preserved exactly)."""
    if max_block_lines <= 1:
        from collections import Counter

        counts = Counter(lines)
        emitted = set()
        result = []
        for line in lines:
            cnt = counts[line]
            if cnt > threshold:
                if line not in emitted:
                    emitted.add(line)
                    result.append(f"{line}  ({cnt} repeats)")
            else:
                result.append(line)
        return result

    # Multi-line path: greedy largest-block-first contiguous run detection.
    consumed = [False] * len(lines)
    result = []
    i = 0
    n = len(lines)

    while i < n:
        if consumed[i]:
            i += 1
            continue

        collapsed = False
        for h in range(min(max_block_lines, n - i), 0, -1):
            block = tuple(lines[i : i + h])
            j = i
            repeats = 0
            while j + h <= n and tuple(lines[j : j + h]) == block:
                if any(consumed[k] for k in range(j, j + h)):
                    break
                repeats += 1
                j += h

            if repeats > threshold:
                for k, line in enumerate(block[:-1]):
                    result.append(line)
                    consumed[i + k] = True
                result.append(f"{block[-1]}  ({repeats} repeats)")
                for k in range(h * repeats):
                    consumed[i + k] = True
                i = i + h * repeats
                collapsed = True
                break

        if not collapsed:
            result.append(lines[i])
            consumed[i] = True
            i += 1

    return result


def _compat_dedup_output(output: str, threshold: int = 3, *, max_block_lines: int = 1) -> str:
    """Mirror of tools/common.py::_dedup_output."""
    if not output:
        return ""
    return "\n".join(
        _compat_dedup_lines(
            output.splitlines(), threshold, max_block_lines=max_block_lines
        )
    )


def _compat_fold_line(line: str, fold_col: int) -> list[str]:
    """Wrap a line at fold_col code points (no mid-codepoint splits)."""
    if fold_col <= 0:
        return [line]
    if not line:
        return [""]
    return [line[i : i + fold_col] for i in range(0, len(line), fold_col)]


class _CompatLineProcessor:
    """Pure-Python mirror of the native LineProcessor semantics.

    Mirrors runtime/stream/line_processor.h exactly:
      - ANSI stripping is streaming (safe at chunk boundaries);
      - CRLF / lone-CR are normalized to LF (the CR may span chunk edges);
      - dedup_mode 0 emits completed lines on feed() (streaming);
      - dedup modes 1/2 buffer raw lines and emit the deduped result at
        flush();
      - the final unterminated line is emitted at flush().
    """

    def __init__(
        self,
        strip_ansi: bool = True,
        dedup_mode: int = 0,
        threshold: int = 3,
        block_window: int = 3,
        max_bytes: int = 0,
        max_lines: int = 0,
        fold_col: int = 0,
    ) -> None:
        self._strip_ansi = strip_ansi
        self._dedup_mode = dedup_mode
        self._threshold = threshold
        self._block_window = block_window
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._fold_col = fold_col
        self._line_buf = ""  # current (possibly partial) line, no terminator
        self._pending_cr = False  # saw '\r', waiting to see if '\n' follows
        self._lines: list[str] = []  # buffered raw lines for dedup modes
        self._bytes = 0
        self._cps = 0
        self._count = 0
        self._exhausted = False

    # -- public API ------------------------------------------------------
    def feed(self, chunk) -> list[str]:
        if isinstance(chunk, (bytes, bytearray)):
            chunk = bytes(chunk).decode("utf-8", "surrogatepass")
        chunk = str(chunk)
        if self._strip_ansi:
            chunk = _ANSI_ESCAPE_RE.sub("", chunk)
        out: list[str] = []
        for ch in chunk:
            if self._pending_cr:
                self._pending_cr = False
                if ch == "\n":
                    # CRLF: consume both; the LF terminates the line.
                    self._finish_line(out)
                    continue
                # lone CR from the previous chunk terminates the line, then
                # the current char is processed normally below.
                self._finish_line(out)
            if ch == "\r":
                self._pending_cr = True
                continue
            if ch == "\n":
                self._finish_line(out)
                continue
            self._line_buf += ch
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        # Resolve trailing state: a pending lone CR at EOF terminates the
        # line; otherwise emit the final unterminated line if it has content.
        if self._pending_cr:
            self._pending_cr = False
            self._finish_line(out)
        elif self._line_buf:
            self._finish_line(out)
        if self._dedup_mode == 1:
            lines = self._dedup_counter(self._lines)
        elif self._dedup_mode == 2:
            lines = self._dedup_block(self._lines)
        else:
            lines = self._lines  # mode 0 already emitted on feed; nothing left
        self._lines = []
        for line in lines:
            for seg in _compat_fold_line(line, self._fold_col):
                self._emit(seg, out)
        return out

    def reset(self) -> None:
        self._line_buf = ""
        self._pending_cr = False
        self._lines = []
        self._bytes = 0
        self._cps = 0
        self._count = 0
        self._exhausted = False

    # -- internals -------------------------------------------------------
    def _finish_line(self, out: list[str]) -> None:
        """A line terminator was reached: process the completed line."""
        if self._dedup_mode == 0:
            for seg in _compat_fold_line(self._line_buf, self._fold_col):
                self._emit(seg, out)
        else:
            self._lines.append(self._line_buf)
        self._line_buf = ""

    def bytes_written(self) -> int:
        return self._bytes

    def code_points_written(self) -> int:
        return self._cps

    def lines_written(self) -> int:
        return self._count

    # -- internals -------------------------------------------------------
    def _emit(self, seg: str, out: list[str]) -> None:
        if self._exhausted:
            return
        if self._max_lines > 0 and self._count >= self._max_lines:
            self._exhausted = True
            return
        if self._max_bytes > 0 and self._bytes + len(seg.encode("utf-8")) > self._max_bytes:
            self._exhausted = True
            return
        out.append(seg)
        self._bytes += len(seg.encode("utf-8"))
        self._cps += len(seg)
        self._count += 1

    def _dedup_counter(self, lines: list[str]) -> list[str]:
        return _compat_dedup_lines(lines, self._threshold, max_block_lines=1)

    def _dedup_block(self, lines: list[str]) -> list[str]:
        return _compat_dedup_lines(
            lines, self._threshold, max_block_lines=self._block_window
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_ansi(text: str | bytes) -> str:
    """One-shot ANSI escape stripping (no CRLF normalization)."""
    if isinstance(text, (bytes, bytearray)):
        data = bytes(text)
    else:
        data = str(text).encode("utf-8", "surrogatepass")
    if not use_native("STREAM") or _native is None:
        return _ANSI_ESCAPE_RE.sub(
            "", data.decode("utf-8", "surrogatepass")
        )
    return _native.stream.strip_ansi(data).decode("utf-8", "surrogatepass")


def filter_output(text: str) -> str:
    """Mirror of tools/common.py::filter_output (ANSI strip + CRLF->LF)."""
    if not isinstance(text, str):
        raise TypeError("filter_output expects a string")
    if not use_native("STREAM") or _native is None:
        return _compat_filter_output(text)
    data = text.encode("utf-8", "surrogatepass")
    return _native.stream.filter_output(data).decode("utf-8", "surrogatepass")


class LineProcessor:
    """Single-pass line stream processor.

    feed(chunk) accepts str or bytes and returns the lines completed by that
    chunk (best effort — dedup modes buffer and emit at flush()); flush()
    returns the remaining lines. ``"\\n".join(flush_lines)`` is the processed
    output. bytes_written()/code_points_written()/lines_written() report the
    emitted content cumulatively.
    """

    def __init__(
        self,
        strip_ansi: bool = True,
        dedup_mode: int = 0,
        threshold: int = 3,
        block_window: int = 3,
        max_bytes: int = 0,
        max_lines: int = 0,
        fold_col: int = 0,
    ) -> None:
        self._use_native = use_native("STREAM") and _native is not None
        if self._use_native:
            self._impl = _native.stream.LineProcessor(
                strip_ansi, dedup_mode, threshold, block_window,
                max_bytes, max_lines, fold_col,
            )
        else:
            self._impl = _CompatLineProcessor(
                strip_ansi=strip_ansi,
                dedup_mode=dedup_mode,
                threshold=threshold,
                block_window=block_window,
                max_bytes=max_bytes,
                max_lines=max_lines,
                fold_col=fold_col,
            )
        self._native = self._use_native

    def feed(self, chunk: str | bytes) -> list[str]:
        if self._native and isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "surrogatepass")
        lines = self._impl.feed(chunk)
        if self._native:
            return [b.decode("utf-8", "surrogatepass") for b in lines]
        return lines

    def flush(self) -> list[str]:
        lines = self._impl.flush()
        if self._native:
            return [b.decode("utf-8", "surrogatepass") for b in lines]
        return lines

    def reset(self) -> None:
        self._impl.reset()

    def bytes_written(self) -> int:
        return self._impl.bytes_written()

    def code_points_written(self) -> int:
        return self._impl.code_points_written()

    def lines_written(self) -> int:
        return self._impl.lines_written()
