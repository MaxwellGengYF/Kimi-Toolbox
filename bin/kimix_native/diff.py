"""kimix_native.diff — diff kernels: unified diff, hunk extraction, inline diff.

Native implementations live in ``runtime_py.diff`` (compiled kernels, GIL
released). The pure-Python functions below mirror ``difflib`` and
``kimi_cli/utils/rich/diff_render.py`` exactly so ``use_native('DIFF') is False``
yields identical behavior.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from difflib import unified_diff as _py_unified_diff

from . import _native, use_native

# ---------------------------------------------------------------------------
# _compat — exact mirrors of the Python reference implementations
# ---------------------------------------------------------------------------


def _compat_unified_diff(
    old_text: bytes,
    new_text: bytes,
    path: str = "",
    include_file_header: bool = True,
    lineterm: str = "\n",
) -> bytes:
    """Pure-Python unified_diff using difflib."""
    old_s = old_text.decode("utf-8", "surrogatepass")
    new_s = new_text.decode("utf-8", "surrogatepass")

    old_lines = old_s.splitlines(keepends=True)
    new_lines = new_s.splitlines(keepends=True)

    # Ensure lines end with newline for proper diff formatting.
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    fromfile = f"a/{path}" if path else "a/file"
    tofile = f"b/{path}" if path else "b/file"

    diff = list(
        _py_unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm=lineterm,
        )
    )

    if (
        not include_file_header
        and len(diff) >= 2
        and diff[0].startswith("--- ")
        and diff[1].startswith("+++ ")
    ):
        diff = diff[2:]

    return "".join(diff).encode("utf-8", "surrogatepass")


def _compat_diff_hunks(
    old_text: bytes,
    new_text: bytes,
    context_lines: int = 3,
) -> list[dict]:
    """Pure-Python diff_hunks using SequenceMatcher.get_grouped_opcodes."""
    old_lines = old_text.decode("utf-8", "surrogatepass").splitlines()
    new_lines = new_text.decode("utf-8", "surrogatepass").splitlines()

    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[dict] = []

    for group in matcher.get_grouped_opcodes(n=context_lines):
        hunk: dict = {
            "old_start": group[0][1] + 1,
            "new_start": group[0][3] + 1,
            "old_lines": [],
            "new_lines": [],
        }
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                hunk["old_lines"].extend(old_lines[i1:i2])
                hunk["new_lines"].extend(new_lines[j1:j2])
            elif tag == "delete":
                hunk["old_lines"].extend(old_lines[i1:i2])
            elif tag == "insert":
                hunk["new_lines"].extend(new_lines[j1:j2])
            elif tag == "replace":
                hunk["old_lines"].extend(old_lines[i1:i2])
                hunk["new_lines"].extend(new_lines[j1:j2])
        hunks.append(hunk)

    return hunks


def _build_offset_map(raw: str, tab_size: int) -> list[int]:
    """Mirror of diff_render.py::_build_offset_map (without the fallback path)."""
    offsets: list[int] = []
    col = 0
    for ch in raw:
        offsets.append(col)
        if ch == "\t":
            col += tab_size - (col % tab_size)
        else:
            col += 1
    offsets.append(col)
    return offsets


def _compat_build_offset_map(raw: str, rendered: str, tab_size: int) -> list[int]:
    """Exact mirror of diff_render.py::_build_offset_map (including the
    fallback linear-map branch).

    Returns a list of length ``len(raw) + 1`` where ``result[i]`` is the
    rendered offset (in Python code points) corresponding to raw position *i*.
    """
    if raw == rendered:
        return list(range(len(raw) + 1))
    offsets: list[int] = []
    col = 0
    for ch in raw:
        offsets.append(col)
        if ch == "\t":
            col += tab_size - (col % tab_size)
        else:
            col += 1
    offsets.append(col)
    if col != len(rendered):
        # The highlighter transformed the text in a way we didn't expect.
        # Return a bounded, monotonic best-effort map so inline stylizing
        # can proceed without crashing or producing out-of-range offsets.
        rendered_len = len(rendered)
        raw_len = len(raw)
        if raw_len == 0:
            return [rendered_len]
        return [(i * rendered_len) // raw_len for i in range(raw_len)] + [rendered_len]
    return offsets


def _compat_inline_diff_ranges(
    old_line: str,
    new_line: str,
    min_ratio: float = 0.5,
    *,
    tab_size: int = 4,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Pure-Python inline diff ranges using SequenceMatcher."""
    sm = SequenceMatcher(None, old_line, new_line, autojunk=False)
    if sm.ratio() < min_ratio:
        return [], []

    old_map = _build_offset_map(old_line, tab_size)
    new_map = _build_offset_map(new_line, tab_size)

    deletes: list[tuple[int, int]] = []
    inserts: list[tuple[int, int]] = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("delete", "replace"):
            deletes.append((old_map[i1], old_map[i2]))
        if op in ("insert", "replace"):
            inserts.append((new_map[j1], new_map[j2]))

    return deletes, inserts


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


def unified_diff(
    old_text: bytes,
    new_text: bytes,
    path: str = "",
    include_file_header: bool = True,
    lineterm: str = "\n",
) -> bytes:
    if use_native("DIFF") and _native is not None:
        return _native.diff.unified_diff(
            old_text, new_text, path, include_file_header, lineterm
        )
    return _compat_unified_diff(old_text, new_text, path, include_file_header, lineterm)


def diff_hunks(
    old_text: bytes,
    new_text: bytes,
    context_lines: int = 3,
) -> list[dict]:
    if use_native("DIFF") and _native is not None:
        return _native.diff.diff_hunks(old_text, new_text, context_lines)
    return _compat_diff_hunks(old_text, new_text, context_lines)


def inline_diff_ranges(
    old_line: str,
    new_line: str,
    min_ratio: float = 0.5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if use_native("DIFF") and _native is not None:
        return _native.diff.inline_diff_ranges(
            old_line.encode("utf-8", "surrogatepass"),
            new_line.encode("utf-8", "surrogatepass"),
            min_ratio,
        )
    return _compat_inline_diff_ranges(old_line, new_line, min_ratio)


def build_offset_map(raw: str, rendered: str, tab_size: int) -> list[int]:
    """Build a mapping from raw-string indices to rendered-string indices.

    Mirrors ``kimi_cli.utils.rich.diff_render._build_offset_map`` exactly:
    the highlighter expands tabs via ``str.expandtabs(tab_size)`` before
    tokenising, so this replicates the same column-aware expansion. Returns a
    list of length ``len(raw) + 1`` where ``result[i]`` is the rendered offset
    corresponding to raw position *i* (offsets are Python code points).
    """
    if use_native("DIFF") and _native is not None:
        return _native.diff.build_offset_map(
            raw.encode("utf-8", "surrogatepass"),
            rendered.encode("utf-8", "surrogatepass"),
            tab_size,
        )
    return _compat_build_offset_map(raw, rendered, tab_size)
