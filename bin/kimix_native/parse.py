"""kimix_native.parse - comment parsers + bash/pwsh command scanners.

Native implementations live in ``runtime_py.parse`` (compiled kernels, GIL
released). The pure-Python ``_compat`` mirrors are the reference algorithms
from the kimi-agent repo (vendored in ``_parse_compat.py`` and
``_shell_compat.py``, with attribution), so ``use_native("PARSE") is False``
yields bit-identical behavior.

Public API (mirrors the reference consumers):
  - parse(lang, source) -> ParseResult{language, comments, code_without_comments}
      lang in {"c","python","shell","sql","html","lisp","pascal"}; comments
      are Comment(content, line, column, kind) dataclasses exactly like the
      reference (kind strings "line"/"block"/"doc").
  - comment_spans(lang, data: bytes) -> list[(start, end, kind)]
      low-level span access (native kernel; kind int 0/1/2).
  - fix_bash_command(cmd) -> BashFix{command, replacements, path_changes}
  - _process_unquoted(cmd) -> str       (bash_tool._process_unquoted)
  - fix_pwsh_command(cmd) -> PwshFix | None
  - pwsh_transform(code) -> (str, warnings)

Native-path notes (documented deviations):
  - C and LISP input containing non-ASCII bytes is routed to _compat (the
    kernel's regex-literal / char-literal heuristics are ASCII-only).
  - BASH_FIX / PWSH_FIX / PWSH_TRANSFORM with non-ASCII commands are routed
    to _compat (reference uses Unicode isalnum/isspace there).
  - BASH_FIX: the kernel emits fallback-command edits with the marker
    replacement "\\x01<name>\\x01"; the shim expands it to the wrapper runner
    and composes the fallback-definitions prefix from the vendored _FALLBACKS.
  - PWSH_TRANSFORM native returns no warning strings (informational only);
    the transformed code is byte-identical to the reference.
"""

from __future__ import annotations

import re

from . import _native, use_native
from . import _parse_compat as _compat
from . import _shell_compat as _shell

# Reference Comment / ParseResult / BashFix / PwshFix shapes.
from ._parse_compat import Comment, ParseResult, BaseParser  # noqa: F401
from ._shell_compat import BashFix, PwshFix  # noqa: F401

_LANG_NAMES = ("c", "python", "shell", "sql", "html", "lisp", "pascal")
_KIND_NAMES = ("line", "block", "doc")

# Fallback command names added after the native PARSE kernel was built.  The
# kernel scanner does not know these words, so the shim routes any ASCII
# command that likely contains one of them as an executable word to the
# pure-Python reference implementation (``_shell_compat`` mirrors
# ``bash_fix.py``).  This keeps behaviour bit-identical without waiting for a
# kernel rebuild.
_POST_KERNEL_FALLBACKS = frozenset({
    # Empty: all post-kernel fallback aliases have been promoted into the
    # native scanner.  This set is kept as a documented extension point.
})
_POST_KERNEL_RE = re.compile(
    r"(?:^|[\s;|&(){}!\n])"
    r"(?:" + "|".join(map(re.escape, _POST_KERNEL_FALLBACKS)) + r")"
    r"(?=[\s;|&(){}<>\n]|$)"
)

_COMPAT_PARSERS = {
    "c": _compat.CParser,
    "python": _compat.PythonParser,
    "shell": _compat.ShellParser,
    "sql": _compat.SqlParser,
    "html": _compat.HtmlParser,
    "lisp": _compat.LispParser,
    "pascal": _compat.PascalParser,
}


def _native_enabled(lang: str, data: bytes) -> bool:
    if not use_native("PARSE") or _native is None:
        return False
    if lang in ("c", "lisp") and not data.isascii():
        return False
    return True


def comment_spans(lang: str, data: bytes) -> list[tuple[int, int, int]]:
    """(start, end, kind) byte-offset spans; kind 0=line 1=block 2=doc."""
    if lang not in _LANG_NAMES:
        raise ValueError(f"unknown language: {lang!r}")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    data = bytes(data)
    if _native_enabled(lang, data):
        return _native.parse.comment_spans(lang, data)
    return _compat_spans(lang, data)


# ---------------------------------------------------------------------------
# _compat: reference algorithms (vendored parsers) -> span list
# ---------------------------------------------------------------------------

def _compat_spans(lang: str, data: bytes) -> list[tuple[int, int, int]]:
    """Run the vendored reference parser and convert its comments to spans."""
    text = data.decode("utf-8", "surrogatepass")
    result = _COMPAT_PARSERS[lang]().parse(text)
    spans = []
    for c in result.comments:
        start = _byte_offset(text, c.line, c.column)
        end = start + len(c.content.encode("utf-8", "surrogatepass"))
        spans.append((start, end, _KIND_NAMES.index(c.kind)))
    return spans


def _byte_offset(text: str, line: int, column: int) -> int:
    """Byte offset of the 1-based (line, column) position in *text*."""
    lines = text.splitlines(keepends=True)
    prefix = "".join(lines[: line - 1]) if line > 1 else ""
    # column is 1-based over code points; locate within the line
    line_text = lines[line - 1] if line - 1 < len(lines) else ""
    return len(prefix.encode("utf-8", "surrogatepass")) + len(
        line_text[: column - 1].encode("utf-8", "surrogatepass")
    )


# ---------------------------------------------------------------------------
# parse() - native spans -> Comment list + code_without_comments
# ---------------------------------------------------------------------------

def _span_line_col(text: str, start: int) -> tuple[int, int]:
    prefix = text[:start]
    line = prefix.count("\n") + 1
    col = len(prefix.rsplit("\n", 1)[-1]) + 1
    return line, col


def _marker_offset(lang: str, text: str, s: int, e: int, k: int) -> int:
    """Bytes before the span start occupied by the comment marker.

    The reference records Comment.line/column at the MARKER start for parsers
    whose content excludes the markers (C, SQL, HTML, Pascal); for parsers
    whose content includes the marker (Python, Shell, Lisp) the offset is 0.
    """
    if lang == "c":
        return 2 if k != 2 else 3
    if lang == "sql":
        if k == 1:
            return 2
        return 2 if text[s - 2 : s] == "--" else 1
    if lang == "html":
        return 4 if k == 1 else 2
    if lang == "pascal":
        if k == 0:
            return 2
        return 1 if text[s - 1 : s] == "{" else 2
    return 0


def _byte_to_char_offsets(data: bytes, positions) -> dict[int, int]:
    """Map byte offsets to character offsets in ONE pass (O(n)).

    Only needed for non-ASCII input (ASCII bytes == chars).
    """
    pos_set = sorted(set(positions))
    if not pos_set:
        return {}
    text = data.decode("utf-8", "surrogatepass")
    out: dict[int, int] = {}
    byte_i = 0
    pi = 0
    for char_i, ch in enumerate(text):
        while pi < len(pos_set) and pos_set[pi] <= byte_i:
            out[pos_set[pi]] = char_i
            pi += 1
        byte_i += len(ch.encode("utf-8", "surrogatepass"))
    while pi < len(pos_set):
        out[pos_set[pi]] = len(text)
        pi += 1
    return out


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _comments_from_spans(lang: str, text: str, spans) -> list[Comment]:
    import bisect

    starts = _line_starts(text)
    comments = []
    for s, e, k in spans:
        content = text[s:e]
        off = _marker_offset(lang, text, s, e, k)
        line = bisect.bisect_right(starts, s - off)
        col = (s - off) - starts[line - 1] + 1
        comments.append(Comment(content=content, line=line, column=col,
                                kind=_KIND_NAMES[k]))
    return comments


def _base_build_result(text: str, comments: list[Comment]) -> str:
    """base.py::BaseParser._build_result (Python/Shell/Lisp parsers)."""
    lines = text.splitlines(keepends=True)
    for comment in sorted(comments, key=lambda c: (c.line, c.column), reverse=True):
        if 1 <= comment.line <= len(lines):
            line = lines[comment.line - 1]
            col = comment.column - 1
            end_col = min(col + len(comment.content), len(line))
            if col < len(line):
                lines[comment.line - 1] = (
                    line[:col] + " " * len(comment.content) + line[end_col:]
                )
    return "".join(lines)


def _code_without_comments(lang: str, text: str, spans) -> str:
    if lang in ("python", "shell", "lisp"):
        return _base_build_result(text, _comments_from_spans(lang, text, spans))
    if lang == "html":
        ranges = []
        for s, e, k in spans:
            if k == 1:  # <!-- ... -->
                ranges.append((s - 4, e + 3))
            else:       # <? ... ?>
                ranges.append((s - 2, e + 2))
        chars = list(text)
        for rs, re_ in sorted(ranges, reverse=True):
            for j in range(max(rs, 0), min(re_, len(chars))):
                if chars[j] != "\n":
                    chars[j] = " "
        return "".join(chars)
    if lang == "c":
        out = []
        prev = 0
        for s, e, k in spans:
            if k == 0:  # // line: marker + content removed, newline kept
                out.append(text[prev : s - 2])
                prev = e
            else:       # block/doc: marker removed, content masked, */ -> 2 spaces
                ms = s - (3 if k == 2 else 2)
                out.append(text[prev:ms])
                seg = text[s:e]
                out.append("".join(ch if ch == "\n" else " " for ch in seg))
                if e + 2 <= len(text):
                    out.append("  ")
                    prev = e + 2
                else:
                    prev = e  # unclosed at EOF: no closing marker
        out.append(text[prev:])
        return "".join(out)
    if lang == "sql":
        def _sql_block_mask(seg: str) -> str:
            """Block-comment content -> spaces/newlines, nested markers dropped
            (the reference never emits nested /* */ pairs)."""
            out = []
            i = 0
            n = len(seg)
            while i < n:
                if seg[i] == "/" and i + 1 < n and seg[i + 1] == "*":
                    i += 2
                elif seg[i] == "*" and i + 1 < n and seg[i + 1] == "/":
                    i += 2
                elif seg[i] == "\n":
                    out.append("\n")
                    i += 1
                else:
                    out.append(" ")
                    i += 1
            return "".join(out)

        out = []
        prev = 0
        for s, e, k in spans:
            if k == 0:
                ms = s - 2 if text[s - 2 : s] == "--" else s - 1
                out.append(text[prev:ms])
                prev = e
            else:
                out.append(text[prev : s - 2])
                out.append(_sql_block_mask(text[s:e]))
                if e + 2 <= len(text):
                    out.append("  ")
                    prev = e + 2
                else:
                    prev = e  # unclosed at EOF: no closing marker
        out.append(text[prev:])
        return "".join(out)
    if lang == "pascal":
        def _comment_area(sl, sc, el, ec):
            parts = []
            cur_l, cur_c = sl, sc
            while cur_l < el:
                parts.append(" ")
                parts.append("\n")
                cur_l += 1
                cur_c = 1
            while cur_c < ec:
                parts.append(" ")
                cur_c += 1
            return "".join(parts)

        out = []
        prev = 0
        for s, e, k in spans:
            if k == 0:  # // line
                out.append(text[prev : s - 2])
                prev = e
            elif text[s - 1] == "{":  # brace comment
                sl, sc = _span_line_col(text, s - 1)
                el, ec = _span_line_col(text, e)
                out.append(text[prev : s - 1])
                # content newlines are emitted during the scan (reference
                # emit_char(ch)); the area then adds " \n" per line transition
                out.append("".join(ch for ch in text[s:e] if ch == "\n"))
                if e + 1 <= len(text):
                    out.append(_comment_area(sl, sc, el, ec))
                    out.append(" ")  # the closing }
                    prev = e + 1
                else:
                    prev = e  # unclosed at EOF: no area, no closer
            else:  # (* ... *)
                sl, sc = _span_line_col(text, s - 2)
                el, ec = _span_line_col(text, e)
                out.append(text[prev : s - 2])
                out.append("".join(ch for ch in text[s:e] if ch == "\n"))
                if e + 2 <= len(text):
                    out.append(_comment_area(sl, sc, el, ec))
                    out.append("  ")  # the closing *)
                    prev = e + 2
                else:
                    prev = e  # unclosed at EOF: no area, no closer
        out.append(text[prev:])
        return "".join(out)
    raise ValueError(f"unknown language: {lang!r}")


def parse(lang: str, source: str) -> ParseResult:
    """Mirror of the reference BaseParser.parse (comments + code without)."""
    if lang not in _LANG_NAMES:
        raise ValueError(f"unknown language: {lang!r}")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    data = source.encode("utf-8", "surrogatepass")
    if _native_enabled(lang, data):
        spans = _native.parse.comment_spans(lang, data)
        # native spans are BYTE offsets; the str-based conversion below needs
        # character offsets (they coincide for pure-ASCII input)
        if data.isascii():
            char_spans = spans
        else:
            mapping = _byte_to_char_offsets(
                data, [p for s, e, _ in spans for p in (s, e)]
            )
            char_spans = [(mapping[s], mapping[e], k) for s, e, k in spans]
        comments = _comments_from_spans(lang, source, char_spans)
        return ParseResult(
            language=lang,
            comments=comments,
            code_without_comments=_code_without_comments(lang, source, char_spans),
        )
    return _COMPAT_PARSERS[lang]().parse(source)


# ---------------------------------------------------------------------------
# shell scanners (bash_fix / bash_tool / pwsh_fix / process_pwsh)
# ---------------------------------------------------------------------------

def _apply_edits(data: bytes, edits) -> bytes:
    pieces = []
    previous = 0
    for start, end, replacement in sorted(edits):
        pieces.append(data[previous:start])
        pieces.append(replacement)
        previous = end
    pieces.append(data[previous:])
    return b"".join(pieces)


def _expand_bash_markers(edits) -> list[tuple[int, int, bytes]]:
    """Expand "\\x01<name>\\x01" fallback markers to wrapper runners."""
    out = []
    for start, end, replacement in edits:
        if replacement.startswith(b"\x01") and replacement.endswith(b"\x01") and len(replacement) > 2:
            name = replacement[1:-1].decode("utf-8", "surrogatepass")
            out.append((start, end, _shell._wrapper_runner(name).encode("utf-8", "surrogatepass")))
        else:
            out.append((start, end, replacement))
    return out


def _shell_native(dialect: str, cmd: str) -> bool:
    if not use_native("PARSE") or _native is None:
        return False
    if not cmd.isascii():
        return False
    return True


def fix_bash_command(cmd: str) -> BashFix:
    """Mirror of bash_fix.fix_bash_command (Windows Git Bash rewrites)."""
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")
    if not _shell_native("bash_fix", cmd):
        return _shell.fix_bash_command(cmd)
    # Fallback commands added after the compiled kernel was built are not
    # recognised by the native scanner.  Route commands that may contain such
    # newer aliases through the pure-Python reference so behaviour stays
    # bit-identical with ``bash_fix.py`` until the kernel is rebuilt.
    if _POST_KERNEL_RE.search(cmd):
        return _shell.fix_bash_command(cmd)
    data = cmd.encode("utf-8", "surrogatepass")
    edits, names_bytes, notes_bytes = _native.parse.shell_scan("bash_fix", data)
    if not names_bytes and not edits:
        return BashFix(cmd)
    names = [n.decode("utf-8", "surrogatepass") for n in names_bytes]
    source = _apply_edits(data, _expand_bash_markers(edits)).decode("utf-8", "surrogatepass")
    definitions = "\n".join(_shell._FALLBACKS[n] for n in dict.fromkeys(names))
    prefix = definitions + "\n" if definitions else ""
    path_changes = tuple(n.decode("utf-8", "surrogatepass") for n in notes_bytes)
    return BashFix(prefix + source, tuple(names), path_changes)


def _process_unquoted(cmd: str) -> str:
    """Mirror of bash_tool._process_unquoted (backslash -> slash, unquoted)."""
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")
    if not use_native("PARSE") or _native is None:
        return _shell._process_unquoted(cmd)
    # No Unicode-sensitive decisions in this scanner: native on any input.
    data = cmd.encode("utf-8", "surrogatepass")
    edits, _, _ = _native.parse.shell_scan("bash_process_unquoted", data)
    return _apply_edits(data, edits).decode("utf-8", "surrogatepass")


_PWSH_WARNINGS = {
    0: "",
    1: _shell._W_UNCLOSED_DQ,
    2: _shell._W_UNCLOSED_SQ,
    3: _shell._W_UNCLOSED_HDQ,
    4: _shell._W_UNCLOSED_HSQ,
    5: _shell._W_UNCLOSED_BLOCK,
    6: _shell._W_TRAILING_COMMENT,
    7: _shell._W_STOP_PARSING,
    8: _shell._W_COMMENT_ONLY,
    9: _shell._W_TRAILING_CONTINUATION,
}


def fix_pwsh_command(cmd: str) -> PwshFix | None:
    """Mirror of pwsh_fix.fix_pwsh_command (validate + repair quoting)."""
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")
    if not cmd or not cmd.strip():
        return None
    if not _shell_native("pwsh_fix", cmd):
        return _shell.fix_pwsh_command(cmd)
    data = cmd.encode("utf-8", "surrogatepass")
    code = _native.parse.pwsh_fix_warning(data)
    if code == -1:
        return None
    fixed = _native.parse.shell_transform("pwsh_fix", data).decode("utf-8", "surrogatepass")
    warning = _PWSH_WARNINGS.get(code & 0x0F, "")
    if code & 0x10:
        warning = f"{warning}\n{_shell._W_TRAILING_CONTINUATION}" if warning else _shell._W_TRAILING_CONTINUATION
    return PwshFix(command=fixed, warning=warning)


def pwsh_transform(code: str) -> tuple[str, list[str]]:
    """Mirror of process_pwsh.pwsh_transform (PS7 -> PS5.1).

    Native path returns byte-identical transformed code; the warning strings
    are informational and empty in native mode (documented deviation).
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not _shell_native("pwsh_transform", code):
        return _shell.pwsh_transform(code)
    data = code.encode("utf-8", "surrogatepass")
    out = _native.parse.shell_transform("pwsh_transform", data).decode("utf-8", "surrogatepass")
    return out, []
