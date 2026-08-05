"""kimix_native.json -- incremental JSON lexer (plan 010).

Native implementation lives in ``runtime_py.json.IncrementalJsonLexer``
(compiled kernel, GIL released). The pure-Python ``_compat`` mirror below is
a line-for-line port of the C++ state machine (same rules, same outputs):

- feed(bytes): incremental; strings/escapes/\\uXXXX/comments/numbers/literals
  persist across chunk boundaries.
- is_complete(): a complete top-level JSON value (depth 0 after a value,
  no pending comma). Whitespace/comments after the value are fine.
- has_error(): malformed input (unmatched close, `{"a": }`, bad escape,
  invalid number/literal, a second top-level value after completion).
- value_span(key): (start, end) byte offsets of the top-level key's value
  in the internal buffer, or None. The buffer is append-only so offsets
  stay valid; ``buffer()`` exposes the bytes for slicing.
- Relaxed: trailing commas (`[1,2,]`, `{"a":1,}`) and ``//`` / ``/* */``
  comments anywhere whitespace is allowed (outside strings); raw newlines
  inside strings are accepted (LLM streams emit them unescaped).
"""

from __future__ import annotations

import contextlib
import json

from . import _native, use_native


def _enc(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def _dec(b: bytes) -> str:
    return b.decode("utf-8", "surrogatepass")


def _compact(obj) -> bytes:
    """orjson-like compact JSON bytes (no spaces, raw UTF-8)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8", "surrogatepass"
    )

_USE = use_native("JSON") and _native is not None

# ---------------------------------------------------------------------------
# _compat -- pure-Python mirror of the kernel state machine
# ---------------------------------------------------------------------------

# frame substates
_KOBJ_KEY_EXPECTED, _KOBJ_COLON_EXPECTED, _KOBJ_VALUE_EXPECTED, _KOBJ_VALUE_DONE = 0, 1, 2, 3
_KARR_VALUE_EXPECTED, _KARR_VALUE_DONE = 0, 1
_KFRAME_OBJ, _KFRAME_ARR = 0, 1

# scanner states
(_KSCAN_NORMAL, _KSCAN_STRING, _KSCAN_STRING_ESC, _KSCAN_UNICODE,
 _KSCAN_LINE_COMMENT, _KSCAN_BLOCK_COMMENT, _KSCAN_BLOCK_COMMENT_STAR,
 _KSCAN_NUMBER, _KSCAN_LITERAL, _KSCAN_SLASH_PENDING) = range(10)

# number sub-states
(_KNUM_LEADING_SIGN, _KNUM_INT, _KNUM_FRAC_OR_EXP, _KNUM_FRAC,
 _KNUM_EXP_SIGN, _KNUM_EXP_SIGNED, _KNUM_EXP) = range(7)

_LITERALS = ("true", "false", "null")


def _is_ws(c: int) -> bool:
    return c in (0x20, 0x09, 0x0A, 0x0D)


def _is_digit(c: int) -> bool:
    return 0x30 <= c <= 0x39


def _hex_value(c: int) -> int:
    if 0x30 <= c <= 0x39:
        return c - 0x30
    if 0x61 <= c <= 0x66:
        return c - 0x61 + 10
    if 0x41 <= c <= 0x46:
        return c - 0x41 + 10
    return -1


def _is_number_delim(c: int) -> bool:
    return _is_ws(c) or c in (0x2C, 0x7D, 0x5D, 0x2F)  # , } ] /


def _encode_utf8(cp: int) -> bytes:
    if cp < 0x80:
        return bytes([cp])
    if cp < 0x800:
        return bytes([0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)])
    if cp < 0x10000:
        return bytes([0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)])
    return bytes([0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F),
                  0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)])


class _CompatIncrementalJsonLexer:
    """Exact port of src/runtime/json/incremental_lexer.cpp."""

    __slots__ = (
        "_buf", "_processed", "_complete", "_error", "_root_done", "_stack",
        "_scan", "_number_sub", "_literal_kind", "_literal_pos",
        "_unicode_need", "_unicode_val", "_comment_star", "_in_key",
        "_str_buf", "_pending_high", "_pending_high_val", "_have_key",
        "_cur_key", "_want_span", "_value_start", "_spans",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._processed = 0
        self._complete = False
        self._error = False
        self._root_done = False
        self._stack: list[list[int]] = []  # [kind, sub]
        self._scan = _KSCAN_NORMAL
        self._number_sub = 0
        self._literal_kind = 0
        self._literal_pos = 0
        self._unicode_need = 0
        self._unicode_val = 0
        self._comment_star = False
        self._in_key = False
        self._str_buf = bytearray()
        self._pending_high = False
        self._pending_high_val = 0
        self._have_key = False
        self._cur_key = ""
        self._want_span = False
        self._value_start = -1
        self._spans: list[tuple[str, int, int]] = []

    def feed(self, chunk: bytes) -> None:
        if not chunk or self._error:
            return
        self._buf += chunk
        self._scan_buf()
        self._processed = len(self._buf)

    def is_complete(self) -> bool:
        return self._complete and not self._error

    def has_error(self) -> bool:
        return self._error

    def value_span(self, key: str):
        for k, start, end in reversed(self._spans):
            if k == key:
                return start, end
        return None

    def top_level_keys(self) -> list[str]:
        return [k for k, _, _ in self._spans]

    def buffer(self) -> bytes:
        return bytes(self._buf)

    # -- scanner -----------------------------------------------------------

    def _record_span(self, end: int) -> None:
        self._spans.append((self._cur_key, self._value_start, end))

    def _scan_buf(self) -> None:
        size = len(self._buf)
        i = self._processed
        while i < size:
            c = self._buf[i]
            if self._error:
                return
            if self._scan == _KSCAN_NORMAL:
                if _is_ws(c):
                    pass
                elif c == 0x7B:  # {
                    if not self._value_position_ok():
                        self._error = True
                    else:
                        self._value_started(i)
                        self._stack.append([_KFRAME_OBJ, _KOBJ_KEY_EXPECTED])
                elif c == 0x5B:  # [
                    if not self._value_position_ok():
                        self._error = True
                    else:
                        self._value_started(i)
                        self._stack.append([_KFRAME_ARR, _KARR_VALUE_EXPECTED])
                elif c == 0x7D:  # }
                    if not self._stack or self._stack[-1][0] != _KFRAME_OBJ:
                        self._error = True
                    else:
                        self._close_container(i)
                elif c == 0x5D:  # ]
                    if not self._stack or self._stack[-1][0] != _KFRAME_ARR:
                        self._error = True
                    else:
                        self._close_container(i)
                elif c == 0x3A:  # :
                    f = self._stack[-1] if self._stack else None
                    if (f is None or f[0] != _KFRAME_OBJ or
                            f[1] != _KOBJ_COLON_EXPECTED):
                        self._error = True
                    else:
                        f[1] = _KOBJ_VALUE_EXPECTED
                        if len(self._stack) == 1 and self._have_key:
                            self._want_span = True
                            self._value_start = -1
                elif c == 0x2C:  # ,
                    if not self._stack:
                        self._error = True
                    else:
                        f = self._stack[-1]
                        if f[0] == _KFRAME_OBJ:
                            if f[1] == _KOBJ_VALUE_DONE:
                                f[1] = _KOBJ_KEY_EXPECTED
                            else:
                                self._error = True
                        else:
                            if f[1] == _KARR_VALUE_DONE:
                                f[1] = _KARR_VALUE_EXPECTED
                            else:
                                self._error = True
                elif c == 0x22:  # "
                    key_pos = bool(self._stack) and self._stack[-1][0] == _KFRAME_OBJ \
                        and self._stack[-1][1] == _KOBJ_KEY_EXPECTED
                    if not key_pos and not self._value_position_ok():
                        self._error = True
                    else:
                        self._value_started(i)
                        if key_pos:
                            self._in_key = True
                            self._str_buf = bytearray()
                        self._scan = _KSCAN_STRING
                elif c == 0x2D or _is_digit(c):  # - or digit
                    if not self._value_position_ok():
                        self._error = True
                    else:
                        self._value_started(i)
                        self._scan = _KSCAN_NUMBER
                        self._number_sub = _KNUM_LEADING_SIGN if c == 0x2D else _KNUM_INT
                elif c in (0x74, 0x66, 0x6E):  # t f n
                    if not self._value_position_ok():
                        self._error = True
                    else:
                        self._value_started(i)
                        self._scan = _KSCAN_LITERAL
                        self._literal_kind = 0 if c == 0x74 else (1 if c == 0x66 else 2)
                        self._literal_pos = 1
                elif c == 0x2F:  # /
                    if i + 1 < size:
                        n = self._buf[i + 1]
                        if n == 0x2F:
                            self._scan = _KSCAN_LINE_COMMENT
                            i += 1
                        elif n == 0x2A:
                            self._scan = _KSCAN_BLOCK_COMMENT
                            i += 1
                        else:
                            self._error = True
                    else:
                        self._scan = _KSCAN_SLASH_PENDING
                elif self._root_done:
                    self._error = True
                else:
                    self._error = True
            elif self._scan == _KSCAN_STRING:
                if c == 0x22:  # "
                    if self._in_key:
                        self._finish_key()
                    else:
                        self._scalar_complete(i + 1)
                    self._scan = _KSCAN_NORMAL
                elif c == 0x5C:  # backslash
                    self._scan = _KSCAN_STRING_ESC
                elif self._in_key:
                    self._str_buf.append(c)
            elif self._scan == _KSCAN_STRING_ESC:
                if c == 0x75:  # u
                    self._scan = _KSCAN_UNICODE
                    self._unicode_need = 4
                    self._unicode_val = 0
                elif c in (0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74):
                    if self._in_key:
                        self._str_buf.append({0x22: 0x22, 0x5C: 0x5C, 0x2F: 0x2F,
                                              0x62: 0x08, 0x66: 0x0C, 0x6E: 0x0A,
                                              0x72: 0x0D, 0x74: 0x09}[c])
                    self._scan = _KSCAN_STRING
                else:
                    self._error = True
            elif self._scan == _KSCAN_UNICODE:
                h = _hex_value(c)
                if h < 0:
                    self._error = True
                else:
                    self._unicode_val = (self._unicode_val << 4) | h
                    self._unicode_need -= 1
                    if self._unicode_need == 0:
                        if self._in_key:
                            cp = self._unicode_val
                            if 0xD800 <= cp <= 0xDBFF:
                                if self._pending_high:
                                    self._str_buf += _encode_utf8(self._pending_high_val)
                                self._pending_high = True
                                self._pending_high_val = cp
                            elif 0xDC00 <= cp <= 0xDFFF:
                                if self._pending_high:
                                    combined = 0x10000 + ((self._pending_high_val - 0xD800) << 10) + (cp - 0xDC00)
                                    self._str_buf += _encode_utf8(combined)
                                    self._pending_high = False
                                else:
                                    self._str_buf += _encode_utf8(cp)
                            else:
                                if self._pending_high:
                                    self._str_buf += _encode_utf8(self._pending_high_val)
                                    self._pending_high = False
                                self._str_buf += _encode_utf8(cp)
                        self._scan = _KSCAN_STRING
            elif self._scan == _KSCAN_LINE_COMMENT:
                if c == 0x0A:
                    self._scan = _KSCAN_NORMAL
            elif self._scan == _KSCAN_BLOCK_COMMENT:
                if c == 0x2A:  # *
                    self._scan = _KSCAN_BLOCK_COMMENT_STAR
            elif self._scan == _KSCAN_BLOCK_COMMENT_STAR:
                if c == 0x2F:  # /
                    self._scan = _KSCAN_NORMAL
                elif c != 0x2A:
                    self._scan = _KSCAN_BLOCK_COMMENT
            elif self._scan == _KSCAN_SLASH_PENDING:
                if c == 0x2F:
                    self._scan = _KSCAN_LINE_COMMENT
                elif c == 0x2A:
                    self._scan = _KSCAN_BLOCK_COMMENT
                else:
                    self._error = True
            elif self._scan == _KSCAN_NUMBER:
                done = False
                sub = self._number_sub
                if sub == _KNUM_LEADING_SIGN:
                    if _is_digit(c):
                        self._number_sub = _KNUM_INT
                    else:
                        self._error = True
                elif sub == _KNUM_INT:
                    if _is_digit(c):
                        pass
                    elif c == 0x2E:  # .
                        self._number_sub = _KNUM_FRAC_OR_EXP
                    elif c in (0x65, 0x45):  # e E
                        self._number_sub = _KNUM_EXP_SIGN
                    elif _is_number_delim(c):
                        done = True
                    else:
                        self._error = True
                elif sub == _KNUM_FRAC_OR_EXP:
                    if _is_digit(c):
                        self._number_sub = _KNUM_FRAC
                    else:
                        self._error = True
                elif sub == _KNUM_FRAC:
                    if _is_digit(c):
                        pass
                    elif c in (0x65, 0x45):
                        self._number_sub = _KNUM_EXP_SIGN
                    elif _is_number_delim(c):
                        done = True
                    else:
                        self._error = True
                elif sub == _KNUM_EXP_SIGN:
                    if _is_digit(c):
                        self._number_sub = _KNUM_EXP
                    elif c in (0x2B, 0x2D):  # + -
                        self._number_sub = _KNUM_EXP_SIGNED
                    else:
                        self._error = True
                elif sub == _KNUM_EXP_SIGNED:
                    if _is_digit(c):
                        self._number_sub = _KNUM_EXP
                    else:
                        self._error = True
                elif sub == _KNUM_EXP:
                    if _is_digit(c):
                        pass
                    elif _is_number_delim(c):
                        done = True
                    else:
                        self._error = True
                if done:
                    self._scan = _KSCAN_NORMAL
                    self._scalar_complete(i)
                    i -= 1  # reprocess the delimiter
            elif self._scan == _KSCAN_LITERAL:
                lit = _LITERALS[self._literal_kind]
                if c == ord(lit[self._literal_pos]):
                    self._literal_pos += 1
                    if self._literal_pos == len(lit):
                        self._scan = _KSCAN_NORMAL
                        self._scalar_complete(i + 1)
                else:
                    self._error = True
            i += 1

        # NOTE: a number token is NEVER completed at a feed boundary (see the
        # kernel comment in incremental_lexer.cpp): chunked feeding stays
        # byte-identical to one-shot feeding.

    # -- helpers -----------------------------------------------------------

    def _value_position_ok(self) -> bool:
        if not self._stack:
            return not self._root_done
        f = self._stack[-1]
        if f[0] == _KFRAME_OBJ:
            return f[1] == _KOBJ_VALUE_EXPECTED
        return f[1] == _KARR_VALUE_EXPECTED

    def _value_started(self, pos: int) -> None:
        if len(self._stack) == 1 and self._want_span:
            self._value_start = pos
            self._want_span = False

    def _scalar_complete(self, end: int) -> None:
        if not self._stack:
            self._complete = True
            self._root_done = True
            return
        f = self._stack[-1]
        if f[0] == _KFRAME_OBJ:
            if f[1] != _KOBJ_VALUE_EXPECTED:
                self._error = True
                return
            f[1] = _KOBJ_VALUE_DONE
        else:
            if f[1] != _KARR_VALUE_EXPECTED:
                self._error = True
                return
            f[1] = _KARR_VALUE_DONE
        if len(self._stack) == 1 and self._have_key and not self._want_span:
            self._record_span(end)
            self._have_key = False
            self._want_span = False
            self._value_start = -1
            self._cur_key = ""

    def _close_container(self, pos: int) -> None:
        end = pos + 1
        f = self._stack[-1]
        if f[0] == _KFRAME_OBJ:
            if f[1] not in (_KOBJ_KEY_EXPECTED, _KOBJ_VALUE_DONE):
                self._error = True
                return
        else:
            if f[1] not in (_KARR_VALUE_EXPECTED, _KARR_VALUE_DONE):
                self._error = True
                return
        self._stack.pop()
        if not self._stack:
            self._complete = True
            self._root_done = True
            return
        parent = self._stack[-1]
        if parent[0] == _KFRAME_OBJ:
            if parent[1] == _KOBJ_VALUE_EXPECTED:
                parent[1] = _KOBJ_VALUE_DONE
        else:
            if parent[1] == _KARR_VALUE_EXPECTED:
                parent[1] = _KARR_VALUE_DONE
        if len(self._stack) == 1 and self._have_key and not self._want_span:
            self._record_span(end)
            self._have_key = False
            self._want_span = False
            self._value_start = -1
            self._cur_key = ""

    def _finish_key(self) -> None:
        if self._pending_high:
            self._str_buf += _encode_utf8(self._pending_high_val)
            self._pending_high = False
        if len(self._stack) == 1:
            self._cur_key = bytes(self._str_buf).decode("utf-8", "surrogatepass")
            self._have_key = True
        self._str_buf = bytearray()
        self._in_key = False
        f = self._stack[-1]
        if f[0] == _KFRAME_OBJ and f[1] == _KOBJ_KEY_EXPECTED:
            f[1] = _KOBJ_COLON_EXPECTED
        else:
            self._error = True


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


class IncrementalJsonLexer:
    """Incremental JSON lexer: feed chunks, detect completeness, extract
    top-level key -> value byte spans without reparsing."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.json.IncrementalJsonLexer()
        else:
            self._compat = _CompatIncrementalJsonLexer()
        # Shadow of every byte fed (offsets from value_span slice into this).
        self._shadow = bytearray()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._native is not None:
            if self._native.has_error():
                return  # kernel ignores further input after an error
            self._native.feed(bytes(chunk))
            self._shadow += chunk
        else:
            if self._compat.has_error():
                return
            self._compat.feed(chunk)
            self._shadow += chunk

    def is_complete(self) -> bool:
        if self._native is not None:
            return bool(self._native.is_complete())
        return self._compat.is_complete()

    def has_error(self) -> bool:
        if self._native is not None:
            return bool(self._native.has_error())
        return self._compat.has_error()

    def value_span(self, key: str):
        """(start, end) byte offsets of the top-level key's value in
        ``buffer()``, or None. Copy the bytes you need before the next
        feed (the native offsets are invalidated by the next feed)."""
        if self._native is not None:
            span = self._native.value_span(key)
            if span is None:
                return None
            return int(span[0]), int(span[1])
        return self._compat.value_span(key)

    def value_bytes(self, key: str):
        """Convenience: the raw value bytes for a top-level key, or None."""
        span = self.value_span(key)
        if span is None:
            return None
        start, end = span
        return bytes(self._shadow[start:end])

    def top_level_keys(self) -> list[str]:
        if self._native is not None:
            return [str(k) for k in self._native.top_level_keys()]
        return self._compat.top_level_keys()

    def buffer(self) -> bytes:
        """Every byte fed so far (mirrors the kernel's internal buffer)."""
        return bytes(self._shadow)

    def reset(self) -> None:
        self._shadow = bytearray()
        if self._native is not None:
            self._native.reset()
        else:
            self._compat.reset()


# ---------------------------------------------------------------------------
# Plan 016: JsonStore / SchemaOps / notification batch scan.
# ---------------------------------------------------------------------------


class JsonStore:
    """One native JSON document per task/notification file.

    load() parses once; update() deep-merges a partial object (nested objects
    merge key-by-key, scalars replace); get()/save_atomic() serialize with
    orjson OPT_INDENT_2-compatible bytes; save_atomic writes tmp + rename.
    """

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.json.JsonStore()
        else:
            self._doc: dict = {}

    def load(self, data: bytes) -> None:
        if self._native is not None:
            self._native.load(bytes(data))
            return
        try:
            parsed = json.loads(_dec(bytes(data)))
        except ValueError:
            self._doc = {}
            return
        self._doc = parsed if isinstance(parsed, dict) else {}

    def update(self, data: bytes) -> None:
        if self._native is not None:
            self._native.update(bytes(data))
            return
        try:
            parsed = json.loads(_dec(bytes(data)))
        except ValueError:
            return
        if isinstance(parsed, dict):
            self._doc = _deep_merge(self._doc, parsed)

    def get(self) -> bytes:
        if self._native is not None:
            return bytes(self._native.get())
        return _compat_indent2(self._doc)

    def keys(self) -> list[str]:
        if self._native is not None:
            return [str(k) for k in self._native.keys()]
        return list(self._doc.keys())

    def save_atomic(self, path: str) -> bytes:
        """Atomically write the pretty serialization (tmp + rename)."""
        if self._native is not None:
            return bytes(self._native.save_atomic(str(path)))
        blob = self.get()
        _compat_atomic_write(str(path), blob)
        return blob

    def clear(self) -> None:
        if self._native is not None:
            self._native.clear()
        else:
            self._doc = {}

    def loaded(self) -> bool:
        if self._native is not None:
            return bool(self._native.loaded())
        return True


def _deep_merge(dst: dict, src: dict) -> dict:
    """dict-style deep merge: nested dicts merge; scalars replace."""
    out = dict(dst)
    for k, v in src.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _compat_indent2(obj) -> bytes:
    """orjson OPT_INDENT_2-style pretty bytes (2-space indent, raw UTF-8)."""

    def esc(s: str) -> str:
        out = []
        for ch in s:
            o = ord(ch)
            if ch == '"':
                out.append('\\"')
            elif ch == "\\":
                out.append("\\\\")
            elif ch == "\b":
                out.append("\b")
            elif ch == "\f":
                out.append("\f")
            elif ch == "\n":
                out.append("\n")
            elif ch == "\r":
                out.append("\r")
            elif ch == "\t":
                out.append("\t")
            elif o < 0x20:
                out.append("\\u%04x" % o)
            else:
                out.append(ch)
        return "".join(out)

    def render(v, level: int) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, (int, float)):
            if isinstance(v, float):
                r = repr(v)
                if "e" not in r and "E" not in r and "." not in r:
                    r += ".0"
                return r
            return str(v)
        if isinstance(v, str):
            return '"' + esc(v) + '"'
        if isinstance(v, list):
            if not v:
                return "[]"
            pad = "  " * (level + 1)
            items = [pad + render(x, level + 1) for x in v]
            return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
        if isinstance(v, dict):
            if not v:
                return "{}"
            pad = "  " * (level + 1)
            items = [pad + '"' + esc(str(k)) + '": ' + render(x, level + 1)
                     for k, x in v.items()]
            return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
        return '"' + esc(str(v)) + '"'

    return _enc(render(obj, 0))


def _compat_atomic_write(path: str, blob: bytes) -> None:
    import os
    import tempfile

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def scan_notifications(jsonl: bytes, now_ms: int = 0) -> list[dict]:
    """One parse + one sort of a JSONL of {event, delivery} views.

    Returns [{id, created_at, event, delivery}] sorted by created_at desc
    (stable; ties keep input order).
    """
    if _USE:
        rows = _native.json.scan_notifications(bytes(jsonl), int(now_ms))
        return [
            {
                "id": str(r["id"]),
                "created_at": float(r["created_at"]),
                "event": dict(r["event"]),
                "delivery": dict(r["delivery"]),
            }
            for r in rows
        ]
    rows = []
    for line in bytes(jsonl).split(b"\n"):
        if not line:
            continue
        try:
            view = json.loads(_dec(line))
        except ValueError:
            continue
        event = view.get("event")
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "id": event.get("id", ""),
                "created_at": float(event.get("created_at", 0.0) or 0.0),
                "event": event,
                "delivery": view.get("delivery") or {},
            }
        )
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


def deref_json_schema(schema: bytes, registry: list[bytes] | None = None) -> bytes:
    """Inline local $ref pointers; drop dead definition buckets."""
    if _USE:
        reg = list(registry) if registry is not None else []
        return bytes(_native.json.deref_json_schema(bytes(schema), reg))
    return _compact(_compat_deref_json_schema(json.loads(_dec(bytes(schema)))))


def _compat_deref_json_schema(schema: dict) -> dict:
    """Port of kosong/utils/jsonschema.py::deref_json_schema."""
    import copy as _copy

    full_schema = _copy.deepcopy(schema)
    visited: set[str] = set()

    def resolve_pointer(pointer: str):
        if pointer == "#":
            return True, full_schema
        current = full_schema
        for raw_part in pointer[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                if part not in current:
                    return False, None
                current = current[part]
            elif isinstance(current, list):
                if not (part == "0" or (part.isdigit() and part[0] != "0")):
                    return False, None
                index = int(part)
                if index >= len(current):
                    return False, None
                current = current[index]
            else:
                return False, None
        return True, current

    def traverse(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref == "#" or ref.startswith("#/"):
                    if ref in visited:
                        return node
                    found, target = resolve_pointer(ref)
                    if found:
                        visited.add(ref)
       
                        resolved = traverse(target)
                        visited.discard(ref)
                        if isinstance(resolved, dict):
                            merged = dict(resolved)
                            for key, value in node.items():
                                if key == "$ref":
                                    continue
                                merged[key] = traverse(value)
                            return merged
                        return resolved
                return node
            return {key: traverse(value) for key, value in node.items()}
        if isinstance(node, list):
            return [traverse(item) for item in node]
        return node

    resolved = traverse(full_schema)

    def has_unresolved(node, bucket_key: str) -> bool:
        if isinstance(node, list):
            return any(has_unresolved(child, bucket_key) for child in node)
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(f"#/{bucket_key}/"):
                return True
            return any(
                has_unresolved(value, bucket_key)
                for key, value in node.items()
                if key != bucket_key
            )
        return False

    if not has_unresolved(resolved, "$defs"):
        resolved.pop("$defs", None)
    if not has_unresolved(resolved, "definitions"):
        resolved.pop("definitions", None)
    return resolved


def ensure_property_types(schema: bytes) -> bytes:
    """Deep copy with an explicit `type` on every nested property schema."""
    if _USE:
        return bytes(_native.json.ensure_property_types(bytes(schema)))
    return _compact(_compat_ensure_property_types(json.loads(_dec(bytes(schema)))))


_TYPE_SKIP_KEYS = {"$ref", "allOf", "anyOf", "else", "if", "not", "oneOf", "then"}
_CHILD_SLOTS = (
    ("$defs", "map"), ("definitions", "map"), ("dependencies", "map"),
    ("dependentSchemas", "map"), ("patternProperties", "map"), ("properties", "map"),
    ("additionalItems", "single"), ("additionalProperties", "single"),
    ("contains", "single"), ("contentSchema", "single"), ("else", "single"),
    ("if", "single"), ("not", "single"), ("propertyNames", "single"),
    ("then", "single"), ("unevaluatedItems", "single"),
    ("unevaluatedProperties", "single"), ("allOf", "array"), ("anyOf", "array"),
    ("oneOf", "array"), ("prefixItems", "array"), ("items", "schema-or-array"),
)
_OBJECT_KEYS = {"additionalProperties", "dependentRequired", "dependentSchemas",
                "maxProperties", "minProperties", "patternProperties",
                "properties", "required", "unevaluatedProperties"}
_ARRAY_KEYS = {"additionalItems", "contains", "items", "maxContains", "maxItems",
               "minContains", "minItems", "prefixItems", "unevaluatedItems",
               "uniqueItems"}
_STRING_KEYS = {"contentEncoding", "contentMediaType", "contentSchema", "format",
                "maxLength", "minLength", "pattern"}
_NUMERIC_KEYS = {"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum",
                 "multipleOf"}


def _classify(value) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return None


def _infer_from_values(values: list) -> str:
    inferred = {_classify(v) for v in values}
    if None in inferred:
        return "string"
    if len(inferred) == 1:
        return next(iter(inferred))
    if inferred == {"integer", "number"}:
        return "number"
    return "string"


def _try_single(values: list) -> str | None:
    inferred = {_classify(v) for v in values}
    if None in inferred:
        return None
    if "number" in inferred:
        inferred.discard("integer")
    if len(inferred) == 1:
        return next(iter(inferred))
    return None


def _infer_from_structure(node: dict) -> str:
    if any(k in node for k in _OBJECT_KEYS):
        return "object"
    if any(k in node for k in _ARRAY_KEYS):
        return "array"
    if any(k in node for k in _STRING_KEYS):
        return "string"
    if any(k in node for k in _NUMERIC_KEYS):
        return "number"
    return "string"


def _compat_ensure_property_types(schema: dict) -> dict:
    import copy as _copy

    result = _copy.deepcopy(schema)

    def recurse(node) -> None:
        if not isinstance(node, dict):
            return
        for key, kind in _CHILD_SLOTS:
            value = node.get(key)
            if kind == "single":
                if isinstance(value, dict):
                    normalize(value)
            elif kind == "array":
                if isinstance(value, list):
                    for item in value:
                        normalize(item)
            elif kind == "map":
                if isinstance(value, dict):
                    for item in value.values():
                        normalize(item)
            else:
                if isinstance(value, dict):
                    normalize(value)
                elif isinstance(value, list):
                    for item in value:
                        normalize(item)

    def normalize(node) -> None:
        if not isinstance(node, dict):
            return
        has_skip = any(k in node for k in _TYPE_SKIP_KEYS)
        if "type" not in node and not has_skip:
            enum_values = node.get("enum")
            if isinstance(enum_values, list) and enum_values:
                node["type"] = _infer_from_values(enum_values)
            elif "const" in node:
                node["type"] = _infer_from_values([node["const"]])
            else:
                node["type"] = _infer_from_structure(node)
        elif not has_skip and isinstance(node.get("type"), str):
            enum_values = node.get("enum")
            values = None
            if isinstance(enum_values, list) and enum_values:
                values = enum_values
            elif "const" in node:
                values = [node["const"]]
            if values is not None:
                inferred = _try_single(values)
                if inferred is not None and node["type"] != inferred:
                    node["type"] = inferred
                    if inferred != "object":
                        for k in _OBJECT_KEYS:
                            node.pop(k, None)
                    if inferred != "array":
                        for k in _ARRAY_KEYS:
                            node.pop(k, None)
        recurse(node)

    recurse(result)
    return result
