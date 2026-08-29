"""Terminal printing primitives for kimix (P8: extracted from kimix.base).

Color enums, ANSI helpers, the custom ``print``, ``PrintStream`` and the
``print_*`` convenience functions. Import from here instead of
``kimix.base`` for new code.
"""

from __future__ import annotations

import functools
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from kimi_cli.native_loader import (
    get_compat as _native_get_compat,
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_STREAM = _native_get_module("stream")
# Pure-Python reference implementation (canonical copy lives in the shim);
# resolved lazily to avoid an import-time dependency on the shim package.
_COMPAT_STREAM = None


def _compat_stream():
    global _COMPAT_STREAM
    if _COMPAT_STREAM is None:
        _COMPAT_STREAM = _native_get_compat("stream")
    return _COMPAT_STREAM

_threads: list[threading.Thread] = []
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# 1a. Switch the Windows console code page to UTF-8 (CP 65001).
#     This ensures that child processes reading from the console
#     (including PowerShell's own [Console]::OutputEncoding when
#     not overridden) see UTF-8 rather than cp1252.  The
#     per-subprocess [Console]::OutputEncoding preamble is a
#     belt-and-suspenders complement; this system-level setting
#     catches everything else.
try:
    import ctypes
    _CP_UTF8 = 65001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(_CP_UTF8)
    kernel32.SetConsoleOutputCP(_CP_UTF8)
except Exception:
    # Non-fatal — the per-subprocess preamble still works.
    pass
class MessageType(Enum):
    """Message type for print_agent_json output function."""
    Text = "text"
    Thinking = "thinking"
    ToolCalling = "tool_calling"
    ToolCallingPart = "tool_calling_part"
    ToolResult = "tool_result"


class Color(Enum):
    """ANSI color codes for foreground colors."""
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97


class BgColor(Enum):
    """ANSI color codes for background colors."""
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47
    BRIGHT_BLACK = 100
    BRIGHT_RED = 101
    BRIGHT_GREEN = 102
    BRIGHT_YELLOW = 103
    BRIGHT_BLUE = 104
    BRIGHT_MAGENTA = 105
    BRIGHT_CYAN = 106
    BRIGHT_WHITE = 107


class Style(Enum):
    """ANSI style codes."""
    RESET = 0
    BOLD = 1
    DIM = 2
    ITALIC = 3
    UNDERLINE = 4
    BLINK = 5
    REVERSE = 7
    HIDDEN = 8
    STRIKETHROUGH = 9


@dataclass(frozen=True)
class Color256:
    """256-color mode (8-bit) foreground color."""
    value: int


@dataclass(frozen=True)
class BgColor256:
    """256-color mode (8-bit) background color."""
    value: int


@dataclass(frozen=True)
class TrueColor:
    """24-bit true color (RGB) foreground."""
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, hex_color: str) -> "TrueColor":
        hex_color = hex_color.lstrip("#")
        return cls(
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )


@dataclass(frozen=True)
class BgTrueColor:
    """24-bit true color (RGB) background."""
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, hex_color: str) -> "BgTrueColor":
        hex_color = hex_color.lstrip("#")
        return cls(
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )


# Common 256-color grayscale colors (232-255)
GRAY_NEAR_BLACK = Color256(232)
GRAY_DARK = Color256(240)
GRAY = Color256(245)
GRAY_LIGHT = Color256(250)

# Common true color grayscale
TRUE_GRAY = TrueColor(128, 128, 128)


def _strip_ansi(text: str) -> str:
    if "\x1b" not in text:
        return text
    # Native acceleration: kimix_native.stream.strip_ansi uses the identical
    # ANSI escape pattern; the pure-Python body below is unchanged.
    if _native_use_native("STREAM") and _NATIVE_STREAM is not None:
        return _NATIVE_STREAM.strip_ansi(text)
    return _compat_stream()._ANSI_ESCAPE_RE.sub("", text)


def _sgr_end(word: str, i: int, end: int) -> int:
    """If an SGR sequence (``\\x1b[<digits/semicolons>m``) starts at ``i``,
    return its end offset; otherwise return -1.

    This is exactly the sequence shape emitted by ``colorful_text`` /
    ``_ansi_prefix``, so the common colored-output path needs no regex.
    """
    j = i + 2
    while j < end and (word[j].isdigit() or word[j] == ';'):
        j += 1
    return j + 1 if j < end and word[j] == 'm' else -1


def _ends_with_newline(word: str) -> bool:
    """Return whether ``word`` ends with a newline, ignoring ANSI sequences.

    Equivalent to ``_strip_ansi(word).endswith('\\n')`` but avoids the regex
    substitution for the common cases (plain text, or text wrapped in SGR
    color sequences — which is how ``colorful_text`` emits colored output).
    Falls back to exact stripping for anything else (OSC sequences,
    malformed escapes, ...).
    """
    end = len(word)
    for _ in range(8):
        if end == 0:
            return False
        if word[end - 1] == '\n':
            return True
        i = word.rfind('\x1b[', 0, end)
        if i < 0:
            break
        j = _sgr_end(word, i, end)
        if j < 0:
            break  # Non-SGR CSI or malformed; exact fallback below.
        if j < end:
            # Something follows the last SGR sequence. If it contains no
            # ESC it is literal text whose last char is not a newline.
            if '\x1b' not in word[j:end]:
                return False
            break  # e.g. an OSC/APC sequence; exact fallback below.
        end = i
    return _strip_ansi(word[:end]).endswith('\n')


_colorful_print = True
_print_func: Callable = print


def _resolve_native_print() -> Callable | None:
    """Resolve ``runtime_py.print.native_print`` (no ``kimix_native.print`` shim yet).

    The compiled extension is reached through the shim's ``_native`` handle —
    the same handle every ``kimix_native.<kernel>`` shim binds. Returns None
    when native is unavailable or the runtime predates the print submodule, in
    which case printing stays on the builtin ``print``.
    """
    try:
        import kimix_native as _kn
        native = getattr(_kn, "_native", None)
        if native is None:
            return None
        return getattr(getattr(native, "print", None), "native_print", None)
    except Exception:
        return None


_NATIVE_PRINT = _resolve_native_print()


def _native_print_func(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    """Native ``print`` — thin forwarder; all logic lives in C++ now.

    The binding used to receive pre-formatted bytes; it has since absorbed
    the whole builtin-print contract (``str()`` coercion, ``sep``/``end``
    joining with the usual ``None`` defaults, ``surrogatepass`` UTF-8
    encoding and the non-stdout fallback to ``builtins.print``), so this
    wrapper only forwards the arguments to keep the hot path at one hop.
    Writes go through ``runtime_py.print.native_print`` — GIL released,
    fflushed when ``flush=True``.
    """
    _NATIVE_PRINT(*values, sep=sep, end=end, file=file, flush=flush)


if _native_use_native("PRINT") and _NATIVE_PRINT is not None:
    _print_func = _native_print_func


def print(*values: object, sep: str | None = " ", end: str | None = "\n", file: Any = None, flush: bool = False):
    _print_func(*values, sep=sep, end=end, file=file, flush=flush)


@functools.lru_cache(maxsize=256)
def _ansi_prefix(
    fg_value: int | str | None,
    bg_value: int | str | None,
    styles_tuple: tuple[int, ...],
) -> str | None:
    codes: list[str] = []
    if styles_tuple:
        codes.extend(map(str, styles_tuple))
    if fg_value is not None:
        codes.append(str(fg_value))
    if bg_value is not None:
        codes.append(str(bg_value))
    if codes:
        return f"\033[{';'.join(codes)}m"
    return None


def _resolve_fg(color: Color | Color256 | TrueColor | None) -> int | str | None:
    if color is None:
        return None
    if isinstance(color, Color):
        return color.value
    if isinstance(color, Color256):
        return f"38;5;{color.value}"
    if isinstance(color, TrueColor):
        return f"38;2;{color.r};{color.g};{color.b}"
    return None


def _resolve_bg(color: BgColor | BgColor256 | BgTrueColor | None) -> int | str | None:
    if color is None:
        return None
    if isinstance(color, BgColor):
        return color.value
    if isinstance(color, BgColor256):
        return f"48;5;{color.value}"
    if isinstance(color, BgTrueColor):
        return f"48;2;{color.r};{color.g};{color.b}"
    return None


def colorful_text(
    text: str,
    fg: Color | Color256 | TrueColor | None = None,
    bg: BgColor | BgColor256 | BgTrueColor | None = None,
    styles: list[Style] | None = None,
) -> str:
    if not _colorful_print:
        return text
    prefix = _ansi_prefix(
        _resolve_fg(fg),
        _resolve_bg(bg),
        tuple(s.value for s in styles) if styles else (),
    )
    if prefix:
        text = f"{prefix}{text}\033[0m"
    return text


def colorful_print(
    text: str,
    fg: Color | Color256 | TrueColor | None = None,
    bg: BgColor | BgColor256 | BgTrueColor | None = None,
    styles: list[Style] | None = None,
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    if not _colorful_print:
        _print_func(text, end=end, file=file, flush=flush)
        return
    text = colorful_text(text, fg, bg, styles)
    _print_func(text, end=end, file=file, flush=flush)


class StreamPrintState(Enum):
    Text = 0
    Thinking = 1
    Other = 2


class PrintStream:
    """A stream wrapper that tracks whether the last printed character was a newline.

    Provides print_word(word) that automatically inserts a leading
    newline when the previous output didn't end with one.
    """

    def __init__(self, print_func: Callable = _print_func) -> None:
        self._print_func = print_func
        self._last_char_was_newline = True
        self._state = StreamPrintState.Other

    def print_word(self, word: str, require_new_line: bool, raw_word: str | None = None, flush: bool = False) -> None:
        """Print a word, auto-inserting a leading newline when the previous
        output didn't end with one. Pass ``flush=True`` for live streaming output."""
        if not word:
            if require_new_line and not self._last_char_was_newline:
                self._print_func('', end='\n', flush=flush)
                self._last_char_was_newline = True
            return

        if require_new_line and not self._last_char_was_newline:
            self._print_func('', end='\n', flush=flush)

        self._print_func(word, end='', flush=flush)
        check_word = raw_word if raw_word is not None else word
        self._last_char_was_newline = _ends_with_newline(check_word)

    def colorful_print_word(
            self, word: str,
            require_new_line: bool,
            fg: Color | Color256 | TrueColor | None = None,
            bg: BgColor | BgColor256 | BgTrueColor | None = None,
            styles: list[Style] | None = None,
            flush: bool = False) -> None:
        self.print_word(colorful_text(word, fg, bg, styles),
                        require_new_line=require_new_line, raw_word=word, flush=flush)


_quiet = False


def print_success(text: str, end: str = "\n") -> None:
    """Print success message in green."""
    colorful_print(text, fg=Color.BRIGHT_GREEN, styles=[Style.BOLD], end=end)


def print_string(text: str, end: str = "\n", file: Any = None, flush: bool = False) -> None:
    _print_func(text, end=end, file=file, flush=flush)


def print_error(text: str, end: str = "\n") -> None:
    """Print error message in red."""
    colorful_print(text, fg=Color.BRIGHT_RED, styles=[Style.BOLD], end=end)


def print_warning(text: str, end: str = "\n") -> None:
    """Print warning message in yellow."""
    colorful_print(text, fg=Color.BRIGHT_YELLOW, styles=[Style.BOLD], end=end)


def print_info(text: str, end: str = "\n") -> None:
    """Print info message in blue."""
    colorful_print(text, fg=Color.BRIGHT_MAGENTA, end=end)


def print_debug(text: str, end: str = "\n") -> None:
    """Print debug message in cyan."""
    if _quiet:
        return
    colorful_print(text, fg=Color.BRIGHT_CYAN, end=end)


def _process_lru() -> None:
    """Limit the number of threads to 8 by waiting and removing completed ones."""
    global _threads
    MAX_PROCESSES = 8

    _threads = [p for p in _threads if p.is_alive()]

    while len(_threads) >= MAX_PROCESSES:
        time.sleep(0.1)
        _threads = [p for p in _threads if p.is_alive()]


_stream = PrintStream()
