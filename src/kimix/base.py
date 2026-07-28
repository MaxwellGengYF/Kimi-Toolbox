from __future__ import annotations

import asyncio
import functools
import io
import os
import regex as re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import orjson
from kimi_cli.wire.types import (
    ApprovalRequest,
    BackgroundTaskDisplayBlock,
    BriefDisplayBlock,
    CompactionBegin,
    CompactionEnd,
    DiffDisplayBlock,
    DisplayBlock,
    ShellDisplayBlock,
    StepBegin,
    StepInterrupted,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    ToolCall,
    ToolCallPart,
    ToolResult,
    UnknownDisplayBlock,
)

if TYPE_CHECKING:
    from kimi_agent_sdk import Session

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


_ANSI_ESCAPE = re.compile(
    r"\x1B(?:"
    r"\][^\x07\x1B]*(?:\x07|\x1B\\)|"  # OSC sequences (BEL or ST terminated)
    r"[P^_][^\x07\x1B]*(?:\x07|\x1B\\)|"  # DCS / PM / APC sequences
    r"[@-Z\\-_]|"              # Single-character Fe sequences
    r"\[[0-?]*[ -/]*[@-~]"      # CSI sequences
    r")"
)


def _strip_ansi(text: str) -> str:
    if "\x1b" not in text:
        return text
    return _ANSI_ESCAPE.sub("", text)


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
_text_buffer: io.StringIO | None = None


_TOOL_TYPES = (ToolCall, ToolCallPart, ToolResult)
_PRINT_AGENT_JSON_MESSAGE_TYPE_ATTR = "_kimix_print_agent_json_message_type"


def _message_transition_type(wire_msg: Any) -> MessageType | None:
    if isinstance(wire_msg, TextPart):
        return MessageType.Text
    if isinstance(wire_msg, ThinkPart):
        return MessageType.Thinking
    if isinstance(wire_msg, _TOOL_TYPES):
        return MessageType.ToolCalling
    return None


def _print_transition_usage(session: Session, message_type: MessageType | None) -> None:
    if message_type is None:
        return
    previous_type = getattr(session, _PRINT_AGENT_JSON_MESSAGE_TYPE_ATTR, None)
    if previous_type is not None and previous_type != message_type:
        split_str = '=' * 20
        usage = percentage_and_token(session)
        left = f"{split_str} Context usage: {usage} "
        target_width = 80
        right_split = '=' * max(target_width - len(left), 1)
        _stream.colorful_print_word(
            f"{left}{right_split}\n",
            fg=GRAY,
            require_new_line=True,
            flush=True,
        )
    setattr(session, _PRINT_AGENT_JSON_MESSAGE_TYPE_ATTR, message_type)


def _format_display_blocks(display: list[Any]) -> str | None:
    """Format display blocks into a colored terminal string.

    Returns a string ending with ``\n`` so that ``PrintStream.print_word``
    correctly tracks ``_last_char_was_newline`` after the output.
    """
    if not display:
        return None
    parts: list[str] = []
    for block in display:
        if isinstance(block, BriefDisplayBlock):
            if block.text:
                parts.append(colorful_text(block.text, fg=Color.BRIGHT_BLACK))
        elif isinstance(block, DiffDisplayBlock):
            parts.append(colorful_text(
                f"Diff: {block.path}", fg=Color.BRIGHT_YELLOW))
            for line in block.old_text.splitlines():
                parts.append(colorful_text(f"- {line}", fg=Color.BRIGHT_RED))
            for line in block.new_text.splitlines():
                parts.append(colorful_text(f"+ {line}", fg=Color.BRIGHT_GREEN))
        elif isinstance(block, TodoDisplayBlock):
            for item in block.items:
                status = item.status.replace("_", " ").lower()
                if status == "done":
                    parts.append(colorful_text(
                        f"- ~~{item.title}~~", fg=Color.BRIGHT_BLACK))
                elif status == "in progress":
                    parts.append(colorful_text(
                        f"- {item.title} \u2190", fg=Color.BRIGHT_YELLOW))
                else:
                    parts.append(colorful_text(
                        f"- {item.title}", fg=GRAY_LIGHT))
        elif isinstance(block, ShellDisplayBlock):
            parts.append(colorful_text(
                f"$ {block.command}", fg=Color.BRIGHT_BLUE))
        elif isinstance(block, BackgroundTaskDisplayBlock):
            parts.append(
                colorful_text(
                    f"[{block.status}] {block.task_id}: {block.description}", fg=Color.BRIGHT_BLACK)
            )
        elif isinstance(block, UnknownDisplayBlock):
            parts.append(colorful_text(str(block.data), fg=Color.BRIGHT_BLACK))
        elif isinstance(block, DisplayBlock):
            data = block.model_dump()
            if data:
                parts.append(colorful_text(str(data), fg=GRAY_LIGHT))
    if not parts:
        return None
    return "\n".join(parts) + "\n"


def _format_tool_result(result: ToolResult) -> str:
    """Format a ToolResult for the output function."""
    rv = result.return_value
    return rv.message or ""


_LAST_TOOL_CALL_KEY = "_kimix_last_tool_call"
_TOOL_CALL_STREAM_KEY = "_kimix_tool_call_stream"
_TOOL_HEADER_PRINTED_KEY = "_kimix_tool_header_printed"
_TOOL_CALL_PART_PENDING_KEY = "_kimix_tool_call_part_pending"
_TOOL_CALL_PART_EMITTED_LEN_KEY = "_kimix_tool_call_part_emitted_len"
_TOOL_CALL_MERGE_TARGET_KEY = "_kimix_tool_call_merge_target"
# The id of the tool call whose header was last printed (paired with
# _TOOL_HEADER_PRINTED_KEY).  This makes the header-printed gate work
# per-call, which is essential when parallel tool calls are in flight:
# finishing one call's arguments and printing its header must not prevent
# the next call's header from being printed.
_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY = "_kimix_tool_call_header_printed_tc_id"

# Minimum payload size before a cumulative ``ToolCallingPart`` snapshot is
# emitted to ``output_function``; afterwards the emission threshold doubles
# each time. Consumers replace (not append) the previous snapshot, so
# coalescing intermediate snapshots is invisible while avoiding O(N^2)
# bytes for long streamed tool arguments.
_TOOL_CALL_PART_MIN_EMIT_BYTES = 4096


def _flush_tool_call_part_output(
    session: Session,
    output_function: Callable[[str, MessageType], Any] | None,
) -> None:
    """Emit any pending coalesced ``ToolCallingPart`` snapshot.

    Called when the tool call finishes (a non-``ToolCallPart`` wire message
    arrives) or is superseded by a new ``ToolCall``, guaranteeing the final
    full snapshot is always delivered exactly once.
    """
    if output_function is None:
        return
    tmp_data = getattr(session, "_tmp_data", None)
    if not tmp_data or not tmp_data.pop(_TOOL_CALL_PART_PENDING_KEY, None):
        return
    last_tc: ToolCall | None = tmp_data.get(_LAST_TOOL_CALL_KEY)
    if last_tc is None:
        return
    payload = f"{last_tc.function.name} {last_tc.function.arguments or ''}"
    output_function(payload, MessageType.ToolCallingPart)
    tmp_data[_TOOL_CALL_PART_EMITTED_LEN_KEY] = len(payload)

# Mapping from argument-key aliases (pydantic Field(alias=...)) to the
# canonical key used for streaming decisions, color lookup, and display.
_ARG_KEY_ALIASES: dict[str, str] = {
    "old_string": "old",
    "new_string": "new",
    "text": "content",
    "source_code": "code",
    "code_file": "code",
    "task": "prompt",
    "file_path": "path",
    "cmd": "command",
    "session": "session_id",
    "edits": "edit",
    "items": "todos",
    "block": "wait",
    "token_kill": "deduplicate_output",
}


def _canonical_key(key: str) -> str:
    """Return the canonical form of *key*, resolving pydantic Field aliases."""
    return _ARG_KEY_ALIASES.get(key, key)


# Tool names eligible for streaming argument display.
# Only these tools benefit from the incremental decoded output;
# all other tools (e.g. Grep, Powershell, Bash) use the legacy
# compact one-line format regardless of the stream_tool_args flag.
_STREAM_TOOL_NAMES = frozenset({
    "WriteFile",
    "WritePlan",
    "Python",
    "Agent",
    "EditFile",
    "EditPlan",
    "Run",
    "Powershell",
    "Bash",
    # "Goal" and "RunGoal" removed — both are deleted
})

# Tool-call argument keys whose (potentially very long) string values are
# printed decoded, token by token, as they stream in from the LLM.
# Aliases (e.g. old_string, new_string, text, source_code, task) are
# canonicalized via _canonical_key() before lookup.
_STREAM_ARG_KEYS = frozenset({
    "content",       # WriteFile / WritePlan
    "code",          # Python
    "prompt",        # Agent
    "old", "new",    # EditFile / EditPlan edit items
    "question", "context", "instruction",
    "command",       # Run / Powershell / Bash
})

# Tool-call argument keys whose streamed string values are printed *inline*
# after the tool header (space separator, no "key:\n" label).  Used so that
# ``⚡ Powershell Get-Date`` stays on one line instead of breaking the
# command onto a new ``command:\n`` line.
#
# Keys are canonical (resolved via _canonical_key()), so the "cmd" alias
# is automatically covered — no need to list it separately.
_INLINE_ARG_KEYS = frozenset({
    "command",       # Run / Powershell / Bash (alias "cmd" also covered)
})

# Foreground color for the "⚡ ToolName" header printed when a tool call
# starts. All tool names use the same BRIGHT_MAGENTA color.
_TOOL_HEADER_COLOR: Color = Color.BRIGHT_MAGENTA


def _tool_header_color(name: str) -> Color:
    """Return the foreground color for the tool-call header '⚡ Name'.

    All tool names use BRIGHT_MAGENTA; the result colors (success green,
    failure red) are handled separately when the tool result is printed.
    """
    del name  # unused: every tool header uses the same color
    return _TOOL_HEADER_COLOR


def _format_tool_args(name: str, args: str | None) -> str | None:
    """Format tool arguments into a friendly one-line sentence."""
    if args is None:
        return None
    if args == "":
        return ""
    try:
        parsed = orjson.loads(args)
    except (orjson.JSONDecodeError, TypeError):
        return None

    try:
        if not isinstance(parsed, dict):
            return orjson.dumps(parsed).decode("utf-8")

        def _fmt(v: Any, max_len: int = 60) -> str:
            if v is None:
                return "None"
            s = str(v)
            if len(s) > max_len:
                return s[:max_len] + "..."
            return s

        def _collect(*keys: str | tuple[str, ...], hide: set[str] | None = None) -> list[str]:
            hide = hide or set()
            parts: list[str] = []
            for key in keys:
                if isinstance(key, str):
                    # Single key lookup.
                    if key in parsed:
                        display = key
                        if key in hide:
                            parts.append(f"{display}: ...")
                        else:
                            parts.append(f"{display}: {_fmt(parsed[key])}")
                else:
                    # Tuple of alternative keys — use the first one present.
                    found_key = None
                    for alt in key:
                        if alt in parsed:
                            found_key = alt
                            break
                    if found_key is not None:
                        display = key[0]  # Use first alias as display name
                        if found_key in hide:
                            parts.append(f"{display}: ...")
                        else:
                            parts.append(f"{display}: {_fmt(parsed[found_key])}")
            return parts

        match name:
            case "Bash":
                return ", ".join(_collect(("command", "cmd"), "mode", "timeout", "interactive", "task_id", "wait_for_pattern", "max_lines", ("deduplicate_output", "token_kill")))
            case "Powershell":
                return ", ".join(_collect(("command", "cmd"), "mode", "timeout", "interactive", "task_id", "wait_for_pattern", "max_lines", ("deduplicate_output", "token_kill")))
            case "Run":
                return ", ".join(_collect(("command", "cmd"), "cwd", "timeout", "output_path", "env", "mode", "shell", "run_in_background", "task_id", "wait_for_pattern", "max_lines", ("deduplicate_output", "token_kill")))
            case "Python":
                return ", ".join(_collect(("code", "source_code"), "file", "output_path", "timeout", "mode", "run_in_background", "task_id", "wait_for_pattern", "max_lines", ("deduplicate_output", "token_kill"), hide={"code", "source_code"}))
            case "TaskOutput":
                return ", ".join(_collect("task_id", "action", ("wait", "block"), "timeout", "output_path", "kill"))
            case "TodoList":
                parts: list[str] = []
                if "todos" in parsed or "items" in parsed:
                    todos = parsed.get("todos", parsed.get("items"))
                    if todos is None:
                        parts.append("todos=None")
                    elif isinstance(todos, list):
                        parts.append(f"todos=[{len(todos)} items]")
                    else:
                        parts.append("todos=[1 item]")
                if parsed.get("mode") and parsed.get("mode") != "append":
                    parts.append(f"mode={parsed['mode']}")
                return ", ".join(parts)
            case "ReadFile":
                return ", ".join(_collect(("path", "file_path"), "line_offset", "n_lines", "max_char", "char_offset", "glob", "show_line_numbers"))
            case "EditFile":
                parts = []
                _path = parsed.get("path", parsed.get("file_path"))
                if _path is not None:
                    parts.append(f"path={_fmt(_path)}")
                if "edit" in parsed or "edits" in parsed:
                    edit = parsed.get("edit", parsed.get("edits"))
                    if edit is None:
                        parts.append("edit=None")
                    elif isinstance(edit, list):
                        parts.append(f"edit=[{len(edit)} edit(s)]")
                    else:
                        parts.append("edit=[1 edit]")
                return ", ".join(parts)
            case "WriteFile":
                return ", ".join(_collect(("path", "file_path"), ("content", "text"), "mode", "auto_fix_json", "mkdir", "show_diff", hide={"content", "text"}))
            case "Glob":
                return ", ".join(_collect("pattern", ("directory", "path"), "include_dirs", "respect_gitignore", "include_ignored", "verbose", "timeout"))
            case "Grep":
                return ", ".join(
                    _collect(
                        "pattern", "path", "glob", "output_mode",
                        ("before_context", "-B"), ("after_context", "-A"), ("context", "-C"),
                        ("line_number", "-n"), ("ignore_case", "-i"),
                        "type", "head_limit", "offset", "multiline", "include_ignored",
                    )
                )
            case "FetchURL":
                return ", ".join(_collect("url", "output_path"))
            case "Agent":
                return ", ".join(_collect(("prompt", "task"), ("session_id", "session"), "close_session", "history_format", "context_files"))
            case "AgentList":
                return ""
            case "AgentClose":
                return ", ".join(_collect(("session_id", "session")))
            case "WritePlan":
                return ", ".join(_collect(("content", "text"), "mode", hide={"content", "text"}))
            case "ReadPlan":
                return ", ".join(_collect("line_offset", "n_lines", "max_char", "char_offset"))
            case "EditPlan":
                parts = []
                if "edit" in parsed or "edits" in parsed:
                    edit = parsed.get("edit", parsed.get("edits"))
                    if edit is None:
                        parts.append("edit=None")
                    elif isinstance(edit, list):
                        parts.append(f"edit=[{len(edit)} edit(s)]")
                    else:
                        parts.append("edit=[1 edit]")
                return ", ".join(parts)
            case "AskParent":
                return ", ".join(_collect("question", "context"))
            case "ContextUsage":
                return ""
            case "Compact":
                return ", ".join(_collect("instruction", "mode"))
            case "Memory":
                return ", ".join(_collect("action", "topic", "query", "max_results", hide={"content"}))
            case _:
                return orjson.dumps(parsed).decode("utf-8")
    except TypeError:
        return None


class _ToolCallStreamPrinter:
    """Incrementally lexes streamed tool-call arguments JSON and prints
    decoded argument values live (token by token) as fragments arrive.

    Lifecycle: created when a ``ToolCall`` arrives; :meth:`feed` is called per
    ``ToolCallPart`` fragment; :meth:`finish` is called when the arguments JSON
    parses completely, when a new ``ToolCall`` supersedes this one, or when any
    non-``ToolCallPart`` wire message arrives (safety net for truncated or
    malformed JSON).

    Each argument — streamed or compact — is printed on its own line beneath
    the ``⚡ Name`` header. String values for keys in :data:`_STREAM_ARG_KEYS`
    are printed decoded, fragment by fragment, each in a per-key color from
    :attr:`_STREAM_KEY_COLORS` (fallback ``GRAY_LIGHT``). Other short scalar
    values are buffered and printed as compact ``key: value`` lines on
    completion.
    """

    # Lexer states.
    _EXPECT_KEY = 0
    _IN_KEY = 1
    _EXPECT_COLON = 2
    _EXPECT_VALUE = 3
    _IN_STRING = 4
    _IN_BARE = 5
    _AFTER_VALUE = 6
    _DONE = 7

    _SIMPLE_ESCAPES = {
        "n": "\n", "t": "\t", "r": "\r", '"': '"',
        "\\": "\\", "/": "/", "b": "\b", "f": "\f",
    }

    # Flush streamed output to the terminal at least every this many bytes.
    # Keeps long values visibly "live" without paying one terminal flush
    # (a syscall on real consoles/pipes) per LLM fragment.
    _FLUSH_INTERVAL_BYTES = 256
    _BARE_LITERALS = {"true": "True", "false": "False", "null": "None"}

    # Key -> foreground color for streamed argument values printed live by
    # _flush_emit. Keys not listed here fall back to GRAY_LIGHT.
    _STREAM_KEY_COLORS: dict[str, Color | Color256] = {
        "old": Color.BRIGHT_RED,
        "new": Color.BRIGHT_GREEN,
        "code": Color.BRIGHT_BLUE,
        "prompt": Color.BRIGHT_YELLOW,
        "question": Color.BRIGHT_YELLOW,
        "instruction": Color.BRIGHT_YELLOW,
        "content": Color.BRIGHT_BLACK,
        "context": GRAY,
        "source_code": Color.BRIGHT_CYAN,
        "text": GRAY_LIGHT,
        "task": Color.BRIGHT_YELLOW,
        "command": Color.BRIGHT_BLUE,
    }

    def __init__(self, tool_name: str, session: Session) -> None:
        self._tool_name = tool_name
        self._session = session
        self._state = self._EXPECT_VALUE
        self._stack: list[str] = []
        self._current_key = ""
        self._key_chars: list[str] = []
        self._value_chars: list[str] = []
        self._emit_chars: list[str] = []
        self._escape_buf = ""
        self._in_escape = False
        self._pending_high_surrogate: int | None = None
        self._string_streamed = False
        self._stream_color: Color | Color256 = GRAY_LIGHT  # resolved per-key in _begin_string_value
        self._json_parts: list[str] = []
        self._finished = False
        self._broken = False
        self._bytes_since_flush = 0

    @staticmethod
    def _stream_color_for_key(key: str) -> Color | Color256:
        """Return the foreground color for a streamed argument key."""
        return _ToolCallStreamPrinter._STREAM_KEY_COLORS.get(key, GRAY_LIGHT)

    # ------------------------------------------------------------------ API

    def feed(self, fragment: str) -> None:
        """Feed one raw JSON fragment; prints decoded output as it goes."""
        if self._finished:
            return
        if fragment:
            self._json_parts.append(fragment)
            if self._broken:
                _stream.colorful_print_word(
                    fragment, fg=GRAY_LIGHT, require_new_line=False, flush=True)
            else:
                try:
                    self._lex(fragment)
                    self._flush_emit()
                except Exception:
                    # Defensive fallback: never let a lexer error break output.
                    self._broken = True
                    _stream.colorful_print_word(
                        fragment, fg=GRAY_LIGHT, require_new_line=False, flush=True)
        self._check_complete()

    def finish(self) -> None:
        """Flush pending buffers, terminate the line and detach from the session."""
        if self._finished:
            return
        self._finished = True
        try:
            if self._in_escape and self._escape_buf:
                # Incomplete escape at end of input: emit verbatim.
                self._append_value_char(self._escape_buf)
                self._in_escape = False
                self._escape_buf = ""
            if self._pending_high_surrogate is not None:
                self._append_value_char("\ufffd")
                self._pending_high_surrogate = None
            if self._state == self._IN_STRING:
                if self._string_streamed:
                    self._flush_emit(flush=True)
                elif self._value_chars:
                    self._emit_compact("".join(self._value_chars) + "...")
            elif self._state == self._IN_BARE and self._value_chars:
                self._end_bare_value()
            else:
                self._flush_emit(flush=True)
        except Exception:
            pass
        _stream.print_word("", True, flush=True)
        _stream._state = StreamPrintState.Other
        # Release accumulated fragments promptly; they are no longer needed.
        self._json_parts.clear()
        if self._session._tmp_data.get(_TOOL_CALL_STREAM_KEY) is self:
            self._session._tmp_data.pop(_TOOL_CALL_STREAM_KEY, None)

    # ------------------------------------------------------------- internal

    def _check_complete(self) -> None:
        if not self._json_parts:
            return
        # Structural gate: only run the full JSON validation when the document
        # may actually be complete. Re-joining and re-parsing every accumulated
        # fragment on each feed() is O(N^2) in the number of fragments.
        if self._broken:
            # Lexer unavailable (defensive fallback): a complete JSON document
            # can only end right after a container close or a closing quote.
            tail = self._json_parts[-1].rstrip()
            if not tail or tail[-1] not in '}]"':
                return
        elif self._state != self._DONE:
            # The incremental lexer tracks container balance. The document can
            # only be complete once the outermost container has closed (_DONE),
            # or when a top-level string/bare value has just ended (empty stack).
            if self._stack or self._state not in (self._AFTER_VALUE, self._IN_BARE):
                return
        try:
            orjson.loads("".join(self._json_parts))
        except (orjson.JSONDecodeError, TypeError, ValueError):
            return
        self.finish()

    # Terminates a bare (unquoted) JSON value: comma, container close, or
    # JSON insignificant whitespace. Mirrors the set handled in _feed_char.
    _BARE_VALUE_TERMINATOR = re.compile(r"[,}\] \t\r\n]")

    def _lex(self, fragment: str) -> None:
        """Lex one raw JSON fragment.

        Fast paths consume "boring" spans in bulk via C-level ``str.find`` /
        regex search — string content without quotes or escapes, bare
        literals, and insignificant whitespace. Only boundary characters go
        through the per-char state machine, which keeps the emitted output
        byte-for-byte identical while cutting the per-char dispatch cost.
        """
        i = 0
        n = len(fragment)
        while i < n:
            if self._in_escape:
                self._feed_escape_char(fragment[i])
                i += 1
                continue
            state = self._state
            if state == self._DONE:
                # Characters after a complete document are ignored.
                return
            if state == self._IN_STRING or state == self._IN_KEY:
                # Bulk-consume up to the next quote or escape introducer.
                q = fragment.find('"', i)
                b = fragment.find('\\', i)
                if q == -1:
                    j = b if b != -1 else n
                elif b == -1:
                    j = q
                else:
                    j = q if q < b else b
                if j > i:
                    self._append_value_char(fragment[i:j])
                    i = j
                    continue
            elif state == self._IN_BARE:
                m = self._BARE_VALUE_TERMINATOR.search(fragment, i)
                j = m.start() if m is not None else n
                if j > i:
                    self._value_chars.append(fragment[i:j])
                    i = j
                    continue
            elif fragment[i] in ' \t\r\n':
                # Insignificant whitespace outside strings / bare values is
                # ignored by every remaining state; skip without dispatch.
                i += 1
                continue
            self._feed_char(fragment[i])
            i += 1

    def _feed_char(self, ch: str) -> None:
        if self._in_escape:
            self._feed_escape_char(ch)
            return
        state = self._state
        if state == self._DONE:
            return
        if state == self._EXPECT_KEY:
            if ch == '"':
                self._key_chars = []
                self._state = self._IN_KEY
            elif ch == '}':
                self._close_container()
        elif state == self._IN_KEY:
            if ch == '\\':
                self._in_escape = True
                self._escape_buf = "\\"
            elif ch == '"':
                self._current_key = "".join(self._key_chars)
                self._state = self._EXPECT_COLON
            else:
                self._key_chars.append(ch)
        elif state == self._EXPECT_COLON:
            if ch == ':':
                self._state = self._EXPECT_VALUE
        elif state == self._EXPECT_VALUE:
            if ch == '"':
                self._begin_string_value()
            elif ch == '{':
                self._stack.append('{')
                self._state = self._EXPECT_KEY
            elif ch == '[':
                self._stack.append('[')
            elif ch == ']' or ch == '}':
                self._close_container()
            elif ch not in ' \t\r\n':
                self._value_chars = [ch]
                self._state = self._IN_BARE
        elif state == self._IN_STRING:
            if ch == '\\':
                self._in_escape = True
                self._escape_buf = "\\"
            elif ch == '"':
                self._end_string_value()
            else:
                self._append_value_char(ch)
        elif state == self._IN_BARE:
            if ch == ',':
                self._end_bare_value()
                self._after_comma()
            elif ch == '}' or ch == ']':
                self._end_bare_value()
                self._close_container()
            elif ch in ' \t\r\n':
                self._end_bare_value()
            else:
                self._value_chars.append(ch)
        elif state == self._AFTER_VALUE:
            if ch == ',':
                self._after_comma()
            elif ch == '}' or ch == ']':
                self._close_container()

    def _feed_escape_char(self, ch: str) -> None:
        self._escape_buf += ch
        buf = self._escape_buf
        if len(buf) == 2 and buf[1] != 'u':
            decoded = self._SIMPLE_ESCAPES.get(ch, ch)
            self._reset_escape()
            self._append_value_char(decoded)
        elif buf.startswith("\\u") and len(buf) == 6:
            self._reset_escape()
            try:
                cp = int(buf[2:], 16)
            except ValueError:
                self._append_value_char(buf)
                return
            self._handle_code_point(cp)
        elif len(buf) > 6 or (len(buf) > 2 and not buf.startswith("\\u")):
            # Should not happen; emit verbatim and recover.
            self._reset_escape()
            self._append_value_char(buf)

    def _reset_escape(self) -> None:
        self._in_escape = False
        self._escape_buf = ""

    def _handle_code_point(self, cp: int) -> None:
        hi = self._pending_high_surrogate
        if hi is not None:
            self._pending_high_surrogate = None
            if 0xDC00 <= cp <= 0xDFFF:
                self._append_value_char(
                    chr(0x10000 + ((hi - 0xD800) << 10) + (cp - 0xDC00)))
                return
            self._append_value_char("\ufffd")
        if 0xD800 <= cp <= 0xDBFF:
            self._pending_high_surrogate = cp
        elif 0xDC00 <= cp <= 0xDFFF:
            self._append_value_char("\ufffd")
        else:
            self._append_value_char(chr(cp))

    def _append_value_char(self, s: str) -> None:
        if self._state == self._IN_KEY:
            self._key_chars.append(s)
        elif self._string_streamed:
            self._emit_chars.append(s)
        else:
            self._value_chars.append(s)

    def _begin_string_value(self) -> None:
        self._current_key = _canonical_key(self._current_key)
        self._string_streamed = self._current_key in _STREAM_ARG_KEYS
        self._value_chars = []
        self._state = self._IN_STRING
        if self._string_streamed:
            self._stream_color = self._stream_color_for_key(self._current_key)
            if self._current_key in _INLINE_ARG_KEYS:
                # Inline: print a space, no label — value follows on same line.
                _stream.colorful_print_word(
                    " ", fg=GRAY, require_new_line=False, flush=True)
            else:
                _stream.colorful_print_word(
                    f"{self._separator()}{self._current_key}:\n",
                    fg=GRAY, require_new_line=False, flush=True)

    def _end_string_value(self) -> None:
        if self._string_streamed:
            if self._pending_high_surrogate is not None:
                self._emit_chars.append("\ufffd")
                self._pending_high_surrogate = None
            self._flush_emit(flush=True)
        else:
            self._emit_compact("".join(self._value_chars))
        self._value_chars = []
        self._state = self._AFTER_VALUE

    def _end_bare_value(self) -> None:
        text = "".join(self._value_chars)
        self._value_chars = []
        self._emit_compact(self._BARE_LITERALS.get(text, text))
        self._state = self._AFTER_VALUE

    def _emit_compact(self, text: str) -> None:
        if len(text) > 60:
            text = text[:60] + "..."
        canonical_key = _canonical_key(self._current_key) if self._current_key else ""
        segment = f"{self._separator()}{canonical_key}:\n{text}" if canonical_key \
            else f"{self._separator()}{text}"
        _stream.colorful_print_word(
            segment, fg=Color.BRIGHT_MAGENTA, require_new_line=False, flush=True)

    def _separator(self) -> str:
        # Each tool argument starts on its own line beneath the tool header.
        return "\n"

    def _flush_emit(self, flush: bool = False) -> None:
        """Print buffered decoded output.

        ``flush=True`` forces a terminal flush (value boundaries, finish);
        otherwise a flush happens only once every
        :data:`_FLUSH_INTERVAL_BYTES` streamed bytes. Printed bytes are
        identical either way — only the flush cadence changes.
        """
        if not self._emit_chars:
            return
        chunk = "".join(self._emit_chars)
        self._emit_chars = []
        self._bytes_since_flush += len(chunk)
        if flush or self._bytes_since_flush >= self._FLUSH_INTERVAL_BYTES:
            self._bytes_since_flush = 0
            flush = True
        _stream.colorful_print_word(
            chunk, fg=self._stream_color, require_new_line=False, flush=flush)

    def _after_comma(self) -> None:
        if self._stack and self._stack[-1] == '{':
            self._state = self._EXPECT_KEY
        elif self._stack:
            self._state = self._EXPECT_VALUE

    def _close_container(self) -> None:
        if self._stack:
            self._stack.pop()
        self._state = self._AFTER_VALUE if self._stack else self._DONE


def _json_tail_may_complete(args: str) -> bool:
    """Cheap structural gate before attempting a full JSON parse of an
    accumulated (usually still incomplete) tool-arguments string.

    Tool-call arguments are JSON objects, so a complete document can only end
    with ``}`` (or ``]`` / ``"`` for exotic non-object args). Skipping the
    parse attempt for every other fragment avoids re-parsing the whole
    growing string per fragment — O(N^2) failed parses per tool call.
    """
    tail = args[-256:].rstrip()
    return bool(tail) and tail[-1] in '}]"'


def _print_compact_tool_header(session: Session, last_tc: ToolCall) -> bool:
    """Print the legacy compact ``⚡ name args`` header once the accumulated
    arguments parse as complete JSON. No-op if already printed (for this
    specific tool call) or the arguments are still incomplete/invalid.

    Returns ``True`` if the header was actually printed, ``False`` otherwise.
    """
    tmp_data = session._tmp_data
    # Per-call gate: only skip if the flag is set AND it belongs to THIS call.
    if tmp_data.get(_TOOL_HEADER_PRINTED_KEY) and tmp_data.get(_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY) == last_tc.id:
        return False
    args = last_tc.function.arguments
    if not args:
        return False
    formatted = _format_tool_args(last_tc.function.name, args)
    if formatted:
        _stream.colorful_print_word(
            f"⚡ {last_tc.function.name} {formatted}",
            fg=_tool_header_color(last_tc.function.name), require_new_line=True, flush=True)
        tmp_data[_TOOL_HEADER_PRINTED_KEY] = True
        tmp_data[_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY] = last_tc.id
        return True
    return False


def _finish_tool_call_stream(session: Session) -> None:
    """Finish and remove any active tool-call stream printer for the session."""
    tmp_data = getattr(session, "_tmp_data", None)
    if not tmp_data:
        return
    printer = tmp_data.pop(_TOOL_CALL_STREAM_KEY, None)
    if printer is not None:
        printer.finish()
    # Compact path: if the per-fragment gate never saw a completable tail
    # (e.g. exotic non-object arguments), make one final parse attempt now
    # that no more fragments can arrive for this tool call.
    last_tc: ToolCall | None = tmp_data.get(_LAST_TOOL_CALL_KEY)
    if last_tc is not None:
        already_printed = (
            tmp_data.get(_TOOL_HEADER_PRINTED_KEY)
            and tmp_data.get(_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY) == last_tc.id
        )
        if not already_printed:
            printed = _print_compact_tool_header(session, last_tc)
            # No more fragments can arrive for this tool call; don't re-attempt
            # the (failed) parse on every later non-toolcall message.
            tmp_data[_TOOL_HEADER_PRINTED_KEY] = True
            # Only remember the call id when the header actually printed;
            # otherwise a subsequent ToolCallPart for a different call
            # would be falsely gated.
            if printed:
                tmp_data[_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY] = last_tc.id
    # Clear any stale merge target (safety net for truncated streams).
    tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)


def _handle_tool_call(
    wire_msg: ToolCall | ToolCallPart,
    output_function: Callable[[str, MessageType], Any] | None,
    session: Session,
    format_output: bool = False,
    stream_tool_args: bool = False,
) -> None:
    if isinstance(wire_msg, ToolCall):
        # A new tool call supersedes any previous one: flush its pending
        # coalesced output first so callbacks stay in wire order.
        _flush_tool_call_part_output(session, output_function)
        session._tmp_data.pop(_TOOL_CALL_PART_EMITTED_LEN_KEY, None)
        # Clear previous header-printed flag for the new tool call.
        session._tmp_data.pop(_TOOL_HEADER_PRINTED_KEY, None)
        session._tmp_data[_LAST_TOOL_CALL_KEY] = wire_msg
        session._tmp_data[wire_msg.id] = wire_msg
        name = wire_msg.function.name
        args = wire_msg.function.arguments
        # Track merge target for ToolCallPart routing: when parallel tool
        # calls arrive, each streamed fragment must merge into the correct
        # pending call (not just the last one).  A call with empty or
        # still-incomplete arguments becomes the merge target; a call with
        # already-complete arguments clears it.
        if args is None or args == "" or not _json_tail_may_complete(args):
            if _TOOL_CALL_MERGE_TARGET_KEY not in session._tmp_data:
                session._tmp_data[_TOOL_CALL_MERGE_TARGET_KEY] = wire_msg
        else:
            session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)
        if stream_tool_args:
            if name not in _STREAM_TOOL_NAMES:
                # Non-streamable tool: fall through to the legacy compact
                # format path.
                # NOTE: empty initial arguments (``args == ""``) must NOT take
                # this path for streamable tools. Anthropic-protocol and
                # OpenAI-Responses providers emit the streamed call header as
                # ``ToolCall(arguments="")`` and deliver the arguments via
                # subsequent ``ToolCallPart`` fragments; treating the empty
                # header as "compact" left no stream printer behind, so the
                # terminal showed nothing until the whole arguments JSON had
                # finished streaming (the WritePlan stall).
                pass
            else:
                # A new tool call supersedes any previous stream printer.
                # Mark the header as printed *before* finishing any previous
                # stream printer, so _finish_tool_call_stream does not attempt
                # the compact path (which would print a second header).
                session._tmp_data[_TOOL_HEADER_PRINTED_KEY] = True
                session._tmp_data[_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY] = wire_msg.id
                _finish_tool_call_stream(session)
                _stream.colorful_print_word(
                    f"⚡ {name}", fg=_tool_header_color(name), require_new_line=True, flush=True)
                _stream._state = StreamPrintState.Other
                printer = _ToolCallStreamPrinter(name, session)
                session._tmp_data[_TOOL_CALL_STREAM_KEY] = printer
                if args:
                    printer.feed(args)
                if output_function:
                    output_function(
                        f"{name} {args or ''}", MessageType.ToolCalling)
                return
        formatted = _format_tool_args(name, args)
        if formatted:
            header = f"⚡ {name} {formatted}"
        else:
            return
        _stream.colorful_print_word(
            header, fg=_tool_header_color(name), require_new_line=True, flush=True)
        _stream._state = StreamPrintState.Other
        session._tmp_data[_TOOL_HEADER_PRINTED_KEY] = True
        session._tmp_data[_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY] = wire_msg.id
        if output_function:
            output_function(
                f"{name} {args or ''}", MessageType.ToolCalling)
    else:  # ToolCallPart
        # Route the fragment to the correct pending call.  When multiple
        # parallel tool calls are in flight, ``_LAST_TOOL_CALL_KEY`` may
        # point to a *later* call whose parts have already started arriving,
        # while this fragment belongs to an earlier still-pending call.  Use
        # the dedicated merge-target pointer when available.
        last_tc: ToolCall = (
            session._tmp_data.get(_TOOL_CALL_MERGE_TARGET_KEY)
            or session._tmp_data.get(_LAST_TOOL_CALL_KEY)
        )
        if last_tc is not None:
            last_tc.merge_in_place(wire_msg)
        # Clear the merge target when the merged arguments become
        # structurally complete (valid JSON).  This lets the next
        # ToolCallPart fall back to ``_LAST_TOOL_CALL_KEY`` (the next
        # pending call).
        if last_tc is not None and _TOOL_CALL_MERGE_TARGET_KEY in session._tmp_data:
            merged_args = last_tc.function.arguments
            if merged_args and _json_tail_may_complete(merged_args):
                try:
                    orjson.loads(merged_args)
                    session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)
                except (orjson.JSONDecodeError, TypeError, ValueError):
                    pass
        printer: _ToolCallStreamPrinter | None = None
        if stream_tool_args:
            printer = session._tmp_data.get(_TOOL_CALL_STREAM_KEY)
        if printer is not None:
            printer.feed(wire_msg.arguments_part or "")
        elif last_tc is not None:
            # Per-call gate: only skip printing if the header was already
            # printed for THIS specific tool call.
            _printed_id = session._tmp_data.get(_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY)
            if not session._tmp_data.get(_TOOL_HEADER_PRINTED_KEY) or _printed_id != last_tc.id:
                args = last_tc.function.arguments
                # Only attempt the full parse when the accumulated string
                # may actually be complete; otherwise orjson scans (and
                # fails on) the whole growing string per fragment — O(N^2).
                if args and _json_tail_may_complete(args):
                    _print_compact_tool_header(session, last_tc)
        if last_tc is not None:
            if output_function:
                # Coalesce cumulative snapshots: building and emitting the
                # full accumulated arguments for every fragment is O(N^2) in
                # the number of fragments. Consumers replace the previous
                # snapshot, so only emit at geometrically growing thresholds;
                # the final snapshot is flushed by
                # _flush_tool_call_part_output when the tool call ends.
                name = last_tc.function.name
                payload_len = len(name) + 1 + len(last_tc.function.arguments or '')
                emitted_len: int = session._tmp_data.get(
                    _TOOL_CALL_PART_EMITTED_LEN_KEY, 0)
                if payload_len >= max(
                        _TOOL_CALL_PART_MIN_EMIT_BYTES, emitted_len * 2):
                    output_function(
                        f"{name} {last_tc.function.arguments or ''}",
                        MessageType.ToolCallingPart)
                    session._tmp_data[_TOOL_CALL_PART_EMITTED_LEN_KEY] = payload_len
                    session._tmp_data.pop(_TOOL_CALL_PART_PENDING_KEY, None)
                else:
                    session._tmp_data[_TOOL_CALL_PART_PENDING_KEY] = True
        elif output_function:
            part = wire_msg.arguments_part or ""
            if part:
                output_function(part, MessageType.ToolCallingPart)
        if printer is None:
            _stream.print_word('', True)
        _stream._state = StreamPrintState.Other


def _handle_tool_result(wire_msg: ToolResult, output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    rv = wire_msg.return_value
    display_text = _format_display_blocks(rv.display)
    _stream.print_word(display_text, require_new_line=True)
    result_text = _format_tool_result(wire_msg)
    flushed = False
    if result_text:
        prefix = ("✗ " if rv.is_error else "✓ ")
        if display_text:
            tc: ToolCall | None = _session._tmp_data.pop(wire_msg.tool_call_id, None)
            # The tool call is finished: drop the stale "last tool call"
            # reference (only if it still points to this call, so a newer
            # in-flight call is not clobbered) together with the header flag.
            # Otherwise the next non-toolcall wire message would make
            # _finish_tool_call_stream re-print this call's header.
            #
            # Both pops are conditional on the id match: when the result
            # belongs to an *earlier* call while a later one is still in
            # flight, touching either entry corrupts the in-flight call's
            # state — clearing the header flag makes _finish_tool_call_stream
            # re-print the in-flight call's header once per arriving result.
            last_tc: ToolCall | None = _session._tmp_data.get(_LAST_TOOL_CALL_KEY)
            if last_tc is not None and last_tc.id == wire_msg.tool_call_id:
                _session._tmp_data.pop(_LAST_TOOL_CALL_KEY, None)
                _session._tmp_data.pop(_TOOL_HEADER_PRINTED_KEY, None)
                _session._tmp_data.pop(_TOOL_CALL_HEADER_PRINTED_TC_ID_KEY, None)
                _session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)
                if tc is None:
                    tc = last_tc
            # Safety: if the tool call was stored by id, it's done — clean up
            # any stale merge target that might still point to this call.
            if tc is not None:
                _session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)
            if tc:
                tool_name = tc.function.name if tc else "tool"
                _stream.colorful_print_word(
                    f"{prefix}{tool_name}",
                    fg=Color.BRIGHT_RED if rv.is_error else Color.BRIGHT_GREEN,
                    require_new_line=True,
                    flush=True,
                )
                flushed = True
        else:
            _stream.colorful_print_word(
                f"{prefix}{result_text}",
                fg=Color.BRIGHT_RED if rv.is_error else Color.BRIGHT_GREEN,
                require_new_line=True,
                flush=True,
            )
            flushed = True
    if not flushed:
        _stream.print_word('', True, flush=True)
    _stream._state = StreamPrintState.Other
    if output_function:
        formatted = f"[ToolResult] {_format_tool_result(wire_msg)}"
        if formatted:
            output_function(formatted, MessageType.ToolResult)


def _handle_approval_request(wire_msg: ApprovalRequest, _output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    wire_msg.resolve("approve")


def _handle_noop(_wire_msg: Any, _output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    pass


def _handle_compaction_begin(_wire_msg: Any, _output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    _stream.colorful_print_word(
        "Compacting...", require_new_line=True, fg=Color.BRIGHT_MAGENTA, flush=True)


def _handle_think_part(wire_msg: ThinkPart, output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    think_content = wire_msg.think
    if not _quiet:
        if output_function:
            output_function(think_content, MessageType.Thinking)
        if _stream._state != StreamPrintState.Thinking:
            _stream.colorful_print_word(
                f"[Think] {think_content}", fg=Color.BRIGHT_CYAN, require_new_line=True)
        else:
            _stream.colorful_print_word(
                f"{think_content}", fg=Color.BRIGHT_CYAN, require_new_line=False)
        _stream._state = StreamPrintState.Thinking


def _handle_text_part(wire_msg: TextPart, output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    chunk = wire_msg.text
    if output_function:
        output_function(chunk, MessageType.Text)
    if format_output:
        global _text_buffer
        if _text_buffer is None:
            _text_buffer = io.StringIO()
        _text_buffer.write(chunk)
    else:
        _stream.print_word(
            chunk, require_new_line=_stream._state != StreamPrintState.Text)
    _stream._state = StreamPrintState.Text



def _handle_other(_wire_msg: Any, _output_function: Callable[[str, MessageType], Any] | None, _session: Session, format_output: bool = False) -> None:
    _stream._state = StreamPrintState.Other


_PRINT_AGENT_JSON_DISPATCH: dict[type, Callable[[Any, Callable[[str, MessageType], Any] | None, Session, bool], None]] = {
    ToolCall: _handle_tool_call,
    ToolCallPart: _handle_tool_call,
    ToolResult: _handle_tool_result,
    ApprovalRequest: _handle_approval_request,
    StepBegin: _handle_noop,
    StepInterrupted: _handle_noop,
    CompactionEnd: _handle_noop,
    CompactionBegin: _handle_compaction_begin,
    ThinkPart: _handle_think_part,
    TextPart: _handle_text_part,
}


def _flush_agent_json_text() -> None:
    """Flush any buffered text parts as formatted markdown."""
    global _text_buffer
    if _text_buffer is not None:
        text = _text_buffer.getvalue()
        _text_buffer.close()
        _text_buffer = None
        if text:
            from kimix.cli_impl.utils import render_markdown
            _stream.print_word(render_markdown(text), require_new_line=True, flush=True)


def print_agent_json_flush_text() -> None:
    """Public helper to flush buffered text parts as formatted markdown."""
    _flush_agent_json_text()


async def print_agent_json(
    wire_msg: Any,
    session: Session,
    output_function: Callable[[str, MessageType], Any] | None = None,
    format_output: bool = False,
    stream_tool_args: bool = True,
) -> None:
    """Pretty-print a streaming wire message from an agent session.

    Awaitable; the internal handlers are synchronous (printing is sync I/O).

    When ``stream_tool_args`` is True (default), each tool argument starts on
    its own line beneath the tool header; long whitelisted string values
    (e.g. the ``content`` parameter of ``WriteFile``) are streamed decoded
    token by token as ``ToolCallPart`` fragments arrive from the LLM, and
    short scalar arguments print as compact ``key: value`` lines. Pass
    ``stream_tool_args=False`` to restore the legacy compact output (which
    hides long values such as ``content``) byte-for-byte.

    With ``merge_wire_messages=True`` a single complete ``ToolCall`` arrives,
    so the full decoded value is printed in one go; pass
    ``stream_tool_args=False`` for the old compact, hidden-content output.
    """
    if format_output and _stream._state == StreamPrintState.Text and not isinstance(wire_msg, TextPart):
        _flush_agent_json_text()
        _stream._state = StreamPrintState.Other
    if not isinstance(wire_msg, (ToolCall, ToolCallPart)):
        # Terminate any streamed tool-call argument line before other output
        # (tool results, text parts, usage banners, ...).
        _finish_tool_call_stream(session)
        # Deliver the final coalesced ToolCallingPart snapshot (if any)
        # before this message's own output.
        _flush_tool_call_part_output(session, output_function)
    _print_transition_usage(session, _message_transition_type(wire_msg))
    if isinstance(wire_msg, (ToolCall, ToolCallPart)):
        _handle_tool_call(wire_msg, output_function, session, format_output,
                          stream_tool_args=stream_tool_args)
        return
    handler = _PRINT_AGENT_JSON_DISPATCH.get(type(wire_msg))
    if handler is not None:
        handler(wire_msg, output_function, session, format_output)
    else:
        _handle_other(wire_msg, output_function, session, format_output)


def run_thread(
    function: Callable[..., Any], args: tuple[Any, ...] | None = None
) -> threading.Thread:
    assert callable(function)
    global _threads
    _process_lru()

    if args is None:
        args = ()
    elif type(args) is not tuple:
        args = (args, )
    thd = threading.Thread(target=function, args=args)
    thd.start()
    _threads.append(thd)
    return thd


def run_script(path: str | Path) -> Any:
    return subprocess.Popen(
        [sys.executable, str(path)], creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def sync_all() -> None:
    global _threads
    for thd in _threads:
        thd.join()
    _threads.clear()


def _run_process_with_log(command: str) -> tuple[str, int]:
    print_info(f"Shell: {command}")
    result = subprocess.run(command, shell=True, capture_output=True)
    output = result.stdout.decode(
        "utf-8", errors="replace") if result.stdout else ""
    if result.stderr:
        output += "\n" + result.stderr.decode("utf-8", errors="replace")
    return output, result.returncode


async def _run_process_with_log_async(command: str) -> tuple[str, int]:
    print_info(f"Shell: {command}")
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    if stderr:
        output += "\n" + stderr.decode("utf-8", errors="replace")
    return output, proc.returncode


def _filter_error_output(
    result: str, code: int, keycode: tuple[str, ...] | None, skip_success: bool
) -> str | None:
    if skip_success and code == 0:
        return None
    if not keycode:
        return result
    lines = result.splitlines()
    for idx, line in enumerate(lines):
        lower_line = line.lower()
        for c in keycode:
            if c in lower_line:
                return "\n".join(lines[idx:])
    return result


def run_process_with_error(
    command: str,
    keycode: tuple[str, ...] | None,
    skip_success: bool = True,
) -> str | None:
    result, code = _run_process_with_log(command)
    return _filter_error_output(result, code, keycode, skip_success)


async def run_process_with_error_async(
    command: str,
    keycode: tuple[str, ...] | None,
    skip_success: bool = True,
) -> str | None:
    result, code = await _run_process_with_log_async(command)
    return _filter_error_output(result, code, keycode, skip_success)


def percentage_str(num: float) -> str:
    return f"{num * 100:.1f}%"


def percentage_and_token(session: Any) -> str:
    status = session.status
    return f"{status.context_usage * 100:.1f}% ({status.context_tokens} tokens)"


_default_thinking: bool = True
_default_yolo: bool = True
_default_agent_file_dir: Path = Path(__file__).parent
_default_agent_file: Path = _default_agent_file_dir / "agent_worker.json"
_default_skill_dirs: list[Any] = []
_default_provider: dict[str, Any] | None = None
_default_sub_providers: list[dict[str, Any]] = []
_default_manually_cot: bool = False
_default_ralph: int | None = None

# Common skill directory paths (relative to current working directory)
COMMON_SKILL_DIRS: list[str] = [
    ".agents/skills",
    ".config/.agents/skills",
    ".opencode/skills",
    ".claude/skills",
    ".codex/skills",
    ".skills",
    "skills",
]


def set_default_thinking(value: bool) -> None:
    global _default_thinking
    _default_thinking = value


def set_default_yolo(value: bool) -> None:
    global _default_yolo
    _default_yolo = value


def set_default_agent_file_dir(value: Path) -> None:
    global _default_agent_file_dir
    _default_agent_file_dir = value


def set_default_agent_file(value: Path) -> None:
    global _default_agent_file
    _default_agent_file = value


def set_default_skill_dirs(value: list[Any]) -> None:
    global _default_skill_dirs
    _default_skill_dirs = value


def set_default_manually_cot(value: bool) -> None:
    global _default_manually_cot
    _default_manually_cot = value


def set_default_provider(value: dict[str, Any] | None) -> None:
    global _default_provider
    _default_provider = value


def set_default_sub_providers(providers: list[dict[str, Any]] | None) -> None:
    global _default_sub_providers
    _default_sub_providers = list(providers or [])


def get_default_sub_provider(role: str = "sub_agent") -> dict[str, Any] | None:
    for p in _default_sub_providers:
        if p.get("role", "sub_agent") == role:
            return p
    return None


# The failed-list for tool call that
# tuple: function-name, arguments, output, message


def get_skill_dirs(use_kaos_path: bool = True) -> list[Any]:
    from kaos.path import KaosPath

    global _default_skill_dirs
    if _default_skill_dirs:
        if use_kaos_path:
            return [KaosPath(str(i)) for i in _default_skill_dirs]
        return _default_skill_dirs

    _default_skill_dirs = [
        p for rel in COMMON_SKILL_DIRS if (p := Path(os.curdir) / rel).exists()
    ]
    # If there's a `skills` subdirectory under the skill dir, use `*/skills` pattern
    _default_skill_dirs = [
        p / "*/skills" if (p / "skills").is_dir() else p
        for p in _default_skill_dirs
    ]
    if _default_skill_dirs:
        for d in _default_skill_dirs:
            print_debug(f"skill dir: {str(d)}")
        if use_kaos_path:
            return [KaosPath(str(d)) for d in _default_skill_dirs]
        return _default_skill_dirs
    return []


generate_memory = """---

Compact the above agent conversation context according to the following priorities and rules.

**Priorities:**
- **Current Task State** — what is being worked on right now
- **Errors & Solutions** — all errors encountered and how they were resolved
- **Code Evolution** — final working versions only (drop intermediate attempts)
- **System Context** — project structure, dependencies, environment setup
- **Design Decisions** — architectural choices and rationale
- **TODO Items** — unfinished tasks and known issues
- **Project Overview** — purpose, scope, tech stack
- **Key Decisions** — critical choices, rationale, rejected alternatives
- **Current State** — what works, what's merged/verified, active branch, test results
- **Important Files** — key paths and their roles (add, modify, delete)
- **Architecture / Data Flow** — major components, interfaces, schema changes
- **Dependencies** — added, removed, upgraded packages or services
- **Risks / Rollback** — breaking changes, migration steps, revert strategy
- **Technical Notes** — patterns, constraints, APIs, env setup, performance or security considerations

**Rules:**
- **Keep:** error messages, stack traces, working solutions, current task
- **Merge:** similar discussions into single summary points
- **Remove:** redundant explanations, failed attempts (retain lessons learned), verbose comments
- **Condense:** long code blocks → signatures + key logic only

**Special Handling:**
- **Code:** keep full version if < 20 lines; otherwise keep signature + key logic
- **Errors:** keep full error message + final solution
- **Discussions:** extract decisions and action items only
```"""
