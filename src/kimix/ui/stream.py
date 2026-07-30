"""Wire-message streaming and rendering for kimix (P8: extracted from kimix.base).

Tool-call stream printing, display-block formatting and
``print_agent_json``. Import from here instead of ``kimix.base`` for new
code.
"""

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
from kosong.tooling import (
    TOOL_NAME_REDIRECTS,
    normalize_tool_name,
    resolve_tool_name,
)
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

from kimix.ui.printing import (
    Color,
    Color256,
    GRAY,
    GRAY_LIGHT,
    MessageType,
    PrintStream,
    StreamPrintState,
    _quiet,
    _stream,
    colorful_text,
    print,
)

if TYPE_CHECKING:
    from kimi_agent_sdk import Session

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
            # Command output is shown via the success/failure message
            pass
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
_TOOL_CALL_PART_PENDING_KEY = "_kimix_tool_call_part_pending"
_TOOL_CALL_PART_EMITTED_LEN_KEY = "_kimix_tool_call_part_emitted_len"
_TOOL_CALL_MERGE_TARGET_KEY = "_kimix_tool_call_merge_target"

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

# Mapping from argument-key aliases to the canonical key used for
# streaming decisions, color lookup, and display.
#
# This must stay in parity with the execution-side key repair in
# ``kosong.tooling`` (the pydantic ``Field(alias=...)`` declarations plus
# the ``FIELD_ALIASES_*`` tables applied by ``_repair_dict_for_model``):
# the toolset repairs hallucinated keys at execution time, but the wire
# messages shown here still carry the *raw* keys the model sent.  Keys
# that kosong repairs but this map misses would silently lose the live
# decoded streaming display (they fall back to a truncated 60-char
# compact line).  Only alias targets relevant to the streaming display
# are listed — kosong's tables contain conflicting entries for other
# tools (e.g. ``text -> title``, ``path -> directory``) that must NOT be
# applied here.
_ARG_KEY_ALIASES: dict[str, str] = {
    # pydantic Field(alias=...) declarations.
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
    # kosong FIELD_ALIASES_FILE parity (common LLM key substitutions,
    # frequent with kimi/anthropic providers — e.g. Claude's native editor
    # uses ``old_str`` / ``new_str``).
    "old_str": "old",
    "new_str": "new",
    "old_content": "old",
    "new_content": "new",
    "original": "old",
    "replace_with": "new",
    "data": "content",
    "body": "content",
    "file": "path",
    "filepath": "path",
    "filename": "path",
    "file_name": "path",
    "changes": "edit",
    "modifications": "edit",
    # Grep CLI-style flag aliases (parity with Grep.field_aliases).
    "-A": "after_context",
    "-B": "before_context",
    "-C": "context",
    "-n": "line_number",
    "-i": "ignore_case",
}


def _canonical_key(key: str) -> str:
    """Return the canonical form of *key*, resolving known LLM aliases.

    Case-insensitive fallback mirrors the case-insensitive fuzzy key
    matching applied by kosong at execution time."""
    canonical = _ARG_KEY_ALIASES.get(key)
    if canonical is not None:
        return canonical
    return _ARG_KEY_ALIASES.get(key.lower(), key)


# Pre-normalized redirect map for tool-name resolution, mirroring
# ``kosong.tooling._TOOL_NAME_REDIRECTS_NORMALIZED`` (built here from the
# public table to avoid relying on a private name).
_TOOL_NAME_REDIRECTS_NORM: dict[str, str] = {
    normalize_tool_name(k): v for k, v in TOOL_NAME_REDIRECTS.items() if k != v
}


def _session_tool_names(session: Session) -> tuple[str, ...]:
    """Names of the tools registered in the session's live toolset.

    Returns an empty tuple when the toolset is not reachable (tests,
    custom session implementations); name resolution then falls back to
    the redirect table only.
    """
    try:
        toolset = session._cli.soul.agent.toolset  # type: ignore[attr-defined]
        return tuple(tool.name for tool in toolset.tools)
    except AttributeError:
        return ()


def _resolve_display_tool_name(name: str, session: Session) -> str:
    """Resolve a (possibly hallucinated) tool name for the header display.

    Every tool is streamable — there is no whitelist.  This only picks the
    *canonical name* shown in the ``⚡ Name`` header, mirroring the
    execution-side resolution in ``kosong.tooling`` (``resolve_tool_name``
    + ``TOOL_NAME_REDIRECTS``, applied by ``KimiToolset.handle``): a model
    that sends ``write_file`` or ``AppendFile`` has its call auto-corrected
    to ``WriteFile`` at execution time, but the wire message shown here
    still carries the raw name.

    Future-compatible: the candidate names come from the session's live
    toolset, so newly registered tools resolve with no code change here.
    Falls back to the raw wire name when nothing matches.
    """
    candidates = _session_tool_names(session)
    if not candidates:
        return _TOOL_NAME_REDIRECTS_NORM.get(normalize_tool_name(name), name)
    resolution = resolve_tool_name(
        name, candidates, redirects=_TOOL_NAME_REDIRECTS_NORM
    )
    return resolution.name if resolution.name is not None else name

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

# Streamed argument keys (a subset of :data:`_STREAM_ARG_KEYS`) whose values
# are printed *inline* after the tool header (space separator, no "key:\n"
# label).  Used so that ``⚡ Powershell Get-Date`` stays on one line instead
# of breaking the command onto a new ``command:\n`` line.
#
# Keys are canonical (resolved via _canonical_key()), so the "cmd" alias
# is automatically covered — no need to list it separately.
#
# All other (short) argument values are printed inline as `` key:value``
# on the header line — see ``_ToolCallStreamPrinter._emit_compact``.
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


# Maximum length of a single argument value in the compact one-line
# summary produced by :func:`format_tool_args` (and in the inline
# `` key:value`` segments printed by the stream printer).
_COMPACT_VALUE_MAX_LEN = 60


def format_tool_args(args: str | None) -> str | None:
    """Format raw tool-call arguments JSON as a compact one-line summary.

    Fully generic — every argument renders as ``key:value`` separated by
    spaces, so new tools need no per-tool display code.  Keys are
    canonicalized via :func:`_canonical_key` (alias spellings such as
    ``cmd`` or ``file_path`` display under their canonical name); values
    longer than :data:`_COMPACT_VALUE_MAX_LEN` characters are truncated.

    Returns ``None`` when *args* is ``None`` or not valid JSON, ``""`` for
    empty arguments.
    """
    if args is None:
        return None
    if args == "":
        return ""
    try:
        parsed = orjson.loads(args)
    except (orjson.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return orjson.dumps(parsed).decode("utf-8")
    parts: list[str] = []
    for key, value in parsed.items():
        text = str(value)
        if len(text) > _COMPACT_VALUE_MAX_LEN:
            text = text[:_COMPACT_VALUE_MAX_LEN] + "..."
        parts.append(f"{_canonical_key(key)}:{text}")
    return " ".join(parts)


class _ToolCallStreamPrinter:
    """Incrementally lexes streamed tool-call arguments JSON and prints
    decoded argument values live (token by token) as fragments arrive.

    Lifecycle: created when a ``ToolCall`` arrives; :meth:`feed` is called per
    ``ToolCallPart`` fragment; :meth:`finish` is called when the arguments JSON
    parses completely, when a new ``ToolCall`` supersedes this one, or when any
    non-``ToolCallPart`` wire message arrives (safety net for truncated or
    malformed JSON).

    Each argument is printed beneath the ``⚡ Name`` header. String values
    for keys in :data:`_STREAM_ARG_KEYS` are printed decoded, fragment by
    fragment, each in a per-key color from :attr:`_STREAM_KEY_COLORS`
    (fallback ``GRAY_LIGHT``); keys in :data:`_INLINE_ARG_KEYS` follow the
    header inline (space separator, no label) while the rest get a
    ``key:\n`` label on their own line. Other short scalar values are
    buffered and printed on completion — inline as `` key: value`` for
    keys in :data:`_INLINE_ARG_KEYS`, otherwise as compact ``key:\nvalue``
    lines.
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
        """Print a short (non-streamed) argument value.

        Short arguments never get their own line: they stay inline on the
        header line as `` key:value`` segments, e.g.::

            ⚡ EditFile path:C:/dev/x.py line_offset:1125 max_char:15000

        Only values whose keys are in :data:`_STREAM_ARG_KEYS` are printed
        on a new line (decoded, token by token) — see
        :meth:`_begin_string_value`.
        """
        if len(text) > _COMPACT_VALUE_MAX_LEN:
            text = text[:_COMPACT_VALUE_MAX_LEN] + "..."
        canonical_key = _canonical_key(self._current_key) if self._current_key else ""
        segment = f" {canonical_key}:{text}" if canonical_key \
            else f" {text}"
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


def _finish_tool_call_stream(session: Session) -> None:
    """Finish and remove any active tool-call stream printer for the session."""
    tmp_data = getattr(session, "_tmp_data", None)
    if not tmp_data:
        return
    printer = tmp_data.pop(_TOOL_CALL_STREAM_KEY, None)
    if printer is not None:
        printer.finish()
    # Clear any stale merge target (safety net for truncated streams).
    tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)


def _handle_tool_call(
    wire_msg: ToolCall | ToolCallPart,
    output_function: Callable[[str, MessageType], Any] | None,
    session: Session,
    format_output: bool = False,
) -> None:
    if isinstance(wire_msg, ToolCall):
        # A new tool call supersedes any previous one: flush its pending
        # coalesced output first so callbacks stay in wire order.
        _flush_tool_call_part_output(session, output_function)
        session._tmp_data.pop(_TOOL_CALL_PART_EMITTED_LEN_KEY, None)
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
        # Every tool call streams — there is no whitelist and no legacy
        # compact path.  A new tool call supersedes any previous stream
        # printer.
        _finish_tool_call_stream(session)
        # The header shows the resolved canonical name — the name the
        # toolset will actually execute after its own auto-correction (the
        # raw wire name may be a hallucination such as ``write_file`` or
        # ``AppendFile``).
        resolved_name = _resolve_display_tool_name(name, session)
        _stream.colorful_print_word(
            f"⚡ {resolved_name}", fg=_tool_header_color(resolved_name), require_new_line=True, flush=True)
        _stream._state = StreamPrintState.Other
        # NOTE: empty initial arguments (``args == ""`` or ``None``) must NOT
        # skip the stream printer. Anthropic-protocol and OpenAI-Responses
        # providers emit the streamed call header as ``ToolCall(arguments="")``
        # and deliver the arguments via subsequent ``ToolCallPart`` fragments.
        printer = _ToolCallStreamPrinter(resolved_name, session)
        session._tmp_data[_TOOL_CALL_STREAM_KEY] = printer
        if args:
            printer.feed(args)
        if output_function:
            output_function(
                f"{name} {args or ''}", MessageType.ToolCalling)
        return
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
        printer: _ToolCallStreamPrinter | None = session._tmp_data.get(_TOOL_CALL_STREAM_KEY)
        if printer is not None:
            printer.feed(wire_msg.arguments_part or "")
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

    prefix = ("✗ " if rv.is_error else "✓ ")
    tc: ToolCall | None = _session._tmp_data.pop(wire_msg.tool_call_id, None)

    # The tool call is finished: drop the stale "last tool call"
    # reference (only if it still points to this call, so a newer
    # in-flight call is not clobbered) together with the merge target.
    #
    # The pops are conditional on the id match: when the result
    # belongs to an *earlier* call while a later one is still in
    # flight, touching either entry corrupts the in-flight call's
    # state.
    last_tc: ToolCall | None = _session._tmp_data.get(_LAST_TOOL_CALL_KEY)
    if last_tc is not None and last_tc.id == wire_msg.tool_call_id:
        _session._tmp_data.pop(_LAST_TOOL_CALL_KEY, None)
        _session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)
        if tc is None:
            tc = last_tc
    # Safety: if the tool call was stored by id, it's done — clean up
    # any stale merge target that might still point to this call.
    if tc is not None:
        _session._tmp_data.pop(_TOOL_CALL_MERGE_TARGET_KEY, None)

    if tc:
        # Primary: show "✓ ToolName" (or "✗ ToolName")
        _stream.colorful_print_word(
            f"{prefix}{tc.function.name}",
            fg=Color.BRIGHT_RED if rv.is_error else Color.BRIGHT_GREEN,
            require_new_line=True,
            flush=True,
        )
        # Supplementary: show message detail in dim text if non-trivial
        if result_text and result_text not in ("success", "failed", "[rtk] success", "[rtk] failed"):
            _stream.colorful_print_word(
                f"  {result_text}",
                fg=Color.BRIGHT_BLACK,
                require_new_line=True,
                flush=True,
            )
    elif result_text:
        # Fallback: no tool call available, show message directly
        _stream.colorful_print_word(
            f"{prefix}{result_text}",
            fg=Color.BRIGHT_RED if rv.is_error else Color.BRIGHT_GREEN,
            require_new_line=True,
            flush=True,
        )
    else:
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
) -> None:
    """Pretty-print a streaming wire message from an agent session.

    Awaitable; the internal handlers are synchronous (printing is sync I/O).

    Every tool call is printed via the incremental stream printer: the
    ``⚡ Name`` header appears as soon as the ``ToolCall`` arrives, long
    string values whose keys are in ``_STREAM_ARG_KEYS`` (e.g. the
    ``content`` parameter of ``WriteFile``) are streamed decoded token by
    token as ``ToolCallPart`` fragments arrive from the LLM, each on its
    own ``key:\n`` line, while short arguments print inline on the header
    line as `` key:value`` segments.  The design is fully generic — new
    tools need no per-tool display code.

    With ``merge_wire_messages=True`` a single complete ``ToolCall`` arrives,
    so the full decoded value is printed in one go by the same stream
    printer.
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
        _handle_tool_call(wire_msg, output_function, session, format_output)
        return
    handler = _PRINT_AGENT_JSON_DISPATCH.get(type(wire_msg))
    if handler is not None:
        handler(wire_msg, output_function, session, format_output)
    else:
        _handle_other(wire_msg, output_function, session, format_output)




def percentage_str(num: float) -> str:
    return f"{num * 100:.1f}%"


def percentage_and_token(session: Any) -> str:
    status = session.status
    return f"{status.context_usage * 100:.1f}% ({status.context_tokens} tokens)"
