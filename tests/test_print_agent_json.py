from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import orjson
from kimi_cli.wire.types import (
    ShellDisplayBlock,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
)
from kosong.tooling import ToolReturnValue

import kimix.base as base

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float
    context_tokens: int


class FakeSession:
    def __init__(self, context_usage: float = 0.125, context_tokens: int = 1024) -> None:
        self.status = FakeStatus(context_usage=context_usage, context_tokens=context_tokens)
        self.cancelled = False
        self._cancel_event = None
        self._tmp_data = {}

    async def prompt(self, _prompt: str, *, merge_wire_messages: bool = False) -> Any:
        del merge_wire_messages
        yield TextPart(text="prompt output")

    def cancel(self) -> None:
        self.cancelled = True


def _capture_base_stream(monkeypatch: Any) -> list[str]:
    chunks: list[str] = []

    def print_func(*values: object, sep: str = " ", end: str = "\n", **_kwargs: Any) -> None:
        chunks.append(sep.join(str(value) for value in values) + end)

    monkeypatch.setattr(base, "_stream", base.PrintStream(print_func=print_func))
    monkeypatch.setattr(base, "_text_buffer", None)
    monkeypatch.setattr(base, "_quiet", False)
    monkeypatch.setattr(base, "_colorful_print", True)
    return chunks


def _plain(chunks: list[str]) -> str:
    """Join captured chunks and strip ANSI codes (streamed segments are
    printed as separate colored writes, so substrings are only contiguous
    after stripping)."""
    return base._strip_ansi("".join(chunks))


async def test_print_agent_json_prints_black_usage_when_text_switches_to_thinking(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession(context_usage=0.125, context_tokens=1024)

    await base.print_agent_json(TextPart(text="hello"), session)
    await base.print_agent_json(TextPart(text=" world"), session)
    await base.print_agent_json(ThinkPart(think="hmm"), session)

    output = "".join(chunks)

    assert output.count("Context usage: 12.5% (1024 tokens)") == 1
    assert "\x1b[38;5;245m==================== Context usage: 12.5% (1024 tokens) ========================\n\x1b[0m" in output
    assert "hello world" in output
    assert "\x1b[96m[Think] hmm\x1b[0m" in output


async def test_print_agent_json_groups_tool_parts_before_tool_to_text_transition(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession(context_usage=0.5, context_tokens=4096)
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Run", arguments='{"command": "pytest"}'),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"more": true}'), session)
    await base.print_agent_json(TextPart(text="done"), session)

    output = "".join(chunks)

    assert output.count("Context usage: 50.0% (4096 tokens)") == 1
    assert "\x1b[38;5;245m==================== Context usage: 50.0% (4096 tokens) ========================\n\x1b[0m" in output
    # Complete-args ToolCall: header + streamed inline command printed via the
    # stream printer, which finishes immediately; the stray ToolCallPart stays silent.
    assert output.count("⚡ Run") == 1
    plain = _plain(chunks)
    # Command is inline after the header (no "command:" label): ``⚡ Run pytest``
    assert "pytest" in plain
    assert "more" not in plain
    assert "done" in plain


def test_prompt_async_passes_session_to_print_agent_json(monkeypatch: Any) -> None:
    import asyncio

    calls: list[tuple[object, object, object]] = []
    session = FakeSession()

    async def fake_print_agent_json(wire_msg: object, passed_session: object, output_function: object, format_output: bool = False) -> None:
        calls.append((wire_msg, passed_session, output_function, format_output))

    monkeypatch.setattr(prompt_mod, "print_agent_json", fake_print_agent_json)
    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *args, **kwargs: None)

    asyncio.run(prompt_mod.prompt_async("hello", session=session))

    assert len(calls) == 1
    assert isinstance(calls[0][0], TextPart)
    assert calls[0][1] is session
    assert calls[0][2] is None
    assert calls[0][3] is False


async def test_print_agent_json_format_output_buffers_text_until_mode_change(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(TextPart(text="hello "), session, format_output=True)
    await base.print_agent_json(TextPart(text="world"), session, format_output=True)
    assert "hello world" not in "".join(chunks)

    await base.print_agent_json(ThinkPart(think="hmm"), session, format_output=True)
    output = "".join(chunks)
    assert "hello world" in output
    assert "[Think] hmm" in output


async def test_print_agent_json_format_output_flushes_remaining_text_at_end(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(TextPart(text="hello"), session, format_output=True)
    assert "hello" not in "".join(chunks)

    base.print_agent_json_flush_text()
    assert "hello" in "".join(chunks)


async def test_print_agent_json_format_output_still_calls_output_function(monkeypatch: Any) -> None:
    monkeypatch.setattr(base, "_text_buffer", None)
    session = FakeSession()
    received: list[str] = []

    def output_function(text: str, _msg_type: object) -> None:
        received.append(text)

    await base.print_agent_json(TextPart(text="chunk1"), session, output_function=output_function, format_output=True)
    await base.print_agent_json(TextPart(text="chunk2"), session, output_function=output_function, format_output=True)

    assert received == ["chunk1", "chunk2"]


async def test_print_agent_json_streams_writefile_content_token_by_token(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"path": "x.py", "content": "hello'), session)
    await base.print_agent_json(ToolCallPart(arguments_part=" wor"), session)
    await base.print_agent_json(ToolCallPart(arguments_part='ld"}'), session)

    output = "".join(chunks)
    plain = _plain(chunks)

    # Header printed exactly once when the ToolCall arrives.
    assert output.count("⚡ WriteFile") == 1
    # Non-whitelisted short values print as compact segments.
    assert "path:\nx.py" in plain
    # The long content value is printed decoded, across fragments.
    assert "hello world" in plain
    # No stray JSON quotes/braces leak into the streamed content.
    assert 'ld"}' not in plain
    # Complete JSON finishes the stream: the line is terminated.
    assert base._stream._last_char_was_newline is True


async def test_print_agent_json_stream_decodes_escapes(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"content": "line1\\nline2 \\"q\\""}'),
        session,
    )

    plain = _plain(chunks)

    assert "line1\nline2" in plain
    assert '"q"' in plain
    # Raw escape sequences are not printed verbatim.
    assert "\\nline2" not in plain


async def test_print_agent_json_stream_handles_split_unicode_escape(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"content": "\\u4f'), session)
    assert "你" not in "".join(chunks)

    await base.print_agent_json(ToolCallPart(arguments_part='60"}'), session)

    plain = _plain(chunks)
    assert plain.count("你") == 1
    assert "\\u4f" not in plain


async def test_print_agent_json_stream_prints_compact_short_values(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "mode": "overwrite", "content": "body"}'),
        session,
    )

    plain = _plain(chunks)

    assert "path:\nx.py" in plain
    # Short scalar args (``mode``) print inline on the header line.
    assert " mode: overwrite" in plain
    assert "mode:\noverwrite" not in plain
    assert "body" in plain


async def test_streamed_short_args_print_inline(monkeypatch: Any) -> None:
    """Short scalar arguments (e.g. ``timeout``) print inline after the tool
    header — ``⚡ Powershell Get-Date timeout: 30`` — instead of each
    occupying its own ``timeout:\n30`` line beneath the header."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-inline-timeout",
            function=ToolCall.FunctionBody(name="Powershell", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"command": "Get-Date", "timeout": 30}'),
        session,
    )

    plain = _plain(chunks)
    assert "Get-Date timeout: 30" in plain
    assert "timeout:\n30" not in plain


async def test_print_agent_json_stream_finished_by_tool_result(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"content": "partial'), session)
    assert base._stream._last_char_was_newline is False

    tool_result = ToolResult(
        tool_call_id="call-1",
        return_value=ToolReturnValue(is_error=False, message="ok", output="", display=[]),
    )
    await base.print_agent_json(tool_result, session)

    plain = _plain(chunks)
    assert "partial" in plain
    # The truncated stream line is terminated before the tool result prints.
    assert "partial\n" in plain
    assert "\n✓ ok" in plain
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_print_agent_json_stream_opt_out_restores_compact_output(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="WriteFile",
            arguments='{"path": "x.py", "content": "secret body"}'),
    )

    await base.print_agent_json(tool_call, session, stream_tool_args=False)

    plain = _plain(chunks)
    assert "⚡ WriteFile path: x.py, content: ..." in plain
    assert "secret body" not in plain


async def test_print_agent_json_merged_tool_call_prints_full_content_once(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    args = orjson.dumps({"path": "big.py", "content": "full body here"}).decode("utf-8")
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=args),
    )

    await base.print_agent_json(tool_call, session)

    plain = _plain(chunks)
    assert plain.count("full body here") == 1
    assert "path:\nbig.py" in plain
    assert base._stream._last_char_was_newline is True




async def test_non_whitelisted_tool_uses_compact_format(monkeypatch: Any) -> None:
    """Non-whitelisted tools (Grep, etc.) should use the
    legacy compact ``_format_tool_args`` output even with ``stream_tool_args=True``."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    # A complete ToolCall for Grep (not in _STREAM_TOOL_NAMES).
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="Grep",
            arguments='{"pattern": "def ", "path": ".", "-n": true}',
        ),
    )

    await base.print_agent_json(tool_call, session)  # stream_tool_args=True by default

    output = "".join(chunks)
    plain = _plain(chunks)

    # Header printed exactly once via _format_tool_args compact format.
    assert output.count("⚡ Grep") == 1
    assert "pattern: def " in plain
    assert "path: ." in plain
    # No streaming artifacts.
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_non_whitelisted_tool_streaming_fragments_still_compact(monkeypatch: Any) -> None:
    """Non-whitelisted tool with ToolCall(arguments=None) + fragments should
    still use the legacy compact format, not the stream printer."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    tool_call = ToolCall(
        id="call-2",
        function=ToolCall.FunctionBody(name="Grep", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    # No stream printer was created.
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data

    # Send fragments that build up complete JSON.
    await base.print_agent_json(ToolCallPart(arguments_part='{"pattern": "def ", "path": "'), session)
    await base.print_agent_json(ToolCallPart(arguments_part='.", "-n": true}'), session)

    plain = _plain(chunks)

    # The legacy format path prints the compact summary.
    assert "⚡ Grep" in plain
    assert "pattern: def " in plain
    assert "path: ." in plain
    # Only one header.
    assert plain.count("⚡ Grep") == 1
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_whitelisted_tool_empty_initial_args_streams_live(monkeypatch: Any) -> None:
    """Regression (WritePlan stall): Anthropic-protocol and OpenAI-Responses
    providers emit the streamed call header as ``ToolCall(arguments="")`` and
    deliver the arguments via subsequent ``ToolCallPart`` fragments.

    The ``args == ""`` guard used to force whitelisted tools down the legacy
    compact path: no stream printer was created, so the terminal showed
    nothing until the whole arguments JSON had finished streaming, and then
    only the compact hidden-content one-liner appeared. The header must print
    at header time and the decoded content must stream live.
    """
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="WritePlan", arguments="")),
        session,
    )
    # Header printed immediately at ToolCall time; stream printer created.
    assert "⚡ WritePlan" in _plain(chunks)
    assert base._TOOL_CALL_STREAM_KEY in session._tmp_data

    # Fragments are JSON-escaped, exactly as a provider's input_json_delta sends them.
    fragments = ['{"content": "# Plan', '\\n\\n1. first', '\\n2. second', '"}']
    per_fragment_output: list[str] = []
    for frag in fragments:
        before = len("".join(chunks))
        await base.print_agent_json(ToolCallPart(arguments_part=frag), session)
        per_fragment_output.append("".join(chunks)[before:])

    plain = _plain(chunks)
    # Decoded content streamed live and is fully visible.
    assert "# Plan" in plain
    assert "1. first" in plain
    assert "2. second" in plain
    # Mid-stream fragments produced visible output before the JSON completed.
    assert any(base._strip_ansi(text).strip() for text in per_fragment_output[:-1])
    # The compact hidden-content one-liner (bug signature) must not appear.
    assert "⚡ WritePlan content: ..." not in plain
    # Printer finished once the arguments JSON completed.
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_non_whitelisted_tool_unknown_no_stream(monkeypatch: Any) -> None:
    """An unknown tool (not in whitelist) with partial JSON produces no output
    via the legacy path and doesn't break when fragments arrive."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    tool_call = ToolCall(
        id="call-3",
        function=ToolCall.FunctionBody(name="UnknownTool", arguments='{"a": '),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='"x", "b": [1, 2'), session)
    # Truncated stream ends when a non-part message arrives — must not raise.
    await base.print_agent_json(TextPart(text="next"), session)

    plain = _plain(chunks)
    # The truncated line is terminated before the text prints.
    assert "\nnext" in plain
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


def test_tool_header_color_always_bright_magenta() -> None:
    for name in (
        "Python", "WriteFile", "WritePlan", "EditFile", "Bash", "Powershell",
        "Grep", "ReadFile", "TodoList", "Agent", "Compact", "NoSuchTool",
    ):
        assert base._tool_header_color(name) is base.Color.BRIGHT_MAGENTA


def test_stream_color_for_key_mapping_and_fallback() -> None:
    printer = base._ToolCallStreamPrinter
    assert printer._stream_color_for_key("old") is base.Color.BRIGHT_RED
    assert printer._stream_color_for_key("new") is base.Color.BRIGHT_GREEN
    assert printer._stream_color_for_key("code") is base.Color.BRIGHT_BLUE
    assert printer._stream_color_for_key("prompt") is base.Color.BRIGHT_YELLOW
    assert printer._stream_color_for_key("content") is base.Color.BRIGHT_BLACK
    assert printer._stream_color_for_key("context") is base.GRAY
    assert printer._stream_color_for_key("command") is base.Color.BRIGHT_BLUE
    # Aliases are canonicalized before lookup, so the raw color dict
    # returns the fallback for alias keys.
    assert printer._stream_color_for_key("old_string") is base.GRAY_LIGHT
    assert printer._stream_color_for_key("new_string") is base.GRAY_LIGHT
    assert printer._stream_color_for_key("anything_else") is base.GRAY_LIGHT


async def test_stream_prints_with_alias_old_string(monkeypatch: Any) -> None:
    """When the LLM sends old_string / new_string aliases, the stream
    printer canonicalizes them and displays old: / new: with correct
    colors (red for old, green for new)."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="EditFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"path": "f.py", "edit": [{"old_string": "aaa", "new_string": "bbb"}]}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi("".join(chunks))

    # Header printed.
    assert "\x1b[95m\u26a1 EditFile\x1b[0m" in output
    # Old value displayed in bright red, labeled "old:" (canonical).
    assert "\x1b[91maaa\x1b[0m" in output, "old_string value should be bright red"
    assert "\nold:\n" in plain, "alias old_string should display as canonical 'old:'"
    # New value displayed in bright green, labeled "new:" (canonical).
    assert "\x1b[92mbbb\x1b[0m" in output, "new_string value should be bright green"
    assert "\nnew:\n" in plain, "alias new_string should display as canonical 'new:'"
    # Stream finished cleanly.
    assert base._stream._last_char_was_newline is True


async def test_stream_prints_editplan_with_aliases(monkeypatch: Any) -> None:
    """EditPlan streaming works with old_string / new_string aliases."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="EditPlan", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"edit": [{"old_string": "aaa", "new_string": "bbb"}]}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi("".join(chunks))

    # EditPlan header printed (bright magenta).
    assert "\x1b[95m\u26a1 EditPlan\x1b[0m" in output
    # Canonical labels displayed.
    assert "\nold:\n" in plain
    assert "\nnew:\n" in plain
    # Values streamed with correct colors.
    assert "\x1b[91maaa\x1b[0m" in output
    assert "\x1b[92mbbb\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_alias_text_maps_to_content(monkeypatch: Any) -> None:
    """The 'text' alias streams as 'content' with bright black color."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"path": "x.py", "text": "hello world"}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi("".join(chunks))

    # Label is canonical "content:".
    assert "\ncontent:\n" in plain, "alias 'text' should display as canonical 'content:'"
    # Value is streamed (not compact) with bright black color.
    assert "\x1b[90mhello world\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_alias_source_code_maps_to_code(monkeypatch: Any) -> None:
    """The 'source_code' alias streams as 'code' with bright blue color."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Python", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"source_code": "print(1)"}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi("".join(chunks))

    # Label is canonical "code:".
    assert "\ncode:\n" in plain, "alias 'source_code' should display as canonical 'code:'"
    # Value is streamed with bright blue color.
    assert "\x1b[94mprint(1)\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_alias_task_maps_to_prompt(monkeypatch: Any) -> None:
    """The 'task' alias streams as 'prompt' with bright yellow color."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Agent", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"task": "do it now"}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi("".join(chunks))

    # Label is canonical "prompt:".
    assert "\nprompt:\n" in plain, "alias 'task' should display as canonical 'prompt:'"
    # Value is streamed with bright yellow color.
    assert "\x1b[93mdo it now\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_colors_writefile_header_and_content_white(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "content": "hello"}'), session)

    output = "".join(chunks)
    # Header is always bright magenta.
    assert "\x1b[95m⚡ WriteFile\x1b[0m" in output
    # Streamed content value color-coded bright black.
    assert "\x1b[90mhello\x1b[0m" in output
    # Non-streamed compact segments keep the legacy magenta and start on
    # their own line (newline prefix).
    assert "\x1b[95m\npath:\nx.py\x1b[0m" in output


async def test_stream_colors_editfile_old_red_new_green(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="EditFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "f.py", "edit": [{"old": "aaa", "new": "bbb"}]}'),
        session,
    )

    output = "".join(chunks)
    assert "\x1b[95m⚡ EditFile\x1b[0m" in output   # header bright magenta
    assert "\x1b[91maaa\x1b[0m" in output            # old -> bright red
    assert "\x1b[92mbbb\x1b[0m" in output            # new -> bright green


async def test_stream_prints_each_argument_on_new_line(monkeypatch: Any) -> None:
    """Each streamed tool argument starts on its own line (no ", " joins)."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="EditFile", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "f.py", "edit": [{"old": "aaa", "new": "bbb"}]}'),
        session,
    )

    plain = _plain(chunks)

    # Header on its own line, then exactly one argument per line.
    assert "⚡ EditFile\n" in plain
    assert "\npath:\nf.py" in plain
    assert "\nold:\naaa" in plain
    assert "\nnew:\nbbb" in plain
    # No comma-separated arguments remain.
    assert ", old:" not in plain
    assert ", new:" not in plain
    assert ", path:" not in plain
    # Stream is still terminated cleanly.
    assert base._stream._last_char_was_newline is True


async def test_stream_colors_python_code_blue(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Python", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"code": "print(1)"}'), session)

    output = "".join(chunks)
    assert "\x1b[95m⚡ Python\x1b[0m" in output      # header bright magenta
    assert "\x1b[94mprint(1)\x1b[0m" in output       # code -> bright blue


async def test_stream_colors_agent_prompt(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Agent", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"prompt": "do it"}'), session)

    output = "".join(chunks)
    assert "\x1b[95m⚡ Agent\x1b[0m" in output       # header magenta (unchanged)
    assert "\x1b[93mdo it\x1b[0m" in output          # prompt -> bright yellow


async def test_tool_header_color_compact_path_and_fallback(monkeypatch: Any) -> None:
    # Non-whitelisted tool (compact path): Grep header is bright magenta too.
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(id="c1", function=ToolCall.FunctionBody(
            name="Grep", arguments='{"pattern": "def "}')), session)
    assert "\x1b[95m⚡ Grep" in "".join(chunks)

    # Unknown tool (compact path, default case): also bright magenta header.
    chunks2 = _capture_base_stream(monkeypatch)
    session2 = FakeSession()
    await base.print_agent_json(
        ToolCall(id="c2", function=ToolCall.FunctionBody(
            name="MysteryTool", arguments='{"a": 1}')), session2)
    assert "\x1b[95m⚡ MysteryTool" in "".join(chunks2)


async def test_tool_header_not_reprinted_after_tool_result(monkeypatch: Any) -> None:
    """Regression: the compact tool header must not be printed again after
    the tool result arrives.

    _handle_tool_result clears _TOOL_HEADER_PRINTED_KEY but (when the tool
    call is found by id) left the stale _LAST_TOOL_CALL_KEY behind. The next
    non-toolcall wire message then triggered _finish_tool_call_stream, whose
    final parse attempt re-printed the header a second time."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="Powershell", arguments='{"cmd": "git diff --stat"}'),
    )

    await base.print_agent_json(tool_call, session)
    tool_result = ToolResult(
        tool_call_id="call-1",
        return_value=ToolReturnValue(
            is_error=False,
            message="ok",
            output="",
            display=[ShellDisplayBlock(command="git diff --stat", language="powershell")],
        ),
    )
    await base.print_agent_json(tool_result, session)
    # Any subsequent non-toolcall message (next step, text, ...) must not
    # re-print the finished tool call's header.
    await base.print_agent_json(TextPart(text="next step"), session)

    plain = _plain(chunks)
    assert plain.count("⚡ Powershell") == 1
    assert "✓ Powershell" in plain


async def test_tool_header_not_reprinted_for_in_flight_call_on_earlier_results(
    monkeypatch: Any,
) -> None:
    """Regression: while the last streamed tool call is still in flight,
    results of earlier parallel calls must not re-print its header.

    _handle_tool_result used to clear _TOOL_HEADER_PRINTED_KEY for *any*
    result with display blocks. With parallel tool calls (the OpenAI
    Responses wire format: ``ToolCall(args='')`` + one ``ToolCallPart`` per
    call), the results of earlier calls then cleared the flag of the last,
    still-pending call, so each arriving result made
    _finish_tool_call_stream re-print the pending call's ``⚡`` header."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    # Proxy-style stream: header with empty args, then a single full-args part.
    await base.print_agent_json(
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="Glob", arguments="")),
        session,
    )
    await base.print_agent_json(ToolCallPart(arguments_part='{"pattern": "*.a"}'), session)
    await base.print_agent_json(
        ToolCall(id="call-2", function=ToolCall.FunctionBody(name="Glob", arguments="")),
        session,
    )
    await base.print_agent_json(ToolCallPart(arguments_part='{"pattern": "*.b"}'), session)

    def _result(call_id: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call_id,
            return_value=ToolReturnValue(
                is_error=False,
                message="ok",
                output="",
                display=[ShellDisplayBlock(command="glob", language="text")],
            ),
        )

    # Earlier call's result arrives while the last call is still in flight.
    await base.print_agent_json(_result("call-1"), session)
    await base.print_agent_json(_result("call-2"), session)
    await base.print_agent_json(TextPart(text="next step"), session)

    plain = _plain(chunks)
    assert plain.count("⚡ Glob") == 2
    assert plain.count("✓ Glob") == 2


async def test_tool_result_colors_success_green_error_red(monkeypatch: Any) -> None:
    # Success result -> bright green.
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolResult(
            tool_call_id="c1",
            return_value=ToolReturnValue(is_error=False, message="ok", output="", display=[]),
        ),
        session,
    )
    assert "\x1b[92m✓ ok\x1b[0m" in "".join(chunks)

    # Failed result -> bright red.
    chunks2 = _capture_base_stream(monkeypatch)
    session2 = FakeSession()
    await base.print_agent_json(
        ToolResult(
            tool_call_id="c2",
            return_value=ToolReturnValue(is_error=True, message="boom", output="", display=[]),
        ),
        session2,
    )
    assert "\x1b[91m✗ boom\x1b[0m" in "".join(chunks2)


def test_format_tool_args_powershell_bash_prints_cmd_and_command_alias() -> None:
    """Regression: ``PowershellParams``/``BashParams`` (pwsh_tool.py /
    bash_tool.py) declare ``cmd`` with the pydantic alias ``command``, so the
    JSON schema advertised to the LLM names the field ``command`` and models
    typically send ``{"command": ...}``. The compact ``_format_tool_args``
    header must print the command for both the ``command`` alias and the
    ``cmd`` field name — previously only ``cmd`` was looked up, so the header
    showed no command at all when the LLM used the advertised alias."""
    for tool_name in ("Powershell", "Bash"):
        for key in ("command", "cmd"):
            formatted = base._format_tool_args(
                tool_name, orjson.dumps({key: "Get-Date", "timeout": 30}).decode("utf-8")
            )
            assert formatted is not None, f"{tool_name}: no formatted output for {key!r}"
            assert "Get-Date" in formatted, (
                f"{tool_name}: command missing from header for {key!r} args: {formatted!r}"
            )
            assert "timeout: 30" in formatted


async def test_powershell_bash_tool_call_header_prints_command_alias(monkeypatch: Any) -> None:
    """End-to-end: the printed ``⚡ Powershell`` / ``⚡ Bash`` tool-call header
    must include the command **inline** (on the same line) regardless of whether
    the LLM sent it under the advertised ``command`` alias or the ``cmd`` field
    name.  Both forms produce identical inline output ``⚡ Name Get-Date``."""
    for tool_name in ("Powershell", "Bash"):
        for key in ("command", "cmd"):
            chunks = _capture_base_stream(monkeypatch)
            session = FakeSession()
            await base.print_agent_json(
                ToolCall(
                    id=f"call-{tool_name}-{key}",
                    function=ToolCall.FunctionBody(
                        name=tool_name,
                        arguments=orjson.dumps({key: "Get-Date"}).decode("utf-8"),
                    ),
                ),
                session,
            )
            plain = _plain(chunks)
            # Header and command appear on the same line (inline).
            # The output looks like: ⚡ Powershell Get-Date
            assert f"\u26a1 {tool_name} " in plain, (
                f"{tool_name}: inline header+command missing for {key!r} args: {plain!r}"
            )
            assert "Get-Date" in plain, (
                f"{tool_name}: header missing command for {key!r} args: {plain!r}"
            )
            # No "command:" label — command is inline.
            assert "command:" not in plain, (
                f"{tool_name}: unexpected 'command:' label for {key!r} args: {plain!r}"
            )


async def test_powershell_bash_streamed_fragments_print_command_alias(monkeypatch: Any) -> None:
    """Anthropic/OpenAI-Responses style streaming: ToolCall with empty
    arguments followed by ToolCallPart fragments.  Since Powershell/Bash
    are now in ``_STREAM_TOOL_NAMES``, the stream printer is created and
    the command is printed **inline** after the header as fragments arrive."""
    for tool_name in ("Powershell", "Bash"):
        chunks = _capture_base_stream(monkeypatch)
        session = FakeSession()
        await base.print_agent_json(
            ToolCall(id=f"call-{tool_name}", function=ToolCall.FunctionBody(name=tool_name, arguments="")),
            session,
        )
        # Stream printer should have been created.
        assert base._TOOL_CALL_STREAM_KEY in session._tmp_data
        await base.print_agent_json(ToolCallPart(arguments_part='{"command": "Get-Da'), session)
        await base.print_agent_json(ToolCallPart(arguments_part='te"}'), session)
        plain = _plain(chunks)
        # Header + inline command on the same line.
        assert f"\u26a1 {tool_name} " in plain
        assert "Get-Date" in plain, (
            f"{tool_name}: streamed header missing command: {plain!r}"
        )
        # No "command:" label — command is inline.
        assert "command:" not in plain, (
            f"{tool_name}: unexpected 'command:' label in streamed output: {plain!r}"
        )


# ---------------------------------------------------------------------------
# Flush tests — verify that tool headers and results use flush=True
# so output appears immediately rather than sitting in stdout's buffer.
# ---------------------------------------------------------------------------


def _capture_base_stream_with_flush(monkeypatch: Any) -> tuple[list[str], list[bool]]:
    """Like _capture_base_stream but also captures ``flush`` flags per call."""
    chunks: list[str] = []
    flush_flags: list[bool] = []

    def print_func(
        *values: object,
        sep: str = " ",
        end: str = "\n",
        flush: bool = False,
        **kwargs: Any,
    ) -> None:
        chunks.append(sep.join(str(value) for value in values) + end)
        flush_flags.append(flush)

    monkeypatch.setattr(base, "_stream", base.PrintStream(print_func=print_func))
    monkeypatch.setattr(base, "_text_buffer", None)
    monkeypatch.setattr(base, "_quiet", False)
    monkeypatch.setattr(base, "_colorful_print", True)
    return chunks, flush_flags


async def test_compact_path_tool_header_flushes(monkeypatch: Any) -> None:
    """Fix 1: Compact-path headers (TaskOutput, etc.) must use ``flush=True``
    so the ``\u26a1`` line appears immediately, not buffered for seconds."""
    chunks, flush_flags = _capture_base_stream_with_flush(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(
                name="TaskOutput",
                arguments='{"task_id": "pwsh_xxx"}',
            ),
        ),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 TaskOutput" in plain
    assert any(flush_flags), (
        "No flush=True found — compact-path tool header did not flush"
    )


async def test_fragmented_compact_header_flushes(monkeypatch: Any) -> None:
    """Fix 1b: When compact-path arguments arrive via ToolCallPart fragments,
    ``_print_compact_tool_header`` must use ``flush=True``."""
    chunks, flush_flags = _capture_base_stream_with_flush(monkeypatch)
    session = FakeSession()

    # Empty-args header + one fragment that completes the JSON.
    await base.print_agent_json(
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="Glob", arguments="")),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"pattern": "*.py"}'),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 Glob" in plain
    assert any(flush_flags), (
        "No flush=True found — fragmented compact header did not flush"
    )


async def test_stream_path_header_flushes(monkeypatch: Any) -> None:
    """Fix 2: Stream-path tool headers must use ``flush=True`` for immediate
    visibility even before argument fragments arrive."""
    chunks, flush_flags = _capture_base_stream_with_flush(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="WriteFile", arguments=None),
        ),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 WriteFile" in plain
    assert any(flush_flags), (
        "No flush=True found — stream-path tool header did not flush"
    )


async def test_tool_result_flushes(monkeypatch: Any) -> None:
    """Fix 3: Tool-result ``\u2713/\u2717`` lines must use ``flush=True`` so
    success/failure feedback appears immediately."""
    chunks, flush_flags = _capture_base_stream_with_flush(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False,
                message="ok",
                output="",
                display=[],
            ),
        ),
        session,
    )

    plain = _plain(chunks)
    assert "\u2713 ok" in plain
    assert any(flush_flags), (
        "No flush=True found — tool result did not flush"
    )
