from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
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
import kimix.ui.stream as stream_mod

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float
    context_tokens: int


# Default tool names exposed by FakeSession's stub toolset, mirroring the
# real kimix toolset so display-name resolution (write_file -> write,
# AppendFile -> write, ...) works exactly as in production.
_DEFAULT_TOOL_NAMES = (
    "write", "WritePlan", "read", "ReadPlan", "edit", "EditPlan",
    "python", "subagent", "list_agents", "interrupt_agent", "Run", "pwsh",
    "bash", "grep", "glob", "fetch_url", "todo_write", "job_output",
    "compact", "retrieve",
)


class FakeSession:
    def __init__(
        self,
        context_usage: float = 0.125,
        context_tokens: int = 1024,
        tool_names: tuple[str, ...] | None = _DEFAULT_TOOL_NAMES,
    ) -> None:
        self.status = FakeStatus(context_usage=context_usage, context_tokens=context_tokens)
        self.cancelled = False
        self._cancel_event = None
        self._tmp_data = {}
        if tool_names is not None:
            # Stub the session._cli.soul.agent.toolset.tools chain used by
            # kimix.ui.stream._session_tool_names for display-name resolution.
            # ``find`` is stubbed too: kimix.utils.prompt looks up the
            # todo_write tool instance via toolset.find("todo_write").
            self._cli = SimpleNamespace(
                soul=SimpleNamespace(
                    agent=SimpleNamespace(
                        toolset=SimpleNamespace(
                            tools=[SimpleNamespace(name=n) for n in tool_names],
                            find=lambda *a, **k: None,
                        )
                    )
                )
            )

    async def prompt(self, _prompt: str, *, merge_wire_messages: bool = False) -> Any:
        del merge_wire_messages
        yield TextPart(text="prompt output")

    def cancel(self) -> None:
        self.cancelled = True


def _capture_base_stream(monkeypatch: Any) -> list[str]:
    chunks: list[str] = []

    def print_func(*values: object, sep: str = " ", end: str = "\n", **_kwargs: Any) -> None:
        chunks.append(sep.join(str(value) for value in values) + end)

    new_stream = base.PrintStream(print_func=print_func)
    monkeypatch.setattr(base, "_stream", new_stream)
    monkeypatch.setattr(stream_mod, "_stream", new_stream)
    monkeypatch.setattr(base, "_text_buffer", None)
    monkeypatch.setattr(stream_mod, "_text_buffer", None)
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
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"path": "x.py", "content": "hello'), session)
    await base.print_agent_json(ToolCallPart(arguments_part=" wor"), session)
    await base.print_agent_json(ToolCallPart(arguments_part='ld"}'), session)

    output = "".join(chunks)
    plain = _plain(chunks)

    # Header printed exactly once when the ToolCall arrives.
    assert output.count("⚡ write") == 1
    # Short arguments print inline on the header line (key:value).
    assert " path:x.py" in plain
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
        function=ToolCall.FunctionBody(name="write", arguments=None),
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
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"content": "\\u4f'), session)
    assert "你" not in "".join(chunks)

    await base.print_agent_json(ToolCallPart(arguments_part='60"}'), session)

    plain = _plain(chunks)
    assert plain.count("你") == 1
    assert "\\u4f" not in plain


async def test_print_agent_json_split_unicode_escape_right_after_backslash_u(monkeypatch: Any) -> None:
    """A valid ``\\uXXXX`` escape split immediately after the ``\\u`` (the
    most fragile boundary for the escape state machine) must still decode."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(ToolCallPart(arguments_part='{"content": "\\u'), session)
    await base.print_agent_json(ToolCallPart(arguments_part='4f60"}'), session)

    plain = _plain(chunks)
    assert plain.count("你") == 1
    assert "\\u4f" not in plain
    assert base._stream._last_char_was_newline is True


async def test_stream_malformed_unicode_escape_does_not_swallow_quote(monkeypatch: Any) -> None:
    """Regression (Python tool): when the LLM emits a single-backslash ``\\u``
    that is not a real unicode escape (e.g. a Python raw string ``r"\\u"``
    serialized without JSON-escaped backslashes), the lexer used to swallow
    the closing quote, leak the rest of the JSON document into the streamed
    ``code:`` value and get stuck mid-string — the displayed code became
    ``print(r\\u", "mode`` and the tool result merged into the garbage.

    The malformed escape must be emitted verbatim, the string must still
    terminate, and the remaining arguments must print normally.
    """
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="python", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(ToolCallPart(arguments_part='{"code": "print(r'), session)
    await base.print_agent_json(ToolCallPart(arguments_part='\\u", "mode": "run"}'), session)
    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False, message="success", output="", display=[]
            ),
        ),
        session,
    )

    plain = _plain(chunks)
    # The malformed escape is emitted verbatim, not dropped.
    assert "print(r\\u" in plain
    # No JSON scaffolding leaks into the streamed code value.
    assert '", "mode' not in plain
    assert '"}' not in plain
    # The remaining argument prints normally under its own label.
    assert " mode:run" in plain
    # The tool result renders on its own line, not merged into the code.
    assert "\n✓ python" in plain


async def test_stream_malformed_unicode_escape_recovers_mid_string(monkeypatch: Any) -> None:
    """A malformed ``\\u`` followed by more string content (e.g. ``\\u12z``)
    must emit the buffered escape verbatim and continue decoding the rest of
    the value without duplicating or dropping characters."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="python", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"code": "x = \\u12z", "mode": "run"}'),
        session,
    )

    plain = _plain(chunks)
    # ``\\u12`` emitted verbatim, ``z`` appended exactly once (no duplication).
    assert "x = \\u12z" in plain
    assert "x = \\u12zz" not in plain
    assert " mode:run" in plain
    # The streamed line is terminated cleanly before the tool result.
    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False, message="success", output="", display=[]
            ),
        ),
        session,
    )
    plain = _plain(chunks)
    assert "\n✓ python" in plain


async def test_stream_single_backslash_windows_path_preserved(monkeypatch: Any) -> None:
    """LLMs frequently emit single-backslash Windows paths in tool
    arguments (``r'C:\\dev\\src'`` — invalid JSON escapes, but common).
    Unknown two-char escapes must keep their backslash instead of decoding
    to the bare character, so the path is not displayed mangled
    (``C:devsrc``)."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="python", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"code": "p = r\'C:\\dev\\src\'", "timeout": 30}'),
        session,
    )

    plain = _plain(chunks)
    assert "p = r'C:\\dev\\src'" in plain, f"path mangled: {plain!r}"
    assert " timeout:30" in plain
    # The streamed line is terminated cleanly before the tool result.
    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False, message="success", output="", display=[]
            ),
        ),
        session,
    )
    plain = _plain(chunks)
    assert "\n✓ python" in plain


async def test_stream_unknown_escape_keeps_backslash(monkeypatch: Any) -> None:
    """Regex escapes emitted without JSON escaping (``\\d``, ``\\s``) keep
    their backslash in the streamed display instead of collapsing to the
    bare letter."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="python", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part="{\"code\": \"import re\\nr = re.compile(r'\\d+\\s')\"}"),
        session,
    )

    plain = _plain(chunks)
    assert "r'\\d+\\s'" in plain, f"regex escapes mangled: {plain!r}"
    # The streamed line is terminated cleanly before the tool result.
    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False, message="success", output="", display=[]
            ),
        ),
        session,
    )
    plain = _plain(chunks)
    assert "\n✓ python" in plain


async def test_print_agent_json_stream_prints_compact_short_values(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "mode": "overwrite", "content": "body"}'),
        session,
    )

    plain = _plain(chunks)

    assert " path:x.py" in plain
    # Short scalar args (``mode``) print inline on the header line.
    assert " mode:overwrite" in plain
    assert "mode:\noverwrite" not in plain
    assert "body" in plain


async def test_streamed_short_args_print_inline(monkeypatch: Any) -> None:
    """Short scalar arguments (e.g. ``timeout``) print inline after the tool
    header — ``⚡ pwsh Get-Date timeout:30`` — instead of each
    occupying its own ``timeout:\n30`` line beneath the header."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-inline-timeout",
            function=ToolCall.FunctionBody(name="pwsh", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"command": "Get-Date", "timeout": 30}'),
        session,
    )

    plain = _plain(chunks)
    assert "Get-Date timeout:30" in plain
    assert "timeout:\n30" not in plain


async def test_print_agent_json_stream_finished_by_tool_result(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write", arguments=None),
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
    # Tracked tool call renders as `✓ {ToolName}` header + dim `  {message}`.
    assert "\n✓ write" in plain
    assert "\n  ok" in plain
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_print_agent_json_merged_tool_call_prints_full_content_once(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    args = orjson.dumps({"path": "big.py", "content": "full body here"}).decode("utf-8")
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write", arguments=args),
    )

    await base.print_agent_json(tool_call, session)

    plain = _plain(chunks)
    assert plain.count("full body here") == 1
    assert " path:big.py" in plain
    assert base._stream._last_char_was_newline is True




async def test_any_tool_streams_short_args_inline(monkeypatch: Any) -> None:
    """Every tool streams — there is no whitelist.  A complete ToolCall for
    grep prints the header plus all short arguments inline on one line."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="grep",
            arguments='{"pattern": "def ", "path": ".", "-n": true}',
        ),
    )

    await base.print_agent_json(tool_call, session)

    output = "".join(chunks)
    plain = _plain(chunks)

    # Header printed exactly once, with all short args inline (key:value).
    assert output.count("⚡ grep") == 1
    assert "⚡ grep pattern:def " in plain
    assert " path:." in plain
    # The grep CLI-flag alias ``-n`` displays under its canonical name.
    assert " line_number:True" in plain
    # Stream finished (complete JSON): no printer left behind.
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


async def test_any_tool_streams_fragmented_args(monkeypatch: Any) -> None:
    """A tool with ToolCall(arguments=None) + fragments streams live via the
    stream printer — the header prints at ToolCall time, args appear as the
    fragments complete."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    tool_call = ToolCall(
        id="call-2",
        function=ToolCall.FunctionBody(name="grep", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    # Header printed immediately; stream printer was created.
    assert "⚡ grep" in _plain(chunks)
    assert base._TOOL_CALL_STREAM_KEY in session._tmp_data

    # Send fragments that build up complete JSON.
    await base.print_agent_json(ToolCallPart(arguments_part='{"pattern": "def ", "path": "'), session)
    await base.print_agent_json(ToolCallPart(arguments_part='.", "-n": true}'), session)

    plain = _plain(chunks)

    assert "⚡ grep pattern:def " in plain
    assert " path:." in plain
    # Only one header.
    assert plain.count("⚡ grep") == 1
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


async def test_unknown_tool_truncated_stream_recovers(monkeypatch: Any) -> None:
    """An unknown tool (not in the session toolset) still prints its header
    and streams generically; a truncated argument stream is terminated
    cleanly when a non-part message arrives — must not raise."""
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
    # Header printed once with the raw name; short args streamed inline.
    assert plain.count("⚡ UnknownTool") == 1
    assert " a:x" in plain
    # The truncated line is terminated before the text prints.
    assert "\nnext" in plain
    assert base._TOOL_CALL_STREAM_KEY not in session._tmp_data


def test_tool_header_color_always_bright_magenta() -> None:
    for name in (
        "python", "write", "WritePlan", "edit", "bash", "pwsh",
        "grep", "read", "todo_write", "subagent", "compact", "NoSuchTool",
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
        function=ToolCall.FunctionBody(name="edit", arguments=None),
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
    assert "\x1b[95m\u26a1 edit\x1b[0m" in output
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
        function=ToolCall.FunctionBody(name="write", arguments=None),
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
        function=ToolCall.FunctionBody(name="python", arguments=None),
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
        function=ToolCall.FunctionBody(name="subagent", arguments=None),
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
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "content": "hello"}'), session)

    output = "".join(chunks)
    # Header is always bright magenta.
    assert "\x1b[95m⚡ write\x1b[0m" in output
    # Streamed content value color-coded bright black.
    assert "\x1b[90mhello\x1b[0m" in output
    # Short args stay inline on the header line (magenta, space prefix).
    assert "\x1b[95m path:x.py\x1b[0m" in output


async def test_stream_colors_editfile_old_red_new_green(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="edit", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "f.py", "edit": [{"old": "aaa", "new": "bbb"}]}'),
        session,
    )

    output = "".join(chunks)
    assert "\x1b[95m⚡ edit\x1b[0m" in output   # header bright magenta
    assert "\x1b[91maaa\x1b[0m" in output            # old -> bright red
    assert "\x1b[92mbbb\x1b[0m" in output            # new -> bright green


async def test_stream_prints_only_stream_keys_on_new_line(monkeypatch: Any) -> None:
    """Short args stay inline on the header line; only ``_STREAM_ARG_KEYS``
    values (old/new here) start on their own ``key:\\n`` line."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="edit", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "f.py", "edit": [{"old": "aaa", "new": "bbb"}]}'),
        session,
    )

    plain = _plain(chunks)

    # Short arg (path) is inline on the header line.
    assert "⚡ edit path:f.py" in plain
    # Streamed old/new values get their own labeled lines.
    assert "\nold:\naaa" in plain
    assert "\nnew:\nbbb" in plain
    # No per-line short-argument formatting remains.
    assert "\npath:" not in plain
    # Stream is still terminated cleanly.
    assert base._stream._last_char_was_newline is True


async def test_stream_colors_python_code_blue(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="python", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"code": "print(1)"}'), session)

    output = "".join(chunks)
    assert "\x1b[95m⚡ python\x1b[0m" in output      # header bright magenta
    assert "\x1b[94mprint(1)\x1b[0m" in output       # code -> bright blue


async def test_stream_colors_agent_prompt(monkeypatch: Any) -> None:
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="subagent", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"prompt": "do it"}'), session)

    output = "".join(chunks)
    assert "\x1b[95m⚡ subagent\x1b[0m" in output       # header magenta (unchanged)
    assert "\x1b[93mdo it\x1b[0m" in output          # prompt -> bright yellow


async def test_tool_header_color_compact_path_and_fallback(monkeypatch: Any) -> None:
    # Non-whitelisted tool (compact path): grep header is bright magenta too.
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(id="c1", function=ToolCall.FunctionBody(
            name="grep", arguments='{"pattern": "def "}')), session)
    assert "\x1b[95m⚡ grep" in "".join(chunks)

    # Unknown tool (compact path, default case): also bright magenta header.
    chunks2 = _capture_base_stream(monkeypatch)
    session2 = FakeSession()
    await base.print_agent_json(
        ToolCall(id="c2", function=ToolCall.FunctionBody(
            name="MysteryTool", arguments='{"a": 1}')), session2)
    assert "\x1b[95m⚡ MysteryTool" in "".join(chunks2)


async def test_tool_header_not_reprinted_after_tool_result(monkeypatch: Any) -> None:
    """Regression: the tool header must not be printed again after the tool
    result arrives.

    _handle_tool_result must drop the stale _LAST_TOOL_CALL_KEY (when the
    tool call is found by id); otherwise the next non-toolcall wire message
    would trigger _finish_tool_call_stream on the finished call."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="pwsh", arguments='{"cmd": "git diff --stat"}'),
    )

    await base.print_agent_json(tool_call, session)
    tool_result = ToolResult(
        tool_call_id="call-1",
        return_value=ToolReturnValue(
            is_error=False,
            message="ok",
            output="",
            display=[ShellDisplayBlock(language="powershell")],
        ),
    )
    await base.print_agent_json(tool_result, session)
    # Any subsequent non-toolcall message (next step, text, ...) must not
    # re-print the finished tool call's header.
    await base.print_agent_json(TextPart(text="next step"), session)

    plain = _plain(chunks)
    assert plain.count("⚡ pwsh") == 1
    # Result renders as `✓ {ToolName}` (tracked call) + dim `  {message}`.
    assert "✓ pwsh" in plain
    assert "\n  ok" in plain


async def test_tool_header_not_reprinted_for_in_flight_call_on_earlier_results(
    monkeypatch: Any,
) -> None:
    """Regression: while the last streamed tool call is still in flight,
    results of earlier parallel calls must not re-print its header.

    _handle_tool_result must only touch the state of the call the result
    belongs to. With parallel tool calls (the OpenAI Responses wire format:
    ``ToolCall(args='')`` + one ``ToolCallPart`` per call), clearing the
    last, still-pending call's state for an earlier result corrupted the
    in-flight call's display."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    # Proxy-style stream: header with empty args, then a single full-args part.
    await base.print_agent_json(
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="glob", arguments="")),
        session,
    )
    await base.print_agent_json(ToolCallPart(arguments_part='{"pattern": "*.a"}'), session)
    await base.print_agent_json(
        ToolCall(id="call-2", function=ToolCall.FunctionBody(name="glob", arguments="")),
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
                display=[ShellDisplayBlock(language="text")],
            ),
        )

    # Earlier call's result arrives while the last call is still in flight.
    await base.print_agent_json(_result("call-1"), session)
    await base.print_agent_json(_result("call-2"), session)
    await base.print_agent_json(TextPart(text="next step"), session)

    plain = _plain(chunks)
    assert plain.count("⚡ glob") == 2
    # Each tracked result renders as `✓ glob` header + dim `  ok` line.
    assert plain.count("✓ glob") == 2
    assert plain.count("\n  ok") == 2


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


def test_format_tool_args_cmd_and_command_alias() -> None:
    """Regression: ``PowershellParams``/``BashParams`` (pwsh_tool.py /
    bash_tool.py) declare ``cmd`` with the pydantic alias ``command``, so the
    JSON schema advertised to the LLM names the field ``command`` and models
    typically send ``{"command": ...}``. The generic ``format_tool_args``
    summary must print the command under its canonical key for both the
    ``command`` alias and the ``cmd`` field name — previously only ``cmd``
    was looked up, so the header showed no command at all when the LLM used
    the advertised alias."""
    for key in ("command", "cmd"):
        formatted = stream_mod.format_tool_args(
            orjson.dumps({key: "Get-Date", "timeout": 30}).decode("utf-8")
        )
        assert formatted == "command:Get-Date timeout:30", (
            f"unexpected summary for {key!r} args: {formatted!r}"
        )


def test_format_tool_args_generic_new_tool() -> None:
    """Future-compatible: a brand-new tool name needs no display code — the
    generic formatter summarizes any arguments dict as ``key:value`` pairs."""
    formatted = stream_mod.format_tool_args(
        orjson.dumps({"query": "abc", "limit": 5, "verbose": True}).decode("utf-8")
    )
    assert formatted == "query:abc limit:5 verbose:True"
    # Non-dict / invalid inputs behave like before.
    assert stream_mod.format_tool_args(None) is None
    assert stream_mod.format_tool_args("") == ""
    assert stream_mod.format_tool_args("not json") is None
    assert stream_mod.format_tool_args('[1, 2]') == "[1,2]"


async def test_powershell_bash_tool_call_header_prints_command_alias(monkeypatch: Any) -> None:
    """End-to-end: the printed ``⚡ pwsh`` / ``⚡ bash`` tool-call header
    must include the command **inline** (on the same line) regardless of whether
    the LLM sent it under the advertised ``command`` alias or the ``cmd`` field
    name.  Both forms produce identical inline output ``⚡ Name Get-Date``."""
    for tool_name in ("pwsh", "bash"):
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
            # The output looks like: ⚡ pwsh Get-Date
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
    arguments followed by ToolCallPart fragments.  The stream printer is
    created for every tool and the command is printed **inline** after the
    header as fragments arrive."""
    for tool_name in ("pwsh", "bash"):
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

    new_stream = base.PrintStream(print_func=print_func)
    monkeypatch.setattr(base, "_stream", new_stream)
    monkeypatch.setattr(stream_mod, "_stream", new_stream)
    monkeypatch.setattr(base, "_text_buffer", None)
    monkeypatch.setattr(stream_mod, "_text_buffer", None)
    monkeypatch.setattr(base, "_quiet", False)
    monkeypatch.setattr(base, "_colorful_print", True)
    return chunks, flush_flags


async def test_compact_path_tool_header_flushes(monkeypatch: Any) -> None:
    """Fix 1: Compact-path headers (job_output, etc.) must use ``flush=True``
    so the ``\u26a1`` line appears immediately, not buffered for seconds."""
    chunks, flush_flags = _capture_base_stream_with_flush(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(
                name="job_output",
                arguments='{"task_id": "pwsh_xxx"}',
            ),
        ),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 job_output" in plain
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
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="glob", arguments="")),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"pattern": "*.py"}'),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 glob" in plain
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
            function=ToolCall.FunctionBody(name="write", arguments=None),
        ),
        session,
    )

    plain = _plain(chunks)
    assert "\u26a1 write" in plain
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


# ---------------------------------------------------------------------------
# Fuzzy-matching parity tests (kimi / anthropic provider policy).
#
# Commit e673243 ("Add fuzzy tool argument call matching to defeat
# hallucination") made the *execution* layer (kosong.tooling) tolerate
# hallucinated tool names (auto-correct / TOOL_NAME_REDIRECTS) and a wide
# set of argument-key aliases (FIELD_ALIASES_FILE: old_str, new_str, data,
# file, changes, ...).  The streaming display in kimix/ui/stream.py mirrors
# the same resolution — names resolve against the session's live toolset
# (future-compatible: no hardcoded tool list) and keys canonicalize via
# _ARG_KEY_ALIASES — so calls that execute fine (after kosong repairs
# them) keep the live decoded streaming display.  Kimi/Anthropic models
# hit this frequently — e.g. Claude's native editor keys are
# ``old_str``/``new_str`` and snake_case tool names like ``write_file``
# are common.
# ---------------------------------------------------------------------------


async def test_short_args_print_on_one_header_line(monkeypatch: Any) -> None:
    """Short arguments never break onto their own lines — the whole call
    renders as a single ``⚡ Name key:value key:value`` line::

        ⚡ edit path:C:\\dev\\kimi-agent\\src\\kimix\\ui\\stream.py line_offset:1125 max_char:15000
    """
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    path = "C:\\dev\\kimi-agent\\src\\kimix\\ui\\stream.py"

    await base.print_agent_json(
        ToolCall(
            id="call-short-args",
            function=ToolCall.FunctionBody(
                name="edit",
                arguments=orjson.dumps(
                    {"path": path, "line_offset": 1125, "max_char": 15000}
                ).decode("utf-8"),
            ),
        ),
        session,
    )

    plain = _plain(chunks)
    expected = f"⚡ edit path:{path} line_offset:1125 max_char:15000"
    assert expected in plain, f"one-line header+args missing: {plain!r}"
    # The header line is a single line: no newline between header and args.
    header_line = next(
        line for line in plain.splitlines() if line.startswith("⚡ edit")
    )
    assert header_line == expected


async def test_new_tool_streams_without_code_change(monkeypatch: Any) -> None:
    """Future-compatible: a tool name this codebase has never heard of still
    streams generically — header + inline short args, long whitelisted keys
    on their own line — and hallucinated spellings of newly registered
    tools resolve via the session's live toolset."""
    # Brand-new tool registered in the session toolset; the model sends a
    # snake_case hallucination of it.
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession(tool_names=("MyCustomTool",))

    await base.print_agent_json(
        ToolCall(
            id="call-new",
            function=ToolCall.FunctionBody(name="my_custom_tool", arguments=None),
        ),
        session,
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"query": "abc", "limit": 5}'),
        session,
    )

    plain = _plain(chunks)
    # Header shows the resolved canonical name from the live toolset.
    assert "⚡ MyCustomTool" in plain
    # Generic inline short args, no per-tool code required.
    assert " query:abc" in plain
    assert " limit:5" in plain
    assert base._stream._last_char_was_newline is True


async def test_stream_fuzzy_alias_old_str_new_str(monkeypatch: Any) -> None:
    """Anthropic-style edit: Claude's native editor uses ``old_str`` /
    ``new_str``.  kosong's FIELD_ALIASES_FILE repairs them to ``old`` /
    ``new`` at execution, so the streamed display must show the canonical
    ``old:`` / ``new:`` labels with live red/green values — not a truncated
    60-char compact fallback."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="edit", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"path": "f.py", "edit": [{"old_str": "aaa", "new_str": "bbb"}]}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi(output)

    assert "\nold:\n" in plain, "alias 'old_str' should display as canonical 'old:'"
    assert "\nnew:\n" in plain, "alias 'new_str' should display as canonical 'new:'"
    # Values streamed live with correct colors (not compact/truncated).
    assert "\x1b[91maaa\x1b[0m" in output
    assert "\x1b[92mbbb\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_fuzzy_alias_data_maps_to_content(monkeypatch: Any) -> None:
    """kimi-style write: models frequently emit ``data`` (or ``body``)
    instead of ``content``; kosong repairs it, so the display must stream it
    under the canonical ``content:`` label."""
    for key in ("data", "body"):
        chunks = _capture_base_stream(monkeypatch)
        session = FakeSession()
        tool_call = ToolCall(
            id=f"call-{key}",
            function=ToolCall.FunctionBody(name="write", arguments=None),
        )

        await base.print_agent_json(tool_call, session)
        await base.print_agent_json(
            ToolCallPart(arguments_part='{"path": "x.py", "%s": "hello world"}' % key),
            session,
        )

        output = "".join(chunks)
        plain = base._strip_ansi(output)

        assert "\ncontent:\n" in plain, (
            f"alias {key!r} should display as canonical 'content:': {plain!r}"
        )
        assert "\x1b[90mhello world\x1b[0m" in output
        assert base._stream._last_char_was_newline is True


async def test_stream_fuzzy_alias_file_maps_to_path(monkeypatch: Any) -> None:
    """Models often emit ``file`` instead of ``path``; kosong repairs it.
    The path must not be mislabeled — it displays under the canonical
    ``path:`` key, inline on the header line."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"file": "x.py", "content": "hello"}'),
        session,
    )

    plain = _plain(chunks)
    assert " path:x.py" in plain, (
        f"alias 'file' should display as canonical 'path:': {plain!r}"
    )
    assert "\ncontent:\nhello" in plain


async def test_stream_snake_case_tool_name_write_file(monkeypatch: Any) -> None:
    """kimi-style hallucinated tool name ``write_file``: kosong's
    normalize_tool_name auto-correct resolves it to ``write`` at
    execution, so the display must resolve it too — the header shows the
    canonical name resolved against the session's live toolset and the
    content streams live."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="write_file", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    # Stream printer must be created for the resolved tool.
    assert base._TOOL_CALL_STREAM_KEY in session._tmp_data, (
        "no stream printer for resolved name 'write_file' -> 'write'"
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "content": "hello world"}'),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi(output)

    # Header shows the resolved canonical name.
    assert "\x1b[95m\u26a1 write\x1b[0m" in output
    assert "\ncontent:\n" in plain
    assert "\x1b[90mhello world\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_stream_redirected_tool_names(monkeypatch: Any) -> None:
    """kosong's TOOL_NAME_REDIRECTS maps common wrong names onto real tools
    (AppendFile -> write, ReplaceFile -> edit).  The streaming
    display must follow the same redirect so these calls stream live."""
    # AppendFile -> write: content streams.
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-append",
            function=ToolCall.FunctionBody(name="AppendFile", arguments=None),
        ),
        session,
    )
    assert base._TOOL_CALL_STREAM_KEY in session._tmp_data, (
        "no stream printer for redirected name 'AppendFile' -> 'write'"
    )
    await base.print_agent_json(
        ToolCallPart(arguments_part='{"path": "x.py", "content": "appended"}'),
        session,
    )
    output = "".join(chunks)
    plain = base._strip_ansi(output)
    assert "\x1b[95m\u26a1 write\x1b[0m" in output
    assert "\ncontent:\n" in plain
    assert "\x1b[90mappended\x1b[0m" in output

    # ReplaceFile -> edit: old/new stream with colors.
    chunks2 = _capture_base_stream(monkeypatch)
    session2 = FakeSession()
    await base.print_agent_json(
        ToolCall(
            id="call-replace",
            function=ToolCall.FunctionBody(name="ReplaceFile", arguments=None),
        ),
        session2,
    )
    assert base._TOOL_CALL_STREAM_KEY in session2._tmp_data, (
        "no stream printer for redirected name 'ReplaceFile' -> 'edit'"
    )
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"path": "f.py", "edit": [{"old": "aaa", "new": "bbb"}]}'
        ),
        session2,
    )
    output2 = "".join(chunks2)
    plain2 = base._strip_ansi(output2)
    assert "\x1b[95m\u26a1 edit\x1b[0m" in output2
    assert "\nold:\n" in plain2
    assert "\nnew:\n" in plain2
    assert "\x1b[91maaa\x1b[0m" in output2
    assert "\x1b[92mbbb\x1b[0m" in output2


async def test_stream_fuzzy_alias_changes_maps_to_edit(monkeypatch: Any) -> None:
    """Models sometimes emit ``changes`` instead of ``edit``/``edits`` for
    edit; kosong repairs the key.  The nested old/new values must still
    stream under the canonical labels."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="edit", arguments=None),
    )

    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolCallPart(
            arguments_part='{"path": "f.py", "changes": [{"old": "aaa", "new": "bbb"}]}'
        ),
        session,
    )

    output = "".join(chunks)
    plain = base._strip_ansi(output)

    assert "\nold:\n" in plain, "nested 'old' under 'changes' should still stream"
    assert "\nnew:\n" in plain
    assert "\x1b[91maaa\x1b[0m" in output
    assert "\x1b[92mbbb\x1b[0m" in output
    assert base._stream._last_char_was_newline is True


async def test_print_agent_json_think_parts_across_tool_boundary_all_printed(monkeypatch: Any) -> None:
    """Regression test for "thinking received by the provider but not printed".

    Multi-step agents think again after a tool result; every ThinkPart must
    reach the terminal regardless of the surrounding stream state
    (Thinking -> ToolCalling -> Thinking transitions).
    """
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()

    await base.print_agent_json(ThinkPart(think="step1 thinking"), session)
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="Run", arguments='{"command": "ls"}'),
    )
    await base.print_agent_json(tool_call, session)
    await base.print_agent_json(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(is_error=False, message="ok", output="", display=[]),
        ),
        session,
    )
    await base.print_agent_json(ThinkPart(think="step2 "), session)
    await base.print_agent_json(ThinkPart(think="thinking"), session)

    plain = _plain(chunks)
    assert "step1 thinking" in plain
    assert "step2 thinking" in plain
    # Each step's first ThinkPart re-prints the [Think] banner after the
    # state was reset by the tool call in between.
    assert plain.count("[Think]") == 2


async def test_print_agent_json_think_part_reaches_output_function(monkeypatch: Any) -> None:
    """Every ThinkPart must be forwarded to output_function as Thinking."""
    chunks = _capture_base_stream(monkeypatch)
    session = FakeSession()
    received: list[tuple[str, object]] = []

    def output_function(text: str, msg_type: object) -> None:
        received.append((text, msg_type))

    await base.print_agent_json(ThinkPart(think="abc"), session, output_function=output_function)
    await base.print_agent_json(ThinkPart(think="def"), session, output_function=output_function)

    from kimix.ui.printing import MessageType

    thinking = [t for t, ty in received if ty == MessageType.Thinking]
    assert thinking == ["abc", "def"]
    assert "abcdef" in _plain(chunks)


async def test_print_agent_json_think_part_emits_reasoning_debug(monkeypatch: Any, capsys: Any) -> None:
    """The KIMIX_DEBUG_REASONING stub must log ThinkPart receipt to stderr."""
    _capture_base_stream(monkeypatch)
    monkeypatch.setattr(stream_mod, "_REASONING_DEBUG_ENABLED", True)
    session = FakeSession()

    await base.print_agent_json(ThinkPart(think="debug me"), session)

    err = capsys.readouterr().err
    assert "[reasoning-debug]" in err
    assert "ThinkPart" in err
    assert "8 chars" in err
