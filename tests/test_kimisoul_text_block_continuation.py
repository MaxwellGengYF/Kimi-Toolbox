from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, LLM
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Agent
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, _MAX_TEXT_BLOCK_CONTINUATION_ROUNDS
from kimi_cli.wire import Wire
from kimi_cli.wire.types import TextPart
from kosong.message import ContentPart, Message, ToolCall
from kosong.tooling import HandleResult, Tool, ToolResult as ToolResultObj, ToolReturnValue
from kosong.tooling.empty import EmptyToolset

from tests.test_prompt_steer import _make_runtime


class _NoopToolset(EmptyToolset):
    """Toolset with a single dummy tool that always succeeds."""

    @property
    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="noop",
                description="A no-op tool used for testing.",
                parameters={"type": "object", "properties": {}},
            )
        ]

    def handle(self, tool_call: ToolCall) -> HandleResult:
        if tool_call.function.name == "noop":
            return ToolResultObj(
                tool_call_id=tool_call.id,
                return_value=ToolReturnValue(
                    is_error=False,
                    output="done",
                    message="noop completed",
                    display=[],
                ),
            )
        return super().handle(tool_call)


class _StreamedMessage:
    def __init__(self, parts: list[ContentPart]) -> None:
        self._parts = list(parts)

    async def _iter(self):
        for part in self._parts:
            yield part

    def __aiter__(self):
        return self._iter()

    @property
    def id(self) -> str | None:
        return "test"

    @property
    def usage(self):
        return None


class _ScriptedChatProvider:
    name = "scripted"

    def __init__(self, sequences: list[list[ContentPart]]) -> None:
        self._sequences = [list(parts) for parts in sequences]
        self._calls = 0
        self._generation_kwargs: dict[str, object] = {}

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def thinking_effort(self):
        return None

    async def generate(self, system_prompt, tools, history):
        index = min(self._calls, len(self._sequences) - 1)
        self._calls += 1
        return _StreamedMessage(self._sequences[index])

    def with_thinking(self, effort):
        return self


class _RecordingScriptedChatProvider(_ScriptedChatProvider):
    """Scripted provider that also records the history passed to each LLM call."""

    def __init__(self, sequences: list[list[ContentPart]]) -> None:
        super().__init__(sequences)
        self._histories: list[list[Message]] = []

    async def generate(self, system_prompt, tools, history):
        self._histories.append(list(history))
        return await super().generate(system_prompt, tools, history)


class _InstructionGatedChatProvider(_ScriptedChatProvider):
    """Returns empty text unless the continuation instruction reaches the LLM.

    Simulates the real-world failure: after a tool call the model's next
    response is empty/whitespace, and it only produces a final text block when
    it actually *sees* the forced-continuation user message. If that message is
    stripped before the step (the regression), every forced step comes back
    empty and the session ends on no final text.
    """

    def __init__(self, empty_parts: list[ContentPart] | None = None) -> None:
        super().__init__([])
        self._empty_parts = empty_parts if empty_parts is not None else [TextPart(text="   ")]
        self._final_text = [TextPart(text="All done.")]

    async def generate(self, system_prompt, tools, history):
        self._calls += 1
        continuation_seen = any(
            m.role == "user" and "did not end with a plain text block" in m.extract_text(" ")
            for m in history
        )
        return _StreamedMessage(self._final_text if continuation_seen else self._empty_parts)


def _make_soul(tmp_path: Path, runtime) -> KimiSoul:
    agent = Agent(
        name="Text Block Gate Test Agent",
        system_prompt="Test prompt.",
        toolset=_NoopToolset(),
        runtime=runtime,
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "context.jsonl"))


@pytest.mark.asyncio
async def test_soul_forces_continuation_after_empty_final_message(
    tmp_path: Path,
) -> None:
    """After a tool call, an empty/truncated assistant message should trigger
    the soul-level text-block gate, forcing a continuation step that ends on
    a plain text block.
    """
    # Sequence:
    # 1. assistant calls noop tool
    # 2. assistant returns whitespace-only message (non-text final block)
    # 3. assistant returns final text after the soul-level continuation prompt
    provider = _ScriptedChatProvider(
        [
            [ToolCall(id="call-1", function=ToolCall.FunctionBody(name="noop", arguments="{}"))],
            [TextPart(text="   ")],
            [TextPart(text="All done.")],
        ]
    )
    runtime = _make_runtime(tmp_path, provider)
    soul = _make_soul(tmp_path, runtime)

    received: list[object] = []

    async def ui_loop(wire: Wire) -> None:
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except Exception:
                return
            received.append(msg)

    await run_soul(soul, "do the thing", ui_loop, asyncio.Event())

    # The final assistant message must be plain text, not empty.
    final = soul.context.history[-1]
    assert final.role == "assistant"
    assert final.content == [TextPart(text="All done.")]

    # The continuation user message should have been injected as a real
    # user message (so it survives ``strip_system_reminders`` in the next
    # step and actually reaches the LLM). Verify both the wire and the
    # persisted history.
    assert any(
        isinstance(msg, TextPart)
        and "did not end with a plain text block" in msg.text
        for msg in received
    ), f"No continuation text was sent on the wire. Received: {received}"
    user_messages = [m for m in soul.context.history if m.role == "user"]
    assert any(
        "did not end with a plain text block" in m.extract_text()
        for m in user_messages
    ), "Continuation prompt was stripped before reaching the LLM"


@pytest.mark.asyncio
async def test_continuation_prompt_reaches_next_llm_step(
    tmp_path: Path,
) -> None:
    """The text-block continuation must actually reach the forced LLM step.

    Regression test for the session-truncation bug where the continuation was
    appended as a ``<system-reminder>`` user message, which
    ``strip_system_reminders`` removes at the start of the very next step —
    so the LLM never saw the instruction and the turn ended on an empty
    message right after a tool call ("Finished" with no final text block).
    """
    provider = _RecordingScriptedChatProvider(
        [
            [ToolCall(id="call-1", function=ToolCall.FunctionBody(name="noop", arguments="{}"))],
            [TextPart(text="   ")],
            [TextPart(text="All done.")],
        ]
    )
    runtime = _make_runtime(tmp_path, provider)
    soul = _make_soul(tmp_path, runtime)

    await run_soul(soul, "do the thing", _collecting_ui_loop, asyncio.Event())

    # The final assistant message must be plain text, not empty.
    final = soul.context.history[-1]
    assert final.role == "assistant"
    assert final.content == [TextPart(text="All done.")]

    # The 3rd LLM call is the forced continuation step.  Its input history
    # must contain the continuation prompt — otherwise the gate is a no-op.
    assert len(provider._histories) >= 3
    third_history = provider._histories[2]
    third_text = " ".join(
        m.extract_text(" ") for m in third_history if m.role == "user"
    )
    assert "did not end with a plain text block" in third_text
    # And it must be a real user instruction, not a strippable reminder.
    assert "<system-reminder>" not in third_text


@pytest.mark.asyncio
async def test_turn_does_not_end_empty_when_continuation_reaches_llm(
    tmp_path: Path,
) -> None:
    """A turn that would end on empty text must keep forcing steps until the
    continuation instruction is actually seen, then finish with text.

    The provider deliberately returns empty text unless the continuation user
    message reaches its input history. Before the fix the continuation was
    stripped as a ``<system-reminder>``, so the turn ended on the empty
    message (the reported 'session truncation' symptom).
    """
    provider = _InstructionGatedChatProvider()
    runtime = _make_runtime(tmp_path, provider)
    soul = _make_soul(tmp_path, runtime)

    await run_soul(soul, "do the thing", _collecting_ui_loop, asyncio.Event())

    final = soul.context.history[-1]
    assert final.role == "assistant"
    assert final.content == [TextPart(text="All done.")]
    assert provider._calls <= 1 + _MAX_TEXT_BLOCK_CONTINUATION_ROUNDS + 1


async def _collecting_ui_loop(wire: Wire) -> None:
    wire_ui = wire.ui_side(merge=False)
    while True:
        try:
            await wire_ui.receive()
        except Exception:
            return
