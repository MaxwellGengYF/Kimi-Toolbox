from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, LLM
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Agent
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
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

    # The continuation user message should have been injected into history.
    user_messages = [m for m in soul.context.history if m.role == "user"]
    assert any(
        "did not end with a plain text block" in m.extract_text() for m in user_messages
    )
