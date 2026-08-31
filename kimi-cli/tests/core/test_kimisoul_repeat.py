from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Self

import pytest
from kosong.chat_provider import StreamedMessagePart, ThinkingEffort, TokenUsage
from kosong.message import Message, ToolCall
from kosong.tooling import CallableTool2, Tool, ToolOk, ToolReturnValue
from pydantic import BaseModel

import kimi_cli.soul.kimisoul as kimisoul_module
from kimi_cli.llm import LLM
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.toolset import (
    _CYCLE_FORCE_STOP,
    _DIFF_ARGS_HARD_STOP_START,
    _REPEAT_FORCE_STOP_STREAK,
    _TURN_TOOL_CALL_HARD_STOP,
    KimiToolset,
)


class _Params(BaseModel):
    value: str = ""


class _DummyTool(CallableTool2[_Params]):
    name = "ToolA"
    description = "Dummy tool that always succeeds."
    params = _Params

    async def __call__(self, params: _Params) -> ToolReturnValue:
        return ToolOk(output="a")


class _DummyToolB(CallableTool2[_Params]):
    name = "ToolB"
    description = "Second dummy tool for interleaved-call tests."
    params = _Params

    async def __call__(self, params: _Params) -> ToolReturnValue:
        return ToolOk(output="b")


class _RepeatStream:
    def __init__(self, parts: list[StreamedMessagePart]) -> None:
        self._iter = self._to_stream(parts)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    async def _to_stream(
        self, parts: list[StreamedMessagePart]
    ) -> AsyncIterator[StreamedMessagePart]:
        for part in parts:
            yield part

    @property
    def id(self) -> str | None:
        return "repeat"

    @property
    def usage(self) -> TokenUsage | None:
        return None


class _RepeatChatProvider:
    """Yields one tool call per step.

    ``cycle`` is the sequence of (tool_name, arguments) pairs replayed round
    robin; a one-element sequence reproduces a plain adjacent repeat, a longer
    one reproduces an interleaved cycle.
    """

    name = "repeat"

    def __init__(self, cycle: Sequence[tuple[str, str]] | None = None) -> None:
        self._n = 0
        self._cycle: Sequence[tuple[str, str]] = cycle or [("ToolA", '{"value":"x"}')]

    @property
    def model_name(self) -> str:
        return "repeat"

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> _RepeatStream:
        name, arguments = self._cycle[self._n % len(self._cycle)]
        self._n += 1
        tc = ToolCall(
            id=f"tc-{self._n}",
            function=ToolCall.FunctionBody(name=name, arguments=arguments),
        )
        return _RepeatStream([tc])

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        return self


def _runtime_with_llm(runtime: Runtime, llm: LLM) -> Runtime:
    return Runtime(
        config=runtime.config,
        llm=llm,
        session=runtime.session,
        builtin_args=runtime.builtin_args,
        denwa_renji=runtime.denwa_renji,
        approval=runtime.approval,
        labor_market=runtime.labor_market,
        environment=runtime.environment,
        notifications=runtime.notifications,
        background_tasks=runtime.background_tasks,
        skills=runtime.skills,
        oauth=runtime.oauth,
        additional_dirs=runtime.additional_dirs,
        skills_dirs=runtime.skills_dirs,
        role=runtime.role,
    )


def _make_soul(runtime: Runtime, llm: LLM, toolset: KimiToolset, tmp_path: Path) -> KimiSoul:
    agent = Agent(
        name="Repeat Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=_runtime_with_llm(runtime, llm),
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


@pytest.mark.asyncio
async def test_turn_force_stops_on_adjacent_identical_calls(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn replaying one identical call ends via the loop guard.

    Whichever ceiling is hit first wins: the consecutive-streak force stop
    (``_REPEAT_FORCE_STOP_STREAK``) or the absolute per-turn call ceiling
    (``_TURN_TOOL_CALL_HARD_STOP``).
    """
    toolset = KimiToolset()
    toolset.add(_DummyTool())
    llm = LLM(
        chat_provider=_RepeatChatProvider(),
        max_context_size=100_000,
        capabilities=set(),
    )
    soul = _make_soul(runtime, llm, toolset, tmp_path)

    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _msg: None)

    async def _noop_checkpoint() -> None:
        return None

    monkeypatch.setattr(soul, "_checkpoint", _noop_checkpoint)
    monkeypatch.setattr(soul._denwa_renji, "set_n_checkpoints", lambda _n: None)

    outcome = await soul._turn(Message(role="user", content="go"))

    assert outcome.stop_reason == "tool_call_repeat"
    assert outcome.step_count == min(_REPEAT_FORCE_STOP_STREAK, _TURN_TOOL_CALL_HARD_STOP)


@pytest.mark.asyncio
async def test_turn_force_stops_on_interleaved_cycle(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``A(x) -> B(x) -> A(x) -> B(x)`` must stop via the cycle detector.

    The consecutive streak can never grow here, and neither the per-tool
    (``_DIFF_ARGS_HARD_STOP_START``) nor the per-turn
    (``_TURN_TOOL_CALL_HARD_STOP``) ceiling is reached — only the cycle-aware
    punishment stops the loop.
    """
    toolset = KimiToolset()
    toolset.add(_DummyTool())
    toolset.add(_DummyToolB())
    args = '{"value":"x"}'
    llm = LLM(
        chat_provider=_RepeatChatProvider([("ToolA", args), ("ToolB", args)]),
        max_context_size=100_000,
        capabilities=set(),
    )
    soul = _make_soul(runtime, llm, toolset, tmp_path)

    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _msg: None)

    async def _noop_checkpoint() -> None:
        return None

    monkeypatch.setattr(soul, "_checkpoint", _noop_checkpoint)
    monkeypatch.setattr(soul._denwa_renji, "set_n_checkpoints", lambda _n: None)

    outcome = await soul._turn(Message(role="user", content="go"))

    assert outcome.stop_reason == "tool_call_repeat"
    # ToolA replays its identical call on steps 1,3,5,7 -> cycle stop at 2*n-1
    expected_step = 2 * _CYCLE_FORCE_STOP - 1
    assert outcome.step_count == expected_step
    # strictly below every other ceiling
    assert expected_step < _TURN_TOOL_CALL_HARD_STOP
    assert expected_step // 2 < _DIFF_ARGS_HARD_STOP_START

