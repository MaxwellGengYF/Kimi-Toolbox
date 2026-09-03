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
    _REPEAT_FORCE_STOP_STREAK,
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
        self._histories: list[list[Message]] = []

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
        self._histories.append(list(history))
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
    """A turn replaying one identical call must NOT end silently.

    The toolset's consecutive-streak guard fires; the soul then runs bounded
    loop-recovery prompts and finally returns a synthesized plain-text answer
    that references the user requirement, never an empty tool-only stop.
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

    # The turn must end on a plain-text answer, never with an empty stop.
    assert outcome.stop_reason == "no_tool_calls"
    assert outcome.final_message is not None
    assert outcome.final_message.content
    text = outcome.final_message.extract_text(" ")
    assert "Original request: go" in text
    # The loop is caught at the consecutive-streak guard, then recovery runs.
    assert outcome.step_count >= _REPEAT_FORCE_STOP_STREAK


@pytest.mark.asyncio
async def test_turn_force_stops_on_interleaved_cycle(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``A(x) -> B(x) -> A(x) -> B(x)`` must NOT end silently.

    The consecutive streak can never grow here, and the per-tool
    different-args ceiling is not reached — only the cycle-aware punishment
    stops the loop. The soul must convert that into a recovery prompt and a
    final plain-text answer for the user requirement.
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

    assert outcome.stop_reason == "no_tool_calls"
    assert outcome.final_message is not None
    assert outcome.final_message.content
    text = outcome.final_message.extract_text(" ")
    assert "Original request: go" in text
    # ToolA replays its identical call on steps 1,3,5,7 -> cycle stop at 2*n-1,
    # then the soul runs its bounded recovery rounds.
    cycle_stop = 2 * _CYCLE_FORCE_STOP - 1
    assert outcome.step_count >= cycle_stop


@pytest.mark.asyncio
async def test_loop_recovery_prompt_reaches_llm_and_requirement_kept(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the cycle detector fires, recovery prompts must reach the LLM.

    Regression for the session-truncation symptom: the toolset's
    ``Interleaved ('cyclic') repeat detection`` used to force-stop the turn
    with no final text.  Now the soul injects plain-user recovery prompts
    that restate the top-level user requirement, and if the model keeps
    looping the turn ends on a synthesized text fallback referencing it.
    """
    toolset = KimiToolset()
    toolset.add(_DummyTool())
    toolset.add(_DummyToolB())
    args = '{"value":"x"}'
    provider = _RepeatChatProvider([("ToolA", args), ("ToolB", args)])
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    soul = _make_soul(runtime, llm, toolset, tmp_path)

    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _msg: None)

    async def _noop_checkpoint() -> None:
        return None

    monkeypatch.setattr(soul, "_checkpoint", _noop_checkpoint)
    monkeypatch.setattr(soul._denwa_renji, "set_n_checkpoints", lambda _n: None)

    outcome = await soul._turn(Message(role="user", content="deploy the service"))

    # Never ends without a top-level answer.
    assert outcome.stop_reason == "no_tool_calls"
    assert outcome.final_message is not None
    final_text = outcome.final_message.extract_text(" ")
    assert "deploy the service" in final_text

    # The recovery prompts must actually be visible to later LLM steps
    # (not stripped as system reminders).
    recovery_calls = [
        hist
        for hist in provider._histories
        if any(
            m.role == "user" and "[loop-recovery]" in m.extract_text(" ")
            for m in hist
        )
    ]
    assert recovery_calls, "No LLM call ever saw a loop-recovery prompt"
    recovery_texts = [
        m.extract_text(" ")
        for hist in recovery_calls
        for m in hist
        if m.role == "user" and "[loop-recovery]" in m.extract_text(" ")
    ]
    assert any("deploy the service" in t for t in recovery_texts)

