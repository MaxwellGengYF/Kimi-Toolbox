"""Tests for the mid-stream ``Steer`` API.

Covers:
  - ``Steer`` resolution (direct / from session with ``_cli.soul`` / ``soul`` / ``_soul`` / None)
  - ``push`` while idle returns False and queues (stale-discarded at turn init)
  - e2e mid-text interrupt (partial output dropped, steer injected, fresh step)
  - e2e mid-reasoning interrupt (interrupt fires while ThinkPart is printing)
  - e2e with ``MockChatProvider`` proving provider-agnosticism
  - external cancellation is NOT swallowed by the steer mechanism
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from kosong.chat_provider.mock import MockChatProvider
from kosong.message import ContentPart, Message, TextPart, ThinkPart
from kosong.tooling.empty import EmptyToolset

from kimi_cli.llm import LLM
from kimi_cli.soul import RunCancelled, run_soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.steer import Steer
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire
from kimi_cli.wire.types import SteerInput, StepBegin, StepInterrupted, TurnBegin, TurnEnd


@pytest.fixture
def approval() -> Approval:
    """Override global yolo=True fixture; steer tests don't need yolo."""
    return Approval(yolo=False)


def _make_soul(runtime: Runtime, tmp_path: Path) -> KimiSoul:
    agent = Agent(
        name="Steer Test Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


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
    )


def _llm_with_provider(runtime: Runtime, provider: object) -> LLM:
    assert runtime.llm is not None
    return LLM(
        chat_provider=provider,
        max_context_size=runtime.llm.max_context_size,
        capabilities=runtime.llm.capabilities,
    )


class _SleepyStreamedMessage:
    """Streams parts one by one with a sleep gap between them, so a steer can
    reliably interrupt the step mid-stream."""

    def __init__(self, parts: list[ContentPart], gap: float) -> None:
        self._parts = list(parts)
        self._gap = gap

    def __aiter__(self):
        return self

    async def __anext__(self) -> ContentPart:
        if not self._parts:
            raise StopAsyncIteration
        part = self._parts.pop(0)
        await asyncio.sleep(self._gap)
        return part

    @property
    def id(self) -> str | None:
        return "sleepy"

    @property
    def usage(self):
        return None


class _SleepyChatProvider:
    name = "sleepy"

    def __init__(self, sequences: list[list[ContentPart]], gap: float = 0.2) -> None:
        self._sequences = [list(parts) for parts in sequences]
        self._calls = 0
        self._gap = gap

    @property
    def model_name(self) -> str:
        return "sleepy"

    @property
    def thinking_effort(self):
        return None

    async def generate(self, system_prompt, tools, history):
        index = min(self._calls, len(self._sequences) - 1)
        self._calls += 1
        return _SleepyStreamedMessage(self._sequences[index], self._gap)

    def with_thinking(self, effort):
        return self


def _soul_with_provider(runtime: Runtime, tmp_path: Path, provider: object) -> KimiSoul:
    runtime.config.loop_control.context_meter_enabled = False
    agent = Agent(
        name="Steer Test Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=_runtime_with_llm(runtime, _llm_with_provider(runtime, provider)),
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_steer_resolves_directly_from_soul(runtime: Runtime, tmp_path: Path) -> None:
    soul = _make_soul(runtime, tmp_path)
    steer = Steer(soul)
    assert steer._soul is soul


def test_steer_from_session_with_cli_soul(runtime: Runtime, tmp_path: Path) -> None:
    soul = _make_soul(runtime, tmp_path)
    session = SimpleNamespace(_cli=SimpleNamespace(soul=soul))
    steer = Steer.from_session(session)
    assert steer is not None
    assert steer._soul is soul


def test_steer_from_session_with_public_soul_attr(runtime: Runtime, tmp_path: Path) -> None:
    soul = _make_soul(runtime, tmp_path)
    steer = Steer.from_session(SimpleNamespace(soul=soul))
    assert steer is not None
    assert steer._soul is soul


def test_steer_from_session_with_underscore_soul_attr(runtime: Runtime, tmp_path: Path) -> None:
    soul = _make_soul(runtime, tmp_path)
    steer = Steer.from_session(SimpleNamespace(_soul=soul))
    assert steer is not None
    assert steer._soul is soul


def test_steer_from_session_returns_none_without_soul() -> None:
    assert Steer.from_session(object()) is None
    assert Steer.from_session(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Idle push
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_while_idle_returns_false_and_queues(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _make_soul(runtime, tmp_path)
    steer = Steer(soul)

    assert soul.is_running() is False
    assert await steer.push("stale steer") is False
    assert steer.pending() == 1

    steer.clear()
    assert steer.pending() == 0

    steer.close()
    assert steer._soul is None
    assert await steer.push("after close") is False


@pytest.mark.asyncio
async def test_idle_push_is_discarded_at_turn_init(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _soul_with_provider(runtime, tmp_path, _SleepyChatProvider([[TextPart(text="answer")]]))
    steer = Steer(soul)

    # Soul is idle — push queues but reports not-delivered.
    assert await steer.push("stale steer") is False
    assert steer.pending() == 1

    seen: list[object] = []

    async def ui_loop(wire: Wire) -> None:
        wire_ui = wire.ui_side(merge=True)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)

    await run_soul(soul, "original question", ui_loop, asyncio.Event())

    # The stale steer was discarded at turn init — never injected.
    assert soul.context.history == [
        Message(role="user", content=[TextPart(text="original question")]),
        Message(role="assistant", content=[TextPart(text="answer")]),
    ]
    assert [msg for msg in seen if isinstance(msg, SteerInput)] == []


# ---------------------------------------------------------------------------
# e2e mid-stream interrupts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steer_interrupts_mid_text_stream(runtime: Runtime, tmp_path: Path) -> None:
    soul = _soul_with_provider(
        runtime,
        tmp_path,
        _SleepyChatProvider(
            [
                [
                    TextPart(text="partial-1"),
                    TextPart(text="partial-2"),
                    TextPart(text="final answer"),
                ],
                [TextPart(text="second answer")],
            ]
        ),
    )
    steer = Steer(soul)
    seen: list[object] = []
    pushed = False

    async def ui_loop(wire: Wire) -> None:
        nonlocal pushed
        # merge=False so the individual streamed parts (partial-1, partial-2,
        # ...) are observed separately — merge=True would aggregate them into
        # a single TextPart.
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)
            if not pushed and msg == TextPart(text="partial-1"):
                # The steer is pushed while the first step is still streaming
                # (partial-2 / final answer are still pending on the wire).
                assert await steer.push("steer now") is True
                pushed = True

    await run_soul(soul, "original question", ui_loop, asyncio.Event())

    assert pushed is True
    # The interrupted step's partial output is NOT grown into the context —
    # the steer becomes the next user message and a fresh step answers it.
    assert soul.context.history == [
        Message(role="user", content=[TextPart(text="original question")]),
        Message(role="user", content=[TextPart(text="steer now")]),
        Message(role="assistant", content=[TextPart(text="second answer")]),
    ]
    assert [msg for msg in seen if isinstance(msg, TurnBegin)] == [
        TurnBegin(user_input="original question")
    ]
    assert [msg for msg in seen if isinstance(msg, SteerInput)] == [
        SteerInput(user_input="steer now")
    ]
    assert [msg for msg in seen if isinstance(msg, StepBegin)] == [
        StepBegin(n=1),
        StepBegin(n=2),
    ]
    assert [msg for msg in seen if isinstance(msg, StepInterrupted)] == [StepInterrupted()]
    # The partial parts were streamed to the wire before the interrupt.
    assert TextPart(text="partial-1") in seen
    assert TextPart(text="partial-2") not in seen
    assert isinstance(seen[-1], TurnEnd)


@pytest.mark.asyncio
async def test_steer_interrupts_mid_reasoning_stream(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _soul_with_provider(
        runtime,
        tmp_path,
        _SleepyChatProvider(
            [
                [
                    ThinkPart(think="reasoning-1"),
                    ThinkPart(think="reasoning-2"),
                    TextPart(text="final answer"),
                ],
                [TextPart(text="second answer")],
            ]
        ),
    )
    steer = Steer(soul)
    seen: list[object] = []
    pushed = False

    async def ui_loop(wire: Wire) -> None:
        nonlocal pushed
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)
            if not pushed and msg == ThinkPart(think="reasoning-1"):
                # The interrupt fires while reasoning is still printing.
                assert await steer.push("steer now") is True
                pushed = True

    await run_soul(soul, "original question", ui_loop, asyncio.Event())

    assert pushed is True
    # No reasoning content lands in the context; only the completed second
    # answer does.
    assert soul.context.history == [
        Message(role="user", content=[TextPart(text="original question")]),
        Message(role="user", content=[TextPart(text="steer now")]),
        Message(role="assistant", content=[TextPart(text="second answer")]),
    ]
    assert [msg for msg in seen if isinstance(msg, SteerInput)] == [
        SteerInput(user_input="steer now")
    ]
    assert [msg for msg in seen if isinstance(msg, StepBegin)] == [
        StepBegin(n=1),
        StepBegin(n=2),
    ]
    assert [msg for msg in seen if isinstance(msg, StepInterrupted)] == [StepInterrupted()]
    assert isinstance(seen[-1], TurnEnd)


@pytest.mark.asyncio
async def test_steer_with_mock_provider_is_provider_agnostic(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _soul_with_provider(runtime, tmp_path, MockChatProvider([TextPart(text="mock answer")]))
    steer = Steer(soul)
    seen: list[object] = []
    pushed = False

    async def ui_loop(wire: Wire) -> None:
        nonlocal pushed
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)
            if not pushed and msg == TextPart(text="mock answer"):
                assert await steer.push("mock steer") is True
                pushed = True

    await run_soul(soul, "original question", ui_loop, asyncio.Event())

    assert pushed is True
    assert [msg for msg in seen if isinstance(msg, SteerInput)] == [
        SteerInput(user_input="mock steer")
    ]
    # Exactly two steps run either way: the first step is either interrupted
    # or completes and picks the steer up at the step boundary; the second
    # step answers the steer and ends the turn.
    assert [msg for msg in seen if isinstance(msg, StepBegin)] == [
        StepBegin(n=1),
        StepBegin(n=2),
    ]
    assert isinstance(seen[-1], TurnEnd)
    assert any(
        m.role == "user" and m.content == [TextPart(text="mock steer")]
        for m in soul.context.history
    )


@pytest.mark.asyncio
async def test_external_cancellation_not_swallowed_by_steer(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul = _soul_with_provider(
        runtime,
        tmp_path,
        _SleepyChatProvider(
            [
                [
                    TextPart(text="partial-1"),
                    TextPart(text="partial-2"),
                    TextPart(text="final answer"),
                ],
                [TextPart(text="second answer")],
            ]
        ),
    )
    steer = Steer(soul)
    cancel_event = asyncio.Event()
    seen: list[object] = []
    pushed = False

    async def ui_loop(wire: Wire) -> None:
        nonlocal pushed
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)
            if not pushed and msg == TextPart(text="partial-1"):
                assert await steer.push("steer now") is True
                pushed = True
                # Cancel the whole run right after pushing the steer — the
                # steer interrupt must NOT swallow the external cancellation.
                cancel_event.set()

    with pytest.raises(RunCancelled):
        await run_soul(soul, "original question", ui_loop, cancel_event)

    assert pushed is True


# ---------------------------------------------------------------------------
# Event-loop reuse across prompts
# ---------------------------------------------------------------------------


def test_soul_reuse_across_separate_event_loops(
    runtime: Runtime, tmp_path: Path
) -> None:
    """A KimiSoul reused across prompts survives fresh event loops.

    Regression test: the soul persists across prompts and each prompt may run
    in its own event loop (``kimix.utils.prompt.prompt`` wraps every prompt in
    a fresh ``asyncio.run``). ``asyncio.Event``/``Queue`` bind to the loop on
    first use, so reusing the same soul previously raised ``RuntimeError:
    <asyncio.locks.Event ...> is bound to a different event loop`` on the
    second prompt's first step.
    """
    soul = _soul_with_provider(
        runtime, tmp_path, _SleepyChatProvider([[TextPart(text="answer")]])
    )

    def _run_once(prompt: str) -> list[object]:
        seen: list[object] = []

        async def ui_loop(wire: Wire) -> None:
            wire_ui = wire.ui_side(merge=True)
            while True:
                try:
                    msg = await wire_ui.receive()
                except QueueShutDown:
                    return
                seen.append(msg)

        asyncio.run(run_soul(soul, prompt, ui_loop, asyncio.Event()))
        return seen

    first = _run_once("first question")
    assert any(isinstance(m, TurnEnd) for m in first)

    # Second prompt in a brand-new event loop must not raise the
    # "is bound to a different event loop" RuntimeError.
    second = _run_once("second question")
    assert any(isinstance(m, TurnEnd) for m in second)
