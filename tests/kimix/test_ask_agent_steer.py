"""E2E test: ``AskAgent`` delivers a message through the real ``Steer`` API.

The parent agent runs with a dummy (sleepy) chat provider. While the parent is
mid-turn (streaming text), a sub-agent calls ``AskAgent``; the tool resolves the
parent session from ``parent_session_id``, pushes the question via
``kimi_cli.soul.steer.Steer``, and the parent's running loop is interrupted
mid-stream. The question lands in the parent's context as a follow-up user
message and a fresh step answers it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, LLM
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire
from kimi_cli.wire.types import SteerInput, StepBegin, StepInterrupted, TurnEnd
from kosong.chat_provider.mock import MockChatProvider
from kosong.message import ContentPart, Message, TextPart
from kosong.tooling.empty import EmptyToolset

from kimix.tools.agent import (
    AskAgent,
    AskAgentParams,
    _register_agent_session,
    _unregister_agent_session,
)


class _SleepyStreamedMessage:
    """Streams parts with a sleep gap so a steer can interrupt mid-stream."""

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


def _make_runtime(tmp_path: Path, provider: object) -> Runtime:
    """Build a real ``Runtime`` with a dummy chat provider (mirrors the
    ``runtime`` fixture in ``kimi-cli/tests/conftest.py``)."""
    from kaos.path import KaosPath
    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.background import BackgroundTaskManager
    from kimi_cli.config import get_default_config
    from kimi_cli.metadata import WorkDirMeta
    from kimi_cli.notifications import NotificationManager
    from kimi_cli.session import Session
    from kimi_cli.session_state import SessionState
    from kimi_cli.soul.agent import BuiltinSystemPromptArgs, LaborMarket
    from kimi_cli.soul.approval import Approval
    from kimi_cli.soul.denwarenji import DenwaRenji
    from kimi_cli.utils.environment import Environment
    from kimi_cli.wire.file import WireFile

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    share_dir = tmp_path / "share"
    share_dir.mkdir(parents=True, exist_ok=True)

    work_kaos = KaosPath.unsafe_from_local_path(work_dir)
    config = get_default_config()
    config.loop_control.context_meter_enabled = False
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=ALL_MODEL_CAPABILITIES,
    )
    builtin_args = BuiltinSystemPromptArgs(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=work_kaos,
        KIMI_WORK_DIR_LS="Test ls content",
        KIMI_AGENTS_MD="Test agents content",
        KIMI_SKILLS="No skills found.",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS="Windows",
        KIMI_SHELL="pwsh",
    )
    session = Session(
        id="parent-cli",
        work_dir=work_kaos,
        work_dir_meta=WorkDirMeta(path=str(work_kaos), kaos="local"),
        context_file=share_dir / "context.jsonl",
        wire_file=WireFile(path=share_dir / "wire.jsonl"),
        state=SessionState(),
        title="Test Session",
        updated_at=0.0,
        custom_data={},
        custom_config={},
    )
    notifications = NotificationManager(share_dir / "notifications", config.notifications)
    return Runtime(
        config=config,
        llm=llm,
        builtin_args=builtin_args,
        denwa_renji=DenwaRenji(),
        session=session,
        approval=Approval(yolo=False),
        labor_market=LaborMarket(),
        environment=Environment(
            os_kind="Windows",
            os_arch="x86_64",
            os_version="1.0",
            shell_name="pwsh",
            shell_path=KaosPath(r"C:\Program Files\Git\bin\bash.exe"),
        ),
        notifications=notifications,
        background_tasks=BackgroundTaskManager(
            session, config.background, notifications=notifications
        ),
        skills={},
        oauth=OAuthManager(config),
        additional_dirs=[],
        skills_dirs=[],
        role="root",
    )


@pytest.mark.asyncio
async def test_ask_agent_steers_parent_mid_stream(tmp_path: Path) -> None:
    runtime = _make_runtime(
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
    parent_agent = Agent(
        name="Parent Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    parent_soul = KimiSoul(
        parent_agent, context=Context(file_backend=tmp_path / "context.jsonl")
    )

    parent_id = "parent-1"
    parent_cli_session = SimpleNamespace(id=parent_id)
    parent_sdk = SimpleNamespace(
        _cli=SimpleNamespace(soul=parent_soul, session=parent_cli_session)
    )
    _register_agent_session(parent_id, parent_sdk)

    # The sub-agent's AskAgent tool instance: a sub-agent whose parent is
    # ``parent-1``. Its own loop need not be running for this test — only the
    # parent must be running so the steer can interrupt it.
    child_session = SimpleNamespace(
        id="child-1",
        custom_config={"is_sub_agent": True, "parent_session_id": parent_id},
        custom_data={},
    )

    seen: list[object] = []
    ask_result: object = None

    try:
        async def ui_loop(wire: Wire) -> None:
            nonlocal ask_result
            wire_ui = wire.ui_side(merge=False)
            while True:
                try:
                    msg = await wire_ui.receive()
                except QueueShutDown:
                    return
                seen.append(msg)
                if ask_result is None and msg == TextPart(text="partial-1"):
                    # The sub-agent asks its parent while the parent is
                    # mid-stream printing text.
                    ask_agent = AskAgent(child_session)
                    ask_result = await ask_agent(
                        AskAgentParams(question="What format do you want?")
                    )

        await run_soul(parent_soul, "original question", ui_loop, asyncio.Event())
    finally:
        _unregister_agent_session(parent_id)

    assert ask_result is not None and not ask_result.is_error
    assert "parent-1" in ask_result.output
    assert "delivered" in ask_result.output

    # The question was injected into the parent's context as a follow-up user
    # message (with the sender attributed); the interrupted step's partial
    # output was dropped; a fresh step answered the question.
    assert parent_soul.context.history == [
        Message(role="user", content=[TextPart(text="original question")]),
        Message(
            role="user",
            content=[TextPart(text="Message from agent 'child-1':\nWhat format do you want?")],
        ),
        Message(role="assistant", content=[TextPart(text="second answer")]),
    ]
    assert [msg for msg in seen if isinstance(msg, SteerInput)] == [
        SteerInput(user_input="Message from agent 'child-1':\nWhat format do you want?")
    ]
    assert [msg for msg in seen if isinstance(msg, StepBegin)] == [
        StepBegin(n=1),
        StepBegin(n=2),
    ]
    assert [msg for msg in seen if isinstance(msg, StepInterrupted)] == [StepInterrupted()]
    assert TextPart(text="partial-1") in seen
    assert TextPart(text="partial-2") not in seen
    assert isinstance(seen[-1], TurnEnd)


@pytest.mark.asyncio
async def test_ask_agent_uses_mock_provider_registry_resolution(tmp_path: Path) -> None:
    """AskAgent resolution also works with the kosong ``MockChatProvider``."""
    runtime = _make_runtime(tmp_path, MockChatProvider([TextPart(text="mock answer")]))
    parent_agent = Agent(
        name="Parent Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    parent_soul = KimiSoul(
        parent_agent, context=Context(file_backend=tmp_path / "context.jsonl")
    )
    parent_id = "parent-2"
    parent_sdk = SimpleNamespace(
        _cli=SimpleNamespace(soul=parent_soul, session=SimpleNamespace(id=parent_id))
    )
    _register_agent_session(parent_id, parent_sdk)
    child_session = SimpleNamespace(
        id="child-2",
        custom_config={"is_sub_agent": True, "parent_session_id": parent_id},
        custom_data={},
    )
    seen: list[object] = []
    ask_result: object = None

    try:
        async def ui_loop(wire: Wire) -> None:
            nonlocal ask_result
            wire_ui = wire.ui_side(merge=False)
            while True:
                try:
                    msg = await wire_ui.receive()
                except QueueShutDown:
                    return
                seen.append(msg)
                if ask_result is None and msg == TextPart(text="mock answer"):
                    ask_agent = AskAgent(child_session)
                    ask_result = await ask_agent(AskAgentParams(question="hello?"))
                    # Stop the run quickly after the message is delivered.
                    return

        await run_soul(parent_soul, "original question", ui_loop, asyncio.Event())
    finally:
        _unregister_agent_session(parent_id)

    assert ask_result is not None and not ask_result.is_error
    assert any(
        m.role == "user"
        and m.content == [TextPart(text="Message from agent 'child-2':\nhello?")]
        for m in parent_soul.context.history
    )
