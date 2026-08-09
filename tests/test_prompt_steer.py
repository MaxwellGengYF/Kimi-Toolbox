"""Tests for ``kimix.utils.prompt.steer_session`` / ``steer_session_sync``.

Covers:
  - returning False when the session has no resolvable soul
  - pushing into a real ``KimiSoul`` running with a dummy provider (async)
  - pushing from another thread via the sync wrapper
"""
from __future__ import annotations

import asyncio
import threading
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
from kimi_cli.wire.types import TurnEnd
from kosong.message import ContentPart, Message, TextPart
from kosong.tooling.empty import EmptyToolset

from kimix.utils.prompt import steer_session, steer_session_sync


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
    from kimi_cli.soul.agent import BuiltinSystemPromptArgs, LaborMarket, Runtime
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
        id="test",
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


def _make_soul(tmp_path: Path, runtime: Runtime) -> KimiSoul:
    agent = Agent(
        name="Steer Test Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "context.jsonl"))


# ---------------------------------------------------------------------------
# No resolvable soul
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steer_session_returns_false_without_soul() -> None:
    assert await steer_session(object(), "hi") is False
    assert await steer_session(SimpleNamespace(), "hi") is False


def test_steer_session_sync_returns_false_without_soul() -> None:
    assert steer_session_sync(object(), "hi") is False
    assert steer_session_sync(SimpleNamespace(), "hi") is False


# ---------------------------------------------------------------------------
# Async push into a real running soul
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steer_session_pushes_into_running_soul(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share_env"))
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
    soul = _make_soul(tmp_path, runtime)
    session = SimpleNamespace(_cli=SimpleNamespace(soul=soul))
    seen: list[object] = []
    pushed = False

    async def ui_loop(wire: Wire) -> None:
        nonlocal pushed
        # merge=False so individual streamed parts are observable.
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            seen.append(msg)
            if not pushed and msg == TextPart(text="partial-1"):
                assert await steer_session(session, "steer now") is True
                pushed = True

    await run_soul(soul, "original question", ui_loop, asyncio.Event())

    assert pushed is True
    # The steer landed in the context and the interrupted step's partial
    # output was dropped.
    assert soul.context.history == [
        Message(role="user", content=[TextPart(text="original question")]),
        Message(role="user", content=[TextPart(text="steer now")]),
        Message(role="assistant", content=[TextPart(text="second answer")]),
    ]
    assert isinstance(seen[-1], TurnEnd)


# ---------------------------------------------------------------------------
# Sync push from another thread into a real running soul
# ---------------------------------------------------------------------------


def test_steer_session_sync_pushes_into_running_soul_from_other_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share_env"))
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
    soul = _make_soul(tmp_path, runtime)
    session = SimpleNamespace(_cli=SimpleNamespace(soul=soul))
    first_part_seen = threading.Event()
    run_done = threading.Event()
    errors: list[BaseException] = []

    async def ui_loop(wire: Wire) -> None:
        wire_ui = wire.ui_side(merge=False)
        while True:
            try:
                msg = await wire_ui.receive()
            except QueueShutDown:
                return
            if msg == TextPart(text="partial-1"):
                first_part_seen.set()

    def _run() -> None:
        try:
            asyncio.run(run_soul(soul, "original question", ui_loop, asyncio.Event()))
        except BaseException as exc:  # pragma: no cover - failure reporting only
            errors.append(exc)
        finally:
            run_done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        assert first_part_seen.wait(timeout=15), "first part never streamed"
        # The soul runs in the background thread's loop; push from THIS thread.
        assert steer_session_sync(session, "steer now") is True
    finally:
        run_done.wait(timeout=15)
        thread.join(timeout=15)

    assert errors == []
    assert any(
        m.role == "user" and m.content == [TextPart(text="steer now")]
        for m in soul.context.history
    )
