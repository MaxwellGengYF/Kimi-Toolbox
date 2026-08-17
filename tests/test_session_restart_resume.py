"""Tests that an auto-restarted Session resumes the prior conversation context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from kaos.path import KaosPath
from kimi_cli.soul import SessionRestartRequired
from kimi_cli.wire.types import TextPart
from kosong.chat_provider import APIStatusError

from kimi_agent_sdk._session import Session


class _FakeChatProvider:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class _FakeLLM:
    chat_provider: _FakeChatProvider


@dataclass
class _FakeRuntime:
    llm: _FakeLLM | None


@dataclass
class _FakeAgent:
    toolset: Any = None


@dataclass
class _FakeSoul:
    agent: _FakeAgent
    _runtime: _FakeRuntime

    async def close(self) -> None:
        pass


@dataclass
class _FakeWireFile:
    path: Path


@dataclass
class _FakeCLISession:
    work_dir: KaosPath
    session_id: str
    context_file: Path
    dir: Path
    wire_file: _FakeWireFile
    custom_data: dict[str, Any] = field(default_factory=dict)
    custom_config: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.session_id

    async def close_context_db(self) -> None:
        pass


class _FakeCLI:
    """Fake CLI whose first run raises SessionRestartRequired and second succeeds."""

    calls: int = 0

    def __init__(self, session: _FakeCLISession) -> None:
        self.session = session
        self.soul = _FakeSoul(
            agent=_FakeAgent(),
            _runtime=_FakeRuntime(llm=_FakeLLM(chat_provider=_FakeChatProvider())),
        )

    async def run(
        self,
        user_input: Any,
        cancel_event: Any,
        *,
        merge_wire_messages: bool = False,
    ) -> Any:
        if _FakeCLI.calls == 0:
            _FakeCLI.calls += 1
            raise SessionRestartRequired(
                "simulated failure",
                original_error=APIStatusError(500, "Internal Server Error"),
            )
        _FakeCLI.calls += 1
        yield TextPart(text="success after restart")


def _make_session(tmp_path: Path) -> tuple[Session, _FakeCLISession, Path, Path]:
    work_dir = KaosPath.unsafe_from_local_path(tmp_path)
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    context_file = session_dir / "context.jsonl"
    context_file.write_text('{"role": "user", "content": "previous turn"}\n', encoding="utf-8")
    state_file = session_dir / "state.json"
    state_file.write_text("{}", encoding="utf-8")
    wire_file_path = session_dir / "wire.jsonl"
    cli_session = _FakeCLISession(
        work_dir=work_dir,
        session_id="s1",
        context_file=context_file,
        dir=session_dir,
        wire_file=_FakeWireFile(wire_file_path),
        custom_data={"key": "value"},
        custom_config={},
    )
    cli = _FakeCLI(cli_session)
    session = Session(cli)
    session._create_kwargs = {"yolo": True}
    return session, cli_session, context_file, state_file


@pytest.mark.asyncio
async def test_prompt_restart_preserves_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a SessionRestartRequired the session must resume, not start fresh."""
    _FakeCLI.calls = 0
    session, _old_cli_session, context_file, state_file = _make_session(tmp_path)

    import kimi_agent_sdk._session as session_mod

    find_called = False
    create_called = False
    captured_kwargs: dict[str, Any] = {}

    async def fake_find(work_dir: KaosPath, session_id: str) -> _FakeCLISession:
        nonlocal find_called
        find_called = True
        return _old_cli_session

    async def fake_create(cli_session: _FakeCLISession, **kwargs: Any) -> _FakeCLI:
        nonlocal create_called
        create_called = True
        captured_kwargs.update(kwargs)
        return _FakeCLI(cli_session)

    monkeypatch.setattr(session_mod.CliSession, "find", classmethod(lambda cls, *a, **kw: fake_find(*a, **kw)))
    monkeypatch.setattr(session_mod.CliSession, "create", classmethod(lambda cls, *a, **kw: fake_create(a[1] if len(a) > 1 else None, **kw)))
    monkeypatch.setattr(session_mod.KimiCLI, "create", classmethod(lambda cls, *a, **kw: fake_create(a[0] if a else None, **kw)))

    messages: list[Any] = []
    async for msg in session.prompt("hello", max_restarts=1):
        messages.append(msg)

    # Prior context/state must survive the restart.
    assert context_file.exists(), "context.jsonl was deleted during restart"
    assert "previous turn" in context_file.read_text(encoding="utf-8")
    assert state_file.exists(), "state.json was deleted during restart"

    # The fixed implementation must reload the existing session, not create a new one.
    assert find_called, "CliSession.find should be used to resume the existing session"

    # The new CLI must be told that it is resuming an existing session.
    assert captured_kwargs.get("resumed") is True

    # Custom data attached to the old session must be carried over.
    assert session._cli.session.custom_data.get("key") == "value"

    texts = [m.text for m in messages if isinstance(m, TextPart)]
    assert any("Connection lost" in t for t in texts)
    assert any("success after restart" in t for t in texts)
