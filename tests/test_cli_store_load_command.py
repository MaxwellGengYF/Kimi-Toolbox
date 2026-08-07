from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kaos.path import KaosPath
from kimi_cli.session import Session as CliSession
from kimi_cli.wire.file import WireFileMetadata, WireMessageRecord
from kimi_cli.wire.protocol import WIRE_PROTOCOL_VERSION
from kimi_cli.wire.types import TextPart, TurnBegin
from kosong.message import Message

from kimix.cli_impl import commands


def _make_globals(
    _default_session=None,
    _default_role=None,
) -> SimpleNamespace:
    """Create a SimpleNamespace mock for commands._globals."""
    return SimpleNamespace(
        _default_session=_default_session,
        _default_role=_default_role,
        _cli_sessions={},
        _add_cli_session=lambda *args, **kwargs: None,
    )


def _fake_session(
    session_id: str = "current",
    work_dir: KaosPath | None = None,
    anonymous: bool = False,
    empty: bool = True,
):
    work_dir = work_dir or KaosPath(".")
    token_count = 0 if empty else 123
    return SimpleNamespace(
        id=session_id,
        _anonymous=anonymous,
        _closed=False,
        _cancel_event=None,
        _cleanup_tools=AsyncMock(),
        _close_chat_provider=AsyncMock(),
        status=SimpleNamespace(
            context_usage=0.0,
            context_tokens=token_count,
        ),
        _cli=SimpleNamespace(
            session=SimpleNamespace(
                work_dir=work_dir,
                id=session_id,
                title="Test Session",
                updated_at=time.time(),
                close_context_db=AsyncMock(),
                is_empty=lambda: empty,
            ),
            soul=SimpleNamespace(
                context=SimpleNamespace(token_count=token_count)
            ),
        ),
    )


def _fake_new_session(session_id: str = "current"):
    return _fake_session(session_id=session_id)


@pytest.fixture
def isolated_cli_share_dir(monkeypatch, tmp_path: Path) -> Path:
    share_dir = tmp_path / "share"
    share_dir.mkdir()

    def _get_share_dir() -> Path:
        share_dir.mkdir(parents=True, exist_ok=True)
        return share_dir

    monkeypatch.setattr("kimi_cli.share.get_share_dir", _get_share_dir)
    monkeypatch.setattr("kimi_cli.metadata.get_share_dir", _get_share_dir)
    return share_dir


# ---------------------------------------------------------------------------
# Registration / help
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /store unit tests
# ---------------------------------------------------------------------------


def test_store_command_requires_argument(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_default_session", lambda: _fake_session())
    commands._cmd_store(["store"], [])
    out = capsys.readouterr().out
    assert "Command must be /store:session_id" in out


def test_store_command_rejects_same_id(monkeypatch, capsys):
    session = _fake_session(session_id="current")
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    commands._cmd_store(["store", "current"], [])
    out = capsys.readouterr().out
    assert "Target session name must be different" in out


def test_store_command_rejects_existing_target_and_resumes_original(monkeypatch, capsys):
    session = _fake_session(session_id="current", anonymous=False)
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(side_effect=ValueError("Target session already exists")),
    )

    fake_new = _fake_new_session("current")
    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return fake_new

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(commands, "_globals", _make_globals())

    commands._cmd_store(["store", "backup"], [])

    out = capsys.readouterr().out
    assert "Target session already exists" in out
    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "current"
    assert created_calls[0]["resume"] is True
    assert created_calls[0]["anonymous"] is False
    assert commands._globals._default_session is fake_new


def test_store_command_resumes_original_on_copy_failure(monkeypatch, capsys):
    session = _fake_session(session_id="current", anonymous=True)
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(side_effect=RuntimeError("Disk full")),
    )

    fake_new = _fake_new_session("current")
    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return fake_new

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_store(["store", "backup"], [])

    out = capsys.readouterr().out
    assert "Disk full" in out
    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "current"
    assert created_calls[0]["anonymous"] is True
    assert commands._globals._default_session is fake_new


def test_store_command_preserves_anonymous_flag(monkeypatch, capsys):
    work_dir = KaosPath(".")
    session = _fake_session(session_id="current", work_dir=work_dir, anonymous=True)
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="backup")),
    )

    fake_new = _fake_new_session("current")
    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return fake_new

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_store(["store", "backup"], [])

    out = capsys.readouterr().out
    assert "Session stored as backup" in out
    assert len(created_calls) == 1
    assert created_calls[0]["anonymous"] is True
    assert commands._globals._default_session is fake_new


def test_store_command_marks_old_session_closed(monkeypatch, capsys):
    session = _fake_session(session_id="current")
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="backup")),
    )
    monkeypatch.setattr(commands, "create_session", lambda **kw: _fake_new_session("current"))
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_store(["store", "backup"], [])

    assert session._closed is True


def test_store_command_success_updates_default_session(monkeypatch, capsys):
    session = _fake_session(session_id="current")
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="backup")),
    )

    fake_new = _fake_new_session("current")
    monkeypatch.setattr(commands, "create_session", lambda **kw: fake_new)
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_store(["store", "backup"], [])

    out = capsys.readouterr().out
    assert "Session stored as backup" in out
    assert commands._globals._default_session is fake_new


# ---------------------------------------------------------------------------
# /load unit tests
# ---------------------------------------------------------------------------


def test_load_command_requires_argument(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_default_session", lambda: None)
    commands._cmd_load(["load"], [])
    out = capsys.readouterr().out
    assert "Command must be /load:session_id" in out


def test_load_command_reports_missing_source(monkeypatch, capsys):
    current = _fake_session(session_id="current")
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(side_effect=ValueError("Source session not found")),
    )

    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))

    commands._cmd_load(["load", "missing"], [])

    out = capsys.readouterr().out
    assert "Source session not found" in out
    assert len(closed_sessions) == 0


def test_load_command_success_creates_anonymous_copy(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_default_session", lambda: None)
    monkeypatch.setattr(commands.uuid, "uuid4", lambda: SimpleNamespace(hex="anon123"))
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="anon123")),
    )

    fake_new = _fake_new_session("anon123")
    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return fake_new

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(commands, "_globals", _make_globals())

    commands._cmd_load(["load", "saved"], [])

    out = capsys.readouterr().out
    assert "Loaded session saved into anonymous session anon123" in out
    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "anon123"
    assert created_calls[0]["anonymous"] is True
    assert commands._globals._default_session is fake_new


def test_load_command_closes_current_only_after_successful_copy(monkeypatch, capsys):
    current = _fake_session(session_id="current")
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(side_effect=ValueError("Target session already exists")),
    )

    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))

    commands._cmd_load(["load", "saved"], [])

    out = capsys.readouterr().out
    assert "Target session already exists" in out
    assert len(closed_sessions) == 0


def test_load_command_warns_and_cancels_on_non_empty_current(monkeypatch, capsys):
    current = _fake_session(session_id="current", empty=False)
    monkeypatch.setattr(commands, "get_default_session", lambda: current)

    copy_mock = AsyncMock(return_value=SimpleNamespace(id="anon123"))
    monkeypatch.setattr("kimi_cli.session.Session.copy", copy_mock)

    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))
    monkeypatch.setattr(
        commands, "_globals", _make_globals(_default_session=current)
    )

    commands._cmd_load(["load", "saved"], ["n"])

    out = capsys.readouterr().out
    assert "has 123 context tokens" in out
    assert "Load cancelled" in out
    copy_mock.assert_not_awaited()
    assert len(closed_sessions) == 0
    assert commands._globals._default_session is current


def test_load_command_warns_and_confirms_on_non_empty_current(monkeypatch, capsys):
    current = _fake_session(session_id="current", empty=False)
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr(commands.uuid, "uuid4", lambda: SimpleNamespace(hex="anon123"))
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="anon123")),
    )

    fake_new = _fake_new_session("anon123")
    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return fake_new

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(
        commands, "_globals", _make_globals(_default_session=current)
    )

    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))

    commands._cmd_load(["load", "saved"], ["y"])

    out = capsys.readouterr().out
    assert "has 123 context tokens" in out
    assert "Loaded session saved into anonymous session anon123" in out
    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "anon123"
    assert created_calls[0]["anonymous"] is True
    assert len(closed_sessions) == 1
    assert closed_sessions[0] is current
    assert commands._globals._default_session is fake_new


def test_load_command_rejects_invalid_confirmation_input(monkeypatch, capsys):
    current = _fake_session(session_id="current", empty=False)
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr(commands.uuid, "uuid4", lambda: SimpleNamespace(hex="anon123"))
    monkeypatch.setattr(
        "kimi_cli.session.Session.copy",
        AsyncMock(return_value=SimpleNamespace(id="anon123")),
    )

    fake_new = _fake_new_session("anon123")
    monkeypatch.setattr(commands, "create_session", lambda **kw: fake_new)
    monkeypatch.setattr(
        commands, "_globals", _make_globals(_default_session=current)
    )
    monkeypatch.setattr(commands, "close_session", lambda s: None)

    commands._cmd_load(["load", "saved"], ["maybe", "", "y"])

    out = capsys.readouterr().out
    assert "Please enter y or n." in out
    assert "Loaded session saved into anonymous session anon123" in out
    assert commands._globals._default_session is fake_new


# ---------------------------------------------------------------------------
# /store integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def work_dir(tmp_path: Path) -> KaosPath:
    path = tmp_path / "work"
    path.mkdir()
    return KaosPath.unsafe_from_local_path(path)


def test_store_command_copies_session_on_disk(
    isolated_cli_share_dir: Path, work_dir: KaosPath, monkeypatch, capsys
):
    cli_session = asyncio.run(CliSession.create(work_dir, "current"))

    wire_file = cli_session.dir / "wire.jsonl"
    metadata = WireFileMetadata(protocol_version=WIRE_PROTOCOL_VERSION)
    record = WireMessageRecord.from_wire_message(
        TurnBegin(user_input=[TextPart(text="store integration")]),
        timestamp=time.time(),
    )
    with wire_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(metadata.model_dump(mode="json")) + "\n")
        f.write(json.dumps(record.model_dump(mode="json")) + "\n")

    session = _fake_session(session_id="current", work_dir=work_dir, anonymous=False)
    session._cli.session = cli_session
    monkeypatch.setattr(commands, "get_default_session", lambda: session)

    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return _fake_new_session(kw.get("session_id", "current"))

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_store(["store", "backup"], [])

    backup = asyncio.run(CliSession.find(work_dir, "backup"))
    assert backup is not None
    assert backup.context_file.exists()
    assert (backup.dir / "wire.jsonl").exists()

    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "current"
    assert created_calls[0]["anonymous"] is False


# ---------------------------------------------------------------------------
# /load integration tests
# ---------------------------------------------------------------------------


def test_load_command_copies_named_session_to_anonymous(
    isolated_cli_share_dir: Path, work_dir: KaosPath, monkeypatch, capsys
):
    source = asyncio.run(CliSession.create(work_dir, "saved"))

    context_file = source.context_file
    message = Message(role="user", content=[TextPart(text="hello from saved")])
    context_file.write_text(message.model_dump_json(exclude_none=True) + "\n", encoding="utf-8")

    current = _fake_session(session_id="other", work_dir=work_dir, anonymous=False)
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))
    monkeypatch.setattr(commands.uuid, "uuid4", lambda: SimpleNamespace(hex="anonload1"))

    created_calls = []

    def _create_session(**kw):
        created_calls.append(kw)
        return _fake_new_session(kw.get("session_id", "anonload1"))

    monkeypatch.setattr(commands, "create_session", _create_session)
    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )

    commands._cmd_load(["load", "saved"], [])

    loaded = asyncio.run(CliSession.find(work_dir, "anonload1"))
    assert loaded is not None
    assert asyncio.run(CliSession.find(work_dir, "saved")) is not None

    assert len(created_calls) == 1
    assert created_calls[0]["session_id"] == "anonload1"
    assert created_calls[0]["anonymous"] is True
    assert len(closed_sessions) == 1
    assert closed_sessions[0] is current


def test_load_command_from_current_session_preserves_source(
    isolated_cli_share_dir: Path, work_dir: KaosPath, monkeypatch, capsys
):
    cli_session = asyncio.run(CliSession.create(work_dir, "saved"))

    session = _fake_session(session_id="saved", work_dir=work_dir, anonymous=False)
    session._cli.session = cli_session
    monkeypatch.setattr(commands, "get_default_session", lambda: session)
    monkeypatch.setattr(commands.uuid, "uuid4", lambda: SimpleNamespace(hex="anonload2"))

    closed_sessions = []
    monkeypatch.setattr(commands, "close_session", lambda s: closed_sessions.append(s))

    monkeypatch.setattr(
        commands, "_globals", _make_globals()
    )
    monkeypatch.setattr(
        commands, "create_session", lambda **kw: _fake_new_session(kw.get("session_id", "anonload2"))
    )

    commands._cmd_load(["load", "saved"], [])

    assert asyncio.run(CliSession.find(work_dir, "saved")) is not None
    assert len(closed_sessions) == 1
    assert closed_sessions[0] is session
