from __future__ import annotations

from types import SimpleNamespace

import kimix.utils._globals as _globals
from kimix.cli_impl import commands


def test_sessions_command_prints_empty_state(monkeypatch, capsys):
    # Clear the global sessions cache to simulate no sessions
    _globals._cli_sessions.clear()
    monkeypatch.setattr(commands, "get_default_session", lambda: None)

    commands._cmd_sessions(["sessions"], [])

    assert "No sessions found." in capsys.readouterr().out


def test_sessions_command_prints_sessions_in_returned_order(monkeypatch, capsys):
    # Populate the global sessions cache
    _globals._cli_sessions.clear()
    _globals._add_cli_session(
        "s1", "older title", 1_700_000_000.0,
        context_usage=0.3, context_tokens=1500,
    )
    _globals._add_cli_session(
        "s2", "current title", 1_700_000_100.0,
        context_usage=0.5, context_tokens=2500,
    )

    current = SimpleNamespace(
        _cli=SimpleNamespace(session=SimpleNamespace(id="s2")),
        status=SimpleNamespace(context_usage=0.5, context_tokens=2500),
    )
    monkeypatch.setattr(commands, "get_default_session", lambda: current)

    commands._cmd_sessions(["sessions"], [])

    output = capsys.readouterr().out
    assert "session id" in output
    assert "context usage" in output
    assert "*  s2" in output
    assert "   s1" in output
    assert "50.0%" in output
    assert "30.0%" in output
    assert output.index("s2") < output.index("s1")
