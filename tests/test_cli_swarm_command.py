"""Tests for the /swarm CLI command."""
from __future__ import annotations

from types import SimpleNamespace

from kimix.cli_impl import commands
from kimix.utils.system_prompt import SystemPromptType


def test_swarm_command_cancelled(monkeypatch, capsys):
    monkeypatch.setattr(commands, "_input", lambda _prompt, _arr: "/cancel")
    commands._cmd_swarm(["swarm"], [])
    output = capsys.readouterr().out
    # The prompt banner contains the word "cancel", so just ensure the empty
    # input warning is not printed (i.e. the command aborted cleanly).
    assert "No input provided for swarm." not in output


def test_swarm_command_empty_input(monkeypatch, capsys):
    monkeypatch.setattr(commands, "_input", lambda _prompt, _arr: "/end")
    commands._cmd_swarm(["swarm"], [])
    assert "No input provided for swarm." in capsys.readouterr().out


def test_swarm_command_creates_session_and_prompts(monkeypatch, capsys):
    inputs = ["line 1", "line 2", "/end"]
    monkeypatch.setattr(
        commands, "_input", lambda _prompt, _arr: inputs.pop(0)
    )

    fake_session = SimpleNamespace(id="swarm-session", custom_data={})
    created = {"session": fake_session}
    prompt_calls: list[str] = []
    closed: list[SimpleNamespace] = []

    def fake_create_session(*, agent_file, agent_type, custom_data=None, **kwargs):
        created["session"] = fake_session
        created["agent_type"] = agent_type
        created["custom_data"] = custom_data
        return fake_session

    def fake_prompt(prompt_str, session=None, **kwargs):
        prompt_calls.append(prompt_str)

    def fake_close_session(session):
        closed.append(session)

    monkeypatch.setattr(commands, "create_session", fake_create_session)
    monkeypatch.setattr(commands, "prompt", fake_prompt)
    monkeypatch.setattr(commands, "close_session", fake_close_session)

    commands._cmd_swarm(["swarm"], [])

    assert created["session"] is fake_session
    assert created["agent_type"] is SystemPromptType.SwarmLeader
    assert created["custom_data"] == {"is_swarm_session": True}
    assert len(prompt_calls) == 1
    # The orchestration instructions are now part of the SwarmLeader system
    # prompt, so the user prompt is sent unchanged.
    assert prompt_calls[0] == "line 1\nline 2"
    assert closed == [fake_session]


def test_swarm_command_passes_custom_data(monkeypatch):
    inputs = ["task", "/end"]
    monkeypatch.setattr(
        commands, "_input", lambda _prompt, _arr: inputs.pop(0)
    )

    captured: dict[str, object] = {}

    def fake_create_session(*, agent_file, agent_type, custom_data=None, **kwargs):
        captured["agent_type"] = agent_type
        captured["custom_data"] = custom_data
        return SimpleNamespace(id="s", custom_data={})

    monkeypatch.setattr(commands, "create_session", fake_create_session)
    monkeypatch.setattr(commands, "prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(commands, "close_session", lambda _s: None)

    commands._cmd_swarm(["swarm"], [])
    assert captured["agent_type"] is SystemPromptType.SwarmLeader
    assert captured["custom_data"] == {"is_swarm_session": True}
