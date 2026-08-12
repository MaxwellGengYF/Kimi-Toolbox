from __future__ import annotations

from types import SimpleNamespace

import kimix.cli_impl.commands as commands


def _fake_session() -> SimpleNamespace:
    """A minimal session stub with a soul context usable by the reflection builder."""
    return SimpleNamespace(
        _cli=SimpleNamespace(
            soul=SimpleNamespace(
                context=SimpleNamespace(history=[], token_count=0),
            ),
        ),
        status=SimpleNamespace(context_usage=0.5, context_tokens=100),
    )


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def test_build_reflection_prompt_includes_architecture_map():
    prompt = commands._build_reflection_prompt(_fake_session())
    norm = _norm(prompt)

    # Exact source paths the reflection task must know about.
    assert "src/kimix/utils/system_prompt.py" in norm
    assert "src/kimix/agent_worker.json" in norm
    assert "kimi-cli/src/kimi_cli/soul" in norm
    assert "kimi-cli/src/kimi_cli/soul/dynamic_injections" in norm
    assert "kimi-cli/src/kimi_cli/config.py" in norm

    # System prompt summary.
    assert "systemprompttype" in norm
    assert "get_system_prompt()" in prompt
    assert "max_system_prompt_tokens" in prompt

    # Tool manifest summary lists the worker tools.
    for tool in (
        "Bash",
        "Powershell",
        "Run",
        "Python",
        "TaskOutput",
        "TodoList",
        "Retrieve",
        "ReadFile",
        "EditFile",
        "WriteFile",
        "Agent",
        "AgentSwarm",
        "Glob",
        "Grep",
        "FetchURL",
        "SearchWeb",
        "ContextUsage",
        "Compact",
    ):
        assert tool in prompt

    # Soul runtime files.
    for soul_file in (
        "agent.py",
        "kimisoul.py",
        "compaction.py",
        "context.py",
        "context_db.py",
        "context_pruning.py",
        "message.py",
        "dynamic_injection.py",
        "verification_gate.py",
        "approval.py",
        "btw.py",
        "steer.py",
        "slash.py",
        "toolset.py",
        "tool_taxonomy.py",
    ):
        assert soul_file in prompt

    # Dynamic injectors (reminders).
    for injector in (
        "budget_reminder.py",
        "compact_reminder.py",
        "context_meter.py",
        "target_churn.py",
        "todo_reminder.py",
    ):
        assert injector in prompt

    # Config summary.
    assert "loopcontrol" in norm
    assert "config" in prompt.lower()


def test_build_reflection_prompt_embeds_agents_md():
    prompt = commands._build_reflection_prompt(_fake_session())
    # AGENTS.md is read from the repo and embedded verbatim.
    assert "## AGENTS.md" in prompt
    assert "syntax_check.py" in prompt
    assert "git_diff.py" in prompt


def test_build_reflection_prompt_default_report_path():
    prompt = commands._build_reflection_prompt(_fake_session())
    assert "docs/reflection_report_" in _norm(prompt)


def test_cmd_reflection_requires_active_session(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_default_session", lambda: None)
    result = commands._cmd_reflection(["reflection"], [])
    assert result == (None, False)
    assert "No active session" in capsys.readouterr().out


def test_cmd_reflection_requires_non_empty_context(monkeypatch, capsys):
    empty = SimpleNamespace(
        _cli=SimpleNamespace(
            soul=SimpleNamespace(
                context=SimpleNamespace(history=[], token_count=0),
            ),
        ),
        status=SimpleNamespace(context_usage=0.0, context_tokens=0),
    )
    monkeypatch.setattr(commands, "get_default_session", lambda: empty)
    result = commands._cmd_reflection(["reflection"], [])
    assert result == (None, False)
    assert "Context is empty" in capsys.readouterr().out
