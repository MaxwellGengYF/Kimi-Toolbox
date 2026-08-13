"""Tests for the TargetChurnProvider (P1: target-level anti-loop)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import orjson
from kosong.message import Message, TextPart, ToolCall

from kimi_cli.soul.dynamic_injections.target_churn import TargetChurnProvider


def _soul(step: int, turn: str = "t1") -> Any:
    soul = MagicMock()
    soul._current_step_no = step
    soul._current_turn_id = turn
    return soul


def _assistant_call(tool_name: str, args: dict[str, Any]) -> Message:
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id="call-1",
                function=ToolCall.FunctionBody(
                    name=tool_name,
                    arguments=orjson.dumps(args).decode(),
                ),
            )
        ],
    )


def _tool_ok(text: str = "ok") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id="call-1")


def _tool_error(text: str) -> Message:
    return Message(
        role="tool",
        content=[TextPart(text=f"<system>ERROR: {text}</system>")],
        tool_call_id="call-1",
    )


def _edit(path: str, tool: str = "edit") -> Message:
    return _assistant_call(tool, {"path": path, "content": "x"})


async def _collect_alerts(
    provider: TargetChurnProvider,
    history: list[Message],
    steps: list[int],
    turn: str = "t1",
) -> list[str]:
    """Feed history incrementally, collecting all injection contents."""
    alerts: list[str] = []
    for end, step in enumerate(steps, start=1):
        injections = await provider.get_injections(history[:end], _soul(step, turn))
        alerts.extend(inj.content for inj in injections)
    return alerts


# ---------------------------------------------------------------------------
# 1. Same-file churn triggers exactly one reminder
# ---------------------------------------------------------------------------


async def test_same_file_six_edits_one_alert() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [_edit("a.py") for _ in range(6)]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4, 5, 6])
    assert len(alerts) == 1
    assert "a.py" in alerts[0]


async def test_below_threshold_no_alert() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [_edit("a.py") for _ in range(4)]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4])
    assert alerts == []


async def test_strong_alert_at_higher_threshold() -> None:
    provider = TargetChurnProvider(file_warn=3, file_strong=6, error_warn=3, cooldown_steps=0)
    history = [_edit("a.py") for _ in range(8)]
    alerts = await _collect_alerts(provider, history, steps=list(range(1, 9)))
    assert len(alerts) == 2
    assert "Rewrite the file as a whole" in alerts[1] or "Rewrite" in alerts[1]


# ---------------------------------------------------------------------------
# 2. Cross-tool counting (write -> edit -> shell redirects/sed)
# ---------------------------------------------------------------------------


async def test_cross_tool_counting() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [
        _edit("src/x.py", "write"),
        _edit("src/x.py", "edit"),
        _assistant_call("pwsh", {"command": "sed -i 's/a/b/' src/x.py"}),
        _assistant_call("bash", {"command": "echo line >> src/x.py"}),
        _assistant_call("pwsh", {"command": "echo y > src/x.py"}),
    ]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4, 5])
    assert len(alerts) == 1
    assert "src\\x.py" in alerts[0] or "src/x.py" in alerts[0]


# ---------------------------------------------------------------------------
# 3. Error-signature streaks
# ---------------------------------------------------------------------------


async def test_same_error_streak_triggers() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [
        _tool_error("ValueError: bad input at line 42"),
        _tool_error("ValueError: bad input at line 57"),
        _tool_error("ValueError: bad input at line 91"),
    ]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3])
    assert len(alerts) == 1
    assert "same error" in alerts[0]


async def test_different_errors_no_trigger() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [
        _tool_error("ValueError: bad input at line 42"),
        _tool_error("TypeError: wrong type at line 42"),
        _tool_error("KeyError: missing at line 42"),
    ]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3])
    assert alerts == []


async def test_success_breaks_error_streak() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=0)
    history = [
        _tool_error("ValueError: bad at line 1"),
        _tool_error("ValueError: bad at line 2"),
        _tool_ok("command succeeded"),
        _tool_error("ValueError: bad at line 3"),
        _tool_error("ValueError: bad at line 4"),
    ]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4, 5])
    assert alerts == []


# ---------------------------------------------------------------------------
# 4. Cooldown suppresses repeated alerts
# ---------------------------------------------------------------------------


async def test_cooldown_suppresses_alerts() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [_edit("a.py") for _ in range(5)] + [_edit("b.py") for _ in range(5)]
    steps = list(range(1, 11))
    alerts = await _collect_alerts(provider, history, steps=steps)
    # a.py alerts at step 5; b.py reaches threshold at step 10 — within
    # cooldown (10 - 5 < 6) -> suppressed.
    assert len(alerts) == 1
    assert "a.py" in alerts[0]

    # After cooldown expires, b.py can alert.
    more = await provider.get_injections(history, _soul(12, "t1"))
    assert len(more) == 1
    assert "b.py" in more[0].content


# ---------------------------------------------------------------------------
# 5. Compaction resets state
# ---------------------------------------------------------------------------


async def test_compaction_resets_state() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=6)
    history = [_edit("a.py") for _ in range(5)]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4, 5])
    assert len(alerts) == 1

    await provider.on_context_compacted()

    # After compaction the counter starts fresh: 5 new edits alert again.
    new_history = [_edit("a.py") for _ in range(5)]
    alerts2 = await _collect_alerts(provider, new_history, steps=[6, 7, 8, 9, 10])
    assert len(alerts2) == 1


# ---------------------------------------------------------------------------
# 6. Batch legitimate refactor does not trigger
# ---------------------------------------------------------------------------


async def test_batch_refactor_no_false_positive() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=0)
    history = [_edit(f"mod_{i}.py") for i in range(4) for _ in range(2)]
    alerts = await _collect_alerts(provider, history, steps=list(range(1, 9)))
    assert alerts == []


# ---------------------------------------------------------------------------
# 7. Turn boundary resets per-turn alert dedup
# ---------------------------------------------------------------------------


async def test_turn_boundary_resets_per_turn_state() -> None:
    provider = TargetChurnProvider(file_warn=5, file_strong=8, error_warn=3, cooldown_steps=0)
    history = [_edit("a.py") for _ in range(5)]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3, 4, 5], turn="t1")
    assert len(alerts) == 1

    # Same file keeps accumulating across turns, but the per-turn dedup
    # resets: another edit in a new turn may alert once more.
    history.append(_edit("a.py"))
    injections = await provider.get_injections(history, _soul(6, "t2"))
    assert len(injections) == 1


# ---------------------------------------------------------------------------
# Robustness: malformed args, defensive cursor reset
# ---------------------------------------------------------------------------


async def test_malformed_arguments_ignored() -> None:
    provider = TargetChurnProvider(file_warn=3, file_strong=5, error_warn=2, cooldown_steps=0)
    bad = Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id="c1",
                function=ToolCall.FunctionBody(name="edit", arguments="{not json"),
            )
        ],
    )
    history = [bad, bad, bad]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3])
    assert alerts == []


async def test_cursor_reset_on_shorter_history() -> None:
    provider = TargetChurnProvider(file_warn=3, file_strong=5, error_warn=2, cooldown_steps=0)
    history = [_edit("a.py") for _ in range(3)]
    alerts = await _collect_alerts(provider, history, steps=[1, 2, 3])
    assert len(alerts) == 1

    # History shrinks without compaction notice (e.g. revert) — provider
    # must not crash and restarts counting from scratch.
    short_history = [_edit("a.py")]
    injections = await provider.get_injections(short_history, _soul(4, "t1"))
    assert injections == []
