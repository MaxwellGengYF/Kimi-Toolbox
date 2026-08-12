"""Tests for the soul-layer VerificationGate (P2, B-3)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import orjson
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.verification_gate import VerificationGate


def _todo(title: str, status: str) -> Any:
    return SimpleNamespace(title=title, status=status, notes=None)


def _make_soul(
    *,
    turn_id: str = "t1",
    todos: list[Any] | None = None,
    history: list[Message] | None = None,
    todo_tool: Any = None,
) -> Any:
    soul = MagicMock()
    soul._current_turn_id = turn_id
    soul._load_todo_states_for_reminder = MagicMock(return_value=todos or [])
    soul.context.history = history or []
    if todo_tool is not None:
        soul.agent.toolset.find = MagicMock(return_value=todo_tool)
    else:
        soul.agent.toolset.find = MagicMock(side_effect=Exception("no toolset"))
    return soul


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _reminder(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=f"<system-reminder>\n{text}\n</system-reminder>")])


def _assistant_call(tool_name: str, args: dict[str, Any]) -> Message:
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id="c1",
                function=ToolCall.FunctionBody(
                    name=tool_name, arguments=orjson.dumps(args).decode()
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Condition 1: unfinished todos
# ---------------------------------------------------------------------------


async def test_unfinished_todos_block_turn() -> None:
    gate = VerificationGate(max_nudges=2)
    soul = _make_soul(todos=[_todo("task A", "pending"), _todo("task B", "done")])
    msg = await gate.check(soul)
    assert msg is not None
    assert "task A" in msg
    assert "task B" not in msg


async def test_all_todos_done_no_block() -> None:
    gate = VerificationGate(max_nudges=2)
    soul = _make_soul(todos=[_todo("task A", "done")])
    assert await gate.check(soul) is None


async def test_no_todos_no_edits_clean() -> None:
    gate = VerificationGate(max_nudges=2)
    soul = _make_soul()
    assert await gate.check(soul) is None


# ---------------------------------------------------------------------------
# Condition 2: edits without verification
# ---------------------------------------------------------------------------


async def test_edits_without_verification_block() -> None:
    gate = VerificationGate(max_nudges=2)
    history = [
        _user("implement the feature"),
        _assistant_call("WriteFile", {"path": "a.py", "content": "x"}),
        _reminder("some injected reminder"),
        _assistant_call("EditFile", {"path": "a.py", "old_string": "x", "new_string": "y"}),
    ]
    soul = _make_soul(history=history)
    msg = await gate.check(soul)
    assert msg is not None
    assert "no verification" in msg


async def test_edits_with_test_run_no_block() -> None:
    gate = VerificationGate(max_nudges=2)
    history = [
        _user("implement the feature"),
        _assistant_call("WriteFile", {"path": "a.py", "content": "x"}),
        _assistant_call("Powershell", {"command": "pytest tests/ -q"}),
    ]
    soul = _make_soul(history=history)
    assert await gate.check(soul) is None


async def test_edits_with_todolist_done_no_block() -> None:
    gate = VerificationGate(max_nudges=2)
    history = [
        _user("implement the feature"),
        _assistant_call("WriteFile", {"path": "a.py", "content": "x"}),
        _assistant_call("TodoList", {"todos": [{"title": "impl", "status": "done"}]}),
    ]
    soul = _make_soul(history=history)
    assert await gate.check(soul) is None


async def test_previous_turn_edits_do_not_block_current_turn() -> None:
    gate = VerificationGate(max_nudges=2)
    history = [
        _user("old turn"),
        _assistant_call("WriteFile", {"path": "a.py", "content": "x"}),
        _user("new turn: just chat"),
        Message(role="assistant", content=[TextPart(text="sure, here is the answer")]),
    ]
    soul = _make_soul(history=history)
    assert await gate.check(soul) is None


# ---------------------------------------------------------------------------
# Nudge cap (deadlock prevention)
# ---------------------------------------------------------------------------


async def test_max_nudges_cap() -> None:
    gate = VerificationGate(max_nudges=2)
    soul = _make_soul(todos=[_todo("task A", "pending")])
    assert await gate.check(soul) is not None  # nudge 1
    assert await gate.check(soul) is not None  # nudge 2
    assert await gate.check(soul) is None      # released


async def test_nudges_reset_on_new_turn() -> None:
    gate = VerificationGate(max_nudges=1)
    soul = _make_soul(todos=[_todo("task A", "pending")], turn_id="t1")
    assert await gate.check(soul) is not None
    assert await gate.check(soul) is None
    soul._current_turn_id = "t2"
    assert await gate.check(soul) is not None  # fresh budget in the new turn


# ---------------------------------------------------------------------------
# Construction inside KimiSoul (config wiring)
# ---------------------------------------------------------------------------


def test_kimisoul_constructs_gate(runtime: Runtime, tmp_path) -> None:
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    soul = KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))
    assert soul._verification_gate._max_nudges == soul._loop_control.verification_gate_max_nudges
