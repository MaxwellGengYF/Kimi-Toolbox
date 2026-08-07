"""Tests for _maybe_build_code_todo_reminder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class FakeTodo:
    """Minimal todo-like object for testing."""
    def __init__(self, title: str, status: str, code: str | None = None) -> None:
        self.title = title
        self.status = status
        self.code = code


class FakeTodoDict:
    """Dict-based todo for subagent state testing."""
    def __init__(self, title: str, status: str, code: str | None = None) -> None:
        self.title = title
        self.status = status
        self.code = code


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock session with minimal structure."""
    session = MagicMock()
    cli = MagicMock()
    runtime = MagicMock()
    runtime.role = "root"
    cli._runtime = runtime
    soul = MagicMock()
    agent = MagicMock()
    toolset = MagicMock()
    todo_tool = MagicMock()
    todo_tool._verify_and_set_todo_status = MagicMock()
    toolset.find.return_value = todo_tool
    agent.toolset = toolset
    soul.agent = agent
    cli.soul = soul
    session._cli = cli
    return session


@pytest.mark.asyncio
async def test_returns_none_when_no_todos_have_code(mock_session: MagicMock) -> None:
    """When no todos have code, returns None."""
    from kimix.utils.prompt import _maybe_build_code_todo_reminder

    state = MagicMock()
    state.todos = [FakeTodo("task1", "pending"), FakeTodo("task2", "done")]
    mock_session._cli.session = MagicMock(state=state)

    result = await _maybe_build_code_todo_reminder(mock_session)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_code_todos_all_done(mock_session: MagicMock) -> None:
    """When code todos are all done, returns None (skipped by continue)."""
    from kimix.utils.prompt import _maybe_build_code_todo_reminder

    state = MagicMock()
    state.todos = [FakeTodo("task1", "done", code="print('ok')")]
    mock_session._cli.session = MagicMock(state=state)

    result = await _maybe_build_code_todo_reminder(mock_session)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_no_todos_at_all(mock_session: MagicMock) -> None:
    """Empty todos list should return None."""
    from kimix.utils.prompt import _maybe_build_code_todo_reminder

    state = MagicMock()
    state.todos = []
    mock_session._cli.session = MagicMock(state=state)

    result = await _maybe_build_code_todo_reminder(mock_session)
    assert result is None
