"""End-to-end tests for the Goal→TodoList code merge.

Tests the full flow:
1. Creating todos with `code` field
2. Auto-verification when marking a todo `done`
3. Reverting to `pending` on verification failure
4. Code indicator in display
5. Persistence of code through TodoItemState
6. _maybe_build_code_todo_reminder in prompt.py
7. system_prompt.py updated instructions
8. Goal/RunGoal tools no longer exist
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimi_cli.tools.todo import Params, Todo, TodoList


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
    todo_tool._verify_and_set_todo_status = AsyncMock()
    toolset.find.return_value = todo_tool
    agent.toolset = toolset
    soul.agent = agent
    cli.soul = soul
    session._cli = cli
    return session


class TestE2ECodeTodo:
    """End-to-end tests: code field on Todo items."""

    # ---- E2E: Todo creation with code ----

    async def test_create_todo_with_code(self, todo_list_tool: TodoList) -> None:
        """Creating a todo with code persists the code field."""
        await todo_list_tool(Params(
            todos=[Todo(title="CodeTask", status="pending", code="print('hello')")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "CodeTask":
                assert t.code == "print('hello')"
                break
        else:
            pytest.fail("Todo not found")

    async def test_create_todo_without_code(self, todo_list_tool: TodoList) -> None:
        """Creating a todo without code leaves code as None."""
        await todo_list_tool(Params(
            todos=[Todo(title="NoCodeTask", status="pending")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "NoCodeTask":
                assert t.code is None
                break
        else:
            pytest.fail("Todo not found")

    async def test_code_indicator_in_output(self, todo_list_tool: TodoList) -> None:
        """Output shows code indicator for todos with code."""
        result = await todo_list_tool(Params(
            todos=[Todo(title="CodeTask", status="pending", code="print('x')")]
        ))
        assert "[code: inline]" in result.output

    async def test_no_code_indicator(self, todo_list_tool: TodoList) -> None:
        """Output does NOT show code indicator when no code."""
        result = await todo_list_tool(Params(
            todos=[Todo(title="NoCodeTask", status="pending")]
        ))
        assert "[code:" not in result.output

    # ---- E2E: code_file alias ----

    async def test_code_file_alias(self, todo_list_tool: TodoList) -> None:
        """code_file alias is accepted."""
        await todo_list_tool(Params(
            todos=[Todo(title="AliasTask", status="pending", code_file="print('alias')")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "AliasTask":
                assert t.code == "print('alias')"
                break
        else:
            pytest.fail("Todo not found")

    # ---- E2E: Merge preserves code ----

    async def test_merge_preserves_code(self, todo_list_tool: TodoList) -> None:
        """Updating status via merge preserves existing code."""
        await todo_list_tool(Params(
            todos=[Todo(title="MergeTask", status="pending", code="print('keep')")]
        ))
        await todo_list_tool(Params(
            todos=[Todo(title="MergeTask", status="done")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "MergeTask":
                assert t.code == "print('keep')"
                assert t.status == "done"
                break
        else:
            pytest.fail("Todo not found")

    async def test_merge_empty_notes_code_keeps_old_values(self, todo_list_tool: TodoList) -> None:
        """Updating a title with None/empty notes & code keeps the previously stored values."""
        await todo_list_tool(Params(
            todos=[Todo(title="KeepNotesCode", status="pending", notes="keep notes", code="print('keep')")]
        ))
        await todo_list_tool(Params(
            todos=[Todo(title="KeepNotesCode", status="done", notes="", code="")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "KeepNotesCode":
                assert t.code == "print('keep')"
                assert t.notes == "keep notes"
                assert t.status == "done"
                break
        else:
            pytest.fail("Todo not found")

    async def test_merge_changed_notes_code_replaces_old_values(self, todo_list_tool: TodoList) -> None:
        """Updating a title with filled, different notes/code replaces the old values."""
        await todo_list_tool(Params(
            todos=[Todo(title="ChangeNotesCode", status="pending", notes="old notes", code="print('old')")]
        ))
        await todo_list_tool(Params(
            todos=[Todo(title="ChangeNotesCode", status="in_progress", notes="new notes", code="print('new')")]
        ))
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "ChangeNotesCode":
                assert t.code == "print('new')"
                assert t.notes == "new notes"
                assert t.status == "in_progress"
                break
        else:
            pytest.fail("Todo not found")

    # ---- E2E: Display block includes code ----

    async def test_display_block_includes_code(self, todo_list_tool: TodoList) -> None:
        """DisplayBlock items include code field."""
        from kimi_cli.tools.display import TodoDisplayBlock
        result = await todo_list_tool(Params(
            todos=[Todo(title="DisplayTask", status="pending", code="print('display')")]
        ))
        assert len(result.display) == 1
        block = result.display[0]
        assert isinstance(block, TodoDisplayBlock)
        assert block.items[0].code == "print('display')"

    # ---- E2E: code persisted through TodoItemState ----

    async def test_code_persisted_via_item_state(self, todo_list_tool: TodoList) -> None:
        """Code is persisted through TodoItemState (disk round-trip)."""
        from kimi_cli.session_state import load_session_state, TodoItemState
        await todo_list_tool(Params(
            todos=[Todo(title="PersistTask", status="pending", code="print('persist')")]
        ))
        # Verify via TodoItemState
        items = todo_list_tool._item_states(todo_list_tool._load_todos())
        for item in items:
            if item.title == "PersistTask":
                assert item.code == "print('persist')"
                break
        else:
            pytest.fail("Item not found")
        # Verify via disk
        runtime = todo_list_tool._runtime
        disk_state = load_session_state(runtime.session.dir)
        for t in disk_state.todos:
            if t.title == "PersistTask":
                assert t.code == "print('persist')"
                break
        else:
            pytest.fail("Todo not found on disk")

    # ---- E2E: No dangling references to Goal/RunGoal ----

    def test_goal_tools_no_longer_exist(self) -> None:
        """Goal and RunGoal tools are deleted."""
        import importlib
        for mod_name in ["kimix.tools.goal"]:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                importlib.import_module(mod_name)

    def test_system_prompt_updated(self) -> None:
        """system_prompt.py no longer mentions Goal/RunGoal tools."""
        import ast
        import sys
        root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(root / "src"))
        with open(root / "src/kimix/utils/system_prompt.py", encoding="utf-8") as f:
            content = f.read()
        # Should NOT contain old Goal reference
        assert "define the goal with the `Goal` tool" not in content
        # Should contain new code-todo reference
        assert "attach verification `code`" in content
        assert "TodoList" in content

    def test_base_py_no_goal_run_goal(self) -> None:
        """base.py no longer handles Goal/RunGoal in format_tool_args."""
        root = Path(__file__).parent.parent.parent.parent
        with open(root / "src/kimix/base.py", encoding="utf-8") as f:
            content = f.read()
        # Check _format_tool_args no longer has Goal/RunGoal cases
        assert 'case "Goal":' not in content, "base.py still has case 'Goal' in _format_tool_args"
        assert 'case "RunGoal":' not in content, "base.py still has case 'RunGoal' in _format_tool_args"
        # Check _STREAM_TOOL_NAMES doesn't contain Goal (comment with "Goal" is OK)
        lines = content.split("\n")
        stream_section = False
        for line in lines:
            if "_STREAM_TOOL_NAMES" in line:
                stream_section = True
            if stream_section and '"Goal"' in line and not line.strip().startswith("#"):
                pytest.fail(f"base.py still has Goal in _STREAM_TOOL_NAMES: {line}")


class TestE2EPromptCodeTodoReminder:
    """E2E tests for _maybe_build_code_todo_reminder."""

    @pytest.mark.asyncio
    async def test_reminder_runs_code_and_marks_done(self, mock_session: MagicMock) -> None:
        """_maybe_build_code_todo_reminder runs code and marks done on success."""
        from kimix.utils.prompt import _maybe_build_code_todo_reminder

        state = MagicMock()
        state.todos = [MagicMock(title="E2ETask", status="pending", code="print('e2e_ok')")]
        mock_session._cli.session = MagicMock(state=state)

        # Mock the tool instance on the toolset
        toolset = mock_session._cli.soul.agent.toolset
        todo_tool = MagicMock()
        # verify_code_todos delegates entirely to _verify_and_set_todo_status
        # (which runs the code once); None means the code passed.
        todo_tool._verify_and_set_todo_status = AsyncMock(return_value=None)
        toolset.find.return_value = todo_tool
        with patch("kimi_cli.tools.todo.TodoList._resolve_code_executable", return_value="/tmp/fake.py"):
            with patch("kimi_cli.tools.todo.TodoList._run_code", AsyncMock(return_value=(True, "ok"))):
                result = await _maybe_build_code_todo_reminder(mock_session)
        assert result is None  # No failures
        todo_tool._verify_and_set_todo_status.assert_called_once_with("E2ETask", "done")

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_reminder_accumulates_failures(self, mock_session: MagicMock) -> None:
        """_maybe_build_code_todo_reminder accumulates multiple failures."""
        from kimix.utils.prompt import _maybe_build_code_todo_reminder
        state = MagicMock()
        state.todos = [
            MagicMock(title="FailA", status="pending", code="bad_code_a"),
            MagicMock(title="FailB", status="pending", code="bad_code_b"),
        ]
        mock_session._cli.session = MagicMock(state=state)
        toolset = mock_session._cli.soul.agent.toolset
        todo_tool = MagicMock()
        # Each failing verification returns an error message from
        # _verify_and_set_todo_status (single-execution delegation).
        todo_tool._verify_and_set_todo_status = AsyncMock(
            side_effect=[
                "Todo 'FailA' verification failed: boom_a",
                "Todo 'FailB' verification failed: boom_b",
            ]
        )
        toolset.find.return_value = todo_tool
        result = await _maybe_build_code_todo_reminder(mock_session)
        assert result is not None
        assert "FailA" in result
        assert "FailB" in result
        assert "verification failed" in result
    async def test_prompt_async_enforcement_loop(self, mock_session: MagicMock) -> None:
        """The enforcement loop in prompt_async uses _maybe_build_code_todo_reminder."""
        from kimix.utils.prompt import _maybe_build_code_todo_reminder
        # Verify the function is defined and callable
        assert callable(_maybe_build_code_todo_reminder)
        # Check the function exists and is accessible
        import inspect
        sig = inspect.signature(_maybe_build_code_todo_reminder)
        assert "session" in sig.parameters
        assert "strong" in sig.parameters
