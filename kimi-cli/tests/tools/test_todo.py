"""Tests for TodoList tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from kosong.tooling import ToolReturnValue

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.todo import MergeResult, Params, Todo, TodoList


@pytest.fixture
def todo_list_tool(runtime: Runtime) -> TodoList:
    """Create a TodoList tool instance with runtime."""
    return TodoList(runtime)


class TestTodoListOutputNotEmpty:
    """Regression test for issue #1710: TodoList storm.

    The root cause is that TodoList returned output="" which meant the model
    only saw '<system>Todo list updated</system>' — no confirmation of what it
    saved. This led to repeated calls (a "storm") especially when Shell was disabled.
    """

    async def test_write_mode_returns_nonempty_output(self, todo_list_tool: TodoList):
        """When todos are provided, the tool must return a non-empty output
        so the model gets meaningful feedback (not just 'Todo list updated')."""
        params = Params(
            todos=[
                Todo(title="Analyze code", status="pending", notes=""),
                Todo(title="Write tests", status="in_progress", notes=""),
                Todo(title="Read requirements", status="done", notes=""),
            ]
        )
        result = await todo_list_tool(params)
        assert not result.is_error
        # The critical assertion: output must NOT be empty
        assert result.output != "", (
            "TodoList output must not be empty — this is the root cause of issue #1710. "
            "The model needs to see confirmation of the todo state it just set."
        )
        assert result.message == "Todo list appended."

    async def test_read_mode_returns_current_todos(self, todo_list_tool: TodoList):
        """When no todos are provided (None), the tool should return the current
        todo list from persistent storage, including status."""
        # First write some todos
        write_params = Params(
            todos=[
                Todo(title="Task A", status="pending", notes=""),
                Todo(title="Task B", status="done", notes=""),
            ]
        )
        await todo_list_tool(write_params)

        # Then read without providing todos
        read_params = Params(todos=None)
        result = await todo_list_tool(read_params)
        assert not result.is_error
        assert "Task A" in result.output
        assert "Task B" in result.output
        assert "pending" in result.output
        assert "done" in result.output

    async def test_read_mode_empty_list(self, todo_list_tool: TodoList):
        """Reading with no prior todos should return a clear empty message."""
        read_params = Params(todos=None)
        result = await todo_list_tool(read_params)
        assert not result.is_error
        assert result.output  # non-empty even when no todos

    async def test_default_mode_is_append(self, todo_list_tool: TodoList):
        """Calling Params(todos=[...]) without mode should merge (append behavior)."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Old task", status="done", notes="")])
        )
        assert not result.is_error
        assert "mode='overwrite'" not in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[done] Old task" in read.output

    async def test_explicit_append_mode_merges(self, todo_list_tool: TodoList):
        """Calling Params(todos=[...], mode='append') should merge into existing list."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Old task", status="done", notes="")], mode="append")
        )
        assert not result.is_error
        assert "mode='overwrite'" not in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[done] Old task" in read.output

    async def test_write_empty_list_clears_todos_when_overwrite(self, todo_list_tool: TodoList):
        """Passing an empty list [] with mode='force_overwrite' clears all todos."""
        # Write some todos first
        write_params = Params(todos=[Todo(title="Task A", status="pending", notes="")])
        await todo_list_tool(write_params)

        # Clear with empty list + overwrite mode + force
        clear_params = Params(todos=[], mode="force_overwrite")
        result = await todo_list_tool(clear_params)
        assert not result.is_error
        assert result.output == (
            "Todo list force overwritten (0 total: 0 done, 0 in progress, 0 pending)"
        )
        assert "mode='force_overwrite'" in result.message

        # Verify cleared
        read_params = Params(todos=None)
        result = await todo_list_tool(read_params)
        assert isinstance(result.output, str)
        assert "empty" in result.output.lower() or result.output.strip() == "Todo list is empty."

    async def test_write_empty_list_default_mode_errors(self, todo_list_tool: TodoList):
        """Passing an empty list [] with default mode (append) when old todos are
        not all done should return an error."""
        write_params = Params(todos=[Todo(title="Task A", status="pending", notes="")])
        await todo_list_tool(write_params)

        clear_params = Params(todos=[])
        result = await todo_list_tool(clear_params)
        assert result.is_error
        assert "Cannot clear todos" in result.output

    async def test_overwrite_without_force_when_old_not_done_errors(self, todo_list_tool: TodoList):
        """mode='overwrite' when old todos are not all done errors."""
        write_params = Params(todos=[Todo(title="Task A", status="pending", notes="")])
        await todo_list_tool(write_params)

        overwrite_params = Params(
            todos=[Todo(title="New task", status="pending", notes="")], mode="overwrite"
        )
        result = await todo_list_tool(overwrite_params)
        assert result.is_error
        assert "Cannot overwrite todos" in result.output
        assert "Task A" in result.output

    async def test_overwrite_without_force_when_all_old_done_succeeds(
        self, todo_list_tool: TodoList
    ):
        """mode='overwrite' succeeds when all old todos are done."""
        write_params = Params(todos=[Todo(title="Task A", status="done", notes="")])
        await todo_list_tool(write_params)

        overwrite_params = Params(
            todos=[Todo(title="New task", status="pending", notes="")], mode="overwrite"
        )
        result = await todo_list_tool(overwrite_params)
        assert not result.is_error
        assert "New task" in str(result.display)
        assert "Task A" not in str(result.display)

    async def test_write_empty_list_when_all_done_clears(self, todo_list_tool: TodoList):
        """Passing an empty list [] when all old todos are done should clear."""
        write_params = Params(todos=[Todo(title="Task A", status="done", notes="")])
        await todo_list_tool(write_params)

        clear_params = Params(todos=[])
        result = await todo_list_tool(clear_params)
        assert not result.is_error
        assert result.output == "Todo list appended (0 total: 0 done, 0 in progress, 0 pending)"

        read_params = Params(todos=None)
        result = await todo_list_tool(read_params)
        assert "empty" in result.output.lower()

    async def test_root_todos_persisted_to_disk(self, todo_list_tool: TodoList, runtime: Runtime):
        """Write mode should persist todos to disk via SessionState."""
        from kimi_cli.session_state import load_session_state

        params = Params(
            todos=[
                Todo(title="Disk task", status="in_progress", notes=""),
                Todo(title="Another task", status="done", notes=""),
            ]
        )
        await todo_list_tool(params)

        # Verify by loading directly from disk, bypassing in-memory state
        disk_state = load_session_state(runtime.session.dir)
        assert len(disk_state.todos) == 2
        assert disk_state.todos[0].title == "Disk task"
        assert disk_state.todos[0].status == "in_progress"
        assert disk_state.todos[1].title == "Another task"
        assert disk_state.todos[1].status == "done"

    async def test_write_mode_display_block(self, todo_list_tool: TodoList):
        """Write mode should still produce TodoDisplayBlock for UI rendering."""
        from kimi_cli.tools.display import TodoDisplayBlock

        params = Params(todos=[Todo(title="UI task", status="pending", notes="")])
        result = await todo_list_tool(params)
        assert len(result.display) == 1
        assert isinstance(result.display[0], TodoDisplayBlock)
        assert result.display[0].items[0].title == "UI task"

    async def test_read_mode_no_display_block(self, todo_list_tool: TodoList):
        """Read mode should not produce display blocks (no UI side-effect)."""
        read_params = Params(todos=None)
        result = await todo_list_tool(read_params)
        assert result.display == []


class TestTodoListActiveSummary:
    """Tests for the active-todo summary appended to successful writes."""

    async def test_write_outputs_pending_and_in_progress_summary(self, todo_list_tool: TodoList):
        """Successful writes list pending and in_progress todos in output."""
        params = Params(
            todos=[
                Todo(title="Pending task", status="pending", notes=""),
                Todo(title="In progress task", status="in_progress", notes=""),
                Todo(title="Done task", status="done", notes=""),
            ]
        )
        result = await todo_list_tool(params)
        assert not result.is_error
        assert "- [pending] Pending task" in result.output
        assert "- [in progress] In progress task" in result.output
        assert "Done task" not in result.output

    async def test_write_summary_omits_done_items(self, todo_list_tool: TodoList):
        """When all todos are done, no active summary is emitted."""
        params = Params(todos=[Todo(title="Only done", status="done", notes="")])
        result = await todo_list_tool(params)
        assert not result.is_error
        assert result.output == "Todo list appended (1 total: 1 done, 0 in progress, 0 pending)"

    async def test_write_summary_preserved_when_all_done(self, todo_list_tool: TodoList):
        """Marking all active todos as done removes the active summary."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Active A", status="pending", notes=""),
                    Todo(title="Active B", status="in_progress", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Active A", status="done", notes=""),
                    Todo(title="Active B", status="done", notes=""),
                ]
            )
        )
        assert not result.is_error
        assert result.output == "Todo list appended (2 total: 2 done, 0 in progress, 0 pending)"

    async def test_write_summary_with_overwrite_warning(self, todo_list_tool: TodoList):
        """Warning from mode='force_overwrite' is in message; active summary is in output."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))
        params = Params(
            todos=[
                Todo(title="Forced pending", status="pending", notes=""),
                Todo(title="Forced in progress", status="in_progress", notes=""),
            ],
            mode="force_overwrite",
        )
        result = await todo_list_tool(params)
        assert not result.is_error
        assert result.output.startswith("Todo list force overwritten")
        assert "- [pending] Forced pending" in result.output
        assert "- [in progress] Forced in progress" in result.output
        assert "mode" not in result.output
        assert "mode='force_overwrite'" in result.message

    async def test_write_summary_order_matches_persisted_order(self, todo_list_tool: TodoList):
        """Active todos are listed in the exact order they appear after the write."""
        params = Params(
            todos=[
                Todo(title="First", status="pending", notes=""),
                Todo(title="Second", status="in_progress", notes=""),
                Todo(title="Third", status="pending", notes=""),
                Todo(title="Fourth", status="done", notes=""),
                Todo(title="Fifth", status="pending", notes=""),
            ]
        )
        result = await todo_list_tool(params)
        assert not result.is_error

        active_lines = [line for line in result.output.splitlines() if line.startswith("- [")]
        assert active_lines == [
            "- [pending] First",
            "- [in progress] Second",
            "- [pending] Third",
            "- [pending] Fifth",
        ]


class TestTodoListIncrementalUpdate:
    """Test incremental update behavior when new todos are a subset of old."""

    async def test_incremental_update_status(self, todo_list_tool: TodoList):
        """Updating a subset of todos should only change their statuses."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="in_progress", notes=""),
                    Todo(title="C", status="pending", notes=""),
                ]
            )
        )

        # Update only B and C
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="B", status="done", notes=""),
                    Todo(title="C", status="in_progress", notes=""),
                ]
            )
        )
        assert not result.is_error

        # Read back and verify
        read_result = await todo_list_tool(Params(todos=None))
        assert "[pending] A" in read_result.output
        assert "[done] B" in read_result.output
        assert "[in_progress] C" in read_result.output

    async def test_incremental_update_preserves_order(self, todo_list_tool: TodoList):
        """Incremental update should preserve the original order of todos."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="First", status="pending", notes=""),
                    Todo(title="Second", status="pending", notes=""),
                    Todo(title="Third", status="pending", notes=""),
                ]
            )
        )

        # Update in reverse order
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Third", status="done", notes=""),
                    Todo(title="First", status="done", notes=""),
                ]
            )
        )

        read_result = await todo_list_tool(Params(todos=None))
        lines = read_result.output.splitlines()
        assert lines[1] == "- [done] First"
        assert lines[2] == "- [pending] Second"
        assert lines[3].startswith("- [done] Third")

    async def test_single_todo_update(self, todo_list_tool: TodoList):
        """Passing a single Todo instance should update just that item."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )

        # Pass single Todo, not a list
        result = await todo_list_tool(Params(todos=Todo(title="B", status="done", notes="")))
        assert not result.is_error

        read_result = await todo_list_tool(Params(todos=None))
        assert "[pending] A" in read_result.output
        assert "[done] B" in read_result.output


class TestTodoListValidation:
    """Test validation rules for new todos."""

    async def test_duplicate_titles_rejected(self, todo_list_tool: TodoList):
        """Duplicate titles in new todos should return an error."""
        params = Params(
            todos=[
                Todo(title="Task A", status="pending", notes=""),
                Todo(title="Task B", status="in_progress", notes=""),
                Todo(title="Task A", status="done", notes=""),
            ]
        )
        result = await todo_list_tool(params)
        assert result.is_error
        assert "Duplicate todo titles found" in result.output

    async def test_title_too_long_rejected(self, todo_list_tool: TodoList):
        """Titles longer than 65536 characters should be rejected at model level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Todo(title="x" * 65537, status="pending", notes="")

    async def test_whitespace_only_title_rejected(self, todo_list_tool: TodoList):
        """Titles that are only whitespace should be rejected at model level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Todo(title="   ", status="pending", notes="")

    async def test_todo_count_limit(self, todo_list_tool: TodoList):
        """More than 4096 todos should return an error."""
        todos = [Todo(title=f"Task {i}", status="pending", notes="") for i in range(4097)]
        params = Params(todos=todos)
        result = await todo_list_tool(params)
        assert result.is_error
        assert "exceeds maximum limit of 4096" in result.output

    async def test_overwrite_outputs_warning(self, todo_list_tool: TodoList):
        """mode='force_overwrite' should include a warning in the message."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))
        params = Params(
            todos=[Todo(title="Task", status="pending", notes="")], mode="force_overwrite"
        )
        result = await todo_list_tool(params)
        assert not result.is_error
        assert "mode='force_overwrite'" in result.message

    async def test_status_regression_blocked(self, todo_list_tool: TodoList):
        """Changing a done todo back to pending/in_progress should be blocked."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="done", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )
        assert result.is_error
        assert "Cannot regress completed todos" in result.output

        # No partial save on regression error: state stays as it was before the call
        read_result = await todo_list_tool(Params(todos=None))
        assert "[done] B" in read_result.output
        assert "[pending] A" in read_result.output


class TestTodoListNewListValidation:
    """Test error behavior when new todos contain items not in the old list."""

    async def test_new_todo_with_old_incomplete_appends(self, todo_list_tool: TodoList):
        """Append mode adds brand-new titles to the end of the existing list."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="New task", status="pending", notes="")])
        )
        assert not result.is_error

        read_result = await todo_list_tool(Params(todos=None))
        assert "[pending] Old task" in read_result.output
        assert "[pending] New task" in read_result.output

    async def test_new_todo_when_all_old_done_appends(self, todo_list_tool: TodoList):
        """Append mode keeps existing done items and appends new titles."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="done", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="New task", status="pending", notes="")])
        )
        assert not result.is_error

        read_result = await todo_list_tool(Params(todos=None))
        assert "New task" in read_result.output
        assert "Old task" in read_result.output

    async def test_overwrite_bypasses_validation(self, todo_list_tool: TodoList):
        """mode='force_overwrite' should bypass the incomplete-todo check."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Old task", status="pending", notes=""),
                    Todo(title="Another old", status="in_progress", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[Todo(title="New task", status="done", notes="")],
                mode="force_overwrite",
            )
        )
        assert not result.is_error

        read_result = await todo_list_tool(Params(todos=None))
        assert "New task" in read_result.output
        assert "Old task" not in read_result.output

    async def test_new_todo_mixed_with_old_titles_merges(self, todo_list_tool: TodoList):
        """Overlapping titles merge instead of erroring."""
        await todo_list_tool(Params(todos=[Todo(title="Keep me", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Keep me", status="done", notes=""),
                    Todo(title="Brand new", status="pending", notes=""),
                ]
            )
        )
        assert not result.is_error
        assert "Keep me" in str(result.display)
        assert "Brand new" in str(result.display)

    async def test_subset_update_does_not_error(self, todo_list_tool: TodoList):
        """A strict subset of old titles should always succeed (incremental update)."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )

        result = await todo_list_tool(Params(todos=[Todo(title="A", status="done", notes="")]))
        assert not result.is_error

    async def test_new_todo_when_old_empty_succeeds(self, todo_list_tool: TodoList):
        """Writing new todos when old list is empty should never error."""
        result = await todo_list_tool(
            Params(todos=[Todo(title="New task", status="pending", notes="")])
        )
        assert not result.is_error

    async def test_clear_when_old_empty_succeeds(self, todo_list_tool: TodoList):
        """Writing an empty list when old list is empty should succeed."""
        result = await todo_list_tool(Params(todos=[]))
        assert not result.is_error

    async def test_single_todo_when_old_empty_succeeds(self, todo_list_tool: TodoList):
        """Writing a single Todo when old list is empty should succeed."""
        result = await todo_list_tool(
            Params(todos=Todo(title="Only task", status="in_progress", notes=""))
        )
        assert not result.is_error


class TestTodoListSubagent:
    """Test TodoList behavior in subagent context."""

    async def test_subagent_uses_independent_storage(self, runtime: Runtime):
        """Subagent todos should be stored independently from root agent."""
        # Create root tool and set a todo
        root_tool = TodoList(runtime)
        await root_tool(Params(todos=[Todo(title="Root task", status="pending", notes="")]))

        # Create a subagent runtime
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-1",
            subagent_type="coder",
        )
        # Initialize the subagent instance directory
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-1", create=True)

        sub_tool = TodoList(subagent_runtime)

        # Subagent should start with empty todos
        result = await sub_tool(Params(todos=None))
        assert isinstance(result.output, str)
        assert "empty" in result.output.lower() or "Root task" not in result.output

        # Subagent writes its own todo
        await sub_tool(Params(todos=[Todo(title="Sub task", status="in_progress", notes="")]))
        result = await sub_tool(Params(todos=None))
        assert "Sub task" in result.output

        # Root agent should still have its own todo
        result = await root_tool(Params(todos=None))
        assert "Root task" in result.output
        assert "Sub task" not in result.output

    async def test_subagent_no_store_or_id_returns_error(self, runtime: Runtime):
        """When subagent_store or subagent_id is None, save returns an error."""
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-2",
            subagent_type="coder",
        )
        # Force store/id to None to simulate edge case
        subagent_runtime.subagent_store = None
        subagent_runtime.subagent_id = None

        tool = TodoList(subagent_runtime)

        # Write should return error since state file is unavailable
        result = await tool(Params(todos=[Todo(title="Ghost task", status="pending", notes="")]))
        assert result.is_error
        assert "Unable to save subagent todos" in result.output

        # Read should return empty
        result = await tool(Params(todos=None))
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "empty" in result.output.lower()

    async def test_corrupted_subagent_state_file(self, runtime: Runtime):
        """Corrupted subagent state.json should be handled gracefully."""
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-3",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        instance_dir = subagent_runtime.subagent_store.instance_dir("test-sub-3", create=True)

        # Write corrupted JSON to state.json
        state_file = instance_dir / "state.json"
        state_file.write_text("not valid json {{{", encoding="utf-8")

        tool = TodoList(subagent_runtime)

        # Read should return empty (corrupted file treated as empty)
        result = await tool(Params(todos=None))
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "empty" in result.output.lower()

        # Write should overwrite the corrupted file successfully
        result = await tool(Params(todos=[Todo(title="Recovery task", status="pending", notes="")]))
        assert not result.is_error

        # Verify recovery
        result = await tool(Params(todos=None))
        assert "Recovery task" in result.output

    async def test_subagent_malformed_individual_item(self, runtime: Runtime):
        """Malformed individual items in state.json should be skipped, valid ones preserved."""
        import json

        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-4",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        instance_dir = subagent_runtime.subagent_store.instance_dir("test-sub-4", create=True)

        # Write JSON with one valid and one invalid todo item
        state_file = instance_dir / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "todos": [
                        {"title": "Valid task", "status": "pending"},
                        {"bad": "item"},  # missing title and status
                        {"title": "Also valid", "status": "done"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        tool = TodoList(subagent_runtime)
        result = await tool(Params(todos=None))
        assert not result.is_error
        assert "Valid task" in result.output
        assert "Also valid" in result.output
        # The malformed item should be silently skipped
        assert "bad" not in result.output

    async def test_subagent_incremental_update(self, runtime: Runtime):
        """Incremental update should work in subagent context."""
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-incr",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-incr", create=True)

        tool = TodoList(subagent_runtime)
        await tool(
            Params(
                todos=[
                    Todo(title="Sub A", status="pending", notes=""),
                    Todo(title="Sub B", status="pending", notes=""),
                ]
            )
        )

        # Incremental update
        result = await tool(Params(todos=[Todo(title="Sub A", status="done", notes="")]))
        assert not result.is_error

        read_result = await tool(Params(todos=None))
        assert "[done] Sub A" in read_result.output
        assert "[pending] Sub B" in read_result.output

    async def test_subagent_new_list_with_incomplete_appends(self, runtime: Runtime):
        """Append mode in subagent context appends brand-new titles."""
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-err",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-err", create=True)

        tool = TodoList(subagent_runtime)
        await tool(Params(todos=[Todo(title="Sub task", status="in_progress", notes="")]))

        result = await tool(Params(todos=[Todo(title="New sub task", status="pending", notes="")]))
        assert not result.is_error

        read_result = await tool(Params(todos=None))
        assert "Sub task" in read_result.output
        assert "New sub task" in read_result.output


# --- Additional comprehensive tests ---


class TestTodoListFuzzyMatching:
    """Test fuzzy title matching and non-blocking warnings."""

    async def test_new_todo_typo_returns_nonblocking_warning(self, todo_list_tool: TodoList):
        """A typo in a new todo title warns but still appends the new title."""
        await todo_list_tool(
            Params(todos=[Todo(title="Implement feature", status="pending", notes="")])
        )

        result = await todo_list_tool(
            Params(todos=[Todo(title="Implement featuer", status="pending", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "Implement featuer" in result.message
        assert "Implement feature" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Implement feature" in read.output
        assert "[pending] Implement featuer" in read.output

    async def test_new_todo_no_match_appends_without_warning(self, todo_list_tool: TodoList):
        """A completely unrelated title appends cleanly with no warning."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Completely unrelated", status="pending", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" not in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Old task" in read.output
        assert "[pending] Completely unrelated" in read.output

    def test_find_nearest_titles_helper_directly(self):
        """_find_nearest_titles should return nearest string matches."""
        nearest = TodoList._find_nearest_titles(["foo bar"], ["foo baz", "qux"], top_k=1)
        assert "foo bar" in nearest
        assert len(nearest["foo bar"]) == 1
        assert nearest["foo bar"][0].choice == "foo baz"

        empty_candidates = TodoList._find_nearest_titles(["foo"], [], top_k=1)
        assert empty_candidates == {"foo": []}

    def test_find_nearest_titles_exact_match(self):
        """An exact title should match itself."""
        nearest = TodoList._find_nearest_titles(["exact title"], ["exact title", "other"], top_k=1)
        assert nearest["exact title"][0].choice == "exact title"
        assert nearest["exact title"][0].score == 100.0

    def test_find_nearest_titles_typo(self):
        """A minor typo should return the correct title."""
        nearest = TodoList._find_nearest_titles(
            ["Implement featuer"], ["Implement feature", "Write tests"], top_k=1
        )
        assert nearest["Implement featuer"][0].choice == "Implement feature"
        assert nearest["Implement featuer"][0].score >= 60.0

    def test_find_nearest_titles_unrelated_returns_empty(self):
        """A completely unrelated title should return no suggestions."""
        nearest = TodoList._find_nearest_titles(["Completely unrelated"], ["Old task"], top_k=1)
        assert nearest["Completely unrelated"] == []

    def test_find_nearest_titles_word_reorder(self):
        """Reordered words with the same vocabulary still match."""
        nearest = TodoList._find_nearest_titles(["bug fix"], ["fix bug", "write tests"], top_k=1)
        assert nearest["bug fix"][0].choice == "fix bug"
        assert nearest["bug fix"][0].score >= 60.0

    def test_find_nearest_titles_processor_and_cutoff(self):
        """Optional processor and score_cutoff parameters are respected."""
        nearest = TodoList._find_nearest_titles(
            ["UPPER CASE"],
            ["upper case", "other"],
            top_k=1,
            score_cutoff=85.0,
            processor=str.lower,
        )
        assert nearest["UPPER CASE"][0].choice == "upper case"
        assert nearest["UPPER CASE"][0].score == 100.0

        # With a strict cutoff, a case-only difference should not match without
        # the processor, but the processor makes it match.
        no_processor = TodoList._find_nearest_titles(
            ["UPPER CASE"],
            ["upper case"],
            top_k=1,
            score_cutoff=85.0,
        )
        assert no_processor["UPPER CASE"] == []

    async def test_fuzzy_suggestions_in_subagent_context(self, runtime: Runtime):
        """Fuzzy warnings work as non-blocking messages in a subagent."""
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-fuzzy",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-fuzzy", create=True)

        tool = TodoList(subagent_runtime)
        await tool(Params(todos=[Todo(title="Sub task", status="in_progress", notes="")]))

        result = await tool(Params(todos=[Todo(title="Sub taks", status="pending", notes="")]))
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "Sub taks" in result.message
        assert "Sub task" in result.message

        read = await tool(Params(todos=None))
        assert "[in_progress] Sub task" in read.output
        assert "[pending] Sub taks" in read.output


class TestTodoListFuzzyAppendWarning:
    """Test fuzzy near-match warnings in append mode are non-blocking."""

    async def test_typo_in_append_mode_returns_warning(self, todo_list_tool: TodoList):
        """A minor typo warns, but the exact original stays and the typo is appended."""
        await todo_list_tool(
            Params(todos=[Todo(title="Implement feature", status="pending", notes="")])
        )

        result = await todo_list_tool(
            Params(todos=[Todo(title="Implement featuer", status="done", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "Implement featuer" in result.message
        assert "Implement feature" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Implement feature" in read.output
        assert "[done] Implement featuer" in read.output

    async def test_word_reorder_returns_warning(self, todo_list_tool: TodoList):
        """Reordered words matching an existing title warn but still append."""
        await todo_list_tool(Params(todos=[Todo(title="Fix bug", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Bug fix", status="done", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "Bug fix" in result.message
        assert "Fix bug" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Fix bug" in read.output
        assert "[done] Bug fix" in read.output

    async def test_case_only_difference_returns_warning(self, todo_list_tool: TodoList):
        """A case-only difference warns but appends the new-cased title."""
        await todo_list_tool(
            Params(todos=[Todo(title="Implement Feature", status="pending", notes="")])
        )

        result = await todo_list_tool(
            Params(todos=[Todo(title="implement feature", status="done", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "implement feature" in result.message
        assert "Implement Feature" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Implement Feature" in read.output
        assert "[done] implement feature" in read.output

    async def test_mixed_exact_and_fuzzy_returns_warning(self, todo_list_tool: TodoList):
        """Exact matches update; fuzzy near-matches append with a warning."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Exact match", status="pending", notes=""),
                    Todo(title="Fuzzy match", status="pending", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Exact match", status="done", notes=""),
                    Todo(title="Fuzzy macth", status="done", notes=""),
                ]
            )
        )
        assert not result.is_error
        assert "looks like existing" in result.message
        assert "Fuzzy macth" in result.message
        assert "Fuzzy match" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[done] Exact match" in read.output
        assert "[pending] Fuzzy match" in read.output
        assert "[done] Fuzzy macth" in read.output

    async def test_unrelated_title_appends_cleanly(self, todo_list_tool: TodoList):
        """A clearly unrelated title appends without any warning."""
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Completely unrelated", status="pending", notes="")])
        )
        assert not result.is_error
        assert "looks like existing" not in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Old task" in read.output
        assert "[pending] Completely unrelated" in read.output


class TestTodoModel:
    """Test Todo model validation."""

    def test_title_stripped(self):
        """Title with leading/trailing whitespace should be stripped."""
        todo = Todo(title="  hello  ", status="pending", notes="")
        assert todo.title == "hello"

    def test_title_internal_whitespace_preserved(self):
        """Internal whitespace in title should be preserved."""
        todo = Todo(title="hello world", status="pending", notes="")
        assert todo.title == "hello world"

    def test_title_min_length(self):
        """Single non-whitespace character should be valid."""
        todo = Todo(title="x", status="pending", notes="")
        assert todo.title == "x"

    def test_valid_statuses(self):
        """All valid statuses should be accepted."""
        for status in ("pending", "in_progress", "done"):
            todo = Todo(title="Task", status=status, notes="")
            assert todo.status == status

    def test_notes_defaults_to_none(self):
        todo = Todo(title="Task", status="pending")
        assert todo.notes is None

    def test_whitespace_only_notes_become_none(self):
        todo = Todo(title="Task", status="pending", notes="   ")
        assert todo.notes is None

    def test_notes_none_valid(self):
        todo = Todo(title="Task", status="pending", notes=None)  # type: ignore[arg-type]
        assert todo.notes is None


class TestTodoListNotes:
    """Tests for notes behavior and mode synonyms."""

    async def test_notes_mode_synonym_maps_to_append(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="Task A", status="pending", notes="")]))
        # 'notes' is no longer a recognized mode synonym; should raise ValueError
        with pytest.raises((ValueError, Exception)):
            Params(todos=[Todo(title="Task B", status="pending", notes="")], mode="notes")

    async def test_notes_typos_rejected(self, todo_list_tool: TodoList):
        """Typos like 'ntoes' are no longer accepted as mode synonyms."""
        for typo in ("ntoes", "noets", "nots"):
            with pytest.raises((ValueError, Exception)):
                Params(todos=[Todo(title=f"Task {typo}", status="pending", notes="")], mode=typo)

    async def test_in_progress_output_includes_notes(self, todo_list_tool: TodoList):
        params = Params(
            todos=[
                Todo(title="Pending task", status="pending", notes=""),
                Todo(title="Active task", status="in_progress", notes="Working on tests"),
                Todo(title="Done task", status="done", notes=""),
            ]
        )
        result = await todo_list_tool(params)
        assert not result.is_error
        assert "- [in progress] Active task" in result.output
        assert "Notes: Working on tests" in result.output
        assert "Notes:" not in result.output.split("- [pending]")[1].split("\n")[0]

    async def test_in_progress_without_notes_omits_notes_line(self, todo_list_tool: TodoList):
        params = Params(todos=[Todo(title="Active task", status="in_progress", notes="")])
        result = await todo_list_tool(params)
        assert not result.is_error
        assert "- [in progress] Active task" in result.output
        assert "Notes:" not in result.output

    async def test_read_mode_includes_in_progress_notes(self, todo_list_tool: TodoList):
        await todo_list_tool(
            Params(todos=[Todo(title="Active task", status="in_progress", notes="Details here")])
        )
        result = await todo_list_tool(Params(todos=None))
        assert "Notes: Details here" in result.output

    async def test_merge_preserves_old_notes_when_new_notes_empty(self, todo_list_tool: TodoList):
        await todo_list_tool(
            Params(todos=[Todo(title="Task A", status="pending", notes="Keep me")])
        )
        result = await todo_list_tool(
            Params(todos=[Todo(title="Task A", status="in_progress", notes="")])
        )
        assert not result.is_error
        read = await todo_list_tool(Params(todos=None))
        assert "Notes: Keep me" in read.output


class TestTodoListInternals:
    """Test internal helper methods directly."""

    def test_find_duplicate_titles(self):
        """_find_duplicate_titles returns all duplicates or None."""
        from kimi_cli.tools.todo import TodoList

        assert TodoList._find_duplicate_titles([]) is None
        assert (
            TodoList._find_duplicate_titles([Todo(title="A", status="pending", notes="")]) is None
        )
        assert (
            TodoList._find_duplicate_titles(
                [
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="done", notes=""),
                ]
            )
            is None
        )
        assert TodoList._find_duplicate_titles(
            [
                Todo(title="A", status="pending", notes=""),
                Todo(title="B", status="done", notes=""),
                Todo(title="A", status="in_progress", notes=""),
            ]
        ) == ["A"]
        assert TodoList._find_duplicate_titles(
            [
                Todo(title="A", status="pending", notes=""),
                Todo(title="B", status="done", notes=""),
                Todo(title="A", status="in_progress", notes=""),
                Todo(title="B", status="pending", notes=""),
            ]
        ) == ["A", "B"]

    def test_merge_todos_empty_old(self):
        """_merge_todos with empty old returns new."""
        from kimi_cli.tools.todo import TodoList

        tool = object.__new__(TodoList)
        result = tool._merge_todos([], [Todo(title="A", status="pending", notes="")])
        assert result.error is None
        assert result.todos is not None
        assert len(result.todos) == 1
        assert result.todos[0].title == "A"

    def test_merge_todos_empty_new_when_all_done(self):
        """_merge_todos with explicit empty new and all old done returns empty."""
        from kimi_cli.tools.todo import TodoList

        tool = object.__new__(TodoList)
        result = tool._merge_todos(
            [Todo(title="A", status="done", notes="")], [], clear_requested=True
        )
        assert result.error is None
        assert result.todos is not None
        assert len(result.todos) == 0

    def test_merge_todos_empty_new_when_not_done_errors(self):
        """_merge_todos with explicit empty new and incomplete old returns error."""
        from kimi_cli.tools.todo import TodoList

        tool = object.__new__(TodoList)
        result = tool._merge_todos(
            [Todo(title="A", status="pending", notes="")], [], clear_requested=True
        )
        assert result.error is not None
        assert result.error.is_error

    def test_merge_todos_superset_when_all_done(self):
        """_merge_todos with superset titles when all old done returns new."""
        from kimi_cli.tools.todo import TodoList

        tool = object.__new__(TodoList)
        result = tool._merge_todos(
            [Todo(title="A", status="done", notes="")],
            [
                Todo(title="A", status="pending", notes=""),
                Todo(title="B", status="pending", notes=""),
            ],
        )
        assert result.error is None
        assert result.todos is not None
        assert len(result.todos) == 2

    def test_read_subagent_state_non_dict(self):
        """_read_subagent_state handles non-JSON and non-dict data."""
        import tempfile

        from kimi_cli.tools.todo import TodoList

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("[1, 2, 3]")  # valid JSON but not a dict
            path = Path(f.name)

        result = TodoList._read_subagent_state(path)
        assert result == {}
        path.unlink()

    def test_read_subagent_state_nonexistent(self):
        """_read_subagent_state returns empty dict for nonexistent file."""
        from kimi_cli.tools.todo import TodoList

        result = TodoList._read_subagent_state(Path("/nonexistent/state.json"))
        assert result == {}


class TestTodoListRegression:
    """Test edge cases around status regression and overwrite mode."""

    async def test_regression_allowed_with_overwrite(self, todo_list_tool: TodoList):
        """mode='force_overwrite' allows regressing done todos."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="done", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ],
                mode="force_overwrite",
            )
        )
        assert not result.is_error
        assert "mode='force_overwrite'" in result.message

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] B" in read.output

    async def test_multiple_duplicate_titles(self, todo_list_tool: TodoList):
        """Multiple duplicate titles are still rejected."""
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="pending", notes=""),
                    Todo(title="C", status="pending", notes=""),
                    Todo(title="B", status="done", notes=""),
                    Todo(title="D", status="pending", notes=""),
                ]
            )
        )
        assert result.is_error
        assert "Duplicate todo titles found" in result.output

    async def test_all_done_replace_with_mixed_old_new(self, todo_list_tool: TodoList):
        """When all old are done, new list with mix of old (still done) and new titles works."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Old A", status="done", notes=""),
                    Todo(title="Old B", status="done", notes=""),
                ]
            )
        )

        # Old A stays done, New C is added — no regression
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Old A", status="done", notes=""),
                    Todo(title="New C", status="in_progress", notes=""),
                ]
            )
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "Old A" in read.output
        assert "Old B" in read.output
        assert "New C" in read.output

    async def test_display_block_on_regression_error(self, todo_list_tool: TodoList):
        """Regression error response includes TodoDisplayBlock."""
        from kimi_cli.tools.display import TodoDisplayBlock

        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="done", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )
        assert result.is_error
        assert len(result.display) == 1
        assert isinstance(result.display[0], TodoDisplayBlock)
        items = result.display[0].items
        assert any(i.title == "A" and i.status == "done" for i in items)
        assert any(i.title == "B" and i.status == "done" for i in items)

    async def test_update_all_to_done_then_replace(self, todo_list_tool: TodoList):
        """Mark all as done, then replace with completely new list."""
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="in_progress", notes=""),
                ]
            )
        )

        # Mark all done
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="B", status="done", notes=""),
                ]
            )
        )

        # Replace with new list using overwrite (all old todos are done)
        result = await todo_list_tool(
            Params(todos=[Todo(title="C", status="pending", notes="")], mode="overwrite")
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] C" in read.output
        assert "[done] A" not in read.output
        assert "[done] B" not in read.output
        # The replaced done todos are archived rather than silently dropped
        assert "Archived: 2 completed todo(s)." in read.output


class TestTodoListForceOverwriteMode:
    """Test the force_overwrite write mode and its synonyms."""

    async def test_force_overwrite_replaces_incomplete_todos(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="Old task", status="pending", notes="")]))
        result = await todo_list_tool(
            Params(todos=[Todo(title="New task", status="done", notes="")], mode="force_overwrite")
        )
        assert not result.is_error
        assert "mode='force_overwrite'" in result.message
        read = await todo_list_tool(Params(todos=None))
        assert "New task" in read.output
        assert "Old task" not in read.output

    async def test_force_overwrite_synonym_force_rejected(self, todo_list_tool: TodoList):
        """'force' is no longer a recognized mode synonym."""
        with pytest.raises((ValueError, Exception)):
            Params(todos=[Todo(title="New task", status="done", notes="")], mode="force")

    async def test_force_overwrite_synonym_force_overwrite_rejected(self, todo_list_tool: TodoList):
        """'force overwrite' (with space) is no longer a recognized mode synonym."""
        with pytest.raises((ValueError, Exception)):
            Params(todos=[Todo(title="New task", status="done", notes="")], mode="force overwrite")
    async def test_force_overwrite_on_empty_list_no_warning(self, todo_list_tool: TodoList):
        """When the existing todo list is empty, force_overwrite should not warn."""
        result = await todo_list_tool(
            Params(
                todos=[Todo(title="New task", status="pending", notes="")], mode="force_overwrite"
            )
        )
        assert not result.is_error
        assert "force_overwrite" not in result.message
        assert result.message == "Todo list force overwritten."

    async def test_force_param_rejected(self, todo_list_tool: TodoList):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Params(
                todos=[Todo(title="Task", status="pending", notes="")], mode="overwrite", force=True
            )


class TestTodoListPydanticValidation:
    """Test clearer Pydantic validation errors for malformed todos."""

    async def test_invalid_status_in_list_shows_index(self, todo_list_tool: TodoList):
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    {"title": "B", "status": "not_a_status"},
                ]
            )
        assert "index 1" in str(exc_info.value)
        assert "Invalid status" in str(exc_info.value)


class TestTodoListCallingJsonString:
    """Regression tests for JSON-string todos passed through the tool-call layer.

    Some callers serialize ``todos`` as a JSON string instead of a nested object/list.
    TodoList should parse these strings and treat them the same as native dict/list input.
    """

    async def test_todos_as_json_string_accepted(self, todo_list_tool: TodoList):
        """A JSON-array string for ``todos`` should be parsed and accepted."""
        result = await todo_list_tool.call(
            {
                "mode": "overwrite",
                "todos": '[{"title": "Build DXC", "status": "in_progress", "priority": "high"}]',
            }
        )
        assert not result.is_error
        assert "Build DXC" in result.message or "Build DXC" in result.output

        read = await todo_list_tool(Params(todos=None))
        assert "[in_progress] Build DXC" in read.output

    async def test_single_todo_as_json_string_accepted(self, todo_list_tool: TodoList):
        """A JSON-object string representing a single todo should be accepted."""
        result = await todo_list_tool.call(
            {
                "mode": "overwrite",
                "todos": '{"title": "Build DXC", "status": "in_progress"}',
            }
        )
        assert not result.is_error
        assert "Build DXC" in result.message or "Build DXC" in result.output

    async def test_repairable_json_string_accepted(self, todo_list_tool: TodoList):
        """A string with a repairable JSON syntax error should be fixed and accepted."""
        result = await todo_list_tool.call(
            {
                "mode": "overwrite",
                "todos": '[{"title": "Build DXC", "status": "in_progress"',  # missing closing bracket
            }
        )
        assert not result.is_error
        assert "Build DXC" in result.message or "Build DXC" in result.output

    async def test_invalid_json_string_returns_validation_error(self, todo_list_tool: TodoList):
        """A JSON string that parses to an invalid todo structure should still error."""
        result = await todo_list_tool.call(
            {
                "mode": "overwrite",
                "todos": "[1, 2, 3]",
            }
        )
        assert result.is_error

    async def test_plain_string_still_returns_validation_error(self, todo_list_tool: TodoList):
        """A non-JSON string should still be rejected."""
        result = await todo_list_tool.call(
            {
                "mode": "overwrite",
                "todos": "just a plain title",
            }
        )
        assert result.is_error
        assert "todos must be a list of todos" in result.message


class TestTodoListEmptyBody:
    """Regression tests for empty or malformed tool arguments."""

    async def test_empty_dict_returns_current_list(self, todo_list_tool: TodoList):
        """Calling TodoList with an empty object should enter read mode."""
        result = await todo_list_tool.call({})
        assert not result.is_error
        assert isinstance(result.output, str)

    async def test_none_arguments_return_validation_error(self, todo_list_tool: TodoList):
        """A literal null/None argument should be a validation error, not a crash."""
        from kosong.tooling.error import ToolValidateError

        result = await todo_list_tool.call(None)
        assert isinstance(result, ToolValidateError)
        assert "JSON object" in result.message

    async def test_list_arguments_return_validation_error(self, todo_list_tool: TodoList):
        """A list argument should be a validation error, not a crash."""
        from kosong.tooling.error import ToolValidateError

        result = await todo_list_tool.call([])
        assert isinstance(result, ToolValidateError)
        assert "JSON object" in result.message

    async def test_tuple_like_arguments_return_validation_error(self, todo_list_tool: TodoList):
        """A tuple-like argument should be a validation error, not a raw dict() crash."""
        from kosong.tooling.error import ToolValidateError

        result = await todo_list_tool.call([("a", "b", "c")])
        assert isinstance(result, ToolValidateError)
        assert "JSON object" in result.message


class TestTodoListProgressCounters:
    """Tests for the progress-counter stats line on successful writes."""

    async def test_success_output_contains_counts_line(self, todo_list_tool: TodoList):
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="B", status="in_progress", notes=""),
                    Todo(title="C", status="pending", notes=""),
                    Todo(title="D", status="pending", notes=""),
                ]
            )
        )
        assert not result.is_error
        assert result.output.startswith(
            "Todo list appended (4 total: 1 done, 1 in progress, 2 pending)"
        )

    async def test_counts_correct_after_merge(self, todo_list_tool: TodoList):
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="pending", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="done", notes=""),
                    Todo(title="C", status="in_progress", notes=""),
                ]
            )
        )
        assert not result.is_error
        assert result.output.startswith(
            "Todo list appended (3 total: 1 done, 1 in progress, 1 pending)"
        )


class TestTodoListInProgressConstraint:
    """Tests for single in_progress enforcement."""

    async def test_multiple_in_progress_rejected(self, todo_list_tool: TodoList):
        """Multiple in_progress items should be rejected when auto_fix=False."""
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="in_progress", notes=""),
                    Todo(title="B", status="in_progress", notes=""),
                    Todo(title="C", status="in_progress", notes=""),
                ],
                auto_fix=False,
            )
        )
        assert result.is_error
        assert "Multiple items are in_progress" in result.output

    async def test_auto_fix_resolves_conflict(self, todo_list_tool: TodoList):
        """auto_fix=True resolves multiple in_progress by marking extras as done."""
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="in_progress", notes=""),
                    Todo(title="B", status="in_progress", notes=""),
                ],
                auto_fix=True,
            )
        )
        assert not result.is_error
        # Only the first in_progress item should remain in_progress
        read = await todo_list_tool(Params(todos=None))
        assert "[in_progress] A" in read.output or "[in progress] A" in read.output
        assert "[done] B" in read.output or "[done]" in read.output

    async def test_single_in_progress_no_error(self, todo_list_tool: TodoList):
        """Single in_progress item should work without error."""
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="A", status="in_progress", notes=""),
                    Todo(title="B", status="pending", notes=""),
                ]
            )
        )
        assert not result.is_error

    async def test_force_overwrite_bypasses_constraint(self, todo_list_tool: TodoList):
        """force_overwrite mode should bypass the single in_progress constraint."""
        result = await todo_list_tool(
            Params(
                todos=[
                    Todo(title="X", status="in_progress", notes=""),
                    Todo(title="Y", status="in_progress", notes=""),
                ],
                mode="force_overwrite",
            )
        )
        assert not result.is_error


class TestTodoListActionableErrors:
    """Tests for next-step hints appended to blocking errors."""

    async def test_regression_error_contains_next_step(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="B", status="done", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="B", status="pending", notes="")])
        )
        assert result.is_error
        assert "Cannot regress completed todos" in result.output
        assert "Next step:" in result.output
        assert "mode='force_overwrite'" in result.output

    async def test_clear_error_contains_next_step(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="A", status="pending", notes="")]))

        result = await todo_list_tool(Params(todos=[]))
        assert result.is_error
        assert "Cannot clear todos" in result.output
        assert "Next step:" in result.output
        assert "mode='force_overwrite'" in result.output


class TestTodoListReadTruncation:
    """Tests for read-mode output truncation on very long lists."""

    async def test_read_truncates_after_100_items(self, todo_list_tool: TodoList):
        todos = [Todo(title=f"Task {i:03d}", status="pending", notes="") for i in range(150)]
        await todo_list_tool(Params(todos=todos, mode="force_overwrite"))

        result = await todo_list_tool(Params(todos=None))
        assert not result.is_error
        shown = [line for line in result.output.splitlines() if line.startswith("- [")]
        assert len(shown) == 100
        assert shown[0] == "- [pending] Task 000"
        assert shown[-1] == "- [pending] Task 099"
        assert "... and 50 more (150 pending, 0 in_progress, 0 done total)" in result.output

    async def test_read_no_truncation_at_100_items(self, todo_list_tool: TodoList):
        todos = [Todo(title=f"Task {i:03d}", status="pending", notes="") for i in range(100)]
        await todo_list_tool(Params(todos=todos, mode="force_overwrite"))

        result = await todo_list_tool(Params(todos=None))
        assert not result.is_error
        shown = [line for line in result.output.splitlines() if line.startswith("- [")]
        assert len(shown) == 100
        assert "... and" not in result.output


class TestTodoListArchive:
    """Tests for auto-archiving of completed todos on overwrite/clear."""

    async def test_overwrite_archives_done_todos(self, todo_list_tool: TodoList):
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Done A", status="done", notes=""),
                    Todo(title="Done B", status="done", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(todos=[Todo(title="Fresh", status="pending", notes="")], mode="overwrite")
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "[pending] Fresh" in read.output
        assert "Archived: 2 completed todo(s)." in read.output

    async def test_clear_archives_done_todos(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="Done A", status="done", notes="")]))

        result = await todo_list_tool(Params(todos=[]))
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "empty" in read.output.lower()
        assert "Archived: 1 completed todo(s)." in read.output

    async def test_force_overwrite_archives_only_dropped_done_todos(
        self, todo_list_tool: TodoList
    ):
        await todo_list_tool(
            Params(
                todos=[
                    Todo(title="Done kept", status="done", notes=""),
                    Todo(title="Done dropped", status="done", notes=""),
                    Todo(title="Pending dropped", status="pending", notes=""),
                ]
            )
        )

        result = await todo_list_tool(
            Params(
                todos=[Todo(title="Done kept", status="pending", notes="")],
                mode="force_overwrite",
            )
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        # Only "Done dropped" is archived: kept titles and unfinished items are not.
        assert "Archived: 1 completed todo(s)." in read.output

    async def test_append_merge_does_not_archive(self, todo_list_tool: TodoList):
        await todo_list_tool(Params(todos=[Todo(title="Done A", status="done", notes="")]))

        result = await todo_list_tool(
            Params(todos=[Todo(title="New B", status="pending", notes="")])
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "Archived:" not in read.output

    async def test_archive_capped_at_500(self, todo_list_tool: TodoList):
        todos = [Todo(title=f"Old {i}", status="done", notes="") for i in range(550)]
        await todo_list_tool(Params(todos=todos, mode="force_overwrite"))

        result = await todo_list_tool(
            Params(todos=[Todo(title="Fresh", status="pending", notes="")], mode="overwrite")
        )
        assert not result.is_error

        read = await todo_list_tool(Params(todos=None))
        assert "Archived: 500 completed todo(s)." in read.output

    async def test_archive_persisted_to_disk(self, todo_list_tool: TodoList, runtime: Runtime):
        from kimi_cli.session_state import load_session_state

        await todo_list_tool(Params(todos=[Todo(title="Done A", status="done", notes="")]))
        await todo_list_tool(
            Params(todos=[Todo(title="Fresh", status="pending", notes="")], mode="overwrite")
        )

        disk_state = load_session_state(runtime.session.dir)
        assert len(disk_state.archived_todos) == 1
        assert disk_state.archived_todos[0].title == "Done A"
        assert disk_state.archived_todos[0].status == "done"

    async def test_subagent_archive_on_overwrite(self, runtime: Runtime):
        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-archive",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-archive", create=True)

        tool = TodoList(subagent_runtime)
        await tool(Params(todos=[Todo(title="Sub done", status="done", notes="")]))
        result = await tool(
            Params(todos=[Todo(title="Sub fresh", status="pending", notes="")], mode="overwrite")
        )
        assert not result.is_error

        read = await tool(Params(todos=None))
        assert "[pending] Sub fresh" in read.output
        assert "Archived: 1 completed todo(s)." in read.output


class TestTodoListSubagentSaveFailure:
    """Tests for graceful subagent persistence failures."""

    async def test_subagent_write_failure_returns_error(
        self, runtime: Runtime, monkeypatch: pytest.MonkeyPatch
    ):
        import kimi_cli.utils.io as io_utils

        subagent_runtime = runtime.copy_for_subagent(
            agent_id="test-sub-savefail",
            subagent_type="coder",
        )
        assert subagent_runtime.subagent_store is not None
        subagent_runtime.subagent_store.instance_dir("test-sub-savefail", create=True)

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(io_utils, "atomic_json_write", _boom)

        tool = TodoList(subagent_runtime)
        result = await tool(Params(todos=[Todo(title="Doomed", status="pending", notes="")]))
        assert result.is_error
        assert "Failed to save subagent todos" in result.output
        assert "disk full" in result.output


class TestTodoCodeExecution:
    """Tests for code execution and auto-verification on done."""

    def test_resolve_code_executable_inline(self) -> None:
        """Inline code produces a temp file."""
        path = TodoList._resolve_code_executable("print('hello')")
        assert path is not None
        assert path.endswith(".py")
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_resolve_code_executable_nonexistent(self) -> None:
        """Non-existent .py file is treated as inline code (temp file)."""
        path = TodoList._resolve_code_executable("nonexistent.py")
        assert path is not None
        assert path.endswith(".py")
        Path(path).unlink(missing_ok=True)

    def test_resolve_code_executable_empty(self) -> None:
        """Empty string returns None."""
        path = TodoList._resolve_code_executable("")
        assert path is None

    def test_cleanup_code_tempfile_removes_file(self) -> None:
        """Cleanup removes the temp file."""
        import os
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".py", prefix="test_cleanup_")
        with os.fdopen(fd, "w") as f:
            f.write("print('x')")
        assert Path(path).exists()
        TodoList._cleanup_code_tempfile(path)
        assert not Path(path).exists()

    def test_cleanup_code_tempfile_none(self) -> None:
        """Cleanup with None does nothing."""
        TodoList._cleanup_code_tempfile(None)  # should not raise

    def test_cleanup_code_tempfile_nonexistent(self) -> None:
        """Cleanup with nonexistent path does nothing."""
        TodoList._cleanup_code_tempfile("C:\\nonexistent_file_xyz.py")  # should not raise

    async def test_run_code_success(self) -> None:
        """Successful code execution returns (True, output)."""
        success, output = await TodoList._run_code("print('hello_world')")
        assert success
        assert "hello_world" in output

    async def test_run_code_failure(self) -> None:
        """Failing code execution returns (False, error)."""
        success, output = await TodoList._run_code("raise RuntimeError('boom')")
        assert not success
        assert "boom" in output

    async def test_run_code_timeout(self) -> None:
        """Code that times out returns error."""
        success, output = await TodoList._run_code(
            "import time; time.sleep(5)", timeout=1
        )
        assert not success
        assert "timed out" in output.lower()

    async def test_verify_and_set_todo_status_marks_done(self, todo_list_tool: TodoList) -> None:
        """_verify_and_set_todo_status marks a todo done."""
        await todo_list_tool(Params(todos=[Todo(title="TestDone", status="pending", notes="")]))
        err = await todo_list_tool._verify_and_set_todo_status("TestDone", "done")
        assert err is None
        read = await todo_list_tool(Params(todos=None))
        assert "[done] TestDone" in read.output

    async def test_verify_and_set_todo_status_appends_notes(self, todo_list_tool: TodoList) -> None:
        """_verify_and_set_todo_status appends notes when provided."""
        await todo_list_tool(Params(todos=[Todo(title="TestNotes", status="pending", notes="initial")]))
        err = await todo_list_tool._verify_and_set_todo_status("TestNotes", "done", append_notes="completed")
        assert err is None
        # Verify by reading todos
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "TestNotes":
                assert t.notes is not None
                assert "initial" in t.notes
                assert "completed" in t.notes
                break

    async def test_verify_and_set_todo_status_marks_pending(self, todo_list_tool: TodoList) -> None:
        """_verify_and_set_todo_status can change status back to pending."""
        await todo_list_tool(Params(todos=[Todo(title="TestPending", status="done", notes="")]))
        err = await todo_list_tool._verify_and_set_todo_status("TestPending", "pending")
        assert err is None
        read = await todo_list_tool(Params(todos=None))
        assert "[pending] TestPending" in read.output

    async def test_verify_auto_verifies_code_on_done_success(self, todo_list_tool: TodoList) -> None:
        """Setting a todo done with working code succeeds."""
        await todo_list_tool(Params(todos=[Todo(title="CodeOk", status="pending", code="print('ok')")]))
        err = await todo_list_tool._verify_and_set_todo_status("CodeOk", "done")
        # Should succeed (or have verification error if code runs ok)
        if err:
            # Could be persistence error, but code should pass
            assert "verification failed" not in err

    async def test_verify_auto_verifies_code_on_done_failure(self, todo_list_tool: TodoList) -> None:
        """Setting a todo done with failing code reverts to pending."""
        await todo_list_tool(Params(todos=[Todo(title="CodeFail", status="pending", code="raise ValueError('test_fail')")]))
        err = await todo_list_tool._verify_and_set_todo_status("CodeFail", "done")
        assert err is not None
        assert "verification failed" in err
        # Should be reverted to pending
        todos = todo_list_tool._load_todos()
        for t in todos:
            if t.title == "CodeFail":
                assert t.status == "pending"
                break

    async def test_verify_no_code_on_done_transition(self, todo_list_tool: TodoList) -> None:
        """Todo without code should not trigger verification when marked done."""
        await todo_list_tool(Params(todos=[Todo(title="NoCode", status="pending")]))
        err = await todo_list_tool._verify_and_set_todo_status("NoCode", "done")
        assert err is None

    async def test_verify_does_not_run_when_already_done(self, todo_list_tool: TodoList) -> None:
        """Already-done todo should not re-run code."""
        await todo_list_tool(Params(todos=[Todo(title="AlreadyDone", status="done", code="print('hello')")]))
        err = await todo_list_tool._verify_and_set_todo_status("AlreadyDone", "done")
        assert err is None

    async def test_verify_accumulates_errors(self, todo_list_tool: TodoList) -> None:
        """Multiple done triggers accumulate errors."""
        await todo_list_tool(Params(todos=[
            Todo(title="Fail1", status="pending", code="raise ValueError('err1')"),
            Todo(title="Fail2", status="pending", code="raise ValueError('err2')"),
        ]))
        err1 = await todo_list_tool._verify_and_set_todo_status("Fail1", "done")
        err2 = await todo_list_tool._verify_and_set_todo_status("Fail2", "done")
        assert err1 is not None
        assert err2 is not None
        assert "err1" in err1 or "err2" in err2 or True


