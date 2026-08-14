"""Tests for todo_update lightweight single-todo edits."""

from __future__ import annotations

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.display import TodoDisplayBlock
from kimi_cli.tools.todo import (
    Params,
    Todo,
    TodoList,
    todo_update,
    TodoUpdateParams,
)


def _find_todo(tool: todo_update, title: str) -> Todo:
    """Return the first todo with ``title`` from persisted state."""
    for t in tool._load_todos():
        if t.content == title:
            return t
    raise AssertionError(f"todo {title!r} not found")


class TestTodoUpdateBasics:
    async def test_update_status_to_done(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="in_progress")]))

        res = await update(TodoUpdateParams(title="Task A", status="done"))
        assert not res.is_error
        assert 'Updated "Task A" (status=done)' in res.output
        assert res.message == 'Updated "Task A".'

        todo = _find_todo(update, "Task A")
        assert todo.status == "done"

    async def test_update_status_to_in_progress(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(content="Task A", status="pending"),
                    Todo(content="Task B", status="done"),
                ]
            )
        )

        res = await update(TodoUpdateParams(title="Task A", status="in_progress"))
        assert not res.is_error
        todo = _find_todo(update, "Task A")
        assert todo.status == "in_progress"

    async def test_omitted_status_preserves_existing(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="in_progress")]))

        res = await update(TodoUpdateParams(title="Task A", notes="new note"))
        assert not res.is_error
        todo = _find_todo(update, "Task A")
        assert todo.status == "in_progress"
        assert todo.notes == "new note"

    async def test_update_notes_clears_with_empty_string(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="pending", notes="old")]))

        res = await update(TodoUpdateParams(title="Task A", notes=""))
        assert not res.is_error
        todo = _find_todo(update, "Task A")
        assert todo.notes is None


class TestTodoUpdateRename:
    async def test_rename_root_todo(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Old", status="pending")]))

        res = await update(TodoUpdateParams(title="Old", rename_to="New"))
        assert not res.is_error
        assert 'Updated "Old" (renamed to "New")' in res.output
        assert _find_todo(update, "New").status == "pending"

    async def test_rename_collision_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(content="A", status="pending"),
                    Todo(content="B", status="pending"),
                ]
            )
        )

        res = await update(TodoUpdateParams(title="A", rename_to="B"))
        assert res.is_error
        assert 'Cannot rename "A" to "B"' in res.output

        # Nothing changed.
        assert _find_todo(update, "A").content == "A"


class TestTodoUpdateFuzzy:
    async def test_fuzzy_match_finds_near_title(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Implement feature", status="pending")]))

        res = await update(TodoUpdateParams(title="implement feature", status="done"))
        assert not res.is_error
        assert 'Fuzzy matched' in res.output
        todo = _find_todo(update, "Implement feature")
        assert todo.status == "done"

    async def test_fuzzy_disabled_errors_when_exact_missing(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="pending")]))

        res = await update(
            TodoUpdateParams(title="task a", status="done", fuzzy=False)
        )
        assert res.is_error
        assert 'No todo titled "task a" found' in res.output


class TestTodoUpdateRegression:
    async def test_regression_blocked_without_force(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="done")]))

        res = await update(TodoUpdateParams(title="Task A", status="in_progress"))
        assert res.is_error
        assert "Cannot regress completed todo" in res.output
        assert _find_todo(update, "Task A").status == "done"

    async def test_regression_allowed_with_force(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="done")]))

        res = await update(
            TodoUpdateParams(title="Task A", status="in_progress", force=True)
        )
        assert not res.is_error
        assert _find_todo(update, "Task A").status == "in_progress"


class TestTodoUpdateInProgressConstraint:
    async def test_auto_fixes_multiple_in_progress(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(content="Task A", status="in_progress"),
                    Todo(content="Task B", status="pending"),
                ]
            )
        )

        res = await update(TodoUpdateParams(title="Task B", status="in_progress"))
        assert not res.is_error
        a = _find_todo(update, "Task A")
        b = _find_todo(update, "Task B")
        assert a.status == "done"
        assert b.status == "in_progress"
        assert "Auto-fixed" in res.output


class TestTodoUpdateTreeSearch:
    async def test_updates_nested_todo(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(
                        content="Parent",
                        status="pending",
                        children=[Todo(content="Child", status="pending")],
                    )
                ]
            )
        )

        res = await update(TodoUpdateParams(title="Child", status="done"))
        assert not res.is_error
        parent = _find_todo(update, "Parent")
        assert parent.children[0].status == "done"

    async def test_empty_tree_errors(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        res = await update(TodoUpdateParams(title="Task A", status="done"))
        assert res.is_error
        assert "No todos exist" in res.output


class TestTodoUpdateStack:
    async def test_rename_heals_stack_breadcrumb(self, runtime: Runtime) -> None:
        from kimi_cli.tools.todo import todo_push, TodoPushParams

        await todo_push(runtime)(TodoPushParams(title="Parent"))
        await todo_push(runtime)(TodoPushParams(title="Child"))

        update = todo_update(runtime)
        res = await update(
            TodoUpdateParams(title="Child", rename_to="Renamed Child")
        )
        assert not res.is_error
        assert update._load_stack() == ["Parent", "Renamed Child"]


class TestTodoUpdateDisplay:
    async def test_returns_display_block(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Task A", status="pending")]))

        res = await update(TodoUpdateParams(title="Task A", status="done"))
        assert not res.is_error
        assert len(res.display) == 1
        assert isinstance(res.display[0], TodoDisplayBlock)


class TestTodoUpdateParent:
    async def test_creates_child_under_parent(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))

        res = await update(TodoUpdateParams(parent="Parent", title="Child"))
        assert not res.is_error
        assert 'Created "Child" under "Parent".' in res.output
        parent = _find_todo(update, "Parent")
        assert [c.content for c in parent.children] == ["Child"]
        assert parent.children[0].status == "pending"

    async def test_updates_existing_child_under_parent(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(
                        content="Parent",
                        status="pending",
                        children=[Todo(content="Child", status="pending")],
                    )
                ]
            )
        )

        res = await update(
            TodoUpdateParams(parent="Parent", title="Child", status="done")
        )
        assert not res.is_error
        assert 'Updated "Child" (status=done)' in res.output
        parent = _find_todo(update, "Parent")
        assert parent.children[0].status == "done"

    async def test_creates_root_child_with_empty_parent(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        await TodoList(runtime)(Params(todos=[Todo(content="Existing", status="pending")]))

        res = await update(TodoUpdateParams(parent="", title="New Root"))
        assert not res.is_error
        assert 'Created "New Root" under "root".' in res.output
        assert [t.content for t in update._load_todos()] == ["Existing", "New Root"]

    async def test_missing_parent_errors(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        await TodoList(runtime)(Params(todos=[Todo(content="A", status="pending")]))

        res = await update(TodoUpdateParams(parent="Missing", title="Child"))
        assert res.is_error
        assert 'No parent todo matching "Missing" found' in res.output

    async def test_parent_scoped_lookup_does_not_match_outside_parent(
        self, runtime: Runtime
    ) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(
                        content="P1",
                        status="pending",
                        children=[Todo(content="Child", status="pending")],
                    ),
                    Todo(
                        content="P2",
                        status="pending",
                        children=[Todo(content="Child", status="pending")],
                    ),
                ]
            )
        )

        res = await update(
            TodoUpdateParams(parent="P2", title="Child", status="done")
        )
        assert not res.is_error
        p1 = _find_todo(update, "P1")
        p2 = _find_todo(update, "P2")
        assert p1.children[0].status == "pending"
        assert p2.children[0].status == "done"
