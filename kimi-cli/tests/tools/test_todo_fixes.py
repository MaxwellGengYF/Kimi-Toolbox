"""Tests for the todo tool usability fixes.

Covers the behavior of the simplified todo toolset (kimi_cli.tools.todo):

1. todo_update(parent=...) child creation and bare same-title calls preserve
   status; done items need force=True to regress.
2. mode='clear' is explicit; an empty todos=[] in append mode is a no-op.
3. todo_write writes merge root-level titles only, and warn (non-blocking) when
   a new root title already exists deeper in the tree.
4. todo_update(complete=True) marks a todo and all its sub-todos done in one
   call (replaces the removed todo_pop).
5. Maximum tree nesting depth (todo_max_layers + 1) is enforced at write time.

Mirrors test_todo_stack.py style: async tests with the ``runtime`` fixture;
tools are instantiated directly as ``TodoList(runtime)`` / ``todo_update(runtime)``.
"""

from __future__ import annotations

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.todo import (
    Params,
    Todo,
    TodoList,
    TodoUpdateParams,
    todo_update,
)


def _read_root_todo(tool: TodoList, title: str) -> Todo:
    """Return the root-level todo with ``title`` from persisted state."""
    for t in tool._load_todos():
        if t.content == title:
            return t
    raise AssertionError(f"todo {title!r} not found")


# ---------------------------------------------------------------------------
# Fix 1: status preservation + regression guard (force param)
# ---------------------------------------------------------------------------


class TestTodoUpdateStatusPreservationAndRegression:
    """Bare same-title calls must not reset status; done items need force=True."""

    async def test_bare_same_title_call_preserves_status(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", status="in_progress", notes="keep"))

        # Bare same-title call (no status) must NOT reset to pending.
        res = await update(TodoUpdateParams(parent="Parent", title="child"))
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "in_progress"
        assert child.notes == "keep"

    async def test_bare_same_title_call_on_done_preserves_done(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", status="done"))

        res = await update(TodoUpdateParams(parent="Parent", title="child"))
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "done"

    async def test_regress_done_to_pending_errors_without_force(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", status="done"))

        res = await update(TodoUpdateParams(parent="Parent", title="child", status="pending"))
        assert res.is_error
        assert "Cannot regress completed todo" in res.output
        assert "force=True" in res.output
        # State unchanged after the failed write.
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "done"

    async def test_regress_done_to_in_progress_errors_without_force(
        self, runtime: Runtime
    ) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", status="done"))

        res = await update(TodoUpdateParams(parent="Parent", title="child", status="in_progress"))
        assert res.is_error
        assert "Cannot regress completed todo" in res.output

    async def test_regress_done_with_force_succeeds(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", status="done"))

        res = await update(
            TodoUpdateParams(parent="Parent", title="child", status="pending", force=True)
        )
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "pending"

    async def test_explicit_pending_on_pending_stays_pending(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child"))

        res = await update(TodoUpdateParams(parent="Parent", title="child", status="pending"))
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "pending"

    async def test_rename_done_item_without_status_change_ok(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="old", status="done"))

        res = await update(TodoUpdateParams(parent="Parent", title="old", rename_to="new"))
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.content == "new"
        assert child.status == "done"


# ---------------------------------------------------------------------------
# Fix 4: complete=True finishes a subtree in one call
# ---------------------------------------------------------------------------


class TestCompleteSubtreeGuard:
    async def test_complete_marks_unfinished_items(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="c1"))  # pending
        await update(TodoUpdateParams(parent="Parent", title="c2", status="in_progress"))

        res = await update(TodoUpdateParams(title="Parent", complete=True))
        assert not res.is_error
        assert "completed with 3 sub-todos marked done" in res.output
        # Existing message contract preserved.
        assert res.message == 'Updated "Parent".'
        parent = _read_root_todo(update, "Parent")
        assert parent.status == "done"
        assert [c.status for c in parent.children] == ["done", "done"]

    async def test_complete_all_done_is_noop_style_success(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="done")]))
        await update(TodoUpdateParams(parent="Parent", title="c1", status="done"))
        await update(TodoUpdateParams(parent="Parent", title="c2", status="done"))

        res = await update(TodoUpdateParams(title="Parent", complete=True))
        assert not res.is_error
        assert "completed with 3 sub-todos marked done" in res.output
        parent = _read_root_todo(update, "Parent")
        assert parent.status == "done"
        assert all(c.status == "done" for c in parent.children)


# ---------------------------------------------------------------------------
# Fix 2: explicit mode='clear' + todos=[] no-op
# ---------------------------------------------------------------------------


class TestTodoListClearModeAndNoop:
    async def test_clear_mode_with_todos_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=[Todo(content="B", status="pending")], mode="clear"))
        assert res.is_error
        assert "mode='clear' cannot be combined with todos" in res.output

    async def test_clear_mode_noop_on_empty_list(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        res = await lst(Params(todos=[], mode="clear"))
        assert not res.is_error
        assert res.output == "Todo list cleared (0 total: 0 done, 0 in progress, 0 pending)"

    async def test_clear_mode_archives_done(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(
            Params(todos=[Todo(content="D1", status="done"), Todo(content="D2", status="done")])
        )
        res = await lst(Params(todos=[], mode="clear"))
        assert not res.is_error
        read = await lst(Params(todos=None))
        assert "Archived: 2 completed todo(s)." in read.output

    async def test_append_empty_list_noop_shows_current_state(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="in_progress")]))
        res = await lst(Params(todos=[]))
        assert not res.is_error
        assert "unchanged" in res.output
        assert "- [in progress] A" in res.output
        # Display block reflects the current (unchanged) state.
        assert len(res.display) == 1

    async def test_clear_error_names_force_escape_hatch(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=[], mode="clear"))
        assert res.is_error
        assert "force=True" in res.output

    async def test_clear_with_force_discards_unfinished(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=[], mode="clear", force=True))
        assert not res.is_error
        read = await lst(Params(todos=None))
        assert "empty" in read.output.lower()


# ---------------------------------------------------------------------------
# Fix 3: todo_write root-scope merge warns on nested scope duplicates
# ---------------------------------------------------------------------------


class TestTodoListScopeDuplicateWarning:
    async def test_append_nested_title_warns(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="Parent"))
        await update(TodoUpdateParams(parent="Parent", title="child"))
        # child exists only under Parent; appending it at root must warn.
        res = await lst(Params(todos=[Todo(content="child", status="done")]))
        assert not res.is_error
        assert '"child" already exists in the tree (under "Parent")' in res.message
        # The warning is non-blocking: the new root item IS appended.
        todos = lst._load_todos()
        assert [t.content for t in todos] == ["Parent", "child"]

    async def test_append_root_title_no_warning(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="Parent"))
        await update(TodoUpdateParams(parent="Parent", title="child"))
        # Updating the root title Parent merges in place — no scope warning.
        res = await lst(Params(todos=[Todo(content="Parent", status="done")]))
        assert not res.is_error
        assert "already exists in the tree" not in res.message

    async def test_append_deep_nested_title_warns_with_path(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="A"))
        await update(TodoUpdateParams(parent="A", title="B"))
        await update(TodoUpdateParams(parent="B", title="deep"))
        res = await lst(Params(todos=[Todo(content="deep", status="done")]))
        assert not res.is_error
        assert '"deep" already exists in the tree (under "A > B")' in res.message


# ---------------------------------------------------------------------------
# Fix 5: max tree depth enforced at write time
# ---------------------------------------------------------------------------


class TestTodoListDepthCap:
    async def test_default_max_depth_5_allowed(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        # depth 5 = max_layers(4) + one todo_update(parent=...) level under the deepest parent.
        deep = Todo(
            content="L1",
            status="pending",
            children=[
                Todo(
                    content="L2",
                    status="pending",
                    children=[
                        Todo(
                            content="L3",
                            status="pending",
                            children=[
                                Todo(
                                    content="L4",
                                    status="pending",
                                    children=[Todo(content="L5", status="pending")],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        res = await lst(Params(todos=[deep]))
        assert not res.is_error

    async def test_depth_6_rejected(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        deep = Todo(
            content="L1",
            status="pending",
            children=[
                Todo(
                    content="L2",
                    status="pending",
                    children=[
                        Todo(
                            content="L3",
                            status="pending",
                            children=[
                                Todo(
                                    content="L4",
                                    status="pending",
                                    children=[
                                        Todo(
                                            content="L5",
                                            status="pending",
                                            children=[Todo(content="L6", status="pending")],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        res = await lst(Params(todos=[deep]))
        assert res.is_error
        assert "maximum nesting depth" in res.output
        assert "5 levels" in res.output
        # Nothing persisted.
        read = await lst(Params(todos=None))
        assert "empty" in read.output.lower()

    async def test_depth_cap_honors_config(self, runtime: Runtime) -> None:
        runtime.config.loop_control.todo_max_layers = 1
        lst = TodoList(runtime)
        # max_depth = max_layers(1) + 1 = 2; a 3-deep tree is rejected.
        deep = Todo(
            content="L1",
            status="pending",
            children=[
                Todo(
                    content="L2",
                    status="pending",
                    children=[Todo(content="L3", status="pending")],
                )
            ],
        )
        res = await lst(Params(todos=[deep]))
        assert res.is_error
        assert "maximum nesting depth of 2 levels" in res.output

    async def test_depth_cap_applies_to_replace_with_force(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        deep = Todo(
            content="L1",
            status="pending",
            children=[
                Todo(
                    content="L2",
                    status="pending",
                    children=[
                        Todo(
                            content="L3",
                            status="pending",
                            children=[
                                Todo(
                                    content="L4",
                                    status="pending",
                                    children=[
                                        Todo(
                                            content="L5",
                                            status="pending",
                                            children=[Todo(content="L6", status="pending")],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        res = await lst(Params(todos=[deep], mode="replace", force=True))
        assert res.is_error
        assert "maximum nesting depth" in res.output


class TestTodoUpdateDepthGuard:
    """Defensive depth guard: todo_update cannot add below max_layers + 1."""

    async def test_update_depth_guard_with_limited_layers(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        # Simulate a runtime where the layer budget was later reduced.
        runtime.config.loop_control.todo_max_layers = 0
        res = await update(TodoUpdateParams(parent="A", title="child"))
        assert res.is_error
        assert "Cannot add children deeper than 1 layers" in res.output

    async def test_update_depth_guard_allows_at_limit(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        # Build depth 4 via nested todo_write tree.
        deep = Todo(
            content="A",
            status="pending",
            children=[
                Todo(
                    content="B",
                    status="pending",
                    children=[
                        Todo(
                            content="C",
                            status="pending",
                            children=[Todo(content="D", status="pending")],
                        )
                    ],
                )
            ],
        )
        await lst(Params(todos=[deep]))
        # todo_update(parent=...) may still add one level under the deepest parent.
        res = await update(TodoUpdateParams(parent="D", title="leaf"))
        assert not res.is_error
        node = _read_root_todo(lst, "A").children[0].children[0].children[0]
        assert [c.content for c in node.children] == ["leaf"]
