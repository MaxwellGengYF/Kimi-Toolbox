"""Tests for the todo tool usability fixes.

Covers the fixes applied to kimi_cli.tools.todo:

1. TodoSub bare same-title calls preserve status (no fake "read" that resets
   items to pending); done items need force=True to regress.
2. mode='clear' is explicit; an empty todos=[] in append mode is a no-op.
3. TodoList writes merge root-level titles only, and warn (non-blocking) when
   a new root title already exists deeper in the tree.
4. TodoPop warns when it marks unfinished items done.
5. Maximum tree nesting depth (todo_max_layers + 1) is enforced at write time.

Mirrors test_todo_stack.py style: async tests with the ``runtime`` fixture;
tools are instantiated directly as ``TodoPush(runtime)`` etc.
"""

from __future__ import annotations

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.todo import (
    Params,
    Todo,
    TodoList,
    TodoPop,
    TodoPopParams,
    TodoPush,
    TodoPushParams,
    TodoSub,
    TodoSubParams,
)


def _read_root_todo(tool: TodoList, title: str) -> Todo:
    """Return the root-level todo with ``title`` from persisted state."""
    for t in tool._load_todos():
        if t.title == title:
            return t
    raise AssertionError(f"todo {title!r} not found")


# ---------------------------------------------------------------------------
# Fix 1: TodoSub status preservation + regression guard (force param)
# ---------------------------------------------------------------------------


class TestTodoSubStatusPreservationAndRegression:
    """Bare TodoSub calls must not reset status; done items need force=True."""

    async def test_bare_same_title_call_preserves_status(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", status="in_progress", notes="keep"))

        # Bare same-title call (no status) must NOT reset to pending.
        res = await sub(TodoSubParams(title="child"))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "in_progress"
        assert child.notes == "keep"

    async def test_bare_same_title_call_on_done_preserves_done(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", status="done"))

        res = await sub(TodoSubParams(title="child"))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "done"

    async def test_regress_done_to_pending_errors_without_force(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", status="done"))

        res = await sub(TodoSubParams(title="child", status="pending"))
        assert res.is_error
        assert "Cannot regress completed todo" in res.output
        assert "force=True" in res.output
        # State unchanged after the failed write.
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "done"

    async def test_regress_done_to_in_progress_errors_without_force(
        self, runtime: Runtime
    ) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", status="done"))

        res = await sub(TodoSubParams(title="child", status="in_progress"))
        assert res.is_error
        assert "Cannot regress completed todo" in res.output

    async def test_regress_done_with_force_succeeds(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", status="done"))

        res = await sub(TodoSubParams(title="child", status="pending", force=True))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "pending"

    async def test_explicit_pending_on_pending_stays_pending(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child"))

        res = await sub(TodoSubParams(title="child", status="pending"))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "pending"

    async def test_rename_done_item_without_status_change_ok(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="old", status="done"))

        res = await sub(TodoSubParams(title="old", rename_to="new"))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.title == "new"
        assert child.status == "done"


# ---------------------------------------------------------------------------
# Fix 4: TodoPop warns when it marks unfinished items done
# ---------------------------------------------------------------------------


class TestTodoPopUnfinishedWarning:
    async def test_pop_warns_when_marking_unfinished_items(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        pop = TodoPop(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="c1"))  # pending
        await sub(TodoSubParams(title="c2", status="in_progress"))

        res = await pop(TodoPopParams())
        assert not res.is_error
        assert "Note:" in res.output
        assert "had 3 unfinished item(s)" in res.output
        assert "force=True" in res.output
        # Existing message contract preserved.
        assert res.message == 'Popped "Parent".'

    async def test_pop_all_done_no_warning(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        pop = TodoPop(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="c1", status="done"))
        await sub(TodoSubParams(title="c2", status="done"))
        # Mark the parent done too via TodoList root merge.
        await lst(Params(todos=[Todo(title="Parent", status="done")]))

        res = await pop(TodoPopParams())
        assert not res.is_error
        assert 'Popped "Parent" — 3 sub-todo(s) marked done.' in res.output
        assert "Note:" not in res.output


# ---------------------------------------------------------------------------
# Fix 2: explicit mode='clear' + todos=[] no-op
# ---------------------------------------------------------------------------


class TestTodoListClearModeAndNoop:
    async def test_clear_mode_with_todos_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(title="A", status="pending")]))
        res = await lst(Params(todos=[Todo(title="B", status="pending")], mode="clear"))
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
            Params(todos=[Todo(title="D1", status="done"), Todo(title="D2", status="done")])
        )
        res = await lst(Params(todos=[], mode="clear"))
        assert not res.is_error
        read = await lst(Params(todos=None))
        assert "Archived: 2 completed todo(s)." in read.output

    async def test_append_empty_list_noop_shows_current_state(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(title="A", status="in_progress")]))
        res = await lst(Params(todos=[]))
        assert not res.is_error
        assert "unchanged" in res.output
        assert "- [in progress] A" in res.output
        # Display block reflects the current (unchanged) state.
        assert len(res.display) == 1

    async def test_clear_error_names_force_overwrite_escape_hatch(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(title="A", status="pending")]))
        res = await lst(Params(todos=[], mode="clear"))
        assert res.is_error
        assert "mode='force_overwrite'" in res.output


# ---------------------------------------------------------------------------
# Fix 3: TodoList root-scope merge warns on nested scope duplicates
# ---------------------------------------------------------------------------


class TestTodoListScopeDuplicateWarning:
    async def test_append_nested_title_warns(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child"))
        # child exists only under Parent; appending it at root must warn.
        res = await lst(Params(todos=[Todo(title="child", status="done")]))
        assert not res.is_error
        assert '"child" already exists in the tree (under "Parent")' in res.message
        # The warning is non-blocking: the new root item IS appended.
        todos = lst._load_todos()
        assert [t.title for t in todos] == ["Parent", "child"]

    async def test_append_root_title_no_warning(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child"))
        # Updating the root title Parent merges in place — no scope warning.
        res = await lst(Params(todos=[Todo(title="Parent", status="done")]))
        assert not res.is_error
        assert "already exists in the tree" not in res.message

    async def test_append_deep_nested_title_warns_with_path(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await push(TodoPushParams(title="B"))
        await sub(TodoSubParams(title="deep"))
        res = await lst(Params(todos=[Todo(title="deep", status="done")]))
        assert not res.is_error
        assert '"deep" already exists in the tree (under "A > B")' in res.message


# ---------------------------------------------------------------------------
# Fix 5: max tree depth enforced at write time
# ---------------------------------------------------------------------------


class TestTodoListDepthCap:
    async def test_default_max_depth_5_allowed(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        # depth 5 = max_layers(4) + one TodoSub level under the deepest push.
        deep = Todo(
            title="L1",
            status="pending",
            children=[
                Todo(
                    title="L2",
                    status="pending",
                    children=[
                        Todo(
                            title="L3",
                            status="pending",
                            children=[
                                Todo(
                                    title="L4",
                                    status="pending",
                                    children=[Todo(title="L5", status="pending")],
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
            title="L1",
            status="pending",
            children=[
                Todo(
                    title="L2",
                    status="pending",
                    children=[
                        Todo(
                            title="L3",
                            status="pending",
                            children=[
                                Todo(
                                    title="L4",
                                    status="pending",
                                    children=[
                                        Todo(
                                            title="L5",
                                            status="pending",
                                            children=[Todo(title="L6", status="pending")],
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
            title="L1",
            status="pending",
            children=[
                Todo(
                    title="L2",
                    status="pending",
                    children=[Todo(title="L3", status="pending")],
                )
            ],
        )
        res = await lst(Params(todos=[deep]))
        assert res.is_error
        assert "maximum nesting depth of 2 levels" in res.output

    async def test_depth_cap_applies_to_force_overwrite(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        deep = Todo(
            title="L1",
            status="pending",
            children=[
                Todo(
                    title="L2",
                    status="pending",
                    children=[
                        Todo(
                            title="L3",
                            status="pending",
                            children=[
                                Todo(
                                    title="L4",
                                    status="pending",
                                    children=[
                                        Todo(
                                            title="L5",
                                            status="pending",
                                            children=[Todo(title="L6", status="pending")],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        res = await lst(Params(todos=[deep], mode="force_overwrite"))
        assert res.is_error
        assert "maximum nesting depth" in res.output


class TestTodoSubDepthGuard:
    """Defensive depth guard: TodoSub cannot add below max_layers + 1."""

    async def test_sub_depth_guard_with_limited_layers(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        await push(TodoPushParams(title="A"))
        # Simulate a runtime where the layer budget was later reduced below the
        # current stack depth (only reachable via config changes / old state).
        runtime.config.loop_control.todo_max_layers = 0
        res = await sub(TodoSubParams(title="child"))
        assert res.is_error
        assert "Cannot add sub-todos deeper than 1 layers" in res.output

    async def test_sub_depth_guard_allows_at_limit(self, runtime: Runtime) -> None:
        push = TodoPush(runtime)
        sub = TodoSub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await push(TodoPushParams(title="B"))
        await push(TodoPushParams(title="C"))
        await push(TodoPushParams(title="D"))  # deepest pushed level (4/4)
        # TodoSub may still add one level under the deepest pushable parent.
        res = await sub(TodoSubParams(title="leaf"))
        assert not res.is_error
        node = _read_root_todo(lst, "A").children[0].children[0].children[0]
        assert [c.title for c in node.children] == ["leaf"]
