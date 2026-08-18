"""Tests for the simplified todo tree tools (todo_write + todo_update).

The todo toolset was consolidated from four tools (todo_write, todo_push,
todo_pop, todo_update) down to two:

- ``todo_write``: read / write / clear the whole tree.
- ``todo_update``: targeted single/batch edits, child creation via
  ``parent=...``, and one-call subtree completion via ``complete=True``.

The stack-based ``todo_push``/``todo_pop`` tools and the persisted
``todo_stack`` breadcrumb were removed: trees are addressed purely by title
(+ ``parent`` scope), which removes scope-state mistakes and reduces the
number of calls needed for hierarchical work.

Also covers ``format_todo_injection`` (session_state) and
``TodoReminderProvider`` signature behavior, and native-gated recursive
status counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import orjson
import pytest

from kimi_cli.session_state import (
    TODO_INJECTION_HEADER,
    TodoItemState,
    format_todo_injection,
    load_session_state,
)
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.dynamic_injections.todo_reminder import TodoReminderProvider
from kimi_cli.tools.display import TodoDisplayBlock
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


def _find_todo(tool: todo_update, title: str) -> Todo:
    """Return the first todo with ``title`` from persisted state."""
    for t in tool._load_todos():
        if t.content == title:
            return t
    raise AssertionError(f"todo {title!r} not found")


# ---------------------------------------------------------------------------
# 1. Tree creation — todo_update(parent=...) replaces todo_push/todo_sub
# ---------------------------------------------------------------------------


class TestTreeCreation:
    async def test_create_root_item_with_empty_parent(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        res = await update(TodoUpdateParams(parent="", title="Root item", notes="notes"))
        assert not res.is_error
        assert 'Created "Root item" under "root".' in res.output
        assert res.message == 'Created "Root item" under "root".'

        todos = update._load_todos()
        assert len(todos) == 1
        assert todos[0].content == "Root item"
        assert todos[0].status == "pending"
        assert todos[0].notes == "notes"

    async def test_create_children_under_parent(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))

        r1 = await update(TodoUpdateParams(parent="Parent", title="child one"))
        assert not r1.is_error
        assert 'Created "child one" under "Parent".' in r1.output
        assert "  - [pending] child one" in r1.output
        assert r1.message == 'Created "child one" under "Parent".'

        r2 = await update(TodoUpdateParams(parent="Parent", title="child two"))
        assert not r2.is_error

        parent = _read_root_todo(update, "Parent")
        assert [c.content for c in parent.children] == ["child one", "child two"]
        assert all(c.status == "pending" for c in parent.children)

    async def test_many_children_under_one_parent(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        for i in range(5):
            res = await update(TodoUpdateParams(parent="Parent", title=f"child {i}"))
            assert not res.is_error

        parent = _read_root_todo(update, "Parent")
        assert [c.content for c in parent.children] == [f"child {i}" for i in range(5)]
        assert update._count_all(update._load_todos()) == 6

        # Display block flattens depth-first: parent (0), then children (1).
        res = await update(TodoUpdateParams(parent="Parent", title="child 5"))
        assert not res.is_error
        block = res.display[0]
        assert [(i.title, i.depth) for i in block.items] == [
            ("Parent", 0),
            *[(f"child {i}", 1) for i in range(6)],
        ]

    async def test_deeper_nesting_via_nested_parents(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="P", status="pending")]))
        await update(TodoUpdateParams(parent="P", title="P2"))
        await update(TodoUpdateParams(parent="P2", title="c1"))
        await update(TodoUpdateParams(parent="P2", title="c2"))

        todos = update._load_todos()
        assert [c.content for c in todos[0].children] == ["P2"]
        assert [c.content for c in todos[0].children[0].children] == ["c1", "c2"]

    async def test_batch_create_root_children_when_empty(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        res = await update(TodoUpdateParams(parent="", updates=[{"title": "A"}, {"title": "B"}]))
        assert not res.is_error
        todos = update._load_todos()
        assert [t.content for t in todos] == ["A", "B"]


# ---------------------------------------------------------------------------
# 2. Tree edits — same-title updates, rename, parent-scoped lookup
# ---------------------------------------------------------------------------


class TestTreeEdits:
    async def test_same_title_update_keeps_old_notes_when_empty(
        self, runtime: Runtime
    ) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", notes="keep me"))

        res = await update(TodoUpdateParams(parent="Parent", title="child", status="in_progress"))
        assert not res.is_error
        assert 'Updated "child" (status=in_progress' in res.output
        child = _read_root_todo(update, "Parent").children[0]
        assert child.status == "in_progress"
        assert child.notes == "keep me"

    async def test_same_title_update_replaces_nonempty_notes(
        self, runtime: Runtime
    ) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="child", notes="old"))

        res = await update(
            TodoUpdateParams(parent="Parent", title="child", status="in_progress", notes="new")
        )
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.notes == "new"
        assert child.status == "in_progress"

    async def test_rename_edits_title(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="old name"))

        res = await update(TodoUpdateParams(parent="Parent", title="old name", rename_to="new name"))
        assert not res.is_error
        child = _read_root_todo(update, "Parent").children[0]
        assert child.content == "new name"
        assert len(_read_root_todo(update, "Parent").children) == 1

    async def test_rename_collision_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="Parent", status="pending")]))
        await update(TodoUpdateParams(parent="Parent", title="a"))
        await update(TodoUpdateParams(parent="Parent", title="b"))

        res = await update(TodoUpdateParams(parent="Parent", title="a", rename_to="b"))
        assert res.is_error
        assert 'Cannot rename "a" to "b"' in res.output
        assert 'Use todo_update "b" to update the existing item instead of renaming.' in res.brief
        # Nothing changed.
        children = [c.content for c in _read_root_todo(update, "Parent").children]
        assert children == ["a", "b"]

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

        res = await update(TodoUpdateParams(parent="P2", title="Child", status="done"))
        assert not res.is_error
        p1 = _find_todo(update, "P1")
        p2 = _find_todo(update, "P2")
        assert p1.children[0].status == "pending"
        assert p2.children[0].status == "done"

    async def test_missing_parent_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))

        res = await update(TodoUpdateParams(parent="Missing", title="Child"))
        assert res.is_error
        assert 'No parent todo matching "Missing" found' in res.output


# ---------------------------------------------------------------------------
# 3. complete=True — one-call subtree finish (replaces todo_pop)
# ---------------------------------------------------------------------------


class TestCompleteSubtree:
    async def test_complete_marks_all_descendants_done(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(
                        content="Parent",
                        status="pending",
                        children=[
                            Todo(content="c1", status="pending"),
                            Todo(
                                content="c2",
                                status="in_progress",
                                children=[Todo(content="grand", status="pending")],
                            ),
                        ],
                    )
                ]
            )
        )

        res = await update(TodoUpdateParams(title="Parent", complete=True))
        assert not res.is_error
        assert "completed with 4 sub-todos marked done" in res.output
        assert res.message == 'Updated "Parent".'

        parent = _read_root_todo(update, "Parent")
        assert parent.status == "done"
        assert all(c.status == "done" for c in parent.children)
        assert parent.children[1].children[0].status == "done"

    async def test_complete_child_leaves_siblings_untouched(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(
            Params(
                todos=[
                    Todo(
                        content="Parent",
                        status="pending",
                        children=[
                            Todo(content="c1", status="pending"),
                            Todo(
                                content="c2",
                                status="in_progress",
                                children=[Todo(content="grand", status="pending")],
                            ),
                            Todo(content="c3", status="pending"),
                        ],
                    )
                ]
            )
        )

        res = await update(TodoUpdateParams(parent="Parent", title="c2", complete=True))
        assert not res.is_error
        parent = _read_root_todo(update, "Parent")
        c2 = next(c for c in parent.children if c.content == "c2")
        assert c2.status == "done"
        assert c2.children[0].status == "done"
        # Siblings untouched.
        c1 = next(c for c in parent.children if c.content == "c1")
        c3 = next(c for c in parent.children if c.content == "c3")
        assert c1.status == "pending"
        assert c3.status == "pending"

    async def test_complete_on_missing_title_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))

        res = await update(TodoUpdateParams(title="ghost", complete=True))
        assert res.is_error
        assert "found" in res.output  # fuzzy path: 'No todo matching "ghost" found.'

    async def test_complete_with_status_pending_errors(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))

        res = await update(TodoUpdateParams(title="A", status="pending", complete=True))
        assert res.is_error
        assert "complete=True cannot be combined with status=" in res.output
        assert _find_todo(update, "A").status == "pending"


# ---------------------------------------------------------------------------
# 4. Max tree depth
# ---------------------------------------------------------------------------


class TestMaxDepth:
    async def test_default_max_depth_5_allowed(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        # max_depth = max_layers(4) + one todo_update(parent=...) level.
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

    async def test_update_depth_guard_with_limited_layers(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        # max_layers=0 means children may not be added below the root level.
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


# ---------------------------------------------------------------------------
# 5. Persistence round-trip (root + subagent)
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    async def test_root_persists_todos(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="P"))
        await update(TodoUpdateParams(parent="P", title="c1"))

        disk = load_session_state(runtime.session.dir)
        assert len(disk.todos) == 1
        assert disk.todos[0].title == "P"
        assert disk.todos[0].children[0].title == "c1"
        # In-memory session state agrees with disk.
        assert runtime.session.state.todos == disk.todos

    async def test_subagent_persists_todos(self, runtime: Runtime) -> None:
        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-persist", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-persist", create=True)

        lst = TodoList(sub_runtime)
        update = todo_update(sub_runtime)
        await update(TodoUpdateParams(parent="", title="S"))
        await update(TodoUpdateParams(parent="S", title="c"))

        state_file = sub_runtime.subagent_store.instance_dir("sub-persist") / "state.json"
        data = orjson.loads(state_file.read_bytes())
        assert data["todos"][0]["title"] == "S"
        assert data["todos"][0]["children"][0]["title"] == "c"
        # Root scope is untouched by subagent writes.
        assert runtime.session.state.todos == []

    async def test_subagent_state_isolated_from_root(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="RootP", status="pending")]))

        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-iso", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-iso", create=True)
        sub_update = todo_update(sub_runtime)
        await sub_update(TodoUpdateParams(parent="", title="SubP"))

        assert [t.content for t in sub_update._load_todos()] == ["SubP"]
        assert [t.content for t in lst._load_todos()] == ["RootP"]

    async def test_subagent_complete_persists_subtree(self, runtime: Runtime) -> None:
        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-complete", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-complete", create=True)

        lst = TodoList(sub_runtime)
        update = todo_update(sub_runtime)
        await update(TodoUpdateParams(parent="", title="S"))
        await update(TodoUpdateParams(parent="S", title="c1"))
        await update(TodoUpdateParams(parent="S", title="c2", status="in_progress"))

        res = await update(TodoUpdateParams(title="S", complete=True))
        assert not res.is_error
        assert "completed with 3 sub-todos marked done" in res.output

        state_file = sub_runtime.subagent_store.instance_dir("sub-complete") / "state.json"
        data = orjson.loads(state_file.read_bytes())
        assert data["todos"][0]["status"] == "done"
        assert all(c["status"] == "done" for c in data["todos"][0]["children"])
        # Root session is untouched.
        assert runtime.session.state.todos == []

    async def test_create_persists_notes_field(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        res = await update(TodoUpdateParams(parent="", title="P", notes="hello"))
        assert not res.is_error
        assert _read_root_todo(update, "P").notes == "hello"

    async def test_todolist_same_title_update_keeps_children(self, runtime: Runtime) -> None:
        """todo_write append on the same root title must not destroy the tree."""
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="P"))
        await update(TodoUpdateParams(parent="P", title="c1"))

        res = await lst(Params(todos=[Todo(content="P", status="in_progress")]))
        assert not res.is_error
        todos = lst._load_todos()
        assert todos[0].status == "in_progress"
        assert [c.content for c in todos[0].children] == ["c1"]


# ---------------------------------------------------------------------------
# 6. format_todo_injection with stack + tree (session_state still supports it)
# ---------------------------------------------------------------------------


class TestFormatTodoInjectionStack:
    def test_stack_breadcrumb_and_indentation(self) -> None:
        parent = TodoItemState(
            title="Parent",
            status="in_progress",
            children=[TodoItemState(title="child", status="pending")],
        )
        text = format_todo_injection([parent], stack=["Root", "Parent"])
        assert text is not None
        lines = text.splitlines()
        assert lines[0] == TODO_INJECTION_HEADER
        assert lines[1] == "- (stack: Root > Parent)"
        assert "- [>] Parent (in_progress)" in text
        assert "  - [ ] child (pending)" in text

    def test_flat_lists_byte_identical_with_and_without_stack(self) -> None:
        flat = [
            TodoItemState(title="A", status="pending"),
            TodoItemState(title="B", status="in_progress"),
        ]
        default_text = format_todo_injection(flat)
        assert default_text is not None
        assert default_text == format_todo_injection(flat, stack=[])
        assert default_text == format_todo_injection(flat, stack=None)
        assert "- (stack:" not in default_text

    def test_done_parent_still_reveals_pending_child(self) -> None:
        parent = TodoItemState(
            title="Parent",
            status="done",
            children=[TodoItemState(title="child", status="pending")],
        )
        text = format_todo_injection([parent], stack=["Parent"])
        assert text is not None
        assert "- (stack: Parent)" in text
        assert "Parent (done)" not in text
        assert "  - [ ] child (pending)" in text


# ---------------------------------------------------------------------------
# 7. TodoReminderProvider signature sensitivity
# ---------------------------------------------------------------------------


class TestReminderSignature:
    def test_signature_changes_on_child_status_change(self) -> None:
        items = [
            TodoItemState(
                title="P", status="pending", children=[TodoItemState(title="c", status="pending")]
            )
        ]
        sig_pending = TodoReminderProvider._signature(items)
        changed = [
            TodoItemState(
                title="P", status="pending", children=[TodoItemState(title="c", status="done")]
            )
        ]
        assert TodoReminderProvider._signature(changed) != sig_pending

    def test_signature_changes_on_stack_change(self) -> None:
        items = [TodoItemState(title="P", status="pending")]
        base = TodoReminderProvider._signature(items)
        assert TodoReminderProvider._signature(items, stack=["A"]) != base
        assert (
            TodoReminderProvider._signature(items, stack=["A", "B"])
            != TodoReminderProvider._signature(items, stack=["A"])
        )

    async def test_get_injections_includes_stack_breadcrumb(self) -> None:
        provider = TodoReminderProvider(
            todos_loader=lambda: [TodoItemState(title="P", status="pending")],
            stack_loader=lambda: ["A", "B"],
        )
        soul = SimpleNamespace(_current_step_no=1)
        injections = await provider.get_injections([], soul)  # type: ignore[arg-type]
        assert len(injections) == 1
        assert "- (stack: A > B)" in injections[0].content
        assert "- [pending] P" in injections[0].content

    async def test_stack_change_retriggers_injection(self) -> None:
        state = {"stack": []}
        provider = TodoReminderProvider(
            todos_loader=lambda: [TodoItemState(title="P", status="pending")],
            stack_loader=lambda: state["stack"],
            interval_steps=100,
        )
        soul = SimpleNamespace(_current_step_no=1)
        first = await provider.get_injections([], soul)  # type: ignore[arg-type]
        assert len(first) == 1

        # Same stack, same signature → throttled within the interval.
        soul._current_step_no = 2
        assert await provider.get_injections([], soul) == []  # type: ignore[arg-type]

        # Stack change alters the signature → re-injected.
        state["stack"] = ["A"]
        again = await provider.get_injections([], soul)  # type: ignore[arg-type]
        assert len(again) == 1
        assert "- (stack: A)" in again[0].content


# ---------------------------------------------------------------------------
# 8. Native-gated recursive status counts
# ---------------------------------------------------------------------------


class TestNativeGating:
    def test_status_counts_recursive_with_children(self) -> None:
        todos = [
            Todo(
                content="P",
                status="pending",
                children=[
                    Todo(content="c1", status="in_progress"),
                    Todo(
                        content="c2",
                        status="done",
                        children=[Todo(content="g", status="pending")],
                    ),
                ],
            ),
            Todo(content="Q", status="done"),
        ]
        assert TodoList._status_counts(todos) == {
            "pending": 2,
            "in_progress": 1,
            "done": 2,
        }

    def test_status_counts_flat(self) -> None:
        todos = [
            Todo(content="a", status="pending"),
            Todo(content="b", status="in_progress"),
            Todo(content="c", status="done"),
        ]
        assert TodoList._status_counts(todos) == {
            "pending": 1,
            "in_progress": 1,
            "done": 1,
        }

    def test_count_all_recursive(self) -> None:
        todos = [
            Todo(
                content="P",
                status="pending",
                children=[
                    Todo(content="c1", status="pending"),
                    Todo(
                        content="c2",
                        status="done",
                        children=[Todo(content="g", status="pending")],
                    ),
                ],
            )
        ]
        assert TodoList._count_all(todos) == 4

    def test_count_unfinished_descendants(self) -> None:
        todo = Todo(
            content="P",
            status="pending",
            children=[
                Todo(content="c1", status="pending"),
                Todo(content="c2", status="done"),
                Todo(content="c3", status="in_progress", children=[Todo(content="g", status="done")]),
            ],
        )
        assert TodoList._count_unfinished_descendants(todo) == 2  # c1 + c3

    def test_mark_subtree_done_recursive(self) -> None:
        todo = Todo(
            content="P",
            status="pending",
            children=[
                Todo(content="c1", status="in_progress"),
                Todo(content="c2", status="pending", children=[Todo(content="g", status="pending")]),
            ],
        )
        TodoList._mark_subtree_done(todo)
        assert todo.status == "done"
        assert all(c.status == "done" for c in todo.children)
        assert todo.children[1].children[0].status == "done"


# ---------------------------------------------------------------------------
# 9. Cross-tool Next:/Hint output contracts
# ---------------------------------------------------------------------------


class TestCrossToolHints:
    async def test_todolist_write_success_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        res = await tool(Params(todos=[Todo(content="A", status="pending")]))
        assert not res.is_error
        assert res.output.endswith(
            "Next: todo_update to edit one or more items, or todo_write to read the tree."
        )
        # Hint is output-only, never in message.
        assert "Next:" not in res.message

    async def test_todolist_read_success_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        await tool(Params(todos=[Todo(content="A", status="pending")]))
        res = await tool(Params(todos=None))
        assert not res.is_error
        assert "Next: todo_update to edit one or more items, or todo_write to read the tree." in res.output
        assert res.message == "Current todo list displayed."

    async def test_todolist_zero_total_write_suppresses_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        await tool(Params(todos=[Todo(content="A", status="done")]))
        res = await tool(Params(todos=[], mode="clear"))
        assert not res.is_error
        assert res.output == "Todo list cleared (0 total: 0 done, 0 in progress, 0 pending)"
        assert "Next:" not in res.output

    async def test_create_success_hint_names_sibling_tools(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="P", status="pending")]))
        res = await update(TodoUpdateParams(parent="P", title="c"))
        assert not res.is_error
        assert "todo_update(parent=" in res.output
        assert "todo_write" in res.output

    async def test_update_success_hint_names_sibling_tools(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="P", status="pending")]))
        res = await update(TodoUpdateParams(title="P", status="done"))
        assert not res.is_error
        assert "todo_update" in res.output
        assert "todo_write" in res.output

    async def test_no_todos_error_names_creation_path(self, runtime: Runtime) -> None:
        update = todo_update(runtime)
        res = await update(TodoUpdateParams(title="A", status="done"))
        assert res.is_error
        assert "No todos exist" in res.output
        assert 'todo_update(parent="", title="...")' in res.brief
        assert "todo_write" in res.brief

    async def test_max_depth_error_names_todo_update(self, runtime: Runtime) -> None:
        runtime.config.loop_control.todo_max_layers = 1
        lst = TodoList(runtime)
        deep = Todo(
            content="A",
            status="pending",
            children=[
                Todo(
                    content="B",
                    status="pending",
                    children=[Todo(content="C", status="pending")],
                )
            ],
        )
        res = await lst(Params(todos=[deep]))
        assert res.is_error
        assert "todo_update(parent=" in res.output

    async def test_rename_collision_error_names_update_path(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await lst(Params(todos=[Todo(content="P", status="pending")]))
        await update(TodoUpdateParams(parent="P", title="a"))
        await update(TodoUpdateParams(parent="P", title="b"))
        res = await update(TodoUpdateParams(parent="P", title="a", rename_to="b"))
        assert res.is_error
        assert 'Use todo_update "b" to update the existing item instead of renaming.' in res.brief


class TestTodoListReadTreeRendering:
    """Read mode renders the tree with indented children (no stack breadcrumb)."""

    async def test_read_shows_indented_children(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="Parent"))
        await update(TodoUpdateParams(parent="Parent", title="Child A"))
        await update(TodoUpdateParams(parent="Parent", title="Child B"))
        res = await lst(Params(todos=None))
        assert not res.is_error
        assert "Stack:" not in res.output
        assert "- [pending] Parent" in res.output
        assert "  - [pending] Child A" in res.output
        assert "  - [pending] Child B" in res.output

    async def test_read_nested_children_indent_deeper(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        update = todo_update(runtime)
        await update(TodoUpdateParams(parent="", title="P1"))
        await update(TodoUpdateParams(parent="P1", title="P2"))
        await update(TodoUpdateParams(parent="P2", title="grandchild"))
        res = await lst(Params(todos=None))
        assert "Stack:" not in res.output
        assert "- [pending] P1" in res.output
        assert "  - [pending] P2" in res.output
        assert "    - [pending] grandchild" in res.output

    async def test_read_flat_list_renders_unindented(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=None))
        assert "Stack:" not in res.output
        assert "\n- [pending] A" in res.output


class TestTodoListErrorHints:
    """todo_write error paths carry corrective sibling-tool hints."""

    async def test_duplicate_error_names_todo_update(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        res = await lst(
            Params(
                todos=[
                    Todo(content="Dup", status="pending"),
                    Todo(content="Dup", status="done"),
                ]
            )
        )
        assert res.is_error
        assert "Duplicate todo titles found" in res.output
        assert "Hint: " in res.output
        assert "todo_update(parent=" in res.output

    async def test_clear_error_names_todolist_and_update(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=[], mode="clear"))
        assert res.is_error
        assert "Cannot clear todos" in res.output
        assert "Hint: " in res.output
        assert "todo_write" in res.output
        assert "todo_update" in res.output

    async def test_empty_read_carries_next_hint(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        res = await lst(Params(todos=None))
        assert not res.is_error
        assert "Todo list is empty." in res.output
        assert "Next: todo_update to edit one or more items, or todo_write to read the tree." in res.output
