"""Tests for todo_write Stack & Tree Structure (todo_push / todo_pop / todo_update).

Phase 10 of the "todo_write Stack & Tree Structure" plan:

- todo_push: push a parent onto the stack scope (breadcrumb), depth limits.
- todo_update(parent=...): add / update / rename children under an explicit parent.
- todo_pop: pop the focus subtree (errors when unfinished unless complete=True).
- Stack persistence (root + subagent), auto-heal of broken stacks.
- format_todo_injection stack breadcrumb + tree indentation.
- TodoReminderProvider signature sensitivity to stack / child changes.
- Native-gated recursive status counts.
- Cross-tool "Next:" / corrective-hint output contracts.

Mirrors ``test_todo.py`` style: ``async def`` tests with the ``runtime``
fixture; tools are instantiated directly as ``todo_push(runtime)`` etc.
"""

from __future__ import annotations

from types import SimpleNamespace

import orjson
import pytest
from kosong.tooling import ToolReturnValue

from kimi_cli.session_state import (
    TODO_INJECTION_HEADER,
    TODO_INJECTION_TRUNCATION_MARKER,
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
    todo_pop,
    TodoPopParams,
    todo_push,
    TodoPushParams,
    todo_update,
    TodoUpdateParams,
)


# Compatibility wrapper: the removed todo_sub tool is emulated by todo_update
# with the current stack top as the explicit parent.
class TodoSubParams(TodoUpdateParams):
    pass


class todo_sub:
    def __init__(self, runtime: Runtime) -> None:
        self._update = todo_update(runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._update, name)

    async def __call__(self, params: TodoSubParams) -> ToolReturnValue:
        stack = self._update._load_stack()
        parent = stack[-1] if stack else ""
        data = params.model_dump(by_alias=True)
        data["parent"] = parent
        return await self._update(TodoUpdateParams(**data))


def _read_root_todo(tool: TodoList, title: str) -> Todo:
    """Return the root-level todo with ``title`` from persisted state."""
    for t in tool._load_todos():
        if t.content == title:
            return t
    raise AssertionError(f"todo {title!r} not found")


# ---------------------------------------------------------------------------
# 1. todo_push — root append, breadcrumb, descending depth
# ---------------------------------------------------------------------------


class Testtodo_pushRootAndDepth:
    async def test_push_at_root_appends_parent(self, runtime: Runtime) -> None:
        tool = todo_push(runtime)
        res = await tool(TodoPushParams(title="Parent A", notes="notes A"))
        assert not res.is_error
        assert 'Pushed "Parent A" (depth 1/4).' in res.output
        assert "Stack: Parent A" in res.output
        assert "todo_update(parent=" in res.output
        assert res.message == 'Pushed "Parent A".'

        # Breadcrumb persisted and reloadable.
        assert tool._load_stack() == ["Parent A"]
        todos = tool._load_todos()
        assert len(todos) == 1
        assert todos[0].content == "Parent A"
        assert todos[0].status == "pending"
        assert todos[0].notes == "notes A"

    async def test_second_push_descends_into_child(self, runtime: Runtime) -> None:
        tool = todo_push(runtime)
        await tool(TodoPushParams(title="Parent"))
        res = await tool(TodoPushParams(title="Child"))
        assert not res.is_error
        assert 'Pushed "Child" (depth 2/4).' in res.output
        assert tool._load_stack() == ["Parent", "Child"]

        todos = tool._load_todos()
        assert todos[0].content == "Parent"
        assert todos[0].children[0].content == "Child"
        assert todos[0].children[0].status == "pending"

        # Display block carries depth (root = 0).
        assert len(res.display) == 1
        block = res.display[0]
        assert isinstance(block, TodoDisplayBlock)
        assert [(i.title, i.depth) for i in block.items] == [("Parent", 0), ("Child", 1)]

    async def test_push_duplicate_title_in_scope_errors(self, runtime: Runtime) -> None:
        tool = todo_push(runtime)
        sub = todo_sub(runtime)
        await tool(TodoPushParams(title="A"))
        await sub(TodoSubParams(title="B"))  # B is now a child of A (current scope)
        res = await tool(TodoPushParams(title="B"))
        assert res.is_error
        assert 'Error: Duplicate todo title "B" in this scope.' in res.output
        assert 'Use todo_update(parent=..., title="B") to update the existing item.' in res.brief

    async def test_push_duplicate_title_checks_scope_not_whole_tree(
        self, runtime: Runtime
    ) -> None:
        """The same title is allowed at different scopes (only the current
        scope's children are checked, not the whole tree)."""
        tool = todo_push(runtime)
        await tool(TodoPushParams(title="A"))
        await tool(TodoPushParams(title="B"))
        # Root list already contains "A", but the current scope (B's children)
        # is empty, so pushing "A" again descends instead of erroring.
        res = await tool(TodoPushParams(title="A"))
        assert not res.is_error
        assert tool._load_stack() == ["A", "B", "A"]
        todos = tool._load_todos()
        assert todos[0].content == "A"
        assert todos[0].children[0].content == "B"
        assert todos[0].children[0].children[0].content == "A"


# ---------------------------------------------------------------------------
# 2. todo_sub — children, same-title updates, rename, verification
# ---------------------------------------------------------------------------


class TestTodoUpdateChildren:
    async def test_sub_adds_children_under_scope(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))

        r1 = await sub(TodoSubParams(title="child one"))
        assert not r1.is_error
        assert 'Created "child one" under "Parent".' in r1.output
        assert "  - [pending] child one" in r1.output
        assert "todo_update(parent=" in r1.output and "todo_pop" in r1.output
        assert r1.message == 'Created "child one" under "Parent".'

        r2 = await sub(TodoSubParams(title="child two"))
        assert not r2.is_error

        todos = sub._load_todos()
        parent = todos[0]
        assert [c.content for c in parent.children] == ["child one", "child two"]
        assert all(c.status == "pending" for c in parent.children)
        # Stack is unchanged by todo_sub.
        assert sub._load_stack() == ["Parent"]

    async def test_sub_at_root_scope_with_empty_stack(self, runtime: Runtime) -> None:
        """With an empty stack, todo_sub operates on the root list."""
        sub = todo_sub(runtime)
        res = await sub(TodoSubParams(title="root item"))
        assert not res.is_error
        assert 'Created "root item" under "root".' in res.output
        todos = sub._load_todos()
        assert len(todos) == 1
        assert todos[0].content == "root item"

    async def test_same_title_update_keeps_old_notes_when_empty(
        self, runtime: Runtime
    ) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", notes="keep me"))

        res = await sub(TodoSubParams(title="child", status="in_progress"))
        assert not res.is_error
        assert 'Updated "child" (status=in_progress' in res.output
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.status == "in_progress"
        assert child.notes == "keep me"

    async def test_same_title_update_replaces_nonempty_notes(
        self, runtime: Runtime
    ) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="child", notes="old"))

        res = await sub(
            TodoSubParams(title="child", status="in_progress", notes="new")
        )
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.notes == "new"
        assert child.status == "in_progress"

    async def test_rename_edits_title(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="old name"))

        res = await sub(TodoSubParams(title="old name", rename_to="new name"))
        assert not res.is_error
        child = _read_root_todo(sub, "Parent").children[0]
        assert child.content == "new name"
        assert len(_read_root_todo(sub, "Parent").children) == 1

    async def test_rename_heals_stack_top(self, runtime: Runtime) -> None:
        """Renaming a child whose title equals the stack top updates the breadcrumb."""
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="P"))
        # Give root P a child also named P (titles may repeat across scopes).
        await lst(
            Params(
                todos=[
                    Todo(
                        content="P",
                        status="pending",
                        children=[Todo(content="P", status="pending")],
                    )
                ]
            )
        )
        assert sub._load_stack() == ["P"]
        res = await sub(TodoSubParams(title="P", rename_to="P2"))
        assert not res.is_error
        # Stack top healed to the renamed child.
        assert sub._load_stack() == ["P2"]
        child = _read_root_todo(sub, "P").children[0]
        assert child.content == "P2"

    async def test_rename_collision_errors(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="a"))
        await sub(TodoSubParams(title="b"))

        res = await sub(TodoSubParams(title="a", rename_to="b"))
        assert res.is_error
        assert 'Cannot rename "a" to "b"' in res.output
        assert 'Use todo_update "b" to update the existing item instead of renaming.' in res.brief
        # Nothing changed.
        children = [c.content for c in _read_root_todo(sub, "Parent").children]
        assert children == ["a", "b"]


class TestMultipleSubTodos:
    async def test_many_children_under_one_parent(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="Parent"))
        for i in range(5):
            res = await sub(TodoSubParams(title=f"child {i}"))
            assert not res.is_error

        parent = _read_root_todo(sub, "Parent")
        assert [c.content for c in parent.children] == [f"child {i}" for i in range(5)]
        assert sub._count_all(sub._load_todos()) == 6

        # Display block flattens depth-first: parent (0), then children (1).
        res = await sub(TodoSubParams(title="child 5"))
        assert not res.is_error
        block = res.display[0]
        assert [(i.title, i.depth) for i in block.items] == [
            ("Parent", 0),
            *[(f"child {i}", 1) for i in range(6)],
        ]

    async def test_deeper_nesting_via_push_after_sub(self, runtime: Runtime) -> None:
        """Push after Sub descends from the last sub-todo's scope? No — Push
        always descends from the current stack top (the parent), so a push adds
        a sibling-of-subs parent; todo_sub then nests under it."""
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="P"))
        await sub(TodoSubParams(title="c1"))
        await push(TodoPushParams(title="P2"))
        await sub(TodoSubParams(title="c2"))

        todos = push._load_todos()
        assert [c.content for c in todos[0].children] == ["c1", "P2"]
        assert todos[0].children[1].children[0].content == "c2"
        assert push._load_stack() == ["P", "P2"]


# ---------------------------------------------------------------------------
# 4. todo_pop — mark subtree done, ascend, empty-stack error
# ---------------------------------------------------------------------------


class Testtodo_pop:
    async def test_pop_marks_all_descendants_done_and_ascends(
        self, runtime: Runtime
    ) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        pop = todo_pop(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="c1"))
        await sub(TodoSubParams(title="c2", status="in_progress"))
        await push(TodoPushParams(title="Child"))
        await sub(TodoSubParams(title="grand"))

        res = await pop(TodoPopParams(complete=True))
        assert not res.is_error
        assert 'Popped "Child" — 2 sub-todo(s) marked done.' in res.output
        assert "Next: todo_push to start the next parent, or todo_write to read the tree." in res.output
        assert res.message == 'Popped "Child".'
        # Ascended to the parent scope.
        assert pop._load_stack() == ["Parent"]

        parent = _read_root_todo(pop, "Parent")
        assert parent.status == "pending"  # parent itself untouched yet
        child = next(c for c in parent.children if c.content == "Child")
        assert child.status == "done"  # even the in-progress c2 got marked done
        assert child.children[0].status == "done"
        # Sibling sub-todos under Parent are untouched by popping Child.
        c1 = next(c for c in parent.children if c.content == "c1")
        assert c1.status == "pending"
        assert len(parent.children) == 3  # c1, c2, Child

    async def test_pop_finishes_parent_subtree(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        pop = todo_pop(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="c1"))
        await sub(TodoSubParams(title="c2"))

        res = await pop(TodoPopParams(complete=True))
        assert not res.is_error
        assert 'Popped "Parent" — 3 sub-todo(s) marked done.' in res.output
        assert pop._load_stack() == []
        todos = pop._load_todos()
        assert todos[0].status == "done"
        assert all(c.status == "done" for c in todos[0].children)

    async def test_pop_empty_stack_errors(self, runtime: Runtime) -> None:
        pop = todo_pop(runtime)
        res = await pop(TodoPopParams())
        assert res.is_error
        assert "No parent todo to pop" in res.output
        assert "Use todo_push to create one, or todo_write to read the tree." in res.brief
        assert res.message == "No parent todo to pop."

    async def test_pop_broken_stack_errors(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        pop = todo_pop(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await lst(Params(todos=[Todo(content="Z", status="pending")], mode="replace", force=True))

        res = await pop(TodoPopParams())
        assert res.is_error
        assert "Todo stack is broken" in res.output
        assert "todo_write" in res.brief and "todo_push" in res.brief


# ---------------------------------------------------------------------------
# 5. Max layers
# ---------------------------------------------------------------------------


class TestMaxLayers:
    async def test_default_max_layers_4(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        for title in ("L1", "L2", "L3", "L4"):
            res = await push(TodoPushParams(title=title))
            assert not res.is_error, title
        # Depth 3 push succeeded (L4) — pushing at depth 4 now errors.
        res = await push(TodoPushParams(title="L5"))
        assert res.is_error
        assert "Cannot push deeper than 4 layers" in res.output
        assert "todo_update(parent=" in res.brief
        assert res.message == "Cannot push deeper than 4 layers."
        # Stack is unchanged after the failed push.
        assert push._load_stack() == ["L1", "L2", "L3", "L4"]

    async def test_override_max_layers_2(self, runtime: Runtime) -> None:
        runtime.config.loop_control.todo_max_layers = 2
        push = todo_push(runtime)
        res = await push(TodoPushParams(title="L1"))
        assert not res.is_error
        res = await push(TodoPushParams(title="L2"))
        assert not res.is_error
        res = await push(TodoPushParams(title="L3"))
        assert res.is_error
        assert "Cannot push deeper than 2 layers" in res.output
        assert push._load_stack() == ["L1", "L2"]

    async def test_error_output_has_hint_and_breadcrumb(self, runtime: Runtime) -> None:
        runtime.config.loop_control.todo_max_layers = 1
        push = todo_push(runtime)
        await push(TodoPushParams(title="A"))
        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        assert "Hint:" in res.output
        assert "Use todo_update(parent=...) to add sub-todos at this level instead of pushing deeper." in res.output


# ---------------------------------------------------------------------------
# 6. Auto-heal of broken stacks
# ---------------------------------------------------------------------------


class TestAutoHeal:
    async def test_partial_breakage_truncates_stack_and_warns(
        self, runtime: Runtime
    ) -> None:
        push = todo_push(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await push(TodoPushParams(title="B"))
        # Remove B (a child of A) via replace+force — A becomes a leaf.
        await lst(
            Params(todos=[Todo(content="A", status="pending")], mode="replace", force=True)
        )

        res = await push(TodoPushParams(title="C"))
        assert not res.is_error
        assert "Todo stack healed" in res.output
        assert "'B' no longer exists" in res.output
        # Stack was truncated to the longest valid prefix, then C pushed on top.
        assert push._load_stack() == ["A", "C"]
        assert [c.content for c in _read_root_todo(push, "A").children] == ["C"]

    async def test_sub_auto_heals_after_ancestor_removed(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await push(TodoPushParams(title="B"))
        await lst(
            Params(todos=[Todo(content="A", status="pending")], mode="replace", force=True)
        )

        res = await sub(TodoSubParams(title="new child"))
        assert not res.is_error
        assert "Todo stack healed" in res.output
        assert sub._load_stack() == ["A"]
        assert [c.content for c in _read_root_todo(sub, "A").children] == ["new child"]

    async def test_pop_auto_heals(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        pop = todo_pop(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await push(TodoPushParams(title="B"))
        await lst(
            Params(todos=[Todo(content="A", status="pending")], mode="replace", force=True)
        )

        res = await pop(TodoPopParams(complete=True))
        assert not res.is_error
        assert "Todo stack healed" in res.output
        assert 'Popped "A" — 1 sub-todo(s) marked done.' in res.output
        assert pop._load_stack() == []
        assert _read_root_todo(pop, "A").status == "done"

    async def test_completely_broken_stack_errors(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        # Replace the entire root list, dropping A entirely.
        await lst(
            Params(todos=[Todo(content="Z", status="pending")], mode="replace", force=True)
        )

        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        assert "Todo stack is broken" in res.output
        assert "Use todo_write to read the tree and todo_push to re-enter a parent." in res.brief

    async def test_broken_stack_does_not_mutate_tree(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await lst(
            Params(todos=[Todo(content="Z", status="pending")], mode="replace", force=True)
        )

        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        # The failed push must not have created anything.
        assert [t.content for t in push._load_todos()] == ["Z"]


# ---------------------------------------------------------------------------
# 7. Persistence round-trip (root + subagent)
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    async def test_root_persists_todos_and_stack(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="P"))
        await sub(TodoSubParams(title="c1"))

        disk = load_session_state(runtime.session.dir)
        assert disk.todo_stack == ["P"]
        assert len(disk.todos) == 1
        assert disk.todos[0].title == "P"
        assert disk.todos[0].children[0].title == "c1"
        # In-memory session state agrees with disk.
        assert runtime.session.state.todo_stack == ["P"]

    async def test_subagent_persists_todos_and_stack(self, runtime: Runtime) -> None:
        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-persist", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-persist", create=True)

        push = todo_push(sub_runtime)
        sub = todo_sub(sub_runtime)
        await push(TodoPushParams(title="S"))
        await sub(TodoSubParams(title="c"))

        state_file = sub_runtime.subagent_store.instance_dir("sub-persist") / "state.json"
        data = orjson.loads(state_file.read_bytes())
        assert data["todo_stack"] == ["S"]
        assert data["todos"][0]["title"] == "S"
        assert data["todos"][0]["children"][0]["title"] == "c"
        # Root scope is untouched by subagent writes.
        assert runtime.session.state.todo_stack == []

    async def test_subagent_stack_isolated_from_root(self, runtime: Runtime) -> None:
        root_push = todo_push(runtime)
        await root_push(TodoPushParams(title="RootP"))

        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-iso", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-iso", create=True)
        sub_push = todo_push(sub_runtime)
        await sub_push(TodoPushParams(title="SubP"))

        assert sub_push._load_stack() == ["SubP"]
        assert root_push._load_stack() == ["RootP"]

    async def test_subagent_pop_persists_stack_and_subtree(self, runtime: Runtime) -> None:
        sub_runtime = runtime.copy_for_subagent(
            agent_id="sub-pop", subagent_type="coder"
        )
        assert sub_runtime.subagent_store is not None
        sub_runtime.subagent_store.instance_dir("sub-pop", create=True)

        push = todo_push(sub_runtime)
        sub = todo_sub(sub_runtime)
        pop = todo_pop(sub_runtime)
        await push(TodoPushParams(title="S"))
        await sub(TodoSubParams(title="c1"))
        await sub(TodoSubParams(title="c2", status="in_progress"))

        res = await pop(TodoPopParams(complete=True))
        assert not res.is_error
        assert 'Popped "S" — 3 sub-todo(s) marked done.' in res.output
        assert pop._load_stack() == []

        state_file = sub_runtime.subagent_store.instance_dir("sub-pop") / "state.json"
        data = orjson.loads(state_file.read_bytes())
        assert data["todo_stack"] == []
        assert data["todos"][0]["status"] == "done"
        assert all(c["status"] == "done" for c in data["todos"][0]["children"])
        # Root session is untouched.
        assert runtime.session.state.todo_stack == []

    async def test_push_persists_notes_field(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        res = await push(TodoPushParams(title="P", notes="hello"))
        assert not res.is_error
        assert _read_root_todo(push, "P").notes == "hello"

    async def test_todolist_same_title_update_keeps_children(self, runtime: Runtime) -> None:
        """todo_write append on the same root title must not destroy the tree."""
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="P"))
        await sub(TodoSubParams(title="c1"))

        res = await lst(Params(todos=[Todo(content="P", status="in_progress")]))
        assert not res.is_error
        todos = lst._load_todos()
        assert todos[0].status == "in_progress"
        assert [c.content for c in todos[0].children] == ["c1"]
        # Stack is untouched by a todo_write write.
        assert lst._load_stack() == ["P"]


# ---------------------------------------------------------------------------
# 8. format_todo_injection with stack + tree
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

    def test_max_items_truncation_counts_stack_line(self) -> None:
        todos = [TodoItemState(title=f"Task {i}", status="pending") for i in range(20)]
        text = format_todo_injection(todos, stack=["A"], max_items=5)
        assert text is not None
        lines = text.splitlines()
        assert "- (stack: A)" in lines
        assert any(line.startswith("- … and ") for line in lines)
        # header + (stack + 4 items) + overflow line = 7 total lines.
        assert len(lines) == 7

    def test_max_chars_truncation_appends_marker(self) -> None:
        todos = [TodoItemState(title=f"Task {i}", status="pending") for i in range(30)]
        text = format_todo_injection(todos, stack=["A"], max_chars=200)
        assert text is not None
        assert len(text) <= 200
        assert text.startswith(TODO_INJECTION_HEADER)
        assert text.endswith(TODO_INJECTION_TRUNCATION_MARKER)


# ---------------------------------------------------------------------------
# 9. TodoReminderProvider signature / stack sensitivity
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
# 11. Native-gated recursive status counts
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
# 12. Cross-tool Next:/Hint output contracts
# ---------------------------------------------------------------------------


class TestCrossToolHints:
    async def test_todolist_write_success_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        res = await tool(Params(todos=[Todo(content="A", status="pending")]))
        assert not res.is_error
        assert res.output.endswith(
            "Next: todo_push to start a parent todo, todo_update to edit one or more, or todo_write to read the tree."
        )
        # Hint is output-only, never in message.
        assert "Next:" not in res.message

    async def test_todolist_read_success_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        await tool(Params(todos=[Todo(content="A", status="pending")]))
        res = await tool(Params(todos=None))
        assert not res.is_error
        assert "Next: todo_push to start a parent todo, todo_update to edit one or more, or todo_write to read the tree." in res.output
        assert res.message == "Current todo list displayed."

    async def test_todolist_zero_total_write_suppresses_hint(self, runtime: Runtime) -> None:
        tool = TodoList(runtime)
        await tool(Params(todos=[Todo(content="A", status="done")]))
        res = await tool(Params(todos=[], mode="clear"))
        assert not res.is_error
        assert res.output == "Todo list cleared (0 total: 0 done, 0 in progress, 0 pending)"
        assert "Next:" not in res.output

    async def test_push_success_hint_names_sibling_tools(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        res = await push(TodoPushParams(title="A"))
        assert not res.is_error
        assert "todo_update(parent=" in res.output
        assert "todo_pop" in res.output

    async def test_sub_success_hint_names_sibling_tools(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="P"))
        res = await sub(TodoSubParams(title="c"))
        assert not res.is_error
        assert "todo_update(parent=" in res.output
        assert "todo_pop" in res.output

    async def test_pop_success_hint_names_sibling_tools(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        pop = todo_pop(runtime)
        await push(TodoPushParams(title="A"))
        res = await pop(TodoPopParams(complete=True))
        assert not res.is_error
        assert "Next: todo_push to start the next parent, or todo_write to read the tree." in res.output

    async def test_pop_empty_error_names_corrective_tool(self, runtime: Runtime) -> None:
        pop = todo_pop(runtime)
        res = await pop(TodoPopParams())
        assert res.is_error
        assert "todo_push" in res.brief
        assert "todo_write" in res.brief
        assert "todo_push" in res.output and "todo_write" in res.output

    async def test_max_layer_error_names_todo_update(self, runtime: Runtime) -> None:
        runtime.config.loop_control.todo_max_layers = 1
        push = todo_push(runtime)
        await push(TodoPushParams(title="A"))
        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        assert "todo_update(parent=" in res.brief
        assert "todo_update(parent=" in res.output

    async def test_duplicate_push_error_names_todo_update(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="A"))
        await sub(TodoSubParams(title="B"))
        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        assert 'Use todo_update(parent=..., title="B") to update the existing item.' in res.brief

    async def test_rename_collision_error_names_update_path(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        await push(TodoPushParams(title="P"))
        await sub(TodoSubParams(title="a"))
        await sub(TodoSubParams(title="b"))
        res = await sub(TodoSubParams(title="a", rename_to="b"))
        assert res.is_error
        assert 'Use todo_update "b" to update the existing item instead of renaming.' in res.brief

    async def test_broken_stack_error_names_todolist_and_push(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="A"))
        await lst(
            Params(todos=[Todo(content="Z", status="pending")], mode="replace", force=True)
        )
        res = await push(TodoPushParams(title="B"))
        assert res.is_error
        assert "todo_write" in res.brief
        assert "todo_push" in res.brief


class TestTodoListReadTreeRendering:
    """Read mode renders the tree with a stack breadcrumb and indented children."""

    async def test_read_shows_breadcrumb_and_indented_children(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="Parent"))
        await sub(TodoSubParams(title="Child A"))
        await sub(TodoSubParams(title="Child B"))
        res = await lst(Params(todos=None))
        assert not res.is_error
        assert "Stack: Parent" in res.output
        assert "  - [pending] Child A" in res.output
        assert "  - [pending] Child B" in res.output

    async def test_read_nested_children_indent_deeper(self, runtime: Runtime) -> None:
        push = todo_push(runtime)
        sub = todo_sub(runtime)
        lst = TodoList(runtime)
        await push(TodoPushParams(title="P1"))
        await push(TodoPushParams(title="P2"))
        await sub(TodoSubParams(title="grandchild"))
        res = await lst(Params(todos=None))
        assert "Stack: P1 > P2" in res.output
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

    async def test_clear_error_names_todolist_and_push(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        await lst(Params(todos=[Todo(content="A", status="pending")]))
        res = await lst(Params(todos=[], mode="clear"))
        assert res.is_error
        assert "Cannot clear todos" in res.output
        assert "Hint: " in res.output
        assert "todo_write" in res.output
        assert "todo_push" in res.output

    async def test_empty_read_carries_next_hint(self, runtime: Runtime) -> None:
        lst = TodoList(runtime)
        res = await lst(Params(todos=None))
        assert not res.is_error
        assert "Todo list is empty." in res.output
        assert "Next: todo_push to start a parent todo, todo_update to edit one or more, or todo_write to read the tree." in res.output
