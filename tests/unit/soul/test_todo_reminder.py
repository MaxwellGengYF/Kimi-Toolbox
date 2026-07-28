"""Unit tests for TodoReminderProvider (recency-edge plan re-injection)."""

from __future__ import annotations

from types import SimpleNamespace

from kimi_cli.session_state import TodoItemState
from kimi_cli.soul.dynamic_injections.todo_reminder import TodoReminderProvider


def _todo(title: str, status: str = "pending") -> TodoItemState:
    return TodoItemState(title=title, status=status)  # type: ignore[arg-type]


def _soul(step_no: int, *, subagent: bool = False) -> SimpleNamespace:
    return SimpleNamespace(_current_step_no=step_no, is_subagent=subagent)


class TestInjection:
    async def test_injects_unfinished_todos(self) -> None:
        provider = TodoReminderProvider(lambda: [_todo("task A"), _todo("task B", "in_progress")])
        injections = await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        assert len(injections) == 1
        assert injections[0].type == "todo_reminder"
        assert "[pending] task A" in injections[0].content
        assert "[in_progress] task B" in injections[0].content

    async def test_no_injection_when_all_done(self) -> None:
        provider = TodoReminderProvider(lambda: [_todo("task A", "done")])
        assert await provider.get_injections([], _soul(1)) == []  # type: ignore[arg-type]

    async def test_no_injection_when_empty(self) -> None:
        provider = TodoReminderProvider(lambda: [])
        assert await provider.get_injections([], _soul(1)) == []  # type: ignore[arg-type]

    async def test_loader_exception_is_swallowed(self) -> None:
        def _boom() -> list[TodoItemState]:
            raise RuntimeError("corrupt state")

        provider = TodoReminderProvider(_boom)
        assert await provider.get_injections([], _soul(1)) == []  # type: ignore[arg-type]


class TestThrottling:
    async def test_throttled_within_interval(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=10)
        assert await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        # same signature, steps 2..10 -> throttled
        for step in range(2, 11):
            assert await provider.get_injections([], _soul(step)) == []  # type: ignore[arg-type]
        # step 11 -> interval elapsed -> re-inject
        assert await provider.get_injections([], _soul(11))  # type: ignore[arg-type]

    async def test_signature_change_reinjects_immediately(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=100)
        assert await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        todos.append(_todo("task B"))
        # changed signature -> inject despite interval
        assert await provider.get_injections([], _soul(2))  # type: ignore[arg-type]

    async def test_status_change_reinjects(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=100)
        await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        todos[0].status = "in_progress"  # type: ignore[assignment]
        assert await provider.get_injections([], _soul(2))  # type: ignore[arg-type]

    async def test_all_done_resets_throttle(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=100)
        await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        todos[0].status = "done"  # type: ignore[assignment]
        assert await provider.get_injections([], _soul(2)) == []  # type: ignore[arg-type]
        # new unfinished todo -> immediate injection even within old interval
        todos.append(_todo("task C"))
        assert await provider.get_injections([], _soul(3))  # type: ignore[arg-type]

    async def test_compaction_resets_throttle(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=100)
        await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        await provider.on_context_compacted()
        # after compaction the plan is re-anchored immediately
        assert await provider.get_injections([], _soul(2))  # type: ignore[arg-type]

    async def test_afk_change_resets_throttle(self) -> None:
        todos = [_todo("task A")]
        provider = TodoReminderProvider(lambda: todos, interval_steps=100)
        await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        await provider.on_afk_changed(True)
        assert await provider.get_injections([], _soul(2))  # type: ignore[arg-type]


class TestRendering:
    async def test_max_items_truncation(self) -> None:
        todos = [_todo(f"task {i}") for i in range(25)]
        provider = TodoReminderProvider(lambda: todos, max_items=5)
        injections = await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        content = injections[0].content
        assert "task 4" in content
        assert "task 5" not in content
        assert "20 more" in content

    async def test_done_items_not_shown(self) -> None:
        todos = [_todo("finished", "done"), _todo("open")]
        provider = TodoReminderProvider(lambda: todos)
        injections = await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        assert "- [done] finished" not in injections[0].content
        assert "- [pending] open" in injections[0].content

    async def test_guidance_line_present(self) -> None:
        provider = TodoReminderProvider(lambda: [_todo("x")])
        injections = await provider.get_injections([], _soul(1))  # type: ignore[arg-type]
        assert "in_progress" in injections[0].content
