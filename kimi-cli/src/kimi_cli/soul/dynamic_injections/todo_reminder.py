"""Re-inject unfinished todos at the end of the context window.

LLM attention is strongest at the beginning (primacy) and end (recency) of
the context. A todo list written 50 tool calls ago drifts into the "lost in
the middle" zone — or out of the attended region entirely with
sparse/sliding-window attention. This provider periodically re-surfaces the
remaining plan as a system-reminder appended at the tail of the context,
so the model's goals stay inside the attended region.

The reminder is re-injected when:
- unfinished todos are seen for the first time,
- the todo signature (titles + statuses) changed since the last injection,
- ``interval_steps`` steps have passed since the last injection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import xxhash
from kosong.message import Message

from kimi_cli.session_state import TodoItemState
from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_TODO_REMINDER_TYPE = "todo_reminder"

_MAX_REMINDER_ITEMS = 20


class TodoReminderProvider(DynamicInjectionProvider):
    """Periodically re-injects unfinished TodoList items into the context tail."""

    def __init__(
        self,
        todos_loader: Callable[[], list[TodoItemState]],
        *,
        interval_steps: int = 10,
        max_items: int = _MAX_REMINDER_ITEMS,
    ) -> None:
        """
        Args:
            todos_loader: Zero-arg callable returning the current todo states
                (root or subagent scope). Must never raise.
            interval_steps: Minimum steps between repeated injections of an
                unchanged todo list.
            max_items: Maximum number of unfinished items shown per reminder.
        """
        self._todos_loader = todos_loader
        self._interval_steps = max(1, interval_steps)
        self._max_items = max_items
        self._last_injected_step: int | None = None
        self._last_signature: str | None = None

    @staticmethod
    def _signature(unfinished: list[TodoItemState]) -> str:
        """Hash of (title, status) pairs — detects any meaningful change."""
        digest = xxhash.xxh64()
        for item in unfinished:
            digest.update(item.status.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(item.title.encode("utf-8"))
            digest.update(b"\x01")
        return digest.hexdigest()

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        _ = history

        try:
            todos = self._todos_loader()
        except Exception:
            return []

        unfinished = [t for t in todos if t.status != "done"]
        if not unfinished:
            # Reset so the next unfinished todo triggers an immediate reminder.
            self._last_injected_step = None
            self._last_signature = None
            return []

        signature = self._signature(unfinished)
        step_no = soul._current_step_no

        if self._last_injected_step is not None:
            unchanged = signature == self._last_signature
            within_interval = (step_no - self._last_injected_step) < self._interval_steps
            if unchanged and within_interval:
                return []

        self._last_injected_step = step_no
        self._last_signature = signature

        lines = [
            "Reminder — unfinished TodoList tasks (re-injected to keep your plan in focus):",
        ]
        for item in unfinished[: self._max_items]:
            lines.append(f"- [{item.status}] {item.title}")
        if len(unfinished) > self._max_items:
            lines.append(f"- … and {len(unfinished) - self._max_items} more (call `TodoList` to read all)")
        lines.append(
            "Keep exactly one item `in_progress` and mark items `done` as you finish them."
        )
        return [DynamicInjection(type=_TODO_REMINDER_TYPE, content="\n".join(lines))]

    async def on_context_compacted(self) -> None:
        """Reset throttling so the plan is re-anchored right after compaction."""
        self._last_injected_step = None
        self._last_signature = None

    async def on_afk_changed(self, enabled: bool) -> None:
        _ = enabled
        self._last_injected_step = None
        self._last_signature = None
