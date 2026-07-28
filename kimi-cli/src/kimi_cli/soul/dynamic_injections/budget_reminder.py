"""Budget-awareness reminders (P3, gap G4).

Agents finish long tasks better when they can *see* the remaining budget.
This provider injects a reminder when step (or optional wall-clock) usage
crosses configured ratios of ``max_steps_per_turn``:

- first ratio (default 0.7): "plan your wrap-up";
- second ratio (default 0.9): "wrap up NOW — minimal verification + summary".

Each level fires at most once per turn. Compaction does not reset real
consumption (steps are actually spent), but the highest level is allowed
to re-alert once after compaction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from kosong.message import Message

from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_BUDGET_REMINDER_TYPE = "budget_reminder"


class BudgetReminderProvider(DynamicInjectionProvider):
    """Injects budget warnings as step/wall-clock usage crosses thresholds."""

    def __init__(
        self,
        *,
        warn_ratios: tuple[float, ...] = (0.7, 0.9),
        wall_clock_seconds: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            warn_ratios: Ascending usage ratios that each trigger one
                reminder per turn (e.g. ``(0.7, 0.9)``).
            wall_clock_seconds: Optional per-turn wall-clock budget in
                seconds; 0 disables the wall-clock dimension.
            clock: Monotonic clock (injectable for tests).
        """
        self._warn_ratios = tuple(sorted(warn_ratios))
        self._wall_clock_seconds = max(0, wall_clock_seconds)
        self._clock = clock

        self._turn_id = ""
        self._turn_start: float | None = None
        self._warned_levels: set[int] = set()

    def _sync_turn(self, soul: KimiSoul) -> None:
        turn_id = soul._current_turn_id  # pyright: ignore[reportPrivateUsage]
        if turn_id and turn_id != self._turn_id:
            self._turn_id = turn_id
            self._turn_start = self._clock()
            self._warned_levels = set()
        elif self._turn_start is None:
            self._turn_start = self._clock()

    def _usages(self, soul: KimiSoul) -> tuple[float, int, float]:
        """Return ``(usage_ratio, remaining_steps, remaining_seconds)``."""
        max_steps = max(1, soul._loop_control.max_steps_per_turn)  # pyright: ignore[reportPrivateUsage]
        step_no = soul._current_step_no  # pyright: ignore[reportPrivateUsage]
        step_usage = step_no / max_steps
        remaining_steps = max(0, max_steps - step_no)

        remaining_seconds = float("inf")
        wall_usage = 0.0
        if self._wall_clock_seconds > 0 and self._turn_start is not None:
            elapsed = self._clock() - self._turn_start
            wall_usage = elapsed / self._wall_clock_seconds
            remaining_seconds = max(0.0, self._wall_clock_seconds - elapsed)

        return max(step_usage, wall_usage), remaining_steps, remaining_seconds

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        _ = history
        self._sync_turn(soul)
        if not self._warn_ratios:
            return []

        usage, remaining_steps, remaining_seconds = self._usages(soul)

        # Fire the highest crossed level that has not fired yet this turn.
        level = -1
        for idx, ratio in enumerate(self._warn_ratios):
            if usage >= ratio and idx not in self._warned_levels:
                level = idx
        if level < 0:
            return []

        # Consume this level and all lower ones: once a stronger warning has
        # fired, a weaker late warning would be confusing.
        self._warned_levels.update(range(level + 1))
        remaining_minutes = (
            ""
            if remaining_seconds == float("inf")
            else f" / ~{remaining_seconds / 60:.0f} minutes"
        )

        if level >= len(self._warn_ratios) - 1:
            content = (
                f"Budget almost exhausted (usage ≥ {self._warn_ratios[level]:.0%}; "
                f"~{remaining_steps} steps{remaining_minutes} left). "
                "Wrap up immediately: run the minimal verification and summarize "
                "the current state. Do not start new sub-tasks."
            )
        else:
            content = (
                f"Budget notice: {usage:.0%} of the step/time budget used "
                f"(~{remaining_steps} steps{remaining_minutes} left). "
                "Plan your wrap-up: prioritize remaining todos, and reserve "
                "enough budget for verification and a final summary."
            )
        return [DynamicInjection(type=_BUDGET_REMINDER_TYPE, content=content)]

    async def on_context_compacted(self) -> None:
        """Allow the highest level to re-alert once after compaction.

        Real consumption is not reset — spent steps stay spent — but the
        model may have lost the earlier warning in the summary, so the most
        urgent level may fire one more time.
        """
        if self._warned_levels:
            highest = max(self._warned_levels)
            # Keep all but the highest level marked as fired, so the most
            # urgent warning may fire once more post-compaction.
            self._warned_levels = set(range(highest))
