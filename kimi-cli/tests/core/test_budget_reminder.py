"""Tests for the BudgetReminderProvider (P3: budget-awareness reminders)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from kimi_cli.soul.dynamic_injections.budget_reminder import BudgetReminderProvider


def _soul(step: int, turn: str = "t1", max_steps: int = 100) -> Any:
    soul = MagicMock()
    soul._current_step_no = step
    soul._current_turn_id = turn
    soul._loop_control.max_steps_per_turn = max_steps
    return soul


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Step-budget triggers
# ---------------------------------------------------------------------------


async def test_70_and_90_percent_each_fire_once() -> None:
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9))

    # Below the first ratio: silent.
    assert await provider.get_injections([], _soul(50)) == []

    # Cross 0.7: planning reminder fires once.
    first = await provider.get_injections([], _soul(70))
    assert len(first) == 1
    assert "Plan your wrap-up" in first[0].content

    # Repeat at the same level: suppressed.
    assert await provider.get_injections([], _soul(75)) == []

    # Cross 0.9: urgent reminder fires once.
    second = await provider.get_injections([], _soul(90))
    assert len(second) == 1
    assert "Wrap up immediately" in second[0].content

    # Repeat: suppressed.
    assert await provider.get_injections([], _soul(95)) == []


async def test_highest_level_wins_when_multiple_crossed() -> None:
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9))
    # Jump straight past both ratios: only the urgent level fires.
    injections = await provider.get_injections([], _soul(95))
    assert len(injections) == 1
    assert "Wrap up immediately" in injections[0].content
    # The lower level never fires afterwards (already crossed).
    assert await provider.get_injections([], _soul(96)) == []


async def test_default_budget_short_turn_no_trigger() -> None:
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9))
    # Default max_steps_per_turn is 15000; a short turn stays far below 0.7.
    for step in (1, 10, 100, 1000):
        assert await provider.get_injections([], _soul(step, max_steps=15000)) == []


# ---------------------------------------------------------------------------
# Wall-clock dimension
# ---------------------------------------------------------------------------


async def test_wall_clock_dimension_independent() -> None:
    clock = _FakeClock()
    provider = BudgetReminderProvider(
        warn_ratios=(0.7, 0.9), wall_clock_seconds=100, clock=clock
    )
    # First call establishes the turn start.
    assert await provider.get_injections([], _soul(1, max_steps=15000)) == []

    # Advance past 70% of the wall budget with tiny step usage.
    clock.advance(71)
    first = await provider.get_injections([], _soul(2, max_steps=15000))
    assert len(first) == 1
    assert "minutes" in first[0].content

    # Advance past 90%: urgent.
    clock.advance(21)
    second = await provider.get_injections([], _soul(3, max_steps=15000))
    assert len(second) == 1
    assert "Wrap up immediately" in second[0].content


async def test_wall_clock_disabled_by_default() -> None:
    clock = _FakeClock()
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9), wall_clock_seconds=0, clock=clock)
    await provider.get_injections([], _soul(1, max_steps=15000))
    clock.advance(10_000)
    assert await provider.get_injections([], _soul(2, max_steps=15000)) == []


# ---------------------------------------------------------------------------
# Turn boundaries & compaction
# ---------------------------------------------------------------------------


async def test_new_turn_resets_levels() -> None:
    clock = _FakeClock()
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9), clock=clock)
    first = await provider.get_injections([], _soul(70, turn="t1"))
    assert len(first) == 1
    # New turn: the level may fire again.
    second = await provider.get_injections([], _soul(71, turn="t2"))
    assert len(second) == 1


async def test_compaction_allows_top_level_realert() -> None:
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9))
    assert len(await provider.get_injections([], _soul(70))) == 1
    assert len(await provider.get_injections([], _soul(90))) == 1
    assert await provider.get_injections([], _soul(95)) == []

    await provider.on_context_compacted()
    # Highest (urgent) level may fire once more; the lower stays fired.
    realert = await provider.get_injections([], _soul(96))
    assert len(realert) == 1
    assert "Wrap up immediately" in realert[0].content
    assert await provider.get_injections([], _soul(97)) == []


async def test_compaction_without_alerts_is_noop() -> None:
    provider = BudgetReminderProvider(warn_ratios=(0.7, 0.9))
    await provider.on_context_compacted()  # must not raise
    assert len(await provider.get_injections([], _soul(70))) == 1
