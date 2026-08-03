"""Tests for ContextMeterProvider.

The critical property under test: the meter must NEVER be injected too
frequently. It may only fire when usage has materially changed
(>= ``min_delta``) and enough steps have passed (>= ``cooldown_steps``) since
the last injection — except for a one-shot report right after a real
compaction (and a one-shot re-anchor after an AFK toggle).

The harness strips stale system-reminder messages from history on every step
before collecting fresh injections; that strip must NOT reset the provider's
throttle state (see the regression tests at the bottom).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from kosong.message import Message

from kimi_cli.soul.dynamic_injections.context_meter import (
    _CONTEXT_METER_TYPE,
    ContextMeterProvider,
)
from kimi_cli.soul.message import strip_system_reminders, system_reminder

# Config defaults used by KimiSoul when constructing the provider.
MIN_DELTA = 0.15
COOLDOWN_STEPS = 30
SUPPRESS_ABOVE = 0.70
MIN_USAGE = 0.20


def _make_provider(**kwargs) -> ContextMeterProvider:
    return ContextMeterProvider(
        min_delta=kwargs.pop("min_delta", MIN_DELTA),
        cooldown_steps=kwargs.pop("cooldown_steps", COOLDOWN_STEPS),
        suppress_above=kwargs.pop("suppress_above", SUPPRESS_ABOVE),
        min_usage=kwargs.pop("min_usage", MIN_USAGE),
        **kwargs,
    )


def _mock_soul(
    usage: float = 0.0,
    step_no: int = 1,
    max_context_tokens: int = 100_000,
    is_subagent: bool = False,
) -> MagicMock:
    soul = MagicMock()
    soul.is_subagent = is_subagent
    soul._current_step_no = step_no
    status = MagicMock()
    status.max_context_tokens = max_context_tokens
    soul.status = status
    context = MagicMock()
    context.token_count_with_pending = int(usage * max_context_tokens)
    soul.context = context
    return soul


# ── Basic gates ────────────────────────────────────────────────────────


async def test_injected_once_above_min_usage() -> None:
    provider = _make_provider()
    result = await provider.get_injections([], _mock_soul(usage=0.30, step_no=1))
    assert len(result) == 1
    assert result[0].type == _CONTEXT_METER_TYPE
    assert "Memory" in result[0].content


async def test_silent_below_min_usage_on_first_call() -> None:
    provider = _make_provider()
    result = await provider.get_injections([], _mock_soul(usage=0.10, step_no=1))
    assert result == []


async def test_silent_when_max_tokens_unknown() -> None:
    provider = _make_provider()
    soul = _mock_soul(usage=0.50, step_no=1)
    soul.status.max_context_tokens = 0
    result = await provider.get_injections([], soul)
    assert result == []


async def test_silent_for_subagent() -> None:
    provider = _make_provider()
    result = await provider.get_injections(
        [], _mock_soul(usage=0.50, step_no=1, is_subagent=True)
    )
    assert result == []


async def test_silent_above_suppress_threshold() -> None:
    provider = _make_provider()
    result = await provider.get_injections([], _mock_soul(usage=0.80, step_no=1))
    assert result == []


async def test_silent_at_suppress_threshold() -> None:
    provider = _make_provider()
    result = await provider.get_injections([], _mock_soul(usage=SUPPRESS_ABOVE, step_no=1))
    assert result == []


async def test_suppress_disabled_when_none() -> None:
    provider = _make_provider(suppress_above=None)
    result = await provider.get_injections([], _mock_soul(usage=0.90, step_no=1))
    assert len(result) == 1


# ── Frequency guards: cooldown + delta ─────────────────────────────────


async def test_throttled_by_cooldown_even_with_usage_growth() -> None:
    provider = _make_provider()
    # First injection at step 1.
    assert len(await provider.get_injections([], _mock_soul(usage=0.25, step_no=1))) == 1
    # Usage grew a lot, but cooldown has not elapsed -> silent.
    result = await provider.get_injections([], _mock_soul(usage=0.60, step_no=5))
    assert result == []


async def test_throttled_by_delta_even_after_cooldown() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.25, step_no=1))) == 1
    # Cooldown elapsed, but usage barely changed -> silent.
    result = await provider.get_injections([], _mock_soul(usage=0.26, step_no=31))
    assert result == []


async def test_reinjects_when_cooldown_and_delta_satisfied() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.25, step_no=1))) == 1
    # Cooldown elapsed and usage grew by >= min_delta -> inject again.
    result = await provider.get_injections([], _mock_soul(usage=0.45, step_no=31))
    assert len(result) == 1
    assert result[0].type == _CONTEXT_METER_TYPE


async def test_reinjects_on_usage_drop_after_cooldown() -> None:
    """A material usage *decrease* (e.g. pruning) also re-anchors the meter."""
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.50, step_no=1))) == 1
    result = await provider.get_injections([], _mock_soul(usage=0.30, step_no=31))
    assert len(result) == 1


async def test_delta_boundary_exactly_at_min_delta() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.25, step_no=1))) == 1
    # delta == min_delta (0.40 - 0.25) -> injects.
    result = await provider.get_injections([], _mock_soul(usage=0.40, step_no=31))
    assert len(result) == 1


# ── Compaction ─────────────────────────────────────────────────────────


async def test_compaction_reports_fresh_low_usage_once() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.50, step_no=1))) == 1
    # Real compaction: usage drops well below min_usage.
    await provider.on_context_compacted()
    result = await provider.get_injections([], _mock_soul(usage=0.10, step_no=21))
    assert len(result) == 1  # one-shot fresh report


async def test_no_second_injection_right_after_compaction_report() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.50, step_no=1))) == 1
    await provider.on_context_compacted()
    assert len(await provider.get_injections([], _mock_soul(usage=0.10, step_no=21))) == 1
    # The one-shot is consumed: same low usage must not re-fire on the next step.
    result = await provider.get_injections([], _mock_soul(usage=0.11, step_no=22))
    assert result == []


async def test_compaction_before_first_injection_bypasses_min_usage() -> None:
    """Compaction before any injection: report the fresh low usage once."""
    provider = _make_provider()
    await provider.on_context_compacted()
    result = await provider.get_injections([], _mock_soul(usage=0.10, step_no=1))
    assert len(result) == 1


async def test_compaction_reinject_requires_material_change_after_cooldown() -> None:
    """After the one-shot report, normal throttling applies again."""
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.50, step_no=1))) == 1
    await provider.on_context_compacted()
    assert len(await provider.get_injections([], _mock_soul(usage=0.10, step_no=21))) == 1
    # After cooldown (step 51) with only a tiny growth -> still silent.
    result = await provider.get_injections([], _mock_soul(usage=0.12, step_no=51))
    assert result == []
    # Material growth after cooldown -> fires.
    result = await provider.get_injections([], _mock_soul(usage=0.30, step_no=51))
    assert len(result) == 1


# ── AFK ────────────────────────────────────────────────────────────────


async def test_afk_reanchors_and_fires_once_on_resume() -> None:
    provider = _make_provider()
    assert len(await provider.get_injections([], _mock_soul(usage=0.25, step_no=1))) == 1
    # AFK toggle re-anchors the meter so it fires once when the agent resumes.
    await provider.on_afk_changed(True)
    result = await provider.get_injections([], _mock_soul(usage=0.26, step_no=2))
    assert len(result) == 1
    # Throttling returns immediately after the one-shot re-anchor.
    result = await provider.get_injections([], _mock_soul(usage=0.27, step_no=3))
    assert result == []


async def test_afk_resume_respects_min_usage() -> None:
    """Resuming into a nearly-empty session must not nag."""
    provider = _make_provider()
    await provider.on_afk_changed(True)
    result = await provider.get_injections([], _mock_soul(usage=0.05, step_no=1))
    assert result == []


# ── Strip-cycle regressions ────────────────────────────────────────────
# The harness strips stale reminders from history on every step before
# collecting injections. This must NOT reset the provider's throttle state,
# or the meter would be re-injected on every single step.


def _append_reminder(history: list[Message], injections) -> None:
    for inj in injections:
        history.append(Message(role="user", content=[system_reminder(inj.content)]))


async def test_strip_cycle_does_not_reinject_every_step() -> None:
    """Fixed harness: strip reminders each step WITHOUT resetting providers."""
    provider = _make_provider()
    history: list[Message] = []
    fired: list[int] = []

    usage = 0.25
    for step in range(1, 41):
        strip_system_reminders(history)  # old reminders removed, no provider reset
        injections = await provider.get_injections(
            list(history), _mock_soul(usage=usage, step_no=step)
        )
        if injections:
            fired.append(step)
            _append_reminder(history, injections)
        if step == 20:
            usage = 0.45  # material growth after step 20

    # Fires at step 1 (first injection), then only when BOTH cooldown (30 steps)
    # and delta (>= 15%) are satisfied — never on every step.
    assert fired == [1, 31]


async def test_strip_cycle_never_more_than_one_per_cooldown_window() -> None:
    """Even a hostile harness that resets providers on strip stays throttled
    by the delta guard (defense in depth)."""
    provider = _make_provider()
    history: list[Message] = []
    fired: list[int] = []

    usage = 0.25
    for step in range(1, 41):
        strip_system_reminders(history)
        await provider.on_context_compacted()  # hostile per-step reset
        injections = await provider.get_injections(
            list(history), _mock_soul(usage=usage, step_no=step)
        )
        if injections:
            fired.append(step)
            _append_reminder(history, injections)
        if step == 20:
            usage = 0.45

    # Identical outcome: the delta guard keeps the meter quiet until usage
    # materially changes, no matter how often the reset hook is invoked.
    assert fired == [1, 31]


async def test_real_compaction_within_strip_cycle_fires_once() -> None:
    """A genuine compaction inside the step loop yields exactly one
    post-compaction report, then the meter stays quiet until usage changes."""
    provider = _make_provider()
    history: list[Message] = []
    fired: list[int] = []

    for step in range(1, 26):
        if step == 21:
            await provider.on_context_compacted()  # real compaction
            history.clear()
        strip_system_reminders(history)
        usage = 0.10 if step >= 21 else 0.50
        injections = await provider.get_injections(
            list(history), _mock_soul(usage=usage, step_no=step)
        )
        if injections:
            fired.append(step)
            _append_reminder(history, injections)

    # Step 1: first injection. Step 21: one-shot post-compaction report.
    assert fired == [1, 21]
