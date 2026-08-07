"""Unit tests for ContextMeterProvider (context-pressure telemetry)."""

from __future__ import annotations

from types import SimpleNamespace

from kimi_cli.soul.dynamic_injections.context_meter import (
    ContextMeterProvider,
)


def _soul(
    step_no: int,
    tokens: int,
    max_tokens: int = 200_000,
    *,
    subagent: bool = False,
) -> SimpleNamespace:
    status = SimpleNamespace(max_context_tokens=max_tokens)
    context = SimpleNamespace(token_count_with_pending=tokens)
    return SimpleNamespace(
        _current_step_no=step_no,
        is_subagent=subagent,
        status=status,
        context=context,
    )


class TestInjection:
    async def test_first_injection(self) -> None:
        provider = ContextMeterProvider()
        injections = await provider.get_injections([], _soul(1, 72_000))  # type: ignore[arg-type]
        assert len(injections) == 1
        assert injections[0].type == "context_meter"
        assert "Context is volatile" in injections[0].content
        assert "Memory" in injections[0].content

    async def test_subagent_skipped(self) -> None:
        provider = ContextMeterProvider()
        assert await provider.get_injections([], _soul(1, 72_000, subagent=True)) == []  # type: ignore[arg-type]

    async def test_no_max_tokens_skipped(self) -> None:
        provider = ContextMeterProvider()
        assert await provider.get_injections([], _soul(1, 100, max_tokens=0)) == []  # type: ignore[arg-type]

    async def test_suppressed_above_compact_threshold(self) -> None:
        """The high-usage region belongs to CompactReminderProvider."""
        provider = ContextMeterProvider(suppress_above=0.70)
        assert await provider.get_injections([], _soul(1, 150_000)) == []  # type: ignore[arg-type]

    async def test_suppression_disabled(self) -> None:
        provider = ContextMeterProvider(suppress_above=None)
        assert await provider.get_injections([], _soul(1, 150_000))  # type: ignore[arg-type]

    async def test_boundary_not_suppressed(self) -> None:
        provider = ContextMeterProvider(suppress_above=0.70)
        # 69.5% < 70% -> still meters
        assert await provider.get_injections([], _soul(1, 139_000))  # type: ignore[arg-type]


class TestThrottling:
    async def test_small_delta_throttled(self) -> None:
        provider = ContextMeterProvider(min_delta=0.05, cooldown_steps=2)
        assert await provider.get_injections([], _soul(1, 60_000))  # type: ignore[arg-type]
        # +2% (< 5% min_delta) -> throttled even past cooldown
        assert await provider.get_injections([], _soul(5, 64_000)) == []  # type: ignore[arg-type]

    async def test_large_delta_injects(self) -> None:
        provider = ContextMeterProvider(min_delta=0.05, cooldown_steps=2)
        await provider.get_injections([], _soul(1, 60_000))  # type: ignore[arg-type]
        # +10% (>= 5%) and >= 2 steps -> inject
        assert await provider.get_injections([], _soul(3, 80_000))  # type: ignore[arg-type]

    async def test_cooldown_blocks_even_with_delta(self) -> None:
        provider = ContextMeterProvider(min_delta=0.05, cooldown_steps=5)
        await provider.get_injections([], _soul(1, 60_000))  # type: ignore[arg-type]
        # +10% delta but only 1 step later -> throttled
        assert await provider.get_injections([], _soul(2, 80_000)) == []  # type: ignore[arg-type]

    async def test_usage_drop_also_reports(self) -> None:
        """Delta is absolute — a big drop (e.g. after manual pruning) is reported."""
        provider = ContextMeterProvider(min_delta=0.05, cooldown_steps=1)
        await provider.get_injections([], _soul(1, 100_000))  # type: ignore[arg-type]
        injections = await provider.get_injections([], _soul(2, 60_000))  # type: ignore[arg-type]
        assert injections
        assert "Context is volatile" in injections[0].content

    async def test_compaction_resets(self) -> None:
        """After a real compaction the fresh (much lower) usage is reported once."""
        provider = ContextMeterProvider(min_delta=0.5, cooldown_steps=100)
        await provider.get_injections([], _soul(1, 100_000))  # type: ignore[arg-type]
        await provider.on_context_compacted()
        # 100k (50%) -> 0 (0%): delta 0.5 >= min_delta bypasses the cooldown.
        assert await provider.get_injections([], _soul(2, 0))  # type: ignore[arg-type]
