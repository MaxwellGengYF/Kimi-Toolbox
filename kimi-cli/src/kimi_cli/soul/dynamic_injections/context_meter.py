"""Always-on context-usage telemetry for agent self-regulation.

Research shows agents manage memory better when context pressure is
*observable*: a cheap status line lets the model pick a good moment to
checkpoint, flush durable memory, or start a fresh session — instead of the
harness compacting at a bad time.

This provider injects a reminder to persist important facts with the ``Memory``
tool when usage has materially changed. It deliberately stays quiet in the
high-usage region, which is owned by :class:`CompactReminderProvider`
(actionable advice), so the two never double-inject.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kosong.message import Message

from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_CONTEXT_METER_TYPE = "context_meter"


class ContextMeterProvider(DynamicInjectionProvider):
    """Injects a reminder to use the ``Memory`` tool when context usage materially changes."""

    def __init__(
        self,
        *,
        min_delta: float = 0.05,
        cooldown_steps: int = 5,
        suppress_above: float | None = 0.70,
    ) -> None:
        """
        Args:
            min_delta: Minimum usage-ratio change since the last injection
                required to inject again.
            cooldown_steps: Minimum steps between injections.
            suppress_above: Usage ratio at/above which this provider stays
                silent (the compact reminder owns that region). ``None``
                disables suppression.
        """
        self._min_delta = min_delta
        self._cooldown_steps = max(0, cooldown_steps)
        self._suppress_above = suppress_above
        self._last_injected_step: int | None = None
        self._last_injected_usage: float | None = None

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        _ = history

        # Only meter root sessions (skip subagents).
        if soul.is_subagent:
            return []

        max_tokens = soul.status.max_context_tokens
        if not max_tokens:
            return []
        tokens = soul.context.token_count_with_pending
        usage = tokens / max_tokens

        # High-usage region belongs to the compact reminder.
        if self._suppress_above is not None and usage >= self._suppress_above:
            return []

        step_no = soul._current_step_no
        if self._last_injected_step is not None and self._last_injected_usage is not None:
            steps_since = step_no - self._last_injected_step
            usage_delta = abs(usage - self._last_injected_usage)
            if steps_since < self._cooldown_steps or usage_delta < self._min_delta:
                return []

        self._last_injected_step = step_no
        self._last_injected_usage = usage

        content = (
            "Context is volatile — persist important facts with the `Memory` tool; "
            "they survive compaction."
        )
        return [DynamicInjection(type=_CONTEXT_METER_TYPE, content=content)]

    async def on_context_compacted(self) -> None:
        """Reset so the post-compaction (much lower) usage is reported once."""
        self._last_injected_step = None
        self._last_injected_usage = None

    async def on_afk_changed(self, enabled: bool) -> None:
        _ = enabled
        self._last_injected_step = None
        self._last_injected_usage = None
