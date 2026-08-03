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
        min_usage: float = 0.20,
    ) -> None:
        """
        Args:
            min_delta: Minimum usage-ratio change since the last injection
                required to inject again.
            cooldown_steps: Minimum steps between injections.
            suppress_above: Usage ratio at/above which this provider stays
                silent (the compact reminder owns that region). ``None``
                disables suppression.
            min_usage: Usage ratio threshold for the very first injection.
                Below this, the provider stays silent.
        """
        self._min_delta = min_delta
        self._cooldown_steps = max(0, cooldown_steps)
        self._suppress_above = suppress_above
        self._min_usage = min_usage
        self._last_injected_step: int | None = None
        self._last_injected_usage: float | None = None
        self._compaction_pending: bool = False

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

        # The post-compaction "fresh report" is a one-shot: consume the flag on
        # the first evaluation regardless of outcome. If it stayed armed until
        # the next injection, a harness that re-notifies providers on every
        # step would keep the cooldown bypass alive forever.
        compaction_pending = self._compaction_pending
        self._compaction_pending = False

        # ── Frequency guards ────────────────────────────────────────────────
        # The cooldown and delta guards below are the ONLY thing standing
        # between a periodic reminder and a per-step nag, so they must hold even
        # after on_context_compacted() / on_afk_changed() ran since the last
        # injection: those hooks never clear the throttle anchors here, so a
        # spurious reset can never re-arm the meter.
        if self._last_injected_step is not None:
            steps_since = step_no - self._last_injected_step
            cooldown_ok = steps_since >= self._cooldown_steps
            # A real compaction drops usage below the last injected level; only
            # then may the one-shot bypass the cooldown so the fresh low usage
            # is reported promptly. A spurious reset does not drop usage, so the
            # cooldown keeps applying.
            usage_dropped = bool(
                self._last_injected_usage is not None
                and usage < self._last_injected_usage
            )
            if not cooldown_ok and not (compaction_pending and usage_dropped):
                return []
            if self._last_injected_usage is not None:
                usage_delta = abs(usage - self._last_injected_usage)
                if usage_delta < self._min_delta:
                    return []
        elif not compaction_pending and usage < self._min_usage:
            # First injection (or right after an AFK resume): require a minimum
            # usage so we don't nag during a nearly-empty session.
            return []

        self._last_injected_step = step_no
        self._last_injected_usage = usage

        content = (
            "Context is volatile — persist important facts with the `Memory` tool; "
            "they survive compaction. When unsure about history or memory, "
            "recall past decisions, file paths, or errors with the "
            "`Memory` tool (action='retrieve'). "
            "There is no need to call the `Memory` tool frequently — this reminder "
            "only fires when context usage materially changes, so write or retrieve "
            "only when there is something genuinely important."
        )
        return [DynamicInjection(type=_CONTEXT_METER_TYPE, content=content)]

    async def on_context_compacted(self) -> None:
        """Mark that a real compaction happened so the fresh (much lower) usage
        is reported promptly, even below ``min_usage``.

        The throttle anchors (last step / last usage) are intentionally NOT
        cleared here: they keep the cooldown and delta guards active.
        ``_compaction_pending`` is a one-shot flag — it is consumed on the first
        evaluation and only bypasses the *cooldown* guard (the delta guard still
        applies) — so even if this hook were called on every step (e.g. if the
        harness started resetting providers when it strips stale reminders), the
        meter would stay quiet until both cooldown and a material usage change
        are satisfied.
        """
        self._compaction_pending = True

    async def on_afk_changed(self, enabled: bool) -> None:
        _ = enabled
        # Re-anchor after AFK so the meter fires once when the agent resumes
        # (still gated by ``min_usage`` via the first-injection path above).
        self._last_injected_step = None
        self._last_injected_usage = None
