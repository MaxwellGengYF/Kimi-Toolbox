"""Codex-parity context accounting.

Mirrors openai/codex so ChatGPT subscription models get the same usable window
here as they do in the official harness:

    codex-rs/protocol/src/openai_models.rs
      usable_context_window()    = context_window * effective_context_window_percent / 100
      auto_compact_token_limit() = context_window * 9 / 10
    codex-rs/core/src/session/context_window.rs
      token_limit_reached = scope_tokens >= auto_compact_token_limit + fallback_buffer
                         || active_tokens >= usable_context_window

Codex reserves output headroom as a flat 5% of the window; it never subtracts
the model's maximum output length (``ModelInfo`` has no ``max_output_tokens``
field at all). Neither can we: the Codex backend rejects explicit output-token
limits, so ``OpenAICodex._request_kwargs`` sends no ``max_output_tokens``
and the provider's ``_generation_kwargs`` carries no ``max_tokens`` key. Letting
``max_tokens`` shrink the input window would reserve ~129k tokens on a 272k
model to protect a request field that is never sent.

This module is intentionally dependency-free so the config layer
(:mod:`kimi_cli.config`) and tests can use it without an import cycle.
"""

from __future__ import annotations

from typing import Any

CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 95
"""``ModelInfo.effective_context_window_percent`` default in openai/codex."""
CODEX_AUTO_COMPACT_PERCENT = 90
"""``auto_compact_token_limit()`` is derived as 90% of the context window."""
CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS = 16_384
"""``token_budget.auto_compact_fallback_buffer_tokens`` for the GPT-5.6 family."""
CODEX_TOKEN_BUDGET_REMINDER_TOKENS = 6_144
"""``token_budget.reminder_threshold_tokens``: remaining-token point at which
codex injects its context-window reminder."""

CODEX_LOOP_CONTROL_KEYS = (
    "reserved_context_size",
    "compaction_trigger_ratio",
    "compact_reminder_threshold",
)
"""``LoopControl`` fields :func:`codex_loop_control` is allowed to fill in."""


def codex_trigger_point(max_context_size: int) -> int:
    """Token count at which codex forces a compaction for ``max_context_size``."""

    usable = max_context_size * CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT // 100
    buffered = (
        max_context_size * CODEX_AUTO_COMPACT_PERCENT // 100
        + CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS
    )
    return min(usable, buffered)


def codex_loop_control(max_context_size: int) -> dict[str, Any]:
    """Return the ``loop_control`` overrides that reproduce codex's compaction points.

    ``should_auto_compact`` fires on whichever comes first::

        token_count >= max_context_size * compaction_trigger_ratio
        token_count + min(max(tool_buffer, reserved_context_size, max_tokens + margin),
                          max_context_size - reserved_context_size) >= max_context_size

    Setting ``reserved_context_size`` to codex's buffered auto-compact limit caps
    the second rule's reservation at ``max_context_size - reserved_context_size``,
    which collapses the reserved rule to exactly that buffered limit and removes
    ``max_tokens`` / ``tool_call_buffer_tokens`` from the decision. Together with
    ``compaction_trigger_ratio`` = 95% the pair becomes::

        min(usable_context_window, auto_compact_token_limit + fallback_buffer)

    i.e. codex's ``token_limit_reached`` (see :func:`codex_trigger_point`).
    Returns an empty mapping when the context window is unknown, so the normal
    defaults stay in effect.
    """
    if max_context_size <= 0:
        return {}

    auto_compact_limit = max_context_size * CODEX_AUTO_COMPACT_PERCENT // 100
    # Codex adds the fallback buffer *on top of* its limit (grace room for the
    # model to checkpoint its notes) instead of subtracting it up front.
    buffered_limit = auto_compact_limit + CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS
    reminder_at = max(0, auto_compact_limit - CODEX_TOKEN_BUDGET_REMINDER_TOKENS)
    return {
        "reserved_context_size": max(1_000, buffered_limit),
        "compaction_trigger_ratio": CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT / 100,
        "compact_reminder_threshold": min(0.95, max(0.5, reminder_at / max_context_size)),
    }
