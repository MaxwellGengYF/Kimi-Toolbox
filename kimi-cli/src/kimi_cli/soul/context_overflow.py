"""Context-overflow detection and per-step recovery budget (Phase 4 §6.1).

Port of DSH's ``agent/request-error`` + ``{kind:'retry'}`` recovery loop: when a
provider confirms the context window was exceeded (a 4xx ``APIStatusError`` whose
message matches the overflow markers), the soul force-compacts and re-runs the
step instead of immediately interrupting the session with
``SessionRestartRequired``.

``classify_api_error`` (kimisoul.py) already returns ``"context_overflow"`` for
these errors; the marker list lives here so both paths share one source of truth
(importing from kimisoul would create a circular import).
"""

from __future__ import annotations

from kosong.chat_provider import APIStatusError

CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "max tokens",
    "maximum context",
    "too many tokens",
)
"""Substrings (matched case-insensitively) that identify a provider-confirmed
context-window-exceeded error message."""


def is_context_overflow_error(exc: BaseException) -> bool:
    """Return True for an ``APIStatusError`` 4xx whose message matches the markers.

    Mirrors ``classify_api_error``'s ``context_overflow`` branch: the status must
    be in ``[400, 500)`` and any marker must appear in the lowercased message.
    Statuses that ``classify_api_error`` resolves *before* the 4xx branch
    (401/403 → ``auth``, 429 → ``rate_limit``) are rejected here so the two
    functions stay consistent — a rate-limit or auth error is never an overflow,
    even when its body happens to mention tokens. 5xx, network errors, timeouts,
    and non-``APIStatusError`` exceptions are rejected as well.
    """
    if not isinstance(exc, APIStatusError):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", 0)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    if not (400 <= status < 500):
        return False
    if status in (401, 403, 429):
        # classify_api_error precedence: auth / rate_limit win over overflow.
        return False
    msg_lower = str(exc).lower()
    return any(marker in msg_lower for marker in CONTEXT_OVERFLOW_MARKERS)


class OverflowRecoveryState:
    """Per-step overflow retry budget (reset on each step begin / success).

    The soul creates one instance at the top of each *top-level* ``_step`` and
    shares it across the re-entrant ``_step`` calls made by the overflow recovery
    loop (passed as an explicit argument), so the budget is consumed across
    re-entries but naturally resets when the step completes.
    """

    def __init__(self, max_retries: int) -> None:
        # Negative/zero max_retries → never retry.
        self._max_retries = max(0, max_retries)
        self._remaining = self._max_retries

    def can_retry(self) -> bool:
        """True while at least one overflow retry is still available."""
        return self._remaining > 0

    def consumed(self) -> None:
        """Record that one overflow retry was used."""
        if self._remaining > 0:
            self._remaining -= 1

    def reset(self) -> None:
        """Restore the full retry budget."""
        self._remaining = self._max_retries
