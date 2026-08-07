from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kosong.message import Message, ToolCall

from kimi_cli.native_loader import kernel_module
from kimi_cli.notifications.llm import is_notification_message
from kimi_cli.soul.message import (
    coalesce_tool_metadata,
    is_system_reminder_message,
    system,
)
from kimi_cli.tools.file.micro_compress import (
    MicroCompressConfig,
    compress as _mc_compress,
)
from kimi_cli.utils.tokens import count_message_tokens, count_tokens
from kimi_cli.wire.types import ContentPart, TextPart


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ElidedRecord:
    """A record of a Tier B elision for archival/retrieval indexing.

    The original content is stored so it can be re-indexed for retrieval.
    """

    index: int
    """Index in the original (unpruned) history."""
    role: str
    """Message role ('assistant', 'tool', etc.)."""
    kind: str
    """Elision category, e.g. 'superseded_read', 'oversized_output'."""
    summary: str
    """Short human-readable summary of the elided content."""
    original_text: str
    """The original text content that was elided."""
    ref: str
    """Stable reference ID (aligned with HistoryIndex turn_id concept)."""


@dataclass
class PruningResult:
    """Result of a prune pass."""

    messages: list[Message]
    """New LLM-visible message list (Tier A dropped, Tier B stubbed)."""
    elided: list[ElidedRecord]
    """Tier B originals for archiving/indexing (Tier A needs none)."""
    freed_tokens: int
    """Estimated number of tokens freed by this prune pass."""
    earliest_removed_index: int | None
    """The earliest (smallest) index at which a change was made.
    ``None`` if nothing was removed/elided. Used for cache-depth logging."""


# ---------------------------------------------------------------------------
# Native shim helpers
# ---------------------------------------------------------------------------


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Serialize a kosong Message to the dict schema used by kimix_native.soul."""
    d: dict[str, Any] = {"role": message.role}
    content = message.content
    if isinstance(content, str):
        d["content"] = content
    elif isinstance(content, (list, tuple)):
        d["content"] = [part.model_dump(exclude_none=True) for part in content]
    elif content is not None:
        d["content"] = [content.model_dump(exclude_none=True)]
    if message.tool_calls:
        d["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
    if message.tool_call_id is not None:
        d["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        d["name"] = message.name
    if message.partial is not None:
        d["partial"] = message.partial
    return d


def _dict_to_message(d: dict[str, Any]) -> Message:
    """Reconstruct a kosong Message from a kimix_native.soul result dict."""
    content_in = d.get("content")
    if isinstance(content_in, str):
        content_out: list[Any] = [TextPart(text=content_in)]
    elif isinstance(content_in, (list, tuple)):
        content_out = []
        for part in content_in:
            if isinstance(part, dict):
                content_out.append(ContentPart.model_validate(part))
            else:
                content_out.append(TextPart(text=str(part)))
    else:
        content_out = []

    tool_calls = None
    if d.get("tool_calls"):
        tool_calls = [
            ToolCall(
                type=tc.get("type", "function"),
                id=tc.get("id", ""),
                function=ToolCall.FunctionBody(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments"),
                ),
            )
            for tc in d["tool_calls"]
        ]

    return Message(
        role=d["role"],
        content=content_out,
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
        partial=d.get("partial"),
    )


# ---------------------------------------------------------------------------
# Tier A — Ephemeral message detectors
# ---------------------------------------------------------------------------


def _is_active_task_snapshot_message(message: Message) -> bool:
    """Check if *message* is a post-compaction active-task snapshot."""
    if message.role != "user":
        return False
    text = ""
    for part in message.content:
        if isinstance(part, TextPart):
            text += part.text
    return "<active-background-tasks>" in text or "active background tasks" in text.lower()


def _is_dmail_notice_message(message: Message) -> bool:
    """Check if *message* is a D-Mail notice from the future self."""
    if message.role != "user":
        return False
    text = ""
    for part in message.content:
        if isinstance(part, TextPart):
            text += part.text
    return "D-Mail from your future self" in text


def _is_checkpoint_marker_message(message: Message) -> bool:
    """Check if *message* is a CHECKPOINT marker."""
    if message.role not in ("user", "system"):
        return False
    text = ""
    for part in message.content:
        if isinstance(part, TextPart):
            text += part.text
    text_stripped = text.strip()
    return "CHECKPOINT" in text and (text_stripped.startswith("<system>CHECKPOINT") or "<system>CHECKPOINT" in text_stripped)


def _is_ephemeral_message(
    message: Message,
    *,
    check_notifications: bool = True,
    check_task_snapshots: bool = True,
    check_dmail: bool = True,
    check_checkpoints: bool = False,
) -> bool:
    """Check if *message* is any kind of ephemeral injected message.

    This generalizes ``is_system_reminder_message`` to cover all
    auto-injected accumulating ephemera.
    """
    if is_system_reminder_message(message):
        return True
    if check_notifications and is_notification_message(message):
        return True
    if check_task_snapshots and _is_active_task_snapshot_message(message):
        return True
    if check_dmail and _is_dmail_notice_message(message):
        return True
    if check_checkpoints and _is_checkpoint_marker_message(message):
        return True
    return False


# ---------------------------------------------------------------------------
# Protected set helpers
# ---------------------------------------------------------------------------


def _protect_tool_pair_indices(
    history: Sequence[Message],
    protected: set[int],
) -> set[int]:
    """Extend *protected* so every tool-call pair in it is kept as a unit.

    For any protected assistant message with ``tool_calls``, the matching
    ``role="tool"`` result messages (by ``tool_call_id``) are also protected.
    This preserves provider invariants (OpenAI, Anthropic, Kimi) that require
    every assistant tool call to have a corresponding tool result.
    """
    protected = set(protected)
    n = len(history)
    assistant_tool_ids: dict[int, set[str]] = {}

    for idx in protected:
        msg = history[idx]
        if msg.role == "assistant" and msg.tool_calls:
            assistant_tool_ids[idx] = {tc.id for tc in msg.tool_calls}

    for idx, tool_ids in assistant_tool_ids.items():
        # Tool results normally follow immediately after the assistant call,
        # but scan the whole history to be robust to reordering/normalization.
        for j in range(n):
            if j in protected:
                continue
            candidate = history[j]
            if candidate.role == "tool" and candidate.tool_call_id in tool_ids:
                protected.add(j)

    return protected


def _compute_protected_indices(
    history: Sequence[Message],
    *,
    stable_prefix_messages: int,
    recent_messages_protected: int,
    current_turn_index: int | None = None,
) -> set[int]:
    """Compute the set of protected indices that must never be pruned.

    Includes:
    - First ``stable_prefix_messages`` messages (head stability).
    - Last ``recent_messages_protected`` user/assistant turns + their tool
      messages (recency window).
    - Current turn's user message and anything appended this turn.
    - Any assistant-with-tool_calls whose tool responses lie in the
      protected tail (protected as a unit).
    """
    protected: set[int] = set()
    n = len(history)

    # Head protection
    for i in range(min(stable_prefix_messages, n)):
        protected.add(i)

    # Tail protection — find last K user/assistant turns
    tail_turn_indices: list[int] = []
    for i in range(n - 1, -1, -1):
        if len(tail_turn_indices) >= recent_messages_protected:
            break
        if history[i].role in ("user", "assistant"):
            tail_turn_indices.append(i)

    # Add tail turn indices
    for idx in tail_turn_indices:
        protected.add(idx)

    # Protect tool-call pairs for any protected assistant message
    protected = _protect_tool_pair_indices(history, protected)

    # Current turn protection
    if current_turn_index is not None:
        for i in range(current_turn_index, n):
            protected.add(i)

    return protected


# ---------------------------------------------------------------------------
# Tier A — Ephemeral candidate selection
# ---------------------------------------------------------------------------


def _tier_a_candidates(
    history: Sequence[Message],
    protected: set[int],
    *,
    drop_notifications: bool = True,
    drop_task_snapshots: bool = True,
    drop_dmail: bool = True,
    drop_checkpoints: bool = False,
) -> list[tuple[int, int]]:
    """Find Tier A (ephemeral) drop candidates, returning ``(index, savings)``.

    Only messages *outside* the protected set are considered.
    For task snapshots, only the most recent one is kept.
    """
    candidates: list[tuple[int, int]] = []

    # First pass: collect all ephemeral messages outside protected
    ephemeral_indices: list[int] = []
    for i in range(len(history)):
        if i in protected:
            continue
        if _is_ephemeral_message(
            history[i],
            check_notifications=drop_notifications,
            check_task_snapshots=drop_task_snapshots,
            check_dmail=drop_dmail,
            check_checkpoints=drop_checkpoints,
        ):
            ephemeral_indices.append(i)

    if not ephemeral_indices:
        return []

    # For task snapshots: keep only the most recent one
    if drop_task_snapshots:
        snapshot_indices = [
            i for i in ephemeral_indices if _is_active_task_snapshot_message(history[i])
        ]
        if len(snapshot_indices) > 1:
            # Keep the latest (highest index), drop the rest
            latest_snapshot = max(snapshot_indices)
            for idx in snapshot_indices:
                if idx != latest_snapshot:
                    tokens = len(history[idx].content[0].text) // 4 if history[idx].content else 0
                    candidates.append((idx, max(tokens, 1)))

    # Add all other ephemeral messages (notifications, dmail, etc.)
    for idx in ephemeral_indices:
        if _is_active_task_snapshot_message(history[idx]):
            continue  # handled above
        # Estimate token savings
        tokens = len(history[idx].content[0].text) // 4 if history[idx].content else 0
        candidates.append((idx, max(tokens, 1)))

    return candidates


# ---------------------------------------------------------------------------
# Tier B — Substantive elision detectors (stubs, Phase 3)
# ---------------------------------------------------------------------------


def _is_superseded_read(
    history: Sequence[Message],
    index: int,
) -> tuple[bool, str, int]:
    """Check if a tool result at *index* is a superseded read operation.

    Returns ``(is_superseded, kind, savings)``.
    """
    msg = history[index]
    if msg.role != "tool":
        return (False, "", 0)

    text = ""
    for part in msg.content:
        if isinstance(part, TextPart):
            text += part.text

    if not text.strip():
        return (False, "", 0)

    # Rough estimate of savings
    savings = max(len(text) // 4, 1)

    # Check if a later tool result for the same path exists (simple heuristic)
    # Look for a later tool result with similar content
    for j in range(index + 1, len(history)):
        if history[j].role == "tool":
            later_text = ""
            for part in history[j].content:
                if isinstance(part, TextPart):
                    later_text += part.text
            # If later result is shorter/success message after error, mark as superseded
            if later_text and "Tool output is empty" in later_text:
                return (True, "superseded_read", savings)
            if later_text and len(later_text) < len(text) // 2:
                return (True, "superseded_read", savings)

    return (False, "", 0)


def _is_oversized_output(
    history: Sequence[Message],
    index: int,
    min_tokens: int = 512,
) -> tuple[bool, str, int]:
    """Check if a tool result at *index* is oversized.

    Returns ``(is_oversized, kind, savings)``.
    """
    msg = history[index]
    if msg.role != "tool":
        return (False, "", 0)

    text = ""
    for part in msg.content:
        if isinstance(part, TextPart):
            text += part.text

    token_count = max(len(text) // 4, 1)
    if token_count >= min_tokens:
        return (True, "oversized_output", token_count)

    return (False, "", 0)


def _is_resolved_error(
    history: Sequence[Message],
    index: int,
) -> tuple[bool, str, int]:
    """Check if a tool result at *index* is an error that was later resolved.

    Returns ``(is_resolved, kind, savings)``.
    """
    msg = history[index]
    if msg.role != "tool":
        return (False, "", 0)

    text = ""
    for part in msg.content:
        if isinstance(part, TextPart):
            text += part.text

    if "<system>ERROR:" not in text:
        return (False, "", 0)

    savings = max(len(text) // 4, 1)

    # Check if a later same-tool success exists
    for j in range(index + 1, len(history)):
        if history[j].role == "tool":
            later_text = ""
            for part in history[j].content:
                if isinstance(part, TextPart):
                    later_text += part.text
            if later_text and "<system>ERROR:" not in later_text:
                return (True, "resolved_error", savings)

    return (False, "", 0)


def _tier_b_candidates(
    history: Sequence[Message],
    protected: set[int],
    *,
    min_output_tokens: int = 512,
) -> list[tuple[int, int, str]]:
    """Find Tier B (substantive elision) candidates.

    Returns ``(index, savings, kind)`` tuples.
    """
    candidates: list[tuple[int, int, str]] = []

    for i in range(len(history)):
        if i in protected:
            continue

        # Superseded reads
        is_sup, kind, savings = _is_superseded_read(history, i)
        if is_sup:
            candidates.append((i, savings, kind))
            continue

        # Oversized outputs
        is_oversized, kind, savings = _is_oversized_output(history, i, min_tokens=min_output_tokens)
        if is_oversized:
            candidates.append((i, savings, kind))
            continue

        # Resolved errors
        is_resolved, kind, savings = _is_resolved_error(history, i)
        if is_resolved:
            candidates.append((i, savings, kind))

    return candidates


# ---------------------------------------------------------------------------
# Tier C — Micro-compress in place (plan.md §8.3, Phase 4)
# ---------------------------------------------------------------------------

# ReadFile-style line-number prefix (``{n:6d}\t``), mirrored from
# micro_compress.Stage 5 so history-time detection stays independent.
_LINENO_RE = re.compile(r"^\s*(\d+)\t")

# Markers emitted by the *annotated* micro-compress stages (3-A3, 4, 6, 7,
# 8, 9).  Tier C emits an ElidedRecord only when one of these appears in the
# compressed text; pure-lossless stages (1, 2, 3-A1/A2/A4, 5) are applied
# silently because they are reversible with no information loss.
_ANNOTATED_MARKERS = (
    "[common-indent:",
    "[prefix:",
    "[ts-prefix folded",
    "banner lines dropped",
    "near-dup",
    "chars elided]",
    "license lines elided",
    "comment lines elided",
    "lines of generated content",
)


def _looks_like_readfile_output(text: str) -> bool:
    """True when *text* is ReadFile-style (every substantial line is ``N\t…``)."""
    lines = text.split("\n")
    substantial = 0
    numbered = 0
    for ln in lines:
        if not ln.strip():
            continue
        substantial += 1
        if _LINENO_RE.match(ln):
            numbered += 1
    return substantial > 0 and numbered == substantial


def _output_text(message: Message) -> str:
    """Concatenate the non-``<system>`` TextPart text of *message*.

    The ``<system>…</system>`` metadata parts are preserved untouched — they
    are Layer-1 boilerplate handled by ``coalesce_tool_metadata`` instead.
    """
    parts: list[str] = []
    for p in message.content:
        if isinstance(p, TextPart) and not p.text.strip().startswith("<system>"):
            parts.append(p.text)
    return "".join(parts)


def _has_annotated_marker(text: str) -> bool:
    """True when *text* contains a marker from an annotated compression stage."""
    return any(marker in text for marker in _ANNOTATED_MARKERS)


def _text_length(message: Message) -> int:
    """Total TextPart character length of *message* (for delta accounting)."""
    return sum(len(p.text) for p in message.content if isinstance(p, TextPart))


def _tier_c_candidates(
    history: Sequence[Message],
    excluded: set[int],
    *,
    min_saved_chars: int = 64,
) -> list[tuple[int, int, str]]:
    """Find Tier C (micro-compress in place) candidates.

    For stale ``role="tool"`` messages outside *excluded*, re-runs
    ``micro_compress.compress`` on the output ``TextPart`` text (system
    metadata parts are preserved) and keeps messages whose compressed form
    saves at least *min_saved_chars* characters.

    Because the transform is deterministic and idempotent, re-running on
    already-compressed text yields zero delta and is skipped automatically —
    safe to run on every prune pass.

    Returns ``(index, savings_tokens, content_kind)`` tuples.
    """
    candidates: list[tuple[int, int, str]] = []
    for i in range(len(history)):
        if i in excluded:
            continue
        msg = history[i]
        if msg.role != "tool":
            continue
        out_text = _output_text(msg)
        if not out_text.strip():
            continue
        # ReadFile-style content is treated as code: line numbers survive and
        # the destructive whitespace/prefix stages stay disabled.
        kind = "code" if _looks_like_readfile_output(out_text) else "log"
        compressed = _mc_compress(out_text, kind=kind, config=MicroCompressConfig())
        saved_chars = len(out_text) - len(compressed)
        if saved_chars >= min_saved_chars:
            candidates.append((i, max(saved_chars // 4, 1), kind))
    return candidates


def _apply_tier_c(
    history: Sequence[Message],
    excluded: set[int],
    *,
    min_saved_chars: int = 64,
    ref_counter: int = 0,
) -> tuple[list[Message], list[ElidedRecord], int, set[int], int]:
    """Micro-compress stale tool messages in place (Tier C, plan.md §8.3).

    Every ``role="tool"`` message outside *excluded* whose output text
    compresses by at least *min_saved_chars* characters is replaced by a
    deep copy carrying the compacted text.  The message role, ``tool_call_id``
    and non-text parts (images, thinking) are preserved; only the output
    ``TextPart``(s) are rewritten.

    An :class:`ElidedRecord` with ``kind="micro_compress"`` is emitted only
    when an *annotated* stage actually fired (Stages 4/6/7/8 — markers such as
    ``[prefix: …]``, ``[N banner lines dropped]``, ``[×k near-dup …]``);
    lossless-only changes (Stages 1-3/5) are applied silently.  The original
    text is archived on the record so ``Memory``/``HistoryIndex`` retrieval
    stays lossless.

    Returns ``(work_history, records, freed_tokens, changed_indices, next_ref)``.
    The caller's *history* is never mutated (changed messages are copies).
    """
    candidates = _tier_c_candidates(
        history, excluded, min_saved_chars=min_saved_chars
    )
    if not candidates:
        return list(history), [], 0, set(), ref_counter

    by_index = {i: kind for i, _, kind in candidates}
    work: list[Message] = []
    records: list[ElidedRecord] = []
    freed = 0
    changed: set[int] = set()

    for i, msg in enumerate(history):
        if i not in by_index:
            work.append(msg)
            continue

        sys_parts = [
            p
            for p in msg.content
            if isinstance(p, TextPart) and p.text.strip().startswith("<system>")
        ]
        out_parts = [
            p
            for p in msg.content
            if isinstance(p, TextPart) and not p.text.strip().startswith("<system>")
        ]
        non_text = [p for p in msg.content if not isinstance(p, TextPart)]
        text = "".join(p.text for p in out_parts)

        compressed = _mc_compress(
            text, kind=by_index[i], config=MicroCompressConfig()
        )
        saved_chars = len(text) - len(compressed)
        if saved_chars <= 0 or compressed == text:
            work.append(msg)
            continue

        new_content: list[ContentPart] = [*sys_parts]
        if compressed:
            new_content.append(TextPart(text=compressed))
        new_content.extend(non_text)

        new_msg = msg.model_copy(deep=True)
        new_msg.content = new_content
        work.append(new_msg)
        freed += max(saved_chars // 4, 1)
        changed.add(i)

        if _has_annotated_marker(compressed):
            ref = f"prune_{ref_counter}"
            ref_counter += 1
            records.append(
                ElidedRecord(
                    index=i,
                    role=msg.role,
                    kind="micro_compress",
                    summary=f"micro-compressed {saved_chars} chars at index {i}",
                    original_text=text,
                    ref=ref,
                )
            )

    return work, records, freed, changed, ref_counter


# ---------------------------------------------------------------------------
# Main pruner class
# ---------------------------------------------------------------------------


class ContextPruner:
    """Smart context history removal system.

    Runs inside `_step`, right where ``strip_system_reminders`` already runs,
    so pruning and the existing reminder churn share **one** cache-break event.

    **Three tiers:**

    * **Tier A — Ephemeral injected messages** (primary, safest, default).
      Drops consumed/superseded accumulating ephemera (notifications, task
      snapshots, D-Mail notices) from the LLM-visible history. No tool pairing,
      negligible long-term value → dropped outright.
    * **Tier C — Micro-compress in place** (plan.md §8.3, Phase 4). Re-runs
      the deterministic, idempotent ``micro_compress`` pipeline on *stale*
      surviving tool messages, shrinking redundant whitespace, prefixes,
      banners and repetition *inside* text that is otherwise kept verbatim.
      Lossless-only changes are applied silently; annotated stages emit an
      ``ElidedRecord`` with ``kind="micro_compress"``. Cheapest tier — runs
      before Tier B and works with both the native and Python paths.
    * **Tier B — Stale/oversized substantive content** (escalation only).
      Elides (not deletes) superseded reads, oversized tool outputs, resolved
      errors — replaces with a compact stub + retrieval ref.

    **Cache-conservative policy:**
    1. Protect the recent tail (hot cache + high value).
    2. Protect a stable head (long permanent-cached prefix).
    3. Prune only the middle band, tail-inward.
    4. Rare + batched (cooldown).
    5. Min-payoff gate.
    6. Deterministic + idempotent.
    7. Prefer Tier A over Tier B.
    8. Piggyback the existing break.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        trigger_ratio: float = 0.0,
        target_ratio: float = 0.0,
        stable_prefix_messages: int = 4,
        recent_messages_protected: int = 6,
        min_free_tokens: int = 2_000,
        cooldown_steps: int = 4,
        min_usage_growth: float = 0.05,
        max_fraction_per_pass: float = 0.5,
        ephemeral_enabled: bool = True,
        ephemeral_notifications: bool = True,
        ephemeral_task_snapshots: bool = True,
        ephemeral_dmail_notices: bool = True,
        ephemeral_checkpoint_markers: bool = False,
        substantive_enabled: bool = True,
        tool_output_min_tokens: int = 512,
        micro_compress_enabled: bool = False,
        micro_compress_min_saved_chars: int = 64,
    ) -> None:
        self._enabled = enabled
        self._trigger_ratio = trigger_ratio
        self._target_ratio = target_ratio
        self._stable_prefix_messages = stable_prefix_messages
        self._recent_messages_protected = recent_messages_protected
        self._min_free_tokens = min_free_tokens
        self._cooldown_steps = cooldown_steps
        self._min_usage_growth = min_usage_growth
        self._max_fraction_per_pass = max_fraction_per_pass

        # Tier A toggles
        self._ephemeral_enabled = ephemeral_enabled
        self._ephemeral_notifications = ephemeral_notifications
        self._ephemeral_task_snapshots = ephemeral_task_snapshots
        self._ephemeral_dmail_notices = ephemeral_dmail_notices
        self._ephemeral_checkpoint_markers = ephemeral_checkpoint_markers

        # Tier B toggles
        self._substantive_enabled = substantive_enabled
        self._tool_output_min_tokens = tool_output_min_tokens

        # Tier C toggles (micro-compress in place, plan.md §8.3)
        self._micro_compress_enabled = micro_compress_enabled
        self._micro_compress_min_saved_chars = micro_compress_min_saved_chars

        # Hysteresis state
        self._last_prune_step: int = -1
        self._last_prune_usage: float = 0.0
        self._ref_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        history: Sequence[Message],
        *,
        current_step: int = 0,
        context_usage: float = 0.0,
        max_context_size: int = 128_000,
        current_turn_index: int | None = None,
        model: str | None = None,
    ) -> PruningResult:
        """Run a prune pass on *history*.

        Args:
            history: The full message history.
            current_step: Current step number (for cooldown check).
            context_usage: Current context usage ratio (0.0 to 1.0).
            max_context_size: Maximum context size in tokens.
            current_turn_index: Index of the current turn's first message.
            model: Model name for token estimation.

        Returns:
            A ``PruningResult`` with the modified message list.
        """
        if not self._enabled:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Policy #4: Cooldown check
        if self._in_cooldown(current_step, context_usage):
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Policy: Trigger check
        if context_usage < self._trigger_ratio:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        target_tokens = int(max_context_size * self._target_ratio)
        current_tokens = count_message_tokens(history, model=model)
        budget = current_tokens - target_tokens

        if budget <= 0:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Cap budget by max_fraction_per_pass
        max_prune = int(current_tokens * self._max_fraction_per_pass)
        budget = min(budget, max_prune)

        # ── Tier C — micro-compress stale tool messages in place (plan §8.3) ──
        # Cheapest tier: deterministic, idempotent character-level compression
        # applied *before* any drop/elide decisions, so the A/B tiers operate
        # on already-compressed history.  Tier B candidates are excluded so
        # their stubs archive the true (uncompressed) original text.
        work_history: Sequence[Message] = history
        tier_c_records: list[ElidedRecord] = []
        tier_c_freed = 0
        tier_c_changes: set[int] = set()
        if self._micro_compress_enabled:
            protected = _compute_protected_indices(
                history,
                stable_prefix_messages=self._stable_prefix_messages,
                recent_messages_protected=self._recent_messages_protected,
                current_turn_index=current_turn_index,
            )
            excluded = set(protected)
            if self._substantive_enabled:
                excluded |= {
                    idx
                    for idx, _, _ in _tier_b_candidates(
                        history,
                        protected,
                        min_output_tokens=self._tool_output_min_tokens,
                    )
                }
            work_history, tier_c_records, tier_c_freed, tier_c_changes, next_ref = (
                _apply_tier_c(
                    history,
                    excluded,
                    min_saved_chars=self._micro_compress_min_saved_chars,
                    ref_counter=self._ref_counter,
                )
            )
            self._ref_counter = next_ref

        # Native fast path (GIL-released C++ kernel) on the compressed history
        soul_mod = kernel_module("SOUL")
        if soul_mod is not None:
            try:
                base = self._native_prune(
                    work_history,
                    budget,
                    current_turn_index=current_turn_index,
                )
            except Exception:
                # Fall through to the pure-Python implementation on any native
                # error so pruning never blocks the soul.
                base = self._python_prune(
                    work_history,
                    max_context_size=max_context_size,
                    current_turn_index=current_turn_index,
                    model=model,
                )
        else:
            base = self._python_prune(
                work_history,
                max_context_size=max_context_size,
                current_turn_index=current_turn_index,
                model=model,
            )

        return self._finalize_prune_result(
            history,
            base,
            tier_c_records=tier_c_records,
            tier_c_freed=tier_c_freed,
            tier_c_changes=tier_c_changes,
            current_step=current_step,
            context_usage=context_usage,
        )

    def _python_prune(
        self,
        history: Sequence[Message],
        *,
        max_context_size: int = 128_000,
        current_turn_index: int | None = None,
        model: str | None = None,
    ) -> PruningResult:
        """Pure-Python Tier A/B prune implementation.

        Operates on *history* (which may already carry Tier C in-place
        compression).  Returns a ``PruningResult`` for the A/B tiers only —
        the min-payoff gate and hysteresis are applied by
        :meth:`_finalize_prune_result` so Tier C savings count towards the
        combined pass.
        """
        target_tokens = int(max_context_size * self._target_ratio)
        current_tokens = count_message_tokens(history, model=model)
        budget = current_tokens - target_tokens

        if budget <= 0:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Cap budget by max_fraction_per_pass
        max_prune = int(current_tokens * self._max_fraction_per_pass)
        budget = min(budget, max_prune)

        # Compute protected set
        protected = _compute_protected_indices(
            history,
            stable_prefix_messages=self._stable_prefix_messages,
            recent_messages_protected=self._recent_messages_protected,
            current_turn_index=current_turn_index,
        )

        # Collect candidates
        candidates: list[tuple[int, int, str, str]] = []  # (index, savings, tier, kind)

        # Tier A
        if self._ephemeral_enabled:
            tier_a = _tier_a_candidates(
                history,
                protected,
                drop_notifications=self._ephemeral_notifications,
                drop_task_snapshots=self._ephemeral_task_snapshots,
                drop_dmail=self._ephemeral_dmail_notices,
                drop_checkpoints=self._ephemeral_checkpoint_markers,
            )
            for idx, savings in tier_a:
                candidates.append((idx, savings, "A", "ephemeral"))

        # Tier B (only if Tier A alone is insufficient and we're near compaction)
        tier_a_savings = sum(s for _, s, _, _ in candidates if _[2] == "A")
        need_more = budget - tier_a_savings
        if need_more > 0 and self._substantive_enabled:
            tier_b = _tier_b_candidates(
                history,
                protected,
                min_output_tokens=self._tool_output_min_tokens,
            )
            for idx, savings, kind in tier_b:
                # Avoid duplicates (already in Tier A)
                if any(c[0] == idx for c in candidates):
                    continue
                candidates.append((idx, savings, "B", kind))

        if not candidates:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Policy #3: Tail-inward selection — prefer latest-index first
        # Policy #7: Prefer Tier A over Tier B (Tier C already ran as a pre-pass)
        candidates.sort(key=lambda x: (-x[0], 0 if x[2] == "A" else 1, -x[1]))

        # Greedy selection
        selected_indices: set[int] = set()
        total_freed = 0
        for idx, savings, tier, kind in candidates:
            if total_freed >= budget:
                break
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            total_freed += savings

        # Build result
        result_messages: list[Message] = []
        elided_records: list[ElidedRecord] = []
        changes: set[int] = set()

        for i, msg in enumerate(history):
            if i in selected_indices:
                changes.add(i)
                # Check if Tier A (drop) or Tier B (elide)
                is_tier_a = _is_ephemeral_message(
                    history[i],
                    check_notifications=self._ephemeral_notifications,
                    check_task_snapshots=self._ephemeral_task_snapshots,
                    check_dmail=self._ephemeral_dmail_notices,
                    check_checkpoints=self._ephemeral_checkpoint_markers,
                )
                if is_tier_a:
                    # Tier A: drop the message entirely
                    continue
                else:
                    # Tier B: elide — replace content with stub
                    text = ""
                    for part in msg.content:
                        if isinstance(part, TextPart):
                            text += part.text

                    kind = "elided"
                    for _idx, _sav, _tier, _kind in candidates:
                        if _idx == i:
                            kind = _kind
                            break

                    ref = f"prune_{self._ref_counter}"
                    self._ref_counter += 1

                    stub_text = (
                        f"<system>[context-elided: {kind} — content elided. "
                        f"~{savings} tokens freed. "
                        f"Retrieve full content with Memory action='retrieve' id={ref}]</system>"
                    )

                    elided_records.append(
                        ElidedRecord(
                            index=i,
                            role=msg.role,
                            kind=kind,
                            summary=f"{kind} at index {i}",
                            original_text=text,
                            ref=ref,
                        )
                    )

                    result_messages.append(
                        Message(
                            role=msg.role,
                            content=[TextPart(text=stub_text)],
                            tool_call_id=msg.tool_call_id,
                        )
                    )
            else:
                result_messages.append(msg)

        earliest = min(changes) if changes else None

        return PruningResult(
            messages=result_messages,
            elided=elided_records,
            freed_tokens=total_freed,
            earliest_removed_index=earliest,
        )

    def _finalize_prune_result(
        self,
        original_history: Sequence[Message],
        base: PruningResult,
        *,
        tier_c_records: list[ElidedRecord],
        tier_c_freed: int,
        tier_c_changes: set[int],
        current_step: int,
        context_usage: float,
    ) -> PruningResult:
        """Merge Tier C in-place compression and Layer 1 metadata coalescing
        into the Tier A/B result, then apply the min-payoff gate (policy #5).

        If the *combined* pass (Tier C + A/B + metadata coalescing) fails the
        gate, the original history is returned untouched — Tier C changes are
        rolled back with everything else.
        """
        messages = base.messages
        elided = [*tier_c_records, *base.elided]
        freed = tier_c_freed + base.freed_tokens
        changes = set(tier_c_changes)
        if base.earliest_removed_index is not None:
            changes.add(base.earliest_removed_index)

        if self._micro_compress_enabled:
            # Layer 1 (plan §8.2) — coalesce adjacent identical <system>
            # metadata (Class C3).  Operate on deep copies so the caller's
            # history is never mutated.
            coalesce_work = [m.model_copy(deep=True) for m in messages]
            before = [_text_length(m) for m in coalesce_work]
            removed_parts = coalesce_tool_metadata(coalesce_work)
            after = [_text_length(m) for m in coalesce_work]
            coalesced_chars = 0
            coalesced_first: int | None = None
            for i, (b, a) in enumerate(zip(before, after)):
                if b > a:
                    coalesced_chars += b - a
                    if coalesced_first is None:
                        coalesced_first = i
            if removed_parts and coalesced_chars:
                messages = coalesce_work
                freed += max(coalesced_chars // 4, 1)
                if coalesced_first is not None:
                    changes.add(coalesced_first)

        # Policy #5: Min-payoff gate (whole pass, including Tier C)
        if not changes or freed < self._min_free_tokens:
            return PruningResult(
                messages=list(original_history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # Update hysteresis
        self._last_prune_step = current_step
        self._last_prune_usage = context_usage

        return PruningResult(
            messages=messages,
            elided=elided,
            freed_tokens=freed,
            earliest_removed_index=min(changes),
        )

    def estimate_after_prune(
        self,
        history: Sequence[Message],
        *,
        context_usage: float = 0.0,
        max_context_size: int = 128_000,
        current_step: int = 0,
        model: str | None = None,
    ) -> int:
        """Estimate token count after a prune pass without actually pruning.

        Returns the estimated token count of the pruned history.
        """
        result = self.prune(
            history,
            current_step=current_step,
            context_usage=context_usage,
            max_context_size=max_context_size,
            model=model,
        )
        if result.earliest_removed_index is None:
            return count_message_tokens(history, model=model)
        return count_message_tokens(result.messages, model=model)

    def prune_with_policy(
        self,
        history: Sequence[Message],
        *,
        remove_reasoning: bool = True,
        remove_tool_results: bool = True,
        keep_recent_turns: int = 6,
        target_token_count: int | None = None,
        max_context_size: int = 128_000,
        current_step: int = 0,
        model: str | None = None,
    ) -> PruningResult:
        """Run a policy-driven prune pass suitable for manual invocation.

        This method configures a fresh :class:`ContextPruner` instance from the
        high-level parameters and runs ``prune()``.  It bypasses hysteresis so
        that a manual ``ContextPrune`` tool call always acts when content is
        available to drop/elide.

        Args:
            history: The full message history.
            remove_reasoning: For ``mode="prune"`` this is a no-op because the
                existing Tier-B detector only elides ``role="tool"`` messages;
                reasoning removal is handled by ``strip_reasoning`` mode.
            remove_tool_results: When ``False``, disable Tier-B substantive
                elision entirely (only ephemeral Tier-A drops are performed).
            keep_recent_turns: Number of recent user/assistant turns to protect.
            target_token_count: Optional explicit token target.  When supplied,
                the pruner's ``target_ratio`` is derived as
                ``target_token_count / max_context_size``.
            max_context_size: Maximum context size in tokens.
            current_step: Current step number (passed through for logging).
            model: Model name for token estimation.

        Returns:
            A ``PruningResult`` with the modified message list.
        """
        if target_token_count is not None:
            target_ratio = max(0.0, min(1.0, target_token_count / max_context_size))
        else:
            target_ratio = self._target_ratio if self._target_ratio > 0 else 0.5

        ephemeral_enabled = self._ephemeral_enabled
        substantive_enabled = self._substantive_enabled and remove_tool_results

        # Build a fresh pruner so manual invocation does not disturb the soul's
        # hysteresis state (cooldowns, last usage, etc.).
        pruner = ContextPruner(
            enabled=True,
            trigger_ratio=0.0,  # manual invocation always runs
            target_ratio=target_ratio,
            stable_prefix_messages=self._stable_prefix_messages,
            recent_messages_protected=keep_recent_turns,
            min_free_tokens=1,  # manual invocation bypasses min-payoff gate
            cooldown_steps=0,  # manual invocation bypasses cooldown
            min_usage_growth=0.0,
            max_fraction_per_pass=self._max_fraction_per_pass,
            ephemeral_enabled=ephemeral_enabled,
            ephemeral_notifications=self._ephemeral_notifications,
            ephemeral_task_snapshots=self._ephemeral_task_snapshots,
            ephemeral_dmail_notices=self._ephemeral_dmail_notices,
            ephemeral_checkpoint_markers=self._ephemeral_checkpoint_markers,
            substantive_enabled=substantive_enabled,
            tool_output_min_tokens=self._tool_output_min_tokens,
            micro_compress_enabled=self._micro_compress_enabled,
            micro_compress_min_saved_chars=self._micro_compress_min_saved_chars,
        )
        pruner._ref_counter = self._ref_counter

        result = pruner.prune(
            history,
            current_step=current_step,
            context_usage=1.0 if target_ratio > 0 else 0.0,
            max_context_size=max_context_size,
            model=model,
        )

        # Propagate the ref counter back so elided IDs stay unique across calls.
        self._ref_counter = pruner._ref_counter

        # Guard: even though the current Tier-B detector only elides tool
        # messages, if remove_reasoning is False we restore any assistant
        # message that might have been elided (e.g. future detectors).
        if not remove_reasoning and result.elided:
            elided_indices = {rec.index for rec in result.elided if rec.role == "assistant"}
            if elided_indices:
                elided_by_index = {rec.index: rec for rec in result.elided}
                restored_messages: list[Message] = []
                for i, msg in enumerate(history):
                    if i in elided_indices:
                        rec = elided_by_index[i]
                        restored_messages.append(
                            Message(
                                role=rec.role,
                                content=[TextPart(text=rec.original_text)],
                                tool_call_id=msg.tool_call_id,
                            )
                        )
                    else:
                        restored_messages.append(msg)
                kept_elided = [rec for rec in result.elided if rec.role != "assistant"]
                freed = sum(max(len(rec.original_text) // 4, 1) for rec in kept_elided)
                earliest = min({rec.index for rec in kept_elided}) if kept_elided else None
                return PruningResult(
                    messages=restored_messages,
                    elided=kept_elided,
                    freed_tokens=freed,
                    earliest_removed_index=earliest,
                )

        return result

    # ------------------------------------------------------------------
    # Native fast path
    # ------------------------------------------------------------------

    def _native_prune(
        self,
        history: Sequence[Message],
        budget: int,
        *,
        current_turn_index: int | None = None,
    ) -> PruningResult:
        """Run the C++ prune_history kernel and convert the result back."""
        soul_mod = kernel_module("SOUL")
        if soul_mod is None:
            raise RuntimeError("native SOUL module is not available")

        messages = [_message_to_dict(m) for m in history]

        policy = {
            "stable_prefix_messages": self._stable_prefix_messages,
            "recent_messages_protected": self._recent_messages_protected,
            "current_turn_index": current_turn_index,
            "drop_notifications": self._ephemeral_notifications,
            "drop_task_snapshots": self._ephemeral_task_snapshots,
            "drop_dmail": self._ephemeral_dmail_notices,
            "drop_checkpoints": self._ephemeral_checkpoint_markers,
            "max_elision_tokens": max(0, int(budget)),
            "tool_output_min_tokens": self._tool_output_min_tokens,
            "superseded_read_enabled": self._substantive_enabled,
            "oversized_output_enabled": self._substantive_enabled,
            "stale_tool_result_enabled": self._substantive_enabled,
        }

        result = soul_mod.prune_history(messages, policy)

        if not result["elided"] and result["freed_tokens"] == 0:
            return PruningResult(
                messages=list(history),
                elided=[],
                freed_tokens=0,
                earliest_removed_index=None,
            )

        # NOTE: the min-payoff gate is applied in _finalize_prune_result so
        # Tier C savings count towards the combined pass.

        elided_by_index = {rec["index"]: rec for rec in result["elided"]}
        result_messages_dicts = result["messages"]
        result_idx = 0
        out_messages: list[Message] = []
        elided_records: list[ElidedRecord] = []
        ref_offset = self._ref_counter

        for i, orig_dict in enumerate(messages):
            if (
                result_idx < len(result_messages_dicts)
                and result_messages_dicts[result_idx] is orig_dict
            ):
                out_messages.append(_dict_to_message(orig_dict))
                result_idx += 1
                continue

            if i not in elided_by_index:
                # Tier A drop
                removed_indices.append(i)
                continue

            rec = elided_by_index[i]
            old_ref = rec["ref"]
            new_ref = f"prune_{ref_offset + len(elided_records)}"
            rec["ref"] = new_ref

            stub_dict = result_messages_dicts[result_idx]
            stub_text = stub_dict["content"][0]["text"]
            stub_text = stub_text.replace(f"id={old_ref}", f"id={new_ref}")
            stub_dict["content"][0]["text"] = stub_text

            out_messages.append(_dict_to_message(stub_dict))
            elided_records.append(
                ElidedRecord(
                    index=i,
                    role=rec["role"],
                    kind=rec["kind"],
                    summary=rec["summary"],
                    original_text=rec["original_text"],
                    ref=new_ref,
                )
            )
            result_idx += 1

        self._ref_counter += len(elided_records)

        removed_indices.extend(rec.index for rec in elided_records)
        earliest = min(removed_indices) if removed_indices else None

        return PruningResult(
            messages=out_messages,
            elided=elided_records,
            freed_tokens=result["freed_tokens"],
            earliest_removed_index=earliest,
        )

    # ------------------------------------------------------------------
    # Hysteresis helpers
    # ------------------------------------------------------------------

    def _in_cooldown(self, current_step: int, current_usage: float) -> bool:
        """Check whether the pruner is in a cooldown period."""
        if self._last_prune_step < 0:
            return False

        # Step cooldown
        if current_step - self._last_prune_step < self._cooldown_steps:
            return True

        # Usage growth cooldown
        usage_growth = current_usage - self._last_prune_usage
        if usage_growth < self._min_usage_growth:
            return True

        return False

    def reset_cooldown(self) -> None:
        """Reset hysteresis state (e.g., after compaction)."""
        self._last_prune_step = -1
        self._last_prune_usage = 0.0


def is_pruned_stub(message: Message) -> bool:
    """Check if *message* is a Tier B elision stub."""
    for part in message.content:
        if isinstance(part, TextPart) and "[context-elided:" in part.text:
            return True
    return False