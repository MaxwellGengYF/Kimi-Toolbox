from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, runtime_checkable

import kosong
from kosong.chat_provider import TokenUsage
from kosong.message import Message
from kosong.tooling import Tool
from kosong.tooling.empty import EmptyToolset

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.session_state import TodoItemState, format_todo_injection
from kimi_cli.soul.compaction_ledger import CompactionLedger, CompactionRecord
from kimi_cli.soul.llm_request_recorder import LLMRequestRecorder
from kimi_cli.soul.message import system
from kimi_cli.soul.tool_pairing import balanced_cut_indices, nearest_balanced_cut_before
from kimi_cli.utils.logging import logger
from kimi_cli.utils.tokens import count_message_tokens
from kimi_cli.wire.types import ContentPart, TextPart, ThinkPart


class CompactMode(str, Enum):
    """High-level compaction style presets."""

    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    RETENTIVE = "retentive"
    TECHNICAL = "technical"


_MODE_GUIDANCE: dict[CompactMode, str] = {
    CompactMode.BALANCED: (
        "**Compaction Style Guidance:** Be balanced. Preserve essential context "
        "while condensing redundant information. Keep current task state, errors "
        "and solutions, code state, design decisions, and TODO items."
    ),
    CompactMode.AGGRESSIVE: (
        "**Compaction Style Guidance:** Be aggressive. Prioritize brevity, drop "
        "intermediate attempts, exploratory dead-ends, and low-priority details. "
        "Keep only the essential facts, decisions, and current state."
    ),
    CompactMode.RETENTIVE: (
        "**Compaction Style Guidance:** Be retentive. Preserve more verbatim detail, "
        "especially recent reasoning steps, exact values, file paths, and user "
        "preferences. Do not over-compress."
    ),
    CompactMode.TECHNICAL: (
        "**Compaction Style Guidance:** Focus on technical specifics. Prioritize "
        "code snippets, file paths, error messages, stack traces, architectural "
        "decisions, and current implementation state. Summarize conversational filler."
    ),
}


@dataclass(frozen=True, slots=True)
class CompactionOptions:
    """Per-compaction options that do not override session-level preserve config."""

    avoid_cascade: bool = False
    mode: CompactMode = CompactMode.BALANCED
    todos_max_items: int | None = None
    """Maximum unfinished todos re-injected into the compaction output.
    ``None`` uses :func:`format_todo_injection`'s default (20)."""
    preserve_depth_override: int | None = None
    """Override the preserve depth for this compaction (bypasses adaptive/normal
    depth). None uses the configured depth."""


class SurfaceChangedError(Exception):
    """The conversation surface changed while the compaction LLM call was in flight.

    Raised when the caller mutates ``messages`` (length or last-message text)
    between the pre-call fingerprint snapshot and the post-call re-check, so a
    stale summary is never applied over a changed history.
    """


class CompactionShrinkError(Exception):
    """The compaction summary is not smaller than the content it replaces.

    Raised when ``summary_tokens >= shadowed_tokens`` (DSH ``region.ts``
    shrink check): applying the summary would not actually free context.
    """


class ManualCompactionError(Exception):
    """Classified failure of a user-requested (manual) compaction.

    ``slash.py /compact`` wraps :meth:`KimiSoul.compact_context` and maps the
    codes to user-facing messages: ``changed`` → history changed during
    compaction; ``summary`` → summary not smaller than compacted content;
    ``commit``/``persistence`` → compaction did not commit cleanly;
    ``busy`` → compaction already in progress; ``cancelled`` → aborted.
    """

    def __init__(
        self,
        code: Literal["busy", "cancelled", "changed", "summary", "commit", "persistence"],
        message: str | None = None,
    ) -> None:
        super().__init__(message or f"manual compaction failed: {code}")
        self.code = code


@dataclass(frozen=True, slots=True)
class SummarizationInput:
    """KV-cache-aligned input for the compaction LLM call (Phase 2).

    Replays the conversation's real system prompt, tools and the contiguous
    ``to_compact`` region verbatim, appending only the compaction instruction as
    the final user message — so the provider's cacheable request prefix is
    identical to the main loop's up to the compaction point.
    """

    system_prompt: str | None          # None → omit (generic fallback)
    tools: Sequence[Tool]              # real schemas, for prefix alignment
    messages: Sequence[Message]        # the contiguous to_compact region (original messages)
    instruction: Message               # final user message with the compaction prompt


@dataclass(frozen=True, slots=True)
class SurfaceFingerprint:
    """Cheap snapshot of the conversation surface for the stability check."""

    history_len: int
    token_count: int
    last_message_text: str | None     # cheap content fingerprint of last message


class CompactionResult(NamedTuple):
    messages: Sequence[Message]
    usage: TokenUsage | None
    compaction_id: str = ""
    """uuid4 hex identifying the compaction transaction (empty for no-op)."""
    shadowed_tokens: int = 0
    """Estimated tokens of the region replaced by the summary."""

    @property
    def estimated_token_count(self) -> int:
        """Estimate the token count of the compacted messages.

        When LLM usage is available, ``usage.output`` gives the exact token count
        of the generated summary (the first message).  Preserved messages (all
        subsequent messages) are estimated from their text length.

        When usage is not available (no compaction LLM call was made), all
        messages are estimated from text length.

        The estimate is intentionally conservative — it will be replaced by the
        real value on the next LLM call.
        """
        return self.estimated_token_count_for_model()

    def estimated_token_count_for_model(self, model: str | None = None) -> int:
        """Model-aware token count estimate.

        Args:
            model: Optional model name for tiktoken-based counting.
        """
        if self.usage is not None and len(self.messages) > 0:
            summary_tokens = self.usage.output
            preserved_tokens = count_message_tokens(self.messages[1:], model=model)
            return summary_tokens + preserved_tokens

        return count_message_tokens(self.messages, model=model)


def estimate_text_tokens(messages: Sequence[Message], model: str | None = None) -> int:
    """Estimate tokens from message text content.

    Backwards-compatible wrapper around :func:`count_message_tokens`.
    """
    return count_message_tokens(messages, model=model)


def _surface_fingerprint(messages: Sequence[Message]) -> SurfaceFingerprint:
    """Snapshot the conversation surface for the post-call stability check."""
    last = messages[-1] if messages else None
    return SurfaceFingerprint(
        history_len=len(messages),
        token_count=count_message_tokens(messages),
        last_message_text=last.extract_text(" ") if last is not None else None,
    )


def _legacy_reconstruct_to_compact(
    messages: Sequence[Message], to_preserve: Sequence[Message]
) -> list[Message]:
    """Reconstruct the compacted region for the legacy flattened path.

    :class:`PrepareResult` does not expose ``to_compact`` (the existing snapshot
    tests pin the NamedTuple shape), so when ``summarization_input`` is absent
    the region is derived from the preserved tail:

    - Phase-6 first-message re-insertion: ``to_preserve = [first] + tail`` and
      ``to_compact = messages[1:len(messages) - len(to_preserve) + 1]``.
    - Otherwise (balanced-cut fallback, or no re-insertion): ``to_compact`` is
      the prefix of length ``len(messages) - len(to_preserve)``.

    Only used for ledger shadow accounting on the legacy path (the KV-aligned
    path gets the exact region from ``summarization_input.messages``).
    """
    if not messages:
        return []
    n_preserved = len(to_preserve)
    if (
        to_preserve
        and n_preserved < len(messages)
        and to_preserve[0] is messages[0]
    ):
        # Phase-6 first-message re-insertion: the first message is preserved
        # verbatim and removed from the compaction input.
        return list(messages[1 : len(messages) - n_preserved + 1])
    return list(messages[: len(messages) - n_preserved])


def _detect_cascade_depth(messages: Sequence[Message]) -> int:
    """Count how many messages are already compaction summaries."""
    depth = 0
    for msg in messages:
        for part in msg.content:
            if isinstance(part, TextPart) and "Previous context has been compacted" in part.text:
                depth += 1
                break
    return depth


SAFETY_MARGIN_TOKENS: int = 4096
"""Extra tokens reserved for unforeseen growth (tool metadata, formatting, etc.)."""


def should_auto_compact(
    token_count: int,
    max_context_size: int,
    *,
    trigger_ratio: float,
    reserved_context_size: int,
    max_tokens: int | None = None,
    tool_call_buffer_tokens: int = 0,
    safety_margin_tokens: int = SAFETY_MARGIN_TOKENS,
) -> bool:
    """Determine whether auto-compaction should be triggered.

    Returns True when either condition is met (whichever fires first):
    - Ratio-based: token_count >= max_context_size * trigger_ratio
    - Reserved-based: token_count + effective_reserved >= max_context_size

    ``effective_reserved`` follows the context budget formula:

        Reserved Token = max(Tool Call Buffer, Reserved Context, Max Output Token + Safety Margin)

    i.e. ``max(tool_call_buffer_tokens, reserved_context_size,
    (max_tokens or 0) + safety_margin_tokens)``. Only the *largest* single
    reservation counts — reservations are no longer summed, so a large per-tool
    output buffer no longer shrinks the usable input window to a small fraction
    of the context. The result is capped at ``max_context_size -
    reserved_context_size`` so at least ``reserved_context_size`` tokens always
    remain available for input; the configuration stays usable even when the
    configured output budget is pathologically large for the model. When
    ``max_tokens`` is ``None`` it is treated as 0.
    """
    output_size = (max_tokens or 0) + safety_margin_tokens
    reserved = max(tool_call_buffer_tokens, reserved_context_size, output_size)
    # Cap the reservation so the configuration is always usable: at least
    # ``reserved_context_size`` tokens must remain available for input.
    min_input_room = max(0, max_context_size - reserved_context_size)
    effective_reserved = min(reserved, min_input_room)
    return (
        token_count >= max_context_size * trigger_ratio
        or token_count + effective_reserved >= max_context_size
    )


def adaptive_preserve_depth(
    messages: Sequence[Message],
    *,
    min_preserved: int = 1,
    max_preserved: int = 10,
) -> int:
    """Heuristically determine how many recent turns to preserve verbatim.

    Signals examined (only the most recent turn is inspected for speed):
    - Contains ``error`` / ``exception`` / ``failed``           → +1
    - Tool call with >2 file edits                              → +1
    - Contains :class:`ThinkPart` (reasoning)                   → +1
    - Pure Q&A (no tools)                                       → baseline (no boost)

    The result is clamped to ``[min_preserved, max_preserved]``.
    """
    depth = min_preserved
    if not messages:
        return depth

    # Inspect only the most recent user/assistant turn for speed.
    last_turn: Message | None = None
    for msg in reversed(messages):
        if msg.role in {"user", "assistant"}:
            last_turn = msg
            break

    if last_turn is None:
        return depth

    text = ""
    has_think = False
    for part in last_turn.content:
        if isinstance(part, TextPart):
            text += part.text
        elif isinstance(part, ThinkPart):
            has_think = True

    lowered = text.lower()
    if any(k in lowered for k in ("error", "exception", "failed")):
        depth += 1
    if has_think:
        depth += 1
    # Heuristic for "tool call with >2 file edits" – look for multiple file paths
    # in tool results (common pattern: ``file:`` or ``.py``, ``.md``, etc.).
    file_refs = lowered.count("file:") + lowered.count(".py") + lowered.count(".md")
    if file_refs > 2:
        depth += 1

    return min(max(depth, min_preserved), max_preserved)


@runtime_checkable
class Compaction(Protocol):
    async def compact(
        self,
        messages: Sequence[Message],
        llm: LLM,
        *,
        custom_instruction: str = "",
        options: CompactionOptions | None = None,
    ) -> CompactionResult:
        """
        Compact a sequence of messages into a new sequence of messages.

        Args:
            messages (Sequence[Message]): The messages to compact.
            llm (LLM): The LLM to use for compaction.
            custom_instruction: Optional user instruction to guide compaction focus.

        Returns:
            CompactionResult: The compacted messages and token usage from the compaction LLM call.

        Raises:
            ChatProviderError: When the chat provider returns an error.
        """
        ...


if TYPE_CHECKING:

    def type_check(simple: SimpleCompaction):
        _: Compaction = simple


_DECISION_SECTION_GUIDANCE = (
    "\n\n**Required Summary Sections:**\n"
    "Your summary MUST include these two sections with exact headings:\n"
    "## Decisions & Conclusions\n"
    "- Decisions already made and their rationale; approaches already "
    "evaluated and rejected (with the rejection reason); assumptions "
    "currently treated as valid.\n"
    "## Verification Status\n"
    "- What has been verified to work (and how it was verified); what "
    "remains unverified."
)


class SimpleCompaction:
    def __init__(
        self,
        max_preserved_messages: int = 2,
        *,
        preserve_depth: int | Callable[[Sequence[Message]], int] | None = None,
        decision_section_enabled: bool = False,
        balanced_cuts: bool = True,
    ) -> None:
        self.max_preserved_messages = max_preserved_messages
        self.preserve_depth = preserve_depth
        self.decision_section_enabled = decision_section_enabled
        self.balanced_cuts = balanced_cuts

    def _build_prompt_text(
        self,
        to_compact: Sequence[Message],
        options: CompactionOptions,
        custom_instruction: str,
    ) -> tuple[str, int]:
        """Build the compaction instruction text shared by both transports.

        Returns ``(prompt_text, cascade_depth)``. The legacy flattened path
        appends ``prompt_text`` as the final ``TextPart`` of ``compact_message``;
        the KV-aligned path wraps the *same* text in the instruction user
        message — keeping both paths byte-identical is what makes the aligned
        transport a drop-in replacement.
        """
        cascade_depth = _detect_cascade_depth(to_compact)
        if options.avoid_cascade:
            prompt_text = "\n" + prompts.COMPACT
        elif cascade_depth >= 3:
            prompt_text = "\n" + prompts.COMPACT_CASCADE
        else:
            prompt_text = "\n" + prompts.COMPACT

        mode_guidance = _MODE_GUIDANCE.get(options.mode)
        if mode_guidance:
            prompt_text += "\n\n" + mode_guidance

        if self.decision_section_enabled:
            prompt_text += _DECISION_SECTION_GUIDANCE

        if custom_instruction:
            prompt_text += (
                "\n\n**User's Custom Compaction Instruction:**\n"
                "Prioritize this user focus over the default priorities and style guidance:\n"
                f"{custom_instruction}"
            )
        return prompt_text, cascade_depth

    def _resolve_preserve_depth(
        self,
        messages: Sequence[Message],
        *,
        preserve_depth_override: int | None = None,
    ) -> int:
        if preserve_depth_override is not None:
            return preserve_depth_override
        if self.preserve_depth is None:
            return self.max_preserved_messages
        if callable(self.preserve_depth):
            return self.preserve_depth(messages)
        return self.preserve_depth

    async def compact(
        self,
        messages: Sequence[Message],
        llm: LLM,
        *,
        custom_instruction: str = "",
        options: CompactionOptions | None = None,
        recorder: LLMRequestRecorder | None = None,
        todos_loader: Callable[[], Sequence[TodoItemState]] | None = None,
        todos_stack_loader: Callable[[], Sequence[str]] | None = None,
        aligned_system_prompt: str | None = None,
        aligned_tools: Sequence[Tool] | None = None,
        ledger: CompactionLedger | None = None,
        trigger: Literal["auto", "manual", "overflow"] = "auto",
    ) -> CompactionResult:
        """Compact *messages* into a summary + preserved tail.

        Phase 2: when ``aligned_system_prompt`` is provided (and ``prepare``
        built a :class:`SummarizationInput`), the compaction LLM call replays the
        real system prompt / tools / history prefix via ``kosong.generate`` so
        the provider KV cache stays aligned. Otherwise the legacy flattened path
        (``kosong.step`` + ``EmptyToolset``) is used unchanged.

        Phase 3: every LLM-backed compaction is a transaction — a ``compaction_id``
        is generated up front, a pre-call :class:`SurfaceFingerprint` snapshot is
        taken, and after the call the surface is re-checked (stability) and the
        summary is required to be smaller than the shadowed region (shrink).
        When a ``ledger`` is provided the transaction is persisted; ledger I/O
        failures never propagate (they degrade to warnings).
        """
        options = options if options is not None else CompactionOptions()
        compaction_id = uuid.uuid4().hex
        prepare_result = self.prepare(
            messages,
            custom_instruction=custom_instruction,
            options=options,
            aligned_system_prompt=aligned_system_prompt,
            aligned_tools=aligned_tools,
        )
        compact_message = prepare_result.compact_message
        to_preserve = prepare_result.to_preserve
        if compact_message is None:
            # No compaction LLM call — skip the whole transactional envelope.
            return CompactionResult(messages=to_preserve, usage=None)

        if prepare_result.summarization_input is not None:
            to_compact = list(prepare_result.summarization_input.messages)
        else:
            to_compact = _legacy_reconstruct_to_compact(messages, to_preserve)
        shadowed_tokens = count_message_tokens(to_compact)

        # Call kosong.generate/step to get the compacted context
        # TODO: set max completion tokens
        if prepare_result.cascade_depth >= 3 and not options.avoid_cascade:
            logger.debug(
                "Compacting context with cascade prompt (depth={depth})...",
                depth=prepare_result.cascade_depth,
            )
        else:
            logger.debug("Compacting context...")

        # Phase 3: persist the transaction start before the LLM call. The real
        # summary_tokens / shrank are unknown here and finalized by record_end.
        ledger_started = False
        if ledger is not None:
            try:
                ledger.record_start(
                    CompactionRecord(
                        compaction_id=compaction_id,
                        trigger=trigger,
                        started_at=time.time(),
                        shadowed_range=(0, len(to_compact)),
                        shadowed_tokens=shadowed_tokens,
                        summary_tokens=0,
                        preserved_tokens=count_message_tokens(to_preserve),
                        shrank=False,
                    )
                )
                ledger_started = True
            except Exception as exc:  # noqa: BLE001 — ledger must never mask compaction
                logger.warning(
                    "Failed to record compaction start for {cid}: {err}",
                    cid=compaction_id,
                    err=exc,
                )

        surface_before = _surface_fingerprint(messages)
        try:
            if (
                aligned_system_prompt is not None
                and prepare_result.summarization_input is not None
            ):
                # KV-cache-aligned path: replay system + tools + region verbatim
                # and append only the instruction (generate, not step — the
                # instruction forbids tool calls and nothing is dispatched).
                summarization_input = prepare_result.summarization_input
                history: list[Message] = [
                    *summarization_input.messages,
                    summarization_input.instruction,
                ]
                if recorder is not None:
                    recorder.record(
                        llm.chat_provider,
                        aligned_system_prompt,
                        list(aligned_tools or []),
                        history,
                        kind="compaction",
                        dropped_count=len(messages) - len(to_preserve),
                    )
                result = await kosong.generate(
                    chat_provider=llm.chat_provider,
                    system_prompt=aligned_system_prompt,
                    tools=list(aligned_tools or []),
                    history=history,
                )
            else:
                # Legacy flattened path (kept for tests/back-compat).
                system_prompt = (
                    "You are a helpful assistant that compacts conversation context."
                )
                toolset = EmptyToolset()
                if recorder is not None:
                    recorder.record(
                        llm.chat_provider,
                        system_prompt,
                        toolset.tools,
                        [compact_message],
                        kind="compaction",
                        dropped_count=len(messages) - len(to_preserve),
                    )
                result = await kosong.step(
                    chat_provider=llm.chat_provider,
                    system_prompt=system_prompt,
                    toolset=toolset,
                    history=[compact_message],
                )
            if result.usage:
                logger.debug(
                    "Compaction used {input} input tokens and {output} output tokens",
                    input=result.usage.input,
                    output=result.usage.output,
                )

            # Phase 3 stability check (§5.2 item 3): the caller must not have
            # mutated the conversation while the summary was being generated.
            # history_len and last_message_text are the authoritative signals;
            # token_count is compared too but never raises on its own (it is
            # noisy under concurrent edits) — logged for diagnostics only.
            surface_after = _surface_fingerprint(messages)
            if (
                surface_after.history_len != surface_before.history_len
                or surface_after.last_message_text != surface_before.last_message_text
            ):
                raise SurfaceChangedError("conversation changed during compaction")
            if surface_after.token_count != surface_before.token_count:
                logger.debug(
                    "Compaction surface token count drifted ({before} -> {after}); "
                    "history length and tail unchanged",
                    before=surface_before.token_count,
                    after=surface_after.token_count,
                )

            # Phase 3 shrink check (§5.2 item 4): never apply a summary that is
            # not smaller than the region it replaces.
            summary_tokens = (
                result.usage.output
                if result.usage is not None
                else count_message_tokens([result.message])
            )
            if summary_tokens >= shadowed_tokens:
                raise CompactionShrinkError(
                    f"compaction summary ({summary_tokens} tokens) is not smaller than "
                    f"the compacted content ({shadowed_tokens} tokens)"
                )

            content: list[ContentPart] = [
                system("Previous context has been compacted. Here is the compaction output:")
            ]
            compacted_msg = result.message

            # drop thinking parts if any
            content.extend(
                part for part in compacted_msg.content if not isinstance(part, ThinkPart)
            )

            # Hermes-style re-injection: deterministically append the active
            # (unfinished) todo list so the plan survives context compression.
            # Failure-isolated — a broken loader must never break compaction.
            if todos_loader is not None:
                try:
                    stack = (
                        todos_stack_loader() if todos_stack_loader is not None else None
                    )
                    injection = format_todo_injection(
                        todos_loader(),
                        max_items=options.todos_max_items or 20,
                        stack=stack,
                    )
                except Exception:
                    injection = None
                if injection:
                    content.append(TextPart(text="\n\n" + injection))
            compacted_messages: list[Message] = [Message(role="user", content=content)]
            compacted_messages.extend(to_preserve)

            if ledger is not None:
                try:
                    ledger.record_end(
                        compaction_id,
                        summary_tokens=summary_tokens,
                        shrank=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to finalize compaction {cid} in ledger: {err}",
                        cid=compaction_id,
                        err=exc,
                    )
            return CompactionResult(
                messages=compacted_messages,
                usage=result.usage,
                compaction_id=compaction_id,
                shadowed_tokens=shadowed_tokens,
            )
        except Exception as exc:
            if ledger is not None and ledger_started:
                try:
                    ledger.record_end(compaction_id, error=str(exc))
                except Exception as ledger_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to record compaction failure {cid} in ledger: {err}",
                        cid=compaction_id,
                        err=ledger_exc,
                    )
            raise

    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]
        cascade_depth: int = 0
        summarization_input: SummarizationInput | None = None

    def prepare(
        self,
        messages: Sequence[Message],
        *,
        custom_instruction: str = "",
        options: CompactionOptions | None = None,
        aligned_system_prompt: str | None = None,
        aligned_tools: Sequence[Tool] | None = None,
    ) -> PrepareResult:
        """Prepare a compaction: split the history into a compacted region and a
        preserved tail, and build the legacy flattened ``compact_message``.

        When ``aligned_system_prompt`` is provided and there is a non-empty
        ``to_compact`` region, also build a :class:`SummarizationInput` carrying
        the original region messages plus an instruction user message whose text
        is byte-identical to the legacy ``compact_message`` tail — the KV-aligned
        transport replays this prefix verbatim.
        """
        options = options if options is not None else CompactionOptions()
        preserve_depth = self._resolve_preserve_depth(
            messages, preserve_depth_override=options.preserve_depth_override
        )
        if not messages or preserve_depth <= 0:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        history = list(messages)
        preserve_start_index = len(history)
        n_preserved = 0
        for index in range(len(history) - 1, -1, -1):
            if history[index].role in {"user", "assistant"}:
                n_preserved += 1
                if n_preserved == preserve_depth:
                    preserve_start_index = index
                    break

        if n_preserved < preserve_depth:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        # Balanced tool-pairing cuts: snap the boundary left to the nearest
        # balanced cut so the preserved tail never starts mid call/result pair.
        if self.balanced_cuts:
            preserve_start_index = nearest_balanced_cut_before(
                history, preserve_start_index
            )

        to_compact = history[:preserve_start_index]
        to_preserve = list(history[preserve_start_index:])

        # Phase 6: Sliding-Window + First-Turn Preservation
        # Always keep the very first message (primacy bias) if it's not already preserved.
        if history and history[0] not in to_preserve:
            to_preserve.insert(0, history[0])
            # Ensure the first message is not part of the compaction input
            if history[0] in to_compact:
                to_compact = [m for m in to_compact if m is not history[0]]

        if self.balanced_cuts and len(to_compact) not in balanced_cut_indices(history):
            # The Phase-6 first-message re-insertion shifted the boundary off a
            # balanced cut (e.g. the last compacted message was a tool result
            # answering a call that stays in to_compact). Keep the first message
            # in to_compact instead of forcing a balance violation: prefer the
            # largest balanced cut below the preserve point; if none exists
            # (pathological), fall back to preserve_start_index = 1.
            cuts = balanced_cut_indices(history)
            smaller = [c for c in cuts if c < preserve_start_index]
            if smaller:
                preserve_start_index = max(smaller)
            else:
                logger.warning(
                    "No balanced tool-pairing cut below preserve point {index}; "
                    "falling back to preserve_start_index=1",
                    index=preserve_start_index,
                )
                preserve_start_index = 1
            to_compact = list(history[:preserve_start_index])
            to_preserve = list(history[preserve_start_index:])

        if not to_compact:
            # Let's hope this won't exceed the context size limit
            return self.PrepareResult(compact_message=None, to_preserve=to_preserve)

        # Create input message for compaction
        compact_message = Message(role="user", content=[])
        for i, msg in enumerate(to_compact):
            compact_message.content.append(
                TextPart(text=f"## Message {i + 1}\nRole: {msg.role}\nContent:\n")
            )
            compact_message.content.extend(
                part for part in msg.content if isinstance(part, TextPart)
            )
        prompt_text, cascade_depth = self._build_prompt_text(
            to_compact, options, custom_instruction
        )
        compact_message.content.append(TextPart(text=prompt_text))

        # Phase 2: KV-cache-aligned input. The instruction message reuses the
        # exact ``prompt_text`` appended to ``compact_message`` so both transports
        # produce identical instruction text.
        summarization_input: SummarizationInput | None = None
        if aligned_system_prompt is not None:
            summarization_input = SummarizationInput(
                system_prompt=aligned_system_prompt,
                tools=tuple(aligned_tools or ()),
                messages=tuple(to_compact),
                instruction=Message(role="user", content=[TextPart(text=prompt_text)]),
            )
        return self.PrepareResult(
            compact_message=compact_message,
            to_preserve=to_preserve,
            cascade_depth=cascade_depth,
            summarization_input=summarization_input,
        )
