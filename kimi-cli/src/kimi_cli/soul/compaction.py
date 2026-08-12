from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import kosong
from kosong.chat_provider import TokenUsage
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.session_state import TodoItemState, format_todo_injection
from kimi_cli.soul.llm_request_recorder import LLMRequestRecorder
from kimi_cli.soul.message import system
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


class CompactionResult(NamedTuple):
    messages: Sequence[Message]
    usage: TokenUsage | None

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


def _detect_cascade_depth(messages: Sequence[Message]) -> int:
    """Count how many messages are already compaction summaries."""
    depth = 0
    for msg in messages:
        for part in msg.content:
            if isinstance(part, TextPart) and "Previous context has been compacted" in part.text:
                depth += 1
                break
    return depth


def should_auto_compact(
    token_count: int,
    max_context_size: int,
    *,
    trigger_ratio: float,
    reserved_context_size: int,
) -> bool:
    """Determine whether auto-compaction should be triggered.

    Returns True when either condition is met (whichever fires first):
    - Ratio-based: token_count >= max_context_size * trigger_ratio
    - Reserved-based: token_count + reserved_context_size >= max_context_size
    """
    return (
        token_count >= max_context_size * trigger_ratio
        or token_count + reserved_context_size >= max_context_size
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
    ) -> None:
        self.max_preserved_messages = max_preserved_messages
        self.preserve_depth = preserve_depth
        self.decision_section_enabled = decision_section_enabled

    def _resolve_preserve_depth(self, messages: Sequence[Message]) -> int:
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
    ) -> CompactionResult:
        options = options if options is not None else CompactionOptions()
        prepare_result = self.prepare(
            messages, custom_instruction=custom_instruction, options=options
        )
        compact_message = prepare_result.compact_message
        to_preserve = prepare_result.to_preserve
        if compact_message is None:
            return CompactionResult(messages=to_preserve, usage=None)

        # Call kosong.step to get the compacted context
        # TODO: set max completion tokens
        if prepare_result.cascade_depth >= 3 and not options.avoid_cascade:
            logger.debug(
                "Compacting context with cascade prompt (depth={depth})...",
                depth=prepare_result.cascade_depth,
            )
        else:
            logger.debug("Compacting context...")
        system_prompt = "You are a helpful assistant that compacts conversation context."
        toolset = EmptyToolset()
        if recorder is not None:
            recorder.record(
                llm.chat_provider,
                system_prompt,
                toolset.tools,
                [compact_message],
                kind="compaction",
                dropped_count=len(messages) - len(prepare_result.to_preserve),
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

        content: list[ContentPart] = [
            system("Previous context has been compacted. Here is the compaction output:")
        ]
        compacted_msg = result.message

        # drop thinking parts if any
        content.extend(part for part in compacted_msg.content if not isinstance(part, ThinkPart))

        # Hermes-style re-injection: deterministically append the active
        # (unfinished) todo list so the plan survives context compression.
        # Failure-isolated — a broken loader must never break compaction.
        if todos_loader is not None:
            try:
                stack = todos_stack_loader() if todos_stack_loader is not None else None
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
        return CompactionResult(messages=compacted_messages, usage=result.usage)

    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]
        cascade_depth: int = 0

    def prepare(
        self,
        messages: Sequence[Message],
        *,
        custom_instruction: str = "",
        options: CompactionOptions | None = None,
    ) -> PrepareResult:
        options = options if options is not None else CompactionOptions()
        preserve_depth = self._resolve_preserve_depth(messages)
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

        to_compact = history[:preserve_start_index]
        to_preserve = list(history[preserve_start_index:])

        # Phase 6: Sliding-Window + First-Turn Preservation
        # Always keep the very first message (primacy bias) if it's not already preserved.
        if history and history[0] not in to_preserve:
            to_preserve.insert(0, history[0])
            # Ensure the first message is not part of the compaction input
            if history[0] in to_compact:
                to_compact = [m for m in to_compact if m is not history[0]]

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
        compact_message.content.append(TextPart(text=prompt_text))
        return self.PrepareResult(compact_message=compact_message, to_preserve=to_preserve, cascade_depth=cascade_depth)
