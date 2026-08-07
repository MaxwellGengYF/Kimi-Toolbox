from __future__ import annotations

from collections.abc import Sequence

import regex as re
from kosong.message import Message
from kosong.tooling.error import ToolRuntimeError

from kimi_cli.llm import ModelCapability
from kimi_cli.wire.types import (
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    ToolResult,
    VideoURLPart,
)


def system(message: str) -> ContentPart:
    return TextPart(text=f"<system>{message}</system>")


def system_reminder(message: str) -> TextPart:
    return TextPart(text=f"<system-reminder>\n{message}\n</system-reminder>")


def is_system_reminder_message(message: Message) -> bool:
    """Check whether a message is an internal system-reminder user message."""
    if message.role != "user" or len(message.content) != 1:
        return False
    part = message.content[0]
    return isinstance(part, TextPart) and part.text.strip().startswith("<system-reminder>")


def strip_system_reminders(history: list[Message]) -> int:
    """Remove all standalone system-reminder user messages from *history* in-place.

    Returns the number of messages removed.
    """
    removed = 0
    i = 0
    while i < len(history):
        if is_system_reminder_message(history[i]):
            history.pop(i)
            removed += 1
        else:
            i += 1
    return removed


def tool_result_to_message(tool_result: ToolResult) -> Message:
    """Convert a tool result to a message."""
    if tool_result.return_value.is_error:
        assert tool_result.return_value.message, "Error return value should have a message"
        message = tool_result.return_value.message
        if isinstance(tool_result.return_value, ToolRuntimeError):
            message += "\nThis is an unexpected error and the tool is probably not working."
        content: list[ContentPart] = [system(f"ERROR: {message}")]
        if tool_result.return_value.output:
            content.extend(_output_to_content_parts(tool_result.return_value.output))
    else:
        content: list[ContentPart] = []
        if tool_result.return_value.message:
            content.append(system(tool_result.return_value.message))
        if tool_result.return_value.output:
            content.extend(_output_to_content_parts(tool_result.return_value.output))
        if not content:
            content.append(system("Tool output is empty."))
        elif not any(isinstance(part, TextPart) for part in content):
            # Ensure at least one TextPart exists so the LLM API won't reject
            # the message with "text content is empty" (see #1663).
            content.insert(0, system("Tool returned non-text content."))

    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_result.tool_call_id,
    )


# ── Layer 1 — coalesce identical <system> metadata ──────────────────────

_SYSTEM_RE = re.compile(r"^<system>(.*)</system>$", re.DOTALL)
_SYSTEM_OPEN = "<system>"
_SYSTEM_CLOSE = "</system>"


def _extract_system_text(part: TextPart) -> str | None:
    """Return the inner text of a ``<system>…</system>`` TextPart, or ``None``."""
    if not isinstance(part, TextPart):
        return None
    text = part.text.strip()
    m = _SYSTEM_RE.match(text)
    if m:
        return m.group(1).strip()
    return None


def coalesce_tool_metadata(history: list[Message]) -> int:
    """Merge adjacent identical ``<system>…</system>` metadata across tool messages.

    When consecutive tool-role messages start with the **same** system
    metadata line (e.g. ``<system>Results truncated to 20 lines.</system>``),
    keep only the first occurrence and remove the redundant copies from
    subsequent messages.  The first message's system block is annotated
    with ``[×N]`` so the model knows *N* results shared this note.

    This is Class C3 from plan.md §6 (Stage 6).  It is safe because the
    information is purely cosmetic metadata, not user-visible tool output.

    Args:
        history: The conversation messages (modified **in-place**).

    Returns:
        The number of system TextParts removed.
    """
    removed = 0
    i = 0
    while i < len(history) - 1:
        msg = history[i]
        next_msg = history[i + 1]
        if (
            msg.role == "tool"
            and next_msg.role == "tool"
            and msg.content
            and next_msg.content
        ):
            sys_text = _extract_system_text(msg.content[0])
            if sys_text is not None:
                # Look ahead — count how many consecutive tool messages
                # share this exact system metadata.
                run = 0
                j = i + 1
                while (
                    j < len(history)
                    and history[j].role == "tool"
                    and history[j].content
                    and _extract_system_text(history[j].content[0]) == sys_text
                ):
                    run += 1
                    j += 1
                if run > 0:
                    # Annotate first occurrence if more than 1 share it.
                    total = run + 1
                    if total > 1:
                        new_inner = f"[×{total}] {sys_text}"
                        msg.content[0] = TextPart(
                            text=f"{_SYSTEM_OPEN}{new_inner}{_SYSTEM_CLOSE}"
                        )
                    # Remove the system part from each subsequent message,
                    # but never leave a message with *empty* content (provider
                    # invariants require non-empty tool results).
                    for k in range(i + 1, j):
                        if len(history[k].content) > 1:
                            history[k].content = history[k].content[1:]
                            removed += 1
                    i = j  # skip past the coalesced run
                    continue
        i += 1
    return removed


def coalesce_content_parts(content: list[ContentPart]) -> list[ContentPart]:
    """Merge adjacent ``<system>…</system>`` TextParts within a single content list.

    Multiple system blocks that are adjacent (separated only by other system
    blocks) are merged into one, with their inner text concatenated.

    Returns a new list (does not modify *content*).
    """
    if not content:
        return content
    result: list[ContentPart] = []
    pending_system_texts: list[str] = []

    def flush() -> None:
        if pending_system_texts:
            merged = ". ".join(pending_system_texts)
            result.append(TextPart(text=f"{_SYSTEM_OPEN}{merged}{_SYSTEM_CLOSE}"))
            pending_system_texts.clear()

    for part in content:
        sys_text = _extract_system_text(part)
        if sys_text is not None:
            pending_system_texts.append(sys_text)
        else:
            flush()
            result.append(part)
    flush()
    return result


def _output_to_content_parts(
    output: str | ContentPart | Sequence[ContentPart],
) -> list[ContentPart]:
    content: list[ContentPart] = []
    match output:
        case str(text):
            if text:
                content.append(TextPart(text=text))
        case ContentPart():
            content.append(output)
        case _:
            content.extend(output)
    return content


def check_message(
    message: Message, model_capabilities: set[ModelCapability]
) -> set[ModelCapability]:
    """Check the message content, return the missing model capabilities."""
    capabilities_needed = set[ModelCapability]()
    for part in message.content:
        if isinstance(part, ImageURLPart):
            capabilities_needed.add("image_in")
        elif isinstance(part, VideoURLPart):
            capabilities_needed.add("video_in")
        elif isinstance(part, ThinkPart):
            capabilities_needed.add("thinking")
    return capabilities_needed - model_capabilities
