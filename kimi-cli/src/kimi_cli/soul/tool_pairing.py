from __future__ import annotations

from collections.abc import Sequence

from kosong.message import Message, ToolCall, ToolCallPart


def is_tool_call_part(part: object) -> bool:
    """True for ToolCall or ToolCallPart instances (both are call markers)."""
    return isinstance(part, (ToolCall, ToolCallPart))


def message_tool_call_delta(msg: Message) -> int:
    """+N for an assistant message with N tool-call parts; -1 for a tool result; 0 otherwise.

    Assistant messages persist completed :class:`ToolCall` objects in
    ``msg.tool_calls`` (see ``kosong._generate._message_append``), but other
    flows may carry streamed :class:`ToolCallPart` instances inside
    ``msg.content`` — both are counted.
    """
    if msg.role == "assistant":
        n = len(msg.tool_calls) if msg.tool_calls else 0
        n += sum(1 for part in msg.content if isinstance(part, ToolCallPart))
        return n
    if msg.role == "tool":
        return -1
    return 0


def balanced_cut_indices(messages: Sequence[Message]) -> set[int]:
    """Return the set of *cut indices* (0..len) where no unanswered tool call
    crosses the cut.

    ``cut i`` is the boundary between ``messages[i - 1]`` and ``messages[i]``.
    A fold over the history maintains ``in_progress`` (the number of assistant
    tool calls not yet answered by a ``tool`` result); a cut is balanced iff
    ``in_progress == 0`` after processing ``messages[:i]``. Cut ``0`` (start)
    and cut ``len`` (after the last message) are always balanced.

    Raises:
        ValueError: On unbalanced history — a ``tool`` result with no matching
            call (``in_progress`` goes negative). A trailing assistant tool
            call with no following result is tolerated leniently (only a
            negative count is an error), mirroring DSH.
    """
    cuts: set[int] = {0}
    in_progress = 0
    for i, msg in enumerate(messages, start=1):
        in_progress += message_tool_call_delta(msg)
        if in_progress < 0:
            raise ValueError(
                "unbalanced tool history: tool result with no matching call at index "
                f"{i - 1}"
            )
        if in_progress == 0:
            cuts.add(i)
    # The cut after the last message is always balanced (nothing crosses it).
    cuts.add(len(messages))
    return cuts


def nearest_balanced_cut_before(messages: Sequence[Message], index: int) -> int:
    """Largest balanced cut <= index (never splits a call/result pair).

    ``index == len`` means "after the last message". Out-of-range indices are
    clamped: ``index < 0`` → ``0``, ``index > len`` → ``len``.
    """
    if index < 0:
        return 0
    if index > len(messages):
        return len(messages)
    cuts = balanced_cut_indices(messages)
    return max(c for c in cuts if c <= index)
