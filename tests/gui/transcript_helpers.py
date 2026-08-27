"""Small semantic transcript fixtures shared by Qt tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import count

from kimix_gui.qt.transcript import Transcript
from kimix_gui.tool_display import tool_family
from kimix_gui.transcript_data import (
    ActivityEntry,
    ActivityField,
    AppendText,
    HistoryEntry,
    NoticeEntry,
    RawBlock,
    ReplaceEntry,
    TextBlock,
    TextEntry,
    ToolActivity,
    ToolCallContent,
    ToolIdentity,
    ToolResultContent,
    TranscriptEntry,
    entry_body_text,
    entry_kind,
    literal,
)
from kimix_gui.transcript_layout import activity_status

_KEYS = count()


def next_key(prefix: str = "test") -> str:
    return f"{prefix}-{next(_KEYS)}"


def text_entry(
    kind: str,
    text: str,
    *,
    key: str | None = None,
    complete: bool = True,
    markdown: bool = False,
) -> TranscriptEntry:
    """Build a semantic text or notice entry without a display-text parser."""

    entry_key = key or next_key(kind)
    block = TextBlock(literal(text), format="markdown" if markdown else "plain")
    if kind in {"user", "assistant", "thinking"}:
        return TextEntry(
            key=entry_key,
            kind=kind,  # type: ignore[arg-type]
            blocks=(block,),
            complete=complete,
        )
    notice_kind = kind if kind in {"system", "error", "approval"} else "system"
    return NoticeEntry(
        key=entry_key,
        kind=notice_kind,  # type: ignore[arg-type]
        blocks=(block,),
        complete=complete,
    )


def activity_entry(
    wire_name: str,
    *,
    call_id: str | None = None,
    summary: str = "",
    call_text: str = "",
    result_summary: str = "",
    result_text: str = "",
    status: str | None = None,
    fields: tuple[ActivityField, ...] = (),
    key: str | None = None,
) -> ActivityEntry:
    """Build a complete activity snapshot; callers state every semantic component."""

    resolved_call_id = call_id or next_key("call")
    call = ToolCallContent(
        summary_parts=(literal(summary),) if summary else (),
        fields=fields,
        details=(RawBlock(call_text),) if call_text else (),
    )
    result = None
    if status is not None:
        result = ToolResultContent(
            status=status,  # type: ignore[arg-type]
            summary_parts=(literal(result_summary),) if result_summary else (),
            blocks=(TextBlock(literal(result_text)),) if result_text else (),
        )
    return ActivityEntry(
        key=key or f"activity:{resolved_call_id}",
        activity=ToolActivity(
            call_id=resolved_call_id,
            identity=ToolIdentity(wire_name=wire_name, family=tool_family(wire_name)),
            call=call,
            result=result,
        ),
        complete=status is not None,
    )


def append_entry(transcript: Transcript, entry: TranscriptEntry) -> None:
    transcript.apply_mutation(ReplaceEntry(entry))


def append_text(
    transcript: Transcript,
    kind: str,
    text: str,
    *,
    key: str | None = None,
    markdown: bool = False,
) -> str:
    entry = text_entry(kind, text, key=key, markdown=markdown)
    append_entry(transcript, entry)
    return entry.key


def append_texts(transcript: Transcript, items: Iterable[tuple[str, str]]) -> None:
    for kind, text in items:
        append_text(transcript, kind, text)


def append_fragment(
    transcript: Transcript,
    kind: str,
    fragment: str,
    *,
    key: str,
    markdown: bool = False,
) -> None:
    transcript.apply_mutation(
        AppendText(
            key=key,
            kind=kind,  # type: ignore[arg-type]
            source=text_entry(kind, "").source,
            block=0,
            fragment=fragment,
            format="markdown" if markdown else "plain",
        )
    )


def history_entries(
    items: Sequence[tuple[str, str] | tuple[str, str, int]],
) -> tuple[HistoryEntry, ...]:
    ordinals: dict[int, int] = {}
    result: list[HistoryEntry] = []
    for fallback_turn, item in enumerate(items):
        kind, text, *turn_value = item
        turn = turn_value[0] if turn_value else fallback_turn
        ordinal = ordinals.get(turn, 0)
        ordinals[turn] = ordinal + 1
        result.append(
            HistoryEntry(
                entry=text_entry(kind, text, key=f"history:{turn}:{ordinal}"),
                turn=turn,
                ordinal=ordinal,
            )
        )
    return tuple(result)


def row_text(row) -> str:
    return entry_body_text(row.entry)


def row_kind(row) -> str:
    return entry_kind(row.entry)


def row_status(row) -> str:
    return activity_status(row.entry)


def replace_activity(transcript: Transcript, entry: ActivityEntry) -> None:
    transcript.apply_mutation(ReplaceEntry(entry))
