from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kimix_gui.transcript_data import (
    ROOT_SOURCE,
    ActivityEntry,
    AppendText,
    ClearTranscript,
    EntrySource,
    FinishEntry,
    RawBlock,
    ReplaceEntry,
    TextBlock,
    TextEntry,
    ToolActivity,
    ToolCallContent,
    ToolIdentity,
    ToolResultContent,
    TranscriptReducer,
    entry_body_text,
    entry_copy_text,
    entry_cost,
    literal,
    localized,
    resolve_text,
)


def test_append_for_a_trimmed_target_recreates_a_deterministic_row() -> None:
    reducer = TranscriptReducer()
    effect = reducer.apply(
        AppendText(
            key="assistant:7",
            kind="assistant",
            source=ROOT_SOURCE,
            block=2,
            fragment="survives trimming",
            format="markdown",
        )
    )

    assert effect.action == "insert"
    assert len(reducer.entries) == 1
    entry = reducer.entries[0]
    assert isinstance(entry, TextEntry)
    assert (entry.key, entry.kind, entry.complete) == ("assistant:7", "assistant", False)
    assert len(entry.blocks) == 3
    assert entry_body_text(entry) == "survives trimming"
    assert all(
        isinstance(block, TextBlock) and resolve_text(block.text) == ""
        for block in entry.blocks[:2]
    )
    target = entry.blocks[2]
    assert isinstance(target, TextBlock) and target.format == "markdown"


def test_replace_upserts_finish_is_typed_and_clear_is_explicit() -> None:
    reducer = TranscriptReducer()
    entry = TextEntry("answer", "assistant", (TextBlock(literal("done")),), complete=False)

    assert reducer.apply(ReplaceEntry(entry)).action == "insert"
    assert reducer.apply(FinishEntry("answer")).action == "finish"
    assert reducer.entries[0].complete is True
    assert reducer.apply(ClearTranscript()).action == "clear"
    assert reducer.entries == []


def test_entry_cost_counts_nested_raw_and_original_payloads() -> None:
    raw_arguments = '{"code":"' + ("x" * 200) + '"}'
    raw_result = "y" * 300
    entry = ActivityEntry(
        key="activity:1",
        activity=ToolActivity(
            call_id="call-1",
            identity=ToolIdentity("python", "python"),
            call=ToolCallContent(
                details=(RawBlock(raw_arguments, parse_state="partial"),),
                raw_arguments=raw_arguments,
                parse_state="partial",
            ),
            result=ToolResultContent(
                status="error",
                blocks=(RawBlock(raw_result, parse_state="invalid"),),
                raw_fallback=raw_result,
            ),
        ),
    )

    assert entry_cost(entry) >= len(raw_arguments) * 2 + len(raw_result) * 2


def test_cross_layer_entries_are_frozen_and_slotted() -> None:
    entry = TextEntry("user:1", "user", (TextBlock(literal("hello")),))

    with pytest.raises(FrozenInstanceError):
        entry.key = "changed"  # type: ignore[misc]
    assert not hasattr(entry, "__dict__")


def test_localized_text_resolves_named_arguments_only_at_the_formatter_boundary() -> None:
    text = localized("Session: {id}", id="abc")

    assert resolve_text(text) == "Session: abc"
    assert resolve_text(
        text, lambda msgid: "会话：{id}" if msgid.startswith("Session") else msgid
    ) == ("会话：abc")


def test_copy_formatter_keeps_structured_source_metadata() -> None:
    entry = TextEntry(
        "child:1",
        "assistant",
        (TextBlock(literal("answer")),),
        source=EntrySource(
            kind="subagent",
            identifier="child-1",
            label=localized("Subagent {kind} · {id}", kind="explore", id="child-1"),
        ),
    )

    assert entry_copy_text(entry) == "Subagent explore · child-1\n\nanswer"
