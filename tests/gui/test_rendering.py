from __future__ import annotations

from types import SimpleNamespace

import pytest

from kimi_agent_sdk import (
    ApprovalRequest,
    BriefDisplayBlock,
    DiffDisplayBlock,
    StatusUpdate,
    SubagentEvent,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    TodoDisplayItem,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
)
from kimix_gui.rendering import (
    McpCounts,
    McpServer,
    StatusValues,
    TokenCounts,
    WireNormalizer,
    display_blocks,
    status_values,
)
from kimix_gui.transcript_data import (
    ActivityEntry,
    AppendText,
    DiffBlock,
    FieldListBlock,
    RawBlock,
    ReplaceEntry,
    StartEntry,
    TextEntry,
    TodoBlock,
    TranscriptReducer,
    block_text,
    entry_body_text,
    resolve_text,
)


def _entry(normalized: object) -> object:
    mutation = normalized.mutations[-1]  # type: ignore[attr-defined]
    if isinstance(mutation, StartEntry | ReplaceEntry):
        return mutation.entry
    raise TypeError(type(mutation).__name__)


def test_text_and_thinking_are_typed_stream_mutations() -> None:
    normalizer = WireNormalizer()
    text = normalizer.normalize(TextPart(text="hello"))
    thinking = normalizer.normalize(ThinkPart(think="considering"))

    text_entry = _entry(text)
    thinking_entry = _entry(thinking)
    assert isinstance(text_entry, TextEntry)
    assert (text_entry.kind, entry_body_text(text_entry), text_entry.complete) == (
        "assistant",
        "hello",
        False,
    )
    assert isinstance(thinking_entry, TextEntry)
    assert thinking_entry.kind == "thinking"
    assert entry_body_text(thinking_entry) == "considering"
    assert any(
        mutation.key == text_entry.key
        for mutation in thinking.mutations
        if hasattr(mutation, "key")
    )


def test_multiple_text_parts_reduce_to_one_entry() -> None:
    normalizer = WireNormalizer()
    reducer = TranscriptReducer()
    first = normalizer.normalize(TextPart(text="hel"))
    second = normalizer.normalize(TextPart(text="lo"))

    for mutation in (*first.mutations, *second.mutations):
        reducer.apply(mutation)

    assert len(reducer.entries) == 1
    assert entry_body_text(reducer.entries[0]) == "hello"
    assert isinstance(second.mutations[-1], AppendText)


def test_tool_arguments_are_decoded_into_fields_once() -> None:
    normalized = WireNormalizer().normalize(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="read", arguments='{"path":"a.py"}'),
        )
    )

    entry = _entry(normalized)
    assert isinstance(entry, ActivityEntry)
    assert entry.activity.identity.wire_name == "read"
    assert entry.activity.call is not None
    assert entry.activity.call.parse_state == "complete"
    assert entry.activity.call.raw_arguments == '{"path":"a.py"}'
    assert [(field.name, field.role, field.hint) for field in entry.activity.call.fields] == [
        ("path", "primary", "path")
    ]
    assert resolve_text(entry.activity.call.summary_parts[0]) == "a.py"


def test_partial_and_invalid_arguments_remain_lossless_raw_blocks() -> None:
    partial = _entry(
        WireNormalizer().normalize(
            ToolCall(
                id="partial",
                function=ToolCall.FunctionBody(name="read", arguments='{"path":'),
            )
        )
    )
    invalid = _entry(
        WireNormalizer().normalize(
            ToolCall(
                id="invalid",
                function=ToolCall.FunctionBody(name="read", arguments="{]"),
            )
        )
    )

    assert isinstance(partial, ActivityEntry) and partial.activity.call is not None
    assert partial.activity.call.parse_state == "partial"
    assert isinstance(partial.activity.call.details[0], RawBlock)
    assert partial.activity.call.details[0].payload == '{"path":'
    assert isinstance(invalid, ActivityEntry) and invalid.activity.call is not None
    assert invalid.activity.call.parse_state == "invalid"
    assert invalid.activity.call.raw_arguments == "{]"


def test_a_complete_tool_part_replaces_the_partial_snapshot() -> None:
    normalizer = WireNormalizer()
    first = normalizer.normalize(
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="read", arguments=""))
    )
    partial = normalizer.normalize(ToolCallPart(arguments_part='{"path":'))
    complete = normalizer.normalize(ToolCallPart(arguments_part='"a.py"}'))

    keys = [_entry(item).key for item in (first, partial, complete)]
    assert keys == ["root:activity:call-1"] * 3
    entry = _entry(complete)
    assert isinstance(entry, ActivityEntry) and entry.activity.call is not None
    assert entry.activity.call.parse_state == "complete"
    assert resolve_text(entry.activity.call.fields[0].value) == "a.py"


def test_parallel_streamed_calls_route_parts_to_each_incomplete_activity() -> None:
    normalizer = WireNormalizer()
    normalizer.normalize(
        ToolCall(id="call-a", function=ToolCall.FunctionBody(name="read", arguments=""))
    )
    normalizer.normalize(
        ToolCall(id="call-b", function=ToolCall.FunctionBody(name="read", arguments=""))
    )

    first = _entry(normalizer.normalize(ToolCallPart(arguments_part='{"path":"a.py"}')))
    second = _entry(normalizer.normalize(ToolCallPart(arguments_part='{"path":"b.py"}')))

    assert isinstance(first, ActivityEntry) and first.activity.call is not None
    assert isinstance(second, ActivityEntry) and second.activity.call is not None
    assert first.activity.call_id == "call-a"
    assert second.activity.call_id == "call-b"
    assert resolve_text(first.activity.call.fields[0].value) == "a.py"
    assert resolve_text(second.activity.call.fields[0].value) == "b.py"


def test_tool_result_updates_the_same_full_activity_snapshot() -> None:
    normalizer = WireNormalizer()
    normalizer.normalize(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="read", arguments='{"path":"a.py"}'),
            extras={"provider": "test"},
        )
    )
    result = normalizer.normalize(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False,
                output="file contents",
                message="success",
                display=[BriefDisplayBlock(text="12 lines")],
                extras={"bytes": 20},
            ),
        )
    )

    entry = _entry(result)
    assert isinstance(entry, ActivityEntry)
    assert entry.key == "root:activity:call-1"
    assert entry.complete is True
    assert entry.activity.call is not None
    assert entry.activity.result is not None
    assert entry.activity.result.status == "ok"
    assert "file contents" in entry_body_text(entry)
    assert any(isinstance(block, RawBlock) for block in entry.activity.result.blocks)


def test_result_first_then_call_upserts_one_activity() -> None:
    normalizer = WireNormalizer()
    reducer = TranscriptReducer()
    result = normalizer.normalize(
        ToolResult(
            tool_call_id="late",
            return_value=ToolReturnValue(is_error=True, output="", message="missing", display=[]),
        )
    )
    call = normalizer.normalize(
        ToolCall(
            id="late",
            function=ToolCall.FunctionBody(name="read", arguments='{"path":"a.py"}'),
        )
    )
    for mutation in (*result.mutations, *call.mutations):
        reducer.apply(mutation)

    assert len(reducer.entries) == 1
    entry = reducer.entries[0]
    assert isinstance(entry, ActivityEntry)
    assert entry.activity.call is not None and entry.activity.result is not None
    assert entry.activity.result.status == "error"


def test_repeated_result_update_keeps_the_decoded_call_snapshot() -> None:
    normalizer = WireNormalizer()
    normalizer.normalize(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="read", arguments='{"path":"a.py"}'),
        )
    )
    for output in ("first", "second"):
        latest = _entry(
            normalizer.normalize(
                ToolResult(
                    tool_call_id="call-1",
                    return_value=ToolReturnValue(
                        is_error=False, output=output, message="success", display=[]
                    ),
                )
            )
        )

    assert isinstance(latest, ActivityEntry)
    assert latest.activity.call is not None and latest.activity.result is not None
    assert resolve_text(latest.activity.call.fields[0].value) == "a.py"
    assert "second" in entry_body_text(latest)


def test_absent_approval_metadata_does_not_turn_into_literal_none_fields() -> None:
    entry = _entry(
        WireNormalizer().normalize(
            ApprovalRequest(
                id="approval-1",
                tool_call_id="call-1",
                sender="write",
                action="write file",
                description="Write a.py",
            )
        )
    )

    assert not any(
        resolve_text(field.value) == "None"
        for block in entry.blocks  # type: ignore[union-attr]
        if isinstance(block, FieldListBlock)
        for field in block.fields
    )


def test_native_display_blocks_preserve_diff_and_todo_structure() -> None:
    blocks = display_blocks(
        [
            BriefDisplayBlock(text="Updated files"),
            DiffDisplayBlock(path="a.py", old_text="old", new_text="new", old_start=4, new_start=5),
            TodoDisplayBlock(
                items=[
                    TodoDisplayItem(title="Implement", status="in_progress", notes="now"),
                    TodoDisplayItem(title="Verify", status="done", depth=1),
                ]
            ),
        ]
    )

    assert isinstance(blocks[1], DiffBlock)
    assert isinstance(blocks[2], TodoBlock)
    rendered = "\n".join(block_text(block) for block in blocks)
    for detail in ("Updated files", "Diff: a.py", "@@ -4 +5 @@", "[>] Implement — now"):
        assert detail in rendered


def test_status_is_an_independent_structured_update() -> None:
    status = StatusUpdate(context_tokens=1_000, max_context_tokens=10_000, context_usage=0.1)
    normalized = WireNormalizer().normalize(status)

    assert normalized.mutations == ()
    assert normalized.status == StatusValues(
        context_tokens=1_000,
        max_context_tokens=10_000,
        usage_percent=pytest.approx(10.0),
    )


def test_status_includes_tokens_and_mcp_details() -> None:
    status = SimpleNamespace(
        context_tokens=2_000,
        max_context_tokens=20_000,
        context_usage=0.1,
        token_usage=TokenUsage(
            input_other=100,
            input_cache_read=800,
            input_cache_creation=50,
            output=75,
        ),
        message_id="msg-1",
        mcp_status=SimpleNamespace(
            loading=False,
            connected=1,
            total=2,
            tools=8,
            servers=(SimpleNamespace(name="github", status="connected"),),
        ),
    )

    values = status_values(status)
    assert values.tokens == TokenCounts(
        total_input=950, new_input=100, cache_read=800, cache_write=50, output=75
    )
    assert values.mcp == McpCounts(
        connected=1,
        total=2,
        tools=8,
        loading=False,
        servers=(McpServer(name="github", state="connected"),),
    )
    assert not hasattr(values, "message_id")


def test_subagent_entries_carry_source_metadata_instead_of_a_text_prefix() -> None:
    normalizer = WireNormalizer()
    main = _entry(normalizer.normalize(TextPart(text="main")))
    child = _entry(
        normalizer.normalize(
            SubagentEvent(
                agent_id="child-1",
                subagent_type="explore",
                event=TextPart(text="child"),
            )
        )
    )

    assert isinstance(main, TextEntry) and main.source.kind == "root"
    assert isinstance(child, TextEntry)
    assert child.source.kind == "subagent"
    assert child.source.identifier == "child-1"
    assert entry_body_text(child) == "child"


def test_display_fields_are_not_reparsed_when_values_contain_colons() -> None:
    normalizer = WireNormalizer()
    entry = _entry(
        normalizer.normalize(
            ToolCall(
                id="colon",
                function=ToolCall.FunctionBody(
                    name="read", arguments='{"path":"C:/src/key: value.py"}'
                ),
            )
        )
    )

    assert isinstance(entry, ActivityEntry) and entry.activity.call is not None
    assert isinstance(FieldListBlock(entry.activity.call.fields), FieldListBlock)
    assert resolve_text(entry.activity.call.fields[0].value) == "C:/src/key: value.py"
