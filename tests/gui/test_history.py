from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kimi_agent_sdk import (
    BriefDisplayBlock,
    StepInterrupted,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
    TurnBegin,
    TurnEnd,
)
from kimix_gui.history import (
    HistoryAccumulator,
    Timeline,
    _read_wire_turns,
    _scan_wire_history_index,
    entries_from_wire_messages,
    load_session_history,
    load_wire_history_page,
    take_last_turns,
)
from kimix_gui.rendering import WireNormalizer
from kimix_gui.transcript_data import (
    ActivityEntry,
    HistoryEntry,
    TextBlock,
    TextEntry,
    TranscriptReducer,
    entry_body_text,
    entry_kind,
    literal,
)


class SteerInput:
    def __init__(self, user_input: str) -> None:
        self.user_input = user_input


def _texts(entries: object, kind: str | None = None) -> list[str]:
    values = entries.entries if hasattr(entries, "entries") else entries
    return [
        entry_body_text(item.entry)
        for item in values
        if kind is None or entry_kind(item.entry) == kind
    ]


def _turns(count: int) -> list[list[TextEntry]]:
    return [
        [
            TextEntry(f"user:{index}", "user", (TextBlock(literal(f"q{index}")),)),
            TextEntry(f"assistant:{index}", "assistant", (TextBlock(literal(f"a{index}")),)),
        ]
        for index in range(count)
    ]


def test_mutations_merge_streams_and_keep_user_turns() -> None:
    entries = entries_from_wire_messages(
        [
            TurnBegin(user_input="fix login"),
            ThinkPart(think="looking "),
            ThinkPart(think="at auth"),
            TextPart(text="Check "),
            TextPart(text="the redirect."),
            TurnEnd(),
            SteerInput("also cookies"),
        ]
    )

    assert [(entry_kind(item.entry), entry_body_text(item.entry)) for item in entries] == [
        ("user", "fix login"),
        ("thinking", "looking at auth"),
        ("assistant", "Check the redirect."),
        ("user", "also cookies"),
    ]


def test_history_keeps_full_dialogue_messages() -> None:
    user = "user " + ("u" * 4_500)
    assistant = "assistant " + ("a" * 4_500)
    entries = entries_from_wire_messages(
        [
            TurnBegin(user_input=user),
            TextPart(text=assistant[:2_000]),
            TextPart(text=assistant[2_000:]),
        ]
    )
    assert _texts(entries) == [user, assistant]


def test_take_last_turns_uses_history_metadata_not_user_text() -> None:
    entries = tuple(
        HistoryEntry(entry=entry, turn=turn, ordinal=ordinal)
        for turn, values in enumerate(_turns(4))
        for ordinal, entry in enumerate(values)
    )
    history = take_last_turns(entries, max_turns=2)
    assert history.omitted_turns == 2
    assert _texts(history) == ["q2", "a2", "q3", "a3"]


async def test_load_session_history_uses_injected_messages(tmp_path: Path) -> None:
    history = await load_session_history(
        tmp_path,
        "sess-1",
        max_turns=1,
        messages=[
            TurnBegin(user_input="older"),
            TextPart(text="old reply"),
            TurnBegin(user_input="newest"),
            TextPart(text="new reply"),
        ],
    )
    assert history.omitted_turns == 1
    assert _texts(history) == ["newest", "new reply"]


async def test_load_session_history_empty_when_session_missing(tmp_path: Path) -> None:
    history = await load_session_history(tmp_path, "missing-session")
    assert history.entries == ()
    assert history.omitted_turns == 0


async def test_load_session_history_keeps_all_turns_by_default(tmp_path: Path) -> None:
    messages: list[object] = []
    for index in range(6):
        messages.extend((TurnBegin(user_input=f"q{index}"), TextPart(text=f"a{index}")))
    history = await load_session_history(tmp_path, "all", messages=messages)
    assert _texts(history, "user") == [f"q{index}" for index in range(6)]


async def test_tail_loader_reads_only_requested_turns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    from kaos.path import KaosPath
    from kimi_cli.session import Session as CliSession

    work_dir = tmp_path / "project"
    session = await CliSession.create(
        KaosPath.unsafe_from_local_path(work_dir.resolve()).canonical(), "tail-session"
    )
    await session.wire_file.open()
    try:
        for index in range(5):
            await session.wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            await session.wire_file.append_message(TextPart(text=f"a{index}"))
            await session.wire_file.append_message(TurnEnd())
    finally:
        await session.wire_file.close()
    history = await load_session_history(work_dir, session.id, max_turns=2, max_blocks=10)
    assert history.omitted_turns == 3
    assert _texts(history, "user") == ["q3", "q4"]


async def test_indexed_pages_read_only_requested_turns(tmp_path: Path) -> None:
    from kimi_cli.wire.file import WireFile

    wire = WireFile(tmp_path / "wire.jsonl")
    await wire.open()
    try:
        for index in range(6):
            await wire.append_message(TurnBegin(user_input=f"q{index}"))
            await wire.append_message(TextPart(text=f"a{index}"))
            await wire.append_message(TurnEnd())
    finally:
        await wire.close()
    index = _scan_wire_history_index(wire.path)
    assert index.turn_record_counts == (3, 3, 3, 3, 3, 3)
    latest = await load_wire_history_page(index, end_turn=6, page_turns=2)
    older = await load_wire_history_page(index, end_turn=4, page_turns=2)
    assert (latest.start_turn, latest.end_turn, latest.has_older) == (4, 6, True)
    assert _texts(latest, "user") == ["q4", "q5"]
    assert _texts(older, "user") == ["q2", "q3"]
    assert [item.turn for item in older.entries if entry_kind(item.entry) == "user"] == [2, 3]


async def test_indexed_page_caps_auxiliary_entries_but_keeps_dialogue(tmp_path: Path) -> None:
    from kimi_cli.wire.file import WireFile

    wire = WireFile(tmp_path / "verbose.jsonl")
    await wire.open()
    try:
        for index in range(4):
            await wire.append_message(TurnBegin(user_input=f"q{index}"))
            for _ in range(20):
                await wire.append_message(StepInterrupted())
            await wire.append_message(TextPart(text=f"a{index}"))
            await wire.append_message(TurnEnd())
    finally:
        await wire.close()
    page = await load_wire_history_page(
        _scan_wire_history_index(wire.path), end_turn=4, page_turns=4, max_blocks=8
    )
    assert _texts(page, "user") == ["q0", "q1", "q2", "q3"]
    assert _texts(page, "assistant") == ["a0", "a1", "a2", "a3"]
    assert all(entry_kind(item.entry) in {"user", "assistant"} for item in page.entries)


async def test_indexed_history_disables_process_wide_string_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import pydantic_core
    from kimi_cli.wire.file import WireFile

    wire = WireFile(tmp_path / "wire.jsonl")
    await wire.open()
    try:
        await wire.append_message(TurnBegin(user_input="q"))
        await wire.append_message(TextPart(text="a"))
        await wire.append_message(TurnEnd())
    finally:
        await wire.close()
    settings: list[object] = []
    real = pydantic_core.from_json

    def recording(data: object, *args: object, **kwargs: object) -> object:
        settings.append(kwargs.get("cache_strings"))
        return real(data, *args, **kwargs)

    monkeypatch.setattr(pydantic_core, "from_json", recording)
    await load_wire_history_page(_scan_wire_history_index(wire.path), end_turn=1, page_turns=1)
    assert settings and set(settings) == {False}


def test_unknown_sdk_object_without_data_is_ignored() -> None:
    assert entries_from_wire_messages([SimpleNamespace(noise=True), TurnEnd()]) == ()


def test_tool_fragments_and_result_fold_into_one_activity_entry() -> None:
    entries = entries_from_wire_messages(
        [
            TurnBegin(user_input="read it"),
            ToolCall(id="call-1", function=ToolCall.FunctionBody(name="read", arguments="")),
            ToolCallPart(arguments_part='{"path":'),
            ToolCallPart(arguments_part='"a.py"}'),
            ToolResult(
                tool_call_id="call-1",
                return_value=ToolReturnValue(
                    is_error=False,
                    output="contents",
                    message="success",
                    display=[BriefDisplayBlock(text="12 lines")],
                ),
            ),
        ]
    )
    assert [entry_kind(item.entry) for item in entries] == ["user", "tool"]
    activity = entries[1].entry
    assert isinstance(activity, ActivityEntry)
    assert activity.activity.call_id == "call-1"
    assert activity.activity.call is not None and activity.activity.result is not None
    assert activity.activity.result.status == "ok"
    assert "contents" in entry_body_text(activity)


def test_history_and_live_reducers_produce_identical_ast() -> None:
    messages = [
        TurnBegin(user_input="read it"),
        TextPart(text="I will check. "),
        TextPart(text="Now."),
        ToolCall(id="call-1", function=ToolCall.FunctionBody(name="read", arguments="")),
        ToolCallPart(arguments_part='{"path":'),
        ToolCallPart(arguments_part='"C:/src/key: value.py"}'),
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False,
                output="key: value\nnot metadata",
                message="success",
                display=[BriefDisplayBlock(text="2 lines")],
            ),
        ),
        TurnEnd(),
    ]
    normalizer = WireNormalizer()
    reducer = TranscriptReducer()
    for message in messages:
        normalized = normalizer.normalize(message)
        for mutation in normalized.mutations:
            reducer.apply(mutation)
    for mutation in normalizer.finish():
        reducer.apply(mutation)

    historical = entries_from_wire_messages(messages)
    assert tuple(item.entry for item in historical) == tuple(reducer.entries)


def test_accumulator_turn_and_record_budgets_keep_dialogue() -> None:
    accumulator = HistoryAccumulator(max_turns=2, max_blocks=4)
    for index in range(5):
        accumulator.feed(TurnBegin(user_input=f"q{index}"))
        for _ in range(5):
            accumulator.feed(StepInterrupted())
        accumulator.feed(TextPart(text=f"a{index}"))
    history = accumulator.finish()
    assert history.omitted_turns == 3
    assert _texts(history, "user") == ["q3", "q4"]
    assert _texts(history, "assistant") == ["a3", "a4"]


def test_timeline_exposes_frozen_history_entries() -> None:
    timeline = Timeline.from_turn_entries(_turns(3))
    assert timeline.total_turns == 3
    assert timeline.materialized_turn_count == 3
    assert _texts(timeline.history_entries()) == ["q0", "a0", "q1", "a1", "q2", "a2"]
    assert timeline.turn_at_line(timeline.first_line_of_turn(2)) == 2
    assert timeline.virtual_lines() > 0


async def test_timeline_drops_whole_turns_and_rematerializes_complete_entries() -> None:
    timeline = Timeline.from_turn_entries(_turns(8))
    timeline.unload_distant(keep_turn=7, radius=1)
    assert timeline.entries_for_turn(0) is None
    assert _texts(timeline.entries_for_turn(6) or ()) == ["q6", "a6"]
    await timeline.materialize_turns(0, 1)
    assert _texts(timeline.entries_for_turn(0) or ()) == ["q0", "a0"]


async def test_timeline_slide_obeys_record_budget_and_skips_gap() -> None:
    timeline = Timeline.from_turn_entries(_turns(20))
    timeline.hydrated_record_budget = 8
    await timeline.slide_to(0)
    assert (timeline.first_materialized_turn(), timeline.last_materialized_turn()) == (0, 3)
    assert timeline.entries_for_turn(19) is None
    await timeline.slide_to(19)
    assert timeline.entries_for_turn(0) is None
    assert (timeline.first_materialized_turn(), timeline.last_materialized_turn()) == (16, 19)


async def test_timeline_uses_recursive_entry_cost() -> None:
    turns = [
        [TextEntry(f"entry:{index}", "user", (TextBlock(literal(str(index) * 100)),))]
        for index in range(10)
    ]
    timeline = Timeline.from_turn_entries(turns)
    timeline.hydrated_budget = 280
    timeline.hydrated_record_budget = 100
    await timeline.slide_to(4)
    assert timeline.entries_for_turn(4) is not None
    assert timeline.hydrated_chars() <= 280
    assert timeline.materialized_turn_count <= 2


async def test_timeline_keeps_one_oversized_anchor_as_soft_limit() -> None:
    timeline = Timeline.from_turn_entries(
        [
            [TextEntry("small", "user", (TextBlock(literal("small")),))],
            [TextEntry("large", "assistant", (TextBlock(literal("x" * 1_000)),))],
        ]
    )
    timeline.hydrated_budget = 10
    await timeline.slide_to(1)
    assert timeline.materialized_turn_count == 1
    assert _texts(timeline.entries_for_turn(1) or ()) == ["x" * 1_000]


def test_parallel_tool_results_update_their_own_activities() -> None:
    entries = entries_from_wire_messages(
        [
            TurnBegin(user_input="go"),
            ToolCall(id="call-a", function=ToolCall.FunctionBody(name="read", arguments="{}")),
            ToolCall(id="call-b", function=ToolCall.FunctionBody(name="write", arguments="{}")),
            ToolResult(
                tool_call_id="call-b",
                return_value=ToolReturnValue(
                    is_error=True, output="", message="disk full", display=[]
                ),
            ),
            ToolResult(
                tool_call_id="call-a",
                return_value=ToolReturnValue(
                    is_error=False, output="contents", message="success", display=[]
                ),
            ),
        ]
    )
    activities = [item.entry for item in entries if isinstance(item.entry, ActivityEntry)]
    assert [entry.activity.call_id for entry in activities] == ["call-a", "call-b"]
    assert [entry.activity.result.status for entry in activities if entry.activity.result] == [
        "ok",
        "error",
    ]


def test_valid_unknown_envelope_becomes_a_raw_entry(tmp_path: Path) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_bytes(
        b'{"type":"message","message":{"type":"FutureEvent","payload":{"key":"value"}}}\n'
    )
    turns = _read_wire_turns(path, 0, path.stat().st_size)
    assert len(turns) == 1
    assert entry_kind(turns[0][0]) == "system"
    assert '"key": "value"' in entry_body_text(turns[0][0])


def test_malformed_json_is_logged_and_skipped(tmp_path: Path, caplog) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_bytes(b"{broken}\n")
    assert _read_wire_turns(path, 0, path.stat().st_size) == []
    assert "Skipping malformed wire JSON" in caplog.text
