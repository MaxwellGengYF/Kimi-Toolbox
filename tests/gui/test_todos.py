from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kimi_agent_sdk import (
    BriefDisplayBlock,
    TodoDisplayBlock,
    ToolResult,
    ToolReturnValue,
)
from kimix_gui.todos import (
    EMPTY_SNAPSHOT,
    TodoEntry,
    TodoSnapshot,
    flatten_todo_tree,
    read_snapshot_file,
    session_state_file,
    snapshot_from_display_blocks,
    snapshot_from_state_data,
    snapshot_from_wire_message,
)


def _state(*, todos: list[dict], archived: int = 0) -> dict:
    return {
        "version": 1,
        "todos": todos,
        "archived_todos": [
            {"title": f"old {index}", "status": "done"} for index in range(archived)
        ],
    }


def test_flatten_tree_keeps_depth_first_order_and_depth() -> None:
    entries = flatten_todo_tree(
        [
            {
                "title": "Parent",
                "status": "in_progress",
                "notes": "  keep going  ",
                "children": [
                    {"title": "Child", "status": "done"},
                    {
                        "title": "Grandparent",
                        "status": "pending",
                        "children": [
                            {"title": "Deep", "status": "pending"},
                        ],
                    },
                ],
            },
            {"title": "Sibling", "status": "pending"},
        ]
    )
    assert entries == (
        TodoEntry("Parent", "in_progress", "keep going", 0),
        TodoEntry("Child", "done", "", 1),
        TodoEntry("Grandparent", "pending", "", 1),
        TodoEntry("Deep", "pending", "", 2),
        TodoEntry("Sibling", "pending", "", 0),
    )


def test_flatten_tree_survives_malformed_input() -> None:
    assert flatten_todo_tree(None) == ()
    assert flatten_todo_tree("nope") == ()
    assert flatten_todo_tree([{"status": "pending"}]) == ()
    assert flatten_todo_tree([{"title": "  a  b ", "status": "weird"}]) == (
        TodoEntry("a b", "pending", "", 0),
    )
    assert flatten_todo_tree([{"title": "x", "status": "completed"}]) == (
        TodoEntry("x", "done", "", 0),
    )


def test_snapshot_counts_and_active_item() -> None:
    snapshot = snapshot_from_state_data(
        _state(
            todos=[
                {"title": "a", "status": "done"},
                {"title": "b", "status": "in_progress"},
                {"title": "c", "status": "pending"},
                {"title": "d", "status": "pending"},
            ],
            archived=2,
        )
    )
    assert snapshot.total == 4
    assert (snapshot.done, snapshot.in_progress, snapshot.pending) == (1, 1, 2)
    assert snapshot.archived == 2
    assert snapshot.all_done is False
    assert snapshot.active is not None
    assert snapshot.active.title == "b"


def test_snapshot_active_falls_back_to_first_pending() -> None:
    snapshot = snapshot_from_state_data(
        _state(todos=[{"title": "a", "status": "done"}, {"title": "b", "status": "pending"}])
    )
    assert snapshot.active is not None
    assert snapshot.active.title == "b"


def test_snapshot_all_done_has_no_active_item() -> None:
    snapshot = snapshot_from_state_data(_state(todos=[{"title": "a", "status": "done"}]))
    assert snapshot.all_done is True
    assert snapshot.active is None


def test_snapshot_from_state_data_rejects_non_mapping() -> None:
    assert snapshot_from_state_data([]) == EMPTY_SNAPSHOT
    assert snapshot_from_state_data(None) == EMPTY_SNAPSHOT


def test_snapshot_from_display_blocks_uses_last_todo_block() -> None:
    snapshot = snapshot_from_display_blocks(
        [
            BriefDisplayBlock(text="ignored"),
            TodoDisplayBlock(items=[{"title": "stale", "status": "pending"}]),
            TodoDisplayBlock(
                items=[
                    {"title": "root", "status": "in_progress"},
                    {"title": "leaf", "status": "pending", "depth": 1, "notes": "later"},
                ]
            ),
        ]
    )
    assert snapshot == TodoSnapshot(
        entries=(
            TodoEntry("root", "in_progress", "", 0),
            TodoEntry("leaf", "pending", "later", 1),
        )
    )


def test_snapshot_from_display_blocks_without_todo_block() -> None:
    assert snapshot_from_display_blocks([BriefDisplayBlock(text="x")]) is None
    assert snapshot_from_display_blocks([]) is None


def test_snapshot_from_wire_message_reads_tool_results() -> None:
    message = ToolResult(
        tool_call_id="call-1",
        return_value=ToolReturnValue(
            is_error=False,
            output="",
            message="",
            display=[TodoDisplayBlock(items=[{"title": "ship it", "status": "pending"}])],
        ),
    )
    snapshot = snapshot_from_wire_message(message)
    assert snapshot == TodoSnapshot(entries=(TodoEntry("ship it", "pending", "", 0),))


def test_snapshot_from_wire_message_ignores_other_messages() -> None:
    assert snapshot_from_wire_message(SimpleNamespace()) is None
    assert snapshot_from_wire_message("text") is None
    assert (
        snapshot_from_wire_message(
            ToolResult(
                tool_call_id="c",
                return_value=ToolReturnValue(is_error=False, output="", message="", display=[]),
            )
        )
        is None
    )


def test_with_archived_preserves_entries() -> None:
    snapshot = TodoSnapshot(entries=(TodoEntry("a", "pending"),))
    assert snapshot.with_archived(3).archived == 3
    assert snapshot.with_archived(3).entries == snapshot.entries
    assert snapshot.with_archived(0) is snapshot


def test_read_snapshot_file_handles_missing_and_corrupt(tmp_path: Path) -> None:
    assert read_snapshot_file(None) == EMPTY_SNAPSHOT
    assert read_snapshot_file(tmp_path / "nope.json") == EMPTY_SNAPSHOT
    broken = tmp_path / "state.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_snapshot_file(broken) == EMPTY_SNAPSHOT


def test_read_snapshot_file_reads_state_json(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(_state(todos=[{"title": "resumed", "status": "in_progress"}], archived=1)),
        encoding="utf-8",
    )
    snapshot = read_snapshot_file(state_file)
    assert snapshot.entries == (TodoEntry("resumed", "in_progress", "", 0),)
    assert snapshot.archived == 1


def test_session_state_file_returns_none_for_unknown_work_dir(tmp_path: Path) -> None:
    assert session_state_file(tmp_path, "") is None
    assert session_state_file(tmp_path, "missing-session") is None
