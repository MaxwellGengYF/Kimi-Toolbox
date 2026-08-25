"""Tests for conflict-marker and staleness guards in edit/write."""

from __future__ import annotations

import pytest
from kaos.path import KaosPath

from kimi_cli.tools.file import EditFile
from kimi_cli.tools.file.edit.params import EditParams, ReplaceEditItem
from kimi_cli.tools.file.write import Params as WriteParams, WriteFile


@pytest.fixture
def conflict_edit_tool(edit_file_tool: EditFile) -> EditFile:
    return edit_file_tool


async def test_edit_refuses_conflict_markers(conflict_edit_tool, temp_work_dir):
    file_path = temp_work_dir / "conflict.txt"
    await file_path.write_text("before\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\nafter\n")

    result = await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="before", new="start")],
        )
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message
    assert await file_path.read_text() == "before\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\nafter\n"


async def test_edit_allows_conflict_markers_when_flag_set(conflict_edit_tool, temp_work_dir):
    file_path = temp_work_dir / "conflict.txt"
    await file_path.write_text("before\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\nafter\n")

    result = await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="before", new="start")],
            allow_conflicts=True,
        )
    )
    assert not result.is_error
    assert await file_path.read_text() == "start\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\nafter\n"


async def test_write_refuses_conflict_markers(write_file_tool: WriteFile, temp_work_dir):
    file_path = temp_work_dir / "conflict.txt"
    await file_path.write_text("old\n")

    result = await write_file_tool(
        WriteParams(path=str(file_path), content="<<<<<<< HEAD\na\n=======\nb\n>>>>>>> branch\n")
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message


async def test_write_allows_overwriting_conflict_markers(write_file_tool: WriteFile, temp_work_dir):
    file_path = temp_work_dir / "conflict.txt"
    await file_path.write_text("old\n")

    result = await write_file_tool(
        WriteParams(
            path=str(file_path),
            content="resolved\n",
        )
    )
    assert not result.is_error
    assert await file_path.read_text() == "resolved\n"
