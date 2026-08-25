"""Tests for patch mode of the edit tool."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file import EditFile
from kimi_cli.tools.file.edit.params import EditParams, PatchEntry


@pytest.fixture
def patch_tool(edit_file_tool: EditFile) -> EditFile:
    return edit_file_tool


async def test_patch_update_exact_hunk(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "sample.txt"
    await file_path.write_text("line1\nline2\nline3\n")

    diff = "@@ -2,3 +2,3 @@\n line2\n-line3\n+line3changed\n"
    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="update", diff=diff)])
    )

    assert not result.is_error
    assert "successfully patched" in result.message
    assert await file_path.read_text() == "line1\nline2\nline3changed\n"


async def test_patch_update_multiple_hunks(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "sample.txt"
    await file_path.write_text("a\nb\nc\nd\ne\n")

    diff = (
        "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
        "@@ -4,3 +4,3 @@\n d\n-e\n+E\n"
    )
    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="update", diff=diff)])
    )

    assert not result.is_error
    assert await file_path.read_text() == "a\nB\nc\nd\nE\n"


async def test_patch_create_file(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "created.txt"
    diff = "+hello\n+world\n"

    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="create", diff=diff)])
    )

    assert not result.is_error
    assert await file_path.read_text() == "hello\nworld\n"


async def test_patch_create_refuses_overwrite(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "existing.txt"
    await file_path.write_text("existing")

    diff = "+new\n"
    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="create", diff=diff)])
    )

    assert result.is_error
    assert "already exists" in result.message
    assert await file_path.read_text() == "existing"


async def test_patch_delete_file(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "delete_me.txt"
    await file_path.write_text("bye")

    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="delete")])
    )

    assert not result.is_error
    assert not await file_path.exists()


async def test_patch_update_missing_file(patch_tool, temp_work_dir):
    file_path = temp_work_dir / "missing.txt"
    diff = "@@ -1,2 +1,2 @@\n-old\n+new\n"

    result = await patch_tool(
        EditParams(path=str(file_path), mode="patch", edits=[PatchEntry(op="update", diff=diff)])
    )

    assert result.is_error
    assert "not found" in result.message.lower()


async def test_patch_update_rename(patch_tool, temp_work_dir):
    source = temp_work_dir / "old_name.txt"
    dest = temp_work_dir / "new_name.txt"
    await source.write_text("a\nb\n")

    diff = "@@ -2,1 +2,1 @@\n-b\n+B\n"
    result = await patch_tool(
        EditParams(
            path=str(source),
            mode="patch",
            edits=[PatchEntry(op="update", diff=diff, rename=str(dest))],
        )
    )

    assert not result.is_error
    assert not await source.exists()
    assert await dest.read_text() == "a\nB\n"
