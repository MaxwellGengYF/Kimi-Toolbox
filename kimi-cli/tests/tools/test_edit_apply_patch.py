"""Tests for the standalone apply_patch tool."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file import ApplyPatchFile, EditFile
from kimi_cli.tools.file.edit.modes.patch import ApplyPatchInput, parse_apply_patch
from tests.conftest import tool_call_context


@pytest.fixture
def apply_patch_tool(runtime, approval, session) -> ApplyPatchFile:
    with tool_call_context("apply_patch"):
        yield ApplyPatchFile(runtime, approval, session)


async def test_apply_patch_update_single_file(apply_patch_tool, temp_work_dir):
    file_path = temp_work_dir / "sample.txt"
    await file_path.write_text("line1\nline2\nline3\n")

    envelope = (
        "*** Begin Patch\n"
        "*** Update File: sample.txt\n"
        "@@ -2,2 +2,2 @@\n"
        " line2\n"
        "-line3\n"
        "+line3changed\n"
        "*** End Patch\n"
    )

    result = await apply_patch_tool(ApplyPatchInput(input=envelope))
    assert not result.is_error
    assert await file_path.read_text() == "line1\nline2\nline3changed\n"


async def test_apply_patch_create_and_update(apply_patch_tool, temp_work_dir):
    existing = temp_work_dir / "existing.txt"
    await existing.write_text("a\nb\n")
    new_file = temp_work_dir / "new_file.txt"

    envelope = (
        "*** Begin Patch\n"
        "*** Update File: existing.txt\n"
        "@@ -2,1 +2,1 @@\n"
        "-b\n"
        "+B\n"
        "*** Add File: new_file.txt\n"
        "+hello\n"
        "+world\n"
        "*** End Patch\n"
    )

    result = await apply_patch_tool(ApplyPatchInput(input=envelope))
    assert not result.is_error
    assert await existing.read_text() == "a\nB\n"
    assert await new_file.read_text() == "hello\nworld"


async def test_apply_patch_delete_file(apply_patch_tool, temp_work_dir):
    doomed = temp_work_dir / "doomed.txt"
    await doomed.write_text("delete me")

    envelope = (
        "*** Begin Patch\n"
        "*** Delete File: doomed.txt\n"
        "*** End Patch\n"
    )

    result = await apply_patch_tool(ApplyPatchInput(input=envelope))
    assert not result.is_error
    assert not await doomed.exists()


async def test_apply_patch_parse_error_missing_end(apply_patch_tool):
    envelope = "*** Begin Patch\n*** Update File: foo.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    result = await apply_patch_tool(ApplyPatchInput(input=envelope))
    assert result.is_error
    assert "last line" in result.message.lower() or "parse" in result.message.lower()


async def test_apply_patch_rejects_partial_batch_on_failure(apply_patch_tool, temp_work_dir):
    existing = temp_work_dir / "a.txt"
    await existing.write_text("unchanged\n")
    missing = temp_work_dir / "b.txt"

    envelope = (
        "*** Begin Patch\n"
        "*** Update File: a.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-unchanged\n"
        "+changed\n"
        "*** Update File: b.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-missing\n"
        "+present\n"
        "*** End Patch\n"
    )

    result = await apply_patch_tool(ApplyPatchInput(input=envelope))
    assert result.is_error
    assert await existing.read_text() == "changed\n"
    assert "b.txt" in result.message
    assert "already applied" in result.message or "NOT applied" in result.message


async def test_apply_patch_envelope_via_edit_tool(edit_file_tool, temp_work_dir):
    file_path = temp_work_dir / "via_edit.txt"
    await file_path.write_text("one\ntwo\nthree\n")

    envelope = (
        "*** Begin Patch\n"
        "*** Update File: via_edit.txt\n"
        "@@ -2,2 +2,2 @@\n"
        " two\n"
        "-three\n"
        "+THREE\n"
        "*** End Patch\n"
    )

    from kimi_cli.tools.file.edit.params import EditParams

    result = await edit_file_tool(EditParams(input=envelope))
    assert not result.is_error
    assert await file_path.read_text() == "one\ntwo\nTHREE\n"


def test_parse_apply_patch_heredoc_wrapper():
    text = (
        "<<EOF\n"
        "*** Begin Patch\n"
        "*** Add File: x.txt\n"
        "+hi\n"
        "*** End Patch\n"
        "EOF\n"
    )
    hunks = parse_apply_patch(text)
    assert len(hunks) == 1
    assert hunks[0].path == "x.txt"
    assert hunks[0].op == "create"
    assert hunks[0].diff == "hi"
