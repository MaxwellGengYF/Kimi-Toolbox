"""Tests for the edit_file tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from kaos.path import KaosPath

from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval, ApprovalResult
from kimi_cli.tools.file.replace import Edit, EditFile, Params
from tests.conftest import tool_call_context


async def test_replace_single_occurrence(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing a single occurrence."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world! This is a test."
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new="universe"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    # The diff is intentionally not attached to the result display: it was
    # already shown during approval, and the streamed old/new arguments are
    # printed live by the CLI printer (see kimix.base).
    assert result.display == []
    assert await file_path.read_text() == "Hello universe! This is a test."


async def test_replace_all_occurrences(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing all occurrences."""
    file_path = temp_work_dir / "test.txt"
    original_content = "apple banana apple cherry apple"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="apple", new="fruit", replace_all=True),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "fruit banana fruit cherry fruit"


async def test_replace_multiple_edits(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test applying multiple edits."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world! Goodbye world!"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=[
                Edit(old="Hello", new="Hi"),
                Edit(old="Goodbye", new="See you"),
            ],
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hi world! See you world!"


async def test_replace_multiline_content(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing multi-line content."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Line 1\nLine 2\nLine 3\n"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="Line 2\nLine 3", new="Modified line 2\nModified line 3"),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Line 1\nModified line 2\nModified line 3\n"


async def test_replace_unicode_content(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing unicode content."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello 世界! café"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="世界", new="地球"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hello 地球! café"


async def test_replace_no_match(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test replacing when the old string is not found."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world!"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="notfound", new="replacement"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message
    assert await file_path.read_text() == original_content  # Content unchanged


async def test_call_with_string_edit_is_repaired_and_succeeds(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Reproduce report #6: edit serialized as a JSON string instead of an object.

    When the model emits the ``edit`` argument as a JSON-encoded string rather
    than a nested object/list, the shared argument-repair machinery should
    parse the JSON string into the expected nested model and apply the edit.
    """
    file_path = temp_work_dir / "test.txt"
    original_content = "  // OpCopyMemory is core — no capabilities needed\nvoid foo() {}\n"
    await file_path.write_text(original_content)

    # This is how the broken tool call looked: `edit` is a string, not an object.
    result = await edit_file_tool.call(
        {
            "path": str(file_path),
            "edit": '{"old": "  // OpCopyMemory is core — no capabilities needed", "new": "  // OpCopyMemory is core - no capabilities needed"}',
        }
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert (
        await file_path.read_text()
        == "  // OpCopyMemory is core - no capabilities needed\nvoid foo() {}\n"
    )


async def test_replace_with_relative_path(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing with a relative path inside the work directory."""
    relative_dir = temp_work_dir / "relative" / "path"
    await relative_dir.mkdir(parents=True, exist_ok=True)
    file_path = relative_dir / "file.txt"
    await file_path.write_text("old content")

    result = await edit_file_tool(
        Params(path="relative/path/file.txt", edit=Edit(old="old", new="new"))
    )

    assert not result.is_error
    assert await file_path.read_text() == "new content"


async def test_replace_outside_work_directory(
    edit_file_tool: EditFile, outside_file: Path
):
    """Test replacing outside the working directory with an absolute path."""
    outside_file.write_text("old content", encoding="utf-8")

    result = await edit_file_tool(
        Params(path=str(outside_file), edit=Edit(old="old", new="new"))
    )

    assert not result.is_error
    assert outside_file.read_text(encoding="utf-8") == "new content"


async def test_replace_outside_work_directory_with_prefix(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Paths sharing the work dir prefix but outside should still be editable
    with absolute paths."""
    base = Path(str(temp_work_dir))
    sneaky_dir = base.parent / f"{base.name}-sneaky"
    sneaky_dir.mkdir(parents=True, exist_ok=True)
    sneaky_file = sneaky_dir / "test.txt"
    sneaky_file.write_text("content", encoding="utf-8")

    result = await edit_file_tool(
        Params(path=str(sneaky_file), edit=Edit(old="content", new="new"))
    )

    assert not result.is_error
    assert sneaky_file.read_text() == "new"


async def test_replace_nonexistent_file(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing in a non-existent file."""
    file_path = temp_work_dir / "nonexistent.txt"

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="old", new="new"))
    )

    assert result.is_error
    assert "does not exist" in result.message


async def test_replace_directory_instead_of_file(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing in a directory instead of a file."""
    dir_path = temp_work_dir / "directory"
    await dir_path.mkdir()

    result = await edit_file_tool(
        Params(path=str(dir_path), edit=Edit(old="old", new="new"))
    )

    assert result.is_error
    assert "is not a file" in result.message


async def test_replace_mixed_multiple_edits(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test multiple edits with different replace_all settings."""
    file_path = temp_work_dir / "test.txt"
    original_content = "apple apple banana apple cherry"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=[
                Edit(old="apple", new="fruit", replace_all=False),  # Only first occurrence
                Edit(
                    old="banana", new="tasty", replace_all=True
                ),  # All occurrences (though only one)
            ],
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "fruit apple tasty apple cherry"


async def test_replace_empty_strings(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing with empty strings."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world!"
    await file_path.write_text(original_content)

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new=""))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hello !"


# --- Comprehensive edge-case tests ---


async def test_replace_empty_path(edit_file_tool: EditFile):
    """Test replacing with an empty path."""
    result = await edit_file_tool(
        Params(path="", edit=Edit(old="old", new="new"))
    )
    assert result.is_error
    assert "File path cannot be empty" in result.message


async def test_replace_relative_path_outside_workspace(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test replacing with a relative path that resolves outside the workspace."""
    result = await edit_file_tool(
        Params(path="../outside.txt", edit=Edit(old="old", new="new"))
    )
    assert result.is_error
    assert "absolute path" in result.message.lower()


async def test_replace_approval_rejected(runtime: Runtime, session: Session, temp_work_dir: KaosPath):
    """Test that a rejected approval returns a ToolRejectedError."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("old content")

    approval = Approval(yolo=False)
    request_mock = AsyncMock(return_value=ApprovalResult(approved=False))
    approval.request = cast(Any, request_mock)

    with tool_call_context("EditFile"):
        tool = EditFile(runtime, approval, session)
        result = await tool(
            Params(path=str(file_path), edit=Edit(old="old", new="new"))
        )

    assert result.is_error
    assert "rejected" in result.message.lower()
    request_mock.assert_awaited_once()


async def test_replace_invalid_json(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test editing a JSON file to make it invalid returns a format error."""
    file_path = temp_work_dir / "test.json"
    await file_path.write_text('{"key": "value"}')

    with patch("json_repair.repair_json", return_value=""):
        result = await edit_file_tool(
            Params(
                path=str(file_path),
                edit=Edit(old='"key": "value"', new='"key": broken'),
            )
        )

    assert result.is_error
    assert "successfully edited" in result.message
    assert "JSON decode error" in result.message
    assert await file_path.read_text() == '{"key": broken}'


async def test_replace_valid_json(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test editing a JSON file with valid JSON succeeds without format error."""
    file_path = temp_work_dir / "test.json"
    await file_path.write_text('{"key": "old"}')

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old='"old"', new='"new"'),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == '{"key": "new"}'


async def test_replace_invalid_xml(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test editing an XML file to make it invalid returns a format error."""
    file_path = temp_work_dir / "test.xml"
    await file_path.write_text("<root>old</root>")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="old", new="<broken"),
        )
    )

    assert result.is_error
    assert "successfully edited" in result.message
    assert "XML parse error" in result.message
    assert await file_path.read_text() == "<root><broken</root>"


async def test_replace_valid_xml(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test editing an XML file with valid XML succeeds without format error."""
    file_path = temp_work_dir / "test.xml"
    await file_path.write_text("<root>old</root>")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="old", new="new"),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "<root>new</root>"


async def test_replace_edit_empty_old(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test that an edit with an empty old string results in no replacements."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello world!")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="", new="X"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message


async def test_replace_edit_old_equals_new(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Test that an edit where old equals new results in no replacements."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello world!")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new="world"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message


async def test_replace_all_no_match(edit_file_tool: EditFile, temp_work_dir: KaosPath):
    """Test replace_all when the old string is not found."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello world!")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="notfound", new="X", replace_all=True))
    )

    assert result.is_error
    assert "No replacements were made" in result.message


async def test_replace_oserror_on_write(runtime: Runtime, session: Session, temp_work_dir: KaosPath):
    """Test that an OSError during write is handled gracefully."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("old content")

    with tool_call_context("EditFile"):
        tool = EditFile(runtime, Approval(yolo=True), session)
        with patch("kaos.path.KaosPath.write_text", side_effect=OSError("disk full")):
            result = await tool(
                Params(path=str(file_path), edit=Edit(old="old", new="new"))
            )

    assert result.is_error
    assert "Failed to edit" in result.message
    assert "disk full" in result.message


async def test_replace_memory_error_propagated(runtime: Runtime, session: Session, temp_work_dir: KaosPath):
    """Test that MemoryError is propagated and not caught."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("old content")

    with tool_call_context("EditFile"):
        tool = EditFile(runtime, Approval(yolo=True), session)
        with patch(
            "kaos.path.KaosPath.read_text", side_effect=MemoryError("out of memory")
        ):
            try:
                await tool(
                    Params(path=str(file_path), edit=Edit(old="old", new="new"))
                )
                raise AssertionError("MemoryError should have been raised")
            except MemoryError as e:
                assert "out of memory" in str(e)


def test_apply_edit_single_replacement():
    """Test _apply_edit with a single replacement."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit("hello world", Edit(old="world", new="universe"))
    assert content == "hello universe"
    assert count == 1
    assert suggestion is None


def test_apply_edit_replace_all():
    """Test _apply_edit with replace_all."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit(
        "a b a c a", Edit(old="a", new="X", replace_all=True)
    )
    assert content == "X b X c X"
    assert count == 3
    assert suggestion is None


def test_apply_edit_empty_old():
    """Test _apply_edit with empty old string returns no changes."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit("hello", Edit(old="", new="X"))
    assert content == "hello"
    assert count == 0
    assert suggestion is None


def test_apply_edit_old_equals_new():
    """Test _apply_edit when old equals new returns no changes."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit("hello", Edit(old="hello", new="hello"))
    assert content == "hello"
    assert count == 0
    assert suggestion is None


def test_apply_edit_no_match():
    """Test _apply_edit when old string is not found."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit("hello", Edit(old="xyz", new="abc"))
    assert content == "hello"
    assert count == 0
    assert suggestion is None


def test_apply_edit_replace_all_no_match():
    """Test _apply_edit with replace_all when old string is not found."""
    tool = object.__new__(EditFile)
    content, count, suggestion = tool._apply_edit("hello", Edit(old="xyz", new="abc", replace_all=True))
    assert content == "hello"
    assert count == 0
    assert suggestion is None


# --- [out of work-dir] warning tests ---


async def test_edit_outside_work_dir_has_warning(
    edit_file_tool: EditFile, outside_file: Path
):
    """Editing outside work-dir should include [out of work-dir] in success message."""
    outside_file.write_text("original content", encoding="utf-8")
    result = await edit_file_tool(
        Params(path=str(outside_file), edit=Edit(old="original", new="modified"))
    )
    assert not result.is_error
    assert "[out of work-dir]" in result.message


async def test_edit_inside_work_dir_no_warning(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Editing inside work-dir should NOT include [out of work-dir] in message."""
    file_path = temp_work_dir / "inside.txt"
    await file_path.write_text("original content")
    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="original", new="modified"))
    )
    assert not result.is_error
    assert "[out of work-dir]" not in result.message


async def test_edit_outside_work_dir_nonexistent_has_warning(
    edit_file_tool: EditFile, outside_file: Path
):
    """Error editing non-existent outside file should include [out of work-dir]."""
    result = await edit_file_tool(
        Params(path=str(outside_file), edit=Edit(old="foo", new="bar"))
    )
    assert result.is_error
    assert "[out of work-dir]" in result.message


async def test_edit_outside_work_dir_directory_has_warning(
    edit_file_tool: EditFile
):
    """Editing a directory outside work-dir should include [out of work-dir]."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await edit_file_tool(
            Params(path=tmpdir, edit=Edit(old="foo", new="bar"))
        )
        assert result.is_error
        assert "[out of work-dir]" in result.message


# --- mark_dirty / external modification tests ---


async def test_edit_after_external_modification_succeeds(
    edit_file_tool: EditFile, temp_work_dir: KaosPath, session
):
    """Editing a file modified externally since the last read succeeds.

    The FileMTime guard was removed: EditFile no longer refuses to edit a
    file whose mtime changed externally. The edit applies to the current
    on-disk content.
    """
    file_path = temp_work_dir / "changed.txt"
    await file_path.write_text("original content")

    # Simulate an external modification: pre-populate a stale read baseline
    # so the recorded read time is older than the file's current mtime.
    from kimi_cli.utils.path import kaos_path_from_user_input
    key = str(kaos_path_from_user_input(str(file_path)).canonical())
    st = await file_path.stat()
    session.file_mtime._read_times[key] = st.st_mtime - 10
    assert session.file_mtime._read_times[key] < (await file_path.stat()).st_mtime

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="original", new="modified"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "modified content"


async def test_edit_new_file_not_in_dict_succeeds(
    edit_file_tool: EditFile, temp_work_dir: KaosPath
):
    """Editing a file not in tracker dict should succeed normally."""
    file_path = temp_work_dir / "new_untracked.txt"
    await file_path.write_text("original content")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="original", new="modified"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "modified content"
