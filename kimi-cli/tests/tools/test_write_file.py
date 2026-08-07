"""Tests for the write_file tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from kaos.path import KaosPath

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval, ApprovalResult
from kimi_cli.tools.file.read import Params as ReadParams
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.write import Params, WriteFile
from tests.conftest import tool_call_context


async def test_write_new_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing a new file."""
    file_path = temp_work_dir / "new_file.txt"
    content = "Hello, World!"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert "successfully overwritten" in result.message
    # The diff is intentionally not attached to the result display: it was
    # already shown during approval, and the streamed content argument is
    # printed live by the CLI printer (see kimix.base).
    assert result.display == []
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_overwrite_existing_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test overwriting an existing file."""
    file_path = temp_work_dir / "existing.txt"
    original_content = "Original content"
    await file_path.write_text(original_content)

    new_content = "New content"
    result = await write_file_tool(Params(path=str(file_path), content=new_content))

    assert not result.is_error
    assert "successfully overwritten" in result.message
    assert await file_path.read_text() == new_content


async def test_append_to_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test appending to an existing file."""
    file_path = temp_work_dir / "append_test.txt"
    original_content = "First line\n"
    await file_path.write_text(original_content)

    append_content = "Second line\n"
    result = await write_file_tool(
        Params(path=str(file_path), content=append_content, mode="append")
    )

    assert not result.is_error
    assert "successfully appended to" in result.message
    expected_content = original_content + append_content
    assert await file_path.read_text() == expected_content


async def test_write_unicode_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing unicode content."""
    file_path = temp_work_dir / "unicode.txt"
    content = "Hello 世界 🌍\nUnicode: café, naïve, résumé"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text(encoding="utf-8") == content


async def test_write_empty_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing empty content."""
    file_path = temp_work_dir / "empty.txt"
    content = ""

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_write_multiline_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing multiline content."""
    file_path = temp_work_dir / "multiline.txt"
    content = "Line 1\nLine 2\nLine 3\n"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.read_text() == content


async def test_write_with_relative_path(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing with a relative path inside the work directory."""
    relative_dir = temp_work_dir / "relative" / "path"
    await relative_dir.mkdir(parents=True, exist_ok=True)

    result = await write_file_tool(Params(path="relative/path/file.txt", content="content"))

    assert not result.is_error
    assert await (temp_work_dir / "relative" / "path" / "file.txt").read_text() == "content"


async def test_write_outside_work_directory(write_file_tool: WriteFile, outside_file: Path):
    """Test writing outside the working directory with an absolute path."""
    result = await write_file_tool(Params(path=str(outside_file), content="content"))

    assert not result.is_error
    assert outside_file.read_text() == "content"


async def test_write_outside_work_directory_with_prefix(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Paths sharing the same prefix as work dir should still be writable with absolute paths."""
    base = Path(str(temp_work_dir))
    sneaky_dir = base.parent / f"{base.name}-sneaky"
    sneaky_dir.mkdir(parents=True, exist_ok=True)
    sneaky_file = sneaky_dir / "file.txt"

    result = await write_file_tool(Params(path=str(sneaky_file), content="content"))

    assert not result.is_error
    assert sneaky_file.read_text() == "content"


async def test_write_to_nonexistent_directory(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing to a non-existent directory auto-creates parents."""
    file_path = temp_work_dir / "nonexistent" / "file.txt"

    result = await write_file_tool(Params(path=str(file_path), content="content"))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == "content"


async def test_append_to_nonexistent_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test appending to a non-existent file (should create it)."""
    file_path = temp_work_dir / "new_append.txt"
    content = "New content\n"

    result = await write_file_tool(Params(path=str(file_path), content=content, mode="append"))

    assert not result.is_error
    assert "successfully appended to" in result.message
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_write_large_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing large content."""
    file_path = temp_work_dir / "large.txt"
    content = "Large content line\n" * 1000

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == content


# --- Comprehensive edge-case tests ---


async def test_write_empty_path(write_file_tool: WriteFile):
    """Test writing with an empty path."""
    result = await write_file_tool(Params(path="", content="content"))
    assert result.is_error
    assert "File path cannot be empty" in result.message


async def test_write_relative_path_outside_workspace(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Test writing with a relative path that resolves outside the workspace."""
    result = await write_file_tool(Params(path="../outside.txt", content="content"))
    assert result.is_error
    assert "absolute path" in result.message.lower()


async def test_write_to_directory(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing to an existing directory path."""
    dir_path = temp_work_dir / "subdir"
    await dir_path.mkdir()

    result = await write_file_tool(Params(path=str(dir_path), content="content"))
    assert result.is_error
    assert "is a directory" in result.message


async def test_write_valid_json(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing valid JSON content succeeds."""
    file_path = temp_work_dir / "test.json"
    content = '{"key": "value", "num": 42}'

    result = await write_file_tool(Params(path=str(file_path), content=content))
    assert not result.is_error
    assert await file_path.read_text() == content


async def test_write_invalid_json(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing invalid JSON content returns a format validation error."""
    file_path = temp_work_dir / "test.json"
    content = '{"key": broken}'

    with patch("kimi_cli.tools.file.write.json_repair.repair_json", return_value=""):
        result = await write_file_tool(Params(path=str(file_path), content=content))
    assert result.is_error
    assert "Format validation failed" in result.brief
    assert "JSON decode error" in result.message


async def test_write_valid_xml(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing valid XML content succeeds."""
    file_path = temp_work_dir / "test.xml"
    content = "<root><item>value</item></root>"

    result = await write_file_tool(Params(path=str(file_path), content=content))
    assert not result.is_error
    assert await file_path.read_text() == content


async def test_write_invalid_xml(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing invalid XML content returns a format validation error."""
    file_path = temp_work_dir / "test.xml"
    content = "<root><unclosed></root>"

    result = await write_file_tool(Params(path=str(file_path), content=content))
    assert result.is_error
    assert "Format validation failed" in result.brief
    assert "XML parse error" in result.message


async def test_write_approval_rejected(runtime: Runtime, session, temp_work_dir: KaosPath):
    """Test that a rejected approval returns a ToolRejectedError."""
    file_path = temp_work_dir / "test.txt"

    approval = Approval(yolo=False)
    request_mock = AsyncMock(return_value=ApprovalResult(approved=False))
    approval.request = cast(Any, request_mock)

    with tool_call_context("WriteFile"):
        tool = WriteFile(runtime, approval, session)
        result = await tool(Params(path=str(file_path), content="content"))

    assert result.is_error
    assert "rejected" in result.message.lower()
    request_mock.assert_awaited_once()


async def test_write_exception_during_write(runtime: Runtime, session, temp_work_dir: KaosPath):
    """Test that an exception during write is handled gracefully."""
    file_path = temp_work_dir / "test.txt"

    with tool_call_context("WriteFile"):
        tool = WriteFile(runtime, Approval(yolo=True), session)
        with patch("kaos.path.KaosPath.write_text", side_effect=OSError("disk full")):
            result = await tool(Params(path=str(file_path), content="content"))

    assert result.is_error
    assert "Failed to write" in result.message
    assert "disk full" in result.message


# --- [out of work-dir] warning tests ---


async def test_write_outside_work_dir_has_warning(write_file_tool: WriteFile, outside_file: Path):
    """Writing outside work-dir should include [out of work-dir] in success message."""
    result = await write_file_tool(Params(path=str(outside_file), content="content"))
    assert not result.is_error
    assert "[out of work-dir]" in result.message


async def test_write_inside_work_dir_no_warning(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Writing inside work-dir should NOT include [out of work-dir] in message."""
    file_path = temp_work_dir / "inside.txt"
    result = await write_file_tool(Params(path=str(file_path), content="content"))
    assert not result.is_error
    assert "[out of work-dir]" not in result.message


async def test_write_outside_work_dir_error_has_warning(
    write_file_tool: WriteFile, outside_file: Path
):
    """Error writing outside work-dir should include [out of work-dir] in error message."""
    # Make outside_file a directory to trigger an error
    outside_dir = outside_file.parent / "outside_dir"
    outside_dir.mkdir(parents=True, exist_ok=True)
    result = await write_file_tool(Params(path=str(outside_dir), content="content"))
    assert result.is_error
    assert "[out of work-dir]" in result.message


async def test_write_outside_relative_path_error_has_warning(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Relative path outside workspace error should include [out of work-dir]."""
    result = await write_file_tool(Params(path="../outside.txt", content="content"))
    assert result.is_error
    assert "[out of work-dir]" in result.message


# --- mark_dirty tests ---


async def test_write_after_file_changed_returns_error(
    write_file_tool: WriteFile, temp_work_dir: KaosPath, session
):
    """Writing to a file whose mtime matches the recorded mtime returns error."""
    file_path = temp_work_dir / "changed.txt"
    await file_path.write_text("original content")

    # Pre-populate the tracker with the current mtime so mark_dirty
    # finds an equal timestamp and returns False.
    from kimi_cli.utils.path import kaos_path_from_user_input
    key = str(kaos_path_from_user_input(str(file_path)).canonical())
    st = await file_path.stat()
    session.file_mtime._times[key] = st.st_mtime

    result = await write_file_tool(Params(path=str(file_path), content="new content"))

    assert result.is_error
    assert "File modified" in result.message
    assert "read file first" in result.message


async def test_write_new_file_not_in_dict_succeeds(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Writing a new file (not in tracker dict) should succeed normally."""
    file_path = temp_work_dir / "new_untracked.txt"

    result = await write_file_tool(Params(path=str(file_path), content="new content"))

    assert not result.is_error
    assert "successfully overwritten" in result.message
    assert await file_path.read_text() == "new content"


# --- FileMTime external-modification integration tests ---


def _bump_mtime(path: KaosPath, delta: float = 2.0) -> None:
    """Deterministically move *path*'s mtime forward by *delta* seconds."""
    st = os.stat(str(path))
    os.utime(str(path), (st.st_atime, st.st_mtime + delta))


async def test_write_blocked_after_external_modification(
    read_file_tool: ReadFile,
    write_file_tool: WriteFile,
    temp_work_dir: KaosPath,
):
    """Read via ReadFile, external modification, then WriteFile is blocked."""
    file_path = temp_work_dir / "ext_mod.txt"
    await file_path.write_text("original content")

    # Read it — records the read baseline via clean_file
    result = await read_file_tool(ReadParams(path=str(file_path)))
    assert not result.is_error

    # External modification bumps the mtime past the read baseline
    _bump_mtime(file_path)

    result = await write_file_tool(Params(path=str(file_path), content="clobbered"))
    assert result.is_error
    assert "File modified" in result.message
    assert "read file first" in result.message


async def test_write_after_write_without_read_blocked(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Two consecutive writes with no read in between: the second is blocked."""
    file_path = temp_work_dir / "double_write.txt"

    result1 = await write_file_tool(Params(path=str(file_path), content="first"))
    assert not result1.is_error

    result2 = await write_file_tool(Params(path=str(file_path), content="second"))
    assert result2.is_error
    assert "File modified" in result2.message
    assert "read file first" in result2.message


async def test_write_succeeds_after_read(
    read_file_tool: ReadFile,
    write_file_tool: WriteFile,
    temp_work_dir: KaosPath,
):
    """Reading a file via ReadFile legitimizes a subsequent WriteFile."""
    file_path = temp_work_dir / "read_then_write.txt"
    await file_path.write_text("original content")

    result = await read_file_tool(ReadParams(path=str(file_path)))
    assert not result.is_error

    result = await write_file_tool(Params(path=str(file_path), content="new content"))
    assert not result.is_error
    assert "successfully overwritten" in result.message
    assert await file_path.read_text() == "new content"


# --- Post-write size verification tests ---


async def test_write_message_includes_verification(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Success message notes that post-write size verification passed."""
    file_path = temp_work_dir / "verified.txt"
    result = await write_file_tool(Params(path=str(file_path), content="hello world"))
    assert not result.is_error
    assert "Verified: size matches." in result.message


async def test_write_verification_size_mismatch(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """A post-write stat size mismatch reports a verification error."""
    file_path = temp_work_dir / "mismatch.txt"

    class FakeStat:
        st_mode = 0o100644  # regular-file mode so is_dir()/exists() keep working
        st_size = 999999  # wrong size on purpose

    async def fake_stat(self, follow_symlinks: bool = True) -> FakeStat:
        return FakeStat()

    # The real write happens; only the post-write stat reports a wrong size.
    with patch("kaos.path.KaosPath.stat", fake_stat):
        result = await write_file_tool(Params(path=str(file_path), content="hello"))

    assert result.is_error
    assert "verification failed" in result.message.lower()
    assert "Verification failed" in result.brief


# ============================================================================
# Fuzzy mode matching tests
# ============================================================================


class TestWriteFileFuzzyMode:
    """Test fuzzy matching for the mode field."""

    # ── overwrite synonyms ──

    @pytest.mark.parametrize("synonym", [
        "replace", "write", "create", "new", "truncate", "rewrite", "set", "put",
        "over-write", "over_write",
    ])
    async def test_fuzzy_overwrite_synonyms(
        self, write_file_tool: WriteFile, temp_work_dir: KaosPath, synonym: str
    ):
        """All overwrite synonyms should be accepted and canonicalized to 'overwrite'."""
        file_path = temp_work_dir / f"fuzzy_overwrite_{synonym}.txt"
        result = await write_file_tool(Params(path=str(file_path), content="hello", mode=synonym))  # type: ignore[reportArgumentType]
        assert not result.is_error
        assert "successfully overwritten" in result.message
        assert await file_path.read_text() == "hello"

    # ── append synonyms ──

    @pytest.mark.parametrize("synonym", [
        "add", "concat", "concatenate", "extend", "attach", "insert", "prepend", "after",
    ])
    async def test_fuzzy_append_synonyms(
        self, write_file_tool: WriteFile, temp_work_dir: KaosPath, synonym: str
    ):
        """All append synonyms should be accepted and canonicalized to 'append'."""
        file_path = temp_work_dir / f"fuzzy_append_{synonym}.txt"
        await file_path.write_text("existing\n")
        result = await write_file_tool(Params(path=str(file_path), content="added\n", mode=synonym))  # type: ignore[reportArgumentType]
        assert not result.is_error
        assert "successfully appended to" in result.message
        assert await file_path.read_text() == "existing\nadded\n"

    # ── normalisation ──

    async def test_fuzzy_mode_case_insensitive(
        self, write_file_tool: WriteFile, temp_work_dir: KaosPath
    ):
        """Mode should be case-insensitive."""
        file_path = temp_work_dir / "case_test.txt"
        result = await write_file_tool(Params(path=str(file_path), content="x", mode="OVERWRITE"))  # type: ignore[reportArgumentType]
        assert not result.is_error
        assert "successfully overwritten" in result.message

    async def test_fuzzy_mode_strips_whitespace(
        self, write_file_tool: WriteFile, temp_work_dir: KaosPath
    ):
        """Mode should be stripped of surrounding whitespace."""
        file_path = temp_work_dir / "strip_test.txt"
        result = await write_file_tool(Params(path=str(file_path), content="x", mode="  append  "))  # type: ignore[reportArgumentType]
        assert not result.is_error
        assert "successfully appended to" in result.message

    # ── invalid modes ──


# ── New params tests ─────────────────────────────────────────────────────────


async def test_auto_fix_json_enabled_repairs_broken_json(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """auto_fix_json=True (default) repairs broken JSON."""
    file_path = temp_work_dir / "config.json"
    broken_json = '{"key": "value"'
    result = await write_file_tool(
        Params(path=str(file_path), content=broken_json)
    )
    assert not result.is_error
    # Should have been repaired and written
    written = await file_path.read_text()
    assert '"key": "value"' in written
    # Should end with curly brace (repaired)
    assert written.strip().endswith("}")


async def test_auto_fix_json_disabled_reports_error(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """auto_fix_json=False reports format error instead of repairing."""
    file_path = temp_work_dir / "broken.json"
    broken_json = '{"key": "value"'
    result = await write_file_tool(
        Params(path=str(file_path), content=broken_json, auto_fix_json=False)
    )
    # With auto_fix_json=False, the JSON error should be reported
    assert result.is_error
    assert "format" in result.message.lower() or "JSON" in result.message


async def test_mkdir_true_creates_parents(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """mkdir=True (default) creates parent directories."""
    file_path = temp_work_dir / "newdir" / "subdir" / "test.txt"
    result = await write_file_tool(
        Params(path=str(file_path), content="hello", mkdir=True)
    )
    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == "hello"


async def test_mkdir_false_fails_on_missing_parent(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """mkdir=False fails when parent directory does not exist."""
    file_path = temp_work_dir / "nonexistent_parent" / "test.txt"
    result = await write_file_tool(
        Params(path=str(file_path), content="hello", mkdir=False)
    )
    assert result.is_error
    assert "Parent directory does not exist" in result.message


async def test_show_diff_includes_diff_in_output(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """show_diff=True includes a diff in the tool output."""
    file_path = temp_work_dir / "diff_test.txt"
    content = "Hello, World!"
    result = await write_file_tool(
        Params(path=str(file_path), content=content, show_diff=True)
    )
    assert not result.is_error
    # Output should be non-empty (diff was included)
    assert len(result.output) > 0
    assert "Diff:" in result.output or "diff" in result.output.lower()


# ── JSON auto-repair diff tests ─────────────────────────────────────────────


async def test_json_auto_repair_reported_in_message(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """When auto_fix_json repairs broken JSON, the success message mentions the repair."""
    file_path = temp_work_dir / "repair_test.json"
    broken_json = '{"key": "value"'  # missing closing brace
    result = await write_file_tool(
        Params(path=str(file_path), content=broken_json)
    )
    assert not result.is_error
    # Success message should mention JSON was auto-repaired
    assert "JSON was auto-repaired" in result.message
    # File should contain valid JSON (repaired)
    written = await file_path.read_text()
    assert written.strip().endswith("}")


async def test_json_auto_repair_diff_in_output(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """When auto_fix_json repairs broken JSON, the repair diff appears in output."""
    file_path = temp_work_dir / "repair_diff_test.json"
    broken_json = '{"key": "value"'  # missing closing brace
    result = await write_file_tool(
        Params(path=str(file_path), content=broken_json)
    )
    assert not result.is_error
    # Output should contain the repair diff
    assert "JSON auto-repair diff" in result.output


async def test_json_auto_repair_disabled_no_diff(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """When auto_fix_json=False, no repair diff should appear."""
    file_path = temp_work_dir / "no_repair.json"
    broken_json = '{"key": "value"'
    result = await write_file_tool(
        Params(path=str(file_path), content=broken_json, auto_fix_json=False)
    )
    assert result.is_error
    # There should be no repair diff in output
    assert "JSON auto-repair diff" not in result.output
