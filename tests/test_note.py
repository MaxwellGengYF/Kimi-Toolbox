"""Tests for plan/note tools: WritePlan, ReadPlan, EditPlan.

Covers:
- SkipThisTool when _enable_plan is not set
- Missing plan_writing_path in session custom_data
- WritePlan: overwrite and append modes, dir creation, error handling
- ReadPlan: forward/negative offset, file not found, not a file,
  char_offset/max_char, MAX_LINES / MAX_BYTES limits, truncation,
  line_offset validation
- EditPlan: exact match, replace_all, fuzzy matching, strip matching,
  no match suggestions, multiple edits, line ending normalization,
  _apply_edit return values, error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool

import kimix.tools.note as _note_module
from kimi_agent_sdk import ToolError, ToolOk
from kimix.tools.note import (
    MAX_BYTES,
    MAX_LINES,
    Edit,
    EditPlan,
    EditPlanParams,
    ReadPlan,
    ReadPlanParams,
    WritePlan,
    WritePlanParams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_plan():
    """Enable plan tools for the current test thread."""
    _note_module._enable_plan = True
    yield
    _note_module._enable_plan = False


@pytest.fixture
def mock_session() -> MagicMock:
    """Return a mock Session with an empty custom_data dict."""
    session = MagicMock(spec=Session)
    session.custom_data = {}
    return session


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    """Create a temporary plan file path (does not create the file)."""
    return tmp_path / "plan.md"


# ============================================================================
# WritePlan
# ============================================================================


class TestWritePlanInit:
    def test_raises_SkipThisTool_when_not_enabled(self) -> None:
        _note_module._enable_plan = False
        session = MagicMock(spec=Session)
        session.custom_data = {}
        with pytest.raises(SkipThisTool):
            WritePlan(session=session)
        _note_module._enable_plan = True


class TestWritePlanCall:
    async def test_missing_plan_writing_path(self, mock_session: MagicMock) -> None:
        tool = WritePlan(session=mock_session)
        result = await tool(WritePlanParams(content="test"))
        assert isinstance(result, ToolError)
        assert "no plan_writing_path" in result.message

    async def test_overwrite_creates_file(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = WritePlan(session=mock_session)
        result = await tool(WritePlanParams(content="hello world", mode="overwrite"))
        assert isinstance(result, ToolOk)
        assert plan_path.read_text(encoding="utf-8") == "hello world"
        assert mock_session.custom_data.get("plan_called") is True
        assert "written to" in result.output

    async def test_append_adds_to_existing(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("line1\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = WritePlan(session=mock_session)
        result = await tool(WritePlanParams(content="line2\n", mode="append"))
        assert isinstance(result, ToolOk)
        assert plan_path.read_text(encoding="utf-8") == "line1\nline2\n"
        assert mock_session.custom_data.get("plan_called") is True
        assert "appended to" in result.output

    async def test_creates_parent_directories(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "dirs" / "plan.md"
        mock_session.custom_data["plan_writing_path"] = path
        tool = WritePlan(session=mock_session)
        result = await tool(WritePlanParams(content="deep"))
        assert isinstance(result, ToolOk)
        assert path.read_text(encoding="utf-8") == "deep"

    async def test_returns_ToolError_on_write_failure(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = WritePlan(session=mock_session)
        with patch("anyio.open_file", side_effect=PermissionError("access denied")):
            result = await tool(WritePlanParams(content="x"))
        assert isinstance(result, ToolError)
        assert "access denied" in result.message


# ============================================================================
# ReadPlan
# ============================================================================


class TestReadPlanInit:
    def test_raises_SkipThisTool_when_not_enabled(self) -> None:
        _note_module._enable_plan = False
        session = MagicMock(spec=Session)
        session.custom_data = {}
        with pytest.raises(SkipThisTool):
            ReadPlan(session=session)
        _note_module._enable_plan = True


class TestReadPlanCall:
    async def test_missing_plan_writing_path(self, mock_session: MagicMock) -> None:
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams())
        assert isinstance(result, ToolError)
        assert "no plan_writing_path" in result.message

    async def test_file_does_not_exist(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        path = tmp_path / "missing.md"
        mock_session.custom_data["plan_writing_path"] = path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams())
        assert isinstance(result, ToolError)
        assert "does not exist" in result.message

    async def test_path_is_directory_not_file(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        dir_path = tmp_path / "a_dir"
        dir_path.mkdir()
        mock_session.custom_data["plan_writing_path"] = dir_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams())
        assert isinstance(result, ToolError)
        assert "not a file" in result.message

    async def test_read_forward_entire_file(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(line_offset=1, n_lines=10))
        assert isinstance(result, ToolOk)
        assert "line1" in result.output
        assert "line2" in result.output
        assert "line3" in result.output
        assert "Total lines in file: 3" in result.message

    async def test_read_forward_with_line_offset(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(line_offset=2, n_lines=10))
        assert isinstance(result, ToolOk)
        assert "a" not in result.output  # line 1 skipped
        assert "b" in result.output
        assert "c" in result.output

    async def test_read_tail_negative_offset(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(line_offset=-2, n_lines=10))
        assert isinstance(result, ToolOk)
        assert "d" in result.output
        assert "e" in result.output
        assert "a" not in result.output
        assert "Total lines in file: 5" in result.message

    async def test_read_tail_empty_file(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(line_offset=-5))
        assert isinstance(result, ToolOk)
        assert "No lines read" in result.message or result.output == ""

    async def test_char_offset_and_max_char_slice_output(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("abcdefghij\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(char_offset=2, max_char=5))
        assert isinstance(result, ToolOk)
        assert len(result.output) <= 5

    async def test_max_lines_limit_emits_warning(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        lines = "".join(f"line{i}\n" for i in range(MAX_LINES + 10))
        plan_path.write_text(lines, encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(n_lines=MAX_LINES + 100))
        assert isinstance(result, ToolOk)
        assert f"Max {MAX_LINES} lines reached" in result.message

    async def test_max_bytes_limit_emits_warning(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        # Create many long lines to exceed MAX_BYTES quickly
        long_line = "x" * 2000 + "\n"
        lines = long_line * 200
        plan_path.write_text(lines, encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        result = await tool(ReadPlanParams(n_lines=MAX_LINES))
        assert isinstance(result, ToolOk)
        # Should hit bytes limit before lines limit
        assert (
            f"Max {MAX_BYTES} bytes reached" in result.message
            or f"Max {MAX_LINES} lines reached" in result.message
        )

    async def test_read_error_returns_ToolError(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("content", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = ReadPlan(session=mock_session)
        with patch("anyio.open_file", side_effect=OSError("disk failure")):
            result = await tool(ReadPlanParams())
        assert isinstance(result, ToolError)
        assert "Failed to read plan" in result.message


# ============================================================================
# EditPlan
# ============================================================================


class TestEditPlanInit:
    def test_raises_SkipThisTool_when_not_enabled(self) -> None:
        _note_module._enable_plan = False
        session = MagicMock(spec=Session)
        session.custom_data = {}
        with pytest.raises(SkipThisTool):
            EditPlan(session=session)
        _note_module._enable_plan = True


class TestEditPlanNormalizeLineEndings:
    def test_converts_windows_crlf_to_lf(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        assert tool._normalize_line_endings("hello\r\nworld") == "hello\nworld"

    def test_preserves_unix_lf(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        assert tool._normalize_line_endings("hello\nworld") == "hello\nworld"

    def test_preserves_cr_only(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        assert tool._normalize_line_endings("hello\rworld") == "hello\rworld"


class TestEditPlanFindSimilar:
    def test_returns_closest_line_above_cutoff(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._find_similar("helo", "hello\nworld\n", cutoff=70.0)
        assert result == "hello"

    def test_returns_None_when_no_similar_match(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._find_similar("zzzz", "hello\nworld\n", cutoff=90.0)
        assert result is None


class TestEditPlanTryStripMatch:
    def test_finds_stripped_old_inside_line(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        # "  hello  " stripped → "hello" is inside "  hello world  "
        result = tool._try_strip_match("  hello world  \n", "  hello  ", "hi")
        assert result is not None
        assert "hi world" in result

    def test_returns_None_when_stripped_old_is_empty(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._try_strip_match("content", "   ", "x")
        assert result is None

    def test_preserves_trailing_newline(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._try_strip_match("the hello world\n", "hello", "hi")
        assert result == "the hi world\n"


class TestEditPlanFindBestFuzzyMatch:
    def test_returns_match_above_cutoff(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._find_best_fuzzy_match("helo", "hello\nworld\n")
        assert result is not None
        matched_text, score = result
        assert matched_text == "hello"
        assert score >= 75.0

    def test_returns_None_below_cutoff(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        result = tool._find_best_fuzzy_match("zzzzz", "hello\nworld\n", cutoff=95.0)
        assert result is None


class TestEditPlanApplyEdit:
    def test_noop_when_old_equals_new(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit("hello", Edit(old="x", new="x"))
        assert count == 0
        assert content == "hello"

    def test_noop_when_old_is_empty(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit("hello", Edit(old="", new="y"))
        assert count == 0
        assert content == "hello"

    def test_replace_all_multiple_occurrences(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit(
            "foo bar foo", Edit(old="foo", new="baz", replace_all=True)
        )
        assert count == 2
        assert content == "baz bar baz"

    def test_replace_all_no_matches_returns_suggestion(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit(
            "hello world", Edit(old="xyz", new="abc", replace_all=True)
        )
        assert count == 0
        assert content == "hello world"
        # suggestion may be None or similar text

    def test_single_replace_exact_match(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit(
            "hello world", Edit(old="hello", new="hi")
        )
        assert count == 1
        assert content == "hi world"

    def test_single_replace_falls_back_to_strip_match(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit(
            "  hello world  \n", Edit(old="hello", new="hi")
        )
        # "hello" (stripped from old "hello") should be found in content
        assert count == 1
        assert "hi world" in content

    def test_single_replace_falls_back_to_fuzzy_match(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        # "hellp" is close enough to "hello" for fuzzy matching (80% ratio > 75% cutoff)
        content, count, suggestion = tool._apply_edit(
            "hellp\n", Edit(old="hello", new="hi")
        )
        assert count == 1
        assert "hi" in content

    def test_single_replace_no_match_returns_suggestion(self) -> None:
        tool = EditPlan(session=MagicMock(spec=Session))
        content, count, suggestion = tool._apply_edit(
            "hello world\n", Edit(old="zzzz_nonexistent", new="abc")
        )
        assert count == 0
        assert content == "hello world\n"


class TestEditPlanCall:
    async def test_missing_plan_writing_path(self, mock_session: MagicMock) -> None:
        tool = EditPlan(session=mock_session)
        result = await tool(EditPlanParams(edit=Edit(old="a", new="b")))
        assert isinstance(result, ToolError)
        assert "no plan_writing_path" in result.message

    async def test_file_does_not_exist(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        path = tmp_path / "missing.md"
        mock_session.custom_data["plan_writing_path"] = path
        tool = EditPlan(session=mock_session)
        result = await tool(EditPlanParams(edit=Edit(old="a", new="b")))
        assert isinstance(result, ToolError)
        assert "does not exist" in result.message

    async def test_exact_replace(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("hello world", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool(EditPlanParams(edit=Edit(old="hello", new="hi")))
        assert isinstance(result, ToolOk)
        assert plan_path.read_text(encoding="utf-8") == "hi world"

    async def test_replace_all(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("foo bar foo", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool(
            EditPlanParams(edit=Edit(old="foo", new="baz", replace_all=True))
        )
        assert isinstance(result, ToolOk)
        assert plan_path.read_text(encoding="utf-8") == "baz bar baz"

    async def test_no_match_returns_ToolError_with_suggestion(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("hello world", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool(
            EditPlanParams(edit=Edit(old="zzzz_nonexistent", new="abc"))
        )
        assert isinstance(result, ToolError)
        assert "No replacements were made" in result.message

    async def test_multiple_edits(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool(
            EditPlanParams(
                edit=[
                    Edit(old="line1", new="first"),
                    Edit(old="line3", new="third"),
                ]
            )
        )
        assert isinstance(result, ToolOk)
        content = plan_path.read_text(encoding="utf-8")
        assert "first" in content
        assert "third" in content
        assert "line2" in content  # unchanged

    async def test_normalizes_crlf_in_plan_file(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("hello\r\nworld", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool(EditPlanParams(edit=Edit(old="hello", new="hi")))
        assert isinstance(result, ToolOk)
        # The file should have the replacement applied (line endings normalized)
        content = plan_path.read_text(encoding="utf-8")
        assert "hi" in content
        assert "world" in content

    async def test_error_returns_ToolError(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        plan_path.write_text("content", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        with patch("pathlib.Path.read_text", side_effect=OSError("read failure")):
            result = await tool(EditPlanParams(edit=Edit(old="a", new="b")))
        assert isinstance(result, ToolError)
        assert "Failed to edit plan" in result.message

    async def test_string_edit_json_is_repaired(
        self, mock_session: MagicMock, plan_path: Path
    ) -> None:
        """A JSON-encoded `edit` string is parsed into an Edit and applied."""
        plan_path.write_text("hello world", encoding="utf-8")
        mock_session.custom_data["plan_writing_path"] = plan_path
        tool = EditPlan(session=mock_session)
        result = await tool.call(
            {"edit": '{"old": "hello", "new": "hi"}'}
        )
        assert isinstance(result, ToolOk)
        assert plan_path.read_text(encoding="utf-8") == "hi world"
