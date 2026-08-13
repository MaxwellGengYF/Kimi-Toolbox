"""Tests for ToolCallReason tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kosong.tooling import CallableTool2
from pydantic import BaseModel

from kimi_cli.tools.reason import ToolCallReason


class MockWriteFileParams(BaseModel):
    file_path: str
    content: str
    reason: str = ""


class MockEditFileParams(BaseModel):
    file_path: str
    edit: Any
    reason: str = ""


class MockWriteFileTool(CallableTool2[MockWriteFileParams]):
    name: str = "write"
    description: str = "Mock write"
    params: type[MockWriteFileParams] = MockWriteFileParams

    async def __call__(self, params: MockWriteFileParams) -> Any:
        return None


class MockEditFileTool(CallableTool2[MockEditFileParams]):
    name: str = "edit"
    description: str = "Mock edit"
    params: type[MockEditFileParams] = MockEditFileParams

    async def __call__(self, params: MockEditFileParams) -> Any:
        return None


class _MockWrongParams(BaseModel):
    pass


class MockWrongTool(CallableTool2[_MockWrongParams]):
    name: str = "WrongTool"
    description: str = "Mock wrong"
    params: type[_MockWrongParams] = _MockWrongParams

    async def __call__(self, params: _MockWrongParams) -> Any:
        return None


@pytest.fixture
def tracker() -> ToolCallReason:
    return ToolCallReason()


@pytest.fixture
def write_tool() -> MockWriteFileTool:
    return MockWriteFileTool()


@pytest.fixture
def edit_tool() -> MockEditFileTool:
    return MockEditFileTool()


@pytest.fixture
def wrong_tool() -> MockWrongTool:
    return MockWrongTool()


class TestToolCallReasonAdd:
    """Test add_tool_call_reason method."""

    def test_add_write_file(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        params = MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="hello", reason="create file")
        tracker.add_tool_call_reason(params, write_tool)

        assert len(tracker) == 1
        abs_path = str((tmp_path / "a.py").resolve())
        assert abs_path in tracker._records
        assert tracker._records[abs_path] == ["write"]

    def test_add_edit_file_single(self, tracker: ToolCallReason, edit_tool: MockEditFileTool, tmp_path: Path):
        from pydantic import BaseModel, Field

        class Edit(BaseModel):
            old: str = Field(default="")
            new: str = Field(default="")
            replace_all: bool = Field(default=False)

        params = MockEditFileParams(
            file_path=str(tmp_path / "b.py"),
            edit=Edit(old="foo", new="bar"),
            reason="fix typo",
        )
        tracker.add_tool_call_reason(params, edit_tool)

        abs_path = str((tmp_path / "b.py").resolve())
        assert tracker._records[abs_path] == ["edit"]

    def test_add_edit_file_list(self, tracker: ToolCallReason, edit_tool: MockEditFileTool, tmp_path: Path):
        from pydantic import BaseModel, Field

        class Edit(BaseModel):
            old: str = Field(default="")
            new: str = Field(default="")
            replace_all: bool = Field(default=False)

        params = MockEditFileParams(
            file_path=str(tmp_path / "c.py"),
            edit=[Edit(old="a", new="1"), Edit(old="b", new="2")],
            reason="batch update",
        )
        tracker.add_tool_call_reason(params, edit_tool)

        abs_path = str((tmp_path / "c.py").resolve())
        assert tracker._records[abs_path] == ["edit"]

    def test_add_edit_file_none_edit(self, tracker: ToolCallReason, edit_tool: MockEditFileTool, tmp_path: Path):
        params = MockEditFileParams(file_path=str(tmp_path / "d.py"), edit=None, reason="noop")
        tracker.add_tool_call_reason(params, edit_tool)

        abs_path = str((tmp_path / "d.py").resolve())
        assert tracker._records[abs_path] == ["edit"]

    def test_add_wrong_tool_raises(self, tracker: ToolCallReason, wrong_tool: MockWrongTool, tmp_path: Path):
        params = MockWriteFileParams(file_path=str(tmp_path / "x.py"), content="x")
        with pytest.raises(ValueError, match="Expected write or edit"):
            tracker.add_tool_call_reason(params, wrong_tool)

    def test_add_empty_path_raises(self, tracker: ToolCallReason, write_tool: MockWriteFileTool):
        params = MockWriteFileParams(file_path="", content="x")
        with pytest.raises(ValueError, match="non-empty 'file_path'"):
            tracker.add_tool_call_reason(params, write_tool)

    def test_add_multiple_same_path(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        path = str(tmp_path / "multi.py")
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v1", reason="first"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v2", reason="second"), write_tool)

        abs_path = str((tmp_path / "multi.py").resolve())
        assert tracker._records[abs_path] == ["write", "write"]


class TestToolCallReasonFormattedPrint:
    """Test formatted_print method."""

    def test_formatted_print_no_records(self, tracker: ToolCallReason, tmp_path: Path):
        result = tracker.formatted_print([str(tmp_path / "missing.py")])
        assert "no record" in result

    def test_formatted_print_single_write_file(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        path = str(tmp_path / "a.py")
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="hello world", reason="init"), write_tool)

        result = tracker.formatted_print([path])
        abs_path = str(Path(path).resolve())
        assert f"- {abs_path}" in result
        assert "(write)" in result
        assert "hello world" not in result
        assert "--- old ---" not in result

    def test_formatted_print_single_edit_file(self, tracker: ToolCallReason, edit_tool: MockEditFileTool, tmp_path: Path):
        from pydantic import BaseModel, Field

        class Edit(BaseModel):
            old: str = Field(default="")
            new: str = Field(default="")
            replace_all: bool = Field(default=False)

        path = str(tmp_path / "b.py")
        tracker.add_tool_call_reason(
            MockEditFileParams(file_path=path, edit=Edit(old="old_text", new="new_text"), reason="update"),
            edit_tool,
        )

        result = tracker.formatted_print([path])
        assert "(edit)" in result
        assert "--- old ---" not in result
        assert "old_text" not in result
        assert "new_text" not in result

    def test_formatted_print_multiple_paths(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        path1 = str(tmp_path / "a.py")
        path2 = str(tmp_path / "b.py")
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path1, content="a", reason="ra"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path2, content="b", reason="rb"), write_tool)

        result = tracker.formatted_print([path1, path2])
        assert "(write)" in result
        assert "- " in result

    def test_formatted_print_multiple_records_same_path(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        path = str(tmp_path / "a.py")
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v1", reason="r1"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v2", reason="r2"), write_tool)

        result = tracker.formatted_print([path])
        assert "(write, write)" in result

    def test_formatted_print_returns_string_not_prints(self, tracker: ToolCallReason, tmp_path: Path):
        result = tracker.formatted_print([str(tmp_path / "none.py")])
        assert isinstance(result, str)


class TestToolCallReasonChangedFiles:
    """Test changed_files and to_markdown."""

    def test_changed_files_sorted(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "z.py"), content="z", reason="rz"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a", reason="ra"), write_tool)
        assert tracker.changed_files == sorted(tracker.changed_files)
        assert len(tracker.changed_files) == 2

    def test_to_markdown_empty(self, tracker: ToolCallReason):
        assert tracker.to_markdown() == ""

    def test_to_markdown_content(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a", reason="ra"), write_tool)
        md = tracker.to_markdown()
        assert md.startswith("Changed files:")
        assert "a.py" in md
        assert "(write)" in md

    def test_to_markdown_multiple_records(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        path = str(tmp_path / "a.py")
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v1", reason="r1"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=path, content="v2", reason="r2"), write_tool)
        md = tracker.to_markdown()
        assert "(write, write)" in md


class TestToolCallReasonLifecycle:
    """Test clear, len, bool."""

    def test_len_empty(self, tracker: ToolCallReason):
        assert len(tracker) == 0

    def test_bool_empty(self, tracker: ToolCallReason):
        assert not tracker

    def test_len_and_bool_with_records(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a"), write_tool)
        assert len(tracker) == 1
        assert bool(tracker)

    def test_clear(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a"), write_tool)
        tracker.clear()
        assert len(tracker) == 0
        assert not tracker

    def test_len_multiple_paths_and_records(self, tracker: ToolCallReason, write_tool: MockWriteFileTool, tmp_path: Path):
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "b.py"), content="b"), write_tool)
        tracker.add_tool_call_reason(MockWriteFileParams(file_path=str(tmp_path / "a.py"), content="a2"), write_tool)
        assert len(tracker) == 3
