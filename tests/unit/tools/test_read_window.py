"""Tests for ReadFile char-window truncation notices.

The ``char_offset``/``max_char`` window is applied after the line/byte
budgets are rendered, so without a notice the message could claim "End of
file reached." while the output was silently cut. These tests pin the
explicit window notice (head / tail / middle) that lets the agent know a
read was partial.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaos.path import KaosPath
from kosong.tooling import ToolOk

from kimi_cli.tools.file.read import ReadFile, Params as ReadFileParams


@pytest.fixture
def read_tool(tmp_path: Path) -> ReadFile:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = KaosPath(str(tmp_path))
    runtime.additional_dirs = []
    session = MagicMock()
    session.id = "test"
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    return ReadFile(runtime, session)


class TestReadCharWindowNotice:
    async def test_head_window_reports_truncation(
        self, read_tool: ReadFile, tmp_path: Path
    ) -> None:
        f = tmp_path / "big.txt"
        f.write_text("x" * 1000, encoding="utf-8")
        result = await read_tool(ReadFileParams(path=str(f), max_char=100))
        assert isinstance(result, ToolOk)
        assert len(result.output) == 100
        assert "head chars 0..100" in result.message
        assert "content after is hidden" in result.message

    async def test_full_read_has_no_notice(
        self, read_tool: ReadFile, tmp_path: Path
    ) -> None:
        f = tmp_path / "small.txt"
        f.write_text("hello\n", encoding="utf-8")
        result = await read_tool(ReadFileParams(path=str(f), max_char=16000))
        assert isinstance(result, ToolOk)
        assert "NOTE:" not in result.message
        assert "End of file reached." in result.message

    async def test_char_offset_tail_reports_hidden_prefix(
        self, read_tool: ReadFile, tmp_path: Path
    ) -> None:
        f = tmp_path / "big2.txt"
        f.write_text("".join(f"line {i}\n" for i in range(50)), encoding="utf-8")
        result = await read_tool(ReadFileParams(path=str(f), char_offset=200, max_char=16000))
        assert isinstance(result, ToolOk)
        assert "tail chars 200.." in result.message
        assert "content before is hidden" in result.message

    async def test_middle_window_reports_both_hidden(
        self, read_tool: ReadFile, tmp_path: Path
    ) -> None:
        f = tmp_path / "big3.txt"
        f.write_text("x" * 1000, encoding="utf-8")
        result = await read_tool(ReadFileParams(path=str(f), char_offset=100, max_char=100))
        assert isinstance(result, ToolOk)
        assert "middle chars 100..200" in result.message
        assert "content before and after is hidden" in result.message
