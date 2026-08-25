"""Tests for rich-format parameter validation and dispatch precedence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaos.path import KaosPath
from kosong.tooling import ToolError

from kimi_cli.tools.file.read import Params as ReadFileParams, ReadFile


@pytest.fixture
def read_tool(tmp_path: Path) -> ReadFile:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = KaosPath(str(tmp_path))
    runtime.additional_dirs = []
    runtime.llm.capabilities = set()
    session = MagicMock()
    session.id = "test"
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    return ReadFile(runtime, session)


class TestRichParamsValidation:
    async def test_pdf_page_conflicts_with_archive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ReadFileParams(path="x.pdf", pdf_page=1, archive_member="a")

    async def test_sql_query_conflicts_with_table(self) -> None:
        with pytest.raises(ValueError, match="sql_query cannot be combined"):
            ReadFileParams(path="x.db", sql_query="SELECT 1", sql_table="t")

    async def test_sql_limit_requires_table(self) -> None:
        with pytest.raises(ValueError, match="sql_limit/sql_offset require"):
            ReadFileParams(path="x.db", sql_limit=10)


class TestRichDispatch:
    async def test_pdf_page_on_non_pdf_errors(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        result = await read_tool(ReadFileParams(path=str(f), pdf_page=1))
        assert result.is_error
        assert "only valid for PDF" in result.message
