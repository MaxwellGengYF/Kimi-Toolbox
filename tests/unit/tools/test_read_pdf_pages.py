"""Tests for PDF page screenshot rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pymupdf

from kaos.path import KaosPath
from kosong.tooling import ToolError, ToolOk

from kimi_cli.tools.file.read import Params as ReadFileParams, ReadFile
from kimi_cli.tools.file.read_pdf_pages import render_pdf_page


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 100), "Hello PDF")
    doc.save(str(pdf))
    doc.close()
    return pdf


@pytest.fixture
def read_tool(tmp_path: Path) -> ReadFile:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = KaosPath(str(tmp_path))
    runtime.additional_dirs = []
    runtime.llm.capabilities = {"image_in"}
    session = MagicMock()
    session.id = "test"
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    return ReadFile(runtime, session)


class TestPdfPages:
    async def test_render_pdf_page_no_image_capability(self, sample_pdf: Path) -> None:
        result = render_pdf_page(str(sample_pdf), 1, capabilities=set())
        assert isinstance(result, ToolError)
        assert "image input" in result.message.lower()

    async def test_render_pdf_page_out_of_range(self, sample_pdf: Path) -> None:
        result = render_pdf_page(str(sample_pdf), 5, capabilities={"image_in"})
        assert isinstance(result, ToolError)
        assert "out of range" in result.message.lower()

    @patch("kimi_cli.tools.file.read_pdf_pages.compress_image_for_model")
    async def test_read_pdf_page_returns_media(
        self,
        mock_compress,
        read_tool: ReadFile,
        sample_pdf: Path,
    ) -> None:
        class FakeCompressed:
            changed = False
            width = 612
            height = 792
            final_byte_length = 1024
            mime_type = "image/png"
            data = b"pngdata"

        mock_compress.return_value = FakeCompressed()
        result = await read_tool(ReadFileParams(path=str(sample_pdf), pdf_page=1))
        assert isinstance(result, ToolOk)
        assert isinstance(result.output, list)
        assert "PDF page" in result.message
    @patch("kimi_cli.tools.file.read_pdf_pages.compress_image_for_model")
    async def test_read_pdf_page_dpi_fallback(
        self,
        mock_compress,
        read_tool: ReadFile,
        sample_pdf: Path,
    ) -> None:
        class FakeCompressed:
            changed = False
            width = 612
            height = 792
            final_byte_length = 1024
            mime_type = "image/png"
            data = b"pngdata"

        # First call over budget, second succeeds.
        mock_compress.side_effect = [
            type("C", (), {"changed": False, "width": 612, "height": 792, "final_byte_length": 999_999, "mime_type": "image/png", "data": b"x"})(),
            FakeCompressed(),
        ]
        with patch("kimi_cli.tools.file.read_pdf_pages.mipmap_downsample") as mock_mipmap:
            class FakeMipmap:
                changed = True
                width = 612
                height = 792
                final_byte_length = 1024
                mime_type = "image/png"
                data = b"pngdata"
            mock_mipmap.return_value = FakeMipmap()
            result = await read_tool(ReadFileParams(path=str(sample_pdf), pdf_page=1))
        assert isinstance(result, ToolOk)
        assert mock_compress.call_count >= 1
