"""Tests for document extraction in ReadFile (.ipynb / .docx / .xlsx)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import orjson
import pytest
from kaos.path import KaosPath

from kimi_cli.tools.file import read as read_module
from kimi_cli.tools.file.read import Params, ReadFile
from kimi_cli.tools.file.read_extract import (
    EXTRACTABLE_EXTENSIONS,
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_zip(path: KaosPath, files: dict[str, str | bytes]) -> None:
    """Write a minimal zip archive from {entry_name: content}."""
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)


def _write_bytes(path: KaosPath, data: bytes) -> None:
    """Synchronous write for module-level (non-async) tests."""
    Path(str(path)).write_bytes(data)


NB_JSON = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "Some *text*."]},
        {"cell_type": "code", "source": "print('hello')"},
        {"cell_type": "raw", "source": "raw content"},
    ]
}

DOCX_MINIMAL: dict[str, str] = {
    "word/document.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_NS_W}">'
        "<w:body>"
        "<w:p><w:r><w:t>Hello</w:t><w:tab/><w:t>World</w:t><w:br/></w:r></w:p>"
        "<w:p><w:r><w:t>Second para</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
}

XLSX_MINIMAL: dict[str, str] = {
    "xl/workbook.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{_NS_S}" xmlns:r="{_NS_REL}">'
        "<sheets>"
        '<sheet name="Data" sheetId="1" r:id="rId1"/>'
        '<sheet name="HiddenSheet" sheetId="2" state="hidden" r:id="rId2"/>'
        "</sheets>"
        "</workbook>"
    ),
    "xl/_rels/workbook.xml.rels": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_NS_PKG_REL}">'
        '<Relationship Id="rId1" '
        f'Type="{_NS_REL}/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        f'Type="{_NS_REL}/worksheet" Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    ),
    "xl/worksheets/sheet1.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{_NS_S}">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>1</v></c></row>'
        "</sheetData>"
        "</worksheet>"
    ),
    "xl/worksheets/sheet2.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{_NS_S}">'
        "<sheetData></sheetData>"
        "</worksheet>"
    ),
    "xl/sharedStrings.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="{_NS_S}">'
        "<si><t>Name</t></si>"
        "<si><t>Alice</t></si>"
        "</sst>"
    ),
}


# ── module-level extraction helpers ─────────────────────────────────────────


def test_is_extractable_document():
    assert set(EXTRACTABLE_EXTENSIONS) == {".ipynb", ".docx", ".xlsx"}
    assert is_extractable_document("notes.ipynb")
    assert is_extractable_document("doc.DOCX")
    assert is_extractable_document("a/b/c.xlsx")
    assert not is_extractable_document("notes.txt")
    assert not is_extractable_document("noext")
    assert not is_extractable_document("archive.zip")


def test_extract_notebook_markdown_and_code_cells(temp_work_dir: KaosPath):
    nb = temp_work_dir / "nb.ipynb"
    _write_bytes(nb, orjson.dumps(NB_JSON))

    text = extract_document_text(str(nb))
    assert text == (
        "# ── Markdown cell 1 ──\n"
        "# Title\n"
        "Some *text*.\n"
        "\n"
        "# ── Code cell 1 ──\n"
        "print('hello')\n"
        "\n"
        "# ── Raw cell ──\n"
        "raw content\n"
    )


def test_extract_notebook_legacy_worksheets(temp_work_dir: KaosPath):
    """Legacy `worksheets`-based notebooks are supported."""
    nb = temp_work_dir / "legacy.ipynb"
    _write_bytes(
        nb,
        orjson.dumps(
            {
                "worksheets": [
                    {"cells": [{"cell_type": "code", "source": "x = 1"}]},
                ]
            }
        ),
    )
    text = extract_document_text(str(nb))
    assert text == "# ── Code cell 1 ──\nx = 1\n"


def test_extract_notebook_no_cells_raises(temp_work_dir: KaosPath):
    nb = temp_work_dir / "empty.ipynb"
    _write_bytes(nb, orjson.dumps({"cells": []}))
    with pytest.raises(ExtractionError):
        extract_document_text(str(nb))


def test_extract_docx_paragraphs_tabs_breaks(temp_work_dir: KaosPath):
    docx = temp_work_dir / "doc.docx"
    _write_zip(docx, DOCX_MINIMAL)

    text = extract_document_text(str(docx))
    assert text == "Hello\tWorld\n\nSecond para\n"


def test_extract_xlsx_sheets_and_shared_strings(temp_work_dir: KaosPath):
    xlsx = temp_work_dir / "book.xlsx"
    _write_zip(xlsx, XLSX_MINIMAL)

    text = extract_document_text(str(xlsx))
    assert text == "# ── Sheet: Data ──\nName\t42\nAlice\n"


def test_extract_bad_docx_raises(temp_work_dir: KaosPath):
    bad = temp_work_dir / "bad.docx"
    _write_bytes(bad, b"this is not a zip file at all")
    with pytest.raises(ExtractionError):
        extract_document_text(str(bad))


# ── tool-level tests ────────────────────────────────────────────────────────


async def test_read_ipynb_via_tool(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    nb = temp_work_dir / "nb.ipynb"
    await nb.write_bytes(orjson.dumps(NB_JSON))
    display_path = str(nb).replace("\\", "/")

    result = await read_file_tool(Params(path=str(nb)))
    assert not result.is_error
    assert "# ── Markdown cell 1 ──" in result.output
    assert "# ── Code cell 1 ──" in result.output
    assert "print('hello')" in result.output
    assert "raw content" in result.output
    assert result.message.startswith(
        "9 lines read from file starting from line 1. Total lines in file: 9. End of file reached."
    )
    assert result.message.endswith(f" Path: {display_path} (extracted from .ipynb document)")


async def test_read_docx_via_tool(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    docx = temp_work_dir / "doc.docx"
    _write_zip(docx, DOCX_MINIMAL)
    display_path = str(docx).replace("\\", "/")

    result = await read_file_tool(Params(path=str(docx)))
    assert not result.is_error
    assert "Hello\tWorld" in result.output
    assert "Second para" in result.output
    assert result.message.startswith(
        "3 lines read from file starting from line 1. Total lines in file: 3. End of file reached."
    )
    assert result.message.endswith(f" Path: {display_path} (extracted from .docx document)")


async def test_read_xlsx_via_tool(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    xlsx = temp_work_dir / "book.xlsx"
    _write_zip(xlsx, XLSX_MINIMAL)
    display_path = str(xlsx).replace("\\", "/")

    result = await read_file_tool(Params(path=str(xlsx)))
    assert not result.is_error
    assert "# ── Sheet: Data ──" in result.output
    assert "Name\t42" in result.output
    assert "Alice" in result.output
    # Hidden sheet is skipped.
    assert "HiddenSheet" not in result.output
    assert result.message.endswith(f" Path: {display_path} (extracted from .xlsx document)")


async def test_read_malformed_docx_bad_zip(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    bad = temp_work_dir / "bad.docx"
    await bad.write_bytes(b"\x00\x01\x02 this is not a zip")

    result = await read_file_tool(Params(path=str(bad)))
    assert result.is_error
    assert "could not be extracted" in result.message
    assert result.brief == "Document extraction failed"


async def test_read_docx_missing_document_xml(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    docx = temp_work_dir / "nodoc.docx"
    _write_zip(docx, {"word/styles.xml": "<styles/>"})

    result = await read_file_tool(Params(path=str(docx)))
    assert result.is_error
    assert "could not be extracted" in result.message
    assert "word/document.xml" in result.message
    assert result.brief == "Document extraction failed"


async def test_read_ipynb_no_cells(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    nb = temp_work_dir / "empty.ipynb"
    await nb.write_bytes(orjson.dumps({"cells": []}))

    result = await read_file_tool(Params(path=str(nb)))
    assert result.is_error
    assert "could not be extracted" in result.message
    assert result.brief == "Document extraction failed"


async def test_read_extract_size_guard(
    read_file_tool: ReadFile, temp_work_dir: KaosPath, monkeypatch
):
    """Documents over MAX_EXTRACT_BYTES are refused without extraction."""
    docx = temp_work_dir / "big.docx"
    _write_zip(docx, DOCX_MINIMAL)
    monkeypatch.setattr(read_module, "MAX_EXTRACT_BYTES", 4)

    result = await read_file_tool(Params(path=str(docx)))
    assert result.is_error
    assert "larger than" in result.message
    assert "cannot be extracted" in result.message
    assert result.brief == "File not readable"


async def test_read_extract_pagination(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """char_offset/max_char pagination applies to extracted text."""
    nb = temp_work_dir / "nb.ipynb"
    await nb.write_bytes(orjson.dumps(NB_JSON))

    full = await read_file_tool(Params(path=str(nb), char_offset=0, max_char=200000))
    p1 = await read_file_tool(Params(path=str(nb), char_offset=0, max_char=30))
    p2 = await read_file_tool(Params(path=str(nb), char_offset=30, max_char=30))

    assert not full.is_error and not p1.is_error and not p2.is_error
    assert len(p1.output) <= 30
    assert len(p2.output) <= 30
    assert p1.output == full.output[0:30]
    assert p2.output == full.output[30:60]
    assert p1.output != p2.output
