"""Markit-style conversions for the ``read`` tool.

Converts supported documents to markdown-flavored text and converts
markdown/HTML files to plain text.
"""

from __future__ import annotations

from pathlib import Path

import regex as re

from .read_extract import (
    EXTRACTABLE_EXTENSIONS,
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)

__all__ = [
    "DOCUMENT_MARKDOWN_EXTENSIONS",
    "extract_document_markdown",
    "markdown_to_text",
    "html_to_text",
    "is_markdown_document",
    "is_html_document",
]

# Matches the existing extractors' section headers like "# ── Sheet: name ──".
_SECTION_HEADER_RE = re.compile(r"^# ── (.+?) ──$")

DOCUMENT_MARKDOWN_EXTENSIONS = frozenset(EXTRACTABLE_EXTENSIONS)


def _tab_row_to_markdown(row: str) -> str:
    cells = [cell.strip() for cell in row.split("\t")]
    return "| " + " | ".join(cells) + " |"


def _is_table_row(line: str) -> bool:
    return "\t" in line and not line.startswith("#")


def _convert_extracted_to_markdown(text: str) -> str:
    """Convert the legacy tab-separated extractor output into markdown."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _SECTION_HEADER_RE.match(line)
        if m:
            out.append(f"## {m.group(1)}")
            i += 1
            continue

        if _is_table_row(line):
            rows: list[str] = []
            while i < len(lines) and _is_table_row(lines[i]):
                rows.append(lines[i])
                i += 1
            if rows:
                cells = [c.strip() for c in rows[0].split("\t")]
                out.append("| " + " | ".join(cells) + " |")
                out.append("|" + "|".join(" --- " for _ in cells) + "|")
                for r in rows[1:]:
                    out.append(_tab_row_to_markdown(r))
                out.append("")
            continue

        out.append(line)
        i += 1

    return "\n".join(out).rstrip("\n") + "\n"


def extract_document_markdown(path: str) -> str:
    """Extract a document as markdown-flavored text."""
    if not is_extractable_document(path):
        raise ExtractionError(f"Unsupported document type: {path!r}")
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return _convert_docx_to_markdown(path)
    if ext == ".ipynb":
        return _convert_notebook_to_markdown(path)
    extracted = extract_document_text(path)
    return _convert_extracted_to_markdown(extracted)


def _convert_docx_to_markdown(path: str) -> str:
    """Extract a .docx as markdown with heading and table markers."""
    try:
        from docx import Document
        from docx.text.paragraph import Paragraph
        from docx.table import Table
    except ImportError as exc:
        raise ExtractionError(f"python-docx is required for .docx extraction: {exc}") from exc
    try:
        document = Document(path)
    except Exception as exc:
        raise ExtractionError(f"Not a valid DOCX: {exc}") from exc

    out: list[str] = []

    def _table_rows(table: Table) -> list[str]:
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\t", " ") for cell in row.cells]
            rows.append("\t".join(cells))
        return rows

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                out.append("")
                continue
            style_name = ""
            try:
                if item.style and item.style.name:
                    style_name = item.style.name.lower()
            except Exception:
                pass
            if style_name.startswith("heading "):
                try:
                    level = min(int(style_name.split()[1]) + 1, 6)
                except (IndexError, ValueError):
                    level = 2
                out.append(f"{'#' * level} {text}")
            elif style_name == "title":
                out.append(f"## {text}")
            else:
                out.append(text)
        elif isinstance(item, Table):
            rows = _table_rows(item)
            if rows:
                cells = [c.strip() for c in rows[0].split("\t")]
                out.append("| " + " | ".join(cells) + " |")
                out.append("|" + "|".join(" --- " for _ in cells) + "|")
                for r in rows[1:]:
                    out.append(_tab_row_to_markdown(r))
                out.append("")

    if not any(line.strip() for line in out):
        raise ExtractionError("DOCX contains no extractable text")

    # Collapse consecutive blank lines.
    cleaned: list[str] = []
    prev_blank = False
    for line in out:
        if line == "":
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).rstrip("\n") + "\n"


def _convert_notebook_to_markdown(path: str) -> str:
    """Extract a Jupyter notebook as markdown with fenced code cells."""
    try:
        import nbformat
    except ImportError as exc:
        raise ExtractionError(f"nbformat is required for .ipynb extraction: {exc}") from exc
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    out: list[str] = []
    counts: dict[str, int] = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown cell", "code": "Code cell", "raw": "Raw cell"}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        out.append(f"## {labels[typ]} {counts[typ]}")
        source = _source_text(cell.get("source", "")).rstrip("\n")
        if typ == "code":
            out.append("```python")
            out.append(source)
            out.append("```")
        else:
            out.append(source)
        out.append("")
    if not out:
        raise ExtractionError("Notebook contains no readable cells")
    return "\n".join(out).rstrip("\n") + "\n"


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def is_markdown_document(path: str) -> bool:
    return Path(path).suffix.lower() in {".md", ".markdown"}


def is_html_document(path: str) -> bool:
    return Path(path).suffix.lower() in {".html", ".htm"}


def markdown_to_text(md: str) -> str:
    """Convert markdown to compact plain text."""
    # Fenced code blocks -> placeholder.
    md = re.sub(
        r"```[\s\S]*?```",
        lambda m: f"[code block: {m.group(0).count(chr(10))} lines]",
        md,
    )
    # Inline code -> placeholder so later transformations (emphasis, links,
    # headings) never rewrite code content. Restored at the end.
    inline_code: list[str] = []

    def _capture_inline_code(m: "re.Match[str]") -> str:
        inline_code.append(m.group(1))
        return f"\x00CODE{len(inline_code) - 1}\x00"

    md = re.sub(r"`([^`]+)`", _capture_inline_code, md)

    # Bold/italic markers. The underscore form is word-bounded: CommonMark
    # disables intraword emphasis for `_`, so identifiers like `foo_bar` or
    # module paths (`kimi_cli/soul/...`) are preserved verbatim instead of
    # being treated as italic pairs.
    md = re.sub(r"\*\*([^*]+)\*\*", r"\1", md)
    md = re.sub(r"\*([^*]+)\*", r"\1", md)
    md = re.sub(r"__([^_]+)__", r"\1", md)
    md = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", md)
    # Links.
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", md)
    # Images.
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \2]", md)
    # Heading markers.
    md = re.sub(r"^#+\s*(.+)$", r"\1", md, flags=re.MULTILINE)
    # Horizontal rules.
    md = re.sub(r"^---+$", "", md, flags=re.MULTILINE)
    # Restore inline code.
    for i, code in enumerate(inline_code):
        md = md.replace(f"\x00CODE{i}\x00", code)
    # Collapse blank runs.
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text via markdownify then markdown_to_text."""
    try:
        import markdownify
    except ImportError as exc:
        raise ExtractionError(f"markdownify is required for HTML conversion: {exc}") from exc
    try:
        md = markdownify.markdownify(html, heading_style="ATX")
    except Exception as exc:
        raise ExtractionError(f"Failed to convert HTML: {exc}") from exc
    return markdown_to_text(md)
