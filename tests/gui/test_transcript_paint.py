from __future__ import annotations

from itertools import count

from PySide6.QtCore import QRect

from kimix_gui.design import DARK
from kimix_gui.qt.paint import layout_record
from kimix_gui.qt.transcript import Transcript
from kimix_gui.transcript_data import (
    ActivityEntry,
    ActivityField,
    NoticeEntry,
    StartEntry,
    TextBlock,
    TextEntry,
    ToolActivity,
    ToolCallContent,
    ToolIdentity,
    ToolResultContent,
    entry_copy_text,
    literal,
)
from kimix_gui.transcript_layout import BodySection, bar_color_name, cell_len, record_label

_KEYS = count()


def _text(kind: str, text: str, *, markdown: bool = False) -> TextEntry:
    return TextEntry(
        key=f"text:{next(_KEYS)}",
        kind=kind,  # type: ignore[arg-type]
        blocks=(TextBlock(literal(text), format="markdown" if markdown else "plain"),),
    )


def _notice(kind: str, text: str) -> NoticeEntry:
    return NoticeEntry(
        key=f"notice:{next(_KEYS)}",
        kind=kind,  # type: ignore[arg-type]
        blocks=(TextBlock(literal(text)),),
    )


def _activity(
    name: str,
    summary: tuple[str, ...],
    *,
    fields: tuple[ActivityField, ...] = (),
    status: str | None = None,
    result_summary: tuple[str, ...] = (),
    result_blocks: tuple[TextBlock, ...] = (),
) -> ActivityEntry:
    family = {
        "read": "read",
        "grep": "grep",
        "todo_write": "todo",
        "bash": "shell",
        "python": "python",
        "write": "write",
    }.get(name, "generic")
    result = (
        ToolResultContent(
            status=status,  # type: ignore[arg-type]
            summary_parts=tuple(literal(part) for part in result_summary),
            blocks=result_blocks,
        )
        if status is not None
        else None
    )
    return ActivityEntry(
        key=f"activity:{next(_KEYS)}",
        activity=ToolActivity(
            call_id=f"call:{next(_KEYS)}",
            identity=ToolIdentity(name, family),
            call=ToolCallContent(
                summary_parts=tuple(literal(part) for part in summary), fields=fields
            ),
            result=result,
        ),
        complete=result is not None,
    )


def test_user_keeps_cyan_label_and_raw_markdown() -> None:
    entry = _text("user", "please use **bold** here")
    layout = layout_record(entry, width=48)
    assert layout.label == "You"
    assert "**bold**" in layout.body
    assert layout.summary == ""
    assert layout.bar_color == "cyan"
    assert layout.body_sections[0].format == "plain"


def test_assistant_markdown_is_built_by_qtextdocument_fragment(qtbot) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(400, 200)
    entry = _text("assistant", "## Title\n\nUse **bold** and `code`.", markdown=True)
    transcript.apply_mutation(StartEntry(entry))
    layout = layout_record(entry, width=48)
    index = transcript.model().index(0, 0)
    document = transcript._delegate.document_for(QRect(0, 0, 400, 0), index)
    assert document is not None
    rendered = document.toPlainText()
    assert layout.label == "AI"
    assert layout.body_sections[0].format == "markdown"
    assert "Title" in rendered and "bold" in rendered and "**bold**" not in rendered
    assert layout.bar_color == "green"


def test_markdown_links_use_the_high_contrast_theme_color(qtbot) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    entry = _text("assistant", "[settings](/src/kimix_gui/qt/settings_list.py)", markdown=True)
    transcript.apply_mutation(StartEntry(entry))
    record = transcript.records[0]
    layout = layout_record(record.entry, width=320)
    document = transcript.bodies.document(record, 320, layout)
    link = document.find("settings")
    assert link.charFormat().isAnchor()
    assert link.charFormat().foreground().color().name() == DARK.palette.link


def test_thinking_uses_muted_italic() -> None:
    layout = layout_record(_text("thinking", "considering options"), width=48, expanded=True)
    assert layout.header == "Think"
    assert "considering options" in layout.body
    assert layout.italic_body is True
    assert layout.body_sections[0].tone == "muted"
    assert layout.bar_color == "muted"


def test_notice_records_are_compacted_from_semantic_blocks() -> None:
    layout = layout_record(
        NoticeEntry(
            key="notice:structured",
            kind="system",
            blocks=(
                TextBlock(literal("Ready")),
                TextBlock(literal("a very detailed result")),
            ),
        ),
        width=48,
    )
    assert layout.compact is True
    assert layout.header == "System Ready"
    assert "detailed" not in layout.header
    assert layout.body == ""


def test_tool_summary_never_reads_the_code_payload() -> None:
    entry = _activity(
        "write",
        ("demo.py",),
        fields=(
            ActivityField("path", literal("demo.py"), role="primary", hint="path"),
            ActivityField(
                "content", literal("print('hello')\nprint('world')"), role="detail", hint="code"
            ),
        ),
    )
    layout = layout_record(entry, width=48)
    assert layout.header == "Write demo.py"
    assert "print" not in layout.header


def test_expanded_tool_call_uses_explicit_field_sections() -> None:
    entry = _activity(
        "write",
        ("demo.py",),
        fields=(
            ActivityField("path", literal("demo.py"), role="primary", hint="path"),
            ActivityField("content", literal("line one\nline two"), role="detail", hint="code"),
        ),
    )
    layout = layout_record(entry, width=64, expanded=True)
    assert layout.header == "Write demo.py"
    assert "content:\nline one\nline two" in layout.body
    assert any(section.format == "code" for section in layout.body_sections)


def test_known_tools_use_identity_for_titles_and_colors() -> None:
    cases = (
        ("grep", "Grep", "cyan"),
        ("read", "Read", "cyan"),
        ("todo_write", "Todo", "yellow"),
        ("bash", "Bash", "green"),
    )
    for name, title, color in cases:
        entry = _activity(name, ("target",))
        layout = layout_record(entry, width=48)
        assert layout.header.startswith(title)
        assert layout.bar_color == color
        assert record_label(entry) == title


def test_uncategorized_and_notice_rows_use_muted_bars() -> None:
    for entry in (
        _activity("custom_mcp", ("query",)),
        _text("thinking", "hmm"),
        _notice("system", "ready"),
        _notice("approval", "allow this"),
    ):
        assert bar_color_name(entry) == "muted"


def test_compacted_records_use_ascii_ellipsis_and_display_cells() -> None:
    layout = layout_record(_notice("system", "会话" * 60), width=48)
    assert layout.compact is True
    assert layout.header.endswith("...")
    assert cell_len(layout.header) <= 48
    assert not {"▸", "▾", "⧉"}.intersection(layout.header)


def test_pending_activity_exposes_state_without_a_text_glyph() -> None:
    layout = layout_record(_activity("read", ("a.py",)), width=64)
    assert layout.status == "pending"
    assert layout.header == "Read a.py"
    assert not {"▸", "▾", "◌", "✓", "✗"}.intersection(layout.header)


def test_finished_activity_folds_result_summary_into_header() -> None:
    layout = layout_record(
        _activity(
            "read",
            ("a.py",),
            status="ok",
            result_summary=("12 lines",),
            result_blocks=(TextBlock(literal("file contents")),),
        ),
        width=80,
    )
    assert layout.header == "Read a.py · 12 lines"
    assert "file contents" not in layout.header


def test_failed_activity_uses_red_bar_from_structured_status() -> None:
    entry = _activity(
        "read",
        ("missing.py",),
        status="error",
        result_summary=("failed",),
        result_blocks=(TextBlock(literal("No such file")),),
    )
    layout = layout_record(entry, width=80)
    assert layout.header == "Read missing.py · failed"
    assert layout.status == "error"
    assert layout.bar_color == "red"
    assert bar_color_name(entry) == "red"


def test_expanded_read_uses_context_section_then_primary_output() -> None:
    entry = _activity(
        "read",
        ("a.py",),
        fields=(
            ActivityField("path", literal("a.py"), role="primary", hint="path"),
            ActivityField("offset", literal("1"), role="secondary", hint="count"),
        ),
        status="ok",
        result_summary=("120 lines",),
        result_blocks=(TextBlock(literal("file contents here\nsecond line")),),
    )
    layout = layout_record(entry, width=94, expanded=True)
    assert layout.body == "a.py · offset 1\n\nfile contents here\nsecond line"
    assert layout.body_sections == (
        BodySection("a.py · offset 1", tone="context"),
        BodySection("file contents here\nsecond line", spacing="paragraph"),
    )
    assert "Result" not in layout.body and "└" not in layout.body


def test_expanded_shell_uses_context_then_error_output() -> None:
    entry = _activity(
        "bash",
        ("pytest -q", "/tmp/work"),
        fields=(
            ActivityField("command", literal("pytest -q"), role="primary", hint="command"),
            ActivityField("cwd", literal("/tmp/work"), role="secondary"),
        ),
        status="error",
        result_summary=("2 tests failed",),
        result_blocks=(TextBlock(literal("AssertionError")),),
    )
    layout = layout_record(entry, width=94, expanded=True)
    assert layout.body == "pytest -q · cwd /tmp/work\n\nAssertionError"
    assert layout.body_sections[0].tone == "context"


def test_expanded_python_preserves_multiline_code_without_result_label() -> None:
    entry = _activity(
        "python",
        ("def run():",),
        fields=(
            ActivityField("code", literal("def run():\n    return 3"), role="primary", hint="code"),
        ),
        status="ok",
        result_summary=("1 line",),
        result_blocks=(TextBlock(literal("3")),),
    )
    layout = layout_record(entry, width=94, expanded=True)
    assert layout.body == "code:\ndef run():\n    return 3\n\n3"
    assert layout.body_sections[0].format == "code"
    assert "Result" not in layout.body


def test_copy_text_traverses_the_activity_ast() -> None:
    entry = _activity(
        "read",
        ("a.py",),
        fields=(ActivityField("path", literal("a.py"), role="primary", hint="path"),),
        status="ok",
        result_summary=("120 lines",),
        result_blocks=(TextBlock(literal("file contents")),),
    )
    copied = entry_copy_text(entry)
    assert copied == "read  a.py\npath: a.py\n\nread  120 lines\nfile contents"
    assert "└" not in copied
