"""The Qt half of the status line: figures in, one localized phrase out.

``rendering.status_values`` is covered in ``tests/gui/test_rendering.py`` and never
produces English, so the wording is only asserted here. No ``QApplication`` is needed:
``QCoreApplication.translate`` returns the msgid when no catalog is installed, which
is the English text.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from kimix_gui.qt.i18n import catalog_file, set_active_language
from kimix_gui.qt.status_line import format_status_line
from kimix_gui.rendering import McpCounts, McpServer, StatusValues, TokenCounts

needs_catalog = pytest.mark.skipif(
    not catalog_file("zh_CN").is_file(),
    reason=f"{catalog_file('zh_CN').name} not built; run: "
    "uv run python scripts/gui/build_translations.py",
)


@pytest.fixture
def chinese(qtbot) -> Iterator[None]:
    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "zh_CN")
    yield
    set_active_language(qt_app, "en")


def test_nothing_to_report_renders_the_idle_word() -> None:
    assert format_status_line(StatusValues()) == "ready"
    # ``None`` is the "no status at all" case ``KimixBridge`` passes on startup.
    assert format_status_line(None) == "ready"


def test_context_and_usage_join_with_the_glossary_separator() -> None:
    line = format_status_line(
        StatusValues(context_tokens=1_000, max_context_tokens=10_000, usage_percent=10.0)
    )

    assert line == "context 1,000/10,000 · 10.0%"


def test_context_without_a_limit_uses_its_own_sentence() -> None:
    assert format_status_line(StatusValues(context_tokens=1_000)) == "context 1,000"


def test_tokens_and_mcp_details_render_in_full() -> None:
    line = format_status_line(
        StatusValues(
            context_tokens=2_000,
            max_context_tokens=20_000,
            usage_percent=10.0,
            tokens=TokenCounts(
                total_input=950, new_input=100, cache_read=800, cache_write=50, output=75
            ),
            mcp=McpCounts(
                connected=1,
                total=2,
                tools=8,
                servers=(McpServer(name="github", state="connected"),),
            ),
        )
    )

    assert "context 2,000/20,000" in line
    assert "tokens in 950 (new 100, cache read 800, cache write 50)" in line
    assert "out 75" in line
    assert "MCP 1/2 ready, 8 tools [github:connected]" in line


def test_loading_mcp_uses_a_whole_sentence_not_a_spliced_word() -> None:
    loading = format_status_line(StatusValues(mcp=McpCounts(connected=0, total=2, loading=True)))
    ready = format_status_line(StatusValues(mcp=McpCounts(connected=2, total=2)))

    assert loading == "MCP 0/2 loading, 0 tools"
    assert ready == "MCP 2/2 ready, 0 tools"


def test_the_run_mode_flags_append_their_own_fields() -> None:
    line = format_status_line(StatusValues(yolo_enabled=True, afk_enabled=True))

    assert line == "YOLO enabled · AFK enabled"


@needs_catalog
def test_the_status_line_renders_in_chinese(chinese) -> None:
    line = format_status_line(
        StatusValues(
            context_tokens=2_000,
            max_context_tokens=20_000,
            usage_percent=10.0,
            tokens=TokenCounts(
                total_input=950, new_input=100, cache_read=800, cache_write=50, output=75
            ),
            mcp=McpCounts(connected=1, total=2, tools=8),
        )
    )

    assert format_status_line(StatusValues()) == "就绪"
    assert "上下文 2,000/20,000" in line
    assert "输入 950 tokens" in line
    assert "输出 75" in line
    assert "MCP 1/2 就绪，8 个工具" in line
