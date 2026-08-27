"""The word catalog that lets pure-layer labels stay English msgids.

``qt/labels.py`` is the only place that knows those words are translatable, so the
danger is a new pure-layer label that nobody adds to the catalog: it would render
English forever and nothing would fail. These tests close that gap by comparing the
catalog against the pure-layer tables directly.

No ``QApplication`` is required: ``QCoreApplication.translate`` returns the msgid when
no catalog is installed, and the msgid is the English text.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from kimix_gui.qt.i18n import catalog_file, set_active_language
from kimix_gui.qt.labels import (
    RECORD_LABELS,
    TOOL_FAMILY_LABELS,
    UNTITLED_SESSION,
    translate_label,
    translate_no_details,
    translate_session_title,
)
from kimix_gui.qt.paint import layout_record
from kimix_gui.session_index import UNTITLED_TITLE
from kimix_gui.tool_display import _FAMILY_LABEL
from kimix_gui.transcript_layout import LABELS, NO_DETAILS

from .transcript_helpers import activity_entry, text_entry

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


def test_every_record_label_is_in_the_catalog() -> None:
    assert set(LABELS.values()) == set(RECORD_LABELS)


def test_every_tool_family_label_is_in_the_catalog() -> None:
    assert set(_FAMILY_LABEL.values()) == set(TOOL_FAMILY_LABELS)


def test_the_untitled_placeholder_is_single_sourced() -> None:
    assert UNTITLED_SESSION == UNTITLED_TITLE


def test_unknown_words_pass_straight_through() -> None:
    """Labels derived from a wire tool name are not copy and must survive verbatim."""

    for word in ("Bash", "Pwsh", "Custom Mcp", "", "Δ", "read_media"):
        assert translate_label(word) == word


def test_english_lookups_return_the_msgid() -> None:
    assert translate_label("You") == "You"
    assert translate_label("Read") == "Read"
    assert translate_session_title("Untitled") == "Untitled"
    assert translate_session_title("Fix login") == "Fix login"


@needs_catalog
def test_the_catalog_translates_at_the_rendering_boundary(chinese) -> None:
    """The pure layer still returns English; ``layout_record`` is where it changes."""

    from kimix_gui.transcript_layout import record_label

    assert record_label(text_entry("user", "hello")) == "You"
    assert layout_record(text_entry("user", "hello"), width=48).label == "你"
    read = activity_entry("read", summary="a.py")
    assert layout_record(read, width=64).label == "读取"
    assert layout_record(read, width=64).header == "读取 a.py"
    # An unknown tool keeps its wire-derived title even in Chinese.
    assert layout_record(activity_entry("custom_mcp", summary="ping"), width=64).label == (
        "Custom Mcp"
    )


#: Every catalog word and its Chinese rendering, spelled out so that "the ``.ts`` was
#: never translated" and "this word is deliberately kept in English" cannot be
#: confused. The identity entries are tool and brand identities: ``grep`` / ``glob``
#: are command names a reader matches against the wire tool name, and ``AI`` /
#: ``Python`` are used as-is in Chinese; see ``src/kimix_gui/translations/GLOSSARY.md``.
_EXPECTED_CHINESE = {
    "You": "你",
    "AI": "AI",
    "Think": "思考",
    "Tool": "工具",
    "Approval": "批准",
    "System": "系统",
    "Error": "错误",
    "Read": "读取",
    "Grep": "Grep",
    "Glob": "Glob",
    "Write": "写入",
    "Edit": "编辑",
    "Python": "Python",
    "Todo": "待办",
    "Search": "搜索",
    "Fetch": "抓取",
    "Agent": "智能体",
}


@needs_catalog
def test_every_catalog_word_has_its_agreed_chinese_rendering(chinese) -> None:
    """A word left out of the ``.ts`` silently stays English, so pin the whole table."""

    assert set(_EXPECTED_CHINESE) == set(RECORD_LABELS) | set(TOOL_FAMILY_LABELS)
    assert {word: translate_label(word) for word in _EXPECTED_CHINESE} == _EXPECTED_CHINESE


@needs_catalog
def test_the_untitled_placeholder_renders_in_chinese(chinese) -> None:
    assert translate_session_title(UNTITLED_TITLE) == "无标题"


@needs_catalog
def test_the_no_details_placeholder_reaches_a_row_in_chinese(chinese) -> None:
    """A row with nothing but a tool name is where the placeholder shows up.

    It used to be pinned to English by ``qt/paint.py`` comparing the composed summary
    against the constant; ``layout_record`` asks the pure layer for the *absence* now,
    so this is the whole round trip: msgid in the pure layer, words from the catalog.
    """

    assert translate_no_details() == "（无详情）"
    assert layout_record(activity_entry("read"), width=64).header.endswith("（无详情）")
    assert NO_DETAILS == "(no details)", "the msgid itself stays English"
