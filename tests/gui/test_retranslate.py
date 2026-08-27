"""Does a widget that already exists change language, and on what does that rest?

Three layers, deliberately separate:

1. The two Qt behaviours ``Retranslator`` is built on. If either changes in a future
   PySide6, the design is wrong rather than merely broken, so they are pinned here
   with no product code involved.
2. ``Retranslator`` itself.
3. End to end: build a view in English, install the Chinese catalog, pump, and read
   the text off the same widget objects. These retain the shared missing-catalog guard
   described in ``tests/gui/test_i18n_runtime.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QObject, QRect
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from kimix_gui.llm_config import inspect_llm_config
from kimix_gui.qt.bridge import KimixBridge
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.i18n import catalog_file, set_active_language
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.todo_panel import TodoPanel
from kimix_gui.qt.transcript import Transcript
from kimix_gui.session_index import SessionSummary
from kimix_gui.todos import TodoEntry, TodoSnapshot

from .qtutil import find, widget_text
from .transcript_helpers import append_text

CHINESE_CATALOG_MISSING = not catalog_file("zh_CN").is_file()
needs_catalog = pytest.mark.skipif(
    CHINESE_CATALOG_MISSING,
    reason=(
        f"{catalog_file('zh_CN').name} not built; run: "
        "uv run python scripts/gui/build_translations.py"
    ),
)

SUMMARY = SessionSummary(
    id="s-1",
    title="Untitled",
    updated_at=1_700_000_000.0,
    is_last=True,
    size_bytes=2048,
    file_count=3,
    todo_count=2,
)


def _provider_config(tmp_path: Path) -> Path:
    """A config file that inspects cleanly, so the home view has a model label."""

    path = tmp_path / "provider.json"
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def chinese(qtbot) -> Iterator[None]:
    """Install the Chinese catalog for one test, then put English back.

    Restoring matters more here than elsewhere: these tests exist to prove copy
    changes underneath live widgets, so leaking the catalog would change the copy
    every later test reads.
    """

    qt_app = QApplication.instance()
    assert qt_app is not None
    yield
    set_active_language(qt_app, "en")


def _switch_to_chinese() -> None:
    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "zh_CN")
    # Posted, not sent. Without this the widgets are still in English and every
    # assertion below would report the design as broken.
    qt_app.processEvents()


# ---- layer 1: the Qt behaviours the design rests on ------------------------------


def test_language_change_reaches_plain_qobject_children(qtbot) -> None:
    """A ``QObject`` child is enough. This is why no host overrides ``changeEvent``."""

    class Listener(QObject):
        def __init__(self, parent: QObject) -> None:
            super().__init__(parent)
            self.hits = 0

        def event(self, event: QEvent) -> bool:
            if event.type() == QEvent.Type.LanguageChange:
                self.hits += 1
            return super().event(event)

    root = QWidget()
    qtbot.addWidget(root)
    deep_label = QLabel("deep", root)
    on_root = Listener(root)
    under_label = Listener(deep_label)
    orphan = Listener(QObject())

    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "zh_CN")
    qt_app.processEvents()
    set_active_language(qt_app, "en")
    qt_app.processEvents()

    assert on_root.hits > 0, "a plain QObject child of a widget is notified"
    assert under_label.hits > 0, "so is one nested under a leaf label"
    assert orphan.hits == 0, "an unparented QObject is not; parenting is the subscription"


def test_language_change_is_posted_not_sent(qtbot) -> None:
    """Nothing is retranslated until the event loop turns.

    The same trap as ``QApplication.setFont``. It is pinned because the failure mode
    is a test that proves the opposite of what it claims: skip the pump and every
    widget still reads English, which looks exactly like a broken design.
    """

    class Listener(QObject):
        def __init__(self, parent: QObject) -> None:
            super().__init__(parent)
            self.hits = 0

        def event(self, event: QEvent) -> bool:
            if event.type() == QEvent.Type.LanguageChange:
                self.hits += 1
            return super().event(event)

    root = QWidget()
    qtbot.addWidget(root)
    listener = Listener(root)
    qt_app = QApplication.instance()
    assert qt_app is not None

    set_active_language(qt_app, "zh_CN")
    assert listener.hits == 0, "install alone delivers nothing"
    qt_app.processEvents()
    assert listener.hits > 0, "the pump is what delivers it"

    set_active_language(qt_app, "en")
    qt_app.processEvents()


# ---- layer 2: Retranslator ------------------------------------------------------


def test_bind_runs_the_statement_immediately(qtbot) -> None:
    """The reason a converted call site renders identically to the original."""

    host = QWidget()
    qtbot.addWidget(host)
    label = QLabel(host)
    Retranslator(host).bind(lambda: label.setText("first"))
    assert label.text() == "first"


def test_retranslate_reruns_every_statement(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    runs: list[str] = []
    i18n = Retranslator(host)
    i18n.bind(lambda: runs.append("a"))
    i18n.bind(lambda: runs.append("b"))
    assert runs == ["a", "b"]
    i18n.retranslate()
    assert runs == ["a", "b", "a", "b"], "in bind order, so later statements still win"
    assert i18n.statement_count == 2


def test_a_language_change_drives_the_retranslator(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    calls: list[int] = []
    i18n = Retranslator(host)
    i18n.bind(lambda: calls.append(1))

    QApplication.sendEvent(i18n, QEvent(QEvent.Type.LanguageChange))
    assert len(calls) == 2, "once from bind, once from the event"

    QApplication.sendEvent(i18n, QEvent(QEvent.Type.FontChange))
    assert len(calls) == 2, "and only for that event"


def test_binding_does_not_grow_when_derived_copy_is_recomputed(qtbot) -> None:
    """Why dynamic copy binds an existing recompute method instead of re-binding.

    Re-binding inside a refresh would append a statement per refresh, and the leak
    would be invisible: the copy stays correct, the list just gets longer forever.
    """

    panel = TodoPanel()
    qtbot.addWidget(panel)
    before = panel._i18n.statement_count
    for index in range(5):
        panel.set_snapshot(TodoSnapshot(entries=(TodoEntry(f"todo {index}", "pending"),)))
    assert panel._i18n.statement_count == before


# ---- layer 3: end to end --------------------------------------------------------


@needs_catalog
def test_a_live_todo_panel_changes_language(qtbot, chinese) -> None:
    panel = TodoPanel()
    qtbot.addWidget(panel)
    panel.set_snapshot(
        TodoSnapshot(
            entries=(TodoEntry("write it", "in_progress"), TodoEntry("ship it", "pending"))
        )
    )
    assert widget_text(panel, "todo-title") == "TODOS"
    assert "in progress" in widget_text(panel, "todo-footer")

    _switch_to_chinese()

    assert widget_text(panel, "todo-title") == "待办"
    assert "进行中" in widget_text(panel, "todo-footer")
    assert "点击收起" in panel._header.toolTip()


@needs_catalog
def test_a_live_chat_view_changes_language(qtbot, chinese) -> None:
    bridge = KimixBridge()
    view = ChatView(bridge)
    qtbot.addWidget(view)
    assert widget_text(view, "chat-title") == "CHAT"
    assert widget_text(view, "leave-session") == "Home"
    assert widget_text(view, "status") == "connecting…"
    assert widget_text(view, "history-info") == "History · connecting…"

    _switch_to_chinese()

    assert widget_text(view, "chat-title") == "对话"
    assert widget_text(view, "leave-session") == "主页"
    assert widget_text(view, "status") == "连接中…", "a derived status line, re-derived"
    assert widget_text(view, "history-info") == "历史 · 连接中…"
    assert find(view, "send-prompt", QPushButton).text() == "发送"
    assert "Enter 发送" in view.prompt.placeholderText()
    expand = find(view.prompt, "expand-prompt", QPushButton)
    assert expand.toolTip() == "撰写长消息", "the Composer owns its own retranslator"


@needs_catalog
def test_a_live_home_view_changes_language(qtbot, tmp_path, chinese) -> None:
    reference = inspect_llm_config(_provider_config(tmp_path))
    view = HomeView(
        tmp_path,
        default_config=reference,
        session_config_loader=lambda _session_id: None,
    )
    qtbot.addWidget(view)
    view.show_sessions([SUMMARY])
    assert widget_text(view, "home-title") == "Sessions"
    assert widget_text(view, "detail-overline") == "Details"
    assert widget_text(view, "session-badge") == "Last"
    assert widget_text(view, "session-title") == "Untitled"
    assert widget_text(view, "detail-state") == "Last active session"

    _switch_to_chinese()

    assert widget_text(view, "home-title") == "会话"
    assert widget_text(view, "detail-overline") == "详情"
    assert widget_text(view, "select-shown") == "全选"
    assert widget_text(view, "session-badge") == "最近", "a row built before the switch"
    assert widget_text(view, "session-title") == "无标题"
    assert widget_text(view, "detail-state") == "最近活动的会话"
    assert widget_text(view, "detail-storage").endswith("个文件"), "a derived detail value"


@needs_catalog
def test_the_metadata_field_names_change_language(qtbot, tmp_path, chinese) -> None:
    """The names, not just the values: they live in a shared component.

    ``KeyValueList`` takes its labels already translated, so it needed a way to be
    told them again. The accessible name is asserted with the visible one because
    they are set together and drifting apart would be silent.
    """

    view = HomeView(
        tmp_path,
        default_config=inspect_llm_config(_provider_config(tmp_path)),
        session_config_loader=lambda _session_id: None,
    )
    qtbot.addWidget(view)
    view.show_sessions([SUMMARY])
    keys = [
        label.text()
        for label in find(view, "detail-metadata").findChildren(QLabel, "key-value-key")
    ]
    assert "Session ID" in keys

    _switch_to_chinese()

    keys = [
        label.text()
        for label in find(view, "detail-metadata").findChildren(QLabel, "key-value-key")
    ]
    assert "会话 ID" in keys
    assert find(view, "detail-id").accessibleName() == "会话 ID"


def test_a_live_transcript_drops_its_document_cache(qtbot) -> None:
    """No catalog needed: the assertion is that Qt is told to paint again.

    The transcript's translated copy is painted, not stored, so there is no text to
    re-set. Dropping the cache is what makes the labels come back through
    ``translate_label`` on the next paint.
    """

    transcript = Transcript()
    qtbot.addWidget(transcript)
    append_text(transcript, "assistant", "a paragraph long enough to need a document")
    index = transcript.model().index(0, 0)
    assert transcript._delegate.document_for(QRect(0, 0, 600, 0), index) is not None
    assert transcript.bodies.documents, "the delegate cached a document to invalidate"

    QApplication.sendEvent(transcript._i18n, QEvent(QEvent.Type.LanguageChange))

    assert not transcript.bodies.documents, "a language change invalidates it"
