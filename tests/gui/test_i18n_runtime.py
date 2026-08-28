"""End-to-end proof that an installed catalog actually renders localized copy.

The point is the rendered widget text, not the presence of a ``.qm`` file: if this
module passes, converting the remaining UI copy to ``tr()`` is mechanical work.

Application ``.qm`` catalogs are gitignored build output, rebuilt by
``scripts/gui/build_translations.py``. The localized behaviour tests keep a skip guard so
a missing file reports one clear catalog failure instead of a large cascade; the
catalog-presence test in ``test_i18n.py`` remains the hard failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPushButton,
)

from kimix_gui.i18n import SUPPORTED_LANGUAGES
from kimix_gui.llm import AXIS_THINKING_EFFORT, resolved_provider_file
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.bridge import KimixBridge
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.components import DISCLOSURE_COLLAPSED, DISCLOSURE_EXPANDED, DisclosureHeader
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.i18n import (
    TRANSLATIONS_DIR,
    active_language,
    apply_language,
    catalog_file,
    set_active_language,
)
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog
from kimix_gui.qt.todo_panel import TodoPanel
from kimix_gui.session_index import SessionSummary
from kimix_gui.todos import TodoEntry, TodoSnapshot

from .qtutil import find, widget_text

CHINESE_CATALOG_MISSING = not catalog_file("zh_CN").is_file()
SKIP_REASON = (
    f"{catalog_file('zh_CN').name} not built; run: uv run python scripts/gui/build_translations.py"
)

needs_catalog = pytest.mark.skipif(CHINESE_CATALOG_MISSING, reason=SKIP_REASON)


@pytest.fixture
def restore_english(qtbot) -> Iterator[None]:
    """Undo the catalog swap so later tests keep seeing English copy."""

    yield
    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "en")


@pytest.fixture
def chinese(restore_english) -> None:
    """Install the Chinese catalogs for the duration of one test."""

    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "zh_CN")


def _provider_config(tmp_path: Path) -> Path:
    path = tmp_path / "provider.json"
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "capabilities": ["thinking"],
                "thinking_effort": "high",
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_skip_guard_matches_the_build_output() -> None:
    """Keep the skip honest: it must point at the path the build script writes.

    Without this, renaming the output directory would turn the localization proof
    into a permanent, invisible skip.
    """

    expected = TRANSLATIONS_DIR / "kimix_gui_zh_CN.qm"

    assert catalog_file("zh_CN") == expected
    assert TRANSLATIONS_DIR.parent.name == "kimix_gui"
    assert set(SUPPORTED_LANGUAGES) == {"en", "zh_CN"}


@needs_catalog
def test_chinese_catalog_renders_chinese_widget_text(qtbot, restore_english) -> None:
    qt_app = QApplication.instance()
    assert qt_app is not None
    assert apply_language(qt_app, InterfacePreferences(language="zh_CN")) == "zh_CN"
    assert active_language() == "zh_CN"

    dialog = PreferencesDialog(InterfacePreferences(language="zh_CN"), font_families=list)
    qtbot.addWidget(dialog)

    assert find(dialog, "preferences-subtitle", QLabel).text() == "应用偏好设置"
    assert find(dialog, "font-preview-label", QLabel).text() == "预览"
    descriptions = [
        label.text()
        for label in dialog.findChildren(QLabel, "preferences-description")
        if label.parent() is not None
    ]
    assert "选择 Kimix 全局使用的等宽字体。" in descriptions
    assert "改动保存后立即生效。" in descriptions
    assert find(dialog, "manage-llm-settings", QPushButton).toolTip() == "打开服务商配置"


@needs_catalog
def test_language_endonyms_are_never_translated(qtbot, restore_english) -> None:
    """The picker must stay readable whichever catalog happens to be installed."""

    qt_app = QApplication.instance()
    assert qt_app is not None
    set_active_language(qt_app, "zh_CN")

    dialog = PreferencesDialog(InterfacePreferences(language="zh_CN"), font_families=list)
    qtbot.addWidget(dialog)
    picker = find(dialog, "interface-language", QComboBox)

    assert [picker.itemText(index) for index in range(picker.count())] == [
        "跟随系统 · System",
        "English",
        "中文",
    ]


def test_english_preference_renders_english_widget_text(qtbot, restore_english) -> None:
    """Not skipped: English must hold even when no catalog was ever built.

    Also covers the swap back, which is where a leaked translator would show up.
    """

    qt_app = QApplication.instance()
    assert qt_app is not None
    if not CHINESE_CATALOG_MISSING:
        set_active_language(qt_app, "zh_CN")
    assert apply_language(qt_app, InterfacePreferences(language="en")) == "en"
    assert active_language() == "en"

    dialog = PreferencesDialog(InterfacePreferences(language="en"), font_families=list)
    qtbot.addWidget(dialog)

    assert find(dialog, "preferences-subtitle", QLabel).text() == "Application preferences"
    assert find(dialog, "font-preview-label", QLabel).text() == "PREVIEW"
    assert (
        find(dialog, "manage-llm-settings", QPushButton).toolTip() == "Open provider configuration"
    )


def test_the_bundled_qt_catalog_is_installed_too(qtbot, restore_english) -> None:
    """Second translator: Qt's own copy (dialog buttons, shortcuts, file dialogs).

    Never skipped -- ``qtbase_*.qm`` ships inside PySide6, so it is present in any
    checkout that can import Qt at all.
    """

    qt_app = QApplication.instance()
    assert qt_app is not None
    buttons = QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel

    # The box has to outlive the comprehension: dropping it frees its buttons.
    set_active_language(qt_app, "zh_CN")
    localized_box = QDialogButtonBox(buttons)
    qtbot.addWidget(localized_box)
    localized = [button.text() for button in localized_box.buttons()]
    set_active_language(qt_app, "en")
    english_box = QDialogButtonBox(buttons)
    qtbot.addWidget(english_box)
    english = [button.text() for button in english_box.buttons()]

    assert english == ["Save", "Cancel"]
    assert localized == ["\u4fdd\u5b58", "\u53d6\u6d88"]


def test_a_missing_catalog_degrades_to_the_english_msgid(qtbot, restore_english) -> None:
    """A language with no catalog at all must not raise and must stay English."""

    qt_app = QApplication.instance()
    assert qt_app is not None
    assert not catalog_file("kl_GL").exists()
    set_active_language(qt_app, "kl_GL")

    dialog = PreferencesDialog(InterfacePreferences(), font_families=list)
    qtbot.addWidget(dialog)

    assert active_language() == "kl_GL"
    assert find(dialog, "preferences-subtitle", QLabel).text() == "Application preferences"


@needs_catalog
def test_home_view_renders_chinese_chrome(qtbot, chinese, tmp_path: Path) -> None:
    """HomeView, built directly so no ``KimixGuiApp`` re-pins the language."""

    reference = resolved_provider_file(_provider_config(tmp_path))
    home = HomeView(tmp_path, default_config=reference, session_config_loader=lambda _id: None)
    qtbot.addWidget(home)
    home.show_sessions(
        [
            SessionSummary(
                id="sess-1",
                title="Fix login",
                updated_at=1_700_000_000.0,
                is_archived=True,
                size_bytes=42 * 1024,
                file_count=4,
                storage_format="SQLite",
            )
        ]
    )

    assert widget_text(home, "home-title") == "会话"
    assert find(home, "start-new-session", QPushButton).text() == "新建会话"
    assert find(home, "open-settings", QPushButton).text() == "设置"
    assert widget_text(home, "history-title") == "历史"
    assert find(home, "select-shown", QPushButton).text() == "全选"
    assert find(home, "session-search", QLineEdit).placeholderText() == "搜索会话"
    assert widget_text(home, "detail-overline") == "详情"
    assert find(home, "open-session", QPushButton).text() == "打开会话"
    assert find(home, "configure-session", QPushButton).text() == "配置"
    assert widget_text(home, "home-model") == "新会话 · Test Model"
    assert widget_text(home, "session-count") == "共 1 个"
    assert widget_text(home, "detail-state") == "已归档会话"
    # Placeholder substitution has to survive translation: the count is filled in
    # from the Chinese string, not appended to an English stem.
    assert widget_text(home, "detail-storage") == "SQLite · 4 个文件"
    assert find(home.session_rows()[0], "session-badge", QLabel).text() == "已归档"
    # The pure layer returned ``FileSize("42", "KB")``; the unit phrase is this view's.
    assert widget_text(home, "detail-size") == "42 KB"


@needs_catalog
def test_home_view_localizes_the_pure_layer_formatters(qtbot, chinese, tmp_path: Path) -> None:
    """Relative time, file size and the untitled placeholder all word themselves here.

    ``session_index`` only ever returns buckets and figures, so this is the only place
    the Chinese wording can be observed.
    """

    from kimix_gui.qt.session_copy import format_file_size, format_relative_time

    now = 1_700_000_000.0
    reference = resolved_provider_file(_provider_config(tmp_path))
    home = HomeView(tmp_path, default_config=reference, session_config_loader=lambda _id: None)
    qtbot.addWidget(home)
    home.show_sessions(
        [
            SessionSummary(
                id="sess-1",
                title="Untitled",
                updated_at=now,
                size_bytes=1536,
                file_count=1,
                storage_format="JSONL",
            )
        ]
    )

    assert format_relative_time(0, now=now) == "未知"
    assert format_relative_time(now - 10, now=now) == "刚刚"
    assert format_relative_time(now - 60, now=now) == "1 分钟前"
    assert format_relative_time(now - 120, now=now) == "2 分钟前"
    assert format_relative_time(now - 4000, now=now) == "1 小时前"
    assert format_relative_time(now - 8000, now=now) == "2 小时前"
    assert format_relative_time(now - 100_000, now=now) == "昨天"
    # A date pattern rather than an English month abbreviation.
    assert format_relative_time(now - 10 * 86400, now=now).endswith("日")
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(5 * 1024 * 1024) == "5 MB"
    # The placeholder title is a pure-layer word translated at the boundary.
    assert find(home.session_rows()[0], "session-title", QLabel).text() == "无标题"
    assert widget_text(home, "detail-title") == "无标题"
    assert widget_text(home, "detail-storage") == "JSONL · 1 个文件"


@needs_catalog
def test_transcript_rows_render_chinese_labels(qtbot, chinese) -> None:
    """The pure-layer label words, translated where the row becomes pixels."""

    from kimix_gui.qt.paint import layout_record

    from .transcript_helpers import activity_entry, text_entry

    assert layout_record(text_entry("user", "hi"), width=64).label == "你"
    assert layout_record(text_entry("thinking", "hmm"), width=64, expanded=True).header == "思考"
    assert layout_record(activity_entry("write", summary="a.py"), width=64).label == "写入"
    assert layout_record(activity_entry("todo_write", summary="2 items"), width=64).label == "待办"
    assert layout_record(text_entry("error", "boom"), width=64).label == "错误"
    merged = layout_record(
        activity_entry(
            "read",
            summary="a.py",
            call_text="a.py",
            result_summary="12 lines",
            result_text="file contents",
            status="ok",
        ),
        width=64,
        expanded=True,
    )
    assert merged.body == "a.py\n\nfile contents"
    assert "结果" not in merged.body
    # Wire-derived titles are not copy and stay verbatim.
    assert layout_record(activity_entry("bash", summary="ls"), width=64).label == "Bash"


@needs_catalog
def test_chat_view_renders_chinese_chrome(qtbot, chinese) -> None:
    """ChatView on an unstarted bridge: no worker thread, only the chrome."""

    bridge = KimixBridge()
    chat = ChatView(bridge)
    qtbot.addWidget(chat)

    assert widget_text(chat, "chat-title") == "对话"
    assert widget_text(chat, "status") == "连接中…"
    assert find(chat, "open-settings", QPushButton).text() == "设置"
    assert find(chat, "leave-session", QPushButton).text() == "主页"
    assert widget_text(chat, "history-info") == "历史 · 连接中…"
    assert find(chat, "send-prompt", QPushButton).text() == "发送"
    assert find(chat, "send-prompt", QPushButton).toolTip() == "发送消息"
    assert find(chat, "cancel-prompt", QPushButton).text() == "取消"
    assert find(chat, "cancel-prompt", QPushButton).toolTip() == "停止生成"
    assert find(chat, "load-older", QPushButton).toolTip() == "上一轮"
    assert find(chat, "jump-latest", QPushButton).toolTip() == "跳到最新"
    assert chat.prompt.placeholderText().startswith("向 AI 提问")
    assert find(chat, "history-turn", QLineEdit).placeholderText() == "轮次"


@needs_catalog
def test_llm_settings_dialog_renders_chinese_chrome(qtbot, chinese, tmp_path: Path) -> None:
    reference = resolved_provider_file(_provider_config(tmp_path))
    dialog = LLMSettingsDialog(
        current=reference,
        models=(reference.model,),
        scope_label="新会话",
        manage_library=True,
    )
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "LLM 配置"
    assert widget_text(dialog, "settings-title") == "LLM 配置"
    path_input = find(dialog, "config-path", QLineEdit)
    assert path_input.placeholderText() == "服务商 JSON 路径"
    assert path_input.accessibleName() == "服务商 JSON 路径"
    assert find(dialog, "browse-config", QPushButton).text() == "浏览…"
    assert find(dialog, "load-config", QPushButton).text() == "添加"
    assert widget_text(dialog, "config-sources-title") == "提供方"
    assert widget_text(dialog, "config-details-title") == "模型选择"
    assert widget_text(dialog, "selection-source") == "Provider 文件"
    assert widget_text(dialog, "selection-status") == "可用"
    chatgpt_group = find(dialog, "chatgpt-config-group", DisclosureHeader)
    provider_group = find(dialog, "provider-config-group", DisclosureHeader)
    assert chatgpt_group.text() == f"{DISCLOSURE_COLLAPSED}  ChatGPT 订阅 · 0"
    assert provider_group.text() == f"{DISCLOSURE_EXPANDED}  Provider 文件 · 1"
    assert chatgpt_group.toolTip() == "展开提供方"
    assert provider_group.toolTip() == "收起提供方"
    assert [item.text() for item in dialog.model_items()] == ["Test Model"]
    assert find(dialog, "apply-settings", QPushButton).text() == "使用模型"
    assert find(dialog, "delete-config", QPushButton).text() == "移除"
    assert find(dialog, "cancel-settings", QPushButton).text() == "取消"
    # ``format_tokens`` is a module function, so its copy goes through
    # ``QCoreApplication.translate`` instead of ``self.tr`` -- prove it lands too.
    assert widget_text(dialog, "model-context") == "100,000 tokens"
    assert widget_text(dialog, "model-capabilities") == "thinking"
    assert widget_text(dialog, "provider-thinking") == "由模型参数选择"
    assert widget_text(dialog, f"param-{AXIS_THINKING_EFFORT}-label") == "思考强度"
    effort = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert effort is not None
    assert effort.itemText(effort.currentIndex()) == "高（high） · 模型默认"
    assert effort.accessibleDescription() == "选择要为此模型参数保存的确切值"


@needs_catalog
def test_todo_panel_renders_chinese_summary(qtbot, chinese) -> None:
    panel = TodoPanel()
    qtbot.addWidget(panel)
    panel.set_snapshot(
        TodoSnapshot(
            entries=(
                TodoEntry("写面板", "done"),
                TodoEntry("翻译文案", "in_progress"),
                TodoEntry("跑测试", "pending"),
            )
        )
    )

    assert widget_text(panel, "todo-title") == "待办"
    assert widget_text(panel, "todo-footer") == "1 项进行中 · 1 项待处理 · 1 项已完成"
    assert "当前: 翻译文案" in panel._header.toolTip()
    assert "点击收起" in panel._header.toolTip()
