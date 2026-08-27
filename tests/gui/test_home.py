from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QWidget,
)

from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.design import DARK
from kimix_gui.llm import KimixGuiConfigStore, resolved_provider_file
from kimix_gui.qt import theme
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.request_dialogs import DeleteSessionsDialog
from kimix_gui.qt.session_copy import format_file_size, format_relative_time
from kimix_gui.qt.session_row import SessionRow
from kimix_gui.session_index import SessionSummary, relative_time
from kimix_gui.transcript_data import entry_body_text

from .qtutil import find, launch_app, wait_chat_ready, wait_home, wait_idle, widget_text


def _config_store(tmp_path: Path) -> KimixGuiConfigStore:
    config_file = tmp_path / "test-provider.json"
    config_file.write_text(
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
    store = KimixGuiConfigStore(
        tmp_path / "kimix-gui.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )
    store.set_default(tmp_path, resolved_provider_file(config_file).selection)
    return store


class FakeSession:
    def __init__(self, session_id: str = "fake-session") -> None:
        self.id = session_id
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self.prompts: list[str] = []
        self.closed = False

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        self.prompts.append(user_input)
        assert merge_wire_messages is False
        if False:  # pragma: no cover
            yield None

    def cancel(self) -> None:
        return None

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _summaries() -> list[SessionSummary]:
    return [
        SessionSummary(
            id="sess-1",
            title="Fix login",
            updated_at=1_700_000_000.0,
            is_last=True,
            size_bytes=1_572_864,
            file_count=4,
            storage_format="SQLite",
            todo_count=2,
            additional_dir_count=1,
        ),
        SessionSummary(
            id="sess-2",
            title="Untitled",
            updated_at=1_699_996_400.0,
            size_bytes=42 * 1024,
            file_count=2,
            storage_format="JSONL",
            is_archived=True,
        ),
    ]


async def _history_loader(_work_dir: Path) -> list[SessionSummary]:
    return list(reversed(_summaries()))


async def _empty_loader(_work_dir: Path) -> list[SessionSummary]:
    return []


def test_missing_session_id_opens_home(qtbot, tmp_path: Path) -> None:
    session = FakeSession()

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert [row.summary.id for row in home.session_rows()] == ["sess-1", "sess-2"]
    assert home.summary == _summaries()[0]
    assert widget_text(home, "detail-size") == "1.5 MB"
    assert widget_text(home, "detail-storage") == "SQLite · 4 files"
    assert widget_text(home, "detail-todos") == "2"
    assert widget_text(home, "detail-directories") == "1"


def test_new_session_shortcut_skips_resume(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession()

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_N)
    chat = wait_chat_ready(qtbot, app)
    assert opened[0].session_id is None
    assert chat.prompt_enabled is True


def test_new_session_button_starts_chat(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession()

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_empty_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "start-new-session"), Qt.MouseButton.LeftButton)
    wait_chat_ready(qtbot, app)
    assert opened[0].session_id is None
    assert isinstance(app.screen, ChatView)


def test_click_previews_session_before_opening(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession("sess-2")

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    qtbot.mouseClick(rows[1], Qt.MouseButton.LeftButton, pos=QPoint(90, 14))

    assert app.screen is home
    assert opened == []
    assert home.summary is not None
    assert home.summary.id == "sess-2"
    assert widget_text(home, "detail-state") == "Archived session"
    assert widget_text(home, "detail-size") == "42 KB"

    qtbot.mouseClick(find(home, "open-session"), Qt.MouseButton.LeftButton)
    wait_chat_ready(qtbot, app)
    assert opened[0].session_id == "sess-2"
    assert isinstance(app.screen, ChatView)


def test_double_click_opens_session(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession("sess-2")

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    qtbot.mouseDClick(rows[1], Qt.MouseButton.LeftButton, pos=QPoint(90, 14))
    wait_chat_ready(qtbot, app)
    assert opened[0].session_id == "sess-2"
    assert isinstance(app.screen, ChatView)


def test_click_session_check_toggles_selection(qtbot, tmp_path: Path) -> None:
    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    first = rows[0].summary
    mark = find(rows[1], "session-check", QCheckBox)
    assert mark.text() == ""
    qtbot.mouseClick(mark, Qt.MouseButton.LeftButton)

    assert rows[1].selected is True
    assert mark.isChecked() is True
    session_list = find(home, "session-list")
    list_y = session_list.y()
    list_height = session_list.height()
    assert widget_text(home, "selection-count") == "1 selected"
    assert find(home, "selection-count").isVisible()
    assert not find(home, "history-title").isVisible()
    assert find(home, "delete-sessions", QPushButton).isVisible()
    assert find(home, "delete-sessions", QPushButton).isEnabled()
    assert session_list.y() == list_y
    assert session_list.height() == list_height
    assert home.summary == first

    qtbot.mouseClick(mark, Qt.MouseButton.LeftButton)
    assert rows[1].selected is False
    assert mark.isChecked() is False
    assert not find(home, "selection-count").isVisible()
    assert find(home, "history-title").isVisible()
    assert not find(home, "delete-sessions", QPushButton).isVisible()
    assert session_list.y() == list_y
    assert session_list.height() == list_height


def test_home_session_rows_use_badges_instead_of_ascii_checks(qtbot, tmp_path: Path) -> None:
    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    assert find(rows[0], "session-badge", QLabel).text() == "Last"
    assert find(rows[1], "session-badge", QLabel).text() == "Archived"
    # Structural, not `"[ ]" not in ...`: the ASCII markers are gone, so a negative
    # substring check holds for any title and proves nothing. What the test is
    # really about is that selection is a real widget and the title is only the
    # title, so assert exactly that.
    for row, summary in zip(rows, _summaries(), strict=True):
        mark = find(row, "session-check", QCheckBox)
        assert mark.text() == ""
        assert mark.isVisibleTo(row)
        assert widget_text(row, "session-title") == summary.title


def test_home_filters_sessions_by_title(qtbot, tmp_path: Path) -> None:
    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    search = find(home, "session-search", QLineEdit)
    search.setText("LOGIN")
    assert [row.summary.id for row in home.session_rows()] == ["sess-1"]
    assert widget_text(home, "session-count") == "1 of 2"

    search.setText("missing title")
    assert home.session_rows() == []
    assert "No sessions match" in widget_text(home, "home-status")


def test_home_selects_and_deletes_sessions_in_batch(qtbot, tmp_path: Path) -> None:
    deleted: list[str] = []

    async def deleter(_work_dir: Path, session_ids: Sequence[str]) -> None:
        deleted.extend(session_ids)

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        session_deleter=deleter,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    session_list = find(home, "session-list")
    list_y = session_list.y()
    find(home, "select-shown", QPushButton).click()
    assert widget_text(home, "selection-count") == "2 selected"
    assert find(home, "selection-count").isVisible()
    assert find(home, "delete-sessions", QPushButton).isEnabled()
    assert session_list.y() == list_y
    assert all(row.selected for row in home.session_rows())
    assert all(find(row, "session-check", QCheckBox).isChecked() for row in home.session_rows())
    assert find(home, "select-shown", QPushButton).text() == "Clear"

    find(home, "delete-sessions", QPushButton).click()
    qtbot.waitUntil(lambda: isinstance(app.screen, DeleteSessionsDialog), timeout=10_000)
    find(app.screen, "confirm-delete", QPushButton).click()
    qtbot.waitUntil(lambda: home.session_rows() == [], timeout=10_000)
    wait_idle(qtbot, app)

    assert deleted == ["sess-1", "sess-2"]
    assert app.screen is home
    assert home.session_rows() == []
    assert not find(home, "selection-count").isVisible()
    assert find(home, "history-title").isVisible()
    assert not find(home, "delete-sessions", QPushButton).isVisible()


def test_enter_resumes_highlighted_session(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession("sess-1")

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    chat = wait_chat_ready(qtbot, app)
    assert opened[0].session_id == "sess-1"
    assert any(
        entry_body_text(record.entry) == "Session: sess-1" for record in chat.transcript.records
    )
    assert find(chat, "leave-session", QPushButton).text() == "Home"


def test_escape_from_chat_returns_home_and_releases_session(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    sessions: list[FakeSession] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        session = FakeSession(options.session_id or "created")
        sessions.append(session)
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    wait_home(qtbot, app)
    qtbot.keyClick(app.screen, Qt.Key.Key_Return)
    chat = wait_chat_ready(qtbot, app)
    assert chat.session_id == sessions[0].id
    assert any(
        entry_body_text(record.entry) == "Session: sess-1" for record in chat.transcript.records
    )

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    home = wait_home(qtbot, app)
    assert sessions[0].closed is True
    assert window.chat is None
    assert chat.busy is False
    assert app._options.session_id is None

    qtbot.keyClick(home, Qt.Key.Key_N)
    new_chat = wait_chat_ready(qtbot, app)
    assert opened[1].session_id is None
    assert new_chat.session_id == sessions[1].id
    assert sessions[1].closed is False
    # Against the session id, not the English opening line: the id cannot be
    # translated away, so this keeps proving the old session's rows are gone.
    assert all(
        "sess-1" not in entry_body_text(record.entry) for record in new_chat.transcript.records
    )


def test_quit_command_returns_home(qtbot, tmp_path: Path) -> None:
    session = FakeSession("fake-session")

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText("/quit")
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    wait_home(qtbot, app)
    assert session.closed is True
    assert app.window is not None
    assert app.window.chat is None


def test_home_uses_master_detail_layout_on_wide_window(qtbot, tmp_path: Path) -> None:
    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app, size=(1100, 700))
    home = wait_home(qtbot, app)
    browser = find(home, "session-browser")
    details = find(home, "session-detail")
    assert details.x() > browser.x()
    assert abs(details.y() - browser.y()) < 8
    assert find(home, "home-workspace", QSplitter).orientation() == Qt.Orientation.Horizontal


def test_home_stacks_sections_on_narrow_window(qtbot, tmp_path: Path) -> None:
    app = KimixGuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app, size=(700, 700))
    home = wait_home(qtbot, app)
    window.setMinimumSize(400, 400)
    window.resize(700, 700)
    QApplication.processEvents()
    home._sync_narrow(window.width())
    QApplication.processEvents()
    browser = find(home, "session-browser")
    details = find(home, "session-detail")
    assert find(home, "home-workspace", QSplitter).orientation() == Qt.Orientation.Vertical
    assert details.y() > browser.y()
    assert abs(details.x() - browser.x()) < 8
    assert find(home, "open-session").geometry().bottom() <= details.geometry().bottom()
    assert find(home, "configure-session").geometry().right() <= details.geometry().right()
    assert find(home, "select-shown").geometry().right() <= browser.geometry().right()
    assert find(home, "session-list").height() > 0


# --- The Qt half of the region-sensitive formatters -------------------------
# ``session_index`` returns figures; wording happens here. English is asserted
# without a catalog installed (``translate`` falls back to the msgid), Chinese in
# ``tests/gui/test_i18n_runtime.py``.


def test_relative_time_wording_covers_every_bucket() -> None:
    now = 1_700_000_000.0

    assert format_relative_time(0, now=now) == "unknown"
    assert format_relative_time(now - 10, now=now) == "just now"
    # Singular and plural are two whole strings, so this reads "1 minute", not "1 minutes".
    assert format_relative_time(now - 60, now=now) == "1 minute ago"
    assert format_relative_time(now - 120, now=now) == "2 minutes ago"
    assert format_relative_time(now - 4000, now=now) == "1 hour ago"
    assert format_relative_time(now - 8000, now=now) == "2 hours ago"
    assert format_relative_time(now - 100_000, now=now) == "yesterday"


#: English month abbreviations, written out so the expectation does not depend on the
#: developer machine's C locale the way ``strftime("%b")`` would.
_MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def test_relative_time_formats_old_dates_through_the_locale() -> None:
    now = 1_700_000_000.0
    this_year = relative_time(now - 10 * 86400, now=now)
    earlier = relative_time(now - 800 * 86400, now=now)
    assert this_year.moment is not None
    assert earlier.moment is not None

    # The Qt date *pattern* is the translated string: ``MMM d`` in English becomes
    # ``M月d日`` in Chinese, so no English month name can leak into a Chinese UI.
    month = _MONTH_ABBREVIATIONS[this_year.moment.month - 1]
    assert format_relative_time(now - 10 * 86400, now=now) == f"{month} {this_year.moment.day}"
    assert format_relative_time(now - 800 * 86400, now=now) == earlier.moment.strftime("%Y-%m-%d")


def test_file_size_wording_picks_the_unit_phrase() -> None:
    assert format_file_size(0) == "0 KB"
    assert format_file_size(512) == "0.5 KB"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(1024 * 1024) == "1 MB"
    assert format_file_size(int(5.5 * 1024 * 1024)) == "5.5 MB"


def _render(widget: QWidget) -> QImage:
    """Paint only this widget, not its children, so the fill is unobstructed."""

    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.black)
    widget.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawWindowBackground)
    return image


def _row(selected: bool = False) -> SessionRow:
    row = SessionRow(
        SessionSummary(
            id="sess-1",
            title="Fix login",
            updated_at=1_700_000_000.0,
            size_bytes=1024,
            file_count=1,
            storage_format="SQLite",
        ),
        selected=selected,
    )
    row.resize(320, DARK.session_list.row_height)
    return row


def test_the_active_session_row_paints_the_live_theme(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is hand-painted, so this is the only thing keeping it themeable.

    Nothing in the stylesheet selects ``#session-row``'s fills; the painter reads
    ``active_theme()`` on every call, and that chain is what breaks silently if
    someone reintroduces an import-time colour copy.
    """

    row = _row()
    qtbot.addWidget(row)
    row.set_active(True)

    default = _render(row).pixelColor(160, DARK.session_list.row_height // 2)
    assert default.name() == DARK.palette.boost

    probe = replace(DARK, palette=replace(DARK.palette, boost="#ff00ff"))
    monkeypatch.setattr(theme, "_ACTIVE_THEME", probe)
    repainted = _render(row).pixelColor(160, DARK.session_list.row_height // 2)
    assert repainted.name() == "#ff00ff"


def test_the_active_marker_sits_where_the_token_puts_it(qtbot) -> None:
    """The inset rounded marker is why this row cannot become a stylesheet rule."""

    metrics = DARK.session_list
    row = _row()
    qtbot.addWidget(row)
    row.set_active(True)
    image = _render(row)

    middle = metrics.row_height // 2
    inside = image.pixelColor(int(metrics.marker_x) + 1, middle)
    assert inside.name() == DARK.palette.accent
    above = image.pixelColor(int(metrics.marker_x) + 1, metrics.marker_inset_y // 2)
    assert above.name() == DARK.palette.boost, "the marker stops short of the top edge"
