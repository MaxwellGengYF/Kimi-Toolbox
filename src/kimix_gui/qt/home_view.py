"""Home view: session browser, search, batch delete, and details pane."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import (
    QSize,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.llm import ChatGPTTarget, ProviderFileTarget, ResolvedLLMSelection
from kimix_gui.qt import keys
from kimix_gui.qt.components import Card, KeyValueList
from kimix_gui.qt.labels import translate_session_title
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.session_copy import format_file_size, format_relative_time, format_timestamp
from kimix_gui.qt.session_row import SessionRow
from kimix_gui.qt.styling import (
    CardLevel,
    Level,
    Role,
    Surface,
    Tone,
    Variant,
    style,
)
from kimix_gui.session_index import (
    SessionSummary,
)

SessionConfigLoader = Callable[[str], ResolvedLLMSelection | None]

# Import-time bound, like the transcript metrics: the names stay, the numbers
# moved to the token layer.
_MARK_SIZE = DARK.session_list.mark_size
_ROW_HEIGHT = DARK.session_list.row_height


class HomeView(QWidget):
    """Browse project sessions or start a new one."""

    new_session = Signal()
    resume_session = Signal(str)
    open_settings = Signal()
    # Carries the session to configure, or ``None`` for the folder's default. It is an
    # ``object`` signal for the sake of that ``None``: ``F4`` on this page configures
    # the default, and ``KimixGuiApp.open_llm_settings`` already takes ``str | None``.
    configure_session = Signal(object)
    quit_requested = Signal()
    delete_requested = Signal(list)
    llm_required = Signal(object)

    def __init__(
        self,
        work_dir: Path,
        *,
        default_config: ResolvedLLMSelection,
        session_config_loader: SessionConfigLoader,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("home-view")
        self._i18n = Retranslator(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._work_dir = work_dir
        self._default_config = default_config
        self._session_config_loader = session_config_loader
        self._summaries: list[SessionSummary] = []
        self._loaded = False
        self._load_error: str | None = None
        self._selected_ids: set[str] = set()
        self._narrow = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*DARK.card_padding.wide)
        root.setSpacing(12)
        root.addWidget(self._toolbar())

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("home-workspace")
        browser = self._browser()
        self._splitter.addWidget(browser)
        details = self._details_pane()
        browser.setMinimumWidth(200)
        details.setMinimumWidth(200)
        self._splitter.addWidget(details)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([380, 640])
        root.addWidget(self._splitter, 1)

        self._i18n.bind(self._refresh_model_label)
        self._i18n.bind(self._refresh_browser_copy)
        self._i18n.bind(self._refresh_details_copy)
        self._open.clicked.connect(self._open_current)
        self._configure.clicked.connect(self._configure_current)
        self._select_shown.clicked.connect(self._toggle_shown)
        self._delete.clicked.connect(self.request_delete)
        self._search.textChanged.connect(self._render_sessions)
        keys.install(
            self,
            keys.HOME,
            {
                "new-session": self.request_new_session,
                "quit": self.quit_requested.emit,
                "toggle-selection": self.toggle_selected,
                "delete-selection": self.request_delete,
                "open-highlighted": self.open_highlighted,
                "focus-search": self.focus_search,
                "configure-llm": self.request_default_configuration,
            },
        )
        # Last, because binding it is what seeds the selection count and the
        # select-all label. No explicit row list: it derives one from the list widget,
        # which is empty now and will not be when a language change re-runs this.
        self._i18n.bind(self._update_selection_controls)

    def _toolbar(self) -> QFrame:
        """Page header: work dir, active model, and the two actions that leave the browser."""
        toolbar = QFrame()
        toolbar.setObjectName("home-toolbar")
        style(toolbar, surface=Surface.BAR)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 4, 4, 8)
        context = QVBoxLayout()
        context.setSpacing(2)
        title = QLabel()
        title.setObjectName("home-title")
        style(title, role=Role.DISPLAY, level=Level.ONE)
        self._i18n.bind(lambda: title.setText(self.tr("Sessions")))
        path = QLabel(str(self._work_dir))
        path.setObjectName("home-path")
        style(path, tone=Tone.MUTED)
        self._home_model = QLabel()
        self._home_model.setObjectName("home-model")
        style(self._home_model, tone=Tone.MUTED)
        context.addWidget(title)
        context.addWidget(path)
        context.addWidget(self._home_model)
        toolbar_layout.addLayout(context, 1)
        new_btn = QPushButton()
        new_btn.setObjectName("start-new-session")
        style(new_btn, variant=Variant.PRIMARY)
        self._new_btn = new_btn
        self._i18n.bind(lambda: new_btn.setText(self.tr("New session")))
        settings = QPushButton()
        settings.setObjectName("open-settings")
        self._i18n.bind(lambda: settings.setText(self.tr("Settings")))
        toolbar_layout.addWidget(new_btn)
        toolbar_layout.addWidget(settings)
        # Connected here rather than with the rest of the wiring in ``_build``: both
        # buttons are locals, so this is the last place that can still reach them.
        new_btn.clicked.connect(self.request_new_session)
        settings.clicked.connect(self.open_settings.emit)
        return toolbar

    def _browser(self) -> QWidget:
        """Left splitter pane: the session list with its header, search box and status line."""
        browser = QWidget()
        browser.setObjectName("session-browser")
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(0, 0, 8, 0)
        browser_layout.setSpacing(8)
        browser_layout.addWidget(self._history_header())
        self._search = QLineEdit()
        self._search.setObjectName("session-search")
        self._i18n.bind(lambda: self._search.setPlaceholderText(self.tr("Search sessions")))
        self._search.setClearButtonEnabled(True)
        browser_layout.addWidget(self._search)

        self._status = QLabel()
        self._status.setObjectName("home-status")
        style(self._status, tone=Tone.MUTED)
        browser_layout.addWidget(self._status)
        self._list = QListWidget()
        self._list.setObjectName("session-list")
        self._list.setUniformItemSizes(True)
        self._list.setSpacing(2)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._list.installEventFilter(self)
        self._list.currentRowChanged.connect(self._on_row_changed)
        browser_layout.addWidget(self._list, 1)
        return browser

    def _history_header(self) -> QWidget:
        """Bar above the session list: heading, selection count, and the batch actions."""
        header_bar = QWidget()
        header_bar.setObjectName("history-header")
        header_bar.setFixedHeight(32)
        header = QHBoxLayout(header_bar)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._history_title = QLabel()
        self._history_title.setObjectName("history-title")
        style(self._history_title, role=Role.SECTION)
        self._i18n.bind(lambda: self._history_title.setText(self.tr("History")))
        self._selection_count = QLabel()
        self._selection_count.setObjectName("selection-count")
        style(self._selection_count, role=Role.SECTION)
        self._session_count = QLabel("")
        self._session_count.setObjectName("session-count")
        style(self._session_count, tone=Tone.MUTED)
        self._select_shown = QPushButton()
        self._select_shown.setObjectName("select-shown")
        style(self._select_shown, variant=Variant.GHOST)
        self._select_shown.setEnabled(False)
        self._select_shown.setFlat(True)
        self._select_shown.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_shown.setFixedHeight(DARK.sizing.compact_control_height)
        self._delete = QPushButton()
        self._delete.setObjectName("delete-sessions")
        style(self._delete, variant=Variant.DANGER)
        self._i18n.bind(lambda: self._delete.setText(self.tr("Delete")))
        self._delete.setEnabled(False)
        self._delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete.setFixedHeight(DARK.sizing.compact_control_height)
        header.addWidget(self._history_title)
        header.addWidget(self._selection_count)
        header.addStretch()
        header.addWidget(self._select_shown)
        header.addWidget(self._delete)
        header.addWidget(self._session_count)
        return header_bar

    def _details_pane(self) -> Card:
        """Right splitter pane: the highlighted session's actions and metadata."""
        self._details = Card(CardLevel.PANEL)
        self._details.setObjectName("session-detail")
        details_layout = self._details.body
        details_layout.setSpacing(8)
        overline = QLabel()
        overline.setObjectName("detail-overline")
        style(overline, role=Role.OVERLINE)
        self._i18n.bind(lambda: overline.setText(self.tr("Details")))
        self._detail_title = QLabel()
        self._detail_title.setObjectName("detail-title")
        self._detail_title.setWordWrap(True)
        self._detail_state = QLabel()
        self._detail_state.setObjectName("detail-state")
        style(self._detail_state, tone=Tone.MUTED)
        details_layout.addWidget(overline)
        details_layout.addWidget(self._detail_title)
        details_layout.addWidget(self._detail_state)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._open = QPushButton()
        self._open.setObjectName("open-session")
        style(self._open, variant=Variant.PRIMARY)
        self._open.setEnabled(False)
        self._i18n.bind(lambda: self._open.setText(self.tr("Open session")))
        self._configure = QPushButton()
        self._configure.setObjectName("configure-session")
        self._configure.setEnabled(False)
        self._i18n.bind(lambda: self._configure.setText(self.tr("Configure")))
        actions.addWidget(self._open)
        actions.addWidget(self._configure)
        actions.addStretch()
        details_layout.addLayout(actions)
        self._meta = KeyValueList(self._metadata_rows())
        self._meta.setObjectName("detail-metadata")
        self._i18n.bind(lambda: self._meta.set_labels(self._metadata_rows()))
        self._detail_values = self._meta.values
        # Double the card's usual gap, separating what you can *do* with a session
        # from what is *true* about it. It used to be a top content margin on the
        # metadata block; the shared component does not own outer margins, so the
        # separation is stated here, where the grouping decision lives.
        details_layout.addSpacing(DARK.spacing.md)
        details_layout.addWidget(self._meta)
        details_layout.addStretch()
        return self._details

    @property
    def summary(self) -> SessionSummary | None:
        item = self._list.currentItem()
        row = self._row_of(item) if item is not None else None
        return row.summary if row is not None else None

    @property
    def configuration_available(self) -> bool:
        summary = self.summary
        if summary is None:
            return False
        saved = self._session_config_loader(summary.id)
        return (saved or self._default_config).available

    def session_rows(self) -> list[SessionRow]:
        rows: list[SessionRow] = []
        for index in range(self._list.count()):
            row = self._row_at(index)
            if row is not None:
                rows.append(row)
        return rows

    def show_sessions(self, summaries: list[SessionSummary]) -> None:
        self._summaries = summaries
        self._loaded = True
        self._load_error = None
        self._selected_ids.intersection_update(summary.id for summary in summaries)
        self._render_sessions()

    def show_load_error(self, message: str) -> None:
        self._list.clear()
        self._list.hide()
        self._status.show()
        self._load_error = message
        self._refresh_browser_copy()
        self._refresh_details_copy()
        self._update_selection_controls([])

    def refresh_configuration(self, default_config: ResolvedLLMSelection) -> None:
        self._default_config = default_config
        self._refresh_model_label()
        if self.summary is not None:
            self._show_session(self.summary)

    def request_new_session(self) -> None:
        if not self._default_config.available:
            self.llm_required.emit(None)
            return
        self.new_session.emit()

    def open_highlighted(self) -> None:
        summary = self.summary
        if summary is not None:
            self._open_summary(summary)

    def request_default_configuration(self) -> None:
        """Configure the folder's default LLM rather than one session's.

        ``MainWindow`` used to own this: a window-wide ``F4`` whose handler asked which
        page was showing and then called the controller directly, bypassing this view's
        own signal. The key belongs to the page that gives it meaning.
        """
        self.configure_session.emit(None)

    def focus_search(self) -> None:
        self._search.setFocus()

    def toggle_selected(self) -> None:
        item = self._list.currentItem()
        row = self._row_of(item) if item is not None else None
        if row is not None:
            self._toggle_row(row)

    def request_delete(self) -> None:
        if self._selected_ids:
            self.delete_requested.emit(sorted(self._selected_ids))

    def apply_deleted(self, ids: list[str]) -> None:
        id_set = set(ids)
        self._summaries = [summary for summary in self._summaries if summary.id not in id_set]
        self._selected_ids.difference_update(id_set)
        self._render_sessions()

    def take_keyboard_focus(self) -> None:
        """Make this page the keyboard's target: the session list, not the search field.

        Arrow keys should move through sessions the moment the page is in front, and the
        bindings this page owns only fire while the focus is inside it. See
        ``keys.ensure_focus``.
        """
        keys.ensure_focus(self, self._list)

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        self._sync_narrow(self.width())
        self.take_keyboard_focus()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._sync_narrow(self.width())

    def _sync_narrow(self, width: int) -> None:
        win = self.window()
        measured = min(width, win.width()) if win is not None else width
        narrow = measured < DARK.breakpoints.home_narrow
        if narrow == self._narrow and self._splitter.orientation() == (
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        ):
            return
        self._narrow = narrow
        self._splitter.setOrientation(
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        )

    def _refresh_model_label(self) -> None:
        # Two whole sentences rather than a translated suffix glued onto a
        # translated stem: the " · missing" tail has no meaning on its own.
        if self._default_config.available:
            template = self.tr("New sessions · {config}")
        elif isinstance(self._default_config.selection.target, ChatGPTTarget):
            template = self.tr("New sessions · {config} · connect ChatGPT")
        else:
            template = self.tr("New sessions · {config} · missing")
        self._home_model.setText(template.format(config=self._default_config.label))

    def _metadata_rows(self) -> tuple[tuple[str, str], ...]:
        """The details pane's field names, paired with the object name of their value."""

        return (
            ("detail-updated", self.tr("Updated")),
            ("detail-llm", self.tr("LLM")),
            ("detail-provider", self.tr("Provider")),
            ("detail-config", self.tr("Config")),
            ("detail-size", self.tr("Size")),
            ("detail-storage", self.tr("Storage")),
            ("detail-todos", self.tr("Active todos")),
            ("detail-directories", self.tr("Extra folders")),
            ("detail-id", self.tr("Session ID")),
            ("detail-path", self.tr("Folder")),
        )

    def _refresh_browser_copy(self) -> None:
        """Derive the list's status line and its count from what has been loaded.

        Only the text: visibility is decided by ``_render_sessions``, which knows
        whether the list or the status line is the thing on screen. The status line is
        phrased even while hidden, so this stays a pure function of state -- which is
        also why its two "nothing to show" branches read oddly when there *are* rows.
        ``_render_sessions`` only ever shows it when the filtered list is empty, and
        that is the case both branches are phrased for.
        """

        if self._load_error is not None:
            self._status.setText(
                self.tr("Could not load sessions: {reason}").format(reason=self._load_error)
            )
            self._session_count.setText(self.tr("Unavailable"))
            return
        if not self._loaded:
            self._status.setText(self.tr("Loading sessions…"))
            return
        total = len(self._summaries)
        query = self._search.text().strip()
        self._session_count.setText(
            self.tr("{shown} of {total}").format(shown=len(self._filtered()), total=total)
            if query
            else self.tr("{total} total").format(total=total)
        )
        if total and query:
            self._status.setText(self.tr("No sessions match this title"))
        else:
            self._status.setText(self.tr("No saved sessions in this folder"))

    def _refresh_details_copy(self) -> None:
        """Derive the details pane: a selected session, or why there is none.

        There are five reasons for an empty pane and each used to be phrased at the
        moment it was discovered, from four different call sites. Deriving them from
        state instead is what lets the pane be re-phrased in another language.
        """

        if self._load_error is not None:
            self._show_empty(
                self.tr("Sessions unavailable"), self.tr("Start a new session to continue")
            )
            return
        summary = self.summary
        if summary is not None:
            self._show_session(summary)
            return
        if not self._loaded:
            self._show_empty(self.tr("Loading sessions…"), self.tr("Reading this project"))
            return
        if not self._filtered():
            if self._summaries and self._search.text().strip():
                self._show_empty(self.tr("No matching sessions"), self.tr("Try another title"))
            else:
                self._show_empty(
                    self.tr("No sessions yet"), self.tr("Start a new session in this folder")
                )
            return
        self._show_empty(self.tr("No session selected"), "")

    def _filtered(self) -> list[SessionSummary]:
        query = self._search.text().strip().casefold()
        # Matched against the *displayed* title so a user can search for what is on
        # screen: an untitled session reads "Untitled" only in English.
        return [
            summary
            for summary in self._summaries
            if not query or query in translate_session_title(summary.title).casefold()
        ]

    def _render_sessions(self) -> None:
        filtered = self._filtered()
        current_id = self.summary.id if self.summary is not None else None
        self._list.clear()
        self._refresh_browser_copy()
        if not filtered:
            self._list.hide()
            self._status.show()
            self._refresh_details_copy()
            self._update_selection_controls(filtered)
            if not self._search.hasFocus():
                self._new_btn.setFocus()
            return
        self._status.hide()
        self._list.show()
        select_row = 0
        for index, summary in enumerate(filtered):
            row = SessionRow(summary, selected=summary.id in self._selected_ids)
            row.check_toggled.connect(self._on_mark_clicked)
            row.opened.connect(self._open_id)
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, _ROW_HEIGHT))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            if summary.id == current_id:
                select_row = index
        self._list.setCurrentRow(select_row)
        self._update_selection_controls(filtered)
        if not self._search.hasFocus():
            self._list.setFocus()

    def _row_at(self, index: int) -> SessionRow | None:
        item = self._list.item(index)
        widget = self._list.itemWidget(item) if item is not None else None
        return widget if isinstance(widget, SessionRow) else None

    def _row_of(self, item: QListWidgetItem | None) -> SessionRow | None:
        if item is None:
            return None
        widget = self._list.itemWidget(item)
        return widget if isinstance(widget, SessionRow) else None

    def _on_row_changed(self, row: int) -> None:
        for index in range(self._list.count()):
            session_row = self._row_at(index)
            if session_row is not None:
                session_row.set_active(index == row)
        session_row = self._row_at(row)
        if session_row is None:
            self._refresh_details_copy()
            return
        self._show_session(session_row.summary)
        self._update_selection_controls()

    def _show_session(self, summary: SessionSummary) -> None:
        self._meta.show()
        self._detail_title.setText(translate_session_title(summary.title))
        if summary.is_archived:
            state = self.tr("Archived session")
        elif summary.is_last:
            state = self.tr("Last active session")
        else:
            state = self.tr("Saved session")
        self._detail_state.setText(state)
        relative = format_relative_time(summary.updated_at)
        timestamp = format_timestamp(summary.updated_at)
        self._detail_values["detail-updated"].setText(f"{relative} · {timestamp}")
        saved_config = self._session_config_loader(summary.id)
        effective = saved_config or self._default_config
        self._detail_values["detail-llm"].setText(effective.label)
        self._detail_values["detail-provider"].setText(effective.model.provider_type)
        config_source = (
            self.tr("ChatGPT subscription")
            if isinstance(effective.selection.target, ChatGPTTarget)
            else str(effective.selection.target.path)
            if isinstance(effective.selection.target, ProviderFileTarget)
            else effective.model.endpoint
        )
        if saved_config is None:
            config_source = self.tr("{path} · project default").format(path=config_source)
        if not effective.available:
            config_source = (
                self.tr("{path} · connect ChatGPT")
                if isinstance(effective.selection.target, ChatGPTTarget)
                else self.tr("{path} · missing")
            ).format(path=config_source)
        self._detail_values["detail-config"].setText(config_source)
        self._detail_values["detail-size"].setText(format_file_size(summary.size_bytes))
        # Spelled-out singular / plural instead of ``%n``: the msgid is what renders
        # when no catalog is installed (a fresh clone, and the test suite), so
        # ``%n file(s)`` would put a literal "(s)" on screen.
        self._detail_values["detail-storage"].setText(
            (
                self.tr("{format} · 1 file")
                if summary.file_count == 1
                else self.tr("{format} · {count} files")
            ).format(format=summary.storage_format, count=summary.file_count)
        )
        self._detail_values["detail-todos"].setText(str(summary.todo_count))
        self._detail_values["detail-directories"].setText(str(summary.additional_dir_count))
        self._detail_values["detail-id"].setText(summary.id)
        self._detail_values["detail-path"].setText(str(self._work_dir))
        self._open.setEnabled(True)
        self._configure.setEnabled(True)

    def _show_empty(self, title: str, state: str) -> None:
        self._detail_title.setText(title)
        self._detail_state.setText(state)
        self._meta.hide()
        self._open.setEnabled(False)
        self._configure.setEnabled(False)

    def _update_selection_controls(self, filtered: list[SessionSummary] | None = None) -> None:
        count = len(self._selected_ids)
        selecting = count > 0
        self._selection_count.setText(self.tr("{count} selected").format(count=count))
        self._selection_count.setVisible(selecting)
        self._history_title.setVisible(not selecting)
        self._session_count.setVisible(not selecting)
        self._delete.setEnabled(selecting)
        self._delete.setVisible(selecting)
        if filtered is None:
            filtered = [row.summary for row in self.session_rows()]
        has_rows = bool(filtered)
        self._select_shown.setEnabled(has_rows)
        self._select_shown.setVisible(has_rows)
        all_selected = has_rows and all(item.id in self._selected_ids for item in filtered)
        self._select_shown.setText(self.tr("Clear") if all_selected else self.tr("Select all"))
        for row in self.session_rows():
            row.set_selection_mode(selecting)

    def _toggle_shown(self) -> None:
        rows = self.session_rows()
        if not rows:
            return
        ids = {row.summary.id for row in rows}
        if ids.issubset(self._selected_ids):
            self._selected_ids.difference_update(ids)
        else:
            self._selected_ids.update(ids)
        for row in rows:
            row.set_selected(row.summary.id in self._selected_ids)
        self._update_selection_controls()

    def _on_mark_clicked(self, session_id: str) -> None:
        for row in self.session_rows():
            if row.summary.id != session_id:
                continue
            if row.checked:
                self._selected_ids.add(session_id)
            else:
                self._selected_ids.discard(session_id)
            row.selected = row.checked
            self._update_selection_controls()
            return

    def _toggle_row(self, row: SessionRow) -> None:
        session_id = row.summary.id
        if session_id in self._selected_ids:
            self._selected_ids.remove(session_id)
        else:
            self._selected_ids.add(session_id)
        row.set_selected(session_id in self._selected_ids)
        self._update_selection_controls()

    def _open_current(self) -> None:
        summary = self.summary
        if summary is not None:
            self._open_summary(summary)

    def _open_id(self, session_id: str) -> None:
        for row in self.session_rows():
            if row.summary.id == session_id:
                self._open_summary(row.summary)
                return

    def _open_summary(self, summary: SessionSummary) -> None:
        saved = self._session_config_loader(summary.id)
        if not (saved or self._default_config).available:
            self.llm_required.emit(summary.id)
            return
        self.resume_session.emit(summary.id)

    def _configure_current(self) -> None:
        summary = self.summary
        if summary is not None:
            self.configure_session.emit(summary.id)
