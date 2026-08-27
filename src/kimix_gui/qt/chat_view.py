"""Chat view: transcript, composer, history toolbar, and Kimix bridge wiring."""

from __future__ import annotations

from contextlib import suppress
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.qt import keys
from kimix_gui.qt.bridge import (
    HistoryPage,
    KimixBridge,
    StatusLineUpdate,
    TodoUpdate,
)
from kimix_gui.qt.composer import Composer, ComposerPad
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.status_line import format_status_line
from kimix_gui.qt.styling import Metric, Role, Surface, Tone, Variant, style
from kimix_gui.qt.todo_panel import TodoPanel
from kimix_gui.qt.transcript import Transcript
from kimix_gui.rendering import StatusValues
from kimix_gui.transcript_data import (
    ClearTranscript,
    NoticeEntry,
    StartEntry,
    TextBlock,
    TranscriptUpdate,
    literal,
)

# What the status line is currently reporting. Stored instead of the sentence it
# produces, so the sentence can be produced again in another language.
SessionState = Literal["connecting", "ready", "running", "cancelling", "unavailable"]


class ChatView(QWidget):
    """Run one SDK session inside a full-window chat interface."""

    leave_requested = Signal()
    open_settings = Signal()
    approval_asked = Signal(object)
    question_asked = Signal(object)
    notify = Signal(str, str, str)

    def __init__(self, bridge: KimixBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chat-view")
        self.bridge = bridge
        self._epoch = 0
        self._pending_config_label: str | None = None
        self._history_total = 0
        self._history_loading = False
        self._history_ready = False
        self._session_state: SessionState = "connecting"
        self._session_id_text = ""
        self._context_text = ""
        self._pad: ComposerPad | None = None
        self._i18n = Retranslator(self)
        self._build()
        self._connect_bridge()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._header())

        self.transcript = Transcript()
        root.addWidget(self.transcript, 1)
        # The panel floats inside the transcript and keeps its own position; handing it
        # the host is the whole contract.
        self.todo_panel = TodoPanel(self.transcript)

        root.addWidget(self._composer_dock())

        self._older.clicked.connect(self.load_older_history)
        self._newer.clicked.connect(self.load_newer_history)
        self._latest.clicked.connect(self.jump_to_latest)
        self._turn_input.returnPressed.connect(self._submit_turn)
        self.prompt.submitted.connect(self._submit_prompt)
        self.prompt.textChanged.connect(self._sync_composer_actions)
        self.prompt.expand_requested.connect(self._open_composer_pad)
        self._send.clicked.connect(self._send_prompt)
        self._cancel.clicked.connect(self.cancel_prompt)
        self.transcript.reached_top.connect(self._on_reached_top)
        self.transcript.reached_bottom.connect(self._on_reached_bottom)
        self.transcript.viewport_turn_changed.connect(lambda _turn: self._update_history_toolbar())
        self.transcript.record_copied.connect(
            lambda label: self.notify.emit(
                self.tr("{label} message copied").format(label=label), "information", ""
            )
        )
        # The status line and the history label are derived from state rather than
        # phrased on arrival, so binding the two derivations both seeds them now and
        # re-derives them on a language change.
        self._i18n.bind(self._refresh_status)
        self._i18n.bind(self._refresh_history_copy)
        keys.install(
            self,
            keys.CHAT,
            {
                "leave-session": self.leave_requested.emit,
                "focus-prompt": self.focus_prompt,
                "focus-history-turn": self.focus_history_turn,
                "session-settings": self.open_settings.emit,
                "cancel-generation": self.cancel_prompt,
                "load-older-history": self.load_older_history,
                "jump-to-latest": self.jump_to_latest,
                "toggle-todos": self.toggle_todo_panel,
            },
        )
        self._sync_composer_actions()

    def _header(self) -> QFrame:
        """Top bar: the chat toolbar stacked over the history navigation strip."""
        header = QFrame()
        header.setObjectName("chat-header")
        style(header, surface=Surface.BAR)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)
        header_layout.addWidget(self._toolbar())
        header_layout.addWidget(self._history_toolbar())
        return header

    def _toolbar(self) -> QFrame:
        """Chat toolbar: session title, status line, and the settings / home actions."""
        toolbar = QFrame()
        toolbar.setObjectName("chat-toolbar")
        style(toolbar, surface=Surface.BAR)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        title = QLabel()
        title.setObjectName("chat-title")
        style(title, role=Role.TITLE)
        self._i18n.bind(lambda: title.setText(self.tr("CHAT")))
        self._status = QLabel()
        self._status.setObjectName("status")
        style(self._status, tone=Tone.MUTED)
        settings = QPushButton()
        settings.setObjectName("open-settings")
        self._i18n.bind(lambda: settings.setText(self.tr("Settings")))
        home = QPushButton()
        home.setObjectName("leave-session")
        self._i18n.bind(lambda: home.setText(self.tr("Home")))
        toolbar_layout.addWidget(title)
        toolbar_layout.addWidget(self._status, 1)
        toolbar_layout.addWidget(settings)
        toolbar_layout.addWidget(home)
        # Connected here rather than with the rest of the wiring in ``_build``: both
        # buttons are locals, so this is the last place that can still reach them.
        settings.clicked.connect(self.open_settings.emit)
        home.clicked.connect(self.leave_requested.emit)
        return toolbar

    def _history_toolbar(self) -> QFrame:
        """History strip: how far back the transcript is, and the controls that move it."""
        history = QFrame()
        history.setObjectName("history-toolbar")
        style(history, surface=Surface.BAR)
        history_layout = QHBoxLayout(history)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        self._history_info = QLabel()
        self._history_info.setObjectName("history-info")
        style(self._history_info, tone=Tone.MUTED)
        self._older = QPushButton("←")
        self._older.setObjectName("load-older")
        style(self._older, metric=Metric.NAV)
        self._i18n.bind(lambda: self._older.setToolTip(self.tr("Previous turn")))
        self._older.setFixedHeight(DARK.sizing.compact_control_height)
        self._turn_input = QLineEdit()
        self._turn_input.setObjectName("history-turn")
        self._turn_input.setValidator(QIntValidator(1, 1_000_000, self))
        self._turn_input.setEnabled(False)
        self._turn_input.setFixedWidth(72)
        self._turn_input.setFixedHeight(DARK.sizing.compact_control_height)
        self._i18n.bind(lambda: self._turn_input.setToolTip(self.tr("Seek to turn")))
        self._newer = QPushButton("→")
        self._newer.setObjectName("load-newer")
        style(self._newer, metric=Metric.NAV)
        self._newer.setEnabled(False)
        self._i18n.bind(lambda: self._newer.setToolTip(self.tr("Next turn")))
        self._newer.setFixedHeight(DARK.sizing.compact_control_height)
        self._latest = QPushButton("↓")
        self._latest.setObjectName("jump-latest")
        style(self._latest, metric=Metric.NAV)
        self._latest.setEnabled(False)
        self._i18n.bind(lambda: self._latest.setToolTip(self.tr("Jump to latest")))
        self._latest.setFixedHeight(DARK.sizing.compact_control_height)
        history_layout.addWidget(self._history_info, 1)
        history_layout.addWidget(self._older)
        history_layout.addWidget(self._turn_input)
        history_layout.addWidget(self._newer)
        history_layout.addWidget(self._latest)
        return history

    def _composer_dock(self) -> QFrame:
        """Bottom dock: the prompt with its send / cancel actions and the context readout."""
        dock = QFrame()
        dock.setObjectName("composer-dock")
        dock_layout = QVBoxLayout(dock)
        dock_layout.setContentsMargins(16, 8, 16, 8)
        dock_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.prompt = Composer()
        self._i18n.bind(
            lambda: self.prompt.setPlaceholderText(
                self.tr("Ask AI, or type /help. Enter to send · Ctrl+Enter for a new line")
            )
        )
        self.prompt.setEnabled(False)
        self._cancel = QPushButton()
        self._cancel.setObjectName("cancel-prompt")
        style(self._cancel, variant=Variant.DANGER, metric=Metric.ACTION)
        self._cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._i18n.bind(lambda: self._cancel.setText(self.tr("Cancel")))
        self._i18n.bind(lambda: self._cancel.setToolTip(self.tr("Stop generation")))
        self._cancel.setFixedHeight(Composer.ACTION_HEIGHT)
        self._send = QPushButton()
        self._send.setObjectName("send-prompt")
        style(self._send, variant=Variant.PRIMARY, metric=Metric.ACTION)
        self._send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._i18n.bind(lambda: self._send.setText(self.tr("Send")))
        self._i18n.bind(lambda: self._send.setToolTip(self.tr("Send message")))
        self._send.setFixedHeight(Composer.ACTION_HEIGHT)
        row.addWidget(self.prompt, 1)
        row.addWidget(self._cancel, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)
        dock_layout.addLayout(row)
        self._context = QLabel("")
        self._context.setObjectName("context")
        style(self._context, tone=Tone.MUTED)
        self._context.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dock_layout.addWidget(self._context)
        return dock

    def _connect_bridge(self) -> None:
        bridge = self.bridge
        bridge.session_opened.connect(self._on_session_opened)
        bridge.session_failed.connect(self._on_session_failed)
        bridge.transcript_updated.connect(self._on_update)
        bridge.status_changed.connect(self._on_status_changed)
        bridge.history_page.connect(self._on_history_page)
        bridge.history_loading.connect(self._on_history_loading)
        bridge.todos_changed.connect(self._on_todos_changed)
        bridge.input_enabled.connect(self._on_input_enabled)
        bridge.approval_asked.connect(self._forward_approval)
        bridge.question_asked.connect(self._forward_question)
        bridge.generation_started.connect(self._on_generation_started)
        bridge.generation_finished.connect(self._on_generation_finished)

    def disconnect_bridge(self) -> None:
        bridge = self.bridge
        for signal, slot in (
            (bridge.session_opened, self._on_session_opened),
            (bridge.session_failed, self._on_session_failed),
            (bridge.transcript_updated, self._on_update),
            (bridge.status_changed, self._on_status_changed),
            (bridge.history_page, self._on_history_page),
            (bridge.history_loading, self._on_history_loading),
            (bridge.todos_changed, self._on_todos_changed),
            (bridge.input_enabled, self._on_input_enabled),
            (bridge.approval_asked, self._forward_approval),
            (bridge.question_asked, self._forward_question),
            (bridge.generation_started, self._on_generation_started),
            (bridge.generation_finished, self._on_generation_finished),
        ):
            with suppress(RuntimeError, TypeError):
                signal.disconnect(slot)
        if self._pad is not None:
            self._pad.close()
            self._pad = None

    def take_keyboard_focus(self) -> None:
        """Make this page the keyboard's target: the prompt, where typing belongs.

        Unlike ``focus_prompt`` this does not care whether a generation is running --
        the point is that the focus is somewhere inside the page, because that is what
        lets the page's own key bindings fire. See ``keys.ensure_focus``.
        """
        keys.ensure_focus(self, self.prompt)

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        self.take_keyboard_focus()

    def _forward_approval(self, ask: object) -> None:
        self.approval_asked.emit(ask)

    def _forward_question(self, ask: object) -> None:
        self.question_asked.emit(ask)

    @property
    def busy(self) -> bool:
        return self.bridge.busy

    @property
    def session_id(self) -> str | None:
        return self.bridge.session_id

    @property
    def prompt_enabled(self) -> bool:
        return self.prompt.isEnabled()

    def set_pending_config(self, label: str) -> None:
        self._pending_config_label = label
        self._refresh_status()

    def reset_session_label(self) -> None:
        """Put the status line back to its pre-connection text.

        Exists so ``KimixGuiApp`` does not have to spell the copy a second time:
        it is not a ``QObject`` and cannot call ``tr()``.
        """

        self._set_session_state("connecting")

    def focus_prompt(self) -> None:
        if not self.busy:
            self.prompt.setFocus()

    def focus_history_turn(self) -> None:
        if self._turn_input.isEnabled():
            self._turn_input.setFocus()

    def cancel_prompt(self) -> None:
        if self.busy:
            self.bridge.cancel_prompt()
            self._set_session_state("cancelling")
            self._sync_composer_actions()
        else:
            self.focus_prompt()

    def _send_prompt(self) -> None:
        self._submit_prompt(self.prompt.text)

    def _open_composer_pad(self) -> None:
        if self._pad is not None:
            self._pad.raise_()
            self._pad.activateWindow()
            return
        pad = ComposerPad(
            self.prompt.text,
            running=self.busy,
            enabled=self.prompt.isEnabled(),
            parent=self,
        )
        self._pad = pad
        pad.submitted.connect(self._submit_prompt)
        pad.cancelled.connect(self.cancel_prompt)
        pad.finished.connect(self._on_pad_finished)
        pad.open()

    def _on_pad_finished(self, _result: int) -> None:
        pad = self._pad
        self._pad = None
        if pad is None:
            return
        if not pad.sent:
            self.prompt.text = pad.text
        self._sync_composer_actions()
        self.focus_prompt()

    def load_older_history(self) -> None:
        self.bridge.load_older(self._display_turn())

    def load_newer_history(self) -> None:
        self.bridge.load_newer(self._display_turn())

    def jump_to_latest(self) -> None:
        self.transcript.jump_to_latest()
        self.bridge.jump_to_latest()

    def jump_to_history_turn(self, turn: int) -> None:
        self.bridge.jump_to_turn(turn)

    def _submit_prompt(self, text: str) -> None:
        stripped = text.strip()
        if not stripped or self.session_id is None or self.busy:
            return
        self.prompt.clear()
        if stripped.startswith("/"):
            if stripped.partition(" ")[0] == "/quit":
                self.leave_requested.emit()
                return
            self.bridge.run_command(stripped)
            return
        self.bridge.run_prompt(stripped)

    def _submit_turn(self) -> None:
        value = self._turn_input.text().strip()
        if not value:
            return
        try:
            turn = int(value)
        except ValueError:
            self.notify.emit(self.tr("Enter a numeric turn"), "warning", "")
            return
        if turn < 1 or turn > self._history_total:
            self.notify.emit(
                self.tr("Turn must be between 1 and {total}").format(total=self._history_total),
                "warning",
                "",
            )
            return
        self.jump_to_history_turn(turn)

    def _on_session_opened(self, session_id: str, status: object, epoch: int) -> None:
        self._epoch = epoch
        self._session_id_text = session_id
        self._context_text = (
            format_status_line(status) if isinstance(status, StatusValues) else str(status)
        )
        self._set_session_state("ready")
        self._context.setText(self._context_text)

    def _on_session_failed(self, message: str, epoch: int) -> None:
        self._epoch = epoch
        self.transcript.apply_mutation(
            StartEntry(
                NoticeEntry(
                    key=f"session-error:{epoch}",
                    kind="error",
                    blocks=(TextBlock(literal(message)),),
                )
            )
        )
        self._set_session_state("unavailable")

    def _on_update(self, update: object) -> None:
        if not isinstance(update, TranscriptUpdate) or update.epoch != self.bridge.epoch:
            return
        self.transcript.apply_mutation(update.mutation)
        if isinstance(update.mutation, ClearTranscript):
            self._history_total = 0
            self._update_history_toolbar()

    def _on_status_changed(self, update: object) -> None:
        if not isinstance(update, StatusLineUpdate) or update.epoch != self.bridge.epoch:
            return
        self._context_text = format_status_line(update.values)
        self._context.setText(self._context_text)

    def _on_history_page(self, page: object) -> None:
        if not isinstance(page, HistoryPage) or page.epoch != self.bridge.epoch:
            return
        if page.entries:
            self.transcript.replace_history(
                page.entries,
                target_turn=page.target_turn,
                pin_latest=page.pin_latest,
            )
            self._history_total = page.total_turns
            if page.pin_latest:
                self.transcript.jump_to_latest()
            elif page.target_turn is not None:
                self.transcript.jump_to_turn(page.target_turn)
        elif page.pin_latest:
            self.transcript.jump_to_latest()
            self._history_total = page.total_turns
        else:
            self.transcript.mark_history_window()
            self._history_total = page.total_turns
        self._update_history_toolbar()

    def _on_history_loading(self, loading: bool, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self._history_loading = loading
        self._update_history_toolbar()

    def _on_todos_changed(self, update: object) -> None:
        if not isinstance(update, TodoUpdate) or update.epoch != self.bridge.epoch:
            return
        self.todo_panel.set_snapshot(update.snapshot)

    def toggle_todo_panel(self) -> None:
        """Collapse or expand the todo panel when the session has todos."""

        if self.todo_panel.isVisible():
            self.todo_panel.toggle()

    def _on_generation_started(self, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        if self.session_id:
            self._session_id_text = self.session_id
            self._set_session_state("running")
        self._sync_composer_actions()

    def _on_input_enabled(self, enabled: bool, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self.prompt.setEnabled(enabled)
        self._sync_composer_actions()
        if enabled:
            self.prompt.setFocus()
        if enabled and self.session_id:
            self._session_id_text = self.session_id
            self._set_session_state("ready")
            if self._context_text:
                self._context.setText(self._context_text)

    def _on_generation_finished(self, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self._sync_composer_actions()

    def _on_reached_top(self) -> None:
        if self._history_total <= 0 or self._history_loading:
            return
        self.bridge.prefetch_older()

    def _on_reached_bottom(self) -> None:
        if self._history_total <= 0 or self._history_loading:
            return
        self.bridge.prefetch_newer()

    def _sync_composer_actions(self) -> None:
        running = self.busy
        source = self._pad.text if self._pad is not None else self.prompt.text
        has_text = bool(source.strip())
        self._send.setVisible(not running)
        self._send.setEnabled(self.prompt.isEnabled() and has_text and not running)
        self._cancel.setVisible(running)
        self._cancel.setEnabled(running)
        if self._pad is not None:
            self._pad.set_running(running)
            self._pad.set_enabled(self.prompt.isEnabled())

    def _display_turn(self) -> int:
        total = self._history_total
        if self.transcript.pinned_to_latest:
            return total
        viewport = self.transcript.viewport_turn()
        if viewport is not None:
            return viewport + 1
        return total

    def _set_session_state(self, state: SessionState) -> None:
        self._session_state = state
        self._refresh_status()

    def _session_label(self) -> str:
        """The session half of the status line, without the pending-config suffix."""

        state = self._session_state
        if state == "connecting":
            return self.tr("connecting…")
        if state == "unavailable":
            return self.tr("session unavailable")
        if state == "running":
            return self.tr("session {id} · running").format(id=self._session_id_text)
        if state == "cancelling":
            return self.tr("{session} · cancelling…").format(
                session=self.tr("session {id}").format(id=self._session_id_text)
            )
        return self.tr("session {id}").format(id=self._session_id_text)

    def _refresh_status(self) -> None:
        """Derive the status line from state, so it can be derived again later.

        Every branch used to be phrased at the moment its event arrived and then
        stored as a finished sentence, which meant a language change had nothing left
        to re-phrase. The sentence is now a function of ``_session_state``, so the
        retranslator can call this and get the same words in the new language.
        """

        text = self._session_label()
        if self._pending_config_label:
            text = self.tr("{status} · next: {config}").format(
                status=text, config=self._pending_config_label
            )
        self._status.setText(text)

    def _refresh_history_copy(self) -> None:
        """Derive the history label and the turn placeholder from state.

        Split out of ``_update_history_toolbar`` so the retranslator can re-run the
        copy without also re-running the button enablement, which depends on scroll
        position and would fight whatever the user is doing at that moment.
        """

        if not self._history_ready:
            self._history_info.setText(self.tr("History · connecting…"))
        elif self._history_loading:
            self._history_info.setText(self.tr("History · loading…"))
        elif self._history_total <= 0:
            self._history_info.setText(self.tr("History · no turns yet"))
        else:
            self._history_info.setText(
                self.tr("History · Turn {current} of {total}").format(
                    current=self._display_turn(), total=self._history_total
                )
            )
        if self._history_total <= 0:
            self._turn_input.setPlaceholderText(self.tr("Turn #"))
        else:
            self._turn_input.setPlaceholderText(
                self.tr("Turn 1-{total}").format(total=self._history_total)
            )

    def _update_history_toolbar(self) -> None:
        self._history_ready = True
        self._refresh_history_copy()
        if self._history_total <= 0:
            self._older.setEnabled(False)
            self._turn_input.setEnabled(False)
            self._newer.setEnabled(False)
            self._latest.setEnabled(False)
            return
        current = self._display_turn()
        self._older.setEnabled(not self._history_loading and current > 1)
        self._turn_input.setEnabled(not self._history_loading)
        self._newer.setEnabled(not self._history_loading and current < self._history_total)
        self._latest.setEnabled(
            not self._history_loading
            and not (current >= self._history_total and self.transcript.pinned_to_latest)
        )
