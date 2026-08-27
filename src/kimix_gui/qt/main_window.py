"""Stacked main window: home, chat, and modal dialogs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGraphicsOpacityEffect,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog

if TYPE_CHECKING:
    from kimix_gui.app import KimixGuiApp


class Toast(QLabel):
    """Centered snackbar that auto-hides after a short delay."""

    INFO_MS = 2_200
    WARN_MS = 4_000

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toast")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade.finished.connect(self._on_fade_finished)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._hiding = False
        self.hide()

    def show_message(
        self,
        message: str,
        title: str = "",
        *,
        severity: str = "information",
        duration_ms: int | None = None,
    ) -> None:
        self._hiding = False
        self._fade.stop()
        self._timer.stop()
        self._effect.setOpacity(1.0)
        self.setText(f"{title}\n{message}" if title else message)
        parent = self.parentWidget()
        self.setMaximumWidth(max(240, (parent.width() if parent is not None else 420) - 48))
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        if duration_ms is None:
            duration_ms = self.WARN_MS if severity == "warning" else self.INFO_MS
        if duration_ms > 0:
            self._timer.start(duration_ms)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = max(16, (parent.width() - self.width()) // 2)
        y = max(16, parent.height() - self.height() - 28)
        self.move(x, y)

    def mousePressEvent(self, event: object) -> None:
        if isinstance(event, QMouseEvent):
            self._timer.stop()
            self._fade_out()
            event.accept()
            return
        super().mousePressEvent(event)  # type: ignore[arg-type]

    def _fade_out(self) -> None:
        if not self.isVisible():
            return
        self._hiding = True
        self._fade.stop()
        self._fade.setDuration(DARK.motion.fade_ms)
        self._fade.setStartValue(max(0.0, self._effect.opacity()))
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self) -> None:
        if not self._hiding:
            return
        self.hide()
        self._hiding = False
        self._effect.setOpacity(1.0)


class MainWindow(QMainWindow):
    """Top-level window hosting Home and Chat views."""

    def __init__(self, app: KimixGuiApp) -> None:
        super().__init__()
        self.controller = app
        self.setWindowTitle("Kimix")
        self.resize(1100, 720)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self.home: HomeView | None = None
        self.chat: ChatView | None = None
        self._toast = Toast(self)
        self.controller.bridge.codex_auth_changed.connect(self._on_codex_auth_changed)
        self.controller.bridge.codex_browser_challenge.connect(self._on_codex_browser_challenge)
        self.controller.bridge.codex_catalog_changed.connect(self._on_codex_catalog_changed)
        # No key bindings here. Each page installs its own, scoped to its own focus
        # subtree, which is why there is no longer a set of handlers in this class that
        # begin by asking which page is showing. See ``qt/keys.py``.

    @property
    def current_view(self) -> QWidget | None:
        """What the user is looking at: the dialog on top, else the page on the stack.

        Qt keeps this record, so this class does not. It used to: every site that
        opened a dialog told the window to remember it and told it again to forget --
        six pairs across two files, one of them spelled as a truthiness lambda -- and a
        missed half left ``current_view`` naming a dialog the user had already closed,
        with nothing to notice it.

        ``activeModalWidget()`` is the topmost modal widget in the application.
        Measured on the offscreen platform: it appears on ``open()`` and on ``show()``,
        reports the inner dialog while one is stacked on another, uncovers the outer one
        when that closes, and returns to ``None`` afterwards. Every dialog in this app is
        modal, so there is nothing it misses.

        Two things it reports that the old record did not, both of them true: a dialog
        opened from another dialog (the LLM library reached through Preferences), and the
        composer pad, which no page ever registered.
        """

        modal = QApplication.activeModalWidget()
        if modal is not None:
            return modal
        return self._stack.currentWidget()

    def show_home(self, *, reload: bool = True) -> HomeView:
        created = self.home is None
        if created:
            self.home = HomeView(
                self.controller.options.work_dir,
                default_config=self.controller.default_config,
                session_config_loader=self.controller.session_config,
            )
            self._connect_home(self.home)
            self._stack.addWidget(self.home)
        self._stack.setCurrentWidget(self.home)
        if reload or created:
            self.controller.bridge.load_sessions(self.controller.options.work_dir)
        self.home.refresh_configuration(self.controller.default_config)
        return self.home

    def show_chat(self) -> ChatView:
        self.remove_chat()
        self.chat = ChatView(self.controller.bridge)
        self._connect_chat(self.chat)
        self._stack.addWidget(self.chat)
        self._stack.setCurrentWidget(self.chat)
        return self.chat

    def remove_chat(self) -> None:
        if self.chat is None:
            return
        self.chat.disconnect_bridge()
        chat = self.chat
        self.chat = None
        self._stack.removeWidget(chat)
        chat.deleteLater()

    def _connect_home(self, home: HomeView) -> None:
        home.new_session.connect(self.controller.start_new_session)
        home.resume_session.connect(self.controller.resume_session)
        home.open_settings.connect(self.controller.open_preferences)
        home.configure_session.connect(self.controller.open_llm_settings)
        home.quit_requested.connect(self.close)
        home.delete_requested.connect(self._confirm_delete)
        home.llm_required.connect(self._home_llm_required)
        self.controller.bridge.sessions_listed.connect(home.show_sessions)
        self.controller.bridge.sessions_list_failed.connect(home.show_load_error)
        self.controller.bridge.sessions_deleted.connect(home.apply_deleted)
        self.controller.bridge.notify.connect(self.show_notification)

    def _connect_chat(self, chat: ChatView) -> None:
        chat.leave_requested.connect(self.controller.leave_chat)
        chat.open_settings.connect(self.controller.open_chat_settings)
        chat.approval_asked.connect(self._show_approval)
        chat.question_asked.connect(self._show_question)
        chat.notify.connect(self.show_notification)

    def notify_llm_required(self) -> None:
        """Warn that the selected LLM configuration cannot be used.

        Single owner of this copy. ``KimixGuiApp.start_new_session`` used to hold a
        byte-identical duplicate, and ``KimixGuiApp`` is not a ``QObject``, so it
        cannot call ``tr()`` at all -- keeping the strings here is what lets them be
        translated exactly once.
        """

        self.show_notification(
            self.tr("Select a valid LLM configuration to continue."),
            "warning",
            self.tr("LLM configuration required"),
        )

    def _home_llm_required(self, session_id: object) -> None:
        self.notify_llm_required()
        self.controller.open_llm_settings(session_id if isinstance(session_id, str) else None)

    def _confirm_delete(self, ids: list[str]) -> None:
        dialog = DeleteSessionsDialog(len(ids), self)

        def _done(result: int) -> None:
            if result != int(QDialog.DialogCode.Accepted):
                return
            self.controller.bridge.delete_sessions(self.controller.options.work_dir, ids)
            # Spelled-out singular / plural instead of ``%n``: the msgid is what a
            # clone with no compiled catalog renders, and "(s)" is not English.
            message = (
                self.tr("Deleted 1 session")
                if len(ids) == 1
                else self.tr("Deleted {count} sessions").format(count=len(ids))
            )
            self.show_notification(message, "information", "")

        dialog.finished.connect(_done)
        dialog.open()

    def _show_approval(self, ask: object) -> None:
        from kimix_gui.qt.bridge import ApprovalAsk

        if not isinstance(ask, ApprovalAsk):
            return
        dialog = ApprovalDialog(ask.title, ask.description, self)

        def _done(decision: str) -> None:
            self.controller.bridge.resolve_request(ask.token, ask.epoch, decision)

        dialog.decided.connect(_done)
        dialog.open()

    def _show_question(self, ask: object) -> None:
        from kimix_gui.qt.bridge import QuestionAsk

        if not isinstance(ask, QuestionAsk):
            return
        dialog = QuestionDialog(ask.prompt, ask.body, self)

        def _done(answer: object) -> None:
            self.controller.bridge.resolve_request(ask.token, ask.epoch, answer)

        dialog.answered.connect(_done)
        dialog.open()

    def show_notification(
        self, message: str, severity: str = "information", title: str = ""
    ) -> None:
        self.controller.note(message, severity, title)
        self._toast.show_message(message, title, severity=severity)

    def _on_codex_auth_changed(self, snapshot: object) -> None:
        from kimi_cli.auth.codex import CodexAuthSnapshot

        if isinstance(snapshot, CodexAuthSnapshot):
            self.controller.on_codex_auth_changed(snapshot)

    def _on_codex_browser_challenge(self, challenge: object) -> None:
        from kimi_cli.auth.codex import CodexBrowserChallenge

        if isinstance(challenge, CodexBrowserChallenge):
            self.controller.on_codex_browser_challenge(challenge)

    def _on_codex_catalog_changed(self, catalog: object) -> None:
        from kimi_cli.auth.codex import CodexModelCatalog

        if isinstance(catalog, CodexModelCatalog):
            self.controller.on_codex_catalog_changed(catalog)

    def closeEvent(self, event: object) -> None:
        self.controller.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self._toast.isVisible():
            self._toast.reposition()
