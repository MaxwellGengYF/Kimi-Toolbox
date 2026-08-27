"""ChatGPT Codex account card and browser OAuth dialogs."""

from __future__ import annotations

import time

from PySide6.QtCore import QCoreApplication, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.codex_auth import (
    AUTH_CONNECTED,
    AUTH_CONNECTING,
    AUTH_LOGIN_REQUIRED,
    AUTH_RETRY_LATER,
    PROBLEM_CALLBACK_UNAVAILABLE,
    PROBLEM_CANCELLED,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_MODEL_UNAVAILABLE,
    PROBLEM_RATE_LIMITED,
    PROBLEM_TIMEOUT,
    CodexAuthSnapshot,
    CodexBrowserChallenge,
    CodexModelCatalog,
    CodexProblem,
)
from kimix_gui.qt.components import Card, DialogFooter
from kimix_gui.qt.styling import CardLevel, Level, Role, Tone, Variant, style


class CodexAccountCard(Card):
    """Preferences summary and actions for the one global ChatGPT account."""

    connect_requested = Signal()
    refresh_requested = Signal()
    disconnect_requested = Signal()

    def __init__(
        self,
        snapshot: CodexAuthSnapshot,
        catalog: CodexModelCatalog,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(CardLevel.INSET, parent=parent)
        self.setObjectName("codex-account-card")
        self._snapshot = snapshot
        self._catalog = catalog
        self.body.setSpacing(8)

        title = QLabel(self.tr("ChatGPT / Codex subscription"))
        title.setObjectName("codex-account-title")
        style(title, role=Role.TITLE)
        description = QLabel(
            self.tr("Use models included with your ChatGPT subscription through the Codex backend.")
        )
        description.setObjectName("codex-account-description")
        style(description, tone=Tone.MUTED)
        description.setWordWrap(True)
        self._status = QLabel()
        self._status.setObjectName("codex-account-status")
        style(self._status, tone=Tone.MUTED)
        self._status.setWordWrap(True)
        self.body.addWidget(title)
        self.body.addWidget(description)
        self.body.addWidget(self._status)

        actions = QHBoxLayout()
        self._connect = QPushButton(self.tr("Connect ChatGPT"))
        self._connect.setObjectName("connect-chatgpt")
        style(self._connect, variant=Variant.PRIMARY)
        self._refresh = QPushButton(self.tr("Refresh models"))
        self._refresh.setObjectName("refresh-codex-models")
        style(self._refresh, variant=Variant.GHOST)
        self._disconnect = QPushButton(self.tr("Disconnect"))
        self._disconnect.setObjectName("disconnect-chatgpt")
        style(self._disconnect, variant=Variant.DANGER)
        actions.addWidget(self._connect)
        actions.addWidget(self._refresh)
        actions.addWidget(self._disconnect)
        actions.addStretch()
        self.body.addLayout(actions)
        self._connect.clicked.connect(self.connect_requested.emit)
        self._refresh.clicked.connect(self.refresh_requested.emit)
        self._disconnect.clicked.connect(self.disconnect_requested.emit)
        self._sync()

    def set_snapshot(self, snapshot: CodexAuthSnapshot) -> None:
        self._snapshot = snapshot
        self._sync()

    def set_catalog(self, catalog: CodexModelCatalog) -> None:
        self._catalog = catalog
        self._sync()

    def _sync(self) -> None:
        snapshot = self._snapshot
        connected = snapshot.state == AUTH_CONNECTED
        self._connect.setVisible(not connected and snapshot.state != AUTH_CONNECTING)
        self._refresh.setVisible(connected)
        self._disconnect.setVisible(connected)
        self._refresh.setEnabled(snapshot.state != AUTH_CONNECTING)
        if snapshot.state == AUTH_CONNECTING:
            text = self.tr("Waiting for ChatGPT authorization…")
        elif connected:
            count = len(self._catalog.models)
            text = (
                self.tr("Connected · 1 model")
                if count == 1
                else self.tr("Connected · {count} models").format(count=count)
            )
            if self._catalog.stale:
                text = self.tr("{status} · cached list").format(status=text)
            if self._catalog.problem is not None:
                text = self.tr("{status} · {problem}").format(
                    status=text,
                    problem=codex_problem_message(self._catalog.problem),
                )
        elif snapshot.state == AUTH_LOGIN_REQUIRED:
            text = self.tr("Sign in again to continue using subscription models.")
        elif snapshot.state == AUTH_RETRY_LATER:
            text = codex_problem_message(snapshot.problem)
        else:
            text = self.tr("Not connected. Your external provider files are unaffected.")
        self._status.setText(text)


class CodexLoginDialog(QDialog):
    """Open and monitor the browser-based ChatGPT OAuth flow."""

    retry_requested = Signal()
    cancel_requested = Signal(int)

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("codex-login-dialog")
        self.setWindowTitle(self.tr("Connect ChatGPT"))
        self.setModal(True)
        self.resize(520, 280)
        self._operation_id = 0
        self._challenge: CodexBrowserChallenge | None = None
        self._opened_operation = 0
        self._login_pending = False
        self._timer = QTimer(self)
        self._timer.setInterval(1_000)
        self._timer.timeout.connect(self._update_countdown)
        self._build()

    @property
    def operation_id(self) -> int:
        return self._operation_id

    def begin(self, operation_id: int) -> None:
        self._operation_id = operation_id
        self._login_pending = True
        self._challenge = None
        self._status.setText(self.tr("Preparing browser sign-in…"))
        self._retry.hide()
        self._open.setEnabled(False)
        self._timer.stop()

    def show_challenge(self, challenge: CodexBrowserChallenge) -> None:
        if challenge.operation_id != self._operation_id:
            return
        self._challenge = challenge
        self._open.setEnabled(True)
        self._timer.start()
        self._update_countdown()
        if self._opened_operation != challenge.operation_id:
            self._opened_operation = challenge.operation_id
            QDesktopServices.openUrl(QUrl(challenge.authorization_url))

    def set_snapshot(self, snapshot: CodexAuthSnapshot) -> None:
        if snapshot.operation_id != self._operation_id:
            return
        if snapshot.state == AUTH_CONNECTED:
            self._login_pending = False
            self._timer.stop()
            self.accept()
            return
        if snapshot.state == AUTH_CONNECTING:
            return
        self._login_pending = False
        self._timer.stop()
        if snapshot.problem is not None and snapshot.problem.code == PROBLEM_CANCELLED:
            self.reject()
        elif snapshot.problem is not None:
            self._status.setText(codex_problem_message(snapshot.problem))
            self._retry.show()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        title = QLabel(self.tr("Connect ChatGPT"))
        title.setObjectName("codex-login-title")
        style(title, role=Role.DISPLAY, level=Level.TWO)
        description = QLabel(
            self.tr(
                "A browser will open. Complete the ChatGPT sign-in there; Kimix will continue automatically."
            )
        )
        description.setObjectName("codex-login-description")
        style(description, tone=Tone.MUTED)
        description.setWordWrap(True)
        self._status = QLabel(self.tr("Preparing browser sign-in…"))
        self._status.setObjectName("codex-login-status")
        style(self._status, tone=Tone.MUTED)
        self._status.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)
        root.addWidget(self._status)

        actions = QHBoxLayout()
        self._open = QPushButton(self.tr("Open sign-in page again"))
        self._open.setObjectName("open-codex-browser")
        style(self._open, variant=Variant.PRIMARY)
        self._open.setEnabled(False)
        actions.addWidget(self._open)
        actions.addStretch()
        root.addLayout(actions)

        cancel = QPushButton(self.tr("Cancel"))
        cancel.setObjectName("cancel-codex-login")
        self._retry = QPushButton(self.tr("Try again"))
        self._retry.setObjectName("retry-codex-login")
        style(self._retry, variant=Variant.PRIMARY)
        self._retry.hide()
        root.addWidget(DialogFooter(dismiss=cancel, extra=(self._retry,), parent=self))
        self._open.clicked.connect(self._open_page)
        self._retry.clicked.connect(self.retry_requested.emit)
        cancel.clicked.connect(self._cancel)

    def _open_page(self) -> None:
        if self._challenge is not None:
            QDesktopServices.openUrl(QUrl(self._challenge.authorization_url))

    def _cancel(self) -> None:
        self.reject()

    def reject(self) -> None:
        if self._login_pending:
            self._login_pending = False
            self.cancel_requested.emit(self._operation_id)
        super().reject()

    def _update_countdown(self) -> None:
        challenge = self._challenge
        if challenge is None:
            return
        remaining = max(0, int(challenge.expires_at - time.time()))
        minutes, seconds = divmod(remaining, 60)
        self._status.setText(
            self.tr("Waiting for browser authorization · {minutes}:{seconds}").format(
                minutes=minutes,
                seconds=f"{seconds:02d}",
            )
        )
        if remaining == 0:
            self._timer.stop()


class DisconnectChatGPTDialog(QDialog):
    """Confirm closing an active subscription-backed session before logout."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("disconnect-chatgpt-dialog")
        self.setWindowTitle(self.tr("Disconnect ChatGPT"))
        self.setModal(True)
        root = QVBoxLayout(self)
        title = QLabel(self.tr("Disconnect ChatGPT?"))
        title.setObjectName("disconnect-chatgpt-title")
        style(title, role=Role.SECTION)
        detail = QLabel(
            self.tr(
                "The active ChatGPT session will be stopped. Saved model selections remain available after you sign in again."
            )
        )
        detail.setObjectName("disconnect-chatgpt-detail")
        detail.setWordWrap(True)
        cancel = QPushButton(self.tr("Cancel"))
        confirm = QPushButton(self.tr("Disconnect"))
        confirm.setObjectName("confirm-disconnect-chatgpt")
        style(confirm, variant=Variant.DANGER)
        root.addWidget(title)
        root.addWidget(detail)
        root.addWidget(DialogFooter(dismiss=cancel, confirm=confirm, parent=self))
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)


def codex_problem_message(problem: CodexProblem | None) -> str:
    if problem is None:
        return QCoreApplication.translate(
            "CodexLoginDialog", "ChatGPT connection is unavailable. Try again later."
        )
    if problem.code == PROBLEM_RATE_LIMITED:
        if problem.retry_after is not None:
            return QCoreApplication.translate(
                "CodexLoginDialog", "Too many requests. Try again in {seconds} seconds."
            ).format(seconds=max(1, int(problem.retry_after)))
        return QCoreApplication.translate(
            "CodexLoginDialog", "Too many requests. Try again shortly."
        )
    if problem.code == PROBLEM_TIMEOUT:
        return QCoreApplication.translate(
            "CodexLoginDialog", "Browser sign-in timed out. Try again."
        )
    if problem.code == PROBLEM_CALLBACK_UNAVAILABLE:
        return QCoreApplication.translate(
            "CodexLoginDialog",
            "Could not start the local sign-in callback. Close other sign-in windows and try again.",
        )
    if problem.code in {
        PROBLEM_LOGIN_REQUIRED,
        "invalid_grant",
        "invalid_token",
        "refresh_token_reused",
    }:
        return QCoreApplication.translate(
            "CodexLoginDialog", "Your ChatGPT sign-in expired. Sign in again to continue."
        )
    if problem.code == PROBLEM_MODEL_UNAVAILABLE:
        return QCoreApplication.translate(
            "CodexLoginDialog", "This model is not available for the connected account."
        )
    return QCoreApplication.translate(
        "CodexLoginDialog", "Could not connect to ChatGPT. Check your network and try again."
    )
