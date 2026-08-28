"""Qt-side ChatGPT login and disconnect orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from kimi_cli.auth.codex import (
    AUTH_DISCONNECTED,
    CodexAuthSnapshot,
    CodexBrowserChallenge,
    CodexModelCatalog,
)
from PySide6.QtWidgets import QDialog, QWidget

from kimix_gui.llm.registry import ModelCatalogService
from kimix_gui.qt.codex_dialog import CodexLoginDialog, DisconnectChatGPTDialog


class CodexBridge(Protocol):
    """Account operations exposed by the worker-thread bridge."""

    codex_auth_changed: Any
    codex_browser_challenge: Any
    codex_catalog_changed: Any

    @property
    def uses_chatgpt(self) -> bool: ...

    def connect_chatgpt(self) -> int: ...

    def cancel_chatgpt_login(self, operation_id: int) -> None: ...

    def refresh_codex_models(self) -> int: ...

    def disconnect_chatgpt(self, *, close_active_session: bool = False) -> int: ...


class CodexController:
    """Own transient Codex dialogs while the pure catalog service owns state."""

    def __init__(
        self,
        bridge: CodexBridge,
        catalogs: ModelCatalogService,
        *,
        default_parent: Callable[[], QWidget | None],
        active_session_disconnected: Callable[[], None],
    ) -> None:
        self._bridge = bridge
        self._catalogs = catalogs
        self._default_parent = default_parent
        self._active_session_disconnected = active_session_disconnected
        self._login_dialog: CodexLoginDialog | None = None
        self._disconnecting_active = False
        bridge.codex_auth_changed.connect(self.on_auth_changed)
        bridge.codex_browser_challenge.connect(self.on_browser_challenge)
        bridge.codex_catalog_changed.connect(self.on_catalog_changed)

    @property
    def login_dialog(self) -> CodexLoginDialog | None:
        return self._login_dialog

    def connect(self, *, parent: QWidget | None = None) -> None:
        dialog = self._login_dialog
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        owner = parent or self._default_parent()
        dialog = CodexLoginDialog(parent=owner)
        self._login_dialog = dialog
        dialog.retry_requested.connect(lambda: self._retry(dialog))
        dialog.cancel_requested.connect(self._bridge.cancel_chatgpt_login)
        dialog.finished.connect(lambda: self._clear_login_dialog(dialog))
        dialog.open()
        dialog.begin(self._bridge.connect_chatgpt())

    def refresh_models(self) -> None:
        self._bridge.refresh_codex_models()

    def disconnect(self, *, parent: QWidget | None = None) -> None:
        if not self._bridge.uses_chatgpt:
            self._bridge.disconnect_chatgpt()
            return
        owner = parent or self._default_parent()
        dialog = DisconnectChatGPTDialog(parent=owner)

        def done(result: int) -> None:
            try:
                if result == int(QDialog.DialogCode.Accepted):
                    self._disconnecting_active = True
                    self._bridge.disconnect_chatgpt(close_active_session=True)
            finally:
                dialog.deleteLater()

        dialog.finished.connect(done)
        dialog.open()

    def on_browser_challenge(self, challenge: object) -> None:
        if not isinstance(challenge, CodexBrowserChallenge):
            return
        dialog = self._login_dialog
        if dialog is not None:
            dialog.show_challenge(challenge)

    def on_auth_changed(self, snapshot: object) -> None:
        if not isinstance(snapshot, CodexAuthSnapshot):
            return
        if not self._catalogs.update_codex_auth(snapshot):
            return
        dialog = self._login_dialog
        if dialog is not None:
            dialog.set_snapshot(snapshot)
        if snapshot.state == AUTH_DISCONNECTED and self._disconnecting_active:
            self._disconnecting_active = False
            self._active_session_disconnected()

    def on_catalog_changed(self, catalog: object) -> None:
        if isinstance(catalog, CodexModelCatalog):
            self._catalogs.update_codex_catalog(catalog)

    def _retry(self, dialog: CodexLoginDialog) -> None:
        if self._login_dialog is dialog:
            dialog.begin(self._bridge.connect_chatgpt())

    def _clear_login_dialog(self, dialog: CodexLoginDialog) -> None:
        if self._login_dialog is dialog:
            self._login_dialog = None
        dialog.deleteLater()
