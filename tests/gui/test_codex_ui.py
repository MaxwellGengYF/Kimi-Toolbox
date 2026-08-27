from __future__ import annotations

import asyncio
import time

import pytest
from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    AUTH_DISCONNECTED,
    AUTH_RETRY_LATER,
    PROBLEM_RATE_LIMITED,
    CodexAuthSnapshot,
    CodexBrowserChallenge,
    CodexModel,
    CodexModelCatalog,
    CodexProblem,
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QPushButton, QWidget

from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.llm_config import ChatGPTModelReference, ChatGPTSource
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.bridge import KimixBridge
from kimix_gui.qt.codex_dialog import CodexLoginDialog
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog, LLMSettingsResult

from .qtutil import find, widget_text


def _catalog(*, stale: bool = False) -> CodexModelCatalog:
    return CodexModelCatalog(
        operation_id=3,
        models=(CodexModel("gpt-5.4"), CodexModel("gpt-5.6-sol")),
        stale=stale,
    )


def test_preferences_account_card_exposes_connected_actions(qtbot) -> None:
    dialog = PreferencesDialog(
        InterfacePreferences(),
        font_families=list,
        codex_snapshot=CodexAuthSnapshot(3, AUTH_CONNECTED, model_count=2, stale=False),
        codex_catalog=_catalog(),
    )
    qtbot.addWidget(dialog)
    dialog.show_category(dialog.CATEGORY_MODELS)
    dialog.show()
    refreshed: list[bool] = []
    disconnected: list[bool] = []
    dialog.refresh_codex_models.connect(lambda: refreshed.append(True))
    dialog.disconnect_chatgpt.connect(lambda: disconnected.append(True))

    assert "Connected · 2 models" in widget_text(dialog, "codex-account-status")
    assert find(dialog, "connect-chatgpt", QPushButton).isHidden()
    find(dialog, "refresh-codex-models", QPushButton).click()
    find(dialog, "disconnect-chatgpt", QPushButton).click()

    assert refreshed == [True]
    assert disconnected == [True]


def test_login_dialog_opens_browser_can_reopen_and_retry(
    qtbot,
    monkeypatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    dialog = CodexLoginDialog()
    qtbot.addWidget(dialog)
    dialog.begin(8)
    dialog.show()
    challenge = CodexBrowserChallenge(
        operation_id=8,
        authorization_url="https://auth.openai.com/oauth/authorize?state=opaque",
        expires_at=time.time() + 120,
    )

    dialog.show_challenge(challenge)
    find(dialog, "open-codex-browser", QPushButton).click()

    assert opened == [
        "https://auth.openai.com/oauth/authorize?state=opaque",
        "https://auth.openai.com/oauth/authorize?state=opaque",
    ]
    assert dialog.findChild(QPushButton, "copy-codex-code") is None
    assert "Waiting for browser authorization" in widget_text(dialog, "codex-login-status")

    retries: list[bool] = []
    dialog.retry_requested.connect(lambda: retries.append(True))
    dialog.set_snapshot(
        CodexAuthSnapshot(
            8,
            AUTH_RETRY_LATER,
            problem=CodexProblem(PROBLEM_RATE_LIMITED, retry_after=4),
        )
    )
    retry = find(dialog, "retry-codex-login", QPushButton)
    assert retry.isVisible()
    retry.click()
    assert retries == [True]


def test_login_dialog_cancel_carries_operation_id(qtbot) -> None:
    dialog = CodexLoginDialog()
    qtbot.addWidget(dialog)
    dialog.begin(11)
    dialog.show()
    cancelled: list[int] = []
    dialog.cancel_requested.connect(cancelled.append)

    find(dialog, "cancel-codex-login", QPushButton).click()

    assert cancelled == [11]


def test_builtin_chatgpt_model_applies_only_after_explicit_click(qtbot) -> None:
    reference = ChatGPTModelReference(
        model_name="gpt-5.4",
        supported_efforts=("low", "medium", "high", "xhigh"),
        default_reasoning_effort="medium",
        available=True,
        stale=False,
    )
    dialog = LLMSettingsDialog(
        current=reference,
        references=(reference,),
        scope_label="New sessions",
        manage_library=True,
        chatgpt_connected=True,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    applied: list[LLMSettingsResult] = []
    dialog.applied.connect(applied.append)

    assert applied == []
    assert dialog.selected_config() == reference
    assert not find(dialog, "delete-config", QPushButton).isEnabled()
    assert widget_text(dialog, "config-context") == "272,000 tokens"
    assert widget_text(dialog, "config-output") == "128,000 tokens"
    assert widget_text(dialog, "config-thinking") == (
        "default medium · efforts low, medium, high, xhigh"
    )
    find(dialog, "apply-settings", QPushButton).click()

    assert applied == [LLMSettingsResult(reference)]


def test_disconnected_model_picker_offers_login_without_making_model_available(
    qtbot,
) -> None:
    reference = ChatGPTModelReference(model_name="gpt-5.4")
    dialog = LLMSettingsDialog(
        current=reference,
        references=(),
        scope_label="New sessions",
        chatgpt_connected=False,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    requested: list[bool] = []
    dialog.connect_chatgpt.connect(lambda: requested.append(True))

    assert not find(dialog, "apply-settings", QPushButton).isEnabled()
    assert any("Connect ChatGPT" in item.text() for item in dialog.config_items())
    find(dialog, "connect-chatgpt-models", QPushButton).click()

    assert requested == [True]


def test_connected_account_marks_absent_saved_model_unavailable(qtbot) -> None:
    reference = ChatGPTModelReference(
        model_name="retired-model",
        available=False,
        problem_code="model_unavailable",
    )
    dialog = LLMSettingsDialog(
        current=reference,
        references=(),
        scope_label="Saved session",
        chatgpt_connected=True,
    )
    qtbot.addWidget(dialog)
    dialog.set_chatgpt_references((), connected=True)
    dialog.show()

    assert "not available" in widget_text(dialog, "settings-error")
    assert find(dialog, "connect-chatgpt-models", QPushButton).isHidden()
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_saved_chatgpt_session_waits_for_initial_model_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    app = KimixGuiApp(
        SessionOptions(
            tmp_path,
            session_id="saved-session",
            llm_source=ChatGPTSource("account-only-model"),
        )
    )
    app._pending_codex_startup = True
    started_with_available_model: list[bool] = []
    monkeypatch.setattr(
        app,
        "_startup",
        lambda: started_with_available_model.append(app.default_config.available),
    )

    app.on_codex_auth_changed(CodexAuthSnapshot(9, AUTH_CONNECTED))

    assert started_with_available_model == []
    app.on_codex_catalog_changed(
        CodexModelCatalog(
            operation_id=9,
            models=(CodexModel("account-only-model"),),
            stale=False,
        )
    )
    assert started_with_available_model == [True]


def test_active_chatgpt_disconnect_requires_confirmation(qtbot, tmp_path) -> None:
    calls: list[bool] = []

    class Bridge:
        uses_chatgpt = True

        def disconnect_chatgpt(self, *, close_active_session: bool = False) -> int:
            calls.append(close_active_session)
            return 1

        def stop(self) -> None:
            return None

    app = KimixGuiApp(SessionOptions(tmp_path))
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    app.window = window  # type: ignore[assignment]
    app.bridge = Bridge()  # type: ignore[assignment]

    app.disconnect_chatgpt()
    dialog = find(window, "disconnect-chatgpt-dialog")
    assert dialog.isVisible()
    assert calls == []

    find(dialog, "confirm-disconnect-chatgpt", QPushButton).click()
    assert calls == [True]


@pytest.mark.asyncio
async def test_account_operations_serialize_refresh_before_disconnect() -> None:
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    events: list[str] = []

    class Service:
        async def refresh_models(self, operation_id: int) -> CodexModelCatalog:
            events.append("refresh-started")
            refresh_started.set()
            await release_refresh.wait()
            events.append("refresh-saved")
            return CodexModelCatalog(operation_id, (CodexModel("model"),), False)

        async def snapshot(self, operation_id: int) -> CodexAuthSnapshot:
            return CodexAuthSnapshot(operation_id, AUTH_CONNECTED)

        async def disconnect(self, operation_id: int) -> CodexAuthSnapshot:
            events.append("disconnected")
            return CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(
        session_factory=unused_factory,
        codex_service=Service(),  # type: ignore[arg-type]
    )
    refresh_operation = bridge._next_codex_operation()
    refresh_task = asyncio.create_task(bridge._refresh_codex_models(refresh_operation))
    await refresh_started.wait()
    disconnect_operation = bridge._next_codex_operation()
    disconnect_task = asyncio.create_task(
        bridge._disconnect_chatgpt(disconnect_operation, close_active_session=False)
    )
    release_refresh.set()

    await asyncio.gather(refresh_task, disconnect_task)

    assert events == ["refresh-started", "refresh-saved", "disconnected"]
