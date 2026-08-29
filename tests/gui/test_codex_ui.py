from __future__ import annotations

import asyncio
import time

import pytest
from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    AUTH_CONNECTING,
    AUTH_DISCONNECTED,
    AUTH_LOGIN_REQUIRED,
    AUTH_RETRY_LATER,
    PROBLEM_CANCELLED,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_LOGIN_SUPERSEDED,
    PROBLEM_RATE_LIMITED,
    CodexAccountState,
    CodexAuthError,
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
from kimix_gui.llm import (
    AXIS_THINKING_EFFORT,
    PROBLEM_MODEL_UNAVAILABLE,
    ChatGPTTarget,
    chatgpt_model_descriptor,
    chatgpt_selection,
    resolve_selection,
    unavailable_model,
)
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
    model = chatgpt_model_descriptor(
        CodexModel(
            "gpt-5.4",
            reasoning_efforts=("low", "medium", "high", "xhigh"),
            default_reasoning_effort="medium",
        ),
        connected=True,
        stale=False,
    )
    selection = chatgpt_selection("gpt-5.4", "medium")
    resolved = resolve_selection(selection, [model])
    dialog = LLMSettingsDialog(
        current=resolved,
        models=(model,),
        scope_label="New sessions",
        manage_library=True,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    applied: list[LLMSettingsResult] = []
    dialog.applied.connect(applied.append)

    assert applied == []
    assert dialog.selected_selection() == selection
    assert not find(dialog, "delete-config", QPushButton).isEnabled()
    assert widget_text(dialog, "model-context") == "272,000 tokens"
    assert widget_text(dialog, "model-output") == "128,000 tokens"
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert picker is not None
    assert picker.currentData() == "medium"
    find(dialog, "apply-settings", QPushButton).click()

    assert applied == [LLMSettingsResult(selection)]


def test_disconnected_model_picker_offers_login_without_making_model_available(
    qtbot,
) -> None:
    model = chatgpt_model_descriptor(CodexModel("gpt-5.4"), connected=False, stale=True)
    selection = chatgpt_selection("gpt-5.4", "medium")
    reference = resolve_selection(selection, [model])
    dialog = LLMSettingsDialog(
        current=reference,
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    requested: list[bool] = []
    dialog.connect_chatgpt.connect(lambda: requested.append(True))

    assert not find(dialog, "apply-settings", QPushButton).isEnabled()
    find(dialog, "connect-chatgpt-models", QPushButton).click()

    assert requested == [True]


def test_connected_account_marks_absent_saved_model_unavailable(qtbot) -> None:
    selection = chatgpt_selection("retired-model", "medium")
    model = unavailable_model(ChatGPTTarget("retired-model"), PROBLEM_MODEL_UNAVAILABLE)
    reference = resolve_selection(selection, [model])
    dialog = LLMSettingsDialog(
        current=reference,
        models=(),
        scope_label="Saved session",
    )
    qtbot.addWidget(dialog)
    dialog.set_models(())
    dialog.show()

    assert "not available" in widget_text(dialog, "settings-error")
    login = dialog.findChild(QPushButton, "connect-chatgpt-models")
    assert login is None or login.isHidden()
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_saved_chatgpt_session_waits_for_initial_model_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    app = KimixGuiApp(
        SessionOptions(
            tmp_path,
            session_id="saved-session",
            llm_selection=chatgpt_selection("account-only-model", "medium"),
        )
    )
    app._pending_catalog_startup = True
    started_with_available_model: list[bool] = []
    monkeypatch.setattr(
        app,
        "_startup",
        lambda: started_with_available_model.append(app.default_config.available),
    )

    app.codex_controller.on_auth_changed(CodexAuthSnapshot(9, AUTH_CONNECTED))

    assert started_with_available_model == []
    app.codex_controller.on_catalog_changed(
        CodexModelCatalog(
            operation_id=9,
            models=(
                CodexModel(
                    "account-only-model",
                    reasoning_efforts=("low", "medium"),
                    default_reasoning_effort="medium",
                ),
            ),
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
    app.codex_controller._bridge = Bridge()  # type: ignore[assignment]

    app.codex_controller.disconnect(parent=window)
    dialog = find(window, "disconnect-chatgpt-dialog")
    assert dialog.isVisible()
    assert calls == []

    find(dialog, "confirm-disconnect-chatgpt", QPushButton).click()
    assert calls == [True]


@pytest.mark.asyncio
async def test_initialize_publishes_one_atomic_account_state(monkeypatch) -> None:
    calls = 0
    problem = CodexProblem(PROBLEM_LOGIN_REQUIRED)

    async def initialize(operation_id: int) -> CodexAccountState:
        nonlocal calls
        calls += 1
        return CodexAccountState(
            CodexAuthSnapshot(
                operation_id,
                AUTH_LOGIN_REQUIRED,
                problem=problem,
            ),
            CodexModelCatalog(
                operation_id,
                (CodexModel("cached"),),
                True,
                problem,
            ),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.initialize_codex_account", initialize)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    auth_states: list[str] = []
    bridge.codex_auth_changed.connect(lambda snapshot: auth_states.append(snapshot.state))
    operation_id = bridge._next_codex_operation()

    await bridge._initialize_codex_locked(operation_id)

    assert calls == 1
    assert auth_states == [AUTH_LOGIN_REQUIRED]


@pytest.mark.asyncio
async def test_account_operations_serialize_refresh_before_disconnect(monkeypatch) -> None:
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    events: list[str] = []

    async def refresh_account(operation_id: int) -> CodexAccountState:
        events.append("refresh-started")
        refresh_started.set()
        await release_refresh.wait()
        events.append("refresh-saved")
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_CONNECTED),
            CodexModelCatalog(operation_id, (CodexModel("model"),), False),
        )

    async def disconnect(operation_id: int) -> CodexAccountState:
        events.append("disconnected")
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED),
            CodexModelCatalog(operation_id, (), True),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.refresh_codex_account", refresh_account)
    monkeypatch.setattr("kimix_gui.qt.bridge.disconnect_codex_account", disconnect)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
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


@pytest.mark.asyncio
async def test_superseded_disconnect_stops_after_releasing_active_session(
    tmp_path,
    monkeypatch,
) -> None:
    release_started = asyncio.Event()
    finish_release = asyncio.Event()
    disconnect_calls: list[int] = []

    async def release_session() -> int:
        release_started.set()
        await finish_release.wait()
        return 7

    async def disconnect(operation_id: int) -> CodexAccountState:
        disconnect_calls.append(operation_id)
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED),
            CodexModelCatalog(operation_id, (), True),
        )

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    bridge._session = object()  # type: ignore[assignment]
    bridge._options = SessionOptions(
        tmp_path,
        llm_selection=chatgpt_selection("gpt-5.4", "medium"),
    )
    monkeypatch.setattr(bridge, "_release_session", release_session)
    monkeypatch.setattr("kimix_gui.qt.bridge.disconnect_codex_account", disconnect)
    operation_id = bridge._next_codex_operation()
    task = asyncio.create_task(
        bridge._disconnect_chatgpt_locked(operation_id, close_active_session=True)
    )
    await release_started.wait()

    bridge._next_codex_operation()
    finish_release.set()
    await task

    assert disconnect_calls == []


@pytest.mark.asyncio
async def test_successful_disconnect_publishes_matching_stale_catalog(monkeypatch) -> None:
    async def disconnect(operation_id: int) -> CodexAccountState:
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED),
            CodexModelCatalog(
                operation_id,
                (CodexModel("fallback-model"),),
                True,
            ),
        )

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    monkeypatch.setattr("kimix_gui.qt.bridge.disconnect_codex_account", disconnect)
    bridge = KimixBridge(session_factory=unused_factory)
    snapshots: list[CodexAuthSnapshot] = []
    catalogs: list[CodexModelCatalog] = []
    bridge.codex_auth_changed.connect(snapshots.append)
    bridge.codex_catalog_changed.connect(catalogs.append)
    operation_id = bridge._next_codex_operation()

    await bridge._disconnect_chatgpt_locked(operation_id, close_active_session=False)

    assert [snapshot.state for snapshot in snapshots] == [AUTH_DISCONNECTED]
    assert [model.slug for model in catalogs[0].models] == ["fallback-model"]
    assert catalogs[0].operation_id == snapshots[0].operation_id
    assert catalogs[0].stale is True


@pytest.mark.asyncio
async def test_bridge_can_cancel_login_before_worker_starts_operation(
    monkeypatch,
) -> None:
    instances: list[object] = []

    class LoginOperation:
        def __init__(self, operation_id: int, _challenge_callback) -> None:
            self.operation_id = operation_id
            self.cancelled = False
            instances.append(self)

        def cancel(self) -> None:
            self.cancelled = True

        async def run(self):
            assert self.cancelled
            raise CodexAuthError(CodexProblem(PROBLEM_CANCELLED))

    async def account_state(operation_id: int) -> CodexAccountState:
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED),
            CodexModelCatalog(operation_id, (), True),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.CodexLoginOperation", LoginOperation)
    monkeypatch.setattr("kimix_gui.qt.bridge.codex_account_state", account_state)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    submitted: list[object] = []
    monkeypatch.setattr(bridge, "submit", submitted.append)
    auth_states: list[str] = []
    bridge.codex_auth_changed.connect(lambda value: auth_states.append(value.state))

    operation_id = bridge.connect_chatgpt()
    bridge.cancel_chatgpt_login(operation_id)
    assert len(submitted) == 1
    await submitted[0]  # type: ignore[misc]

    assert len(instances) == 1
    assert instances[0].cancelled is True  # type: ignore[attr-defined]
    assert bridge._active_codex_login is None
    assert auth_states == [AUTH_CONNECTING, AUTH_DISCONNECTED]


@pytest.mark.asyncio
async def test_superseded_login_publishes_connected_authoritative_account(
    monkeypatch,
) -> None:
    class LoginOperation:
        def __init__(self, operation_id: int) -> None:
            self.operation_id = operation_id

        async def run(self):
            raise CodexAuthError(CodexProblem(PROBLEM_LOGIN_SUPERSEDED))

    async def account_state(operation_id: int) -> CodexAccountState:
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_CONNECTED, model_count=1),
            CodexModelCatalog(
                operation_id,
                (CodexModel("external-account-model"),),
                False,
            ),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.codex_account_state", account_state)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    auth_snapshots: list[CodexAuthSnapshot] = []
    catalogs: list[CodexModelCatalog] = []
    bridge.codex_auth_changed.connect(auth_snapshots.append)
    bridge.codex_catalog_changed.connect(catalogs.append)
    operation_id = bridge._next_codex_operation()

    await bridge._connect_chatgpt_locked(LoginOperation(operation_id))  # type: ignore[arg-type]

    assert [snapshot.state for snapshot in auth_snapshots] == [AUTH_CONNECTED]
    assert [model.slug for model in catalogs[0].models] == ["external-account-model"]
    assert auth_snapshots[0].problem is None


@pytest.mark.asyncio
async def test_refresh_supersedes_queued_login_and_clears_its_handle(
    monkeypatch,
) -> None:
    instances: list[object] = []

    class LoginOperation:
        def __init__(self, operation_id: int, _challenge_callback) -> None:
            self.operation_id = operation_id
            self.cancelled = False
            instances.append(self)

        def cancel(self) -> None:
            self.cancelled = True

        async def run(self):
            raise AssertionError("a superseded queued login must not run")

    async def refresh(operation_id: int) -> CodexAccountState:
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_CONNECTED),
            CodexModelCatalog(operation_id, (CodexModel("model"),), False),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.CodexLoginOperation", LoginOperation)
    monkeypatch.setattr("kimix_gui.qt.bridge.refresh_codex_account", refresh)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    submitted: list[object] = []
    monkeypatch.setattr(bridge, "submit", submitted.append)

    bridge.connect_chatgpt()
    bridge.refresh_codex_models()
    assert len(submitted) == 2
    await submitted[0]  # type: ignore[misc]
    await submitted[1]  # type: ignore[misc]

    assert len(instances) == 1
    assert instances[0].cancelled is True  # type: ignore[attr-defined]
    assert bridge._active_codex_login is None


@pytest.mark.asyncio
async def test_disconnect_cancels_active_login_before_waiting_for_account_lock(
    monkeypatch,
) -> None:
    login_started = asyncio.Event()
    login_cancelled = asyncio.Event()
    cancellation_calls = 0

    class LoginOperation:
        def __init__(self, operation_id: int, _challenge_callback) -> None:
            self.operation_id = operation_id

        def cancel(self) -> None:
            nonlocal cancellation_calls
            cancellation_calls += 1
            login_cancelled.set()

        async def run(self):
            login_started.set()
            await login_cancelled.wait()
            raise CodexAuthError(CodexProblem(PROBLEM_CANCELLED))

    async def disconnect(operation_id: int) -> CodexAccountState:
        return CodexAccountState(
            CodexAuthSnapshot(operation_id, AUTH_DISCONNECTED),
            CodexModelCatalog(operation_id, (), True),
        )

    monkeypatch.setattr("kimix_gui.qt.bridge.CodexLoginOperation", LoginOperation)
    monkeypatch.setattr("kimix_gui.qt.bridge.disconnect_codex_account", disconnect)

    async def unused_factory(_options: SessionOptions):
        raise AssertionError("session factory should not run")

    bridge = KimixBridge(session_factory=unused_factory)
    submitted: list[object] = []
    monkeypatch.setattr(bridge, "submit", submitted.append)

    bridge.connect_chatgpt()
    login_task = asyncio.create_task(submitted.pop(0))  # type: ignore[arg-type]
    await login_started.wait()
    bridge.disconnect_chatgpt()
    disconnect_task = asyncio.create_task(submitted.pop(0))  # type: ignore[arg-type]

    await asyncio.gather(login_task, disconnect_task)

    assert cancellation_calls == 1
    assert bridge._active_codex_login is None
