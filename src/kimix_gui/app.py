"""Application-level routing between home and chat windows."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import partial

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QDialog, QWidget

from kimix_gui.backend import SessionOptions, create_sdk_session
from kimix_gui.codex_auth import (
    AUTH_CONNECTED,
    AUTH_DISCONNECTED,
    CodexAuthService,
    CodexAuthSnapshot,
    CodexBrowserChallenge,
    CodexModelCatalog,
    fallback_catalog,
)
from kimix_gui.history import HistoryLoader
from kimix_gui.llm_config import (
    ChatGPTModelReference,
    ChatGPTSource,
    ConfigFileSource,
    KimixGuiConfigStore,
    LLMConfigError,
    LLMReference,
    default_config_path,
    inspect_llm_config,
    llm_reference_available,
    reference_from_source,
    unavailable_config_reference,
)
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.bridge import KimixBridge, SessionFactory
from kimix_gui.qt.codex_dialog import CodexLoginDialog, DisconnectChatGPTDialog
from kimix_gui.qt.i18n import apply_language
from kimix_gui.qt.main_window import MainWindow
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.settings_dialog import (
    LLMSettingsDialog,
    LLMSettingsResult,
    config_problem_message,
)
from kimix_gui.qt.theme import apply_interface_font, apply_theme_preference
from kimix_gui.session_index import SessionDeleter, SessionLoader

# ``KimixGuiApp`` is a plain class, not a ``QObject``, so it has no ``tr()``. The copy
# it hands to widgets goes through ``QCoreApplication.translate`` with an explicit
# context instead, spelled out at every call site because lupdate only reads literal
# arguments -- a helper or a constant would extract nothing.
# (Plain ``#``, never ``#:`` -- lupdate reads ``#:`` as an extracomment and would
# staple this paragraph onto the next translatable string in the file.)


@dataclass(frozen=True, slots=True)
class AppNotification:
    message: str
    title: str = ""
    severity: str = "information"


class KimixGuiApp:
    """Route between home and chat, owning GUI config and the Kimix bridge."""

    TITLE = "Kimix"

    def __init__(
        self,
        options: SessionOptions,
        *,
        session_factory: SessionFactory = create_sdk_session,
        session_loader: SessionLoader | None = None,
        history_loader: HistoryLoader | None = None,
        config_store: KimixGuiConfigStore | None = None,
        session_deleter: SessionDeleter | None = None,
        codex_service: CodexAuthService | None = None,
    ) -> None:
        self._options = options
        self._codex_service = codex_service or CodexAuthService()
        resolved_session_factory = (
            partial(create_sdk_session, codex_service=self._codex_service)
            if session_factory is create_sdk_session
            else session_factory
        )
        self._session_factory = resolved_session_factory
        self._session_loader = session_loader
        self._history_loader = history_loader
        self._session_deleter = session_deleter
        self._config_store = config_store or KimixGuiConfigStore()
        self._codex_snapshot = CodexAuthSnapshot(operation_id=0, state=AUTH_DISCONNECTED)
        self._codex_catalog = fallback_catalog()
        self._codex_operation = 0
        self._codex_initialized = False
        self._codex_login_dialog: CodexLoginDialog | None = None
        self._preferences_dialog: PreferencesDialog | None = None
        self._llm_dialogs: list[LLMSettingsDialog] = []
        self._pending_codex_startup = False
        self._disconnecting_active_codex = False
        startup_source = options.llm_source
        if isinstance(startup_source, ChatGPTSource):
            startup_config: LLMReference = reference_from_source(startup_source)
        else:
            config_source = startup_source or ConfigFileSource(default_config_path())
            try:
                startup_config = inspect_llm_config(
                    config_source.path,
                    model_override=config_source.model_override,
                )
            except LLMConfigError:
                startup_config = unavailable_config_reference(
                    config_source.path,
                    model_override=config_source.model_override,
                )
        saved_default = self._config_store.default_for(options.work_dir)
        if saved_default is not None:
            saved_default = self._refresh_reference(saved_default)
        has_startup_override = options.llm_source is not None
        self._default_config = (
            startup_config if has_startup_override else saved_default or startup_config
        )
        self._active_config: LLMReference | None = None
        self.bridge = KimixBridge(
            session_factory=resolved_session_factory,
            history_loader=history_loader,
            session_loader=session_loader,
            session_deleter=session_deleter,
            codex_service=self._codex_service,
        )
        self.window: MainWindow | None = None
        self._notifications: list[AppNotification] = []
        self._qt_app: QApplication | None = None

    @property
    def options(self) -> SessionOptions:
        return self._options

    @property
    def default_config(self) -> LLMReference:
        return self._default_config

    @property
    def interface_preferences(self) -> InterfacePreferences:
        return self._config_store.interface

    @property
    def screen(self) -> object:
        if self.window is None:
            return None
        return self.window.current_view

    def note(self, message: str, severity: str = "information", title: str = "") -> None:
        self._notifications.append(AppNotification(message, title, severity))

    def session_config(self, session_id: str) -> LLMReference | None:
        reference = self._config_store.session_for(self._options.work_dir, session_id)
        return self._refresh_reference(reference) if reference is not None else None

    def _refresh_reference(self, reference: LLMReference) -> LLMReference:
        if isinstance(reference, ChatGPTModelReference):
            return self._chatgpt_reference(reference.model_name)
        try:
            refreshed = inspect_llm_config(
                reference.path,
                model_override=reference.model_override,
            )
        except LLMConfigError as exc:
            return replace(reference, error=exc.problem)
        return refreshed

    def ensure_application(self) -> QApplication:
        existing = QApplication.instance()
        if existing is None:
            self._qt_app = QApplication(sys.argv)
            self._configure_application(self._qt_app)
            return self._qt_app
        self._configure_application(existing)
        return existing

    def _configure_application(self, qt_app: QApplication) -> None:
        """Install language, theme and font before any widget is constructed.

        Language goes first: ``tr()`` is evaluated while widgets are built, so a
        catalog installed afterwards would leave the already-created labels English.
        """

        apply_language(qt_app, self.interface_preferences)
        apply_theme_preference(qt_app, self.interface_preferences)
        apply_interface_font(qt_app, self.interface_preferences)

    def create_window(self) -> MainWindow:
        self.ensure_application()
        self.window = MainWindow(self)
        self.bridge.start()
        startup_reference = (
            self.session_config(self._options.session_id) or self._default_config
            if self._options.session_id
            else self._default_config
        )
        self._pending_codex_startup = bool(
            self._options.session_id and isinstance(startup_reference, ChatGPTModelReference)
        )
        if self._pending_codex_startup:
            self._show_home()
        else:
            self._startup()
        return self.window

    def run(self) -> None:
        qt_app = self.ensure_application()
        window = self.create_window()
        window.show()
        raise SystemExit(qt_app.exec())

    def shutdown(self) -> None:
        self.bridge.stop()

    def _startup(self) -> None:
        if self._options.session_id:
            reference = self.session_config(self._options.session_id) or self._default_config
            if not llm_reference_available(reference):
                self._show_home()
                self.open_llm_settings(self._options.session_id)
                return
            self._prepare_session_options(self._options.session_id, reference)
            self._show_chat(reference, record_session_config=False)
            return
        self._show_home()

    def _show_home(self, *, reload: bool = True) -> None:
        if self.window is None:
            return
        self.window.show_home(reload=reload)

    def start_new_session(self) -> None:
        if not llm_reference_available(self._default_config):
            if self.window is not None:
                # Copy lives on MainWindow: this used to be a byte-identical
                # duplicate of ``MainWindow._home_llm_required``, and only a
                # ``QObject`` can call ``tr()``.
                self.window.notify_llm_required()
            self.open_llm_settings(None)
            return
        self._prepare_session_options(None, self._default_config)
        self._show_chat(self._default_config, record_session_config=True)

    def resume_session(self, session_id: str) -> None:
        reference = self.session_config(session_id) or self._default_config
        self._prepare_session_options(session_id, reference)
        self._show_chat(reference, record_session_config=False)

    def leave_chat(self) -> None:
        self.bridge.close_session()
        self._active_config = None
        self._options = replace(self._options, session_id=None)
        self._show_home(reload=False)
        if self.window is not None:
            self.window.remove_chat()

    def open_chat_settings(self) -> None:
        if self._active_config is None or self.window is None:
            return
        dialog = LLMSettingsDialog(
            current=self._active_config,
            references=(),
            scope_label=QCoreApplication.translate("KimixGuiApp", "Current session · in use"),
            read_only=True,
            parent=self.window,
        )
        dialog.open()

    def _show_chat(self, reference: LLMReference, *, record_session_config: bool) -> None:
        if self.window is None:
            return
        if not llm_reference_available(reference):
            session_id = self._options.session_id
            if isinstance(reference, ChatGPTModelReference):
                problem_message = (
                    QCoreApplication.translate(
                        "KimixGuiApp",
                        "This ChatGPT model is not available for the connected account.",
                    )
                    if reference.problem_code == "model_unavailable"
                    else QCoreApplication.translate(
                        "KimixGuiApp",
                        "Connect ChatGPT to use this subscription model.",
                    )
                )
            else:
                problem_message = config_problem_message(reference.error)
            self.window.show_notification(
                problem_message
                or QCoreApplication.translate("KimixGuiApp", "LLM configuration is unavailable"),
                "error",
                "",
            )
            self._show_home()
            self.open_llm_settings(session_id)
            return
        chat = self.window.show_chat()
        self._active_config = reference
        on_opened = (
            partial(self._record_session_config, reference=reference)
            if record_session_config
            else None
        )
        self.bridge.open_session(self._options, on_session_opened=on_opened)
        chat.reset_session_label()

    def _prepare_session_options(
        self,
        session_id: str | None,
        reference: LLMReference,
    ) -> None:
        self._options = replace(
            self._options,
            session_id=session_id,
            llm_source=reference.source,
        )

    def _record_session_config(self, session_id: str, *, reference: LLMReference) -> None:
        self._config_store.set_session(self._options.work_dir, session_id, reference)

    def open_llm_settings(
        self,
        session_id: str | None,
        *,
        parent: QWidget | None = None,
        preserve_parent_dialog: bool = False,
    ) -> None:
        if self.window is None:
            return
        parent = parent if parent is not None else self.window
        session_config = self.session_config(session_id) if session_id is not None else None
        current = session_config or self._default_config
        references = [
            self._refresh_reference(reference)
            for reference in self._config_store.references_for(self._options.work_dir)
        ]
        scope_label = (
            QCoreApplication.translate("KimixGuiApp", "Session {id}").format(id=session_id)
            if session_id is not None
            else QCoreApplication.translate("KimixGuiApp", "New sessions")
        )
        dialog = LLMSettingsDialog(
            current=current,
            references=references,
            scope_label=scope_label,
            project_default=self._default_config if session_id is not None else None,
            inherits_project_default=session_id is not None and session_config is None,
            manage_library=session_id is None,
            chatgpt_connected=self._codex_snapshot.state == AUTH_CONNECTED,
            registrar=self._config_store.add_config,
            remover=self._config_store.remove_config,
            parent=parent,
        )
        dialog.set_chatgpt_references(
            self._chatgpt_references(),
            connected=self._codex_snapshot.state == AUTH_CONNECTED,
        )
        dialog.connect_chatgpt.connect(lambda: self.connect_chatgpt(parent=dialog))
        self._llm_dialogs.append(dialog)

        def _forget_dialog() -> None:
            if dialog in self._llm_dialogs:
                self._llm_dialogs.remove(dialog)

        dialog.finished.connect(_forget_dialog)

        def _applied(result: object) -> None:
            if isinstance(result, LLMSettingsResult):
                self._on_llm_settings(result, session_id)

        dialog.applied.connect(_applied)
        if preserve_parent_dialog:
            # Opened from another dialog, which stays up underneath: hand it back the
            # top once this one closes.
            dialog.finished.connect(lambda: (parent.raise_(), parent.activateWindow()))
        dialog.open()

    def open_preferences(self) -> None:
        if self.window is None:
            return
        dialog = PreferencesDialog(
            self.interface_preferences,
            codex_snapshot=self._codex_snapshot,
            codex_catalog=self._codex_catalog,
            parent=self.window,
        )
        self._preferences_dialog = dialog
        dialog.applied.connect(self._on_preferences_applied)
        dialog.manage_llm.connect(lambda: self._open_llm_from_preferences(dialog))
        dialog.connect_chatgpt.connect(lambda: self.connect_chatgpt(parent=dialog))
        dialog.refresh_codex_models.connect(self.bridge.refresh_codex_models)
        dialog.disconnect_chatgpt.connect(self.disconnect_chatgpt)
        dialog.finished.connect(lambda: self._clear_preferences_dialog(dialog))
        dialog.open()

    def _clear_preferences_dialog(self, dialog: PreferencesDialog) -> None:
        if self._preferences_dialog is dialog:
            self._preferences_dialog = None

    def connect_chatgpt(self, *, parent: QWidget | None = None) -> None:
        dialog = self._codex_login_dialog
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        owner = parent or self.window
        dialog = CodexLoginDialog(parent=owner)
        self._codex_login_dialog = dialog
        dialog.retry_requested.connect(lambda: self._retry_chatgpt_login(dialog))
        dialog.cancel_requested.connect(self.bridge.cancel_chatgpt_login)
        dialog.finished.connect(lambda: self._clear_codex_login_dialog(dialog))
        dialog.open()
        dialog.begin(self.bridge.connect_chatgpt())

    def _retry_chatgpt_login(self, dialog: CodexLoginDialog) -> None:
        if self._codex_login_dialog is dialog:
            dialog.begin(self.bridge.connect_chatgpt())

    def _clear_codex_login_dialog(self, dialog: CodexLoginDialog) -> None:
        if self._codex_login_dialog is dialog:
            self._codex_login_dialog = None

    def disconnect_chatgpt(self) -> None:
        if self.window is None:
            return
        if not self.bridge.uses_chatgpt:
            self.bridge.disconnect_chatgpt()
            return
        dialog = DisconnectChatGPTDialog(parent=self.window)

        def _done(result: int) -> None:
            if result != int(QDialog.DialogCode.Accepted):
                return
            self._disconnecting_active_codex = True
            self.bridge.disconnect_chatgpt(close_active_session=True)

        dialog.finished.connect(_done)
        dialog.open()

    def on_codex_browser_challenge(self, challenge: CodexBrowserChallenge) -> None:
        dialog = self._codex_login_dialog
        if dialog is not None:
            dialog.show_challenge(challenge)

    def on_codex_auth_changed(self, snapshot: CodexAuthSnapshot) -> None:
        if snapshot.operation_id < self._codex_operation:
            return
        self._codex_operation = snapshot.operation_id
        self._codex_initialized = True
        self._codex_snapshot = snapshot
        dialog = self._codex_login_dialog
        if dialog is not None:
            dialog.set_snapshot(snapshot)
        preferences = self._preferences_dialog
        if preferences is not None:
            preferences.set_codex_snapshot(snapshot)
        self._refresh_chatgpt_references()
        if self._pending_codex_startup and snapshot.state != AUTH_CONNECTED:
            self._pending_codex_startup = False
            self._startup()
        if snapshot.state == AUTH_DISCONNECTED and self._disconnecting_active_codex:
            self._disconnecting_active_codex = False
            self._active_config = None
            self._options = replace(self._options, session_id=None)
            self._show_home(reload=False)
            if self.window is not None:
                self.window.remove_chat()

    def on_codex_catalog_changed(self, catalog: CodexModelCatalog) -> None:
        if catalog.operation_id < self._codex_operation:
            return
        self._codex_operation = catalog.operation_id
        self._codex_catalog = catalog
        preferences = self._preferences_dialog
        if preferences is not None:
            preferences.set_codex_catalog(catalog)
        self._refresh_chatgpt_references()
        if self._pending_codex_startup and self._codex_snapshot.state == AUTH_CONNECTED:
            self._pending_codex_startup = False
            self._startup()

    def _chatgpt_references(self) -> tuple[ChatGPTModelReference, ...]:
        if self._codex_snapshot.state != AUTH_CONNECTED:
            return ()
        return tuple(self._chatgpt_reference(model.slug) for model in self._codex_catalog.models)

    def _chatgpt_reference(self, model_name: str) -> ChatGPTModelReference:
        model = next(
            (entry for entry in self._codex_catalog.models if entry.slug == model_name),
            None,
        )
        connected = self._codex_snapshot.state == AUTH_CONNECTED
        if model is None:
            return ChatGPTModelReference(
                model_name=model_name,
                available=False,
                stale=True,
                problem_code="model_unavailable" if connected else self._codex_snapshot.state,
            )
        capabilities = ["thinking"]
        normalized_modalities = {item.lower() for item in model.input_modalities}
        if normalized_modalities & {"image", "images", "vision"}:
            capabilities.append("image_in")
        if normalized_modalities & {"video", "videos"}:
            capabilities.append("video_in")
        return ChatGPTModelReference(
            model_name=model.slug,
            display_name=model.display_name,
            max_context_size=model.max_context_size,
            max_tokens=model.max_tokens,
            capabilities=tuple(capabilities),
            supported_efforts=model.reasoning_efforts,
            default_reasoning_effort=model.default_reasoning_effort,
            input_modalities=model.input_modalities,
            priority=model.priority,
            available=connected,
            stale=self._codex_catalog.stale,
            problem_code=None if connected else self._codex_snapshot.state,
        )

    def _refresh_chatgpt_references(self) -> None:
        if isinstance(self._default_config, ChatGPTModelReference):
            self._default_config = self._chatgpt_reference(self._default_config.model_name)
        if isinstance(self._active_config, ChatGPTModelReference):
            self._active_config = self._chatgpt_reference(self._active_config.model_name)
        references = self._chatgpt_references()
        connected = self._codex_snapshot.state == AUTH_CONNECTED
        for dialog in tuple(self._llm_dialogs):
            dialog.set_chatgpt_references(references, connected=connected)
        if self.window is not None and self.window.home is not None:
            self.window.home.refresh_configuration(self._default_config)

    def _open_llm_from_preferences(self, dialog: PreferencesDialog) -> None:
        self.open_llm_settings(None, parent=dialog, preserve_parent_dialog=True)

    def _on_preferences_applied(self, result: object) -> None:
        if not isinstance(result, InterfacePreferences):
            return
        try:
            self._config_store.set_interface(result)
        except OSError as exc:
            if self.window is not None:
                self.window.show_notification(
                    QCoreApplication.translate(
                        "KimixGuiApp", "Failed to save preferences: {reason}"
                    ).format(reason=exc),
                    "error",
                    "",
                )
            return
        qt_app = QApplication.instance()
        if qt_app is not None:
            # Theme before font, and for the same reason as at startup: ``apply_theme``
            # resets the application font to the theme's own base, so the interface
            # font has to be re-stated after it. No event is dispatched between the
            # two calls, so nothing ever paints with the intermediate font.
            apply_theme_preference(qt_app, result)
            apply_interface_font(qt_app, result)
            # Installing a catalog makes Qt post ``LanguageChange`` to every QObject,
            # which is what the live widgets' ``Retranslator`` children react to. It is
            # posted, not sent, so the new copy appears on the next trip through the
            # event loop rather than inside this call.
            apply_language(qt_app, result)

    def _on_llm_settings(self, result: LLMSettingsResult, session_id: str | None) -> None:
        reference = result.reference
        try:
            if session_id is None:
                self._config_store.set_default(self._options.work_dir, reference)
                self._default_config = reference
            elif result.use_project_default:
                self._config_store.clear_session(self._options.work_dir, session_id)
            else:
                self._config_store.set_session(self._options.work_dir, session_id, reference)
        except OSError as exc:
            if self.window is not None:
                self.window.show_notification(
                    QCoreApplication.translate(
                        "KimixGuiApp", "Failed to save LLM configuration metadata: {reason}"
                    ).format(reason=exc),
                    "error",
                    "",
                )
            return
        if self.window is None:
            return
        if self.window.home is not None:
            self.window.home.refresh_configuration(self._default_config)
        if self.window.chat is not None:
            label = (
                QCoreApplication.translate("KimixGuiApp", "Project default · {config}").format(
                    config=reference.label
                )
                if result.use_project_default
                else reference.label
            )
            self.window.chat.set_pending_config(label)
