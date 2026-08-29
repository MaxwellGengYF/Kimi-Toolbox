"""Application-level routing between home and chat windows."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QWidget

from kimix_gui.backend import SessionOptions, create_sdk_session
from kimix_gui.history import HistoryLoader
from kimix_gui.llm import (
    PROBLEM_INVALID_SESSION_SELECTION,
    KimixGuiConfigStore,
    LLMSelection,
    ModelCatalogService,
    ProviderFileTarget,
    ResolvedLLMSelection,
    configured_selection,
    default_provider_file_path,
    default_provider_registry,
    unavailable_model,
)
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.bridge import KimixBridge, SessionFactory
from kimix_gui.qt.codex_controller import CodexController
from kimix_gui.qt.i18n import apply_language
from kimix_gui.qt.llm_text import problem_message
from kimix_gui.qt.main_window import MainWindow
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog, LLMSettingsResult
from kimix_gui.qt.theme import apply_interface_font, apply_theme_preference
from kimix_gui.session_index import SessionDeleter, SessionLoader


@dataclass(frozen=True, slots=True)
class AppNotification:
    message: str
    title: str = ""
    severity: str = "information"


class KimixGuiApp:
    """Route between home and chat, owning exact LLM selections and snapshots."""

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
    ) -> None:
        self._options = options
        self._session_loader = session_loader
        self._history_loader = history_loader
        self._session_deleter = session_deleter
        self._config_store = config_store or KimixGuiConfigStore()
        provider_registry = default_provider_registry()
        self._catalogs = ModelCatalogService(
            provider_registry,
            self._config_store,
            options.work_dir,
        )
        # Catalog resolution and runtime creation share one provider registry so a
        # provider plugin owns the target from discovery through session startup.
        resolved_session_factory = (
            partial(
                create_sdk_session,
                provider_registry=provider_registry,
            )
            if session_factory is create_sdk_session
            else session_factory
        )
        self._session_factory = resolved_session_factory
        self._preferences_dialog: PreferencesDialog | None = None
        self._llm_dialogs: dict[LLMSettingsDialog, str | None] = {}
        self._pending_catalog_startup = False

        saved_default = self._config_store.default_selection_for(options.work_dir)
        self._default_selection = (
            options.llm_selection
            or saved_default
            or configured_selection(ProviderFileTarget(default_provider_file_path()))
        )
        self._default_is_persisted = options.llm_selection is None and saved_default is not None
        self._default_config = self._catalogs.resolve(self._default_selection)
        self._active_config: ResolvedLLMSelection | None = None

        self.bridge = KimixBridge(
            session_factory=resolved_session_factory,
            history_loader=history_loader,
            session_loader=session_loader,
            session_deleter=session_deleter,
        )
        self.window: MainWindow | None = None
        self.codex_controller = CodexController(
            self.bridge,
            self._catalogs,
            default_parent=lambda: self.window,
            active_session_disconnected=self._on_active_codex_disconnected,
        )
        self._catalogs.subscribe(self._on_catalog_state_changed)
        self._notifications: list[AppNotification] = []
        self._qt_app: QApplication | None = None

    @property
    def options(self) -> SessionOptions:
        return self._options

    @property
    def default_config(self) -> ResolvedLLMSelection:
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

    def session_config(self, session_id: str) -> ResolvedLLMSelection | None:
        stored = self._config_store.session_selection_for(self._options.work_dir, session_id)
        if stored.problem is not None:
            model = replace(
                unavailable_model(
                    self._default_selection.target,
                    PROBLEM_INVALID_SESSION_SELECTION,
                    reason=stored.problem.reason,
                ),
                problem=stored.problem,
            )
            return ResolvedLLMSelection(
                selection=self._default_selection,
                model=model,
                materialized_selection=self._default_selection,
                problem=stored.problem,
            )
        selection = stored.selection
        if selection is None:
            return None
        selection = self._catalogs.materialize_session(session_id, selection)
        return self._catalogs.resolve(selection)

    def ensure_application(self) -> QApplication:
        existing = QApplication.instance()
        if existing is None:
            self._qt_app = QApplication(sys.argv)
            self._configure_application(self._qt_app)
            return self._qt_app
        qt_app = cast(QApplication, existing)
        self._configure_application(qt_app)
        return qt_app

    def _configure_application(self, qt_app: QApplication) -> None:
        apply_language(qt_app, self.interface_preferences)
        apply_theme_preference(qt_app, self.interface_preferences)
        apply_interface_font(qt_app, self.interface_preferences)

    def create_window(self) -> MainWindow:
        self.ensure_application()
        self.window = MainWindow(self)
        self.bridge.start()
        startup_config = (
            self.session_config(self._options.session_id) or self._default_config
            if self._options.session_id
            else self._default_config
        )
        self._pending_catalog_startup = bool(
            self._options.session_id
            and not self._catalogs.ready_for(startup_config.selection)
        )
        if self._pending_catalog_startup:
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
            config = self.session_config(self._options.session_id) or self._default_config
            if not config.available:
                self._show_home()
                self.open_llm_settings(self._options.session_id)
                return
            self._prepare_session_options(self._options.session_id, config)
            self._show_chat(config, record_session_config=False)
            return
        self._show_home()

    def _show_home(self, *, reload: bool = True) -> None:
        if self.window is not None:
            self.window.show_home(reload=reload)

    def start_new_session(self) -> None:
        if not self._default_config.available:
            if self.window is not None:
                self.window.notify_llm_required()
            self.open_llm_settings(None)
            return
        self._prepare_session_options(None, self._default_config)
        self._show_chat(self._default_config, record_session_config=True)

    def resume_session(self, session_id: str) -> None:
        config = self.session_config(session_id) or self._default_config
        self._prepare_session_options(session_id, config)
        self._show_chat(config, record_session_config=False)

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
            models=(self._active_config.model,),
            scope_label=QCoreApplication.translate("KimixGuiApp", "Current session · in use"),
            read_only=True,
            parent=self.window,
        )
        dialog.open()

    def _show_chat(
        self,
        config: ResolvedLLMSelection,
        *,
        record_session_config: bool,
    ) -> None:
        if self.window is None:
            return
        if not config.available:
            session_id = self._options.session_id
            self.window.show_notification(
                problem_message(config.problem or config.model.problem)
                or QCoreApplication.translate("KimixGuiApp", "LLM configuration is unavailable"),
                "error",
                "",
            )
            self._show_home()
            self.open_llm_settings(session_id)
            return
        chat = self.window.show_chat()
        # This immutable resolved value is the startup snapshot for the running session.
        self._active_config = config
        on_opened = (
            partial(self._record_session_config, selection=config.materialized_selection)
            if record_session_config
            else None
        )
        self.bridge.open_session(self._options, on_session_opened=on_opened)
        chat.reset_session_label()

    def _prepare_session_options(
        self,
        session_id: str | None,
        config: ResolvedLLMSelection,
    ) -> None:
        self._options = replace(
            self._options,
            session_id=session_id,
            llm_selection=config.materialized_selection,
            llm_runtime=config.runtime,
        )

    def _record_session_config(self, session_id: str, *, selection: LLMSelection) -> None:
        self._config_store.set_session(self._options.work_dir, session_id, selection)

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
        models = self._catalogs.models_for(current.selection, self._default_selection)
        scope_label = (
            QCoreApplication.translate("KimixGuiApp", "Session {id}").format(id=session_id)
            if session_id is not None
            else QCoreApplication.translate("KimixGuiApp", "New sessions")
        )
        dialog = LLMSettingsDialog(
            current=current,
            models=models,
            scope_label=scope_label,
            project_default=self._default_config if session_id is not None else None,
            inherits_project_default=session_id is not None and session_config is None,
            manage_library=session_id is None,
            registrar=self._config_store.add_provider_file,
            remover=self._config_store.remove_provider_file,
            parent=parent,
        )
        dialog.connect_chatgpt.connect(
            lambda: self.codex_controller.connect(parent=dialog)
        )
        self._llm_dialogs[dialog] = session_id

        def _forget_dialog() -> None:
            self._llm_dialogs.pop(dialog, None)

        dialog.finished.connect(_forget_dialog)

        def _applied(result: object) -> None:
            if isinstance(result, LLMSettingsResult):
                self._on_llm_settings(result, session_id)

        dialog.applied.connect(_applied)
        if preserve_parent_dialog:

            def restore_parent() -> None:
                parent.raise_()
                parent.activateWindow()

            dialog.finished.connect(restore_parent)
        dialog.open()

    def open_preferences(self) -> None:
        if self.window is None:
            return
        dialog = PreferencesDialog(
            self.interface_preferences,
            codex_snapshot=self._catalogs.context.codex_snapshot,
            codex_catalog=self._catalogs.context.codex_catalog,
            parent=self.window,
        )
        self._preferences_dialog = dialog
        dialog.applied.connect(self._on_preferences_applied)
        dialog.manage_llm.connect(lambda: self._open_llm_from_preferences(dialog))
        dialog.connect_chatgpt.connect(
            lambda: self.codex_controller.connect(parent=dialog)
        )
        dialog.refresh_codex_models.connect(self.codex_controller.refresh_models)
        dialog.disconnect_chatgpt.connect(
            lambda: self.codex_controller.disconnect(parent=self.window)
        )
        dialog.finished.connect(lambda: self._clear_preferences_dialog(dialog))
        dialog.open()

    def _clear_preferences_dialog(self, dialog: PreferencesDialog) -> None:
        if self._preferences_dialog is dialog:
            self._preferences_dialog = None

    def _on_catalog_state_changed(self) -> None:
        self._default_selection, self._default_config = self._catalogs.refresh_default(
            self._default_selection,
            persisted=self._default_is_persisted,
        )
        context = self._catalogs.context
        preferences = self._preferences_dialog
        if preferences is not None:
            preferences.set_codex_snapshot(context.codex_snapshot)
            preferences.set_codex_catalog(context.codex_catalog)
        for dialog, session_id in tuple(self._llm_dialogs.items()):
            session_config = self.session_config(session_id) if session_id is not None else None
            current = session_config or self._default_config
            dialog.set_models(
                self._catalogs.models_for(current.selection, self._default_selection),
                current=current,
                project_default=self._default_config if session_id is not None else None,
                inherits_project_default=session_id is not None and session_config is None,
            )
        # Deliberately do not replace ``_active_config``: it is the running snapshot.
        if self.window is not None and self.window.home is not None:
            self.window.home.refresh_configuration(self._default_config)
        if self._pending_catalog_startup:
            startup = (
                self.session_config(self._options.session_id) or self._default_config
                if self._options.session_id
                else self._default_config
            )
            if self._catalogs.ready_for(startup.selection):
                self._pending_catalog_startup = False
                self._startup()

    def _on_active_codex_disconnected(self) -> None:
        self._active_config = None
        self._options = replace(self._options, session_id=None)
        self._show_home(reload=False)
        if self.window is not None:
            self.window.remove_chat()

    def _open_llm_from_preferences(self, dialog: PreferencesDialog) -> None:
        self.open_llm_settings(None, parent=dialog, preserve_parent_dialog=True)

    def _on_preferences_applied(self, result: object) -> None:
        if not isinstance(result, InterfacePreferences):
            return
        try:
            self._config_store.set_interface(result)
        except (OSError, TypeError, ValueError) as exc:
            if self.window is not None:
                self.window.show_notification(
                    QCoreApplication.translate(
                        "KimixGuiApp", "Failed to save preferences: {reason}"
                    ).format(reason=exc),
                    "error",
                    "",
                )
            return
        instance = QApplication.instance()
        if instance is not None:
            qt_app = cast(QApplication, instance)
            apply_theme_preference(qt_app, result)
            apply_interface_font(qt_app, result)
            apply_language(qt_app, result)

    def _on_llm_settings(self, result: LLMSettingsResult, session_id: str | None) -> None:
        selection = result.selection
        try:
            if session_id is None:
                self._config_store.set_default(self._options.work_dir, selection)
                self._default_selection = selection
                self._default_is_persisted = True
                self._default_config = self._catalogs.resolve(selection)
            elif result.use_project_default:
                self._config_store.clear_session(self._options.work_dir, session_id)
            else:
                self._config_store.set_session(self._options.work_dir, session_id, selection)
        except (OSError, TypeError, ValueError) as exc:
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
            pending = (
                self._default_config
                if result.use_project_default
                else self._catalogs.resolve(selection)
            )
            label = (
                QCoreApplication.translate("KimixGuiApp", "Project default · {config}").format(
                    config=pending.label
                )
                if result.use_project_default
                else pending.label
            )
            self.window.chat.set_pending_config(label)
