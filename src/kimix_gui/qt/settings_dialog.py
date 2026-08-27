"""LLM Provider, Model and Variant selection dialog."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.llm import (
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_FILE_MISSING,
    PROBLEM_INVALID_JSON,
    PROBLEM_INVALID_PROVIDER_FILE,
    PROBLEM_INVALID_SESSION_SELECTION,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_MODEL_UNAVAILABLE,
    PROBLEM_NOT_AN_OBJECT,
    PROBLEM_NOT_JSON,
    PROBLEM_PROVIDER_FILE_UNAVAILABLE,
    PROBLEM_VARIANT_UNAVAILABLE,
    PROBLEM_VARIANT_UNRESOLVED,
    ChatGPTTarget,
    LLMInspectionError,
    LLMModelDescriptor,
    LLMProblem,
    LLMSelection,
    LLMVariantDescriptor,
    LLMVariantKey,
    ProviderFileTarget,
    ResolvedLLMSelection,
    inspect_provider_file,
    resolve_selection,
    target_key,
    unavailable_model,
)
from kimix_gui.qt.components import (
    DialogFooter,
    DisclosureHeader,
    KeyValueList,
    SettingsList,
    VariantOption,
    VariantPicker,
)
from kimix_gui.qt.styling import Level, Role, Tone, Variant, style

ModelInspector = Callable[[Path], LLMModelDescriptor]
ProviderFileRegistrar = Callable[[ProviderFileTarget], None]
ProviderFileRemover = Callable[[Path], None]

CHATGPT_GROUP = "chatgpt"
PROVIDER_FILE_GROUP = "provider_file"


def _model_group(model: LLMModelDescriptor) -> str:
    return CHATGPT_GROUP if isinstance(model.target, ChatGPTTarget) else PROVIDER_FILE_GROUP


@dataclass(frozen=True, slots=True)
class LLMSettingsResult:
    selection: LLMSelection
    use_project_default: bool = False


class ModelListItem(QListWidgetItem):
    """One left-pane row per Model, independent of its Variant count."""

    def __init__(self, model: LLMModelDescriptor) -> None:
        super().__init__()
        self.model_descriptor = model
        self.refresh()

    def refresh(self) -> None:
        label = self.model_descriptor.label
        self.setText(label)
        self.setToolTip(label)


class LLMSettingsDialog(QDialog):
    """Select a Model row and an exact Variant for one persistence scope."""

    applied = Signal(object)
    connect_chatgpt = Signal()

    def __init__(
        self,
        *,
        current: ResolvedLLMSelection,
        models: Iterable[LLMModelDescriptor],
        scope_label: str,
        project_default: ResolvedLLMSelection | None = None,
        inherits_project_default: bool = False,
        manage_library: bool = False,
        read_only: bool = False,
        chatgpt_connected: bool = False,
        inspector: ModelInspector = inspect_provider_file,
        registrar: ProviderFileRegistrar | None = None,
        remover: ProviderFileRemover | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings-dialog")
        self.setWindowTitle(self.tr("LLM configuration"))
        self.setModal(True)
        self.resize(920, 580)
        self._current = current
        self._project_default = project_default
        self._inherits_project_default = inherits_project_default
        self._scope_label = scope_label
        self._manage_library = manage_library
        self._read_only = read_only
        self._chatgpt_connected = chatgpt_connected
        self._inspector = inspector
        self._registrar = registrar
        self._remover = remover
        self._models: dict[str, LLMModelDescriptor] = {}
        for model in [*models, current.model]:
            self._models[target_key(model.target)] = model
        if project_default is not None:
            self._models[target_key(project_default.model.target)] = project_default.model
        self._selected_model: LLMModelDescriptor | None = None
        self._selected_selection: LLMSelection | None = None
        self._resolved_preview: ResolvedLLMSelection | None = None
        self._use_project_default = False
        self._variant_keys: dict[str, LLMVariantKey] = {}
        current_group = _model_group(current.model)
        self._group_expanded = {
            CHATGPT_GROUP: current_group == CHATGPT_GROUP,
            PROVIDER_FILE_GROUP: current_group == PROVIDER_FILE_GROUP,
        }
        self._group_rows: dict[str, list[QListWidgetItem]] = {}
        self._group_headers: dict[str, DisclosureHeader] = {}
        self._narrow = False
        self._build()
        self._select_initial()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.addLayout(self._header())

        self._error = QLabel("")
        self._error.setObjectName("settings-error")
        self._error.setWordWrap(True)
        style(self._error, tone=Tone.DANGER)
        root.addWidget(self._error)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("settings-body")
        self._splitter.addWidget(self._sources_pane())
        self._splitter.addWidget(self._details_pane())
        self._splitter.setSizes([330, 550])
        root.addWidget(self._splitter, 1)
        root.addWidget(self._footer())

        self._populate_list()
        self._list.currentItemChanged.connect(self._on_current_changed)

    def _header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        title = QLabel(self.tr("LLM configuration"))
        title.setObjectName("settings-title")
        style(title, role=Role.DISPLAY, level=Level.TWO)
        scope = QLabel(self._scope_label)
        scope.setObjectName("settings-scope")
        style(scope, tone=Tone.MUTED)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(scope)
        return header

    def _sources_pane(self) -> QWidget:
        sources = QWidget()
        sources.setObjectName("config-sources")
        layout = QVBoxLayout(sources)
        title = QLabel(self.tr("ACTIVE MODEL") if self._read_only else self.tr("AVAILABLE MODELS"))
        title.setObjectName("config-sources-title")
        style(title, role=Role.TITLE)
        layout.addWidget(title)

        self._inherit = None
        if self._project_default is not None and not self._read_only:
            inherit = QCheckBox(self.tr("Follow project default"))
            inherit.setObjectName("inherit-project-default")
            inherit.setAccessibleDescription(
                self.tr("Use the project's saved model and variant for this session")
            )
            inherit.setChecked(self._inherits_project_default)
            inherit.toggled.connect(self._inherit_toggled)
            self._inherit = inherit
            layout.addWidget(inherit)

        self._list = SettingsList()
        self._list.setObjectName("config-list")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        layout.addWidget(self._list)
        return sources

    def _details_pane(self) -> QWidget:
        details = QWidget()
        details.setObjectName("config-details")
        layout = QVBoxLayout(details)

        title = QLabel(self.tr("MODEL SELECTION"))
        title.setObjectName("config-details-title")
        style(title, role=Role.TITLE)
        layout.addWidget(title)

        self._summary_fields = KeyValueList(
            (
                ("selection-model", self.tr("Current model")),
                ("selection-source", self.tr("Source")),
                ("selection-status", self.tr("Status")),
            )
        )
        layout.addWidget(self._summary_fields)

        variant_row = QHBoxLayout()
        self._variant_label = QLabel(self.tr("Variant"))
        self._variant_label.setObjectName("variant-picker-label")
        self._variant_picker = VariantPicker()
        self._variant_picker.setAccessibleName(self.tr("Model variant"))
        self._variant_picker.setAccessibleDescription(
            self.tr("Choose the exact runtime variant saved for this model")
        )
        self._variant_picker.key_activated.connect(self._variant_activated)
        variant_row.addWidget(self._variant_label)
        variant_row.addWidget(self._variant_picker, 1)
        layout.addLayout(variant_row)

        model_title = QLabel(self.tr("MODEL DETAILS"))
        model_title.setObjectName("model-details-title")
        style(model_title, role=Role.OVERLINE)
        layout.addWidget(model_title)
        self._model_fields = KeyValueList(
            (
                ("model-id", self.tr("Model ID")),
                ("model-context", self.tr("Context")),
                ("model-output", self.tr("Max output")),
                ("model-capabilities", self.tr("Capabilities")),
                ("model-modalities", self.tr("Input")),
            )
        )
        layout.addWidget(self._model_fields)

        provider_title = QLabel(self.tr("PROVIDER DETAILS"))
        provider_title.setObjectName("provider-details-title")
        style(provider_title, role=Role.OVERLINE)
        layout.addWidget(provider_title)
        self._provider_fields = KeyValueList(
            (
                ("provider-type", self.tr("Provider")),
                ("provider-endpoint", self.tr("Endpoint")),
                ("provider-credential", self.tr("Credential")),
                ("provider-format", self.tr("Format")),
                ("provider-thinking", self.tr("Provider thinking")),
            )
        )
        layout.addWidget(self._provider_fields)
        layout.addStretch()
        return details

    def _footer(self) -> DialogFooter:
        self._delete = None
        extra: list[QPushButton] = []
        if self._manage_library and not self._read_only:
            delete = QPushButton(self.tr("Remove"))
            delete.setObjectName("delete-config")
            delete.setEnabled(False)
            delete.clicked.connect(self._remove_selected)
            self._delete = delete
            extra.append(delete)

        close = QPushButton(self.tr("Close") if self._read_only else self.tr("Cancel"))
        close.setObjectName("close-settings" if self._read_only else "cancel-settings")
        self._apply = None
        if not self._read_only:
            apply_button = QPushButton(self.tr("Use model"))
            apply_button.setObjectName("apply-settings")
            style(apply_button, variant=Variant.PRIMARY)
            apply_button.setEnabled(False)
            apply_button.clicked.connect(self._apply_clicked)
            self._apply = apply_button
        close.clicked.connect(self.reject)
        return DialogFooter(dismiss=close, extra=extra, confirm=self._apply, parent=self)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        narrow = self.width() < DARK.breakpoints.settings_narrow
        if narrow == self._narrow:
            return
        self._narrow = narrow
        self._splitter.setOrientation(
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        )

    def _populate_list(self) -> None:
        selected_target = (
            target_key(self._selected_model.target) if self._selected_model is not None else None
        )
        self._list.clear()
        self._group_rows = {}
        self._group_headers = {}
        grouped: dict[str, list[ModelListItem]] = {
            CHATGPT_GROUP: [],
            PROVIDER_FILE_GROUP: [],
        }
        for model in sorted(self._models.values(), key=lambda item: item.priority):
            grouped[_model_group(model)].append(ModelListItem(model))
        self._add_group(CHATGPT_GROUP, grouped[CHATGPT_GROUP])
        if grouped[PROVIDER_FILE_GROUP] or self._manage_library or self._read_only:
            self._add_group(PROVIDER_FILE_GROUP, grouped[PROVIDER_FILE_GROUP])
        if selected_target is not None:
            self._select_target(selected_target)

    def _add_group(self, group: str, items: list[ModelListItem]) -> None:
        title = (
            QCoreApplication.translate("LLMSettingsDialog", "CHATGPT SUBSCRIPTION · {count}")
            if group == CHATGPT_GROUP
            else QCoreApplication.translate("LLMSettingsDialog", "PROVIDER FILES · {count}")
        ).format(count=len(items))
        header_item = QListWidgetItem()
        header_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(header_item)
        header = DisclosureHeader(
            title,
            expanded=self._group_expanded[group],
            expanded_tooltip=self.tr("Collapse section"),
            collapsed_tooltip=self.tr("Expand section"),
            parent=self._list,
        )
        header.setObjectName(
            "chatgpt-config-group" if group == CHATGPT_GROUP else "provider-config-group"
        )
        header.toggled.connect(
            lambda expanded, source_group=group: self._set_group_expanded(
                source_group,
                expanded,
            )
        )
        self._list.set_sized_item_widget(header_item, header)
        self._group_headers[group] = header

        rows: list[QListWidgetItem] = []
        if group == CHATGPT_GROUP and not self._chatgpt_connected and not self._read_only:
            action_item = QListWidgetItem()
            action_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(action_item)
            login = QPushButton(self.tr("Connect ChatGPT"), self._list)
            login.setObjectName("connect-chatgpt-models")
            login.setToolTip(self.tr("Use ChatGPT subscription models"))
            login.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            style(login, variant=Variant.PRIMARY)
            login.clicked.connect(self.connect_chatgpt.emit)
            self._list.set_sized_item_widget(action_item, login)
            rows.append(action_item)
        if group == PROVIDER_FILE_GROUP and self._manage_library and not self._read_only:
            rows.append(self._add_provider_picker())
        for item in items:
            self._list.addItem(item)
            rows.append(item)
        if not items:
            empty = QListWidgetItem(
                self.tr("No subscription models available")
                if group == CHATGPT_GROUP
                else self.tr("No Provider files added")
            )
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(empty)
            rows.append(empty)
        self._group_rows[group] = rows
        self._set_group_expanded(group, self._group_expanded[group])

    def _add_provider_picker(self) -> QListWidgetItem:
        item = QListWidgetItem()
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(item)
        container = QWidget(self._list)
        container.setObjectName("provider-file-picker")
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        path_input = QLineEdit(container)
        path_input.setObjectName("config-path")
        path_input.setPlaceholderText(self.tr("Provider JSON path"))
        path_input.setAccessibleName(self.tr("Provider JSON path"))
        browse = QPushButton(self.tr("Browse..."), container)
        browse.setObjectName("browse-config")
        add = QPushButton(self.tr("Add"), container)
        add.setObjectName("load-config")
        row.addWidget(path_input, 1)
        row.addWidget(browse)
        row.addWidget(add)
        self._path_input = path_input
        path_input.returnPressed.connect(lambda: self._add_path(Path(path_input.text())))
        browse.clicked.connect(self._browse_path)
        add.clicked.connect(lambda: self._add_path(Path(path_input.text())))
        self._list.set_sized_item_widget(item, container)
        return item

    def _set_group_expanded(self, group: str, expanded: bool) -> None:
        self._group_expanded[group] = expanded
        header = self._group_headers.get(group)
        if header is not None and header.isChecked() != expanded:
            header.setChecked(expanded)
        for item in self._group_rows.get(group, ()):
            item.setHidden(not expanded)
            item_widget = self._list.itemWidget(item)
            if item_widget is not None:
                item_widget.setVisible(expanded)

    def _select_initial(self) -> None:
        if self._inherits_project_default and self._project_default is not None:
            self._show_project_default()
            return
        if not self._select_target(target_key(self._current.selection.target)):
            self._show_resolved(self._current)

    def _select_target(self, key: str) -> bool:
        for item in self.model_items():
            if target_key(item.model_descriptor.target) == key:
                group = _model_group(item.model_descriptor)
                self._set_group_expanded(group, True)
                self._list.setCurrentItem(item)
                return True
        return False

    def _on_current_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if not isinstance(current, ModelListItem):
            return
        if self._inherit is not None and self._inherit.isChecked():
            self._inherit.blockSignals(True)
            self._inherit.setChecked(False)
            self._inherit.blockSignals(False)
        model = current.model_descriptor
        self._selected_model = model
        self._use_project_default = False
        if target_key(model.target) == target_key(self._current.selection.target):
            variant = self._current.selection.variant
        else:
            default_variant = model.default_variant
            variant = default_variant.key if default_variant is not None else None
        self._render_model(model, variant)

    def _inherit_toggled(self, checked: bool) -> None:
        if checked:
            self._show_project_default()
            return
        current_key = target_key(self._current.selection.target)
        if not self._select_target(current_key) and self._selected_model is not None:
            variant = self._selected_selection.variant if self._selected_selection else None
            self._render_model(self._selected_model, variant)

    def _show_project_default(self) -> None:
        if self._project_default is None:
            return
        self._list.clearSelection()
        self._list.setCurrentItem(None)
        self._use_project_default = True
        self._render_model(
            self._project_default.model,
            self._project_default.selection.variant,
        )
        self._variant_picker.setEnabled(False)

    def _render_model(
        self,
        model: LLMModelDescriptor,
        variant: LLMVariantKey | None,
    ) -> None:
        self._selected_model = model
        self._variant_keys = {item.key.id: item.key for item in model.variants}
        selected_key = variant.id if variant is not None else None
        options = [VariantOption(item.key.id, self._variant_text(item)) for item in model.variants]
        missing_label = self.tr("Unavailable variant · {variant}").format(
            variant=selected_key or self.tr("not selected")
        )
        self._variant_picker.set_options(
            options,
            selected_key=selected_key,
            missing_label=missing_label,
        )
        is_chatgpt = isinstance(model.target, ChatGPTTarget)
        self._variant_label.setVisible(is_chatgpt)
        self._variant_picker.setVisible(is_chatgpt)
        self._variant_picker.setEnabled(
            is_chatgpt and not self._read_only and model.problem is None
        )

        if variant is None:
            fallback_selection = (
                LLMSelection(model.target, model.variants[0].key)
                if model.variants
                else self._current.selection
            )
            self._selected_selection = None
            resolved = ResolvedLLMSelection(
                fallback_selection,
                model,
                None,
                LLMProblem(PROBLEM_VARIANT_UNRESOLVED),
            )
        else:
            selection = LLMSelection(model.target, variant)
            self._selected_selection = selection
            resolved = resolve_selection(selection, [model])
        self._show_resolved(resolved)

    def _show_resolved(self, resolved: ResolvedLLMSelection) -> None:
        self._resolved_preview = resolved
        model = resolved.model
        problem = resolved.problem or model.problem
        self._error.setText(llm_problem_message(problem))
        self._summary_fields.set_value("selection-model", model.label)
        self._summary_fields.set_value(
            "selection-source",
            self.tr("ChatGPT subscription")
            if isinstance(model.target, ChatGPTTarget)
            else self.tr("Provider file"),
        )
        status = self.tr("Available") if resolved.available else _short_problem(problem)
        if resolved.available and model.catalog_stale:
            status = self.tr("Available · cached catalog")
        self._summary_fields.set_value("selection-status", status)

        self._model_fields.set_value("model-id", model.model_id)
        self._model_fields.set_value("model-context", _format_tokens(model.max_context_size))
        self._model_fields.set_value("model-output", _format_tokens(model.max_tokens))
        self._model_fields.set_value(
            "model-capabilities",
            ", ".join(model.capabilities) or self.tr("Not specified"),
        )
        self._model_fields.set_value(
            "model-modalities",
            ", ".join(model.input_modalities) or self.tr("Not specified"),
        )
        self._provider_fields.set_value("provider-type", model.provider_type)
        self._provider_fields.set_value("provider-endpoint", model.endpoint)
        self._provider_fields.set_value("provider-credential", model.credential)
        self._provider_fields.set_value("provider-format", model.file_format)
        self._provider_fields.set_value(
            "provider-thinking",
            _provider_thinking_text(model),
        )
        if self._apply is not None:
            self._apply.setEnabled(resolved.available)
        if self._delete is not None:
            self._delete.setEnabled(
                isinstance(model.target, ProviderFileTarget)
                and not self._use_project_default
                and target_key(model.target) != target_key(self._current.selection.target)
            )

    def _variant_activated(self, key: str) -> None:
        model = self._selected_model
        variant = self._variant_keys.get(key)
        if model is None or variant is None:
            return
        self._use_project_default = False
        self._render_model(model, variant)

    def _variant_text(self, variant: LLMVariantDescriptor) -> str:
        key = variant.key
        if key.kind == "configured":
            label = self.tr("Configured in Provider file")
        elif key.kind == "provider_default":
            label = self.tr("Provider default")
        elif key.kind == "reasoning_effort":
            label = _reasoning_effort_text(key.value or "")
        else:
            label = self.tr("Choose a variant")
        if variant.is_default:
            label = self.tr("{variant} · Model default").format(variant=label)
        return label

    def _browse_start_directory(self) -> str:
        if hasattr(self, "_path_input"):
            text = self._path_input.text().strip()
            if text:
                expanded = Path(text).expanduser()
                if expanded.is_dir():
                    return str(expanded)
                if expanded.parent.is_dir():
                    return str(expanded.parent)
        return str(Path.home())

    def _browse_path(self) -> None:
        selected = pick_json_file(self, self._browse_start_directory())
        if selected is not None:
            self._add_path(selected)

    def _add_path(self, path: Path) -> None:
        try:
            model = self._inspector(path)
        except LLMInspectionError as exc:
            self._error.setText(llm_problem_message(exc.problem))
            if self._apply is not None:
                self._apply.setEnabled(False)
            return
        target = model.target
        if not isinstance(target, ProviderFileTarget):
            return
        try:
            if self._registrar is not None:
                self._registrar(target)
        except OSError as exc:
            self._error.setText(
                self.tr("Failed to save Provider file metadata: {reason}").format(reason=exc)
            )
            return
        self._models[target_key(target)] = model
        self._group_expanded[PROVIDER_FILE_GROUP] = True
        self._populate_list()
        self._select_target(target_key(target))

    def _remove_selected(self) -> None:
        model = self._selected_model
        if model is None or not isinstance(model.target, ProviderFileTarget):
            return
        if target_key(model.target) == target_key(self._current.selection.target):
            return
        try:
            if self._remover is not None:
                self._remover(model.target.path)
        except OSError as exc:
            self._error.setText(
                self.tr("Failed to remove Provider file: {reason}").format(reason=exc)
            )
            return
        self._models.pop(target_key(model.target), None)
        self._selected_model = None
        self._selected_selection = None
        self._populate_list()
        self._select_target(target_key(self._current.selection.target))

    def _apply_clicked(self) -> None:
        resolved = self._resolved_preview
        if resolved is None or not resolved.available:
            return
        selection = (
            self._project_default.selection
            if self._use_project_default and self._project_default is not None
            else self._selected_selection
        )
        if selection is None:
            return
        self.applied.emit(LLMSettingsResult(selection, self._use_project_default))
        self.accept()

    def set_models(
        self,
        models: Iterable[LLMModelDescriptor],
        *,
        chatgpt_connected: bool,
        current: ResolvedLLMSelection | None = None,
        project_default: ResolvedLLMSelection | None = None,
        inherits_project_default: bool | None = None,
    ) -> None:
        """Refresh catalog metadata without changing a stored Variant key."""

        refreshed = {
            key: model
            for key, model in self._models.items()
            if isinstance(model.target, ProviderFileTarget)
        }
        refreshed.update({target_key(model.target): model for model in models})

        current_selection = current.selection if current is not None else self._current.selection
        project_selection = (
            project_default.selection
            if project_default is not None
            else self._project_default.selection
            if self._project_default is not None
            else None
        )
        for selection in (current_selection, project_selection):
            if selection is None or target_key(selection.target) in refreshed:
                continue
            target = selection.target
            if isinstance(target, ChatGPTTarget):
                problem = PROBLEM_MODEL_UNAVAILABLE if chatgpt_connected else PROBLEM_LOGIN_REQUIRED
                refreshed[target_key(target)] = unavailable_model(target, problem)
            else:
                previous = self._models.get(target_key(target))
                if previous is not None:
                    refreshed[target_key(target)] = previous

        descriptors = tuple(refreshed.values())
        self._models = refreshed
        self._chatgpt_connected = chatgpt_connected
        self._current = current or resolve_selection(current_selection, descriptors)
        if project_default is not None:
            self._project_default = project_default
        elif self._project_default is not None:
            self._project_default = resolve_selection(project_selection, descriptors)
        if inherits_project_default is not None:
            self._inherits_project_default = inherits_project_default
            if self._inherit is not None:
                self._inherit.blockSignals(True)
                self._inherit.setChecked(inherits_project_default)
                self._inherit.blockSignals(False)
        self._populate_list()
        if (
            self._use_project_default or self._inherits_project_default
        ) and self._project_default is not None:
            self._show_project_default()
        else:
            self._select_target(target_key(self._current.selection.target))

    def select_provider_file(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        for item in self.model_items():
            target = item.model_descriptor.target
            if isinstance(target, ProviderFileTarget) and target.path == resolved:
                self._list.setCurrentItem(item)
                return True
        return False

    def select_project_default(self) -> bool:
        if self._inherit is None:
            return False
        self._inherit.setChecked(True)
        return True

    def selected_selection(self) -> LLMSelection | None:
        return self._selected_selection

    def model_items(self) -> list[ModelListItem]:
        return [
            item
            for row in range(self._list.count())
            if isinstance((item := self._list.item(row)), ModelListItem)
        ]


def pick_json_file(parent: QWidget | None, start_directory: str) -> Path | None:
    selected, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        QCoreApplication.translate("LLMSettingsDialog", "Select Kimix Provider file"),
        start_directory,
        QCoreApplication.translate("LLMSettingsDialog", "JSON files (*.json)"),
    )
    return Path(selected) if selected else None


def llm_problem_message(problem: LLMProblem | None) -> str:
    """Translate one machine-readable LLM problem at the Qt boundary."""

    if problem is None:
        return ""
    template = _PROBLEM_TEMPLATES.get(problem.kind)
    if template is None:
        return str(problem)
    return template().format(path=problem.path or "", reason=problem.reason)


def _short_problem(problem: LLMProblem | None) -> str:
    if problem is None:
        return QCoreApplication.translate("LLMSettingsDialog", "Unavailable")
    labels = {
        PROBLEM_CREDENTIAL_MISSING: QCoreApplication.translate(
            "LLMSettingsDialog", "API key missing"
        ),
        PROBLEM_FILE_MISSING: QCoreApplication.translate("LLMSettingsDialog", "File missing"),
        PROBLEM_LOGIN_REQUIRED: QCoreApplication.translate("LLMSettingsDialog", "Connect ChatGPT"),
        PROBLEM_MODEL_UNAVAILABLE: QCoreApplication.translate(
            "LLMSettingsDialog", "Model unavailable"
        ),
        PROBLEM_VARIANT_UNAVAILABLE: QCoreApplication.translate(
            "LLMSettingsDialog", "Variant unavailable"
        ),
        PROBLEM_VARIANT_UNRESOLVED: QCoreApplication.translate(
            "LLMSettingsDialog", "Choose a variant"
        ),
    }
    return labels.get(
        problem.kind,
        QCoreApplication.translate("LLMSettingsDialog", "Unavailable"),
    )


def _reasoning_effort_text(effort: str) -> str:
    known = {
        "none": lambda: QCoreApplication.translate("LLMSettingsDialog", "None ({value})"),
        "minimal": lambda: QCoreApplication.translate("LLMSettingsDialog", "Minimal ({value})"),
        "low": lambda: QCoreApplication.translate("LLMSettingsDialog", "Low ({value})"),
        "medium": lambda: QCoreApplication.translate("LLMSettingsDialog", "Medium ({value})"),
        "high": lambda: QCoreApplication.translate("LLMSettingsDialog", "High ({value})"),
        "xhigh": lambda: QCoreApplication.translate("LLMSettingsDialog", "Extra high ({value})"),
        "max": lambda: QCoreApplication.translate("LLMSettingsDialog", "Maximum ({value})"),
    }
    formatter = known.get(effort)
    return formatter().format(value=effort) if formatter is not None else effort


def _provider_thinking_text(model: LLMModelDescriptor) -> str:
    if isinstance(model.target, ChatGPTTarget):
        return QCoreApplication.translate("LLMSettingsDialog", "Selected by Variant")
    effort = model.configured_reasoning_effort
    stream = model.show_thinking_stream
    if effort is None and stream is None:
        return QCoreApplication.translate("LLMSettingsDialog", "Configured in Provider file")
    stream_text = (
        QCoreApplication.translate("LLMSettingsDialog", "stream on")
        if stream
        else QCoreApplication.translate("LLMSettingsDialog", "stream off")
    )
    return QCoreApplication.translate("LLMSettingsDialog", "effort {effort} · {stream}").format(
        effort=effort or QCoreApplication.translate("LLMSettingsDialog", "not specified"),
        stream=stream_text,
    )


def _format_tokens(value: int | None) -> str:
    if value is None:
        return QCoreApplication.translate("LLMSettingsDialog", "Not specified")
    return QCoreApplication.translate("LLMSettingsDialog", "{count} tokens").format(
        count=f"{value:,}"
    )


_PROBLEM_TEMPLATES: dict[str, Callable[[], str]] = {
    PROBLEM_NOT_JSON: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Kimix Provider file must be JSON: {path}"
    ),
    PROBLEM_FILE_MISSING: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Provider file does not exist: {path}"
    ),
    PROBLEM_INVALID_JSON: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid Provider JSON {path}: {reason}"
    ),
    PROBLEM_NOT_AN_OBJECT: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Provider JSON must contain an object: {path}"
    ),
    PROBLEM_INVALID_PROVIDER_FILE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid Kimix Provider file {path}: {reason}"
    ),
    PROBLEM_PROVIDER_FILE_UNAVAILABLE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Provider file is unavailable: {path}"
    ),
    PROBLEM_CREDENTIAL_MISSING: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "No API key or OAuth credential is configured: {path}"
    ),
    PROBLEM_INVALID_SESSION_SELECTION: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid session LLM selection: {path}"
    ),
    PROBLEM_LOGIN_REQUIRED: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Connect ChatGPT to use this subscription model."
    ),
    PROBLEM_MODEL_UNAVAILABLE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "This model is not available for the connected account."
    ),
    PROBLEM_VARIANT_UNAVAILABLE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "The saved model variant is no longer available."
    ),
    PROBLEM_VARIANT_UNRESOLVED: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Choose a model variant before using this configuration."
    ),
}
