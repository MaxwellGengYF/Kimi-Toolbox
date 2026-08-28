"""Provider-grouped LLM configuration and parameter selection dialogs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.llm import (
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_MODEL_UNAVAILABLE,
    LLMInspectionError,
    LLMModelDescriptor,
    LLMSelection,
    ParameterAssignment,
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
    ParameterForm,
    ParameterPicker,
)
from kimix_gui.qt.llm_text import (
    format_tokens,
    problem_message,
    provider_thinking_text,
    provider_title,
    short_problem,
)
from kimix_gui.qt.styling import CardLevel, Level, Role, Tone, Variant, style

ModelInspector = Callable[[Path], LLMModelDescriptor]
ProviderFileRegistrar = Callable[[ProviderFileTarget], None]
ProviderFileRemover = Callable[[Path], None]

@dataclass(frozen=True, slots=True)
class LLMSettingsResult:
    """All metadata needed to commit one dialog selection."""

    selection: LLMSelection
    use_project_default: bool = False


class ModelListItem(QPushButton):
    """One selectable model row nested inside a provider card."""

    def __init__(self, model: LLMModelDescriptor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_descriptor = model
        self.setObjectName("provider-model-row")
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.refresh()

    def refresh(self) -> None:
        label = self.model_descriptor.label
        self.setText(label)
        self.setToolTip(label)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        height = max(
            DARK.sizing.settings_row_min_height,
            self.fontMetrics().lineSpacing() * 2 + DARK.sizing.settings_row_padding,
        )
        return QSize(hint.width(), height)


class LLMSettingsDialog(QDialog):
    """Select a model card and exact values for every parameter it exposes."""

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
        inspector: ModelInspector = inspect_provider_file,
        registrar: ProviderFileRegistrar | None = None,
        remover: ProviderFileRemover | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings-dialog")
        self.setWindowTitle(self.tr("LLM configuration"))
        self.setModal(True)
        self.resize(960, 620)
        self._current = current
        self._project_default = project_default
        self._inherits_project_default = inherits_project_default
        self._scope_label = scope_label
        self._manage_library = manage_library
        self._read_only = read_only
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
        self._model_buttons: dict[str, ModelListItem] = {}
        self._provider_cards: dict[str, QFrame] = {}
        current_provider = current.model.target.provider_id
        self._provider_expanded: dict[str, bool] = {current_provider: True}
        self._narrow = False
        self._build()
        self._select_initial()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.addLayout(self._header())
        self._error = QLabel("", self)
        self._error.setObjectName("settings-error")
        self._error.setWordWrap(True)
        style(self._error, tone=Tone.DANGER)
        root.addWidget(self._error)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setObjectName("settings-body")
        self._splitter.addWidget(self._sources_pane())
        self._splitter.addWidget(self._details_pane())
        self._splitter.setSizes([360, 600])
        root.addWidget(self._splitter, 1)
        root.addWidget(self._footer())
        self._populate_cards()

    def _header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        title = QLabel(self.tr("LLM configuration"), self)
        title.setObjectName("settings-title")
        style(title, role=Role.DISPLAY, level=Level.TWO)
        scope = QLabel(self._scope_label, self)
        scope.setObjectName("settings-scope")
        style(scope, tone=Tone.MUTED)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(scope)
        return header

    def _sources_pane(self) -> QWidget:
        sources = QWidget(self)
        sources.setObjectName("config-sources")
        layout = QVBoxLayout(sources)
        title = QLabel(self.tr("ACTIVE MODEL") if self._read_only else self.tr("PROVIDERS"))
        title.setObjectName("config-sources-title")
        style(title, role=Role.TITLE)
        layout.addWidget(title)

        self._inherit: QCheckBox | None = None
        if self._project_default is not None and not self._read_only:
            inherit = QCheckBox(self.tr("Follow project default"), sources)
            inherit.setObjectName("inherit-project-default")
            inherit.setAccessibleDescription(
                self.tr("Use the project's saved model and parameters for this session")
            )
            inherit.setChecked(self._inherits_project_default)
            inherit.toggled.connect(self._inherit_toggled)
            self._inherit = inherit
            layout.addWidget(inherit)

        self._cards_scroll = QScrollArea(sources)
        self._cards_scroll.setObjectName("provider-cards")
        self._cards_scroll.setWidgetResizable(True)
        self._cards_host = QWidget(self._cards_scroll)
        self._cards_host.setObjectName("provider-cards-content")
        self._cards_layout = QVBoxLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._cards_scroll.setWidget(self._cards_host)
        layout.addWidget(self._cards_scroll, 1)
        return sources

    def _details_pane(self) -> QWidget:
        details = QWidget(self)
        details.setObjectName("config-details")
        layout = QVBoxLayout(details)
        title = QLabel(self.tr("MODEL SELECTION"), details)
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

        self._parameters_title = QLabel(self.tr("MODEL PARAMETERS"), details)
        self._parameters_title.setObjectName("model-parameters-title")
        style(self._parameters_title, role=Role.OVERLINE)
        layout.addWidget(self._parameters_title)
        self._parameter_form = ParameterForm(details)
        self._parameter_form.setObjectName("model-parameters")
        self._parameter_form.value_activated.connect(self._parameter_activated)
        layout.addWidget(self._parameter_form)

        model_title = QLabel(self.tr("MODEL DETAILS"), details)
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

        provider_title = QLabel(self.tr("PROVIDER DETAILS"), details)
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
        self._delete: QPushButton | None = None
        extra: list[QPushButton] = []
        if self._manage_library and not self._read_only:
            delete = QPushButton(self.tr("Remove"), self)
            delete.setObjectName("delete-config")
            delete.setEnabled(False)
            delete.clicked.connect(self._remove_selected)
            self._delete = delete
            extra.append(delete)
        close = QPushButton(self.tr("Close") if self._read_only else self.tr("Cancel"), self)
        close.setObjectName("close-settings" if self._read_only else "cancel-settings")
        self._apply: QPushButton | None = None
        if not self._read_only:
            apply_button = QPushButton(self.tr("Use model"), self)
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

    def _populate_cards(self) -> None:
        selected_key = (
            target_key(self._selected_model.target) if self._selected_model is not None else None
        )
        while (item := self._cards_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._model_buttons.clear()
        self._provider_cards.clear()
        grouped: dict[str, list[LLMModelDescriptor]] = {}
        for model in sorted(self._models.values(), key=lambda item: (item.priority, item.label)):
            grouped.setdefault(model.target.provider_id, []).append(model)
        provider_ids = list(grouped)
        if self._manage_library and "provider_file" not in provider_ids:
            provider_ids.append("provider_file")
        if not self._read_only and "chatgpt" not in provider_ids:
            provider_ids.insert(0, "chatgpt")
        for provider_id in provider_ids:
            self._add_provider_card(provider_id, grouped.get(provider_id, []))
        self._cards_layout.addStretch()
        if selected_key is not None:
            self._select_target(selected_key)

    def _add_provider_card(
        self,
        provider_id: str,
        models: list[LLMModelDescriptor],
    ) -> None:
        card = QFrame(self._cards_host)
        card.setObjectName("provider-card")
        card.setProperty("providerId", provider_id)
        style(card, card=CardLevel.PANEL)
        card_layout = QVBoxLayout(card)
        expanded = self._provider_expanded.get(
            provider_id,
            provider_id == self._current.model.target.provider_id,
        )
        header = DisclosureHeader(
            self.tr("{provider} · {count}").format(
                provider=provider_title(provider_id),
                count=len(models),
            ),
            expanded=expanded,
            expanded_tooltip=self.tr("Collapse provider"),
            collapsed_tooltip=self.tr("Expand provider"),
            parent=card,
        )
        header.setObjectName(
            "chatgpt-config-group"
            if provider_id == "chatgpt"
            else "provider-config-group"
        )
        card_layout.addWidget(header)
        body = QWidget(card)
        body.setObjectName("provider-card-body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        requires_login = not models or any(
            model.problem is not None and model.problem.kind == PROBLEM_LOGIN_REQUIRED
            for model in models
        )
        if provider_id == "chatgpt" and requires_login and not self._read_only:
            login = QPushButton(self.tr("Connect ChatGPT"), body)
            login.setObjectName("connect-chatgpt-models")
            style(login, variant=Variant.PRIMARY)
            login.clicked.connect(self.connect_chatgpt.emit)
            body_layout.addWidget(login)
        if provider_id == "provider_file" and self._manage_library and not self._read_only:
            body_layout.addWidget(self._provider_file_picker(body))

        button_group = QButtonGroup(card)
        button_group.setExclusive(True)
        for model in models:
            button = ModelListItem(model, body)
            button.clicked.connect(lambda _checked=False, item=model: self._select_model(item))
            button_group.addButton(button)
            body_layout.addWidget(button)
            self._model_buttons[target_key(model.target)] = button
        if not models:
            empty = QLabel(
                self.tr("No subscription models available")
                if provider_id == "chatgpt"
                else self.tr("No models configured"),
                body,
            )
            empty.setObjectName("empty-provider-card")
            style(empty, tone=Tone.MUTED)
            body_layout.addWidget(empty)
        card_layout.addWidget(body)
        body.setVisible(expanded)
        header.toggled.connect(
            lambda is_expanded, source=provider_id, panel=body: self._toggle_provider(
                source,
                panel,
                is_expanded,
            )
        )
        self._provider_cards[provider_id] = card
        self._cards_layout.addWidget(card)

    def _provider_file_picker(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
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
        return container

    def _toggle_provider(self, provider_id: str, body: QWidget, expanded: bool) -> None:
        self._provider_expanded[provider_id] = expanded
        body.setVisible(expanded)

    def _select_initial(self) -> None:
        if self._inherits_project_default and self._project_default is not None:
            self._show_project_default()
            return
        if not self._select_target(target_key(self._current.selection.target)):
            self._show_resolved(self._current)

    def _select_target(self, key: str) -> bool:
        button = self._model_buttons.get(key)
        if button is None:
            return False
        provider_id = button.model_descriptor.target.provider_id
        card = self._provider_cards.get(provider_id)
        if card is not None:
            body = card.findChild(QWidget, "provider-card-body")
            header = card.findChild(DisclosureHeader)
            if body is not None:
                body.setVisible(True)
            if header is not None:
                header.setChecked(True)
            self._provider_expanded[provider_id] = True
        button.setChecked(True)
        self._select_model(button.model_descriptor)
        return True

    def _select_model(self, model: LLMModelDescriptor) -> None:
        if self._inherit is not None and self._inherit.isChecked():
            self._inherit.blockSignals(True)
            self._inherit.setChecked(False)
            self._inherit.blockSignals(False)
        self._selected_model = model
        self._use_project_default = False
        assignment = (
            self._current.materialized_selection.parameters
            if target_key(model.target) == target_key(self._current.selection.target)
            else model.default_assignment
        )
        self._render_model(model, assignment)

    def _inherit_toggled(self, checked: bool) -> None:
        if checked:
            self._show_project_default()
            return
        current_key = target_key(self._current.selection.target)
        if not self._select_target(current_key) and self._selected_model is not None:
            assignment = (
                self._selected_selection.parameters
                if self._selected_selection is not None
                else self._selected_model.default_assignment
            )
            self._render_model(self._selected_model, assignment)

    def _show_project_default(self) -> None:
        if self._project_default is None:
            return
        for button in self._model_buttons.values():
            button.setChecked(False)
        self._use_project_default = True
        self._render_model(
            self._project_default.model,
            self._project_default.materialized_selection.parameters,
        )
        self._parameter_form.set_controls_enabled(False)

    def _render_model(
        self,
        model: LLMModelDescriptor,
        assignment: ParameterAssignment,
    ) -> None:
        self._selected_model = model
        materialized = assignment
        for parameter in model.parameters:
            if materialized.get(parameter.axis) is None and parameter.default is not None:
                materialized = materialized.with_value(parameter.axis, parameter.default.value)
        selection = LLMSelection(model.target, materialized, pinned=True)
        resolved = resolve_selection(selection, [model])
        self._selected_selection = selection
        self._render_parameter_controls(model, materialized, resolved)
        self._show_resolved(resolved)

    def _render_parameter_controls(
        self,
        model: LLMModelDescriptor,
        assignment: ParameterAssignment,
        resolved: ResolvedLLMSelection,
    ) -> None:
        self._parameter_form.set_parameters(
            model.parameters,
            assignment,
            resolved=resolved.resolved,
            enabled=(
                not self._read_only
                and not self._use_project_default
                and model.problem is None
            ),
        )
        self._parameters_title.setVisible(bool(self._parameter_form.pickers))

    def _show_resolved(self, resolved: ResolvedLLMSelection) -> None:
        self._resolved_preview = resolved
        model = resolved.model
        problem = resolved.problem or model.problem
        parameter_problem = any(item.problem == problem for item in resolved.resolved)
        self._error.setText("" if parameter_problem else problem_message(problem))
        self._summary_fields.set_value("selection-model", model.label)
        self._summary_fields.set_value(
            "selection-source",
            provider_title(model.target.provider_id),
        )
        status = self.tr("Available") if resolved.available else short_problem(problem)
        if resolved.available and model.catalog_stale:
            status = self.tr("Available · cached catalog")
        self._summary_fields.set_value("selection-status", status)
        self._model_fields.set_value("model-id", model.model_id)
        self._model_fields.set_value("model-context", format_tokens(model.max_context_size))
        self._model_fields.set_value("model-output", format_tokens(model.max_tokens))
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
        self._provider_fields.set_value("provider-thinking", provider_thinking_text(resolved))
        if self._apply is not None:
            self._apply.setEnabled(resolved.available)
        if self._delete is not None:
            self._delete.setEnabled(
                isinstance(model.target, ProviderFileTarget)
                and not self._use_project_default
                and target_key(model.target) != target_key(self._current.selection.target)
            )

    def _parameter_activated(self, axis: str, value: str) -> None:
        model = self._selected_model
        selection = self._selected_selection
        if model is None or selection is None:
            return
        parameter = next((item for item in model.parameters if item.axis == axis), None)
        if parameter is None or parameter.option(value) is None:
            return
        self._use_project_default = False
        self._render_model(model, selection.parameters.with_value(axis, value))

    def parameter_picker(self, axis: str) -> ParameterPicker | None:
        return self._parameter_form.picker(axis)

    def _browse_start_directory(self) -> str:
        text = self._path_input.text().strip() if hasattr(self, "_path_input") else ""
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
            self._error.setText(problem_message(exc.problem))
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
        self._provider_expanded["provider_file"] = True
        self._populate_cards()
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
        self._populate_cards()
        self._select_target(target_key(self._current.selection.target))

    def _apply_clicked(self) -> None:
        resolved = self._resolved_preview
        if resolved is None or not resolved.available:
            return
        selection = (
            self._project_default.materialized_selection
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
        current: ResolvedLLMSelection | None = None,
        project_default: ResolvedLLMSelection | None = None,
        inherits_project_default: bool | None = None,
    ) -> None:
        """Refresh provider catalogs without changing stored parameter assignments."""

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
            refreshed[target_key(selection.target)] = unavailable_model(
                selection.target,
                PROBLEM_MODEL_UNAVAILABLE,
            )
        descriptors = tuple(refreshed.values())
        self._models = refreshed
        self._current = current or resolve_selection(current_selection, descriptors)
        if project_default is not None:
            self._project_default = project_default
        elif self._project_default is not None and project_selection is not None:
            self._project_default = resolve_selection(project_selection, descriptors)
        if inherits_project_default is not None:
            self._inherits_project_default = inherits_project_default
            if self._inherit is not None:
                self._inherit.blockSignals(True)
                self._inherit.setChecked(inherits_project_default)
                self._inherit.blockSignals(False)
        self._populate_cards()
        if (
            self._use_project_default or self._inherits_project_default
        ) and self._project_default is not None:
            self._show_project_default()
        else:
            self._select_target(target_key(self._current.selection.target))

    def select_provider_file(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        for button in self.model_items():
            target = button.model_descriptor.target
            if isinstance(target, ProviderFileTarget) and target.path == resolved:
                self._select_target(target_key(target))
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
        return list(self._model_buttons.values())


def pick_json_file(parent: QWidget | None, start_directory: str) -> Path | None:
    selected, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        QCoreApplication.translate("LLMSettingsDialog", "Select Kimix Provider file"),
        start_directory,
        QCoreApplication.translate("LLMSettingsDialog", "JSON files (*.json)"),
    )
    return Path(selected) if selected else None
