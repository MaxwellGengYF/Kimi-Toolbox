"""LLM configuration dialog: library on the left, redacted details on the right."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.llm_config import (
    CONFIG_FILE_MISSING,
    CONFIG_INVALID,
    CONFIG_INVALID_JSON,
    CONFIG_INVALID_SESSION_REFERENCE,
    CONFIG_NOT_AN_OBJECT,
    CONFIG_NOT_JSON,
    CONFIG_UNAVAILABLE,
    ChatGPTModelReference,
    ConfigProblem,
    LLMConfigError,
    LLMConfigReference,
    LLMReference,
    inspect_llm_config,
    llm_reference_available,
)
from kimix_gui.qt.components import DialogFooter, KeyValueList, SettingsList
from kimix_gui.qt.styling import Level, Role, Tone, Variant, style

# ``QListWidgetItem`` is not a ``QObject`` and module functions have no ``self``, so
# the copy in those places goes through ``QCoreApplication.translate`` with an
# explicit context. It is spelled out at every call site because lupdate only reads
# literal arguments -- a helper wrapper or a constant would extract nothing.
# ``LLMSettingsDialog`` is deliberately reused as that context: the dialog owns the
# list and the formatters, and sharing it keeps one catalog entry for strings both
# sides show (``Not specified``).
# (Plain ``#``, never ``#:`` -- lupdate treats ``#:`` as an extracomment and would
# staple this paragraph onto the next translatable string in the file.)

ConfigInspector = Callable[[Path], LLMConfigReference]
ConfigRegistrar = Callable[[LLMConfigReference], None]
ConfigRemover = Callable[[Path], None]


def _reference_key(reference: LLMReference) -> str:
    if isinstance(reference, ChatGPTModelReference):
        return f"chatgpt:{reference.model_name}"
    return f"config_file:{reference.path.resolve(strict=False)}"


@dataclass(frozen=True, slots=True)
class LLMSettingsResult:
    reference: LLMReference
    use_project_default: bool = False


class ConfigListItem(QListWidgetItem):
    def __init__(
        self,
        reference: LLMReference,
        *,
        project_default: bool = False,
        active: bool = False,
    ) -> None:
        super().__init__()
        self.reference = reference
        self.project_default = project_default
        self.active = active
        self._refresh()

    def update_reference(self, reference: LLMReference) -> None:
        self.reference = reference
        self._refresh()

    def _refresh(self) -> None:
        available = llm_reference_available(self.reference)
        chatgpt_unavailable = isinstance(self.reference, ChatGPTModelReference) and not available
        if self.project_default:
            name = QCoreApplication.translate("LLMSettingsDialog", "Project default")
            if chatgpt_unavailable:
                status = QCoreApplication.translate(
                    "LLMSettingsDialog", "Connect ChatGPT · {label}"
                ).format(label=self.reference.label)
            elif not available:
                status = QCoreApplication.translate(
                    "LLMSettingsDialog", "Missing · {label}"
                ).format(label=self.reference.label)
            else:
                status = f"{self.reference.label} · {self.reference.provider_type}"
        else:
            name = self.reference.label
            if chatgpt_unavailable:
                status = QCoreApplication.translate(
                    "LLMSettingsDialog", "Connect ChatGPT · {model}"
                ).format(model=self.reference.model_name)
            elif not available:
                status = QCoreApplication.translate(
                    "LLMSettingsDialog", "Missing · {provider} · {model}"
                ).format(
                    provider=self.reference.provider_type,
                    model=self.reference.model_name,
                )
            else:
                status = f"{self.reference.provider_type} · {self.reference.model_name}"
        if self.active:
            status = QCoreApplication.translate("LLMSettingsDialog", "IN USE · {status}").format(
                status=status
            )
        self.setText(f"{name}\n{status}")


class LLMSettingsDialog(QDialog):
    """Choose or inspect a config path and preview its redacted LLM settings."""

    applied = Signal(object)
    connect_chatgpt = Signal()

    def __init__(
        self,
        *,
        current: LLMReference,
        references: Iterable[LLMReference],
        scope_label: str,
        project_default: LLMReference | None = None,
        inherits_project_default: bool = False,
        manage_library: bool = False,
        read_only: bool = False,
        chatgpt_connected: bool = False,
        inspector: ConfigInspector = inspect_llm_config,
        registrar: ConfigRegistrar | None = None,
        remover: ConfigRemover | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings-dialog")
        self.setWindowTitle(self.tr("LLM configuration"))
        self.setModal(True)
        self.resize(920, 560)
        self._current = current
        self._references: dict[str, LLMReference] = {}
        for reference in [*references, current]:
            self._references[_reference_key(reference)] = reference
        self._scope_label = scope_label
        self._project_default = project_default
        self._inherits_project_default = inherits_project_default
        self._manage_library = manage_library
        self._read_only = read_only
        self._chatgpt_connected = chatgpt_connected
        self._inspector = inspector
        self._registrar = registrar
        self._remover = remover
        self._preview: LLMReference | None = None
        self._selected_reference: LLMReference | None = None
        self._use_project_default = False
        self._narrow = False
        self._build()
        self._select_initial()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.addLayout(self._header())

        self._add_path_picker(root)

        self._error = QLabel("")
        self._error.setObjectName("settings-error")
        style(self._error, tone=Tone.DANGER)
        root.addWidget(self._error)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("settings-body")
        self._splitter.addWidget(self._sources_pane())
        self._splitter.addWidget(self._details_pane())
        self._splitter.setSizes([320, 560])
        root.addWidget(self._splitter, 1)
        root.addWidget(self._footer())

        self._populate_list()
        self._list.currentItemChanged.connect(self._on_current_changed)

    def _header(self) -> QHBoxLayout:
        """Dialog heading: what this dialog edits, and which scope it will save to."""
        header = QHBoxLayout()
        title = QLabel(self.tr("LLM configuration"))
        title.setObjectName("settings-title")
        # ``display`` level 2 is what a dialog heading is, and it is what
        # ``preferences-title`` already used. As ``title`` this heading rendered at the
        # same size and weight as the ``Sources`` / ``Details`` section titles *inside*
        # the dialog, so the hierarchy was flat where it mattered most.
        style(title, role=Role.DISPLAY, level=Level.TWO)
        scope = QLabel(self._scope_label)
        scope.setObjectName("settings-scope")
        style(scope, tone=Tone.MUTED)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(scope)
        return header

    def _add_path_picker(self, root: QVBoxLayout) -> None:
        """The provider-file row: path field, Browse, and Add, above the config list."""
        # Builds into ``root`` instead of returning a widget: the overline and the row
        # are siblings of the dialog's own layout, and giving them a container of their
        # own would nest a layout that was never there.
        if self._manage_library:
            path_label = QLabel(self.tr("KIMIX PROVIDER CONFIG (.JSON)"))
            path_label.setObjectName("config-path-label")
            # Hand-uppercased because Qt style sheets have no ``text-transform``;
            # that shape is what ``overline`` is for (see ``PREVIEW``, ``TODOS``).
            style(path_label, role=Role.OVERLINE)
            root.addWidget(path_label)
            path_row = QHBoxLayout()
            initial_path = (
                str(self._current.path) if isinstance(self._current, LLMConfigReference) else ""
            )
            self._path_input = QLineEdit(initial_path)
            self._path_input.setObjectName("config-path")
            self._path_input.setPlaceholderText(r"C:\path\to\provider.json")
            browse = QPushButton(self.tr("Browse..."))
            browse.setObjectName("browse-config")
            browse.setToolTip(self.tr("Choose a JSON file"))
            load = QPushButton(self.tr("Add config"))
            load.setObjectName("load-config")
            path_row.addWidget(self._path_input, 1)
            path_row.addWidget(browse)
            path_row.addWidget(load)
            root.addLayout(path_row)
            self._path_input.returnPressed.connect(
                lambda: self._add_path(Path(self._path_input.text()))
            )
            browse.clicked.connect(self._browse_path)
            load.clicked.connect(lambda: self._add_path(Path(self._path_input.text())))
        else:
            self._path_input = None

    def _sources_pane(self) -> QWidget:
        """Left splitter pane: the configs this dialog can choose between."""
        sources = QWidget()
        sources.setObjectName("config-sources")
        sources_layout = QVBoxLayout(sources)
        sources_title = QLabel(
            self.tr("ACTIVE CONFIG") if self._read_only else self.tr("AVAILABLE CONFIGS")
        )
        sources_title.setObjectName("config-sources-title")
        # The two splitter panes are a symmetric pair, so their headings have to
        # match. Only the details one was ever styled; this is the omission.
        style(sources_title, role=Role.TITLE)
        self._list = SettingsList()
        self._list.setObjectName("config-list")
        sources_layout.addWidget(sources_title)
        self._chatgpt_login = QPushButton(self.tr("Connect ChatGPT"))
        self._chatgpt_login.setObjectName("connect-chatgpt-models")
        self._chatgpt_login.setToolTip(self.tr("Use ChatGPT subscription models"))
        style(self._chatgpt_login, variant=Variant.PRIMARY)
        self._chatgpt_login.setVisible(not self._chatgpt_connected and not self._read_only)
        self._chatgpt_login.clicked.connect(self.connect_chatgpt.emit)
        sources_layout.addWidget(self._chatgpt_login)
        sources_layout.addWidget(self._list)
        return sources

    def _details_pane(self) -> QWidget:
        """Right splitter pane: the redacted fields of whichever config is selected."""
        details = QWidget()
        details.setObjectName("config-details")
        details_layout = QVBoxLayout(details)
        details_title = QLabel(self.tr("CONFIG DETAILS"))
        details_title.setObjectName("config-details-title")
        style(details_title, role=Role.TITLE)
        details_layout.addWidget(details_title)
        fields = KeyValueList(
            (
                ("config-model", self.tr("Name")),
                ("config-format", self.tr("Format")),
                ("config-model-id", self.tr("Model ID")),
                ("config-provider", self.tr("Provider")),
                ("config-endpoint", self.tr("Endpoint")),
                ("config-credential", self.tr("Credential")),
                ("config-context", self.tr("Context")),
                ("config-output", self.tr("Max output")),
                ("config-capabilities", self.tr("Capabilities")),
                ("config-thinking", self.tr("Thinking")),
            )
        )
        self._fields = fields
        details_layout.addWidget(fields)
        details_layout.addStretch()
        return details

    def _footer(self) -> DialogFooter:
        """Dialog footer, minus whichever buttons this mode does not offer."""
        self._delete = None
        extra: list[QPushButton] = []
        if self._manage_library and not self._read_only:
            self._delete = QPushButton(self.tr("Remove"))
            self._delete.setObjectName("delete-config")
            self._delete.setEnabled(False)
            self._delete.clicked.connect(self._remove_selected)
            extra.append(self._delete)
        close = QPushButton(self.tr("Close") if self._read_only else self.tr("Cancel"))
        close.setObjectName("close-settings" if self._read_only else "cancel-settings")
        self._apply: QPushButton | None = None
        if not self._read_only:
            apply_btn = QPushButton(self.tr("Use config"))
            apply_btn.setObjectName("apply-settings")
            style(apply_btn, variant=Variant.PRIMARY)
            apply_btn.setEnabled(False)
            self._apply = apply_btn
            apply_btn.clicked.connect(self._apply_clicked)
        # Was ``Remove, Use config, Close``: the only footer in the app that did not
        # end with the action, so the corner of this dialog meant "never mind" while
        # the corner of every other one meant "do it".
        footer = DialogFooter(dismiss=close, extra=extra, confirm=self._apply, parent=self)
        close.clicked.connect(self.reject)
        return footer

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
        self._list.clear()
        if self._read_only:
            self._list.addItem(ConfigListItem(self._current, active=True))
        else:
            if self._project_default is not None:
                self._list.addItem(ConfigListItem(self._project_default, project_default=True))
            chatgpt = [
                reference
                for reference in self._references.values()
                if isinstance(reference, ChatGPTModelReference)
            ]
            external = [
                reference
                for reference in self._references.values()
                if isinstance(reference, LLMConfigReference)
            ]
            if chatgpt:
                self._add_group_header(
                    QCoreApplication.translate("LLMSettingsDialog", "CHATGPT SUBSCRIPTION")
                )
            for reference in chatgpt:
                self._list.addItem(ConfigListItem(reference))
            if external:
                self._add_group_header(
                    QCoreApplication.translate("LLMSettingsDialog", "PROVIDER FILES")
                )
            for reference in external:
                self._list.addItem(ConfigListItem(reference))

    def _add_group_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
        self._list.addItem(item)

    def set_chatgpt_references(
        self,
        references: Iterable[ChatGPTModelReference],
        *,
        connected: bool,
    ) -> None:
        """Refresh built-in rows without changing the selected default implicitly."""

        selected = self.selected_config()
        selected_key = _reference_key(selected) if selected is not None else None
        self._references = {
            key: reference
            for key, reference in self._references.items()
            if not key.startswith("chatgpt:")
        }
        for reference in references:
            self._references[_reference_key(reference)] = reference
            if isinstance(self._current, ChatGPTModelReference) and (
                self._current.model_name == reference.model_name
            ):
                self._current = reference
            if isinstance(self._project_default, ChatGPTModelReference) and (
                self._project_default.model_name == reference.model_name
            ):
                self._project_default = reference
        if isinstance(self._current, ChatGPTModelReference):
            current_key = _reference_key(self._current)
            if current_key not in self._references:
                self._current = replace(
                    self._current,
                    available=False,
                    problem_code="model_unavailable" if connected else "login_required",
                )
                self._references[current_key] = self._current
        if isinstance(self._project_default, ChatGPTModelReference) and not connected:
            self._project_default = replace(
                self._project_default,
                available=False,
                problem_code="login_required",
            )
        self._chatgpt_connected = connected
        self._chatgpt_login.setVisible(not connected and not self._read_only)
        self._populate_list()
        if selected_key is not None:
            for row in range(self._list.count()):
                item = self._list.item(row)
                if (
                    isinstance(item, ConfigListItem)
                    and _reference_key(item.reference) == selected_key
                ):
                    self._list.setCurrentRow(row)
                    return
        self._select_initial()

    def _select_initial(self) -> None:
        if self._project_default is not None and self._inherits_project_default:
            for item in self.config_items():
                if item.project_default:
                    self._list.setCurrentItem(item)
                    break
            self._select_project_default(self._project_default)
            return
        current_key = _reference_key(self._current)
        selected = next(
            (
                item
                for item in self.config_items()
                if not item.project_default and _reference_key(item.reference) == current_key
            ),
            None,
        )
        if selected is None:
            selected = next(
                (item for item in self.config_items() if not item.project_default), None
            )
        if selected is not None:
            self._list.setCurrentItem(selected)
        self._show_reference(
            self._current,
            error=self._current.error if isinstance(self._current, LLMConfigReference) else None,
        )

    def _on_current_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if not isinstance(current, ConfigListItem):
            return
        if current.project_default:
            self._select_project_default(current.reference)
        else:
            self._select_reference(current.reference)

    def _select_reference(self, reference: LLMReference) -> None:
        if isinstance(reference, LLMConfigReference):
            self._update_path_input(reference.path)
        self._show_reference(
            reference,
            error=reference.error if isinstance(reference, LLMConfigReference) else None,
        )

    def _select_project_default(self, reference: LLMReference) -> None:
        if isinstance(reference, LLMConfigReference):
            self._update_path_input(reference.path)
        self._show_reference(
            reference,
            error=reference.error if isinstance(reference, LLMConfigReference) else None,
            use_project_default=True,
        )

    def _update_path_input(self, path: Path) -> None:
        if self._path_input is not None:
            self._path_input.setText(str(path))

    def _browse_start_directory(self) -> str:
        candidates: list[Path] = []
        if self._path_input is not None:
            typed = self._path_input.text().strip()
            if typed:
                candidates.append(Path(typed))
        if isinstance(self._current, LLMConfigReference):
            candidates.append(self._current.path)
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded.is_file():
                return str(expanded.parent)
            if expanded.is_dir():
                return str(expanded)
            parent = expanded.parent
            if parent.is_dir():
                return str(parent)
        return str(Path.home())

    def _browse_path(self) -> None:
        selected = pick_json_file(self, self._browse_start_directory())
        if selected is None:
            return
        self._add_path(selected)

    def _add_path(self, path: Path) -> None:
        try:
            reference = self._inspector(path)
        except LLMConfigError as exc:
            self._preview = None
            self._error.setText(config_problem_message(exc.problem))
            self._apply.setEnabled(False)
            return
        try:
            if self._registrar is not None:
                self._registrar(reference)
        except OSError as exc:
            self._preview = None
            self._error.setText(
                self.tr("Failed to save configuration metadata: {reason}").format(reason=exc)
            )
            self._apply.setEnabled(False)
            return
        key = _reference_key(reference)
        if key not in self._references:
            self._references[key] = reference
            self._list.addItem(ConfigListItem(reference))
        else:
            self._references[key] = reference
            for row in range(self._list.count()):
                item = self._list.item(row)
                if (
                    isinstance(item, ConfigListItem)
                    and not item.project_default
                    and _reference_key(item.reference) == key
                ):
                    item.update_reference(reference)
                    break
        self._update_path_input(reference.path)
        self._show_reference(reference)

    def _remove_selected(self) -> None:
        reference = self._selected_reference
        if (
            not self._manage_library
            or reference is None
            or not isinstance(reference, LLMConfigReference)
            or _reference_key(reference) == _reference_key(self._current)
        ):
            return
        try:
            if self._remover is not None:
                self._remover(reference.path)
        except OSError as exc:
            self._error.setText(
                self.tr("Failed to remove configuration reference: {reason}").format(reason=exc)
            )
            return
        key = _reference_key(reference)
        self._references.pop(key, None)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if (
                isinstance(item, ConfigListItem)
                and not item.project_default
                and _reference_key(item.reference) == key
            ):
                self._list.takeItem(row)
                break
        current_key = _reference_key(self._current)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if (
                isinstance(item, ConfigListItem)
                and not item.project_default
                and _reference_key(item.reference) == current_key
            ):
                self._list.setCurrentRow(row)
                break
        self._select_reference(self._current)

    def _show_reference(
        self,
        reference: LLMReference,
        *,
        error: ConfigProblem | None = None,
        use_project_default: bool = False,
    ) -> None:
        self._selected_reference = reference
        available = error is None and llm_reference_available(reference)
        self._preview = reference if available else None
        self._use_project_default = use_project_default and available
        if isinstance(reference, ChatGPTModelReference) and not reference.available:
            self._error.setText(
                self.tr("This model is not available for the connected ChatGPT account.")
                if reference.problem_code == "model_unavailable"
                else self.tr("Connect ChatGPT to use this subscription model.")
            )
        else:
            self._error.setText(config_problem_message(error))
        values = {
            "config-model": reference.label,
            "config-format": (
                self.tr("Built-in")
                if isinstance(reference, ChatGPTModelReference)
                else reference.file_format
            ),
            "config-model-id": reference.model_name,
            "config-provider": reference.provider_type,
            "config-endpoint": reference.base_url,
            "config-credential": (
                self.tr("ChatGPT OAuth")
                if isinstance(reference, ChatGPTModelReference)
                else reference.credential
            ),
            "config-context": _format_tokens(reference.max_context_size),
            "config-output": _format_tokens(reference.max_tokens),
            "config-capabilities": ", ".join(reference.capabilities) or self.tr("Not specified"),
            # Two whole sentences instead of a translatable bare "on"/"off", which
            # has no single right rendering out of context.
            "config-thinking": self._thinking_description(reference),
        }
        for key, value in values.items():
            self._fields.set_value(key, value)
        if self._apply is not None:
            self._apply.setEnabled(available)
        if self._delete is not None:
            self._delete.setEnabled(
                isinstance(reference, LLMConfigReference)
                and not use_project_default
                and _reference_key(reference) != _reference_key(self._current)
            )

    def _thinking_description(self, reference: LLMReference) -> str:
        if isinstance(reference, ChatGPTModelReference):
            efforts = ", ".join(reference.supported_efforts)
            default_effort = reference.default_reasoning_effort
            if efforts and default_effort:
                return self.tr("default {default} · efforts {efforts}").format(
                    default=default_effort,
                    efforts=efforts,
                )
            if efforts:
                return self.tr("server default · efforts {efforts}").format(efforts=efforts)
            return self.tr("server default")
        return (
            self.tr("effort {effort} · stream on")
            if reference.show_thinking_stream
            else self.tr("effort {effort} · stream off")
        ).format(effort=reference.thinking_effort or self.tr("not specified"))

    def _apply_clicked(self) -> None:
        if self._preview is None:
            return
        result = LLMSettingsResult(self._preview, self._use_project_default)
        self.applied.emit(result)
        self.accept()

    def select_config(self, path: Path) -> bool:
        """Select the configuration stored at ``path``, as clicking its row would.

        Returns whether it was listed at all. The list widget is an implementation
        detail: what a caller means is "this configuration is the selected one now", and
        saying that through a row index makes them depend on how the library happens to
        be ordered -- and on the invisible extra row the project default adds.
        """
        target = path.resolve(strict=False)
        for item in self.config_items():
            if item.project_default:
                continue
            if (
                isinstance(item.reference, LLMConfigReference)
                and item.reference.path.resolve(strict=False) == target
            ):
                self._list.setCurrentItem(item)
                return True
        return False

    def select_project_default(self) -> bool:
        """Select the inherit-the-project-default row, if this scope offers one."""
        for item in self.config_items():
            if item.project_default:
                self._list.setCurrentItem(item)
                return True
        return False

    def selected_config(self) -> LLMReference | None:
        """The configuration whose details are on show, or ``None`` before any."""
        item = self._list.currentItem()
        return item.reference if isinstance(item, ConfigListItem) else None

    def config_items(self) -> list[ConfigListItem]:
        return [
            item
            for row in range(self._list.count())
            if isinstance((item := self._list.item(row)), ConfigListItem)
        ]


def pick_json_file(parent: QWidget | None, start_directory: str) -> Path | None:
    selected, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        QCoreApplication.translate("LLMSettingsDialog", "Select Kimix provider config"),
        start_directory,
        QCoreApplication.translate("LLMSettingsDialog", "JSON files (*.json)"),
    )
    if not selected:
        return None
    return Path(selected)


def config_problem_message(problem: ConfigProblem | None) -> str:
    """Phrase a pure-layer :class:`ConfigProblem` for the user.

    The pure layer raises structured problems because it cannot call ``tr()``; this is
    the single place that turns one into a sentence, and ``app.py`` reuses it for the
    toast it raises on the same failures. An unknown ``kind`` degrades to the
    exception's own English text rather than to an empty label.
    """

    if problem is None:
        return ""
    template = _PROBLEM_TEMPLATES.get(problem.kind)
    if template is None:
        return str(problem)
    return template().format(path=problem.path, reason=problem.reason)


# Lambdas, not a dict of strings: ``translate`` has to run at display time so that a
# catalog installed after import still applies. Each msgid is a literal argument,
# which is the only shape lupdate extracts.
_PROBLEM_TEMPLATES: dict[str, Callable[[], str]] = {
    CONFIG_NOT_JSON: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Kimix configuration must be a JSON file: {path}"
    ),
    CONFIG_FILE_MISSING: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Configuration file does not exist: {path}"
    ),
    CONFIG_INVALID_JSON: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid JSON configuration {path}: {reason}"
    ),
    CONFIG_NOT_AN_OBJECT: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Kimix configuration must contain a JSON object: {path}"
    ),
    CONFIG_INVALID: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid Kimix configuration {path}: {reason}"
    ),
    CONFIG_UNAVAILABLE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Configuration file is unavailable: {path}"
    ),
    CONFIG_INVALID_SESSION_REFERENCE: lambda: QCoreApplication.translate(
        "LLMSettingsDialog", "Invalid session configuration reference: {path}"
    ),
}


def _format_tokens(value: int | None) -> str:
    if value is None:
        return QCoreApplication.translate("LLMSettingsDialog", "Not specified")
    return QCoreApplication.translate("LLMSettingsDialog", "{count} tokens").format(
        count=f"{value:,}"
    )
