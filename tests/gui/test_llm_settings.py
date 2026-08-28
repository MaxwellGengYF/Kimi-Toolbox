from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import orjson
from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    CodexAuthSnapshot,
    CodexModel,
    CodexModelCatalog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
)

from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.llm import (
    AXIS_CONTEXT_WINDOW,
    AXIS_THINKING_EFFORT,
    PROBLEM_CREDENTIAL_MISSING,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMModelDescriptor,
    LLMSelection,
    ParameterAssignment,
    ParameterOption,
    ParameterSpec,
    ProviderFileTarget,
    RuntimeOverrides,
    chatgpt_model_descriptor,
    chatgpt_selection,
    resolve_selection,
    resolved_provider_file,
)
from kimix_gui.qt.components import DisclosureHeader
from kimix_gui.qt.settings_dialog import LLMSettingsDialog, LLMSettingsResult

from .qtutil import find, widget_text


def write_provider_file(path: Path, *, api_key: str = "test-key") -> Path:
    path.write_bytes(
        orjson.dumps(
            {
                "model": "provider-model",
                "name": "Provider Model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": api_key,
            }
        )
    )
    return path


def chatgpt_model(
    *,
    connected: bool = True,
    efforts: tuple[str, ...] = ("low", "medium", "high", "ultra"),
    default: str | None = "medium",
):
    return chatgpt_model_descriptor(
        CodexModel(
            "gpt-test",
            display_name="GPT Test",
            reasoning_efforts=efforts,
            default_reasoning_effort=default,
        ),
        connected=connected,
        stale=False,
    )


def dialog_for_model(model, *, effort: str = "medium", **kwargs) -> LLMSettingsDialog:
    selection = chatgpt_selection(model.model_id, effort)
    return LLMSettingsDialog(
        current=resolve_selection(selection, [model]),
        models=(model,),
        scope_label="New sessions",
        **kwargs,
    )


def test_chatgpt_parameter_values_share_one_model_row(qtbot) -> None:
    model = chatgpt_model()
    dialog = dialog_for_model(model)
    qtbot.addWidget(dialog)

    assert len(dialog.model_items()) == 1
    assert dialog.model_items()[0].text() == "GPT Test"
    assert "medium" not in dialog.model_items()[0].text()


def test_parameter_picker_uses_stable_order_and_pins_explicit_choice(qtbot) -> None:
    model = chatgpt_model(efforts=("low", "ultra", "max", "off", "future"), default="ultra")
    dialog = dialog_for_model(model, effort="max")
    qtbot.addWidget(dialog)
    selected: list[LLMSettingsResult] = []
    dialog.applied.connect(selected.append)
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert picker is not None

    assert [picker.itemData(index) for index in range(picker.count())] == [
        "low",
        "max",
    ]
    assert "Model default" in picker.itemText(1)
    picker.setCurrentIndex(0)
    picker.activated.emit(0)
    find(dialog, "apply-settings", QPushButton).click()

    assert selected == [LLMSettingsResult(chatgpt_selection("gpt-test", "low"))]


def test_dialog_renders_each_future_parameter_axis_without_structural_changes(qtbot) -> None:
    thinking = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption("low", RuntimeOverrides(thinking_effort="low"), True),
            ParameterOption("high", RuntimeOverrides(thinking_effort="high")),
        ),
        order=10,
    )
    context = ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (
            ParameterOption("small", RuntimeOverrides(max_context_size=100_000)),
            ParameterOption("large", RuntimeOverrides(max_context_size=1_000_000), True),
        ),
        order=20,
    )
    model = LLMModelDescriptor(
        target=ChatGPTTarget("gpt-test"),
        model_id="gpt-test",
        provider_type="openai-codex",
        endpoint="ChatGPT subscription",
        credential="Connected account",
        file_format="Built-in",
        parameters=(thinking, context),
    )
    selection = LLMSelection(
        model.target,
        ParameterAssignment(
            {AXIS_THINKING_EFFORT: "low", AXIS_CONTEXT_WINDOW: "large"}
        ),
    )
    dialog = LLMSettingsDialog(
        current=resolve_selection(selection, [model]),
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    thinking_picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    context_picker = dialog.parameter_picker(AXIS_CONTEXT_WINDOW)

    assert thinking_picker is not None
    assert context_picker is not None
    assert thinking_picker.objectName() == f"param-{AXIS_THINKING_EFFORT}"
    assert context_picker.objectName() == f"param-{AXIS_CONTEXT_WINDOW}"
    assert thinking_picker.selected_value() == "low"
    assert context_picker.selected_value() == "large"
    context_picker.setCurrentIndex(context_picker.findData("small"))
    context_picker.activated.emit(context_picker.currentIndex())
    assert dialog.selected_selection() == LLMSelection(
        model.target,
        ParameterAssignment(
            {AXIS_THINKING_EFFORT: "low", AXIS_CONTEXT_WINDOW: "small"}
        ),
    )


def test_parameter_picker_is_keyboard_focusable_and_accessible(qtbot) -> None:
    dialog = dialog_for_model(chatgpt_model())
    qtbot.addWidget(dialog)
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert picker is not None
    picker.show()
    picker.setFocus()

    assert picker.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert picker.accessibleName() == "Thinking effort"
    assert "exact value" in picker.accessibleDescription()


def test_removed_saved_parameter_value_stays_visible_and_blocks_apply(qtbot) -> None:
    model = chatgpt_model(efforts=("low", "high"), default="low")
    dialog = dialog_for_model(model, effort="medium")
    qtbot.addWidget(dialog)
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert picker is not None
    picker_model = picker.model()

    assert picker.currentData() == "medium"
    assert picker.property("state") == "unavailable"
    assert isinstance(picker_model, QStandardItemModel)
    assert not picker_model.item(0).isEnabled()
    assert widget_text(dialog, "settings-error") == ""
    assert "no longer available" in widget_text(
        dialog,
        f"param-{AXIS_THINKING_EFFORT}-problem",
    )
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_catalog_refresh_keeps_a_removed_current_model_as_an_unavailable_row(qtbot) -> None:
    dialog = dialog_for_model(chatgpt_model())
    qtbot.addWidget(dialog)

    dialog.set_models(())

    assert len(dialog.model_items()) == 1
    assert dialog.model_items()[0].model_descriptor.model_id == "gpt-test"
    assert dialog.model_items()[0].text() == "gpt-test"
    assert widget_text(dialog, "selection-status") == "Model unavailable"
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_unknown_future_parameter_value_is_retained_as_a_raw_placeholder(qtbot) -> None:
    model = chatgpt_model(efforts=("future-effort",), default="future-effort")
    dialog = dialog_for_model(model, effort="future-effort")
    qtbot.addWidget(dialog)
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert isinstance(picker, QComboBox)

    assert picker.itemText(0) == "Unavailable value · future-effort"


def test_provider_thinking_summary_comes_from_resolved_parameters(qtbot) -> None:
    thinking = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption("low", RuntimeOverrides(thinking_effort="low"), True),
            ParameterOption("high", RuntimeOverrides(thinking_effort="high")),
        ),
    )
    model = LLMModelDescriptor(
        target=ProviderFileTarget(Path("thinking.json")),
        model_id="provider-thinking",
        provider_type="openai_legacy",
        endpoint="http://127.0.0.1:8000/v1",
        credential="Configured in file",
        file_format="JSON",
        parameters=(thinking,),
    )
    selection = LLMSelection(
        model.target,
        ParameterAssignment({AXIS_THINKING_EFFORT: "high"}),
    )
    dialog = LLMSettingsDialog(
        current=resolve_selection(selection, [model]),
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    assert widget_text(dialog, "provider-thinking") == "Selected by model parameters"


def test_provider_file_has_no_editable_runtime_parameters(
    qtbot,
    tmp_path: Path,
) -> None:
    resolved = resolved_provider_file(write_provider_file(tmp_path / "provider.json"))
    dialog = LLMSettingsDialog(
        current=resolved,
        models=(resolved.model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    assert resolved.model.parameters == ()
    assert dialog.parameter_picker(AXIS_THINKING_EFFORT) is None
    assert find(dialog, "model-parameters").isHidden()


def test_provider_add_controls_live_inside_provider_group(qtbot, tmp_path: Path) -> None:
    resolved = resolved_provider_file(write_provider_file(tmp_path / "provider.json"))
    dialog = LLMSettingsDialog(
        current=resolved,
        models=(resolved.model,),
        scope_label="New sessions",
        manage_library=True,
    )
    qtbot.addWidget(dialog)
    picker = find(dialog, "provider-file-picker")
    provider_header = find(dialog, "provider-config-group", DisclosureHeader)
    cards = find(dialog, "provider-cards", QScrollArea)

    assert cards.isAncestorOf(picker)
    provider_body = find(provider_header.parentWidget(), "provider-card-body")
    provider_header.setChecked(False)
    assert provider_body.isHidden()
    provider_header.setChecked(True)
    assert not provider_body.isHidden()


def test_adding_provider_file_registers_and_selects_it(qtbot, tmp_path: Path) -> None:
    first = resolved_provider_file(write_provider_file(tmp_path / "first.json"))
    second_path = write_provider_file(tmp_path / "second.json")
    registered = []
    dialog = LLMSettingsDialog(
        current=first,
        models=(first.model,),
        scope_label="New sessions",
        manage_library=True,
        registrar=registered.append,
    )
    qtbot.addWidget(dialog)
    path_input = find(dialog, "config-path", QLineEdit)
    path_input.setText(str(second_path))

    find(dialog, "load-config", QPushButton).click()

    assert registered == [resolved_provider_file(second_path).selection.target]
    assert dialog.selected_selection() == resolved_provider_file(second_path).selection


def test_missing_provider_credential_is_marked_unavailable_not_preflighted(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("KIMI_API_KEY", "KIMIX_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    resolved = resolved_provider_file(
        write_provider_file(tmp_path / "missing-key.json", api_key="")
    )
    dialog = LLMSettingsDialog(
        current=resolved,
        models=(resolved.model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    assert resolved.problem is not None
    assert resolved.problem.kind == PROBLEM_CREDENTIAL_MISSING
    assert "No API key" in widget_text(dialog, "settings-error")
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_session_inheritance_is_a_control_above_the_model_list(qtbot) -> None:
    project_model = chatgpt_model()
    project_selection = chatgpt_selection("gpt-test", "medium")
    project = resolve_selection(project_selection, [project_model])
    override_model = chatgpt_model(efforts=("low", "high"), default="high")
    dialog = LLMSettingsDialog(
        current=project,
        models=(override_model,),
        scope_label="Session test",
        project_default=project,
        inherits_project_default=True,
    )
    qtbot.addWidget(dialog)
    inherit = find(dialog, "inherit-project-default", QCheckBox)

    assert inherit.isChecked()
    assert inherit.parent() is not find(dialog, "provider-cards", QScrollArea)
    assert len(dialog.model_items()) == 1
    picker = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    assert picker is not None
    assert picker.selected_value() == "medium"
    assert not picker.isEnabled()
    dialog.model_items()[0].click()
    assert not inherit.isChecked()


def test_follow_project_default_emits_the_exact_project_selection(qtbot) -> None:
    model = chatgpt_model()
    project = resolve_selection(chatgpt_selection("gpt-test", "medium"), [model])
    session = resolve_selection(chatgpt_selection("gpt-test", "high"), [model])
    dialog = LLMSettingsDialog(
        current=session,
        models=(model,),
        scope_label="Session test",
        project_default=project,
    )
    qtbot.addWidget(dialog)
    applied: list[LLMSettingsResult] = []
    dialog.applied.connect(applied.append)

    assert dialog.select_project_default()
    find(dialog, "apply-settings", QPushButton).click()

    assert applied == [LLMSettingsResult(project.selection, use_project_default=True)]


def test_narrow_dialog_stacks_source_and_detail_panes(qtbot) -> None:
    dialog = dialog_for_model(chatgpt_model())
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.resize(600, 700)
    qtbot.wait(10)
    assert find(dialog, "settings-body", QSplitter).orientation() == Qt.Orientation.Vertical

    dialog.resize(1000, 700)
    qtbot.wait(10)
    assert find(dialog, "settings-body", QSplitter).orientation() == Qt.Orientation.Horizontal


def test_model_list_uses_only_the_full_name_and_the_view_may_elide_overflow(qtbot) -> None:
    long_name = "GPT " + "very-long-model-name-" * 12
    model = chatgpt_model_descriptor(
        CodexModel(
            "gpt-test",
            display_name=long_name,
            reasoning_efforts=("medium",),
            default_reasoning_effort="medium",
        ),
        connected=True,
        stale=False,
    )
    dialog = dialog_for_model(model)
    qtbot.addWidget(dialog)
    item = dialog.model_items()[0]

    assert item.text() == long_name
    assert "\n" not in item.text()
    assert item.toolTip() == long_name
    assert item.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_fresh_catalog_pins_legacy_project_default_once(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 4,
                "configs": [],
                "work_dirs": {
                    str(work_dir.resolve()): {"default": {"kind": "chatgpt", "model": "gpt-test"}}
                },
            }
        )
    )
    store = KimixGuiConfigStore(metadata)
    app = KimixGuiApp(SessionOptions(work_dir), config_store=store)
    assert not app.default_config.selection.pinned

    app.codex_controller.on_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    app.codex_controller.on_catalog_changed(
        CodexModelCatalog(
            1,
            (
                CodexModel(
                    "gpt-test",
                    reasoning_efforts=("low", "medium", "high"),
                    default_reasoning_effort="medium",
                ),
            ),
            False,
        )
    )

    assert app.default_config.selection == chatgpt_selection("gpt-test", "medium")
    assert store.default_selection_for(work_dir) == app.default_config.selection


def test_fresh_catalog_without_a_valid_default_does_not_guess_for_legacy_selection(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 4,
                "configs": [],
                "work_dirs": {
                    str(work_dir.resolve()): {"default": {"kind": "chatgpt", "model": "gpt-test"}}
                },
            }
        )
    )
    store = KimixGuiConfigStore(metadata)
    app = KimixGuiApp(SessionOptions(work_dir), config_store=store)

    app.codex_controller.on_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    app.codex_controller.on_catalog_changed(
        CodexModelCatalog(
            1,
            (
                CodexModel(
                    "gpt-test",
                    reasoning_efforts=("low", "high"),
                    default_reasoning_effort=None,
                ),
            ),
            False,
        )
    )

    assert not app.default_config.selection.pinned
    assert not app.default_config.available
    assert store.default_selection_for(work_dir) == app.default_config.selection


def test_fresh_catalog_pins_an_accessed_legacy_session_without_scanning_others(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    session_file = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    session_file.parent.mkdir(parents=True)
    session_file.write_bytes(
        orjson.dumps(
            {
                "version": 2,
                "llm": {"kind": "chatgpt", "model": "gpt-test"},
            }
        )
    )
    store = KimixGuiConfigStore(
        tmp_path / "metadata.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )
    app = KimixGuiApp(SessionOptions(work_dir), config_store=store)
    refreshes: list[dict[str, Any]] = []

    class OpenDialog:
        def set_models(self, _models: object, **kwargs: Any) -> None:
            refreshes.append(kwargs)

    dialog = OpenDialog()
    app._llm_dialogs[cast(Any, dialog)] = "session-1"

    app.codex_controller.on_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    assert store.session_selection_for(work_dir, "session-1").selection == LLMSelection(
        ChatGPTTarget("gpt-test"),
        pinned=False,
    )

    app.codex_controller.on_catalog_changed(
        CodexModelCatalog(
            1,
            (
                CodexModel(
                    "gpt-test",
                    reasoning_efforts=("low", "medium", "high"),
                    default_reasoning_effort="medium",
                ),
            ),
            False,
        )
    )

    pinned = chatgpt_selection("gpt-test", "medium")
    assert store.session_selection_for(work_dir, "session-1").selection == pinned
    assert refreshes[-1]["current"].selection == pinned


def test_catalog_refresh_does_not_mutate_running_session_snapshot(tmp_path: Path) -> None:
    selection = chatgpt_selection("gpt-test", "medium")
    app = KimixGuiApp(SessionOptions(tmp_path, llm_selection=selection))
    app.codex_controller.on_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    initial_catalog = CodexModelCatalog(
        1,
        (
            CodexModel(
                "gpt-test",
                reasoning_efforts=("low", "medium", "high"),
                default_reasoning_effort="medium",
            ),
        ),
        False,
    )
    app.codex_controller.on_catalog_changed(initial_catalog)
    running = app.default_config
    app._active_config = running

    app.codex_controller.on_catalog_changed(
        CodexModelCatalog(
            2,
            (
                CodexModel(
                    "gpt-test",
                    reasoning_efforts=("low", "high"),
                    default_reasoning_effort="low",
                ),
            ),
            False,
        )
    )

    assert app._active_config is running
    assert app._active_config.available
    assert not app.default_config.available


def test_session_override_and_project_default_store_the_same_selection_shape(
    tmp_path: Path,
) -> None:
    store = KimixGuiConfigStore(
        tmp_path / "metadata.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )
    project = chatgpt_selection("gpt-test", "medium")
    override = chatgpt_selection("gpt-test", "high")
    store.set_default(tmp_path, project)
    store.set_session(tmp_path, "session-1", override)

    global_value = orjson.loads((tmp_path / "metadata.json").read_bytes())["work_dirs"][
        str(tmp_path.resolve())
    ]["default_llm"]
    session_value = orjson.loads(
        (tmp_path / "sessions" / "session-1" / "kimix-gui.json").read_bytes()
    )["llm"]

    assert global_value.keys() == session_value.keys() == {
        "target",
        "parameters",
        "pinned",
    }
    assert global_value["parameters"][AXIS_THINKING_EFFORT] == "medium"
    assert session_value["parameters"][AXIS_THINKING_EFFORT] == "high"


def test_session_selection_type_is_immutable() -> None:
    selection = LLMSelection(
        ChatGPTTarget("gpt-test"),
        ParameterAssignment({AXIS_THINKING_EFFORT: "medium"}),
    )

    assert selection == chatgpt_selection("gpt-test", "medium")


def test_provider_file_target_renders_multiple_metadata_driven_parameter_axes(qtbot) -> None:
    target = ProviderFileTarget(Path("provider.json"))
    model = LLMModelDescriptor(
        target=target,
        model_id="provider-model",
        provider_type="test",
        endpoint="memory://test",
        credential="None",
        file_format="Test",
        parameters=(
            ParameterSpec(
                AXIS_CONTEXT_WINDOW,
                (
                    ParameterOption(
                        "200k",
                        RuntimeOverrides(max_context_size=200_000),
                        is_default=True,
                    ),
                    ParameterOption("1m", RuntimeOverrides(max_context_size=1_000_000)),
                ),
                order=20,
            ),
            ParameterSpec(
                AXIS_THINKING_EFFORT,
                (
                    ParameterOption("low", RuntimeOverrides(thinking_effort="low")),
                    ParameterOption(
                        "high",
                        RuntimeOverrides(thinking_effort="high"),
                        is_default=True,
                    ),
                ),
                order=10,
            ),
        ),
    )
    current = resolve_selection(LLMSelection(target), [model])
    dialog = LLMSettingsDialog(
        current=current,
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    thinking = dialog.parameter_picker(AXIS_THINKING_EFFORT)
    context = dialog.parameter_picker(AXIS_CONTEXT_WINDOW)

    assert thinking is not None
    assert context is not None
    assert thinking.currentData() == "high"
    assert context.currentData() == "200k"
    assert thinking.objectName() == f"param-{AXIS_THINKING_EFFORT}"
    assert context.objectName() == f"param-{AXIS_CONTEXT_WINDOW}"


def test_unknown_saved_axis_is_a_disabled_raw_token_row_in_the_dialog(qtbot) -> None:
    target = ProviderFileTarget(Path("provider.json"))
    parameter = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (ParameterOption("low", is_default=True),),
    )
    model = LLMModelDescriptor(
        target=target,
        model_id="provider-model",
        provider_type="test",
        endpoint="memory://test",
        credential="None",
        file_format="Test",
        parameters=(parameter,),
    )
    selection = LLMSelection(
        target,
        ParameterAssignment(
            {AXIS_THINKING_EFFORT: "low", "future_axis": "future_value"}
        ),
    )
    current = resolve_selection(selection, [model])
    dialog = LLMSettingsDialog(
        current=current,
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    picker = dialog.parameter_picker("future_axis")

    assert picker is not None
    assert picker.currentData() == "future_value"
    assert picker.accessibleName() == "future_axis"
    assert not picker.isEnabled()
    assert picker.property("state") == "unavailable"
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_parameter_axis_without_a_default_is_unresolved_until_selected(qtbot) -> None:
    target = ProviderFileTarget(Path("provider.json"))
    parameter = ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (ParameterOption("200k"), ParameterOption("1m")),
    )
    model = LLMModelDescriptor(
        target=target,
        model_id="provider-model",
        provider_type="test",
        endpoint="memory://test",
        credential="None",
        file_format="Test",
        parameters=(parameter,),
    )
    current = resolve_selection(LLMSelection(target), [model])
    dialog = LLMSettingsDialog(
        current=current,
        models=(model,),
        scope_label="New sessions",
    )
    qtbot.addWidget(dialog)

    picker = dialog.parameter_picker(AXIS_CONTEXT_WINDOW)

    assert picker is not None
    assert picker.currentIndex() == -1
    assert picker.property("state") == "unavailable"
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()
