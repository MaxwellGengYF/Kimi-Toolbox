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
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QListWidget, QPushButton, QSplitter

from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.llm import (
    LEGACY_DEFAULT_VARIANT,
    PROBLEM_CREDENTIAL_MISSING,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMSelection,
    chatgpt_model_descriptor,
    chatgpt_selection,
    configured_selection,
    reasoning_effort_variant,
    resolve_selection,
    resolved_provider_file,
)
from kimix_gui.qt.components import DisclosureHeader, VariantPicker
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
        chatgpt_connected=model.problem is None,
        **kwargs,
    )


def test_chatgpt_variants_share_one_model_row(qtbot) -> None:
    model = chatgpt_model()
    dialog = dialog_for_model(model)
    qtbot.addWidget(dialog)

    assert len(dialog.model_items()) == 1
    assert dialog.model_items()[0].text().splitlines()[0] == "GPT Test"
    assert "medium" not in dialog.model_items()[0].text().splitlines()[0]


def test_variant_picker_preserves_catalog_order_and_pins_explicit_choice(qtbot) -> None:
    model = chatgpt_model(efforts=("low", "ultra", "max", "off", "future"), default="ultra")
    dialog = dialog_for_model(model, effort="max")
    qtbot.addWidget(dialog)
    selected: list[LLMSettingsResult] = []
    dialog.applied.connect(selected.append)
    picker = find(dialog, "variant-picker", VariantPicker)

    assert [picker.itemData(index) for index in range(picker.count())] == [
        "reasoning_effort/low",
        "reasoning_effort/max",
        "reasoning_effort/future",
    ]
    assert "Model default" in picker.itemText(1)
    picker.setCurrentIndex(0)
    picker.activated.emit(0)
    find(dialog, "apply-settings", QPushButton).click()

    assert selected == [LLMSettingsResult(chatgpt_selection("gpt-test", "low"))]


def test_variant_picker_is_keyboard_focusable_and_accessible(qtbot) -> None:
    dialog = dialog_for_model(chatgpt_model())
    qtbot.addWidget(dialog)
    picker = find(dialog, "variant-picker", VariantPicker)
    picker.show()
    picker.setFocus()

    assert picker.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert picker.accessibleName() == "Model variant"
    assert "exact runtime variant" in picker.accessibleDescription()


def test_removed_saved_variant_stays_visible_and_blocks_apply(qtbot) -> None:
    model = chatgpt_model(efforts=("low", "high"), default="low")
    dialog = dialog_for_model(model, effort="medium")
    qtbot.addWidget(dialog)
    picker = find(dialog, "variant-picker", VariantPicker)
    picker_model = picker.model()

    assert picker.currentData() == "reasoning_effort/medium"
    assert picker.property("state") == "unavailable"
    assert isinstance(picker_model, QStandardItemModel)
    assert not picker_model.item(0).isEnabled()
    assert "no longer available" in widget_text(dialog, "settings-error")
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_catalog_refresh_keeps_a_removed_current_model_as_an_unavailable_row(qtbot) -> None:
    dialog = dialog_for_model(chatgpt_model())
    qtbot.addWidget(dialog)

    dialog.set_models((), chatgpt_connected=True)

    assert len(dialog.model_items()) == 1
    assert dialog.model_items()[0].model_descriptor.model_id == "gpt-test"
    assert "Model unavailable" in dialog.model_items()[0].text()
    assert not find(dialog, "apply-settings", QPushButton).isEnabled()


def test_unknown_future_variant_uses_its_raw_catalog_label(qtbot) -> None:
    model = chatgpt_model(efforts=("future-effort",), default="future-effort")
    dialog = dialog_for_model(model, effort="future-effort")
    qtbot.addWidget(dialog)
    picker = find(dialog, "variant-picker", QComboBox)

    assert picker.itemText(0) == "future-effort · Model default"


def test_provider_file_has_one_configured_variant_without_editable_picker(
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

    assert len(resolved.model.variants) == 1
    assert resolved.model.variants[0].key == configured_selection(resolved.selection.target).variant
    assert find(dialog, "variant-picker", VariantPicker).isHidden()


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
    model_list = find(dialog, "config-list", QListWidget)

    assert model_list.isAncestorOf(picker)
    provider_header.setChecked(False)
    assert picker.isHidden()
    provider_header.setChecked(True)
    assert not picker.isHidden()


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
        chatgpt_connected=True,
    )
    qtbot.addWidget(dialog)
    inherit = find(dialog, "inherit-project-default", QCheckBox)

    assert inherit.isChecked()
    assert inherit.parent() is not find(dialog, "config-list", QListWidget)
    assert len(dialog.model_items()) == 1
    picker = find(dialog, "variant-picker", VariantPicker)
    assert picker.selected_key() == reasoning_effort_variant("medium").id
    assert not picker.isEnabled()
    dialog._list.setCurrentItem(dialog.model_items()[0])
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
        chatgpt_connected=True,
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


def test_model_text_is_full_and_only_the_view_may_elide_overflow(qtbot) -> None:
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

    assert item.text().startswith(long_name)
    assert not item.text().splitlines()[0].endswith("...")
    assert item.toolTip() == item.text()
    assert find(dialog, "config-list", QListWidget).textElideMode() == Qt.TextElideMode.ElideRight


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
    assert app.default_config.selection.variant == LEGACY_DEFAULT_VARIANT

    app.on_codex_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    app.on_codex_catalog_changed(
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

    assert app.default_config.selection.variant == reasoning_effort_variant("medium")
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

    app.on_codex_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    app.on_codex_catalog_changed(
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

    assert app.default_config.selection.variant == LEGACY_DEFAULT_VARIANT
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

    app.on_codex_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
    assert store.session_selection_for(work_dir, "session-1").selection == LLMSelection(
        ChatGPTTarget("gpt-test"),
        LEGACY_DEFAULT_VARIANT,
    )

    app.on_codex_catalog_changed(
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
    app.on_codex_auth_changed(CodexAuthSnapshot(1, AUTH_CONNECTED))
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
    app.on_codex_catalog_changed(initial_catalog)
    running = app.default_config
    app._active_config = running

    app.on_codex_catalog_changed(
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

    assert global_value.keys() == session_value.keys() == {"target", "variant"}
    assert global_value["variant"]["value"] == "medium"
    assert session_value["variant"]["value"] == "high"


def test_session_selection_type_is_immutable() -> None:
    selection = LLMSelection(ChatGPTTarget("gpt-test"), reasoning_effort_variant("medium"))

    assert selection == chatgpt_selection("gpt-test", "medium")
