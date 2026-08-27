from __future__ import annotations

from pathlib import Path

import orjson
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QComboBox, QPushButton, QSpinBox

from kimix_gui.llm import KimixGuiConfigStore, resolved_provider_file
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.components import SettingsList
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog
from kimix_gui.qt.theme import interface_font

from .qtutil import find


def test_gui_config_store_round_trips_font_priorities(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    store = KimixGuiConfigStore(store_file)

    store.set_interface(InterfacePreferences(("Cascadia Mono", "Consolas", "Cascadia Mono"), 17))

    assert KimixGuiConfigStore(store_file).interface == InterfacePreferences(
        ("Cascadia Mono", "Consolas"), 17
    )
    assert orjson.loads(store_file.read_bytes()) == {
        "version": 5,
        "interface": {
            "font_families": ["Cascadia Mono", "Consolas"],
            "font_size": 17,
            "language": "auto",
            "theme": "auto",
        },
        "provider_files": [],
        "work_dirs": {},
    }
    assert not store_file.with_suffix(".json.tmp").exists()


def test_preferences_written_before_language_existed_still_load(tmp_path) -> None:
    # A version 3 GUI config from before the ``language`` key migrates in memory while
    # the missing key still falls back to the documented default.
    store_file = tmp_path / "kimix-gui.json"
    store_file.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "interface": {"font_families": ["Consolas"], "font_size": 15},
                "configs": [],
                "work_dirs": {},
            }
        )
    )

    assert KimixGuiConfigStore.VERSION == 5
    assert KimixGuiConfigStore(store_file).interface == InterfacePreferences(
        ("Consolas",), 15, "auto", "auto"
    )


def test_preferences_written_before_the_theme_key_existed_still_load(tmp_path) -> None:
    # Same contract, one key later: a file that already knows ``language`` but not
    # ``theme``. Kept as its own case because the interesting part is that adding the
    # second optional key did not require the bump the first one didn't either.
    store_file = tmp_path / "kimix-gui.json"
    store_file.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "interface": {
                    "font_families": ["Consolas"],
                    "font_size": 15,
                    "language": "zh_CN",
                },
                "configs": [],
                "work_dirs": {},
            }
        )
    )

    assert KimixGuiConfigStore.VERSION == 5
    assert KimixGuiConfigStore(store_file).interface == InterfacePreferences(
        ("Consolas",), 15, "zh_CN", "auto"
    )


def test_an_unknown_stored_theme_is_normalized_on_the_way_in_and_out(tmp_path) -> None:
    """A hand-edited or future-written theme name must not reach the Qt layer."""

    store_file = tmp_path / "kimix-gui.json"
    store_file.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "interface": {"font_families": [], "theme": "solarized"},
                "configs": [],
                "work_dirs": {},
            }
        )
    )
    assert KimixGuiConfigStore(store_file).interface.theme == "auto"

    store = KimixGuiConfigStore(store_file)
    store.set_interface(InterfacePreferences(theme="solarized"))
    assert orjson.loads(store_file.read_bytes())["interface"]["theme"] == "auto"


def test_the_theme_preference_round_trips(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    store = KimixGuiConfigStore(store_file)

    store.set_interface(InterfacePreferences(theme="light"))

    assert KimixGuiConfigStore(store_file).interface.theme == "light"
    assert orjson.loads(store_file.read_bytes())["interface"]["theme"] == "light"


def test_unknown_stored_language_falls_back_to_auto(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    store_file.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "interface": {"font_families": [], "language": "kl_GL"},
                "configs": [],
                "work_dirs": {},
            }
        )
    )

    assert KimixGuiConfigStore(store_file).interface.language == "auto"


def test_gui_config_store_round_trips_language(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    KimixGuiConfigStore(store_file).set_interface(InterfacePreferences(language="zh_CN"))

    assert KimixGuiConfigStore(store_file).interface.language == "zh_CN"


def test_existing_gui_config_without_interface_preserves_llm_metadata(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    config_path = str((tmp_path / "provider.json").resolve())
    work_dir = str((tmp_path / "project").resolve())
    store_file.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "configs": [config_path],
                "work_dirs": {work_dir: {"default": config_path}},
            }
        )
    )

    store = KimixGuiConfigStore(store_file)
    assert store.interface == InterfacePreferences()

    store.set_interface(InterfacePreferences(language="zh_CN"))

    saved = orjson.loads(store_file.read_bytes())
    assert saved["provider_files"] == [config_path]
    assert saved["work_dirs"] == {
        work_dir: {
            "default_llm": {
                "target": {"kind": "provider_file", "path": config_path},
                "variant": {"kind": "configured"},
            }
        }
    }
    assert saved["interface"]["language"] == "zh_CN"


def test_invalid_preferences_file_uses_system_font(tmp_path) -> None:
    store_file = tmp_path / "kimix-gui.json"
    store_file.write_text("not json", encoding="utf-8")

    assert KimixGuiConfigStore(store_file).interface == InterfacePreferences()


def test_interface_font_keeps_system_fixed_width_fallback(qtbot) -> None:
    font = interface_font(InterfacePreferences(("Missing font",)))
    system_fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()

    assert font.families() == [system_fixed]


def test_interface_font_uses_saved_font_size(qtbot) -> None:
    font = interface_font(InterfacePreferences(("Missing font",), 18))

    assert font.pixelSize() == 18


def test_settings_sidebars_share_font_aware_item_height(qtbot, tmp_path: Path) -> None:
    preferences = PreferencesDialog(InterfacePreferences(), font_families=list)
    llm = LLMSettingsDialog(
        current=resolved_provider_file(tmp_path / "missing.json"),
        models=(),
        scope_label="New sessions",
    )
    qtbot.addWidget(preferences)
    qtbot.addWidget(llm)

    category = find(preferences, "preferences-categories", SettingsList).item(0)
    config = llm.model_items()[0]
    assert category.sizeHint().height() == config.sizeHint().height()
    assert category.sizeHint().height() >= 44


def test_preferences_dialog_saves_the_selected_language(qtbot) -> None:
    dialog = PreferencesDialog(InterfacePreferences(), font_families=list)
    qtbot.addWidget(dialog)
    selected: list[InterfacePreferences] = []
    dialog.applied.connect(selected.append)

    picker = find(dialog, "interface-language", QComboBox)
    assert [picker.itemData(index) for index in range(picker.count())] == [
        "auto",
        "en",
        "zh_CN",
    ]
    # Endonyms, deliberately not translated, so each option reads in its own language.
    assert [picker.itemText(index) for index in range(picker.count())] == [
        "\u8ddf\u968f\u7cfb\u7edf \u00b7 System",
        "English",
        "\u4e2d\u6587",
    ]
    assert picker.currentData() == "auto"

    picker.setCurrentIndex(picker.findData("zh_CN"))
    find(dialog, "apply-preferences", QPushButton).click()

    assert selected == [InterfacePreferences((), 13, "zh_CN")]


def test_preferences_dialog_preselects_the_stored_language(qtbot) -> None:
    dialog = PreferencesDialog(InterfacePreferences(language="en"), font_families=list)
    qtbot.addWidget(dialog)

    assert find(dialog, "interface-language", QComboBox).currentData() == "en"


def test_preferences_dialog_saves_primary_and_fallback_fonts(qtbot) -> None:
    dialog = PreferencesDialog(
        InterfacePreferences(),
        font_families=lambda: ["Cascadia Mono", "Consolas"],
    )
    qtbot.addWidget(dialog)
    selected: list[InterfacePreferences] = []
    dialog.applied.connect(selected.append)

    picker = find(dialog, "font-primary", QComboBox)
    picker.setCurrentIndex(picker.findData("Consolas"))
    fallback = find(dialog, "font-fallback", QComboBox)
    fallback.setCurrentIndex(fallback.findData("Cascadia Mono"))
    find(dialog, "font-size", QSpinBox).setValue(16)
    find(dialog, "apply-preferences", QPushButton).click()

    assert selected == [InterfacePreferences(("Consolas", "Cascadia Mono"), 16)]
    assert dialog.result() == int(dialog.DialogCode.Accepted)
