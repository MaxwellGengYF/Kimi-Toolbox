"""Does choosing a theme actually repaint the app, all the way to the pixels?

Three layers have to move together, and each can fail on its own:

* the module global ``qt.theme._ACTIVE_THEME`` plus the ``COLORS`` map derived from
  it, which is what the hand-painted widgets read;
* the stylesheet, which is what every declaratively styled widget reads;
* the actual painted output, which is the only thing that proves the first two
  reached a widget rather than merely being stored.

The pixel checks work by *attribution*: no hex may appear in both palettes (a guard
in ``test_design_tokens.py`` keeps the ramps from overlapping, and ``LIGHT.text`` was
deliberately moved off ``DARK.bg`` for this reason), so a color found in a light
render can be blamed on the dark theme with no ambiguity.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QColor, QImage, QPalette, QRegion
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from kimix_gui.design import DARK, LIGHT, SUPPORTED_THEMES, THEMES, Theme
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt import theme as theme_module
from kimix_gui.qt.preferences_dialog import THEME_LABELS, PreferencesDialog
from kimix_gui.qt.session_row import SelectionMark, SessionRow
from kimix_gui.qt.theme import (
    apply_theme,
    apply_theme_preference,
    build_stylesheet,
    desktop_prefers_dark,
)
from kimix_gui.qt.todo_panel import TodoPanel
from kimix_gui.qt.transcript import Transcript
from kimix_gui.session_index import SessionSummary
from kimix_gui.todos import TodoEntry, TodoSnapshot

from .qtutil import find
from .transcript_helpers import append_text

SUMMARY = SessionSummary(
    id="s-1",
    title="A session with a title",
    updated_at=1_700_000_000.0,
    size_bytes=2048,
    file_count=3,
    todo_count=2,
)

SNAPSHOT = TodoSnapshot(
    entries=(
        TodoEntry(title="finished", status="done"),
        TodoEntry(title="running", status="in_progress"),
        TodoEntry(title="waiting", status="pending"),
    )
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _hexes(widget: QWidget, size: QSize) -> set[str]:
    """Render ``widget`` alone and return every distinct opaque color in it.

    ``DrawWindowBackground`` on its own renders the widget without descending into
    children, which keeps each assertion about the widget actually named.
    """

    widget.resize(size)
    image = QImage(size, QImage.Format.Format_ARGB32)
    image.fill(0)
    widget.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawWindowBackground)
    found: set[str] = set()
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() > 0:
                found.add(pixel.name())
    return found


def _palette_hexes(theme: Theme) -> set[str]:
    """Every hex this theme declares, as ``QColor.name()`` spells it."""

    values = set(theme.palette.__slots__)
    hexes = {getattr(theme.palette, name) for name in values}
    hexes |= set(theme.categories.as_map().values())
    return {value for value in hexes if value.startswith("#")}


def _exclusive_to(theme: Theme) -> set[str]:
    other = LIGHT if theme is DARK else DARK
    return _palette_hexes(theme) - _palette_hexes(other)


def _link_color(transcript: Transcript, index: object) -> str:
    """Read the color actually written into the cached document's anchor run."""

    document = transcript._delegate.document_for(QRect(0, 0, 600, 0), index)
    assert document is not None
    block = document.begin()
    while block.isValid():
        for span in block.textFormats():
            if span.format.isAnchor():
                return span.format.foreground().color().name()
        block = block.next()
    raise AssertionError("no anchor fragment in the rendered markdown")


@pytest.fixture
def light(app_appearance: QApplication) -> Iterator[QApplication]:
    """Run the body under the light theme, with ``app_appearance`` putting it back."""

    apply_theme(app_appearance, LIGHT)
    yield app_appearance


# --------------------------------------------------------------------------- #
# The two palettes must be tellable apart at all
# --------------------------------------------------------------------------- #


def test_no_hex_is_shared_between_the_two_palettes() -> None:
    """The premise every pixel check below rests on.

    If the themes shared a hex, finding it in a light render would prove nothing,
    and these tests would quietly weaken into tests of nothing.
    """

    assert _palette_hexes(DARK) & _palette_hexes(LIGHT) == set()
    assert _exclusive_to(DARK) == _palette_hexes(DARK)
    assert _exclusive_to(LIGHT) == _palette_hexes(LIGHT)


# --------------------------------------------------------------------------- #
# Layer 1: the module globals the hand-painted widgets read
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_applying_a_theme_moves_the_one_global_the_painters_read(
    app_appearance: QApplication, theme_name: str
) -> None:
    """``active_theme()`` is the whole handoff now.

    There used to be a second global beside it: a flat ``name -> hex`` map that
    ``set_active_theme`` mutated *in place*, because the painters had done
    ``from ... import COLORS`` and would otherwise have kept a stale dict forever.
    Every painter resolves through ``active_theme()`` at paint time, so the map and
    the aliasing it protected are both gone.
    """

    theme = THEMES[theme_name]

    apply_theme(app_appearance, theme)

    assert theme_module.active_theme() is theme
    assert not hasattr(theme_module, "COLORS"), "no second source of truth for colors"


# --------------------------------------------------------------------------- #
# Layer 2: the stylesheet on the application
# --------------------------------------------------------------------------- #


def test_switching_themes_swaps_the_installed_stylesheet(app_appearance: QApplication) -> None:
    apply_theme(app_appearance, DARK)
    dark_sheet = app_appearance.styleSheet()

    apply_theme(app_appearance, LIGHT)

    assert app_appearance.styleSheet() == build_stylesheet(LIGHT)
    assert app_appearance.styleSheet() != dark_sheet
    assert LIGHT.palette.bg in app_appearance.styleSheet()
    for value in _exclusive_to(DARK):
        assert value not in app_appearance.styleSheet(), value


def test_switching_themes_swaps_the_qpalette(app_appearance: QApplication) -> None:
    """The ``QPalette`` matters on its own: it is what unstyled Qt internals read."""

    apply_theme(app_appearance, LIGHT)

    palette = app_appearance.palette()
    assert palette.color(QPalette.ColorRole.Window).name() == LIGHT.palette.bg
    assert palette.color(QPalette.ColorRole.WindowText).name() == LIGHT.palette.text
    assert palette.color(QPalette.ColorRole.Base).name() == LIGHT.palette.surface


def _assert_combo_surfaces_follow(combo: QComboBox, expected: Theme) -> None:
    """Check the effective brushes Qt gives both halves of a combo box.

    The popup is a lazily materialized top-level window.  It is therefore absent
    from a static widget gallery, and a transparent brush there falls through to
    the platform backing surface instead of to the dialog behind it.
    """

    combo.showPopup()
    view = combo.view()
    surfaces = {
        "combo": (combo, QPalette.ColorRole.Window),
        "popup view": (view, QPalette.ColorRole.Base),
        "popup viewport": (view.viewport(), QPalette.ColorRole.Base),
        "popup window": (view.window(), QPalette.ColorRole.Window),
    }
    wrong: list[str] = []
    for label, (widget, role) in surfaces.items():
        color = widget.palette().color(role)
        if color.alpha() != 255 or color.name() != expected.palette.panel:
            wrong.append(
                f"{label}: {color.name(QColor.NameFormat.HexArgb)} != {expected.palette.panel}"
            )
    combo.hidePopup()
    assert wrong == []


def test_a_combo_box_and_its_popup_are_opaque_light_theme_surfaces(
    qtbot, app_appearance: QApplication
) -> None:
    apply_theme(app_appearance, LIGHT)
    dialog = PreferencesDialog(InterfacePreferences(theme="light"), font_families=list)
    qtbot.addWidget(dialog)
    dialog.show()
    app_appearance.processEvents()

    picker = find(dialog, "interface-theme", QComboBox)

    _assert_combo_surfaces_follow(picker, LIGHT)


def test_an_existing_combo_popup_follows_a_runtime_theme_switch(
    qtbot, app_appearance: QApplication
) -> None:
    apply_theme(app_appearance, DARK)
    dialog = PreferencesDialog(InterfacePreferences(theme="dark"), font_families=list)
    qtbot.addWidget(dialog)
    dialog.show()
    app_appearance.processEvents()
    picker = find(dialog, "interface-theme", QComboBox)

    # Materialize Qt's private popup under the first theme before switching.  A
    # newly constructed popup alone would not catch a stale internal window.
    _assert_combo_surfaces_follow(picker, DARK)

    apply_theme(app_appearance, LIGHT)
    app_appearance.processEvents()

    _assert_combo_surfaces_follow(picker, LIGHT)


# --------------------------------------------------------------------------- #
# Layer 3: pixels out of the widgets that paint themselves
# --------------------------------------------------------------------------- #


def test_a_session_row_repaints_under_the_light_theme(qtbot, app_appearance: QApplication) -> None:
    """``SessionRow.paintEvent`` reads ``active_theme()`` with no QSS involved."""

    row = SessionRow(SUMMARY)
    qtbot.addWidget(row)
    row.set_active(True)  # the branch that paints an accent-tinted fill
    size = QSize(400, 64)

    apply_theme(app_appearance, DARK)
    dark_pixels = _hexes(row, size)
    apply_theme(app_appearance, LIGHT)
    light_pixels = _hexes(row, size)

    assert dark_pixels != light_pixels
    assert light_pixels & _exclusive_to(DARK) == set()
    assert light_pixels & _palette_hexes(LIGHT) != set(), "it did paint tokens, not nothing"


def test_a_selection_mark_repaints_under_the_light_theme(
    qtbot, app_appearance: QApplication
) -> None:
    mark = SelectionMark()
    qtbot.addWidget(mark)
    mark.setChecked(True)
    size = QSize(22, 22)

    apply_theme(app_appearance, DARK)
    dark_pixels = _hexes(mark, size)
    apply_theme(app_appearance, LIGHT)
    light_pixels = _hexes(mark, size)

    assert dark_pixels != light_pixels
    assert light_pixels & _exclusive_to(DARK) == set()
    assert light_pixels & _palette_hexes(LIGHT) != set(), "it did paint tokens, not nothing"


def test_the_todo_progress_strip_repaints_under_the_light_theme(
    qtbot, app_appearance: QApplication
) -> None:
    """The strip's three segments come from ``categories``, the one group most at
    risk of being copied between themes without being re-picked."""

    panel = TodoPanel()
    qtbot.addWidget(panel)
    panel.set_snapshot(SNAPSHOT)
    panel.set_expanded(True)
    strip = panel.findChild(QWidget, "todo-progress")
    assert strip is not None, "the strip is addressed by object name here"
    size = QSize(200, 6)

    apply_theme(app_appearance, DARK)
    dark_pixels = _hexes(strip, size)
    apply_theme(app_appearance, LIGHT)
    light_pixels = _hexes(strip, size)

    assert dark_pixels != light_pixels
    assert light_pixels & _exclusive_to(DARK) == set()
    assert light_pixels & _palette_hexes(LIGHT) != set(), "it did paint tokens, not nothing"


def test_the_transcript_link_color_follows_a_real_theme_swap(
    qtbot, app_appearance: QApplication
) -> None:
    """Markdown link color is baked into the cached ``QTextDocument`` by Qt's reader.

    Repainting cannot fix it; only dropping the cache can, which is what
    ``Transcript.changeEvent`` does. ``test_appearance.py`` drives this with a
    synthetic probe theme; here it is a real swap, which additionally proves the
    light theme picked its own cyan instead of inheriting the dark one.
    """

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(
        transcript,
        "assistant",
        "see [the docs](https://example.invalid) for more",
        markdown=True,
    )
    index = transcript.model().index(0, 0)

    apply_theme(app_appearance, DARK)
    assert _link_color(transcript, index) == DARK.categories.cyan

    apply_theme(app_appearance, LIGHT)

    assert _link_color(transcript, index) == LIGHT.categories.cyan
    assert LIGHT.categories.cyan != DARK.categories.cyan


# --------------------------------------------------------------------------- #
# Resolving the preference, end to end through Qt
# --------------------------------------------------------------------------- #


def test_apply_theme_preference_honours_an_explicit_choice(
    app_appearance: QApplication,
) -> None:
    assert apply_theme_preference(app_appearance, InterfacePreferences(theme="light")) is LIGHT
    assert theme_module.active_theme() is LIGHT

    assert apply_theme_preference(app_appearance, InterfacePreferences(theme="dark")) is DARK
    assert theme_module.active_theme() is DARK


@pytest.mark.parametrize(("hint", "expected"), [(True, DARK), (False, LIGHT), (None, DARK)])
def test_apply_theme_preference_follows_the_desktop_under_auto(
    app_appearance: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    hint: bool | None,
    expected: Theme,
) -> None:
    """The hint has to be faked: the offscreen platform never reports one.

    Only ``auto`` consults it -- that half is pinned at the pure layer by
    ``test_design_tokens.py::test_resolve_theme``, which shows an explicit choice
    resolving the same way for every hint value.
    """

    monkeypatch.setattr(theme_module, "desktop_prefers_dark", lambda _app: hint)

    applied = apply_theme_preference(app_appearance, InterfacePreferences(theme="auto"))

    assert applied is expected
    assert theme_module.active_theme() is expected
    assert app_appearance.styleSheet() == build_stylesheet(expected)


def test_the_offscreen_platform_reports_no_preference(app_appearance: QApplication) -> None:
    """Records why the ``auto`` path cannot be tested against the real hint.

    ``QStyleHints.setColorScheme`` exists but the offscreen plugin ignores it, so
    ``colorScheme()`` stays ``Unknown`` here no matter what. Following the desktop
    live (Qt emits ``colorSchemeChanged``) is therefore also unverified.
    """

    assert desktop_prefers_dark(app_appearance) is None


# --------------------------------------------------------------------------- #
# The dialog that makes it selectable
# --------------------------------------------------------------------------- #


def test_the_preferences_dialog_offers_every_theme(qtbot) -> None:
    dialog = PreferencesDialog(InterfacePreferences(), font_families=list)
    qtbot.addWidget(dialog)

    picker = find(dialog, "interface-theme", QComboBox)

    assert [picker.itemData(row) for row in range(picker.count())] == ["auto", "dark", "light"]
    assert all(picker.itemText(row) for row in range(picker.count())), "every option is labelled"


def test_the_preferences_dialog_shows_the_stored_theme(qtbot) -> None:
    dialog = PreferencesDialog(InterfacePreferences(theme="light"), font_families=list)
    qtbot.addWidget(dialog)

    assert find(dialog, "interface-theme", QComboBox).currentData() == "light"


def test_the_preferences_dialog_emits_the_selected_theme(qtbot) -> None:
    dialog = PreferencesDialog(InterfacePreferences(), font_families=list)
    qtbot.addWidget(dialog)
    picker = find(dialog, "interface-theme", QComboBox)
    picker.setCurrentIndex(picker.findData("light"))
    emitted: list[object] = []
    dialog.applied.connect(emitted.append)

    dialog.findChild(QWidget, "apply-preferences").click()

    assert len(emitted) == 1
    assert isinstance(emitted[0], InterfacePreferences)
    assert emitted[0].theme == "light"


def test_an_unselected_theme_stays_unselected(qtbot) -> None:
    """The picker must not quietly rewrite ``auto`` into whatever it resolves to."""

    dialog = PreferencesDialog(InterfacePreferences(theme="auto"), font_families=list)
    qtbot.addWidget(dialog)
    emitted: list[object] = []
    dialog.applied.connect(emitted.append)

    dialog.findChild(QWidget, "apply-preferences").click()

    assert emitted[0].theme == "auto"


def test_the_dialog_and_the_design_layer_agree_on_the_option_list() -> None:
    """The picker builds itself from ``SUPPORTED_THEMES`` and must stay able to.

    ``THEME_LABELS[name]`` raises on a missing key rather than skipping the option,
    so a third theme added to the design layer fails loudly here instead of shipping
    a picker that silently cannot select it.
    """

    assert set(THEME_LABELS) == set(SUPPORTED_THEMES)
