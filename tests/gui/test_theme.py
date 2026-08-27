"""Characterization tests for the color chain and for theme/font application."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields, replace

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from kimix_gui.design import CATEGORY_NAMES, DARK, LIGHT, THEMES, Palette, Theme
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt import theme
from kimix_gui.qt import transcript as transcript_module
from kimix_gui.qt.paint import qcolor
from kimix_gui.qt.theme import build_stylesheet
from kimix_gui.qt.todo_panel import status_color
from kimix_gui.qt.transcript import Transcript

from .transcript_helpers import append_text

#: The generated global stylesheet, replacing the former ``APP_STYLE`` constant.
APP_STYLE = build_stylesheet(DARK)

#: Color literals in the stylesheet that resolve through no design token at all.
#: This used to hold five entries (ink on accent, accent hover, danger surface,
#: danger border, translucent panel surface); each is now a named token, so the
#: guard below only has to stop new bare literals from appearing.
KNOWN_BARE_COLORS: frozenset[str] = frozenset()

#: The ink used on top of an accent fill. Only the ``on_accent`` token holds the
#: literal now: both the stylesheet and the transcript delegate read it from there.
ON_ACCENT_INK = "#042f2e"

#: The one name both palettes publish. Every other ``Palette`` role is unreachable
#: through ``qcolor()``, which resolves through the category palette alone.
SHARED_WITH_THE_CATEGORIES = "muted"


#: ``app_appearance`` lives in ``tests/gui/conftest.py``: ``tests/gui/test_appearance.py``
#: needs the same save/restore, and two copies of a fixture that unwinds
#: process-wide state is how one of them ends up not unwinding it.


# --------------------------------------------------------------------------- #
# Group 1: the color resolution chain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(CATEGORY_NAMES))
def test_qcolor_resolves_each_layout_name_to_the_palette_value(name: str) -> None:
    assert qcolor(name).name() == DARK.categories.resolve(name)


def test_qcolor_downgrades_unknown_names_to_muted() -> None:
    """The fallback is silent, so a typo shows up as gray rather than as an error."""

    assert qcolor("chartreuse").name() == DARK.categories.muted
    assert qcolor("").name() == DARK.categories.muted
    assert qcolor("bg").name() == DARK.categories.muted, "role names are not category names"


def test_qcolor_covers_the_category_names_and_nothing_else() -> None:
    """``qcolor()`` resolves through the category palette, so its domain is fixed."""

    assert set(CATEGORY_NAMES) == {"cyan", "green", "red", "yellow", "blue", "magenta", "muted"}
    assert {name: qcolor(name).name() for name in CATEGORY_NAMES} == DARK.categories.as_map()
    roles = {field.name for field in fields(Palette)}
    assert roles & set(CATEGORY_NAMES) == {SHARED_WITH_THE_CATEGORIES}
    for role in roles - {SHARED_WITH_THE_CATEGORIES}:
        assert qcolor(role).name() == DARK.categories.muted, f"{role} is a role, not a hue"


def test_todo_status_colors_resolve_through_the_semantic_palette() -> None:
    """``qt/todo_panel.py`` reads state colors from ``Palette``, not from hues."""

    assert status_color("done").name() == DARK.palette.success
    assert status_color("in_progress").name() == DARK.palette.accent
    assert status_color("pending").name() == DARK.palette.muted
    assert status_color("archived").name() == DARK.palette.muted, "unknown states stay muted"


def _token_color_values(theme_tokens: Theme) -> set[str]:
    """Every color string the design layer publishes for a theme, lowercased."""

    values: set[str] = set()
    for group in (theme_tokens.palette, theme_tokens.categories):
        for field in fields(group):
            value = getattr(group, field.name)
            if isinstance(value, str):
                values.add(value.lower())
    return values


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_stylesheet_bare_colors_stay_on_the_known_debt_list(theme_name: str) -> None:
    """Every color in the generated QSS must come from that theme's own tokens.

    Run against both themes, because a literal copied from the dark palette would
    still look like "a token" to a single-theme version of this check -- it would
    just be the wrong theme's token, and would stay dark on a light page.
    """

    theme_tokens = THEMES[theme_name]
    stylesheet = build_stylesheet(theme_tokens)
    tokens = _token_color_values(theme_tokens)
    literals = set(re.findall(r"#[0-9a-fA-F]{3,8}", stylesheet))
    literals |= set(re.findall(r"rgba?\([^)]*\)", stylesheet))
    bare = {literal for literal in literals if literal.lower() not in tokens}
    assert bare == set(KNOWN_BARE_COLORS)


def test_the_two_themes_produce_different_stylesheets() -> None:
    """Guards the one thing the scan above cannot see: that the QSS actually moved.

    A ``build_stylesheet`` that ignored its argument would pass every bare-color
    check for both themes, since the dark literals are all dark tokens.
    """

    dark = build_stylesheet(DARK)
    light = build_stylesheet(LIGHT)
    assert dark != light
    assert DARK.palette.bg not in light
    assert LIGHT.palette.bg not in dark
    assert len(dark.splitlines()) == len(light.splitlines()), "same rules, different values"


def test_on_accent_ink_is_one_value_shared_by_the_stylesheet_and_the_delegate() -> None:
    """The on-accent ink now lives only in the token; both uses read it from there.

    ``qt/transcript.py`` used to paint selected body text with a bare ``#042f2e``.
    It reads ``palette.on_accent`` now, so rebinding that token has to move the
    delegate *and* both stylesheet uses, which is what proves the wiring.
    """

    source = inspect.getsource(transcript_module)
    assert ON_ACCENT_INK not in source, "the delegate keeps no copy of the ink"
    assert "fmt.setForeground(QColor(palette.on_accent))" in source

    assert DARK.palette.on_accent == ON_ACCENT_INK, "the token owns the value"
    accent_pair = f"background: {DARK.palette.accent};\n    color: {ON_ACCENT_INK};"
    assert APP_STYLE.count(ON_ACCENT_INK) == 2
    assert APP_STYLE.count(accent_pair) == 2, "both stylesheet uses sit on an accent background"

    probe = replace(DARK, palette=replace(DARK.palette, on_accent="#123456"))
    repainted = build_stylesheet(probe)
    assert ON_ACCENT_INK not in repainted, "nothing hardcodes the ink anymore"
    assert repainted.count("#123456") == 2, "both uses follow the token"

    assert ON_ACCENT_INK not in _token_color_values(DARK) - {DARK.palette.on_accent}, (
        "no second token carries the same literal"
    )


# --------------------------------------------------------------------------- #
# Group 4: theme / font application, plus two known defects
# --------------------------------------------------------------------------- #


def test_apply_theme_is_reentrant(app_appearance: QApplication) -> None:
    theme.apply_theme(app_appearance)
    first_palette = QPalette(app_appearance.palette())
    first_font = QFont(app_appearance.font())
    first_stylesheet = app_appearance.styleSheet()

    theme.apply_theme(app_appearance)

    assert app_appearance.styleSheet() == first_stylesheet == build_stylesheet(DARK)
    assert app_appearance.palette() == first_palette
    assert app_appearance.font() == first_font


def test_apply_theme_maps_the_palette_roles_to_tokens(app_appearance: QApplication) -> None:
    theme.apply_theme(app_appearance)
    palette = app_appearance.palette()
    role = QPalette.ColorRole
    tokens = DARK.palette
    expected = {
        role.Window: tokens.bg,
        role.WindowText: tokens.text,
        role.Base: tokens.surface,
        role.AlternateBase: tokens.panel,
        role.Text: tokens.text,
        role.Button: tokens.panel,
        role.ButtonText: tokens.text,
        role.Highlight: tokens.boost,
        role.HighlightedText: tokens.text,
        role.PlaceholderText: tokens.muted,
    }
    assert {key: palette.color(key).name() for key in expected} == expected


def test_apply_theme_resets_the_selected_interface_font(app_appearance: QApplication) -> None:
    """Pins today's order-sensitive behavior, which Phase 4 is expected to fix.

    ``apply_theme`` unconditionally installs ``Segoe UI`` at 13px, so it wipes any
    font the user picked. ``app.py`` only survives this because it calls
    ``apply_interface_font`` on the very next line (``app.py:127-131``). Once the
    reset moves out of ``apply_theme``, this test should be inverted.
    """

    theme.apply_interface_font(app_appearance, InterfacePreferences(font_families=(), font_size=24))
    assert app_appearance.font().pixelSize() == 24

    theme.apply_theme(app_appearance)

    assert app_appearance.font().family() == "Segoe UI"
    assert app_appearance.font().pixelSize() == 13


def _one_row_transcript(qtbot) -> tuple[Transcript, QStyleOptionViewItem]:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(
        transcript,
        "assistant",
        "a moderately long assistant answer that has to wrap across the row width " * 2,
    )
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 600, 0)
    return transcript, option


def test_invalidating_the_delegate_reflows_after_a_font_change(
    qtbot, app_appearance: QApplication
) -> None:
    """The workaround that the defect below lacks: an explicit cache drop reflows.

    ``QApplication.setFont`` posts its change, and a body is measured with the font
    the view will paint with, so the event has to land before the cache is dropped.
    Without the drop the row keeps the height it was first measured at, which is the
    whole point of the claim being pinned here.
    """

    transcript, option = _one_row_transcript(qtbot)
    index = transcript.model().index(0, 0)
    theme.apply_interface_font(app_appearance, InterfacePreferences(font_size=13))
    app_appearance.processEvents()
    small = transcript._delegate.sizeHint(option, index).height()

    theme.apply_interface_font(app_appearance, InterfacePreferences(font_size=26))
    app_appearance.processEvents()
    transcript.bodies.invalidate()

    assert transcript._delegate.sizeHint(option, index).height() > small


#: The strict xfail that used to sit here is now a passing test:
#: ``tests/gui/test_appearance.py::test_a_font_change_reflows_the_transcript``. The
#: defect was that nothing listened for ``FontChange``; the fix is a
#: ``changeEvent`` override on the view. The workaround test above stays, because
#: it pins the narrower claim that dropping the cache is what does the work.


def test_qcolor_follows_a_changed_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repainted palette reaches the painter; this used to be an xfail.

    The old defect was ``COLOR_HEX``, an import-time copy in ``qt/paint.py``.
    ``qcolor()`` now resolves through ``theme.active_theme()`` on every call, so the
    chain under test is ``active theme -> CategoryPalette.resolve -> QColor``.
    ``monkeypatch.setattr`` restores the module global afterwards.
    """

    probe = replace(DARK, categories=replace(DARK.categories, cyan="#123456"))
    monkeypatch.setattr(theme, "_ACTIVE_THEME", probe)

    assert qcolor("cyan").name() == "#123456"
    assert qcolor("green").name() == DARK.categories.green, "untouched hues keep their value"
