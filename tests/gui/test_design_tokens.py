"""Unit tests for the framework-free design token layer (no Qt needed)."""

from __future__ import annotations

import pathlib
import re
from dataclasses import MISSING, FrozenInstanceError, fields, replace

import pytest

import kimix_gui.design as design_package
from kimix_gui.design import (
    CATEGORY_NAMES,
    DARK,
    DEFAULT_THEME,
    LIGHT,
    SUPPORTED_THEMES,
    SYSTEM_THEME,
    THEMES,
    Breakpoints,
    CardPadding,
    CategoryPalette,
    ComposerMetrics,
    Motion,
    Palette,
    RadiusScale,
    Sizing,
    SpacingScale,
    Theme,
    TodoPanelMetrics,
    TrackingScale,
    TranscriptMetrics,
    TypeScale,
    normalize_theme_preference,
    resolve_theme,
)
from kimix_gui.tool_display import KNOWN_TOOL_FAMILIES, is_known_tool_family
from kimix_gui.transcript_layout import BAR_COLOR_NAME, FAMILY_BAR_NAME

_COLOR_RE = re.compile(r"^(#[0-9a-f]{6}|rgba\(\d+, \d+, \d+, \d+\))$")


# --------------------------------------------------------------------------- #
# Layering: the token layer must stay Qt-free
# --------------------------------------------------------------------------- #


def _design_sources() -> list[pathlib.Path]:
    package = pathlib.Path(design_package.__file__).parent
    return sorted(package.glob("*.py"))


def test_the_design_package_never_imports_pyside6() -> None:
    """``src/kimix_gui/design`` is a pure layer; importing Qt there breaks it."""

    sources = _design_sources()
    assert [path.name for path in sources] == [
        "__init__.py",
        "categories.py",
        "palette.py",
        "scale.py",
        "theme.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "PySide6" not in text, f"{path.name} imports Qt"
        assert "QtCore" not in text, f"{path.name} imports QtCore"


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("group", "attribute", "value"),
    [
        (DARK, "name", "light"),
        (DARK.palette, "accent", "#ff0000"),
        (DARK.categories, "cyan", "#ff0000"),
        (DARK.spacing, "md", 99),
        (DARK.radius, "md", 99),
        (DARK.type_scale, "base", 99),
        (DARK.tracking, "wide", 9.0),
        (DARK.sizing, "border_width", 99),
        (DARK.motion, "fade_ms", 99),
        (DARK.breakpoints, "home_narrow", 99),
        (DARK.card_padding, "wide", (1, 1, 1, 1)),
        (DARK.transcript, "pad_x", 99),
        (DARK.todo_panel, "margin", 99),
        (DARK.composer, "min_height", 99),
    ],
)
def test_every_token_group_is_frozen(group: object, attribute: str, value: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(group, attribute, value)


def test_token_groups_have_no_instance_dict() -> None:
    """``slots=True`` everywhere, so a typo cannot silently add a token."""

    for group in (
        DARK,
        DARK.palette,
        DARK.categories,
        DARK.spacing,
        DARK.radius,
        DARK.type_scale,
        DARK.tracking,
        DARK.sizing,
        DARK.card_padding,
        DARK.motion,
        DARK.breakpoints,
        DARK.transcript,
        DARK.todo_panel,
        DARK.composer,
    ):
        assert not hasattr(group, "__dict__")


def test_replace_produces_an_independent_variant() -> None:
    """The builder tests derive probe themes this way; DARK must stay untouched."""

    probe = replace(DARK, name="probe", palette=replace(DARK.palette, accent="#ff0000"))
    assert probe.palette.accent == "#ff0000"
    assert DARK.palette.accent == "#5eead4"
    assert probe.spacing == DARK.spacing


# --------------------------------------------------------------------------- #
# Scale values
# --------------------------------------------------------------------------- #


def test_spacing_scale_steps() -> None:
    assert (DARK.spacing.none, DARK.spacing.hairline, DARK.spacing.xxs) == (0, 1, 2)
    assert (DARK.spacing.xs, DARK.spacing.sm, DARK.spacing.md) == (4, 6, 8)
    assert (DARK.spacing.lg, DARK.spacing.lg_plus, DARK.spacing.xl) == (10, 11, 12)
    assert (DARK.spacing.xxl, DARK.spacing.xxxl) == (14, 18)


def test_radius_scale_steps() -> None:
    assert (DARK.radius.xs, DARK.radius.sm, DARK.radius.md, DARK.radius.lg) == (4, 6, 8, 10)
    assert (DARK.radius.xl, DARK.radius.xxl) == (12, 16)
    assert (DARK.radius.pill, DARK.radius.pill_lg) == (17, 18)


def test_type_scale_steps() -> None:
    assert (DARK.type_scale.micro, DARK.type_scale.xs, DARK.type_scale.sm) == (9, 11, 12)
    assert (DARK.type_scale.base, DARK.type_scale.md, DARK.type_scale.lg) == (13, 14, 16)
    assert (DARK.type_scale.xl, DARK.type_scale.xxl, DARK.type_scale.display) == (18, 20, 22)
    assert (DARK.type_scale.weight_semibold, DARK.type_scale.weight_bold) == (600, 700)
    assert DARK.type_scale.base_family == "Segoe UI"


def test_tracking_scale_steps() -> None:
    assert (DARK.tracking.none, DARK.tracking.wide, DARK.tracking.wider) == (0.0, 0.4, 0.8)


def test_sizing_and_motion_and_breakpoints() -> None:
    assert DARK.sizing.border_width == 1
    assert DARK.sizing.selection_bar_width == 3
    assert DARK.sizing.icon_button_min_width == 22
    assert DARK.sizing.nav_button_min_width == 28
    assert DARK.sizing.action_button_min_width == 64
    assert DARK.sizing.scrollbar_width == 10
    assert DARK.sizing.scrollbar_handle_min_height == 24
    assert DARK.sizing.compact_control_height == 28, "history nav + home toolbar buttons"
    assert DARK.sizing.action_control_height == 36, "send / cancel buttons"
    assert DARK.motion.fade_ms == 160, "Toast._fade_out duration"
    assert DARK.motion.flash_ms == 1600, "TodoPanel collapsed-pill flash"
    assert DARK.breakpoints.home_narrow == 780, "HomeView._sync_narrow"
    assert DARK.breakpoints.settings_narrow == 880, "LLMSettingsDialog.resizeEvent"


def test_spacing_steps_ascend() -> None:
    names = [field.name for field in fields(SpacingScale)]
    assert names == [
        "none",
        "hairline",
        "xxs",
        "xs",
        "sm",
        "md",
        "lg",
        "lg_plus",
        "xl",
        "xxl",
        "xxxl",
    ]
    values = [getattr(DARK.spacing, name) for name in names]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "no two spacing steps share a value"


def test_radius_steps_ascend() -> None:
    names = [field.name for field in fields(RadiusScale)]
    values = [getattr(DARK.radius, name) for name in names]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "no two radius steps share a value"


def test_scales_default_construct_to_the_dark_values() -> None:
    """The scales carry defaults, so a future theme only re-declares colors."""

    assert DARK.spacing == SpacingScale()
    assert DARK.radius == RadiusScale()
    assert DARK.type_scale == TypeScale()
    assert DARK.tracking == TrackingScale()
    assert DARK.sizing == Sizing()
    assert DARK.card_padding == CardPadding()
    assert DARK.motion == Motion()
    assert DARK.breakpoints == Breakpoints()
    assert DARK.transcript == TranscriptMetrics()
    assert DARK.todo_panel == TodoPanelMetrics()
    assert DARK.composer == ComposerMetrics()


def test_transcript_metrics_match_the_painted_geometry() -> None:
    """The eleven literals ``qt/transcript.py`` used to declare inline.

    ``tests/gui/test_render_geometry.py`` pins the same numbers from the Qt side; this
    is the pure-layer half of the same contract.
    """

    metrics = DARK.transcript
    assert (metrics.pad_x, metrics.pad_y) == (12, 8)
    assert (metrics.card_margin_x, metrics.card_margin_y) == (8, 4)
    assert (metrics.bar_width, metrics.bar_width_hover, metrics.bar_hit_width) == (3, 5, 12)
    assert (metrics.header_height, metrics.line_height, metrics.copy_width) == (22, 18, 36)
    assert metrics.bar_width < metrics.bar_width_hover < metrics.bar_hit_width


def test_todo_panel_metrics_match_the_panel_geometry() -> None:
    """The twelve literals ``qt/todo_panel.py`` used to declare inline.

    ``MIN_BODY_HEIGHT`` is intentionally absent: it is derived as
    ``row_height + 2 * body_padding``, so it never was a literal of its own.
    """

    metrics = DARK.todo_panel
    assert (metrics.margin, metrics.card_width, metrics.min_card_width) == (14, 336, 200)
    assert (metrics.header_height, metrics.bar_height, metrics.footer_height) == (34, 3, 26)
    assert (metrics.row_height, metrics.row_with_notes_height, metrics.row_spacing) == (28, 44, 2)
    assert (metrics.body_padding, metrics.max_body_height, metrics.indent_step) == (6, 320, 14)
    assert metrics.row_height + 2 * metrics.body_padding < metrics.max_body_height


def test_composer_metrics_and_card_padding_steps() -> None:
    """Composer heights, plus the three card inset combinations found in the views."""

    assert (DARK.composer.min_height, DARK.composer.max_height) == (52, 130)
    assert DARK.composer.min_height < DARK.composer.max_height
    assert DARK.card_padding.wide == (20, 16, 20, 16)
    assert DARK.card_padding.detail == (18, 16, 18, 16)
    assert DARK.card_padding.compact == (14, 12, 14, 12)
    for inset in (DARK.card_padding.wide, DARK.card_padding.detail, DARK.card_padding.compact):
        left, top, right, bottom = inset
        assert (left, top) == (right, bottom), "every card inset is symmetric today"


# --------------------------------------------------------------------------- #
# Palette shape and cross-token references
# --------------------------------------------------------------------------- #


def test_every_theme_is_a_named_theme() -> None:
    for name, theme in THEMES.items():
        assert isinstance(theme, Theme)
        assert theme.name == name, "the registry key is the theme's own name"
        assert isinstance(theme.palette, Palette)
        assert isinstance(theme.categories, CategoryPalette)


def test_the_registry_holds_exactly_the_themes_that_exist() -> None:
    """Two themes, and the light one is not the dark one wearing a different name."""

    assert set(THEMES) == {"dark", "light"}
    assert THEMES["dark"] is DARK
    assert THEMES["light"] is LIGHT
    assert DARK.palette != LIGHT.palette
    assert DARK.categories != LIGHT.categories


def test_the_registry_cannot_be_mutated() -> None:
    """A registry a caller can append to is a registry that stops matching reality."""

    with pytest.raises(TypeError):
        THEMES["solarized"] = DARK  # type: ignore[index]


@pytest.mark.parametrize("theme_name", sorted(THEMES))
@pytest.mark.parametrize("group_name", ["palette", "categories"])
def test_color_tokens_are_qss_ready_strings(group_name: str, theme_name: str) -> None:
    group = getattr(THEMES[theme_name], group_name)
    for field in fields(group):
        value = getattr(group, field.name)
        assert _COLOR_RE.match(value), f"{theme_name}.{group_name}.{field.name} = {value!r}"


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_every_theme_declares_every_color(theme_name: str) -> None:
    """No theme may leave a color to a default: colors are the one thing themes own.

    ``Palette`` and ``CategoryPalette`` have no field defaults, so this is really a
    guard on the dataclasses staying that way. The moment one grows a default,
    a new theme can silently inherit a dark-theme hex.
    """

    for group in (Palette, CategoryPalette):
        for field in fields(group):
            assert field.default is MISSING, f"{group.__name__}.{field.name} has a default"


def test_the_six_newly_named_semantics_exist() -> None:
    """The literals that used to be bare in the QSS now have names."""

    assert DARK.palette.on_accent == "#042f2e"
    assert DARK.palette.accent_hover == "#2dd4bf"
    assert DARK.palette.danger_surface == "#3f1d22"
    assert DARK.palette.danger_border == "#7f1d1d"
    assert DARK.palette.overlay == "rgba(24, 29, 39, 247)"


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_shared_values_are_declared_once_per_role(theme_name: str) -> None:
    """Roles that intentionally share a hex value stay pinned to each other."""

    theme = THEMES[theme_name]
    assert theme.palette.muted == theme.categories.muted
    assert theme.palette.error == theme.categories.red
    assert theme.palette.success == theme.categories.green
    assert theme.palette.focus_ring == theme.palette.accent, "same hue today, separate role"
    assert theme.palette.link == theme.categories.cyan, "prose links vs. tool-call bars"


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_the_surface_ramp_moves_away_from_bg_one_step_at_a_time(theme_name: str) -> None:
    """Each fill above ``bg`` is further from ``bg``, and never doubles back.

    The direction is the theme's own business: the dark theme lightens as it rises,
    the light theme darkens. What both owe is monotonicity, because the ramp is how
    elevation is expressed -- a ``panel`` that lands between ``bg`` and ``surface``
    would read as *below* the thing it sits on.
    """

    palette = THEMES[theme_name].palette
    ramp = [palette.bg, palette.surface, palette.panel, palette.boost]
    luma = [sum(int(color[i : i + 2], 16) for i in (1, 3, 5)) for color in ramp]
    assert luma == sorted(luma) or luma == sorted(luma, reverse=True), luma
    assert len(set(luma)) == len(luma), "two rungs at the same brightness is one rung"


def test_the_two_themes_ramp_in_opposite_directions() -> None:
    """Pins the fact the previous test deliberately stops short of asserting.

    Without this, a "light" theme that merely lightened the dark ramp a little would
    pass every guard above while still being a dark theme.
    """

    def luma(color: str) -> int:
        return sum(int(color[i : i + 2], 16) for i in (1, 3, 5))

    assert luma(DARK.palette.bg) < luma(DARK.palette.boost)
    assert luma(LIGHT.palette.bg) > luma(LIGHT.palette.boost)
    assert luma(LIGHT.palette.bg) > luma(DARK.palette.boost), "no overlap between ramps"
    assert luma(LIGHT.palette.text) < luma(DARK.palette.text), "ink flips with the page"


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_text_and_muted_carry_contrast_against_the_page(theme_name: str) -> None:
    """Cheap readability floor: the light theme's first draft failed this on yellow.

    Not WCAG. Full contrast validation needs the real relative-luminance formula and
    a decision about which pairs actually meet on screen; this only catches a token
    that was copied across themes without being re-picked.

    The 350 floor was chosen by measurement, not taste: the tightest hue either theme
    actually ships sits at 371 (dark green on the dark page) and 394 (light blue on
    white), while the dark theme's yellow -- the one hue a careless copy would most
    likely leave behind -- lands at 290 against white. A looser floor let that copy
    through, which is how this number got picked.
    """

    theme = THEMES[theme_name]

    def luma(color: str) -> int:
        return sum(int(color[i : i + 2], 16) for i in (1, 3, 5))

    page = luma(theme.palette.bg)
    assert abs(luma(theme.palette.text) - page) > 400, "body text"
    assert abs(luma(theme.palette.muted) - page) > 350, "secondary text"
    for name, hue in theme.categories.as_map().items():
        assert abs(luma(hue) - page) > 350, f"category {name} on the page"
    assert abs(luma(theme.palette.on_accent) - luma(theme.palette.accent)) > 200, "ink on accent"


# --------------------------------------------------------------------------- #
# Resolving a stored preference to a theme
# --------------------------------------------------------------------------- #


def test_the_theme_preference_vocabulary_mirrors_the_language_one() -> None:
    """Same shape as ``kimix_gui.i18n``, so neither has to be learned twice."""

    assert SYSTEM_THEME == "auto"
    assert DEFAULT_THEME == "dark"
    assert SUPPORTED_THEMES == ("auto", "dark", "light")
    assert SUPPORTED_THEMES[0] == SYSTEM_THEME, "follow-the-desktop is offered first"
    assert set(SUPPORTED_THEMES) - {SYSTEM_THEME} == set(THEMES)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("dark", "dark"),
        ("light", "light"),
        ("auto", "auto"),
        ("Dark", "auto"),
        ("solarized", "auto"),
        ("", "auto"),
        (None, "auto"),
        (7, "auto"),
        (True, "auto"),
        (["dark"], "auto"),
    ],
)
def test_normalize_theme_preference_only_keeps_what_can_be_offered(
    value: object, expected: str
) -> None:
    assert normalize_theme_preference(value) == expected


@pytest.mark.parametrize(
    ("preference", "prefers_dark", "expected"),
    [
        ("dark", None, DARK),
        ("dark", True, DARK),
        ("dark", False, DARK),
        ("light", None, LIGHT),
        ("light", True, LIGHT),
        ("light", False, LIGHT),
        ("auto", True, DARK),
        ("auto", False, LIGHT),
        ("auto", None, DARK),
        ("solarized", False, DARK),
    ],
)
def test_resolve_theme(preference: str, prefers_dark: bool | None, expected: Theme) -> None:
    """An explicit choice wins over the desktop; ``auto`` is the only one that asks."""

    assert resolve_theme(preference, prefers_dark) is expected


def test_an_unknown_desktop_hint_lands_on_the_shipped_default() -> None:
    """``None`` means "could not tell", which must not silently mean "light"."""

    assert resolve_theme(SYSTEM_THEME, None) is THEMES[DEFAULT_THEME]


# --------------------------------------------------------------------------- #
# Category palette vs. the layout that addresses it
# --------------------------------------------------------------------------- #


def test_category_names_match_the_layout_domain() -> None:
    """``transcript_layout`` may only emit names the category palette can resolve."""

    emitted = set(BAR_COLOR_NAME.values()) | set(FAMILY_BAR_NAME.values())
    assert emitted <= set(CATEGORY_NAMES)
    assert set(DARK.categories.as_map()) == set(CATEGORY_NAMES)
    assert len(CATEGORY_NAMES) == 7


def test_the_specialized_families_and_their_colors_are_one_list() -> None:
    """``KNOWN_TOOL_FAMILIES`` and ``FAMILY_BAR_NAME`` must name the same families.

    They were three tables: ``tool_display`` also carried a family -> Rich markup map
    and a family -> bar color map whose values nothing read, and whose colors had
    already drifted (``bright_cyan`` here, plain ``cyan`` in the layout). Only the
    layout emits colors now, so a family added on one side and forgotten on the other
    would silently get a muted bar.
    """
    assert KNOWN_TOOL_FAMILIES == set(FAMILY_BAR_NAME)
    assert is_known_tool_family("read") is True
    assert is_known_tool_family("bash") is True  # an alias of the shell family
    assert is_known_tool_family("some_random_tool") is False
    assert is_known_tool_family(None) is False


def test_category_resolve_falls_back_to_muted() -> None:
    assert DARK.categories.resolve("cyan") == DARK.categories.cyan
    assert DARK.categories.resolve("chartreuse") == DARK.categories.muted
    assert DARK.categories.resolve("") == DARK.categories.muted


def test_ui_roles_and_category_hues_stay_in_separate_namespaces() -> None:
    """No hue name leaks into the semantic palette, no role name into the hues."""

    role_names = {field.name for field in fields(Palette)}
    hue_names = set(CATEGORY_NAMES) - {"muted"}
    assert role_names & hue_names == set()


def test_session_list_metrics_match_the_hand_painted_rows() -> None:
    """The literals ``SessionRow`` and ``SelectionMark`` used to draw with.

    Those two keep their painters -- an inset rounded marker has no style sheet
    equivalent -- so a token is the only way a theme can reach their geometry.
    """

    metrics = DARK.session_list
    assert (metrics.row_height, metrics.row_radius, metrics.mark_size) == (58, 10, 22)
    assert metrics.mark_idle_opacity == 0.42
    assert (metrics.marker_x, metrics.marker_width) == (1.5, 3)
    assert (metrics.marker_inset_y, metrics.marker_radius) == (14, 1.5)
    assert 0.0 < metrics.mark_idle_opacity < 1.0
    assert metrics.mark_size < metrics.row_height, "the mark has to fit inside the row"
    assert 2 * metrics.marker_inset_y < metrics.row_height, "the marker would be inverted"
    assert metrics.row_radius == DARK.radius.lg, "same corner as the composer prompt"
