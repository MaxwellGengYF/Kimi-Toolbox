"""The ``Theme`` aggregate and the two themes that exist."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from kimix_gui.design.categories import CategoryPalette
from kimix_gui.design.palette import Palette
from kimix_gui.design.scale import (
    Breakpoints,
    CardPadding,
    ComposerMetrics,
    Motion,
    RadiusScale,
    SessionListMetrics,
    Sizing,
    SpacingScale,
    TodoPanelMetrics,
    TrackingScale,
    TranscriptMetrics,
    TypeScale,
)


@dataclass(frozen=True, slots=True)
class Theme:
    """One complete set of design tokens.

    Pure data: no Qt import anywhere below this module, so the whole token layer is
    reachable from plain pytest.
    """

    name: str
    palette: Palette
    categories: CategoryPalette
    spacing: SpacingScale = field(default_factory=SpacingScale)
    radius: RadiusScale = field(default_factory=RadiusScale)
    type_scale: TypeScale = field(default_factory=TypeScale)
    tracking: TrackingScale = field(default_factory=TrackingScale)
    sizing: Sizing = field(default_factory=Sizing)
    card_padding: CardPadding = field(default_factory=CardPadding)
    motion: Motion = field(default_factory=Motion)
    breakpoints: Breakpoints = field(default_factory=Breakpoints)
    transcript: TranscriptMetrics = field(default_factory=TranscriptMetrics)
    todo_panel: TodoPanelMetrics = field(default_factory=TodoPanelMetrics)
    session_list: SessionListMetrics = field(default_factory=SessionListMetrics)
    composer: ComposerMetrics = field(default_factory=ComposerMetrics)


_MUTED = "#8b95a8"
_CYAN = "#22d3ee"
_RED = "#f87171"
_GREEN = "#4ade80"
_ACCENT = "#5eead4"

DARK = Theme(
    name="dark",
    palette=Palette(
        bg="#0f1115",
        surface="#161a22",
        panel="#1c212c",
        boost="#252b38",
        overlay="rgba(24, 29, 39, 247)",
        border="#2c3342",
        focus_ring=_ACCENT,
        text="#e8edf5",
        muted=_MUTED,
        link=_CYAN,
        accent=_ACCENT,
        accent_hover="#2dd4bf",
        on_accent="#042f2e",
        error=_RED,
        success=_GREEN,
        danger_surface="#3f1d22",
        danger_border="#7f1d1d",
    ),
    categories=CategoryPalette(
        cyan=_CYAN,
        green=_GREEN,
        red=_RED,
        yellow="#facc15",
        blue="#60a5fa",
        magenta="#e879f9",
        muted=_MUTED,
    ),
)
"""The dark theme: the one this app shipped with, and the default."""

_LIGHT_MUTED = "#5c6779"
_LIGHT_CYAN = "#0e7490"
_LIGHT_RED = "#dc2626"
_LIGHT_GREEN = "#16a34a"
_LIGHT_ACCENT = "#0d9488"

LIGHT = Theme(
    name="light",
    palette=Palette(
        # The surface ramp runs the other way here: ``bg`` is the *lightest* fill and
        # each step above it is darker. The invariant that survives both themes is
        # "further from the page means further from ``bg``", which is what
        # ``tests/gui/test_design_tokens.py`` checks -- not "lighter".
        bg="#ffffff",
        surface="#f5f7fa",
        panel="#eceff4",
        boost="#dfe4ec",
        overlay="rgba(246, 248, 250, 247)",
        border="#cfd6e0",
        focus_ring=_LIGHT_ACCENT,
        # Not the dark theme's ``bg`` (``#0f1115``), tempting as the symmetry is: a hex
        # shared between the two palettes cannot be attributed to a theme, and the
        # pixel guards in ``tests/gui/test_theme_switching.py`` work by attribution.
        text="#111826",
        muted=_LIGHT_MUTED,
        link=_LIGHT_CYAN,
        # Two steps darker than the dark theme's teal. The same hue at 300 weight is
        # unreadable as ink on white, and this token is ink as often as it is fill
        # (``role=marker`` text, focus rings, the ``level`` badges).
        accent=_LIGHT_ACCENT,
        accent_hover="#0f766e",
        on_accent="#f0fdfa",
        error=_LIGHT_RED,
        success=_LIGHT_GREEN,
        danger_surface="#fee2e2",
        danger_border="#fca5a5",
    ),
    categories=CategoryPalette(
        # 600-weight hues, for the same reason as the accent: the dark theme's 400s
        # are chosen to glow against ``#0f1115`` and turn to pastel mush on white.
        cyan=_LIGHT_CYAN,
        green=_LIGHT_GREEN,
        red=_LIGHT_RED,
        yellow="#a16207",
        blue="#2563eb",
        magenta="#a21caf",
        muted=_LIGHT_MUTED,
    ),
)
"""The light theme. Exists to keep the token layer honest: a second instance is the
only thing that can prove no color is hiding in the widgets."""

#: Every theme, by the name stored in the preferences file. A mapping proxy because
#: this is a registry, not a scratch dict.
THEMES: Final[Mapping[str, Theme]] = MappingProxyType({DARK.name: DARK, LIGHT.name: LIGHT})

#: What ``auto`` means when nothing can say what the desktop prefers, and what an
#: unrecognised stored value falls back to.
DEFAULT_THEME: Final = DARK.name

#: Stored value meaning "whatever the desktop is set to". Mirrors the language
#: preference, which uses the same word for the same idea.
SYSTEM_THEME: Final = "auto"

#: Everything the preferences dialog may offer, in the order it offers them.
SUPPORTED_THEMES: Final[tuple[str, ...]] = (SYSTEM_THEME, DARK.name, LIGHT.name)


def normalize_theme_preference(value: object) -> str:
    """Coerce a stored or user-supplied theme preference to something storable.

    Anything unrecognised becomes ``auto`` rather than ``dark``: a preferences file
    written by a future version that knows a third theme should degrade to
    "follow the desktop", not to "force the old default".
    """

    if isinstance(value, str) and value in SUPPORTED_THEMES:
        return value
    return SYSTEM_THEME


def resolve_theme(preference: str, desktop_prefers_dark: bool | None) -> Theme:
    """Turn a stored preference plus the desktop's hint into an actual theme.

    ``desktop_prefers_dark`` is ``None`` when nothing could be determined, which is
    the common case on a platform Qt has no color-scheme hint for. It is passed in
    rather than detected here because detecting it needs Qt, and this layer has
    none -- the same split as ``resolve_language`` and ``system_locale_name``.
    """

    if preference == SYSTEM_THEME:
        if desktop_prefers_dark is None:
            return THEMES[DEFAULT_THEME]
        return DARK if desktop_prefers_dark else LIGHT
    return THEMES.get(preference, THEMES[DEFAULT_THEME])
