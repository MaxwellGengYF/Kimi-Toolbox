"""UI semantic color palette: what a surface, a line, or an ink *means*."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    """Role-named colors for chrome, text, and interactive states.

    Values are QSS-ready strings (``"#0f1115"`` or ``"rgba(24, 29, 39, 247)"``) so
    this layer stays free of Qt. Names describe the *role*, never the hue: the
    transcript's record-classification hues live in
    :class:`kimix_gui.design.categories.CategoryPalette` instead.
    """

    # Surfaces, from the window backdrop up to the most raised fill.
    bg: str
    surface: str
    panel: str
    boost: str
    #: Translucent fill for floating panels layered over content.
    overlay: str

    # Lines.
    border: str
    #: Outline of a focused input. Currently the same hue as ``accent``, but it is
    #: a distinct role: a re-tint of ``accent`` must not silently move focus rings.
    focus_ring: str

    # Ink.
    text: str
    muted: str
    #: Ink for a hyperlink inside rendered Markdown. Shares ``categories.cyan``'s
    #: value today, but it is a role, not a hue: a link is chrome, and re-tinting
    #: the "this record is a tool call" bar must not repaint prose.
    link: str

    # Accent (primary action) triad.
    accent: str
    accent_hover: str
    #: Ink painted *on top of* an ``accent`` fill (badges, primary buttons).
    on_accent: str

    # Feedback.
    error: str
    success: str
    danger_surface: str
    danger_border: str
