"""The dynamic-property vocabulary QSS selects on.

``objectName`` answers *which widget is this* (and stays the test hook). These
properties answer *how should it look*, so one QSS rule can dress every widget
that shares an intent instead of one rule per name.

``tests/gui/test_styling.py`` cross-checks these constants against the attribute
selectors in :mod:`kimix_gui.qt.theme`, in both directions: a value declared
here with no rule, or a rule naming a value not declared here, is a failure.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtWidgets import QWidget

VARIANT: Final = "variant"
"""How a control is filled: see :class:`Variant`."""

ROLE: Final = "role"
"""What a piece of text is for: see :class:`Role`."""

TONE: Final = "tone"
"""Colour intent on its own, with no metrics attached: see :class:`Tone`."""

LEVEL: Final = "level"
"""Rank within a role, for roles that come in more than one size."""

SURFACE: Final = "surface"
"""Container background treatment: see :class:`Surface`."""

CARD: Final = "card"
"""How far a card sits off the page: see :class:`CardLevel`."""

METRIC: Final = "metric"
"""A named metric preset (min-width and padding): see :class:`Metric`.

``size`` would have been the obvious name and is a trap: ``QWidget`` already
declares a ``size`` property of type ``QSize``, so ``setProperty("size", ...)``
silently resizes the widget instead of tagging it.
"""

STATE: Final = "state"
"""Domain state, not an appearance choice -- mirrors ``TodoEntry.status``."""

KIND: Final = "kind"
"""Domain kind, not an appearance choice -- mirrors the session badge text."""

MODE: Final = "mode"
"""Collapsed/expanded shape of the todo panel."""

FLASH: Final = "flash"
"""Transient attention border on the todo panel."""


class Variant:
    """Visual treatments for interactive controls."""

    PRIMARY: Final = "primary"
    DANGER: Final = "danger"
    GHOST: Final = "ghost"
    ICON: Final = "icon"
    DISCLOSURE: Final = "disclosure"


class Role:
    """Typographic roles for ``QLabel``."""

    DISPLAY: Final = "display"
    TITLE: Final = "title"
    SECTION: Final = "section"
    OVERLINE: Final = "overline"
    CAPTION: Final = "caption"
    FOOTNOTE: Final = "footnote"
    MARKER: Final = "marker"


class Tone:
    """Colour-only intents, usable on any widget."""

    MUTED: Final = "muted"
    DANGER: Final = "danger"


class Surface:
    """Container background treatments."""

    BAR: Final = "bar"


class CardLevel:
    """How far a bordered container sits off the page.

    Three cards existed as three unrelated ID rules, each with its own pick of
    background and radius, three hundred lines apart in the style sheet. They are
    one family with three steps: the further off the page, the rounder the corner
    and the roomier the padding.
    """

    # Above the window entirely -- the composer pad's card.
    FLOATING: Final = "floating"
    # A region of the page -- the home view's session detail.
    PANEL: Final = "panel"
    # Nested inside a form -- the preferences font preview.
    INSET: Final = "inset"


class Metric:
    """Named metric presets for ``QPushButton``."""

    ACTION: Final = "action"
    NAV: Final = "nav"


class Level:
    """Ranks for roles that exist in more than one size."""

    ONE: Final = "1"
    TWO: Final = "2"


def repolish(widget: QWidget) -> None:
    """Re-run the style engine after a selector-visible property changed.

    Qt resolves QSS once per polish, so a ``setProperty`` after that is
    invisible until the widget is unpolished and polished again. ``update()``
    is part of the contract: ``polish()`` alone does not schedule a repaint,
    which is how the two hand-rolled copies of this helper used to disagree.
    """

    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_style_property(widget: QWidget, name: str, value: object) -> bool:
    """Set one property and repolish only when the value actually changed.

    Returns whether a repolish happened, so callers can chain further work.
    """

    if widget.property(name) == value:
        return False
    widget.setProperty(name, value)
    repolish(widget)
    return True


def style(
    widget: QWidget,
    *,
    variant: str | None = None,
    role: str | None = None,
    tone: str | None = None,
    level: str | None = None,
    surface: str | None = None,
    metric: str | None = None,
    card: str | None = None,
) -> QWidget:
    """Declare a widget's appearance intent, returning it for inline use.

    Meant for construction time, before the first polish, so it does not
    repolish; use :func:`set_style_property` for changes made later.
    """

    for name, value in (
        (VARIANT, variant),
        (ROLE, role),
        (TONE, tone),
        (LEVEL, level),
        (SURFACE, surface),
        (METRIC, metric),
        (CARD, card),
    ):
        if value is not None:
            widget.setProperty(name, value)
    return widget


__all__ = [
    "CARD",
    "FLASH",
    "KIND",
    "LEVEL",
    "METRIC",
    "MODE",
    "ROLE",
    "STATE",
    "SURFACE",
    "TONE",
    "VARIANT",
    "CardLevel",
    "Level",
    "Metric",
    "Role",
    "Surface",
    "Tone",
    "Variant",
    "repolish",
    "set_style_property",
    "style",
]
