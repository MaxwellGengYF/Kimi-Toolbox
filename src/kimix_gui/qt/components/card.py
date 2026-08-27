"""A bordered container that holds content off the page."""

from __future__ import annotations

from typing import Final

from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from kimix_gui.design import DARK
from kimix_gui.qt.styling import CardLevel, style

# Padding per level. The three existing cards already used three different
# insets, and they lined up with the three levels one-to-one, so the mapping is
# a record of what was there rather than a new decision. The token names come
# from ``CardPadding``, which predates the levels and is also used for a page
# root, so they are not renamed to match.
_PADDING: Final = {
    CardLevel.FLOATING: DARK.card_padding.wide,
    CardLevel.PANEL: DARK.card_padding.detail,
    CardLevel.INSET: DARK.card_padding.compact,
}


class Card(QFrame):
    """Content in a bordered box, at one of three elevations.

    Background, border and radius come from the level, so a new card picks a step
    on that scale instead of inventing another background/radius pair -- which is
    how three cards ended up with three radii and no stated relationship between
    them.

    Padding comes from the level too. Spacing does not: the three call sites use
    different internal rhythms and none of them is wrong, so that stays with the
    caller, reachable through :attr:`body`.
    """

    def __init__(self, level: str, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        style(self, card=level)
        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(*_PADDING[level])

    @property
    def body(self) -> QVBoxLayout:
        """The layout to put content in."""
        return self._body
