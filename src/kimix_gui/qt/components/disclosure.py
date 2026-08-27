"""Reusable text header for showing and hiding a section."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QToolButton, QWidget

from kimix_gui.qt.styling import Variant, style

DISCLOSURE_EXPANDED = "▴"
DISCLOSURE_COLLAPSED = "▾"


class DisclosureHeader(QToolButton):
    """A checkable section label whose leading glyph reflects its state.

    The caller owns the label and translated tooltips. The component owns the
    interaction and presentation of that state, so every disclosure uses the
    same glyphs, keyboard behaviour, and accessibility shape.
    """

    def __init__(
        self,
        label: str,
        *,
        expanded: bool,
        expanded_tooltip: str = "",
        collapsed_tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._expanded_tooltip = expanded_tooltip
        self._collapsed_tooltip = collapsed_tooltip
        self.setAccessibleName(label)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        style(self, variant=Variant.DISCLOSURE)
        self.toggled.connect(self._sync_state)
        self.setChecked(expanded)
        self._sync_state(expanded)

    @property
    def label(self) -> str:
        """Return the label without its state glyph."""

        return self._label

    def set_label(self, label: str) -> None:
        """Replace the caller-owned label while preserving disclosure state."""

        self._label = label
        self.setAccessibleName(label)
        self._sync_state(self.isChecked())

    def _sync_state(self, expanded: bool) -> None:
        glyph = DISCLOSURE_EXPANDED if expanded else DISCLOSURE_COLLAPSED
        self.setText(f"{glyph}  {self._label}")
        tooltip = self._expanded_tooltip if expanded else self._collapsed_tooltip
        self.setToolTip(tooltip)
        self.setAccessibleDescription(tooltip)
