"""The category sidebar shared by the two settings dialogs."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QSize
from PySide6.QtWidgets import QListWidget, QWidget

from kimix_gui.design import DARK


class SettingsList(QListWidget):
    """One column of category rows, each two lines of the *current* font tall.

    This replaces ``apply_settings_item_height()``, a free function the two
    dialogs called at the end of their constructors. A function stamped on the
    height exactly once, which left two ways to be wrong: change the interface
    font and every row keeps the size it had under the old one, or add a row later
    and it comes back at Qt's default. Owning the rule inside the widget removes
    both, and removes the "remember to call it" step as well.

    Two lines rather than one because a category label may wrap, and the row has
    to look the same whether it wrapped or not.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model().rowsInserted.connect(self._on_rows_inserted)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        # Qt sends this to every widget whose font actually moved, nested or not.
        # A widget that had ``setFont`` called on it is excluded, which is correct
        # here: nobody pins this list's font.
        if event.type() == QEvent.Type.FontChange:
            self._apply_row_height()

    def _on_rows_inserted(self, _parent: QModelIndex, first: int, last: int) -> None:
        height = self._row_height()
        for row in range(first, last + 1):
            item = self.item(row)
            if item is not None:
                item.setSizeHint(QSize(0, height))

    def _apply_row_height(self) -> None:
        height = self._row_height()
        for row in range(self.count()):
            self.item(row).setSizeHint(QSize(0, height))

    def _row_height(self) -> int:
        return max(
            DARK.sizing.settings_row_min_height,
            self.fontMetrics().lineSpacing() * 2 + DARK.sizing.settings_row_padding,
        )
