"""The category sidebar shared by the two settings dialogs."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QSize, Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from kimix_gui.design import DARK

_MINIMUM_ROW_HEIGHT_ROLE = int(Qt.ItemDataRole.UserRole) + 1


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
                self._apply_item_height(item, height)

    def _apply_row_height(self) -> None:
        height = self._row_height()
        for row in range(self.count()):
            self._apply_item_height(self.item(row), height)

    def set_item_minimum_height(self, item: QListWidgetItem, height: int) -> None:
        """Keep enough total row height for an item-widget's styled content.

        ``QListWidget::item`` padding is taken from the row before Qt lays out an
        item widget. Callers therefore pass the complete row height they need, not
        just the child widget's height. The value is retained across font changes.
        """

        item.setData(_MINIMUM_ROW_HEIGHT_ROLE, max(0, height))
        self._apply_item_height(item, self._row_height())

    def set_sized_item_widget(self, item: QListWidgetItem, widget: QWidget) -> None:
        """Attach a control and reserve room for it plus the styled row padding."""

        self.setItemWidget(item, widget)
        widget.ensurePolished()
        self.set_item_minimum_height(
            item,
            widget.sizeHint().height() + DARK.sizing.settings_row_padding,
        )

    @staticmethod
    def _apply_item_height(item: QListWidgetItem, default_height: int) -> None:
        minimum = item.data(_MINIMUM_ROW_HEIGHT_ROLE)
        height = max(default_height, minimum) if isinstance(minimum, int) else default_height
        item.setSizeHint(QSize(0, height))

    def _row_height(self) -> int:
        return max(
            DARK.sizing.settings_row_min_height,
            self.fontMetrics().lineSpacing() * 2 + DARK.sizing.settings_row_padding,
        )
