"""Accessible selector for one stable LLM parameter value."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QWidget

from kimix_gui.qt.styling import STATE, set_style_property


@dataclass(frozen=True, slots=True)
class ParameterValueOption:
    """Qt-boundary presentation for one stable parameter value."""

    value: str
    label: str
    enabled: bool = True


class ParameterPicker(QComboBox):
    """A native combobox that identifies values by token, never by row number."""

    value_activated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(16)
        self.activated.connect(self._emit_value)

    def set_options(
        self,
        options: Iterable[ParameterValueOption],
        *,
        selected_value: str | None,
        missing_label: str,
    ) -> None:
        """Replace options and retain a removed value as a disabled placeholder."""
        self.blockSignals(True)
        self.clear()
        options_by_value = {option.value: option for option in options}
        missing = selected_value is not None and selected_value not in options_by_value
        if missing:
            self.addItem(missing_label, selected_value)
            self._set_item_enabled(0, False)
        for option in options_by_value.values():
            self.addItem(option.label, option.value)
            self._set_item_enabled(self.count() - 1, option.enabled)
        index = self.findData(selected_value) if selected_value is not None else -1
        self.setCurrentIndex(index)
        self.blockSignals(False)
        selected_option = options_by_value.get(selected_value or "")
        unavailable = (
            selected_value is None
            or missing
            or (selected_option is not None and not selected_option.enabled)
        )
        set_style_property(self, STATE, "unavailable" if unavailable else "available")

    def selected_value(self) -> str | None:
        value = self.currentData()
        return value if isinstance(value, str) and value else None

    def _set_item_enabled(self, index: int, enabled: bool) -> None:
        model = self.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(index)
            if item is not None:
                item.setEnabled(enabled)

    def _emit_value(self, _index: int) -> None:
        value = self.selected_value()
        if value is not None:
            self.value_activated.emit(value)
