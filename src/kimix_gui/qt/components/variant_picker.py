"""Accessible selector for stable model-variant keys."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QWidget

from kimix_gui.qt.styling import STATE, set_style_property


@dataclass(frozen=True, slots=True)
class VariantOption:
    """Presentation supplied by the Qt boundary for one stable Variant key."""

    key: str
    label: str
    enabled: bool = True


class VariantPicker(QComboBox):
    """A compact native combobox that never identifies a Variant by row number."""

    key_activated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("variant-picker")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(16)
        self.activated.connect(self._emit_key)

    def set_options(
        self,
        options: Iterable[VariantOption],
        *,
        selected_key: str | None,
        missing_label: str,
    ) -> None:
        """Replace options and retain a removed selection as a disabled placeholder."""

        self.blockSignals(True)
        self.clear()
        options_by_key = {option.key: option for option in options}
        missing = selected_key is not None and selected_key not in options_by_key
        if missing:
            self.addItem(missing_label, selected_key)
            self._set_item_enabled(0, False)
        for option in options_by_key.values():
            self.addItem(option.label, option.key)
            self._set_item_enabled(self.count() - 1, option.enabled)
        index = self.findData(selected_key) if selected_key is not None else -1
        self.setCurrentIndex(index)
        self.blockSignals(False)
        set_style_property(self, STATE, "unavailable" if missing else "available")

    def selected_key(self) -> str | None:
        value = self.currentData()
        return value if isinstance(value, str) and value else None

    def _set_item_enabled(self, index: int, enabled: bool) -> None:
        model = self.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(index)
            if item is not None:
                item.setEnabled(enabled)

    def _emit_key(self, _index: int) -> None:
        key = self.selected_key()
        if key is not None:
            self.key_activated.emit(key)
