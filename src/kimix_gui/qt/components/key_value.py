"""A labelled column of read-only values: session details, config details."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QLabel, QWidget

from kimix_gui.design import DARK
from kimix_gui.qt.styling import Tone, style


class KeyValueList(QWidget):
    """Field name on the left, value on the right, for facts the user reads.

    Two panes were already doing this and neither did it the same way. The home
    view built rows by hand out of ``QHBoxLayout`` and pinned the name column to
    ``setFixedWidth(110)``; the config details pane used a ``QFormLayout`` and let
    Qt size the column. A pinned column is a translation bug waiting for a longer
    word -- "Extra folders" fits in 110px, ``Zusätzliche Ordner`` does not -- so
    the form layout wins and the column sizes itself to whatever the labels say.

    The two panes also disagreed on the details. Names are muted here, as they
    were in the home view: the value is the information, the name is furniture.
    Values are selectable, as they were in the home view: a session id or an
    endpoint you cannot copy is a session id you have to retype.

    Rows are ``(object_name, label)``. The object name goes on the *value*, which
    is what callers update and what tests read; the label is display copy and is
    expected to arrive already translated.
    """

    def __init__(
        self,
        rows: Iterable[tuple[str, str]],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(DARK.spacing.md)
        form.setVerticalSpacing(DARK.spacing.md)
        # Without this, a long value stretches the widget instead of wrapping, and
        # the details pane pushes the splitter around as the selection changes.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        keys: dict[str, QLabel] = {}
        values: dict[str, QLabel] = {}
        for object_name, label in rows:
            name = QLabel(label)
            name.setObjectName("key-value-key")
            keys[object_name] = name
            style(name, tone=Tone.MUTED)
            value = QLabel("")
            value.setObjectName(object_name)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            # A screen reader reading down the column would otherwise announce a
            # run of bare values. The name is not focusable, so it cannot serve as
            # a buddy; saying it here is what makes the pairing audible.
            value.setAccessibleName(label)
            values[object_name] = value
            form.addRow(name, value)
        self._keys = keys
        self._values = values

    @property
    def values(self) -> Mapping[str, QLabel]:
        """The value labels, keyed by object name."""
        return self._values

    def set_value(self, object_name: str, text: str) -> None:
        self._values[object_name].setText(text)

    def set_labels(self, rows: Iterable[tuple[str, str]]) -> None:
        """Restate the field names, for callers whose copy can change after build.

        The accessible name moves with the visible one. They are the same string at
        construction, and a screen reader announcing the previous language while the
        screen shows the current one is the same bug, only harder to notice.
        """

        for object_name, label in rows:
            self._keys[object_name].setText(label)
            self._values[object_name].setAccessibleName(label)
