"""The one selectable row of the session browser, and its circular select mark.

Split out of ``home_view`` because a list item is not the view that lists it: the row
owns its own hover and selection state, paints itself from theme tokens, and says its own
copy again on a language change. Nothing here knows about search, batch delete, or the
details pane.

The two classes stay together: ``SelectionMark`` exists only inside a row (it is what
makes the row batch-selectable) and both are hand-painted against the same tokens, so
splitting them again would separate one look into two files. Both are painted rather
than styled on purpose -- the active-row marker is an inset rounded bar and the mark is a
circle with a check path, and no style sheet declaration can place either.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.qt.labels import translate_session_title
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.session_copy import format_file_size, format_relative_time
from kimix_gui.qt.styling import KIND, Role, repolish, style
from kimix_gui.qt.theme import active_theme
from kimix_gui.session_index import SessionSummary

# Import-time bound, like the transcript metrics: the names stay, the numbers
# moved to the token layer.
_MARK_SIZE = DARK.session_list.mark_size


class SelectionMark(QCheckBox):
    """Circular checkbox used for batch-selecting sessions."""

    def __init__(
        self, parent: QWidget | None = None, *, object_name: str = "session-check"
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setText("")
        self.setFixedSize(_MARK_SIZE, _MARK_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._forced = False

    def set_forced(self, forced: bool) -> None:
        self._forced = forced
        self.update()

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        tokens = active_theme()
        palette_tokens = tokens.palette
        checked = self.isChecked()
        mixed = self.checkState() == Qt.CheckState.PartiallyChecked
        idle = not (checked or mixed or self._forced or self.underMouse() or self.isDown())
        painter.setOpacity(1.0 if not idle else tokens.session_list.mark_idle_opacity)
        box = QRectF(2.5, 2.5, 17, 17)
        if checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(palette_tokens.accent))
            painter.drawEllipse(box)
            pen = QPen(QColor(palette_tokens.on_accent), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mark = QPainterPath()
            mark.moveTo(7.2, 11.2)
            mark.lineTo(10.0, 14.1)
            mark.lineTo(15.2, 8.0)
            painter.drawPath(mark)
            return
        active = mixed or self.underMouse() or self.isDown()
        border = QColor(palette_tokens.accent if active else palette_tokens.muted)
        painter.setPen(QPen(border, 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(box)
        if mixed:
            dash = QPen(QColor(palette_tokens.accent), 2.0)
            dash.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(dash)
            painter.drawLine(7, 11, 15, 11)


class SessionRow(QWidget):
    """Selectable session card: click previews, the mark batch-selects."""

    check_toggled = Signal(str)
    opened = Signal(str)

    def __init__(self, summary: SessionSummary, *, selected: bool = False) -> None:
        super().__init__()
        self.summary = summary
        self.selected = selected
        self._active = False
        self._hovered = False
        self._selection_mode = False
        self.setObjectName("session-row")
        self._i18n = Retranslator(self)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 12, 8)
        layout.setSpacing(10)
        self._check = SelectionMark(self)
        self._check.setChecked(selected)
        self._check.clicked.connect(self._emit_check)
        layout.addWidget(self._check, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        self._title = QLabel()
        title = self._title
        title.setObjectName("session-title")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._meta = QLabel()
        meta = self._meta
        meta.setObjectName("session-meta")
        style(meta, role=Role.CAPTION)
        meta.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text.addWidget(title)
        text.addWidget(meta)
        layout.addLayout(text, 1)
        self._badge = QLabel()
        self._badge.setObjectName("session-badge")
        self._badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if summary.is_archived:
            self._badge.setProperty(KIND, "archived")
        elif summary.is_last:
            self._badge.setProperty(KIND, "last")
        else:
            self._badge.hide()
        repolish(self._badge)
        self._i18n.bind(self._refresh_copy)
        layout.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set_selected(selected)

    def _refresh_copy(self) -> None:
        """Every string on this row is derived from its summary, including the two
        that look static: the title falls back to a translated "Untitled", and the
        meta line is a relative time and a file size, both locale-shaped."""

        summary = self.summary
        self._title.setText(translate_session_title(summary.title))
        self._meta.setText(
            f"{format_relative_time(summary.updated_at)} · {format_file_size(summary.size_bytes)}"
        )
        if summary.is_archived:
            self._badge.setText(self.tr("Archived"))
        elif summary.is_last:
            self._badge.setText(self.tr("Last"))

    @property
    def checked(self) -> bool:
        return self._check.isChecked()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._check.blockSignals(True)
        self._check.setChecked(selected)
        self._check.blockSignals(False)
        self._check.set_forced(self._selection_mode or self._hovered or selected)
        self._check.update()

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.update()

    def set_selection_mode(self, active: bool) -> None:
        self._selection_mode = active
        self._check.set_forced(active or self._hovered or self.selected)

    def _emit_check(self) -> None:
        self.check_toggled.emit(self.summary.id)

    def enterEvent(self, event: object) -> None:
        self._hovered = True
        self._check.set_forced(True)
        self.update()
        super().enterEvent(event)  # type: ignore[arg-type]

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self._check.set_forced(self._selection_mode or self.selected)
        self.update()
        super().leaveEvent(event)  # type: ignore[arg-type]

    def mousePressEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if self.childAt(event.position().toPoint()) is self._check:
            return
        list_widget = self._list_widget()
        if list_widget is not None:
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if list_widget.itemWidget(item) is self:
                    list_widget.setCurrentRow(index)
                    break
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if self.childAt(event.position().toPoint()) is self._check:
            return
        self.opened.emit(self.summary.id)
        event.accept()

    def paintEvent(self, event: object) -> None:
        """Hand-painted, and staying that way.

        The fills alone would be two style sheet rules, but the active-row marker is
        an inset rounded bar, and no declaration can place that. Splitting the row
        between a rule and a painter would put its look in two places, so the whole
        row is drawn here -- from tokens, so a theme change still reaches it.
        """

        del event
        tokens = active_theme()
        palette_tokens = tokens.palette
        metrics = tokens.session_list
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._active:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(palette_tokens.boost))
            painter.drawRoundedRect(rect, metrics.row_radius, metrics.row_radius)
            painter.setBrush(QColor(palette_tokens.accent))
            painter.drawRoundedRect(
                QRectF(
                    metrics.marker_x,
                    metrics.marker_inset_y,
                    metrics.marker_width,
                    rect.height() - 2 * metrics.marker_inset_y,
                ),
                metrics.marker_radius,
                metrics.marker_radius,
            )
        elif self._hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(palette_tokens.panel))
            painter.drawRoundedRect(rect, metrics.row_radius, metrics.row_radius)

    def _list_widget(self) -> QListWidget | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QListWidget):
                return parent
            parent = parent.parentWidget()
        return None
