"""Virtualized chat transcript: list model + custom delegate, no per-row widgets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from PySide6.QtCore import (
    QEvent,
    QModelIndex,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QListView,
    QMenu,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from kimix_gui.qt.appearance import APPEARANCE_CHANGED
from kimix_gui.qt.labels import translate_transcript_text
from kimix_gui.qt.paint import qcolor
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.theme import active_theme
from kimix_gui.qt.transcript_cards import (
    CARD_MARGIN_Y,
    HEADER_HEIGHT,
    CardBodies,
    bar_hit_rect,
    bar_rect,
    body_rect,
    body_width,
    card_rect,
    copy_rect,
    disclosure_rect,
    header_rect,
    header_text_rect,
    layout_for,
    status_rect,
)
from kimix_gui.qt.transcript_model import (
    MAX_TRANSCRIPT_CHARS,
    BodySelection,
    RecordId,
    TranscriptModel,
    TranscriptRecord,
)
from kimix_gui.transcript_data import (
    HistoryEntry,
    TranscriptMutation,
    entry_copy_text,
    is_dialogue_entry,
)
from kimix_gui.transcript_layout import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_PENDING,
    record_label,
)

_PRESENTATION_OVERSCAN_VIEWPORTS = 3
_MIN_ACTIVE_PRESENTATION_RECORDS = 24
_MAX_ACTIVE_PRESENTATION_RECORDS = 96
_SCROLL_FRAME_INTERVAL_MS = 16


def _centered_icon_rect(rect: QRect, scale: float = 0.55) -> QRectF:
    size = min(rect.width(), rect.height()) * scale
    center = rect.center()
    return QRectF(center.x() - size / 2, center.y() - size / 2, size, size)


def _paint_status_icon(painter: QPainter, rect: QRect, status: str, category: str) -> None:
    """Paint semantic activity state without relying on a font glyph."""

    palette = active_theme().palette
    color = QColor(
        palette.success
        if status == STATUS_OK
        else palette.error
        if status == STATUS_ERROR
        else qcolor(category)
    )
    box = _centered_icon_rect(rect)
    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(color)
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    if status == STATUS_PENDING:
        painter.drawArc(box, 35 * 16, 285 * 16)
    else:
        painter.drawEllipse(box)
    if status == STATUS_OK:
        painter.drawLine(
            QPointF(box.left() + box.width() * 0.22, box.top() + box.height() * 0.53),
            QPointF(box.left() + box.width() * 0.43, box.top() + box.height() * 0.72),
        )
        painter.drawLine(
            QPointF(box.left() + box.width() * 0.43, box.top() + box.height() * 0.72),
            QPointF(box.left() + box.width() * 0.78, box.top() + box.height() * 0.32),
        )
    elif status == STATUS_ERROR:
        center_x = box.center().x()
        painter.drawLine(
            QPointF(center_x, box.top() + box.height() * 0.25),
            QPointF(center_x, box.top() + box.height() * 0.57),
        )
        dot = box.width() * 0.07
        painter.setBrush(color)
        painter.drawEllipse(
            QRectF(center_x - dot, box.top() + box.height() * 0.72 - dot, dot * 2, dot * 2)
        )
    painter.restore()


def _paint_disclosure_icon(painter: QPainter, rect: QRect, *, expanded: bool) -> None:
    """Paint a small chevron whose direction mirrors the disclosure state."""

    center = rect.center()
    half_width = rect.width() * 0.16
    half_height = rect.height() * 0.11
    outer_y = center.y() + half_height if expanded else center.y() - half_height
    inner_y = center.y() - half_height if expanded else center.y() + half_height
    pen = QPen(QColor(active_theme().palette.muted))
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.save()
    painter.setPen(pen)
    painter.drawLine(QPointF(center.x() - half_width, outer_y), QPointF(center.x(), inner_y))
    painter.drawLine(QPointF(center.x(), inner_y), QPointF(center.x() + half_width, outer_y))
    painter.restore()


def _paint_copy_icon(painter: QPainter, rect: QRect) -> None:
    """Paint the copy affordance as chrome rather than a Unicode character."""

    box = _centered_icon_rect(rect, 0.5)
    unit = box.width() * 0.18
    back = QRectF(box.left() + unit, box.top(), box.width() - unit, box.height() - unit)
    front = QRectF(box.left(), box.top() + unit, box.width() - unit, box.height() - unit)
    pen = QPen(QColor(active_theme().palette.muted))
    pen.setWidthF(1.25)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(pen)
    radius = unit * 0.7
    painter.drawRoundedRect(back, radius, radius)
    painter.drawRoundedRect(front, radius, radius)
    painter.restore()


class TranscriptDelegate(QStyledItemDelegate):
    """Paints one message card and answers where the pointer landed in it.

    Geometry, layout and measurement belong to ``transcript_cards``; what is left here
    is the Qt item-delegate protocol and the brush work.
    """

    def __init__(self, view: Transcript) -> None:
        super().__init__(view)
        self._view = view
        self.bodies = CardBodies(self, font=view.font)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return QSize(option.rect.width(), HEADER_HEIGHT)
        width = max(120, option.rect.width() or self._view.viewport().width())
        return QSize(width, self.bodies.row_height(record, width))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return
        rect = card_rect(option.rect)
        layout = layout_for(record, option.rect.width())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        hovered = self._view.hovered_bar_row == index.row()
        palette = active_theme().palette
        dialogue = is_dialogue_entry(record.entry)
        if layout.status and record.expanded:
            fill = palette.surface
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(fill))
            radius = active_theme().radius.md
            painter.drawRoundedRect(QRectF(rect), radius, radius)
        # "muted" is the layout's way of saying *uncategorized*, so chrome roles answer
        # for it: an uncategorized row brightens to plain text ink under the cursor,
        # and its label stays dim instead of borrowing a record hue.
        uncategorized = layout.bar_color == "muted"
        bar_color = QColor(palette.text) if hovered and uncategorized else qcolor(layout.bar_color)
        painter.fillRect(bar_rect(rect, hovered=hovered), bar_color)
        if layout.status:
            _paint_status_icon(painter, status_rect(rect), layout.status, layout.bar_color)
        header_band = header_text_rect(
            rect,
            has_status=bool(layout.status),
            has_disclosure=not dialogue,
        )
        if layout.status == STATUS_OK and not uncategorized:
            # Completion is already communicated by the green status icon. Keep the
            # action itself tied to its family (Read cyan, Todo yellow, and so on),
            # while the object/outcome summary remains neutral beside it.
            label_color = qcolor(layout.bar_color)
        elif layout.status:
            label_color = QColor(palette.text)
        elif uncategorized:
            label_color = QColor(palette.muted)
        else:
            label_color = qcolor(layout.bar_color)
        painter.setPen(label_color)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        label_width = min(
            header_band.width(), painter.fontMetrics().horizontalAdvance(layout.label)
        )
        label_band = QRect(header_band.left(), header_band.top(), label_width, header_band.height())
        painter.drawText(
            label_band,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            layout.label,
        )
        font.setBold(False)
        painter.setFont(font)
        if layout.summary:
            gap = active_theme().spacing.sm
            summary_left = label_band.right() + 1 + gap
            summary_band = QRect(
                summary_left,
                header_band.top(),
                max(0, header_band.right() - summary_left + 1),
                header_band.height(),
            )
            painter.setPen(QColor(palette.muted))
            painter.drawText(
                summary_band,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                layout.summary,
            )
        if not dialogue:
            _paint_disclosure_icon(painter, disclosure_rect(rect), expanded=not layout.compact)
        if self._view.hovered_row == index.row():
            _paint_copy_icon(painter, copy_rect(rect))
        if not layout.compact and layout.body:
            body_band = body_rect(rect)
            document = self.bodies.document(record, body_width(option.rect.width()), layout)
            painter.translate(body_band.topLeft())
            painter.setClipRect(QRect(0, 0, body_band.width(), body_band.height()))
            ctx = QAbstractTextDocumentLayout.PaintContext()
            primary_body = is_dialogue_entry(record.entry) or bool(layout.status)
            color = QColor(palette.text if primary_body else palette.muted)
            ctx.palette.setColor(QPalette.ColorRole.Text, color)
            ctx.palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
            ctx.palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 0))
            selection = self._view.body_selection
            if selection is not None and selection.row == index.row() and not selection.is_empty:
                extra = QAbstractTextDocumentLayout.Selection()
                cursor = QTextCursor(document)
                start, end = selection.normalized()
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                extra.cursor = cursor
                fmt = QTextCharFormat()
                fmt.setBackground(QColor(palette.accent))
                fmt.setForeground(QColor(palette.on_accent))
                extra.format = fmt
                ctx.selections = [extra]
            document.documentLayout().draw(painter, ctx)
        painter.restore()

    def copy_hit(self, option_rect: QRect, pos: QPoint) -> bool:
        return copy_rect(card_rect(option_rect)).contains(pos)

    def header_hit(self, option_rect: QRect, pos: QPoint) -> bool:
        return header_rect(card_rect(option_rect)).contains(pos) and not self.copy_hit(
            option_rect, pos
        )

    def bar_hit(self, option_rect: QRect, pos: QPoint) -> bool:
        """Whether ``pos`` is on the left gutter, which toggles the row."""

        return bar_hit_rect(card_rect(option_rect)).contains(pos)

    def body_hit(
        self,
        option_rect: QRect,
        pos: QPoint,
        index: QModelIndex,
        *,
        clamp: bool = False,
    ) -> int | None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return None
        layout = layout_for(record, option_rect.width())
        if layout.compact or not layout.body:
            return None
        body_band = body_rect(card_rect(option_rect))
        if not body_band.contains(pos):
            if (
                not clamp
                or not body_band.isValid()
                or body_band.width() <= 0
                or body_band.height() <= 0
            ):
                return None
            pos = QPoint(
                min(max(pos.x(), body_band.left()), body_band.right() - 1),
                min(max(pos.y(), body_band.top()), body_band.bottom() - 1),
            )
        document = self.bodies.document(record, body_width(option_rect.width()), layout)
        local = QPointF(pos.x() - body_band.left(), pos.y() - body_band.top())
        hit = document.documentLayout().hitTest(local, Qt.HitTestAccuracy.FuzzyHit)
        if hit < 0:
            return None
        return min(hit, max(0, document.characterCount() - 1))

    def document_for(self, option_rect: QRect, index: QModelIndex) -> QTextDocument | None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return None
        layout = layout_for(record, option_rect.width())
        if layout.compact or not layout.body:
            return None
        return self.bodies.document(record, body_width(option_rect.width()), layout)


class Transcript(QListView):
    """Scrollable chat log that virtualizes painting and bounds memory."""

    reached_top = Signal()
    reached_bottom = Signal()
    viewport_turn_changed = Signal(object)
    record_copied = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        max_chars: int = MAX_TRANSCRIPT_CHARS,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("transcript")
        self._model = TranscriptModel(self, max_chars=max_chars)
        self._delegate = TranscriptDelegate(self)
        self.setModel(self._model)
        self.setItemDelegate(self._delegate)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSpacing(2)
        self.setUniformItemSizes(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._stick_to_bottom = True
        self._top_event_armed = True
        self._bottom_event_armed = True
        self._last_viewport_turn: int | None = None
        self._wrap_width = 0
        self._selection: BodySelection | None = None
        self._selecting = False
        self._hovered_row: int | None = None
        self._hovered_bar_row: int | None = None
        self._history_at_latest = True
        self._shifting_window = False
        self._pending_scroll_value: int | None = None
        self._scroll_processing_suspended = 0
        self._scroll_frame = QTimer(self)
        self._scroll_frame.setSingleShot(True)
        self._scroll_frame.setInterval(_SCROLL_FRAME_INTERVAL_MS)
        self._scroll_frame.timeout.connect(self._flush_scroll)
        self.verticalScrollBar().valueChanged.connect(self._queue_scroll)
        self.viewport().installEventFilter(self)
        self._i18n = Retranslator(self)
        self._i18n.bind(self._repaint_labels)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Drop every cached document when the font or the theme moves.

        The delegate caches measured heights and ``QTextDocument`` bodies by stable
        record revision and width. Font is intentionally outside those keys, and the
        theme accent is baked into markdown links. A font change would otherwise leave
        rows measured for the old size, while a theme change would leave stale links.

        Qt delivers all three of these to every widget, so there is nothing to
        subscribe to and no manager to keep in sync. ``StyleChange`` is the one that
        stands for "theme reapplied": ``apply_theme`` always calls
        ``setStyleSheet``, and Qt sends ``StyleChange`` even when the new sheet is
        byte-identical to the old one.

        Dropping the cache is the whole fix. An explicit
        ``scheduleDelayedItemsLayout()`` was here and was removed: with it gone the
        view's item rectangles still move, so it could not be shown to do anything.
        ``QListView`` gets there by itself, whether from the base ``changeEvent`` or
        from its own lazy relayout was not isolated.
        """

        super().changeEvent(event)
        if event.type() in APPEARANCE_CHANGED:
            self.bodies.invalidate()

    def _repaint_labels(self) -> None:
        """Make Qt repaint the record labels the delegate translates.

        The only translated copy in this view -- "TOOL", "ERROR", the tool family
        names -- is painted straight to the viewport rather than held in a widget, so
        there is no text to re-set. What a language change needs here is for the
        paint to happen again, which is why the bound statement invalidates instead
        of assigning. Qt does not repaint on ``LanguageChange`` by itself.
        """

        self.bodies.invalidate()
        self.viewport().update()

    @property
    def records(self) -> list[TranscriptRecord]:
        return self._model.records

    @property
    def hovered_bar_row(self) -> int | None:
        """Row whose left gutter currently holds the pointer, if any."""

        return self._hovered_bar_row

    @property
    def hovered_row(self) -> int | None:
        """Transcript row currently under the pointer, if any."""

        return self._hovered_row

    @property
    def body_selection(self) -> BodySelection | None:
        """The in-body text selection the delegate has to highlight, if any.

        Read-only on purpose: dragging owns the anchor. It is a property rather than
        a private attribute the delegate reaches into because painting a selection is
        the one thing the delegate needs from the view's interaction state.
        """

        return self._selection

    @property
    def omitted_records(self) -> int:
        return self._model.omitted_records

    @property
    def history_window(self) -> tuple[int, int] | None:
        return self._model.history_window

    @property
    def has_hidden_older_history(self) -> bool:
        return self._model.has_hidden_older_history

    @property
    def has_hidden_newer_history(self) -> bool:
        return self._model.has_hidden_newer_history

    @property
    def pinned_to_latest(self) -> bool:
        return self._stick_to_bottom

    @property
    def bodies(self) -> CardBodies:
        """The laid-out bodies of the rows shown so far, measured and cached."""

        return self._delegate.bodies

    def _wrap_cells(self) -> int:
        """Viewport width in character cells, the granularity a reflow is worth at.

        Only used to decide whether a resize changed anything: the cached bodies are
        keyed by pixel width, so dropping them on every pixel of a drag would relay
        out the whole viewport dozens of times per second.
        """

        return max(8, self.viewport().width() // 8 or 80)

    def _presentation_record_limit(self) -> int:
        """Rows covering three compact viewports, capped before layout becomes eager."""

        item_extent = HEADER_HEIGHT + CARD_MARGIN_Y * 2 + self.spacing() * 2
        viewport_rows = (max(1, self.viewport().height()) + item_extent - 1) // item_extent
        return max(
            _MIN_ACTIVE_PRESENTATION_RECORDS,
            min(
                _MAX_ACTIVE_PRESENTATION_RECORDS,
                viewport_rows * _PRESENTATION_OVERSCAN_VIEWPORTS,
            ),
        )

    def _sync_presentation_record_limit(self) -> bool:
        return self._model.set_presentation_record_limit(self._presentation_record_limit())

    def mark_history_window(self, start: int | None = None, end: int | None = None) -> None:
        self._model.mark_history_window(start, end)

    def apply_mutation(self, mutation: TranscriptMutation) -> None:
        """Apply one typed live update; the AST remains the semantic input."""

        self._model.apply_mutation(mutation)
        self._maybe_scroll_end()
        self._notify_viewport_turn()

    def apply_mutations(self, mutations: Sequence[TranscriptMutation]) -> None:
        self._model.apply_mutations(mutations)
        self._maybe_scroll_end()
        self._notify_viewport_turn()

    def replace_history(
        self,
        entries: Sequence[HistoryEntry],
        *,
        target_turn: int | None = None,
        pin_latest: bool = False,
    ) -> None:
        self._clear_selection()
        self._set_hovered_row(None)
        self._set_hovered_bar_row(None)
        self._sync_presentation_record_limit()
        with self._suspend_scroll_processing():
            self._model.replace_history(
                entries,
                target_turn=target_turn,
                pin_latest=pin_latest,
            )
            self._history_at_latest = pin_latest
            self._stick_to_bottom = pin_latest
            self._top_event_armed = False
            self._bottom_event_armed = False
            self.verticalScrollBar().setValue(0)
        self._notify_viewport_turn()

    def clear_messages(self) -> None:
        self.bodies.invalidate()
        self._set_hovered_row(None)
        self._set_hovered_bar_row(None)
        with self._suspend_scroll_processing():
            self._model.clear_messages()
        self._history_at_latest = True
        self._stick_to_bottom = True
        self._top_event_armed = True
        self._bottom_event_armed = True

    def jump_to_latest(self) -> None:
        self._clear_selection()
        self._set_hovered_row(None)
        self._set_hovered_bar_row(None)
        self._sync_presentation_record_limit()
        with self._suspend_scroll_processing():
            self._model.reveal_turn(0, pin_latest=True)
            self._history_at_latest = True
            self._stick_to_bottom = True
            self._top_event_armed = True
            self._bottom_event_armed = False
            self._maybe_scroll_end()
        self._notify_viewport_turn()

    def jump_to_turn(self, turn: int) -> None:
        self._clear_selection()
        self._set_hovered_row(None)
        self._set_hovered_bar_row(None)
        self._sync_presentation_record_limit()
        with self._suspend_scroll_processing():
            self._model.reveal_turn(turn)
            self.doItemsLayout()
            start = self._model._history_start or 0
            end = self._model._history_end or len(self.records)
            target_index = None
            for index in range(start, end):
                if self.records[index].turn == turn:
                    target_index = index
                    break
            if target_index is None:
                return
            self._history_at_latest = False
            self._stick_to_bottom = False
            self._top_event_armed = False
            self._bottom_event_armed = False
            self.scrollTo(
                self._model.index(target_index),
                QAbstractItemView.ScrollHint.PositionAtTop,
            )
            self._stick_to_bottom = False
            self._top_event_armed = True
            self._bottom_event_armed = True
        self._notify_viewport_turn()

    def _reveal_local_history(self, *, older: bool) -> bool:
        """Shift the bounded Qt slice and keep the edge record visually anchored."""

        self._sync_presentation_record_limit()
        anchor = self._viewport_anchor(from_bottom=not older)
        self._shifting_window = True
        try:
            changed = (
                self._model.reveal_older_history() if older else self._model.reveal_newer_history()
            )
            if not changed:
                return False
            self._clear_selection()
            self._set_hovered_row(None)
            self._set_hovered_bar_row(None)
            self.doItemsLayout()
            if anchor is not None:
                self._restore_viewport_anchor(anchor, from_bottom=not older)
            self._stick_to_bottom = False
            self._top_event_armed = True
            self._bottom_event_armed = True
            self._notify_viewport_turn()
            return True
        finally:
            self._shifting_window = False

    def _viewport_anchor(self, *, from_bottom: bool) -> tuple[RecordId, int] | None:
        viewport = self.viewport().rect()
        index = QModelIndex()
        for distance in range(2, HEADER_HEIGHT + 8):
            y = viewport.bottom() - distance if from_bottom else viewport.top() + distance
            index = self.indexAt(QPoint(max(12, viewport.center().x()), y))
            if index.isValid():
                break
        if not index.isValid():
            return None
        rect = self.visualRect(index)
        offset = rect.bottom() - viewport.bottom() if from_bottom else rect.top() - viewport.top()
        return self.records[index.row()].record_id, offset

    def _restore_viewport_anchor(
        self,
        anchor: tuple[RecordId, int],
        *,
        from_bottom: bool,
    ) -> None:
        record_id, offset = anchor
        row = self._model.row_for_record(record_id)
        if row is None:
            return
        index = self._model.index(row)
        hint = (
            QAbstractItemView.ScrollHint.PositionAtBottom
            if from_bottom
            else QAbstractItemView.ScrollHint.PositionAtTop
        )
        self.scrollTo(index, hint)
        rect = self.visualRect(index)
        viewport = self.viewport().rect()
        current = rect.bottom() if from_bottom else rect.top()
        wanted = (viewport.bottom() if from_bottom else viewport.top()) + offset
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() + current - wanted)

    def viewport_turn(self) -> int | None:
        index = self.indexAt(QPoint(12, 4))
        if not index.isValid():
            return None
        record = self.records[index.row()]
        return record.turn

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is self.viewport():
            if isinstance(event, QMouseEvent):
                handled = self._handle_mouse_event(event)
                if handled:
                    return True
            if event.type() == QEvent.Type.Leave:
                self.viewport().unsetCursor()
                self._set_hovered_row(None)
                self._set_hovered_bar_row(None)
        return super().eventFilter(watched, event)  # type: ignore[arg-type]

    def keyPressEvent(self, event: object) -> None:
        if (
            isinstance(event, QKeyEvent)
            and event.matches(QKeySequence.StandardKey.Copy)
            and self._copy_selection()
        ):
            event.accept()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def wheelEvent(self, event: object) -> None:
        """Request another history window when the scrollbar cannot move further."""

        if not isinstance(event, QWheelEvent):
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        pixel_delta = event.pixelDelta().y()
        delta = pixel_delta if pixel_delta else event.angleDelta().y()
        super().wheelEvent(event)
        bar = self.verticalScrollBar()
        if delta > 0 and bar.value() <= 0:
            self._cancel_pending_scroll()
            self._activate_top_edge()
        elif delta < 0 and bar.value() >= bar.maximum():
            self._cancel_pending_scroll()
            self._activate_bottom_edge()

    def _handle_mouse_event(self, event: QMouseEvent) -> bool:
        etype = event.type()
        if etype == QEvent.Type.MouseButtonDblClick:
            return self._handle_mouse_double_click(event)
        if etype == QEvent.Type.MouseMove:
            return self._handle_mouse_move(event)
        if etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._selecting:
                self._selecting = False
                return True
            return False
        if etype == QEvent.Type.MouseButtonPress:
            return self._handle_mouse_press(event)
        return False

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        if not index.isValid():
            self._clear_selection()
            self._set_hovered_row(None)
            return False
        self._set_hovered_row(index.row())
        record = self.records[index.row()]
        option_rect = self.visualRect(index)
        copy_clicked = self._delegate.copy_hit(option_rect, pos)
        header = self._delegate.header_hit(option_rect, pos)
        bar = self._delegate.bar_hit(option_rect, pos)
        if event.button() == Qt.MouseButton.RightButton:
            if copy_clicked:
                self._copy_record(record)
                return True
            if self.selected_text():
                self._popup_copy_menu(event.globalPosition().toPoint())
            return True
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if copy_clicked:
            self._copy_record(record)
            return True
        if not is_dialogue_entry(record.entry) and (not record.expanded or header or bar):
            self._model.toggle_expanded(index.row())
            self.updateGeometries()
            self._maybe_scroll_end()
            self._clear_selection()
            return True
        hit = self._delegate.body_hit(option_rect, pos, index)
        if hit is None:
            self._clear_selection()
            return False
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._selection = BodySelection(index.row(), hit, hit)
        self._selecting = True
        self.viewport().update()
        return True

    def _handle_mouse_double_click(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        if not index.isValid():
            return False
        self._set_hovered_row(index.row())
        option_rect = self.visualRect(index)
        if self._delegate.copy_hit(option_rect, pos):
            return True
        hit = self._delegate.body_hit(option_rect, pos, index)
        document = self._delegate.document_for(option_rect, index)
        if hit is None or document is None:
            return False
        cursor = QTextCursor(document)
        cursor.setPosition(min(hit, max(0, document.characterCount() - 1)))
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        self._selection = BodySelection(
            index.row(),
            cursor.selectionStart(),
            cursor.selectionEnd(),
        )
        self._selecting = False
        self.viewport().update()
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        pos = event.position().toPoint()
        if self._selecting and self._selection is not None:
            hovered = self.indexAt(pos)
            self._set_hovered_row(hovered.row() if hovered.isValid() else None)
            index = self._model.index(self._selection.row)
            if index.isValid():
                hit = self._delegate.body_hit(self.visualRect(index), pos, index, clamp=True)
                if hit is not None:
                    self._selection.position = hit
                    self.viewport().update()
            return True
        self._sync_cursor(pos)
        return False

    def _sync_cursor(self, pos: QPoint) -> None:
        index = self.indexAt(pos)
        if not index.isValid():
            self.viewport().unsetCursor()
            self._set_hovered_row(None)
            self._set_hovered_bar_row(None)
            return
        self._set_hovered_row(index.row())
        option_rect = self.visualRect(index)
        record = self.records[index.row()]
        on_bar = not is_dialogue_entry(record.entry) and self._delegate.bar_hit(option_rect, pos)
        self._set_hovered_bar_row(index.row() if on_bar else None)
        if self._delegate.copy_hit(option_rect, pos):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            return
        if not is_dialogue_entry(record.entry) and (
            on_bar or self._delegate.header_hit(option_rect, pos)
        ):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            return
        if self._delegate.body_hit(option_rect, pos, index) is not None:
            self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
            return
        self.viewport().unsetCursor()

    def _set_hovered_row(self, row: int | None) -> None:
        if row == self._hovered_row:
            return
        previous = self._hovered_row
        self._hovered_row = row
        for candidate in (previous, row):
            if candidate is not None and 0 <= candidate < len(self.records):
                self.viewport().update(self.visualRect(self._model.index(candidate)))

    def _set_hovered_bar_row(self, row: int | None) -> None:
        if row == self._hovered_bar_row:
            return
        previous = self._hovered_bar_row
        self._hovered_bar_row = row
        for candidate in (previous, row):
            if candidate is not None and 0 <= candidate < len(self.records):
                self.viewport().update(self.visualRect(self._model.index(candidate)))

    def selected_text(self) -> str:
        selection = self._selection
        if selection is None or selection.is_empty:
            return ""
        index = self._model.index(selection.row)
        if not index.isValid():
            return ""
        document = self._delegate.document_for(self.visualRect(index), index)
        if document is None:
            return ""
        start, end = selection.normalized()
        limit = max(0, document.characterCount() - 1)
        start = max(0, min(start, limit))
        end = max(0, min(end, limit))
        cursor = QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText().replace("\u2029", "\n")

    def _copy_selection(self) -> bool:
        text = self.selected_text()
        if not text:
            return False
        QApplication.clipboard().setText(text)
        return True

    def _clear_selection(self) -> None:
        self._selecting = False
        if self._selection is not None:
            self._selection = None
            self.viewport().update()

    def _popup_copy_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        action = menu.addAction(self.tr("Copy"))
        action.triggered.connect(self._copy_selection)
        menu.popup(global_pos)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        presentation_changed = self._sync_presentation_record_limit()
        if presentation_changed and self._model.has_history_source:
            self._refit_history_projection()
        cells = self._wrap_cells()
        if cells != self._wrap_width:
            self._wrap_width = cells
            self.bodies.invalidate()
            self.scheduleDelayedItemsLayout()
            self._maybe_scroll_end()

    def _refit_history_projection(self) -> None:
        """Resize the exact row window while preserving the reader's pixel anchor."""

        from_bottom = self._stick_to_bottom
        anchor = self._viewport_anchor(from_bottom=from_bottom)
        turn = self.viewport_turn()
        if turn is None:
            start = self._model._history_start or 0
            if start < len(self.records):
                turn = self.records[start].turn
        self._shifting_window = True
        try:
            with self._suspend_scroll_processing():
                if from_bottom:
                    self._model.reveal_turn(0, pin_latest=True)
                elif turn is not None:
                    self._model.reveal_turn(turn)
                self.doItemsLayout()
                if anchor is not None:
                    self._restore_viewport_anchor(anchor, from_bottom=from_bottom)
                elif from_bottom:
                    self.scrollToBottom()
        finally:
            self._shifting_window = False

    def _copy_record(self, record: TranscriptRecord) -> None:
        QApplication.clipboard().setText(entry_copy_text(record.entry, translate_transcript_text))
        self.record_copied.emit(translate_transcript_text(record_label(record.entry)))

    def _maybe_scroll_end(self) -> None:
        if not self._stick_to_bottom:
            return
        self.scrollToBottom()

    @contextmanager
    def _suspend_scroll_processing(self) -> Iterator[None]:
        self._scroll_processing_suspended += 1
        self._cancel_pending_scroll()
        try:
            yield
        finally:
            self._scroll_processing_suspended -= 1

    def _cancel_pending_scroll(self) -> None:
        self._pending_scroll_value = None
        self._scroll_frame.stop()

    def _queue_scroll(self, value: int) -> None:
        """Keep only the final scrollbar position until the next event-loop frame."""

        if self._shifting_window or self._scroll_processing_suspended:
            return
        self._pending_scroll_value = value
        if not self._scroll_frame.isActive():
            self._scroll_frame.start()

    def _flush_scroll(self) -> None:
        self._scroll_frame.stop()
        value = self._pending_scroll_value
        self._pending_scroll_value = None
        if value is None or self._shifting_window or self._scroll_processing_suspended:
            return
        self._process_scroll_value(value)

    def _process_scroll_value(self, value: int) -> None:
        if self._shifting_window:
            return
        bar = self.verticalScrollBar()
        at_top = value <= 0
        at_bottom = value >= bar.maximum()
        if (
            at_bottom
            and value > 0
            and self._history_at_latest
            and not self._model.has_hidden_newer_history
        ):
            self._stick_to_bottom = True
        elif not at_bottom:
            self._stick_to_bottom = False
        if at_top:
            if self._top_event_armed and not at_bottom and self._activate_top_edge():
                return
        else:
            self._top_event_armed = True
        if at_bottom:
            if self._bottom_event_armed and not at_top and self._activate_bottom_edge():
                return
        else:
            self._bottom_event_armed = True
        self._notify_viewport_turn()

    def _activate_top_edge(self) -> bool:
        if not self._top_event_armed:
            return False
        self._top_event_armed = False
        if self._reveal_local_history(older=True):
            return True
        self.reached_top.emit()
        return True

    def _activate_bottom_edge(self) -> bool:
        if not self._bottom_event_armed:
            return False
        self._bottom_event_armed = False
        if self._reveal_local_history(older=False):
            return True
        self.reached_bottom.emit()
        return True

    def _notify_viewport_turn(self) -> None:
        turn = self.viewport_turn()
        if turn != self._last_viewport_turn:
            self._last_viewport_turn = turn
            self.viewport_turn_changed.emit(turn)

    def visible_text(self) -> str:
        """Plain text of the rows currently in the viewport.

        Laid out through ``layout_for``, the same function the painter uses, so what
        this reports is what is on screen rather than a second opinion about it.
        """

        lines: list[str] = []
        viewport = self.viewport().rect()
        width = self.viewport().width()
        for row in range(len(self.records)):
            rect = self.visualRect(self._model.index(row))
            if not rect.intersects(viewport):
                continue
            layout = layout_for(self.records[row], width)
            lines.append(layout.header)
            if layout.body:
                lines.append(layout.body)
        return "\n".join(lines)
