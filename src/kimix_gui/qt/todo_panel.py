"""Collapsible todo panel pinned to the transcript's top-right corner."""

from __future__ import annotations

from contextlib import suppress

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.qt.appearance import FONT_CHANGED
from kimix_gui.qt.components.disclosure import DISCLOSURE_COLLAPSED, DISCLOSURE_EXPANDED
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.styling import (
    FLASH,
    MODE,
    STATE,
    Role,
    repolish,
    set_style_property,
    style,
)
from kimix_gui.qt.theme import active_theme
from kimix_gui.todos import EMPTY_SNAPSHOT, TodoEntry, TodoSnapshot

# Panel geometry, from the design tokens. Bound at import because these feed
# ``setFixedHeight`` calls made once during construction.
_METRICS = DARK.todo_panel
MARGIN = _METRICS.margin
CARD_WIDTH = _METRICS.card_width
MIN_CARD_WIDTH = _METRICS.min_card_width
HEADER_HEIGHT = _METRICS.header_height
BAR_HEIGHT = _METRICS.bar_height
FOOTER_HEIGHT = _METRICS.footer_height
ROW_HEIGHT = _METRICS.row_height
ROW_WITH_NOTES_HEIGHT = _METRICS.row_with_notes_height
ROW_SPACING = _METRICS.row_spacing
BODY_PADDING = _METRICS.body_padding
MIN_BODY_HEIGHT = ROW_HEIGHT + 2 * BODY_PADDING
MAX_BODY_HEIGHT = _METRICS.max_body_height
INDENT_STEP = _METRICS.indent_step
FLASH_MS = DARK.motion.flash_ms

_GLYPHS = {"done": "✓", "in_progress": "▸", "pending": "○"}

# Anchor events that mean "the corner the panel sits in has moved". ``LayoutRequest``
# is in the set because a container can rearrange its children without resizing itself.
_HOST_MOVED = frozenset({QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest})


def status_color(status: str) -> QColor:
    """Return the feedback color for a todo ``status``, from the active theme.

    These are semantic states, not transcript record categories, so they resolve
    through ``Palette`` (``success`` / ``accent`` / ``muted``) rather than through
    ``CategoryPalette``. Resolved per call, so a theme swap repaints correctly.
    """

    palette = active_theme().palette
    if status == "done":
        return QColor(palette.success)
    if status == "in_progress":
        return QColor(palette.accent)
    return QColor(palette.muted)


class _ElidedLabel(QLabel):
    """Single-line label that elides its text and keeps the full text as tooltip."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._apply_elide()

    @property
    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._apply_elide()

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Re-place the ellipsis: how much fits changed, the width did not.

        ``_apply_elide`` measured with ``QFontMetrics`` but only ever ran on resize,
        so after a font change the cut stayed where the old metrics put it -- too
        early at a smaller size, and past the right edge at a larger one.
        """

        super().changeEvent(event)
        if event.type() in FONT_CHANGED:
            self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(0, self.width())
        if width <= 4:
            super().setText(self._full)
            return
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, width))


class _ClickableFrame(QFrame):
    """Frame that reports left-clicks, used as the panel header/pill."""

    clicked = Signal()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _ProgressStrip(QWidget):
    """Three-tone progress strip: done, in progress, remaining."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("todo-progress")
        self.setFixedHeight(BAR_HEIGHT)
        self._done = 0
        self._in_progress = 0
        self._total = 0

    def set_counts(self, done: int, in_progress: int, total: int) -> None:
        self._done = max(0, done)
        self._in_progress = max(0, in_progress)
        self._total = max(0, total)
        self.update()

    def paintEvent(self, _event: QEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor(active_theme().palette.border))
        if self._total <= 0 or width <= 0:
            return
        done_width = round(width * self._done / self._total)
        active_width = round(width * self._in_progress / self._total)
        if done_width > 0:
            painter.fillRect(0, 0, done_width, height, status_color("done"))
        if active_width > 0:
            painter.fillRect(
                done_width,
                0,
                min(active_width, max(0, width - done_width)),
                height,
                status_color("in_progress"),
            )


class _TodoRow(QFrame):
    """One todo line: status glyph, elided title, optional notes."""

    def __init__(self, entry: TodoEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("todo-row")
        self.setProperty(STATE, entry.status)
        self.setFixedHeight(ROW_WITH_NOTES_HEIGHT if entry.notes else ROW_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8 + entry.depth * INDENT_STEP, 2, 8, 2)
        layout.setSpacing(7)

        glyph = QLabel(_GLYPHS.get(entry.status, "○"))
        glyph.setObjectName("todo-glyph")
        style(glyph, role=Role.MARKER)
        glyph.setProperty(STATE, entry.status)
        glyph.setFixedWidth(12)
        glyph.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(glyph, 0, Qt.AlignmentFlag.AlignTop)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(1)
        title = _ElidedLabel(entry.title)
        title.setObjectName("todo-item-title")
        # The strike-through for a done entry is a style sheet declaration now; see
        # the note on that rule for why a hand-built QFont was the wrong tool.
        title.setProperty(STATE, entry.status)
        column.addWidget(title)
        if entry.notes:
            notes = _ElidedLabel(entry.notes)
            notes.setObjectName("todo-item-notes")
            style(notes, role=Role.FOOTNOTE)
            column.addWidget(notes)
        layout.addLayout(column, 1)

        tooltip = entry.title if not entry.notes else f"{entry.title}\n{entry.notes}"
        self.setToolTip(tooltip)


class TodoPanel(QFrame):
    """Floating, collapsible view of the session's todo list.

    The panel hides itself while a session has no todos, shows a compact pill
    when collapsed, and expands into a card with a progress strip, the indented
    todo tree, and a status summary.
    """

    toggled = Signal(bool)

    def __init__(self, host: QWidget | None = None) -> None:
        """Float inside ``host``, which is also the Qt parent.

        The panel is positioned by hand rather than by a layout, so it has to know
        when the area it floats in moves. It watches for that itself: the host used to
        be the one that knew, and forwarded three widget events and a scrollbar signal
        on the panel's behalf, which meant a second widget had to be taught the panel's
        anchoring rules to host it at all.
        """

        super().__init__(host)
        self.setObjectName("todo-panel")
        self.setProperty(MODE, "pill")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._snapshot = EMPTY_SNAPSHOT
        self._expanded = True
        self._user_collapsed = False
        self._rows: list[_TodoRow] = []
        self._active_row: _TodoRow | None = None
        self._i18n = Retranslator(self)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.setInterval(FLASH_MS)
        self._flash_timer.timeout.connect(lambda: self._set_flash(False))
        self._build()
        self._watch_host()
        self.setVisible(False)

    # ---- construction ------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)

        self._header = _ClickableFrame()
        self._header.setObjectName("todo-header")
        self._header.setFixedHeight(HEADER_HEIGHT)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(11, 0, 9, 0)
        header_layout.setSpacing(8)
        self._dot = QLabel("●")
        self._dot.setObjectName("todo-dot")
        style(self._dot, role=Role.MARKER)
        self._dot.setProperty(STATE, "pending")
        self._title = QLabel()
        self._title.setObjectName("todo-title")
        style(self._title, role=Role.OVERLINE)
        self._count = QLabel("0/0")
        self._count.setObjectName("todo-count")
        self._chevron = QLabel(DISCLOSURE_EXPANDED)
        self._chevron.setObjectName("todo-chevron")
        style(self._chevron, role=Role.FOOTNOTE)
        header_layout.addWidget(self._dot)
        header_layout.addWidget(self._title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._count)
        header_layout.addWidget(self._chevron)
        root.addWidget(self._header)

        self._strip = _ProgressStrip()
        root.addWidget(self._strip)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("todo-scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list = QWidget()
        self._list.setObjectName("todo-list")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(BODY_PADDING, BODY_PADDING, BODY_PADDING, BODY_PADDING)
        self._list_layout.setSpacing(ROW_SPACING)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list)
        root.addWidget(self._scroll, 1)

        self._footer = QLabel("")
        self._footer.setObjectName("todo-footer")
        style(self._footer, role=Role.FOOTNOTE)
        self._footer.setFixedHeight(FOOTER_HEIGHT)
        root.addWidget(self._footer)

        self._header.clicked.connect(self._on_header_clicked)
        # Every translatable string in this panel is derived from the snapshot, so the
        # re-runnable statement is the method that derives them. Binding it also seeds
        # the title, the footer and the tooltip, which is why none of them are given a
        # starting text above: one derivation, stated once.
        self._i18n.bind(self._sync_header)

    # ---- public API --------------------------------------------------------

    @property
    def snapshot(self) -> TodoSnapshot:
        return self._snapshot

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_snapshot(self, snapshot: TodoSnapshot) -> None:
        """Adopt a new todo snapshot, rebuilding the rows when it changed."""

        if snapshot == self._snapshot:
            return
        was_empty = self._snapshot.is_empty
        self._snapshot = snapshot
        self._rebuild_rows()
        self._sync_header()
        if snapshot.is_empty:
            self._flash_timer.stop()
            self._set_flash(False)
            self.setVisible(False)
            return
        if was_empty and not self._user_collapsed:
            self._expanded = True
        self.setVisible(True)
        self._apply_mode()
        if not self._expanded:
            self._set_flash(True)
            self._flash_timer.start()
        self.relayout()
        self.raise_()
        if self._expanded:
            self._scroll_to_active()

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._user_collapsed = not expanded
        if expanded:
            self._flash_timer.stop()
            self._set_flash(False)
        self._apply_mode()
        self._sync_header()
        self.relayout()
        self.raise_()
        if expanded:
            self._scroll_to_active()
        self.toggled.emit(expanded)

    def relayout(self) -> None:
        """Re-anchor the panel to the top-right corner of its host widget."""

        host = self.parentWidget()
        anchor = self._anchor()
        if host is None or anchor is None or self.isHidden():
            return
        rect = self._anchor_rect(host, anchor)
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if self._expanded:
            width = max(MIN_CARD_WIDTH, min(CARD_WIDTH, rect.width() - 2 * MARGIN))
            height = min(self._expanded_height(), max(HEADER_HEIGHT, rect.height() - 2 * MARGIN))
        else:
            width = min(self._pill_width(), max(MIN_CARD_WIDTH, rect.width() - 2 * MARGIN))
            height = HEADER_HEIGHT + 2
        left = max(rect.left(), rect.left() + rect.width() - MARGIN - width)
        self.setGeometry(QRect(left, rect.top() + MARGIN, width, height))

    # ---- internals ---------------------------------------------------------

    def _watch_host(self) -> None:
        """Follow the widget the panel is anchored to, so nobody has to push it."""

        anchor = self._anchor()
        if anchor is not None:
            anchor.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched is self._anchor() and event.type() in _HOST_MOVED:
            self.relayout()
        return super().eventFilter(watched, event)

    def _anchor(self) -> QWidget | None:
        """The widget whose corner the panel sits in.

        For a scroll area that is the viewport, not the host itself: the viewport is
        what is left once the scrollbars have taken their width, and Qt resizes it when
        one appears without resizing the host. Watching the host instead meant a
        scrollbar could narrow the usable area with no event to react to -- the panel
        then overlapped it until something else moved. Deriving the width by hand was
        the other half of that: a hidden ``QScrollBar`` still reports a width (measured
        at 100 while invisible against 10 while shown), so the arithmetic only worked
        because it asked ``isVisible()`` first.

        Any other widget can host the panel, and then it is the anchor itself: a plain
        widget has no scrollbars to leave room for.
        """

        host = self.parentWidget()
        if host is None:
            return None
        return host.viewport() if isinstance(host, QAbstractScrollArea) else host

    @staticmethod
    def _anchor_rect(host: QWidget, anchor: QWidget) -> QRect:
        """The anchor's area, in the coordinates the panel's geometry is set in.

        The panel is a child of the host, so a viewport anchor has to be offset by
        where it sits inside it -- which is also what makes a framed host come out
        right. ``mapTo`` covers both cases: mapping a widget to itself returns the
        point unchanged, so a plain host needs no branch here.
        """

        return QRect(anchor.mapTo(host, QPoint(0, 0)), anchor.size())

    def _pill_width(self) -> int:
        return max(132, self._header.sizeHint().width() + 12)

    def _expanded_height(self) -> int:
        content = 2 * BODY_PADDING
        for index, row in enumerate(self._rows):
            content += row.height()
            if index:
                content += ROW_SPACING
        body = max(MIN_BODY_HEIGHT, min(MAX_BODY_HEIGHT, content))
        return HEADER_HEIGHT + BAR_HEIGHT + body + FOOTER_HEIGHT + 2

    def _apply_mode(self) -> None:
        self.setProperty(MODE, "card" if self._expanded else "pill")
        self._strip.setVisible(self._expanded)
        self._scroll.setVisible(self._expanded)
        self._footer.setVisible(self._expanded)
        self._chevron.setText(DISCLOSURE_EXPANDED if self._expanded else DISCLOSURE_COLLAPSED)
        repolish(self)

    def _set_flash(self, active: bool) -> None:
        set_style_property(self, FLASH, active)

    def _on_header_clicked(self) -> None:
        self.toggle()

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._list_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        self._active_row = None

    def _rebuild_rows(self) -> None:
        self._clear_rows()
        active = self._snapshot.active
        for index, entry in enumerate(self._snapshot.entries):
            row = _TodoRow(entry, self._list)
            self._list_layout.insertWidget(index, row)
            self._rows.append(row)
            if active is not None and entry == active and self._active_row is None:
                self._active_row = row

    def _sync_header(self) -> None:
        snapshot = self._snapshot
        self._count.setText(f"{snapshot.done}/{snapshot.total}")
        self._strip.set_counts(snapshot.done, snapshot.in_progress, snapshot.total)
        state = "pending"
        if snapshot.all_done:
            state = "done"
        elif snapshot.in_progress:
            state = "in_progress"
        set_style_property(self._dot, STATE, state)
        self._title.setText(self._title_text(expanded=self._expanded))
        self._footer.setText(self._summary_text())
        self._header.setToolTip(self._header_tooltip())

    def _title_text(self, *, expanded: bool) -> str:
        """One msgid, cased here.

        The expanded header is an overline and reads as caps; the pill does not. That
        was two separate translatable strings (``TODOS`` and ``Todos``) that a
        translator had to keep in sync for a difference no other language even has --
        Qt style sheets have no ``text-transform``, so the case has to happen in code,
        and here it happens after the lookup instead of inside it.
        """

        label = self.tr("Todos")
        return label.upper() if expanded else label

    def _summary_text(self) -> str:
        snapshot = self._snapshot
        if snapshot.is_empty:
            return self.tr("No todos")
        parts: list[str] = []
        # ``{count}`` rather than ``%n``: an uninstalled catalog falls back to the
        # msgid, and these counts read the same for one or many in both languages.
        if snapshot.in_progress:
            parts.append(self.tr("{count} in progress").format(count=snapshot.in_progress))
        if snapshot.pending:
            parts.append(self.tr("{count} pending").format(count=snapshot.pending))
        if snapshot.done:
            parts.append(self.tr("{count} done").format(count=snapshot.done))
        if snapshot.archived:
            parts.append(self.tr("{count} archived").format(count=snapshot.archived))
        return " · ".join(parts) or self.tr("No todos")

    def _header_tooltip(self) -> str:
        snapshot = self._snapshot
        if snapshot.is_empty:
            return self.tr("No todos yet")
        hint = self.tr("Click to collapse") if self._expanded else self.tr("Click to expand")
        active = snapshot.active
        if active is None:
            # Only the separators are interpolated; every word is already localized,
            # so these f-strings hold no translatable text. The tr() calls stay out
            # of the f-string braces because lupdate does not look inside them.
            all_done = self.tr("All {count} todos done").format(count=snapshot.total)
            return f"{all_done} · {hint}"
        label = self.tr("Now") if active.status == "in_progress" else self.tr("Next")
        return f"{label}: {active.title}\n{self._summary_text()} · {hint}"

    def _scroll_to_active(self) -> None:
        if self._active_row is None or not self._scroll.isVisible():
            return
        QTimer.singleShot(0, self._ensure_active_visible)

    def _ensure_active_visible(self) -> None:
        row = self._active_row
        if row is None:
            return
        with suppress(RuntimeError):
            self._scroll.ensureWidgetVisible(row, 0, 12)
