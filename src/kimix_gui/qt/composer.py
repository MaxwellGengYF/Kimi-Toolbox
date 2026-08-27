"""Chat composer and an expanded pad for long prompts."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.qt.appearance import FONT_CHANGED
from kimix_gui.qt.components import Card
from kimix_gui.qt.retranslate import Retranslator
from kimix_gui.qt.styling import CardLevel, Metric, Role, Variant, style


class _GlyphButton(QPushButton):
    """Flat icon button used inside the composer and pad chrome."""

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(22, 22)

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)  # type: ignore[arg-type]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self.palette().color(self.foregroundRole()))
        if not self.underMouse():
            color.setAlpha(180)
        painter.setPen(QPen(color, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        r = self.rect().adjusted(6, 6, -6, -6)
        if self._kind == "expand":
            painter.drawLine(r.topLeft(), r.topLeft() + QPoint(5, 0))
            painter.drawLine(r.topLeft(), r.topLeft() + QPoint(0, 5))
            painter.drawLine(r.bottomRight(), r.bottomRight() - QPoint(5, 0))
            painter.drawLine(r.bottomRight(), r.bottomRight() - QPoint(0, 5))
            painter.drawLine(r.topLeft(), r.topLeft() + QPoint(4, 4))
            painter.drawLine(r.bottomRight(), r.bottomRight() - QPoint(4, 4))
        else:
            painter.drawLine(r.topLeft(), r.bottomRight())
            painter.drawLine(r.topRight(), r.bottomLeft())


class Composer(QPlainTextEdit):
    """Compact chat input: Enter sends, Ctrl/Shift+Enter inserts a newline."""

    submitted = Signal(str)
    expand_requested = Signal()
    # Shared with the chat view's send / cancel buttons, hence a ``Sizing`` step.
    ACTION_HEIGHT = DARK.sizing.action_control_height
    MIN_HEIGHT = DARK.composer.min_height
    MAX_HEIGHT = DARK.composer.max_height

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("prompt")
        self.setPlaceholderText(placeholder)
        self.setTabChangesFocus(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(2)
        self.setViewportMargins(0, 0, 26, 0)
        self.setFixedHeight(self.MIN_HEIGHT)
        self.textChanged.connect(self._sync_height)
        self._i18n = Retranslator(self)
        self._expand = _GlyphButton("expand", self)
        self._expand.setObjectName("expand-prompt")
        style(self._expand, variant=Variant.ICON)
        self._i18n.bind(lambda: self._expand.setToolTip(self.tr("Write a longer message")))
        self._expand.clicked.connect(self.expand_requested.emit)
        self._place_expand()
        self._sync_height()

    @property
    def text(self) -> str:
        return self.toPlainText()

    @text.setter
    def text(self, value: str) -> None:
        self.setPlainText(value)

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._sync_height()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
            ):
                self.insertPlainText("\n")
                self._sync_height()
                return
            self.submitted.emit(self.toPlainText())
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._place_expand()

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        self._sync_height()

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Re-measure the box when the font changes; the text inside it just did.

        ``_sync_height`` turns a line count into a pixel height via
        ``fontMetrics().lineSpacing()`` and pins the result with
        ``setFixedHeight``. It ran on construction, on every text change and on
        show -- every trigger except the one that changes ``lineSpacing()``. Raising
        the interface font therefore grew the text inside a box still sized for the
        old font, and the lines past the first were clipped.
        """

        super().changeEvent(event)
        if event.type() in FONT_CHANGED:
            self._sync_height()

    def _place_expand(self) -> None:
        margin = 6
        self._expand.move(
            self.width() - self._expand.width() - margin,
            self.height() - self._expand.height() - margin,
        )
        self._expand.raise_()

    def _sync_height(self) -> None:
        lines = max(1, self.toPlainText().count("\n") + 1)
        line_height = max(18, self.fontMetrics().lineSpacing())
        needed = self.MIN_HEIGHT + (lines - 1) * line_height
        target = min(self.MAX_HEIGHT, max(self.MIN_HEIGHT, needed))
        if target != self.height():
            self.setFixedHeight(target)
        overflowing = needed > self.MAX_HEIGHT
        policy = (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if overflowing
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        if self.verticalScrollBarPolicy() != policy:
            self.setVerticalScrollBarPolicy(policy)

    def sizeHint(self) -> QSize:
        return QSize(200, self.height())


class _PadEditor(QPlainTextEdit):
    """Writing surface: Enter newlines, Ctrl+Enter is ignored, Send is a button."""

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()
            return
        super().keyPressEvent(event)


class ComposerPad(QDialog):
    """Large writing pad: Enter inserts a newline, sending requires the Send button."""

    submitted = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        text: str = "",
        *,
        running: bool = False,
        enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("composer-pad")
        self.setModal(True)
        self.setWindowTitle(self.tr("Compose"))
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.sent = False
        self._running = running
        self._enabled = enabled
        self._drag_offset: QPoint | None = None
        self.resize(760, 520)
        self._build(text)
        self.set_running(running)
        self._sync_send()

    def _build(self, text: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = Card(CardLevel.FLOATING)
        card.setObjectName("composer-pad-card")
        layout = card.body
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("composer-pad-header")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.tr("Compose"))
        title.setObjectName("composer-pad-title")
        close = _GlyphButton("close")
        close.setObjectName("close-composer-pad")
        style(close, variant=Variant.ICON)
        close.setToolTip(self.tr("Close"))
        close.clicked.connect(self.reject)
        header_row.addWidget(title, 1)
        header_row.addWidget(close)
        header.installEventFilter(self)
        layout.addWidget(header)

        hint = QLabel(self.tr("Enter for a new line · Click Send"))
        hint.setObjectName("composer-pad-hint")
        style(hint, role=Role.CAPTION)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self._editor = _PadEditor()
        self._editor.setObjectName("prompt-pad")
        self._editor.setPlaceholderText(self.tr("Write or paste a long prompt"))
        self._editor.setPlainText(text)
        self._editor.setEnabled(self._enabled)
        cursor = self._editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._editor.setTextCursor(cursor)
        self._cancel = QPushButton(self.tr("Cancel"))
        self._cancel.setObjectName("cancel-pad")
        style(self._cancel, variant=Variant.DANGER, metric=Metric.ACTION)
        self._cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel.setAutoDefault(False)
        self._cancel.setDefault(False)
        self._cancel.setFixedHeight(Composer.ACTION_HEIGHT)
        self._send = QPushButton(self.tr("Send"))
        self._send.setObjectName("send-pad")
        style(self._send, variant=Variant.PRIMARY, metric=Metric.ACTION)
        self._send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send.setAutoDefault(False)
        self._send.setDefault(False)
        self._send.setFixedHeight(Composer.ACTION_HEIGHT)
        row.addWidget(self._editor, 1)
        row.addWidget(self._cancel, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(row, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self._count = QLabel("")
        self._count.setObjectName("composer-pad-count")
        style(self._count, role=Role.CAPTION)
        footer.addWidget(self._count, 1)
        footer.addWidget(QSizeGrip(card), 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(footer)
        root.addWidget(card)

        self._editor.textChanged.connect(self._on_text_changed)
        self._send.clicked.connect(self._send_text)
        self._cancel.clicked.connect(self.cancelled.emit)
        self._on_text_changed()

    @property
    def text(self) -> str:
        return self._editor.toPlainText()

    def set_running(self, running: bool) -> None:
        self._running = running
        self._send.setVisible(not running)
        self._cancel.setVisible(running)
        self._cancel.setEnabled(running)
        self._sync_send()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._editor.setEnabled(enabled)
        self._sync_send()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched.objectName() == "composer-pad-header" and isinstance(event, QMouseEvent):
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._drag_offset = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                return True
            if (
                event.type() == QEvent.Type.MouseMove
                and self._drag_offset is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_offset = None
        return super().eventFilter(watched, event)

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        parent = self.parentWidget()
        if parent is not None:
            host = parent.window().frameGeometry()
            self.move(host.center() - self.rect().center())
        self._editor.setFocus()

    def _on_text_changed(self) -> None:
        n = len(self.text)
        # Grouped digits stay in ``%1``: ``%n`` would render 1234 without separators.
        self._count.setText(
            self.tr("Empty") if n == 0 else self.tr("{count} characters").format(count=f"{n:,}")
        )
        self._sync_send()

    def _sync_send(self) -> None:
        self._send.setEnabled(self._enabled and bool(self.text.strip()) and not self._running)

    def _send_text(self) -> None:
        if not self._send.isEnabled():
            return
        self.sent = True
        self.submitted.emit(self.text)
        self.accept()
