"""Where the parts of a transcript card sit, and how tall one is.

A record plus the width of its row goes in; rects, a laid-out :class:`RecordLayout`
and a measured body document come out. Splitting this away from the delegate is what
makes measuring and painting agree, because both now ask the same functions for the
same numbers instead of each doing the arithmetic its own way.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from PySide6.QtCore import QObject, QRect
from PySide6.QtGui import (
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextDocumentFragment,
    QTextOption,
)

from kimix_gui.design import DARK
from kimix_gui.qt.paint import RecordLayout, layout_record
from kimix_gui.qt.theme import active_theme
from kimix_gui.qt.transcript_model import TranscriptRecord
from kimix_gui.transcript_layout import BodySection, is_compact_entry

# Bodies kept laid out per view. A memory/latency trade-off, not a design decision, so
# it stays out of the theme.
DOCUMENT_CACHE_SIZE = 32
DOCUMENT_CACHE_CHARS = 2 * 1024 * 1024
HEIGHT_CACHE_SIZE = 2048

# Painted card geometry, from the design tokens. Bound once at import because the rect
# helpers below are module functions; a theme swap does not re-run them.
_METRICS = DARK.transcript
PAD_X = _METRICS.pad_x
PAD_Y = _METRICS.pad_y
CARD_MARGIN_X = _METRICS.card_margin_x
CARD_MARGIN_Y = _METRICS.card_margin_y
BAR_WIDTH = _METRICS.bar_width
# Painted width while the pointer sits in the bar's hit zone.
BAR_WIDTH_HOVER = _METRICS.bar_width_hover
# The bar is only ``BAR_WIDTH`` wide, so the whole left gutter toggles the row. It
# stops where the body text starts, keeping text selection untouched.
BAR_HIT_WIDTH = _METRICS.bar_hit_width
HEADER_HEIGHT = _METRICS.header_height
LINE_HEIGHT = _METRICS.line_height
COPY_WIDTH = _METRICS.copy_width
# The two header icons are square bands derived from the header itself. They are
# geometry, not text: reserving them here keeps clipping, painting and hit testing on
# the same model without adding another independent pixel scale.
STATUS_WIDTH = HEADER_HEIGHT
DISCLOSURE_WIDTH = HEADER_HEIGHT

# Rough pixels per character cell. ``layout_record`` counts cells, not pixels, because
# it decides where to truncate a one-line summary; this is the only place that converts.
_CELL_WIDTH = 8
_MIN_CELLS = 24
# Narrowest body Qt is asked to wrap at, so a collapsed pane still lays out.
_MIN_BODY_WIDTH = 40


def layout_for(record: TranscriptRecord, option_width: int) -> RecordLayout:
    """Lay out one record for a row this wide.

    The width is the row rect's, and the cell count comes off the *card*, which is the
    band the header is actually painted in. Both callers used to convert for
    themselves, from different rects, so a summary could be truncated for the painter
    at a width the hit test never saw.
    """

    return layout_record(
        record.entry,
        width=max(_MIN_CELLS, card_rect(QRect(0, 0, option_width, 0)).width() // _CELL_WIDTH),
        expanded=record.expanded,
    )


def card_rect(option_rect: QRect) -> QRect:
    """Inset a row rect to the painted card; the single source of row margins."""

    return option_rect.adjusted(
        CARD_MARGIN_X,
        CARD_MARGIN_Y,
        -CARD_MARGIN_X,
        -CARD_MARGIN_Y,
    )


def bar_rect(card: QRect, *, hovered: bool = False) -> QRect:
    """Accent bar spans the whole card so short rows keep the same coverage."""

    width = BAR_WIDTH_HOVER if hovered else BAR_WIDTH
    return QRect(card.left(), card.top(), width, card.height())


def bar_hit_rect(card: QRect) -> QRect:
    """Clickable gutter around the accent bar, full card height."""

    return QRect(card.left(), card.top(), BAR_HIT_WIDTH, card.height())


def header_rect(card: QRect) -> QRect:
    return QRect(card.left(), card.top(), card.width(), HEADER_HEIGHT)


def copy_rect(card: QRect) -> QRect:
    return QRect(card.right() - COPY_WIDTH, card.top(), COPY_WIDTH, HEADER_HEIGHT)


def disclosure_rect(card: QRect) -> QRect:
    """Disclosure action immediately before the copy action."""

    copy = copy_rect(card)
    return QRect(copy.left() - DISCLOSURE_WIDTH, card.top(), DISCLOSURE_WIDTH, HEADER_HEIGHT)


def status_rect(card: QRect) -> QRect:
    """Status icon band at the activity row's leading edge."""

    return QRect(card.left() + PAD_X, card.top(), STATUS_WIDTH, HEADER_HEIGHT)


def header_text_rect(
    card: QRect,
    *,
    has_status: bool,
    has_disclosure: bool,
) -> QRect:
    """Text band between optional leading status and trailing actions."""

    left = status_rect(card).right() + 1 if has_status else card.left() + PAD_X
    trailing = disclosure_rect(card).left() if has_disclosure else copy_rect(card).left()
    return QRect(left, card.top(), max(0, trailing - left), HEADER_HEIGHT)


def body_rect(card: QRect) -> QRect:
    return QRect(
        card.left() + PAD_X,
        card.top() + HEADER_HEIGHT,
        card.width() - PAD_X * 2,
        card.height() - HEADER_HEIGHT - PAD_Y,
    )


def body_width(option_width: int) -> int:
    """Wrap width of the body text in a row this wide.

    Derived from the same rects the painter uses, so measuring and painting cannot
    disagree. They used to: the measuring path subtracted a hard-coded 16, which
    happened to equal ``CARD_MARGIN_X * 2``, and nothing said so.
    """

    return max(_MIN_BODY_WIDTH, body_rect(card_rect(QRect(0, 0, option_width, 0))).width())


class CardBodies(QObject):
    """The laid-out bodies of the rows a view has shown, and how tall they are.

    Documents use the stable record identity and content revision rather than retaining
    the complete body in their keys. Integer heights have a larger independent cache,
    so QListView's repeated size-hint passes do not rebuild documents evicted from the
    much smaller laid-out-body cache.

    ``font`` is the seam. It is called on every miss rather than captured, so a font
    change is picked up as soon as the cache is dropped, and a test can pin what a
    body is measured with instead of moving the whole application's font.
    """

    def __init__(self, parent: QObject, *, font: Callable[[], QFont]) -> None:
        super().__init__(parent)
        self._font = font
        self._documents: OrderedDict[tuple[object, ...], QTextDocument] = OrderedDict()
        self._document_costs: dict[tuple[object, ...], int] = {}
        self._document_chars = 0
        self._heights: OrderedDict[tuple[object, ...], int] = OrderedDict()

    @property
    def documents(self) -> OrderedDict[tuple[object, ...], QTextDocument]:
        """The cache itself, for tests that assert on eviction and destruction."""

        return self._documents

    @property
    def heights(self) -> OrderedDict[tuple[object, ...], int]:
        """Cheap measured heights retained independently from laid-out documents."""

        return self._heights

    def invalidate(self) -> None:
        """Drop every document, because a font, a theme or a width changed."""

        for document in self._documents.values():
            _release(document)
        self._documents.clear()
        self._document_costs.clear()
        self._document_chars = 0
        self._heights.clear()

    def row_height(self, record: TranscriptRecord, option_width: int) -> int:
        """Height of the whole card, card margins included."""

        if is_compact_entry(record.entry, expanded=record.expanded):
            return HEADER_HEIGHT + CARD_MARGIN_Y * 2
        body = self.body_height(record, option_width)
        return HEADER_HEIGHT + body + PAD_Y + CARD_MARGIN_Y * 2

    def body_height(self, record: TranscriptRecord, option_width: int) -> int:
        """Measured height of the body text, or zero when the row has none."""

        layout = layout_for(record, option_width)
        if layout.compact or not layout.body:
            return 0
        key = (record.record_id, record.content_revision, option_width)
        cached = self._heights.pop(key, None)
        if cached is not None:
            self._heights[key] = cached
            return cached
        document = self.document(record, body_width(option_width), layout)
        height = max(LINE_HEIGHT, int(document.size().height()))
        self._heights[key] = height
        while len(self._heights) > HEIGHT_CACHE_SIZE:
            self._heights.popitem(last=False)
        return height

    def document(
        self,
        record: TranscriptRecord,
        width: int,
        layout: RecordLayout,
    ) -> QTextDocument:
        key = (
            record.record_id,
            record.content_revision,
            width,
            layout.italic_body,
        )
        document = self._documents.pop(key, None)
        cost = self._document_costs.pop(key, 0)
        self._document_chars -= cost
        if document is None:
            # Parented on purpose: see ``_release``.
            document = QTextDocument(self)
            font = QFont(self._font())
            font.setItalic(layout.italic_body)
            document.setDefaultFont(font)
            document.setDefaultStyleSheet("body, p, pre, code { background-color: transparent; }")
            _insert_sections(document, layout.body_sections)
            _apply_markdown_link_color(document)
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            document.setDefaultTextOption(option)
            document.setDocumentMargin(0)
            document.setTextWidth(max(_MIN_BODY_WIDTH, width))
            cost = len(layout.body)
        self._documents[key] = document
        self._document_costs[key] = cost
        self._document_chars += cost
        while (
            len(self._documents) > DOCUMENT_CACHE_SIZE
            or self._document_chars > DOCUMENT_CACHE_CHARS
        ) and len(self._documents) > 1:
            evicted_key, evicted = self._documents.popitem(last=False)
            self._document_chars -= self._document_costs.pop(evicted_key, 0)
            _release(evicted)
        return document


def _release(document: QTextDocument) -> None:
    """Destroy a cached document on the GUI thread, whoever drops the last reference.

    A Qt object may only be destroyed on the thread that owns it, and PySide6 deletes
    the C++ half of an *unparented* Qt object when its Python wrapper dies. The wrapper
    for a cached document dies inside a reference cycle -- the view holds the delegate,
    the delegate holds the view, and every ``Retranslator`` statement closes over its
    host -- so it is the cyclic collector that frees it, and the cyclic collector runs
    on whichever thread happened to cross the allocation threshold. In this application
    that is very often ``kimix-bridge``, which allocates continuously. Destroying a
    ``QTextDocument`` from there is undefined behaviour, and it showed up as access
    violations and heap corruption in the faulting thread named ``kimix-bridge``.

    Two changes make that unreachable. The documents are constructed with the delegate
    as their parent, which moves ownership to C++, so collecting the wrapper from any
    thread only frees the wrapper. Ownership by C++ then means an evicted document
    would live as long as the delegate, so eviction hands it to ``deleteLater``, which
    destroys it on the thread the object lives on -- the GUI thread -- and turns the
    cached wrapper into a harmless dangling one.

    Turning off automatic collection and driving ``gc.collect()`` from a GUI-thread
    timer would also have hidden the crash, and is what ``tests/gui/conftest.py`` does to
    keep the suite alive. It is not what the product does: it changes when every cycle
    in the process is reclaimed in order to fix one container, and it leaves the actual
    defect -- a Qt object whose destruction thread is unspecified -- in place.
    """
    document.deleteLater()


def _apply_markdown_link_color(document: QTextDocument) -> None:
    """Apply the theme color because Qt Markdown anchors ignore document CSS.

    Resolved per call, not per import: a theme swap only reaches these anchors
    because the cache is dropped and every body is rebuilt through here.
    """

    color = QTextCharFormat()
    color.setForeground(QColor(active_theme().palette.link))
    block = document.begin()
    while block.isValid():
        fragments = block.begin()
        while not fragments.atEnd():
            fragment = fragments.fragment()
            if fragment.isValid() and fragment.charFormat().isAnchor():
                cursor = QTextCursor(document)
                cursor.setPosition(fragment.position())
                cursor.setPosition(
                    fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor
                )
                cursor.mergeCharFormat(color)
            fragments += 1
        block = block.next()


def _insert_sections(document: QTextDocument, sections: tuple[BodySection, ...]) -> None:
    """Compose explicit sections; Markdown parsing remains Qt's responsibility."""

    cursor = QTextCursor(document)
    cursor.movePosition(QTextCursor.MoveOperation.End)
    for index, section in enumerate(sections):
        if index:
            cursor.insertBlock()
            if section.spacing == "paragraph":
                cursor.insertBlock()
        start = cursor.position()
        if section.format == "markdown":
            cursor.insertFragment(QTextDocumentFragment.fromMarkdown(section.text))
        else:
            cursor.insertText(section.text)
        _apply_section_tone(document, start, cursor.position(), section.tone)


def _apply_section_tone(document: QTextDocument, start: int, end: int, tone: str) -> None:
    if end <= start:
        return
    palette = active_theme().palette
    color = (
        palette.error if tone == "error" else palette.text if tone == "primary" else palette.muted
    )
    cursor = QTextCursor(document)
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    formatting = QTextCharFormat()
    formatting.setForeground(QColor(color))
    cursor.mergeCharFormat(formatting)
