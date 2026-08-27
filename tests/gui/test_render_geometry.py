"""Golden geometry snapshots and pixel probes for the transcript delegate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import fields

import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
from PySide6.QtWidgets import QStyleOptionViewItem

from kimix_gui.design import DARK
from kimix_gui.qt import transcript_cards as cards
from kimix_gui.qt.paint import RecordLayout, layout_record
from kimix_gui.qt.transcript import BodySelection, Transcript
from kimix_gui.transcript_data import ActivityField, literal
from kimix_gui.transcript_layout import BodySection, HeaderRun

from .transcript_helpers import activity_entry, append_entry, append_text, text_entry

#: Chrome roles and record hues, kept apart on purpose: a bar answers "what kind of
#: record is this" (``HUES``), a copy icon answers "what role does this play"
#: (``PALETTE``). The probes below say which one they mean.
PALETTE = DARK.palette
HUES = DARK.categories

#: Layout width in *cells* (the unit ``layout_record`` takes), held fixed so the
#: golden headers below stay comparable.
GOLDEN_CELL_WIDTH = 64
#: Row width in *pixels* handed to ``sizeHint``. Passed explicitly instead of
#: relying on the viewport so a scrollbar cannot shift the golden heights.
GOLDEN_ROW_WIDTH = 600
#: The font metrics every golden *pixel* height below is a function of. Qt's offscreen
#: platform ships a stub font engine -- one advance per character equal to the pixel
#: size, ``height == pixelSize``, ``lineSpacing == pixelSize + 2`` -- and that stub, not
#: a font installed here, produced these numbers: under ``offscreen``
#: ``QFontInfo(app.font()).family()`` is empty and ``QFontDatabase.families()`` has no
#: ``Segoe UI`` at all, so pinning the family would have guarded nothing. An offscreen
#: build that measures through fontconfig instead reports other numbers, and the golden
#: heights are skipped there; ``test_a_row_is_its_header_its_body_and_its_margins``
#: carries the part of the contract that holds on any engine.
GOLDEN_FONT_METRICS = {"height": 13, "line_spacing": 15, "advance": 13}
#: Canvas used by the pixel probes; height is generous enough for a wrapped body.
PROBE_WIDTH = 600
PROBE_HEIGHT = 90

_USER_TEXT = "What does layout_record return?"
_MARKDOWN_ANSWER = "## Result\n\nUse `layout_record` and **read** it."
_PLAIN_ANSWER = "plain answer without markup"
_THINKING_TEXT = "weighing the options"
_READ_CALL = "read  a.py"
_READ_RESULT = "read  12 lines\nfile contents"
_FAILED_CALL = "read  missing.py\npath: missing.py"
_FAILED_RESULT = "read  failed\nNo such file"

#: One golden row per case: the ``layout_record`` arguments, every field of the
#: resulting ``RecordLayout``, and the delegate's ``sizeHint`` height.
#:
#: These numbers are not sacred -- they are simply what the renderer does today.
#: A diff here is fine as long as it was intended; an unexplained diff means the
#: refactor moved visual behavior by accident.
GOLDEN_ROWS: tuple[tuple[str, dict[str, object], dict[str, object], int], ...] = (
    (
        "user-plain",
        {"entry": text_entry("user", _USER_TEXT)},
        {
            "header_runs": (HeaderRun("You", "label", True),),
            "body_sections": (BodySection(_USER_TEXT),),
            "compact": False,
            "label": "You",
            "bar_color": "cyan",
            "italic_body": False,
            "status": "",
            "full_summary": "",
        },
        56,
    ),
    (
        "assistant-markdown",
        {"entry": text_entry("assistant", _MARKDOWN_ANSWER, markdown=True)},
        {
            "header_runs": (HeaderRun("AI", "label", True),),
            "body_sections": (BodySection(_MARKDOWN_ANSWER, format="markdown"),),
            "compact": False,
            "label": "AI",
            "bar_color": "green",
            "italic_body": False,
            "status": "",
            "full_summary": "",
        },
        76,
    ),
    (
        "assistant-plain",
        {"entry": text_entry("assistant", _PLAIN_ANSWER)},
        {
            "header_runs": (HeaderRun("AI", "label", True),),
            "body_sections": (BodySection(_PLAIN_ANSWER),),
            "compact": False,
            "label": "AI",
            "bar_color": "green",
            "italic_body": False,
            "status": "",
            "full_summary": "",
        },
        56,
    ),
    (
        "thinking-expanded",
        {"entry": text_entry("thinking", _THINKING_TEXT), "expanded": True},
        {
            "header_runs": (HeaderRun("Think", "label", True),),
            "body_sections": (BodySection(_THINKING_TEXT, tone="muted"),),
            "compact": False,
            "label": "Think",
            "bar_color": "muted",
            "italic_body": True,
            "status": "",
            "full_summary": "",
        },
        56,
    ),
    (
        "tool-compact-pending",
        {"entry": activity_entry("read", call_id="golden-pending", summary="a.py")},
        {
            "header_runs": (
                HeaderRun("Read", "label", True),
                HeaderRun("a.py", "summary"),
            ),
            "body_sections": (),
            "compact": True,
            "label": "Read",
            "bar_color": "cyan",
            "italic_body": False,
            "status": "pending",
            "full_summary": "a.py",
        },
        30,
    ),
    (
        "tool-compact-ok",
        {
            "entry": activity_entry(
                "read",
                call_id="golden-ok",
                summary="a.py",
                result_summary="12 lines",
                result_text="file contents",
                status="ok",
            )
        },
        {
            "header_runs": (
                HeaderRun("Read", "label", True),
                HeaderRun("a.py · 12 lines", "summary"),
            ),
            "body_sections": (),
            "compact": True,
            "label": "Read",
            "bar_color": "cyan",
            "italic_body": False,
            "status": "ok",
            "full_summary": "a.py · 12 lines",
        },
        30,
    ),
    (
        "tool-expanded-error",
        {
            "entry": activity_entry(
                "read",
                call_id="golden-error",
                summary="missing.py",
                call_text="missing.py",
                result_summary="failed",
                result_text="No such file",
                status="error",
            ),
            "expanded": True,
        },
        {
            "header_runs": (
                HeaderRun("Read", "label", True),
                HeaderRun("missing.py · failed", "summary"),
            ),
            "body_sections": (
                BodySection("missing.py", format="code", tone="context"),
                BodySection("No such file", spacing="paragraph"),
            ),
            "compact": False,
            "label": "Read",
            "bar_color": "red",
            "italic_body": False,
            "status": "error",
            "full_summary": "missing.py · failed",
        },
        83,
    ),
    (
        "system-compact",
        {"entry": text_entry("system", "Session: demo")},
        {
            "header_runs": (
                HeaderRun("System", "label", True),
                HeaderRun("Session: demo", "summary"),
            ),
            "body_sections": (),
            "compact": True,
            "label": "System",
            "bar_color": "muted",
            "italic_body": False,
            "status": "",
            "full_summary": "Session: demo",
        },
        30,
    ),
)


@pytest.fixture
def pinned_font(qapp) -> Iterator[None]:
    """Pin the application font so golden heights do not depend on test order.

    ``theme.apply_theme`` mutates the process-wide font, so any earlier test that
    builds a ``KimixGuiApp`` changes what these snapshots would otherwise measure.
    """

    previous = QFont(qapp.font())
    font = QFont("Segoe UI")
    font.setPixelSize(13)
    qapp.setFont(font)
    yield
    qapp.setFont(previous)


def _layout_for_case(case: dict[str, object]) -> RecordLayout:
    arguments = dict(case)
    entry = arguments.pop("entry")
    return layout_record(entry, width=GOLDEN_CELL_WIDTH, **arguments)  # type: ignore[arg-type]


def _golden_transcript(qtbot) -> Transcript:
    """Build the golden rows in ``GOLDEN_ROWS`` order through public API only."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(transcript, GOLDEN_ROWS[0][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[1][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[2][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[3][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[4][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[5][1]["entry"])  # type: ignore[arg-type]
    append_entry(transcript, GOLDEN_ROWS[6][1]["entry"])  # type: ignore[arg-type]
    transcript.model().toggle_expanded(6)
    append_entry(transcript, GOLDEN_ROWS[7][1]["entry"])  # type: ignore[arg-type]
    assert len(transcript.records) == len(GOLDEN_ROWS)
    return transcript


def _font_metrics_fingerprint(font: QFont) -> dict[str, int]:
    metrics = QFontMetrics(font)
    return {
        "height": metrics.height(),
        "line_spacing": metrics.lineSpacing(),
        "advance": metrics.horizontalAdvance("x"),
    }


def _skip_unless_the_golden_engine(font: QFont) -> None:
    measured = _font_metrics_fingerprint(font)
    if measured != GOLDEN_FONT_METRICS:
        pytest.skip(f"golden heights need {GOLDEN_FONT_METRICS}; this engine reports {measured}")


def _size_hint_option() -> QStyleOptionViewItem:
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, GOLDEN_ROW_WIDTH, 0)
    return option


def _render_row(transcript: Transcript, row: int, *, hovered: bool = False) -> QImage:
    """Paint one row through the delegate onto an opaque background image."""

    image = QImage(PROBE_WIDTH, PROBE_HEIGHT, QImage.Format.Format_ARGB32)
    image.fill(QColor(PALETTE.bg))
    painter = QPainter(image)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, PROBE_WIDTH, PROBE_HEIGHT)
    transcript._set_hovered_row(row if hovered else None)
    try:
        transcript._delegate.paint(painter, option, transcript.model().index(row, 0))
    finally:
        painter.end()
    return image


def _probe_card() -> QRect:
    return cards.card_rect(QRect(0, 0, PROBE_WIDTH, PROBE_HEIGHT))


def _color_counts(image: QImage, area: QRect) -> Counter[str]:
    counts: Counter[str] = Counter()
    for y in range(area.top(), area.bottom() + 1):
        for x in range(area.left(), area.right() + 1):
            counts[image.pixelColor(x, y).name()] += 1
    return counts


def _assert_pixel(image: QImage, x: int, y: int, expected: str, *, where: str) -> None:
    """Compare a probed pixel to a token color, tolerating slight channel drift.

    Offscreen Qt currently paints the bar with ``fillRect`` on an integer rect and
    renders text without blending, so probes come out bit-exact. The tolerance is
    only insurance against a platform that enables subpixel blending; anything
    larger than 2 per channel means the token itself changed.
    """

    actual = image.pixelColor(x, y)
    wanted = QColor(expected)
    drift = max(
        abs(actual.red() - wanted.red()),
        abs(actual.green() - wanted.green()),
        abs(actual.blue() - wanted.blue()),
    )
    assert drift <= 2, f"{where} at ({x}, {y}): {actual.name()} is not close to {expected}"


def test_transcript_geometry_constants_are_pinned() -> None:
    """Every painted dimension in one place, so a silent tweak breaks a test."""

    assert cards.PAD_X == 12
    assert cards.PAD_Y == 8
    assert cards.CARD_MARGIN_X == 8
    assert cards.CARD_MARGIN_Y == 4
    assert cards.BAR_WIDTH == 3
    assert cards.BAR_WIDTH_HOVER == 5
    assert cards.BAR_HIT_WIDTH == 12
    assert cards.HEADER_HEIGHT == 22
    assert cards.LINE_HEIGHT == 18
    assert cards.COPY_WIDTH == 36
    assert cards.STATUS_WIDTH == cards.HEADER_HEIGHT
    assert cards.DISCLOSURE_WIDTH == cards.HEADER_HEIGHT
    assert cards.DOCUMENT_CACHE_SIZE == 32


@pytest.mark.parametrize(
    ("name", "case", "golden", "_height"),
    GOLDEN_ROWS,
    ids=[row[0] for row in GOLDEN_ROWS],
)
def test_layout_record_matches_the_golden_snapshot(
    qapp, name: str, case: dict[str, object], golden: dict[str, object], _height: int
) -> None:
    layout = _layout_for_case(case)
    assert {field.name for field in fields(RecordLayout)} == set(golden), (
        f"{name}: RecordLayout gained or lost a field; extend the golden snapshot"
    )
    for field, expected in golden.items():
        assert getattr(layout, field) == expected, f"{name}.{field} changed"


def test_measuring_and_painting_share_one_body(qtbot, pinned_font) -> None:
    """A row that is measured and then painted must not lay its body out twice.

    Documents used to be keyed by row, and the measuring path passed ``-1`` with a
    width it computed by subtracting a hard-coded 16, so every visible row held two
    copies of its body: the one its height came from and the one the painter drew.
    """

    transcript = _golden_transcript(qtbot)
    option = _size_hint_option()
    index = transcript.model().index(0, 0)
    transcript.bodies.invalidate()

    assert transcript._delegate.sizeHint(option, index).height() > cards.HEADER_HEIGHT
    measured = dict(transcript.bodies.documents)
    assert len(measured) == 1
    painted = transcript._delegate.document_for(option.rect, index)

    assert len(transcript.bodies.documents) == 1
    assert painted is next(iter(measured.values()))


def test_size_hints_match_the_golden_snapshot(qtbot, pinned_font) -> None:
    """Row heights for the fixed golden rows at a fixed 600px width."""

    transcript = _golden_transcript(qtbot)
    _skip_unless_the_golden_engine(transcript.font())
    option = _size_hint_option()
    measured = {
        name: transcript._delegate.sizeHint(option, transcript.model().index(row, 0)).height()
        for row, (name, _case, _golden, _height) in enumerate(GOLDEN_ROWS)
    }
    expected = {name: height for name, _case, _golden, height in GOLDEN_ROWS}
    assert measured == expected
    for row in range(len(GOLDEN_ROWS)):
        hint = transcript._delegate.sizeHint(option, transcript.model().index(row, 0))
        assert hint.width() == GOLDEN_ROW_WIDTH


def test_a_row_is_its_header_its_body_and_its_margins(qtbot, pinned_font) -> None:
    """The golden heights as an identity instead of as numbers.

    A row is the header band, the measured body, the bottom padding, and the card
    margin on each side -- true whatever the font engine measures, so this is the guard
    that survives on a machine where the absolute snapshot has to be skipped.
    """

    transcript = _golden_transcript(qtbot)
    option = _size_hint_option()
    margins = cards.CARD_MARGIN_Y * 2
    for row, (name, _case, golden, _height) in enumerate(GOLDEN_ROWS):
        index = transcript.model().index(row, 0)
        height = transcript._delegate.sizeHint(option, index).height()
        if golden["compact"]:
            assert height == cards.HEADER_HEIGHT + margins, name
            continue
        body = transcript.bodies.body_height(transcript.records[row], GOLDEN_ROW_WIDTH)
        assert body > 0, name
        assert height == cards.HEADER_HEIGHT + body + cards.PAD_Y + margins, name


def test_compact_rows_are_exactly_one_header_band_tall(qtbot, pinned_font) -> None:
    """Collapsed rows must stay header-height plus the two card margins."""

    transcript = _golden_transcript(qtbot)
    option = _size_hint_option()
    compact_rows = [
        row for row, (_name, _case, golden, _height) in enumerate(GOLDEN_ROWS) if golden["compact"]
    ]
    assert compact_rows
    for row in compact_rows:
        hint = transcript._delegate.sizeHint(option, transcript.model().index(row, 0))
        assert hint.height() == cards.HEADER_HEIGHT + (cards.CARD_MARGIN_Y * 2)


def test_hit_regions_keep_their_pixel_geometry() -> None:
    """Card and every interactive/text band for a 600x90 activity row."""

    option_rect = QRect(0, 0, 600, 90)
    card = cards.card_rect(option_rect)
    bar = cards.bar_rect(card)
    bar_hover = cards.bar_rect(card, hovered=True)
    gutter = cards.bar_hit_rect(card)
    header = cards.header_rect(card)
    copy = cards.copy_rect(card)
    disclosure = cards.disclosure_rect(card)
    status = cards.status_rect(card)
    activity_text = cards.header_text_rect(card, has_status=True, has_disclosure=True)
    body = cards.body_rect(card)

    assert card == QRect(8, 4, 584, 82)
    assert bar == QRect(8, 4, 3, 82)
    assert bar_hover == QRect(8, 4, 5, 82)
    assert gutter == QRect(8, 4, 12, 82)
    assert header == QRect(8, 4, 584, 22)
    assert copy == QRect(555, 4, 36, 22)
    assert disclosure == QRect(533, 4, 22, 22)
    assert status == QRect(20, 4, 22, 22)
    assert activity_text == QRect(42, 4, 491, 22)
    assert body == QRect(20, 26, 560, 52)


def test_hit_regions_keep_their_left_to_right_order() -> None:
    """Bar, text and vector actions keep a stable left-to-right order."""

    option_rect = QRect(0, 0, 600, 90)
    card = cards.card_rect(option_rect)
    bar = cards.bar_rect(card)
    gutter = cards.bar_hit_rect(card)
    header = cards.header_rect(card)
    copy = cards.copy_rect(card)
    disclosure = cards.disclosure_rect(card)
    status = cards.status_rect(card)
    activity_text = cards.header_text_rect(card, has_status=True, has_disclosure=True)
    body = cards.body_rect(card)

    assert bar.left() == gutter.left() == header.left() == card.left()
    assert bar.width() < gutter.width()
    assert gutter.right() < status.left() < activity_text.left()
    assert activity_text.right() < disclosure.left() < copy.left()
    assert gutter.right() < copy.left()
    # ``_copy_rect`` anchors by ``card.right() - _COPY_WIDTH``, which lands one
    # pixel short of the card edge; pinned so a fix is a deliberate change.
    assert copy.right() == card.right() - 1
    assert header.top() == card.top()
    assert body.top() == header.bottom() + 1
    assert body.left() > gutter.right()
    assert bar.height() == card.height(), "the bar spans the whole card, not just the header"


def test_header_summary_uses_real_pixels_up_to_the_disclosure_boundary(qapp, pinned_font) -> None:
    full_summary = "i" * 400
    layout = layout_record(
        activity_entry("read", call_id="pixel-summary", summary=full_summary),
        width=GOLDEN_CELL_WIDTH,
    )
    card = cards.card_rect(QRect(0, 0, 600, 90))
    line = cards.header_line(layout, card, qapp.font(), has_disclosure=True)
    text_band = cards.header_text_rect(card, has_status=True, has_disclosure=True)
    regular_font = QFont(qapp.font())
    regular_font.setBold(False)
    metrics = QFontMetrics(regular_font)

    assert layout.summary.endswith("...")
    assert layout.full_summary == full_summary
    assert line.summary == metrics.elidedText(
        full_summary,
        Qt.TextElideMode.ElideRight,
        line.summary_rect.width(),
    )
    assert line.summary.endswith("…")
    assert line.summary_rect.right() == text_band.right()
    assert metrics.horizontalAdvance(line.summary) <= line.summary_rect.width()
    assert line.summary_rect.width() - metrics.horizontalAdvance(line.summary) <= max(
        1, metrics.horizontalAdvance("i")
    )

    wide_card = cards.card_rect(QRect(0, 0, 900, 90))
    wide_line = cards.header_line(layout, wide_card, qapp.font(), has_disclosure=True)
    assert len(wide_line.summary) > len(line.summary)


def test_delegate_hit_tests_follow_the_pixel_geometry(qtbot) -> None:
    """Which region owns a point. Click *behavior* is covered in test_widgets.py."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(transcript, activity_entry("read", summary="a.py", call_text="path: a.py"))
    delegate = transcript._delegate
    option_rect = QRect(0, 0, 600, 90)

    gutter_low = QPoint(10, 80)
    assert delegate.bar_hit(option_rect, gutter_low) is True
    assert delegate.header_hit(option_rect, gutter_low) is False
    assert delegate.copy_hit(option_rect, gutter_low) is False

    # The gutter reaches into the header band, so both predicates fire there.
    gutter_in_header = QPoint(10, 10)
    assert delegate.bar_hit(option_rect, gutter_in_header) is True
    assert delegate.header_hit(option_rect, gutter_in_header) is True

    header_middle = QPoint(300, 10)
    assert delegate.bar_hit(option_rect, header_middle) is False
    assert delegate.header_hit(option_rect, header_middle) is True
    assert delegate.copy_hit(option_rect, header_middle) is False

    copy_point = QPoint(570, 10)
    assert delegate.copy_hit(option_rect, copy_point) is True
    assert delegate.header_hit(option_rect, copy_point) is False, "copy wins inside the header"
    assert delegate.copy_hit(option_rect, QPoint(555, 10)) is True, "left edge of the copy region"
    assert delegate.copy_hit(option_rect, QPoint(554, 10)) is False, "one pixel outside it"

    body_point = QPoint(300, 60)
    assert delegate.bar_hit(option_rect, body_point) is False
    assert delegate.header_hit(option_rect, body_point) is False
    assert delegate.copy_hit(option_rect, body_point) is False


@pytest.mark.parametrize(
    ("kind", "text", "token"),
    [
        ("user", "a question", "cyan"),
        ("assistant", "an answer", "green"),
        ("thinking", "musing about it", "muted"),
        ("error", "boom", "red"),
        ("tool", "read  a.py", "cyan"),
    ],
)
def test_bar_pixels_use_the_token_color_of_the_record_kind(
    qtbot, pinned_font, kind: str, text: str, token: str
) -> None:
    """Probe the painted bar itself rather than the color *name* it resolved."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    if kind == "tool":
        append_entry(transcript, activity_entry("read", summary="a.py"))
    else:
        append_text(transcript, kind, text)
    image = _render_row(transcript, 0)
    card = _probe_card()
    middle = card.top() + card.height() // 2

    for offset in range(cards.BAR_WIDTH):
        _assert_pixel(image, card.left() + offset, middle, HUES.resolve(token), where=f"{kind} bar")
    _assert_pixel(
        image,
        card.left() + cards.BAR_WIDTH,
        middle,
        PALETTE.bg,
        where=f"{kind} just right of the bar",
    )
    for y in (card.top(), card.bottom()):
        _assert_pixel(image, card.left() + 1, y, HUES.resolve(token), where=f"{kind} bar edge")


def test_hovered_muted_bar_widens_and_switches_to_the_text_color(qtbot, pinned_font) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(transcript, "thinking", "musing about it")
    card = _probe_card()
    middle = card.top() + card.height() // 2

    transcript._set_hovered_bar_row(0)
    image = _render_row(transcript, 0)
    for offset in range(cards.BAR_WIDTH_HOVER):
        _assert_pixel(image, card.left() + offset, middle, PALETTE.text, where="hovered bar")
    _assert_pixel(
        image,
        card.left() + cards.BAR_WIDTH_HOVER,
        middle,
        PALETTE.bg,
        where="right of the hovered bar",
    )

    transcript._set_hovered_bar_row(None)
    image = _render_row(transcript, 0)
    _assert_pixel(image, card.left() + 1, middle, HUES.muted, where="unhovered bar")


def test_hovered_colored_bar_only_widens(qtbot, pinned_font) -> None:
    """Hover swaps the color for muted bars only; colored bars keep their token."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(transcript, "user", "a question")
    card = _probe_card()
    middle = card.top() + card.height() // 2

    transcript._set_hovered_bar_row(0)
    image = _render_row(transcript, 0)
    for offset in range(cards.BAR_WIDTH_HOVER):
        _assert_pixel(image, card.left() + offset, middle, HUES.cyan, where="hovered user bar")
    _assert_pixel(
        image,
        card.left() + cards.BAR_WIDTH_HOVER,
        middle,
        PALETTE.bg,
        where="right of the hovered user bar",
    )


@pytest.mark.parametrize(
    ("kind", "text", "header_token", "body_token"),
    [
        ("user", "body text sample here", "cyan", "text"),
        ("assistant", "an answer with body", "green", "text"),
        ("thinking", "muted body sample here", "muted", "muted"),
    ],
)
def test_header_and_body_ink_use_their_token_colors(
    qtbot, pinned_font, kind: str, text: str, header_token: str, body_token: str
) -> None:
    """Glyph pixels, so only presence is asserted: counts depend on font metrics."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(transcript, kind, text)
    image = _render_row(transcript, 0)
    card = _probe_card()
    header_band = cards.header_text_rect(
        card,
        has_status=False,
        has_disclosure=kind not in {"user", "assistant"},
    )
    copy_band = cards.copy_rect(card)
    body_band = cards.body_rect(card)

    header_counts = _color_counts(image, header_band)
    assert header_counts[HUES.resolve(header_token)] > 0, "header label ink"
    assert _color_counts(image, copy_band)[PALETTE.muted] == 0, "copy icon stays hidden"
    assert _color_counts(image, body_band)[getattr(PALETTE, body_token)] > 0, "body ink"


def test_copy_icon_is_only_painted_for_the_hovered_row(qtbot, pinned_font) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(transcript, "assistant", "hover this row to copy it")
    copy_band = cards.copy_rect(_probe_card())

    idle = _color_counts(_render_row(transcript, 0), copy_band)
    hovered = _color_counts(_render_row(transcript, 0, hovered=True), copy_band)

    assert idle[PALETTE.muted] == 0
    assert hovered[PALETTE.muted] > 0


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pending", HUES.cyan),
        ("ok", PALETTE.success),
        ("error", PALETTE.error),
    ],
)
def test_activity_state_is_painted_as_a_vector_icon(
    qtbot, pinned_font, status: str, expected: str
) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(
        transcript,
        activity_entry(
            "read",
            call_id="call-1",
            summary="a.py",
            result_summary="done" if status != "pending" else "",
            status=status if status != "pending" else None,
        ),
    )

    image = _render_row(transcript, 0)
    icon_band = cards.status_rect(_probe_card())

    assert _color_counts(image, icon_band)[expected] > 0


@pytest.mark.parametrize(
    ("wire_name", "summary", "family_color"),
    [
        ("read", "a.py", HUES.cyan),
        ("todo_write", "2 items", HUES.yellow),
    ],
)
def test_successful_activity_label_keeps_its_family_color(
    qtbot, pinned_font, wire_name: str, summary: str, family_color: str
) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(
        transcript,
        activity_entry(
            wire_name,
            call_id="call-1",
            summary=summary,
            result_summary="done",
            status="ok",
        ),
    )

    image = _render_row(transcript, 0)
    header_band = cards.header_text_rect(
        _probe_card(),
        has_status=True,
        has_disclosure=True,
    )

    assert _color_counts(image, header_band)[family_color] > 0


def test_disclosure_is_painted_as_a_vector_icon(qtbot, pinned_font) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(transcript, activity_entry("read", summary="a.py"))

    image = _render_row(transcript, 0)
    icon_band = cards.disclosure_rect(_probe_card())

    assert _color_counts(image, icon_band)[PALETTE.muted] > 0


def test_expanded_activity_demotes_context_and_promotes_output(qtbot, pinned_font) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(
        transcript,
        activity_entry(
            "read",
            call_id="call-1",
            summary="a.py",
            fields=(
                ActivityField("path", literal("src/a.py"), role="primary", hint="path"),
                ActivityField("offset", literal("4"), role="secondary", hint="count"),
            ),
            result_summary="2 lines",
            result_text="first\nsecond",
            status="ok",
        ),
    )
    transcript.model().toggle_expanded(0)

    image = _render_row(transcript, 0)
    body_band = cards.body_rect(_probe_card())
    counts = _color_counts(image, body_band)
    document = transcript._delegate.document_for(
        QRect(0, 0, PROBE_WIDTH, 0), transcript.model().index(0, 0)
    )

    assert counts[PALETTE.muted] > 0, "invocation context stays secondary"
    assert counts[PALETTE.text] > 0, "returned content becomes the primary body"
    assert document is not None
    context = document.find("src/a.py")
    assert context.charFormat().foreground().color().name() == PALETTE.muted


def test_expanded_error_uses_a_neutral_surface(qtbot, pinned_font) -> None:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_entry(
        transcript,
        activity_entry(
            "bash",
            call_id="call-1",
            summary="pytest -q",
            call_text="command: pytest -q",
            result_summary="failed",
            result_text="2 tests failed",
            status="error",
        ),
    )
    transcript.model().toggle_expanded(0)

    image = _render_row(transcript, 0)
    card = _probe_card()

    _assert_pixel(
        image,
        card.right() - 20,
        card.bottom() - 20,
        PALETTE.surface,
        where="expanded error surface",
    )
    assert _color_counts(image, card)[PALETTE.danger_surface] == 0


def test_selection_paints_the_accent_highlight_over_the_body(qtbot, pinned_font) -> None:
    """Selected body text gets an accent background and the on-accent ink."""

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(transcript, "user", "hello there this is a long enough user message to select")
    card = _probe_card()
    body_band = cards.body_rect(card)

    unselected = _color_counts(_render_row(transcript, 0), body_band)
    assert unselected[PALETTE.accent] == 0
    assert unselected["#042f2e"] == 0

    transcript._selection = BodySelection(0, 0, 30)
    selected = _color_counts(_render_row(transcript, 0), body_band)

    assert selected[PALETTE.accent] > 500, "accent highlight fills the selected run"
    assert selected["#042f2e"] > 0, "selected glyphs switch to the on-accent ink"
    assert selected[PALETTE.text] < unselected[PALETTE.text], "selected ink left the base color"
