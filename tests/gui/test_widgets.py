from __future__ import annotations

import asyncio

import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QPlainTextEdit,
    QPushButton,
    QStyleOptionViewItem,
    QWidget,
)

from kimix_gui.qt.bridge import KimixBridge, StreamCoalescer
from kimix_gui.qt.composer import Composer, ComposerPad
from kimix_gui.qt.main_window import Toast
from kimix_gui.qt.transcript import MAX_TRANSCRIPT_CHARS, Transcript
from kimix_gui.qt.transcript_cards import CARD_MARGIN_Y, HEADER_HEIGHT
from kimix_gui.qt.transcript_model import MAX_PRESENTATION_RECORDS
from kimix_gui.transcript_data import ROOT_SOURCE, AppendText, ReplaceEntry, TranscriptUpdate

from .transcript_helpers import (
    activity_entry,
    append_entry,
    append_fragment,
    append_text,
    append_texts,
    history_entries,
    row_kind,
    row_status,
    row_text,
)

# What a collapsed row measures: one header band inside the card margins.
_COMPACT_HEIGHT = HEADER_HEIGHT + CARD_MARGIN_Y * 2


def _shown(
    qtbot, *, max_chars: int = MAX_TRANSCRIPT_CHARS, width: int = 640, height: int = 400
) -> Transcript:
    transcript = Transcript(max_chars=max_chars)
    qtbot.addWidget(transcript)
    transcript.resize(width, height)
    transcript.show()
    qtbot.waitUntil(lambda: transcript.isVisible(), timeout=5_000)
    return transcript


def _copy_pos(transcript: Transcript, row: int = 0) -> QPoint:
    rect = transcript.visualRect(transcript.model().index(row, 0))
    return QPoint(rect.right() - 10, rect.top() + 10)


def _header_pos(transcript: Transcript, row: int = 0) -> QPoint:
    rect = transcript.visualRect(transcript.model().index(row, 0))
    return QPoint(rect.left() + 24, rect.top() + 10)


def _send_wheel(transcript: Transcript, delta: int) -> None:
    viewport = transcript.viewport()
    pos = QPointF(viewport.rect().center())
    global_pos = QPointF(viewport.mapToGlobal(pos.toPoint()))
    event = QWheelEvent(
        pos,
        global_pos,
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(viewport, event)


def test_transcript_keeps_records_for_scrollback(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "user", "one")
    append_text(transcript, "user", "two")
    append_text(transcript, "user", "three")
    append_text(transcript, "user", "four")
    assert [row_text(record) for record in transcript.records] == ["one", "two", "three", "four"]
    assert transcript.verticalScrollBar().maximum() >= 0
    assert all(
        transcript.indexWidget(transcript.model().index(row, 0)) is None
        for row in range(len(transcript.records))
    )


def test_transcript_keeps_full_dialogue_text_when_streaming(qtbot) -> None:
    transcript = _shown(qtbot)
    first = "a" * 100
    second = "b" * 4_500
    append_fragment(transcript, "assistant", first, key="assistant-stream")
    append_fragment(transcript, "assistant", second, key="assistant-stream")
    assert len(transcript.records) == 1
    assert row_text(transcript.records[0]) == first + second
    user_text = "question " + ("x" * 4_500)
    append_text(transcript, "user", user_text)
    assert row_text(transcript.records[-1]) == user_text


def test_transcript_keeps_full_auxiliary_text(qtbot) -> None:
    transcript = _shown(qtbot)
    tool_text = "x" * 4_500
    append_entry(transcript, activity_entry("read", summary="large output", call_text=tool_text))
    assert tool_text in row_text(transcript.records[0])
    stream_first = "a" * 100
    stream_second = "b" * 4_500
    append_fragment(transcript, "thinking", stream_first, key="thinking-stream")
    append_fragment(transcript, "thinking", stream_second, key="thinking-stream")
    assert row_text(transcript.records[-1]) == stream_first + stream_second


def test_transcript_paints_records_lazily_and_bounds_memory(qtbot) -> None:
    transcript = _shown(qtbot, max_chars=32)
    append_texts(transcript, [("user", f"message-{index}-{'x' * 20}") for index in range(10)])
    assert transcript.omitted_records > 0
    assert len(transcript.bodies.documents) <= 32


def test_transcript_batch_mutations_keep_full_history(qtbot) -> None:
    transcript = _shown(qtbot)
    append_texts(transcript, [("user", "a"), ("assistant", "b"), ("user", "c")])
    assert [row_text(record) for record in transcript.records] == ["a", "b", "c"]


def test_transcript_model_exposes_ast_accessibility_text(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "user", "hello")

    accessible = transcript.model().index(0, 0).data(Qt.ItemDataRole.AccessibleTextRole)
    assert accessible == "You\nhello"


def test_non_dialogue_records_take_one_header_band(qtbot) -> None:
    transcript = _shown(qtbot, width=320)
    append_entry(
        transcript,
        activity_entry("read", summary="example.py", call_text="Path: example.py\nArguments: {}"),
    )
    assert transcript.sizeHintForRow(0) == _COMPACT_HEIGHT
    assert transcript.visible_text().startswith("Read")


def test_only_explicit_copy_action_copies_dialogue_message(qtbot) -> None:
    transcript = _shown(qtbot)
    append_texts(
        transcript,
        [("system", "Session: demo"), ("user", "First question"), ("assistant", "First answer")],
    )
    append_entry(
        transcript,
        activity_entry(
            "read",
            summary="example.py",
            call_text="Path: example.py",
            result_summary="10 lines",
            result_text="10 lines",
            status="ok",
        ),
    )
    append_texts(transcript, [("user", "Second question"), ("assistant", "Second answer")])
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(2, 0)).height() > 0)
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    assistant_rect = transcript.visualRect(transcript.model().index(2, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(assistant_rect.left() + 24, assistant_rect.top() + 36),
    )
    assert clipboard.text() == "unchanged"
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 2))
    assert clipboard.text() == "First answer"
    qtbot.mouseClick(
        transcript.viewport(), Qt.MouseButton.RightButton, pos=_header_pos(transcript, 3)
    )
    assert clipboard.text() == "First answer"


def test_clicking_copy_action_on_system_record_copies_only_that_record(qtbot) -> None:
    transcript = _shown(qtbot)
    append_texts(
        transcript,
        [
            ("system", "Session: demo"),
            ("user", "Question"),
            ("assistant", "Answer"),
        ],
    )
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 0))
    assert QApplication.clipboard().text() == "Session: demo"


def test_mouse_move_tracks_the_row_that_reveals_the_copy_action(qtbot) -> None:
    transcript = _shown(qtbot, height=300)
    append_texts(transcript, [("user", "Question"), ("assistant", "Answer")])
    second = transcript.visualRect(transcript.model().index(1, 0))

    qtbot.mouseMove(
        transcript.viewport(),
        pos=QPoint(transcript.viewport().width() // 2, transcript.viewport().height() - 2),
    )
    assert transcript.hovered_row is None

    qtbot.mouseMove(transcript.viewport(), pos=second.center())
    assert transcript.hovered_row == 1


def test_drag_selects_body_text_and_ctrl_c_copies_selection(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    append_text(transcript, "user", "ALPHA-SELECTABLE-TEXT OMEGA")
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    rect = transcript.visualRect(transcript.model().index(0, 0))
    start = QPoint(rect.left() + 24, rect.top() + 36)
    end = QPoint(rect.left() + 220, rect.top() + 36)
    qtbot.mousePress(transcript.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(transcript.viewport(), pos=end)
    qtbot.mouseRelease(transcript.viewport(), Qt.MouseButton.LeftButton, pos=end)
    selected = transcript.selected_text()
    assert selected
    assert selected in "ALPHA-SELECTABLE-TEXT OMEGA"
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    transcript.setFocus()
    qtbot.keyClick(transcript, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert clipboard.text() == selected


def test_clicking_body_does_not_copy_without_selection(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "assistant", "Selectable assistant reply")
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    rect = transcript.visualRect(transcript.model().index(0, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 24, rect.top() + 36),
    )
    assert clipboard.text() == "unchanged"
    assert transcript.selected_text() == ""
    transcript.setFocus()
    qtbot.keyClick(transcript, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert clipboard.text() == "unchanged"


def test_auxiliary_record_expands_collapses_and_copies_from_action(qtbot) -> None:
    transcript = _shown(qtbot, width=320, height=400)
    append_entry(
        transcript,
        activity_entry("read", summary="file", call_text="Path: example.py"),
    )
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    qtbot.mouseClick(
        transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript, 0)
    )
    assert transcript.records[0].expanded is True
    assert transcript.sizeHintForRow(0) > _COMPACT_HEIGHT
    assert "Path: example.py" in transcript.visible_text()

    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    rect = transcript.visualRect(transcript.model().index(0, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 24, rect.top() + 40),
    )
    assert clipboard.text() == "unchanged"
    assert transcript.records[0].expanded is True

    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 0))
    assert clipboard.text() == "read  file\nPath: example.py"
    qtbot.mouseClick(
        transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript, 0)
    )
    assert transcript.records[0].expanded is False
    assert transcript.sizeHintForRow(0) == _COMPACT_HEIGHT


def test_thinking_records_start_expanded_in_italic(qtbot) -> None:
    transcript = _shown(qtbot, width=384)
    append_text(transcript, "thinking", "considering options")
    visible = transcript.visible_text()
    assert transcript.records[0].expanded is True
    assert transcript.sizeHintForRow(0) > _COMPACT_HEIGHT
    assert visible.startswith("Think\n")
    assert "considering options" in visible
    # The body is its own line: the header names the record, it does not quote it.
    assert "considering options" not in visible.splitlines()[0]


def test_clicking_scrolled_message_body_does_not_copy(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    append_texts(
        transcript,
        [
            item
            for turn in range(12)
            for item in (("user", f"Question {turn}"), ("assistant", f"Answer {turn}"))
        ],
    )
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    transcript.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    target = transcript.model().index(10, 0)
    transcript.scrollTo(target)
    QApplication.clipboard().setText("unchanged")
    rect = transcript.visualRect(target)
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 20, rect.top() + 36),
    )
    assert QApplication.clipboard().text() == "unchanged"


def test_transcript_opens_scrolled_to_latest_history(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    append_text(transcript, "system", "Session: demo")
    append_texts(
        transcript,
        [("user", f"history-first-{i:03d}") for i in range(50)]
        + [("assistant", "history-latest-reply")],
    )
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    assert transcript.verticalScrollBar().value() == transcript.verticalScrollBar().maximum()
    visible = transcript.visible_text()
    assert "history-first-000" not in visible
    assert "history-latest-reply" in visible


def test_transcript_stays_put_when_user_scrolls_up(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    append_texts(transcript, [("user", f"turn-{i:03d}") for i in range(40)])
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    transcript.verticalScrollBar().setValue(0)
    QApplication.processEvents()
    assert transcript.verticalScrollBar().value() == 0
    assert transcript.pinned_to_latest is False
    append_text(transcript, "user", "new-after-scroll")
    assert transcript.verticalScrollBar().value() == 0
    visible = transcript.visible_text()
    assert "turn-000" in visible
    assert "new-after-scroll" not in visible


def test_replacing_history_window_keeps_live_rows(qtbot) -> None:
    transcript = _shown(qtbot)
    append_texts(
        transcript, [("system", "Session: demo"), ("user", "old-history"), ("assistant", "live")]
    )
    transcript.mark_history_window(1, 2)
    transcript.replace_history(
        history_entries([("user", "page-history"), ("assistant", "page-reply")])
    )
    assert [(row_kind(record), row_text(record)) for record in transcript.records] == [
        ("system", "Session: demo"),
        ("user", "page-history"),
        ("assistant", "page-reply"),
        ("assistant", "live"),
    ]
    assert transcript.history_window == (1, 3)


def test_jump_to_turn_unpins_and_scrolls_immediately(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.replace_history(
        history_entries([("user", f"q{index}", index) for index in range(20)])
    )
    transcript.jump_to_latest()
    assert transcript.pinned_to_latest is True
    transcript.jump_to_turn(0)
    assert transcript.pinned_to_latest is False
    assert transcript.viewport_turn() == 0
    assert "q0" in transcript.visible_text()
    assert "q19" not in transcript.visible_text()


def test_jump_to_latest_pins_without_leaving_older_rows(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.replace_history(
        history_entries([("user", f"q{index}", index) for index in range(20)])
    )
    transcript.jump_to_turn(0)
    assert transcript.pinned_to_latest is False
    transcript.jump_to_latest()
    assert transcript.pinned_to_latest is True
    assert transcript.verticalScrollBar().value() == transcript.verticalScrollBar().maximum()
    assert "q19" in transcript.visible_text()


def test_short_history_page_requests_older_on_each_outward_wheel(qtbot) -> None:
    transcript = _shown(qtbot)
    top_requests: list[None] = []
    transcript.reached_top.connect(lambda: top_requests.append(None))
    transcript.replace_history(
        history_entries([("user", "latest", 9)]),
        pin_latest=True,
    )
    transcript.jump_to_latest()
    QApplication.processEvents()

    assert transcript.verticalScrollBar().maximum() == 0
    _send_wheel(transcript, 120)
    assert len(top_requests) == 1

    # An edge request is one-shot until its replacement window arrives.
    _send_wheel(transcript, 120)
    assert len(top_requests) == 1

    transcript.replace_history(
        history_entries([("user", "older", 8)]),
        target_turn=8,
    )
    transcript.jump_to_turn(8)
    QApplication.processEvents()

    assert transcript.verticalScrollBar().maximum() == 0
    _send_wheel(transcript, 120)
    assert len(top_requests) == 2


def test_replace_history_keeps_in_flight_stream(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "system", "Session: demo")
    transcript.mark_history_window()
    transcript.replace_history(history_entries([("user", "q0", 0)]))
    append_fragment(transcript, "assistant", "hel", key="live-answer")
    append_fragment(transcript, "assistant", "lo", key="live-answer")
    transcript.replace_history(history_entries([("user", "q0", 0), ("assistant", "old", 0)]))
    append_fragment(transcript, "assistant", "!", key="live-answer")
    assert [row_text(record) for record in transcript.records] == [
        "Session: demo",
        "q0",
        "old",
        "hello!",
    ]


def test_history_window_skips_fifo_trim(qtbot) -> None:
    transcript = _shown(qtbot, max_chars=32)
    transcript.mark_history_window()
    append_texts(transcript, [("user", f"message-{index}-{'x' * 20}") for index in range(10)])
    assert transcript.omitted_records == 0
    assert len(transcript.records) == 10


def test_history_exposes_only_a_bounded_qt_presentation_window(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=240)
    transcript.replace_history(
        history_entries([("user", f"q{turn}", turn) for turn in range(400)]),
        pin_latest=True,
    )

    active_limit = transcript.model().presentation_record_limit
    assert 1 < active_limit < MAX_PRESENTATION_RECORDS
    assert len(transcript.records) == active_limit
    assert transcript.has_hidden_older_history is True
    first_turn = transcript.records[0].turn
    first_id = transcript.records[0].record_id
    resets: list[None] = []
    inserted: list[tuple[int, int]] = []
    removed: list[tuple[int, int]] = []
    transcript.model().modelReset.connect(lambda: resets.append(None))
    transcript.model().rowsInserted.connect(
        lambda _parent, first, last: inserted.append((first, last))
    )
    transcript.model().rowsRemoved.connect(
        lambda _parent, first, last: removed.append((first, last))
    )

    assert transcript._reveal_local_history(older=True) is True
    assert len(transcript.records) <= active_limit
    assert transcript.records[0].turn < first_turn
    assert any(record.record_id == first_id for record in transcript.records)
    assert resets == []
    assert inserted
    assert removed
    anchored_row = next(
        row for row, record in enumerate(transcript.records) if record.record_id == first_id
    )
    anchored_rect = transcript.visualRect(transcript.model().index(anchored_row, 0))
    assert anchored_rect.top() <= transcript.viewport().rect().top() + 4
    assert transcript.verticalScrollBar().value() > 0


def test_history_presentation_limit_tracks_the_viewport(qtbot) -> None:
    compact = _shown(qtbot, width=640, height=240)
    tall = _shown(qtbot, width=640, height=900)
    entries = history_entries([("user", f"q{turn}", turn) for turn in range(400)])

    compact.replace_history(entries, pin_latest=True)
    tall.replace_history(entries, pin_latest=True)

    compact_limit = compact.model().presentation_record_limit
    tall_limit = tall.model().presentation_record_limit
    assert compact_limit < tall_limit <= MAX_PRESENTATION_RECORDS
    assert len(compact.records) == compact_limit
    assert len(tall.records) == tall_limit

    compact.resize(640, 900)
    qtbot.waitUntil(
        lambda: compact.model().presentation_record_limit == tall_limit,
        timeout=1_000,
    )
    assert len(compact.records) == tall_limit


def test_local_history_shift_preserves_rows_outside_the_projection(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=240)
    append_text(transcript, "system", "Session: demo")
    transcript.mark_history_window()
    transcript.replace_history(
        history_entries([("user", f"q{turn}", turn) for turn in range(400)]),
        pin_latest=True,
    )
    append_text(transcript, "assistant", "live tail")
    prefix = transcript.records[0]
    suffix = transcript.records[-1]

    assert transcript._reveal_local_history(older=True) is True

    assert transcript.records[0] is prefix
    assert transcript.records[-1] is suffix
    assert transcript.history_window == (1, len(transcript.records) - 1)


def test_local_history_projection_stays_consistent_across_repeated_shifts(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=240)
    transcript.replace_history(
        history_entries([("user", f"q{turn}", turn) for turn in range(400)]),
        pin_latest=True,
    )
    model = transcript.model()
    resets: list[None] = []
    model.modelReset.connect(lambda: resets.append(None))

    for older in (True,) * 20 + (False,) * 20:
        assert transcript._reveal_local_history(older=older) is True
        source = model._history_source[model._history_source_start : model._history_source_end]
        start, end = transcript.history_window or (0, 0)
        assert transcript.records[start:end] == source
        assert len(source) <= model.presentation_record_limit

    assert resets == []


def test_scroll_values_are_coalesced_until_the_next_event_loop_frame(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _shown(qtbot)
    processed: list[int] = []
    transcript._cancel_pending_scroll()
    monkeypatch.setattr(transcript, "_process_scroll_value", processed.append)

    transcript._queue_scroll(10)
    transcript._queue_scroll(20)
    transcript._queue_scroll(30)

    assert processed == []
    qtbot.waitUntil(lambda: bool(processed), timeout=1_000)
    assert processed == [30]


def test_cached_height_survives_qtextdocument_eviction(qtbot) -> None:
    transcript = _shown(qtbot)
    append_texts(transcript, [("assistant", f"row {row} " + "body " * 30) for row in range(40)])
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 600, 0)
    first = transcript.model().index(0, 0)
    transcript.bodies.invalidate()

    height = transcript._delegate.sizeHint(option, first).height()
    for row in range(1, 40):
        transcript._delegate.document_for(option.rect, transcript.model().index(row, 0))
    before = list(transcript.bodies.documents)

    assert transcript._delegate.sizeHint(option, first).height() == height
    assert list(transcript.bodies.documents) == before
    assert transcript.bodies.heights


def test_overlapping_history_snapshots_reuse_document_by_stable_record_id(qtbot) -> None:
    transcript = _shown(qtbot)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 600, 0)
    item = history_entries([("assistant", "same body", 7)])
    transcript.replace_history(item, target_turn=7)
    first = transcript._delegate.document_for(option.rect, transcript.model().index(0, 0))

    transcript.replace_history(item, target_turn=7)
    reused = transcript._delegate.document_for(option.rect, transcript.model().index(0, 0))

    assert reused is first

    transcript.replace_history(
        history_entries([("assistant", "changed body", 7)]),
        target_turn=7,
    )
    changed = transcript._delegate.document_for(option.rect, transcript.model().index(0, 0))
    assert changed is not first
    assert changed.toPlainText() == "changed body"


def test_prompt_enter_submits_without_inserting_newline(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "hi")
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    assert submitted == ["hi"]
    assert prompt.text == "hi"


def test_prompt_ctrl_enter_inserts_newline_then_enter_sends(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "hi")
    qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(prompt, "there")
    assert prompt.text == "hi\nthere"
    assert submitted == []
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    assert submitted == ["hi\nthere"]


def test_prompt_shift_enter_inserts_newline(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "a")
    qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    qtbot.keyClicks(prompt, "b")
    assert prompt.text == "a\nb"
    assert submitted == []


def test_prompt_grows_with_lines_and_caps_height(qtbot) -> None:
    prompt = Composer("Ask")
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    initial_height = prompt.height()
    assert initial_height >= Composer.MIN_HEIGHT
    assert prompt.verticalScrollBar().isVisible() is False
    prompt.setPlainText("one line of prompt text")
    assert prompt.height() == initial_height
    assert prompt.verticalScrollBar().isVisible() is False
    for _ in range(20):
        qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert prompt.height() == Composer.MAX_HEIGHT
    assert prompt.text.count("\n") == 20
    assert prompt.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded


def test_composer_expand_emits_and_pad_sends_long_text(qtbot) -> None:
    prompt = Composer("Ask")
    opened: list[int] = []
    prompt.expand_requested.connect(lambda: opened.append(1))
    qtbot.addWidget(prompt)
    prompt.show()
    expand = prompt.findChild(QPushButton, "expand-prompt")
    assert expand is not None
    expand.click()
    assert opened == [1]
    assert prompt.height() == Composer.MIN_HEIGHT

    submitted: list[str] = []
    long_text = "line\n" * 80 + "end"
    pad = ComposerPad(long_text)
    pad.submitted.connect(submitted.append)
    qtbot.addWidget(pad)
    pad.show()
    assert pad.findChild(QWidget, "prompt-pad") is not None
    send = pad.findChild(QPushButton, "send-pad")
    assert send is not None
    send.click()
    assert submitted == [long_text]
    assert pad.sent is True


def test_composer_pad_enter_newlines_send_requires_click(qtbot) -> None:
    pad = ComposerPad("hello")
    submitted: list[str] = []
    pad.submitted.connect(submitted.append)
    qtbot.addWidget(pad)
    pad.show()
    editor = pad.findChild(QPlainTextEdit, "prompt-pad")
    assert editor is not None
    editor.setFocus()
    editor.setPlainText("hello\nworld")
    cursor = editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    assert submitted == []
    assert editor.toPlainText() == "hello\nworld\n"
    qtbot.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert submitted == []
    assert editor.toPlainText() == "hello\nworld\n"
    send = pad.findChild(QPushButton, "send-pad")
    assert send is not None
    send.click()
    assert submitted == ["hello\nworld\n"]


def test_toast_centers_at_bottom_and_auto_hides(qtbot) -> None:
    host = QWidget()
    host.resize(800, 600)
    qtbot.addWidget(host)
    host.show()
    toast = Toast(host)
    toast.show_message("AI message copied", duration_ms=80)
    assert toast.isVisible()
    mid = toast.x() + toast.width() / 2
    assert abs(mid - 400) < 24
    assert toast.y() + toast.height() > 520
    qtbot.waitUntil(lambda: not toast.isVisible(), timeout=3_000)


def test_toast_click_dismisses_before_timer(qtbot) -> None:
    host = QWidget()
    host.resize(800, 600)
    qtbot.addWidget(host)
    host.show()
    toast = Toast(host)
    toast.show_message("Copied", duration_ms=10_000)
    assert toast.isVisible()
    qtbot.mouseClick(toast, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not toast.isVisible(), timeout=3_000)


def test_transcript_replaces_a_tool_call_with_its_complete_snapshot(qtbot) -> None:
    transcript = _shown(qtbot)
    pending = activity_entry("read", call_id="call-1", summary="a.py")
    append_entry(transcript, pending)

    assert [row_kind(record) for record in transcript.records] == ["tool"]
    assert row_status(transcript.records[0]) == "pending"

    complete = activity_entry(
        "read",
        call_id="call-1",
        summary="a.py",
        result_summary="12 lines",
        result_text="file contents",
        status="ok",
    )
    transcript.apply_mutation(ReplaceEntry(complete))

    assert [row_kind(record) for record in transcript.records] == ["tool"]
    assert row_status(transcript.records[0]) == "ok"
    assert "file contents" in row_text(transcript.records[0])


def test_transcript_replaces_parallel_tool_results_by_entry_key(qtbot) -> None:
    transcript = _shown(qtbot)
    append_entry(transcript, activity_entry("read", call_id="call-a", summary="a.py"))
    append_entry(transcript, activity_entry("read", call_id="call-b", summary="b.py"))

    assert len(transcript.records) == 2

    transcript.apply_mutation(
        ReplaceEntry(
            activity_entry(
                "read", call_id="call-b", summary="b.py", result_text="b done", status="ok"
            )
        )
    )
    transcript.apply_mutation(
        ReplaceEntry(
            activity_entry(
                "read",
                call_id="call-a",
                summary="a.py",
                result_text="a failed",
                status="error",
            )
        )
    )

    assert len(transcript.records) == 2
    assert [row_status(record) for record in transcript.records] == ["error", "ok"]
    assert "a failed" in row_text(transcript.records[0])
    assert "b done" in row_text(transcript.records[1])


def test_transcript_ignores_an_identical_activity_snapshot(qtbot) -> None:
    transcript = _shown(qtbot)
    complete = activity_entry(
        "read", call_id="call-1", summary="a.py", result_text="12 lines", status="ok"
    )
    append_entry(transcript, complete)
    revision = transcript.records[0].content_revision

    transcript.apply_mutation(ReplaceEntry(complete))
    assert transcript.records[0].content_revision == revision
    assert "12 lines" in row_text(transcript.records[0])


def test_transcript_replace_upserts_a_missing_activity(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "system", "no tools yet")

    transcript.apply_mutation(
        ReplaceEntry(
            activity_entry(
                "read",
                call_id="call-missing",
                summary="missing",
                result_text="done",
                status="ok",
            )
        )
    )
    assert [row_kind(record) for record in transcript.records] == ["system", "tool"]


def test_transcript_copies_the_merged_call_and_result_body(qtbot) -> None:
    transcript = _shown(qtbot)
    append_entry(
        transcript,
        activity_entry(
            "read",
            call_id="call-1",
            summary="a.py",
            call_text="path: a.py",
            result_summary="12 lines",
            result_text="file contents",
            status="ok",
        ),
    )
    transcript._copy_record(transcript.records[0])

    clipboard = QApplication.clipboard().text()

    assert "path: a.py" in clipboard
    assert "file contents" in clipboard


async def test_coalescer_splits_streamed_tool_calls_by_call_id() -> None:
    emitted: list[TranscriptUpdate] = []
    coalescer = StreamCoalescer(emitted.append, asyncio.get_running_loop())

    coalescer.feed(ReplaceEntry(activity_entry("read", call_id="call-a", summary="a.py")), 1)
    coalescer.feed(ReplaceEntry(activity_entry("read", call_id="call-b", summary="b.py")), 1)
    coalescer.flush()

    snapshots = [update.mutation for update in emitted]
    assert all(isinstance(snapshot, ReplaceEntry) for snapshot in snapshots)
    assert [snapshot.entry.activity.call_id for snapshot in snapshots] == [  # type: ignore[union-attr]
        "call-a",
        "call-b",
    ]


async def test_coalescer_keeps_text_fragments_of_one_entry_together() -> None:
    emitted: list[TranscriptUpdate] = []
    coalescer = StreamCoalescer(emitted.append, asyncio.get_running_loop())

    coalescer.feed(
        AppendText(
            key="assistant-a",
            kind="assistant",
            source=ROOT_SOURCE,
            block=0,
            fragment="read",
        ),
        1,
    )
    coalescer.feed(
        AppendText(
            key="assistant-a",
            kind="assistant",
            source=ROOT_SOURCE,
            block=0,
            fragment="  a.py",
        ),
        1,
    )
    coalescer.flush()

    assert len(emitted) == 1
    mutation = emitted[0].mutation
    assert isinstance(mutation, AppendText)
    assert mutation.fragment == "read  a.py"
    assert mutation.key == "assistant-a"


def _bar_pos(transcript: Transcript, row: int = 0, *, from_bottom: int = 12) -> QPoint:
    """A point on the left gutter, low enough to miss the header band."""

    rect = transcript.visualRect(transcript.model().index(row, 0))
    return QPoint(rect.left() + 12, rect.bottom() - from_bottom)


def test_transcript_bar_click_collapses_an_expanded_row(qtbot) -> None:
    transcript = _shown(qtbot)
    append_entry(
        transcript,
        activity_entry("read", summary="a.py", call_text="\n".join(f"line {i}" for i in range(40))),
    )
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript))

    assert transcript.records[0].expanded is True

    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_bar_pos(transcript))

    assert transcript.records[0].expanded is False


def test_transcript_bar_click_expands_a_collapsed_row(qtbot) -> None:
    transcript = _shown(qtbot)
    append_entry(transcript, activity_entry("read", summary="a.py", call_text="path: a.py"))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=_bar_pos(transcript, from_bottom=8),
    )

    assert transcript.records[0].expanded is True


def test_transcript_bar_click_leaves_dialogue_rows_alone(qtbot) -> None:
    transcript = _shown(qtbot)
    append_text(transcript, "user", "a question\n" + "\n".join(f"line {i}" for i in range(20)))
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_bar_pos(transcript))

    assert transcript.records[0].expanded is False


def test_transcript_tracks_the_hovered_bar_row(qtbot) -> None:
    transcript = _shown(qtbot)
    append_entry(
        transcript,
        activity_entry("read", summary="a.py", call_text="\n".join(f"line {i}" for i in range(40))),
    )
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript))

    assert transcript.hovered_bar_row is None

    transcript._sync_cursor(_bar_pos(transcript))

    assert transcript.hovered_bar_row == 0

    rect = transcript.visualRect(transcript.model().index(0, 0))
    transcript._sync_cursor(QPoint(rect.center().x(), rect.bottom() - 12))

    assert transcript.hovered_bar_row is None


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_submit_reports_work_in_flight_before_the_worker_thread_starts_it(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_idle()`` must go false when work is submitted, not when the worker picks it up.

    ``wait_idle`` in ``tests/gui/qtutil.py`` waits for the idle edge only. If the barrier
    reported idle during the window between ``submit()`` returning on the UI thread and
    the worker thread entering ``_tracked``, every end-to-end test could sail straight
    past pending work -- which is exactly how the history pagination test flaked.
    """

    bridge = KimixBridge()
    bridge.start()
    try:
        qtbot.waitUntil(bridge.is_idle, timeout=5_000)

        scheduled: list[object] = []
        monkeypatch.setattr(
            asyncio, "run_coroutine_threadsafe", lambda coro, _loop: scheduled.append(coro)
        )

        async def _work() -> None:
            return None

        bridge.submit(_work())
        assert scheduled, "submit() must hand the coroutine to the worker loop"
        assert bridge.is_idle() is False
    finally:
        monkeypatch.undo()
        for coro in scheduled:
            coro.close()
        bridge.stop()
