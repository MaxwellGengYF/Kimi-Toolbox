from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea, QWidget

from kimix_gui.design import DARK
from kimix_gui.qt.theme import build_stylesheet
from kimix_gui.qt.todo_panel import CARD_WIDTH, MARGIN, TodoPanel, _TodoRow
from kimix_gui.todos import EMPTY_SNAPSHOT, TodoEntry, TodoSnapshot


def _snapshot(*entries: TodoEntry, archived: int = 0) -> TodoSnapshot:
    return TodoSnapshot(entries=entries, archived=archived)


def _host(qtbot) -> tuple[QWidget, TodoPanel]:
    host = QWidget()
    host.resize(900, 600)
    panel = TodoPanel(host)
    qtbot.addWidget(host)
    host.show()
    return host, panel


def _scrolling_host(qtbot, *, overflow: bool) -> tuple[QScrollArea, TodoPanel]:
    """A scroll area host, with or without enough content to show a scrollbar."""

    host = QScrollArea()
    host.resize(900, 600)
    inner = QWidget()
    inner.resize(400, 4000 if overflow else 100)
    host.setWidget(inner)
    panel = TodoPanel(host)
    qtbot.addWidget(host)
    host.show()
    QApplication.processEvents()
    return host, panel


def _rows(panel: TodoPanel) -> list[_TodoRow]:
    return panel.findChildren(_TodoRow)


def test_panel_hidden_until_todos_arrive(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    assert panel.isVisible() is False

    panel.set_snapshot(_snapshot(TodoEntry("Write the panel", "in_progress")))
    assert panel.isVisible() is True
    assert panel.expanded is True
    assert len(_rows(panel)) == 1


def test_panel_hides_again_when_todos_are_cleared(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    panel.set_snapshot(EMPTY_SNAPSHOT)
    assert panel.isVisible() is False
    assert _rows(panel) == []


def test_panel_anchors_to_top_right_of_host(qtbot) -> None:
    host, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    geometry = panel.geometry()
    assert geometry.width() == CARD_WIDTH
    assert geometry.top() == MARGIN
    assert host.rect().right() - geometry.right() == MARGIN


def test_collapse_shrinks_to_a_pill_and_hides_the_body(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending"), TodoEntry("b", "done")))
    expanded_height = panel.height()

    panel.toggle()
    assert panel.expanded is False
    assert panel.property("mode") == "pill"
    assert panel.width() < CARD_WIDTH
    assert panel.height() < expanded_height
    footer = panel.findChild(QLabel, "todo-footer")
    assert footer is not None
    assert footer.isVisible() is False

    panel.toggle()
    assert panel.expanded is True
    assert panel.property("mode") == "card"
    assert panel.height() == expanded_height


def test_collapsed_panel_stays_collapsed_across_updates(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    panel.toggle()

    panel.set_snapshot(_snapshot(TodoEntry("a", "in_progress")))
    assert panel.expanded is False
    assert panel.property("flash") is True

    panel.set_snapshot(EMPTY_SNAPSHOT)
    panel.set_snapshot(_snapshot(TodoEntry("b", "pending")))
    assert panel.expanded is False, "a manual collapse must survive a session's todo reset"


def test_header_reflects_progress_and_summary(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(
        _snapshot(
            TodoEntry("done one", "done"),
            TodoEntry("doing", "in_progress", "halfway"),
            TodoEntry("later", "pending"),
            TodoEntry("nested", "pending", depth=1),
            archived=2,
        )
    )
    count = panel.findChild(QLabel, "todo-count")
    footer = panel.findChild(QLabel, "todo-footer")
    dot = panel.findChild(QLabel, "todo-dot")
    assert count is not None and footer is not None and dot is not None
    assert count.text() == "1/4"
    assert footer.text() == "1 in progress · 2 pending · 1 done · 2 archived"
    assert dot.property("state") == "in_progress"
    assert "Now: doing" in panel._header.toolTip()


def test_all_done_marks_the_panel_complete(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "done"), TodoEntry("b", "done")))
    dot = panel.findChild(QLabel, "todo-dot")
    count = panel.findChild(QLabel, "todo-count")
    assert dot is not None and count is not None
    assert dot.property("state") == "done"
    assert count.text() == "2/2"
    assert "All 2 todos done" in panel._header.toolTip()


def test_rows_carry_status_and_indentation(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(
        _snapshot(
            TodoEntry("parent", "in_progress"),
            TodoEntry("child", "pending", "with notes", depth=1),
        )
    )
    rows = _rows(panel)
    assert [row.property("state") for row in rows] == ["in_progress", "pending"]
    parent_indent = rows[0].layout().contentsMargins().left()
    child_indent = rows[1].layout().contentsMargins().left()
    assert child_indent > parent_indent
    assert rows[1].height() > rows[0].height(), "a note adds a second line"
    assert rows[1].toolTip() == "child\nwith notes"


def test_repeated_identical_snapshot_does_not_rebuild(qtbot) -> None:
    _host_widget, panel = _host(qtbot)
    snapshot = _snapshot(TodoEntry("a", "pending"))
    panel.set_snapshot(snapshot)
    first = _rows(panel)[0]
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    assert _rows(panel)[0] is first


def test_relayout_clamps_to_a_narrow_host(qtbot) -> None:
    host, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    host.resize(260, 200)
    panel.relayout()
    assert panel.width() <= 260 - 2 * MARGIN
    assert panel.height() <= 200 - 2 * MARGIN


def test_a_done_entry_is_struck_through_by_the_stylesheet(qtbot) -> None:
    """The strike-through used to be a hand-built QFont on the label.

    Setting a font also pins it, so a done row stopped following the interface font
    preference. The declaration reaches the same font flag, and the row keeps
    inheriting its size -- which is what the second half of this test pins.
    """

    app = QApplication.instance()
    assert app is not None
    app.setStyleSheet(build_stylesheet(DARK))
    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(
        _snapshot(TodoEntry("done thing", "done"), TodoEntry("todo thing", "pending"))
    )
    titles = [
        label
        for row in _rows(panel)
        for label in row.findChildren(QLabel)
        if label.objectName() == "todo-item-title"
    ]
    assert len(titles) == 2
    for title in titles:
        title.ensurePolished()
    assert [title.font().strikeOut() for title in titles] == [True, False]
    assert not any(title.testAttribute(Qt.WidgetAttribute.WA_SetFont) for title in titles)


def test_the_panel_title_is_one_msgid_cased_for_the_two_shapes(qtbot) -> None:
    """The expanded overline reads as caps, the pill does not.

    That used to be two translatable strings a translator had to keep in sync for a
    distinction most languages do not have. The case is a presentation transform now,
    applied after the lookup.
    """

    _host_widget, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    title = panel.findChild(QLabel, "todo-title")
    assert title is not None
    expanded = title.text()
    panel.set_expanded(False)
    collapsed = title.text()

    assert expanded == "TODOS"
    assert collapsed == "Todos"
    assert expanded == collapsed.upper(), "one msgid, two casings"


def test_the_panel_re_anchors_when_its_host_resizes(qtbot) -> None:
    """Nobody has to tell it. The host used to forward three events for the panel.

    A host that has to know the panel's anchoring rules cannot be swapped for another
    widget without repeating them, and forgetting one is invisible: the panel simply
    stays where the old geometry put it.
    """

    host, panel = _host(qtbot)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))

    host.resize(600, 400)
    QApplication.processEvents()

    assert host.rect().right() - panel.geometry().right() == MARGIN
    assert panel.geometry().top() == MARGIN


def test_the_panel_dodges_a_visible_scrollbar(qtbot) -> None:
    """It anchors to the viewport, so the scrollbar's width is already accounted for."""

    host, panel = _scrolling_host(qtbot, overflow=True)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))

    bar = host.verticalScrollBar()
    assert bar.isVisible() is True
    viewport = host.viewport().geometry()
    assert viewport.right() - panel.geometry().right() == MARGIN
    assert panel.geometry().top() == viewport.top() + MARGIN
    # And it really is clear of the scrollbar, not merely inside the host.
    assert panel.geometry().right() + MARGIN <= host.rect().right() - bar.width()


def test_a_hidden_scrollbar_takes_no_width_from_the_panel(qtbot) -> None:
    """The counterpart: a hidden ``QScrollBar`` still reports a width (measured: 100).

    Subtracting it unconditionally would leave the panel floating far short of the
    edge, which is why the viewport is the thing to ask.
    """

    host, panel = _scrolling_host(qtbot, overflow=False)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))

    assert host.verticalScrollBar().isVisible() is False
    assert host.viewport().geometry().right() - panel.geometry().right() == MARGIN


def test_the_panel_re_anchors_when_the_scrollbar_appears(qtbot) -> None:
    """A scrollbar appearing narrows the viewport without resizing the host."""

    host, panel = _scrolling_host(qtbot, overflow=False)
    panel.set_snapshot(_snapshot(TodoEntry("a", "pending")))
    before = panel.geometry().right()

    host.widget().resize(400, 4000)
    QApplication.processEvents()

    assert host.verticalScrollBar().isVisible() is True
    assert panel.geometry().right() < before
    assert host.viewport().geometry().right() - panel.geometry().right() == MARGIN
