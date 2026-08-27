"""Facts the test suite's own fixtures depend on.

These are not tests of the product. They pin the Qt behaviour that
``conftest.py`` is built on, so that a fixture whose premise stops holding fails
here with a name that says what happened, instead of showing up as "the suite got
four times slower for no visible reason".
"""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget


def test_process_events_does_not_deliver_deferred_deletes(qapp: QApplication) -> None:
    """``processEvents()`` leaves ``deleteLater()`` widgets alive. Qt documents this.

    ``pytest-qt`` retires widgets with ``close()`` + ``deleteLater()`` and then calls
    ``processEvents()``, which reads like a full cleanup and is not one: Qt only
    delivers ``DeferredDelete`` when the event loop that posted it unwinds. Nothing
    in this suite runs such a loop, so without the explicit flush in
    ``_flush_deferred_widget_deletions`` the widgets accumulate for the whole
    session -- and ``QApplication.setStyleSheet`` is linear in the live population.
    """

    before = len(QApplication.allWidgets())
    widget = QWidget()
    QLabel("child", widget)
    assert len(QApplication.allWidgets()) == before + 2

    widget.close()
    widget.deleteLater()
    qapp.processEvents()

    assert len(QApplication.allWidgets()) == before + 2, "still alive, as Qt intends"

    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert len(QApplication.allWidgets()) == before, "the flush is what reclaims them"


def test_the_flush_runs_before_every_test(qapp: QApplication) -> None:
    """The population a test inherits is bounded, whatever ran before it.

    A loose bound on purpose: the point is that it is bounded at all. Before the
    flush existed this number reached 20,780 partway through the session.
    """

    assert len(QApplication.allWidgets()) < 500, len(QApplication.allWidgets())
