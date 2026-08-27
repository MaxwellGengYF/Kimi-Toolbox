"""Re-run the statements that put translated copy into a widget.

Qt posts ``LanguageChange`` after a translator is installed, but nothing in Qt
re-runs the ``setText(self.tr(...))`` calls that already happened. Widgets built
before the switch keep the copy they were born with.

The usual answer is a hand-written ``retranslateUi()`` per widget, which states
every piece of copy twice: once where the widget is built and once in the handler.
Two lists that must agree, and the only symptom of them disagreeing is one label
staying in the old language.

This module keeps a single list. A call site changes from::

    title.setText(self.tr("Recent sessions"))

to::

    self._i18n.bind(lambda: title.setText(self.tr("Recent sessions")))

``bind`` runs the statement immediately -- so the widget is populated exactly as
before -- and runs it again on every ``LanguageChange``. The copy is written once,
and there is no second list to forget.

Two facts this rests on, both probed on PySide6 6.11 (see
``tests/gui/test_retranslate.py``):

* ``LanguageChange`` reaches **every** ``QObject`` under a widget, not just widgets,
  so ``Retranslator`` can be a plain ``QObject`` child and no host needs to override
  ``changeEvent``. An *unparented* ``QObject`` gets nothing.
* The event is **posted**, not sent. Nothing has been retranslated until the event
  loop turns, exactly like ``QApplication.setFont``. Tests must pump.

For text that is derived from state rather than fixed at construction ("12
sessions"), bind the method that already recomputes it. It will re-derive from
whatever the state is at the time, which is what makes this work for a list that
has been reloaded twenty times since the widget was built.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject


class Retranslator(QObject):
    """Holds re-runnable copy statements for one host widget.

    Parent it to the widget whose copy it owns. It has no other state, and its
    lifetime is the host's, so bound closures may safely capture the host's
    children.
    """

    def __init__(self, host: QObject) -> None:
        super().__init__(host)
        self._statements: list[Callable[[], None]] = []

    def bind(self, statement: Callable[[], None]) -> None:
        """Run ``statement`` now, and again whenever the language changes."""

        self._statements.append(statement)
        statement()

    def retranslate(self) -> None:
        """Re-run every bound statement. Called on ``LanguageChange``."""

        for statement in self._statements:
            statement()

    @property
    def statement_count(self) -> int:
        """How many statements are bound. For tests and for spotting leaks."""

        return len(self._statements)

    def event(self, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate()
        return super().event(event)


__all__ = ["Retranslator"]
