"""Modal dialogs for approvals, questions, and session deletion."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.qt import keys
from kimix_gui.qt.components import DialogFooter
from kimix_gui.qt.styling import Role, Tone, Variant, style


class ApprovalDialog(QDialog):
    """Resolve a SDK approval or hook request."""

    decided = Signal(str)

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("approval-dialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 280)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setObjectName("dialog-title")
        # A heading that rendered at body size and weight, so the dialog opened with
        # no visual entry point. ``section`` is the same treatment the history pane
        # heading gets: emphasis without the accent colour a brand title would take.
        style(heading, role=Role.SECTION)
        heading.setWordWrap(True)
        body = QTextEdit()
        # Not ``dialog-body``: that name also sat on the question dialog's plain
        # ``QLabel``, so one hook meant two widget classes and no per-name style could
        # apply to either. This one is the scrollable payload of a tool call.
        body.setObjectName("approval-payload")
        body.setReadOnly(True)
        body.setPlainText(description)
        layout.addWidget(heading)
        layout.addWidget(body)
        reject = QPushButton(self.tr("Reject"))
        reject.setObjectName("reject")
        style(reject, variant=Variant.DANGER)
        session = QPushButton(self.tr("Approve session"))
        session.setObjectName("approve-for-session")
        approve = QPushButton(self.tr("Approve"))
        approve.setObjectName("approve")
        style(approve, variant=Variant.PRIMARY)
        layout.addWidget(
            DialogFooter(dismiss=reject, extra=(session,), confirm=approve, parent=self)
        )
        reject.clicked.connect(lambda: self._choose("reject"))
        session.clicked.connect(lambda: self._choose("approve_for_session"))
        approve.clicked.connect(lambda: self._choose("approve"))
        # A / S / R stay bound to these three keys in every language: they are muscle
        # memory, they are in the README key table, and rebinding per locale would make
        # the documentation wrong. What does not survive translation is the *hint* --
        # "批准" does not start with A -- so the Chinese copy of the three button labels
        # carries the letter explicitly ("批准 (A)"). That is a translation decision, so
        # the English msgids stay clean and the letters live in the catalog.
        keys.install(
            self,
            keys.APPROVAL,
            {
                "approve": lambda: self._choose("approve"),
                "approve-for-session": lambda: self._choose("approve_for_session"),
                "reject": lambda: self._choose("reject"),
            },
            context=Qt.ShortcutContext.WindowShortcut,
        )

    def _choose(self, decision: str) -> None:
        self.decided.emit(decision)
        self.accept()


class QuestionDialog(QDialog):
    """Collect a free-form answer for a public SDK question request."""

    answered = Signal(object)

    def __init__(self, prompt: str, body: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("question-dialog")
        self.setWindowTitle(self.tr("Question"))
        self.setModal(True)
        self.resize(520, 240)
        layout = QVBoxLayout(self)
        heading = QLabel(prompt)
        heading.setObjectName("dialog-title")
        style(heading, role=Role.SECTION)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        if body:
            detail = QLabel(body)
            detail.setObjectName("question-detail")
            detail.setWordWrap(True)
            layout.addWidget(detail)
        self._answer = QLineEdit()
        self._answer.setObjectName("answer")
        self._answer.setPlaceholderText(self.tr("Type an option label or a free-form answer"))
        layout.addWidget(self._answer)
        self._resolved = False
        self._answer.returnPressed.connect(self._submit)
        self._answer.setFocus()

    def _submit(self) -> None:
        text = self._answer.text().strip()
        if not text:
            return
        self._emit_answer(text)
        self.accept()

    def reject(self) -> None:  # type: ignore[override]
        self._emit_answer(None)
        super().reject()

    def _emit_answer(self, value: object) -> None:
        if self._resolved:
            return
        self._resolved = True
        self.answered.emit(value)


class DeleteSessionsDialog(QDialog):
    """Confirm permanent deletion of one or more sessions."""

    def __init__(self, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("delete-dialog")
        self.setModal(True)
        # Spelled-out singular / plural instead of ``%n``: the msgid is what renders
        # when no catalog is installed (a fresh clone, and the test suite), so
        # ``%n session(s)`` would put a literal "(s)" in the title bar.
        question = (
            self.tr("Delete 1 session?")
            if count == 1
            else self.tr("Delete {count} sessions?").format(count=count)
        )
        self.setWindowTitle(question)
        layout = QVBoxLayout(self)
        title = QLabel(question)
        title.setObjectName("delete-title")
        style(title, role=Role.SECTION)
        copy = QLabel(
            self.tr("Conversation history and session files will be permanently removed.")
        )
        copy.setObjectName("delete-copy")
        # Subtext under a heading is muted everywhere else in the app
        # (``preferences-description``, ``detail-state``, ``settings-scope``).
        style(copy, tone=Tone.MUTED)
        copy.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(copy)
        cancel = QPushButton(self.tr("Cancel"))
        cancel.setObjectName("cancel-delete")
        confirm = QPushButton(self.tr("Delete"))
        confirm.setObjectName("confirm-delete")
        style(confirm, variant=Variant.DANGER)
        layout.addWidget(DialogFooter(dismiss=cancel, confirm=confirm, parent=self))
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
