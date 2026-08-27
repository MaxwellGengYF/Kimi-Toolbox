"""The row of buttons that closes a dialog, and the Return key policy with it."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

from kimix_gui.design import DARK
from kimix_gui.qt.styling import VARIANT, Variant


class DialogFooter(QWidget):
    """The buttons that close a dialog: one order, and one deliberate default.

    Four dialogs grew their own footer and none of them agreed. Three ended with
    the button that performs the action and ``LLMSettingsDialog`` ended with
    ``Close``, so the corner of the dialog meant "do it" in some places and
    "never mind" in others.

    The Return key was worse. ``QPushButton`` inside a ``QDialog`` defaults to
    ``autoDefault``, and Qt promotes the first such button it finds to be the
    default one, so the button Return activates was whichever the dialog
    happened to construct first:

    ==========================  =========================  ====================
    dialog                      button Qt promoted         what Return did
    ==========================  =========================  ====================
    ``ApprovalDialog``          ``approve``                approved (deliberate)
    ``DeleteSessionsDialog``    ``cancel-delete``          cancelled (luck)
    ``PreferencesDialog``       ``manage-llm-settings``    nothing, or opened a
                                                           nested dialog -- it
                                                           sits on a hidden page
                                                           of the stack, and
                                                           saving was
                                                           unreachable
    ``LLMSettingsDialog``       ``browse-config``          opened a file picker
    ==========================  =========================  ====================

    Arguments name what a button *means* rather than where it goes. Only two
    positions carry meaning -- the way out is leftmost, the action is rightmost --
    so anything else goes between them:

    ``dismiss``
        Leaves the dialog alone. Required: Escape is wired separately, but a
        dialog still needs a visible way out.
    ``confirm``
        Performs the thing the dialog asks about. Absent for a dialog that only
        reports, such as the read-only config viewer.
    ``extra``
        Neither: a second way to say yes, or an action on the selection rather
        than on the dialog. Keeps the order the caller gives.
    """

    def __init__(
        self,
        *,
        dismiss: QPushButton,
        confirm: QPushButton | None = None,
        extra: Sequence[QPushButton] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dialog-footer")
        # A row of buttons has one right height. Left at the default policy the
        # footer accepts a share of whatever vertical slack the dialog has, and it
        # takes that space from the content above it -- the delete confirmation's
        # two lines of copy lost 35px between them to a footer growing into the
        # gap. The bare layouts this replaces did not have that problem, because a
        # layout is not a widget and cannot be stretched.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._dismiss = dismiss
        self._confirm = confirm
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # The gap every other row of controls uses -- the home header, the chat
        # toolbar, the detail actions. The footers had 6 (the platform default,
        # nobody's choice) and 16 (inherited from the preferences dialog's root
        # spacing, meant for the gap between sections, not between buttons).
        layout.setSpacing(DARK.spacing.md)
        layout.addStretch()
        layout.addWidget(dismiss)
        for button in extra:
            layout.addWidget(button)
        if confirm is not None:
            layout.addWidget(confirm)
        self._claimed = False

    @property
    def default_button(self) -> QPushButton:
        """The button Return activates.

        Confirming, unless confirming destroys something. ``DeleteSessionsDialog``
        confirms with a ``danger`` button, and a dialog that deletes sessions on
        Return is a dialog that deletes sessions by accident, so there the way out
        takes the key and deleting stays a deliberate click.
        """
        confirm = self._confirm
        if confirm is None or confirm.property(VARIANT) == Variant.DANGER:
            return self._dismiss
        return confirm

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        if not self._claimed:
            self._claimed = True
            self._claim_the_return_key()

    def _claim_the_return_key(self) -> None:
        """Take ``autoDefault`` away from every other button in the dialog.

        Setting the default is not enough on its own: Qt re-promotes whichever
        ``autoDefault`` button holds focus, so tabbing to ``Remove`` would quietly
        make Return mean "remove". Clearing the flag leaves Return meaning one
        thing and Space meaning "the button I am on", which is the convention
        users already have.

        Done on first show rather than in ``__init__`` because a footer is built
        before or after the rest of the dialog depending on the dialog, and by the
        time it is shown every button exists either way.
        """
        chosen = self.default_button
        for button in self.window().findChildren(QPushButton):
            # ``qt_`` names belong to Qt's own internals, such as a line edit's
            # clear button; their focus behaviour is not ours to change.
            if button.objectName().startswith("qt_"):
                continue
            button.setAutoDefault(button is chosen)
            button.setDefault(button is chosen)
