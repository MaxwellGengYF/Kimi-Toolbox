"""Guards for the shared components in ``kimix_gui.qt.components``.

The point of a component is that a decision is made once. These tests are the
other half of that: they fail when a decision starts being made somewhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.llm_config import inspect_llm_config
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.components import Card, DialogFooter, KeyValueList
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog
from kimix_gui.qt.styling import CARD, TONE, CardLevel, Tone


def _provider_config(tmp_path: Path) -> Path:
    path = tmp_path / "provider.json"
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    return path


def _dialogs(tmp_path: Path) -> dict[str, QDialog]:
    """Every dialog in the app, keyed by the name used in the tables below."""

    reference = inspect_llm_config(_provider_config(tmp_path))
    return {
        "approval": ApprovalDialog("Run bash", "ls -la"),
        "question": QuestionDialog("Which branch?", "main or dev"),
        "delete-sessions": DeleteSessionsDialog(3),
        "preferences": PreferencesDialog(InterfacePreferences(), font_families=list),
        "llm-settings": LLMSettingsDialog(
            current=reference,
            references=(reference,),
            scope_label="New session",
            manage_library=True,
        ),
        "llm-settings-read-only": LLMSettingsDialog(
            current=reference,
            references=(reference,),
            scope_label="Active session",
            read_only=True,
        ),
    }


@pytest.fixture
def dialogs(qtbot, tmp_path: Path) -> dict[str, QDialog]:
    built = _dialogs(tmp_path)
    for dialog in built.values():
        qtbot.addWidget(dialog)
        dialog.show()
    return built


def _buttons(widget: QWidget) -> list[QPushButton]:
    return [
        button
        for button in widget.findChildren(QPushButton)
        if not button.objectName().startswith("qt_")
    ]


#: The button Return activates, per dialog. Every entry here was wrong before the
#: footer existed except ``approval``: ``QPushButton`` in a ``QDialog`` defaults to
#: ``autoDefault``, so Qt promoted whichever button the dialog happened to build
#: first, and nothing said otherwise.
#:
#: The two destructive dialogs deliberately point at the way out. A confirmation
#: whose Return key performs the deletion is a confirmation that does not confirm
#: anything.
RETURN_ACTIVATES: dict[str, str] = {
    "approval": "approve",
    "delete-sessions": "cancel-delete",
    "preferences": "apply-preferences",
    "llm-settings": "apply-settings",
    "llm-settings-read-only": "close-settings",
}


@pytest.mark.parametrize("name", sorted(RETURN_ACTIVATES))
def test_return_activates_the_intended_button(dialogs: dict[str, QDialog], name: str) -> None:
    """Before the footer: Return opened a file picker in the config dialog."""

    dialog = dialogs[name]
    defaults = [button.objectName() for button in _buttons(dialog) if button.isDefault()]
    assert defaults == [RETURN_ACTIVATES[name]]


@pytest.mark.parametrize("name", sorted(RETURN_ACTIVATES))
def test_no_other_button_can_steal_the_return_key(dialogs: dict[str, QDialog], name: str) -> None:
    """Setting the default is not enough on its own.

    Qt re-promotes whichever ``autoDefault`` button holds focus, so tabbing onto
    ``Remove`` would quietly redefine Return. Clearing the flag everywhere else
    leaves Return meaning one thing and Space meaning "the button I am on".
    """

    dialog = dialogs[name]
    keeping = sorted(button.objectName() for button in _buttons(dialog) if button.autoDefault())
    assert keeping == [RETURN_ACTIVATES[name]]


def test_return_in_the_preferences_dialog_saves(qtbot, dialogs: dict[str, QDialog]) -> None:
    """It used to open the nested LLM dialog, or do nothing.

    ``manage-llm-settings`` sits on a hidden page of the category stack and was
    built before the footer, so Qt made it the default button: Return did nothing
    on the Appearance page, opened another dialog on the Models page, and never
    once saved. Saving was unreachable from the keyboard.
    """

    dialog = dialogs["preferences"]
    saved: list[object] = []
    opened: list[object] = []
    dialog.applied.connect(saved.append)
    dialog.manage_llm.connect(lambda: opened.append(True))
    qtbot.keyClick(dialog, Qt.Key.Key_Return)
    assert opened == []
    assert len(saved) == 1


def test_return_in_the_delete_dialog_does_not_delete(qtbot, dialogs: dict[str, QDialog]) -> None:
    dialog = dialogs["delete-sessions"]
    qtbot.keyClick(dialog, Qt.Key.Key_Return)
    assert dialog.result() == QDialog.DialogCode.Rejected


#: Button order per dialog, left to right. The way out is leftmost, the action is
#: rightmost, anything else sits between them. ``LLMSettingsDialog`` used to read
#: ``Remove, Use config, Close`` -- the only footer in the app that did not end
#: with the action, so its corner meant "never mind" while every other corner
#: meant "do it".
FOOTER_ORDER: dict[str, list[str]] = {
    "approval": ["reject", "approve-for-session", "approve"],
    "delete-sessions": ["cancel-delete", "confirm-delete"],
    "preferences": ["cancel-preferences", "apply-preferences"],
    "llm-settings": ["cancel-settings", "delete-config", "apply-settings"],
    "llm-settings-read-only": ["close-settings"],
}


@pytest.mark.parametrize("name", sorted(FOOTER_ORDER))
def test_the_footer_puts_the_action_in_the_corner(dialogs: dict[str, QDialog], name: str) -> None:
    footer = dialogs[name].findChild(DialogFooter)
    assert footer is not None
    order = [button.objectName() for button in _buttons(footer)]
    assert order == FOOTER_ORDER[name]


#: Buttons that are not footer buttons, and why. Anything else in a dialog has to
#: live in the footer, which is what keeps the Return key policy from being
#: bypassed by the next dialog that grows a button.
BUTTONS_OUTSIDE_THE_FOOTER: dict[str, str] = {
    "browse-config": "opens a file picker for the path field it sits beside",
    "connect-chatgpt": "account-card action: starts the global OAuth dialog",
    "connect-chatgpt-models": "source-pane action: starts the global OAuth dialog",
    "disconnect-chatgpt": "account-card action: disconnects the global account",
    "load-config": "acts on the path field it sits beside",
    "manage-llm-settings": "page content: opens another dialog from the Models page",
    "refresh-codex-models": "account-card action: refreshes its model catalog",
}


@pytest.mark.parametrize("name", sorted(FOOTER_ORDER))
def test_every_dialog_button_is_either_in_the_footer_or_declared(
    dialogs: dict[str, QDialog], name: str
) -> None:
    dialog = dialogs[name]
    footer = dialog.findChild(DialogFooter)
    stray = sorted(
        button.objectName()
        for button in _buttons(dialog)
        if (footer is None or not footer.isAncestorOf(button))
        and button.objectName() not in BUTTONS_OUTSIDE_THE_FOOTER
    )
    assert stray == [], stray


def test_no_dialog_declares_a_footer_button_it_does_not_have(
    dialogs: dict[str, QDialog],
) -> None:
    """The register shrinks with the code, or it turns into fiction."""

    present = {button.objectName() for dialog in dialogs.values() for button in _buttons(dialog)}
    assert sorted(set(BUTTONS_OUTSIDE_THE_FOOTER) - present) == []


def test_the_footer_does_not_grow_into_the_content_above_it(qtbot) -> None:
    """A bare layout cannot be stretched; a widget can, and this one was.

    Wrapping the buttons in a widget handed the footer a share of the dialog's
    spare vertical space, and it took that space from the copy above: the delete
    confirmation's two lines lost 35px between them.
    """

    host = QDialog()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    filler = QLabel("content")
    filler.setWordWrap(True)
    footer = DialogFooter(dismiss=QPushButton("Cancel"), confirm=QPushButton("OK"))
    layout.addWidget(filler)
    layout.addWidget(footer)
    host.resize(400, 600)
    host.show()
    qtbot.waitExposed(host)

    assert footer.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert footer.height() == footer.sizeHint().height()


def test_a_destructive_confirmation_leaves_the_return_key_on_the_way_out(qtbot) -> None:
    """The rule, stated on its own so it survives the dialogs changing."""

    from kimix_gui.qt.styling import Variant, style

    dismiss = QPushButton("Cancel")
    destructive = QPushButton("Delete")
    style(destructive, variant=Variant.DANGER)
    footer = DialogFooter(dismiss=dismiss, confirm=destructive)
    qtbot.addWidget(footer)
    assert footer.default_button is dismiss

    safe = QPushButton("Save")
    style(safe, variant=Variant.PRIMARY)
    ordinary = DialogFooter(dismiss=QPushButton("Cancel"), confirm=safe)
    qtbot.addWidget(ordinary)
    assert ordinary.default_button is safe


def test_the_key_column_sizes_itself_to_the_labels(qtbot) -> None:
    """It used to be ``setFixedWidth(110)``, which is a translation waiting to break.

    A pinned column either clips a longer word or wastes the room a shorter one
    does not need, and which of those happens depends on the catalog installed.
    """

    short = KeyValueList((("a", "ID"),))
    wide = KeyValueList((("a", "A considerably longer field name"),))
    for widget in (short, wide):
        qtbot.addWidget(widget)
        widget.resize(600, 80)
        widget.show()
        qtbot.waitExposed(widget)

    assert wide.values["a"].x() > short.values["a"].x()
    for widget in (short, wide):
        key = widget.findChild(QLabel, "key-value-key")
        assert key is not None
        assert key.maximumWidth() > key.sizeHint().width()


def test_the_key_column_is_muted_and_the_value_is_readable(qtbot) -> None:
    widget = KeyValueList((("detail-id", "Session ID"),))
    qtbot.addWidget(widget)
    key = widget.findChild(QLabel, "key-value-key")
    value = widget.values["detail-id"]
    assert key is not None
    assert key.property(TONE) == Tone.MUTED
    assert value.property(TONE) is None
    # A session id you cannot copy is a session id you have to retype.
    assert value.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    # The name is not focusable, so it cannot act as a buddy; without this a
    # screen reader reads a column of bare values.
    assert value.accessibleName() == "Session ID"


def test_a_key_value_row_is_one_form_row(qtbot) -> None:
    """Hand-built ``QHBoxLayout`` rows do not line up; form rows share a column."""

    widget = KeyValueList((("a", "One"), ("b", "Two")))
    qtbot.addWidget(widget)
    form = widget.findChild(QFormLayout)
    assert form is not None
    assert form.rowCount() == 2


CARD_PADDING: dict[str, str] = {
    CardLevel.FLOATING: "wide",
    CardLevel.PANEL: "detail",
    CardLevel.INSET: "compact",
}


@pytest.mark.parametrize("level", sorted(CARD_PADDING))
def test_a_card_takes_its_padding_from_the_level(qtbot, level: str) -> None:
    from kimix_gui.design import DARK

    card = Card(level)
    qtbot.addWidget(card)
    expected = getattr(DARK.card_padding, CARD_PADDING[level])
    margins = card.body.contentsMargins()
    actual = (margins.left(), margins.top(), margins.right(), margins.bottom())
    assert actual == expected
    assert card.property(CARD) == level


def test_every_card_level_is_reachable_from_the_stylesheet() -> None:
    """A level with no rule is a card with no background."""

    from kimix_gui.qt.theme import build_stylesheet

    stylesheet = build_stylesheet()
    for level in (CardLevel.FLOATING, CardLevel.PANEL, CardLevel.INSET):
        assert f'QFrame[card="{level}"]' in stylesheet


def test_the_level_is_what_paints_the_card(qtbot, qapp) -> None:
    """The string being in the sheet is not the same as the sheet reaching the card.

    ``QFrame[card=...]`` is one point of specificity plus one attribute; the base
    ``QWidget`` rule that makes everything transparent is one. If that ever
    inverts, the cards go transparent and nothing else changes, which is the kind
    of regression a screenshot review misses.
    """

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QRegion

    from kimix_gui.design import DARK
    from kimix_gui.qt.theme import build_stylesheet

    qapp.setStyleSheet(build_stylesheet(DARK))
    painted: dict[str, str] = {}
    for level in (CardLevel.FLOATING, CardLevel.PANEL, CardLevel.INSET):
        card = Card(level)
        qtbot.addWidget(card)
        card.resize(120, 80)
        card.show()
        qtbot.waitExposed(card)
        image = QImage(card.size(), QImage.Format.Format_ARGB32)
        image.fill(0)
        card.render(image, QPoint(), QRegion(), card.RenderFlag.DrawWindowBackground)
        painted[level] = QImage(image).pixelColor(60, 40).name()

    assert painted[CardLevel.PANEL] == DARK.palette.panel
    assert painted[CardLevel.FLOATING] == DARK.palette.surface
    assert painted[CardLevel.INSET] == DARK.palette.surface
