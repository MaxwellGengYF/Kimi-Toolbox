"""Guard the keyboard tables and the one mechanism that installs them.

Two invariants are worth spending tests on.

The table has to stay a *table*: every sequence must parse, no scope may bind the same
key twice, and ``qt.keys.SCOPES`` must list every table the module defines -- discovered
by reflection, because a hand-written list of scopes would pass while a new scope went
uninstalled and untested.

And the table has to stay *the* source: each page's shortcuts are compared against it by
key sequence, so deleting a row without touching a view (or the reverse) fails here
rather than in the product. ``install`` refusing mismatched handlers covers the third
direction, where a row and a handler drift apart by name.

The behavioural tests all drive real key events instead of calling handlers, because the
thing being asserted is Qt's dispatch: that a binding fires only while the focus is
inside the page that owns it. Calling ``home.open_highlighted()`` would prove nothing
about the keyboard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QLineEdit, QStackedWidget, QVBoxLayout, QWidget

from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.llm_config import inspect_llm_config
from kimix_gui.qt import keys
from kimix_gui.qt.bridge import KimixBridge
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.request_dialogs import ApprovalDialog
from kimix_gui.session_index import SessionSummary

from .qtutil import reactivate_window

REPOSITORY = Path(__file__).resolve().parents[2]
USER_GUIDE = REPOSITORY / "docs" / "gui" / "user-guide.md"


def _tables() -> dict[str, tuple[keys.Binding, ...]]:
    """Every binding table the module defines, found by reflection."""
    found = {}
    for name in dir(keys):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(keys, name)
        if isinstance(value, tuple) and value and all(isinstance(x, keys.Binding) for x in value):
            found[name] = value
    return found


def _sequences(host: QWidget) -> list[str]:
    return sorted(
        child.key().toString() for child in host.children() if isinstance(child, QShortcut)
    )


def _expected(bindings: tuple[keys.Binding, ...]) -> list[str]:
    return sorted(binding.key.toString() for binding in bindings)


def _home(tmp_path: Path) -> HomeView:
    config = tmp_path / "provider.json"
    config.write_text(
        '{"model": "m", "name": "M", "max_context_size": 1000, '
        '"type": "openai_legacy", "url": "https://example.test/v1", "api_key": "k"}',
        encoding="utf-8",
    )
    home = HomeView(
        tmp_path,
        default_config=inspect_llm_config(config),
        session_config_loader=lambda _id: None,
    )
    home.resize(900, 600)
    home.show_sessions([SessionSummary(id="s-1", title="one", updated_at=1.0)])
    return home


def _press(
    target: QWidget,
    key: Qt.Key,
    text: str = "",
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Send a real key press, so Qt's shortcut map decides who handles it.

    ``text`` matters: a text widget looks at ``event.text()`` when it decides whether to
    veto a binding, so a synthesised press with an empty one is ignored and propagates,
    which looks exactly like the veto that is being tested for.
    """
    for kind in (QKeyEvent.Type.KeyPress, QKeyEvent.Type.KeyRelease):
        QApplication.sendEvent(target, QKeyEvent(kind, key, modifier, text))


# --- the tables themselves ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_tables()))
def test_every_sequence_parses(name: str) -> None:
    for binding in _tables()[name]:
        parsed = binding.key
        assert parsed.count() == 1, f"{name}: {binding.sequence!r} is not one key sequence"
        assert parsed.toString(), f"{name}: {binding.sequence!r} parsed to nothing"


@pytest.mark.parametrize("name", sorted(_tables()))
def test_no_scope_binds_the_same_sequence_twice(name: str) -> None:
    parsed = [binding.key.toString() for binding in _tables()[name]]
    assert len(parsed) == len(set(parsed)), f"{name} binds a key twice: {sorted(parsed)}"


@pytest.mark.parametrize("name", sorted(_tables()))
def test_action_names_are_stable_identifiers(name: str) -> None:
    for binding in _tables()[name]:
        assert re.fullmatch(r"[a-z]+(-[a-z]+)*", binding.action), binding.action


def test_scopes_lists_every_table() -> None:
    listed = {id(table) for table in keys.SCOPES.values()}
    for name, table in _tables().items():
        assert id(table) in listed, f"{name} is not in keys.SCOPES"


def test_actions_collapses_shared_sequences() -> None:
    # Return and the keypad Enter are one action under two sequences.
    assert len(keys.actions(keys.HOME)) < len(keys.HOME)
    assert "open-highlighted" in keys.actions(keys.HOME)


# --- install ---------------------------------------------------------------------


def test_install_rejects_a_missing_handler(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    table = (keys.Binding("F5", "refresh"), keys.Binding("F6", "reload"))
    with pytest.raises(ValueError, match="missing reload"):
        keys.install(host, table, {"refresh": lambda: None})


def test_install_rejects_an_unreachable_handler(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    with pytest.raises(ValueError, match="extra reload"):
        keys.install(
            host,
            (keys.Binding("F5", "refresh"),),
            {"refresh": lambda: None, "reload": lambda: None},
        )


def test_install_rejects_the_same_key_twice(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    # Spelled differently on purpose: the check compares the parsed key, not the text.
    table = (keys.Binding("Esc", "quit"), keys.Binding("Escape", "quit"))
    with pytest.raises(ValueError, match="bound twice"):
        keys.install(host, table, {"quit": lambda: None})


def test_install_scopes_and_names_every_row(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    installed = keys.install(host, keys.HOME, dict.fromkeys(keys.actions(keys.HOME), lambda: None))
    assert len(installed) == len(keys.HOME)
    assert [s.objectName() for s in installed] == [f"key:{b.action}" for b in keys.HOME]
    for shortcut in installed:
        assert shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_install_honours_an_explicit_context(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    installed = keys.install(
        host,
        (keys.Binding("F5", "refresh"),),
        {"refresh": lambda: None},
        context=Qt.ShortcutContext.WindowShortcut,
    )
    assert installed[0].context() == Qt.ShortcutContext.WindowShortcut


# --- the pages install their own table -------------------------------------------


def test_home_installs_the_home_table(qtbot, tmp_path: Path) -> None:
    home = _home(tmp_path)
    qtbot.addWidget(home)
    assert _sequences(home) == _expected(keys.HOME)


def test_chat_installs_the_chat_table(qtbot) -> None:
    chat = ChatView(KimixBridge())
    qtbot.addWidget(chat)
    assert _sequences(chat) == _expected(keys.CHAT)


def test_approval_dialog_installs_the_approval_table(qtbot) -> None:
    dialog = ApprovalDialog("Run a command", "cat file")
    qtbot.addWidget(dialog)
    assert _sequences(dialog) == _expected(keys.APPROVAL)


def test_the_window_owns_no_bindings(qtbot, tmp_path: Path) -> None:
    """Every key belongs to a page now. A window-wide one would fire from anywhere."""
    controller = KimixGuiApp(SessionOptions(tmp_path))
    window = controller.create_window()
    qtbot.addWidget(window)
    try:
        assert [s.objectName() for s in window.children() if isinstance(s, QShortcut)] == []
    finally:
        controller.shutdown()


# --- dispatch --------------------------------------------------------------------


def test_a_home_key_fires_while_the_focus_is_in_the_page(qtbot, tmp_path: Path) -> None:
    home = _home(tmp_path)
    qtbot.addWidget(home)
    home.show()
    qtbot.waitExposed(home)
    reactivate_window(home)
    with qtbot.waitSignal(home.new_session, timeout=1000):
        _press(QApplication.focusWidget() or home, Qt.Key.Key_N, "n")


def test_a_home_key_stays_quiet_while_another_page_has_the_focus(qtbot, tmp_path: Path) -> None:
    """The scoping test. F4 is bound on both pages and must not be ambiguous."""
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    home = _home(tmp_path)
    chat = ChatView(KimixBridge())
    stack.addWidget(home)
    stack.addWidget(chat)
    stack.setCurrentWidget(chat)
    stack.show()
    qtbot.waitExposed(stack)
    reactivate_window(stack)

    fired: list[str] = []
    home.configure_session.connect(lambda _id: fired.append("home"))
    chat.open_settings.connect(lambda: fired.append("chat"))
    chat.take_keyboard_focus()
    _press(QApplication.focusWidget() or chat, Qt.Key.Key_F4)
    assert fired == ["chat"]


def test_a_key_stays_quiet_while_focus_is_outside_the_page(qtbot, tmp_path: Path) -> None:
    outside = QWidget()
    qtbot.addWidget(outside)
    layout = QVBoxLayout(outside)
    home = _home(tmp_path)
    field = QLineEdit()
    layout.addWidget(home)
    layout.addWidget(field)
    outside.show()
    qtbot.waitExposed(outside)
    reactivate_window(outside)
    field.setFocus()
    assert QApplication.focusWidget() is field

    fired: list[str] = []
    home.configure_session.connect(lambda _id: fired.append("home"))
    _press(field, Qt.Key.Key_F4)
    assert fired == []


def test_a_text_field_keeps_a_plain_letter_for_itself(qtbot, tmp_path: Path) -> None:
    """``n`` must reach the search box, not start a session, while it has the focus."""
    home = _home(tmp_path)
    qtbot.addWidget(home)
    home.show()
    qtbot.waitExposed(home)
    reactivate_window(home)
    search = home.findChild(QLineEdit, "session-search")
    assert search is not None
    search.setFocus()

    fired: list[str] = []
    home.new_session.connect(lambda: fired.append("new"))
    _press(search, Qt.Key.Key_N, "n")
    assert fired == []
    assert search.text() == "n"


# --- ensure_focus ------------------------------------------------------------------


def test_ensure_focus_moves_focus_into_the_page(qtbot, tmp_path: Path) -> None:
    outside = QWidget()
    qtbot.addWidget(outside)
    layout = QVBoxLayout(outside)
    home = _home(tmp_path)
    field = QLineEdit()
    layout.addWidget(home)
    layout.addWidget(field)
    outside.show()
    qtbot.waitExposed(outside)
    reactivate_window(outside)
    field.setFocus()

    home.take_keyboard_focus()
    focused = QApplication.focusWidget()
    assert focused is not None
    assert home.isAncestorOf(focused)


def test_ensure_focus_leaves_focus_that_is_already_inside_alone(qtbot, tmp_path: Path) -> None:
    home = _home(tmp_path)
    qtbot.addWidget(home)
    home.show()
    qtbot.waitExposed(home)
    reactivate_window(home)
    search = home.findChild(QLineEdit, "session-search")
    assert search is not None
    search.setFocus()

    home.take_keyboard_focus()
    assert QApplication.focusWidget() is search


def test_ensure_focus_picks_a_widget_that_can_use_the_keyboard(qtbot, tmp_path: Path) -> None:
    """Not merely *inside* the page: on the page's keyboard entry point.

    Focusing the page itself would satisfy the shortcuts -- a widget-scoped binding
    matches the host as well as its children, and ``setFocus()`` on a ``Qt.NoFocus``
    container does take effect (measured; it is not the no-op it looks like). But it
    would leave arrow keys and typing with nowhere to go, so the target is the list.
    """
    home = _home(tmp_path)
    qtbot.addWidget(home)
    home.show()
    qtbot.waitExposed(home)
    reactivate_window(home)

    home.take_keyboard_focus()
    focused = QApplication.focusWidget()
    assert focused is not home
    assert focused is not None
    assert home.isAncestorOf(focused)
    assert focused.focusPolicy() != Qt.FocusPolicy.NoFocus


# --- the README says the same thing -------------------------------------------------


def test_every_binding_appears_in_the_readme() -> None:
    """One direction only: the README may explain more than the table declares.

    Sequences are compared in Qt's own portable text so the check follows a renamed key
    rather than a hand-kept translation of it.
    """
    #: User-facing spellings of the same key. The README is written for fingers, so it
    #: says the arrow rather than the word and lists one row for Return and the keypad
    #: Enter. Anything not in here has to appear literally.
    aliases = {"Del": "Delete", "Return": "Enter", "Ctrl+Up": "Ctrl+\u2191"}
    readme = USER_GUIDE.read_text(encoding="utf-8")
    missing = []
    for name, table in _tables().items():
        for binding in table:
            text = binding.key.toString()
            spellings = {text, aliases.get(text, text)}
            # Case-insensitively: the README writes letter keys the way they are typed
            # (`n`), Qt names them as keys (`N`).
            if not any(f"`{spelling}`".lower() in readme.lower() for spelling in spellings):
                missing.append(f"{name}: {text}")
    assert missing == [], f"not documented in README.md: {missing}"


def test_readme_uses_qts_spelling_of_a_shortcut() -> None:
    """Keeps the test above honest: it can only pass by matching real Qt text."""
    assert QKeySequence("Ctrl+F").toString() == "Ctrl+F"
