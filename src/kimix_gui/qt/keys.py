"""Every keyboard binding in the application, declared in one place.

There used to be three mechanisms doing this job, and they disagreed.

``MainWindow`` installed eight window-wide ``QShortcut``s whose handlers each began by
asking which page was on screen (``isinstance(self.current_view, ChatView)``), because a
window-wide binding fires from anywhere and most of these keys only mean something on
one page. ``ChatView`` re-implemented seven of those same keys in ``keyPressEvent`` --
unreachable in the product, because Qt consults the shortcut map before delivering a
key to any widget, so the window's binding always won; only a ``ChatView`` with no such
window as an ancestor ever ran that code, which is to say only a test. ``HomeView``
hand-rolled a third mechanism: ``keyPressEvent`` plus an ``eventFilter`` on its list,
dispatching five keys through an if-chain.

All three collapse into ``Qt.ShortcutContext.WidgetWithChildrenShortcut``, which fires a
binding only while focus is inside the widget that owns it. Measured on PySide6 6.11:

* A page of a ``QStackedWidget`` that is not the current one is hidden, and Qt skips
  hidden widgets' shortcuts. Page switching *is* the scope test, so the ``isinstance``
  dispatch was re-deriving something Qt already knew.
* A focused ``QLineEdit`` vetoes a plain-letter or ``Space`` binding through
  ``ShortcutOverride`` and types the character instead, so ``n`` and ``q`` do not need
  protecting from the search field by hand. ``Return`` is *not* vetoed by a
  ``QLineEdit`` (which never inserts it) but is by a ``QPlainTextEdit`` (which does).
* A focused ``QListWidget`` vetoes none of ``n`` / ``Space`` / ``Del`` / ``Return``, so
  the event filter on the session list was unnecessary.
* A modal dialog takes focus, so a page's bindings stop firing while one is open. That
  is the guard ``MainWindow`` was spelling out as ``if self._modal is not None``.

What is left is a table per keyboard scope and one installer. The action names are the
seam: the table says which keys exist, each view says what its actions do, and
``install`` refuses to run if the two do not line up exactly -- so a renamed handler or
a new row is a failure at construction, not a key that silently does nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True, slots=True)
class Binding:
    """One key sequence and the name of what it does.

    ``sequence`` is portable ``QKeySequence`` text rather than a ``Qt.Key``, so the
    table reads the way the README and the user's fingers do. ``action`` is a stable
    identifier: it is what a view supplies a handler for, what a test names, and what a
    future shortcut-help list would group by. Two rows may share an action -- ``Return``
    and the keypad ``Enter`` are one behaviour under two sequences.
    """

    sequence: str
    action: str

    @property
    def key(self) -> QKeySequence:
        return QKeySequence(self.sequence)


# The session browser. ``Esc`` and ``q`` both leave the application, which is what a
# list-shaped launcher screen does; there is nothing to back out to.
HOME: tuple[Binding, ...] = (
    Binding("N", "new-session"),
    Binding("Q", "quit"),
    Binding("Esc", "quit"),
    Binding("Space", "toggle-selection"),
    Binding("Del", "delete-selection"),
    Binding("Return", "open-highlighted"),
    Binding("Enter", "open-highlighted"),
    Binding("Ctrl+F", "focus-search"),
    Binding("F4", "configure-llm"),
)

# The chat page. ``Esc`` closes the session instead of the application, and ``F4``
# configures the open session rather than the folder default -- the same two keys
# meaning something else, which is exactly why these are separate scopes and not one
# window-wide table with a page check in every handler.
CHAT: tuple[Binding, ...] = (
    Binding("Esc", "leave-session"),
    Binding("F2", "focus-prompt"),
    Binding("F3", "focus-history-turn"),
    Binding("F4", "session-settings"),
    Binding("Ctrl+G", "cancel-generation"),
    Binding("Ctrl+Up", "load-older-history"),
    Binding("Ctrl+End", "jump-to-latest"),
    Binding("Ctrl+T", "toggle-todos"),
)

# The approval dialog. The letters stay the same in every language: they are muscle
# memory and they are in the README. ``Esc`` is a decision here, not a dismissal -- the
# SDK request is waiting on an answer, and closing the dialog without emitting one
# would leave it waiting forever, which is why this scope cannot lean on ``QDialog``'s
# built-in ``Esc``.
APPROVAL: tuple[Binding, ...] = (
    Binding("A", "approve"),
    Binding("S", "approve-for-session"),
    Binding("R", "reject"),
    Binding("Esc", "reject"),
)

# Every table, for the tests that check the set as a whole.
SCOPES: Mapping[str, tuple[Binding, ...]] = {
    "home": HOME,
    "chat": CHAT,
    "approval": APPROVAL,
}


def actions(bindings: Iterable[Binding]) -> frozenset[str]:
    return frozenset(binding.action for binding in bindings)


def install(
    host: QWidget,
    bindings: Iterable[Binding],
    handlers: Mapping[str, Callable[[], None]],
    *,
    context: Qt.ShortcutContext = Qt.ShortcutContext.WidgetWithChildrenShortcut,
) -> tuple[QShortcut, ...]:
    """Bind ``bindings`` on ``host``, and fail loudly if they do not match ``handlers``.

    The default context is the reason this is worth centralising: a binding fires only
    while focus is inside ``host``, so a page owns its keys and stops responding the
    moment another page or a modal dialog has the focus. Dialogs that are their own
    window pass ``WindowShortcut`` instead, since they have no page to be inside of.

    Both directions of the coverage check matter. A missing handler is a key that does
    nothing; an extra handler is a behaviour with no way to reach it, which is how the
    duplicated key handling grew in the first place.
    """
    rows = tuple(bindings)
    duplicates = _duplicate_sequences(rows)
    if duplicates:
        raise ValueError(f"the same sequence is bound twice: {', '.join(duplicates)}")
    declared = actions(rows)
    provided = frozenset(handlers)
    if declared != provided:
        missing = ", ".join(sorted(declared - provided)) or "(none)"
        extra = ", ".join(sorted(provided - declared)) or "(none)"
        raise ValueError(f"handlers do not match the table: missing {missing}; extra {extra}")
    shortcuts = []
    for binding in rows:
        shortcut = QShortcut(binding.key, host, handlers[binding.action])
        shortcut.setContext(context)
        shortcut.setObjectName(f"key:{binding.action}")
        shortcuts.append(shortcut)
    return tuple(shortcuts)


def ensure_focus(page: QWidget, target: QWidget) -> None:
    """Put focus inside ``page``, so the bindings it owns can fire.

    A widget-scoped binding matches against the focus widget, so a page with the focus
    nowhere inside it is a page whose keyboard does nothing at all.

    Focus is only moved when it is not already inside the page. Qt hands the focus back
    on its own in the cases it knows about -- a modal dialog closing over a page that was
    never hidden restores the widget that had it before, measured on Windows -- and
    guessing again would take the focus away from whatever the user was using. So this is
    a floor, not a policy: it fills in the one case Qt cannot know about, which is a page
    being put in front while the focus sits somewhere else entirely.

    ``target`` is the page's keyboard entry point rather than the page itself. Focusing
    the page would be enough for the shortcuts -- a widget-scoped binding matches the host
    as well as its children, and ``setFocus()`` on a ``Qt.NoFocus`` container does take
    effect, measured -- but it would leave the focus on something that cannot use a key
    it did not bind. Arrow keys would move nothing and typing would go nowhere.
    """
    focused = QApplication.focusWidget()
    if focused is not None and (focused is page or page.isAncestorOf(focused)):
        return
    target.setFocus()


def _duplicate_sequences(bindings: Iterable[Binding]) -> tuple[str, ...]:
    seen: set[str] = set()
    repeated: list[str] = []
    for binding in bindings:
        # Compare the parsed form: "Esc" and "Escape" are the same key written twice.
        text = binding.key.toString()
        if text in seen:
            repeated.append(binding.sequence)
        seen.add(text)
    return tuple(repeated)
