from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel, QWidget

from kimix_gui.app import KimixGuiApp
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.home_view import HomeView


def find[T: QWidget](root: QWidget, name: str, cls: type[T] = QWidget) -> T:
    widget = root.findChild(cls, name)
    if widget is None:
        raise AssertionError(f"widget #{name} not found under {type(root).__name__}")
    return widget


def widget_text(root: QWidget, name: str) -> str:
    return find(root, name).text()


def wait_idle(qtbot, app: KimixGuiApp, timeout: int = 10_000) -> None:
    qtbot.waitUntil(lambda: app.bridge.is_idle(), timeout=timeout)
    for _ in range(8):
        QApplication.processEvents()
        if app.bridge.is_idle():
            return
        qtbot.wait(20)
    QApplication.processEvents()


def reactivate_window(widget: QWidget) -> None:
    """Give the widget's window the keyboard back, as a real platform would.

    Qt only reports an application focus widget for the *active* window, and the
    offscreen platform never activates anything again after a modal dialog closes: the
    measured state is ``activeWindow() is None`` and ``focusWidget() is None``, while the
    window itself still remembers the widget that had the focus. A real platform hands it
    straight back (measured on Windows), so activating here restores what the tests would
    see on a desktop rather than inventing a focus target -- Qt picks it.

    It matters because the pages' key bindings are scoped to their own focus subtree, and
    a shortcut with no focus widget can never match. Without this, synthesising a key
    press after a dialog closes silently does nothing.
    """
    window = widget.window()
    if QApplication.activeWindow() is window:
        return
    window.activateWindow()
    # Activation is delivered as an event, so the focus widget only reappears once it has
    # been processed.
    for _ in range(8):
        QApplication.processEvents()
        if QApplication.activeWindow() is window:
            return


def wait_chat_ready(qtbot, app: KimixGuiApp, timeout: int = 10_000) -> ChatView:
    qtbot.waitUntil(
        lambda: isinstance(app.screen, ChatView) and app.screen.prompt_enabled,
        timeout=timeout,
    )
    wait_idle(qtbot, app, timeout=timeout)
    chat = app.screen
    assert isinstance(chat, ChatView)
    reactivate_window(chat)
    return chat


def wait_home(qtbot, app: KimixGuiApp, timeout: int = 10_000) -> HomeView:
    qtbot.waitUntil(lambda: isinstance(app.screen, HomeView), timeout=timeout)
    home = app.screen
    assert isinstance(home, HomeView)

    # Structural readiness, never display copy: #session-count is built empty and
    # HomeView fills it in both terminal branches of a load (`show_sessions` ->
    # `_render_sessions` writes "N total", `show_load_error` writes its own text),
    # so a non-empty label means the bridge answered. `wait_idle` then covers a
    # reload of an already populated view, because `load_sessions()` is submitted
    # to the bridge and keeps it non-idle until the listing finishes.
    count = find(home, "session-count", QLabel)
    qtbot.waitUntil(lambda: count.text() != "", timeout=timeout)
    wait_idle(qtbot, app, timeout=timeout)
    reactivate_window(home)
    return home


def launch_app(qtbot, app: KimixGuiApp, size: tuple[int, int] = (1100, 720)):
    window = app.create_window()
    window.resize(*size)
    qtbot.addWidget(window)
    window.show()
    wait_idle(qtbot, app)
    return window
