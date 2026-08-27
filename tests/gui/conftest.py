from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pin the UI language to English so assertions on interface copy do not depend on
# the developer machine's locale (this repo is developed on Chinese Windows).
# Assigned unconditionally rather than with ``setdefault``: an inherited
# ``LANG=zh_CN`` would silently flip the language back once i18n lands.
#
# Measured on Windows 11 + PySide6 with a zh-CN host (probe, not documentation):
#   * ``LANG=en_US.UTF-8``     -> ``QLocale.system().name() == "en_US"``,
#     ``QLocale.system().language()`` is ``English``, ``QLocale().name() == "en_US"``.
#   * ``LC_ALL`` / ``LANGUAGE`` alone -> ignored by Qt on Windows (stays ``zh_CN``);
#     they are set anyway because POSIX hosts honor them.
#   * ``QLocale.system().uiLanguages()`` is unaffected by every variable above and
#     keeps returning the OS preferred UI languages (``zh-Hans-CN`` first).
#
# Consequence: this pins any language decision derived from
# ``QLocale.system().language()`` / ``.name()`` (the ``auto`` behaviour), but a
# ``QTranslator.load(QLocale.system(), ...)`` call walks ``uiLanguages()`` and would
# still find Chinese -- which is why ``kimix_gui.qt.i18n`` never uses that overload,
# and why ``_pin_english_ui`` below does not rely on these variables either.
_LOCALE_ENV = ("LANG", "LC_ALL", "LANGUAGE")
_PREVIOUS_LOCALE_ENV = {name: os.environ.get(name) for name in _LOCALE_ENV}
os.environ.update(
    {
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "LANGUAGE": "en_US:en",
    }
)

# Imported below the assignments on purpose: importing Qt reads the environment.
import pytest
from PySide6.QtCore import QEvent, QLocale
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import QApplication

from kimix_gui import app as app_module
from kimix_gui.app import KimixGuiApp
from kimix_gui.qt import theme as theme_module
from kimix_gui.qt.i18n import set_active_language

# Qt caches the system locale after the first query. Restore the process environment
# immediately so this nested conftest does not leak LANG changes into non-GUI tests.
QLocale.system()
for _name, _value in _PREVIOUS_LOCALE_ENV.items():
    if _value is None:
        os.environ.pop(_name, None)
    else:
        os.environ[_name] = _value


@pytest.fixture(scope="session")
def _gui_share_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a session-owned Kimi share tree for GUI tests only."""

    return tmp_path_factory.mktemp("kimix-gui-share")


@pytest.fixture(autouse=True)
def _isolate_gui_share_dir(
    monkeypatch: pytest.MonkeyPatch,
    _gui_share_dir: Path,
) -> None:
    """Keep GUI preferences out of both real user state and non-GUI tests."""

    monkeypatch.setenv("KIMI_SHARE_DIR", str(_gui_share_dir))


@pytest.fixture
def app_appearance(qapp: QApplication) -> Iterator[QApplication]:
    """Restore every piece of process-wide appearance state the test may move.

    ``apply_theme`` and ``apply_interface_font`` reach past ``QApplication`` into a
    module global as well: ``theme._ACTIVE_THEME``, which every painter resolves
    through at paint time. Restoring only the Qt side would leave a test that
    applied a probe theme leaking magenta into whatever ran next.

    Shared by ``test_theme.py`` and ``test_appearance.py``; a second copy is a
    second chance to forget one of the four.
    """

    font = QFont(qapp.font())
    palette = QPalette(qapp.palette())
    stylesheet = qapp.styleSheet()
    active_theme = theme_module._ACTIVE_THEME
    yield qapp
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(stylesheet)
    theme_module._ACTIVE_THEME = active_theme


@pytest.fixture(autouse=True)
def _pin_english_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every app under test onto the English catalogs.

    ``KimixGuiApp`` builds a real ``KimixGuiConfigStore`` when none is injected, so
    without this the developer's own saved language would decide what ~140
    English copy assertions see. Replacing the call at the assembly point is the
    narrowest pin available: it leaves config parsing untouched (the preference tests
    still observe real stored values) and does not depend on ``LANG``, which Qt ignores
    for ``uiLanguages()`` on Windows.
    """

    monkeypatch.setattr(
        app_module,
        "apply_language",
        lambda qt_app, _preferences: set_active_language(qt_app, "en"),
    )


# Automatic cyclic collection is deliberately left ON.
#
# It used to be disabled here. Crash signature, at roughly 1 run in 8: the faulting
# thread was always named ``kimix-bridge`` and always sat in ``Garbage-collecting``,
# seen as both ``0xC0000005`` (access violation) and ``0xC0000374`` (heap corruption).
# Automatic generational GC fires on whichever thread crosses an allocation threshold,
# and a bridge worker allocates continuously, so it regularly ran PySide6 finalizers
# for Qt objects from a non-GUI thread -- which Qt does not allow.
#
# The cause was one container: ``TranscriptDelegate`` cached unparented
# ``QTextDocument`` objects, so Python owned their C++ half, and the cache was only
# reachable through a cycle. ``transcript._release`` fixes that at the source. Leaving
# collection on is what keeps this suite honest: it runs the same collection behaviour
# as the product, so a future unparented Qt object stored in a cycle fails here rather
# than only on a user's machine. Cost is about 4% of the suite's runtime.
#
# Do not disable it again to stabilise a test. The readable signal lives in
# ``tests/gui/test_gc_safety.py``; this is only the backstop for what that misses, and its
# signal is terrible -- the whole process dies with no report.


@pytest.fixture(autouse=True)
def _flush_deferred_widget_deletions() -> Iterator[None]:
    """Actually destroy the widgets ``qtbot`` retired at the end of the last test.

    ``pytest-qt`` closes every registered widget and calls ``deleteLater()`` on it,
    then calls ``processEvents()``. That is not enough: ``processEvents`` deliberately
    does **not** deliver ``DeferredDelete`` -- Qt only flushes those when the event
    loop that posted them unwinds, which never happens in a test that never ran a
    loop. So the C++ objects stayed alive, and with automatic GC disabled
    (see ``_collect_garbage_only_on_the_main_thread``) nothing else reclaimed them.

    Measured before this fixture: ``test_components.py`` added 180 live widgets per
    test and the application reached 20,780 live widgets by the time the theme tests
    ran. ``QApplication.setStyleSheet`` walks every live widget, so one
    ``apply_theme`` cost 3.6 seconds there against 6 milliseconds on a clean
    application -- which is what turned a 50 second suite into a 220 second one as
    soon as tests that switch themes were added. ``gc.collect()`` does not help;
    these are not cycles, they are objects Qt is holding for a queued delete.

    Flushing at setup rather than teardown is deliberate: ``pytest-qt`` posts the
    deletes from a ``pytest_runtest_teardown`` wrapper marked ``trylast``, which runs
    after every fixture finalizer, so at teardown time there is nothing to flush yet.
    """

    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    yield


@pytest.fixture(autouse=True)
def _shut_down_bridges(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop every ``KimixGuiApp`` bridge this test started before the next one runs.

    Diagnosed from two captured crashes (``0xC0000005`` access violation and
    ``0xC0000374`` heap corruption, roughly 1 run in 8). Both had the identical
    shape: the faulting thread was named ``kimix-bridge`` and was inside
    ``Garbage-collecting`` while the main thread sat in ``wait_idle``.

    Cause: ``KimixBridge.start()`` spawns a daemon thread running an asyncio loop,
    and the ~40 tests that build a ``KimixGuiApp`` inline never call ``shutdown()``.
    Those orphaned worker threads stay alive for the rest of the session, so when a
    GC pass happens to run on one of them it finalizes Qt objects belonging to a
    window ``qtbot`` already deleted on the C++ side -- a cross-thread use of freed
    memory. Reaping the threads per test removes the whole window.

    ``KimixBridge.stop()`` returns early when no loop is running, so this is safe
    for apps that were built but never started.
    """
    started: list[KimixGuiApp] = []
    original_init = KimixGuiApp.__init__

    def tracking_init(self: KimixGuiApp, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        started.append(self)

    monkeypatch.setattr(KimixGuiApp, "__init__", tracking_init)
    yield
    monkeypatch.undo()
    if not started:
        return
    for app in started:
        app.shutdown()
    # Every worker loop is joined now, so reclaim the cycles the disabled collector
    # left behind. On the main thread, which is where these Qt objects were created.
    # Skipped entirely for the ~320 tests that never build an app: collecting after
    # each of those turned a 6 second run into 25 seconds.
    gc.collect()
