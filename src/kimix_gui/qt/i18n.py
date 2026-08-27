"""Qt binding for the language preference: catalog lookup and QTranslator install."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

from kimix_gui.i18n import DEFAULT_LANGUAGE, resolve_language
from kimix_gui.preferences import InterfacePreferences

# Where ``scripts/gui/build_translations.py`` writes the compiled catalogs. They are
# committed inside the package so clean source checkouts and wheels carry them.
TRANSLATIONS_DIR = Path(__file__).resolve().parent.parent / "translations"

# Basename prefix of the application catalogs (``kimix_gui_zh_CN.qm``).
CATALOG_PREFIX = "kimix_gui"

_ACTIVE_LANGUAGE: str = DEFAULT_LANGUAGE
_INSTALLED: list[QTranslator] = []


def active_language() -> str:
    """Return the language code the currently installed catalogs were resolved to.

    Qt code that needs to branch on the language reads this instead of asking
    ``QLocale``, because the language is a stored preference and only incidentally
    related to the system locale.
    """

    return _ACTIVE_LANGUAGE


def system_locale_name() -> str:
    """Return the system locale name (``zh_CN``, ``en_US``) for ``auto`` resolution.

    Deliberately ``QLocale.system().name()`` and not ``uiLanguages()``: the latter
    ignores ``LANG``/``LC_ALL``/``LANGUAGE`` on Windows, which would make the tests
    depend on the developer machine's OS language.
    """

    return QLocale.system().name()


def catalog_file(language: str) -> Path:
    """Return the expected ``.qm`` path for ``language``, whether or not it exists."""

    return TRANSLATIONS_DIR / f"{CATALOG_PREFIX}_{language}.qm"


def set_active_language(app: QCoreApplication, language: str) -> str:
    """Install the catalogs for ``language``, replacing whatever was installed.

    Loads by explicit file name rather than handing a ``QLocale`` to
    ``QTranslator.load()``: that overload walks ``QLocale.system().uiLanguages()``
    and would pick the OS language regardless of the stored preference. Missing or
    unreadable catalogs are skipped silently -- ``tr()`` then returns the English
    msgid, which is the correct text anyway.
    """

    global _ACTIVE_LANGUAGE
    for translator in _INSTALLED:
        app.removeTranslator(translator)
    _INSTALLED.clear()
    _ACTIVE_LANGUAGE = language
    _install(app, f"{CATALOG_PREFIX}_{language}", TRANSLATIONS_DIR)
    _install(app, f"qtbase_{language}", _qt_translations_dir())
    return language


def apply_language(app: QCoreApplication, preferences: InterfacePreferences) -> str:
    """Resolve ``preferences.language`` against the system locale and install it."""

    return set_active_language(app, resolve_language(preferences.language, system_locale_name()))


def _install(app: QCoreApplication, name: str, directory: Path) -> bool:
    translator = QTranslator(app)
    if not translator.load(name, str(directory)):
        return False
    if not app.installTranslator(translator):
        return False
    _INSTALLED.append(translator)
    return True


def _qt_translations_dir() -> Path:
    """Return the PySide6-bundled catalog directory holding ``qtbase_*.qm``."""

    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))


__all__ = [
    "CATALOG_PREFIX",
    "TRANSLATIONS_DIR",
    "active_language",
    "apply_language",
    "catalog_file",
    "set_active_language",
    "system_locale_name",
]
