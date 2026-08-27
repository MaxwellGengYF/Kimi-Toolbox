"""Pure-layer language resolution: preference plus system locale to a language code."""

from __future__ import annotations

#: Preference value meaning "follow the operating system".
SYSTEM_LANGUAGE = "auto"

#: Language the sources are written in; also the fallback whenever nothing matches.
#: ``tr()`` returns the English msgid when no catalog is installed, so ``en`` is the
#: one language that can never fail to load.
DEFAULT_LANGUAGE = "en"

#: Languages with a shipped catalog, in menu order.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "zh_CN")

#: Everything :class:`~kimix_gui.preferences.InterfacePreferences` accepts.
LANGUAGE_PREFERENCES: tuple[str, ...] = (SYSTEM_LANGUAGE, *SUPPORTED_LANGUAGES)

_SUPPORTED_BY_KEY = {
    language.replace("_", "").lower(): language for language in SUPPORTED_LANGUAGES
}

#: Base language subtag to catalog. ``zh_TW`` is deliberately folded into ``zh_CN``
#: for now: there is no Traditional catalog, and Simplified beats English for a
#: Chinese-reading user.
_BASE_LANGUAGE_TO_CATALOG = {"en": "en", "zh": "zh_CN"}


def _catalog_key(value: str) -> str:
    return value.replace("-", "").replace("_", "").lower()


def normalize_language_preference(preference: object) -> str:
    """Coerce a stored or user-picked value into one of :data:`LANGUAGE_PREFERENCES`.

    Unknown input becomes :data:`SYSTEM_LANGUAGE` because that is the documented
    default for the setting: an unreadable value should behave like "never chosen",
    not like an explicit English choice. Compare :func:`resolve_language`, which
    answers a different question and prefers :data:`DEFAULT_LANGUAGE`.
    """

    if not isinstance(preference, str):
        return SYSTEM_LANGUAGE
    candidate = preference.strip()
    if candidate.lower() == SYSTEM_LANGUAGE:
        return SYSTEM_LANGUAGE
    return _SUPPORTED_BY_KEY.get(_catalog_key(candidate), SYSTEM_LANGUAGE)


def resolve_language(preference: str, system_locale: str) -> str:
    """Resolve an ``auto``/``en``/``zh_CN`` preference into the catalog to load.

    ``system_locale`` is a locale name such as ``zh_CN``, ``zh-Hans-CN`` or
    ``en_US`` (the Qt caller passes ``QLocale.system().name()``); it is only read
    when ``preference`` is ``auto``. Anything unrecognized resolves to
    :data:`DEFAULT_LANGUAGE`, so a corrupt preference or an exotic locale yields the
    untranslated sources instead of a guess.
    """

    if not isinstance(preference, str):
        return DEFAULT_LANGUAGE
    candidate = preference.strip()
    if candidate.lower() != SYSTEM_LANGUAGE:
        explicit = _SUPPORTED_BY_KEY.get(_catalog_key(candidate))
        return explicit if explicit is not None else DEFAULT_LANGUAGE
    return language_for_locale(system_locale)


def language_for_locale(system_locale: str) -> str:
    """Map a locale name onto a supported catalog, falling back to English."""

    if not isinstance(system_locale, str):
        return DEFAULT_LANGUAGE
    base = system_locale.strip().replace("-", "_").split("_", 1)[0].lower()
    return _BASE_LANGUAGE_TO_CATALOG.get(base, DEFAULT_LANGUAGE)


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_PREFERENCES",
    "SUPPORTED_LANGUAGES",
    "SYSTEM_LANGUAGE",
    "language_for_locale",
    "normalize_language_preference",
    "resolve_language",
]
