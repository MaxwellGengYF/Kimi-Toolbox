"""Qt-free interface preference values and normalization."""

from __future__ import annotations

from dataclasses import dataclass

from kimix_gui.design import SYSTEM_THEME, normalize_theme_preference
from kimix_gui.i18n import SYSTEM_LANGUAGE, normalize_language_preference


@dataclass(frozen=True, slots=True)
class InterfacePreferences:
    """Interface font families (descending priority), size, UI language, and theme."""

    font_families: tuple[str, ...] = ()
    font_size: int = 13
    language: str = SYSTEM_LANGUAGE
    theme: str = SYSTEM_THEME


def normalize_interface_preferences(preferences: InterfacePreferences) -> InterfacePreferences:
    """Return preferences constrained to the values the interface supports."""

    return InterfacePreferences(
        _normalize_families(preferences.font_families),
        _normalize_font_size(preferences.font_size),
        normalize_language_preference(preferences.language),
        normalize_theme_preference(preferences.theme),
    )


def parse_interface_preferences(data: object) -> InterfacePreferences:
    """Decode the optional ``interface`` mapping from the GUI config document."""

    if not isinstance(data, dict):
        return InterfacePreferences()
    families = data.get("font_families")
    if not isinstance(families, list):
        return InterfacePreferences()
    return InterfacePreferences(
        font_families=_normalize_families(families),
        font_size=_normalize_font_size(data.get("font_size")),
        language=normalize_language_preference(data.get("language")),
        theme=normalize_theme_preference(data.get("theme")),
    )


def serialize_interface_preferences(preferences: InterfacePreferences) -> dict[str, object]:
    """Encode normalized preferences for the shared GUI config document."""

    normalized = normalize_interface_preferences(preferences)
    return {
        "font_families": list(normalized.font_families),
        "font_size": normalized.font_size,
        "language": normalized.language,
        "theme": normalized.theme,
    }


def _normalize_font_size(size: object) -> int:
    if isinstance(size, int) and not isinstance(size, bool):
        return min(max(size, 9), 32)
    return 13


def _normalize_families(families: object) -> tuple[str, ...]:
    if not isinstance(families, (list, tuple)):
        return ()
    cleaned = tuple(
        family.strip() for family in families if isinstance(family, str) and family.strip()
    )
    return tuple(dict.fromkeys(cleaned))
