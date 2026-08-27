"""Framework-free design tokens: palettes and numeric scales, no Qt anywhere."""

from kimix_gui.design.categories import CATEGORY_NAMES, CategoryPalette
from kimix_gui.design.palette import Palette
from kimix_gui.design.scale import (
    Breakpoints,
    CardPadding,
    ComposerMetrics,
    Motion,
    RadiusScale,
    SessionListMetrics,
    Sizing,
    SpacingScale,
    TodoPanelMetrics,
    TrackingScale,
    TranscriptMetrics,
    TypeScale,
)
from kimix_gui.design.theme import (
    DARK,
    DEFAULT_THEME,
    LIGHT,
    SUPPORTED_THEMES,
    SYSTEM_THEME,
    THEMES,
    Theme,
    normalize_theme_preference,
    resolve_theme,
)

__all__ = [
    "CATEGORY_NAMES",
    "DARK",
    "DEFAULT_THEME",
    "LIGHT",
    "SUPPORTED_THEMES",
    "SYSTEM_THEME",
    "THEMES",
    "Breakpoints",
    "CardPadding",
    "CategoryPalette",
    "ComposerMetrics",
    "Motion",
    "Palette",
    "RadiusScale",
    "SessionListMetrics",
    "Sizing",
    "SpacingScale",
    "Theme",
    "TodoPanelMetrics",
    "TrackingScale",
    "TranscriptMetrics",
    "TypeScale",
    "normalize_theme_preference",
    "resolve_theme",
]
