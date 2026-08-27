"""Transcript record-classification hues, kept apart from the UI semantic palette.

``transcript_layout.BAR_COLOR_NAME`` / ``FAMILY_BAR_NAME`` answer "what kind of
record is this", which is a different question from "what role does this control
play". Mixing both into one flat namespace is what let ``COLORS`` drift into a
bag of hue names, so the two groups stay separate here.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The seven names ``transcript_layout`` may emit for a record's color bar.
CATEGORY_NAMES = ("cyan", "green", "red", "yellow", "blue", "magenta", "muted")


@dataclass(frozen=True, slots=True)
class CategoryPalette:
    """Hue-named colors addressed by layout-produced category names."""

    cyan: str
    green: str
    red: str
    yellow: str
    blue: str
    magenta: str
    #: Neutral fallback for uncategorized records; shares its value with
    #: ``Palette.muted`` because a dim record and dim chrome read the same.
    muted: str

    def as_map(self) -> dict[str, str]:
        """Return the palette keyed by the names the layout emits."""

        return {name: getattr(self, name) for name in CATEGORY_NAMES}

    def resolve(self, name: str) -> str:
        """Return the color for ``name``, falling back to ``muted``."""

        return self.as_map().get(name, self.muted)
