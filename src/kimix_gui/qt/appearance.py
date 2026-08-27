"""Which Qt events mean "what you cached from the appearance is now stale".

There is no theme manager in this app and no change signal to subscribe to,
because Qt already broadcasts. Probed on PySide6 6.11 (Windows, offscreen), with
an event filter on a tree of widgets three containers deep:

* ``app.setFont()`` sends ``FontChange`` to **every** widget, not only top-levels
  -- except a widget that had ``setFont()`` called on it, which carries
  ``WA_SetFont`` and is skipped. That exclusion is a feature: the preferences font
  preview is supposed to keep showing the font being previewed. Setting the same
  font again sends nothing.
* ``app.setPalette()`` sends ``PaletteChange`` to every widget.
* ``app.setStyleSheet()`` sends ``StyleChange`` to every widget, **including when
  the new sheet is byte-identical to the old one**. That is what makes it a usable
  stand-in for "a theme was applied": ``theme.apply_theme`` always sets the sheet.
* ``app.installTranslator()`` sends ``LanguageChange`` to every ``QObject`` --
  widgets, layouts and item delegates alike. With one catch: an *empty*
  translator (a ``.qm`` that failed to load) is ignored and sends nothing, so a
  missing catalog is silent rather than a spurious retranslate.

So the rule is: if a widget stores something derived from the font or the theme,
it overrides ``changeEvent`` and recomputes on :data:`APPEARANCE_CHANGED`. The
places that do are listed in ``tests/gui/test_appearance.py`` and asserted there,
because a new cache without a new listener is exactly the bug this file exists to
prevent.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent

# Something the font or the theme feeds has moved; recompute measurements and
# re-resolve colors. ``StyleChange`` is here rather than only ``PaletteChange``
# because this app's theme lives in the stylesheet, and the stylesheet is what
# ``apply_theme`` swaps.
APPEARANCE_CHANGED: Final[frozenset[QEvent.Type]] = frozenset(
    {
        QEvent.Type.FontChange,
        QEvent.Type.PaletteChange,
        QEvent.Type.StyleChange,
    }
)

# Only the measurements moved, not the colors. Widgets that cache a size but no
# color use this to avoid recomputing on every restyle.
FONT_CHANGED: Final[frozenset[QEvent.Type]] = frozenset({QEvent.Type.FontChange})

__all__ = ["APPEARANCE_CHANGED", "FONT_CHANGED"]
