"""Qt binding for the design tokens: palette mapping, QSS builder, font helpers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from kimix_gui.design import (
    DARK,
    CategoryPalette,
    Palette,
    RadiusScale,
    Sizing,
    SpacingScale,
    Theme,
    TrackingScale,
    TypeScale,
    resolve_theme,
)
from kimix_gui.preferences import InterfacePreferences

_ACTIVE_THEME: Theme = DARK


def active_theme() -> Theme:
    """Return the theme Qt code must resolve colors through.

    Painters call this instead of holding an import-time snapshot, so re-applying a
    theme at runtime reaches every brush on the next repaint.
    """

    return _ACTIVE_THEME


def set_active_theme(theme: Theme) -> None:
    """Install ``theme`` as the source for runtime color lookups.

    The only global the painters read. A flat ``name -> hex`` map used to sit beside
    it, mutated in place so painters that had done ``from ... import COLORS`` would
    not hold a stale dict; every painter resolves through :func:`active_theme` now,
    so there is one answer to "which theme is installed" instead of two that had to
    be kept agreeing.
    """

    global _ACTIVE_THEME
    _ACTIVE_THEME = theme


def _tracking(value: float) -> str:
    """Format a letter-spacing step, always with a unit.

    A bare ``0`` looks like valid CSS and is not: Qt's parser drops the whole
    declaration, so a rule meant to *reset* tracking silently loses to the
    less specific rule it was written to override. Emitting ``0px`` is what
    actually resets it. ``tests/gui/test_theme.py`` pins this.
    """

    return f"{value}px"


def _base_section(p: Palette, z: Sizing) -> str:
    # ``QWidget`` matches subclasses too, so this one declaration is what makes every
    # widget transparent unless a container claims a background. Eleven rules used to
    # restate it; a pixel sweep of all twelve scenes showed none of them changed
    # anything, and ``tests/gui/test_styling.py`` now keeps them from creeping back.
    return f"""QWidget {{
    background: transparent;
    color: {p.text};
}}
QMainWindow, QDialog, QStackedWidget,
QWidget#home-view, QWidget#chat-view {{
    background: {p.bg};
}}
QMenu, QToolTip {{
    background: {p.panel};
    color: {p.text};
    border: {z.border_width}px solid {p.border};
}}
"""


def _card_section(p: Palette, r: RadiusScale, z: Sizing) -> str:
    """The three elevations of :class:`kimix_gui.qt.styling.CardLevel`.

    These were ``#composer-pad-card``, ``#session-detail`` and ``#font-preview``:
    the same background/border/radius triple written three times, three hundred
    lines apart, with three radii and nothing saying how they related. Adjacent
    and named, the scale is visible, and a fourth card picks a step instead of
    inventing a pair.
    """

    return f"""QFrame[card="floating"] {{
    background: {p.surface};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.xxl}px;
}}
QFrame[card="panel"] {{
    background: {p.panel};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.xl}px;
}}
QFrame[card="inset"] {{
    background: {p.surface};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.md}px;
}}
"""


def _typography_section(
    p: Palette, s: SpacingScale, r: RadiusScale, t: TypeScale, k: TrackingScale
) -> str:
    return f"""QLabel[role="display"] {{
    color: {p.accent};
    font-weight: {t.weight_bold};
    letter-spacing: {_tracking(k.none)};
}}
QLabel[role="display"][level="1"] {{
    font-size: {t.display}px;
}}
QLabel[role="display"][level="2"] {{
    font-size: {t.xxl}px;
}}
QLabel[role="title"] {{
    color: {p.accent};
    font-weight: {t.weight_semibold};
    letter-spacing: {_tracking(k.wide)};
}}
QLabel[role="section"] {{
    color: {p.text};
    font-size: {t.md}px;
    font-weight: {t.weight_semibold};
    letter-spacing: {_tracking(k.none)};
}}
QLabel[role="overline"] {{
    color: {p.muted};
    font-size: {t.xs}px;
    font-weight: {t.weight_semibold};
    letter-spacing: {_tracking(k.wider)};
}}
QLabel[role="caption"] {{
    color: {p.muted};
    font-size: {t.sm}px;
}}
QLabel[role="footnote"] {{
    color: {p.muted};
    font-size: {t.xs}px;
}}
QLabel[tone="muted"] {{
    color: {p.muted};
}}
QLabel[tone="danger"] {{
    color: {p.error};
}}
QLabel#detail-title {{
    font-size: {t.xl}px;
    font-weight: {t.weight_semibold};
}}
QLabel#session-title {{
    font-weight: {t.weight_semibold};
}}
QLabel#session-badge {{
    font-size: {t.xs}px;
    font-weight: {t.weight_semibold};
    border-radius: {r.md}px;
    padding: {s.xxs}px {s.md}px;
}}
QLabel#session-badge[kind="last"] {{
    background: {p.accent};
    color: {p.on_accent};
}}
QLabel#session-badge[kind="archived"] {{
    background: {p.boost};
    color: {p.muted};
}}
"""


def _home_rows_section(s: SpacingScale) -> str:
    return f"""QCheckBox#session-check {{
    spacing: {s.none};
    border: none;
}}
QPushButton#delete-sessions {{
    padding: {s.xs}px {s.lg}px;
}}
"""


def _input_section(p: Palette, s: SpacingScale, r: RadiusScale, t: TypeScale, z: Sizing) -> str:
    # Leave ``::drop-down`` / ``::down-arrow`` to Qt together. Styling only the
    # first half suppresses the native arrow; complex-widget subcontrols have to
    # be customized as a complete set. The themed frame and popup do not need it.
    return f"""QLineEdit, QPlainTextEdit, QSpinBox, QTextEdit, QComboBox {{
    background: {p.panel};
    color: {p.text};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.lg}px;
    selection-background-color: {p.boost};
    selection-color: {p.text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border: {z.border_width}px solid {p.focus_ring};
}}
QComboBox QAbstractItemView {{
    background: {p.panel};
    color: {p.text};
    border: {z.border_width}px solid {p.border};
    outline: none;
    selection-background-color: {p.boost};
    selection-color: {p.text};
}}
QPlainTextEdit#prompt {{
    background: {p.panel};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.lg}px;
    padding: {s.md}px {s.xl}px;
}}
QPlainTextEdit#prompt:focus {{
    border: {z.border_width}px solid {p.focus_ring};
}}
QDialog#composer-pad {{
    background: transparent;
}}
QLabel#composer-pad-title {{
    font-size: {t.lg}px;
    font-weight: {t.weight_semibold};
}}
QPlainTextEdit#prompt-pad {{
    background: {p.panel};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.xl}px;
    padding: {s.xl}px {s.xxl}px;
    font-size: {t.md}px;
}}
QPlainTextEdit#prompt-pad:focus {{
    border: {z.border_width}px solid {p.focus_ring};
}}
QFrame#composer-dock {{
    background: {p.surface};
}}
"""


def _button_section(p: Palette, s: SpacingScale, r: RadiusScale, t: TypeScale, z: Sizing) -> str:
    # Order matters where specificity ties: the ``[variant=...]`` rules weigh the
    # same as the bare ``:hover`` / ``:disabled`` rules above them, so they have
    # to come later to keep winning the way the old ID selectors did.
    return f"""QPushButton[metric="action"] {{
    min-width: {z.action_button_min_width}px;
    padding: {s.xs}px {s.xl}px;
}}
QPushButton[metric="nav"] {{
    min-width: {z.nav_button_min_width}px;
    padding: {s.xxs}px {s.md}px;
}}
QPushButton {{
    background: {p.panel};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.xl}px;
    color: {p.text};
}}
QPushButton:hover {{
    background: {p.boost};
}}
QPushButton:disabled {{
    color: {p.muted};
}}
QPushButton[variant="primary"] {{
    background: {p.accent};
    color: {p.on_accent};
    border: none;
    font-weight: {t.weight_semibold};
}}
QPushButton[variant="primary"]:hover {{
    background: {p.accent_hover};
}}
QPushButton[variant="danger"] {{
    background: {p.danger_surface};
    color: {p.error};
    border: {z.border_width}px solid {p.danger_border};
}}
QPushButton[variant="primary"][metric="action"]:disabled,
QPushButton[variant="danger"][metric="action"]:disabled {{
    background: {p.panel};
    color: {p.muted};
    border: {z.border_width}px solid {p.border};
}}
QPushButton[variant="ghost"], QPushButton[variant="icon"] {{
    background: transparent;
    border: none;
    color: {p.muted};
}}
QPushButton[variant="ghost"] {{
    padding: {s.xs}px {s.md}px;
}}
QPushButton[variant="icon"] {{
    padding: {s.none};
    min-width: {z.icon_button_min_width}px;
    border-radius: {r.sm}px;
}}
QPushButton[variant="ghost"]:hover, QPushButton[variant="icon"]:hover {{
    background: {p.boost};
    color: {p.text};
}}
QPushButton[variant="ghost"]:disabled {{
    background: transparent;
}}
QToolButton[variant="disclosure"] {{
    background: {p.panel};
    color: {p.accent};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.md}px;
    font-weight: {t.weight_semibold};
    text-align: left;
}}
QToolButton[variant="disclosure"]:hover {{
    background: {p.boost};
}}
QToolButton[variant="disclosure"]:focus {{
    border-color: {p.focus_ring};
}}
"""


def _list_section(p: Palette, s: SpacingScale, r: RadiusScale, z: Sizing) -> str:
    return f"""QListWidget, QListView {{
    border: none;
    outline: none;
    padding: {s.xs}px;
}}
QListView#transcript {{
    background: transparent;
    border: none;
    padding: {s.none};
}}
QListView#transcript::item {{
    background: transparent;
    border: none;
    padding: {s.none};
}}
QListWidget::item, QListView::item {{
    border-radius: {r.md}px;
    padding: {s.xs}px;
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {p.boost};
    border-left: {z.selection_bar_width}px solid {p.accent};
}}
QListWidget#session-list {{
    border: none;
    outline: none;
    padding: {s.xxs}px {s.none};
}}
QListWidget#session-list::item {{
    padding: {s.none};
    margin: {s.hairline}px {s.none};
    border: none;
    background: transparent;
}}
QListWidget#session-list::item:selected {{
    background: transparent;
    border: none;
}}
QListWidget#session-list::item:hover {{
    background: transparent;
}}
QScrollBar:vertical {{
    width: {z.scrollbar_width}px;
    margin: {s.xs}px {s.xxs}px;
}}
QScrollBar::handle:vertical {{
    background: {p.border};
    border-radius: {r.xs}px;
    min-height: {z.scrollbar_handle_min_height}px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: {s.none};
}}
QSplitter::handle {{
    background: {p.border};
}}
"""


def _chrome_section(p: Palette, s: SpacingScale, r: RadiusScale, z: Sizing) -> str:
    return f"""QFrame[surface="bar"] {{
    background: {p.surface};
    border: none;
}}
QLineEdit#history-turn {{
    padding: {s.xxs}px {s.md}px;
}}
"""


def _todo_section(p: Palette, s: SpacingScale, r: RadiusScale, t: TypeScale, z: Sizing) -> str:
    return f"""QFrame#todo-panel {{
    background: {p.overlay};
    border: {z.border_width}px solid {p.border};
}}
QFrame#todo-panel[mode="card"] {{
    border-radius: {r.xl}px;
}}
QFrame#todo-panel[mode="pill"] {{
    border-radius: {r.pill}px;
}}
QFrame#todo-panel[flash="true"] {{
    border: {z.border_width}px solid {p.accent};
}}
QFrame#todo-header {{
    border: none;
}}
QLabel#todo-count {{
    color: {p.text};
    font-size: {t.sm}px;
    font-weight: {t.weight_semibold};
}}
QLabel#todo-chevron {{
    padding-left: {s.xxs}px;
}}
QLabel#todo-dot {{
    font-size: {t.micro}px;
}}
QLabel[role="marker"] {{
    color: {p.muted};
}}
QLabel[role="marker"][state="in_progress"] {{
    color: {p.accent};
}}
QLabel[role="marker"][state="done"] {{
    color: {p.success};
}}
QScrollArea#todo-scroll {{
    border: none;
    border-top: {z.border_width}px solid {p.border};
}}
QFrame#todo-row {{
    border: none;
    border-radius: {r.sm}px;
}}
QFrame#todo-row[state="in_progress"] {{
    background: {p.boost};
}}
QLabel#todo-glyph {{
    font-size: {t.sm}px;
}}
QLabel#todo-item-title {{
    color: {p.text};
    font-size: {t.sm}px;
}}
QLabel#todo-item-title[state="in_progress"] {{
    font-weight: {t.weight_semibold};
}}
QLabel#todo-item-title[state="done"] {{
    color: {p.muted};
    /* Was a hand-built QFont with setStrikeOut in ``_TodoRow``. Setting a font on
       the widget also pinned it, so a done row stopped following the interface
       font preference; the declaration does the same thing and stays themeable. */
    text-decoration: line-through;
}}
QLabel#todo-footer {{
    border-top: {z.border_width}px solid {p.border};
    padding: {s.none} {s.lg_plus}px;
}}
"""


def _toast_section(p: Palette, s: SpacingScale, r: RadiusScale, t: TypeScale, z: Sizing) -> str:
    return f"""QLabel#toast {{
    background: {p.boost};
    color: {p.text};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.pill_lg}px;
    padding: {s.lg}px {s.xxxl}px;
    font-size: {t.base}px;
}}
"""


def _preferences_section(p: Palette, s: SpacingScale, r: RadiusScale, z: Sizing) -> str:
    return f"""QDialog#preferences-dialog {{
    background: {p.bg};
}}
QListWidget#preferences-categories, QListWidget#config-list {{
    background: {p.surface};
    border: {z.border_width}px solid {p.border};
    border-radius: {r.md}px;
    padding: {s.sm}px;
}}
QListWidget#preferences-categories::item, QListWidget#config-list::item {{
    padding: {z.settings_row_padding // 2}px {s.lg}px;
}}
QComboBox#variant-picker[state="unavailable"] {{
    border-color: {p.danger_border};
}}
"""


def build_stylesheet(theme: Theme = DARK) -> str:
    """Render the global QSS for ``theme``.

    Replaces the former import-time ``APP_STYLE`` f-string: nothing is evaluated
    until a caller asks, which is what lets a second theme exist later.
    """

    p = theme.palette
    s = theme.spacing
    r = theme.radius
    t = theme.type_scale
    k = theme.tracking
    z = theme.sizing
    sections = (
        _base_section(p, z),
        _card_section(p, r, z),
        _typography_section(p, s, r, t, k),
        _home_rows_section(s),
        _input_section(p, s, r, t, z),
        _button_section(p, s, r, t, z),
        _list_section(p, s, r, z),
        _chrome_section(p, s, r, z),
        _todo_section(p, s, r, t, z),
        _toast_section(p, s, r, t, z),
        _preferences_section(p, s, r, z),
    )
    return "\n" + "".join(sections)


def apply_theme(app: QApplication, theme: Theme = DARK) -> None:
    """Apply ``theme``'s palette and stylesheet to ``app``."""

    set_active_theme(theme)
    font = QFont(theme.type_scale.base_family)
    font.setPixelSize(theme.type_scale.base)
    app.setFont(font)
    tokens = theme.palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens.bg))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens.surface))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(tokens.panel))
    palette.setColor(QPalette.ColorRole.Text, QColor(tokens.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens.panel))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(tokens.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens.boost))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(tokens.muted))
    app.setPalette(palette)
    app.setStyleSheet(build_stylesheet(theme))


def desktop_prefers_dark(app: QApplication) -> bool | None:
    """Report the desktop's color-scheme hint, or ``None`` when there is none.

    ``QStyleHints.colorScheme()`` returns ``Unknown`` wherever the platform plugin
    cannot answer -- the ``offscreen`` platform the tests run on always does, which
    is why the ``auto`` preference has to resolve to a concrete default rather than
    guessing. Qt tracks this hint live and emits ``colorSchemeChanged``; nothing
    subscribes yet, so following the desktop mid-session is not implemented.
    """

    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return None


def apply_theme_preference(app: QApplication, preferences: InterfacePreferences) -> Theme:
    """Resolve the stored theme preference against the desktop and apply it.

    The Qt-free half of this decision lives in ``kimix_gui.design.resolve_theme``;
    all this adds is the one fact only Qt knows. Returns the theme that was applied
    so a caller can report it without re-deriving it.
    """

    theme = resolve_theme(preferences.theme, desktop_prefers_dark(app))
    apply_theme(app, theme)
    return theme


def available_monospace_families() -> list[str]:
    """Return installed fixed-pitch families suitable for interface selection."""

    return [family for family in QFontDatabase.families() if QFontDatabase.isFixedPitch(family)]


def interface_font(preferences: InterfacePreferences, theme: Theme = DARK) -> QFont:
    """Build a configured fixed-width font, retaining the legacy default when unset."""

    if not preferences.font_families:
        font = QFont(theme.type_scale.base_family)
        font.setPixelSize(preferences.font_size)
        return font
    available = set(QFontDatabase.families())
    families = [family for family in preferences.font_families if family in available]
    system_fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
    if system_fixed and system_fixed not in families:
        families.append(system_fixed)
    font = QFont()
    font.setFamilies(families)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(preferences.font_size)
    return font


def apply_interface_font(app: QApplication, preferences: InterfacePreferences) -> None:
    """Override the default font when the user selected a fixed-width typeface."""

    app.setFont(interface_font(preferences))


__all__ = [
    "DARK",
    "CategoryPalette",
    "Theme",
    "active_theme",
    "apply_interface_font",
    "apply_theme",
    "available_monospace_families",
    "build_stylesheet",
    "interface_font",
    "set_active_theme",
]
