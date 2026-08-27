"""Numeric token scales: spacing, radius, type, tracking, sizing, motion, geometry.

Every step exists because some rule in the stylesheet or some painter already used
that value. Steps that look off-grid (``SpacingScale.lg_plus``,
``RadiusScale.pill*``, every field of :class:`TranscriptMetrics`) are kept as their
own step on purpose: collapsing them would move pixels, and the token extraction is
required to be visually inert.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpacingScale:
    """Padding / margin steps, in device-independent pixels."""

    none: int = 0
    hairline: int = 1
    xxs: int = 2
    xs: int = 4
    sm: int = 6
    md: int = 8
    lg: int = 10
    #: Off-grid step used only by the TODO footer's inline padding, which lines up
    #: with the row text rather than with the spacing grid.
    lg_plus: int = 11
    xl: int = 12
    xxl: int = 14
    xxxl: int = 18


@dataclass(frozen=True, slots=True)
class RadiusScale:
    """Corner radii, in device-independent pixels."""

    #: Scrollbar handles.
    xs: int = 4
    #: Icon buttons and TODO rows.
    sm: int = 6
    #: Default control radius: inputs, buttons, list items, badges.
    md: int = 8
    #: Composer prompt.
    lg: int = 10
    #: Cards: session detail, prompt pad, expanded TODO panel.
    xl: int = 12
    #: Composer pad shell.
    xxl: int = 16
    #: Collapsed TODO pill; half of its 34px height, so it is not a grid step.
    pill: int = 17
    #: Toast capsule; one pixel above ``pill`` in the original stylesheet.
    pill_lg: int = 18


@dataclass(frozen=True, slots=True)
class TypeScale:
    """Font sizes (px), weights, and the interface family default."""

    micro: int = 9
    xs: int = 11
    sm: int = 12
    #: Application default size.
    base: int = 13
    md: int = 14
    lg: int = 16
    xl: int = 18
    xxl: int = 20
    display: int = 22

    weight_semibold: int = 600
    weight_bold: int = 700

    #: Family installed by ``apply_theme`` and used as the fallback in
    #: ``interface_font`` when the user picked no fixed-width family.
    base_family: str = "Segoe UI"


@dataclass(frozen=True, slots=True)
class TrackingScale:
    """Letter-spacing steps, in px.

    Always emitted with a unit. A bare ``0`` is an invalid declaration that Qt drops
    on the floor, which used to make the reset rules lose to the very rules they
    were written to override.
    """

    none: float = 0.0
    wide: float = 0.4
    wider: float = 0.8


@dataclass(frozen=True, slots=True)
class Sizing:
    """Control metrics that are neither spacing nor radius."""

    #: Every ``1px solid`` line in the stylesheet.
    border_width: int = 1
    #: Left marker on a selected list row.
    selection_bar_width: int = 3
    #: Bare icon buttons (expand prompt, close pad).
    icon_button_min_width: int = 22
    #: History navigation buttons.
    nav_button_min_width: int = 28
    #: Send / cancel buttons.
    action_button_min_width: int = 64
    scrollbar_width: int = 10
    scrollbar_handle_min_height: int = 24
    #: Fixed height of compact chrome controls: the history navigation buttons and
    #: the home toolbar's select / delete buttons. Numerically equal to
    #: ``nav_button_min_width`` by coincidence, so the two stay separate roles.
    compact_control_height: int = 28
    #: Fixed height of the primary action row (send / cancel, in the composer and
    #: in the prompt pad).
    action_control_height: int = 36
    #: Floor for a settings sidebar row, for the case where two lines of a small
    #: font would make the row too cramped to click comfortably.
    settings_row_min_height: int = 44
    #: Vertical padding added around the two lines of a settings sidebar row.
    settings_row_padding: int = 14


@dataclass(frozen=True, slots=True)
class CardPadding:
    """``setContentsMargins`` insets for card-like containers, ``(l, t, r, b)``.

    Three combinations exist today. They are recorded as three named steps rather
    than collapsed onto one value, because every collapse would move pixels.
    """

    #: Page-level cards: the home view root and the composer pad card.
    wide: tuple[int, int, int, int] = (20, 16, 20, 16)
    #: The home view's session detail card.
    detail: tuple[int, int, int, int] = (18, 16, 18, 16)
    #: Small inline cards, such as the preferences font preview.
    compact: tuple[int, int, int, int] = (14, 12, 14, 12)


@dataclass(frozen=True, slots=True)
class Motion:
    """Animation durations, in milliseconds."""

    #: Toast fade-out (``qt/main_window.py``).
    fade_ms: int = 160
    #: How long the collapsed TODO pill keeps its accent outline after an update
    #: (``qt/todo_panel.py``).
    flash_ms: int = 1600


@dataclass(frozen=True, slots=True)
class TranscriptMetrics:
    """Painted geometry of one transcript card (``qt/transcript.py``).

    None of these are grid steps: they are the literals the delegate has always
    used, pinned byte-for-byte by ``tests/gui/test_render_geometry.py``.
    """

    #: Horizontal inset of header and body text inside the card.
    pad_x: int = 12
    #: Bottom padding below the body text.
    pad_y: int = 8
    #: Gap between the row rect and the painted card.
    card_margin_x: int = 8
    card_margin_y: int = 4
    #: Record color bar, at rest and while the pointer is in its hit zone.
    bar_width: int = 3
    bar_width_hover: int = 5
    #: The bar is only ``bar_width`` wide, so the whole left gutter toggles the
    #: row. It stops where the body text starts, keeping text selection untouched.
    bar_hit_width: int = 12
    #: Height of the header band, which also drives the compact row height.
    header_height: int = 22
    #: Floor for a measured body height, and the step used to estimate scroll.
    line_height: int = 18
    #: Width of the copy affordance parked at the card's right edge.
    copy_width: int = 36


@dataclass(frozen=True, slots=True)
class TodoPanelMetrics:
    """Geometry of the floating TODO panel (``qt/todo_panel.py``)."""

    #: Gap kept between the panel and its host's top-right corner.
    margin: int = 14
    #: Preferred and minimum width of the expanded card.
    card_width: int = 336
    min_card_width: int = 200
    #: Clickable header band, which is also the collapsed pill's body.
    header_height: int = 34
    #: Three-tone progress strip under the header.
    bar_height: int = 3
    #: Status summary line at the bottom of the expanded card.
    footer_height: int = 26
    #: One todo row, without and with a notes line.
    row_height: int = 28
    row_with_notes_height: int = 44
    row_spacing: int = 2
    #: Inset around the row list; also feeds the minimum body height, which is
    #: ``row_height + 2 * body_padding`` rather than a literal of its own.
    body_padding: int = 6
    max_body_height: int = 320
    #: Extra left inset per nesting level of the todo tree.
    indent_step: int = 14


@dataclass(frozen=True, slots=True)
class SessionListMetrics:
    """Geometry of the home view's session list (``qt/home_view.py``).

    These rows are hand-painted, and stay that way: the active-row marker is an
    inset rounded bar, which no style sheet declaration can place. What the painter
    must not keep is the numbers, so they live here and its colours come from
    :class:`~kimix_gui.design.palette.Palette` -- a new theme reaches the rows even
    though no rule selects them.
    """

    #: Fixed height of one row, set as the list item's size hint.
    row_height: int = 58
    #: Corner radius of the hover and active fills.
    row_radius: int = 10
    #: Side of the circular selection mark.
    mark_size: int = 22
    #: An unchecked, un-hovered mark is drawn faded instead of hidden, so the
    #: column keeps its width and the affordance stays discoverable.
    mark_idle_opacity: float = 0.42
    #: Active-row marker: left inset, width, vertical inset at each end, radius.
    #: Half-pixel offsets keep the 1px stroke on the pixel grid.
    marker_x: float = 1.5
    marker_width: int = 3
    marker_inset_y: int = 14
    marker_radius: float = 1.5


@dataclass(frozen=True, slots=True)
class ComposerMetrics:
    """Heights of the chat composer (``qt/composer.py``).

    The action row height lives in :class:`Sizing` instead, because the chat view's
    send / cancel buttons share it.
    """

    #: Collapsed height, and the floor the auto-grow never goes below.
    min_height: int = 52
    #: Ceiling after which the composer scrolls instead of growing.
    max_height: int = 130


@dataclass(frozen=True, slots=True)
class Breakpoints:
    """Width thresholds (px) at which a split layout stacks vertically.

    Both steps answer the same question -- "is this window too narrow to keep the
    splitter horizontal" -- yet carry different values, with no comment in the
    original code explaining why. ``settings_narrow`` happens to equal the sum of
    that dialog's default pane sizes (``320 + 560``) while ``home_narrow`` does not
    match the home splitter's ``380 + 640``, which is suggestive but not proof.
    **Values differ, reason unknown, to be confirmed**: they are kept apart so that
    naming them cannot change either view's behavior.
    """

    #: ``qt/home_view.py``, ``HomeView._sync_narrow``.
    home_narrow: int = 780
    #: ``qt/settings_dialog.py``, ``LLMSettingsDialog.resizeEvent``.
    settings_narrow: int = 880
