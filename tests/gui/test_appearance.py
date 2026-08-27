"""Does a runtime font / theme change actually reach what it should?

Qt broadcasts ``FontChange``, ``PaletteChange`` and ``StyleChange`` to *every*
widget, nested or not (probed: a label three containers deep hears all three).
What was missing was anyone listening. Four places cached a measurement or a
color taken from the font or the theme and never recomputed it, so the app could
be showing text at 26px inside rows measured for 13px.

These tests change the application font or the theme and then assert on the
cached thing, with no manual invalidation in between. That "no manual step" is
the whole point: ``test_theme.py`` already has a test showing that an explicit
``delegate.invalidate()`` reflows, which is the workaround, not the fix.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QStyleOptionViewItem

from kimix_gui.design import DARK
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt import theme
from kimix_gui.qt.components import SettingsList
from kimix_gui.qt.composer import Composer
from kimix_gui.qt.theme import build_stylesheet
from kimix_gui.qt.todo_panel import TodoPanel, _ElidedLabel
from kimix_gui.qt.transcript import Transcript
from kimix_gui.todos import TodoEntry, TodoSnapshot

from .transcript_helpers import append_text

SMALL = InterfacePreferences(font_size=13)
LARGE = InterfacePreferences(font_size=26)


def _resize_to(app: QApplication, preferences: InterfacePreferences) -> None:
    """Change the interface font the way the preferences dialog does, then let it land.

    ``QApplication.setFont`` does **not** update existing widgets synchronously: it
    stores the new default and posts ``ApplicationFontChange``, and the propagation
    down the widget tree happens when that event is delivered. Probed: without a
    turn of the event loop, ``widget.font()`` still reports the old family and
    ``widget.fontMetrics().lineSpacing()`` the old value, so an assertion made
    immediately after ``setFont`` tests nothing and passes for the wrong reason.

    Harmless in the running app -- the loop turns on the next iteration -- but every
    test here has to pump, and so does anything that reads a metric right after
    applying a font.
    """

    theme.apply_interface_font(app, preferences)
    app.processEvents()


# --------------------------------------------------------------------------- #
# The transcript: a document cache keyed without the font
# --------------------------------------------------------------------------- #


def _one_row_transcript(qtbot) -> tuple[Transcript, QStyleOptionViewItem]:
    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(
        transcript,
        "assistant",
        "a moderately long assistant answer that has to wrap across the row width " * 2,
    )
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 600, 0)
    return transcript, option


def test_a_font_change_reflows_the_transcript(qtbot, app_appearance: QApplication) -> None:
    """Doubling the interface font must make a wrapped row taller.

    This was a strict xfail: the cached body's key is
    ``(expanded, status, width, body, markdown, italic)`` with no font dimension, so
    every already-measured row kept the height it had when it was first laid out.
    """

    transcript, option = _one_row_transcript(qtbot)
    index = transcript.model().index(0, 0)
    _resize_to(app_appearance, SMALL)
    small = transcript._delegate.sizeHint(option, index).height()

    _resize_to(app_appearance, LARGE)

    assert transcript._delegate.sizeHint(option, index).height() > small


def test_a_font_change_relayouts_the_transcript_view(qtbot, app_appearance: QApplication) -> None:
    """The rows the user sees move, not just the numbers ``sizeHint`` returns.

    ``QListView`` keeps its own item rectangles, so a taller ``sizeHint`` only
    reaches the screen once the items layout runs again. It does, without the view
    asking: an explicit ``scheduleDelayedItemsLayout()`` in ``changeEvent`` was
    tried and removed because removing it changed nothing here.

    Same fix as the test above, different end of it -- that one reads the delegate's
    answer, this one reads the geometry the viewport will actually paint.
    """

    transcript, _ = _one_row_transcript(qtbot)
    transcript.show()
    qtbot.waitExposed(transcript)
    index = transcript.model().index(0, 0)
    _resize_to(app_appearance, SMALL)
    qtbot.waitUntil(lambda: transcript.visualRect(index).height() > 0)
    small = transcript.visualRect(index).height()

    _resize_to(app_appearance, LARGE)

    qtbot.waitUntil(lambda: transcript.visualRect(index).height() > small)


def _link_color(transcript: Transcript, index) -> str:
    """Read the color actually written into the cached document's anchor run."""

    document = transcript._delegate.document_for(QRect(0, 0, 600, 0), index)
    assert document is not None
    block = document.begin()
    while block.isValid():
        for span in block.textFormats():
            if span.format.isAnchor():
                return span.format.foreground().color().name()
        block = block.next()
    raise AssertionError("no anchor fragment in the rendered markdown")


def test_a_theme_change_recolors_cached_markdown_links(qtbot, app_appearance: QApplication) -> None:
    """The link color is baked into the document, so the cache holds it too.

    ``_apply_markdown_link_color`` merges the hue into the char format because Qt's
    Markdown reader ignores document CSS. That makes a cached document carry a
    color from a theme that may no longer be installed -- the one kind of staleness
    that repainting alone cannot fix, since the wrong color is *in* the document.
    """

    transcript = Transcript()
    qtbot.addWidget(transcript)
    transcript.resize(640, 400)
    append_text(
        transcript,
        "assistant",
        "see [the docs](https://example.invalid) for more",
        markdown=True,
    )
    index = transcript.model().index(0, 0)

    assert _link_color(transcript, index) == DARK.palette.link

    recolored = replace(DARK, palette=replace(DARK.palette, link="#ff00ff"))
    theme.apply_theme(app_appearance, recolored)

    assert _link_color(transcript, index) == "#ff00ff"

    # And the hue it shares a value with is a different role: retinting the palette
    # of *record categories* has no business repainting prose.
    hue_only = replace(recolored, categories=replace(recolored.categories, cyan="#00ff00"))
    theme.apply_theme(app_appearance, hue_only)

    assert _link_color(transcript, index) == "#ff00ff"


# --------------------------------------------------------------------------- #
# The composer: a fixed height computed from line spacing
# --------------------------------------------------------------------------- #


def test_a_font_change_resizes_the_composer(qtbot, app_appearance: QApplication) -> None:
    """Three lines of text need three lines of room, at whatever the current size is.

    ``_sync_height`` reads ``fontMetrics().lineSpacing()`` and calls
    ``setFixedHeight``. It ran on construction, on every text change and on show,
    but not on a font change -- so the box stayed sized for the old font while the
    text inside it grew, and the last lines were simply cut off.
    """

    composer = Composer()
    qtbot.addWidget(composer)
    composer.setPlainText("one\ntwo\nthree")
    _resize_to(app_appearance, SMALL)
    small = composer.height()

    _resize_to(app_appearance, LARGE)

    assert composer.height() > small


# --------------------------------------------------------------------------- #
# The settings sidebar: item size hints baked at construction
# --------------------------------------------------------------------------- #


def test_a_font_change_resizes_settings_sidebar_rows(qtbot, app_appearance: QApplication) -> None:
    """Row height is two lines of the current font, and it has to stay that way.

    The height used to be applied once by a free function at construction time,
    which meant the only way to get it right was to build the dialog again.
    """

    items = SettingsList()
    qtbot.addWidget(items)
    items.addItem("Appearance")
    items.addItem("Models")
    _resize_to(app_appearance, SMALL)
    small = items.item(0).sizeHint().height()

    _resize_to(app_appearance, LARGE)

    assert items.item(0).sizeHint().height() > small
    assert items.item(1).sizeHint().height() > small, "every row, not just the first"


def test_settings_rows_added_later_get_the_current_height(
    qtbot, app_appearance: QApplication
) -> None:
    """A row inserted after the font changed must not come back at the old height."""

    items = SettingsList()
    qtbot.addWidget(items)
    items.addItem("Appearance")
    _resize_to(app_appearance, LARGE)
    tall = items.item(0).sizeHint().height()

    items.addItem("added afterwards")

    assert items.item(1).sizeHint().height() == tall


# --------------------------------------------------------------------------- #
# The todo panel: an elide point measured against the old font
# --------------------------------------------------------------------------- #


def test_a_font_change_re_elides_a_truncated_label(qtbot) -> None:
    """More pixels per character means fewer characters fit before the ellipsis.

    ``_apply_elide`` ran on resize only, so after a font change the label kept a cut
    placed with the old metrics: too early at a smaller size (wasting room) and past
    the right edge at a larger one (clipping).

    The font is set on the label rather than on the application, because this label
    gets its size from the stylesheet -- see the test below.
    """

    label = _ElidedLabel("a deliberately long todo title that cannot possibly fit")
    qtbot.addWidget(label)
    label.resize(160, 20)
    small = theme.interface_font(SMALL)
    label.setFont(small)
    before = label.text()
    assert "…" in before, "the label is not narrow enough for this test to mean anything"

    label.setFont(theme.interface_font(LARGE))

    assert len(label.text()) < len(before)
    assert label.toolTip() == label.full_text, "the full text stays reachable"


def test_a_stylesheet_font_size_pins_a_widget_against_the_font_preference(
    qtbot, app_appearance: QApplication
) -> None:
    """Todo titles do not follow the interface font size, and this records that.

    ``QLabel#todo-item-title { font-size: ... }`` wins over the application font, so
    raising the interface font leaves these rows at their stylesheet size. Found the
    hard way: an earlier version of the test above drove the font through
    ``QApplication`` and passed alone but failed in the full suite, where an earlier
    test had installed the stylesheet.

    Recorded rather than fixed: whether the todo panel *should* track the preference
    is a design question, and silently making it track would move pixels nobody
    asked to move.
    """

    app_appearance.setStyleSheet(build_stylesheet(DARK))
    panel = TodoPanel()
    qtbot.addWidget(panel)
    panel.resize(240, 300)
    panel.set_snapshot(
        TodoSnapshot(entries=(TodoEntry("a title long enough to be cut short", "in_progress"),))
    )
    panel.set_expanded(True)
    panel.show()
    qtbot.waitExposed(panel)

    def title() -> QLabel:
        for label in panel.findChildren(QLabel):
            if label.objectName() == "todo-item-title":
                return label
        raise AssertionError("no todo-item-title in the expanded panel")

    _resize_to(app_appearance, SMALL)
    pinned = title().font().pixelSize()

    _resize_to(app_appearance, LARGE)

    assert title().font().pixelSize() == pinned == DARK.type_scale.sm


# --------------------------------------------------------------------------- #
# The registry: everything that caches an appearance-derived value listens
# --------------------------------------------------------------------------- #

#: Widget classes that hold state computed from the font or the theme, and so must
#: recompute it when Qt says either changed. Adding a cache without adding a
#: listener is the bug this whole file is about, so the list is asserted, not
#: documented.
LISTENS_FOR_APPEARANCE_CHANGES: dict[str, str] = {
    "Transcript": "delegate height/document caches: wrapping and link color",
    "Composer": "fixed height computed from fontMetrics().lineSpacing()",
    "SettingsList": "per-item sizeHint computed from fontMetrics().lineSpacing()",
    "_ElidedLabel": "elide point measured with QFontMetrics",
}


@pytest.mark.parametrize("name", sorted(LISTENS_FOR_APPEARANCE_CHANGES))
def test_the_widget_declares_a_change_handler(name: str) -> None:
    """Each listed class overrides ``changeEvent``; inheriting Qt's does nothing."""

    from kimix_gui.qt import composer, todo_panel, transcript
    from kimix_gui.qt.components import settings_list

    lookup = {
        "Transcript": transcript.Transcript,
        "Composer": composer.Composer,
        "SettingsList": settings_list.SettingsList,
        "_ElidedLabel": todo_panel._ElidedLabel,
    }
    widget = lookup[name]
    assert "changeEvent" in vars(widget), (
        f"{name} caches {LISTENS_FOR_APPEARANCE_CHANGES[name]} but does not override changeEvent"
    )


def test_a_widget_with_its_own_font_stays_deaf_on_purpose(
    qtbot, app_appearance: QApplication
) -> None:
    """The font preview must not follow the interface font; that is its whole job.

    Qt implements this for us: a widget with an explicitly assigned font carries
    ``WA_SetFont`` and receives no ``FontChange`` when the application font moves.
    Pinned here because it is the one place where the deafness is the feature, and
    a future "just call setFont everywhere" fix would break it silently.
    """

    from PySide6.QtCore import Qt

    label = QLabel("preview")
    qtbot.addWidget(label)
    pinned = theme.interface_font(InterfacePreferences(font_size=11))
    label.setFont(pinned)
    assert label.testAttribute(Qt.WidgetAttribute.WA_SetFont)

    _resize_to(app_appearance, LARGE)

    assert label.font().pixelSize() == 11


def test_settings_list_is_a_qlistwidget(qtbot) -> None:
    """The two settings dialogs pass it to code that expects the Qt API."""

    items = SettingsList()
    qtbot.addWidget(items)
    assert isinstance(items, QListWidget)
