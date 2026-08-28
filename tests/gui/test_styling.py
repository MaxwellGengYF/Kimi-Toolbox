"""Guards for the dynamic-property style vocabulary.

The point of :mod:`kimix_gui.qt.styling` is that ``objectName`` stops carrying
appearance. These tests keep the two halves honest: the vocabulary must match
the attribute selectors in the stylesheet in *both* directions, and the widgets
must actually be tagged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QWidget,
)

from kimix_gui.design import DARK
from kimix_gui.llm import resolved_provider_file
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt import styling
from kimix_gui.qt.bridge import KimixBridge
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.composer import ComposerPad
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.preferences_dialog import PreferencesDialog
from kimix_gui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog
from kimix_gui.qt.styling import (
    CARD,
    FLASH,
    KIND,
    LEVEL,
    METRIC,
    MODE,
    ROLE,
    STATE,
    SURFACE,
    TONE,
    VARIANT,
    CardLevel,
    Level,
    Metric,
    Role,
    Surface,
    Tone,
    Variant,
    repolish,
    set_style_property,
    style,
)
from kimix_gui.qt.theme import build_stylesheet
from kimix_gui.qt.todo_panel import TodoPanel
from kimix_gui.session_index import SessionSummary
from kimix_gui.todos import TodoEntry, TodoSnapshot


def _provider_config(tmp_path: Path) -> Path:
    path = tmp_path / "provider.json"
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "capabilities": ["thinking"],
                "thinking_effort": "high",
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    return path


#: Every property name the stylesheet is allowed to select on.
PROPERTY_NAMES: frozenset[str] = frozenset(
    {VARIANT, ROLE, TONE, LEVEL, SURFACE, METRIC, CARD, STATE, KIND, MODE, FLASH}
)

#: ``property name -> the class that owns its values``. ``state`` / ``kind`` /
#: ``mode`` / ``flash`` are missing on purpose: they carry domain values that
#: belong to their widget, not to the appearance vocabulary.
VALUE_HOLDERS: dict[str, type] = {
    VARIANT: Variant,
    ROLE: Role,
    TONE: Tone,
    SURFACE: Surface,
    METRIC: Metric,
    LEVEL: Level,
    CARD: CardLevel,
}


def _constants(holder: type) -> frozenset[str]:
    return frozenset(
        value
        for name, value in vars(holder).items()
        if not name.startswith("_") and isinstance(value, str)
    )


#: Derived rather than hand-written: a value added to a holder class without a
#: matching rule then has to fail the round-trip below, instead of sitting as
#: dead vocabulary until someone remembers to update a literal here.
DECLARED_VALUES: dict[str, frozenset[str]] = {
    name: _constants(holder) for name, holder in VALUE_HOLDERS.items()
}

#: ``objectName``s that used to carry appearance and must not select style any
#: more. They stay on the widgets as test hooks, so only the QSS is checked.
FREED_OBJECT_NAMES: frozenset[str] = frozenset(
    {
        "apply-preferences",
        "apply-settings",
        "approve",
        "cancel-pad",
        "cancel-prompt",
        "chatgpt-config-group",
        "chat-header",
        "chat-title",
        "chat-toolbar",
        "composer-pad-card",
        "composer-pad-count",
        "composer-pad-hint",
        "config-details-title",
        "confirm-delete",
        "context",
        "detail-overline",
        "detail-state",
        "expand-prompt",
        "close-composer-pad",
        "font-preview",
        "font-preview-label",
        "history-info",
        "history-title",
        "history-toolbar",
        "home-model",
        "home-path",
        "home-status",
        "home-title",
        "home-toolbar",
        "jump-latest",
        "load-newer",
        "load-older",
        "open-session",
        "preferences-description",
        "preferences-page-title",
        "preferences-subtitle",
        "preferences-title",
        "provider-config-group",
        "reject",
        "select-shown",
        "selection-count",
        "send-pad",
        "send-prompt",
        "session-count",
        "session-detail",
        "session-meta",
        "settings-scope",
        "settings-title",
        "start-new-session",
        "status",
        "todo-item-notes",
        "todo-title",
    }
)

_ATTRIBUTE = re.compile(r"\[(?P<name>[\w-]+)\s*=\s*\"(?P<value>[^\"]*)\"\]")


def _selectors(stylesheet: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"([^{}]+)\{[^}]*\}", stylesheet)]


@pytest.fixture(scope="module")
def stylesheet() -> str:
    return build_stylesheet(DARK)


def test_the_stylesheet_only_selects_on_declared_property_names(stylesheet: str) -> None:
    used = {match.group("name") for match in _ATTRIBUTE.finditer(stylesheet)}
    assert used <= PROPERTY_NAMES, sorted(used - PROPERTY_NAMES)


def test_the_stylesheet_only_selects_on_declared_values(stylesheet: str) -> None:
    """A typo in either half is a silently unstyled widget, so pin both."""

    used: dict[str, set[str]] = {}
    for match in _ATTRIBUTE.finditer(stylesheet):
        used.setdefault(match.group("name"), set()).add(match.group("value"))
    for name, declared in DECLARED_VALUES.items():
        assert used.get(name, set()) == declared, name


def test_every_declared_value_is_actually_used_by_a_rule(stylesheet: str) -> None:
    """The other direction: a value with no rule is dead vocabulary."""

    used = {(m.group("name"), m.group("value")) for m in _ATTRIBUTE.finditer(stylesheet)}
    declared = {(name, value) for name, values in DECLARED_VALUES.items() for value in values}
    assert declared - used == set()


@pytest.mark.parametrize("name", sorted(PROPERTY_NAMES))
def test_no_property_name_shadows_a_real_qt_property(qapp, name: str) -> None:
    """``size`` looked like the obvious name for the metric preset and is a trap.

    ``QWidget.size`` is a declared ``QSize`` property, so ``setProperty`` on it
    resizes the widget instead of tagging it -- silently, and only visible as
    wrong geometry. Any future name has to clear the same bar.
    """

    classes = (
        QWidget,
        QLabel,
        QPushButton,
        QFrame,
        QLineEdit,
        QPlainTextEdit,
        QListWidget,
        QScrollArea,
        QDialog,
    )
    for cls in classes:
        meta = cls.staticMetaObject
        declared = {meta.property(i).name() for i in range(meta.propertyCount())}
        assert name not in declared, f"{name} shadows {cls.__name__}.{name}"


@pytest.mark.parametrize("name", sorted(FREED_OBJECT_NAMES))
def test_freed_object_names_no_longer_appear_in_the_stylesheet(stylesheet: str, name: str) -> None:
    """These names are test hooks now; their looks come from properties."""

    offenders = [selector for selector in _selectors(stylesheet) if f"#{name}" in selector]
    assert offenders == []


def test_letter_spacing_is_never_emitted_without_a_unit(stylesheet: str) -> None:
    """A bare ``0`` is dropped by Qt's parser, so a reset rule loses silently.

    That is not hypothetical: ``home-title`` and ``preferences-title`` shipped
    with the 0.6px tracking they were written to cancel.
    """

    bare = re.findall(r"letter-spacing:\s*([\d.]+)\s*;", stylesheet)
    assert bare == [], bare
    assert re.search(r"letter-spacing:\s*0\s*;", stylesheet) is None


def test_style_sets_only_the_properties_it_was_given(qapp) -> None:
    widget = QLabel()
    style(widget, role=Role.CAPTION)
    assert widget.property(ROLE) == Role.CAPTION
    assert widget.property(TONE) is None
    assert widget.property(VARIANT) is None


def test_style_returns_the_widget_for_inline_use(qapp) -> None:
    widget = QLabel()
    assert style(widget, tone=Tone.MUTED) is widget


def test_set_style_property_reports_whether_it_changed_anything(qapp) -> None:
    widget = QLabel()
    assert set_style_property(widget, TONE, Tone.MUTED) is True
    assert set_style_property(widget, TONE, Tone.MUTED) is False
    assert set_style_property(widget, TONE, Tone.DANGER) is True


def test_repolish_makes_a_late_property_change_take_effect(qapp) -> None:
    """Without the repolish the new property is invisible until the next polish."""

    app = QApplication.instance()
    assert app is not None
    app.setStyleSheet(build_stylesheet(DARK))
    label = QLabel("boom")
    label.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    label.show()
    label.ensurePolished()
    plain = label.palette().color(label.palette().ColorRole.WindowText).name()

    label.setProperty(TONE, Tone.DANGER)
    assert label.palette().color(label.palette().ColorRole.WindowText).name() == plain

    repolish(label)
    assert label.palette().color(label.palette().ColorRole.WindowText).name() == DARK.palette.error


def test_repolish_schedules_a_repaint(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    """The two hand-rolled copies disagreed on this; only one called update()."""

    label = QLabel()
    calls: list[int] = []
    monkeypatch.setattr(type(label), "update", lambda self, *a: calls.append(1))
    repolish(label)
    assert calls == [1]


def test_the_vocabulary_classes_expose_only_string_constants() -> None:
    """Keeps ``_constants`` honest: it reads every public attribute, so a helper
    or a nested class on a holder would silently widen the declared vocabulary.
    """

    for holder in VALUE_HOLDERS.values():
        public = {name: value for name, value in vars(holder).items() if not name.startswith("_")}
        assert public
        for name, value in public.items():
            assert name.isupper(), f"{holder.__name__}.{name}"
            assert isinstance(value, str), f"{holder.__name__}.{name}"


def test_the_module_exports_match_its_public_names() -> None:
    exported = set(styling.__all__)
    public = {name for name in vars(styling) if not name.startswith("_") and name.isupper()}
    assert public <= exported


# ---- the other half: widgets have to actually carry the properties ----------


#: ``objectName -> expected style properties``, for the widgets whose looks moved
#: from an ID selector onto a property. A rule with nothing carrying its property
#: is as broken as a widget with no rule, and neither shows up in the QSS checks.
EXPECTED_PROPERTIES: dict[str, dict[str, str]] = {
    # chat view
    "chat-header": {SURFACE: Surface.BAR},
    "chat-toolbar": {SURFACE: Surface.BAR},
    "history-toolbar": {SURFACE: Surface.BAR},
    "chat-title": {ROLE: Role.TITLE},
    "status": {TONE: Tone.MUTED},
    "history-info": {TONE: Tone.MUTED},
    "context": {TONE: Tone.MUTED},
    "load-older": {METRIC: Metric.NAV},
    "load-newer": {METRIC: Metric.NAV},
    "jump-latest": {METRIC: Metric.NAV},
    "send-prompt": {VARIANT: Variant.PRIMARY, METRIC: Metric.ACTION},
    "cancel-prompt": {VARIANT: Variant.DANGER, METRIC: Metric.ACTION},
    "expand-prompt": {VARIANT: Variant.ICON},
    # home view
    "home-toolbar": {SURFACE: Surface.BAR},
    "home-title": {ROLE: Role.DISPLAY, LEVEL: Level.ONE},
    "home-path": {TONE: Tone.MUTED},
    "home-model": {TONE: Tone.MUTED},
    "home-status": {TONE: Tone.MUTED},
    "session-count": {TONE: Tone.MUTED},
    "history-title": {ROLE: Role.SECTION},
    "selection-count": {ROLE: Role.SECTION},
    "select-shown": {VARIANT: Variant.GHOST},
    "delete-sessions": {VARIANT: Variant.DANGER},
    "start-new-session": {VARIANT: Variant.PRIMARY},
    "detail-overline": {ROLE: Role.OVERLINE},
    "detail-state": {TONE: Tone.MUTED},
    "key-value-key": {TONE: Tone.MUTED},
    "open-session": {VARIANT: Variant.PRIMARY},
    "session-detail": {CARD: CardLevel.PANEL},
    # composer pad
    "composer-pad-card": {CARD: CardLevel.FLOATING},
    "close-composer-pad": {VARIANT: Variant.ICON},
    "composer-pad-hint": {ROLE: Role.CAPTION},
    "composer-pad-count": {ROLE: Role.CAPTION},
    "send-pad": {VARIANT: Variant.PRIMARY, METRIC: Metric.ACTION},
    "cancel-pad": {VARIANT: Variant.DANGER, METRIC: Metric.ACTION},
    # todo panel
    "todo-title": {ROLE: Role.OVERLINE},
    "todo-chevron": {ROLE: Role.FOOTNOTE},
    "todo-footer": {ROLE: Role.FOOTNOTE},
    # ``state`` is domain data driven by the snapshot, so only the role is pinned
    # here; ``tests/gui/test_todo_panel.py`` owns the state transitions.
    "todo-dot": {ROLE: Role.MARKER},
    "todo-glyph": {ROLE: Role.MARKER},
    "todo-item-notes": {ROLE: Role.FOOTNOTE},
    # dialogs
    "settings-title": {ROLE: Role.DISPLAY, LEVEL: Level.TWO},
    "settings-scope": {TONE: Tone.MUTED},
    "settings-error": {TONE: Tone.DANGER},
    "config-details-title": {ROLE: Role.TITLE},
    "model-details-title": {ROLE: Role.OVERLINE},
    "provider-details-title": {ROLE: Role.OVERLINE},
    "provider-card": {CARD: CardLevel.PANEL},
    "empty-provider-card": {TONE: Tone.MUTED},
    "model-parameters-title": {ROLE: Role.OVERLINE},
 "param-thinking_effort": {STATE: "available"},
    "param-thinking_effort-problem": {TONE: Tone.DANGER},
    "chatgpt-config-group": {VARIANT: Variant.DISCLOSURE},
    "provider-config-group": {VARIANT: Variant.DISCLOSURE},
    "apply-settings": {VARIANT: Variant.PRIMARY},
    "preferences-title": {ROLE: Role.DISPLAY, LEVEL: Level.TWO},
    "preferences-subtitle": {TONE: Tone.MUTED},
    "preferences-page-title": {ROLE: Role.TITLE},
    "preferences-description": {TONE: Tone.MUTED},
    "apply-preferences": {VARIANT: Variant.PRIMARY},
    "codex-account-card": {CARD: CardLevel.INSET},
    "codex-account-title": {ROLE: Role.TITLE},
    "codex-account-description": {TONE: Tone.MUTED},
    "codex-account-status": {TONE: Tone.MUTED},
    "connect-chatgpt": {VARIANT: Variant.PRIMARY},
    "connect-chatgpt-models": {VARIANT: Variant.PRIMARY},
    "refresh-codex-models": {VARIANT: Variant.GHOST},
    "disconnect-chatgpt": {VARIANT: Variant.DANGER},
    "approve": {VARIANT: Variant.PRIMARY},
    "reject": {VARIANT: Variant.DANGER},
    "confirm-delete": {VARIANT: Variant.DANGER},
    "session-meta": {ROLE: Role.CAPTION},
    "font-preview": {CARD: CardLevel.INSET},
    "font-preview-label": {ROLE: Role.OVERLINE},
    "config-sources-title": {ROLE: Role.TITLE},
    "dialog-title": {ROLE: Role.SECTION},
    "delete-title": {ROLE: Role.SECTION},
    "delete-copy": {TONE: Tone.MUTED},
}


_OPAQUE_CONTROL_DECLARATIONS = frozenset({"background", "color"})

#: ``objectName -> (widget type, exact element selector, required declarations)``.
#: Unlike a prose reason, every part of this decision is checked against both the
#: runtime gallery and the generated QSS.  A false claim such as "QComboBox element
#: rule" therefore cannot make an actually unstyled control look accounted for.
ELEMENT_STYLED: dict[str, tuple[type[QWidget], str, frozenset[str]]] = {
    # Dialog and view roots.
    "approval-dialog": (QDialog, "QDialog", frozenset({"background"})),
    "question-dialog": (QDialog, "QDialog", frozenset({"background"})),
    "delete-dialog": (QDialog, "QDialog", frozenset({"background"})),
    "settings-dialog": (QDialog, "QDialog", frozenset({"background"})),
    "preferences-pages": (QStackedWidget, "QStackedWidget", frozenset({"background"})),
    # Inputs.
    "answer": (QLineEdit, "QLineEdit", _OPAQUE_CONTROL_DECLARATIONS),
    "config-path": (QLineEdit, "QLineEdit", _OPAQUE_CONTROL_DECLARATIONS),
    "session-search": (QLineEdit, "QLineEdit", _OPAQUE_CONTROL_DECLARATIONS),
    "font-primary": (QComboBox, "QComboBox", _OPAQUE_CONTROL_DECLARATIONS),
    "font-fallback": (QComboBox, "QComboBox", _OPAQUE_CONTROL_DECLARATIONS),
    "interface-language": (QComboBox, "QComboBox", _OPAQUE_CONTROL_DECLARATIONS),
    "interface-theme": (QComboBox, "QComboBox", _OPAQUE_CONTROL_DECLARATIONS),
    "font-size": (QSpinBox, "QSpinBox", _OPAQUE_CONTROL_DECLARATIONS),
    "approval-payload": (QTextEdit, "QTextEdit", _OPAQUE_CONTROL_DECLARATIONS),
    # Secondary buttons: the bare QPushButton rule is their intentional look.
    **{
        name: (QPushButton, "QPushButton", _OPAQUE_CONTROL_DECLARATIONS)
        for name in (
            "approve-for-session",
            "browse-config",
            "cancel-delete",
            "cancel-preferences",
            "cancel-settings",
            "close-settings",
            "configure-session",
            "delete-config",
            "leave-session",
            "load-config",
            "manage-llm-settings",
            "open-settings",
            "provider-model-row",
        )
    },
}


#: ``objectName -> why it deliberately has no dedicated QSS decision``. Every
#: name must land in this register, an ID rule, a dynamic property rule, or the
#: mechanically checked ``ELEMENT_STYLED`` register above.
NO_QSS_BY_DESIGN: dict[str, str] = {
    # Layout-only containers: they exist to be found by tests and to hold a
    # layout. Giving them a surface would double-paint the panel behind them.
    "config-details": "layout-only container",
    "dialog-footer": "layout-only button row; DialogFooter owns the order, not a look",
    "config-sources": "layout-only container",
    "provider-file-picker": "layout-only container inside the Provider files group",
    "provider-card-body": "layout-only container inside one provider card",
    "provider-cards": "scrolling viewport; provider cards own their surfaces",
    "provider-cards-content": "layout-only host for provider cards",
    "model-parameters": "layout-only container for metadata-driven parameter controls",
       "param-thinking_effort-row": "layout-only parameter row",
    "param-thinking_effort-label": "base label style paired with its picker",
    "detail-metadata": "layout-only container",
    "preferences-appearance": "layout-only page container",
    "preferences-models": "layout-only page container",
    "session-browser": "layout-only container",
    "composer-pad-header": "layout-only container",
    "history-header": "layout-only container",
    "todo-list": "layout-only container",
    "session-row": "hand-painted; see SessionRow.paintEvent for why it stays that way",
    "home-workspace": "QSplitter, layout only",
    "settings-body": "QSplitter, layout only",
    "question-detail": "body copy at base label style, next to the section heading",
    # Value labels of the two key/value lists: base label style on purpose, the
    # muted key next to them carries the contrast.
    **{
        name: "key/value list value, base label style"
        for name in (
            "selection-model",
            "selection-source",
            "selection-status",
            "model-id",
            "model-context",
            "model-output",
            "model-capabilities",
            "model-modalities",
            "provider-type",
            "provider-endpoint",
            "provider-credential",
            "provider-format",
            "provider-thinking",
            "detail-updated",
            "detail-llm",
            "detail-provider",
            "detail-config",
            "detail-size",
            "detail-storage",
            "detail-todos",
            "detail-directories",
            "detail-id",
            "detail-path",
        )
    },
    # Custom painters: their colours come from tokens inside ``paintEvent``.
    "todo-progress": "custom paintEvent, colours read from tokens",
    "font-preview-text": "font is set per preview, that is the whole point",
    "inherit-project-default": "native checkbox; it carries selection semantics, not appearance",
}


def _id_selectors(stylesheet: str) -> set[str]:
    without_colours = re.sub(r"#[0-9a-fA-F]{6}\b", "", stylesheet)
    return set(re.findall(r"#([a-z][\w-]*)", without_colours))


def _tagged_widgets(roots: list[QWidget]) -> dict[str, list[QWidget]]:
    found: dict[str, list[QWidget]] = {}
    for root in roots:
        for widget in [root, *root.findChildren(QWidget)]:
            name = widget.objectName()
            # ``qt_`` names belong to Qt's own internals (scroll area viewports,
            # spin box line edits, splitter handles); they are not ours to style.
            if name and not name.startswith("qt_"):
                found.setdefault(name, []).append(widget)
    return found


@pytest.fixture
def style_gallery(qtbot, tmp_path: Path) -> dict[str, list[QWidget]]:
    """One instance of every view that owns a tagged widget."""

    reference = resolved_provider_file(_provider_config(tmp_path))
    home = HomeView(tmp_path, default_config=reference, session_config_loader=lambda _id: None)
    home.show_sessions(
        [
            SessionSummary(
                id="sess-1",
                title="Fix login",
                updated_at=1_700_000_000.0,
                size_bytes=42 * 1024,
                file_count=4,
                storage_format="SQLite",
            )
        ]
    )
    # ``KimixBridge()`` without ``start()``: the chrome needs no worker thread,
    # and an orphan loop is the GC hazard ``conftest`` spends effort avoiding.
    roots: list[QWidget] = [
        home,
        ChatView(KimixBridge()),
        ComposerPad("draft"),
        PreferencesDialog(InterfacePreferences(), font_families=list),
        LLMSettingsDialog(
            current=reference,
            models=(reference.model,),
            scope_label="New session",
            project_default=reference,
            manage_library=True,
        ),
        # Read-only is a second chrome, not a disabled variant of the first: it
        # swaps ``cancel-settings`` for ``close-settings`` and drops the library
        # controls, so both spellings need an instance to be reachable here.
        LLMSettingsDialog(
            current=reference,
            models=(reference.model,),
            scope_label="Active session",
            read_only=True,
        ),
        ApprovalDialog("Run bash", "ls -la"),
        QuestionDialog("Which branch?", "main or dev"),
        DeleteSessionsDialog(3),
    ]
    panel = TodoPanel()
    panel.set_snapshot(
        TodoSnapshot(
            entries=(
                TodoEntry("Write the panel", "in_progress", notes="with a note"),
                TodoEntry("Run tests", "pending"),
            )
        )
    )
    roots.append(panel)
    for root in roots:
        qtbot.addWidget(root)
    return _tagged_widgets(roots)


def test_every_freed_object_name_carries_its_replacement_property(
    qtbot, style_gallery: dict[str, list[QWidget]]
) -> None:
    missing = sorted(FREED_OBJECT_NAMES - set(EXPECTED_PROPERTIES))
    assert missing == [], missing
    unreachable = sorted(set(EXPECTED_PROPERTIES) - set(style_gallery))
    assert unreachable == [], unreachable

    wrong: list[str] = []
    for name, expected in EXPECTED_PROPERTIES.items():
        for widget in style_gallery[name]:
            actual = {key: widget.property(key) for key in expected}
            if actual != expected:
                wrong.append(f"{name}: {actual} != {expected}")
    assert wrong == []


def test_the_two_full_size_dialogs_head_themselves_the_same_way(
    style_gallery: dict[str, list[QWidget]],
) -> None:
    """One answer to "what is a dialog heading", and it outranks the sections below it.

    ``settings-title`` was ``title`` -- the same role its own ``Sources`` and
    ``Details`` section titles carry -- while ``preferences-title`` was ``display``
    level 2. Two dialogs of the same weight opened with headings of different sizes,
    and one of them had no heading level at all.
    """
    heading = {ROLE: Role.DISPLAY, LEVEL: Level.TWO}
    for name in ("settings-title", "preferences-title"):
        for widget in style_gallery[name]:
            assert {key: widget.property(key) for key in heading} == heading, name

    for name in ("config-sources-title", "config-details-title", "preferences-page-title"):
        for widget in style_gallery[name]:
            assert widget.property(ROLE) == Role.TITLE, name

    for name in ("model-details-title", "provider-details-title"):
        for widget in style_gallery[name]:
            assert widget.property(ROLE) == Role.OVERLINE, name


def test_unavailable_parameter_style_is_axis_generic(stylesheet: str) -> None:
    assert 'QComboBox[state="unavailable"]' in stylesheet
    assert "model-parameter-picker" not in stylesheet


def test_every_object_name_has_a_styling_decision(
    stylesheet: str, style_gallery: dict[str, list[QWidget]]
) -> None:
    """No name may sit in the gap between the four mechanically distinct answers.

    ``settings-error`` lived in that gap: it looked like a styled widget because
    it had a name, and the missing rule only showed as an error message rendered
    in ordinary text. This is the check that closes the gap for good.
    """

    styled_by_id = _id_selectors(stylesheet)
    undecided = sorted(
        set(style_gallery)
        - styled_by_id
        - set(EXPECTED_PROPERTIES)
        - set(ELEMENT_STYLED)
        - set(NO_QSS_BY_DESIGN)
    )
    assert undecided == [], undecided


def test_no_styling_decision_is_stale(
    stylesheet: str, style_gallery: dict[str, list[QWidget]]
) -> None:
    """Registers must stay reachable and mutually exclusive."""

    registered = set(ELEMENT_STYLED) | set(NO_QSS_BY_DESIGN)
    gone = sorted(name for name in registered if name not in style_gallery)
    assert gone == [], gone
    base_decisions = {
        "property": set(EXPECTED_PROPERTIES),
        "element": set(ELEMENT_STYLED),
        "no QSS": set(NO_QSS_BY_DESIGN),
    }
    contradictions: list[str] = []
    labels = list(base_decisions)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            for name in sorted(base_decisions[left] & base_decisions[right]):
                contradictions.append(f"{name}: {left} and {right}")
    for name in sorted(_id_selectors(stylesheet) & set(NO_QSS_BY_DESIGN)):
        contradictions.append(f"{name}: ID rule and no QSS")
    assert contradictions == []


@pytest.mark.parametrize("name", sorted(NO_QSS_BY_DESIGN))
def test_a_widget_left_unstyled_carries_no_style_property(
    style_gallery: dict[str, list[QWidget]], name: str
) -> None:
    for widget in style_gallery[name]:
        carried = {key: widget.property(key) for key in PROPERTY_NAMES}
        assert {key: value for key, value in carried.items() if value is not None} == {}, name


#: Every rule allowed to declare ``background: transparent``, and why it needs to.
#: ``QWidget`` matches subclasses, so its declaration already reaches every widget in
#: the app; a second one is only load-bearing where a *more specific* rule would
#: otherwise win. Eleven rules restated the base for nothing -- proven by deleting
#: each one and re-rendering all twelve gallery scenes, which moved no pixels.
TRANSPARENT_ON_PURPOSE: dict[str, str] = {
    "QWidget": "the base rule; the single place transparency is declared",
    # Deleting this one moves no pixels either, but it is not a restatement of the
    # base rule -- it contradicts ``QDialog { background: bg }``. The reason it is
    # unobservable is ``WA_TranslucentBackground``, set in Python, which makes Qt skip
    # the window fill entirely. Two mechanisms, one of them in the other layer, so the
    # declaration stays and says what the pad wants on its own terms.
    "QDialog#composer-pad": "contradicts the QDialog base rule, which paints bg",
    'QPushButton[variant="ghost"], QPushButton[variant="icon"]': (
        "overrides the QPushButton element rule, which paints panel"
    ),
    'QPushButton[variant="ghost"]:disabled': "same, for the disabled state",
    "QListView#transcript": "overrides the QListWidget, QListView element rule",
    "QListView#transcript::item": "a subcontrol; base rules do not reach it",
    "QListWidget#session-list::item": "a subcontrol; base rules do not reach it",
    "QListWidget#session-list::item:selected": "a subcontrol, and Qt would paint a "
    "selection highlight here",
    "QListWidget#session-list::item:hover": "a subcontrol, and Qt would paint a "
    "hover highlight here",
}

_RULE = re.compile(r"(?P<selector>[^{}]+?)\s*\{(?P<body>[^{}]*)\}", re.DOTALL)
_DECLARATION = re.compile(r"(?P<name>[\w-]+)\s*:")


def _declarations_for_selector(stylesheet: str, selector: str) -> set[str]:
    declarations: set[str] = set()
    for match in _RULE.finditer(stylesheet):
        alternatives = {" ".join(part.split()) for part in match.group("selector").split(",")}
        if selector in alternatives:
            declarations.update(
                declaration.group("name")
                for declaration in _DECLARATION.finditer(match.group("body"))
            )
    return declarations


def test_every_element_style_decision_names_a_real_rule(
    stylesheet: str, style_gallery: dict[str, list[QWidget]]
) -> None:
    """Prose must not be able to claim an element rule that the QSS lacks."""

    wrong: list[str] = []
    for name, (widget_type, selector, required) in ELEMENT_STYLED.items():
        for widget in style_gallery[name]:
            if not isinstance(widget, widget_type):
                wrong.append(f"{name}: {type(widget).__name__} is not {widget_type.__name__}")
        declarations = _declarations_for_selector(stylesheet, selector)
        missing = sorted(required - declarations)
        if missing:
            wrong.append(f"{name}: {selector} lacks {missing}")
    assert wrong == []


def test_combo_popup_rule_claims_an_opaque_theme_surface(stylesheet: str) -> None:
    required = {"background", "color", "selection-background-color", "selection-color"}
    declarations = _declarations_for_selector(stylesheet, "QComboBox QAbstractItemView")
    assert required - declarations == set()


def _selectors_declaring(stylesheet: str, declaration: str) -> set[str]:
    return {
        " ".join(match.group("selector").split())
        for match in _RULE.finditer(stylesheet)
        if declaration in match.group("body")
    }


def test_transparency_is_declared_once_and_overridden_only_on_purpose(
    stylesheet: str,
) -> None:
    """A restated default is worse than no rule: it reads as a decision.

    Someone changing the base rule has to find every copy of it, and a copy that
    happens to sit on a more specific selector silently outranks the change.
    """
    declaring = _selectors_declaring(stylesheet, "background: transparent")
    assert sorted(declaring) == sorted(TRANSPARENT_ON_PURPOSE), sorted(
        declaring.symmetric_difference(TRANSPARENT_ON_PURPOSE)
    )
