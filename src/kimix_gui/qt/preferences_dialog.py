"""Expandable application preferences dialog with an appearance page."""

from __future__ import annotations

from collections.abc import Callable

from kimi_cli.auth.codex import (
    AUTH_DISCONNECTED,
    CodexAuthSnapshot,
    CodexModelCatalog,
    fallback_catalog,
)
from PySide6.QtCore import QT_TRANSLATE_NOOP, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from kimix_gui.design import SUPPORTED_THEMES, SYSTEM_THEME
from kimix_gui.i18n import SUPPORTED_LANGUAGES, SYSTEM_LANGUAGE
from kimix_gui.preferences import InterfacePreferences
from kimix_gui.qt.codex_dialog import CodexAccountCard
from kimix_gui.qt.components import Card, DialogFooter, SettingsList
from kimix_gui.qt.styling import CardLevel, Level, Role, Tone, Variant, style
from kimix_gui.qt.theme import available_monospace_families, interface_font

# Endonyms for the language picker: every option reads in the language it selects, so
# it stays recognizable no matter which catalog is installed. Never wrapped in tr().
# (Plain ``#`` comments, not ``#:`` -- lupdate reads ``#:`` as an extracomment and
# would staple these lines onto the next translatable string in this file.)
LANGUAGE_ENDONYMS: dict[str, str] = {"en": "English", "zh_CN": "中文"}

# The only entry with no language of its own. Both spellings are shown so the option
# stays readable to either audience without depending on the active catalog.
SYSTEM_LANGUAGE_LABEL = "跟随系统 · System"

# Theme names, unlike language names, are ordinary UI copy: they are read in whatever
# language the interface is already in, so they go through tr() like everything else.
# Declared here rather than inline so the picker cannot drift out of sync with
# ``SUPPORTED_THEMES`` -- a missing key raises instead of quietly dropping an option.
THEME_LABELS: dict[str, str] = {
    SYSTEM_THEME: QT_TRANSLATE_NOOP("PreferencesDialog", "Follow system"),
    "dark": QT_TRANSLATE_NOOP("PreferencesDialog", "Dark"),
    "light": QT_TRANSLATE_NOOP("PreferencesDialog", "Light"),
}


class PreferencesDialog(QDialog):
    """Edit application preferences while leaving room for later categories."""

    applied = Signal(object)

    # Category rows, in the order they are added below. The list and the stack are
    # index-coupled by ``currentRowChanged`` -> ``setCurrentIndex``, so the order is
    # real behaviour rather than presentation; naming it keeps callers (and tests) from
    # spelling out a bare number whose meaning is somewhere else.
    CATEGORY_APPEARANCE = 0
    CATEGORY_MODELS = 1
    manage_llm = Signal()
    connect_chatgpt = Signal()
    refresh_codex_models = Signal()
    disconnect_chatgpt = Signal()

    def __init__(
        self,
        preferences: InterfacePreferences,
        *,
        font_families: Callable[[], list[str]] = available_monospace_families,
        codex_snapshot: CodexAuthSnapshot | None = None,
        codex_catalog: CodexModelCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("preferences-dialog")
        self.setWindowTitle(self.tr("Settings"))
        self.setModal(True)
        self.resize(760, 500)
        self._preferences = preferences
        self._font_families = font_families()
        self._codex_snapshot = codex_snapshot or CodexAuthSnapshot(
            operation_id=0,
            state=AUTH_DISCONNECTED,
        )
        self._codex_catalog = codex_catalog or fallback_catalog()
        self._codex_card: CodexAccountCard | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel(self.tr("Settings"))
        title.setObjectName("preferences-title")
        style(title, role=Role.DISPLAY, level=Level.TWO)
        subtitle = QLabel(self.tr("Application preferences"))
        subtitle.setObjectName("preferences-subtitle")
        style(subtitle, tone=Tone.MUTED)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(20)
        self._categories = SettingsList()
        self._categories.setObjectName("preferences-categories")
        self._categories.setFixedWidth(150)
        for label in (self.tr("Appearance"), self.tr("Models")):
            self._categories.addItem(QListWidgetItem(label))
        body.addWidget(self._categories)

        self._pages = QStackedWidget()
        self._pages.setObjectName("preferences-pages")
        self._pages.addWidget(self._appearance_page())
        self._pages.addWidget(self._models_page())
        body.addWidget(self._pages, 1)
        root.addLayout(body, 1)

        cancel = QPushButton(self.tr("Cancel"))
        cancel.setObjectName("cancel-preferences")
        apply_button = QPushButton(self.tr("Save changes"))
        apply_button.setObjectName("apply-preferences")
        style(apply_button, variant=Variant.PRIMARY)
        root.addWidget(DialogFooter(dismiss=cancel, confirm=apply_button, parent=self))

        self._categories.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._categories.setCurrentRow(self.CATEGORY_APPEARANCE)
        cancel.clicked.connect(self.reject)
        apply_button.clicked.connect(self._apply)

    def show_category(self, category: int) -> None:
        """Switch categories the way clicking a row does. See ``CATEGORY_*``."""
        self._categories.setCurrentRow(category)

    def _appearance_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("preferences-appearance")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel(self.tr("Appearance"))
        title.setObjectName("preferences-page-title")
        style(title, role=Role.TITLE)
        description = QLabel(self.tr("Choose the fixed-width typeface used throughout Kimix."))
        description.setObjectName("preferences-description")
        style(description, tone=Tone.MUTED)
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        form = QFormLayout()
        form.setSpacing(10)
        self._primary_font = QComboBox()
        self._primary_font.setObjectName("font-primary")
        self._primary_font.addItem(self.tr("Application default"), userData="")
        for family in self._font_families:
            self._primary_font.addItem(family, userData=family)
        selected = self._preferences.font_families[0] if self._preferences.font_families else ""
        index = self._primary_font.findData(selected)
        self._primary_font.setCurrentIndex(max(index, 0))
        form.addRow(self.tr("Typeface"), self._primary_font)
        self._fallback_font = QComboBox()
        self._fallback_font.setObjectName("font-fallback")
        self._fallback_font.addItem(self.tr("System fixed-width font"), userData="")
        for family in self._font_families:
            self._fallback_font.addItem(family, userData=family)
        fallback = (
            self._preferences.font_families[1] if len(self._preferences.font_families) > 1 else ""
        )
        index = self._fallback_font.findData(fallback)
        self._fallback_font.setCurrentIndex(max(index, 0))
        form.addRow(self.tr("Fallback"), self._fallback_font)
        self._font_size = QSpinBox()
        self._font_size.setObjectName("font-size")
        self._font_size.setRange(9, 32)
        self._font_size.setSuffix(" px")
        self._font_size.setValue(self._preferences.font_size)
        form.addRow(self.tr("Size"), self._font_size)
        self._theme = QComboBox()
        self._theme.setObjectName("interface-theme")
        for name in SUPPORTED_THEMES:
            self._theme.addItem(self.tr(THEME_LABELS[name]), userData=name)
        index = self._theme.findData(self._preferences.theme)
        self._theme.setCurrentIndex(max(index, 0))
        form.addRow(self.tr("Theme"), self._theme)
        self._language = QComboBox()
        self._language.setObjectName("interface-language")
        self._language.addItem(SYSTEM_LANGUAGE_LABEL, userData=SYSTEM_LANGUAGE)
        for code in SUPPORTED_LANGUAGES:
            self._language.addItem(LANGUAGE_ENDONYMS[code], userData=code)
        index = self._language.findData(self._preferences.language)
        self._language.setCurrentIndex(max(index, 0))
        form.addRow(self.tr("Language"), self._language)
        layout.addLayout(form)

        preview = Card(CardLevel.INSET)
        preview.setObjectName("font-preview")
        preview_layout = preview.body
        preview_layout.setSpacing(4)
        preview_label = QLabel(self.tr("PREVIEW"))
        preview_label.setObjectName("font-preview-label")
        style(preview_label, role=Role.OVERLINE)
        self._preview = QLabel("The quick brown fox 0123456789\n中文字符 preview")
        self._preview.setObjectName("font-preview-text")
        preview_layout.addWidget(preview_label)
        preview_layout.addWidget(self._preview)
        layout.addWidget(preview)
        # One sentence, because there is no longer an exception to note. The language
        # used to need a restart and said so here; live widgets now restate their copy
        # on Qt's ``LanguageChange``, so typeface, theme and language all land at once.
        applies_note = QLabel(self.tr("Saved changes apply immediately."))
        applies_note.setObjectName("preferences-description")
        style(applies_note, tone=Tone.MUTED)
        applies_note.setWordWrap(True)
        layout.addWidget(applies_note)
        layout.addStretch()
        self._primary_font.currentIndexChanged.connect(self._update_preview)
        self._fallback_font.currentIndexChanged.connect(self._update_preview)
        self._font_size.valueChanged.connect(self._update_preview)
        self._update_preview()
        return page

    def _models_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("preferences-models")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        title = QLabel(self.tr("Models"))
        title.setObjectName("preferences-page-title")
        style(title, role=Role.TITLE)
        description = QLabel(
            self.tr(
                "Provider files and default model selection are managed per project or session."
            )
        )
        description.setObjectName("preferences-description")
        style(description, tone=Tone.MUTED)
        description.setWordWrap(True)
        self._codex_card = CodexAccountCard(
            self._codex_snapshot,
            self._codex_catalog,
        )
        self._codex_card.connect_requested.connect(self.connect_chatgpt.emit)
        self._codex_card.refresh_requested.connect(self.refresh_codex_models.emit)
        self._codex_card.disconnect_requested.connect(self.disconnect_chatgpt.emit)
        manage = QPushButton(self.tr("Manage LLM configurations"))
        manage.setObjectName("manage-llm-settings")
        manage.setToolTip(self.tr("Open provider configuration"))
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self._codex_card)
        layout.addWidget(manage, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        manage.clicked.connect(self.manage_llm.emit)
        return page

    def set_codex_snapshot(self, snapshot: CodexAuthSnapshot) -> None:
        self._codex_snapshot = snapshot
        if self._codex_card is not None:
            self._codex_card.set_snapshot(snapshot)

    def set_codex_catalog(self, catalog: CodexModelCatalog) -> None:
        self._codex_catalog = catalog
        if self._codex_card is not None:
            self._codex_card.set_catalog(catalog)

    def _update_preview(self) -> None:
        self._preview.setFont(
            interface_font(InterfacePreferences(self._selected_families(), self._font_size.value()))
        )

    def _apply(self) -> None:
        self.applied.emit(
            InterfacePreferences(
                self._selected_families(),
                self._font_size.value(),
                self._selected_language(),
                self._selected_theme(),
            )
        )
        self.accept()

    def _selected_families(self) -> tuple[str, ...]:
        families = (self._primary_font.currentData(), self._fallback_font.currentData())
        return tuple(family for family in families if isinstance(family, str) and family)

    def _selected_language(self) -> str:
        selected = self._language.currentData()
        return selected if isinstance(selected, str) and selected else SYSTEM_LANGUAGE

    def _selected_theme(self) -> str:
        selected = self._theme.currentData()
        return selected if isinstance(selected, str) and selected else SYSTEM_THEME
