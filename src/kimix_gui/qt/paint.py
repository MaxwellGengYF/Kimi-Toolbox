"""Qt wording and colors for semantic transcript entries; no geometry or drawing."""

from __future__ import annotations

from PySide6.QtGui import QColor

from kimix_gui.qt.labels import translate_no_details, translate_transcript_text
from kimix_gui.qt.theme import active_theme
from kimix_gui.transcript_data import TranscriptEntry
from kimix_gui.transcript_layout import RecordLayout, layout_entry


def qcolor(name: str) -> QColor:
    """Resolve a semantic category against the active theme on every call."""

    return QColor(active_theme().categories.resolve(name))


def layout_record(
    entry: TranscriptEntry,
    *,
    width: int,
    expanded: bool = False,
) -> RecordLayout:
    """Translate and compose a row directly from its immutable AST entry."""

    return layout_entry(
        entry,
        width=width,
        expanded=expanded,
        translate=translate_transcript_text,
        placeholder=translate_no_details(),
    )


__all__ = ["RecordLayout", "layout_record", "qcolor"]
