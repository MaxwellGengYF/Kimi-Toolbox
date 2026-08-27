"""Word the session list's timestamps and file sizes from the pure layer's figures.

Same split as ``qt/status_line.py``: ``session_index`` computes (``RelativeTime``,
``FileSize``) and this module phrases. It has to be this way round -- the pure layer must
not import PySide6, so it cannot reach a translation catalog, and word order is not the
pure layer's business.

Singular and plural are two whole strings, never ``%n``: a missing ``.qm`` makes ``tr``
fall back to the msgid, and ``%n minute(s) ago`` would then show a literal ``(s)``. Units
are whole phrases for the same reason a language may want the unit before the number.

The translation context stays ``"HomeView"`` even though the code moved out of that
module. A context is the bucket a translator reads, and every string here is copy the
home view shows; renaming it would orphan finished translations to describe a file layout
the reader of a ``.ts`` cannot see. ``QCoreApplication.translate`` with a literal context
rather than ``self.tr`` because these are module functions with no ``QObject`` to ask,
and ``pyside6-lupdate`` only extracts literal arguments.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QCoreApplication, QDate, QLocale

from kimix_gui.qt.i18n import active_language
from kimix_gui.session_index import (
    RELATIVE_DATE_THIS_YEAR,
    RELATIVE_HOURS,
    RELATIVE_JUST_NOW,
    RELATIVE_MINUTES,
    RELATIVE_YESTERDAY,
    SIZE_UNIT_MB,
    FileSize,
    RelativeTime,
    file_size,
    relative_time,
)


def format_timestamp(updated_at: float) -> str:
    if updated_at <= 0:
        return QCoreApplication.translate("HomeView", "Unknown time")
    try:
        return datetime.fromtimestamp(updated_at).astimezone().strftime("%Y-%m-%d %H:%M")
    except OSError, OverflowError, ValueError:
        return QCoreApplication.translate("HomeView", "Unknown time")


def format_relative_time(updated_at: float, *, now: float | None = None) -> str:
    """Turn ``session_index.relative_time`` data into one localized phrase."""
    return _relative_phrase(relative_time(updated_at, now=now))


def format_file_size(size_bytes: int) -> str:
    """Turn ``session_index.file_size`` data into one localized phrase."""
    return _file_size_phrase(file_size(size_bytes))


def _relative_phrase(relative: RelativeTime) -> str:
    if relative.unit == RELATIVE_JUST_NOW:
        return QCoreApplication.translate("HomeView", "just now")
    if relative.unit == RELATIVE_MINUTES:
        template = (
            QCoreApplication.translate("HomeView", "1 minute ago")
            if relative.count == 1
            else QCoreApplication.translate("HomeView", "{count} minutes ago")
        )
        return template.format(count=relative.count)
    if relative.unit == RELATIVE_HOURS:
        template = (
            QCoreApplication.translate("HomeView", "1 hour ago")
            if relative.count == 1
            else QCoreApplication.translate("HomeView", "{count} hours ago")
        )
        return template.format(count=relative.count)
    if relative.unit == RELATIVE_YESTERDAY:
        return QCoreApplication.translate("HomeView", "yesterday")
    if relative.moment is not None:
        # A translatable Qt date *pattern* rather than ``strftime``: ``%b`` emits an
        # English month abbreviation whatever the UI language, and field order is
        # part of the locale. ``MMM d`` stays as-is in English and becomes ``M月d日``
        # in Chinese.
        pattern = (
            QCoreApplication.translate("HomeView", "MMM d")
            if relative.unit == RELATIVE_DATE_THIS_YEAR
            else QCoreApplication.translate("HomeView", "yyyy-MM-dd")
        )
        # The locale comes from the installed catalog, not from ``QLocale.system()``:
        # the UI language is a stored preference and only incidentally related to the
        # operating system's locale.
        return QLocale(active_language()).toString(QDate(relative.moment.date()), pattern)
    return QCoreApplication.translate("HomeView", "unknown")


def _file_size_phrase(size: FileSize) -> str:
    template = (
        QCoreApplication.translate("HomeView", "{value} MB")
        if size.unit == SIZE_UNIT_MB
        else QCoreApplication.translate("HomeView", "{value} KB")
    )
    return template.format(value=size.value)
