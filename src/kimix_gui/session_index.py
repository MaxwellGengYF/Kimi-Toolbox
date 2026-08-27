"""List historical Kimix sessions from the kimi-cli work-dir store."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import time
from typing import Any

from kaos.path import KaosPath

from kimix_gui.kimi_workdir import resolve_kimi_work_dir

_LAST_ID_UNSET: object = object()

#: Stand-in title for a session that never got one. Public because the Qt layer has
#: to recognize it to translate it (see ``qt/labels.py``); the pure layer keeps
#: returning the English word, which is the msgid.
UNTITLED_TITLE = "Untitled"

#: Buckets :func:`relative_time` can report. Machine-readable on purpose: the wording
#: lives in the Qt layer, which is the only place that can call ``tr()``.
RELATIVE_UNKNOWN = "unknown"
RELATIVE_JUST_NOW = "just_now"
RELATIVE_MINUTES = "minutes"
RELATIVE_HOURS = "hours"
RELATIVE_YESTERDAY = "yesterday"
RELATIVE_DATE_THIS_YEAR = "date_this_year"
RELATIVE_DATE_EARLIER = "date_earlier"

#: Units :func:`file_size` can report.
SIZE_UNIT_KB = "KB"
SIZE_UNIT_MB = "MB"

SessionLister = Callable[[KaosPath], Awaitable[Sequence[Any]]]
SessionLoader = Callable[[Path], Awaitable[list["SessionSummary"]]]
SessionDeleter = Callable[[Path, Sequence[str]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Display fields for one resumable session on the GUI home screen."""

    id: str
    title: str
    updated_at: float
    is_last: bool = False
    size_bytes: int = 0
    file_count: int = 0
    storage_format: str = "Unknown"
    is_archived: bool = False
    todo_count: int = 0
    additional_dir_count: int = 0


@dataclass(frozen=True, slots=True)
class RelativeTime:
    """How long ago something happened, as data rather than as a sentence.

    ``unit`` is one of the ``RELATIVE_*`` constants. ``count`` carries the number of
    minutes or hours for those two buckets and is ``0`` otherwise. ``moment`` is the
    local-time instant, set only for the two date buckets so the Qt layer can format
    it with the user's locale instead of a hardcoded ``strftime`` pattern.
    """

    unit: str
    count: int = 0
    moment: datetime | None = None


def relative_time(updated_at: float, *, now: float | None = None) -> RelativeTime:
    """Bucket a timestamp for home-screen session details.

    Returns structured data, never a sentence: "2 hours ago" has a different word
    order in other languages, and only the Qt layer can call ``tr()``.
    """

    if updated_at <= 0:
        return RelativeTime(RELATIVE_UNKNOWN)
    current = now if now is not None else time()
    delta = max(0, int(current - updated_at))
    if delta < 45:
        return RelativeTime(RELATIVE_JUST_NOW)
    if delta < 90:
        return RelativeTime(RELATIVE_MINUTES, 1)
    if delta < 3600:
        return RelativeTime(RELATIVE_MINUTES, delta // 60)
    if delta < 5400:
        return RelativeTime(RELATIVE_HOURS, 1)
    if delta < 86400:
        return RelativeTime(RELATIVE_HOURS, delta // 3600)
    if delta < 172800:
        return RelativeTime(RELATIVE_YESTERDAY)
    local_tz = datetime.now(UTC).astimezone().tzinfo
    moment = datetime.fromtimestamp(updated_at, tz=local_tz)
    current_dt = datetime.fromtimestamp(current, tz=local_tz)
    unit = RELATIVE_DATE_THIS_YEAR if current_dt.year == moment.year else RELATIVE_DATE_EARLIER
    return RelativeTime(unit, moment=moment)


@dataclass(frozen=True, slots=True)
class FileSize:
    """A byte count reduced to one binary unit.

    ``value`` is the already-rounded number as text (``"0"``, ``"0.5"``, ``"1.5"``):
    rounding is arithmetic, so it stays here. ``unit`` is ``SIZE_UNIT_KB`` or
    ``SIZE_UNIT_MB``, and joining the two into a phrase is the Qt layer's job.
    """

    value: str
    unit: str


def file_size(size_bytes: int) -> FileSize:
    """Reduce a byte count to a rounded value plus a binary unit."""

    size = max(0, size_bytes)
    if size == 0:
        return FileSize("0", SIZE_UNIT_KB)
    if size < 1024 * 1024:
        return FileSize(_trimmed(max(0.1, size / 1024)), SIZE_UNIT_KB)
    return FileSize(_trimmed(size / (1024 * 1024)), SIZE_UNIT_MB)


def _trimmed(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _session_storage_stats(session: Any) -> tuple[int, int, str]:
    try:
        session_dir = Path(session.dir)
    except AttributeError, TypeError, OSError:
        session_dir = None

    size_bytes = 0
    file_count = 0
    if session_dir is not None:
        for root, _dirs, files in session_dir.walk(on_error=lambda _error: None):
            for name in files:
                try:
                    size_bytes += (root / name).stat().st_size
                    file_count += 1
                except OSError:
                    continue

    try:
        suffix = Path(session.context_file).suffix.lower()
    except AttributeError, TypeError:
        suffix = ""
    storage_format = {".db": "SQLite", ".jsonl": "JSONL"}.get(suffix, "Unknown")
    return size_bytes, file_count, storage_format


def _collection_size(value: object) -> int:
    return len(value) if isinstance(value, Sized) else 0


def summaries_from_sessions(
    sessions: Sequence[Any],
    *,
    last_session_id: str | None = None,
) -> list[SessionSummary]:
    """Map kimi-cli session objects (or test doubles) into home-screen rows."""

    summaries: list[SessionSummary] = []
    for session in sessions:
        session_id = str(getattr(session, "id", "") or "")
        if not session_id:
            continue
        title = str(getattr(session, "title", "") or UNTITLED_TITLE)
        updated_at = float(getattr(session, "updated_at", 0.0) or 0.0)
        size_bytes, file_count, storage_format = _session_storage_stats(session)
        state = getattr(session, "state", None)
        summaries.append(
            SessionSummary(
                id=session_id,
                title=title,
                updated_at=updated_at,
                is_last=session_id == last_session_id,
                size_bytes=size_bytes,
                file_count=file_count,
                storage_format=storage_format,
                is_archived=bool(getattr(state, "archived", False)),
                todo_count=_collection_size(getattr(state, "todos", ())),
                additional_dir_count=_collection_size(getattr(state, "additional_dirs", ())),
            )
        )
    return sorted(summaries, key=lambda summary: summary.updated_at, reverse=True)


async def _list_cli_sessions(work_dir: KaosPath) -> Sequence[Any]:
    from kimi_cli.session import Session as CliSession

    return await CliSession.list(work_dir)


def _last_session_id(work_dir: KaosPath) -> str | None:
    from kimi_cli.metadata import load_metadata

    meta = load_metadata().get_work_dir_meta(work_dir)
    if meta is None:
        return None
    return meta.last_session_id


async def list_session_summaries(
    work_dir: Path,
    *,
    list_sessions: SessionLister | None = None,
    last_session_id: str | None | object = _LAST_ID_UNSET,
) -> list[SessionSummary]:
    """List non-empty sessions for ``work_dir`` using kimi-cli storage rules."""

    kaos_dir = resolve_kimi_work_dir(work_dir)
    sessions = await (list_sessions or _list_cli_sessions)(kaos_dir)
    resolved_last = (
        _last_session_id(kaos_dir) if last_session_id is _LAST_ID_UNSET else last_session_id
    )
    last_id = resolved_last if isinstance(resolved_last, str) else None
    return summaries_from_sessions(sessions, last_session_id=last_id)


async def delete_sessions(work_dir: Path, session_ids: Sequence[str]) -> None:
    """Permanently delete Kimi sessions and clear a deleted last-session pointer."""

    from kimi_cli.metadata import load_metadata, save_metadata
    from kimi_cli.session import Session as CliSession

    ids = list(dict.fromkeys(session_id for session_id in session_ids if session_id))
    if not ids:
        return
    kaos_dir = resolve_kimi_work_dir(work_dir)
    for session_id in ids:
        session = await CliSession.find(kaos_dir, session_id)
        if session is not None:
            await session.delete()

    metadata = load_metadata()
    work_dir_meta = metadata.get_work_dir_meta(kaos_dir)
    if work_dir_meta is not None and work_dir_meta.last_session_id in ids:
        work_dir_meta.last_session_id = None
        save_metadata(metadata)
