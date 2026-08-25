"""Session file recorder for grep (plans/23-grep-rich.md §4.4).

Port of oh-my-pi's ``createFileRecorder`` (deduplicating, insertion-ordered
list of relative paths) plus a kimi-specific session persistence layer: the
deduplicated matched-file list is stored on ``session.custom_data`` under
``grep.files`` so a follow-up ``read``/``edit`` pass can operate on exactly
the files the last grep surfaced.
"""

from __future__ import annotations

import pendulum

__all__ = [
    "FileRecorder",
    "RECORDER_SESSION_KEY",
    "RECORDER_FILES_KEY",
    "RECORDER_CAP",
    "record_grep_files",
    "get_recorded_grep_files",
]

RECORDER_SESSION_KEY = "grep"
RECORDER_FILES_KEY = "files"
RECORDER_CAP = 500  # bounded; oldest entries are dropped on overflow


class FileRecorder:
    """Deduplicating, insertion-ordered list of relative paths."""

    __slots__ = ("_seen", "_list")

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._list: list[str] = []

    def record(self, relative_path: str) -> None:
        """Record a path, skipping duplicates and empty entries."""
        if not relative_path or relative_path in self._seen:
            return
        self._seen.add(relative_path)
        self._list.append(relative_path)

    @property
    def list(self) -> list[str]:
        return list(self._list)

    def __len__(self) -> int:
        return len(self._list)


def _merged(existing: list[str], new: list[str]) -> list[str]:
    """Merge two lists preserving insertion order (existing first)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in [*existing, *new]:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def record_grep_files(session, files: list[str], *, cwd: str | None = None) -> None:
    """Persist a deduplicated matched-file list on the session.

    Merges into ``session.custom_data["grep"]["files"]`` preserving insertion
    order, capped at ``RECORDER_CAP`` (drop from the front). Also stores
    ``cwd`` (the workspace the paths are relative to) and ``updated_at``
    (ISO timestamp via pendulum, per the performance rule).
    """
    if not files:
        return
    data = session.custom_data.setdefault(RECORDER_SESSION_KEY, {})
    if not isinstance(data, dict):
        data = {}
        session.custom_data[RECORDER_SESSION_KEY] = data
    existing = data.get(RECORDER_FILES_KEY) or []
    if not isinstance(existing, list):
        existing = []
    merged = _merged(existing, files)
    if len(merged) > RECORDER_CAP:
        merged = merged[-RECORDER_CAP:]
    data[RECORDER_FILES_KEY] = merged
    if cwd is not None:
        data["cwd"] = cwd
    data["updated_at"] = pendulum.now().isoformat()


def get_recorded_grep_files(session) -> list[str]:
    """Return the recorded matched-file list (empty when nothing recorded)."""
    data = session.custom_data.get(RECORDER_SESSION_KEY)
    if not isinstance(data, dict):
        return []
    files = data.get(RECORDER_FILES_KEY)
    if not isinstance(files, list):
        return []
    return [f for f in files if isinstance(f, str)]
