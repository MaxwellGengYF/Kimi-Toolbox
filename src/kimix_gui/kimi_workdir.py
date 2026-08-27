"""Resolve Windows work-directory casing against Kimi's metadata store."""

from __future__ import annotations

import os
from pathlib import Path

from kaos import get_current_kaos
from kaos.path import KaosPath
from kimi_cli.metadata import WorkDirMeta, load_metadata, save_metadata


class WorkDirCaseConflict(RuntimeError):
    """Raised when multiple case variants contain session data."""


def resolve_kimi_work_dir(work_dir: Path) -> KaosPath:
    """Return a Kimi-compatible path, repairing a safe drive-case mismatch."""

    local_path = work_dir.expanduser().resolve()
    canonical = KaosPath.unsafe_from_local_path(local_path).canonical()
    if os.name != "nt":
        return canonical

    metadata = load_metadata()
    if metadata.get_work_dir_meta(canonical) is not None:
        return canonical

    canonical_text = str(canonical)
    kaos_name = get_current_kaos().name
    matches = [
        item
        for item in metadata.work_dirs
        if item.kaos == kaos_name and _same_windows_path(item.path, canonical_text)
    ]
    if not matches:
        return canonical
    if len(matches) > 1:
        raise WorkDirCaseConflict(
            f"Multiple Kimi metadata entries match {canonical_text} with different casing"
        )

    stored = matches[0]
    canonical_meta = WorkDirMeta(path=canonical_text, kaos=stored.kaos)
    stored_has_sessions = _directory_has_entries(stored.sessions_dir)
    canonical_has_sessions = _directory_has_entries(canonical_meta.sessions_dir)
    if stored_has_sessions and canonical_has_sessions:
        raise WorkDirCaseConflict(
            "Kimi session data exists under both path-case variants: "
            f"{stored.path} and {canonical_text}"
        )
    if canonical_has_sessions:
        stored.path = canonical_text
        save_metadata(metadata)
        return canonical

    return KaosPath.unsafe_from_local_path(Path(stored.path)).canonical()


def _same_windows_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))


def _directory_has_entries(path: Path) -> bool:
    try:
        next(path.iterdir())
    except OSError, StopIteration:
        return False
    return True
