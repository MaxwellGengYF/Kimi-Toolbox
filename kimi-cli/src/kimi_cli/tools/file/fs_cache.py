"""FS-cache invalidation after write/delete/rename (plan 25).

Port of oh-my-pi ``tools/fs-cache-invalidation.ts``.  This repo has no native
directory-scan cache (glob re-walks the tree every call), so each call routes
to the caches that actually exist:

* the glob gitignore rule cache (``glob._GITIGNORE_CACHE``) — busted when a
  ``.gitignore`` could have changed;
* registered scan-cache invalidators (:data:`_INVALIDATORS`) — the registry
  seam so future scan caches (native glob dir scan, grep file recorders) can
  hook in without touching the tools;
* the session file snapshot store (``invalidate``/``relocate``) — handled
  here for delete/rename because those paths have no content argument
  (write/edit record fresh content themselves).

The auto-generated marker cache needs no explicit bust (self-consistent via
its ``(mtime_ns, size)`` key); ``VFS._resolve_rel`` and the grep
pattern/classification lru_caches are pure input→output caches (documented
no-ops in the plan).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from kimi_cli.utils.logging import logger

_INVALIDATORS: list[Callable[[str], None]] = []


def register_invalidator(fn: Callable[[str], None]) -> None:
    """Register a scan-cache invalidator.

    *fn* receives the canonical path of every file affected by a
    write/delete/rename.  Future scan caches call this at import time; the
    write/edit tools never change again.  Registering the same callable
    twice is a no-op.
    """
    if fn not in _INVALIDATORS:
        _INVALIDATORS.append(fn)


def unregister_invalidator(fn: Callable[[str], None]) -> None:
    """Remove a previously registered invalidator (test hook)."""
    try:
        _INVALIDATORS.remove(fn)
    except ValueError:
        pass


def _canonical(path: str) -> str:
    from kimi_cli.tools.file.snapshot_store import canonical_snapshot_key

    return canonical_snapshot_key(path)


def _bust_gitignore_cache(path: str) -> None:
    try:
        from kimi_cli.tools.file.glob import invalidate_gitignore_cache

        invalidate_gitignore_cache(path)
    except Exception:
        logger.debug("gitignore cache bust failed for {path}", path=path)


def _run_registered_invalidators(path: str) -> None:
    for fn in list(_INVALIDATORS):
        try:
            fn(path)
        except Exception:
            logger.debug("fs-cache invalidator failed: {fn}", fn=fn)


def _session_snapshot_store(session: Any | None) -> Any | None:
    if session is None:
        return None
    return getattr(session, "file_snapshot_store", None)


def invalidate_fs_scan_after_write(path: str) -> None:
    """Bust scan caches after a successful write to *path*.

    Drops the glob gitignore cache entries affected by the write when a
    ``.gitignore`` was written, then runs the registered invalidators on the
    canonical path.  (Snapshot + file_mtime handling are done by the write
    tools directly because they own the content.)
    """
    try:
        if Path(path).name.lower() == ".gitignore":
            _bust_gitignore_cache(path)
    except (TypeError, ValueError):
        pass
    _run_registered_invalidators(_canonical(path))


def invalidate_fs_scan_after_delete(path: str, session: Any | None = None) -> None:
    """Bust scan caches after deleting *path*.

    A deleted directory may have held ``.gitignore`` files, so every
    affected glob cache entry is dropped unconditionally; the snapshot-store
    entry for the canonical path is invalidated.
    """
    _bust_gitignore_cache(path)
    canon = _canonical(path)
    _run_registered_invalidators(canon)
    store = _session_snapshot_store(session)
    if store is not None:
        try:
            store.invalidate(path)
        except Exception:
            logger.debug("snapshot invalidate failed for {path}", path=path)


def invalidate_fs_scan_after_rename(
    old_path: str, new_path: str, session: Any | None = None
) -> None:
    """Bust scan caches after renaming *old_path* → *new_path*.

    A ``.gitignore`` could move, so both endpoints bust the gitignore cache;
    registered invalidators run for both canonical paths; the snapshot
    history is relocated when the two paths canonicalize differently.
    """
    _bust_gitignore_cache(old_path)
    _bust_gitignore_cache(new_path)
    old_canon = _canonical(old_path)
    new_canon = _canonical(new_path)
    _run_registered_invalidators(old_canon)
    if new_canon != old_canon:
        _run_registered_invalidators(new_canon)
    store = _session_snapshot_store(session)
    if store is not None and old_canon != new_canon:
        try:
            store.relocate(old_path, new_path)
        except Exception:
            logger.debug("snapshot relocate failed for {path}", path=old_path)
