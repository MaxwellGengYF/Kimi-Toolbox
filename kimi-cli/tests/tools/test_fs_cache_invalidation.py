"""Tests for fs-cache invalidation (plan 25 §4.3)."""

from __future__ import annotations

import pytest

from kimi_cli.session import Session
from kimi_cli.tools.file import glob as glob_mod
from kimi_cli.tools.file.fs_cache import (
    _INVALIDATORS,
    invalidate_fs_scan_after_delete,
    invalidate_fs_scan_after_rename,
    invalidate_fs_scan_after_write,
    register_invalidator,
    unregister_invalidator,
)
from kimi_cli.tools.file.snapshot_store import (
    get_file_snapshot_store,
    record_content_snapshot,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the global gitignore cache + invalidator registry per test."""
    saved_cache = dict(glob_mod._GITIGNORE_CACHE)
    saved_invalidators = list(_INVALIDATORS)
    glob_mod._GITIGNORE_CACHE.clear()
    _INVALIDATORS.clear()
    yield
    glob_mod._GITIGNORE_CACHE.clear()
    glob_mod._GITIGNORE_CACHE.update(saved_cache)
    _INVALIDATORS.clear()
    _INVALIDATORS.extend(saved_invalidators)


def _seed_gitignore_cache(root_str: str) -> None:
    glob_mod._GITIGNORE_CACHE[root_str] = glob_mod._GitignoreCacheEntry()


# ═══════════════════════════════════════════════════════════════════════════
# gitignore cache busting
# ═══════════════════════════════════════════════════════════════════════════


def test_write_gitignore_drops_affected_root(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    gi = tmp_path / ".gitignore"
    invalidate_fs_scan_after_write(str(gi))
    assert root not in glob_mod._GITIGNORE_CACHE


def test_write_regular_file_keeps_cache(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    invalidate_fs_scan_after_write(str(tmp_path / "code.py"))
    assert root in glob_mod._GITIGNORE_CACHE


def test_write_gitignore_case_insensitive(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    invalidate_fs_scan_after_write(str(tmp_path / ".GITIGNORE"))
    assert root not in glob_mod._GITIGNORE_CACHE


def test_write_gitignore_keeps_unrelated_root(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    _seed_gitignore_cache("/some/other/root")
    invalidate_fs_scan_after_write(str(tmp_path / ".gitignore"))
    assert root not in glob_mod._GITIGNORE_CACHE
    assert "/some/other/root" in glob_mod._GITIGNORE_CACHE


def test_delete_drops_ancestor_entries(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    # deleting a directory that may have held .gitignore files
    invalidate_fs_scan_after_delete(str(tmp_path / "subdir"))
    assert root not in glob_mod._GITIGNORE_CACHE


def test_rename_busts_both_endpoints(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    invalidate_fs_scan_after_rename(
        str(tmp_path / "a" / ".gitignore"), str(tmp_path / "b" / ".gitignore")
    )
    assert root not in glob_mod._GITIGNORE_CACHE


def test_invalidate_gitignore_cache_direct(tmp_path):
    root = str(tmp_path.resolve())
    _seed_gitignore_cache(root)
    glob_mod.invalidate_gitignore_cache(str(tmp_path / ".gitignore"))
    assert root not in glob_mod._GITIGNORE_CACHE
    # missing path is a safe no-op
    glob_mod.invalidate_gitignore_cache(str(tmp_path / "missing" / ".gitignore"))


# ═══════════════════════════════════════════════════════════════════════════
# registry seam
# ═══════════════════════════════════════════════════════════════════════════


def test_register_invalidator_receives_write_path(tmp_path):
    seen: list[str] = []
    register_invalidator(seen.append)
    f = tmp_path / "file.txt"
    invalidate_fs_scan_after_write(str(f))
    assert len(seen) == 1
    assert seen[0].endswith("file.txt")
    unregister_invalidator(seen.append)


def test_register_invalidator_delete_and_rename(tmp_path):
    seen: list[str] = []
    register_invalidator(seen.append)
    invalidate_fs_scan_after_delete(str(tmp_path / "gone.txt"))
    assert len(seen) == 1
    invalidate_fs_scan_after_rename(
        str(tmp_path / "old.txt"), str(tmp_path / "new.txt")
    )
    assert len(seen) == 3  # both endpoints
    assert any(p.endswith("old.txt") for p in seen)
    assert any(p.endswith("new.txt") for p in seen)
    unregister_invalidator(seen.append)


def test_register_same_invalidator_twice_is_noop():
    calls: list[str] = []
    register_invalidator(calls.append)
    register_invalidator(calls.append)
    invalidate_fs_scan_after_write("/x/y.txt")
    assert len(calls) == 1
    unregister_invalidator(calls.append)


def test_unregister_missing_is_noop():
    unregister_invalidator(lambda p: None)  # no exception


def test_failing_invalidator_is_isolated(tmp_path):
    seen: list[str] = []

    def boom(_path: str) -> None:
        raise RuntimeError("invalidator blew up")

    register_invalidator(boom)
    register_invalidator(seen.append)
    invalidate_fs_scan_after_write(str(tmp_path / "ok.txt"))
    assert len(seen) == 1  # second invalidator still ran
    unregister_invalidator(boom)
    unregister_invalidator(seen.append)


# ═══════════════════════════════════════════════════════════════════════════
# snapshot store interaction on delete/rename
# ═══════════════════════════════════════════════════════════════════════════


def test_delete_invalidates_snapshot(session: Session, tmp_path):
    f = tmp_path / "doomed.txt"
    record_content_snapshot(session, str(f), "about to die\n")
    store = get_file_snapshot_store(session)
    assert store.head(str(f)) is not None
    invalidate_fs_scan_after_delete(str(f), session=session)
    assert store.head(str(f)) is None


def test_delete_without_session_is_safe(tmp_path):
    invalidate_fs_scan_after_delete(str(tmp_path / "x.txt"), session=None)


def test_rename_relocates_snapshot_history(session: Session, tmp_path):
    old = tmp_path / "old.txt"
    new = tmp_path / "new.txt"
    tag = record_content_snapshot(session, str(old), "moving\n")
    store = get_file_snapshot_store(session)
    invalidate_fs_scan_after_rename(str(old), str(new), session=session)
    assert store.head(str(old)) is None
    head = store.head(str(new))
    assert head is not None
    assert head.hash == tag
    assert head.text == "moving\n"


def test_rename_same_canonical_keeps_history(session: Session, tmp_path):
    f = tmp_path / "same.txt"
    tag = record_content_snapshot(session, str(f), "staying\n")
    store = get_file_snapshot_store(session)
    invalidate_fs_scan_after_rename(str(f), str(f), session=session)
    head = store.head(str(f))
    assert head is not None
    assert head.hash == tag


def test_write_does_not_touch_snapshot_store(session: Session, tmp_path):
    """write owns its content — fs_cache write must not invalidate snapshots."""
    f = tmp_path / "keep.txt"
    record_content_snapshot(session, str(f), "keep me\n")
    invalidate_fs_scan_after_write(str(f))
    assert get_file_snapshot_store(session).head(str(f)) is not None


# ═══════════════════════════════════════════════════════════════════════════
# integration: glob sees fresh gitignore state after an invalidated write
# ═══════════════════════════════════════════════════════════════════════════


def test_glob_rule_cache_refreshed_after_gitignore_write(tmp_path):
    """A .gitignore write + invalidation yields fresh rules on next glob."""
    from kimi_cli.tools.file.glob import _get_gitignore_rules

    root = tmp_path.resolve()
    gi = root / ".gitignore"
    gi.write_bytes(b"*.log\n")
    rules = _get_gitignore_rules(root)
    assert any(r.pattern == "*.log" for r in rules)

    # rewrite the gitignore and invalidate
    gi.write_bytes(b"*.tmp\n")
    invalidate_fs_scan_after_write(str(gi))
    rules = _get_gitignore_rules(root)
    assert any(r.pattern == "*.tmp" for r in rules)
    assert not any(r.pattern == "*.log" for r in rules)
