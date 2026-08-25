"""Tests for the edit snapshot store."""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.session import Session
from kimi_cli.tools.file.snapshot_store import (
    SNAPSHOT_MAX_BYTES,
    STORE_MAX_ENTRIES,
    EditSnapshotStore,
    canonical_snapshot_key,
    get_edit_snapshot_store,
    parse_seen_lines_from_hashline_body,
)


def test_canonical_key_collapses(tmp_path):
    file = tmp_path / "a" / "b.txt"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("hi")
    key1 = canonical_snapshot_key(str(file))
    key2 = canonical_snapshot_key(str(file.resolve()))
    assert key1 == key2


def test_store_records_tag(session: Session):
    store = get_edit_snapshot_store(session)
    tag = store.record("/fake/path.py", "line1\nline2\n")
    assert tag is not None
    entry = store.lookup("/fake/path.py")
    assert entry is not None
    assert entry.content == "line1\nline2\n"
    assert entry.line1_hash is not None


def test_store_skips_oversized_content():
    store = EditSnapshotStore()
    huge = "x" * (SNAPSHOT_MAX_BYTES + 1)
    assert store.record("/big.txt", huge) is None


def test_store_prunes_distinct_paths():
    store = EditSnapshotStore(max_entries=3)
    store.record("/a/1.txt", "one")
    store.record("/a/2.txt", "two")
    store.record("/a/3.txt", "three")
    # Re-recording same path does not cause eviction.
    store.record("/a/3.txt", "three-v2")
    assert len(store) == 3
    store.record("/a/4.txt", "four")
    assert len(store) == 3


def test_seen_lines_parsing():
    body = "1#AB:first\n3#CD:third\n5-7#EF:range\n"
    seen = parse_seen_lines_from_hashline_body(body)
    assert 1 in seen
    assert 3 in seen
    assert 5 in seen
    assert 7 in seen
