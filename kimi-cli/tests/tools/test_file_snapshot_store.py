"""Tests for the plan-25 versioned file snapshot store.

Covers ``InMemorySnapshotStore`` (bounds, dedup, LRU, ring), the helper API
(``record_content_snapshot`` / ``record_file_snapshot`` / seen-lines), text
normalization/hashing, and lazy ``Session.file_snapshot_store`` attachment.
"""

from __future__ import annotations

import os

import pytest

from kimi_cli.session import Session
from kimi_cli.tools.file.snapshot_store import (
    DEFAULT_MAX_PATHS,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_MAX_VERSIONS_PER_PATH,
    SNAPSHOT_MAX_BYTES,
    InMemorySnapshotStore,
    Snapshot,
    canonical_snapshot_key,
    compute_file_hash,
    detect_line_ending,
    get_file_snapshot_store,
    normalize_to_lf,
    parse_seen_lines_from_body,
    record_content_snapshot,
    record_file_snapshot,
    record_seen_lines,
    record_seen_lines_from_body,
)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization / hashing
# ═══════════════════════════════════════════════════════════════════════════


def test_normalize_to_lf_crlf():
    assert normalize_to_lf("a\r\nb\r\n") == "a\nb\n"


def test_normalize_to_lf_lone_cr():
    assert normalize_to_lf("a\rb\rc") == "a\nb\nc"


def test_normalize_to_lf_bom():
    assert normalize_to_lf("\ufeffhello") == "hello"


def test_normalize_to_lf_mixed():
    assert normalize_to_lf("\ufeffa\r\nb\rc") == "a\nb\nc"


def test_compute_file_hash_format_and_stability():
    tag = compute_file_hash("hello\nworld\n")
    assert len(tag) == 4
    assert tag == tag.upper()
    assert all(c in "0123456789ABCDEF" for c in tag)
    # stable across calls
    assert compute_file_hash("hello\nworld\n") == tag


def test_compute_file_hash_lf_normalized():
    # CRLF and LF forms hash identically (normalization before hashing).
    assert compute_file_hash("a\r\nb\r\n") == compute_file_hash("a\nb\n")
    assert compute_file_hash("\ufeffa\nb\n") == compute_file_hash("a\nb\n")


def test_detect_line_ending():
    assert detect_line_ending("a\r\nb\r\n") == "\r\n"
    assert detect_line_ending("a\nb\n") == "\n"
    assert detect_line_ending("no newlines") == "\n"


def test_canonical_snapshot_key_missing_file(tmp_path):
    missing = tmp_path / "no" / "such" / "file.txt"
    key = canonical_snapshot_key(str(missing))
    # realpath fallback produces an absolute path
    assert os.path.isabs(key)
    assert key.endswith("file.txt")


# ═══════════════════════════════════════════════════════════════════════════
# record / head / by_hash / by_content
# ═══════════════════════════════════════════════════════════════════════════


def test_record_returns_tag_and_head():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/a.py", "one\ntwo\n")
    assert tag == compute_file_hash("one\ntwo\n")
    head = store.head("/fake/a.py")
    assert head is not None
    assert head.text == "one\ntwo\n"
    assert head.hash == tag


def test_head_missing_path():
    store = InMemorySnapshotStore()
    assert store.head("/nope.py") is None


def test_record_oversized_returns_none():
    store = InMemorySnapshotStore()
    huge = "x" * (SNAPSHOT_MAX_BYTES + 1)
    assert store.record("/fake/big.txt", huge) is None
    assert store.head("/fake/big.txt") is None


def test_same_content_re_record_same_tag_refreshes_recency():
    store = InMemorySnapshotStore()
    t1 = store.record("/fake/a.py", "same\n")
    t2 = store.record("/fake/a.py", "same\n")
    assert t1 == t2
    # only one version in the ring
    versions = store._paths[canonical_snapshot_key("/fake/a.py")]
    assert len(versions) == 1
    # recency refreshed: head is the same object, recorded_at moved
    assert store.head("/fake/a.py") is versions[-1]


def test_by_hash_and_by_content():
    store = InMemorySnapshotStore()
    tag1 = store.record("/fake/a.py", "v1\n")
    tag2 = store.record("/fake/a.py", "v2\n")
    assert store.by_hash("/fake/a.py", tag1) is not None
    assert store.by_hash("/fake/a.py", tag2) is not None
    assert store.by_hash("/fake/a.py", "ZZZZ") is None
    assert store.by_content("/fake/a.py", "v1\n") is not None
    assert store.by_content("/fake/a.py", "v1\r\n") is not None  # normalized
    assert store.by_content("/fake/a.py", "missing\n") is None
    # missing path lookups are None
    assert store.by_hash("/nope", tag1) is None
    assert store.by_content("/nope", "v1\n") is None


def test_find_by_hash_across_paths():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/a.py", "shared\n")
    store.record("/fake/b.py", "other\n")
    found = store.find_by_hash(tag)
    assert len(found) == 1
    assert found[0].text == "shared\n"
    assert store.find_by_hash("ZZZZ") == []


def test_dedup_by_full_text_not_just_tag():
    """Two texts with a colliding 16-bit tag must stay two versions."""
    store = InMemorySnapshotStore()
    # Force a tag collision by monkeypatching compute_file_hash used in record.
    import kimi_cli.tools.file.snapshot_store as ss

    real = ss.compute_file_hash
    try:
        ss.compute_file_hash = lambda text: "BEEF"
        store.record("/fake/a.py", "text-one\n")
        store.record("/fake/a.py", "text-two\n")
    finally:
        ss.compute_file_hash = real
    versions = store._paths[canonical_snapshot_key("/fake/a.py")]
    assert len(versions) == 2
    texts = {v.text for v in versions}
    assert texts == {"text-one\n", "text-two\n"}
    assert {v.hash for v in versions} == {"BEEF"}


def test_record_normalizes_content():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/a.py", "\ufeffa\r\nb\r")
    head = store.head("/fake/a.py")
    assert head.text == "a\nb\n"
    assert tag == compute_file_hash("a\nb\n")


# ═══════════════════════════════════════════════════════════════════════════
# Bounds: ring / max_paths / max_total_bytes
# ═══════════════════════════════════════════════════════════════════════════


def test_ring_drops_oldest_beyond_max_versions():
    store = InMemorySnapshotStore(max_versions_per_path=4)
    for i in range(6):
        store.record("/fake/a.py", f"v{i}\n")
    versions = store._paths[canonical_snapshot_key("/fake/a.py")]
    assert len(versions) == 4
    assert [v.text for v in versions] == ["v2\n", "v3\n", "v4\n", "v5\n"]


def test_max_paths_lru_eviction():
    store = InMemorySnapshotStore(max_paths=3, max_versions_per_path=4)
    store.record("/fake/1.txt", "one\n")
    store.record("/fake/2.txt", "two\n")
    store.record("/fake/3.txt", "three\n")
    # touch /fake/1.txt (re-record same content refreshes recency)
    store.record("/fake/1.txt", "one\n")
    store.record("/fake/4.txt", "four\n")
    assert len(store) == 3
    # /fake/2.txt was the least recently used → evicted
    assert store.head("/fake/2.txt") is None
    assert store.head("/fake/1.txt") is not None
    assert store.head("/fake/3.txt") is not None
    assert store.head("/fake/4.txt") is not None


def test_max_total_bytes_evicts_lru_histories():
    store = InMemorySnapshotStore(max_paths=10, max_total_bytes=100)
    store.record("/fake/a.txt", "a" * 60)
    store.record("/fake/b.txt", "b" * 60)
    # total would be 120 > 100 → oldest path history evicted
    assert store.head("/fake/a.txt") is None
    assert store.head("/fake/b.txt") is not None
    assert len(store) == 1


def test_default_bounds_constants():
    assert DEFAULT_MAX_PATHS == 256
    assert DEFAULT_MAX_VERSIONS_PER_PATH == 4
    assert DEFAULT_MAX_TOTAL_BYTES == 64 * 1024 * 1024
    assert SNAPSHOT_MAX_BYTES == 4 * 1024 * 1024


def test_total_bytes_tracking_under_ring_trim():
    store = InMemorySnapshotStore(max_versions_per_path=2, max_total_bytes=10_000)
    store.record("/fake/a.txt", "x" * 50)
    store.record("/fake/a.txt", "y" * 50)
    store.record("/fake/a.txt", "z" * 50)  # drops first, ring capped at 2
    assert store._total_bytes == 100


# ═══════════════════════════════════════════════════════════════════════════
# seen_lines
# ═══════════════════════════════════════════════════════════════════════════


def test_seen_lines_union_across_records():
    store = InMemorySnapshotStore()
    store.record("/fake/a.py", "same\n", seen_lines=[1, 2])
    store.record("/fake/a.py", "same\n", seen_lines=[3])
    head = store.head("/fake/a.py")
    assert head.seen_lines == {1, 2, 3}


def test_record_seen_lines_by_tag():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/a.py", "text\n")
    store.record_seen_lines("/fake/a.py", tag, [5, 6])
    head = store.head("/fake/a.py")
    assert head.seen_lines == {5, 6}
    # missing tag → no-op
    store.record_seen_lines("/fake/a.py", "ZZZZ", [9])
    assert head.seen_lines == {5, 6}
    # missing path → no-op
    store.record_seen_lines("/nope.py", tag, [9])


def test_seen_lines_none_default():
    store = InMemorySnapshotStore()
    store.record("/fake/a.py", "text\n")
    assert store.head("/fake/a.py").seen_lines is None


# ═══════════════════════════════════════════════════════════════════════════
# invalidate / relocate / clear
# ═══════════════════════════════════════════════════════════════════════════


def test_invalidate():
    store = InMemorySnapshotStore()
    store.record("/fake/a.py", "one\n")
    store.invalidate("/fake/a.py")
    assert store.head("/fake/a.py") is None
    assert len(store) == 0
    assert store._total_bytes == 0
    # invalidating a missing path is a no-op
    store.invalidate("/fake/a.py")


def test_relocate_moves_history():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/old.py", "content\n")
    store.relocate("/fake/old.py", "/fake/new.py")
    assert store.head("/fake/old.py") is None
    head = store.head("/fake/new.py")
    assert head is not None
    assert head.text == "content\n"
    assert head.hash == tag


def test_relocate_same_canonical_is_noop():
    store = InMemorySnapshotStore()
    tag = store.record("/fake/a.py", "content\n")
    store.relocate("/fake/a.py", "/fake/a.py")
    assert store.head("/fake/a.py") is not None
    assert store.head("/fake/a.py").hash == tag


def test_relocate_overwrites_destination():
    store = InMemorySnapshotStore()
    store.record("/fake/old.py", "old content\n")
    store.record("/fake/new.py", "new content\n")
    store.relocate("/fake/old.py", "/fake/new.py")
    head = store.head("/fake/new.py")
    assert head.text == "old content\n"


def test_clear():
    store = InMemorySnapshotStore()
    store.record("/fake/a.py", "one\n")
    store.record("/fake/b.py", "two\n")
    store.clear()
    assert len(store) == 0
    assert store._total_bytes == 0
    assert store.head("/fake/a.py") is None


# ═══════════════════════════════════════════════════════════════════════════
# Session wiring + helper API
# ═══════════════════════════════════════════════════════════════════════════


def test_get_file_snapshot_store_lazy_attaches(session: Session):
    assert session.file_snapshot_store is None
    store = get_file_snapshot_store(session)
    assert isinstance(store, InMemorySnapshotStore)
    assert session.file_snapshot_store is store
    assert get_file_snapshot_store(session) is store


def test_record_content_snapshot(session: Session):
    tag = record_content_snapshot(session, "/fake/x.py", "hello\n")
    assert tag is not None
    store = get_file_snapshot_store(session)
    assert store.head("/fake/x.py").text == "hello\n"


def test_record_content_snapshot_oversized(session: Session):
    huge = "x" * (SNAPSHOT_MAX_BYTES + 1)
    assert record_content_snapshot(session, "/fake/big.txt", huge) is None
    assert get_file_snapshot_store(session).head("/fake/big.txt") is None


def _run(coro):
    import asyncio

    return asyncio.run(coro)


async def test_record_file_snapshot_reads_disk(session: Session, tmp_path):
    f = tmp_path / "src.py"
    f.write_bytes(b"print(1)\n")
    tag = await record_file_snapshot(session, str(f))
    assert tag is not None
    assert get_file_snapshot_store(session).head(str(f)).text == "print(1)\n"


async def test_record_file_snapshot_missing(session: Session, tmp_path):
    assert await record_file_snapshot(session, str(tmp_path / "missing.txt")) is None


async def test_record_file_snapshot_oversized_skips(session: Session, tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (SNAPSHOT_MAX_BYTES + 1024))
    assert await record_file_snapshot(session, str(f)) is None


async def test_record_file_snapshot_normalizes_crlf(session: Session, tmp_path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"a\r\nb\r\n")
    tag = await record_file_snapshot(session, str(f))
    head = get_file_snapshot_store(session).head(str(f))
    assert head.text == "a\nb\n"
    assert tag == compute_file_hash("a\nb\n")


def test_record_seen_lines_helper(session: Session):
    tag = record_content_snapshot(session, "/fake/y.py", "line\n")
    record_seen_lines(session, "/fake/y.py", tag, [1, 2])
    head = get_file_snapshot_store(session).head("/fake/y.py")
    assert head.seen_lines == {1, 2}


def test_parse_seen_lines_from_body():
    body = "     1\tread line\n  12#AB:hashline row\n *  5\tgrep hit\n  20-25\tspan\njunk"
    seen = parse_seen_lines_from_body(body)
    assert 1 in seen
    assert 12 in seen
    assert 5 in seen
    assert 20 in seen and 25 in seen


def test_record_seen_lines_from_body(session: Session):
    tag = record_content_snapshot(session, "/fake/z.py", "a\nb\nc\n")
    record_seen_lines_from_body(session, "/fake/z.py", tag, "   2\thello\n")
    head = get_file_snapshot_store(session).head("/fake/z.py")
    assert head.seen_lines == {2}
    # empty body → no-op
    record_seen_lines_from_body(session, "/fake/z.py", tag, "no lines here")
    assert head.seen_lines == {2}


def test_snapshot_dataclass_fields():
    snap = Snapshot(path="/x", text="t", hash="AAAA", recorded_at=1.0)
    assert snap.seen_lines is None
