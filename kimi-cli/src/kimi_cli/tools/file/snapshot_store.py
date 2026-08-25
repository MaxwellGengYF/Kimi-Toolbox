"""Session-bound file snapshot store for edit safety and hashline anchoring.

Two stores live here:

1. ``EditSnapshotStore`` — the original single-version-per-path store used by
   the hashline edit mode (``edit/modes/hashline.py``) and ``edit_safety.py``.
   Kept fully backward compatible.
2. ``InMemorySnapshotStore`` — the plan-25 versioned store (port of
   oh-my-pi ``packages/hashline/src/snapshots.ts``): a bounded, LRU-pruned
   ring of whole-file versions per canonical path, content-addressed with a
   4-hex uppercase xxHash32 tag, dedup-by-full-text.

Session wiring
--------------
``get_file_snapshot_store(session)`` lazily attaches an
``InMemorySnapshotStore`` to ``Session.file_snapshot_store`` (declared field,
``slots=True``).  ``get_edit_snapshot_store`` keeps using
``session.custom_data`` so existing consumers are untouched.

Producers (``read`` / ``HashRead`` / ``write`` / ``edit``) record what the
model saw or wrote; consumers (``HashEdit`` stale-anchor recovery, ``write``
pre-overwrite safety snapshot) resolve stale anchors back to recorded text.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import xxhash

from kimi_cli.session import Session

if TYPE_CHECKING:
    pass

SNAPSHOT_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB per-file cap
STORE_MAX_ENTRIES = 64  # legacy EditSnapshotStore capacity

DEFAULT_MAX_PATHS = 256
DEFAULT_MAX_VERSIONS_PER_PATH = 4
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024  # 64 MiB


# ═══════════════════════════════════════════════════════════════════════════
# Text normalization / tagging
# ═══════════════════════════════════════════════════════════════════════════


def normalize_to_lf(text: str) -> str:
    """Normalize *text* for hashing/storage: strip BOM, CRLF→LF, lone CR→LF."""
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def detect_line_ending(text: str) -> str:
    """Return the dominant line ending of *text* (``"\\r\\n"`` or ``"\\n"``)."""
    return "\r\n" if "\r\n" in text else "\n"


def compute_file_hash(text: str) -> str:
    """Whole-file content tag: xxHash32 low 16 bits, 4-hex uppercase.

    Matches oh-my-pi ``computeFileHash`` — LF/BOM-normalized before hashing.
    """
    digest = xxhash.xxh32(normalize_to_lf(text).encode("utf-8"), 0).intdigest()
    return f"{digest & 0xFFFF:04X}"


def canonical_snapshot_key(absolute_path: str) -> str:
    """Collapse symlink/relative forms into a stable canonical key."""
    path = Path(absolute_path)
    try:
        resolved = path.resolve(strict=False)
        return str(resolved)
    except (OSError, ValueError):
        try:
            parent = path.parent.resolve(strict=False)
            return str(parent / path.name)
        except (OSError, ValueError):
            return absolute_path


# ═══════════════════════════════════════════════════════════════════════════
# Versioned store (plan 25)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class Snapshot:
    """One recorded whole-file version."""

    path: str
    text: str  # LF-normalized, no BOM
    hash: str  # whole-file tag, 4-hex uppercase
    recorded_at: float
    seen_lines: set[int] | None = None


class SnapshotStore:
    """Abstract snapshot-store API (port of hashline ``SnapshotStore``)."""

    def head(self, path: str) -> Snapshot | None:
        raise NotImplementedError

    def by_hash(self, path: str, hash_: str) -> Snapshot | None:
        raise NotImplementedError

    def by_content(self, path: str, full_text: str) -> Snapshot | None:
        raise NotImplementedError

    def find_by_hash(self, hash_: str) -> list[Snapshot]:
        return []

    def versions(self, path: str) -> list[Snapshot]:
        """Return recorded versions for *path*, newest first."""
        return []

    def record(
        self,
        path: str,
        full_text: str,
        seen_lines: Iterable[int] | None = None,
    ) -> str | None:
        raise NotImplementedError

    def record_seen_lines(self, path: str, hash_: str, lines: Iterable[int]) -> None:
        raise NotImplementedError

    def invalidate(self, path: str) -> None:
        raise NotImplementedError

    def relocate(self, from_path: str, to_path: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemorySnapshotStore(SnapshotStore):
    """Bounded in-memory versioned snapshot store.

    ``OrderedDict`` of canonical path → version ring (oldest first, head
    last).  LRU by path recency (refreshed on record); per-path ring capped
    at ``max_versions_per_path``; total normalized-text bytes kept under
    ``max_total_bytes`` by evicting whole LRU path histories.
    """

    def __init__(
        self,
        max_paths: int = DEFAULT_MAX_PATHS,
        max_versions_per_path: int = DEFAULT_MAX_VERSIONS_PER_PATH,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self._paths: OrderedDict[str, list[Snapshot]] = OrderedDict()
        self._max_paths = max_paths
        self._max_versions = max_versions_per_path
        self._max_total_bytes = max_total_bytes
        self._total_bytes = 0

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _size(snap: Snapshot) -> int:
        return len(snap.text.encode("utf-8"))

    def _drop_history(self, key: str) -> None:
        versions = self._paths.pop(key, None)
        if versions:
            for snap in versions:
                self._total_bytes -= self._size(snap)

    def _enforce_bounds(self) -> None:
        while len(self._paths) > self._max_paths:
            oldest_key, _ = next(iter(self._paths.items()))
            self._drop_history(oldest_key)
        while self._total_bytes > self._max_total_bytes and self._paths:
            oldest_key, _ = next(iter(self._paths.items()))
            self._drop_history(oldest_key)

    # -- API -----------------------------------------------------------------

    def head(self, path: str) -> Snapshot | None:
        versions = self._paths.get(canonical_snapshot_key(path))
        return versions[-1] if versions else None

    def by_hash(self, path: str, hash_: str) -> Snapshot | None:
        versions = self._paths.get(canonical_snapshot_key(path))
        if not versions:
            return None
        for snap in versions:
            if snap.hash == hash_:
                return snap
        return None

    def by_content(self, path: str, full_text: str) -> Snapshot | None:
        versions = self._paths.get(canonical_snapshot_key(path))
        if not versions:
            return None
        normalized = normalize_to_lf(full_text)
        for snap in versions:
            if snap.text == normalized:
                return snap
        return None

    def find_by_hash(self, hash_: str) -> list[Snapshot]:
        found: list[Snapshot] = []
        for versions in self._paths.values():
            for snap in versions:
                if snap.hash == hash_:
                    found.append(snap)
        return found

    def versions(self, path: str) -> list[Snapshot]:
        ring = self._paths.get(canonical_snapshot_key(path))
        return list(reversed(ring)) if ring else []

    def record(
        self,
        path: str,
        full_text: str,
        seen_lines: Iterable[int] | None = None,
    ) -> str | None:
        """Record a version; returns its 4-hex tag, or ``None`` if oversized.

        Dedup by (hash AND full text) — a 16-bit tag collision must never
        fuse two distinct texts (oh-my-pi issue #4075).  Re-recording the
        same text refreshes recency, promotes the version to head, and
        unions ``seen_lines``.
        """
        text = normalize_to_lf(full_text)
        size = len(text.encode("utf-8"))
        if size > SNAPSHOT_MAX_BYTES:
            return None
        key = canonical_snapshot_key(path)
        tag = compute_file_hash(text)
        lines = set(seen_lines) if seen_lines else None

        versions = self._paths.get(key)
        if versions is not None:
            self._paths.move_to_end(key)
            for snap in versions:
                if snap.hash == tag and snap.text == text:
                    snap.recorded_at = time.time()
                    if lines:
                        if snap.seen_lines is None:
                            snap.seen_lines = set()
                        snap.seen_lines |= lines
                    if snap is not versions[-1]:
                        versions.remove(snap)
                        versions.append(snap)
                    return tag
        else:
            versions = []
            self._paths[key] = versions

        snap = Snapshot(
            path=key,
            text=text,
            hash=tag,
            recorded_at=time.time(),
            seen_lines=lines,
        )
        versions.append(snap)
        self._total_bytes += size
        while len(versions) > self._max_versions:
            dropped = versions.pop(0)
            self._total_bytes -= self._size(dropped)
        self._enforce_bounds()
        return tag

    def record_seen_lines(self, path: str, hash_: str, lines: Iterable[int]) -> None:
        key = canonical_snapshot_key(path)
        versions = self._paths.get(key)
        if not versions:
            return
        new_lines = set(lines)
        if not new_lines:
            return
        for snap in versions:
            if snap.hash == hash_:
                if snap.seen_lines is None:
                    snap.seen_lines = set()
                snap.seen_lines |= new_lines
                return

    def invalidate(self, path: str) -> None:
        self._drop_history(canonical_snapshot_key(path))

    def relocate(self, from_path: str, to_path: str) -> None:
        src = canonical_snapshot_key(from_path)
        dst = canonical_snapshot_key(to_path)
        if src == dst:
            return
        versions = self._paths.pop(src, None)
        if versions is None:
            return
        self._drop_history(dst)
        self._paths[dst] = versions
        self._paths.move_to_end(dst)

    def clear(self) -> None:
        self._paths.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._paths)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy single-version store (kept for hashline mode + edit_safety)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SnapshotEntry:
    """One recorded snapshot of a file."""

    tag: str
    content: str  # LF-normalized
    line1_hash: str | None = None
    seen_lines: set[int] = field(default_factory=set)
    recorded_at: float = field(default_factory=time.time)


class EditSnapshotStore:
    """In-memory per-session snapshot store."""

    def __init__(self, max_entries: int = STORE_MAX_ENTRIES) -> None:
        self._store: dict[str, SnapshotEntry] = {}
        self._max_entries = max_entries

    def _prune(self) -> None:
        """Evict the oldest entry for a *different* path when over capacity.

        Re-recording the same path replaces its entry in place, so the same
        path does not cause eviction of other paths.
        """
        while len(self._store) > self._max_entries:
            oldest_other: tuple[str, SnapshotEntry] | None = None
            for key, entry in self._store.items():
                if oldest_other is None or entry.recorded_at < oldest_other[1].recorded_at:
                    oldest_other = (key, entry)
            if oldest_other is None:
                break
            del self._store[oldest_other[0]]

    def record(
        self,
        absolute_path: str,
        content: str,
        seen_lines: Iterable[int] | None = None,
    ) -> str | None:
        """Record a normalized snapshot and return its tag, or None if too large."""
        content = content.replace("\r\n", "\n")
        if len(content) > SNAPSHOT_MAX_BYTES:
            return None
        key = canonical_snapshot_key(absolute_path)
        tag = _content_tag(content)
        line1_hash = _line1_hash(content)
        entry = self._store.get(key)
        if entry is None:
            entry = SnapshotEntry(tag=tag, content=content, line1_hash=line1_hash)
            self._store[key] = entry
        else:
            entry.tag = tag
            entry.content = content
            entry.line1_hash = line1_hash
            entry.recorded_at = time.time()
        if seen_lines:
            entry.seen_lines.update(seen_lines)
        self._prune()
        return tag

    def record_seen_lines(self, absolute_path: str, tag: str, lines: Iterable[int]) -> None:
        """Attach displayed line numbers to an existing snapshot by tag."""
        key = canonical_snapshot_key(absolute_path)
        entry = self._store.get(key)
        if entry is None or entry.tag != tag:
            return
        entry.seen_lines.update(lines)

    def lookup(self, absolute_path: str) -> SnapshotEntry | None:
        """Return the current snapshot for a path, or None."""
        return self._store.get(canonical_snapshot_key(absolute_path))

    def __len__(self) -> int:
        return len(self._store)


_store_key = "__edit_snapshot_store__"


def get_edit_snapshot_store(session: Session) -> EditSnapshotStore:
    """Lazily fetch the per-session legacy snapshot store."""
    store = session.custom_data.get(_store_key)
    if store is None:
        store = EditSnapshotStore()
        session.custom_data[_store_key] = store
    return store


def get_file_snapshot_store(session: Session) -> InMemorySnapshotStore:
    """Lazily fetch (and attach) the per-session versioned snapshot store."""
    store = getattr(session, "file_snapshot_store", None)
    if store is None:
        store = InMemorySnapshotStore()
        session.file_snapshot_store = store
    return store


def _content_tag(content: str) -> str:
    """Return the first 8 hex chars of xxhash64 of the content."""
    return xxhash.xxh64(content.encode("utf-8")).hexdigest()[:8]


def _line1_hash(content: str) -> str:
    """Return the 2-char cumulative hash for line 1 of the content."""
    from kimi_cli.tools.file.hash_line import compute_line_hash

    first = content.split("\n")[0].rstrip("\r")
    return compute_line_hash(1, first, None)


# ═══════════════════════════════════════════════════════════════════════════
# Producer helpers
# ═══════════════════════════════════════════════════════════════════════════


def _record_both_stores(
    session: Session,
    absolute_path: str,
    text: str,
    seen_lines: Iterable[int] | None,
) -> str | None:
    """Best-effort record into the legacy + versioned stores; new tag back."""
    tag: str | None = None
    try:
        tag = get_file_snapshot_store(session).record(absolute_path, text, seen_lines)
    except Exception:
        tag = None
    try:
        get_edit_snapshot_store(session).record(absolute_path, text, seen_lines)
    except Exception:
        pass
    return tag


async def record_file_snapshot(
    session: Session,
    absolute_path: str,
    seen_lines: Iterable[int] | None = None,
) -> str | None:
    """Read ``absolute_path``, record its LF-normalized snapshot, return the tag."""
    try:
        from kaos.path import KaosPath

        path = KaosPath(absolute_path)
        if not await path.exists():
            return None
        stat = await path.stat()
        if stat.st_size > SNAPSHOT_MAX_BYTES:
            return None
        text = await path.read_text(encoding="utf-8")
        normalized = normalize_to_lf(text)
        return _record_both_stores(session, absolute_path, normalized, seen_lines)
    except Exception:
        return None


def record_content_snapshot(
    session: Session,
    absolute_path: str,
    full_text: str,
    seen_lines: Iterable[int] | None = None,
) -> str | None:
    """Record text a caller already holds (avoids double I/O); returns the tag."""
    try:
        if len(full_text.encode("utf-8")) > SNAPSHOT_MAX_BYTES:
            return None
    except Exception:
        return None
    try:
        return _record_both_stores(session, absolute_path, full_text, seen_lines)
    except Exception:
        return None


def record_seen_lines(
    session: Session,
    absolute_path: str,
    tag: str,
    lines: Iterable[int],
) -> None:
    """Attach displayed line numbers to a versioned snapshot by tag."""
    try:
        get_file_snapshot_store(session).record_seen_lines(absolute_path, tag, lines)
        get_edit_snapshot_store(session).record_seen_lines(absolute_path, tag, lines)
    except Exception:
        pass


def parse_seen_lines_from_hashline_body(body: str) -> list[int]:
    """Extract 1-indexed displayed lines from a tool output body.

    Recognizes ``NN\t`` / ``NN:`` (read), ``NN#HASH:`` (HashRead), an
    optional leading ``*`` match marker (grep), and ``NN-MM`` collapsed
    ranges (both boundaries recorded).
    """
    import regex as re

    seen: list[int] = []
    prefix = re.compile(r"^[ *]*(\d+)(?:#[^:\t]*)?(?:-(\d+)(?:#[^:\t]*)?)?[:\t]")
    for row in body.splitlines():
        match = prefix.match(row)
        if not match:
            continue
        seen.append(int(match.group(1)))
        end = match.group(2)
        if end is not None:
            seen.append(int(end))
    return seen


def parse_seen_lines_from_body(body: str) -> list[int]:
    """Extract displayed lines from any tool body.

    Recognizes ``NN:`` (read), ``NN#HASH:`` (HashRead), an optional leading
    ``*`` match marker (grep), and ``NN-MM:`` collapsed ranges (both
    boundaries recorded).
    """
    return parse_seen_lines_from_hashline_body(body)


def record_seen_lines_from_body(
    session: Session,
    absolute_path: str,
    tag: str,
    body: str,
) -> None:
    """Record seen lines parsed from a tool body against a snapshot tag."""
    lines = parse_seen_lines_from_body(body)
    if not lines:
        return
    record_seen_lines(session, absolute_path, tag, lines)
