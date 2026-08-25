"""Session-bound file snapshot store for edit safety and hashline anchoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import xxhash

from kimi_cli.session import Session

SNAPSHOT_MAX_BYTES = 4 * 1024 * 1024
STORE_MAX_ENTRIES = 64


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
    """Lazily fetch the per-session snapshot store."""
    store = session.custom_data.get(_store_key)
    if store is None:
        store = EditSnapshotStore()
        session.custom_data[_store_key] = store
    return store


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


def _content_tag(content: str) -> str:
    """Return the first 8 hex chars of xxhash64 of the content."""
    return xxhash.xxh64(content.encode("utf-8")).hexdigest()[:8]


def _line1_hash(content: str) -> str:
    """Return the 2-char cumulative hash for line 1 of the content."""
    from kimi_cli.tools.file.hash_line import compute_line_hash

    first = content.split("\n")[0].rstrip("\r")
    return compute_line_hash(1, first, None)


async def record_file_snapshot(
    session: Session,
    absolute_path: str,
    seen_lines: Iterable[int] | None = None,
) -> str | None:
    """Read `absolute_path`, record its LF-normalized snapshot, and return the tag."""
    try:
        from kaos.path import KaosPath

        path = KaosPath(absolute_path)
        if not await path.exists():
            return None
        stat = await path.stat()
        if stat.st_size > SNAPSHOT_MAX_BYTES:
            return None
        text = await path.read_text(encoding="utf-8")
        normalized = text.replace("\r\n", "\n")
        return get_edit_snapshot_store(session).record(absolute_path, normalized, seen_lines)
    except Exception:
        return None


def parse_seen_lines_from_hashline_body(body: str) -> list[int]:
    """Extract 1-indexed displayed lines from a hashline-formatted body."""
    import regex as re

    seen: list[int] = []
    prefix = re.compile(r"^[ *]?(\d+)(?:#[^:]*)?(?:-(\d+)(?:#[^:]*)?)?:")
    for row in body.splitlines():
        match = prefix.match(row)
        if not match:
            continue
        seen.append(int(match.group(1)))
        end = match.group(2)
        if end is not None:
            seen.append(int(end))
    return seen


def record_seen_lines_from_body(
    session: Session,
    absolute_path: str,
    tag: str,
    body: str,
) -> None:
    """Record seen lines parsed from a hashline body against a snapshot tag."""
    lines = parse_seen_lines_from_hashline_body(body)
    if not lines:
        return
    get_edit_snapshot_store(session).record_seen_lines(absolute_path, tag, lines)
