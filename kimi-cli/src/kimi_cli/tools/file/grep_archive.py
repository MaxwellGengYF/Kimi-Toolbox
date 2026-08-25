"""Archive member search support for richer grep (plans/23-grep-rich.md §4.6).

Port of oh-my-pi's ``resolveArchiveSearchPaths`` + ``pi-utils/ar``: a path
like ``bundle.zip:src/foo.ts`` is parsed into (archive, member) candidates;
each UTF-8 text member is extracted to a temp scratch file so the existing
ripgrep/pure-Python pipeline can search it; after search, results are
remapped back to the ``archive:member`` selector.

Reuses ``read_archive.ArchiveReader`` (plans/21) for member reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import regex as re

from kimi_cli.tools.file.grep_selectors import GrepPathSpec
from kimi_cli.tools.file.read_archive import (
    ARCHIVE_EXTENSIONS,
    ArchiveReader,
    is_archive_path,
)

__all__ = [
    "ARCHIVE_EXTENSIONS",
    "MAX_ARCHIVE_MEMBER_BYTES",
    "MAX_ARCHIVE_TOTAL_BYTES",
    "parse_archive_path_candidates",
    "read_archive_member_bytes",
    "materialize_archive_members",
]

MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024  # per-member cap (plan 21 parity)
MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024  # total materialized cap (bomb guard)


def parse_archive_path_candidates(entry: str) -> list[tuple[str, str]]:
    """Split ``entry`` on ':' rightmost-first into (archive, member) pairs.

    A candidate is returned when the left side looks like an archive path
    (name ends in a known archive extension) and the right side is non-empty.
    Pairs are ordered rightmost-split-first so the innermost member wins.
    """
    candidates: list[tuple[str, str]] = []
    rest = entry
    while True:
        idx = rest.rfind(":")
        if idx <= 0:
            break
        left, member = rest[:idx], rest[idx + 1 :]
        if member and is_archive_path(left):
            candidates.append((left, member))
            # Keep splitting: the left part itself may contain another
            # archive layer (nested archives are rare but cheap to support).
            rest = left
            continue
        # Not an archive on the left; no deeper candidate can qualify.
        break
    return candidates


def read_archive_member_bytes(archive_path: Path, member: str) -> bytes:
    """Read a member's raw bytes from an archive, honoring the member cap.

    Raises on missing member, unreadable archive, or size-cap violations.
    """
    with ArchiveReader(str(archive_path)) as reader:
        data = reader.read_file(member)
    if len(data) > MAX_ARCHIVE_MEMBER_BYTES:
        raise ValueError(
            f"Archive member {member!r} exceeds the {MAX_ARCHIVE_MEMBER_BYTES}-byte cap."
        )
    return data


def _safe_scratch_name(member: str) -> str:
    """Sanitize a member path into a flat scratch-file basename."""
    base = os.path.basename(member.replace("\\", "/")) or "member"
    safe = re.sub(r"[^\w.-]+", "_", base)
    return safe or "member"


async def materialize_archive_members(
    path_specs: list[GrepPathSpec],
    cwd: Path,
    scratch_dir: Path,
) -> tuple[list[GrepPathSpec], dict[str, str], list[str]]:
    """Extract archive members to scratch files for searching.

    For each spec whose ``clean`` path parses as ``archive:member`` (possibly
    with a trailing selector already peeled into ``spec.ranges`` semantics by
    the caller), the member is read, rejected when binary (NUL byte) or not
    strict UTF-8, and otherwise written to ``scratch_dir/<idx>-<safe_base>``.

    Returns ``(rewritten_specs, display_map, unreadable_notes)`` where
    ``display_map`` maps each scratch path to the original ``archive:member``
    selector and ``unreadable_notes`` lists human-readable skip reasons.
    """
    import asyncio

    rewritten: list[GrepPathSpec] = []
    display_map: dict[str, str] = {}
    unreadable: list[str] = []
    total_bytes = 0

    def _materialize_sync() -> None:
        nonlocal total_bytes
        idx = 0
        for spec in path_specs:
            candidates = parse_archive_path_candidates(spec.clean)
            if not candidates:
                rewritten.append(spec)
                continue
            archive_str, member = candidates[0]
            archive_path = Path(archive_str)
            if not archive_path.is_absolute():
                archive_path = (cwd / archive_path).resolve()
            # Display keeps the entry as the caller wrote it (relative to cwd
            # when it was relative) so results round-trip through read syntax.
            original = spec.original
            display_key = f"{archive_str}:{member}"
            try:
                data = read_archive_member_bytes(archive_path, member)
            except Exception as exc:  # noqa: BLE001 — surface as a skip note
                unreadable.append(f"{original}: {exc}")
                continue
            if b"\x00" in data:
                unreadable.append(f"{original}: binary member (text members only)")
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                unreadable.append(f"{original}: not UTF-8 text")
                continue
            total_bytes += len(data)
            if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                unreadable.append(f"{original}: archive size cap exceeded")
                continue
            scratch = scratch_dir / f"{idx}-{_safe_scratch_name(member)}"
            scratch.write_text(text, encoding="utf-8", newline="\n")
            idx += 1
            display_map[str(scratch)] = display_key
            rewritten.append(
                GrepPathSpec(
                    original=spec.original,
                    clean=str(scratch),
                    literal_filesystem_match=False,
                    ranges=spec.ranges,
                )
            )

    await asyncio.to_thread(_materialize_sync)
    return rewritten, display_map, unreadable
