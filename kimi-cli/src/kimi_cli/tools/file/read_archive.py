"""Archive browsing/reading helpers for the ``read`` tool.

Supports the stdlib archive families first (zip/jar/war/apk/whl/etc, tar
and tar.*, plus bare gzip/bzip2/xz payloads). Optional formats such as 7z,
rar, and epub are left for a later phase.
"""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import regex as re

__all__ = [
    "ARCHIVE_EXTENSIONS",
    "is_archive_path",
    "sniff_archive",
    "ArchiveEntry",
    "ArchiveReader",
    "normalize_member_path",
    "MAX_ARCHIVE_MEMBER_BYTES",
    "MAX_ARCHIVE_TOTAL_BYTES",
    "MAX_ARCHIVE_LIST_ENTRIES",
]

# Longest extension first so ``.tar.gz`` wins over ``.gz``.
ARCHIVE_EXTENSIONS: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
    ".tgz",
    ".tbz2",
    ".tbz",
    ".txz",
    ".zip",
    ".jar",
    ".war",
    ".ear",
    ".apk",
    ".whl",
    ".xpi",
    ".vsix",
    ".nupkg",
    ".cbz",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
)

_MAX_COMPRESSION_RATIO = 10
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024  # 8 MiB per member
MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024  # 32 MiB total in-memory budget
MAX_ARCHIVE_LIST_ENTRIES = 500  # default cap for directory listings


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    name: str
    is_dir: bool
    size: int | None


def _longest_archive_extension(path: str) -> str:
    """Return the longest matching archive extension, or empty string."""
    lower = path.lower()
    for ext in ARCHIVE_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return ""


def is_archive_path(path: str) -> bool:
    return bool(_longest_archive_extension(path))


def sniff_archive(header: bytes) -> bool:
    """Return True when *header* looks like a supported archive."""
    if len(header) < 4:
        return False
    if header[:4] == b"PK\x03\x04":
        return True  # zip / jar / etc.
    if header[:2] == b"\x1f\x8b":
        return True  # gzip
    if header[:3] == b"BZh":
        return True  # bzip2
    if header[:6] == b"\xfd7zXZ\x00":
        return True  # xz
    if len(header) >= 262 and header[257:265] == b"ustar\x00":
        return True  # tar
    if header[:4] == b"\x28\xb5\x2f\xfd":
        return True  # zstd
    return False


def normalize_member_path(member: str) -> str | None:
    """Normalize an archive member path for safe lookup.

    Returns ``None`` for traversal attempts (``..``), absolute paths,
    backslash escapes, or NUL bytes. Otherwise returns a forward-slash
    normalized path with redundant ``.``/empty segments removed.
    """
    if not member:
        return None
    if "\x00" in member:
        return None
    # Reject absolute paths.
    if member.startswith("/"):
        return None
    # Windows-style separators are not allowed as path separators inside
    # archives; treat them as a zip-slip-style escape attempt.
    if "\\" in member:
        return None
    parts = member.replace("\\", "/").split("/")
    cleaned: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            return None
        cleaned.append(part)
    if not cleaned:
        return None
    return "/".join(cleaned)


class _SizeGuard:
    """Tracks decompressed bytes to guard against decompression bombs."""

    def __init__(self, per_member: int, total: int) -> None:
        self.per_member = per_member
        self.total_limit = total
        self.total_used = 0

    def check(self, n: int) -> bool:
        if n > self.per_member:
            return False
        if self.total_used + n > self.total_limit:
            return False
        self.total_used += n
        return True


class ArchiveReader:
    """Context-manager wrapper around stdlib archive readers.

    Usage::

        with ArchiveReader(path) as reader:
            entries = reader.list_directory()
            data = reader.read_file("src/main.py")
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader: zipfile.ZipFile | tarfile.TarFile | None = None
        self._kind: str = ""
        self._guard = _SizeGuard(MAX_ARCHIVE_MEMBER_BYTES, MAX_ARCHIVE_TOTAL_BYTES)

    def __enter__(self) -> "ArchiveReader":
        ext = _longest_archive_extension(str(self.path))
        if ext in {".zip", ".jar", ".war", ".ear", ".apk", ".whl", ".xpi", ".vsix", ".nupkg", ".cbz"}:
            self._reader = zipfile.ZipFile(self.path, "r")
            self._kind = "zip"
        elif ext in {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tbz", ".tar.xz", ".txz", ".tar.zst"}:
            self._reader = tarfile.open(self.path, "r:*")
            self._kind = "tar"
        elif ext in {".gz", ".bz2", ".xz"}:
            # Bare compressed payload: expose it as a single named member.
            self._kind = f"compressed-{ext.lstrip('.')}"
        else:
            # Fallback: try zip, then tar, then bare compressed.
            try:
                self._reader = zipfile.ZipFile(self.path, "r")
                self._kind = "zip"
            except zipfile.BadZipFile:
                try:
                    self._reader = tarfile.open(self.path, "r:*")
                    self._kind = "tar"
                except tarfile.TarError as exc:
                    raise ValueError(f"Unsupported or malformed archive: {self.path}") from exc
        return self

    def __exit__(self, *exc) -> None:
        if self._reader is not None:
            self._reader.close()

    def list_directory(self, prefix: str = "", *, offset: int = 1, limit: int = MAX_ARCHIVE_LIST_ENTRIES) -> list[ArchiveEntry]:
        """List archive entries under *prefix* with 1-indexed pagination.

        ``offset`` is 1-indexed; ``limit`` caps the returned entries. Directory
        entries are synthesized for tar archives and for zip archives that do
        not contain explicit directory entries.
        """
        if self._kind == "compressed-gz" or self._kind == "compressed-bz2" or self._kind == "compressed-xz":
            # Single synthetic member named after the decompressed stem.
            stem = self.path.name
            for ext in ARCHIVE_EXTENSIONS:
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            size = self.path.stat().st_size
            return [ArchiveEntry(name=stem or "data", is_dir=False, size=size)]

        entries: dict[str, ArchiveEntry] = {}
        if self._kind == "zip" and isinstance(self._reader, zipfile.ZipFile):
            for info in self._reader.infolist():
                name = normalize_member_path(info.filename)
                if name is None:
                    continue
                if prefix and not name.startswith(prefix.rstrip("/") + "/") and name != prefix:
                    continue
                rel = name[len(prefix) :] if prefix else name
                if not rel:
                    continue
                first = rel.split("/", 1)[0]
                is_dir = rel.endswith("/") or info.is_dir()
                rest = rel[len(first) + 1 :]
                if not is_dir and rest:
                    # Synthesize a directory entry for the parent prefix.
                    entries[first] = ArchiveEntry(name=first, is_dir=True, size=None)
                elif is_dir:
                    entries[first] = ArchiveEntry(name=first, is_dir=True, size=None)
                else:
                    entries.setdefault(first, ArchiveEntry(name=first, is_dir=False, size=info.file_size))
        elif self._kind == "tar" and isinstance(self._reader, tarfile.TarFile):
            for member in self._reader.getmembers():
                name = normalize_member_path(member.name)
                if name is None:
                    continue
                if prefix and not name.startswith(prefix.rstrip("/") + "/") and name != prefix:
                    continue
                rel = name[len(prefix) :] if prefix else name
                if not rel:
                    continue
                first = rel.split("/", 1)[0]
                is_dir = member.isdir() or rel.endswith("/")
                rest = rel[len(first) + 1 :]
                if not is_dir and rest:
                    entries[first] = ArchiveEntry(name=first, is_dir=True, size=None)
                elif is_dir:
                    entries[first] = ArchiveEntry(name=first, is_dir=True, size=None)
                else:
                    size = member.size if not member.issym() and not member.islnk() else None
                    entries.setdefault(first, ArchiveEntry(name=first, is_dir=False, size=size))

        sorted_entries = sorted(entries.values(), key=lambda e: (not e.is_dir, e.name.lower()))
        start = max(0, offset - 1)
        return sorted_entries[start : start + limit]

    def read_file(self, member: str) -> bytes:
        """Read *member* bytes with decompression-bomb guards.

        Raises ``ValueError`` when the member would exceed caps.
        """
        norm = normalize_member_path(member)
        if norm is None:
            raise ValueError(f"Invalid or unsafe archive member path: {member!r}")

        if self._kind == "compressed-gz":
            return self._read_compressed_single(lambda: __import__("gzip").open(self.path, "rb"))
        if self._kind == "compressed-bz2":
            return self._read_compressed_single(lambda: __import__("bz2").open(self.path, "rb"))
        if self._kind == "compressed-xz":
            return self._read_compressed_single(lambda: __import__("lzma").open(self.path, "rb"))

        if self._reader is None:
            raise ValueError("Archive reader is not open")

        if self._kind == "zip" and isinstance(self._reader, zipfile.ZipFile):
            info = self._reader.getinfo(norm)
            # Rough guard: compressed size * ratio is a cheap upper bound.
            est = info.file_size or (info.compress_size * _MAX_COMPRESSION_RATIO)
            if est > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(
                    f"Archive member '{member}' is too large to read safely "
                    f"(estimated {est} bytes; limit {MAX_ARCHIVE_MEMBER_BYTES})."
                )
            data = self._reader.read(norm)
            if not self._guard.check(len(data)):
                raise ValueError(
                    f"Archive member '{member}' exceeds the safe in-memory budget."
                )
            return data

        if self._kind == "tar" and isinstance(self._reader, tarfile.TarFile):
            m = self._reader.getmember(norm)
            if m is None:
                raise ValueError(f"Archive member not found: {member!r}")
            if m.size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(
                    f"Archive member '{member}' is too large to read safely "
                    f"({m.size} bytes; limit {MAX_ARCHIVE_MEMBER_BYTES})."
                )
            f = self._reader.extractfile(m)
            if f is None:
                raise ValueError(f"Archive member '{member}' is not a regular file")
            data = f.read()
            if not self._guard.check(len(data)):
                raise ValueError(
                    f"Archive member '{member}' exceeds the safe in-memory budget."
                )
            return data

        raise ValueError(f"Unsupported archive kind: {self._kind}")

    def _read_compressed_single(self, opener) -> bytes:
        # Opener returns a file-like object; read up to the per-member cap.
        with opener() as f:
            data = f.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
        if len(data) > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(
                f"Compressed file is too large to read safely "
                f"({len(data)} bytes; limit {MAX_ARCHIVE_MEMBER_BYTES})."
            )
        if not self._guard.check(len(data)):
            raise ValueError("Compressed file exceeds the safe in-memory budget.")
        return data


def is_binary_data(data: bytes, sample_size: int = 1024) -> bool:
    """Best-effort binary detection using NUL bytes."""
    chunk = data[:sample_size]
    return b"\x00" in chunk or (len(chunk) > 0 and any(b > 127 for b in chunk) and b"\n" not in chunk)


def format_archive_listing(entries: list[ArchiveEntry]) -> str:
    """Render archive entries as plain text lines."""
    lines: list[str] = []
    for entry in entries:
        if entry.is_dir:
            lines.append(f"{entry.name}/")
        elif entry.size is not None:
            lines.append(f"{entry.name} ({entry.size} bytes)")
        else:
            lines.append(entry.name)
    return "\n".join(lines)
