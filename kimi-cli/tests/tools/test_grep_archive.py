"""Tests for grep_archive.py (plans/23-grep-rich.md §7)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from kimi_cli.tools.file.grep_archive import (
    MAX_ARCHIVE_MEMBER_BYTES,
    MAX_ARCHIVE_TOTAL_BYTES,
    materialize_archive_members,
    parse_archive_path_candidates,
    read_archive_member_bytes,
)
from kimi_cli.tools.file.grep_selectors import GrepPathSpec, LineRange


@pytest.fixture
def bundle(tmp_path):
    """Build bundle.zip with src/foo.ts (text) and img.bin (binary)."""
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("src/foo.ts", "export const x = 1;\nexport const y = 2;\n")
        zf.writestr("img.bin", b"\x00\x01\x02binary")
        zf.writestr("bad.txt", b"\xff\xfe invalid utf8")
    return tmp_path, zpath


@pytest.fixture
def tar_bundle(tmp_path):
    tpath = tmp_path / "pkg.tar.gz"
    src = tmp_path / "inner.txt"
    src.write_text("hello tar\n")
    with tarfile.open(tpath, "w:gz") as tf:
        tf.add(src, arcname="inner.txt")
    return tmp_path, tpath


class TestParseArchivePathCandidates:
    def test_zip(self):
        assert parse_archive_path_candidates("bundle.zip:src/foo.ts") == [
            ("bundle.zip", "src/foo.ts")
        ]

    def test_tar_gz(self):
        assert parse_archive_path_candidates("pkg.tar.gz:a/b.txt") == [
            ("pkg.tar.gz", "a/b.txt")
        ]

    def test_tar(self):
        assert parse_archive_path_candidates("x.tar:y.txt") == [("x.tar", "y.txt")]

    def test_non_archive_left(self):
        assert parse_archive_path_candidates("src/foo.py:50-100") == []

    def test_empty_member(self):
        assert parse_archive_path_candidates("bundle.zip:") == []

    def test_no_colon(self):
        assert parse_archive_path_candidates("bundle.zip") == []


class TestReadArchiveMemberBytes:
    def test_read_member(self, bundle):
        tmp_path, zpath = bundle
        data = read_archive_member_bytes(zpath, "src/foo.ts")
        assert data == b"export const x = 1;\nexport const y = 2;\n"

    def test_missing_member_raises(self, bundle):
        _, zpath = bundle
        with pytest.raises(Exception):
            read_archive_member_bytes(zpath, "nope.txt")

    def test_member_cap(self, tmp_path):
        zpath = tmp_path / "big.zip"
        big = b"a" * (MAX_ARCHIVE_MEMBER_BYTES + 1)
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("big.bin", big)
        with pytest.raises(ValueError, match="byte cap|too large"):
            read_archive_member_bytes(zpath, "big.bin")


class TestMaterializeArchiveMembers:
    async def test_remapping(self, bundle):
        tmp_path, zpath = bundle
        specs = [GrepPathSpec(original="bundle.zip:src/foo.ts", clean="bundle.zip:src/foo.ts")]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            specs, tmp_path, scratch
        )
        assert len(rewritten) == 1
        scratch_path = rewritten[0].clean
        assert Path(scratch_path).is_file()
        assert display_map[scratch_path] == "bundle.zip:src/foo.ts"
        assert unreadable == []
        assert "export const x" in Path(scratch_path).read_text()

    async def test_binary_member_skipped(self, bundle):
        tmp_path, zpath = bundle
        specs = [GrepPathSpec(original="bundle.zip:img.bin", clean="bundle.zip:img.bin")]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            specs, tmp_path, scratch
        )
        assert rewritten == []
        assert any("binary" in n for n in unreadable)

    async def test_non_utf8_skipped(self, bundle):
        tmp_path, zpath = bundle
        specs = [GrepPathSpec(original="bundle.zip:bad.txt", clean="bundle.zip:bad.txt")]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            specs, tmp_path, scratch
        )
        assert rewritten == []
        assert any("UTF-8" in n for n in unreadable)

    async def test_non_archive_passthrough(self, bundle):
        tmp_path, _ = bundle
        spec = GrepPathSpec(original="src/foo.py:1-5", clean="src/foo.py")
        spec.ranges = [LineRange(1, 5)]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            [spec], tmp_path, scratch
        )
        assert rewritten == [spec]
        assert display_map == {}
        assert unreadable == []

    async def test_tar_member(self, tar_bundle):
        tmp_path, tpath = tar_bundle
        specs = [GrepPathSpec(original="pkg.tar.gz:inner.txt", clean="pkg.tar.gz:inner.txt")]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            specs, tmp_path, scratch
        )
        assert len(rewritten) == 1
        assert "hello tar" in Path(rewritten[0].clean).read_text()

    async def test_ranges_preserved(self, bundle):
        tmp_path, zpath = bundle
        spec = GrepPathSpec(
            original="bundle.zip:src/foo.ts:1-1", clean="bundle.zip:src/foo.ts"
        )
        spec.ranges = [LineRange(1, 1)]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, _, _ = await materialize_archive_members([spec], tmp_path, scratch)
        assert rewritten[0].ranges == [LineRange(1, 1)]

    async def test_total_cap(self, bundle, monkeypatch):
        import kimi_cli.tools.file.grep_archive as ga

        monkeypatch.setattr(ga, "MAX_ARCHIVE_TOTAL_BYTES", 10)
        tmp_path, zpath = bundle
        specs = [GrepPathSpec(original="bundle.zip:src/foo.ts", clean="bundle.zip:src/foo.ts")]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rewritten, display_map, unreadable = await materialize_archive_members(
            specs, tmp_path, scratch
        )
        assert rewritten == []
        assert any("size cap" in n for n in unreadable)
