"""Direct unit tests for FileMTime external-modification detection."""

from __future__ import annotations

import os
from pathlib import Path

from kimi_cli.file_mtine import FileMTime


def _bump_mtime(path: Path, delta: float = 2.0) -> None:
    """Deterministically move *path*'s mtime forward by *delta* seconds."""
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + delta))


class TestFileMTime:
    def test_read_then_write_allowed(self, tmp_path: Path) -> None:
        """A fresh read (clean_file) legitimizes a subsequent write."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        tracker.clean_file(str(f))
        assert tracker.mark_dirty(str(f)) is True

    def test_mark_dirty_twice_blocks_second_write(self, tmp_path: Path) -> None:
        """mark_dirty twice with no change: the second call returns False."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        assert tracker.mark_dirty(str(f)) is True
        assert tracker.mark_dirty(str(f)) is False

    def test_external_modification_after_read_blocks_write(self, tmp_path: Path) -> None:
        """clean_file (read) -> external modify -> mark_dirty returns False."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        tracker.clean_file(str(f))
        _bump_mtime(f)
        assert tracker.mark_dirty(str(f)) is False

    def test_our_write_after_read_blocks_second_write(self, tmp_path: Path) -> None:
        """clean_file -> mark_dirty True -> our own write (mtime bump) -> mark_dirty False."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        tracker.clean_file(str(f))
        assert tracker.mark_dirty(str(f)) is True
        _bump_mtime(f)  # simulate the mtime bump caused by our own write
        assert tracker.mark_dirty(str(f)) is False

    def test_reread_after_external_modification_allows_write(self, tmp_path: Path) -> None:
        """clean_file -> external modify -> clean_file (re-read) -> mark_dirty True."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        tracker.clean_file(str(f))
        _bump_mtime(f)
        tracker.clean_file(str(f))  # a fresh read resets the baseline
        assert tracker.mark_dirty(str(f)) is True

    def test_clean_file_drops_write_baseline(self, tmp_path: Path) -> None:
        """A read between two writes allows the second write."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        assert tracker.mark_dirty(str(f)) is True  # first write
        _bump_mtime(f)  # the write bumped the mtime
        tracker.clean_file(str(f))  # read in between
        assert tracker.mark_dirty(str(f)) is True  # second write allowed

    def test_missing_file_allowed(self, tmp_path: Path) -> None:
        """Writing a file that does not exist is always allowed."""
        tracker = FileMTime()
        f = tmp_path / "never.txt"
        assert tracker.mark_dirty(str(f)) is True

    def test_file_deleted_after_read_allowed(self, tmp_path: Path) -> None:
        """A file deleted after a read is safe to (re)create."""
        f = tmp_path / "f.txt"
        f.write_text("original")
        tracker = FileMTime()

        tracker.clean_file(str(f))
        f.unlink()
        assert tracker.mark_dirty(str(f)) is True
