"""Tests for grep_recorder.py (plans/23-grep-rich.md §7)."""

from __future__ import annotations

from kimi_cli.tools.file.grep_recorder import (
    RECORDER_CAP,
    FileRecorder,
    get_recorded_grep_files,
    record_grep_files,
)


class TestFileRecorder:
    def test_dedup_insertion_order(self):
        rec = FileRecorder()
        rec.record("b.py")
        rec.record("a.py")
        rec.record("b.py")  # duplicate
        assert rec.list == ["b.py", "a.py"]

    def test_empty_skipped(self):
        rec = FileRecorder()
        rec.record("")
        assert rec.list == []

    def test_len(self):
        rec = FileRecorder()
        rec.record("a")
        rec.record("b")
        assert len(rec) == 2


class TestSessionPersistence:
    def test_record_and_get(self, session):
        record_grep_files(session, ["a.py", "b.py"], cwd="/work")
        assert get_recorded_grep_files(session) == ["a.py", "b.py"]

    def test_merge_preserves_order(self, session):
        record_grep_files(session, ["a.py"])
        record_grep_files(session, ["b.py", "a.py"])  # a.py dup across calls
        assert get_recorded_grep_files(session) == ["a.py", "b.py"]

    def test_metadata_stored(self, session):
        record_grep_files(session, ["a.py"], cwd="/work")
        data = session.custom_data["grep"]
        assert data["cwd"] == "/work"
        assert "updated_at" in data

    def test_cap_drops_oldest(self, session):
        first = [f"file{i}.py" for i in range(RECORDER_CAP)]
        record_grep_files(session, first)
        record_grep_files(session, ["newest.py"])
        files = get_recorded_grep_files(session)
        assert len(files) == RECORDER_CAP
        assert files[-1] == "newest.py"
        assert files[0] == "file1.py"  # file0.py dropped from the front

    def test_empty_record_noop(self, session):
        record_grep_files(session, [])
        assert get_recorded_grep_files(session) == []
        assert "grep" not in session.custom_data

    def test_get_empty_when_nothing(self, session):
        assert get_recorded_grep_files(session) == []

    def test_corrupt_custom_data_tolerated(self, session):
        session.custom_data["grep"] = "not-a-dict"
        record_grep_files(session, ["a.py"])
        assert get_recorded_grep_files(session) == ["a.py"]
