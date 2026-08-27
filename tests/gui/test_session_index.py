from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kimix_gui.session_index import (
    RELATIVE_HOURS,
    RELATIVE_JUST_NOW,
    RELATIVE_MINUTES,
    RELATIVE_UNKNOWN,
    RELATIVE_YESTERDAY,
    SIZE_UNIT_KB,
    SIZE_UNIT_MB,
    FileSize,
    RelativeTime,
    delete_sessions,
    file_size,
    list_session_summaries,
    relative_time,
    summaries_from_sessions,
)


def test_summaries_mark_last_session_and_skip_blank_ids() -> None:
    sessions = [
        SimpleNamespace(id="sess-a", title="Fix login redirect", updated_at=100.0),
        SimpleNamespace(id="", title="ignored", updated_at=90.0),
        SimpleNamespace(id="sess-b", title="", updated_at=50.0),
    ]

    summaries = summaries_from_sessions(sessions, last_session_id="sess-a")

    assert [item.id for item in summaries] == ["sess-a", "sess-b"]
    assert summaries[0].is_last is True
    assert summaries[1].title == "Untitled"
    assert summaries[1].is_last is False


def test_relative_time_reports_a_bucket_and_a_count() -> None:
    """Structured buckets, not sentences: the wording lives in ``qt/home_view.py``.

    Stronger than the previous string assertions in two ways: it pins the singular
    boundaries (which decide *which* of the two spelled-out plural forms the view
    picks) and it proves ``count`` is unset for the count-less buckets.
    """

    now = 1_700_000_000.0

    assert relative_time(0, now=now) == RelativeTime(RELATIVE_UNKNOWN)
    assert relative_time(now - 10, now=now) == RelativeTime(RELATIVE_JUST_NOW)
    assert relative_time(now - 60, now=now) == RelativeTime(RELATIVE_MINUTES, 1)
    assert relative_time(now - 120, now=now) == RelativeTime(RELATIVE_MINUTES, 2)
    assert relative_time(now - 4000, now=now) == RelativeTime(RELATIVE_HOURS, 1)
    assert relative_time(now - 8000, now=now) == RelativeTime(RELATIVE_HOURS, 2)
    assert relative_time(now - 100_000, now=now) == RelativeTime(RELATIVE_YESTERDAY)


def test_relative_time_hands_dates_over_as_a_datetime() -> None:
    """Old timestamps carry the instant, so the view can format it with its locale."""

    now = 1_700_000_000.0
    this_year = relative_time(now - 10 * 86400, now=now)
    earlier = relative_time(now - 800 * 86400, now=now)

    assert this_year.unit == "date_this_year"
    assert earlier.unit == "date_earlier"
    for bucket in (this_year, earlier):
        assert bucket.moment is not None
        assert bucket.moment.tzinfo is not None
        assert bucket.count == 0


def test_file_size_reports_a_rounded_value_and_a_unit() -> None:
    """Rounding is arithmetic and stays pure; joining value and unit is view work."""

    assert file_size(0) == FileSize("0", SIZE_UNIT_KB)
    assert file_size(512) == FileSize("0.5", SIZE_UNIT_KB)
    assert file_size(1536) == FileSize("1.5", SIZE_UNIT_KB)
    assert file_size(1024 * 1024) == FileSize("1", SIZE_UNIT_MB)
    assert file_size(int(5.5 * 1024 * 1024)) == FileSize("5.5", SIZE_UNIT_MB)
    # Negative input is clamped, not reported as a negative size.
    assert file_size(-5) == FileSize("0", SIZE_UNIT_KB)


def test_summary_collects_session_directory_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "stored-session"
    nested_dir = session_dir / "subagents" / "worker"
    nested_dir.mkdir(parents=True)
    context_file = session_dir / "context.db"
    context_file.write_bytes(b"x" * 1536)
    (nested_dir / "wire.jsonl").write_bytes(b"y" * 1024)
    session = SimpleNamespace(
        id="stored-session",
        title="Stored session",
        updated_at=100.0,
        dir=session_dir,
        context_file=context_file,
        state=SimpleNamespace(
            archived=True,
            todos=[object(), object()],
            additional_dirs=["D:/shared"],
        ),
    )

    summary = summaries_from_sessions([session])[0]

    assert summary.size_bytes == 2560
    assert summary.file_count == 2
    assert summary.storage_format == "SQLite"
    assert summary.is_archived is True
    assert summary.todo_count == 2
    assert summary.additional_dir_count == 1


async def test_list_session_summaries_uses_injected_lister(tmp_path: Path) -> None:
    sessions = [
        SimpleNamespace(id="older", title="Oldest", updated_at=10.0),
        SimpleNamespace(id="newer", title="Newest", updated_at=20.0),
    ]

    async def fake_list(_work_dir: object) -> list[SimpleNamespace]:
        return sessions

    summaries = await list_session_summaries(
        tmp_path,
        list_sessions=fake_list,
        last_session_id="older",
    )

    assert [item.id for item in summaries] == ["newer", "older"]
    assert summaries[1].is_last is True


async def test_list_session_summaries_empty(tmp_path: Path) -> None:
    async def fake_list(_work_dir: object) -> list[SimpleNamespace]:
        return []

    summaries = await list_session_summaries(
        tmp_path,
        list_sessions=fake_list,
        last_session_id=None,
    )

    assert summaries == []


async def test_delete_sessions_uses_kimi_storage_and_clears_last_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    from kimi_cli.metadata import load_metadata, save_metadata
    from kimi_cli.session import Session as CliSession

    from kimix_gui.kimi_workdir import resolve_kimi_work_dir

    work_dir = tmp_path / "project"
    resolved = resolve_kimi_work_dir(work_dir)
    metadata = load_metadata()
    work_dir_meta = metadata.new_work_dir_meta(resolved)
    work_dir_meta.last_session_id = "session-1"
    save_metadata(metadata)
    deleted: list[str] = []

    class FakeSession:
        async def delete(self) -> None:
            deleted.append("session-1")

    async def find(_work_dir: object, session_id: str) -> FakeSession | None:
        return FakeSession() if session_id == "session-1" else None

    monkeypatch.setattr(CliSession, "find", staticmethod(find))

    await delete_sessions(work_dir, ["session-1", "missing", "session-1"])

    assert deleted == ["session-1"]
    saved_meta = load_metadata().get_work_dir_meta(resolved)
    assert saved_meta is not None
    assert saved_meta.last_session_id is None
