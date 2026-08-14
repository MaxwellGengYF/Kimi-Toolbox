from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest

from kimi_cli.soul.compaction_ledger import CompactionLedger, CompactionRecord


def _record(compaction_id: str = "a" * 32, **overrides) -> CompactionRecord:
    base = dict(
        compaction_id=compaction_id,
        trigger="auto",
        started_at=1234.0,
        shadowed_range=(0, 5),
        shadowed_tokens=100,
        summary_tokens=40,
        preserved_tokens=60,
        shrank=True,
    )
    base.update(overrides)
    return CompactionRecord(**base)


def test_record_start_end_round_trip(tmp_path: Path) -> None:
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    ledger.record_start(_record())

    assert ledger.latest() == _record()

    ledger.record_end("a" * 32, summary_tokens=25, shrank=True)
    final = ledger.latest()
    assert final is not None
    assert final.error is None
    assert final.summary_tokens == 25
    assert final.shrank is True


def test_latest_returns_most_recent_record(tmp_path: Path) -> None:
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    ledger.record_start(_record(compaction_id="1" * 32, shadowed_tokens=10))
    ledger.record_start(_record(compaction_id="2" * 32, shadowed_tokens=20))
    ledger.record_start(_record(compaction_id="3" * 32, shadowed_tokens=30))

    latest = ledger.latest()
    assert latest is not None
    assert latest.compaction_id == "3" * 32
    assert latest.shadowed_tokens == 30


def test_record_end_sets_error(tmp_path: Path) -> None:
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    ledger.record_start(_record(shrank=False))
    ledger.record_end("a" * 32, error="boom")

    final = ledger.latest()
    assert final is not None
    assert final.error == "boom"
    assert final.shrank is False


def test_record_end_rewrites_in_place_one_line_per_transaction(tmp_path: Path) -> None:
    """Design check: the file stays one line per transaction (rewrite-in-place)."""
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    ledger.record_start(_record(compaction_id="1" * 32))
    ledger.record_start(_record(compaction_id="2" * 32))
    ledger.record_end("1" * 32, error="failed")

    lines = (tmp_path / "ledger.jsonl").read_bytes().strip().splitlines()
    assert len(lines) == 2
    first = orjson.loads(lines[0])
    assert first["compaction_id"] == "1" * 32
    assert first["error"] == "failed"
    second = orjson.loads(lines[1])
    assert second["compaction_id"] == "2" * 32
    assert "error" not in second or second["error"] is None


def test_jsonl_parseable_with_orjson(tmp_path: Path) -> None:
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    ledger.record_start(_record())
    ledger.record_start(_record(compaction_id="b" * 32, trigger="manual"))

    raw = (tmp_path / "ledger.jsonl").read_bytes()
    records = [orjson.loads(line) for line in raw.strip().splitlines()]
    assert [r["compaction_id"] for r in records] == ["a" * 32, "b" * 32]
    assert records[1]["trigger"] == "manual"
    assert records[0]["shadowed_range"] == [0, 5]
    assert records[0]["shrank"] is True


def test_malformed_line_is_skipped_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"{not-json}\n")
    ledger = CompactionLedger(path)
    ledger.record_start(_record())

    # the malformed line is skipped; the valid record is still read
    assert ledger.latest() == _record()


def test_for_session_disabled_is_noop_no_file_created(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    ledger = CompactionLedger.for_session(session_dir, enabled=False)

    assert ledger._path is None  # noqa: SLF001
    ledger.record_start(_record())
    ledger.record_end("a" * 32)
    assert ledger.latest() is None
    assert not (session_dir / ".kimix_cache" / "compaction_ledger.jsonl").exists()


def test_for_session_enabled_creates_cache_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    ledger = CompactionLedger.for_session(session_dir, enabled=True)

    expected = session_dir / ".kimix_cache" / "compaction_ledger.jsonl"
    assert ledger._path == expected  # noqa: SLF001
    assert expected.parent.is_dir()

    ledger.record_start(_record())
    assert expected.exists()


def test_for_session_enabled_with_broken_dir_degrades_to_noop(tmp_path: Path) -> None:
    # parent of .kimix_cache is a *file* → mkdir must fail → no-op ledger
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    session_dir = blocker / "nested"  # path whose ancestor is a file

    ledger = CompactionLedger.for_session(session_dir, enabled=True)
    assert ledger._path is None  # noqa: SLF001
    ledger.record_start(_record())  # must not raise
    assert ledger.latest() is None


def test_failure_isolation_broken_path_does_not_raise_out_of_compact(tmp_path: Path) -> None:
    """A ledger whose path cannot be written must never raise out of the
    transactional envelope — record_start/record_end degrade to warnings."""
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("file")
    # path "inside" a file: parent is a regular file → open/append fails
    ledger = CompactionLedger(blocker / "ledger.jsonl")

    # simulate the compact() envelope exactly
    async def fake_compact() -> None:
        ledger.record_start(_record())
        try:
            raise RuntimeError("llm failure")
        except Exception as exc:
            ledger.record_end("a" * 32, error=str(exc))
            raise

    with pytest.raises(RuntimeError, match="llm failure"):
        asyncio.run(fake_compact())

    # no ledger file/artifact created
    assert not (blocker / "ledger.jsonl").exists()


def test_latest_empty_file_returns_none(tmp_path: Path) -> None:
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    assert ledger.latest() is None
    ledger.record_end("a" * 32)  # no start record → no-op, no raise
    assert ledger.latest() is None


def test_record_end_unknown_id_leaves_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = CompactionLedger(path)
    ledger.record_start(_record(compaction_id="1" * 32))
    before = path.read_bytes()

    ledger.record_end("deadbeef" * 8, error="orphan")
    assert path.read_bytes() == before
    assert ledger.latest() == _record(compaction_id="1" * 32)
