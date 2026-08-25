"""Tests for the edit blackbox recorder."""

from __future__ import annotations

import orjson
import pytest
from kaos.path import KaosPath

from kimi_cli.session import Session
from kimi_cli.tools.file.blackbox import (
    AppliedEditSnapshot,
    EditBlackboxRecorder,
    NoOpBlackboxRecorder,
    create_edit_blackbox_recorder,
)


async def test_recorder_appends_jsonl(session: Session, temp_work_dir: KaosPath) -> None:
    log_path = temp_work_dir / ".kimix_cache" / "edit-blackbox.jsonl"
    recorder = EditBlackboxRecorder(log_path, model="test-model", snapshot_max_bytes=1024 * 1024)
    snapshot = AppliedEditSnapshot(path="foo.py", prev="x = 1\n", next="x =\n")
    await recorder.record(snapshot, variant="replace", arg={"file_path": "foo.py"})

    assert await log_path.exists()
    lines = (await log_path.read_text()).strip().split("\n")
    assert len(lines) == 1
    data = orjson.loads(lines[0])
    assert data["path"] == "foo.py"
    assert data["prev"] == "x = 1\n"
    assert data["next"] == "x =\n"
    assert data["model"] == "test-model"
    assert data["variant"] == "replace"
    assert "ts" in data


async def test_recorder_size_guard_skips_oversized(session: Session, temp_work_dir: KaosPath) -> None:
    log_path = temp_work_dir / ".kimix_cache" / "edit-blackbox.jsonl"
    recorder = EditBlackboxRecorder(log_path, model="test-model", snapshot_max_bytes=5)
    snapshot = AppliedEditSnapshot(path="foo.py", prev="x = 1\n", next="x =\n")
    await recorder.record(snapshot, variant="replace", arg={})
    assert not await log_path.exists() or (await log_path.read_text()) == ""


async def test_create_recorder_disabled(session: Session) -> None:
    session.custom_config = {
        "config_json": {"edit": {"blackbox": {"enabled": False}}}
    }
    recorder = create_edit_blackbox_recorder(session, "replace", {})
    assert isinstance(recorder, NoOpBlackboxRecorder)


async def test_noop_recorder_never_raises() -> None:
    recorder = NoOpBlackboxRecorder()
    snapshot = AppliedEditSnapshot(path="foo.py", prev="x = 1\n", next="x =\n")
    await recorder.record(snapshot, variant="replace", arg={})
