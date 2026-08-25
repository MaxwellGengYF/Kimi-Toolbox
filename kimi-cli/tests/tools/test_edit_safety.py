"""Tests for EditParseGuard orchestrator."""

from __future__ import annotations

import pytest
from kaos.path import KaosPath

from kimi_cli.session import Session
from kimi_cli.tools.file.edit_safety import EditParseGuard


async def test_observe_stashes_parse_regression_and_returns_warning(session: Session) -> None:
    guard = EditParseGuard(session, variant="replace", arg={})
    await guard.observe_applied("C:\\foo.py", "x = 1\n", "x =\n")
    notes = await guard.finish()
    assert len(notes) == 1
    assert "no longer parses" in notes[0]
    assert "foo.py" in notes[0]


async def test_observe_drops_regression_when_later_parse_restored(session: Session) -> None:
    guard = EditParseGuard(session, variant="replace", arg={})
    await guard.observe_applied("C:\\foo.py", "x = 1\n", "x =\n")
    await guard.observe_applied("C:\\foo.py", "x = 1\n", "x = 2\n")
    notes = await guard.finish()
    assert notes == []


async def test_observe_records_no_regression_for_unknown_language(session: Session) -> None:
    guard = EditParseGuard(session, variant="replace", arg={})
    await guard.observe_applied("C:\\foo.txt", "x = 1\n", "x =\n")
    notes = await guard.finish()
    assert notes == []


async def test_finish_never_raises(session: Session) -> None:
    guard = EditParseGuard(session, variant="replace", arg={})
    # Force a malformed snapshot to exercise failure isolation.
    guard._parse_failures["C:\\foo.py"] = object()  # type: ignore[assignment]
    notes = await guard.finish()
    assert len(notes) == 1
    assert "no longer parses" in notes[0]


async def test_observe_records_snapshot_for_covered_language(
    session: Session, temp_work_dir: KaosPath
) -> None:
    file_path = temp_work_dir / "guard.py"
    await file_path.write_text("x = 1\n")
    guard = EditParseGuard(session, variant="replace", arg={})
    await guard.observe_applied(str(file_path), "x = 1\n", "x = 2\n")
    notes = await guard.finish()
    assert notes == []
    from kimi_cli.tools.file.snapshot_store import get_edit_snapshot_store

    assert get_edit_snapshot_store(session).lookup(str(file_path)) is not None
