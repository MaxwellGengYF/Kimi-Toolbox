"""Plan 24 extensions for the edit conflict guard (registration + bookkeeping).

Existing plan-04 tests live above; marker strings are assembled at runtime.
"""

from __future__ import annotations

import pytest
from kaos.path import KaosPath

from kimi_cli.tools.file import EditFile
from kimi_cli.tools.file.conflict_detect import get_conflict_history
from kimi_cli.tools.file.edit.params import EditParams, ReplaceEditItem

OURS_M = "<" * 7
SEP_M = "=" * 7
THEIRS_M = ">" * 7


def _block(ours: str, theirs: str) -> str:
    return f"{OURS_M} HEAD\n{ours}\n{SEP_M}\n{theirs}\n{THEIRS_M} branch"


def _conflicted(before: str = "before", after: str = "after") -> str:
    return f"{before}\n{_block('ours', 'theirs')}\n{after}\n"


@pytest.fixture
def conflict_edit_tool(edit_file_tool: EditFile) -> EditFile:
    return edit_file_tool


async def test_edit_registers_history_blocks(
    conflict_edit_tool, temp_work_dir, session
):
    file_path = temp_work_dir / "reg.txt"
    await file_path.write_bytes(_conflicted().encode("utf-8"))

    await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="before", new="start")],
            allow_conflicts=True,
        )
    )
    entries = get_conflict_history(session).entries()
    assert len(entries) == 1
    assert entries[0].start_line == 2


async def test_edit_resolving_markers_invalidates_history(
    conflict_edit_tool, temp_work_dir, session
):
    file_path = temp_work_dir / "resolve.txt"
    await file_path.write_bytes(_conflicted().encode("utf-8"))

    # Replace the whole marker block with resolved text. The pre-apply guard
    # still requires allow_conflicts=True because the file itself holds
    # markers; the edit resolves them and history is invalidated afterwards.
    result = await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old=_block("ours", "theirs"), new="resolved")],
            allow_conflicts=True,
        )
    )
    assert not result.is_error
    assert await file_path.read_bytes() == b"before\nresolved\nafter\n"
    assert get_conflict_history(session).entries() == []


async def test_edit_introducing_markers_warns(
    conflict_edit_tool, temp_work_dir, session
):
    file_path = temp_work_dir / "introduce.txt"
    await file_path.write_bytes(b"clean\nmarker-holder\nend\n")

    result = await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="marker-holder", new=_block("a", "b"))],
            allow_conflicts=True,
        )
    )
    assert not result.is_error
    assert "introduced" in result.message
    assert "conflict marker block" in result.message


async def test_edit_keeps_ids_stable_with_read_registration(
    conflict_edit_tool, temp_work_dir, session
):
    file_path = temp_work_dir / "stable_ids.txt"
    await file_path.write_bytes(_conflicted().encode("utf-8"))

    # First edit registers block #1.
    await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="before", new="start")],
            allow_conflicts=True,
        )
    )
    # Second edit on the same file must reuse id 1 (same path + start_line).
    await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="after", new="tail")],
            allow_conflicts=True,
        )
    )
    ids = [e.id for e in get_conflict_history(session).entries()]
    assert ids == [1]


async def test_edit_outside_region_still_refused_without_flag(
    conflict_edit_tool, temp_work_dir
):
    """Deliberate design decision: the plan-04 whole-file guard is stricter
    than plan 24's span-intersection guard.  An edit OUTSIDE the conflict
    region is still refused without allow_conflicts — this test encodes that
    choice so it is not silently loosened later."""
    file_path = temp_work_dir / "outside.txt"
    await file_path.write_bytes(_conflicted().encode("utf-8"))
    result = await conflict_edit_tool(
        EditParams(
            path=str(file_path),
            mode="replace",
            edits=[ReplaceEditItem(old="before", new="changed")],
        )
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message
    # the file is untouched
    assert await file_path.read_bytes() == _conflicted().encode("utf-8")
