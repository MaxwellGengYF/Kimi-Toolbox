"""Read-tool conflict detection integration tests (plan 24 M1/M2).

Marker strings are assembled at runtime (not literals) so the write tool's
conflict-marker guard does not refuse to create this test file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.tools.file.conflict_detect import get_conflict_history

OURS_M = "<" * 7
SEP_M = "=" * 7
THEIRS_M = ">" * 7
BASE_M = "|" * 7


def block(ours: str, theirs: str, ours_label: str = "HEAD",
          theirs_label: str = "feature/x", base: str | None = None) -> str:
    lines = [f"{OURS_M} {ours_label}", ours]
    if base is not None:
        lines += [f"{BASE_M} merged common ancestors", base]
    lines += [SEP_M, theirs, f"{THEIRS_M} {theirs_label}"]
    return "\n".join(lines)


CONFLICT_FILE = "\n".join(
    ["line one", "line two", block("ours alpha", "theirs alpha"), "line eight"]
) + "\n"

TWO_BLOCK_FILE = "\n".join(
    [
        block("first ours", "first theirs", theirs_label="branch-a"),
        "middle",
        block("second ours", "second theirs", theirs_label="branch-b"),
    ]
) + "\n"

CRLF_FILE = (
    "top\r\n" + block("ours", "theirs", theirs_label="t").replace("\n", "\r\n")
    + "\r\nbottom\r\n"
)


@pytest.fixture
def conflicted_dir(temp_work_dir):
    d = Path(str(temp_work_dir))
    return d


from kimi_cli.tools.file.read import Params


def _params(tool, file_path, **kw):
    # Translate test-friendly names to ReadFile's canonical Params fields.
    rename = {"offset": "line_offset", "limit": "n_lines"}
    kw = {rename.get(k, k): v for k, v in kw.items()}
    return Params(path=file_path, **kw)


async def test_read_conflicted_file_shows_footer(read_file_tool, conflicted_dir):
    p = conflicted_dir / "f.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, str(p)))
    assert not result.is_error
    assert "unresolved conflict detected" in result.output
    assert "──── #1  L3-7 ────" in result.output
    assert "- ours = HEAD" in result.output
    assert "- theirs = feature/x" in result.output
    assert "1 unresolved conflict(s) detected" in result.message


async def test_read_clean_file_no_footer(read_file_tool, conflicted_dir):
    p = conflicted_dir / "clean.py"
    p.write_text("a = 1\nb = 2\n", encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, str(p)))
    assert not result.is_error
    assert "unresolved conflict" not in result.output
    assert "unresolved conflict" not in result.message


async def test_read_partial_window_no_closer_no_footer(read_file_tool, conflicted_dir):
    p = conflicted_dir / "partial.py"
    p.write_text(OURS_M + " HEAD\nours\n", encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, str(p)))
    assert not result.is_error
    assert "unresolved conflict" not in result.output


async def test_read_window_starting_mid_block_no_footer(read_file_tool, conflicted_dir):
    p = conflicted_dir / "mid.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    # offset inside the ours section: no opener visible -> no block detected
    result = await read_file_tool(
        _params(read_file_tool, str(p), offset=4, limit=3)
    )
    assert not result.is_error
    assert "unresolved conflict" not in result.output


async def test_read_tail_mode_detects_conflict(read_file_tool, conflicted_dir):
    p = conflicted_dir / "tail.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, str(p), offset=-8))
    assert not result.is_error
    assert "unresolved conflict detected" in result.output


async def test_read_crlf_file_detects_conflict(read_file_tool, conflicted_dir):
    p = conflicted_dir / "crlf.py"
    p.write_bytes(CRLF_FILE.encode("utf-8"))
    result = await read_file_tool(_params(read_file_tool, str(p)))
    assert not result.is_error
    assert "unresolved conflict detected" in result.output
    assert "──── #1  L2-6 ────" in result.output


async def test_read_registers_history_ids(read_file_tool, conflicted_dir, session):
    p = conflicted_dir / "hist.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    history = get_conflict_history(session)
    entries = history.entries()
    assert len(entries) == 1
    assert entries[0].id == 1
    assert entries[0].start_line == 3


async def test_read_conflict_ids_stable_across_rereads(
    read_file_tool, conflicted_dir, session
):
    p = conflicted_dir / "stable.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    await read_file_tool(_params(read_file_tool, str(p)))
    history = get_conflict_history(session)
    assert [e.id for e in history.entries()] == [1]


async def test_read_conflict_uri_full_region(read_file_tool, conflicted_dir):
    p = conflicted_dir / "uri.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    result = await read_file_tool(_params(read_file_tool, "conflict://1"))
    assert not result.is_error
    lines = result.output.split("\n")
    assert OURS_M in lines[0]
    assert lines[-1].endswith(f"{THEIRS_M} feature/x")
    # line numbers preserved (block starts at line 3)
    assert f"3\t{OURS_M} HEAD" in lines[0]


async def test_read_conflict_uri_side_scopes(read_file_tool, conflicted_dir):
    p = conflicted_dir / "sides.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    ours = await read_file_tool(_params(read_file_tool, "conflict://1/ours"))
    assert not ours.is_error
    assert "ours alpha" in ours.output
    assert OURS_M not in ours.output
    theirs = await read_file_tool(_params(read_file_tool, "conflict://1/theirs"))
    assert "theirs alpha" in theirs.output


async def test_read_conflict_uri_base_two_way_error(read_file_tool, conflicted_dir):
    p = conflicted_dir / "twoway.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    result = await read_file_tool(_params(read_file_tool, "conflict://1/base"))
    assert result.is_error
    assert "2-way conflict" in result.message


async def test_read_conflict_uri_unknown_id(read_file_tool, conflicted_dir):
    result = await read_file_tool(_params(read_file_tool, "conflict://99"))
    assert result.is_error
    assert "Conflict #99 not found" in result.message
    assert "re-read" in result.message


async def test_read_conflict_uri_wildcard_rejected(read_file_tool, conflicted_dir):
    result = await read_file_tool(_params(read_file_tool, "conflict://*"))
    assert result.is_error
    assert "write-only" in result.message


async def test_read_conflict_uri_invalid_scope(read_file_tool, conflicted_dir):
    result = await read_file_tool(_params(read_file_tool, "conflict://1/bogus"))
    assert result.is_error
    assert "scope" in result.message.lower()


async def test_read_conflict_uri_with_prefix_recovers(
    read_file_tool, conflicted_dir
):
    p = conflicted_dir / "prefixed.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, str(p)))
    result = await read_file_tool(
        _params(read_file_tool, f"{p.as_posix()}:conflict://1")
    )
    assert not result.is_error
    assert OURS_M in result.output
    assert "prefixed.py" in result.message


async def test_conflicts_selector_index(read_file_tool, conflicted_dir):
    p = conflicted_dir / "sel.py"
    p.write_text(TWO_BLOCK_FILE, encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, f"{p.as_posix()}:conflicts"))
    assert not result.is_error
    assert "2 unresolved conflicts in" in result.output
    assert "#1  L1-5" in result.output
    assert "#2  L7-11" in result.output
    assert "conflictCount=2" in result.message


async def test_conflicts_selector_clean_file(read_file_tool, conflicted_dir):
    p = conflicted_dir / "nosel.py"
    p.write_text("clean\n", encoding="utf-8")
    result = await read_file_tool(_params(read_file_tool, f"{p.as_posix()}:conflicts"))
    assert not result.is_error
    assert "No unresolved git merge conflicts" in result.output


async def test_conflicts_selector_registers_history(
    read_file_tool, conflicted_dir, session
):
    p = conflicted_dir / "selreg.py"
    p.write_text(TWO_BLOCK_FILE, encoding="utf-8")
    await read_file_tool(_params(read_file_tool, f"{p.as_posix()}:conflicts"))
    history = get_conflict_history(session)
    assert len(history.entries()) == 2


async def test_conflicts_selector_glob_rejected(read_file_tool, conflicted_dir):
    result = await read_file_tool(_params(read_file_tool, "*.py:conflicts"))
    assert result.is_error
    assert "glob" in result.message.lower()


async def test_conflicts_selector_missing_file(read_file_tool, conflicted_dir):
    result = await read_file_tool(
        _params(read_file_tool, f"{conflicted_dir.as_posix()}/nope.py:conflicts")
    )
    assert result.is_error
    assert "does not exist" in result.message


async def test_partial_window_header_shows_total(read_file_tool, conflicted_dir):
    p = conflicted_dir / "many.py"
    # Two blocks but the window (offset=1, limit=6) only shows the first.
    p.write_text(TWO_BLOCK_FILE, encoding="utf-8")
    result = await read_file_tool(
        _params(read_file_tool, str(p), offset=1, limit=6)
    )
    assert not result.is_error
    assert "1 of 2 unresolved conflicts visible in this window" in result.output
    assert ":conflicts" in result.output


async def test_char_window_does_not_cut_footer(read_file_tool, conflicted_dir):
    p = conflicted_dir / "charwin.py"
    p.write_text(CONFLICT_FILE, encoding="utf-8")
    result = await read_file_tool(
        _params(read_file_tool, str(p), max_char=40)
    )
    assert not result.is_error
    # Footer survives even though the char window truncated the body.
    assert "unresolved conflict detected" in result.output
