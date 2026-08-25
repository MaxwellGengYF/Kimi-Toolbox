"""Write-tool conflict resolution + guard integration tests (plan 24 M3).

Marker strings are assembled at runtime so this file itself carries no
literal conflict markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.tools.file.conflict_detect import get_conflict_history
from kimi_cli.tools.file.read import Params as ReadParams
from kimi_cli.tools.file.write import Params as WriteParams

OURS_M = "<" * 7
SEP_M = "=" * 7
THEIRS_M = ">" * 7


def block(ours: str, theirs: str, ours_label: str = "HEAD",
          theirs_label: str = "feature/x") -> str:
    return "\n".join(
        [f"{OURS_M} {ours_label}", ours, SEP_M, theirs, f"{THEIRS_M} {theirs_label}"]
    )


def conflict_text(before="top", after="bottom"):
    return f"{before}\n{block('ours side', 'theirs side')}\n{after}\n"


@pytest.fixture
def wdir(temp_work_dir):
    return Path(str(temp_work_dir))


async def _register(read_file_tool, path: Path):
    await read_file_tool(ReadParams(path=str(path)))


# ---------------------------------------------------------------------------
# conflict://<N> single resolve


async def test_resolve_single_conflict_ours(
    write_file_tool, read_file_tool, wdir, session
):
    p = wdir / "single.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)

    result = await write_file_tool(
        WriteParams(path="conflict://1", content="@ours")
    )
    assert not result.is_error
    assert "Resolved conflict #1" in result.message
    assert p.read_text(encoding="utf-8") == "top\nours side\nbottom\n"
    assert get_conflict_history(session).entries() == []


async def test_resolve_single_conflict_theirs_and_literal(
    write_file_tool, read_file_tool, wdir
):
    p = wdir / "two.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)

    result = await write_file_tool(
        WriteParams(path="conflict://1", content="@theirs")
    )
    assert not result.is_error
    assert p.read_text(encoding="utf-8") == "top\ntheirs side\nbottom\n"


async def test_resolve_single_conflict_literal_replacement(
    write_file_tool, read_file_tool, wdir
):
    p = wdir / "literal.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)

    result = await write_file_tool(
        WriteParams(path="conflict://1", content="manually merged\nsecond line")
    )
    assert not result.is_error
    assert p.read_text(encoding="utf-8") == "top\nmanually merged\nsecond line\nbottom\n"


async def test_resolve_unknown_id(write_file_tool, wdir):
    result = await write_file_tool(
        WriteParams(path="conflict://42", content="@ours")
    )
    assert result.is_error
    assert "Conflict #42 not found" in result.message


async def test_resolve_scope_is_read_only(write_file_tool, read_file_tool, wdir):
    p = wdir / "scope.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(path="conflict://1/ours", content="x")
    )
    assert result.is_error
    assert "read-only" in result.message


async def test_resolve_shifted_block(write_file_tool, read_file_tool, wdir):
    p = wdir / "shifted.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    # Out-of-band edit shifts the block down by two lines.
    p.write_text("new1\nnew2\n" + conflict_text(), encoding="utf-8")

    result = await write_file_tool(
        WriteParams(path="conflict://1", content="@ours")
    )
    assert not result.is_error
    assert p.read_text(encoding="utf-8") == "new1\nnew2\ntop\nours side\nbottom\n"


async def test_resolve_altered_block_fails(write_file_tool, read_file_tool, wdir):
    p = wdir / "altered.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    # Out-of-band edit alters the ours side -> recorded block no longer matches.
    p.write_bytes(
        conflict_text().replace("ours side", "edited ours").encode("utf-8")
    )
    result = await write_file_tool(
        WriteParams(path="conflict://1", content="@theirs")
    )
    assert result.is_error
    assert "no longer matches" in result.message
    # File untouched.
    assert "edited ours" in p.read_text(encoding="utf-8")


async def test_resolve_base_token_two_way_error(
    write_file_tool, read_file_tool, wdir
):
    p = wdir / "base2way.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(path="conflict://1", content="@base")
    )
    assert result.is_error
    assert "2-way conflict" in result.message


async def test_resolve_echo_trim_note(write_file_tool, read_file_tool, wdir):
    # Replacement echoes adjacent context -> dropped with a note.
    text = "top\n" + block("ours side", "theirs side") + "\nbottom\nend\n"
    p = wdir / "echo.py"
    p.write_text(text, encoding="utf-8")
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(
            path="conflict://1", content="top\nours side\nbottom\nend"
        )
    )
    assert not result.is_error
    assert "boundary-echo" in result.message
    assert p.read_text(encoding="utf-8") == "top\nours side\nbottom\nend\n"


# ---------------------------------------------------------------------------
# conflict://* bulk resolve


async def test_bulk_resolve_all_with_shared_content(
    write_file_tool, read_file_tool, wdir, session
):
    p = wdir / "bulk.py"
    p.write_text(
        block("first ours", "first theirs") + "\nmid\n" + block("second ours", "second theirs") + "\n",
        encoding="utf-8",
    )
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(path="conflict://*", content="@ours")
    )
    assert not result.is_error
    assert "Resolved 2 conflict(s)" in result.message
    assert p.read_text(encoding="utf-8") == "first ours\nmid\nsecond ours\n"
    assert get_conflict_history(session).entries() == []


async def test_bulk_resolve_per_id_directives(
    write_file_tool, read_file_tool, wdir
):
    p = wdir / "directives.py"
    p.write_text(
        block("a ours", "a theirs") + "\nmid\n" + block("b ours", "b theirs") + "\n",
        encoding="utf-8",
    )
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(path="conflict://*", content="1: @ours\n2: @theirs\n")
    )
    assert not result.is_error
    assert p.read_text(encoding="utf-8") == "a ours\nmid\nb theirs\n"


async def test_bulk_resolve_unknown_directive_id(
    write_file_tool, read_file_tool, wdir
):
    p = wdir / "unknown.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    result = await write_file_tool(
        WriteParams(path="conflict://*", content="1: @ours\n9: @theirs\n")
    )
    assert result.is_error
    assert "#9" in result.message


async def test_bulk_resolve_empty_history(write_file_tool, wdir):
    result = await write_file_tool(
        WriteParams(path="conflict://*", content="@ours")
    )
    assert result.is_error
    assert "nothing to resolve" in result.message


# ---------------------------------------------------------------------------
# Regular-write guard


async def test_overwrite_with_new_markers_refused(write_file_tool, wdir):
    p = wdir / "clean_target.py"
    p.write_text("existing\n", encoding="utf-8")
    result = await write_file_tool(
        WriteParams(path=str(p), content=block("a", "b") + "\n")
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message
    assert p.read_text(encoding="utf-8") == "existing\n"


async def test_overwrite_with_new_markers_allow_conflicts(write_file_tool, wdir):
    p = wdir / "allow_target.py"
    p.write_text("existing\n", encoding="utf-8")
    result = await write_file_tool(
        WriteParams(path=str(p), content=block("a", "b") + "\n", allow_conflicts=True)
    )
    assert not result.is_error
    assert "Warning" in result.message
    assert OURS_M in p.read_text(encoding="utf-8")


async def test_clean_overwrite_of_conflicted_file_invalidates_history(
    write_file_tool, read_file_tool, wdir, session
):
    p = wdir / "resolvebywrite.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    await _register(read_file_tool, p)
    assert len(get_conflict_history(session).entries()) == 1
    result = await write_file_tool(
        WriteParams(path=str(p), content="fully resolved\n")
    )
    assert not result.is_error
    assert get_conflict_history(session).entries() == []
    assert p.read_text(encoding="utf-8") == "fully resolved\n"


async def test_new_file_with_markers_refused(write_file_tool, wdir):
    p = wdir / "newfile.py"
    result = await write_file_tool(
        WriteParams(path=str(p), content=block("a", "b") + "\n")
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message
    assert not p.exists()


async def test_append_with_markers_refused(write_file_tool, wdir):
    p = wdir / "appendtarget.py"
    p.write_text("head\n", encoding="utf-8")
    result = await write_file_tool(
        WriteParams(path=str(p), content=block("a", "b") + "\n", mode="append")
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message


async def test_append_to_dangling_opener_refused(write_file_tool, wdir):
    p = wdir / "dangling.py"
    p.write_text("head\n" + OURS_M + " HEAD\nours body\n", encoding="utf-8")
    result = await write_file_tool(
        WriteParams(path=str(p), content="more\n", mode="append")
    )
    assert result.is_error
    assert "Conflict markers detected" in result.message


async def test_append_clean_to_conflicted_file_warns(write_file_tool, wdir):
    p = wdir / "appendconflicted.py"
    p.write_bytes(conflict_text().encode("utf-8"))
    result = await write_file_tool(
        WriteParams(path=str(p), content="appended\n", mode="append")
    )
    assert not result.is_error
    assert "Note" in result.message
    assert "unresolved conflict marker block" in result.message
