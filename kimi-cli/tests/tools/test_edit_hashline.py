"""Tests for hashline mode of the edit tool."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file import EditFile, HashRead
from kimi_cli.tools.file.edit.params import EditParams
from kimi_cli.tools.file.hash_line import HashReadParams, compute_line_hash


def _line1_hash(content: str) -> str:
    line = content.splitlines()[0] if content.splitlines() else ""
    return compute_line_hash(1, line, None)


@pytest.fixture
def hashline_tool(edit_file_tool: EditFile) -> EditFile:
    return edit_file_tool


async def test_hashline_replace_lines(hashline_tool, temp_work_dir):
    file_path = temp_work_dir / "sample.txt"
    content = "line1\nline2\nline3\n"
    await file_path.write_text(content)

    input_text = (
        f"[{file_path}#{_line1_hash(content)}]\n"
        "PUT 2.=3:\n"
        "+NEW2\n"
        "+NEW3\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await file_path.read_text() == "line1\nNEW2\nNEW3\n"


async def test_hashline_insert_after(hashline_tool, temp_work_dir):
    file_path = temp_work_dir / "sample.txt"
    content = "line1\nline2\n"
    await file_path.write_text(content)

    input_text = (
        f"[{file_path}#{_line1_hash(content)}]\n"
        "PUT >2:\n"
        "+line3\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await file_path.read_text() == "line1\nline2\nline3\n"


async def test_hashline_delete_file(hashline_tool, temp_work_dir):
    file_path = temp_work_dir / "remove.txt"
    await file_path.write_text("bye")

    input_text = (
        f"[{file_path}#{_line1_hash('bye')}]\n"
        "REM\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert not await file_path.exists()


async def test_hashline_stale_tag(hashline_tool, temp_work_dir):
    file_path = temp_work_dir / "stale.txt"
    await file_path.write_text("line1\nline2\n")

    input_text = (
        f"[{file_path}#ZZ]\n"
        "PUT 1.=1:\n"
        "+changed\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert result.is_error
    assert "Hashline mismatch" in result.brief or "changed since" in result.message


async def test_hashline_move_file(hashline_tool, temp_work_dir):
    source = temp_work_dir / "source.txt"
    dest = temp_work_dir / "dest.txt"
    content = "line1\nline2\n"
    await source.write_text(content)

    input_text = (
        f"[{source}#{_line1_hash(content)}]\n"
        "PUT 1.=2:\n"
        "+NEW1\n"
        "+NEW2\n"
        f"MV {dest}\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert not await source.exists()
    assert await dest.read_text() == "NEW1\nNEW2\n"


async def test_hashline_named_register_across_sections(hashline_tool, temp_work_dir):
    source = temp_work_dir / "cut.txt"
    dest = temp_work_dir / "paste.txt"
    src_content = "alpha\nbeta\ngamma\n"
    dst_content = "one\ntwo\n"
    await source.write_text(src_content)
    await dest.write_text(dst_content)

    input_text = (
        f"[{source}#{_line1_hash(src_content)}]\n"
        "CUT 1.=2 @chunk\n"
        f"[{dest}#{_line1_hash(dst_content)}]\n"
        "PUT >1 @chunk\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await source.read_text() == "gamma\n"
    assert await dest.read_text() == "one\nalpha\nbeta\ntwo\n"


async def test_hashline_auto_detected_from_input(hashline_tool, temp_work_dir):
    file_path = temp_work_dir / "auto.txt"
    content = "line1\nline2\n"
    await file_path.write_text(content)

    result = await hashline_tool(
        EditParams(
            input=(
                f"[{file_path}#{_line1_hash(content)}]\n"
                "PUT 2.=2:\n"
                "+NEW2\n"
            )
        )
    )
    assert not result.is_error
    assert await file_path.read_text() == "line1\nNEW2\n"


async def test_hashline_recovers_from_snapshot(
    hashline_tool, hash_line_tool: HashRead, temp_work_dir
):
    file_path = temp_work_dir / "snap.txt"
    content = "line1\nline2\n"
    await file_path.write_text(content)

    # Read records a snapshot with line1 hash.
    read_result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not read_result.is_error

    # Change the file externally so the current line1 hash differs.
    await file_path.write_text("changed\nline2\n")

    # Edit using the original line1 hash should recover from the snapshot.
    input_text = (
        f"[{file_path}#{_line1_hash(content)}]\n"
        "PUT 1.=1:\n"
        "+RECOVERED\n"
    )
    result = await hashline_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await file_path.read_text() == "RECOVERED\nline2\n"
