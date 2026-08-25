"""Integration tests for edit/write parse-failure detection and auto-repair."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from kaos.path import KaosPath

from kimi_cli.session import Session
from kimi_cli.tools.file.replace import Edit, EditFile, Params
from kimi_cli.tools.file.write import Params as WriteParams, WriteFile


async def test_write_introduced_parse_failure_records_blackbox_and_warns(
    write_file_tool: WriteFile, temp_work_dir: KaosPath, session: Session
) -> None:
    file_path = temp_work_dir / "written.py"

    result = await write_file_tool(
        WriteParams(path=str(file_path), content="x =\n")
    )

    assert not result.is_error
    assert "no longer parses" in result.message
    assert await file_path.read_text() == "x =\n"

    log_path = Path(str(temp_work_dir)) / ".kimix_cache" / "edit-blackbox.jsonl"
    assert log_path.exists()


async def test_edit_introduced_parse_failure_records_blackbox_and_warns(
    edit_file_tool: EditFile, temp_work_dir: KaosPath, session: Session
) -> None:
    file_path = temp_work_dir / "broken.py"
    await file_path.write_text("x = 1\n")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="x = 1", new="x ="))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert "no longer parses" in result.message
    assert await file_path.read_text() == "x =\n"

    log_path = Path(str(temp_work_dir)) / ".kimix_cache" / "edit-blackbox.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = orjson.loads(lines[0])
    assert data["path"] == str(file_path)
    assert data["prev"] == "x = 1\n"
    assert data["next"] == "x =\n"


async def test_edit_indentation_drift_is_auto_repaired(
    edit_file_tool: EditFile, temp_work_dir: KaosPath, session: Session
) -> None:
    file_path = temp_work_dir / "indent.py"
    await file_path.write_text("def foo():\n    if True:\n        pass\n")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="        pass", new="    pass"),
        )
    )

    assert not result.is_error
    assert "automatic syntax repair" in result.message
    assert "def foo():" in await file_path.read_text()
    # Ensure the repaired file parses.
    import ast
    ast.parse(await file_path.read_text())


async def test_edit_parse_failure_disabled_auto_repair_warns_only(
    edit_file_tool: EditFile, temp_work_dir: KaosPath, session: Session
) -> None:
    session.custom_config = {
        "config_json": {"edit": {"autoRepair": {"enabled": False}}}
    }
    file_path = temp_work_dir / "norepair.py"
    await file_path.write_text("x = 1\n")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="x = 1", new="x ="))
    )

    assert not result.is_error
    assert "no longer parses" in result.message
    assert "automatic syntax repair" not in result.message


async def test_edit_blackbox_disabled_does_not_write_log(
    edit_file_tool: EditFile, temp_work_dir: KaosPath, session: Session
) -> None:
    session.custom_config = {
        "config_json": {"edit": {"blackbox": {"enabled": False}}}
    }
    file_path = temp_work_dir / "nolog.py"
    await file_path.write_text("x = 1\n")

    await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="x = 1", new="x ="))
    )

    log_path = Path(str(temp_work_dir)) / ".kimix_cache" / "edit-blackbox.jsonl"
    assert not log_path.exists()
