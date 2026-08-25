"""Tests for sloppy mode of the edit tool."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file import EditFile
from kimi_cli.tools.file.edit.params import EditParams


@pytest.fixture
def sloppy_tool(edit_file_tool: EditFile) -> EditFile:
    return edit_file_tool


async def test_sloppy_inline_selection(sloppy_tool, temp_work_dir):
    file_path = temp_work_dir / "config.ts"
    await file_path.write_text("const timeout = 1000;\nconst retries = 3;\n")

    input_text = (
        f"§{file_path}\n"
        "const timeout = ⟪1000│5000⟫;\n"
        "const retries = ⟪3│5⟫;\n"
    )
    result = await sloppy_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await file_path.read_text() == "const timeout = 5000;\nconst retries = 5;\n"


async def test_sloppy_block_rewrite(sloppy_tool, temp_work_dir):
    file_path = temp_work_dir / "render.ts"
    await file_path.write_text(
        "function legacyPipeline(input: Frame): Frame {\n"
        "    const staged = stage(input);\n"
        "    return commit(staged);\n"
        "}\n"
    )

    input_text = (
        f"§{file_path}\n"
        "function legacyPipeline(input: Frame): Frame {\n"
        "    const staged = stage(input);\n"
        "    return commit(staged);\n"
        "}\n"
        "»\n"
        "const renderPipeline = (input: Frame): Frame => commit(stage(input));\n"
    )
    result = await sloppy_tool(EditParams(input=input_text))
    assert not result.is_error
    assert "const renderPipeline" in await file_path.read_text()


async def test_sloppy_block_delete(sloppy_tool, temp_work_dir):
    file_path = temp_work_dir / "util.ts"
    await file_path.write_text(
        "const helper = () => {\n"
        "    return 1;\n"
        "};\n"
        "run(target);\n"
    )

    input_text = (
        f"§{file_path}\n"
        "const helper = () => {\n"
        "    return 1;\n"
        "};\n"
        "»\n"
    )
    result = await sloppy_tool(EditParams(input=input_text))
    assert not result.is_error
    assert await file_path.read_text() == "run(target);\n"


async def test_sloppy_failure_returns_corrected_payload(sloppy_tool, temp_work_dir):
    file_path = temp_work_dir / "missing.ts"
    await file_path.write_text("const a = 1;\n")

    input_text = (
        f"§{file_path}\n"
        "const nonexistent = ⟪999│777⟫;\n"
    )
    result = await sloppy_tool(EditParams(input=input_text))
    assert result.is_error
    assert "Corrected payload" in result.message or "Could not locate" in result.message
    assert await file_path.read_text() == "const a = 1;\n"
