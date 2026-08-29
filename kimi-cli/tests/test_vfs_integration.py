"""Tests for VFS integration with file tools."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from kaos import reset_current_kaos, set_current_kaos
from kaos.local import LocalKaos
from kaos.path import KaosPath

from kimi_cli.soul.toolset import current_tool_call
from kimi_cli.tools.file import Glob, Grep, ReadFile, WriteFile
from kimi_cli.tools.file.glob import Params as GlobParams
from kimi_cli.tools.file.grep_local import Params as GrepParams
from kimi_cli.tools.file.read import Params as ReadFileParams
from kimi_cli.tools.file.write import Params as WriteFileParams
from kimi_cli.vfs import VFS
from kimi_cli.vfs.core import merge as vfs_merge
from kimi_cli.wire.types import ToolCall


@pytest.fixture
def vfs_setup():
    """Create VFS with temp virtual_root over a temp work_dir, with kaos context."""
    with tempfile.TemporaryDirectory() as work_dir_tmp, tempfile.TemporaryDirectory() as virtual_root_tmp:
            work_dir = Path(work_dir_tmp).resolve()
            virtual_root = Path(virtual_root_tmp).resolve()
            original_cwd = Path.cwd()
            os.chdir(work_dir)
            token = set_current_kaos(LocalKaos())
            try:
                vfs = VFS(virtual_root=virtual_root, work_dir=work_dir)
                yield work_dir, virtual_root, vfs
            finally:
                reset_current_kaos(token)
                os.chdir(original_cwd)


@contextmanager
def _tool_call_ctx(tool_name: str) -> Generator[None]:
    """Set the current tool call context for approval."""
    token = current_tool_call.set(
        ToolCall(id="test", function=ToolCall.FunctionBody(name=tool_name, arguments=None))
    )
    try:
        yield
    finally:
        current_tool_call.reset(token)


def _mock_runtime(work_dir: Path):
    """Create a minimal mock Runtime with the required attributes."""

    class MockBuiltinArgs:
        KIMI_WORK_DIR = KaosPath.unsafe_from_local_path(work_dir)

    class MockEnvironment:
        os_kind = "Windows"

    class MockRuntime:
        builtin_args = MockBuiltinArgs()
        additional_dirs = []
        skills_dirs = []
        environment = MockEnvironment()
        session = _mock_session()

    return MockRuntime()


def _mock_approval():
    """Create a mock Approval that auto-approves all requests."""

    class MockApprovalResult:
        def __bool__(self):
            return True

        def rejection_error(self):
            return None

    class MockApproval:
        async def request(self, tool_name, action, description, display=None):
            return MockApprovalResult()

    return MockApproval()


def _mock_session():
    """Create a minimal mock Session with the required attributes."""

    class MockFileMtime:
        def mark_dirty(self, path: str) -> bool:
            return True
        def clean_file(self, path: str) -> None:
            pass

    class MockSession:
        id = "test-session"
        custom_data = {}
        custom_config = {}
        file_mtime = MockFileMtime()

    return MockSession()


class TestVFSOverlay:
    async def test_write_marks_dirty(self, vfs_setup):
        """WriteFile marks an existing file as dirty in VFS."""
        work_dir, virtual_root, vfs = vfs_setup
        test_path = Path(work_dir) / "test.txt"
        test_path.write_text("original")

        tool = WriteFile(
            runtime=_mock_runtime(work_dir),
            approval=_mock_approval(),
            session=_mock_session(),
            vfs=vfs,
        )
        with _tool_call_ctx("WriteFile"):
            result = await tool(WriteFileParams(path="test.txt", content="hello"))

        assert not result.is_error
        assert vfs.is_dirty(test_path)
        assert (Path(virtual_root) / "test.txt").exists()
        assert test_path.read_text() == "original"  # Original unchanged

    async def test_read_sees_virtual_file(self, vfs_setup):
        """ReadFile sees the virtual content for a dirty file."""
        work_dir, _, vfs = vfs_setup
        test_path = Path(work_dir) / "test.txt"
        test_path.write_text("original")

        write_tool = WriteFile(
            runtime=_mock_runtime(work_dir),
            approval=_mock_approval(),
            session=_mock_session(),
            vfs=vfs,
        )
        read_tool = ReadFile(
            runtime=_mock_runtime(work_dir),
            session=_mock_session(),
            vfs=vfs,
        )

        with _tool_call_ctx("WriteFile"):
            await write_tool(WriteFileParams(path="test.txt", content="virtual content"))
        result = await read_tool(ReadFileParams(path="test.txt"))

        assert not result.is_error
        assert "virtual content" in result.output

    async def test_glob_sees_virtual_files(self, vfs_setup):
        """Glob includes dirty files in results (originals still exist in work_dir)."""
        work_dir, _, vfs = vfs_setup
        write_tool = WriteFile(
            runtime=_mock_runtime(work_dir),
            approval=_mock_approval(),
            session=_mock_session(),
            vfs=vfs,
        )
        glob_tool = Glob(runtime=_mock_runtime(work_dir), vfs=vfs)

        subdir = Path(work_dir) / "subdir"
        subdir.mkdir()
        (subdir / "new.txt").write_text("original")

        with _tool_call_ctx("WriteFile"):
            await write_tool(WriteFileParams(path="subdir/new.txt", content="x"))
        result = await glob_tool(
            GlobParams(pattern="subdir/*.txt", directory=str(work_dir))
        )

        assert not result.is_error
        assert "new.txt" in result.output

    async def test_grep_finds_in_virtual_file(self, vfs_setup):
        """Grep searches virtual file content when files are dirty."""
        work_dir, _, vfs = vfs_setup
        write_tool = WriteFile(
            runtime=_mock_runtime(work_dir),
            approval=_mock_approval(),
            session=_mock_session(),
            vfs=vfs,
        )
        grep_tool = Grep(runtime=_mock_runtime(work_dir), vfs=vfs)

        test_path = Path(work_dir) / "test.txt"
        test_path.write_text("original text here")

        with _tool_call_ctx("WriteFile"):
            await write_tool(WriteFileParams(path="test.txt", content="virtual text here"))
        result = await grep_tool(
            GrepParams(
                pattern="virtual",
                path=str(work_dir),
                output_mode="files_with_matches",
            )
        )

        assert not result.is_error
        assert "test.txt" in result.output

    async def test_read_original_when_clean(self, vfs_setup):
        """ReadFile reads from work_dir when the file is not dirty."""
        work_dir, _, vfs = vfs_setup
        test_path = Path(work_dir) / "existing.txt"
        test_path.write_text("original content")

        read_tool = ReadFile(
            runtime=_mock_runtime(work_dir),
            session=_mock_session(),
            vfs=vfs,
        )
        result = await read_tool(ReadFileParams(path="existing.txt"))

        assert not result.is_error
        assert "original content" in result.output

    async def test_write_new_file_not_dirty(self, vfs_setup):
        """WriteFile creates new files in work_dir and does not mark them dirty."""
        work_dir, _, vfs = vfs_setup

        tool = WriteFile(
            runtime=_mock_runtime(work_dir),
            approval=_mock_approval(),
            session=_mock_session(),
            vfs=vfs,
        )
        with _tool_call_ctx("WriteFile"):
            result = await tool(WriteFileParams(path="new_file.txt", content="new content"))

        assert not result.is_error
        assert (Path(work_dir) / "new_file.txt").exists()
        assert (Path(work_dir) / "new_file.txt").read_text() == "new content"
        assert not vfs.is_dirty(Path(work_dir) / "new_file.txt")


# --- Merged from tests/test_vfs_integration.py: multi-VFS merge tests ---


def test_multi_vfs_shared_workdir(tmp_path: Path) -> None:
    """Multiple VFS instances share a work_dir and make independent changes."""
    wd = tmp_path / "work"
    wd.mkdir()

    vfs_list = [
        VFS(tmp_path / f"virtual_{i}", wd)
        for i in range(4)
    ]

    (wd / "shared.txt").write_text("base_shared")
    (wd / "base.txt").write_text("base_base")

    vfs_list[0].get(wd / "shared.txt")
    (vfs_list[0].virtual_root / "shared.txt").write_text("vfs0_edit")

    vfs_list[1].get(wd / "shared.txt")
    (vfs_list[1].virtual_root / "shared.txt").write_text("vfs1_edit")

    vfs_list[2].get(wd / "base.txt")
    vfs_list[3].get(wd / "base.txt")
    (vfs_list[2].virtual_root / "base.txt").write_text("same_edit")
    (vfs_list[3].virtual_root / "base.txt").write_text("same_edit")

    new_file_0 = vfs_list[0].virtual_root / "new0.txt"
    new_file_0.parent.mkdir(parents=True, exist_ok=True)
    new_file_0.write_text("new0")

    new_file_1 = vfs_list[1].virtual_root / "new1.txt"
    new_file_1.parent.mkdir(parents=True, exist_ok=True)
    new_file_1.write_text("new1")

    conflicts, applied = vfs_merge(*vfs_list, apply=True)

    assert Path("new0.txt") in applied
    assert Path("new1.txt") in applied
    assert Path("base.txt") in applied
    assert (wd / "new0.txt").read_text() == "new0"
    assert (wd / "new1.txt").read_text() == "new1"
    assert (wd / "base.txt").read_text() == "same_edit"

    assert Path("shared.txt") in conflicts
    assert (wd / "shared.txt").read_text() == "base_shared"

    conflict_entries = conflicts[Path("shared.txt")]
    chosen_idx, chosen_data = random.choice(conflict_entries)
    dest = wd / "shared.txt"
    dest.write_bytes(chosen_data)

    for vfs in vfs_list:
        rel = Path("shared.txt")
        vfile = vfs.virtual_root / rel
        if vfile.exists():
            vfile.unlink()

    assert (wd / "shared.txt").read_text() in {"vfs0_edit", "vfs1_edit"}
    assert not any(vfs.is_dirty(wd / "base.txt") for vfs in vfs_list)
    assert not any(vfs.is_dirty(wd / "new0.txt") for vfs in vfs_list)
    assert not any(vfs.is_dirty(wd / "new1.txt") for vfs in vfs_list)
