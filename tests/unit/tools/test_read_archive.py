"""Tests for archive reading via ReadFile."""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaos.path import KaosPath
from kosong.tooling import ToolOk

from kimi_cli.tools.file.read import Params as ReadFileParams, ReadFile
from kimi_cli.tools.file.read_archive import (
    ArchiveReader,
    MAX_ARCHIVE_MEMBER_BYTES,
    MAX_ARCHIVE_TOTAL_BYTES,
    normalize_member_path,
)


@pytest.fixture
def read_tool(tmp_path: Path) -> ReadFile:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = KaosPath(str(tmp_path))
    runtime.additional_dirs = []
    runtime.llm.capabilities = set()
    session = MagicMock()
    session.id = "test"
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    return ReadFile(runtime, session)


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _make_targz(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class TestArchiveHelpers:
    def test_normalize_rejects_traversal(self) -> None:
        assert normalize_member_path("../etc/passwd") is None
        assert normalize_member_path("/absolute") is None
        assert normalize_member_path("a\\b") is None
        assert normalize_member_path("a\x00b") is None
        assert normalize_member_path("a/./b/c") == "a/b/c"

    def test_archive_reader_zip_list_and_read(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"src/main.py": b"print('hello')\n", "README.md": b"# hi\n"})
        with ArchiveReader(str(archive)) as reader:
            entries = reader.list_directory()
            names = {e.name for e in entries}
            assert names == {"README.md", "src"}
            data = reader.read_file("src/main.py")
            assert data == b"print('hello')\n"

    def test_archive_reader_targz_list_and_read(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.gz"
        _make_targz(archive, {"a.txt": b"alpha\n", "b.txt": b"beta\n"})
        with ArchiveReader(str(archive)) as reader:
            entries = reader.list_directory()
            assert {e.name for e in entries} == {"a.txt", "b.txt"}
            assert reader.read_file("a.txt") == b"alpha\n"

    def test_archive_reader_bare_gz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.gz"
        with gzip.open(archive, "wb") as f:
            f.write(b"compressed payload\n")
        with ArchiveReader(str(archive)) as reader:
            entries = reader.list_directory()
            assert len(entries) == 1
            assert entries[0].name == "data"
            assert reader.read_file("data") == b"compressed payload\n"

    def test_archive_reader_pagination(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        files = {f"f{i}.txt": b"x" for i in range(10)}
        _make_zip(archive, files)
        with ArchiveReader(str(archive)) as reader:
            assert len(reader.list_directory(offset=1, limit=5)) == 5
            assert len(reader.list_directory(offset=6, limit=5)) == 5
            assert reader.list_directory(offset=11, limit=5) == []


class TestArchiveReadFile:
    async def test_list_root(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"src/main.py": b"print('hello')\n", "README.md": b"# hi\n"})
        result = await read_tool(ReadFileParams(path=str(archive)))
        print("OUTPUT:", repr(result.output))
        print("MESSAGE:", repr(result.message))
        assert isinstance(result, ToolOk)
        assert "src/" in result.output
        assert "README.md" in result.output

    async def test_read_member(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"src/main.py": b"print('hello')\nline two\n"})
        result = await read_tool(
            ReadFileParams(path=str(archive), archive_member="src/main.py", limit=1)
        )
        assert isinstance(result, ToolOk)
        assert "print('hello')" in result.output
        assert "line two" not in result.output

    async def test_member_not_found(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"a.txt": b"hi"})
        result = await read_tool(
            ReadFileParams(path=str(archive), archive_member="missing.txt")
        )
        assert result.is_error
        assert "not found" in result.message.lower()

    async def test_traversal_member_rejected(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"a.txt": b"hi"})
        result = await read_tool(
            ReadFileParams(path=str(archive), archive_member="../a.txt")
        )
        assert result.is_error

    async def test_binary_member_notice(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"blob.bin": b"\x00\x01\x02"})
        result = await read_tool(
            ReadFileParams(path=str(archive), archive_member="blob.bin")
        )
        assert isinstance(result, ToolOk)
        assert "Cannot read binary archive member" in result.output

    async def test_archive_member_overrides_unknown_extension(self, read_tool: ReadFile, tmp_path: Path) -> None:
        # When archive_member is explicitly set on a non-archive file, error.
        f = tmp_path / "not_really.zip"
        f.write_text("plain text file\n")
        result = await read_tool(
            ReadFileParams(path=str(f), archive_member="x.txt")
        )
        assert result.is_error
        assert "Cannot read archive" in result.message

    async def test_non_archive_zip_extension_falls_through(self, read_tool: ReadFile, tmp_path: Path) -> None:
        # A text file with a .zip extension that fails to open as an archive
        # should still be readable as plain text.
        f = tmp_path / "not_really.zip"
        f.write_text("plain text file\n")
        result = await read_tool(ReadFileParams(path=str(f)))
        assert isinstance(result, ToolOk)
        assert "plain text file" in result.output

    async def test_size_cap(self, read_tool: ReadFile, tmp_path: Path) -> None:
        archive = tmp_path / "big.zip"
        _make_zip(archive, {"huge.txt": b"x" * (MAX_ARCHIVE_MEMBER_BYTES + 10)})
        result = await read_tool(
            ReadFileParams(path=str(archive), archive_member="huge.txt")
        )
        assert result.is_error
        assert "too large" in result.message.lower()
