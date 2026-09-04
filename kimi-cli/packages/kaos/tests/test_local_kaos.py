from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from kaos import reset_current_kaos, set_current_kaos
from kaos.local import LocalKaos
from kaos.path import KaosPath


@pytest.fixture
def local_kaos(tmp_path: Path) -> Generator[LocalKaos]:
    """Set LocalKaos as the current Kaos and switch cwd to a temp directory."""
    local = LocalKaos()
    token = set_current_kaos(local)
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        yield local
    finally:
        os.chdir(old_cwd)
        reset_current_kaos(token)


def test_pathclass_gethome_and_getcwd(local_kaos: LocalKaos):
    path_class = local_kaos.pathclass()
    if os.name == "nt":
        assert issubclass(path_class, PureWindowsPath)
    else:
        assert issubclass(path_class, PurePosixPath)

    assert str(local_kaos.gethome()) == str(Path.home())
    assert str(local_kaos.getcwd()) == str(Path.cwd())


async def test_chdir_and_stat(local_kaos: LocalKaos):
    new_dir = local_kaos.getcwd() / "nested"
    await local_kaos.mkdir(new_dir)

    await local_kaos.chdir(new_dir)
    assert Path.cwd() == new_dir.unsafe_to_local_path()

    file_path = new_dir / "file.txt"
    await local_kaos.writetext(file_path, "hello world")

    stat_result = await local_kaos.stat(file_path)
    assert stat_result.st_size == len("hello world")


async def test_iterdir_and_glob(local_kaos: LocalKaos):
    tmp_path = local_kaos.getcwd()
    await local_kaos.mkdir(tmp_path / "alpha")
    await local_kaos.writetext(tmp_path / "bravo.txt", "bravo")
    await local_kaos.writetext(tmp_path / "charlie.TXT", "charlie")

    entries = [entry async for entry in local_kaos.iterdir(tmp_path)]
    assert {entry.name for entry in entries} == {"alpha", "bravo.txt", "charlie.TXT"}
    assert all(isinstance(entry, KaosPath) for entry in entries)

    matched = [entry.name async for entry in local_kaos.glob(tmp_path, "*.txt")]
    assert set(matched) == {"bravo.txt"}


async def test_glob_includes_hidden_files(local_kaos: LocalKaos):
    """Glob should match dotfiles (hidden files) with * and ** patterns."""
    tmp_path = local_kaos.getcwd()

    # Create hidden and visible files
    await local_kaos.writetext(tmp_path / ".gitlab-ci.yml", "stages: [build]")
    await local_kaos.writetext(tmp_path / "config.yml", "key: value")
    await local_kaos.mkdir(tmp_path / "src")
    await local_kaos.mkdir(tmp_path / "src" / ".config")
    await local_kaos.writetext(tmp_path / "src" / ".config" / "settings.yml", "debug: true")
    await local_kaos.writetext(tmp_path / "src" / "main.py", "pass")

    # *.yml should match .gitlab-ci.yml
    matched = {entry.name async for entry in local_kaos.glob(tmp_path, "*.yml")}
    assert ".gitlab-ci.yml" in matched
    assert "config.yml" in matched

    # src/**/*.yml should find files in hidden directories
    deep_matched = [
        str(entry.relative_to(tmp_path))
        async for entry in local_kaos.glob(tmp_path, "src/**/*.yml")
    ]
    assert any(".config" in p for p in deep_matched)


async def test_read_write_and_append_text(local_kaos: LocalKaos):
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "note.txt"

    written = await local_kaos.writetext(file_path, "line1")
    assert written == len("line1")

    content = await local_kaos.readtext(file_path)
    assert content == "line1"

    await local_kaos.writetext(file_path, "\nline2", mode="a")
    lines = [line async for line in local_kaos.readlines(file_path)]
    assert "".join(lines) == "line1\nline2"


async def test_writetext_preserves_lf_line_endings(local_kaos: LocalKaos):
    """writetext should not convert LF to CRLF on any platform."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "lf.txt"

    await local_kaos.writetext(file_path, "hello\nworld\n")

    # Read back as binary to check actual bytes on disk
    raw = await local_kaos.readbytes(file_path)
    assert raw == b"hello\nworld\n", f"Expected LF line endings, got {raw!r}"


async def test_writetext_preserves_crlf_line_endings(local_kaos: LocalKaos):
    """writetext should preserve CRLF if explicitly present in content."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "crlf.txt"

    await local_kaos.writetext(file_path, "hello\r\nworld\r\n")

    raw = await local_kaos.readbytes(file_path)
    assert raw == b"hello\r\nworld\r\n", f"Expected CRLF preserved, got {raw!r}"


async def test_mkdir_with_parents(local_kaos: LocalKaos):
    tmp_path = local_kaos.getcwd()
    nested_dir = tmp_path / "a" / "b" / "c"

    await local_kaos.mkdir(nested_dir, parents=True)
    assert await nested_dir.is_dir()


async def test_read_write_bytes(local_kaos: LocalKaos):
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "data.bin"
    await local_kaos.writebytes(file_path, b"\x00\x01\xff")
    assert await local_kaos.readbytes(file_path) == b"\x00\x01\xff"


def _python_code_args(code: str) -> tuple[str, str, str]:
    return sys.executable, "-c", code


async def test_exec_runs_command_and_streams(local_kaos: LocalKaos):
    code = "import sys\nsys.stdout.write('hello\\n')\nsys.stderr.write('stderr line\\n')\n"

    process = await local_kaos.exec(*_python_code_args(code))

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_data, stderr_data = await asyncio.gather(process.stdout.read(), process.stderr.read())
    assert await process.wait() == 0
    assert stdout_data.decode("utf-8").strip() == "hello"
    assert stderr_data.decode("utf-8").strip() == "stderr line"


async def test_exec_runs_command_wait_before_read(local_kaos: LocalKaos):
    code = "import sys\nsys.stdout.write('hello\\n')\nsys.stderr.write('stderr line\\n')\n"

    process = await local_kaos.exec(*_python_code_args(code))

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    assert await process.wait() == 0
    stdout_data, stderr_data = await asyncio.gather(process.stdout.read(), process.stderr.read())
    assert stdout_data.decode("utf-8").strip() == "hello"
    assert stderr_data.decode("utf-8").strip() == "stderr line"


async def test_exec_non_zero_exit(local_kaos: LocalKaos):
    process = await local_kaos.exec(*_python_code_args("import sys; sys.exit(7)"))

    exit_code = await process.wait()
    assert exit_code == 7


async def test_exec_wait_timeout(local_kaos: LocalKaos):
    process = await local_kaos.exec(*_python_code_args("import time; time.sleep(1)"))
    assert process.pid > 0

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=0.01)
    finally:
        if process.returncode is None:
            await process.kill()
        await process.wait()


async def test_writetext_encode_failure_preserves_existing_file(local_kaos: LocalKaos):
    """A strict-encode failure on overwrite must not truncate the target."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "keep.txt"
    await local_kaos.writetext(file_path, "original content")
    with pytest.raises(UnicodeEncodeError):
        # '€' is unencodable in strict ascii; previously the file was
        # truncated by the mode="w" open before the encode error surfaced.
        await local_kaos.writetext(file_path, "euro: €", encoding="ascii")
    assert await local_kaos.readtext(file_path) == "original content"


async def test_writetext_append_encode_failure_appends_nothing(local_kaos: LocalKaos):
    """A strict-encode failure on append must not touch the file at all."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "append_keep.txt"
    await local_kaos.writetext(file_path, "base")
    with pytest.raises(UnicodeEncodeError):
        await local_kaos.writetext(file_path, "€", mode="a", encoding="ascii")
    assert await local_kaos.readtext(file_path) == "base"


async def test_writetext_overwrite_is_atomic_and_leaves_no_temp_files(local_kaos: LocalKaos):
    """Overwrite publishes via temp+replace; no *.tmp siblings remain."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "atomic.txt"
    await local_kaos.writetext(file_path, "v1")
    await local_kaos.writetext(file_path, "v2-longer-content")
    assert await local_kaos.readtext(file_path) == "v2-longer-content"
    leftovers = [p async for p in local_kaos.iterdir(tmp_path)]
    assert [p.name for p in leftovers] == ["atomic.txt"]


async def test_writebytes_is_atomic(local_kaos: LocalKaos):
    """writebytes overwrites fully and leaves no temp files behind."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "data.bin"
    await local_kaos.writebytes(file_path, b"old")
    await local_kaos.writebytes(file_path, b"new-bytes")
    assert await local_kaos.readbytes(file_path) == b"new-bytes"
    leftovers = [p.name async for p in local_kaos.iterdir(tmp_path)]
    assert leftovers == ["data.bin"]


async def test_writetext_non_utf8_encoding_roundtrip(local_kaos: LocalKaos):
    """Explicit encodings (e.g. utf-8-sig BOM) still round-trip correctly."""
    tmp_path = local_kaos.getcwd()
    file_path = tmp_path / "bom.txt"
    await local_kaos.writetext(file_path, "你好", encoding="utf-8-sig")
    raw = await local_kaos.readbytes(file_path)
    assert raw == b"\xef\xbb\xbf" + "你好".encode("utf-8")
    assert await local_kaos.readtext(file_path, encoding="utf-8-sig") == "你好"
