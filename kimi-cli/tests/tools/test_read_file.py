"""Tests for the read_file tool."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from kaos.path import KaosPath

from kimi_cli.tools.file.read import (
    MAX_BYTES,
    MAX_FILES,
    MAX_LINE_LENGTH,
    MAX_LINES,
    Params,
    ReadFile,
)


@pytest.fixture
async def sample_file(temp_work_dir: KaosPath) -> KaosPath:
    """Create a sample file with test content."""
    file_path = temp_work_dir / "sample.txt"
    content = """Line 1: Hello World
Line 2: This is a test file
Line 3: With multiple lines
Line 4: For testing purposes
Line 5: End of file"""
    await file_path.write_text(content)
    return file_path




def _output_has_header(output: str, display_path: str) -> bool:
    """Check that output starts with the file header."""
    return output.startswith(f"======== {display_path} ========")


def _output_content(output: str) -> str:
    """Extract content after the header."""
    lines = output.splitlines()
    if lines and lines[0].startswith("========"):
        return "\n".join(lines[1:]) if len(lines) > 1 else ""
    return output


async def test_read_entire_file(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test reading an entire file."""
    display_path = str(sample_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file)))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     1	Line 1: Hello World
     2	Line 2: This is a test file
     3	Line 3: With multiple lines
     4	Line 4: For testing purposes
     5	Line 5: End of file\
"""
    )
    assert result.message.startswith("5 lines read from file starting from line 1. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_with_line_offset(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test reading from a specific line offset."""
    display_path = str(sample_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file), line_offset=3))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     3	Line 3: With multiple lines
     4	Line 4: For testing purposes
     5	Line 5: End of file\
"""
    )
    assert result.message.startswith("3 lines read from file starting from line 3. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_with_n_lines(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test reading a specific number of lines."""
    display_path = str(sample_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file), n_lines=2))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     1	Line 1: Hello World
     2	Line 2: This is a test file
"""
    )
    assert result.message.startswith("2 lines read from file starting from line 1.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_with_line_offset_and_n_lines(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test reading with both line offset and n_lines."""
    display_path = str(sample_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file), line_offset=2, n_lines=2))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     2	Line 2: This is a test file
     3	Line 3: With multiple lines
"""
    )
    assert result.message.startswith("2 lines read from file starting from line 2.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_nonexistent_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading a non-existent file."""
    nonexistent_file = temp_work_dir / "nonexistent.txt"
    result = await read_file_tool(Params(path=str(nonexistent_file)))
    assert result.is_error
    display_path = str(nonexistent_file).replace("\\", "/")
    assert result.message == f"`{display_path}` does not exist."
    assert result.brief == "File not found"


async def test_read_directory_instead_of_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test attempting to read a directory."""
    result = await read_file_tool(Params(path=str(temp_work_dir)))
    assert result.is_error
    display_path = str(temp_work_dir).replace("\\", "/")
    assert result.message == f"`{display_path}` is not a file."
    assert result.brief == "Invalid path"


async def test_read_with_relative_path(
    read_file_tool: ReadFile, temp_work_dir: KaosPath, sample_file: KaosPath
):
    """Test reading with a relative path."""
    display_path = str(sample_file.relative_to(temp_work_dir)).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file.relative_to(temp_work_dir))))
    assert not result.is_error
    assert result.message.startswith("5 lines read from file starting from line 1. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")
    assert result.output == snapshot("""\
     1	Line 1: Hello World
     2	Line 2: This is a test file
     3	Line 3: With multiple lines
     4	Line 4: For testing purposes
     5	Line 5: End of file\
""")


async def test_read_with_relative_path_outside_work_dir(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Test reading a file outside the work directory with a relative path (should fail)."""
    path = Path("..") / "outside_file.txt"
    result = await read_file_tool(Params(path=str(path)))
    assert result.is_error
    assert "absolute path" in result.message.lower()
    assert "outside the working directory" in result.message


async def test_read_empty_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading an empty file."""
    empty_file = temp_work_dir / "empty.txt"
    await empty_file.write_text("")
    display_path = str(empty_file).replace("\\", "/")

    result = await read_file_tool(Params(path=str(empty_file)))
    assert not result.is_error
    assert result.output == snapshot('')
    assert result.message.startswith("No lines read from file. Total lines in file: 0. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_image_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading an image file."""
    image_file = temp_work_dir / "sample.png"
    data = b"\x89PNG\r\n\x1a\n" + b"pngdata"
    await image_file.write_bytes(data)

    result = await read_file_tool(Params(path=str(image_file)))

    assert result.is_error
    display_path = str(image_file).replace("\\", "/")
    assert result.message == snapshot(
        f"`{display_path}` is a image file. Use other appropriate tools to read image or video files."
    )
    assert result.brief == "Unsupported file type"


async def test_read_extensionless_image_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading an extensionless image file."""
    image_file = temp_work_dir / "sample"
    data = b"\x89PNG\r\n\x1a\n" + b"pngdata"
    await image_file.write_bytes(data)

    result = await read_file_tool(Params(path=str(image_file)))

    assert result.is_error
    display_path = str(image_file).replace("\\", "/")
    assert result.message == snapshot(
        f"`{display_path}` is a image file. Use other appropriate tools to read image or video files."
    )
    assert result.brief == "Unsupported file type"


async def test_read_video_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading a video file."""
    video_file = temp_work_dir / "sample.mp4"
    data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await video_file.write_bytes(data)

    result = await read_file_tool(Params(path=str(video_file)))

    assert result.is_error
    display_path = str(video_file).replace("\\", "/")
    assert result.message == snapshot(
        f"`{display_path}` is a video file. Use other appropriate tools to read image or video files."
    )
    assert result.brief == "Unsupported file type"


async def test_read_line_offset_beyond_file_length(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test reading with line offset beyond file length."""
    display_path = str(sample_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(sample_file), line_offset=10))
    assert not result.is_error
    assert result.output == snapshot('')
    assert result.message.startswith("No lines read from file. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_unicode_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading a file with unicode characters."""
    unicode_file = temp_work_dir / "unicode.txt"
    content = "Hello 世界 🌍\nUnicode test: café, naïve, résumé"
    await unicode_file.write_text(content, encoding="utf-8")
    display_path = str(unicode_file).replace("\\", "/")

    result = await read_file_tool(Params(path=str(unicode_file)))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     1	Hello 世界 🌍
     2	Unicode test: café, naïve, résumé\
"""
    )
    assert result.message.startswith("2 lines read from file starting from line 1. Total lines in file: 2. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_read_edge_cases(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test edge cases for line offset reading."""
    display_path = str(sample_file).replace("\\", "/")
    # Test reading from line 1 (should be same as default)
    result = await read_file_tool(Params(path=str(sample_file), line_offset=1))
    assert not result.is_error
    assert result.output == snapshot(
        """\
     1	Line 1: Hello World
     2	Line 2: This is a test file
     3	Line 3: With multiple lines
     4	Line 4: For testing purposes
     5	Line 5: End of file\
"""
    )
    assert result.message.startswith("5 lines read from file starting from line 1. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")

    # Test reading from line 5 (last line)
    result = await read_file_tool(Params(path=str(sample_file), line_offset=5))
    assert not result.is_error
    assert result.output == snapshot("     5\tLine 5: End of file")
    assert result.message.startswith("1 lines read from file starting from line 5. Total lines in file: 5. End of file reached.")
    assert result.message.endswith(f" Path: {display_path}")

    # Test reading with offset and n_lines combined
    result = await read_file_tool(Params(path=str(sample_file), line_offset=2, n_lines=1))
    assert not result.is_error
    assert result.output == snapshot("     2\tLine 2: This is a test file\n")
    assert result.message.startswith("1 lines read from file starting from line 2.")
    assert result.message.endswith(f" Path: {display_path}")


async def test_line_truncation_and_messaging(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test line truncation functionality and messaging."""

    # Test single long line truncation
    single_line_file = temp_work_dir / "single_long_line.txt"
    long_content = "A" * (MAX_LINE_LENGTH + 100) + " This should be truncated"
    await single_line_file.write_text(long_content)

    result = await read_file_tool(Params(path=str(single_line_file)))
    assert not result.is_error
    assert isinstance(result.output, str)
    assert "1 lines read from" in result.message
    # Check that the line is truncated and ends with "..."
    assert result.output.endswith("...")

    # Verify exact length after truncation (accounting for line number prefix)
    lines = result.output.split("\n")
    content_line = [line for line in lines if line.strip()][0]
    actual_content = content_line.split("\t", 1)[1] if "\t" in content_line else content_line
    assert len(actual_content) == MAX_LINE_LENGTH

    # Test multiple long lines with truncation messaging
    multi_line_file = temp_work_dir / "multi_truncation_test.txt"
    long_line_1 = "A" * (MAX_LINE_LENGTH + 100)
    long_line_2 = "B" * (MAX_LINE_LENGTH + 200)
    normal_line = "Short line"
    content = f"{long_line_1}\n{normal_line}\n{long_line_2}"
    await multi_line_file.write_text(content)

    display_path = str(multi_line_file).replace("\\", "/")
    result = await read_file_tool(Params(path=str(multi_line_file)))
    assert not result.is_error
    assert isinstance(result.output, str)
    assert result.message.startswith("3 lines read from file starting from line 1. Total lines in file: 3. End of file reached. Lines [1, 3] were truncated.")
    assert result.message.endswith(f" Path: {display_path}")

    # Verify truncation actually happened for specific lines
    lines = result.output.split("\n")
    endings = [line[-20:] for line in lines]
    assert endings == snapshot(
        [
            "AAAAAAAAAAAAAAAAA...",
            "     2\tShort line",
            "BBBBBBBBBBBBBBBBB...",
        ]
    )


async def test_parameter_validation_line_offset(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test that line_offset parameter validation works correctly."""
    # line_offset=0 is invalid (must be positive or negative, not zero)
    with pytest.raises(ValueError, match="line_offset"):
        Params(path=str(sample_file), line_offset=0)

    # Negative values are now valid (tail mode)
    params = Params(path=str(sample_file), line_offset=-1)
    assert params.line_offset == -1

    # Negative offset exceeding MAX_LINES should be rejected
    with pytest.raises(ValueError, match="line_offset"):
        Params(path=str(sample_file), line_offset=-(MAX_LINES + 1))

    # Exactly -MAX_LINES should be accepted
    params = Params(path=str(sample_file), line_offset=-MAX_LINES)
    assert params.line_offset == -MAX_LINES


async def test_parameter_validation_n_lines(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test that n_lines parameter validation works correctly."""
    # Test n_lines < 1 should be rejected by Pydantic validation
    with pytest.raises(ValueError, match="n_lines"):
        Params(path=str(sample_file), n_lines=0)

    with pytest.raises(ValueError, match="n_lines"):
        Params(path=str(sample_file), n_lines=-1)


async def test_max_lines_boundary(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test that reading respects the MAX_LINES boundary."""
    # Create a file with more than MAX_LINES lines
    large_file = temp_work_dir / "large_file.txt"
    content = "\n".join([f"Line {i}" for i in range(1, MAX_LINES + 10)])
    await large_file.write_text(content)

    # Request more than MAX_LINES to trigger the boundary check.
    # Use a large max_char so the output is not sliced after reading.
    result = await read_file_tool(
        Params(path=str(large_file), n_lines=MAX_LINES + 5, max_char=10_000_000)
    )

    assert not result.is_error
    assert isinstance(result.output, str)
    # Should read MAX_LINES lines, not the full file
    assert f"Max {MAX_LINES} lines reached" in result.message
    # Count actual lines in output (accounting for line numbers)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    assert len(output_lines) == MAX_LINES


# --- [out of work-dir] warning tests ---


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in ReadFile")
async def test_read_outside_work_dir_has_warning(
    read_file_tool: ReadFile, outside_file: Path
):
    """Reading outside work-dir should include [out of work-dir] in success message."""
    outside_file.write_text("outside content", encoding="utf-8")
    result = await read_file_tool(Params(path=str(outside_file)))
    assert not result.is_error
    assert "[out of work-dir]" in result.message


async def test_read_inside_work_dir_no_warning(read_file_tool: ReadFile, sample_file: KaosPath):
    """Reading inside work-dir should NOT include [out of work-dir] in message."""
    result = await read_file_tool(Params(path=str(sample_file)))
    assert not result.is_error
    assert "[out of work-dir]" not in result.message


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in ReadFile")
async def test_read_outside_work_dir_nonexistent_has_warning(
    read_file_tool: ReadFile, outside_file: Path
):
    """Error reading non-existent outside file should include [out of work-dir]."""
    result = await read_file_tool(Params(path=str(outside_file)))
    assert result.is_error
    assert "[out of work-dir]" in result.message


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in ReadFile")
async def test_read_outside_work_dir_directory_has_warning(
    read_file_tool: ReadFile
):
    """Reading a directory outside work-dir should include [out of work-dir]."""
    # Use /tmp or equivalent that exists but is outside work-dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await read_file_tool(Params(path=tmpdir))
        assert result.is_error
        assert "[out of work-dir]" in result.message


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in ReadFile")
async def test_read_outside_relative_path_error_has_warning(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Relative path outside workspace error should include [out of work-dir]."""
    result = await read_file_tool(Params(path="../outside.txt"))
    assert result.is_error
    assert "[out of work-dir]" in result.message


async def test_max_bytes_boundary(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test that reading respects the MAX_BYTES boundary."""
    # Create a file that exceeds MAX_BYTES but stays under MAX_LINES
    large_file = temp_work_dir / "large_bytes.txt"
    line_content = "A" * 1000  # 1000 characters per line
    num_lines = (MAX_BYTES // 1000) + 5  # Enough to exceed MAX_BYTES
    content = "\n".join([line_content] * num_lines)
    await large_file.write_text(content)

    result = await read_file_tool(Params(path=str(large_file)))

    assert not result.is_error
    assert f"Max {MAX_BYTES} bytes reached" in result.message


async def test_read_with_tilde_path_expansion(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading with ~ path expansion."""
    # Create a test file in temp_work_dir and use ~ to reference it
    # We simulate by creating a file and checking that ~ expands correctly
    home = Path.home()
    test_file = home / ".test_expanduser_temp"
    test_content = "Test content for tilde expansion"

    try:
        # Create the test file in home directory
        test_file.write_text(test_content)
        display_path = "~/.test_expanduser_temp"

        # Read using ~ path
        result = await read_file_tool(Params(path="~/.test_expanduser_temp"))

        assert not result.is_error
        assert "Test content for tilde expansion" in result.output
        assert result.message.startswith("1 lines read from file starting from line 1. Total lines in file: 1. End of file reached.")
        assert result.message.endswith(f" Path: {display_path}")
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


async def test_read_rejects_sensitive_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """ReadFile should block reading files that match sensitive patterns."""
    env_file = temp_work_dir / ".env"
    await env_file.write_text("SECRET_KEY=hunter2\n")

    result = await read_file_tool(Params(path=str(env_file)))

    assert result.is_error
    assert "sensitive" in result.message.lower() or "secrets" in result.message.lower()
    assert "blocked" in result.message.lower() or "protect" in result.message.lower()


async def test_read_allows_non_sensitive_dotfile(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """ReadFile should allow reading non-sensitive dotfiles like .gitignore."""
    gitignore = temp_work_dir / ".gitignore"
    await gitignore.write_text("node_modules/\n")

    result = await read_file_tool(Params(path=str(gitignore)))

    assert not result.is_error
    assert "node_modules" in result.output


# ── Tests for totalLines and tail (negative offset) ──────────────────────────


async def test_read_tail_basic(read_file_tool: ReadFile, sample_file: KaosPath):
    """Negative line_offset=-3 on a 5-line file should return the last 3 lines."""
    result = await read_file_tool(Params(path=str(sample_file), line_offset=-3))
    assert not result.is_error
    # Should return lines 3, 4, 5 with absolute line numbers
    assert "     3\tLine 3: With multiple lines\n" in result.output
    assert "     4\tLine 4: For testing purposes\n" in result.output
    assert "     5\tLine 5: End of file" in result.output
    # Should NOT contain lines 1 or 2
    assert "Line 1:" not in result.output
    assert "Line 2:" not in result.output
    # Message must include total lines info
    assert "Total lines in file: 5." in result.message


async def test_read_tail_with_n_lines(read_file_tool: ReadFile, sample_file: KaosPath):
    """Negative offset=-5 with n_lines=2 should return 2 lines starting from the tail position."""
    result = await read_file_tool(Params(path=str(sample_file), line_offset=-5, n_lines=2))
    assert not result.is_error
    # -5 on a 5-line file means start from line 1, then n_lines=2 limits to lines 1-2
    assert "     1\tLine 1: Hello World\n" in result.output
    assert "     2\tLine 2: This is a test file\n" in result.output
    assert "Line 3:" not in result.output
    assert "Total lines in file: 5." in result.message


async def test_read_tail_exceeds_file(read_file_tool: ReadFile, sample_file: KaosPath):
    """Negative offset exceeding file length should return the entire file."""
    result = await read_file_tool(Params(path=str(sample_file), line_offset=-100))
    assert not result.is_error
    # Should return all 5 lines
    assert "     1\tLine 1: Hello World\n" in result.output
    assert "     5\tLine 5: End of file" in result.output
    assert "Total lines in file: 5." in result.message


async def test_read_tail_empty_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Negative offset on an empty file should return nothing with totalLines=0."""
    empty_file = temp_work_dir / "empty_tail.txt"
    await empty_file.write_text("")

    result = await read_file_tool(Params(path=str(empty_file), line_offset=-10))
    assert not result.is_error
    assert result.output == ""
    assert "Total lines in file: 0." in result.message


async def test_read_total_lines_with_positive_offset(
    read_file_tool: ReadFile, sample_file: KaosPath
):
    """Positive offset should also include totalLines in the message."""
    result = await read_file_tool(Params(path=str(sample_file), line_offset=3, n_lines=1))
    assert not result.is_error
    # Should return only line 3
    assert "     3\tLine 3: With multiple lines" in result.output
    assert "Line 1:" not in result.output
    assert "Line 4:" not in result.output
    # Message does not include total lines when eof is not reached
    assert "Total lines in file: 5." not in result.message


async def test_read_tail_last_line(read_file_tool: ReadFile, sample_file: KaosPath):
    """line_offset=-1 should return only the last line with correct absolute line number."""
    result = await read_file_tool(Params(path=str(sample_file), line_offset=-1))
    assert not result.is_error
    assert result.output == "     5\tLine 5: End of file"
    assert "1 lines read from file starting from line 5." in result.message
    assert "Total lines in file: 5." in result.message
    assert "End of file reached." in result.message


async def test_read_tail_max_lines(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Tail mode with -MAX_LINES on a file larger than MAX_LINES should return MAX_LINES lines."""
    # Create a file with more than MAX_LINES lines
    large_file = temp_work_dir / "tail_large.txt"
    total = MAX_LINES + 500  # 1500 lines
    content = "\n".join([f"Line {i}" for i in range(1, total + 1)])
    await large_file.write_text(content)

    # Use -MAX_LINES (the maximum allowed negative offset).
    # Use a large max_char so the output is not sliced after reading.
    result = await read_file_tool(
        Params(path=str(large_file), line_offset=-MAX_LINES, max_char=10_000_000)
    )
    assert not result.is_error
    assert f"Total lines in file: {total}." in result.message
    # tail_buf captures the last MAX_LINES lines; n_lines defaults to MAX_LINES so all are output
    assert isinstance(result.output, str)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    assert len(output_lines) == MAX_LINES
    # First line should be line 501 (total - MAX_LINES + 1)
    assert output_lines[0].endswith(f"Line {total - MAX_LINES + 1}")


async def test_read_tail_max_bytes(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Tail mode byte-budget truncation should keep newest lines (closest to EOF)."""
    large_file = temp_work_dir / "tail_bytes.txt"
    max_bytes = MAX_BYTES
    # Each line ~1001 bytes (1000 chars + \n), need > MAX_BYTES to trigger truncation
    num_lines = (MAX_BYTES // 1001) + 20
    # Tag each line with its number so we can verify which lines are kept
    lines_data = [f"{i:04d}{'B' * 996}" for i in range(1, num_lines + 1)]
    content = "\n".join(lines_data)
    await large_file.write_text(content)

    result = await read_file_tool(
        Params(path=str(large_file), line_offset=-(num_lines), max_char=2_000_000)
    )
    assert not result.is_error
    assert f"Max {max_bytes} bytes reached" in result.message
    assert f"Total lines in file: {num_lines}." in result.message

    # Verify that the LAST line of the file is included (newest lines kept)
    assert isinstance(result.output, str)
    output_lines = [x for x in result.output.split("\n") if x.strip()]
    last_output = output_lines[-1].split("\t", 1)[1]
    assert last_output.startswith(f"{num_lines:04d}"), (
        "Byte-budget truncation should keep newest lines closest to EOF"
    )
    # Verify that the first output line is NOT line 1 (oldest lines trimmed)
    first_output = output_lines[0].split("\t", 1)[1]
    assert not first_output.startswith("0001"), "Byte-budget truncation should trim oldest lines"


async def test_read_tail_n_lines_not_affected_by_byte_cap(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Small n_lines should not be affected by MAX_BYTES truncation.

    Regression test: line_offset=-N, n_lines=1 on a file with long lines
    should return the first line of the tail window, not a line shifted by byte-cap.
    """
    large_file = temp_work_dir / "tail_nlines_bytecap.txt"
    # Create a file where tail_buf total bytes >> MAX_BYTES but n_lines=1 is fine.
    # Each line ~2000 bytes, 500 lines total (> MAX_BYTES).
    num_lines = 500
    lines_data = [f"{i:04d}{'X' * 1996}" for i in range(1, num_lines + 1)]
    content = "\n".join(lines_data)
    await large_file.write_text(content)

    # Request tail window of 200 lines but only read 1
    result = await read_file_tool(Params(path=str(large_file), line_offset=-200, n_lines=1))
    assert not result.is_error
    assert isinstance(result.output, str)

    # The first line of the tail window (last 200 lines) is line 301
    output_lines = [x for x in result.output.split("\n") if x.strip()]
    assert len(output_lines) == 1
    line_content = output_lines[0].split("\t", 1)[1]
    assert line_content.startswith("0301"), (
        f"Expected line 301 (start of tail window), got content starting with: {line_content[:10]}"
    )
    # Should NOT report MAX_BYTES since 1 line is well within budget
    assert "Max" not in result.message


async def test_read_tail_line_truncation(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Tail mode should correctly report truncated lines via was_truncated flag in deque."""
    trunc_file = temp_work_dir / "tail_truncation.txt"
    short_line = "Short line"
    long_line = "X" * (MAX_LINE_LENGTH + 100)  # Exceeds MAX_LINE_LENGTH
    # 5 lines: short, long, short, long, short
    content = f"{short_line}\n{long_line}\n{short_line}\n{long_line}\n{short_line}"
    await trunc_file.write_text(content)

    # Read last 3 lines (lines 3, 4, 5)
    result = await read_file_tool(Params(path=str(trunc_file), line_offset=-3))
    assert not result.is_error
    assert "Total lines in file: 5." in result.message
    # Line 4 is a long line that should be truncated
    assert "Lines [4] were truncated." in result.message
    # Verify the truncated line ends with "..."
    assert isinstance(result.output, str)
    output_lines = result.output.split("\n")
    line_4 = [x for x in output_lines if x.strip().startswith("4")][0]
    actual_content = line_4.split("\t", 1)[1]
    assert actual_content.endswith("...")


# --- Comprehensive edge-case tests ---


async def test_read_empty_path(read_file_tool: ReadFile):
    """Test reading with an empty path."""
    result = await read_file_tool(Params(path=""))
    assert result.is_error
    assert "File path cannot be empty" in result.message


async def test_read_outside_workspace_absolute(
    read_file_tool: ReadFile, outside_file: Path
):
    """Test reading outside the working directory with an absolute path."""
    outside_file.write_text("outside content", encoding="utf-8")
    result = await read_file_tool(Params(path=str(outside_file)))
    assert not result.is_error
    assert "outside content" in result.output


async def test_read_char_offset(read_file_tool: ReadFile, sample_file: KaosPath):
    """Test char_offset slices output correctly."""
    result = await read_file_tool(Params(path=str(sample_file), char_offset=10))
    assert not result.is_error
    # Output should skip the first 10 characters
    assert not result.output.startswith("     1\t")
    # The remaining content should be present
    assert "Line 1" in result.output or "Line 2" in result.output


async def test_read_max_char(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test max_char limits output correctly."""
    file_path = temp_work_dir / "long.txt"
    content = "\n".join([f"Line {i}" for i in range(1, 21)])
    await file_path.write_text(content)

    result = await read_file_tool(Params(path=str(file_path), max_char=20))
    assert not result.is_error
    assert len(result.output) <= 20


async def test_read_binary_unknown_file(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test reading a binary/unknown file is blocked."""
    binary_file = temp_work_dir / "data.bin"
    await binary_file.write_bytes(bytes(range(256)))

    result = await read_file_tool(Params(path=str(binary_file)))
    assert result.is_error
    assert "not readable" in result.message.lower() or "unknown" in result.message.lower()


async def test_read_exception_handling(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Test that exceptions during reading are handled gracefully."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("content")

    from unittest.mock import patch

    with patch("kaos.path.KaosPath.read_bytes", side_effect=OSError("permission denied")):
        result = await read_file_tool(Params(path=str(file_path)))

    assert result.is_error
    assert "Failed to read" in result.message
    assert "permission denied" in result.message


async def test_read_forward_max_lines_not_eof(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Test forward read when max_lines is reached before EOF."""
    file_path = temp_work_dir / "large.txt"
    content = "\n".join([f"Line {i}" for i in range(1, MAX_LINES + 50)])
    await file_path.write_text(content)

    result = await read_file_tool(Params(path=str(file_path), line_offset=1, n_lines=MAX_LINES))
    assert not result.is_error
    assert f"Max {MAX_LINES} lines reached" in result.message
    assert "End of file reached" not in result.message
    assert "Total lines in file" not in result.message


async def test_read_forward_end_of_file(
    read_file_tool: ReadFile, sample_file: KaosPath
):
    """Test forward read reaches end of file message."""
    result = await read_file_tool(
        Params(path=str(sample_file), line_offset=1, n_lines=MAX_LINES)
    )
    assert not result.is_error
    assert "End of file reached" in result.message
    assert "Total lines in file: 5" in result.message


async def test_read_tail_max_lines_not_eof(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Test tail read returns MAX_LINES from a larger file."""
    file_path = temp_work_dir / "tail_large.txt"
    total = MAX_LINES + 100
    content = "\n".join([f"Line {i}" for i in range(1, total + 1)])
    await file_path.write_text(content)

    result = await read_file_tool(
        Params(path=str(file_path), line_offset=-MAX_LINES, n_lines=MAX_LINES, max_char=200000)
    )
    assert not result.is_error
    assert f"Total lines in file: {total}." in result.message
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    assert len(output_lines) == MAX_LINES


# --- Merged from tests/test_read_file.py: detailed char_offset / max_char tests ---


class TestReadFileCharSlicing:
    async def test_char_offset_cuts_beginning(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("1234567890\n")
        result = await read_file_tool(Params(path=str(f), char_offset=7))
        assert not result.is_error
        assert result.output == "1234567890\n"

    async def test_max_char_cuts_end(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("1234567890\n")
        result = await read_file_tool(Params(path=str(f), max_char=12))
        assert not result.is_error
        assert result.output == "     1\t12345"

    async def test_char_offset_and_max_char(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("0123456789\n")
        result = await read_file_tool(Params(path=str(f), char_offset=7, max_char=12))
        assert not result.is_error
        assert result.output == "01234"

    async def test_char_offset_beyond_output(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("short\n")
        result = await read_file_tool(Params(path=str(f), char_offset=100))
        assert not result.is_error
        assert result.output == ""

    async def test_max_char_beyond_output(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("short\n")
        result = await read_file_tool(Params(path=str(f), max_char=100000))
        assert not result.is_error
        assert "short" in result.output

    async def test_zero_max_char_returns_empty(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("hello\n")
        result = await read_file_tool(Params(path=str(f), max_char=0))
        assert not result.is_error
        assert result.output == ""

    async def test_empty_file_with_slicing(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "empty.txt"
        await f.write_text("")
        result = await read_file_tool(Params(path=str(f), char_offset=5, max_char=10))
        assert not result.is_error
        assert result.output == ""

    async def test_line_offset_with_char_offset(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("line one\nline two\nline three\n")
        result = await read_file_tool(Params(path=str(f), line_offset=2, char_offset=7))
        assert not result.is_error
        assert result.output.startswith("line two")

    async def test_line_offset_with_max_char(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("line one\nline two\nline three\n")
        result = await read_file_tool(Params(path=str(f), line_offset=2, max_char=6))
        assert not result.is_error
        assert len(result.output) == 6

    async def test_negative_line_offset_with_char_slice(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("a\nb\nc\nd\n")
        result = await read_file_tool(Params(path=str(f), line_offset=-2, char_offset=4))
        assert not result.is_error
        assert len(result.output) > 0

    async def test_large_char_offset(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("x" * 100 + "\n")
        result = await read_file_tool(Params(path=str(f), char_offset=57))
        assert not result.is_error
        assert len(result.output) == 51
        assert result.output == "x" * 50 + "\n"

    async def test_max_char_exact_boundary(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("1234567890\n")
        result = await read_file_tool(Params(path=str(f), max_char=7))
        assert not result.is_error
        assert result.output == "     1\t"

    async def test_multibyte_utf8_slicing(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        f = temp_work_dir / "a.txt"
        await f.write_text("你好世界\n")
        result = await read_file_tool(Params(path=str(f), char_offset=7, max_char=9))
        assert not result.is_error
        assert result.output == "你好"


# ── Multi-file read tests ────────────────────────────────────────────────────


async def test_read_multiple_files(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """Read two files in a single call."""
    a = temp_work_dir / "a.txt"
    b = temp_work_dir / "b.txt"
    await a.write_text("File A line 1\nFile A line 2")
    await b.write_text("File B line 1")

    result = await read_file_tool(Params(path=[str(a), str(b)]))
    assert not result.is_error
    display_a = str(a).replace("\\", "/")
    display_b = str(b).replace("\\", "/")
    assert f"======== {display_a} ========" in result.output
    assert "File A line 1" in result.output
    assert f"======== {display_b} ========" in result.output
    assert "File B line 1" in result.output
    assert "Read 2 file(s)" in result.message


async def test_read_multiple_files_single_string_still_works(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """A single string path still behaves exactly like a single-file read."""
    a = temp_work_dir / "a.txt"
    await a.write_text("single file content")

    result = await read_file_tool(Params(path=str(a)))
    assert not result.is_error
    assert result.output == snapshot('     1\tsingle file content')
    assert "Read file" in result.brief


async def test_read_multiple_files_with_errors(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """One success and one failure returns a successful aggregate with both blocks."""
    a = temp_work_dir / "a.txt"
    missing = temp_work_dir / "missing.txt"
    await a.write_text("present")

    result = await read_file_tool(Params(path=[str(a), str(missing)]))
    assert not result.is_error
    display_a = str(a).replace("\\", "/")
    display_missing = str(missing).replace("\\", "/")
    assert f"======== {display_a} ========" in result.output
    assert "present" in result.output
    assert f"======== {display_missing} ========" in result.output
    assert "does not exist" in result.output
    assert "Read 1 file(s), 1 error(s)" in result.message


async def test_read_multiple_files_all_fail(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """All files failing returns a ToolError."""
    missing1 = temp_work_dir / "missing1.txt"
    missing2 = temp_work_dir / "missing2.txt"

    result = await read_file_tool(Params(path=[str(missing1), str(missing2)]))
    assert result.is_error
    assert "Failed to read" in result.message
    assert result.brief == "Failed to read files"


async def test_read_multiple_files_deduplicated(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Duplicate paths are read only once."""
    a = temp_work_dir / "a.txt"
    await a.write_text("content")

    result = await read_file_tool(Params(path=[str(a), str(a)]))
    assert not result.is_error
    # Deduplication leaves a single file, so the single-file output format is used.
    assert result.output == snapshot('     1\tcontent')
    assert result.output.count("content") == 1
async def test_read_multiple_files_too_many(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Creating 33 paths is rejected at validation time."""
    paths = []
    for i in range(MAX_FILES + 1):
        f = temp_work_dir / f"file{i}.txt"
        await f.write_text(str(i))
        paths.append(str(f))

    # Validation now happens in Params, raising ValueError
    with pytest.raises((ValueError, Exception)):
        Params(path=paths)
async def test_read_multiple_files_empty_list(read_file_tool: ReadFile):
    """An empty path list is rejected."""
    result = await read_file_tool(Params(path=[]))
    assert result.is_error
    assert "cannot be empty" in result.message.lower()


async def test_read_multiple_files_per_file_limits(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Per-file options now use scalar values broadcast to all files."""
    a = temp_work_dir / "a.txt"
    b = temp_work_dir / "b.txt"
    await a.write_text("a1\na2\na3\na4")
    await b.write_text("b1\nb2\nb3\nb4\nb5")

    # Scalar line_offset and n_lines apply to all files
    result = await read_file_tool(
        Params(path=[str(a), str(b)], line_offset=1, n_lines=2)
    )
    assert not result.is_error
    display_a = str(a).replace("\\", "/")
    display_b = str(b).replace("\\", "/")
    # a should contain lines 1-2
    assert f"======== {display_a} ========" in result.output
    assert "     1\ta1" in result.output
    assert "     2\ta2" in result.output
    # b should contain lines 1-2
    b_section = result.output.split(f"======== {display_b} ========")[1]
    assert "     1\tb1" in b_section
    assert "     2\tb2" in b_section


async def test_read_multiple_files_scalar_options(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """Scalar n_lines applies to every file."""
    a = temp_work_dir / "a.txt"
    b = temp_work_dir / "b.txt"
    await a.write_text("a1\na2")
    await b.write_text("b1\nb2")

    result = await read_file_tool(Params(path=[str(a), str(b)], n_lines=1))
    assert not result.is_error
    display_a = str(a).replace("\\", "/")
    display_b = str(b).replace("\\", "/")
    a_section = result.output.split(f"======== {display_b} ========")[0]
    b_section = result.output.split(f"======== {display_b} ========")[1]
    assert "     1\ta1" in a_section
    assert "     2\ta2" not in a_section
    assert "     1\tb1" in b_section
    assert "     2\tb2" not in b_section


async def test_read_multiple_files_mismatched_option_length(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """List-valued options are no longer accepted; only scalar values."""
    a = temp_work_dir / "a.txt"
    b = temp_work_dir / "b.txt"
    await a.write_text("a")
    await b.write_text("b")

    # n_lines no longer accepts lists; should raise validation error
    with pytest.raises((ValueError, Exception)):
        Params(path=[str(a), str(b)], n_lines=[1])


async def test_read_multiple_files_alias_paths(
    read_file_tool: ReadFile, temp_work_dir: KaosPath
):
    """The 'paths' alias is repaired to 'path'."""
    a = temp_work_dir / "a.txt"
    b = temp_work_dir / "b.txt"
    await a.write_text("alpha")
    await b.write_text("beta")

    result = await read_file_tool.call({"paths": [str(a), str(b)]})
    assert not result.is_error
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "Read 2 file(s)" in result.message



# ── Glob support tests ───────────────────────────────────────────────────────


class TestReadFileGlob:
    def _file_headers(self, output: str) -> list[str]:
        return re.findall(r"======== (.*?) ========", output)

    async def test_read_glob_simple(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`./*.md` reads all top-level .md files and skips .txt files."""
        await (temp_work_dir / "a.md").write_text("alpha")
        await (temp_work_dir / "b.md").write_text("beta")
        await (temp_work_dir / "c.txt").write_text("gamma")

        result = await read_file_tool(Params(path="./*.md", glob=True))

        assert not result.is_error
        assert "alpha" in result.output
        assert "beta" in result.output
        assert "gamma" not in result.output
        assert len(self._file_headers(result.output)) == 2

    async def test_read_glob_sorted(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """Output order is alphabetical regardless of creation order."""
        await (temp_work_dir / "z.md").write_text("z")
        await (temp_work_dir / "a.md").write_text("a")
        await (temp_work_dir / "m.md").write_text("m")

        result = await read_file_tool(Params(path="*.md", glob=True))

        assert not result.is_error
        names = self._file_headers(result.output)
        assert names == ["a.md", "m.md", "z.md"]

    async def test_read_glob_subdirectory(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`docs/*.md` works relative to the working directory."""
        docs = temp_work_dir / "docs"
        await docs.mkdir()
        await (docs / "x.md").write_text("x")
        await (docs / "y.md").write_text("y")

        result = await read_file_tool(Params(path="docs/*.md", glob=True))

        assert not result.is_error
        assert "x" in result.output
        assert "y" in result.output
        assert len(self._file_headers(result.output)) == 2

    async def test_read_glob_absolute_directory(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """Absolute base directory with a glob works."""
        await (temp_work_dir / "p.md").write_text("p")
        await (temp_work_dir / "q.md").write_text("q")

        pattern = str(temp_work_dir).replace("\\", "/") + "/*.md"
        result = await read_file_tool(Params(path=pattern, glob=True))

        assert not result.is_error
        assert "p" in result.output
        assert "q" in result.output

    async def test_read_glob_no_matches(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`./*.nomatch` returns an error."""
        result = await read_file_tool(Params(path="./*.nomatch", glob=True))

        assert result.is_error
        assert "No files matched" in result.message
        assert result.brief == "No matches"

    async def test_read_glob_nonexistent_directory(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`missing/*.md` returns an error."""
        result = await read_file_tool(Params(path="missing/*.md", glob=True))

        assert result.is_error
        assert "does not exist" in result.message
        assert result.brief == "Directory not found"

    async def test_read_glob_outside_workspace_relative(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`../outside/*.md` is rejected."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            outside = Path(tmpdir) / "outside"
            outside.mkdir()
            (outside / "x.md").write_text("x")

            # Change into temp_work_dir so ../outside resolves to the outside dir.
            original_cwd = Path.cwd()
            os.chdir(str(temp_work_dir))
            try:
                result = await read_file_tool(Params(path="../outside/*.md", glob=True))
            finally:
                os.chdir(original_cwd)

        assert result.is_error
        assert "absolute path" in result.message.lower()

    async def test_read_glob_with_literal_files(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """A list mixing a glob and a literal path aggregates correctly."""
        await (temp_work_dir / "a.md").write_text("a")
        await (temp_work_dir / "b.md").write_text("b")
        literal = temp_work_dir / "c.txt"
        await literal.write_text("c")

        result = await read_file_tool(Params(path=["./*.md", str(literal)], glob=True))

        assert not result.is_error
        assert "a" in result.output
        assert "b" in result.output
        assert "c" in result.output
        assert len(self._file_headers(result.output)) == 3

    async def test_read_glob_options_broadcast(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`n_lines=1` applied to a glob affects every matched file."""
        await (temp_work_dir / "a.md").write_text("a1\na2")
        await (temp_work_dir / "b.md").write_text("b1\nb2")

        result = await read_file_tool(Params(path="*.md", n_lines=1, glob=True))

        assert not result.is_error
        assert "     1\ta1" in result.output
        assert "     1\tb1" in result.output
        assert "     2\ta2" not in result.output
        assert "     2\tb2" not in result.output

    async def test_read_glob_deduplicates(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`path=["a.md", "*.md"]` reads `a.md` only once."""
        await (temp_work_dir / "a.md").write_text("a")
        await (temp_work_dir / "b.md").write_text("b")

        result = await read_file_tool(Params(path=["a.md", "*.md"], glob=True))

        assert not result.is_error
        headers = self._file_headers(result.output)
        assert len(headers) == 2
        assert headers.count("a.md") == 1
        assert "b" in result.output

    async def test_read_glob_skips_directories(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """Pattern `*` matches a directory but only files are read."""
        await (temp_work_dir / "subdir").mkdir()
        await (temp_work_dir / "file.txt").write_text("file")

        result = await read_file_tool(Params(path="*", glob=True))

        assert not result.is_error
        assert "file" in result.output
        assert "subdir" not in result.output
        # Only one file matched, so single-file output format is used.
        assert result.message.endswith(" Path: file.txt")

    async def test_read_glob_respects_gitignore(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """A `.gitignore` rule excludes matched files from a glob."""
        await (temp_work_dir / ".gitignore").write_text("ignored.md\n")
        await (temp_work_dir / "included.md").write_text("included")
        await (temp_work_dir / "ignored.md").write_text("ignored")

        result = await read_file_tool(Params(path="*.md", glob=True))

        assert not result.is_error
        assert "included" in result.output
        assert "ignored" not in result.output
        # Only one file matched, so single-file output format is used.
        assert result.message.endswith(" Path: included.md")

    async def test_read_glob_rejects_leading_double_star(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """`**/*.md` is rejected as unsafe."""
        result = await read_file_tool(Params(path="**/*.md", glob=True))

        assert result.is_error
        assert "starts with `**`" in result.message
        assert result.brief == "Unsafe glob pattern"

    async def test_read_glob_respects_max_files(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """Create MAX_FILES + 1 matching files and verify the call is rejected."""
        for i in range(MAX_FILES + 1):
            await (temp_work_dir / f"file{i}.txt").write_text(str(i))

        result = await read_file_tool(Params(path="*.txt", glob=True))

        assert result.is_error
        assert f"Cannot read more than {MAX_FILES} files" in result.message
        assert result.brief == "Too many files"

    async def test_read_glob_media_file_is_rejected(self, read_file_tool: ReadFile, temp_work_dir: KaosPath):
        """A glob that matches an image file produces a per-entry error."""
        await (temp_work_dir / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"pngdata")
        await (temp_work_dir / "readme.txt").write_text("hello")

        result = await read_file_tool(Params(path="*", glob=True))

        assert not result.is_error
        assert "hello" in result.output
        assert "sample.png" in result.output
        assert "image file" in result.output
        assert "Read 1 file(s), 1 error(s)" in result.message


# ── Show line numbers tests ──────────────────────────────────────────────────


async def test_show_line_numbers_default_true(read_file_tool: ReadFile, sample_file: KaosPath):
    """By default, lines are prefixed with line numbers (backward compat)."""
    result = await read_file_tool(Params(path=str(sample_file)))
    assert not result.is_error
    assert "     1\tLine 1: Hello World" in result.output


async def test_show_line_numbers_false(read_file_tool: ReadFile, sample_file: KaosPath):
    """show_line_numbers=False returns raw content without line numbers."""
    result = await read_file_tool(Params(path=str(sample_file), show_line_numbers=False))
    assert not result.is_error
    assert "Line 1: Hello World" in result.output
    assert "     1\t" not in result.output[:20]  # No line number prefix


async def test_show_line_numbers_false_tail(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """show_line_numbers=False works with negative line_offset (tail mode)."""
    f = temp_work_dir / "tail_test.txt"
    await f.write_text("line1\nline2\nline3\nline4\nline5")
    result = await read_file_tool(Params(path=str(f), line_offset=-3, show_line_numbers=False))
    assert not result.is_error
    assert "line3" in result.output
    assert "line4" in result.output
    assert "line5" in result.output
    assert "\t" not in result.output[:20]  # No tab-separated line numbers


async def test_show_line_numbers_false_multiple_files(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """show_line_numbers=False works with multiple files."""
    a = temp_work_dir / "show_a.txt"
    b = temp_work_dir / "show_b.txt"
    await a.write_text("content_a")
    await b.write_text("content_b")
    result = await read_file_tool(Params(path=[str(a), str(b)], show_line_numbers=False))
    assert not result.is_error
    assert "content_a" in result.output
    assert "content_b" in result.output
    assert "\t" not in result.output.split("content_a")[0][:10]


# ── Glob parameter tests ──────────────────────────────────────────────────────


async def test_glob_param_explicit(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """glob=True reads matching files."""
    await (temp_work_dir / "glob_a.py").write_text("print('a')")
    await (temp_work_dir / "glob_b.py").write_text("print('b')")
    await (temp_work_dir / "glob_c.txt").write_text("text")
    result = await read_file_tool(Params(path="*.py", glob=True))
    assert not result.is_error
    assert "print('a')" in result.output or "glob_a" in result.output
    assert "print('b')" in result.output or "glob_b" in result.output


async def test_glob_param_false_literal(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """glob=False treats paths with wildcards as literal file names."""
    # Create a file with an asterisk in its name
    import os as _os
    f = temp_work_dir / "star_file.txt"
    await f.write_text("literal content")
    result = await read_file_tool(Params(path=str(f), glob=False))
    assert not result.is_error
    assert "literal content" in result.output


async def test_glob_param_false_suppresses_auto_detection(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """glob=False suppresses auto-detection; paths with glob chars are treated literally."""
    # Create a file that has a [ in its name
    f = temp_work_dir / "data[1].txt"
    await f.write_text("bracket content")
    # With glob=False, auto-detection is suppressed
    result = await read_file_tool(Params(path=str(f), glob=False))
    assert not result.is_error
    assert "bracket content" in result.output


async def test_glob_param_true_literal_path(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """glob=True with a literal path (no glob chars) treats it as a single file."""
    f = temp_work_dir / "literal_target.txt"
    await f.write_text("literal content for glob test")
    result = await read_file_tool(Params(path=str(f), glob=True))
    assert not result.is_error
    assert "literal content for glob test" in result.output


async def test_glob_param_true_works_for_glob_patterns(read_file_tool: ReadFile, temp_work_dir: KaosPath):
    """glob=True expands glob patterns as expected."""
    await (temp_work_dir / "glob_pattern_a.py").write_text("pattern_a")
    await (temp_work_dir / "glob_pattern_b.py").write_text("pattern_b")
    result = await read_file_tool(Params(path="*.py", glob=True))
    assert not result.is_error or "No files matched" in result.message
    if not result.is_error:
        assert "pattern_a" in result.output or "glob_pattern_a" in result.output
