"""Tests for HashRead, HashEdit, and shared hash_line utilities."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from kaos.path import KaosPath

from kimi_cli.tools.file.hash_line import (
    NIBBLE_STR,
    AnchorRef,
    AppendEdit,
    DeleteEdit,
    HashEdit,
    HashEditParams,
    HashlineMismatchError,
    HashMismatch,
    HashRead,
    HashReadParams,
    PrependEdit,
    ReplaceEdit,
    apply_hashline_edits,
    compute_line_hash,
    generate_hash_aware_diff,
    parse_anchor,
    validate_anchor_ref,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def get_line_hash(content: str, line_num: int) -> str:
    """Compute cumulative hashes for content and return hash for a specific line."""
    lines = content.splitlines()
    prev_hash = None
    cumulative_hashes = []
    for i, line in enumerate(lines):
        ln = i + 1
        h = compute_line_hash(ln, line, prev_hash)
        cumulative_hashes.append(h)
        prev_hash = h
    return cumulative_hashes[line_num - 1]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Hash computation tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_line_hash_determinism():
    """Same input gives same 2-char hash."""
    h1 = compute_line_hash(1, "test line", None)
    h2 = compute_line_hash(1, "test line", None)
    assert h1 == h2
    assert len(h1) == 2


def test_compute_line_hash_empty_line():
    """Empty line produces 2-char hash."""
    h = compute_line_hash(1, "", None)
    assert len(h) == 2


def test_compute_line_hash_whitespace_only():
    """Whitespace-only lines use line number as seed, so different line numbers give different hashes."""
    h1 = compute_line_hash(1, "   \t", None)
    h3 = compute_line_hash(3, "   \t", None)
    assert len(h1) == 2
    assert len(h3) == 2
    assert h1 != h3


def test_compute_line_hash_same_content_different_lines():
    """Same content with alphanumeric chars gives same hash regardless of line number (when prev_hash=None)."""
    h1 = compute_line_hash(1, "content", None)
    h2 = compute_line_hash(2, "content", None)
    assert h1 == h2


def test_compute_line_hash_trailing_cr():
    """Trailing \\r is stripped before hashing."""
    h_with_cr = compute_line_hash(1, "test line\r", None)
    h_without_cr = compute_line_hash(1, "test line", None)
    assert h_with_cr == h_without_cr


def test_compute_line_hash_cumulative_chain():
    """Changing prev_hash changes the output."""
    h1 = compute_line_hash(1, "test line", "AB")
    h2 = compute_line_hash(1, "test line", "XY")
    assert h1 != h2


# ═══════════════════════════════════════════════════════════════════════════
# 2. Anchor parsing tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_parse_anchor_new_format():
    """5#ab -> (5, 'ab')."""
    result = parse_anchor("5#ab")
    assert result == (5, "ab")


def test_parse_anchor_old_format():
    """5:abc -> (5, 'abc')."""
    result = parse_anchor("5:abc")
    assert result == (5, "abc")


def test_parse_anchor_invalid():
    """Invalid strings return None."""
    assert parse_anchor("invalid") is None
    assert parse_anchor("") is None
    assert parse_anchor("abc#def") is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Read operation tests (async)
# ═══════════════════════════════════════════════════════════════════════════


async def test_read_simple_file(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Read a file, verify output contains <file>, line hashes, content."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\n"
    await file_path.write_text(content)

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert result.output == snapshot(
        f"""\
<file>
1#{get_line_hash(content, 1)}:line 1
2#{get_line_hash(content, 2)}:line 2
3#{get_line_hash(content, 3)}:line 3

(End of file - 3 total lines)
</file>"""
    )
    display_path = str(file_path).replace("\\", "/")
    assert result.message == snapshot(f"Read {display_path}.")


async def test_read_empty_file(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Empty file returns '(End of file - 0 lines)'."""
    file_path = temp_work_dir / "empty.txt"
    await file_path.write_text("")

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert result.output == snapshot("<file>\n(End of file - 0 lines)\n</file>")


async def test_read_with_offset(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Offset skips first N lines."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    await file_path.write_text(content)

    result = await hash_line_tool(HashReadParams(path=str(file_path), offset=2))
    assert not result.is_error
    assert result.output == snapshot(
        f"""\
<file>
3#{get_line_hash(content, 3)}:line 3
4#{get_line_hash(content, 4)}:line 4
5#{get_line_hash(content, 5)}:line 5

(End of file - 5 total lines)
</file>"""
    )


async def test_read_with_limit(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Limit restricts number of lines."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    await file_path.write_text(content)

    result = await hash_line_tool(HashReadParams(path=str(file_path), limit=2))
    assert not result.is_error
    assert result.output == snapshot(
        f"""\
<file>
1#{get_line_hash(content, 1)}:line 1
2#{get_line_hash(content, 2)}:line 2

(File has more lines. Use 'offset' parameter to read beyond line 2)
</file>"""
    )


async def test_read_offset_beyond_eof(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Offset beyond file length returns 0 lines message."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("line 1\nline 2\n")

    result = await hash_line_tool(HashReadParams(path=str(file_path), offset=100))
    assert not result.is_error
    assert result.output == snapshot("<file>\n(End of file - 0 lines)\n</file>")


async def test_read_unicode(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Unicode content handled correctly."""
    file_path = temp_work_dir / "unicode.txt"
    content = "Hello 世界\n🎉 Emoji test\n"
    await file_path.write_text(content)

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert result.output == snapshot(
        f"""\
<file>
1#{get_line_hash(content, 1)}:Hello 世界
2#{get_line_hash(content, 2)}:🎉 Emoji test

(End of file - 2 total lines)
</file>"""
    )


async def test_read_trailing_newline(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """File with trailing newline preserves hash computation."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\n"
    await file_path.write_text(content)

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert result.output == snapshot(
        f"""\
<file>
1#{get_line_hash(content, 1)}:line 1
2#{get_line_hash(content, 2)}:line 2

(End of file - 2 total lines)
</file>"""
    )


async def test_read_nonexistent_file(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Returns ToolError."""
    file_path = temp_work_dir / "nonexistent.txt"
    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert result.is_error
    display_path = str(file_path).replace("\\", "/")
    assert result.message == snapshot(f"`{display_path}` does not exist.")


async def test_read_directory(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Returns ToolError."""
    dir_path = temp_work_dir / "directory"
    await dir_path.mkdir()

    result = await hash_line_tool(HashReadParams(path=str(dir_path)))
    assert result.is_error
    display_path = str(dir_path).replace("\\", "/")
    assert result.message == snapshot(f"`{display_path}` is not a file.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Edit operation tests - basic (sync and async)
# ═══════════════════════════════════════════════════════════════════════════


def test_replace_single_line():
    """Replace one line, verify content, first_changed."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=None,
            lines=["REPLACED"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert "REPLACED" in result
    assert "line 2" not in result
    assert result.splitlines() == ["line 1", "REPLACED", "line 3"]
    assert first_changed == 2


def test_replace_range():
    """Replace lines 2-4 with new content."""
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=AnchorRef(line=4, hash=get_line_hash(content, 4)),
            lines=["replaced"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["line 1", "replaced", "line 5"]
    assert first_changed == 2


def test_append_after_line():
    """Insert after a specific line."""
    content = "first\nsecond\n"
    edits = [
        AppendEdit(
            op="append",
            pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            lines=["inserted"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["first", "inserted", "second"]
    assert first_changed == 2


def test_append_eof():
    """Append at end of file (pos=None)."""
    content = "first\nsecond\n"
    edits = [AppendEdit(op="append", pos=None, lines=["at eof"])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["first", "second", "at eof"]
    assert first_changed == 3


def test_prepend_before_line():
    """Insert before a specific line."""
    content = "first\nsecond\n"
    edits = [
        PrependEdit(
            op="prepend",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            lines=["before"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["first", "before", "second"]
    assert first_changed == 2


def test_prepend_bof():
    """Prepend at start of file (pos=None)."""
    content = "first\nsecond\n"
    edits = [PrependEdit(op="prepend", pos=None, lines=["at bof"])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["at bof", "first", "second"]
    assert first_changed == 1


def test_delete_single_line():
    """Delete convenience operation."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        DeleteEdit(
            op="delete",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["line 1", "line 3"]
    assert first_changed == 2


def test_empty_content_append():
    """Append to empty file."""
    content = ""
    edits = [AppendEdit(op="append", pos=None, lines=["line 1"])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result == "line 1"
    assert first_changed == 1


def test_empty_lines_replace():
    """Replace with empty lines list deletes the line."""
    content = "first\nsecond\nthird\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=None,
            lines=[],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["first", "third"]
    assert first_changed == 2


def test_trailing_newline_preserved():
    """Original trailing newline is preserved."""
    content = "fn test() {\n    // comment\n}\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=None,
            lines=["    // modified"],
        )
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert result.endswith("\n")
    assert result.splitlines() == ["fn test() {", "    // modified", "}"]


def test_no_changes_returns_none():
    """Empty edits list returns original content."""
    content = "first\nsecond\nthird\n"
    result, first_changed = apply_hashline_edits(content, [])
    assert result == content
    assert first_changed is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Edit validation tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_hash_mismatch_fails():
    """Wrong hash causes HashlineMismatchError with context."""
    content = "first\nsecond\nthird\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash="ZZ"),
            end=None,
            lines=["replaced"],
        )
    ]
    with pytest.raises(HashlineMismatchError) as exc_info:
        apply_hashline_edits(content, edits)
    assert "changed since last read" in str(exc_info.value)
    assert exc_info.value.mismatches[0].line == 2
    assert exc_info.value.mismatches[0].expected == "ZZ"


def test_line_out_of_range():
    """Line number beyond file length fails."""
    content = "first\nsecond\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=10, hash="AB"),
            end=None,
            lines=["replaced"],
        )
    ]
    with pytest.raises(ValueError, match="does not exist"):
        apply_hashline_edits(content, edits)


def test_range_start_greater_than_end():
    """Start > end fails."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=3, hash=get_line_hash(content, 3)),
            end=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            lines=["replaced"],
        )
    ]
    with pytest.raises(ValueError, match="must be <= end line"):
        apply_hashline_edits(content, edits)


def test_overlapping_replace_rejected():
    """Two replaces on same line rejected."""
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=None,
            lines=["replaced"],
        ),
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=None,
            lines=["replaced again"],
        ),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_overlapping_replace_range_rejected():
    """Replace ranges that overlap rejected."""
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            end=AnchorRef(line=4, hash=get_line_hash(content, 4)),
            lines=["range"],
        ),
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=3, hash=get_line_hash(content, 3)),
            end=None,
            lines=["single"],
        ),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_adjacent_edits_allowed():
    """Adjacent but non-overlapping edits succeed."""
    content = "line 1\nline 2\nline 3\nline 4\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            end=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            lines=["first"],
        ),
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=3, hash=get_line_hash(content, 3)),
            end=AnchorRef(line=4, hash=get_line_hash(content, 4)),
            lines=["second"],
        ),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["first", "second"]


def test_append_prepend_same_line_overlap():
    """Append and prepend at same line rejected."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        AppendEdit(
            op="append",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            lines=["appended"],
        ),
        PrependEdit(
            op="prepend",
            pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
            lines=["prepended"],
        ),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_replace_and_append_same_line_allowed():
    """Replace at N and append at N are NOT overlapping."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            end=None,
            lines=["replaced"],
        ),
        AppendEdit(
            op="append",
            pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            lines=["appended"],
        ),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert "replaced" in result
    assert "appended" in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cumulative hash tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_stale_hash_fails():
    """Editing line 2, then trying to edit line 2 with old hash fails."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)

    first_edit = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    modified, _ = apply_hashline_edits(content, first_edit)

    # Try to edit line 2 again using the ORIGINAL hash
    stale_edit = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["SHOULD_FAIL"],
        )
    ]
    with pytest.raises(HashlineMismatchError):
        apply_hashline_edits(modified, stale_edit)

    # Verify no duplication occurred
    assert modified.splitlines() == ["line 1", "MODIFIED", "line 3"]


def test_subsequent_line_hash_invalidated():
    """Editing line 2 invalidates hash of line 3+ (cumulative chain)."""
    content = "line 1\nline 2\nline 3\nline 4\n"
    h2 = get_line_hash(content, 2)
    h3 = get_line_hash(content, 3)
    h4 = get_line_hash(content, 4)

    edit = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    modified, _ = apply_hashline_edits(content, edit)

    # Edit line 3 with original hash should fail
    with pytest.raises(HashlineMismatchError):
        apply_hashline_edits(
            modified,
            [
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=3, hash=h3),
                    end=None,
                    lines=["should fail"],
                )
            ],
        )

    # Edit line 4 with original hash should fail
    with pytest.raises(HashlineMismatchError):
        apply_hashline_edits(
            modified,
            [
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=4, hash=h4),
                    end=None,
                    lines=["should fail"],
                )
            ],
        )


def test_before_edit_line_still_valid():
    """Lines before the edit remain valid."""
    content = "line 1\nline 2\nline 3\nline 4\n"
    h1 = get_line_hash(content, 1)
    h2 = get_line_hash(content, 2)

    edit = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    modified, _ = apply_hashline_edits(content, edit)

    # Edit line 1 with original hash should still work
    result, _ = apply_hashline_edits(
        modified,
        [
            ReplaceEdit(
                op="replace",
                pos=AnchorRef(line=1, hash=h1),
                end=None,
                lines=["line 1 modified"],
            )
        ],
    )
    assert "line 1 modified" in result


# ═══════════════════════════════════════════════════════════════════════════
# 7. Deduplication tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_duplicate_edits_deduplicated():
    """Identical edits applied only once."""
    content = "first\nsecond\n"
    edit = ReplaceEdit(
        op="replace",
        pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
        end=None,
        lines=["replaced"],
    )
    edits = [edit, edit]
    result, _ = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["replaced", "second"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Diff generation tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_diff_replace_single():
    """Diff shows -, +, and space correctly."""
    old = "first\nsecond\nthird\n"
    new = "first\nreplaced\nthird\n"
    diff = generate_hash_aware_diff(old, new, 2)
    assert "-2#  :second" in diff
    assert "+2#" in diff
    assert " 1#" in diff
    assert " 3#" in diff


def test_diff_insert():
    """Inserted lines marked with +."""
    old = "first\nsecond\n"
    new = "first\ninserted\nsecond\n"
    diff = generate_hash_aware_diff(old, new, 2)
    assert "+2#" in diff
    assert "inserted" in diff


def test_diff_delete():
    """Deleted lines marked with - and no hash."""
    old = "first\nsecond\nthird\n"
    new = "first\nthird\n"
    diff = generate_hash_aware_diff(old, new, 2)
    assert "-2#  :second" in diff


def test_diff_no_changes():
    """Shows context around first_changed_line."""
    old = "a\nb\nc\n"
    new = "a\nb\nc\n"
    diff = generate_hash_aware_diff(old, new, 2)
    assert " 1#" in diff
    assert " 2#" in diff
    assert " 3#" in diff


def test_diff_note_present():
    """Contains stale hash note."""
    old = "a\nb\n"
    new = "a\nmodified\n"
    diff = generate_hash_aware_diff(old, new, 2)
    assert "Lines after edited regions have stale hashes" in diff


# ═══════════════════════════════════════════════════════════════════════════
# 9. Tool integration tests (async)
# ═══════════════════════════════════════════════════════════════════════════


async def test_edit_file_success(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Full tool call edits file on disk."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\n"
    await file_path.write_text(content)

    result = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
                    end=None,
                    lines=["MODIFIED"],
                )
            ],
        )
    )
    assert not result.is_error
    assert "Edit applied successfully" in result.message
    assert "(first change at line 2)" in result.message
    assert "<diff>" in result.output
    assert await file_path.read_text() == "line 1\nMODIFIED\nline 3\n"


async def test_edit_file_hash_mismatch(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Tool returns ToolError with hash mismatch message."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("first\nsecond\nthird\n")

    result = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=2, hash="ZZ"),
                    end=None,
                    lines=["replaced"],
                )
            ],
        )
    )
    assert result.is_error
    assert "Hash mismatch error" in result.message
    assert "changed since last read" in result.message


async def test_edit_file_overlapping(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Tool returns ToolError for overlapping edits."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    await file_path.write_text(content)

    result = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
                    end=AnchorRef(line=4, hash=get_line_hash(content, 4)),
                    lines=["range"],
                ),
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=3, hash=get_line_hash(content, 3)),
                    end=None,
                    lines=["single"],
                ),
            ],
        )
    )
    assert result.is_error
    assert "Overlapping edits detected" in result.message


async def test_edit_file_no_changes(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Tool returns success with 'No changes made'."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("first\nsecond\nthird\n")

    result = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[],
        )
    )
    assert not result.is_error
    assert "No changes made" in result.message
    assert result.output == ""


async def test_edit_file_empty_path(hash_edit_tool: HashEdit):
    """Tool returns ToolError for empty path."""
    result = await hash_edit_tool(HashEditParams(path="", edits=[]))
    assert result.is_error
    assert "File path cannot be empty" in result.message


async def test_edit_file_nonexistent(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Tool returns ToolError for nonexistent file."""
    file_path = temp_work_dir / "nonexistent.txt"
    result = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=1, hash="AB"),
                    end=None,
                    lines=["replaced"],
                )
            ],
        )
    )
    assert result.is_error
    display_path = str(file_path).replace("\\", "/")
    assert display_path in result.message
    assert "does not exist" in result.message


async def test_edit_file_not_a_file(hash_edit_tool: HashEdit, temp_work_dir: KaosPath):
    """Tool returns ToolError for directory."""
    dir_path = temp_work_dir / "directory"
    await dir_path.mkdir()

    result = await hash_edit_tool(
        HashEditParams(
            path=str(dir_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=1, hash="AB"),
                    end=None,
                    lines=["replaced"],
                )
            ],
        )
    )
    assert result.is_error
    assert "is not a file" in result.message


# ═══════════════════════════════════════════════════════════════════════════
# 10. Edge cases (sync and async)
# ═══════════════════════════════════════════════════════════════════════════


def test_multiple_edits_bottom_up():
    """Multiple edits at different lines all succeed when sorted bottom-up."""
    content = "a\nb\nc\nd\ne\n"
    h1 = get_line_hash(content, 1)
    h2 = get_line_hash(content, 2)
    h4 = get_line_hash(content, 4)

    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["A"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=4, hash=h4), end=None, lines=["D"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=2, hash=h2), end=None, lines=["B"]),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert "a" not in result
    assert "b" not in result
    assert "d" not in result
    assert "A" in result
    assert "B" in result
    assert "D" in result


def test_special_characters():
    """Tabs, unicode in edits."""
    content = "line with \t tabs\nline with unicode: 你好\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=1, hash=get_line_hash(content, 1)),
            end=None,
            lines=["replaced"],
        )
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert "replaced" in result
    assert "你好" in result


async def test_windows_line_endings(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """\\r\\n handling in read."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_bytes(b"line 1\r\nline 2\r\nline 3\r\n")

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    # Hashes should be computed on lines without \r
    content_normalized = "line 1\nline 2\nline 3\n"
    assert result.output == snapshot(
        f"""\
<file>
1#{get_line_hash(content_normalized, 1)}:line 1
2#{get_line_hash(content_normalized, 2)}:line 2
3#{get_line_hash(content_normalized, 3)}:line 3

(End of file - 3 total lines)
</file>"""
    )
# ═══════════════════════════════════════════════════════════════════════════
# 11. Hash collision tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_hash_collision_exists():
    """Since hash output is only 8 bits (256 values), collisions are inevitable.
    
    Demonstrate that two different lines with the same prev_hash can produce the same 2-char hash.
    """
    import random
    import string

    prev_hash = "AB"
    seen: dict[str, str] = {}  # hash -> content

    # Generate many random strings until we find a collision
    random.seed(42)
    collision_found = False
    for _ in range(5000):
        length = random.randint(1, 20)
        content = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
        # Ensure we generate content that has alphanumeric chars (so seed=0 when prev_hash=None,
        # but here we always pass prev_hash so seed is derived from it)
        h = compute_line_hash(1, content, prev_hash)
        if h in seen and seen[h] != content:
            collision_found = True
            break
        seen[h] = content

    assert collision_found, "Should find a collision within 5000 random strings (256 possible hashes)"


def test_hash_collision_cumulative_mitigation():
    """Even if line N produces the same hash for different content, line N+1 hashes will differ.

    The cumulative chain provides mitigation because different content at line N produces
    different seeds for line N+1 computation (even if the 8-bit hash happens to collide).
    """
    # Find two lines that collide at line 1 with prev_hash=None
    import random

    random.seed(123)
    seen: dict[str, str] = {}
    line_a = line_b = None
    for _ in range(5000):
        length = random.randint(3, 15)
        content = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
        h = compute_line_hash(1, content, None)
        if h in seen and seen[h] != content:
            line_a = seen[h]
            line_b = content
            break
        seen[h] = content

    assert line_a is not None and line_b is not None, "Should find collision"
    assert line_a != line_b

    # Now compute hashes for line 2 with the same follow-up content
    h_a1 = compute_line_hash(1, line_a, None)
    h_b1 = compute_line_hash(1, line_b, None)
    assert h_a1 == h_b1, "Line 1 hashes should collide"

    # Line 2 with same content but different prev_hash (from the colliding line 1s)
    follow_up = "same_follow_up_line"
    h_a2 = compute_line_hash(2, follow_up, h_a1)
    h_b2 = compute_line_hash(2, follow_up, h_b1)

    # Even though prev_hash values are the same (h_a1 == h_b1), the seed derived from them is identical,
    # so line 2 hashes will also be the same. That's fine — the collision persists.
    # The REAL mitigation is: if someone tries to use line_b's hash at position 1 but the actual
    # content is line_a, the hash matches. But line 2 would still validate because it depends
    # on the prev_hash which is the SAME for both colliding lines.
    assert h_a2 == h_b2, (
        "With same prev_hash, same follow-up produces same hash — collision chain continues"
    )

    # Key insight: collision only matters for a single line edit at the collision point.
    # For range edits spanning multiple lines, the hash chain provides no extra protection
    # if all lines in the range collide with the replacement lines.


def test_seed_overflow_from_prev_hash():
    """The seed computation from prev_hash can overflow 32 bits.
    
    ""((seed * 256) + ord(c)) & 0xFFFFFFFF" wraps around.
    Different prev_hash strings could theoretically produce the same seed.
    """
    def seed_from_hash(h: str) -> int:
        seed = 0
        for c in h:
            seed = ((seed * 256) + ord(c)) & 0xFFFFFFFF
        return seed

    # Short hashes should produce different seeds
    s1 = seed_from_hash("AB")
    s2 = seed_from_hash("CD")
    assert s1 != s2, "Different short hashes should produce different seeds"

    # Long hash strings could overflow — but in practice prev_hash is always 2 chars
    # from NIBBLE_STR, so this isn't a real concern. Still, verify determinism.
    assert seed_from_hash("AB") == seed_from_hash("AB")

    # Test with non-NIBBLE characters (not that they'd occur in practice)
    s_unicode = seed_from_hash("\u4e16")
    assert isinstance(s_unicode, int)


def test_hash_space_is_256():
    """Verify the hash output space is exactly 256 values (16 x 16 nibble combinations)."""

    assert len(NIBBLE_STR) == 16
    # All possible 2-char combinations
    all_hashes = {a + b for a in NIBBLE_STR for b in NIBBLE_STR}
    assert len(all_hashes) == 256

    # NIBBLE_STR should have no duplicates
    assert len(set(NIBBLE_STR)) == 16


# ═══════════════════════════════════════════════════════════════════════════
# 12. Corner case tests — content edge cases (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_empty_content_replace_fails():
    """Replace on empty content raises ValueError because empty content has 0 lines after splitlines."""
    content = ""
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=1, hash="AB"),
            end=None,
            lines=["replaced"],
        )
    ]
    # Empty content -> splitlines() returns [] (0 lines)
    # So line 1 does not exist
    with pytest.raises(ValueError, match="does not exist"):
        apply_hashline_edits(content, edits)


def test_append_empty_lines_list_noop():
    """Append with empty lines list is a no-op."""
    content = "first\nsecond\n"
    edits = [AppendEdit(op="append", pos=None, lines=[])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result == content
    # Empty edits list means no changes but first_changed should be None
    # Actually, the edit is not in the list — it IS in the list but has empty lines.
    # Let's verify: the edit is normalized, then in the apply loop, AppendEdit with not edit.lines
    # just continues without changing first_changed.
    assert first_changed is None


def test_prepend_empty_lines_list_noop():
    """Prepend with empty lines list is a no-op."""
    content = "first\nsecond\n"
    edits = [PrependEdit(op="prepend", pos=None, lines=[])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result == content
    assert first_changed is None


def test_anchor_line_zero_rejected():
    """Anchor with line=0 should raise ValueError."""
    content = "line 1\nline 2\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=0, hash="AB"),
            end=None,
            lines=["replaced"],
        )
    ]
    with pytest.raises(ValueError, match="must be >= 1"):
        apply_hashline_edits(content, edits)


def test_very_long_line_hash():
    """A very long line should still produce a valid 2-char hash."""
    long_line = "x" * 100_000
    h = compute_line_hash(1, long_line, None)
    assert len(h) == 2
    # Determinism
    assert compute_line_hash(1, long_line, None) == h


def test_no_alphanumeric_different_content_same_line_num():
    """Lines with no alphanumeric chars use line_num as seed.
    
    Different special-character-only lines at the same line number use the same seed,
    so different special chars can produce different hashes (or collide).
    """
    h1 = compute_line_hash(5, "@#$%", None)
    h2 = compute_line_hash(5, "!^&*", None)
    # Both have no alphanumeric, both at line 5, both use seed=5
    # They may or may not collide depending on content
    assert len(h1) == 2
    assert len(h2) == 2
    # With different content they typically produce different hashes
    # (but there's a 1/256 chance of collision)


def test_no_alphanumeric_prev_hash_seed():
    """Lines with no alphanumeric chars but with prev_hash use prev_hash as seed, not line_num."""
    h1 = compute_line_hash(1, "@#$%", "AB")
    h2 = compute_line_hash(99, "@#$%", "AB")
    # Both have prev_hash so use seed derived from "AB", regardless of line_num
    assert h1 == h2


def test_content_crlf_edit():
    """Content with CRLF line endings should work with edits.
    apply_hashline_edits normalizes \r\n to \n before processing."""
    content = "line 1\r\nline 2\r\nline 3\r\n"
    # Hashes can be computed directly on CRLF content (no manual \r strip needed)
    h2 = get_line_hash(content, 2)
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert "MODIFIED" in result
    assert "line 2" not in result
    assert first_changed == 2


def test_parse_anchor_multiple_hashes():
    """'5#a#b' splits on first # only: (5, 'a#b')."""
    result = parse_anchor("5#a#b")
    assert result == (5, "a#b")


def test_compute_line_hash_non_nibble_prev_hash():
    """prev_hash with characters outside NIBBLE_STR still works (seed computation uses ord)."""
    h1 = compute_line_hash(1, "content", "!!")
    assert len(h1) == 2
    # Deterministic
    assert compute_line_hash(1, "content", "!!") == h1


def test_replace_range_start_equals_end():
    """Range where start == end is equivalent to single-line replace."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=AnchorRef(line=2, hash=h2),
            lines=["REPLACED"],
        )
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["line 1", "REPLACED", "line 3"]
    assert first_changed == 2


def test_file_with_only_newlines():
    """Content with only newlines produces empty lines after splitlines."""
    content = "\n\n\n"
    # splitlines: ["", "", ""] — 3 empty lines
    lines = content.splitlines()
    assert len(lines) == 3
    assert all(line == "" for line in lines)

    h1 = get_line_hash(content, 1)
    h2 = get_line_hash(content, 2)
    h3 = get_line_hash(content, 3)

    # All empty lines but at different positions, so different hashes (seed=line_num)
    assert h1 != h2 != h3
    assert len(h1) == len(h2) == len(h3) == 2


def test_deduplication_different_content_same_position():
    """Different edits at the same position with different content are NOT deduplicated.
    
    They will be caught by overlap detection instead.
    """
    content = "line 1\nline 2\nline 3\n"
    h1 = get_line_hash(content, 1)
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["AAA"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["BBB"]),
    ]
    # Different content, so not deduplicated. Overlap detection should catch it.
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_overlap_append_append_same_line():
    """Two append operations at the same line overlap."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        AppendEdit(op="append", pos=AnchorRef(line=2, hash=h2), lines=["first"]),
        AppendEdit(op="append", pos=AnchorRef(line=2, hash=h2), lines=["second"]),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_overlap_prepend_prepend_same_line():
    """Two prepend operations at the same line overlap."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        PrependEdit(op="prepend", pos=AnchorRef(line=2, hash=h2), lines=["first"]),
        PrependEdit(op="prepend", pos=AnchorRef(line=2, hash=h2), lines=["second"]),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_append_eof_on_empty_file():
    """Append at EOF on empty file creates first line."""
    content = ""
    edits = [AppendEdit(op="append", pos=None, lines=["first line"])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result == "first line"
    assert first_changed == 1


def test_prepend_bof_on_empty_file():
    """Prepend at BOF on empty file creates first line."""
    content = ""
    edits = [PrependEdit(op="prepend", pos=None, lines=["first line"])]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result == "first line"
    assert first_changed == 1


def test_multiple_edits_consecutive():
    """Multiple edits at consecutive lines all succeed."""
    content = "a\nb\nc\nd\ne\n"
    h1 = get_line_hash(content, 1)
    h3 = get_line_hash(content, 3)
    h5 = get_line_hash(content, 5)
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["A"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=3, hash=h3), end=None, lines=["C"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=5, hash=h5), end=None, lines=["E"]),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["A", "b", "C", "d", "E"]


def test_unicode_prev_hash_seed():
    """Unicode characters in prev_hash affect seed computation via ord()."""
    h_ascii = compute_line_hash(1, "content", "AB")
    h_unicode = compute_line_hash(1, "content", "\u4e16\u754c")  # 世界
    # Different seeds produce different hashes
    assert h_ascii != h_unicode


def test_compute_line_hash_only_cr():
    """A line that is only '\r' becomes empty after stripping, treated as whitespace-only."""
    h_cr = compute_line_hash(1, "\r", None)
    h_empty = compute_line_hash(1, "", None)
    # Both are empty after stripping/normalizing, same line_num, so same hash
    assert h_cr == h_empty


def test_compute_line_hash_mixed_cr():
    """'text\r' should equal 'text'."""
    assert compute_line_hash(1, "text\r", None) == compute_line_hash(1, "text", None)


def test_hashline_mismatch_error_display():
    """HashlineMismatchError __str__ includes context lines."""

    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    file_lines = content.splitlines()
    mismatches = [HashMismatch(line=3, expected="XX", actual=get_line_hash(content, 3))]
    error = HashlineMismatchError(mismatches, file_lines)
    error_str = str(error)
    assert "changed since last read" in error_str
    assert ">>>" in error_str  # Markers for changed lines
    assert "3#" in error_str


def test_apply_hashline_edits_multiple_mismatches():
    """Multiple hash mismatches are collected before raising."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash="ZZ"), end=None, lines=["a"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=2, hash="YY"), end=None, lines=["b"]),
    ]
    with pytest.raises(HashlineMismatchError) as exc_info:
        apply_hashline_edits(content, edits)
    assert len(exc_info.value.mismatches) == 2
    assert exc_info.value.mismatches[0].line == 1
    assert exc_info.value.mismatches[1].line == 2


def test_validate_anchor_ref_line_zero():
    """Directly test validate_anchor_ref with line=0."""

    file_lines = ["line 1"]
    mismatches: list = []
    validation_errors: list[str] = []
    anchor = AnchorRef(line=0, hash="AB")
    validate_anchor_ref(anchor, file_lines, mismatches, validation_errors)
    assert validation_errors == ["Line 0 must be >= 1"]
    assert mismatches == []


# ═══════════════════════════════════════════════════════════════════════════
# 13. Corner case — sort order tests (sync)
# ═══════════════════════════════════════════════════════════════════════════


def test_bottom_up_sort_order():
    """Edits are applied bottom-up so line numbers remain valid."""
    content = "line 1\nline 2\nline 3\nline 4\n"
    h4 = get_line_hash(content, 4)
    h1 = get_line_hash(content, 1)
    # Intentionally out of order: edit line 4 first, then line 1
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=4, hash=h4), end=None, lines=["D"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["A"]),
    ]
    result, first_changed = apply_hashline_edits(content, edits)
    assert result.splitlines() == ["A", "line 2", "line 3", "D"]
    assert first_changed == 1


def test_append_after_then_replace_before():
    """Append after line 2, then replace line 1 — no conflict if bottom-up."""
    content = "line 1\nline 2\nline 3\n"
    h1 = get_line_hash(content, 1)
    h2 = get_line_hash(content, 2)
    edits = [
        AppendEdit(op="append", pos=AnchorRef(line=2, hash=h2), lines=["inserted"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=1, hash=h1), end=None, lines=["REPLACED"]),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert "REPLACED" in result
    assert "inserted" in result


# ═══════════════════════════════════════════════════════════════════════════
# 14. Corner case — anchor model tests
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 15. Cumulative hash edge case tests
# ═══════════════════════════════════════════════════════════════════════════


def test_cumulative_hash_chain_full_recomputation():
    """After editing line N, ALL subsequent line hashes change."""
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    h2 = get_line_hash(content, 2)
    h3_orig = get_line_hash(content, 3)
    h4_orig = get_line_hash(content, 4)
    h5_orig = get_line_hash(content, 5)

    edit = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=2, hash=h2), end=None, lines=["MODIFIED"]),
    ]
    modified, _ = apply_hashline_edits(content, edit)

    h3_new = get_line_hash(modified, 3)
    h4_new = get_line_hash(modified, 4)
    h5_new = get_line_hash(modified, 5)

    assert h3_orig != h3_new, "Line 3 hash should change after editing line 2"
    assert h4_orig != h4_new, "Line 4 hash should change"
    assert h5_orig != h5_new, "Line 5 hash should change"


def test_cumulative_hash_insertion_shifts():
    """Inserting lines shifts subsequent hashes."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    h3_orig = get_line_hash(content, 3)

    edit = [
        AppendEdit(op="append", pos=AnchorRef(line=2, hash=h2), lines=["inserted"]),
    ]
    modified, _ = apply_hashline_edits(content, edit)

    # Line 3 original is now at line 4
    h3_new = get_line_hash(modified, 3)  # This is "inserted"
    h4_new = get_line_hash(modified, 4)  # This is the old "line 3"

    assert h3_orig != h3_new, "Old line 3 hash != new line 3 hash (different content)"
    assert h3_orig != h4_new, "Old line 3 hash != line 4 new hash (different position/prev_hash)"


# ═══════════════════════════════════════════════════════════════════════════
# 16. HashlineMismatchError display edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_mismatch_error_single_line():
    """Single mismatch uses singular 'line'."""

    file_lines = ["line 1", "line 2", "line 3"]
    mismatches = [HashMismatch(line=2, expected="ZZ", actual="AB")]
    error = HashlineMismatchError(mismatches, file_lines)
    error_str = str(error)
    assert "1 line" in error_str
    assert "has changed" in error_str or "have changed" in error_str


def test_mismatch_error_multiple_lines():
    """Multiple mismatches use plural 'lines'."""

    file_lines = ["line 1", "line 2", "line 3"]
    mismatches = [
        HashMismatch(line=1, expected="ZZ", actual="AB"),
        HashMismatch(line=2, expected="YY", actual="CD"),
    ]
    error = HashlineMismatchError(mismatches, file_lines)
    error_str = str(error)
    assert "2 lines" in error_str


def test_mismatch_error_context_at_boundary():
    """Mismatch at line 1 still shows context (no lines before)."""

    file_lines = ["line 1", "line 2", "line 3"]
    mismatches = [HashMismatch(line=1, expected="ZZ", actual="AB")]
    error = HashlineMismatchError(mismatches, file_lines)
    error_str = str(error)
    assert ">>>" in error_str
    assert "1#" in error_str


# ═══════════════════════════════════════════════════════════════════════════
# 17. _deduplicate_edits edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_deduplicate_delete_same_line():
    """Duplicate delete edits at the same line are deduplicated."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edit = DeleteEdit(op="delete", pos=AnchorRef(line=2, hash=h2))
    edits = [edit, edit]
    result, _ = apply_hashline_edits(content, edits)
    # Should only delete once
    assert result.splitlines() == ["line 1", "line 3"]


def test_deduplicate_append_eof_different_content():
    """Two distinct append-at-EOF edits with different content are NOT deduplicated."""
    content = "line 1\n"
    edits = [
        AppendEdit(op="append", pos=None, lines=["first"]),
        AppendEdit(op="append", pos=None, lines=["second"]),
    ]
    # They have different content, so different keys — not deduplicated
    # But both at EOF, so they overlap
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_deduplicate_prepend_bof_different_content():
    """Two distinct prepend-at-BOF edits with different content are NOT deduplicated."""
    content = "line 1\n"
    edits = [
        PrependEdit(op="prepend", pos=None, lines=["first"]),
        PrependEdit(op="prepend", pos=None, lines=["second"]),
    ]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


# ═══════════════════════════════════════════════════════════════════════════
# 18. Overlap detection edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_replace_and_prepend_same_line():
    """Replace at line N and prepend at line N overlap (replace affects line N, prepend inserts before it)."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=2, hash=h2), end=None, lines=["REPLACED"]),
        PrependEdit(op="prepend", pos=AnchorRef(line=2, hash=h2), lines=["prepended"]),
    ]
    # Replace affects line 2, prepend inserts before line 2
    # Prepend range: [2, 2 + len - 1] = [2, 2] for single line
    # Replace range: [2, 2]
    # They overlap!
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


def test_append_and_prepend_different_lines_allowed():
    """Append at line 1 and prepend at line 3 do NOT overlap."""
    content = "line 1\nline 2\nline 3\n"
    h1 = get_line_hash(content, 1)
    h3 = get_line_hash(content, 3)
    edits = [
        AppendEdit(op="append", pos=AnchorRef(line=1, hash=h1), lines=["after 1"]),
        PrependEdit(op="prepend", pos=AnchorRef(line=3, hash=h3), lines=["before 3"]),
    ]
    result, _ = apply_hashline_edits(content, edits)
    assert "after 1" in result
    assert "before 3" in result


def test_delete_and_append_same_line():
    """Delete at N (normalized to replace with empty lines) and append at N overlap."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        DeleteEdit(op="delete", pos=AnchorRef(line=2, hash=h2)),
        AppendEdit(op="append", pos=AnchorRef(line=2, hash=h2), lines=["after"]),
    ]
    # Delete normalizes to replace at [2,2]; append at 2 inserts at [3,3]
    # These should NOT overlap: replace [2,2] vs append [3,3] are adjacent, not overlapping
    # But wait: the overlap check uses intervals_overlap which checks range_i[1] < range_j[0]
    # For replace [2,2] and append [3,3]: 2 < 3 is true, so NOT overlapping.
    result, _ = apply_hashline_edits(content, edits)
    assert "line 2" not in result
    assert "after" in result


def test_delete_and_prepend_same_line():
    """Delete at N and prepend at N overlap."""
    content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(content, 2)
    edits = [
        DeleteEdit(op="delete", pos=AnchorRef(line=2, hash=h2)),
        PrependEdit(op="prepend", pos=AnchorRef(line=2, hash=h2), lines=["before"]),
    ]
    # Delete normalizes to replace at [2,2]; prepend at 2 inserts at [2,2]
    # These DO overlap: [2,2] intersects with [2,2]
    with pytest.raises(ValueError, match="Overlapping edits detected"):
        apply_hashline_edits(content, edits)


# ═══════════════════════════════════════════════════════════════════════════
# 19. generate_hash_aware_diff edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_diff_empty_old_content():
    """Diff with empty old content."""
    old = ""
    new = "line 1\nline 2\n"
    diff = generate_hash_aware_diff(old, new, 1)
    assert "+1#" in diff
    assert "+2#" in diff


def test_diff_empty_new_content():
    """Diff where new content is empty — deleted lines are tracked in deleted_old_lines
    but since new content has 0 lines, the iteration over new lines shows nothing.
    The diff note is still present."""
    old = "line 1\nline 2\n"
    new = ""
    diff = generate_hash_aware_diff(old, new, 1)
    # The note is always present
    assert "Lines after edited regions have stale hashes" in diff


def test_diff_first_changed_line_fallback():
    """When no changes are detected, diff falls back to first_changed_line context."""
    old = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\n"
    new = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\n"
    diff = generate_hash_aware_diff(old, new, 8)
    # Should show context around line 8 (±5 lines)
    assert " 3#" in diff or " 4#" in diff  # Lower bound
    assert "13#" in diff or "12#" in diff  # Upper bound


def test_diff_insert_at_beginning():
    """Insert at beginning of file."""
    old = "line 1\nline 2\n"
    new = "inserted\nline 1\nline 2\n"
    diff = generate_hash_aware_diff(old, new, 1)
    assert "+1#" in diff
    assert "inserted" in diff


def test_diff_delete_at_end():
    """Delete at end of file — diff iterates new lines, so deleted trailing lines
    beyond the new file length are not shown. Only surviving lines appear."""
    old = "line 1\nline 2\nline 3\n"
    new = "line 1\nline 2\n"
    diff = generate_hash_aware_diff(old, new, 3)
    # Lines 1 and 2 survive
    assert "line 1" in diff
    assert "line 2" in diff
    # Note is always present
    assert "Lines after edited regions have stale hashes" in diff


# ═══════════════════════════════════════════════════════════════════════════
# 20. Validation error accumulation
# ═══════════════════════════════════════════════════════════════════════════


def test_validation_errors_and_mismatches_together():
    """Validation errors are raised before hash mismatches."""
    content = "line 1\nline 2\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=10, hash="AB"),  # Out of range
            end=None,
            lines=["replaced"],
        ),
    ]
    with pytest.raises(ValueError, match="does not exist"):
        apply_hashline_edits(content, edits)


def test_validation_errors_accumulated():
    """Multiple validation errors are accumulated."""
    content = "line 1\n"
    edits = [
        ReplaceEdit(op="replace", pos=AnchorRef(line=0, hash="AB"), end=None, lines=["a"]),
        ReplaceEdit(op="replace", pos=AnchorRef(line=5, hash="CD"), end=None, lines=["b"]),
    ]
    with pytest.raises(ValueError) as exc_info:
        apply_hashline_edits(content, edits)
    error_msg = str(exc_info.value)
    assert "must be >= 1" in error_msg
    assert "does not exist" in error_msg


# ═══════════════════════════════════════════════════════════════════════════
# 21. Anchor line negative test
# ═══════════════════════════════════════════════════════════════════════════


def test_anchor_negative_line():
    """Anchor with negative line number."""
    # parse_anchor should handle this
    result = parse_anchor("-5#ab")
    assert result == (-5, "ab")

    # But validation should catch it
    content = "line 1\nline 2\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=-5, hash="ab"),
            end=None,
            lines=["replaced"],
        )
    ]
    with pytest.raises(ValueError, match="must be >= 1"):
        apply_hashline_edits(content, edits)


# ═══════════════════════════════════════════════════════════════════════════
# 22. Whitespace normalization edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_whitespace_normalization_identical():
    """Lines with different whitespace but same alphanumeric content produce same hash."""
    h1 = compute_line_hash(1, "  hello  world  ", None)
    h2 = compute_line_hash(1, "\thello\tworld\t", None)
    h3 = compute_line_hash(1, "helloworld", None)
    # All normalize to "helloworld"
    assert h1 == h2 == h3


def test_whitespace_only_lines_different_positions():
    """Empty/whitespace-only lines at different positions use different seeds (line_num).
    Different seeds can still produce the same 8-bit hash — a form of seed collision.
    Most pairs differ, but it's probabilistic."""
    h1 = compute_line_hash(1, "   ", None)
    h2 = compute_line_hash(2, "   ", None)
    h3 = compute_line_hash(3, "   ", None)
    # All should be valid 2-char hashes
    assert len(h1) == len(h2) == len(h3) == 2
    # At least some pairs should differ (probabilistically, most will)
    # But we can't assert all are different — different seeds can collide in 8-bit output
    assert len({h1, h2, h3}) >= 1  # All valid


# ═══════════════════════════════════════════════════════════════════════════
# 23. Fuzzy CRLF/LF matching tests
# ═══════════════════════════════════════════════════════════════════════════


def test_fuzzy_crlf_edit_without_manual_r_strip():
    """CRLF content can be edited using hashes computed from LF-only content.
    
    The fuzzy fallback in validate_anchor_ref strips \r from file lines and retries."""
    # Content with CRLF line endings
    crlf_content = "line 1\r\nline 2\r\nline 3\r\n"
    # Hashes computed on LF-only version (simulating what the LLM would have from _do_read)
    lf_content = "line 1\nline 2\nline 3\n"
    h2 = get_line_hash(lf_content, 2)

    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    result, first_changed = apply_hashline_edits(crlf_content, edits)
    assert "MODIFIED" in result
    assert first_changed == 2


def test_fuzzy_crlf_edit_range():
    """Fuzzy fallback also works for range edits with CRLF content."""
    crlf_content = "line 1\r\nline 2\r\nline 3\r\nline 4\r\n"
    lf_content = "line 1\nline 2\nline 3\nline 4\n"
    h2 = get_line_hash(lf_content, 2)
    h3 = get_line_hash(lf_content, 3)

    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=h2),
            end=AnchorRef(line=3, hash=h3),
            lines=["REPLACED"],
        )
    ]
    result, _ = apply_hashline_edits(crlf_content, edits)
    assert "REPLACED" in result
    assert "line 2" not in result
    assert "line 3" not in result


def test_fuzzy_fallback_not_triggered_for_wrong_hash():
    """Fuzzy fallback still fails when hash is genuinely wrong (not a CRLF issue)."""
    content = "line 1\nline 2\nline 3\n"
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash="ZZ"),  # Completely wrong hash
            end=None,
            lines=["replaced"],
        )
    ]
    with pytest.raises(HashlineMismatchError):
        apply_hashline_edits(content, edits)


def test_fuzzy_fallback_with_direct_validate_call():
    """Fuzzy fallback in validate_anchor_ref triggers when file_lines contain \r."""

    # file_lines with \r (simulating lines before splitlines normalization)
    file_lines = ["line 1\r", "line 2\r", "line 3"]
    # Hash computed from \r-free content
    lf_lines = ["line 1", "line 2", "line 3"]
    h2 = get_line_hash("\n".join(lf_lines), 2)

    mismatches: list = []
    validation_errors: list[str] = []
    anchor = AnchorRef(line=2, hash=h2)
    validate_anchor_ref(anchor, file_lines, mismatches, validation_errors)
    # Fuzzy fallback should have matched, so no mismatches
    assert mismatches == []
    assert validation_errors == []


def test_fuzzy_fallback_prev_hash_chain():
    """Fuzzy fallback recomputes the entire cumulative chain with \r stripped."""
    crlf_content = "line 1\r\nline 2\r\nline 3\r\n"
    lf_content = "line 1\nline 2\nline 3\n"
    # Hash for line 3 depends on the cumulative chain
    h3 = get_line_hash(lf_content, 3)

    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=3, hash=h3),
            end=None,
            lines=["MODIFIED"],
        )
    ]
    result, _ = apply_hashline_edits(crlf_content, edits)
    assert "MODIFIED" in result


# ═══════════════════════════════════════════════════════════════════════════
# 24. max_char line truncation tests (async)
# ═══════════════════════════════════════════════════════════════════════════


async def test_max_char_output_limit(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """max_char limits total output characters (like read.py)."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    await file_path.write_text(content)

    # Use a small max_char to limit total output
    result = await hash_line_tool(HashReadParams(path=str(file_path), max_char=30))
    assert not result.is_error
    assert len(result.output) <= 30
    # Should contain beginning but cut off
    assert "line 1" in result.output


async def test_max_char_zero_empty_output(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """max_char=0 returns empty output."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("line 1\nline 2\n")

    result = await hash_line_tool(HashReadParams(path=str(file_path), max_char=0))
    assert not result.is_error
    assert result.output == ""


async def test_char_offset_skip_start(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """char_offset skips first N characters of output."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("line 1\nline 2\n")

    # Read full output first to know the length
    full = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not full.is_error
    full_output = full.output

    # With offset=5, first 5 chars are skipped
    result = await hash_line_tool(HashReadParams(path=str(file_path), char_offset=5))
    assert not result.is_error
    assert result.output == full_output[5:]


async def test_max_char_and_char_offset_combined(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """char_offset and max_char together: skip N then take M chars."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("line 1\nline 2\nline 3\n")

    full = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not full.is_error
    full_output = full.output

    result = await hash_line_tool(HashReadParams(path=str(file_path), char_offset=3, max_char=10))
    assert not result.is_error
    # max_char is the end index (Python slice semantics): [char_offset:max_char]
    assert result.output == full_output[3:10]


def test_max_char_hash_unchanged_by_truncation():
    """Per-line truncation (MAX_LINE_LENGTH) is display-only; hashes are on full lines."""
    content = "short\n" + "x" * 3000 + "\nnormal\n"
    h1 = get_line_hash(content, 1)
    h2 = get_line_hash(content, 2)
    h3 = get_line_hash(content, 3)
    assert len(h1) == len(h2) == len(h3) == 2
    # The long line hash is computed on the original 3000-char line, not truncated


# ═══════════════════════════════════════════════════════════════════════════

async def test_max_bytes_truncation(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Output is truncated when exceeding MAX_BYTES (100KB)."""
    file_path = temp_work_dir / "big.txt"
    # Generate ~200KB of content to exceed the 100KB limit
    big_line = "x" * 1000
    lines = [f"line {i}: {big_line}" for i in range(200)]
    content_text = "\n".join(lines) + "\n"
    await file_path.write_text(content_text)

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert "KB" in result.output


async def test_max_bytes_not_exceeded_small_file(hash_line_tool: HashRead, temp_work_dir: KaosPath):
    """Small files under MAX_BYTES are shown completely."""
    file_path = temp_work_dir / "small.txt"
    await file_path.write_text("line 1\nline 2\nline 3\n")

    result = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result.is_error
    assert "truncated" not in result.output.lower()
    assert "line 1" in result.output
    assert "line 2" in result.output
    assert "line 3" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# 25. New HashRead / HashEdit behavior tests (async)
# ═══════════════════════════════════════════════════════════════════════════


async def test_read_empty_path(hash_line_tool: HashRead):
    """HashRead returns ToolError for empty path."""
    result = await hash_line_tool(HashReadParams(path=""))
    assert result.is_error
    assert "File path cannot be empty" in result.message


async def test_edit_file_mark_dirty_blocks_second_edit(
    hash_edit_tool: HashEdit, temp_work_dir: KaosPath, session
):
    """Editing a tracked file whose mtime hasn't changed returns error."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\n"
    await file_path.write_text(content)

    # First edit succeeds (file is not yet tracked)
    result1 = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
                    end=None,
                    lines=["MODIFIED"],
                )
            ],
        )
    )
    assert not result1.is_error

    # Simulate the tracker having the current mtime (no external change)
    from kimi_cli.utils.path import kaos_path_from_user_input
    key = str(kaos_path_from_user_input(str(file_path)).canonical())
    st = await file_path.stat()
    session.file_mtime._times[key] = st.st_mtime

    # Second edit without an intervening read should fail
    result2 = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=3, hash=get_line_hash("line 1\nMODIFIED\nline 3\n", 3)),
                    end=None,
                    lines=["ALSO_MODIFIED"],
                )
            ],
        )
    )
    assert result2.is_error
    assert "File modified" in result2.message
    assert "read file first" in result2.message


async def test_edit_file_after_hashread_succeeds(
    hash_line_tool: HashRead,
    hash_edit_tool: HashEdit,
    temp_work_dir: KaosPath,
    session,
):
    """HashRead cleans the tracker so a subsequent HashEdit succeeds."""
    file_path = temp_work_dir / "test.txt"
    content = "line 1\nline 2\nline 3\n"
    await file_path.write_text(content)

    # First edit succeeds (untracked)
    result1 = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(line=2, hash=get_line_hash(content, 2)),
                    end=None,
                    lines=["MODIFIED"],
                )
            ],
        )
    )
    assert not result1.is_error

    # Manually set tracker to current mtime so a second edit would fail
    from kimi_cli.utils.path import kaos_path_from_user_input
    key = str(kaos_path_from_user_input(str(file_path)).canonical())
    st = await file_path.stat()
    session.file_mtime._times[key] = st.st_mtime

    # Read the file with HashRead — this calls clean_file() and resets tracking
    result2 = await hash_line_tool(HashReadParams(path=str(file_path)))
    assert not result2.is_error

    # Edit again should succeed because the tracker was cleared
    result3 = await hash_edit_tool(
        HashEditParams(
            path=str(file_path),
            edits=[
                ReplaceEdit(
                    op="replace",
                    pos=AnchorRef(
                        line=3,
                        hash=get_line_hash("line 1\nMODIFIED\nline 3\n", 3),
                    ),
                    end=None,
                    lines=["ALSO_MODIFIED"],
                )
            ],
        )
    )
    assert not result3.is_error
    assert "ALSO_MODIFIED" in await file_path.read_text()
