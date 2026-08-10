from __future__ import annotations

import contextlib
from typing import Annotated, Literal, override

import xxhash
from rapidfuzz.distance import Levenshtein
from kaos.path import KaosPath
from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field, model_validator

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import ToolResultBuilder, truncate_line
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import (
    is_within_directory,
    is_within_workspace,
    kaos_path_from_tool_input,
    kaos_path_from_user_input,
)
from kimi_cli.utils.sensitive import is_sensitive_file
from kimi_cli.vfs import VFS

from .utils import resolve_vfs

NIBBLE_STR = "ZPMQVRWSNKTXJBYH"
HASH_SEED = 0
MAX_LINE_LENGTH = 2000
MAX_BYTES = 100 << 10  # 100KB

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")

# Precomputed lookup: hash byte → 2-char nibble string
_NIBBLE_LOOKUP: list[str] = [NIBBLE_STR[i >> 4] + NIBBLE_STR[i & 0x0F] for i in range(256)]

# ═══════════════════════════════════════════════════════════════════════════
# Hash Computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_line_hash(line_num: int, line: str, prev_hash: str | None) -> str:
    """Compute a 2-char xxHash32 line hash with cumulative chaining."""
    # Strip trailing carriage return
    if line.endswith("\r"):
        line = line[:-1]

    # Single pass: collect non-whitespace chars, detect alphanumeric content
    chars: list[str] = []
    has_significant = False
    for c in line:
        if not c.isspace():
            chars.append(c)
            if not has_significant and c.isalnum():
                has_significant = True

    # Build seed from previous hash or use defaults
    if prev_hash is not None:
        seed = 0
        for c in prev_hash:
            seed = ((seed * 256) + ord(c)) & 0xFFFFFFFF
    elif has_significant:
        seed = HASH_SEED
    else:
        seed = line_num

    # NOTE: the per-line native kernel (kimix_native.tools.line_hash) is not
    # wired here: benchmarks show the Python->C boundary dominates for
    # per-line calls (0.5x vs xxhash.xxh32 which is itself a C extension).
    # The bulk kernel (tools.line_hashes / compute_line_hashes) stays
    # available for whole-file hashing if a caller ever needs it.
    data = "".join(chars).encode("utf-8")
    hash_val = xxhash.xxh32(data, seed).intdigest() & 0xFF
    return _NIBBLE_LOOKUP[hash_val]


def _cumulative_hashes(file_lines: list[str]) -> list[str]:
    """Compute cumulative (chained) hashes for every line in *file_lines*.

    Bulk native fast path: ``kimix_native.tools.compute_line_hashes`` hashes
    the whole file in a single call (GIL released). The kernel splits on
    ``\n`` and treats a trailing ``\n`` as the end marker (dropping the
    trailing empty element), so it is used only when the line list has no
    trailing empty line — earlier hashes never depend on later lines and the
    pure-Python loop below is bit-identical otherwise.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None:
        if not file_lines or file_lines[-1] != "":
            return _NATIVE_TOOLS.compute_line_hashes("\n".join(file_lines))
    hashes: list[str] = []
    prev_hash: str | None = None
    for i, line in enumerate(file_lines, 1):
        h = compute_line_hash(i, line, prev_hash)
        hashes.append(h)
        prev_hash = h
    return hashes

# ═══════════════════════════════════════════════════════════════════════════
# Anchor Parsing
# ═══════════════════════════════════════════════════════════════════════════


def parse_anchor(anchor: str) -> tuple[int, str] | None:
    """Parse 'LINE#HASH' into (line_num, hash). Supports legacy 'LINE:HASH'."""
    parts = anchor.split("#", 1)
    if len(parts) == 2:
        try:
            line_num = int(parts[0])
        except ValueError:
            return None
        return (line_num, parts[1])

    parts = anchor.split(":", 1)
    if len(parts) == 2:
        try:
            line_num = int(parts[0])
        except ValueError:
            return None
        return (line_num, parts[1])

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════


class AnchorRef(BaseModel):
    line: int
    hash: str

    @model_validator(mode="before")
    @classmethod
    def _parse_string(cls, v):
        if isinstance(v, str):
            parts = v.split("#", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid anchor format '{v}', expected format 'LINE#HASH' (e.g., '8#RT')"
                )
            try:
                line = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Invalid line number '{parts[0]}' in anchor '{v}', expected format 'LINE#HASH' (e.g., '8#RT')"
                ) from None
            return {"line": line, "hash": parts[1]}
        return v


class ReplaceEdit(BaseModel):
    op: Literal["replace"]
    pos: AnchorRef
    end: AnchorRef | None = None
    lines: list[str]


class AppendEdit(BaseModel):
    op: Literal["append"]
    pos: AnchorRef | None = None
    lines: list[str]


class PrependEdit(BaseModel):
    op: Literal["prepend"]
    pos: AnchorRef | None = None
    lines: list[str]


class DeleteEdit(BaseModel):
    op: Literal["delete"]
    pos: AnchorRef


HashlineEdit = Annotated[
    ReplaceEdit | AppendEdit | PrependEdit | DeleteEdit,
    Field(discriminator="op"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════


class HashMismatch:
    def __init__(self, line: int, expected: str, actual: str):
        self.line = line
        self.expected = expected
        self.actual = actual


class HashlineMismatchError(Exception):
    def __init__(self, mismatches: list[HashMismatch], file_lines: list[str]):
        self.mismatches = mismatches
        self.file_lines = file_lines

    def __str__(self) -> str:
        mismatch_set = {m.line for m in self.mismatches}
        lines_word = "lines" if len(self.mismatches) > 1 else "line"
        parts: list[str] = []
        parts.append(
            f"{len(self.mismatches)} {lines_word} have changed since last read. "
            "Use the updated LINE#ID references shown below (>>> marks changed lines)."
        )
        parts.append("")

        # Collect lines to display (mismatch lines + 2 context)
        display_lines_set: set[int] = set()
        for m in self.mismatches:
            lo = max(m.line - 2, 1)
            hi = min(m.line + 2, len(self.file_lines))
            for i in range(lo, hi + 1):
                display_lines_set.add(i)
        display_lines = sorted(display_lines_set)

        # Pre-compute all cumulative hashes for the file
        cumulative_hashes = _cumulative_hashes(self.file_lines)

        prev_line = 0
        for line_num in display_lines:
            if prev_line != 0 and line_num > prev_line + 1:
                parts.append("    ...")
            prev_line = line_num

            text = self.file_lines[line_num - 1]
            hash_str = cumulative_hashes[line_num - 1]
            if line_num in mismatch_set:
                parts.append(f">>> {line_num}#{hash_str}:{text}")
            else:
                parts.append(f"    {line_num}#{hash_str}:{text}")

        return "\n".join(parts)


def validate_anchor_ref(
    anchor: AnchorRef,
    file_lines: list[str],
    mismatches: list[HashMismatch],
    validation_errors: list[str],
    cumulative_hashes: list[str] | None = None,
    fuzzy_hashes: list[str] | None = None,
) -> None:
    if anchor.line < 1:
        validation_errors.append(f"Line {anchor.line} must be >= 1")
        return
    if anchor.line > len(file_lines):
        validation_errors.append(
            f"Line {anchor.line} does not exist (file has {len(file_lines)} lines)"
        )
        return

    # Use precomputed cumulative hashes if provided, else compute
    if cumulative_hashes is not None:
        actual_hash = cumulative_hashes[anchor.line - 1]
    else:
        # Compute all cumulative hashes; earlier hashes do not depend on
        # later lines, so this is bit-identical to the early-stop loop.
        computed = _cumulative_hashes(file_lines)
        actual_hash = computed[anchor.line - 1]

    if actual_hash != anchor.hash:
        # Fuzzy fallback: try with \r stripped from file lines
        fuzzy_match = False
        if fuzzy_hashes is not None:
            # Use precomputed fuzzy hashes
            if fuzzy_hashes[anchor.line - 1] == anchor.hash:
                fuzzy_match = True
        elif any("\r" in line for line in file_lines[:anchor.line]):
            prev_hash_fuzzy: str | None = None
            for i, line in enumerate(file_lines):
                line_num = i + 1
                fuzzy_line = line.replace("\r", "")
                hash_str_fuzzy = compute_line_hash(line_num, fuzzy_line, prev_hash_fuzzy)
                prev_hash_fuzzy = hash_str_fuzzy
                if line_num == anchor.line:
                    if hash_str_fuzzy == anchor.hash:
                        fuzzy_match = True
                    break
        if not fuzzy_match:
            mismatches.append(
                HashMismatch(
                    line=anchor.line,
                    expected=anchor.hash,
                    actual=actual_hash,
                )
            )


# ═══════════════════════════════════════════════════════════════════════════
# Edit Application
# ═══════════════════════════════════════════════════════════════════════════


def _deduplicate_edits(edits: list[HashlineEdit]) -> list[HashlineEdit]:
    seen: dict[str, int] = {}
    result: list[HashlineEdit] = []

    for i, edit in enumerate(edits):
        if isinstance(edit, ReplaceEdit):
            if edit.end is not None:
                line_key = f"r:{edit.pos.line}:{edit.end.line}"
            else:
                line_key = f"s:{edit.pos.line}"
            key = f"{line_key}:{chr(10).join(edit.lines)}"
        elif isinstance(edit, AppendEdit):
            line_key = f"i:{edit.pos.line}" if edit.pos is not None else "ieof"
            key = f"{line_key}:{chr(10).join(edit.lines)}"
        elif isinstance(edit, PrependEdit):
            line_key = f"ib:{edit.pos.line}" if edit.pos is not None else "ibef"
            key = f"{line_key}:{chr(10).join(edit.lines)}"
        elif isinstance(edit, DeleteEdit):
            key = f"d:{edit.pos.line}"
        else:
            key = f"unknown:{i}"

        if key not in seen:
            seen[key] = i
            result.append(edit)

    return result


def _track_first_changed(first: list[int | None], line: int) -> None:
    if first[0] is None or line < first[0]:
        first[0] = line


def _normalize_edit(edit: HashlineEdit) -> HashlineEdit:
    if isinstance(edit, DeleteEdit):
        return ReplaceEdit(op="replace", pos=edit.pos, end=None, lines=[])
    return edit


def apply_hashline_edits(
    content: str, edits: list[HashlineEdit]
) -> tuple[str, int | None]:
    if not edits:
        return content, None

    # Normalize line endings for fuzzy CRLF/LF matching
    content = content.replace("\r\n", "\n")

    # Normalize delete edits to replace with empty lines
    edits = [_normalize_edit(e) for e in edits]

    # Track if original content ends with newline
    ends_with_newline = content.endswith("\n")

    file_lines = content.splitlines()
    first_changed_line: list[int | None] = [None]

    # Precompute cumulative hashes once for all anchor validations
    file_hashes: list[str] = []
    prev_hash: str | None = None
    for i, line in enumerate(file_lines):
        line_num = i + 1
        hash_str = compute_line_hash(line_num, line, prev_hash)
        file_hashes.append(hash_str)
        prev_hash = hash_str

    # Precompute fuzzy hashes (\r-stripped) if any lines contain \r
    fuzzy_hashes: list[str] | None = None
    if any("\r" in line for line in file_lines):
        fuzzy_hashes = []
        prev_fuzzy: str | None = None
        for i, line in enumerate(file_lines):
            line_num = i + 1
            fuzzy_line = line.replace("\r", "")
            hash_str = compute_line_hash(line_num, fuzzy_line, prev_fuzzy)
            fuzzy_hashes.append(hash_str)
            prev_fuzzy = hash_str

    # Pre-validate: collect all hash mismatches and check for invalid ranges
    mismatches: list[HashMismatch] = []
    validation_errors: list[str] = []

    for edit in edits:
        if isinstance(edit, ReplaceEdit):
            if edit.end is not None and edit.pos.line > edit.end.line:
                validation_errors.append(
                    f"Range start line {edit.pos.line} must be <= end line {edit.end.line}"
                )
            validate_anchor_ref(edit.pos, file_lines, mismatches, validation_errors, file_hashes, fuzzy_hashes)
            if edit.end is not None:
                validate_anchor_ref(
                    edit.end, file_lines, mismatches, validation_errors, file_hashes, fuzzy_hashes
                )
        elif isinstance(edit, (AppendEdit, PrependEdit)):
            if edit.pos is not None:
                validate_anchor_ref(
                    edit.pos, file_lines, mismatches, validation_errors, file_hashes, fuzzy_hashes
                )

    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    if mismatches:
        raise HashlineMismatchError(mismatches, file_lines)

    # Deduplicate edits targeting same location with same content
    edits = _deduplicate_edits(edits)

    # Check for overlapping edits
    overlapping: list[str] = []
    file_len = len(file_lines)

    def _get_edit_range(edit: HashlineEdit) -> tuple[int, int] | None:
        if isinstance(edit, ReplaceEdit):
            end_line = edit.end.line if edit.end is not None else edit.pos.line
            return (edit.pos.line, end_line)
        elif isinstance(edit, AppendEdit):
            if not edit.lines:
                return None
            ref_line = edit.pos.line if edit.pos is not None else file_len
            # Append inserts after ref_line, so range is [ref_line+1, ref_line+lines.len()]
            return (ref_line + 1, ref_line + len(edit.lines))
        elif isinstance(edit, PrependEdit):
            if not edit.lines:
                return None
            ref_line = edit.pos.line if edit.pos is not None else 1
            # Prepend inserts before ref_line, so range is [ref_line, ref_line+lines.len()-1]
            return (ref_line, ref_line + len(edit.lines) - 1)
        return None

    for i in range(len(edits)):
        range_i = _get_edit_range(edits[i])
        if range_i is None:
            continue
        for j in range(i + 1, len(edits)):
            range_j = _get_edit_range(edits[j])
            if range_j is None:
                continue

            # Check if ranges overlap (intervals intersect)
            intervals_overlap = not (range_i[1] < range_j[0] or range_j[1] < range_i[0])

            # Special case: Append and Prepend at same ref line are conceptually at the same position
            same_ref_line = False
            if isinstance(edits[i], AppendEdit) and isinstance(edits[j], PrependEdit):
                pos_a = edits[i].pos
                pos_b = edits[j].pos
                ref_a = pos_a.line if pos_a is not None else file_len
                ref_b = pos_b.line if pos_b is not None else 1
                same_ref_line = ref_a == ref_b and pos_a is not None and pos_b is not None
            elif isinstance(edits[i], PrependEdit) and isinstance(edits[j], AppendEdit):
                pos_a = edits[i].pos
                pos_b = edits[j].pos
                ref_a = pos_a.line if pos_a is not None else 1
                ref_b = pos_b.line if pos_b is not None else file_len
                same_ref_line = ref_a == ref_b and pos_a is not None and pos_b is not None

            if intervals_overlap or same_ref_line:
                op_i = (
                    "replace"
                    if isinstance(edits[i], ReplaceEdit)
                    else "append"
                    if isinstance(edits[i], AppendEdit)
                    else "prepend"
                )
                op_j = (
                    "replace"
                    if isinstance(edits[j], ReplaceEdit)
                    else "append"
                    if isinstance(edits[j], AppendEdit)
                    else "prepend"
                )
                overlapping.append(
                    f"  - {op_i} at lines {range_i[0]}-{range_i[1]} overlaps with {op_j} at lines {range_j[0]}-{range_j[1]}"
                )

    if overlapping:
        raise ValueError(
            "Overlapping edits detected. Combine overlapping edits into a single operation:\n"
            + "\n".join(overlapping)
        )

    # Sort edits bottom-up (highest line first)
    annotated: list[tuple[int, int, HashlineEdit]] = []
    for idx, edit in enumerate(edits):
        if isinstance(edit, ReplaceEdit):
            end_line = edit.end.line if edit.end is not None else edit.pos.line
            sort_line = end_line
        elif isinstance(edit, AppendEdit):
            sort_line = edit.pos.line if edit.pos is not None else file_len
        elif isinstance(edit, PrependEdit):
            sort_line = edit.pos.line if edit.pos is not None else 0
        else:
            sort_line = 0
        annotated.append((idx, sort_line, edit))

    # Sort by line descending, then by original index descending
    annotated.sort(key=lambda x: (-x[1], -x[0]))

    # Apply edits
    for _idx, _sort_line, edit in annotated:
        if isinstance(edit, ReplaceEdit):
            if edit.end is not None:
                # Replace range
                count = edit.end.line - edit.pos.line + 1
                start_idx = edit.pos.line - 1
                file_lines[start_idx : start_idx + count] = edit.lines
            else:
                # Replace single line
                start_idx = edit.pos.line - 1
                file_lines[start_idx : start_idx + 1] = edit.lines
            _track_first_changed(first_changed_line, edit.pos.line)
        elif isinstance(edit, AppendEdit):
            if not edit.lines:
                continue
            if edit.pos is not None:
                # Insert after specified line
                file_lines[edit.pos.line : edit.pos.line] = edit.lines
                _track_first_changed(first_changed_line, edit.pos.line + 1)
            else:
                # Append at end of file
                if len(file_lines) == 1 and file_lines[0] == "":
                    file_lines.clear()
                start_idx = len(file_lines)
                file_lines.extend(edit.lines)
                _track_first_changed(first_changed_line, start_idx + 1)
        elif isinstance(edit, PrependEdit):
            if not edit.lines:
                continue
            if edit.pos is not None:
                # Insert before specified line
                file_lines[edit.pos.line - 1 : edit.pos.line - 1] = edit.lines
                _track_first_changed(first_changed_line, edit.pos.line)
            else:
                # Prepend at start of file
                if len(file_lines) == 1 and file_lines[0] == "":
                    file_lines.clear()
                file_lines[0:0] = edit.lines
                _track_first_changed(first_changed_line, 1)

    result = "\n".join(file_lines)
    if ends_with_newline and result and not result.endswith("\n"):
        result += "\n"
    return result, first_changed_line[0]


# ═══════════════════════════════════════════════════════════════════════════
# Hash-aware Diff
# ═══════════════════════════════════════════════════════════════════════════


def generate_hash_aware_diff(
    old_content: str, new_content: str, first_changed_line: int
) -> str:
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    total_new_lines = len(new_lines)

    # Compute cumulative hashes for all new lines
    new_line_hashes: list[str] = []
    prev_hash: str | None = None
    for i, line in enumerate(new_lines):
        line_num = i + 1
        hash_str = compute_line_hash(line_num, line, prev_hash)
        new_line_hashes.append(hash_str)
        prev_hash = hash_str

    # Use RapidFuzz Levenshtein to find changes
    ops = Levenshtein.editops(old_lines, new_lines)
    changed_new_lines: set[int] = set()
    deleted_old_lines: set[int] = set()

    for op in ops:
        if op.tag == "insert":
            changed_new_lines.add(op.dest_pos + 1)
        elif op.tag == "delete":
            deleted_old_lines.add(op.src_pos + 1)
        elif op.tag == "replace":
            changed_new_lines.add(op.dest_pos + 1)
            deleted_old_lines.add(op.src_pos + 1)

    # Calculate display range: +/- 5 lines around changes
    display_ranges: list[tuple[int, int]] = []
    for line in changed_new_lines:
        start = max(line - 5, 1)
        end = min(line + 5, total_new_lines)
        display_ranges.append((start, end))

    # Merge overlapping ranges
    display_ranges.sort(key=lambda r: r[0])
    merged_ranges: list[tuple[int, int]] = []
    for start, end in display_ranges:
        if merged_ranges:
            last = merged_ranges[-1]
            if start <= last[1] + 1:
                merged_ranges[-1] = (last[0], max(last[1], end))
                continue
        merged_ranges.append((start, end))

    # If no merged ranges, show context around first_changed_line
    if not merged_ranges:
        start = max(first_changed_line - 5, 1)
        end = min(first_changed_line + 5, total_new_lines)
        merged_ranges.append((start, end))

    # Build output
    output_lines: list[str] = []
    prev_end = 0

    for range_start, range_end in merged_ranges:
        if prev_end > 0 and range_start > prev_end + 1:
            output_lines.append("...")

        for line_num in range(range_start, range_end + 1):
            new_line_content = new_lines[line_num - 1]
            new_hash = new_line_hashes[line_num - 1]

            was_deleted = line_num in deleted_old_lines
            was_inserted = line_num in changed_new_lines

            if was_deleted:
                old_content_line = (
                    old_lines[line_num - 1] if line_num <= len(old_lines) else ""
                )
                output_lines.append(f"-{line_num}#  :{old_content_line}")

            if was_inserted or not was_deleted:
                sign = "+" if was_inserted else " "
                output_lines.append(
                    f"{sign}{line_num}#{new_hash}:{new_line_content}"
                )

        prev_end = range_end

    output_lines.append("")
    output_lines.append(
        "Note: Lines after edited regions have stale hashes. Use hashread to refresh."
    )

    return "\n".join(output_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Tool
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# HashRead — read files with hash-anchored line references
# ═══════════════════════════════════════════════════════════════════════════


class HashReadParams(BaseModel):
    path: str = Field(description="File path.")
    offset: int = Field(default=0, description="0-based line offset.")
    limit: int = Field(default=2000, description="Max lines to read.")
    max_char: int = Field(
        default=65536,
        ge=0,
        description="Max characters to return.",
    )
    char_offset: int = Field(
        default=0,
        ge=0,
        description="Character offset to start from.",
    )


class HashRead(CallableTool2[HashReadParams]):
    name: str = "HashRead"
    description: str = "Read files using hash-anchored line references."
    params: type[HashReadParams] = HashReadParams

    def __init__(
        self, runtime: Runtime, session: Session, vfs: VFS | None = None
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._session = session
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._vfs = vfs

    async def _validate_path(
        self, path: KaosPath, raw_path: str
    ) -> tuple[ToolError | None, bool]:
        resolved_path = path.canonical()
        original_is_absolute = kaos_path_from_user_input(raw_path).is_absolute()
        inside = is_within_workspace(
            resolved_path, self._work_dir, self._additional_dirs
        )
        if not inside and not original_is_absolute:
            return (
                ToolError(
                    message=(
                        f"`{raw_path}` is not an absolute path. "
                        "You must provide an absolute path to access a file "
                        "outside the working directory."
                    ),
                    brief="Invalid path",
                ),
                False,
            )

        protected_paths = (
            self._session.custom_config.get("config_json", {}).get("protected_read_paths")
        )
        if protected_paths:
            from .utils import check_path_protected

            if matched := check_path_protected(
                resolved_path, protected_paths, self._work_dir
            ):
                return (
                    ToolError(
                        message=f"Reading `{path}` is blocked by protected path rule: `{matched}`.",
                        brief="Protected path",
                    ),
                    False,
                )

        return None, inside

    @override
    async def __call__(self, params: HashReadParams) -> ToolReturnValue:
        if not params.path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )
        try:
            return await self._do_read(params)
        except Exception as e:
            return ToolError(message=str(e), brief='Internal error.')

    async def _do_read(self, params: HashReadParams) -> ToolReturnValue:
        display_path = params.path.replace("\\", "/")
        p = kaos_path_from_tool_input(params.path, self._work_dir)
        logical_path = p
        err, _ = await self._validate_path(p, params.path)
        if err:
            return err

        p = await resolve_vfs(params.path, self._vfs, for_write=False, work_dir=self._work_dir)

        if is_sensitive_file(str(logical_path)):
            return ToolError(
                message=(
                    f"`{display_path}` appears to contain secrets "
                    "(matched sensitive file pattern). "
                    "Reading this file is blocked to protect credentials."
                ),
                brief=f"Sensitive file: {display_path}",
            )

        try:
            if not await p.exists():
                return ToolError(
                    message=f"`{display_path}` does not exist.",
                    brief=f"File not found: {display_path}",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{display_path}` is not a file.",
                    brief=f"Invalid path: {display_path}",
                )
        except Exception as e:
            return ToolError(
                message=f"Failed to stat {display_path}. Error: {e}",
                brief=f"Failed to read file: {display_path}",
            )

        try:
            content = await p.read_text(errors="replace")
        except Exception as e:
            return ToolError(
                message=f"Failed to read {display_path}. Error: {e}",
                brief=f"Failed to read file: {display_path}",
            )

        lines = content.splitlines()
        total_lines = len(lines)
        start = params.offset
        count = params.limit
        end = min(start + count, total_lines)

        builder = ToolResultBuilder()
        builder.write("<file>\n")

        if start >= total_lines:
            builder.write("(End of file - 0 lines)\n")
            builder.write("</file>")
            return builder.ok(message=f"Read {display_path}", brief=f"Read {display_path}")

        # Compute cumulative hashes for all lines
        cumulative_hashes: list[str] = []
        prev_hash: str | None = None
        for i, line in enumerate(lines):
            line_num = i + 1
            hash_str = compute_line_hash(line_num, line, prev_hash)
            cumulative_hashes.append(hash_str)
            prev_hash = hash_str

        n_bytes = 0
        max_bytes_reached = False
        actual_end = end
        for i in range(start, end):
            line_num = i + 1
            line = lines[i]
            hash_str = cumulative_hashes[i]
            # Truncate long lines (hash is computed on original line)
            display_line = truncate_line(line, MAX_LINE_LENGTH)
            line_output = f"{line_num}#{hash_str}:{display_line}\n"
            b_len = len(line_output.encode("utf-8"))
            if n_bytes + b_len > MAX_BYTES and i > start:
                max_bytes_reached = True
                actual_end = i
                break
            builder.write(line_output)
            n_bytes += b_len

        if max_bytes_reached:
            builder.write(
                f"\n(Output truncated at {MAX_BYTES // 1024}KB limit. Use 'offset' to read beyond line {actual_end})\n"
            )
        elif end < total_lines:
            builder.write(
                f"\n(File has more lines. Use 'offset' parameter to read beyond line {end})\n"
            )
        else:
            builder.write(f"\n(End of file - {total_lines} total lines)\n")

        builder.write("</file>")
        self._session.file_mtime.clean_file(params.path)
        result = builder.ok(message=f"Read {display_path}", brief=f"Read {display_path}")
        # Apply char_offset / max_char slicing (like read.py)
        if isinstance(result.output, str):
            result.output = result.output[params.char_offset:params.max_char]

        return result

# ═══════════════════════════════════════════════════════════════════════════
# HashEdit — edit files using hash-anchored line references
# ═══════════════════════════════════════════════════════════════════════════


class HashEditParams(BaseModel):
    path: str = Field(description="File path.")
    edits: list[HashlineEdit] = Field(description="Edits to apply.")

class HashEdit(CallableTool2[HashEditParams]):
    name: str = "HashEdit"
    description: str = "Edit files with hash-anchored line references for robustness against concurrent changes."
    params: type[HashEditParams] = HashEditParams

    def __init__(
        self, runtime: Runtime, session: Session, vfs: VFS | None = None
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._session = session
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._vfs = vfs

    async def _validate_path(
        self, path: KaosPath, raw_path: str
    ) -> tuple[ToolError | None, bool]:
        resolved_path = path.canonical()
        original_is_absolute = kaos_path_from_user_input(raw_path).is_absolute()
        inside = is_within_workspace(
            resolved_path, self._work_dir, self._additional_dirs
        )
        if not inside and not original_is_absolute:
            return (
                ToolError(
                    message=(
                        f"`{raw_path}` is not an absolute path. "
                        "You must provide an absolute path to access a file "
                        "outside the working directory."
                    ),
                    brief="Invalid path",
                ),
                False,
            )

        protected_paths = (
            self._session.custom_config.get("config_json", {}).get("protected_write_paths")
        )
        if protected_paths:
            from .utils import check_path_protected

            if matched := check_path_protected(
                resolved_path, protected_paths, self._work_dir
            ):
                return (
                    ToolError(
                        message=f"Editing `{path}` is blocked by protected path rule: `{matched}`.",
                        brief="Protected path",
                    ),
                    False,
                )

        return None, inside

    @override
    async def __call__(self, params: HashEditParams) -> ToolReturnValue:
        if not params.path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )
        try:
            return await self._do_edit(params)
        except Exception as e:
            return ToolError(message=str(e), brief='Internal error.')

    async def _do_edit(self, params: HashEditParams) -> ToolReturnValue:
        try:
            p = kaos_path_from_tool_input(params.path, self._work_dir)
            logical_path = p
            display_logical_path = str(logical_path).replace("\\", "/")
            _outside = not is_within_directory(
                logical_path.canonical(), self._work_dir
            )
            err, path_is_inside = await self._validate_path(p, params.path)
            if err:
                if _outside:
                    err.message = f"[out of work-dir] {err.message}"
                return err

            p = await resolve_vfs(params.path, self._vfs, for_write=True, work_dir=self._work_dir)

            try:
                st = await p.stat()
                from stat import S_ISREG

                if not S_ISREG(st.st_mode):
                    return ToolError(
                        message=f"{'[out of work-dir] ' if _outside else ''}`{display_logical_path}` is not a file.",
                        brief="Invalid path",
                    )
            except FileNotFoundError:
                return ToolError(
                    message=f"{'[out of work-dir] ' if _outside else ''}`{display_logical_path}` does not exist.",
                    brief="File not found",
                )

            content = await p.read_text(errors="replace")
            original_content = content

            new_content, first_changed = apply_hashline_edits(content, params.edits)

            if new_content == original_content:
                return ToolReturnValue(
                    is_error=False,
                    output="",
                    message="No changes made.",
                    display=[BriefDisplayBlock(text=f"No changes: {display_logical_path}")],
                )

            first_changed_line = first_changed if first_changed is not None else 1
            first_line_msg = f" (first change at line {first_changed_line})"

            diff_output = generate_hash_aware_diff(
                original_content, new_content, first_changed_line
            )

            diff_blocks = await build_diff_blocks(
                display_logical_path, original_content, new_content
            )

            await p.write_text(new_content, errors="replace")

            builder = ToolResultBuilder()
            builder.display(*diff_blocks)
            builder.write(
                f"Edit applied successfully{first_line_msg}.\n\n"
            )
            builder.write(f"<diff>\n--- {display_logical_path}\n+++ {display_logical_path}\n")
            builder.write(diff_output)
            builder.write("\n</diff>")

            return builder.ok(
                message=f"Edit applied successfully{first_line_msg}.",
                brief=f"Edited {display_logical_path}",
            )

        except HashlineMismatchError as e:
            return ToolError(
                message=f"Hash mismatch error:\n{e}",
                brief="Hash mismatch",
            )
        except ValueError as e:
            return ToolError(
                message=f"Edit failed: {e}",
                brief="Edit validation failed",
            )
        except Exception as e:
            logger.warning(
                "HashEdit edit failed: {path}: {error}",
                path=params.path,
                error=e,
            )
            _outside_ex = False
            with contextlib.suppress(Exception):
                _outside_ex = not is_within_directory(
                    kaos_path_from_tool_input(params.path, self._work_dir).canonical(),
                    self._work_dir,
                )
            return ToolError(
                message=f"{'[out of work-dir] ' if _outside_ex else ''}Failed to edit. Error: {e}",
                brief="Failed to edit file",
            )


# Backward-compat aliases
HashLine = HashRead  # noqa: F811
Params = HashReadParams  # noqa: F811
