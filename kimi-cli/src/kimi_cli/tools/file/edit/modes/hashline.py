"""Hashline-mode executor for the multi-mode edit tool."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from stat import S_ISREG
from typing import Literal

from kaos.path import KaosPath
from kosong.tooling import ToolError, ToolReturnValue

from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.edit.params import EditMode, EditParams
from kimi_cli.tools.file.hash_line import (
    AnchorRef,
    AppendEdit,
    HashlineMismatchError,
    HashMismatch,
    PrependEdit,
    ReplaceEdit,
    apply_hashline_edits,
    compute_line_hash,
)
from kimi_cli.tools.file.snapshot_store import (
    canonical_snapshot_key,
    get_edit_snapshot_store,
)
from kimi_cli.tools.file.edit_safety import create_edit_parse_guard
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.path import is_within_directory, kaos_path_from_tool_input

from ..base import BaseEditTool


class HashlineClipboard:
    """Per-session hashline named-register storage."""

    _key = "__hashline_clipboard__"

    @classmethod
    def get(cls, session) -> "HashlineClipboard":
        store = session.custom_data.get(cls._key)
        if store is None:
            store = cls()
            session.custom_data[cls._key] = store
        return store

    def __init__(self) -> None:
        self.named: dict[str, list[str]] = {}


@dataclass
class HashlineOp:
    """A parsed hashline operation inside a section."""

    kind: Literal["put", "cut", "rem", "mv"]
    line_text: str
    start: int | None = None
    end: int | None = None
    insert_where: Literal["replace", "before", "after"] = "replace"
    register: str | None = None
    body: list[str] = field(default_factory=list)
    dest: str | None = None


@dataclass
class HashlineSection:
    """One [path#tag] section."""

    path: str
    tag: str
    ops: list[HashlineOp] = field(default_factory=list)


@dataclass
class _PreparedSection:
    section: HashlineSection
    canonical_path: str
    display_path: str
    p: KaosPath
    base_content: str
    using_snapshot: bool
    ops: list[HashlineOp]
    is_delete: bool


def _line1_hash(content: str) -> str:
    """Return the 2-char cumulative hash for line 1 of *content*."""
    lines = content.replace("\r\n", "\n").splitlines()
    if not lines:
        return ""
    return compute_line_hash(1, lines[0], None)


_HEADER_RE = re.compile(r"^\[(?P<path>[^\]\n#]+)#(?P<tag>[^\]\n]+)\]\s*$")

# Body PUT requires a trailing colon; register paste PUT is colonless.
_PUT_RE = re.compile(
    r"^PUT\s+"
    r"(?:(?P<start>\d+)(?:\.=|\.)(?P<end>\d+|\$)|"
    r"<\s*(?P<before>\d+|\$)|"
    r">\s*(?P<after>\d+|\$))"
    r"(?:\s*@\s*(?P<reg>[A-Za-z_]\w*))?"
    r"\s*:"
)

_PUT_PASTE_RE = re.compile(
    r"^PUT\s+"
    r"(?:(?P<start>\d+)(?:\.=|\.)(?P<end>\d+|\$)|"
    r"<\s*(?P<before>\d+|\$)|"
    r">\s*(?P<after>\d+|\$))"
    r"\s+@\s*(?P<reg>[A-Za-z_]\w*)"
    r"\s*$"
)

_CUT_RE = re.compile(
    r"^CUT\s+"
    r"(?P<start>\d+)(?:\.=|\.)(?P<end>\d+|\$)"
    r"(?:\s*@\s*(?P<reg>[A-Za-z_]\w*))?"
    r"\s*$"
)

_REM_RE = re.compile(r"^REM\s*$")
_MV_RE = re.compile(r"^MV\s+(?P<dest>.+?)\s*$")


def _parse_section_body(body_lines: list[tuple[int, str]]) -> list[HashlineOp]:
    """Parse the body of one section into a list of operations."""
    ops: list[HashlineOp] = []
    pending: HashlineOp | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            while pending.body and pending.body[-1] == "":
                pending.body.pop()
            ops.append(pending)
            pending = None

    for _line_num, raw in body_lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        if stripped == "":
            if pending is not None:
                pending.body.append("")
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            flush()
            continue
        if stripped.startswith("*** "):
            flush()
            continue

        put_match = _PUT_RE.match(stripped) or _PUT_PASTE_RE.match(stripped)
        if put_match:
            flush()
            reg = put_match.group("reg")
            if put_match.group("start") is not None:
                start = int(put_match.group("start"))
                end_str = put_match.group("end")
                end = start if end_str == "$" else int(end_str)
                pending = HashlineOp(
                    kind="put",
                    line_text=stripped,
                    start=start,
                    end=end,
                    insert_where="replace",
                    register=reg,
                ) if put_match.re is _PUT_RE and reg is None else HashlineOp(
                    kind="put",
                    line_text=stripped,
                    start=start,
                    end=end,
                    insert_where="replace",
                    register=reg,
                )
            elif put_match.group("before") is not None:
                before_str = put_match.group("before")
                before = 1 if before_str == "$" else int(before_str)
                pending = HashlineOp(
                    kind="put",
                    line_text=stripped,
                    start=before,
                    insert_where="before",
                    register=reg,
                )
            else:
                after_str = put_match.group("after")
                after = None if after_str == "$" else int(after_str)
                pending = HashlineOp(
                    kind="put",
                    line_text=stripped,
                    start=after,
                    insert_where="after",
                    register=reg,
                )
            # Colonless register paste has no body; finalize immediately.
            if put_match.re is _PUT_PASTE_RE:
                flush()
            continue

        cut_match = _CUT_RE.match(stripped)
        if cut_match:
            flush()
            start = int(cut_match.group("start"))
            end_str = cut_match.group("end")
            end = start if end_str == "$" else int(end_str)
            ops.append(
                HashlineOp(
                    kind="cut",
                    line_text=stripped,
                    start=start,
                    end=end,
                    register=cut_match.group("reg"),
                )
            )
            continue

        if _REM_RE.match(stripped):
            flush()
            ops.append(HashlineOp(kind="rem", line_text=stripped))
            continue

        mv_match = _MV_RE.match(stripped)
        if mv_match:
            flush()
            ops.append(HashlineOp(kind="mv", line_text=stripped, dest=mv_match.group("dest").strip()))
            continue

        if pending is not None:
            if stripped.startswith("+"):
                text = line[1:] if len(line) > 1 else ""
                pending.body.append(text)
            elif stripped.startswith("-"):
                raise ValueError(
                    f"Body rows under `{pending.line_text}` must start with `+`; "
                    f"rejecting `-` row: {line!r}"
                )
            else:
                raise ValueError(
                    f"Body rows under `{pending.line_text}` must start with `+`; "
                    f"got: {line!r}"
                )
            continue

        raise ValueError(
            f"Unexpected hashline line: {line!r}. "
            "Expected a section header `[path#tag]`, a `PUT ...:`, `CUT ...`, `REM`, or `MV ...`."
        )

    flush()
    return ops


def parse_hashline_input(input_text: str) -> list[HashlineSection]:
    """Parse hashline-mode input into sections and operations."""
    lines = input_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    while lines and lines[0].strip().startswith("***"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("***"):
        lines.pop()

    sections: list[HashlineSection] = []
    current_body: list[tuple[int, str]] = []
    current_section: HashlineSection | None = None

    def flush_section() -> None:
        nonlocal current_section, current_body
        if current_section is not None:
            current_section.ops = _parse_section_body(current_body)
            sections.append(current_section)
        current_section = None
        current_body = []

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        if stripped == "":
            continue

        header_match = _HEADER_RE.match(stripped)
        if header_match:
            flush_section()
            current_section = HashlineSection(
                path=header_match.group("path").strip(),
                tag=header_match.group("tag").strip(),
            )
            continue

        if stripped.startswith("*** "):
            continue

        current_body.append((i, line))

    flush_section()

    if not sections:
        raise ValueError("No `[path#tag]` hashline sections found in input.")
    return sections


class HashlineModeExecutor:
    """Executor for hashline text-grammar edits."""

    mode: EditMode = "hashline"
    description: str = "Edit files using hash-anchored line references."

    async def execute(self, tool: BaseEditTool, params: EditParams) -> ToolReturnValue:
        if not params.input:
            return ToolError(message="hashline mode requires an input payload.", brief="Missing input")

        try:
            sections = parse_hashline_input(params.input)
        except ValueError as e:
            return ToolError(message=f"Failed to parse hashline input: {e}", brief="Parse error")

        try:
            prepared = await self._preflight(tool, sections)
        except HashlineMismatchError as e:
            return ToolError(message=str(e), brief="Hashline mismatch")
        except ValueError as e:
            return ToolError(message=f"Hashline preflight failed: {e}", brief="Preflight error")

        results: list[str] = []
        anon_register: list[str] | None = None
        clipboard = HashlineClipboard.get(tool._session)

        for prepared_section in prepared:
            if prepared_section.is_delete:
                err = await self._delete_file(tool, params, prepared_section)
                if err:
                    return err
                results.append(f"Deleted `{prepared_section.display_path}`.")
                continue

            # Resolve paths and registers into concrete HashlineEdits.
            try:
                edits, anon_register = self._build_edits(
                    prepared_section.ops,
                    prepared_section.base_content,
                    clipboard,
                    anon_register,
                )
            except ValueError as e:
                return ToolError(
                    message=f"Register error in `{prepared_section.display_path}`: {e}",
                    brief="Register error",
                )

            new_content, first_changed = apply_hashline_edits(prepared_section.base_content, edits)

            if new_content == prepared_section.base_content:
                return ToolError(
                    message=(
                        f"Edits to `{prepared_section.display_path}` parsed and applied cleanly, "
                        "but produced no change: your body row(s) are byte-identical to the file at the targeted lines. "
                        "The bug is somewhere else — re-read the file before issuing another edit. "
                        "Do NOT widen the payload or add lines; verify the anchor first."
                    ),
                    brief="No change",
                )

            original_for_diff = (
                prepared_section.base_content if prepared_section.using_snapshot
                else await tool._read_text(prepared_section.p)
            )
            diff_blocks = await build_diff_blocks(
                prepared_section.display_path, original_for_diff, new_content
            )
            action = FileActions.EDIT if tool._is_within_workspace(prepared_section.p) else FileActions.EDIT_OUTSIDE
            approval_result = await tool._approval.request(
                "edit",
                action,
                f"Edit file `{prepared_section.display_path}`"
                + (f" — {params.justification}" if params.justification else ""),
                display=diff_blocks,
            )
            if not approval_result:
                return approval_result.rejection_error()

            conflict_err = await self._check_conflicts(tool, prepared_section, params)
            if conflict_err:
                return conflict_err
            if not prepared_section.using_snapshot:
                stale_err = await self._check_staleness(tool, prepared_section)
                if stale_err:
                    return stale_err

            dest = self._section_dest(prepared_section)
            if dest is not None:
                dest_p = await tool._resolve_for_write(dest)
                await tool._ensure_parent(dest_p)
                await tool._write_text(dest_p, new_content)
                await tool._remove_file(prepared_section.p)
                results.append(
                    f"Edited `{prepared_section.display_path}` and moved to `{dest}`."
                )
            else:
                await tool._write_text(prepared_section.p, new_content)
                results.append(f"Edited `{prepared_section.display_path}`.")

            guard = create_edit_parse_guard(
                tool._session,
                variant="hashline",
                arg=params.model_dump(),
            )
            await guard.observe_applied(str(prepared_section.p), original_for_diff, new_content)
            notes = await guard.finish()
            if notes:
                results[-1] += "\n" + "\n".join(notes)

        return ToolReturnValue(
            is_error=False,
            output="",
            message="\n".join(results),
            display=[],
        )

    async def _preflight(self, tool: BaseEditTool, sections: list[HashlineSection]) -> list[_PreparedSection]:
        prepared: list[_PreparedSection] = []
        seen_paths: set[str] = set()

        for section in sections:
            p = kaos_path_from_tool_input(section.path, tool._work_dir)
            display_path = str(p).replace("\\", "/")
            err, _ = await tool._validate_path(p, section.path)
            if err:
                raise ValueError(f"Invalid path `{display_path}`: {err.message}")

            canonical = canonical_snapshot_key(str(p))
            if canonical in seen_paths:
                raise ValueError(
                    f"Multiple hashline sections resolve to the same file `{display_path}`. "
                    "Merge their ops under one header."
                )
            seen_paths.add(canonical)

            using_snapshot = False
            base_content = ""
            try:
                current_content = await tool._read_text(p)
                current_line1 = _line1_hash(current_content)
            except FileNotFoundError:
                current_content = ""
                current_line1 = ""

            if current_line1 == section.tag:
                base_content = current_content
            else:
                store = get_edit_snapshot_store(tool._session)
                entry = store.lookup(str(p))
                if entry is not None:
                    snap_line1 = _line1_hash(entry.content)
                    if snap_line1 == section.tag:
                        base_content = entry.content
                        using_snapshot = True

            if not base_content and not current_content:
                raise ValueError(f"File `{display_path}` does not exist.")
            if not base_content:
                file_lines = current_content.replace("\r\n", "\n").splitlines()
                raise HashlineMismatchError(
                    [HashMismatch(1, section.tag, current_line1)],
                    file_lines,
                )

            file_len = len(base_content.replace("\r\n", "\n").splitlines())
            is_delete = False
            validated_ops: list[HashlineOp] = []

            for op in section.ops:
                if op.kind == "rem":
                    if validated_ops:
                        raise ValueError("`REM` deletes the whole file and cannot be combined with line ops.")
                    is_delete = True
                    validated_ops.append(op)
                    continue
                if op.kind == "mv":
                    validated_ops.append(op)
                    continue
                if op.kind == "cut":
                    start = op.start or 1
                    end = op.end if op.end is not None else start
                    self._check_range(start, end, file_len, op.line_text)
                    validated_ops.append(op)
                    continue
                if op.kind == "put":
                    if op.insert_where == "replace":
                        start = op.start or 1
                        end = op.end if op.end is not None else start
                        self._check_range(start, end, file_len, op.line_text)
                    elif op.insert_where == "before":
                        line = op.start if op.start is not None else 1
                        self._check_line(line, file_len, op.line_text)
                    else:  # after
                        if op.start is not None:
                            self._check_line(op.start, file_len, op.line_text)
                    validated_ops.append(op)

            prepared.append(
                _PreparedSection(
                    section=section,
                    canonical_path=canonical,
                    display_path=display_path,
                    p=p,
                    base_content=base_content,
                    using_snapshot=using_snapshot,
                    ops=validated_ops,
                    is_delete=is_delete,
                )
            )

        return prepared

    @staticmethod
    def _check_line(line: int, file_len: int, line_text: str) -> None:
        if line < 1 or line > file_len:
            raise ValueError(f"Invalid line in `{line_text}`: file has {file_len} lines.")

    @staticmethod
    def _check_range(start: int, end: int, file_len: int, line_text: str) -> None:
        if start < 1 or end < start or start > file_len or end > file_len:
            raise ValueError(f"Invalid range in `{line_text}`: file has {file_len} lines.")

    def _build_edits(
        self,
        ops: list[HashlineOp],
        base_content: str,
        clipboard: HashlineClipboard,
        anon_register: list[str] | None,
    ) -> tuple[list, list[str] | None]:
        """Convert ops to HashlineEdit models, resolving registers."""
        lines = base_content.replace("\r\n", "\n").splitlines()
        edits: list = []

        # Compute cumulative hashes for anchor validation.
        line_hashes: list[str] = []
        prev: str | None = None
        for i, line in enumerate(lines, 1):
            h = compute_line_hash(i, line, prev)
            line_hashes.append(h)
            prev = h

        def _hash(line: int) -> str:
            return line_hashes[line - 1] if 1 <= line <= len(line_hashes) else ""

        for op in ops:
            if op.kind == "cut":
                start = op.start or 1
                end = op.end if op.end is not None else start
                captured = lines[start - 1 : end]
                if op.register is None:
                    anon_register = captured
                else:
                    clipboard.named[op.register] = captured
                edits.append(
                    ReplaceEdit(
                        op="replace",
                        pos=AnchorRef(line=start, hash=_hash(start)),
                        end=AnchorRef(line=end, hash=_hash(end)),
                        lines=[],
                    )
                )
                continue

            if op.kind == "put":
                body = self._resolve_body(op, anon_register, clipboard)
                if op.insert_where == "replace":
                    start = op.start or 1
                    end = op.end if op.end is not None else start
                    edits.append(
                        ReplaceEdit(
                            op="replace",
                            pos=AnchorRef(line=start, hash=_hash(start)),
                            end=AnchorRef(line=end, hash=_hash(end)),
                            lines=body,
                        )
                    )
                elif op.insert_where == "before":
                    line = op.start if op.start is not None else 1
                    edits.append(
                        PrependEdit(
                            op="prepend",
                            pos=AnchorRef(line=line, hash=_hash(line)),
                            lines=body,
                        )
                    )
                else:  # after
                    pos = None if op.start is None else AnchorRef(line=op.start, hash=_hash(op.start))
                    edits.append(AppendEdit(op="append", pos=pos, lines=body))

        return edits, anon_register

    @staticmethod
    def _resolve_body(
        op: HashlineOp,
        anon_register: list[str] | None,
        clipboard: HashlineClipboard,
    ) -> list[str]:
        if op.register is None:
            return list(op.body)
        if op.register in clipboard.named:
            return list(clipboard.named[op.register])
        raise ValueError(f"Register `@{op.register}` is not defined.")

    @staticmethod
    def _section_dest(prepared: _PreparedSection) -> str | None:
        for op in prepared.ops:
            if op.kind == "mv":
                return op.dest
        return None

    async def _delete_file(
        self, tool: BaseEditTool, params: EditParams, prepared: _PreparedSection
    ) -> ToolError | None:
        try:
            st = await prepared.p.stat()
            if not S_ISREG(st.st_mode):
                return ToolError(
                    message=f"`{prepared.display_path}` is not a file.",
                    brief="Invalid path",
                )
        except FileNotFoundError:
            return ToolError(
                message=f"`{prepared.display_path}` does not exist.",
                brief="File not found",
            )
        action = FileActions.EDIT if tool._is_within_workspace(prepared.p) else FileActions.EDIT_OUTSIDE
        approval_result = await tool._approval.request(
            "edit",
            action,
            f"Delete file `{prepared.display_path}`",
            display=[],
        )
        if not approval_result:
            return approval_result.rejection_error()
        await tool._remove_file(prepared.p)
        return None

    async def _check_conflicts(
        self, tool: BaseEditTool, prepared: _PreparedSection, params: EditParams
    ) -> ToolError | None:
        if params.allow_conflicts:
            return None
        content = await tool._read_text(prepared.p)
        markers = ["<<<<<<<", "=======", ">>>>>>>"]
        found: list[tuple[int, str]] = []
        for i, line in enumerate(content.replace("\r\n", "\n").splitlines(), 1):
            stripped = line.strip()
            if stripped in markers or stripped.startswith("<<<<<<< ") or stripped.startswith(">>>>>>> "):
                found.append((i, line))
        if found:
            lines_str = "\n".join(f"  line {n}: {text}" for n, text in found)
            return ToolError(
                message=(
                    f"Conflict markers detected in `{prepared.display_path}`; refusing to edit.\n{lines_str}\n"
                    "Resolve the conflict first or pass allow_conflicts=true."
                ),
                brief="Conflict markers detected",
            )
        return None

    async def _check_staleness(self, tool: BaseEditTool, prepared: _PreparedSection) -> ToolError | None:
        if not tool._session.file_mtime.mark_dirty(str(prepared.p)):
            return ToolError(
                message=(
                    f"`{prepared.display_path}` changed externally or was written after the last read. "
                    "Re-read the file and re-issue the edit."
                ),
                brief="Stale file",
            )
        return None
