"""Patch-mode executor for the multi-mode edit tool."""

from __future__ import annotations

from stat import S_ISREG

from kaos.path import KaosPath
from kosong.tooling import ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.edit.diff import (
    ApplyPatchError,
    apply_diff_hunks,
    normalize_create_content,
    parse_diff_hunks,
)
from kimi_cli.tools.file.edit.params import EditMode, EditParams, PatchEntry
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import is_within_directory, kaos_path_from_tool_input

from ..base import BaseEditTool


class PatchModeExecutor:
    """Executor for patch-mode edits: create/update/delete with unified-diff hunks."""

    mode: EditMode = "patch"
    description: str = "Apply unified-diff style edits to a file."

    async def execute(self, tool: BaseEditTool, params: EditParams) -> ToolReturnValue:
        file_path = params.file_path or ""
        if not file_path:
            return ToolError(message="File path cannot be empty.", brief="Empty file path")

        entries: list[PatchEntry]
        if params.edit is None:
            return ToolError(message="Patch mode requires edits.", brief="Missing edits")
        if isinstance(params.edit, list):
            entries = [e if isinstance(e, PatchEntry) else PatchEntry.model_validate(e) for e in params.edit]
        else:
            entries = [PatchEntry.model_validate(params.edit)]

        if not entries:
            return ToolError(message="Patch mode requires at least one edit.", brief="Empty edits")

        base_p = kaos_path_from_tool_input(file_path, tool._work_dir)
        display_path = str(base_p).replace("\\", "/")
        _outside = not is_within_directory(base_p.canonical(), tool._work_dir)

        err, _ = await tool._validate_path(base_p, file_path)
        if err:
            if _outside:
                err.message = f"[out of work-dir] {err.message}"
            return err

        applied: list[str] = []
        skipped: list[str] = []

        for idx, entry in enumerate(entries):
            try:
                await self._apply_entry(tool, file_path, entry, _outside, params.allow_conflicts)
            except ApplyPatchError as e:
                skipped = [f"{display_path}#{i}" for i in range(idx, len(entries))]
                msg_parts = [f"{tool._out_prefix(_outside)}Patch failed for `{display_path}`: {e.message}"]
                if applied:
                    msg_parts.append(f"Entries already applied: {', '.join(applied)}.")
                if skipped[1:]:
                    msg_parts.append(f"Entries NOT applied: {', '.join(skipped[1:])}.")
                return ToolError(message="\n\n".join(msg_parts), brief="Patch failed")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("patch failed: {path}: {error}", path=file_path, error=e)
                return ToolError(
                    message=f"{tool._out_prefix(_outside)}Failed to apply patch. Error: {e} Path: {display_path}",
                    brief="Failed to apply patch",
                )
            applied.append(f"{display_path}#{idx} ({entry.op})")

        return ToolReturnValue(
            is_error=False,
            output="",
            message=f"{tool._out_prefix(_outside)}File successfully patched. Applied {len(entries)} patch operation(s).",
            display=[],
        )

    async def _apply_entry(
        self,
        tool: BaseEditTool,
        base_path: str,
        entry: PatchEntry,
        _outside: bool,
        allow_conflicts: bool = False,
    ) -> None:
        base_p = kaos_path_from_tool_input(base_path, tool._work_dir)
        display_path = str(base_p).replace("\\", "/")

        if entry.op == "create":
            p = await tool._resolve_for_write(base_path)
            try:
                await p.stat()
                exists = True
            except FileNotFoundError:
                exists = False
            if exists:
                raise ApplyPatchError(f"File already exists: `{display_path}`.")
            content = normalize_create_content(entry.diff or "")
            diff_blocks = await build_diff_blocks(str(base_p), "", content)
            action = FileActions.EDIT if tool._is_within_workspace(p) else FileActions.EDIT_OUTSIDE
            approval = await tool._approval.request(
                "edit",
                action,
                f"Create file `{display_path}`",
                display=diff_blocks,
            )
            if not approval:
                raise ApplyPatchError(approval.rejection_error().message)
            conflict_err = await tool._check_conflicts(display_path, "", allow_conflicts=allow_conflicts)
            if conflict_err:
                raise ApplyPatchError(conflict_err.message)

            await tool._ensure_parent(p)
            fmt_error, is_json = await tool._check_format(str(base_p), content)
            if is_json and fmt_error:
                repaired = await tool._try_repair_json(content)
                if repaired is not None:
                    content = repaired
                    fmt_error = None
            await tool._write_text(p, content)
            if fmt_error:
                raise ApplyPatchError(f"File created, but {fmt_error}")
            return

        if entry.op == "delete":
            p = await tool._resolve_for_write(base_path)
            try:
                st = await p.stat()
                if not S_ISREG(st.st_mode):
                    raise ApplyPatchError(f"`{display_path}` is not a file.")
            except FileNotFoundError:
                raise ApplyPatchError(f"File not found: `{display_path}`.")
            action = FileActions.EDIT if tool._is_within_workspace(p) else FileActions.EDIT_OUTSIDE
            approval = await tool._approval.request(
                "edit",
                action,
                f"Delete file `{display_path}`",
                display=[],
            )
            if not approval:
                raise ApplyPatchError(approval.rejection_error().message)
            await tool._remove_file(p)
            return

        # op == "update"
        p = await tool._resolve_for_write(base_path)
        try:
            st = await p.stat()
            if not S_ISREG(st.st_mode):
                raise ApplyPatchError(f"`{display_path}` is not a file.")
        except FileNotFoundError:
            raise ApplyPatchError(f"File not found: `{display_path}`.")

        original_content = await tool._read_text(p)
        hunks = parse_diff_hunks(entry.diff or "")
        if not hunks:
            raise ApplyPatchError("No patch hunks found in diff.")
        new_content, _ = apply_diff_hunks(original_content, hunks, allow_fuzzy=True, threshold=0.75)

        rename_path: KaosPath | None = None
        display_rename: str | None = None
        if entry.rename:
            rename_p = kaos_path_from_tool_input(entry.rename, tool._work_dir)
            rename_path = rename_p
            display_rename = str(rename_p).replace("\\", "/")
            err, _ = await tool._validate_path(rename_p, entry.rename)
            if err:
                raise ApplyPatchError(f"Invalid rename target: {err.message}")

        diff_blocks = await build_diff_blocks(str(base_p), original_content, new_content)
        action = FileActions.EDIT if tool._is_within_workspace(p) else FileActions.EDIT_OUTSIDE
        approval = await tool._approval.request(
            "edit",
            action,
            f"Patch file `{display_path}`"
            + (f" and move to `{display_rename}`" if rename_path else ""),
            display=diff_blocks,
        )
        if not approval:
            raise ApplyPatchError(approval.rejection_error().message)

        fmt_error, is_json = await tool._check_format(str(base_p), new_content)
        if is_json and fmt_error:
            repaired = await tool._try_repair_json(new_content)
            if repaired is not None:
                new_content = repaired
                fmt_error = None

        conflict_err = await tool._check_conflicts(display_path, original_content, allow_conflicts=allow_conflicts)
        if conflict_err:
            raise ApplyPatchError(conflict_err.message)

        await tool._write_text(p, new_content)
        if fmt_error:
            raise ApplyPatchError(f"File patched, but {fmt_error}")

        if rename_path:
            rename_target = await tool._resolve_for_write(entry.rename or "")
            await tool._ensure_parent(rename_target)
            await tool._write_text(rename_target, new_content)
            await tool._remove_file(p)


class ApplyPatchInput(BaseModel):
    """Parameters for the standalone apply_patch tool."""

    input: str = Field(description="The full Codex apply_patch envelope.")


class ApplyPatchHunk:
    """A single file operation parsed from a Codex apply_patch envelope."""

    def __init__(self, path: str, op: str, rename: str | None = None, diff: str | None = None) -> None:
        self.path = path
        self.op = op
        self.rename = rename
        self.diff = diff


BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "


def parse_apply_patch(input_text: str) -> list[ApplyPatchHunk]:
    """Parse a Codex apply_patch envelope into a list of file operations."""
    lines = input_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Drop empty leading/trailing lines so a trailing newline doesn't defeat marker checks.
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    # Strip heredoc wrapper if present.
    if len(lines) >= 2:
        first = lines[0].strip()
        last = lines[-1].strip()
        if first in {"<<EOF", "<<'EOF'", '<<"EOF"'} and last == "EOF":
            lines = lines[1:-1]
            while lines and lines[-1].strip() == "":
                lines.pop()

    if not lines or lines[0].strip() != BEGIN_PATCH_MARKER:
        raise ApplyPatchError("The first line of the patch must be '*** Begin Patch'")
    if len(lines) == 1 or lines[-1].strip() != END_PATCH_MARKER:
        raise ApplyPatchError("The last line of the patch must be '*** End Patch'")

    remaining = lines[1:-1]
    hunks: list[ApplyPatchHunk] = []

    while remaining:
        line = remaining[0].strip()
        if line == "":
            remaining = remaining[1:]
            continue

        if line.startswith(ADD_FILE_MARKER):
            path = line[len(ADD_FILE_MARKER) :]
            contents: list[str] = []
            consumed = 1
            for raw in remaining[1:]:
                if raw.startswith("+"):
                    contents.append(raw[1:])
                    consumed += 1
                else:
                    break
            hunks.append(ApplyPatchHunk(path=path, op="create", diff="\n".join(contents)))
            remaining = remaining[consumed:]
            continue

        if line.startswith(DELETE_FILE_MARKER):
            path = line[len(DELETE_FILE_MARKER) :]
            hunks.append(ApplyPatchHunk(path=path, op="delete"))
            remaining = remaining[1:]
            continue

        if line.startswith(UPDATE_FILE_MARKER):
            path = line[len(UPDATE_FILE_MARKER) :]
            remaining = remaining[1:]
            move_path: str | None = None
            if remaining and remaining[0].startswith(MOVE_TO_MARKER):
                move_path = remaining[0][len(MOVE_TO_MARKER) :]
                remaining = remaining[1:]

            diff_lines: list[str] = []
            while remaining:
                next_line = remaining[0]
                if (
                    next_line.startswith("*** Add File:")
                    or next_line.startswith("*** Delete File:")
                    or next_line.startswith("*** Update File:")
                ):
                    break
                diff_lines.append(next_line)
                remaining = remaining[1:]

            if not diff_lines:
                raise ApplyPatchError(f"Update file hunk for path '{path}' is empty")
            hunks.append(ApplyPatchHunk(path=path, op="update", rename=move_path, diff="\n".join(diff_lines)))
            continue

        raise ApplyPatchError(
            f"'{line}' is not a valid hunk header. "
            "Valid headers: '*** Add File: {{path}}', '*** Delete File: {{path}}', '*** Update File: {{path}}'"
        )

    if not hunks:
        raise ApplyPatchError("No files were modified.")
    return hunks


class ApplyPatchModeExecutor:
    """Executor for the Codex apply_patch envelope."""

    mode: EditMode = "apply_patch"
    description: str = "Apply a Codex-style apply_patch envelope to multiple files."

    async def execute(self, tool: BaseEditTool, params: EditParams | ApplyPatchInput) -> ToolReturnValue:
        input_text = params.input if isinstance(params, ApplyPatchInput) else params.input
        if not input_text:
            return ToolError(message="apply_patch requires an input envelope.", brief="Missing input")

        try:
            hunks = parse_apply_patch(input_text)
        except ApplyPatchError as e:
            return ToolError(message=f"Failed to parse apply_patch envelope: {e.message}", brief="Parse error")

        # Preflight: resolve and validate every path before touching files.
        preflight: list[tuple[str, ApplyPatchHunk]] = []
        for hunk in hunks:
            p = kaos_path_from_tool_input(hunk.path, tool._work_dir)
            display_path = str(p).replace("\\", "/")
            err, _ = await tool._validate_path(p, hunk.path)
            if err:
                return ToolError(
                    message=f"Invalid path `{display_path}`: {err.message}",
                    brief="Invalid path",
                )
            preflight.append((display_path, hunk))

        patch_executor = PatchModeExecutor()
        applied: list[str] = []
        skipped: list[str] = []

        for display_path, hunk in preflight:
            entry = PatchEntry(op=hunk.op, diff=hunk.diff, rename=hunk.rename)
            edit_params = EditParams(file_path=hunk.path, edit=[entry], mode="patch")
            try:
                result = await patch_executor.execute(tool, edit_params)
            except ApplyPatchError as e:
                skipped.append(display_path)
                msg_parts = [f"Patch failed for `{display_path}`: {e.message}"]
                if applied:
                    msg_parts.append(f"Files already applied: {', '.join(applied)}.")
                if skipped[1:]:
                    msg_parts.append(
                        f"Files NOT applied: {', '.join(skipped[1:])}; "
                        "re-read the affected files and re-issue only the failed and unapplied files."
                    )
                return ToolError(message="\n\n".join(msg_parts), brief="Apply patch failed")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("apply_patch failed for {path}: {error}", path=hunk.path, error=e)
                return ToolError(
                    message=f"Failed to apply patch to `{display_path}`: {e}",
                    brief="Failed to apply patch",
                )
            if result.is_error:
                skipped.append(display_path)
                msg_parts = [f"Patch failed for `{display_path}`: {result.message}"]
                if applied:
                    msg_parts.append(f"Files already applied: {', '.join(applied)}.")
                if skipped[1:]:
                    msg_parts.append(
                        f"Files NOT applied: {', '.join(skipped[1:])}; "
                        "re-read the affected files and re-issue only the failed and unapplied files."
                    )
                return ToolError(message="\n\n".join(msg_parts), brief="Apply patch failed")
            applied.append(display_path)

        return ToolReturnValue(
            is_error=False,
            output="",
            message=f"Successfully applied patch to {len(applied)} file(s).",
            display=[],
        )
