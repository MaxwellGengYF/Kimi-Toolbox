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
