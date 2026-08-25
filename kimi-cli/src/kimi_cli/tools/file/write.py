import contextlib
from typing import Literal, override

import json_repair
from kaos.path import KaosPath
from kosong.tooling import CallableTool2, DisplayBlock, ToolError, ToolReturnValue, alias_note
from pydantic import AliasChoices, BaseModel, Field, field_validator

from kimi_cli import logger
from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import DiffDisplayBlock
from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.edit_safety import create_edit_parse_guard
from kimi_cli.tools.file.check_fmt import (
    check_json_text,
    check_toml_text,
    check_xml_text,
    check_yaml_text,
)
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.path import (
    is_within_directory,
    is_within_workspace,
    kaos_path_from_tool_input,
    kaos_path_from_user_input,
)
from kimi_cli.vfs import VFS

from .utils import resolve_vfs

_BASE_DESCRIPTION = "Create or fully replace a UTF-8 text file."

# Fuzzy mode map — maps common synonyms to canonical values
_MODE_MAP: dict[str, Literal["overwrite", "append"]] = {
    # overwrite synonyms
    "overwrite": "overwrite",
    "over_write": "overwrite",
    "over-write": "overwrite",
    "replace": "overwrite",
    "write": "overwrite",
    "create": "overwrite",
    "new": "overwrite",
    "truncate": "overwrite",
    "rewrite": "overwrite",
    "set": "overwrite",
    "put": "overwrite",
    # append synonyms
    "append": "append",
    "add": "append",
    "concat": "append",
    "concatenate": "append",
    "extend": "append",
    "attach": "append",
    "insert": "append",
    "prepend": "append",
    "after": "append",
}


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str = Field(
        validation_alias=AliasChoices("file_path", "path"),
        description="Path to write, resolved by the filesystem backend. "
        + alias_note("file_path", "path", word=False),
    )
    content: str = Field(
        validation_alias=AliasChoices("content", "text"),
        description="Full UTF-8 text content to write. "
        + alias_note("content", "text", word=False),
    )
    sandbox_permissions: Literal["workspace-write", "danger-full-access"] | None = Field(
        default=None,
        description=(
            "The wider sandbox mode this file operation needs "
            "(`workspace-write` or `danger-full-access`). Only valid as a "
            "one-shot retry of an operation the sandbox just denied; requires "
            "justification and user approval."
        ),
    )
    justification: str | None = Field(
        default=None,
        description=(
            "Required with sandbox_permissions: one sentence for the user "
            "explaining why this exact file operation needs the wider access."
        ),
    )
    mode: Literal["overwrite", "append"] = Field(
        description="Write mode: overwrite or append.",
        default="overwrite",
    )
    auto_fix_json: bool = Field(
        default=True,
        description="When True (default), attempt to repair broken JSON before writing. "
        "When False, fail with a format error if JSON is invalid.",
    )
    mkdir: bool = Field(
        default=True,
        description="When True (default), automatically create parent directories. "
        "When False, fail if the parent directory does not exist.",
    )
    show_diff: bool = Field(
        default=False,
        description="When True, include a unified diff in the tool output.",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        normalized = v.strip().lower().replace("-", "_")
        canonical = _MODE_MAP.get(normalized)
        if canonical is None:
            raise ValueError(
                f"Invalid mode '{v}'. Must be 'overwrite' or 'append' (or a known synonym)."
            )
        return canonical


class WriteFile(CallableTool2[Params]):
    name: str = "write"
    description: str = _BASE_DESCRIPTION
    params: type[Params] = Params

    def __init__(self, runtime: Runtime, approval: Approval, session: Session, vfs: VFS | None = None):
        super().__init__()
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._approval = approval
        self._session = session
        self._vfs = vfs
    async def _validate_path(
        self, path: KaosPath, raw_path: str
    ) -> tuple[ToolError | None, bool]:
        """Validate that the path is safe to write.

        Returns:
            A tuple of (error_or_none, is_inside_workspace).
        """
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
                        "You must provide an absolute path to write a file "
                        "outside the working directory."
                    ),
                    brief="Invalid path",
                ),
                False,
            )

        protected_paths = self._session.custom_config.get("config_json", {}).get("protected_write_paths")
        if protected_paths:
            from .utils import check_path_protected
            if matched := check_path_protected(resolved_path, protected_paths, self._work_dir):
                return (
                    ToolError(
                        message=f"Writing to `{path}` is blocked by protected path rule: `{matched}`.",
                        brief="Protected path",
                    ),
                    False,
                )

        return None, inside

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        display_path = params.file_path.replace("\\", "/")
        # TODO: checks:
        # - check if the path may contain secrets
        if not params.file_path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )

        try:
            p = kaos_path_from_tool_input(params.file_path, self._work_dir)
            logical_path = p
            display_logical_path = str(logical_path).replace("\\", "/")
            _outside = not is_within_directory(logical_path.canonical(), self._work_dir)
            err, path_is_inside = await self._validate_path(logical_path, params.file_path)
            if err:
                err.message = f"[out of work-dir] {err.message}" if _outside else err.message
                return err

            p = await resolve_vfs(params.file_path, self._vfs, for_write=True, work_dir=self._work_dir)
            display_p = str(p).replace("\\", "/")

            if await p.is_dir():
                return ToolError(
                    message=f"{'[out of work-dir] ' if _outside else ''}`{display_p}` is a directory, not a file.",
                    brief="Path is a directory",
                )

            # Create parent directories (or fail if mkdir=False)
            parent = p.parent
            if not await parent.exists():
                if params.mkdir:
                    try:
                        await parent.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        return ToolError(
                            message=f"{'[out of work-dir] ' if _outside else ''}Failed to create parent directory for {display_p}: {e}",
                            brief="Parent directory not found",
                        )
                else:
                    return ToolError(
                        message=f"{'[out of work-dir] ' if _outside else ''}Parent directory does not exist: {parent}. Set mkdir=True to create it.",
                        brief="Directory not found",
                    )

            old_text = ""
            file_existed = False
            try:
                old_text = await p.read_text(encoding="utf-8", errors="strict")
                file_existed = True
            except FileNotFoundError:
                pass

            new_text = params.content if params.mode == "overwrite" else old_text + params.content

            # In-memory format validation & fix (before any write)
            fmt_error = None
            file_path_str = str(logical_path)
            is_json = file_path_str.lower().endswith(".json")
            if is_json:
                fmt_error = check_json_text(new_text)
            elif file_path_str.lower().endswith((".yaml", ".yml")):
                fmt_error = check_yaml_text(new_text)
            elif file_path_str.lower().endswith(".toml"):
                fmt_error = check_toml_text(new_text)
            elif file_path_str.lower().endswith(".xml"):
                fmt_error = check_xml_text(new_text)

            # Try to repair broken JSON before building diff (if auto_fix_json is enabled)
            _repair_diff = None
            if is_json and fmt_error:
                if params.auto_fix_json:
                    try:
                        repaired_text = json_repair.repair_json(new_text, return_objects=False)
                        if repaired_text and repaired_text != new_text:
                            repair_diff_blocks = await build_diff_blocks(
                                file_path_str, new_text, repaired_text
                            )
                            new_text = repaired_text
                            fmt_error = None
                            _repair_diff = repair_diff_blocks
                    except Exception:
                        pass
                # else: auto_fix_json=False — leave fmt_error as-is, error reported below

            # Build diff blocks
            diff_blocks: list[DisplayBlock]
            if params.mode == "append" and file_existed:
                # Fast path: synthetic diff for append
                old_lines = old_text.splitlines()
                old_start = max(1, len(old_lines) - 2)
                old_context = "\n".join(old_lines[old_start - 1 :]) if old_lines else ""
                new_context = (
                    (old_context + "\n" if old_context else "") + params.content
                ).rstrip("\n")
                diff_blocks = [
                    DiffDisplayBlock(
                        path=file_path_str,
                        old_text=old_context,
                        new_text=new_context,
                        old_start=old_start,
                        new_start=old_start,
                    )
                ]
            else:
                diff_blocks = await build_diff_blocks(
                    file_path_str,
                    old_text,
                    new_text,
                )

            action = (
                FileActions.EDIT
                if path_is_inside
                else FileActions.EDIT_OUTSIDE
            )

            # Request approval
            approval_text = f"Write file `{display_logical_path}`"
            if params.justification:
                approval_text += f" — {params.justification}"
            result = await self._approval.request(
                self.name,
                action,
                approval_text,
                display=diff_blocks,
            )
            if not result:
                return result.rejection_error()

            # Write content to file
            conflict_err = await self._check_conflicts(display_p, new_text)
            if conflict_err:
                return conflict_err

            if params.mode == "append" and file_existed:
                await p.append_text(params.content, encoding="utf-8", errors="strict")
                expected_size = len(old_text.encode("utf-8")) + len(
                    params.content.encode("utf-8")
                )
            else:
                await p.write_text(new_text, encoding="utf-8", errors="strict")
                expected_size = len(new_text.encode("utf-8"))

            # Post-write size verification
            try:
                actual_size = (await p.stat()).st_size
            except Exception as e:
                logger.warning(
                    "write verification failed: {path}: {error}", path=params.file_path, error=e
                )
                return ToolError(
                    message=(
                        f"{'[out of work-dir] ' if _outside else ''}Write verification failed "
                        f"for {display_p}: {e}."
                    ),
                    brief="Verification failed",
                )
            if actual_size != expected_size:
                return ToolError(
                    message=(
                        f"{'[out of work-dir] ' if _outside else ''}Write verification failed "
                        "(size mismatch): "
                        f"expected {expected_size} bytes, got {actual_size} bytes. "
                        f"Path: {display_path}"
                    ),
                    brief="Verification failed",
                )

            # Compute file size in-memory
            file_size = expected_size
            action_desc = "overwritten" if params.mode == "overwrite" else "appended to"

            guard = create_edit_parse_guard(
                self._session,
                variant="write",
                arg=params.model_dump(),
                enabled_for_write=True,
            )
            await guard.observe_applied(str(logical_path), old_text, new_text)
            notes = await guard.finish()

            if fmt_error:
                msg = f"{'[out of work-dir] ' if _outside else ''}File successfully {action_desc}, but {fmt_error} Path: {display_path}"
                if notes:
                    msg += "\n\n" + "\n\n".join(notes)
                return ToolError(
                    message=msg,
                    brief="Format validation failed",
                )

            # Build result output
            result_parts: list[str] = []
            if params.show_diff:
                # Include diff in output if requested
                diff_text = "\n".join(str(b) for b in diff_blocks)
                if diff_text:
                    result_parts.append(f"Diff:\n{diff_text}\n")
            if _repair_diff:
                repair_text = "\n".join(str(b) for b in _repair_diff)
                result_parts.append(f"JSON auto-repair diff:\n{repair_text}")

            result_output = "\n".join(result_parts)
            if result_output:
                result_output = result_output.rstrip("\n") + "\n"

            message_parts = [
                f"{'[out of work-dir] ' if _outside else ''}File successfully {action_desc}. "
                f"Current size: {file_size} bytes. Path: {display_path}",
                " Verified: size matches.",
            ]
            if _repair_diff:
                message_parts.append(" JSON was auto-repaired.")
            if notes:
                message_parts.append("\n\n" + "\n\n".join(notes))

            return ToolReturnValue(
                is_error=False,
                output=result_output,
                message="".join(message_parts),
                display=[],
            )

        except Exception as e:
            logger.warning(
                "write failed: {path}: {error}", path=params.file_path, error=e
            )
            _outside_ex = False
            with contextlib.suppress(Exception):
                _outside_ex = not is_within_directory(
                    kaos_path_from_tool_input(params.file_path, self._work_dir).canonical(),
                    self._work_dir,
                )
            return ToolError(
                message=f"{'[out of work-dir] ' if _outside_ex else ''}Failed to write to {display_path}. Error: {e}.",
                brief="Failed to write file",
            )

    async def _check_conflicts(
        self,
        display_path: str,
        content: str,
        *,
        allow_conflicts: bool = False,
    ) -> ToolError | None:
        """Pre-apply guard: refuse files containing conflict markers."""
        if allow_conflicts:
            return None
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
                    f"Conflict markers detected in `{display_path}`; refusing to write.\n{lines_str}\n"
                    "Resolve the conflict first."
                ),
                brief="Conflict markers detected",
            )
        return None
