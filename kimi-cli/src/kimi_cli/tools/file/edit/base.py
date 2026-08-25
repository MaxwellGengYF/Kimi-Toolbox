"""Shared base class and flow for edit-mode executors."""

from __future__ import annotations

import contextlib
from pathlib import Path
from stat import S_ISREG
from typing import Any

import json_repair
from kaos.path import KaosPath
from kosong.tooling import ToolError, ToolReturnValue

from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import DisplayBlock
from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.check_fmt import (
    check_json_text,
    check_toml_text,
    check_xml_text,
    check_yaml_text,
)
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import (
    is_within_directory,
    is_within_workspace,
    kaos_path_from_tool_input,
    kaos_path_from_user_input,
)
from kimi_cli.vfs import VFS

from ..utils import check_path_protected, resolve_vfs


class BaseEditTool:
    """Shared flow for file edit tools: path validation, VFS, approval, format checks."""

    def __init__(self, runtime: Runtime, approval: Approval, session: Session, vfs: VFS | None = None):
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._approval = approval
        self._session = session
        self._vfs = vfs

    def _out_prefix(self, is_outside: bool) -> str:
        return "[out of work-dir] " if is_outside else ""

    async def _validate_path(
        self, path: KaosPath, raw_path: str
    ) -> tuple[ToolError | None, bool]:
        """Validate that the path is safe to edit.

        Returns a tuple of (error_or_none, is_inside_workspace).
        """
        resolved_path = path.canonical()
        original_is_absolute = kaos_path_from_user_input(raw_path).is_absolute()

        inside = is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
        if not inside and not original_is_absolute:
            return (
                ToolError(
                    message=(
                        f"`{raw_path}` is not an absolute path. "
                        "You must provide an absolute path to edit a file "
                        "outside the working directory."
                    ),
                    brief="Invalid path",
                ),
                False,
            )
        protected_paths = self._session.custom_config.get("config_json", {}).get("protected_write_paths")
        if protected_paths:
            if matched := check_path_protected(resolved_path, protected_paths, self._work_dir):
                return (
                    ToolError(
                        message=f"Editing `{path}` is blocked by protected path rule: `{matched}`.",
                        brief="Protected path",
                    ),
                    False,
                )
        return None, inside

    async def _resolve_for_write(self, path_str: str) -> KaosPath:
        return await resolve_vfs(
            path_str, self._vfs, for_write=True, work_dir=self._work_dir
        )

    def _is_within_workspace(self, path: KaosPath) -> bool:
        return is_within_workspace(path, self._work_dir, self._additional_dirs)

    async def _read_text(self, path: KaosPath) -> str:
        return await path.read_text(errors="replace")

    async def _write_text(self, path: KaosPath, content: str) -> None:
        await path.write_text(content, errors="replace")

    async def _remove_file(self, path: KaosPath) -> None:
        Path(str(path)).unlink()

    async def _ensure_parent(self, path: KaosPath) -> None:
        parent = Path(str(path)).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    async def _check_format(self, file_path_str: str, content: str) -> tuple[str | None, bool]:
        """Return (format_error_message, is_json) for *content*."""
        suffix = Path(file_path_str).suffix.lower()
        is_json = suffix == ".json"
        fmt_error: str | None = None
        if is_json:
            fmt_error = check_json_text(content)
        elif suffix in (".yaml", ".yml"):
            fmt_error = check_yaml_text(content)
        elif suffix == ".toml":
            fmt_error = check_toml_text(content)
        elif suffix == ".xml":
            fmt_error = check_xml_text(content)
        return fmt_error, is_json

    async def _try_repair_json(self, content: str) -> str | None:
        try:
            repaired = json_repair.repair_json(content, return_objects=False)
            if repaired:
                return repaired
        except Exception:
            logger.debug("json repair failed")
        return None

    async def _request_approval(
        self,
        name: str,
        action: FileActions,
        display_path: str,
        justification: str | None,
        original_content: str,
        new_content: str,
    ) -> ToolReturnValue | None:
        """Request approval and return a ToolReturnValue on rejection, or None if approved."""
        diff_blocks: list[DisplayBlock] = await build_diff_blocks(
            display_path, original_content, new_content
        )
        prompt_text = f"Edit file `{display_path}`"
        if justification:
            prompt_text += f" — {justification}"
        result = await self._approval.request(
            name,
            action,
            prompt_text,
            display=diff_blocks,
        )
        if not result:
            return result.rejection_error()
        return None

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
                    f"Conflict markers detected in `{display_path}`; refusing to edit.\n{lines_str}\n"
                    "Resolve the conflict first or pass allow_conflicts=true."
                ),
                brief="Conflict markers detected",
            )
        return None

    def _check_staleness(self, display_path: str, p: KaosPath) -> ToolError | None:
        """Staleness guard before writing."""
        if not self._session.file_mtime.mark_dirty(str(p)):
            return ToolError(
                message=(
                    f"`{display_path}` changed externally or was written after the last read. "
                    "Re-read the file and re-issue the edit."
                ),
                brief="Stale file",
            )
        return None

    async def _stat_is_regular_file(self, path: KaosPath) -> tuple[bool, bool]:
        """Return (exists_and_is_regular_file, is_directory_or_other)."""
        try:
            st = await path.stat()
            return S_ISREG(st.st_mode), not S_ISREG(st.st_mode)
        except FileNotFoundError:
            return False, False

    def _wrap_exception(
        self, params: Any, display_path: str, exc: Exception
    ) -> ToolError:
        logger.warning("edit failed: {path}: {error}", path=params.file_path, error=exc)
        _outside = False
        with contextlib.suppress(Exception):
            _outside = not is_within_directory(
                kaos_path_from_tool_input(params.file_path or display_path, self._work_dir).canonical(),
                self._work_dir,
            )
        return ToolError(
            message=f"{self._out_prefix(_outside)}Failed to edit. Error: {exc} Path: {display_path}",
            brief="Failed to edit file",
        )
