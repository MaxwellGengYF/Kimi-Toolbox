"""Multi-mode edit tool dispatcher."""

from __future__ import annotations

from kosong.tooling import CallableTool2, ToolError, ToolReturnValue

from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.vfs import VFS

from .base import BaseEditTool
from .modes import MODE_REGISTRY
from .modes.replace import ReplaceModeExecutor
from .params import EditMode, EditParams, ReplaceEditItem, normalize_edit_mode


class EditFile(CallableTool2[EditParams]):
    """Edit tool supporting replace, patch, hashline, and sloppy modes."""

    name: str = "edit"
    description: str = ReplaceModeExecutor.description
    params: type[EditParams] = EditParams

    def __init__(self, runtime: Runtime, approval: Approval, session: Session, vfs: VFS | None = None):
        super().__init__()
        self._tool = BaseEditTool(runtime, approval, session, vfs)

    @property
    def _work_dir(self):
        return self._tool._work_dir

    @property
    def _approval(self):
        return self._tool._approval

    @property
    def _session(self):
        return self._tool._session

    @property
    def _vfs(self):
        return self._tool._vfs

    async def __call__(self, params: EditParams) -> ToolReturnValue:
        if params.resolved_mode is None:
            # Validation should always populate this, but repair if needed.
            params.resolved_mode = normalize_edit_mode(params.mode) or "replace"

        executor_cls = MODE_REGISTRY.get(params.resolved_mode)
        if executor_cls is None:
            return ToolError(
                message=f"Invalid edit mode: {params.mode}",
                brief="Invalid edit mode",
            )
        return await executor_cls().execute(self._tool, params)

    # -----------------------------------------------------------------------
    # Backward-compatible helpers exposed by the old EditFile implementation.
    # These are stateless so they work when the class is instantiated via
    # object.__new__ (used by the existing test suite).
    # -----------------------------------------------------------------------
    def _apply_edit(self, content: str, edit: ReplaceEditItem) -> tuple[str, int, str | None]:
        return ReplaceModeExecutor()._apply_edit(content, edit)

    def _normalize_line_endings(self, text: str) -> str:
        return ReplaceModeExecutor()._normalize_line_endings(text)

    def _find_similar(self, target: str, content: str, cutoff: float = 75.0) -> str | None:
        return ReplaceModeExecutor()._find_similar(target, content, cutoff)

    def _try_strip_match(self, content: str, old: str, new: str) -> str | None:
        return ReplaceModeExecutor()._try_strip_match(content, old, new)

    def _find_best_fuzzy_match(
        self, target: str, content: str, cutoff: float = 75.0
    ) -> tuple[str, float] | None:
        return ReplaceModeExecutor()._find_best_fuzzy_match(target, content, cutoff)

    def _apply_replace_all(
        self,
        content: str,
        norm_content: str,
        norm_old: str,
        norm_new: str,
        edit: ReplaceEditItem,
    ) -> tuple[str, int, str | None]:
        return ReplaceModeExecutor()._apply_replace_all(content, norm_content, norm_old, norm_new, edit)

    def _apply_fuzzy_fallback(
        self,
        content: str,
        norm_content: str,
        norm_old: str,
        norm_new: str,
        edit: ReplaceEditItem,
    ) -> tuple[str, int, str | None]:
        return ReplaceModeExecutor()._apply_fuzzy_fallback(content, norm_content, norm_old, norm_new, edit)


__all__ = [
    "EditFile",
    "EditMode",
    "EditParams",
    "ReplaceEditItem",
    "normalize_edit_mode",
]
