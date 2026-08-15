import asyncio
import contextlib
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal, override

import json_repair
from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolReturnValue, alias_note
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from rapidfuzz import fuzz, process

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

from .utils import resolve_vfs

_BASE_DESCRIPTION = "Edit an existing UTF-8 text file by replacing literal text."


class Edit(BaseModel):
    model_config = {"populate_by_name": True}

    old: str = Field(
        alias="old_string",  # common LLM variant
        description="String to replace. " + alias_note("old", "old_string", word=False),
    )
    new: str = Field(
        alias="new_string",  # common LLM variant
        description="Replacement string. " + alias_note("new", "new_string", word=False),
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences. When False, only the first occurrence is replaced.",
    )
    max_replacements: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of occurrences to replace when replace_all=True. "
        "None means unlimited.",
    )
    match_mode: Literal["exact", "fuzzy"] = Field(
        default="fuzzy",
        description="'fuzzy' (default): Use fuzzy matching when exact match fails "
        "(may match similar text). 'exact': Only replace literal matches of `old`.",
    )


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str = Field(
        validation_alias=AliasChoices("file_path", "path"),
        description="Path to edit, resolved by the filesystem backend. "
        + alias_note("file_path", "path", word=False),
    )
    edit: Edit | list[Edit] = Field(
        alias="edits",  # common LLM variant (plural)
        description=(
            "One or more edits to apply, in order. "
            + alias_note("edit", "edits", word=False)
            + " Each item: `old_string` (literal text to replace), "
            "`new_string` (replacement text; empty string deletes the match), "
            "`replace_all` (default false; when false, old_string must appear "
            "exactly once), plus optional `max_replacements` and `match_mode`."
        ),
    )
    old_string: str | None = Field(
        default=None,
        description="Literal text to replace. Must match exactly. "
        "Single-edit shorthand for `edit` — mutually exclusive with it.",
    )
    new_string: str | None = Field(
        default=None,
        description="Literal replacement text. Use an empty string to delete the match. "
        "Single-edit shorthand for `edit` — mutually exclusive with it.",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "Replace all matches. Defaults to false; when false, old_string "
            "must appear exactly once."
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_single_edit(cls, data: Any) -> Any:
        """Build the `edit` list from top-level old_string/new_string shorthand.

        A bare ``{file_path, old_string, new_string, replace_all}`` call (the
        report's canonical shape) is normalized into the single-edit path so
        the existing ``params.edit`` execution flow is shared.
        """
        if isinstance(data, dict) and "edit" not in data and "edits" not in data:
            if data.get("old_string") is not None or data.get("new_string") is not None:
                edit = {
                    "old_string": data.get("old_string", ""),
                    "new_string": data.get("new_string", ""),
                    "replace_all": data.get("replace_all", False),
                }
                data = {**data, "edit": [edit]}
        return data

    @field_validator("edit", mode="before")
    @classmethod
    def _normalize_edit(cls, v: Any) -> list[dict]:
        """Auto-wrap a single Edit dict into a list for backward compat."""
        if isinstance(v, dict):
            return [v]
        if isinstance(v, Edit):
            return [v.model_dump()]
        return v

class EditFile(CallableTool2[Params]):
    name: str = "edit"
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
        """Validate that the path is safe to edit.

        Returns:
            A tuple of (error_or_none, is_inside_workspace).
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
            from .utils import check_path_protected
            if matched := check_path_protected(resolved_path, protected_paths, self._work_dir):
                return (
                    ToolError(
                        message=f"Editing `{path}` is blocked by protected path rule: `{matched}`.",
                        brief="Protected path",
                    ),
                    False,
                )
        return None, inside

    def _normalize_line_endings(self, text: str) -> str:
        """Normalize \\r\\n to \\n for comparison."""
        return text.replace("\r\n", "\n")

    def _find_similar(self, target: str, content: str, cutoff: float = 75.0) -> str | None:
        """Find the most similar line or chunk in content to target."""
        norm_target = self._normalize_line_endings(target)
        norm_content = self._normalize_line_endings(content)
        lines = norm_content.splitlines()
        if not lines:
            return None

        def _best(candidates: list[str]) -> str | None:
            result = process.extractOne(norm_target, candidates, scorer=fuzz.ratio)
            if result and result[1] >= cutoff:
                return result[0]
            return None

        # Line-level matching first (covers single- and multi-line targets).
        match = _best(lines)
        if match is not None:
            return match

        # Fallback: sliding windows of equal line count for multi-line targets.
        target_lines = norm_target.splitlines()
        target_line_count = len(target_lines)
        if target_line_count > 1 and len(lines) >= target_line_count:
            windows = [
                "\n".join(lines[i : i + target_line_count])
                for i in range(len(lines) - target_line_count + 1)
            ]
            return _best(windows)

        return None

    def _try_strip_match(
        self, content: str, old: str, new: str
    ) -> str | None:
        """Try to find *old* inside any line of *content* ignoring leading/trailing whitespace.

        Returns the updated content with the first such occurrence replaced, or None.
        """
        old_stripped = old.strip()
        if not old_stripped:
            return None

        # Search line-by-line so we can map back to the original line text
        for line in content.splitlines(keepends=True):
            line_core = line.rstrip("\n").rstrip("\r")
            idx = line_core.find(old_stripped)
            if idx != -1:
                # Rebuild the line: preserve prefix/suffix whitespace around the match
                prefix = line_core[:idx]
                suffix = line_core[idx + len(old_stripped) :]
                # Keep original line ending
                ending = ""
                if line.endswith("\r\n"):
                    ending = "\r\n"
                elif line.endswith("\n"):
                    ending = "\n"
                elif line.endswith("\r"):
                    ending = "\r"
                new_line = prefix + new + suffix + ending
                # Replace only the first occurrence of this exact line in content
                return content.replace(line, new_line, 1)
        return None

    def _find_best_fuzzy_match(
        self, target: str, content: str, cutoff: float = 75.0
    ) -> tuple[str, float] | None:
        """Find the best fuzzy match of target in content.

        Returns the matched original text and similarity score, or None.
        """
        norm_target = self._normalize_line_endings(target)
        norm_content = self._normalize_line_endings(content)

        best_score = 0.0
        best_original = None

        target_lines = norm_target.splitlines()
        target_line_count = len(target_lines)

        # Split original content into lines (without line endings)
        original_lines = content.splitlines()
        norm_lines = norm_content.splitlines()

        if target_line_count == 1:
            for orig_line, norm_line in zip(original_lines, norm_lines, strict=False):
                score = fuzz.ratio(norm_target, norm_line)
                if score > best_score:
                    best_score = score
                    best_original = orig_line
        else:
            for i in range(len(norm_lines) - target_line_count + 1):
                window = "\n".join(norm_lines[i : i + target_line_count])
                score = fuzz.ratio(norm_target, window)
                if score > best_score:
                    best_score = score
                    best_original = "\n".join(
                        original_lines[i : i + target_line_count]
                    )

        if best_score >= cutoff:
            return best_original, best_score

        return None

    def _apply_replace_all(
        self,
        content: str,
        norm_content: str,
        norm_old: str,
        norm_new: str,
        edit: Edit,
    ) -> tuple[str, int, str | None]:
        """Apply a replace-all edit using exact matching only.

        Fuzzy fallback is intentionally NOT applied here: blindly fuzzy-matching
        "all" occurrences is too ambiguous to be safe.
        """
        if edit.max_replacements is not None:
            # Replace only the first max_replacements occurrences
            count = 0
            result = norm_content
            while count < edit.max_replacements:
                idx = result.find(norm_old)
                if idx == -1:
                    break
                result = result[:idx] + norm_new + result[idx + len(norm_old) :]
                count += 1
            if count == 0:
                return content, 0, self._find_similar(edit.old, content)
            return result, count, None

        count = norm_content.count(norm_old)
        if count == 0:
            return content, 0, self._find_similar(edit.old, content)
        return norm_content.replace(norm_old, norm_new), count, None

    def _apply_fuzzy_fallback(
        self,
        content: str,
        norm_content: str,
        norm_old: str,
        norm_new: str,
        edit: Edit,
    ) -> tuple[str, int, str | None]:
        """Fallback chain for fuzzy mode after an exact single match fails.

        Tries, in order:
          1. a strip match (ignores leading/trailing whitespace),
          2. a best-effort fuzzy match,
          3. a similarity suggestion for the error message.
        """
        # Try strip match (ignores leading/trailing spaces)
        stripped = self._try_strip_match(content, edit.old, edit.new)
        if stripped is not None:
            return stripped, 1, None

        # Strip match failed — try fuzzy match
        fuzzy = self._find_best_fuzzy_match(edit.old, content)
        if fuzzy is not None:
            matched_text, score = fuzzy
            # Replace in normalized content so line endings stay consistent
            new_content = norm_content.replace(
                self._normalize_line_endings(matched_text), norm_new, 1
            )
            # Return score info via suggestion field for logging
            suggestion = f"fuzzy-matched at {score:.0f}%: '{matched_text[:80]}'"
            return new_content, 1, suggestion

        # No match at all — return suggestion for error message
        return content, 0, self._find_similar(edit.old, content)

    def _apply_edit(self, content: str, edit: Edit) -> tuple[str, int, str | None]:
        """Apply a single edit to the content.

        Returns (new_content, replacements_made, suggestion_or_None).
        """
        if not edit.old or edit.old == edit.new:
            return content, 0, None

        norm_content = self._normalize_line_endings(content)
        norm_old = self._normalize_line_endings(edit.old)
        norm_new = self._normalize_line_endings(edit.new)

        if edit.replace_all:
            return self._apply_replace_all(
                content, norm_content, norm_old, norm_new, edit
            )

        # Single replacement with normalized line endings
        idx = norm_content.find(norm_old)
        if idx != -1:
            return norm_content.replace(norm_old, norm_new, 1), 1, None

        # Exact match failed. In exact mode no fuzzy heuristics are applied —
        # report the closest line as a suggestion only.
        if edit.match_mode == "exact":
            return content, 0, self._find_similar(edit.old, content)

        # Fuzzy mode (the default): try progressively looser fallbacks.
        return self._apply_fuzzy_fallback(
            content, norm_content, norm_old, norm_new, edit
        )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        display_path = params.file_path.replace("\\", "/")
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
            err, _ = await self._validate_path(p, params.file_path)
            if err:
                if _outside:
                    err.message = f"[out of work-dir] {err.message}"
                return err

            p = await resolve_vfs(params.file_path, self._vfs, for_write=True, work_dir=self._work_dir)

            try:
                st = await p.stat()
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

            # Read the file content
            content = await p.read_text(errors="replace")

            original_content = content
            edits = params.edit

            def _work() -> tuple[str, int, str | None]:
                text = content
                total = 0
                last_suggestion = None
                for edit in edits:
                    text, n, suggestion = self._apply_edit(text, edit)
                    total += n
                    if suggestion:
                        last_suggestion = suggestion
                return text, total, last_suggestion

            new_content, total_replacements, suggestion = await asyncio.to_thread(_work)

            # Check if any changes were made
            if new_content == original_content:
                msg = f"{'[out of work-dir] ' if _outside else ''}No replacements were made. The old string was not found in the file."
                if suggestion:
                    msg += f"\n\nDid you mean:\n  {suggestion}"
                return ToolError(
                    message=msg,
                    brief="No replacements made",
                )

            # Build result message with fuzzy match info if applicable
            result_msg_parts = [
                f"{'[out of work-dir] ' if _outside else ''}File successfully edited. "
                f"Applied {len(edits)} edit(s) with {total_replacements} total replacement(s)."
            ]
            if suggestion and "fuzzy-matched" in suggestion:
                result_msg_parts.append(f" ({suggestion})")

            diff_blocks: list[DisplayBlock] = await build_diff_blocks(
                str(logical_path), original_content, new_content
            )

            action = (
                FileActions.EDIT
                if is_within_workspace(p, self._work_dir, self._additional_dirs)
                else FileActions.EDIT_OUTSIDE
            )

            result = await self._approval.request(
                self.name,
                action,
                f"Edit file `{display_logical_path}`"
                + (f" — {params.justification}" if params.justification else ""),
                display=diff_blocks,
            )
            if not result:
                return result.rejection_error()

            # Fix JSON format before writing if needed
            file_path_str = str(logical_path)
            fmt_error = None
            suffix = Path(file_path_str).suffix.lower()
            is_json = suffix == ".json"
            if is_json:
                fmt_error = check_json_text(new_content)
            elif suffix in (".yaml", ".yml"):
                fmt_error = check_yaml_text(new_content)
            elif suffix == ".toml":
                fmt_error = check_toml_text(new_content)
            elif suffix == ".xml":
                fmt_error = check_xml_text(new_content)

            # Try to repair broken JSON before writing
            if is_json and fmt_error:
                try:
                    repaired_text = json_repair.repair_json(new_content, return_objects=False)
                    if repaired_text:
                        new_content = repaired_text
                        fmt_error = None
                        diff_blocks = await build_diff_blocks(
                            str(logical_path), original_content, new_content
                        )
                except Exception:
                    pass

            # Write the modified content back to the file
            await p.write_text(new_content, errors="replace")

            if fmt_error:
                return ToolError(
                    message=f"{'[out of work-dir] ' if _outside else ''}File successfully edited, but {fmt_error}",
                    brief="Format validation failed",
                )

            # Note: the diff is intentionally NOT attached to the result display.
            # It was already shown during approval, and the streamed old/new
            # argument values are printed live (formatted and colored) by the
            # CLI printer while the tool call is generated (see kimix.base).
            # Attaching diff_blocks here would print the old -> new content twice.
            return ToolReturnValue(
                is_error=False,
                output="",
                message="".join(result_msg_parts),
                display=[],
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("edit failed: {path}: {error}", path=params.file_path, error=e)
            _outside_ex = False
            with contextlib.suppress(Exception):
                _outside_ex = not is_within_directory(
                    kaos_path_from_tool_input(params.file_path, self._work_dir).canonical(),
                    self._work_dir,
                )
            return ToolError(
                message=f"{'[out of work-dir] ' if _outside_ex else ''}Failed to edit. Error: {e} Path: {display_path}",
                brief="Failed to edit file",
            )
        except MemoryError:
            raise
