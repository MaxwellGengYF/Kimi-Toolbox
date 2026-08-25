"""Parameter models for the multi-mode edit tool."""

from __future__ import annotations

from typing import Any, Literal, Union

from kosong.tooling import ToolError, alias_note
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

EditMode = Literal["replace", "patch", "hashline", "sloppy", "apply_patch"]


class ReplaceEditItem(BaseModel):
    """A single literal replace edit."""

    model_config = {"populate_by_name": True}

    old: str = Field(
        alias="old_string",
        description="String to replace. " + alias_note("old", "old_string", word=False),
    )
    new: str = Field(
        alias="new_string",
        description="Replacement text. " + alias_note("new", "new_string", word=False),
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


class PatchEntry(BaseModel):
    """A single patch entry for patch mode."""

    model_config = {"populate_by_name": True}

    op: Literal["create", "update", "delete"] = Field(default="update")
    diff: str | None = Field(
        default=None,
        description="Unified-diff hunk text for update; full file content for create.",
    )
    rename: str | None = Field(
        default=None,
        description="Destination path for update+move (relative to workdir).",
    )


class EditParams(BaseModel):
    """Parameters for the multi-mode edit tool."""

    model_config = {"populate_by_name": True}

    mode: Literal["auto", "replace", "patch", "hashline", "sloppy", "apply_patch"] = Field(
        default="auto",
        description="Edit mode. 'auto' detects the mode from the payload shape.",
    )

    file_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("file_path", "path"),
        description="Path to edit. " + alias_note("file_path", "path", word=False),
    )

    edit: Union[
        ReplaceEditItem, PatchEntry, list[Union[ReplaceEditItem, PatchEntry]], None
    ] = Field(
        default=None,
        alias="edits",
        description="One or more replace edits, or patch entries when mode='patch'. "
        + alias_note("edit", "edits", word=False),
    )

    old_string: str | None = Field(
        default=None,
        description="Literal text to replace. Single-edit shorthand for `edit`.",
    )
    new_string: str | None = Field(
        default=None,
        description="Literal replacement text. Single-edit shorthand for `edit`.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all matches. Only used with the single-edit shorthand.",
    )

    input: str | None = Field(
        default=None,
        description="Input text for hashline / sloppy / apply_patch modes.",
    )

    sandbox_permissions: Literal["workspace-write", "danger-full-access"] | None = Field(
        default=None,
        description="The wider sandbox mode this file operation needs.",
    )
    justification: str | None = Field(
        default=None,
        description="Required with sandbox_permissions: explanation for the user.",
    )
    allow_conflicts: bool = Field(
        default=False,
        description="When True, allow editing files that contain conflict markers.",
    )

    # Internal: resolved mode populated by model_validator.
    resolved_mode: EditMode | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_single_edit(cls, data: Any) -> Any:
        """Build the `edit` list from top-level old_string/new_string shorthand."""
        if not isinstance(data, dict):
            return data
        if "edit" not in data and "edits" not in data:
            if data.get("old_string") is not None or data.get("new_string") is not None:
                data = {
                    **data,
                    "edit": [
                        {
                            "old_string": data.get("old_string", ""),
                            "new_string": data.get("new_string", ""),
                            "replace_all": data.get("replace_all", False),
                        }
                    ],
                }
        return data

    @field_validator("edit", mode="before")
    @classmethod
    def _normalize_edit(cls, v: Any) -> Any:
        """Auto-wrap a single edit dict into a list and route patch/replace models."""
        if isinstance(v, dict):
            return [cls._coerce_edit(v)]
        if isinstance(v, list):
            return [cls._coerce_edit(item) for item in v]
        if isinstance(v, ReplaceEditItem):
            return [v]
        if isinstance(v, PatchEntry):
            return [v]
        return v

    @classmethod
    def _coerce_edit(cls, item: Any) -> ReplaceEditItem | PatchEntry:
        if isinstance(item, (ReplaceEditItem, PatchEntry)):
            return item
        if isinstance(item, dict):
            op = item.get("op")
            if op in {"create", "update", "delete"}:
                return PatchEntry.model_validate(item)
            return ReplaceEditItem.model_validate(item)
        raise ValueError(f"Invalid edit item: {item!r}")

    @model_validator(mode="after")
    def _resolve_mode(self) -> "EditParams":
        """Resolve auto-detected mode and validate payload shape."""
        raw_mode = self.mode
        if raw_mode != "auto":
            normalized = normalize_edit_mode(raw_mode)
            if normalized is None:
                raise ValueError(f"Invalid edit mode: {raw_mode}")
            self.resolved_mode = normalized
        else:
            self.resolved_mode = detect_mode(self)
        return self


def normalize_edit_mode(raw: str) -> EditMode | None:
    """Normalize a user-provided mode string to a canonical EditMode."""
    if not raw:
        return None
    key = raw.lower().replace("-", "_").replace(" ", "_")
    mapping: dict[str, EditMode] = {
        "replace": "replace",
        "patch": "patch",
        "hashline": "hashline",
        "hash_line": "hashline",
        "sloppy": "sloppy",
        "apply_patch": "apply_patch",
        "applypatch": "apply_patch",
        "applypatchfile": "apply_patch",
    }
    return mapping.get(key)


def detect_mode(params: EditParams) -> EditMode:
    """Detect edit mode from the payload shape."""
    if params.input:
        first_non_blank = ""
        for line in params.input.splitlines():
            stripped = line.strip()
            if stripped:
                first_non_blank = stripped
                break
        if first_non_blank.startswith("*** Begin Patch") or first_non_blank.startswith("*** Begin"):
            return "apply_patch"
        if first_non_blank.startswith("["):
            return "hashline"
        if first_non_blank.startswith("§"):
            return "sloppy"
    edits = params.edit
    if isinstance(edits, list) and edits:
        all_patch_entries = True
        for item in edits:
            if isinstance(item, PatchEntry):
                continue
            if isinstance(item, dict):
                op = item.get("op")
                if op in {"create", "update", "delete"}:
                    continue
            all_patch_entries = False
            break
        if all_patch_entries:
            return "patch"
    if params.old_string is not None or params.new_string is not None:
        return "replace"
    if edits is not None:
        return "replace"
    raise ValueError(
        "Could not determine edit mode. Supported payloads: "
        "replace ({file_path, old_string, new_string}), "
        "patch ({file_path, edits: [{op, diff}]}), "
        "apply_patch ({input: '*** Begin Patch ... *** End Patch'}), "
        "hashline ({input: '[path#TAG] ...'}), "
        "sloppy ({input: '§path ...'})."
    )
