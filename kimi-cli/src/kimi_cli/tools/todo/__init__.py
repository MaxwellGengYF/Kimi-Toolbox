"""Todo list tracking tool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, override

import orjson
import rapidfuzz
from kosong.tooling import CallableTool2, ToolError, ToolReturnValue, alias_note
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kimi_cli import logger
from kimi_cli.session_state import TodoItemState, TodoStatus
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.display import TodoDisplayBlock, TodoDisplayItem
from kimi_cli.tools.utils import repair_json_string

_TODOLIST_DESCRIPTION = (
    "Read or write the whole todo tree. Omit `todos` to read the current tree; "
    "send the complete list to set the plan. For targeted single/batch edits "
    "(status, notes, rename, or children via parent=...) use todo_update.\n\n"
    "Write modes:\n"
    "- append (default): merges root-level todos by exact title; new titles are appended.\n"
    "- replace: replaces the whole list; only allowed when all existing todos are done (use force=True to override).\n"
    "- clear: empties the list; only allowed when all todos are done (use force=True to override).\n\n"
    "Notes:\n"
    "- Send the complete list each write; there are no partial edits.\n"
    "- Keep exactly one item in_progress at a time; auto_fix=True resolves conflicts by keeping the last listed item.\n"
    "- Statuses: pending, in_progress, done (or completed)."
)

def _truncate_prompt(text: str, max_len: int = 200) -> str:
    """Truncate long text, keeping head and tail.

    When ``len(text) > max_len``, keeps the first 100 chars and the last
    100 chars with ``...`` in between. Otherwise returns the text as-is.
    """
    if len(text) > max_len:
        return text[:100] + "..." + text[-100:]
    return text

_ALL_DONE_REMINDER = (
    "All todos are done. "
    "Please review the requirements again to ensure nothing is left unfinished."
)
"""Default reminder shown when all todos are done."""

# Hard limits for harness safety.
_MAX_TODOS = 4096
# Maximum number of archived todos kept in state (oldest are dropped first).
_MAX_ARCHIVED_TODOS = 500
# Maximum number of items printed by read mode before truncating.
_MAX_READ_ITEMS = 100

# ── Cross-tool reference hints (anti-hallucination core) ────────────────────
# Compact one-line hints appended to tool output. Success outputs carry a
# ``Next: Todo<X> ...`` hint; error outputs carry a corrective ``Hint: ...``
# sentence naming the right tool(s). Kept generic (≤1 sentence) on purpose.

def _hint_next(text: str) -> str:
    """Render a one-line 'Next:' hint block appended to success output."""
    return "\nNext: " + text

def _hint_error(text: str) -> str:
    """Render a one-line corrective hint block appended to error output."""
    return "\nHint: " + text

_TODOLIST_SUCCESS_HINT = (
    "todo_update to edit one or more items, or todo_write to read the tree."
)

# Mode map — only canonical values accepted
_MODE_MAP: dict[str, Literal["append", "replace", "clear"]] = {
    "append": "append",
    "replace": "replace",
    # Legacy spelling: 'overwrite' is the old name for replace-with-all-done-guard.
    "overwrite": "replace",
    "clear": "clear",
}

# Status map — only canonical values accepted; `completed` (report spelling)
# is accepted as an alias of the internal `done` value.
_STATUS_MAP: dict[str, TodoStatus] = {
    "pending": "pending",
    "in_progress": "in_progress",
    "done": "done",
    "completed": "done",
}

def _canonical_status(v: Any) -> TodoStatus:
    """Normalize a status value to its canonical form."""
    if not isinstance(v, str):
        raise ValueError(
            f"Invalid status '{v}'. Must be one of: pending, in_progress, done (or completed)."
        )
    normalized = v.strip().lower().replace("-", "_")
    canonical = _STATUS_MAP.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Invalid status '{v}'. Must be one of: pending, in_progress, done (or completed)."
        )
    return canonical

@dataclass(frozen=True)
class _FuzzyResult:
    """Typed wrapper for rapidfuzz>=3 ``(choice, score, index)`` match tuples."""

    choice: str
    score: float
    index: int

class Todo(BaseModel):
    model_config = {"populate_by_name": True}

    content: str = Field(
        validation_alias=AliasChoices("content", "title"),
        description="Title (report item shape: `content`).",
        min_length=1,
        max_length=65536,
    )
    status: TodoStatus = Field(description="Status")
    notes: str | None = Field(
        default=None,
        description="Notes. MUST write, be comprehensively, detailed.",
        max_length=65536,
    )
    # Sub todos (children). Leave empty for a leaf. Pydantic recurses
    # automatically; all field validators apply to children too.
    children: list[Todo] = Field(
        default_factory=list,
        description="Sub todos (children). Leave empty for a leaf.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_description_alias(cls, data: Any) -> Any:
        """Accept `description` as an alias for `notes`."""
        if isinstance(data, dict):
            if "description" in data and "notes" not in data:
                data["notes"] = data.pop("description")
        return data

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: Any) -> str:
        return _canonical_status(v)

    @field_validator("notes", mode="before")
    @classmethod
    def _validate_notes(cls, v: Any) -> str | None:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator("content")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Title cannot be empty or contain only whitespace")
        return stripped

class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    todos: list[Todo] | Todo | None = Field(
        default=None,
        validation_alias=AliasChoices("todos", "items"),
        description=(
            "The COMPLETE task list, replacing any previous list. Each item: "
            "`content` (string, short imperative line) and `status` (enum: "
            "pending/in_progress/completed). "
            "Passing an empty list [] is a no-op (use mode='clear' to empty the list). "
            + alias_note("todos", "items", word=False)
        ),
    )
    mode: Literal["append", "replace", "clear"] = Field(
        default="append",
        description=(
            "Write mode: 'append' merges the provided todos into the existing list "
            "(existing root titles are updated, new titles are appended; empty list is a no-op); "
            "'replace' replaces the existing todo list only when every existing todo is done "
            "(errors otherwise); 'clear' empties the list (errors unless every old todo is done). "
            "Set force=True to replace or clear even with unfinished todos."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "When True, mode='replace' and mode='clear' bypass the all-done guard "
            "(and skip regression and single-in_progress checks). "
            "Legacy 'force_overwrite' mode maps to mode='replace' with force=True."
        ),
    )
    auto_fix: bool = Field(
        default=True,
        description=(
            "When True and multiple items are in_progress, automatically mark the extra "
            "items as done before applying the update. The LAST in_progress item in the "
            "list (depth-first order) is treated as the current focus and kept; earlier "
            "in_progress items are completed. Set False to get an error instead."
        ),
    )
    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError(
                "Invalid mode. Must be 'append', 'replace', or 'clear'."
            )
        normalized = v.strip().lower().replace("-", "_")
        canonical = _MODE_MAP.get(normalized)
        if canonical is None:
            raise ValueError(
                f"Invalid mode '{v}'. Must be 'append', 'replace', or 'clear'."
            )
        return canonical

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_force_modes(cls, data: Any) -> Any:
        """Translate legacy 'force_overwrite' mode spellings into replace+force.

        Runs before field validation so the canonical ``mode``/``force`` values
        reach ``_validate_mode`` and the write branches.
        """
        if isinstance(data, dict):
            mode = data.get("mode")
            if isinstance(mode, str):
                norm = mode.strip().lower().replace("-", "_").replace(" ", "_")
                if norm in (
                    "force_overwrite",
                    "force_override",
                    "force",
                    "forcewrite",
                    "forceoverride",
                ):
                    data["mode"] = "replace"
                    data["force"] = True
        return data

    @field_validator("todos", mode="before")
    @classmethod
    def _validate_todos(cls, v: Any) -> list[Todo] | Todo | None:
        if v is None:
            return None
        if isinstance(v, Todo):
            return v
        if isinstance(v, str):
            parsed = repair_json_string(v)
            if parsed is None:
                raise ValueError(
                    "todos must be a list of todos, a single todo dict/object, or None"
                )
            v = parsed
        if isinstance(v, dict):
            try:
                return Todo.model_validate(v)
            except ValidationError as exc:
                msg = _first_pydantic_message(exc)
                raise ValueError(f"Invalid todo: {msg}") from exc
        if isinstance(v, list):
            out: list[Todo] = []
            for idx, item in enumerate(v):
                if isinstance(item, Todo):
                    out.append(item)
                    continue
                if isinstance(item, dict):
                    try:
                        out.append(Todo.model_validate(item))
                    except ValidationError as exc:
                        msg = _first_pydantic_message(exc)
                        raise ValueError(f"Invalid todo at index {idx}: {msg}") from exc
                    continue
                raise ValueError(
                    f"Invalid todo at index {idx}: expected a dict or Todo, got {type(item).__name__}"
                )
            return out
        raise ValueError("todos must be a list of todos, a single todo dict/object, or None")

def _first_pydantic_message(exc: ValidationError) -> str:
    """Return the first human-readable message from a Pydantic ValidationError."""
    errors = exc.errors()
    if errors:
        return errors[0].get("msg", str(exc))
    return str(exc)

@dataclass
class MergeResult:
    """Result of merging old and new todo lists.

    Attributes:
        todos: Merged todo list on success (``None`` means error).
        error: Error value set when the merge cannot proceed.
        warnings: Non-blocking warnings accumulated during the merge.
    """

    todos: list[Todo] | None = None
    error: ToolReturnValue | None = None
    warnings: list[str] = field(default_factory=list)

class TodoList(CallableTool2[Params]):
    name: str = "todo_write"
    description: str = _TODOLIST_DESCRIPTION
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if params.todos is not None:
            return await self._write_todos(params.todos, params)
        return self._read_todos()

    # ---- Write mode --------------------------------------------------------

    @staticmethod
    def _enforce_single_in_progress(todos: list[Todo]) -> list[str] | None:
        """Return list of titles that are in_progress if >1, else None.

        Recursive: the constraint is global across the whole tree (a child
        ``in_progress`` counts just like a root one).
        """
        in_progress: list[str] = []

        def walk(items: list[Todo]) -> None:
            for t in items:
                if t.status == "in_progress":
                    in_progress.append(t.content)
                walk(t.children)

        walk(todos)
        if len(in_progress) > 1:
            return in_progress
        return None

    @staticmethod
    def _auto_fix_in_progress(todos: list[Todo], warnings: list[str]) -> list[Todo]:
        """Enforce single in_progress, keeping the LAST item in DFS pre-order.

        Collects every in_progress node as a ``(container, index)`` pair in the
        same depth-first pre-order as ``_enforce_single_in_progress``, keeps the
        final one (the most recently listed item — the current focus), and
        demotes every earlier one to ``done``. Walks the whole tree so nested
        children participate too; ``warnings`` receives one line per demotion.

        Mutates only the list slots (element replacement), never the Todo
        objects themselves, so aliasing with persisted old items is safe.
        """
        slots: list[tuple[list[Todo], int]] = []

        def walk(items: list[Todo]) -> None:
            for i, item in enumerate(items):
                if item.status == "in_progress":
                    slots.append((items, i))
                walk(item.children)

        walk(todos)
        if len(slots) <= 1:
            return todos
        keep_container, keep_idx = slots[-1]
        kept_title = keep_container[keep_idx].content
        for container, idx in slots[:-1]:
            item = container[idx]
            warnings.append(
                f'Auto-fixed "{item.content}": set to done (only one item may be '
                f'in_progress; keeping "{kept_title}" in_progress)'
            )
            container[idx] = item.model_copy(update={"status": "done"})
        return todos

    async def _write_todos(
        self,
        raw_todos: list[Todo] | Todo,
        params: Params,
    ) -> ToolReturnValue:
        """Validate, merge, and persist todos, saving exactly once on success."""
        new_todos: list[Todo] = [raw_todos] if isinstance(raw_todos, Todo) else list(raw_todos)

        # 0. mode='clear' is a write of nothing — combining it with todos is a mistake.
        if params.mode == "clear" and new_todos:
            return self._error(
                "Error: mode='clear' cannot be combined with todos. "
                "Use mode='append' or 'replace' to write todos, or call with no todos to read.",
                "mode='clear' cannot be combined with todos.",
            )

        # 1. Validate new inputs
        if params.mode != "clear":
            duplicates = self._find_duplicate_titles(new_todos)
            if duplicates:
                return self._error(
                    f"Error: Duplicate todo titles found: {duplicates}",
                    f"Duplicate todo titles found: {duplicates}",
                    hint='todo_update(parent=...) to target a specific duplicate, or todo_write to read the tree.',
                )

            if self._count_all(new_todos) > _MAX_TODOS:
                return self._error(
                    f"Error: Todo list exceeds maximum limit of {_MAX_TODOS} items.",
                    f"Todo list exceeds maximum limit of {_MAX_TODOS} items.",
                )

        # 2. Load existing state
        old_todos = self._load_todos()
        old_archived = self._load_archived_todos()

        # 3. Branch on write mode. ``replaces_list`` marks modes that drop old
        # items (replace/clear) so completed ones get archived.
        warnings: list[str] = []
        replaces_list = False
        if params.mode == "clear":
            if old_todos and not all(t.status == "done" for t in old_todos):
                if not params.force:
                    unfinished = "\n".join(t.content for t in old_todos if t.status != "done")
                    return self._error(
                        "Error: Cannot clear todos while old todos are not all done. "
                        "Next step: mark them done first, "
                        "or call with mode='clear' and force=True to discard them intentionally.\n"
                        f"Unfinished:\n{unfinished}",
                        "Cannot clear todos while old todos are not all done.",
                        display=[self._build_display_block(old_todos)],
                    )
            final_todos = []
            replaces_list = True
        elif params.mode == "replace":
            if old_todos and not all(t.status == "done" for t in old_todos):
                if not params.force:
                    unfinished = "\n".join(t.content for t in old_todos if t.status != "done")
                    return self._error(
                        "Error: Cannot replace todos while old todos are not all done. "
                        "Use force=True if you really want to discard unfinished work.\n"
                        f"Unfinished:\n{unfinished}",
                        "Cannot replace todos while old todos are not all done.",
                    )
            final_todos = list(new_todos)
            replaces_list = True
        else:  # append
            if not new_todos:
                # Explicitly empty list is a no-op — use mode='clear' to empty.
                return self._build_noop_response(old_todos)
            result = self._merge_todos(old_todos, new_todos)
            if result.error is not None:
                return result.error
            final_todos = result.todos or []
            warnings.extend(result.warnings)

        # 3b. Enforce maximum tree nesting depth (all modes) so created trees
        # stay reachable via todo_write/todo_update(parent=...): the deepest
        # explicit-parent level is max_layers, plus one todo_update level under it.
        max_layers = self._max_layers()
        max_depth = max_layers + 1
        if self._max_tree_depth(final_todos) > max_depth:
            return self._error(
                f"Error: Todo tree exceeds maximum nesting depth of {max_depth} levels "
                f"(todo_max_layers={max_layers}). Flatten the tree, or build it with todo_write/todo_update(parent=...).",
                f"Todo tree exceeds maximum nesting depth of {max_depth} levels.",
                display=[self._build_display_block(final_todos)],
            )

        # 4. Regression detection
        if not params.force and params.mode != "clear" and old_todos:
            final_todos, regressions = self._check_regressions(old_todos, final_todos)
            if regressions:
                return self._error(
                    "Error: Cannot regress completed todos back to pending/in_progress: "
                    + ", ".join(regressions)
                    + "\nNext step: resend with these items kept as 'done', "
                    "or use force=True to restart them intentionally.",
                    "Cannot regress completed todos.",
                    display=[self._build_display_block(final_todos)],
                )

        # 5. Archive completed todos dropped by replace/clear
        archived = list(old_archived)
        if replaces_list and old_todos:
            kept_titles = {t.content for t in final_todos}
            newly_archived = [
                t for t in old_todos if t.status == "done" and t.content not in kept_titles
            ]
            if newly_archived:
                archived.extend(self._item_states(newly_archived))
                archived = archived[-_MAX_ARCHIVED_TODOS:]

        # 5b. Enforce single in_progress (unless auto_fix or force)
        if not params.force and params.mode != "clear":
            conflicts = self._enforce_single_in_progress(final_todos)
            if conflicts:
                if params.auto_fix:
                    # Auto-fix: keep the LAST in_progress node (depth-first
                    # pre-order — the most recently listed item, i.e. the
                    # current focus) and demote every earlier one to done.
                    # Recursive so children count too, matching the global
                    # single-in_progress invariant in _enforce_single_in_progress.
                    final_todos = self._auto_fix_in_progress(final_todos, warnings)
                else:
                    return self._error(
                        f"Error: Multiple items are in_progress: {conflicts}. "
                        "Keep exactly one item in_progress at a time. "
                        "Mark the current item as 'done' before starting another, "
                        "use force=True to override, "
                        "or set auto_fix=True to automatically resolve conflicts.",
                        "Multiple items in_progress",
                        display=[self._build_display_block(final_todos)],
                    )

        # 6. Persist exactly once
        save_error = self._save_todos(final_todos, archived)
        if save_error:
            return self._error(save_error, "Failed to save todos.")

        # 7. Build response
        result = self._build_success_response(
            final_todos, params.mode, bool(old_todos), warnings, params.force
        )
        return result

    @staticmethod
    def _error(
        output: str,
        message: str,
        display: list[Any] | None = None,
        hint: str | None = None,
    ) -> ToolReturnValue:
        """Build an error ToolReturnValue with consistent shape.

        ``hint`` (default: generic) names the sibling tools to use next so the
        model's tool inventory stays consistent after a failure.
        """
        if hint is None:
            hint = "todo_write to read the tree, or todo_update to edit one or more items."
        return ToolReturnValue(
            is_error=True,
            output=output + _hint_error(hint),
            message=message,
            display=display if display is not None else [],
        )

    @staticmethod
    def _find_duplicate_titles(todos: list[Todo]) -> list[str] | None:
        """Return a sorted list of all duplicate titles, or None if all unique."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for t in todos:
            if t.content in seen:
                duplicates.add(t.content)
            else:
                seen.add(t.content)
        return sorted(duplicates) if duplicates else None

    @staticmethod
    def _format_todos(
        todos: list[Todo],
        *,
        status_filter: tuple[TodoStatus, ...] = (
            "pending",
            "in_progress",
        ),
        display_status: dict[TodoStatus, str] | None = None,
    ) -> str:
        """Return a dense Markdown summary of selected todos, or '' if none."""
        if display_status is None:
            display_status = {
                "pending": "pending",
                "in_progress": "in progress",
                "done": "done",
            }
        selected = [t for t in todos if t.status in status_filter]
        if not selected:
            return ""
        lines: list[str] = []
        for t in selected:
            todo = f"- [{display_status[t.status]}] {t.content}"
            if t.status == "in_progress" and t.notes:
                todo += f"  Notes: {t.notes}"
            lines.append(todo)

        return "\n".join(lines)

    # Score threshold for user-facing title suggestions. rapidfuzz returns a
    # normalized similarity in [0, 100]; 60 catches minor typos while avoiding
    # suggestions that share only a few characters.
    _FUZZY_TITLE_CUTOFF: float = 60.0

    # Warning threshold for append-mode titles that are fuzzy near-matches of
    # existing titles. Kept moderate; the warning is now non-blocking, so it
    # should flag likely typos without rejecting legitimate new todos that share
    # common words.
    _FUZZY_WARNING_CUTOFF: float = 75.0

    @staticmethod
    def _find_nearest_titles(
        query_titles: list[str],
        candidate_titles: list[str],
        top_k: int = 1,
        *,
        score_cutoff: float | None = None,
        processor: Callable[[str], str] | None = None,
        scorer: Callable[..., float] | None = None,
    ) -> dict[str, list[_FuzzyResult]]:
        """Return nearest candidate titles for each query title.

        Uses a lightweight string similarity matcher (rapidfuzz) instead of
        rebuilding a full inverted index on every call. Returns a mapping
        ``query_title -> [_FuzzyResult(...), ...]``. If no candidate titles
        exist or no match clears the cutoff, the list is empty.

        Args:
            query_titles: Titles to look up.
            candidate_titles: Titles to search against.
            top_k: Maximum number of nearest matches to return per query.
            score_cutoff: Minimum rapidfuzz score to include. Defaults to
                ``_FUZZY_TITLE_CUTOFF`` for backward compatibility.
            processor: Optional preprocessing function applied to both query and
                candidate strings before scoring. The returned candidate title is
                the original (unprocessed) value.
            scorer: rapidfuzz scorer to use. Defaults to ``token_sort_ratio``.
        """
        if not candidate_titles or not query_titles:
            return {q: [] for q in query_titles}

        cutoff = score_cutoff if score_cutoff is not None else TodoList._FUZZY_TITLE_CUTOFF
        scorer = scorer if scorer is not None else rapidfuzz.fuzz.token_sort_ratio

        results: dict[str, list[_FuzzyResult]] = {}
        for query in query_titles:
            matches = rapidfuzz.process.extract(
                query,
                candidate_titles,
                scorer=scorer,
                limit=top_k,
                score_cutoff=cutoff,
                processor=processor,
            )
            # rapidfuzz>=3 process.extract returns (choice, score, index) tuples.
            results[query] = [
                _FuzzyResult(choice=str(choice), score=float(score), index=int(index))
                for choice, score, index in matches
            ]
        return results

    def _merge_todos(
        self,
        old_todos: list[Todo],
        new_todos: list[Todo],
    ) -> MergeResult:
        """Merge ``new_todos`` into ``old_todos`` using append/update semantics.

        * Existing root titles update status (and any provided metadata) in place.
        * Brand-new titles are appended to the end.
        * Fuzzy near-matches and titles that already exist deeper in the tree
          (scope duplicates) are reported as non-blocking warnings.
        """
        if not old_todos:
            return MergeResult(todos=list(new_todos))

        old_title_list = [t.content for t in old_todos]
        old_title_set = set(old_title_list)

        warnings = self._detect_fuzzy_warnings(new_todos, old_title_set, old_title_list)
        warnings.extend(self._detect_scope_duplicates(new_todos, old_todos))

        merged = self._merge_by_title_update(old_todos, new_todos)
        return MergeResult(todos=merged, warnings=warnings)

    @staticmethod
    def _detect_scope_duplicates(
        new_todos: list[Todo], old_todos: list[Todo]
    ) -> list[str]:
        """Warn when a new root title already exists deeper in the existing tree.

        todo_write append merges root-level titles only, so a title that exists
        only inside a stack scope (a child/grandchild) would be appended as a
        brand-new root item instead of updating the nested one. The warning is
        non-blocking because identical titles in different scopes are otherwise
        legal; it names the nested parent so the caller can switch to todo_update(parent=...).
        """
        if not old_todos or not new_todos:
            return []
        root_titles = {t.content for t in old_todos}
        nested: dict[str, str] = {}

        def walk(items: list[Todo], path: list[str]) -> None:
            for t in items:
                if path:
                    nested.setdefault(t.content, " > ".join(path))
                walk(t.children, [*path, t.content])

        walk(old_todos, [])
        warnings: list[str] = []
        for t in new_todos:
            if t.content in root_titles:
                continue
            parent = nested.get(t.content)
            if parent:
                warnings.append(
                    f'"{t.content}" already exists in the tree (under "{parent}"); '
                    "todo_write merges root-level titles only — use todo_update(parent=\"{parent}\", title=\"{t.content}\") to update it."
                )
        return warnings

    @staticmethod
    def _max_tree_depth(todos: list[Todo]) -> int:
        """Return the maximum nesting depth of the tree (root items = depth 1)."""
        max_depth = 0

        def walk(items: list[Todo], depth: int) -> None:
            nonlocal max_depth
            for t in items:
                if depth > max_depth:
                    max_depth = depth
                walk(t.children, depth + 1)

        walk(todos, 1)
        return max_depth

    def _build_noop_response(self, todos: list[Todo]) -> ToolReturnValue:
        """Response for an append-mode write with an empty todos list (no-op)."""
        counts = self._status_counts(todos)
        total = self._count_all(todos)
        stats = (
            f"({total} total: {counts['done']} done, "
            f"{counts['in_progress']} in progress, {counts['pending']} pending)"
        )
        output = f"Todo list unchanged; no todos provided {stats}"
        active_summary = self._format_todos(todos)
        if active_summary:
            output += "\n" + active_summary
        if total > 0:
            output += _hint_next(_TODOLIST_SUCCESS_HINT)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message="No todos provided; todo list unchanged.",
            display=[self._build_display_block(todos)] if todos else [],
        )

    def _detect_fuzzy_warnings(
        self,
        new_todos: list[Todo],
        old_title_set: set[str],
        old_title_list: list[str],
    ) -> list[str]:
        """Return non-blocking warnings for new titles that look like existing ones."""
        if not old_title_list:
            return []
        warnings: list[str] = []
        for new_todo in new_todos:
            if new_todo.content in old_title_set:
                continue
            nearest = self._find_nearest_titles(
                [new_todo.content],
                old_title_list,
                top_k=1,
                score_cutoff=TodoList._FUZZY_WARNING_CUTOFF,
                processor=str.lower,
            )
            hits = nearest.get(new_todo.content, [])
            if hits:
                warnings.append(f'"{new_todo.content}" looks like existing "{hits[0].choice}"')
        return warnings

    def _merge_by_title_update(self, old_todos: list[Todo], new_todos: list[Todo]) -> list[Todo]:
        """Update existing titles and append brand-new ones."""
        new_by_title = {t.content: t for t in new_todos}
        merged: list[Todo] = []
        seen: set[str] = set()

        for old in old_todos:
            new = new_by_title.get(old.content)
            if new is not None:
                merged.append(self._merge_one(old, new))
            else:
                merged.append(old)
            seen.add(old.content)

        for new in new_todos:
            if new.content not in seen:
                merged.append(new)
                seen.add(new.content)

        return merged

    @staticmethod
    def _merge_one(old: Todo, new: Todo) -> Todo:
        """Produce an updated todo preserving old notes when new omits them.

        For a same-title update, ``notes`` is replaced only when the new value
        is neither ``None`` nor empty/whitespace-only; otherwise the previously
        stored value is kept. Children of an updated parent are kept unless the
        new todo explicitly provided its own ``children`` (tracked via
        pydantic's ``model_fields_set``).
        """
        return Todo(
            content=old.content,
            status=new.status,
            notes=(
                new.notes
                if new.notes is not None and new.notes.strip()
                else old.notes
            ),
            children=(
                new.children
                if "children" in new.model_fields_set
                else old.children
            ),
        )

    @staticmethod
    def _check_regressions(
        old_todos: list[Todo], final_todos: list[Todo]
    ) -> tuple[list[Todo], list[str]]:
        """Detect done todos being moved back to pending/in_progress.

        Recursive: the old-status map is built across the whole tree and the
        clamp (done → pending/in_progress reverted to ``done``) applies to
        every descendant. Returns the final list with regressed items clamped
        back to ``done``, plus the list of regressed titles.
        """
        old_status_map: dict[str, str] = {}

        def collect(items: list[Todo]) -> None:
            for t in items:
                old_status_map[t.content] = t.status
                collect(t.children)

        collect(old_todos)

        regressions: list[str] = []

        def clamp(items: list[Todo]) -> list[Todo]:
            out: list[Todo] = []
            for t in items:
                if old_status_map.get(t.content) == "done" and t.status != "done":
                    regressions.append(t.content)
                    t = t.model_copy(update={"status": "done"})
                if t.children:
                    t = t.model_copy(update={"children": clamp(t.children)})
                out.append(t)
            return out

        return clamp(final_todos), regressions

    def _build_success_response(
        self,
        todos: list[Todo],
        mode: str,
        had_old_todos: bool,
        warnings: list[str],
        force: bool = False,
    ) -> ToolReturnValue:
        display_block = self._build_display_block(todos)
        active_summary = self._format_todos(todos)
        counts = self._status_counts(todos)

        mode_msg = {
            "append": "appended",
            "replace": "replaced",
            "clear": "cleared",
        }[mode]

        stats = (
            f"({self._count_all(todos)} total: {counts['done']} done, "
            f"{counts['in_progress']} in progress, {counts['pending']} pending)"
        )
        output_lines: list[str] = [f"Todo list {mode_msg} {stats}"]
        if active_summary:
            output_lines.append(active_summary)
        output = "\n".join(output_lines)

        # Append all-done reminder when all todos are done.
        all_done_reminder = _ALL_DONE_REMINDER
        # Append the original user prompt as context when available.
        current_prompt = getattr(self._runtime, "current_prompt", None)
        if current_prompt:
            all_done_reminder += "\nOriginal prompt:\n\n" + _truncate_prompt(current_prompt)
        if counts["pending"] == 0 and counts["in_progress"] == 0 and len(todos) > 0:
            output_lines.append(all_done_reminder)
            output = "\n".join(output_lines)

        # One-line cross-tool hint (output ONLY — never message; suppressed for
        # the 0-total case so exact-output assertions on empty writes hold).
        if self._count_all(todos) > 0:
            output += _hint_next(_TODOLIST_SUCCESS_HINT)

        message_lines: list[str] = [f"Todo list {mode_msg}."]
        if counts["pending"] == 0 and counts["in_progress"] == 0 and len(todos) > 0:
            message_lines.append(all_done_reminder)
        if force and had_old_todos:
            message_lines.append(
                "Warning: force=True bypassed the all-done guard and replaced the existing todo list."
            )
        if counts["in_progress"] > 1:
            message_lines.append(
                f"Note: {counts['in_progress']} items are in_progress; "
                "prefer exactly one at a time."
            )
        if warnings:
            message_lines.extend(["", *warnings])
        message = "\n".join(message_lines)

        return ToolReturnValue(
            is_error=False,
            output=output,
            message=message,
            display=[display_block],
        )

    @staticmethod
    def _status_counts(todos: list[Todo]) -> dict[TodoStatus, int]:
        """Count todos by status across the whole tree (recursive)."""
        counts: dict[TodoStatus, int] = {"pending": 0, "in_progress": 0, "done": 0}

        def walk(items: list[Todo]) -> None:
            for t in items:
                counts[t.status] += 1
                walk(t.children)

        walk(todos)
        return counts

    @staticmethod
    def _count_all(todos: list[Todo]) -> int:
        """Recursive total item count across the whole tree."""
        total = 0
        pending: list[Todo] = list(todos)
        while pending:
            t = pending.pop()
            total += 1
            pending.extend(t.children)
        return total

    @staticmethod
    def _count_unfinished_descendants(todo: Todo) -> int:
        """Count unfinished descendants (children and deeper) of a todo."""
        total = 0

        def walk(items: list[Todo]) -> None:
            nonlocal total
            for t in items:
                if t.status != "done":
                    total += 1
                walk(t.children)

        walk(todo.children)
        return total

    @staticmethod
    def _mark_subtree_done(node: Todo) -> None:
        """Recursively mark a node and all its descendants done."""
        node.status = "done"
        for child in node.children:
            TodoList._mark_subtree_done(child)

    @staticmethod
    def _build_display_block(todos: list[Todo]) -> TodoDisplayBlock:
        """Build a flattened display block with per-item ``depth`` (root = 0).

        Depth-first so the frontend can indent children under their parent.
        """
        items: list[TodoDisplayItem] = []

        def walk(items_list: list[Todo], depth: int) -> None:
            for todo in items_list:
                items.append(
                    TodoDisplayItem(
                        title=todo.content,
                        status=todo.status,
                        notes=todo.notes,
                        depth=depth,
                    )
                )
                walk(todo.children, depth + 1)

        walk(todos, 0)
        return TodoDisplayBlock(items=items)

    # ---- Read mode ---------------------------------------------------------

    def _render_read_tree(
        self, todos: list[Todo], *, max_lines: int = _MAX_READ_ITEMS
    ) -> str:
        """Render the todo tree for read mode: all statuses, children indented.

        Depth-0 lines are byte-identical to the legacy flat renderer (so flat
        lists render unchanged); children are indented 2 spaces per depth.
        Stops after ``max_lines`` lines (depth-first).
        """
        display_status = {
            "pending": "pending",
            "in_progress": "in_progress",
            "done": "done",
        }
        lines: list[str] = []

        def walk(items: list[Todo], depth: int) -> None:
            for t in items:
                if len(lines) >= max_lines:
                    return
                line = f"{'  ' * depth}- [{display_status[t.status]}] {t.content}"
                if t.status == "in_progress" and t.notes:
                    line += f"  Notes: {t.notes}"
                lines.append(line)
                walk(t.children, depth + 1)

        walk(todos, 0)
        return "\n".join(lines)

    def _read_todos(self) -> ToolReturnValue:
        todos = self._load_todos()
        archived = self._load_archived_todos()

        if not todos:
            empty_lines = ["Todo list is empty."]
            if archived:
                empty_lines.append(f"Archived: {len(archived)} completed todo(s).")
            return ToolReturnValue(
                is_error=False,
                output="\n".join(empty_lines) + _hint_next(_TODOLIST_SUCCESS_HINT),
                message="Todo list is empty.",
                display=[],
            )

        # Recursive counts across the whole tree (identical for flat lists).
        counts = self._status_counts(todos)
        total = self._count_all(todos)
        all_done = total > 0 and counts["pending"] == 0 and counts["in_progress"] == 0

        # Render the tree (children indented 2 spaces per depth), truncating
        # the flattened line list to _MAX_READ_ITEMS.
        output_lines = ["Current todo list:"]
        formatted = self._render_read_tree(todos, max_lines=_MAX_READ_ITEMS)
        if formatted:
            output_lines.append(formatted)

        if total > _MAX_READ_ITEMS:
            output_lines.append(
                f"... and {total - _MAX_READ_ITEMS} more "
                f"({counts['pending']} pending, {counts['in_progress']} in_progress, "
                f"{counts['done']} done total)"
            )
        if archived:
            output_lines.append(f"Archived: {len(archived)} completed todo(s).")

        # Append all-done reminder.
        all_done_reminder = _ALL_DONE_REMINDER
        # Append the original user prompt as context when available.
        current_prompt = getattr(self._runtime, "current_prompt", None)
        if current_prompt:
            all_done_reminder += "\nOriginal prompt:\n\n" + _truncate_prompt(current_prompt)
        if all_done:
            output_lines.append(all_done_reminder)

        # One-line cross-tool hint (output ONLY — never message).
        output_lines.append("Next: " + _TODOLIST_SUCCESS_HINT)

        return ToolReturnValue(
            is_error=False,
            output="\n".join(output_lines),
            message=all_done_reminder if all_done else "Current todo list displayed.",
            display=[],
        )

    # ---- Persistence -------------------------------------------------------

    def _save_todos(self, active: list[Todo], archived: list[TodoItemState]) -> str | None:
        """Persist active and archived todos. Returns error message on failure."""
        active_items = self._item_states(active)

        if self._runtime.role == "root":
            return self._save_root_todos(active_items, archived)
        return self._save_subagent_todos(active_items, archived)

    def _load_todos(self) -> list[Todo]:
        """Load active todos from the appropriate state file."""
        if self._runtime.role == "root":
            return self._load_root_todos()
        return self._load_subagent_todos()

    def _load_archived_todos(self) -> list[TodoItemState]:
        """Load archived todos from the appropriate state file."""
        if self._runtime.role == "root":
            return list(self._runtime.session.state.archived_todos)
        return self._load_subagent_archived_todos()

    def _max_layers(self) -> int:
        """Maximum todo tree depth (layers). Default 4.

        Used to cap how deep a tree can be built with todo_write or
        ``todo_update(parent=...)``: the deepest explicit-parent level is
        ``max_layers``, plus one todo_update level under it.
        """
        try:
            value = self._runtime.config.loop_control.todo_max_layers
        except Exception:
            return 4
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return 4

    def _save_root_todos(
        self, items: list[TodoItemState], archived: list[TodoItemState]
    ) -> str | None:
        try:
            session = self._runtime.session
            session.state.todos = items
            session.state.archived_todos = archived
            session.save_state()
            return None
        except Exception as exc:
            return f"Error: Failed to save root todos: {exc}"

    def _load_root_todos(self) -> list[Todo]:
        from kimi_cli.session_state import load_session_state

        session = self._runtime.session
        fresh = load_session_state(session.dir)
        session.state.todos = fresh.todos
        session.state.archived_todos = fresh.archived_todos
        result: list[Todo] = []
        for t in fresh.todos:
            try:
                result.append(Todo.model_validate(t.model_dump()))
            except Exception:
                logger.warning("Skipping malformed todo item in root state: {t}", t=t)
        return result

    def _save_subagent_todos(
        self, items: list[TodoItemState], archived: list[TodoItemState]
    ) -> str | None:
        state_file = self._subagent_state_file()
        if state_file is None:
            return "Error: Unable to save subagent todos: state file is not available."
        data = self._read_subagent_state(state_file)
        data["todos"] = [item.model_dump() for item in items]
        data["archived_todos"] = [item.model_dump() for item in archived]
        try:
            self._write_subagent_state(state_file, data)
        except Exception as exc:
            return f"Error: Failed to save subagent todos: {exc}"
        return None

    def _load_subagent_todos(self) -> list[Todo]:
        state_file = self._subagent_state_file()
        if state_file is None:
            return []
        data = self._read_subagent_state(state_file)
        raw_todos_val = data.get("todos", [])
        raw_todos = cast(list[Any], raw_todos_val) if isinstance(raw_todos_val, list) else []
        result: list[Todo] = []
        for item in raw_todos:
            try:
                result.append(Todo.model_validate(item))
            except Exception:
                logger.warning("Skipping malformed todo item in subagent state: {item}", item=item)
        return result

    def _load_subagent_archived_todos(self) -> list[TodoItemState]:
        state_file = self._subagent_state_file()
        if state_file is None:
            return []
        data = self._read_subagent_state(state_file)
        raw_archived_val = data.get("archived_todos", [])
        raw_archived = (
            cast(list[Any], raw_archived_val) if isinstance(raw_archived_val, list) else []
        )
        result: list[TodoItemState] = []
        for item in raw_archived:
            try:
                result.append(TodoItemState.model_validate(item))
            except Exception:
                logger.warning(
                    "Skipping malformed archived todo item in subagent state: {item}", item=item
                )
        return result

    @staticmethod
    def _item_states(todos: list[Todo]) -> list[TodoItemState]:
        return [
            TodoItemState(
                title=todo.content,
                status=todo.status,
                notes=todo.notes,
                children=TodoList._item_states(todo.children),
            )
            for todo in todos
        ]

    def _subagent_state_file(self) -> Path | None:
        store = self._runtime.subagent_store
        agent_id = self._runtime.subagent_id
        if store is None or agent_id is None:
            return None
        return store.instance_dir(agent_id) / "state.json"

    @staticmethod
    def _read_subagent_state(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = orjson.loads(path.read_text(encoding="utf-8"))
        except (orjson.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Corrupted subagent todo state, using defaults: {path}", path=path)
            return {}
        if not isinstance(data, dict):
            logger.warning("Invalid subagent todo state type, using defaults: {path}", path=path)
            return {}
        return cast(dict[str, Any], data)

    @staticmethod
    def _write_subagent_state(path: Path, data: dict[str, Any]) -> None:
        from kimi_cli.utils.io import atomic_json_write

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(data, path)



class TodoUpdateItem(BaseModel):
    """Single update operation for todo_update."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(
        description="Title of the todo to update or create. Exact match is tried first; fuzzy match is used when enabled and exact match fails.",
        min_length=1,
        max_length=65536,
    )
    status: TodoStatus | None = Field(
        default=None,
        description="New status. One of: pending, in_progress, done. Omit to keep the current status (new items default to pending).",
    )
    notes: str | None = Field(
        default=None,
        description="New notes. Omit to keep current notes; pass an empty string to clear notes.",
        max_length=65536,
    )
    rename_to: str | None = Field(
        default=None,
        description="Rename the matched todo to this title.",
    )
    parent: str | None = Field(
        default=None,
        description=(
            "Parent todo title that scopes the lookup and creation. "
            "When provided, the title is searched only under that parent. "
            "If the title does not exist there, a new child is created. "
            "Use an empty string for the root scope (creation allowed); omit to search globally and update only."
        ),
    )
    fuzzy: bool = Field(
        default=True,
        description="When True and the exact title is not found, use fuzzy matching to find the nearest title.",
    )
    force: bool = Field(
        default=False,
        description="Allow regressing a 'done' item back to pending/in_progress, or allow renaming that would collide with a done item.",
    )
    complete: bool = Field(
        default=False,
        description=(
            "When True, mark the matched todo and all of its sub-todos done "
            "(one-call subtree finish; replaces the old todo_pop). "
            "Cannot be combined with status='pending'/'in_progress'."
        ),
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: Any) -> str | None:
        if v is None:
            return None
        return _canonical_status(v)

    @field_validator("title", "parent")
    @classmethod
    def _validate_title(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()

    @field_validator("rename_to")
    @classmethod
    def _validate_rename_to(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class TodoUpdateParams(BaseModel):
    """Parameters for todo_update: one or more lightweight todo edits without rewriting the tree."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str | None = Field(
        default=None,
        description=(
            "Title of the todo to update or create when using a single top-level "
            "update. Use `updates` to batch multiple edits."
        ),
        min_length=1,
        max_length=65536,
    )
    status: TodoStatus | None = Field(
        default=None,
        description=(
            "New status for the single top-level update. Ignored when `updates` is provided."
        ),
    )
    notes: str | None = Field(
        default=None,
        description=(
            "New notes for the single top-level update. Ignored when `updates` is provided."
        ),
        max_length=65536,
    )
    rename_to: str | None = Field(
        default=None,
        description=(
            "Rename for the single top-level update. Ignored when `updates` is provided."
        ),
    )
    parent: str | None = Field(
        default=None,
        description=(
            "Optional common parent title applied to items in `updates` that do not "
            "specify their own parent. Also usable as a top-level parent for a single update."
        ),
    )
    fuzzy: bool = Field(
        default=True,
        description=(
            "Fuzzy matching setting for the single top-level update. "
            "Ignored when `updates` is provided."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Force setting for the single top-level update. "
            "Ignored when `updates` is provided."
        ),
    )
    complete: bool = Field(
        default=False,
        description=(
            "When True, mark the matched todo and all of its sub-todos done. "
            "Ignored when `updates` is provided."
        ),
    )
    updates: list[TodoUpdateItem] | TodoUpdateItem | None = Field(
        default=None,
        validation_alias=AliasChoices("updates", "todos"),
        description=(
            "One or more update operations. Each item has the same shape as a single "
            "todo_update call (title, status, notes, rename_to, parent, fuzzy, force, complete). "
            "Use this to batch multiple lightweight edits in one call. When provided, "
            "top-level title/status/notes/rename_to/complete must not be used."
        ),
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: Any) -> str | None:
        if v is None:
            return None
        return _canonical_status(v)

    @field_validator("title", "parent")
    @classmethod
    def _validate_title(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()

    @field_validator("rename_to")
    @classmethod
    def _validate_rename_to(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("updates", mode="before")
    @classmethod
    def _validate_updates(cls, v: Any) -> list[TodoUpdateItem] | TodoUpdateItem | None:
        if v is None:
            return None
        if isinstance(v, TodoUpdateItem):
            return [v]
        if isinstance(v, dict):
            return [TodoUpdateItem.model_validate(v)]
        if isinstance(v, list):
            out: list[TodoUpdateItem] = []
            for idx, item in enumerate(v):
                if isinstance(item, TodoUpdateItem):
                    out.append(item)
                    continue
                if isinstance(item, dict):
                    try:
                        out.append(TodoUpdateItem.model_validate(item))
                    except ValidationError as exc:
                        msg = _first_pydantic_message(exc)
                        raise ValueError(f"Invalid update at index {idx}: {msg}") from exc
                    continue
                raise ValueError(
                    f"Invalid update at index {idx}: expected a dict or TodoUpdateItem, "
                    f"got {type(item).__name__}"
                )
            return out
        raise ValueError(
            "updates must be a list of updates, a single update dict/object, or None"
        )

    @model_validator(mode="after")
    def _check_no_mixed_fields(self) -> TodoUpdateParams:
        if self.updates is not None:
            mixed = {"title", "status", "notes", "rename_to", "complete"} & self.model_fields_set
            if mixed:
                raise ValueError(
                    "Cannot mix top-level "
                    f"{sorted(mixed)} with `updates`; pass all edits inside `updates`."
                )
        return self


class todo_update(TodoList):
    """Lightweight edit or creation of one or more todos by title; no need to resend the whole tree."""

    name: str = "todo_update"
    description: str = (
        "Create, update, rename, or complete one or more todos by title — no need to "
        "resend the whole tree. Pass a single edit directly (title=..., status=...), or "
        "pass updates=[...] (alias todos=[...]) to batch several edits in one call.\n"
        "- title: the todo to update or create.\n"
        "- parent: scope the lookup/creation — omit to search the whole tree (update only), "
        "  \"\" for the root scope, or a parent title to create/update a child under it.\n"
        "- status: pending/in_progress/done; omit keeps the current status (new items default to pending).\n"
        "- notes: replace notes (\"\" clears, omit keeps).\n"
        "- rename_to: rename the matched todo.\n"
        "- complete: True marks the matched todo and all its sub-todos done (one call finishes a subtree).\n"
        "- force: allow reopening a done item or renaming over a done item.\n"
        "- fuzzy: default True — match near-miss titles when the exact title is not found."
    )
    params: type[TodoUpdateParams] = TodoUpdateParams

    def __init__(self, runtime: Runtime) -> None:
        # Subclass todo_write for shared persistence/helpers, but use our own metadata.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoUpdateParams) -> ToolReturnValue:
        items, error = self._normalize_update_items(params)
        if error is not None:
            return error

        todos = self._load_todos()
        warnings: list[str] = []
        summaries: list[str] = []

        messages: list[str] = []
        for item in items:
            result = self._apply_one_update(todos, item, warnings)
            if isinstance(result, ToolReturnValue):
                return result
            todos, summary, message, _ = result
            summaries.append(summary)
            messages.append(message)

            conflicts = self._enforce_single_in_progress(todos)
            if conflicts:
                todos = self._auto_fix_in_progress(todos, warnings)

        archived = self._load_archived_todos()
        save_error = self._save_todos(todos, archived)
        if save_error:
            hint = "Use todo_write to read the tree and retry todo_update."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )

        output_lines: list[str] = ["Current todo list:"]
        tree = self._render_read_tree(todos, max_lines=_MAX_READ_ITEMS)
        if tree:
            output_lines.append(tree)

        output = "\n".join(output_lines) + "\n" + "\n".join(summaries)
        next_hint = (
            "todo_update to edit another item, todo_update(parent=...) to add a child, "
            "or todo_write to read the tree."
        )
        output += _hint_next(next_hint)
        if warnings:
            output += "\n" + "\n".join(warnings)

        message = "; ".join(messages)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message=message,
            display=[self._build_display_block(todos)],
        )

    def _normalize_update_items(
        self, params: TodoUpdateParams
    ) -> tuple[list[TodoUpdateItem], ToolReturnValue | None]:
        """Normalize params into a list of ``TodoUpdateItem`` and an optional error."""
        if params.updates is not None:
            items = list(params.updates)
            common_parent = params.parent.strip() if params.parent is not None else None
            if common_parent is not None:
                applied: list[TodoUpdateItem] = []
                for item in items:
                    if item.parent is None:
                        applied.append(item.model_copy(update={"parent": common_parent}))
                    else:
                        applied.append(item)
                items = applied
            return items, None

        # Single top-level update mode.
        if params.title is None:
            hint = (
                "Provide a title for a single update, or pass updates=[...] for multiple updates."
            )
            return [], ToolError(
                message="No todo title provided.",
                brief=hint,
                output="Error: title is required when updates is not provided." + _hint_error(hint),
            )
        single = TodoUpdateItem(
            title=params.title,
            status=params.status,
            notes=params.notes,
            rename_to=params.rename_to,
            parent=params.parent,
            fuzzy=params.fuzzy,
            force=params.force,
            complete=params.complete,
        )
        return [single], None

    def _apply_one_update(
        self,
        todos: list[Todo],
        params: TodoUpdateItem,
        warnings: list[str],
    ) -> tuple[list[Todo], str, str, tuple[str, str] | None] | ToolReturnValue:
        """Apply a single update item to an in-memory todo tree.

        Returns ``(new_tree, summary, message, rename_pair)`` on success, where
        ``summary`` is the detailed human-readable line, ``message`` is the
        short confirmation, and ``rename_pair`` is ``(old_title, new_title)`` if
        a rename happened. On failure returns a ``ToolReturnValue`` error.
        """
        parent_raw = params.parent.strip() if params.parent is not None else None

        if parent_raw is None:
            if not todos:
                hint = (
                    'Use todo_update(parent="", title="...") to create a root todo, '
                    "or todo_write to set the whole list."
                )
                return ToolError(
                    message="No todos to update.",
                    brief=hint,
                    output="Error: No todos exist." + _hint_error(hint),
                )
            return self._update_global_in_memory(todos, params, warnings)

        return self._update_or_create_under_parent_in_memory(todos, parent_raw, params, warnings)

    def _update_global_in_memory(
        self,
        todos: list[Todo],
        params: TodoUpdateItem,
        warnings: list[str],
    ) -> tuple[list[Todo], str, str, tuple[str, str] | None] | ToolReturnValue:
        target_title = params.title
        path = self._find_path(todos, target_title)
        matched_title = target_title

        if path is None:
            if not params.fuzzy:
                hint = 'Use todo_write to read the tree, or set fuzzy=True to search by similarity.'
                return ToolError(
                    message=f'Todo "{target_title}" not found.',
                    brief=hint,
                    output=f'Error: No todo titled "{target_title}" found.' + _hint_error(hint),
                )
            nearest = self._find_nearest_titles(
                [target_title],
                self._collect_titles(todos),
                top_k=1,
                score_cutoff=TodoList._FUZZY_TITLE_CUTOFF,
                processor=str.lower,
            )
            hits = nearest.get(target_title, [])
            if not hits:
                hint = "Use todo_write to read the tree."
                return ToolError(
                    message=f'No todo matching "{target_title}" found.',
                    brief=hint,
                    output=f'Error: No todo matching "{target_title}" found.' + _hint_error(hint),
                )
            matched_title = hits[0].choice
            path = self._find_path(todos, matched_title)
            warnings.append(f'Fuzzy matched "{target_title}" to "{matched_title}".')

        assert path is not None
        return self._apply_update_to_tree(todos, path, matched_title, params, warnings)

    def _update_or_create_under_parent_in_memory(
        self,
        todos: list[Todo],
        parent_title: str,
        params: TodoUpdateItem,
        warnings: list[str],
    ) -> tuple[list[Todo], str, str, tuple[str, str] | None] | ToolReturnValue:
        if parent_title == "":
            parent_path: list[int] | None = []
            parent_node: Todo | None = None
            resolved_parent_title = "root"
        else:
            parent_path = self._find_path(todos, parent_title)
            if parent_path is None:
                if not params.fuzzy:
                    hint = 'Use todo_write to read the tree, or set fuzzy=True to search by similarity.'
                    return ToolError(
                        message=f'Parent todo "{parent_title}" not found.',
                        brief=hint,
                        output=f'Error: No parent todo titled "{parent_title}" found.' + _hint_error(hint),
                    )
                nearest = self._find_nearest_titles(
                    [parent_title],
                    self._collect_titles(todos),
                    top_k=1,
                    score_cutoff=TodoList._FUZZY_TITLE_CUTOFF,
                    processor=str.lower,
                )
                hits = nearest.get(parent_title, [])
                if not hits:
                    hint = "Use todo_write to read the tree."
                    return ToolError(
                        message=f'No parent todo matching "{parent_title}" found.',
                        brief=hint,
                        output=f'Error: No parent todo matching "{parent_title}" found.' + _hint_error(hint),
                    )
                resolved_parent_title = hits[0].choice
                parent_path = self._find_path(todos, resolved_parent_title)
                warnings.append(f'Fuzzy matched parent "{parent_title}" to "{resolved_parent_title}".')
                assert parent_path is not None
                parent_node = self._node_at_path(todos, parent_path)
            else:
                parent_node = self._node_at_path(todos, parent_path)
                resolved_parent_title = parent_node.content

        scope = parent_node.children if parent_node is not None else todos
        target_title = params.title
        child_index = next((i for i, t in enumerate(scope) if t.content == target_title), None)

        if child_index is not None:
            path = [*parent_path, child_index] if parent_path else [child_index]
            return self._apply_update_to_tree(todos, path, target_title, params, warnings)

        # New child creation under the resolved parent.
        if params.rename_to is not None:
            hint = f'Cannot rename a new child; use title="{params.rename_to}" to create it.'
            return ToolError(
                message=f'Cannot rename non-existent todo "{target_title}".',
                brief=hint,
                output=f'Error: "{target_title}" does not exist under "{resolved_parent_title}".' + _hint_error(hint),
            )

        if params.complete:
            hint = (
                f'complete=True requires an existing todo; "{target_title}" does not exist '
                f'under "{resolved_parent_title}". Create it first or use status="done".'
            )
            return ToolError(
                message=f'Cannot complete non-existent todo "{target_title}".',
                brief=hint,
                output=f'Error: "{target_title}" does not exist under "{resolved_parent_title}".' + _hint_error(hint),
            )

        parent_depth = len(parent_path)
        max_layers = self._max_layers()
        if parent_depth > max_layers:
            hint = f"Cannot add children deeper than {max_layers + 1} layers."
            return ToolError(
                message=f'Cannot add a child under "{resolved_parent_title}": too deep.',
                brief=hint,
                output=(
                    f'Error: "{resolved_parent_title}" is at depth {parent_depth}; '
                    f"children would exceed the maximum depth ({max_layers + 1})."
                    + _hint_error(hint)
                ),
            )

        new_child = Todo(
            content=target_title,
            status=params.status if params.status is not None else "pending",
            notes=params.notes,
        )
        final_todos = self._insert_child_at_path(todos, parent_path, new_child)

        summary = f'Created "{target_title}" under "{resolved_parent_title}".'
        return final_todos, summary, summary, None

    def _apply_update_to_tree(
        self,
        todos: list[Todo],
        path: list[int],
        matched_title: str,
        params: TodoUpdateItem,
        warnings: list[str],
    ) -> tuple[list[Todo], str, str, tuple[str, str] | None] | ToolReturnValue:
        old_node = self._node_at_path(todos, path)
        new_status = params.status if params.status is not None else old_node.status

        # complete=True marks the subtree done — it cannot be combined with a
        # pending/in_progress status (ambiguous intent).
        if params.complete and params.status in ("pending", "in_progress"):
            hint = f'Use todo_update "{matched_title}" with status="done" instead of complete=True, or omit status.'
            return ToolError(
                message=f'complete=True cannot be combined with status="{params.status}".',
                brief=hint,
                output=(
                    f'Error: complete=True cannot be combined with status="{params.status}" '
                    f'for "{matched_title}". complete=True always marks everything done.'
                    + _hint_error(hint)
                ),
            )

        # Regression guard: done -> pending/in_progress is blocked unless force=True.
        if not params.force and old_node.status == "done" and new_status != "done":
            hint = (
                f'Use todo_update "{matched_title}" with force=True to reopen a done item, '
                "or todo_write with mode='replace' and force=True to restart the whole list."
            )
            return ToolError(
                message=f'Cannot regress completed todo "{matched_title}" back to {new_status}.',
                brief=hint,
                output=(
                    f'Error: Cannot regress completed todo "{matched_title}" back to {new_status}.'
                    + _hint_error(hint)
                ),
            )

        # Rename collision guard within the matched scope.
        final_title = matched_title
        rename_pair: tuple[str, str] | None = None
        if params.rename_to is not None and params.rename_to != matched_title:
            new_title = params.rename_to
            parent_path = path[:-1]
            siblings = (
                self._node_at_path(todos, parent_path).children
                if parent_path
                else todos
            )
            if any(t.content == new_title for i, t in enumerate(siblings) if i != path[-1]):
                hint = f'Use todo_update "{new_title}" to update the existing item instead of renaming.'
                return ToolError(
                    message=f'Cannot rename "{matched_title}" to "{new_title}": title already exists.',
                    brief=hint,
                    output=(
                        f'Error: Cannot rename "{matched_title}" to "{new_title}": '
                        "title already exists in this scope."
                        + _hint_error(hint)
                    ),
                )
            final_title = new_title
            rename_pair = (matched_title, final_title)

        # Notes: None means keep existing; empty string clears notes.
        final_notes = old_node.notes if params.notes is None else (params.notes.strip() or None)

        final_todos = self._update_at_path(
            todos,
            path,
            {
                "content": final_title,
                "status": new_status,
                "notes": final_notes,
            },
        )

        change_parts: list[str] = []
        if params.complete:
            # Mark the matched node and every descendant done (in-place on the
            # fresh copies produced by _update_at_path — safe to mutate).
            node = self._node_at_path(final_todos, path)
            self._mark_subtree_done(node)
            n = self._count_all([node])
            change_parts.append(f"completed with {n} sub-todo{'s' if n != 1 else ''} marked done")
        if params.status is not None:
            change_parts.append(f"status={new_status}")
        if params.notes is not None:
            change_parts.append("notes updated")
        if params.rename_to is not None:
            change_parts.append(f'renamed to "{final_title}"')
        change_summary = ", ".join(change_parts) if change_parts else "no changes"
        summary = f'Updated "{matched_title}" ({change_summary}).'
        message = f'Updated "{matched_title}".'
        return final_todos, summary, message, rename_pair

    @staticmethod
    def _insert_child_at_path(
        items: list[Todo], parent_path: list[int] | None, child: Todo
    ) -> list[Todo]:
        """Return a new tree with ``child`` appended to the children of the node at ``parent_path``.

        ``parent_path`` of ``None`` or ``[]`` appends to the root list.
        """
        if not parent_path:
            return [*items, child]
        new_items = list(items)
        if len(parent_path) == 1:
            parent = items[parent_path[0]]
            new_items[parent_path[0]] = parent.model_copy(update={"children": [*parent.children, child]})
        else:
            child_updates = todo_update._insert_child_at_path(
                items[parent_path[0]].children, parent_path[1:], child
            )
            new_items[parent_path[0]] = items[parent_path[0]].model_copy(
                update={"children": child_updates}
            )
        return new_items

    @staticmethod
    def _collect_titles(items: list[Todo]) -> list[str]:
        """Return every title in the tree, depth-first."""
        titles: list[str] = []
        for t in items:
            titles.append(t.content)
            titles.extend(todo_update._collect_titles(t.children))
        return titles

    @staticmethod
    def _find_path(items: list[Todo], title: str, path: list[int] | None = None) -> list[int] | None:
        """Return the index path to the first todo with the given title, or None."""
        if path is None:
            path = []
        for i, t in enumerate(items):
            if t.content == title:
                return [*path, i]
            child_path = todo_update._find_path(t.children, title, [*path, i])
            if child_path is not None:
                return child_path
        return None

    @staticmethod
    def _node_at_path(items: list[Todo], path: list[int]) -> Todo:
        """Return the node at the given index path."""
        node = items[path[0]]
        for idx in path[1:]:
            node = node.children[idx]
        return node

    @staticmethod
    def _update_at_path(items: list[Todo], path: list[int], updates: dict[str, Any]) -> list[Todo]:
        """Return a new tree with the node at ``path`` updated via ``updates``."""
        new_items = list(items)
        if len(path) == 1:
            new_items[path[0]] = items[path[0]].model_copy(update=updates)
        else:
            child_updates = todo_update._update_at_path(
                items[path[0]].children, path[1:], updates
            )
            new_items[path[0]] = items[path[0]].model_copy(update={"children": child_updates})
        return new_items
