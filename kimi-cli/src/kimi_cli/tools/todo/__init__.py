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
    "Record and update a structured task list for the current work. Send the "
    "ENTIRE list every call — it REPLACES the previous list (there are no "
    "partial updates, no per-item edits). Use it to plan multi-step work and "
    "show progress: add one todo per concrete step before you start. Mark "
    "every todo being actively worked on `in_progress` — several at once when "
    "work genuinely runs in parallel (e.g. concurrent subagents or background "
    "commands), one for sequential work; while work remains, at least one "
    "task should be `in_progress`. Mark a todo `completed` the moment it is "
    "done (do not batch completions), and allow no `in_progress` item only "
    "once all work is complete. Skip the list for trivial single-step tasks. "
    "Statuses: `pending` (not started), `in_progress` (being worked on now), "
    "`completed` (finished).\n"
    "\n"
    "Project notes: mode='append' (default) merges by exact content at the "
    "ROOT level: root items are updated, new items are appended.\n"
    "NOTE: todo_write writes only match root-level items. Items inside the "
    "active stack scope (children) are NOT updated by this tool — use todo_sub "
    "to add or edit items under the current parent. "
    "Passing todos=[] is a no-op (list unchanged); use mode='clear' to empty the list.\n"
    "mode='replace' replaces the list only when every existing todo is done (errors otherwise); "
    "set force=True to replace unconditionally; "
    "mode='clear' empties the list (errors unless every old todo is done, or force=True).\n"
    "Keep exactly one item in_progress at a time and mark items done immediately after finishing them. "
    "If you send several in_progress items with auto_fix enabled (default), the LAST one is "
    "treated as the current focus and kept; the earlier ones are auto-completed.\n"
    "When a todo stack is active (a parent was pushed), the tree is shown with a `Stack:` breadcrumb — "
    "use todo_sub to add or edit items under the current parent and todo_pop to finish it."
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
    "todo_push to start a parent todo, or todo_write to read the tree."
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
    match_mode: Literal["exact", "fuzzy"] = Field(
        default="exact",
        description=(
            "'exact' (default): Match titles exactly. "
            "'fuzzy': Use fuzzy matching for near-miss titles when appending/updating."
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
                    hint='todo_sub to update an existing todo, or todo_write to read the tree.',
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
        # stay reachable via todo_push/todo_sub stack navigation: the deepest
        # pushable level is max_layers, plus one todo_sub level under it.
        max_layers = self._max_layers()
        max_depth = max_layers + 1
        if self._max_tree_depth(final_todos) > max_depth:
            return self._error(
                f"Error: Todo tree exceeds maximum nesting depth of {max_depth} levels "
                f"(todo_max_layers={max_layers}). Flatten the tree, or build it with todo_push/todo_sub.",
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
            hint = "todo_write to read the tree, or todo_push to start a parent todo."
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
        legal; it names the nested parent so the caller can switch to todo_sub.
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
                    "todo_write merges root-level titles only — use todo_sub to update it."
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
        stack = self._load_stack()

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

        # Breadcrumb (tree navigation): always echo where the model is.
        output_lines = ["Current todo list:"]
        if stack:
            output_lines.append(f"Stack: {' > '.join(stack)}")

        # Render the tree (children indented 2 spaces per depth), truncating
        # the flattened line list to _MAX_READ_ITEMS.
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

    # ---- Stack scope (tree navigation) -------------------------------------

    def _load_stack(self) -> list[str]:
        """Load the current todo stack (breadcrumb of ancestor titles).

        Root scope reads ``SessionState.todo_stack`` (reloaded from disk so it
        agrees with the persisted state); subagent scope reads the
        ``todo_stack`` key of the subagent state.json. Never raises — broken
        or missing state falls back to an empty stack.
        """
        if self._runtime.role == "root":
            from kimi_cli.session_state import load_session_state

            session = self._runtime.session
            fresh = load_session_state(session.dir)
            session.state.todo_stack = fresh.todo_stack
            return list(fresh.todo_stack)
        state_file = self._subagent_state_file()
        if state_file is None:
            return []
        data = self._read_subagent_state(state_file)
        raw = data.get("todo_stack")
        return [str(t) for t in raw] if isinstance(raw, list) else []

    def _save_stack(self, stack: list[str]) -> str | None:
        """Persist the todo stack. Returns an error message on failure."""
        clean = [str(t) for t in stack]
        if self._runtime.role == "root":
            try:
                session = self._runtime.session
                session.state.todo_stack = clean
                session.save_state()
                return None
            except Exception as exc:
                return f"Error: Failed to save todo stack: {exc}"
        state_file = self._subagent_state_file()
        if state_file is None:
            return "Error: Unable to save todo stack: state file is not available."
        data = self._read_subagent_state(state_file)
        data["todo_stack"] = clean
        try:
            self._write_subagent_state(state_file, data)
        except Exception as exc:
            return f"Error: Failed to save todo stack: {exc}"
        return None

    def _resolve_scope(
        self, todos: list[Todo]
    ) -> tuple[Todo | None, int, list[str], list[str]]:
        """Resolve the current stack scope against the todo tree.

        Walks the stack titles top-down through ``children``. When a title is
        missing (broken stack — e.g. the parent was force-overwritten), the
        stack is auto-healed: truncated to the longest existing prefix and a
        non-blocking warning is returned.

        Returns:
            ``(node, depth, healed_stack, warnings)`` — ``node`` is the
            current focus parent (``None`` when the stack is empty → root
            scope), ``depth`` is the node's layer (number of matched stack
            titles; 0 = root), ``healed_stack`` is the stack to persist (may
            be truncated), and ``warnings`` are non-blocking heal notices.
        """
        stack = self._load_stack()
        if not stack:
            return None, 0, [], []
        warnings: list[str] = []
        node: Todo | None = None
        scope = todos
        depth = 0
        for title in stack:
            child = next((t for t in scope if t.content == title), None)
            if child is None:
                healed = stack[:depth]
                warnings.append(
                    f"Todo stack healed: '{title}' no longer exists in the tree; "
                    f"stack truncated to {(' > '.join(healed)) if healed else 'root'}."
                )
                return node, depth, healed, warnings
            node = child
            scope = child.children
            depth += 1
        return node, depth, list(stack), warnings

    def _max_layers(self) -> int:
        """Maximum todo_write tree/stack depth (layers). Default 4."""
        try:
            value = self._runtime.config.loop_control.todo_max_layers
        except Exception:
            return 4
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return 4

    def _render_scope(self, todos: list[Todo], stack: list[str]) -> str:
        """Render the current stack scope: breadcrumb + scope children.

        Shows only the children of the current focus node (root list when the
        stack is empty), in ``_format_todos`` style with a 2-space indent per
        depth. Deeper unfinished descendants are collapsed into a per-item
        ``… N sub-tasks`` count so the output stays short.
        """
        scope = todos
        for title in stack:
            child = next((t for t in scope if t.content == title), None)
            if child is None:
                break
            scope = child.children

        lines: list[str] = []
        if stack:
            lines.append(f"Stack: {' > '.join(stack)}")
        if not scope:
            return "\n".join(lines) if lines else "(no todos in this scope)"

        labels = {"pending": "pending", "in_progress": "in progress", "done": "done"}
        base_indent = "  " * len(stack)
        for child in scope:
            if child.status == "done":
                continue  # keep the scope view short: unfinished items only
            line = f"{base_indent}- [{labels[child.status]}] {child.content}"
            deeper = self._count_unfinished_descendants(child)
            if deeper:
                line += f" … {deeper} sub-task{'s' if deeper != 1 else ''}"
            lines.append(line)
        return "\n".join(lines)

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

# ── Stack/tree tools: todo_push / todo_pop / todo_sub ───────────────────────────
# These operate on the same persisted todo tree as todo_write but navigate a
# ``todo_stack`` breadcrumb (root → current focus parent) instead of replacing
# the whole list. See plan: todo_write Stack & Tree Structure.


class TodoPushParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(description="Title", min_length=1, max_length=65536)
    notes: str | None = Field(
        default=None,
        description="Notes. MUST write, be comprehensively, detailed.",
        max_length=65536,
    )

    @field_validator("title")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Title cannot be empty or contain only whitespace")
        return stripped


class todo_push(TodoList):
    name: str = "todo_push"
    description: str = (
        "Push a new parent todo onto the current stack scope, making it the focus "
        "parent. Add sub-todos under it with todo_sub and finish it with todo_pop. "
        "Pushing deeper than the configured max layers errors."
    )
    params: type[TodoPushParams] = TodoPushParams

    def __init__(self, runtime: Runtime) -> None:
        # Subclass todo_write so all persistence/scope helpers are shared, but
        # bypass TodoList.__init__ (it bakes the TodoList description) and let
        # CallableTool2.__init__ read this class's own name/description/params
        # attrs.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoPushParams) -> ToolReturnValue:
        title = params.title
        todos = self._load_todos()
        stack = self._load_stack()
        node, depth, healed_stack, warnings = self._resolve_scope(todos)

        # Completely broken stack (non-empty original, nothing resolved): do not
        # silently fall back to root scope.
        if stack and node is None:
            hint = "Use todo_write to read the tree and todo_push to re-enter a parent."
            return ToolError(
                message="Todo stack is broken.",
                brief=hint,
                output=(
                    "Error: Todo stack is broken; cannot resolve the current scope."
                    + _hint_error(hint)
                ),
            )

        max_layers = self._max_layers()
        if depth >= max_layers:
            hint = "Use todo_sub to add sub-todos at this level instead of pushing deeper."
            breadcrumb = f"Stack: {' > '.join(healed_stack)}" if healed_stack else "Stack: (root)"
            return ToolError(
                message=f"Cannot push deeper than {max_layers} layers.",
                brief=hint,
                output=(
                    f"Error: Cannot push deeper than {max_layers} layers "
                    f"(current depth {depth}).\n{breadcrumb}"
                    + _hint_error(hint)
                ),
            )

        scope = node.children if node is not None else todos
        if any(t.content == title for t in scope):
            hint = f'Use todo_sub "{title}" to update the existing item.'
            return ToolError(
                message=f'Duplicate todo title "{title}" in this scope.',
                brief=hint,
                output=(
                    f'Error: Duplicate todo title "{title}" in this scope.'
                    + _hint_error(hint)
                ),
            )

        scope.append(Todo(content=title, status="pending", notes=params.notes))
        save_error = self._save_todos(todos, self._load_archived_todos())
        if save_error:
            hint = "Use todo_write to read the tree and retry todo_push."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        new_stack = [*healed_stack, title]
        stack_error = self._save_stack(new_stack)
        if stack_error:
            hint = "Use todo_write to read the tree and retry todo_push."
            return ToolError(
                message="Failed to save todo stack.",
                brief=hint,
                output=stack_error + _hint_error(hint),
            )

        render = self._render_scope(todos, new_stack)
        output = (
            render
            + "\n"
            + f'Pushed "{title}" (depth {depth + 1}/{max_layers}).'
            + _hint_next(
                f'todo_sub "<sub>" to add sub-todos under "{title}"; todo_pop to finish it.'
            )
        )
        if warnings:
            output += "\n" + "\n".join(warnings)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message=f'Pushed "{title}".',
            display=[self._build_display_block(todos)],
        )


class TodoPopParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: bool = Field(
        default=False,
        description=(
            "When True, mark the current focus parent and all its sub-todos done "
            "before popping. Default False: pop succeeds only when everything under "
            "the focus parent is already done; otherwise it errors."
        ),
    )


class todo_pop(TodoList):
    name: str = "todo_pop"
    description: str = (
        "Pop the current focus parent and return to the parent scope. "
        "Errors when the focus parent or any sub-todo is unfinished, unless "
        "complete=True (marks them all done first)."
    )
    params: type[TodoPopParams] = TodoPopParams

    def __init__(self, runtime: Runtime) -> None:
        # See todo_push.__init__: subclass todo_write for shared helpers, bypass
        # TodoList.__init__ and use this class's own name/description/params.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoPopParams) -> ToolReturnValue:
        _ = params
        todos = self._load_todos()
        stack = self._load_stack()
        if not stack:
            hint = "No parent to pop — use todo_push to create one, or todo_write to read the tree."
            return ToolError(
                message="No parent todo to pop.",
                brief="Use todo_push to create one, or todo_write to read the tree.",
                output="Error: No parent todo to pop (stack is empty)." + _hint_error(hint),
            )

        node, _depth, healed_stack, warnings = self._resolve_scope(todos)
        if node is None:
            hint = "Use todo_write to read the tree and todo_push to re-enter a parent."
            return ToolError(
                message="Todo stack is broken.",
                brief=hint,
                output=(
                    "Error: Todo stack is broken; cannot resolve the current scope."
                    + _hint_error(hint)
                ),
            )

        popped_title = node.content
        unfinished = self._count_unfinished_descendants(node)
        if node.status != "done":
            unfinished += 1

        # Pure scope-exit guard: popping must not silently complete unfinished work.
        if unfinished and not params.complete:
            hint = (
                f'"{popped_title}" has {unfinished} unfinished item(s). '
                "Finish them with todo_sub, or call todo_pop with complete=True "
                "to mark them done and pop."
            )
            return ToolError(
                message=f'Cannot pop "{popped_title}": {unfinished} unfinished item(s).',
                brief="Finish them or pass complete=True.",
                output=(
                    f'Error: Cannot pop "{popped_title}" — {unfinished} unfinished item(s).\n'
                    f"Stack: {' > '.join(healed_stack)}"
                    + _hint_error(hint)
                ),
            )

        self._mark_subtree_done(node)
        n = self._count_all([node])
        new_stack = healed_stack[:-1]

        save_error = self._save_todos(todos, self._load_archived_todos())
        if save_error:
            hint = "Use todo_write to read the tree and retry todo_pop."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        stack_error = self._save_stack(new_stack)
        if stack_error:
            hint = "Use todo_write to read the tree and retry todo_pop."
            return ToolError(
                message="Failed to save todo stack.",
                brief=hint,
                output=stack_error + _hint_error(hint),
            )

        render = self._render_scope(todos, new_stack)
        if params.complete:
            output = (
                render
                + "\n"
                + f'Popped "{popped_title}" — {n} sub-todo(s) marked done.'
            )
        else:
            output = render + "\n" + f'Popped "{popped_title}".'
        if unfinished:
            output += (
                f'\nNote: "{popped_title}" had {unfinished} unfinished item(s) — '
                "all marked done. Use todo_sub with force=True or todo_write "
                "mode='replace' with force=True to reopen if that was a mistake."
            )
        output += _hint_next("todo_push to start the next parent, or todo_write to read the tree.")
        if warnings:
            output += "\n" + "\n".join(warnings)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message=f'Popped "{popped_title}".',
            display=[self._build_display_block(todos)],
        )

class TodoSubParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(description="Title", min_length=1, max_length=65536)
    status: TodoStatus | None = Field(
        default=None,
        description=(
            "Status. One of: pending, in_progress, done. "
            "Omit to keep the existing status (new items default to pending)."
        ),
    )
    notes: str | None = Field(
        default=None,
        description="Notes. MUST write, be comprehensively, detailed.",
        max_length=65536,
    )
    rename_to: str | None = Field(
        default=None,
        description="Rename the matched sub-todo to this title.",
    )
    force: bool = Field(
        default=False,
        description="Allow regressing a 'done' item back to pending/in_progress.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: Any) -> str | None:
        if v is None:
            return None
        return _canonical_status(v)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Title cannot be empty or contain only whitespace")
        return stripped

    @field_validator("rename_to")
    @classmethod
    def _validate_rename_to(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

class todo_sub(TodoList):
    name: str = "todo_sub"
    description: str = (
        "Create/update a sub-todo under the current parent scope. Same-title "
        "calls update status/notes (status is preserved when omitted); new titles "
        "append a child. Set force=True to reopen a 'done' item."
    )
    params: type[TodoSubParams] = TodoSubParams

    def __init__(self, runtime: Runtime) -> None:
        # See todo_push.__init__: subclass todo_write for shared helpers, bypass
        # TodoList.__init__ and use this class's own name/description/params.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoSubParams) -> ToolReturnValue:
        todos = self._load_todos()
        stack = self._load_stack()
        node, _depth, healed_stack, warnings = self._resolve_scope(todos)
        if stack and node is None:
            hint = "Use todo_write to read the tree and todo_push to re-enter a parent."
            return ToolError(
                message="Todo stack is broken.",
                brief=hint,
                output=(
                    "Error: Todo stack is broken; cannot resolve the current scope."
                    + _hint_error(hint)
                ),
            )

        # Defensive depth guard: children of the deepest pushable node sit at
        # max_layers + 1; anything deeper is unreachable via stack navigation.
        max_layers = self._max_layers()
        if len(healed_stack) + 1 > max_layers + 1:
            hint = "Use todo_write to read the tree and flatten it, then todo_push/todo_sub to rebuild."
            return ToolError(
                message=f"Cannot add sub-todos deeper than {max_layers + 1} layers.",
                brief=hint,
                output=(
                    f"Error: Cannot add sub-todos deeper than {max_layers + 1} layers "
                    f"(todo_max_layers={max_layers})."
                    + _hint_error(hint)
                ),
            )

        scope = node.children if node is not None else todos
        parent_title = node.content if node is not None else "root"
        title = params.title
        existing = next((t for t in scope if t.content == title), None)

        if existing is None:
            # New title → append child (same-title would have matched above).
            scope.append(
                Todo(
                    content=title,
                    status=params.status if params.status is not None else "pending",
                    notes=params.notes,
                )
            )
            verb = "added"
            final_title = title
        else:
            # Same title → write: rename, then update status/notes.
            # Regression guard (mirrors todo_write): done → pending/in_progress
            # is blocked unless force=True. Status is preserved when omitted,
            # so a bare same-title call never resets an item to pending.
            new_status = params.status if params.status is not None else existing.status
            if existing.status == "done" and new_status != "done" and not params.force:
                hint = (
                    f'Use todo_sub "{title}" with force=True to reopen a done item, '
                    "or todo_write with mode='replace' and force=True to restart the whole list."
                )
                return ToolError(
                    message=f'Cannot regress completed todo "{title}" back to {new_status}.',
                    brief=hint,
                    output=(
                        f'Error: Cannot regress completed todo "{title}" back to {new_status}.'
                        + _hint_error(hint)
                    ),
                )
            if params.rename_to and params.rename_to != title:
                if any(t.content == params.rename_to for t in scope if t is not existing):
                    hint = f'Use todo_sub "{params.rename_to}" to update instead of renaming.'
                    return ToolError(
                        message=f'Cannot rename "{title}" to "{params.rename_to}": title already exists.',
                        brief=hint,
                        output=(
                            f'Error: Cannot rename "{title}" to "{params.rename_to}": '
                            "title already exists in this scope."
                            + _hint_error(hint)
                        ),
                    )
                old_title = existing.content
                existing.content = params.rename_to
                # If the renamed child IS the stack top, update the breadcrumb.
                if healed_stack and healed_stack[-1] == old_title:
                    healed_stack[-1] = params.rename_to
                title = params.rename_to

            # None/empty keeps old values, mirroring TodoList._merge_one.
            if params.notes is not None and params.notes.strip():
                existing.notes = params.notes
            existing.status = new_status
            verb = "updated"
            final_title = title

        save_error = self._save_todos(todos, self._load_archived_todos())
        if save_error:
            hint = "Use todo_write to read the tree and retry todo_sub."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        stack_error = self._save_stack(healed_stack)
        if stack_error:
            hint = "Use todo_write to read the tree and retry todo_sub."
            return ToolError(
                message="Failed to save todo stack.",
                brief=hint,
                output=stack_error + _hint_error(hint),
            )

        render = self._render_scope(todos, healed_stack)
        output = render + "\n" + f'Sub-todo "{final_title}" {verb} under "{parent_title}".'
        output += _hint_next(
            f'todo_sub "<next>" for more sub-todos, or todo_pop to finish "{parent_title}".'
        )
        if warnings:
            output += "\n" + "\n".join(warnings)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message=f'Sub-todo "{final_title}" {verb}.',
            display=[self._build_display_block(todos)],
        )
