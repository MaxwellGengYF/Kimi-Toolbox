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
    "Track progress with a todo list.\n"
    "Call with no arguments to read the current list. "
    "mode='append' (default) merges by exact title at the ROOT level: root titles are updated, new titles are appended.\n"
    "NOTE: TodoList writes only match root-level titles. Titles inside the active stack scope (children) are "
    "NOT updated by this tool — use TodoSub to add or edit items under the current parent. "
    "Passing todos=[] is a no-op (list unchanged); use mode='clear' to empty the list.\n"
    "mode='overwrite' replaces the list only when every existing todo is done; "
    "mode='force_overwrite' replaces the list unconditionally; "
    "mode='clear' empties the list (errors unless every old todo is done).\n"
    "Keep exactly one item in_progress at a time and mark items done immediately after finishing them.\n"
    "When a todo stack is active (a parent was pushed), the tree is shown with a `Stack:` breadcrumb — "
    "use TodoSub to add or edit items under the current parent and TodoPop to finish it."
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
    "TodoPush to start a parent todo, or TodoList to read the tree."
)

# Mode map — only canonical values accepted
_MODE_MAP: dict[str, Literal["overwrite", "append", "force_overwrite", "clear"]] = {
    "overwrite": "overwrite",
    "append": "append",
    "force_overwrite": "force_overwrite",
    "clear": "clear",
}

# Status map — only canonical values accepted
_STATUS_MAP: dict[str, TodoStatus] = {
    "pending": "pending",
    "in_progress": "in_progress",
    "done": "done",
}

def _canonical_status(v: Any) -> TodoStatus:
    """Normalize a status value to its canonical form."""
    if not isinstance(v, str):
        raise ValueError(
            f"Invalid status '{v}'. Must be one of: pending, in_progress, done."
        )
    normalized = v.strip().lower().replace("-", "_")
    canonical = _STATUS_MAP.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Invalid status '{v}'. Must be one of: pending, in_progress, done."
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

    title: str = Field(description="Title", min_length=1, max_length=65536)
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

    @field_validator("title")
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
        alias="items",  # common LLM variant
        description="Updated list, a single Todo item, or omit to return current list unchanged. "
        "Passing an empty list [] is a no-op (use mode='clear' to empty the list). "
        + alias_note("todos", "items", word=False),
    )
    mode: Literal["overwrite", "append", "force_overwrite", "clear"] = Field(
        default="append",
        description=(
            "Write mode: 'overwrite' safely replaces the existing todo list only when all old todos are done; "
            "'append' merges the provided todos into the existing list (existing root titles are updated, new titles are appended; empty list is a no-op); "
            "'force_overwrite' replaces the existing todo list unconditionally; "
            "'clear' empties the list (errors unless all old todos are done)."
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
            "items as done before applying the update."
        ),
    )
    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError(
                "Invalid mode. Must be 'overwrite', 'append', 'force_overwrite', or 'clear'."
            )
        normalized = v.strip().lower().replace("-", "_")
        canonical = _MODE_MAP.get(normalized)
        if canonical is None:
            raise ValueError(
                f"Invalid mode '{v}'. Must be 'overwrite', 'append', 'force_overwrite', or 'clear'."
            )
        return canonical

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
    name: str = "TodoList"
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
                    in_progress.append(t.title)
                walk(t.children)

        walk(todos)
        if len(in_progress) > 1:
            return in_progress
        return None

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
                "Use mode='append' or 'overwrite' to write todos, or call with no todos to read.",
                "mode='clear' cannot be combined with todos.",
            )

        # 1. Validate new inputs
        if params.mode != "clear":
            duplicates = self._find_duplicate_titles(new_todos)
            if duplicates:
                return self._error(
                    f"Error: Duplicate todo titles found: {duplicates}",
                    f"Duplicate todo titles found: {duplicates}",
                    hint='TodoSub to update an existing todo, or TodoList to read the tree.',
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
        # items (overwrite/force_overwrite/clear) so completed ones get archived.
        warnings: list[str] = []
        replaces_list = False
        if params.mode == "clear":
            if old_todos and not all(t.status == "done" for t in old_todos):
                unfinished = "\n".join(t.title for t in old_todos if t.status != "done")
                return self._error(
                    "Error: Cannot clear todos while old todos are not all done. "
                    "Next step: mark them done first, "
                    "or use mode='force_overwrite' with todos=[] to discard them intentionally.\n"
                    f"Unfinished:\n{unfinished}",
                    "Cannot clear todos while old todos are not all done.",
                    display=[self._build_display_block(old_todos)],
                )
            final_todos = []
            replaces_list = True
        elif params.mode == "force_overwrite":
            final_todos = list(new_todos)
            replaces_list = True
        elif params.mode == "overwrite":
            if old_todos and not all(t.status == "done" for t in old_todos):
                unfinished = "\n".join(t.title for t in old_todos if t.status != "done")
                return self._error(
                    "Error: Cannot overwrite todos while old todos are not all done. "
                    "Use mode='force_overwrite' if you really want to discard unfinished work.\n"
                    f"Unfinished:\n{unfinished}",
                    "Cannot overwrite todos while old todos are not all done.",
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
        # stay reachable via TodoPush/TodoSub stack navigation: the deepest
        # pushable level is max_layers, plus one TodoSub level under it.
        max_layers = self._max_layers()
        max_depth = max_layers + 1
        if self._max_tree_depth(final_todos) > max_depth:
            return self._error(
                f"Error: Todo tree exceeds maximum nesting depth of {max_depth} levels "
                f"(todo_max_layers={max_layers}). Flatten the tree, or build it with TodoPush/TodoSub.",
                f"Todo tree exceeds maximum nesting depth of {max_depth} levels.",
                display=[self._build_display_block(final_todos)],
            )

        # 4. Regression detection
        if params.mode not in ("force_overwrite", "clear") and old_todos:
            final_todos, regressions = self._check_regressions(old_todos, final_todos)
            if regressions:
                return self._error(
                    "Error: Cannot regress completed todos back to pending/in_progress: "
                    + ", ".join(regressions)
                    + "\nNext step: resend with these items kept as 'done', "
                    "or use mode='force_overwrite' to restart them intentionally.",
                    "Cannot regress completed todos.",
                    display=[self._build_display_block(final_todos)],
                )

        # 5. Archive completed todos dropped by overwrite/force_overwrite/clear
        archived = list(old_archived)
        if replaces_list and old_todos:
            kept_titles = {t.title for t in final_todos}
            newly_archived = [
                t for t in old_todos if t.status == "done" and t.title not in kept_titles
            ]
            if newly_archived:
                archived.extend(self._item_states(newly_archived))
                archived = archived[-_MAX_ARCHIVED_TODOS:]

        # 5b. Enforce single in_progress (unless auto_fix or force_overwrite)
        if params.mode not in ("force_overwrite", "clear"):
            conflicts = self._enforce_single_in_progress(final_todos)
            if conflicts:
                if params.auto_fix:
                    # Auto-fix: mark extra in_progress items as done
                    fixed_todos: list[Todo] = []
                    seen_in_progress = False
                    for t in final_todos:
                        if t.status == "in_progress":
                            if seen_in_progress:
                                fixed_todos.append(t.model_copy(update={"status": "done"}))
                                warnings.append(f'Auto-fixed "{t.title}": set to done (only one item may be in_progress)')
                            else:
                                seen_in_progress = True
                                fixed_todos.append(t)
                        else:
                            fixed_todos.append(t)
                    final_todos = fixed_todos
                else:
                    return self._error(
                        f"Error: Multiple items are in_progress: {conflicts}. "
                        "Keep exactly one item in_progress at a time. "
                        "Mark the current item as 'done' before starting another, "
                        "use mode='force_overwrite' to override, "
                        "or set auto_fix=True to automatically resolve conflicts.",
                        "Multiple items in_progress",
                        display=[self._build_display_block(final_todos)],
                    )

        # 6. Persist exactly once
        save_error = self._save_todos(final_todos, archived)
        if save_error:
            return self._error(save_error, "Failed to save todos.")

        # 7. Build response
        result = self._build_success_response(final_todos, params.mode, bool(old_todos), warnings)
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
            hint = "TodoList to read the tree, or TodoPush to start a parent todo."
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
            if t.title in seen:
                duplicates.add(t.title)
            else:
                seen.add(t.title)
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
            todo = f"- [{display_status[t.status]}] {t.title}"
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

        old_title_list = [t.title for t in old_todos]
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

        TodoList append merges root-level titles only, so a title that exists
        only inside a stack scope (a child/grandchild) would be appended as a
        brand-new root item instead of updating the nested one. The warning is
        non-blocking because identical titles in different scopes are otherwise
        legal; it names the nested parent so the caller can switch to TodoSub.
        """
        if not old_todos or not new_todos:
            return []
        root_titles = {t.title for t in old_todos}
        nested: dict[str, str] = {}

        def walk(items: list[Todo], path: list[str]) -> None:
            for t in items:
                if path:
                    nested.setdefault(t.title, " > ".join(path))
                walk(t.children, [*path, t.title])

        walk(old_todos, [])
        warnings: list[str] = []
        for t in new_todos:
            if t.title in root_titles:
                continue
            parent = nested.get(t.title)
            if parent:
                warnings.append(
                    f'"{t.title}" already exists in the tree (under "{parent}"); '
                    "TodoList merges root-level titles only — use TodoSub to update it."
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
            if new_todo.title in old_title_set:
                continue
            nearest = self._find_nearest_titles(
                [new_todo.title],
                old_title_list,
                top_k=1,
                score_cutoff=TodoList._FUZZY_WARNING_CUTOFF,
                processor=str.lower,
            )
            hits = nearest.get(new_todo.title, [])
            if hits:
                warnings.append(f'"{new_todo.title}" looks like existing "{hits[0].choice}"')
        return warnings

    def _merge_by_title_update(self, old_todos: list[Todo], new_todos: list[Todo]) -> list[Todo]:
        """Update existing titles and append brand-new ones."""
        new_by_title = {t.title: t for t in new_todos}
        merged: list[Todo] = []
        seen: set[str] = set()

        for old in old_todos:
            new = new_by_title.get(old.title)
            if new is not None:
                merged.append(self._merge_one(old, new))
            else:
                merged.append(old)
            seen.add(old.title)

        for new in new_todos:
            if new.title not in seen:
                merged.append(new)
                seen.add(new.title)

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
            title=old.title,
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
                old_status_map[t.title] = t.status
                collect(t.children)

        collect(old_todos)

        regressions: list[str] = []

        def clamp(items: list[Todo]) -> list[Todo]:
            out: list[Todo] = []
            for t in items:
                if old_status_map.get(t.title) == "done" and t.status != "done":
                    regressions.append(t.title)
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
    ) -> ToolReturnValue:
        display_block = self._build_display_block(todos)
        active_summary = self._format_todos(todos)
        counts = self._status_counts(todos)

        mode_msg = {
            "append": "appended",
            "overwrite": "overwritten",
            "force_overwrite": "force overwritten",
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
        if mode == "force_overwrite" and had_old_todos:
            message_lines.append(
                "Warning: mode='force_overwrite' replaces the existing todo list and bypasses merge validation logic."
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
                        title=todo.title,
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
                line = f"{'  ' * depth}- [{display_status[t.status]}] {t.title}"
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
            child = next((t for t in scope if t.title == title), None)
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
        """Maximum TodoList tree/stack depth (layers). Default 4."""
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
            child = next((t for t in scope if t.title == title), None)
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
            line = f"{base_indent}- [{labels[child.status]}] {child.title}"
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
        return [TodoItemState(**todo.model_dump()) for todo in todos]

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

# ── Stack/tree tools: TodoPush / TodoPop / TodoSub ───────────────────────────
# These operate on the same persisted todo tree as TodoList but navigate a
# ``todo_stack`` breadcrumb (root → current focus parent) instead of replacing
# the whole list. See plan: TodoList Stack & Tree Structure.


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


class TodoPush(TodoList):
    name: str = "TodoPush"
    description: str = (
        "Push a new parent todo onto the current stack scope, making it the focus "
        "parent. Add sub-todos under it with TodoSub and finish it with TodoPop. "
        "Pushing deeper than the configured max layers errors."
    )
    params: type[TodoPushParams] = TodoPushParams

    def __init__(self, runtime: Runtime) -> None:
        # Subclass TodoList so all persistence/scope helpers are shared, but
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
            hint = "Use TodoList to read the tree and TodoPush to re-enter a parent."
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
            hint = "Use TodoSub to add sub-todos at this level instead of pushing deeper."
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
        if any(t.title == title for t in scope):
            hint = f'Use TodoSub "{title}" to update the existing item.'
            return ToolError(
                message=f'Duplicate todo title "{title}" in this scope.',
                brief=hint,
                output=(
                    f'Error: Duplicate todo title "{title}" in this scope.'
                    + _hint_error(hint)
                ),
            )

        scope.append(Todo(title=title, status="pending", notes=params.notes))
        save_error = self._save_todos(todos, self._load_archived_todos())
        if save_error:
            hint = "Use TodoList to read the tree and retry TodoPush."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        new_stack = [*healed_stack, title]
        stack_error = self._save_stack(new_stack)
        if stack_error:
            hint = "Use TodoList to read the tree and retry TodoPush."
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
                f'TodoSub "<sub>" to add sub-todos under "{title}"; TodoPop to finish it.'
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


class TodoPop(TodoList):
    name: str = "TodoPop"
    description: str = (
        "Pop the current focus parent: mark it and all its sub-todos done and "
        "return to the parent scope."
    )
    params: type[TodoPopParams] = TodoPopParams

    def __init__(self, runtime: Runtime) -> None:
        # See TodoPush.__init__: subclass TodoList for shared helpers, bypass
        # TodoList.__init__ and use this class's own name/description/params.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoPopParams) -> ToolReturnValue:
        _ = params
        todos = self._load_todos()
        stack = self._load_stack()
        if not stack:
            hint = "No parent to pop — use TodoPush to create one, or TodoList to read the tree."
            return ToolError(
                message="No parent todo to pop.",
                brief="Use TodoPush to create one, or TodoList to read the tree.",
                output="Error: No parent todo to pop (stack is empty)." + _hint_error(hint),
            )

        node, _depth, healed_stack, warnings = self._resolve_scope(todos)
        if node is None:
            hint = "Use TodoList to read the tree and TodoPush to re-enter a parent."
            return ToolError(
                message="Todo stack is broken.",
                brief=hint,
                output=(
                    "Error: Todo stack is broken; cannot resolve the current scope."
                    + _hint_error(hint)
                ),
            )

        popped_title = node.title
        unfinished = self._count_unfinished_descendants(node)
        if node.status != "done":
            unfinished += 1
        self._mark_subtree_done(node)
        n = self._count_all([node])
        new_stack = healed_stack[:-1]

        save_error = self._save_todos(todos, self._load_archived_todos())
        if save_error:
            hint = "Use TodoList to read the tree and retry TodoPop."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        stack_error = self._save_stack(new_stack)
        if stack_error:
            hint = "Use TodoList to read the tree and retry TodoPop."
            return ToolError(
                message="Failed to save todo stack.",
                brief=hint,
                output=stack_error + _hint_error(hint),
            )

        render = self._render_scope(todos, new_stack)
        output = (
            render
            + "\n"
            + f'Popped "{popped_title}" — {n} sub-todo(s) marked done.'
        )
        if unfinished:
            output += (
                f'\nNote: "{popped_title}" had {unfinished} unfinished item(s) — '
                "all marked done. Use TodoSub with force=True or TodoList "
                "mode='force_overwrite' to reopen if that was a mistake."
            )
        output += _hint_next("TodoPush to start the next parent, or TodoList to read the tree.")
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

class TodoSub(TodoList):
    name: str = "TodoSub"
    description: str = (
        "Create/update a sub-todo under the current parent scope. Same-title "
        "calls update status/notes (status is preserved when omitted); new titles "
        "append a child. Set force=True to reopen a 'done' item."
    )
    params: type[TodoSubParams] = TodoSubParams

    def __init__(self, runtime: Runtime) -> None:
        # See TodoPush.__init__: subclass TodoList for shared helpers, bypass
        # TodoList.__init__ and use this class's own name/description/params.
        CallableTool2.__init__(self)
        self._runtime = runtime

    @override
    async def __call__(self, params: TodoSubParams) -> ToolReturnValue:
        todos = self._load_todos()
        stack = self._load_stack()
        node, _depth, healed_stack, warnings = self._resolve_scope(todos)
        if stack and node is None:
            hint = "Use TodoList to read the tree and TodoPush to re-enter a parent."
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
            hint = "Use TodoList to read the tree and flatten it, then TodoPush/TodoSub to rebuild."
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
        parent_title = node.title if node is not None else "root"
        title = params.title
        existing = next((t for t in scope if t.title == title), None)

        if existing is None:
            # New title → append child (same-title would have matched above).
            scope.append(
                Todo(
                    title=title,
                    status=params.status if params.status is not None else "pending",
                    notes=params.notes,
                )
            )
            verb = "added"
            final_title = title
        else:
            # Same title → write: rename, then update status/notes.
            # Regression guard (mirrors TodoList): done → pending/in_progress
            # is blocked unless force=True. Status is preserved when omitted,
            # so a bare same-title call never resets an item to pending.
            new_status = params.status if params.status is not None else existing.status
            if existing.status == "done" and new_status != "done" and not params.force:
                hint = (
                    f'Use TodoSub "{title}" with force=True to reopen a done item, '
                    "or TodoList with mode='force_overwrite' to restart the whole list."
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
                if any(t.title == params.rename_to for t in scope if t is not existing):
                    hint = f'Use TodoSub "{params.rename_to}" to update instead of renaming.'
                    return ToolError(
                        message=f'Cannot rename "{title}" to "{params.rename_to}": title already exists.',
                        brief=hint,
                        output=(
                            f'Error: Cannot rename "{title}" to "{params.rename_to}": '
                            "title already exists in this scope."
                            + _hint_error(hint)
                        ),
                    )
                old_title = existing.title
                existing.title = params.rename_to
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
            hint = "Use TodoList to read the tree and retry TodoSub."
            return ToolError(
                message="Failed to save todos.",
                brief=hint,
                output=save_error + _hint_error(hint),
            )
        stack_error = self._save_stack(healed_stack)
        if stack_error:
            hint = "Use TodoList to read the tree and retry TodoSub."
            return ToolError(
                message="Failed to save todo stack.",
                brief=hint,
                output=stack_error + _hint_error(hint),
            )

        render = self._render_scope(todos, healed_stack)
        output = render + "\n" + f'Sub-todo "{final_title}" {verb} under "{parent_title}".'
        output += _hint_next(
            f'TodoSub "<next>" for more sub-todos, or TodoPop to finish "{parent_title}".'
        )
        if warnings:
            output += "\n" + "\n".join(warnings)
        return ToolReturnValue(
            is_error=False,
            output=output,
            message=f'Sub-todo "{final_title}" {verb}.',
            display=[self._build_display_block(todos)],
        )
