"""Todo list tracking tool."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, override

import orjson
import rapidfuzz
from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import (
    AliasChoices, BaseModel, ConfigDict, Field,
    ValidationError, field_validator, model_validator,
)

from kimi_cli import logger
from kimi_cli.session_state import TodoItemState, TodoStatus
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.display import TodoDisplayBlock, TodoDisplayItem
from kimi_cli.tools.utils import repair_json_string


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



# Mode map — only canonical values accepted
_MODE_MAP: dict[str, Literal["overwrite", "append", "force_overwrite"]] = {
    "overwrite": "overwrite",
    "append": "append",
    "force_overwrite": "force_overwrite",
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
        description="Notes.",
        max_length=65536,
    )
    code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("code", "code_file"),
        description=(
            "Verification code: inline Python, a `.py` file path, a shell command "
            "prefixed with `!` (e.g. `!pytest tests/ -x -q`), or a `.sh`/`.ps1` file path. "
            "Todos involving code changes should attach verification code. "
            "Omit if this todo has no executable code. "
            "Pass empty string to clear previously set code. "
            "Accepts `code` or `code_file`."
        ),
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

    @field_validator("code", mode="before")
    @classmethod
    def _validate_code(cls, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    todos: list[Todo] | Todo | None = Field(
        default=None,
        alias="items",  # common LLM variant
        description="Updated list, a single Todo item, or omit to return current list unchanged. "
        "Accepts `todos` or `items`.",
    )
    mode: Literal["overwrite", "append", "force_overwrite"] = Field(
        default="append",
        description=(
            "Write mode: 'overwrite' safely replaces the existing todo list only when all old todos are done; "
            "'append' merges the provided todos into the existing list (existing titles are updated, new titles are appended); "
            "'force_overwrite' replaces the existing todo list unconditionally."
        ),
    )
    match_mode: Literal["exact", "fuzzy"] = Field(
        default="exact",
        description=(
            "'exact': Match titles exactly. "
            "'fuzzy' (default): Use fuzzy matching for near-miss titles when appending/updating."
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
                "Invalid mode. Must be 'overwrite', 'append', or 'force_overwrite'."
            )
        normalized = v.strip().lower().replace("-", "_")
        canonical = _MODE_MAP.get(normalized)
        if canonical is None:
            raise ValueError(
                f"Invalid mode '{v}'. Must be 'overwrite', 'append', or 'force_overwrite'."
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
    description: str = (
        "Track progress with a todo list.\n"
        "Call with no arguments to read the current list. "
        "mode='append' (default) merges by exact title: existing titles are updated, new titles are appended.\n"
        "mode='overwrite' replaces the list only when every existing todo is done; "
        "use mode='force_overwrite' to intentionally discard unfinished items.\n"
        "Keep exactly one item in_progress at a time and mark items done immediately after finishing them.\n"
        "Each todo may include a `code` field with inline Python, a `.py` file path, "
        "a `!`-prefixed shell command (e.g. `!pytest tests/ -x -q`), or a `.sh`/`.ps1` file path. "
        "Todos involving code changes should attach verification `code`.\n"
        "When a todo is marked as `done` (previous status was not done), its `code` is automatically executed for verification.\n"
        "If verification fails, errors are accumulated and returned; multiple `done` triggers accumulate multiple errors."
    )
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if params.todos is not None:
            return self._write_todos(params.todos, params)
        return self._read_todos()

    # ---- Write mode --------------------------------------------------------

    @staticmethod
    def _enforce_single_in_progress(todos: list[Todo]) -> list[str] | None:
        """Return list of titles that are in_progress if >1, else None."""
        in_progress = [t.title for t in todos if t.status == "in_progress"]
        if len(in_progress) > 1:
            return in_progress
        return None

    def _write_todos(
        self,
        raw_todos: list[Todo] | Todo,
        params: Params,
    ) -> ToolReturnValue:
        """Validate, merge, and persist todos, saving exactly once on success."""
        new_todos: list[Todo] = [raw_todos] if isinstance(raw_todos, Todo) else list(raw_todos)

        # 1. Validate new inputs
        duplicates = self._find_duplicate_titles(new_todos)
        if duplicates:
            return self._error(
                f"Error: Duplicate todo titles found: {duplicates}",
                f"Duplicate todo titles found: {duplicates}",
            )

        if len(new_todos) > _MAX_TODOS:
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
        if params.mode == "force_overwrite":
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
            result = self._merge_todos(old_todos, new_todos, clear_requested=True)
            if result.error is not None:
                return result.error
            final_todos = result.todos or []
            warnings.extend(result.warnings)
            replaces_list = not new_todos  # explicit [] means clear

        # 4. Regression detection
        if params.mode != "force_overwrite" and old_todos:
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
        if params.mode != "force_overwrite":
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
        return self._build_success_response(final_todos, params.mode, bool(old_todos), warnings)

    @staticmethod
    def _error(
        output: str,
        message: str,
        display: list[Any] | None = None,
    ) -> ToolReturnValue:
        """Build an error ToolReturnValue with consistent shape."""
        return ToolReturnValue(
            is_error=True,
            output=output,
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
            if t.code:
                stripped = t.code.strip()
                if stripped.startswith("!"):
                    kind_label = "shell"
                elif stripped.lower().endswith((".py", ".sh", ".ps1")) and Path(stripped).is_file():
                    kind_label = "file"
                else:
                    kind_label = "inline"
                todo += f"  `[code: {kind_label}]`"
            if t.status == "in_progress" and t.notes:
                todo += f"  Notes: {t.notes}"
            lines.append(todo)

        return "\n".join(lines)

    @staticmethod
    def _resolve_code_executable(code: str) -> tuple[str, str] | None:
        """Resolve how to execute a todo's code string.

        Returns a ``(kind, payload)`` tuple, or ``None`` when the code has
        no runnable content. Kinds:

        - ``("shell", command)`` — code starts with ``!`` (shell command);
        - ``("shell_file", path)`` — code points to a ``.sh``/``.ps1`` file;
        - ``("python", path)`` — code points to an existing ``.py`` file;
        - ``("python_inline", temp_path)`` — inline Python written to a temp file.
        """
        if not code:
            return None
        stripped = code.strip()
        if stripped.startswith("!"):
            command = stripped[1:].strip()
            return ("shell", command) if command else None
        fp = Path(code)
        if fp.is_file():
            lowered = code.lower()
            if lowered.endswith(".py"):
                return ("python", str(fp))
            if lowered.endswith((".sh", ".ps1")):
                return ("shell_file", str(fp))
            return None
        # Inline Python code — write to temp file
        import tempfile as _tf
        fd, path = _tf.mkstemp(suffix=".py", prefix="run_todo_code_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        return ("python_inline", path)

    @staticmethod
    async def _run_process(
        argv: list[str],
        timeout: int,
        *,
        not_found_hint: str,
    ) -> tuple[bool, str]:
        """Run a subprocess with timeout, returning ``(success, output)``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return False, not_found_hint
        except Exception as exc:
            return False, f"Code execution error: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Code execution timed out after {timeout}s."

        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            error_text = stderr.decode("utf-8", errors="replace")
            if output:
                output += "\n" + error_text
            else:
                output = error_text

        if proc.returncode == 0:
            return True, output or "Code executed successfully (no output)."
        return False, f"Code failed (exit code {proc.returncode}):\n{output}"

    @staticmethod
    def _shell_argv(command: str) -> tuple[list[str], str]:
        """Build the argv for a shell command, plus a not-found hint."""
        if sys.platform == "win32":
            import shutil

            shell_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
            return (
                [shell_exe, "-NoProfile", "-NonInteractive", "-Command", command],
                f"Shell executable not found: {shell_exe}",
            )
        return (["bash", "-l", "-c", command], "bash executable not found")

    @staticmethod
    async def _run_code(
        code: str,
        timeout: int = 30,
        python_exe: str | None = None,
        executable: tuple[str, str] | str | None = None,
    ) -> tuple[bool, str]:
        """Execute a todo's code and return ``(success, output_or_error)``.

        ``executable`` may be a ``(kind, payload)`` tuple as returned by
        ``_resolve_code_executable``, or a legacy plain ``.py`` path string.
        When omitted, the executable is resolved from ``code``.
        """
        if executable is None:
            resolved = TodoList._resolve_code_executable(code)
        elif isinstance(executable, tuple):
            resolved = executable
        else:
            resolved = ("python", executable)  # legacy plain path
        if resolved is None:
            return False, "Todo has no runnable code."
        kind, payload = resolved

        if kind == "shell":
            argv, hint = TodoList._shell_argv(payload)
            return await TodoList._run_process(argv, timeout, not_found_hint=hint)
        if kind == "shell_file":
            if payload.lower().endswith(".ps1"):
                import shutil

                shell_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
                argv = [shell_exe, "-NoProfile", "-NonInteractive", "-File", payload]
                hint = f"Shell executable not found: {shell_exe}"
            else:
                argv = ["bash", "-l", payload]
                hint = "bash executable not found"
            return await TodoList._run_process(argv, timeout, not_found_hint=hint)

        # python / python_inline
        if python_exe is None:
            python_exe = sys.executable
        return await TodoList._run_process(
            [python_exe, payload],
            timeout,
            not_found_hint=f"Python executable not found: {python_exe}",
        )

    @staticmethod
    def _cleanup_code_tempfile(executable: tuple[str, str] | str | None) -> None:
        """Clean up a temp file created for inline code execution.

        Accepts either a ``(kind, payload)`` tuple (only ``python_inline``
        payloads are removed) or a legacy plain path string.
        """
        if executable is None:
            return
        if isinstance(executable, tuple):
            kind, payload = executable
            if kind != "python_inline":
                return
            temp_file_path: str | None = payload
        else:
            temp_file_path = executable
        if temp_file_path:
            try:
                Path(temp_file_path).unlink(missing_ok=True)
            except OSError:
                pass

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
        clear_requested: bool = False,
    ) -> MergeResult:
        """Merge ``new_todos`` into ``old_todos`` using append/update semantics.

        * Existing titles update status (and any provided metadata) in place.
        * Brand-new titles are appended to the end.
        * An explicitly empty ``new_todos`` (``clear_requested=True``) clears the
          list only when all old todos are done.
        * Fuzzy near-matches are reported as non-blocking warnings.
        """
        if not old_todos:
            return MergeResult(todos=list(new_todos))

        old_title_list = [t.title for t in old_todos]
        old_title_set = set(old_title_list)

        warnings = self._detect_fuzzy_warnings(new_todos, old_title_set, old_title_list)

        # Explicitly empty list: treat as clear operation
        if clear_requested and not new_todos:
            if not all(t.status == "done" for t in old_todos):
                unfinished = ", ".join(t.title for t in old_todos if t.status != "done")
                return MergeResult(
                    error=self._error(
                        "Error: Cannot clear todos while old todos are not all done. "
                        f"Unfinished: {unfinished}\n"
                        "Next step: mark them done first, "
                        "or use mode='force_overwrite' to discard them intentionally.",
                        "Cannot clear todos while old todos are not all done.",
                    )
                )
            return MergeResult(todos=[])

        merged = self._merge_by_title_update(old_todos, new_todos)
        return MergeResult(todos=merged, warnings=warnings)

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
        """Produce an updated todo preserving old notes/code when new omits them."""
        return Todo(
            title=old.title,
            status=new.status,
            notes=new.notes if new.notes is not None else old.notes,
            code=new.code if new.code is not None else old.code,
        )

    @staticmethod
    def _check_regressions(
        old_todos: list[Todo], final_todos: list[Todo]
    ) -> tuple[list[Todo], list[str]]:
        """Detect done todos being moved back to pending/in_progress.

        Returns the final list with regressed items clamped back to ``done``,
        plus the list of regressed titles.
        """
        old_status_map = {t.title: t.status for t in old_todos}

        regressions: list[str] = []
        clamped: list[Todo] = []
        for t in final_todos:
            if old_status_map.get(t.title) == "done" and t.status != "done":
                regressions.append(t.title)
                clamped.append(t.model_copy(update={"status": "done"}))
            else:
                clamped.append(t)
        return clamped, regressions

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
        }[mode]

        stats = (
            f"({len(todos)} total: {counts['done']} done, "
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
        """Count todos by status."""
        counts: dict[TodoStatus, int] = {"pending": 0, "in_progress": 0, "done": 0}
        for t in todos:
            counts[t.status] += 1
        return counts

    @staticmethod
    def _build_display_block(todos: list[Todo]) -> TodoDisplayBlock:
        return TodoDisplayBlock(
            items=[
                TodoDisplayItem(
                    title=todo.title,
                    status=todo.status,
                    notes=todo.notes,
                    code=todo.code,
                )
                for todo in todos
            ]
        )

    async def _verify_and_set_todo_status(self, todo_title: str, status: TodoStatus, append_notes: str = "") -> str | None:
        """Set a todo's status and auto-verify code when marking `done`.

        When marking a todo `done` (previous status was NOT `done`), if the todo has
        a `code` field, it is executed for verification. If execution fails, the error
        is collected and returned as part of the result string.
        Multiple `done` status changes can each trigger their own verification;
        errors from each are accumulated into a single newline-separated string.

        Returns error message string on verification/persistence failure, None on success.
        """
        todos = self._load_todos()
        archived = self._load_archived_todos()
        errors: list[str] = []
        for t in todos:
            if t.title == todo_title:
                was_done = t.status == "done"
                is_becoming_done = status == "done" and not was_done
                t.status = status
                if append_notes:
                    t.notes = ((t.notes or "") + "\n" + append_notes).strip() or None
                # Auto-verify code when transitioning to done
                if is_becoming_done and t.code:
                    executable = TodoList._resolve_code_executable(t.code)
                    if executable is not None:
                        try:
                            success, output = await TodoList._run_code(t.code, executable=executable)
                        except Exception as exc:
                            success, output = False, str(exc)
                        finally:
                            TodoList._cleanup_code_tempfile(executable)

                        if not success:
                            err_msg = f"Todo '{t.title}' verification failed: {output}"
                            errors.append(err_msg)
                            # Revert status to pending on verification failure
                            t.status = "pending"
                            t.notes = ((t.notes or "") + "\n" + err_msg).strip() or None
                break
        combined_errors = "\n".join(errors) if errors else None
        persist_err = self._save_todos(todos, archived)
        if persist_err and combined_errors:
            return persist_err + "\n" + combined_errors
        return persist_err or combined_errors

    # ---- Read mode ---------------------------------------------------------

    def _read_todos(self) -> ToolReturnValue:
        todos = self._load_todos()
        archived = self._load_archived_todos()

        if not todos:
            empty_lines = ["Todo list is empty."]
            if archived:
                empty_lines.append(f"Archived: {len(archived)} completed todo(s).")
            return ToolReturnValue(
                is_error=False,
                output="\n".join(empty_lines),
                message="Todo list is empty.",
                display=[],
            )

        # NEW: Check if all todos are done
        all_done = all(t.status == "done" for t in todos)

        shown = todos[:_MAX_READ_ITEMS]
        formatted = self._format_todos(
            shown,
            status_filter=("pending", "in_progress", "done"),
            display_status={
                "pending": "pending",
                "in_progress": "in_progress",
                "done": "done",
            },
        )
        output_lines = ["Current todo list:"]
        if formatted:
            output_lines.append(formatted)

        if len(todos) > _MAX_READ_ITEMS:
            counts = self._status_counts(todos)
            output_lines.append(
                f"... and {len(todos) - _MAX_READ_ITEMS} more "
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
