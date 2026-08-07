from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel, Field, ValidationError

from kimi_cli.utils.io import atomic_json_write
from kimi_cli.utils.logging import logger

# Shared status type — single source of truth for valid todo statuses
TodoStatus = Literal["pending", "in_progress", "done"]

STATE_FILE_NAME = "state.json"


class ApprovalStateData(BaseModel):
    yolo: bool = False
    afk: bool = False
    auto_approve_actions: set[str] = Field(default_factory=set)


class TodoItemState(BaseModel):
    """A single todo item stored in session or subagent state."""

    title: str
    status: TodoStatus
    notes: str | None = None
    code: str | None = None


# ── Compression re-injection (Hermes parity) ────────────────────────────────
# The active (unfinished) todo list is deterministically appended to the
# context-compaction output under this stable header so the plan survives
# summarization. The header wording is part of the contract — keep it
# byte-identical to Hermes's `TODO_INJECTION_HEADER`.
TODO_INJECTION_HEADER = (
    "[Your active task list was preserved across context compression]"
)

# Hermes markers are `[x]`/`[>]`/`[ ]`/`[~]`; Kimi only has the two
# non-done statuses below (`done` items are deliberately excluded).
TODO_INJECTION_MARKERS: dict[TodoStatus, str] = {
    "pending": "[ ]",
    "in_progress": "[>]",
}

TODO_INJECTION_TRUNCATION_MARKER = "… [truncated]"


# A plain (non-`None`) title is required for an item to be injected.
def format_todo_injection(
    todos: Sequence[TodoItemState],
    *,
    max_items: int = 20,
    max_chars: int = 4096,
    per_title_chars: int = 200,
) -> str | None:
    """Render the active (unfinished) todo list for compaction re-injection.

    Hermes-style parity: only ``pending``/``in_progress`` items are included
    (``done`` items are excluded so the model does not re-do finished work)
    and the result is appended to the compaction output under
    :data:`TODO_INJECTION_HEADER`.

    Pure function: no I/O, and it must never raise on malformed input
    (defensive ``getattr`` on ``status``/``title``; malformed items are
    skipped).

    Args:
        todos: The todo items to render.
        max_items: Maximum number of lines to emit; excess items are replaced
            by a single overflow line.
        max_chars: Hard cap on the total output length; lines are dropped
            from the tail (never a partial line) and the truncation marker
            is appended.
        per_title_chars: Per-title truncation length.

    Returns:
        The formatted injection text, or ``None`` when there is nothing to
        inject (empty input or no unfinished items).
    """
    if not todos:
        return None
    lines: list[str] = []
    for item in todos:
        status = getattr(item, "status", None)
        title = getattr(item, "title", None)
        if status not in TODO_INJECTION_MARKERS or not title:
            continue  # defensive: skip malformed items, never raise
        if len(title) > per_title_chars:
            title = title[:per_title_chars] + TODO_INJECTION_TRUNCATION_MARKER
        lines.append(f"- {TODO_INJECTION_MARKERS[status]} {title} ({status})")
    if not lines:
        return None
    if len(lines) > max_items:
        overflow = len(lines) - max_items
        lines = lines[:max_items]
        lines.append(f"- … and {overflow} more (call TodoList to read all)")

    text = TODO_INJECTION_HEADER + "\n" + "\n".join(lines)
    if len(text) > max_chars:
        kept: list[str] = []
        for line in lines:
            candidate = TODO_INJECTION_HEADER + "\n" + "\n".join([*kept, line])
            if len(candidate) + len("\n" + TODO_INJECTION_TRUNCATION_MARKER) > max_chars:
                break
            kept.append(line)
        if not kept:
            return None  # the header alone does not fit — nothing to inject
        text = (
            TODO_INJECTION_HEADER
            + "\n"
            + "\n".join(kept)
            + "\n"
            + TODO_INJECTION_TRUNCATION_MARKER
        )
    return text


class SessionState(BaseModel):
    version: int = 1
    approval: ApprovalStateData = Field(default_factory=ApprovalStateData)
    additional_dirs: list[str] = Field(default_factory=list)
    custom_title: str | None = None
    title_generated: bool = False
    title_generate_attempts: int = 0
    # System-prompt stability (cache-04): the wall-clock value rendered into
    # the system prompt, persisted at session creation so every resumed
    # process reuses the identical value (minute precision) instead of a
    # fresh microsecond timestamp — keeping the prompt byte-identical across
    # process boundaries for provider prefix-cache continuity.
    system_prompt_now: str | None = None
    # Archive state (previously in metadata.json)
    wire_mtime: float | None = None
    archived: bool = False
    archived_at: float | None = None
    auto_archive_exempt: bool = False
    # Todo list state
    todos: list[TodoItemState] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    archived_todos: list[TodoItemState] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


_LEGACY_METADATA_FILENAME = "metadata.json"


def _migrate_legacy_metadata(session_dir: Path, state: SessionState) -> str:
    """Migrate fields from legacy metadata.json into SessionState.

    Returns:
        "migrated" - fields were merged into state, caller should save and delete legacy file
        "no_change" - legacy file parsed but no fields needed, caller can delete legacy file
        "skip" - legacy file missing or unreadable, caller should not touch it
    """
    metadata_file = session_dir / _LEGACY_METADATA_FILENAME
    if not metadata_file.exists():
        return "skip"
    try:
        data = orjson.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        # Leave the file intact for future retry — it may be temporarily unreadable
        return "skip"

    changed = False

    # Migrate title fields (only if state has defaults)
    if state.custom_title is None and data.get("title") and data["title"] != "Untitled":
        state.custom_title = data["title"]
        changed = True
    if not state.title_generated and data.get("title_generated"):
        state.title_generated = True
        changed = True
    if state.title_generate_attempts == 0 and data.get("title_generate_attempts", 0) > 0:
        state.title_generate_attempts = data["title_generate_attempts"]
        changed = True

    # Migrate archive fields
    if not state.archived and data.get("archived"):
        state.archived = True
        changed = True
    if state.archived_at is None and data.get("archived_at") is not None:
        state.archived_at = data["archived_at"]
        changed = True
    if not state.auto_archive_exempt and data.get("auto_archive_exempt"):
        state.auto_archive_exempt = True
        changed = True

    # Migrate wire_mtime
    if state.wire_mtime is None and data.get("wire_mtime") is not None:
        state.wire_mtime = data["wire_mtime"]
        changed = True

    return "migrated" if changed else "no_change"


def load_session_state(session_dir: Path) -> SessionState:
    state_file = session_dir / STATE_FILE_NAME
    if not state_file.exists():
        state = SessionState()
    else:
        try:
            with open(state_file, encoding="utf-8") as f:
                state = SessionState.model_validate(orjson.loads(f.read()))
        except (orjson.JSONDecodeError, ValidationError, UnicodeDecodeError):
            logger.warning("Corrupted state file, using defaults: {path}", path=state_file)

            state = SessionState()

    # One-time migration from legacy metadata.json (best-effort)
    migration = _migrate_legacy_metadata(session_dir, state)
    if migration in ("migrated", "no_change"):
        try:
            if migration == "migrated":
                save_session_state(state, session_dir)
            (session_dir / _LEGACY_METADATA_FILENAME).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Failed to persist migration for {path}, will retry next load",
                path=session_dir,
            )

    return state


def save_session_state(state: SessionState, session_dir: Path) -> None:
    state_file = session_dir / STATE_FILE_NAME
    atomic_json_write(state.model_dump(mode="json"), state_file)
