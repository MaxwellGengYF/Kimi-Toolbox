"""Session todo state: read Kimi's ``state.json`` and live ``TodoDisplayBlock``s.

Kimi keeps the root agent's todo tree in ``<session_dir>/state.json`` under
``todos`` (plus ``archived_todos``). The ``todo_write`` / ``todo_update`` tools
persist that file and return a flattened :class:`TodoDisplayBlock` on every
mutation, so the UI has two sources:

* the display block on a tool result — instant, arrives mid-generation;
* ``state.json`` — authoritative, and the only source when a session is
  resumed or after ``/clear`` (which deletes the file).

Subagent todos live in their own ``subagents/<id>/state.json`` and are
deliberately ignored here: the panel mirrors the root agent's plan only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson

TodoStatus = Literal["pending", "in_progress", "done"]

STATE_FILE_NAME = "state.json"

_STATUSES: tuple[TodoStatus, ...] = ("pending", "in_progress", "done")
_MAX_ENTRIES = 500
_MAX_DEPTH = 12


@dataclass(frozen=True, slots=True)
class TodoEntry:
    """One todo, flattened depth-first out of the stored tree."""

    title: str
    status: TodoStatus
    notes: str = ""
    depth: int = 0


@dataclass(frozen=True, slots=True)
class TodoSnapshot:
    """A complete view of a session's todo list at one point in time."""

    entries: tuple[TodoEntry, ...] = ()
    archived: int = 0

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    @property
    def done(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "done")

    @property
    def in_progress(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "in_progress")

    @property
    def pending(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "pending")

    @property
    def all_done(self) -> bool:
        return bool(self.entries) and self.done == self.total

    @property
    def active(self) -> TodoEntry | None:
        """The item the agent says it is working on, else the next pending one."""

        for entry in self.entries:
            if entry.status == "in_progress":
                return entry
        for entry in self.entries:
            if entry.status == "pending":
                return entry
        return None

    def with_archived(self, archived: int) -> TodoSnapshot:
        if archived == self.archived:
            return self
        return TodoSnapshot(entries=self.entries, archived=max(0, archived))


EMPTY_SNAPSHOT = TodoSnapshot()


def _coerce_status(value: object) -> TodoStatus:
    text = str(value or "").strip().lower()
    if text in _STATUSES:
        return text  # type: ignore[return-value]
    if text in {"completed", "complete", "finished"}:
        return "done"
    if text in {"in-progress", "inprogress", "active", "running"}:
        return "in_progress"
    return "pending"


def _coerce_title(value: object) -> str:
    return " ".join(str(value or "").split())


def _coerce_notes(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _item_field(item: object, *names: str) -> object:
    """Read the first present field, accepting dicts and pydantic models alike."""

    for name in names:
        if isinstance(item, dict):
            if name in item:
                return item[name]
            continue
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def flatten_todo_tree(raw: object) -> tuple[TodoEntry, ...]:
    """Flatten a stored todo tree (``state.json`` shape) depth-first.

    Never raises: malformed nodes are skipped, untitled nodes are dropped, and
    the result is bounded so a corrupted state file cannot stall the UI.
    """

    entries: list[TodoEntry] = []

    def walk(items: object, depth: int) -> None:
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            return
        for item in items:
            if len(entries) >= _MAX_ENTRIES:
                return
            title = _coerce_title(_item_field(item, "title", "content"))
            if title:
                entries.append(
                    TodoEntry(
                        title=title,
                        status=_coerce_status(_item_field(item, "status")),
                        notes=_coerce_notes(_item_field(item, "notes")),
                        depth=depth,
                    )
                )
            if depth < _MAX_DEPTH:
                walk(_item_field(item, "children"), depth + 1)

    walk(raw, 0)
    return tuple(entries)


def _flatten_display_items(raw: object) -> tuple[TodoEntry, ...]:
    """Convert ``TodoDisplayBlock.items`` (already flat, with ``depth``)."""

    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    entries: list[TodoEntry] = []
    for item in raw:
        if len(entries) >= _MAX_ENTRIES:
            break
        title = _coerce_title(_item_field(item, "title", "content"))
        if not title:
            continue
        depth = _item_field(item, "depth")
        entries.append(
            TodoEntry(
                title=title,
                status=_coerce_status(_item_field(item, "status")),
                notes=_coerce_notes(_item_field(item, "notes")),
                depth=max(0, min(_MAX_DEPTH, int(depth) if isinstance(depth, int) else 0)),
            )
        )
    return tuple(entries)


def snapshot_from_state_data(data: object) -> TodoSnapshot:
    """Build a snapshot from parsed ``state.json`` content."""

    if not isinstance(data, dict):
        return EMPTY_SNAPSHOT
    archived = data.get("archived_todos")
    return TodoSnapshot(
        entries=flatten_todo_tree(data.get("todos")),
        archived=len(archived) if isinstance(archived, Sequence) else 0,
    )


def snapshot_from_display_block(block: object) -> TodoSnapshot | None:
    """Build a snapshot from a ``TodoDisplayBlock``, or ``None`` if it is not one."""

    if getattr(block, "type", None) != "todo" or not hasattr(block, "items"):
        return None
    return TodoSnapshot(entries=_flatten_display_items(block.items))


def snapshot_from_display_blocks(display: Iterable[object]) -> TodoSnapshot | None:
    """Return the snapshot of the last todo block in ``display``, if any."""

    found: TodoSnapshot | None = None
    for block in display:
        snapshot = snapshot_from_display_block(block)
        if snapshot is not None:
            found = snapshot
    return found


def snapshot_from_wire_message(message: object) -> TodoSnapshot | None:
    """Extract a todo snapshot from a root-agent tool result.

    Returns ``None`` for every message that does not carry a todo display
    block. Subagent traffic arrives wrapped in ``SubagentEvent`` and therefore
    never reaches this branch.
    """

    return_value = getattr(message, "return_value", None)
    if return_value is None:
        return None
    display = getattr(return_value, "display", None)
    if not display:
        return None
    return snapshot_from_display_blocks(display)


def session_state_file(work_dir: Path, session_id: str) -> Path | None:
    """Resolve ``state.json`` for a session using kimi-cli's storage layout."""

    if not session_id:
        return None
    try:
        from kimi_cli.metadata import load_metadata

        from kimix_gui.kimi_workdir import resolve_kimi_work_dir

        kaos_dir = resolve_kimi_work_dir(work_dir)
        work_dir_meta = load_metadata().get_work_dir_meta(kaos_dir)
    except Exception:  # noqa: BLE001 - a missing/broken store just means "no todos"
        return None
    if work_dir_meta is None:
        return None
    return work_dir_meta.sessions_dir / session_id / STATE_FILE_NAME


def read_snapshot_file(state_file: Path | None) -> TodoSnapshot:
    """Read one ``state.json``; a missing or corrupt file yields an empty snapshot."""

    if state_file is None:
        return EMPTY_SNAPSHOT
    try:
        raw = state_file.read_bytes()
    except OSError:
        return EMPTY_SNAPSHOT
    try:
        return snapshot_from_state_data(orjson.loads(raw))
    except orjson.JSONDecodeError:
        return EMPTY_SNAPSHOT


def read_snapshot(work_dir: Path, session_id: str) -> TodoSnapshot | None:
    """Blocking read of a session's persisted todo state.

    Returns ``None`` when the state file cannot even be located (unknown work
    dir or session), which means "no authoritative answer" rather than "no
    todos" — callers should keep whatever they already have. A resolvable but
    missing file yields an empty snapshot, which is what ``/clear`` produces.
    """

    state_file = session_state_file(work_dir, session_id)
    if state_file is None:
        return None
    return read_snapshot_file(state_file)


async def load_snapshot(work_dir: Path, session_id: str) -> TodoSnapshot | None:
    """Read a session's persisted todo state off the event loop."""

    return await asyncio.to_thread(read_snapshot, work_dir, session_id)
