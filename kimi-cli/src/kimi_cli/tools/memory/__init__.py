"""Durable agent-managed memory tool.

Implements the "disk is the source of truth" pattern for long-running agents:
facts written here survive context compaction, pruning, and session restarts,
because they live in files under the session directory rather than in the
(volatile) LLM context window.

The search index is rebuilt from disk on every ``search`` call, so recall
works regardless of what happened to the context window in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import regex as re
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field, field_validator

from kimi_cli import logger
from kimi_cli.soul.agent import Runtime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_DIR_NAME = "memory"
"""Directory (inside the session dir) holding durable memory files."""

DEFAULT_TOPIC = "memory"
"""Default topic (file name) when the caller does not specify one."""

PRE_COMPACT_FLUSH_TOPIC = "pre_compact_state"
"""Topic used by the automatic pre-compaction state flush."""

_MAX_TOPICS = 64
"""Maximum number of distinct memory files per session."""

_MAX_FILE_BYTES = 262_144
"""Maximum size of a single memory file (256 KiB)."""

_MAX_WRITE_BYTES = 65_536
"""Maximum size of a single write/append payload (64 KiB)."""

_MAX_READ_CHARS = 32_000
"""Maximum characters returned by a single ``read`` call."""

_MAX_SEARCH_RESULTS_CAP = 20
"""Hard cap on search results regardless of ``max_results``."""

_SEARCH_SNIPPET_RADIUS = 80
"""Characters of context on each side of a search match."""

_TOPIC_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_QUERY_TERM_RE = re.compile(r"[\w-]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Path helpers (shared with the soul for pre-compaction flush / restore)
# ---------------------------------------------------------------------------


def sanitize_topic(topic: str) -> str:
    """Normalize a topic into a safe file stem (no path traversal)."""
    cleaned = _TOPIC_SANITIZE_RE.sub("_", topic.strip()).strip("_").lower()
    return cleaned[:64] or DEFAULT_TOPIC


def memory_dir_for_session(session_dir: Path) -> Path:
    """Return the durable memory directory for a session."""
    return session_dir / MEMORY_DIR_NAME


def _topic_path(memory_dir: Path, topic: str) -> Path:
    return memory_dir / f"{sanitize_topic(topic)}.md"


def list_memory_files(memory_dir: Path) -> list[Path]:
    """List memory files (newest first). Missing dir yields an empty list."""
    try:
        files = [p for p in memory_dir.iterdir() if p.is_file() and p.suffix == ".md"]
    except OSError:
        return []
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# Pre-compaction flush / post-compaction restore helpers
# ---------------------------------------------------------------------------


def flush_pre_compact_state(
    session_dir: Path,
    *,
    trigger_reason: str,
    context_tokens: int,
    max_context_tokens: int,
    unfinished_todos: list[tuple[str, str]],
    extra_notes: str | None = None,
) -> Path | None:
    """Write a durable snapshot of agent state before compaction fires.

    This is the "pre-compaction flush": whatever the summary destroys can be
    recovered from disk. Returns the written path, or ``None`` when there is
    nothing worth persisting.
    """
    if not unfinished_todos and not extra_notes:
        return None

    import pendulum

    lines: list[str] = [
        "# Pre-compaction state flush",
        "",
        f"- Flushed at: {pendulum.now().to_iso8601_string()}",
        f"- Trigger: {trigger_reason}",
        f"- Context at flush: {context_tokens}/{max_context_tokens} tokens",
        "",
    ]
    if unfinished_todos:
        lines.append("## Unfinished TodoList tasks")
        lines.append("")
        for status, title in unfinished_todos:
            lines.append(f"- [{status}] {title}")
        lines.append("")
    if extra_notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(extra_notes)
        lines.append("")

    memory_dir = memory_dir_for_session(session_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{PRE_COMPACT_FLUSH_TOPIC}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_memory_restore_text(session_dir: Path, *, max_excerpt_chars: int = 600) -> str | None:
    """Build the post-compaction "memory pointer" text.

    After compaction the model has lost its working context; this message
    re-surfaces the durable memory directory at the *end* of the context,
    where attention is strongest. Returns ``None`` when no memory exists.
    """
    memory_dir = memory_dir_for_session(session_dir)
    files = list_memory_files(memory_dir)
    if not files:
        return None

    listing = ", ".join(f"{p.name} ({p.stat().st_size} bytes)" for p in files[:10])
    parts: list[str] = [
        f"Durable memory files survive compaction at `{memory_dir}`: {listing}.",
        "Use the `Memory` tool (action='read'/'search') to recall anything you need.",
    ]

    flush_path = memory_dir / f"{PRE_COMPACT_FLUSH_TOPIC}.md"
    if flush_path.is_file():
        try:
            excerpt = flush_path.read_text(encoding="utf-8")[:max_excerpt_chars]
        except OSError:
            excerpt = ""
        if excerpt.strip():
            parts.append(f"Latest pre-compaction flush:\n{excerpt}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SearchHit:
    topic: str
    line_no: int
    snippet: str
    score: int


def _search_files(memory_dir: Path, query: str, max_results: int) -> list[_SearchHit]:
    """Keyword search over all memory files, rebuilt from disk every call."""
    terms = [t.lower() for t in _QUERY_TERM_RE.findall(query) if t.strip()]
    if not terms:
        return []

    hits: list[_SearchHit] = []
    for path in list_memory_files(memory_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        topic = path.stem
        lowered_lines = text.lower().splitlines()
        raw_lines = text.splitlines()
        for idx, lowered in enumerate(lowered_lines):
            score = sum(lowered.count(term) for term in terms)
            if score <= 0:
                continue
            line = raw_lines[idx].strip()
            if len(line) > 2 * _SEARCH_SNIPPET_RADIUS + 3:
                # center the snippet on the first term occurrence
                pos = min(
                    (lowered.find(t) for t in terms if lowered.find(t) >= 0),
                    default=0,
                )
                start = max(0, pos - _SEARCH_SNIPPET_RADIUS)
                end = min(len(line), pos + _SEARCH_SNIPPET_RADIUS)
                line = ("…" if start else "") + line[start:end] + ("…" if end < len(raw_lines[idx].strip()) else "")
            hits.append(_SearchHit(topic=topic, line_no=idx + 1, snippet=line, score=score))

    hits.sort(key=lambda h: (-h.score, h.topic, h.line_no))
    return hits[: min(max_results, _MAX_SEARCH_RESULTS_CAP)]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class Params(BaseModel):
    action: Literal["write", "append", "read", "list", "search"] = Field(
        default="read",
        description=(
            "'write': overwrite a memory topic with `content`. "
            "'append': add `content` to the end of a topic. "
            "'read': read a topic (default). "
            "'list': list all memory topics. "
            "'search': keyword-search all memory topics with `query`."
        ),
    )
    topic: str = Field(
        default=DEFAULT_TOPIC,
        description=(
            "Memory topic name (stored as `<topic>.md` in the session memory dir). "
            "Sanitized to `[a-z0-9_-]`; defaults to 'memory'."
        ),
    )
    content: str | None = Field(
        default=None,
        description="Content for 'write'/'append' actions.",
    )
    query: str = Field(
        default="",
        description="Search query for the 'search' action (keywords or phrase).",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=_MAX_SEARCH_RESULTS_CAP,
        description="Maximum number of search hits to return.",
    )

    @field_validator("topic", mode="before")
    @classmethod
    def _validate_topic(cls, v: object) -> str:
        if v is None:
            return DEFAULT_TOPIC
        return sanitize_topic(str(v))


class Memory(CallableTool2[Params]):
    name: str = "Memory"
    description: str = (
        "Durable long-term memory that survives context compaction. "
        "The context window is volatile working memory — anything important "
        "(decisions, file paths, user preferences, invariants, debugging findings) "
        "MUST be written here, or it may be lost when context is compacted.\n"
        "Memory lives in Markdown files under the session directory, one file per topic. "
        "'search' rebuilds its index from disk on every call, so recall works "
        "no matter what happened to the conversation context.\n"
        "Use 'write'/'append' to persist facts, 'read' to recall a topic, "
        "'search' to find facts across all topics, 'list' to see what you have stored."
    )
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._memory_dir = memory_dir_for_session(runtime.session.dir)

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            if params.action == "write":
                return self._write(params.topic, params.content, append=False)
            if params.action == "append":
                return self._write(params.topic, params.content, append=True)
            if params.action == "read":
                return self._read(params.topic)
            if params.action == "list":
                return self._list()
            return self._search(params.query, params.max_results)
        except OSError as exc:
            logger.warning("Memory tool I/O failure: {error}", error=exc)
            return ToolError(
                message=f"Memory operation failed: {exc}",
                brief="Memory I/O error",
            )

    # ---- actions -----------------------------------------------------------

    def _write(self, topic: str, content: str | None, *, append: bool) -> ToolReturnValue:
        if content is None or not content.strip():
            return ToolError(
                message=f"'{ 'append' if append else 'write' }' requires non-empty `content`.",
                brief="Missing content",
            )
        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_WRITE_BYTES:
            return ToolError(
                message=(
                    f"Content is {len(encoded)} bytes, exceeding the per-write limit "
                    f"of {_MAX_WRITE_BYTES} bytes. Split it into smaller writes."
                ),
                brief="Content too large",
            )

        self._memory_dir.mkdir(parents=True, exist_ok=True)
        path = _topic_path(self._memory_dir, topic)
        if not path.exists() and len(list_memory_files(self._memory_dir)) >= _MAX_TOPICS:
            return ToolError(
                message=f"Memory topic limit reached ({_MAX_TOPICS}). Reuse or consolidate topics.",
                brief="Too many topics",
            )

        existing = path.read_text(encoding="utf-8") if append and path.exists() else ""
        new_text = f"{existing.rstrip()}\n\n{content.strip()}\n" if existing else f"{content.strip()}\n"
        if len(new_text.encode("utf-8")) > _MAX_FILE_BYTES:
            return ToolError(
                message=(
                    f"Memory topic '{path.stem}' would exceed the per-file limit of "
                    f"{_MAX_FILE_BYTES} bytes. Consolidate it into a new topic."
                ),
                brief="Topic too large",
            )
        path.write_text(new_text, encoding="utf-8")

        verb = "Appended to" if append else "Wrote"
        return ToolOk(
            output=f"{verb} memory topic '{path.stem}' ({path}).",
            message=f"{verb} topic '{path.stem}'",
        )

    def _read(self, topic: str) -> ToolReturnValue:
        path = _topic_path(self._memory_dir, topic)
        if not path.is_file():
            return ToolError(
                message=f"No memory topic named '{path.stem}'. Use action='list' to see available topics.",
                brief="Topic not found",
            )
        text = path.read_text(encoding="utf-8")
        truncated = False
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS]
            truncated = True
        output = f"Memory topic '{path.stem}':\n\n{text}"
        if truncated:
            output += f"\n\n[Truncated at {_MAX_READ_CHARS} chars; use 'search' to find specific facts.]"
        return ToolOk(output=output, message=f"Read topic '{path.stem}'")

    def _list(self) -> ToolReturnValue:
        files = list_memory_files(self._memory_dir)
        if not files:
            return ToolOk(
                output="No memory topics yet. Use action='write' to persist important facts.",
                message="No memory topics",
            )
        lines = [f"{len(files)} memory topic(s) in {self._memory_dir}:"]
        for p in files:
            lines.append(f"- {p.stem} ({p.stat().st_size} bytes)")
        return ToolOk(output="\n".join(lines), message=f"{len(files)} topic(s)")

    def _search(self, query: str, max_results: int) -> ToolReturnValue:
        if not query.strip():
            return ToolError(
                message="'search' requires a non-empty `query`.",
                brief="Missing query",
            )
        hits = _search_files(self._memory_dir, query, max_results)
        if not hits:
            return ToolOk(
                output=f"No memory entries match {query!r}.",
                message="No results",
            )
        lines = [f"{len(hits)} memory hit(s) for {query!r}:"]
        for hit in hits:
            lines.append(f"- [{hit.topic}:{hit.line_no}] {hit.snippet}")
        return ToolOk(output="\n".join(lines), message=f"{len(hits)} hit(s)")
