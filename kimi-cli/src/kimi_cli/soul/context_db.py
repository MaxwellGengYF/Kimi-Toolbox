from __future__ import annotations

import orjson
import regex as re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiosqlite
import threading
from kosong.message import Message

from kimi_cli.soul.context_records import ExportedContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHECKPOINT_PATTERN = re.compile(r"^<system>CHECKPOINT \d+</system>$")


def _is_checkpoint_content(content: Any) -> bool:
    """Check if message content contains a synthetic checkpoint marker.

    Handles both string content (``"<system>CHECKPOINT N</system>"``) and
    list content (``[{"type": "text", "text": "..."}, ...]``).
    """
    if isinstance(content, str):
        return bool(_CHECKPOINT_PATTERN.fullmatch(content.strip()))
    if isinstance(content, list):
        return any(_is_checkpoint_part(part) for part in content)
    return False


def _is_checkpoint_part(part: Any) -> bool:
    """Check if a single content part is a synthetic checkpoint marker."""
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str) and _CHECKPOINT_PATTERN.fullmatch(text.strip()):
            return True
    elif isinstance(part, str) and _CHECKPOINT_PATTERN.fullmatch(part.strip()):
        return True
    return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS system_prompt (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    content     TEXT NOT NULL,
    updated_at  REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id            INTEGER NOT NULL,
    message_rowid INTEGER,
    created_at    REAL NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS usage_snapshots (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    token_count INTEGER NOT NULL,
    created_at  REAL NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""


# ---------------------------------------------------------------------------
# ContextDB
# ---------------------------------------------------------------------------


class ContextDB:
    """SQLite-backed storage for conversation context.

    Lifecycle:
        >>> db = ContextDB(db_path)
        >>> await db.initialize()
        >>> ...  # use methods
        >>> await db.close()
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._in_transaction: bool = False
        self._last_message_rowid: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        """Open connection, enable WAL mode, and create tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=-32000")  # 32 MB cache
        await self._conn.execute("PRAGMA temp_store=MEMORY")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def stop_sync(self) -> None:
        """Synchronously stop the aiosqlite worker thread and close SQLite.

        Used by process-exit cleanup paths (e.g. ``Session.__del__`` after a
        ``KeyboardInterrupt``) where awaiting ``close()`` is impossible.  On
        Windows the aiosqlite worker thread keeps ``context.db`` (and its WAL
        files) locked until the underlying connection is closed, which makes
        ``shutil.rmtree`` fail silently; stopping the thread releases the
        handles so the session directory can be deleted.

        The worker thread is joined with a timeout; if it is stuck inside a
        long-running query the connection may stay locked and the caller
        should retry the directory removal.
        """
        conn = self._conn
        if conn is None:
            return
        try:
            conn.stop()
        except Exception:
            # aiosqlite.Connection.stop is best-effort (it even tolerates a
            # missing event loop); never let cleanup raise.
            pass
        thread = getattr(conn, "_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._conn = None

    async def _ensure_open(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.initialize()
        return self._conn  # type: ignore[return-value]

    async def _maybe_commit(self, conn: aiosqlite.Connection) -> None:
        """Commit if not inside an explicit transaction."""
        if not self._in_transaction:
            await conn.commit()

    async def begin_transaction(self) -> None:
        """Begin an explicit transaction for bulk operations."""
        conn = await self._ensure_open()
        await conn.execute("BEGIN")
        self._in_transaction = True

    async def commit_transaction(self) -> None:
        """Commit the current explicit transaction."""
        if self._conn is not None:
            await self._conn.execute("COMMIT")
            self._in_transaction = False

    async def rollback_transaction(self) -> None:
        """Rollback the current explicit transaction."""
        if self._conn is not None:
            try:
                await self._conn.execute("ROLLBACK")
            finally:
                self._in_transaction = False

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    async def get_system_prompt(self) -> str | None:
        conn = await self._ensure_open()
        cursor = await conn.execute("SELECT content FROM system_prompt WHERE id = 1")
        row = await cursor.fetchone()
        await cursor.close()
        return row["content"] if row else None

    async def set_system_prompt(self, content: str) -> None:
        conn = await self._ensure_open()
        await conn.execute(
            "INSERT OR REPLACE INTO system_prompt (id, content, updated_at) VALUES (1, ?, unixepoch())",
            (content,),
        )
        await self._maybe_commit(conn)

    # ------------------------------------------------------------------ #
    # Messages (append + read)
    # ------------------------------------------------------------------ #

    async def append_messages(self, messages: Sequence[Message]) -> None:
        conn = await self._ensure_open()
        params = [
            (msg.role, msg.model_dump_json(exclude_none=True))
            for msg in messages
        ]
        if params:
            await conn.executemany(
                "INSERT INTO messages (role, content) VALUES (?, ?)", params
            )
        await self._maybe_commit(conn)

    async def get_messages(
        self,
        *,
        after_rowid: int = 0,
        limit: int | None = None,
    ) -> list[Message]:
        rows = await self._get_message_rows(after_rowid=after_rowid, limit=limit)
        return [Message.model_validate_json(row["content"]) for row in rows]

    async def get_messages_with_meta(
        self,
        *,
        after_rowid: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Like get_messages() but returns dicts with rowid, role, content, created_at."""
        return await self._get_message_rows(after_rowid=after_rowid, limit=limit)

    async def _get_message_rows(
        self,
        *,
        after_rowid: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Shared helper: returns dict rows with rowid, role, content, created_at."""
        conn = await self._ensure_open()
        query = "SELECT rowid, role, content, created_at FROM messages WHERE rowid > ? ORDER BY rowid"
        params: list[Any] = [after_rowid]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def get_message_count(self) -> int:
        conn = await self._ensure_open()
        cursor = await conn.execute("SELECT COUNT(*) FROM messages")
        row = await cursor.fetchone()
        await cursor.close()
        return row[0]  # type: ignore[index]

    async def has_visible_messages(self) -> bool:
        """Check if there are messages with non-meta roles (user/assistant/tool)."""
        conn = await self._ensure_open()
        cursor = await conn.execute(
            "SELECT 1 FROM messages WHERE role NOT IN ('_system_prompt', '_usage', '_checkpoint') LIMIT 1"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    async def get_last_message_rowid(self) -> int:
        conn = await self._ensure_open()
        cursor = await conn.execute("SELECT MAX(rowid) FROM messages")
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row and row[0] else 0  # type: ignore[index]

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #

    async def create_checkpoint(self, checkpoint_id: int) -> int:
        """Record a checkpoint and return the current max message rowid.

        Uses a single SQL subquery for efficiency instead of SELECT + INSERT.
        """
        conn = await self._ensure_open()
        cursor = await conn.execute(
            "INSERT INTO checkpoints (id, message_rowid) VALUES (?, (SELECT MAX(rowid) FROM messages))",
            (checkpoint_id,),
        )
        await self._maybe_commit(conn)
        # Fetch the actual max rowid that was stored
        cp = await self.get_checkpoint_message_rowid(checkpoint_id)
        return cp or 0

    async def get_latest_checkpoint_id(self) -> int:
        conn = await self._ensure_open()
        cursor = await conn.execute("SELECT COALESCE(MAX(id), -1) FROM checkpoints")
        row = await cursor.fetchone()
        await cursor.close()
        return row[0]  # type: ignore[index]

    async def get_checkpoint_message_rowid(self, checkpoint_id: int) -> int | None:
        conn = await self._ensure_open()
        cursor = await conn.execute(
            "SELECT message_rowid FROM checkpoints WHERE id = ?", (checkpoint_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row["message_rowid"] if row else None

    async def revert_to_checkpoint(self, checkpoint_id: int) -> None:
        """Delete all messages, checkpoints, and usage snapshots after the given checkpoint."""
        message_rowid = await self.get_checkpoint_message_rowid(checkpoint_id)
        if message_rowid is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        conn = await self._ensure_open()
        already_in_tx = self._in_transaction
        if not already_in_tx:
            await conn.execute("BEGIN")
        try:
            await conn.execute("DELETE FROM messages WHERE rowid > ?", (message_rowid,))
            await conn.execute("DELETE FROM checkpoints WHERE id >= ?", (checkpoint_id,))
            # Delete usage snapshots whose rowid exceeds the max that maps to the surviving messages
            await conn.execute(
                "DELETE FROM usage_snapshots WHERE rowid > COALESCE((SELECT MAX(rowid) FROM usage_snapshots WHERE rowid <= ?), 0)",
                (message_rowid,),
            )

            if not already_in_tx:
                await conn.execute("COMMIT")
        except Exception:
            if not already_in_tx:
                await conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ #
    # Usage snapshots
    # ------------------------------------------------------------------ #

    async def record_usage(self, token_count: int) -> None:
        conn = await self._ensure_open()
        await conn.execute(
            "INSERT INTO usage_snapshots (token_count) VALUES (?)",
            (token_count,),
        )
        await self._maybe_commit(conn)

    async def get_latest_usage(self) -> int | None:
        conn = await self._ensure_open()
        cursor = await conn.execute(
            "SELECT token_count FROM usage_snapshots ORDER BY rowid DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row["token_count"] if row else None

    # ------------------------------------------------------------------ #
    # Bulk operations
    # ------------------------------------------------------------------ #

    async def clear(self) -> None:
        conn = await self._ensure_open()
        already_in_tx = self._in_transaction
        if not already_in_tx:
            await conn.execute("BEGIN")
        try:
            await conn.execute("DELETE FROM messages")
            await conn.execute("DELETE FROM system_prompt")
            await conn.execute("DELETE FROM checkpoints")
            await conn.execute("DELETE FROM usage_snapshots")
            if not already_in_tx:
                await conn.execute("COMMIT")
        except Exception:
            if not already_in_tx:
                await conn.execute("ROLLBACK")
            raise

    async def export(self) -> ExportedContext:
        """Export all context data atomically in a transaction."""
        conn = await self._ensure_open()

        result = ExportedContext()

        # Use a transaction to get a consistent snapshot across all tables
        if not self._in_transaction:
            await conn.execute("BEGIN")
        try:
            # system prompt
            cursor = await conn.execute("SELECT content FROM system_prompt WHERE id = 1")
            row = await cursor.fetchone()
            await cursor.close()
            if row:
                result.system_prompt = row["content"]

            # messages
            cursor = await conn.execute("SELECT content FROM messages ORDER BY rowid")
            rows = await cursor.fetchall()
            await cursor.close()
            result.messages = [Message.model_validate_json(row["content"]) for row in rows]

            # checkpoints
            cursor = await conn.execute("SELECT id FROM checkpoints ORDER BY id")
            rows = await cursor.fetchall()
            await cursor.close()
            result.checkpoints = [row["id"] for row in rows]

            # usage snapshots
            cursor = await conn.execute("SELECT token_count FROM usage_snapshots ORDER BY rowid")
            rows = await cursor.fetchall()
            await cursor.close()
            result.usages = [row["token_count"] for row in rows]

            await conn.execute("COMMIT")
        except Exception:
            await conn.execute("ROLLBACK")
            raise

        return result

    # ------------------------------------------------------------------ #
    # Turn boundaries
    # ------------------------------------------------------------------ #

    async def get_messages_up_to_turn(self, turn_index: int) -> list[tuple[str, int]]:
        """Return (json_line, rowid) pairs for all messages up to and including the given turn.

        Turn detection is based on real user messages, excluding synthetic checkpoint
        user entries like ``<system>CHECKPOINT N</system>``.

        Uses a streaming cursor to avoid loading all rows into memory.
        """
        conn = await self._ensure_open()
        cursor = await conn.execute(
            "SELECT rowid, role, content FROM messages ORDER BY rowid"
        )

        result: list[tuple[str, int]] = []
        current_turn = -1

        async for row in cursor:
            role = row["role"]
            content = row["content"]

            # Detect user turn (excluding synthetic checkpoint markers)
            if role == "user":
                # Fast-path: only attempt JSON parsing if content contains checkpoint marker
                if "CHECKPOINT" in content:
                    # Fast sub-string check before JSON parsing
                    if '"<system>CHECKPOINT' in content:
                        try:
                            parsed = orjson.loads(content)
                            # Case 1: content is a plain string
                            if isinstance(parsed, str) and _CHECKPOINT_PATTERN.fullmatch(parsed.strip()):
                                pass  # skip checkpoint
                            # Case 2: content is a list (legacy format — raw content array)
                            elif isinstance(parsed, list):
                                if _is_checkpoint_content(parsed):
                                    pass
                                else:
                                    current_turn += 1
                            # Case 3: content is a full Message dict (SQLite storage format)
                            elif isinstance(parsed, dict):
                                msg_content = parsed.get("content")
                                if _is_checkpoint_content(msg_content):
                                    pass  # skip checkpoint
                                else:
                                    current_turn += 1
                            else:
                                current_turn += 1
                        except (orjson.JSONDecodeError, TypeError):
                            current_turn += 1
                    else:
                        # Contains 'CHECKPOINT' but not '<system>CHECKPOINT' — real user message
                        current_turn += 1
                else:
                    current_turn += 1

                if current_turn > turn_index:
                    break

            if current_turn <= turn_index:
                result.append((content, row["rowid"]))

        await cursor.close()
        return result

    # ------------------------------------------------------------------ #
    # Migration helpers
    # ------------------------------------------------------------------ #

    async def import_jsonl_line(self, line_json: dict[str, Any]) -> None:
        """Import a single parsed JSONL line into the appropriate table.

        Used during JSONL → SQLite migration.
        Tracks the last inserted message rowid so that checkpoints can
        reference the correct message boundary.
        """
        conn = await self._ensure_open()
        role = line_json.get("role")

        if role == "_system_prompt":
            content = line_json.get("content", "")
            await conn.execute(
                "INSERT OR REPLACE INTO system_prompt (id, content, updated_at) VALUES (1, ?, unixepoch())",
                (content,),
            )
        elif role == "_usage":
            token_count = line_json.get("token_count", 0)
            await conn.execute(
                "INSERT INTO usage_snapshots (token_count) VALUES (?)",
                (token_count,),
            )
        elif role == "_checkpoint":
            cpid = line_json.get("id", 0)
            await conn.execute(
                "INSERT INTO checkpoints (id, message_rowid) VALUES (?, ?)",
                (cpid, self._last_message_rowid),
            )
        else:
            cursor = await conn.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, orjson.dumps(line_json).decode()),
            )
            self._last_message_rowid = cursor.lastrowid or self._last_message_rowid

    async def finalize_migration(self) -> None:
        """After all JSONL lines are imported, update checkpoint message_rowid references.

        Since checkpoints now get correct message_rowid during import_jsonl_line,
        this is a no-op retained for backward compatibility.
        """
        # No longer needed — message_rowid is set correctly during import

    async def fix_checkpoint_message_rowids(self) -> None:
        """Fix checkpoint message_rowid to point to actual message boundaries.

        Since import_jsonl_line now tracks the last message rowid during import,
        this method only fixes checkpoints with message_rowid=0 (from pre-fix
        migrations or edge cases). Uses a single UPDATE for efficiency.
        """
        conn = await self._ensure_open()
        # For any checkpoints still set to 0, assign message_rowid = id
        # as a reasonable fallback (checkpoint ids and message rowids
        # are both monotonically increasing during sequential import).
        await conn.execute(
            "UPDATE checkpoints SET message_rowid = id WHERE message_rowid = 0"
        )
        await self._maybe_commit(conn)
