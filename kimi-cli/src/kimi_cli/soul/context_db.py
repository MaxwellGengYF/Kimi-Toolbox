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
from kimi_cli.soul.fts5_search import (
    contains_cjk,
    escape_like,
    extract_text_from_content,
    has_lone_cjk_run,
    quote_fts_tokens,
    sanitize_fts5_query,
    trigram_eligible_tokens,
)

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
    content_text TEXT,
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

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""

# External-content FTS5 tables over messages.content_text (Hermes FTS_SQL /
# FTS_TRIGRAM_SQL pattern). Triggers are created separately (see
# _FTS_TRIGGER_SQL) so the content_text backfill can install them before the
# chunked rebuild; every trigger is gated on the rebuild markers so a
# never-indexed row never receives an external-content 'delete' command
# (which would corrupt the index — exactly the failure Hermes' rebuild-gated
# triggers prevent).
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content_text,
    content='messages',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content_text,
    content='messages',
    content_rowid='rowid',
    tokenize='trigram'
);
"""

# Shared rebuild gate predicate (ported verbatim from
# hermes_state_common.py:416-453, adapted to the ``meta`` table and
# ``messages.rowid``): a row is indexed iff  rowid <= fts_rebuild_progress
# (already backfilled)  OR  rowid > fts_rebuild_high_water (inserted after the
# rebuild started; AUTOINCREMENT rowids are always greater).  Rows in
# (progress, high_water] are not indexed yet, so triggers must stay silent for
# them.  When no rebuild is pending both keys are absent and COALESCE turns the
# predicate into a tautology (id > -1 OR id <= -1), i.e. normal operation.
_FTS_REBUILD_HIGH_WATER_KEY = "fts_rebuild_high_water"
_FTS_REBUILD_PROGRESS_KEY = "fts_rebuild_progress"
_FTS_STALE_KEY = "fts_stale"
_FTS_REBUILD_KEYS = (
    _FTS_REBUILD_HIGH_WATER_KEY,
    _FTS_REBUILD_PROGRESS_KEY,
    _FTS_STALE_KEY,
)

_FTS_REBUILD_GATE = (
    "(new.rowid > COALESCE((SELECT CAST(value AS INTEGER) FROM meta "
    f"WHERE key = '{_FTS_REBUILD_HIGH_WATER_KEY}'), -1) "
    "OR new.rowid <= COALESCE((SELECT CAST(value AS INTEGER) FROM meta "
    f"WHERE key = '{_FTS_REBUILD_PROGRESS_KEY}'), -1))"
)
_FTS_REBUILD_GATE_OLD = _FTS_REBUILD_GATE.replace("new.rowid", "old.rowid")

_FTS_TRIGGER_SQL = f"""
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN {_FTS_REBUILD_GATE}
BEGIN
    INSERT INTO messages_fts(rowid, content_text) VALUES (new.rowid, new.content_text);
    INSERT INTO messages_fts_trigram(rowid, content_text) VALUES (new.rowid, new.content_text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN {_FTS_REBUILD_GATE_OLD}
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.rowid, old.content_text);
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content_text) VALUES ('delete', old.rowid, old.content_text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF content_text ON messages
WHEN old.content_text IS NOT new.content_text
   AND {_FTS_REBUILD_GATE_OLD}
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.rowid, old.content_text);
    INSERT INTO messages_fts(rowid, content_text) VALUES (new.rowid, new.content_text);
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content_text) VALUES ('delete', old.rowid, old.content_text);
    INSERT INTO messages_fts_trigram(rowid, content_text) VALUES (new.rowid, new.content_text);
END;
"""

_FTS_TRIGGER_NAMES = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)

_FTS_TABLE_NAMES = ("messages_fts", "messages_fts_trigram")

# Chunk size for the Hermes-style chunked FTS backfill.
_FTS_REBUILD_CHUNK_ROWS = 500
# Bound for the incremental ``merge`` command (max pages merged per call).
_FTS_MERGE_MAX_PAGES = 200


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
        await self._conn.executescript(_FTS_SQL)
        await self._migrate_and_backfill_fts()
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
            (msg.role, msg.model_dump_json(exclude_none=True), extract_text_from_content(msg.content))
            for msg in messages
        ]
        if params:
            await conn.executemany(
                "INSERT INTO messages (role, content, content_text) VALUES (?, ?, ?)", params
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
    # Full-text search (Phase B)
    # ------------------------------------------------------------------ #

    async def _migrate_and_backfill_fts(self) -> None:
        """Ensure ``messages.content_text`` exists/populated and the FTS
        indexes are in sync with ``messages`` (Hermes marker pattern).

        Runs at ``initialize()`` time on a single connection.  When the
        docsize shadow counts disagree with ``messages`` (pre-FTS migration,
        interrupted rebuild, corruption), a chunked backfill runs with
        ``fts_rebuild_high_water`` / ``fts_rebuild_progress`` markers in the
        ``meta`` table; the triggers are gated on those markers so rows
        written during the rebuild are still indexed live and never-indexed
        rows never receive external-content 'delete' commands.
        """
        conn = self._conn
        assert conn is not None

        # 1. Ensure content_text column on pre-FTS databases.
        cursor = await conn.execute("PRAGMA table_info(messages)")
        cols = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "content_text" not in cols:
            await conn.execute("ALTER TABLE messages ADD COLUMN content_text TEXT")

        # 2. Always (re)install the gated sync triggers.
        await conn.executescript(_FTS_TRIGGER_SQL)

        # 3. Rebuild when the docsize shadow counts disagree with messages.
        msg_count = await self._count_rows(conn, "messages")
        fts_count = await self._count_rows(conn, "messages_fts_docsize")
        trigram_count = await self._count_rows(conn, "messages_fts_trigram_docsize")
        if msg_count != fts_count or msg_count != trigram_count:
            await self._fts_rebuild_start(conn)
            await self._backfill_content_text(conn)
            await self._fts_rebuild_chunked(conn)
            await self._fts_rebuild_finish(conn)

    async def _count_rows(self, conn: aiosqlite.Connection, table: str) -> int:
        try:
            cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            await cursor.close()
            return int(row[0]) if row else 0  # type: ignore[index]
        except Exception:
            return 0

    async def _get_meta(self, conn: aiosqlite.Connection, key: str) -> str | None:
        cursor = await conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = await cursor.fetchone()
        await cursor.close()
        return row["value"] if row else None

    async def _set_meta(self, conn: aiosqlite.Connection, key: str, value: str) -> None:
        await conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    async def _reset_fts_index_to_empty(self, conn: aiosqlite.Connection) -> None:
        """O(1) truncate for external-content FTS5 tables."""
        for table in _FTS_TABLE_NAMES:
            await conn.execute(f"INSERT INTO {table}({table}) VALUES('delete-all')")

    async def _fts_rebuild_start(self, conn: aiosqlite.Connection) -> None:
        """Clean FTS surface and seed the rebuild markers."""
        await self._reset_fts_index_to_empty(conn)
        cursor = await conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM messages")
        row = await cursor.fetchone()
        await cursor.close()
        high_water = int(row[0]) if row else 0
        await self._set_meta(conn, _FTS_REBUILD_HIGH_WATER_KEY, str(high_water))
        await self._set_meta(conn, _FTS_REBUILD_PROGRESS_KEY, "0")
        # Publish markers before the content_text backfill so the gated
        # triggers stay silent for old rows while new rows stay indexed.
        await conn.execute("COMMIT")

    async def _backfill_content_text(self, conn: aiosqlite.Connection) -> None:
        """Fill NULL/empty content_text from the stored JSON blob.

        The gated update trigger does not fire for rows <= high_water, so a
        never-indexed row never receives an external-content 'delete'.
        """
        cursor = await conn.execute(
            "SELECT rowid, content FROM messages "
            "WHERE content_text IS NULL OR content_text = ''"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        if rows:
            updates: list[tuple[str, int]] = []
            for row in rows:
                content = row["content"]
                text = ""
                try:
                    data = orjson.loads(content)
                    if isinstance(data, dict):
                        text = extract_text_from_content(data.get("content"))
                    else:
                        text = extract_text_from_content(data)
                except Exception:
                    text = ""
                updates.append((text, row["rowid"]))
            await conn.executemany(
                "UPDATE messages SET content_text = ? WHERE rowid = ?", updates
            )
        await conn.execute("COMMIT")

    async def _fts_rebuild_chunked(self, conn: aiosqlite.Connection) -> None:
        """Backfill the FTS indexes chunk-by-chunk, crash-atomically.

        Each chunk (rows + progress marker) commits in one transaction, so an
        interrupted rebuild resumes from the last published progress.
        """
        high_water_raw = await self._get_meta(conn, _FTS_REBUILD_HIGH_WATER_KEY)
        if high_water_raw is None:
            return
        high_water = int(high_water_raw)
        progress_raw = await self._get_meta(conn, _FTS_REBUILD_PROGRESS_KEY)
        progress = int(progress_raw or "0")
        chunk = _FTS_REBUILD_CHUNK_ROWS
        while progress < high_water:
            upper = min(progress + chunk, high_water)
            await conn.execute(
                "INSERT INTO messages_fts(rowid, content_text) "
                "SELECT rowid, content_text FROM messages "
                "WHERE rowid > ? AND rowid <= ?",
                (progress, upper),
            )
            await conn.execute(
                "INSERT INTO messages_fts_trigram(rowid, content_text) "
                "SELECT rowid, content_text FROM messages "
                "WHERE rowid > ? AND rowid <= ?",
                (progress, upper),
            )
            await self._set_meta(conn, _FTS_REBUILD_PROGRESS_KEY, str(upper))
            await conn.execute("COMMIT")
            progress = upper

    async def _fts_rebuild_finish(self, conn: aiosqlite.Connection) -> None:
        """Boundary sweep around high_water, then clear the markers."""
        high_water_raw = await self._get_meta(conn, _FTS_REBUILD_HIGH_WATER_KEY)
        if high_water_raw is not None:
            high_water = int(high_water_raw)
            lo, hi = high_water - _FTS_REBUILD_CHUNK_ROWS, high_water + _FTS_REBUILD_CHUNK_ROWS
            for table in _FTS_TABLE_NAMES:
                await conn.execute(
                    f"INSERT INTO {table}(rowid, content_text) "
                    "SELECT m.rowid, m.content_text FROM messages m "
                    f"WHERE m.rowid > ? AND m.rowid <= ? "
                    f"AND NOT EXISTS (SELECT 1 FROM {table}_docsize d WHERE d.id = m.rowid)",
                    (lo, hi),
                )
        await conn.execute(
            "DELETE FROM meta WHERE key IN (?, ?)",
            (_FTS_REBUILD_HIGH_WATER_KEY, _FTS_REBUILD_PROGRESS_KEY),
        )
        await conn.execute("COMMIT")

    async def rebuild_fts(self) -> None:
        """Drop + recreate the FTS tables and backfill from ``messages``.

        Also clears the ``fts_stale`` breadcrumb so FTS serving resumes.
        """
        conn = await self._ensure_open()
        for trigger in _FTS_TRIGGER_NAMES:
            await conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        for table in _FTS_TABLE_NAMES:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.executescript(_FTS_SQL)
        await conn.executescript(_FTS_TRIGGER_SQL)
        await conn.execute("DELETE FROM meta WHERE key = ?", (_FTS_STALE_KEY,))
        await self._backfill_content_text(conn)
        await self._fts_rebuild_start(conn)
        await self._fts_rebuild_chunked(conn)
        await self._fts_rebuild_finish(conn)
        await conn.commit()

    async def fts_rebuild_status(self) -> dict[str, Any] | None:
        """Return rebuild progress, or None when no rebuild is pending.

        Mirrors Hermes' ``fts_rebuild_status`` surface (plan §4 #2).
        """
        conn = await self._ensure_open()
        high_water_raw = await self._get_meta(conn, _FTS_REBUILD_HIGH_WATER_KEY)
        if high_water_raw is None:
            return None
        high_water = int(high_water_raw)
        progress_raw = await self._get_meta(conn, _FTS_REBUILD_PROGRESS_KEY)
        progress = int(progress_raw or "0")
        if high_water <= 0:
            return None
        percent = min(100, int(100 * progress / high_water))
        return {
            "pending": True,
            "total": high_water,
            "indexed": progress,
            "percent": percent,
        }

    async def search_messages(
        self,
        query: str,
        *,
        role: str | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 full-text search over message content.

        Returns ``{rowid, role, content, content_text, created_at, snippet,
        score}``.  Routing: no CJK → unicode61 FTS; CJK (>=3-char tokens) →
        trigram; short/lone CJK runs → LIKE substring scan.  Any FTS error
        degrades to the LIKE fallback (never raises to callers).
        """
        conn = await self._ensure_open()
        q = sanitize_fts5_query(query)
        if not q:
            return []

        # Stale-index breadcrumb (plan §4 #3 / §6): a previously-corrupted FTS
        # table serves LIKE results until rebuild_fts() clears the marker.
        try:
            stale = await self._get_meta(conn, _FTS_STALE_KEY)
            if stale is not None:
                raw_query = q.strip('"').strip()
                return await self._search_messages_like(
                    conn, raw_query, "", [], limit, offset
                )
        except Exception:
            pass

        sort_norm = sort.strip().lower() if isinstance(sort, str) else None
        if sort_norm not in ("newest", "oldest"):
            sort_norm = None
        if sort_norm == "newest":
            order_by = "ORDER BY m.created_at DESC, rank, m.rowid DESC"
        elif sort_norm == "oldest":
            order_by = "ORDER BY m.created_at ASC, rank, m.rowid ASC"
        else:
            order_by = "ORDER BY rank, m.rowid"

        role_clause = ""
        role_params: list[str] = []
        if role:
            role_clause = "AND m.role = ?"
            role_params = [role]

        try:
            if contains_cjk(q):
                raw_query = q.strip('"').strip()
                if trigram_eligible_tokens(q) and not has_lone_cjk_run(raw_query):
                    trigram_query = quote_fts_tokens(raw_query)
                    sql = f"""
                        SELECT m.rowid, m.role, m.content, m.content_text, m.created_at,
                               snippet(messages_fts_trigram, -1, '>>>', '<<<', '...', 40) AS snippet,
                               -bm25(messages_fts_trigram) AS score
                        FROM messages_fts_trigram
                        JOIN messages m ON m.rowid = messages_fts_trigram.rowid
                        WHERE messages_fts_trigram MATCH ? {role_clause}
                        {order_by}
                        LIMIT ? OFFSET ?
                    """
                    return await self._run_search(
                        conn, sql, [trigram_query, *role_params, limit, offset]
                    )
                return await self._search_messages_like(conn, raw_query, role_clause, role_params, limit, offset)

            sql = f"""
                SELECT m.rowid, m.role, m.content, m.content_text, m.created_at,
                       snippet(messages_fts, -1, '>>>', '<<<', '...', 40) AS snippet,
                       -bm25(messages_fts) AS score
                FROM messages_fts
                JOIN messages m ON m.rowid = messages_fts.rowid
                WHERE messages_fts MATCH ? {role_clause}
                {order_by}
                LIMIT ? OFFSET ?
            """
            return await self._run_search(conn, sql, [q, *role_params, limit, offset])
        except Exception:
            # Any FTS failure (syntax, corrupt index, unavailable tokenizer)
            # degrades to the LIKE substring path over the canonical table and
            # sets the stale breadcrumb so later searches skip straight to LIKE.
            try:
                await self._set_meta(conn, _FTS_STALE_KEY, "1")
                await conn.commit()
            except Exception:
                pass
            raw_query = q.strip('"').strip()
            return await self._search_messages_like(conn, raw_query, role_clause, role_params, limit, offset)

    async def _run_search(
        self,
        conn: aiosqlite.Connection,
        sql: str,
        params: list[Any],
    ) -> list[dict[str, Any]]:
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def _search_messages_like(
        self,
        conn: aiosqlite.Connection,
        raw_query: str,
        role_clause: str,
        role_params: list[str],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        tokens = [
            t for t in raw_query.split() if t.upper() not in {"AND", "OR", "NOT"}
        ] or [raw_query]
        clauses: list[str] = []
        like_params: list[str] = []
        for tok in tokens:
            esc = escape_like(tok)
            clauses.append("m.content_text LIKE ? ESCAPE '\\'")
            like_params.append(f"%{esc}%")
        where = " OR ".join(clauses)
        sql = f"""
            SELECT m.rowid, m.role, m.content, m.content_text, m.created_at,
                   substr(m.content_text,
                          max(1, instr(m.content_text, ?) - 40),
                          120) AS snippet,
                   0.0 AS score
            FROM messages m
            WHERE ({where}) {role_clause}
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT ? OFFSET ?
        """
        params: list[Any] = [tokens[0], *like_params, *role_params, limit, offset]
        return await self._run_search(conn, sql, params)

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
            await conn.execute("DELETE FROM meta WHERE key IN (?, ?, ?)", _FTS_REBUILD_KEYS)
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
            content = line_json.get("content", "")
            cursor = await conn.execute(
                "INSERT INTO messages (role, content, content_text) VALUES (?, ?, ?)",
                (role, orjson.dumps(line_json).decode(), extract_text_from_content(content)),
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
