"""HistoryIndex: durable FTS5-backed full-text index over conversation turns.

The primary backend is a per-session SQLite ``history.db`` opened through
``apsw`` (synchronous, bundles a recent SQLite with FTS5 + trigram support).
The public API is unchanged from the old in-memory BM25 implementation so
``kimisoul.py``, the ``retrieve`` tool, ``context_prune``, and tests keep
working:

- ``index_messages`` batches turns into ``history.turns``; FTS triggers keep
  the unicode61 + trigram virtual tables in sync.
- ``search`` routes Latin queries to ``turns_fts``, CJK (>=3-char tokens) to
  ``turns_fts_trigram``, and short/lone CJK runs to a LIKE substring scan.
- No 500-turn cap: the durable store holds the whole session history.
- ``persist_path`` mode (legacy JSON + in-memory BM25) is retained for
  backward compatibility and as the FTS5-unavailable fallback.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import apsw
import orjson

from kosong.message import Message

from kimi_cli.soul import fts5_search
from kimi_cli.soul.fts5_search import (
    contains_cjk,
    escape_like,
    has_lone_cjk_run,
    quote_fts_tokens,
    sanitize_fts5_query,
    trigram_eligible_tokens,
)
from kimi_cli.wire.types import TextPart

# Legacy in-memory cap (kept for the old JSON backend and the bounded
# ``_turns`` compat property). The FTS5 backend indexes everything.
_MAX_TURNS: int = 500
# Bound for the ``_turns`` compat property read in FTS5 mode — kimisoul only
# uses it to find the last few non-compacted turn ids, so a bounded window is
# plenty and keeps the property O(1)-ish on huge histories.
_MAX_TURNS_COMPAT: int = 5000

# Phase C (plan §4): stale-index breadcrumb + incremental merge bounds.
_FTS_STALE_KEY = "fts_stale"
_FTS_MERGE_INTERVAL = 500  # run a bounded merge after this many writes
_FTS_MERGE_MAX_PAGES = 200  # max FTS5 b-tree pages merged per call

_FTS_TRIGGER_NAMES = (
    "turns_fts_insert",
    "turns_fts_delete",
    "turns_fts_update",
    "turns_fts_trigram_insert",
    "turns_fts_trigram_delete",
    "turns_fts_trigram_update",
)

_SCHEMA_STATEMENTS = [
    """
CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp REAL NOT NULL,
    is_compacted INTEGER NOT NULL DEFAULT 0
)
""",
    """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
""",
    """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    text,
    content='turns',
    content_rowid='turn_id',
    tokenize='unicode61'
)
""",
    """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts_trigram USING fts5(
    text,
    content='turns',
    content_rowid='turn_id',
    tokenize='trigram'
)
""",
    """
CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);
    INSERT INTO turns_fts_trigram(rowid, text) VALUES (new.turn_id, new.text);
END;
""",
    """
CREATE TRIGGER IF NOT EXISTS turns_fts_delete AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.turn_id, old.text);
    INSERT INTO turns_fts_trigram(turns_fts_trigram, rowid, text) VALUES ('delete', old.turn_id, old.text);
END;
""",
    """
CREATE TRIGGER IF NOT EXISTS turns_fts_update AFTER UPDATE OF text ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.turn_id, old.text);
    INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);
    INSERT INTO turns_fts_trigram(turns_fts_trigram, rowid, text) VALUES ('delete', old.turn_id, old.text);
    INSERT INTO turns_fts_trigram(rowid, text) VALUES (new.turn_id, new.text);
END;
""",
]



class HistoryIndex:
    """Index over conversation turns, backed by SQLite FTS5 (or legacy BM25).

    Each turn (user / assistant / tool message) is one row in ``turns`` with
    an extracted plain-text ``text`` column.  FTS triggers keep the unicode61
    and trigram virtual tables in sync so search never needs a full rebuild.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        *,
        db_path: Path | None = None,
        legacy_json_path: Path | None = None,
    ) -> None:
        self._persist_path = persist_path
        self._db_path = Path(db_path) if db_path is not None else None
        self._legacy_json_path = (
            Path(legacy_json_path) if legacy_json_path is not None else None
        )
        self._conn: apsw.Connection | None = None
        self._in_transaction: bool = False
        self._fts_stale: bool = False
        self._writes_since_merge: int = 0

        # Legacy in-memory BM25 state (always initialized; used only when no
        # db_path is configured or FTS5 is unavailable).
        self._index: Any = None
        self._tokenizer: Any = None
        self._searcher: Any = None
        self._turns_list: list[dict[str, Any]] = []
        self._doc_id_counter = 0
        self._init_legacy_state()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _init_legacy_state(self) -> None:
        from kimix.retrieval import InvertedIndex, NgramTokenizer, Searcher

        self._index = InvertedIndex()
        self._tokenizer = NgramTokenizer(n=2)
        self._searcher = None
        self._turns_list = []
        self._doc_id_counter = 0

    @property
    def _use_fts(self) -> bool:
        return self._db_path is not None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> apsw.Connection:
        if self._conn is None:
            if self._db_path is None:
                raise RuntimeError("HistoryIndex has no db_path (legacy mode)")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = apsw.Connection(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._conn
        assert conn is not None
        # apsw has no executescript(); execute each DDL statement separately.
        for statement in _SCHEMA_STATEMENTS:
            statement = statement.strip()
            if statement:
                conn.execute(statement)
        self._sync_doc_id_counter()
        self._load_fts_stale()

    def _load_fts_stale(self) -> None:
        """Read the persisted stale-index breadcrumb (plan §4 #3)."""
        self._fts_stale = False
        if self._conn is None:
            return
        try:
            cursor = self._conn.execute(
                "SELECT 1 FROM meta WHERE key = ?", (_FTS_STALE_KEY,)
            )
            try:
                self._fts_stale = cursor.fetchone() is not None
            finally:
                cursor.close()
        except Exception:
            self._fts_stale = False

    def _set_fts_stale(self) -> None:
        """Persist the stale-index breadcrumb so later searches skip FTS."""
        self._fts_stale = True
        try:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_FTS_STALE_KEY, "1"),
            )
        except Exception:
            pass

    def _clear_fts_stale(self) -> None:
        """Clear the stale-index breadcrumb after a successful rebuild."""
        self._fts_stale = False
        try:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM meta WHERE key = ?", (_FTS_STALE_KEY,))
        except Exception:
            pass

    def _sync_doc_id_counter(self) -> None:
        """Track the next turn_id (0-based, matching the legacy in-memory
        counter so ``prune_<n>`` references stay valid)."""
        if self._conn is None:
            return
        cursor = self._conn.execute("SELECT COALESCE(MAX(turn_id), -1) + 1 FROM turns")
        try:
            row = cursor.fetchone()
            self._doc_id_counter = int(row[0] if row and row[0] is not None else 0)
        finally:
            cursor.close()

    def _begin(self) -> None:
        if not self._in_transaction:
            self._ensure_conn().execute("BEGIN")
            self._in_transaction = True

    def _commit(self) -> None:
        if self._in_transaction and self._conn is not None:
            self._conn.execute("COMMIT")
            self._in_transaction = False

    def _rollback(self) -> None:
        if self._in_transaction and self._conn is not None:
            try:
                self._conn.execute("ROLLBACK")
            finally:
                self._in_transaction = False

    def close(self) -> None:
        """Close the SQLite connection (best-effort; safe to call twice)."""
        self._commit()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _message_to_text(self, message: Message) -> str:
        parts: list[str] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append(part.text)
        if not parts:
            # Fall back to the generic extractor (handles str content and
            # raw dict parts from migrated/imported rows).
            return fts5_search.extract_text_from_content(message.content)
        return "\n".join(parts)

    def index_messages(self, messages: Sequence[Message]) -> None:
        """Add *messages* to the index, including tool role messages.

        In FTS5 mode the inserts run inside a single transaction — each row
        fires the FTS triggers, so batching is required for fast backfills
        (measured ~0.4 s for 5,000 rows batched vs ~12.9 s autocommit).
        """
        if not self._use_fts:
            self._index_messages_legacy(messages)
            return

        rows: list[tuple[int, str, str, float]] = []
        now = time.time()
        next_id = self._doc_id_counter
        for msg in messages:
            if msg.role not in {"user", "assistant", "tool"}:
                continue
            text = self._message_to_text(msg)
            if not text.strip():
                continue
            rows.append((next_id, msg.role, text, now))
            next_id += 1

        if not rows:
            return
        try:
            self._begin()
            conn = self._ensure_conn()
            conn.executemany(
                "INSERT INTO turns (turn_id, role, text, timestamp) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._doc_id_counter = next_id
            self._commit()
            self._maybe_merge_fts()
        except Exception:
            self._rollback()
            raise

    def _maybe_merge_fts(self) -> None:
        """Run a bounded FTS5 ``merge`` after N writes (plan §4 #1).

        Keeps the FTS b-tree balanced without blocking a write for long.
        """
        if not self._use_fts:
            return
        self._writes_since_merge += 1
        if self._writes_since_merge < _FTS_MERGE_INTERVAL:
            return
        self._writes_since_merge = 0
        try:
            conn = self._ensure_conn()
            for table in ("turns_fts", "turns_fts_trigram"):
                conn.execute(
                    f"INSERT INTO {table}({table}, rank) VALUES('merge', ?)",
                    (_FTS_MERGE_MAX_PAGES,),
                )
        except Exception:
            pass

    def _index_messages_legacy(self, messages: Sequence[Message]) -> None:
        if self._index is None:
            self._init_legacy_state()
        # If the index has been finalized (e.g. after a search), rebuild it
        # from the existing turns so new documents can be added.
        if getattr(self._index, "_finalized", False):
            old_turns = list(self._turns_list)
            self._init_legacy_state()
            for turn in old_turns:
                tokens = self._tokenizer.tokenize(turn["text"])
                self._index.add_document(turn["turn_id"], tokens)

        for msg in messages:
            if msg.role not in {"user", "assistant", "tool"}:
                continue
            text = self._message_to_text(msg)
            if not text.strip():
                continue

            turn = {
                "turn_id": self._doc_id_counter,
                "timestamp": time.time(),
                "role": msg.role,
                "text": text,
                "is_compacted": False,
            }
            self._turns_list.append(turn)
            tokens = self._tokenizer.tokenize(text)
            self._index.add_document(self._doc_id_counter, tokens)
            self._doc_id_counter += 1

        # Enforce size bound — drop oldest turns (legacy in-memory backend).
        while len(self._turns_list) > _MAX_TURNS:
            self._turns_list.pop(0)

        self._searcher = None  # invalidate cached searcher

    def mark_compacted(self) -> None:
        """Mark all currently-indexed turns as compacted/archived."""
        if self._use_fts:
            try:
                self._begin()
                self._ensure_conn().execute("UPDATE turns SET is_compacted = 1")
                self._commit()
            except Exception:
                self._rollback()
                raise
            return
        for turn in self._turns_list:
            turn["is_compacted"] = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the top-*k* matching turns as dicts.

        Routing (FTS5 mode): no CJK → unicode61 index; CJK with >=3-char
        tokens → trigram index; short/lone CJK runs → LIKE substring scan.
        The returned ``score`` is ``-bm25(...)`` so higher = more relevant,
        preserving the legacy positive-score API semantics.
        """
        if not self._use_fts:
            return self._search_legacy(query, top_k)
        if self._db_path is not None and not self._db_path.exists():
            return []
        try:
            conn = self._ensure_conn()
        except Exception:
            return []

        q = sanitize_fts5_query(query)
        if not q:
            return []

        # Stale-index breadcrumb (plan §4 #3): a previously-corrupted FTS table
        # serves LIKE results until rebuild_fts() clears the marker.
        if self._fts_stale:
            return self._search_like(conn, q, top_k)

        try:
            if contains_cjk(q):
                raw_query = q.strip('"').strip()
                if trigram_eligible_tokens(q) and not has_lone_cjk_run(raw_query):
                    try:
                        return self._search_fts_trigram(conn, raw_query, top_k)
                    except apsw.OperationalError:
                        self._set_fts_stale()
                        return self._search_like(conn, raw_query, top_k)
                return self._search_like(conn, raw_query, top_k)
            try:
                return self._search_fts_unicode61(conn, q, top_k)
            except apsw.OperationalError:
                self._set_fts_stale()
                return self._search_like(conn, q, top_k)
        except Exception:
            # Never raise to the LLM tool — degrade to LIKE on any FTS error.
            self._set_fts_stale()
            return self._search_like(conn, q, top_k)

    def _row_to_turn(self, row: tuple, *, with_score: bool = False) -> dict[str, Any]:
        """Convert an apsw result row to the legacy turn dict shape."""
        keys = ("turn_id", "role", "text", "timestamp", "is_compacted")
        if with_score:
            keys += ("score",)
        out = dict(zip(keys, row))
        out["is_compacted"] = bool(out["is_compacted"])
        return out

    def _search_fts_unicode61(
        self, conn: apsw.Connection, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        cursor = conn.execute(
            """
            SELECT t.turn_id, t.role, t.text, t.timestamp, t.is_compacted,
                   -bm25(turns_fts) AS score
            FROM turns_fts
            JOIN turns t ON t.turn_id = turns_fts.rowid
            WHERE turns_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (query, top_k),
        )
        return [self._row_to_turn(row, with_score=True) for row in cursor]

    def _search_fts_trigram(
        self, conn: apsw.Connection, raw_query: str, top_k: int
    ) -> list[dict[str, Any]]:
        trigram_query = quote_fts_tokens(raw_query)
        cursor = conn.execute(
            """
            SELECT t.turn_id, t.role, t.text, t.timestamp, t.is_compacted,
                   -bm25(turns_fts_trigram) AS score
            FROM turns_fts_trigram
            JOIN turns t ON t.turn_id = turns_fts_trigram.rowid
            WHERE turns_fts_trigram MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (trigram_query, top_k),
        )
        return [self._row_to_turn(row, with_score=True) for row in cursor]

    def _search_like(
        self, conn: apsw.Connection, raw_query: str, top_k: int
    ) -> list[dict[str, Any]]:
        tokens = [
            t for t in raw_query.split() if t.upper() not in {"AND", "OR", "NOT"}
        ] or [raw_query]
        clauses: list[str] = []
        params: list[str] = []
        for tok in tokens:
            esc = escape_like(tok)
            clauses.append("text LIKE ? ESCAPE '\\'")
            params.append(f"%{esc}%")
        where = " OR ".join(clauses)
        cursor = conn.execute(
            f"""
            SELECT turn_id, role, text, timestamp, is_compacted, 0.0 AS score
            FROM turns
            WHERE {where}
            ORDER BY timestamp DESC, turn_id DESC
            LIMIT ?
            """,
            (*params, top_k),
        )
        return [self._row_to_turn(row, with_score=True) for row in cursor]

    def _search_legacy(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._turns_list:
            return []
        if self._index is None:
            return []
        if not getattr(self._index, "_finalized", False):
            self._index.finalize()
        if self._searcher is None:
            from kimix.retrieval import Searcher

            self._searcher = Searcher(self._index, tokenizer=self._tokenizer)
        results = self._searcher.search(query, top_k=top_k)
        out: list[dict[str, Any]] = []
        for doc_id, score in results:
            for turn in self._turns_list:
                if turn["turn_id"] == doc_id:
                    out.append({**turn, "score": score})
                    break
        return out

    def get_by_id(self, ref: str) -> dict[str, Any] | None:
        """Retrieve a turn by its reference ID (turn_id).

        The *ref* may be a plain integer string (``"42"``) or a prefixed
        reference like ``"prune_42"`` — the numeric suffix is tried.
        """
        ref_str = str(ref)
        if ref_str.startswith("prune_"):
            ref_str = ref_str[len("prune_"):]
        try:
            turn_id = int(ref_str)
        except ValueError:
            return None

        if self._use_fts:
            cursor = None
            try:
                conn = self._ensure_conn()
                cursor = conn.execute(
                    "SELECT turn_id, role, text, timestamp, is_compacted FROM turns WHERE turn_id = ?",
                    (turn_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_turn(row)
            except Exception:
                return None
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

        for turn in self._turns_list:
            if turn["turn_id"] == turn_id:
                return dict(turn)
        return None

    def search_with_recency(
        self,
        query: str,
        *,
        top_k: int = 3,
        recency_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """BM25 search with recency boosting.

        boosted_score = bm25_score * (1 + recency_weight * exp(-hours_ago / 24.0))
        """
        turns = self._turns if self._use_fts else self._turns_list
        if not turns:
            return []

        # Fetch a larger candidate pool so recency re-ranking has room to work
        candidates = self.search(query, top_k=top_k * 3)
        if not candidates:
            return []

        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for turn in candidates:
            bm25_score = turn.get("score", 0.0)
            hours_ago = (now - turn["timestamp"]) / 3600.0
            boost = 1.0 + recency_weight * math.exp(-hours_ago / 24.0)
            boosted_score = bm25_score * boost
            scored.append((boosted_score, {**turn, "boosted_score": boosted_score}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [turn for _, turn in scored[:top_k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist state.  No-op for the FTS5 backend (index is durable)."""
        if self._use_fts:
            try:
                self._commit()
            except Exception:
                self._rollback()
            return
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "doc_id_counter": self._doc_id_counter,
            "turns": self._turns_list,
        }
        self._persist_path.write_text(
            orjson.dumps(data).decode("utf-8"), encoding="utf-8"
        )

    def load(self) -> bool:
        """Load turn metadata from disk.  Returns ``True`` on success.

        FTS5 mode:
        - ``history.db`` exists → open + init schema → True
        - legacy JSON exists → migrate into ``history.db``, rename JSON to
          ``.json.bak`` → True
        - neither → create empty schema → False (same contract as today)
        """
        if self._use_fts:
            return self._load_fts()

        if self._persist_path is None or not self._persist_path.exists():
            return False
        try:
            data = orjson.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        turns = data.get("turns", [])
        self._init_legacy_state()
        self._doc_id_counter = data.get("doc_id_counter", 0)
        self._turns_list = turns
        for turn in self._turns_list:
            tokens = self._tokenizer.tokenize(turn["text"])
            self._index.add_document(turn["turn_id"], tokens)
        self._searcher = None
        return True

    def _load_fts(self) -> bool:
        assert self._db_path is not None
        db_existed = self._db_path.exists()
        try:
            conn = self._ensure_conn()
        except Exception:
            # FTS5 unavailable or DB unreadable — fall back to legacy mode
            # (in-memory BM25) so sessions stay usable.
            self._db_path = None
            self.close()
            self._init_legacy_state()
            return self.load()

        if db_existed:
            return True

        # Legacy JSON migration (mirrors _migrate_jsonl_to_sqlite).
        legacy = self._legacy_json_path
        if legacy is not None and legacy.exists():
            try:
                data = orjson.loads(legacy.read_text(encoding="utf-8"))
                turns = data.get("turns", [])
                rows = [
                    (
                        int(turn.get("turn_id", i)),
                        turn.get("role", "user"),
                        turn.get("text", ""),
                        float(turn.get("timestamp", 0.0) or 0.0),
                        int(bool(turn.get("is_compacted", False))),
                    )
                    for i, turn in enumerate(turns)
                ]
                try:
                    self._begin()
                    conn.executemany(
                        "INSERT INTO turns (turn_id, role, text, timestamp, is_compacted) VALUES (?, ?, ?, ?, ?)",
                        rows,
                    )
                    self._commit()
                except Exception:
                    self._rollback()
                    raise
                self._sync_doc_id_counter()
                legacy.replace(legacy.with_suffix(".json.bak"))
                return True
            except Exception:
                # Broken/missing legacy JSON is not fatal — fresh index.
                return False
        return False

    def clear(self) -> None:
        """Clear all data and delete the persisted files."""
        if self._use_fts:
            self.close()
            for path in self._db_path.parent.glob(f"{self._db_path.name}*"):
                try:
                    path.unlink()
                except OSError:
                    pass
            self._init_legacy_state()
            self._doc_id_counter = 0
            self._fts_stale = False
            self._writes_since_merge = 0
            if self._legacy_json_path is not None and self._legacy_json_path.exists():
                try:
                    self._legacy_json_path.unlink()
                except OSError:
                    pass
            return

        self._init_legacy_state()
        if self._persist_path is not None and self._persist_path.exists():
            self._persist_path.unlink()

    # ------------------------------------------------------------------
    # FTS maintenance (plan §6 / Phase C)
    # ------------------------------------------------------------------

    def rebuild_fts(self) -> None:
        """Drop + recreate the FTS tables and backfill from ``turns``.

        Clears the ``fts_stale`` breadcrumb so FTS serving resumes.  No-op in
        legacy (persist_path) mode.  The connection is closed and reopened
        first so any caller-held read cursor cannot block the DDL (this is a
        repair path; callers must not use cursors across the call).
        """
        if not self._use_fts:
            return
        self.close()
        conn = self._ensure_conn()
        try:
            self._begin()
            for trigger in _FTS_TRIGGER_NAMES:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            for table in ("turns_fts", "turns_fts_trigram"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._commit()
            # Recreate the FTS tables + triggers (statements after the base
            # turns/meta tables).
            for statement in _SCHEMA_STATEMENTS[2:]:
                statement = statement.strip()
                if statement:
                    conn.execute(statement)
            self._begin()
            conn.execute(
                "INSERT INTO turns_fts(rowid, text) SELECT turn_id, text FROM turns"
            )
            conn.execute(
                "INSERT INTO turns_fts_trigram(rowid, text) SELECT turn_id, text FROM turns"
            )
            self._commit()
            self._clear_fts_stale()
        except Exception:
            self._rollback()
            raise

    # ------------------------------------------------------------------
    # Compat surface
    # ------------------------------------------------------------------

    @property
    def _turns(self) -> list[dict[str, Any]]:
        """Current turns as a list (oldest first), bounded for the compat path."""
        if not self._use_fts:
            return self._turns_list
        try:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "SELECT turn_id, role, text, timestamp, is_compacted FROM turns "
                "ORDER BY turn_id DESC LIMIT ?",
                (_MAX_TURNS_COMPAT,),
            )
            try:
                rows = list(cursor)
            finally:
                cursor.close()
        except Exception:
            return []
        # Reverse so the list is oldest → newest (legacy list order).
        return [self._row_to_turn(row) for row in reversed(rows)]
