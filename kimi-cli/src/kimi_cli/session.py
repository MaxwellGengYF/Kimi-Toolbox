from __future__ import annotations

import asyncio
import builtins
import shutil
import time
import uuid

import orjson
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
from kaos.path import KaosPath
from kosong.message import Message
from kosong.utils.jsonx import loads_relaxed

from kimi_cli.file_mtine import FileMTime
from kimi_cli.metadata import WorkDirMeta, load_metadata, save_metadata
from kimi_cli.session_state import SessionState, load_session_state, save_session_state
from kimi_cli.soul.context_db import ContextDB
from kimi_cli.soul.context_records import ExportedContext
from kimi_cli.utils.logging import logger
from kimi_cli.utils.string import shorten
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.types import TurnBegin


@dataclass(slots=True, kw_only=True)
class Session:
    """A session of a work directory."""

    # static metadata
    id: str
    """The session ID."""
    work_dir: KaosPath
    """The absolute path of the work directory."""
    work_dir_meta: WorkDirMeta
    """The metadata of the work directory."""
    context_file: Path
    """The absolute path to the file storing the message history (backward compat)."""
    wire_file: WireFile
    """The wire message log file wrapper."""

    # session state
    state: SessionState
    """Persisted session state (approval settings, plan mode, workspace scope, etc.)."""

    # refreshable metadata
    title: str
    """The title of the session."""
    updated_at: float
    """The timestamp of the last update to the session."""

    custom_data: dict[str, Any]
    
    custom_config: dict[str, Any]

    file_mtime: FileMTime = field(default_factory=FileMTime)

    # Internal: lazy-loaded ConflictHistory (plan 24) attached by
    # kimi_cli.tools.file.conflict_detect.get_conflict_history on first use.
    # Typed ``Any`` to avoid an import cycle with the file tools package.
    conflict_history: Any | None = field(default=None, repr=False, compare=False)
    # Internal: lazy-loaded versioned InMemorySnapshotStore (plan 25) attached
    # by kimi_cli.tools.file.snapshot_store.get_file_snapshot_store on first
    # use.  Typed ``Any`` to avoid an import cycle with the file tools package.
    file_snapshot_store: Any | None = field(default=None, repr=False, compare=False)

    # Internal: lazy-loaded ContextDB for this session
    _context_db: ContextDB | None = field(default=None, repr=False, compare=False)

    # Internal: optional HistoryIndex attached by KimiSoul (FTS5 backend).
    # Closed before session-dir operations so Windows file locks are released.
    _history_index: Any | None = field(default=None, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        """The absolute path of the session directory."""
        path = self.work_dir_meta.sessions_dir / self.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def subagents_dir(self) -> Path:
        """The absolute path of the subagent instances directory."""
        path = self.dir / "subagents"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def context_db_file(self) -> Path:
        """The absolute path to the SQLite database file."""
        return self.dir / "context.db"

    def _get_context_db(self) -> ContextDB:
        """Get or create a ContextDB for this session."""
        if self._context_db is None:
            self._context_db = ContextDB(self.context_db_file)
        return self._context_db

    async def close_context_db(self) -> None:
        """Close the internal ContextDB connection if open."""
        if self._context_db is not None:
            await self._context_db.close()
            self._context_db = None

    def close_history_index(self) -> None:
        """Close the attached HistoryIndex connection if any (best-effort).

        The FTS5 ``history.db`` is held open by an apsw connection; on Windows
        that handle keeps the session directory locked until closed, so
        delete/rename/copy paths release it first.
        """
        idx = getattr(self, "_history_index", None)
        if idx is not None:
            close = getattr(idx, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def is_empty(self) -> bool:
        """Whether the session has any context history or a custom title."""
        if self.state.custom_title:
            return False
        if not self.wire_file.is_empty():
            return False

        # Check SQLite first, fall back to JSONL
        db_file = self.context_db_file
        if db_file.exists():
            try:
                import apsw
                conn = apsw.Connection(str(db_file))
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT 1 FROM messages WHERE role NOT IN ('_system_prompt', '_usage', '_checkpoint') LIMIT 1"
                    )
                    row = cursor.fetchone()
                    return row is None
                finally:
                    conn.close()
            except Exception:
                pass

        # Fallback to JSONL
        try:
            with self.context_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    role = loads_relaxed(line).get("role")
                    if isinstance(role, str) and not role.startswith("_"):
                        return False
        except FileNotFoundError:
            return True
        except (OSError, ValueError, TypeError):
            logger.exception("Failed to read context file {file}:", file=self.context_file)
            return False
        return True

    def save_state(self) -> None:
        """Persist the session state to disk.

        Reloads externally-mutable fields (title, archive) from disk first
        to avoid overwriting concurrent changes made by the web API.
        """
        fresh = load_session_state(self.dir)
        self.state.custom_title = fresh.custom_title
        self.state.title_generated = fresh.title_generated
        self.state.title_generate_attempts = fresh.title_generate_attempts
        self.state.archived = fresh.archived
        self.state.archived_at = fresh.archived_at
        self.state.auto_archive_exempt = fresh.auto_archive_exempt
        save_session_state(self.state, self.dir)

    async def delete(self) -> None:
        """Delete the session directory."""
        await self.close_context_db()
        self.close_history_index()
        session_dir = self.work_dir_meta.sessions_dir / self.id
        if not session_dir.exists():
            return
        await asyncio.to_thread(shutil.rmtree, session_dir, True)
        
    def delete_sync(self) -> None:
        """Delete the session directory.

        Best-effort: the aiosqlite worker thread (if any) is stopped first so
        SQLite files are not locked during removal on Windows, and the removal
        is retried briefly because the thread may still hold the file handles
        for a moment after it has been stopped.
        """
        session_dir = self.work_dir_meta.sessions_dir / self.id
        if not session_dir.exists():
            return
        db = self._context_db
        if db is not None:
            db.stop_sync()
        self.close_history_index()
        deadline = time.monotonic() + 2.0
        while True:
            shutil.rmtree(session_dir, True)
            if not session_dir.exists() or time.monotonic() >= deadline:
                return
            time.sleep(0.05)

    async def refresh(self) -> None:
        self.title = "Untitled"

        # Check mtime from context.db first, then context.jsonl
        db_file = self.context_db_file
        jsonl_file = self.context_file
        if db_file.exists():
            self.updated_at = db_file.stat().st_mtime
        elif jsonl_file.exists():
            self.updated_at = jsonl_file.stat().st_mtime
        else:
            self.updated_at = 0.0

        if self.state.custom_title:
            self.title = self.state.custom_title
            return

        async for record in self.wire_file.iter_records():
            try:
                wire_msg = record.to_wire_message()
            except Exception:
                # Unknown/unparseable records (e.g. observability records written
                # by a newer binary) must not abort title derivation.
                logger.debug(
                    "Skipping unparseable wire record in {file} during title derivation",
                    file=self.wire_file.path,
                )
                continue
            if isinstance(wire_msg, TurnBegin):
                self.title = shorten(
                    Message(role="user", content=wire_msg.user_input).extract_text(" "),
                    width=50,
                )
                return

    async def export(self) -> ExportedContext:
        """Export all data from the session's context file.

        Uses SQLite backend if available, falls back to JSONL.

        Returns:
            ExportedContext: Structured representation of the context file.
        """
        from pydantic import ValidationError

        from kimi_cli.soul.context_records import (
            CheckpointRecord,
            SystemPromptRecord,
            UsageRecord,
        )

        # Try SQLite first
        db_file = self.context_db_file
        if db_file.exists():
            db = ContextDB(db_file)
            try:
                await db.initialize()
                return await db.export()
            except Exception:
                logger.exception("Failed to export from SQLite, falling back to JSONL")
            finally:
                await db.close()

        # Fallback to JSONL
        result = ExportedContext()
        if not self.context_file.exists() or self.context_file.stat().st_size == 0:
            return result

        async with aiofiles.open(self.context_file, encoding="utf-8", errors="replace") as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    line_json = loads_relaxed(line)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(line_json, dict):
                    continue

                role = line_json.get("role")
                if not isinstance(role, str):
                    continue
                if role == "_system_prompt":
                    try:
                        record = SystemPromptRecord.model_validate(line_json)
                        result.system_prompt = record.content
                    except ValidationError:
                        continue
                elif role == "_usage":
                    try:
                        record = UsageRecord.model_validate(line_json)
                        result.usages.append(record.token_count)
                    except ValidationError:
                        continue
                elif role == "_checkpoint":
                    try:
                        record = CheckpointRecord.model_validate(line_json)
                        result.checkpoints.append(record.id)
                    except ValidationError:
                        continue
                else:
                    try:
                        message = Message.model_validate(line_json)
                        result.messages.append(message)
                    except ValidationError:
                        continue

        return result

    @staticmethod
    async def create(
        work_dir: KaosPath,
        session_id: str | None = None,
        _context_file: Path | None = None,
    ) -> Session:
        """Create a new session for a work directory."""
        work_dir = work_dir.canonical()
        logger.debug("Creating new session for work directory: {work_dir}", work_dir=work_dir)

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            work_dir_meta = metadata.new_work_dir_meta(work_dir)

        if session_id is None:
            session_id = uuid.uuid4().hex
        session_dir = work_dir_meta.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if _context_file is not None:
            # Custom context file provided (backward compat / tests)
            logger.warning(
                "Using provided context file: {context_file}", context_file=_context_file
            )
            _context_file.parent.mkdir(parents=True, exist_ok=True)
            if _context_file.exists():
                assert _context_file.is_file()
            context_file = _context_file
            if context_file.exists():
                logger.warning(
                    "Context file already exists, truncating: {context_file}", context_file=context_file
                )
                context_file.unlink()
            context_file.touch()
        else:
            # Default: use .db file for new sessions
            context_file = session_dir / "context.db"
            context_file.touch()

        save_metadata(metadata)

        session = Session(
            id=session_id,
            work_dir=work_dir,
            work_dir_meta=work_dir_meta,
            context_file=context_file,
            wire_file=WireFile(path=session_dir / "wire.jsonl"),
            state=SessionState(),
            title="",
            updated_at=0.0,
            custom_data={},
            custom_config={},
        )
        await session.refresh()
        return session

    @staticmethod
    async def find(work_dir: KaosPath, session_id: str) -> Session | None:
        """Find a session by work directory and session ID."""
        work_dir = work_dir.canonical()
        logger.debug(
            "Finding session for work directory: {work_dir}, session ID: {session_id}",
            work_dir=work_dir,
            session_id=session_id,
        )

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            logger.debug("Work directory never been used")
            return None

        _migrate_session_context_file(work_dir_meta, session_id)

        session_dir = work_dir_meta.sessions_dir / session_id
        if not session_dir.is_dir():
            logger.debug("Session directory not found: {session_dir}", session_dir=session_dir)
            return None

        # Look for .db first, then .jsonl
        context_file = session_dir / "context.db"
        if not context_file.exists():
            context_file = session_dir / "context.jsonl"
            if not context_file.exists():
                logger.debug(
                    "Session context file not found: {context_file}", context_file=context_file
                )
                return None

        session = Session(
            id=session_id,
            work_dir=work_dir,
            work_dir_meta=work_dir_meta,
            context_file=context_file,
            wire_file=WireFile(path=session_dir / "wire.jsonl"),
            state=load_session_state(session_dir),
            title="",
            updated_at=0.0,
            custom_data={},
            custom_config={},
        )
        await session.refresh()
        return session

    @staticmethod
    async def list(work_dir: KaosPath) -> builtins.list[Session]:
        """List all sessions for a work directory."""
        work_dir = work_dir.canonical()
        logger.debug("Listing sessions for work directory: {work_dir}", work_dir=work_dir)

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            logger.debug("Work directory never been used")
            return []

        session_ids = set()
        for path in work_dir_meta.sessions_dir.iterdir():
            if path.is_dir():
                session_ids.add(path.name)
            elif path.suffix in (".jsonl", ".db") and path.stem not in session_ids:
                # Legacy flat-file sessions
                session_ids.add(path.stem)

        sessions: list[Session] = []
        for session_id in session_ids:
            _migrate_session_context_file(work_dir_meta, session_id)
            session_dir = work_dir_meta.sessions_dir / session_id
            if not session_dir.is_dir():
                logger.debug("Session directory not found: {session_dir}", session_dir=session_dir)
                continue

            # Look for .db first, then .jsonl
            context_file = session_dir / "context.db"
            if not context_file.exists():
                context_file = session_dir / "context.jsonl"
                if not context_file.exists():
                    logger.debug(
                        "Session context file not found: {context_file}", context_file=context_file
                    )
                    continue

            session = Session(
                id=session_id,
                work_dir=work_dir,
                work_dir_meta=work_dir_meta,
                context_file=context_file,
                wire_file=WireFile(path=session_dir / "wire.jsonl"),
                state=load_session_state(session_dir),
                title="",
                updated_at=0.0,
                custom_data={},
                custom_config={},
            )
            if session.is_empty():
                logger.debug(
                    "Session context file is empty: {context_file}", context_file=context_file
                )
                continue
            await session.refresh()
            sessions.append(session)
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions

    @classmethod
    async def list_all(cls) -> builtins.list[Session]:
        """List sessions across all known work directories."""
        all_sessions: list[Session] = []
        for wd in load_metadata().work_dirs:
            sessions = await cls.list(KaosPath.unsafe_from_local_path(Path(wd.path)))
            all_sessions.extend(sessions)
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return all_sessions

    @staticmethod
    async def continue_(work_dir: KaosPath) -> Session | None:
        """Get the last session for a work directory."""
        work_dir = work_dir.canonical()
        logger.debug("Continuing session for work directory: {work_dir}", work_dir=work_dir)

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            logger.debug("Work directory never been used")
            return None
        if work_dir_meta.last_session_id is None:
            logger.debug("Work directory never had a session")
            return None

        logger.debug(
            "Found last session for work directory: {session_id}",
            session_id=work_dir_meta.last_session_id,
        )
        return await Session.find(work_dir, work_dir_meta.last_session_id)

    @staticmethod
    async def rename(
        work_dir: KaosPath, session_id: str, new_session_id: str
    ) -> Session | None:
        """Rename a session to a new session ID.

        Args:
            work_dir: Working directory containing the session.
            session_id: The current session ID to rename.
            new_session_id: The new session ID.

        Returns:
            Session | None: The renamed session, or None if the session
                was not found or the new session ID already exists.
        """
        work_dir = work_dir.canonical()
        logger.debug(
            "Renaming session for work directory: {work_dir}, "
            "session ID: {session_id} to {new_session_id}",
            work_dir=work_dir,
            session_id=session_id,
            new_session_id=new_session_id,
        )

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            logger.debug("Work directory never been used")
            return None

        old_session_dir = work_dir_meta.sessions_dir / session_id
        if not old_session_dir.is_dir():
            logger.debug(
                "Session directory not found: {session_dir}",
                session_dir=old_session_dir,
            )
            return None

        # Check for either .db or .jsonl
        old_context_file_db = old_session_dir / "context.db"
        old_context_file_jsonl = old_session_dir / "context.jsonl"
        if not old_context_file_db.exists() and not old_context_file_jsonl.exists():
            logger.debug(
                "Session context file not found (checked .db and .jsonl): {session_dir}",
                session_dir=old_session_dir,
            )
            return None

        new_session_dir = work_dir_meta.sessions_dir / new_session_id
        if new_session_dir.exists():
            logger.debug(
                "Target session directory already exists: {session_dir}",
                session_dir=new_session_dir,
            )
            return None

        # Rename the session directory.  On Windows the SQLite WAL file can
        # remain locked for a short moment after the connection is closed, so
        # retry a few times before giving up.
        for attempt in range(5):
            try:
                old_session_dir.rename(new_session_dir)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                logger.debug(
                    "Rename attempt {attempt} failed for {old} -> {new}, retrying",
                    attempt=attempt,
                    old=old_session_dir,
                    new=new_session_dir,
                )
                await asyncio.sleep(0.05)

        if work_dir_meta.last_session_id == session_id:
            work_dir_meta.last_session_id = new_session_id
            save_metadata(metadata)

        return await Session.find(work_dir, new_session_id)

    @staticmethod
    async def copy(
        work_dir: KaosPath, source_session_id: str, target_session_id: str
    ) -> Session:
        """Copy an existing session directory to a new session ID.

        Args:
            work_dir: Working directory containing the sessions.
            source_session_id: ID of the session to copy.
            target_session_id: ID for the new copied session.

        Returns:
            Session: The newly created target session.

        Raises:
            ValueError: If the source session does not exist, the target ID is empty,
                or a session with the target ID already exists.
        """
        work_dir = work_dir.canonical()
        if not target_session_id:
            raise ValueError("Target session ID cannot be empty")
        if source_session_id == target_session_id:
            raise ValueError("Target session ID must differ from source session ID")

        metadata = load_metadata()
        work_dir_meta = metadata.get_work_dir_meta(work_dir)
        if work_dir_meta is None:
            raise ValueError(f"Source session not found: {source_session_id}")

        _migrate_session_context_file(work_dir_meta, source_session_id)
        source_dir = work_dir_meta.sessions_dir / source_session_id
        if not source_dir.is_dir():
            raise ValueError(f"Source session not found: {source_session_id}")

        target_dir = work_dir_meta.sessions_dir / target_session_id
        if target_dir.exists():
            raise ValueError(f"Target session already exists: {target_session_id}")

        await asyncio.to_thread(shutil.copytree, source_dir, target_dir)

        copied = await Session.find(work_dir, target_session_id)
        if copied is None:
            raise RuntimeError(f"Failed to open copied session: {target_session_id}")
        return copied


def _migrate_session_context_file(work_dir_meta: WorkDirMeta, session_id: str) -> None:
    """Migrate legacy session context files.

    Handles two migrations:
    1. Flat JSONL file (session_id.jsonl) → session_dir/context.jsonl
    2. context.jsonl → context.db (auto-migration on access)
    """
    session_dir = work_dir_meta.sessions_dir / session_id

    # Migration 1: Flat JSONL → session dir
    old_flat_file = work_dir_meta.sessions_dir / f"{session_id}.jsonl"
    new_context_jsonl = session_dir / "context.jsonl"
    if old_flat_file.exists() and not new_context_jsonl.exists():
        new_context_jsonl.parent.mkdir(parents=True, exist_ok=True)
        old_flat_file.rename(new_context_jsonl)
        logger.info(
            "Migrated session context file from {old} to {new}",
            old=old_flat_file,
            new=new_context_jsonl,
        )

    # Migration 2: JSONL → SQLite (only if JSONL exists and DB doesn't)
    if new_context_jsonl.exists() and not (session_dir / "context.db").exists():
        _migrate_jsonl_to_sqlite(new_context_jsonl)


def _migrate_jsonl_to_sqlite(jsonl_path: Path) -> None:
    """Migrate a context.jsonl file to SQLite context.db.

    Reads the JSONL file line by line and inserts into the SQLite database.
    On success, renames the JSONL file to .bak.
    """
    import orjson as json_module

    db_path = jsonl_path.with_suffix(".db")
    if db_path.exists():
        logger.debug("SQLite DB already exists, skipping migration: {db}", db=db_path)
        return

    if not jsonl_path.exists():
        return

    logger.info("Migrating context from JSONL to SQLite: {jsonl} -> {db}", jsonl=jsonl_path, db=db_path)

    try:
        from kimi_cli.soul.context_db import ContextDB

        # Read all lines from JSONL
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            lines = [line for line in f if line.strip()]

        if not lines:
            logger.debug("Empty JSONL file, nothing to migrate")
            # Still create an empty DB
            db_path.touch()
            return

        import asyncio

        async def _do_migration():
            db = ContextDB(db_path)
            await db.initialize()

            await db.begin_transaction()
            try:
                for line in lines:
                    try:
                        line_json = loads_relaxed(line)
                    except json_module.JSONDecodeError:
                        continue
                    if not isinstance(line_json, dict):
                        continue

                    await db.import_jsonl_line(line_json)

                # Fix checkpoint message_rowid references
                await db.fix_checkpoint_message_rowids()
                await db.commit_transaction()
            except Exception:
                await db.rollback_transaction()
                raise
            finally:
                await db.close()

        asyncio.run(_do_migration())

        # Verify row counts match
        async def _verify():
            db = ContextDB(db_path)
            await db.initialize()
            db_msg_count = await db.get_message_count()
            await db.close()
            return db_msg_count

        db_msg_count = asyncio.run(_verify())

        jsonl_line_count = len([l for l in lines if l.strip()])
        logger.debug(
            "Migration verification: JSONL lines={jsonl}, DB messages={db}",
            jsonl=jsonl_line_count,
            db=db_msg_count,
        )

        # Rename JSONL to .bak
        backup_path = jsonl_path.with_suffix(".jsonl.bak")
        jsonl_path.rename(backup_path)
        logger.info("Migration complete. Backup at: {backup}", backup=backup_path)

    except Exception:
        logger.exception("Failed to migrate context from JSONL to SQLite: {jsonl}", jsonl=jsonl_path)
        # Clean up partial DB on failure
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass
        raise
