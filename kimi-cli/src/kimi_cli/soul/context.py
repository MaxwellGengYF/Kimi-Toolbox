from __future__ import annotations

import asyncio
import regex as re

import orjson
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import aiofiles
import aiofiles.os
from kosong.message import Message
from kosong.utils.jsonx import loads_relaxed
from pydantic import ValidationError

from kimi_cli.soul.compaction import estimate_text_tokens
from kimi_cli.soul.context_records import ExportedContext
from kimi_cli.soul.message import system
from kimi_cli.utils.logging import logger


# ---------------------------------------------------------------------------
# ContextStorage protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ContextStorage(Protocol):
    """Protocol for context storage backends.

    Implementations must provide durable, ordered storage for conversation
    context including messages, system prompt, checkpoints, and usage data.
    """

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def get_system_prompt(self) -> str | None: ...
    async def set_system_prompt(self, content: str) -> None: ...
    async def append_messages(self, messages: Sequence[Message]) -> None: ...
    async def get_messages(self, *, after_rowid: int = 0, limit: int | None = None) -> list[Message]: ...
    async def get_message_count(self) -> int: ...
    async def has_visible_messages(self) -> bool: ...
    async def create_checkpoint(self, checkpoint_id: int) -> int: ...
    async def get_latest_checkpoint_id(self) -> int: ...
    async def revert_to_checkpoint(self, checkpoint_id: int) -> None: ...
    async def record_usage(self, token_count: int) -> None: ...
    async def get_latest_usage(self) -> int | None: ...
    async def clear(self) -> None: ...
    async def export(self) -> ExportedContext: ...

    async def restore_full(self) -> tuple[str | None, list[Message], int | None, int, list[Message]]:
        """Full restore returning (system_prompt, all_messages, latest_usage, next_checkpoint_id, messages_after_last_usage).

        This is needed for correct pending token estimation where messages after
        the last usage record are tracked separately.
        """
        ...


# ---------------------------------------------------------------------------
# JsonlContextStorage — legacy JSONL backend
# ---------------------------------------------------------------------------


class JsonlContextStorage:
    """JSONL-based context storage (legacy backend).

    Maintains full backward compatibility with existing ``context.jsonl``
    files on disk.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._write_handle: Any | None = None

    @property
    def storage_path(self) -> Path:
        return self._file_path

    async def initialize(self) -> None:
        """Ensure the JSONL file exists and open write handle."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_handle = await aiofiles.open(self._file_path, mode="a", encoding="utf-8")

    async def close(self) -> None:
        """Close the write handle."""
        if self._write_handle is not None:
            await self._write_handle.close()
            self._write_handle = None

    # ---- System prompt ---- #

    async def get_system_prompt(self) -> str | None:
        """Scan the JSONL file for the system prompt (first _system_prompt line)."""
        if not self._file_path.exists():
            return None
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    line_json = loads_relaxed(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(line_json, dict) and line_json.get("role") == "_system_prompt":
                    content = line_json.get("content")
                    if isinstance(content, str):
                        return content
        return None

    async def set_system_prompt(self, content: str) -> None:
        """Write or prepend the system prompt."""
        # Close the write handle before rewriting to avoid Windows file-lock conflicts
        if self._write_handle is not None:
            await self._write_handle.close()
            self._write_handle = None

        prompt_line = (
            orjson.dumps({"role": "_system_prompt", "content": content}).decode("utf-8") + "\n"
        )

        def _write_sync() -> None:
            if not self._file_path.exists() or self._file_path.stat().st_size == 0:
                self._file_path.write_text(prompt_line, encoding="utf-8")
                return
            tmp_path = self._file_path.with_suffix(".tmp")
            with (
                tmp_path.open("w", encoding="utf-8") as tmp_f,
                self._file_path.open(encoding="utf-8") as src_f,
            ):
                tmp_f.write(prompt_line)
                while True:
                    chunk = src_f.read(64 * 1024)
                    if not chunk:
                        break
                    tmp_f.write(chunk)
            tmp_path.replace(self._file_path)

        await asyncio.to_thread(_write_sync)

        # Reopen the write handle for subsequent append operations
        self._write_handle = await aiofiles.open(self._file_path, mode="a", encoding="utf-8")

    # ---- Messages ---- #

    async def append_messages(self, messages: Sequence[Message]) -> None:
        if self._write_handle is not None:
            for message in messages:
                await self._write_handle.write(message.model_dump_json(exclude_none=True) + "\n")
            await self._write_handle.flush()
        else:
            async with aiofiles.open(self._file_path, "a", encoding="utf-8") as f:
                for message in messages:
                    await f.write(message.model_dump_json(exclude_none=True) + "\n")

    async def get_messages(
        self, *, after_rowid: int = 0, limit: int | None = None
    ) -> list[Message]:
        messages: list[Message] = []
        line_no = 0
        if not self._file_path.exists():
            return messages
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                line_no += 1
                if line_no <= after_rowid:
                    continue
                if not line.strip():
                    continue
                try:
                    line_json = loads_relaxed(line)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(line_json, dict):
                    continue
                role = line_json.get("role")
                # Skip meta-records (system prompt, usage, checkpoint)
                if not isinstance(role, str) or role.startswith("_"):
                    continue
                try:
                    message = Message.model_validate(line_json)
                except ValidationError:
                    continue
                messages.append(message)
                if limit is not None and len(messages) >= limit:
                    break
        return messages

    async def get_message_count(self) -> int:
        count = 0
        if not self._file_path.exists():
            return 0
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                if line.strip():
                    count += 1
        return count

    async def has_visible_messages(self) -> bool:
        """Check if there are non-meta role messages."""
        if not self._file_path.exists():
            return False
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                role = None
                try:
                    role = loads_relaxed(line).get("role")
                except (orjson.JSONDecodeError, ValueError, TypeError):
                    continue
                if isinstance(role, str) and not role.startswith("_"):
                    return True
        return False

    # ---- Checkpoints ---- #

    async def create_checkpoint(self, checkpoint_id: int) -> int:
        """Append a checkpoint line and return 0 (unused by caller)."""
        line = orjson.dumps({"role": "_checkpoint", "id": checkpoint_id}).decode("utf-8") + "\n"
        if self._write_handle is not None:
            await self._write_handle.write(line)
            await self._write_handle.flush()
        else:
            async with aiofiles.open(self._file_path, "a", encoding="utf-8") as f:
                await f.write(line)
        # Return a dummy rowid; the caller (Context.checkpoint) ignores it.
        return 0

    async def get_latest_checkpoint_id(self) -> int:
        latest_id = -1
        if not self._file_path.exists():
            return latest_id
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    line_json = loads_relaxed(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(line_json, dict) and line_json.get("role") == "_checkpoint":
                    cpid = line_json.get("id")
                    if isinstance(cpid, int) and cpid > latest_id:
                        latest_id = cpid
        return latest_id

    _CHECKPOINT_USER_PATTERN = re.compile(r"^<system>CHECKPOINT \d+</system>$")

    async def revert_to_checkpoint(self, checkpoint_id: int) -> None:
        """Revert by rewriting the file up to (and excluding) the checkpoint.

        Only keeps valid context lines (messages, system prompt, usage, checkpoint) and
        skips malformed/invalid lines, matching the original ``Context.revert_to()`` behavior.
        """
        # Close the write handle before rewriting to avoid Windows file-lock conflicts
        if self._write_handle is not None:
            await self._write_handle.close()
            self._write_handle = None

        tmp_path = self._file_path.with_suffix(".tmp")
        found = False
        try:
            async with (
                aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as old_file,
                aiofiles.open(tmp_path, "w", encoding="utf-8") as new_file,
            ):
                async for line in old_file:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        line_json = loads_relaxed(stripped)
                    except orjson.JSONDecodeError:
                        continue
                    if not isinstance(line_json, dict):
                        continue
                    role = line_json.get("role")
                    if not isinstance(role, str):
                        # Skip structurally invalid lines (e.g. {"oops": 1})
                        continue
                    if role == "_checkpoint" and line_json.get("id") == checkpoint_id:
                        found = True
                        break
                    await new_file.write(line)
            if not found:
                if tmp_path.exists():
                    await aiofiles.os.unlink(tmp_path)
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            await aiofiles.os.replace(tmp_path, self._file_path)
        except Exception:
            if tmp_path.exists():
                await aiofiles.os.unlink(tmp_path)
            raise
        finally:
            # Reopen the write handle for subsequent append operations
            self._write_handle = await aiofiles.open(self._file_path, mode="a", encoding="utf-8")

    # ---- Usage ---- #

    async def record_usage(self, token_count: int) -> None:
        line = orjson.dumps({"role": "_usage", "token_count": token_count}).decode("utf-8") + "\n"
        if self._write_handle is not None:
            await self._write_handle.write(line)
            await self._write_handle.flush()
        else:
            async with aiofiles.open(self._file_path, "a", encoding="utf-8") as f:
                await f.write(line)

    async def get_latest_usage(self) -> int | None:
        latest = None
        if not self._file_path.exists():
            return None
        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    line_json = loads_relaxed(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(line_json, dict) and line_json.get("role") == "_usage":
                    tc = line_json.get("token_count")
                    if isinstance(tc, int):
                        latest = tc
        return latest

    # ---- Bulk ---- #

    async def clear(self) -> None:
        """Clear the JSONL file (truncate)."""
        # Close existing handle, truncate, then reopen in append mode
        if self._write_handle is not None:
            await self._write_handle.close()
            self._write_handle = None
        async with aiofiles.open(self._file_path, "w", encoding="utf-8") as f:
            pass
        self._write_handle = await aiofiles.open(self._file_path, mode="a", encoding="utf-8")

    async def restore_full(self) -> tuple[str | None, list[Message], int | None, int, list[Message]]:
        """Full restore scanning the entire JSONL file.

        Returns:
            (system_prompt, all_messages, latest_usage, next_checkpoint_id, messages_after_last_usage)
        """
        system_prompt: str | None = None
        messages: list[Message] = []
        messages_after_last_usage: list[Message] = []
        latest_usage: int | None = None
        next_checkpoint_id: int = 0

        if not self._file_path.exists() or self._file_path.stat().st_size == 0:
            return system_prompt, messages, latest_usage, next_checkpoint_id, messages_after_last_usage

        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
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
                    content = line_json.get("content")
                    if isinstance(content, str):
                        system_prompt = content
                elif role == "_usage":
                    tc = line_json.get("token_count")
                    if isinstance(tc, int):
                        latest_usage = tc
                        messages_after_last_usage.clear()
                elif role == "_checkpoint":
                    cpid = line_json.get("id")
                    if isinstance(cpid, int):
                        next_checkpoint_id = cpid + 1
                else:
                    try:
                        message = Message.model_validate(line_json)
                    except ValidationError:
                        continue
                    messages.append(message)
                    messages_after_last_usage.append(message)

        return system_prompt, messages, latest_usage, next_checkpoint_id, messages_after_last_usage

    async def export(self) -> ExportedContext:
        from pydantic import ValidationError

        from kimi_cli.soul.context_records import (
            CheckpointRecord,
            SystemPromptRecord,
            UsageRecord,
        )

        result = ExportedContext()
        if not self._file_path.exists() or self._file_path.stat().st_size == 0:
            return result

        async with aiofiles.open(self._file_path, encoding="utf-8", errors="replace") as f:
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


# ---------------------------------------------------------------------------
# SqliteContextStorage — new SQLite backend
# ---------------------------------------------------------------------------


class SqliteContextStorage:
    """SQLite-based context storage.

    Wraps ``ContextDB`` and implements the ``ContextStorage`` protocol.
    """

    def __init__(self, db_path: Path) -> None:
        from kimi_cli.soul.context_db import ContextDB

        self._db = ContextDB(db_path)

    @property
    def storage_path(self) -> Path:
        return self._db.db_path

    @property
    def _db_instance(self):
        return self._db

    async def initialize(self) -> None:
        await self._db.initialize()

    async def close(self) -> None:
        await self._db.close()

    def close_sync(self) -> None:
        """Synchronously stop the aiosqlite worker thread (process-exit cleanup)."""
        self._db.stop_sync()

    async def get_system_prompt(self) -> str | None:
        return await self._db.get_system_prompt()

    async def set_system_prompt(self, content: str) -> None:
        await self._db.set_system_prompt(content)

    async def append_messages(self, messages: Sequence[Message]) -> None:
        await self._db.append_messages(messages)

    async def get_messages(
        self, *, after_rowid: int = 0, limit: int | None = None
    ) -> list[Message]:
        return await self._db.get_messages(after_rowid=after_rowid, limit=limit)

    async def get_message_count(self) -> int:
        return await self._db.get_message_count()

    async def has_visible_messages(self) -> bool:
        return await self._db.has_visible_messages()

    async def create_checkpoint(self, checkpoint_id: int) -> int:
        return await self._db.create_checkpoint(checkpoint_id)

    async def get_latest_checkpoint_id(self) -> int:
        return await self._db.get_latest_checkpoint_id()

    async def revert_to_checkpoint(self, checkpoint_id: int) -> None:
        await self._db.revert_to_checkpoint(checkpoint_id)

    async def record_usage(self, token_count: int) -> None:
        await self._db.record_usage(token_count)

    async def get_latest_usage(self) -> int | None:
        return await self._db.get_latest_usage()

    async def clear(self) -> None:
        await self._db.clear()

    async def export(self) -> ExportedContext:
        return await self._db.export()

    async def restore_full(self) -> tuple[str | None, list[Message], int | None, int, list[Message]]:
        """Full restore from SQLite.

        Since SQLite stores messages and usage in separate tables, all messages
        are considered "after the last usage" for pending token estimation.
        """
        system_prompt = await self._db.get_system_prompt()
        messages = await self._db.get_messages()
        latest_usage = await self._db.get_latest_usage()
        latest_cp = await self._db.get_latest_checkpoint_id()
        next_checkpoint_id = latest_cp + 1 if latest_cp >= 0 else 0
        # All messages are after the last usage since usage is stored separately
        messages_after_last_usage = list(messages)
        if latest_usage is not None:
            # If usage was recorded, messages that were stored before the last
            # usage should not be counted as pending. But in SQLite we can't
            # distinguish. For correct behavior, we return all messages.
            # Context.restore() will handle this correctly.
            pass
        return system_prompt, messages, latest_usage, next_checkpoint_id, messages_after_last_usage

    # ---- Migration support ---- #

    async def get_db(self):
        """Return the underlying ContextDB instance for migration operations."""
        return self._db


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def _detect_storage_backend(path: Path) -> type[JsonlContextStorage] | type[SqliteContextStorage]:
    """Detect which backend to use for a given path."""
    if path.suffix == ".db":
        return SqliteContextStorage
    return JsonlContextStorage


def _resolve_storage_path(path: Path) -> Path:
    """If path is a .jsonl file, return the corresponding .db path."""
    if path.suffix == ".jsonl":
        return path.with_suffix(".db")
    return path


def _needs_migration(path: Path) -> bool:
    """Check if a JSONL file exists and needs migration to SQLite."""
    jsonl_path = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")
    db_path = path if path.suffix == ".db" else path.with_suffix(".db")
    return jsonl_path.exists() and not db_path.exists()


# ---------------------------------------------------------------------------
# Context — the main orchestrator
# ---------------------------------------------------------------------------


class Context:
    """Conversation context manager.

    Accepts either a ``file_backend`` path (backward-compatible) or a
    ``storage`` instance implementing ``ContextStorage``.
    """

    def __init__(
        self,
        file_backend: Path | None = None,
        on_append: Callable[[Sequence[Message]], None] | None = None,
        model_name: str | None = None,
        storage: ContextStorage | None = None,
    ):
        if storage is not None:
            self._storage = storage
        elif file_backend is not None:
            self._storage = self._auto_create_storage(file_backend)
        else:
            raise ValueError("Either file_backend or storage must be provided")

        self._history: list[Message] = []
        self._token_count: int = 0
        self._pending_token_estimate: int = 0
        self._next_checkpoint_id: int = 0
        """The ID of the next checkpoint, starting from 0, incremented after each checkpoint."""
        self._system_prompt: str | None = None
        self._on_append = on_append
        self._model_name = model_name

    @staticmethod
    def _auto_create_storage(path: Path) -> ContextStorage:
        """Auto-detect and create the appropriate storage backend."""
        cls = _detect_storage_backend(path)
        return cls(path)

    @classmethod
    def from_file_backend(cls, path: Path, **kwargs: Any) -> Context:
        """Backward-compatible factory: creates Context from a file path.

        Auto-detects the storage backend based on the path suffix.
        """
        return cls(file_backend=path, **kwargs)

    @classmethod
    def from_db(cls, db_path: Path, **kwargs: Any) -> Context:
        """Factory for creating a Context backed by SQLite.

        Args:
            db_path: Path to the SQLite database file.
            **kwargs: Additional arguments passed to Context.__init__.

        Returns:
            A Context instance backed by SqliteContextStorage.
        """
        storage = SqliteContextStorage(db_path)
        return cls(storage=storage, **kwargs)

    @property
    def storage(self) -> ContextStorage:
        return self._storage

    @property
    def history(self) -> Sequence[Message]:
        return self._history

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def token_count_with_pending(self) -> int:
        return self._token_count + self._pending_token_estimate

    @property
    def n_checkpoints(self) -> int:
        return self._next_checkpoint_id

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def file_backend(self) -> Path:
        """Return the storage path (backward-compatible property)."""
        if hasattr(self._storage, "storage_path"):
            return self._storage.storage_path
        return Path(".")

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str | None) -> None:
        self._model_name = value

    # ------------------------------------------------------------------ #
    # Restore
    # ------------------------------------------------------------------ #

    async def restore(self) -> bool:
        """Restore context from storage.

        Returns True if context was restored, False if empty/missing.
        """
        logger.debug("Restoring context from storage: {storage}", storage=self.file_backend)

        if self._history:
            logger.error("The context storage is already modified")
            raise RuntimeError("The context storage is already modified")

        # Initialize storage
        await self._storage.initialize()

        # Full restore via storage backend (handles pending estimate correctly)
        system_prompt, messages, latest_usage, next_checkpoint_id, messages_after_last_usage = (
            await self._storage.restore_full()
        )

        if system_prompt is not None:
            self._system_prompt = system_prompt

        self._history.extend(messages)

        if latest_usage is not None:
            self._token_count = latest_usage

        if next_checkpoint_id > 0:
            self._next_checkpoint_id = next_checkpoint_id

        # Strip stale system-reminder messages
        from kimi_cli.soul.message import strip_system_reminders

        strip_system_reminders(self._history)

        self._pending_token_estimate = estimate_text_tokens(messages_after_last_usage, model=self._model_name)
        return len(self._history) > 0 or self._system_prompt is not None

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    async def write_system_prompt(self, prompt: str) -> None:
        """Write the system prompt to storage."""
        await self._storage.set_system_prompt(prompt)
        self._system_prompt = prompt

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #

    async def append_message(self, message: Message | Sequence[Message]) -> None:
        """Append one or more messages to the context."""
        logger.debug("Appending message(s) to context: {message}", message=message)
        messages = [message] if isinstance(message, Message) else message
        self._history.extend(messages)
        self._pending_token_estimate += estimate_text_tokens(messages, model=self._model_name)

        if self._on_append is not None:
            self._on_append(messages)

        await self._storage.append_messages(messages)

    async def replace_history(self, messages: Sequence[Message]) -> None:
        """Atomically replace the persisted message history.

        Clears storage and rewrites the system prompt plus the given messages.
        Checkpoints and usage records are reset; callers should re-create a
        checkpoint if checkpoint semantics are required.
        """
        logger.debug(
            "Replacing context history with {count} messages", count=len(messages)
        )
        await self._storage.clear()
        if self._system_prompt is not None:
            await self._storage.set_system_prompt(self._system_prompt)
        if messages:
            await self._storage.append_messages(messages)
        self._history = list(messages)
        new_token_estimate = estimate_text_tokens(messages, model=self._model_name)
        # The persisted usage records were cleared, so the in-memory count can
        # only be safely lowered to the estimate.  A higher value is left for
        # the next API usage update to correct.
        if new_token_estimate < self._token_count:
            self._token_count = new_token_estimate
        self._pending_token_estimate = 0
        self._next_checkpoint_id = 0

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #

    async def checkpoint(self, add_user_message: bool) -> None:
        """Create a checkpoint."""
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        logger.debug("Checkpointing, ID: {id}", id=checkpoint_id)

        await self._storage.create_checkpoint(checkpoint_id)

        if add_user_message:
            await self.append_message(
                Message(role="user", content=[system(f"CHECKPOINT {checkpoint_id}")])
            )

    async def revert_to(self, checkpoint_id: int) -> None:
        """Revert the context to the specified checkpoint."""
        logger.debug("Reverting checkpoint, ID: {id}", id=checkpoint_id)
        if checkpoint_id >= self._next_checkpoint_id:
            logger.error("Checkpoint {checkpoint_id} does not exist", checkpoint_id=checkpoint_id)
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist")

        await self._storage.revert_to_checkpoint(checkpoint_id)

        # Rebuild in-memory state by doing a full restore
        self._history.clear()
        self._token_count = 0
        self._next_checkpoint_id = 0
        self._system_prompt = None

        system_prompt, messages, latest_usage, next_checkpoint_id, messages_after_last_usage = (
            await self._storage.restore_full()
        )

        if system_prompt is not None:
            self._system_prompt = system_prompt

        self._history.extend(messages)

        if latest_usage is not None:
            self._token_count = latest_usage

        if next_checkpoint_id > 0:
            self._next_checkpoint_id = next_checkpoint_id

        self._pending_token_estimate = estimate_text_tokens(messages_after_last_usage, model=self._model_name)

    # ------------------------------------------------------------------ #
    # Token count
    # ------------------------------------------------------------------ #

    async def update_token_count(self, token_count: int) -> None:
        """Record a token count snapshot."""
        logger.debug("Updating token count in context: {token_count}", token_count=token_count)
        self._token_count = token_count
        self._pending_token_estimate = 0
        await self._storage.record_usage(token_count)

    # ------------------------------------------------------------------ #
    # Clear
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the context storage backend.

        This ensures any background resources (e.g. aiosqlite worker threads)
        are properly shut down before process exit.
        """
        logger.debug("Closing context storage")
        await self._storage.close()

    def close_sync(self) -> None:
        """Synchronously stop the storage backend's worker thread.

        Used by synchronous process-exit cleanup (``__del__`` after
        ``KeyboardInterrupt``) where awaiting ``close()`` is impossible.  On
        Windows this releases the SQLite file handles held by the aiosqlite
        worker thread so the session directory can be deleted.
        """
        close_sync = getattr(self._storage, "close_sync", None)
        if close_sync is not None:
            try:
                close_sync()
            except Exception:
                logger.debug("Failed to synchronously close context storage")
                pass

    async def clear(self) -> None:
        """Clear the context history."""
        logger.debug("Clearing context")
        await self._storage.clear()
        self._history.clear()
        self._token_count = 0
        self._pending_token_estimate = 0
        self._next_checkpoint_id = 0
        self._system_prompt = None
