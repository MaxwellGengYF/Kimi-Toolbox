# -*- coding: utf-8 -*-
"""Server-side session manager: creates, tracks, and runs kimix sessions.

Emits opencode-style SSE events that match the protocol specification:
- ID prefixes: ses_, msg_, prt_
- message.part.updated with proper part.type sub-fields
- Tool call lifecycle: pending → running → completed/error
- Reasoning and text use cumulative text + delta pattern
- step-finish carries reason, cost, tokens at part level
"""

from __future__ import annotations
import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import xxhash
from kimi_agent_sdk import Session
from kaos.path import KaosPath
from kimi_cli.soul import RunCancelled

from kimix.server.bus import bus, BusEvent
from kimix.utils import (
    _create_session_async,
    close_session_async,
)
from kimi_cli.session_state import load_session_state
from kosong.utils.jsonx import loads_relaxed

logger = logging.getLogger(__name__)

# ── ID Generators (opencode-style prefixed IDs) ────────────────────

_id_counter_lock = threading.Lock()
_id_counter = 0


def _gen_id(prefix: str) -> str:
    """Generate an opencode-style prefixed ID.

    Format: {prefix}_{hex_timestamp}{random_hex}
    Examples: ses_2325232b2ffe0XLh, msg_dcdae34e30014kj6, prt_dcdae47b2001AnTv
    """
    global _id_counter
    with _id_counter_lock:
        _id_counter += 1
        counter = _id_counter
    ts_hex = format(int(time.time() * 1000), "x")
    cnt_hex = format(counter, "04x")
    # Add some entropy from hash
    raw = f"{ts_hex}{cnt_hex}{os.getpid()}{time.monotonic_ns()}"
    h = xxhash.xxh64(raw.encode()).hexdigest()[:12]
    return f"{prefix}_{ts_hex}{h}"


def _gen_session_id() -> str:
    return _gen_id("ses")


def _gen_message_id() -> str:
    return _gen_id("msg")


def _gen_part_id() -> str:
    return _gen_id("prt")


def _snapshot_hash() -> str:
    """Generate a snapshot hash for step-start/step-finish."""
    raw = f"{time.time()}{time.monotonic_ns()}{os.getpid()}"
    return xxhash.xxh64(raw.encode()).hexdigest()[:16]


def _now_ms() -> int:
    """Current timestamp in milliseconds."""
    return int(time.time() * 1000)


# ── Server feature capabilities (opencode-sse extensions) ──────────
#
# Capability discovery for opencode-sse extension clients. Anything beyond the
# core opencode surface (compact/summarize, interactive permissions, …) is
# advertised here so a client can disable the matching UI when talking to a
# server that does not implement it. Extend this map as new extension features
# land — clients treat a missing key as "disabled".

SERVER_FEATURES: Dict[str, Dict[str, Any]] = {
    "compact": {
        "enabled": True,
        "title": "Compact context",
        "description": "Summarize the conversation to shrink the context window.",
    },
    "abort": {
        "enabled": True,
        "title": "Abort",
        "description": "Cancel the prompt that is currently running.",
    },
    "permissions": {
        "enabled": True,
        "title": "Permissions",
        "description": "Interactive permission prompts for tool calls.",
    },
    "plan": {
        "enabled": True,
        "title": "Plan Mode",
        "description": "Generate implementation plans using a dedicated planner agent.",
    },
}


# ── Data Models ──────────────────────────────────────────────────


@dataclass
class MessagePart:
    """A part of a message (text, tool call, reasoning, etc.)."""

    id: str = ""
    type: str = "text"  # text | tool | reasoning | step-start | step-finish
    sessionID: str = ""
    messageID: str = ""
    # text / reasoning fields
    text: str = ""
    # tool fields
    tool: str = ""
    callID: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    # step-start / step-finish fields
    snapshot: str = ""
    reason: str = ""
    cost: float = 0.0
    prompt: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, Any] = field(default_factory=dict)
    # time tracking
    time: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to opencode-compatible dict for SSE emission."""
        d: Dict[str, Any] = {
            "id": self.id,
            "sessionID": self.sessionID,
            "messageID": self.messageID,
            "type": self.type,
        }
        if self.type == "text":
            d["text"] = self.text
            if self.time:
                d["time"] = self.time
        elif self.type == "reasoning":
            d["text"] = self.text
            if self.time:
                d["time"] = self.time
        elif self.type == "tool":
            d["callID"] = self.callID
            d["tool"] = self.tool
            d["state"] = self.state
        elif self.type == "step-start":
            d["snapshot"] = self.snapshot
        elif self.type == "step-finish":
            d["reason"] = self.reason
            d["snapshot"] = self.snapshot
            if self.cost:
                d["cost"] = self.cost
            if self.prompt:
                d["prompt"] = self.prompt
            if self.tokens:
                d["tokens"] = self.tokens
        return d


@dataclass
class MessageInfo:
    """Message metadata (opencode MessageV2.Info shape)."""

    id: str = ""
    role: str = "assistant"  # user | assistant | system
    sessionID: str = ""
    agent: str = ""
    modelID: str = ""
    providerID: str = ""
    mode: str = ""
    time: Dict[str, Any] = field(default_factory=dict)
    parentID: str = ""
    path: Dict[str, str] = field(default_factory=dict)
    cost: float = 0.0
    tokens: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "sessionID": self.sessionID,
            "role": self.role,
            "time": self.time or {"created": _now_ms()},
        }
        if self.role == "assistant":
            # opencode AssistantMessage requires these fields.
            d["parentID"] = self.parentID
            d["modelID"] = self.modelID
            d["providerID"] = self.providerID
            d["mode"] = self.mode or "chat"
            d["agent"] = self.agent
            d["path"] = self.path or {"cwd": "", "root": ""}
            d["cost"] = self.cost
            d["tokens"] = self.tokens or {
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            }
        else:
            d["agent"] = self.agent
            if self.modelID or self.providerID:
                d["model"] = {
                    "providerID": self.providerID,
                    "modelID": self.modelID,
                }
        return d


@dataclass
class MessageWithParts:
    info: MessageInfo = field(default_factory=MessageInfo)
    parts: List[MessagePart] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "info": self.info.to_dict(),
            "parts": [p.to_dict() for p in self.parts],
        }


@dataclass
class SessionInfo:
    """Public session info exposed via API (opencode Session.Info shape)."""

    id: str = ""
    title: str = ""
    projectID: str = "kimix"
    directory: str = ""
    version: str = "0.1.0"
    createdAt: int = 0  # unix ms
    updatedAt: int = 0  # unix ms
    parentID: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "projectID": self.projectID,
            "directory": self.directory,
            "title": self.title,
            "version": self.version,
            "time": {
                "created": self.createdAt,
                "updated": self.updatedAt,
            },
        }
        if self.parentID:
            d["parentID"] = self.parentID
        return d


@dataclass
class SessionStatus:
    type: str = "idle"  # idle | busy | retry
    time: float = 0.0
    token_count: int = 0
    context_usage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "token_count": self.token_count,
            "context_usage": self.context_usage,
        }


# ── Managed Session Entry ────────────────────────────────────────


@dataclass
class ManagedSession:
    """Internal session entry tracked by the manager."""

    info: SessionInfo
    sdk_session: Optional[Session] = None
    work_dir: Optional[str] = None
    status: SessionStatus = field(default_factory=SessionStatus)
    messages: List[MessageWithParts] = field(default_factory=list)
    _cancel_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _planner_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _planner_session: Optional[Session] = field(default=None, repr=False)
    _plan_confirm_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _plan_feedback: Optional[str] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# ── Event Helper ─────────────────────────────────────────────────


def _emit_part_to_message(
    asst_msg: MessageWithParts,
    session_id: str,
    part: MessagePart,
    delta: str = "",
) -> None:
    """Emit a message.part.updated SSE event and update the assistant message parts list.

    Replaces an existing part with the same ID to prevent duplicate accumulation
    of incremental chunks (reasoning, text).
    """
    found = False
    for i, existing in enumerate(asst_msg.parts):
        if existing.id == part.id:
            asst_msg.parts[i] = part
            found = True
            break
    if not found:
        asst_msg.parts.append(part)
    props: Dict[str, Any] = {"part": part.to_dict()}
    if delta:
        props["delta"] = delta
    bus.emit(BusEvent(type="message.part.updated", properties=props))


# ── Session Manager ──────────────────────────────────────────────


class SessionManager:
    """Manages all sessions for the kimix serve process."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ManagedSession] = {}
        self._lock = threading.Lock()

    def _current_work_dir(self) -> KaosPath:
        return KaosPath.unsafe_from_local_path(Path(os.getcwd()))

    async def _refresh_disk_sessions(self) -> None:
        """Index persisted sessions for the current work directory."""
        try:
            from kimi_cli.session import Session as CliSession

            disk_sessions = await CliSession.list(self._current_work_dir())
        except Exception:
            logger.debug("Error listing persisted sessions", exc_info=True)
            return

        with self._lock:
            existing = set(self._sessions)

        for disk_session in disk_sessions:
            if disk_session.id in existing:
                continue
            context_file = disk_session.context_file
            created_at = self._mtime_ms(context_file)
            updated_at = (
                int(disk_session.updated_at * 1000) if disk_session.updated_at else created_at
            )
            entry = ManagedSession(
                info=SessionInfo(
                    id=disk_session.id,
                    title=disk_session.title or f"Session {disk_session.id[:12]}",
                    directory=str(disk_session.work_dir),
                    createdAt=created_at,
                    updatedAt=updated_at,
                ),
                sdk_session=None,
                work_dir=str(disk_session.work_dir),
                messages=self._load_context_messages(disk_session.id, context_file),
            )
            with self._lock:
                self._sessions.setdefault(disk_session.id, entry)

    async def _ensure_sdk_session(self, entry: ManagedSession) -> Session:
        """Resume a persisted SDK session only when it is actually used."""
        if entry.sdk_session is not None:
            return entry.sdk_session

        if not entry.work_dir:
            raise ValueError(f"Session {entry.info.id} has no active SDK session")

        work_dir = KaosPath.unsafe_from_local_path(Path(entry.work_dir))
        try:
            from kimi_cli.session import Session as CliSession

            disk_session = await CliSession.find(work_dir, entry.info.id)
        except Exception as exc:
            raise ValueError(f"Session {entry.info.id} could not be restored") from exc
        if disk_session is None:
            raise ValueError(f"Session {entry.info.id} could not be restored")

        restored = await _create_session_async(
            session_id=entry.info.id,
            work_dir=work_dir,
            resume=True,
        )
        if restored is None:
            raise ValueError(f"Session {entry.info.id} could not be restored")
        entry.sdk_session = restored
        return restored

    async def search_messages(
        self,
        session_id: str,
        query: str,
        *,
        role: str | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 full-text search over a persisted session's ``context.db``.

        Returns the ``ContextDB.search_messages`` result shape
        (``{rowid, role, content, content_text, created_at, snippet, score}``)
        or ``[]`` when the session is unknown / unsearchable.
        """
        entry = self._sessions.get(session_id)
        if entry is None or not entry.work_dir:
            return []
        try:
            from kimi_cli.session import Session as CliSession
            from kimi_cli.soul.context_db import ContextDB

            work_dir = KaosPath.unsafe_from_local_path(Path(entry.work_dir))
            disk_session = await CliSession.find(work_dir, session_id)
            if disk_session is None:
                return []
            db = ContextDB(disk_session.context_db_file)
            try:
                await db.initialize()
                return await db.search_messages(
                    query,
                    role=role,
                    limit=limit,
                    offset=offset,
                    sort=sort,
                )
            finally:
                await db.close()
        except Exception:
            logger.exception(
                "Failed to search session {session_id} messages", session_id=session_id
            )
            return []

    def _load_context_messages(self, session_id: str, context_file: Path) -> List[MessageWithParts]:
        messages: List[MessageWithParts] = []
        created = self._mtime_ms(context_file)

        # Try SQLite backend first (for .db files)
        if context_file.suffix == ".db" and context_file.exists():
            try:
                import apsw

                conn = apsw.Connection(str(context_file))
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT role, content FROM messages WHERE role NOT IN "
                        "('_system_prompt', '_usage', '_checkpoint') "
                        "ORDER BY rowid DESC LIMIT 100"
                    )
                    rows = list(cursor)

                    for index, row in enumerate(reversed(rows)):
                        role = row[0]
                        content_data = loads_relaxed(row[1])
                        content = self._context_content_text(content_data)
                        if not content:
                            continue

                        message_id = f"msg_restored_{index}"
                        messages.append(
                            MessageWithParts(
                                info=MessageInfo(
                                    id=message_id,
                                    role=role if role in {"user", "assistant", "system"} else "assistant",
                                    sessionID=session_id,
                                    time={"created": created},
                                ),
                                parts=[TextPart(content=content)],
                            )
                        )
                finally:
                    conn.close()
            except Exception:
                logger.exception("Failed to load context messages from SQLite")
            return messages

        # Fallback to JSONL parsing
        try:
            lines = context_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return messages

        for index, line in enumerate(lines[-100:]):
            if not line.strip():
                continue
            try:
                data = loads_relaxed(line)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            role = data.get("role")
            if not isinstance(role, str) or role.startswith("_"):
                continue
            content = self._context_content_text(data.get("content"))
            if not content:
                continue

            message_id = f"msg_restored_{index}"
            messages.append(
                MessageWithParts(
                    info=MessageInfo(
                        id=message_id,
                        role=role if role in {"user", "assistant", "system"} else "assistant",
                        sessionID=session_id,
                        time={"created": created},
                    ),
                    parts=[
                        MessagePart(
                            id=f"prt_restored_{index}",
                            type="text",
                            text=content,
                            sessionID=session_id,
                            messageID=message_id,
                        )
                    ],
                )
            )
        return messages

    def _context_content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    value = part.get("text") or part.get("content")
                    if isinstance(value, str):
                        text_parts.append(value)
            if text_parts:
                return "\n".join(text_parts)
        if content is None:
            return ""
        try:
            return orjson.dumps(content).decode("utf-8")
        except Exception:
            return str(content)

    def _mtime_ms(self, path: Path) -> int:
        try:
            return int(path.stat().st_mtime * 1000)
        except OSError:
            return _now_ms()

    # ── Session CRUD ─────────────────────────────────────────────

    async def create_session(
        self,
        title: Optional[str] = None,
        parent_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> SessionInfo:
        session_id = _gen_session_id()
        now = _now_ms()
        info = SessionInfo(
            id=session_id,
            title=title or f"Session {session_id[:12]}",
            directory=os.getcwd(),
            createdAt=now,
            updatedAt=now,
            parentID=parent_id,
        )
        sdk_session = await _create_session_async(session_id=session_id)
        entry = ManagedSession(
            info=info,
            sdk_session=sdk_session,
            work_dir=os.getcwd(),
        )
        with self._lock:
            self._sessions[session_id] = entry

        bus.emit_type(
            "session.created", info=info.to_dict()
        )
        logger.info("[SessionManager] Created session %s", session_id)
        return info

    async def get_session(self, session_id: str) -> SessionInfo:
        entry = await self._get_entry_or_restore(session_id)
        return entry.info

    def get_sdk_session(self, session_id: str) -> Optional[Session]:
        entry = self._get_entry(session_id)
        return entry.sdk_session

    async def list_sessions(self) -> List[SessionInfo]:
        await self._refresh_disk_sessions()
        with self._lock:
            entries = list(self._sessions.values())
        return sorted(
            [e.info for e in entries],
            key=lambda s: s.updatedAt,
            reverse=True,
        )

    async def delete_session(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None:
            await self._refresh_disk_sessions()
            with self._lock:
                entry = self._sessions.pop(session_id, None)
            if entry is None:
                return False
        if entry.sdk_session:
            try:
                await close_session_async(entry.sdk_session)
            except Exception:
                logger.debug("Error closing sdk session", exc_info=True)
        if entry.work_dir:
            try:
                from kimi_cli.session import Session as CliSession

                disk_session = await CliSession.find(
                    KaosPath.unsafe_from_local_path(Path(entry.work_dir)),
                    session_id,
                )
                if disk_session is not None:
                    await disk_session.delete()
            except Exception:
                logger.debug("Error deleting persisted session", exc_info=True)
        bus.emit_type("session.deleted", info=entry.info.to_dict())
        logger.info("[SessionManager] Deleted session %s", session_id)
        return True

    def get_session_status(self, session_id: str) -> SessionStatus:
        """Get status for a specific session.

        Returns :class:`SessionStatus` for the given session ID.
        Raises KeyError if the session does not exist.
        """
        return self._get_entry(session_id).status

    def get_all_session_statuses(self) -> Dict[str, Dict[str, Any]]:
        # opencode only tracks non-idle sessions in the status map.
        with self._lock:
            return {
                sid: entry.status.to_dict()
                for sid, entry in self._sessions.items()
                if entry.status.type != "idle"
            }

    def get_todos(self, session_id: str) -> List[Dict[str, Any]]:
        entry = self._get_entry(session_id)
        sdk_session = entry.sdk_session
        if sdk_session is None:
            return []
        try:
            cli = getattr(sdk_session, "_cli", None)
            if cli is not None and cli.session is not None:
                todos = load_session_state(cli.session.dir).todos
                return [
                    {
                        "id": getattr(t, "id", str(i)),
                        "content": getattr(t, "content", ""),
                        "status": getattr(t, "status", "pending"),
                        "priority": getattr(t, "priority", "medium"),
                    }
                    for i, t in enumerate(todos)
                ]
        except Exception:
            logger.debug("Error loading todos", exc_info=True)
        return []

    # ── Messages ─────────────────────────────────────────────────

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        entry = await self._get_entry_or_restore(session_id)
        msgs = entry.messages
        if limit and limit > 0:
            msgs = msgs[-limit:]
        return [m.to_dict() for m in msgs]

    # ── Bootstrap / instance metadata ────────────────────────────

    def _model_ref(self) -> Dict[str, str]:
        """Resolve the configured (providerID, modelID) pair."""
        try:
            from kimix.utils.config import _create_config

            cfg, _provider_dict = _create_config(None)
            provider_id = cfg.provider.type if cfg.provider is not None else "kimix"
            model_id = cfg.model.model if cfg.model is not None else "unknown"
            return {"providerID": provider_id, "modelID": model_id}
        except Exception:
            logger.debug("Error resolving model ref", exc_info=True)
            return {"providerID": "kimix", "modelID": "unknown"}

    def get_config(self) -> Dict[str, Any]:
        """Return an opencode-compatible Config object."""
        ref = self._model_ref()
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": f"{ref['providerID']}/{ref['modelID']}",
            "username": os.environ.get("USER") or os.environ.get("USERNAME") or "user",
        }

    def get_features(self) -> Dict[str, Any]:
        """Return advertised server capabilities for opencode-sse extensions.

        Shape: ``{"features": {name: {enabled, title, description}}}``.
        Clients should disable any extension UI whose feature is missing here
        or reported with ``enabled == false``.
        """
        return {"features": {k: dict(v) for k, v in SERVER_FEATURES.items()}}

    def get_providers(self) -> Dict[str, Any]:
        """Return opencode ConfigProvidersResponse: {providers, default}."""
        ref = self._model_ref()
        provider = {
            "id": ref["providerID"],
            "name": ref["providerID"],
            "env": [],
            "models": {
                ref["modelID"]: {
                    "id": ref["modelID"],
                    "name": ref["modelID"],
                    "release_date": "",
                    "attachment": False,
                    "reasoning": True,
                    "temperature": True,
                    "tool_call": True,
                    "cost": {"input": 0, "output": 0},
                    "limit": {"context": 0, "output": 0},
                    "options": {},
                }
            },
        }
        return {
            "providers": [provider],
            "default": {ref["providerID"]: ref["modelID"]},
        }

    def get_project(self) -> Dict[str, Any]:
        """Return opencode Project.Info for the current working directory."""
        cwd = os.getcwd()
        return {
            "id": "kimix",
            "worktree": cwd,
            "vcs": "git" if os.path.isdir(os.path.join(cwd, ".git")) else None,
            "time": {"created": _now_ms(), "initialized": _now_ms()},
        }

    def get_path(self) -> Dict[str, Any]:
        """Return opencode PathInfo: {home, state, config, worktree, directory}."""
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        return {
            "home": home,
            "state": os.path.join(home, ".kimi"),
            "config": os.path.join(home, ".kimi"),
            "worktree": cwd,
            "directory": cwd,
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return the list of available agents in opencode Agent shape."""
        ref = self._model_ref()
        return [
            {
                "name": "chat",
                "description": "Default kimix agent",
                "mode": "primary",
                "builtIn": True,
                "model": ref,
                "tools": {},
                "options": {},
            }
        ]

    # ── Prompt (sync wait) ───────────────────────────────────────


    async def prompt(
        self,
        session_id: str,
        text: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a prompt and wait for the full response.

        Emits opencode-compatible SSE events:
        - step-start / step-finish with snapshot, reason, cost, tokens
        - reasoning with cumulative text + delta + time
        - text with cumulative text + delta + time
        - tool with pending → running → completed lifecycle + callID
        """
        # Lazy imports of wire message types
        from kimi_cli.wire.types import (
            ApprovalRequest,
            ContentPart,
            StepBegin,
            StepInterrupted,
            TextPart,
            ThinkPart,
            ToolCall,
            ToolCallPart,
            ToolResult,
        )

        entry = await self._get_entry_or_restore(session_id)
        sdk_session = await self._ensure_sdk_session(entry)

        self._set_status(entry, "busy")
        entry._cancel_event = None

        # ── Create user message ──────────────────────────────────
        user_msg_id = _gen_message_id()
        now_ms = _now_ms()
        user_msg = MessageWithParts(
            info=MessageInfo(
                id=user_msg_id,
                role="user",
                sessionID=session_id,
                agent=agent or "",
                time={"created": now_ms},
            ),
            parts=[
                MessagePart(
                    id=_gen_part_id(),
                    type="text",
                    text=text,
                    sessionID=session_id,
                    messageID=user_msg_id,
                )
            ],
        )
        entry.messages.append(user_msg)
        bus.emit_type(
            "message.updated",
            info=user_msg.info.to_dict(),
        )

        # ── Create assistant message placeholder ─────────────────
        asst_msg_id = _gen_message_id()
        _ref = self._model_ref()
        _cwd = os.getcwd()
        asst_msg = MessageWithParts(
            info=MessageInfo(
                id=asst_msg_id,
                role="assistant",
                sessionID=session_id,
                agent=agent or "",
                parentID=user_msg_id,
                modelID=_ref["modelID"],
                providerID=_ref["providerID"],
                mode="chat",
                path={"cwd": _cwd, "root": _cwd},
                time={"created": _now_ms()},
            ),
            parts=[],
        )
        entry.messages.append(asst_msg)

        # ── Event emission helper ────────────────────────────────
        def _emit_part(part: MessagePart, delta: str = "") -> None:
            # Replace existing part with same ID instead of always appending.
            # This prevents duplicate accumulation of incremental chunks (reasoning, text).
            found = False
            for i, existing in enumerate(asst_msg.parts):
                if existing.id == part.id:
                    asst_msg.parts[i] = part
                    found = True
                    break
            if not found:
                asst_msg.parts.append(part)
            props: Dict[str, Any] = {"part": part.to_dict()}
            if delta:
                props["delta"] = delta
            bus.emit(BusEvent(type="message.part.updated", properties=props))

        # ── Emit initial step-start ──────────────────────────────
        step_snapshot = _snapshot_hash()
        _emit_part(
            MessagePart(
                id=_gen_part_id(),
                type="step-start",
                sessionID=session_id,
                messageID=asst_msg_id,
                snapshot=step_snapshot,
            )
        )

        # ── Accumulation state ───────────────────────────────────
        # Text: cumulative buffer
        text_buf: List[str] = []
        text_part_id = _gen_part_id()
        text_time_start: Optional[int] = None

        # Reasoning: cumulative buffer
        reasoning_buf: List[str] = []
        reasoning_part_id = _gen_part_id()
        reasoning_time_start: Optional[int] = None

        # Tool tracking: wire_tool_call_id → (part_id, tool_name)
        active_tool_parts: Dict[str, tuple[str, str]] = {}

        # Metrics accumulation
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0

        error_msg: Optional[str] = None
        step_finish_reason = "stop"

        def _flush_reasoning() -> None:
            """Flush accumulated reasoning with time.end."""
            nonlocal reasoning_buf, reasoning_part_id, reasoning_time_start
            if reasoning_buf and reasoning_time_start:
                full = "".join(reasoning_buf)
                _emit_part(
                    MessagePart(
                        id=reasoning_part_id,
                        type="reasoning",
                        text=full,
                        sessionID=session_id,
                        messageID=asst_msg_id,
                        time={"start": reasoning_time_start, "end": _now_ms()},
                    )
                )
            reasoning_buf = []
            reasoning_part_id = _gen_part_id()
            reasoning_time_start = None

        def _flush_text() -> None:
            """Flush accumulated text with time.end."""
            nonlocal text_buf, text_part_id, text_time_start
            if text_buf and text_time_start:
                full = "".join(text_buf)
                _emit_part(
                    MessagePart(
                        id=text_part_id,
                        type="text",
                        text=full,
                        sessionID=session_id,
                        messageID=asst_msg_id,
                        time={"start": text_time_start, "end": _now_ms()},
                    )
                )
            text_buf = []
            text_part_id = _gen_part_id()
            text_time_start = None

        def _emit_step_finish(reason: str) -> None:
            """Emit a step-finish event."""
            nonlocal step_snapshot
            _emit_part(
                MessagePart(
                    id=_gen_part_id(),
                    type="step-finish",
                    sessionID=session_id,
                    messageID=asst_msg_id,
                    reason=reason,
                    snapshot=step_snapshot,
                    cost=total_cost,
                    prompt={"total": total_input_tokens + total_output_tokens},
                    tokens={
                        "input": total_input_tokens,
                        "output": total_output_tokens,
                        "reasoning": total_reasoning_tokens,
                        "cache": {
                            "read": total_cache_read,
                            "write": total_cache_write,
                        },
                    },
                )
            )

        try:
            text = text.strip()
            if text.startswith("/"):
                task = text[1:]
                split_idx = task.find(":")
                if split_idx >= 0:
                    cmd_name = task[:split_idx]
                    cmd_args = task[split_idx + 1 :].split(":")
                else:
                    cmd_name = task
                    cmd_args = []
                try:
                    handler = getattr(self, "_cmd_" + cmd_name, self._cmd_unknown)
                    new_text, response = await handler(session_id, cmd_args)
                except:
                    response = None
                    pass
                response = None
                if response is not None:
                    _emit_part(
                        MessagePart(
                            id=_gen_part_id(),
                            type="text",
                            text=orjson.dumps(response).decode("utf-8"),
                            sessionID=session_id,
                            messageID=asst_msg_id,
                        )
                    )
                    asst_msg.info.time["completed"] = _now_ms()
                    entry.info.updatedAt = _now_ms()
                    bus.emit_type(
                        "message.updated",
                        info=asst_msg.info.to_dict(),
                    )
                    return asst_msg.to_dict()
                if new_text is not None:
                    text = new_text.strip()
                else:
                    asst_msg.info.time["completed"] = _now_ms()
                    entry.info.updatedAt = _now_ms()
                    bus.emit_type(
                        "message.updated",
                        info=asst_msg.info.to_dict(),
                    )
                    return asst_msg.to_dict()

            ralph_count = 0
            try:
                ralph_count = sdk_session._cli._runtime.config.loop_control.max_ralph_iterations or 0
            except AttributeError:
                pass

            if ralph_count < 0:
                loop_iter = iter(int, 1)
            elif ralph_count > 0:
                loop_iter = range(ralph_count + 1)
            else:
                loop_iter = range(1)

            for _ in loop_iter:
                if entry._cancel_event is not None and entry._cancel_event.is_set():
                    break
                async for wire_msg in sdk_session.prompt(text, merge_wire_messages=True):
                    # ── ApprovalRequest: auto-approve in server mode ─
                    if isinstance(wire_msg, ApprovalRequest):
                        logger.info(
                            "[SessionManager] Auto-approving: %s (%s)",
                            wire_msg.action,
                            wire_msg.description,
                        )
                        wire_msg.resolve("approve")
                        continue

                    # ── StepBegin: new step boundary ─────────────────
                    if isinstance(wire_msg, StepBegin):
                        _flush_reasoning()
                        _flush_text()
                        _emit_step_finish("tool-calls")

                        # Inter-step events
                        asst_msg.info.time["completed"] = _now_ms()
                        bus.emit_type(
                            "message.updated",
                            info=asst_msg.info.to_dict(),
                        )
                        bus.emit_type(
                            "session.status",
                            sessionID=session_id,
                            status={"type": "busy"},
                        )

                        # New step
                        step_snapshot = _snapshot_hash()
                        _emit_part(
                            MessagePart(
                                id=_gen_part_id(),
                                type="step-start",
                                sessionID=session_id,
                                messageID=asst_msg_id,
                                snapshot=step_snapshot,
                            )
                        )
                        step_finish_reason = "stop"
                        continue

                    # ── StepInterrupted ──────────────────────────────
                    if isinstance(wire_msg, StepInterrupted):
                        step_finish_reason = "tool-calls"
                        continue

                    # ── ThinkPart (reasoning) — cumulative ───────────
                    if isinstance(wire_msg, ThinkPart):
                        chunk = wire_msg.think
                        if chunk:
                            if reasoning_time_start is None:
                                reasoning_time_start = _now_ms()
                            reasoning_buf.append(chunk)
                            full_so_far = "".join(reasoning_buf)
                            _emit_part(
                                MessagePart(
                                    id=reasoning_part_id,
                                    type="reasoning",
                                    text=full_so_far,
                                    sessionID=session_id,
                                    messageID=asst_msg_id,
                                    time={"start": reasoning_time_start},
                                ),
                                delta=chunk,
                            )
                        continue

                    # ── TextPart — cumulative ────────────────────────
                    if isinstance(wire_msg, TextPart):
                        chunk = wire_msg.text
                        if chunk:
                            # Flush reasoning first if we're switching to text
                            if reasoning_buf:
                                _flush_reasoning()

                            if text_time_start is None:
                                text_time_start = _now_ms()
                            text_buf.append(chunk)
                            full_so_far = "".join(text_buf)
                            _emit_part(
                                MessagePart(
                                    id=text_part_id,
                                    type="text",
                                    text=full_so_far,
                                    sessionID=session_id,
                                    messageID=asst_msg_id,
                                    time={"start": text_time_start},
                                ),
                                delta=chunk,
                            )
                        continue

                    # ── ToolCall: pending → running ──────────────────
                    if isinstance(wire_msg, ToolCall):
                        # Flush reasoning before tool calls
                        if reasoning_buf:
                            _flush_reasoning()

                        tool_name = (
                            wire_msg.function.name if wire_msg.function else "unknown"
                        )
                        tool_args_raw = (
                            wire_msg.function.arguments if wire_msg.function else ""
                        )
                        wire_tc_id = wire_msg.id or _gen_id("toolu")
                        tool_part_id = _gen_part_id()
                        active_tool_parts[wire_tc_id] = (tool_part_id, tool_name)

                        # Phase 1: pending
                        _emit_part(
                            MessagePart(
                                id=tool_part_id,
                                type="tool",
                                tool=tool_name,
                                callID=wire_tc_id,
                                sessionID=session_id,
                                messageID=asst_msg_id,
                                state={
                                    "status": "pending",
                                    "input": {},
                                    "raw": "",
                                },
                            )
                        )

                        # Phase 2: running (parse input args)
                        parsed_input: Any = {}
                        if tool_args_raw:
                            try:
                                parsed_input = orjson.loads(tool_args_raw)
                            except (orjson.JSONDecodeError, TypeError):
                                parsed_input = {"raw": str(tool_args_raw)}

                        _emit_part(
                            MessagePart(
                                id=tool_part_id,
                                type="tool",
                                tool=tool_name,
                                callID=wire_tc_id,
                                sessionID=session_id,
                                messageID=asst_msg_id,
                                state={
                                    "status": "running",
                                    "input": parsed_input,
                                    "time": {"start": _now_ms()},
                                },
                            )
                        )
                        continue

                    # ── ToolCallPart: streaming argument chunks ──────
                    if isinstance(wire_msg, ToolCallPart):
                        # Incremental argument chunk — we already emitted running
                        continue

                    # ── ToolResult: completed / error ────────────────
                    if isinstance(wire_msg, ToolResult):
                        tc_id = wire_msg.tool_call_id
                        tool_part_id, tool_name = active_tool_parts.pop(
                            tc_id, (_gen_part_id(), "")
                        )
                        rv = wire_msg.return_value
                        is_error = rv.is_error

                        # Extract output text
                        output = ""
                        if isinstance(rv.output, str):
                            output = rv.output
                        elif isinstance(rv.output, list):
                            parts_text = []
                            for cp in rv.output:
                                if isinstance(cp, TextPart):
                                    parts_text.append(cp.text)
                                else:
                                    parts_text.append(f"[{type(cp).__name__}]")
                            output = "".join(parts_text)
                        if not output and rv.message:
                            output = rv.message

                        status = "error" if is_error else "completed"
                        state: Dict[str, Any] = {
                            "status": status,
                            "input": {},  # input was already sent in running phase
                        }
                        if is_error:
                            state["error"] = rv.message or output[:4000]
                        else:
                            state["output"] = output[:4000]

                        _emit_part(
                            MessagePart(
                                id=tool_part_id,
                                type="tool",
                                tool=tool_name,
                                callID=tc_id,
                                sessionID=session_id,
                                messageID=asst_msg_id,
                                state=state,
                            )
                        )
                        continue

                    # ── Other ContentPart subtypes (images etc.) ─────
                    if isinstance(wire_msg, ContentPart):
                        try:
                            raw = wire_msg.model_dump()
                            part_type_str = raw.get("type", "unknown")
                            part_text = orjson.dumps(raw).decode("utf-8")
                            if text_time_start is None:
                                text_time_start = _now_ms()
                            text_buf.append(f"[{part_type_str}] {part_text}")
                            full_so_far = "".join(text_buf)
                            _emit_part(
                                MessagePart(
                                    id=text_part_id,
                                    type="text",
                                    text=full_so_far,
                                    sessionID=session_id,
                                    messageID=asst_msg_id,
                                    time={"start": text_time_start},
                                ),
                                delta=part_text,
                            )
                        except Exception:
                            pass
                        continue

                    if ralph_count != 0:
                        todos = []
                        try:
                            if hasattr(sdk_session, '_cli') and sdk_session._cli is not None and sdk_session._cli.session is not None:
                                todos = load_session_state(sdk_session._cli.session.dir).todos
                        except Exception:
                            pass

                        if not todos or all(todo.status == 'done' for todo in todos):
                            break
        except (asyncio.CancelledError, RunCancelled):
            error_msg = "cancelled"
        except Exception as exc:
            error_msg = str(exc)
            logger.error(
                "[SessionManager] Prompt error: %s", exc, exc_info=True
            )
        finally:
            # Finalize the turn BEFORE signaling idle. The terminal
            # `session.idle` event must be the LAST thing the client sees:
            # if we flush the trailing text/step-finish parts *after* idle,
            # the webview receives idle (stops "busy"), then a late
            # message.part.updated re-opens the busy state with no following
            # idle — leaving the UI stuck "running" forever.
            _flush_reasoning()
            _flush_text()

            reason = error_msg or step_finish_reason
            _emit_step_finish(reason)

            asst_msg.info.time["completed"] = _now_ms()
            entry.info.updatedAt = _now_ms()
            entry._cancel_event = None

            bus.emit_type(
                "message.updated",
                info=asst_msg.info.to_dict(),
            )

            # Idle last — terminal event for this turn.
            self._set_status(entry, "idle")

        return asst_msg.to_dict()

    # ── Prompt Async (fire-and-forget) ───────────────────────────

    async def prompt_async(
        self,
        session_id: str,
        text: str,
        agent: Optional[str] = None,
    ) -> None:
        """Fire-and-forget: start prompt in background, events via SSE."""
        entry = await self._get_entry_or_restore(session_id)
        await self._ensure_sdk_session(entry)

        async def _run() -> None:
            try:
                await self.prompt(session_id, text, agent)
            except Exception as exc:
                logger.error(
                    "[SessionManager] prompt_async error: %s",
                    exc,
                    exc_info=True,
                )
                try:
                    entry = self._get_entry(session_id)
                    self._set_status(entry, "idle")
                except Exception:
                    pass

        asyncio.ensure_future(_run())

    # ── Abort ────────────────────────────────────────────────────

    def abort_session(self, session_id: str) -> bool:
        entry = self._get_entry(session_id)
        if entry._cancel_event:
            entry._cancel_event.set()
        if entry.sdk_session:
            try:
                entry.sdk_session.cancel()
            except Exception:
                pass
        # Also cancel the planner session if running
        if entry._planner_session:
            try:
                entry._planner_session.cancel()
            except Exception:
                pass
        if entry._planner_task and not entry._planner_task.done():
            entry._planner_task.cancel()
        self._set_status(entry, "idle")
        return True

    # ── Session Operations ───────────────────────────────────────

    async def clear_session(self, session_id: str) -> bool:
        """Clear a session's SDK state and local message history."""
        entry = self._get_entry(session_id)
        print('cleared server')
        if entry.sdk_session:
            await entry.sdk_session.clear()
        entry.messages.clear()
        entry.info.updatedAt = _now_ms()
        logger.info("[SessionManager] Cleared session %s", session_id)
        return True

    async def compact_session(self, session_id: str, keep: Optional[int] = None) -> bool:
        """Compact a session by trimming message history."""
        entry = self._get_entry(session_id)
        if entry.sdk_session:
            await entry.sdk_session.compact()
        if keep is not None and len(entry.messages) > keep:
            entry.messages = entry.messages[-keep:]
        entry.info.updatedAt = _now_ms()
        logger.info("[SessionManager] Compacted session %s", session_id)
        return True

    async def export_session(
        self, session_id: str, output_path: Optional[str] = None
    ) -> tuple[str, int]:
        """Export a session to a file. Returns (output_path, message_count)."""
        entry = self._get_entry(session_id)
        if entry.sdk_session is None:
            raise ValueError(f"Session {session_id} has no active SDK session")
        output, count = await entry.sdk_session.export(output_path=output_path)
        logger.info("[SessionManager] Exported session %s to %s", session_id, output)
        return output, count

    async def get_session_context(self, session_id: str, keep: Optional[int] = None) -> Dict[str, Any]:
        """Return context usage for a specific session."""
        entry = self._get_entry(session_id)
        print('cleared server')
        if entry.sdk_session:
            await entry.sdk_session.clear()
        entry.messages.clear()
        entry.info.updatedAt = time.time()
        logger.info("[SessionManager] Cleared session %s", session_id)
        return True
        # entry = self._get_entry(session_id)
        # usage: Any = None
        # if entry.sdk_session:
        #     try:
        #         usage = entry.sdk_session.status.context_usage
        #     except Exception:
        #         pass
        # if keep is not None and len(entry.messages) > keep:
        #     entry.messages = entry.messages[-keep:]
        # entry.messages.clear()
        # entry.info.updatedAt = time.time()
        # logger.info("[SessionManager] Get session  %s", session_id)
        # return {"sessionID": session_id, "context_usage": usage}

    # ── Plan ──────────────────────────────────────────────────

    async def plan_async(self, session_id: str, text: str) -> None:
        """POST /session/{sessionID}/plan

        Fire-and-forget: run planner in background, events via SSE.
        Follows the same design as prompt_async() — creates user/assistant
        messages, emits message.part.updated events, and is abortable.
        """
        entry = await self._get_entry_or_restore(session_id)

        async def _run_plan() -> None:
            from kimix.base import _default_agent_file_dir
            from kimix.tools.note import _set_enable_plan
            from kimix.utils.system_prompt import SystemPromptType

            planner_session = None
            plan_file = Path("plan.md")

            # ── Create user message ──────────────────────────────
            user_msg_id = _gen_message_id()
            now_ms = _now_ms()
            user_msg = MessageWithParts(
                info=MessageInfo(
                    id=user_msg_id,
                    role="user",
                    sessionID=session_id,
                    time={"created": now_ms},
                ),
                parts=[
                    MessagePart(
                        id=_gen_part_id(),
                        type="text",
                        text=text,
                        sessionID=session_id,
                        messageID=user_msg_id,
                    )
                ],
            )
            entry.messages.append(user_msg)
            bus.emit_type(
                "message.updated",
                info=user_msg.info.to_dict(),
            )

            # ── Create assistant message placeholder ─────────────
            asst_msg_id = _gen_message_id()
            asst_msg = MessageWithParts(
                info=MessageInfo(
                    id=asst_msg_id,
                    role="assistant",
                    sessionID=session_id,
                    parentID=user_msg_id,
                    modelID="",
                    providerID="",
                    mode="plan",
                    path={"cwd": os.getcwd(), "root": os.getcwd()},
                    time={"created": _now_ms()},
                ),
                parts=[],
            )
            entry.messages.append(asst_msg)

            # ── Emit initial step-start ──────────────────────────
            step_snapshot = _snapshot_hash()
            _emit_part_to_message(
                asst_msg,
                session_id,
                MessagePart(
                    id=_gen_part_id(),
                    type="step-start",
                    sessionID=session_id,
                    messageID=asst_msg_id,
                    snapshot=step_snapshot,
                ),
            )

            # ── Accumulation state ───────────────────────────────
            text_buf: List[str] = []
            text_part_id = _gen_part_id()
            step_finish_reason = "stop"
            plan_generated = False

            try:
                self._set_status(entry, "busy")
                entry._cancel_event = None

                if plan_file.is_file():
                    plan_file.unlink()

                # Create a planner SDK session
                planner_session = await _create_session_async(
                    agent_type=SystemPromptType.TodoMaker,
                    agent_file=_default_agent_file_dir / "agent_planner.json",
                )
                planner_session.get_custom_data()["plan_writing_path"] = plan_file
                entry._planner_session = planner_session

                # Enable plan tools
                _set_enable_plan(True)

                # Run the planner
                prompt_text = (
                    "read the following requirement carefully and generate a comprehensive plan. "
                    "save the complete plan to a file using the WritePlan tool.\n\n"
                    f"Requirement:\n{text.strip()}"
                )

                max_plan_attempts = 3

                for attempt in range(max_plan_attempts):
                    # Check cancellation before each attempt
                    if entry._cancel_event is not None and entry._cancel_event.is_set():
                        step_finish_reason = "cancelled"
                        break

                    try:
                        async for wire_msg in planner_session.prompt(prompt_text, merge_wire_messages=True):
                            # Check cancellation during streaming
                            if entry._cancel_event is not None and entry._cancel_event.is_set():
                                step_finish_reason = "cancelled"
                                break

                            # Stream wire messages as standard message.part.updated events
                            from kimi_cli.wire.types import (
                                ContentPart,
                                StepBegin,
                                StepInterrupted,
                                TextPart,
                                ThinkPart,
                                ToolCall,
                                ToolCallPart,
                                ToolResult,
                            )

                            # ── StepBegin: new step boundary ─────────────────
                            if isinstance(wire_msg, StepBegin):
                                _emit_part_to_message(
                                    asst_msg,
                                    session_id,
                                    MessagePart(
                                        id=_gen_part_id(),
                                        type="step-start",
                                        sessionID=session_id,
                                        messageID=asst_msg_id,
                                        snapshot=step_snapshot,
                                    ),
                                )
                                continue

                            # ── StepInterrupted ──────────────────────────────
                            if isinstance(wire_msg, StepInterrupted):
                                continue

                            # ── ToolCall: emit tool event ────────────────────
                            if isinstance(wire_msg, ToolCall):
                                tool_name = (
                                    wire_msg.function.name if wire_msg.function else "unknown"
                                )
                                tool_args_raw = (
                                    wire_msg.function.arguments if wire_msg.function else ""
                                )
                                tool_part_id = _gen_part_id()
                                _emit_part_to_message(
                                    asst_msg,
                                    session_id,
                                    MessagePart(
                                        id=tool_part_id,
                                        type="tool",
                                        tool=tool_name,
                                        callID=wire_msg.id,
                                        sessionID=session_id,
                                        messageID=asst_msg_id,
                                        state={
                                            "status": "running",
                                            "input": tool_args_raw,
                                        },
                                    ),
                                )
                                continue

                            # ── ToolCallPart: streaming argument chunks ──────
                            if isinstance(wire_msg, ToolCallPart):
                                continue

                            # ── ToolResult: emit tool_result event ───────────
                            if isinstance(wire_msg, ToolResult):
                                rv = wire_msg.return_value
                                output = rv.output if isinstance(rv.output, str) else (rv.message or "")
                                is_error = rv.is_error
                                status = "error" if is_error else "completed"
                                tool_part_id = _gen_part_id()
                                _emit_part_to_message(
                                    asst_msg,
                                    session_id,
                                    MessagePart(
                                        id=tool_part_id,
                                        type="tool",
                                        tool="",
                                        callID=wire_msg.tool_call_id,
                                        sessionID=session_id,
                                        messageID=asst_msg_id,
                                        state={
                                            "status": status,
                                            "output": output[:4000] if output else "",
                                        },
                                    ),
                                )
                                continue

                            # ── ContentPart (e.g. ImageURLPart) — emit as text ──
                            if isinstance(wire_msg, ContentPart):
                                try:
                                    raw = wire_msg.model_dump()
                                    part_text = orjson.dumps(raw).decode("utf-8")
                                    text_buf.append(f"[{raw.get('type', 'unknown')}] {part_text}")
                                    full_so_far = "".join(text_buf)
                                    _emit_part_to_message(
                                        asst_msg,
                                        session_id,
                                        MessagePart(
                                            id=text_part_id,
                                            type="text",
                                            text=full_so_far,
                                            sessionID=session_id,
                                            messageID=asst_msg_id,
                                        ),
                                        delta=part_text,
                                    )
                                except Exception:
                                    pass
                                continue

                            # ── ThinkPart (reasoning) — cumulative ───────────
                            if isinstance(wire_msg, ThinkPart):
                                chunk = wire_msg.think
                                if chunk:
                                    text_buf.append(chunk)
                                    full_so_far = "".join(text_buf)
                                    _emit_part_to_message(
                                        asst_msg,
                                        session_id,
                                        MessagePart(
                                            id=text_part_id,
                                            type="text",
                                            text=full_so_far,
                                            sessionID=session_id,
                                            messageID=asst_msg_id,
                                        ),
                                        delta=chunk,
                                    )
                                continue

                            # ── TextPart — cumulative ────────────────────────
                            if isinstance(wire_msg, TextPart):
                                chunk = wire_msg.text
                                if chunk:
                                    text_buf.append(chunk)
                                    full_so_far = "".join(text_buf)
                                    _emit_part_to_message(
                                        asst_msg,
                                        session_id,
                                        MessagePart(
                                            id=text_part_id,
                                            type="text",
                                            text=full_so_far,
                                            sessionID=session_id,
                                            messageID=asst_msg_id,
                                        ),
                                        delta=chunk,
                                    )

                        if step_finish_reason == "cancelled":
                            break

                        if plan_file.exists() and plan_file.stat().st_size > 0:
                            plan_generated = True
                            # Read and display the plan content
                            plan_content = plan_file.read_text(encoding="utf-8", errors="replace")
                            plan_header = "\n\n--- Generated Plan ---\n"
                            text_buf.append(plan_header)
                            text_buf.append(plan_content)
                            full_so_far = "".join(text_buf)
                            _emit_part_to_message(
                                asst_msg,
                                session_id,
                                MessagePart(
                                    id=text_part_id,
                                    type="text",
                                    text=full_so_far,
                                    sessionID=session_id,
                                    messageID=asst_msg_id,
                                ),
                                delta=plan_header + plan_content,
                            )

                            # ── Confirmation flow ────────────────────────────────
                            entry._plan_confirm_event = asyncio.Event()
                            entry._plan_feedback = None

                            # Emit plan.confirm event for frontend dialog
                            _emit_part_to_message(
                                asst_msg,
                                session_id,
                                MessagePart(
                                    id=_gen_part_id(),
                                    type="plan.confirm",
                                    text=plan_content,
                                    sessionID=session_id,
                                    messageID=asst_msg_id,
                                ),
                            )

                            # Wait for confirmation with timeout
                            try:
                                await asyncio.wait_for(entry._plan_confirm_event.wait(), timeout=300.0)
                            except asyncio.TimeoutError:
                                entry._plan_feedback = None
                                pass

                            confirm_event = entry._plan_confirm_event
                            entry._plan_confirm_event = None

                            if confirm_event is not None and confirm_event.is_set():
                                feedback = entry._plan_feedback
                                entry._plan_feedback = None
                                if feedback is not None and feedback != "accept":
                                    # Revise: loop back with revision feedback
                                    prompt_text = (
                                        "The user reviewed the plan and requests the following changes:\n"
                                        f"{feedback}\n\n"
                                        "Please revise the plan accordingly and save the updated plan "
                                        "using the WritePlan tool. Overwrite the existing plan file."
                                    )
                                    plan_generated = False
                                    continue

                            # Accepted or timeout — proceed to finish
                            break

                        if attempt < max_plan_attempts - 1:
                            prompt_text = (
                                "The plan file was not generated. "
                                "Please generate the plan and save it using the WritePlan tool.\n\n"
                                f"Requirement:\n{text.strip()}"
                            )
                    except Exception as exc:
                        logger.error("Plan generation attempt %d failed: %s", attempt + 1, exc)
                        if attempt == max_plan_attempts - 1:
                            step_finish_reason = str(exc)
                            break
                        await asyncio.sleep(1)

                if not plan_generated and step_finish_reason == "stop":
                    step_finish_reason = "Plan file was not generated after multiple attempts"

            except (asyncio.CancelledError, RunCancelled):
                step_finish_reason = "cancelled"
            except Exception as exc:
                step_finish_reason = str(exc)
                logger.error("[SessionManager] Plan error: %s", exc, exc_info=True)
            finally:
                # Emit step-finish
                _emit_part_to_message(
                    asst_msg,
                    session_id,
                    MessagePart(
                        id=_gen_part_id(),
                        type="step-finish",
                        sessionID=session_id,
                        messageID=asst_msg_id,
                        reason=step_finish_reason,
                        snapshot=step_snapshot,
                    ),
                )

                asst_msg.info.time["completed"] = _now_ms()
                entry.info.updatedAt = _now_ms()
                entry._cancel_event = None

                bus.emit_type(
                    "message.updated",
                    info=asst_msg.info.to_dict(),
                )

                # Clean up planner session
                _set_enable_plan(False)
                if planner_session is not None:
                    try:
                        await close_session_async(planner_session)
                    except Exception:
                        pass
                entry._planner_session = None
                entry._planner_task = None

                # Idle last
                self._set_status(entry, "idle")

        task = asyncio.create_task(_run_plan())
        entry._planner_task = task

    async def confirm_plan_async(self, session_id: str, action: str, feedback: Optional[str] = None) -> bool:
        """POST /session/{sessionID}/plan/confirm

        Confirm or revise a plan after generation.
        "action" is "accept" or "revise".
        "feedback" is revision instructions when action="revise".
        """
        logger.info(
            "[SessionManager] confirm_plan_async("
            "session_id=%s, action=%s, feedback=%s)",
            session_id, action, feedback,
        )
        entry = self._get_entry(session_id)
        if entry._plan_confirm_event is None:
            return False  # No confirmation pending
        entry._plan_feedback = action if action == "accept" else feedback
        entry._plan_confirm_event.set()
        return True

    # ── Special command handlers ─────────────────────────────────

    async def _cmd_clear(self, session_id: str, args: list[str]) -> tuple[Optional[str], Optional[Any]]:
        await self.clear_session(session_id)
        return None, None

    async def _cmd_compact(self, session_id: str, args: list[str]) -> tuple[Optional[str], Optional[Any]]:
        await self.compact_session(session_id)
        return None, None

    async def _cmd_context(self, session_id: str, args: list[str]) -> tuple[Optional[str], Optional[Any]]:
        await self.clear_session(session_id)
        # result = self.get_session_context(session_id)
        return None, None

    async def _cmd_export(self, session_id: str, args: list[str]) -> tuple[Optional[str], Optional[Any]]:
        output_path = ":".join(args) if args else None
        output, count = await self.export_session(session_id, output_path=output_path)
        return None, {"output": output, "count": count}

    async def _cmd_unknown(self, session_id: str, args: list[str]) -> tuple[Optional[str], Optional[Any]]:
        return None, {"detail": "Unknown command"}

    # ── Helpers ───────────────────────────────────────────────────

    def _get_entry(self, session_id: str) -> ManagedSession:
        with self._lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError(f"Session not found: {session_id}")
        return entry

    async def _get_entry_or_restore(self, session_id: str) -> ManagedSession:
        try:
            return self._get_entry(session_id)
        except KeyError:
            await self._refresh_disk_sessions()
            return self._get_entry(session_id)

    def _set_status(self, entry: ManagedSession, status_type: str) -> None:
        token_count = 0
        context_usage = 0.0
        if entry.sdk_session is not None:
            try:
                sdk_status = entry.sdk_session.status
                token_count = getattr(sdk_status, "context_tokens", 0)
                context_usage = getattr(sdk_status, "context_usage", 0.0)
            except Exception:
                pass
        entry.status = SessionStatus(
            type=status_type,
            time=time.time(),
            token_count=token_count,
            context_usage=context_usage,
        )
        bus.emit_type(
            "session.status",
            sessionID=entry.info.id,
            status=entry.status.to_dict(),
        )
        if status_type == "idle":
            bus.emit_type("session.idle", sessionID=entry.info.id)


# Global singleton
session_manager = SessionManager()
