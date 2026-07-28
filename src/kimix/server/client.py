# -*- coding: utf-8 -*-
"""Kimix opencode-style HTTP / SSE async client.

API surface mirrors the opencode client protocol for full compatibility.

Supports:
- REST API: session CRUD, prompt_async (204), abort, permissions
- SSE: /event global stream with auto-reconnect
- Event parsing: message.part.updated → text/reasoning/tool/step-start/step-finish
"""

from __future__ import annotations

import asyncio
import orjson
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx

from kimix.ui.printing import print  # noqa: F811 - use ui.printing.print for flush support

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────


class MessagePartType(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    TOOL_CALLING_PART = "tool_calling_part"
    TOOL_RESULT = "tool_result"
    STEP_START = "step-start"
    STEP_FINISH = "step-finish"
    UNKNOWN = "unknown"


@dataclass
class MessagePart:
    type: MessagePartType
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_status: Optional[str] = None
    tool_state: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    call_id: Optional[str] = None
    reason: Optional[str] = None
    cost: Optional[float] = None
    tokens: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessagePart":
        part_type = data.get("type", "text")
        # Map opencode type strings to MessagePartType values
        _OPENCODE_TYPE_MAP: Dict[str, str] = {
            "tool": "tool_calling",
            "reasoning": "thinking",
        }
        part_type = _OPENCODE_TYPE_MAP.get(part_type, part_type)
        try:
            msg_type = MessagePartType(part_type)
        except ValueError:
            return cls(type=MessagePartType.UNKNOWN, raw_data=data)

        part = cls(type=msg_type)
        if msg_type == MessagePartType.TEXT:
            part.text = data.get("text")
        elif msg_type == MessagePartType.TOOL_CALLING:
            part.tool_name = data.get("tool")
            part.call_id = data.get("callID")
            state = data.get("state", {})
            part.tool_status = state.get("status")
            part.tool_state = state
        elif msg_type == MessagePartType.THINKING:
            part.text = data.get("text")
        elif msg_type in (MessagePartType.TOOL_CALLING_PART, MessagePartType.TOOL_RESULT):
            part.text = data.get("text")
        elif msg_type == MessagePartType.STEP_FINISH:
            # opencode: reason, cost, tokens are at part level
            part.reason = data.get("reason")
            part.cost = data.get("cost")
            part.tokens = data.get("tokens")
        elif msg_type == MessagePartType.UNKNOWN:
            part.raw_data = data
        return part


@dataclass
class Message:
    id: str
    role: str
    parts: List[MessagePart] = field(default_factory=list)
    created_at: Optional[float] = None

    @property
    def text_content(self) -> str:
        return "".join(
            p.text for p in self.parts
            if p.type == MessagePartType.TEXT and p.text
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        # Handle dummy session manager format from _process_prompt:
        # {"type": "Text|Thinking|ToolCalling", "text": "...", "time": ts}
        if "info" not in data:
            msg_type = data.get("type", "Text")
            text = data.get("text", "")
            created_at = data.get("time")
            # Map MessageType.name values to MessagePartType
            _TYPE_MAP: Dict[str, MessagePartType] = {
                "Text": MessagePartType.TEXT,
                "Thinking": MessagePartType.THINKING,
                "ToolCalling": MessagePartType.TOOL_CALLING,
                "ToolCallingPart": MessagePartType.TOOL_CALLING_PART,
                "ToolResult": MessagePartType.TOOL_RESULT,
            }
            part_type = _TYPE_MAP.get(msg_type, MessagePartType.UNKNOWN)
            part = MessagePart(type=part_type, text=text)
            if data.get("tool_name"):
                part.tool_name = data["tool_name"]
            if data.get("tool_result"):
                part.tool_result = data["tool_result"]
            return cls(
                id="",
                role="assistant",
                parts=[part],
                created_at=created_at,
            )

        # Opencode format: {"info": {"id":..., "role":..., "time":...}, "parts": [...]}
        info = data.get("info", {})
        time_info = info.get("time", {})
        return cls(
            id=info.get("id", ""),
            role=info.get("role", "assistant"),
            parts=[MessagePart.from_dict(p) for p in data.get("parts", [])],
            created_at=time_info.get("created"),
        )


@dataclass
class Session:
    id: str
    title: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    parent_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=data.get("id", ""),
            title=data.get("title"),
            created_at=data.get("createdAt"),
            updated_at=data.get("updatedAt"),
            parent_id=data.get("parentID"),
        )


@dataclass
class SSEEvent:
    """Parsed Server-Sent Event.

    In opencode protocol, there is no SSE `event:` field.
    All type info is in the JSON data.type.
    """

    event: str = ""
    data: str = ""
    id: Optional[str] = None

    def json(self) -> Any:
        if not self.data:
            return None
        try:
            return orjson.loads(self.data)
        except orjson.JSONDecodeError:
            return None

    @property
    def is_reconnect(self) -> bool:
        return self.event == "__reconnected__"


# ── Event Type / Parsed Event ─────────────────────────────────────


class EventType(str, Enum):
    TEXT = "text"
    TEXT_DELTA = "text_delta"
    REASONING = "reasoning"
    TOOL = "tool"
    PERMISSION = "permission"
    STEP_START = "step-start"
    STEP_FINISH = "step-finish"
    SESSION_IDLE = "session_idle"
    RECONNECTED = "reconnected"
    SKIP = "skip"
    UNKNOWN = "unknown"


@dataclass
class ParsedEvent:
    type: EventType = EventType.UNKNOWN
    text: str = ""
    delta: str = ""
    tool_name: str = ""
    tool_status: str = ""
    tool_title: str = ""
    tool_input: Any = ""
    tool_output: str = ""
    tool_error: str = ""
    tool_call_id: str = ""
    created_at: float = 0.0
    permission_id: str = ""
    finished: bool = False
    cost: float = 0.0
    tokens: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if this event signals the end of a prompt response.

        Terminal conditions (per opencode protocol):
        1. session.status with type == "idle"
        2. step-finish with reason != "tool-calls" (e.g. reason == "stop")
        """
        if self.type == EventType.SESSION_IDLE:
            return True
        if self.type == EventType.STEP_FINISH:
            return self.text not in ("tool-calls", "tool_calls")
        return False


# ── SSE Event Parser ─────────────────────────────────────────────


def parse_event(event: SSEEvent, session_id: str = "") -> ParsedEvent:
    """Parse a raw SSE event into a structured ParsedEvent.

    Compatible with opencode protocol format.
    """
    if event.event == "__reconnected__":
        return ParsedEvent(
            type=EventType.RECONNECTED,
            text=f"SSE reconnected (attempt {event.data})",
        )

    data = event.json()
    if not data:
        return ParsedEvent(type=EventType.SKIP)

    event_type: str = data.get("type", "")

    if event_type in ("server.connected", "server.heartbeat"):
        return ParsedEvent(type=EventType.SKIP)

    # Session filtering
    if session_id and not _matches_session(data, session_id):
        return ParsedEvent(type=EventType.SKIP)

    if event_type == "message.part.updated":
        return _parse_part_updated(data)

    if event_type in (
        "message.updated",
        "message.created",
        "session.updated",
        "session.created",
        "session.diff",
    ):
        return ParsedEvent(type=EventType.SKIP)

    if event_type in ("session.idle", "session.status"):
        return _parse_session_status(data, event_type)

    return ParsedEvent(type=EventType.UNKNOWN, raw=data)


def _parse_part_updated(data: Dict[str, Any]) -> ParsedEvent:
    props = data.get("properties", {})
    part = props.get("part", {})
    delta = props.get("delta", "")
    part_type = part.get("type", "")

    if part_type == "text":
        return ParsedEvent(
            type=EventType.TEXT,
            delta=delta,
            text=part.get("text", ""),
            raw=data,
        )
    if part_type == "tool":
        state = part.get("state", {})
        tool_name = part.get("tool", "")
        call_id = part.get("callID", "")
        status = state.get("status", "")
        tool_input = state.get("input", {})
        return ParsedEvent(
            type=EventType.TOOL,
            tool_name=tool_name,
            tool_status=status,
            tool_title=tool_name,
            tool_input=tool_input,
            tool_output=state.get("output", ""),
            tool_error=state.get("error", ""),
            tool_call_id=call_id,
            raw=data,
        )
    if part_type == "reasoning":
        return ParsedEvent(
            type=EventType.REASONING,
            text=part.get("text", ""),
            delta=delta,
            raw=data,
        )
    if part_type == "step-start":
        return ParsedEvent(type=EventType.STEP_START, raw=data)
    if part_type == "step-finish":
        # opencode: reason, cost, tokens at part level (not in state)
        reason = part.get("reason", "")
        finished = reason not in ("tool-calls", "tool_calls")
        return ParsedEvent(
            type=EventType.STEP_FINISH,
            text=reason,
            finished=finished,
            cost=part.get("cost", 0.0),
            tokens=part.get("tokens", {}),
            raw=data,
        )
    return ParsedEvent(type=EventType.SKIP)


def _parse_session_status(data: Dict[str, Any], event_type: str) -> ParsedEvent:
    if event_type == "session.status":
        props = data.get("properties", {})
        status = props.get("status", {})
        status_type = status.get("type", "") if isinstance(status, dict) else ""
        if status_type != "idle":
            return ParsedEvent(type=EventType.SKIP)
    return ParsedEvent(type=EventType.SESSION_IDLE, finished=True, raw=data)


def _matches_session(data: Dict[str, Any], session_id: str) -> bool:
    """Check if an event belongs to the given session.

    Checks multiple possible locations per opencode protocol:
    - properties.sessionID
    - properties.part.sessionID
    - properties.info.sessionID
    """
    props = data.get("properties", {})
    sid: Optional[str] = props.get("sessionID")
    if not sid:
        part = props.get("part", {})
        if isinstance(part, dict):
            sid = part.get("sessionID")
    if not sid:
        info = props.get("info", {})
        if isinstance(info, dict):
            sid = info.get("sessionID")
    # If no sessionID found in event, don't filter it out
    return not sid or sid == session_id


# ── Sync Helpers ──────────────────────────────────────────────────


def check_health_sync(
    port: int,
    host: str = "127.0.0.1",
    timeout: float = 3.0,
) -> bool:
    """Synchronous health check for process management contexts."""
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), trust_env=False) as c:
            resp = c.get(f"http://{host}:{port}/global/health")
            return bool(resp.json().get("healthy", False))
    except Exception:
        return False


def abort_session_sync(
    session_id: str,
    port: int,
    host: str = "127.0.0.1",
    timeout: float = 10.0,
) -> bool:
    """Synchronous session abort."""
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), trust_env=False) as c:
            resp = c.post(f"http://{host}:{port}/session/{session_id}/abort")
            return resp.status_code == 200
    except Exception:
        return False


# ── Async Client ──────────────────────────────────────────────────


class KimixAsyncClient:
    """Async HTTP + SSE client for kimix serve (opencode-style API).

    Example::

        async with KimixAsyncClient(port=4096) as client:
            sess = await client.create_session("My Task")
            ok = await client.send_prompt_async(sess.id, "write tests")
            async for event in client.stream_events_robust(sess.id):
                parsed = parse_event(event, sess.id)
                if parsed.type == EventType.TEXT:
                    print(parsed.delta, end="", flush=True)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4096,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), trust_env=False
        )

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except RuntimeError as exc:
            # Transports bound to a now-closed ProactorEventLoop (Windows
            # Python 3.14) raise RuntimeError('Event loop is closed').
            # The OS will reclaim the socket, so we swallow it.
            if "Event loop is closed" not in str(exc):
                raise

    async def __aenter__(self) -> "KimixAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        await self.close()
        return False

    # ── Health ────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/global/health")
            data = resp.json()
            return bool(data.get("healthy", False))
        except Exception as exc:
            logger.warning(
                "[KimixClient] health_check: %s: %s", type(exc).__name__, exc
            )
            return False

    # ── Session CRUD ─────────────────────────────────────────

    async def create_session(self, title: Optional[str] = None) -> Session:
        body = {"title": title} if title else {}
        resp = await self._client.post(f"{self._base_url}/session", json=body)
        resp.raise_for_status()
        return Session.from_dict(resp.json())

    async def get_session(self, session_id: str) -> Session:
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}"
        )
        resp.raise_for_status()
        return Session.from_dict(resp.json())

    async def delete_session(self, session_id: str) -> bool:
        resp = await self._client.delete(
            f"{self._base_url}/session/{session_id}"
        )
        return resp.status_code == 200

    async def list_sessions(self) -> List[Session]:
        resp = await self._client.get(f"{self._base_url}/session")
        resp.raise_for_status()
        return [Session.from_dict(s) for s in resp.json()]

    async def get_messages(
        self, session_id: str, limit: int = 10
    ) -> List[Message]:
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}/message",
            params={"limit": limit},
        )
        resp.raise_for_status()
        return [Message.from_dict(m) for m in resp.json()]

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """GET /session/{sessionID}/status

        Returns running/idle status with context usage and token count when idle.
        """
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}/status"
        )
        resp.raise_for_status()
        return resp.json()

    # ── Messaging ────────────────────────────────────────────

    async def send_prompt_async(
        self,
        session_id: str,
        text: str,
        agent: Optional[str] = None,
    ) -> bool:
        """Fire-and-forget prompt (HTTP 204)."""
        body: Dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if agent:
            body["agent"] = agent
        resp = await self._client.post(
            f"{self._base_url}/session/{session_id}/prompt_async", json=body
        )
        return resp.status_code == 204

    async def abort_session(self, session_id: str) -> bool:
        resp = await self._client.post(
            f"{self._base_url}/session/{session_id}/abort"
        )
        return resp.status_code == 200

    async def clear_session(self, session_id: str) -> bool:
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}/clear"
        )
        return resp.status_code == 200

    async def compact_session(
        self, session_id: str, keep: Optional[int] = None
    ) -> bool:
        params = {}
        if keep is not None:
            params["keep"] = keep
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}/compact", params=params
        )
        return resp.status_code == 200

    async def export_session(
        self, session_id: str, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {}
        if output_path:
            params["output_path"] = output_path
        resp = await self._client.get(
            f"{self._base_url}/session/{session_id}/export", params=params
        )
        resp.raise_for_status()
        return resp.json()

    # ── SSE Streaming ────────────────────────────────────────

    async def stream_events(
        self,
        session_id: str,
        timeout: float = 14400.0,
    ) -> AsyncIterator[SSEEvent]:
        """Stream SSE events from /event endpoint."""
        url = f"{self._base_url}/event"
        request = self._client.build_request(
            "GET",
            url,
            timeout=httpx.Timeout(timeout, connect=10.0, read=timeout),
        )
        response = await self._client.send(request, stream=True)
        try:
            response.raise_for_status()
            async for event in _parse_sse_stream(response):
                yield event
        finally:
            await response.aclose()

    async def stream_events_robust(
        self,
        session_id: str,
        timeout: float = 14400.0,
        max_reconnects: int = 5,
        reconnect_delay: float = 2.0,
        on_reconnect: Optional[Callable[[int], None]] = None,
    ) -> AsyncIterator[SSEEvent]:
        """SSE stream with auto-reconnect."""
        reconnects = 0
        while reconnects <= max_reconnects:
            try:
                async for event in self.stream_events(session_id, timeout):
                    reconnects = 0
                    yield event
                return
            except (
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as exc:
                reconnects += 1
                if reconnects > max_reconnects:
                    logger.error("[SSE] Max reconnects reached: %s", exc)
                    raise
                logger.warning(
                    "[SSE] Reconnecting (%d/%d): %s",
                    reconnects,
                    max_reconnects,
                    exc,
                )
                if on_reconnect:
                    on_reconnect(reconnects)
                await asyncio.sleep(reconnect_delay * reconnects)
                yield SSEEvent(
                    event="__reconnected__", data=str(reconnects)
                )


# ── SSE Stream Parser (internal) ──────────────────────────────────


async def _parse_sse_stream(
    response: httpx.Response,
) -> AsyncIterator[SSEEvent]:
    """Parse HTTP response body into SSEEvent stream.

    Handles the opencode SSE format where:
    - No `event:` field is used
    - Events are `data: {json}` followed by blank line
    - Comments (`: ...`) are ignored
    """
    current = SSEEvent()
    data_lines: List[str] = []

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r\n")

        if not line:
            if data_lines or current.event:
                current.data = "\n".join(data_lines)
                yield current
                current = SSEEvent()
                data_lines = []
            continue

        if line.startswith(":"):
            continue  # SSE comment / heartbeat

        if ":" in line:
            field_name, _, value = line.partition(":")
            value = value.lstrip(" ")
        else:
            field_name = line
            value = ""

        if field_name == "event":
            current.event = value
        elif field_name == "data":
            data_lines.append(value)
        elif field_name == "id":
            current.id = value

    if data_lines or current.event:
        current.data = "\n".join(data_lines)
        yield current
