from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel
from kimi_agent_sdk import Session

from kimix.utils import close_session_async


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system", "tool", "error"]
    content: str | list[Any]
    timestamp: float
    metadata: dict[str, Any] | None = None


class AgentSessionEntry:
    def __init__(
        self,
        session: Session,
        session_id: str,
        created_at: float,
        last_accessed: float,
        conversation_history: list[ConversationTurn],
        total_turns: int,
        is_active: bool = True,
        pending_question: str | None = None,
        state: Literal["running", "awaiting_response", "completed"] = "running",
    ) -> None:
        self.session = session
        self.session_id = session_id
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.conversation_history = conversation_history
        self.total_turns = total_turns
        self.is_active = is_active
        self.pending_question = pending_question
        self.state = state


class AgentSessionStore:
    MAX_SESSIONS: int = 10

    def __init__(self) -> None:
        self.entries: dict[str, AgentSessionEntry] = {}

    def get(self, session_id: str) -> AgentSessionEntry | None:
        return self.entries.get(session_id)

    def put(self, entry: AgentSessionEntry) -> None:
        self.entries[entry.session_id] = entry

    def close(self, session_id: str) -> bool:
        entry = self.entries.pop(session_id, None)
        return entry is not None

    def list_active(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in self.entries.values():
            if entry.is_active:
                result.append({
                    "session_id": entry.session_id,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "total_turns": entry.total_turns,
                    "state": entry.state,
                    "is_active": entry.is_active,
                })
        return result

    async def evict_lru_if_needed(self) -> None:
        while len(self.entries) >= self.MAX_SESSIONS:
            lru_id = min(
                self.entries.keys(),
                key=lambda sid: self.entries[sid].last_accessed,
            )
            entry = self.entries.pop(lru_id)
            entry.is_active = False
            try:
                await close_session_async(entry.session)
            except Exception:
                pass
