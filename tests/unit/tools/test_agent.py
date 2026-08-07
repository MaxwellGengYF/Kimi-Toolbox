"""Tests for Defects 11.1-11.4: Agent tool improvements."""
from __future__ import annotations

from unittest.mock import MagicMock

from kimix.tools.agent import AgentRespond, AgentRespondParams


class TestAgentRespond:
    async def test_agentrespond_missing_session(self, mock_session: MagicMock) -> None:
        ar = AgentRespond(session=mock_session)
        result = await ar(AgentRespondParams(session_id="nonexistent", response="answer"))
        assert result.is_error
        assert "not found" in result.message.lower()
