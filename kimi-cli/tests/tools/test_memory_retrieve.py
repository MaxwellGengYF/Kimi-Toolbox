"""Tests for the Retrieve tool (history-only retrieval via HistoryIndex).

Retrieve searches past conversation turns (compacted/archived included)
by natural-language query with a recency boost, or fetches a specific
turn by id (a plain turn id or a ``prune_<n>`` reference).
"""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.tools.memory import Params, Retrieve
from kimi_cli.wire.types import TextPart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_tool() -> Retrieve:
    """Retrieve with NO history index attached."""
    return Retrieve()


@pytest.fixture
def history_index() -> HistoryIndex:
    """Pre-populated HistoryIndex with API-related turns."""
    idx = HistoryIndex()
    idx.index_messages([
        Message(role="user", content=[TextPart(text="What did we decide about the API design?")]),
        Message(role="assistant", content=[TextPart(text="We decided to use REST over GraphQL.")]),
    ])
    return idx


@pytest.fixture
def tool_with_history(memory_tool: Retrieve, history_index: HistoryIndex) -> Retrieve:
    """Retrieve with the history index attached."""
    memory_tool.attach_history_index(history_index)
    return memory_tool


# ---------------------------------------------------------------------------
# Query search
# ---------------------------------------------------------------------------


class TestRetrieveSearch:
    @pytest.mark.asyncio
    async def test_query_search_finds_turns(self, tool_with_history: Retrieve):
        """Query search returns matching history turns with relevance scores."""
        result = await tool_with_history(Params(query="REST", k=5))
        assert "[Conversation history]" in result.output
        assert "We decided to use REST over GraphQL." in result.output
        assert "relevance:" in result.output
        assert result.output.count("> **") == 2

    @pytest.mark.asyncio
    async def test_no_index_guidance(self, memory_tool: Retrieve):
        """Without an attached index, a clear guidance message is returned."""
        result = await memory_tool(Params(query="REST", k=5))
        assert "No history index attached" in result.output

    @pytest.mark.asyncio
    async def test_no_results_message(self, tool_with_history: Retrieve):
        """No matching turns yield a clear no-results message."""
        result = await tool_with_history(Params(query="zzz_nonexistent_zzz", k=5))
        assert "No matching results found in conversation history" in result.output

    @pytest.mark.asyncio
    async def test_empty_history_index(self, memory_tool: Retrieve):
        """An attached but empty history index behaves like no results."""
        memory_tool.attach_history_index(HistoryIndex())
        result = await memory_tool(Params(query="REST", k=5))
        assert "No matching results found" in result.output

    @pytest.mark.asyncio
    async def test_header_retrieved_count(self, tool_with_history: Retrieve):
        """Header reports the total number of retrieved turns."""
        result = await tool_with_history(Params(query="REST", k=10))
        assert "Retrieved 2 result(s):" in result.output

    @pytest.mark.asyncio
    async def test_message_reflects_count(self, tool_with_history: Retrieve):
        """ToolReturnValue.message shows the result count."""
        result = await tool_with_history(Params(query="REST", k=10))
        assert "Found" in result.message
        assert "result(s)" in result.message

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, tool_with_history: Retrieve):
        """Queries with special characters do not crash."""
        result = await tool_with_history(Params(query="REST!@#$%", k=5))
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_unicode_in_history(self):
        """Unicode history content is searchable."""
        idx = HistoryIndex()
        idx.index_messages([
            Message(role="user", content=[TextPart(text="café concept and naïve approach")]),
            Message(role="user", content=[TextPart(text="the rest of the plan uses unicode correctly")]),
        ])
        tool = Retrieve()
        tool.attach_history_index(idx)
        result = await tool(Params(query="café", k=5))
        assert "café" in result.output


# ---------------------------------------------------------------------------
# k parameter
# ---------------------------------------------------------------------------


class TestRetrieveK:
    @pytest.mark.asyncio
    async def test_k_limits_results(self, tool_with_history: Retrieve):
        """k caps the number of returned turns."""
        result = await tool_with_history(Params(query="REST", k=1))
        assert result.output.count("> **") <= 1

    @pytest.mark.asyncio
    async def test_k_validation(self):
        """k is constrained to 1..10 by the Params schema."""
        with pytest.raises(Exception):
            Params.model_validate({"query": "REST", "k": 0})
        with pytest.raises(Exception):
            Params.model_validate({"query": "REST", "k": 11})


# ---------------------------------------------------------------------------
# id-based retrieval
# ---------------------------------------------------------------------------


class TestRetrieveById:
    @pytest.mark.asyncio
    async def test_id_fetch_plain(self, tool_with_history: Retrieve):
        """A plain turn id fetches the turn."""
        result = await tool_with_history(Params(id="0"))
        assert "turn id='0'" in result.output
        assert "[current]" in result.output
        assert "API design" in result.output

    @pytest.mark.asyncio
    async def test_id_fetch_prune_prefix(self, tool_with_history: Retrieve):
        """A prune_<n> reference fetches the same turn."""
        result = await tool_with_history(Params(id="prune_0"))
        assert "turn id='prune_0'" in result.output
        assert "API design" in result.output

    @pytest.mark.asyncio
    async def test_id_not_found(self, tool_with_history: Retrieve):
        """An unknown id yields a not-found message."""
        result = await tool_with_history(Params(id="prune_999"))
        assert "No turn found with id=" in result.output

    @pytest.mark.asyncio
    async def test_id_without_index(self, memory_tool: Retrieve):
        """An id fetch without an attached index yields guidance."""
        result = await memory_tool(Params(id="0"))
        assert "No history index attached" in result.output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRetrieveEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_query_guidance(self, tool_with_history: Retrieve):
        """Empty query returns guidance, not an error."""
        result = await tool_with_history(Params(query="", k=3))
        assert "No query provided" in result.output
        assert "`id`" in result.output

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, tool_with_history: Retrieve):
        """Whitespace-only query is treated as empty."""
        result = await tool_with_history(Params(query="   \n\t  ", k=3))
        assert "No query provided" in result.output

    @pytest.mark.asyncio
    async def test_compacted_marker(self, tool_with_history: Retrieve, history_index: HistoryIndex):
        """Turns marked compacted show a [compacted] marker."""
        history_index.mark_compacted()
        result = await tool_with_history(Params(query="REST", k=5))
        assert "[compacted]" in result.output
        assert "[current]" not in result.output

    @pytest.mark.asyncio
    async def test_output_is_valid_string(self, tool_with_history: Retrieve):
        """Output is always a non-empty string."""
        result = await tool_with_history(Params(query="REST", k=5))
        assert isinstance(result.output, str)
        assert len(result.output) > 0
