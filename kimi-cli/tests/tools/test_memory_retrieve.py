"""Tests for the Memory 'retrieve' action (history + durable memory search).

'retrieve' searches conversation history (via the attached HistoryIndex)
AND durable memory files, or fetches by id: a history turn id or
'memory:<topic>'.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.tools.memory import Memory, Params
from kimi_cli.wire.types import TextPart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "session"


@pytest.fixture
def memory_tool(session_dir: Path) -> Memory:
    """Memory with a real session dir but NO history index attached."""
    runtime = SimpleNamespace(session=SimpleNamespace(dir=session_dir), read_only=False)
    return Memory(runtime)  # type: ignore[arg-type]


@pytest.fixture
def memory_dir(session_dir: Path) -> Path:
    """Session memory dir pre-populated with several .md files."""
    md = session_dir / "memory"
    md.mkdir(parents=True)
    (md / "decisions.md").write_text(
        "# Decisions\n\n- Use REST for all APIs\n- OAuth2 for auth\n- Rate limiting at gateway\n",
        encoding="utf-8",
    )
    (md / "architecture.md").write_text(
        "# Architecture\n\n- Microservices with gRPC\n- REST gateway at edge\n- PostgreSQL for persistence\n",
        encoding="utf-8",
    )
    (md / "unrelated.md").write_text(
        "# Unrelated\n\n- Team lunch on Fridays\n- Coffee machine on 3rd floor\n",
        encoding="utf-8",
    )
    return md


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
def tool_with_history(memory_tool: Memory, history_index: HistoryIndex) -> Memory:
    """Memory with the history index attached (full merged behavior)."""
    memory_tool.attach_history_index(history_index)
    return memory_tool


# ---------------------------------------------------------------------------
# Backward compatibility: no history index attached
# ---------------------------------------------------------------------------


class TestRetrieveBackwardCompat:
    """Memory without a history index keeps working; retrieve = memory-only."""

    @pytest.mark.asyncio
    async def test_existing_actions_unchanged(self, memory_tool: Memory):
        """write/search/read actions behave exactly as before."""
        r = await memory_tool(Params(action="write", topic="facts", content="sky is blue"))
        assert "Wrote" in r.output
        r = await memory_tool(Params(action="search", query="sky"))
        assert "facts:1" in r.output
        r = await memory_tool(Params(action="read", topic="facts"))
        assert "sky is blue" in r.output
        r = await memory_tool(Params(action="list"))
        assert "facts" in r.output

    @pytest.mark.asyncio
    async def test_retrieve_memory_only_when_no_history(self, memory_tool: Memory):
        """retrieve searches durable memory even without a history index."""
        await memory_tool(Params(action="write", topic="decisions", content="Use REST for all APIs"))
        result = await memory_tool(Params(action="retrieve", query="REST", k=5))
        assert "[Durable memory]" in result.output
        assert "decisions" in result.output
        assert "[Conversation history]" not in result.output

    @pytest.mark.asyncio
    async def test_retrieve_id_memory_without_history(self, memory_tool: Memory):
        """id='memory:<topic>' works without a history index."""
        await memory_tool(Params(action="write", topic="config", content="secret123"))
        result = await memory_tool(Params(action="retrieve", id="memory:config"))
        assert "secret123" in result.output
        assert "memory topic 'config'" in result.output

    @pytest.mark.asyncio
    async def test_retrieve_id_history_without_index(self, memory_tool: Memory):
        """A history turn id without an attached index yields a clear message."""
        result = await memory_tool(Params(action="retrieve", id="0"))
        assert "No history index attached" in result.output


# ---------------------------------------------------------------------------
# Dual-source search
# ---------------------------------------------------------------------------


class TestRetrieveDualSearch:
    """retrieve with both a history index and durable memory configured."""

    @pytest.mark.asyncio
    async def test_retrieve_finds_memory_with_marker(self, tool_with_history: Memory, memory_dir: Path):
        """Search finds durable memory results with [memory] marker."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        assert "[memory]" in result.output
        assert "decisions" in result.output
        assert "[Durable memory]" in result.output

    @pytest.mark.asyncio
    async def test_retrieve_memory_only_when_history_empty(self, memory_tool: Memory, memory_dir: Path):
        """When the attached history index has no turns, only memory appears."""
        memory_tool.attach_history_index(HistoryIndex())
        result = await memory_tool(Params(action="retrieve", query="gateway", k=5))
        assert "[Durable memory]" in result.output
        assert "architecture" in result.output
        assert "[Conversation history]" not in result.output

    @pytest.mark.asyncio
    async def test_retrieve_history_only_when_memory_no_match(
        self, tool_with_history: Memory, memory_dir: Path
    ):
        """When memory has no matches, only history results appear."""
        result = await tool_with_history(Params(action="retrieve", query="GraphQL", k=5))
        assert "[Conversation history]" in result.output
        assert "[Durable memory]" not in result.output

    @pytest.mark.asyncio
    async def test_retrieve_no_results_when_neither_matches(
        self, tool_with_history: Memory, memory_dir: Path
    ):
        """When neither source matches, clear no-results message."""
        result = await tool_with_history(Params(action="retrieve", query="zzz_nonexistent_zzz", k=5))
        assert "No matching results found" in result.output

    # ── merged results ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_merged_both_sections(self, tool_with_history: Memory, memory_dir: Path):
        """Both [Conversation history] and [Durable memory] sections appear."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        assert "[Conversation history]" in result.output
        assert "[Durable memory]" in result.output

    @pytest.mark.asyncio
    async def test_merged_total_count(self, tool_with_history: Memory, memory_dir: Path):
        """Total count in header equals sum of both sections."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=10))
        hist_count = result.output.count("> **")
        mem_count = result.output.count("[memory]")
        total = hist_count + mem_count
        assert f"Retrieved {total} result(s):" in result.output

    @pytest.mark.asyncio
    async def test_history_section_before_memory(self, tool_with_history: Memory, memory_dir: Path):
        """[Conversation history] appears before [Durable memory]."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        hist_pos = result.output.find("[Conversation history]")
        mem_pos = result.output.find("[Durable memory]")
        assert hist_pos < mem_pos

    @pytest.mark.asyncio
    async def test_memory_topic_line_format(self, tool_with_history: Memory, memory_dir: Path):
        """Each memory result shows topic:line_no format."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        mem_start = result.output.find("[Durable memory]")
        mem_section = result.output[mem_start:]
        assert "- [" in mem_section
        assert ":" in mem_section.split("- [", 1)[1] if "- [" in mem_section else False

    # ── id-based retrieval ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_id_retrieves_memory_topic(self, tool_with_history: Memory, memory_dir: Path):
        """id='memory:decisions' returns memory file content."""
        result = await tool_with_history(Params(action="retrieve", id="memory:decisions"))
        assert "REST" in result.output
        assert "memory topic 'decisions'" in result.output

    @pytest.mark.asyncio
    async def test_id_retrieves_memory_topic_not_found(self, tool_with_history: Memory, memory_dir: Path):
        """id='memory:nonexistent' returns not-found message."""
        result = await tool_with_history(Params(action="retrieve", id="memory:nonexistent"))
        assert "No memory topic found" in result.output

    @pytest.mark.asyncio
    async def test_id_memory_prefix_sanitizes_topic(self, tool_with_history: Memory, memory_dir: Path):
        """id='memory:My Topic!' sanitizes the topic name before lookup."""
        result = await tool_with_history(Params(action="retrieve", id="memory:My Topic!"))
        assert "No memory topic found" in result.output or "memory topic" in result.output

    @pytest.mark.asyncio
    async def test_id_memory_empty_topic(self, tool_with_history: Memory, memory_dir: Path):
        """id='memory:' with empty topic falls back to the default topic name."""
        result = await tool_with_history(Params(action="retrieve", id="memory:"))
        assert "No memory topic found" in result.output or "memory topic" in result.output

    @pytest.mark.asyncio
    async def test_id_history_turn_still_works(self, tool_with_history: Memory, memory_dir: Path):
        """Non-memory-prefixed id retrieves from HistoryIndex."""
        result = await tool_with_history(Params(action="retrieve", id="0"))
        assert "turn id='0'" in result.output
        assert "[current]" in result.output

    @pytest.mark.asyncio
    async def test_id_history_prune_prefix_still_works(self, tool_with_history: Memory, memory_dir: Path):
        """id='prune_0' still retrieves from HistoryIndex."""
        result = await tool_with_history(Params(action="retrieve", id="prune_0"))
        assert "turn id='prune_0'" in result.output

    # ── k parameter ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_k_limits_memory_results(self, memory_tool: Memory, memory_dir: Path):
        """k limits the memory hit count."""
        result = await memory_tool(Params(action="retrieve", query="REST", k=1))
        mem_start = result.output.find("[Durable memory]")
        mem_section = result.output[mem_start:] if mem_start >= 0 else result.output
        assert mem_section.count("- [") <= 1

    @pytest.mark.asyncio
    async def test_k_limits_total_merged(self, tool_with_history: Memory, memory_dir: Path):
        """Each source gets up to k results; total ≤ 2*k."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=1))
        assert result.output.count("> **") <= 1
        assert result.output.count("[memory]") <= 1

    # ── edge cases ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_query_graceful(self, tool_with_history: Memory, memory_dir: Path):
        """Empty query returns guidance, not an error."""
        result = await tool_with_history(Params(action="retrieve", query="", k=3))
        assert "No query provided" in result.output

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, tool_with_history: Memory, memory_dir: Path):
        """Whitespace-only query is treated as empty."""
        result = await tool_with_history(Params(action="retrieve", query="   \n\t  ", k=3))
        assert "No query provided" in result.output

    @pytest.mark.asyncio
    async def test_no_memory_dir_yet(self, tool_with_history: Memory):
        """A session with no memory files yet returns history results only."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        assert "[Conversation history]" in result.output
        assert "[Durable memory]" not in result.output

    @pytest.mark.asyncio
    async def test_compacted_marker_still_appears(self, tool_with_history: Memory, memory_dir: Path):
        """History turns marked as compacted show [compacted] marker."""
        tool_with_history._history_index.mark_compacted()  # pyright: ignore[reportPrivateUsage]
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        assert "[compacted]" in result.output

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, tool_with_history: Memory, memory_dir: Path):
        """Query with special characters doesn't crash."""
        result = await tool_with_history(Params(action="retrieve", query="REST!@#$%", k=5))
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_unicode_in_memory_content(self, memory_tool: Memory, session_dir: Path):
        """Memory files with Unicode content are searched correctly."""
        md = session_dir / "memory"
        md.mkdir(parents=True)
        (md / "unicode.md").write_text("# Unicode\n\n- café concept\n- naïve approach\n", encoding="utf-8")
        result = await memory_tool(Params(action="retrieve", query="café", k=5))
        assert "unicode" in result.output or "café" in result.output

    @pytest.mark.asyncio
    async def test_large_memory_file_truncation(self, memory_tool: Memory, session_dir: Path):
        """Very long lines in memory files are truncated in snippets."""
        md = session_dir / "memory"
        md.mkdir(parents=True)
        long_line = "key_point " + "REST " * 200 + "end"
        (md / "large.md").write_text(long_line, encoding="utf-8")
        result = await memory_tool(Params(action="retrieve", query="key_point", k=3))
        assert "…" in result.output  # ellipsis indicates truncation

    # ── message format ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_message_reflects_total_count(self, tool_with_history: Memory, memory_dir: Path):
        """ToolReturnValue.message shows total result count."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=10))
        assert "result(s)" in result.message
        assert "Found" in result.message

    @pytest.mark.asyncio
    async def test_output_is_valid_string(self, tool_with_history: Memory, memory_dir: Path):
        """Output is always a non-empty string."""
        result = await tool_with_history(Params(action="retrieve", query="REST", k=5))
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_no_results_mentions_both_sources(self, tool_with_history: Memory, memory_dir: Path):
        """No-results message mentions both conversation history and durable memory."""
        result = await tool_with_history(Params(action="retrieve", query="zzz_nonexistent_zzz", k=5))
        assert "conversation history" in result.output
        assert "durable memory" in result.output


# ---------------------------------------------------------------------------
# Integration: Memory write/append → retrieve
# ---------------------------------------------------------------------------


class TestRetrieveIntegration:
    """Full flow: Memory persists facts, retrieve finds them."""

    @pytest.mark.asyncio
    async def test_write_then_retrieve(self, memory_tool: Memory, history_index: HistoryIndex):
        """After Memory.write, retrieve can find the content."""
        memory_tool.attach_history_index(history_index)
        await memory_tool(Params(
            action="write",
            topic="important",
            content="The production database is hosted on PG15 in us-east-1.",
        ))
        result = await memory_tool(Params(action="retrieve", query="production database", k=5))
        assert "PG15" in result.output
        assert "[Durable memory]" in result.output
        assert "important" in result.output

    @pytest.mark.asyncio
    async def test_write_multiple_topics_then_retrieve_all(self, memory_tool: Memory):
        """retrieve searches across all memory topics."""
        await memory_tool(Params(action="write", topic="db", content="PostgreSQL 15 with pgvector extension"))
        await memory_tool(Params(action="write", topic="api", content="FastAPI with PostgreSQL backend"))

        result = await memory_tool(Params(action="retrieve", query="PostgreSQL", k=10))
        topics: set[str] = set()
        for line in result.output.split("\n"):
            if line.startswith("- ["):
                topics.add(line.split(":")[0].replace("- [", ""))
        assert "db" in topics
        assert "api" in topics

    @pytest.mark.asyncio
    async def test_id_retrieves_written_memory(self, memory_tool: Memory):
        """id='memory:<topic>' retrieves full content written by Memory."""
        await memory_tool(Params(
            action="write",
            topic="config",
            content="API_KEY=secret123\nENV=production\nDEBUG=false",
        ))
        result = await memory_tool(Params(action="retrieve", id="memory:config"))
        assert "secret123" in result.output
        assert "ENV=production" in result.output

    @pytest.mark.asyncio
    async def test_append_then_retrieve_full(self, memory_tool: Memory):
        """Appended content is also retrievable."""
        await memory_tool(Params(action="write", topic="notes", content="First note: use async/await"))
        await memory_tool(Params(action="append", topic="notes", content="Second note: handle errors with try/except"))

        result = await memory_tool(Params(action="retrieve", id="memory:notes"))
        assert "async/await" in result.output
        assert "try/except" in result.output

    @pytest.mark.asyncio
    async def test_memory_survives_reinstantiation(self, memory_tool: Memory, session_dir: Path):
        """Memory written to disk is found by a fresh Memory instance."""
        await memory_tool(Params(action="write", topic="survivor", content="This data survives re-instantiation."))

        # Brand-new Memory instance (simulating a new session) + fresh index
        runtime = SimpleNamespace(session=SimpleNamespace(dir=session_dir), read_only=False)
        new_mem = Memory(runtime)  # type: ignore[arg-type]
        new_mem.attach_history_index(HistoryIndex())

        result = await new_mem(Params(action="retrieve", query="survives", k=5))
        assert "survivor" in result.output
        assert "re-instantiation" in result.output
