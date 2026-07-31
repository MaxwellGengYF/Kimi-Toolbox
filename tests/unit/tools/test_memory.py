"""Unit tests for the durable Memory tool (kimi_cli.tools.memory)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from kosong.tooling import ToolError, ToolOk

from kimi_cli.tools.memory import (
    _MAX_FILE_BYTES,
    _MAX_READ_CHARS,
    _MAX_TOPICS,
    _MAX_WRITE_BYTES,
    Memory,
    Params,
    build_memory_restore_text,
    flush_pre_compact_state,
    list_memory_files,
    memory_dir_for_session,
    sanitize_topic,
)


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "session"


@pytest.fixture
def tool(session_dir: Path) -> Memory:
    runtime = SimpleNamespace(session=SimpleNamespace(dir=session_dir), read_only=False)
    return Memory(runtime)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSanitizeTopic:
    def test_plain(self) -> None:
        assert sanitize_topic("decisions") == "decisions"

    def test_path_traversal_removed(self) -> None:
        assert sanitize_topic("../evil/../../etc") == "evil_etc"

    def test_spaces_and_case(self) -> None:
        assert sanitize_topic("My Notes!") == "my_notes"

    def test_empty_falls_back_to_default(self) -> None:
        assert sanitize_topic("") == "memory"
        assert sanitize_topic("!!!") == "memory"

    def test_truncated_to_64_chars(self) -> None:
        assert len(sanitize_topic("a" * 200)) == 64

    def test_validator_applied_in_params(self) -> None:
        assert Params(action="read", topic="../X Y").topic == "x_y"


class TestMemoryDir:
    def test_path(self, session_dir: Path) -> None:
        assert memory_dir_for_session(session_dir) == session_dir / "memory"

    def test_list_missing_dir(self, tmp_path: Path) -> None:
        assert list_memory_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# write / append / read
# ---------------------------------------------------------------------------


class TestWrite:
    async def test_write_creates_file(self, tool: Memory, session_dir: Path) -> None:
        result = await tool(Params(action="write", topic="facts", content="sky is blue"))
        assert isinstance(result, ToolOk)
        path = session_dir / "memory" / "facts.md"
        assert path.read_text(encoding="utf-8") == "sky is blue\n"

    async def test_write_overwrites(self, tool: Memory, session_dir: Path) -> None:
        await tool(Params(action="write", topic="facts", content="v1"))
        await tool(Params(action="write", topic="facts", content="v2"))
        assert (session_dir / "memory" / "facts.md").read_text(encoding="utf-8") == "v2\n"

    async def test_write_requires_content(self, tool: Memory) -> None:
        result = await tool(Params(action="write", topic="facts", content=None))
        assert isinstance(result, ToolError)

    async def test_write_rejects_blank_content(self, tool: Memory) -> None:
        result = await tool(Params(action="write", topic="facts", content="   "))
        assert isinstance(result, ToolError)

    async def test_write_enforces_per_write_limit(self, tool: Memory) -> None:
        result = await tool(
            Params(action="write", topic="big", content="x" * (_MAX_WRITE_BYTES + 1))
        )
        assert isinstance(result, ToolError)
        assert "per-write limit" in result.message

    async def test_append_adds_content(self, tool: Memory, session_dir: Path) -> None:
        await tool(Params(action="write", topic="log", content="line1"))
        result = await tool(Params(action="append", topic="log", content="line2"))
        assert isinstance(result, ToolOk)
        text = (session_dir / "memory" / "log.md").read_text(encoding="utf-8")
        assert "line1" in text and "line2" in text
        assert text.index("line1") < text.index("line2")

    async def test_append_creates_when_missing(self, tool: Memory, session_dir: Path) -> None:
        await tool(Params(action="append", topic="new", content="hello"))
        assert "hello" in (session_dir / "memory" / "new.md").read_text(encoding="utf-8")

    async def test_append_enforces_file_limit(self, tool: Memory, session_dir: Path) -> None:
        path = session_dir / "memory"
        path.mkdir(parents=True)
        (path / "full.md").write_bytes(b"y" * _MAX_FILE_BYTES)
        result = await tool(Params(action="append", topic="full", content="more"))
        assert isinstance(result, ToolError)
        assert "per-file limit" in result.message

    async def test_topic_limit(self, tool: Memory, session_dir: Path) -> None:
        mem = session_dir / "memory"
        mem.mkdir(parents=True)
        for i in range(_MAX_TOPICS):
            (mem / f"t{i}.md").write_text("x", encoding="utf-8")
        result = await tool(Params(action="write", topic="one_too_many", content="x"))
        assert isinstance(result, ToolError)
        assert "topic limit" in result.message


class TestRead:
    async def test_read_returns_content(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="facts", content="remember this"))
        result = await tool(Params(action="read", topic="facts"))
        assert isinstance(result, ToolOk)
        assert "remember this" in result.output

    async def test_read_missing_topic(self, tool: Memory) -> None:
        result = await tool(Params(action="read", topic="ghost"))
        assert isinstance(result, ToolError)
        assert "ghost" in result.message

    async def test_read_missing_topic_suggests_fuzzy_match(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="facts", content="some facts"))
        await tool(Params(action="write", topic="decisions", content="some decisions"))
        # "fact" is a close match to "facts" — should trigger suggestion
        result = await tool(Params(action="read", topic="fact"))
        assert isinstance(result, ToolError)
        assert "Did you mean: 'facts'?" in result.message
        # "decisions" is too far from "fact" phonetically — should not appear
        assert "decisions" not in result.message

    async def test_read_missing_topic_no_fuzzy_match(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="xyz", content="something"))
        # "abcdef" shares no meaningful similarity with "xyz"
        result = await tool(Params(action="read", topic="abcdef"))
        assert isinstance(result, ToolError)
        assert "Did you mean:" not in result.message
        assert "Use action='list'" in result.message
        assert "No memory topic named 'abcdef'" in result.message

    async def test_read_missing_topic_empty_memory_dir(self, tool: Memory) -> None:
        """No memory files exist at all — fall back to list hint."""
        result = await tool(Params(action="read", topic="anything"))
        assert isinstance(result, ToolError)
        assert "Use action='list'" in result.message
        assert "Did you mean:" not in result.message

    async def test_read_exact_match_unaffected(self, tool: Memory) -> None:
        """Exact topic match must still work — not fall through to fuzzy."""
        await tool(Params(action="write", topic="fact", content="short name"))
        await tool(Params(action="write", topic="facts", content="plural name"))
        result = await tool(Params(action="read", topic="fact"))
        assert isinstance(result, ToolOk)
        assert "short name" in result.output
        assert "plural name" not in result.output

    async def test_read_truncates_huge_files(self, tool: Memory, session_dir: Path) -> None:
        mem = session_dir / "memory"
        mem.mkdir(parents=True)
        (mem / "huge.md").write_text("z" * (_MAX_READ_CHARS + 5000), encoding="utf-8")
        result = await tool(Params(action="read", topic="huge"))
        assert isinstance(result, ToolOk)
        assert "Truncated" in result.output
        assert len(result.output) < _MAX_READ_CHARS + 500
    async def test_default_action_is_retrieve(self, tool: Memory) -> None:
        """Default action is 'retrieve'; with no query/id it returns guidance."""
        result = await tool(Params())
        assert isinstance(result, ToolOk)
        assert "No query provided" in result.output

    async def test_read_default_topic_explicit_action(self, tool: Memory) -> None:
        """Explicit action='read' still reads the default 'memory' topic."""
        await tool(Params(action="write", topic="memory", content="default topic"))
        result = await tool(Params(action="read", topic="memory"))
        assert isinstance(result, ToolOk)
        assert "default topic" in result.output


class TestList:
    async def test_list_empty(self, tool: Memory) -> None:
        result = await tool(Params(action="list"))
        assert isinstance(result, ToolOk)
        assert "No memory topics" in result.output

    async def test_list_shows_topics(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="a", content="1"))
        await tool(Params(action="write", topic="b", content="2"))
        result = await tool(Params(action="list"))
        assert "2 memory topic(s)" in result.output
        assert "- a " in result.output and "- b " in result.output


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_requires_query(self, tool: Memory) -> None:
        result = await tool(Params(action="search", query=""))
        assert isinstance(result, ToolError)

    async def test_search_finds_match(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="facts", content="the password is hunter2"))
        result = await tool(Params(action="search", query="password"))
        assert isinstance(result, ToolOk)
        assert "hunter2" in result.output
        assert "facts:1" in result.output

    async def test_search_no_results(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="facts", content="nothing relevant"))
        result = await tool(Params(action="search", query="unobtainium"))
        assert isinstance(result, ToolOk)
        assert "No memory entries" in result.output

    async def test_search_rebuilt_from_disk(self, tool: Memory, session_dir: Path) -> None:
        """Search must see files written outside the tool (survives any context event)."""
        mem = session_dir / "memory"
        mem.mkdir(parents=True)
        (mem / "external.md").write_text("written by another process", encoding="utf-8")
        result = await tool(Params(action="search", query="another process"))
        assert "external:1" in result.output

    async def test_search_ranks_by_score(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="many", content="foo foo foo"))
        await tool(Params(action="write", topic="few", content="foo"))
        result = await tool(Params(action="search", query="foo"))
        lines = result.output.splitlines()
        assert "many" in lines[1]

    async def test_search_respects_max_results(self, tool: Memory, session_dir: Path) -> None:
        mem = session_dir / "memory"
        mem.mkdir(parents=True)
        (mem / "multi.md").write_text("\n".join(f"hit line {i}" for i in range(10)), encoding="utf-8")
        result = await tool(Params(action="search", query="hit", max_results=3))
        assert "3 memory hit(s)" in result.output

    async def test_search_long_line_snippet_centered(self, tool: Memory) -> None:
        await tool(Params(action="write", topic="long", content="a" * 200 + "NEEDLE" + "b" * 200))
        result = await tool(Params(action="search", query="NEEDLE"))
        assert "NEEDLE" in result.output
        # snippet should be much shorter than the full 406-char line
        snippet_line = [l for l in result.output.splitlines() if "NEEDLE" in l][0]
        assert len(snippet_line) < 250


# ---------------------------------------------------------------------------
# pre-compaction flush / post-compaction restore
# ---------------------------------------------------------------------------


class TestFlushAndRestore:
    def test_flush_writes_snapshot(self, session_dir: Path) -> None:
        path = flush_pre_compact_state(
            session_dir,
            trigger_reason="auto",
            context_tokens=72_000,
            max_context_tokens=200_000,
            unfinished_todos=[("pending", "fix bug"), ("in_progress", "write tests")],
        )
        assert path is not None and path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "Pre-compaction state flush" in text
        assert "Trigger: auto" in text
        assert "72000/200000" in text
        assert "- [pending] fix bug" in text
        assert "- [in_progress] write tests" in text
        assert "done" not in text

    def test_flush_returns_none_when_nothing_to_persist(self, session_dir: Path) -> None:
        path = flush_pre_compact_state(
            session_dir,
            trigger_reason="manual",
            context_tokens=1,
            max_context_tokens=2,
            unfinished_todos=[],
        )
        assert path is None
        assert not memory_dir_for_session(session_dir).exists()

    def test_flush_overwrites_previous(self, session_dir: Path) -> None:
        flush_pre_compact_state(
            session_dir, trigger_reason="auto", context_tokens=1,
            max_context_tokens=2, unfinished_todos=[("pending", "old")],
        )
        flush_pre_compact_state(
            session_dir, trigger_reason="auto", context_tokens=1,
            max_context_tokens=2, unfinished_todos=[("pending", "new")],
        )
        text = (memory_dir_for_session(session_dir) / "pre_compact_state.md").read_text(encoding="utf-8")
        assert "new" in text and "old" not in text

    def test_restore_none_without_memory(self, session_dir: Path) -> None:
        assert build_memory_restore_text(session_dir) is None

    def test_restore_lists_files_and_flush_excerpt(self, session_dir: Path) -> None:
        flush_pre_compact_state(
            session_dir, trigger_reason="auto", context_tokens=1,
            max_context_tokens=2, unfinished_todos=[("pending", "task A")],
        )
        (memory_dir_for_session(session_dir) / "facts.md").write_text("fact", encoding="utf-8")
        text = build_memory_restore_text(session_dir)
        assert text is not None
        assert "Memory" in text and "survive compaction" in text
        assert "facts.md" in text and "pre_compact_state.md" in text
        assert "task A" in text  # flush excerpt included

    def test_restore_without_flush_still_lists(self, session_dir: Path) -> None:
        mem = memory_dir_for_session(session_dir)
        mem.mkdir(parents=True)
        (mem / "facts.md").write_text("fact", encoding="utf-8")
        text = build_memory_restore_text(session_dir)
        assert text is not None
        assert "facts.md" in text
        assert "Latest pre-compaction flush" not in text
