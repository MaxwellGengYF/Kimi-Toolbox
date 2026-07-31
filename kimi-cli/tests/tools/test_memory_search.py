"""Tests for public memory search utilities (search_memory_files, SearchHit, topic_path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.tools.memory import (
    SearchHit,
    sanitize_topic,
    search_memory_files,
    topic_path,
)


class TestSearchMemoryFiles:
    """Unit tests for search_memory_files function."""

    @pytest.fixture
    def memory_dir(self, tmp_path: Path) -> Path:
        """Create a temporary memory dir with multiple .md files."""
        md = tmp_path / "memory"
        md.mkdir()
        (md / "decisions.md").write_text(
            "# Decisions\n\n- Use REST for all APIs\n- OAuth2 for auth\n- Rate limiting at gateway\n",
            encoding="utf-8",
        )
        (md / "architecture.md").write_text(
            "# Architecture\n\n- Microservices with gRPC\n- REST gateway at edge\n- PostgreSQL for persistence\n",
            encoding="utf-8",
        )
        (md / "onboarding.md").write_text(
            "# Onboarding\n\n- Setup dev environment\n- Run tests with pytest\n",
            encoding="utf-8",
        )
        return md

    def test_returns_SearchHit_instances(self, memory_dir: Path):
        """Results are SearchHit dataclass instances with correct fields."""
        hits = search_memory_files(memory_dir, "REST", 5)
        assert len(hits) >= 1
        for hit in hits:
            assert isinstance(hit, SearchHit)
            assert isinstance(hit.topic, str)
            assert isinstance(hit.line_no, int)
            assert isinstance(hit.snippet, str)
            assert isinstance(hit.score, int)
            assert hit.score > 0

    def test_finds_matches_across_multiple_files(self, memory_dir: Path):
        """Search matches spans multiple .md files."""
        hits = search_memory_files(memory_dir, "REST", 10)
        topics = {h.topic for h in hits}
        assert "decisions" in topics
        assert "architecture" in topics

    def test_no_match_returns_empty(self, memory_dir: Path):
        """Query with no matches returns empty list."""
        hits = search_memory_files(memory_dir, "zzz_nonexistent_zzz", 5)
        assert hits == []

    def test_empty_query_returns_empty(self, memory_dir: Path):
        """Empty or whitespace-only query returns empty list."""
        assert search_memory_files(memory_dir, "", 5) == []
        assert search_memory_files(memory_dir, "   ", 5) == []

    def test_respects_max_results(self, memory_dir: Path):
        """max_results parameter limits the number of returned hits."""
        hits = search_memory_files(memory_dir, "REST", 1)
        assert len(hits) <= 1

    def test_scores_higher_for_more_matches(self, memory_dir: Path):
        """Lines with more term occurrences score higher."""
        # Write a file with varying match density
        (memory_dir / "scoring.md").write_text(
            "REST REST REST\nREST\nno match here\n",
            encoding="utf-8",
        )
        hits = search_memory_files(memory_dir, "REST", 5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)  # sorted descending

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        """Non-existent directory returns empty list gracefully."""
        hits = search_memory_files(tmp_path / "nonexistent", "REST", 5)
        assert hits == []

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        """Directory with no .md files returns empty list."""
        md = tmp_path / "empty_memory"
        md.mkdir()
        hits = search_memory_files(md, "REST", 5)
        assert hits == []

    def test_non_md_files_ignored(self, memory_dir: Path):
        """.txt and other non-.md files are not searched."""
        (memory_dir / "notes.txt").write_text("REST API notes", encoding="utf-8")
        hits = search_memory_files(memory_dir, "REST", 10)
        topics = {h.topic for h in hits}
        assert "notes" not in topics

    def test_case_insensitive_search(self, memory_dir: Path):
        """Search is case-insensitive."""
        hits_lower = search_memory_files(memory_dir, "rest", 10)
        hits_upper = search_memory_files(memory_dir, "REST", 10)
        assert len(hits_lower) == len(hits_upper)

    def test_snippet_truncation_long_lines(self, memory_dir: Path):
        """Long lines are truncated with ellipsis in snippet."""
        long_line = "prefix " + "REST " * 100 + "suffix"
        (memory_dir / "long.md").write_text(long_line, encoding="utf-8")
        hits = search_memory_files(memory_dir, "REST", 1)
        assert len(hits) == 1
        # Snippet should be shorter than the full line
        assert len(hits[0].snippet) < len(long_line)
        assert "…" in hits[0].snippet

    def test_topic_name_uses_stem_not_full_path(self, memory_dir: Path):
        """Topic in SearchHit is the file stem, not full path."""
        hits = search_memory_files(memory_dir, "REST", 1)
        for hit in hits:
            assert "/" not in hit.topic
            assert "\\" not in hit.topic
            assert ".md" not in hit.topic


class TestTopicPath:
    """Tests for topic_path public function."""

    def test_returns_correct_path(self, tmp_path: Path):
        path = topic_path(tmp_path, "my_topic")
        assert path == tmp_path / "my_topic.md"

    def test_sanitizes_topic_name(self, tmp_path: Path):
        """Topic names with special chars are sanitized."""
        path = topic_path(tmp_path, "My Topic!@#")
        expected = tmp_path / f"{sanitize_topic('My Topic!@#')}.md"
        assert path == expected
        assert "!" not in str(path)

    def test_empty_topic_defaults(self, tmp_path: Path):
        """Empty topic name falls back to default."""
        path = topic_path(tmp_path, "")
        assert path.name == "memory.md"  # DEFAULT_TOPIC


class TestSearchHitDataclass:
    """Tests for SearchHit dataclass."""

    def test_immutable(self):
        hit = SearchHit(topic="test", line_no=1, snippet="hello", score=5)
        with pytest.raises(Exception):
            hit.score = 10  # frozen=True prevents mutation

    def test_equality(self):
        a = SearchHit(topic="t", line_no=1, snippet="s", score=5)
        b = SearchHit(topic="t", line_no=1, snippet="s", score=5)
        assert a == b

    def test_hashable(self):
        """Frozen dataclass with slots=True should be hashable."""
        hit = SearchHit(topic="t", line_no=1, snippet="s", score=5)
        assert hash(hit) is not None
        # Can be used in a set
        s = {hit}
        assert len(s) == 1
