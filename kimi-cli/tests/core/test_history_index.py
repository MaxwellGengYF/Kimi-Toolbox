"""Tests for kimi_cli.soul.history_index — BM25 index over conversation turns."""

from __future__ import annotations

import time
from pathlib import Path

from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.wire.types import TextPart


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextPart(text=text)])


class TestHistoryIndex:
    def test_search_empty_index(self):
        idx = HistoryIndex()
        assert idx.search("anything") == []

    def test_index_and_search(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "How do I compile Python?")])
        idx.index_messages([_msg("assistant", "Use pyinstaller or cx_Freeze.")])

        results = idx.search("compile Python", top_k=2)
        # Only the user message should match the query
        assert len(results) == 1
        assert results[0]["role"] == "user"
        assert "compile Python" in results[0]["text"]

    def test_skips_system_and_tool_roles(self):
        idx = HistoryIndex()
        idx.index_messages([
            Message(role="system", content=[TextPart(text="System prompt")]),
            _msg("user", "Hello"),
        ])
        assert len(idx._turns) == 1
        assert idx._turns[0]["role"] == "user"

    def test_skips_empty_messages(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "   ")])
        assert len(idx._turns) == 0

    def test_mark_compacted(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "Question 1")])
        idx.mark_compacted()
        idx.index_messages([_msg("user", "Question 2")])

        assert idx._turns[0]["is_compacted"] is True
        assert idx._turns[1]["is_compacted"] is False

    def test_max_turns_bound(self):
        idx = HistoryIndex()
        for i in range(510):
            idx.index_messages([_msg("user", f"Message {i}")])
        assert len(idx._turns) == 500

    def test_persistence_roundtrip(self, tmp_path: Path):
        persist_path = tmp_path / "history.json"
        idx = HistoryIndex(persist_path=persist_path)
        # Index at least 2 docs so BM25 stop-word pruning doesn't remove all terms
        idx.index_messages([_msg("user", "Persistent question")])
        idx.index_messages([_msg("assistant", "Unrelated answer about Java")])
        idx.save()

        idx2 = HistoryIndex(persist_path=persist_path)
        assert idx2.load() is True
        assert len(idx2._turns) == 2
        assert idx2._turns[0]["text"] == "Persistent question"

        # Search should work after reload
        results = idx2.search("persistent", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Persistent question"

    def test_clear(self, tmp_path: Path):
        persist_path = tmp_path / "history.json"
        idx = HistoryIndex(persist_path=persist_path)
        idx.index_messages([_msg("user", "To be cleared")])
        idx.save()
        idx.clear()

        assert len(idx._turns) == 0
        assert not persist_path.exists()

    def test_load_missing_file(self):
        idx = HistoryIndex(persist_path=Path("/nonexistent/path.json"))
        assert idx.load() is False

    def test_search_filters_by_compacted(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "First question about Python")])
        idx.mark_compacted()
        idx.index_messages([_msg("user", "Second question about Java")])

        results = idx.search("Python", top_k=3)
        compacted = [r for r in results if r.get("is_compacted")]
        assert len(compacted) >= 1
        assert compacted[0]["text"] == "First question about Python"

    def test_search_returns_verbatim(self):
        idx = HistoryIndex()
        text = "The exact original text must be preserved."
        msgs = [
            Message(role="user", content=[TextPart(text=text)]),
            Message(role="assistant", content=[TextPart(text="Something completely different.")]),
        ]
        idx.index_messages(msgs)
        results = idx.search("original text", top_k=1)
        assert len(results) >= 1
        assert results[0]["text"] == text

    def test_search_with_recency_boosts_newer_docs(self):
        idx = HistoryIndex()
        # Index docs with a shared rare term plus unrelated docs so BM25 IDF
        # for the shared term is non-zero (needs enough docs for IDF > 0).
        idx.index_messages([_msg("user", "older document about alpha programming")])
        idx._turns[0]["timestamp"] = time.time() - 7200  # 2 hours ago
        idx.index_messages([_msg("assistant", "unrelated answer about java")])
        idx.index_messages([_msg("user", "newer document about alpha programming")])
        idx.index_messages([_msg("assistant", "another unrelated answer about golang")])

        results = idx.search_with_recency("alpha programming", top_k=2, recency_weight=1.0)
        # Filter to the two matching docs
        matching = [r for r in results if "alpha" in r["text"]]
        assert len(matching) == 2
        # With the default 24-hour decay, the 2-hour-old doc gets ~1.92 boost, the new doc gets ~2.0 boost
        # so the newer doc should rank first despite identical BM25 scores
        assert matching[0]["turn_id"] == 2
        assert matching[1]["turn_id"] == 0
        assert matching[0]["boosted_score"] > matching[1]["boosted_score"]

    def test_search_with_recency_weight_zero_falls_back_to_bm25(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "older document about alpha programming")])
        idx._turns[0]["timestamp"] = time.time() - 7200
        idx.index_messages([_msg("assistant", "unrelated answer about java")])
        idx.index_messages([_msg("user", "newer document about alpha programming")])
        idx.index_messages([_msg("assistant", "another unrelated answer about golang")])

        results = idx.search_with_recency("alpha programming", top_k=2, recency_weight=0.0)
        # With zero recency weight, ordering should match pure BM25
        bm25_results = idx.search("alpha programming", top_k=2)
        assert [r["turn_id"] for r in results] == [r["turn_id"] for r in bm25_results]

    def test_search_with_recency_preserves_original_score(self):
        idx = HistoryIndex()
        idx.index_messages([_msg("user", "unique search term xyz")])
        idx.index_messages([_msg("assistant", "something unrelated about java")])
        results = idx.search_with_recency("unique search term xyz", top_k=1)
        assert len(results) == 1
        assert "score" in results[0]
        assert "boosted_score" in results[0]
        assert results[0]["boosted_score"] >= results[0]["score"]


class TestHistoryIndexFts:
    """FTS5-backed HistoryIndex (history.db) — durable, uncapped, CJK-capable."""

    def _make(self, tmp_path: Path, name: str = "history.db") -> HistoryIndex:
        idx = HistoryIndex(db_path=tmp_path / name)
        idx.load()
        return idx

    def test_load_returns_false_when_missing(self, tmp_path: Path):
        idx = HistoryIndex(db_path=tmp_path / "missing.db")
        assert idx.load() is False
        idx.close()

    def test_search_over_500_turns(self, tmp_path: Path):
        idx = self._make(tmp_path)
        for i in range(510):
            idx.index_messages([_msg("user", f"Message {i} about sqlite")])
        assert len(idx._turns) == 510
        results = idx.search("sqlite", top_k=3)
        assert len(results) == 3
        idx.close()

    def test_search_returns_verbatim(self, tmp_path: Path):
        idx = self._make(tmp_path)
        text = "The exact original text must be preserved."
        msgs = [
            Message(role="user", content=[TextPart(text=text)]),
            Message(role="assistant", content=[TextPart(text="Something completely different.")]),
        ]
        idx.index_messages(msgs)
        results = idx.search("original text", top_k=1)
        assert len(results) >= 1
        assert results[0]["text"] == text
        idx.close()

    def test_skips_system_and_empty(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([
            Message(role="system", content=[TextPart(text="System prompt")]),
            _msg("user", "   "),
            _msg("user", "Hello"),
        ])
        assert len(idx._turns) == 1
        assert idx._turns[0]["role"] == "user"
        idx.close()

    def test_mark_compacted_and_search(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([_msg("user", "First question about Python")])
        idx.mark_compacted()
        idx.index_messages([_msg("user", "Second question about Java")])

        assert idx._turns[0]["is_compacted"] is True
        assert idx._turns[1]["is_compacted"] is False
        results = idx.search("Python", top_k=3)
        compacted = [r for r in results if r.get("is_compacted")]
        assert len(compacted) >= 1
        assert compacted[0]["text"] == "First question about Python"
        idx.close()

    def test_cjk_search_trigram(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([
            _msg("user", "关于Python安装和配置"),
            _msg("assistant", "日本語の検索テストです"),
        ])
        results = idx.search("日本語", top_k=3)
        assert any("日本語" in r["text"] for r in results)
        results = idx.search("配置", top_k=3)
        assert any("配置" in r["text"] for r in results)
        idx.close()

    def test_lone_cjk_char_uses_like_fallback(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([_msg("user", "日本語のテストです")])
        results = idx.search("日", top_k=3)
        assert any("日本語" in r["text"] for r in results)
        idx.close()

    def test_like_fallback_escapes_wildcards(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([_msg("user", "progress is 50% done")])
        # '%' is preserved for CJK-ish LIKE; for Latin the sanitizer strips it,
        # so search for the literal phrase instead.
        results = idx.search("50 done", top_k=3)
        assert any("50% done" in r["text"] for r in results)
        idx.close()

    def test_persistence_reopen(self, tmp_path: Path):
        db_path = tmp_path / "history.db"
        idx = HistoryIndex(db_path=db_path)
        assert idx.load() is False
        idx.index_messages([_msg("user", "Persistent question")])
        idx.index_messages([_msg("assistant", "Unrelated answer about Java")])
        idx.mark_compacted()
        idx.save()
        idx.close()

        idx2 = HistoryIndex(db_path=db_path)
        assert idx2.load() is True
        assert len(idx2._turns) == 2
        assert idx2._turns[0]["text"] == "Persistent question"
        results = idx2.search("persistent", top_k=1)
        assert len(results) == 1
        assert results[0]["is_compacted"] is True
        idx2.close()

    def test_legacy_json_migration(self, tmp_path: Path):
        import orjson

        legacy = tmp_path / "history_index" / "sess.json"
        legacy.parent.mkdir(parents=True)
        turns = [
            {"turn_id": 0, "role": "user", "text": "Old question", "timestamp": 100.0, "is_compacted": False},
            {"turn_id": 1, "role": "assistant", "text": "Old answer", "timestamp": 101.0, "is_compacted": True},
        ]
        legacy.write_text(orjson.dumps({"doc_id_counter": 2, "turns": turns}).decode(), encoding="utf-8")

        db_path = tmp_path / "history.db"
        idx = HistoryIndex(db_path=db_path, legacy_json_path=legacy)
        assert idx.load() is True
        assert len(idx._turns) == 2
        assert idx._turns[1]["is_compacted"] is True
        assert not legacy.exists()  # renamed to .bak
        assert legacy.with_suffix(".json.bak").exists()
        results = idx.search("old question", top_k=1)
        assert len(results) == 1
        idx.close()

    def test_get_by_id_with_prune_prefix(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([_msg("user", "First"), _msg("user", "Second")])
        turn = idx.get_by_id("prune_1")
        assert turn is not None
        assert turn["turn_id"] == 1
        assert idx.get_by_id("prune_99") is None
        assert idx.get_by_id("not-a-number") is None
        idx.close()

    def test_clear_removes_db(self, tmp_path: Path):
        db_path = tmp_path / "history.db"
        idx = HistoryIndex(db_path=db_path)
        idx.load()
        idx.index_messages([_msg("user", "To be cleared")])
        assert db_path.exists()
        idx.clear()
        assert not db_path.exists()
        assert len(idx._turns) == 0
        # Index works again after clear (fresh empty schema)
        idx.index_messages([_msg("user", "Fresh start")])
        assert len(idx._turns) == 1
        idx.close()

    def test_search_with_recency_works(self, tmp_path: Path):
        idx = self._make(tmp_path)
        idx.index_messages([_msg("user", "older document about alpha programming")])
        # _turns returns copies in FTS mode, so update the persisted timestamp
        # through the connection directly.
        idx._ensure_conn().execute(
            "UPDATE turns SET timestamp = ? WHERE turn_id = 0", (time.time() - 7200,)
        )
        idx.index_messages([_msg("assistant", "unrelated answer about java")])
        idx.index_messages([_msg("user", "newer document about alpha programming")])
        idx.index_messages([_msg("assistant", "another unrelated answer about golang")])

        results = idx.search_with_recency("alpha programming", top_k=2, recency_weight=1.0)
        matching = [r for r in results if "alpha" in r["text"]]
        assert len(matching) == 2
        # 0-based turn ids: older=0, newer=2
        assert matching[0]["turn_id"] == 2
        assert matching[1]["turn_id"] == 0
        assert matching[0]["boosted_score"] > matching[1]["boosted_score"]
        assert matching[0]["boosted_score"] >= matching[0]["score"]
        idx.close()


class TestHistoryIndexPhaseC:
    """Phase C Hermes parity: incremental merge, stale breadcrumbs, rebuild_fts."""

    def test_rebuild_fts_recovers_corrupt_index(self, tmp_path: Path):
        idx = HistoryIndex(db_path=tmp_path / "history.db")
        idx.load()
        idx.index_messages([_msg("user", "Question about sqlite")])
        # Corrupt: drop the unicode61 index
        idx._ensure_conn().execute("DROP TABLE turns_fts")
        results = idx.search("sqlite", top_k=3)
        assert len(results) == 1  # LIKE fallback still answers
        assert idx._fts_stale is True

        idx.rebuild_fts()
        assert idx._fts_stale is False
        results = idx.search("sqlite", top_k=3)
        assert len(results) == 1  # FTS restored
        idx.close()

    def test_stale_marker_persisted(self, tmp_path: Path):
        db_path = tmp_path / "history.db"
        idx = HistoryIndex(db_path=db_path)
        idx.load()
        idx.index_messages([_msg("user", "stale marker test about sqlite")])
        idx._ensure_conn().execute("DROP TABLE turns_fts")
        idx.search("sqlite", top_k=3)  # sets stale marker
        idx.close()

        idx2 = HistoryIndex(db_path=db_path)
        idx2.load()
        assert idx2._fts_stale is True
        # Searches go straight to LIKE while stale
        results = idx2.search("sqlite", top_k=3)
        assert len(results) == 1
        idx2.close()

    def test_incremental_merge_no_error(self, tmp_path: Path):
        from kimi_cli.soul.history_index import _FTS_MERGE_INTERVAL

        idx = HistoryIndex(db_path=tmp_path / "history.db")
        idx.load()
        for i in range(_FTS_MERGE_INTERVAL + 10):
            idx.index_messages([_msg("user", f"merge message {i} about sqlite")])
        results = idx.search("sqlite", top_k=3)
        assert len(results) == 3
        assert idx._writes_since_merge == 10  # reset after merge fired
        idx.close()

    def test_rebuild_fts_with_legacy_mode_is_noop(self, tmp_path: Path):
        idx = HistoryIndex(persist_path=tmp_path / "history.json")
        idx.rebuild_fts()  # must not raise
        idx.close()
