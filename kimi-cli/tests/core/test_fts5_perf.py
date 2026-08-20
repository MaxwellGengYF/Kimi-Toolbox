"""Performance smoke tests guarding the FTS5 adoption win (plan §7.4).

Marked ``pytest.mark.benchmark`` so CI can skip them with
``-m "not benchmark"``; loose bounds avoid flakiness on slow machines.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.wire.types import TextPart

pytestmark = pytest.mark.benchmark


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextPart(text=text)])


def _seed_history(tmp_path: Path, count: int = 5000) -> Path:
    db_path = tmp_path / "history.db"
    idx = HistoryIndex(db_path=db_path)
    idx.load()
    batch = [
        _msg("user", f"Message {i} about sqlite and python deployment") for i in range(count)
    ]
    started = time.perf_counter()
    idx.index_messages(batch)
    elapsed = time.perf_counter() - started
    idx.close()
    assert elapsed < 2.0, f"batched indexing of {count} turns took {elapsed:.2f}s"
    return db_path


def test_fts_search_common_keyword_returns_rows(tmp_path: Path):
    db_path = _seed_history(tmp_path)
    idx = HistoryIndex(db_path=db_path)
    idx.load()
    started = time.perf_counter()
    results = idx.search("sqlite", top_k=5)
    elapsed = time.perf_counter() - started
    idx.close()
    assert len(results) >= 1  # no silent stop-term pruning
    assert elapsed < 0.05, f"FTS search took {elapsed:.3f}s"


def test_reopen_and_first_search_is_fast(tmp_path: Path):
    db_path = _seed_history(tmp_path)
    started = time.perf_counter()
    idx = HistoryIndex(db_path=db_path)
    idx.load()
    results = idx.search("deployment", top_k=1)
    elapsed = time.perf_counter() - started
    idx.close()
    assert len(results) >= 1
    assert elapsed < 0.05, f"reopen + first search took {elapsed:.3f}s"


def test_history_db_size_bounded(tmp_path: Path):
    db_path = _seed_history(tmp_path)
    size = db_path.stat().st_size
    assert size < 10 * 1024 * 1024, f"history.db grew to {size} bytes"


def test_cjk_trigram_search(tmp_path: Path):
    db_path = tmp_path / "history.db"
    idx = HistoryIndex(db_path=db_path)
    idx.load()
    idx.index_messages([_msg("user", "关于Python安装和配置")] * 200)
    idx.close()
    idx = HistoryIndex(db_path=db_path)
    idx.load()
    started = time.perf_counter()
    results = idx.search("安装", top_k=3)
    elapsed = time.perf_counter() - started
    idx.close()
    assert len(results) >= 1
    assert elapsed < 0.05, f"trigram search took {elapsed:.3f}s"
