"""Tests for edit auto-repair helpers."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file.auto_repair import (
    build_hunks,
    compute_repair_region,
    isolate_culprit_hunks,
    repair_parse_regression,
    revert_hunks,
)
from kimi_cli.tools.file.parse_check import source_parses


def test_build_hunks_merges_adjacent_changes() -> None:
    prev = "a\nb\nc\n"
    next_ = "a\nX\nY\nc\n"
    hunks, a, b = build_hunks(prev, next_)
    assert len(hunks) == 1
    h = hunks[0]
    assert a[h.a_start:h.a_end] == ["b"]
    assert b[h.b_start:h.b_end] == ["X", "Y"]


def test_revert_hunks_restores_prev() -> None:
    prev = "a\nb\nc\n"
    next_ = "a\nX\nY\nc\n"
    hunks, a, b = build_hunks(prev, next_)
    restored = revert_hunks(a, b, hunks, [0])
    assert restored == prev


def test_isolate_culprit_hunks_finds_single() -> None:
    prev = "def foo():\n    pass\n"
    # One bad hunk plus one good hunk.
    next_ = "def foo():\n    pass\nbad syntax\n"
    hunks, a, b = build_hunks(prev, next_)
    culprits = isolate_culprit_hunks("foo.py", a, b, hunks)
    assert culprits is not None
    assert len(culprits) == 1


def test_isolate_culprit_hunks_pre_image_broken_returns_none() -> None:
    prev = "def foo(\n"
    next_ = "def foo():\n    pass\n"
    hunks, a, b = build_hunks(prev, next_)
    culprits = isolate_culprit_hunks("foo.py", a, b, hunks)
    assert culprits is None


def test_compute_repair_region_bounds() -> None:
    prev = "\n".join([f"line{i}" for i in range(20)]) + "\n"
    next_ = prev.replace("line10", "line10 syntax error")
    region = compute_repair_region("foo.py", prev, next_)
    assert region is not None
    assert region.b_start <= 10 <= region.b_end
    assert region.b_end - region.b_start <= 150
    assert source_parses(region.reference_text, "foo.py")


@pytest.mark.asyncio
async def test_repair_parse_regression_deterministic_indentation_drift() -> None:
    prev = "def foo():\n    if True:\n        pass\n"
    # model dropped indentation on the `pass` line.
    next_ = "def foo():\n    if True:\n    pass\n"
    repair = await repair_parse_regression("foo.py", prev, next_)
    assert repair is not None
    assert source_parses(repair.content, "foo.py")
    assert "    pass" in repair.content


@pytest.mark.asyncio
async def test_repair_parse_regression_rejects_plain_revert() -> None:
    prev = "x = 1\n"
    next_ = "x =\n"
    repair = await repair_parse_regression("foo.py", prev, next_)
    # A plain revert is rejected; no deterministic candidate parses.
    assert repair is None


@pytest.mark.asyncio
async def test_repair_parse_regression_model_phase_with_fake_completer() -> None:
    prev = "x = 1\n"
    next_ = "x =\n"

    async def fake_complete(prompt: str) -> str:
        return "x = 2\n"

    repair = await repair_parse_regression("foo.py", prev, next_, complete=fake_complete)
    assert repair is not None
    assert source_parses(repair.content, "foo.py")
    assert "x = 2" in repair.content
