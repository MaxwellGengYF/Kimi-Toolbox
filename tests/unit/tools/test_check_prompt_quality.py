"""Tests for the Q1-Q5 prompt-quality lint (tools/check_prompt_quality.py)."""

from __future__ import annotations

from tools.check_prompt_quality import _check_description

CLEAN = (
    "Read a UTF-8 text file and return line-numbered content. "
    "Negative offset = tail mode."
)


def test_clean_description_passes() -> None:
    assert _check_description("read", CLEAN) == []


def test_q1_flags_hoisted_convention_duplication() -> None:
    violations = _check_description(
        "bash", "Run a command. Output longer than max_lines is head+tail folded."
    )
    assert any(v.startswith("Q1:") for v in violations)


def test_q2_flags_schema_redundant_range() -> None:
    violations = _check_description(
        "bash", "Run a command. Timeout in seconds (1-900)."
    )
    assert any(v.startswith("Q2:") for v in violations)


def test_q3_flags_boilerplate() -> None:
    violations = _check_description(
        "glob", "This tool is a tool that finds files. Use it to search."
    )
    assert any(v.startswith("Q3:") for v in violations)


def test_q4_flags_duplicate_sentence() -> None:
    violations = _check_description(
        "read",
        "Read files and return line-numbered content. "
        "Read files and return line-numbered content.",
    )
    assert any(v.startswith("Q4:") for v in violations)


def test_q5_flags_empty_and_truncated() -> None:
    assert any(v.startswith("Q5:") for v in _check_description("x", ""))
    assert any(
        v.startswith("Q5:") for v in _check_description("x", "Find files by name...")
    )
