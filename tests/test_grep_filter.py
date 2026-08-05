"""Tests for OUTPUT_LIMIT (16384) and original file saving."""

import pytest

from kimix.tools.common import (
    OUTPUT_LIMIT,
    _maybe_export_output_async,
    _build_session_output_block,
)


# ---------------------------------------------------------------------------
# OUTPUT_LIMIT
# ---------------------------------------------------------------------------

def test_output_limit_is_16384() -> None:
    """OUTPUT_LIMIT must be 16384."""
    assert OUTPUT_LIMIT == 16384, (
        f"OUTPUT_LIMIT expected 16384, got {OUTPUT_LIMIT}"
    )


async def test_output_under_limit_not_exported() -> None:
    """Output <= OUTPUT_LIMIT should NOT be exported to file."""
    output = "a" * OUTPUT_LIMIT
    result = await _maybe_export_output_async(output)
    # Result should be the same string (not a file export message)
    assert result == output


async def test_output_over_limit_is_exported() -> None:
    """Output > OUTPUT_LIMIT should be exported to file."""
    output = "a" * (OUTPUT_LIMIT + 1)
    result = await _maybe_export_output_async(output)
    assert "exported to file" in result.lower()
    assert str(OUTPUT_LIMIT + 1) not in result  # The content itself should not be in the message


# ---------------------------------------------------------------------------
# _build_session_output_block with original_path
# ---------------------------------------------------------------------------

def test_build_block_includes_original_path_when_set() -> None:
    block = _build_session_output_block(
        task_id="test_task",
        status="completed",
        output="hello",
        original_path="/tmp/original.txt",
    )
    assert "original_path: /tmp/original.txt" in block


def test_build_block_original_path_null_when_none() -> None:
    block = _build_session_output_block(
        task_id="test_task",
        status="completed",
        output="hello",
    )
    assert "original_path: null" in block

