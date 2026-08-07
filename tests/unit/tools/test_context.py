"""Tests for Defects 14.1-14.3: ContextUsage / Compact improvements."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from kimix.tools.context import ContextUsage, ContextUsageParams


class TestContextUsageExtras:
    async def test_extras_contains_structured_data(self, mock_soul: MagicMock) -> None:
        # Patch get_current_soul_or_none to return our mock
        with patch("kimix.tools.context.get_current_soul_or_none", return_value=mock_soul):
            mock_soul.status.context_usage = 0.425
            mock_soul.status.context_tokens = 128000
            mock_soul.status.max_context_tokens = 300000
            cu = ContextUsage()
            result = await cu(ContextUsageParams())
            assert result.extras is not None
            assert result.extras["context_usage_pct"] == 42.5
            assert result.extras["used_tokens"] == 128000
            assert result.extras["max_tokens"] == 300000
            assert result.extras["free_tokens"] == 172000

    async def test_output_still_human_readable(self, mock_soul: MagicMock) -> None:
        with patch("kimix.tools.context.get_current_soul_or_none", return_value=mock_soul):
            mock_soul.status.context_usage = 0.5
            mock_soul.status.context_tokens = 50000
            mock_soul.status.max_context_tokens = 100000
            cu = ContextUsage()
            result = await cu(ContextUsageParams())
            assert "%" in result.output
            assert "tokens" in result.output.lower()
