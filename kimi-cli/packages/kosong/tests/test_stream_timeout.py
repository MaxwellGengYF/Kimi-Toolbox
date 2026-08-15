import asyncio
from collections.abc import AsyncIterator

import pytest

from kosong.chat_provider import APITimeoutError, StreamedMessagePart
from kosong.contrib.chat_provider.common import (
    BaseStreamedMessage,
    get_stream_iteration_timeout,
    with_stream_timeout,
)
from kosong.message import TextPart


async def _fast_iterator() -> AsyncIterator[int]:
    for i in range(3):
        yield i


async def _stalled_iterator() -> AsyncIterator[int]:
    yield 1
    await asyncio.sleep(3600)
    yield 2


class _TestStreamedMessage(BaseStreamedMessage):
    def __init__(self, iterator: AsyncIterator[StreamedMessagePart]) -> None:
        self._iter = iterator


async def _text_iterator(*texts: str) -> AsyncIterator[StreamedMessagePart]:
    for text in texts:
        yield TextPart(text=text)


class TestStreamTimeoutHelper:
    """Tests for the shared streaming per-chunk timeout wrapper."""

    @pytest.mark.asyncio
    async def test_passes_through_all_items(self) -> None:
        result = [item async for item in with_stream_timeout(_fast_iterator(), timeout=10.0)]
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_raises_timeout_when_iterator_stalls(self) -> None:
        wrapped = with_stream_timeout(_stalled_iterator(), timeout=0.05)
        items: list[int] = []
        with pytest.raises(APITimeoutError) as exc_info:
            async for item in wrapped:
                items.append(item)
        assert items == [1]
        assert "0.05" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_aiter_returns_same_wrapper(self) -> None:
        wrapped = with_stream_timeout(_fast_iterator(), timeout=1.0)
        assert wrapped.__aiter__() is wrapped

    @pytest.mark.asyncio
    async def test_propagates_non_timeout_exceptions(self) -> None:
        async def _failing_iterator() -> AsyncIterator[int]:
            yield 1
            raise ValueError("boom")

        wrapped = with_stream_timeout(_failing_iterator(), timeout=1.0)
        with pytest.raises(ValueError, match="boom"):
            async for _ in wrapped:
                pass


class TestGetStreamIterationTimeout:
    """Tests for the default/env configurable timeout value."""

    def test_default_is_60_seconds(self) -> None:
        assert get_stream_iteration_timeout() == 60.0

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOSONG_STREAM_ITERATION_TIMEOUT", "30.5")
        assert get_stream_iteration_timeout() == 30.5

    def test_invalid_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOSONG_STREAM_ITERATION_TIMEOUT", "not-a-number")
        assert get_stream_iteration_timeout() == 60.0


class TestBaseStreamedMessageTimeout:
    """Tests that BaseStreamedMessage wraps its iterator with a per-chunk timeout."""

    @pytest.mark.asyncio
    async def test_applies_timeout_to_stalled_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOSONG_STREAM_ITERATION_TIMEOUT", "0.05")

        async def _stalled_text_iterator() -> AsyncIterator[StreamedMessagePart]:
            yield TextPart(text="hello")
            await asyncio.sleep(3600)

        msg = _TestStreamedMessage(_stalled_text_iterator())
        parts: list[StreamedMessagePart] = []
        with pytest.raises(APITimeoutError):
            async for part in msg:
                parts.append(part)

        assert parts == [TextPart(text="hello")]

    @pytest.mark.asyncio
    async def test_does_not_timeout_fast_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOSONG_STREAM_ITERATION_TIMEOUT", "10.0")

        msg = _TestStreamedMessage(_text_iterator("a", "b", "c"))
        parts = [part async for part in msg]
        assert parts == [TextPart(text="a"), TextPart(text="b"), TextPart(text="c")]

    @pytest.mark.asyncio
    async def test_aiter_returns_consistent_wrapper(self) -> None:
        msg = _TestStreamedMessage(_text_iterator("a"))
        first = msg.__aiter__()
        second = msg.__aiter__()
        assert first is second


class TestStreamTimeoutIteratorInternals:
    """Direct tests for the internal iterator class."""

    @pytest.mark.asyncio
    async def test_slots_and_attributes(self) -> None:
        it = with_stream_timeout(_fast_iterator(), timeout=1.0)
        assert [item async for item in it] == [0, 1, 2]
