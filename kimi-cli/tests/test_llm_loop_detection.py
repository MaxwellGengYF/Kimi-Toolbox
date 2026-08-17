from __future__ import annotations

import pytest
from kosong.chat_provider.mock import MockChatProvider
from kosong.message import TextPart, ThinkPart

from kimi_cli.llm import (
    LoopDetectedError,
    TextLoopDetector,
    _wrap_generate_with_loop_detection,
)


def test_char_loop_detected() -> None:
    detector = TextLoopDetector(char_threshold=50)
    assert detector.feed("a" * 50)


def test_char_loop_not_triggered_by_short_run() -> None:
    detector = TextLoopDetector(char_threshold=50)
    assert not detector.feed("a" * 49)


def test_char_loop_not_triggered_by_whitespace() -> None:
    detector = TextLoopDetector(char_threshold=50)
    assert not detector.feed(" " * 50)


def test_word_loop_detected() -> None:
    detector = TextLoopDetector(word_threshold=5, word_window=10)
    assert detector.feed("foo " * 5)


def test_word_loop_not_triggered_below_threshold() -> None:
    detector = TextLoopDetector(word_threshold=5, word_window=10)
    assert not detector.feed("foo " * 4)


def test_word_loop_across_chunks() -> None:
    detector = TextLoopDetector(word_threshold=5, word_window=10)
    for _ in range(4):
        assert not detector.feed("foo ")
    assert not detector.feed("fo")
    assert detector.feed("o ")


def test_word_loop_resets_across_whitespace_boundaries() -> None:
    detector = TextLoopDetector(word_threshold=3, word_window=10)
    assert not detector.feed("foo foo ")
    assert detector.feed("foo ")


def test_no_loop_on_normal_text() -> None:
    detector = TextLoopDetector()
    text = "The quick brown fox jumps over the lazy dog. "
    assert not detector.feed(text * 100)


def test_empty_feed_does_not_trigger() -> None:
    detector = TextLoopDetector()
    assert not detector.feed("")
    assert not detector.feed("foo")


@pytest.mark.asyncio
async def test_wrapper_raises_loop_detected_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    monkeypatch.setenv("KIMIX_LOOP_WORD_THRESHOLD", "5")
    parts = [TextPart(text="loop ")] * 60
    provider = MockChatProvider(parts)
    _wrap_generate_with_loop_detection(provider)
    stream = await provider.generate("system", [], [])
    with pytest.raises(LoopDetectedError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_wrapper_raises_loop_detected_error_think(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    monkeypatch.setenv("KIMIX_LOOP_WORD_THRESHOLD", "5")
    parts = [ThinkPart(think="loop ")] * 60
    provider = MockChatProvider(parts)
    _wrap_generate_with_loop_detection(provider)
    stream = await provider.generate("system", [], [])
    with pytest.raises(LoopDetectedError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_wrapper_ignores_encrypted_think(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    parts = [ThinkPart(think="loop ", encrypted="sig")] * 60
    provider = MockChatProvider(parts)
    _wrap_generate_with_loop_detection(provider)
    stream = await provider.generate("system", [], [])
    count = 0
    async for _ in stream:
        count += 1
    assert count == 60


@pytest.mark.asyncio
async def test_wrapper_no_loop_on_varied_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    parts = [TextPart(text="The quick brown fox jumps over the lazy dog. ")] * 5
    provider = MockChatProvider(parts)
    _wrap_generate_with_loop_detection(provider)
    stream = await provider.generate("system", [], [])
    count = 0
    async for _ in stream:
        count += 1
    assert count == 5


@pytest.mark.asyncio
async def test_wrapper_preserves_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    parts = [TextPart(text="hello ")]
    provider = MockChatProvider(parts)
    _wrap_generate_with_loop_detection(provider)
    stream = await provider.generate("system", [], [])
    assert stream.id == "mock"
