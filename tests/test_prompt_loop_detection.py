from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
import importlib

import pytest
from kosong.message import TextPart

from kimi_cli.llm import LoopDetectedError

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float = 0.0
    context_tokens: int = 0


class FakeSession:
    def __init__(
        self,
        parts: list[Any] | None = None,
        raises_at: int | None = None,
    ) -> None:
        self.status = FakeStatus()
        self._cancel_event = None
        self.cancelled = False
        self.parts = parts or []
        self.raises_at = raises_at

    async def prompt(self, prompt_str: str, *, merge_wire_messages: bool = False) -> Any:
        if False:
            yield None
        for i, part in enumerate(self.parts):
            if i == self.raises_at:
                raise LoopDetectedError("loop detected")
            yield part

    def cancel(self) -> None:
        self.cancelled = True


def _suppress_output(monkeypatch: Any) -> list[str]:
    printed: list[str] = []

    def _capture(text: str, *args: Any, **kwargs: Any) -> None:
        printed.append(text)

    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", _capture)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *args, **kwargs: None)
    return printed


@pytest.mark.asyncio
async def test_run_prompt_iter_cancels_on_word_loop(monkeypatch: Any) -> None:
    """A high-level word loop should cancel the session and print a warning."""
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    monkeypatch.setenv("KIMIX_LOOP_WORD_THRESHOLD", "5")
    printed = _suppress_output(monkeypatch)
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    parts = [TextPart(text="loop ")] * 60
    session = FakeSession(parts=parts)

    await prompt_mod._run_single_prompt(
        session, "hi", output_function=None, cancel_callable=None,
        merge_wire_messages=False, info_print=False,
    )

    assert session.cancelled
    assert any("Loop detected" in text for text in printed)
    # Loop detection should not be retried as a transient failure.
    assert sleeps == []


@pytest.mark.asyncio
async def test_run_prompt_iter_cancels_on_low_level_loop_detected_error(monkeypatch: Any) -> None:
    """A LoopDetectedError raised inside the stream should cancel gracefully."""
    monkeypatch.setenv("KIMIX_LOOP_DETECTION_ENABLED", "1")
    printed = _suppress_output(monkeypatch)
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    parts = [TextPart(text="hello ")] * 10
    session = FakeSession(parts=parts, raises_at=5)

    await prompt_mod._run_single_prompt(
        session, "hi", output_function=None, cancel_callable=None,
        merge_wire_messages=False, info_print=False,
    )

    assert session.cancelled
    assert any("Loop detected" in text for text in printed)
    assert sleeps == []
