"""Tests for the current-turn start index helpers (cache-02 call-site wiring).

``KimiSoul._current_turn_start_index`` (used by ``_step``) and the
``context_prune`` tool helper must both find the current turn's first *real*
user message, skipping injected ``<system-reminder>`` user messages.
"""

from __future__ import annotations

from kosong.message import Message, TextPart

from kimi_cli.soul.kimisoul import _current_turn_start_index as soul_index
from kimi_cli.soul.message import system_reminder
from kimi_cli.tools.context_prune import _current_turn_start_index as tool_index


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _reminder(text: str = "r") -> Message:
    return Message(role="user", content=[system_reminder(text)])


def test_returns_last_real_user_index() -> None:
    history = [
        _user("u0"),
        _reminder("old"),
        _user("u1"),  # current turn starts here
        _reminder("fresh"),
    ]
    assert soul_index(history) == 2
    assert tool_index(history) == 2


def test_reminder_only_tail() -> None:
    history = [_user("u0"), _reminder("r1"), _reminder("r2")]
    assert soul_index(history) == 0
    assert tool_index(history) == 0


def test_no_real_user_returns_none() -> None:
    assert soul_index([]) is None
    assert soul_index([_reminder("only")]) is None
    assert tool_index([]) is None
    assert tool_index([_reminder("only")]) is None


def test_non_user_tail_ignored() -> None:
    history = [
        _user("u0"),
        Message(role="assistant", content=[TextPart(text="a")]),
        Message(role="tool", content=[TextPart(text="t")], tool_call_id="c1"),
    ]
    assert soul_index(history) == 0
    assert tool_index(history) == 0
