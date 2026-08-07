"""Tests for the /btw side-question context construction (cache-01).

`_build_btw_context` must strip stale `<system-reminder>` user messages before
normalizing, so the side-question request shares the exact prefix of the next
main-loop step (system prompt + tools + normalized history without stale
reminders) — maximizing provider prompt-cache hits.
"""

from __future__ import annotations

from types import SimpleNamespace

from kosong.message import Message, TextPart

from kimi_cli.soul.btw import _build_btw_context
from kimi_cli.soul.dynamic_injection import normalize_history
from kimi_cli.soul.message import strip_system_reminders, system_reminder


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _tool(text: str) -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id="c1")


def _reminder(text: str) -> Message:
    return Message(role="user", content=[system_reminder(text)])


class _FakeAgent:
    def __init__(self, tools: list) -> None:
        self.toolset = SimpleNamespace(tools=tools)

    def get_system_prompt(self) -> str:
        return "sys-prompt"


class _FakeSoul:
    def __init__(self, history: list[Message], tools: list | None = None) -> None:
        self.context = SimpleNamespace(history=history)
        self._agent = _FakeAgent(tools or [])


def _text_of(msg: Message) -> str:
    return "".join(
        part.text for part in msg.content if isinstance(part, TextPart)
    )


def test_btw_context_strips_stale_reminders() -> None:
    """The normalized prefix must contain no stale system-reminder text."""
    history = [
        _user("u0"),
        _assistant("a0"),
        _tool("t0"),
        _user("u1"),
        _reminder("stale reminder"),
    ]
    soul = _FakeSoul(history)

    system_prompt, out_history, toolset = _build_btw_context(soul, "question?")

    assert system_prompt == "sys-prompt"
    assert toolset.tools == []
    # Side question is the last message; its prefix is the stripped+normalized history.
    assert len(out_history) == len(history)  # 4 real + 1 side message

    prefix = out_history[:-1]
    joined = "\n".join(_text_of(m) for m in prefix)
    assert "<system-reminder>" not in joined
    assert _text_of(out_history[-1]).endswith("question?")


def test_btw_context_prefix_equals_normalized_stripped_history() -> None:
    """The side-question history equals [*normalize(stripped), side_message]."""
    history = [
        _user("u0"),
        _assistant("a0"),
        _tool("t0"),
        _user("u1"),
        _reminder("stale"),
    ]
    soul = _FakeSoul(history)

    _, out_history, _ = _build_btw_context(soul, "side?")

    stripped = list(history)
    strip_system_reminders(stripped)
    expected_prefix = normalize_history(stripped)
    assert len(expected_prefix) == 4
    for mine, ref in zip(out_history[:-1], expected_prefix):
        assert mine.role == ref.role
        assert mine.content == ref.content
