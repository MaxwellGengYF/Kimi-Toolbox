"""Tests for normalize_history in the dynamic_injection module."""

from __future__ import annotations

from kosong.message import ContentPart, Message, TextPart

from kimi_cli.soul.dynamic_injection import normalize_history
from kimi_cli.soul.message import system_reminder


def _text(part: ContentPart) -> str:
    assert isinstance(part, TextPart)
    return part.text


def test_empty_history() -> None:
    assert normalize_history([]) == []


def test_single_user_message() -> None:
    msgs = [Message(role="user", content=[TextPart(text="hello")])]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert result[0].role == "user"
    assert _text(result[0].content[0]) == "hello"


def test_single_assistant_message() -> None:
    msgs = [Message(role="assistant", content=[TextPart(text="hi")])]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert result[0].role == "assistant"


def test_adjacent_user_messages_merged() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="user", content=[TextPart(text="B")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert result[0].role == "user"
    assert len(result[0].content) == 2
    assert _text(result[0].content[0]) == "A"
    assert _text(result[0].content[1]) == "B"


def test_three_adjacent_user_messages_merged() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="user", content=[TextPart(text="B")]),
        Message(role="user", content=[TextPart(text="C")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert len(result[0].content) == 3


def test_non_adjacent_users_not_merged() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="assistant", content=[TextPart(text="X")]),
        Message(role="user", content=[TextPart(text="B")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 3
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[2].role == "user"


def test_adjacent_assistant_not_merged() -> None:
    msgs = [
        Message(role="assistant", content=[TextPart(text="X")]),
        Message(role="assistant", content=[TextPart(text="Y")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 2


def test_mixed_roles_complex() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="user", content=[TextPart(text="B")]),
        Message(role="assistant", content=[TextPart(text="X")]),
        Message(role="user", content=[TextPart(text="C")]),
        Message(role="user", content=[TextPart(text="D")]),
        Message(role="assistant", content=[TextPart(text="Y")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 4
    assert result[0].role == "user"
    assert len(result[0].content) == 2  # A + B merged
    assert result[1].role == "assistant"
    assert result[2].role == "user"
    assert len(result[2].content) == 2  # C + D merged
    assert result[3].role == "assistant"


def test_multipart_content_preserved() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="A"), TextPart(text="B")]),
        Message(role="user", content=[TextPart(text="C")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert len(result[0].content) == 3
    assert _text(result[0].content[0]) == "A"
    assert _text(result[0].content[1]) == "B"
    assert _text(result[0].content[2]) == "C"


def test_notification_messages_not_merged_with_user_messages() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="user input")]),
        Message(
            role="user",
            content=[
                TextPart(
                    text='<notification id="n1" category="task" type="task.completed">x</notification>'
                )
            ],
        ),
    ]
    result = normalize_history(msgs)
    assert len(result) == 2


# ======================================================================
# cache-01: <system-reminder> messages must never be merged (KV-cache
# boundary churn — see plans/01-reminder-merge-boundary-churn-cache-miss.md)
# ======================================================================


def test_system_reminder_not_merged_with_user_message() -> None:
    msgs = [
        Message(role="user", content=[TextPart(text="u1")]),
        Message(role="user", content=[system_reminder("inj1")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 2
    # The real user message must be byte-identical to normalizing it alone.
    without = normalize_history(msgs[:1])
    assert result[0].role == "user"
    assert result[0].content == without[0].content
    # The reminder stays a standalone tail message.
    assert _text(result[1].content[0]).startswith("<system-reminder>")


def test_system_reminder_prefix_stable_across_steps() -> None:
    """Plan cache-01 reproduction: [u0, a0, t0, u1, inj1] keeps u1 and inj1
    separate, and the first 4 messages equal normalize([u0, a0, t0, u1])."""
    msgs = [
        Message(role="user", content=[TextPart(text="u0")]),
        Message(role="assistant", content=[TextPart(text="a0")]),
        Message(role="tool", content=[TextPart(text="t0")], tool_call_id="c1"),
        Message(role="user", content=[TextPart(text="u1")]),
        Message(role="user", content=[system_reminder("inj1")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 5  # u1 and inj1 stay separate messages

    base = normalize_history(msgs[:4])
    assert len(base) == 4
    for mine, ref in zip(result[:4], base):
        assert mine.role == ref.role
        assert mine.content == ref.content
    assert _text(result[4].content[0]).startswith("<system-reminder>")


def test_reminder_between_user_messages_blocks_merge() -> None:
    """A reminder between two real user messages prevents their merge."""
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="user", content=[system_reminder("r")]),
        Message(role="user", content=[TextPart(text="B")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 3
    assert _text(result[1].content[0]).startswith("<system-reminder>")


def test_adjacent_user_messages_still_merge_without_reminder() -> None:
    """Regression guard: merging still applies to plain adjacent user messages."""
    msgs = [
        Message(role="user", content=[TextPart(text="A")]),
        Message(role="user", content=[TextPart(text="B")]),
    ]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert len(result[0].content) == 2
