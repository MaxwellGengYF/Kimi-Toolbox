from __future__ import annotations

import pytest
from kosong.message import Message, TextPart, ToolCall, ToolCallPart

from kimi_cli.soul.compaction import SimpleCompaction
from kimi_cli.soul.tool_pairing import (
    balanced_cut_indices,
    is_tool_call_part,
    message_tool_call_delta,
    nearest_balanced_cut_before,
)


def _msg(role: str, text: str = "", *, tool_calls=None, tool_call_id: str | None = None):
    content = [TextPart(text=text)] if text else []
    return Message(
        role=role,  # type: ignore[arg-type]
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def _call(call_id: str = "c1", name: str = "bash") -> ToolCall:
    return ToolCall(id=call_id, function=ToolCall.FunctionBody(name=name, arguments="{}"))


def _assistant_with_content_tool_call_part() -> Message:
    """A message carrying a streamed ToolCallPart inside content.

    Normal pydantic construction rejects ToolCallPart in ``content`` (it is
    not a ContentPart), so build via model_construct to model the "other
    flows" the tool-pairing detector must tolerate.
    """
    return Message.model_construct(role="assistant", content=[ToolCallPart(arguments_part="{}")])


# --- is_tool_call_part ---------------------------------------------------


def test_is_tool_call_part():
    assert is_tool_call_part(_call())
    assert is_tool_call_part(ToolCallPart(arguments_part="{}"))
    assert not is_tool_call_part(TextPart(text="hi"))
    assert not is_tool_call_part(object())
    assert not is_tool_call_part(None)


# --- message_tool_call_delta ---------------------------------------------


def test_delta_assistant_with_two_tool_calls():
    msg = _msg("assistant", tool_calls=[_call("c1"), _call("c2")])
    assert message_tool_call_delta(msg) == 2


def test_delta_assistant_with_tool_call_part_in_content():
    assert message_tool_call_delta(_assistant_with_content_tool_call_part()) == 1


def test_delta_assistant_with_both_tool_calls_and_content_parts():
    msg = _msg("assistant", tool_calls=[_call("c1")])
    with_content = Message.model_construct(
        role="assistant",
        content=[ToolCallPart(arguments_part="{}")],
        tool_calls=[_call("c2")],
    )
    assert message_tool_call_delta(msg) == 1
    assert message_tool_call_delta(with_content) == 2


def test_delta_tool_result():
    assert message_tool_call_delta(_msg("tool", "ok", tool_call_id="c1")) == -1


def test_delta_plain_messages_are_zero():
    assert message_tool_call_delta(_msg("user", "hello")) == 0
    assert message_tool_call_delta(_msg("system", "sys")) == 0
    assert message_tool_call_delta(_msg("assistant", "plain reply")) == 0


# --- balanced_cut_indices ------------------------------------------------


def test_balanced_cut_indices_empty():
    assert balanced_cut_indices([]) == {0}


def test_balanced_cut_indices_plain_chat_all_balanced():
    messages = [
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
        _msg("user", "Q2"),
        _msg("assistant", "A2"),
    ]
    assert balanced_cut_indices(messages) == {0, 1, 2, 3, 4}


def test_balanced_cut_indices_mixed_history():
    """user → call → tool → user → call → tool → user → assistant(no call)."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", tool_calls=[_call("c2")]),
        _msg("tool", "R2", tool_call_id="c2"),
        _msg("user", "Q2"),
        _msg("assistant", "A2"),
    ]
    assert balanced_cut_indices(messages) == {0, 1, 3, 4, 6, 7, 8}


def test_balanced_cut_indices_trailing_unbalanced_is_lenient():
    """A trailing assistant tool call with no result keeps cuts balanced only
    where in_progress returns to 0; len is still included."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", tool_calls=[_call("c2")]),  # no result follows
    ]
    assert balanced_cut_indices(messages) == {0, 1, 3, 4, 5}


def test_balanced_cut_indices_raises_on_orphan_tool_result():
    with pytest.raises(ValueError, match="unbalanced tool history"):
        balanced_cut_indices([_msg("tool", "orphan", tool_call_id="c1")])


def test_balanced_cut_indices_raises_when_in_progress_goes_negative():
    messages = [
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("tool", "orphan", tool_call_id="c2"),
    ]
    with pytest.raises(ValueError, match="unbalanced tool history"):
        balanced_cut_indices(messages)


# --- nearest_balanced_cut_before -----------------------------------------


def _pair_history():
    # user → call → tool → user → assistant
    return [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]


def test_nearest_balanced_cut_before_mid_pair():
    # cut 2 would split (call c1, R1); largest balanced cut <= 2 is 1
    messages = _pair_history()
    assert balanced_cut_indices(messages) == {0, 1, 3, 4, 5}
    assert nearest_balanced_cut_before(messages, 2) == 1


def test_nearest_balanced_cut_before_at_len():
    messages = _pair_history()
    assert nearest_balanced_cut_before(messages, len(messages)) == len(messages)


def test_nearest_balanced_cut_before_at_zero():
    messages = _pair_history()
    assert nearest_balanced_cut_before(messages, 0) == 0


def test_nearest_balanced_cut_before_clamps():
    messages = _pair_history()
    assert nearest_balanced_cut_before(messages, -3) == 0
    assert nearest_balanced_cut_before(messages, len(messages) + 10) == len(messages)


def test_nearest_balanced_cut_before_exact_balanced_index():
    messages = _pair_history()
    assert nearest_balanced_cut_before(messages, 3) == 3
    assert nearest_balanced_cut_before(messages, 4) == 4


# --- prepare boundary guarantees -----------------------------------------


def test_prepare_never_splits_call_result_pair():
    """A raw boundary inside a call/result pair is snapped left so the
    preserved tail keeps the pair together."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", "Thinking", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    result = SimpleCompaction(max_preserved_messages=1).prepare(messages)

    assert result.compact_message is not None
    # pair (call c1, R1) is intact inside to_compact (both compacted together)
    compact_text = result.compact_message.extract_text(" ")
    assert "Thinking" in compact_text
    assert "R1" in compact_text
    # preserved tail is exactly [Q0 (Phase 6), A1] — no tool message leads it
    assert result.to_preserve == [messages[0], messages[4]]
    # boundary is a balanced cut
    assert len(messages) - len(result.to_preserve) in balanced_cut_indices(messages)


def test_prepare_balanced_cuts_snaps_boundary_left_of_mid_pair():
    """When the raw boundary would split (call, interleaved user, result),
    balanced cuts snap left — the whole history is preserved instead of
    compacting mid-pair."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("user", "M"),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    result = SimpleCompaction(max_preserved_messages=3).prepare(messages)

    assert result.compact_message is None
    assert result.to_preserve == messages


def test_prepare_balanced_cuts_false_keeps_legacy_split():
    """With the opt-out flag, the legacy boundary (raw index 2) splits the
    call/result pair exactly as before."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("user", "M"),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    result = SimpleCompaction(max_preserved_messages=3, balanced_cuts=False).prepare(messages)

    assert result.compact_message is not None
    # legacy: the call is compacted while its result is preserved (pair split)
    assert messages[1] not in result.to_preserve
    assert messages[3] in result.to_preserve


def test_prepare_first_message_reinsertion_keeps_boundary_balanced():
    """Phase-6 first-message re-insertion must not push the boundary off a
    balanced cut: when it would, the first message stays in to_compact."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    # raw boundary at index 3 (balanced); Phase 6 re-insertion of Q0 would
    # shift len(to_compact) to 2 which is NOT a balanced cut → fallback keeps
    # the first message inside to_compact and rebuilds at cut 1.
    assert result.compact_message is not None
    assert result.to_preserve == [messages[1], messages[2], messages[3], messages[4]]
    assert messages[0] not in result.to_preserve
    # the boundary is a balanced cut of the original history
    assert len(messages) - len(result.to_preserve) in balanced_cut_indices(messages)


def test_prepare_balanced_cuts_no_split_across_many_depths():
    """Property check: for every preserve depth, the final boundary is a
    balanced cut and the preserved tail never starts with a tool message."""
    messages = [
        _msg("user", "Q0"),
        _msg("assistant", tool_calls=[_call("c1")]),
        _msg("tool", "R1", tool_call_id="c1"),
        _msg("user", "Q1"),
        _msg("assistant", tool_calls=[_call("c2")]),
        _msg("tool", "R2", tool_call_id="c2"),
        _msg("user", "Q2"),
        _msg("assistant", "A2"),
    ]
    for depth in range(1, 6):
        result = SimpleCompaction(max_preserved_messages=depth).prepare(messages)
        if result.to_preserve:
            assert result.to_preserve[0].role != "tool", f"depth={depth}"
        boundary = len(messages) - len(result.to_preserve)
        assert boundary in balanced_cut_indices(messages), f"depth={depth}"
        # no tool message in to_preserve whose call is inside to_compact:
        compacted_ids = {
            tc.id
            for m in messages[:boundary]
            for tc in (m.tool_calls or [])
        }
        for m in result.to_preserve:
            if m.role == "tool" and m.tool_call_id in compacted_ids:
                pytest.fail(f"depth={depth}: tool result {m.tool_call_id} split from its call")


def test_prepare_balanced_cuts_raises_on_corrupt_history():
    messages = [_msg("tool", "orphan", tool_call_id="c1"), _msg("user", "Q1")]
    with pytest.raises(ValueError, match="unbalanced tool history"):
        SimpleCompaction(max_preserved_messages=1).prepare(messages)


def test_prepare_balanced_cuts_false_tolerates_corrupt_history():
    messages = [_msg("tool", "orphan", tool_call_id="c1"), _msg("user", "Q1")]
    result = SimpleCompaction(max_preserved_messages=1, balanced_cuts=False).prepare(messages)
    assert result.compact_message is None
    assert result.to_preserve == messages
