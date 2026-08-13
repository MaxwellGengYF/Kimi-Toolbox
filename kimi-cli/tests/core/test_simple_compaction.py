from __future__ import annotations

from inline_snapshot import snapshot
from kosong.chat_provider import TokenUsage
from kosong.message import AudioURLPart, ImageURLPart, Message, VideoURLPart

import kimi_cli.prompts as prompts
from kimi_cli.soul.compaction import (
    CompactionResult,
    SimpleCompaction,
    should_auto_compact,
)
from kimi_cli.soul.compaction import CompactMode, _MODE_GUIDANCE
from kimi_cli.wire.types import TextPart, ThinkPart


def test_prepare_returns_original_when_not_enough_messages():
    messages = [Message(role="user", content=[TextPart(text="Only one message")])]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result == snapshot(
        SimpleCompaction.PrepareResult(
            compact_message=None,
            to_preserve=[Message(role="user", content=[TextPart(text="Only one message")])],
        )
    )


def test_prepare_skips_compaction_with_only_preserved_messages():
    messages = [
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest reply")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result == snapshot(
        SimpleCompaction.PrepareResult(
            compact_message=None,
            to_preserve=[
                Message(role="user", content=[TextPart(text="Latest question")]),
                Message(role="assistant", content=[TextPart(text="Latest reply")]),
            ],
        )
    )


def test_prepare_builds_compact_message_and_preserves_tail():
    messages = [
        Message(role="system", content=[TextPart(text="System note")]),
        Message(
            role="user",
            content=[TextPart(text="Old question"), ThinkPart(think="Hidden thoughts")],
        ),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    # Phase 6: first message (system) is always preserved
    assert result.compact_message == snapshot(
        Message(
            role="user",
            content=[
                TextPart(text="## Message 1\nRole: user\nContent:\n"),
                TextPart(text="Old question"),
                TextPart(text="## Message 2\nRole: assistant\nContent:\n"),
                TextPart(text="Old answer"),
                TextPart(text="\n" + prompts.COMPACT + "\n\n" + _MODE_GUIDANCE[CompactMode.BALANCED]),
            ],
        )
    )
    assert result.to_preserve == snapshot(
        [
            Message(role="system", content=[TextPart(text="System note")]),
            Message(role="user", content=[TextPart(text="Latest question")]),
            Message(role="assistant", content=[TextPart(text="Latest answer")]),
        ]
    )


# --- CompactionResult.estimated_token_count tests ---


def test_estimated_token_count_with_usage_uses_output_tokens_for_summary():
    """When usage is available, the summary (first message) uses exact output tokens
    and preserved messages (remaining) use character-based estimation."""
    summary_msg = Message(role="user", content=[TextPart(text="compacted summary")])
    preserved_msg = Message(
        role="user",
        content=[TextPart(text="a" * 80)],  # 80 chars → 20 tokens
    )
    usage = TokenUsage(input_other=1000, output=150, input_cache_read=0)

    result = CompactionResult(messages=[summary_msg, preserved_msg], usage=usage)

    assert result.estimated_token_count == 150 + 20


def test_estimated_token_count_without_usage_estimates_all_from_text():
    """Without usage (no LLM call), all messages are estimated from text content."""
    messages = [
        Message(role="user", content=[TextPart(text="a" * 100)]),
        Message(role="assistant", content=[TextPart(text="b" * 200)]),
    ]
    result = CompactionResult(messages=messages, usage=None)

    assert result.estimated_token_count == 300 // 4


def test_estimated_token_count_ignores_non_text_parts():
    """Non-text parts (think, etc.) should not inflate the estimate."""
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="a" * 40),
                ThinkPart(think="internal reasoning " * 100),
            ],
        ),
    ]
    result = CompactionResult(messages=messages, usage=None)

    assert result.estimated_token_count == 40 // 4


def test_estimated_token_count_empty_messages():
    """Empty message list should return 0."""
    result = CompactionResult(messages=[], usage=None)
    assert result.estimated_token_count == 0


def test_prepare_appends_custom_instruction():
    messages = [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(
        messages, custom_instruction="Preserve all discussions about the database"
    )

    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    # Custom instruction should be merged into the same TextPart as the COMPACT prompt
    assert last_part.text.startswith("\n" + prompts.COMPACT)
    assert "User's Custom Compaction Instruction" in last_part.text
    assert "Preserve all discussions about the database" in last_part.text


def test_prepare_without_custom_instruction_unchanged():
    """When no custom_instruction is given, the compact message should end with the COMPACT prompt."""
    messages = [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    assert last_part.text == "\n" + prompts.COMPACT + "\n\n" + _MODE_GUIDANCE[CompactMode.BALANCED]


# --- should_auto_compact tests ---


class TestShouldAutoCompact:
    """Test the auto-compaction trigger logic across different model context sizes."""

    def test_200k_model_triggers_by_reserved(self):
        """200K model with default config: reserved (50K) fires first at 150K (75%)."""
        # At 150K tokens: ratio check = 150K >= 170K (False), reserved check = 200K >= 200K (True)
        assert should_auto_compact(
            150_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_200k_model_below_threshold(self):
        """200K model: 140K tokens should NOT trigger (below both thresholds)."""
        assert not should_auto_compact(
            140_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_1m_model_triggers_by_ratio(self):
        """1M model with default config: ratio (85%) fires first at 850K."""
        # At 850K tokens: ratio check = 850K >= 850K (True)
        assert should_auto_compact(
            850_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_1m_model_below_ratio_threshold(self):
        """1M model: 840K tokens should NOT trigger (below 85% ratio, well above reserved)."""
        assert not should_auto_compact(
            840_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_custom_ratio_triggers_earlier(self):
        """Custom ratio=0.7 triggers at 70% of context."""
        # 200K * 0.7 = 140K
        assert should_auto_compact(
            140_000, 200_000, trigger_ratio=0.7, reserved_context_size=50_000
        )
        assert not should_auto_compact(
            139_999, 200_000, trigger_ratio=0.7, reserved_context_size=50_000
        )

    def test_zero_tokens_never_triggers(self):
        """Empty context should never trigger compaction."""
        assert not should_auto_compact(0, 200_000, trigger_ratio=0.85, reserved_context_size=50_000)

    def test_compaction_never_skippable_above_reserved_boundary(self):
        """The pruning-skip decision must keep the reserved-output boundary.

        Compaction may be skipped only when pruning brings the input strictly
        below ``max_context_size - reserved_context_size``. Whenever the
        (post-prune) input reaches the boundary — even below the ratio
        threshold — ``should_auto_compact`` must still fire so the context is
        compacted before ``input_token_size >= context_token_size -
        max_output_token_size`` holds (input + output must fit in the window).
        """
        max_context = 200_000
        trigger_ratio = 0.8
        reserved = 75_000
        boundary = max_context - reserved  # 125_000

        # At/over the boundary (125K) but below the ratio threshold (160K)
        # -> must still fire, otherwise compaction would be skipped too late.
        assert should_auto_compact(
            boundary, max_context, trigger_ratio=trigger_ratio, reserved_context_size=reserved
        )
        assert should_auto_compact(
            boundary + 1, max_context, trigger_ratio=trigger_ratio, reserved_context_size=reserved
        )
        assert should_auto_compact(
            150_000, max_context, trigger_ratio=trigger_ratio, reserved_context_size=reserved
        )

        # Only strictly below the boundary is skipping compaction safe.
        assert not should_auto_compact(
            boundary - 1, max_context, trigger_ratio=trigger_ratio, reserved_context_size=reserved
        )


def test_prepare_only_keeps_text_parts_in_compaction():
    """Compaction input should only contain TextPart (whitelist approach).

    Non-text parts (media, think, etc.) are filtered out because the compaction
    API endpoint only supports text content.

    Fixes: https://github.com/MoonshotAI/kimi-cli/issues/1395
    Fixes: https://github.com/MoonshotAI/kimi-cli/issues/1390
    """
    # Phase 6: prepend a system message so the media-rich user msg stays in to_compact
    messages = [
        Message(role="system", content=[TextPart(text="System prompt")]),
        Message(
            role="user",
            content=[
                TextPart(text="Analyze these files:"),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,IMG")),
                AudioURLPart(audio_url=AudioURLPart.AudioURL(url="data:audio/mp3;base64,AUD")),
                VideoURLPart(video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,VID")),
                ThinkPart(think="internal reasoning"),
            ],
        ),
        Message(role="assistant", content=[TextPart(text="I can see all the media files.")]),
        Message(role="user", content=[TextPart(text="What's your conclusion?")]),
    ]

    result = SimpleCompaction(max_preserved_messages=1).prepare(messages)

    assert result.compact_message is not None
    # Verify only TextPart remains in the compaction request
    for part in result.compact_message.content:
        assert isinstance(part, TextPart), (
            f"Only TextPart should be in compaction input, got {type(part).__name__}"
        )

    # Text content should be preserved
    texts = [p.text for p in result.compact_message.content if isinstance(p, TextPart)]
    assert any("Analyze these files:" in t for t in texts)
    assert any("I can see all the media files." in t for t in texts)


def test_prepare_preserves_media_parts_in_recent_messages():
    """Media parts in preserved (recent) messages should remain untouched."""
    messages = [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(
            role="user",
            content=[
                TextPart(text="Look at this video:"),
                VideoURLPart(video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,VID")),
            ],
        ),
        Message(role="assistant", content=[TextPart(text="Nice video!")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    # Phase 6: first message is prepended, so video is in to_preserve[1]
    # Find the preserved message that contains the video part
    video_msg = None
    for msg in result.to_preserve:
        if any(isinstance(p, VideoURLPart) for p in msg.content):
            video_msg = msg
            break
    assert video_msg is not None, "Expected a preserved message with VideoURLPart"


def test_prepare_selects_cascade_prompt_at_depth_3():
    """When 3+ messages in to_compact are already compaction summaries, prepare()
    should select the COMPACT_CASCADE prompt and report cascade_depth >= 3."""
    from kimi_cli.soul.message import system

    summary_prefix = "Previous context has been compacted. Here is the compaction output:"
    messages = [
        Message(role="user", content=[TextPart(text="Original request")]),
        Message(role="user", content=[system(summary_prefix), TextPart(text="summary 1")]),
        Message(role="user", content=[system(summary_prefix), TextPart(text="summary 2")]),
        Message(role="user", content=[system(summary_prefix), TextPart(text="summary 3")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    # Original request is preserved by Phase 6; Latest Q+A are preserved by tail.
    # to_compact should contain the 3 summary messages.
    assert result.cascade_depth >= 3
    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    assert prompts.COMPACT_CASCADE in last_part.text


def test_prepare_selects_normal_prompt_below_depth_3():
    """When fewer than 3 messages are compaction summaries, prepare() should select
    the normal COMPACT prompt."""
    from kimi_cli.soul.message import system

    summary_prefix = "Previous context has been compacted. Here is the compaction output:"
    messages = [
        Message(role="user", content=[system(summary_prefix), TextPart(text="summary 1")]),
        Message(role="assistant", content=[TextPart(text="ok")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.cascade_depth < 3
    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    assert prompts.COMPACT in last_part.text
    assert prompts.COMPACT_CASCADE not in last_part.text


# --- Merged from tests/test_first_turn_preservation.py ---

class TestFirstTurnPreservation:
    """Test that the first message is always preserved."""

    def test_first_message_preserved_when_not_in_tail(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Original request: build a web app")]),
            Message(role="assistant", content=[TextPart(text="Okay, let me start.")]),
            Message(role="user", content=[TextPart(text="Use React")]),
            Message(role="assistant", content=[TextPart(text="Sure.")]),
            Message(role="user", content=[TextPart(text="Add routing")]),
            Message(role="assistant", content=[TextPart(text="Done.")]),
        ]
        result = SimpleCompaction(max_preserved_messages=2).prepare(msgs)
        assert result.to_preserve[0] == msgs[0]
        if result.compact_message is not None:
            texts = [p.text for p in result.compact_message.content if isinstance(p, TextPart)]
            assert "Original request" not in " ".join(texts)

    def test_no_duplicate_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="First")]),
            Message(role="assistant", content=[TextPart(text="Second")]),
            Message(role="user", content=[TextPart(text="Third")]),
        ]
        result = SimpleCompaction(max_preserved_messages=3).prepare(msgs)
        assert result.compact_message is None
        ids = [id(m) for m in result.to_preserve]
        assert len(ids) == len(set(ids))

    def test_first_message_already_in_tail_no_duplicate(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Only one")]),
            Message(role="assistant", content=[TextPart(text="Reply")]),
        ]
        result = SimpleCompaction(max_preserved_messages=2).prepare(msgs)
        assert result.compact_message is None
        assert len(result.to_preserve) == 2

    def test_system_first_message_preserved(self):
        msgs = [
            Message(role="system", content=[TextPart(text="System prompt")]),
            Message(role="user", content=[TextPart(text="User asks something")]),
            Message(role="assistant", content=[TextPart(text="Assistant replies")]),
            Message(role="user", content=[TextPart(text="Follow up")]),
        ]
        result = SimpleCompaction(max_preserved_messages=1).prepare(msgs)
        assert result.to_preserve[0] == msgs[0]


# --- Merged from tests/test_compaction_cascade.py ---

class TestDetectCascadeDepth:
    """Test cascade depth detection."""

    def test_no_compaction_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Hello")]),
            Message(role="assistant", content=[TextPart(text="Hi")]),
        ]
        from kimi_cli.soul.compaction import _detect_cascade_depth

        assert _detect_cascade_depth(msgs) == 0

    def test_one_compaction_message(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. Summary.")]),
            Message(role="user", content=[TextPart(text="New question")]),
        ]
        from kimi_cli.soul.compaction import _detect_cascade_depth

        assert _detect_cascade_depth(msgs) == 1

    def test_three_compaction_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. A")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. B")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. C")]),
        ]
        from kimi_cli.soul.compaction import _detect_cascade_depth

        assert _detect_cascade_depth(msgs) == 3


# ---------------------------------------------------------------------------
# P4: decision & verification summary sections
# ---------------------------------------------------------------------------


def _compaction_messages() -> list[Message]:
    return [
        Message(role="system", content=[TextPart(text="System note")]),
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Older question 2")]),
        Message(role="assistant", content=[TextPart(text="Older answer 2")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]


def test_prepare_includes_decision_sections_when_enabled():
    result = SimpleCompaction(
        max_preserved_messages=2, decision_section_enabled=True
    ).prepare(_compaction_messages())

    assert result.compact_message is not None
    prompt_text = result.compact_message.extract_text(" ")
    assert "## Decisions & Conclusions" in prompt_text
    assert "## Verification Status" in prompt_text


def test_prepare_omits_decision_sections_by_default():
    result = SimpleCompaction(max_preserved_messages=2).prepare(_compaction_messages())

    assert result.compact_message is not None
    prompt_text = result.compact_message.extract_text(" ")
    assert "## Decisions & Conclusions" not in prompt_text
    assert "## Verification Status" not in prompt_text
