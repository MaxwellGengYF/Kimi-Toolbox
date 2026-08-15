from __future__ import annotations

from collections.abc import Sequence

import pytest
from inline_snapshot import snapshot
from kosong.chat_provider import TokenUsage
from kosong.message import AudioURLPart, ImageURLPart, Message, ToolCall, VideoURLPart
from kosong.tooling import Tool

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.soul.compaction import (
    CompactionOptions,
    CompactionResult,
    CompactionShrinkError,
    SimpleCompaction,
    SummarizationInput,
    SurfaceChangedError,
    should_auto_compact,
)
from kimi_cli.soul.compaction import CompactMode, _MODE_GUIDANCE
from kimi_cli.soul.compaction_ledger import CompactionLedger
from kimi_cli.soul.tool_pairing import balanced_cut_indices
from kimi_cli.utils.tokens import count_message_tokens
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
        """200K model: with no output budget, reserved_context_size (50K) is the reservation."""
        # ratio check = 150K >= 170K (False)
        # reserved check = 150K + 50K >= 200K (True)
        assert should_auto_compact(
            150_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )
        # One token below the reserved boundary -> no trigger.
        # reserved check = 149_999 + 50K = 199_999 < 200K (False)
        assert not should_auto_compact(
            149_999, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )
        # ratio check = 170K >= 170K (True)
        assert should_auto_compact(
            170_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
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

        With only the safety margin reserved (smaller than reserved_context_size),
        compaction may be skipped only when pruning brings the input strictly below
        ``max_context_size - reserved_context_size``. Whenever the (post-prune)
        input reaches the boundary — even below the ratio threshold —
        ``should_auto_compact`` must still fire so the context is compacted before
        ``input_token_size >= context_token_size - max_output_token_size`` holds
        (input + output must fit in the window).
        """
        max_context = 200_000
        # Use a high ratio so the reserved boundary is the only trigger.
        trigger_ratio = 0.99
        reserved = 75_000
        safety_margin = 4096
        boundary = max_context - reserved  # 125_000 (reserved dominates the 4096 safety margin)

        # At/over the boundary -> must still fire.
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=trigger_ratio,
            reserved_context_size=reserved,
            safety_margin_tokens=safety_margin,
        )
        assert should_auto_compact(
            boundary + 1,
            max_context,
            trigger_ratio=trigger_ratio,
            reserved_context_size=reserved,
            safety_margin_tokens=safety_margin,
        )

        # Only strictly below the boundary is skipping compaction safe.
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=trigger_ratio,
            reserved_context_size=reserved,
            safety_margin_tokens=safety_margin,
        )

    def test_large_max_tokens_expands_reserved_boundary(self):
        """A large max_tokens budget must reserve enough input headroom.

        Regression: when max_tokens (384k) plus the 4096 token safety margin is
        larger than reserved_context_size (75k), compaction must trigger at
        ``max_context_size - max_tokens - safety_margin``, not at
        ``max_context_size - reserved_context_size``.
        """
        max_context = 1_048_576
        reserved = 75_000
        max_tokens = 384_000
        safety_margin = 4096
        boundary = max_context - max_tokens - safety_margin  # 660_480

        # One token above the boundary -> must trigger.
        assert should_auto_compact(
            boundary + 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            safety_margin_tokens=safety_margin,
        )
        # Exactly at the boundary -> must trigger.
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            safety_margin_tokens=safety_margin,
        )
        # Just below the boundary -> must not trigger.
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            safety_margin_tokens=safety_margin,
        )

    def test_tool_call_buffer_dominates_reserved_boundary(self):
        """A dynamic tool-call output buffer expands the boundary when it is the
        largest single reservation."""
        max_context = 200_000
        reserved = 75_000
        max_tokens = 50_000
        tool_buffer = 100_000
        safety_margin = 4096
        output_size = max_tokens + safety_margin  # 54_096
        boundary = max_context - max(tool_buffer, reserved, output_size)  # 100_000

        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )

    def test_safety_margin_expands_reserved_boundary(self):
        """The 4096 token safety margin expands the boundary when it exceeds a tiny
        reserved_context_size (safety is bundled into the output reservation)."""
        max_context = 200_000
        reserved = 3_000
        safety_margin = 4096
        boundary = max_context - safety_margin  # 195_904

        # Use a high ratio so the reserved boundary is the only trigger.
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.99,
            reserved_context_size=reserved,
            safety_margin_tokens=safety_margin,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.99,
            reserved_context_size=reserved,
            safety_margin_tokens=safety_margin,
        )

    def test_output_budget_capped_at_context_minus_reserved(self):
        """If the full output budget would leave no input room, it is capped."""
        max_context = 100_000
        reserved = 75_000
        # Unbounded output budget would be 384k + 50k + 4096 = 438_096,
        # which is far larger than the context. effective_reserved is capped at
        # max_context - reserved = 25_000, so the trigger boundary is 75_000.
        boundary = 75_000
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=384_000,
            tool_call_buffer_tokens=50_000,
            safety_margin_tokens=4096,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=384_000,
            tool_call_buffer_tokens=50_000,
            safety_margin_tokens=4096,
        )

    def test_small_max_tokens_reserved_context_dominates(self):
        """When max_tokens + safety margin is smaller than reserved_context_size,
        reserved_context_size is the reservation."""
        max_context = 200_000
        reserved = 75_000
        max_tokens = 50_000
        safety_margin = 4096
        boundary = max_context - reserved  # 125_000

        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            safety_margin_tokens=safety_margin,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            safety_margin_tokens=safety_margin,
        )

    def test_none_max_tokens_reserved_context_dominates(self):
        """When max_tokens is None, reserved_context_size is the reservation
        (the 4096 safety margin alone is smaller)."""
        max_context = 200_000
        reserved = 75_000
        safety_margin = 4096
        boundary = max_context - reserved  # 125_000

        # Use a high ratio so the reserved boundary is the only trigger.
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.99,
            reserved_context_size=reserved,
            max_tokens=None,
            safety_margin_tokens=safety_margin,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.99,
            reserved_context_size=reserved,
            max_tokens=None,
            safety_margin_tokens=safety_margin,
        )

    def test_256k_context_with_64k_max_tokens_no_premature_trigger(self):
        """Regression: a 256K context with 64K max_tokens should not compact at ~27% usage.

        The dynamic tool-call output buffer for a 256K context used to be ~166K tokens,
        which pulled the reserved boundary down to ~90K tokens. With a correctly capped
        buffer the reservation is dominated by ``reserved_context_size`` (75K), so the
        trigger stays near ``max_context - reserved_context_size`` (~181K).
        """
        from kimi_cli.soul.toolset import KimiToolset

        max_context = 256_000
        current_tokens = 71_265
        reserved = 75_000
        max_tokens = 64_000
        safety_margin = 1024

        ts = KimiToolset()
        tool_buffer = ts.estimate_tool_output_token_budget(max_context, current_tokens)

        # At the user's observed usage (~27%) compaction must not fire.
        assert not should_auto_compact(
            current_tokens,
            max_context,
            trigger_ratio=0.8,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )

        # Even at 120K tokens it must not fire; the trigger should be far higher.
        assert not should_auto_compact(
            120_000,
            max_context,
            trigger_ratio=0.8,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )

        # The trigger should be at the reserved_context_size boundary.
        boundary = max_context - reserved
        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.8,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )

    def test_large_tool_buffer_does_not_shrink_output_boundary(self):
        """Regression: a large tool buffer must not shrink the boundary below the
        output reservation.

        With a 1M context, 384K max output and the tool buffer at its 1 MiB
        ceiling (262_144 tokens), the boundary stays at
        ``max_context - (max_tokens + safety_margin)`` = 660_480 (~63%), NOT at
        398_336 (~38%) as it would if the reservations were summed.
        """
        max_context = 1_048_576
        reserved = 75_000
        max_tokens = 384_000
        tool_buffer = 262_144
        safety_margin = 4096
        boundary = max_context - max(tool_buffer, reserved, max_tokens + safety_margin)  # 660_480

        assert should_auto_compact(
            boundary,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
        )
        assert not should_auto_compact(
            boundary - 1,
            max_context,
            trigger_ratio=0.85,
            reserved_context_size=reserved,
            max_tokens=max_tokens,
            tool_call_buffer_tokens=tool_buffer,
            safety_margin_tokens=safety_margin,
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


# --- preserve_depth_override (Phase 4 prerequisite) -----------------------


def test_compaction_options_preserve_depth_override_defaults_none():
    assert CompactionOptions().preserve_depth_override is None


def test_prepare_respects_preserve_depth_override():
    """preserve_depth_override bypasses the configured preserve depth entirely."""
    messages = [
        Message(role="user", content=[TextPart(text="Q1")]),
        Message(role="assistant", content=[TextPart(text="A1")]),
        Message(role="user", content=[TextPart(text="Q2")]),
        Message(role="assistant", content=[TextPart(text="A2")]),
        Message(role="user", content=[TextPart(text="Q3")]),
        Message(role="assistant", content=[TextPart(text="A3")]),
    ]
    result = SimpleCompaction(max_preserved_messages=1).prepare(
        messages, options=CompactionOptions(preserve_depth_override=2)
    )

    assert result.compact_message is not None
    texts = " ".join(
        p.text for m in result.to_preserve for p in m.content if isinstance(p, TextPart)
    )
    # override=2 keeps the last 2 user/assistant messages (Q3, A3) plus the
    # Phase-6 first message (Q1); the older pair is compacted
    assert "Q3" in texts and "A3" in texts
    assert "Q2" not in texts and "A2" not in texts
    assert "A1" not in texts


def test_prepare_without_override_uses_configured_depth():
    messages = [
        Message(role="user", content=[TextPart(text="Q1")]),
        Message(role="assistant", content=[TextPart(text="A1")]),
        Message(role="user", content=[TextPart(text="Q2")]),
        Message(role="assistant", content=[TextPart(text="A2")]),
        Message(role="user", content=[TextPart(text="Q3")]),
        Message(role="assistant", content=[TextPart(text="A3")]),
    ]
    result = SimpleCompaction(max_preserved_messages=1).prepare(messages)

    assert result.compact_message is not None
    texts = " ".join(
        p.text for m in result.to_preserve for p in m.content if isinstance(p, TextPart)
    )
    # configured depth=1 keeps only the last 1 user/assistant message (A3) plus
    # the Phase-6 first message (Q1); Q3 is compacted
    assert "A3" in texts
    assert "Q3" not in texts
    assert "Q2" not in texts and "A2" not in texts


# --- balanced tool-pairing cuts in prepare --------------------------------


def _tool_pair_history():
    call1 = ToolCall(id="c1", function=ToolCall.FunctionBody(name="bash", arguments="{}"))
    return [
        Message(role="user", content=[TextPart(text="Q0")]),
        Message(
            role="assistant",
            content=[TextPart(text="Thinking")],
            tool_calls=[call1],
        ),
        Message(role="tool", content=[TextPart(text="R1")], tool_call_id="c1"),
        Message(role="user", content=[TextPart(text="Q1")]),
        Message(role="assistant", content=[TextPart(text="A1")]),
    ]


def test_prepare_balanced_cuts_do_not_split_tool_pairs():
    """With balanced cuts on, the preserved tail never splits a call/result pair."""
    messages = _tool_pair_history()
    result = SimpleCompaction(max_preserved_messages=1).prepare(messages)

    assert result.compact_message is not None
    # the (call c1, R1) pair is compacted together
    compact_text = result.compact_message.extract_text(" ")
    assert "Thinking" in compact_text
    assert "R1" in compact_text
    # preserved tail = [Q0 (Phase 6), A1]
    assert result.to_preserve == [messages[0], messages[4]]
    assert len(result.to_preserve) == 2
    # the final boundary is a balanced cut of the original history
    assert len(messages) - len(result.to_preserve) in balanced_cut_indices(messages)


def test_prepare_balanced_cuts_snaps_mid_pair_boundary():
    """A history ending mid call/result pair: the preserved tail must not split it.

    The raw boundary (index 2) falls between the assistant call and its result;
    balanced cuts snap it left so the whole history is preserved instead.
    """
    call1 = ToolCall(id="c1", function=ToolCall.FunctionBody(name="bash", arguments="{}"))
    messages = [
        Message(role="user", content=[TextPart(text="Q0")]),
        Message(role="assistant", content=[TextPart(text="C")], tool_calls=[call1]),
        Message(role="user", content=[TextPart(text="M")]),
        Message(role="tool", content=[TextPart(text="R1")], tool_call_id="c1"),
        Message(role="user", content=[TextPart(text="Q1")]),
        Message(role="assistant", content=[TextPart(text="A1")]),
    ]
    result = SimpleCompaction(max_preserved_messages=3).prepare(messages)

    assert result.compact_message is None
    assert result.to_preserve == messages


def test_prepare_balanced_cuts_false_keeps_legacy_boundary():
    """Opt-out (balanced_cuts=False) restores the legacy boundary exactly: the
    call/result pair may be split."""
    messages = _tool_pair_history()
    balanced = SimpleCompaction(max_preserved_messages=1).prepare(messages)
    legacy = SimpleCompaction(max_preserved_messages=1, balanced_cuts=False).prepare(messages)

    # no tool messages in this history → identical legacy and balanced output
    assert legacy == balanced


def test_prepare_balanced_cuts_false_legacy_split_mid_pair():
    """On a history where the legacy boundary splits a pair, balanced_cuts=False
    keeps the split exactly as before."""
    call1 = ToolCall(id="c1", function=ToolCall.FunctionBody(name="bash", arguments="{}"))
    messages = [
        Message(role="user", content=[TextPart(text="Q0")]),
        Message(role="assistant", content=[TextPart(text="C")], tool_calls=[call1]),
        Message(role="user", content=[TextPart(text="M")]),
        Message(role="tool", content=[TextPart(text="R1")], tool_call_id="c1"),
        Message(role="user", content=[TextPart(text="Q1")]),
        Message(role="assistant", content=[TextPart(text="A1")]),
    ]
    result = SimpleCompaction(max_preserved_messages=3, balanced_cuts=False).prepare(messages)

    assert result.compact_message is not None
    # legacy: the call is compacted while its result is preserved (pair split)
    assert messages[1] not in result.to_preserve
    assert messages[3] in result.to_preserve


# ---------------------------------------------------------------------------
# Phase 2: KV-cache-aligned summarization input + generate path
# ---------------------------------------------------------------------------


def _aligned_tools() -> list[Tool]:
    return [
        Tool(
            name="bash",
            description="Run a shell command",
            parameters={"type": "object", "properties": {}},
        )
    ]


class StaticStreamedMessage:
    """Minimal kosong ``StreamedMessage`` for fake providers."""

    def __init__(self, parts: Sequence[object], usage: TokenUsage | None = None) -> None:
        self._parts = list(parts)
        self._usage = usage

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._parts:
            raise StopAsyncIteration
        return self._parts.pop(0)

    @property
    def id(self) -> str:
        return "compaction-fake"

    @property
    def usage(self) -> TokenUsage | None:
        return self._usage


class RecordingProvider:
    """Fake chat provider that records ``generate(...)`` arguments and returns
    a static stream. ``on_generate`` (optional) runs inside ``generate`` and can
    mutate state to simulate mid-flight conversation changes."""

    name = "recording-compaction-provider"

    def __init__(
        self,
        parts: Sequence[object],
        usage: TokenUsage | None = None,
        on_generate=None,
    ) -> None:
        self._parts = parts
        self._usage = usage
        self._on_generate = on_generate
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def thinking_effort(self):
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> StaticStreamedMessage:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "tools": list(tools),
                "history": list(history),
            }
        )
        if self._on_generate is not None:
            await self._on_generate(self, system_prompt, tools, history)
        return StaticStreamedMessage(self._parts, usage=self._usage)

    def with_thinking(self, effort):
        return self


def _fake_llm(provider) -> LLM:
    return LLM(chat_provider=provider, max_context_size=200_000, capabilities=set())


def test_prepare_builds_summarization_input_when_aligned_args_passed():
    messages = _compaction_messages()
    tools = _aligned_tools()

    result = SimpleCompaction(max_preserved_messages=2).prepare(
        messages, aligned_system_prompt="SYSTEM PROMPT", aligned_tools=tools
    )

    si = result.summarization_input
    assert si is not None
    assert isinstance(si, SummarizationInput)
    assert si.system_prompt == "SYSTEM PROMPT"
    assert list(si.tools) == tools
    # region is the ORIGINAL to_compact messages (indices 1..4 of the input)
    assert list(si.messages) == messages[1:5]
    # instruction is a user message whose text is byte-identical to the legacy
    # compact_message tail (same prompt-building code path)
    assert si.instruction.role == "user"
    assert si.instruction.content == [result.compact_message.content[-1]]
    assert prompts.COMPACT in si.instruction.extract_text(" ")


def test_prepare_summarization_input_none_without_aligned_args():
    result = SimpleCompaction(max_preserved_messages=2).prepare(_compaction_messages())
    assert result.summarization_input is None


def test_prepare_summarization_input_none_when_region_empty():
    """No aligned input when prepare decides nothing needs compaction."""
    messages = [
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest reply")]),
    ]
    result = SimpleCompaction(max_preserved_messages=2).prepare(
        messages, aligned_system_prompt="SYSTEM", aligned_tools=_aligned_tools()
    )
    assert result.compact_message is None
    assert result.summarization_input is None


async def test_compact_kv_aligned_path_calls_generate_with_region_and_instruction():
    messages = _compaction_messages()
    tools = _aligned_tools()
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)
    compactor = SimpleCompaction(max_preserved_messages=2)
    prepared = compactor.prepare(
        messages, aligned_system_prompt="SYSTEM PROMPT", aligned_tools=tools
    )

    result = await compactor.compact(
        messages,
        llm,
        aligned_system_prompt="SYSTEM PROMPT",
        aligned_tools=tools,
    )

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["system_prompt"] == "SYSTEM PROMPT"
    assert call["tools"] == tools
    assert call["history"] == [
        *prepared.summarization_input.messages,
        prepared.summarization_input.instruction,
    ]
    instruction = call["history"][-1]
    assert instruction.role == "user"
    assert prompts.COMPACT in instruction.extract_text(" ")

    # downstream handling identical to the legacy path
    assert result.messages[0].role == "user"
    assert result.messages[0].extract_text(" ").startswith(
        "<system>Previous context has been compacted. Here is the compaction output:</system>"
    )
    assert result.messages[1:] == list(prepared.to_preserve)
    assert result.usage is None


async def test_compact_legacy_path_uses_flattened_message_and_empty_tools():
    messages = _compaction_messages()
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)

    result = await SimpleCompaction(max_preserved_messages=2).compact(messages, llm)

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["system_prompt"] == (
        "You are a helpful assistant that compacts conversation context."
    )
    assert call["tools"] == []
    assert len(call["history"]) == 1
    assert call["history"][0].role == "user"
    assert prompts.COMPACT in call["history"][0].extract_text(" ")
    # the envelope applies to the legacy path too
    assert len(result.compaction_id) == 32
    assert result.shadowed_tokens == count_message_tokens(messages[1:5])


# ---------------------------------------------------------------------------
# Phase 3: transactional envelope — shrink / stability / ledger
# ---------------------------------------------------------------------------


async def test_compact_shrink_check_raises_when_usage_output_not_smaller():
    messages = _compaction_messages()
    # usage.output (1000) >= shadowed tokens of the compacted region
    usage = TokenUsage(input_other=0, output=1000)
    provider = RecordingProvider(parts=[TextPart(text="tiny")], usage=usage)
    llm = _fake_llm(provider)

    with pytest.raises(CompactionShrinkError, match="not smaller"):
        await SimpleCompaction(max_preserved_messages=2).compact(
            messages, llm, aligned_system_prompt="SYSTEM PROMPT", aligned_tools=[]
        )


async def test_compact_shrink_check_raises_for_long_summary_without_usage():
    messages = _compaction_messages()
    # no usage → summary tokens estimated from text; a 20k-char summary is
    # far larger than the compacted region
    provider = RecordingProvider(parts=[TextPart(text="x" * 20_000)])
    llm = _fake_llm(provider)

    with pytest.raises(CompactionShrinkError, match="not smaller"):
        await SimpleCompaction(max_preserved_messages=2).compact(
            messages, llm, aligned_system_prompt="SYSTEM PROMPT"
        )


async def test_compact_stability_check_raises_when_messages_mutated():
    messages = _compaction_messages()

    async def mutate(_provider, _system_prompt, _tools, _history):
        messages.append(Message(role="user", content=[TextPart(text="intruder")]))

    provider = RecordingProvider(
        parts=[TextPart(text="short summary")], on_generate=mutate
    )
    llm = _fake_llm(provider)

    with pytest.raises(SurfaceChangedError, match="conversation changed"):
        await SimpleCompaction(max_preserved_messages=2).compact(
            messages, llm, aligned_system_prompt="SYSTEM PROMPT", aligned_tools=[]
        )


def test_compaction_result_defaults_for_legacy_construction():
    result = CompactionResult(messages=[], usage=None)
    assert result.compaction_id == ""
    assert result.shadowed_tokens == 0


async def test_compact_result_propagates_compaction_id_and_shadowed_tokens():
    messages = _compaction_messages()
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)

    result = await SimpleCompaction(max_preserved_messages=2).compact(
        messages, llm, aligned_system_prompt="SYSTEM PROMPT", aligned_tools=[]
    )

    assert len(result.compaction_id) == 32
    assert all(c in "0123456789abcdef" for c in result.compaction_id)
    # shadowed region is the original messages compacted (indices 1..4)
    assert result.shadowed_tokens == count_message_tokens(messages[1:5])


async def test_compact_writes_ledger_record_on_success(tmp_path):
    messages = _compaction_messages()
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)

    result = await SimpleCompaction(max_preserved_messages=2).compact(
        messages,
        llm,
        aligned_system_prompt="SYSTEM PROMPT",
        aligned_tools=[],
        ledger=ledger,
    )

    rec = ledger.latest()
    assert rec is not None
    assert rec.compaction_id == result.compaction_id
    assert rec.error is None
    assert rec.shrank is True
    assert rec.shadowed_range == (0, 4)
    assert rec.shadowed_tokens == result.shadowed_tokens
    assert rec.summary_tokens == count_message_tokens(
        [Message(role="user", content=[TextPart(text="short summary")])]
    )


async def test_compact_writes_ledger_error_on_failure(tmp_path):
    messages = _compaction_messages()
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    # huge usage.output → shrink check fails after the transaction started
    usage = TokenUsage(input_other=0, output=999_999)
    provider = RecordingProvider(parts=[TextPart(text="tiny")], usage=usage)
    llm = _fake_llm(provider)

    with pytest.raises(CompactionShrinkError):
        await SimpleCompaction(max_preserved_messages=2).compact(
            messages,
            llm,
            aligned_system_prompt="SYSTEM PROMPT",
            aligned_tools=[],
            ledger=ledger,
        )

    rec = ledger.latest()
    assert rec is not None
    assert rec.error is not None
    assert "not smaller" in rec.error
    assert rec.shrank is False


async def test_compact_legacy_path_with_ledger_records_shadowed_tokens(tmp_path):
    """The legacy path reconstructs to_compact exactly for ledger accounting."""
    messages = _compaction_messages()
    ledger = CompactionLedger(tmp_path / "ledger.jsonl")
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)

    result = await SimpleCompaction(max_preserved_messages=2).compact(
        messages, llm, ledger=ledger
    )

    rec = ledger.latest()
    assert rec is not None
    assert rec.shrank is True
    assert rec.shadowed_tokens == result.shadowed_tokens
    assert rec.shadowed_tokens == count_message_tokens(messages[1:5])


async def test_compact_with_ledger_never_raises_on_broken_ledger_path(tmp_path):
    """Failure isolation: a ledger that cannot be written must not break compact."""
    messages = _compaction_messages()
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("file")
    ledger = CompactionLedger(blocker / "ledger.jsonl")  # parent is a file
    provider = RecordingProvider(parts=[TextPart(text="short summary")])
    llm = _fake_llm(provider)

    result = await SimpleCompaction(max_preserved_messages=2).compact(
        messages,
        llm,
        aligned_system_prompt="SYSTEM PROMPT",
        aligned_tools=[],
        ledger=ledger,
    )

    assert len(result.compaction_id) == 32
    assert result.messages[0].extract_text(" ") == (
        "<system>Previous context has been compacted. Here is the compaction output:</system>"
        " short summary"
    )


async def test_compact_error_ledger_failure_does_not_mask_real_exception(tmp_path):
    """Even when the ledger record_end fails on the error path, the original
    exception still propagates."""
    messages = _compaction_messages()
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("file")
    ledger = CompactionLedger(blocker / "ledger.jsonl")  # broken path
    usage = TokenUsage(input_other=0, output=999_999)
    provider = RecordingProvider(parts=[TextPart(text="tiny")], usage=usage)
    llm = _fake_llm(provider)

    with pytest.raises(CompactionShrinkError):
        await SimpleCompaction(max_preserved_messages=2).compact(
            messages,
            llm,
            aligned_system_prompt="SYSTEM PROMPT",
            aligned_tools=[],
            ledger=ledger,
        )


def test_manual_compaction_error_codes_exist():
    from kimi_cli.soul.compaction import ManualCompactionError

    for code in ("busy", "cancelled", "changed", "summary", "commit", "persistence"):
        exc = ManualCompactionError(code)  # type: ignore[arg-type]
        assert exc.code == code
    assert "summary" in str(ManualCompactionError("summary"))


def test_surface_fingerprint_helper():
    from kimi_cli.soul.compaction import _surface_fingerprint

    messages = [
        Message(role="user", content=[TextPart(text="hello")]),
        Message(role="assistant", content=[TextPart(text="world")]),
    ]
    fp = _surface_fingerprint(messages)
    assert fp.history_len == 2
    assert fp.token_count == count_message_tokens(messages)
    assert fp.last_message_text == "world"
    assert _surface_fingerprint([]).last_message_text is None
