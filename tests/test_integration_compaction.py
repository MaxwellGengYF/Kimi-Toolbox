"""Integration test simulating a multi-turn session with compaction (Phases 1-6)."""

from __future__ import annotations

from kimi_cli.soul.compaction import (
    CompactionOptions,
    CompactMode,
    SimpleCompaction,
    adaptive_preserve_depth,
    should_auto_compact,
)
from kimi_cli.utils.tokens import count_message_tokens
from kimi_cli.wire.types import TextPart
from kosong.message import Message


class TestIntegrationCompaction:
    """Simulate a 20-turn session with forced compactions."""

    def _make_messages(self, n_turns: int):
        msgs = [Message(role="user", content=[TextPart(text="Original task: refactor auth module")])]
        for i in range(1, n_turns):
            if i % 2 == 1:
                msgs.append(Message(role="assistant", content=[TextPart(text=f"Step {i} done")]))
            else:
                msgs.append(Message(role="user", content=[TextPart(text=f"Request {i}")]))
        return msgs

    def test_first_user_message_survives_first_compaction(self):
        """Phase 6: the very first message survives the first compaction round."""
        msgs = self._make_messages(20)
        compactor = SimpleCompaction(max_preserved_messages=2)

        result = compactor.prepare(msgs)
        assert result.compact_message is not None
        # The original first message should be in to_preserve
        first_msg = msgs[0]
        assert first_msg in result.to_preserve

    def test_cascade_prompt_triggers_at_depth_3(self):
        import kimi_cli.prompts as prompts
        # Start with enough compaction summaries so that even after Phase 6 preserves
        # the first one, there are still 3+ in to_compact.
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. A")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. B")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. C")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. D")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. E")]),
        ]
        msgs.extend(self._make_messages(10))
        compactor = SimpleCompaction(max_preserved_messages=2)

        result = compactor.prepare(msgs)
        assert result.cascade_depth >= 3
        last_part = result.compact_message.content[-1]
        assert prompts.COMPACT_CASCADE in last_part.text

    def test_system_prompt_stays_under_budget(self):
        from pathlib import Path
        from types import SimpleNamespace

        from kimix.utils.system_prompt import SystemPromptType, get_system_prompt

        tmp_path = Path("/tmp/fake_work_dir")
        prompt_func = get_system_prompt(
            work_dir=None,
            agent_role=SystemPromptType.Worker,
            max_system_prompt_tokens=4_000,
        )
        runtime = SimpleNamespace(
            builtin_args=SimpleNamespace(
                KIMI_NOW="t",
                KIMI_WORK_DIR=tmp_path,
                KIMI_WORK_DIR_LS="",
                KIMI_AGENTS_MD="",
                KIMI_SKILLS="",
                KIMI_ADDITIONAL_DIRS_INFO="",
                KIMI_OS="Linux",
                KIMI_SHELL="bash",
            ),
            session=SimpleNamespace(dir=tmp_path, id="s", custom_data={}),
        )
        prompt = prompt_func(runtime)
        assert count_message_tokens([Message(role="system", content=[TextPart(text=prompt)])]) <= 4_000

    def test_adaptive_preserve_increases_on_error(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Normal question")]),
            Message(role="assistant", content=[TextPart(text="Normal answer")]),
        ]
        depth_normal = adaptive_preserve_depth(msgs)

        msgs_error = [
            Message(role="user", content=[TextPart(text="Normal question")]),
            Message(role="assistant", content=[TextPart(text="It failed with an error")]),
        ]
        depth_error = adaptive_preserve_depth(msgs_error)

        assert depth_error > depth_normal

    def test_auto_compact_triggered_by_ratio(self):
        assert should_auto_compact(
            850_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_auto_compact_not_triggered_below_threshold(self):
        assert not should_auto_compact(
            100_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_avoid_cascade_uses_compact_prompt_at_depth_3(self):
        """Tool-initiated compaction must always use the structured COMPACT prompt."""
        import kimi_cli.prompts as prompts

        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. A")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. B")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. C")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. D")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. E")]),
        ]
        msgs.extend(self._make_messages(10))
        compactor = SimpleCompaction(max_preserved_messages=2)

        result = compactor.prepare(msgs, options=CompactionOptions(avoid_cascade=True))
        assert result.cascade_depth >= 3
        last_part = result.compact_message.content[-1]
        assert prompts.COMPACT in last_part.text
        assert prompts.COMPACT_CASCADE not in last_part.text

    def test_mode_guidance_appended_to_prompt(self):
        """Each non-balanced mode appends the expected style guidance."""
        msgs = self._make_messages(10)
        compactor = SimpleCompaction(max_preserved_messages=2)

        balanced = compactor.prepare(msgs, options=CompactionOptions(mode=CompactMode.BALANCED))
        assert "Be balanced" in balanced.compact_message.content[-1].text

        aggressive = compactor.prepare(msgs, options=CompactionOptions(mode=CompactMode.AGGRESSIVE))
        assert "Be aggressive" in aggressive.compact_message.content[-1].text

        retentive = compactor.prepare(msgs, options=CompactionOptions(mode=CompactMode.RETENTIVE))
        assert "Be retentive" in retentive.compact_message.content[-1].text

        technical = compactor.prepare(msgs, options=CompactionOptions(mode=CompactMode.TECHNICAL))
        assert "Focus on technical specifics" in technical.compact_message.content[-1].text

    def test_custom_instruction_takes_precedence_after_mode_guidance(self):
        """Mode guidance appears before the user's custom instruction."""
        msgs = self._make_messages(10)
        compactor = SimpleCompaction(max_preserved_messages=2)

        result = compactor.prepare(
            msgs,
            custom_instruction="Keep every file edit.",
            options=CompactionOptions(mode=CompactMode.TECHNICAL),
        )
        text = result.compact_message.content[-1].text
        guidance_pos = text.find("Compaction Style Guidance")
        instruction_pos = text.find("User's Custom Compaction Instruction")
        assert guidance_pos != -1
        assert instruction_pos != -1
        assert guidance_pos < instruction_pos
