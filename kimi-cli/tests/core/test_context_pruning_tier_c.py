"""Tests for Tier C — micro-compress in place (plan.md §8.3, Phase 4).

Covers the ``_tier_c_candidates`` detector, the ``_apply_tier_c`` pre-pass,
and the ``ContextPruner.prune()`` integration: in-place compression of stale
tool messages, ElidedRecord emission (annotated stages only), idempotency,
the combined min-payoff gate, Layer 1 ``<system>`` metadata coalescing, and
the protected-set / Tier-B-exclusion guards.
"""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.context_pruning import (
    ContextPruner,
    _apply_tier_c,
    _looks_like_readfile_output,
    _tier_c_candidates,
    is_pruned_stub,
)
from kimi_cli.wire.types import ImageURLPart, TextPart, ThinkPart

# A grep-style common path prefix, long enough that Stage 4 folding fires.
_PREFIX = "D:\\kimi-agent\\src\\kimix\\tools\\file\\"


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _tool(text: str, tool_call_id: str = "call_1", system_meta: str | None = None) -> Message:
    parts = []
    if system_meta:
        parts.append(TextPart(text=f"<system>{system_meta}</system>"))
    parts.append(TextPart(text=text))
    return Message(role="tool", content=parts, tool_call_id=tool_call_id)


def _grep_style_text(n: int = 30) -> str:
    """n grep-result lines sharing a long absolute-path prefix."""
    return "\n".join(f"{_PREFIX}mod{i}.py:10:match line {i}" for i in range(n))


def _default_pruner(**kw) -> ContextPruner:
    defaults = dict(
        trigger_ratio=0.0,
        target_ratio=0.0,
        stable_prefix_messages=0,
        recent_messages_protected=0,
        min_free_tokens=0,
        cooldown_steps=0,
        # Tier C is opt-in by default (config default is False); these tests
        # exercise the feature, so enable it explicitly.
        micro_compress_enabled=True,
    )
    defaults.update(kw)
    return ContextPruner(**defaults)


# ======================================================================
# ReadFile-style detection
# ======================================================================


class TestLookLikeReadFile:
    def test_line_numbered_detected(self):
        assert _looks_like_readfile_output("     1\tfoo\n    10\tbar") is True

    def test_plain_text_not_detected(self):
        assert _looks_like_readfile_output("foo\nbar") is False

    def test_partially_numbered_not_detected(self):
        assert _looks_like_readfile_output("1\tfoo\nnot numbered") is False

    def test_empty_not_detected(self):
        assert _looks_like_readfile_output("") is False


# ======================================================================
# _tier_c_candidates
# ======================================================================


class TestTierCCandidates:
    def test_finds_compressible_tool_message(self):
        history = [_user("hello"), _tool(_grep_style_text())]
        cands = _tier_c_candidates(history, excluded=set(), min_saved_chars=32)
        assert len(cands) == 1
        idx, savings, kind = cands[0]
        assert idx == 1
        assert savings > 0
        assert kind == "log"

    def test_skips_excluded_indices(self):
        history = [_user("hello"), _tool(_grep_style_text())]
        cands = _tier_c_candidates(history, excluded={1}, min_saved_chars=32)
        assert cands == []

    def test_skips_non_tool_messages(self):
        history = [_user(_grep_style_text())]
        cands = _tier_c_candidates(history, excluded=set(), min_saved_chars=32)
        assert cands == []

    def test_skips_small_messages_below_threshold(self):
        history = [_tool("tiny output")]
        cands = _tier_c_candidates(history, excluded=set(), min_saved_chars=32)
        assert cands == []

    def test_skips_system_only_messages(self):
        msg = Message(role="tool", content=[TextPart(text="<system>Tool output is empty.</system>")])
        cands = _tier_c_candidates([msg], excluded=set(), min_saved_chars=32)
        assert cands == []

    def test_already_compressed_is_noop(self):
        """Idempotency: a second pass over compressed text yields no candidates."""
        history = [_user("hello"), _tool(_grep_style_text())]
        cands1 = _tier_c_candidates(history, excluded=set(), min_saved_chars=32)
        work, _, _, changed, _ = _apply_tier_c(
            history, set(), min_saved_chars=32, ref_counter=0
        )
        assert changed == {1}
        cands2 = _tier_c_candidates(work, excluded=set(), min_saved_chars=32)
        assert cands2 == []

    def test_line_numbered_content_uses_code_kind(self):
        text = "\n".join(f"{i + 1}\t    x = {i}   # c" for i in range(30))
        cands = _tier_c_candidates([_tool(text)], excluded=set(), min_saved_chars=32)
        # code-kind compress is a no-op for this input → no candidates
        assert cands == []


# ======================================================================
# _apply_tier_c
# ======================================================================


class TestApplyTierC:
    def test_compresses_in_place_and_preserves_metadata(self):
        history = [
            _user("hello"),
            _tool(_grep_style_text(), tool_call_id="call_x", system_meta="Results truncated."),
        ]
        work, records, freed, changed, next_ref = _apply_tier_c(
            history, set(), min_saved_chars=32, ref_counter=7
        )
        assert changed == {1}
        assert freed > 0
        assert next_ref == 8
        new_msg = work[1]
        # system metadata part preserved, output part compressed
        assert new_msg.content[0].text == "<system>Results truncated.</system>"
        assert "[prefix:" in new_msg.content[1].text
        assert new_msg.tool_call_id == "call_x"
        # annotated Stage 4 fired → ElidedRecord with the original archived
        assert len(records) == 1
        rec = records[0]
        assert rec.kind == "micro_compress"
        assert rec.ref == "prune_7"
        assert rec.original_text == _grep_style_text()
        # caller history is never mutated
        assert history[1].content[-1].text == _grep_style_text()

    def test_lossless_only_change_emits_no_record(self):
        text = "line one   \nline two\t\n" * 10
        work, records, freed, changed, _ = _apply_tier_c(
            [_tool(text)], set(), min_saved_chars=32, ref_counter=0
        )
        assert changed == {0}
        assert freed > 0
        assert records == []  # lossless stages only → silent
        assert work[0].content[0].text == "line one\nline two\n" * 10

    def test_preserves_non_text_parts(self):
        text = _grep_style_text()
        msg = Message(
            role="tool",
            content=[
                TextPart(text=text),
                ThinkPart(think="reasoning"),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url="file:///x.png")),
            ],
            tool_call_id="call_1",
        )
        work, _, _, changed, _ = _apply_tier_c([msg], set(), min_saved_chars=32)
        assert changed == {0}
        parts = work[0].content
        kinds = [type(p).__name__ for p in parts]
        assert kinds == ["TextPart", "ThinkPart", "ImageURLPart"]
        assert parts[1].think == "reasoning"

    def test_no_candidates_returns_original_list(self):
        history = [_user("hello"), _tool("tiny")]
        work, records, freed, changed, next_ref = _apply_tier_c(
            history, set(), min_saved_chars=32, ref_counter=3
        )
        assert work == list(history)
        assert records == [] and freed == 0 and changed == set() and next_ref == 3


# ======================================================================
# prune() integration
# ======================================================================


class TestPruneTierC:
    def test_tier_c_applied_in_place(self):
        pruner = _default_pruner()
        history = [_user("hello"), _tool(_grep_style_text())]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        assert result.earliest_removed_index == 1
        assert result.freed_tokens > 0
        text = "".join(p.text for p in result.messages[1].content if isinstance(p, TextPart))
        assert "[prefix:" in text
        assert len(result.messages) == 2  # nothing dropped, compressed in place
        assert result.elided and result.elided[0].kind == "micro_compress"
        # caller history untouched
        assert history[1].content[0].text == _grep_style_text()

    def test_second_pass_is_noop(self):
        pruner = _default_pruner()
        history = [_user("hello"), _tool(_grep_style_text())]
        result1 = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        assert result1.earliest_removed_index is not None
        pruner.reset_cooldown()
        result2 = pruner.prune(result1.messages, context_usage=0.8, max_context_size=100_000)
        assert result2.earliest_removed_index is None
        assert result2.freed_tokens == 0

    def test_disabled_micro_compress_is_passthrough(self):
        pruner = _default_pruner(micro_compress_enabled=False)
        history = [_user("hello"), _tool(_grep_style_text())]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        assert result.earliest_removed_index is None
        assert result.messages == list(history)

    def test_protected_tail_untouched(self):
        pruner = _default_pruner(recent_messages_protected=1)
        history = [
            _user("old"),
            _tool(_grep_style_text(), tool_call_id="call_old"),
            _user("recent"),
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        # Only index 1 is outside the protected tail (index 2 protected)
        assert result.earliest_removed_index == 1
        text = "".join(p.text for p in result.messages[1].content if isinstance(p, TextPart))
        assert "[prefix:" in text
        assert result.messages[2].content[0].text == "recent"

    def test_combined_min_payoff_gate(self):
        """Tier C savings count toward the min-payoff gate."""
        pruner = _default_pruner(min_free_tokens=10_000)
        history = [_user("hello"), _tool(_grep_style_text())]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        # ~2.4k chars saved < 10_000-token gate → whole pass rolled back
        assert result.earliest_removed_index is None
        assert result.messages == list(history)

    def test_tier_b_candidates_excluded_from_tier_c(self):
        """Messages Tier B will stub are not Tier C compressed first, so the
        archived original stays uncompressed."""
        pruner = _default_pruner(
            ephemeral_enabled=False,
            tool_output_min_tokens=50,
        )
        big = _grep_style_text(n=60) + "\n" + "padding " * 400  # > 50 tokens
        history = [_user("hello"), _tool(big)]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        assert result.earliest_removed_index == 1
        assert is_pruned_stub(result.messages[1])
        tier_b_recs = [r for r in result.elided if r.kind != "micro_compress"]
        assert tier_b_recs, "Tier B elision should have produced a record"
        assert "[prefix:" not in tier_b_recs[0].original_text  # true original archived

    def test_layer1_metadata_coalescing(self):
        """Adjacent identical <system> metadata is merged during the prune pass."""
        pruner = _default_pruner()
        history = [
            _user("hello"),
            _tool("x" * 200, tool_call_id="call_a", system_meta="Results truncated to 20 lines."),
            _tool("y" * 200, tool_call_id="call_b", system_meta="Results truncated to 20 lines."),
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert len(tool_msgs) == 2
        # Second tool message lost its redundant system part
        assert not any(
            isinstance(p, TextPart) and p.text.strip().startswith("<system>")
            for p in tool_msgs[1].content
        )
        assert any(
            isinstance(p, TextPart) and "[×2]" in p.text for p in tool_msgs[0].content
        )

    def test_coalescing_never_empties_system_only_message(self):
        pruner = _default_pruner()
        history = [
            _user("hello"),
            Message(role="tool", content=[TextPart(text="<system>Results truncated to 20 lines.</system>")], tool_call_id="call_a"),
            Message(role="tool", content=[TextPart(text="<system>Results truncated to 20 lines.</system>")], tool_call_id="call_b"),
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        for m in tool_msgs:
            assert len(m.content) >= 1  # never empty
            assert m.tool_call_id is not None

    def test_python_path_merges_tier_c(self):
        """Tier C runs on the pure-Python path (native SOUL kernel removed)."""
        pruner = _default_pruner()
        history = [_user("hello"), _tool(_grep_style_text())]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        assert result.earliest_removed_index == 1
        text = "".join(p.text for p in result.messages[1].content if isinstance(p, TextPart))
        assert "[prefix:" in text

    def test_prune_with_policy_propagates_flag(self):
        pruner = _default_pruner()
        history = [_user("hello"), _tool(_grep_style_text())]
        result = pruner.prune_with_policy(history, target_token_count=1)
        assert result.earliest_removed_index is not None
        text = "".join(p.text for p in result.messages[1].content if isinstance(p, TextPart))
        assert "[prefix:" in text

    def test_freed_tokens_reported_for_lossless_only(self):
        pruner = _default_pruner()
        text = "line one   \nline two\t\n" * 20  # ~80 chars of trailing-ws noise
        history = [_user("hello"), _tool(text)]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100_000)
        # lossless stages alone are enough to fire Tier C (savings ≥ 64 chars)
        assert result.earliest_removed_index == 1
        assert result.freed_tokens > 0
        # no annotated stage fired → no ElidedRecord
        assert all(r.kind != "micro_compress" for r in result.elided)
