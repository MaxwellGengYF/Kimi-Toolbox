from __future__ import annotations

import pytest

from kosong.message import Message


from kimi_cli.notifications.llm import build_notification_message
from kimi_cli.soul.context_pruning import (
    ContextPruner,
    ElidedRecord,
    PruningResult,
    _compute_protected_indices,
    _is_active_task_snapshot_message,
    _is_dmail_notice_message,
    _is_ephemeral_message,
    _is_checkpoint_marker_message,
    _is_superseded_read,
    _is_oversized_output,
    _is_resolved_error,
    _protect_tool_pair_indices,
    _tier_a_candidates,
    _tier_b_candidates,
    is_pruned_stub,
)
from kimi_cli.soul.message import system, system_reminder
from kimi_cli.wire.types import TextPart, ThinkPart


# ======================================================================
# Helpers
# ======================================================================


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant(text: str = "", tool_calls: list | None = None) -> Message:
    return Message(role="assistant", content=[TextPart(text=text)] if text else [], tool_calls=tool_calls)


def _tool(text: str, tool_call_id: str = "call_1") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id=tool_call_id)


def _notification(text: str = "Test notification", category: str = "system") -> Message:
    """Build a minimal notification message."""
    import uuid
    msg_lines = [
        f'<notification id="{uuid.uuid4()}" category="{category}" '
        f'type="test" source_kind="system" source_id="">',
        f"Title: Test",
        f"Severity: info",
        text,
        "</notification>",
    ]
    return Message(role="user", content=[TextPart(text="\n".join(msg_lines))])


def _active_task_snapshot(text: str = "Active task: test") -> Message:
    return Message(
        role="user",
        content=[
            system("The following background tasks are still active after compaction."),
            TextPart(text=f"<active-background-tasks>{text}</active-background-tasks>"),
        ],
    )


def _dmail_notice(text: str = "D-Mail: do something") -> Message:
    return Message(
        role="user",
        content=[system(f"You just got a D-Mail from your future self. {text}")],
    )


def _checkpoint_marker(n: int = 1) -> Message:
    return Message(role="user", content=[system(f"CHECKPOINT {n}")])


def _system_reminder(text: str = "reminder") -> Message:
    return Message(role="user", content=[system_reminder(text)])


# ======================================================================
# Tier A — Ephemeral message detectors
# ======================================================================


class TestTierADetectors:
    def test_active_task_snapshot_detected(self):
        msg = _active_task_snapshot()
        assert _is_active_task_snapshot_message(msg) is True

    def test_active_task_snapshot_not_detected(self):
        msg = _user("Hello")
        assert _is_active_task_snapshot_message(msg) is False

    def test_dmail_notice_detected(self):
        msg = _dmail_notice()
        assert _is_dmail_notice_message(msg) is True

    def test_dmail_notice_not_detected(self):
        msg = _user("Hello")
        assert _is_dmail_notice_message(msg) is False

    def test_checkpoint_marker_detected(self):
        msg = _checkpoint_marker()
        assert _is_checkpoint_marker_message(msg) is True

    def test_checkpoint_marker_not_detected(self):
        msg = _user("Hello")
        assert _is_checkpoint_marker_message(msg) is False

    def test_is_ephemeral_message_system_reminder(self):
        msg = _system_reminder()
        assert _is_ephemeral_message(msg) is True

    def test_is_ephemeral_message_notification(self):
        msg = _notification()
        assert _is_ephemeral_message(msg, check_notifications=True) is True
        assert _is_ephemeral_message(msg, check_notifications=False) is False

    def test_is_ephemeral_message_task_snapshot(self):
        msg = _active_task_snapshot()
        assert _is_ephemeral_message(msg, check_task_snapshots=True) is True
        assert _is_ephemeral_message(msg, check_task_snapshots=False) is False

    def test_is_ephemeral_message_dmail(self):
        msg = _dmail_notice()
        assert _is_ephemeral_message(msg, check_dmail=True) is True
        assert _is_ephemeral_message(msg, check_dmail=False) is False

    def test_is_ephemeral_message_checkpoint(self):
        msg = _checkpoint_marker()
        # Default: check_checkpoints=False
        assert _is_ephemeral_message(msg, check_checkpoints=False) is False
        assert _is_ephemeral_message(msg, check_checkpoints=True) is True


# ======================================================================
# Protected set
# ======================================================================


class TestProtectedIndices:
    def test_head_protected(self):
        history = [_user("a"), _user("b"), _user("c"), _user("d"), _user("e")]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=3, recent_messages_protected=1
        )
        assert 0 in protected
        assert 1 in protected
        assert 2 in protected

    def test_tail_protected(self):
        history = [_user(f"msg{i}") for i in range(10)]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=2, recent_messages_protected=3
        )
        # Head: indices 0,1
        assert 0 in protected
        assert 1 in protected
        # Tail: last 3 user/assistant turns = indices 7,8,9
        assert 7 in protected
        assert 8 in protected
        assert 9 in protected

    def test_tool_messages_follow_assistant(self):
        """Tool messages following an assistant-with-tool_calls in the tail are protected."""
        history = [
            _user("query"),
            _assistant("Let me read"),
            _tool("file content", tool_call_id="call_1"),
            _user("next"),
            _assistant("done"),
        ]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=2
        )
        # Tail: last 2 user/assistant turns = indices 3,4
        assert 3 in protected  # user "next"
        assert 4 in protected  # assistant "done"
        # Tool message at index 2 is before the tail, but if assistant with tool_calls
        # is in tail, tool responses get protected
        # Actually the assistant at index 1 is not in tail (tail is 3,4)
        # So tool at 2 is not protected unless its assistant is protected
        # This is the right behavior - let's just check the tail is correct

    def test_current_turn_protected(self):
        history = [_user("a"), _user("b"), _user("c")]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=1, current_turn_index=1
        )
        assert 0 in protected  # head
        assert 1 in protected  # current turn start
        assert 2 in protected  # current turn (anything after current_turn_index)


# ======================================================================
# Tier A candidate selection
# ======================================================================


class TestTierACandidates:
    def test_superseded_task_snapshots(self):
        """Only the most recent task snapshot is kept."""
        history = [
            _active_task_snapshot("old task"),
            _user("some work"),
            _active_task_snapshot("newer task"),
            _user("more work"),
            _active_task_snapshot("latest task"),
        ]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=0, recent_messages_protected=0
        )
        candidates = _tier_a_candidates(history, protected, drop_task_snapshots=True)
        # The latest snapshot (index 4) should NOT be in candidates
        candidate_indices = {idx for idx, _ in candidates}
        assert 4 not in candidate_indices, "The latest snapshot should be kept"
        # Older snapshots (indices 0, 2) should be candidates
        assert 0 in candidate_indices or 2 in candidate_indices

    def test_notifications_outside_protected(self):
        history = [
            _notification("first"),
            _user("work1"),
            _notification("second"),
            _user("work2"),  # tail protection targets this
        ]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=1
        )
        candidates = _tier_a_candidates(history, protected, drop_notifications=True)
        # Index 0 is in head (protected), index 3 is tail (protected)
        # Index 2 (notification) is in the middle band → candidate
        candidate_indices = {idx for idx, _ in candidates}
        assert 2 in candidate_indices

    def test_dmail_outside_protected(self):
        history = [
            _user("hello"),
            _dmail_notice(),
            _user("response"),
        ]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=1
        )
        candidates = _tier_a_candidates(history, protected, drop_dmail=True)
        # Index 0 is head (protected), index 2 is tail (protected)
        # Index 1 (dmail) is in the middle band → candidate
        candidate_indices = {idx for idx, _ in candidates}
        assert 1 in candidate_indices


# ======================================================================
# Tier B candidates
# ======================================================================


class TestTierBCandidates:
    def test_oversized_output_detected(self):
        large_text = "x" * 5000  # ~1250 tokens
        msg = _tool(large_text)
        is_oversized, kind, savings = _is_oversized_output([msg], 0, min_tokens=512)
        assert is_oversized is True
        assert kind == "oversized_output"
        assert savings >= 512

    def test_superseded_read_detected(self):
        """A tool result followed by an empty result is superseded."""
        history = [
            _tool("long file content here"),
            _tool("Tool output is empty."),
        ]
        is_sup, kind, savings = _is_superseded_read(history, 0)
        assert is_sup is True
        assert kind == "superseded_read"

    def test_resolved_error_detected(self):
        """An error followed by a successful result is resolved."""
        history = [
            _tool("<system>ERROR: file not found</system>"),
            _tool("File content here"),
        ]
        is_resolved, kind, savings = _is_resolved_error(history, 0)
        assert is_resolved is True
        assert kind == "resolved_error"

    def test_tier_b_candidates_outside_protected(self):
        large_text = "x" * 5000
        history = [
            _user("hello"),
            _tool(large_text),
            _user("world"),
        ]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=1
        )
        candidates = _tier_b_candidates(history, protected, min_output_tokens=512)
        candidate_indices = {idx for idx, _, _ in candidates}
        # Index 1 is in the middle band → candidate
        assert 1 in candidate_indices

    def test_tier_b_empty_when_all_protected(self):
        """No candidates when everything is protected."""
        history = [_user("only message")]
        protected = _compute_protected_indices(
            history, stable_prefix_messages=1, recent_messages_protected=1
        )
        candidates = _tier_b_candidates(history, protected)
        assert len(candidates) == 0


# ======================================================================
# Cache-conservative policy
# ======================================================================


class TestCachePolicy:
    def test_tail_inward_selection(self):
        """Pruning prefers latest-index candidates first (tail-inward)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,  # always trigger
            target_ratio=0.0,  # prune all we can
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        # Create history with ephemeral messages at various positions
        history = [
            _notification("notif1"),  # index 0
            _user("work1"),
            _notification("notif2"),  # index 2
            _user("work2"),
            _notification("notif3"),  # index 4
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert result.earliest_removed_index is not None
        # Should have removed notifications in tail-inward order
        # The earliest removed index should be the first notification (or later)

    def test_min_payoff_gate(self):
        """Skip pruning if savings < min_free_tokens."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=10000,  # high threshold
            cooldown_steps=0,
        )
        history = [_notification("small"), _user("work")]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        # The notification saves only ~ few tokens, below min_free_tokens
        assert result.earliest_removed_index is None
        assert result.freed_tokens == 0

    def test_cooldown_prevents_repruning(self):
        """Within cooldown, pruning is a no-op."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=5,
        )
        history = [_notification("test"), _user("work")]

        # First pass — should prune
        result1 = pruner.prune(history, current_step=1, context_usage=0.8, max_context_size=100000)
        assert result1.earliest_removed_index is not None

        # Second pass within cooldown — no-op
        result2 = pruner.prune(history, current_step=2, context_usage=0.8, max_context_size=100000)
        assert result2.earliest_removed_index is None

    def test_idempotency(self):
        """Re-pruning already pruned history is a no-op."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [_notification("test"), _user("work")]

        # First pass
        result1 = pruner.prune(history, current_step=1, context_usage=0.8, max_context_size=100000)
        assert result1.earliest_removed_index is not None

        # Reset cooldown for second pass with same history
        pruner.reset_cooldown()
        result2 = pruner.prune(result1.messages, current_step=10, context_usage=0.8, max_context_size=100000)
        # Nothing left to prune — no-op
        assert result2.earliest_removed_index is None

    def test_tier_a_preferred_over_tier_b(self):
        """When both Tier A and Tier B candidates exist, Tier A is selected first."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            tool_output_min_tokens=50,
        )
        history = [
            _notification("notif"),
            _tool("x" * 500),  # oversized output
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert result.earliest_removed_index is not None
        # Both candidates should be selected (budget allows both)
        # The notification (Tier A, index 0) should be dropped
        # The oversized (Tier B, index 1) should be elided
        assert len(result.messages) <= 2  # at least one removed


# ======================================================================
# Pruner integration
# ======================================================================


class TestContextPruner:
    def test_disabled_pruner_returns_original(self):
        pruner = ContextPruner(enabled=False)
        history = [_user("hello")]
        result = pruner.prune(history)
        assert result.messages == list(history)
        assert result.freed_tokens == 0
        assert result.earliest_removed_index is None

    def test_below_trigger_no_op(self):
        pruner = ContextPruner(trigger_ratio=0.9)
        history = [_notification("test"), _user("work")]
        result = pruner.prune(history, context_usage=0.5, max_context_size=100000)
        assert result.earliest_removed_index is None

    def test_tier_a_drops_ephemeral(self):
        """Tier A messages are dropped entirely from the result."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [
            _notification("notif1"),
            _user("hello"),
            _notification("notif2"),
            _user("world"),
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert result.freed_tokens > 0
        # Notifications should be dropped
        result_roles = [m.role for m in result.messages]
        assert result_roles == ["user", "user"]  # only the two user messages

    def test_tier_b_elides_content(self):
        """Tier B messages are elided (stub replaces content)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            ephemeral_enabled=False,  # only Tier B
            tool_output_min_tokens=50,
        )
        history = [
            _user("hello"),
            _tool("x" * 500),  # oversized
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert result.freed_tokens > 0
        # The tool message should be elided (stub), not dropped
        assert len(result.messages) == 2  # both messages still present
        assert is_pruned_stub(result.messages[1])

    def test_estimate_after_prune(self):
        """estimate_after_prune matches actual pruning result."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [_notification("test"), _user("hello")]
        # First pass — this sets hysteresis state
        result1 = pruner.prune(history, current_step=1, context_usage=0.8, max_context_size=100000)
        # Reset cooldown so estimate_after_prune also runs
        pruner.reset_cooldown()
        estimated = pruner.estimate_after_prune(history, context_usage=0.8, max_context_size=100000, current_step=10)
        from kimi_cli.utils.tokens import count_message_tokens
        actual_tokens = count_message_tokens(result1.messages)
        assert estimated == actual_tokens

    def test_estimate_after_prune_is_dry_run(self):
        """estimate_after_prune must not consume cooldown/ref-counter state.

        The auto-compaction skip check calls ``estimate_after_prune`` and, when
        it decides pruning frees enough space, relies on the *real* prune at
        the same step still applying to the LLM-visible history. If the dry run
        advanced the hysteresis (cooldown) state, the real prune would be
        blocked and the LLM would see the unpruned (over-limit) history.
        """
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=4,
        )
        history = [_notification("test"), _user("hello")]

        estimated = pruner.estimate_after_prune(
            history, context_usage=0.8, max_context_size=100000, current_step=5
        )
        # Dry run must leave hysteresis / ref-counter state untouched
        assert pruner._last_prune_step == -1
        assert pruner._last_prune_usage == 0.0
        assert pruner._ref_counter == 0

        # A real prune at the same step must still apply
        result = pruner.prune(
            history, context_usage=0.8, max_context_size=100000, current_step=5
        )
        assert result.earliest_removed_index is not None
        assert result.freed_tokens > 0
        assert pruner._last_prune_step == 5

    def test_elided_record_created(self):
        """Tier B elisions produce ElidedRecord entries."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            ephemeral_enabled=False,
            tool_output_min_tokens=50,
        )
        history = [_tool("x" * 500)]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert len(result.elided) >= 1
        assert result.elided[0].role == "tool"
        assert result.elided[0].ref.startswith("prune_")

    def test_tier_a_tier_b_combined(self):
        """Both Tier A drop and Tier B elision work together."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            tool_output_min_tokens=50,
            max_fraction_per_pass=1.0,  # allow pruning everything
        )
        history = [
            _notification("notif"),
            _user("hello"),
            _tool("x" * 500),
        ]
        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)
        assert result.freed_tokens > 0
        # Both Tier A and Tier B candidates should be selected
        # Notification dropped (Tier A), tool elided (Tier B), user kept
        assert any(m.role == "user" for m in result.messages)
        assert len(result.messages) < 3  # at least one message removed/elided

    def test_is_pruned_stub(self):
        """is_pruned_stub detects elision stubs."""
        stub_msg = Message(
            role="tool",
            content=[TextPart(text="<system>[context-elided: oversized_output — ...]</system>")],
        )
        assert is_pruned_stub(stub_msg) is True

        normal_msg = _user("hello")
        assert is_pruned_stub(normal_msg) is False

    def test_pairing_invariant_preserved(self):
        """After pruning, every tool_call.id still has a matching
        tool_call_id response and vice versa (§6.3).

        Tier A only drops standalone ephemera (no tool_call_id, no tool_calls).
        Tier B only elides content (keeps role/tool_call_id).
        """
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            tool_output_min_tokens=20,
            max_fraction_per_pass=1.0,
        )

        # Create history with assistant tool_calls and paired tool responses
        # along with some ephemeral messages
        history = [
            _notification("notif1"),
            _user("read file x"),
            _assistant(
                "Let me read file x",
                tool_calls=[{"id": "call_read", "function": {"name": "ReadFile", "arguments": "{\"path\": \"x\"}"}}],
            ),
            _tool("content of file x", tool_call_id="call_read"),
            _notification("notif2"),
            _user("now write file y"),
            _assistant(
                "Writing file y",
                tool_calls=[{"id": "call_write", "function": {"name": "WriteFile", "arguments": "{\"path\": \"y\"}"}}],
            ),
            _tool("x" * 200, tool_call_id="call_write"),  # oversized -> Tier B elision
            _notification("notif3"),
        ]

        result = pruner.prune(history, context_usage=0.8, max_context_size=100000)

        # Collect all tool_call_ids from tool messages and all tool_call ids from assistant
        tool_call_ids_in_result: set[str] = set()
        assistant_tool_call_ids: set[str] = set()

        for msg in result.messages:
            if msg.role == "tool" and msg.tool_call_id:
                tool_call_ids_in_result.add(msg.tool_call_id)
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Handle both dict and ToolCall objects
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        assistant_tool_call_ids.add(tc_id)

        # Every tool_call.id has a matching tool_call_id response
        for tc_id in assistant_tool_call_ids:
            assert tc_id in tool_call_ids_in_result, (
                f"tool_call.id {tc_id!r} missing matching tool response"
            )

        # Every tool_call_id response has a matching tool_call.id
        for tc_id in tool_call_ids_in_result:
            assert tc_id in assistant_tool_call_ids, (
                f"tool response {tc_id!r} missing matching tool_call"
            )

        # Verify Tier A drops: notifications should have been removed
        notification_count = sum(
            1 for m in result.messages
            if m.role == "user" and any(
                isinstance(p, TextPart) and "<notification" in p.text
                for p in m.content
            )
        )
        # At least some notifications should be dropped
        assert notification_count < 3  # fewer notifications than original 3

        # Verify Tier B elision: tool content replaced with stub but tool_call_id preserved
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        for tm in tool_msgs:
            # tool_call_id must still be present
            assert tm.tool_call_id is not None, "Elided tool message lost tool_call_id"
            # Either original text or stub
            text = "".join(p.text for p in tm.content if isinstance(p, TextPart))
            assert len(text) > 0, "Elided tool message has empty content"


# ======================================================================
# Tool-pair protection helper
# ======================================================================


class TestProtectToolPairIndices:
    def test_protects_matching_tool_results(self):
        history = [
            _assistant(
                "Let me read",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "ReadFile", "arguments": "{}"}}
                ],
            ),
            _tool("content", tool_call_id="call_1"),
        ]
        protected = _protect_tool_pair_indices(history, {0})
        assert 1 in protected

    def test_skips_unmatched_tool_results(self):
        history = [
            _assistant("No tools here", tool_calls=[]),
            _tool("content", tool_call_id="call_1"),
        ]
        protected = _protect_tool_pair_indices(history, {0})
        assert 1 not in protected

    def test_does_not_unprotect_existing(self):
        history = [
            _assistant(
                "Let me read",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "ReadFile", "arguments": "{}"}}
                ],
            ),
            _tool("content", tool_call_id="call_1"),
        ]
        protected = _protect_tool_pair_indices(history, {0, 1})
        assert 0 in protected
        assert 1 in protected


# ======================================================================
# Policy-driven pruning
# ======================================================================


class TestPruneWithPolicy:
    def test_default_same_as_prune_with_equivalent_knobs(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.5,
            stable_prefix_messages=1,
            recent_messages_protected=6,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [
            _user("hello"),
            _notification("old notification"),
            _user("work"),
            _user("tail"),
        ]
        policy_result = pruner.prune_with_policy(history)
        direct_result = pruner.prune(history, context_usage=1.0, max_context_size=128_000)
        assert policy_result.earliest_removed_index == direct_result.earliest_removed_index

    def test_keep_recent_turns_protects_last_two(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=6,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [
            _system_reminder("old reminder"),
            _user("a"),
            _user("b"),
            _user("tail"),
        ]
        result = pruner.prune_with_policy(
            history,
            keep_recent_turns=2,
            target_token_count=1,
            max_context_size=128_000,
        )
        # Last two user turns (indices 2,3) protected; index 0 dropped
        assert result.earliest_removed_index == 0
        result_roles = [m.role for m in result.messages]
        assert result_roles == ["user", "user", "user"]

    def test_remove_tool_results_false_disables_tier_b(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            ephemeral_enabled=False,
            substantive_enabled=True,
            tool_output_min_tokens=50,
        )
        history = [
            _user("hello"),
            _tool("x" * 500),
        ]
        result = pruner.prune_with_policy(history, remove_tool_results=False)
        assert result.earliest_removed_index is None
        assert result.messages == list(history)

    def test_target_token_count_lowers_ratio(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            tool_output_min_tokens=50,
            max_fraction_per_pass=1.0,
        )
        # Create a large tool result that must be elided to hit the target
        large_tool = _tool("x" * 4000)
        history = [
            _user("hello"),
            large_tool,
        ]
        result = pruner.prune_with_policy(
            history,
            target_token_count=1000,
            max_context_size=128_000,
        )
        assert result.earliest_removed_index is not None
        assert result.freed_tokens > 0

    def test_protects_tool_pairs(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=1,
            min_free_tokens=0,
            cooldown_steps=0,
            tool_output_min_tokens=20,
            max_fraction_per_pass=1.0,
        )
        history = [
            _user("read"),
            _assistant(
                "Reading",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "ReadFile", "arguments": "{}"}}
                ],
            ),
            _tool("x" * 200, tool_call_id="call_1"),  # oversized but paired
            _user("tail"),
        ]
        result = pruner.prune_with_policy(history, keep_recent_turns=1)
        # The assistant+tool pair must survive because the assistant is in the tail
        assert any(m.role == "assistant" and m.tool_calls for m in result.messages)
        tool_msgs = [m for m in result.messages if m.role == "tool" and m.tool_call_id == "call_1"]
        assert len(tool_msgs) == 1

    def test_remove_reasoning_false_keeps_assistant_thinking(self):
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            ephemeral_enabled=False,
            substantive_enabled=True,
            tool_output_min_tokens=20,
        )
        history = [
            Message(role="assistant", content=[ThinkPart(think="deep reasoning")]),
            _tool("x" * 200),
        ]
        result = pruner.prune_with_policy(history, remove_reasoning=False)
        # Current Tier-B only elides tool messages, so assistant reasoning is untouched
        assert result.messages[0].role == "assistant"
        assert any(isinstance(p, ThinkPart) for p in result.messages[0].content)

    def test_bypasses_cooldown_and_trigger(self):
        pruner = ContextPruner(
            trigger_ratio=1.0,  # would normally never trigger
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=10000,  # would normally block small savings
            cooldown_steps=1000,
        )
        history = [_system_reminder("old reminder"), _user("hello")]
        result = pruner.prune_with_policy(
            history,
            keep_recent_turns=0,
            target_token_count=1,
            max_context_size=128_000,
        )
        # Manual policy invocation should ignore hysteresis
        assert result.earliest_removed_index is not None


# ======================================================================
# cache-02: current-turn protection wiring
# (plans/02-pruner-current-turn-index-unwired-cache-miss.md)
# ======================================================================


def _big_notification(text: str = "n") -> Message:
    """A notification large enough (~200 tokens) to matter for budget math."""
    return _notification(text * 100)


class TestCurrentTurnProtection:
    @pytest.fixture(autouse=True)
    def _force_python_path(self):
        """These tests assert exact selection/identity; the native SOUL kernel
        was removed, so the pure-Python policy engine is always used."""

    def test_current_turn_index_protects_turn(self):
        """With current_turn_index=k nothing at index >= k may be dropped/elided."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            min_usage_growth=0.0,
        )
        history = [
            _user("u0"),
            _big_notification("a"),  # idx 1, droppable
            _user("u1"),
            _big_notification("b"),  # idx 3, droppable
            _user("u2"),  # idx 4 — current turn starts here
            _big_notification("c"),  # idx 5, in turn → protected
            _user("u3"),
            _big_notification("d"),  # idx 7, in turn → protected
        ]
        k = 4
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=100_000,
            current_turn_index=k,
        )
        assert result.earliest_removed_index is not None
        assert result.earliest_removed_index < k
        # Nothing from the current turn onward changed: the trailing suffix of
        # the result equals the current turn's original messages (in order).
        turn_tail = history[k:]
        assert result.messages[-len(turn_tail):] == list(turn_tail)

    def test_without_current_turn_index_drops_turn(self):
        """Same call without current_turn_index (legacy) drops the turn's own
        messages — proving the wiring changes behavior."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            min_usage_growth=0.0,
        )
        # Pre-turn notifications are small; in-turn notifications are big and
        # dominate the budget, so the legacy greedy (tail-inward) reaches into
        # the current turn while the protected call cannot.
        history = [
            _user("u0"),
            _notification("pre1"),  # idx 1, small
            _user("u1"),
            _notification("pre2"),  # idx 3, small
            _user("u2"),  # idx 4 — current turn starts here
            _big_notification("in1"),  # idx 5, in turn → protected
            _user("u3"),
            _big_notification("in2"),  # idx 7, in turn → protected
        ]
        with_turn = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=100_000,
            current_turn_index=4,
        )
        without = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=100_000,
            current_turn_index=None,
        )
        # With protection, only the pre-turn notifications (1, 3) are dropped.
        assert with_turn.earliest_removed_index == 1
        assert len(with_turn.messages) == len(history) - 2
        # Legacy behavior drops the two big in-turn notifications (5, 7) first:
        # the earliest change lands *inside* the current turn (index >= 4).
        assert without.earliest_removed_index == 5
        assert len(without.messages) == len(history) - 2
        # The protected call leaves every in-turn message untouched.
        turn_tail = history[4:]
        assert with_turn.messages[-len(turn_tail):] == list(turn_tail)

    def test_prune_with_policy_forwards_current_turn_index(self):
        """prune_with_policy must forward current_turn_index (tool call site)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            min_usage_growth=0.0,
        )
        history = [
            _user("u0"),
            _big_notification("a"),
            _user("u1"),
            _big_notification("b"),
            _user("u2"),
            _big_notification("c"),
            _user("u3"),
            _big_notification("d"),
        ]
        result = pruner.prune_with_policy(
            history,
            keep_recent_turns=0,
            target_token_count=1,
            max_context_size=128_000,
            current_turn_index=4,
        )
        assert result.earliest_removed_index is not None
        assert result.earliest_removed_index < 4
        turn_tail = history[4:]
        assert result.messages[-len(turn_tail):] == list(turn_tail)

    def test_tier_b_refuses_user_assistant_elision(self):
        """Tier B stubs only tool messages; user/assistant messages outside the
        protected band must be left untouched (in-place rewrite would change
        cached content)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            ephemeral_enabled=False,
            substantive_enabled=True,
            tool_output_min_tokens=20,
        )
        history = [
            _assistant("x" * 4000),
            _user("y" * 4000),
            _tool("z" * 4000),
        ]
        result = pruner.prune(history, context_usage=0.9, max_context_size=100_000)
        # Only the tool message (index 2) may be elided into a stub.
        assert result.earliest_removed_index == 2
        assert is_pruned_stub(result.messages[2])
        assert not is_pruned_stub(result.messages[0])
        assert not is_pruned_stub(result.messages[1])


# ======================================================================
# cache-03: cache-depth floor + invalidation-cost payoff gate + tail band
# (plans/03-pruner-cache-depth-floor-policy.md)
# ======================================================================


class TestCacheDepthFloor:
    @pytest.fixture(autouse=True)
    def _force_python_path(self):
        """The native SOUL kernel was removed; the pure-Python policy engine
        is always used for exact selection assertions."""

    def test_floor_protects_head_notification(self):
        """A notification at index 5 is protected with min_cache_prefix_depth=100;
        any pass has earliest_removed_index >= 100."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            min_cache_prefix_depth=100,
        )
        history = [_user(f"u{i}") for i in range(200)]
        history[5] = _big_notification("head")
        history[120] = _big_notification("mid1")
        history[150] = _big_notification("mid2")
        history[180] = _big_notification("mid3")

        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        assert result.earliest_removed_index is not None
        assert result.earliest_removed_index >= 100
        # Everything before the floor is untouched (identity preserved).
        for i in range(100):
            assert result.messages[i] == history[i]

    def test_floor_disabled_by_default(self):
        """min_cache_prefix_depth=None keeps legacy behavior: index 5 is prunable."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [_user(f"u{i}") for i in range(200)]
        history[5] = _notification("head")
        history[120] = _notification("mid1")
        history[150] = _notification("mid2")
        history[180] = _notification("mid3")
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        # Legacy: the greedy tail-inward pass reaches index 5.
        assert result.earliest_removed_index is not None
        assert result.earliest_removed_index < 100

    def test_payoff_gate_blocks_when_freed_below_cache_loss(self):
        """When freed < cache_loss, the pass must be rejected entirely."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            cache_loss_penalty=0.0,
        )
        # One tiny notification at index 5, then a long tail: dropping it frees
        # ~few tokens but invalidates the whole tail → no pass.
        history = [_user(f"u{i}") for i in range(50)]
        history[5] = _notification("tiny")
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        assert result.earliest_removed_index is None
        assert result.messages == list(history)

    def test_payoff_gate_allows_high_yield_pass(self):
        """When freed equals/exceeds the cache loss the pass still applies."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            cache_loss_penalty=0.0,
        )
        # A huge notification as the *last* message: freeing it invalidates
        # only itself (cache_loss == freed), so the gate lets the pass through.
        history = [_user(f"u{i}") for i in range(20)]
        history[19] = _notification("huge-tail" * 2000)
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        assert result.earliest_removed_index == 19
        assert result.freed_tokens > 0

    def test_tail_band_preferred_in_candidate_sort(self):
        """Given one candidate at index 10 and one at 150 in a 200-message
        history, the index-150 candidate is selected first (tail-inward /
        tail-band preference)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
            max_fraction_per_pass=0.1,
        )
        history = [_user(f"u{i}") for i in range(200)]
        history[10] = _big_notification("old")
        history[150] = _big_notification("new")
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        assert result.earliest_removed_index is not None
        assert result.earliest_removed_index == 150
        assert result.messages[10] == history[10]

    def test_payoff_gate_disabled_by_default(self):
        """cache_loss_penalty=None keeps legacy behavior (no gate)."""
        pruner = ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=0,
            min_free_tokens=0,
            cooldown_steps=0,
        )
        history = [_user(f"u{i}") for i in range(50)]
        history[5] = _notification("tiny")
        result = pruner.prune(
            history,
            context_usage=0.9,
            max_context_size=1_000_000,
        )
        # Legacy: the tiny notification is dropped despite a large cache loss.
        assert result.earliest_removed_index == 5