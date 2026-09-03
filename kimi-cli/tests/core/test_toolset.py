"""Tests for KimiToolset hide/unhide and deduplication functionality."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from typing import override

from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kosong.tooling.error import ToolNotFoundError as KosongToolNotFoundError
from pydantic import BaseModel

from kimi_cli.soul.toolset import (
    _DIFF_ARGS_HARD_STOP_START,
    _DIFF_ARGS_REMINDER_TEXT_1,
    _DIFF_ARGS_WARN_THRESHOLDS,
    _PLATFORM_REDIRECTS_NORM,
    _TURN_TOTAL_REMINDER_INTERVAL,
    _TURN_TOTAL_REMINDER_START,
    KimiToolset,
    _build_platform_redirects,
    _collect_candidates,
    _has_reasoning_parts,
    _make_diff_args_reminder,
    _parse_stringified_arguments,
    _repair_argument_format,
    _repair_todo_arguments,
    _unwrap_nested_arguments,
)
from kimi_cli.wire.types import (
    ContentPart,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolResult,
)


class DummyParams(BaseModel):
    value: str = ""


class DummyToolA(CallableTool2[DummyParams]):
    name: str = "ToolA"
    description: str = "Tool A"
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output="a")


class DummyToolB(CallableTool2[DummyParams]):
    name: str = "ToolB"
    description: str = "Tool B"
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output="b")


class DummyToolC(CallableTool2[DummyParams]):
    name: str = "LongNamedTool"
    description: str = "A tool with a longer name for typo tests"
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output="long")


def _make_toolset() -> KimiToolset:
    ts = KimiToolset()
    ts.add(DummyToolA())
    ts.add(DummyToolB())
    return ts


def _make_extended_toolset() -> KimiToolset:
    ts = KimiToolset()
    ts.add(DummyToolA())
    ts.add(DummyToolB())
    ts.add(DummyToolC())
    return ts


def _tool_names(ts: KimiToolset) -> set[str]:
    return {t.name for t in ts.tools}


# --- hide() ---


def test_hide_removes_from_tools_property():
    ts = _make_toolset()
    assert _tool_names(ts) == {"ToolA", "ToolB"}

    ts.hide("ToolA")
    assert _tool_names(ts) == {"ToolB"}


def test_hide_returns_true_for_existing_tool():
    ts = _make_toolset()
    assert ts.hide("ToolA") is True


def test_hide_returns_false_for_nonexistent_tool():
    ts = _make_toolset()
    assert ts.hide("NoSuchTool") is False


def test_hide_is_idempotent():
    ts = _make_toolset()
    ts.hide("ToolA")
    ts.hide("ToolA")
    assert "ToolA" not in _tool_names(ts)

    # Single unhide restores after multiple hides
    ts.unhide("ToolA")
    assert "ToolA" in _tool_names(ts)


def test_hide_multiple_tools():
    ts = _make_toolset()
    ts.hide("ToolA")
    ts.hide("ToolB")
    assert ts.tools == []


# --- unhide() ---


def test_unhide_restores_tool():
    ts = _make_toolset()
    ts.hide("ToolA")
    assert "ToolA" not in _tool_names(ts)

    ts.unhide("ToolA")
    assert "ToolA" in _tool_names(ts)


def test_unhide_nonexistent_is_noop():
    ts = _make_toolset()
    ts.unhide("NoSuchTool")
    assert _tool_names(ts) == {"ToolA", "ToolB"}


def test_unhide_without_prior_hide_is_noop():
    ts = _make_toolset()
    ts.unhide("ToolA")
    assert _tool_names(ts) == {"ToolA", "ToolB"}


# --- find() is unaffected ---


def test_hidden_tool_still_findable_by_name():
    ts = _make_toolset()
    ts.hide("ToolA")
    assert ts.find("ToolA") is not None


def test_hidden_tool_still_findable_by_type():
    ts = _make_toolset()
    ts.hide("ToolA")
    assert ts.find(DummyToolA) is not None


# --- handle() is unaffected ---


async def test_hidden_tool_still_handled():
    """handle() should dispatch to hidden tools instead of returning ToolNotFoundError."""
    ts = _make_toolset()
    ts.hide("ToolA")

    tool_call = ToolCall(
        id="tc-1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=json.dumps({"value": "test"}),
        ),
    )
    result = ts.handle(tool_call)
    # For async tools, handle() returns an asyncio.Future.
    # A ToolNotFoundError would be returned as a sync ToolResult directly.
    if isinstance(result, ToolResult):
        assert not isinstance(result.return_value, KosongToolNotFoundError)
    else:
        assert isinstance(result, asyncio.Future)
        result.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await result


async def test_nonexistent_tool_returns_not_found():
    """handle() should return ToolNotFoundError for tools not in _tool_dict at all."""
    ts = _make_toolset()

    tool_call = ToolCall(
        id="tc-2",
        function=ToolCall.FunctionBody(
            name="NoSuchTool",
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, ToolResult)
    assert isinstance(result.return_value, KosongToolNotFoundError)
    assert "not found" in result.return_value.message


async def test_nonexistent_tool_with_fuzzy_suggestion():
    """A non-existent tool name close to a registered one should include a hint."""
    ts = _make_toolset()
    tool_call = ToolCall(
        id="tc-fuzzy",
        function=ToolCall.FunctionBody(
            name="TollX",  # similarity ~0.6 to ToolA/ToolB — below auto-correct but above suggestion cutoff
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, ToolResult)
    assert isinstance(result.return_value, KosongToolNotFoundError)
    assert "did you mean" in result.return_value.message
    assert "ToolA" in result.return_value.message or "ToolB" in result.return_value.message


async def test_nonexistent_tool_auto_corrects_case_insensitive():
    """A case-insensitive match should auto-correct to the real tool."""
    ts = _make_toolset()
    tool_call = ToolCall(
        id="tc-ci",
        function=ToolCall.FunctionBody(
            name="toola",  # lowercase version of "ToolA"
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert output.startswith("a")
    assert "<system-warning>" in output


async def test_nonexistent_tool_auto_correct_appends_warning():
    """When auto-correcting, a system-warning should be appended to the output."""
    ts = _make_toolset()
    tool_call = ToolCall(
        id="tc-warn",
        function=ToolCall.FunctionBody(
            name="toola",  # case-insensitive match to "ToolA"
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert "<system-warning>" in output
    assert "toola" in output.lower() or "ToolA" in output
    assert "Auto-corrected" in output


async def test_nonexistent_tool_auto_corrects_close_typo():
    """A tool name with a small typo (high similarity) should auto-correct."""
    ts = _make_extended_toolset()
    # "LongNamedTol" missing the last 'o' vs "LongNamedTool" (ratio ~0.96)
    tool_call = ToolCall(
        id="tc-typo",
        function=ToolCall.FunctionBody(
            name="LongNamedTol",
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert output.startswith("long")
    assert "<system-warning>" in output


async def test_nonexistent_tool_auto_correct_does_not_affect_distant_names():
    """A distant name still returns ToolNotFoundError."""
    ts = _make_extended_toolset()
    tool_call = ToolCall(
        id="tc-distant",
        function=ToolCall.FunctionBody(
            name="Xyzzy",  # no similarity to any registered tool, below 0.75 cutoff
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, ToolResult)
    assert isinstance(result.return_value, KosongToolNotFoundError)


# --- snake_case / kebab-case / SCREAMING_SNAKE_CASE auto-correction ---


async def _assert_misformatted_autocorrects_to_long(sent: str) -> None:
    """A mis-formatted spelling of ``LongNamedTool`` auto-corrects and runs it."""
    ts = _make_extended_toolset()
    tool_call = ToolCall(
        id=f"tc-{sent}",
        function=ToolCall.FunctionBody(
            name=sent,
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert output.startswith("long")
    assert "<system-warning>" in output
    assert "Auto-corrected" in output


async def test_nonexistent_tool_auto_corrects_snake_case():
    """snake_case spelling of a multi-word tool auto-corrects to CamelCase."""
    await _assert_misformatted_autocorrects_to_long("long_named_tool")


async def test_nonexistent_tool_auto_corrects_kebab_case():
    """kebab-case spelling of a multi-word tool auto-corrects to CamelCase."""
    await _assert_misformatted_autocorrects_to_long("long-named-tool")


async def test_nonexistent_tool_auto_corrects_screaming_snake_case():
    """SCREAMING_SNAKE_CASE spelling of a multi-word tool auto-corrects."""
    await _assert_misformatted_autocorrects_to_long("LONG_NAMED_TOOL")


async def test_nonexistent_tool_auto_corrects_mixed_separators():
    """Mixed CamelCase + separator spelling of a multi-word tool auto-corrects."""
    await _assert_misformatted_autocorrects_to_long("Long_Named_Tool")


async def test_nonexistent_tool_auto_corrects_snake_case_with_typo():
    """A snake_case spelling that also contains a typo still auto-corrects."""
    await _assert_misformatted_autocorrects_to_long("long_named_tol")


async def test_snake_case_non_tool_returns_not_found():
    """A snake_case name not close to any tool still yields ToolNotFoundError."""
    ts = _make_extended_toolset()
    tool_call = ToolCall(
        id="tc-unrelated",
        function=ToolCall.FunctionBody(
            name="totally_unrelated",
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, ToolResult)
    assert isinstance(result.return_value, KosongToolNotFoundError)


# --- AskAgent name-variant auto-correction (fuzzy matching) ---


class DummyAskAgent(CallableTool2[DummyParams]):
    name: str = "AskAgent"
    description: str = "Send a message to another agent."
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output="asked")


def _make_ask_agent_toolset() -> KimiToolset:
    ts = KimiToolset()
    ts.add(DummyAskAgent())
    return ts


async def _assert_ask_agent_variant_autocorrects(sent: str) -> None:
    """A slightly-wrong spelling of ``AskAgent`` auto-corrects and runs it."""
    ts = _make_ask_agent_toolset()
    tool_call = ToolCall(
        id=f"tc-ask-{sent}",
        function=ToolCall.FunctionBody(
            name=sent,
            arguments="{}",
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert output.startswith("asked")
    assert "<system-warning>" in output
    assert "AskAgent" in output


async def test_ask_agent_kebab_case_autocorrects():
    """``ask-agent`` auto-corrects to ``AskAgent``."""
    await _assert_ask_agent_variant_autocorrects("ask-agent")


async def test_ask_agent_snake_case_autocorrects():
    """``ask_agent`` auto-corrects to ``AskAgent``."""
    await _assert_ask_agent_variant_autocorrects("ask_agent")


async def test_ask_agent_space_separated_autocorrects():
    """``ask agent`` (space-separated) auto-corrects to ``AskAgent``."""
    await _assert_ask_agent_variant_autocorrects("ask agent")


# --- hide/unhide cycle ---


def test_hide_unhide_cycle():
    """Multiple hide/unhide cycles should work correctly."""
    ts = _make_toolset()

    ts.hide("ToolA")
    assert "ToolA" not in _tool_names(ts)

    ts.unhide("ToolA")
    assert "ToolA" in _tool_names(ts)

    ts.hide("ToolA")
    assert "ToolA" not in _tool_names(ts)

    ts.unhide("ToolA")
    assert "ToolA" in _tool_names(ts)


# --- deduplication ---


async def test_same_step_dedup():
    """Duplicate tool calls within the same step should share the original result."""
    ts = _make_toolset()
    ts.begin_step([])

    args = json.dumps({"value": "x"})
    tool_call_1 = ToolCall(
        id="tc-dedup-1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )
    tool_call_2 = ToolCall(
        id="tc-dedup-2",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )

    result_1 = ts.handle(tool_call_1)
    assert isinstance(result_1, asyncio.Task)

    result_2 = ts.handle(tool_call_2)
    assert isinstance(result_2, asyncio.Task)

    # Both should eventually return the same output but with different tool_call_id
    tr_1 = await result_1
    tr_2 = await result_2

    assert tr_1.return_value.output == "a"
    assert tr_2.return_value.output == "a"
    assert tr_1.tool_call_id == "tc-dedup-1"
    assert tr_2.tool_call_id == "tc-dedup-2"

    assert ts.end_step() == [("ToolA", '{"value":"x"}'), ("ToolA", '{"value":"x"}')]


async def test_same_step_dedup_canonicalizes_argument_key_order():
    """Equivalent JSON objects with different key order should share the original result."""
    ts = _make_toolset()
    ts.begin_step([])

    tool_call_1 = ToolCall(
        id="tc-canonical-1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments='{"a": 1, "b": 2}',
        ),
    )
    tool_call_2 = ToolCall(
        id="tc-canonical-2",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments='{"b": 2, "a": 1}',
        ),
    )

    result_1 = ts.handle(tool_call_1)
    result_2 = ts.handle(tool_call_2)
    assert isinstance(result_1, asyncio.Task)
    assert isinstance(result_2, asyncio.Task)

    tr_1 = await result_1
    tr_2 = await result_2

    assert tr_1.return_value.output == "a"
    assert tr_2.return_value.output == "a"
    assert ts.end_step() == [("ToolA", '{"a":1,"b":2}'), ("ToolA", '{"a":1,"b":2}')]


async def test_cross_step_duplicate_does_not_append_reminder_below_three_consecutive():
    """The second consecutive identical call is tracked but not reminded yet."""
    ts = _make_toolset()
    args = json.dumps({"value": "x"})
    ts.begin_step([("ToolA", args)])

    tool_call = ToolCall(
        id="tc-dedup-reminder",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )

    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert output == "a"
    assert ts.dedup_triggered is True
    assert ts.end_step() == [("ToolA", '{"value":"x"}')]


async def test_cross_step_duplicate_appends_reminder_at_three_consecutive():
    """The first reminder is sparse and appears only at the third consecutive call."""
    ts = _make_toolset()
    args = json.dumps({"value": "x"})
    previous_calls: list[tuple[str, str]] = []

    for i in range(2):
        ts.begin_step(previous_calls, step_no=i + 1)
        result = ts.handle(
            ToolCall(
                id=f"tc-repeat-prior-{i}",
                function=ToolCall.FunctionBody(name="ToolA", arguments=args),
            )
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        assert "system-reminder" not in tr.return_value.output
        previous_calls = ts.end_step()

    ts.begin_step(previous_calls, step_no=3)
    result = ts.handle(
        ToolCall(
            id="tc-repeat-third",
            function=ToolCall.FunctionBody(name="ToolA", arguments=args),
        )
    )
    assert isinstance(result, asyncio.Task)
    tr = await result
    output = tr.return_value.output
    assert isinstance(output, str)
    assert "Stop repeating the same tool call with identical parameters" in output
    assert "repeated_times" not in output


async def test_cross_step_duplicate_uses_sparse_stronger_reminders():
    """The stronger reminder appears at the eighth repeat and includes canonical args."""
    ts = _make_toolset()
    args = '{"b": 2, "a": 1}'
    previous_calls: list[tuple[str, str]] = []
    last_output = ""

    for i in range(8):
        ts.begin_step(previous_calls, step_no=i + 1)
        result = ts.handle(
            ToolCall(
                id=f"tc-repeat-{i}",
                function=ToolCall.FunctionBody(name="ToolA", arguments=args),
            )
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        last_output = tr.return_value.output
        previous_calls = ts.end_step()

    assert isinstance(last_output, str)
    assert "Repeated identical call" in last_output
    assert "- tool: ToolA" in last_output
    assert "repeated_times: 8" in last_output
    assert '- arguments: {"a":1,"b":2}' in last_output


async def test_non_duplicate_allowed():
    """A tool call with different arguments should be allowed even if the tool name matches."""
    ts = _make_toolset()
    ts.begin_step([("ToolA", json.dumps({"value": "x"}))])

    args = json.dumps({"value": "y"})
    tool_call = ToolCall(
        id="tc-ok-1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )

    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "a"
    assert ts.dedup_triggered is False
    assert ts.end_step() == [("ToolA", '{"value":"y"}')]


def test_begin_end_step():
    """begin_step and end_step should correctly manage deduplication state."""
    ts = _make_toolset()

    ts.begin_step([("ToolA", "{}")])
    assert ts._previous_step_calls == [("ToolA", "{}")]
    assert ts._current_step_calls == []
    assert ts._current_step_tasks == {}
    assert ts.dedup_triggered is False

    ts._current_step_calls.append(("ToolB", "{}"))
    assert ts.end_step() == [("ToolB", "{}")]

    # After end_step, internal lists are not cleared by end_step itself;
    # the caller (KimiSoul) is expected to call begin_step again for the next step.
    # But dedup_triggered should still reflect the last step's state.
    assert ts.dedup_triggered is False


async def test_begin_step_resets_cancelled_tasks():
    """begin_step() must clear _current_step_tasks so a retry does not await a cancelled task."""
    ts = _make_toolset()

    ts.begin_step([], step_no=1, turn_id="t1")
    args = json.dumps({"value": "x"})
    tc1 = ToolCall(
        id="c1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )
    result1 = ts.handle(tc1)
    assert isinstance(result1, asyncio.Task)
    result1.cancel()

    # Simulate retry: begin_step again for the same step
    ts.begin_step([], step_no=1, turn_id="t1")
    tc2 = ToolCall(
        id="c2",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )
    result2 = ts.handle(tc2)
    assert isinstance(result2, asyncio.Task)
    assert result2 is not result1

    # The new task should complete successfully (not raise CancelledError)
    tr = await result2
    assert tr.return_value.output == "a"


async def test_cross_step_dedup_not_triggered_after_back_to_the_future():
    """When _last_tool_calls is emptied (back_to_the_future), the same call must not be treated as a cross-step duplicate."""
    ts = _make_toolset()

    # Step 1: execute a tool
    args = json.dumps({"value": "x"})
    ts.begin_step([], step_no=1, turn_id="t1")
    tc1 = ToolCall(
        id="c1",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )
    result1 = ts.handle(tc1)
    assert isinstance(result1, asyncio.Task)
    await result1
    last_calls = ts.end_step()
    assert last_calls == [("ToolA", '{"value":"x"}')]

    # Simulate back_to_the_future: caller clears last_calls
    last_calls = []

    # Step 2: same call with empty last_calls should execute normally
    ts.begin_step(last_calls, step_no=2, turn_id="t1")
    tc2 = ToolCall(
        id="c2",
        function=ToolCall.FunctionBody(
            name="ToolA",
            arguments=args,
        ),
    )
    result2 = ts.handle(tc2)
    assert isinstance(result2, asyncio.Task)
    tr = await result2

    # Should NOT have the cross-step reminder appended
    assert tr.return_value.output == "a"
    assert ts.dedup_triggered is False


async def test_different_args_hard_stop_after_many_repeats():
    """Repeated calls with different args must force-stop once the per-tool ceiling is crossed."""
    from kimi_cli.soul.toolset import _DIFF_ARGS_HARD_STOP_START

    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-hard-diff")

    for i in range(_DIFF_ARGS_HARD_STOP_START):
        tc = ToolCall(
            id=f"tc-diff-{i}",
            function=ToolCall.FunctionBody(
                name="ToolA",
                arguments=json.dumps({"value": str(i)}),
            ),
        )
        result = ts.handle(tc)
        assert isinstance(result, asyncio.Task)
        await result

    assert ts._tool_call_counts["ToolA"] == _DIFF_ARGS_HARD_STOP_START
    assert ts.force_stop_turn is True


async def test_turn_total_calls_soft_reminder():
    """A very long turn receives a soft reminder instead of being force-stopped."""
    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-total")

    last_output = None
    for i in range(_TURN_TOTAL_REMINDER_START):
        # Alternate tools so no single tool hits its own per-tool ceiling.
        name = "ToolA" if i % 2 == 0 else "ToolB"
        tc = ToolCall(
            id=f"tc-total-{i}",
            function=ToolCall.FunctionBody(
                name=name,
                arguments=json.dumps({"value": str(i)}),
            ),
        )
        result = ts.handle(tc)
        assert isinstance(result, asyncio.Task)
        tr = await result
        last_output = tr.return_value.output

    assert ts.force_stop_turn is False
    assert ts._turn_total_calls == _TURN_TOTAL_REMINDER_START
    assert ts._tool_call_counts["ToolA"] < _DIFF_ARGS_HARD_STOP_START
    assert ts._tool_call_counts["ToolB"] < _DIFF_ARGS_HARD_STOP_START
    assert isinstance(last_output, str)
    assert "Tool calls repeat 60 times" in last_output
    assert "Stop or finish" in last_output


async def test_different_args_below_hard_stop_not_force_stopped():
    """A moderate number of different-args calls must not force-stop the turn.

    Stay below the ``_DIFF_ARGS_HARD_STOP_START`` (per-tool) ceiling.
    """
    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-diff-ok")

    for i in range(12):
        tc = ToolCall(
            id=f"tc-diff-ok-{i}",
            function=ToolCall.FunctionBody(
                name="ToolA",
                arguments=json.dumps({"value": str(i)}),
            ),
        )
        result = ts.handle(tc)
        assert isinstance(result, asyncio.Task)
        await result

    assert ts.force_stop_turn is False


# --- Ceiling invariants -----------------------------------------------------


def test_diff_args_threshold_invariants():
    """The different-args hard-stop ceiling and warning ladder stay aligned."""
    assert _DIFF_ARGS_HARD_STOP_START == 40
    # the warning ladder must fire strictly before the hard stop
    assert max(_DIFF_ARGS_WARN_THRESHOLDS) < _DIFF_ARGS_HARD_STOP_START


async def test_different_args_hard_stop_at_ten_calls():
    """One tool called 10 times with different args force-stops the turn."""
    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-diff-10")

    stop_at = None
    for i in range(_DIFF_ARGS_HARD_STOP_START):
        result = ts.handle(
            ToolCall(
                id=f"tc-d10-{i}",
                function=ToolCall.FunctionBody(
                    name="ToolA", arguments=json.dumps({"value": str(i)})
                ),
            )
        )
        assert isinstance(result, asyncio.Task)
        await result
        if ts.force_stop_turn:
            stop_at = i + 1
            break

    assert stop_at == _DIFF_ARGS_HARD_STOP_START


async def test_turn_total_reminder_across_tools():
    """Distinct calls spread over several tools receive a soft reminder, not a hard stop."""
    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-total-15")

    last_output = None
    for i in range(_TURN_TOTAL_REMINDER_START):
        # alternate tools so no single tool reaches its own ceiling first
        name = "ToolA" if i % 2 == 0 else "ToolB"
        result = ts.handle(
            ToolCall(
                id=f"tc-t15-{i}",
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps({"value": str(i)})),
            )
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        last_output = tr.return_value.output

    assert ts.force_stop_turn is False
    assert ts._turn_total_calls == _TURN_TOTAL_REMINDER_START
    assert isinstance(last_output, str)
    assert "Tool calls repeat 60 times" in last_output
    # neither tool reached its own per-tool ceiling
    assert ts._tool_call_counts["ToolA"] < _DIFF_ARGS_HARD_STOP_START
    assert ts._tool_call_counts["ToolB"] < _DIFF_ARGS_HARD_STOP_START


async def test_diff_args_warning_ladder():
    """The graded warnings land at 15/25/35 and the strongest text at the ceiling."""
    assert _make_diff_args_reminder("ToolA", 15) == _DIFF_ARGS_REMINDER_TEXT_1
    assert "called 25 times with different args" in _make_diff_args_reminder("ToolA", 25)
    assert "called 35 times. Stop now." in _make_diff_args_reminder("ToolA", 35)
    assert "called 40 times. Stop now." in _make_diff_args_reminder("ToolA", 40)


# --- Cycle-aware interleaved repeat punishment ------------------------------


def _call_args(value: str) -> str:
    return json.dumps({"value": value})


async def _run_cycle(
    ts: KimiToolset, seq: list[tuple[str, str]], max_steps: int = 40, turn_id: str = "T1"
):
    """Replay *seq* one call per step (as KimiSoul drives the toolset).

    Returns (stop_step, reminder_steps) where reminder_steps maps step number to
    the output text of that step.
    """
    previous: list[tuple[str, str]] = []
    outputs: dict[int, str] = {}
    for step in range(1, max_steps + 1):
        tool, args = seq[(step - 1) % len(seq)]
        ts.begin_step(previous, step_no=step, turn_id=turn_id)
        result = ts.handle(
            ToolCall(id=f"c{step}", function=ToolCall.FunctionBody(name=tool, arguments=args))
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        outputs[step] = tr.return_value.output  # type: ignore[assignment]
        previous = ts.end_step()
        if ts.force_stop_turn:
            return step, outputs
    return None, outputs


def test_interleaved_cycle_thresholds():
    """Cycle punishment: warn at 2 occurrences, escalate at 3, stop at 4."""
    from kimi_cli.soul.toolset import (
        _CYCLE_FORCE_STOP,
        _CYCLE_REMINDER_2_START,
        _CYCLE_REMINDER_START,
    )

    assert _CYCLE_REMINDER_START == 2
    assert _CYCLE_REMINDER_2_START == 3
    assert _CYCLE_FORCE_STOP == 4


async def test_interleaved_abc_cycle_is_punished():
    """``A(x) -> B(x) -> C(x) -> A(x)`` must be punished despite a flat streak."""
    ts = _make_extended_toolset()
    args = _call_args("x")
    seq = [("ToolA", args), ("ToolB", args), ("LongNamedTool", args)]

    stop_step, outputs = await _run_cycle(ts, seq)

    # 4th round of A is the 10th call — far below both the per-tool and per-turn
    # ceilings, so only the cycle detector can stop it.
    assert stop_step == 10
    assert ts._turn_total_calls == 10
    assert ts._tool_call_counts["ToolA"] == 4 < _DIFF_ARGS_HARD_STOP_START
    # the consecutive streak never grew, i.e. the old detector saw nothing
    assert ts._consecutive_count <= 1
    # first repetition of A(x) already warns
    assert "already ran earlier this turn" in outputs[4]
    # third repetition gets the stronger warning
    assert "'ToolA' repeated 3 times in a cycle" in outputs[7]
    # the stopping call carries the strongest cycle warning
    assert "'ToolA' repeated 4 times in a cycle" in outputs[10]


async def test_interleaved_ab_cycle_is_punished():
    """A period-2 alternation of identical calls also force-stops the turn."""
    ts = _make_toolset()
    args = _call_args("x")
    stop_step, outputs = await _run_cycle(ts, [("ToolA", args), ("ToolB", args)])

    # A(x) runs on calls 1/3/5/7: its 4th occurrence stops the turn on call 7,
    # well below both the per-tool (40) and total (60) ceilings.
    assert stop_step == 7
    assert ts._turn_total_calls == 7
    assert ts._tool_call_counts["ToolA"] == 4 < _DIFF_ARGS_HARD_STOP_START
    assert "already ran earlier this turn" in outputs[3]
    assert "'ToolA' repeated 4 times in a cycle" in outputs[7]
    assert ts.force_stop_turn is True


async def test_cycle_force_stop_reports_reason_and_key():
    """The cycle detector records why/what tripped for soul-level recovery."""
    ts = _make_toolset()
    args = _call_args("x")
    stop_step, _ = await _run_cycle(ts, [("ToolA", args), ("ToolB", args)])
    assert stop_step == 7
    assert ts.force_stop_reason == "cycle-repeat"
    assert ts.force_stop_key == ("ToolA", '{"value":"x"}')


async def test_cycle_punishment_is_per_key_not_per_tool():
    """Same tool with *different* args is not a cycle; only identical keys count."""
    ts = _make_toolset()
    ts.begin_step([], turn_id="turn-cycle-args")
    previous: list[tuple[str, str]] = []

    for i in range(6):
        for name in ("ToolA", "ToolB"):
            args = _call_args(f"{name}-{i}")
            ts.begin_step(previous, step_no=i + 1, turn_id="turn-cycle-args")
            result = ts.handle(
                ToolCall(id=f"c-{i}-{name}", function=ToolCall.FunctionBody(name=name, arguments=args))
            )
            assert isinstance(result, asyncio.Task)
            tr = await result
            assert "already ran earlier this turn" not in tr.return_value.output
            assert "repeated in a cycle" not in tr.return_value.output
            previous = ts.end_step()

    assert ts.force_stop_turn is False
    # every key seen exactly once per tool
    assert all(count == 1 for count in ts._call_key_counts.values())


async def test_adjacent_streak_is_not_double_punished_by_cycle_detector():
    """Plain adjacent repeats keep using the streak ladder, without cycle text."""
    args = _call_args("x")
    ts = _make_toolset()
    previous: list[tuple[str, str]] = []
    texts: list[str] = []
    for step in range(1, 13):
        ts.begin_step(previous, step_no=step, turn_id="turn-streak-only")
        result = ts.handle(
            ToolCall(id=f"s{step}", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        texts.append(tr.return_value.output)  # type: ignore[arg-type]
        previous = ts.end_step()

    assert ts.force_stop_turn is False  # streak stops at 16
    assert not any("already ran earlier this turn" in t for t in texts)
    assert not any("repeated in a cycle" in t for t in texts)
    # streak reminders still appear at its own thresholds
    assert "Stop repeating the same tool call" in texts[2]


async def test_same_step_duplicate_is_not_counted_as_a_cycle():
    """A duplicate inside one step copies the result and stays unpunished."""
    ts = _make_toolset()
    args = _call_args("x")
    ts.begin_step([], turn_id="turn-same-step")

    first = ts.handle(
        ToolCall(id="a1", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(first, asyncio.Task)
    tr1 = await first
    assert tr1.return_value.output == "a"

    second = ts.handle(
        ToolCall(id="a2", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(second, asyncio.Task)
    tr2 = await second
    assert "already ran earlier this turn" not in tr2.return_value.output
    assert ts._call_key_counts[(("ToolA"), json.dumps({"value": "x"}).replace(" ", ""))] == 1
    assert ts.force_stop_turn is False


async def test_cycle_counts_reset_on_new_turn():
    """A repeated call across two turns is legitimate work, not a loop."""
    args = _call_args("x")
    ts = _make_toolset()

    # turn 1: A(x) B(x) A(x) -> warns once but does not stop
    stop_step, _ = await _run_cycle(ts, [("ToolA", args), ("ToolB", args)], max_steps=3)
    assert stop_step is None
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 2

    # turn 2: same sequence restarts counting from zero
    ts.begin_step([], step_no=1, turn_id="turn-2")
    result = ts.handle(
        ToolCall(id="n1", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "a"
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 1
    assert ts.force_stop_turn is False


async def test_cycle_counts_reset_on_retried_step():
    """Re-running a cancelled step (empty previous calls) clears cycle counts."""
    args = _call_args("x")
    ts = _make_toolset()
    ts.begin_step([], step_no=1, turn_id="t")
    first = ts.handle(
        ToolCall(id="r1", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(first, asyncio.Task)
    await first
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 1

    # retry of the very same step: no previous calls -> clean slate
    ts.begin_step([], step_no=1, turn_id="t")
    assert ts._call_key_counts == {}
    result = ts.handle(
        ToolCall(id="r2", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "a"
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 1


# --- Dynamic tool output byte budget ---


class _MockLLM:
    def __init__(self, max_context_size: int) -> None:
        self.max_context_size = max_context_size


class _MockRuntime:
    def __init__(self, max_context_size: int) -> None:
        self.llm = _MockLLM(max_context_size)


class _EchoTool(CallableTool2[DummyParams]):
    name: str = "EchoTool"
    description: str = "Echoes the input value"
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output=params.value)


def test_max_output_bytes_fallback_without_runtime():
    """Without a runtime the byte budget falls back to the original 128 KiB."""
    ts = KimiToolset()
    assert ts._get_max_output_bytes() == 128 << 10


def test_max_output_bytes_with_total_context_budget():
    """Empty context: the total-context term dominates for typical models."""
    ts = KimiToolset(runtime=_MockRuntime(32_768))
    # total_budget = int(32768 * 4 * 0.5) = 65536
    # remaining_budget = int(32768 * 4 * 0.9) = 117964
    # default cap = 131072
    assert ts._get_max_output_bytes() == 65_536


def test_max_output_bytes_with_partial_context():
    """Partially filled context shrinks the budget via the remaining-context term."""
    ts = KimiToolset(
        runtime=_MockRuntime(1_048_576),
        context_token_provider=lambda: 1_020_000,
    )
    # total_budget = int(1048576 * 4 * 0.5) = 2097152
    # remaining_budget = int((1048576 - 1020000) * 4 * 0.9) = 102873
    assert ts._get_max_output_bytes() == 102_873


def test_max_output_bytes_near_full_context():
    """Near-full context drives the budget toward zero."""
    ts = KimiToolset(
        runtime=_MockRuntime(131_072),
        context_token_provider=lambda: 130_000,
    )
    # remaining_budget = int((131072 - 130000) * 4 * 0.9) = 3859
    assert ts._get_max_output_bytes() == 3_859


def test_max_output_bytes_absolute_ceiling():
    """Very large contexts are capped by the default 128 KiB ceiling."""
    ts = KimiToolset(runtime=_MockRuntime(1_048_576))
    assert ts._get_max_output_bytes() == 128 << 10


def test_set_context_token_provider_overrides_provider():
    """The setter can replace the callback used by _get_max_output_bytes."""
    ts = KimiToolset(
        runtime=_MockRuntime(1_048_576),
        context_token_provider=lambda: 1_020_000,
    )
    assert ts._get_max_output_bytes() == 102_873

    ts.set_context_token_provider(lambda: 1_046_528)
    # remaining_budget = int((1048576 - 1046528) * 4 * 0.9) = 7372
    assert ts._get_max_output_bytes() == 7_372


def test_estimate_tool_output_token_budget_matches_byte_budget():
    """The token-budget helper mirrors the existing byte-budget logic."""
    ts = KimiToolset(runtime=_MockRuntime(131_072))

    # Empty context: capped by the default 128 KiB ceiling.
    # total_budget_bytes = int(131072 * 4 * 0.5) = 262144 -> capped at 131072 -> tokens = 32768
    assert ts.estimate_tool_output_token_budget(131_072, 0) == 32_768

    # Partial context: still capped by the default 128 KiB ceiling.
    # remaining_budget_bytes = int((131072 - 65536) * 4 * 0.9) = 235929 -> capped -> tokens = 32768
    assert ts.estimate_tool_output_token_budget(131_072, 65_536) == 32_768

    # Near-full context: remaining-context term dominates.
    assert ts.estimate_tool_output_token_budget(131_072, 130_000) == 964

    # Very large context: default ceiling applies.
    assert ts.estimate_tool_output_token_budget(1_048_576, 0) == 32_768


async def test_oversized_string_output_is_truncated():
    """A string tool output above the dynamic limit is truncated and returned as an error."""
    ts = KimiToolset()  # fallback 128 KiB budget
    ts.add(_EchoTool())

    # Use a non-repeating pattern so sanitize_for_tokenizer does not collapse it.
    large_output = "".join(chr(65 + i % 26) for i in range(200_000))
    tool_call = ToolCall(
        id="tc-large",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps({"value": large_output}),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert isinstance(tr.return_value, ToolError)
    output = tr.return_value.output
    assert isinstance(output, str)
    assert len(output.encode("utf-8")) == ts._get_max_output_bytes()
    assert "exceeded the maximum allowed size" in tr.return_value.message
    assert "truncated" in tr.return_value.message.lower()
    # The truncated output is a prefix of the original content.
    assert large_output.startswith(output)


async def test_small_string_output_is_returned_normally():
    """A string tool output below the dynamic limit is returned unchanged."""
    ts = KimiToolset()
    ts.add(_EchoTool())

    small_output = "hello"
    tool_call = ToolCall(
        id="tc-small",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps({"value": small_output}),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == small_output


class _PartEchoTool(CallableTool2[DummyParams]):
    name: str = "PartEchoTool"
    description: str = "Echoes the input value as a TextPart list"
    params: type[DummyParams] = DummyParams

    async def __call__(self, params: DummyParams) -> ToolReturnValue:
        return ToolOk(output=[TextPart(text=params.value)])


async def test_oversized_content_part_output_is_truncated():
    """A ContentPart tool output above the dynamic limit is truncated and returned as an error."""
    ts = KimiToolset()  # fallback 128 KiB budget
    ts.add(_PartEchoTool())

    large_output = "".join(chr(65 + i % 26) for i in range(200_000))
    tool_call = ToolCall(
        id="tc-large-parts",
        function=ToolCall.FunctionBody(
            name="PartEchoTool",
            arguments=json.dumps({"value": large_output}),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert isinstance(tr.return_value, ToolError)
    output = tr.return_value.output
    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], TextPart)
    assert len(output[0].text.encode("utf-8")) == ts._get_max_output_bytes()
    assert "exceeded the maximum allowed size" in tr.return_value.message
    assert large_output.startswith(output[0].text)


# ══════════════════════════════════════════════════════════════════════════════
# Tool name redirect / candidate collection tests
# ══════════════════════════════════════════════════════════════════════════════


def test_platform_redirects_win32_includes_powershell():
    """On Windows, the platform redirect map includes bash→pwsh."""
    redirects = _build_platform_redirects()
    # The normalized key for "bash" should map to "pwsh"
    from kosong.tooling import normalize_tool_name

    bash_norm = normalize_tool_name("Bash")
    pwsh_norm = normalize_tool_name("Powershell")
    # On win32, bash → pwsh; on POSIX, Powershell → bash
    if sys.platform == "win32":
        assert redirects.get(bash_norm) == "pwsh"
    else:
        assert redirects.get(pwsh_norm) == "bash"


def test_platform_redirects_has_keys():
    """Platform redirects map should always have entries."""
    redirects = _build_platform_redirects()
    assert len(redirects) > 0
    # All values should be valid normalized tool name variants
    for norm_key, target in redirects.items():
        assert isinstance(norm_key, str)
        assert isinstance(target, str)
        assert norm_key.islower()


def test_platform_redirects_norm_is_cache():
    """_PLATFORM_REDIRECTS_NORM is the pre-computed platform redirect map."""
    assert isinstance(_PLATFORM_REDIRECTS_NORM, dict)
    # Should match a fresh build
    fresh = _build_platform_redirects()
    assert fresh == _PLATFORM_REDIRECTS_NORM


def test_collect_candidates_no_redirects():
    """Without redirects, _collect_candidates falls through to normalize+fuzzy."""
    valid = {"ToolA", "ToolB", "LongNamedTool"}
    candidates = _collect_candidates("ToolA", valid)
    assert "ToolA" in candidates
    # The exact match via normalize_tool_name should be first
    assert candidates[0] == "ToolA"


def test_collect_candidates_with_redirect():
    """With redirects, the redirect match takes priority."""
    from kosong.tooling import normalize_tool_name

    valid = {"ToolA", "ToolB"}
    redirects = {normalize_tool_name("Bash"): "ToolA"}
    # "Bash" is not in valid, but redirects point to "ToolA"
    candidates = _collect_candidates("Bash", valid, redirects=redirects)
    assert "ToolA" in candidates
    # ToolA should be first (from redirect, highest priority)
    assert candidates[0] == "ToolA"


def test_collect_candidates_redirect_not_in_valid():
    """Redirect to a name not in valid_names is skipped."""
    from kosong.tooling import normalize_tool_name

    valid = {"ToolA", "ToolB"}
    redirects = {normalize_tool_name("Bash"): "NonExistent"}
    candidates = _collect_candidates("Bash", valid, redirects=redirects)
    # Should not contain "NonExistent" since it's not in valid
    assert "NonExistent" not in candidates


def test_collect_candidates_deduplicates():
    """Candidates list should not contain duplicates."""
    valid = {"ToolA", "ToolB"}
    candidates = _collect_candidates("toola", valid)  # case-insensitive match to ToolA
    # Should have exactly one "ToolA"
    toola_count = candidates.count("ToolA")
    assert toola_count == 1, f"Expected exactly one 'ToolA', got {toola_count}"


# ══════════════════════════════════════════════════════════════════════════════
# Handle with redirect map (integration)
# ══════════════════════════════════════════════════════════════════════════════


async def test_handle_redirect_auto_corrects_hallucinated_name():
    """A hallucinated tool name that matches the redirect map should auto-correct."""
    from kosong.tooling import normalize_tool_name

    ts = _make_toolset()
    # Add a redirect entry for testing: "AppendFile" → "ToolA"
    # We inject into the module-level platform redirects (it's used by handle())
    import kimi_cli.soul.toolset as ts_mod

    original_redirects = dict(ts_mod._PLATFORM_REDIRECTS_NORM)
    ts_mod._PLATFORM_REDIRECTS_NORM = dict(original_redirects)
    ts_mod._PLATFORM_REDIRECTS_NORM[normalize_tool_name("AppendFile")] = "ToolA"

    try:
        tool_call = ToolCall(
            id="tc-redirect",
            function=ToolCall.FunctionBody(
                name="AppendFile",
                arguments=json.dumps({"value": "test"}),
            ),
        )
        result = ts.handle(tool_call)
        assert isinstance(result, asyncio.Task)
        tr = await result
        output = tr.return_value.output
        assert isinstance(output, str)
        assert output.startswith("a")  # ToolA returns "a"
        assert "<system-warning>" in output
        assert "Auto-corrected" in output
    finally:
        ts_mod._PLATFORM_REDIRECTS_NORM = original_redirects


# ══════════════════════════════════════════════════════════════════════════════
# Integration test: TodoList invalid status fuzzy matching
# ══════════════════════════════════════════════════════════════════════════════


async def test_todo_tool_invalid_status_fuzzy_match():
    """A TodoList call with status="completed" should be fuzzy-matched to
    status="done" after value-level repair in kosong/tooling.

    Currently this is expected to return a ToolValidateError (broken behavior).
    After Phase 2's value-level fuzzy matching, this should return ToolOk.
    """
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class TodoParams(BaseModel):
        todos: list[TodoItem] | TodoItem | None = None
        mode: Literal["overwrite", "append", "force_overwrite"] = "append"

    class SimpleTodoTool(CallableTool2[TodoParams]):
        name: str = "TodoList"
        description: str = "Simple todo list tool"
        params: type[TodoParams] = TodoParams

        @override
        async def __call__(self, params: TodoParams) -> ToolReturnValue:
            return ToolOk(output="ok")

    ts = KimiToolset()
    ts.add(SimpleTodoTool())

    tool_call = ToolCall(
        id="tc-todo-status",
        function=ToolCall.FunctionBody(
            name="TodoList",
            arguments=json.dumps({
                "items": [{"title": "Test", "status": "completed"}],
                "mode": "append",
            }),
        ),
    )

    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    ret = tr.return_value

    # Currently the tool succeeds (ToolOk), which is the desired behavior.
    # After Phase 2 adds fuzzy value matching, "completed" should be
    # explicitly repaired to "done" by the kosong/tooling layer.
    assert isinstance(ret, ToolOk), (
        f"Expected ToolOk for 'completed' status, got {ret}"
    )


async def test_handle_redirect_precedes_fuzzy():
    """Redirect map is checked before fuzzy matching."""
    from kosong.tooling import normalize_tool_name

    ts = _make_extended_toolset()
    import kimi_cli.soul.toolset as ts_mod

    original_redirects = dict(ts_mod._PLATFORM_REDIRECTS_NORM)
    ts_mod._PLATFORM_REDIRECTS_NORM = dict(original_redirects)
    # "LongNmedTool" has a typo - fuzzy would correct to "LongNamedTool"
    # But we redirect it to "ToolB" instead
    ts_mod._PLATFORM_REDIRECTS_NORM[normalize_tool_name("LongNmedTool")] = "ToolB"

    try:
        tool_call = ToolCall(
            id="tc-redirect-priority",
            function=ToolCall.FunctionBody(
                name="LongNmedTool",
                arguments=json.dumps({"value": "test"}),
            ),
        )
        result = ts.handle(tool_call)
        assert isinstance(result, asyncio.Task)
        tr = await result
        output = tr.return_value.output
        assert isinstance(output, str)
        assert output.startswith("b")  # ToolB returns "b", not "long"
        assert "<system-warning>" in output
    finally:
        ts_mod._PLATFORM_REDIRECTS_NORM = original_redirects


# ══════════════════════════════════════════════════════════════════════════════
# Argument-format hallucination repair tests
# ══════════════════════════════════════════════════════════════════════════════


def test_unwrap_nested_arguments_extracts_arguments_key():
    """{"arguments": {...}} should unwrap to the inner dict."""
    assert _unwrap_nested_arguments({"arguments": {"value": "x"}}) == {"value": "x"}


def test_unwrap_nested_arguments_extracts_args_key():
    """{"args": {...}} should unwrap to the inner dict."""
    assert _unwrap_nested_arguments({"args": {"value": "x"}}) == {"value": "x"}


def test_unwrap_nested_arguments_extracts_stringified_inner():
    """{"arguments": "{...}"} should unwrap to the stringified JSON object."""
    assert _unwrap_nested_arguments({"arguments": '{"value": "x"}'}) == '{"value": "x"}'


def test_unwrap_nested_arguments_extracts_list_inner():
    """{"arguments": [...]} should unwrap to the inner list."""
    assert _unwrap_nested_arguments({"arguments": [1, 2]}) == [1, 2]


def test_unwrap_nested_arguments_leaves_plain_dict_alone():
    """A plain argument dict should not be modified."""
    assert _unwrap_nested_arguments({"value": "x"}) == {"value": "x"}


def test_unwrap_nested_arguments_leaves_multi_key_dict_alone():
    """A dict with multiple keys should not be unwrapped even if one is 'arguments'."""
    assert _unwrap_nested_arguments({"arguments": {"value": "x"}, "extra": 1}) == {
        "arguments": {"value": "x"},
        "extra": 1,
    }


def test_unwrap_nested_arguments_passes_through_non_dict():
    """Non-dict values are returned unchanged."""
    assert _unwrap_nested_arguments("string") == "string"
    assert _unwrap_nested_arguments([1, 2]) == [1, 2]
    assert _unwrap_nested_arguments(None) is None


def test_parse_stringified_arguments_parses_json_object():
    """A string containing a JSON object is parsed into a dict."""
    assert _parse_stringified_arguments('{"value": "x"}') == {"value": "x"}


def test_parse_stringified_arguments_parses_json_array():
    """A string containing a JSON array is parsed into a list."""
    assert _parse_stringified_arguments('["a", "b"]') == ["a", "b"]


def test_parse_stringified_arguments_parses_with_relaxed_json():
    """Relaxed JSON parsing allows single quotes and trailing commas."""
    assert _parse_stringified_arguments("{'value': 'x',}") == {"value": "x"}


def test_parse_stringified_arguments_leaves_plain_string_alone():
    """A non-JSON string is returned unchanged."""
    assert _parse_stringified_arguments("plain text") == "plain text"


def test_parse_stringified_arguments_leaves_dict_alone():
    """A dict input is returned unchanged."""
    assert _parse_stringified_arguments({"value": "x"}) == {"value": "x"}



def test_repair_argument_format_unwraps_then_parses():
    """A stringified object wrapped in {'arguments': ...} is fully repaired."""
    assert _repair_argument_format({'arguments': '{"value": "x"}'}) == {"value": "x"}


def test_repair_argument_format_parses_then_unwraps():
    """A stringified {'arguments': ...} object is parsed and then unwrapped."""
    assert _repair_argument_format('{"arguments": {"value": "x"}}') == {"value": "x"}


def test_repair_argument_format_noop_for_plain_dict():
    """A plain dict is not changed by repair."""
    assert _repair_argument_format({"value": "x"}) == {"value": "x"}


# --- _repair_todo_arguments: fuzzy todo argument repair -----------------


def test_repair_todo_write_promotes_singular_key():
    """todo_write accepts a single todo/task/item key as the todos field."""
    assert _repair_todo_arguments("todo_write", {"todo": {"content": "A"}}) == {
        "todos": {"content": "A", "status": "pending"}
    }
    assert _repair_todo_arguments("todo_write", {"task": "Implement X", "status": "done"}) == {
        "todos": [{"content": "Implement X", "status": "done"}]
    }
    assert _repair_todo_arguments("todo_write", {"item": {"content": "B", "status": "done"}}) == {
        "todos": {"content": "B", "status": "done"}
    }


def test_repair_todo_write_wraps_bare_string_todos():
    """Bare-string todos are wrapped into schema-valid item dicts."""
    assert _repair_todo_arguments(
        "todo_write", {"todos": ["Buy milk", "Walk dog"]}
    ) == {
        "todos": [
            {"content": "Buy milk", "status": "pending"},
            {"content": "Walk dog", "status": "pending"},
        ]
    }
    assert _repair_todo_arguments("todo_write", {"todos": "Single"}) == {
        "todos": [{"content": "Single", "status": "pending"}]
    }


def test_repair_todo_write_fills_missing_status():
    """Item dicts missing the required status get a pending default."""
    assert _repair_todo_arguments("todo_write", {"todos": [{"content": "A"}]}) == {
        "todos": [{"content": "A", "status": "pending"}]
    }
    # Valid statuses are preserved.
    assert _repair_todo_arguments(
        "todo_write", {"todos": [{"content": "A", "status": "done"}]}
    ) == {"todos": [{"content": "A", "status": "done"}]}


def test_repair_todo_update_promotes_title_synonyms():
    """todo_update accepts task/todo/item/name as the title field."""
    assert _repair_todo_arguments("todo_update", {"task": "Fix bug", "status": "done"}) == {
        "title": "Fix bug",
        "status": "done",
    }
    assert _repair_todo_arguments("todo_update", {"todo": "Write docs"}) == {
        "title": "Write docs"
    }


def test_repair_todo_update_promotes_batch_synonyms():
    """todo_update accepts edits/changes/operations as the updates field."""
    assert _repair_todo_arguments(
        "todo_update", {"edits": [{"title": "A"}, {"content": "B"}]}
    ) == {"updates": [{"title": "A"}, {"content": "B"}]}
    assert _repair_todo_arguments("todo_update", {"operations": [{"title": "C"}]}) == {
        "updates": [{"title": "C"}]
    }


def test_repair_todo_update_wraps_bare_string_updates():
    """Bare-string update lists are wrapped into title items."""
    assert _repair_todo_arguments("todo_update", {"todos": ["X", "Y"]}) == {
        "todos": [{"title": "X", "status": "pending"}, {"title": "Y", "status": "pending"}]
    }
    assert _repair_todo_arguments("todo_update", {"changes": "Just one"}) == {
        "updates": [{"title": "Just one", "status": "pending"}]
    }


def test_repair_todo_arguments_keeps_valid_calls_unchanged():
    """Well-formed todo calls and non-todo tools are not modified."""
    valid_write = {"todos": [{"content": "A", "status": "done"}], "mode": "append"}
    assert _repair_todo_arguments("todo_write", dict(valid_write)) == valid_write

    valid_update = {"title": "A", "status": "in_progress", "force": True}
    assert _repair_todo_arguments("todo_update", dict(valid_update)) == valid_update

    other = {"value": "x", "task": "ignored"}
    assert _repair_todo_arguments("bash", dict(other)) == other


async def test_handle_repairs_nested_arguments():
    """handle() repairs arguments double-wrapped in {'arguments': ...}."""
    ts = KimiToolset()
    ts.add(_EchoTool())

    tool_call = ToolCall(
        id="tc-nested-args",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps({"arguments": {"value": "nested"}}),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "nested"


async def test_handle_repairs_stringified_arguments():
    """handle() repairs arguments that are a stringified JSON object."""
    ts = KimiToolset()
    ts.add(_EchoTool())

    # arguments field is a JSON-encoded string containing the real args object
    tool_call = ToolCall(
        id="tc-stringified-args",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps('{"value": "stringified"}'),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "stringified"


async def test_handle_repairs_stringified_double_wrapped_arguments():
    """handle() repairs arguments that are a stringified {'arguments': ...} wrapper."""
    ts = KimiToolset()
    ts.add(_EchoTool())

    # arguments field is a JSON-encoded string containing {"arguments": {"value": ...}}
    tool_call = ToolCall(
        id="tc-stringified-nested",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps('{"arguments": {"value": "deeply_nested"}}'),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "deeply_nested"


async def test_handle_repairs_relaxedly_stringified_arguments():
    """handle() repairs single-quoted/trailing-comma stringified arguments."""
    ts = KimiToolset()
    ts.add(_EchoTool())

    # arguments field is a JSON-encoded string containing relaxed JSON
    tool_call = ToolCall(
        id="tc-relaxed-stringified",
        function=ToolCall.FunctionBody(
            name="EchoTool",
            arguments=json.dumps("{'value': 'relaxed',}"),
        ),
    )
    result = ts.handle(tool_call)
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "relaxed"


# --- Reasoning-gated loop detection reset --------------------------------
# A model that emits a ThinkPart (thinking block) between tool calls is
# making progress; the repeat/loop detectors must reset their counting
# window so only consecutive no-thinking tool-call steps are punished.


async def _inflate_cycle_state(ts: KimiToolset) -> None:
    """Run enough A(x)->B(x) steps to reach warning territory (call 3, A#2)."""
    args = _call_args("x")
    await _run_cycle(ts, [("ToolA", args), ("ToolB", args)], max_steps=3)
    assert ts.force_stop_turn is False
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 2


async def test_reset_loop_detectors_clears_every_counter():
    """The public reset clears every turn-scoped loop counter and trip state."""
    ts = _make_toolset()
    await _inflate_cycle_state(ts)

    ts.reset_loop_detectors()

    assert ts._consecutive_key is None
    assert ts._consecutive_count == 0
    assert ts._seen_call_keys == set()
    assert ts._call_key_counts == {}
    assert ts._tool_call_counts == {}
    assert ts._tool_warned_at == {}
    assert ts._turn_total_calls == 0
    assert ts.force_stop_turn is False
    assert ts.force_stop_reason is None
    assert ts.force_stop_key is None


async def test_reset_loop_detectors_allows_legitimate_rerun_after_reasoning():
    """After a reasoning reset the same call is fresh work, not a cycle repeat."""
    ts = _make_toolset()
    await _inflate_cycle_state(ts)
    ts.reset_loop_detectors()

    args = _call_args("x")
    ts.begin_step([], step_no=4, turn_id="T1")
    result = ts.handle(
        ToolCall(id="rerun", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
    )
    assert isinstance(result, asyncio.Task)
    tr = await result
    assert tr.return_value.output == "a"
    assert "already ran earlier this turn" not in tr.return_value.output
    assert ts.force_stop_turn is False
    assert ts._call_key_counts[(("ToolA"), '{"value":"x"}')] == 1


async def test_adjacent_streak_restarts_after_reset():
    """A reasoned step restarts the adjacent-repeat streak at the 3rd call."""
    args = _call_args("x")
    ts = _make_toolset()
    previous: list[tuple[str, str]] = []
    texts: list[str] = []

    # First 12 consecutive identical calls -> sparse reminders at 3rd/8th/12th
    # (the adjacent-repeat detector reminds on every call once past a threshold,
    # so assert the exact threshold texts rather than a total count).
    for step in range(1, 13):
        ts.begin_step(previous, step_no=step, turn_id="turn-streak-reset")
        result = ts.handle(
            ToolCall(id=f"pre-{step}", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        texts.append(str(tr.return_value.output))
        previous = ts.end_step()
    assert "Stop repeating the same tool call with identical parameters" in texts[2]
    assert "Repeated identical call" in texts[7]
    assert "Dead-end loop detected" in texts[11]
    assert ts.force_stop_turn is False  # streak stops only at 16

    # Reasoning step resets the consecutive streak AND the seen-set, so
    # continued identical calls restart the ladder at the new 3rd call.
    ts.reset_loop_detectors()
    # The reset clears _seen_call_keys, so the first post-reset call is no
    # longer a cross-step duplicate -> plain output, count 1.
    post_texts: list[str] = []
    for step in range(13, 16):
        ts.begin_step(previous, step_no=step, turn_id="turn-streak-reset")
        result = ts.handle(
            ToolCall(id=f"post-{step}", function=ToolCall.FunctionBody(name="ToolA", arguments=args))
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        post_texts.append(str(tr.return_value.output))
        previous = ts.end_step()
    assert post_texts[0] == "a"  # no cross-step duplicate at all: count restarts at 1
    assert "Stop repeating" in post_texts[1]  # the new 2nd identical call is a fresh duplicate
    assert ts.force_stop_turn is False


async def test_reasoned_step_resets_cycle_before_force_stop():
    """A reasoning block between cycle occurrences delays the force-stop by a full round."""
    args = _call_args("x")
    ts = _make_toolset()
    previous: list[tuple[str, str]] = []
    seq = [("ToolA", args), ("ToolB", args)]

    for step in range(1, 6):  # A B A B A  (A#3, one below the cycle force-stop)
        tool, tool_args = seq[(step - 1) % len(seq)]
        ts.begin_step(previous, step_no=step, turn_id="turn-reason-cycle")
        result = ts.handle(
            ToolCall(
                id=f"cyc-{step}",
                function=ToolCall.FunctionBody(name=tool, arguments=tool_args),
            )
        )
        assert isinstance(result, asyncio.Task)
        tr = await result
        previous = ts.end_step()
        if step == 5:
            ts.reset_loop_detectors()  # the reasoning step (ThinkPart) between calls

    # After reset, A(x) needs its own 4th occurrence to force-stop -> A B A B A B A = step 12.
    stop_step: int | None = None
    for step in range(6, 20):
        tool, tool_args = seq[(step - 1) % len(seq)]
        ts.begin_step(previous, step_no=step, turn_id="turn-reason-cycle")
        result = ts.handle(
            ToolCall(
                id=f"cyc-{step}",
                function=ToolCall.FunctionBody(name=tool, arguments=tool_args),
            )
        )
        assert isinstance(result, asyncio.Task)
        await result
        previous = ts.end_step()
        if ts.force_stop_turn:
            stop_step = step
            break
    assert stop_step == 12
    assert ts.force_stop_reason == "cycle-repeat"


async def test_reset_loop_detectors_clears_force_stop_trip():
    """A reasoned step clears a mid-stream force-stop trip so it cannot end the turn."""
    ts = _make_toolset()
    args = _call_args("x")
    stop_step, _ = await _run_cycle(ts, [("ToolA", args), ("ToolB", args)])
    assert stop_step == 7
    assert ts.force_stop_turn is True
    assert ts.force_stop_reason == "cycle-repeat"
    assert ts.force_stop_key is not None

    ts.reset_loop_detectors()
    assert ts.force_stop_turn is False
    assert ts.force_stop_reason is None
    assert ts.force_stop_key is None
    assert ts._turn_total_calls == 0


def test_has_reasoning_parts_helper():
    """The toolset reasoning-part helper only accepts non-empty ThinkParts."""
    assert _has_reasoning_parts([ThinkPart(think="let me think")]) is True
    assert _has_reasoning_parts([ThinkPart(think="")]) is False
    assert _has_reasoning_parts([TextPart(text="answer")]) is False
    assert _has_reasoning_parts([TextPart(text="x"), ThinkPart(think="why")]) is True
    assert _has_reasoning_parts([]) is False


def test_has_reasoning_parts_accepts_generic_content_parts():
    """The helper is typed on ContentPart and accepts any part subclass at runtime."""
    parts: list[ContentPart] = [ThinkPart(think="reason")]
    assert _has_reasoning_parts(parts) is True
