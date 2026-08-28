from __future__ import annotations

from kimix_gui.llm import (
    AXIS_CONTEXT_WINDOW,
    AXIS_THINKING_EFFORT,
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_PARAMETER_UNKNOWN,
    PROBLEM_PARAMETER_UNRESOLVED,
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
    ChatGPTTarget,
    LLMModelDescriptor,
    LLMProblem,
    LLMSelection,
    ParameterAssignment,
    ParameterOption,
    ParameterSpec,
    RuntimeOverrides,
    resolve_selection,
)


def model_with(*parameters: ParameterSpec, problem: LLMProblem | None = None) -> LLMModelDescriptor:
    return LLMModelDescriptor(
        target=ChatGPTTarget("gpt-test"),
        model_id="gpt-test",
        provider_type="test",
        endpoint="memory://test",
        credential="None",
        file_format="Test",
        parameters=parameters,
        problem=problem,
    )


def test_parameter_assignment_is_frozen_hashable_sorted_and_stable() -> None:
    first = ParameterAssignment(
        {
            AXIS_THINKING_EFFORT: "high",
            AXIS_CONTEXT_WINDOW: "1m",
        }
    )
    second = ParameterAssignment(reversed(first.entries))

    assert first.entries == (
        (AXIS_CONTEXT_WINDOW, "1m"),
        (AXIS_THINKING_EFFORT, "high"),
    )
    assert first == second
    assert hash(first) == hash(second)
    assert first.id == "context_window=1m;thinking_effort=high"
    assert first.with_value(AXIS_THINKING_EFFORT, "low").get(
        AXIS_THINKING_EFFORT
    ) == "low"
    assert first.without(AXIS_CONTEXT_WINDOW) == ParameterAssignment(
        {AXIS_THINKING_EFFORT: "high"}
    )


def test_runtime_overrides_merge_with_later_precedence_and_stable_deduplication() -> None:
    earlier = RuntimeOverrides(
        thinking_effort="low",
        max_context_size=200_000,
        beta_features=("existing", "shared"),
        generation_kwargs=(("temperature", 0.1), ("top_p", 0.8)),
    )
    later = RuntimeOverrides(
        thinking_effort="high",
        max_tokens=32_000,
        beta_features=("shared", "context-1m"),
        generation_kwargs=(("temperature", 0.2),),
    )

    merged = earlier.merge(later)

    assert merged.thinking_effort == "high"
    assert merged.max_context_size == 200_000
    assert merged.max_tokens == 32_000
    assert merged.beta_features == ("existing", "shared", "context-1m")
    assert merged.generation_kwargs == (("temperature", 0.2), ("top_p", 0.8))


def test_resolve_selection_materializes_a_missing_axis_default_and_requests_writeback() -> None:
    spec = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption("low", RuntimeOverrides(thinking_effort="low")),
            ParameterOption(
                "high",
                RuntimeOverrides(thinking_effort="high"),
                is_default=True,
            ),
        ),
    )
    selection = LLMSelection(ChatGPTTarget("gpt-test"))

    resolved = resolve_selection(selection, [model_with(spec)])

    assert resolved.available
    assert resolved.needs_writeback
    assert resolved.materialized_selection.parameters == ParameterAssignment(
        {AXIS_THINKING_EFFORT: "high"}
    )
    assert resolved.resolved[0].stored_value is None
    assert resolved.resolved[0].option is spec.default
    assert resolved.runtime.thinking_effort == "high"


def test_resolve_selection_retains_unknown_axis_and_removed_value_tokens() -> None:
    spec = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (ParameterOption("low", is_default=True),),
    )
    selection = LLMSelection(
        ChatGPTTarget("gpt-test"),
        ParameterAssignment(
            {
                AXIS_THINKING_EFFORT: "retired",
                "future_axis": "future_value",
            }
        ),
    )

    resolved = resolve_selection(selection, [model_with(spec)])

    assert not resolved.available
    assert resolved.materialized_selection == selection
    assert resolved.resolved[0].problem is not None
    assert resolved.resolved[0].problem.kind == PROBLEM_PARAMETER_VALUE_UNAVAILABLE
    assert resolved.resolved[0].stored_value == "retired"
    assert resolved.resolved[1].problem is not None
    assert resolved.resolved[1].problem.kind == PROBLEM_PARAMETER_UNKNOWN
    assert resolved.resolved[1].stored_value == "future_value"


def test_resolve_selection_propagates_option_problem_and_no_default_problem() -> None:
    blocked = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption(
                "high",
                problem=LLMProblem(PROBLEM_PARAMETER_VALUE_UNAVAILABLE, reason="account"),
            ),
        ),
    )
    unresolved = ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (ParameterOption("200k"), ParameterOption("1m")),
        order=20,
    )

    blocked_result = resolve_selection(
        LLMSelection(
            ChatGPTTarget("gpt-test"),
            ParameterAssignment({AXIS_THINKING_EFFORT: "high"}),
        ),
        [model_with(blocked)],
    )
    unresolved_result = resolve_selection(
        LLMSelection(ChatGPTTarget("gpt-test")),
        [model_with(unresolved)],
    )

    assert blocked_result.problem == blocked.options[0].problem
    assert unresolved_result.problem is not None
    assert unresolved_result.problem.kind == PROBLEM_PARAMETER_UNRESOLVED
    assert unresolved_result.resolved[0].stored_value is None


def test_stale_catalog_never_writes_a_materialized_default() -> None:
    spec = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (ParameterOption("high", is_default=True),),
    )
    model = model_with(spec)
    stale_model = LLMModelDescriptor(
        target=model.target,
        model_id=model.model_id,
        provider_type=model.provider_type,
        endpoint=model.endpoint,
        credential=model.credential,
        file_format=model.file_format,
        parameters=model.parameters,
        catalog_stale=True,
    )

    resolved = resolve_selection(LLMSelection(model.target), [stale_model])

    assert resolved.available
    assert not resolved.needs_writeback
    assert resolved.materialized_selection.parameters.get(AXIS_THINKING_EFFORT) == "high"


def test_model_problem_still_resolves_saved_axes_for_display_without_writeback() -> None:
    selection = LLMSelection(
        ChatGPTTarget("gpt-test"),
        ParameterAssignment({"future_axis": "future_value"}),
        pinned=False,
    )
    problem = LLMProblem(PROBLEM_CREDENTIAL_MISSING)

    resolved = resolve_selection(selection, [model_with(problem=problem)])

    assert resolved.problem is problem
    assert not resolved.available
    assert not resolved.needs_writeback
    assert resolved.materialized_selection == selection
    assert resolved.resolved[0].axis == "future_axis"
    assert resolved.resolved[0].stored_value == "future_value"
    assert resolved.resolved[0].problem is not None
    assert resolved.resolved[0].problem.kind == PROBLEM_PARAMETER_UNKNOWN


def test_model_preserves_provider_option_order_while_canonicalizing_axis_order() -> None:
    context = ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (ParameterOption("1m"), ParameterOption("200k", is_default=True)),
        order=20,
    )
    thinking = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption("high"),
            ParameterOption("low", is_default=True),
            ParameterOption("max"),
        ),
        order=10,
    )

    model = model_with(context, thinking)

    assert [parameter.axis for parameter in model.parameters] == [
        AXIS_THINKING_EFFORT,
        AXIS_CONTEXT_WINDOW,
    ]
    assert [option.value for option in model.parameters[0].options] == [
        "high",
        "low",
        "max",
    ]
