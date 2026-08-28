"""Immutable domain values for selecting and resolving an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from kimix_gui.llm.axes import AXIS_THINKING_EFFORT, axis_sort_key
from kimix_gui.llm.parameters import (
    EMPTY_ASSIGNMENT,
    ParameterAssignment,
    ParameterOption,
    ParameterSpec,
    RuntimeOverrides,
)

PROBLEM_NOT_JSON = "not_json"
PROBLEM_FILE_MISSING = "file_missing"
PROBLEM_INVALID_JSON = "invalid_json"
PROBLEM_NOT_AN_OBJECT = "not_an_object"
PROBLEM_INVALID_PROVIDER_FILE = "invalid_provider_file"
PROBLEM_PROVIDER_FILE_UNAVAILABLE = "provider_file_unavailable"
PROBLEM_CREDENTIAL_MISSING = "credential_missing"
PROBLEM_INVALID_SESSION_SELECTION = "invalid_session_selection"
PROBLEM_LOGIN_REQUIRED = "login_required"
PROBLEM_MODEL_UNAVAILABLE = "model_unavailable"
PROBLEM_PARAMETER_UNKNOWN = "parameter_unknown"
PROBLEM_PARAMETER_VALUE_UNAVAILABLE = "parameter_value_unavailable"
PROBLEM_PARAMETER_UNRESOLVED = "parameter_unresolved"

_PROBLEM_TEXT: dict[str, str] = {
    PROBLEM_NOT_JSON: "Kimix Provider file must be JSON: {path}",
    PROBLEM_FILE_MISSING: "Provider file does not exist: {path}",
    PROBLEM_INVALID_JSON: "Invalid Provider JSON {path}: {reason}",
    PROBLEM_NOT_AN_OBJECT: "Provider JSON must contain an object: {path}",
    PROBLEM_INVALID_PROVIDER_FILE: "Invalid Kimix Provider file {path}: {reason}",
    PROBLEM_PROVIDER_FILE_UNAVAILABLE: "Provider file is unavailable: {path}",
    PROBLEM_CREDENTIAL_MISSING: "No API key or OAuth credential is configured: {path}",
    PROBLEM_INVALID_SESSION_SELECTION: "Invalid session LLM selection: {path}",
    PROBLEM_LOGIN_REQUIRED: "Connect ChatGPT to use this subscription model.",
    PROBLEM_MODEL_UNAVAILABLE: "This model is not available for the connected account.",
    PROBLEM_PARAMETER_UNKNOWN: "The saved model parameter is not recognized.",
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE: "The saved model parameter value is no longer available.",
    PROBLEM_PARAMETER_UNRESOLVED: "Choose a model parameter before using this configuration.",
}


@dataclass(frozen=True, slots=True)
class LLMProblem:
    """Machine-readable reason a Model or LLM selection cannot be used."""

    kind: str
    path: Path | None = None
    reason: str = ""

    def __str__(self) -> str:
        template = _PROBLEM_TEXT.get(self.kind, self.kind)
        return template.format(path=self.path or "", reason=self.reason)


class LLMInspectionError(ValueError):
    """Raised when a Provider file cannot produce a Model descriptor."""

    def __init__(self, problem: LLMProblem) -> None:
        super().__init__(str(problem))
        self.problem = problem


class LLMSelectionError(RuntimeError):
    """Raised when an exact persisted selection cannot be used at runtime."""

    def __init__(self, problem: LLMProblem) -> None:
        super().__init__(str(problem))
        self.problem = problem


@runtime_checkable
class ProviderTarget(Protocol):
    """Stable structural identity owned by exactly one provider plugin."""

    @property
    def kind(self) -> str: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def key(self) -> str: ...


@dataclass(frozen=True, slots=True, init=False)
class ProviderFileTarget:
    """A user-owned Provider file and optional model override."""

    kind: Literal["provider_file"]
    path: Path
    model_override: str | None

    def __init__(
        self,
        path: Path,
        model_override: str | None = None,
        *,
        kind: Literal["provider_file"] = "provider_file",
    ) -> None:
        if kind != "provider_file":
            raise ValueError(f"Invalid Provider target kind: {kind}")
        normalized_override = model_override.strip() if model_override else None
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path.expanduser().resolve(strict=False))
        object.__setattr__(self, "model_override", normalized_override or None)

    @property
    def provider_id(self) -> str:
        return "provider_file"

    @property
    def key(self) -> str:
        return f"provider_file:{self.path}:{self.model_override or ''}"


@dataclass(frozen=True, slots=True, init=False)
class ChatGPTTarget:
    """One model served through the connected ChatGPT subscription."""

    kind: Literal["chatgpt"]
    model: str

    def __init__(
        self,
        model: str,
        *,
        kind: Literal["chatgpt"] = "chatgpt",
    ) -> None:
        if kind != "chatgpt":
            raise ValueError(f"Invalid Provider target kind: {kind}")
        normalized = model.strip()
        if not normalized:
            raise ValueError("ChatGPT model must not be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "model", normalized)

    @property
    def provider_id(self) -> str:
        return "chatgpt"

    @property
    def key(self) -> str:
        return f"chatgpt:{self.model}"


@dataclass(frozen=True, slots=True)
class LLMModelDescriptor:
    """Refreshable, secret-free metadata and parameter axes for one Provider target."""

    target: ProviderTarget
    model_id: str
    provider_type: str
    endpoint: str
    credential: str
    file_format: str
    display_name: str | None = None
    max_context_size: int | None = None
    max_tokens: int | None = None
    capabilities: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()
    priority: int = 10_000
    parameters: tuple[ParameterSpec, ...] = ()
    problem: LLMProblem | None = None
    catalog_stale: bool = False
    show_thinking_stream: bool | None = None

    def __post_init__(self) -> None:
        axes = [parameter.axis for parameter in self.parameters]
        if len(axes) != len(set(axes)):
            raise ValueError("Model descriptor has duplicate parameter axes")
        ordered = tuple(
            sorted(
                self.parameters,
                key=lambda parameter: (
                    parameter.order,
                    *axis_sort_key(parameter.axis),
                ),
            )
        )
        if ordered != self.parameters:
            object.__setattr__(self, "parameters", ordered)

    @property
    def default_assignment(self) -> ParameterAssignment:
        """Return every currently declared default as a frozen assignment."""

        return ParameterAssignment(
            (parameter.axis, parameter.default.value)
            for parameter in self.parameters
            if parameter.default is not None
        )

    @property
    def label(self) -> str:
        if self.display_name:
            return self.display_name
        if self.model_id:
            return self.model_id
        if isinstance(self.target, ProviderFileTarget):
            return self.target.path.name
        if isinstance(self.target, ChatGPTTarget):
            return self.target.model
        return self.target.key

    @property
    def available(self) -> bool:
        return self.problem is None


@dataclass(frozen=True, slots=True)
class LLMSelection:
    """A Provider target, free-form parameter assignment, and pinning state."""

    target: ProviderTarget
    parameters: ParameterAssignment = EMPTY_ASSIGNMENT
    pinned: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedParameter:
    """Resolution of one stored, defaulted, unknown, or unavailable parameter value."""

    axis: str
    spec: ParameterSpec | None
    stored_value: str | None
    option: ParameterOption | None
    problem: LLMProblem | None = None


@dataclass(frozen=True, slots=True)
class ResolvedLLMSelection:
    """An LLM selection matched against current model and parameter metadata."""

    selection: LLMSelection
    model: LLMModelDescriptor
    materialized_selection: LLMSelection
    resolved: tuple[ResolvedParameter, ...] = ()
    needs_writeback: bool = False
    problem: LLMProblem | None = None

    @property
    def available(self) -> bool:
        return (
            self.problem is None
            and self.model.available
            and all(item.option is not None and item.problem is None for item in self.resolved)
        )

    @property
    def label(self) -> str:
        return self.model.label

    @property
    def runtime(self) -> RuntimeOverrides:
        overrides = RuntimeOverrides()
        for item in self.resolved:
            if item.option is not None and item.problem is None:
                overrides = overrides.merge(item.option.overrides)
        return overrides


def target_key(target: ProviderTarget) -> str:
    """Return the stable identity used to match selections to Model metadata."""

    return target.key


def selection_key(selection: LLMSelection) -> str:
    """Return a key that distinguishes target, assignment, and deferred pinning state."""

    pinned = "pinned" if selection.pinned else "unpinned"
    return f"{target_key(selection.target)}:{selection.parameters.id}:{pinned}"


def configured_selection(target: ProviderTarget) -> LLMSelection:
    """Build the configured, no-override selection for a Provider file."""

    return LLMSelection(target)


def chatgpt_selection(model: str, effort: str) -> LLMSelection:
    """Build an explicitly pinned ChatGPT thinking-effort selection."""

    return LLMSelection(
        ChatGPTTarget(model),
        ParameterAssignment({AXIS_THINKING_EFFORT: effort}),
    )


def resolve_selection(
    selection: LLMSelection,
    models: tuple[LLMModelDescriptor, ...] | list[LLMModelDescriptor],
) -> ResolvedLLMSelection:
    """Resolve each axis without discarding unavailable or unknown stored tokens."""

    key = target_key(selection.target)
    model = next((candidate for candidate in models if target_key(candidate.target) == key), None)
    if model is None:
        model = unavailable_model(selection.target, PROBLEM_MODEL_UNAVAILABLE)
    materialized = selection.parameters
    resolved: list[ResolvedParameter] = []
    parameter_problem: LLMProblem | None = None
    known_axes = {spec.axis for spec in model.parameters}
    for spec in model.parameters:
        stored_value = selection.parameters.get(spec.axis)
        option = spec.option(stored_value) if stored_value is not None else spec.default
        if stored_value is None and option is not None:
            materialized = materialized.with_value(spec.axis, option.value)
        item_problem: LLMProblem | None
        if option is None:
            item_problem = LLMProblem(
                PROBLEM_PARAMETER_VALUE_UNAVAILABLE
                if stored_value is not None
                else PROBLEM_PARAMETER_UNRESOLVED,
                reason=f"{spec.axis}={stored_value}" if stored_value is not None else spec.axis,
            )
        else:
            item_problem = option.problem
        if parameter_problem is None and item_problem is not None:
            parameter_problem = item_problem
        resolved.append(
            ResolvedParameter(spec.axis, spec, stored_value, option, item_problem)
        )

    for axis, value in selection.parameters.entries:
        if axis in known_axes:
            continue
        unknown_problem = LLMProblem(PROBLEM_PARAMETER_UNKNOWN, reason=f"{axis}={value}")
        if parameter_problem is None:
            parameter_problem = unknown_problem
        resolved.append(ResolvedParameter(axis, None, value, None, unknown_problem))

    pin_problem: LLMProblem | None = None
    materialized_pinned = selection.pinned
    if not selection.pinned:
        if model.catalog_stale:
            pin_problem = LLMProblem(PROBLEM_PARAMETER_UNRESOLVED, reason="catalog_stale")
        elif model.problem is None and parameter_problem is None:
            materialized_pinned = True
    selection_problem = model.problem or parameter_problem or pin_problem
    materialized_selection = LLMSelection(selection.target, materialized, materialized_pinned)
    needs_writeback = (
        selection_problem is None
        and not model.catalog_stale
        and materialized_selection != selection
    )
    return ResolvedLLMSelection(
        selection=selection,
        model=model,
        materialized_selection=materialized_selection,
        resolved=tuple(resolved),
        needs_writeback=needs_writeback,
        problem=selection_problem,
    )


def unavailable_model(target: ProviderTarget, kind: str, *, reason: str = "") -> LLMModelDescriptor:
    """Build a displayable placeholder for an unavailable Provider target."""

    if isinstance(target, ProviderFileTarget):
        path = target.path
        model_id = target.model_override or "Configuration unavailable"
    elif isinstance(target, ChatGPTTarget):
        path = None
        model_id = target.model
    else:
        path = None
        model_id = target.key
    return LLMModelDescriptor(
        target=target,
        model_id=model_id,
        provider_type="Unavailable",
        endpoint="Unavailable",
        credential="Unavailable",
        file_format="JSON" if path else "Built-in",
        problem=LLMProblem(kind, path, reason),
    )
