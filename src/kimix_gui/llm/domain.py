"""Immutable domain values for selecting and resolving an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
PROBLEM_VARIANT_UNAVAILABLE = "variant_unavailable"
PROBLEM_VARIANT_UNRESOLVED = "variant_unresolved"

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
    PROBLEM_VARIANT_UNAVAILABLE: "The saved model variant is no longer available.",
    PROBLEM_VARIANT_UNRESOLVED: "Choose a model variant before using this configuration.",
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


ProviderTarget = ProviderFileTarget | ChatGPTTarget

VariantKind = Literal[
    "configured",
    "provider_default",
    "reasoning_effort",
    "legacy_default",
]


@dataclass(frozen=True, slots=True, init=False)
class LLMVariantKey:
    """Stable, secret-free identity of one executable Model Variant."""

    kind: VariantKind
    value: str | None

    def __init__(self, kind: VariantKind, value: str | None = None) -> None:
        if kind not in {
            "configured",
            "provider_default",
            "reasoning_effort",
            "legacy_default",
        }:
            raise ValueError(f"Unknown Variant kind: {kind}")
        normalized = value.strip() if value else None
        if kind == "reasoning_effort":
            if (
                normalized is None
                or len(normalized) > 64
                or any(character.isspace() for character in normalized)
            ):
                raise ValueError("Reasoning effort must be a non-empty token")
        elif normalized is not None:
            raise ValueError(f"Variant {kind!r} does not accept a value")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", normalized)

    @property
    def id(self) -> str:
        return f"{self.kind}/{self.value}" if self.value is not None else self.kind


CONFIGURED_VARIANT = LLMVariantKey("configured")
PROVIDER_DEFAULT_VARIANT = LLMVariantKey("provider_default")
LEGACY_DEFAULT_VARIANT = LLMVariantKey("legacy_default")


def reasoning_effort_variant(effort: str) -> LLMVariantKey:
    return LLMVariantKey("reasoning_effort", effort)


@dataclass(frozen=True, slots=True)
class LLMRuntimeOptions:
    """Runtime parameters represented by a Variant."""

    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class LLMVariantDescriptor:
    """Current catalog metadata for one Model Variant."""

    key: LLMVariantKey
    options: LLMRuntimeOptions = LLMRuntimeOptions()
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class LLMModelDescriptor:
    """Refreshable, secret-free metadata for one Provider target."""

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
    variants: tuple[LLMVariantDescriptor, ...] = ()
    problem: LLMProblem | None = None
    catalog_stale: bool = False
    configured_reasoning_effort: str | None = None
    show_thinking_stream: bool | None = None

    @property
    def label(self) -> str:
        if self.display_name:
            return self.display_name
        if self.model_id:
            return self.model_id
        if isinstance(self.target, ProviderFileTarget):
            return self.target.path.name
        return self.target.model

    @property
    def available(self) -> bool:
        return self.problem is None

    @property
    def default_variant(self) -> LLMVariantDescriptor | None:
        return next((variant for variant in self.variants if variant.is_default), None)


@dataclass(frozen=True, slots=True)
class LLMSelection:
    """Exact Provider target and Variant stored for a project or session."""

    target: ProviderTarget
    variant: LLMVariantKey

    def __post_init__(self) -> None:
        if isinstance(self.target, ProviderFileTarget):
            if self.variant != CONFIGURED_VARIANT:
                raise ValueError("Provider files only support the configured Variant")
            return
        if self.variant.kind not in {
            "reasoning_effort",
            "provider_default",
            "legacy_default",
        }:
            raise ValueError(f"Invalid ChatGPT Variant: {self.variant.id}")


@dataclass(frozen=True, slots=True)
class ResolvedLLMSelection:
    """An LLM selection matched against current Model and Variant metadata."""

    selection: LLMSelection
    model: LLMModelDescriptor
    variant: LLMVariantDescriptor | None
    problem: LLMProblem | None = None

    @property
    def available(self) -> bool:
        return self.problem is None and self.model.available and self.variant is not None

    @property
    def label(self) -> str:
        return self.model.label


def target_key(target: ProviderTarget) -> str:
    """Return the stable identity used to match selections to Model metadata."""

    if isinstance(target, ChatGPTTarget):
        return f"chatgpt:{target.model}"
    override = target.model_override or ""
    return f"provider_file:{target.path}:{override}"


def selection_key(selection: LLMSelection) -> str:
    return f"{target_key(selection.target)}:{selection.variant.id}"


def configured_selection(target: ProviderFileTarget) -> LLMSelection:
    return LLMSelection(target, CONFIGURED_VARIANT)


def chatgpt_selection(model: str, effort: str) -> LLMSelection:
    return LLMSelection(ChatGPTTarget(model), reasoning_effort_variant(effort))


def resolve_selection(
    selection: LLMSelection,
    models: tuple[LLMModelDescriptor, ...] | list[LLMModelDescriptor],
) -> ResolvedLLMSelection:
    """Resolve without mutating or silently replacing the stored Variant."""

    key = target_key(selection.target)
    model = next((candidate for candidate in models if target_key(candidate.target) == key), None)
    if model is None:
        model = unavailable_model(selection.target, PROBLEM_MODEL_UNAVAILABLE)
    if model.problem is not None:
        return ResolvedLLMSelection(selection, model, None, model.problem)
    if selection.variant == LEGACY_DEFAULT_VARIANT:
        return ResolvedLLMSelection(
            selection,
            model,
            None,
            LLMProblem(PROBLEM_VARIANT_UNRESOLVED),
        )
    variant = next(
        (candidate for candidate in model.variants if candidate.key == selection.variant),
        None,
    )
    if variant is None:
        return ResolvedLLMSelection(
            selection,
            model,
            None,
            LLMProblem(PROBLEM_VARIANT_UNAVAILABLE),
        )
    return ResolvedLLMSelection(selection, model, variant)


def pin_legacy_default(
    selection: LLMSelection,
    model: LLMModelDescriptor,
) -> LLMSelection | None:
    """Replace a legacy default marker with the catalog's exact default Variant."""

    if selection.variant != LEGACY_DEFAULT_VARIANT:
        return selection
    default_variant = model.default_variant
    if default_variant is None:
        return None
    return LLMSelection(selection.target, default_variant.key)


def unavailable_model(target: ProviderTarget, kind: str, *, reason: str = "") -> LLMModelDescriptor:
    """Build a displayable placeholder for an unavailable Provider target."""

    if isinstance(target, ProviderFileTarget):
        path = target.path
        model_id = target.model_override or "Configuration unavailable"
    else:
        path = None
        model_id = target.model
    return LLMModelDescriptor(
        target=target,
        model_id=model_id,
        provider_type="Unavailable",
        endpoint="Unavailable",
        credential="Unavailable",
        file_format="JSON" if path else "Built-in",
        variants=((LLMVariantDescriptor(CONFIGURED_VARIANT, is_default=True),) if path else ()),
        problem=LLMProblem(kind, path, reason),
    )
