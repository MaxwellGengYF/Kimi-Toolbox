"""Provider-independent LLM parameter specifications and frozen assignments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kimix_gui.llm.domain import LLMProblem

_MAX_TOKEN_LENGTH = 64


def _parameter_token(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string token")
    if (
        not value
        or len(value) > _MAX_TOKEN_LENGTH
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{label} must be non-empty, contain no whitespace, and be at most "
            f"{_MAX_TOKEN_LENGTH} characters"
        )
    return value


@dataclass(frozen=True, slots=True)
class RuntimeOverrides:
    """Provider-independent runtime changes contributed by one parameter value."""

    thinking_effort: str | None = None
    max_context_size: int | None = None
    max_tokens: int | None = None
    beta_features: tuple[str, ...] = ()
    generation_kwargs: tuple[tuple[str, object], ...] = ()

    def merge(self, other: RuntimeOverrides) -> RuntimeOverrides:
        """Merge in ``other`` with deterministic later-value precedence."""

        beta_features = tuple(dict.fromkeys((*self.beta_features, *other.beta_features)))
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(other.generation_kwargs)
        return RuntimeOverrides(
            thinking_effort=(
                other.thinking_effort
                if other.thinking_effort is not None
                else self.thinking_effort
            ),
            max_context_size=(
                other.max_context_size
                if other.max_context_size is not None
                else self.max_context_size
            ),
            max_tokens=other.max_tokens if other.max_tokens is not None else self.max_tokens,
            beta_features=beta_features,
            generation_kwargs=tuple(generation_kwargs.items()),
        )


@dataclass(frozen=True, slots=True)
class ParameterOption:
    """One stable token exposed on a parameter axis."""

    value: str
    overrides: RuntimeOverrides = RuntimeOverrides()
    is_default: bool = False
    problem: LLMProblem | None = None

    def __post_init__(self) -> None:
        _parameter_token(self.value, label="Parameter value")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """The current catalog specification for one free-form parameter axis."""

    axis: str
    options: tuple[ParameterOption, ...]
    order: int = 100

    def __post_init__(self) -> None:
        _parameter_token(self.axis, label="Parameter axis")
        values = [option.value for option in self.options]
        if len(values) != len(set(values)):
            raise ValueError(f"Parameter axis {self.axis!r} has duplicate values")
        if sum(option.is_default for option in self.options) > 1:
            raise ValueError(f"Parameter axis {self.axis!r} has multiple defaults")

    @property
    def default(self) -> ParameterOption | None:
        return next((option for option in self.options if option.is_default), None)

    def option(self, value: str) -> ParameterOption | None:
        return next((option for option in self.options if option.value == value), None)


@dataclass(frozen=True, slots=True, init=False)
class ParameterAssignment:
    """A frozen, hashable, axis-sorted mapping from parameter axes to values."""

    entries: tuple[tuple[str, str], ...]

    def __init__(
        self,
        entries: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    ) -> None:
        items = entries.items() if isinstance(entries, Mapping) else entries
        normalized: dict[str, str] = {}
        for raw_axis, raw_value in items:
            axis = _parameter_token(raw_axis, label="Parameter axis")
            value = _parameter_token(raw_value, label=f"Value for parameter axis {axis!r}")
            if axis in normalized:
                raise ValueError(f"Duplicate parameter axis: {axis}")
            normalized[axis] = value
        object.__setattr__(self, "entries", tuple(sorted(normalized.items())))

    def get(self, axis: str) -> str | None:
        return next((value for candidate, value in self.entries if candidate == axis), None)

    def with_value(self, axis: str, value: str) -> ParameterAssignment:
        updated = dict(self.entries)
        updated[axis] = value
        return ParameterAssignment(updated)

    def without(self, axis: str) -> ParameterAssignment:
        return ParameterAssignment((key, value) for key, value in self.entries if key != axis)

    @property
    def id(self) -> str:
        return ";".join(f"{axis}={value}" for axis, value in self.entries)


EMPTY_ASSIGNMENT = ParameterAssignment()
