"""Stable IDs, ordering, and pure derivation for built-in LLM parameter axes."""

from __future__ import annotations

from collections.abc import Iterable

import regex as re

from kimix_gui.llm.parameters import ParameterOption, ParameterSpec, RuntimeOverrides

AXIS_THINKING_EFFORT = "thinking_effort"
AXIS_CONTEXT_WINDOW = "context_window"

ORDER_THINKING_EFFORT = 10
ORDER_CONTEXT_WINDOW = 20

CONTEXT_WINDOW_STANDARD = "200k"
CONTEXT_WINDOW_1M = "1m"
CONTEXT_1M_BETA = "context-1m-2025-08-07"

# Claude's 1M context window is a beta capability. Keyword matching, rather
# than exact slugs, covers aliases such as ``-latest``, dated revisions, and
# platform prefixes without claiming support for unrelated model families.
_ANTHROPIC_1M_KEYWORDS = (
    ("claude", "sonnet", "4"),
    ("claude", "opus", "4"),
)
_MODEL_TOKEN = re.compile(r"[a-z0-9]+")

_AXIS_ORDER = {
    AXIS_THINKING_EFFORT: ORDER_THINKING_EFFORT,
    AXIS_CONTEXT_WINDOW: ORDER_CONTEXT_WINDOW,
}

_OPTION_ORDER = {
    AXIS_THINKING_EFFORT: {
        "off": 0,
        "none": 10,
        "minimal": 20,
        "low": 30,
        "medium": 40,
        "high": 50,
        "xhigh": 60,
        "max": 70,
    },
    AXIS_CONTEXT_WINDOW: {
        CONTEXT_WINDOW_STANDARD: 10,
        CONTEXT_WINDOW_1M: 20,
    },
}


def axis_sort_key(axis: str) -> tuple[int, str]:
    """Return a stable order for known and future parameter axes."""

    return (_AXIS_ORDER.get(axis, 10_000), axis)


def option_sort_key(axis: str, value: str) -> tuple[int, str]:
    """Return a stable semantic order for built-in values, then lexical order."""

    return (_OPTION_ORDER.get(axis, {}).get(value, 10_000), value)


def provider_thinking_parameter(
    capabilities: Iterable[str],
    supported_efforts: Iterable[str],
    configured_effort: str | None,
    *,
    thinking_enabled: bool,
) -> ParameterSpec | None:
    """Derive the effort axis only when model metadata proves support.

    ``always_thinking`` models cannot expose ``off``. Toggleable thinking
    models do expose it, and the default mirrors the Provider file's effective
    startup mode. Unknown effort tokens are ignored because the core provider
    contract currently accepts only the values present in ``supported_efforts``.
    """

    capability_set = set(capabilities)
    always_thinking = "always_thinking" in capability_set
    if "thinking" not in capability_set and not always_thinking:
        return None

    efforts = set(supported_efforts)
    values = [
        value
        for value in _OPTION_ORDER[AXIS_THINKING_EFFORT]
        if value != "off" and value in efforts
    ]
    if not values:
        return None
    if not always_thinking:
        values.insert(0, "off")

    if configured_effort in values:
        default_value = configured_effort
    elif not always_thinking and not thinking_enabled:
        default_value = "off"
    elif "max" in values:
        default_value = "max"
    else:
        default_value = values[-1]

    return ParameterSpec(
        AXIS_THINKING_EFFORT,
        tuple(
            ParameterOption(
                value,
                RuntimeOverrides(thinking_effort=value),
                is_default=value == default_value,
            )
            for value in values
        ),
        order=ORDER_THINKING_EFFORT,
    )


def context_window_parameter(
    provider_type: str,
    model_id: str,
    max_context_size: int | None,
) -> ParameterSpec | None:
    """Derive the opt-in Claude 1M context axis from verified model metadata."""

    if provider_type.casefold() != "anthropic" or max_context_size != 200_000:
        return None
    tokens = set(_MODEL_TOKEN.findall(model_id.casefold()))
    if not any(set(keywords) <= tokens for keywords in _ANTHROPIC_1M_KEYWORDS):
        return None
    return ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (
            ParameterOption(CONTEXT_WINDOW_STANDARD, is_default=True),
            ParameterOption(
                CONTEXT_WINDOW_1M,
                RuntimeOverrides(
                    max_context_size=1_000_000,
                    beta_features=(CONTEXT_1M_BETA,),
                ),
            ),
        ),
        order=ORDER_CONTEXT_WINDOW,
    )
