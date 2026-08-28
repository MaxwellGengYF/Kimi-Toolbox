"""Localized presentation for provider-neutral LLM metadata.

The domain and provider layers expose stable machine tokens only.  Every human
label for those tokens lives here at the Qt boundary so adding a language never
changes selection, persistence, or runtime behavior.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QCoreApplication

from kimix_gui.llm import (
    AXIS_CONTEXT_WINDOW,
    AXIS_THINKING_EFFORT,
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_FILE_MISSING,
    PROBLEM_INVALID_JSON,
    PROBLEM_INVALID_PROVIDER_FILE,
    PROBLEM_INVALID_SESSION_SELECTION,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_MODEL_UNAVAILABLE,
    PROBLEM_NOT_AN_OBJECT,
    PROBLEM_NOT_JSON,
    PROBLEM_PARAMETER_UNKNOWN,
    PROBLEM_PARAMETER_UNRESOLVED,
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
    PROBLEM_PROVIDER_FILE_UNAVAILABLE,
    LLMProblem,
    ParameterOption,
    ParameterSpec,
    ResolvedLLMSelection,
)

_CONTEXT = "LLMSettingsDialog"


def _tr(text: str) -> str:
    return QCoreApplication.translate(_CONTEXT, text)


def _translation_catalog() -> tuple[str, ...]:
    """Expose helper-based copy to ``lupdate`` without translating at import time."""

    # Qt Linguist only extracts direct ``translate(context, source)`` calls.  Runtime
    # presentation still goes through ``_tr`` so one context owns this shared copy.
    return (
        QCoreApplication.translate(
            "LLMSettingsDialog", "Choose the exact value saved for this model parameter"
        ),
        QCoreApplication.translate("LLMSettingsDialog", "ChatGPT subscription"),
        QCoreApplication.translate("LLMSettingsDialog", "Provider files"),
        QCoreApplication.translate("LLMSettingsDialog", "Local OpenAI-compatible"),
        QCoreApplication.translate("LLMSettingsDialog", "ACP adapter"),
        QCoreApplication.translate("LLMSettingsDialog", "Thinking effort"),
        QCoreApplication.translate("LLMSettingsDialog", "Context window"),
        QCoreApplication.translate("LLMSettingsDialog", "Unavailable"),
        QCoreApplication.translate("LLMSettingsDialog", "API key missing"),
        QCoreApplication.translate("LLMSettingsDialog", "File missing"),
        QCoreApplication.translate("LLMSettingsDialog", "Connect ChatGPT"),
        QCoreApplication.translate("LLMSettingsDialog", "Model unavailable"),
        QCoreApplication.translate("LLMSettingsDialog", "Unknown parameter"),
        QCoreApplication.translate("LLMSettingsDialog", "Parameter value unavailable"),
        QCoreApplication.translate("LLMSettingsDialog", "Choose a parameter value"),
        QCoreApplication.translate("LLMSettingsDialog", "Selected by model parameters"),
        QCoreApplication.translate("LLMSettingsDialog", "Managed by provider"),
        QCoreApplication.translate("LLMSettingsDialog", "stream on"),
        QCoreApplication.translate("LLMSettingsDialog", "stream off"),
        QCoreApplication.translate("LLMSettingsDialog", "Not specified"),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Kimix Provider file must be JSON: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Provider file does not exist: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Invalid Provider JSON {path}: {reason}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Provider JSON must contain an object: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Invalid Kimix Provider file {path}: {reason}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Provider file is unavailable: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "No API key or OAuth credential is configured: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Invalid session LLM selection: {path}"
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "Connect ChatGPT to use this subscription model."
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "This model is not available for the connected account."
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "The saved model parameter is no longer recognized."
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog", "The saved model parameter value is no longer available."
        ),
        QCoreApplication.translate(
            "LLMSettingsDialog",
            "Choose a value for every model parameter before continuing.",
        ),
        QCoreApplication.translate("LLMSettingsDialog", "200K tokens"),
        QCoreApplication.translate("LLMSettingsDialog", "1M tokens"),
        QCoreApplication.translate("LLMSettingsDialog", "Unavailable value · {value}"),
        QCoreApplication.translate("LLMSettingsDialog", "effort {effort} · {stream}"),
        QCoreApplication.translate("LLMSettingsDialog", "{count} tokens"),
        QCoreApplication.translate("LLMSettingsDialog", "Off ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "None ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "Minimal ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "Low ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "Medium ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "High ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "Extra high ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "Maximum ({value})"),
        QCoreApplication.translate("LLMSettingsDialog", "{value} · Model default"),
        QCoreApplication.translate("LLMSettingsDialog", "not selected"),
        QCoreApplication.translate("LLMSettingsDialog", "not specified"),
    )


def provider_title(provider_id: str) -> str:
    """Return a localized title for a stable provider plugin ID."""

    known = {
        "chatgpt": _tr("ChatGPT subscription"),
        "provider_file": _tr("Provider files"),
    }
    return known.get(provider_id, provider_id)


def axis_label(axis: str) -> str:
    """Return a localized label for a stable parameter-axis token."""

    known = {
        AXIS_THINKING_EFFORT: _tr("Thinking effort"),
        AXIS_CONTEXT_WINDOW: _tr("Context window"),
    }
    return known.get(axis, axis)


def value_label(axis: str, value: str) -> str:
    """Return a localized label without changing the underlying value token."""

    if axis == AXIS_THINKING_EFFORT:
        known = {
            "off": lambda: _tr("Off ({value})"),
            "none": lambda: _tr("None ({value})"),
            "minimal": lambda: _tr("Minimal ({value})"),
            "low": lambda: _tr("Low ({value})"),
            "medium": lambda: _tr("Medium ({value})"),
            "high": lambda: _tr("High ({value})"),
            "xhigh": lambda: _tr("Extra high ({value})"),
            "max": lambda: _tr("Maximum ({value})"),
        }
        formatter = known.get(value)
        return formatter().format(value=value) if formatter is not None else value
    if axis == AXIS_CONTEXT_WINDOW:
        known_context = {
            "200k": _tr("200K tokens"),
            "1m": _tr("1M tokens"),
        }
        return known_context.get(value, value)
    return value


def parameter_option_label(parameter: ParameterSpec, option: ParameterOption) -> str:
    """Format one parameter option, including its catalog-default marker."""

    label = value_label(parameter.axis, option.value)
    if option.is_default:
        return _tr("{value} · Model default").format(value=label)
    return label


def missing_parameter_label(value: str | None) -> str:
    """Describe a persisted value absent from the current catalog."""

    return _tr("Unavailable value · {value}").format(value=value or _tr("not selected"))


def parameter_picker_description() -> str:
    return _tr("Choose the exact value saved for this model parameter")


def problem_message(problem: LLMProblem | None) -> str:
    """Translate one machine-readable LLM problem at the Qt boundary."""

    if problem is None:
        return ""
    template = _PROBLEM_TEMPLATES.get(problem.kind)
    if template is None:
        return str(problem)
    return template().format(path=problem.path or "", reason=problem.reason)


def short_problem(problem: LLMProblem | None) -> str:
    """Return the compact status label for an LLM problem."""

    if problem is None:
        return _tr("Unavailable")
    labels = {
        PROBLEM_CREDENTIAL_MISSING: _tr("API key missing"),
        PROBLEM_FILE_MISSING: _tr("File missing"),
        PROBLEM_LOGIN_REQUIRED: _tr("Connect ChatGPT"),
        PROBLEM_MODEL_UNAVAILABLE: _tr("Model unavailable"),
        PROBLEM_PARAMETER_UNKNOWN: _tr("Unknown parameter"),
        PROBLEM_PARAMETER_VALUE_UNAVAILABLE: _tr("Parameter value unavailable"),
        PROBLEM_PARAMETER_UNRESOLVED: _tr("Choose a parameter value"),
    }
    return labels.get(problem.kind, _tr("Unavailable"))


def provider_thinking_text(resolved: ResolvedLLMSelection) -> str:
    """Describe effective thinking metadata without branching on provider type."""

    parameter = next(
        (item for item in resolved.resolved if item.axis == AXIS_THINKING_EFFORT),
        None,
    )
    effort = (
        parameter.option.value
        if parameter is not None and parameter.option is not None and parameter.problem is None
        else None
    )
    stream = resolved.model.show_thinking_stream
    if parameter is not None and parameter.option is not None and stream is None:
        return _tr("Selected by model parameters")
    if effort is None and stream is None:
        return _tr("Managed by provider")
    stream_text = _tr("stream on") if stream else _tr("stream off")
    return _tr("effort {effort} · {stream}").format(
        effort=effort or _tr("not specified"),
        stream=stream_text,
    )


def format_tokens(value: int | None) -> str:
    if value is None:
        return _tr("Not specified")
    return _tr("{count} tokens").format(count=f"{value:,}")


_PROBLEM_TEMPLATES: dict[str, Callable[[], str]] = {
    PROBLEM_NOT_JSON: lambda: _tr("Kimix Provider file must be JSON: {path}"),
    PROBLEM_FILE_MISSING: lambda: _tr("Provider file does not exist: {path}"),
    PROBLEM_INVALID_JSON: lambda: _tr("Invalid Provider JSON {path}: {reason}"),
    PROBLEM_NOT_AN_OBJECT: lambda: _tr("Provider JSON must contain an object: {path}"),
    PROBLEM_INVALID_PROVIDER_FILE: lambda: _tr(
        "Invalid Kimix Provider file {path}: {reason}"
    ),
    PROBLEM_PROVIDER_FILE_UNAVAILABLE: lambda: _tr("Provider file is unavailable: {path}"),
    PROBLEM_CREDENTIAL_MISSING: lambda: _tr(
        "No API key or OAuth credential is configured: {path}"
    ),
    PROBLEM_INVALID_SESSION_SELECTION: lambda: _tr("Invalid session LLM selection: {path}"),
    PROBLEM_LOGIN_REQUIRED: lambda: _tr("Connect ChatGPT to use this subscription model."),
    PROBLEM_MODEL_UNAVAILABLE: lambda: _tr(
        "This model is not available for the connected account."
    ),
    PROBLEM_PARAMETER_UNKNOWN: lambda: _tr(
        "The saved model parameter is no longer recognized."
    ),
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE: lambda: _tr(
        "The saved model parameter value is no longer available."
    ),
    PROBLEM_PARAMETER_UNRESOLVED: lambda: _tr(
        "Choose a value for every model parameter before continuing."
    ),
}
