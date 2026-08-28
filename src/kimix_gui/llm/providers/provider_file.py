"""Inspect and run user-owned Kimix Provider files without exposing secrets."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import orjson
from kimi_cli.llm import create_llm
from kosong.chat_provider import ChatProvider
from kosong.providers import get_provider_profile

import kimix
from kimi_agent_sdk import Config
from kimix.utils.config import (
    _create_config,
    _inherit_sub_provider_defaults,
    _pick_main_from_sub_providers,
)
from kimix_gui.llm.axes import context_window_parameter, provider_thinking_parameter
from kimix_gui.llm.domain import (
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_FILE_MISSING,
    PROBLEM_INVALID_JSON,
    PROBLEM_INVALID_PROVIDER_FILE,
    PROBLEM_NOT_AN_OBJECT,
    PROBLEM_NOT_JSON,
    LLMInspectionError,
    LLMModelDescriptor,
    LLMProblem,
    ProviderFileTarget,
    ProviderTarget,
    ResolvedLLMSelection,
    configured_selection,
    resolve_selection,
    unavailable_model,
)
from kimix_gui.llm.parameters import ParameterSpec, RuntimeOverrides
from kimix_gui.llm.providers.base import (
    CatalogContext,
    ProviderKind,
    SessionRuntime,
    apply_overrides,
)


def default_provider_file_path() -> Path:
    """Return Kimix's package-level default Provider file path."""

    package_file = cast(str | None, getattr(kimix, "__file__", None))
    if package_file is None:
        return Path("default_config.json").resolve(strict=False)
    return (Path(package_file).resolve().parent / "default_config.json").resolve(strict=False)


def inspect_provider_file(
    path: Path,
    *,
    model_override: str | None = None,
) -> LLMModelDescriptor:
    """Validate one Provider file and return secret-free model metadata."""

    target = ProviderFileTarget(path, model_override)
    resolved_path, provider_dict = _load_provider_mapping(target.path)
    preview_dict = dict(provider_dict)
    preview_dict.pop("env", None)
    if target.model_override is not None:
        preview_dict["model"] = target.model_override
    # Shape inspection must neither resolve live credentials nor emit the runtime's
    # missing-key warning. Credential state is represented by ``LLMProblem`` below.
    if not preview_dict.get("api_key"):
        preview_dict["api_key"] = "kimix-gui-provider-inspection"
    config = _build_sdk_config(preview_dict, resolved_path)
    model = config.model
    provider = config.provider
    if model is None or provider is None:
        missing = "model" if model is None else "provider"
        raise LLMInspectionError(
            LLMProblem(PROBLEM_INVALID_PROVIDER_FILE, resolved_path, f"no {missing} in config")
        )

    profile_name = provider_dict.get("name")
    display_name = str(profile_name) if profile_name else model.display_name
    provider_type = str(provider.type)
    capabilities = tuple(sorted(str(item) for item in (model.capabilities or ())))
    endpoint = _redact_url(provider.base_url) if provider.base_url else "Not set in file"
    problem: LLMProblem | None = None
    if provider_dict.get("oauth"):
        credential = "OAuth"
    elif provider_dict.get("api_key"):
        credential = "API key configured"
    elif _has_environment_credential(provider_dict, provider_type):
        credential = "Environment"
    else:
        credential = "Not stored in file"
        problem = LLMProblem(PROBLEM_CREDENTIAL_MISSING, resolved_path)

    return LLMModelDescriptor(
        target=target,
        model_id=target.model_override or model.model,
        display_name=display_name,
        provider_type=provider_type,
        endpoint=endpoint,
        credential=credential,
        file_format="JSON",
        max_context_size=model.max_context_size,
        max_tokens=config.max_tokens,
        capabilities=capabilities,
        parameters=provider_file_parameters(config),
        problem=problem,
        show_thinking_stream=(
            bool(provider_dict["show_thinking_stream"])
            if provider_dict.get("show_thinking_stream") is not None
            else None
        ),
    )


def provider_file_model(target: ProviderFileTarget) -> LLMModelDescriptor:
    """Inspect a target, retaining its exact problem as displayable metadata."""

    try:
        return inspect_provider_file(target.path, model_override=target.model_override)
    except LLMInspectionError as exc:
        return replace(
            unavailable_model(target, exc.problem.kind, reason=exc.problem.reason),
            problem=exc.problem,
        )


def resolved_provider_file(
    path: Path,
    *,
    model_override: str | None = None,
) -> ResolvedLLMSelection:
    """Resolve a Provider file's configured selection, including failures."""

    target = ProviderFileTarget(path, model_override)
    model = provider_file_model(target)
    return resolve_selection(configured_selection(target), [model])


def load_provider_mapping(path: Path) -> dict[str, Any]:
    """Load the normalized Provider mapping consumed by Kimix sessions."""

    _resolved, provider_dict = _load_provider_mapping(path)
    return provider_dict


def load_sdk_config(path: Path) -> Config:
    """Load a Provider file as the SDK Config used for a session."""

    resolved, provider_dict = _load_provider_mapping(path)
    return _build_sdk_config(provider_dict, resolved)


class ProviderFileProviderKind(ProviderKind):
    """Provider implementation for one explicit Kimix Provider JSON target."""

    id = "provider_file"
    catalog_managed = False

    def owns(self, target: ProviderTarget) -> bool:
        return isinstance(target, ProviderFileTarget)

    def describe(
        self,
        target: ProviderTarget,
        context: CatalogContext,
    ) -> LLMModelDescriptor:
        del context
        if not isinstance(target, ProviderFileTarget):
            raise TypeError(f"Unsupported Provider-file target: {target!r}")
        return provider_file_model(target)

    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]:
        del context
        return ()

    async def create_runtime(
        self,
        target: ProviderTarget,
        *,
        session_id: str,
        overrides: RuntimeOverrides,
    ) -> SessionRuntime:
        if not isinstance(target, ProviderFileTarget):
            raise TypeError(f"Unsupported Provider-file target: {target!r}")
        resolved_path, provider_dict = _load_provider_mapping(target.path)
        if target.model_override is not None:
            provider_dict["model"] = target.model_override
        config = _build_sdk_config(provider_dict, resolved_path)
        model = config.model
        provider_config = config.provider
        if model is None or provider_config is None:
            missing = "model" if model is None else "provider"
            raise LLMInspectionError(
                LLMProblem(
                    PROBLEM_INVALID_PROVIDER_FILE,
                    resolved_path,
                    f"no {missing} in config",
                )
            )
        provider: ChatProvider | None = None
        needs_provider_override = bool(
            overrides.thinking_effort is not None
            or overrides.beta_features
            or overrides.generation_kwargs
        )
        if needs_provider_override:
            llm = create_llm(
                provider_config,
                model,
                thinking=config.default_thinking,
                session_id=session_id,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                thinking_effort=config.thinking_effort,
            )
            if llm is not None and llm.chat_provider is not None:
                provider = llm.chat_provider
        runtime = SessionRuntime(
            provider_dict=provider_dict,
            model=target.model_override,
            provider=provider,
        )
        return apply_overrides(runtime, overrides)


def provider_file_parameters(config: Config) -> tuple[ParameterSpec, ...]:
    """Derive every adjustable axis proved by one normalized Provider config."""

    model = config.model
    provider = config.provider
    if model is None or provider is None:
        return ()
    capabilities = set(model.capabilities or ())
    candidates = (
        provider_thinking_parameter(
            capabilities,
            model.supported_efforts,
            config.thinking_effort,
            thinking_enabled=config.default_thinking,
        ),
        context_window_parameter(
            str(provider.type),
            model.model,
            model.max_context_size,
        ),
    )
    return tuple(parameter for parameter in candidates if parameter is not None)


def _load_provider_mapping(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.suffix.lower() != ".json":
        raise LLMInspectionError(LLMProblem(PROBLEM_NOT_JSON, resolved))
    if not resolved.is_file():
        raise LLMInspectionError(LLMProblem(PROBLEM_FILE_MISSING, resolved))
    try:
        loaded = orjson.loads(resolved.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise LLMInspectionError(LLMProblem(PROBLEM_INVALID_JSON, resolved, str(exc))) from exc
    if not isinstance(loaded, dict):
        raise LLMInspectionError(LLMProblem(PROBLEM_NOT_AN_OBJECT, resolved))

    provider_dict = deepcopy(loaded)
    sub_provider = provider_dict.pop("sub_provider", None)
    sub_providers = provider_dict.pop("sub_providers", None)
    _inherit_sub_provider_defaults(provider_dict, sub_provider, sub_providers)
    _pick_main_from_sub_providers(provider_dict, sub_provider, sub_providers)
    return resolved, provider_dict


def _build_sdk_config(provider_dict: dict[str, Any], path: Path) -> Config:
    try:
        config, _normalized = _create_config(provider_dict)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise LLMInspectionError(
            LLMProblem(PROBLEM_INVALID_PROVIDER_FILE, path, str(exc))
        ) from exc
    return config


def _has_environment_credential(provider_dict: dict[str, Any], provider_type: str) -> bool:
    credential_names = ["KIMI_API_KEY", "KIMIX_API_KEY"]
    if provider_type in {"openai_legacy", "openai_responses"}:
        credential_names.append("OPENAI_API_KEY")
    profile = get_provider_profile(provider_type)
    if profile is not None:
        credential_names.extend(name for name in profile.env_vars if not name.endswith("_BASE_URL"))
    credential_names = list(dict.fromkeys(credential_names))

    environment = provider_dict.get("env")
    if isinstance(environment, dict) and any(environment.get(key) for key in credential_names):
        return True
    return any(os.environ.get(key) for key in credential_names)


def _redact_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "Configured endpoint"
