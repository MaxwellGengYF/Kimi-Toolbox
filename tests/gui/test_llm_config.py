from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from kimi_cli.auth.codex import CodexModel

from kimix_gui.llm import (
    AXIS_CONTEXT_WINDOW,
    AXIS_THINKING_EFFORT,
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_INVALID_PROVIDER_FILE,
    PROBLEM_INVALID_SESSION_SELECTION,
    PROBLEM_PARAMETER_UNKNOWN,
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMInspectionError,
    LLMModelDescriptor,
    LLMSelection,
    ParameterAssignment,
    ParameterOption,
    ParameterSpec,
    ProviderFileTarget,
    RuntimeOverrides,
    chatgpt_model_descriptor,
    chatgpt_selection,
    configured_selection,
    context_window_parameter,
    default_store_file,
    inspect_provider_file,
    provider_thinking_parameter,
    reasoning_effort_parameter,
    resolve_selection,
    resolved_provider_file,
    session_selection_file,
)
from kimix_gui.llm.providers import provider_file as provider_file_module
from kimix_gui.preferences import InterfacePreferences


def config_store(tmp_path: Path, metadata_file: Path | None = None) -> KimixGuiConfigStore:
    return KimixGuiConfigStore(
        metadata_file or (tmp_path / "metadata.json"),
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )


def write_provider_file(
    path: Path,
    *,
    model: str = "gpt-test",
    display_name: str = "Test Model",
    provider: str = "openai_legacy",
    api_key: str = "top-secret-key",
) -> Path:
    path.write_bytes(
        orjson.dumps(
            {
                "model_name": model,
                "name": display_name,
                "model": model,
                "max_context_size": 131_072,
                "max_tokens": 8192,
                "capabilities": ["thinking"],
                "url": "https://user:password@example.com/v1?api_key=query-secret",
                "type": provider,
                "api_key": api_key,
                "show_thinking_stream": True,
                "thinking_effort": "high",
            }
        )
    )
    return path


def test_provider_file_descriptor_is_redacted_and_exposes_proven_parameters(
    tmp_path: Path,
) -> None:
    path = write_provider_file(tmp_path / "provider.json")

    model = inspect_provider_file(path)

    assert model.target == ProviderFileTarget(path)
    assert model.label == "Test Model"
    assert model.model_id == "gpt-test"
    assert model.provider_type == "openai_legacy"
    assert model.endpoint == "https://example.com/v1"
    assert model.credential == "API key configured"
    assert model.max_context_size == 131_072
    assert model.max_tokens == 8192
    assert model.capabilities == ("thinking",)
    assert model.show_thinking_stream is True
    assert [parameter.axis for parameter in model.parameters] == [AXIS_THINKING_EFFORT]
    thinking = model.parameters[0]
    assert [option.value for option in thinking.options] == [
        "off",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert thinking.default is not None
    assert thinking.default.value == "high"


def test_provider_thinking_axis_respects_capabilities_and_always_thinking() -> None:
    assert provider_thinking_parameter(
        (),
        ("low", "high"),
        None,
        thinking_enabled=False,
    ) is None

    parameter = provider_thinking_parameter(
        ("thinking", "always_thinking"),
        ("high", "low", "max"),
        None,
        thinking_enabled=False,
    )

    assert parameter is not None
    assert [option.value for option in parameter.options] == ["low", "high", "max"]
    assert parameter.default is not None
    assert parameter.default.value == "max"


def test_context_window_axis_is_limited_to_verified_anthropic_model_families() -> None:
    parameter = context_window_parameter(
        "anthropic",
        "bedrock-claude-sonnet-4-20250929",
        200_000,
    )

    assert parameter is not None
    assert [option.value for option in parameter.options] == ["200k", "1m"]
    assert parameter.default is not None
    assert parameter.default.value == "200k"
    long_context = parameter.option("1m")
    assert long_context is not None
    assert long_context.overrides.max_context_size == 1_000_000
    assert long_context.overrides.beta_features == ("context-1m-2025-08-07",)
    assert context_window_parameter("openai_legacy", "claude-sonnet-4", 200_000) is None
    assert context_window_parameter("anthropic", "claude-haiku-4", 200_000) is None
    assert context_window_parameter("anthropic", "claude-sonnet-4", 1_000_000) is None


def test_missing_credentials_are_silent_and_structurally_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("KIMI_API_KEY", "KIMIX_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = write_provider_file(tmp_path / "provider.json", api_key="")

    resolved = resolved_provider_file(path)

    captured = capsys.readouterr()
    assert "api_key not found" not in captured.out
    assert "api_key not found" not in captured.err
    assert not resolved.available
    assert resolved.problem is not None
    assert resolved.problem.kind == PROBLEM_CREDENTIAL_MISSING


def test_environment_credential_does_not_enter_the_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    path = write_provider_file(tmp_path / "provider.json", api_key="")

    model = inspect_provider_file(path)

    assert model.credential == "Environment"
    assert model.problem is None
    assert "environment-secret" not in repr(model)


def test_catalog_parameter_normalizes_legal_values_in_catalog_order() -> None:
    parameter = reasoning_effort_parameter(
        ("high", "ultra", "low", "off", "none", "future-effort", "low"),
        "ultra",
    )

    assert parameter is not None
    model = chatgpt_model_descriptor(
        CodexModel(
            "gpt-test",
            reasoning_efforts=tuple(option.value for option in parameter.options),
            default_reasoning_effort="max",
        ),
        connected=True,
        stale=False,
    )
    parameter = model.parameters[0]
    assert [option.value for option in parameter.options] == [
        "high",
        "max",
        "low",
    ]
    assert parameter.default is not None
    assert parameter.default.value == "max"


def test_model_without_adjustable_values_has_no_parameter_axes() -> None:
    model = chatgpt_model_descriptor(
        CodexModel("plain-model"),
        connected=True,
        stale=False,
    )

    assert model.parameters == ()
    assert model.default_assignment == ParameterAssignment()
    assert resolve_selection(LLMSelection(model.target), [model]).available


def test_removed_saved_parameter_value_is_not_silently_replaced() -> None:
    model = chatgpt_model_descriptor(
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="low",
        ),
        connected=True,
        stale=False,
    )
    selection = chatgpt_selection("gpt-test", "medium")

    resolved = resolve_selection(selection, [model])

    assert resolved.selection == selection
    assert not resolved.available
    assert resolved.problem is not None
    assert resolved.problem.kind == PROBLEM_PARAMETER_VALUE_UNAVAILABLE
    assert resolved.materialized_selection == selection


def test_two_parameter_axes_resolve_and_merge_runtime_without_selection_redesign() -> None:
    thinking = ParameterSpec(
        AXIS_THINKING_EFFORT,
        (
            ParameterOption("low", RuntimeOverrides(thinking_effort="low"), True),
            ParameterOption("high", RuntimeOverrides(thinking_effort="high")),
        ),
        order=10,
    )
    context = ParameterSpec(
        AXIS_CONTEXT_WINDOW,
        (
            ParameterOption("small", RuntimeOverrides(max_context_size=100_000)),
            ParameterOption("large", RuntimeOverrides(max_context_size=1_000_000), True),
        ),
        order=20,
    )
    model = LLMModelDescriptor(
        target=ChatGPTTarget("gpt-test"),
        model_id="gpt-test",
        provider_type="openai-codex",
        endpoint="ChatGPT subscription",
        credential="Connected account",
        file_format="Built-in",
        parameters=(context, thinking),
    )
    selection = LLMSelection(
        model.target,
        ParameterAssignment(
            {
                AXIS_CONTEXT_WINDOW: "small",
                AXIS_THINKING_EFFORT: "high",
            }
        ),
    )

    resolved = resolve_selection(selection, [model])

    assert [parameter.axis for parameter in model.parameters] == [
        AXIS_THINKING_EFFORT,
        AXIS_CONTEXT_WINDOW,
    ]
    assert resolved.available
    assert resolved.runtime.thinking_effort == "high"
    assert resolved.runtime.max_context_size == 100_000
    assert resolved.materialized_selection == selection


def test_unknown_saved_parameter_axis_is_visible_as_a_validation_problem() -> None:
    model = chatgpt_model_descriptor(
        CodexModel("gpt-test", reasoning_efforts=("low",), default_reasoning_effort="low"),
        connected=True,
        stale=False,
    )
    selection = LLMSelection(
        model.target,
        ParameterAssignment({AXIS_THINKING_EFFORT: "low", "future_axis": "future"}),
    )

    resolved = resolve_selection(selection, [model])

    assert not resolved.available
    assert resolved.problem is not None
    assert resolved.problem.kind == PROBLEM_PARAMETER_UNKNOWN
    assert resolved.resolved[-1].axis == "future_axis"
    assert resolved.resolved[-1].stored_value == "future"


def test_unpinned_legacy_selection_materializes_defaults_only_with_fresh_metadata() -> None:
    fresh = chatgpt_model_descriptor(
        CodexModel("gpt-test", reasoning_efforts=("low", "high"), default_reasoning_effort="high"),
        connected=True,
        stale=False,
    )
    deferred = LLMSelection(ChatGPTTarget("gpt-test"), pinned=False)

    resolved = resolve_selection(deferred, [fresh])

    assert resolved.available
    assert resolved.needs_writeback
    assert resolved.materialized_selection == chatgpt_selection("gpt-test", "high")

    stale = chatgpt_model_descriptor(
        CodexModel("gpt-test", reasoning_efforts=("low", "high"), default_reasoning_effort="high"),
        connected=True,
        stale=True,
    )
    stale_result = resolve_selection(deferred, [stale])
    assert not stale_result.available
    assert not stale_result.materialized_selection.pinned


def test_store_persists_exact_project_and_session_assignments_without_secrets(
    tmp_path: Path,
) -> None:
    provider_path = write_provider_file(tmp_path / "provider.json")
    selection = chatgpt_selection("gpt-test", "high")
    metadata_file = tmp_path / "kimix-gui.json"
    work_dir = tmp_path / "project"
    store = config_store(tmp_path, metadata_file)

    store.set_interface(InterfacePreferences(theme="light"))
    store.add_provider_file(ProviderFileTarget(provider_path))
    store.set_default(work_dir, selection)
    store.set_session(work_dir, "session-1", selection)

    reloaded = config_store(tmp_path, metadata_file)
    assert reloaded.default_selection_for(work_dir) == selection
    assert reloaded.session_selection_for(work_dir, "session-1").selection == selection
    persisted = metadata_file.read_text(encoding="utf-8")
    assert "top-secret-key" not in persisted
    assert "password" not in persisted
    assert "query-secret" not in persisted
    data = orjson.loads(metadata_file.read_bytes())
    assert data["version"] == 6
    assert data["work_dirs"][str(work_dir.resolve())]["default_llm"] == {
        "target": {"kind": "chatgpt", "model": "gpt-test"},
        "parameters": {AXIS_THINKING_EFFORT: "high"},
        "pinned": True,
    }
    session_data = orjson.loads(
        (tmp_path / "sessions" / "session-1" / "kimix-gui.json").read_bytes()
    )
    assert session_data == {
        "version": 4,
        "llm": data["work_dirs"][str(work_dir.resolve())]["default_llm"],
    }


@pytest.mark.parametrize("version", [3, 4])
def test_legacy_global_chatgpt_selection_migrates_to_deferred_parameters(
    tmp_path: Path,
    version: int,
) -> None:
    work_dir = tmp_path / "work"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": version,
                "configs": [],
                "work_dirs": {
                    str(work_dir.resolve()): {
                        "default": {"kind": "chatgpt", "model": "gpt-5.4"}
                    }
                },
            }
        )
    )

    selection = config_store(tmp_path, metadata).default_selection_for(work_dir)

    assert selection == LLMSelection(ChatGPTTarget("gpt-5.4"), pinned=False)
    saved = orjson.loads(metadata.read_bytes())
    assert saved["version"] == 6
    assert saved["work_dirs"][str(work_dir.resolve())]["default_llm"] == {
        "target": {"kind": "chatgpt", "model": "gpt-5.4"},
        "parameters": {},
        "pinned": False,
    }


def test_v5_reasoning_variant_migrates_to_exact_parameter_assignment(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 5,
                "provider_files": [],
                "work_dirs": {
                    str(work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "variant": {"kind": "reasoning_effort", "value": "medium"},
                        }
                    }
                },
            }
        )
    )

    assert config_store(tmp_path, metadata).default_selection_for(work_dir) == chatgpt_selection(
        "gpt-test", "medium"
    )
    saved = orjson.loads(metadata.read_bytes())
    assert saved["version"] == 6
    assert saved["work_dirs"][str(work_dir.resolve())]["default_llm"]["parameters"] == {
        AXIS_THINKING_EFFORT: "medium"
    }


@pytest.mark.parametrize(("version", "field"), [(1, "config"), (2, "llm")])
def test_legacy_provider_session_is_upgraded_when_accessed(
    tmp_path: Path,
    version: int,
    field: str,
) -> None:
    path = write_provider_file(tmp_path / "provider.json")
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(orjson.dumps({"version": version, field: str(path.resolve())}))

    stored = config_store(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.selection == configured_selection(ProviderFileTarget(path))
    assert orjson.loads(metadata.read_bytes()) == {
        "version": 4,
        "llm": {
            "target": {"kind": "provider_file", "path": str(path.resolve())},
            "parameters": {},
            "pinned": True,
        },
    }


def test_corrupt_session_returns_a_structured_problem(tmp_path: Path) -> None:
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"not-json")

    stored = config_store(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.selection is None
    assert stored.problem is not None
    assert stored.problem.kind == PROBLEM_INVALID_SESSION_SELECTION


def test_invalid_parameter_tokens_are_isolated_from_valid_global_entries(tmp_path: Path) -> None:
    valid_work_dir = tmp_path / "valid"
    invalid_work_dir = tmp_path / "invalid"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 6,
                "provider_files": [],
                "work_dirs": {
                    str(valid_work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "parameters": {AXIS_THINKING_EFFORT: "medium"},
                            "pinned": True,
                        }
                    },
                    str(invalid_work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "parameters": {"bad axis": "medium"},
                            "pinned": True,
                        }
                    },
                },
            }
        )
    )

    store = config_store(tmp_path, metadata)

    assert store.default_selection_for(valid_work_dir) == chatgpt_selection("gpt-test", "medium")
    assert store.default_selection_for(invalid_work_dir) is None


def test_half_built_provider_is_a_structured_inspection_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_provider_file(tmp_path / "provider.json")
    half_built = SimpleNamespace(
        model=None,
        provider=SimpleNamespace(type="openai", base_url=""),
    )
    monkeypatch.setattr(provider_file_module, "_build_sdk_config", lambda *_args: half_built)

    with pytest.raises(LLMInspectionError) as caught:
        inspect_provider_file(path)

    assert caught.value.problem.kind == PROBLEM_INVALID_PROVIDER_FILE
    assert caught.value.problem.reason == "no model in config"


def test_store_paths_use_kimix_metadata_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    session_path = session_selection_file(tmp_path / "project", "session-1")

    assert session_path.name == "kimix-gui.json"
    assert session_path.parent.name == "session-1"
    assert session_path.parents[2].name == "sessions"
    assert default_store_file() == tmp_path / "share" / "kimix-gui.json"


def test_provider_file_model_override_drives_exact_parameter_metadata(tmp_path: Path) -> None:
    eligible = tmp_path / "eligible.json"
    eligible.write_bytes(
        orjson.dumps(
            {
                "model": "claude-sonnet-4-20250929",
                "max_context_size": 200_000,
                "max_tokens": 32_000,
                "capabilities": [],
                "url": "https://example.test/v1",
                "type": "anthropic",
                "api_key": "test-key",
            }
        )
    )
    ineligible = tmp_path / "ineligible.json"
    ineligible.write_bytes(
        orjson.dumps(
            {
                "model": "gpt-test",
                "max_context_size": 200_000,
                "max_tokens": 32_000,
                "capabilities": [],
                "url": "https://example.test/v1",
                "type": "anthropic",
                "api_key": "test-key",
            }
        )
    )

    removed = inspect_provider_file(eligible, model_override="gpt-unrelated")
    added = inspect_provider_file(ineligible, model_override="claude-opus-4-latest")

    assert removed.model_id == "gpt-unrelated"
    assert AXIS_CONTEXT_WINDOW not in {item.axis for item in removed.parameters}
    assert added.model_id == "claude-opus-4-latest"
    assert AXIS_CONTEXT_WINDOW in {item.axis for item in added.parameters}


def test_provider_file_does_not_guess_thinking_capability_from_model_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provider.json"
    path.write_bytes(
        orjson.dumps(
            {
                "model": "reasoning-mystery",
                "max_context_size": 100_000,
                "max_tokens": 8_000,
                "capabilities": [],
                "url": "https://example.test/v1",
                "type": "openai_legacy",
                "api_key": "test-key",
            }
        )
    )

    descriptor = inspect_provider_file(path)

    assert descriptor.capabilities == ()
    assert descriptor.parameters == ()
