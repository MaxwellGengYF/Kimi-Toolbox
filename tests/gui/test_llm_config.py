from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from kimi_cli.auth.codex import CodexModel

from kimix_gui.llm import (
    CONFIGURED_VARIANT,
    LEGACY_DEFAULT_VARIANT,
    PROBLEM_CREDENTIAL_MISSING,
    PROBLEM_INVALID_PROVIDER_FILE,
    PROBLEM_INVALID_SESSION_SELECTION,
    PROBLEM_VARIANT_UNAVAILABLE,
    PROVIDER_DEFAULT_VARIANT,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMInspectionError,
    LLMSelection,
    ProviderFileTarget,
    chatgpt_model_descriptor,
    chatgpt_selection,
    configured_selection,
    default_store_file,
    inspect_provider_file,
    reasoning_effort_variant,
    reasoning_effort_variants,
    resolve_selection,
    resolved_provider_file,
    session_selection_file,
)
from kimix_gui.llm import provider_file as provider_file_module
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


def test_provider_file_descriptor_is_redacted_and_has_one_configured_variant(
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
    assert model.configured_reasoning_effort == "high"
    assert model.show_thinking_stream is True
    assert tuple(variant.key for variant in model.variants) == (CONFIGURED_VARIANT,)


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
    assert resolved.available is False
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


def test_catalog_variants_preserve_order_normalize_ultra_and_keep_unknown_values() -> None:
    variants = reasoning_effort_variants(
        ("low", "ultra", "max", "off", "none", "future-effort", "low"),
        "ultra",
    )

    assert [variant.key.value for variant in variants] == [
        "low",
        "max",
        "none",
        "future-effort",
    ]
    assert [variant.key.value for variant in variants if variant.is_default] == ["max"]


def test_model_without_reasoning_efforts_exposes_provider_default() -> None:
    model = chatgpt_model_descriptor(
        CodexModel("plain-model"),
        connected=True,
        stale=False,
    )

    assert tuple(variant.key for variant in model.variants) == (PROVIDER_DEFAULT_VARIANT,)
    assert model.default_variant == model.variants[0]


def test_removed_saved_variant_is_not_silently_replaced() -> None:
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
    assert resolved.available is False
    assert resolved.problem is not None
    assert resolved.problem.kind == PROBLEM_VARIANT_UNAVAILABLE


def test_selection_rejects_provider_and_variant_kind_mismatches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Provider files only support"):
        LLMSelection(
            ProviderFileTarget(tmp_path / "provider.json"),
            reasoning_effort_variant("medium"),
        )
    with pytest.raises(ValueError, match="Invalid ChatGPT Variant"):
        LLMSelection(ChatGPTTarget("gpt-test"), CONFIGURED_VARIANT)


def test_store_persists_exact_project_and_session_selection_without_secrets(
    tmp_path: Path,
) -> None:
    provider_path = write_provider_file(tmp_path / "provider.json")
    selection = configured_selection(ProviderFileTarget(provider_path, "override-model"))
    metadata_file = tmp_path / "kimix-gui.json"
    work_dir = tmp_path / "project"
    store = config_store(tmp_path, metadata_file)

    store.set_interface(InterfacePreferences(theme="light"))
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
    assert data["version"] == 5
    assert data["provider_files"] == [str(provider_path.resolve())]
    assert data["work_dirs"][str(work_dir.resolve())]["default_llm"] == {
        "target": {
            "kind": "provider_file",
            "path": str(provider_path.resolve()),
            "model_override": "override-model",
        },
        "variant": {"kind": "configured"},
    }
    session_data = orjson.loads(
        (tmp_path / "sessions" / "session-1" / "kimix-gui.json").read_bytes()
    )
    assert session_data == {
        "version": 3,
        "llm": data["work_dirs"][str(work_dir.resolve())]["default_llm"],
    }
    assert not metadata_file.with_suffix(".json.tmp").exists()


def test_chatgpt_selection_round_trips_without_entering_provider_library(tmp_path: Path) -> None:
    store = config_store(tmp_path)
    work_dir = tmp_path / "work"
    selection = chatgpt_selection("gpt-5.6-terra", "medium")

    store.set_default(work_dir, selection)
    store.set_session(work_dir, "session-1", selection)
    reloaded = config_store(tmp_path)

    assert reloaded.default_selection_for(work_dir) == selection
    assert reloaded.session_selection_for(work_dir, "session-1").selection == selection
    assert reloaded.provider_files() == ()


@pytest.mark.parametrize("version", [3, 4])
def test_legacy_global_provider_file_migrates_immediately_to_v5(
    tmp_path: Path,
    version: int,
) -> None:
    path = write_provider_file(tmp_path / "provider.json")
    work_dir = tmp_path / "work"
    metadata = tmp_path / "kimix-gui.json"
    value: object = (
        str(path.resolve())
        if version == 3
        else {"kind": "config_file", "path": str(path.resolve())}
    )
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": version,
                "configs": [str(path.resolve())],
                "work_dirs": {str(work_dir.resolve()): {"default": value}},
            }
        )
    )

    store = config_store(tmp_path, metadata)

    assert store.default_selection_for(work_dir) == configured_selection(ProviderFileTarget(path))
    saved = orjson.loads(metadata.read_bytes())
    assert saved["version"] == 5
    assert saved["provider_files"] == [str(path.resolve())]
    assert saved["work_dirs"][str(work_dir.resolve())]["default_llm"]["variant"] == {
        "kind": "configured"
    }


@pytest.mark.parametrize("version", [3, 4])
def test_legacy_global_chatgpt_model_migrates_to_deferred_default_marker(
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
                    str(work_dir.resolve()): {"default": {"kind": "chatgpt", "model": "gpt-5.4"}}
                },
            }
        )
    )

    selection = config_store(tmp_path, metadata).default_selection_for(work_dir)

    assert selection == LLMSelection(ChatGPTTarget("gpt-5.4"), LEGACY_DEFAULT_VARIANT)
    saved = orjson.loads(metadata.read_bytes())
    assert saved["work_dirs"][str(work_dir.resolve())]["default_llm"]["variant"] == {
        "kind": "legacy_default"
    }


@pytest.mark.parametrize(
    ("version", "field"),
    [(1, "config"), (2, "llm")],
)
def test_legacy_provider_session_is_upgraded_when_accessed(
    tmp_path: Path,
    version: int,
    field: str,
) -> None:
    path = write_provider_file(tmp_path / "provider.json")
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(orjson.dumps({"version": version, field: str(path.resolve())}))
    store = config_store(tmp_path)

    stored = store.session_selection_for(tmp_path, "session-1")

    assert stored.selection == configured_selection(ProviderFileTarget(path))
    assert orjson.loads(metadata.read_bytes()) == {
        "version": 3,
        "llm": {
            "target": {"kind": "provider_file", "path": str(path.resolve())},
            "variant": {"kind": "configured"},
        },
    }


def test_legacy_chatgpt_session_keeps_deferred_marker_until_catalog_resolution(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 2,
                "llm": {"kind": "chatgpt", "model": "gpt-5.4"},
            }
        )
    )

    stored = config_store(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.selection == LLMSelection(
        ChatGPTTarget("gpt-5.4"),
        LEGACY_DEFAULT_VARIANT,
    )
    assert orjson.loads(metadata.read_bytes())["llm"]["variant"] == {"kind": "legacy_default"}


def test_store_preserves_a_removed_variant_value_exactly(tmp_path: Path) -> None:
    store = config_store(tmp_path)
    work_dir = tmp_path / "work"
    selection = LLMSelection(
        ChatGPTTarget("gpt-test"),
        reasoning_effort_variant("retired-effort"),
    )

    store.set_default(work_dir, selection)

    assert config_store(tmp_path).default_selection_for(work_dir) == selection


def test_corrupt_entries_are_isolated_instead_of_discarding_valid_entries(
    tmp_path: Path,
) -> None:
    valid_work_dir = str((tmp_path / "valid").resolve())
    invalid_work_dir = str((tmp_path / "invalid").resolve())
    metadata = tmp_path / "kimix-gui.json"
    valid = chatgpt_selection("gpt-test", "medium")
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 5,
                "interface": {},
                "provider_files": [7, "", str((tmp_path / "provider.json").resolve())],
                "work_dirs": {
                    valid_work_dir: {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "variant": {"kind": "reasoning_effort", "value": "medium"},
                        }
                    },
                    invalid_work_dir: {"default_llm": "broken"},
                },
            }
        )
    )

    store = config_store(tmp_path, metadata)

    assert store.default_selection_for(Path(valid_work_dir)) == valid
    assert store.default_selection_for(Path(invalid_work_dir)) is None
    assert len(store.provider_files()) == 1


def test_target_variant_mismatch_is_isolated_as_a_corrupt_global_entry(
    tmp_path: Path,
) -> None:
    valid_work_dir = tmp_path / "valid"
    invalid_work_dir = tmp_path / "invalid"
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 5,
                "provider_files": [],
                "work_dirs": {
                    str(valid_work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "variant": {
                                "kind": "reasoning_effort",
                                "value": "medium",
                            },
                        }
                    },
                    str(invalid_work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": "chatgpt", "model": "gpt-test"},
                            "variant": {"kind": "configured"},
                        }
                    },
                },
            }
        )
    )

    store = config_store(tmp_path, metadata)

    assert store.default_selection_for(valid_work_dir) == chatgpt_selection("gpt-test", "medium")
    assert store.default_selection_for(invalid_work_dir) is None


def test_corrupt_session_returns_a_structured_problem(tmp_path: Path) -> None:
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"not-json")

    stored = config_store(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.selection is None
    assert stored.problem is not None
    assert stored.problem.kind == PROBLEM_INVALID_SESSION_SELECTION


def test_loading_global_metadata_never_scans_session_history(tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []

    def resolver(work_dir: Path, session_id: str) -> Path:
        calls.append((work_dir, session_id))
        return tmp_path / session_id / "kimix-gui.json"

    KimixGuiConfigStore(tmp_path / "metadata.json", session_file_resolver=resolver)

    assert calls == []


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


def test_session_selection_file_lives_inside_kimi_session_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    path = session_selection_file(tmp_path / "project", "session-1")

    assert path.name == "kimix-gui.json"
    assert path.parent.name == "session-1"
    assert path.parents[2].name == "sessions"


def test_default_store_file_uses_kimix_gui_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    assert default_store_file() == tmp_path / "share" / "kimix-gui.json"


def test_provider_env_mapping_does_not_mutate_the_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = write_provider_file(tmp_path / "provider.json", api_key="")
    payload = orjson.loads(path.read_bytes())
    payload["env"] = {"OPENAI_API_KEY": "embedded-secret"}
    path.write_bytes(orjson.dumps(payload))

    model = inspect_provider_file(path)

    assert model.credential == "Environment"
    assert "OPENAI_API_KEY" not in os.environ
