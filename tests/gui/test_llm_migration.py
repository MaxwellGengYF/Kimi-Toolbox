from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from kimix_gui.llm import (
    AXIS_THINKING_EFFORT,
    PROBLEM_INVALID_SESSION_SELECTION,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMSelection,
    ParameterAssignment,
)


def store_for(tmp_path: Path, metadata_file: Path | None = None) -> KimixGuiConfigStore:
    return KimixGuiConfigStore(
        metadata_file or (tmp_path / "kimix-gui.json"),
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )


def variant(kind: str, value: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"kind": kind}
    if value is not None:
        result["value"] = value
    return result


@pytest.mark.parametrize(
    ("legacy_variant", "expected"),
    [
        (variant("configured"), LLMSelection(ChatGPTTarget("gpt-test"))),
        (variant("provider_default"), LLMSelection(ChatGPTTarget("gpt-test"))),
        (
            variant("reasoning_effort", "high"),
            LLMSelection(
                ChatGPTTarget("gpt-test"),
                ParameterAssignment({AXIS_THINKING_EFFORT: "high"}),
            ),
        ),
        (
            variant("legacy_default"),
            LLMSelection(ChatGPTTarget("gpt-test"), pinned=False),
        ),
    ],
)
def test_v5_global_variants_migrate_to_v6_idempotently(
    tmp_path: Path,
    legacy_variant: dict[str, object],
    expected: LLMSelection,
) -> None:
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
                            "variant": legacy_variant,
                        }
                    }
                },
            }
        )
    )

    first = store_for(tmp_path, metadata)

    assert first.default_selection_for(work_dir) == expected
    migrated_bytes = metadata.read_bytes()
    assert orjson.loads(migrated_bytes)["version"] == 6

    second = store_for(tmp_path, metadata)

    assert second.default_selection_for(work_dir) == expected
    assert metadata.read_bytes() == migrated_bytes


@pytest.mark.parametrize(
    ("legacy_variant", "expected"),
    [
        (variant("configured"), LLMSelection(ChatGPTTarget("gpt-test"))),
        (variant("provider_default"), LLMSelection(ChatGPTTarget("gpt-test"))),
        (
            variant("reasoning_effort", "medium"),
            LLMSelection(
                ChatGPTTarget("gpt-test"),
                ParameterAssignment({AXIS_THINKING_EFFORT: "medium"}),
            ),
        ),
        (
            variant("legacy_default"),
            LLMSelection(ChatGPTTarget("gpt-test"), pinned=False),
        ),
    ],
)
def test_v3_session_variants_migrate_to_v4(
    tmp_path: Path,
    legacy_variant: dict[str, object],
    expected: LLMSelection,
) -> None:
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 3,
                "llm": {
                    "target": {"kind": "chatgpt", "model": "gpt-test"},
                    "variant": legacy_variant,
                },
            }
        )
    )

    stored = store_for(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.problem is None
    assert stored.selection == expected
    migrated = orjson.loads(metadata.read_bytes())
    assert migrated == {
        "version": 4,
        "llm": {
            "target": {"kind": "chatgpt", "model": "gpt-test"},
            "parameters": dict(expected.parameters.entries),
            "pinned": expected.pinned,
        },
    }


def test_malformed_global_entries_are_isolated_from_valid_siblings(tmp_path: Path) -> None:
    valid_work_dir = tmp_path / "valid"
    malformed_work_dir = tmp_path / "malformed"
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
                            "parameters": {AXIS_THINKING_EFFORT: "high"},
                            "pinned": True,
                        }
                    },
                    str(malformed_work_dir.resolve()): {
                        "default_llm": {
                            "target": {"kind": []},
                            "parameters": {},
                            "pinned": True,
                        }
                    },
                },
            }
        )
    )

    store = store_for(tmp_path, metadata)

    assert store.default_selection_for(valid_work_dir) == LLMSelection(
        ChatGPTTarget("gpt-test"),
        ParameterAssignment({AXIS_THINKING_EFFORT: "high"}),
    )
    assert store.default_selection_for(malformed_work_dir) is None


@pytest.mark.parametrize("version", [[], {}, None, "6"])
def test_malformed_root_version_falls_back_without_raising(
    tmp_path: Path,
    version: object,
) -> None:
    metadata = tmp_path / "kimix-gui.json"
    metadata.write_bytes(orjson.dumps({"version": version, "work_dirs": {}}))

    store = store_for(tmp_path, metadata)

    assert store.default_selection_for(tmp_path) is None


@pytest.mark.parametrize("kind", [[], {}, None, 4])
def test_malformed_session_target_kind_returns_a_structured_problem(
    tmp_path: Path,
    kind: object,
) -> None:
    metadata = tmp_path / "sessions" / "session-1" / "kimix-gui.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(
        orjson.dumps(
            {
                "version": 4,
                "llm": {
                    "target": {"kind": kind},
                    "parameters": {},
                    "pinned": True,
                },
            }
        )
    )

    stored = store_for(tmp_path).session_selection_for(tmp_path, "session-1")

    assert stored.selection is None
    assert stored.problem is not None
    assert stored.problem.kind == PROBLEM_INVALID_SESSION_SELECTION


def test_unknown_axis_and_value_round_trip_through_global_and_session_stores(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "kimix-gui.json"
    work_dir = tmp_path / "work"
    selection = LLMSelection(
        ChatGPTTarget("gpt-test"),
        ParameterAssignment(
            {
                AXIS_THINKING_EFFORT: "future-effort",
                "future_axis": "future_value",
            }
        ),
    )
    store = store_for(tmp_path, metadata)

    store.set_default(work_dir, selection)
    store.set_session(work_dir, "session-1", selection)
    reloaded = store_for(tmp_path, metadata)

    assert reloaded.default_selection_for(work_dir) == selection
    assert reloaded.session_selection_for(work_dir, "session-1").selection == selection
