"""Guards against display copy or flattened text becoming program logic."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

from kimi_agent_sdk import ApprovalRequest, BriefDisplayBlock, ToolResult, ToolReturnValue
from kimix_gui import tool_display, transcript_layout
from kimix_gui.llm_config import (
    config_file_available,
    inspect_llm_config,
    unavailable_config_reference,
)
from kimix_gui.qt import labels, paint
from kimix_gui.rendering import WireNormalizer
from kimix_gui.tool_display import (
    _SDK_BORING_RESULT_MESSAGES,
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    build_tool_result_content,
)
from kimix_gui.transcript_data import (
    ActivityEntry,
    FieldListBlock,
    LocalizedText,
    NoticeEntry,
    ReplaceEntry,
    StartEntry,
    resolve_text,
)
from kimix_gui.transcript_layout import NO_DETAILS, entry_summary, layout_entry


def write_provider_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_inspected_reference_is_marked_inspected(tmp_path: Path) -> None:
    reference = inspect_llm_config(write_provider_config(tmp_path / "provider.json"))
    assert reference.inspected is True
    assert config_file_available(reference) is True


def test_placeholder_reference_is_marked_uninspected(tmp_path: Path) -> None:
    reference = unavailable_config_reference(tmp_path / "missing.json")
    assert reference.inspected is False
    assert reference.provider_type == "Unavailable"


def test_availability_ignores_translated_display_strings(tmp_path: Path) -> None:
    present = tmp_path / "present.json"
    present.write_text("{}", encoding="utf-8")
    translated = replace(
        unavailable_config_reference(present),
        provider_type="不可用",
        base_url="不可用",
        credential="不可用",
        model_name="配置不可用",
        error=None,
    )
    assert config_file_available(translated) is False


def _entry(normalized: object) -> object:
    mutation = normalized.mutations[-1]  # type: ignore[attr-defined]
    if isinstance(mutation, StartEntry | ReplaceEntry):
        return mutation.entry
    raise TypeError(type(mutation).__name__)


def test_approval_metadata_is_structured_and_never_filtered_by_prefix() -> None:
    entry = _entry(
        WireNormalizer().normalize(
            ApprovalRequest(
                id="approval:key: value",
                tool_call_id="call:key: value",
                sender="write",
                action="write file",
                description="Write a.py",
            )
        )
    )

    assert isinstance(entry, NoticeEntry)
    field_block = next(block for block in entry.blocks if isinstance(block, FieldListBlock))
    values = {field.name: resolve_text(field.value) for field in field_block.fields}
    assert values["Request ID"] == "approval:key: value"
    assert values["Tool call ID"] == "call:key: value"
    assert entry_summary(entry) == "write requests: write file"


def test_activity_summary_uses_parts_not_a_placeholder_string_comparison() -> None:
    normalizer = WireNormalizer()
    normalizer.normalize(
        __import__("kimi_agent_sdk").ToolCall(
            id="call-1",
            function=__import__("kimi_agent_sdk").ToolCall.FunctionBody(
                name="read", arguments='{"path":"a.py"}'
            ),
        )
    )
    entry = _entry(
        normalizer.normalize(
            ToolResult(
                tool_call_id="call-1",
                return_value=ToolReturnValue(
                    is_error=False,
                    output="",
                    message="success",
                    display=[BriefDisplayBlock(text="12 lines")],
                ),
            )
        )
    )
    assert isinstance(entry, ActivityEntry)
    assert entry_summary(entry) == "a.py · 12 lines"
    layout = layout_entry(entry, width=80, placeholder="（无详情）")
    assert layout.summary == "a.py · 12 lines"


def test_the_no_details_placeholder_is_only_supplied_at_the_view_boundary() -> None:
    assert NO_DETAILS == labels.NO_DETAILS_TEXT
    assert "NO_DETAILS" not in inspect.getsource(paint)
    assert "placeholder=translate_no_details()" in inspect.getsource(paint)


def test_deleted_text_inference_helpers_cannot_reappear() -> None:
    forbidden = {
        "_simple_fields",
        "informative_line",
        "expanded_body",
        "merged_summary",
        "tool_name_from_text",
        "strip_tool_name",
        "context_chars",
    }
    root = Path(__file__).resolve().parents[2] / "src" / "kimix_gui"
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{name} returned in {path.relative_to(root)}"


def test_layout_and_model_layers_do_not_parse_json() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "kimix_gui"
    paths = (
        root / "transcript_layout.py",
        root / "qt" / "paint.py",
        root / "qt" / "transcript_model.py",
        root / "qt" / "transcript_cards.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "orjson" not in source
        assert "json.loads" not in source


def test_sdk_boring_messages_keep_protocol_spelling() -> None:
    assert _SDK_BORING_RESULT_MESSAGES == frozenset(
        {"success", "succeeded", "ok", "done", "completed"}
    )


def test_display_outcomes_are_localizable_refs_not_protocol_values() -> None:
    assert OUTCOME_SUCCEEDED == "succeeded"
    assert OUTCOME_FAILED == "failed"
    assert OUTCOME_SUCCEEDED in _SDK_BORING_RESULT_MESSAGES

    succeeded = build_tool_result_content(is_error=False, message="succeeded")
    failed = build_tool_result_content(is_error=True, message="")
    assert isinstance(succeeded.summary_parts[0], LocalizedText)
    assert isinstance(failed.summary_parts[0], LocalizedText)
    assert resolve_text(succeeded.summary_parts[0]) == OUTCOME_SUCCEEDED
    assert resolve_text(failed.summary_parts[0]) == OUTCOME_FAILED


def test_empty_result_never_uses_the_old_untranslated_placeholder() -> None:
    for message in ("", "   ", "\n", "ok", "succeeded"):
        for is_error in (False, True):
            result = build_tool_result_content(is_error=is_error, message=message)
            assert resolve_text(result.summary_parts[0]) in {
                OUTCOME_SUCCEEDED,
                OUTCOME_FAILED,
            }
    assert '"(no visible output)"' not in inspect.getsource(tool_display)
    assert "SUMMARY_METADATA_LABELS" not in inspect.getsource(transcript_layout)


def test_tests_run_under_an_english_locale() -> None:
    from PySide6.QtCore import QLocale

    assert QLocale.system().language() == QLocale.Language.English
    assert QLocale().language() == QLocale.Language.English
