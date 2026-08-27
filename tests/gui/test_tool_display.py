from __future__ import annotations

from kimix_gui.tool_display import (
    build_tool_call_content,
    build_tool_result_content,
    tool_family,
    tool_label,
)
from kimix_gui.transcript_data import RawBlock, TextBlock, literal, resolve_text


def test_write_call_marks_summary_and_multiline_content_without_text_parsing() -> None:
    content = "hello\nworld\n" + ("block\n" * 30)
    call = build_tool_call_content(
        "write",
        {"path": "notes.txt", "content": content},
        '{"path":"notes.txt"}',
        parse_state="complete",
    )

    assert [resolve_text(part) for part in call.summary_parts] == ["notes.txt"]
    assert [(field.name, field.role, field.hint) for field in call.fields] == [
        ("path", "primary", "path"),
        ("content", "detail", "code"),
    ]
    assert resolve_text(call.fields[1].value) == content


def test_python_and_shell_fields_carry_explicit_display_hints() -> None:
    code = "def run():\n    return 1 + 2\n"
    python = build_tool_call_content("python", {"code": code}, "raw", parse_state="complete")
    shell = build_tool_call_content(
        "bash",
        {"command": "pytest -q", "cwd": "/tmp/work"},
        "raw",
        parse_state="complete",
    )

    assert python.fields[0].hint == "code"
    assert resolve_text(python.summary_parts[0]) == "def run():"
    assert [(field.name, field.role, field.hint) for field in shell.fields] == [
        ("command", "primary", "command"),
        ("cwd", "secondary", "plain"),
    ]


def test_argument_aliases_are_canonicalized_at_the_boundary() -> None:
    call = build_tool_call_content(
        "edit",
        {"file_path": "app.py", "old_string": "foo = 1", "new_string": "foo = 2"},
        "raw",
        parse_state="complete",
    )

    assert [field.name for field in call.fields] == ["path", "old", "new"]
    assert [field.role for field in call.fields] == ["primary", "detail", "detail"]


def test_unknown_tool_keeps_all_fields_and_raw_arguments() -> None:
    raw = '{"query":"abc","limit":5}'
    call = build_tool_call_content(
        "custom_mcp", {"query": "abc", "limit": 5}, raw, parse_state="complete"
    )

    assert tool_family("custom_mcp") == "generic"
    assert tool_label("custom_mcp") == "Custom Mcp"
    assert call.raw_arguments == raw
    assert [field.name for field in call.fields] == ["query", "limit"]


def test_long_header_sources_survive_until_width_aware_layout() -> None:
    text = "x" * 180
    python = build_tool_call_content("python", {"code": text}, "raw", parse_state="complete")
    agent = build_tool_call_content(
        "subagent", {"description": text, "prompt": "details"}, "raw", parse_state="complete"
    )
    edit = build_tool_call_content(
        "edit",
        {"path": "app.py", "old_string": text, "new_string": "replacement"},
        "raw",
        parse_state="complete",
    )
    generic = build_tool_call_content("custom_mcp", {"value": text}, "raw", parse_state="complete")
    invalid = build_tool_call_content("read", None, text, parse_state="invalid")

    assert resolve_text(python.summary_parts[0]) == text
    assert resolve_text(agent.summary_parts[0]) == text
    assert resolve_text(edit.summary_parts[-1]) == text
    assert resolve_text(generic.summary_parts[0]) == f"value:{text}"
    assert resolve_text(invalid.summary_parts[0]) == text


def test_invalid_call_arguments_are_a_finite_raw_fallback() -> None:
    call = build_tool_call_content(
        "read", None, "{]", parse_state="invalid", extras={"provider": "test"}
    )

    assert call.parse_state == "invalid"
    assert call.raw_arguments == "{]"
    assert len(call.details) == 2
    assert all(isinstance(block, RawBlock) for block in call.details)


def test_result_keeps_blocks_extras_and_a_structured_outcome() -> None:
    result = build_tool_result_content(
        is_error=False,
        message="success",
        display_blocks=(TextBlock(literal("3 matches")),),
        output_blocks=(TextBlock(literal("one\ntwo\nthree")),),
        extras={"bytes": 20},
    )

    assert result.status == "ok"
    assert resolve_text(result.summary_parts[0]) == "3 matches"
    assert any(isinstance(block, RawBlock) for block in result.blocks)
    assert any(
        isinstance(block, TextBlock) and "one\ntwo\nthree" in resolve_text(block.text)
        for block in result.blocks
    )


def test_blank_result_message_uses_localizable_outcome_reference() -> None:
    failed = build_tool_result_content(is_error=True, message="   ")
    succeeded = build_tool_result_content(is_error=False, message="\n")

    assert resolve_text(failed.summary_parts[0]) == "failed"
    assert resolve_text(succeeded.summary_parts[0]) == "succeeded"
