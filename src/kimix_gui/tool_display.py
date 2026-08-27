"""Classify tools and normalize decoded calls/results into semantic content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import orjson

from kimix_gui.transcript_data import (
    ActivityField,
    ContentBlock,
    FieldListBlock,
    LiteralText,
    RawBlock,
    TextBlock,
    TodoBlock,
    TodoItem,
    ToolCallContent,
    ToolResultContent,
    ToolStatus,
    literal,
    localized,
    resolve_text,
)

_SDK_BORING_RESULT_MESSAGES = frozenset({"success", "succeeded", "ok", "done", "completed"})

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"

_COMPACT_VALUE_MAX_LEN = 60
_DETAIL_KEYS = frozenset(
    {
        "content",
        "code",
        "prompt",
        "old",
        "new",
        "question",
        "context",
        "instruction",
        "edit",
        "todos",
    }
)

_ARG_KEY_ALIASES: dict[str, str] = {
    "old_string": "old",
    "new_string": "new",
    "text": "content",
    "source_code": "code",
    "task": "prompt",
    "file_path": "path",
    "cmd": "command",
    "session": "session_id",
    "edits": "edit",
    "items": "todos",
    "block": "wait",
    "token_kill": "deduplicate_output",
    "old_str": "old",
    "new_str": "new",
    "old_content": "old",
    "new_content": "new",
    "original": "old",
    "replace_with": "new",
    "data": "content",
    "body": "content",
    "file": "path",
    "filepath": "path",
    "filename": "path",
    "file_name": "path",
    "changes": "edit",
    "modifications": "edit",
    "-A": "after_context",
    "-B": "before_context",
    "-C": "context",
    "-n": "line_number",
    "-i": "ignore_case",
}

_FAMILY_ALIASES: dict[str, str] = {
    "read": "read",
    "read_file": "read",
    "readfile": "read",
    "read_image": "read",
    "readimage": "read",
    "read_media": "read",
    "readmediafile": "read",
    "grep": "grep",
    "grep_local": "grep",
    "find_str": "grep",
    "findstr": "grep",
    "glob": "glob",
    "write": "write",
    "write_file": "write",
    "writefile": "write",
    "edit": "edit",
    "edit_file": "edit",
    "editfile": "edit",
    "replace": "edit",
    "str_replace": "edit",
    "bash": "shell",
    "shell": "shell",
    "pwsh": "shell",
    "powershell": "shell",
    "run": "shell",
    "python": "python",
    "todo": "todo",
    "todo_write": "todo",
    "todowrite": "todo",
    "todo_update": "todo",
    "todoupdate": "todo",
    "web_search": "search",
    "websearch": "search",
    "search": "search",
    "fetch_url": "fetch",
    "fetchurl": "fetch",
    "web_extract": "fetch",
    "webextract": "fetch",
    "subagent": "agent",
    "agent": "agent",
}

_FAMILY_LABEL: dict[str, str] = {
    "read": "Read",
    "grep": "Grep",
    "glob": "Glob",
    "write": "Write",
    "edit": "Edit",
    "python": "Python",
    "todo": "Todo",
    "search": "Search",
    "fetch": "Fetch",
    "agent": "Agent",
}

KNOWN_TOOL_FAMILIES = frozenset(
    {
        "read",
        "grep",
        "glob",
        "write",
        "edit",
        "shell",
        "python",
        "todo",
        "search",
        "fetch",
        "agent",
    }
)

_PRIMARY_KEYS: dict[str, frozenset[str]] = {
    "read": frozenset({"path"}),
    "grep": frozenset({"pattern"}),
    "glob": frozenset({"pattern"}),
    "write": frozenset({"path"}),
    "edit": frozenset({"path"}),
    "shell": frozenset({"command"}),
    "python": frozenset({"code"}),
    "todo": frozenset({"todos", "title", "content"}),
    "search": frozenset({"query", "search_term"}),
    "fetch": frozenset({"url", "urls"}),
    "agent": frozenset({"description", "prompt"}),
}

_SECONDARY_KEYS: dict[str, frozenset[str]] = {
    "read": frozenset({"offset", "limit", "n_lines", "glob"}),
    "grep": frozenset({"path", "include", "glob", "output_mode", "ignore_case"}),
    "glob": frozenset({"path"}),
    "write": frozenset({"mode"}),
    "edit": frozenset({"mode"}),
    "shell": frozenset({"cwd", "workdir", "working_directory"}),
    "todo": frozenset({"status", "rename_to"}),
}


def normalize_tool_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def tool_family(name: str | None) -> str:
    if not name:
        return "generic"
    key = normalize_tool_name(name)
    if key in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[key]
    return _FAMILY_ALIASES.get(key.replace("_", ""), "generic")


def tool_label(name: str | None) -> str:
    family = tool_family(name)
    if family in _FAMILY_LABEL:
        return _FAMILY_LABEL[family]
    if name:
        return _title_name(name)
    return "Tool"


def is_known_tool_family(name: str | None) -> bool:
    return tool_family(name) in KNOWN_TOOL_FAMILIES


def _title_name(name: str) -> str:
    token = name.strip().split(".")[-1]
    if not token:
        return "Tool"
    return token.replace("_", " ").replace("-", " ").title()


def _canonical_key(key: str) -> str:
    canonical = _ARG_KEY_ALIASES.get(key)
    if canonical is not None:
        return canonical
    return _ARG_KEY_ALIASES.get(key.lower(), key)


def _pretty_value(value: object) -> str:
    try:
        return orjson.dumps(
            value,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=lambda item: str(item),
        ).decode("utf-8")
    except TypeError, ValueError:
        return str(value)


def _one_line(value: object, limit: int = 72) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    return f"{text[: limit - 3]}..."


def _short_path(value: object) -> str:
    if isinstance(value, list | tuple):
        if not value:
            return ""
        if len(value) == 1:
            return str(value[0])
        return f"{value[0]} +{len(value) - 1}"
    return str(value) if value is not None else ""


def _first(parsed: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return parsed[key]
    return None


def _summary_parts(family: str, parsed: Mapping[str, Any]) -> tuple[LiteralText, ...]:
    if family == "read":
        parts = [_short_path(_first(parsed, "file_path", "path"))]
        offset = parsed.get("offset")
        if offset not in (None, 0, 1):
            parts.append(f"offset {offset}")
        limit = _first(parsed, "limit", "n_lines")
        if limit not in (None, ""):
            parts.append(f"{limit} lines")
        if parsed.get("glob"):
            parts.append("glob")
        return tuple(literal(part) for part in parts if part)
    if family == "grep":
        parts = [str(_first(parsed, "pattern") or "")]
        path = parsed.get("path")
        if path not in (None, "", "."):
            parts.append(str(path))
        include = _first(parsed, "include", "glob")
        if include:
            parts.append(str(include))
        mode = parsed.get("output_mode")
        if mode not in (None, "", "files_with_matches"):
            parts.append(str(mode))
        if parsed.get("-i") or parsed.get("case_insensitive"):
            parts.append("ignore-case")
        return tuple(literal(part) for part in parts if part)
    if family == "glob":
        pattern = str(_first(parsed, "pattern") or "")
        path = parsed.get("path")
        parts = [pattern]
        if path not in (None, "", "."):
            parts.append(str(path))
        return tuple(literal(part) for part in parts if part)
    if family in {"write", "edit"}:
        parts = [_short_path(_first(parsed, "file_path", "path"))]
        if family == "write":
            mode = parsed.get("mode")
            if mode not in (None, "", "overwrite"):
                parts.append(str(mode))
        else:
            edits = parsed.get("edits") or parsed.get("edit")
            old = _first(parsed, "old", "old_string")
            new = _first(parsed, "new", "new_string")
            if isinstance(edits, Sequence) and not isinstance(edits, str | bytes) and edits:
                parts.append(f"{len(edits)} edit" + ("s" if len(edits) != 1 else ""))
                first = edits[0]
                if isinstance(first, Mapping):
                    old = first.get("old") or first.get("old_string") or old
            if old:
                parts.append(_one_line(old, 40))
            elif new:
                parts.append(_one_line(new, 40))
        return tuple(literal(part) for part in parts if part)
    if family == "shell":
        parts = [str(_first(parsed, "command", "cmd") or "")]
        cwd = _first(parsed, "working_directory", "cwd", "workdir")
        if cwd:
            parts.append(str(cwd))
        return tuple(literal(part) for part in parts if part)
    if family == "python":
        code = str(_first(parsed, "code", "file") or "")
        first_line = next((line.strip() for line in code.splitlines() if line.strip()), "")
        summary = _one_line(first_line or code, 72)
        return (literal(summary),) if summary else ()
    if family == "todo":
        todos = parsed.get("todos") or parsed.get("updates") or parsed.get("items")
        if isinstance(todos, Sequence) and not isinstance(todos, str | bytes):
            count = len(todos)
            return (literal(f"{count} item" + ("s" if count != 1 else "")),)
        title = _first(parsed, "title", "content")
        if title:
            return (literal(f"{_todo_marker(parsed.get('status'))} {title}"),)
        return ()
    if family == "search":
        value = str(_first(parsed, "query", "search_term") or "")
        return (literal(value),) if value else ()
    if family == "fetch":
        url = _first(parsed, "url")
        if not url:
            urls = parsed.get("urls")
            if isinstance(urls, Sequence) and urls:
                first = urls[0]
                url = first.get("url") or first.get("href") if isinstance(first, Mapping) else first
        return (literal(str(url)),) if url else ()
    if family == "agent":
        value = _one_line(_first(parsed, "description", "prompt") or "", 72)
        return (literal(value),) if value else ()

    parts: list[str] = []
    stream_preview = ""
    for key, value in parsed.items():
        canonical = _canonical_key(str(key))
        if canonical in _DETAIL_KEYS and isinstance(value, str):
            if not stream_preview and value.strip():
                first_line = next(
                    (line.strip() for line in value.splitlines() if line.strip()), value
                )
                stream_preview = _one_line(first_line, 72)
            continue
        text = " ".join(str(value).split())
        if len(text) > _COMPACT_VALUE_MAX_LEN:
            text = text[: _COMPACT_VALUE_MAX_LEN - 3] + "..."
        parts.append(f"{canonical}:{text}")
    summary = " ".join(parts) or stream_preview
    return (literal(summary),) if summary else ()


def _field_hint(key: str, value: object) -> str:
    if key == "path":
        return "path"
    if key in {"url", "urls"}:
        return "url"
    if key == "command":
        return "command"
    if key in {"code", "old", "new", "content"}:
        return "code" if isinstance(value, str) else "json"
    if isinstance(value, Mapping | list | tuple):
        return "json"
    if isinstance(value, str) and "\n" in value:
        return "multiline"
    if isinstance(value, int | float):
        return "count"
    return "plain"


def _field_role(family: str, key: str) -> str:
    if key in _PRIMARY_KEYS.get(family, frozenset()):
        return "primary"
    if key in _SECONDARY_KEYS.get(family, frozenset()):
        return "secondary"
    return "detail"


def _field_value(value: object) -> LiteralText:
    return literal(value if isinstance(value, str) else _pretty_value(value))


def _todo_marker(status: object) -> str:
    key = str(status or "").lower()
    if key in {"done", "completed"}:
        return "[x]"
    if key in {"in_progress", "in-progress", "doing"}:
        return "[>]"
    return "[ ]"


def _todo_items(value: object, depth: int = 0) -> tuple[TodoItem, ...]:
    if isinstance(value, str):
        return (TodoItem(value, depth=depth),)
    if not isinstance(value, Mapping):
        return (TodoItem(str(value), depth=depth),)
    title = str(value.get("content") or value.get("title") or value.get("name") or "")
    items = [
        TodoItem(
            title=title,
            status=str(value.get("status") or "pending"),
            depth=depth,
            notes=str(value.get("notes") or ""),
        )
    ]
    children = value.get("children") or value.get("items") or ()
    if isinstance(children, Sequence) and not isinstance(children, str | bytes):
        for child in children:
            items.extend(_todo_items(child, depth + 1))
    return tuple(items)


def build_tool_call_content(
    name: str,
    parsed: Mapping[str, Any] | None,
    raw_arguments: str | None,
    *,
    parse_state: str,
    extras: object = None,
) -> ToolCallContent:
    """Build a structured call after the wire boundary has decoded its JSON once."""

    family = tool_family(name)
    if parsed is None:
        details: tuple[ContentBlock, ...] = ()
        if raw_arguments:
            details = (RawBlock(raw_arguments, parse_state=parse_state),)
        if extras not in (None, "", {}, []):
            details += (RawBlock(_pretty_value(extras), label=literal("Extras")),)
        preview = _one_line(raw_arguments or "", 72)
        return ToolCallContent(
            summary_parts=(literal(preview),) if preview else (),
            details=details,
            raw_arguments=raw_arguments,
            parse_state=parse_state,
        )

    fields: list[ActivityField] = []
    for raw_key, value in parsed.items():
        if value in (None, ""):
            continue
        key = _canonical_key(str(raw_key))
        fields.append(
            ActivityField(
                name=key,
                value=_field_value(value),
                role=_field_role(family, key),
                hint=_field_hint(key, value),
            )
        )
    details: list[ContentBlock] = []
    todos = parsed.get("todos") or parsed.get("updates") or parsed.get("items")
    if family == "todo" and isinstance(todos, Sequence) and not isinstance(todos, str | bytes):
        todo_items: list[TodoItem] = []
        for item in todos:
            todo_items.extend(_todo_items(item))
        details.append(TodoBlock(tuple(todo_items)))
    if extras not in (None, "", {}, []):
        details.append(RawBlock(_pretty_value(extras), label=literal("Extras")))
    return ToolCallContent(
        summary_parts=_summary_parts(family, parsed),
        fields=tuple(fields),
        details=tuple(details),
        raw_arguments=raw_arguments,
        parse_state="complete",
    )


def _block_summary(block: ContentBlock) -> str:
    if isinstance(block, TextBlock):
        return next(
            (
                " ".join(line.split())
                for line in resolve_text(block.text).splitlines()
                if line.strip()
            ),
            "",
        )
    if isinstance(block, TodoBlock):
        return "Todos"
    if isinstance(block, FieldListBlock):
        if not block.fields:
            return ""
        return resolve_text(block.fields[0].value)
    if isinstance(block, RawBlock) and block.label is not None:
        return resolve_text(block.label)
    return ""


def build_tool_result_content(
    *,
    is_error: bool,
    message: str,
    display_blocks: Sequence[ContentBlock] = (),
    output_blocks: Sequence[ContentBlock] = (),
    extras: object = None,
) -> ToolResultContent:
    """Normalize a tool result without flattening its display blocks."""

    status: ToolStatus = "error" if is_error else "ok"
    outcome = localized(OUTCOME_FAILED if is_error else OUTCOME_SUCCEEDED)
    blocks = (*display_blocks, *output_blocks)
    summary_index: int | None = None
    summary = ""
    for index, block in enumerate(blocks):
        if value := _block_summary(block):
            summary_index = index
            summary = value
            break
    stripped_message = message.strip()
    if not summary and stripped_message.lower() not in _SDK_BORING_RESULT_MESSAGES:
        summary = next(
            (" ".join(line.split()) for line in stripped_message.splitlines() if line.strip()),
            "",
        )
    summary_parts = (literal(summary),) if summary else (outcome,)
    # A one-line BriefDisplayBlock chosen as the header summary is semantic header
    # content, not body output. Decide that here at the normalization boundary so
    # layout never has to compare rendered lines to infer what a block means.
    result_blocks = [
        block
        for index, block in enumerate(blocks)
        if not (
            index < len(display_blocks)
            and index == summary_index
            and isinstance(block, TextBlock)
            and "\n" not in resolve_text(block.text).strip()
        )
    ]
    if stripped_message and stripped_message.lower() not in _SDK_BORING_RESULT_MESSAGES:
        normalized_summary = " · ".join(resolve_text(part) for part in summary_parts)
        if stripped_message != normalized_summary and all(
            not isinstance(block, TextBlock) or resolve_text(block.text).strip() != stripped_message
            for block in result_blocks
        ):
            result_blocks.append(TextBlock(literal(message)))
    if extras not in (None, "", {}, []):
        result_blocks.append(RawBlock(_pretty_value(extras), label=literal("Extras")))
    return ToolResultContent(
        status=status,
        summary_parts=summary_parts,
        blocks=tuple(result_blocks),
    )
