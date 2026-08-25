import json

import orjson
from typing import Any, cast

import streamingjson  # type: ignore[reportMissingTypeStubs]
from kaos.path import KaosPath
from kosong.utils.typing import JsonType

from kimi_cli.utils.string import shorten_middle


class SkipThisTool(Exception):
    """Raised when a tool decides to skip itself from the loading process."""

    pass


def resolve_tool_class(module: Any, attr: str) -> type | None:
    """Return the tool class in *module* referenced by a manifest ``module:attr`` entry.

    ``attr`` may be the class name (e.g. ``Run``) or a tool-name string (e.g.
    ``read``) matching a ``CallableTool``/``CallableTool2`` subclass ``name``.
    Both the toolset loader and the reflection tool listing resolve manifest
    entries through this helper so they can never drift apart.
    """
    from kosong.tooling import CallableTool, CallableTool2

    tool_cls = getattr(module, attr, None)
    if isinstance(tool_cls, type):
        return tool_cls
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        candidate = getattr(module, attr_name, None)
        if not isinstance(candidate, type) or not issubclass(
            candidate, (CallableTool, CallableTool2)
        ):
            continue
        if getattr(candidate, "name", None) == attr:
            return candidate
    return None


def _first_web_extract_url(args: JsonType) -> str | None:
    """Return the first usable URL from a ``web_extract`` arguments dict.

    Accepts URL strings or dict items with a ``url``/``href`` field (search
    results forwarded by the model). Returns None when no usable URL is found.
    """
    if not isinstance(args, dict) or not isinstance(args.get("urls"), list):
        return None
    for item in args["urls"]:
        if isinstance(item, str):
            candidate = item.strip()
            if candidate:
                return candidate
        if isinstance(item, dict):
            candidate = item.get("url") or item.get("href")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return None


def extract_key_argument(
    json_content: str | streamingjson.Lexer,
    tool_name: str,
    work_dir: KaosPath | None = None,
) -> str | None:
    if isinstance(json_content, streamingjson.Lexer):
        json_str = json_content.complete_json()
    else:
        json_str = json_content
    from kosong.utils.jsonx import loads_relaxed

    try:
        curr_args: JsonType = loads_relaxed(json_str)
    except (orjson.JSONDecodeError, json.JSONDecodeError):
        return None
    if not curr_args:
        return None
    match tool_name:
        case "subagent":
            if not isinstance(curr_args, dict) or not curr_args.get("description"):
                return None
            key_argument = str(curr_args["description"])
        case "todo_write":
            return None
        case "read":
            if not isinstance(curr_args, dict) or not (
                curr_args.get("file_path") or curr_args.get("path")
            ):
                return None
            raw_read_path = curr_args.get("file_path") or curr_args.get("path")
            if work_dir is None:
                return None
            key_argument = _normalize_path(str(raw_read_path), work_dir)
        case "read_image":
            pass
        case "glob":
            if not isinstance(curr_args, dict) or not curr_args.get("pattern"):
                return None
            key_argument = str(curr_args["pattern"])
        case "grep":
            if not isinstance(curr_args, dict) or not curr_args.get("pattern"):
                return None
            key_argument = str(curr_args["pattern"])
        case "write":
            pass
        case "edit":
            pass
        case "web_search":
            if not isinstance(curr_args, dict) or not curr_args.get("query"):
                return None
            key_argument = str(curr_args["query"])
        case "fetch_url":
            if not isinstance(curr_args, dict) or not curr_args.get("url"):
                return None
            key_argument = str(curr_args["url"])
        case "web_extract":
            first_url = _first_web_extract_url(curr_args)
            if first_url is None:
                if isinstance(json_content, streamingjson.Lexer):
                    content: list[str] = cast(list[str], json_content.json_content)  # type: ignore[reportUnknownMemberType]
                    key_argument = "".join(content)
                else:
                    key_argument = json_content
            else:
                key_argument = first_url
        case _:
            if isinstance(json_content, streamingjson.Lexer):
                # lexer.json_content is list[str] based on streamingjson source code
                content: list[str] = cast(list[str], json_content.json_content)  # type: ignore[reportUnknownMemberType]
                key_argument = "".join(content)
            else:
                key_argument = json_content
    key_argument = shorten_middle(key_argument, width=50)
    return key_argument


def _normalize_path(path: str, work_dir: KaosPath) -> str:
    cwd = str(work_dir.canonical())
    if path.startswith(cwd):
        path = path[len(cwd) :].lstrip("/\\")
    return path
