"""Schema-level sync tests: the 15 report-synced tools expose the report's
canonical LLM-facing names, params, and descriptions (see report.md), while
keeping every legacy name working as a validation alias.

The plan ("Synchronize kimix Tool Names, Params & Descriptions with
report.md") defines the mapping; this file is the regression gate for it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from kosong.tooling import normalize_tool_name, resolve_tool_name
from kosong.utils.jsonschema import deref_json_schema

# report tool name -> (current class name, canonical params, params to keep
# as legacy aliases).  Only names/params/descriptions are asserted here.
REPORT_TOOLS: dict[str, dict[str, Any]] = {
    "read": {
        "class": "ReadFile",
        "canonical": ["file_path", "offset", "limit"],
        "aliases": {"path": "file_path", "line_offset": "offset", "n_lines": "limit"},
        "desc_lead": "Read a UTF-8 text file and return line-numbered content.",
        "param_desc": {
            "file_path": "Path to read, resolved by the filesystem backend.",
            "offset": "1-based first line to return. Defaults to 1.",
            "limit": "Maximum number of lines to return. Defaults to 2000.",
        },
    },
    "write": {
        "class": "WriteFile",
        "canonical": ["file_path", "content", "sandbox_permissions", "justification"],
        "aliases": {"path": "file_path", "text": "content"},
        "desc_lead": "Create or fully replace a UTF-8 text file.",
        "param_desc": {
            "file_path": "Path to write, resolved by the filesystem backend.",
            "content": "Full UTF-8 text content to write.",
        },
    },
    "edit": {
        "class": "EditFile",
        "canonical": ["file_path", "old_string", "new_string", "replace_all", "sandbox_permissions", "justification"],
        "aliases": {"path": "file_path"},
        "desc_lead": "Edit an existing UTF-8 text file by replacing literal text.",
        "param_desc": {
            "file_path": "Path to edit, resolved by the filesystem backend.",
            "old_string": "Literal text to replace. Must match exactly.",
            "new_string": "Literal replacement text. Use an empty string to delete the match.",
            "replace_all": "Replace all matches. Defaults to false; when false, old_string must appear exactly once.",
        },
    },
    "glob": {
        "class": "Glob",
        "canonical": ["pattern", "path"],
        "aliases": {"directory": "path"},
        "desc_lead": "Find files whose paths match a glob pattern.",
        "param_desc": {
            "pattern": "Glob pattern to match file paths against",
            "path": "Directory to search in.",
        },
    },
    "grep": {
        "class": "Grep",
        "canonical": ["pattern", "path", "include"],
        "aliases": {"glob": "include"},
        "desc_lead": "Search file contents with a ripgrep regular expression.",
        "param_desc": {
            "pattern": "Regular expression to search for (ripgrep syntax).",
            "include": "One glob filter for which files to search",
        },
    },
    "read_image": {
        "class": "ReadMediaFile",
        "canonical": ["file_path"],
        "aliases": {"path": "file_path"},
        "desc_lead": "Read a PNG/JPEG/WebP/GIF file and return the image itself.",
        "param_desc": {
            "file_path": "Path to the image file, resolved by the filesystem backend.",
        },
    },
    "pwsh": {
        "class": "Powershell",
        "canonical": ["command", "description", "timeoutMs", "workdir", "run_in_background", "sandbox_permissions", "justification"],
        "aliases": {"cmd": "command", "timeout": "timeoutMs"},
        "desc_lead": "Execute a PowerShell command",
        "param_desc": {
            "command": "The PowerShell command to execute.",
            "timeoutMs": "Timeout in milliseconds.",
            "workdir": "Working directory for this command.",
        },
    },
    "web_search": {
        "class": "SearchWeb",
        "canonical": ["query"],
        "aliases": {},
        "desc_lead": "Search the web for current information.",
        "param_desc": {"query": "The search query."},
    },
    "subagent": {
        "class": "Agent",
        "canonical": ["description", "prompt", "run_in_background"],
        "aliases": {"task": "prompt"},
        "desc_lead": "Delegate a self-contained task to a subagent",
        "param_desc": {
            "prompt": "The complete, self-contained task for the subagent.",
            "run_in_background": "Whether to run in the background and return a durable subagent id immediately.",
        },
    },
    "send_message": {
        "class": "AskAgent",
        "canonical": ["subagent_id", "message"],
        "aliases": {"id": "subagent_id", "question": "message"},
        "desc_lead": "Send a message to a background subagent by its subagent id",
        "param_desc": {
            "subagent_id": "The subagent id returned when the background subagent was started.",
            "message": "The message to deliver to the subagent.",
        },
    },
    "list_agents": {
        "class": "AgentList",
        "canonical": ["scope"],
        "aliases": {},
        "desc_lead": "List your continuable background subagents by durable id and label.",
        "param_desc": {"scope": "`children` (default) lists direct children only;"},
    },
    "interrupt_agent": {
        "class": "AgentClose",
        "canonical": ["agent_id"],
        "aliases": {"session": "agent_id", "session_id": "agent_id"},
        "desc_lead": "Request cancellation of a background agent's current turn by its agent id.",
        "param_desc": {"agent_id": "The agent id of the running agent to interrupt."},
    },
    "job_output": {
        "class": "TaskOutput",
        "canonical": ["job_id", "wait", "timeout_ms"],
        "aliases": {"task_id": "job_id", "block": "wait", "timeout": "timeout_ms"},
        "desc_lead": "Read a background job.",
        "param_desc": {
            "job_id": "Job id returned by the tool that started the background work.",
            "wait": "Block until the job reaches a terminal status",
            "timeout_ms": "Max wait in milliseconds",
        },
    },
    "todo_write": {
        "class": "TodoList",
        "canonical": ["todos"],
        "aliases": {"items": "todos"},
        "desc_lead": "Record and update a structured task list for the current work.",
        "param_desc": {"todos": "The COMPLETE task list, replacing any previous list."},
    },
    "workflow": {
        "class": "AgentSwarm",
        "canonical": [],  # name/description only (params intentionally not re-shaped)
        "aliases": {},
        "desc_lead": "Run a JavaScript workflow script that orchestrates subagents at scale.",
        "param_desc": {},
    },
}


class _FakeSession:
    def __init__(self) -> None:
        self.custom_data: dict[str, Any] = {}
        self.custom_config = mock.MagicMock()
        self.custom_config.get.side_effect = lambda key, default=None: (
            {} if key == "config_json" else default
        )
        self.id = "test-session-id"


def _fake_runtime() -> mock.MagicMock:
    r = mock.MagicMock()
    r.builtin_args.KIMI_WORK_DIR = Path(tempfile.gettempdir())
    r.additional_dirs = []
    r.skills_dirs = []
    r.environment.os_kind = "Linux"
    r.llm.capabilities = {"image_in", "video_in"}
    r.role = "root"
    r.session = mock.MagicMock()
    return r


def _build_tools() -> dict[str, Any]:
    """Instantiate the 15 report-synced tools with mocked dependencies."""
    session = _FakeSession()
    runtime = _fake_runtime()
    approval = mock.AsyncMock()

    from kimi_cli.tools.file.read import ReadFile
    from kimi_cli.tools.file.write import WriteFile
    from kimi_cli.tools.file.replace import EditFile
    from kimi_cli.tools.file.glob import Glob
    from kimi_cli.tools.file.grep_local import Grep
    from kimi_cli.tools.file.read_media import ReadMediaFile
    from kimi_cli.tools.todo import TodoList

    tools: dict[str, Any] = {
        "read": ReadFile(runtime=runtime, session=session),
        "write": WriteFile(runtime=runtime, approval=approval, session=session),
        "edit": EditFile(runtime=runtime, approval=approval, session=session),
        "glob": Glob(runtime=runtime),
        "grep": Grep(runtime=runtime),
        "read_image": ReadMediaFile(runtime=runtime),
        "todo_write": TodoList(runtime=runtime),
    }

    from kimix.tools.file.bash import pwsh_tool as pt
    with mock.patch.object(pt.sys, "platform", "win32"), mock.patch.object(
        pt._bash_tool, "_should_enable_powershell", return_value=True
    ), mock.patch.object(pt, "find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
        tools["pwsh"] = pt.Powershell(session=session)

    from kimi_cli.tools.web.search import SearchWeb
    from kimi_cli.config import Config
    cfg = Config()
    with mock.patch.object(
        cfg.services, "search",
        mock.MagicMock(base_url="http://x", api_key="k", oauth=None, custom_headers=None),
    ):
        tools["web_search"] = SearchWeb(config=cfg, runtime=runtime)

    from kimix.tools.agent import Agent, AskAgent, AgentList, AgentClose
    tools["subagent"] = Agent(session=session)
    tools["send_message"] = AskAgent(session=session)
    tools["list_agents"] = AgentList(session=session)
    tools["interrupt_agent"] = AgentClose(session=session)

    from kimix.tools.background import TaskOutput
    tools["job_output"] = TaskOutput(session=session)

    from kimix.tools.swarm import AgentSwarm
    s2 = _FakeSession()
    s2.custom_data["is_swarm_session"] = True
    tools["workflow"] = AgentSwarm(session=s2)

    return tools


def _schema_props(tool: Any) -> dict[str, Any]:
    return deref_json_schema(tool.params.model_json_schema())["properties"]


@pytest.mark.parametrize("report_name", sorted(REPORT_TOOLS))
def test_tool_name_matches_report(report_name: str) -> None:
    tool = _build_tools()[report_name]
    assert tool.name == report_name


@pytest.mark.parametrize("report_name", sorted(REPORT_TOOLS))
def test_canonical_params_present_with_report_descriptions(report_name: str) -> None:
    spec = REPORT_TOOLS[report_name]
    tool = _build_tools()[report_name]
    props = _schema_props(tool)
    for canonical in spec["canonical"]:
        assert canonical in props, f"{report_name} missing canonical param {canonical}"
    for param, desc_fragment in spec["param_desc"].items():
        assert param in props, f"{report_name} missing param {param}"
        assert desc_fragment in props[param].get("description", ""), (
            f"{report_name}.{param} description does not carry the report text: "
            f"{props[param].get('description', '')!r}"
        )
    assert spec["desc_lead"] in tool.description, (
        f"{report_name} description lead mismatch: {tool.description[:120]!r}"
    )


@pytest.mark.parametrize("report_name", sorted(REPORT_TOOLS))
def test_legacy_param_aliases_still_validate(report_name: str) -> None:
    """Old param spellings still validate (backward compat via aliases)."""
    spec = REPORT_TOOLS[report_name]
    tool = _build_tools()[report_name]
    for legacy, canonical in spec["aliases"].items():
        props = _schema_props(tool)
        assert canonical in props, f"{report_name} missing canonical {canonical}"
        # The legacy name must be accepted by the params model.
        assert legacy in _accepted_input_keys(tool.params), (
            f"{report_name} legacy alias {legacy} not accepted"
        )


def _accepted_input_keys(params_model: type) -> set[str]:
    """Collect every accepted input key (field names + declared aliases)."""
    keys: set[str] = set(params_model.model_fields.keys())
    for fname, finfo in params_model.model_fields.items():
        if finfo.alias and finfo.alias != fname:
            keys.add(finfo.alias)
        va = getattr(finfo, "validation_alias", None)
        if va is not None and not isinstance(va, str):
            for choice in getattr(va, "choices", ()) or ():
                if isinstance(choice, str):
                    keys.add(choice)
        elif isinstance(va, str):
            keys.add(va)
    return keys


def test_legacy_tool_names_resolve_via_redirects() -> None:
    """Old class names resolve to the report canonical names through the
    platform redirect map (mirrors KimiToolset.handle resolution)."""
    from kimi_cli.soul.toolset import _PLATFORM_REDIRECTS_NORM

    valid = set(REPORT_TOOLS) | {
        "bash", "Run", "python", "todo_push", "todo_sub", "todo_pop",
        "retrieve", "fetch_url", "context_usage", "compact",
    }
    legacy_map = {
        "ReadFile": "read", "WriteFile": "write", "EditFile": "edit",
        "Glob": "glob", "Grep": "grep", "ReadMediaFile": "read_image",
        "SearchWeb": "web_search", "Agent": "subagent",
        "AskAgent": "send_message", "AgentList": "list_agents",
        "AgentClose": "interrupt_agent", "TaskOutput": "job_output",
        "TodoList": "todo_write", "AgentSwarm": "workflow",
    }
    for legacy, expected in legacy_map.items():
        resolution = resolve_tool_name(
            legacy, valid, redirects=_PLATFORM_REDIRECTS_NORM
        )
        assert resolution.name == expected, (
            f"{legacy} should resolve to {expected}, got {resolution.name}"
        )


def test_legacy_shell_names_resolve_to_pwsh_on_windows() -> None:
    """On Windows the shell-name redirects point at pwsh."""
    from kimi_cli.soul.toolset import _build_platform_redirects

    redirects = _build_platform_redirects()
    assert redirects[normalize_tool_name("Powershell")] == "pwsh"
    if sys.platform == "win32":
        assert redirects[normalize_tool_name("Bash")] == "pwsh"
    else:
        assert redirects[normalize_tool_name("Powershell")] == "bash"


def test_todo_item_schema_uses_content() -> None:
    """todo_write items expose the report shape {content, status}."""
    from kimi_cli.tools.todo import Params as TodoParams

    schema = deref_json_schema(TodoParams.model_json_schema())
    item_schemas: list[dict[str, Any]] = []
    for branch in schema["properties"]["todos"].get("anyOf", []):
        if "items" in branch:
            item_schemas.append(branch["items"])
    assert item_schemas, "todos array item schema not found"
    item_props = item_schemas[0].get("properties", {})
    assert "content" in item_props
    assert "status" in item_props
