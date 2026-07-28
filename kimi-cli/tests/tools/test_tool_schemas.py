from __future__ import annotations

# ruff: noqa

from inline_snapshot import snapshot

from kimi_cli.tools.agent import Agent as AgentTool
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditFile
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.todo import TodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


def test_agent_params_schema(agent_tool: AgentTool):
    """Test the schema of Agent tool parameters."""
    assert agent_tool.base.parameters == snapshot(
        {
            "properties": {
                "description": {
                    "description": "Short task label (3–5 words).",
                    "type": "string",
                },
                "prompt": {
                    "description": "Task for the agent.",
                    "type": "string",
                },
                "subagent_type": {
                    "default": "coder",
                    "description": "Built-in agent type (default: coder).",
                    "type": "string",
                },
                "model": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Optional model override.",
                },
                "resume": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Agent ID to resume.",
                },
                "run_in_background": {
                    "default": False,
                    "description": "Run in background.",
                    "type": "boolean",
                },
                "timeout": {
                    "anyOf": [
                        {"maximum": 3600, "minimum": 30, "type": "integer"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "description": "Timeout in seconds (30–3600).",
                },
            },
            "required": ["description", "prompt"],
            "type": "object",
        }
    )


def test_todo_list_params_schema(todo_list_tool: TodoList):
    """Test the schema of TodoList tool parameters."""
    schema = todo_list_tool.base.parameters
    # Verify top-level structure
    assert schema.get("additionalProperties") is False
    assert schema.get("type") == "object"
    props = schema.get("properties", {})
    # Verify required properties exist
    # 'todos' declares alias 'items', so the advertised property is 'items'
    # (both spellings validate via populate_by_name=True).
    assert "items" in props
    assert "mode" in props
    assert "match_mode" in props
    assert "auto_fix" in props
    # Verify mode enum values
    assert props["mode"]["enum"] == ["overwrite", "append", "force_overwrite"]
    assert props["mode"]["default"] == "append"
    # Verify match_mode enum values
    assert props["match_mode"]["enum"] == ["exact", "fuzzy"]
    assert props["match_mode"]["default"] == "exact"
    # Verify auto_fix type
    assert props["auto_fix"]["type"] == "boolean"
    assert props["auto_fix"]["default"] is True
    # Verify todos (advertised as 'items') has Todo structure (may be $ref or inlined)
    todos_props = str(props["items"])
    assert "title" in todos_props and "status" in todos_props
    # Verify sub-schema has notes field in the inline todos definition
    assert "notes" in str(props["items"])


def test_read_file_params_schema(read_file_tool: ReadFile):
    """Test the schema of ReadFile tool parameters."""
    schema = read_file_tool.base.parameters
    props = schema.get("properties", {})
    # Verify top-level properties exist
    # 'path' declares alias 'file_path', so the advertised property is
    # 'file_path' (both spellings validate via populate_by_name=True).
    assert "file_path" in props
    assert "line_offset" in props
    assert "n_lines" in props
    assert "max_char" in props
    assert "char_offset" in props
    assert "glob" in props
    assert "show_line_numbers" in props
    # Verify line_offset is int (not list)
    assert props["line_offset"]["type"] == "integer"
    assert props["n_lines"]["type"] == "integer"
    assert props["max_char"]["type"] == "integer"
    # Verify glob is boolean
    assert props["glob"]["type"] == "boolean"
    # Verify show_line_numbers is boolean
    assert props["show_line_numbers"]["type"] == "boolean"


def test_read_media_file_params_schema(read_media_file_tool: ReadMediaFile):
    """Test the schema of ReadMediaFile tool parameters."""
    schema = read_media_file_tool.base.parameters
    props = schema.get("properties", {})
    # Verify required properties
    assert "path" in props
    assert props["path"]["type"] == "string"
    # Verify new params exist
    assert "info_only" in props
    assert "max_dimension" in props
    assert "quality" in props
    assert "auto_convert" in props
    assert "region_pct" in props
    assert "region" in props
    assert "full_resolution" in props
    # Verify quality bounds
    assert props["quality"]["default"] == 85
    assert props["quality"]["minimum"] == 1
    assert props["quality"]["maximum"] == 100
    # Verify auto_convert default
    assert props["auto_convert"]["default"] is True
    # Verify info_only default
    assert props["info_only"]["default"] is False


def test_glob_params_schema(glob_tool: Glob):
    """Test the schema of Glob tool parameters."""
    schema = glob_tool.base.parameters
    props = schema.get("properties", {})
    # Verify required properties
    assert "pattern" in props
    assert props["pattern"]["type"] == "string"
    # Verify new/modified params
    # 'directory' declares alias 'path', so the advertised property is 'path'
    # (both spellings validate via populate_by_name=True).
    assert "path" in props
    assert "include_dirs" in props
    assert props["include_dirs"]["default"] is False  # Changed from True to False
    assert "respect_gitignore" in props
    assert props["respect_gitignore"]["default"] is True
    assert props["respect_gitignore"]["type"] == "boolean"
    assert "include_ignored" in props  # Deprecated but still present
    assert "verbose" in props
    assert props["verbose"]["default"] is False
    assert props["verbose"]["type"] == "boolean"
    assert "timeout" in props


def test_grep_params_schema(grep_tool: Grep):
    """Test the schema of Grep tool parameters."""
    assert grep_tool.base.parameters == snapshot(
        {
            "properties": {
                "pattern": {
                    "description": "Regex pattern.",
                    "type": "string",
                },
                "path": {
                    "default": ".",
                    "description": "Search target directory or file.",
                    "type": "string",
                },
                "glob": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Glob filter.",
                },
                "output_mode": {
                    "default": "files_with_matches",
                    "description": "Output format: 'files_with_matches', 'count_matches', or 'content'.",
                    "enum": ["files_with_matches", "count_matches", "content"],
                    "type": "string",
                },
                "-B": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Lines before match (content mode only).",
                },
                "-A": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Lines after match (content mode only).",
                },
                "-C": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Lines around match (content mode only).",
                },
                "-n": {
                    "default": True,
                    "description": "Show line numbers (content mode only).",
                    "type": "boolean",
                },
                "-i": {
                    "default": False,
                    "description": "Case-insensitive search.",
                    "type": "boolean",
                },
                "type": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "File type filter.",
                },
                "head_limit": {
                    "anyOf": [{"minimum": 0, "type": "integer"}, {"type": "null"}],
                    "default": 500,
                    "description": "Max results (0 = unlimited).",
                },
                "offset": {
                    "default": 0,
                    "description": "Skip first N results.",
                    "minimum": 0,
                    "type": "integer",
                },
                "multiline": {
                    "default": False,
                    "description": "Multiline regex mode.",
                    "type": "boolean",
                },
                "include_ignored": {
                    "default": False,
                    "description": "Include .gitignore files.",
                    "type": "boolean",
                },
                "timeout": {
                    "default": 60,
                    "description": "Maximum time in seconds to wait for the search to complete.",
                    "minimum": 1,
                    "type": "integer",
                },
                "token_kill": {
                    "default": True,
                    "description": "Deduplicate repeated output lines via rtk (token killer). Set to False to see raw, unfiltered output.",
                    "type": "boolean",
                },
            },
            "required": ["pattern"],
            "type": "object",
        }
    )


def test_write_file_params_schema(write_file_tool: WriteFile):
    """Test the schema of WriteFile tool parameters."""
    schema = write_file_tool.base.parameters
    props = schema.get("properties", {})
    # Verify required properties
    # 'path'/'content' declare aliases 'file_path'/'text', so the advertised
    # properties are the aliases (both spellings validate via populate_by_name=True).
    assert "file_path" in props
    assert "text" in props
    assert "mode" in props
    assert "auto_fix_json" in props
    assert "mkdir" in props
    assert "show_diff" in props
    # Verify types
    assert props["file_path"]["type"] == "string"
    assert props["text"]["type"] == "string"
    assert props["mode"]["enum"] == ["overwrite", "append"]
    assert props["auto_fix_json"]["type"] == "boolean"
    assert props["auto_fix_json"]["default"] is True
    assert props["mkdir"]["type"] == "boolean"
    assert props["mkdir"]["default"] is True
    assert props["show_diff"]["type"] == "boolean"
    assert props["show_diff"]["default"] is False


def test_edit_file_params_schema(edit_file_tool: EditFile):
    """Test the schema of EditFile tool parameters."""
    schema = edit_file_tool.base.parameters
    props = schema.get("properties", {})
    # Verify required properties
    # 'path'/'edit' declare aliases 'file_path'/'edits', so the advertised
    # properties are the aliases (both spellings validate via populate_by_name=True).
    assert "file_path" in props
    assert "edits" in props
    assert props["file_path"]["type"] == "string"
    # edit accepts both a single Edit object and a list of Edit objects
    edit_schema = props["edits"]
    assert "anyOf" in edit_schema
    # Find the array option in anyOf
    array_schema = None
    for opt in edit_schema["anyOf"]:
        if opt.get("type") == "array":
            array_schema = opt
            break
    assert array_schema is not None, "Expected an array option in anyOf"
    # Verify edit item properties include new fields
    # 'old'/'new' declare aliases 'old_string'/'new_string', so the advertised
    # properties are the aliases (both spellings validate via populate_by_name=True).
    item_schema = array_schema.get("items", {})
    item_props = item_schema.get("properties", {})
    assert "old_string" in item_props
    assert "new_string" in item_props
    assert "replace_all" in item_props
    assert "max_replacements" in item_props
    assert "match_mode" in item_props
    # Verify match_mode enum
    assert item_props["match_mode"]["enum"] == ["exact", "fuzzy"]
    assert item_props["match_mode"]["default"] == "fuzzy"
    # Verify max_replacements is nullable integer (None = unlimited)
    assert item_props["max_replacements"]["anyOf"][0]["minimum"] == 1


def test_search_web_params_schema(search_web_tool: SearchWeb):
    """Test the schema of MoonshotSearch tool parameters."""
    assert search_web_tool.base.parameters == snapshot(
        {
            "properties": {
                "query": {
                    "description": "Search query.",
                    "type": "string",
                },
                "limit": {
                    "default": 5,
                    "description": "Number of results. Prefer a specific query over a high limit.",
                    "maximum": 20,
                    "minimum": 1,
                    "type": "integer",
                },
                "include_content": {
                    "default": False,
                    "description": "Include full page content. Increases token usage.",
                    "type": "boolean",
                },
            },
            "required": ["query"],
            "type": "object",
        }
    )


def test_fetch_url_params_schema(fetch_url_tool: FetchURL):
    """Test the schema of FetchURL tool parameters."""
    schema = fetch_url_tool.base.parameters
    props = schema.get("properties", {})
    # Verify required properties
    assert "url" in props
    assert "timeout" in props
    assert "method" in props
    assert "headers" in props
    assert "body" in props
    assert "follow_redirects" in props
    assert "max_redirects" in props
    # Verify types
    assert props["url"]["type"] == "string"
    assert props["timeout"]["default"] == 30.0
    assert props["timeout"]["minimum"] == 1.0
    assert props["timeout"]["maximum"] == 300.0
    assert props["method"]["enum"] == ["GET", "POST"]
    assert props["follow_redirects"]["default"] is True
    assert props["max_redirects"]["default"] == 5
    assert props["max_redirects"]["maximum"] == 20
