from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import ValidationError

from kimi_cli.share import get_share_dir

if TYPE_CHECKING:
    from fastmcp.mcp_config import MCPConfig

GLOBAL_MCP_FILE_NAME = "mcp.json"
PROJECT_MCP_DIR = ".kimix"
PROJECT_MCP_FILE_NAME = "mcp.json"


def get_global_mcp_config_file() -> Path:
    """Get the global MCP config file path."""
    return get_share_dir() / GLOBAL_MCP_FILE_NAME


def load_global_mcp_config() -> dict[str, Any]:
    """Load the global MCP config from ``~/.kimi/mcp.json``."""
    path = get_global_mcp_config_file()
    if not path.exists():
        return {"mcpServers": {}}
    data = orjson.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MCP config file '{path}' must contain a JSON object.")
    return data


def load_project_mcp_config(work_dir: Path | str | None = None) -> dict[str, Any]:
    """Load the project-level MCP config from ``.kimix/mcp.json``.

    The caller must provide the session work directory; this function no longer
    falls back to ``Path.cwd()`` so that project-level discovery is deterministic
    relative to the active session.
    """
    if work_dir is None:
        raise ValueError("work_dir is required to load project-level MCP config")
    work_dir = Path(str(work_dir))
    path = work_dir / PROJECT_MCP_DIR / PROJECT_MCP_FILE_NAME
    if not path.exists():
        return {"mcpServers": {}}
    data = orjson.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MCP config file '{path}' must contain a JSON object.")
    return data


def merge_mcp_configs(
    global_config: dict[str, Any],
    project_config: dict[str, Any],
    explicit_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge MCP configs with explicit > project > global priority."""
    merged: dict[str, Any] = {"mcpServers": {}}

    for source in (global_config, project_config):
        servers = source.get("mcpServers")
        if isinstance(servers, dict):
            for name, server in servers.items():
                if name not in merged["mcpServers"]:
                    merged["mcpServers"][name] = server

    if explicit_configs:
        for explicit in explicit_configs:
            servers = explicit.get("mcpServers")
            if isinstance(servers, dict):
                for name, server in servers.items():
                    merged["mcpServers"][name] = server

    return merged


def discover_mcp_configs(
    work_dir: Path | str | None = None,
    explicit_configs: list[MCPConfig] | list[dict[str, Any]] | None = None,
) -> list[MCPConfig]:
    """Discover and merge global, project, and explicit MCP configs.

    Returns a list containing a single merged ``MCPConfig`` if any servers are
    found, otherwise an empty list.
    """
    global_config = load_global_mcp_config()
    project_config = load_project_mcp_config(work_dir)

    explicit_dicts: list[dict[str, Any]] = []
    if explicit_configs:
        from fastmcp.mcp_config import MCPConfig

        for cfg in explicit_configs:
            if isinstance(cfg, MCPConfig):
                explicit_dicts.append(cfg.model_dump(mode="json"))
            elif isinstance(cfg, dict):
                explicit_dicts.append(cfg)

    merged = merge_mcp_configs(global_config, project_config, explicit_dicts)
    if not merged.get("mcpServers"):
        return []

    # Only pay the (heavy) fastmcp import cost when MCP servers are actually
    # configured and the merged config needs schema validation.
    from fastmcp.mcp_config import MCPConfig

    try:
        validated = MCPConfig.model_validate(merged)
    except ValidationError as e:
        raise ValueError(f"Invalid merged MCP config: {e}") from e

    return [validated]


def validate_mcp_config(config: dict[str, Any]) -> MCPConfig:
    """Validate a dict against fastmcp's ``MCPConfig`` schema."""
    from fastmcp.mcp_config import MCPConfig

    return MCPConfig.model_validate(config)
