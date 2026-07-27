from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kimi_cli.mcp.client import MCPClient
    from kimi_cli.mcp.config import (
        discover_mcp_configs,
        load_global_mcp_config,
        load_project_mcp_config,
        merge_mcp_configs,
    )
    from kimi_cli.mcp.prompts import MCPPromptManager
    from kimi_cli.mcp.resources import MCPResourceManager
    from kimi_cli.mcp.roots import MCPRootsHandler
    from kimi_cli.mcp.sampling import MCPSamplingHandler
    from kimi_cli.mcp.server import MCPKimixServer, serve_http, serve_stdio
    from kimi_cli.mcp.types import MCPConnectionInfo

__all__ = [
    "MCPClient",
    "MCPConnectionInfo",
    "MCPKimixServer",
    "MCPPromptManager",
    "MCPResourceManager",
    "MCPRootsHandler",
    "MCPSamplingHandler",
    "discover_mcp_configs",
    "load_global_mcp_config",
    "load_project_mcp_config",
    "merge_mcp_configs",
    "serve_http",
    "serve_stdio",
]

# Map of public name -> submodule that defines it. Submodules are imported
# lazily on first attribute access so that importing ``kimi_cli.mcp`` (or any
# single submodule) does not eagerly pull in the heavy ``fastmcp``/``mcp``
# dependency trees, which are only needed when MCP is actually used.
_NAME_TO_SUBMODULE = {
    "MCPClient": "client",
    "MCPConnectionInfo": "types",
    "MCPKimixServer": "server",
    "MCPPromptManager": "prompts",
    "MCPResourceManager": "resources",
    "MCPRootsHandler": "roots",
    "MCPSamplingHandler": "sampling",
    "discover_mcp_configs": "config",
    "load_global_mcp_config": "config",
    "load_project_mcp_config": "config",
    "merge_mcp_configs": "config",
    "serve_http": "server",
    "serve_stdio": "server",
}


def __getattr__(name: str) -> Any:
    submodule = _NAME_TO_SUBMODULE.get(name)
    if submodule is not None:
        import importlib

        module = importlib.import_module(f"kimi_cli.mcp.{submodule}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
