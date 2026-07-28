from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from kaos.path import KaosPath
from kimi_cli.app import KimiCLI
from kimi_cli.mcp.config import (
    get_global_mcp_config_file,
    load_global_mcp_config,
    load_project_mcp_config,
    merge_mcp_configs,
)
from kimi_cli.mcp.server import serve_http, serve_stdio
from kimi_cli.session import Session

from kimix.ui.printing import print_debug, print_error, print_info, print_success


def run_mcp_subcommand(args: argparse.Namespace) -> None:
    """Dispatch to the requested MCP subcommand handler."""
    subcommand = args.mcp_command
    if subcommand == "serve":
        mcp_serve(args)
    elif subcommand == "list":
        mcp_list(args)
    elif subcommand == "test":
        mcp_test(args)
    else:
        print_error(f"Unknown mcp subcommand: {subcommand}")
        sys.exit(1)


def _resolve_work_dir(args: argparse.Namespace) -> Path:
    """Resolve the working directory from CLI args or the current directory."""
    work_dir = getattr(args, "work_dir", None)
    if work_dir is not None:
        return Path(work_dir).resolve()
    return Path.cwd()


def mcp_serve(args: argparse.Namespace) -> None:
    """Serve Kimix as an MCP server over stdio or HTTP."""

    async def _run() -> None:
        work_dir = KaosPath(str(_resolve_work_dir(args)))
        session = await Session.create(work_dir)

        agent_file: Path | None = None
        if getattr(args, "agent_file", None) is not None:
            agent_file = Path(args.agent_file)

        cli = await KimiCLI.create(session, agent_file=agent_file)
        runtime = cli.soul.agent.runtime

        options: dict[str, Any] = {
            "agent": cli.soul.agent,
            "include_resources": not getattr(args, "no_resource", False),
            "include_prompts": not getattr(args, "no_prompt", False),
        }

        if args.transport == "stdio":
            print_debug("Starting MCP server over stdio.")
            await serve_stdio(runtime, **options)
        else:
            print_debug(f"Starting MCP server over HTTP at {args.host}:{args.port}.")
            await serve_http(runtime, host=args.host, port=args.port, **options)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def mcp_list(args: argparse.Namespace) -> None:
    """List configured MCP servers from global and project configs."""
    work_dir = _resolve_work_dir(args)
    config = merge_mcp_configs(load_global_mcp_config(), load_project_mcp_config(work_dir))
    servers: dict[str, Any] = config.get("mcpServers", {})

    print_info(f"MCP config file: {get_global_mcp_config_file()}")
    if not servers:
        print_info("No MCP servers configured.")
        return

    for name, server in servers.items():
        if "command" in server:
            command = server["command"]
            command_args = " ".join(server.get("args", []))
            line = f"{name} (stdio): {command} {command_args}".rstrip()
        elif "url" in server:
            transport = server.get("transport") or "http"
            if transport == "streamable-http":
                transport = "http"
            line = f"{name} ({transport}): {server['url']}"
        else:
            line = f"{name}: {server}"
        print_info(f"  {line}")


def mcp_test(args: argparse.Namespace) -> None:
    """Test connection to a configured MCP server and list its tools."""
    import fastmcp

    name = args.server_name
    work_dir = _resolve_work_dir(args)
    config = merge_mcp_configs(load_global_mcp_config(), load_project_mcp_config(work_dir))
    servers: dict[str, Any] = config.get("mcpServers", {})

    if name not in servers:
        print_error(f"MCP server '{name}' not found.")
        sys.exit(1)

    server = servers[name]

    async def _test() -> None:
        client = fastmcp.Client({"mcpServers": {name: server}})
        try:
            async with client:
                tools = await client.list_tools()
        except Exception as exc:
            print_error(f"Connection failed: {type(exc).__name__}: {exc}")
            sys.exit(1)

        print_success(f"Connected to '{name}'")
        print_info(f"Available tools: {len(tools)}")
        if tools:
            print_info("Tools:")
            for tool in tools:
                desc = tool.description or ""
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                print_info(f"  - {tool.name}: {desc}")

    asyncio.run(_test())
