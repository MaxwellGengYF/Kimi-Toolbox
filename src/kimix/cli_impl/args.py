from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import orjson
from kaos.path import KaosPath

import kimix.base as base
from kimix.ui.printing import print_debug, print_error, print_warning

from . import constants, utils  # noqa: F401

# Import moved helpers from config.py
from kimix.utils.config import (
    _normalize_sub_providers,
    _pick_main_from_sub_providers,
    _REQUIRED_PROVIDER_KEYS,
    _SUB_PROVIDER_PICK_PRIORITY,
)


def _load_project_mcp_config() -> dict[str, Any]:
    """Load project-level MCP config from ``.kimix/mcp.json``."""
    mcp_json_path = constants.curr_dir / ".kimix" / "mcp.json"
    if mcp_json_path.exists():
        try:
            config = orjson.loads(mcp_json_path.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                print_debug(f"Loaded MCP config from {mcp_json_path}")
                return config
            print_warning(f"MCP config file {mcp_json_path} must contain a JSON object.")
        except orjson.JSONDecodeError as e:
            print_warning(f"Failed to parse MCP config file {mcp_json_path}: {e}")
        except Exception as e:
            print_warning(f"Failed to load MCP config file {mcp_json_path}: {e}")
    return {}



def _extract_config_from_remaining(args: argparse.Namespace, remaining: list[str]) -> None:
    """Extract ``--config <value>`` or ``--config=<value>`` from remaining args."""
    for i, token in enumerate(remaining):
        if token.startswith("--config="):
            args.config = token.split("=", 1)[1]
            return
        if token == "--config" and i + 1 < len(remaining):
            args.config = remaining[i + 1]
            return


def _maybe_run_default_config_init(args: argparse.Namespace) -> None:
    """Create ``src/kimix/default_config.json`` when no provider is configured.

    Without this guard, starting the CLI without an explicit ``--config`` and
    without the default config file leaves the global provider unset, which
    later surfaces as a cryptic ``LLM not set`` error. When stdin is a TTY we
    run the interactive ``/init`` flow; otherwise we write a minimal template
    config so non-interactive runs (tests, CI) can still boot. The API key can
    be supplied via the ``KIMI_API_KEY`` or ``KIMIX_API_KEY`` environment
    variables when running non-interactively.
    """
    from kimix.utils.config import init as kimix_init

    if getattr(args, "config", None) is not None:
        return
    if base._default_provider is not None:
        return

    default_config_path = Path(__file__).parent.parent / "default_config.json"
    if default_config_path.exists():
        return

    from .init import default_config as _default_config_template
    from .init import init as run_init

    if sys.stdin.isatty():
        try:
            run_init(initialize=False)
        except KeyboardInterrupt:
            return
    else:
        # Non-interactive environment: write a minimal template so the CLI can
        # start. The user must set KIMI_API_KEY/KIMIX_API_KEY or edit the file.
        api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("KIMIX_API_KEY") or ""
        config = orjson.loads(_default_config_template)
        config["api_key"] = api_key
        with open(default_config_path, "wb") as f:
            f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))
        print_warning(
            f"Created default config at {default_config_path}. "
            "Please set KIMI_API_KEY/KIMIX_API_KEY or edit the file."
        )

    # Reload global state with the newly created config file.
    if default_config_path.exists():
        kimix_init(
            config_path=None,
            yolo=not args.no_yolo,
            think=not args.no_think,
            skill_dir=args.skill_dir,
            ralph=args.ralph,
            manually_cot=args.manually_cot,
            colorful_print=not args.no_color,
            clean=args.clean,
        )


def set_arg() -> tuple[str | None, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Kimi Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)

    serve_parser = subparsers.add_parser("serve", description="Kimix HTTP server (opencode-style)")
    serve_parser.add_argument("--host", "--hostname", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=4096, help="Port to bind to")

    gui_parser = subparsers.add_parser("gui", description="Run Kimix backend + TypeScript/Vite frontend")
    gui_parser.add_argument("--host", "--hostname", default="127.0.0.1", help="Host to bind to")
    gui_parser.add_argument("--port", type=int, default=4096, help="Backend port")
    gui_parser.add_argument("--fe-port", type=int, default=5173, help="Frontend dev-server port")
    gui_parser.add_argument("--build", action="store_true", help="Run npm run build before starting the dev server")
    gui_parser.add_argument("--no-fe", action="store_true", help="Skip the frontend and start only the backend (useful when Node.js/npm/Vite are unavailable)")

    sse_cli_parser = subparsers.add_parser("ssecli", description="Kimix SSE CLI for debug")
    sse_cli_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to connect to (for ssecli)"
    )
    sse_cli_parser.add_argument(
        "--port", type=int, default=4096, help="Port to connect to (for ssecli)"
    )
    sse_cli_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print all SSE stream details and save to sse_log_<date>.txt",
    )

    mcp_parser = subparsers.add_parser("mcp", description="MCP server management commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)

    mcp_serve_parser = mcp_subparsers.add_parser(
        "serve", description="Serve Kimix as an MCP server"
    )
    mcp_serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport for the MCP server",
    )
    mcp_serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to for HTTP transport"
    )
    mcp_serve_parser.add_argument(
        "--port", type=int, default=4097, help="Port to bind to for HTTP transport"
    )
    mcp_serve_parser.add_argument(
        "--work-dir", type=str, default=None, help="Workspace directory for resources and session"
    )
    mcp_serve_parser.add_argument(
        "--agent-file", type=str, default=None, help="Agent specification file to load"
    )
    mcp_serve_parser.add_argument(
        "--no-resource", action="store_true", help="Do not expose file resources"
    )
    mcp_serve_parser.add_argument("--no-prompt", action="store_true", help="Do not expose prompts")

    mcp_subparsers.add_parser("list", description="List configured MCP servers")

    mcp_test_parser = mcp_subparsers.add_parser(
        "test", description="Test connection to an MCP server"
    )
    mcp_test_parser.add_argument("server_name", help="Name of the MCP server to test")

    parser.add_argument("-c", "--clean", action="store_true", help="Delete cache file after quit")
    parser.add_argument(
        "-no_color", "--no_color", action="store_true", help="Disable colorful print"
    )
    parser.add_argument(
        "-no_think", "--no_think", action="store_true", help="Disable thinking mode"
    )
    parser.add_argument("-no_yolo", "--no_yolo", action="store_true", help="Disable YOLO mode")
    parser.add_argument("--manually-cot", action="store_true", help="Enable manually CoT mode")
    parser.add_argument(
        "-s",
        "--skill-dir",
        type=str,
        nargs="*",
        default=None,
        help="Specify custom skill directory(s)",
    )

    parser.add_argument(
        "--ralph",
        nargs="?",
        const=1,
        type=int,
        default=None,
        help="Enable Ralph mode (unlimited iterations) or set to specific number",
    )
    # Parse args using parse_known_args first so we can detect a subcommand,
    # then re-parse with the appropriate subparser so subcommand-specific
    # arguments (e.g. --fe-port, --no-fe) are recognized.
    known_args, remaining = parser.parse_known_args()

    if known_args.command is not None:
        # Subcommand detected -- re-parse with its own parser so subcommand-specific
        # arguments (e.g. --fe-port, --no-fe) are recognized.
        # The ``mcp`` command has its own nested subparsers; the top-level parse
        # already consumed ``mcp_command`` and its options, so re-parsing would
        # fail with "required: mcp_command".
        if known_args.command == "mcp":
            args = known_args
        else:
            subparser_lookup: dict[str, argparse.ArgumentParser] = {
                "serve": serve_parser,
                "gui": gui_parser,
                "ssecli": sse_cli_parser,
            }
            sub_parser = subparser_lookup.get(known_args.command)
            if sub_parser is not None:
                args, _ = sub_parser.parse_known_args(remaining, namespace=known_args)
            else:
                args = known_args
    else:
        args = known_args

    # Extract --config from remaining args if not already set
    # (handles the case where --config appears after the subcommand)
    if getattr(args, "config", None) is None:
        _extract_config_from_remaining(args, remaining)

    args.mcp_config = _load_project_mcp_config()

    # Initialize global state via kimix.utils.config.init()
    from kimix.utils.config import init as kimix_init

    kimix_init(
        config_path=getattr(args, "config", None),
        yolo=not args.no_yolo,
        think=not args.no_think,
        skill_dir=args.skill_dir,
        ralph=args.ralph,
        manually_cot=args.manually_cot,
        colorful_print=not args.no_color,
        clean=args.clean,
    )

    # If no provider config was loaded, run the interactive init flow (or fall
    # back to a non-interactive template) so the CLI has a usable LLM config.
    _maybe_run_default_config_init(args)

    if args.command == "mcp":
        print_debug(f"Starting kimix mcp {args.mcp_command}.")
        return "mcp", args

    if args.command == "serve":
        print_debug("Starting kimix serve (opencode-style HTTP server).")
        return "serve", args

    if args.command == "gui":
        print_debug("Starting kimix gui (backend + frontend).")
        return "gui", args

    if args.command == "ssecli":
        print_debug("Starting kimix SSE cli (opencode-style HTTP CLI for debugging).")
        return "ssecli", args
    return None, args
