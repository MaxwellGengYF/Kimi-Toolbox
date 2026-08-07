"""kimix base: session bootstrap, defaults, and process helpers.

The terminal-printing and wire-streaming layers now live in
:mod:`kimix.ui.printing` and :mod:`kimix.ui.stream`; they are re-exported
here for backward compatibility during the migration window (P8).
"""

from __future__ import annotations

import asyncio
import functools
import io
import os
import regex as re
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import orjson
from kimi_cli.wire.types import (
    ApprovalRequest,
    BackgroundTaskDisplayBlock,
    BriefDisplayBlock,
    CompactionBegin,
    CompactionEnd,
    DiffDisplayBlock,
    DisplayBlock,
    ShellDisplayBlock,
    StepBegin,
    StepInterrupted,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    ToolCall,
    ToolCallPart,
    ToolResult,
    UnknownDisplayBlock,
)

if TYPE_CHECKING:
    from kimi_agent_sdk import Session

# --- Re-exports: printing layer -------------------------------------------
from kimix.ui.printing import *  # noqa: F401,F403
from kimix.ui.printing import (  # noqa: F401
    _colorful_print,
    _process_lru,
    _quiet,
    _stream,
    _threads,
    print,
)

# --- Re-exports: stream layer ----------------------------------------------
from kimix.ui.stream import *  # noqa: F401,F403
from kimix.ui.stream import (  # noqa: F401
    percentage_and_token,
    percentage_str,
    print_agent_json,
    print_agent_json_flush_text,
)

# --- Re-exports: private names (backward compatibility) ---------------------
# Legacy private names from the printing/stream layers resolve lazily via
# PEP 562 ``__getattr__`` with a DeprecationWarning (P8 migration window;
# removal in the next minor release — see ChangeLog.md). Core names
# (``_stream``, ``_quiet``, ``_colorful_print``, ``_process_lru``,
# ``_threads``) are imported explicitly above and stay warning-free.
from kimix.ui import printing as _printing_mod
from kimix.ui import stream as _stream_mod

_DEPRECATED_PRIVATE_NAMES: dict[str, Any] = {
    name: value
    for module in (_printing_mod, _stream_mod)
    for name, value in vars(module).items()
    if name.startswith("_") and not name.startswith("__")
}


def __getattr__(name: str) -> Any:  # PEP 562 module-level __getattr__
    """Lazily re-export legacy private names with a deprecation warning."""
    if name in _DEPRECATED_PRIVATE_NAMES:
        warnings.warn(
            f"kimix.base.{name} is deprecated; import it from "
            "kimix.ui.printing / kimix.ui.stream instead. This compatibility "
            "shim will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_PRIVATE_NAMES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def run_thread(
    function: Callable[..., Any], args: tuple[Any, ...] | None = None
) -> threading.Thread:
    assert callable(function)
    global _threads
    _process_lru()

    if args is None:
        args = ()
    elif type(args) is not tuple:
        args = (args, )
    thd = threading.Thread(target=function, args=args)
    thd.start()
    _threads.append(thd)
    return thd


def run_script(path: str | Path) -> Any:
    return subprocess.Popen(
        [sys.executable, str(path)], creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def sync_all() -> None:
    global _threads
    for thd in _threads:
        thd.join()
    _threads.clear()


def _run_process_with_log(command: str) -> tuple[str, int]:
    print_info(f"Shell: {command}")
    result = subprocess.run(command, shell=True, capture_output=True)
    output = result.stdout.decode(
        "utf-8", errors="replace") if result.stdout else ""
    if result.stderr:
        output += "\n" + result.stderr.decode("utf-8", errors="replace")
    return output, result.returncode


async def _run_process_with_log_async(command: str) -> tuple[str, int]:
    print_info(f"Shell: {command}")
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    if stderr:
        output += "\n" + stderr.decode("utf-8", errors="replace")
    return output, proc.returncode


def _filter_error_output(
    result: str, code: int, keycode: tuple[str, ...] | None, skip_success: bool
) -> str | None:
    if skip_success and code == 0:
        return None
    if not keycode:
        return result
    lines = result.splitlines()
    for idx, line in enumerate(lines):
        lower_line = line.lower()
        for c in keycode:
            if c in lower_line:
                return "\n".join(lines[idx:])
    return result


def run_process_with_error(
    command: str,
    keycode: tuple[str, ...] | None,
    skip_success: bool = True,
) -> str | None:
    result, code = _run_process_with_log(command)
    return _filter_error_output(result, code, keycode, skip_success)


async def run_process_with_error_async(
    command: str,
    keycode: tuple[str, ...] | None,
    skip_success: bool = True,
) -> str | None:
    result, code = await _run_process_with_log_async(command)
    return _filter_error_output(result, code, keycode, skip_success)


_default_thinking: bool = True
_default_yolo: bool = True
_default_agent_file_dir: Path = Path(__file__).parent
_default_agent_file: Path = _default_agent_file_dir / "agent_worker.json"
_default_skill_dirs: list[Any] = []
_default_provider: dict[str, Any] | None = None
_default_sub_providers: list[dict[str, Any]] = []
_default_manually_cot: bool = False
_default_ralph: int | None = None

# Common skill directory paths (relative to current working directory)
COMMON_SKILL_DIRS: list[str] = [
    ".agents/skills",
    ".config/.agents/skills",
    ".opencode/skills",
    ".claude/skills",
    ".codex/skills",
    ".skills",
    "skills",
]


def set_default_thinking(value: bool) -> None:
    global _default_thinking
    _default_thinking = value


def set_default_yolo(value: bool) -> None:
    global _default_yolo
    _default_yolo = value


def set_default_agent_file_dir(value: Path) -> None:
    global _default_agent_file_dir
    _default_agent_file_dir = value


def set_default_agent_file(value: Path) -> None:
    global _default_agent_file
    _default_agent_file = value


def set_default_skill_dirs(value: list[Any]) -> None:
    global _default_skill_dirs
    _default_skill_dirs = value


def set_default_manually_cot(value: bool) -> None:
    global _default_manually_cot
    _default_manually_cot = value


def set_default_provider(value: dict[str, Any] | None) -> None:
    global _default_provider
    _default_provider = value


def set_default_sub_providers(providers: list[dict[str, Any]] | None) -> None:
    global _default_sub_providers
    _default_sub_providers = list(providers or [])


def get_default_sub_provider(role: str = "sub_agent") -> dict[str, Any] | None:
    for p in _default_sub_providers:
        if p.get("role", "sub_agent") == role:
            return p
    return None


def get_default_sub_providers_by_role(role: str = "backup") -> list[dict[str, Any]]:
    """Return ALL sub-providers matching *role*, preserving declaration order.

    Unlike ``get_default_sub_provider`` which returns only the first match,
    this returns every entry — essential for ``backup`` failover where
    multiple backup providers are declared.
    """
    return [p for p in _default_sub_providers if p.get("role", "sub_agent") == role]


# The failed-list for tool call that
# tuple: function-name, arguments, output, message


def get_skill_dirs(use_kaos_path: bool = True) -> list[Any]:
    from kaos.path import KaosPath

    global _default_skill_dirs
    if _default_skill_dirs:
        if use_kaos_path:
            return [KaosPath(str(i)) for i in _default_skill_dirs]
        return _default_skill_dirs

    _default_skill_dirs = [
        p for rel in COMMON_SKILL_DIRS if (p := Path(os.curdir) / rel).exists()
    ]
    # If there's a `skills` subdirectory under the skill dir, use `*/skills` pattern
    _default_skill_dirs = [
        p / "*/skills" if (p / "skills").is_dir() else p
        for p in _default_skill_dirs
    ]
    if _default_skill_dirs:
        for d in _default_skill_dirs:
            print_debug(f"skill dir: {str(d)}")
        if use_kaos_path:
            return [KaosPath(str(d)) for d in _default_skill_dirs]
        return _default_skill_dirs
    return []


generate_memory = """---

Compact the above agent conversation context according to the following priorities and rules.

**Priorities:**
- **Current Task State** — what is being worked on right now
- **Errors & Solutions** — all errors encountered and how they were resolved
- **Code Evolution** — final working versions only (drop intermediate attempts)
- **System Context** — project structure, dependencies, environment setup
- **Design Decisions** — architectural choices and rationale
- **TODO Items** — unfinished tasks and known issues
- **Project Overview** — purpose, scope, tech stack
- **Key Decisions** — critical choices, rationale, rejected alternatives
- **Current State** — what works, what's merged/verified, active branch, test results
- **Important Files** — key paths and their roles (add, modify, delete)
- **Architecture / Data Flow** — major components, interfaces, schema changes
- **Dependencies** — added, removed, upgraded packages or services
- **Risks / Rollback** — breaking changes, migration steps, revert strategy
- **Technical Notes** — patterns, constraints, APIs, env setup, performance or security considerations

**Rules:**
- **Keep:** error messages, stack traces, working solutions, current task
- **Merge:** similar discussions into single summary points
- **Remove:** redundant explanations, failed attempts (retain lessons learned), verbose comments
- **Condense:** long code blocks → signatures + key logic only

**Special Handling:**
- **Code:** keep full version if < 20 lines; otherwise keep signature + key logic
- **Errors:** keep full error message + final solution
- **Discussions:** extract decisions and action items only
```"""
