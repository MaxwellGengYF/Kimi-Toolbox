"""Kimi CLI X package.

The package used to eagerly re-export ``kimix.utils`` and ``kimix.base`` at
import time, which pulled the whole ``kimi_agent_sdk`` chain (~500 ms) into
every ``import kimix`` — including ``kimix.cli_impl.main``, the CLI entry
point. Names are now resolved lazily via PEP 562 ``__getattr__``: importing
this package is cheap and each public name triggers the import of exactly one
submodule on first access.

``__all__`` is the union of ``kimix.utils.__all__`` and the public names of
``kimix.base`` (snapshot; keep in sync when the re-exported APIs change —
``tests/test_cli_startup_lazy.py`` verifies parity).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # ---- kimix.utils.__all__ ----
    "SystemPromptType",
    "_default_session",
    "_should_print_usage",
    "_cli_sessions",
    "_add_cli_session",
    "_remove_cli_session",
    "_refresh_cli_sessions",
    "init",
    "_create_config",
    "_SYSTEM_PROMP",
    "get_system_prompt",
    "context_path",
    "delete_session_dir",
    "make_kaos_dir",
    "_ensure_skill_dirs",
    "_create_session_async",
    "create_session",
    "create_supervisor_session",
    "get_tool_call_errors",
    "close_session",
    "close_session_async",
    "get_cancel_event",
    "cancel_prompt",
    "get_default_session",
    "_create_default_session",
    "_create_default_session_async",
    "_print_usage",
    "print_usage",
    "clear_default_context",
    "prompt_async",
    "prompt",
    "prompt_path",
    "prompt_plan",
    "prompt_plan_async",
    "fix_error",
    "async_prompt",
    "async_fix_error",
    "compact_default_context",
    "set_ralph_loop",
    "refresh_env_from_registry",
    # ---- kimix.base public names ----
    "Any",
    "ApprovalRequest",
    "BackgroundTaskDisplayBlock",
    "BgColor",
    "BgColor256",
    "BgTrueColor",
    "BriefDisplayBlock",
    "COMMON_SKILL_DIRS",
    "Callable",
    "Color",
    "Color256",
    "CompactionBegin",
    "CompactionEnd",
    "DiffDisplayBlock",
    "DisplayBlock",
    "Enum",
    "GRAY",
    "GRAY_DARK",
    "GRAY_LIGHT",
    "GRAY_NEAR_BLACK",
    "MessageType",
    "Path",
    "PrintStream",
    "ShellDisplayBlock",
    "StepBegin",
    "StepInterrupted",
    "StreamPrintState",
    "Style",
    "TOOL_NAME_REDIRECTS",
    "TRUE_GRAY",
    "TYPE_CHECKING",
    "TextPart",
    "ThinkPart",
    "TodoDisplayBlock",
    "ToolCall",
    "ToolCallPart",
    "ToolResult",
    "TrueColor",
    "UnknownDisplayBlock",
    "annotations",
    "asyncio",
    "colorful_print",
    "colorful_text",
    "ctypes",
    "dataclass",
    "format_tool_args",
    "functools",
    "generate_memory",
    "get_default_sub_provider",
    "get_default_sub_providers_by_role",
    "get_skill_dirs",
    "io",
    "kernel32",
    "normalize_tool_name",
    "orjson",
    "os",
    "percentage_and_token",
    "percentage_str",
    "print",
    "print_agent_json",
    "print_agent_json_flush_text",
    "print_debug",
    "print_error",
    "print_info",
    "print_string",
    "print_success",
    "print_warning",
    "re",
    "resolve_tool_name",
    "run_process_with_error",
    "run_process_with_error_async",
    "run_script",
    "run_thread",
    "set_default_agent_file",
    "set_default_agent_file_dir",
    "set_default_manually_cot",
    "set_default_provider",
    "set_default_skill_dirs",
    "set_default_sub_providers",
    "set_default_thinking",
    "set_default_yolo",
    "subprocess",
    "sync_all",
    "sys",
    "threading",
    "time",
    "warnings",
]

# Names exported by ``kimix.utils`` (its ``__all__``) — resolved lazily from
# the ``kimix.utils`` package so only the defining submodule is imported.
_UTILS_NAMES: frozenset[str] = frozenset({
    "SystemPromptType",
    "_default_session",
    "_should_print_usage",
    "_cli_sessions",
    "_add_cli_session",
    "_remove_cli_session",
    "_refresh_cli_sessions",
    "init",
    "_create_config",
    "_SYSTEM_PROMP",
    "get_system_prompt",
    "context_path",
    "delete_session_dir",
    "make_kaos_dir",
    "_ensure_skill_dirs",
    "_create_session_async",
    "create_session",
    "create_supervisor_session",
    "get_tool_call_errors",
    "close_session",
    "close_session_async",
    "get_cancel_event",
    "cancel_prompt",
    "get_default_session",
    "_create_default_session",
    "_create_default_session_async",
    "_print_usage",
    "print_usage",
    "clear_default_context",
    "prompt_async",
    "prompt",
    "prompt_path",
    "prompt_plan",
    "prompt_plan_async",
    "fix_error",
    "async_prompt",
    "async_fix_error",
    "compact_default_context",
    "set_ralph_loop",
    "refresh_env_from_registry",
})

# Names re-exported by ``kimix.base`` (its public names, including the lazy
# stream/wire re-exports) — resolved lazily from the ``kimix.base`` module.
_BASE_NAMES: frozenset[str] = frozenset(__all__) - _UTILS_NAMES


def __getattr__(name: str) -> Any:  # PEP 562 module-level __getattr__
    """Resolve a public name lazily from ``kimix.utils`` or ``kimix.base``."""
    if name in _UTILS_NAMES:
        import kimix.utils as _utils

        # Bypass the package attribute table: a submodule sharing the name
        # (e.g. ``kimix.utils.prompt``) would otherwise shadow the function.
        return _utils._resolve_name(name)
    if name in _BASE_NAMES:
        import kimix.base as _base

        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
