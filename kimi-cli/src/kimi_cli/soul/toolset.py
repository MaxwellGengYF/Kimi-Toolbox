from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import orjson
import sys
import time
import typing
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal, cast, overload

from kosong.tooling import (
    CallableTool,
    CallableTool2,
    HandleResult,
    TOOL_NAME_REDIRECTS,
    Tool,
    ToolError,
    ToolOk,
    Toolset,
    normalize_tool_name,
    resolve_tool_by_arguments,
    resolve_tool_name,
)
from kosong.tooling.error import (
    ToolNotFoundError,
    ToolParseError,
    ToolRuntimeError,
)
from kosong.tooling.mcp import convert_mcp_content
from kosong.utils.typing import JsonType

from kimi_cli import logger
from kimi_cli.exception import InvalidToolError, MCPRuntimeError
from kimi_cli.hooks.engine import HookEngine
from kimi_cli.safety_check import sanitize_for_tokenizer
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.utils import repair_tool_arguments
from kimi_cli.wire.types import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    LLMToolSchema,
    MCPServerSnapshot,
    MCPStatusSnapshot,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallRequest,
    ToolResult,
    ToolReturnValue,
    VideoURLPart,
)

if TYPE_CHECKING:
    import mcp
    from fastmcp.client.client import CallToolResult
    from fastmcp.mcp_config import MCPConfig

    from kimi_cli.mcp.client import MCPClient
    from kimi_cli.mcp.prompts import MCPPromptManager
    from kimi_cli.mcp.resources import MCPResourceManager
    from kimi_cli.mcp.roots import MCPRootsHandler
    from kimi_cli.soul.agent import Runtime

current_tool_call = ContextVar[ToolCall | None]("current_tool_call", default=None)

_current_session_id: ContextVar[str] = ContextVar("_current_session_id", default="")
print_tool_func = print


_DEFAULT_TOOL_OUTPUT_MAX_BYTES = 128 << 10  # 128 KiB fallback
_TOOL_OUTPUT_BYTES_PER_TOKEN = 4  # conservative UTF-8 bytes/token estimate
_TOOL_OUTPUT_CONTEXT_FRACTION = 1.0  # budget derived from total context size
_TOOL_OUTPUT_REMAINING_FRACTION = 0.9  # must stay strictly below remaining context
_TOOL_OUTPUT_ABS_MAX_BYTES = 1 << 20  # 1 MiB hard ceiling

# High similarity threshold for auto-correcting a mistyped tool name.
# Only matches at or above this cutoff will be automatically redirected
# to the real tool (with a warning appended to the output).
_AUTO_CORRECT_CUTOFF = 0.75


def _part_byte_size(part: ContentPart) -> int:
    """Return the byte size of a ContentPart for output budget checks."""
    if isinstance(part, TextPart):
        return len(part.text.encode("utf-8"))
    if isinstance(part, ThinkPart):
        return len(part.think.encode("utf-8"))
    if isinstance(part, ImageURLPart):
        return len(part.image_url.url.encode("utf-8"))
    if isinstance(part, AudioURLPart):
        return len(part.audio_url.url.encode("utf-8"))
    if isinstance(part, VideoURLPart):
        return len(part.video_url.url.encode("utf-8"))
    return 0


def _truncate_content_parts(parts: list[ContentPart], max_bytes: int) -> list[ContentPart]:
    """Return a prefix of ``parts`` whose total byte size does not exceed ``max_bytes``."""
    truncated: list[ContentPart] = []
    used = 0
    for part in parts:
        size = _part_byte_size(part)
        if used + size <= max_bytes:
            truncated.append(part)
            used += size
            continue
        room = max_bytes - used
        if room <= 0:
            break
        if isinstance(part, TextPart):
            piece = part.text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
            if piece:
                truncated.append(TextPart(text=piece))
        elif isinstance(part, ThinkPart):
            piece = part.think.encode("utf-8")[:room].decode("utf-8", errors="ignore")
            if piece:
                truncated.append(ThinkPart(think=piece))
        break
    return truncated


def set_session_id(sid: str) -> None:
    _current_session_id.set(sid)


def get_session_id() -> str:
    return _current_session_id.get()


def _get_session_id() -> str:
    return _current_session_id.get()


def get_current_tool_call_or_none() -> ToolCall | None:
    """
    Get the current tool call or None.
    Expect to be not None when called from a `__call__` method of a tool.
    """
    return current_tool_call.get()


type ToolType = CallableTool | CallableTool2[Any]
type ToolCallKey = tuple[str, str]


if TYPE_CHECKING:

    def type_check(kimi_toolset: KimiToolset):
        _: Toolset = kimi_toolset


def _collect_candidates(
    tool_name: str,
    valid_names: Iterable[str],
    redirects: dict[str, str] | None = None,
) -> list[str]:
    """
    Collect candidate real tool names for a hallucinated tool name, ordered by confidence:

    1. Redirect map match (highest confidence — human-curated).
    2. ``normalize_tool_name`` exact match (case/separator fold).
    3. Fuzzy string match candidates (from ``fuzzy_match_tool_name``).

    Deduplicates while preserving priority order.
    """
    from kosong.tooling import fuzzy_match_tool_name

    seen: set[str] = set()
    result: list[str] = []

    norm_name = normalize_tool_name(tool_name)

    # 1. Redirect map (pre-normalized keys)
    if redirects:
        redirected = redirects.get(norm_name)
        if redirected is not None and redirected in valid_names:
            if redirected not in seen:
                result.append(redirected)
                seen.add(redirected)

    # 2. normalize_tool_name exact match against valid names
    for name in valid_names:
        if normalize_tool_name(name) == norm_name and name not in seen:
            result.append(name)
            seen.add(name)

    # 3. Fuzzy matches
    fuzzy = fuzzy_match_tool_name(tool_name, valid_names, n=5, cutoff=0.5)
    for name in fuzzy:
        if name not in seen:
            result.append(name)
            seen.add(name)

    return result


# ── Pre-computed platform-aware normalized redirect map ──
# Built once at module load time instead of on every unknown-tool call
# in ``KimiToolset.handle()``.  ``sys.platform`` is immutable for the
# lifetime of the process, so the result is deterministic.
def _build_platform_redirects() -> dict[str, str]:
    """Build the platform-aware normalized redirect map."""
    _redirects = dict(TOOL_NAME_REDIRECTS)

    if sys.platform == "win32":
        # Windows: redirect all shell names to Powershell.
        _redirects.update({
            "Bash": "Powershell",
            "Shell": "Powershell",
            "Terminal": "Powershell",
            "Cmd": "Powershell",
            "Command": "Powershell",
            "Execute": "Powershell",
            "Exec": "Powershell",
            "RunCommand": "Powershell",
            "Sh": "Powershell",
            "Zsh": "Powershell",
            "ShellCommand": "Powershell",
            "BashCommand": "Powershell",
        })
    else:
        # POSIX (Linux/macOS): PowerShell references → Bash
        _redirects.update({
            "Powershell": "Bash",
            "PowerShell": "Bash",
            "Pwsh": "Bash",
            "PS": "Bash",
        })

    return {
        normalize_tool_name(k): v
        for k, v in _redirects.items()
        if k != v  # skip self-mapping entries (handled by exact match)
    }


_PLATFORM_REDIRECTS_NORM: dict[str, str] = _build_platform_redirects()


_REMINDER_TEXT_1 = (
    "\n\n<system-reminder>\n"
    "You are repeating the exact same tool call with identical parameters."
    " Please carefully analyze the previous result. If the task is not yet complete,"
    " try a different method or parameters instead of repeating the same call."
    "\n</system-reminder>"
)


def _make_reminder_text_2(tool_name: str, repeat_count: int, canonical_args: str) -> str:
    return (
        "\n\n<system-reminder>\n"
        "You have repeatedly called the same tool with identical parameters many times.\n"
        "Repeated tool call detected:\n"
        f"- tool: {tool_name}\n"
        f"- repeated_times: {repeat_count}\n"
        f"- arguments: {canonical_args}\n"
        "The previous repeated calls did not make progress. Do not call this exact same tool "
        "with the exact same arguments again.\n"
        "Carefully inspect the latest tool result and choose a different next action, "
        "different parameters, or finish the task if enough evidence has been gathered."
        "\n</system-reminder>"
    )


_REMINDER_TEXT_3 = (
    "\n\n<system-reminder>\n"
    "You are stuck in a dead end and have repeatedly made the same function call without "
    "progress.\n"
    "Stop all function calls immediately. Do not call any tool in your next response.\n"
    "In analysis, review the current execution state and identify why progress is blocked.\n"
    "Then return a text-only summary to the user that reports the current problem, what has "
    "already been tried, and what information or decision is needed next."
    "\n</system-reminder>"
)


_REPEAT_REMINDER_1_START = 3
_REPEAT_REMINDER_2_START = 8
_REPEAT_REMINDER_3_START = 12
_REPEAT_FORCE_STOP_STREAK = 16

type RepeatAction = Literal["none", "r1", "r2", "r3", "stop"]


def _build_repeat_reminder(
    streak: int, tool_name: str, canonical_args: str
) -> tuple[RepeatAction, str | None]:
    if streak >= _REPEAT_FORCE_STOP_STREAK:
        return "stop", _REMINDER_TEXT_3
    if streak >= _REPEAT_REMINDER_3_START:
        return "r3", _REMINDER_TEXT_3
    if streak >= _REPEAT_REMINDER_2_START:
        return "r2", _make_reminder_text_2(tool_name, streak, canonical_args)
    if streak >= _REPEAT_REMINDER_1_START:
        return "r1", _REMINDER_TEXT_1
    return "none", None


# Different-args tool call repetition thresholds
_DIFF_ARGS_WARN_THRESHOLDS: tuple[int, ...] = (15, 25, 35)

_DIFF_ARGS_REMINDER_TEXT_1 = (
    "\n\n<system-reminder>\n"
    "Same tool called repeatedly with different args. Reconsider your approach."
    "\n</system-reminder>"
)


def _make_diff_args_reminder_text_2(tool_name: str, call_count: int) -> str:
    return (
        "\n\n<system-reminder>\n"
        f"'{tool_name}' called {call_count} times with different args. "
        "Stop if stuck — try a different approach or finish now."
        "\n</system-reminder>"
    )


def _make_diff_args_reminder_text_3(tool_name: str, call_count: int) -> str:
    return (
        "\n\n<system-reminder>\n"
        f"'{tool_name}' called {call_count} times. Stop now. "
        "Change approach or finish the task."
        "\n</system-reminder>"
    )


def _make_diff_args_reminder(tool_name: str, call_count: int) -> str:
    """Return progressively stronger warnings based on the call count."""
    if call_count <= _DIFF_ARGS_WARN_THRESHOLDS[0]:
        return _DIFF_ARGS_REMINDER_TEXT_1
    elif call_count <= _DIFF_ARGS_WARN_THRESHOLDS[1]:
        return _make_diff_args_reminder_text_2(tool_name, call_count)
    else:
        return _make_diff_args_reminder_text_3(tool_name, call_count)


def _sort_json_value(value: object) -> object:
    if isinstance(value, list):
        return [_sort_json_value(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        value_dict = cast("dict[str, object]", value)
        return {key: _sort_json_value(value_dict[key]) for key in sorted(value_dict)}
    return value


def _canonical_tool_arguments(arguments: Any) -> str:
    try:
        return orjson.dumps(
            _sort_json_value(arguments),
        ).decode("utf-8")
    except (TypeError, ValueError):
        return str(arguments)


def _canonical_tool_arguments_text(arguments: str) -> str:
    try:
        return _canonical_tool_arguments(orjson.loads(arguments))
    except orjson.JSONDecodeError:
        return arguments


def _normalize_call_key(tool_name: str, arguments: str) -> ToolCallKey:
    return (tool_name, _canonical_tool_arguments_text(arguments))


def _append_reminder_to_return_value(
    return_value: Any, reminder_text: str = _REMINDER_TEXT_1
) -> Any:
    """Append dedup reminder text to a ToolReturnValue output."""
    from kosong.tooling import ToolReturnValue

    if not isinstance(return_value, ToolReturnValue):
        return return_value

    output = return_value.output

    if isinstance(output, str):
        new_output = output + reminder_text
    else:
        new_output = list(output)
        if new_output and isinstance(new_output[-1], TextPart):
            new_output[-1] = TextPart(text=new_output[-1].text + reminder_text)
        else:
            new_output.append(TextPart(text=reminder_text))

    return return_value.model_copy(update={"output": new_output})


@dataclass(frozen=True, slots=True)
class PendingMCPDiscovery:
    """A verbatim MCP ``tools/list`` discovery parked until a wire is available."""

    server_name: str
    tools: list[LLMToolSchema]
    enabled_names: list[str]
    collisions: list[str]


class KimiToolset:
    def __init__(
        self,
        runtime: Runtime | None = None,
        context_token_provider: Callable[[], int] | None = None,
    ) -> None:
        self._runtime = runtime
        self._context_token_provider = context_token_provider

        self._tool_dict: dict[str, ToolType] = {}
        self._hidden_tools: set[str] = set()
        self._mcp_servers: dict[str, MCPServerInfo] = {}
        self._pending_mcp_discoveries: list[PendingMCPDiscovery] = []
        self._mcp_loading_task: asyncio.Task[None] | None = None
        self._deferred_mcp_load: tuple[list[MCPConfig], Runtime] | None = None
        self._hook_engine: HookEngine = HookEngine()

        # Deduplication state
        self._previous_step_calls: list[ToolCallKey] = []
        self._current_step_calls: list[ToolCallKey] = []
        self._current_step_tasks: dict[ToolCallKey, asyncio.Task[ToolResult]] = {}
        self._seen_call_keys: set[ToolCallKey] = set()
        self._consecutive_key: ToolCallKey | None = None
        self._consecutive_count: int = 0
        self._step_closed: bool = False
        self._dedup_triggered: bool = False
        self._force_stop_turn: bool = False

        # "Different-args" per-tool call tracking (relaxed limitation)
        self._tool_call_counts: dict[str, int] = {}  # tool_name → total calls this turn
        self._tool_warned_at: dict[str, set[int]] = {}  # thresholds already warned
        self._turn_tool_warning_issued: bool = False  # avoid flooding multiple tools
        self._turn_id: str = ""
        self._step_no: int = 0

    def _hook_cwd(self) -> str:
        """Return the cwd to report in lifecycle hook events."""
        if self._runtime is not None:
            return str(self._runtime.session.work_dir)
        return str(Path.cwd())

    def set_hook_engine(self, engine: HookEngine) -> None:
        self._hook_engine = engine

    def set_context_token_provider(self, provider: Callable[[], int] | None) -> None:
        """Set a callback that returns the current context token count."""
        self._context_token_provider = provider

    def _get_max_output_bytes(self) -> int:
        """Return the per-tool output byte budget.

        The budget is the more restrictive of:
          - a fraction of the model's total context size, and
          - a fraction of the currently remaining context tokens.

        Falls back to `_DEFAULT_TOOL_OUTPUT_MAX_BYTES` when no runtime/LLM is available.
        """
        llm = getattr(self._runtime, "llm", None)
        max_context = getattr(llm, "max_context_size", None)
        if not isinstance(max_context, int) or max_context <= 0:
            return _DEFAULT_TOOL_OUTPUT_MAX_BYTES

        # Budget derived from total context size
        total_budget = int(
            max_context * _TOOL_OUTPUT_BYTES_PER_TOKEN * _TOOL_OUTPUT_CONTEXT_FRACTION
        )

        # Budget derived from remaining context (safety constraint)
        current_tokens = 0
        if self._context_token_provider is not None:
            current_tokens = self._context_token_provider()
        remaining_tokens = max(0, max_context - current_tokens)
        remaining_budget = int(
            remaining_tokens * _TOOL_OUTPUT_BYTES_PER_TOKEN * _TOOL_OUTPUT_REMAINING_FRACTION
        )

        # Enforce: max_bytes is less than remaining context usage
        max_bytes = min(total_budget, remaining_budget)
        return max(0, min(max_bytes, _TOOL_OUTPUT_ABS_MAX_BYTES))

    def add(self, tool: ToolType) -> None:
        self._tool_dict[tool.name] = tool

    def hide(self, tool_name: str) -> bool:
        """Hide a tool from the LLM tool list. Returns True if the tool exists."""
        if tool_name in self._tool_dict:
            self._hidden_tools.add(tool_name)
            return True
        return False

    def unhide(self, tool_name: str) -> None:
        """Restore a hidden tool to the LLM tool list."""
        self._hidden_tools.discard(tool_name)

    @overload
    def find(self, tool_name_or_type: str) -> ToolType | None: ...
    @overload
    def find[T: ToolType](self, tool_name_or_type: type[T]) -> T | None: ...
    def find(self, tool_name_or_type: str | type[ToolType]) -> ToolType | None:
        if isinstance(tool_name_or_type, str):
            return self._tool_dict.get(tool_name_or_type)
        else:
            for tool in self._tool_dict.values():
                if isinstance(tool, tool_name_or_type):
                    return tool
        return None

    @property
    def tools(self) -> list[Tool]:
        return [
            tool.base for tool in self._tool_dict.values() if tool.name not in self._hidden_tools
        ]

    def begin_step(
        self,
        previous_calls: list[tuple[str, str]],
        *,
        step_no: int = 0,
        turn_id: str = "",
    ) -> None:
        """Called before each step to set up deduplication state.

        Args:
            previous_calls: Tool calls from the previous step.
            step_no: The current step number (1-based).
            turn_id: The current turn identifier.
        """
        self._previous_step_calls = [
            _normalize_call_key(tool_name, arguments) for tool_name, arguments in previous_calls
        ]
        self._current_step_calls = []
        self._current_step_tasks = {}
        self._step_closed = False
        self._dedup_triggered = False
        self._force_stop_turn = False
        self._step_no = step_no

        # Detect new turn and reset per-tool different-args tracking
        if turn_id and turn_id != self._turn_id:
            self._tool_call_counts.clear()
            self._tool_warned_at.clear()
        self._turn_tool_warning_issued = False  # Reset per-step

        self._turn_id = turn_id
        if not self._previous_step_calls:
            self._seen_call_keys = set()
            self._consecutive_key = None
            self._consecutive_count = 0
            if not turn_id:
                # No turn id provided: rely on previous_calls emptiness
                self._tool_call_counts.clear()
                self._tool_warned_at.clear()
        else:
            self._seen_call_keys.update(self._previous_step_calls)
            if self._consecutive_key is None and self._consecutive_count == 0:
                self._advance_consecutive_streak(self._previous_step_calls)

    def end_step(self) -> list[tuple[str, str]]:
        """Called after each step to capture the calls made in this step."""
        if not self._step_closed:
            self._advance_consecutive_streak(self._current_step_calls)
            self._seen_call_keys.update(self._current_step_calls)
            self._step_closed = True
        return list(self._current_step_calls)

    def _advance_consecutive_streak(self, calls: list[ToolCallKey]) -> None:
        for call_key in calls:
            if call_key == self._consecutive_key:
                self._consecutive_count += 1
            else:
                self._consecutive_key = call_key
                self._consecutive_count = 1

    def _projected_streak_for_call(self, call_index: int) -> int:
        consecutive_key = self._consecutive_key
        consecutive_count = self._consecutive_count
        for call_key in self._current_step_calls[: call_index + 1]:
            if call_key == consecutive_key:
                consecutive_count += 1
            else:
                consecutive_key = call_key
                consecutive_count = 1
        return consecutive_count

    @property
    def dedup_triggered(self) -> bool:
        """Whether a cross-step duplicate was blocked in the current step."""
        return self._dedup_triggered

    @property
    def force_stop_turn(self) -> bool:
        return self._force_stop_turn

    def handle(self, tool_call: ToolCall) -> HandleResult:
        token = current_tool_call.set(tool_call)
        try:
            tool_name = tool_call.function.name
            warning_text: str | None = None

            from kosong.utils.jsonx import loads_relaxed

            try:
                arguments: JsonType = loads_relaxed(tool_call.function.arguments or "{}")
            except (orjson.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Tool call JSON parse error: {tool_name} (call_id={call_id}): {error}",
                    tool_name=tool_name,
                    call_id=tool_call.id,
                    error=e,
                )
                return ToolResult(tool_call_id=tool_call.id, return_value=ToolParseError(str(e)))

            if not isinstance(arguments, dict):
                arguments = {}

            if tool_name not in self._tool_dict:
                # Step 1: Collect candidates from cached redirect map + fuzzy matching.
                candidates = _collect_candidates(
                    tool_name, self._tool_dict.keys(), _PLATFORM_REDIRECTS_NORM
                )

                # Step 2: Try argument-based disambiguation.
                arg_resolution = None
                if arguments and candidates:
                    arg_resolution = resolve_tool_by_arguments(
                        tool_name, arguments, candidates, self._tool_dict
                    )

                if arg_resolution is not None:
                    # High-confidence resolution from argument fit.
                    resolution = arg_resolution
                else:
                    # Fall back to name-only resolution (redirects + fuzzy).
                    resolution = resolve_tool_name(
                        tool_name,
                        self._tool_dict.keys(),
                        auto_correct_cutoff=_AUTO_CORRECT_CUTOFF,
                        redirects=_PLATFORM_REDIRECTS_NORM,
                    )

                if resolution.name is None:
                    # No close enough match — return suggestions error.
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        return_value=ToolNotFoundError(tool_name, resolution.suggestions),
                    )
                if resolution.corrected:
                    logger.info(
                        "Auto-corrected tool name: {original} -> {target}",
                        original=tool_name,
                        target=resolution.name,
                    )
                    warning_text = (
                        f"\n\n<system-warning>\n"
                        f"Tool `{tool_name}` was not found. "
                        f"Auto-corrected to `{resolution.name}`.\n"
                        f"</system-warning>"
                    )
                tool_name = resolution.name

            canonical_args = _canonical_tool_arguments(arguments)
            call_key = (tool_name, canonical_args)
            call_index = len(self._current_step_calls)
            self._current_step_calls.append(call_key)

            # Per-tool different-args call counting (relaxed limitation)
            self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1
            call_count = self._tool_call_counts[tool_name]

            # Same-step dedup: wait for the original task and copy its result.
            if call_key in self._current_step_tasks:
                original_task = self._current_step_tasks[call_key]

                async def _await_dup() -> ToolResult:
                    original_result = await original_task
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        return_value=original_result.return_value,
                    )

                return asyncio.create_task(_await_dup())

            is_cross_step_dup = call_key in self._seen_call_keys
            reminder_text: str | None = None
            if is_cross_step_dup:
                repeat_count = self._projected_streak_for_call(call_index)
                action, reminder_text = _build_repeat_reminder(
                    repeat_count, tool_name, canonical_args
                )
                self._dedup_triggered = True
                if action == "stop":
                    self._force_stop_turn = True

            # Different-args per-tool overuse check (relaxed limitation)
            diff_args_reminder_text: str | None = None
            if not is_cross_step_dup and call_count in _DIFF_ARGS_WARN_THRESHOLDS:
                warned_at = self._tool_warned_at.setdefault(tool_name, set())
                if call_count not in warned_at and not self._turn_tool_warning_issued:
                    warned_at.add(call_count)
                    self._turn_tool_warning_issued = True
                    diff_args_reminder_text = _make_diff_args_reminder(tool_name, call_count)

            # Merge reminder texts if both are set
            if reminder_text is not None and diff_args_reminder_text is not None:
                reminder_text = reminder_text + diff_args_reminder_text
            elif diff_args_reminder_text is not None:
                reminder_text = diff_args_reminder_text

            tool = self._tool_dict[tool_name]

            async def _call():
                tool_input_dict = arguments if isinstance(arguments, dict) else {}

                # --- PreToolUse ---
                from kimi_cli.hooks import events

                results = await self._hook_engine.trigger(
                    "PreToolUse",
                    matcher_value=tool_name,
                    input_data=events.pre_tool_use(
                        session_id=_get_session_id(),
                        cwd=self._hook_cwd(),
                        tool_name=tool_name,
                        tool_input=tool_input_dict,
                        tool_call_id=tool_call.id,
                    ),
                )
                for result in results:
                    if result.action == "block":
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            return_value=ToolError(
                                message=result.reason or "Blocked by PreToolUse hook",
                                brief="Hook blocked",
                            ),
                        )

                # --- Execute tool ---
                t0 = time.monotonic()
                try:
                    repaired_arguments = repair_tool_arguments(tool.params, arguments)
                    ret = await tool.call(repaired_arguments)
                    if isinstance(ret.output, str):
                        ret.output = sanitize_for_tokenizer(ret.output)
                    elif isinstance(ret.output, list):
                        sanitized_parts: list[ContentPart] = []
                        for part in ret.output:
                            if isinstance(part, TextPart):
                                cleaned = sanitize_for_tokenizer(part.text)
                                if cleaned:
                                    part.text = cleaned
                                    sanitized_parts.append(part)
                            elif isinstance(part, ThinkPart):
                                cleaned = sanitize_for_tokenizer(part.think)
                                if cleaned:
                                    part.think = cleaned
                                    sanitized_parts.append(part)
                            else:
                                sanitized_parts.append(part)
                        ret.output = sanitized_parts
                    max_bytes = self._get_max_output_bytes()
                    if isinstance(ret.output, str):
                        output_bytes = ret.output.encode("utf-8")
                        if len(output_bytes) > max_bytes:
                            ret = ToolError(
                                message=(
                                    f"Tool output exceeded the maximum allowed size "
                                    f"({len(output_bytes)} bytes; limit {max_bytes} bytes). "
                                    f"The result has been truncated."
                                ),
                                brief="Output too large",
                                output=output_bytes[:max_bytes].decode("utf-8", errors="ignore"),
                            )
                    else:
                        # Handle list[ContentPart] or single ContentPart
                        parts = ret.output if isinstance(ret.output, list) else [ret.output]
                        total_bytes = sum(_part_byte_size(p) for p in parts)
                        if total_bytes > max_bytes:
                            ret = ToolError(
                                message=(
                                    f"Tool output exceeded the maximum allowed size "
                                    f"({total_bytes} bytes; limit {max_bytes} bytes). "
                                    f"The result has been truncated."
                                ),
                                brief="Output too large",
                                output=_truncate_content_parts(parts, max_bytes),
                            )
                except (TypeError, ValueError) as e:
                    if "dictionary update sequence" in str(e) or "argument" in str(e).lower():
                        logger.exception(
                            "Tool argument coercion failed: {tool_name} (call_id={call_id})",
                            tool_name=tool_name,
                            call_id=tool_call.id,
                        )
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            return_value=ToolValidateError(
                                f"Invalid arguments for tool `{tool_name}`: {e}"
                            ),
                        )
                    raise
                except Exception as e:
                    tool_elapsed = time.monotonic() - t0
                    logger.exception(
                        "Tool execution failed: {tool_name} (call_id={call_id})",
                        tool_name=tool_name,
                        call_id=tool_call.id,
                    )
                    # --- PostToolUseFailure (fire-and-forget) ---
                    _hook_task = asyncio.create_task(
                        self._hook_engine.trigger(
                            "PostToolUseFailure",
                            matcher_value=tool_name,
                            input_data=events.post_tool_use_failure(
                                session_id=_get_session_id(),
                                cwd=self._hook_cwd(),
                                tool_name=tool_name,
                                tool_input=tool_input_dict,
                                error=str(e),
                                tool_call_id=tool_call.id,
                            ),
                        )
                    )
                    _hook_task.add_done_callback(
                        lambda t: t.exception() if not t.cancelled() else None
                    )
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        return_value=ToolRuntimeError(str(e)),
                    )

                tool_elapsed = time.monotonic() - t0
                logger.info(
                    "Tool {tool_name} completed in {elapsed:.1f}s (call_id={call_id})",
                    tool_name=tool_name,
                    elapsed=tool_elapsed,
                    call_id=tool_call.id,
                )

                # --- PostToolUse (fire-and-forget) ---
                _hook_task = asyncio.create_task(
                    self._hook_engine.trigger(
                        "PostToolUse",
                        matcher_value=tool_name,
                        input_data=events.post_tool_use(
                            session_id=_get_session_id(),
                            cwd=self._hook_cwd(),
                            tool_name=tool_name,
                            tool_input=tool_input_dict,
                            tool_output=str(ret)[:2000],
                            tool_call_id=tool_call.id,
                        ),
                    )
                )
                _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

                return ToolResult(tool_call_id=tool_call.id, return_value=ret)

            task = asyncio.create_task(_call())

            # Combine warning (auto-correct) and reminder (dedup) texts
            append_text = ""
            if warning_text is not None:
                append_text += warning_text
            if reminder_text is not None:
                append_text += reminder_text

            if append_text:
                async def _wrap_with_text(
                    inner_task: asyncio.Task[ToolResult],
                    text: str,
                ) -> ToolResult:
                    tr = await inner_task
                    return ToolResult(
                        tool_call_id=tr.tool_call_id,
                        return_value=_append_reminder_to_return_value(tr.return_value, text),
                    )

                task = asyncio.create_task(_wrap_with_text(task, append_text))

            self._current_step_tasks[call_key] = task
            return task
        finally:
            current_tool_call.reset(token)

    def register_external_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if name in self._tool_dict:
            existing = self._tool_dict[name]
            if not isinstance(existing, WireExternalTool):
                return False, "tool name conflicts with existing tool"
        try:
            tool = WireExternalTool(
                name=name,
                description=description,
                parameters=parameters,
            )
        except Exception as e:
            return False, str(e)
        self.add(tool)
        return True, None

    @property
    def mcp_servers(self) -> dict[str, MCPServerInfo]:
        """Get MCP servers info."""
        return self._mcp_servers

    def mcp_status_snapshot(self) -> MCPStatusSnapshot | None:
        """Return a read-only snapshot of current MCP startup state."""
        if not self._mcp_servers:
            return None

        servers = tuple(
            MCPServerSnapshot(
                name=name,
                status=info.status,
                tools=tuple(tool.name for tool in info.tools),
                resources=info.resources,
                prompts=info.prompts,
            )
            for name, info in self._mcp_servers.items()
        )
        return MCPStatusSnapshot(
            loading=self.has_pending_mcp_tools(),
            connected=sum(1 for server in servers if server.status == "connected"),
            total=len(servers),
            tools=sum(len(server.tools) for server in servers),
            servers=servers,
        )

    def defer_mcp_tool_loading(self, mcp_configs: list[MCPConfig], runtime: Runtime) -> None:
        """Store MCP configs for a later background startup."""
        self._deferred_mcp_load = (list(mcp_configs), runtime)

    def has_deferred_mcp_tools(self) -> bool:
        """Return True when MCP loading is configured but has not started yet."""
        return self._deferred_mcp_load is not None

    async def start_deferred_mcp_tool_loading(self) -> bool:
        """Start any deferred MCP loading in the background."""
        if self._deferred_mcp_load is None:
            return False
        if self._mcp_loading_task is not None or self._mcp_servers:
            self._deferred_mcp_load = None
            return False

        mcp_configs, runtime = self._deferred_mcp_load
        self._deferred_mcp_load = None
        await self.load_mcp_tools(mcp_configs, runtime, in_background=True)
        return True

    def load_tools(self, tool_paths: list[str], dependencies: dict[type[Any], Any]) -> None:
        """
        Load tools from paths like `kimi_cli.tools.shell:Shell`.

        Raises:
            InvalidToolError(KimiCLIException, ValueError): When any tool cannot be loaded.
        """

        good_tools: list[str] = []
        bad_tools: list[str] = []

        for tool_path in tool_paths:
            try:
                tool = self._load_tool(tool_path, dependencies)
            except SkipThisTool:
                logger.info("Skipping tool: {tool_path}", tool_path=tool_path)
                continue
            if tool:
                self.add(tool)
                good_tools.append(tool_path)
            else:
                bad_tools.append(tool_path)
        logger.info("Loaded tools: {good_tools}", good_tools=good_tools)
        if bad_tools:
            raise InvalidToolError(f"Invalid tools: {bad_tools}")

    @staticmethod
    def _load_tool(tool_path: str, dependencies: dict[type[Any], Any]) -> ToolType | None:
        logger.debug("Loading tool: {tool_path}", tool_path=tool_path)
        module_name, class_name = tool_path.rsplit(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print_tool_func(str(e))
            logger.warning(
                "Tool module import failed: {module_name}: {error}",
                module_name=module_name,
                error=e,
            )
            return None
        tool_cls = getattr(module, class_name, None)
        if tool_cls is None:
            logger.warning(
                "Tool class not found: {class_name} in {module_name}",
                class_name=class_name,
                module_name=module_name,
            )
            return None
        args: list[Any] = []
        if "__init__" in tool_cls.__dict__:
            # the tool class overrides the `__init__` of base class
            try:
                type_hints = typing.get_type_hints(tool_cls.__init__)
            except Exception:
                type_hints = {}
            for param in inspect.signature(tool_cls).parameters.values():
                if param.kind == inspect.Parameter.KEYWORD_ONLY:
                    # once we encounter a keyword-only parameter, we stop injecting dependencies
                    break
                # all positional parameters should be dependencies to be injected
                annotation = type_hints.get(param.name, param.annotation)
                if annotation not in dependencies:
                    # Handle Optional[X] / X | None
                    origin = typing.get_origin(annotation)
                    args_ = typing.get_args(annotation)
                    if origin is not None and type(None) in args_:
                        non_none = [a for a in args_ if a is not type(None)]
                        if len(non_none) == 1:
                            annotation = non_none[0]
                if annotation not in dependencies:
                    raise ValueError(f"Tool dependency not found: {param.annotation}")
                args.append(dependencies[annotation])
        return tool_cls(*args)

    async def load_mcp_tools(
        self, mcp_configs: list[MCPConfig], runtime: Runtime, in_background: bool = True
    ) -> None:
        """
        Load MCP tools from specified MCP configs.

        Raises:
            MCPRuntimeError(KimiCLIException, RuntimeError): When any MCP server cannot be
                connected.
        """
        from fastmcp.mcp_config import MCPConfig, RemoteMCPServer

        from kimi_cli.mcp.client import MCPClient
        from kimi_cli.mcp.prompts import MCPPromptManager
        from kimi_cli.mcp.resources import MCPResourceManager
        from kimi_cli.mcp.roots import MCPRootsHandler
        from kimi_cli.mcp_oauth import create_mcp_oauth, has_mcp_oauth_tokens

        async def _check_oauth_tokens(server_url: str) -> bool:
            """Check if OAuth tokens exist for the server."""
            return await has_mcp_oauth_tokens(server_url)

        def _toast_mcp(message: str) -> None:
            pass

        def _mark_oauth_unauthorized(server_name: str) -> None:
            logger.warning(
                "Skipping OAuth MCP server '{server_name}': not authorized. "
                "Run 'kimi mcp auth {server_name}' first.",
                server_name=server_name,
            )
            self._mcp_servers[server_name] = MCPServerInfo(
                status="unauthorized", client=None, tools=[]
            )

        async def _connect_server(
            server_name: str, server_info: MCPServerInfo
        ) -> tuple[str, Exception | None]:
            if server_info.status != "pending":
                return server_name, None

            server_info.status = "connecting"
            try:
                assert server_info.client is not None
                client = server_info.client.inner
                async with client:
                    tools = await client.list_tools()
                    # Capture the verbatim tools/list result for the request trace.
                    discovered_schemas = [
                        LLMToolSchema(
                            name=tool.name,
                            description=tool.description or "",
                            parameters=dict(tool.inputSchema),
                        )
                        for tool in tools
                    ]
                    for tool in tools:
                        server_info.tools.append(
                            MCPTool(server_name, tool, server_info.client, runtime=runtime)
                        )

                    # Best-effort resource/prompt discovery
                    try:
                        resources = await MCPResourceManager.list_resources(client)
                        server_info.resources = tuple(str(r.uri) for r in resources[0])
                    except Exception as exc:
                        logger.debug(
                            "MCP server '{server_name}' does not expose resources: {error}",
                            server_name=server_name,
                            error=exc,
                        )

                    try:
                        prompts = await MCPPromptManager.list_prompts(client)
                        server_info.prompts = tuple(p.name for p in prompts)
                    except Exception as exc:
                        logger.debug(
                            "MCP server '{server_name}' does not expose prompts: {error}",
                            server_name=server_name,
                            error=exc,
                        )

                enabled_names: list[str] = []
                collisions: list[str] = []
                for tool in server_info.tools:
                    if tool.name in self._tool_dict:
                        # Name clash with an already-registered tool: skip it
                        # (previously an implicit overwrite; now explicit).
                        collisions.append(tool.name)
                        continue
                    self.add(tool)
                    enabled_names.append(tool.name)
                server_info.tools = [
                    tool for tool in server_info.tools if tool.name not in collisions
                ]
                # Park the discovery: MCP connect may run in a background task
                # where no wire is active. The soul drains this at loop start.
                self._pending_mcp_discoveries.append(
                    PendingMCPDiscovery(
                        server_name=server_name,
                        tools=discovered_schemas,
                        enabled_names=enabled_names,
                        collisions=collisions,
                    )
                )

                server_info.status = "connected"
                logger.info("Connected MCP server: {server_name}", server_name=server_name)
                return server_name, None
            except Exception as e:
                logger.error(
                    "Failed to connect MCP server: {server_name}, error: {error}",
                    server_name=server_name,
                    error=e,
                )
                server_info.status = "failed"
                return server_name, e

        async def _connect():
            _toast_mcp("connecting to mcp servers...")
            tasks = [
                asyncio.create_task(_connect_server(server_name, server_info))
                for server_name, server_info in self._mcp_servers.items()
                if server_info.status == "pending"
            ]
            results = await asyncio.gather(*tasks) if tasks else []
            failed_servers = {name: error for name, error in results if error is not None}

            for mcp_config in mcp_configs:
                # Skip empty MCP configs (no servers defined)
                if not mcp_config.mcpServers:
                    logger.debug("Skipping empty MCP config: {mcp_config}", mcp_config=mcp_config)
                    continue

            if failed_servers:
                _toast_mcp("mcp connection failed")
                raise MCPRuntimeError(f"Failed to connect MCP servers: {failed_servers}")
            if any(info.status == "unauthorized" for info in self._mcp_servers.values()):
                _toast_mcp("mcp authorization needed")
            else:
                _toast_mcp("mcp servers connected")

        for mcp_config in mcp_configs:
            if not mcp_config.mcpServers:
                logger.debug("Skipping empty MCP config: {mcp_config}", mcp_config=mcp_config)
                continue

            for server_name, server_config in mcp_config.mcpServers.items():
                if isinstance(server_config, RemoteMCPServer) and server_config.auth == "oauth":
                    if not await _check_oauth_tokens(server_config.url):
                        _mark_oauth_unauthorized(server_name)
                        continue
                    try:
                        auth = create_mcp_oauth(server_config.url)
                    except Exception as e:
                        logger.debug(
                            "Failed to create MCP OAuth storage for {server_name}: {error}",
                            server_name=server_name,
                            error=e,
                        )
                        _mark_oauth_unauthorized(server_name)
                        continue
                    server_config = server_config.model_copy(update={"auth": auth})

                client = MCPClient(
                    MCPConfig(mcpServers={server_name: server_config}),
                    name=server_name,
                    timeout_ms=runtime.config.mcp.client.tool_call_timeout_ms,
                    roots_handler=MCPRootsHandler(work_dir=runtime.session.work_dir),
                )
                self._mcp_servers[server_name] = MCPServerInfo(
                    status="pending", client=client, tools=[]
                )

        if in_background:
            self._mcp_loading_task = asyncio.create_task(_connect())
        else:
            await _connect()

    def drain_pending_mcp_discoveries(self) -> list[PendingMCPDiscovery]:
        """Pop all parked MCP tool discoveries (see `PendingMCPDiscovery`)."""
        drained = self._pending_mcp_discoveries
        self._pending_mcp_discoveries = []
        return drained

    def has_pending_mcp_tools(self) -> bool:
        """Return True if the background MCP tool-loading task is still running."""
        return self._mcp_loading_task is not None and not self._mcp_loading_task.done()

    async def wait_for_mcp_tools(self) -> None:
        """Wait for background MCP tool loading to finish."""
        task = self._mcp_loading_task
        if not task:
            return
        try:
            await task
        finally:
            if self._mcp_loading_task is task and task.done():
                self._mcp_loading_task = None

    async def cleanup(self) -> None:
        """Cleanup any resources held by the toolset."""
        self._deferred_mcp_load = None
        if self._mcp_loading_task:
            self._mcp_loading_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._mcp_loading_task
        for server_info in self._mcp_servers.values():
            if server_info.client is not None:
                try:
                    await server_info.client.close()
                except Exception:
                    logger.warning("Failed to close MCP client", exc_info=True)


@dataclass(slots=True)
class MCPServerInfo:
    status: Literal["pending", "connecting", "connected", "failed", "unauthorized"]
    client: MCPClient | None
    tools: list[MCPTool[Any]]
    resources: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()


class MCPTool(CallableTool):
    def __init__(
        self,
        server_name: str,
        mcp_tool: mcp.Tool,
        client: MCPClient,
        *,
        runtime: Runtime,
        **kwargs: Any,
    ):
        super().__init__(
            name=mcp_tool.name,
            description=(
                f"This is an MCP (Model Context Protocol) tool from MCP server `{server_name}`.\n\n"
                f"{mcp_tool.description or 'No description provided.'}"
            ),
            parameters=mcp_tool.inputSchema,
            **kwargs,
        )
        self._mcp_tool = mcp_tool
        self._client = client
        self._runtime = runtime
        self._timeout = timedelta(milliseconds=runtime.config.mcp.client.tool_call_timeout_ms)
        self._action_name = f"mcp:{mcp_tool.name}"

    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        description = f"Call MCP tool `{self._mcp_tool.name}`."
        result = await self._runtime.approval.request(self.name, self._action_name, description)
        if not result:
            return result.rejection_error()

        try:
            result = await self._client.call_tool(
                self._mcp_tool.name,
                kwargs,
                timeout_ms=int(self._timeout.total_seconds() * 1000),
            )
            if result.is_error:
                logger.warning(
                    "MCP tool returned error: {tool_name}: {content}",
                    tool_name=self._mcp_tool.name,
                    content=[str(p) for p in result.content][:3],
                )
            return convert_mcp_tool_result(result)
        except Exception as e:
            # fastmcp raises `RuntimeError` on timeout and we cannot tell it from other errors
            exc_msg = str(e).lower()
            if "timeout" in exc_msg or "timed out" in exc_msg:
                logger.warning(
                    "MCP tool call timed out: {tool_name}: {error}",
                    tool_name=self._mcp_tool.name,
                    error=e,
                )
                return ToolError(
                    message=(
                        f"Timeout while calling MCP tool `{self._mcp_tool.name}`. "
                        "You may explain to the user that the timeout config is set too low."
                    ),
                    brief="Timeout",
                )
            logger.error(
                "MCP tool call failed: {tool_name}: {error}",
                tool_name=self._mcp_tool.name,
                error=e,
            )
            raise


class WireExternalTool(CallableTool):
    def __init__(self, *, name: str, description: str, parameters: dict[str, Any]) -> None:
        super().__init__(
            name=name,
            description=description or "No description provided.",
            parameters=parameters,
        )

    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        tool_call = get_current_tool_call_or_none()
        if tool_call is None:
            return ToolError(
                message="External tool calls must be invoked from a tool call context.",
                brief="Invalid tool call",
            )

        from kimi_cli.soul import get_wire_or_none

        wire = get_wire_or_none()
        if wire is None:
            logger.error(
                "Wire is not available for external tool call: {tool_name}", tool_name=self.name
            )
            return ToolError(
                message="Wire is not available for external tool calls.",
                brief="Wire unavailable",
            )

        external_tool_call = ToolCallRequest.from_tool_call(tool_call)
        wire.soul_side.send(external_tool_call)
        try:
            return await external_tool_call.wait()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("External tool call failed: {tool_name}:", tool_name=self.name)
            return ToolError(
                message=f"External tool call failed: {e}",
                brief="External tool error",
            )


# Maximum characters allowed in MCP tool output before truncation.
# Built-in tools use 50K via ToolResultBuilder; MCP gets a wider budget because
# multi-part results (e.g. text + image) are common, but still needs a cap to
# prevent context overflow from tools like Playwright that return full DOMs.
MCP_MAX_OUTPUT_CHARS = 100_000


def _media_part_size(part: ContentPart) -> int | None:
    """Return the payload size of a media part, or ``None`` for non-media parts."""
    if isinstance(part, ImageURLPart):
        return len(part.image_url.url)
    if isinstance(part, AudioURLPart):
        return len(part.audio_url.url)
    if isinstance(part, VideoURLPart):
        return len(part.video_url.url)
    return None


def convert_mcp_tool_result(result: CallToolResult) -> ToolReturnValue:
    """Convert MCP tool result to kosong tool return value.

    All content — text *and* inline media (``data:`` URLs) — is subject to
    a shared *MCP_MAX_OUTPUT_CHARS* character budget.  Text parts are
    truncated in-place; media parts that exceed the remaining budget are
    dropped and replaced with a descriptive placeholder.

    Unsupported content types are caught and replaced with a ``TextPart``
    placeholder instead of crashing the turn.
    """
    content: list[ContentPart] = []
    char_budget = MCP_MAX_OUTPUT_CHARS
    truncated = False

    for part in result.content:
        try:
            converted = convert_mcp_content(part)
        except ValueError as exc:
            logger.warning(
                "Skipping unsupported MCP content part: {error}",
                error=exc,
            )
            converted = TextPart(text=f"[Unsupported content: {exc}]")

        # --- budget enforcement (text) ---
        if isinstance(converted, TextPart):
            if char_budget <= 0:
                truncated = True
                continue
            if len(converted.text) > char_budget:
                converted = TextPart(text=converted.text[:char_budget])
                truncated = True
            char_budget -= len(converted.text)
            content.append(converted)
            continue

        # --- budget enforcement (media: image / audio / video) ---
        media_size = _media_part_size(converted)
        if media_size is not None:
            if media_size > char_budget:
                truncated = True
                continue  # drop the oversized media part silently
            char_budget -= media_size
            content.append(converted)
            continue

        # Unknown ContentPart subclass — pass through without budget impact
        content.append(converted)

    if truncated:
        content.append(
            TextPart(
                text=(
                    f"\n\n[Output truncated: exceeded {MCP_MAX_OUTPUT_CHARS} character limit. "
                    "Use pagination or more specific queries to get remaining content.]"
                )
            )
        )

    if result.is_error:
        return ToolError(
            output=content,
            message="Tool returned an error. The output may be error message or incomplete output",
            brief="",
        )
    else:
        return ToolOk(output=content)
