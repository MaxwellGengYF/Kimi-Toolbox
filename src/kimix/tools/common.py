import asyncio
import atexit
import codecs
import functools
import io
import os
import regex as re
import shutil
import textwrap
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from kimi_agent_sdk import ToolReturnValue

from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time: the runtime environment is stable, so
# ``_native_get_module("stream")`` can never change while the process lives.
_NATIVE_STREAM = _native_get_module("stream")


# ── Standard Error Helpers ────────────────────────────────────────────────


class ErrorKind(str, Enum):
    """Standardized error kinds for tool errors."""

    INVALID_PARAMETER = "invalid_parameter"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    EXECUTION_FAILED = "execution_failed"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    CONTEXT_FULL = "context_full"
    UNSUPPORTED = "unsupported"


# Common suggested actions for frequent error scenarios.
_SUGGESTED_ACTIONS: dict[str, str] = {
    "file_not_found": "Check the file path with `Glob` or verify with `Bash ls -la`.",
    "timeout": "Increase the `timeout` parameter or simplify the command/query.",
    "permission_denied": "Check file permissions with `Bash ls -la`.",
    "network_error": "Check network connectivity and try again. The server may be unreachable.",
    "rate_limited": "Wait a moment and retry with a longer delay between calls.",
    "invalid_parameter": "Review the parameter description and provide a valid value.",
    "execution_failed": "Check the command syntax and try again. Use simpler arguments.",
    "context_full": "Call `Compact` with mode='auto' or mode='aggressive' to free up context.",
    "unsupported": "Use an alternative tool or method that supports this operation.",
}


def make_tool_error(
    kind: ErrorKind,
    *,
    user_message: str = "",
    llm_message: str = "",
    recoverable: bool = True,
    suggested_action: str = "",
    output: str = "",
) -> ToolReturnValue:
    """Build a standardized error ToolReturnValue.

    Args:
        kind: The error category.
        user_message: Short human-readable error summary (appears in brief).
        llm_message: Detailed message for the LLM.
        recoverable: Whether the error is recoverable by the LLM.
        suggested_action: Specific action the LLM can take. Falls back to
            the default for the error kind if empty.
        output: Raw output text.
    """
    action = suggested_action or _SUGGESTED_ACTIONS.get(kind.value, "")
    brief_parts = [f"[{kind.value}] {user_message}"] if user_message else [f"[{kind.value}]"]
    message_parts = [llm_message] if llm_message else []
    if action:
        brief_parts.append(action)
        message_parts.append(f"Suggested action: {action}")

    return ToolReturnValue(
        is_error=True,
        output=output or "",
        message=" ".join(message_parts) if message_parts else (user_message or kind.value),
        brief=" | ".join(brief_parts) if len(brief_parts) > 1 else brief_parts[0],
        extras={
            "error_kind": kind.value,
            "recoverable": recoverable,
            "suggested_action": action,
        },
    )

import queue
import subprocess
import threading
from typing import TYPE_CHECKING

# Common error keywords for detecting error lines in process output
_ERROR_KEYWORDS = [
    "error", "exception", "traceback", "failed", "failure",
    "fatal", "panic", "abort", "assertion", "undefined",
    "syntaxerror", "typeerror", "valueerror", "keyerror",
    "importerror", "modulenotfounderror", "attributeerror",
    "nameerror", "runtimeerror", "oserror", "ioerror",
    "zerodivisionerror", "indexerror", "memoryerror",
    "recursionerror", "unboundlocalerror", "referenceerror",
    "permission denied", "access denied", "not found",
    "cannot find", "does not exist", "no such file",
    "connection refused", "timeout", "unhandled",
]

_ERROR_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _ERROR_KEYWORDS) + r')\b',
    re.IGNORECASE
)


def _find_error_line_index(output: str) -> int | None:
    """Find the 1-based line index of the first line containing a common error keyword."""
    for idx, line in enumerate(output.splitlines(), start=1):
        if _ERROR_PATTERN.search(line):
            return idx
    return None


# RTK (token killer) supported top-level commands.  Subcommands are handled by
# RTK itself, so matching the executable name is sufficient.
_RTK_KNOWN_COMMANDS: frozenset[str] = frozenset(
    [
        # File
        "ls",
        "tree",
        "read",
        "smart",
        "find",
        "grep",
        "rg",
        "diff",
        "wc",
        "json",
        "log",
        "env",
        "deps",
        # Git
        "git",
        # Rust
        "cargo",
        # JS/TS
        "vitest",
        "jest",
        "tsc",
        "lint",
        "prettier",
        "format",
        "next",
        "prisma",
        "playwright",
        "npm",
        "npx",
        "pnpm",
        # Python
        "pytest",
        "ruff",
        "mypy",
        "pip",
        "uv",
        # Go
        "go",
        "golangci-lint",
        # Ruby
        "rspec",
        "rubocop",
        "rake",
        # .NET
        "dotnet",
        # Docker/K8s
        "docker",
        "kubectl",
        "oc",
        # Cloud/CLI
        "aws",
        "gh",
        "glab",
        "gt",
        "curl",
        "wget",
        "psql",
        # Other
        "php",
        "phpunit",
        "phpstan",
        "pest",
        "paratest",
        "ecs",
        "pint",
        "gradlew",
        "mvn",
    ]
)


@functools.lru_cache(maxsize=1)
def _rtk_binary_path() -> Path | None:
    """Return the path to the RTK binary in the share ``bin`` directory, if present."""
    from kimi_cli.share import get_share_dir
    bin_name = "rtk.exe" if os.name == "nt" else "rtk"
    candidate = get_share_dir() / "bin" / bin_name
    return candidate if candidate.is_file() else None


@functools.lru_cache(maxsize=1)
def _rtk_available() -> bool:
    return _rtk_binary_path() is not None


def _is_known_rtk_command(name: str) -> bool:
    # Strip Windows .exe extension before lookup, then normalize case.
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name.lower() in _RTK_KNOWN_COMMANDS


# ANSI escape sequences (colored text, cursor movement, OSC/DCS/PM/APC strings)
_ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:"
    r"\][^\x07\x1B]*(?:\x07|\x1B\\)|"  # OSC sequences (BEL or ST terminated)
    r"[P^_][^\x07\x1B]*(?:\x07|\x1B\\)|"  # DCS / PM / APC sequences
    r"[@-Z\\-_]|"              # Single-character Fe sequences
    r"\[[0-?]*[ -/]*[@-~]"      # CSI sequences
    r")"
)

def filter_output(text: str) -> str:
    """Process process pipeline stdout.

    Steps:
        1. Remove ANSI escape sequences (colored text, cursor movement,
           OSC/DCS/PM/APC strings).
        2. Normalize CRLF and lone CR line endings to LF.

    Args:
        text: Raw stdout text.

    Returns:
        Cleaned plain text.
    """
    if not isinstance(text, str):
        raise TypeError("filter_output expects a string")
    # Native acceleration: kimix_native.stream.filter_output (bit-identical
    # ANSI-strip + CRLF normalize).
    if _native_use_native("STREAM") and _NATIVE_STREAM is not None:
        return _NATIVE_STREAM.filter_output(text)
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


from kimi_cli.session import Session

if TYPE_CHECKING:
    from kimix.tools.background.utils import BackgroundStream
OUTPUT_LIMIT = 16384
_temp_folder = Path('.kimix_cache') / f'tmp_{os.getpid()}'
_temp_folder.mkdir(parents=True, exist_ok=True)
_temp_idx = 0
_temp_set: dict[Path, int] = dict()


def _cleanup_temp_folder() -> None:
    if _temp_folder.exists():
        shutil.rmtree(_temp_folder, ignore_errors=True)


atexit.register(_cleanup_temp_folder)




def _create_temp_file_name(ext: str = '.md') -> str:
    global _temp_idx
    id = _temp_idx
    _temp_idx += 1
    return str(_temp_folder / (str(id) + ext))


def _export_to_temp_file(key: Path | None, content: str, ext: str = '.txt') -> tuple[str, bool]:
    global _temp_idx
    """Export content to a temporary file and return the file path."""
    id = _temp_idx
    new_id = True
    if key:
        v = _temp_set.get(key)
        if v is not None:
            id = v
            new_id = False
        else:
            # Add key to _temp_set with the new id
            _temp_set[key] = id
    if new_id:
        _temp_idx += 1
    _temp_folder.mkdir(parents=True, exist_ok=True)
    name = str(_temp_folder / (str(id) + ext))
    # Append content if key exists, otherwise overwrite/create
    mode = 'a' if not new_id else 'w'
    with open(name, mode, encoding='utf-8') as f:
        f.write(content)
    return name, new_id


async def _export_to_temp_file_async(key: Path | None, content: str, ext: str = '.txt') -> tuple[str, bool]:
    global _temp_idx
    """Async version: Export content to a temporary file and return the file path."""
    import anyio
    id = _temp_idx
    new_id = True
    if key:
        v = _temp_set.get(key)
        if v is not None:
            id = v
            new_id = False
        else:
            # Add key to _temp_set with the new id
            _temp_set[key] = id
    if new_id:
        _temp_idx += 1
    name = _temp_folder / (str(id) + ext)
    # Append content if key exists, otherwise overwrite/create
    mode = 'a' if not new_id else 'w'
    async with await anyio.open_file(name, mode, encoding='utf-8') as f:
        await f.write(content)
    return str(name), new_id


def _maybe_export_output(output: str, key: Path | None = None) -> str:
    """Check if output is too large and export to temp file if needed.

    Args:
        output: The output string to check.
        key: Optional Path to normalize and use in the output message.

    Returns:
        The output string, or a message indicating it was exported to a temp file.
    """
    if not output:
        return ''
    if len(output) > OUTPUT_LIMIT:
        if key is not None:
            if type(key) is not Path:
                key = Path(key)
            key = key.resolve()
        temp_path, new_id = _export_to_temp_file(key, output)
        return f"Output too large, {'exported' if new_id else 'added'} to file `{temp_path}`"
    return output


async def _maybe_export_output_async(output: str, key: Path | None = None) -> str:
    """Async version: Check if output is too large and export to temp file if needed.

    Args:
        output: The output string to check.
        key: Optional Path to normalize and use in the output message.

    Returns:
        The output string, or a message indicating it was exported to a temp file.
    """
    if not output:
        return ''
    if len(output) > OUTPUT_LIMIT:
        if key is not None:
            if type(key) is not Path:
                key = Path(key)
            key = key.resolve()
        temp_path, new_id = await _export_to_temp_file_async(key, output)
        return f"[Output too large, {'exported' if new_id else 'added'} to file: {temp_path}]"
    return output


async def _summarize_long_output_async(session: Session, command: str, output: str) -> str:
    """Process a long command output through an anonymous sub-agent.

    The sub-agent is launched with ``agent_useless.json`` and is asked to
    produce a concise summary of the command output for the parent coding
    agent.

    Args:
        session: The parent tool session.
        command: The command that produced the output.
        output: The full command output.

    Returns:
        A concise summary, or the original output with a note if the
        sub-agent could not be used.
    """
    import kimix.base as base
    from kimix.ui.printing import MessageType
    from kimix.utils import close_session_async, _create_session_async, prompt_async
    from kimix.utils.system_prompt import SystemPromptType

    custom_config = session.custom_config
    chat_provider = custom_config.get("chat_provider")
    default_sub_provider = (
        base.get_default_sub_provider("sub_agent")
        or custom_config.get("provider_dict", base._default_provider)
    )

    sub_session_id = str(uuid.uuid4())
    sub_session = None
    try:
        sub_session = await _create_session_async(
            session_id=sub_session_id,
            agent_file=base._default_agent_file_dir / "agent_useless.json",
            agent_type=SystemPromptType.Reader,
            provider_dict=default_sub_provider,
            chat_provider=chat_provider,
            resume=False,
            anonymous=True,
            max_ralph_iterations=0,
        )
        sub_custom_config = sub_session.get_custom_config()
        if sub_custom_config is not None:
            sub_custom_config["is_sub_agent"] = True

        _OUTPUT_FILE_THRESHOLD = 100 * 1024
        if len(output) > _OUTPUT_FILE_THRESHOLD:
            temp_path, _ = await _export_to_temp_file_async(
                key=None, content=output, ext=".txt"
            )
            display_temp_path = temp_path.replace("\\", "/")
            output_section = (
                f"Output saved to `{display_temp_path}` (>100KB). "
                f"Read the file and summarize it for a coding agent."
            )
        else:
            output_section = f"Output:\n{output}"

        prompt = (
            f"Command:\n{command}\n\n"
            f"{output_section}\n\n"
            "Summarize the output for a coding agent. "
            "Highlight key results, errors, warnings, and next steps. "
            "Do not run commands."
        )

        collected: list[str] = []

        def output_function(text: str, msg_type: MessageType) -> None:
            if text and msg_type == MessageType.Text:
                collected.append(text)

        await prompt_async(
            prompt_str=prompt,
            session=sub_session,
            output_function=output_function,
            info_print=False,
            merge_wire_messages=True,
        )
        summary = "".join(collected).strip()
        if not summary:
            summary = "(no text output)"
        return summary
    except Exception as exc:
        return f"[Summarization failed: {exc}]\n\n{output}"
    finally:
        if sub_session is not None:
            try:
                await close_session_async(sub_session)
            except Exception:
                pass


def _extract_export_path(output: str) -> str | None:
    """If ``output`` is an export-to-temp-file message, return the file path."""
    if not output:
        return None
    markers = [
        "exported to file `",
        "added to file `",
        "exported to file: ",
        "added to file: ",
    ]
    for marker in markers:
        if marker in output:
            return output.split(marker, 1)[-1].rstrip("]`")
    return None


def _build_session_output_block(
    *,
    task_id: str,
    status: str,
    output: str,
    wait_matched: bool | None = None,
    elapsed_seconds: float | None = None,
    exit_code: int | None = None,
    exit_code_meaning: str | None = None,
    failure_hint: str | None = None,
    output_path: str | None = None,
    output_truncated: bool = False,
    original_path: str | None = None,
) -> str:
    """Build a YAML-like metadata block for interactive/background sessions.

    The block is appended to tool output so callers get structured metadata
    (task_id, status, elapsed_seconds, exit-code meaning, failure hint, etc.)
    alongside the raw process output.
    """
    lines = [
        f"task_id: {task_id}",
        f"status: {status}",
        f"exit_code: {exit_code if exit_code is not None else 'null'}",
        f"exit_code_meaning: {exit_code_meaning if exit_code_meaning else 'null'}",
        f"failure_hint: {failure_hint if failure_hint else 'null'}",
        "output: |",
    ]
    if output:
        lines.extend(textwrap.indent(output.rstrip("\n"), "  ").splitlines())
    else:
        lines.append("  (no output)")
    lines.append(f"output_truncated: {str(output_truncated).lower()}")
    lines.append(f"output_path: {output_path if output_path else 'null'}")
    lines.append(
        f"wait_matched: {str(wait_matched).lower() if wait_matched is not None else 'null'}"
    )
    lines.append(
        f"elapsed_seconds: {elapsed_seconds:.2f}" if elapsed_seconds is not None else "elapsed_seconds: null"
    )
    lines.append(
        f"original_path: {original_path if original_path else 'null'}"
    )
    return "\n".join(lines)


def _dedup_output(
    output: str,
    threshold: int = 3,
    *,
    max_block_lines: int = 1,
) -> str:
    """Collapse identical repeated lines or multi-line blocks.

    Lines appearing more than ``threshold`` times are collapsed:
        "ERROR: timeout" x 500  ->  "ERROR: timeout  (500 repeats)"

    When ``max_block_lines`` is greater than 1, contiguous runs of identical
    multi-line blocks are also collapsed. The annotation is placed on the last
    line of the kept block:
        "ERROR: module load failed\n  at /app/main.py:42" x 5
        ->  "ERROR: module load failed\n  at /app/main.py:42  (5 repeats)"

    The first occurrence of a repeated line or block is kept with the count
    annotation; all subsequent duplicates are dropped. Unique lines/blocks and
    those appearing <= threshold times pass through unchanged. Line order is
    preserved.

    Args:
        output: The output string to deduplicate.
        threshold: Minimum repeat count before collapsing (default 3).
        max_block_lines: Maximum height of a repeating block to detect.
            Default 1 keeps the original single-line behavior. Values >= 2
            enable multi-line block detection.

    Returns:
        Deduplicated output string.
    """
    if not output:
        return ""
    # Native acceleration: kimix_native.stream.LineProcessor (dedup_mode
    # 1=counter / 2=block, block_window=max_block_lines).
    if _native_use_native("STREAM") and _NATIVE_STREAM is not None:
        lp = _NATIVE_STREAM.LineProcessor(
            strip_ansi=False,
            dedup_mode=2 if max_block_lines > 1 else 1,
            threshold=threshold,
            block_window=max_block_lines,
        )
        lp.feed(output)
        return "\n".join(lp.flush())
    lines = output.splitlines()

    if max_block_lines <= 1:
        # Count occurrences per line (preserving original form)
        from collections import Counter
        counts = Counter(lines)
        emitted: set[str] = set()
        result: list[str] = []
        for line in lines:
            cnt = counts[line]
            if cnt > threshold:
                if line not in emitted:
                    emitted.add(line)
                    result.append(f"{line}  ({cnt} repeats)")
            else:
                result.append(line)
        return "\n".join(result)

    # Multi-line path: greedy largest-block-first contiguous run detection.
    consumed = [False] * len(lines)
    result: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        if consumed[i]:
            i += 1
            continue

        collapsed = False
        for h in range(min(max_block_lines, n - i), 0, -1):
            block = tuple(lines[i : i + h])
            # Count contiguous repeats starting at i.
            j = i
            repeats = 0
            while j + h <= n and tuple(lines[j : j + h]) == block:
                if any(consumed[k] for k in range(j, j + h)):
                    break
                repeats += 1
                j += h

            if repeats > threshold:
                # Emit one copy of the block.
                for k, line in enumerate(block[:-1]):
                    result.append(line)
                    consumed[i + k] = True
                result.append(f"{block[-1]}  ({repeats} repeats)")
                for k in range(h * repeats):
                    consumed[i + k] = True
                i = i + h * repeats
                collapsed = True
                break

        if not collapsed:
            result.append(lines[i])
            consumed[i] = True
            i += 1

    return "\n".join(result)


def _truncate_lines(output: str, max_lines: int) -> str:
    """Truncate to max_lines with head/tail fold.

    Preserves first floor(max_lines/2) lines and last ceil(max_lines/2)-1 lines.
    The middle is replaced with a fold marker indicating how many lines
    were omitted and referencing the original file if available.

    If total_lines <= max_lines, returns output unchanged.

    Args:
        output: The output string to truncate.
        max_lines: Maximum number of lines to keep (min 3).

    Returns:
        Truncated output string with fold marker.
    """
    if not output or max_lines <= 0:
        return output
    lines = output.splitlines()
    n = len(lines)
    if n <= max_lines:
        return output
    head_n = max_lines // 2
    tail_n = max_lines - head_n - 1  # -1 reserves one line for fold marker
    omitted = n - head_n - tail_n
    head = "\n".join(lines[:head_n])
    tail = "\n".join(lines[-tail_n:]) if tail_n > 0 else ""
    fold = f"\n\n[... {omitted} lines omitted ...]\n\n"
    if tail:
        return head + fold + tail
    return head + fold


def _find_ansi_c_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ``'`` of a ``$'...'`` region."""
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length:
            i += 2
        elif c == "'":
            return i + 1
        else:
            i += 1
    return -1


def _find_backtick_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing backtick of a `` `...` `` region."""
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length:
            i += 2
        elif c == "`":
            return i + 1
        else:
            i += 1
    return -1


def _find_dq_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ``"`` of a double-quoted region."""
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length and cmd[i + 1] in ('"', "\\", "$", "`"):
            i += 2
        elif c == '"':
            return i + 1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "(":
            end = _find_matching_paren(cmd, i + 1)
            if end == -1:
                return -1
            i = end + 1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "'":
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                return -1
            i = end
        elif c == "`":
            end = _find_backtick_end(cmd, i + 1)
            if end == -1:
                return -1
            i = end
        else:
            i += 1
    return -1


def _find_matching_paren(cmd: str, open_pos: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``cmd[open_pos]``."""
    assert cmd[open_pos] == "("
    depth = 1
    i = open_pos + 1
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "'":
            end = cmd.find("'", i + 1)
            if end == -1:
                return -1
            i = end + 1
        elif c == '"':
            end = _find_dq_end(cmd, i + 1)
            if end == -1:
                return -1
            i = end
        elif c == "`":
            end = _find_backtick_end(cmd, i + 1)
            if end == -1:
                return -1
            i = end
        elif c == "$" and i + 1 < length and cmd[i + 1] == "'":
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                return -1
            i = end
        elif c == "$" and i + 1 < length and cmd[i + 1] == "(":
            depth += 1
            i += 2
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
            i += 1
        else:
            i += 1
    return -1


def _split_shell_segments(command: str) -> list[tuple[str, str]]:
    """Split a shell command on ``;``, ``&&``, ``||``, and ``|``.

    Quoted regions and command substitutions are protected so that separators
    inside them do not create spurious segments.  A single ``&`` (background)
    is left inside its segment.
    """
    segments: list[tuple[str, str]] = []
    current: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        c = command[i]
        if c == "'":
            end = command.find("'", i + 1)
            if end == -1:
                current.append(command[i:])
                i = n
                continue
            current.append(command[i : end + 1])
            i = end + 1
        elif c == '"':
            end = _find_dq_end(command, i + 1)
            if end == -1:
                current.append(command[i:])
                i = n
                continue
            current.append(command[i:end])
            i = end
        elif c == "$" and i + 1 < n and command[i + 1] == "'":
            end = _find_ansi_c_end(command, i + 2)
            if end == -1:
                current.append(command[i:])
                i = n
                continue
            current.append(command[i:end])
            i = end
        elif c == "$" and i + 1 < n and command[i + 1] == "(":
            end = _find_matching_paren(command, i + 1)
            if end == -1:
                current.append(command[i:])
                i = n
                continue
            current.append(command[i : end + 1])
            i = end + 1
        elif c == "`":
            end = _find_backtick_end(command, i + 1)
            if end == -1:
                current.append(command[i:])
                i = n
                continue
            current.append(command[i:end])
            i = end
        elif c == ";":
            segments.append(("".join(current), ";"))
            current = []
            i += 1
        elif c == "|":
            if i + 1 < n and command[i + 1] == "|":
                segments.append(("".join(current), "||"))
                current = []
                i += 2
            else:
                # Keep single `|` inside the segment; the per-segment rewriter
                # only rewrites the leftmost command in the pipeline.
                current.append(c)
                i += 1
        elif c == "&":
            if i + 1 < n and command[i + 1] == "&":
                segments.append(("".join(current), "&&"))
                current = []
                i += 2
            else:
                current.append(c)
                i += 1
        else:
            current.append(c)
            i += 1
    segments.append(("".join(current), ""))
    return segments


def _read_shell_word(
    cmd: str, i: int
) -> tuple[str, int, int] | tuple[None, None, None]:
    """Read the next shell word starting at or after ``i``.

    Returns ``(word, word_start, next_index)``.  Quoted regions and command
    substitutions are consumed as part of the word.  An unquoted ``|`` ends
    the word so callers can identify the leftmost command in a pipeline.
    """
    n = len(cmd)
    while i < n and cmd[i].isspace():
        i += 1
    if i >= n:
        return None, None, None
    start = i
    chars: list[str] = []
    while i < n:
        c = cmd[i]
        if c.isspace():
            break
        if c == "|":
            break
        if c == "'":
            end = cmd.find("'", i + 1)
            if end == -1:
                chars.append(cmd[i:])
                i = n
                break
            chars.append(cmd[i : end + 1])
            i = end + 1
        elif c == '"':
            end = _find_dq_end(cmd, i + 1)
            if end == -1:
                chars.append(cmd[i:])
                i = n
                break
            chars.append(cmd[i:end])
            i = end
        elif c == "$" and i + 1 < n and cmd[i + 1] == "'":
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                chars.append(cmd[i:])
                i = n
                break
            chars.append(cmd[i:end])
            i = end
        elif c == "$" and i + 1 < n and cmd[i + 1] == "(":
            end = _find_matching_paren(cmd, i + 1)
            if end == -1:
                chars.append(cmd[i:])
                i = n
                break
            chars.append(cmd[i : end + 1])
            i = end + 1
        elif c == "`":
            end = _find_backtick_end(cmd, i + 1)
            if end == -1:
                chars.append(cmd[i:])
                i = n
                break
            chars.append(cmd[i:end])
            i = end
        else:
            chars.append(c)
            i += 1
    return "".join(chars), start, i


# Prefix commands that should be skipped when finding the real executable token.
# These are modifiers that precede the actual command (e.g. ``sudo git status``).
_PREFIX_SKIP: frozenset[str] = frozenset({"sudo", "time", "nohup", "nice"})

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _is_shell_assignment(word: str) -> bool:
    return bool(_ASSIGNMENT_RE.match(word))


def _rewrite_shell_segment(
    segment: str, exclude_read: bool, pwsh: bool = False
) -> tuple[str, bool]:
    """Rewrite the leftmost command in a single shell segment, if known to RTK."""
    i = 0
    while True:
        word, start, end = _read_shell_word(segment, i)
        if word is None:
            return segment, False
        if word == "RTK_DISABLED=1":
            return segment, False
        if _is_shell_assignment(word):
            i = end
            continue
        if word in _PREFIX_SKIP:
            i = end
            continue
        # First real executable token in this segment/pipeline.
        token = word
        token_start = start
        break

    # Strip surrounding quotes so quoted absolute paths (e.g. the rewritten
    # rtk prefix) are still matched by stem.
    name = Path(token.strip("\"'")).stem
    if name.lower() in ("rtk", "rtk.exe"):
        return segment, False
    if exclude_read and name.lower() == "read":
        return segment, False
    if not _is_known_rtk_command(name):
        return segment, False

    # Use the bare `rtk` executable name. Callers must ensure the shared
    # `bin` directory is first on PATH (see `_env_with_rg_bin_path`) so this
    # resolves to our binary even if another `rtk` exists globally.
    # PowerShell requires the call operator `&` to invoke a command by name.
    prefix = "& rtk " if pwsh else "rtk "
    return segment[:token_start] + prefix + segment[token_start:], True


def _maybe_rewrite_shell_command_with_rtk(
    command: str, token_kill: bool, exclude_read: bool = False, pwsh: bool = False
) -> tuple[str, bool]:
    """Return ``(rewritten_command, did_rewrite)``.

    Performs robust, quote-aware matching on bash/pwsh style command strings
    that may contain ``&&``, ``||``, ``|``, ``;``.  Commands inside quotes or
    command substitutions are not rewritten.  Segments already starting with
    ``rtk`` or prefixed with ``RTK_DISABLED=1`` are left untouched.

    When ``pwsh`` is True the injected ``rtk`` prefix is prefixed with the
    PowerShell call operator ``&`` (required to invoke a command by name).
    """
    if not token_kill:
        return command, False
    if not _rtk_available():
        return command, False
    if not command or command.isspace():
        return command, False

    stripped = command.lstrip()
    if (
        stripped.startswith("rtk ")
        or stripped.startswith("rtk\t")
        or stripped == "rtk"
        or stripped.startswith("rtk.exe")
        or stripped.startswith("& rtk ")
        or stripped == "& rtk"
    ):
        return command, False

    # Also leave commands already starting with the absolute rtk path
    # (quoted or not, with or without the pwsh `&` call operator) untouched.
    rtk_path = _rtk_binary_path()
    if rtk_path is not None:
        rtk_str = str(rtk_path)
        head = stripped[2:].lstrip() if stripped.startswith("& ") else stripped
        if head.startswith(rtk_str) or head.startswith(f'"{rtk_str}"'):
            return command, False

    segments = _split_shell_segments(command)
    new_segments: list[tuple[str, str]] = []
    changed = False
    for seg, sep in segments:
        new_seg, seg_changed = _rewrite_shell_segment(seg, exclude_read, pwsh)
        new_segments.append((new_seg, sep))
        changed |= seg_changed

    if not changed:
        return command, False

    return "".join(seg + sep for seg, sep in new_segments), True


async def _token_filter_output(
    output: str,
    *,
    token_kill: bool = True,
    max_lines: int | None = None,
    rtk_rewritten: bool = False,
    max_block_lines: int = 1,
) -> tuple[str, str | None]:
    """Run the token filter pipeline on shell output.

    Stages run in order:
      1. Strip ANSI escape codes (via rich, merged with dedup step)
      2. Save original to temp file (if any filter is active)
      3. Dedup (collapse repeated lines/blocks, when token_kill=True and RTK was not used)
      4. Truncate (head/tail fold to max_lines)

    Args:
        output: Raw output string (ANSI already stripped by ProcessTask).
        token_kill: Enable token-saving processing. When True and the command
            was not rewritten to RTK, the legacy Python dedup/ANSI pipeline is
            run.  When the command was rewritten to RTK, local dedup is skipped
            because RTK already collapses repeats.
        max_lines: Optional max line count for head/tail truncation.
        rtk_rewritten: Whether the producing command was rewritten to ``rtk``.
        max_block_lines: Maximum height of a repeating block for dedup.
            Default 1 keeps single-line dedup. Values >= 2 enable multi-line
            block detection.

    Returns:
        (filtered_output, original_path).
        original_path is None if no filters were active.
    """
    apply_dedup = token_kill and not rtk_rewritten
    has_filter = apply_dedup or (max_lines is not None)

    # Step 1: Strip ANSI escape codes (via rich, merged with dedup step)
    # When dedup is enabled, first strip any remaining ANSI codes using rich's
    # robust ANSI parser to catch edge cases the regex-based filter_output missed.
    if apply_dedup and output:
        from rich.text import Text
        output = Text.from_ansi(output).plain

    # Step 2: Save original before any destructive transform
    # Even if output is empty, save the file when a filter is active.
    original_path: str | None = None
    if has_filter:
        original_path, _ = await _export_to_temp_file_async(
            key=None, content=output, ext=".txt"
        )

    if not output:
        return output, original_path

    # Step 2.5: Micro-compress — lossless stages (1-3, 5) plus the annotated
    # stages (4, 6, 7, 8): timestamp/path prefix folding, banner drop,
    # intra-line repetition and near-duplicate collapse.  All annotated
    # stages emit count-bearing markers so nothing is hidden silently.
    if apply_dedup:
        from kimi_cli.tools.file.micro_compress import (
            MicroCompressConfig,
            compress as _mc_compress,
        )
        output = _mc_compress(
            output,
            kind="log",
            config=MicroCompressConfig(),
        )

    # Step 3: Dedup (preserves line order)
    if apply_dedup:
        output = _dedup_output(output, max_block_lines=max_block_lines)

    # Step 4: Truncate (head/tail fold)
    if max_lines is not None:
        output = _truncate_lines(output, max_lines)

    return output, original_path


async def _maybe_export_rtk_original_async(output: str) -> tuple[str | None, bool]:
    """Export *output* to a temp file if it contains rtk truncation markers.

    rtk (the token-killing wrapper) injects protocol markers when it folds
    long output, e.g. ``+N more in <path> [see remaining: ...]`` or
    ``+N more files [see remaining: ...]``.  When such markers are present,
    saving the full stream lets the model page through the unfiltered output.

    Returns:
        ``(temp_path, truncated)`` where *temp_path* is the path to the saved
        file (``None`` if no markers were found) and *truncated* is ``True``
        when rtk truncation markers were detected.
    """
    if not output or "[see remaining:" not in output:
        return None, False
    # Import lazily: output_utils is a kimi_cli helper and not needed unless
    # we actually see rtk-style markers.
    from kimi_cli.tools.file.output_utils import parse_rtk_rg_output

    _, meta = parse_rtk_rg_output(output.splitlines())
    if not (meta.get("folded_files") or meta.get("skipped_files")):
        return None, False
    temp_path, _ = await _export_to_temp_file_async(
        key=None, content=output, ext=".txt"
    )
    return temp_path, True


def _interactive_scope_text(*, is_shell: bool = True) -> str:
    """Return the interactive/background usage scope prompt for tool descriptions.

    Shell tools (Bash, Powershell) use ``interactive=True``, while the
    ``Run`` tool uses ``run_in_background=True``.  This is the single source
    of truth so that tool descriptions stay in sync.

    Args:
        is_shell: ``True`` for shell tools (Bash/Powershell), ``False`` for Run.

    Returns:
        A sentence describing how to start interactive/background sessions.
    """
    if is_shell:
        return (
            "Start a persistent session with interactive=True, then reuse the same tool with "
            "task_id=<id> to send input and read output in one step. Use wait_for_pattern to wait "
            "for a prompt. TaskOutput remains available as a fallback for listing/monitoring tasks. "
            "Send 'exit' to close the session."
        )
    return (
        "Start a background session with run_in_background=True, then reuse the same tool with "
        "task_id=<id> to send input and read output in one step. Use wait_for_pattern to wait "
        "for a prompt. TaskOutput remains available as a fallback for listing/monitoring tasks."
    )


def _env_with_rg_bin_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of environment with the shared ``bin`` directory first on PATH.

    The shared ``bin`` directory contains ``rg`` and ``rtk``.  Prepending it
    (and removing any duplicate entry elsewhere in PATH) guarantees that our
    binaries win over any globally-installed copies.

    Args:
        env: Optional base environment dict. If None, ``os.environ`` is used.

    Returns:
        A new dict with ``PATH`` updated so the shared ``bin`` directory is first.
    """
    from kimi_cli.share import get_share_dir

    rg_bin_dir = str(get_share_dir() / "bin")
    result = os.environ.copy() if env is None else env.copy()

    current_path = result.get("PATH", "")
    path_sep = ";" if os.name == "nt" else ":"
    entries = [e for e in current_path.split(path_sep) if e and e != rg_bin_dir]
    result["PATH"] = path_sep.join([rg_bin_dir] + entries)

    return result


class ProcessTask:
    """Run a subprocess in the background with stream output and input support."""

    def __init__(
        self,
        path: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        append_newline: bool = False,
        scrub_env: bool = False,
        redact: bool = False,
    ) -> None:
        import shutil
        # On Windows, subprocess.Popen with shell=False does not resolve .cmd/.bat
        # via PATHEXT. Use shutil.which to find the real executable (e.g. pnpm.CMD).
        if not Path(path).exists():
            resolved = shutil.which(path)
            if resolved:
                path = resolved
        self.path = path
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self.scrub_env = scrub_env
        self.redact = redact
        self._append_newline = append_newline
        self._stop_event = threading.Event()
        self._process_ref: asyncio.subprocess.Process | None = None
        self._stream: 'BackgroundStream' | None = None
        self._task_id: str | None = None
        self._input_queue: queue.Queue[str] = queue.Queue()

    async def _run_process_bg(self, q: queue.Queue[str]) -> tuple[bool, int | None]:
        """Run the process and collect output into the queue.

        Returns:
            ``(success, exit_code)`` where *exit_code* is the subprocess return
            code (may be ``None`` if the process was stopped before completing).
        """
        process = None
        output_buffer = io.StringIO()
        # Lazy imports: kimix.tools.security is a light module, but importing it
        # here (rather than at module top) keeps kimix.tools.common importable
        # even when only the process machinery is needed.  The cap is read from
        # the background.utils module namespace at call time so tests can patch
        # ``BACKGROUND_MAX_OUTPUT_CHARS``.
        from kimix.tools.background.utils import bounded_append, bounded_put
        import kimix.tools.background.utils as _bg_utils
        from kimix.tools.security import redact_sensitive_output, scrub_child_env

        def _write_output(text: str) -> None:
            nonlocal output_buffer
            bounded_append(output_buffer, text, _bg_utils.BACKGROUND_MAX_OUTPUT_CHARS)

        try:
            if self._stop_event.is_set():
                return False, None
            # Start the process.  The child inherits the parent environment;
            # when scrubbing is enabled, credential-looking variables are
            # removed from the base copy FIRST so that a caller-provided
            # ``self.env`` override can still restore a specific variable.
            process_env = os.environ.copy()
            if self.scrub_env:
                process_env = scrub_child_env(process_env)
            if self.env:
                process_env.update(self.env)
            # On Windows, put the child in its own process group and create it
            # without a console window. A separate process group prevents signals
            # sent to the child via GenerateConsoleCtrlEvent from reaching the
            # parent. Running without a console window detaches the child from the
            # parent's console so that Ctrl+C/Ctrl+Break generated by the user is
            # not broadcast back to the parent Python process. This prevents
            # PowerShell and other console apps from raising KeyboardInterrupt in
            # the current process.
            if os.name == "nt":
                process = await asyncio.create_subprocess_exec(
                    self.path,
                    *self.args,
                    cwd=self.cwd,
                    env=process_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    self.path,
                    *self.args,
                    cwd=self.cwd,
                    env=process_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            self._process_ref = process
            # Read stdout and stderr concurrently with stop checking

            if process.stdout is None:
                raise RuntimeError("Subprocess stdout is None")

            async def read_stdout() -> None:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                try:
                    while True:
                        if self._stop_event.is_set():
                            break
                        data = await process.stdout.read(4096)
                        if data:
                            text = decoder.decode(data)
                            if text:
                                text = filter_output(text)
                                if self.redact:
                                    text = redact_sensitive_output(text)
                                bounded_put(q, text)
                                _write_output(text)
                        else:
                            text = decoder.decode(b'', final=True)
                            if text:
                                text = filter_output(text)
                                if self.redact:
                                    text = redact_sensitive_output(text)
                                bounded_put(q, text)
                                _write_output(text)
                            break
                except (IOError, OSError, ValueError):
                    pass
                finally:
                    text = decoder.decode(b'', final=True)
                    if text:
                        text = filter_output(text)
                        if self.redact:
                            text = redact_sensitive_output(text)
                        bounded_put(q, text)
                        _write_output(text)

            async def read_stderr() -> None:
                if process.stderr is None:
                    return
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                try:
                    while True:
                        if self._stop_event.is_set():
                            break
                        data = await process.stderr.read(4096)
                        if data:
                            text = decoder.decode(data)
                            if text:
                                text = filter_output(text)
                                if self.redact:
                                    text = redact_sensitive_output(text)
                                msg = "[stderr] " + text
                                bounded_put(q, msg)
                                _write_output(msg)
                        else:
                            text = decoder.decode(b'', final=True)
                            if text:
                                text = filter_output(text)
                                if self.redact:
                                    text = redact_sensitive_output(text)
                                msg = "[stderr] " + text
                                bounded_put(q, msg)
                                _write_output(msg)
                            break
                except (IOError, OSError, ValueError):
                    pass
                finally:
                    text = decoder.decode(b'', final=True)
                    if text:
                        text = filter_output(text)
                        if self.redact:
                            text = redact_sensitive_output(text)
                        msg = "[stderr] " + text
                        bounded_put(q, msg)
                        _write_output(msg)

            async def write_stdin() -> None:
                try:
                    while True:
                        if self._stop_event.is_set() or process.returncode is not None:
                            break
                        if process.stdin is None:
                            raise RuntimeError("Subprocess stdin is None")
                        try:
                            data = self._input_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue
                        process.stdin.write(data.encode('utf-8', errors='replace'))
                        await process.stdin.drain()
                except (IOError, OSError, ValueError, asyncio.CancelledError):
                    pass

            # Start reader/writer tasks
            stdout_task = asyncio.create_task(read_stdout())
            stderr_task: asyncio.Task[None] | None = None
            if process.stderr is not None:
                stderr_task = asyncio.create_task(read_stderr())
            stdin_task = asyncio.create_task(write_stdin())

            # Wait for process completion with periodic stop checking
            while process.returncode is None:
                if self._stop_event.is_set():
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    break
                await asyncio.sleep(0.1)

            if process.returncode is not None and not self._stop_event.is_set():
                await process.wait()

            # Cancel tasks and wait for them to finish
            stdout_task.cancel()
            if stderr_task is not None:
                stderr_task.cancel()
            stdin_task.cancel()
            try:
                await stdout_task
            except asyncio.CancelledError:
                pass
            if stderr_task is not None:
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            try:
                await stdin_task
            except asyncio.CancelledError:
                pass

            # Read any remaining data from stdout and stderr
            try:
                remaining_stdout = await process.stdout.read()
                if remaining_stdout:
                    text = remaining_stdout.decode('utf-8', errors='replace')
                    text = filter_output(text)
                    if self.redact:
                        text = redact_sensitive_output(text)
                    bounded_put(q, text)
                    _write_output(text)
            except (IOError, OSError, ValueError):
                pass
            if process.stderr is not None:
                try:
                    remaining_stderr = await process.stderr.read()
                    if remaining_stderr:
                        text = remaining_stderr.decode('utf-8', errors='replace')
                        text = filter_output(text)
                        if self.redact:
                            text = redact_sensitive_output(text)
                        msg = "[stderr] " + text
                        bounded_put(q, msg)
                        _write_output(msg)
                except (IOError, OSError, ValueError):
                    pass
            # Report completion status
            return_code = process.returncode
            if self._stop_event.is_set():
                q.put_nowait("\n[Process stopped by user]")
                return False, return_code
            elif return_code is not None and return_code != 0:
                full_output = output_buffer.getvalue()
                error_line = _find_error_line_index(full_output)
                if error_line is not None:
                    q.put_nowait(f"\n[Process exited with code {return_code}, error at line {error_line}]")
                else:
                    q.put_nowait(f"\n[Process exited with code {return_code}]")
                return False, return_code
            return True, return_code

        except Exception as e:
            q.put_nowait(f"\n[Error: {str(e)}]")
            return False, None
        finally:
            self._stop_event.set()
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

    async def _stop_function(self) -> None:
        """Signal the background process to stop."""
        self._stop_event.set()
        # Also try to terminate the process directly if it's running
        proc = self._process_ref
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                pass

    async def _input_function(self, data: str) -> bool:
        """Push data to the process's stdin.

        Args:
            data: The string data to write to stdin.

        Returns:
            True if data was written successfully, False otherwise.
        """
        proc = None
        # Wait for the process to be available
        while True:
            if self._stop_event.is_set():
                return False
            proc = self._process_ref
            if proc is None:
                await asyncio.sleep(0.05)
            else:
                break

        # Write data to stdin
        try:
            if proc.stdin is not None and proc.returncode is None:
                if self._append_newline and not data.endswith("\n"):
                    data += "\n"
                self._input_queue.put_nowait(data)
                return True
        except (IOError, OSError, ValueError):
            # Process may have terminated or stdin is closed
            pass
        return False

    async def start(self, session: Session, kind: str = "run", name: str | None = None) -> str:
        """Start the background process and register it as a task.

        Args:
            session: The session instance.
            kind: Task kind prefix for the task ID.
            name: Optional name for the task ID (defaults to the executable stem).

        Returns:
            The generated task ID.
        """
        from kimix.tools.background.utils import BackgroundStream, generate_task_id, add_task
        self._stream = BackgroundStream()
        # Generate a task ID based on the executable name
        self._task_id = generate_task_id(session, kind, name)
        await self._stream.start(self._run_process_bg,
                           self._stop_function, self._input_function)
        # Register the task
        add_task(session, self._task_id, self._stream)
        assert self._task_id is not None
        return self._task_id

    async def wait(self, timeout: float | None = None) -> None:
        await self._stream.wait(timeout)

    async def wait_with_monitor(
        self,
        timeout: float,
        inactivity_timeout: float | None = None,
    ) -> tuple[bool, float, bool]:
        """Wait for the process, exiting early if output stalls too long.

        Args:
            timeout: Maximum total seconds to wait.
            inactivity_timeout: Seconds of output inactivity that triggers an
                early return when ``timeout`` is larger. Defaults to
                ``DEFAULT_INACTIVITY_TIMEOUT`` at call time so tests can patch
                the module constant.

        Returns:
            ``(completed, elapsed_seconds, inactivity_timed_out)``.
        """
        if inactivity_timeout is None:
            from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
            inactivity_timeout = DEFAULT_INACTIVITY_TIMEOUT
        if self._stream is None:
            return True, 0.0, False
        return await self._stream.wait_with_inactivity_timeout(timeout, inactivity_timeout)

    async def thread_is_alive(self) -> bool:
        return await self._stream.thread_is_alive()

    async def stop(self) -> None:
        """Stop the background process."""
        if self._stream is not None:
            await self._stream.stop()

    async def input(self, data: str) -> bool:
        """Push data to the process's stdin.

        Args:
            data: The string data to write to stdin.

        Returns:
            True if data was written successfully, False otherwise.
        """
        if self._stream is not None:
            return await self._stream.input(data)
        return False

    @property
    def task_id(self) -> str | None:
        """The task ID if the process has been started."""
        return self._task_id

    @property
    def stream(self) -> 'BackgroundStream' | None:
        """The underlying BackgroundStream if the process has been started."""
        return self._stream

    @property
    def completed_event(self) -> threading.Event | None:
        """The threading.Event that is set when the background thread completes."""
        if self._stream is not None:
            return self._stream._completed_event
        return None

