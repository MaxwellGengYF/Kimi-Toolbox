"""Output-side enhancements for the shell tool family.

Pure functions only: no tool imports (import-cycle safe).  Uses only the
standard library plus ``regex`` (imported as ``re``).

Provides:

- :func:`interpret_exit_code` — explain what a non-zero exit code means for
  well-known commands (grep "no matches" is normal, etc.).
- :func:`is_expected_exit` — decide whether a non-zero exit code is the normal,
  expected outcome for a command (grep "no matches", diff "files differ", a
  truncated pipeline), so shell tools report it as an informative success
  instead of a hard failure with retry guidance.
- :func:`annotate_failure` — a single actionable hint for common failure
  signatures in command output (command not found, missing file, missing
  Python module, permission denied).
- :func:`redact_sensitive_output` — mask credentials (JWT, PEM keys, API
  tokens, auth headers, URL userinfo, ``password=`` assignments) so secrets
  never reach the model or the export pipeline.  The implementation moved to
  :mod:`kimix.tools.security`; this module re-exports it for compatibility.
"""

import regex as re

from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")

__all__ = [
    "annotate_failure",
    "interpret_exit_code",
    "is_expected_exit",
    "redact_sensitive_output",
]


def _base_command_name(command: str) -> str:
    """Return the first non-assignment command word, directory-stripped.

    ``/usr/bin/grep -r foo`` -> ``grep``; ``FOO=1 git diff`` -> ``git``;
    ``python -m http.server`` -> ``python``.
    """
    last_segment = command.strip().split("&&")[-1].split("||")[-1]
    last_segment = last_segment.split("|")[-1].split(";")[-1].strip()
    for word in last_segment.split():
        if "=" in word and not word.startswith("-"):
            continue
        stem = word.split("/")[-1]
        return stem[:-4] if stem.lower().endswith(".exe") else stem
    return ""


def _has_top_level_pipe(command: str) -> bool:
    """Return True when *command* contains a top-level ``|`` pipeline operator.

    Quote- and substitution-aware: pipes inside quotes or subshells/command
    substitutions are ignored, and the ``||`` logical-OR operator is not a
    pipeline.  Used to decide whether a SIGPIPE (141) exit is the normal
    consequence of truncating a pipeline (``producer | head``).
    """
    if not command:
        return False
    quote: str | None = None
    escaped = False
    depth = 0
    n = len(command)
    for i, ch in enumerate(command):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if ch != "|" or depth != 0:
            continue
        # A ``||`` logical-OR operator is not a pipeline: skip both pipes.
        if i + 1 < n and command[i + 1] == "|":
            continue
        if i > 0 and command[i - 1] == "|":
            continue
        return True
    return False


def _is_expected_exit_py(command: str, exit_code: int | None) -> bool:
    """Pure-Python expected-exit classification (mirrored by the shim)."""
    if exit_code is None or exit_code == 0:
        return False
    if exit_code == 141 and _has_top_level_pipe(command):
        return True
    name = _base_command_name(command).lower()
    if exit_code != 1:
        return False
    return (
        name in ("grep", "egrep", "fgrep", "rg", "ag", "ack")
        or name in ("diff", "colordiff")
        or name in ("test", "[")
        or name == "find"
    )


def interpret_exit_code(command: str, exit_code: int | None) -> str | None:
    """Explain a non-zero exit code for well-known commands.

    Returns ``None`` for exit 0 / unknown exit codes, or when the meaning is
    not special.  The caller keeps ``ToolError`` semantics; this only enriches
    the message.
    """
    if exit_code is None or exit_code == 0:
        return None
    # Checked before the native fast path: the compiled kernel predates the
    # SIGPIPE rule, so the pipeline-truncation meaning must be decided here to
    # stay identical under native and pure-Python execution.
    if exit_code == 141 and _has_top_level_pipe(command):
        return "SIGPIPE: an upstream pipeline stage was truncated (expected when piping to head/tail)"
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and command.isascii():
        return _NATIVE_TOOLS.interpret_exit_code(command, exit_code)
    name = _base_command_name(command).lower()
    code = exit_code

    if name in ("grep", "egrep", "fgrep", "rg", "ag", "ack") and code == 1:
        return "No matches found (not an error)"
    if name in ("diff", "colordiff") and code == 1:
        return "Files differ (expected, not an error)"
    if name == "find" and code == 1:
        return "Some directories were inaccessible (partial results may still be valid)"
    if name in ("test", "[") and code == 1:
        return "Condition evaluated to false (expected, not an error)"
    if name == "curl":
        notes = {
            6: "Could not resolve host (DNS failure)",
            7: "Failed to connect to host",
            22: "HTTP error (server returned an error status)",
            28: "Connection timed out",
        }
        if code in notes:
            return notes[code]
    if name == "git" and code == 1:
        return "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"
    return None


def is_expected_exit(command: str, exit_code: int | None) -> bool:
    """Return True when *exit_code* is a normal, expected outcome for *command*.

    Covers grep/diff/test/find exit 1 ("no matches", "files differ", …) and
    SIGPIPE (141) inside a pipeline (``producer | head`` truncation).  Shell
    tools use this to report such commands as an informative success instead of
    a hard failure — previously a grep that simply found nothing was surfaced
    as ``failed ... Edit the saved script and run it again``, which misled the
    agent into retrying a command that ran exactly as intended.
    """
    if exit_code is None or exit_code == 0:
        return False
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and command.isascii():
        impl = getattr(_NATIVE_TOOLS, "is_expected_exit", None)
        if impl is not None:
            return impl(command, exit_code)
    return _is_expected_exit_py(command, exit_code)


def annotate_failure(output: str, command: str, exit_code: int | None) -> str | None:
    """Return a single actionable hint for a failed command, or ``None``.

    Scans the first ``min(len(output), 4000)`` characters case-insensitively.
    The *command* argument is kept for signature compatibility (hints are
    output-driven today) and future per-command rules.
    """
    if not output:
        return None
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and output.isascii():
        return _NATIVE_TOOLS.annotate_failure(output, command, exit_code)
    sample = output[:4000]
    lowered = sample.lower()

    if (
        "command not found" in lowered
        or "not recognized as an internal or external command" in lowered
    ):
        return (
            "The command was not found. Check it is installed and on PATH "
            "(use `which <cmd>` / `Get-Command <cmd>`)."
        )
    if "no such file or directory" in lowered:
        return (
            "A file or directory referenced by the command does not exist. "
            "Verify the path with `Glob`/ReadFile."
        )
    module_match = re.search(
        r"modulenotfounderror:\s*no module named '([^']+)'", sample, re.IGNORECASE
    )
    if module_match:
        missing = module_match.group(1)
        return (
            f"Python module {missing} is missing. Install it "
            f"(e.g. `pip install {missing}`) or check the environment."
        )
    if "permission denied" in lowered:
        return "Permission denied. Check file permissions (ls -la) or ownership."
    return None


from kimix.tools.security import redact_sensitive_output as redact_sensitive_output  # noqa: F401
