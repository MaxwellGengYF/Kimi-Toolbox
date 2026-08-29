"""Output-side enhancements for the shell tool family.

Pure functions only: no tool imports (import-cycle safe).

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

The pure-Python algorithm for each function lives once, in the canonical
``kimix_native.tools`` shim (``_compat_*`` fallbacks).  This module keeps the
public API and the native fast-path dispatch and delegates the pure-Python
fallback to the shim.
"""

from kimi_cli.native_loader import (
    get_compat as _native_get_compat,
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")
# Pure-Python reference implementation (canonical copy lives in the shim);
# resolved lazily to avoid an import-time dependency on the shim package.
_COMPAT_TOOLS = None


def _compat_tools():
    global _COMPAT_TOOLS
    if _COMPAT_TOOLS is None:
        _COMPAT_TOOLS = _native_get_compat("tools")
    return _COMPAT_TOOLS

__all__ = [
    "annotate_failure",
    "interpret_exit_code",
    "is_expected_exit",
    "redact_sensitive_output",
]


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
    if exit_code == 141 and _compat_tools()._compat_has_top_level_pipe(command):
        return "SIGPIPE: an upstream pipeline stage was truncated (expected when piping to head/tail)"
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and command.isascii():
        return _NATIVE_TOOLS.interpret_exit_code(command, exit_code)
    return _compat_tools()._compat_interpret_exit_code(command, exit_code)


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
    return _compat_tools()._compat_is_expected_exit(command, exit_code)


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
    return _compat_tools()._compat_annotate_failure(output, command, exit_code)


from kimix.tools.security import redact_sensitive_output as redact_sensitive_output  # noqa: F401
