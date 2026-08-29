"""Shared security primitives for tool execution (pure, import-cycle safe).

This module is the single home for three helpers that the shell family and
the Python tool both rely on:

- :func:`scrub_child_env` — build a child-process environment that cannot
  leak credentials from the parent environment.
- :func:`redact_sensitive_output` — mask credentials (JWT, PEM keys, API
  tokens, auth headers, URL userinfo, ``password=`` assignments) so secrets
  never reach the model or the export pipeline.
- :func:`validate_workdir` — reject control characters and shell
  metacharacters in a user-supplied working directory.

The pure-Python algorithm for each helper lives once, in the canonical
``kimix_native.tools`` shim (``_compat_*`` fallbacks).  This module keeps the
public API and the native fast-path dispatch and delegates the pure-Python
fallback to the shim, so there is exactly one copy of each algorithm.
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
    "redact_sensitive_output",
    "scrub_child_env",
    "validate_workdir",
]


def scrub_child_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of *env* with credential-looking variables removed.

    A variable is kept when its uppercased name starts with one of the safe
    prefixes (``PATH``, ``HOME``, ``VIRTUAL_ENV``, ``KIMIX_*``, ``SSH_*``,
    ``UV_*``, ...).  Otherwise it is dropped when the uppercased name contains
    any secret substring (``KEY``, ``TOKEN``, ``SECRET``, ``PASSWORD``,
    ``AUTH``, ...).  Everything else is kept.  The match is name-only: a value
    such as ``DATABASE_URL`` (no secret substring in the name) is kept.

    The input dict is never mutated and ``None`` is never returned.  Because a
    *merged* env dict cannot remove variables (keys absent from the merge dict
    stay in the base), scrubbing must run on the base copy *before* any
    caller-provided overrides are merged in.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and all(
        name.isascii() for name in env
    ):
        return _NATIVE_TOOLS.scrub_child_env(env)
    return _compat_tools()._compat_scrub_child_env(env)


def redact_sensitive_output(output: str) -> str:
    """Mask credentials in *output* with ``[REDACTED]``.

    Covers JWTs, PEM private keys, GitHub/GitLab/AWS tokens, auth headers,
    URL userinfo, ``password=``-style assignments and bare bearer tokens.
    Plain text (and short values like ``password=x``) is left untouched.
    """
    if not output:
        return output
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and output.isascii():
        return _NATIVE_TOOLS.redact_sensitive_output(output)
    return _compat_tools()._compat_redact_sensitive_output(output)


def validate_workdir(workdir: str | None) -> str | None:
    """Return ``None`` when *workdir* is safe to use as a working directory.

    Rejects control characters and shell metacharacters (``$ ; | & > < ` ( )
    " ' * ? ! { }``).  ``None``/empty input is always safe.  On rejection
    returns an error message naming the first offending character.
    """
    # Pure-Python delegation (the original had no native fast path): the
    # algorithm lives once in the shim's _compat_validate_workdir.
    return _compat_tools()._compat_validate_workdir(workdir)
