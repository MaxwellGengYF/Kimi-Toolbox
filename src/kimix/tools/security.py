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

Uses only the standard library plus ``regex`` (imported as ``re``), so the
module is cheap to import even from the light ``kimix.tools`` namespace.
"""

import regex as re

__all__ = [
    "redact_sensitive_output",
    "scrub_child_env",
    "validate_workdir",
]


# ── Child environment scrubbing ────────────────────────────────────────────

_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                      "AUTH", "DSN", "WEBHOOK", "CREDS", "BEARER", "APIKEY")

_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM", "TMP", "TEMP", "SHELL",
                      "LOGNAME", "XDG_", "PYTHON", "VIRTUAL_ENV", "CONDA", "KIMIX_", "PROCESSOR_",
                      "PROGRAMFILES", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH", "SYSTEM",
                      "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "OS", "COMPUTERNAME",
                      "USERPROFILE", "TZ", "PWD", "SHLVL", "SSH_", "GIT_", "UV_", "PIP_")


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
    if not env:
        return {}
    scrubbed: dict[str, str] = {}
    for name, value in env.items():
        upper = name.upper()
        if any(upper.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES):
            scrubbed[name] = value
        elif any(substring in upper for substring in _SECRET_SUBSTRINGS):
            continue
        else:
            scrubbed[name] = value
    return scrubbed


# ── Output redaction (moved verbatim from kimix/tools/file/bash/output_enhance.py) ──

_REDACTED = "[REDACTED]"

# JSON Web Tokens (header.payload.signature, header starts "eyJ").
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}")
# PEM private keys (RSA / EC / OPENSSH / DSA / ENCRYPTED variants).
_PEM_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    r".*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    re.DOTALL,
)
# GitHub classic tokens (ghp_ / gho_ / ghu_ / ghr_ / ghs_) and PATs.
_GITHUB_TOKEN_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")
_GITHUB_PAT_RE = re.compile(r"github_pat_[A-Za-z0-9_]{20,}")
# GitLab personal access tokens.
_GITLAB_TOKEN_RE = re.compile(r"glpat-[A-Za-z0-9_-]{15,}")
# AWS access key IDs.
_AWS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")
# Authorization / API key headers.
_AUTH_HEADER_RE = re.compile(
    r"(?i)(authorization|x-api-key|apikey|proxy-authorization)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
# URL userinfo (https://user:pass@host) — keep the scheme, mask credentials.
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
# password= / secret: / api_key= style assignments (min value length 6).
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key)"
    r"\s*[=:]\s*(['\"]?)[^\s'\";]{6,}\2"
)
# Generic high-entropy bearer tokens.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")


def _mask_userinfo(match: re.Match[str]) -> str:
    return match.group(1) + _REDACTED + "@"


def redact_sensitive_output(output: str) -> str:
    """Mask credentials in *output* with ``[REDACTED]``.

    Covers JWTs, PEM private keys, GitHub/GitLab/AWS tokens, auth headers,
    URL userinfo, ``password=``-style assignments and bare bearer tokens.
    Plain text (and short values like ``password=x``) is left untouched.
    """
    if not output:
        return output
    output = _URL_USERINFO_RE.sub(_mask_userinfo, output)
    output = _JWT_RE.sub(_REDACTED, output)
    output = _PEM_RE.sub(_REDACTED, output)
    output = _GITHUB_PAT_RE.sub(_REDACTED, output)
    output = _GITHUB_TOKEN_RE.sub(_REDACTED, output)
    output = _GITLAB_TOKEN_RE.sub(_REDACTED, output)
    output = _AWS_KEY_RE.sub(_REDACTED, output)
    output = _AUTH_HEADER_RE.sub(_REDACTED, output)
    output = _ASSIGNMENT_RE.sub(_REDACTED, output)
    output = _BEARER_RE.sub(_REDACTED, output)
    return output


# ── Workdir validation (moved verbatim from kimix/tools/file/bash/safety.py) ──

# Characters allowed in a ``cwd``/``workdir`` value: alphanumerics plus
# spaces, underscore, dot, dash, backslash, forward slash, colon and tilde.
_WORKDIR_ALLOWED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _.-\\/:~"
)


def validate_workdir(workdir: str | None) -> str | None:
    """Return ``None`` when *workdir* is safe to use as a working directory.

    Rejects control characters and shell metacharacters (``$ ; | & > < ` ( )
    " ' * ? ! { }``).  ``None``/empty input is always safe.  On rejection
    returns an error message naming the first offending character.
    """
    if not workdir:
        return None
    for char in workdir:
        if char not in _WORKDIR_ALLOWED:
            return f"Invalid workdir: character {char!r} is not allowed."
    return None
