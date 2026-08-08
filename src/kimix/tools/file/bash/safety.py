"""Command-side safety helpers for the shell tool family (Bash / Powershell / Run).

Pure functions only: no tool imports (import-cycle safe).  Uses only the
standard library plus ``regex`` (imported as ``re``).

The hardline floor blocks commands that are unconditionally destructive —
recursive deletes of root/home, device formatting, ``dd`` to raw devices,
system power commands, fork bombs, ``kill`` of PID 1, and Windows
``format``/``del`` of a drive root — even when spelling tricks (quoting,
backslash escapes, case) are used to obfuscate them.
"""

import regex as re

from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")

__all__ = [
    "check_hardline_blocked",
    "command_detection_variants",
    "detect_hardline_command",
    "foreground_background_guidance",
    "validate_workdir",
]


def command_detection_variants(command: str) -> list[str]:
    """Produce deobfuscation variants of *command* used to defeat quoting tricks.

    Returns a deduped list (at most 3 entries):

    1. the whitespace-collapsed original,
    2. a deobfuscated variant with quote chars and backslash escapes removed
       (``r\\m -rf /`` -> ``rm -rf /``) and lowercased,
    3. the fully lowercase collapsed original.

    ``detect_hardline_command`` is run over every variant so that any single
    variant spelling that matches a hardline pattern is caught.
    """
    if not command or not command.strip():
        return []
    collapsed = " ".join(command.split())
    deobfuscated = re.sub(r"[\\'\"]", "", collapsed).lower()
    lowered = collapsed.lower()
    variants: list[str] = []
    for variant in (collapsed, deobfuscated, lowered):
        if variant and variant not in variants:
            variants.append(variant)
    return variants or [collapsed]


def _segment_tokens(text: str, start: int) -> list[str]:
    """Collect whitespace-separated tokens after *start* within one shell segment.

    The scan stops at shell separators (``;``, ``&&``, ``||``, ``|``, newline)
    so that a later segment's tokens cannot satisfy an earlier command's rule.
    """
    tail = text[start:]
    tail = re.split(r";|\|\||&&|\||\n", tail, maxsplit=1)[0]
    return tail.split()


def _looks_like_flag(token: str) -> bool:
    """Return True when *token* is an option flag (``-rf``, ``--recursive``,
    or Windows ``/s``-style switches).  A bare ``/`` (the root path) is not."""
    if token.startswith("-") and len(token) > 1:
        return True
    if token.startswith("/") and len(token) > 1 and token[1:].isalpha():
        return True
    return False


def _collect_flags(tokens: list[str]) -> set[str]:
    """Collect short/long flag letters (r/f/s/q) from option tokens."""
    flags: set[str] = set()
    for token in tokens:
        if not _looks_like_flag(token):
            continue
        core = token.lstrip("-/")
        if not core:
            continue
        if "recursive" in core:
            flags.add("r")
        if "force" in core:
            flags.add("f")
        for char in core:
            if char in "rfsq":
                flags.add(char)
    return flags


def _rm_target_is_protected(target: str) -> bool:
    """Return True when *target* is a protected root/home path.

    Handles ``/``, ``/*``, ``/.``-style collapses, ``~``, ``$HOME`` /
    ``${HOME}``, and Windows drive roots (``C:\\``, ``C:\\*``, ``C:``).
    Deeper paths (``/tmp/build``, ``C:\\Windows``) are never protected.
    """
    t = target.strip().strip("\"'").lower()
    t = t.replace("${home}", "$home")
    if t.rstrip("/\\") in ("~", "$home"):
        return True
    # Windows drive root, optionally with trailing separator and/or glob.
    if re.match(r"^[a-z]:[\\/]?(?:[\\/]?\*)?$", t):
        return True
    if t.startswith("/"):
        parts = [p for p in t.split("/") if p not in ("", ".", "..")]
        if not parts or parts == ["*"]:
            return True
    return False


def _detect_recursive_delete(text: str) -> str | None:
    """Detect recursive/forced deletes of protected roots (``rm``/``rmdir``/``del``)."""
    for match in re.finditer(r"\b(rm|rmdir|del)(?:\.exe)?\b", text):
        command_word = match.group(1)
        tokens = _segment_tokens(text, match.end())
        flags = _collect_flags(tokens)
        if command_word == "rm" and not ({"r", "f"} & flags):
            continue
        if command_word == "rmdir" and not ({"r", "s"} & flags):
            continue
        if command_word == "del" and not ({"r", "f", "s"} & flags):
            continue
        targets = [t for t in tokens if not _looks_like_flag(t)]
        for target in targets:
            if _rm_target_is_protected(target):
                return f"Recursive delete of protected root/home (`{target}`)"
    return None


def detect_hardline_command(command: str) -> tuple[bool, str | None]:
    """Return ``(True, description)`` when *command* matches a hardline pattern.

    The check is case-insensitive and tolerant of extra whitespace; callers
    should additionally run :func:`command_detection_variants` (see
    :func:`check_hardline_blocked`) to defeat quote/escape obfuscation.
    """
    if not command or not command.strip():
        return False, None
    text = " ".join(command.split()).lower()

    # 1. Recursive delete of root / home / Windows drive root.
    desc = _detect_recursive_delete(text)
    if desc is not None:
        return True, desc

    # 2. Disk formatting (mkfs.* formats devices).
    if re.search(r"\bmkfs(?:\.\w+)?\b", text):
        return True, "Disk formatting command (`mkfs`) is blocked"

    # 3. dd writing to a raw device (of=/dev/sd*, nvme*, disk*, rdisk*).
    if re.search(r"\bdd\b", text) and re.search(
        r"\bof=/dev/(?:sd|nvme|disk|rdisk)[a-z0-9]*", text
    ):
        return True, "`dd` writing to a raw device is blocked"

    # 4. System power commands: shutdown / reboot / poweroff / halt.
    first = text.split()[0] if text.split() else ""
    if first in ("shutdown", "reboot", "poweroff", "halt"):
        return True, f"System `{first}` command is blocked"

    # 5. Fork bomb: `:(){ :|:& };:`
    if re.search(r":\(\)\{", text) and re.search(r":\|:&", text):
        return True, "Fork bomb pattern detected"

    # 6. kill targeting PID 1 (or $PPID — kills the parent shell).
    for match in re.finditer(r"\bkill(?:\.exe)?\b", text):
        tokens = _segment_tokens(text, match.end())
        targets = [t for t in tokens if not _looks_like_flag(t)]
        for target in targets:
            if target == "1" or target == "$ppid":
                return True, "`kill` targeting PID 1 (or `$PPID`) is blocked"

    # 7. Windows: format <drive>: and del /f /s /q <drive>:\*.
    for match in re.finditer(r"\bformat(?:\.exe)?\b", text):
        tokens = _segment_tokens(text, match.end())
        for target in tokens:
            if re.match(r"^[a-z]:[\\/]?$", target):
                return True, "Windows `format` on a drive is blocked"

    return False, None


def check_hardline_blocked(command: str) -> tuple[bool, str | None]:
    """Single entry point: run the hardline detector over all deobfuscation
    variants of *command*.

    Returns ``(True, "<human description>")`` when any variant matches, else
    ``(False, None)``.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and command.isascii():
        return _NATIVE_TOOLS.check_hardline_blocked(command)
    for variant in command_detection_variants(command):
        blocked, desc = detect_hardline_command(variant)
        if blocked:
            return True, desc
    return False, None


# ``validate_workdir`` moved to :mod:`kimix.tools.security`; re-exported
# here for compatibility with existing shell-tool callers.
from kimix.tools.security import validate_workdir as validate_workdir  # noqa: F401


_LONG_RUNNING_PATTERNS = [
    r"\b(?:npm|pnpm|yarn|bun)\s+run\s+(?:dev|start|serve|watch)\b",
    r"\bnext\s+dev\b",
    r"\bvite\b",
    r"\bnodemon\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bpython\s+-m\s+http\.server\b",
    r"\bdocker\s+compose\s+up\b",
    r"\bdocker-compose\s+up\b",
    r"&\s*$",
    r"\bnohup\b",
    r"\bsetsid\b",
]

_FG_BG_HINT = (
    "Long-running process detected. Consider mode='send' (background) + "
    "TaskOutput to avoid blocking on timeout."
)


def _strip_quoted(text: str) -> str:
    """Remove single/double-quoted spans so keywords inside strings do not
    false-positive the long-running detection."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", text)


def foreground_background_guidance(command: str) -> str | None:
    """Return a short hint when *command* looks long-lived (dev servers,
    watchers, ``docker compose up``, trailing ``&``, ``nohup``/``setsid``).

    Quoted content is ignored so keywords inside strings don't match.
    Returns ``None`` for ordinary commands.
    """
    if not command or not command.strip():
        return None
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and command.isascii():
        return _NATIVE_TOOLS.foreground_background_guidance(command)
    stripped = _strip_quoted(command)
    text = " ".join(stripped.split())
    if any(re.search(pattern, text) for pattern in _LONG_RUNNING_PATTERNS):
        return _FG_BG_HINT
    return None
