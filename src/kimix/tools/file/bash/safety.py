"""Command-side safety helpers for the shell tool family (Bash / Powershell / Run).

Pure functions only: no tool imports (import-cycle safe).  Uses only the
standard library plus ``regex`` (imported as ``re``).

The hardline floor blocks commands that are unconditionally destructive —
recursive deletes of root/home, device formatting, ``dd`` to raw devices,
system power commands, fork bombs, ``kill`` of PID 1, and Windows
``format``/``del`` of a drive root — even when spelling tricks (quoting,
backslash escapes, case) are used to obfuscate them.

The self-kill guard (:func:`self_kill_hint`) stops the LLM backend from
accidentally terminating the very process hosting the agent: kill-style
commands (``kill``/``tskill``/``taskkill``/``Stop-Process``/``pkill``/
``killall``/``wmic ... delete``) whose target PID is the agent process or
one of its ancestors, or whose image-name/pattern target matches the
agent's own image name or command line.  PID targets reached through a
shell loop variable — ``for pid in 4100 5000; do taskkill /PID $pid; done``
(bash) and ``foreach ($pid in 4100,5000) { Stop-Process -Id $pid }``
(PowerShell) — are resolved against the loop's literal PID list, so a
batch kill that includes the agent's own PID is still blocked.
"""

import os
import sys

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
    "detect_self_kill",
    "foreground_background_guidance",
    "self_kill_hint",
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


# ── Self-kill guard ──────────────────────────────────────────────────────────
#
# The LLM backend occasionally resolves the wrong PID (or reaches for a broad
# image-name/pattern kill such as ``taskkill /IM python.exe /F`` or
# ``pkill -f python``) and would terminate the very process hosting the
# agent.  ``self_kill_hint`` compares kill targets against the current PID,
# its ancestors, and the agent's own image name / command line, returning a
# smart hint so the tool can refuse to execute such a command.

_AGENT_PIDS_CACHE: set[int] | None = None
_AGENT_IMAGE_NAMES_CACHE: set[str] | None = None
_AGENT_CMDLINE_CACHE: str | None = None

# Words that make a following ``kill`` act on container/remote entities rather
# than host PIDs (``docker kill <container>``, ``kubectl kill`` ...).
_KILL_PRECEDING_SKIP = frozenset({"docker", "podman", "kubectl", "compose"})

# Executable-style suffixes stripped when deriving image-name stems.
_EXECUTABLE_SUFFIXES = ("exe", "com", "bat", "cmd", "py", "sh")


def _posix_ppid(pid: int) -> int | None:
    """Return the parent PID of *pid* via ``/proc`` (None when unavailable)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            stat = fh.read()
        # ``comm`` (field 2) is parenthesized and may contain spaces/parens;
        # the ppid is the second whitespace-separated field after the last ``)``.
        tail = stat.rsplit(")", 1)[1].split()
        return int(tail[1])
    except (OSError, ValueError, IndexError):
        return None


def _windows_parent_map() -> dict[int, int]:
    """Return a ``pid -> parent pid`` map via a Toolhelp32 snapshot.

    Returns an empty dict on any failure so the caller degrades to just
    ``getpid()``/``getppid()``.
    """
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),  # ULONG_PTR
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return {}
    result: dict[int, int] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
        while ok:
            result[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snapshot))
    return result


def _agent_pids() -> set[int]:
    """Return the current PID plus all ancestor PIDs (cached for process life).

    Killing any of these terminates the agent session.  The ancestor walk
    never fails hard: worst case it degenerates to ``{getpid(), getppid()}``.
    """
    global _AGENT_PIDS_CACHE
    if _AGENT_PIDS_CACHE is not None:
        return _AGENT_PIDS_CACHE
    pids = {os.getpid()}
    ppid = os.getppid()
    if ppid > 0:
        pids.add(ppid)
    current = ppid
    if os.name == "nt":
        try:
            parent_map = _windows_parent_map()
        except Exception:
            parent_map = {}
        for _ in range(64):
            parent = parent_map.get(current, 0)
            if parent <= 0 or parent in pids:
                break
            pids.add(parent)
            current = parent
    else:
        for _ in range(64):
            parent = _posix_ppid(current)
            if parent is None or parent <= 1 or parent in pids:
                break
            pids.add(parent)
            current = parent
    _AGENT_PIDS_CACHE = pids
    return pids


def _split_image_name(name: str) -> tuple[str, str]:
    """Return ``(basename, stem)`` of *name*, lowercased.

    The stem drops a known executable suffix (``python.exe`` -> ``python``);
    other dotted names (``python3.12``) keep their full form as the stem.
    """
    base = re.split(r"[\\/]", name.strip().strip("\"'"))[-1].lower()
    stem, dot, ext = base.rpartition(".")
    if dot and stem and ext in _EXECUTABLE_SUFFIXES:
        return base, stem
    return base, base


def _agent_image_names() -> set[str]:
    """Return lowercase image names identifying the agent process (cached).

    Covers the interpreter/launcher basename (``python.exe``) and its stem
    (``python``) for both ``sys.executable`` and ``sys.argv[0]`` (e.g. the
    ``kimi`` console script), so name-based kills of the agent are caught.
    """
    global _AGENT_IMAGE_NAMES_CACHE
    if _AGENT_IMAGE_NAMES_CACHE is not None:
        return _AGENT_IMAGE_NAMES_CACHE
    candidates = [sys.executable]
    if sys.argv and sys.argv[0]:
        candidates.append(sys.argv[0])
    names: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        base, stem = _split_image_name(candidate)
        # Names shorter than 3 chars (e.g. a ``5.py`` scratch-script stem)
        # are too generic to match safely.
        if len(base) >= 3:
            names.add(base)
        if len(stem) >= 3:
            names.add(stem)
    _AGENT_IMAGE_NAMES_CACHE = names
    return names


def _agent_cmdline() -> str:
    """Return the agent's own command line (cached); used for ``pkill -f``."""
    global _AGENT_CMDLINE_CACHE
    if _AGENT_CMDLINE_CACHE is None:
        argv = list(sys.argv) if sys.argv else []
        _AGENT_CMDLINE_CACHE = " ".join([sys.executable, *argv])
    return _AGENT_CMDLINE_CACHE


def _segment_text(text: str, start: int) -> str:
    """Return the shell segment (up to ``;``/``&&``/``||``/``|``/newline) at *start*."""
    return re.split(r";|\|\||&&|\||\n", text[start:], maxsplit=1)[0]


def _numeric_pid_targets(tokens: list[str]) -> list[int]:
    """Return integer PIDs among *tokens* (flags excluded; comma lists split)."""
    pids: list[int] = []
    for token in tokens:
        if _looks_like_flag(token):
            continue
        for part in token.split(","):
            part = part.strip().strip("\"'()")
            if part.isdigit():
                pids.append(int(part))
                continue
            # PowerShell expression style: ``2100).Kill()`` from e.g.
            # ``(Get-Process -Id 2100).Kill()`` — leading digits followed by
            # ``)`` or ``.`` still denote a PID.
            expr = re.match(r"^(\d+)[).]", part)
            if expr:
                pids.append(int(expr.group(1)))
    return pids


# Loop-headers that bind a variable to a literal PID list.  The bash form is
# ``for pid in 4100 5000; do ...``, the PowerShell form is
# ``foreach ($pid in 4100,5000) { ... }``.  Only *literal* numeric list
# entries can be resolved statically; unresolvable sources (command
# substitution, globs, other variables) never feed the guard so unknown
# data is never assumed to be the agent.
_LOOP_PID_HEADERS = (
    # bash/POSIX: ``for pid in 4100 5000; do`` (list ends at the first ``;``)
    r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^;]+)",
    # PowerShell: ``foreach ($pid in 4100,5000) {`` (list ends at ``)``)
    r"\bforeach\s*\(\s*\$([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^)]+)\)",
)


# Characters that mark a loop source as not statically resolvable
# (``$(pgrep ...)``, ``${list}``, globs, backticks, tilde expansion).
_UNRESOLVABLE_LOOP_SOURCE_CHARS = frozenset("$*?[`~")


def _loop_pid_sources(text: str) -> dict[str, list[int]]:
    """Map loop-variable names (lowercased) to their literal numeric PID lists.

    Detects bash ``for pid in 4100 5000; do ...`` and PowerShell
    ``foreach ($pid in 4100,5000) { ... }`` headers.  Comma-separated and
    space-separated lists are both accepted.  Entries that are not plain
    integers (substitutions, globs, word lists) are skipped, so the returned
    map only contains PIDs the command provably iterates over.
    """
    sources: dict[str, list[int]] = {}
    for pattern in _LOOP_PID_HEADERS:
        for match in re.finditer(pattern, text):
            var = match.group(1).lower()
            pids = sources.setdefault(var, [])
            for token in match.group(2).split():
                if _UNRESOLVABLE_LOOP_SOURCE_CHARS.intersection(token):
                    continue
                for part in token.split(","):
                    part = part.strip().strip("\"'")
                    if part.isdigit():
                        pid = int(part)
                        if pid not in pids:
                            pids.append(pid)
    return sources


def _variable_pid_hit(
    tokens: list[str],
    loop_sources: dict[str, list[int]],
    protected_pids: set[int],
    via: str,
) -> str | None:
    """Return a description when *tokens* pass a protected PID via a loop var.

    Kill targets written as ``$pid`` / ``${pid}`` (optionally quoted) inside
    e.g. ``for pid in 4100 5000; do taskkill /PID $pid; done`` are resolved
    through *loop_sources* (see :func:`_loop_pid_sources`); when any bound
    PID is protected the kill is described.  Returns ``None`` when no token
    names a loop variable bound to a protected PID.
    """
    for token in tokens:
        stripped = token.strip().strip("\"'")
        match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", stripped)
        if not match:
            continue
        pids = loop_sources.get(match.group(1).lower())
        if not pids:
            continue
        pid_list = ", ".join(str(pid) for pid in pids)
        for pid in pids:
            if pid in protected_pids:
                return (
                    f"kills PID {pid} via `{via}` through loop variable "
                    f"`${match.group(1)}` (bound to PIDs {pid_list}), which "
                    "is the agent process or one of its parent processes"
                )
    return None


def _name_kill_hit(token: str, image_names: set[str]) -> str | None:
    """Return the matched agent image name when *token* names the agent process.

    Handles exact names (``python.exe``), suffix-less stems (``python``) and
    PowerShell-style trailing wildcards (``python*``).
    """
    base, stem = _split_image_name(token)
    if not base or not re.search(r"[a-z0-9]", base, re.IGNORECASE):
        return None
    if base.endswith("*"):
        prefix = base[:-1]
        if len(prefix) < 3:
            return None
        for name in image_names:
            if name.startswith(prefix):
                return name
        return None
    for name in image_names:
        if base == name or stem == name:
            return name
    return None


def _pattern_kill_hit(pattern: str, haystacks: list[str]) -> str | None:
    """Return the haystack matched by a pkill-style regex *pattern*.

    pkill treats the pattern as an extended regex (substring search); a
    pattern that fails to compile falls back to a plain substring test.
    Tokens without any alphanumeric character (``()``, ``$@`` from shell
    function definitions/wrappers) are never treated as patterns.
    """
    p = pattern.strip().strip("\"'")
    if not p or not re.search(r"[a-z0-9]", p, re.IGNORECASE):
        return None
    for haystack in haystacks:
        if not haystack:
            continue
        try:
            if re.search(p, haystack, re.IGNORECASE):
                return haystack
        except re.error:
            if p.lower() in haystack.lower():
                return haystack
    return None


def _pkill_full_match(tokens: list[str]) -> bool:
    """Return True when pkill *tokens* include the ``-f``/``--full`` flag."""
    for token in tokens:
        tl = token.lower()
        if tl == "--full":
            return True
        if tl.startswith("-") and not tl.startswith("--") and "f" in tl[1:]:
            return True
    return False


def detect_self_kill(
    command: str,
    *,
    protected_pids: set[int] | None = None,
    image_names: set[str] | None = None,
    cmdline: str | None = None,
) -> str | None:
    """Return a short description when *command* would kill the agent process.

    Detects kill-style commands — POSIX ``kill``, Windows ``tskill`` /
    ``taskkill`` (``/PID``, ``/FI "PID eq n"``, ``/IM``), PowerShell
    ``Stop-Process`` (``-Id``/``-Name``, plus ``Get-Process`` piped into a
    kill), ``pkill``/``killall`` name patterns (``pkill -f`` matches against
    the full agent command line), and ``wmic ... ProcessId=n ... delete`` —
    whose target is one of *protected_pids* or whose name/pattern target
    matches the agent's own image name.  Returns ``None`` for safe commands.

    The keyword arguments exist for testing; the defaults describe the live
    agent process (current PID + ancestors, own image names, own cmdline).
    """
    if not command or not command.strip():
        return None
    if protected_pids is None:
        protected_pids = set(_agent_pids())
    if image_names is None:
        image_names = set(_agent_image_names())
    if cmdline is None:
        cmdline = _agent_cmdline()
    if not protected_pids:
        return None
    image_names = {n.lower() for n in image_names if n}
    text = " ".join(command.split()).lower()
    # PID lists bound to shell loop variables (``for pid in 4100 ...`` /
    # ``foreach ($pid in 4100,...)``) so ``taskkill /PID $pid`` inside a
    # batch loop that includes the agent's PID is still caught.
    loop_sources = _loop_pid_sources(text)

    def _pid_hit(tokens: list[str], via: str) -> str | None:
        for pid in _numeric_pid_targets(tokens):
            if pid in protected_pids:
                return (
                    f"targets PID {pid} via {via}, which is the agent process "
                    "or one of its parent processes"
                )
        return None

    # 1. POSIX kill / Windows tskill: numeric PID targets.
    for match in re.finditer(r"\b(?:kill|tskill)(?:\.exe)?\b", text):
        word = match.group(0)
        if word.startswith("kill"):
            prev = text[: match.start()].rstrip().split()
            if prev and prev[-1] in _KILL_PRECEDING_SKIP:
                continue  # docker/podman/kubectl kill: container, not a host PID
        tokens = _segment_tokens(text, match.end())
        hit = _pid_hit(tokens, f"`{word}`")
        if hit is not None:
            return hit
        hit = _variable_pid_hit(tokens, loop_sources, protected_pids, f"`{word}`")
        if hit is not None:
            return hit

    # 2. taskkill: /PID values and /FI pid filters (numeric scan), /IM names.
    for match in re.finditer(r"\btaskkill(?:\.exe)?\b", text):
        tokens = _segment_text(text, match.end()).split()
        hit = _pid_hit(tokens, "`taskkill`")
        if hit is not None:
            return hit
        hit = _variable_pid_hit(tokens, loop_sources, protected_pids, "`taskkill`")
        if hit is not None:
            return hit
        for index, token in enumerate(tokens[:-1]):
            if token == "/im":
                name_hit = _name_kill_hit(tokens[index + 1], image_names)
                if name_hit is not None:
                    return (
                        f"kills by image name `{name_hit}` via `taskkill /IM`, "
                        "which also matches the agent process"
                    )

    # 3. Stop-Process -Id / -Name, and Get-Process piped into a kill.
    for match in re.finditer(r"\bstop-process\b", text):
        tokens = _segment_tokens(text, match.end())
        hit = _pid_hit(tokens, "`Stop-Process`")
        if hit is not None:
            return hit
        hit = _variable_pid_hit(tokens, loop_sources, protected_pids, "`Stop-Process`")
        if hit is not None:
            return hit
        for token in tokens:
            if _looks_like_flag(token):
                continue
            name_hit = _name_kill_hit(token, image_names)
            if name_hit is not None:
                return (
                    f"kills by process name `{name_hit}` via `Stop-Process`, "
                    "which also matches the agent process"
                )
    if "stop-process" in text or "| kill" in text or ".kill()" in text:
        for match in re.finditer(r"\bget-process\b", text):
            tokens = _segment_tokens(text, match.end())
            hit = _pid_hit(tokens, "`Get-Process` piped to a kill")
            if hit is not None:
                return hit
            hit = _variable_pid_hit(
                tokens, loop_sources, protected_pids, "`Get-Process` piped to a kill"
            )
            if hit is not None:
                return hit
            for token in tokens:
                if _looks_like_flag(token):
                    continue
                name_hit = _name_kill_hit(token, image_names)
                if name_hit is not None:
                    return (
                        f"kills by process name `{name_hit}` via `Get-Process` "
                        "piped to a kill, which also matches the agent process"
                    )

    # 4. pkill / killall: name patterns; ``pkill -f`` matches full cmdlines.
    for match in re.finditer(r"\b(?:pkill|killall)(?:\.exe)?\b", text):
        is_pkill = match.group(0).startswith("pkill")
        segment_tokens = _segment_tokens(text, match.end())
        full = is_pkill and _pkill_full_match(segment_tokens)
        for token in segment_tokens:
            if _looks_like_flag(token):
                continue
            if is_pkill:
                haystacks = sorted(image_names)
                if full and cmdline:
                    haystacks.append(cmdline)
                pattern_hit = _pattern_kill_hit(token, haystacks)
                if pattern_hit is not None:
                    display = token.strip().strip("\"'")
                    return (
                        f"kills processes matching `{display}` via "
                        f"`pkill{' -f' if full else ''}`, which also matches "
                        "the agent process"
                    )
            else:
                name_hit = _name_kill_hit(token, image_names)
                if name_hit is not None:
                    return (
                        f"kills by process name `{name_hit}` via `killall`, "
                        "which also matches the agent process"
                    )

    # 5. wmic process where ProcessId=<pid> delete / call terminate.
    for match in re.finditer(r"\bwmic(?:\.exe)?\b", text):
        segment = _segment_text(text, match.end())
        if not re.search(r"\b(?:delete|terminate)\b", segment):
            continue
        pid_match = re.search(r"processid\s*=\s*(\d+)", segment)
        if pid_match and int(pid_match.group(1)) in protected_pids:
            return (
                f"targets PID {pid_match.group(1)} via `wmic`, which is the "
                "agent process or one of its parent processes"
            )
        var_match = re.search(
            r"processid\s*=\s*\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", segment
        )
        if var_match:
            pids = loop_sources.get(var_match.group(1).lower(), [])
            for pid in pids:
                if pid in protected_pids:
                    pid_list = ", ".join(str(p) for p in pids)
                    return (
                        f"targets PID {pid} via `wmic` through loop variable "
                        f"`${var_match.group(1)}` (bound to PIDs {pid_list}), "
                        "which is the agent process or one of its parent "
                        "processes"
                    )

    return None


_SELF_KILL_GUIDANCE = (
    "If you meant to stop a different process, re-check its PID first "
    "(`tasklist` / `Get-Process` / `ps aux`) and retry with a PID that does "
    "not belong to the agent. If the target merely shares the agent's image "
    "name, terminate that specific PID instead of a name/pattern match. If "
    "you really intend to stop or restart the agent itself, ask the user to "
    "do it from outside this session."
)


def self_kill_hint(command: str) -> str | None:
    """Return a smart hint when *command* would kill the running agent process.

    Runs :func:`detect_self_kill` over every deobfuscation variant (see
    :func:`command_detection_variants`) so quoting/backslash/case tricks are
    caught.  The returned message explains what was detected, which PID is
    the agent, and safer alternatives; ``None`` means the command is safe.
    """
    if not command or not command.strip():
        return None
    protected = set(_agent_pids())
    names = set(_agent_image_names())
    cmdline = _agent_cmdline()
    for variant in command_detection_variants(command):
        desc = detect_self_kill(
            variant, protected_pids=protected, image_names=names, cmdline=cmdline
        )
        if desc is not None:
            return (
                f"The command {desc}. Executing it would terminate this agent "
                f"session (current agent PID: {os.getpid()}). "
                + _SELF_KILL_GUIDANCE
            )
    return None
