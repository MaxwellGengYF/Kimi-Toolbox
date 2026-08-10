"""Bash tool that executes commands via the system bash executable."""


import asyncio
import contextlib
import functools
import ntpath
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import orjson
import regex as re
from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.display import ShellDisplayBlock
from pydantic import BaseModel, Field, model_validator

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimix.tools.common import (
    ProcessTask,
    _build_session_output_block,
    _env_with_rg_bin_path,
    _extract_export_path,
    _interactive_scope_text,
    _maybe_export_output_async,
    _maybe_export_rtk_original_async,
    _maybe_rewrite_shell_command_with_rtk,
    _summarize_long_output_async,
    _token_filter_output,
)
from kimix.tools.file.bash.bash_fix import (
    bash_compatibility_prelude,
    # Kept in the module namespace even though one-shot preparation now goes
    # through ``shell_common.prepare_bash_command``: tests patch
    # ``bash_tool.fix_bash_command`` to assert the fixer is never run for
    # forbidden source commands.
    fix_bash_command,
)
from kimix.tools.prompt_common import (
    accepts_alias_text,
    cwd_field,
    deduplicate_output_field,
    max_lines_field,
    mode_field,
    normalize_mode_validator,
    shell_cmd_required_validator,
    task_id_field,
    timeout_field,
    wait_for_pattern_field,
)
from kimix.tools.file.bash.output_enhance import (
    annotate_failure,
    interpret_exit_code,
    redact_sensitive_output,
)
from kimix.tools.file.bash.safety import (
    check_hardline_blocked,
    foreground_background_guidance,
    validate_workdir,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_PARSE = _native_get_module("parse")

if TYPE_CHECKING:
    from kimix.tools.background.utils import BackgroundStream

USE_SYSTEM_SHELL = True


def _encode_startup_script(script: str) -> str:
    """Encode a multi-line startup script as a self-decoding one-liner.

    Long multi-line scripts passed through the Windows command line to MSYS2
    bash get corrupted (argv quoting heuristics).  A single-line base64+gzip
    payload contains only safe ASCII characters and sidesteps every quoting
    issue; the receiving shell decodes and evals it.
    """
    import gzip

    import pybase64

    payload = pybase64.b64encode(gzip.compress(script.encode("utf-8"))).decode("ascii")
    return "eval \"$(printf '%s' '" + payload + "' | base64 -d | gzip -d)\""

# Default Windows shell policy: "Git Bash first, PowerShell as fallback".  The
# Bash tool is enabled whenever a real bash (typically shipped with Git for
# Windows) is installed, and the Powershell tool is enabled only when no bash
# exists (no git install).  Set to True to always prefer PowerShell on Windows
# (disabling the Bash tool there).
USE_SYSTEM_PWSH_ON_WINDOWS = False


def _bash_subprocess_env() -> dict[str, str]:
    """Return the environment for a bash subprocess.

    Starts from ``_env_with_rg_bin_path()`` (shared ``bin`` dir first on
    PATH).  On Windows it additionally opts the Git Bash child out of MSYS
    argv path conversion: without this, Git Bash rewrites arguments that
    *look* like Unix paths (``/FO``, ``/TN``, ``/Create``) into
    ``C:/.../git/FO``-style paths, breaking native Windows tools such as
    ``tasklist``, ``schtasks``, ``wmic``, and ``cmd /c`` invocations.
    Git for Windows bash honors ``MSYS_NO_PATHCONV`` only, while real
    MSYS2/Cygwin bash honors ``MSYS2_ARG_CONV_EXCL``; ``*`` disables all
    conversion.  ``setdefault`` is used so explicit user settings win.
    """
    env = _env_with_rg_bin_path()
    if sys.platform == "win32":
        env.setdefault("MSYS_NO_PATHCONV", "1")
        env.setdefault("MSYS2_ARG_CONV_EXCL", "*")
    return env


def _is_windows_apps_stub(path: str) -> bool:
    """Return True if *path* points into the WindowsApps directory (Store stub).

    Windows ships ``bash.exe`` in ``WindowsApps`` as an App Execution Alias
    (Microsoft Store stub) that only offers to install WSL; it is not a real
    bash and must never be treated as one.
    """
    normalized = os.path.normpath(path).replace("/", "\\")
    return "WindowsApps" in normalized.split("\\")


# Probe used to smoke-test a candidate bash: launch *external* MSYS programs,
# not just builtins.  A builtin-only probe (e.g. ``--version``) passes even
# when Git for Windows cannot fork children under system-wide Mandatory ASLR
# (``ForceRelocateImages``) — bash starts, prints a version, and every real
# command then fails.  ``--noprofile --norc`` keeps a broken login
# post-install from falsely condemning an otherwise usable bash.
_BASH_EXTERNAL_PROGRAM_PROBE = "/usr/bin/true; /usr/bin/cat --version >/dev/null"


def _bash_runs(bash_path: str) -> bool:
    """Smoke-test *bash_path*: return True when it can launch external programs.

    Guards against ``bash`` entries that exist on disk but cannot actually run
    (broken installs, WSL launchers without a distribution, corrupt binaries,
    or a Git for Windows install whose MSYS2 runtime cannot fork children
    under Windows Mandatory ASLR).  A bash that fails this probe is treated as
    "no bash" so that PowerShell becomes the fallback shell on Windows.
    """
    try:
        result = subprocess.run(
            [bash_path, "--noprofile", "--norc", "-c", _BASH_EXTERNAL_PROGRAM_PROBE],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _where_git_executables() -> list[str]:
    """Return candidate git.exe paths reported by ``where.exe git``."""
    try:
        result = subprocess.run(
            ["where.exe", "git"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_bash_candidate_from_git_path(git_path: str) -> Path:
    """Derive ``<gitRoot>/bin/bash.exe`` from the path to ``git.exe``."""
    normalized = ntpath.normpath(ntpath.join(ntpath.dirname(git_path), "..", "bin", "bash.exe"))
    return Path(normalized)


def _git_exec_path(git_path: str) -> str | None:
    """Run ``git --exec-path`` and return the first non-empty line."""
    try:
        result = subprocess.run(
            [git_path, "--exec-path"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        exec_path = line.strip()
        if exec_path:
            return exec_path
    return None


def _git_install_root_from_exec_path(exec_path: str) -> str | None:
    """Return the Git for Windows install root given a ``mingw*/libexec/git-core`` path."""
    current = ntpath.normpath(exec_path)
    while True:
        parent, name = ntpath.split(current)
        if name.casefold() in {"mingw32", "mingw64"}:
            return parent
        if parent == current:
            return None
        current = parent


def _git_bash_candidates_from_exec_path(exec_path: str) -> list[Path]:
    """Return candidate ``bash.exe`` paths derived from ``git --exec-path``."""
    normalized_exec_path = ntpath.normpath(exec_path)
    install_root = _git_install_root_from_exec_path(normalized_exec_path)
    if install_root is not None:
        return [Path(ntpath.join(install_root, "bin", "bash.exe"))]
    return [
        Path(ntpath.normpath(ntpath.join(normalized_exec_path, "..", "..", "bin", "bash.exe")))
    ]


def _is_git_bash_install(bash_path: str) -> bool:
    """Return True when *bash_path* is a bash of a Git for Windows install.

    Both layouts are accepted: the native launcher ``<root>/bin/bash.exe``
    and the real MSYS2 bash ``<root>/usr/bin/bash.exe``.  Git for Windows
    always ships a ``<root>/cmd/git.exe`` marker; MSYS2 (which also ships
    ``usr/bin/bash.exe``) has no such marker, so ``MSYSTEM`` neutralization
    stays limited to Git Bash and never affects real MSYS2 shells.
    """
    if not bash_path:
        return False
    text = ntpath.normpath(bash_path)
    drive, tail = ntpath.splitdrive(text)
    parts = [p.lower() for p in tail.split("\\") if p]
    # expect either ...\usr\bin\bash.exe or ...\bin\bash.exe
    if len(parts) < 3 or parts[-1] != "bash.exe" or parts[-2] != "bin":
        return False
    if parts[-3] == "usr":
        root = "\\".join(parts[:-3])
    else:
        root = "\\".join(parts[:-2])
    # Probe the marker with an *absolute* path: ``ntpath.join(drive, root, ...)``
    # would produce a drive-relative path ("C:foo") that Windows resolves
    # against the process's current directory on drive C:.  When that per-drive
    # CWD is not the drive root (e.g. after code chdirs into a temp dir on C:),
    # the marker lookup silently fails even for a real Git install and MSYSTEM
    # neutralization gets skipped.  Anchoring the drive makes the check
    # CWD-independent.
    root_path = (drive + "\\" if drive else "") + root
    return os.path.isfile(ntpath.join(root_path, "cmd", "git.exe"))


_MSYSTEM_NEUTRALIZE_PREFIX = "export MSYSTEM=; "


def _with_msystem_neutralized(cmd: str, bash_path: str | None) -> str:
    """Prepend an ``MSYSTEM``-neutralizing statement to *cmd* on Git Bash.

    Git Bash's ``bin/bash.exe`` launcher unconditionally injects
    ``MSYSTEM=MINGW64`` into the shell (setting ``MSYSTEM`` in the parent
    environment is useless), and the MSYS2 runtime re-injects the variable
    into children when it is *absent* (``unset`` does not stick).  Exporting
    an *empty* value at the start of the command makes xmake — a child
    process — see an empty ``MSYSTEM`` and default to the ``windows``/MSVC
    platform, while the launcher's PATH setup stays intact.  Limited to Git
    for Windows bash on Windows; all other platforms and shells run the
    command unchanged.
    """
    if sys.platform == "win32" and _is_git_bash_install(bash_path or ""):
        return _MSYSTEM_NEUTRALIZE_PREFIX + cmd
    return cmd


def _find_git_bash_windows() -> str | None:
    """Locate a *working* Git Bash on Windows.

    Every candidate must both exist on disk and pass the external-program
    smoke test (``_bash_runs``): a bash entry that cannot launch external
    programs is treated as "no bash", so PowerShell becomes the fallback
    shell.

    Resolution order:
      1. ``KIMIX_GIT_BASH_PATH`` environment variable.
      2. ``where.exe git`` -> ``<gitDir>/../bin/bash.exe``.
      3. ``git --exec-path`` -> Git for Windows install root -> ``bin/bash.exe``.
      4. Common install locations.
      5. ``bash`` on PATH (WindowsApps Store stubs are ignored — they are not
         a real bash, so a machine without git reports "no bash" and
         PowerShell becomes the fallback shell).
    """
    def _usable(candidate: Path) -> bool:
        return candidate.exists() and _bash_runs(str(candidate))

    override = os.environ.get("KIMIX_GIT_BASH_PATH")
    if override:
        candidate = Path(override)
        if _usable(candidate):
            return str(candidate.resolve())

    for git_path in _where_git_executables():
        bash_candidate = _git_bash_candidate_from_git_path(git_path)
        if _usable(bash_candidate):
            return str(bash_candidate.resolve())

        git_exec_path = _git_exec_path(git_path)
        if git_exec_path:
            for bash_candidate in _git_bash_candidates_from_exec_path(git_exec_path):
                if _usable(bash_candidate):
                    return str(bash_candidate.resolve())

    for candidate in (
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ):
        if _usable(candidate):
            return str(candidate.resolve())

    bash = shutil.which("bash")
    if bash and not _is_windows_apps_stub(bash) and _bash_runs(bash):
        return bash
    # A WindowsApps ``bash.exe`` is only a Microsoft Store stub (installs WSL),
    # not a usable bash: report "no bash" so PowerShell takes over as the
    # fallback shell when there is no git install.
    return None


def _git_bash_for_macos() -> str | None:
    """Return bash bundled with the official Git installer for macOS, if any."""
    git_path = shutil.which("git")
    if not git_path:
        return None
    git_exe = Path(git_path).resolve()
    if git_exe.parent.name.lower() == "bin":
        git_root = git_exe.parent.parent
    else:
        git_root = git_exe.parent
    for subpath in ("bin/bash", "usr/bin/bash"):
        candidate = git_root / subpath
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def _bash_candidates_macos() -> list[Path]:
    """Return well-known bash paths for macOS (Homebrew/MacPorts)."""
    return [
        Path("/opt/homebrew/bin/bash"),
        Path("/usr/local/bin/bash"),
        Path("/opt/local/bin/bash"),
    ]


def _bash_candidates_system() -> list[Path]:
    """Return standard system bash locations (Linux and macOS)."""
    return [Path("/bin/bash"), Path("/usr/bin/bash")]


@functools.lru_cache(maxsize=1)
def find_bash() -> str | None:
    """Find the system bash executable.

    Resolution order on Linux/macOS:
      1. Platform-specific well-known locations
         (Homebrew/MacPorts on macOS).
      2. Bash bundled with the official Git installer for macOS (macOS only).
      3. Standard system locations (``/bin/bash`` and ``/usr/bin/bash``).
      4. ``bash`` on PATH.
    """
    platform: str = sys.platform
    if platform == "win32":
        return _find_git_bash_windows()

    if platform == "darwin":
        # Prefer newer Homebrew/MacPorts bash over the aging system bash.
        for candidate in _bash_candidates_macos():
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

        # Git bash fallback (official Git installer for macOS).
        git_bash = _git_bash_for_macos()
        if git_bash:
            return git_bash

    for candidate in _bash_candidates_system():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    bash = shutil.which("bash")
    if bash:
        return bash
    return None


def _configured_shell() -> str | None:
    """Return the shell tool chosen by agent config, or ``None``.

    Reads the ``agent.shell`` key (``"bash"`` or ``"powershell"``) from the
    agent file used to build sessions (``kimix.base._default_agent_file``,
    i.e. ``src/kimix/agent_worker.json`` by default).  Returns ``None`` when
    the key is absent, holds an unknown value, or the file cannot be read —
    callers then fall back to the legacy platform-based heuristics.
    """
    try:
        from kimix.base import _default_agent_file
        data = orjson.loads(_default_agent_file.read_text(encoding="utf-8"))
        shell = data.get("agent", {}).get("shell")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(shell, str):
        return None
    shell = shell.strip().lower()
    if shell == "bash":
        return "bash"
    if shell in ("powershell", "pwsh"):
        return "powershell"
    return None


def _should_enable_bash() -> bool:
    """Return True when the Bash tool should be enabled on this platform.

    The ``agent.shell`` config key (e.g. ``"shell": "bash"`` in
    ``agent_worker.json``) takes precedence over the platform heuristics:
    ``"bash"`` enables Bash wherever it is installed; ``"powershell"``
    disables Bash on Windows (on non-Windows platforms the Powershell tool is
    unavailable, so Bash remains the fallback).

    With no explicit config on Windows, Git Bash is preferred: Bash is enabled
    whenever a real bash (typically shipped with Git for Windows) is found,
    and PowerShell is used only as the fallback when no bash exists (no git
    install).  Set ``USE_SYSTEM_PWSH_ON_WINDOWS`` to True to always prefer
    PowerShell on Windows instead.
    """
    if not USE_SYSTEM_SHELL:
        return False
    configured = _configured_shell()
    platform: str = sys.platform
    if configured == "powershell":
        if platform == "win32":
            return False
        return find_bash() is not None
    if configured == "bash":
        return find_bash() is not None
    # No explicit config: legacy platform-based behavior.
    if platform == "win32" and USE_SYSTEM_PWSH_ON_WINDOWS:
        return False
    return find_bash() is not None


def _should_enable_powershell() -> bool:
    """Return True when the Powershell tool should be enabled on this platform.

    The ``agent.shell`` config key (e.g. ``"shell": "bash"`` in
    ``agent_worker.json``) takes precedence over the platform heuristics:
    ``"bash"`` disables the tool unless Bash is unavailable (e.g. Windows
    without Git Bash), in which case PowerShell is the fallback shell;
    ``"powershell"`` forces the tool on Windows.

    With no explicit config on Windows, PowerShell is the fallback shell: it
    is enabled only when no bash (no git install) is available.  Set
    ``USE_SYSTEM_PWSH_ON_WINDOWS`` to True to always enable the tool on
    Windows regardless of Bash.
    """
    if sys.platform != "win32":
        return False
    configured = _configured_shell()
    if configured == "bash":
        # Bash is preferred but not installed here — fall back to PowerShell.
        return find_bash() is None
    if configured == "powershell":
        return True
    # No explicit config: legacy platform-based behavior.
    if USE_SYSTEM_PWSH_ON_WINDOWS:
        return True
    return find_bash() is None


# Characters for which a backslash escape must be preserved in bash.
# These are shell metacharacters and other special characters where
# converting \X to /X would change shell syntax or semantics.
_BASH_METACHARACTERS = frozenset("()|;&<>$\"`'\"*?[]{}~!#=% \t\n\r")

# In double quotes, \ only escapes these characters.  $ and ` are included
# because \$, \` inside "..." are literal (the $ / ` is escaped, not triggering
# variable expansion or command substitution).
_DQ_ESCAPED = frozenset(('"', '\\', '$', '`'))

# Precompiled regex for finding the next special character in unquoted mode.
# Matches backslash, single quote, double quote, dollar, or backtick.
_UNQUOTED_SPECIAL_RE = re.compile(r'[\\\'"$`]')


def _find_ansi_c_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ' of a ``$'...'`` region.

    ``start`` is the position right after the opening ``$'`` (i.e. the first
    character inside the region).  Returns ``-1`` if the region is
    unterminated.  Inside ``$'...'`` every ``\\X`` pair is treated as an
    escape (any character after \\ is skipped over).
    """
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
    """Return the index AFTER the closing `` ` `` of a backtick region.

    ``start`` is the position right after the opening `` ` ``.
    Returns ``-1`` if the region is unterminated.  ``\\` `` inside the
    region is an escaped backtick (literal `` ` ``).
    """
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length:
            i += 2  # skip escaped char (including \`)
        elif c == "`":
            return i + 1
        else:
            i += 1
    return -1


def _find_matching_paren(cmd: str, open_pos: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``cmd[open_pos]``.

    Returns ``-1`` if no matching ``)`` is found.  Tracks nested ``$(...)``,
    single-quoted regions, double-quoted regions (including their own
    nested ``$(...)`` and backticks), and backtick regions.
    """
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
            i = _find_dq_end(cmd, i + 1)
            if i == -1:
                return -1
        elif c == "`":
            i = _find_backtick_end(cmd, i + 1)
            if i == -1:
                return -1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "(":
            depth += 1
            i += 2
        elif c == "$" and i + 1 < length and cmd[i + 1] == "'":
            # $'...' ANSI-C quoted region — skip to its closing '
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                return -1
            i = end
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
            i += 1
        else:
            i += 1
    return -1


def _find_dq_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ``"`` of a double-quoted region.

    ``start`` is the position right after the opening ``"``.
    Returns ``-1`` if the region is unterminated.  Recognises ``\\X``
    escapes (``X`` in ``_DQ_ESCAPED``), nested ``$(...)``, ``$'...'``, and
    backtick command substitutions inside the region.
    """
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length and cmd[i + 1] in _DQ_ESCAPED:
            i += 2  # skip \X (X is escaped: ", \, $, `)
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
            # _find_ansi_c_end returns the index AFTER the closing '
            i = end
        elif c == "`":
            end = _find_backtick_end(cmd, i + 1)
            if end == -1:
                return -1
            # _find_backtick_end returns the index AFTER the closing `
            i = end
        else:
            i += 1
    return -1


def _process_unquoted(cmd: str) -> str:
    r"""Convert unquoted backslashes to forward slashes in ``cmd``.

    Walks the string in *unquoted mode* (the same rules that apply at the
    top level of a bash command): a bare ``\`` followed by a non-metachar
    is converted to ``/``, while ``\`` followed by a bash metacharacter,
    or ``\`` inside single / double / ANSI-C quotes, is preserved.

    The function also descends into ``$(...)`` and backtick command
    substitutions, processing their *content* in unquoted mode as well
    (because bash runs the content of ``$(...)`` and `` ` ` `` in a
    subshell where it is parsed unquoted — even when the substitution is
    itself nested inside ``"..."``).

    Native acceleration: kimix_native.parse._process_unquoted (byte-exact).
    """
    if _native_use_native("PARSE") and _NATIVE_PARSE is not None:
        return _NATIVE_PARSE._process_unquoted(cmd)
    result: list[str] = []
    i = 0
    length = len(cmd)

    while i < length:
        # ---- find the next special character ----
        # Use a single regex search (C-accelerated) to bulk-skip non-special chars.
        m = _UNQUOTED_SPECIAL_RE.search(cmd, i)
        if m:
            nxt = m.start()
            if nxt > i:
                result.append(cmd[i:nxt])
                i = nxt
        else:
            # No more special characters — append the remaining suffix and finish.
            result.append(cmd[i:])
            break

        if i >= length:
            break

        char = cmd[i]

        if char == "'":
            # Single-quoted region — copy literally until closing '
            end = cmd.find("'", i + 1)
            if end == -1:
                result.append(cmd[i:])
                break
            result.append(cmd[i : end + 1])
            i = end + 1

        elif char == '"':
            # Double-quoted region.  First find the end of the region,
            # then walk through it and convert the *content* of any
            # $(...) and `...` sub-regions using unquoted-mode rules
            # (bash runs command substitutions in a subshell where the
            # content is parsed unquoted, so backslashes inside must be
            # converted to '/' just like at the top level).
            dq_end = _find_dq_end(cmd, i + 1)
            if dq_end == -1:
                # Unterminated — copy the rest verbatim
                result.append(cmd[i:])
                break
            j = i + 1
            chunk_start = i
            while j < dq_end:
                # Bulk-skip to the next interesting character inside DQ:
                # backslash, dollar, or backtick.
                m2 = _UNQUOTED_SPECIAL_RE.search(cmd, j, dq_end)
                if m2:
                    nxt2 = m2.start()
                    if nxt2 > j:
                        j = nxt2
                else:
                    # No more special chars inside DQ — rest is verbatim
                    j = dq_end
                    break

                c = cmd[j]
                if c == "\\" and j + 1 < dq_end and cmd[j + 1] in _DQ_ESCAPED:
                    # \X inside DQ: X is escaped.  Skip the pair; it will
                    # be included in the next emitted chunk.
                    j += 2
                elif c == "$" and j + 1 < dq_end and cmd[j + 1] == "(":
                    # $(...) command substitution — process content
                    paren_end = _find_matching_paren(cmd, j + 1)
                    if paren_end == -1 or paren_end >= dq_end:
                        # Unterminated or mismatched — treat rest as verbatim
                        j = dq_end
                        break
                    result.append(cmd[chunk_start:j])
                    result.append("$(")
                    result.append(_process_unquoted(cmd[j + 2 : paren_end]))
                    result.append(")")
                    j = paren_end + 1
                    chunk_start = j
                elif c == "$" and j + 1 < dq_end and cmd[j + 1] == "'":
                    # $'...' ANSI-C region — skip through it (copied
                    # verbatim as part of the next chunk).
                    ac_end = _find_ansi_c_end(cmd, j + 2)
                    if ac_end == -1 or ac_end > dq_end:
                        # Unterminated or extends beyond DQ — treat rest as verbatim
                        j = dq_end
                        break
                    j = ac_end
                elif c == "`":
                    # Backtick command substitution — process content
                    bt_end = _find_backtick_end(cmd, j + 1)
                    if bt_end == -1 or bt_end > dq_end:
                        # Unterminated or extends beyond DQ — treat rest as verbatim
                        j = dq_end
                        break
                    result.append(cmd[chunk_start:j])
                    result.append("`")
                    result.append(_process_unquoted(cmd[j + 1 : bt_end - 1]))
                    result.append("`")
                    j = bt_end
                    chunk_start = j
                else:
                    # Should not reach here — char is not one we handle in DQ
                    j += 1
            # Emit the final chunk (up to and including the closing ")
            result.append(cmd[chunk_start:dq_end])
            i = dq_end

        elif char == "$" and i + 1 < length and cmd[i + 1] == "'":
            # $'...' ANSI-C quoted region at top level — copy literally
            ac_end = _find_ansi_c_end(cmd, i + 2)
            if ac_end == -1:
                result.append(cmd[i:])
                break
            result.append(cmd[i:ac_end])
            i = ac_end

        elif char == "`":
            # Backtick command substitution at top level — process content
            bt_end = _find_backtick_end(cmd, i + 1)
            if bt_end == -1:
                result.append(cmd[i:])
                break
            result.append("`")
            result.append(_process_unquoted(cmd[i + 1 : bt_end - 1]))
            result.append("`")
            i = bt_end

        elif char == "\\":
            if i + 1 < length and cmd[i + 1] in _BASH_METACHARACTERS:
                # Backslash is escaping a bash metacharacter — preserve both.
                # Append atomically so the metacharacter (e.g. ' " $) is not
                # re-processed as a quote-start or ANSI-C region on the next
                # iteration.
                result.append("\\")
                result.append(cmd[i + 1])
                i += 2
            else:
                # Unquoted backslash in a path-like context — convert to /
                result.append("/")
                i += 1

        else:
            # Defensive: nxt should always point to a special char we handle.
            result.append(char)
            i += 1

    return "".join(result)


def _prepare_bash_cmd(cmd: str) -> str:
    r"""Prepare a command string for safe use with bash -c.

    On Windows, bash consumes backslashes as escape sequences outside of
    quotes, mangling Windows paths like ``src\kimix\tools\...`` into
    ``srckimixtools...``.  This function converts unquoted backslashes to
    forward slashes so that paths work correctly while preserving backslash
    escapes inside quoted strings (single quotes, double quotes, and ``$'…'``)
    and before bash metacharacters (e.g. ``\(``, ``\)``, ``\|``).

    It also descends into ``$(...)`` and backtick command substitutions
    (including those nested inside double quotes), converting backslashes
    in their content, because bash runs the content of a command
    substitution in a subshell where it is parsed unquoted.

    On non-Windows platforms, returns the command unchanged to preserve
    existing behavior.
    """
    if sys.platform != "win32":
        return cmd
    return _process_unquoted(cmd)


class BashParams(BaseModel):
    """Parameters for the Bash tool — execute a bash command."""

    model_config = {"populate_by_name": True}

    cmd: str = Field(
        default="",
        alias="command",  # LLM can use "command" instead of "cmd"
        description="Bash command or input text for an existing session. " + accepts_alias_text("cmd", "command", word=False)
    )
    mode: Literal["execute", "send", "interactive"] = mode_field(
        execute_desc="Run `cmd` as a shell command.",
        send_desc="Execute `cmd` in background, return task_id immediately.",
        interactive_desc="Start a persistent Bash REPL, return task_id for further input.",
    )
    timeout: int = timeout_field()
    task_id: str | None = task_id_field("cmd")
    wait_for_pattern: str | None = wait_for_pattern_field()
    max_lines: int | None = max_lines_field()
    deduplicate_output: bool = deduplicate_output_field()
    cwd: str | None = cwd_field("command")

    @model_validator(mode="before")
    @classmethod
    def _normalize_mode(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert deprecated boolean flags and mode aliases to canonical names."""
        return normalize_mode_validator(data)

    _validate_cmd = shell_cmd_required_validator("cmd")


class Bash(CallableTool2[BashParams]):
    """Execute a bash command via the system bash, with background task support."""

    name: str = "Bash"
    description: str = (
        "Execute a bash command. Supports Unix-style / POSIX bash syntax. "
        "Prefer `Glob`/`Grep` tools over `find`/`ls`/`grep`/`rg` for file and content search. "
        + _interactive_scope_text(is_shell=True)
    )
    params: type[BashParams] = BashParams

    def __init__(self, session: Session):
        super().__init__()
        if not _should_enable_bash():
            raise SkipThisTool()
        self._session = session
        bash = find_bash()
        if bash is None:
            raise SkipThisTool()
        self._bash = bash

        # Windows-specific experience (verified by TestPrepareBashCmd and
        # TestBashBackslashPaths): unquoted backslash paths are auto-converted.
        if sys.platform == "win32":
            self.description += (
                " On Windows, unquoted backslash paths are auto-converted to forward slashes "
                "(`cat src\\a.py` → `cat src/a.py`); backslashes inside quotes are preserved."
            )

        # Pre-normalize forbidden commands once at init time for O(1) per-call lookup.
        raw_forbidden = self._session.custom_config.get("config_json", {}).get("forbidden_commands", [])
        self._forbidden_keywords: list[str] = []
        seen: set[str] = set()
        for cmd in raw_forbidden:
            if not isinstance(cmd, str) or not cmd:
                continue
            normalized = " ".join(cmd.split())
            if normalized not in seen:
                seen.add(normalized)
                self._forbidden_keywords.append(normalized)

        # Config gates for the hardline safety floor and secret redaction
        # (both default on; explicitly set False to disable).
        shell_cfg = self._session.custom_config.get("config_json", {}).get("shell", {})
        if isinstance(shell_cfg, dict):
            self._hardline_enabled = shell_cfg.get("hardline", True)
            self._redact_secrets = shell_cfg.get("redact_secrets", True)
        else:
            self._hardline_enabled = True
            self._redact_secrets = True

    def _hardline_blocked(self, command: str) -> ToolError | None:
        r"""Return a ToolError when *command* hits the unconditional hardline floor.

        Applies deobfuscation variants (quotes/backslash-escapes/case tricks)
        before matching, so ``r\m -rf /``, ``rm "" -rf /`` and ``Rm -Rf /``
        are all blocked.  Skipped entirely when the ``shell.hardline`` config
        gate is explicitly ``False``.
        """
        if not self._hardline_enabled or not command:
            return None
        blocked, desc = check_hardline_blocked(command)
        if not blocked:
            return None
        return ToolError(
            output="",
            message=(
                f"Blocked (hardline): {desc}. This command cannot be executed "
                "via the agent."
            ),
            brief="Blocked (hardline)",
        )

    async def __call__(self, params: BashParams) -> ToolReturnValue:
        """Execute the bash command via the system bash executable.

        Args:
            params: The parameters specifying the command and its arguments.

        Returns:
            ToolOk on success, ToolError on failure or timeout.
        """
        # Hardline safety floor: never spawn a process for destructive commands.
        blocked = self._hardline_blocked(params.cmd)
        if blocked is not None:
            return blocked

        workdir_err = validate_workdir(params.cwd)
        if workdir_err is not None:
            return ToolError(message=workdir_err, brief="Invalid workdir")

        forbidden = self._forbidden_error(params.cmd)
        if forbidden is not None:
            return forbidden

        # Early dispatch: continue an existing session
        if params.task_id is not None:
            return await self._continue_session(params)

        if params.mode == "send":
            return await self._execute_background(params)

        if params.mode != "interactive" and not params.cmd:
            return ToolError(
                output="Empty command.",
                message="No command specified.",
                brief="Empty command",
            )

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        # Refresh PATH/PATHEXT from registry so that tools installed
        # since the last command (e.g. via WinGet) are discoverable.
        if sys.platform == "win32":
            from kimix.utils.windows_env import refresh_env_from_registry
            refresh_env_from_registry()

        if params.mode == "interactive":
            rtk_rewritten = False
            bootstrap = bash_compatibility_prelude()
            if params.cmd:
                safe_cmd = self._prepare_command(params.cmd)
                if isinstance(safe_cmd, ToolError):
                    return safe_cmd
                rtk_cmd, rtk_rewritten = _maybe_rewrite_shell_command_with_rtk(
                    safe_cmd, params.deduplicate_output, exclude_read=True
                )
                startup_cmd = "\n".join(
                    part for part in (bootstrap, rtk_cmd) if part
                )
            else:
                startup_cmd = bootstrap
            if startup_cmd:
                forbidden = self._forbidden_error(
                    startup_cmd, display_command=params.cmd
                )
                if forbidden is not None:
                    return forbidden
                blocked = self._hardline_blocked(startup_cmd)
                if blocked is not None:
                    return blocked
                encoded = _encode_startup_script(
                    _with_msystem_neutralized(startup_cmd, self._bash)
                )
                bash_args = ["-c", encoded + "; exec bash -i"]
            else:
                bash_args = ["-i"]
            process_task = ProcessTask(self._bash, bash_args, params.cwd, _bash_subprocess_env(), append_newline=True)
            task_id = await process_task.start(self._session, "bash")
            if params.wait_for_pattern is not None and process_task.stream is not None:
                from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
                inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
                output, matched, elapsed = await process_task.stream.wait_for_output(
                    timeout=params.timeout, pattern=pattern,
                    inactivity_timeout=inactivity_timeout,
                )
                alive = await process_task.thread_is_alive()
                status = "running" if alive else "completed"
                return await self._format_session_result(
                    task_id, process_task.stream, params, output, status,
                    wait_matched=matched, elapsed_seconds=elapsed,
                    message=(
                        f"Interactive Bash started. task_id: `{task_id}`. "
                        "Send 'exit' to close the session."
                    ),
                    brief="Interactive Bash started",
                )
            return ToolOk(
                output="",
                message=(
                    f"Interactive Bash started. task_id: `{task_id}`. "
                    "Use task_id to send commands and TaskOutput to read results. "
                    "Send 'exit' to close the session."
                ),
                brief="Interactive Bash started",
            )

        # Build the command line to pass to bash -c
        # On Windows, escape backslashes so bash preserves them in paths.
        safe_cmd = self._prepare_command(params.cmd)
        if isinstance(safe_cmd, ToolError):
            return safe_cmd
        rtk_cmd, rtk_rewritten = _maybe_rewrite_shell_command_with_rtk(
            safe_cmd, params.deduplicate_output, exclude_read=True
        )
        forbidden = self._forbidden_error(rtk_cmd, display_command=params.cmd)
        if forbidden is not None:
            return forbidden
        blocked = self._hardline_blocked(rtk_cmd)
        if blocked is not None:
            return blocked
        process_task = ProcessTask(self._bash, ["-c", _with_msystem_neutralized(rtk_cmd, self._bash)], params.cwd, _bash_subprocess_env())
        task_id = await process_task.start(self._session, "bash")

        wait_matched: bool | None = None
        elapsed_seconds: float | None = None
        try:
            if params.wait_for_pattern is not None and process_task.stream is not None:
                from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
                inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
                output, wait_matched, elapsed_seconds = await process_task.stream.wait_for_output(
                    timeout=params.timeout, pattern=pattern,
                    inactivity_timeout=inactivity_timeout,
                )
                if await process_task.thread_is_alive():
                    return await self._format_session_result(
                        task_id, process_task.stream, params, output, "running",
                        wait_matched=wait_matched, elapsed_seconds=elapsed_seconds,
                        message="Matched pattern, still running",
                        brief="Pattern matched",
                    )
            else:
                await process_task.wait_with_monitor(params.timeout)
        except asyncio.CancelledError:
            # The tool call was cancelled (e.g. by a tool-level timeout or
            # shutdown). Stop the subprocess and return a tool error so the
            # conversation stream can continue.
            with contextlib.suppress(asyncio.CancelledError):
                await process_task.stop()
            from kimix.tools.background.utils import remove_task_id
            remove_task_id(self._session, task_id)
            output = await process_task.stream.pop_output() if process_task.stream else ""
            output = await _maybe_export_output_async(output)
            return ToolError(
                output=output,
                message="Cancelled",
                brief="Command cancelled",
            )

        if await process_task.thread_is_alive():
            output = await process_task.stream.pop_output() if process_task.stream else ""
            output = await _maybe_export_output_async(output)
            guidance = foreground_background_guidance(params.cmd)
            message = f"Running in background. task_id: `{task_id}`. use `TaskOutput`"
            if guidance:
                message += f" {guidance}"
            return ToolError(
                output=output,
                message=message,
                brief="Timeout",
            )

        from kimix.tools.background.utils import remove_task_id
        remove_task_id(self._session, task_id)

        output = await process_task.stream.pop_output() if process_task.stream else ""
        stream = process_task.stream
        success = await stream.success() if stream else False
        real_exit_code = stream.exit_code if stream else None

        # Exit-code semantics + failure hints run on the raw output (the
        # redacted text is what gets displayed/exported below).
        meaning = interpret_exit_code(params.cmd, real_exit_code)
        hint = annotate_failure(output, params.cmd, real_exit_code)

        # Unify success/error path: always pass the real exit code.
        processed, output_path, output_truncated, original_path = await self._process_output(
            params, output, rtk_rewritten=rtk_rewritten
        )
        block = _build_session_output_block(
            task_id=task_id,
            status="completed",
            output=processed,
            exit_code=real_exit_code,
            exit_code_meaning=meaning,
            failure_hint=hint,
            wait_matched=wait_matched,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            output_truncated=output_truncated,
            original_path=original_path,
        )
        if not success:
            msg = "failed" + (f" Hint: {hint}" if hint else "")
            return ToolError(output=block, message=msg, brief="Command execution failed")

        msg = "[rtk] success" if rtk_rewritten else "success"
        return ToolOk(
            output=block,
            message=msg,
            brief="Command executed successfully",
            display_block=ShellDisplayBlock(language="shell"),
        )

    def _forbidden_error(
        self, command: str, *, display_command: str | None = None
    ) -> ToolError | None:
        """Return an error when *command* matches a configured policy rule."""
        if not command or not self._forbidden_keywords:
            return None
        normalized = " ".join(command.split())
        for keyword in self._forbidden_keywords:
            if keyword in normalized:
                shown = command if display_command is None else display_command
                return ToolError(
                    output="",
                    message=f"`{shown}` is forbidden by config rule.",
                    brief="Forbidden command",
                )
        return None

    def _prepare_command(self, command: str) -> str | ToolError:
        """Normalize and add Windows fallbacks, enforcing policy on generated text."""
        from kimix.tools.file.bash import shell_common

        prepared = shell_common.prepare_bash_command(command)
        forbidden = self._forbidden_error(prepared, display_command=command)
        return forbidden if forbidden is not None else prepared

    def _continuation_may_be_incomplete(self, command: str) -> bool:
        """Return whether Bash may combine this fragment with later input.

        Forbidden-command rules are evaluated per API call.  Allowing an
        incomplete fragment while such rules are active would let later calls
        assemble a forbidden command that never appears in any individual
        policy check.  Bash's own no-execute parser handles balanced compound
        commands, arrays, substitutions, and here-documents more faithfully
        than a second hand-written shell parser.
        """
        stripped = command.rstrip(" \t\r\n")
        if not stripped:
            return False

        trailing_backslashes = len(stripped) - len(stripped.rstrip("\\"))
        if trailing_backslashes % 2:
            return True

        try:
            checked = subprocess.run(
                [self._bash, "--noprofile", "--norc", "-n", "-c", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # Fail closed only while command-policy rules are active.
            return True
        if checked.returncode == 0:
            return False

        error = checked.stderr.lower()
        incomplete_markers = (
            "unexpected end of file",
            "unexpected eof while looking for matching",
            "delimited by end-of-file",
        )
        return any(marker in error for marker in incomplete_markers)

    def _compile_pattern(self, wait_for_pattern: str | None) -> re.Pattern[str] | ToolError:
        if wait_for_pattern is None:
            return None
        try:
            return re.compile(wait_for_pattern)
        except re.error as e:
            return ToolError(
                output="",
                message=f"Invalid wait_for_pattern: {e}",
                brief="Invalid pattern",
            )

    async def _execute_background(self, params: BashParams) -> ToolReturnValue:
        """Execute a bash command in background and return immediately with task_id."""
        safe_cmd = self._prepare_command(params.cmd)
        if isinstance(safe_cmd, ToolError):
            return safe_cmd
        rtk_cmd, rtk_rewritten = _maybe_rewrite_shell_command_with_rtk(
            safe_cmd, params.deduplicate_output, exclude_read=True
        )
        forbidden = self._forbidden_error(rtk_cmd, display_command=params.cmd)
        if forbidden is not None:
            return forbidden
        blocked = self._hardline_blocked(rtk_cmd)
        if blocked is not None:
            return blocked
        process_task = ProcessTask(self._bash, ["-c", _with_msystem_neutralized(rtk_cmd, self._bash)], params.cwd, _bash_subprocess_env())
        task_id = await process_task.start(self._session, "bash")

        return ToolOk(
            output=f"Running in background. task_id: `{task_id}`. Use `TaskOutput` tool to retrieve output.",
            message=f"Command started in background. task_id: `{task_id}`",
            brief="Background task started",
        )

    async def _continue_session(self, params: BashParams) -> ToolReturnValue:
        """Send input to an existing Bash session and optionally wait for output."""
        from kimix.tools.background.utils import get_all_tasks

        tasks = get_all_tasks(self._session)
        task_id = params.task_id.strip() if params.task_id else ""
        stream = tasks.get(task_id)
        if stream is None:
            started = [tid for tid, s in tasks.items() if await s.is_started()]
            if not started:
                return ToolError(
                    output="",
                    message=f"Task '{params.task_id}' not found. No running tasks.",
                    brief="Task not found",
                )
            return ToolError(
                output="",
                message=(
                    f"Task '{params.task_id}' not found. "
                    f"Available tasks: [{', '.join(started)}]"
                ),
                brief=f"Task '{params.task_id}' not found",
            )

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        if self._forbidden_keywords and self._continuation_may_be_incomplete(params.cmd):
            return ToolError(
                output="",
                message=(
                    "Incomplete interactive Bash input is disabled while "
                    "forbidden-command rules are configured."
                ),
                brief="Unsafe command fragment",
            )

        # A persistent shell accepts arbitrary parser fragments: a heredoc
        # body, an unfinished quote, or the remainder of a compound command.
        # Rewriting such input as an independent program would corrupt parser
        # state and `$?`.  Compatibility functions were exported when the
        # interactive shell started, so continuation text is sent verbatim.
        rtk_cmd = params.cmd
        rtk_rewritten = False

        # Report only output produced after an accepted input command.  Retain
        # the drained buffer so a process that rejects input cannot lose it.
        prior_output = await stream.pop_output()

        input_text = rtk_cmd
        if not input_text.endswith("\n"):
            input_text += "\n"
        if not await stream.input(input_text):
            return ToolError(
                output=prior_output,
                message=f"Failed to send input to task '{task_id}'",
                brief="Send input failed",
            )

        from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
        inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
        output, matched, elapsed = await stream.wait_for_output(
            timeout=params.timeout, pattern=pattern,
            inactivity_timeout=inactivity_timeout,
        )
        alive = await stream.thread_is_alive()
        status = "running" if alive else "completed"
        return await self._format_session_result(
            task_id, stream, params, output, status,
            wait_matched=matched, elapsed_seconds=elapsed,
            message=(f"[rtk] Data sent to `{task_id}`. Status: {status}." if rtk_rewritten else f"Data sent to `{task_id}`. Status: {status}."),
            brief="Data sent and output retrieved",
            rtk_rewritten=rtk_rewritten,
        )

    async def _process_output(
        self, params: BashParams, output: str, rtk_rewritten: bool = False
    ) -> tuple[str, str | None, bool, str | None]:
        """Summarize/export long output. Returns (display_output, path, truncated, original_path)."""
        # Secret redaction runs first (config-gated) so the dedup/export/
        # summarize pipeline never sees credentials.
        if self._redact_secrets and output:
            output = redact_sensitive_output(output)
        # When rtk itself folded the output, preserve the full stream so the
        # model can page through the unfiltered results.  This is done before
        # the local token filter so the raw rtk stream is captured even when
        # dedup/max_lines are disabled.
        rtk_original_path: str | None = None
        if output and not (
            (params.deduplicate_output and not rtk_rewritten)
            or params.max_lines is not None
        ):
            rtk_original_path, _ = await _maybe_export_rtk_original_async(output)
        # Run token filter pipeline (dedup, truncate)
        output, original_path = await _token_filter_output(
            output,
            token_kill=params.deduplicate_output,
            max_lines=params.max_lines,
            rtk_rewritten=rtk_rewritten,
        )
        if original_path is None:
            original_path = rtk_original_path
        output_truncated = False
        if len(output) > 65536:
            output = await _summarize_long_output_async(self._session, params.cmd, output)
            output_truncated = True
        output = await _maybe_export_output_async(output)
        output_path = _extract_export_path(output)
        return output, output_path, output_truncated, original_path

    async def _format_session_result(
        self,
        task_id: str,
        stream: 'BackgroundStream' | None,
        params: BashParams,
        output: str,
        status: str,
        *,
        wait_matched: bool | None,
        elapsed_seconds: float | None,
        message: str,
        brief: str,
        rtk_rewritten: bool = False,
        exit_code_meaning: str | None = None,
        failure_hint: str | None = None,
    ) -> ToolReturnValue:
        """Build a ToolOk response with a structured output block."""
        processed, output_path, output_truncated, original_path = await self._process_output(
            params, output, rtk_rewritten=rtk_rewritten
        )
        if status != "completed":
            real_exit_code = None
        else:
            real_exit_code = stream.exit_code if stream else None
            if real_exit_code is None and stream is not None:
                real_exit_code = 0 if await stream.success() else None
        if exit_code_meaning is None and real_exit_code is not None:
            exit_code_meaning = interpret_exit_code(params.cmd, real_exit_code)
        block = _build_session_output_block(
            task_id=task_id,
            status=status,
            output=processed,
            exit_code=real_exit_code,
            exit_code_meaning=exit_code_meaning,
            failure_hint=failure_hint,
            wait_matched=wait_matched,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            output_truncated=output_truncated,
            original_path=original_path,
        )
        return ToolOk(output=block, message=message, brief=brief)
