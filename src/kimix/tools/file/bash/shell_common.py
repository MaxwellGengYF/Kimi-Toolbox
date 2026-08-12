"""Shared one-shot shell-execution helpers for the Bash/Powershell tool family.

The Bash tool and the Powershell tool both need to turn a raw command string
into an ``(argv, env)`` pair ready for ``asyncio.create_subprocess_exec``:

* bash: ``_prepare_bash_cmd`` (Windows backslash-path normalization) +
  ``fix_bash_command`` (native-command fallbacks) + ``_with_msystem_neutralized``
  (Git Bash MSYSTEM) run via ``bash [-l] -c`` with ``_bash_subprocess_env``;
* powershell: ``fix_pwsh_command`` (quote-aware validation/repair) +
  ``pwsh_transform`` (PS7 -> PS5.1 downgrade) + ``_PWSH_CONSOLE_INIT`` + the
  try/catch exit-code wrapper and the ``-NoP -NonI -Exec Bypass -NoL`` flags.

This module is the single home for that one-shot machinery.  ``bash_tool`` /
``pwsh_tool`` are imported lazily inside the functions so this module stays
import-safe at any point of the app boot and can be imported from anywhere
(no kimix import cycle); only the pure fixer modules are imported at top.
"""

from __future__ import annotations

import shutil
from typing import Any

from kimix.tools.file.bash import bash_fix
from kimix.tools.file.bash import process_pwsh
from kimix.tools.file.bash import pwsh_fix

# Flags shared by every one-shot PowerShell invocation (pwsh 7 and the
# Windows PowerShell 5.1 fallback alike).
PWSH_ONESHOT_FLAGS = ("-NoP", "-NonI", "-Exec", "Bypass", "-NoL")


def _bash_tool() -> Any:
    """Return the kimix Bash tool module (lazy: import-cycle-safe)."""
    from kimix.tools.file.bash import bash_tool

    return bash_tool


def _pwsh_tool() -> Any:
    """Return the kimix Powershell tool module (lazy: import-cycle-safe)."""
    from kimix.tools.file.bash import pwsh_tool

    return pwsh_tool


# ── Bash ─────────────────────────────────────────────────────────────────────

def prepare_bash_command(command: str) -> str:
    """Normalize a raw command for ``bash -c`` on Windows Git Bash.

    Converts unquoted Windows backslash paths to forward slashes
    (``_prepare_bash_cmd``) and adds native-command fallbacks
    (``fix_bash_command``) — the exact pipeline the Bash tool applies to a
    one-shot command before policy checks / RTK rewriting.
    """
    return bash_fix.fix_bash_command(_bash_tool()._prepare_bash_cmd(command)).command


def bash_argv(command: str, *, login: bool = True) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` executing the *already prepared* *command* via bash.

    ``login`` selects ``bash -l -c`` (login shell; used by the TodoList runner)
    vs ``bash -c`` (used by the Bash tool's one-shot path).  The command is
    MSYSTEM-neutralized on Git Bash and run with the shared bash subprocess
    environment.
    """
    tool = _bash_tool()
    bash_path = tool.find_bash() or "bash"
    neutralized = tool._with_msystem_neutralized(command, bash_path)
    args = (
        [bash_path, "-l", "-c", neutralized]
        if login
        else [bash_path, "-c", neutralized]
    )
    return args, tool._bash_subprocess_env()


def bash_file_argv(script_path: str) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` executing a ``.sh`` script file via bash."""
    tool = _bash_tool()
    bash_path = tool.find_bash() or "bash"
    return [bash_path, "-l", script_path], tool._bash_subprocess_env()


# ── PowerShell ───────────────────────────────────────────────────────────────

def wrap_pwsh_command(command: str) -> str:
    """Wrap *command* with the UTF-8 console init and try/catch exit-code wrapper.

    The wrapper sets UTF-8 output encoding, treats Ctrl+C as input, turns
    terminating errors into a non-zero exit code and preserves the real native
    exit code via ``exit $LASTEXITCODE``.
    """
    return (
        _pwsh_tool()._PWSH_CONSOLE_INIT
        + "try{" + command + "}catch{$_|Out-String|Write-Error;exit 1}"
        + ";exit $LASTEXITCODE"
    )


def pwsh_executable() -> str:
    """Return the PowerShell executable to use (pwsh 7 or the PS 5.1 fallback)."""
    pwsh = _pwsh_tool().find_pwsh()
    if pwsh is not None:
        return pwsh
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"


def pwsh_argv(command: str) -> tuple[list[str], str] | None:
    """Return ``(argv, not_found_hint)`` executing *command* via PowerShell.

    Repairs the command with ``fix_pwsh_command`` (``None`` when the command
    cannot be repaired), downgrades PS7 syntax to PS5.1 when no PowerShell 7
    executable is available, wraps it with :func:`wrap_pwsh_command` and builds
    the one-shot argv.
    """
    fix = pwsh_fix.fix_pwsh_command(command)
    if fix is None:
        return None
    inner = fix.command
    pwsh_tool_mod = _pwsh_tool()
    pwsh = pwsh_tool_mod.find_pwsh()
    if pwsh is not None:
        executable = pwsh
    else:
        # PowerShell 7 not available: downgrade PS7 syntax to PS5.1.
        inner, _warnings = process_pwsh.pwsh_transform(inner)
        executable = shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"
    raw = (
        pwsh_tool_mod._PWSH_CONSOLE_INIT
        + "try{" + inner + "}catch{$_|Out-String|Write-Error;exit 1}"
        + ";exit $LASTEXITCODE"
    )
    return (
        [executable, *PWSH_ONESHOT_FLAGS, "-Command", raw],
        f"PowerShell executable not found: {executable}",
    )


def pwsh_file_argv(script_path: str) -> tuple[list[str], str]:
    """Return ``(argv, not_found_hint)`` executing a ``.ps1`` script file."""
    executable = pwsh_executable()
    return (
        [executable, *PWSH_ONESHOT_FLAGS, "-File", script_path],
        f"PowerShell executable not found: {executable}",
    )
