"""Windows Git Bash compatibility fixes for selected native POSIX commands.

Git for Windows ships a substantial POSIX userland, but a few command names
commonly emitted for Linux or macOS are absent even though an equivalent is
already available.  This module rewrites only verified, behaviorally compatible
command words.  It does not install software and deliberately leaves commands
without a faithful equivalent untouched.

Windows-style backslash paths (``D:\\repo\\src``, ``\\\\server\\share``,
``~\\Desktop``, ``.\\build``) — whether used as arguments, redirection targets,
or as the command word itself (``C:\\tools\\rg.exe``) — are rewritten to the
forward-slash spellings Git Bash understands, and the cmd.exe-only
``cd /d <path>`` form loses its flag
(``cd`` accepts a single argument in Bash).  A redundant leading shell
invocation — ``bash cd /c/dev/x && ...`` or ``bash -c 'cd C:\\x && rev'`` —
is unwrapped (and the ``-c`` inline script is scanned for fallbacks and
paths) because the Bash tool already runs the whole string via bash, so
``bash cd ...`` would otherwise try to open ``cd`` as a script file and fail.
Under an active command wrapper (``env``/``nohup``/``timeout``/...) the shell
word is an operand of that wrapper, so ``bash -c '<script>'`` keeps its shape
and only the inline script is fixed in place.

Command wrappers whose operand is itself a command are scanned as command
contexts so missing POSIX commands behind them get their Git Bash fallback:
``timeout`` (its one DURATION operand is consumed first), ``stdbuf``, ``nice``,
and ``xargs`` (bundled Git Bash executables that exec their operand), plus the
fallback wrappers ``gtimeout`` and ``watch`` (which also record their own
fallback definition; ``watch`` re-runs its command through ``eval "$*"`` in
the same shell, matching procps ``watch``'s ``sh -c`` behavior).  Fallback
definitions are exported (``export -f``) so nested shells — the standalone
runner scripts and ``bash -c`` operands — inherit them.

Rewrites are conservative: the
unquoted word must look unambiguously like a Windows path, so quoted data,
tool-level escape sequences, short ambiguous words such as ``a\\nb``, and
single-segment relative paths such as ``foo\\bar`` are preserved byte-for-byte.
Words whose normalized form needs it (spaces, ``&``, ``;``, ...) are emitted
inside double quotes; glob metacharacters stay unquoted so ``D:/x/*.txt`` still
performs pathname expansion.

The scanner is shell-aware: quoted text, comments, heredoc and here-string
bodies, assignments, case patterns, and ordinary arguments are data, not
commands.  Nested command substitutions and process substitutions are scanned
as their own command contexts.

The scanner implementation is the canonical pure-Python reference that lives
in ``bin/kimix_native/_shell_compat.py`` (the ``kimix_native`` shim); this
module keeps only the public API, the Windows-platform gate, and the native
acceleration fast path, so the scanner logic exists in exactly one place.
"""

from __future__ import annotations

import sys

from kimix.native_loader import (
    get_compat as _native_get_compat,
)
from kimix.native_loader import (
    get_module as _native_get_module,
)
from kimix.native_loader import (
    use_native as _native_use_native,
)

# The canonical pure-Python implementation (the historical body of this
# module) lives in the kimix_native shim so there is exactly one copy of the
# scanner logic.  The shim is importable whenever the kimix package is usable
# (the loader puts the shim directory on ``sys.path`` in every mode).
_shell = _native_get_compat("_shell_compat")
if _shell is None:  # pragma: no cover - shim missing (unbundled install)
    raise ImportError(
        "kimix_native shim unavailable: the pure-Python bash scanner lives in "
        "bin/kimix_native/_shell_compat.py and must be importable. Install the "
        "kimix package with its bundled shim or run from the repository checkout."
    )

# Re-export the reference implementation's public surface (single source of
# truth in the shim).
BashFix = _shell.BashFix
bash_compatibility_prelude = _shell.bash_compatibility_prelude
_fix_heredoc_trailing_operators = _shell._fix_heredoc_trailing_operators
_FALLBACKS = _shell._FALLBACKS
_FALLBACK_BODIES = _shell._FALLBACK_BODIES
_STUB_AWARE_FALLBACKS = _shell._STUB_AWARE_FALLBACKS
_fallback_definition = _shell._fallback_definition
_single_quote = _shell._single_quote
_wrapper_runner = _shell._wrapper_runner
_read_shell_control_operator = _shell._read_shell_control_operator
_apply_heredoc_operator_move = _shell._apply_heredoc_operator_move

# Scanner aliases: the shim renames the historical src names to avoid
# collisions inside the single vendored module; keep the src names working.
_Wrapper = _shell._BashWrapper
_HereDoc = _shell._BashHereDoc
_Scanner = _shell._BashFixScanner

# New-rule tables (also part of the scanner's public surface): wrappers whose
# command operand is scanned as a command context.  ``timeout``/``stdbuf``/
# ``nice``/``xargs`` are bundled Git Bash executables that exec their operand;
# ``gtimeout``/``watch`` are fallback names with the same operand shape.
_FALLBACK_COMMAND_WRAPPERS = _shell._FALLBACK_COMMAND_WRAPPERS
_WRAPPER_OPERAND_COUNTS = _shell._WRAPPER_OPERAND_COUNTS
_SAME_SHELL_WRAPPERS = _shell._SAME_SHELL_WRAPPERS

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_PARSE = _native_get_module("parse")


def fix_bash_command(command: str) -> BashFix:
    """Rewrite selected native POSIX commands for Windows Git Bash.

    Non-Windows input is always returned byte-for-byte unchanged.  On Windows,
    only literal command words with verified equivalents are changed; unknown
    or semantically ambiguous commands are left for Bash to handle normally.
    """
    if sys.platform != "win32" or not command:
        return BashFix(command)
    # Native acceleration: kimix_native.parse.fix_bash_command.
    if _native_use_native("PARSE") and _NATIVE_PARSE is not None:
        result = _NATIVE_PARSE.fix_bash_command(command)
        fixed = _fix_heredoc_trailing_operators(result.command)
        return BashFix(
            command=fixed,
            replacements=tuple(result.replacements),
            path_changes=tuple(result.path_changes),
            shell_wrappers=tuple(getattr(result, "shell_wrappers", ())),
        )
    # Quoting and escaping can form a literal command name without the source
    # containing it contiguously (for example ``r""ev`` or ``\rev``), so a
    # substring fast path would miss legal executable words.  The scanner is
    # linear and exits without allocating generated shell code when unchanged.
    # (``_shell.fix_bash_command`` applies the heredoc-operator fix internally.)
    result = _shell.fix_bash_command(command)
    return BashFix(
        command=result.command,
        replacements=tuple(result.replacements),
        path_changes=tuple(result.path_changes),
        shell_wrappers=tuple(getattr(result, "shell_wrappers", ())),
    )
