"""PowerShell-aware command quoting validator and auto-repair.

``pwsh_tool._validate_command_for_ps`` uses a naive double-quote parity check
that rejects many *valid* PowerShell commands:

    Write-Output 'He said "hi"' # " inside a single-quoted string
    Write-Output "a`"b" # backtick-escaped " inside a dq string
    Write-Output "a""b" # doubled-quote escape inside a dq string
    $x = @"
    line " with " quotes
    "@ # double-quoted here-string
    # note " quote # " inside a comment
    cmd /c echo --% "hello world # " after the --% stop-parsing marker
    Write-Output "a$( "b" )c" # " inside a $(...) sub-expression

This module provides a single-pass tokenizer that follows PowerShell's real
quoting rules (verified empirically against pwsh 7.6.2):

* double-quoted strings: ``"..."`` with backtick escapes (`` `" ``) and
  doubled-quote escapes (``""``);
* single-quoted strings: ``'...'`` with ``''`` escaping a literal quote;
  backticks are literal inside single-quoted strings;
* here-strings: ``@" ... "@`` and ``@' ... '@`` — the opening delimiter must
  be the last thing on its line and the closing delimiter must be at the
  start of its own line (only whitespace may precede it);
* line comments: ``# ...`` — ``#`` starts a comment at a token boundary
  (start-of-input or after a non-word character); ``foo#c`` is a single
  argument token in PowerShell, not a comment;
* block comments: ``<# ... #>`` — PowerShell closes at the *first* ``#>``
  (block comments do not nest);
* the ``--%`` stop-parsing marker makes the rest of the line literal;
* ``$(...)`` sub-expressions may contain their own strings, comments and
  nested parentheses, and must be skipped so an inner ``"`` is not mistaken
  for the outer string's closing delimiter.

The tokenizer is used to decide:

1. **Balance** — if every quote is accounted for (inside a string, comment or
   here-string) the command is valid, even when the naive parity check says
   otherwise.
2. **Repair** — when a construct is left unclosed at the end of the input, the
   matching closing token is appended (``"``, ``'``, ``\n"@``, ``\n'@`` or
   ``#>``) so the command becomes a legal PowerShell command.
3. **Wrapper safety** — the tool wraps the command in ``try{...}catch{...}``.
   A command ending in a line comment or in the ``--%`` marker would swallow
   the wrapper, so a newline is appended.
4. **Failure** — ``fix_pwsh_command`` returns ``None`` when the command cannot
   be repaired (empty/whitespace-only input, or a dangling line-continuation
   backtick at the very end of the command).

The tokenizer implementation is the canonical pure-Python reference that lives
in ``bin/kimix_native/_shell_compat.py`` (the ``kimix_native`` shim); this
module keeps only the public API and the native acceleration fast path, so the
scanner logic exists in exactly one place.
"""

from __future__ import annotations

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
# scanner logic.
_shell = _native_get_compat("_shell_compat")
if _shell is None:  # pragma: no cover - shim missing (unbundled install)
    raise ImportError(
        "kimix_native shim unavailable: the pure-Python PowerShell scanner "
        "lives in bin/kimix_native/_shell_compat.py and must be importable. "
        "Install the kimix package with its bundled shim or run from the "
        "repository checkout."
    )

# Re-export the reference implementation's public surface (single source of
# truth in the shim).
PwshFix = _shell.PwshFix
_Scanner = _shell._PwshScanner
_W_UNCLOSED_DQ = _shell._W_UNCLOSED_DQ
_W_UNCLOSED_SQ = _shell._W_UNCLOSED_SQ
_W_UNCLOSED_HDQ = _shell._W_UNCLOSED_HDQ
_W_UNCLOSED_HSQ = _shell._W_UNCLOSED_HSQ
_W_UNCLOSED_BLOCK = _shell._W_UNCLOSED_BLOCK
_W_TRAILING_COMMENT = _shell._W_TRAILING_COMMENT
_W_STOP_PARSING = _shell._W_STOP_PARSING
_W_COMMENT_ONLY = _shell._W_COMMENT_ONLY
_W_TRAILING_CONTINUATION = _shell._W_TRAILING_CONTINUATION

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_PARSE = _native_get_module("parse")


def fix_pwsh_command(cmd: str) -> PwshFix | None:
    """Validate *cmd* with PowerShell quoting rules and repair it if possible.

    Returns a :class:`PwshFix` (``command`` may equal *cmd* when the command
    is already legal), or ``None`` when the command cannot be repaired.
    """
    if not cmd or not cmd.strip():
        return None
    # Native acceleration: kimix_native.parse.fix_pwsh_command.
    if _native_use_native("PARSE") and _NATIVE_PARSE is not None:
        result = _NATIVE_PARSE.fix_pwsh_command(cmd)
        if result is None:
            return None
        return PwshFix(command=result.command, warning=result.warning)
    return _shell.fix_pwsh_command(cmd)
