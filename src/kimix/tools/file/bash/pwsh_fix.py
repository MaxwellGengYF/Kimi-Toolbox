"""PowerShell-aware command quoting validator and auto-repair.

``pwsh_tool._validate_command_for_ps`` uses a naive double-quote parity check
that rejects many *valid* PowerShell commands:

    Write-Output 'He said "hi"'     # " inside a single-quoted string
    Write-Output "a`"b"             # backtick-escaped " inside a dq string
    Write-Output "a""b"             # doubled-quote escape inside a dq string
    $x = @"
    line " with " quotes
    "@                            # double-quoted here-string
    # note " quote                  # " inside a comment
    cmd /c echo --% "hello world    # " after the --% stop-parsing marker
    Write-Output "a$( "b" )c"       # " inside a $(...) sub-expression

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
"""

from __future__ import annotations

from dataclasses import dataclass

_NORMAL = "normal"
_DQ = "double-quoted"
_SQ = "single-quoted"
_HDQ = "here-double"
_HSQ = "here-single"
_COMMENT = "line-comment"
_BLOCK = "block-comment"

_W_UNCLOSED_DQ = (
    "The command has an unclosed double-quoted string; "
    'appended a closing `"` at the end to make it a legal PowerShell command.'
)
_W_UNCLOSED_SQ = (
    "The command has an unclosed single-quoted string; "
    "appended a closing `'` at the end to make it a legal PowerShell command."
)
_W_UNCLOSED_HDQ = (
    "The command has an unclosed double-quoted here-string; "
    'appended a newline and `"@` at the end to close it.'
)
_W_UNCLOSED_HSQ = (
    "The command has an unclosed single-quoted here-string; "
    "appended a newline and `'@` at the end to close it."
)
_W_UNCLOSED_BLOCK = (
    "The command has an unclosed block comment `<#`; "
    "appended `#>` at the end to close it."
)
_W_TRAILING_COMMENT = (
    "The command ends with a line comment; "
    "appended a newline so the trailing comment does not swallow the "
    "try/catch wrapper used to execute the command."
)
_W_STOP_PARSING = (
    "The command ends with the `--%` stop-parsing marker; "
    "appended a newline so the wrapper is not passed literally to the "
    "native command."
)
_W_COMMENT_ONLY = (
    "The command contains only comments; appended a newline and a no-op "
    "`$null` statement so the try/catch wrapper has a statement to execute."
)
_W_TRAILING_CONTINUATION = (
    "The command ends with a backtick line-continuation; "
    "appended a newline so the continuation does not join with the "
    "try/catch wrapper used to execute the command."
)


@dataclass(frozen=True)
class PwshFix:
    """Result of :func:`fix_pwsh_command`.

    ``command`` is the (possibly repaired) command to execute and ``warning``
    is a human-readable note describing any modification, or ``""`` when the
    command was already valid and left unchanged.
    """

    command: str
    warning: str = ""

    @property
    def changed(self) -> bool:
        """True when the command text differs from the input."""
        return bool(self.warning)


def fix_pwsh_command(cmd: str) -> PwshFix | None:
    """Validate *cmd* with PowerShell quoting rules and repair it if possible.

    Returns a :class:`PwshFix` (``command`` may equal *cmd* when the command
    is already legal), or ``None`` when the command cannot be repaired.
    """
    if not cmd or not cmd.strip():
        return None
    # Fast path: no quote/comment/continuation/here-string/stop-parsing
    # characters at all — the command is valid as-is and cannot affect the
    # try/catch wrapper.  Avoids the O(n) Python scan for plain commands.
    if (
        '"' not in cmd
        and "'" not in cmd
        and "#" not in cmd
        and "`" not in cmd
        and "@" not in cmd
        and "--%" not in cmd
    ):
        return PwshFix(cmd, "")
    return _Scanner(cmd).fix()


class _Scanner:
    """Single-pass tokenizer implementing PowerShell's quoting rules."""

    __slots__ = ("s", "n")

    def __init__(self, s: str) -> None:
        self.s = s
        self.n = len(s)

    # -- helpers used while skipping $(...) sub-expressions -----------------

    def _skip_sq(self, start: int) -> int:
        """Skip a single-quoted string starting at *start*; index after it."""
        s, n = self.s, self.n
        i = start + 1
        while i < n:
            if s[i] == "'":
                if i + 1 < n and s[i + 1] == "'":
                    i += 2  # '' -> literal single quote
                else:
                    return i + 1  # closing quote
            else:
                i += 1
        return i

    def _skip_dq(self, start: int) -> int:
        """Skip a double-quoted string starting at *start*; index after it."""
        s, n = self.s, self.n
        i = start + 1
        while i < n:
            ch = s[i]
            if ch == "`":
                i += 2 if i + 1 < n else 1
            elif ch == '"':
                if i + 1 < n and s[i + 1] == '"':
                    i += 2  # "" -> literal double quote
                else:
                    return i + 1  # closing quote
            elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                i = self._skip_subexpr(i)
            else:
                i += 1
        return i

    def _skip_block(self, start: int) -> int:
        """Skip a block comment starting at *start*; index after it.

        PowerShell closes block comments at the first ``#>`` — they do not
        nest (verified empirically with pwsh 7.6.2).
        """
        s, n = self.s, self.n
        i = start + 2
        while i < n:
            if s[i] == "#" and i + 1 < n and s[i + 1] == ">":
                return i + 2
            i += 1
        return i

    def _skip_subexpr(self, start: int) -> int:
        """Skip a ``$( ... )`` sub-expression starting at *start*.

        Iterative (no recursion) so deeply nested ``$(...)`` cannot hit the
        interpreter recursion limit.  Nested ``$(...)`` are handled by the
        paren-depth counter; strings and comments are skipped before the
        parens are counted, so a ``)`` inside a string is never mistaken for
        the closing paren.

        Returns the index *after* the matching ``)`` (or ``n`` when the
        sub-expression never closes — the caller then treats the enclosing
        string as unclosed).
        """
        s, n = self.s, self.n
        i = start + 2
        depth = 1
        while i < n and depth:
            ch = s[i]
            if ch == "(":
                depth += 1
                i += 1
            elif ch == ")":
                depth -= 1
                i += 1
            elif ch == "'":
                i = self._skip_sq(i)
            elif ch == '"':
                i = self._skip_dq(i)
            elif ch == "`":
                i += 2 if i + 1 < n else 1
            elif ch == "#":
                if self._at_token_start(i):
                    while i < n and s[i] != "\n":
                        i += 1
                else:
                    i += 1
            elif ch == "<" and i + 1 < n and s[i + 1] == "#":
                i = self._skip_block(i)
            else:
                i += 1
        return i

    # -- token-boundary predicate -------------------------------------------

    def _at_token_start(self, i: int) -> bool:
        """True when *i* starts a fresh token (not glued to a word/identifier)."""
        return i == 0 or not (self.s[i - 1].isalnum() or self.s[i - 1] == "_")

    # -- EOF repair ----------------------------------------------------------

    def _dq_closer(self) -> str:
        """Closing quote for an unclosed double-quoted string.

        A trailing backtick would escape a single appended ``"``, so an odd
        run of trailing backticks needs two quotes (one escaped, one closing).
        """
        k = 0
        for ch in reversed(self.s):
            if ch == "`":
                k += 1
            else:
                break
        return '""' if k % 2 == 1 else '"'

    # -- main scan -----------------------------------------------------------

    def fix(self) -> PwshFix | None:
        s, n = self.s, self.n
        mode = _NORMAL
        here_quote = ""   # opening quote of the current here-string
        line_start = 0    # start of the current line inside a here-string
        saw_code = False  # any real statement code seen (not comments/whitespace)
        last_cont_target = -1  # index after the last backtick-newline continuation
        i = 0
        while i < n:
            ch = s[i]
            if mode == _NORMAL:
                if ch == '"':
                    saw_code = True
                    mode = _DQ
                    i += 1
                elif ch == "'":
                    saw_code = True
                    mode = _SQ
                    i += 1
                elif ch == "`":
                    if i + 1 < n:
                        saw_code = True
                        if s[i + 1] == "\n":
                            last_cont_target = i + 2
                        i += 2  # escaped char (or line continuation)
                    else:
                        # Dangling line-continuation backtick: PowerShell
                        # rejects a backtick with nothing after it.
                        return None
                elif ch == "#" and self._at_token_start(i):
                    mode = _COMMENT
                    i += 1
                elif ch == "<" and i + 1 < n and s[i + 1] == "#":
                    mode = _BLOCK
                    i += 2
                elif (
                    ch == "@"
                    and i + 1 < n
                    and s[i + 1] in ("'", '"')
                    and self._at_token_start(i)
                ):
                    # Here-string opener: @' or @" followed by only
                    # whitespace until end-of-line (or end of input).
                    j = i + 2
                    while j < n and s[j] in " \t\r":
                        j += 1
                    if j == n or s[j] == "\n":
                        saw_code = True
                        here_quote = s[i + 1]
                        mode = _HDQ if here_quote == '"' else _HSQ
                        line_start = n if j == n else j + 1
                        i = line_start
                        continue
                    saw_code = True
                    i += 1  # @' / @" with content on the same line: not a here-string
                elif (
                    ch == "-"
                    and s.startswith("--%", i)
                    and self._at_token_start(i)
                ):
                    # --% stop-parsing: the rest of the line is literal.
                    if not saw_code:
                        # `--%` with no command before it is a PowerShell
                        # parse error ("Missing expression after unary
                        # operator '--'") — nothing to repair.
                        return None
                    nl = s.find("\n", i)
                    if nl == -1:
                        # The literal region reaches EOF and would swallow the
                        # try/catch wrapper — terminate the line explicitly.
                        return PwshFix(s + "\n", _W_STOP_PARSING)
                    i = nl + 1
                elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                    saw_code = True
                    i = self._skip_subexpr(i)
                elif ch.isspace():
                    i += 1
                else:
                    saw_code = True
                    i += 1
            elif mode == _DQ:
                if ch == "`":
                    i += 2 if i + 1 < n else 1
                elif ch == '"':
                    if i + 1 < n and s[i + 1] == '"':
                        i += 2  # "" -> literal double quote
                    else:
                        mode = _NORMAL
                        i += 1
                elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                    i = self._skip_subexpr(i)
                else:
                    i += 1
            elif mode == _SQ:
                if ch == "'":
                    if i + 1 < n and s[i + 1] == "'":
                        i += 2  # '' -> literal single quote
                    else:
                        mode = _NORMAL
                        i += 1
                else:
                    i += 1
            elif mode in (_HDQ, _HSQ):
                if ch == "\n":
                    line_start = i + 1
                    i += 1
                elif (
                    ch == here_quote
                    and i + 1 < n
                    and s[i + 1] == "@"
                    and s[line_start:i].strip() == ""
                ):
                    mode = _NORMAL
                    i += 2
                else:
                    i += 1
            elif mode == _COMMENT:
                if ch == "\n":
                    mode = _NORMAL
                    i += 1
                else:
                    i += 1
            elif mode == _BLOCK:
                if ch == "#" and i + 1 < n and s[i + 1] == ">":
                    mode = _NORMAL
                    i += 2
                else:
                    i += 1

        # -- end of input -----------------------------------------------------
        # A line-continuation backtick whose target line runs to the end of
        # the command would join with the try/catch wrapper added by the tool,
        # silently corrupting the command.  Append a newline so the
        # continuation ends on an empty line instead.
        needs_cont_nl = last_cont_target != -1 and s.rfind("\n") < last_cont_target
        if mode == _NORMAL:
            if saw_code:
                if needs_cont_nl:
                    return PwshFix(s + "\n", _W_TRAILING_CONTINUATION)
                return PwshFix(s, "")
            # Only comments/whitespace: give the wrapper a statement to run.
            return PwshFix(s + "\n$null", _W_COMMENT_ONLY)
        if mode == _DQ:
            fixed = s + self._dq_closer()
            warning = _W_UNCLOSED_DQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _SQ:
            fixed = s + "'"
            warning = _W_UNCLOSED_SQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _HDQ:
            fixed = s + '\n"@'
            warning = _W_UNCLOSED_HDQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _HSQ:
            fixed = s + "\n'@"
            warning = _W_UNCLOSED_HSQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _COMMENT:
            if saw_code:
                return PwshFix(s + "\n", _W_TRAILING_COMMENT)
            return PwshFix(s + "\n$null", _W_COMMENT_ONLY)
        if mode == _BLOCK:
            if saw_code:
                fixed = s + "#>"
                warning = _W_UNCLOSED_BLOCK
            else:
                fixed = s + "#>\n$null"
                warning = _W_COMMENT_ONLY
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        return None  # pragma: no cover - unreachable
