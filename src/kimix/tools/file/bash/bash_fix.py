"""Windows Git Bash compatibility fixes for selected native POSIX commands.

Git for Windows ships a substantial POSIX userland, but a few command names
commonly emitted for Linux or macOS are absent even though an equivalent is
already available.  This module rewrites only verified, behaviorally compatible
command words.  It does not install software and deliberately leaves commands
without a faithful equivalent untouched.

The scanner is shell-aware: quoted text, comments, redirection operands,
heredoc bodies, assignments, case patterns, and ordinary arguments are data,
not commands.  Nested command substitutions and process substitutions are
scanned as their own command contexts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import regex as re

_REV_PERL = (
    "perl '-Mopen=:std,:encoding(UTF-8)' -e '"
    "my $zero = shift @ARGV; my $failed = 0; "
    "sub reverse_fh { my ($fh, $zero) = @_; "
    "local $/ = $zero ? qq(\\0) : qq(\\n); "
    "while (my $record = <$fh>) { "
    "my $ended = $zero ? $record =~ s/\\0\\z// : $record =~ s/\\r?\\n\\z//; "
    "print scalar reverse($record); "
    "print($zero ? qq(\\0) : qq(\\n)) if $ended } } "
    "if (@ARGV) { for my $file (@ARGV) { "
    "if (open my $fh, q(<:encoding(UTF-8)), $file) { reverse_fh($fh, $zero); close $fh } "
    "else { warn qq(rev: $file: $!\\n); $failed = 1 } } } "
    "else { reverse_fh(*STDIN, $zero) } exit $failed'"
)

_NATIVE_DELEGATE = (
    "local __kimix_native=''; __kimix_native=$(type -P {name}) || :; "
    "if [[ -n $__kimix_native ]]; then \"$__kimix_native\" \"$@\"; return; fi; "
)

_FALLBACK_BODIES = {
    "gtimeout": "timeout \"$@\"",
    "rev": (
        "local __kimix_zero=0; while (( $# )); do case $1 in "
        "-0|--zero) __kimix_zero=1; shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"rev: unsupported option: $1\" >&2; return 1;; "
        "*) break;; esac; done; "
        + _REV_PERL
        + " -- \"$__kimix_zero\" \"$@\""
    ),
    "xdg-open": "start \"$@\"",
    "open": "start \"$@\"",
    "pbcopy": "clip.exe \"$@\"",
    "pbpaste": (
        "powershell.exe -NoProfile -NonInteractive -Command "
        "'[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "[Console]::Out.Write((Get-Clipboard -Raw))' \"$@\""
    ),
}


def _fallback_definition(name: str) -> str:
    delegate = _NATIVE_DELEGATE.format(name=name)
    body = _FALLBACK_BODIES[name]
    return (
        f"if ! command -v {name} >/dev/null 2>&1; then "
        f"{name}() {{ {delegate}{body}; }}; fi"
    )


def _single_quote(command: str) -> str:
    """Quote *command* as one literal Bash word."""
    return "'" + command.replace("'", "'\"'\"'") + "'"


def _wrapper_runner(name: str) -> str:
    """Return an executable command for wrappers that cannot invoke functions."""
    script = _fallback_definition(name) + f"; {name} \"$@\""
    return "/usr/bin/bash -c " + _single_quote(script) + " --"


_FALLBACKS = {name: _fallback_definition(name) for name in _FALLBACK_BODIES}

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\+)?=")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COMMAND_START_KEYWORDS = frozenset(
    {"!", "{", "if", "then", "elif", "else", "while", "until", "do"}
)
_COMMAND_END_KEYWORDS = frozenset({"fi", "done", "esac"})
_LIST_KEYWORDS = frozenset({"for", "select", "case"})

_COMMAND_WRAPPERS = frozenset(
    {"command", "coproc", "env", "exec", "nohup", "sudo", "time"}
)
_WRAPPER_OPTIONS_WITH_VALUE = {
    "env": frozenset(
        {
            "-u",
            "--unset",
            "-C",
            "--chdir",
            "-S",
            "--split-string",
        }
    ),
    "exec": frozenset({"-a"}),
    "sudo": frozenset(
        {
            "-C",
            "--close-from",
            "-D",
            "--chdir",
            "-g",
            "--group",
            "-h",
            "--host",
            "-p",
            "--prompt",
            "-R",
            "--chroot",
            "-r",
            "--role",
            "-t",
            "--type",
            "-T",
            "--command-timeout",
            "-u",
            "--user",
        }
    ),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}

_OPERATOR_CHARS = frozenset(";&|()<>\n")
_REDIRECTION_START = frozenset("<>")


@dataclass(frozen=True)
class BashFix:
    """Result of :func:`fix_bash_command`.

    ``replacements`` records each original command name in source order.  An
    empty tuple means the command was returned byte-for-byte unchanged.
    """

    command: str
    replacements: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether any compatibility replacement was made."""
        return bool(self.replacements)

    @property
    def warning(self) -> str:
        """Return a concise description of compatibility changes."""
        if not self.replacements:
            return ""
        names = ", ".join(f"`{name}`" for name in self.replacements)
        return f"Added Windows Git Bash fallback(s) for native command(s): {names}."


@dataclass
class _Wrapper:
    kind: str
    skip_next: bool = False
    opaque: bool = False


@dataclass
class _HereDoc:
    delimiter: str | None
    strip_tabs: bool
    expands: bool


class _Scanner:
    """Conservative scanner for Bash executable command positions."""

    __slots__ = ("s", "n", "edits", "names")

    def __init__(self, command: str) -> None:
        self.s = command
        self.n = len(command)
        self.edits: list[tuple[int, int, str]] = []
        self.names: list[str] = []

    def fix(self) -> BashFix:
        try:
            self._scan_range(0, self.n)
        except RecursionError:
            # Malformed or adversarial nesting must never make the Bash tool
            # fail before Bash itself can report the syntax error.
            return BashFix(self.s)
        if not self.names:
            return BashFix(self.s)
        definitions = "\n".join(
            _FALLBACKS[name] for name in dict.fromkeys(self.names)
        )
        if self.edits:
            pieces: list[str] = []
            previous = 0
            for start, end, replacement in sorted(self.edits):
                pieces.extend((self.s[previous:start], replacement))
                previous = end
            pieces.append(self.s[previous:])
            source = "".join(pieces)
        else:
            source = self.s
        return BashFix(definitions + "\n" + source, tuple(self.names))

    @staticmethod
    def _literal_command_name(raw: str) -> str | None:
        """Return the command name produced solely by Bash quote removal.

        Bash permits literal command words such as ``'rev'``, ``\rev`` and
        ``r\"\"ev``.  Only words whose value can be determined without any
        expansion are accepted; parameter/command/arithmetic expansions,
        globbing, and malformed quotes remain untouched for Bash to handle.
        """
        value: list[str] = []
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\":
                if i + 1 >= len(raw):
                    return None
                if raw[i + 1] == "\n":
                    i += 2
                    continue
                value.append(raw[i + 1])
                i += 2
                continue
            if ch == "'":
                close = raw.find("'", i + 1)
                if close < 0:
                    return None
                value.append(raw[i + 1 : close])
                i = close + 1
                continue
            if ch == '"':
                i += 1
                while i < len(raw) and raw[i] != '"':
                    inner = raw[i]
                    if inner in "$`":
                        return None
                    if inner == "\\" and i + 1 < len(raw):
                        escaped = raw[i + 1]
                        if escaped in '$`"\\\n':
                            if escaped != "\n":
                                value.append(escaped)
                            i += 2
                            continue
                    value.append(inner)
                    i += 1
                if i >= len(raw):
                    return None
                i += 1
                continue
            if ch in "$`*?[{~":
                return None
            value.append(ch)
            i += 1
        name = "".join(value)
        return name if name in _FALLBACKS else None

    def _scan_range(self, start: int, end: int) -> None:
        s = self.s
        i = start
        command_expected = True
        redirect_expected = False
        redirect_resume = True
        wrapper: _Wrapper | None = None
        heredoc_operator: str | None = None
        pending_heredocs: list[_HereDoc] = []
        case_stack: list[str] = []
        function_name_expected = False
        function_body_expected = False

        while i < end:
            ch = s[i]

            if ch in " \t\r":
                i += 1
                continue
            if ch == "\\" and i + 1 < end and s[i + 1] == "\n":
                i += 2
                continue
            if ch == "\n":
                i += 1
                if pending_heredocs:
                    i = self._skip_heredoc_bodies(i, end, pending_heredocs)
                    pending_heredocs.clear()
                command_expected = True
                redirect_expected = False
                heredoc_operator = None
                wrapper = None
                continue
            if ch == "#" and self._comment_starts(i, start):
                newline = s.find("\n", i + 1, end)
                i = end if newline < 0 else newline
                continue

            process_substitution = s.startswith("<(", i) or s.startswith(">(", i)
            if not process_substitution and (
                ch in _REDIRECTION_START
                or s.startswith("&>", i)
                or (ch.isdigit() and self._redirection_after_fd(i, end))
            ):
                op_start = i
                if ch.isdigit():
                    while i < end and s[i].isdigit():
                        i += 1
                op, i = self._read_redirection(i, end)
                if op:
                    redirect_resume = command_expected
                    redirect_expected = True
                    if op in {"<<", "<<-"}:
                        # The delimiter is captured when the following word is
                        # read; its body starts only after this command line.
                        pass
                    else:
                        op_start = -1
                    if op_start >= 0:
                        heredoc_operator = op
                    continue
                i = op_start

            if redirect_expected:
                if s.startswith("<(" , i) or s.startswith(">(", i):
                    close = self._find_matching(i + 2, end, ")")
                    self._scan_range(i + 2, close if close < end else end)
                    word_end = close + 1 if close < end else end
                else:
                    scan_substitutions = heredoc_operator not in {"<<", "<<-"}
                    word_end = self._read_word(
                        i, end, scan_substitutions=scan_substitutions
                    )
                if word_end <= i:
                    i += 1
                    continue
                if heredoc_operator in {"<<", "<<-"}:
                    heredoc = self._heredoc_delimiter(s[i:word_end])
                    if heredoc is not None:
                        delimiter, expands = heredoc
                        pending_heredocs.append(
                            _HereDoc(delimiter, heredoc_operator == "<<-", expands)
                        )
                i = word_end
                command_expected = redirect_resume
                redirect_expected = False
                heredoc_operator = None
                continue

            if s.startswith("[[", i):
                function_body_expected = False
                i = self._skip_conditional(i + 2, end)
                command_expected = False
                continue
            if s.startswith("((", i):
                function_body_expected = False
                i = self._skip_arithmetic(i + 2, end)
                command_expected = False
                continue
            if s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                inner_end = close if close < end else end
                self._scan_range(i + 2, inner_end)
                i = close + 1 if close < end else end
                if command_expected:
                    command_expected = False
                continue
            if ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
                if command_expected:
                    command_expected = False
                continue
            if s.startswith("<(", i) or s.startswith(">(", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
                if command_expected:
                    command_expected = False
                continue

            op, op_end = self._read_control_operator(i, end)
            if op:
                i = op_end
                if op == "(" and function_body_expected:
                    function_body_expected = False
                    command_expected = True
                elif op == "(":
                    command_expected = True
                elif op == ")":
                    if case_stack and case_stack[-1] == "patterns":
                        case_stack[-1] = "body"
                        command_expected = True
                    else:
                        command_expected = False
                elif op in {";;", ";&", ";;&"}:
                    if case_stack:
                        case_stack[-1] = "patterns"
                        command_expected = False
                    else:
                        command_expected = True
                else:
                    command_expected = True
                redirect_expected = False
                heredoc_operator = None
                wrapper = None
                continue

            word_start = i
            scan_substitutions = heredoc_operator not in {"<<", "<<-"}
            word_end = self._read_word(i, end, scan_substitutions=scan_substitutions)
            if word_end <= i:
                i += 1
                continue
            raw = s[word_start:word_end]
            i = word_end

            if function_name_expected:
                function_name_expected = False
                function_body_expected = True
                command_expected = False
                declaration_end = self._empty_parentheses_end(i, end)
                if declaration_end is not None:
                    i = declaration_end
                continue

            if function_body_expected:
                function_body_expected = False
                if raw == "{":
                    command_expected = True
                    continue

            if case_stack and case_stack[-1] == "word":
                case_stack[-1] = "await-in"
                command_expected = False
                continue
            if case_stack and case_stack[-1] == "await-in" and raw == "in":
                case_stack[-1] = "patterns"
                command_expected = False
                continue
            if case_stack and case_stack[-1] == "patterns":
                if raw == "esac":
                    case_stack.pop()
                command_expected = False
                continue

            if not command_expected:
                if raw in {"then", "do", "else", "elif"}:
                    command_expected = True
                elif raw == "esac" and case_stack:
                    case_stack.pop()
                continue

            if raw == "function":
                function_name_expected = True
                command_expected = True
                continue
            declaration_end = self._function_declaration_end(raw, i, end)
            if declaration_end is not None:
                i = declaration_end
                function_body_expected = True
                command_expected = False
                continue
            if raw in _COMMAND_START_KEYWORDS:
                command_expected = True
                continue
            if raw in _COMMAND_END_KEYWORDS:
                if raw == "esac" and case_stack:
                    case_stack.pop()
                command_expected = False
                continue
            if raw in _LIST_KEYWORDS:
                if raw == "case":
                    case_stack.append("word")
                command_expected = False
                continue
            if _ASSIGNMENT_RE.match(raw):
                if i < end and s[i] == "(":
                    close = self._find_matching(i + 1, end, ")")
                    self._scan_expansions(i + 1, close if close < end else end)
                    i = close + 1 if close < end else end
                command_expected = True
                continue

            executable_wrapper = (
                wrapper is not None and wrapper.kind not in {"coproc", "time"}
            )
            if wrapper is not None and wrapper.kind == "coproc":
                if self._coproc_name_before_compound(raw, i, end):
                    wrapper = None
                    command_expected = True
                    continue
            if wrapper is not None:
                action = self._consume_wrapper_word(wrapper, raw)
                if action == "skip":
                    command_expected = True
                    continue
                if action == "inspect":
                    command_expected = False
                    wrapper = None
                    continue

            if raw in _COMMAND_WRAPPERS:
                wrapper = _Wrapper(raw)
                command_expected = True
                continue

            fallback_name = self._literal_command_name(raw)
            if fallback_name is not None:
                self.names.append(fallback_name)
                if executable_wrapper:
                    self.edits.append(
                        (word_start, word_end, _wrapper_runner(fallback_name))
                    )
            command_expected = False
            wrapper = None

    def _read_word(
        self, start: int, end: int, *, scan_substitutions: bool = True
    ) -> int:
        s = self.s
        i = start
        while i < end:
            ch = s[i]
            if ch in " \t\r\n" or ch in _OPERATOR_CHARS:
                break
            if ch == "#" and i == start:
                break
            if ch == "\\":
                i += 2 if i + 1 < end else 1
                continue
            if ch == "'":
                i = self._skip_single_quote(i + 1, end)
                continue
            if ch == '"':
                if scan_substitutions:
                    i = self._skip_double_quote(i + 1, end)
                else:
                    i = self._skip_double_quote_for_matching(i + 1, end)
                continue
            if ch == "`":
                close = self._find_backtick_end(i + 1, end)
                if scan_substitutions:
                    self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
                continue
            if s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                if scan_substitutions:
                    self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
                continue
            if s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
                continue
            if s.startswith("${", i):
                if scan_substitutions:
                    i = self._skip_parameter(i + 2, end)
                else:
                    i = self._skip_parameter_literal(i + 2, end)
                continue
            if s.startswith("$'", i):
                i = self._skip_ansi_quote(i + 2, end)
                continue
            i += 1
        return i

    def _skip_single_quote(self, i: int, end: int) -> int:
        close = self.s.find("'", i, end)
        return end if close < 0 else close + 1

    def _skip_ansi_quote(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "'":
                return i + 1
            else:
                i += 1
        return end

    def _skip_double_quote(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "\\" and i + 1 < end and s[i + 1] in '$`"\\\n':
                i += 2
            elif ch == '"':
                return i + 1
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1
        return end

    def _scan_expansions(self, i: int, end: int) -> None:
        """Scan executable substitutions in a region whose plain words are data."""
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s.startswith("$'", i):
                i = self._skip_ansi_quote(i + 2, end)
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote(i + 1, end)
            elif s[i] == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            elif s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1

    def _scan_heredoc_expansions(self, i: int, end: int) -> None:
        """Scan substitutions in an expanding heredoc body.

        Quote characters are literal in heredoc bodies; only a backslash can
        suppress the expansion introducers that Bash recognizes there.
        """
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            elif s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1

    def _skip_conditional(self, i: int, end: int) -> int:
        """Skip a ``[[ ... ]]`` expression while scanning its substitutions."""
        s = self.s
        while i < end:
            if s.startswith("]]", i):
                return i + 2
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s.startswith("$'", i):
                i = self._skip_ansi_quote(i + 2, end)
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote(i + 1, end)
            elif s[i] == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            else:
                i += 1
        return end

    def _skip_parameter_literal(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote_for_matching(i + 1, end)
            elif s[i] == "{":
                depth += 1
                i += 1
            elif s[i] == "}":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            else:
                i += 1
        return end

    def _skip_parameter(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote(i + 1, end)
            elif s[i] == "{":
                depth += 1
                i += 1
            elif s[i] == "}":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            else:
                i += 1
        return end

    def _skip_arithmetic(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            if s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif s.startswith("((", i):
                depth += 1
                i += 2
            elif s.startswith("))", i):
                depth -= 1
                i += 2
                if depth == 0:
                    return i
            elif s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote(i + 1, end)
            else:
                i += 1
        return end

    def _find_backtick_end(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "`":
                return i
            else:
                i += 1
        return end

    def _find_matching(self, i: int, end: int, closing: str) -> int:
        s = self.s
        depth = 0
        pending_heredocs: list[_HereDoc] = []
        case_stack: list[str] = []
        while i < end:
            ch = s[i]
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "\n":
                i += 1
                if pending_heredocs:
                    i = self._skip_heredoc_bodies(
                        i, end, pending_heredocs, scan_expansions=False
                    )
                    pending_heredocs.clear()
            elif s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            elif s.startswith("<<", i) and not s.startswith("<<<", i):
                strip_tabs = s.startswith("<<-", i)
                delimiter_start = i + (3 if strip_tabs else 2)
                while delimiter_start < end and s[delimiter_start] in " \t\r":
                    delimiter_start += 1
                delimiter_end = self._read_word(
                    delimiter_start, end, scan_substitutions=False
                )
                heredoc = self._heredoc_delimiter(s[delimiter_start:delimiter_end])
                if heredoc is not None:
                    delimiter, expands = heredoc
                    pending_heredocs.append(_HereDoc(delimiter, strip_tabs, expands))
                i = delimiter_end if delimiter_end > delimiter_start else delimiter_start
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote_for_matching(i + 1, end)
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                i = close + 1 if close < end else end
            elif ch == "#" and self._comment_starts(i, 0):
                newline = s.find("\n", i + 1, end)
                i = end if newline < 0 else newline
            elif s.startswith(";;&", i):
                if case_stack:
                    case_stack[-1] = "patterns"
                i += 3
            elif s.startswith(";;", i) or s.startswith(";&", i):
                if case_stack:
                    case_stack[-1] = "patterns"
                i += 2
            elif ch not in " \t\r;&|<>()":
                word_end = self._read_word(i, end, scan_substitutions=False)
                if word_end <= i:
                    i += 1
                    continue
                word = s[i:word_end]
                if word == "case":
                    case_stack.append("word")
                elif (
                    case_stack
                    and case_stack[-1] in {"patterns", "body"}
                    and word == "esac"
                ):
                    case_stack.pop()
                elif case_stack and case_stack[-1] == "word":
                    case_stack[-1] = "await-in"
                elif case_stack and case_stack[-1] == "await-in" and word == "in":
                    case_stack[-1] = "patterns"
                i = word_end
            elif ch == "(":
                depth += 1
                i += 1
            elif ch == closing:
                if case_stack and case_stack[-1] == "patterns":
                    case_stack[-1] = "body"
                    i += 1
                elif depth == 0:
                    return i
                else:
                    depth -= 1
                    i += 1
            else:
                i += 1
        return end

    def _skip_double_quote_for_matching(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            if s[i] == "\\" and i + 1 < end and s[i + 1] in '$`"\\\n':
                i += 2
            elif s[i] == '"':
                return i + 1
            elif s[i] == "`":
                close = self._find_backtick_end(i + 1, end)
                i = close + 1 if close < end else end
            elif s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                i = close + 1 if close < end else end
            else:
                i += 1
        return end

    def _read_control_operator(self, i: int, end: int) -> tuple[str, int]:
        s = self.s
        for op in (";;&", ";;", ";&", "&&", "||", "|&"):
            if s.startswith(op, i):
                return op, i + len(op)
        if s[i] in ";&|()":
            return s[i], i + 1
        return "", i

    def _read_redirection(self, i: int, end: int) -> tuple[str, int]:
        s = self.s
        for op in ("&>>", "&>", "<<<", "<<-", "<<", ">>", "<>", ">|", "<&", ">&", "<", ">"):
            if s.startswith(op, i):
                return op, i + len(op)
        return "", i

    def _redirection_after_fd(self, i: int, end: int) -> bool:
        s = self.s
        while i < end and s[i].isdigit():
            i += 1
        return i < end and s[i] in _REDIRECTION_START

    def _comment_starts(self, i: int, range_start: int) -> bool:
        if i <= range_start:
            return True
        return self.s[i - 1] in " \t\r\n;&|()<>"

    def _empty_parentheses_end(self, i: int, end: int) -> int | None:
        s = self.s
        while i < end and s[i] in " \t\r":
            i += 1
        if i >= end or s[i] != "(":
            return None
        i += 1
        while i < end and s[i] in " \t\r":
            i += 1
        return i + 1 if i < end and s[i] == ")" else None

    def _function_declaration_end(self, raw: str, i: int, end: int) -> int | None:
        if not _NAME_RE.fullmatch(raw):
            return None
        return self._empty_parentheses_end(i, end)

    def _consume_wrapper_word(self, wrapper: _Wrapper, raw: str) -> str:
        if wrapper.skip_next:
            wrapper.skip_next = False
            if wrapper.opaque:
                return "inspect"
            return "skip"
        if wrapper.opaque:
            return "inspect"
        if wrapper.kind == "command" and raw in {"-v", "-V"}:
            return "inspect"
        if wrapper.kind == "command" and (
            raw == "-p"
            or (
                raw.startswith("-")
                and not raw.startswith("--")
                and "p" in raw[1:]
            )
        ):
            wrapper.opaque = True
            return "skip"
        if wrapper.kind == "env" and raw in {"-S", "--split-string"}:
            wrapper.opaque = True
            wrapper.skip_next = True
            return "skip"
        if wrapper.kind == "env" and (
            raw.startswith("--split-string=")
            or (raw.startswith("-S") and raw != "-S")
        ):
            return "inspect"
        if raw == "--":
            return "skip"
        if raw in _WRAPPER_OPTIONS_WITH_VALUE.get(wrapper.kind, ()):
            wrapper.skip_next = True
            return "skip"
        if raw.startswith("-"):
            return "skip"
        if wrapper.kind == "env" and _ASSIGNMENT_RE.match(raw):
            return "skip"
        return "command"

    def _coproc_name_before_compound(self, raw: str, i: int, end: int) -> bool:
        if not _NAME_RE.fullmatch(raw):
            return False
        while i < end and self.s[i] in " \t\r":
            i += 1
        if i >= end:
            return False
        if self.s.startswith(("{", "(", "[[", "(("), i):
            return True
        for keyword in ("case", "for", "if", "select", "until", "while"):
            keyword_end = i + len(keyword)
            if self.s.startswith(keyword, i) and (
                keyword_end >= end
                or self.s[keyword_end] in " \t\r\n;&|()<>{}"
            ):
                return True
        return False

    def _heredoc_delimiter(self, raw: str) -> tuple[str | None, bool] | None:
        if not raw:
            return None
        result: list[str] = []
        quoted = False
        matchable = True
        i = 0
        while i < len(raw):
            ch = raw[i]
            if raw.startswith("$'", i):
                quoted = True
                value, i, valid = self._read_ansi_c_delimiter(raw, i + 2)
                result.append(value)
                matchable = matchable and valid
            elif ch == "'":
                quoted = True
                close = raw.find("'", i + 1)
                if close < 0:
                    result.append(raw[i + 1 :])
                    i = len(raw)
                else:
                    result.append(raw[i + 1 : close])
                    i = close + 1
            elif ch == '"':
                quoted = True
                i += 1
                while i < len(raw) and raw[i] != '"':
                    if raw[i] == "\\" and i + 1 < len(raw):
                        escaped = raw[i + 1]
                        if escaped in '$`"\\\n':
                            if escaped != "\n":
                                result.append(escaped)
                            i += 2
                            continue
                    result.append(raw[i])
                    i += 1
                if i < len(raw):
                    i += 1
            elif ch == "\\" and i + 1 < len(raw):
                quoted = True
                result.append(raw[i + 1])
                i += 2
            else:
                result.append(ch)
                i += 1
        delimiter = "".join(result) if matchable else None
        return delimiter, not quoted

    def _read_ansi_c_delimiter(
        self, raw: str, i: int
    ) -> tuple[str, int, bool]:
        result: list[str] = []
        valid = True
        simple = {
            "a": "\a",
            "b": "\b",
            "e": "\x1b",
            "E": "\x1b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
            "\\": "\\",
            "'": "'",
            '"': '"',
            "?": "?",
        }
        while i < len(raw):
            if raw[i] == "'":
                return "".join(result), i + 1, valid
            if raw[i] != "\\" or i + 1 >= len(raw):
                result.append(raw[i])
                i += 1
                continue
            escape = raw[i + 1]
            if escape in simple:
                result.append(simple[escape])
                i += 2
                continue
            if escape in "01234567":
                j = i + 1
                while j < len(raw) and j < i + 4 and raw[j] in "01234567":
                    j += 1
                result.append(chr(int(raw[i + 1 : j], 8)))
                i = j
                continue
            if escape in "xXuU":
                widths = {"x": 2, "X": 2, "u": 4, "U": 8}
                j = i + 2
                limit = min(len(raw), j + widths[escape])
                while j < limit and raw[j] in "0123456789abcdefABCDEF":
                    j += 1
                if j > i + 2:
                    value = int(raw[i + 2 : j], 16)
                    if value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF:
                        result.append(chr(value))
                    else:
                        # Bash accepts byte sequences outside Python's Unicode
                        # scalar range.  Python text cannot represent the same
                        # delimiter, so mark it unmatchable and conservatively
                        # keep all remaining source inside the heredoc.
                        valid = False
                        result.append(raw[i:j])
                    i = j
                    continue
            result.append("\\" + escape)
            i += 2
        return "".join(result), i, valid

    def _skip_heredoc_bodies(
        self,
        i: int,
        end: int,
        documents: list[_HereDoc],
        *,
        scan_expansions: bool = True,
    ) -> int:
        s = self.s
        for document in documents:
            body_start = i
            logical_line = ""
            logical_start = i
            while i < end:
                newline = s.find("\n", i, end)
                line_end = end if newline < 0 else newline
                line = s[i:line_end]
                compare = line.lstrip("\t") if document.strip_tabs else line
                if not logical_line:
                    logical_start = i
                if document.expands and self._heredoc_line_continues(compare):
                    logical_line += compare[:-1]
                    i = end if newline < 0 else newline + 1
                    continue
                logical_line += compare
                if (
                    document.delimiter is not None
                    and logical_line == document.delimiter
                ):
                    if scan_expansions and document.expands:
                        self._scan_heredoc_expansions(body_start, logical_start)
                    i = end if newline < 0 else newline + 1
                    break
                logical_line = ""
                i = end if newline < 0 else newline + 1
            else:
                if scan_expansions and document.expands:
                    self._scan_heredoc_expansions(body_start, end)
        return i

    @staticmethod
    def _heredoc_line_continues(line: str) -> bool:
        trailing = len(line) - len(line.rstrip("\\"))
        return trailing % 2 == 1


def bash_compatibility_prelude() -> str:
    """Return exported fallback definitions for a persistent Git Bash shell.

    Interactive input can be an incomplete Bash fragment (for example a
    heredoc body or the second half of a quote), so it must never be scanned
    and prefixed independently.  The interactive shell instead executes this
    prelude once and exports the fallback functions across ``exec bash -i``.
    """
    if sys.platform != "win32":
        return ""
    definitions = "\n".join(_FALLBACKS.values())
    exports = "\n".join(
        f"if declare -F {name} >/dev/null; then export -f {name}; fi"
        for name in _FALLBACKS
    )
    return definitions + "\n" + exports


def fix_bash_command(command: str) -> BashFix:
    """Rewrite selected native POSIX commands for Windows Git Bash.

    Non-Windows input is always returned byte-for-byte unchanged.  On Windows,
    only literal command words with verified equivalents are changed; unknown
    or semantically ambiguous commands are left for Bash to handle normally.
    """
    if sys.platform != "win32" or not command:
        return BashFix(command)
    # Quoting and escaping can form a literal command name without the source
    # containing it contiguously (for example ``r""ev`` or ``\rev``), so a
    # substring fast path would miss legal executable words.  The scanner is
    # linear and exits without allocating generated shell code when unchanged.
    return _Scanner(command).fix()
