"""Comprehensive tests for the PowerShell quoting parser/fixer.

``kimix.tools.file.bash.pwsh_fix.fix_pwsh_command`` is the PowerShell-aware
validator/repairer that rescues commands rejected by the naive double-quote
parity check in ``pwsh_tool._validate_command_for_ps``.

The test matrix is organized as:

    A. valid commands that the naive validator REJECTS (odd ``"`` count) but
       PowerShell accepts — the parser must not return ``None``, and the
       command must run successfully under real pwsh;
    B. genuinely unbalanced commands — the parser must repair them by
       appending the missing closing token, and the repaired command must run
       where the original failed;
    C. unrepairable commands (empty, whitespace-only, dangling continuation
       backtick) — the parser must return ``None``;
    D. wrapper-safety cases (trailing line comment / ``--%`` marker) — the
       parser appends a newline so the tool's ``try{...}catch{...}`` wrapper
       is not swallowed;
    E. already-valid commands — the parser returns them unchanged;
    F. tool integration — ``Powershell.__call__``/``_execute_background`` run
       the fixed command with a warning instead of returning an error.

All classes that execute commands against a real PowerShell are skipped when
pwsh is not installed.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool

from kimix.tools.file.bash import Powershell
from kimix.tools.file.bash.pwsh_fix import PwshFix, fix_pwsh_command
from kimix.tools.file.bash.pwsh_tool import PowershellParams, _PWSH_CONSOLE_INIT
from kimix.tools.background.utils import _pop_task_data


PWSH = shutil.which("pwsh")
NEEDS_PWSH = pytest.mark.skipif(PWSH is None, reason="pwsh is not installed")


def _run_pwsh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run *cmd* via real pwsh; return (exit_code, stdout+stderr)."""
    import subprocess

    r = subprocess.run(
        [PWSH, "-NoP", "-NonI", "-C", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _is_parse_error(out: str) -> bool:
    """True when *out* looks like a PowerShell parse error (not a runtime one)."""
    low = out.lower()
    return (
        "parsererror" in low
        or "missing the terminator" in low
        or "unexpected token" in low
        or "is not valid in this context" in low
    )


def _force_pwsh_enabled() -> Any:
    """Bypass the platform gate so mocked tool tests run on any host."""
    return patch(
        "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
        return_value=True,
    )


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    session.custom_config.get.return_value = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


# ============================================================================
# C. Irreparable input -> None
# ============================================================================

class TestFixPwshCommandIrreparable:
    @pytest.mark.parametrize(
        "cmd",
        [
            "",
            "   ",
            "\t\n ",
            "`",                       # dangling line-continuation backtick
            "Write-Output `",
            "Get-ChildItem `",
            "--%",                     # --% with no command before it
            "--% foo",
        ],
    )
    def test_returns_none(self, cmd: str) -> None:
        assert fix_pwsh_command(cmd) is None


# ============================================================================
# A. Valid commands that the naive validator rejects (odd double-quote count)
# ============================================================================

# Commands with an ODD number of `"` characters: rejected by the naive
# validator (``_validate_command_for_ps`` returns an error) yet accepted by
# PowerShell.  The parser must accept them and the fixed command must run
# cleanly under real pwsh.
NAIVE_REJECTED_RUN_CLEAN: list[tuple[str, str]] = [
    # double quote inside a single-quoted string
    ("Write-Output 'a\"b'", "a\"b"),
    # backtick-escaped quote inside a double-quoted string
    ("Write-Output \"a`\"b\"", "a\"b"),
    # lone backtick-escaped quote at the top level
    ("Write-Output `\"", "\""),
    # single-quoted strings with '' escapes and a double quote inside
    ("Write-Output 'can''t \" do'", "can't \" do"),
    # double-quoted here-string containing a lone double quote
    ('$x = @"\nline " quote\n"@\nWrite-Output $x', 'line " quote'),
    # single-quoted here-string containing a double quote
    ("$x = @'\nline \" quote\n'@\nWrite-Output $x", 'line " quote'),
    # quotes inside a line comment
    ('# comment " quote\nWrite-Output ok', "ok"),
    ('Write-Output ok # "', "ok"),                     # trailing comment + quote
    ('Write-Output "" # "', ""),                       # empty string + comment quote
    ("Write-Output '' # empty \" comment", ""),
    ('# " and \' quotes\nWrite-Output ok', "ok"),
    # quote inside a block comment
    ('<# comment " quote #>\nWrite-Output ok', "ok"),
    ('<# " only #>\nWrite-Output ok', "ok"),
    # quote after the --% stop-parsing marker (rest of line is literal)
    ('cmd /c echo --% "hello world', '"hello world'),
    # backtick line continuation inside a double-quoted string + escaped quote
    ('Write-Output "a`\nb `"c"', 'a\nb "c'),
    # string concatenation with a double quote inside single quotes
    ('Write-Output ("a" + \'b"\')', 'ab"'),
    # hashtable value containing a double quote (single-quoted)
    ('[pscustomobject]@{name = \'has " quote\'} | Out-String', "has \" quote"),
    # double-quoted string containing a backtick-escaped quote only
    ('Write-Output "`""', '"'),
    # regex / comparison with a quote character
    ('foreach ($f in @("a")) { $f -match \'"\' }', "False"),
    # -replace with quote patterns
    ('$s = "a" -replace \'"\', \'""\'; Write-Output $s', "a"),
    # semicolon-joined statements where a comment carries a quote
    ('Write-Output "a"; # comment " b\nWrite-Output c', "a\nc"),
]

# Commands with tricky quoting but an EVEN number of `"` characters: the
# naive validator accepts them, but the parser must still handle them without
# breaking the quoting.
TRICKY_RUN_CLEAN: list[tuple[str, str]] = [
    ("Write-Output 'He said \"hi\"'", "He said \"hi\""),
    ("Write-Output \"a\"\"b\"", "a\"b"),                # doubled-quote escape
    ("Write-Output `\"hi`\"", "\"hi\""),                # top-level backtick escapes
    ("$x = 'it''s \"fine\"'; Write-Output $x", "it's \"fine\""),
    ("Write-Output 'don''t \"panic\"'", "don't \"panic\""),
    ('Write-Output "a$( "b" )c"', "abc"),               # $(...) sub-expression
    ('Write-Output "x$( "y" )z"', "xyz"),
    ("'{ \"key\": \"value\" }' | Out-String", "{ \"key\": \"value\" }"),
    ('Write-Output "`"hello`""', '"hello"'),
    ("Get-ChildItem -LiteralPath 'C:\\dir with \"quote\"' -ErrorAction SilentlyContinue; Write-Output done", "done"),
]

# These parse fine after the fix but fail at runtime for domain reasons
# (missing file / missing executable / git semantics).  Only the parse result
# is asserted.
TRICKY_PARSE_ONLY: list[str] = [
    'Get-Content \'C:\\path with "quotes".txt\'',
    '& \'C:\\Program Files\\Some "Dir"\\tool.exe\'',
    "git commit -m 'fix \"quotes\"'",
]


@NEEDS_PWSH
class TestFixPwshCommandValidNaiveRejected:
    @pytest.mark.parametrize("cmd,expected", NAIVE_REJECTED_RUN_CLEAN)
    def test_naive_rejected_fixed_and_runs_clean(self, cmd: str, expected: str) -> None:
        # Every one of these commands is REJECTED by the naive validator ...
        assert Powershell._validate_command_for_ps(cmd) is not None
        # ... but the PowerShell-aware parser must accept it (never None).
        fix = fix_pwsh_command(cmd)
        assert fix is not None, f"parser rejected valid command: {cmd!r}"
        rc, out = _run_pwsh(fix.command)
        assert rc == 0, f"fixed command failed: {fix.command!r} -> rc={rc} out={out!r}"
        assert expected in out, f"unexpected output for {fix.command!r}: {out!r}"

    @pytest.mark.parametrize("cmd,expected", TRICKY_RUN_CLEAN)
    def test_tricky_quoting_still_runs_clean(self, cmd: str, expected: str) -> None:
        fix = fix_pwsh_command(cmd)
        assert fix is not None, f"parser rejected valid command: {cmd!r}"
        rc, out = _run_pwsh(fix.command)
        assert rc == 0, f"fixed command failed: {fix.command!r} -> rc={rc} out={out!r}"
        assert expected in out, f"unexpected output for {fix.command!r}: {out!r}"

    @pytest.mark.parametrize("cmd", TRICKY_PARSE_ONLY)
    def test_tricky_parses_without_parse_error(self, cmd: str) -> None:
        fix = fix_pwsh_command(cmd)
        assert fix is not None, f"parser rejected valid command: {cmd!r}"
        rc, out = _run_pwsh(fix.command)
        assert not _is_parse_error(out), (
            f"command should parse fine; got a PowerShell parse error: {out!r}"
        )
        assert rc != 0  # fails for a *runtime* reason, not parsing


# ============================================================================
# B. Genuinely unbalanced commands -> repaired by the parser
# ============================================================================

# (original, expected_fixed, expected_output)
REPAIR_CASES: list[tuple[str, str, str]] = [
    # unclosed double-quoted string
    ('Write-Output "hello', 'Write-Output "hello"', "hello"),
    ('Write-Output "a" "b', 'Write-Output "a" "b"', "a\nb"),
    ('Write-Output "" "', 'Write-Output "" ""', ""),
    # unclosed single-quoted string (naive validator does not even see it)
    ("Write-Output 'hello", "Write-Output 'hello'", "hello"),
    ("'it''s", "'it''s'", "it's"),
    # double-quoted string containing an unclosed single quote
    ('"a\'b', '"a\'b"', "a'b"),
    # unclosed double-quoted string ending in a backtick: the backtick would
    # escape a single appended quote, so two quotes are appended
    ('Write-Output "a`', 'Write-Output "a`""', 'a"'),
    # even run of trailing backticks: a single quote is enough
    ('"a``', '"a``"', "a`"),
    # unclosed here-strings
    ('@"\nunclosed here-string', '@"\nunclosed here-string\n"@', "unclosed here-string"),
    ("@'\nunclosed here-string", "@'\nunclosed here-string\n'@", "unclosed here-string"),
    # unclosed block comment
    ("Write-Output ok <# unclosed comment", "Write-Output ok <# unclosed comment#>", "ok"),
]


@NEEDS_PWSH
class TestFixPwshCommandRepairsUnbalanced:
    @pytest.mark.parametrize("cmd,fixed,expected", REPAIR_CASES)
    def test_repairs_and_runs(self, cmd: str, fixed: str, expected: str) -> None:
        rc0, _ = _run_pwsh(cmd)
        assert rc0 != 0, f"original command should fail: {cmd!r}"
        fix = fix_pwsh_command(cmd)
        assert fix is not None, f"parser could not repair: {cmd!r}"
        assert fix.command == fixed, (
            f"unexpected repair: expected {fixed!r}, got {fix.command!r}"
        )
        assert fix.warning, "repair should carry a warning"
        assert fix.changed
        rc1, out = _run_pwsh(fix.command)
        assert rc1 == 0, f"repaired command failed: {fix.command!r} -> {out!r}"
        assert expected in out, f"unexpected output: {out!r}"

    def test_repair_warning_messages_are_meaningful(self) -> None:
        assert "double-quoted" in fix_pwsh_command('Write-Output "x').warning
        assert "single-quoted" in fix_pwsh_command("Write-Output 'x").warning
        assert "here-string" in fix_pwsh_command('@"\nx').warning
        assert "block comment" in fix_pwsh_command("x <# c").warning


# ============================================================================
# D. Wrapper-safety: trailing line comment / --% marker need a trailing newline
# ============================================================================

@NEEDS_PWSH
class TestFixPwshCommandWrapperSafety:
    @pytest.mark.parametrize(
        "cmd,expected_out,expected_suffix",
        [
            ("Write-Output ok # done", "ok", "\n"),
            ("cmd /c echo --% foo", "foo", "\n"),
            ('Write-Output ok # " done', "ok", "\n"),
            # comment-only commands need a real statement for the wrapper
            ("# just a comment", "", "\n$null"),
            ("<# just a comment", "", "#>\n$null"),
        ],
    )
    def test_trailing_comment_or_stop_parsing_appends_newline(
        self, cmd: str, expected_out: str, expected_suffix: str
    ) -> None:
        fix = fix_pwsh_command(cmd)
        assert fix is not None
        assert fix.changed, f"expected a modification for {cmd!r}"
        assert fix.command == cmd + expected_suffix
        assert "newline" in fix.warning
        # Simulate the tool's wrapper: try{<cmd>}catch{...}
        wrapped = _PWSH_CONSOLE_INIT + "try{" + fix.command + "}catch{$_|Out-String|Write-Error;exit 1}"
        rc, out = _run_pwsh(wrapped)
        assert rc == 0, f"wrapped command failed: {wrapped!r} -> {out!r}"
        assert expected_out in out

    @pytest.mark.parametrize(
        "cmd,expected_out,expected_suffix",
        [
            # continuation into nothing
            ("Write-Output `\n", "", "\n"),
            # continuation target is the last line (silent corruption before fix)
            ("Get-ChildItem `\n-Filter *.ps1", "", "\n"),
            # continuation + unclosed double-quoted string (repair + newline)
            ('Write-Output `\n"hello', "hello", '"\n'),
        ],
    )
    def test_trailing_continuation_appends_newline(
        self, cmd: str, expected_out: str, expected_suffix: str
    ) -> None:
        # Regression: without a trailing newline the tool's try/catch wrapper
        # joins the continued line and silently corrupts the command.
        fix = fix_pwsh_command(cmd)
        assert fix is not None
        assert fix.changed, f"expected a modification for {cmd!r}"
        assert fix.command == cmd + expected_suffix
        assert "continuation" in fix.warning
        wrapped = _PWSH_CONSOLE_INIT + "try{" + fix.command + "}catch{$_|Out-String|Write-Error;exit 1}"
        rc, out = _run_pwsh(wrapped)
        # The wrapper code must never appear inside the output (in the
        # unfixed version it was joined into the continued line).
        assert "$_|Out-String|Write-Error" not in out
        assert expected_out in out


# ============================================================================
# E. Already-valid commands -> unchanged
# ============================================================================

class TestFixPwshCommandUnchanged:
    @pytest.mark.parametrize(
        "cmd",
        [
            "Get-Location",
            "1 + 2",
            'Write-Output "hi"',
            "Write-Output 'a\"b'",        # valid, even though naive check rejects it
            "git status; cargo test",
            "Write-Output \"a\"\"b\"",
            'Write-Output "a$( "b" )c"',
            "$x = @\"\nhi\n\"@",
            "Write-Output \"a`\"b\"",
            "$x = 5 # trailing comment\nWrite-Output $x",
            "Write-Output (Get-Date -Format 'yyyy')",
            'Write-Output "a" -eq "a"',
            "if ($true) { Write-Output 'ok' }",
        ],
    )
    def test_unchanged_no_warning(self, cmd: str) -> None:
        fix = fix_pwsh_command(cmd)
        assert fix is not None
        assert fix.command == cmd
        assert fix.warning == ""
        assert not fix.changed


# ============================================================================
# Extra parser corner cases (unit level, no pwsh needed)
# ============================================================================

class TestFixPwshCommandParserCorners:
    def test_here_string_opener_not_recognized_when_glued_to_word(self) -> None:
        # `a@"` is not a here-string opener in PowerShell; the trailing `"@`
        # closes the string opened by the first `"`.
        fix = fix_pwsh_command('$x = a@"\nhello\n"@')
        assert fix is not None
        assert fix.command == '$x = a@"\nhello\n"@'

    def test_hash_glued_to_word_is_not_a_comment(self) -> None:
        # `foo#c` is a single argument token, not a comment.
        fix = fix_pwsh_command('Write-Output foo#c "x"')
        assert fix is not None
        assert fix.command == 'Write-Output foo#c "x"'

    def test_hash_after_punctuation_is_a_comment(self) -> None:
        fix = fix_pwsh_command('Write-Output (5)#c "x"\nWrite-Output ok')
        assert fix is not None
        # the quote after the comment is inside the comment -> no repair
        assert fix.command == 'Write-Output (5)#c "x"\nWrite-Output ok'

    def test_block_comment_does_not_nest(self) -> None:
        # PowerShell closes the block comment at the FIRST `#>`.
        fix = fix_pwsh_command('<# one <# two #> three #>\nWrite-Output ok')
        assert fix is not None
        assert fix.command == '<# one <# two #> three #>\nWrite-Output ok'

    def test_multi_here_strings(self) -> None:
        cmd = '$a = @"\nx\n"@\n$b = @\'\ny\n\'@\nWrite-Output $a$b'
        fix = fix_pwsh_command(cmd)
        assert fix is not None
        assert fix.command == cmd
        assert not fix.changed

    def test_quote_inside_subexpression_not_closing_outer(self) -> None:
        # The inner `"b"` must not be mistaken for the outer string's close.
        cmd = 'Write-Output "a$( "b" )c"'
        assert fix_pwsh_command(cmd).command == cmd

    def test_stop_parsing_in_middle_ignores_quotes_after(self) -> None:
        fix = fix_pwsh_command('cmd /c echo --% "a" b\nWrite-Output ok')
        assert fix is not None
        assert fix.command == 'cmd /c echo --% "a" b\nWrite-Output ok'

    def test_crlf_line_endings_in_here_string(self) -> None:
        fix = fix_pwsh_command('@"\r\ncontent " q\r\n"@')
        assert fix is not None
        assert not fix.changed

    def test_pwsh_fix_dataclass_api(self) -> None:
        f = fix_pwsh_command('Write-Output "x')
        assert isinstance(f, PwshFix)
        assert f.command == 'Write-Output "x"'
        assert f.changed is True
        assert fix_pwsh_command("Get-Location").changed is False

    @pytest.mark.parametrize(
        "cmd",
        [
            '"',
            '""',
            '"""',
            '""""""',
            '"\'"',
            "''",
            "'''",
            '@\'@\'@',
            '>"@',
            "<#",
            "#>",
            "--%",
            '"$(',
            "'$( \"x\" )",
            '@"@"',
            '"a`"',
            "'a`b'",
            'Write-Output "a$( "b',
            '"a\'b"c',
            "'it''",
            '@"',
            "@'",
            "<# c #>",
            "cmd /c echo --% `",
            'Write-Output `"',
            '"a` `"',
            "# comment `",
            'Write-Output "a"; "b',
            "Write-Output 'a' 'b",
            '"a$( "b )c"',
            '`" `" `"',
            '@"\r\n"@\r\n"@',
        ],
    )
    def test_never_raises(self, cmd: str) -> None:
        # The parser must never throw on arbitrary input.
        result = fix_pwsh_command(cmd)
        assert result is None or isinstance(result, PwshFix)

    @pytest.mark.parametrize(
        "cmd",
        [
            "$(" * 5000 + ")" * 5000,          # deep $( nesting (was RecursionError)
            '"' + "$(" * 5000 + ")" * 5000 + '"',  # same inside a dq string
            "(" * 100000,                       # pathological paren run
            "a `\n" * 5000,                    # many line continuations
        ],
        ids=["deep-subexpr", "deep-subexpr-in-dq", "paren-run", "many-continuations"],
    )
    def test_deeply_nested_input_never_raises(self, cmd: str) -> None:
        # Regression: _skip_subexpr was recursive and crashed with
        # RecursionError on deeply nested $(...).  It is now iterative.
        result = fix_pwsh_command(cmd)
        assert result is None or isinstance(result, PwshFix)

    def test_fast_path_returns_plain_command_unchanged(self) -> None:
        # Commands without quote/comment/continuation/here-string/--% chars
        # skip the scanner entirely (performance fast path).
        for cmd in ("Get-Location", "git status; cargo test", "   Get-Date   "):
            fix = fix_pwsh_command(cmd)
            assert fix is not None
            assert fix.command == cmd
            assert not fix.changed


# ============================================================================
# Naive validator still behaves as before
# ============================================================================

class TestValidateCommandForPs:
    def test_empty_and_whitespace(self) -> None:
        assert Powershell._validate_command_for_ps("") is not None
        assert Powershell._validate_command_for_ps("   ") is not None
        assert Powershell._validate_command_for_ps("\t\n") is not None

    def test_odd_double_quote_rejected(self) -> None:
        assert Powershell._validate_command_for_ps('Write-Output "a') is not None
        assert Powershell._validate_command_for_ps('Write-Output "a" "b') is not None

    def test_even_double_quote_accepted(self) -> None:
        assert Powershell._validate_command_for_ps('Write-Output "a"') is None
        assert Powershell._validate_command_for_ps('Write-Output "a" "b"') is None

    def test_no_quotes_accepted(self) -> None:
        assert Powershell._validate_command_for_ps("Get-Location") is None


# ============================================================================
# F. Tool integration: warning instead of error, fixed command is executed
# ============================================================================

class TestPwshToolValidationFlow:
    @pytest.fixture(autouse=True)
    def _pwsh_enabled(self) -> Any:
        with _force_pwsh_enabled():
            yield

    @pytest.fixture
    def pwsh_tool(self, mock_session: MagicMock) -> Powershell:
        with patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"
        ):
            return Powershell(session=mock_session)

    def _configure_process_task_mock(
        self, mock_pt: MagicMock, task_id: str = "pwsh-test-id", success: bool = True
    ) -> MagicMock:
        instance = MagicMock()
        instance.start = MagicMock(return_value=asyncio.Future())
        instance.start.return_value.set_result(task_id)
        instance.wait_with_monitor = MagicMock(return_value=asyncio.Future())
        instance.wait_with_monitor.return_value.set_result(None)
        instance.thread_is_alive = MagicMock(return_value=asyncio.Future())
        instance.thread_is_alive.return_value.set_result(False)
        instance.stream = MagicMock()
        instance.stream.pop_output = MagicMock(return_value=asyncio.Future())
        instance.stream.pop_output.return_value.set_result("mock output")
        instance.stream.success = MagicMock(return_value=asyncio.Future())
        instance.stream.success.return_value.set_result(success)
        instance.stream.exit_code = 0
        instance.stream.process_elapsed = None
        mock_pt.return_value = instance
        return instance

    def _wrapped_command(self, mock_pt: MagicMock) -> str:
        """Extract the raw command passed to the mocked ProcessTask."""
        return mock_pt.call_args[0][1][-1]

    # -- execute mode --------------------------------------------------------

    async def test_naive_rejected_but_valid_command_runs_with_warning(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd='Write-Output \'a"b\''))

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "parser verified it is valid" in result.message
        assert "try{" + 'Write-Output \'a"b\'' in self._wrapped_command(mock_pt)

    async def test_unbalanced_quote_repaired_runs_with_warning(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd='Write-Output "hello'))

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert 'appended a closing `"`' in result.message
        # The repaired command (with the appended quote) reaches ProcessTask.
        assert "try{Write-Output \"hello\"}" in self._wrapped_command(mock_pt)

    async def test_unclosed_single_quote_repaired_even_though_naive_passes(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd="Write-Output 'hello"))

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "appended a closing `'`" in result.message
        assert "try{Write-Output 'hello'}" in self._wrapped_command(mock_pt)

    async def test_trailing_comment_gets_newline_before_wrapper(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd="Write-Output ok # done"))

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        wrapped = self._wrapped_command(mock_pt)
        # the comment must be terminated before the wrapper's catch block
        assert "# done\n}catch" in wrapped

    async def test_unbalanced_here_string_repaired(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd='@"\nunclosed here-string'))

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert 'appended a newline and `"@`' in result.message
        wrapped = self._wrapped_command(mock_pt)
        assert "unclosed here-string\n\"@}catch" in wrapped

    async def test_irreparable_returns_error(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            result = await pwsh_tool(PowershellParams(cmd="Write-Output `"))
            mock_pt.assert_not_called()

        assert isinstance(result, ToolError)
        assert "Invalid PowerShell command" in result.message

    async def test_unchanged_valid_command_no_warning(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            self._configure_process_task_mock(mock_pt)
            result = await pwsh_tool(PowershellParams(cmd="Get-Location"))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" not in result.message

    async def test_forbidden_command_still_rejected(
        self, mock_session: MagicMock
    ) -> None:
        mock_session.custom_config.get.return_value = {"forbidden_commands": ["rm -rf"]}
        with _force_pwsh_enabled(), patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"
        ):
            pwsh_tool = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            result = await pwsh_tool(PowershellParams(cmd="rm -rf /"))
            mock_pt.assert_not_called()

        assert isinstance(result, ToolError)
        assert "forbidden" in result.message

    # -- send (background) mode ----------------------------------------------

    async def test_background_repaired_command_runs_with_warning(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            instance = MagicMock()
            instance.start = MagicMock(return_value=asyncio.Future())
            instance.start.return_value.set_result("pwsh-bg-id")
            mock_pt.return_value = instance

            result = await pwsh_tool(
                PowershellParams(cmd='Write-Output "hello', mode="send")
            )

        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "pwsh-bg-id" in result.message
        assert "try{Write-Output \"hello\"}" in self._wrapped_command(mock_pt)

    async def test_background_irreparable_returns_error(
        self, pwsh_tool: Powershell, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            result = await pwsh_tool(PowershellParams(cmd="Write-Output `", mode="send"))
            mock_pt.assert_not_called()

        assert isinstance(result, ToolError)
        assert "Invalid PowerShell command" in result.message


# ============================================================================
# End-to-end integration against a real PowerShell (skipped without pwsh)
# ============================================================================

@NEEDS_PWSH
class TestPwshToolEndToEnd:
    """Real pwsh: the tool must succeed where the old validator failed."""

    @pytest.fixture
    def real_tool(self, mock_session: MagicMock) -> Powershell:
        return Powershell(session=mock_session)

    async def test_naive_rejected_command_succeeds(self, real_tool: Powershell) -> None:
        # ODD double-quote count: old behavior returned "Invalid PowerShell
        # command"; new behavior runs it and prints the quoted text.
        params = PowershellParams(cmd='Write-Output \'a"b\'')
        result = await real_tool(params)
        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "parser verified it is valid" in result.message
        assert 'a"b' in result.output

    async def test_unbalanced_quote_is_repaired_and_runs(
        self, real_tool: Powershell
    ) -> None:
        params = PowershellParams(cmd='Write-Output "hello')
        result = await real_tool(params)
        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "hello" in result.output

    async def test_trailing_comment_command_succeeds(self, real_tool: Powershell) -> None:
        # Before the fix the try/catch wrapper was swallowed by the comment.
        params = PowershellParams(cmd="Write-Output ok # done")
        result = await real_tool(params)
        assert isinstance(result, ToolOk), result.message
        assert "[WARNING]" in result.message
        assert "ok" in result.output

    async def test_irreparable_command_still_errors(self, real_tool: Powershell) -> None:
        params = PowershellParams(cmd="Write-Output `")
        result = await real_tool(params)
        assert isinstance(result, ToolError)
        assert "Invalid PowerShell command" in result.message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
