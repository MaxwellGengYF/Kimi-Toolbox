"""Tests for the long multi-operator shell-command advice.

Covers the shared ``_long_pipeline_advice`` helper (smart, cheap detection:
O(1) length gate, C-level operator-character gate, quote-aware early-exit
scan of logical operators) and its integration into the Bash / Powershell
tool return messages.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from kimi_cli.session import Session

from kimi_agent_sdk import ToolOk
from kimix.tools.common import (
    _PIPE_ADVICE_MIN_LENGTH,
    _PIPE_ADVICE_MIN_OPERATORS,
    _long_pipeline_advice,
)
from kimix.tools.file.bash import Bash, BashParams, Powershell
from kimix.tools.file.bash.pwsh_tool import PowershellParams


def _pad(body: str) -> str:
    """Pad *body* with a trailing comment so the total length exceeds 100 chars."""
    return body + " #" + "x" * (_PIPE_ADVICE_MIN_LENGTH + 1 - len(body) - 2)


class TestLongPipelineAdvice:
    def test_empty_command_no_warning(self) -> None:
        assert _long_pipeline_advice("") == ""

    def test_short_command_no_warning(self) -> None:
        assert _long_pipeline_advice("echo a | grep a | wc -l") == ""

    def test_exactly_min_length_no_warning(self) -> None:
        cmd = _pad("echo a | cat | wc -l")
        # Strip the padding comment down to exactly 100 chars of real command.
        cmd = cmd[: _PIPE_ADVICE_MIN_LENGTH]
        assert len(cmd) == _PIPE_ADVICE_MIN_LENGTH
        assert _long_pipeline_advice(cmd) == ""

    def test_min_length_plus_one_warns(self) -> None:
        cmd = _pad("echo a | cat | wc -l | sort | uniq")
        assert len(cmd) > _PIPE_ADVICE_MIN_LENGTH
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_long_single_pipe_no_warning(self) -> None:
        cmd = _pad("cat some/really/long/path/to/a/data/file.txt | grep needle")
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_long_multiple_pipes_warns(self) -> None:
        cmd = _pad("cat some/really/long/path/to/a/data/file.txt | grep needle | sort | uniq | wc -l")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice
        assert "Python" in advice
        assert "prefer the Python tool" in advice

    def test_warning_reports_length_and_operator_count(self) -> None:
        cmd = _pad("cat aaa/bbb/ccc/ddd | grep xyz | awk '{print $1}' | sort -u | head -20")
        advice = _long_pipeline_advice(cmd)
        assert f"{len(cmd)} chars" in advice
        assert f"{_PIPE_ADVICE_MIN_OPERATORS} operators" in advice

    def test_pipes_inside_single_quotes_ignored(self) -> None:
        # Four `|` chars, but all inside a single-quoted string.
        cmd = _pad("echo 'a | b | c | d' && echo done")
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_pipes_inside_double_quotes_ignored(self) -> None:
        cmd = _pad('echo "a | b | c | d" && echo done')
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_pipes_inside_ansi_c_quotes_ignored(self) -> None:
        cmd = _pad(r"echo $'a | b | c | d' && echo done")
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_logical_or_operators_count(self) -> None:
        # `||` is a logical operator: four of them trigger the warning.
        cmd = _pad(
            "run_a_very_long_command_that_might_fail || run_a_backup_command "
            "|| run_a_restore_command || run_a_cleanup_command || echo oops"
        )
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_single_logical_or_no_warning(self) -> None:
        cmd = _pad("run_a_very_long_command_that_might_fail || echo oops")
        assert _long_pipeline_advice(cmd) == ""

    def test_mixed_logical_or_and_real_pipes(self) -> None:
        cmd = _pad("make_something || (cat aaa/bbb/ccc/ddd | grep xyz | sort -u) && echo ok")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_bash_pipe_ampersand_counts_as_pipeline(self) -> None:
        cmd = _pad(
            "cat aaa/bbb/ccc/ddd |& tee /tmp/some/very/long/path.log "
            "|& grep xyz |& sort -u |& wc -l"
        )
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_ampersand_and_chains_warn(self) -> None:
        cmd = _pad(
            "build_a_very_long_target && run_the_tests && deploy_the_artifact "
            "&& notify_team && report_status"
        )
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_single_ampersand_background_not_counted(self) -> None:
        cmd = _pad("sleep_forever & echo one & echo two & echo three")
        assert _long_pipeline_advice(cmd) == ""

    def test_semicolon_separators_not_counted(self) -> None:
        cmd = _pad("first_command; second_command; third_command; fourth_command")
        assert _long_pipeline_advice(cmd) == ""

    def test_pipes_inside_command_substitution_skipped(self) -> None:
        cmd = _pad("result=$(cat aaa/bbb/ccc/ddd | grep xyz | sort -u | wc -l) && echo $result")
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_pipes_inside_backticks_skipped(self) -> None:
        cmd = _pad("result=`cat aaa/bbb/ccc/ddd | grep xyz | sort -u | wc -l` && echo $result")
        assert "|" in cmd
        assert _long_pipeline_advice(cmd) == ""

    def test_single_operator_no_warning(self) -> None:
        cmd = _pad("cat aaa/bbb/ccc/ddd | grep xyz")
        assert cmd.count("|") >= 1
        assert _long_pipeline_advice(cmd) == ""

    def test_unterminated_quote_does_not_crash(self) -> None:
        cmd = _pad("echo 'unterminated | | |")
        assert _long_pipeline_advice(cmd) == ""

    # ── PowerShell word operators ──────────────────────────────────────

    @pytest.mark.parametrize("op", ["-and", "-or", "-xor", "-not"])
    def test_powershell_word_operator_counts(self, op: str) -> None:
        cmd = _pad(f"$a {op} $b {op} $c {op} $d {op} $e")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_powershell_word_operators_case_insensitive(self) -> None:
        cmd = _pad("$a -AND $b -Or $c -XoR $d -Not $e")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_single_powershell_word_operator_no_warning(self) -> None:
        cmd = _pad("$a -and $b")
        assert _long_pipeline_advice(cmd) == ""

    def test_word_operator_glued_to_identifier_ignored(self) -> None:
        # `-android`, `-notepad` and `foo-and` are flags/words, not operators.
        cmd = _pad("ls -android -notepad foo-and bar-and")
        assert _long_pipeline_advice(cmd) == ""

    def test_mixed_symbolic_and_word_operators(self) -> None:
        cmd = _pad("$a -and $b && (cat aaa/bbb/ccc/ddd | grep xyz) -or $c")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    # ── `!` negation ───────────────────────────────────────────────────

    def test_bang_negation_counts(self) -> None:
        cmd = _pad(
            "if (!$a) { Write-Host one } elseif (!$b) { Write-Host two } "
            "elseif (!$c) { Write-Host three } elseif (!$d) { Write-Host four }"
        )
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_bash_test_bang_counts(self) -> None:
        cmd = _pad(
            "[[ ! -f /some/very/long/path/one && ! -d /some/very/long/path/two "
            "&& ! -e /some/very/long/path/three ]]"
        )
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    def test_bang_not_equal_not_counted(self) -> None:
        cmd = _pad('[[ "$a" != "$b" ]] && echo done')
        assert _long_pipeline_advice(cmd) == ""

    def test_bang_glued_to_word_ignored(self) -> None:
        cmd = _pad("echo hello! && echo world!")
        assert _long_pipeline_advice(cmd) == ""

    def test_shebang_comment_bang_not_counted(self) -> None:
        cmd = "#!/bin/bash\n" + _pad("a_command && b_command && c_command && d_command && e_command")
        advice = _long_pipeline_advice(cmd)
        assert "[WARNING]" in advice

    # ── comments ───────────────────────────────────────────────────────

    def test_operators_inside_comments_ignored(self) -> None:
        cmd = "echo hi && echo ok # && echo hidden || echo also_hidden" + "x" * 60
        assert len(cmd) > _PIPE_ADVICE_MIN_LENGTH
        assert _long_pipeline_advice(cmd) == ""


# ============================================================================
# Integration: warning appears in tool return messages
# ============================================================================

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    session.custom_config.get.return_value = {}
    return session


def _make_process_task_mock(output: str = "mock output") -> MagicMock:
    mock_instance = MagicMock()
    mock_instance.start = MagicMock(return_value=asyncio.Future())
    mock_instance.start.return_value.set_result("task-id-1")
    mock_instance.wait_with_monitor = MagicMock(return_value=asyncio.Future())
    mock_instance.wait_with_monitor.return_value.set_result(None)
    mock_instance.thread_is_alive = MagicMock(return_value=asyncio.Future())
    mock_instance.thread_is_alive.return_value.set_result(False)
    mock_instance.stream = MagicMock()
    mock_instance.stream.pop_output = MagicMock(return_value=asyncio.Future())
    mock_instance.stream.pop_output.return_value.set_result(output)
    mock_instance.stream.success = MagicMock(return_value=asyncio.Future())
    mock_instance.stream.success.return_value.set_result(True)
    mock_instance.stream.exit_code = 0
    mock_instance.stream.process_elapsed = None
    return mock_instance


LONG_PIPELINE_CMD = (
    "cat some/really/long/path/to/a/data/file.txt | grep needle | sort -u | head -50 "
    "| wc -l && echo completed"
)
SHORT_CMD = "echo hello"


class TestBashMessageWarning:
    async def test_success_message_warns_on_long_multi_pipe(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_pt.return_value = _make_process_task_mock()
            result = await bash(BashParams(cmd=LONG_PIPELINE_CMD))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" in result.message
        assert "Python tool" in result.message

    async def test_success_message_no_warning_on_short(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_pt.return_value = _make_process_task_mock()
            result = await bash(BashParams(cmd=SHORT_CMD))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" not in result.message

    async def test_background_message_warns_on_long_multi_pipe(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_pt.return_value = _make_process_task_mock()
            result = await bash(BashParams(cmd=LONG_PIPELINE_CMD, mode="send"))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" in result.message
        assert "Python tool" in result.message


class TestPowershellMessageWarning:
    async def test_success_message_warns_on_long_multi_pipe(
        self, mock_session: MagicMock
    ) -> None:
        pwsh_cmd = (
            "Get-ChildItem -Path C:/Some/Really/Long/Directory/Path/With/Deep/Nesting "
            "| Where-Object { $_.Length -gt 1000 } | Sort-Object Length -Descending "
            "| Select-Object -First 20 | Format-Table -AutoSize"
        )
        with patch(
            "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
            return_value=True,
        ), patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh",
            return_value=r"C:\Program Files\PowerShell\7\pwsh.exe",
        ):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_pt.return_value = _make_process_task_mock()
            result = await pwsh(PowershellParams(cmd=pwsh_cmd))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" in result.message
        assert "Python tool" in result.message

    async def test_success_message_no_warning_on_short(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
            return_value=True,
        ), patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh",
            return_value=r"C:\Program Files\PowerShell\7\pwsh.exe",
        ):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_pt.return_value = _make_process_task_mock()
            result = await pwsh(PowershellParams(cmd="Get-Location"))

        assert isinstance(result, ToolOk)
        assert "[WARNING]" not in result.message
