"""Tests for the PowerShell tool interactive mode and shared ProcessTask behavior."""

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from kimi_agent_sdk import ToolError, ToolOk
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool

from kimix.tools.common import ProcessTask, _rtk_binary_path
from kimix.tools.file.bash import Powershell
from kimix.tools.file.bash.pwsh_tool import PowershellParams, _PWSH_CONSOLE_INIT, find_pwsh
from kimix.tools.background.utils import TaskData, _pop_task_data


def _pwsh_is_available() -> bool:
    """Return True when Powershell can be instantiated on this platform."""
    try:
        Powershell(session=MagicMock(spec=Session))
        return True
    except SkipThisTool:
        return False


PWSH_AVAILABLE = _pwsh_is_available()


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
# PowershellParams validation
# ============================================================================

class TestPowershellParams:
    def test_cmd_only_succeeds(self) -> None:
        p = PowershellParams(cmd="Get-Location")
        assert p.cmd == "Get-Location"
        assert p.mode == "execute"

    def test_empty_cmd_execute_raises(self) -> None:
        with pytest.raises(ValueError):
            PowershellParams(cmd="")

    def test_empty_cmd_interactive_succeeds(self) -> None:
        p = PowershellParams(cmd="", mode="interactive")
        assert p.cmd == ""
        assert p.mode == "interactive"

    def test_cmd_and_interactive_succeeds(self) -> None:
        p = PowershellParams(cmd="Get-Location", mode="interactive")
        assert p.cmd == "Get-Location"
        assert p.mode == "interactive"

    def test_run_alias_execute(self) -> None:
        p = PowershellParams(cmd="Get-Date", mode="run")
        assert p.mode == "execute"

    def test_background_alias_send(self) -> None:
        p = PowershellParams(cmd="Get-Date", mode="background")
        assert p.mode == "send"

    def test_send_new_behavior(self) -> None:
        p = PowershellParams(cmd="Get-Date", mode="send")
        assert p.mode == "send"

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowershellParams(cmd="Get-Date", mode="exec")

    def test_mode_description_documents_aliases(self) -> None:
        desc = PowershellParams.model_fields["mode"].description or ""
        assert "run" in desc
        assert "background" in desc


class TestPowershellParamsFullCoverage:
    """Cover every PowershellParams field and alias."""

    def test_command_alias_accepted(self) -> None:
        p = PowershellParams(command="Get-Location")
        assert p.cmd == "Get-Location"

    def test_legacy_interactive_flag_sets_mode(self) -> None:
        p = PowershellParams(cmd="Get-Date", interactive=True)
        assert p.mode == "interactive"

    def test_task_id_with_cmd_valid(self) -> None:
        p = PowershellParams(cmd="dir", task_id="pwsh_1")
        assert p.task_id == "pwsh_1"
        assert p.mode == "execute"

    def test_task_id_with_any_mode_valid(self) -> None:
        for mode in ("execute", "send", "interactive"):
            p = PowershellParams(cmd="dir", task_id="pwsh_1", mode=mode)
            assert p.task_id == "pwsh_1"

    def test_task_id_without_cmd_rejected(self) -> None:
        with pytest.raises(ValueError):
            PowershellParams(cmd="", task_id="pwsh_1", mode="send")

    def test_wait_for_pattern_field(self) -> None:
        p = PowershellParams(cmd="Get-Date", wait_for_pattern="^ok")
        assert p.wait_for_pattern == "^ok"

    def test_timeout_default(self) -> None:
        assert PowershellParams(cmd="x").timeout == 30

    def test_timeout_min_max(self) -> None:
        assert PowershellParams(cmd="x", timeout=1).timeout == 1
        assert PowershellParams(cmd="x", timeout=900).timeout == 900

    def test_timeout_below_min_rejected(self) -> None:
        with pytest.raises(ValueError):
            PowershellParams(cmd="x", timeout=0)

    def test_timeout_above_max_rejected(self) -> None:
        with pytest.raises(ValueError):
            PowershellParams(cmd="x", timeout=901)

    def test_max_lines_field_and_none(self) -> None:
        assert PowershellParams(cmd="x", max_lines=50).max_lines == 50
        assert PowershellParams(cmd="x", max_lines=None).max_lines is None

    def test_max_lines_below_min_rejected(self) -> None:
        with pytest.raises(ValueError):
            PowershellParams(cmd="x", max_lines=2)

    def test_deduplicate_output_default_true(self) -> None:
        assert PowershellParams(cmd="x").deduplicate_output is True

    def test_deduplicate_output_new_name(self) -> None:
        assert PowershellParams(cmd="x", deduplicate_output=False).deduplicate_output is False

    def test_deduplicate_output_token_kill_alias(self) -> None:
        assert PowershellParams(cmd="x", token_kill=False).deduplicate_output is False


# ============================================================================
# Argument building
# ============================================================================

class TestPowershellArgumentBuilding:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock(spec=Session)
        session.custom_data = {}
        session.custom_config.get.return_value = {}
        return session

    async def test_non_interactive_args(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("pwsh-test-id")
            mock_instance.wait_with_monitor = MagicMock(return_value=asyncio.Future())
            mock_instance.wait_with_monitor.return_value.set_result(None)
            mock_instance.thread_is_alive = MagicMock(return_value=asyncio.Future())
            mock_instance.thread_is_alive.return_value.set_result(False)
            mock_instance.stream = MagicMock()
            mock_instance.stream.pop_output = MagicMock(return_value=asyncio.Future())
            mock_instance.stream.pop_output.return_value.set_result("mock output")
            mock_instance.stream.success = MagicMock(return_value=asyncio.Future())
            mock_instance.stream.success.return_value.set_result(True)
            mock_instance.stream.exit_code = 0
            mock_instance.stream.process_elapsed = None
            mock_pt.return_value = mock_instance

            params = PowershellParams(cmd="Get-Location")
            result = await pwsh(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            assert "-NonI" in args[0][1]
            assert "-NoExit" not in args[0][1]
            assert "-Command" in args[0][1] or "-C" in args[0][1]
            assert isinstance(args[0][3], dict) and "PATH" in args[0][3]
            assert args.kwargs.get("append_newline", False) is False

    async def test_interactive_args_with_cmd(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("pwsh-interactive-id")
            mock_pt.return_value = mock_instance

            params = PowershellParams(cmd="Read-Host 'Name'", mode="interactive")
            result = await pwsh(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            ps_args = args[0][1]
            assert "-NonI" not in ps_args
            assert "-NoExit" in ps_args
            assert "-Command" in ps_args or "-C" in ps_args
            assert any("Read-Host 'Name'" in arg for arg in ps_args)
            assert args.kwargs.get("append_newline") is True or args[0][4] is True

    async def test_interactive_args_without_cmd(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("pwsh-interactive-id")
            mock_pt.return_value = mock_instance

            params = PowershellParams(cmd="", mode="interactive")
            result = await pwsh(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            ps_args = args[0][1]
            assert ps_args == ["-NoP", "-Exec", "Bypass", "-NoL", "-NoExit", "-Command", _PWSH_CONSOLE_INIT]

    async def test_interactive_returns_immediately(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
            pwsh = Powershell(session=mock_session)

        with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("task-123")
            mock_pt.return_value = mock_instance

            params = PowershellParams(cmd="", mode="interactive")
            result = await pwsh(params)

            assert isinstance(result, ToolOk)
            assert "task-123" in result.message
            assert "task_id" in result.message
            assert "TaskOutput" in result.message
            mock_instance.wait.assert_not_called()


# ============================================================================
# ProcessTask newline appending
# ============================================================================

class TestProcessTaskAppendNewline:
    def test_appends_newline_when_enabled(self) -> None:
        task = ProcessTask("cmd.exe", ["/c", "echo hello"], append_newline=True)
        # Drive data through the internal queue by calling input on an unstarted task.
        # _input_function waits for the process ref; simulate a fake running process.
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.returncode = None
        task._process_ref = fake_proc

        async def _send() -> bool:
            return await task._input_function("hello")

        assert asyncio.run(_send()) is True
        assert task._input_queue.get_nowait() == "hello\n"

    def test_no_append_when_disabled(self) -> None:
        task = ProcessTask("cmd.exe", ["/c", "echo hello"], append_newline=False)
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.returncode = None
        task._process_ref = fake_proc

        async def _send() -> bool:
            return await task._input_function("hello")

        assert asyncio.run(_send()) is True
        assert task._input_queue.get_nowait() == "hello"

    def test_no_double_newline(self) -> None:
        task = ProcessTask("cmd.exe", ["/c", "echo hello"], append_newline=True)
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.returncode = None
        task._process_ref = fake_proc

        async def _send() -> bool:
            return await task._input_function("hello\n")

        assert asyncio.run(_send()) is True
        assert task._input_queue.get_nowait() == "hello\n"


# ============================================================================
# Powershell session continuation / wait_for_pattern
# ============================================================================

class TestPowershellSessionContinuation:
    @pytest.fixture
    def pwsh_instance(self, mock_session: MagicMock) -> Powershell:
        with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"):
            return Powershell(session=mock_session)

    async def test_continue_nonexistent_task_lists_available(self, pwsh_instance: Powershell) -> None:
        from unittest.mock import AsyncMock

        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        data.tasks = {"pwsh_alive": stream}
        pwsh_instance._session.custom_data["background_task_data"] = data

        result = await pwsh_instance(PowershellParams(cmd="Write-Host hi", task_id="missing"))
        assert isinstance(result, ToolError)
        assert "missing" in result.message
        assert "pwsh_alive" in result.message

    async def test_invalid_wait_for_pattern_returns_error(self, pwsh_instance: Powershell) -> None:
        result = await pwsh_instance(PowershellParams(cmd="Get-Date", wait_for_pattern="["))
        assert isinstance(result, ToolError)
        assert "Invalid wait_for_pattern" in result.message

    async def test_continue_session_sends_input_and_returns_block(self, pwsh_instance: Powershell) -> None:
        from unittest.mock import AsyncMock

        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("hello output", True, 0.12))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"pwsh_42": stream}
        pwsh_instance._session.custom_data["background_task_data"] = data

        result = await pwsh_instance(
            PowershellParams(cmd="Write-Host hello", task_id="pwsh_42", wait_for_pattern="hello")
        )

        assert isinstance(result, ToolOk)
        assert "pwsh_42" in result.output
        assert "status: running" in result.output
        assert "wait_matched: true" in result.output
        stream.input.assert_awaited_once_with("Write-Host hello\n")


# ============================================================================
# Integration tests
# ============================================================================

@pytest.mark.skipif(
    not PWSH_AVAILABLE,
    reason="PowerShell tool is not available on this platform",
)
class TestPowershellInteractiveIntegration:
    async def test_interactive_read_host(self, mock_session: MagicMock) -> None:
        pwsh = Powershell(session=mock_session)
        params = PowershellParams(cmd="Read-Host -Prompt 'Name'", mode="interactive")
        result = await pwsh(params)
        assert isinstance(result, ToolOk)
        task_id = result.message.split("`")[1]

        task_data = mock_session.custom_data.get("background_task_data")
        assert task_data is not None
        task = task_data.tasks.get(task_id)
        assert task is not None

        await task.input("Alice")
        # Poll for output with timeout instead of fixed sleep to handle CI/load variations
        output = ""
        for _ in range(20):
            await asyncio.sleep(0.1)
            output = await task.get_output()
            if "Alice" in output:
                break
        assert "Alice" in output, f"Expected 'Alice' in output, got: {output!r}"

        await task.input("exit")
        await task.wait(timeout=5)

    async def test_persistent_repl_session(self, mock_session: MagicMock) -> None:
        pwsh = Powershell(session=mock_session)
        params = PowershellParams(cmd="", mode="interactive")
        result = await pwsh(params)
        assert isinstance(result, ToolOk)
        task_id = result.message.split("`")[1]

        task_data = mock_session.custom_data.get("background_task_data")
        assert task_data is not None
        task = task_data.tasks.get(task_id)
        assert task is not None

        await task.input("Write-Output hello")
        # Poll for output with timeout instead of fixed sleep to handle CI/load variations
        output = ""
        for _ in range(20):
            await asyncio.sleep(0.1)
            output = await task.get_output()
            if "hello" in output:
                break
        assert "hello" in output, f"Expected 'hello' in output, got: {output!r}"

        await task.input("$x = 42")
        await asyncio.sleep(0.2)
        await task.input("Write-Output $x")
        # Poll for output
        output = ""
        for _ in range(20):
            await asyncio.sleep(0.1)
            output = await task.get_output()
            if "42" in output:
                break
        assert "42" in output, f"Expected '42' in output, got: {output!r}"

        await task.input("exit")
        await task.wait(timeout=5)

    async def test_non_interactive_unchanged(self, mock_session: MagicMock) -> None:
        pwsh = Powershell(session=mock_session)
        params = PowershellParams(cmd="Write-Output noninteractive")
        result = await pwsh(params)
        assert isinstance(result, ToolOk)
        assert "noninteractive" in result.output

    async def test_interactive_start_with_wait_for_pattern(self, mock_session: MagicMock) -> None:
        pwsh = Powershell(session=mock_session)
        params = PowershellParams(
            cmd="Write-Host hello", interactive=True, wait_for_pattern="hello", timeout=10
        )
        result = await pwsh(params)
        assert isinstance(result, ToolOk)
        assert "pwsh" in result.output
        assert "status:" in result.output
        assert "wait_matched: true" in result.output
        assert "hello" in result.output

        task_id = result.output.split("task_id: ", 1)[1].split("\n", 1)[0]
        exit_result = await pwsh(PowershellParams(cmd="exit", task_id=task_id, timeout=5))
        assert isinstance(exit_result, ToolOk)
        assert "status: completed" in exit_result.output


# ============================================================================
# RTK rewrite path
# ============================================================================

class TestPowershellRtkRewrite:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock(spec=Session)
        session.custom_data = {}
        session.custom_config.get.return_value = {}
        return session

    async def test_pwsh_rewrites_known_command_to_bare_rtk(
        self, mock_session: MagicMock
    ) -> None:
        _rtk_binary_path.cache_clear()
        with patch("kimix.tools.common._rtk_available", return_value=True), patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"
        ):
            pwsh = Powershell(session=mock_session)

            with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
                mock_instance = MagicMock()
                mock_instance.start = MagicMock(return_value=asyncio.Future())
                mock_instance.start.return_value.set_result("pwsh-rtk-id")
                mock_instance.wait = MagicMock(return_value=asyncio.Future())
                mock_instance.wait.return_value.set_result(None)
                mock_instance.wait_with_monitor = MagicMock(return_value=asyncio.Future())
                mock_instance.wait_with_monitor.return_value.set_result((False, 0.0, False))
                mock_instance.thread_is_alive = MagicMock(return_value=asyncio.Future())
                mock_instance.thread_is_alive.return_value.set_result(False)
                mock_instance.stream = MagicMock()
                mock_instance.stream.pop_output = MagicMock(return_value=asyncio.Future())
                mock_instance.stream.pop_output.return_value.set_result("mock output")
                mock_instance.stream.success = MagicMock(return_value=asyncio.Future())
                mock_instance.stream.success.return_value.set_result(True)
                mock_instance.stream.exit_code = 0
                mock_instance.stream.process_elapsed = None
                mock_pt.return_value = mock_instance

                params = PowershellParams(cmd="git status")
                result = await pwsh(params)

                assert isinstance(result, ToolOk)
                args = mock_pt.call_args
                assert "& rtk git status" in args[0][1][-1]
                assert "rtk" in result.display[0].command
                assert str(_rtk_binary_path()) not in result.display[0].command
                assert "rtk" in result.message

    async def test_pwsh_compound_command_rewrites_each_segment(
        self, mock_session: MagicMock
    ) -> None:
        _rtk_binary_path.cache_clear()
        with patch("kimix.tools.common._rtk_available", return_value=True), patch(
            "kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"
        ):
            pwsh = Powershell(session=mock_session)

            with patch("kimix.tools.file.bash.pwsh_tool.ProcessTask") as mock_pt:
                mock_instance = MagicMock()
                mock_instance.start = MagicMock(return_value=asyncio.Future())
                mock_instance.start.return_value.set_result("pwsh-rtk-id")
                mock_instance.wait = MagicMock(return_value=asyncio.Future())
                mock_instance.wait.return_value.set_result(None)
                mock_instance.wait_with_monitor = MagicMock(return_value=asyncio.Future())
                mock_instance.wait_with_monitor.return_value.set_result((False, 0.0, False))
                mock_instance.thread_is_alive = MagicMock(return_value=asyncio.Future())
                mock_instance.thread_is_alive.return_value.set_result(False)
                mock_instance.stream = MagicMock()
                mock_instance.stream.pop_output = MagicMock(return_value=asyncio.Future())
                mock_instance.stream.pop_output.return_value.set_result("mock output")
                mock_instance.stream.success = MagicMock(return_value=asyncio.Future())
                mock_instance.stream.success.return_value.set_result(True)
                mock_instance.stream.exit_code = 0
                mock_instance.stream.process_elapsed = None
                mock_pt.return_value = mock_instance

                params = PowershellParams(cmd="git status; cargo test")
                result = await pwsh(params)

                assert isinstance(result, ToolOk)
                args = mock_pt.call_args
                assert "& rtk git status; & rtk cargo test" in args[0][1][-1]
                assert "rtk" in result.display[0].command
                assert str(_rtk_binary_path()) not in result.display[0].command
