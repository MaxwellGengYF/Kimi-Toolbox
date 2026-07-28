"""Tests for the Run tool session continuation and wait_for_pattern support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool

from kimix.tools.background.utils import TaskData
from kimix.tools.file.run import Run, RunParams


def _run_instance(session: Session) -> Run:
    """Create a Run instance even when the platform would normally skip it."""
    with (
        patch("kimix.tools.file.run.USE_SYSTEM_SHELL", True),
        patch("kimix.tools.file.run.USE_SYSTEM_PWSH_ON_WINDOWS", False),
        patch("kimix.tools.file.run.find_bash", return_value=None),
    ):
        return Run(session=session)


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    session.custom_config.get.return_value = {}
    return session


class TestRunParams:
    def test_task_id_with_command_succeeds(self) -> None:
        p = RunParams(command="hello", task_id="run_0")
        assert p.task_id == "run_0"
        assert p.command == "hello"

    def test_wait_for_pattern_optional(self) -> None:
        p = RunParams(command="hello", wait_for_pattern="done")
        assert p.wait_for_pattern == "done"

    def test_deduplicate_output_defaults_true(self) -> None:
        p = RunParams(command="hello")
        assert p.deduplicate_output is True

    def test_deduplicate_output_can_be_disabled(self) -> None:
        p = RunParams(command="hello", deduplicate_output=False)
        assert p.deduplicate_output is False

    def test_deduplicate_output_accepts_token_kill_alias(self) -> None:
        p = RunParams(command="hello", token_kill=False)
        assert p.deduplicate_output is False

    def test_defaults(self) -> None:
        p = RunParams(command="ls")
        assert p.timeout == 30
        assert p.deduplicate_output is True
        assert p.mode == "execute"
        assert p.shell is False

    def test_accepts_cmd_alias(self) -> None:
        p = RunParams(cmd="echo hello")
        assert p.command == "echo hello"

    def test_mode_send_requires_task_id(self) -> None:
        with pytest.raises(ValueError, match="task_id"):
            RunParams(command="hi", mode="send")

    def test_mode_send_with_task_id_succeeds(self) -> None:
        p = RunParams(command="hi", task_id="run_0")
        assert p.mode == "send"
        assert p.task_id == "run_0"

    def test_shell_flag(self) -> None:
        p = RunParams(command="ls -la | head", shell=True)
        assert p.shell is True
        p2 = RunParams(command="ls", shell=False)
        assert p2.shell is False

    def test_timeout_min(self) -> None:
        p = RunParams(command="ls", timeout=1)
        assert p.timeout == 1

    def test_timeout_max(self) -> None:
        with pytest.raises(Exception):
            RunParams(command="ls", timeout=901)

    def test_max_lines_field(self) -> None:
        p = RunParams(command="ls", max_lines=50)
        assert p.max_lines == 50
        p2 = RunParams(command="ls", max_lines=None)
        assert p2.max_lines is None


class TestRunRtkRewrite:
    async def test_run_prepends_rtk_for_known_command(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run._rtk_binary_path", return_value=Path("/fake/share/bin/rtk")),
            patch("kimix.tools.file.run.shutil.which") as mock_which,
        ):
            mock_which.side_effect = lambda name: f"/fake/{name}"
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_rtk")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="mock output")
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="git status"))

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args[0]
            assert args[0] == str(Path("/fake/share/bin/rtk"))
            assert args[1] == ["git", "status"]

    async def test_run_does_not_prepend_rtk_for_unknown_command(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run._rtk_binary_path", return_value=Path("/fake/share/bin/rtk")),
            patch("kimix.tools.file.run.shutil.which") as mock_which,
        ):
            mock_which.side_effect = lambda name: f"/fake/{name}"
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_unknown")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="mock output")
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="git status", deduplicate_output=False))

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args[0]
            # With deduplicate_output=False, RTK is disabled and the command is passed as-is.
            # shutil.which("git") returns "/fake/git" from the mock side_effect.
            assert args[0] == "git"
            assert args[1] == ["status"]

    async def test_run_token_kill_false_does_not_prepend_rtk(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run._rtk_binary_path", return_value=Path("/fake/share/bin/rtk")),
            patch("kimix.tools.file.run.shutil.which") as mock_which,
        ):
            mock_which.side_effect = lambda name: f"/fake/{name}"
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_no_rtk")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="mock output")
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="git status", token_kill=False))

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args[0]
            assert args[0] == "git"
            assert args[1] == ["status"]


class TestRunContinueSession:
    async def test_continue_nonexistent_task_lists_available(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        data.tasks = {"run_alive": stream}
        run._session.custom_data["background_task_data"] = data

        result = await run(RunParams(command="hi", task_id="missing"))
        assert isinstance(result, ToolError)
        assert "missing" in result.message
        assert "run_alive" in result.message

    async def test_invalid_wait_for_pattern_returns_error(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        result = await run(RunParams(command="hi", wait_for_pattern="["))
        assert isinstance(result, ToolError)
        assert "Invalid wait_for_pattern" in result.message

    async def test_continue_session_sends_input_and_returns_block(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("process output", True, 0.12))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"run_42": stream}
        run._session.custom_data["background_task_data"] = data

        result = await run(
            RunParams(command="input line", task_id="run_42", wait_for_pattern="output")
        )

        assert isinstance(result, ToolOk)
        assert "run_42" in result.output
        assert "status: running" in result.output
        assert "wait_matched: true" in result.output
        stream.input.assert_awaited_once_with("input line\n")


class TestRunStartModes:
    async def test_one_shot_command_still_works(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_test")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="mock output")
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="python -c print(1)"))

            assert isinstance(result, ToolOk)
            assert "run_test" in result.output
            assert "status: completed" in result.output
            assert "mock output" in result.output

    async def test_background_with_wait_for_pattern(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_bg")
            instance.stream = AsyncMock()
            instance.stream.wait_for_output = AsyncMock(return_value=("ready", True, 0.05))
            instance.stream.thread_is_alive = AsyncMock(return_value=True)
            mock_pt.return_value = instance

            result = await run(
                RunParams(command="python -c print('ready')", run_in_background=True, wait_for_pattern="ready")
            )

            assert isinstance(result, ToolOk)
            assert "run_bg" in result.output
            assert "status: running" in result.output
            assert "wait_matched: true" in result.output
