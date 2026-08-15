"""Tests for the Run tool session continuation and wait_for_pattern support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kimi_cli.session import Session

from kimi_agent_sdk import ToolError, ToolOk
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

            result = await run(RunParams(command="mycustomcmd --flag"))

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args[0]
            # Dedup is always on, but RTK only wraps *known* commands; an
            # unknown command is passed as-is.
            assert args[0] == "mycustomcmd"
            assert args[1] == ["--flag"]

    def test_deduplicate_output_params_removed(self) -> None:
        """Run no longer exposes ``deduplicate_output``/``token_kill``; cwd stays."""
        props = RunParams.model_json_schema()["properties"]
        assert "cwd" in props
        for gone in ("deduplicate_output", "token_kill"):
            assert gone not in props, f"{gone} must be removed from RunParams"

    def test_cwd_accepts_workdir_alias(self) -> None:
        """Run keeps ``cwd`` and still accepts the ``workdir`` spelling."""
        assert RunParams(command="ls", workdir=r"C:\work").cwd == r"C:\work"
        assert RunParams(command="ls", cwd="/tmp").cwd == "/tmp"


class TestRunShellCwdViaCd:
    """Run's ``cwd`` is translated into a ``cd`` statement inside the shell
    command when delegating to Bash/Powershell (they no longer take ``cwd``)."""

    def test_cd_prefix_bash(self) -> None:
        from kimix.tools.file.run import _cd_prefix

        assert _cd_prefix(None, "bash") == ""
        assert _cd_prefix("", "bash") == ""
        assert _cd_prefix("/tmp/work", "bash") == "cd /tmp/work && "
        assert _cd_prefix("/tmp/a b", "bash") == "cd '/tmp/a b' && "

    def test_cd_prefix_pwsh(self) -> None:
        from kimix.tools.file.run import _cd_prefix

        assert _cd_prefix(None, "pwsh") == ""
        assert _cd_prefix(r"C:\work", "pwsh") == r"cd 'C:\work'; "
        assert _cd_prefix(r"C:\it's", "pwsh") == r"cd 'C:\it''s'; "

    async def test_shell_mode_bash_uses_cd_inside(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        captured: dict[str, object] = {}

        async def fake_bash_call(self: object, params: object) -> ToolOk:
            captured["params"] = params
            return ToolOk(output="", message="ok", brief="ok")

        with (
            patch("kimix.tools.file.run.sys.platform", "darwin"),
            patch("kimix.tools.file.bash.bash_tool.Bash.__call__", new=fake_bash_call),
            patch("kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True),
            patch("kimix.tools.file.bash.bash_tool.find_bash", return_value="/bin/bash"),
        ):
            result = await run(
                RunParams(command="echo hi", shell=True, cwd="/tmp/work")
            )
        assert isinstance(result, ToolOk)
        params = captured["params"]
        assert params.cmd == "cd /tmp/work && echo hi"
        # The removed params are not forwarded.
        assert not hasattr(params, "cwd")
        assert not hasattr(params, "deduplicate_output")

    async def test_shell_mode_pwsh_uses_cd_inside(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        captured: dict[str, object] = {}

        async def fake_pwsh_call(self: object, params: object) -> ToolOk:
            captured["params"] = params
            return ToolOk(output="", message="ok", brief="ok")

        with (
            patch("kimix.tools.file.run.sys.platform", "win32"),
            patch("kimix.tools.file.bash.pwsh_tool.Powershell.__call__", new=fake_pwsh_call),
            patch(
                "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
                return_value=True,
            ),
            patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=r"C:\pwsh\pwsh.exe"),
        ):
            result = await run(
                RunParams(command="echo hi", shell=True, cwd=r"C:\work dir")
            )
        assert isinstance(result, ToolOk)
        params = captured["params"]
        assert params.command == r"cd 'C:\work dir'; echo hi"
        assert not hasattr(params, "cwd")
        assert not hasattr(params, "deduplicate_output")


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


# ============================================================================
# Shell enhancement wiring: hardline floor, workdir, real exit code (WP1/2/6)
# ============================================================================

class TestRunSafetyWiring:
    async def test_hardline_blocked(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with patch("kimix.tools.file.run.ProcessTask") as mock_pt:
            result = await run(RunParams(command="rm -rf /"))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (hardline)"
        assert "hardline" in result.message
        mock_pt.assert_not_called()

    async def test_dangerous_cwd_returns_error(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        with patch("kimix.tools.file.run.ProcessTask") as mock_pt:
            result = await run(RunParams(command="python -c print(1)", cwd="a;b"))
        assert isinstance(result, ToolError)
        assert result.brief == "Invalid workdir"
        assert "Invalid workdir" in result.message
        mock_pt.assert_not_called()

    async def test_exactly_one_run_class(self) -> None:
        """The module exposes exactly one Run class — the real one with a
        ``params`` attribute (the dead duplicate was removed, WP6)."""
        from kimix.tools.file.run import Run as RunImported

        assert RunImported is Run
        assert getattr(Run, "params", None) is RunParams

    async def test_failure_block_carries_real_exit_code(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_fail")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="some error output")
            instance.stream.success = AsyncMock(return_value=False)
            instance.stream.exit_code = 42
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="python -c print(1)"))

        assert isinstance(result, ToolError)
        assert "exit_code: 42" in result.output


class TestRunOriginalSavedSuffix:
    async def test_success_message_includes_original_path_after_dedup(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_dedup")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="ERROR\n" * 10)
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="python -c print(1)"))

        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    async def test_success_message_includes_original_path_after_truncate(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_trunc")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(
                return_value="\n".join(f"line_{i}" for i in range(500))
            )
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="python -c print(1)", max_lines=10))

        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    async def test_no_suffix_when_no_filter(
        self, mock_session: MagicMock
    ) -> None:
        run = _run_instance(mock_session)
        with (
            patch("kimix.tools.file.run.ProcessTask") as mock_pt,
            patch("kimix.tools.file.run.shutil.which", return_value="/fake/python"),
            patch("kimix.tools.file.run.Path.is_file", return_value=True),
        ):
            instance = MagicMock()
            instance.start = AsyncMock(return_value="run_plain")
            instance.wait = AsyncMock(return_value=None)
            instance.thread_is_alive = AsyncMock(return_value=False)
            instance.stream = AsyncMock()
            instance.stream.pop_output = AsyncMock(return_value="plain output")
            instance.stream.success = AsyncMock(return_value=True)
            instance.stream.exit_code = 0
            instance.stream.process_elapsed = None
            mock_pt.return_value = instance

            result = await run(RunParams(command="python -c print(1)"))

        assert isinstance(result, ToolOk)
        assert "[original saved to" not in result.message


# ============================================================================
# Long `python -c` payloads are saved to the shared temp folder
# (not the OS temp dir / session folder), via common._create_script_file.
# ============================================================================

class TestRunLongPythonCScript:
    async def test_long_python_c_script_saved_to_temp_folder(
        self, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys as _sys

        run = _run_instance(mock_session)
        long_code = "print(" + "1" * 40000 + ")"
        assert len(long_code) > 30000
        command = f"python -c {long_code}"

        calls: list[tuple[str, str]] = []
        script_path = r"C:\fake\.kimix_cache\tmp_9\42.py"

        def fake_create(content: str, ext: str = ".py") -> str:
            calls.append((content, ext))
            return script_path

        monkeypatch.setattr("kimix.tools.file.run._create_script_file", fake_create)

        instance = MagicMock()
        instance.start = AsyncMock(return_value="run_long")
        instance.wait = AsyncMock(return_value=None)
        instance.thread_is_alive = AsyncMock(return_value=False)
        instance.stream = AsyncMock()
        instance.stream.pop_output = AsyncMock(return_value="ok")
        instance.stream.success = AsyncMock(return_value=True)
        instance.stream.exit_code = 0
        instance.stream.process_elapsed = None
        pt = MagicMock(return_value=instance)
        monkeypatch.setattr("kimix.tools.file.run.ProcessTask", pt)
        monkeypatch.setattr("kimix.tools.file.run.shutil.which", lambda name: None)

        result = await run(RunParams(command=command))

        assert isinstance(result, ToolOk)
        # The long -c payload went through the temp-folder writer, not
        # tempfile.NamedTemporaryFile, and replaced the -c <code> args.
        assert calls == [(long_code, ".py")]
        pt.assert_called_once()
        ctor_args = pt.call_args[0]
        assert ctor_args[0] == _sys.executable
        assert ctor_args[1] == [script_path]

    async def test_short_python_c_script_not_exported(self, mock_session: MagicMock) -> None:
        run = _run_instance(mock_session)
        command = "python -c print(1)"
        calls: list[tuple[str, str]] = []

        def fake_create(content: str, ext: str = ".py") -> str:
            calls.append((content, ext))
            return r"C:\fake\script.py"

        instance = MagicMock()
        instance.start = AsyncMock(return_value="run_short")
        instance.wait = AsyncMock(return_value=None)
        instance.thread_is_alive = AsyncMock(return_value=False)
        instance.stream = AsyncMock()
        instance.stream.pop_output = AsyncMock(return_value="ok")
        instance.stream.success = AsyncMock(return_value=True)
        instance.stream.exit_code = 0
        instance.stream.process_elapsed = None
        with (
            patch("kimix.tools.file.run._create_script_file", side_effect=fake_create),
            patch("kimix.tools.file.run.shutil.which", lambda name: None),
            patch("kimix.tools.file.run.ProcessTask", return_value=instance),
        ):
            result = await run(RunParams(command=command))

        assert isinstance(result, ToolOk)
        assert calls == []
