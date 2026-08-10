"""Tests for shell-aware TodoList `code` execution (bash vs PowerShell).

Covers the runtime shell-kind detection, the dialect-specific description
builders, the params-schema patching, the fixer-aware argv building in
``TodoList._shell_argv`` / ``_run_code``, and the timeout process-tree kill
in ``TodoList._run_process``.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from kimi_cli.tools.todo import (
    TodoList,
    _detect_shell_kind,
    _get_shell_kind,
    _tool_description,
)


def _collect_code_descriptions(schema: Any) -> list[str]:
    """Return every ``properties.code.description`` found in a JSON schema."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and isinstance(props.get("code"), dict):
                desc = props["code"].get("description")
                if isinstance(desc, str) and desc.startswith("Verification code:"):
                    out.append(desc)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return out


@pytest.fixture(autouse=True)
def _clear_shell_kind_cache() -> Any:
    """Reset the lru_cache so monkeypatched detection never leaks between tests."""
    _detect_shell_kind.cache_clear()
    yield
    _detect_shell_kind.cache_clear()


class TestDetectShellKind:
    def test_bash_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", lambda: True
        )
        _detect_shell_kind.cache_clear()
        assert _detect_shell_kind() == "bash"

    def test_powershell_when_bash_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", lambda: False
        )
        _detect_shell_kind.cache_clear()
        assert _detect_shell_kind() == "powershell"

    def test_get_shell_kind_runtime_resolution(self) -> None:
        # Runtime resolution works and is cached (no import at module load).
        assert _get_shell_kind() in ("bash", "powershell")


class TestDescriptionBuilders:


    def test_instance_description_matches_runtime_kind(self) -> None:
        tl = TodoList(runtime=MagicMock())
        expected = _tool_description(_get_shell_kind())
        assert tl.description == expected
        assert tl._shell_kind == _get_shell_kind()


class TestShellArgv:
    def test_bash_argv_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kimix.tools.file.bash.bash_tool.find_bash", lambda: "/bin/bash")
        argv, hint, env = TodoList._shell_argv("echo hi", shell_kind="bash")  # type: ignore[misc]
        assert argv == ["/bin/bash", "-l", "-c", "echo hi"]
        assert hint == "bash executable not found: /bin/bash"
        assert isinstance(env, dict)

    def test_bash_fixer_applied_before_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import bash_fix, bash_tool

        seen: list[str] = []
        orig_prepare = bash_tool._prepare_bash_cmd
        orig_fix = bash_fix.fix_bash_command

        def fake_prepare(cmd: str) -> str:
            seen.append("prepare")
            return orig_prepare(cmd)

        def fake_fix(cmd: str) -> Any:
            seen.append("fix")
            return orig_fix(cmd)

        monkeypatch.setattr(bash_tool, "_prepare_bash_cmd", fake_prepare)
        monkeypatch.setattr(bash_fix, "fix_bash_command", fake_fix)
        argv, _hint, _env = TodoList._shell_argv("pwd", shell_kind="bash")  # type: ignore[misc]
        assert seen == ["prepare", "fix"]
        assert isinstance(argv[0], str) and argv[0]

    def test_pwsh_argv_with_pwsh7(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: r"C:\pwsh.exe")
        argv, hint, env = TodoList._shell_argv("Write-Output hi", shell_kind="powershell")  # type: ignore[misc]
        assert argv[0] == r"C:\pwsh.exe"
        assert argv[1:7] == ["-NoP", "-NonI", "-Exec", "Bypass", "-NoL", "-Command"]
        raw = argv[7]
        assert "try{Write-Output hi}catch" in raw
        assert ";exit $LASTEXITCODE" in raw
        assert hint == r"PowerShell executable not found: C:\pwsh.exe"
        assert env is None

    def test_pwsh_invalid_command_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import pwsh_fix

        monkeypatch.setattr(pwsh_fix, "fix_pwsh_command", lambda cmd: None)
        assert TodoList._shell_argv("Write-Output x", shell_kind="powershell") is None

    def test_pwsh_ps51_fallback_transform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import process_pwsh, pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: None)
        monkeypatch.setattr(
            process_pwsh, "pwsh_transform", lambda cmd: ("transformed", ["warn"])
        )
        argv, _hint, env = TodoList._shell_argv("Write-Output hi", shell_kind="powershell")  # type: ignore[misc]
        assert "transformed" in argv[7]
        assert argv[0] == "powershell" or argv[0].lower().endswith("powershell.exe")
        assert env is None


class TestRunCode:
    async def test_shell_bash_passes_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import bash_tool

        monkeypatch.setattr(bash_tool, "find_bash", lambda: "/bin/bash")
        captured: dict[str, Any] = {}

        async def fake_run_process(
            argv: list[str],
            timeout: int,
            *,
            not_found_hint: str,
            env: dict[str, str] | None = None,
        ) -> tuple[bool, str]:
            captured["argv"] = argv
            captured["env"] = env
            captured["timeout"] = timeout
            captured["hint"] = not_found_hint
            return True, "ok"

        monkeypatch.setattr(TodoList, "_run_process", fake_run_process)
        ok, out = await TodoList._run_code("!echo hi", shell_kind="bash")
        assert ok and out == "ok"
        assert captured["argv"] == ["/bin/bash", "-l", "-c", "echo hi"]
        assert captured["env"] is not None

    async def test_shell_pwsh_invalid_not_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import pwsh_fix

        monkeypatch.setattr(pwsh_fix, "fix_pwsh_command", lambda cmd: None)
        captured: dict[str, Any] = {}

        async def fake_run_process(*args: Any, **kwargs: Any) -> tuple[bool, str]:
            captured["called"] = True
            return True, "should not happen"

        monkeypatch.setattr(TodoList, "_run_process", fake_run_process)
        ok, out = await TodoList._run_code("!Write-Output x", shell_kind="powershell")
        assert ok is False
        assert out == "Invalid PowerShell command."
        assert "called" not in captured

    async def test_python_path_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_run_process(
            argv: list[str],
            timeout: int,
            *,
            not_found_hint: str,
            env: dict[str, str] | None = None,
        ) -> tuple[bool, str]:
            captured["argv"] = argv
            captured["env"] = env
            return True, "ok"

        monkeypatch.setattr(TodoList, "_run_process", fake_run_process)
        ok, out = await TodoList._run_code("print('hi')")
        assert ok and out == "ok"
        assert captured["argv"][0] == sys.executable
        assert captured["env"] is None


class TestResolveCodeExecutable:
    def test_shell_command(self) -> None:
        assert TodoList._resolve_code_executable("!pytest tests/ -x -q") == (
            "shell",
            "pytest tests/ -x -q",
        )

    def test_shell_file_routing(self, tmp_path: Any) -> None:
        sh = tmp_path / "run.sh"
        sh.write_text("echo hi\n")
        ps1 = tmp_path / "run.ps1"
        ps1.write_text("Write-Output hi\n")
        assert TodoList._resolve_code_executable(str(sh)) == ("shell_file", str(sh))
        assert TodoList._resolve_code_executable(str(ps1)) == ("shell_file", str(ps1))

    def test_python_file(self, tmp_path: Any) -> None:
        py = tmp_path / "run.py"
        py.write_text("print('hi')\n")
        assert TodoList._resolve_code_executable(str(py)) == ("python", str(py))

    def test_empty_and_blank(self) -> None:
        assert TodoList._resolve_code_executable("") is None
        assert TodoList._resolve_code_executable("!") is None

    def test_whitespace_only_is_inline_python(self) -> None:
        # Pre-existing behavior: whitespace-only code falls through to inline
        # Python (the shell/file checks reject it), never None.
        kind, payload = TodoList._resolve_code_executable("   ")  # type: ignore[misc]
        assert kind == "python_inline"
        assert isinstance(payload, str) and payload


class TestRunProcessTreeKill:
    """``_run_process`` must kill the whole tree on timeout, not just the
    direct child, so a grandchild cannot keep the verification running or
    leak as an orphan."""

    @staticmethod
    def _pid_runs_script(pid: int, marker: str) -> bool:
        """Return True if *pid* runs a command line containing *marker*.

        Command-line matching is immune to Windows PID reuse (a busy machine
        recycles PIDs within milliseconds of death), which makes plain
        aliveness checks by PID unreliable.
        """
        if os.name == "nt":
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            ).stdout
            return marker in out
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.exists():
            try:
                cmd = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                return False
            return marker in cmd
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    async def test_timeout_kills_grandchild_tree(self, tmp_path: Any) -> None:
        interp = getattr(sys, "_base_executable", None) or sys.executable
        pidfile = tmp_path / "todo_grandchild.pid"

        grandchild_script = tmp_path / "todo_grandchild.py"
        grandchild_script.write_text(
            "import os, time\n"
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "while True:\n"
            "    time.sleep(1)\n",
            encoding="utf-8",
        )
        parent_script = tmp_path / "todo_parent.py"
        parent_script.write_text(
            "import subprocess, sys\n"
            f"gc = subprocess.Popen([{interp!r}, {str(grandchild_script)!r}])\n"
            "try:\n"
            "    gc.wait()\n"
            "except Exception:\n"
            "    pass\n",
            encoding="utf-8",
        )

        ok, out = await TodoList._run_process(
            [interp, str(parent_script)],
            timeout=1,
            not_found_hint="n/a",
        )
        assert ok is False
        assert "timed out" in out

        grandchild_pid: int | None = None
        try:
            assert pidfile.exists(), "grandchild never started"
            grandchild_pid = int(pidfile.read_text(encoding="utf-8").strip())
            # The grandchild must be dead after the timeout (the whole tree
            # was killed).  Wait briefly for taskkill/killpg to settle.
            deadline = time.monotonic() + 10
            while (
                self._pid_runs_script(grandchild_pid, "todo_grandchild.py")
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.2)
            assert not self._pid_runs_script(
                grandchild_pid, "todo_grandchild.py"
            ), "grandchild survived the timeout tree-kill"
        finally:
            if (
                grandchild_pid is not None
                and self._pid_runs_script(grandchild_pid, "todo_grandchild.py")
            ):
                from kimix.tools.common import kill_child_tree

                kill_child_tree(grandchild_pid, force=True)

    async def test_timeout_invokes_kill_child_tree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The timeout branch must route through kill_child_tree (force), not
        a direct-child-only ``proc.kill()``."""
        with patch("kimix.tools.common.kill_child_tree") as mock_kill:
            ok, out = await TodoList._run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout=1,
                not_found_hint="n/a",
            )
        assert ok is False
        assert "timed out" in out
        mock_kill.assert_called_once()
        args = mock_kill.call_args.args
        assert isinstance(args[0], int) and args[0] > 0
        assert mock_kill.call_args.kwargs.get("force") is True
