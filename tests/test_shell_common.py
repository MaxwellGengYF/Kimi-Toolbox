"""Tests for kimix.tools.file.bash.shell_common (shared one-shot shell helpers).

The Bash tool, the Powershell tool and the TodoList runner all delegate their
one-shot command building here; these tests pin the shared behavior (fixer
pipeline order, argv/env shapes, PowerShell wrapper) so refactors of the
individual call sites cannot silently diverge.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

from kimix.tools.file.bash import shell_common


def _is_powershell_name(path: str) -> bool:
    """True for ``powershell`` or any ``...powershell.exe`` path."""
    return path == "powershell" or path.lower().endswith("powershell.exe")


class TestPrepareBashCommand:
    def test_fixers_applied_in_order(
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
        shell_common.prepare_bash_command("pwd")
        assert seen == ["prepare", "fix"]

    def test_plain_command_unchanged(self) -> None:
        assert shell_common.prepare_bash_command("echo hi") == "echo hi"

    def test_windows_backslash_path_converted(self) -> None:
        result = shell_common.prepare_bash_command(r"cat src\a.py")
        if sys.platform == "win32":
            assert "src/a.py" in result
        else:
            assert result == r"cat src\a.py"


class TestBashArgv:
    def test_login_argv_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import bash_tool

        monkeypatch.setattr(bash_tool, "find_bash", lambda: "/bin/bash")
        argv, env = shell_common.bash_argv("echo hi", login=True)
        assert argv == ["/bin/bash", "-l", "-c", "echo hi"]
        assert isinstance(env, dict)

    def test_non_login_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import bash_tool

        monkeypatch.setattr(bash_tool, "find_bash", lambda: "/bin/bash")
        argv, _env = shell_common.bash_argv("echo hi", login=False)
        assert argv == ["/bin/bash", "-c", "echo hi"]

    def test_fallback_to_bash_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import bash_tool

        monkeypatch.setattr(bash_tool, "find_bash", lambda: None)
        argv, env = shell_common.bash_argv("echo hi")
        assert argv[0] == "bash"
        assert isinstance(env, dict)


class TestBashFileArgv:
    def test_argv_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import bash_tool

        monkeypatch.setattr(bash_tool, "find_bash", lambda: "/bin/bash")
        argv, env = shell_common.bash_file_argv("/tmp/run.sh")
        assert argv == ["/bin/bash", "-l", "/tmp/run.sh"]
        assert isinstance(env, dict)


class TestWrapPwshCommand:
    def test_wrapper_contents(self) -> None:
        wrapped = shell_common.wrap_pwsh_command("Get-Location")
        assert wrapped.startswith("[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;")
        assert "try{Get-Location}catch{$_|Out-String|Write-Error;exit 1}" in wrapped
        assert wrapped.endswith(";exit $LASTEXITCODE")


class TestPwshArgv:
    def test_pwsh7_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: r"C:\pwsh.exe")
        argv, hint = shell_common.pwsh_argv("Write-Output hi")  # type: ignore[misc]
        assert argv[0] == r"C:\pwsh.exe"
        assert argv[1:7] == ["-NoP", "-NonI", "-Exec", "Bypass", "-NoL", "-Command"]
        assert "try{Write-Output hi}catch" in argv[7]
        assert hint == r"PowerShell executable not found: C:\pwsh.exe"

    def test_invalid_command_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import pwsh_fix

        monkeypatch.setattr(pwsh_fix, "fix_pwsh_command", lambda cmd: None)
        assert shell_common.pwsh_argv("whatever") is None

    def test_ps51_fallback_transform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kimix.tools.file.bash import process_pwsh, pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: None)
        monkeypatch.setattr(
            process_pwsh, "pwsh_transform", lambda cmd: ("transformed", ["warn"])
        )
        argv, hint = shell_common.pwsh_argv("Write-Output hi")  # type: ignore[misc]
        assert "transformed" in argv[7]
        assert _is_powershell_name(argv[0])
        assert hint == f"PowerShell executable not found: {argv[0]}"


class TestPwshFileArgv:
    def test_pwsh7_file_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: r"C:\pwsh.exe")
        argv, hint = shell_common.pwsh_file_argv(r"C:\scripts\run.ps1")
        assert argv == [
            r"C:\pwsh.exe",
            "-NoP", "-NonI", "-Exec", "Bypass", "-NoL", "-File",
            r"C:\scripts\run.ps1",
        ]
        assert hint == r"PowerShell executable not found: C:\pwsh.exe"

    def test_fallback_file_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: None)
        argv, _hint = shell_common.pwsh_file_argv("run.ps1")
        assert _is_powershell_name(argv[0])
        assert argv[1:7] == ["-NoP", "-NonI", "-Exec", "Bypass", "-NoL", "-File"]
        assert argv[7] == "run.ps1"


class TestPwshExecutable:
    def test_returns_pwsh7(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: r"C:\pwsh.exe")
        assert shell_common.pwsh_executable() == r"C:\pwsh.exe"

    def test_returns_ps51_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kimix.tools.file.bash import pwsh_tool

        monkeypatch.setattr(pwsh_tool, "find_pwsh", lambda: None)
        exe = shell_common.pwsh_executable()
        assert _is_powershell_name(exe)
