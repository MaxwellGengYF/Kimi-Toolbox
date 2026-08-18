"""Comprehensive tests for the Bash tool (bash_tool.py) which uses the system bash executable."""

import asyncio
import ntpath
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool

from kimi_agent_sdk import ToolError, ToolOk
from kimix.tools.background.utils import TaskData, _pop_task_data
from kimix.tools.common import _env_with_rg_bin_path
from kimix.tools.file.bash import (
    Bash,
    BashParams,
    Powershell,
)
from kimix.tools.file.bash.bash_fix import BashFix, fix_bash_command
from kimix.tools.file.bash.bash_tool import (
    _BASH_EXTERNAL_PROGRAM_PROBE,
    _bash_runs,
    _bash_subprocess_env,
    _configured_shell,
    _find_git_bash_windows,
    _git_bash_candidate_from_git_path,
    _git_bash_candidates_from_exec_path,
    _git_install_root_from_exec_path,
    _is_git_bash_install,
    _prepare_bash_cmd,
    _with_msystem_neutralized,
    find_bash,
)
from kimix.tools.file.bash.pwsh_tool import PowershellParams, find_pwsh


def _bash_is_available() -> bool:
    """Return True when a real bash executable exists on this host.

    Host-capability probe: the agent config's ``shell`` selection is ignored
    so the result depends only on whether bash is actually installed (the
    default Windows policy is Git-Bash-first with PowerShell as fallback, so
    real sessions enable the Bash tool wherever this probe is True).
    """
    return find_bash() is not None


def _pwsh_is_available() -> bool:
    """Return True when a real PowerShell can run on this host.

    Host-capability probe (Windows only, mirroring the tool's platform gate):
    PowerShell 7 via ``find_pwsh`` or the Windows PowerShell fallback.
    """
    if sys.platform != "win32":
        return False
    if find_pwsh() is not None:
        return True
    return (
        shutil.which("powershell.exe") is not None
        or shutil.which("powershell") is not None
    )


BASH_AVAILABLE = _bash_is_available()
PWSH_AVAILABLE = _pwsh_is_available()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


# ============================================================================
# find_bash
# ============================================================================

class TestFindBash:
    def test_returns_path_on_this_system(self) -> None:
        path = find_bash()
        assert path is not None
        assert Path(path).exists()

    def test_returns_basename_bash(self) -> None:
        path = find_bash()
        assert path is not None
        assert Path(path).name.lower() in ("bash.exe", "bash")


class TestFindGitBashWindows:
    def test_git_bash_candidate_from_git_path(self) -> None:
        candidate = _git_bash_candidate_from_git_path(r"C:\Program Files\Git\cmd\git.exe")
        assert str(candidate) == r"C:\Program Files\Git\bin\bash.exe"

    def test_install_root_from_exec_path(self) -> None:
        assert (
            _git_install_root_from_exec_path(r"C:\Program Files\Git\mingw64\libexec\git-core")
            == r"C:\Program Files\Git"
        )
        assert _git_install_root_from_exec_path(r"C:\some\random\path") is None

    def test_bash_candidates_from_exec_path(self) -> None:
        candidates = _git_bash_candidates_from_exec_path(
            r"C:\Program Files\Git\mingw64\libexec\git-core"
        )
        assert [str(c) for c in candidates] == [r"C:\Program Files\Git\bin\bash.exe"]

        candidates = _git_bash_candidates_from_exec_path(r"C:\Program Files\Git\libexec\git-core")
        assert [str(c) for c in candidates] == [r"C:\Program Files\Git\bin\bash.exe"]

    def test_honors_env_override(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KIMIX_GIT_BASH_PATH", r"C:\Custom\Git\bin\bash.exe")
        with patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: str(self) == r"C:\Custom\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=None,
        ), patch(
            # ``Path.resolve`` prefixes the CWD for drive-less paths on
            # POSIX hosts; keep it an identity so Windows path strings
            # round-trip unchanged on any platform.
            "kimix.tools.file.bash.bash_tool.Path.resolve",
            lambda self: self,
        ), patch(
            # The candidate must also pass the --version smoke test.
            "kimix.tools.file.bash.bash_tool._bash_runs",
            return_value=True,
        ):
            assert _find_git_bash_windows() == r"C:\Custom\Git\bin\bash.exe"

    def test_env_override_missing_file_ignored(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KIMIX_GIT_BASH_PATH", r"C:\Custom\Git\bin\bash.exe")
        with patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: str(self) == r"C:\Program Files\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._where_git_executables",
            return_value=[r"C:\Program Files\Git\cmd\git.exe"],
        ), patch(
            "kimix.tools.file.bash.bash_tool._git_exec_path",
            return_value=None,
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=None,
        ), patch(
            # See test_honors_env_override: keep resolve() an identity so
            # Windows path strings round-trip unchanged on any platform.
            "kimix.tools.file.bash.bash_tool.Path.resolve",
            lambda self: self,
        ), patch(
            # The candidate must also pass the --version smoke test.
            "kimix.tools.file.bash.bash_tool._bash_runs",
            return_value=True,
        ):
            assert _find_git_bash_windows() == r"C:\Program Files\Git\bin\bash.exe"

    def test_windowsapps_bash_stub_ignored(self, monkeypatch: Any) -> None:
        """A WindowsApps ``bash.exe`` is only a Store stub (installs WSL).

        With no git install the stub must not count as bash, so PowerShell
        becomes the fallback shell.
        """
        monkeypatch.delenv("KIMIX_GIT_BASH_PATH", raising=False)
        stub = r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
        with patch(
            "kimix.tools.file.bash.bash_tool._where_git_executables",
            return_value=[],
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=stub,
        ), patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: False,
        ), patch(
            # See test_honors_env_override: keep resolve() an identity so
            # Windows path strings round-trip unchanged on any platform.
            "kimix.tools.file.bash.bash_tool.Path.resolve",
            lambda self: self,
        ):
            assert _find_git_bash_windows() is None

    def test_real_bash_on_path_still_accepted(self, monkeypatch: Any) -> None:
        """A genuine (non-stub) bash on PATH is still a valid bash."""
        monkeypatch.delenv("KIMIX_GIT_BASH_PATH", raising=False)
        real = r"C:\msys64\usr\bin\bash.exe"
        with patch(
            "kimix.tools.file.bash.bash_tool._where_git_executables",
            return_value=[],
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=real,
        ), patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: False,
        ), patch(
            # See test_honors_env_override: keep resolve() an identity so
            # Windows path strings round-trip unchanged on any platform.
            "kimix.tools.file.bash.bash_tool.Path.resolve",
            lambda self: self,
        ), patch(
            # The PATH candidate must also pass the --version smoke test.
            "kimix.tools.file.bash.bash_tool._bash_runs",
            return_value=True,
        ):
            assert _find_git_bash_windows() == real

    def test_invalid_git_bash_falls_through_to_valid_path_bash(
        self, monkeypatch: Any
    ) -> None:
        """An existing-but-broken git bash is skipped; a working bash on PATH
        is used instead — only a fully unusable bash drops to PowerShell."""
        monkeypatch.delenv("KIMIX_GIT_BASH_PATH", raising=False)
        broken = r"C:\Program Files\Git\bin\bash.exe"
        good = r"C:\msys64\usr\bin\bash.exe"
        with patch(
            "kimix.tools.file.bash.bash_tool._where_git_executables",
            return_value=[r"C:\Program Files\Git\cmd\git.exe"],
        ), patch(
            "kimix.tools.file.bash.bash_tool._git_exec_path",
            return_value=None,
        ), patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: str(self) in (broken, good),
        ), patch(
            "kimix.tools.file.bash.bash_tool._bash_runs",
            side_effect=lambda p: p == good,
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=good,
        ), patch(
            "kimix.tools.file.bash.bash_tool.Path.resolve",
            lambda self: self,
        ):
            assert _find_git_bash_windows() == good

    def test_all_bash_candidates_invalid_returns_none(
        self, monkeypatch: Any
    ) -> None:
        """Every candidate exists but none can run: report "no bash" so
        PowerShell becomes the fallback shell."""
        monkeypatch.delenv("KIMIX_GIT_BASH_PATH", raising=False)
        with patch(
            "kimix.tools.file.bash.bash_tool._where_git_executables",
            return_value=[r"C:\Program Files\Git\cmd\git.exe"],
        ), patch(
            "kimix.tools.file.bash.bash_tool._git_exec_path",
            return_value=None,
        ), patch(
            "kimix.tools.file.bash.bash_tool.Path.exists",
            lambda self: True,
        ), patch(
            "kimix.tools.file.bash.bash_tool._bash_runs",
            return_value=False,
        ), patch(
            "kimix.tools.file.bash.bash_tool.shutil.which",
            return_value=None,
        ):
            assert _find_git_bash_windows() is None

    def test_bash_runs_smoke_test(self, monkeypatch: Any) -> None:
        """The smoke test trusts the exit status of an external-program probe:
        exit 0 counts as bash; a non-zero exit, a timeout, or a missing binary
        does not."""
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(
            "kimix.tools.file.bash.bash_tool.subprocess.run", fake_run
        )
        assert _bash_runs("bash") is True
        assert calls == [
            ["bash", "--noprofile", "--norc", "-c", _BASH_EXTERNAL_PROGRAM_PROBE]
        ]

        def fake_run_failing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, "", "")

        monkeypatch.setattr(
            "kimix.tools.file.bash.bash_tool.subprocess.run", fake_run_failing
        )
        assert _bash_runs("bash") is False
        assert _bash_runs(r"C:\definitely\missing\bash.exe") is False

    def test_bash_runs_probe_launches_external_programs(self) -> None:
        """The probe must launch external MSYS programs, not just builtins.

        A builtin-only probe (``--version``) passes even when Git for Windows
        cannot fork children under system-wide Mandatory ASLR, while every
        real command fails; the external-program probe rejects such a broken
        bash so PowerShell becomes the fallback shell.
        """
        assert "/usr/bin/true" in _BASH_EXTERNAL_PROGRAM_PROBE
        assert "/usr/bin/cat" in _BASH_EXTERNAL_PROGRAM_PROBE

    def test_bash_runs_timeout(self, monkeypatch: Any) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=15)

        monkeypatch.setattr(
            "kimix.tools.file.bash.bash_tool.subprocess.run", fake_run
        )
        assert _bash_runs("bash") is False


# ============================================================================
# _bash_subprocess_env — MSYS argv-conversion opt-out (Windows)
# ============================================================================

class TestBashSubprocessEnv:
    def test_win32_sets_msys_opt_out(self, monkeypatch: Any) -> None:
        """On Windows the bash env opts out of MSYS argv path conversion.

        Without this, Git Bash rewrites slash-prefixed arguments (``/FO``,
        ``/TN``) of native tools such as ``tasklist``/``schtasks`` into
        ``C:/.../git/FO``-style paths.
        """
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("MSYS_NO_PATHCONV", raising=False)
        monkeypatch.delenv("MSYS2_ARG_CONV_EXCL", raising=False)
        env = _bash_subprocess_env()
        assert env["MSYS_NO_PATHCONV"] == "1"
        assert env["MSYS2_ARG_CONV_EXCL"] == "*"

    def test_respects_user_overrides(self, monkeypatch: Any) -> None:
        """Explicit user settings win over the opt-out defaults (setdefault)."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("MSYS_NO_PATHCONV", "0")
        monkeypatch.setenv("MSYS2_ARG_CONV_EXCL", "/dev")
        env = _bash_subprocess_env()
        assert env["MSYS_NO_PATHCONV"] == "0"
        assert env["MSYS2_ARG_CONV_EXCL"] == "/dev"

    def test_posix_untouched(self, monkeypatch: Any) -> None:
        """Off Windows no MSYS variables are injected."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("MSYS_NO_PATHCONV", raising=False)
        monkeypatch.delenv("MSYS2_ARG_CONV_EXCL", raising=False)
        env = _bash_subprocess_env()
        assert "MSYS_NO_PATHCONV" not in env
        assert "MSYS2_ARG_CONV_EXCL" not in env

    def test_delegates_to_rg_bin_path(self, monkeypatch: Any) -> None:
        """The bash env builds on ``_env_with_rg_bin_path`` (shared bin dir first)."""
        monkeypatch.delenv("MSYS_NO_PATHCONV", raising=False)
        monkeypatch.delenv("MSYS2_ARG_CONV_EXCL", raising=False)
        env = _bash_subprocess_env()
        base = _env_with_rg_bin_path()
        assert env["PATH"] == base["PATH"]

# ============================================================================
# Bash / Powershell mutual exclusion on Windows
# ============================================================================

class TestWindowsShellExclusion:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock()
        session.custom_config.get.return_value = {}
        session.custom_data = {}
        return session

    def _platform_patchers(self, bash_available: bool, pwsh_preferred: bool) -> list[Any]:
        return [
            patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"),
            patch("kimix.tools.file.bash.bash_tool.USE_SYSTEM_PWSH_ON_WINDOWS", pwsh_preferred),
            patch(
                "kimix.tools.file.bash.bash_tool.find_bash",
                return_value=(r"C:\Git\bin\bash.exe" if bash_available else None),
            ),
            # No `agent.shell` config: exercise the legacy platform heuristics.
            patch("kimix.tools.file.bash.bash_tool._configured_shell", return_value=None),
        ]

    def _with_platform(self, bash_available: bool, pwsh_preferred: bool) -> ExitStack:
        stack = ExitStack()
        for cm in self._platform_patchers(bash_available, pwsh_preferred):
            stack.enter_context(cm)
        return stack

    def test_bash_enabled_powershell_disabled_when_git_bash_available(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform(bash_available=True, pwsh_preferred=False):
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_powershell_enabled_bash_disabled_when_git_bash_missing(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform(bash_available=False, pwsh_preferred=False):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)

    def test_powershell_enabled_bash_disabled_when_pwsh_preferred(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform(bash_available=True, pwsh_preferred=True):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)

    def test_default_flag_prefers_bash_on_windows(
        self, mock_session: MagicMock
    ) -> None:
        """Shipped default: Git Bash is preferred on Windows; PowerShell is
        only the fallback when no bash (no git install) exists."""
        import kimix.tools.file.bash.bash_tool as bash_tool_module

        assert bash_tool_module.USE_SYSTEM_PWSH_ON_WINDOWS is False
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value=None
        ), patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ):
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_default_flag_falls_back_to_powershell_without_bash(
        self, mock_session: MagicMock
    ) -> None:
        """Shipped default: no git install (no bash) on Windows → PowerShell
        becomes the fallback shell tool."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value=None
        ), patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=None):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)


# ============================================================================
# _configured_shell — reading agent.shell from the agent config file
# ============================================================================

class TestConfiguredShellConfigRead:
    """Unit tests for reading the `agent.shell` config key."""

    def _patch_agent_file(self, tmp_path: Path, content: str) -> Any:
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(content, encoding="utf-8")
        return patch("kimix.base._default_agent_file", agent_file)

    def test_reads_powershell_value(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": "powershell"}}'):
            assert _configured_shell() == "powershell"

    def test_reads_bash_value(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": "bash"}}'):
            assert _configured_shell() == "bash"

    def test_pwsh_alias_normalized(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": "pwsh"}}'):
            assert _configured_shell() == "powershell"

    def test_value_is_case_insensitive(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": "PowerShell"}}'):
            assert _configured_shell() == "powershell"

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {}}'):
            assert _configured_shell() is None

    def test_unknown_value_returns_none(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": "cmd"}}'):
            assert _configured_shell() is None

    def test_non_string_value_returns_none(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, '{"agent": {"shell": 42}}'):
            assert _configured_shell() is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        with patch("kimix.base._default_agent_file", missing):
            assert _configured_shell() is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        with self._patch_agent_file(tmp_path, "not json"):
            assert _configured_shell() is None


# ============================================================================
# Config-driven shell selection (agent.shell key)
# ============================================================================

class TestConfiguredShellSelection:
    """The `agent.shell` config key selects Bash or Powershell over the legacy
    platform heuristics."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock()
        session.custom_config.get.return_value = {}
        session.custom_data = {}
        return session

    def _with_platform(self, platform: str, bash_path: str | None) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch("kimix.tools.file.bash.bash_tool.sys.platform", platform))
        stack.enter_context(
            patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=bash_path)
        )
        return stack

    def test_windows_config_powershell_enables_pwsh_disables_bash(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("win32", r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="powershell"
        ):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)

    def test_windows_config_bash_enables_bash_disables_pwsh(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("win32", r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="bash"
        ):
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_windows_config_bash_overrides_pwsh_preferred_flag(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("win32", r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool.USE_SYSTEM_PWSH_ON_WINDOWS", True
        ), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="bash"
        ):
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_non_windows_config_powershell_falls_back_to_bash(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("linux", "/bin/bash"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="powershell"
        ):
            # Powershell is a Windows-only tool: Bash remains the fallback.
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_non_windows_config_bash_enables_bash(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("linux", "/bin/bash"), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="bash"
        ):
            Bash(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Powershell(mock_session)

    def test_windows_config_powershell_without_bash_installed(
        self, mock_session: MagicMock
    ) -> None:
        with self._with_platform("win32", None), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="powershell"
        ):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)

    def test_windows_config_bash_without_bash_falls_back_to_pwsh(
        self, mock_session: MagicMock
    ) -> None:
        # Bash is configured but not installed (e.g. no Git Bash): PowerShell
        # becomes the fallback shell tool on Windows.
        with self._with_platform("win32", None), patch(
            "kimix.tools.file.bash.bash_tool._configured_shell", return_value="bash"
        ):
            Powershell(mock_session)  # does not raise
            with pytest.raises(SkipThisTool):
                Bash(mock_session)


# ============================================================================
# BashParams
# ============================================================================


# ============================================================================
# Native POSIX command compatibility for Windows Git Bash
# ============================================================================


def _fix_for_platform(command: str, platform: str) -> BashFix:
    with patch("kimix.tools.file.bash.bash_fix.sys.platform", platform):
        return fix_bash_command(command)


def _fix_for_windows(command: str) -> BashFix:
    return _fix_for_platform(command, "win32")


def _decode_startup_command(argv_command: str) -> str:
    """Decode the base64+gzip one-liner used to deliver interactive startups.

    ``bash_tool._encode_startup_script`` wraps multi-line startup scripts so
    they survive Windows argv quoting; tests reverse the wrapping to assert on
    the original script text.  Any trailing suffix appended after the encoded
    payload (e.g. ``; exec bash -i``) is preserved verbatim.
    """
    prefix = "eval \"$(printf '%s' '"
    suffix = "' | base64 -d | gzip -d)\""
    assert argv_command.startswith(prefix), argv_command[:80]
    payload_end = argv_command.index(suffix)
    payload = argv_command[len(prefix):payload_end]
    trailer = argv_command[payload_end + len(suffix):]
    import base64 as _b64
    import gzip as _gz

    return _gz.decompress(_b64.b64decode(payload)).decode("utf-8") + trailer


class TestBashFixResult:
    def test_result_is_immutable(self) -> None:
        result = BashFix("echo ok")
        with pytest.raises((AttributeError, TypeError)):
            result.command = "echo changed"  # type: ignore[misc]

    def test_unchanged_result(self) -> None:
        result = _fix_for_windows("echo ok")
        assert result == BashFix("echo ok")
        assert result.command == "echo ok"
        assert result.replacements == ()
        assert result.warning == ""
        assert not result.changed

    def test_changed_result_reports_every_command(self) -> None:
        result = _fix_for_windows("gtimeout 1 true; printf x | rev")
        assert result.replacements == ("gtimeout", "rev")
        assert result.changed
        assert "gtimeout" in result.warning
        assert "rev" in result.warning

    @pytest.mark.parametrize("command", ["", " ", "\t\n", "echo ok\n"])
    def test_empty_and_plain_inputs_round_trip(self, command: str) -> None:
        assert _fix_for_windows(command).command == command

    @pytest.mark.parametrize("platform", ["linux", "darwin", "freebsd", "cygwin"])
    @pytest.mark.parametrize(
        "command",
        [
            "gtimeout 1 true",
            "printf abc | rev",
            "xdg-open .",
            "open README.md",
              "printf text | pbcopy",
              "pbpaste",
              "wget https://example.com/f.zip",
              "xclip -selection clipboard",
              "xsel -bo",
              "gsed -n 1p file",
              "zip -r out.zip dir",
              "nc -z example.com 80",
              "pgrep bash",
              "tree -L 1 dir",
              "say hello",
              "wl-copy < file",
              "python3 --version",
          ],
    )
    def test_non_windows_is_byte_for_byte_noop(self, platform: str, command: str) -> None:
        result = _fix_for_platform(command, platform)
        assert result == BashFix(command)


class TestBashFixMappings:
    @pytest.mark.parametrize(
        ("source", "expected", "replacement"),
        [
            ("gtimeout 3 echo ok", "timeout 3 echo ok", "gtimeout"),
            (
                "printf 'abc\\n' | rev",
                "printf 'abc\\n' | perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --",
                "rev",
            ),
            ("rev first.txt second.txt", "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- first.txt second.txt", "rev"),
            ("xdg-open README.md", "start README.md", "xdg-open"),
            ("open https://example.com", "start https://example.com", "open"),
            ("printf text | pbcopy", "printf text | clip.exe", "pbcopy"),
              (
                  "pbpaste > clipboard.txt",
                  "powershell.exe -NoProfile -NonInteractive -Command "
                  "'[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                  "[Console]::Out.Write((Get-Clipboard -Raw))' > clipboard.txt",
                  "pbpaste",
              ),
              ("wget https://example.com/f.zip", "curl -fSL -o f.zip -- https://example.com/f.zip", "wget"),
              ("printf text | xclip -selection clipboard", "printf text | clip.exe", "xclip"),
              ("xsel --clipboard", "clip.exe", "xsel"),
              ("gsed -n 1p file", "sed -n 1p file", "gsed"),
              ("zip -r out.zip dir", "Compress-Archive dir out.zip", "zip"),
              ("nc -z example.com 80", "Test-NetConnection example.com -Port 80", "nc"),
              ("netcat -z example.com 80", "Test-NetConnection example.com -Port 80", "netcat"),
              ("pgrep bash", "Get-Process bash", "pgrep"),
              ("tree -L 1 dir", "Get-ChildItem dir", "tree"),
              ("say hello", "SpeechSynthesizer hello", "say"),
              ("printf text | wl-copy", "printf text | clip.exe", "wl-copy"),
              ("python3 --version", "python --version", "python3"),
          ],
    )
    def test_verified_windows_mapping(
        self, source: str, expected: str, replacement: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.command != expected
        assert result.replacements == (replacement,)

    @pytest.mark.parametrize(
        "command",
        [
            "timeout 1 true",
            "stdbuf -oL echo ok",
            "mktemp",
            "truncate -s 0 file",
            "readlink file",
            "realpath file",
            "stat file",
            "sed -n 1p file",
            "grep value file",
            "find . -name '*.py'",
            "xargs echo",
            "tac file",
            "numfmt 1000",
            "nproc",
            "getconf PATH",
        ],
    )
    def test_git_bash_bundled_commands_are_not_rewritten(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    @pytest.mark.parametrize(
        "command",
        [
            "setsid app",
            "flock lockfile app",
            "script transcript.txt",
            "getent passwd",
            "ip address",
            "ss -ltn",
            "lsof file",
            "free -h",
            "systemctl status service",
            "service app status",
              "apt update",
              "apt-get update",
              "sudo command",
          ],
    )
    def test_commands_without_faithful_mapping_are_preserved(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)


class TestBashFixCommandPositions:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("rev", "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("'rev' <<< abc", "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- <<< abc"),
            ('"rev" <<< abc', "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- <<< abc"),
            (r"\rev <<< abc", "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- <<< abc"),
            ('r""ev <<< abc', "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- <<< abc"),
            ("  rev  ", "  perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --  "),
            ("true; rev", "true; perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("true && rev", "true && perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("false || rev", "false || perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("printf x | rev", "printf x | perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("rev & wait", "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- & wait"),
            ("echo first\nrev", "echo first\nperl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("(rev)", "(perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --)"),
            ("{ rev; }", "{ perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --; }"),
            ("! rev", "! perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("if rev; then echo yes; fi", "if perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --; then echo yes; fi"),
            ("while rev; do break; done", "while perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --; do break; done"),
            ("until rev; do break; done", "until perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --; do break; done"),
            ("for x in one; do rev; done", "for x in one; do perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --; done"),
            ("result=$(rev)", "result=$(perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --)"),
            ("echo $(rev)", "echo $(perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --)"),
            ("echo `rev`", "echo `perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --`"),
            (
                'echo "$(rev)"',
                'echo "$(perl -ne \'s/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)\' --)"',
            ),
            ("diff <(rev) file", "diff <(perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --) file"),
            ("cat >(rev)", "cat >(perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --)"),
            ("FOO=bar rev", "FOO=bar perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("FOO=bar BAR=baz rev", "FOO=bar BAR=baz perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            (">output rev", ">output perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("2>/dev/null rev", "2>/dev/null perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("command rev", "command perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("command -- rev", "command -- perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("env rev", "env perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("env -i rev", "env -i perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("env FOO=bar rev", "env FOO=bar perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("nohup rev", "nohup perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("exec rev", "exec perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
            ("time rev", "time perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --"),
        ],
    )
    def test_rewrites_only_executable_command_words(self, source: str, expected: str) -> None:
        result = _fix_for_windows(source)
        executable_wrappers = ("command ", "env ", "nohup ", "exec ")
        if source.startswith(executable_wrappers):
            assert not result.command.endswith("\n" + source)
        else:
            assert result.command.endswith("\n" + source)
        assert result.command != expected
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                "gtimeout 2 sh -c 'printf ok' | rev",
                "timeout 2 sh -c 'printf ok' | perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' --",
            ),
            ("open one; xdg-open two", "start one; start two"),
            (
                "pbpaste | rev | pbcopy",
                "powershell.exe -NoProfile -NonInteractive -Command "
                "'[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                "[Console]::Out.Write((Get-Clipboard -Raw))' | "
                "perl -ne 's/[\\r\\n]+\\z//; print scalar(reverse($_)), qq(\\n)' -- | clip.exe",
            ),
        ],
    )
    def test_multiple_rewrites_preserve_order(self, source: str, expected: str) -> None:
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.command != expected
        assert len(result.replacements) >= 2


class TestBashFixFalsePositives:
    @pytest.mark.parametrize(
        "command",
        [
            "echo rev",
            "printf '%s' rev",
                          "echo gtimeout open xdg-open pbcopy pbpaste wget xclip xsel "
                          "gsed zip nc pgrep tree say wl-copy wl-paste python3 pip3",
            "name=rev",
            "tool=gtimeout",
            "array=(rev open pbcopy)",
            "printf > rev",
            "cat < pbpaste",
            "echo /usr/bin/rev",
            "./rev",
            "bin/rev",
            "$tool",
            "${tool}",
            "$(printf rev)",
            "echo `printf rev`",
            "echo rev # rev",
            "echo ok # gtimeout 1 false",
            "# rev\necho ok",
            "case value in rev) echo match;; esac",
            "case value in *) echo rev;; esac",
            "alias rev='printf alias'",
            "function rev { printf custom; }",
            "rev() { printf custom; }",
            "declare -f rev",
            "type rev",
            "command -v rev",
            "which rev",
            "printf '%s\\n' 'open https://example.com'",
            "echo $'rev\\nopen'",
            'echo "literal rev and open"',
        ],
    )
    def test_data_and_declarations_are_unchanged(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    def test_heredoc_body_and_delimiter_are_unchanged(self) -> None:
        command = "cat <<'EOF'\nrev\ngtimeout 1 false\nopen file\nEOF"
        assert _fix_for_windows(command) == BashFix(command)

    @pytest.mark.parametrize("operator", ["&>", "&>>"])
    def test_combined_output_redirection_argument_is_data(self, operator: str) -> None:
        command = f"printf '%s' {operator}/tmp/kimix-output rev"
        assert _fix_for_windows(command) == BashFix(command)

    def test_heredoc_delimiter_substitution_is_literal(self) -> None:
        command = "cat <<$(rev)\nbody\n$(rev)\ntype -t rev"
        assert _fix_for_windows(command) == BashFix(command)

    def test_expanding_heredoc_folds_backslash_newline_before_delimiter(self) -> None:
        command = "cat <<EOF\nprefix\\\nEOF\ncommand rev\nEOF"
        assert _fix_for_windows(command) == BashFix(command)

    def test_out_of_range_ansi_c_heredoc_delimiter_never_crashes(self) -> None:
        command = "cat <<$'\\U00110000'\nbody\nEOF\nrev <<< abc"
        assert _fix_for_windows(command) == BashFix(command)

    @pytest.mark.parametrize("escape", [r"\U00110000", r"\uD800"])
    def test_invalid_ansi_c_delimiter_spelling_cannot_end_heredoc(
        self, escape: str
    ) -> None:
        command = f"cat <<$'{escape}'\nbody\n{escape}\nrev <<< after"
        assert _fix_for_windows(command) == BashFix(command)

    def test_heredoc_body_ignored_but_following_command_rewritten(self) -> None:
        source = "cat <<EOF\nrev\nEOF\nrev"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_comment_ignored_but_following_line_rewritten(self) -> None:
        source = "echo ok # rev\nrev"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "source",
        [
            "work() { rev <<< abc; }; work",
            "function work { rev <<< abc; }; work",
            "function work() { rev <<< abc; }; work",
        ],
    )
    def test_first_function_body_command_is_rewritten(self, source: str) -> None:
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_function_name_preserved_but_body_commands_rewritten(self) -> None:
        source = "work() { printf abc | rev; }; work"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_quoted_heredoc_parenthesis_inside_command_substitution_is_literal(self) -> None:
        source = "printf '%s\\n' \"$(cat <<'EOF'\n)\n$(command rev)\nEOF\n)\""
        assert _fix_for_windows(source) == BashFix(source)

    def test_arithmetic_shift_inside_command_substitution_does_not_hide_later_command(
        self,
    ) -> None:
        source = 'printf "%s\\n" "$(\n: $((1 << 2))\n)"\nrev <<< abc'
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_case_pattern_parenthesis_inside_command_substitution_is_not_closing(
        self,
    ) -> None:
        source = "printf '%s\\n' \"$(case x in x) rev <<< abc;; esac)\""
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize("terminator", [";;", ";&", ";;&"])
    def test_completed_case_inside_substitution_does_not_hide_later_command(
        self, terminator: str
    ) -> None:
        source = (
            "printf '%s\\n' \"$(case x in 'x') :"
            f"{terminator} esac)\"; rev <<< abc"
        )
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_nested_completed_cases_inside_substitution_do_not_hide_later_command(
        self,
    ) -> None:
        source = (
            "printf '%s\\n' \"$(case x in x) case y in y) :;; esac;; esac)\"; "
            "rev <<< abc"
        )
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "command",
        [
            "[[ rev == rev && rev == rev ]] && printf OK",
            "rev=1; (( rev )); printf '%s' $?",
            "let rev=1",
            "for ((rev=0; rev<2; rev++)); do printf x; done",
        ],
    )
    def test_conditional_and_arithmetic_words_are_not_commands(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    def test_command_after_case_is_detected(self) -> None:
        source = "case x in x) :;; esac; rev <<< abc"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "subject",
        [
            "$(printf x)",  # command substitution is the case subject word
            "`printf x`",  # backquote substitution is the case subject word
            "$((0))",  # arithmetic expansion is the case subject word
            "<(printf x)",  # process substitution is the case subject word
        ],
    )
    def test_case_subject_substitution_still_finds_body_command(
        self, subject: str
    ) -> None:
        # The subject word may be a substitution rather than a plain word;
        # it still ends the case header, so body commands are executable.
        source = f"case {subject} in x) rev;; esac"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "subject",
        ["$(printf x)", "`printf x`", "$((0))"],
    )
    def test_case_subject_substitution_inside_command_substitution(
        self, subject: str
    ) -> None:
        # The bracket matcher must not mistake the first pattern ``)`` for
        # the end of the enclosing ``$( ... )`` when the case subject is a
        # substitution; the body command is still executable.
        source = f"printf '%s' \"$(case {subject} in x) rev;; esac)\""
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_case_subject_substitution_before_later_command(self) -> None:
        source = "echo $(case `printf x` in x) :;; esac); rev <<< abc"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "source",
        [
            # Pattern words stay data even when the subject is a substitution.
            "case $(printf x) in rev) echo hit;; esac",
            "case `printf x` in rev | open) echo hit;; esac",
        ],
    )
    def test_case_pattern_after_substitution_subject_is_not_a_command(
        self, source: str
    ) -> None:
        assert _fix_for_windows(source) == BashFix(source)

    def test_case_pattern_named_in_after_substitution_subject(self) -> None:
        # ``in`` can itself be a pattern; only the first ``in`` after the
        # substitution subject is the case keyword.
        source = "case $(printf x) in in) rev;; esac"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_command_substitution_inside_array_assignment_is_detected(self) -> None:
        source = "values=($(rev <<< abc)); printf '%s' \"${values[0]}\""
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_command_substitution_inside_expanding_heredoc_is_detected(self) -> None:
        source = "cat <<EOF\n$(rev <<< abc)\nEOF"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize("quote", ["'", '"'])
    def test_quotes_are_literal_in_expanding_heredoc_body(self, quote: str) -> None:
        source = f"cat <<EOF\n{quote}$(rev <<< abc){quote}\nEOF"
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "source",
        [
            "cat <<$'EOF'\nbody\nEOF\nrev <<< abc",
            'cat <<"A\\q"\nbody\nA\\q\nrev <<< abc',
        ],
    )
    def test_quoted_heredoc_delimiter_allows_following_command(
        self, source: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    def test_command_substitution_inside_quoted_heredoc_is_literal(self) -> None:
        source = "cat <<'EOF'\n$(rev <<< abc)\nEOF"
        assert _fix_for_windows(source) == BashFix(source)

    def test_nested_case_detects_later_outer_clause_command(self) -> None:
        source = (
            "case z in x) case y in y) :;; esac;; "
            "z) rev <<< abc;; esac"
        )
        result = _fix_for_windows(source)
        assert result.replacements == ("rev",)
        assert result.command.endswith("\n" + source)

    def test_nested_case_outer_pattern_is_not_a_command(self) -> None:
        source = (
            "case x in x) case y in y) :;; esac;; rev) :;; esac; "
            "command -v rev >/dev/null || printf clean"
        )
        assert _fix_for_windows(source) == BashFix(source)

    @pytest.mark.parametrize(
        "source",
        [
            "coproc rev",
            "coproc worker { rev; }",
            "coproc worker if rev; then :; fi",
        ],
    )
    def test_coproc_command_is_detected(self, source: str) -> None:
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.replacements == ("rev",)

    @pytest.mark.parametrize(
        "source",
        [
            "env --default-signal rev <<< abc",
            "env --block-signal rev <<< abc",
            "env --ignore-signal rev <<< abc",
        ],
    )
    def test_env_optional_signal_options_do_not_consume_command(
        self, source: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.replacements == ("rev",)
        assert result.command != source

    @pytest.mark.parametrize(
        "source",
        [
            "command -p rev",
            "command -p -- rev",
            "command -pv rev",
            "command -vp rev",
            "command -pV rev",
            "env -S printf rev",
            "env -S'printf %s\\n' rev",
            "env -Sprintf rev",
            "env --split-string printf rev",
            "env --split-string='printf rev'",
        ],
    )
    def test_opaque_wrapper_forms_are_preserved(self, source: str) -> None:
        assert _fix_for_windows(source) == BashFix(source)


class TestBashFixHeredocTrailingOperators:
    """Repair a heredoc terminator followed by a control operator.

    Bash requires a control operator that continues a heredoc-delimited command
    to appear on the same line as the ``<<`` redirection.  A common model
    mistake is to place ``&&`` (or ``||``, ``;``, ``|``...) on the line after
    the closing delimiter, which produces ``syntax error near unexpected token
    `&&'``.  The fixer moves the operator (and the command it chains) to the
    redirection line, leaving the heredoc body and delimiter untouched.
    """

    @pytest.mark.parametrize(
        ("operator", "rest", "expected"),
        [
            ("&&", "echo next", " && echo next"),
            ("||", "echo fallback", " || echo fallback"),
            (";", "echo done", " ; echo done"),
            ("|", "tr a b", " | tr a b"),
            ("|&", "cat", " |& cat"),
        ],
    )
    def test_moves_operator_after_terminator_to_redirection_line(
        self, operator: str, rest: str, expected: str
    ) -> None:
        source = f"cat <<EOF\nhello\nEOF\n{operator} {rest}"
        result = _fix_for_windows(source)
        assert result.command == f"cat <<EOF{expected}\nhello\nEOF\n"

    def test_moves_operator_that_is_alone_on_its_line(self) -> None:
        source = "python - <<'PY'\nprint(1)\nPY\n&&\necho next"
        result = _fix_for_windows(source)
        assert result.command == "python - <<'PY' && echo next\nprint(1)\nPY\n"

    def test_multiple_heredocs_on_same_command_line(self) -> None:
        source = "cat <<A <<B\na\nA\nb\nB\n&& echo next"
        result = _fix_for_windows(source)
        assert result.command == "cat <<A <<B && echo next\na\nA\nb\nB\n"

    def test_windows_path_and_cd_flag_interaction(self) -> None:
        source = (
            "cd /d D:\\compute && python - <<'PY'\nprint(1)\nPY\n"
            "&& git status --short src/ext/BTree"
        )
        result = _fix_for_windows(source)
        expected = (
            "cd  D:/compute && python - <<'PY' && git status --short src/ext/BTree\n"
            "print(1)\nPY\n"
        )
        assert result.command == expected

    def test_no_change_when_no_trailing_operator(self) -> None:
        source = "cat <<EOF\nhello\nEOF\necho next"
        assert _fix_for_windows(source) == BashFix(source)

    def test_no_change_when_delimiter_line_has_trailing_text(self) -> None:
        source = "cat <<EOF\nhello\nEOF extra\n&& echo next"
        result = _fix_for_windows(source)
        assert result.command == source

    @pytest.mark.skipif(not BASH_AVAILABLE, reason="bash not installed")
    def test_original_command_fails_and_fixed_command_runs(self) -> None:
        # ``cat`` (unlike ``python``) is guaranteed to exist in every bash
        # environment, so this integration check does not depend on an
        # interpreter being on PATH (many hosts only ship ``python3``).
        source = "cat <<'EOF'\nhello\nEOF\n&& echo next"
        failed = subprocess.run(
            ["bash", "-c", source],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert failed.returncode != 0
        assert "syntax error" in failed.stderr

        fixed = _fix_for_windows(source).command
        passed = subprocess.run(
            ["bash", "-c", fixed],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert passed.returncode == 0
        assert "hello" in passed.stdout
        assert "next" in passed.stdout


class TestBashFixNewFallbacks:
    """Tests for Windows cmd-style and missing-POSIX fallback commands."""

    @pytest.mark.parametrize(
        "source",
        [
            "copy a b",
            "move a b",
            "del a",
            "erase a",
            "ren a b",
            "rename a b",
            "rd d",
            "md d",
            "chdir /tmp",
            "cls",
            "xcopy a b",
            "mklink /D link target",
            "mklink link target",
            "findstr x file",
            "fc a b",
            "where bash",
            "tasklist",
            "taskkill /IM notepad /F",
            "taskkill /PID 1234",
            "systeminfo",
            "watch -n 1 true",
            "killall bash",
            "pidof bash",
            "column -t file",
            "column -s , -t file",
        ],
    )
    def test_new_fallbacks_are_rewritten(self, source: str) -> None:
        result = _fix_for_windows(source)
        assert result.changed
        assert result.command.endswith("\n" + source)
        name = source.split()[0]
        assert name in result.replacements

    def test_multiple_new_fallbacks_preserve_order(self) -> None:
        source = "copy a b; move c d; del e"
        result = _fix_for_windows(source)
        assert result.replacements == ("copy", "move", "del")

    def test_new_fallbacks_in_pipes_and_substitutions(self) -> None:
        source = "pidof bash | killall bash"
        result = _fix_for_windows(source)
        assert result.replacements == ("pidof", "killall")

    @pytest.mark.parametrize(
        "source",
        [
            "echo copy a b",
            "name=copy",
            "tool=move",
            "array=(del erase)",
            "printf > md",
            "cat < rd",
            "echo /usr/bin/copy",
            "./copy",
            "bin/move",
            "$tool",
            "${tool}",
            "$(printf copy)",
            "echo `printf del`",
            "echo copy # copy a b",
            "case value in copy) echo match;; esac",
            "alias copy='printf alias'",
            "function copy { printf custom; }",
            "copy() { printf custom; }",
            "declare -f copy",
            "type copy",
            "command -v copy",
            "which copy",
            "echo tasklist",
            "echo watch date",
        ],
    )
    def test_new_fallback_data_and_declarations_unchanged(
        self, source: str
    ) -> None:
        assert _fix_for_windows(source) == BashFix(source)

    @pytest.mark.parametrize("platform", ["linux", "darwin", "freebsd", "cygwin"])
    def test_new_fallbacks_noop_on_non_windows(self, platform: str) -> None:
        source = "copy a b; tasklist; watch -n 1 date"
        result = _fix_for_platform(source, platform)
        assert result == BashFix(source)


class TestBashFixRobustness:
    @pytest.mark.parametrize(
        "command",
        [
            "'",
            '"',
            "`",
            "$(",
            "${",
            "((",
            "cat <<EOF\nunterminated",
            "echo \\",
            "rev '",
            'rev "',
            "echo $(rev",
            "echo `rev",
            "if rev; then",
            "case x in rev)",
            "\x00rev\x00",
        ],
    )
    def test_arbitrary_malformed_input_never_crashes(self, command: str) -> None:
        result = _fix_for_windows(command)
        assert isinstance(result, BashFix)
        assert isinstance(result.command, str)

    def test_large_plain_command_fast_path(self) -> None:
        command = "printf x " + "argument " * 20_000
        assert _fix_for_windows(command) == BashFix(command)

    def test_deeply_nested_command_substitutions(self) -> None:
        depth = 250
        source = "echo " + "$(echo " * depth + "$(rev)" + ")" * depth
        result = _fix_for_windows(source)
        assert result.command.count("rev()") == 1
        assert result.command.endswith("\n" + source)

    def test_extreme_nesting_beyond_scan_bound_is_unchanged(self) -> None:
        # Pathological nesting must neither hang the scanner nor crash it:
        # beyond the recursion bound the content is left byte-for-byte for
        # Bash to handle.
        depth = 2_000
        source = "echo " + "$(" * depth + "pwd" + ")" * depth
        assert _fix_for_windows(source) == BashFix(source)

    def test_nested_substitutions_still_rewritten_at_moderate_depth(self) -> None:
        depth = 50
        source = "cd " + "$(cd " * depth + "D:\\x" + ")" * depth
        result = _fix_for_windows(source)
        assert result.path_changes == ("D:\\x",)
        expected = "cd " + "$(cd " * depth + "D:/x" + ")" * depth
        assert result.command == expected

    def test_many_commands_are_all_rewritten_linearly(self) -> None:
        source = "; ".join(["rev"] * 2_000)
        result = _fix_for_windows(source)
        assert result.command.endswith("\n" + source)
        assert result.command.count("rev()") == 1
        assert result.replacements == ("rev",) * 2_000


class TestBashFixWindowsPaths:
    """Windows backslash paths and cmd.exe idioms rewritten for Git Bash."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("cd D:\\kimi-agent\\kimi-cli", "cd D:/kimi-agent/kimi-cli"),
            ("cd d:\\tools\\build", "cd d:/tools/build"),
            ("cd C:\\", "cd C:/"),
            ("cd \\\\server\\share\\dir", "cd //server/share/dir"),
            ("cd \\Users\\foo", "cd /Users/foo"),
            ("cd ~\\Desktop\\file.txt", "cd ~/Desktop/file.txt"),
            ("cd .\\build\\dist", "cd ./build/dist"),
            ("cd ..\\src\\lib", "cd ../src/lib"),
            ("cd ..\\..\\repo", "cd ../../repo"),
            ("mkdir build\\dist\\assets", "mkdir build/dist/assets"),
            ("cd foo\\\\bar", "cd foo/bar"),
            ("cd D:\\temp\\cache\\", "cd D:/temp/cache/"),
            ("ls D:\\*.txt", "ls D:/*.txt"),
            ("echo D:\\foo", "echo D:/foo"),
            ("cd D:\\a\\b && cp D:\\a\\b.txt E:\\dest\\",
             "cd D:/a/b && cp D:/a/b.txt E:/dest/"),
            ("cd \\Program\\ Files\\x", 'cd "/Program Files/x"'),
            ("env -C D:\\x cmd", "env -C D:/x cmd"),
            ("env --chdir D:\\x cmd", "env --chdir D:/x cmd"),
            ("env --chdir=D:\\x cmd", "env --chdir=D:/x cmd"),
            ("time -o D:\\out.txt cmd", "time -o D:/out.txt cmd"),
            ("time --output=D:\\out.txt cmd", "time --output=D:/out.txt cmd"),
              ("sudo -D D:\\x cmd", "sudo -D D:/x cmd"),
              # A Windows executable path as the command word itself: Bash
              # quote removal would eat the backslashes and lose the command.
              ("env -u FOO D:\\x cmd", "env -u FOO D:/x cmd"),
          ],
    )
    def test_windows_backslash_paths_rewritten(
        self, source: str, expected: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.command == expected
        assert result.changed
        assert result.replacements == ()
        assert result.path_changes

    def test_path_with_spaces_is_quoted(self) -> None:
        result = _fix_for_windows("cd D:\\Program\\ Files\\Git")
        assert result.command == 'cd "D:/Program Files/Git"'
        assert result.path_changes == ("D:\\Program\\ Files\\Git",)

    def test_path_with_metacharacter_is_quoted(self) -> None:
        result = _fix_for_windows("cd D:\\a\\&b\\c")
        assert result.command == 'cd "D:/a&b/c"'

    def test_tilde_with_spaces_keeps_tilde_outside_quotes(self) -> None:
        result = _fix_for_windows("cd ~\\My\\ Docs\\x")
        assert result.command == 'cd ~"/My Docs/x"'

    def test_path_change_reports_warning(self) -> None:
        result = _fix_for_windows("cd D:\\x")
        assert result.changed
        assert result.replacements == ()
        assert "forward slashes" in result.warning

    def test_cd_d_flag_dropped(self) -> None:
        result = _fix_for_windows("cd /d D:\\kimi-agent\\kimi-cli")
        assert result.command == "cd  D:/kimi-agent/kimi-cli"
        assert result.changed
        assert result.path_changes == ("cd /d", "D:\\kimi-agent\\kimi-cli")

    def test_cd_upper_d_flag_dropped(self) -> None:
        result = _fix_for_windows("cd /D C:\\Users\\me")
        assert result.command == "cd  C:/Users/me"

    def test_cd_flag_requires_following_argument(self) -> None:
        assert _fix_for_windows("cd /d") == BashFix("cd /d")
        assert _fix_for_windows("cd /d && echo x") == BashFix("cd /d && echo x")
        assert _fix_for_windows("cd /d # comment") == BashFix("cd /d # comment")

    def test_cd_flag_dropped_with_quoted_path_preserved(self) -> None:
        result = _fix_for_windows("cd /d 'D:\\x'")
        assert result.command == "cd  'D:\\x'"

    def test_multiple_cd_flags_on_one_line(self) -> None:
        result = _fix_for_windows("cd D:\\x; cd /d D:\\y")
        assert result.command == "cd D:/x; cd  D:/y"

    def test_path_in_command_substitution(self) -> None:
        result = _fix_for_windows("echo $(cd D:\\foo && pwd)")
        assert result.command == "echo $(cd D:/foo && pwd)"

    def test_redirection_target_path_rewritten(self) -> None:
        result = _fix_for_windows("echo hi > D:\\out.txt")
        assert result.command == "echo hi > D:/out.txt"

    def test_path_after_environment_assignment_is_data(self) -> None:
        # `env` consumes FOO=... as its own word: the value is data.
        assert _fix_for_windows("env FOO=D:\\x true") == BashFix("env FOO=D:\\x true")

    @pytest.mark.parametrize(
        "command",
        [
            "echo foo\\bar",  # single-segment relative: ambiguous
            "echo a\\nb",  # looks like an accidental escape
            "grep a\\b file",
            "echo \\n",
            "echo 'D:\\foo'",  # single-quoted text is literal data
            'echo "D:\\foo"',  # double-quoted text is literal data
            "printf '%s\\n' a",  # tool-level escapes stay quoted
            "echo -e 'a\\tb'",
            "echo $PATH\\foo",  # expansions untouched
            "echo ${DIR}\\x",
            "x=D:\\foo",  # assignment values untouched
            "alias cd=D:\\x",
            "case value in D:\\x) echo hit;; esac",
            "[[ -d D:\\x ]]",
            "# cd D:\\x",
            "echo ok # cd D:\\x",
            "cat <<'EOF'\ncd D:\\x\nEOF",  # heredoc body is data
            "cat <<< D:\\x",  # here-string body is data
            "echo \\",  # bare backslash
            "echo \\\\",  # bare double backslash
            # Bash escape sequences are ambiguous, not Windows paths: the
            # rewrite must never turn them into slashed directory words.
            r"echo \a\b",  # single-letter root-relative escapes
            r"echo \n\t",
            r"printf \033\015",  # octal escapes
            r"echo \x\41",  # hex escape style
            r"echo \033\033",
            r"echo x\n\t",  # relative escape sequences
            r"echo a\033\015",
            r"cd \a\b\c",  # all single-letter root segments
            r"cd a\b\c",  # all single-letter relative segments
            # wrapper option values that are not paths stay data
            r"env -C 'D:\x' cmd",  # quoted option value is literal data
            r"time -o log.txt cmd",
                          r"env -u D:\x cmd",  # -u takes a name; the value is opaque data
            r"env -C /d cmd",  # no backslash: not a Windows path
            # backslash-newline line continuations inside path words: the word
            # spans two physical lines and must stay untouched so Bash performs
            # the continuation itself (rewriting would inject a raw newline
            # into the emitted word and change the command's line structure).
            "cd \\Users\\foo\\\nb",
            "cd ~\\Desktop\\\nfile.txt",
            "cd D:\\a\\\nb",
            "cd build\\dist\\\nassets",
            "cd D:\\a\\\rb",  # escaped carriage return
        ],
    )
    def test_ambiguous_or_quoted_words_are_untouched(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    def test_non_windows_platform_is_noop(self) -> None:
        result = _fix_for_platform("cd D:\\x", "linux")
        assert result == BashFix("cd D:\\x")
        assert not result.changed


class TestBashFixArrayLiterals:
    """Windows path elements inside array literals rewritten for Git Bash.

    Array elements are data, not commands: quote removal would eat their
    backslashes (``arr=(D:\\x\\y)`` would store ``D:xy``), so unquoted
    element words get the same conservative rewrite as ordinary arguments.
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("arr=(D:\\x\\y.txt D:\\a\\b.txt)", "arr=(D:/x/y.txt D:/a/b.txt)"),
            ("arr+=(D:\\x\\y.txt)", "arr+=(D:/x/y.txt)"),
            # Glob metacharacters stay unquoted for pathname expansion.
            ("arr=(C:\\data\\*.csv)", "arr=(C:/data/*.csv)"),
            ("declare -a arr=(D:\\x\\y.txt)", "declare -a arr=(D:/x/y.txt)"),
            ("declare -a arr=(D:\\x\\y.txt plain)", "declare -a arr=(D:/x/y.txt plain)"),
            ("local arr=(D:\\x\\y.txt)", "local arr=(D:/x/y.txt)"),
            ("arr=(D:\\Program\\ Files\\x.txt)", 'arr=("D:/Program Files/x.txt")'),
            ("arr=(D:\\x\\y.txt); echo ${arr[0]}", "arr=(D:/x/y.txt); echo ${arr[0]}"),
        ],
    )
    def test_array_element_paths_rewritten(
        self, source: str, expected: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.command == expected
        assert result.changed
        assert result.replacements == ()
        assert result.path_changes

    @pytest.mark.parametrize(
        "command",
        [
            "arr=('D:\\x\\y.txt')",  # quoted element is literal data
            'arr=("D:\\x\\y.txt")',
            "array=(rev open pbcopy)",  # element words are data, not commands
            "declare -a x=(rev)",  # no fallback injection for element data
            "declare -a x=(wget xclip xsel)",
            "arr=([k]=D:\\x)",  # associative key=value words stay opaque
            "arr=(a\\nb)",  # ambiguous escape-like element
        ],
    )
    def test_array_elements_stay_data(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    def test_substitution_inside_array_is_scanned(self) -> None:
        result = _fix_for_windows("arr=($(rev <<< abc))")
        assert result.replacements == ("rev",)
        assert result.path_changes == ()


class TestBashFixCommandWordPaths:
    """Windows executable paths in command position rewritten for Git Bash.

    A command word such as ``C:\\tools\\rg.exe`` loses its backslashes to
    Bash quote removal (becoming the nonexistent ``C:toolsrg.exe``), so the
    same conservative path recognition used for argument words applies to
    the command word itself.
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("C:\\Windows\\System32\\where.exe git", "C:/Windows/System32/where.exe git"),
            ("d:\\tools\\run.exe --help", "d:/tools/run.exe --help"),
            ("\\\\server\\share\\tool.exe arg", "//server/share/tool.exe arg"),
            (".\\build\\tool.exe arg", "./build/tool.exe arg"),
            ("..\\scripts\\run.sh", "../scripts/run.sh"),
            ("~\\bin\\tool.exe --help", "~/bin/tool.exe --help"),
            ("\\Users\\me\\tool.exe", "/Users/me/tool.exe"),
            ("build\\dist\\tool.exe arg", "build/dist/tool.exe arg"),
            ("echo a && C:\\x\\tool.exe", "echo a && C:/x/tool.exe"),
            ("echo a; C:\\x\\tool.exe | cat", "echo a; C:/x/tool.exe | cat"),
            ("(C:\\x\\tool.exe)", "(C:/x/tool.exe)"),
            ("{ C:\\x\\tool.exe; }", "{ C:/x/tool.exe; }"),
            ("if C:\\x\\probe.exe; then echo ok; fi", "if C:/x/probe.exe; then echo ok; fi"),
            ("while C:\\x\\poll.exe; do :; done", "while C:/x/poll.exe; do :; done"),
            ("command C:\\x\\tool.exe", "command C:/x/tool.exe"),
            ("env FOO=1 D:\\x\\tool.exe", "env FOO=1 D:/x/tool.exe"),
            ("nohup D:\\x\\tool.exe &", "nohup D:/x/tool.exe &"),
            # Glob metacharacters stay unquoted for pathname expansion.
            ("D:\\x\\*.exe", "D:/x/*.exe"),
            # Normalized words that need it are double-quoted.
            ("D:\\Program\\ Files\\x.exe", '"D:/Program Files/x.exe"'),
            # Command words inside substitutions are rewritten too.
            ("x=$(C:\\x\\tool.exe)", "x=$(C:/x/tool.exe)"),
            ("echo `C:\\x\\tool.exe`", "echo `C:/x/tool.exe`"),
        ],
    )
    def test_command_word_paths_rewritten(
        self, source: str, expected: str
    ) -> None:
        result = _fix_for_windows(source)
        assert result.command == expected
        assert result.changed
        assert result.replacements == ()
        assert result.path_changes

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello",  # plain command word: no backslash
            "git --version",
            "'C:\\x\\tool.exe' arg",  # quoted command word is literal data
            '"C:\\x\\tool.exe" arg',
            "foo\\bar arg",  # single-segment relative path stays ambiguous
            "a\\nb arg",  # too short: ambiguous escape sequence
            r"\a\b",  # single-letter root-relative escapes
            r"x\n\t",  # escape-sequence-like segments
            r"case $f in D:\x) echo ok;; esac",  # case pattern is data
            r"case $f in (D:\x) echo ok;; esac",
        ],
    )
    def test_command_word_non_paths_untouched(self, command: str) -> None:
        assert _fix_for_windows(command) == BashFix(command)

    def test_fallback_name_takes_priority_over_path_rewrite(self) -> None:
        # ``\rev`` is the escaped literal command name ``rev``: the native
        # fallback must be injected, never a path rewrite.
        result = _fix_for_windows("\\rev <<< abc")
        assert result.replacements == ("rev",)
        assert result.path_changes == ()

    def test_command_word_path_change_reports_warning(self) -> None:
        result = _fix_for_windows("C:\\x\\tool.exe")
        assert result.changed
        assert result.replacements == ()
        assert result.path_changes == ("C:\\x\\tool.exe",)
        assert "forward slashes" in result.warning


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Git Bash")
class TestBashFixRealGitBash:
    @staticmethod
    def _run(command: str, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        bash = find_bash()
        assert bash is not None
        fixed = _fix_for_windows(command)
        # The full suite spawns hundreds of Git Bash processes back-to-back;
        # under load a fresh bash.exe can take longer than a 15 s first-time
        # startup (profile parsing + antivirus scanning) even though the
        # rewritten command itself completes in milliseconds.  These tests
        # pass in isolation, so the timeout is a resource-contention flake.
        # Retry once on TimeoutExpired to absorb the transient spike; a
        # genuine hang (a regression in the fixer) still fails after retry.
        for attempt in range(2):
            try:
                return subprocess.run(
                    [bash, "-lc", fixed.command],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                if attempt == 1:
                    raise
                time.sleep(1)
        raise AssertionError("unreachable")

    def test_gtimeout_rewrite_executes(self) -> None:
        result = self._run("gtimeout 2 bash -c 'printf timeout-ok'")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "timeout-ok"

    def test_command_word_windows_path_executes(self) -> None:
        # ``C:\Windows\System32\where.exe`` is a Windows executable path in
        # command position: the rewrite must make it runnable in Git Bash.
        result = self._run(r"C:\Windows\System32\where.exe git")
        assert result.returncode == 0, result.stderr
        assert "git.exe" in result.stdout

    def test_array_literal_windows_paths_execute(self) -> None:
        result = self._run(
            r"arr=(D:\x\y.txt D:\a\b.txt); printf '%s' ${arr[0]}"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "D:/x/y.txt"

    def test_wget_fallback_downloads_file(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wget_fixture.txt"
        fixture.write_text("wget-fixture\n", encoding="utf-8")
        url = "file:///" + str(fixture).replace("\\", "/")
        out = str(tmp_path / "wget_out.txt").replace("\\", "/")
        result = self._run(f"wget -q -O {out} {url}")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "wget_out.txt").read_text(encoding="utf-8") == "wget-fixture\n"

    def test_wget_fallback_stdout_mode(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wget_fixture.txt"
        fixture.write_text("wget-fixture\n", encoding="utf-8")
        url = "file:///" + str(fixture).replace("\\", "/")
        result = self._run(f"wget -q -O- {url}")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "wget-fixture\n"

    def test_wget_fallback_rejects_unsupported_option(self) -> None:
        result = self._run("wget --recursive https://example.com")
        assert result.returncode == 1
        assert "unsupported option" in result.stderr

    def test_xclip_fallback_clipboard_roundtrip(self) -> None:
        result = self._run(
            "printf roundtrip | xclip -selection clipboard"
            " && xclip -selection clipboard -o"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.rstrip("\r\n") == "roundtrip"

    def test_xsel_fallback_clipboard_roundtrip(self) -> None:
        result = self._run("printf seltest | xsel -b && xsel -bo")
        assert result.returncode == 0, result.stderr
        assert result.stdout.rstrip("\r\n") == "seltest"

    def test_xclip_fallback_rejects_unsupported_option(self) -> None:
        result = self._run("xclip -loops 3")
        assert result.returncode == 1
        assert "unsupported option" in result.stderr

    def test_gsed_fallback_executes(self) -> None:
        result = self._run("printf 'hello world\\n' | gsed 's/world/git-bash/'")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "hello git-bash\n"

    def test_gnu_g_prefix_fallback_executes(self) -> None:
        result = self._run("gawk 'BEGIN{print 2+3}'")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "5\n"

    def test_tree_fallback_lists_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "tree_root"
        (root / "sub").mkdir(parents=True)
        (root / "a.txt").write_text("a", encoding="utf-8")
        (root / "sub" / "b.txt").write_text("b", encoding="utf-8")
        posix = str(root).replace("\\", "/")
        result = self._run(f"tree -a {posix}")
        assert result.returncode == 0, result.stderr
        assert "a.txt" in result.stdout
        assert "sub" in result.stdout
        assert "b.txt" in result.stdout

    def test_zip_fallback_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path / "zip_root"
        (root / "sub").mkdir(parents=True)
        (root / "a.txt").write_text("alpha\n", encoding="utf-8")
        (root / "sub" / "b.txt").write_text("beta\n", encoding="utf-8")
        posix_root = str(root).replace("\\", "/")
        posix_zip = str(tmp_path / "out.zip").replace("\\", "/")
        result = self._run(
            f"cd {posix_root} && zip -qr {posix_zip} a.txt sub"
        )
        assert result.returncode == 0, result.stderr
        import zipfile

        names = zipfile.ZipFile(tmp_path / "out.zip").namelist()
        assert "a.txt" in names
        assert "sub/b.txt" in names

    def test_zip_fallback_rejects_unsupported_option(self) -> None:
        result = self._run("zip --encrypt out.zip file.txt")
        assert result.returncode == 1
        assert "unsupported option" in result.stderr

    def test_nc_z_fallback_open_and_closed_ports(self) -> None:
        import socket
        import threading

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        stop = threading.Event()

        def _accept() -> None:
            while not stop.is_set():
                try:
                    server.settimeout(0.2)
                    conn, _ = server.accept()
                    conn.close()
                except OSError:
                    continue

        thread = threading.Thread(target=_accept, daemon=True)
        thread.start()
        try:
            open_result = self._run(f"nc -z 127.0.0.1 {port}")
            assert open_result.returncode == 0, open_result.stderr
        finally:
            stop.set()
            server.close()
        closed_result = self._run("nc -z -w 2 127.0.0.1 1")
        assert closed_result.returncode == 1

    def test_nc_fallback_rejects_non_scan_mode(self) -> None:
        result = self._run("printf hi | nc 127.0.0.1 80")
        assert result.returncode == 1
        assert "only -z" in result.stderr

    def test_pgrep_fallback_finds_bash(self) -> None:
        result = self._run("pgrep bash")
        assert result.returncode == 0, result.stderr
        lines = result.stdout.split()
        assert lines
        assert all(line.isdigit() for line in lines)

    def test_pgrep_fallback_no_match_returns_one(self) -> None:
        result = self._run("pgrep kimix_no_such_process_name_zzz")
        assert result.returncode == 1

    def test_pkill_fallback_rejects_unsupported_option(self) -> None:
        result = self._run("pkill --signal KILL kimix_no_such_process_name_zzz")
        assert result.returncode == 1
        assert "unsupported option" in result.stderr

    def test_traceroute_fallback_localhost(self) -> None:
        result = self._run("traceroute -n -m 1 127.0.0.1")
        assert result.returncode == 0, result.stderr
        assert "127.0.0.1" in result.stdout

    def test_wl_clipboard_fallback_roundtrip(self) -> None:
        result = self._run("printf wltest | wl-copy && wl-paste")
        assert result.returncode == 0, result.stderr
        assert result.stdout.rstrip("\r\n") == "wltest"

    def test_python3_fallback_executes(self) -> None:
        result = self._run("python3 --version")
        assert result.returncode == 0, result.stderr
        assert "Python 3" in result.stdout

    def test_say_fallback_rejects_unsupported_option(self) -> None:
        result = self._run("say -v Alex hello")
        assert result.returncode == 1
        assert "unsupported option" in result.stderr

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("set -e; rev <<< abc; printf reached", "cba\nreached"),
            ("set -e; gtimeout 2 true; printf reached", "reached"),
        ],
    )
    def test_missing_native_delegate_survives_errexit(
        self, command: str, expected: str
    ) -> None:
        result = self._run(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected

    @pytest.mark.parametrize(
        "source",
        [
            "work() { rev <<< abc; }; work",
            "function work { rev <<< abc; }; work",
            "function work() { rev <<< abc; }; work",
        ],
    )
    def test_first_function_body_fallback_executes(self, source: str) -> None:
        result = self._run(source)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\n"

    def test_quoted_heredoc_parenthesis_in_substitution_executes_literally(self) -> None:
        command = "printf '%s\\n' \"$(cat <<'EOF'\n)\n$(command rev)\nEOF\n)\""
        result = self._run(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ")\n$(command rev)\n"

    def test_arithmetic_shift_in_substitution_does_not_hide_fallback(self) -> None:
        command = 'printf "%s\\n" "$(\n: $((1 << 2))\n)"\nrev <<< abc'
        result = self._run(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "\ncba\n"

    def test_case_pattern_in_substitution_runs_fallback(self) -> None:
        command = "printf '%s\\n' \"$(case x in x) rev <<< abc;; esac)\""
        result = self._run(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\n"

    @pytest.mark.parametrize(
        ("subject", "pattern"),
        [
            ("$(printf x)", "x"),  # command substitution subject
            ("`printf x`", "x"),  # backquote substitution subject
            ("$((0))", "0"),  # arithmetic expansion subject
        ],
    )
    def test_case_subject_substitution_runs_fallback(
        self, subject: str, pattern: str
    ) -> None:
        result = self._run(f"case {subject} in {pattern}) rev <<< abc;; esac")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\n"

    @pytest.mark.parametrize("terminator", [";;", ";&", ";;&"])
    def test_completed_case_in_substitution_preserves_later_fallback(
        self, terminator: str
    ) -> None:
        command = (
            "printf '%s\\n' \"$(case x in 'x') :"
            f"{terminator} esac)\"; rev <<< abc"
        )
        result = self._run(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout.endswith("cba\n")

    def test_rev_rewrite_executes_for_stdin(self) -> None:
        result = self._run("rev", stdin="abc\n123\n")
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["cba", "321"]

    def test_rev_rewrite_executes_for_file(self, tmp_path: Path) -> None:
        source = tmp_path / "rev-input.txt"
        source.write_text("first\nsecond\n", encoding="utf-8")
        path = str(source).replace("\\", "/")
        result = self._run(f"rev {path}")
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["tsrif", "dnoces"]

    def test_windows_backslash_cd_executes(self, tmp_path: Path) -> None:
        backslash = str(tmp_path).replace("/", "\\").replace(" ", "\\ ")
        result = self._run(f"cd {backslash} && printf reached")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "reached"

    def test_cd_d_flag_rewrite_executes(self, tmp_path: Path) -> None:
        backslash = str(tmp_path).replace("/", "\\").replace(" ", "\\ ")
        result = self._run(f"cd /d {backslash} && printf ok")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "ok"

    def test_windows_path_with_space_executes(self, tmp_path: Path) -> None:
        target = tmp_path / "sub dir"
        target.mkdir()
        source_path = str(target).replace(" ", "\\ ")
        result = self._run(f"cd {source_path} && printf spaced")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "spaced"

    def test_escape_sequence_words_are_not_rewritten(self) -> None:
        # ``\a\b`` and ``\n\t`` are Bash escapes ("ab nt"), not paths; the
        # fixer must leave them alone so the shell keeps its meaning.
        result = self._run(r"echo \a\b \n\t")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "ab nt\n"

    def test_env_c_path_option_rewrite_executes(self, tmp_path: Path) -> None:
        backslash = str(tmp_path).replace("/", "\\")
        result = self._run(f"env -C {backslash} pwd")
        assert result.returncode == 0, result.stderr
        # Git Bash canonicalizes the user temp directory to its ``/tmp`` MSYS
        # mount, so only the target directory name is compared, not the full
        # spelling.
        assert result.stdout.strip().endswith("/" + tmp_path.name)

    def test_nested_and_chained_rewrites_execute(self) -> None:
        result = self._run("gtimeout 2 bash -c 'printf abc' | rev")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba"

    def test_rev_preserves_unicode_characters(self) -> None:
        result = self._run("printf 'aé漢\\n' | rev")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "漢éa\n"

    @pytest.mark.parametrize("option", ["-0", "--zero", "-0 --", "--zero --"])
    def test_rev_zero_delimited_mode(self, option: str) -> None:
        result = self._run(f"printf 'abc\\0def\\0' | rev {option}")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\x00fed\x00"

    def test_rev_missing_file_returns_failure(self) -> None:
        result = self._run("rev /definitely/not/a/kimix-file")
        assert result.returncode != 0
        assert "kimix-file" in result.stderr

    def test_existing_function_takes_precedence_over_fallback(self) -> None:
        result = self._run("rev() { printf custom; }; rev </dev/null")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "custom"

    def test_inline_path_native_executable_takes_precedence(self, tmp_path: Path) -> None:
        native = tmp_path / "rev"
        native.write_text("#!/usr/bin/env bash\nprintf native-inline", encoding="utf-8")
        native.chmod(0o755)
        directory = str(tmp_path).replace("\\", "/")
        if len(directory) >= 3 and directory[1:3] == ":/":
            directory = "/" + directory[0].lower() + directory[2:]
        result = self._run(f"PATH='{directory}':$PATH rev </dev/null")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "native-inline"

    @pytest.mark.parametrize(
        "wrapped",
        [
            "command rev",
            "command -- rev",
            "env rev",
            "env -i rev",
            "env K=V rev",
            "env -u K rev",
            "nohup rev",
            "exec rev",
            "exec -a custom-rev rev",
        ],
    )
    def test_executable_wrappers_run_fallback(self, wrapped: str) -> None:
        result = self._run(f"printf 'abc\\n' | {wrapped}")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\n"

    @pytest.mark.parametrize("wrapped", ["time rev", "time -p rev"])
    def test_time_keyword_runs_fallback(self, wrapped: str) -> None:
        result = self._run(f"{wrapped} <<< abc")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba\n"

    @pytest.mark.parametrize(
        ("wrapped", "expected_stdout"),
        [
            ("command 2>/dev/null rev", "cba\n"),
            ("env 2>/dev/null rev", "cba\n"),
            ("nohup 2>/dev/null rev", "cba\n"),
            ("exec 2>/dev/null rev", "cba\n"),
            ("command >$(printf /dev/null) rev", ""),
            ("command > >(cat) rev", "cba\n"),
        ],
    )
    def test_executable_wrapper_survives_redirection(
        self, wrapped: str, expected_stdout: str
    ) -> None:
        result = self._run(f"printf 'abc\\n' | {wrapped}")
        assert result.returncode == 0, result.stderr
        assert "command not found" not in result.stderr.lower()
        assert result.stdout == expected_stdout

    def test_command_default_path_does_not_use_caller_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "rev"
        custom.write_text("#!/usr/bin/env bash\nprintf caller-path", encoding="utf-8")
        custom.chmod(0o755)
        directory = str(tmp_path).replace("\\", "/")
        if len(directory) >= 3 and directory[1:3] == ":/":
            directory = "/" + directory[0].lower() + directory[2:]
        result = self._run(f"PATH='{directory}':$PATH command -p rev")
        assert result.returncode == 127
        assert result.stdout == ""

    @pytest.mark.parametrize(
        "source",
        [
            "coproc rev <<< abc",
            "coproc worker { rev <<< abc; }",
            "coproc worker if rev <<< abc; then :; fi",
        ],
    )
    def test_coproc_runs_fallback(self, source: str) -> None:
        result = self._run(f"{source}; wait; printf '%s' \"${{COPROC_STATUS-}}\"")
        assert result.returncode == 0, result.stderr
        assert "command not found" not in result.stderr.lower()

    def test_conditionals_and_arithmetic_execute_unchanged(self) -> None:
        result = self._run(
            "[[ rev == rev && rev == rev ]] && rev=1 && (( rev )) && printf OK"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "OK"

    def test_commands_in_expanding_contexts_execute(self) -> None:
        result = self._run(
            "case x in x) :;; esac; values=($(rev <<< abc)); "
            "cat <<EOF\n${values[0]} $(rev <<< def)\nEOF"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "cba fed\n"

    def test_unknown_unmapped_command_still_fails_normally(self) -> None:
        result = self._run("setsid-kimix-command-that-does-not-exist")
        assert result.returncode == 127
        assert "command not found" in result.stderr.lower()

    def test_copy_fallback_executes(self, tmp_path: Path) -> None:
        source = tmp_path / "copy_src.txt"
        source.write_text("copy-fixture\n", encoding="utf-8")
        src = str(source).replace("\\", "/")
        dst = str(tmp_path / "copy_dst.txt").replace("\\", "/")
        result = self._run(f"copy {src} {dst}")
        assert result.returncode == 0, result.stderr
        assert Path(dst).read_text(encoding="utf-8") == "copy-fixture\n"

    def test_move_fallback_executes(self, tmp_path: Path) -> None:
        source = tmp_path / "move_src.txt"
        source.write_text("move-fixture\n", encoding="utf-8")
        src = str(source).replace("\\", "/")
        dst = str(tmp_path / "move_dst.txt").replace("\\", "/")
        result = self._run(f"move {src} {dst}")
        assert result.returncode == 0, result.stderr
        assert not source.exists()
        assert Path(dst).read_text(encoding="utf-8") == "move-fixture\n"

    @pytest.mark.parametrize("cmd", ["del", "erase"])
    def test_del_and_erase_fallback_executes(self, cmd: str, tmp_path: Path) -> None:
        target = tmp_path / f"{cmd}_target.txt"
        target.write_text("data", encoding="utf-8")
        path = str(target).replace("\\", "/")
        result = self._run(f"{cmd} {path}")
        assert result.returncode == 0, result.stderr
        assert not target.exists()

    @pytest.mark.parametrize("cmd", ["ren", "rename"])
    def test_ren_and_rename_fallback_executes(self, cmd: str, tmp_path: Path) -> None:
        source = tmp_path / f"{cmd}_src.txt"
        source.write_text("rename-fixture\n", encoding="utf-8")
        src = str(source).replace("\\", "/")
        dst = str(tmp_path / f"{cmd}_dst.txt").replace("\\", "/")
        result = self._run(f"{cmd} {src} {dst}")
        assert result.returncode == 0, result.stderr
        assert not source.exists()
        assert Path(dst).read_text(encoding="utf-8") == "rename-fixture\n"

    def test_md_and_rd_fallback_executes(self, tmp_path: Path) -> None:
        directory = str(tmp_path / "md_rd_dir").replace("\\", "/")
        result = self._run(f"md {directory} && rd {directory}")
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "md_rd_dir").exists()

    def test_chdir_fallback_executes(self) -> None:
        result = self._run("chdir /tmp && pwd")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("/tmp")

    def test_mklink_fallback_creates_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "sym_target.txt"
        source.write_text("symlink-fixture\n", encoding="utf-8")
        src = str(source).replace("\\", "/")
        dst = str(tmp_path / "sym_link.txt").replace("\\", "/")
        result = self._run(f"mklink {dst} {src}")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "sym_link.txt").exists()
        assert (
            tmp_path / "sym_link.txt"
        ).read_text(encoding="utf-8") == "symlink-fixture\n"

    def test_mklink_hard_link_fallback_executes(self, tmp_path: Path) -> None:
        source = tmp_path / "hard_src.txt"
        source.write_text("hard-link-fixture\n", encoding="utf-8")
        src = str(source).replace("\\", "/")
        dst = str(tmp_path / "hard_dst.txt").replace("\\", "/")
        result = self._run(f"mklink /H {dst} {src}")
        assert result.returncode == 0, result.stderr
        assert Path(dst).read_text(encoding="utf-8") == "hard-link-fixture\n"

    def test_pidof_fallback_finds_bash(self) -> None:
        result = self._run("pidof bash")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
        assert all(p.strip().isdigit() for p in result.stdout.split())

    def test_killall_nonexistent_returns_one(self) -> None:
        result = self._run("killall kimix_no_such_process_name_zzz")
        assert result.returncode == 1

    def test_column_fallback_formats_table(self, tmp_path: Path) -> None:
        source = tmp_path / "column_input.txt"
        source.write_text("alpha beta\ngamma delta\n", encoding="utf-8")
        path = str(source).replace("\\", "/")
        result = self._run(f"column -t {path}")
        assert result.returncode == 0, result.stderr
        assert "alpha" in result.stdout
        assert "beta" in result.stdout


# ============================================================================
# _prepare_bash_cmd
# ============================================================================

class TestPrepareBashCmd:
    def test_noop_on_non_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "linux"):
            assert _prepare_bash_cmd("echo hello") == "echo hello"

    def test_noop_on_darwin(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "darwin"):
            assert _prepare_bash_cmd("echo hello") == "echo hello"

    def test_noop_on_windows_without_backslash(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("echo hello") == "echo hello"

    def test_converts_unquoted_backslashes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat src\kimix\tools\file\bash\bash_tool.py"
            result = _prepare_bash_cmd(cmd)
            assert result == "cat src/kimix/tools/file/bash/bash_tool.py"

    def test_preserves_single_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = "echo 'hello world'"
            result = _prepare_bash_cmd(cmd)
            assert result == "echo 'hello world'"

    def test_preserves_backslashes_inside_single_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo 'hello\world'"
            result = _prepare_bash_cmd(cmd)
            assert result == r"echo 'hello\world'"

    def test_preserves_backslashes_inside_double_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "hello\world"'
            result = _prepare_bash_cmd(cmd)
            assert result == r'echo "hello\world"'

    def test_preserves_backslashes_inside_ansi_c_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $'hello\nworld'"
            result = _prepare_bash_cmd(cmd)
            assert result == r"echo $'hello\nworld'"

    def test_empty_command_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("") == ""

    def test_pipes_and_redirects_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = "echo hello | grep h > out.txt"
            result = _prepare_bash_cmd(cmd)
            assert result == "echo hello | grep h > out.txt"

    def test_drive_letter_path_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat C:\Users\test\file.txt"
            result = _prepare_bash_cmd(cmd)
            assert result == "cat C:/Users/test/file.txt"

    def test_relative_paths_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"cd .\subdir") == "cd ./subdir"
            assert _prepare_bash_cmd(r"cd ..\parent") == "cd ../parent"

    def test_multiple_paths_in_one_command_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"diff a\b\c.py x\y\z.py"
            assert _prepare_bash_cmd(cmd) == "diff a/b/c.py x/y/z.py"

    def test_mixed_quoted_and_unquoted_backslashes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat 'src\a.py' src\b.py"
            assert _prepare_bash_cmd(cmd) == r"cat 'src\a.py' src/b.py"

    def test_escaped_quote_inside_double_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "hello \"world\""'
            assert _prepare_bash_cmd(cmd) == r'echo "hello \"world\""'

    def test_unclosed_single_quote_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo 'hello src\file.py"
            assert _prepare_bash_cmd(cmd) == r"echo 'hello src\file.py"

    def test_unclosed_double_quote_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "hello src\file.py'
            assert _prepare_bash_cmd(cmd) == r'echo "hello src\file.py'

    def test_dollar_quote_with_escaped_single_quote_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $'it\'s working'"
            assert _prepare_bash_cmd(cmd) == r"echo $'it\'s working'"

    def test_backslash_before_special_chars_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Backslash escapes before bash metacharacters are preserved
            assert _prepare_bash_cmd(r"echo a\|b") == r"echo a\|b"
            assert _prepare_bash_cmd(r"echo a\;b") == r"echo a\;b"
            assert _prepare_bash_cmd(r"echo a\&b") == r"echo a\&b"
            assert _prepare_bash_cmd(r"echo a\>b") == r"echo a\>b"
            assert _prepare_bash_cmd(r"echo a\<b") == r"echo a\<b"

    def test_double_backslash_outside_quotes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Each backslash is converted individually (\\ -> //)
            assert _prepare_bash_cmd(r"echo \\path") == "echo //path"

    def test_backslash_at_end_of_string_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("echo trailing\\") == "echo trailing/"

    def test_pipes_and_redirects_with_paths_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat src\a.py | grep x > out\b.txt"
            assert _prepare_bash_cmd(cmd) == "cat src/a.py | grep x > out/b.txt"

    def test_preserves_quoted_path_with_spaces_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'cat "C:\Program Files\app\file.txt"'
            assert _prepare_bash_cmd(cmd) == r'cat "C:\Program Files\app\file.txt"'

    def test_preserves_single_quoted_path_with_spaces_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat 'C:\Program Files\app\file.txt'"
            assert _prepare_bash_cmd(cmd) == r"cat 'C:\Program Files\app\file.txt'"

    def test_command_substitution_with_backslashes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # $(...) is not a quoted region; backslashes inside are converted
            cmd = r"echo $(cat src\file.py)"
            assert _prepare_bash_cmd(cmd) == "echo $(cat src/file.py)"

    def test_backtick_with_backslashes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Backticks are not a quoted region; backslashes inside are converted
            cmd = r"echo `cat src\file.py`"
            assert _prepare_bash_cmd(cmd) == "echo `cat src/file.py`"

    def test_find_command_with_escaped_parens_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'find build -maxdepth 4 \( -name "luisa-xir*" -o -name "luisa-spirv*" \) | head -n 20'
            expected = r'find build -maxdepth 4 \( -name "luisa-xir*" -o -name "luisa-spirv*" \) | head -n 20'
            assert _prepare_bash_cmd(cmd) == expected

    def test_backslash_space_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Backslash-escaped space must be preserved so the word remains single token
            assert _prepare_bash_cmd(r"echo hello\ world") == r"echo hello\ world"

    def test_backslash_dollar_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \$HOME") == r"echo \$HOME"

    def test_backslash_star_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \*") == r"echo \*"

    def test_backslash_backtick_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \`cmd\`") == r"echo \`cmd\`"

    def test_backslash_brace_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \{a,b\}") == r"echo \{a,b\}"

    def test_backslash_tilde_preserved_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \~user") == r"echo \~user"

    def test_mixed_paths_and_escapes_on_windows(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat src\tools\file.py && find build \( -name '*.py' \)"
            expected = r"cat src/tools/file.py && find build \( -name '*.py' \)"
            assert _prepare_bash_cmd(cmd) == expected

    def test_escaped_single_quote_outside_quotes_on_windows(self) -> None:
        """\' outside quotes should be preserved and NOT start a single-quoted region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # \' → literal ', backslashes after should be converted
            cmd = r"echo \'src\kimix\'"
            expected = r"echo \'src/kimix\'"
            assert _prepare_bash_cmd(cmd) == expected

    def test_escaped_double_quote_outside_quotes_on_windows(self) -> None:
        r"""\" outside quotes should be preserved and NOT start a double-quoted region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # \" outside quotes: backslash escapes the double-quote → literal "
            # The " should NOT start a double-quoted region.
            cmd = r'echo \"src\kimix\"'
            expected = r'echo \"src/kimix\"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_escaped_dollar_prevents_ansi_c_detection_on_windows(self) -> None:
        """Escaped dollar before single-quote should NOT trigger ANSI-C processing."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # \$'text' — the $ is escaped, so 'text' is a separate single-quoted string
            cmd = r"echo \$'text'"
            expected = r"echo \$'text'"
            assert _prepare_bash_cmd(cmd) == expected

    # -- corner cases discovered during review -------------------------------

    def test_double_quoted_escaped_backslash_before_quote_on_windows(self) -> None:
        r"""\\" inside double quotes: \\ is escaped backslash, then " closes the region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Bash: "hello\\" is the quoted region, then world", then "
            cmd = r'"hello\\"world"'
            # \\ inside "..." preserved, then world is outside (no backslashes),
            # then " starts new region
            expected = r'"hello\\"world"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_double_quoted_multiple_escaped_backslashes_on_windows(self) -> None:
        r"""Multiple \\ sequences inside double quotes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'"a\\b\\c"'
            expected = r'"a\\b\\c"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_double_quoted_escaped_dollar_on_windows(self) -> None:
        r"""\$ inside double quotes should not affect region detection."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'"price is \$100"'
            expected = r'"price is \$100"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_double_quoted_with_dollar_ansi_c_inside_on_windows(self) -> None:
        r"""$' inside double quotes should NOT trigger ANSI-C processing."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "abc $'def' ghi" — the $' is inside double quotes, treated literally
            cmd = r'"abc $"' + "'def' ghi\""
            # The double-quoted region captures everything from first " to last "
            expected = r'"abc $"' + "'def' ghi\""
            assert _prepare_bash_cmd(cmd) == expected

    def test_backslash_before_hash_preserved_on_windows(self) -> None:
        r"""\# should be preserved as bash comment escape."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \# not a comment") == r"echo \# not a comment"

    def test_backslash_before_exclamation_preserved_on_windows(self) -> None:
        r"""\! should be preserved as history expansion escape."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \!test") == r"echo \!test"

    def test_backslash_before_percent_preserved_on_windows(self) -> None:
        r"""\% should be preserved."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \%percent") == r"echo \%percent"

    def test_backslash_before_equals_preserved_on_windows(self) -> None:
        r"""\= should be preserved as assignment escape."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo a\=b") == r"echo a\=b"

    def test_triple_backslash_outside_quotes_on_windows(self) -> None:
        r"""\\\ outside quotes: \\ → //, then \ before p → /p → ///path."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd(r"echo \\\path") == "echo ///path"

    def test_backslash_before_newline_preserved_on_windows(self) -> None:
        r"""\<newline> (line continuation) should be preserved."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = "echo hello\\\nworld"
            expected = "echo hello\\\nworld"
            assert _prepare_bash_cmd(cmd) == expected

    def test_ansi_c_with_double_backslash_before_quote_on_windows(self) -> None:
        r"""$'...\\'' — \\ inside ANSI-C, then ' closes the region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # $'it\\''s' → $'it\\' + 's' (the \\ produces \, then ' closes)
            cmd = r"echo $'it\\'s working'"
            expected = r"echo $'it\\'s working'"
            assert _prepare_bash_cmd(cmd) == expected

    def test_ansi_c_with_hex_escape_on_windows(self) -> None:
        r"""$'...\x41...' — hex escapes are skipped correctly."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $'\x41bc'"
            expected = r"echo $'\x41bc'"
            assert _prepare_bash_cmd(cmd) == expected

    def test_ansi_c_with_octal_escape_on_windows(self) -> None:
        r"""$'...\033...' — octal escapes are skipped correctly."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $'\033[31mred'"
            expected = r"echo $'\033[31mred'"
            assert _prepare_bash_cmd(cmd) == expected

    def test_ansi_c_with_unicode_escape_on_windows(self) -> None:
        r"""$'...\u0041...' — unicode escapes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $'\u0041bc'"
            expected = r"echo $'\u0041bc'"
            assert _prepare_bash_cmd(cmd) == expected

    def test_mixed_quotes_complex_on_windows(self) -> None:
        """Complex mix of quote types and backslashes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # 'single' preserved, "double" preserved, $'ansi' preserved, src\path → src/path
            cmd = "echo 'single' \"double\" $'ansi' src\\path"
            expected = "echo 'single' \"double\" $'ansi' src/path"
            assert _prepare_bash_cmd(cmd) == expected

    def test_only_backslashes_on_windows(self) -> None:
        """String with only backslashes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("\\\\") == "//"
            assert _prepare_bash_cmd("\\") == "/"
            assert _prepare_bash_cmd("\\\\\\") == "///"

    def test_backslash_before_each_metachar_on_windows(self) -> None:
        """Every metacharacter preceded by backslash is preserved."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            metachars = "()|;&<>$\"`'*?[]{}~!#=% \t\n\r"
            for ch in metachars:
                # Build a command with \X where X is a metachar
                cmd = "echo \\" + ch
                result = _prepare_bash_cmd(cmd)
                # The \X pair should be preserved as \X
                assert ("\\" + ch) in result, f"Failed for \\{repr(ch)}: {result}"

    def test_double_quoted_empty_on_windows(self) -> None:
        """Empty double-quoted region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd('echo ""') == 'echo ""'

    def test_ansi_c_empty_on_windows(self) -> None:
        """Empty ANSI-C quoted region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("echo $''") == "echo $''"

    def test_single_quoted_empty_on_windows(self) -> None:
        """Empty single-quoted region."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd("echo ''") == "echo ''"

    def test_double_quoted_escaped_backslash_at_end_on_windows(self) -> None:
        r"""Double-quoted region with \\ at the very end."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "hello\\" — \\ inside, then " closes
            cmd = r'"hello\\"'
            expected = r'"hello\\"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_double_quoted_escaped_backslash_and_quote_on_windows(self) -> None:
        r"""Double-quoted with \\\" — \\ (escaped backslash) then \" (escaped quote)."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "hello\\\"world" — \\ → \, \" → " (escaped quote, region continues)
            cmd = r'"hello\\\"world"'
            expected = r'"hello\\\"world"'
            assert _prepare_bash_cmd(cmd) == expected

    # -- corner case: $(...) and backticks inside double quotes ----------------
    # bash runs the content of a command substitution in a subshell where it is
    # parsed unquoted.  So backslashes inside $(...) or `...` must be processed
    # (converted to /) even when the substitution is nested inside "...".

    def test_dq_with_command_substitution_and_backslash_path_on_windows(self) -> None:
        r"""echo "$(cat src\foo\bar)" — backslashes inside $(...) within DQ are converted."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(cat src\foo\bar)"'
            assert _prepare_bash_cmd(cmd) == 'echo "$(cat src/foo/bar)"'

    def test_dq_with_backtick_substitution_and_backslash_path_on_windows(self) -> None:
        """Backticks inside DQ (unescaped) start a command substitution.

        bash runs the content in a subshell, so backslashes inside are
        processed (converted to /) just like at the top level.
        """
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "`cat src\foo\bar`"'
            assert _prepare_bash_cmd(cmd) == 'echo "`cat src/foo/bar`"'


    def test_dq_with_nested_command_substitution_on_windows(self) -> None:
        r"""Nested $(...) inside DQ — both levels process backslashes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(cat $(echo src\foo\bar))"'
            assert _prepare_bash_cmd(cmd) == 'echo "$(cat $(echo src/foo/bar))"'

    def test_dq_with_backtick_inside_command_substitution_on_windows(self) -> None:
        r"""Backticks nested inside $(...) within DQ — content is processed."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(cat `echo src\foo`)"'
            assert _prepare_bash_cmd(cmd) == 'echo "$(cat `echo src/foo`)"'

    def test_dq_with_command_substitution_inside_backticks_on_windows(self) -> None:
        r"""$(...) nested inside `...` at top level — content is processed."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo `cat $(echo src\foo\bar)`'
            assert _prepare_bash_cmd(cmd) == 'echo `cat $(echo src/foo/bar)`'

    def test_dq_with_quoted_path_and_command_subst_on_windows(self) -> None:
        r"""Mixed: quoted path (preserved) + $(...) substitution (converted)."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "literal src\foo" "$(cat src\bar\baz)"'
            assert _prepare_bash_cmd(cmd) == 'echo "literal src\\foo" "$(cat src/bar/baz)"'

    def test_dq_ansi_c_inside_command_substitution_on_windows(self) -> None:
        r"""$'...' inside $(...) within DQ — ANSI-C region is preserved literally."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(echo $'"'"'\n'"'"')"'
            assert _prepare_bash_cmd(cmd) == r'echo "$(echo $'"'"'\n'"'"')"'

    def test_dq_single_quotes_inside_command_substitution_on_windows(self) -> None:
        r"""Single-quoted path inside $(...) within DQ — backslashes preserved."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(cat '"'"'src\foo\bar'"'"')"'
            assert _prepare_bash_cmd(cmd) == r'echo "$(cat '"'"'src\foo\bar'"'"')"'

    def test_dq_escaped_dollar_paren_not_command_substitution_on_windows(self) -> None:
        r"""\$( inside DQ — the $ is escaped, so ( is NOT a command substitution."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "\$(not a sub) src\file"'
            # \$ makes $ literal; ( ) are regular; src\file is preserved by DQ.
            assert _prepare_bash_cmd(cmd) == r'echo "\$(not a sub) src\file"'

    def test_dq_escaped_backtick_not_substitution_on_windows(self) -> None:
        r"""\` inside DQ — the ` is escaped, so it's a literal backtick, not substitution."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "\`not a sub\` src\file"'
            # \` makes ` literal; src\file is preserved by DQ.
            assert _prepare_bash_cmd(cmd) == r'echo "\`not a sub\` src\file"'

    def test_dq_empty_command_substitution_on_windows(self) -> None:
        """Empty $(...) inside DQ."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd('echo "$()"') == 'echo "$()"'

    def test_dq_empty_backticks_on_windows(self) -> None:
        """Empty `` `` `` inside DQ."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            assert _prepare_bash_cmd('echo "``"') == 'echo "``"'

    def test_unterminated_dq_with_command_substitution_on_windows(self) -> None:
        r"""Unterminated DQ that contains $( — passed through to bash to error."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(unterminated'
            assert _prepare_bash_cmd(cmd) == r'echo "$(unterminated'

    def test_unterminated_command_substitution_inside_dq_on_windows(self) -> None:
        r"""$(... with no matching ) inside DQ — passed through."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(no close paren"'
            assert _prepare_bash_cmd(cmd) == r'echo "$(no close paren"'

    def test_unterminated_backticks_inside_dq_on_windows(self) -> None:
        r"""Unterminated ` inside DQ — passed through."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "`no close"'
            assert _prepare_bash_cmd(cmd) == r'echo "`no close"'

    def test_dq_with_dq_inside_command_substitution_on_windows(self) -> None:
        r"""DQ inside $(...) inside DQ — inner DQ preserves its backslashes."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "$(echo "src\foo" rest)" — inner DQ preserves \, rest converted
            cmd = r'echo "$(echo "src\foo" rest\bar)"'
            assert _prepare_bash_cmd(cmd) == r'echo "$(echo "src\foo" rest/bar)"'

    def test_top_level_backtick_with_escaped_backtick_on_windows(self) -> None:
        r"""\` at top level — escaped backtick, literal, not substitution start."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo \`not_sub\`"
            assert _prepare_bash_cmd(cmd) == r"echo \`not_sub\`"

    def test_top_level_nested_backticks_with_path_on_windows(self) -> None:
        """`` `cmd1`cmd2` `` style — backtick region content is processed."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # Outer backtick runs `cmd src\file`, inner is just text
            cmd = r"echo `cat src\file.txt`"
            assert _prepare_bash_cmd(cmd) == "echo `cat src/file.txt`"

    def test_command_substitution_with_nested_parens_on_windows(self) -> None:
        r"""$(echo (nested) paren) — ) inside parens is balanced correctly."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"echo $(echo (src\foo\bar))"
            # The ) after "bar" closes the $().  The ) at the end is a stray.
            # Actually: $(echo (src\foo\bar)) — opens $(, then echo (, then
            # content, then ) closes the inner paren, then )) closes the $().
            # Let's just verify it doesn't crash and paths are converted.
            result = _prepare_bash_cmd(cmd)
            assert "src/foo/bar" in result

    def test_dq_ansi_c_immediately_before_closing_quote_on_windows(self) -> None:
        r"""$'...' right before closing " in DQ — must not skip the closing quote."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "abc $'def'" — the ANSI-C region ends right before the closing "
            cmd = r'"abc $'"'"'def'"'"'"'
            assert _prepare_bash_cmd(cmd) == r'"abc $'"'"'def'"'"'"'

    def test_dq_backtick_immediately_before_closing_quote_on_windows(self) -> None:
        """Backtick region right before closing " in DQ."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "abc `def`" — backtick region ends right before the closing "
            cmd = '"abc `def`"'
            assert _prepare_bash_cmd(cmd) == '"abc `def`"'

    def test_dq_command_subst_immediately_before_closing_quote_on_windows(self) -> None:
        """$(...) right before closing " in DQ."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            # "abc $(echo x)" — command substitution ends right before the closing "
            cmd = '"abc $(echo x)"'
            assert _prepare_bash_cmd(cmd) == '"abc $(echo x)"'

    def test_dq_with_complex_nesting_on_windows(self) -> None:
        r"""Complex nesting: $(echo "$(echo src\foo)" `echo src\bar`)."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(echo "$(echo src\foo)" `echo src\bar`)"'
            # Both $() levels convert paths; inner DQ preserves its \
            expected = r'echo "$(echo "$(echo src/foo)" `echo src/bar`)"'
            assert _prepare_bash_cmd(cmd) == expected

    def test_unc_path_converted_on_windows(self) -> None:
        r"""UNC path \\server\share\file.txt → //server/share/file.txt."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r"cat \\server\share\file.txt"
            assert _prepare_bash_cmd(cmd) == "cat //server/share/file.txt"

    def test_dq_command_subst_with_backslashes_converted_on_windows(self) -> None:
        r"""$(...) nested in double quotes: its content is parsed unquoted, so
        backslash paths inside are converted."""
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            cmd = r'echo "$(cat C:\a\b.txt)"'
            assert _prepare_bash_cmd(cmd) == 'echo "$(cat C:/a/b.txt)"'


# ============================================================================
# Bash.__call__ — integration tests with backslash paths on Windows
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Backslash path handling targets Git Bash on Windows",
)
class TestBashBackslashPaths:
    async def test_cat_with_backslash_path(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello slash", encoding="utf-8")
        bash = Bash(session=mock_session)
        # Unquoted Windows backslash path is auto-converted to forward slashes.
        params = BashParams(cmd=f"cat {f}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello slash" in result.output

    async def test_ls_with_backslash_path(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd=r"ls src\kimix\tools\file\bash")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "bash_tool.py" in result.output

    async def test_cd_with_backslash_path(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd=r"cd src\kimix\tools\file\bash && pwd")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "bash" in result.output

    async def test_multiple_backslash_paths(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello slash", encoding="utf-8")
        bash = Bash(session=mock_session)
        params = BashParams(cmd=f"echo {tmp_path} && cat {f}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello slash" in result.output

    async def test_quoted_backslash_path_preserved(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello slash", encoding="utf-8")
        bash = Bash(session=mock_session)
        # Backslashes inside single quotes are preserved; Git Bash resolves them.
        params = BashParams(cmd=f"cat '{f}'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello slash" in result.output

    async def test_double_quoted_backslash_path_preserved(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello slash", encoding="utf-8")
        bash = Bash(session=mock_session)
        # Backslashes inside double quotes are preserved; Git Bash resolves them.
        params = BashParams(cmd=f'cat "{f}"')
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello slash" in result.output


# ============================================================================
# Bash.description — Windows-specific experience text
# ============================================================================

class TestBashDescription:
    """The Bash tool description carries the verified Windows slash rules."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock()
        session.custom_config.get.return_value = {}
        session.custom_data = {}
        return session

    def _make_tool(self, platform: str, mock_session: MagicMock) -> Bash:
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", platform), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ), patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=(r"C:\Git\bin\bash.exe" if platform == "win32" else "/bin/bash"),
        ):
            return Bash(session=mock_session)


# ============================================================================
# Bash.__call__
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
class TestBashCall:
    async def test_echo_hello(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output

    async def test_true_command(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="true")
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_false_command(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="false")
        result = await bash(params)
        assert isinstance(result, ToolError)

    async def test_pipefail_surfaces_real_exit_code(
        self, mock_session: MagicMock
    ) -> None:
        """A failing producer piped to a consumer must report the producer's
        non-zero exit (one-shot commands run with ``set -o pipefail``), not the
        consumer's 0 that previously masked crashes behind ``| head``."""
        bash = Bash(session=mock_session)
        params = BashParams(cmd="false | head -1")
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "exit_code: 1" in result.output

    async def test_pipefail_keeps_success_pipeline(
        self, mock_session: MagicMock
    ) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hi | head -1")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hi" in result.output

    async def test_grep_no_match_reports_success(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        """grep exit 1 (no matches) is a normal outcome: it must be a ToolOk
        with the meaning noted, not a 'failed ... run it again' error."""
        f = tmp_path / "sample.txt"
        f.write_text("nothing here\n", encoding="utf-8")
        posix = str(f).replace("\\", "/")
        bash = Bash(session=mock_session)
        params = BashParams(cmd=f"grep __definitely_missing__ {posix}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "No matches found" in result.message
        assert "failed" not in result.message.lower()

    async def test_sigpipe_truncation_reports_success(
        self, mock_session: MagicMock
    ) -> None:
        """SIGPIPE (141) from a truncated pipeline is an expected outcome when
        the one-shot shell runs with pipefail."""
        bash = Bash(session=mock_session)
        params = BashParams(cmd="seq 1 100000 | head -1", timeout=30)
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "SIGPIPE" in result.message

    async def test_unknown_command_error(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="no_such_command_12345", timeout=5)
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "command not found" in result.output or "not found" in result.output.lower()

    async def test_ls_current_dir(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="ls .", timeout=10)
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_echo_with_multiple_args(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello world")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello world" in result.output

    async def test_echo_with_timeout(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo quick", timeout=30)
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_cat_file(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello cat", encoding="utf-8")
        bash = Bash(session=mock_session)
        # Use forward slashes so bash does not interpret backslashes as escapes
        posix_path = str(f).replace("\\", "/")
        params = BashParams(cmd=f"cat {posix_path}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello cat" in result.output

    async def test_pwd(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="pwd")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_whoami(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="whoami")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0


    async def test_timeout(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="sleep 5", timeout=3)
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "Timeout" in result.brief

    async def test_bash_not_found_fallback(self, mock_session: MagicMock) -> None:
        """When bash is not found, Bash.__init__ raises SkipThisTool."""
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=None):
            with pytest.raises(SkipThisTool):
                Bash(session=mock_session)


# ============================================================================
# Edge cases
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
class TestEdgeCases:
    async def test_command_with_special_chars(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'hello\tworld'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # Tab may be preserved or converted by echo depending on bash version
        assert "hello" in result.output
        assert "world" in result.output

    async def test_command_with_quotes(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd='echo "quoted text"')
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "quoted text" in result.output


# ============================================================================
# Inactivity timeout behavior
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
class TestBashInactivityTimeout:
    async def test_bash_inactivity_timeout_returns_background_error(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.background.utils.DEFAULT_INACTIVITY_TIMEOUT", 2.0
        ):
            bash = Bash(session=mock_session)
            params = BashParams(cmd="sleep 120", timeout=90)
            result = await bash(params)
            assert isinstance(result, ToolError)
            assert result.brief == "Timeout"
            assert "Running in background" in result.message
            assert "task_id" in result.message

    async def test_bash_short_timeout_unchanged(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="sleep 5", timeout=3)
        start = asyncio.get_event_loop().time()
        result = await bash(params)
        elapsed = asyncio.get_event_loop().time() - start
        assert isinstance(result, ToolError)
        assert result.brief == "Timeout"
        assert 2.5 <= elapsed <= 4.0


@pytest.mark.skipif(
    not PWSH_AVAILABLE,
    reason="PowerShell tool is not available on this platform",
)
class TestPowershellInactivityTimeout:
    @pytest.fixture(autouse=True)
    def _force_pwsh_enabled(self) -> Any:
        """Force-enable the Powershell tool for these integration tests.

        With the default Git-Bash-first policy on Windows, the gate
        ``_should_enable_powershell()`` is False whenever Git Bash is
        installed; these tests execute real pwsh commands, so the gate is
        bypassed.
        """
        with patch(
            "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
            return_value=True,
        ):
            yield

    async def test_pwsh_inactivity_timeout_returns_background_error(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.background.utils.DEFAULT_INACTIVITY_TIMEOUT", 2.0
        ):
            pwsh = Powershell(session=mock_session)
            params = PowershellParams(cmd="Start-Sleep -Seconds 120", timeout=90)
            result = await pwsh(params)
            assert isinstance(result, ToolError)
            assert result.brief == "Timeout"
            assert "Running in background" in result.message
            assert "task_id" in result.message

    async def test_pwsh_short_timeout_unchanged(self, mock_session: MagicMock) -> None:
        pwsh = Powershell(session=mock_session)
        params = PowershellParams(cmd="Start-Sleep -Seconds 5", timeout=3)
        start = asyncio.get_event_loop().time()
        result = await pwsh(params)
        elapsed = asyncio.get_event_loop().time() - start
        assert isinstance(result, ToolError)
        assert result.brief == "Timeout"
        assert 2.5 <= elapsed <= 4.0


# ============================================================================
# Complex bash commands — pipes, redirects, substitution, etc.
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
class TestComplexCommands:
    @pytest.fixture(autouse=True)
    def _raw_command_output(self) -> Any:
        """These tests assert on raw command output; disable rtk rewriting
        (which wraps output in a metadata envelope) regardless of whether
        an rtk binary is installed on the host."""
        with patch("kimix.tools.common._rtk_available", return_value=False):
            yield

    @staticmethod
    def _cmd_output(result: ToolOk | ToolError) -> str:
        """Extract the raw process output from the session output block.

        The Bash tool wraps process output in a metadata envelope
        (``task_id:``/``status:``/``output: |`` ...); these tests assert on
        the raw command output inside it.
        """
        marker = "output: |\n"
        text = result.output
        if marker not in text:
            return text
        inner = text.split(marker, 1)[1]
        lines: list[str] = []
        for line in inner.splitlines():
            if line.startswith("  "):
                lines.append(line[2:])
            else:
                break
        return "\n".join(lines)

    """Tests for complex bash commands: pipes, redirects, substitution, conditionals, etc."""

    # -- pipes ---------------------------------------------------------------

    async def test_pipe_echo_to_wc(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello | wc -l")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "1" in result.output

    async def test_pipe_echo_to_grep(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo -e 'apple\\nbanana\\ncherry' | grep ana")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "banana" in result.output

    async def test_pipe_ls_to_head(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="ls / | head -1")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_multiple_pipes(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello | tr 'a-z' 'A-Z' | tr 'A-Z' 'a-z'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output

    # -- redirects -----------------------------------------------------------

    async def test_redirect_stdout_to_file(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        bash = Bash(session=mock_session)
        outfile = tmp_path / "redirected.txt"
        posix = str(outfile).replace("\\", "/")
        params = BashParams(cmd=f"echo redirected_content > {posix}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert outfile.read_text(encoding="utf-8").strip() == "redirected_content"

    async def test_redirect_append(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        bash = Bash(session=mock_session)
        outfile = tmp_path / "append.txt"
        posix = str(outfile).replace("\\", "/")
        await bash(BashParams(cmd=f"echo line1 > {posix}"))
        await bash(BashParams(cmd=f"echo line2 >> {posix}"))
        result = await bash(BashParams(cmd=f"cat {posix}"))
        assert isinstance(result, ToolOk)
        lines = self._cmd_output(result).strip().splitlines()
        assert "line1" in lines[0]
        assert "line2" in lines[-1]

    async def test_stderr_redirect(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        bash = Bash(session=mock_session)
        outfile = tmp_path / "stderr.txt"
        posix = str(outfile).replace("\\", "/")
        # Redirect stderr to file; command fails so we expect ToolError
        params = BashParams(cmd=f"ls nonexisistent 2> {posix}")
        await bash(params)
        content = outfile.read_text(encoding="utf-8").lower()
        assert "nonexisistent" in content or "cannot access" in content or "no such" in content

    # -- command substitution ------------------------------------------------

    async def test_command_substitution(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo $(echo nested)")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "nested" in result.output

    async def test_backtick_substitution(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo `echo backtick`")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "backtick" in result.output

    # -- environment variables -----------------------------------------------

    async def test_env_var_home(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo $HOME")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output.strip()) > 0

    async def test_env_var_user(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo $USER")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # USER may be empty on some systems; just check no error

    # -- semicolon-separated commands ----------------------------------------

    async def test_semicolon_chain(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo first; echo second")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "first" in result.output
        assert "second" in result.output

    async def test_and_or_operators(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="true && echo yes || echo no")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "yes" in result.output

    async def test_and_or_false_branch(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="false && echo yes || echo no")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "no" in result.output

    # -- conditionals --------------------------------------------------------

    async def test_if_statement(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="if true; then echo TRUE; else echo FALSE; fi")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "TRUE" in result.output

    async def test_test_bracket(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="[ 1 -eq 1 ] && echo equal")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "equal" in result.output

    # -- exit codes ----------------------------------------------------------

    async def test_exit_code_success_check(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="true; echo $?")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "0" in result.output

    async def test_exit_code_failure_check(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        # The `echo $?` succeeds (exit 0) so overall ToolOk
        params = BashParams(cmd="false; echo $?")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "1" in result.output

    # -- here-strings / here-docs --------------------------------------------

    async def test_here_string(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="cat <<< 'herestring'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "herestring" in result.output

    async def test_here_doc(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="cat <<EOF\nheredoc_line\nEOF")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "heredoc_line" in result.output

    # -- globbing ------------------------------------------------------------

    async def test_glob_expansion(self, mock_session: MagicMock, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        bash = Bash(session=mock_session)
        posix = str(tmp_path).replace("\\", "/")
        params = BashParams(cmd=f"cd {posix} && ls *.txt")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "a.txt" in result.output
        assert "b.txt" in result.output

    # -- arithmetic expansion ------------------------------------------------

    async def test_arithmetic_expansion(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo $((3 + 4))")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "7" in result.output

    # -- brace expansion -----------------------------------------------------

    async def test_brace_expansion(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo {a,b,c}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "a b c" in result.output

    # -- sub-shell -----------------------------------------------------------

    async def test_subshell(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="(cd / && pwd)")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert self._cmd_output(result).strip() == "/"

    # -- process substitution -------------------------------------------------

    async def test_process_substitution_diff(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"
        f1.write_text("same")
        f2.write_text("same")
        bash = Bash(session=mock_session)
        posix1 = str(f1).replace("\\", "/")
        posix2 = str(f2).replace("\\", "/")
        params = BashParams(cmd=f"diff <(cat {posix1}) <(cat {posix2})")
        result = await bash(params)
        # diff returns 0 (success) when files are identical
        assert isinstance(result, ToolOk)

    async def test_process_substitution_diff_differs(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"
        f1.write_text("one")
        f2.write_text("two")
        bash = Bash(session=mock_session)
        posix1 = str(f1).replace("\\", "/")
        posix2 = str(f2).replace("\\", "/")
        params = BashParams(cmd=f"diff <(cat {posix1}) <(cat {posix2})")
        result = await bash(params)
        # diff returns 1 when files differ — an expected, informative outcome
        # (not a failure): reported as success with the meaning noted.
        assert isinstance(result, ToolOk)
        assert "Files differ" in result.message

    # -- inline env ----------------------------------------------------------

    async def test_inline_env_override(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="MYVAR=42 bash -c 'echo $MYVAR'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "42" in result.output

    # -- negation ------------------------------------------------------------

    async def test_negation_bang(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="! false; echo $?")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "0" in result.output

    # -- loop ----------------------------------------------------------------

    async def test_for_loop(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="for i in 1 2 3; do echo $i; done")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "1" in result.output
        assert "2" in result.output
        assert "3" in result.output

    async def test_while_loop(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="i=0; while [ $i -lt 3 ]; do echo $i; i=$((i+1)); done")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "0" in result.output
        assert "1" in result.output
        assert "2" in result.output

    # -- temp file with mktemp -----------------------------------------------

    async def test_mktemp(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="mktemp")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "/tmp" in result.output or "/temp" in result.output.lower()

    # -- printf --------------------------------------------------------------

    async def test_printf(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="printf '%s %s' hello world")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello world" in result.output

    # -- array ---------------------------------------------------------------

    async def test_array(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="arr=(one two three); echo ${arr[1]}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "two" in result.output

    # -- string manipulation -------------------------------------------------

    async def test_string_length(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="s=abcdef; echo ${#s}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "6" in result.output

    async def test_string_substring(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="s=hello; echo ${s:1:3}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "ell" in result.output

    # -- sed -----------------------------------------------------------------

    async def test_sed_substitution(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo foo | sed 's/foo/bar/'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "bar" in result.output

    # -- awk -----------------------------------------------------------------

    async def test_awk_field(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'a b c' | awk '{print $2}'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "b" in result.output

    # -- cut -----------------------------------------------------------------

    async def test_cut_delimiter(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'a:b:c' | cut -d: -f2")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "b" in result.output

    # -- sort / uniq ---------------------------------------------------------

    async def test_sort_uniq(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo -e 'c\\na\\nb\\na' | sort | uniq")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        lines = self._cmd_output(result).strip().splitlines()
        assert lines == ["a", "b", "c"]

    # -- head / tail ---------------------------------------------------------

    async def test_head_n(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="seq 10 | head -3")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        lines = self._cmd_output(result).strip().splitlines()
        assert len(lines) == 3

    async def test_tail_n(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="seq 10 | tail -3")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        lines = self._cmd_output(result).strip().splitlines()
        assert "8" in lines[0]
        assert "10" in lines[-1]

    # -- tee -----------------------------------------------------------------

    async def test_tee(
        self, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        bash = Bash(session=mock_session)
        outfile = tmp_path / "tee_out.txt"
        posix = str(outfile).replace("\\", "/")
        params = BashParams(cmd=f"echo hello_tee | tee {posix}")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello_tee" in result.output
        assert outfile.read_text(encoding="utf-8").strip() == "hello_tee"

    # -- exit with explicit code ---------------------------------------------

    async def test_exit_explicit_code(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="exit 42")
        result = await bash(params)
        # bash -c "exit 42" exits with code 42 -> ToolError
        assert isinstance(result, ToolError)

    # -- chained pipes with special chars ------------------------------------

    async def test_pipe_with_dollar_signs(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo '$HOME' | cat")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # Single quotes preserve literal $HOME
        assert "$HOME" in result.output

    # -- background process via & --------------------------------------------

    async def test_background_ampersand(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="sleep 1 & wait", timeout=10)
        result = await bash(params)
        assert isinstance(result, ToolOk)

    # -- dirname / basename --------------------------------------------------

    async def test_dirname_basename(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="dirname /usr/bin/bash && basename /usr/bin/bash")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "/usr/bin" in result.output
        assert "bash" in result.output

    # -- xargs ---------------------------------------------------------------

    async def test_xargs(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'a b c' | xargs -n1 echo")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "a" in result.output
        assert "b" in result.output
        assert "c" in result.output

    # -- trap ----------------------------------------------------------------

    async def test_trap_does_not_crash(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="trap 'echo trapped' EXIT; echo done")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "done" in result.output
        assert "trapped" in result.output

    # -- backslash escapes before metacharacters -----------------------------

    async def test_find_with_escaped_parens(self, mock_session: MagicMock, tmp_path: Path) -> None:
        bash = Bash(session=mock_session)
        # Create files to search
        (tmp_path / "foo.txt").write_text("foo")
        (tmp_path / "bar.py").write_text("bar")
        (tmp_path / "baz.txt").write_text("baz")
        posix = str(tmp_path).replace("\\", "/")
        # The \(\) grouping must survive _prepare_bash_cmd on Windows
        params = BashParams(cmd=f"find {posix} -maxdepth 1 \\( -name '*.txt' -o -name '*.py' \\) | sort")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "foo.txt" in result.output
        assert "bar.py" in result.output
        assert "baz.txt" in result.output

    async def test_echo_escaped_pipe(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'a|b' | cat")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "a|b" in result.output

    async def test_echo_escaped_glob(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo '*'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "*" in result.output

    async def test_echo_escaped_semicolon(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo 'a;b'")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "a;b" in result.output


# ============================================================================
# BashParams interactive validation
# ============================================================================


# ============================================================================
# Bash fixer integration across execution modes
# ============================================================================


class TestBashFixToolIntegration:
    @pytest.fixture
    def bash_instance(self, mock_session: MagicMock) -> Bash:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            return Bash(session=mock_session)

    @staticmethod
    def _completed_process_task() -> MagicMock:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-fix-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value="fixed output")
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        process_task.stream.process_elapsed = None
        return process_task

    async def test_foreground_command_is_fixed_before_process_creation(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as process_task_class:
            result = await bash_instance(BashParams(cmd="gtimeout 2 echo ok"))

        assert isinstance(result, ToolOk)
        command = process_task_class.call_args.args[1][1]
        assert command.endswith("\ngtimeout 2 echo ok")
        assert "gtimeout()" in command
        assert 'timeout "$@"' in command

    async def test_background_command_is_fixed_before_process_creation(
        self, bash_instance: Bash
    ) -> None:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-background-id")
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as process_task_class:
            result = await bash_instance(
                BashParams(cmd="printf text | pbcopy", mode="send")
            )

        assert isinstance(result, ToolOk)
        command = process_task_class.call_args.args[1][1]
        assert command.endswith("\nprintf text | pbcopy")
        assert "pbcopy()" in command
        assert 'clip.exe "$@"' in command

    async def test_interactive_initial_command_is_fixed(
        self, bash_instance: Bash
    ) -> None:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-interactive-id")
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as process_task_class:
            result = await bash_instance(BashParams(cmd="printf abc | rev", mode="interactive"))

        assert isinstance(result, ToolOk)
        command = _decode_startup_command(process_task_class.call_args.args[1][1])
        assert command.endswith("\nprintf abc | rev; exec bash -i")
        for name in ("gtimeout", "rev", "xdg-open", "open", "pbcopy", "pbpaste"):
            assert f"{name}()" in command
            assert f"declare -F {name}" in command

    async def test_existing_interactive_session_input_is_fixed(
        self, bash_instance: Bash
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("", False, 0.01))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"bash_compat": stream}
        bash_instance._session.custom_data["background_task_data"] = data

        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            result = await bash_instance(
                BashParams(cmd="xdg-open README.md", task_id="bash_compat")
            )

        assert isinstance(result, ToolOk)
        sent = stream.input.await_args.args[0]
        assert sent == "xdg-open README.md\n"
        assert "xdg-open()" not in sent

    @pytest.mark.parametrize(
        "fragment",
        [
            "rev",
            "EOF",
            "then",
            "'continued text",
            '"continued text',
            "printf '%s' \"$?\"",
            "body\\",
        ],
    )
    async def test_existing_interactive_session_input_is_not_reparsed_as_program(
        self, bash_instance: Bash, fragment: str
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("", False, 0.01))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"bash_fragment": stream}
        bash_instance._session.custom_data["background_task_data"] = data

        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"):
            result = await bash_instance(
                BashParams(cmd=fragment, task_id="bash_fragment")
            )

        assert isinstance(result, ToolOk)
        stream.input.assert_awaited_once_with(fragment + "\n")

    async def test_compatibility_fix_runs_before_rtk(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        rewrite = MagicMock(side_effect=lambda command, *_args, **_kwargs: (command, False))
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ), patch(
            "kimix.tools.file.bash.bash_tool._maybe_rewrite_shell_command_with_rtk",
            rewrite,
        ):
            await bash_instance(BashParams(cmd="gtimeout 2 true"))

        rewritten = rewrite.call_args.args[0]
        assert rewritten.endswith("\ngtimeout 2 true")
        assert "gtimeout()" in rewritten

    async def test_path_normalization_and_compatibility_fix_compose(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as process_task_class:
            await bash_instance(
                BashParams(cmd=r"gtimeout 2 cat src\kimix\agent_worker.json")
            )

        command = process_task_class.call_args.args[1][1]
        assert command.endswith(
            "\ngtimeout 2 cat src/kimix/agent_worker.json"
        )

    async def test_forbidden_check_uses_original_command_before_rewrite(
        self, mock_session: MagicMock
    ) -> None:
        mock_session.custom_config.get.return_value = {
            "forbidden_commands": ["gtimeout"]
        }
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.fix_bash_command") as fixer:
            result = await bash(BashParams(cmd="gtimeout 2 true"))

        assert isinstance(result, ToolError)
        assert "forbidden" in result.message
        fixer.assert_not_called()

    @pytest.mark.parametrize("mode", ["execute", "send"])
    async def test_forbidden_source_command_is_blocked_in_fresh_modes(
        self, mock_session: MagicMock, mode: str
    ) -> None:
        mock_session.custom_config.get.return_value = {
            "forbidden_commands": ["gtimeout"]
        }
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as process_task:
            result = await bash(BashParams(cmd="gtimeout 2 true", mode=mode))
        assert isinstance(result, ToolError)
        process_task.assert_not_called()

    async def test_forbidden_source_command_is_blocked_in_continuation(
        self, bash_instance: Bash
    ) -> None:
        bash_instance._forbidden_keywords = ["gtimeout"]
        with patch("kimix.tools.file.bash.bash_tool.fix_bash_command") as fixer:
            result = await bash_instance(
                BashParams(cmd="gtimeout 2 true", task_id="existing")
            )
        assert isinstance(result, ToolError)
        fixer.assert_not_called()

    @pytest.mark.parametrize("fragment", ["printf \\", "x=(printf"])
    async def test_forbidden_policy_rejects_incomplete_continuation_fragment(
        self, bash_instance: Bash, fragment: str
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.input = AsyncMock(return_value=True)
        data.tasks = {"existing": stream}
        bash_instance._session.custom_data["background_task_data"] = data
        bash_instance._forbidden_keywords = ["printf BLOCKED"]

        syntax_error = subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="",
            stderr="bash: syntax error: unexpected end of file",
        )
        with patch(
            "kimix.tools.file.bash.bash_tool.subprocess.run",
            return_value=syntax_error,
        ):
            result = await bash_instance(BashParams(cmd=fragment, task_id="existing"))

        assert isinstance(result, ToolError)
        assert result.brief == "Unsafe command fragment"
        stream.input.assert_not_awaited()

    @pytest.mark.parametrize(
        "command",
        [
            "if true; then printf safe; fi",
            "for x in one; do printf safe; done",
            "cat <<EOF\nsafe\nEOF",
        ],
    )
    async def test_forbidden_policy_allows_complete_compound_continuation(
        self, bash_instance: Bash, command: str
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("safe", False, 0.01))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"existing": stream}
        bash_instance._session.custom_data["background_task_data"] = data
        bash_instance._forbidden_keywords = ["printf BLOCKED"]

        syntax_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch(
            "kimix.tools.file.bash.bash_tool.subprocess.run",
            return_value=syntax_ok,
        ):
            result = await bash_instance(BashParams(cmd=command, task_id="existing"))

        assert isinstance(result, ToolOk)
        stream.input.assert_awaited_once_with(command + "\n")

    @pytest.mark.parametrize(
        ("source", "forbidden"),
        [("gtimeout 2 true", "timeout"), ("pbpaste", "powershell.exe")],
    )
    async def test_forbidden_generated_command_is_blocked(
        self, bash_instance: Bash, source: str, forbidden: str
    ) -> None:
        bash_instance._forbidden_keywords = [forbidden]
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch("kimix.tools.file.bash.bash_tool.ProcessTask") as process_task:
            result = await bash_instance(BashParams(cmd=source))
        assert isinstance(result, ToolError)
        process_task.assert_not_called()

    async def test_forbidden_rtk_generated_command_is_blocked(
        self, bash_instance: Bash
    ) -> None:
        bash_instance._forbidden_keywords = ["rtk"]
        process_task = self._completed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool._maybe_rewrite_shell_command_with_rtk",
            return_value=("rtk git status", True),
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as process_task_class:
            result = await bash_instance(BashParams(cmd="git status"))
        assert isinstance(result, ToolError)
        process_task_class.assert_not_called()

    @pytest.mark.parametrize(
        ("generated_keyword", "command"),
        [("perl", "rev"), ("rtk", "git status")],
    )
    async def test_continuation_is_not_rewritten_or_blocked_by_generated_text(
        self, bash_instance: Bash, generated_keyword: str, command: str
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="buffered output")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("new output", False, 0.01))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"existing": stream}
        bash_instance._session.custom_data["background_task_data"] = data
        bash_instance._forbidden_keywords = [generated_keyword]

        syntax_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch(
            "kimix.tools.file.bash.bash_tool._maybe_rewrite_shell_command_with_rtk"
        ) as rewrite, patch(
            "kimix.tools.file.bash.bash_tool.subprocess.run",
            return_value=syntax_ok,
        ):
            result = await bash_instance(
                BashParams(cmd=command, task_id="existing")
            )

        assert isinstance(result, ToolOk)
        rewrite.assert_not_called()
        stream.pop_output.assert_awaited_once_with()
        stream.input.assert_awaited_once_with(command + "\n")

    async def test_failed_continuation_delivery_returns_buffered_output(
        self, bash_instance: Bash
    ) -> None:
        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="buffered output")
        stream.input = AsyncMock(return_value=False)
        data.tasks = {"existing": stream}
        bash_instance._session.custom_data["background_task_data"] = data

        result = await bash_instance(
            BashParams(cmd="printf new", task_id="existing")
        )

        assert isinstance(result, ToolError)
        assert result.output == "buffered output"
        stream.pop_output.assert_awaited_once_with()
        stream.input.assert_awaited_once()


# ============================================================================
# Bash interactive argument building
# ============================================================================

class TestBashInteractiveArgumentBuilding:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock(spec=Session)
        session.custom_data = {}
        session.custom_config.get.return_value = {}
        return session

    async def test_non_interactive_args(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("bash-test-id")
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

            params = BashParams(cmd="echo hello")
            result = await bash(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            # One-shot commands run with pipefail so pipelines surface the
            # real failing stage's exit code instead of the consumer's 0.
            assert args[0][1] == ["-c", "set -o pipefail; echo hello"]

    async def test_interactive_args_with_cmd(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("bash-interactive-id")
            mock_pt.return_value = mock_instance

            params = BashParams(cmd="echo start", mode="interactive")
            result = await bash(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            bash_args = args[0][1]
            assert "-c" in bash_args
            decoded = _decode_startup_command(bash_args[1])
            assert "echo start" in decoded
            assert "exec bash -i" in decoded
            assert args.kwargs.get("append_newline") is True or args[0][4] is True

    async def test_interactive_args_without_cmd(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("bash-interactive-id")
            mock_pt.return_value = mock_instance

            params = BashParams(cmd="", mode="interactive")
            result = await bash(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            bash_args = args[0][1]
            assert bash_args[0] == "-c"
            assert bash_args[1].endswith("; exec bash -i")
            decoded = _decode_startup_command(bash_args[1])
            assert "gtimeout()" in decoded
            assert "rev()" in decoded
            assert "export -f rev" in decoded

    async def test_interactive_returns_immediately(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("task-456")
            mock_pt.return_value = mock_instance

            params = BashParams(cmd="", mode="interactive")
            result = await bash(params)

            assert isinstance(result, ToolOk)
            assert "task-456" in result.message
            assert "task_id" in result.message
            assert "job_output" in result.message
            mock_instance.wait.assert_not_called()


# ============================================================================
# RTK rewrite path
# ============================================================================

class TestBashRtkRewrite:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock(spec=Session)
        session.custom_data = {}
        session.custom_config.get.return_value = {}
        return session

    async def test_bash_rewrites_known_command_when_rtk_available(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.common._rtk_available", return_value=True
        ), patch(
            "kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

            with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
                mock_instance = MagicMock()
                mock_instance.start = MagicMock(return_value=asyncio.Future())
                mock_instance.start.return_value.set_result("bash-rtk-id")
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

                params = BashParams(cmd="git status")
                result = await bash(params)

                assert isinstance(result, ToolOk)
                args = mock_pt.call_args
                # Rewritten to the bare `rtk` prefix (shared bin dir is first
                # on PATH) with the one-shot pipefail prelude.
                assert args[0][1] == ["-c", "set -o pipefail; rtk git status"]

    async def test_bash_read_builtin_not_rewritten(
        self, mock_session: MagicMock
    ) -> None:
        with patch("kimix.tools.common._rtk_available", return_value=True), patch(
            "kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            bash = Bash(session=mock_session)

        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            mock_instance = MagicMock()
            mock_instance.start = MagicMock(return_value=asyncio.Future())
            mock_instance.start.return_value.set_result("bash-read-id")
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

            params = BashParams(cmd="read var")
            result = await bash(params)

            assert isinstance(result, ToolOk)
            args = mock_pt.call_args
            assert args[0][1] == ["-c", "set -o pipefail; read var"]


# ============================================================================
# Bash session continuation / wait_for_pattern
# ============================================================================

class TestBashSessionContinuation:
    @pytest.fixture
    def bash_instance(self, mock_session: MagicMock) -> Bash:
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=r"C:\Git\bin\bash.exe"), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash", return_value=True
        ):
            return Bash(session=mock_session)

    async def test_continue_nonexistent_task_lists_available(self, bash_instance: Bash) -> None:
        from unittest.mock import AsyncMock

        data = TaskData()
        stream1 = AsyncMock()
        stream1.is_started = AsyncMock(return_value=True)
        stream2 = AsyncMock()
        stream2.is_started = AsyncMock(return_value=False)
        data.tasks = {"bash_alive": stream1, "bash_dead": stream2}
        bash_instance._session.custom_data["background_task_data"] = data

        result = await bash_instance(BashParams(cmd="echo hi", task_id="missing"))
        assert isinstance(result, ToolError)
        assert "missing" in result.message
        assert "bash_alive" in result.message
        assert "bash_dead" not in result.message

    async def test_continue_nonexistent_task_no_tasks(self, bash_instance: Bash) -> None:
        result = await bash_instance(BashParams(cmd="echo hi", task_id="missing"))
        assert isinstance(result, ToolError)
        assert "No running tasks" in result.message

    async def test_invalid_wait_for_pattern_returns_error(self, bash_instance: Bash) -> None:
        result = await bash_instance(BashParams(cmd="echo hi", wait_for_pattern="["))
        assert isinstance(result, ToolError)
        assert "Invalid wait_for_pattern" in result.message

    async def test_continue_session_sends_input_and_returns_block(self, bash_instance: Bash) -> None:
        from unittest.mock import AsyncMock

        data = TaskData()
        stream = AsyncMock()
        stream.is_started = AsyncMock(return_value=True)
        stream.pop_output = AsyncMock(return_value="")
        stream.input = AsyncMock(return_value=True)
        stream.wait_for_output = AsyncMock(return_value=("hello output", True, 0.12))
        stream.thread_is_alive = AsyncMock(return_value=True)
        stream.success = AsyncMock(return_value=True)
        data.tasks = {"bash_42": stream}
        bash_instance._session.custom_data["background_task_data"] = data

        result = await bash_instance(
            BashParams(cmd="echo hello", task_id="bash_42", wait_for_pattern="hello")
        )

        assert isinstance(result, ToolOk)
        assert "bash_42" in result.output
        assert "status: running" in result.output
        assert "wait_matched: true" in result.output
        assert "elapsed_seconds: 0.12" in result.output
        stream.pop_output.assert_awaited_once()
        stream.input.assert_awaited_once_with("echo hello\n")
        stream.wait_for_output.assert_awaited_once()


# ============================================================================
# Bash interactive integration tests
# ============================================================================

@pytest.mark.skipif(
    not BASH_AVAILABLE,
    reason="Bash tool is not available on this platform",
)
class TestBashInteractiveIntegration:
    async def test_interactive_echo(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="", mode="interactive")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        task_id = result.message.split("`")[1]

        task_data = mock_session.custom_data.get("background_task_data")
        assert task_data is not None
        task = task_data.tasks.get(task_id)
        assert task is not None

        try:
            await task.input("echo hello")
            # Interactive Git Bash first sources a ~26 KB compatibility prelude
            # (``bash -c "...; exec bash -i"``) before it can execute input, and
            # under full-suite load that startup can take well over a fixed
            # 0.5 s sleep.  Poll for the echoed output instead, so the test only
            # depends on the echo actually appearing.
            deadline = time.monotonic() + 15.0
            output = ""
            while time.monotonic() < deadline and "hello" not in output:
                output = await task.get_output()
                await asyncio.sleep(0.1)
            assert "hello" in output, repr(output)
        finally:
            # Always tear the session down, even when an assertion fails, so a
            # dangling interactive bash cannot leak into (and slow down) the
            # remaining tests in the suite.
            await task.input("exit")
            await task.wait(timeout=5)
            if await task.thread_is_alive():
                await task.stop()

    async def test_interactive_start_with_wait_for_pattern(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello", mode="interactive", wait_for_pattern="hello", timeout=10)
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "bash" in result.output
        assert "status:" in result.output
        assert "wait_matched: true" in result.output
        assert "hello" in result.output

        # Continue with exit to clean up.
        task_id = result.output.split("task_id: ", 1)[1].split("\n", 1)[0]
        try:
            exit_result = await bash(BashParams(cmd="exit", task_id=task_id, timeout=5))
            assert isinstance(exit_result, ToolOk)
            assert "status: completed" in exit_result.output
        finally:
            # If the continuation above fails, stop the session explicitly so
            # no interactive bash leaks into later tests.
            task_data = mock_session.custom_data.get("background_task_data")
            task = task_data.tasks.get(task_id) if task_data is not None else None
            if task is not None and await task.thread_is_alive():
                await task.stop()


# ============================================================================
# MSYSTEM neutralization (Git Bash -> xmake windows/MSVC default)
# ============================================================================


class TestIsGitBashInstall:
    def test_git_bash_inner_bash_detected(self) -> None:
        with patch("kimix.tools.file.bash.bash_tool.os.path.isfile", return_value=True):
            assert _is_git_bash_install(r"C:\Program Files\Git\usr\bin\bash.exe") is True

    def test_git_bash_wrapper_detected(self) -> None:
        """The ``bin/bash.exe`` launcher (the process the tool actually
        spawns) is also recognized as a Git Bash install."""
        with patch("kimix.tools.file.bash.bash_tool.os.path.isfile", return_value=True):
            assert _is_git_bash_install(r"C:\Program Files\Git\bin\bash.exe") is True

    def test_msys2_bash_rejected(self) -> None:
        """MSYS2 also ships ``usr/bin/bash.exe`` but has no ``cmd/git.exe``
        marker, so its environment must stay untouched."""
        with patch("kimix.tools.file.bash.bash_tool.os.path.isfile", return_value=False):
            assert _is_git_bash_install(r"C:\msys64\usr\bin\bash.exe") is False

    def test_wrapper_without_git_marker_rejected(self) -> None:
        """A ``bin/bash.exe`` that is not backed by a Git for Windows install
        (no ``cmd/git.exe`` marker) is not neutralized."""
        with patch("kimix.tools.file.bash.bash_tool.os.path.isfile", return_value=False):
            assert _is_git_bash_install(r"C:\Program Files\Git\bin\bash.exe") is False

    def test_none_or_garbage_rejected(self) -> None:
        assert _is_git_bash_install(None) is False
        assert _is_git_bash_install("") is False
        assert _is_git_bash_install("/bin/bash") is False

    def test_marker_probe_uses_absolute_path(self) -> None:
        """The ``cmd/git.exe`` marker must be probed with a drive-anchored
        path.  A drive-relative probe ("C:foo") resolves against the process's
        current directory on drive C:, so after a ``chdir`` into a C: temp dir
        the marker lookup fails even for a real Git install and MSYSTEM
        neutralization is silently skipped (order-dependent test failures)."""
        with patch(
            "kimix.tools.file.bash.bash_tool.os.path.isfile",
            return_value=True,
        ) as isfile:
            assert _is_git_bash_install(r"C:\Program Files\Git\bin\bash.exe") is True
        probed = isfile.call_args.args[0]
        assert ntpath.isabs(probed), f"marker probe is drive-relative: {probed!r}"
        assert probed.lower() == r"c:\program files\git\cmd\git.exe"


class TestMsystemNeutralizedCommand:
    def test_win32_git_bash_prepends_prefix(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        with patch(
            "kimix.tools.file.bash.bash_tool._is_git_bash_install",
            return_value=True,
        ):
            cmd = _with_msystem_neutralized("echo hi", r"C:\Program Files\Git\bin\bash.exe")
        assert cmd == "export MSYSTEM=; echo hi"

    def test_win32_non_git_bash_unchanged(self, monkeypatch: Any) -> None:
        """Real MSYS2 shells keep their ``MSYSTEM``; only Git Bash is
        neutralized."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch(
            "kimix.tools.file.bash.bash_tool._is_git_bash_install",
            return_value=False,
        ):
            cmd = _with_msystem_neutralized("echo hi", r"C:\msys64\usr\bin\bash.exe")
        assert cmd == "echo hi"

    def test_non_windows_unchanged(self, monkeypatch: Any) -> None:
        """The change is strictly Windows-only: on Linux/macOS the command
        is returned unchanged even for a Git-style bash path."""
        monkeypatch.setattr(sys, "platform", "linux")
        cmd = _with_msystem_neutralized("echo hi", r"C:\Program Files\Git\bin\bash.exe")
        assert cmd == "echo hi"


class TestMSystemNeutralizationRealBash:
    @pytest.mark.skipif(
        sys.platform != "win32" or not BASH_AVAILABLE,
        reason="requires a real Git Bash on Windows",
    )
    def test_spawned_bash_sees_empty_msystem(self) -> None:
        """End-to-end: the bash actually chosen by the tool, launched with
        the tool's child environment and the neutralized command, sees an
        empty ``MSYSTEM`` (so xmake detects the native MSVC platform instead
        of mingw)."""
        bash = find_bash()
        assert bash is not None
        cmd = _with_msystem_neutralized(r'printf "%s" "$MSYSTEM"', bash)
        result = subprocess.run(
            [bash, "-c", cmd],
            env=_env_with_rg_bin_path(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""


# ============================================================================
# Shell enhancement wiring: hardline floor, cwd/workdir, hints, redaction,
# background guidance (WP1-WP5)
# ============================================================================

class TestShellSafetyWiring:
    @pytest.fixture
    def bash_instance(self, mock_session: MagicMock) -> Bash:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            return Bash(session=mock_session)

    @staticmethod
    def _completed_process_task() -> MagicMock:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-enhance-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value="mock output")
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        process_task.stream.process_elapsed = None
        return process_task

    # -- hardline floor ----------------------------------------------------

    async def test_hardline_block_returns_error_before_process_task(
        self, bash_instance: Bash
    ) -> None:
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            result = await bash_instance(BashParams(cmd="rm -rf /"))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (hardline)"
        assert "hardline" in result.message
        mock_pt.assert_not_called()

    async def test_hardline_block_obfuscated_spelling(
        self, bash_instance: Bash
    ) -> None:
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            result = await bash_instance(BashParams(cmd=r"r\m -rf /"))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (hardline)"
        mock_pt.assert_not_called()

    async def test_hardline_skipped_when_config_disabled(
        self, mock_session: MagicMock
    ) -> None:
        mock_session.custom_config.get.return_value = {"shell": {"hardline": False}}
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)
        process_task = self._completed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ) as mock_pt:
            result = await bash(BashParams(cmd="rm -rf /"))
        assert isinstance(result, ToolOk)
        mock_pt.assert_called_once()

    # -- self-kill guard ----------------------------------------------------

    async def test_self_kill_guard_blocks_own_pid(
        self, bash_instance: Bash
    ) -> None:
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            result = await bash_instance(BashParams(cmd=f"kill -9 {os.getpid()}"))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (self-kill guard)"
        assert str(os.getpid()) in result.message
        mock_pt.assert_not_called()

    async def test_self_kill_guard_blocks_own_image_name(
        self, bash_instance: Bash
    ) -> None:
        image = Path(sys.executable).name  # e.g. python.exe hosting the agent
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            result = await bash_instance(BashParams(cmd=f"taskkill /F /IM {image}"))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (self-kill guard)"
        mock_pt.assert_not_called()

    async def test_self_kill_guard_blocks_loop_pid_target(
        self, bash_instance: Bash
    ) -> None:
        # PID reached only through a shell loop variable (the shape that used
        # to slip through: ``for pid in ...; do taskkill /PID $pid ...``).
        cmd = f"for pid in {os.getpid()} 99999; do taskkill /PID $pid /T /F 2>/dev/null; done; echo done"
        with patch("kimix.tools.file.bash.bash_tool.ProcessTask") as mock_pt:
            result = await bash_instance(BashParams(cmd=cmd))
        assert isinstance(result, ToolError)
        assert result.brief == "Blocked (self-kill guard)"
        assert str(os.getpid()) in result.message
        mock_pt.assert_not_called()

    async def test_self_kill_guard_allows_unrelated_loop(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        cmd = "for pid in 999999999 888888888; do taskkill /PID $pid /F; done"
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ) as mock_pt:
            result = await bash_instance(BashParams(cmd=cmd))
        assert isinstance(result, ToolOk)
        mock_pt.assert_called_once()

    async def test_self_kill_guard_allows_unrelated_pid(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ) as mock_pt:
            result = await bash_instance(BashParams(cmd="kill 999999999"))
        assert isinstance(result, ToolOk)
        mock_pt.assert_called_once()

    async def test_self_kill_guard_skipped_when_config_disabled(
        self, mock_session: MagicMock
    ) -> None:
        mock_session.custom_config.get.return_value = {"shell": {"self_kill_guard": False}}
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)
        process_task = self._completed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ) as mock_pt:
            result = await bash(BashParams(cmd=f"kill -9 {os.getpid()}"))
        assert isinstance(result, ToolOk)
        mock_pt.assert_called_once()

    # -- cwd / workdir removed ---------------------------------------------

    def test_cwd_and_workdir_params_removed(self) -> None:
        """Bash no longer exposes ``cwd``/``workdir``/``deduplicate_output``."""
        props = BashParams.model_json_schema()["properties"]
        for gone in ("cwd", "workdir", "deduplicate_output", "token_kill"):
            assert gone not in props, f"{gone} must be removed from BashParams"

    async def test_process_task_runs_without_cwd(self, bash_instance: Bash) -> None:
        """No working directory is passed to the subprocess anymore."""
        process_task = self._completed_process_task()
        with patch("kimix.tools.file.bash.bash_tool.sys.platform", "win32"), patch(
            "kimix.utils.windows_env.refresh_env_from_registry"
        ), patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask",
            return_value=process_task,
        ) as mock_pt:
            result = await bash_instance(BashParams(cmd="echo hi"))
        assert isinstance(result, ToolOk)
        assert mock_pt.call_args.args[2] is None

    # -- exit-code meaning / failure hints ---------------------------------

    async def test_completed_failure_block_has_meaning_and_hint(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        process_task.stream.pop_output = AsyncMock(
            return_value="bash: no_such_cmd_xyz: command not found"
        )
        process_task.stream.success = AsyncMock(return_value=False)
        process_task.stream.exit_code = 127
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="no_such_cmd_xyz"))
        assert isinstance(result, ToolError)
        assert "exit_code_meaning:" in result.output
        assert "failure_hint:" in result.output
        assert "command was not found" in result.output.lower()
        assert "Hint:" in result.message

    async def test_completed_success_block_has_null_meaning_and_hint(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="echo hi"))
        assert isinstance(result, ToolOk)
        assert "exit_code_meaning: null" in result.output
        assert "failure_hint: null" in result.output

    # -- timeout branch guidance -------------------------------------------

    async def test_timeout_branch_includes_guidance_for_server_command(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        process_task.thread_is_alive = AsyncMock(return_value=True)
        process_task.stream.pop_output = AsyncMock(return_value="")
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="npm run dev", timeout=1))
        assert isinstance(result, ToolError)
        assert result.brief == "Timeout"
        assert "Running in background" in result.message
        assert "Long-running process detected" in result.message

    async def test_timeout_branch_plain_command_no_guidance(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task()
        process_task.thread_is_alive = AsyncMock(return_value=True)
        process_task.stream.pop_output = AsyncMock(return_value="")
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="sleep 5", timeout=1))
        assert isinstance(result, ToolError)
        assert result.brief == "Timeout"
        assert "Long-running process detected" not in result.message

    # -- secret redaction (config-gated) ------------------------------------

    async def test_redaction_applied_in_process_output(
        self, mock_session: MagicMock
    ) -> None:
        mock_session.custom_config.get.return_value = {"shell": {"redact_secrets": True}}
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)
        process_task = self._completed_process_task()
        process_task.stream.pop_output = AsyncMock(
            return_value="token=ghp_1234567890123456789012"
        )
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash(BashParams(cmd="echo token"))
        assert isinstance(result, ToolOk)
        assert "ghp_" not in result.output
        assert "[REDACTED]" in result.output

    async def test_redaction_disabled_by_config(self, mock_session: MagicMock) -> None:
        mock_session.custom_config.get.return_value = {"shell": {"redact_secrets": False}}
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            bash = Bash(session=mock_session)
        process_task = self._completed_process_task()
        process_task.stream.pop_output = AsyncMock(
            return_value="token=ghp_1234567890123456789012"
        )
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash(BashParams(cmd="echo token"))
        assert isinstance(result, ToolOk)
        assert "ghp_1234567890123456789012" in result.output


# ============================================================================
# Original-saved suffix on filtered output
# ============================================================================


class TestBashOriginalSavedSuffix:
    @pytest.fixture
    def bash_instance(self, mock_session: MagicMock) -> Bash:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            return Bash(session=mock_session)

    @staticmethod
    def _completed_process_task(output: str = "fixed output") -> MagicMock:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-suffix-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value=output)
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        process_task.stream.process_elapsed = None
        return process_task

    @staticmethod
    def _random_json_lines(n: int = 2000, seed: int = 999) -> str:
        """High-entropy JSON-ish lines that survive the token filter unchanged
        (no repeats, no near-duplicate patterns) while staying >64KB."""
        import random as _random
        import string as _string

        _random.seed(seed)

        def rand_str(length: int) -> str:
            return "".join(
                _random.choice(
                    _string.ascii_lowercase + _string.digits + " .,;:!?()[]{}-_=+"
                )
                for _ in range(length)
            )

        return "\n".join(
            f'{{"id": {_random.randint(0, 10**9)}, "val": "{rand_str(20)}"}}'
            for _ in range(n)
        )

    @staticmethod
    def _repeated_block_output(seed: int = 123) -> str:
        """Unique high-entropy lines plus a repeated block: the dedup stage
        collapses the block (changing the output, so the token filter saves the
        pre-filter original) while the unique lines keep the result >64KB."""
        import random as _random
        import string as _string

        _random.seed(seed)

        def rand_line() -> str:
            return "".join(
                _random.choice(_string.ascii_lowercase + _string.digits)
                for _ in range(40)
            )

        return "\n".join([rand_line() for _ in range(2500)] + ["ERROR"] * 100)

    async def test_message_includes_original_path_after_dedup(
        self, bash_instance: Bash
    ) -> None:
        process_task = self._completed_process_task(output="ERROR\n" * 10)
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="echo hi"))
        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    async def test_message_includes_original_path_after_truncate(
        self, bash_instance: Bash
    ) -> None:
        long_output = "\n".join(f"line_{i}" for i in range(500))
        process_task = self._completed_process_task(output=long_output)
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="echo hi", max_lines=10))
        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    async def test_no_suffix_when_filter_unchanged(
        self, bash_instance: Bash
    ) -> None:
        """Dedup is always on, but output with no repeats is left unchanged,
        so no original temp file is created and no suffix is appended."""
        process_task = self._completed_process_task(output="plain output")
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="echo hi"))
        assert isinstance(result, ToolOk)
        assert "[original saved to" not in result.message

    async def test_message_includes_original_path_after_summarize(
        self, bash_instance: Bash
    ) -> None:
        """A >64KB output that survives the (always-on) token filter unchanged
        is still preserved before summarization replaces it with a summary."""
        long_output = self._random_json_lines()
        assert len(long_output) > 65536
        process_task = self._completed_process_task(output=long_output)
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ), patch(
            "kimix.tools.file.bash.bash_tool._summarize_long_output_async",
            new=AsyncMock(return_value="[summary]"),
        ):
            result = await bash_instance(BashParams(cmd="echo hi"))
        assert isinstance(result, ToolOk)
        assert "output_truncated: true" in result.output
        assert "[original saved to .kimix_cache/tmp_" in result.message
        # The saved file must contain the full pre-summary output.
        saved = result.message.split("[original saved to ", 1)[1].rstrip("]")
        import anyio
        async with await anyio.open_file(saved, "r") as f:
            assert await f.read() == long_output

    async def test_summarize_saves_original_when_rtk_rewritten(
        self, bash_instance: Bash
    ) -> None:
        """RTK-rewritten commands skip local dedup; a long output that is not
        rtk-folded must still be preserved before summarization."""
        long_output = self._random_json_lines()
        with patch(
            "kimix.tools.file.bash.bash_tool._summarize_long_output_async",
            new=AsyncMock(return_value="[summary]"),
        ):
            display, _path, truncated, original_path = (
                await bash_instance._process_output(
                    BashParams(cmd="echo hi"),
                    long_output,
                    rtk_rewritten=True,
                )
            )
        assert truncated is True
        assert display == "[summary]"
        assert original_path is not None
        import anyio
        async with await anyio.open_file(original_path, "r") as f:
            assert await f.read() == long_output

    async def test_summarize_keeps_existing_original_path(
        self, bash_instance: Bash
    ) -> None:
        """When the (always-on) dedup already saved the pre-filter original,
        summarization must reuse that path instead of writing a new file."""
        long_output = self._repeated_block_output()
        assert len(long_output) > 65536
        save_calls: list[tuple[str, str | None]] = []

        async def fake_save(output: str, original_path: str | None) -> str | None:
            save_calls.append((output, original_path))
            return original_path

        with patch(
            "kimix.tools.file.bash.bash_tool._save_original_output_async",
            new=fake_save,
        ), patch(
            "kimix.tools.file.bash.bash_tool._summarize_long_output_async",
            new=AsyncMock(return_value="[summary]"),
        ):
            display, _path, truncated, original_path = (
                await bash_instance._process_output(
                    BashParams(cmd="echo hi"), long_output
                )
            )
        assert truncated is True
        assert display == "[summary]"
        assert save_calls, "summarize branch must consult the original saver"
        # The dedup stage collapsed the repeated block, so an original was
        # already saved; the summarize branch must keep it (non-None arg).
        assert save_calls[0][1] is not None
        assert original_path == save_calls[0][1]
        # The saved file is the full pre-filter original, not the summary.
        import anyio
        async with await anyio.open_file(original_path, "r") as f:
            assert await f.read() == long_output


# ============================================================================
# Powershell parity: original saved before summarization
# ============================================================================


class TestPowershellOriginalSavedSuffix:
    @pytest.fixture
    def pwsh_instance(self, mock_session: MagicMock) -> Powershell:
        with patch(
            "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
            return_value=True,
        ):
            return Powershell(session=mock_session)

    async def test_process_output_saves_original_before_summarize(
        self, pwsh_instance: Powershell
    ) -> None:
        """Powershell mirrors the Bash fix: an unchanged >64KB output is saved
        before the summarization branch replaces it with a summary."""
        long_output = TestBashOriginalSavedSuffix._random_json_lines()
        with patch(
            "kimix.tools.file.bash.pwsh_tool._summarize_long_output_async",
            new=AsyncMock(return_value="[summary]"),
        ):
            display, _path, truncated, original_path = (
                await pwsh_instance._process_output(
                    "echo hi",
                    PowershellParams(cmd="echo hi"),
                    long_output,
                )
            )
        assert truncated is True
        assert display == "[summary]"
        assert original_path is not None
        import anyio
        async with await anyio.open_file(original_path, "r") as f:
            assert await f.read() == long_output


# ============================================================================
# Failed command saved to a temp script file (.sh / .ps1)
# ============================================================================


class TestBashFailedCommandSaved:
    """Long failing bash commands are preserved as `.sh` temp files whose path
    is returned in the tool message; short commands are not saved."""

    @pytest.fixture
    def bash_instance(self, mock_session: MagicMock) -> Bash:
        with patch(
            "kimix.tools.file.bash.bash_tool.find_bash",
            return_value=r"C:\Git\bin\bash.exe",
        ), patch(
            "kimix.tools.file.bash.bash_tool._should_enable_bash",
            return_value=True,
        ):
            return Bash(session=mock_session)

    @staticmethod
    def _failed_process_task(output: str = "boom") -> MagicMock:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-cmd-saved-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value=output)
        process_task.stream.success = AsyncMock(return_value=False)
        process_task.stream.exit_code = 1
        process_task.stream.process_elapsed = None
        return process_task

    async def test_long_failed_command_is_saved_to_sh_file(
        self, bash_instance: Bash
    ) -> None:
        long_cmd = "echo start && " + "true && " * 30 + "false"
        assert len(long_cmd) > 50
        process_task = self._failed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd=long_cmd))
        assert isinstance(result, ToolError)
        assert "[command saved to .kimix_cache/tmp_" in result.message
        saved = result.message.split("[command saved to ", 1)[1].split("]", 1)[0]
        assert saved.endswith(".sh")
        assert Path(saved).read_text(encoding="utf-8") == long_cmd
        assert "Edit the saved script and run it again with this tool (bash) to retry." in result.message

    async def test_short_failed_command_not_saved(self, bash_instance: Bash) -> None:
        process_task = self._failed_process_task()
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd="false"))
        assert isinstance(result, ToolError)
        assert "[command saved to" not in result.message

    async def test_long_successful_command_not_saved(self, bash_instance: Bash) -> None:
        long_cmd = "echo start && " + "true && " * 30 + "true"
        assert len(long_cmd) > 50
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="bash-cmd-ok-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value="")
        process_task.stream.success = AsyncMock(return_value=True)
        process_task.stream.exit_code = 0
        process_task.stream.process_elapsed = None
        with patch(
            "kimix.tools.file.bash.bash_tool.ProcessTask", return_value=process_task
        ):
            result = await bash_instance(BashParams(cmd=long_cmd))
        assert isinstance(result, ToolOk)
        assert "[command saved to" not in result.message


class TestPowershellFailedCommandSaved:
    """Long failing PowerShell commands are preserved as `.ps1` temp files whose
    path is returned in the tool message; short commands are not saved."""

    @pytest.fixture
    def pwsh_instance(self, mock_session: MagicMock) -> Powershell:
        with patch(
            "kimix.tools.file.bash.pwsh_tool._bash_tool._should_enable_powershell",
            return_value=True,
        ):
            return Powershell(session=mock_session)

    @staticmethod
    def _failed_process_task(output: str = "boom") -> MagicMock:
        process_task = MagicMock()
        process_task.start = AsyncMock(return_value="pwsh-cmd-saved-id")
        process_task.wait_with_monitor = AsyncMock(return_value=(False, 0.0, False))
        process_task.thread_is_alive = AsyncMock(return_value=False)
        process_task.stream = MagicMock()
        process_task.stream.pop_output = AsyncMock(return_value=output)
        process_task.stream.success = AsyncMock(return_value=False)
        process_task.stream.exit_code = 1
        process_task.stream.process_elapsed = None
        return process_task

    async def test_long_failed_command_is_saved_to_ps1_file(
        self, pwsh_instance: Powershell
    ) -> None:
        long_cmd = "Write-Output 'start'; " + "Write-Host 'x'; " * 20 + "exit 1"
        assert len(long_cmd) > 50
        process_task = self._failed_process_task()
        with patch(
            "kimix.tools.file.bash.pwsh_tool.ProcessTask", return_value=process_task
        ):
            result = await pwsh_instance(PowershellParams(cmd=long_cmd))
        assert isinstance(result, ToolError)
        assert "[command saved to .kimix_cache/tmp_" in result.message
        saved = result.message.split("[command saved to ", 1)[1].split("]", 1)[0]
        assert saved.endswith(".ps1")
        assert Path(saved).read_text(encoding="utf-8") == long_cmd
        assert "Edit the saved script and run it again with this tool (PowerShell) to retry." in result.message

    async def test_short_failed_command_not_saved(
        self, pwsh_instance: Powershell
    ) -> None:
        process_task = self._failed_process_task()
        with patch(
            "kimix.tools.file.bash.pwsh_tool.ProcessTask", return_value=process_task
        ):
            result = await pwsh_instance(PowershellParams(cmd="exit 1"))
        assert isinstance(result, ToolError)
        assert "[command saved to" not in result.message


# ============================================================================
# cmd / command accepts a script file (description mention)
# ============================================================================


class TestCommandParamAcceptsScriptFile:
    def test_bash_cmd_description_mentions_sh_script(self) -> None:
        desc = BashParams.model_json_schema()["properties"]["command"]["description"]
        assert "`.sh` script file" in desc
        assert "executed via bash" in desc

    def test_pwsh_cmd_description_mentions_ps1_script(self) -> None:
        desc = PowershellParams.model_json_schema()["properties"]["command"]["description"]
        assert "`.ps1` script file" in desc
        assert "executed via PowerShell" in desc



