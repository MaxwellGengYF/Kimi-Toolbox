"""Comprehensive tests for the Bash tool (bash_tool.py) which uses the system bash executable."""

import asyncio
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from kimi_agent_sdk import ToolError, ToolOk
from kimi_cli.session import Session

from kimi_cli.tools import SkipThisTool
from kimix.tools.file.bash import (
    Bash,
    BashParams,
    Powershell,
)
from kimix.tools.file.bash.pwsh_tool import PowershellParams, find_pwsh
from kimix.tools.file.bash.bash_tool import (
    find_bash,
    _configured_shell,
    _prepare_bash_cmd,
    _find_git_bash_windows,
    _git_bash_candidate_from_git_path,
    _git_bash_candidates_from_exec_path,
    _git_exec_path,
    _git_install_root_from_exec_path,
    _where_git_executables,
)
from kimix.tools.background.utils import TaskData, _pop_task_data


def _bash_is_available() -> bool:
    """Return True when a real bash executable exists on this host.

    Host-capability probe: the agent config's ``shell`` selection is ignored
    so the result depends only on whether bash is actually installed (the
    shipped agent_worker.json selects ``bash``, so real sessions enable the
    Bash tool wherever this probe is True).
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
        ):
            assert _find_git_bash_windows() == r"C:\Program Files\Git\bin\bash.exe"


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

class TestBashParams:
    def test_defaults(self) -> None:
        p = BashParams(cmd="ls")
        assert p.cmd == "ls"
        assert p.timeout == 30
        assert p.deduplicate_output is True
        assert p.mode == "execute"

    def test_full(self) -> None:
        p = BashParams(cmd="cat -n file.txt", timeout=30)
        assert p.cmd == "cat -n file.txt"
        assert p.timeout == 30

    def test_timeout_min(self) -> None:
        # timeout=1 is now valid (ge=1)
        p = BashParams(cmd="ls", timeout=1)
        assert p.timeout == 1

    def test_timeout_max(self) -> None:
        with pytest.raises(Exception):
            BashParams(cmd="ls", timeout=901)

    def test_accepts_command_alias(self) -> None:
        p = BashParams(command="echo hello")
        assert p.cmd == "echo hello"

    def test_mode_send_without_task_id_valid(self) -> None:
        p = BashParams(cmd="echo hi", mode="send")
        assert p.mode == "send"

    def test_mode_send_with_task_id_succeeds(self) -> None:
        p = BashParams(cmd="hi", task_id="bash_0")
        assert p.mode == "execute"
        assert p.task_id == "bash_0"

    def test_task_id_with_any_mode_valid(self) -> None:
        p = BashParams(cmd="input", mode="execute", task_id="bash_0")
        assert p.mode == "execute"
        assert p.task_id == "bash_0"

    def test_run_alias_execute(self) -> None:
        p = BashParams(cmd="echo hi", mode="run")
        assert p.mode == "execute"

    def test_background_alias_send(self) -> None:
        p = BashParams(cmd="echo hi", mode="background")
        assert p.mode == "send"

    def test_deduplicate_output_accepts_token_kill_alias(self) -> None:
        p = BashParams(cmd="ls", token_kill=False)
        assert p.deduplicate_output is False

    def test_deduplicate_output_default_true(self) -> None:
        p = BashParams(cmd="ls")
        assert p.deduplicate_output is True

    def test_max_lines_field(self) -> None:
        p = BashParams(cmd="ls", max_lines=50)
        assert p.max_lines == 50
        p2 = BashParams(cmd="ls", max_lines=None)
        assert p2.max_lines is None


# ============================================================================
# _quote_for_bash_c
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

    def test_win32_description_mentions_slash_conversion(
        self, mock_session: MagicMock
    ) -> None:
        tool = self._make_tool("win32", mock_session)
        assert "auto-converted to forward slashes" in tool.description
        assert "backslashes inside quotes are preserved" in tool.description
        assert "`cat src\\a.py` → `cat src/a.py`" in tool.description

    def test_non_windows_description_has_no_windows_rule(
        self, mock_session: MagicMock
    ) -> None:
        tool = self._make_tool("linux", mock_session)
        assert "auto-converted to forward slashes" not in tool.description
        assert "backslashes inside quotes are preserved" not in tool.description


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

    async def test_empty_command(self, mock_session: MagicMock) -> None:
        # Empty commands in execute mode are rejected by the params model
        # itself before the tool ever runs.
        with pytest.raises(ValidationError, match="cmd cannot be empty"):
            BashParams(cmd="", timeout=5)

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
class TestPowershellParams:
    """Tests for PowershellParams model."""

    def test_defaults(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem")
        assert p.cmd == "Get-ChildItem"
        assert p.timeout == 30
        assert p.deduplicate_output is True
        assert p.mode == "execute"

    def test_accepts_command_alias(self) -> None:
        p = PowershellParams(command="Get-ChildItem")
        assert p.cmd == "Get-ChildItem"

    def test_mode_send_without_task_id_valid(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem", mode="send")
        assert p.mode == "send"

    def test_mode_send_with_task_id_succeeds(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem", task_id="pwsh_0")
        assert p.mode == "execute"
        assert p.task_id == "pwsh_0"

    def test_task_id_with_any_mode_valid(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem", mode="execute", task_id="pwsh_0")
        assert p.mode == "execute"
        assert p.task_id == "pwsh_0"

    def test_run_alias_execute(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem", mode="run")
        assert p.mode == "execute"

    def test_background_alias_send(self) -> None:
        p = PowershellParams(cmd="Get-ChildItem", mode="background")
        assert p.mode == "send"

    def test_deduplicate_output_accepts_token_kill_alias(self) -> None:
        p = PowershellParams(cmd="ls", token_kill=False)
        assert p.deduplicate_output is False

    def test_deduplicate_output_default_true(self) -> None:
        p = PowershellParams(cmd="ls")
        assert p.deduplicate_output is True

    def test_max_lines_field(self) -> None:
        p = PowershellParams(cmd="ls", max_lines=50)
        assert p.max_lines == 50
        p2 = PowershellParams(cmd="ls", max_lines=None)
        assert p2.max_lines is None

    def test_timeout_min(self) -> None:
        p = PowershellParams(cmd="ls", timeout=1)
        assert p.timeout == 1

    def test_timeout_max(self) -> None:
        with pytest.raises(Exception):
            PowershellParams(cmd="ls", timeout=901)


@pytest.mark.skipif(
    not PWSH_AVAILABLE,
    reason="PowerShell tool is not available on this platform",
)
class TestPowershellInactivityTimeout:
    @pytest.fixture(autouse=True)
    def _force_pwsh_enabled(self) -> Any:
        """Force-enable the Powershell tool for these integration tests.

        With the shipped agent_worker.json selecting ``bash``, the gate
        ``_should_enable_powershell()`` is False whenever Git Bash is
        installed on Windows; these tests execute real pwsh commands, so the
        gate is bypassed.
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
        # diff returns 1 (ToolError) when files differ
        assert isinstance(result, ToolError)

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

class TestBashParamsInteractive:
    def test_empty_cmd_execute_raises(self) -> None:
        with pytest.raises(ValueError):
            BashParams(cmd="")

    def test_empty_cmd_interactive_succeeds(self) -> None:
        p = BashParams(cmd="", mode="interactive")
        assert p.cmd == ""
        assert p.mode == "interactive"

    def test_cmd_and_interactive_succeeds(self) -> None:
        p = BashParams(cmd="ls", mode="interactive")
        assert p.cmd == "ls"
        assert p.mode == "interactive"


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
            assert args[0][1] == ["-c", "echo hello"]

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
            assert "echo start" in bash_args[1]
            assert "exec bash -i" in bash_args[1]
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
            assert args[0][1] == ["-i"]

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
            assert "TaskOutput" in result.message
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
                # Rewritten to the bare `rtk` prefix (shared bin dir is first on PATH).
                assert args[0][1] == ["-c", "rtk git status"]

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
            assert args[0][1] == ["-c", "read var"]


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

        await task.input("echo hello")
        await asyncio.sleep(0.5)
        output = await task.get_output()
        assert "hello" in output

        await task.input("exit")
        await task.wait(timeout=5)

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
        exit_result = await bash(BashParams(cmd="exit", task_id=task_id, timeout=5))
        assert isinstance(exit_result, ToolOk)
        assert "status: completed" in exit_result.output
