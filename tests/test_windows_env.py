"""Comprehensive tests for Windows registry PATH/PATHEXT reading.

Covers:
- ``kimix.utils.windows_env`` (re-exports from kimi_cli)
- ``kimi_cli.utils.environment`` (canonical implementation)
- Integration: ``Run`` and ``Powershell`` tools calling refresh before execution
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``winreg``/``ctypes.windll`` only exist on Windows; tests below fake them,
# so the ``patch(..., create=True)`` targets must be created on other platforms.
_requires_win32 = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only registry behaviour"
)


# ============================================================================
# Mocks for winreg / ctypes
# ============================================================================

REG_SZ = 1
REG_EXPAND_SZ = 2


class _FakeKey:
    def __init__(self, data: dict[str, tuple[str, int]]):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_fake_winreg(
    hklm_path: str | None = None,
    hklm_type: int = REG_SZ,
    hkcu_path: str | None = None,
    hkcu_type: int = REG_SZ,
    hklm_pathext: str | None = None,
    hklm_pathext_type: int = REG_SZ,
    hkcu_pathext: str | None = None,
    hkcu_pathext_type: int = REG_SZ,
):
    fake_winreg = MagicMock()
    fake_winreg.HKEY_LOCAL_MACHINE = 0x80000002
    fake_winreg.HKEY_CURRENT_USER = 0x80000001
    fake_winreg.REG_SZ = REG_SZ
    fake_winreg.REG_EXPAND_SZ = REG_EXPAND_SZ
    fake_winreg.KEY_READ = 131097

    def _open_key(hive, subkey, reserved, access):
        if hive == fake_winreg.HKEY_LOCAL_MACHINE:
            data: dict[str, tuple[str, int]] = {}
            if hklm_path is not None:
                data["Path"] = (hklm_path, hklm_type)
            if hklm_pathext is not None:
                data["PATHEXT"] = (hklm_pathext, hklm_pathext_type)
            return _FakeKey(data)
        elif hive == fake_winreg.HKEY_CURRENT_USER:
            data = {}
            if hkcu_path is not None:
                data["Path"] = (hkcu_path, hkcu_type)
            if hkcu_pathext is not None:
                data["PATHEXT"] = (hkcu_pathext, hkcu_pathext_type)
            return _FakeKey(data)
        raise FileNotFoundError()

    fake_winreg.OpenKey = _open_key

    def _query_value_ex(key, name):
        try:
            return key._data[name]
        except KeyError:
            raise FileNotFoundError()

    fake_winreg.QueryValueEx = _query_value_ex
    return fake_winreg


def _fake_expand(value: str, buf: Any, nchars: int) -> int:
    """Simulate ``ExpandEnvironmentStringsW`` (expands ``%VAR%``, keeps unknown).

    Implemented without ``os.path.expandvars`` so it behaves like the Windows
    API on every platform (POSIX ``expandvars`` does not handle ``%VAR%``).
    """
    import regex as re
    expanded = re.sub(r"%([^%]+)%", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if buf is None:
        return len(expanded) + 1
    for i, ch in enumerate(expanded):
        buf[i] = ch
    buf[len(expanded)] = "\0"
    return len(expanded) + 1


# ============================================================================
# Tests: kimix.utils.windows_env — re-export verification
# ============================================================================

class TestRefreshEnvFromRegistry:
    """Verify ``kimix.utils.windows_env`` correctly re-exports from kimi_cli."""


    def test_call_refreshes_path(self) -> None:
        """Smoke test: calling the re-exported function updates PATH."""
        fake_winreg = _make_fake_winreg(
            hklm_path=r"C:\System", hkcu_path=r"C:\User"
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
        ):
            from kimix.utils.windows_env import refresh_env_from_registry
            refresh_env_from_registry()
            assert os.environ["PATH"] == r"C:\System;C:\User"


# ============================================================================
# Tests: _expand_registry_string
# ============================================================================

class TestExpandRegistryString:
    """Tests for ``_expand_registry_string()`` from kimi_cli.utils.environment."""

    def test_no_percent_returns_unchanged(self) -> None:
        from kimi_cli.utils.environment import _expand_registry_string
        assert _expand_registry_string(r"C:\Windows\System32") == r"C:\Windows\System32"

    @_requires_win32
    def test_percent_var_expanded_via_fallback(self) -> None:
        """Without Windows API available, falls back to ``os.path.expandvars``."""
        with patch.dict(os.environ, {"SYSTEMROOT": r"C:\WinNT"}):
            from kimi_cli.utils.environment import _expand_registry_string
            result = _expand_registry_string(r"%SYSTEMROOT%\System32")
            assert result == r"C:\WinNT\System32"

    @_requires_win32
    def test_multiple_percent_vars(self) -> None:
        with patch.dict(os.environ, {"A": "alpha", "B": "beta"}):
            from kimi_cli.utils.environment import _expand_registry_string
            result = _expand_registry_string(r"%A%\%B%")
            assert result == r"alpha\beta"

    def test_unknown_var_kept_literal(self) -> None:
        from kimi_cli.utils.environment import _expand_registry_string
        result = _expand_registry_string(r"%NOEXIST%\bin")
        assert result == r"%NOEXIST%\bin"


# ============================================================================
# Tests: _read_registry_value
# ============================================================================

class TestReadRegistryValue:
    """Tests for ``_read_registry_value()`` from kimi_cli.utils.environment."""

    def test_reads_existing_value(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_path=r"C:\Win")
        with patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True):
            from kimi_cli.utils.environment import _read_registry_value
            val, typ = _read_registry_value(
                fake_winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                "Path",
            )
            assert val == r"C:\Win"
            assert typ == REG_SZ

    def test_reads_expand_sz(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_path=r"%S%\bin", hklm_type=REG_EXPAND_SZ)
        with patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True):
            from kimi_cli.utils.environment import _read_registry_value
            val, typ = _read_registry_value(
                fake_winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                "Path",
            )
            assert val == r"%S%\bin"
            assert typ == REG_EXPAND_SZ

    def test_returns_none_for_missing_value(self) -> None:
        fake_winreg = _make_fake_winreg()
        with patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True):
            from kimi_cli.utils.environment import _read_registry_value
            val, typ = _read_registry_value(
                fake_winreg.HKEY_CURRENT_USER, "Environment", "Path"
            )
            assert val is None
            assert typ is None

    def test_file_not_found_returns_none(self) -> None:
        fake_winreg = MagicMock()
        fake_winreg.KEY_READ = 131097
        fake_winreg.OpenKey.side_effect = FileNotFoundError()
        with patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True):
            from kimi_cli.utils.environment import _read_registry_value
            val, typ = _read_registry_value(0x80000001, "Environment", "Path")
            assert val is None
            assert typ is None


# ============================================================================
# Tests: kimi_cli.utils.environment.refresh_windows_env
# ============================================================================

class TestRefreshWindowsEnv:
    """Tests for ``refresh_windows_env()`` in ``kimi_cli.utils.environment``."""

    def test_pathext_updated(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_pathext=".COM;.EXE", hkcu_pathext=".PS1")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATHEXT"] == ".COM;.EXE;.PS1"

    def test_path_system_and_user(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_path=r"C:\System", hkcu_path=r"C:\User")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATH"] == r"C:\System;C:\User"

    def test_path_expanded(self) -> None:
        import ctypes as real_ctypes

        fake_winreg = _make_fake_winreg(
            hklm_path=r"%SYS%\System32", hklm_type=REG_EXPAND_SZ
        )
        # ``ctypes`` (and its ``windll``) are only imported/exist on Windows;
        # build a fake module keeping the real (platform-independent)
        # ``create_unicode_buffer`` so ``_expand_registry_string`` works here.
        fake_ctypes = MagicMock()
        fake_ctypes.create_unicode_buffer = real_ctypes.create_unicode_buffer
        fake_ctypes.windll.kernel32.ExpandEnvironmentStringsW.side_effect = _fake_expand
        with (
            patch.dict(os.environ, {"SYS": r"C:\Win"}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("kimi_cli.utils.environment.ctypes", fake_ctypes, create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATH"] == r"C:\Win\System32"

    def test_path_deduplicate(self) -> None:
        fake_winreg = _make_fake_winreg(
            hklm_path=r"C:\A;C:\B", hkcu_path=r"C:\B;C:\C"
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATH"] == r"C:\A;C:\B;C:\C"

    # ── Edge cases (merged from former TestEdgeCases) ──────────────────

    def test_only_pathext_no_path(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_pathext=".COM;.EXE")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATHEXT"] == ".COM;.EXE"
            assert "PATH" not in os.environ

    def test_empty_string_values_ignored(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_path="", hkcu_path="")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert "PATH" not in os.environ

    def test_whitespace_only_entries_stripped(self) -> None:
        fake_winreg = _make_fake_winreg(hklm_path=r"C:\A;  ;C:\B")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.winreg", fake_winreg, create=True),
            patch("kimi_cli.utils.environment.sys.platform", "win32"),
            patch("ctypes.windll", create=True),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert os.environ["PATH"] == r"C:\A;C:\B"

    def test_non_windows_is_noop(self) -> None:
        """On non-Windows the function returns early without changes."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("kimi_cli.utils.environment.sys.platform", "linux"),
        ):
            from kimi_cli.utils.environment import refresh_windows_env
            refresh_windows_env()
            assert "PATH" not in os.environ
            assert "PATHEXT" not in os.environ


# ============================================================================
# Tests: Run tool calls refresh
# ============================================================================

class TestRunToolCallsRefresh:
    """Verify ``Run.__call__`` refreshes PATH from registry on Windows."""

    def test_run_calls_refresh_on_windows(self) -> None:
        mock_session = MagicMock()
        mock_session.custom_config.get.return_value = {}
        mock_session.custom_data = {}

        # On Windows, Run.__init__ raises SkipThisTool when
        # USE_SYSTEM_PWSH_ON_WINDOWS is True.  Patch that flag so
        # we can instantiate Run and verify the refresh call.
        import asyncio

        # Create the event loop BEFORE faking ``sys.platform``: asyncio picks
        # its Windows event-loop policy when ``sys.platform == "win32"``,
        # which cannot be imported on other platforms.
        loop = asyncio.new_event_loop()
        try:
            with (
                patch("kimix.tools.file.run.sys.platform", "win32"),
                patch("kimix.tools.file.run.USE_SYSTEM_PWSH_ON_WINDOWS", False),
                patch("kimix.tools.file.run.find_bash", return_value=None),
            ):
                from kimix.tools.file.run import Run, RunParams
                tool = Run(mock_session)

                with patch(
                    "kimix.utils.windows_env.refresh_env_from_registry"
                ) as mock_refresh:
                    with patch(
                        "kimix.tools.common.ProcessTask.start",
                        side_effect=OSError("simulated"),
                    ):
                        try:
                            loop.run_until_complete(
                                tool.__call__(RunParams(command="echo hello"))
                            )
                        except Exception:
                            pass
                        mock_refresh.assert_called_once()
        finally:
            loop.close()


# ============================================================================
# Tests: Powershell tool calls refresh
# ============================================================================

class TestPowershellToolCallsRefresh:
    """Verify ``Powershell.__call__`` refreshes PATH from registry on Windows."""

    def test_pwsh_calls_refresh_on_windows(self) -> None:
        mock_session = MagicMock()
        mock_session.custom_config.get.return_value = {}
        mock_session.custom_data = {}

        import asyncio

        from kimix.tools.file.bash.pwsh_tool import Powershell, PowershellParams

        # Create the event loop BEFORE faking ``sys.platform``: asyncio picks
        # its Windows event-loop policy when ``sys.platform == "win32"``,
        # which cannot be imported on other platforms.  Also stub out
        # ``_resolve_pwsh`` so no real PowerShell probe runs on this host.
        loop = asyncio.new_event_loop()
        try:
            # Keep ``sys.platform`` patched for the ``__call__`` as well: the
            # refresh inside ``Powershell.__call__`` is gated on win32 at call time.
            with (
                patch("kimix.tools.file.bash.pwsh_tool.sys.platform", "win32"),
                patch("kimix.tools.file.bash.pwsh_tool._bash_tool.find_bash", return_value=None),
                patch.object(Powershell, "_resolve_pwsh", lambda self: None),
            ):
                tool = Powershell(mock_session)

                with patch(
                    "kimix.utils.windows_env.refresh_env_from_registry"
                ) as mock_refresh:
                    with patch(
                        "kimix.tools.common.ProcessTask.start",
                        side_effect=OSError("simulated"),
                    ):
                        try:
                            loop.run_until_complete(
                                tool.__call__(PowershellParams(cmd="echo hello"))
                            )
                        except Exception:
                            pass
                        mock_refresh.assert_called_once()
        finally:
            loop.close()

    def test_pwsh_skips_refresh_on_linux(self) -> None:
        mock_session = MagicMock()
        mock_session.custom_config.get.return_value = {}
        mock_session.custom_data = {}

        from kimi_cli.tools import SkipThisTool
        with patch("kimix.tools.file.bash.pwsh_tool.sys.platform", "linux"):
            with pytest.raises(SkipThisTool):
                from kimix.tools.file.bash.pwsh_tool import Powershell
                Powershell(mock_session)


# ============================================================================
# Tests: Environment.detect integration
# ============================================================================

class TestEnvironmentDetectCallsRefresh:
    """Verify ``Environment.detect()`` calls ``refresh_windows_env`` on Windows."""

    @pytest.mark.asyncio
    async def test_detect_calls_refresh_on_windows(self) -> None:
        from kimi_cli.utils.environment import GitBashNotFoundError

        call_log: list[str] = []

        def _fake_refresh():
            call_log.append("refresh")

        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
            patch("platform.version", return_value="10.0.22631"),
            patch(
                "kimi_cli.utils.environment.refresh_windows_env",
                side_effect=_fake_refresh,
            ),
            patch(
                "kimi_cli.utils.environment._find_git_bash_path",
                side_effect=GitBashNotFoundError("no git"),
            ),
            patch("kaos.exec", side_effect=OSError("no where.exe")),
        ):
            from kimi_cli.utils.environment import Environment, GitBashNotFoundError
            env = await Environment.detect()
            assert "refresh" in call_log
            assert env.os_kind == "Windows"

    @pytest.mark.asyncio
    async def test_detect_skips_refresh_on_linux(self) -> None:
        call_log: list[str] = []

        def _fake_refresh():
            call_log.append("refresh")

        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch("platform.version", return_value="6.5.0"),
            patch(
                "kimi_cli.utils.environment.refresh_windows_env",
                side_effect=_fake_refresh,
            ),
        ):
            from kimi_cli.utils.environment import Environment
            env = await Environment.detect()
            assert call_log == []
            assert env.os_kind == "Linux"
