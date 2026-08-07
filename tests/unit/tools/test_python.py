"""Tests for the Python tool interpreter resolution, env building and hints."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kimix.tools.py import Params as PythonParams
from kimix.tools.py import Python


@pytest.fixture
def tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Python:
    session = MagicMock()
    session.custom_data = {}
    session.dir = tmp_path / ".kimi" / "sessions" / "test"
    session.dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("KIMIX_PYTHON_EXECUTABLE", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    # Isolate cwd from any real project .venv on this machine.
    monkeypatch.chdir(tmp_path)
    t = Python(session=session)
    yield t  # type: ignore[misc]


class TestResolvePython:
    def test_fallback_to_sys_executable(self, tool: Python) -> None:
        assert tool._resolve_python(PythonParams(mode="interactive")) == sys.executable

    def test_env_override(
        self, tool: Python, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = tmp_path / "fakepython.exe"
        fake.touch()
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(fake))
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(fake)

    def test_env_override_nonexistent_ignored(
        self, tool: Python, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(tmp_path / "missing.exe"))
        assert tool._resolve_python(PythonParams(mode="interactive")) == sys.executable

    def test_project_venv_discovery(self, tool: Python, tmp_path: Path) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        # session dir is under tmp_path, so discovery walks up and finds .venv
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(venv_py)

    def test_virtual_env_fallback(
        self, tool: Python, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # session dir is NOT under the venv root so only VIRTUAL_ENV matches
        venv_root = tmp_path / "elsewhere" / "myenv"
        venv_py = venv_root / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(venv_py)

    def test_priority_order(
        self, tool: Python, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        override = tmp_path / "override.exe"
        override.touch()
        # project .venv beats VIRTUAL_ENV
        other = tmp_path / "other"
        (other / "Scripts").mkdir(parents=True)
        (other / "Scripts" / "python.exe").touch()
        monkeypatch.setenv("VIRTUAL_ENV", str(other))
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(venv_py)
        # override beats project .venv
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(override))
        tool._resolved_python = None  # bypass cache
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(override)

    def test_cache_revalidated_when_deleted(self, tool: Python, tmp_path: Path) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        assert tool._resolve_python(PythonParams(mode="interactive")) == str(venv_py)
        venv_py.unlink()
        assert tool._resolve_python(PythonParams(mode="interactive")) == sys.executable


class TestBuildEnv:
    @pytest.fixture
    def fake_share_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Patch get_share_dir to a tmp_path containing a fake bin directory."""
        share = tmp_path / "share"
        (share / "bin").mkdir(parents=True)
        monkeypatch.setattr("kimix.tools.py.get_share_dir", lambda: share)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        return share

    def test_env_for_non_venv_prepends_shared_bin(
        self, fake_share_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure the shared bin is NOT already first so we exercise the build path.
        monkeypatch.setenv(
            "PATH", os.pathsep.join(["/usr/bin", "/bin", str(fake_share_dir / "bin")])
        )
        non_venv_python = str(fake_share_dir.parent / "python.exe")
        env = Python._build_env(non_venv_python)
        assert env is not None
        assert env["PATH"].split(os.pathsep)[0] == str(fake_share_dir / "bin")
        assert env.get("VIRTUAL_ENV") is None

    def test_env_when_parent_not_scripts_bin(
        self, fake_share_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe = fake_share_dir.parent / "python.exe"
        exe.touch()
        monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
        env = Python._build_env(str(exe))
        assert env is not None
        assert env["PATH"].split(os.pathsep)[0] == str(fake_share_dir / "bin")
        assert env.get("VIRTUAL_ENV") is None

    def test_env_prepends_shared_bin_then_venv_bin_and_sets_virtual_env(
        self, fake_share_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        venv = fake_share_dir.parent / ".venv"
        scripts = venv / "Scripts"
        scripts.mkdir(parents=True)
        exe = scripts / "python.exe"
        exe.touch()
        (venv / "pyvenv.cfg").touch()
        monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
        env = Python._build_env(str(exe))
        assert env is not None
        path_entries = env["PATH"].split(os.pathsep)
        assert path_entries[0] == str(fake_share_dir / "bin")
        assert path_entries[1] == str(scripts)
        assert env["VIRTUAL_ENV"] == str(venv)

    def test_env_without_pyvenv_cfg(
        self, fake_share_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripts = fake_share_dir.parent / "weird" / "Scripts"
        scripts.mkdir(parents=True)
        exe = scripts / "python.exe"
        exe.touch()
        monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
        env = Python._build_env(str(exe))
        assert env is not None
        assert env["PATH"].split(os.pathsep)[0] == str(fake_share_dir / "bin")
        assert env.get("VIRTUAL_ENV") is None

    def test_global_rtk_is_shadowed_by_shared_bin(
        self, fake_share_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_rtk = fake_share_dir.parent / "global_rtk"
        global_rtk.mkdir()
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(global_rtk), "/usr/bin", "/bin"])
        )
        # Use a non-venv interpreter path so we exercise the non-venv branch.
        non_venv_python = str(fake_share_dir.parent / "python.exe")
        env = Python._build_env(non_venv_python)
        assert env is not None
        path_entries = env["PATH"].split(os.pathsep)
        assert path_entries[0] == str(fake_share_dir / "bin")
        assert path_entries[1] == str(global_rtk)


class TestModuleNotFoundHint:
    def test_hint_present(self) -> None:
        output = "Traceback ...\nModuleNotFoundError: No module named 'foo'\n"
        hint = Python._module_not_found_hint(output, r"C:\proj\.venv\Scripts\python.exe")
        assert r"C:\proj\.venv\Scripts\python.exe" in hint
        assert "-m pip install foo" in hint

    def test_no_hint_for_other_errors(self) -> None:
        assert Python._module_not_found_hint("ValueError: bad", "python") == ""


@pytest.mark.asyncio
async def test_failed_run_includes_hint(tool: Python) -> None:
    """End-to-end: a ModuleNotFoundError failure surfaces the pip hint."""
    result = await tool(PythonParams(code="import definitely_missing_pkg_xyz", timeout=30))
    message = str(result.message)
    assert "-m pip install definitely_missing_pkg_xyz" in message
    assert "interpreter:" in message
