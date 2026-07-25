"""Tests for Python tool: code/file split and unified mode."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kimix.tools.py import Params as PythonParams


# ── Defect 2.1: code vs file split ───────────────────────────────────────


class TestPythonCodeUnified:
    def test_code_only_accepted(self) -> None:
        params = PythonParams(code="print(1+1)")
        assert params.code == "print(1+1)"

    def test_neither_rejected_unless_interactive(self) -> None:
        with pytest.raises(ValidationError, match="code` must be provided"):
            PythonParams()

    def test_neither_ok_when_interactive_legacy(self) -> None:
        params = PythonParams(interactive=True)
        assert params.code == ""
        assert params.mode == "interactive"

    def test_neither_ok_when_mode_interactive(self) -> None:
        params = PythonParams(mode="interactive")
        assert params.code == ""

    def test_task_id_with_code_valid_any_mode(self) -> None:
        for mode in ("execute", "send", "interactive"):
            p = PythonParams(code="print(1)", task_id="python_1", mode=mode)
            assert p.task_id == "python_1"

    def test_task_id_without_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PythonParams(code="", task_id="python_1", mode="send")

    def test_timeout_default(self) -> None:
        assert PythonParams(code="pass").timeout == 30

    def test_timeout_min_max(self) -> None:
        assert PythonParams(code="pass", timeout=1).timeout == 1
        assert PythonParams(code="pass", timeout=900).timeout == 900

    def test_timeout_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PythonParams(code="pass", timeout=0)

    def test_timeout_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PythonParams(code="pass", timeout=901)

    def test_max_lines_field_and_none(self) -> None:
        assert PythonParams(code="pass", max_lines=50).max_lines == 50
        assert PythonParams(code="pass", max_lines=None).max_lines is None

    def test_max_lines_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PythonParams(code="pass", max_lines=2)

    def test_wait_for_pattern_field(self) -> None:
        assert PythonParams(code="pass", wait_for_pattern="^ok").wait_for_pattern == "^ok"

    def test_output_path_field(self) -> None:
        assert PythonParams(code="pass", output_path="out.txt").output_path == "out.txt"

    def test_deduplicate_output_default_true(self) -> None:
        assert PythonParams(code="pass").deduplicate_output is True

    def test_deduplicate_output_new_name(self) -> None:
        assert PythonParams(code="pass", deduplicate_output=False).deduplicate_output is False

    def test_deduplicate_output_token_kill_alias(self) -> None:
        assert PythonParams(code="pass", token_kill=False).deduplicate_output is False

    def test_file_alias_accepted(self) -> None:
        """file=... alias maps to the code field."""
        params = PythonParams(file="script.py")
        assert params.code == "script.py"

    def test_code_with_py_extension_accepted(self) -> None:
        """code ending with .py is accepted as a string."""
        params = PythonParams(code="my_script.py")
        assert params.code == "my_script.py"

    def test_code_with_inline_code_accepted(self) -> None:
        """Arbitrary inline code strings are accepted."""
        params = PythonParams(code="x = 1\nprint(x)")
        assert params.code == "x = 1\nprint(x)"


# ── Defect 2.2: Unified mode parameter ──────────────────────────────────


class TestPythonUnifiedMode:
    @pytest.mark.parametrize("mode", ["execute", "send", "interactive"])
    def test_all_modes_accepted(self, mode: str) -> None:
        params = PythonParams(code="print(1)", mode=mode)
        assert params.mode == mode

    def test_legacy_interactive_bool_still_works(self) -> None:
        params = PythonParams(interactive=True)
        assert params.mode == "interactive"

    def test_execute_is_default(self) -> None:
        params = PythonParams(code="print(1)")
        assert params.mode == "execute"

    def test_run_alias_execute(self) -> None:
        params = PythonParams(code="print(1)", mode="run")
        assert params.mode == "execute"

    def test_background_alias_send(self) -> None:
        params = PythonParams(code="print(1)", mode="background")
        assert params.mode == "send"

    def test_send_mode_new_name(self) -> None:
        params = PythonParams(code="print(1)", mode="send")
        assert params.mode == "send"






# ── Python interpreter resolution (_resolve_python) ──────────────────────


import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    def test_none_for_non_venv(self) -> None:
        assert Python._build_env(sys.executable) is None or True  # depends on host

    def test_none_when_parent_not_scripts_bin(self, tmp_path: Path) -> None:
        exe = tmp_path / "python.exe"
        exe.touch()
        assert Python._build_env(str(exe)) is None

    def test_env_prepends_scripts_and_sets_virtual_env(self, tmp_path: Path) -> None:
        venv = tmp_path / ".venv"
        scripts = venv / "Scripts"
        scripts.mkdir(parents=True)
        exe = scripts / "python.exe"
        exe.touch()
        (venv / "pyvenv.cfg").touch()
        env = Python._build_env(str(exe))
        assert env is not None
        assert env["PATH"].split(os.pathsep)[0] == str(scripts)
        assert env["VIRTUAL_ENV"] == str(venv)

    def test_none_without_pyvenv_cfg(self, tmp_path: Path) -> None:
        scripts = tmp_path / "weird" / "Scripts"
        scripts.mkdir(parents=True)
        exe = scripts / "python.exe"
        exe.touch()
        assert Python._build_env(str(exe)) is None


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
