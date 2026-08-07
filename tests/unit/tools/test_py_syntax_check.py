"""Tests for the PySyntaxCheck tool (ruff-based lint) and its agent registration."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimix.tools.py.check import Params as CheckParams
from kimix.tools.py.check import PySyntaxCheck


@pytest.fixture
def tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PySyntaxCheck:
    session = MagicMock()
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    session.dir = tmp_path / ".kimi" / "sessions" / "test"
    session.dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("KIMIX_PYTHON_EXECUTABLE", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    # Isolate cwd from any real project .venv on this machine.
    monkeypatch.chdir(tmp_path)
    t = PySyntaxCheck(session=session)
    yield t  # type: ignore[misc]


@pytest.fixture
def fake_ruff(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make the tool's ruff availability probe succeed without touching the env."""
    fake = MagicMock()
    fake.find_spec.return_value = SimpleNamespace()  # non-None → ruff available
    monkeypatch.setattr("kimix.tools.py.check._importlib_util", fake)
    return fake


def _agent_json_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[3]  # repo root (D:\\kimi-agent)
    return [
        root / "src" / "kimix" / "agent_worker.json",
        root / "src" / "kimix" / "agent_subagent.json",
    ]


class TestRegistration:
    def test_py_syntax_check_registered_after_python(self) -> None:
        for path in _agent_json_paths():
            data = orjson.loads(path.read_bytes())
            tools = data["agent"]["tools"]
            idx_py = tools.index("kimix.tools.py:Python")
            assert tools[idx_py + 1] == "kimix.tools.py.check:PySyntaxCheck"
            assert tools.count("kimix.tools.py.check:PySyntaxCheck") == 1


class TestResolvePython:
    def test_fallback_to_sys_executable(
        self, tool: PySyntaxCheck, tmp_path: Path
    ) -> None:
        # tmp_path (cwd) has no .venv and VIRTUAL_ENV is unset → sys.executable.
        assert tool._resolve_python() == sys.executable

    def test_env_override(
        self, tool: PySyntaxCheck, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = tmp_path / "fakepython.exe"
        fake.touch()
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(fake))
        assert tool._resolve_python() == str(fake)

    def test_project_venv_discovery(
        self, tool: PySyntaxCheck, tmp_path: Path
    ) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        assert tool._resolve_python() == str(venv_py)

    def test_virtual_env_fallback(
        self, tool: PySyntaxCheck, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        venv_root = tmp_path / "elsewhere" / "myenv"
        venv_py = venv_root / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
        assert tool._resolve_python() == str(venv_py)

    def test_priority_order(
        self, tool: PySyntaxCheck, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        other = tmp_path / "other"
        (other / "Scripts").mkdir(parents=True)
        (other / "Scripts" / "python.exe").touch()
        monkeypatch.setenv("VIRTUAL_ENV", str(other))
        # project .venv beats VIRTUAL_ENV
        assert tool._resolve_python() == str(venv_py)
        # KIMIX_PYTHON_EXECUTABLE beats project .venv
        override = tmp_path / "override.exe"
        override.touch()
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(override))
        tool._resolved_python = None  # bypass cache
        assert tool._resolve_python() == str(override)

    def test_cache_revalidated_when_deleted(
        self, tool: PySyntaxCheck, tmp_path: Path
    ) -> None:
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        assert tool._resolve_python() == str(venv_py)
        venv_py.unlink()
        assert tool._resolve_python() == sys.executable


class TestRuffInvocation:
    @pytest.mark.asyncio
    async def test_ruff_check_runs_with_resolved_interpreter(
        self,
        tool: PySyntaxCheck,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_ruff: MagicMock,
    ) -> None:
        py_file = tmp_path / "sample.py"
        py_file.write_text("def foo():\n    pass\n", encoding="utf-8")
        fake_exe = tmp_path / "fakepython.exe"
        fake_exe.touch()
        monkeypatch.setenv("KIMIX_PYTHON_EXECUTABLE", str(fake_exe))

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(list(cmd))
            return SimpleNamespace(stdout="[]", returncode=0)

        monkeypatch.setattr("kimix.tools.py.check.subprocess.run", fake_run)

        result = await tool(CheckParams(file_path=str(py_file)))
        assert isinstance(result, ToolOk)
        assert "No issues found" in str(result.output)
        assert len(calls) == 2
        assert calls[0] == [str(fake_exe), "-m", "ruff", "check", str(py_file), "--output-format=json"]
        assert calls[1] == [str(fake_exe), "-m", "ruff", "format", str(py_file), "--check", "--output-format=json"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_tool_error(
        self, tool: PySyntaxCheck, monkeypatch: pytest.MonkeyPatch, fake_ruff: MagicMock
    ) -> None:
        # ruff resolves to the real interpreter fallback; no subprocess must run
        # because the file check happens before ruff is invoked.
        monkeypatch.setattr(
            "kimix.tools.py.check.subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("subprocess.run must not be called for missing file")
            ),
        )
        result = await tool(CheckParams(file_path="/nonexistent/missing.py"))
        assert isinstance(result, ToolError)
        assert "not found" in result.message.lower()
