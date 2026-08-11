"""Tests for the Python tool interpreter resolution, env building and hints."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimix.tools.py import Params as PythonParams
from kimix.tools.py import Python


def _long_output() -> str:
    """~176 KB of distinct lines that survive the dedup/micro-compress pipeline
    (random-ish per-line md5 suffix defeats near-duplicate folding)."""
    return "\n".join(
        f"line_{i:05d}_{hashlib.md5(f'rand-{i}-seed'.encode()).hexdigest()}"
        for i in range(4000)
    )


@pytest.fixture
def tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Python:
    session = MagicMock()
    session.custom_data = {}
    # Deterministic config: empty config_json so `python.*` gates read defaults.
    session.custom_config = {"config_json": {}}
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


# ---------------------------------------------------------------------------
# WP2: cwd / workdir / deduplicate_output params removed
# ---------------------------------------------------------------------------
class TestNoCwdParam:
    def test_cwd_workdir_dedup_params_removed(self) -> None:
        props = PythonParams.model_json_schema()["properties"]
        for gone in ("cwd", "workdir", "deduplicate_output", "token_kill"):
            assert gone not in props, f"{gone} must be removed from Params"

    def test_extra_cwd_input_ignored(self) -> None:
        # Pydantic ignores unknown fields by default; the tool no longer
        # reads cwd, so a stray cwd input has no effect.
        p = PythonParams(code="print(1)", cwd=r"C:\work")  # type: ignore[call-arg]
        assert not hasattr(p, "cwd")


# ---------------------------------------------------------------------------
# WP2: ProcessTask wiring — scrub_env / redact kwargs (cwd removed)
# ---------------------------------------------------------------------------
def _fake_process_task(monkeypatch: pytest.MonkeyPatch, output: str = "fake output"):
    """Patch ``kimix.tools.py.ProcessTask`` with a fake that records ctor kwargs
    and behaves like a successfully completed process."""
    import unittest.mock as um

    inst = um.AsyncMock()
    inst.start.return_value = "fake-task-id"
    inst.thread_is_alive.return_value = False
    inst.wait_with_monitor = um.AsyncMock()
    inst.stream = um.AsyncMock()
    inst.stream.pop_output.return_value = output
    inst.stream.success.return_value = True
    inst.stream.exit_code = 0
    mock_cls = um.Mock(return_value=inst)
    monkeypatch.setattr("kimix.tools.py.ProcessTask", mock_cls)
    return mock_cls


class TestProcessTaskWiring:
    @pytest.mark.asyncio
    async def test_process_task_runs_without_cwd(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Python tool no longer forwards a working directory."""
        mock_cls = _fake_process_task(monkeypatch)
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["cwd"] is None

    @pytest.mark.asyncio
    async def test_defaults_forward_scrub_env_and_redact(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cls = _fake_process_task(monkeypatch)
        await tool(PythonParams(code="print('x')"))
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["scrub_env"] is True
        assert kwargs["redact"] is True

    @pytest.mark.asyncio
    async def test_config_disables_scrub_and_redact(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool._session.custom_config = {
            "config_json": {"python": {"scrub_env": False, "redact_secrets": False}}
        }
        mock_cls = _fake_process_task(monkeypatch)
        await tool(PythonParams(code="print('x')"))
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["scrub_env"] is False
        assert kwargs["redact"] is False

    @pytest.mark.asyncio
    async def test_env_passthrough_disables_scrub_only(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool._session.custom_config = {
            "config_json": {"python": {"env_passthrough": True}}
        }
        mock_cls = _fake_process_task(monkeypatch)
        await tool(PythonParams(code="print('x')"))
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["scrub_env"] is False
        assert kwargs["redact"] is True

    @pytest.mark.asyncio
    async def test_undict_config_json_falls_back_to_defaults(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool._session.custom_config = {"config_json": "not-a-dict"}
        mock_cls = _fake_process_task(monkeypatch)
        await tool(PythonParams(code="print('x')"))
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["scrub_env"] is True
        assert kwargs["redact"] is True


class TestOriginalSavedSuffix:
    @pytest.mark.asyncio
    async def test_message_includes_original_path_after_dedup(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repeated = "ERROR\n" * 10
        _fake_process_task(monkeypatch, output=repeated)
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    @pytest.mark.asyncio
    async def test_message_includes_original_path_after_truncate(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_output = "\n".join(f"line_{i}" for i in range(500))
        _fake_process_task(monkeypatch, output=long_output)
        result = await tool(PythonParams(code="print('x')", max_lines=10))
        assert isinstance(result, ToolOk)
        assert "[original saved to .kimix_cache/tmp_" in result.message

    @pytest.mark.asyncio
    async def test_message_no_suffix_when_filter_unchanged(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dedup is always on, but output with no repeats is left unchanged,
        so no original temp file is created and no suffix is appended."""
        _fake_process_task(monkeypatch, output="plain output")
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        assert "[original saved to" not in result.message

    @pytest.mark.asyncio
    async def test_message_includes_original_path_after_summarize(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A >64KB output that survives the (always-on) token filter unchanged
        is still preserved before summarization replaces it with a summary."""
        import unittest.mock as um

        long_output = _long_output()
        assert len(long_output) > 65536
        _fake_process_task(monkeypatch, output=long_output)
        monkeypatch.setattr(
            "kimix.tools.py._summarize_long_output_async",
            um.AsyncMock(return_value="[summary]"),
        )
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        assert "output_truncated: true" in result.output
        assert "[original saved to .kimix_cache/tmp_" in result.message
        saved = result.message.split("[original saved to ", 1)[1].rstrip("]")
        assert Path(saved).read_text(encoding="utf-8") == long_output


# ---------------------------------------------------------------------------
# WP2: fail-fast syntax pre-check (no subprocess spawn on broken source)
# ---------------------------------------------------------------------------
class TestSyntaxPrecheck:
    @pytest.mark.asyncio
    async def test_broken_inline_code_fails_without_spawn(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructed: list[tuple] = []

        def boom(*args: object, **kwargs: object) -> None:
            constructed.append((args, kwargs))
            raise AssertionError("ProcessTask must not be constructed for broken syntax")

        monkeypatch.setattr("kimix.tools.py.ProcessTask", boom)
        result = await tool(PythonParams(code="def broken(:"))
        assert isinstance(result, ToolError)
        assert result.brief == "Syntax error"
        assert "Syntax error detected before execution" in result.message
        assert result.output == ""
        assert constructed == []

    @pytest.mark.asyncio
    async def test_null_bytes_fail_without_spawn(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructed: list[tuple] = []

        def boom(*args: object, **kwargs: object) -> None:
            constructed.append((args, kwargs))
            raise AssertionError("ProcessTask must not be constructed")

        monkeypatch.setattr("kimix.tools.py.ProcessTask", boom)
        # A real NUL byte in the source is rejected by compile() (ValueError).
        result = await tool(PythonParams(code="print('a\x00')"))
        assert isinstance(result, ToolError)
        assert result.brief == "Syntax error"
        assert constructed == []

    @pytest.mark.asyncio
    async def test_broken_file_mode_fails_without_spawn(
        self, tool: Python, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        py_file = tmp_path / "broken.py"
        py_file.write_text("def broken(:", encoding="utf-8")
        constructed: list[tuple] = []

        def boom(*args: object, **kwargs: object) -> None:
            constructed.append((args, kwargs))
            raise AssertionError("ProcessTask must not be constructed")

        monkeypatch.setattr("kimix.tools.py.ProcessTask", boom)
        result = await tool(PythonParams(code=str(py_file)))
        assert isinstance(result, ToolError)
        assert result.brief == "Syntax error"
        assert constructed == []

    @pytest.mark.asyncio
    async def test_valid_code_unaffected(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cls = _fake_process_task(monkeypatch, output="hello_ok")
        result = await tool(PythonParams(code="print('hello_ok')"))
        assert isinstance(result, ToolOk)
        assert "hello_ok" in str(result.output)
        mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_syntax_false_skips_precheck(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool._session.custom_config = {
            "config_json": {"python": {"check_syntax": False}}
        }
        mock_cls = _fake_process_task(monkeypatch)
        result = await tool(PythonParams(code="def broken(:"))
        # Pre-check skipped: ProcessTask is constructed (the fake "runs" it
        # successfully instead of the subprocess failing at runtime).
        assert isinstance(result, ToolOk)
        mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_with_broken_code_fails_without_spawn(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructed: list[tuple] = []

        def boom(*args: object, **kwargs: object) -> None:
            constructed.append((args, kwargs))
            raise AssertionError("ProcessTask must not be constructed")

        monkeypatch.setattr("kimix.tools.py.ProcessTask", boom)
        result = await tool(PythonParams(code="def broken(:", mode="interactive"))
        assert isinstance(result, ToolError)
        assert result.brief == "Syntax error"
        assert constructed == []

    @pytest.mark.asyncio
    async def test_pure_repl_no_initial_code_not_checked(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cls = _fake_process_task(monkeypatch)
        result = await tool(PythonParams(code="", mode="interactive"))
        # Pure REPL: no source to compile, ProcessTask is constructed.
        assert isinstance(result, ToolOk)
        assert "Interactive Python started" in result.message
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# WP2: config gate `python.summarize_long_output` (default True)
# ---------------------------------------------------------------------------
class TestSummarizeGate:
    @pytest.mark.asyncio
    async def test_summarize_disabled_skips_summarizer(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []

        async def fake_summarize(session, context: str, output: str) -> str:
            calls.append((session, context, output))
            return "SUMMARY"

        monkeypatch.setattr("kimix.tools.py._summarize_long_output_async", fake_summarize)
        tool._session.custom_config = {
            "config_json": {"python": {"summarize_long_output": False}}
        }
        # ~176 KB of distinct lines: above the 64 KB summarization threshold and
        # not compressible by the dedup/micro-compress pipeline.
        long_output = _long_output()
        assert len(long_output) > 65536
        _fake_process_task(monkeypatch, output=long_output)
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        assert calls == []
        assert "output_truncated:" in str(result.output)

    @pytest.mark.asyncio
    async def test_summarize_enabled_by_default_calls_summarizer(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []

        async def fake_summarize(session, context: str, output: str) -> str:
            calls.append((session, context, output))
            return "SUMMARY"

        monkeypatch.setattr("kimix.tools.py._summarize_long_output_async", fake_summarize)
        long_output = _long_output()
        _fake_process_task(monkeypatch, output=long_output)
        result = await tool(PythonParams(code="print('x')"))
        assert isinstance(result, ToolOk)
        assert len(calls) == 1
        assert calls[0][1] == "print('x')"


class TestEnvScrubbing:
    """Env scrubbing must survive the full pipeline (regression for the
    full-snapshot merge bug: _build_env returned a complete os.environ copy
    which re-introduced every scrubbed variable after ProcessTask's base scrub)."""

    def test_build_env_scrub_drops_secret_keeps_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        share = tmp_path / "share"
        (share / "bin").mkdir(parents=True)
        monkeypatch.setattr("kimix.tools.py.get_share_dir", lambda: share)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA1234567890ABCD")
        monkeypatch.setenv("VERIFY_SCRUB_TOKEN", "s3cr3t")
        monkeypatch.setenv("SAFE_HOME", "/home/u")
        env = Python._build_env(sys.executable, scrub_env=True)
        assert env is not None
        assert "AWS_ACCESS_KEY_ID" not in env
        assert "VERIFY_SCRUB_TOKEN" not in env
        assert env["SAFE_HOME"] == "/home/u"
        assert "PATH" in env
        assert env["PATH"].split(os.pathsep)[0] == str(share / "bin")

    def test_build_env_no_scrub_keeps_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        share = tmp_path / "share"
        (share / "bin").mkdir(parents=True)
        monkeypatch.setattr("kimix.tools.py.get_share_dir", lambda: share)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("VERIFY_SCRUB_TOKEN", "s3cr3t")
        env = Python._build_env(sys.executable, scrub_env=False)
        assert env is not None
        assert env.get("VERIFY_SCRUB_TOKEN") == "s3cr3t"

    @pytest.mark.asyncio
    async def test_scrub_end_to_end_child_does_not_see_secret(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERIFY_SCRUB_TOKEN", "s3cr3t-value")
        result = await tool(
            PythonParams(
                code="import os; print(os.environ.get('VERIFY_SCRUB_TOKEN', '<scrubbed>'))",
                timeout=30,
            )
        )
        assert not isinstance(result, ToolError)
        assert "<scrubbed>" in str(result.output), f"secret leaked: {result.output!r}"
        assert "s3cr3t-value" not in str(result.output)


# ---------------------------------------------------------------------------
# Scripts are saved to the shared temp folder (not the session folder) and
# return messages show the short relative path.
# ---------------------------------------------------------------------------
class TestScriptTempFolder:
    def test_inline_code_written_to_temp_folder_not_session_dir(
        self, tool: Python
    ) -> None:
        from kimix.tools import common as common_mod

        script_path, is_file_mode = tool._resolve_script_source(
            PythonParams(code="x = 42")
        )
        p = Path(script_path)
        assert is_file_mode is False
        assert p.is_absolute()
        assert p.is_file()
        assert p.read_text(encoding="utf-8") == "x = 42"
        # Inside the shared temp folder, not the session dir.
        temp_root = common_mod._temp_folder.resolve()
        assert temp_root in p.resolve().parents
        assert not list(Path(tool._session.dir).glob("*.py"))

    def test_display_path_is_short_relative(self, tool: Python) -> None:
        from kimix.tools import common as common_mod

        script_path, _ = tool._resolve_script_source(PythonParams(code="x = 42"))
        display = common_mod._display_temp_path(script_path)
        temp_prefix = str(common_mod._temp_folder).replace("\\", "/")
        assert display.startswith(temp_prefix + "/")
        assert "sessions" not in display

    @pytest.mark.asyncio
    async def test_execute_message_uses_relative_temp_path(
        self, tool: Python, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_process_task(monkeypatch, output="ok")
        result = await tool(PythonParams(code="print('ok')"))
        assert isinstance(result, ToolOk)
        assert ".kimix_cache/tmp_" in str(result.message)
        assert "sessions" not in str(result.message)
        assert not list(Path(tool._session.dir).glob("*.py"))
