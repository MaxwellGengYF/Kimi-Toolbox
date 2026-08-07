import asyncio
import importlib.util as _importlib_util
import os
import subprocess
import sys
from pathlib import Path

from kimi_cli.session import Session
from pydantic import BaseModel, Field

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimix.tools.common import _maybe_export_output


class Params(BaseModel):
    file_path: str = Field(description="Python file path to check.")


class PySyntaxCheck(CallableTool2):
    name: str = "PySyntaxCheck"
    description: str = "Check Python syntax with ruff."
    params: type[Params] = Params

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._resolved_python: str | None = None

    def _resolve_python(self) -> str:
        """Resolve the Python executable used to run ruff.

        Mirrors the ``Python`` tool's priority:
          1. ``KIMIX_PYTHON_EXECUTABLE`` env var (explicit override).
          2. A project ``.venv`` next to the session dir / cwd (walking up).
          3. ``VIRTUAL_ENV`` env var (active venv).
          4. ``sys.executable`` (fallback, backward compatible).

        The result is cached, but the cached path is re-validated on each call
        so a deleted/moved interpreter triggers re-resolution.
        """
        if self._resolved_python and Path(self._resolved_python).is_file():
            return self._resolved_python
        self._resolved_python = self._resolve_python_uncached()
        return self._resolved_python

    def _resolve_python_uncached(self) -> str:
        # 1. explicit override
        override = os.environ.get("KIMIX_PYTHON_EXECUTABLE")
        if override and Path(override).is_file():
            return override
        # 2. project .venv next to the session dir / cwd, walking up
        bases: list[Path] = []
        try:
            bases.append(Path(self._session.dir))
        except TypeError:
            pass
        bases.append(Path.cwd())
        for base in bases:
            for parent in (base, *base.parents):
                for candidate in (
                    parent / ".venv" / "Scripts" / "python.exe",  # Windows
                    parent / ".venv" / "bin" / "python",  # POSIX
                ):
                    if candidate.is_file():
                        return str(candidate)
        # 3. VIRTUAL_ENV
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            for candidate in (
                Path(venv) / "Scripts" / "python.exe",
                Path(venv) / "bin" / "python",
            ):
                if candidate.is_file():
                    return str(candidate)
        # 4. fallback
        return sys.executable

    async def __call__(self, params: Params) -> ToolReturnValue:
        # Ensure ruff is available, installing with the resolved interpreter
        # when missing (probe via find_spec — the module itself is never
        # imported because ruff is invoked as a subprocess).
        python_exe = self._resolve_python()
        if _importlib_util.find_spec("ruff") is None:
            try:
                subprocess.check_call([python_exe, "-m", "pip", "install", "ruff"])
                if _importlib_util.find_spec("ruff") is None:
                    raise ImportError("ruff not importable after installation")
            except Exception as e:
                return ToolError(
                    message=f"Failed to install ruff: {str(e)}", brief="Ruff installation failed"
                )

        # Read code from file and use it for ruff analysis
        try:
            file_path = Path(params.file_path)
            if not file_path.exists():
                return ToolError(
                    message=f"File not found: {params.file_path}", brief="File not found"
                )

            # Run ruff check to get errors and warnings
            result = await asyncio.to_thread(
                subprocess.run,
                [python_exe, "-m", "ruff", "check", str(file_path), "--output-format=json"],
                capture_output=True,
                text=True,
            )

            import orjson

            errors = []
            warnings = []
            hints = []

            if result.stdout:
                try:
                    diagnostics = orjson.loads(result.stdout)
                    for diag in diagnostics:
                        message = diag.get("message", "")
                        code = diag.get("code", "")
                        severity = diag.get("severity", "error")
                        location = f"Line {diag.get('location', {}).get('row', '?')}, Col {diag.get('location', {}).get('column', '?')}"

                        item = f"[{code}] {message} ({location})"

                        if severity == "error":
                            errors.append(item)
                        elif severity == "warning":
                            warnings.append(item)
                        else:
                            hints.append(item)
                except orjson.JSONDecodeError:
                    pass

            # Also check for formatting issues as hints
            fmt_result = await asyncio.to_thread(
                subprocess.run,
                [
                    python_exe,
                    "-m",
                    "ruff",
                    "format",
                    str(file_path),
                    "--check",
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
            )

            if fmt_result.stdout:
                try:
                    fmt_diagnostics = orjson.loads(fmt_result.stdout)
                    for diag in fmt_diagnostics:
                        message = diag.get("message", "Formatting issue")
                        location = f"Line {diag.get('start_location', {}).get('row', '?')}"
                        hints.append(f"[format] {message} ({location})")
                except orjson.JSONDecodeError:
                    pass

            output_parts = []
            if errors:
                output_parts.append("Errors:\n" + "\n".join(f"  - {e}" for e in errors))
            if warnings:
                output_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
            if hints:
                output_parts.append("Hints:\n" + "\n".join(f"  - {h}" for h in hints))

            if not output_parts:
                output = "No issues found. Code looks good!"
            else:
                output = "\n\n".join(output_parts)

            output = _maybe_export_output(output)
            return ToolOk(output=output)

        except Exception as e:
            return ToolError(message=str(e), brief="PySyntaxCheck error")
