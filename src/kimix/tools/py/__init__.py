"""Python tool that executes code or runs .py files via the system Python executable."""

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
import regex as re
from kimix.tools.common import (
    _build_session_output_block,
    _extract_export_path,
    _interactive_scope_text,
    _maybe_export_output_async,
    _summarize_long_output_async,
    _token_filter_output,
    ProcessTask,
)
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import AliasChoices, BaseModel, Field, model_validator
from kimi_cli.session import Session

if TYPE_CHECKING:
    from kimix.tools.background.utils import BackgroundStream


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    code: str = Field(
        default="",
        validation_alias=AliasChoices("code", "source_code", "file"),
        description=(
            "Inline Python code to execute. Accepts `code` or `file`. "
            "When the value ends with '.py' and the file exists, "
            "it is treated as a file path."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="Output file path."
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=900,
        description="Timeout in seconds (1-900)."
    )
    mode: Literal["execute", "send", "interactive"] = Field(
        default="execute",
        description=(
            "'execute': Run code and wait for completion (default). "
            "'send': Execute code in background, return immediately with task_id. "
            "'interactive': Start a persistent Python REPL, return task_id for further input."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Existing session/task ID to continue. When provided, 'code' is sent to "
            "the process stdin instead of being executed as a new script."
        ),
    )
    wait_for_pattern: str | None = Field(
        default=None,
        description=(
            "Optional regex pattern. After starting or sending input, the tool blocks up "
            "to 'timeout' seconds until the pattern appears in output."
        ),
    )
    max_lines: int | None = Field(
        default=None,
        ge=3,
        description="Max lines to return via head+tail fold. <N> head lines + <N> tail lines kept; middle collapsed. None = unlimited.",
    )
    deduplicate_output: bool = Field(
        default=True,
        alias="token_kill",  # backward compat with shell tools
        description="Deduplicate repeated output lines from known commands (pytest, ruff, etc.). "
                    "Accepts `deduplicate_output` or `token_kill`. "
                    "Set to False to see raw, unfiltered output.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_mode(cls, data: dict) -> dict:
        """Convert deprecated boolean flags and mode aliases to canonical names."""
        if isinstance(data, dict):
            if data.get('interactive', False):
                data['mode'] = 'interactive'
            if 'mode' in data:
                if data['mode'] == 'run':
                    data['mode'] = 'execute'
                elif data['mode'] == 'background':
                    data['mode'] = 'send'
        return data

    @model_validator(mode="after")
    def _validate_source(self) -> "Params":
        if not self.code and self.task_id is None and self.mode != "interactive":
            raise ValueError("`code` must be provided (unless mode='interactive' or task_id is set).")
        if self.task_id is not None and not self.code:
            raise ValueError("code cannot be empty when continuing a session via task_id")
        return self


class Python(CallableTool2[Params]):
    name: str = "Python"
    description: str = (
        "Execute Python code or run a .py file directly. "
        "Use `code` for inline Python code or a path to an existing .py file (auto-detected). "
        "Output longer than `max_lines` is collapsed via head+tail fold (first N + last N lines, "
        "with middle replaced by a truncation marker). Set `max_lines=None` for unlimited output. "
        "Set `deduplicate_output=False` (or `token_kill=False`) to disable deduplication of repeated "
        "output lines from known commands (pytest, ruff, mypy, pip, etc.). "
        "When invoking known CLI tools (pytest, ruff, mypy, pip, uv, git, npm, etc.) via subprocess, "
        "you can use the \"rtk\" executable to reduce token usage: rtk is a valid binary available on "
        "PATH with the usage pattern `rtk <process> <arguments...>`. For example: "
        "subprocess.run([\"rtk\", \"pytest\", \"-v\"]) or subprocess.run(\"rtk pip list\", shell=True). "
        "rtk automatically deduplicates and truncates the output of the wrapped command. "
        "Scripts run with a resolved interpreter (a project .venv is used when found, otherwise "
        "the backend interpreter). To install packages for scripts run by this tool, use "
        "'<python> -m pip install <pkg>' with the interpreter reported in error messages, or "
        "'uv pip install <pkg>' in the project directory — not bare 'pip install'."
        " "
        + _interactive_scope_text(is_shell=False)
    )
    params: type[Params] = Params

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._semaphore = asyncio.Semaphore(8)
        self._script_counter = 0
        self._resolved_python: str | None = None

    def _resolve_python(self, params: Params) -> str:
        """Resolve the Python executable used to run scripts.

        Priority order:
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
                    parent / ".venv" / "bin" / "python",          # POSIX
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

    @staticmethod
    def _build_env(python_exe: str) -> dict[str, str] | None:
        """Build a child-process env that pins pip/python to the resolved venv.

        Returns None when the interpreter is not inside a virtualenv (e.g. the
        ``sys.executable`` fallback), leaving the default env untouched.
        """
        exe = Path(python_exe)
        if exe.parent.name not in ("Scripts", "bin"):
            return None
        venv_dir = exe.parent.parent
        if not (venv_dir / "pyvenv.cfg").is_file():
            return None
        env = dict(os.environ)
        env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(venv_dir)
        return env

    @staticmethod
    def _module_not_found_hint(output: str, python_exe: str) -> str:
        """Return a remediation hint when the output shows ModuleNotFoundError."""
        match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", output)
        if not match:
            return ""
        return (
            f" Hint: the script ran with interpreter '{python_exe}'. If you installed "
            "the package with plain 'pip install', it may have gone to a different "
            f"environment. Retry with '{python_exe}' -m pip install {match.group(1)}."
        )



    def _resolve_script_source(self, params: Params) -> tuple[str | None, bool]:
        """Resolve the script source from params.

        Priority:
          1. ``params.code`` ending with ``.py`` and existing file — file path mode.
          2. ``params.code`` non-empty — inline code, written to a temp file.
          3. None / empty — returns (None, False)

        Returns ``(script_path, is_file_mode)`` where ``is_file_mode`` is True
        when the source is an existing file (not inline code).
        """
        if not params.code:
            return None, False

        # Priority 1: code ending with .py and existing file → file path mode
        code_path = Path(params.code)
        if params.code.endswith('.py') and code_path.is_file():
            return params.code, True

        # Priority 2: inline code — write to a temp file
        session_dir = Path(self._session.dir)
        script_name = f"{self._script_counter}.py"
        script_path = str(session_dir / script_name)
        self._script_counter += 1
        # Write is done synchronously because we need the path before async ops
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / script_name).write_text(params.code, encoding='utf-8')
        return script_path, False

    async def __call__(self, params: Params) -> ToolReturnValue:
        # Early dispatch: continue an existing session
        if params.task_id is not None:
            return await self._continue_session(params)

        async with self._semaphore:
            if params.mode == "interactive":
                return await self._start_interactive(params)
            elif params.mode == "send":
                # Execute in background mode
                return await self._execute_code(params, background=True)
            else:
                return await self._execute_code(params, background=False)

    async def _start_interactive(self, params: Params) -> ToolReturnValue:
        """Start an interactive Python session."""
        # Determine script path from `code` (auto-detect .py file or inline code)
        script_path, _ = self._resolve_script_source(params)

        if script_path is not None:
            args = ["-i", script_path]
        else:
            # Pure interactive REPL (no initial code)
            args = ["-i"]

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        python_exe = self._resolve_python(params)
        process_task = ProcessTask(
            python_exe, args, env=self._build_env(python_exe), append_newline=True
        )
        task_id = await process_task.start(self._session, "python")

        if params.wait_for_pattern is not None and process_task.stream is not None:
            inactivity_timeout = min(30.0, float(params.timeout))
            output, matched, elapsed = await process_task.stream.wait_for_output(
                timeout=params.timeout, pattern=pattern,
                inactivity_timeout=inactivity_timeout,
            )
            alive = await process_task.thread_is_alive()
            status = "running" if alive else "completed"
            return await self._format_session_result(
                task_id, process_task.stream, params, output, status,
                wait_matched=matched, elapsed_seconds=elapsed,
                message=(
                    f"Interactive Python started. task_id: `{task_id}`. "
                    "Send 'exit()' to close the session."
                ),
                brief="Interactive Python started",
            )

        return ToolOk(
            output="",
            message=(
                f"Interactive Python started. task_id: `{task_id}`. "
                "Use task_id to send commands and TaskOutput to read results. "
                "Send 'exit()' to close the session."
            ),
            brief="Interactive Python started",
        )

    async def _execute_code(self, params: Params, background: bool = False) -> ToolReturnValue:
        """Execute Python code (non-interactive, one-shot).

        Args:
            params: Tool parameters.
            background: If True, start the process and return immediately with task_id.
        """
        # Resolve script source: `file` param takes priority
        script_path, is_file_mode = self._resolve_script_source(params)
        display_script_path = script_path.replace("\\", "/") if script_path else ""

        if is_file_mode:
            source_label = "File"
        elif script_path is not None:
            source_label = "Script"
        else:
            return ToolError(
                output="",
                message="No code or file provided to execute.",
                brief="Missing code/file",
            )

        python_exe = self._resolve_python(params)
        args = [script_path]

        process_task = ProcessTask(python_exe, args, env=self._build_env(python_exe))
        task_id = await process_task.start(self._session, "python")

        if background:
            return ToolOk(
                output=f"{source_label} saved to `{display_script_path}`. Running in background. task_id: `{task_id}`. Use `TaskOutput` tool to retrieve output.",
                brief="Background task started"
            )

        wait_matched: bool | None = None
        elapsed_seconds: float | None = None
        try:
            if params.wait_for_pattern is not None and process_task.stream is not None:
                pattern = self._compile_pattern(params.wait_for_pattern)
                if isinstance(pattern, ToolError):
                    return pattern
                inactivity_timeout = min(30.0, float(params.timeout))
                output, wait_matched, elapsed_seconds = await process_task.stream.wait_for_output(
                    timeout=params.timeout, pattern=pattern,
                    inactivity_timeout=inactivity_timeout,
                )
                if await process_task.thread_is_alive():
                    return await self._format_session_result(
                        task_id, process_task.stream, params, output, "running",
                        wait_matched=wait_matched, elapsed_seconds=elapsed_seconds,
                        message=f"Python code matched pattern and is still running.",
                        brief="Pattern matched",
                    )
            else:
                # Wait for completion with timeout (allow a small buffer for cleanup)
                await process_task.wait_with_monitor(params.timeout)
        except asyncio.CancelledError:
            await process_task.stop()
            from kimix.tools.background.utils import remove_task_id
            remove_task_id(self._session, task_id)
            output = await process_task.stream.get_output() if process_task.stream else ""
            output = await _maybe_export_output_async(output)
            return ToolError(
                output=output,
                message=f"Python execution was cancelled.",
                brief="Execution cancelled",
            )

        if await process_task.thread_is_alive():
            output = await process_task.stream.pop_output() if process_task.stream else ""
            output = await _maybe_export_output_async(output)
            return ToolError(
                output=output,
                message=f"{source_label} saved to `{display_script_path}`. Running in background. task_id: `{task_id}`. use `TaskOutput`",
                brief="Timeout"
            )

        # Clean up foreground task registration
        from kimix.tools.background.utils import remove_task_id
        remove_task_id(self._session, task_id)

        # Get output
        output = await process_task.stream.pop_output() if process_task.stream else ""
        stream = process_task.stream
        success = await stream.success() if stream else False
        real_exit_code = stream.exit_code if stream else None

        # Handle output_path parameter if provided
        if params.output_path:
            async with await anyio.open_file(params.output_path, 'w', encoding='utf-8', errors='replace') as f:
                await f.write(output)
            display_path = params.output_path.replace("\\", "/")
            output = f'output exported to: {display_path}'
            # Use plain output for legacy output_path — skip structured block
            if not success:
                return ToolError(
                    output=output,
                    message=(
                        f"Python execution failed (interpreter: {python_exe})"
                        + self._module_not_found_hint(output, python_exe)
                    ),
                    brief="Python execution error"
                )
            success_message = f"{source_label}: `{display_script_path}`"
            return ToolOk(output=f"{success_message}\n\n{output}", brief=f"Python file executed: {display_script_path}")

        # Process output through token filter and summarization
        processed, output_path, output_truncated, original_path = await self._process_output(
            params, output
        )
        block = _build_session_output_block(
            task_id=task_id,
            status="completed",
            output=processed,
            exit_code=real_exit_code,
            wait_matched=wait_matched,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            output_truncated=output_truncated,
            original_path=original_path,
        )

        if not success:
            return ToolError(
                output=block,
                message=(
                    f"{source_label}: `{display_script_path}` failed "
                    f"(interpreter: {python_exe})"
                    + self._module_not_found_hint(output, python_exe)
                ),
                brief="Python execution error"
            )

        success_message = f"{source_label}: `{display_script_path}`"
        return ToolOk(
            output=block,
            message=success_message,
            brief=f"Python {'file' if is_file_mode else 'code'} executed successfully"
        )

    def _compile_pattern(self, wait_for_pattern: str | None) -> re.Pattern[str] | ToolError:
        """Compile a regex pattern, returning ToolError on invalid input."""
        if wait_for_pattern is None:
            return None
        try:
            return re.compile(wait_for_pattern)
        except re.error as e:
            return ToolError(
                output="",
                message=f"Invalid wait_for_pattern: {e}",
                brief="Invalid pattern",
            )

    async def _continue_session(self, params: Params) -> ToolReturnValue:
        """Send input to an existing Python session and optionally wait for output."""
        from kimix.tools.background.utils import get_all_tasks

        tasks = get_all_tasks(self._session)
        task_id = params.task_id.strip() if params.task_id else ""
        stream = tasks.get(task_id)
        if stream is None:
            started = [tid for tid, s in tasks.items() if await s.is_started()]
            if not started:
                return ToolError(
                    output="",
                    message=f"Task '{params.task_id}' not found. No running tasks.",
                    brief="Task not found",
                )
            return ToolError(
                output="",
                message=(
                    f"Task '{params.task_id}' not found. "
                    f"Available tasks: [{', '.join(started)}]"
                ),
                brief=f"Task '{params.task_id}' not found",
            )

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        # Discard prior output so we only report new output produced after this input.
        await stream.pop_output()

        input_text = params.code
        if not input_text.endswith("\n"):
            input_text += "\n"
        if not await stream.input(input_text):
            return ToolError(
                output="",
                message=f"Failed to send input to task '{task_id}'",
                brief="Send input failed",
            )

        inactivity_timeout = min(30.0, float(params.timeout))
        output, matched, elapsed = await stream.wait_for_output(
            timeout=params.timeout, pattern=pattern,
            inactivity_timeout=inactivity_timeout,
        )
        alive = await stream.thread_is_alive()
        status = "running" if alive else "completed"
        return await self._format_session_result(
            task_id, stream, params, output, status,
            wait_matched=matched, elapsed_seconds=elapsed,
            message=f"Data sent to `{task_id}`. Status: {status}.",
            brief="Data sent and output retrieved",
        )

    async def _process_output(
        self, params: Params, output: str, source_label: str = "Script"
    ) -> tuple[str, str | None, bool, str | None]:
        """Summarize/export long output. Returns (display_output, path, truncated, original_path)."""
        # Run token filter pipeline (dedup, truncate).
        # Python tool doesn't rewrite commands with RTK binary, so rtk_rewritten=False.
        output, original_path = await _token_filter_output(
            output,
            token_kill=params.deduplicate_output,
            max_lines=params.max_lines,
            rtk_rewritten=False,
        )
        output_truncated = False
        if len(output) > 65536:
            # Use the source (file path or inline code) as context for summarization
            source_context = params.code
            output = await _summarize_long_output_async(self._session, source_context, output)
            output_truncated = True
        output = await _maybe_export_output_async(output)
        output_path = _extract_export_path(output)
        return output, output_path, output_truncated, original_path

    async def _format_session_result(
        self,
        task_id: str,
        stream: 'BackgroundStream' | None,
        params: Params,
        output: str,
        status: str,
        *,
        wait_matched: bool | None,
        elapsed_seconds: float | None,
        message: str,
        brief: str,
    ) -> ToolReturnValue:
        """Build a ToolOk response with a structured output block."""
        processed, output_path, output_truncated, original_path = await self._process_output(
            params, output
        )
        if status != "completed":
            real_exit_code = None
        else:
            real_exit_code = stream.exit_code if stream else None
            if real_exit_code is None:
                real_exit_code = 0 if await stream.success() else None
        block = _build_session_output_block(
            task_id=task_id,
            status=status,
            output=processed,
            exit_code=real_exit_code,
            wait_matched=wait_matched,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            output_truncated=output_truncated,
            original_path=original_path,
        )
        return ToolOk(output=block, message=message, brief=brief)
