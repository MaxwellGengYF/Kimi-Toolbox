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
    _create_script_file,
    _display_temp_path,
    _extract_export_path,
    _interactive_scope_text,
    _maybe_export_output_async,
    _maybe_export_rtk_original_async,
    _original_saved_message,
    _summarize_long_output_async,
    _token_filter_output,
    ProcessTask,
)
from kimix.tools.prompt_common import (
    accepts_alias_text,
    cwd_field,
    deduplicate_output_field,
    max_lines_field,
    mode_field,
    normalize_mode_validator,
    task_id_field,
    timeout_field,
    wait_for_pattern_field,
)
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import AliasChoices, BaseModel, Field, model_validator
from kimi_cli.session import Session
from kimi_cli.share import get_share_dir

if TYPE_CHECKING:
    from kimix.tools.background.utils import BackgroundStream


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    code: str = Field(
        default="",
        validation_alias=AliasChoices("code", "source_code", "file"),
        description=(
            "Inline Python code to execute. " + accepts_alias_text("code", "file", word=False) + " "
            "When the value ends with '.py' and the file exists, "
            "it is treated as a file path."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="Output file path."
    )
    timeout: int = timeout_field()
    mode: Literal["execute", "send", "interactive"] = mode_field(
        execute_desc="Run code and wait for completion (default).",
        send_desc="Execute code in background, return immediately with task_id.",
        interactive_desc="Start a persistent Python REPL, return task_id for further input.",
    )
    task_id: str | None = task_id_field("code", tail="being executed as a new script.")
    wait_for_pattern: str | None = wait_for_pattern_field()
    max_lines: int | None = max_lines_field()
    deduplicate_output: bool = deduplicate_output_field(accepts_alias=True)
    cwd: str | None = cwd_field("script", via_alias=False)

    @model_validator(mode="before")
    @classmethod
    def _normalize_mode(cls, data: dict) -> dict:
        """Convert deprecated boolean flags and mode aliases to canonical names."""
        return normalize_mode_validator(data)

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
        "Scripts run with a resolved interpreter (a project .venv is used when found, otherwise "
        "the backend interpreter). To install packages for scripts run by this tool, use "
        "'<python> -m pip install <pkg>' with the interpreter reported in error messages, or "
        "'uv pip install <pkg>' in the project directory — not bare 'pip install'. "
        "By default the child env is scrubbed of secret-looking vars. "
        + _interactive_scope_text(is_shell=False)
    )
    params: type[Params] = Params

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._semaphore = asyncio.Semaphore(8)
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
    def _build_env(python_exe: str, scrub_env: bool = False) -> dict[str, str] | None:
        """Build a child-process env that pins pip/python to the resolved venv.

        The shared ``bin`` directory is always prepended to PATH so that any
        ``rtk`` invoked from user scripts resolves to our binary first.

        When ``scrub_env`` is True the base snapshot starts from
        ``scrub_child_env(os.environ)`` so credential-looking variables are
        removed *before* the PATH/VIRTUAL_ENV tweaks are applied.  This matters
        because the returned dict is a *complete* environment snapshot: if it
        were built from the raw ``os.environ`` it would re-introduce every
        variable that ``ProcessTask``'s own base scrubbing removed.

        Returns None when the interpreter is not inside a virtualenv (e.g. the
        ``sys.executable`` fallback) and the shared ``bin`` directory is already
        first in PATH, preserving the zero-copy fast path.
        """
        from kimix.tools.security import scrub_child_env

        base_env = scrub_child_env(dict(os.environ)) if scrub_env else dict(os.environ)
        bin_dir = str(get_share_dir() / "bin")
        path_sep = os.pathsep
        current_path = os.environ.get("PATH", "")
        already_first = (
            current_path.startswith(bin_dir + path_sep) or current_path == bin_dir
        )

        def _prepend(env: dict[str, str]) -> dict[str, str]:
            if already_first:
                return env
            entries = [e for e in current_path.split(path_sep) if e and e != bin_dir]
            env["PATH"] = path_sep.join([bin_dir] + entries)
            return env

        exe = Path(python_exe)
        is_venv = (
            exe.parent.name in ("Scripts", "bin")
            and (exe.parent.parent / "pyvenv.cfg").is_file()
        )

        if not is_venv:
            return None if already_first else _prepend(base_env)

        env = base_env
        env["VIRTUAL_ENV"] = str(exe.parent.parent)
        venv_bin_dir = str(exe.parent)
        entries = [e for e in current_path.split(path_sep) if e and e != bin_dir]
        env["PATH"] = path_sep.join([bin_dir, venv_bin_dir] + entries)
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

    def _python_config(self) -> dict:
        """Read the ``python`` section of the session's ``config_json``.

        Dict-guarded: a missing/undict ``config_json`` (or ``python`` section)
        falls back to ``{}`` so every consumer uses the documented defaults.
        """
        try:
            raw = self._session.custom_config.get("config_json", {})
        except AttributeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        cfg = raw.get("python", {})
        return cfg if isinstance(cfg, dict) else {}

    def _syntax_check_error(
        self, params: Params, script_path: str | None, is_file_mode: bool
    ) -> ToolError | None:
        """Fail-fast compile pre-check of the script source.

        Compiles the source (inline ``params.code`` or the file-mode script
        content) before any subprocess spawn, catching ``SyntaxError`` and
        ``ValueError`` (e.g. null bytes) with line/column info.  Gated by the
        ``python.check_syntax`` config (default True).  Never runs for a pure
        interactive REPL (no initial code) and is never called from
        ``_continue_session``.

        Returns a ``ToolError`` when the source is not valid Python, else None.
        """
        cfg = self._python_config()
        if not cfg.get("check_syntax", True):
            return None
        if script_path is None:
            # Pure interactive REPL with no initial code.
            return None
        if is_file_mode:
            try:
                source = Path(script_path).read_text(encoding="utf-8")
            except OSError as e:
                return ToolError(
                    output="",
                    message=f"Failed to read script source for syntax check: {e}",
                    brief="Syntax check failed",
                )
        else:
            source = params.code
        try:
            compile(source, "<inline>" if not is_file_mode else script_path, "exec")
        except (SyntaxError, ValueError) as e:
            location = ""
            lineno = getattr(e, "lineno", None)
            offset = getattr(e, "offset", None)
            if lineno is not None:
                location = f" (line {lineno}"
                if offset is not None:
                    location += f", column {offset}"
                location += ")"
            return ToolError(
                output="",
                message=f"Syntax error detected before execution: {e}{location}",
                brief="Syntax error",
            )
        return None



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

        # Priority 2: inline code — write to the shared temp folder (not the
        # session folder) so generated scripts don't accumulate in session
        # state.  The returned path is absolute (subprocess-ready); display it
        # via ``_display_temp_path`` so messages show the short relative form.
        script_path = _create_script_file(params.code, ext='.py')
        return script_path, False

    async def __call__(self, params: Params) -> ToolReturnValue:
        # Early dispatch: continue an existing session
        if params.task_id is not None:
            return await self._continue_session(params)

        # Validate the working directory before any dispatch: a bad cwd must
        # fail fast without spawning a subprocess.
        if params.cwd:
            from kimix.tools.security import validate_workdir
            err = validate_workdir(params.cwd)
            if err:
                return ToolError(message=err, brief="Invalid workdir")

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
        script_path, is_file_mode = self._resolve_script_source(params)

        # Fail-fast syntax pre-check before spawning (config-gated).
        syntax_error = self._syntax_check_error(params, script_path, is_file_mode)
        if syntax_error is not None:
            return syntax_error

        if script_path is not None:
            args = ["-i", script_path]
        else:
            # Pure interactive REPL (no initial code)
            args = ["-i"]

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        python_exe = self._resolve_python(params)
        cfg = self._python_config()
        scrub_on = bool(cfg.get("scrub_env", True)) and not bool(cfg.get("env_passthrough", False))
        redact_on = bool(cfg.get("redact_secrets", True))
        process_task = ProcessTask(
            python_exe,
            args,
            cwd=params.cwd,
            env=self._build_env(python_exe, scrub_env=scrub_on),
            scrub_env=scrub_on,
            redact=redact_on,
            append_newline=True,
        )
        task_id = await process_task.start(self._session, "python")

        if params.wait_for_pattern is not None and process_task.stream is not None:
            from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
            inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
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
                    + self._module_not_found_hint(output, python_exe)
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
        display_script_path = _display_temp_path(script_path) if script_path else ""

        # Fail-fast syntax pre-check before any subprocess spawn (config-gated).
        syntax_error = self._syntax_check_error(params, script_path, is_file_mode)
        if syntax_error is not None:
            return syntax_error

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

        cfg = self._python_config()
        scrub_on = bool(cfg.get("scrub_env", True)) and not bool(cfg.get("env_passthrough", False))
        redact_on = bool(cfg.get("redact_secrets", True))
        process_task = ProcessTask(
            python_exe,
            args,
            cwd=params.cwd,
            env=self._build_env(python_exe, scrub_env=scrub_on),
            scrub_env=scrub_on,
            redact=redact_on,
        )
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
                from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
                inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
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

        suffix = _original_saved_message(original_path)
        if not success:
            msg = (
                f"{source_label}: `{display_script_path}` failed "
                f"(interpreter: {python_exe})"
                + self._module_not_found_hint(output, python_exe)
            )
            if suffix:
                msg = f"{msg} {suffix}"
            return ToolError(
                output=block,
                message=msg,
                brief="Python execution error"
            )

        success_message = f"{source_label}: `{display_script_path}`"
        if suffix:
            success_message = f"{success_message} {suffix}"
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

        from kimix.tools.background.utils import DEFAULT_INACTIVITY_TIMEOUT
        inactivity_timeout = min(DEFAULT_INACTIVITY_TIMEOUT, float(params.timeout))
        output, matched, elapsed = await stream.wait_for_output(
            timeout=params.timeout, pattern=pattern,
            inactivity_timeout=inactivity_timeout,
        )
        alive = await stream.thread_is_alive()
        status = "running" if alive else "completed"
        return await self._format_session_result(
            task_id, stream, params, output, status,
            wait_matched=matched, elapsed_seconds=elapsed,
            message=(
                f"Data sent to `{task_id}`. Status: {status}."
                + self._module_not_found_hint(output, self._resolve_python(params))
            ),
            brief="Data sent and output retrieved",
        )

    async def _process_output(
        self, params: Params, output: str, source_label: str = "Script"
    ) -> tuple[str, str | None, bool, str | None]:
        """Summarize/export long output. Returns (display_output, path, truncated, original_path)."""
        # When rtk itself folded the output, preserve the full stream so the
        # model can page through the unfiltered results.  This is done before
        # the local token filter so the raw rtk stream is captured even when
        # dedup/max_lines are disabled.
        rtk_original_path: str | None = None
        if output and not (params.deduplicate_output or params.max_lines is not None):
            rtk_original_path, _ = await _maybe_export_rtk_original_async(output)
        # Run token filter pipeline (dedup, truncate).
        # Python tool doesn't rewrite commands with RTK binary, so rtk_rewritten=False.
        output, original_path = await _token_filter_output(
            output,
            token_kill=params.deduplicate_output,
            max_lines=params.max_lines,
            rtk_rewritten=False,
        )
        if original_path is None:
            original_path = rtk_original_path
        output_truncated = False
        if len(output) > 65536:
            if self._python_config().get("summarize_long_output", True):
                # Use the source (file path or inline code) as context for summarization
                source_context = params.code
                output = await _summarize_long_output_async(self._session, source_context, output)
                output_truncated = True
            else:
                # Summarization disabled by config: still export, but don't
                # burn an LLM call on the (already token-filtered) output.
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
        suffix = _original_saved_message(original_path)
        if suffix:
            message = f"{message} {suffix}" if message else suffix
        return ToolOk(output=block, message=message, brief=brief)
