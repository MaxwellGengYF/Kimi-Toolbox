"""run tool for executing a process from a path."""
import anyio
import asyncio
import os
from pathlib import Path
from typing import Literal
import regex as re
import shlex
import sys
from kimi_cli.tools import SkipThisTool
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field, model_validator
from kimi_cli.session import Session
from kimix.tools.common import (
    _build_session_output_block,
    _create_script_file,
    _env_with_rg_bin_path,
    _extract_export_path,
    _interactive_scope_text,
    _is_known_rtk_command,
    _maybe_export_output_async,
    _export_to_temp_file_async,
    _rtk_binary_path,
    _summarize_long_output_async,
    _token_filter_output,
    ProcessTask,
)
from kimix.tools.file.bash.output_enhance import (
    annotate_failure,
    interpret_exit_code,
    redact_sensitive_output,
)
from kimix.tools.file.bash.safety import (
    check_hardline_blocked,
    foreground_background_guidance,
    validate_workdir,
)
from kimix.tools.prompt_common import (
    deduplicate_output_field,
    max_lines_field,
    mode_field,
    task_id_field,
    timeout_field,
    wait_for_pattern_field,
)
from kimi_cli.tools.display import ShellDisplayBlock
from kimi_cli.share import get_share_dir
import functools
import shlex
import shutil
_HUGE_CMD_THRESHOLD = 10000
"""Character count above which command display is culled to only the path."""

USE_SYSTEM_PWSH_ON_WINDOWS = True
USE_SYSTEM_SHELL = True

@functools.lru_cache(maxsize=1)
def find_bash() -> str | None:
    """Find the system bash executable."""
    if sys.platform == "darwin":
        # Strategy 1: Homebrew bash (Apple Silicon) – often newer than system bash
        candidate = Path("/opt/homebrew/bin/bash")
        if candidate.exists():
            return str(candidate.resolve())
        # Strategy 2: Homebrew bash (Intel Macs)
        candidate = Path("/usr/local/bin/bash")
        if candidate.exists():
            return str(candidate.resolve())
        # Strategy 3: MacPorts
        candidate = Path("/opt/local/bin/bash")
        if candidate.exists():
            return str(candidate.resolve())
        # Strategy 4: Git bash fallback (official Git installer for macOS)
        git_path = shutil.which("git")
        if git_path:
            git_exe = Path(git_path).resolve()
            if git_exe.parent.name.lower() == "bin":
                git_root = git_exe.parent
            else:
                git_root = git_exe.parent
            for subpath in ("bin/bash", "usr/bin/bash"):
                bash_candidate = git_root / subpath
                if bash_candidate.exists():
                    return str(bash_candidate.resolve())
        # Strategy 5: System bash (older, but guaranteed to exist)
        candidate = Path("/bin/bash")
        if candidate.exists():
            return str(candidate.resolve())

    bash = shutil.which("bash")
    if bash:
        return bash
    return None

class RunParams(BaseModel):
    model_config = {"populate_by_name": True}

    command: str = Field(
        default="",
        alias="cmd",  # backward compat: old "cmd" still works
        description=(
            "Executable command line. Only real executables / processes are accepted — "
            "No shell syntax (pipes, redirects, &&, ||, variables, etc.). "
            "Example: `python -c \"print(1)\"` or `git status`. "
            "Accepts `command` or `cmd` parameter."
        )
    )
    mode: Literal["execute", "send"] = mode_field(
        aliases=False,
        execute_desc="Run `command` as a direct process (default).",
        send_desc="Send `command` as stdin text to an existing session identified by `task_id`.",
    )
    shell: bool = Field(
        default=False,
        description=(
            "When True, execute `command` through the system shell (bash on Linux/macOS, "
            "powershell on Windows). Supports pipes, redirects, variables, etc. "
            "When False (default), execute `command` as a direct process without shell interpretation."
        ),
    )
    timeout: int = timeout_field()
    output_path: str | None = Field(
        default=None,
        description="Output file path."
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory."
    )
    env: str | list[str] | None = Field(
        default=None,
        description="Environment variables to set for the subprocess."
    )
    run_in_background: bool = Field(
        default=False,
        description="Run the process in the background and return immediately."
    )
    task_id: str | None = task_id_field("command")
    wait_for_pattern: str | None = wait_for_pattern_field()
    max_lines: int | None = max_lines_field()
    deduplicate_output: bool = deduplicate_output_field()

    @model_validator(mode="after")
    def _infer_mode(self) -> "RunParams":
        """Backward compat: when task_id is set but mode is default 'execute', auto-switch to 'send'."""
        if self.task_id is not None and self.mode == "execute":
            object.__setattr__(self, 'mode', 'send')
        return self

    @model_validator(mode="after")
    def _validate_cmd(self) -> "RunParams":
        if self.mode == "execute" and not self.command:
            raise ValueError("command cannot be empty when mode='execute'")
        if self.mode == "send":
            if not self.command:
                raise ValueError("command cannot be empty when mode='send'")
            if not self.task_id:
                raise ValueError("mode='send' requires task_id to identify the target session")
        if self.task_id is not None and self.mode != "send":
            raise ValueError("task_id requires mode='send'")
        return self


class Run(CallableTool2[RunParams]):
    name: str = "Run"
    description: str = (
        "Run an executable or bash command. "
        + _interactive_scope_text(is_shell=False)
    )
    params: type[RunParams] = RunParams

    def __init__(self, session: Session):
        super().__init__()
        if USE_SYSTEM_SHELL:
            if sys.platform == "win32" and USE_SYSTEM_PWSH_ON_WINDOWS:
                raise SkipThisTool()
            else:
                if find_bash() is not None:
                    raise SkipThisTool()
        self._session = session
        self._semaphore = asyncio.Semaphore(8)
        self.use_posix = sys.platform != "win32"

        # Pre-normalize forbidden commands once at init time for O(1) per-call lookup.
        raw_forbidden = self._session.custom_config.get(
            "config_json", {}).get("forbidden_commands", [])
        self._forbidden_keywords: list[str] = []
        seen: set[str] = set()
        for cmd in raw_forbidden:
            if not isinstance(cmd, str) or not cmd:
                continue
            normalized = " ".join(cmd.split())
            if normalized not in seen:
                seen.add(normalized)
                self._forbidden_keywords.append(normalized)

        # Config gates for the hardline safety floor and secret redaction
        # (both default on; explicitly set False to disable).
        shell_cfg = self._session.custom_config.get("config_json", {}).get("shell", {})
        if isinstance(shell_cfg, dict):
            self._hardline_enabled = shell_cfg.get("hardline", True)
            self._redact_secrets = shell_cfg.get("redact_secrets", True)
        else:
            self._hardline_enabled = True
            self._redact_secrets = True

    def _hardline_blocked(self, command: str) -> ToolError | None:
        r"""Return a ToolError when *command* hits the unconditional hardline floor.

        Applies deobfuscation variants (quotes/backslash-escapes/case tricks)
        before matching, so ``r\m -rf /``, ``rm "" -rf /`` and ``Rm -Rf /``
        are all blocked.  Skipped entirely when the ``shell.hardline`` config
        gate is explicitly ``False``.
        """
        if not self._hardline_enabled or not command:
            return None
        blocked, desc = check_hardline_blocked(command)
        if not blocked:
            return None
        return ToolError(
            output="",
            message=(
                f"Blocked (hardline): {desc}. This command cannot be executed "
                "via the agent."
            ),
            brief="Blocked (hardline)",
        )

    async def __call__(self, params: RunParams) -> ToolReturnValue:
        # Hardline safety floor: never spawn a process for destructive commands.
        blocked = self._hardline_blocked(params.command)
        if blocked is not None:
            return blocked

        workdir_err = validate_workdir(params.cwd)
        if workdir_err is not None:
            return ToolError(message=workdir_err, brief="Invalid workdir")

        # Mode-based dispatch
        if params.mode == "send":
            if not params.task_id:
                return ToolError(
                    output="",
                    message="mode='send' requires task_id to identify the target session.",
                    brief="Missing task_id",
                )
            return await self._continue_session(params)

        # Shell mode: delegate to Bash or Powershell
        if params.shell:
            return await self._run_via_shell(params)

        pattern = self._compile_pattern(params.wait_for_pattern)
        if isinstance(pattern, ToolError):
            return pattern

        script_path: str | None = None
        use_posix = self.use_posix

        # Split the full command into parts.
        # Use posix=False on Windows to preserve backslashes in paths;
        # on POSIX systems use posix=True for correct quoting/escaping.
        # double quotes are NOT stripped in posix=False mode, so we strip them below.
        cmd_parts: list[str] = shlex.split(params.command, posix=use_posix)
        if not cmd_parts:
            return ToolError(
                output="",
                message="Empty command.",
                brief="Empty command",
            )

        # -- Resolve the executable: progressive prefix lookup for paths with spaces --
        # First, strip outer double quotes from the first element if present.
        # (Only needed when posix=False; posix=True already strips shell quotes.)
        first = cmd_parts[0]
        if not use_posix and first.startswith('"') and first.endswith('"'):
            first = first[1:-1]
        executable = first
        args_raw: list[str] = cmd_parts[1:]

        if len(cmd_parts) > 1:
            # Try progressively longer prefixes to find an existing file,
            # so unquoted paths with spaces are handled correctly.
            for i in range(2, len(cmd_parts) + 1):
                candidate = " ".join(cmd_parts[:i])
                # Strip outer double quotes if present on the candidate
                if not use_posix and candidate.startswith('"') and candidate.endswith('"'):
                    candidate = candidate[1:-1]
                try:
                    if Path(candidate).is_file():
                        executable = candidate
                        args_raw = cmd_parts[i:]
                        break
                except OSError:
                    pass

        # Strip surrounding double quotes from each arg.
        # Only needed when posix=False, because posix=True already strips them.
        if not use_posix:
            args_list: list[str] = []
            for arg in args_raw:
                if arg.startswith('"') and arg.endswith('"'):
                    args_list.append(arg[1:-1])
                else:
                    args_list.append(arg)
        else:
            args_list = list(args_raw)

        try:
            # Check forbidden commands (pre-normalized in __init__)
            if self._forbidden_keywords:
                full_cmd = params.command
                normalized_cmd = " ".join(full_cmd.split())
                for keyword in self._forbidden_keywords:
                    if keyword in normalized_cmd:
                        return ToolError(
                            output="",
                            message=f"Command `{full_cmd}` is forbidden by config rule.",
                            brief="Forbidden command",
                        )

            # Check if executable is a valid process (in PATH or existing file),
            # then fall back to bash built-in commands.
            # Refresh PATH/PATHEXT from registry so that tools installed
            # since the last command (e.g. via WinGet) are discoverable.
            if sys.platform == "win32":
                from kimix.utils.windows_env import refresh_env_from_registry
                refresh_env_from_registry()

            is_process = False
            is_py = False
            if (executable == 'python' and (shutil.which('python') is None) and (not Path('./python').exists())) or (executable == 'python.exe' and (shutil.which('python.exe') is None) and (not Path('./python.exe').exists())):
                executable = sys.executable
                is_process = True
                is_py = True
            elif os.sep in executable or "/" in executable:
                # Contains path separator - check if it's an existing file
                is_process = Path(executable).is_file()
            else:
                # Bare command name - check if it's in PATH
                is_process = shutil.which(executable) is not None

            if not is_process:
                # Not a real process - check if it's a bash built-in command.
                error_msg = " This tool does not support shell commands; use `Bash` tool."
                return ToolError(
                    output='',
                    message=error_msg,
                    brief='Bash not supported.'
                )

            # Rewrite known commands through RTK using the share-bin binary only.
            rtk_rewritten = False
            rtk_path = _rtk_binary_path()
            if (
                params.deduplicate_output
                and rtk_path is not None
                and _is_known_rtk_command(Path(executable).stem)
                and executable not in ("rtk", "rtk.exe")
            ):
                args_list = [executable] + args_list
                executable = str(rtk_path)
                rtk_rewritten = True

            display_executable = "rtk" if rtk_rewritten else executable
            display_args = [
                arg[:100] + '...' if len(arg) > 100 else arg for arg in args_list]
            cmd_str = shlex.join([display_executable] + display_args)
            display_cmd = display_executable if len(
                cmd_str) > _HUGE_CMD_THRESHOLD else cmd_str

            # Handle extremely long python -c scripts via a script file in the
            # shared temp folder (Windows CreateProcessW ~32767 limit).
            if is_py:
                c_idx = next((i for i, a in enumerate(
                    args_list) if a == '-c'), None)
                if c_idx is not None and c_idx + 1 < len(args_list) and len(args_list[c_idx + 1]) > 30000:
                    script_path = _create_script_file(
                        args_list[c_idx + 1], ext='.py')
                    # Replace -c <code> with <script_path>, preserving leading options and trailing args
                    args_list = args_list[:c_idx] + \
                        [script_path] + args_list[c_idx + 2:]

            async with self._semaphore:
                env_dict: dict[str, str] | None = None
                if isinstance(params.env, str):
                    tokens = shlex.split(params.env)
                    env_items = []
                    i = 0
                    while i < len(tokens):
                        if i + 2 < len(tokens) and tokens[i + 1] == '=' and '=' not in tokens[i]:
                            env_items.append(f"{tokens[i]}={tokens[i + 2]}")
                            i += 3
                        else:
                            env_items.append(tokens[i])
                            i += 1
                else:
                    env_items = params.env
                if env_items:
                    env_dict = {}
                    for item in env_items:
                        if '=' in item:
                            key, value = item.split('=', 1)
                            env_dict[key] = value
                        else:
                            env_dict[item] = '1'
                task = ProcessTask(executable, args_list, params.cwd, _env_with_rg_bin_path(env_dict))
                task_id = await task.start(self._session, "run", Path(executable).stem)

                wait_matched: bool | None = None
                elapsed_seconds: float | None = None

                if params.run_in_background:
                    if params.wait_for_pattern is not None and task.stream is not None:
                        output, wait_matched, elapsed_seconds = await task.stream.wait_for_output(
                            timeout=params.timeout, pattern=pattern
                        )
                        return await self._format_session_result(
                            task_id, task.stream, params, output, "running",
                            wait_matched=wait_matched, elapsed_seconds=elapsed_seconds,
                            message=(f"[rtk] Running in background. task_id: `{task_id}`." if rtk_rewritten else f"Running in background. task_id: `{task_id}`."),
                            brief="Background task started",
                            rtk_rewritten=rtk_rewritten,
                        )
                    return ToolOk(
                        output="",
                        message=(f"[rtk] Running in background. task_id: `{task_id}`. Use `TaskOutput` tool to retrieve output." if rtk_rewritten else f"Running in background. task_id: `{task_id}`. Use `TaskOutput` tool to retrieve output."),
                        brief="Background task started",
                        display_block=ShellDisplayBlock(language="shell"),
                    )

                # Wait for completion (or a pattern match) with timeout.
                if params.wait_for_pattern is not None and task.stream is not None:
                    output, wait_matched, elapsed_seconds = await task.stream.wait_for_output(
                        timeout=params.timeout, pattern=pattern
                    )
                    if await task.thread_is_alive():
                        return await self._format_session_result(
                            task_id, task.stream, params, output, "running",
                            wait_matched=wait_matched, elapsed_seconds=elapsed_seconds,
                            message=(f"[rtk] Matched pattern, still running" if rtk_rewritten else "Matched pattern, still running"),
                            brief="Pattern matched",
                            rtk_rewritten=rtk_rewritten,
                        )
                else:
                    wait_timeout = params.timeout
                    await task.wait(wait_timeout)

                if await task.thread_is_alive():
                    output = await task.stream.pop_output() if task.stream else ""
                    output = await _maybe_export_output_async(output)
                    guidance = foreground_background_guidance(params.command)
                    message = f"Running in background. task_id: `{task_id}`. use `TaskOutput`"
                    if guidance:
                        message += f" {guidance}"
                    return ToolError(
                        output=output,
                        message=message,
                        brief="Timeout",
                    )
                # Clean up foreground task registration
                from kimix.tools.background.utils import remove_task_id
                remove_task_id(self._session, task_id)

                # Get output
                output = await task.stream.pop_output() if task.stream else ""

                # Secret redaction runs first (config-gated) so the dedup/
                # export/summarize pipeline never sees credentials.
                if self._redact_secrets and output:
                    output = redact_sensitive_output(output)

                # Apply token filter pipeline (dedup, truncate)
                output, original_path = await _token_filter_output(
                    output,
                    token_kill=params.deduplicate_output,
                    max_lines=params.max_lines,
                    rtk_rewritten=rtk_rewritten,
                )

                # Optionally offload a very long output to a sub-agent
                output_truncated = False
                if len(output) > 65536 and not params.output_path:
                    output = await _summarize_long_output_async(
                        self._session, params.command, output
                    )
                    output_truncated = True

                # Handle output export if needed
                output_path: str | None = None
                if params.output_path:
                    async with await anyio.open_file(params.output_path, 'w', encoding='utf-8', errors='replace') as f:
                        await f.write(output)
                    display_path = params.output_path.replace("\\", "/")
                    output_path = display_path
                    output = f'saved to file `{display_path}`'

                # Check success
                success = await task.stream.success() if task.stream else False
                real_exit_code = task.stream.exit_code if task.stream else None
                meaning = interpret_exit_code(params.command, real_exit_code)
                hint = annotate_failure(output, params.command, real_exit_code)

                if not success:
                    if output and not params.output_path:
                        if len(output) > 65536:
                            output = await _summarize_long_output_async(
                                self._session, params.command, output
                            )
                            output_truncated = True
                        else:
                            temp_path, _ = await _export_to_temp_file_async(key=None, content=output, ext='.txt')
                            display_temp_path = temp_path.replace("\\", "/")
                            output_path = display_temp_path
                            output = f'saved to file `{display_temp_path}`'
                    block = _build_session_output_block(
                        task_id=task_id,
                        status="completed",
                        output=output,
                        exit_code=real_exit_code,
                        exit_code_meaning=meaning,
                        failure_hint=hint,
                        wait_matched=wait_matched,
                        elapsed_seconds=elapsed_seconds,
                        output_path=output_path,
                        output_truncated=output_truncated,
                        original_path=original_path,
                    )
                    elapsed = task.stream.process_elapsed if task.stream else None
                    msg = "[rtk] failed" if rtk_rewritten else "failed"
                    if hint:
                        msg += f" Hint: {hint}"
                    return ToolError(
                        output=block,
                        message=msg,
                        brief="Command execution failed",
                    )

                output = await _maybe_export_output_async(output)
                if not output_path:
                    output_path = _extract_export_path(output)
                block = _build_session_output_block(
                    task_id=task_id,
                    status="completed",
                    output=output,
                    exit_code=real_exit_code,
                    exit_code_meaning=meaning,
                    failure_hint=hint,
                    wait_matched=wait_matched,
                    elapsed_seconds=elapsed_seconds,
                    output_path=output_path,
                    output_truncated=output_truncated,
                    original_path=original_path,
                )
                elapsed = task.stream.process_elapsed if task.stream else None
                msg = "[rtk] success" if rtk_rewritten else "success"
                return ToolOk(
                    output=block,
                    message=msg,
                    brief="Command executed successfully",
                    display_block=ShellDisplayBlock(language="shell"),
                )
        except Exception as e:
            return ToolError(
                output='',
                message='Internal error, quit current session now.',
                brief='Internal error'
            )
        finally:
            if script_path is not None:
                try:
                    os.remove(script_path)
                except Exception:
                    pass

    async def _run_via_shell(self, params: RunParams) -> ToolReturnValue:
        """Execute the command via the system shell (bash/pwsh)."""
        if sys.platform == "win32":
            from kimix.tools.file.bash.pwsh_tool import Powershell, PowershellParams
            try:
                pwsh = Powershell(self._session)
            except Exception:
                return ToolError(
                    output="",
                    message="PowerShell is not available on this system.",
                    brief="PowerShell unavailable",
                )
            ps_params = PowershellParams(
                cmd=params.command,
                timeout=params.timeout,
                deduplicate_output=params.deduplicate_output,
                max_lines=params.max_lines,
                wait_for_pattern=params.wait_for_pattern,
                cwd=params.cwd,
                interactive=False,
            )
            return await pwsh.__call__(ps_params)
        else:
            from kimix.tools.file.bash.bash_tool import Bash, BashParams
            try:
                bash = Bash(self._session)
            except Exception:
                return ToolError(
                    output="",
                    message="Bash is not available on this system.",
                    brief="Bash unavailable",
                )
            bash_params = BashParams(
                cmd=params.command,
                timeout=params.timeout,
                deduplicate_output=params.deduplicate_output,
                max_lines=params.max_lines,
                wait_for_pattern=params.wait_for_pattern,
                cwd=params.cwd,
                interactive=False,
            )
            return await bash.__call__(bash_params)

    def _compile_pattern(self, wait_for_pattern: str | None) -> re.Pattern[str] | ToolError:
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

    async def _continue_session(self, params: RunParams) -> ToolReturnValue:
        """Send input to an existing Run session and optionally wait for output."""
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

        input_text = params.command
        if not input_text.endswith("\n"):
            input_text += "\n"
        if not await stream.input(input_text):
            return ToolError(
                output="",
                message=f"Failed to send input to task '{task_id}'",
                brief="Send input failed",
            )

        output, matched, elapsed = await stream.wait_for_output(
            timeout=params.timeout, pattern=pattern
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
        self, params: RunParams, output: str, rtk_rewritten: bool = False
    ) -> tuple[str, str | None, bool, str | None]:
        """Summarize/export long output. Returns (display_output, path, truncated, original_path)."""
        # Secret redaction runs first (config-gated) so the dedup/export/
        # summarize pipeline never sees credentials.
        if self._redact_secrets and output:
            output = redact_sensitive_output(output)
        # Run token filter pipeline (dedup, truncate)
        output, original_path = await _token_filter_output(
            output,
            token_kill=params.deduplicate_output,
            max_lines=params.max_lines,
            rtk_rewritten=rtk_rewritten,
        )
        output_truncated = False
        if len(output) > 65536:
            output = await _summarize_long_output_async(
                self._session, params.command, output
            )
            output_truncated = True
        output = await _maybe_export_output_async(output)
        output_path = _extract_export_path(output)
        return output, output_path, output_truncated, original_path

    async def _format_session_result(
        self,
        task_id: str,
        stream: 'BackgroundStream' | None,
        params: RunParams,
        output: str,
        status: str,
        *,
        wait_matched: bool | None,
        elapsed_seconds: float | None,
        message: str,
        brief: str,
        rtk_rewritten: bool = False,
        exit_code_meaning: str | None = None,
        failure_hint: str | None = None,
    ) -> ToolReturnValue:
        """Build a ToolOk response with a structured output block."""
        processed, output_path, output_truncated, original_path = await self._process_output(
            params, output, rtk_rewritten=rtk_rewritten
        )
        if status != "completed":
            real_exit_code = None
        else:
            real_exit_code = stream.exit_code if stream else None
            if real_exit_code is None and stream is not None:
                real_exit_code = 0 if await stream.success() else None
        if exit_code_meaning is None and real_exit_code is not None:
            exit_code_meaning = interpret_exit_code(params.command, real_exit_code)
        block = _build_session_output_block(
            task_id=task_id,
            status=status,
            output=processed,
            exit_code=real_exit_code,
            exit_code_meaning=exit_code_meaning,
            failure_hint=failure_hint,
            wait_matched=wait_matched,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            output_truncated=output_truncated,
            original_path=original_path,
        )
        return ToolOk(output=block, message=message, brief=brief)
