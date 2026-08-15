"""Background task management tools."""
import sys
import asyncio

import regex as re
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import AliasChoices, BaseModel, Field, model_validator
from typing import Literal
from kimi_cli.session import Session

from .utils import generate_task_id, remove_task_id, add_task, get_all_tasks, BackgroundStream, discard_all_tasks
from kimix.tools.common import _maybe_export_output_async, _maybe_export_rtk_original_async, _original_saved_message
from kimix.tools.prompt_common import accepts_alias_text, wait_for_pattern_field
from kimi_cli.tools.display import BackgroundTaskDisplayBlock


class TaskOutputParams(BaseModel):
    """Parameters for job_output (TaskOutput)."""
    model_config = {"populate_by_name": True}

    job_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("job_id", "task_id"),
        description=(
            "Job id returned by the tool that started the background work. "
            "When None, lists all tasks. "
            + accepts_alias_text("job_id", "task_id", word=False)
        ),
    )
    action: Literal["get", "list", "kill"] = Field(
        default="get",
        description=(
            "'get': Return output from the job specified by `job_id` (default). "
            "'list': List all jobs (when job_id is empty). "
            "'kill': Force-stop the job specified by `job_id` and return its final output."
        ),
    )
    wait: bool = Field(
        default=False,
        validation_alias=AliasChoices("wait", "block"),
        description=(
            "Block until the job reaches a terminal status or the timeout expires. "
            "A timed-out wait returns [status: running] and leaves the job alive. "
            + accepts_alias_text("wait", "block", word=False)
            + " When False (default), return immediately with whatever output is "
            "available so far."
        ),
    )
    timeout_ms: int = Field(
        default=60000,
        validation_alias=AliasChoices("timeout_ms", "timeout"),
        ge=1,
        le=7200000,
        description=(
            "Max wait in milliseconds (only meaningful with wait: true). "
            "Defaults to the configured wait timeout; capped by the configured "
            "maximum. "
            + accepts_alias_text("timeout_ms", "timeout", word=False)
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="Output file path."
    )
    wait_for_pattern: str | None = wait_for_pattern_field()
    kill: bool = Field(
        default=False,
        description="[Deprecated] Use action='kill' instead.",
    )

    @property
    def timeout(self) -> int:
        """Legacy accessor: timeout in seconds (converted from ``timeout_ms``)."""
        return max(1, self.timeout_ms // 1000)

    @model_validator(mode="after")
    def _normalize_kill(self) -> "TaskOutputParams":
        """Convert deprecated kill=True to action='kill'."""
        if self.kill:
            object.__setattr__(self, 'action', 'kill')
        return self


class TaskOutput(CallableTool2):
    """Get output from a background task, or list all tasks if no job_id is provided."""
    name: str = "job_output"
    description: str = (
        "Read a background job. Stream jobs return only output since the "
        "previous read; final-output jobs return their result after settlement. "
        "Every response ends with `[status: ...]`. Reads are non-blocking "
        "unless `wait: true`, which waits up to the configured cap."
    )
    params: type[BaseModel] = TaskOutputParams

    def __del__(self):
        if sys.is_finalizing():
            return
        session = getattr(self, '_session', None)
        if session is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(discard_all_tasks(session))
                    )
            except RuntimeError:
                pass

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: TaskOutputParams) -> ToolReturnValue:
        """Return the output of a task_id, or list all tasks if task_id is None."""
        try:
            tasks = get_all_tasks(self._session)

            # Action: list all tasks (when action='list' OR task_id is None AND action is default)
            if params.action == "list" or (params.job_id is None and params.action == "get"):
                return await self._list_tasks(tasks)

            # Action: kill a specific task
            if params.action == "kill":
                if not params.job_id:
                    return ToolError(
                        message="task_id is required for action='kill'.",
                        output="",
                        brief="Missing task_id",
                    )
                return await self._kill_task(tasks, params)

            # Action: get output (default)
            return await self._get_output(tasks, params)

        except Exception as e:
            return ToolError(
                message=str(e),
                output="Failed to get task output",
                brief="Task output error"
            )

    async def _list_tasks(self, tasks: dict) -> ToolReturnValue:
        """Return a structured list of all background tasks."""
        tasks_info = []
        for task_id, stream in tasks.items():
            if await stream.is_started():
                alive = await stream.thread_is_alive()
                tasks_info.append({
                    "task_id": task_id,
                    "kind": task_id.split("_")[0] if "_" in task_id else "unknown",
                    "status": "running" if alive else "completed",
                    "elapsed": stream.process_elapsed,
                })

        if not tasks_info:
            return ToolOk(output="No running tasks.", brief="No background tasks")

        # Human-readable markdown table
        lines = ["| Task ID | Kind | Status | Elapsed |", "|---------|------|--------|---------|"]
        for t in tasks_info:
            elapsed_str = f"{t['elapsed']:.1f}s" if t['elapsed'] else "-"
            lines.append(f"| `{t['task_id']}` | {t['kind']} | {t['status']} | {elapsed_str} |")
        output = "\n".join(lines)

        result = ToolOk(output=output, brief=f"{len(tasks_info)} background task(s)")
        result.extras = {"tasks": tasks_info}
        return result

    async def _kill_task(self, tasks: dict, params: TaskOutputParams) -> ToolReturnValue:
        """Kill a specific task and return its final output."""
        stream: BackgroundStream | None = tasks.get(params.job_id.strip())
        if stream is None:
            started = [tid for tid, s in tasks.items() if await s.is_started()]
            if not started:
                return ToolError(
                    message="No running task",
                    output="",
                    brief="No running task"
                )
            task_list = ", ".join(started)
            return ToolError(
                message=f"Task '{params.job_id}' not found. Available tasks: [{task_list}]",
                output="",
                brief=f"Task '{params.job_id}' not found"
            )

        await stream.stop()
        output = await stream.pop_output()
        remove_task_id(self._session, params.job_id.strip())

        processed, message, original_path, _output_path, _output_truncated = await self._process_completed_output(
            stream, output, None
        )
        success = await stream.success()
        if not success:
            elapsed = stream.process_elapsed
            if elapsed is not None:
                message += f" ({elapsed:.1f}s)"
            return ToolError(
                message=message,
                output=processed if processed else "",
                brief=f"Task '{params.job_id}' killed (non-zero exit)"
            )

        return ToolOk(
            output=processed if processed else "(no output)",
            message=message,
            brief=f"Task '{params.job_id}' killed",
        )

    async def _process_completed_output(
        self,
        stream: BackgroundStream,
        output: str,
        wait_matched: bool | None,
    ) -> tuple[str, str, str | None, str | None, bool]:
        """Apply the originating tool's output formatter if one was registered.

        Returns ``(processed_output, message, original_path, output_path,
        output_truncated)``.  When no formatter is available, the legacy generic
        post-processing (rtk export + large-output export) is used.
        """
        if stream.format_output is not None:
            success = await stream.success()
            exit_code = stream.exit_code
            elapsed = stream.process_elapsed
            return await stream.format_output(
                output, success, exit_code, elapsed, wait_matched
            )

        rtk_original_path: str | None = None
        if output:
            rtk_original_path, _ = await _maybe_export_rtk_original_async(output)
        processed = await _maybe_export_output_async(output)
        message = _original_saved_message(rtk_original_path)
        return processed, message, rtk_original_path, None, False

    async def _get_output(self, tasks: dict, params: TaskOutputParams) -> ToolReturnValue:
        """Get output from a specific task."""
        stream: BackgroundStream | None = tasks.get(params.job_id.strip())
        if stream is None:
            started = [tid for tid, s in tasks.items() if await s.is_started()]
            if not started:
                return ToolError(
                    message="No running task",
                    output="",
                    brief="No running task"
                )
            task_list = ", ".join(started)
            return ToolError(
                message=f"Task '{params.job_id}' not found. Available tasks: [{task_list}]",
                output="",
                brief=f"Task '{params.job_id}' not found"
            )

        timeout = params.timeout
        wait_matched: bool | None = None
        if params.wait_for_pattern is not None:
            try:
                pattern = re.compile(params.wait_for_pattern)
            except re.error as exc:
                return ToolError(
                    message=f"Invalid wait_for_pattern: {exc}",
                    output="",
                    brief="Invalid pattern",
                )
            if params.wait:
                inactivity_timeout = min(900, timeout)
                output, wait_matched, _elapsed = await stream.wait_for_output(
                    timeout=timeout,
                    pattern=pattern,
                    inactivity_timeout=inactivity_timeout,
                )
                task_alive = await stream.thread_is_alive()
            else:
                output = await stream.pop_output()
                task_alive = await stream.thread_is_alive()
        else:
            if params.wait:
                inactivity_timeout = min(900, timeout)
                completed, _elapsed, _inactivity_timed_out = await stream.wait_with_inactivity_timeout(
                    timeout, inactivity_timeout
                )
                task_alive = not completed
            else:
                task_alive = await stream.thread_is_alive()

            # Use pop_output to ensure each call returns only new data
            output = await stream.pop_output()

        if not task_alive:
            remove_task_id(self._session, params.job_id.strip())
            processed, message, original_path, _output_path, _output_truncated = await self._process_completed_output(
                stream, output, wait_matched
            )
            if not await stream.success():
                elapsed = stream.process_elapsed
                if elapsed is not None:
                    message += f" ({elapsed:.1f}s)"
                if params.output_path:
                    from pathlib import Path
                    import anyio
                    path = Path(params.output_path)
                    async with await anyio.open_file(path, 'w', encoding='utf-8') as f:
                        await f.write(output)
                    display_path = str(path).replace("\\", "/")
                    output_text = f"output exported to file `{display_path}`"
                else:
                    output_text = processed if processed else "(no output)"
                return ToolError(
                    message=message,
                    output=output_text,
                    brief=f"Task '{params.job_id}' failed"
                )
        else:
            processed = await _maybe_export_output_async(output)
            message = ""
            original_path = None
            if output:
                rtk_original_path, _ = await _maybe_export_rtk_original_async(output)
                if rtk_original_path:
                    message = _original_saved_message(rtk_original_path)
                    original_path = rtk_original_path

        if params.output_path:
            from pathlib import Path
            import anyio
            path = Path(params.output_path)
            async with await anyio.open_file(path, 'w', encoding='utf-8') as f:
                await f.write(output)
            display_path = str(path).replace("\\", "/")
            output_text = f"{f'`{params.job_id}` is still running, call `job_output` again, ' if task_alive else ''}output exported to file `{display_path}`"
        else:
            output_text = processed if processed else "(no output)"
            if not task_alive and original_path:
                display_path = original_path.replace("\\", "/")
                if stream.format_output is not None:
                    output_text += f"\n[original output exported to: {display_path}]"
                else:
                    output_text += f"\n[rtk output exported to: {display_path}]"

        if wait_matched is not None:
            output_text += f"\nwait_matched: {str(wait_matched).lower()}"
        if not task_alive:
            elapsed = stream.process_elapsed
            if elapsed is not None:
                output_text += f"\n[Process completed in {elapsed:.2f}s]"

        # For completed tasks with a registered formatter, the message already
        # contains the original-saved / command-saved suffixes.  Otherwise,
        # derive it from any exported original path.
        if not message and original_path:
            message = _original_saved_message(original_path)

        kind = params.job_id.split("_")[0] if params.job_id else "task"
        status = "running" if task_alive else "completed"
        return ToolOk(
            output=output_text,
            message=message,
            brief="Task output retrieved",
            display_block=BackgroundTaskDisplayBlock(
                task_id=params.job_id,
                kind=kind,
                status=status,
                description=output_text[:200] if output_text else "(no output)",
            ),
        )



__all__ = [
    # Tool classes
    "TaskOutput",
    "TaskOutputParams",
    # Utility functions
    "generate_task_id",
    "remove_task_id",
    "add_task",
    "get_all_tasks",
]
