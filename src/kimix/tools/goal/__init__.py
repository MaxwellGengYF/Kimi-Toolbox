"""Goal tracking tool.

A single-goal equivalent of TodoList: defines the ultimate project goal as
executable Python code (inline or a .py file path) and tracks whether it has
been successfully executed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Literal

import orjson
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.soul.agent import Runtime
from pydantic import AliasChoices, BaseModel, Field


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("code", "code_file"),
        description=(
            "Python code inline or a `.py` file path. "
            "Omit to read the current goal. "
            "Pass empty string to clear the goal. "
            "Accepts `code` or `code_file`."
        ),
    )
    mode: Literal["overwrite", "append", "force_overwrite"] = Field(
        default="append",
        description=(
            "Write mode: "
            "'overwrite' replaces the current goal only when it is already done; "
            "'append' (default) sets or updates the goal; "
            "'force_overwrite' unconditionally replaces the goal."
        ),
    )


class Goal(CallableTool2[Params]):
    name: str = "Goal"
    description: str = (
        "Track a single ultimate project goal as executable Python code. "
        "When called with no arguments, returns the current goal and its status. "
        "Set a goal by providing `code` (inline Python or a .py file path). "
        "Before finishing the session, the system ensures the goal code is "
        "executed and its status is `done`. "
        "Mark the goal as `done` once the code has been successfully executed "
        "and passes all acceptance criteria."
    )
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime
        self._script_counter = 0

    async def __call__(self, params: Params) -> ToolReturnValue:
        # --- If code is None → read current goal (no arguments passed) ---
        if params.code is None:
            return self._read_goal()

        # --- If code is empty string → clear the goal ---
        if not params.code.strip():
            save_err = self._save_goal(None)
            if save_err:
                return ToolError(
                    output="",
                    message=save_err,
                    brief="Failed to clear goal",
                )
            return ToolOk(
                output="Goal cleared.",
                message="Goal has been removed.",
                brief="Goal cleared",
            )

        # --- Resolve the goal code ---
        code, is_file, temp_file_path = self._resolve_goal_code(params.code)

        # --- Load current goal for mode checks ---
        current_goal = self._load_goal()

        # --- Mode-specific merge logic ---
        if params.mode == "force_overwrite":
            # Unconditional replace
            pass  # use resolved values as-is

        elif params.mode == "overwrite":
            if current_goal is not None and current_goal.get("status") != "done":
                return ToolError(
                    output=(
                        "Error: Cannot overwrite goal while it is not done. "
                        "Use mode='force_overwrite' to replace it unconditionally."
                    ),
                    message="Goal is not done yet; use force_overwrite to replace.",
                    brief="Goal not done",
                )

        else:  # append — default, just set/update the goal
            # If goal already exists and is done, replace it (like plan says)
            if current_goal is not None and current_goal.get("status") == "done":
                pass  # replace regardless
            # Otherwise, keep existing attributes that aren't being changed
            # For append mode, if goal exists and is not done, update code
            # (new code replaces old, status resets to pending)

        # --- Build new goal state ---
        goal_state: dict[str, Any] = {
            "code": code,
            "status": "pending",
            "is_file": is_file,
            "temp_file_path": temp_file_path,
        }

        save_err = self._save_goal(goal_state)
        if save_err:
            return ToolError(
                output="",
                message=save_err,
                brief="Failed to save goal",
            )

        # --- Build response ---
        lines = ["Goal set."]
        if is_file:
            lines.append(f"Code reference: `{code}`")
        else:
            lines.append(f"Goal code saved to `{temp_file_path}`" if temp_file_path else "")
        lines.append(f"Status: {goal_state['status']}")
        status_summary = "\n".join(line for line in lines if line)

        return ToolOk(
            output=status_summary,
            message=status_summary,
            brief="Goal set",
        )

    # ---- Read mode --------------------------------------------------------

    def _read_goal(self) -> ToolReturnValue:
        """Read and return the current goal state."""
        goal = self._load_goal()
        if goal is None:
            return ToolOk(
                output="No goal is set.",
                message="No goal is set. Use `Goal` with `code` to define one.",
                brief="No goal set",
            )
        code = goal.get("code", "")
        status = goal.get("status", "pending")
        is_file = goal.get("is_file", False)
        temp_path = goal.get("temp_file_path", None)

        lines = [f"Current goal status: {status}"]
        if is_file:
            lines.append(f"Code reference: `{code}`")
        else:
            lines.append(f"Goal code saved to `{temp_path}`" if temp_path else "")
        output = "\n".join(line for line in lines if line)
        return ToolOk(
            output=output,
            message=output,
            brief=f"Goal status: {status}",
        )

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------

    def _resolve_goal_code(self, code: str) -> tuple[str, bool, str | None]:
        """Resolve the goal code to an executable form.

        Returns ``(code, is_file, temp_file_path)`` where:
        - ``code`` is the actual Python code or file path.
        - ``is_file`` is True when code refers to an existing .py file.
        - ``temp_file_path`` is the path to a temp file (when inline code was saved),
          or None otherwise.
        """
        # Priority 1: .py file path that exists
        code_path = Path(code)
        if code.endswith(".py") and code_path.is_file():
            return code, True, None

        # Priority 2: inline code — save to a temp file
        session_dir = Path(self._runtime.session.dir)
        script_name = f"goal_{self._script_counter}.py"
        self._script_counter += 1
        script_path = session_dir / script_name
        session_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")
        return code, False, str(script_path)


    @staticmethod
    def _resolve_goal_executable(goal_state: dict[str, Any]) -> str | None:
        """Resolve the executable file path from a goal state dict.

        Returns the path to a ``.py`` file that can be run, or ``None`` if
        the goal has no runnable code.
        """
        code = goal_state.get("code", "")
        if not code:
            return None

        is_file = goal_state.get("is_file", False)
        temp_file_path = goal_state.get("temp_file_path")

        # 1. Temp file from an earlier inline-code save
        if temp_file_path:
            tp = Path(temp_file_path)
            if tp.exists():
                return str(tp)

        # 2. Existing .py file reference
        if is_file:
            fp = Path(code)
            if fp.is_file():
                return str(fp)

        # 3. Write inline code to a new temp file
        import tempfile as _tf
        fd, path = _tf.mkstemp(suffix=".py", prefix="run_goal_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    @staticmethod
    async def _run_goal_code(
        goal_state: dict[str, Any],
        timeout: int = 30,
        python_exe: str | None = None,
    ) -> tuple[bool, str]:
        """Execute the goal code and return ``(success, output_or_error)``."""
        executable = Goal._resolve_goal_executable(goal_state)
        if executable is None:
            return False, "Goal has no runnable code."

        if python_exe is None:
            python_exe = sys.executable

        try:
            proc = await asyncio.create_subprocess_exec(
                python_exe,
                executable,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Goal execution timed out after {timeout}s."

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                error_text = stderr.decode("utf-8", errors="replace")
                if output:
                    output += "\n" + error_text
                else:
                    output = error_text

            if proc.returncode == 0:
                return True, output or "Goal executed successfully (no output)."
            else:
                return False, f"Goal failed (exit code {proc.returncode}):\n{output}"
        except FileNotFoundError:
            return False, f"Python executable not found: {python_exe}"
        except Exception as exc:
            return False, f"Goal execution error: {exc}"

    # ------------------------------------------------------------------
    # Persistence — mirrors ``kimi_cli.tools.todo.TodoList`` split:
    #   root  → session.custom_data["goal"]
    #   subagent → state.json under instance dir
    # ------------------------------------------------------------------

    def _load_goal(self) -> dict[str, Any] | None:
        """Load the current goal state, or None if no goal is set."""
        if self._runtime.role == "root":
            raw = self._runtime.session.custom_data.get("goal")
            if isinstance(raw, dict):
                return raw
            return None

        state_file = self._subagent_state_file()
        if state_file is None:
            return None
        data = self._read_subagent_state(state_file)
        raw = data.get("goal")
        if isinstance(raw, dict):
            return raw
        return None

    def _save_goal(self, goal: dict[str, Any] | None) -> str | None:
        """Persist the goal state. Returns error message string on failure, None on success."""
        if self._runtime.role == "root":
            try:
                if goal is None:
                    self._runtime.session.custom_data.pop("goal", None)
                else:
                    self._runtime.session.custom_data["goal"] = goal
                return None
            except Exception as exc:
                return f"Error: Failed to save goal: {exc}"

        # Subagent persistence
        state_file = self._subagent_state_file()
        if state_file is None:
            return "Error: Unable to save subagent goal: state file is not available."
        data = self._read_subagent_state(state_file)
        if goal is None:
            data.pop("goal", None)
        else:
            data["goal"] = goal
        try:
            self._write_subagent_state(state_file, data)
        except Exception as exc:
            return f"Error: Failed to save subagent goal: {exc}"
        return None

    # ------------------------------------------------------------------
    # Subagent helpers — identical pattern to ``kimi_cli.tools.todo.TodoList``
    # ------------------------------------------------------------------

    def _subagent_state_file(self) -> Path | None:
        store = self._runtime.subagent_store
        agent_id = self._runtime.subagent_id
        if store is None or agent_id is None:
            return None
        return store.instance_dir(agent_id) / "state.json"

    @staticmethod
    def _read_subagent_state(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = orjson.loads(path.read_text(encoding="utf-8"))
        except (orjson.JSONDecodeError, OSError, UnicodeDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _write_subagent_state(path: Path, data: dict[str, Any]) -> None:
        from kimi_cli.utils.io import atomic_json_write

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(data, path)



class RunGoalParams(BaseModel):
    model_config = {"populate_by_name": True}

    timeout: int = Field(
        default=30,
        ge=1,
        le=900,
        description="Timeout in seconds for goal code execution (1-900).",
    )


class RunGoal(CallableTool2[RunGoalParams]):
    name: str = "RunGoal"
    description: str = (
        "Execute the current session goal code. "
        "If a goal is set, runs its Python code and returns success or error. "
        "If the code executes successfully (exit code 0), the goal is cleared. "
        "If no goal is set, reports that no goal exists."
    )
    params: type[RunGoalParams] = RunGoalParams

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    async def __call__(self, params: RunGoalParams) -> ToolReturnValue:
        # Load the current goal using the same persistence as Goal
        goal = self._load_goal()
        if goal is None:
            return ToolOk(
                output="No goal is set. Nothing to run.",
                message="No goal is set.",
                brief="No goal set",
            )

        if goal.get("status") == "done":
            # Already done — clear and report
            self._save_goal(None)
            return ToolOk(
                output="Goal was already done. Cleared.",
                message="Goal was already done; cleared.",
                brief="Goal already done",
            )

        # Resolve the goal executable path
        executable = Goal._resolve_goal_executable(goal)
        if executable is None:
            return ToolError(
                output="",
                message="Goal has no runnable code.",
                brief="Goal not runnable",
            )

        # Mark as in_progress
        goal["status"] = "in_progress"
        self._save_goal(goal)

        # Run the code
        try:
            success, output = await Goal._run_goal_code(goal, timeout=params.timeout)
        except Exception as exc:
            goal["status"] = "pending"
            self._save_goal(goal)
            return ToolError(
                output=str(exc),
                message=f"Goal execution raised an unexpected error: {exc}",
                brief="Goal execution error",
            )

        if success:
            # Clear the goal on success
            self._save_goal(None)
            return ToolOk(
                output=output,
                message=f"Goal executed successfully. Goal cleared.\n{output}",
                brief="Goal succeeded",
            )
        else:
            # Restore to pending on failure
            goal["status"] = "pending"
            self._save_goal(goal)
            return ToolError(
                output=output,
                message=output,
                brief="Goal execution failed",
            )

    # ---- Reuse Goal persistence helpers ----------------------------------

    def _load_goal(self) -> dict[str, Any] | None:
        if self._runtime.role == "root":
            raw = self._runtime.session.custom_data.get("goal")
            if isinstance(raw, dict):
                return raw
            return None
        state_file = self._subagent_state_file()
        if state_file is None:
            return None
        data = Goal._read_subagent_state(state_file)
        raw = data.get("goal")
        if isinstance(raw, dict):
            return raw
        return None

    def _save_goal(self, goal: dict[str, Any] | None) -> str | None:
        if self._runtime.role == "root":
            try:
                if goal is None:
                    self._runtime.session.custom_data.pop("goal", None)
                else:
                    self._runtime.session.custom_data["goal"] = goal
                return None
            except Exception as exc:
                return f"Error: Failed to save goal: {exc}"
        state_file = self._subagent_state_file()
        if state_file is None:
            return "Error: Unable to save subagent goal: state file is not available."
        data = Goal._read_subagent_state(state_file)
        if goal is None:
            data.pop("goal", None)
        else:
            data["goal"] = goal
        try:
            Goal._write_subagent_state(state_file, data)
        except Exception as exc:
            return f"Error: Failed to save subagent goal: {exc}"
        return None

    def _subagent_state_file(self) -> Path | None:
        store = self._runtime.subagent_store
        agent_id = self._runtime.subagent_id
        if store is None or agent_id is None:
            return None
        return store.instance_dir(agent_id) / "state.json"
