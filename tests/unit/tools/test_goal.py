"""Comprehensive tests for the Goal tool."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest
from pydantic import ValidationError

from kimix.tools.goal import Params, Goal


# ── Params validation ──────────────────────────────────────────────────


class TestGoalParams:
    def test_defaults(self) -> None:
        """Called with no arguments — code=None (read mode), mode='append'."""
        p = Params()
        assert p.code is None
        assert p.mode == "append"

    def test_code_alias_code_file(self) -> None:
        """Accepts `code_file` as an alias for `code`."""
        p = Params(code_file="print('hello')")
        assert p.code == "print('hello')"

    def test_explicit_empty_string(self) -> None:
        """Empty string is preserved to signal 'clear goal'."""
        p = Params(code="")
        assert p.code == ""

    def test_inline_code(self) -> None:
        p = Params(code="import sys; print(sys.version)")
        assert p.code == "import sys; print(sys.version)"
        assert p.mode == "append"

    @pytest.mark.parametrize("valid_mode", ["overwrite", "append", "force_overwrite"])
    def test_valid_modes(self, valid_mode: str) -> None:
        p = Params(code="x=1", mode=valid_mode)
        assert p.mode == valid_mode

    @pytest.mark.parametrize("invalid_mode", [
        "force", "forced", "write", "set", "put", "merge", "update", "replace",
    ])
    def test_invalid_modes_rejected(self, invalid_mode: str) -> None:
        with pytest.raises(ValidationError):
            Params(code="x=1", mode=invalid_mode)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session() -> MagicMock:
    """A CLI session mock with a mutable custom_data dict and a temp dir."""
    session = MagicMock()
    session.custom_data = {}
    session.dir = tempfile.mkdtemp()
    return session


@pytest.fixture
def mock_runtime_root(mock_session: MagicMock) -> MagicMock:
    """Runtime with role='root' — goal stored in session.custom_data."""
    runtime = MagicMock()
    runtime.role = "root"
    runtime.session = mock_session
    return runtime


@pytest.fixture
def goal_tool_root(mock_runtime_root: MagicMock) -> Goal:
    return Goal(runtime=mock_runtime_root)


@pytest.fixture
def sample_py_file(tmp_path: Path) -> Path:
    """A real .py file for file-reference tests."""
    p = tmp_path / "test_check.py"
    p.write_text("assert 1 + 1 == 2\n")
    return p


# ── Subagent fixtures ──────────────────────────────────────────────────


class FakeSubagentStore:
    """Minimal subagent store backed by a temp directory."""
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def instance_dir(self, agent_id: str) -> Path:
        path = self._base_dir / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path


@pytest.fixture
def subagent_store(tmp_path: Path) -> FakeSubagentStore:
    return FakeSubagentStore(tmp_path / "subagents")


@pytest.fixture
def mock_runtime_subagent(subagent_store: FakeSubagentStore) -> MagicMock:
    """Runtime with role='subagent' — goal stored in state.json."""
    runtime = MagicMock()
    runtime.role = "subagent"
    runtime.subagent_store = subagent_store
    runtime.subagent_id = "agent-test-001"
    # Session is needed for _resolve_goal_code (temp file dir)
    session = MagicMock()
    session.dir = tempfile.mkdtemp()
    runtime.session = session
    return runtime


@pytest.fixture
def goal_tool_subagent(mock_runtime_subagent: MagicMock) -> Goal:
    return Goal(runtime=mock_runtime_subagent)


# ── Read mode ──────────────────────────────────────────────────────────


class TestGoalRead:
    """Calling Goal with no arguments (code=None) reads the current goal."""

    async def test_read_no_goal(self, goal_tool_root: Goal) -> None:
        result = await goal_tool_root(Params())
        assert not result.is_error
        assert "No goal is set" in result.output

    async def test_read_goal_exists(self, goal_tool_root: Goal) -> None:
        # Pre-set a goal in custom_data
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "print('hello')",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        }
        result = await goal_tool_root(Params())
        assert not result.is_error
        assert "pending" in result.output

    async def test_read_goal_done(self, goal_tool_root: Goal) -> None:
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "print('hello')",
            "status": "done",
            "is_file": False,
            "temp_file_path": None,
        }
        result = await goal_tool_root(Params())
        assert not result.is_error
        assert "done" in result.output

    async def test_read_goal_with_file_reference(self, goal_tool_root: Goal) -> None:
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "/path/to/file.py",
            "status": "in_progress",
            "is_file": True,
            "temp_file_path": None,
        }
        result = await goal_tool_root(Params())
        assert not result.is_error
        assert "in_progress" in result.output
        assert "file.py" in result.output


# ── Set with inline code ──────────────────────────────────────────────


class TestGoalSetInline:
    """Setting a goal with inline Python code."""

    async def test_inline_code_creates_temp_file(self, goal_tool_root: Goal) -> None:
        result = await goal_tool_root(Params(code="print('hello world')"))
        assert not result.is_error
        assert "Goal set" in result.output
        assert "Goal code saved to" in result.output
        # Validate goal state in custom_data
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal is not None
        assert goal["code"] == "print('hello world')"
        assert goal["status"] == "pending"
        assert goal["is_file"] is False
        assert goal["temp_file_path"] is not None
        # Verify the temp file was actually created
        temp_path = Path(goal["temp_file_path"])
        assert temp_path.exists()
        assert temp_path.read_text(encoding="utf-8") == "print('hello world')"

    async def test_inline_counter_increments(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="a=1"))
        assert goal_tool_root._script_counter == 1
        await goal_tool_root(Params(code="b=2"))
        assert goal_tool_root._script_counter == 2
        # Two different temp files
        goal1 = goal_tool_root._runtime.session.custom_data.get("goal")
        path1 = goal1["temp_file_path"]
        # Set again
        await goal_tool_root(Params(code="c=3"))
        goal2 = goal_tool_root._runtime.session.custom_data.get("goal")
        path2 = goal2["temp_file_path"]
        assert path1 != path2


# ── Set with .py file path ────────────────────────────────────────────


class TestGoalSetFile:
    """Setting a goal with an existing .py file path."""

    async def test_file_reference(self, goal_tool_root: Goal, sample_py_file: Path) -> None:
        result = await goal_tool_root(Params(code=str(sample_py_file)))
        assert not result.is_error
        assert "Code reference" in result.output
        assert sample_py_file.name in result.output
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal is not None
        assert goal["code"] == str(sample_py_file)
        assert goal["is_file"] is True
        assert goal["temp_file_path"] is None
        assert goal["status"] == "pending"

    async def test_nonexistent_py_file_treated_as_inline(self, goal_tool_root: Goal) -> None:
        """A .py path that does not exist is treated as inline code."""
        result = await goal_tool_root(Params(code="nonexistent.py"))
        assert not result.is_error
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal is not None
        assert goal["is_file"] is False  # not a real file


# ── Clear goal ─────────────────────────────────────────────────────────


class TestGoalClear:
    """Clearing the goal by passing empty code."""

    async def test_clear_goal(self, goal_tool_root: Goal) -> None:
        # First set a goal
        await goal_tool_root(Params(code="print('hello')"))
        assert "goal" in goal_tool_root._runtime.session.custom_data

        # Now clear it
        result = await goal_tool_root(Params(code=""))
        assert not result.is_error
        assert "Goal cleared" in result.output
        assert "goal" not in goal_tool_root._runtime.session.custom_data

    async def test_clear_when_no_goal(self, goal_tool_root: Goal) -> None:
        result = await goal_tool_root(Params(code=""))
        assert not result.is_error
        assert "Goal cleared" in result.output


# ── Mode: overwrite ────────────────────────────────────────────────────


class TestGoalModeOverwrite:
    async def test_overwrite_rejected_when_goal_not_done(self, goal_tool_root: Goal) -> None:
        # Set a goal
        await goal_tool_root(Params(code="x=1"))
        # Try to overwrite with mode='overwrite' while it's pending
        result = await goal_tool_root(Params(code="y=2", mode="overwrite"))
        assert result.is_error
        assert "not done" in result.output.lower()

    async def test_overwrite_accepted_when_goal_done(self, goal_tool_root: Goal) -> None:
        # Set a goal and mark it done
        await goal_tool_root(Params(code="x=1"))
        goal_tool_root._runtime.session.custom_data["goal"]["status"] = "done"

        result = await goal_tool_root(Params(code="y=2", mode="overwrite"))
        assert not result.is_error
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal["code"] == "y=2"
        assert goal["status"] == "pending"

    async def test_overwrite_when_no_previous_goal(self, goal_tool_root: Goal) -> None:
        """Overwrite mode with no existing goal should succeed."""
        result = await goal_tool_root(Params(code="x=1", mode="overwrite"))
        assert not result.is_error


# ── Mode: force_overwrite ──────────────────────────────────────────────


class TestGoalModeForceOverwrite:
    async def test_force_overwrite_always_succeeds(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="x=1"))
        result = await goal_tool_root(Params(code="y=2", mode="force_overwrite"))
        assert not result.is_error
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal["code"] == "y=2"
        assert goal["status"] == "pending"

    async def test_force_overwrite_no_previous(self, goal_tool_root: Goal) -> None:
        result = await goal_tool_root(Params(code="x=1", mode="force_overwrite"))
        assert not result.is_error


# ── Mode: append ───────────────────────────────────────────────────────


class TestGoalModeAppend:
    async def test_append_sets_first_goal(self, goal_tool_root: Goal) -> None:
        result = await goal_tool_root(Params(code="x=1"))
        assert not result.is_error
        assert goal_tool_root._runtime.session.custom_data["goal"]["code"] == "x=1"

    async def test_append_replaces_done_goal(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="x=1"))
        goal_tool_root._runtime.session.custom_data["goal"]["status"] = "done"

        result = await goal_tool_root(Params(code="y=2"))
        assert not result.is_error
        assert goal_tool_root._runtime.session.custom_data["goal"]["code"] == "y=2"
        assert goal_tool_root._runtime.session.custom_data["goal"]["status"] == "pending"

    async def test_append_updates_pending_goal(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="x=1"))
        result = await goal_tool_root(Params(code="y=2"))
        assert not result.is_error
        assert goal_tool_root._runtime.session.custom_data["goal"]["code"] == "y=2"
        assert goal_tool_root._runtime.session.custom_data["goal"]["status"] == "pending"


# ── Root persistence ────────────────────────────────────────────────────


class TestGoalRootPersistence:
    async def test_save_to_custom_data(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="x=1"))
        assert "goal" in goal_tool_root._runtime.session.custom_data
        assert goal_tool_root._runtime.session.custom_data["goal"]["code"] == "x=1"

    async def test_load_from_custom_data(self, goal_tool_root: Goal) -> None:
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "x=1",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        }
        result = await goal_tool_root(Params())
        assert not result.is_error
        assert "pending" in result.output

    async def test_clear_removes_from_custom_data(self, goal_tool_root: Goal) -> None:
        await goal_tool_root(Params(code="x=1"))
        assert "goal" in goal_tool_root._runtime.session.custom_data
        await goal_tool_root(Params(code=""))
        assert "goal" not in goal_tool_root._runtime.session.custom_data


# ── Subagent persistence ───────────────────────────────────────────────


class TestGoalSubagentPersistence:
    async def test_save_to_state_file(self, goal_tool_subagent: Goal) -> None:
        await goal_tool_subagent(Params(code="x=1"))
        state_file = (
            goal_tool_subagent._runtime.subagent_store
            .instance_dir(goal_tool_subagent._runtime.subagent_id)
            / "state.json"
        )
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" in data
        assert data["goal"]["code"] == "x=1"
        assert data["goal"]["status"] == "pending"

    async def test_load_from_state_file(self, goal_tool_subagent: Goal) -> None:
        state_file = (
            goal_tool_subagent._runtime.subagent_store
            .instance_dir(goal_tool_subagent._runtime.subagent_id)
            / "state.json"
        )
        state_file.write_text(json.dumps({
            "goal": {
                "code": "x=1",
                "status": "in_progress",
                "is_file": False,
                "temp_file_path": None,
            }
        }))
        result = await goal_tool_subagent(Params())
        assert not result.is_error
        assert "in_progress" in result.output

    async def test_clear_removes_from_state_file(self, goal_tool_subagent: Goal) -> None:
        await goal_tool_subagent(Params(code="x=1"))
        await goal_tool_subagent(Params(code=""))
        state_file = (
            goal_tool_subagent._runtime.subagent_store
            .instance_dir(goal_tool_subagent._runtime.subagent_id)
            / "state.json"
        )
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data

    async def test_state_file_not_available_returns_error(self) -> None:
        """Subagent with no store/agent_id should return error on save."""
        runtime = MagicMock()
        runtime.role = "subagent"
        runtime.subagent_store = None
        runtime.subagent_id = None
        runtime.session = MagicMock()
        runtime.session.dir = tempfile.mkdtemp()
        tool = Goal(runtime=runtime)
        result = await tool(Params(code="x=1"))
        assert result.is_error
        assert "state file" in result.message.lower()


# ── Error handling ─────────────────────────────────────────────────────


class TestGoalErrorHandling:
    async def test_save_error_returns_tool_error(self) -> None:
        """When custom_data raises, Goal should return a ToolError.

        We simulate a failure by passing a dict subclass whose __setitem__
        and pop raise RuntimeError.
        """
        class _FailingDict(dict):
            def pop(self, key, *args):
                raise RuntimeError("simulated pop failure")
            def __setitem__(self, key, value):
                raise RuntimeError("simulated set failure")

        runtime = MagicMock()
        runtime.role = "root"
        session = MagicMock()
        session.custom_data = _FailingDict()
        runtime.session = session
        tool = Goal(runtime=runtime)

        # Clear path: pop raises
        result = await tool(Params(code=""))
        assert result.is_error
        assert "Failed" in result.message

        # Set path: __setitem__ raises
        result2 = await tool(Params(code="x=1"))
        assert result2.is_error
        assert "Failed" in result2.message

    async def test_corrupted_subagent_state(self, goal_tool_subagent: Goal) -> None:
        """Corrupted state.json should be handled gracefully."""
        state_file = (
            goal_tool_subagent._runtime.subagent_store
            .instance_dir(goal_tool_subagent._runtime.subagent_id)
            / "state.json"
        )
        state_file.write_text("not valid json")
        # Read should return None (no goal) when file is corrupt
        goal = goal_tool_subagent._load_goal()
        assert goal is None
        # Set should still work
        result = await goal_tool_subagent(Params(code="x=1"))
        assert not result.is_error


# ── Subagent state helpers ─────────────────────────────────────────────


class TestGoalSubagentHelpers:
    def test_subagent_state_file_no_store(self) -> None:
        runtime = MagicMock()
        runtime.subagent_store = None
        runtime.subagent_id = "test"
        tool = Goal(runtime=runtime)
        assert tool._subagent_state_file() is None

    def test_subagent_state_file_no_id(self) -> None:
        runtime = MagicMock()
        runtime.subagent_store = MagicMock()
        runtime.subagent_id = None
        tool = Goal(runtime=runtime)
        assert tool._subagent_state_file() is None

    def test_read_subagent_state_missing_file(self, tmp_path: Path) -> None:
        assert Goal._read_subagent_state(tmp_path / "nonexistent.json") == {}

    def test_read_subagent_state_corrupted(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text("{{{")
        assert Goal._read_subagent_state(p) == {}

    def test_read_subagent_state_not_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text('"string"')
        assert Goal._read_subagent_state(p) == {}

    def test_read_subagent_state_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text(json.dumps({"key": "val"}))
        assert Goal._read_subagent_state(p) == {"key": "val"}

    async def test_write_subagent_state_creates_parent(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "state.json"
        Goal._write_subagent_state(p, {"goal": {"code": "x=1"}})
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["goal"]["code"] == "x=1"



# ── RunGoalParams validation ──────────────────────────────────────────


class TestRunGoalParams:
    def test_defaults(self) -> None:
        from kimix.tools.goal import RunGoalParams
        p = RunGoalParams()
        assert p.timeout == 30

    def test_custom_timeout(self) -> None:
        from kimix.tools.goal import RunGoalParams
        p = RunGoalParams(timeout=60)
        assert p.timeout == 60

    def test_timeout_bounds(self) -> None:
        from kimix.tools.goal import RunGoalParams
        with pytest.raises(ValidationError):
            RunGoalParams(timeout=0)
        with pytest.raises(ValidationError):
            RunGoalParams(timeout=901)


# ── RunGoal tool tests ────────────────────────────────────────────────


class TestRunGoal:
    """Tests for the RunGoal tool."""

    async def test_no_goal(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "No goal is set" in result.output

    async def test_goal_already_done(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a goal and mark it done
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "print('ok')",
            "status": "done",
            "is_file": False,
            "temp_file_path": None,
        }
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "already done" in result.output.lower()
        # Goal should be cleared
        assert "goal" not in goal_tool_root._runtime.session.custom_data

    async def test_run_simple_code_success(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a simple goal
        await goal_tool_root(Params(code="print('hello from goal')"))
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "Goal executed successfully" in result.message
        assert "hello from goal" in result.output
        # Goal should be cleared on success
        assert "goal" not in goal_tool_root._runtime.session.custom_data

    async def test_run_code_that_fails(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a goal that will fail
        await goal_tool_root(Params(code="raise RuntimeError('boom')"))
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert result.is_error
        assert "failed" in result.message.lower()
        # Goal should persist (not cleared on failure)
        goal = goal_tool_root._runtime.session.custom_data.get("goal")
        assert goal is not None
        assert goal["status"] == "pending"

    async def test_run_with_exit_code_1(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        await goal_tool_root(Params(code="exit(1)"))
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert result.is_error
        assert "exit code 1" in result.output.lower()

    async def test_run_timeout(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        await goal_tool_root(Params(code="import time; time.sleep(300)"))
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams(timeout=1))
        assert result.is_error
        assert "timed out" in result.message.lower()

    async def test_run_goal_not_runnable(self, goal_tool_root: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a goal with empty code
        goal_tool_root._runtime.session.custom_data["goal"] = {
            "code": "",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        }
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert result.is_error
        assert "not runnable" in result.brief.lower()

    async def test_run_with_file_reference(self, goal_tool_root: Goal, sample_py_file: Path) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a goal with a file reference
        await goal_tool_root(Params(code=str(sample_py_file)))
        tool = RunGoal(runtime=goal_tool_root._runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "Goal executed successfully" in result.message
        # Goal should be cleared
        assert "goal" not in goal_tool_root._runtime.session.custom_data

    async def test_run_subagent_goal(self, goal_tool_subagent: Goal) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        # Set a goal via the subagent tool
        await goal_tool_subagent(Params(code="print('subagent goal ok')"))
        tool = RunGoal(runtime=goal_tool_subagent._runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "Goal executed successfully" in result.message
        # Verify goal cleared from state file
        state_file = (
            goal_tool_subagent._runtime.subagent_store
            .instance_dir(goal_tool_subagent._runtime.subagent_id)
            / "state.json"
        )
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data

    async def test_run_subagent_no_store(self) -> None:
        from kimix.tools.goal import RunGoal, RunGoalParams
        runtime = MagicMock()
        runtime.role = "subagent"
        runtime.subagent_store = None
        runtime.subagent_id = None
        runtime.session = MagicMock()
        runtime.session.dir = tempfile.mkdtemp()
        tool = RunGoal(runtime=runtime)
        result = await tool(RunGoalParams())
        assert not result.is_error
        assert "No goal is set" in result.output


# ── Goal._resolve_goal_executable tests ───────────────────────────────


class TestGoalResolveExecutable:
    def test_empty_code_returns_none(self) -> None:
        assert Goal._resolve_goal_executable({"code": ""}) is None

    def test_no_code_key_returns_none(self) -> None:
        assert Goal._resolve_goal_executable({}) is None

    def test_temp_file_path_exists(self, tmp_path: Path) -> None:
        tp = tmp_path / "goal_0.py"
        tp.write_text("print('hello')")
        result = Goal._resolve_goal_executable({
            "code": "print('hello')",
            "is_file": False,
            "temp_file_path": str(tp),
        })
        assert result == str(tp)

    def test_temp_file_path_missing_falls_back(self) -> None:
        result = Goal._resolve_goal_executable({
            "code": "print('hello')",
            "is_file": False,
            "temp_file_path": "/nonexistent/goal_0.py",
        })
        assert result is not None
        assert result.endswith(".py")
        # Clean up
        import os
        if result and os.path.exists(result):
            os.unlink(result)

    def test_file_reference_exists(self, sample_py_file: Path) -> None:
        result = Goal._resolve_goal_executable({
            "code": str(sample_py_file),
            "is_file": True,
            "temp_file_path": None,
        })
        assert result == str(sample_py_file)

    def test_file_reference_missing_falls_back(self) -> None:
        result = Goal._resolve_goal_executable({
            "code": "/nonexistent/script.py",
            "is_file": True,
            "temp_file_path": None,
        })
        assert result is not None
        assert result.endswith(".py")
        import os
        if result and os.path.exists(result):
            os.unlink(result)


# ── Goal._run_goal_code tests ─────────────────────────────────────────


class TestGoalRunCode:
    async def test_run_simple_code(self) -> None:
        success, output = await Goal._run_goal_code({
            "code": "print('hello')",
            "is_file": False,
            "temp_file_path": None,
        })
        assert success is True
        assert "hello" in output

    async def test_run_failing_code(self) -> None:
        success, output = await Goal._run_goal_code({
            "code": "raise RuntimeError('test error')",
            "is_file": False,
            "temp_file_path": None,
        })
        assert success is False
        assert "test error" in output or "RuntimeError" in output

    async def test_run_exit_code_1(self) -> None:
        success, output = await Goal._run_goal_code({
            "code": "exit(1)",
            "is_file": False,
            "temp_file_path": None,
        })
        assert success is False
        assert "exit code 1" in output

    async def test_run_timeout(self) -> None:
        success, output = await Goal._run_goal_code({
            "code": "import time; time.sleep(300)",
            "is_file": False,
            "temp_file_path": None,
        }, timeout=1)
        assert success is False
        assert "timed out" in output.lower()

    async def test_run_with_python_exe(self) -> None:
        import sys
        success, output = await Goal._run_goal_code({
            "code": "import sys; print(sys.executable)",
            "is_file": False,
            "temp_file_path": None,
        }, python_exe=sys.executable)
        assert success is True
        assert sys.executable in output

    async def test_run_invalid_python_exe(self) -> None:
        success, output = await Goal._run_goal_code({
            "code": "print('hi')",
            "is_file": False,
            "temp_file_path": None,
        }, python_exe="/nonexistent/python")
        assert success is False
        assert "not found" in output.lower()

    async def test_run_multi_line_code(self) -> None:
        code = """
import sys
result = 1 + 1
print(f"1+1={result}")
"""
        success, output = await Goal._run_goal_code({
            "code": code,
            "is_file": False,
            "temp_file_path": None,
        })
        assert success is True
        assert "1+1=2" in output
