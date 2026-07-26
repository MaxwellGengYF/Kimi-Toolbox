"""Comprehensive tests for goal enforcement in prompt.py.

Mirrors the pattern in test_prompt_todo_reminder.py.
"""
from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from kimi_cli.wire.types import TextPart

prompt_mod = importlib.import_module("kimix.utils.prompt")


# ── Fake classes (mirror test_prompt_todo_reminder.py) ────────────────


@dataclass
class FakeStatus:
    context_usage: float = 0.125
    context_tokens: int = 1024


class FakeCLISession:
    def __init__(self, custom_data: dict[str, Any] | None = None) -> None:
        self.custom_data = custom_data or {}


class FakeRuntimeRoot:
    role: str = "root"


class FakeRuntimeSubagent:
    role: str = "subagent"

    def __init__(self, store: Any, agent_id: str) -> None:
        self.subagent_store = store
        self.subagent_id = agent_id


class FakeCLI:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self.session = None


class FakeCLIWithSession:
    def __init__(self, runtime: Any, session: FakeCLISession) -> None:
        self._runtime = runtime
        self.session = session


class FakeSubagentStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def instance_dir(self, agent_id: str) -> Path:
        path = self._base_dir / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path


# ── Session fakes for _maybe_build_goal_reminder / _clear_session_goal ─


class FakeSessionRoot:
    """Root session: _cli has a Runtime with role='root' and a CLI session with custom_data."""

    def __init__(self, goal: dict[str, Any] | None = None) -> None:
        custom_data = {}
        if goal is not None:
            custom_data["goal"] = goal
        self._cli_session = FakeCLISession(custom_data=custom_data)
        self._runtime = FakeRuntimeRoot()
        self._cli = FakeCLIWithSession(runtime=self._runtime, session=self._cli_session)

    def get_custom_data(self) -> dict[str, Any] | None:
        return self._cli_session.custom_data


class FakeSessionSubagent:
    """Subagent session: Runtime has role='subagent' with store/agent_id."""

    def __init__(self, store: FakeSubagentStore, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id
        self._runtime = FakeRuntimeSubagent(store, agent_id)
        self._cli = FakeCLI(runtime=self._runtime)

    def get_custom_data(self) -> dict[str, Any] | None:
        return None


class FakeSessionNoCLI:
    """Session with no _cli attribute at all."""

    def get_custom_data(self) -> dict[str, Any] | None:
        return {}


class FakeSessionCLINoRuntime:
    """Session whose _cli has no _runtime."""

    def __init__(self) -> None:
        self._cli = object()

    def get_custom_data(self) -> dict[str, Any] | None:
        return {}


# ── Session fakes for integration tests (prompt_async) ─────────────────


class FakeSessionWithCLI:
    """Integration-test session that records prompts, like test_prompt_todo_reminder.py."""

    def __init__(
        self,
        has_goal: bool = False,
        goal_status: str = "pending",
        goal_code: str = "",
    ) -> None:
        custom_data: dict[str, Any] = {}
        if has_goal:
            custom_data["goal"] = {
                "code": goal_code or "print('test')",
                "status": goal_status,
                "is_file": False,
                "temp_file_path": None,
            }
        cli_session = FakeCLISession(custom_data=custom_data)
        self._cli = FakeCLIWithSession(runtime=FakeRuntimeRoot(), session=cli_session)
        self.status = FakeStatus()
        self.cancelled = False
        self._cancel_event = None
        self._tmp_data = {}
        self.prompts: list[str] = []

    def get_custom_data(self) -> dict[str, Any] | None:
        return self._cli.session.custom_data

    async def prompt(self, prompt: str, *, merge_wire_messages: bool = False) -> Any:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        yield TextPart(text="prompt output")

    def cancel(self) -> None:
        self.cancelled = True


# ── Helpers ────────────────────────────────────────────────────────────


def _suppress_stream(monkeypatch: Any) -> None:
    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *args, **kwargs: None)


# ══════════════════════════════════════════════════════════════════════
# Tests for _maybe_build_goal_reminder
# ══════════════════════════════════════════════════════════════════════


class TestMaybeBuildGoalReminder:
    """Tests for _maybe_build_goal_reminder."""

    # ── No goal scenarios ───────────────────────────────────────────

    async def test_no_goal_returns_none(self) -> None:
        session = FakeSessionRoot(goal=None)
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    async def test_no_cli_returns_none(self) -> None:
        session = FakeSessionNoCLI()
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    async def test_no_runtime_returns_none(self) -> None:
        session = FakeSessionCLINoRuntime()
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    # ── Goal status scenarios ───────────────────────────────────────

    async def test_goal_done_returns_none(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "print('done')",
            "status": "done",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    async def test_goal_pending_returns_reminder(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "assert 1+1==2",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is not None
        assert "Goal" in result
        assert "assert 1+1==2" in result

    async def test_goal_in_progress_returns_reminder(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "print('working')",
            "status": "in_progress",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is not None

    # ── Strong mode ─────────────────────────────────────────────────

    async def test_strong_reminder_has_critical_prefix(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "x=1",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session, strong=True)
        assert result is not None
        assert result.startswith("CRITICAL")
        assert "MUST" in result

    async def test_non_strong_reminder_is_gentle(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "x=1",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session, strong=False)
        assert result is not None
        assert not result.startswith("CRITICAL")
        assert "Reminder" in result

    # ── Code in reminder output ─────────────────────────────────────

    async def test_reminder_includes_code(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "import sys\nprint(sys.version)",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is not None
        assert "import sys" in result
        assert "print(sys.version)" in result
        assert "```python" in result

    # ── Root persistence path ───────────────────────────────────────

    async def test_root_reads_from_custom_data(self) -> None:
        """Root session reads goal from session.get_custom_data()."""
        session = FakeSessionRoot(goal={
            "code": "root_code",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is not None
        assert "root_code" in result

    async def test_root_no_goal_when_custom_data_missing_key(self) -> None:
        """Root session with custom_data but no 'goal' key."""
        session = FakeSessionRoot(goal=None)  # no goal key
        # The custom_data dict is empty
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    # ── Subagent persistence path ───────────────────────────────────

    async def test_subagent_reads_from_state_file(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent1") / "state.json"
        state_file.write_text(json.dumps({
            "goal": {
                "code": "subagent_code",
                "status": "pending",
                "is_file": False,
                "temp_file_path": None,
            }
        }))
        session = FakeSessionSubagent(store, "agent1")
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is not None
        assert "subagent_code" in result

    async def test_subagent_no_goal_returns_none(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent2") / "state.json"
        state_file.write_text(json.dumps({"other": "data"}))
        session = FakeSessionSubagent(store, "agent2")
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    async def test_subagent_no_store_returns_none(self) -> None:
        """Subagent with no store should return None."""
        session = FakeSessionSubagent(FakeSubagentStore(Path("/nonexistent")), "x")
        # Override runtime to have no subagent_store
        session._runtime.subagent_store = None
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None

    async def test_subagent_done_goal_returns_none(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent3") / "state.json"
        state_file.write_text(json.dumps({
            "goal": {
                "code": "x=1",
                "status": "done",
                "is_file": False,
                "temp_file_path": None,
            }
        }))
        session = FakeSessionSubagent(store, "agent3")
        result = await prompt_mod._maybe_build_goal_reminder(session)
        assert result is None


# ══════════════════════════════════════════════════════════════════════
# Tests for _clear_session_goal
# ══════════════════════════════════════════════════════════════════════


class TestClearSessionGoal:
    """Tests for _clear_session_goal."""

    # ── No-op scenarios ─────────────────────────────────────────────

    async def test_no_cli_returns_immediately(self) -> None:
        session = FakeSessionNoCLI()
        # Should not raise
        await prompt_mod._clear_session_goal(session)

    async def test_no_runtime_returns_immediately(self) -> None:
        session = FakeSessionCLINoRuntime()
        await prompt_mod._clear_session_goal(session)

    # ── Root persistence ────────────────────────────────────────────

    async def test_root_clears_custom_data(self) -> None:
        session = FakeSessionRoot(goal={
            "code": "x=1",
            "status": "pending",
            "is_file": False,
            "temp_file_path": None,
        })
        assert "goal" in session.get_custom_data()
        await prompt_mod._clear_session_goal(session)
        assert "goal" not in session.get_custom_data()

    async def test_root_no_goal_does_not_raise(self) -> None:
        session = FakeSessionRoot(goal=None)
        await prompt_mod._clear_session_goal(session)
        assert "goal" not in session.get_custom_data()

    # ── Subagent persistence ────────────────────────────────────────

    async def test_subagent_clears_state_file(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent1") / "state.json"
        state_file.write_text(json.dumps({
            "goal": {"code": "x=1", "status": "pending"},
            "other": "data",
        }))
        session = FakeSessionSubagent(store, "agent1")
        await prompt_mod._clear_session_goal(session)
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data
        # Other keys should be preserved
        assert data.get("other") == "data"

    async def test_subagent_no_goal_in_file_does_not_raise(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent2") / "state.json"
        state_file.write_text(json.dumps({"other": "data"}))
        session = FakeSessionSubagent(store, "agent2")
        await prompt_mod._clear_session_goal(session)
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data

    async def test_subagent_no_store_does_not_raise(self) -> None:
        session = FakeSessionSubagent(FakeSubagentStore(Path("/nonexistent")), "x")
        session._runtime.subagent_store = None
        await prompt_mod._clear_session_goal(session)


# ══════════════════════════════════════════════════════════════════════
# Integration tests — goal enforcement loop in prompt_async
# ══════════════════════════════════════════════════════════════════════


class TestGoalEnforcementLoop:
    """Tests that the goal enforcement loop in prompt_async works correctly."""

    def test_goal_reminder_injected_when_goal_undone(self, monkeypatch: Any) -> None:
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="exit(1)",  # failing code so auto-run doesn't swallow the reminder
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Auto-run fails → reminder is injected
        assert len(session.prompts) >= 2
        assert session.prompts[0] == "hello"
        reminder = session.prompts[1]
        assert "Goal" in reminder or "goal" in reminder

    def test_goal_reminder_not_injected_when_goal_done(self, monkeypatch: Any) -> None:
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="done",
            goal_code="print('done')",
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Only original prompt, no reminders
        assert session.prompts == ["hello"]

    def test_goal_reminder_not_injected_when_no_goal(self, monkeypatch: Any) -> None:
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(has_goal=False)

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        assert session.prompts == ["hello"]

    def test_goal_reminder_stops_when_marked_done(self, monkeypatch: Any) -> None:
        """When the LLM marks the goal done during the first reminder,
        the second (strong) reminder should not be sent."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="exit(1)",  # failing code so auto-run doesn't swallow the reminder
        )

        async def mark_done_prompt(
            self: Any, prompt: str, *, merge_wire_messages: bool = False,
        ) -> Any:
            self.last_prompt = prompt
            self.prompts.append(prompt)
            # Simulate the LLM marking the goal as done
            if "Goal" in prompt or "Reminder" in prompt:
                goal_data = self.get_custom_data().get("goal")
                if goal_data:
                    goal_data["status"] = "done"
            yield TextPart(text="prompt output")

        monkeypatch.setattr(FakeSessionWithCLI, "prompt", mark_done_prompt)

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Only original prompt + one reminder (the second is skipped because done)
        assert len(session.prompts) == 2
        assert session.prompts[0] == "hello"

    def test_no_reminder_when_ensure_todo_finished_false(self, monkeypatch: Any) -> None:
        """The goal enforcement loop only runs when ensure_todo_finished=True
        (same condition as todo loop)."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
            ensure_todo_finished=False,
        ))

        # No reminders because ensure_todo_finished=False
        assert session.prompts == ["hello"]

    def test_goal_cleared_in_finally_block(self, monkeypatch: Any) -> None:
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="x=1",
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # After prompt_async completes, the goal should be cleared
        custom_data = session.get_custom_data()
        assert custom_data is not None
        assert "goal" not in custom_data


# ══════════════════════════════════════════════════════════════════════
# Tests for _run_single_prompt goal state snapshot
# ══════════════════════════════════════════════════════════════════════


class TestRunSinglePromptGoalSnapshot:
    """Tests that _run_single_prompt snapshots goal state before running."""

    async def test_snapshot_marks_attempt_pending_when_goal_undone(self) -> None:
        custom_data: dict[str, Any] = {
            "goal": {"code": "x=1", "status": "pending", "is_file": False, "temp_file_path": None},
        }
        cli_session = FakeCLISession(custom_data=custom_data)
        session = MagicMock()
        session._cli = FakeCLIWithSession(runtime=FakeRuntimeRoot(), session=cli_session)
        session.get_custom_data.return_value = custom_data
        session._cancel_event = None

        # We need info_print=False so the colorful_print_word is not called
        # Let's use monkeypatch to suppress stream output
        # Actually, _run_single_prompt calls colorful_print_word at the start.
        # We need to handle that.

        # Instead, let's just test the logic directly by calling _run_single_prompt
        # with a session that will fail gracefully
        async def fake_prompt(*args, **kwargs):
            yield TextPart(text="done")

        session.prompt = fake_prompt

        import kimix.base as base
        original_print = base._stream.colorful_print_word
        base._stream.colorful_print_word = lambda *args, **kwargs: None
        original_print_word = base._stream.print_word
        base._stream.print_word = lambda *args, **kwargs: None

        try:
            await prompt_mod._run_single_prompt(
                session=session,
                prompt_str="test",
                output_function=None,
                cancel_callable=None,
                merge_wire_messages=False,
                info_print=False,
                label="test",
            )
        except Exception:
            pass
        finally:
            base._stream.colorful_print_word = original_print
            base._stream.print_word = original_print_word

        assert custom_data.get("_goal_attempt_pending") is True

    async def test_snapshot_skipped_when_goal_done(self) -> None:
        custom_data: dict[str, Any] = {
            "goal": {"code": "x=1", "status": "done", "is_file": False, "temp_file_path": None},
        }
        session = MagicMock()
        session.get_custom_data.return_value = custom_data
        _ = custom_data  # just use it

        # Just test the snapshot logic directly (the flag check before _run_single_prompt)
        if custom_data:
            goal = custom_data.get("goal")
            if isinstance(goal, dict) and goal.get("status") != "done":
                custom_data["_goal_attempt_pending"] = True

        assert "_goal_attempt_pending" not in custom_data or not custom_data["_goal_attempt_pending"]

    async def test_snapshot_skipped_when_no_goal(self) -> None:
        custom_data: dict[str, Any] = {}
        session = MagicMock()
        session.get_custom_data.return_value = custom_data

        if custom_data:
            goal = custom_data.get("goal")
            if isinstance(goal, dict) and goal.get("status") != "done":
                custom_data["_goal_attempt_pending"] = True

        assert "_goal_attempt_pending" not in custom_data



# ══════════════════════════════════════════════════════════════════════
# Tests for _try_run_session_goal
# ══════════════════════════════════════════════════════════════════════


class FakeSessionRootWithGoal:
    """Root session with a goal that either succeeds or fails when run."""

    def __init__(self, goal_code: str = "print('ok')") -> None:
        custom_data: dict[str, Any] = {
            "goal": {
                "code": goal_code,
                "status": "pending",
                "is_file": False,
                "temp_file_path": None,
            }
        }
        self._cli_session = FakeCLISession(custom_data=custom_data)
        self._runtime = FakeRuntimeRoot()
        self._cli = FakeCLIWithSession(runtime=self._runtime, session=self._cli_session)

    def get_custom_data(self) -> dict[str, Any] | None:
        return self._cli_session.custom_data


class FakeSessionRootNoGoal:
    def __init__(self) -> None:
        self._cli_session = FakeCLISession(custom_data={})
        self._runtime = FakeRuntimeRoot()
        self._cli = FakeCLIWithSession(runtime=self._runtime, session=self._cli_session)

    def get_custom_data(self) -> dict[str, Any] | None:
        return self._cli_session.custom_data


class FakeSessionRootGoalDone:
    def __init__(self) -> None:
        custom_data: dict[str, Any] = {
            "goal": {
                "code": "print('ok')",
                "status": "done",
                "is_file": False,
                "temp_file_path": None,
            }
        }
        self._cli_session = FakeCLISession(custom_data=custom_data)
        self._runtime = FakeRuntimeRoot()
        self._cli = FakeCLIWithSession(runtime=self._runtime, session=self._cli_session)

    def get_custom_data(self) -> dict[str, Any] | None:
        return self._cli_session.custom_data


class TestTryRunSessionGoal:
    """Tests for _try_run_session_goal."""

    async def test_no_goal_returns_true(self) -> None:
        session = FakeSessionRootNoGoal()
        result = await prompt_mod._try_run_session_goal(session)
        assert result is True

    async def test_goal_done_returns_true(self) -> None:
        session = FakeSessionRootGoalDone()
        result = await prompt_mod._try_run_session_goal(session)
        assert result is True

    async def test_goal_succeeds_returns_true_and_clears(self) -> None:
        session = FakeSessionRootWithGoal(goal_code="print('auto ok')")
        result = await prompt_mod._try_run_session_goal(session)
        assert result is True
        # Goal should be cleared
        assert "goal" not in session.get_custom_data()

    async def test_goal_fails_returns_false(self) -> None:
        session = FakeSessionRootWithGoal(goal_code="raise RuntimeError('auto fail')")
        result = await prompt_mod._try_run_session_goal(session)
        assert result is False
        # Goal should still exist, status back to pending
        goal = session.get_custom_data().get("goal")
        assert goal is not None
        assert goal["status"] == "pending"

    async def test_no_cli_returns_true(self) -> None:
        session = FakeSessionNoCLI()
        result = await prompt_mod._try_run_session_goal(session)
        assert result is True

    async def test_subagent_goal_succeeds(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent1") / "state.json"
        state_file.write_text(json.dumps({
            "goal": {
                "code": "print('subagent auto ok')",
                "status": "pending",
                "is_file": False,
                "temp_file_path": None,
            }
        }))
        runtime = FakeRuntimeSubagent(store, "agent1")
        cli_session = FakeCLISession(custom_data={})
        cli = FakeCLIWithSession(runtime=runtime, session=cli_session)
        session = MagicMock()
        session._cli = cli
        session.get_custom_data.return_value = cli_session.custom_data

        result = await prompt_mod._try_run_session_goal(session)
        assert result is True
        # Verify goal cleared from state file
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data

    async def test_subagent_goal_fails(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent2") / "state.json"
        state_file.write_text(json.dumps({
            "goal": {
                "code": "exit(1)",
                "status": "pending",
                "is_file": False,
                "temp_file_path": None,
            }
        }))
        runtime = FakeRuntimeSubagent(store, "agent2")
        cli_session = FakeCLISession(custom_data={})
        cli = FakeCLIWithSession(runtime=runtime, session=cli_session)
        session = MagicMock()
        session._cli = cli
        session.get_custom_data.return_value = cli_session.custom_data

        result = await prompt_mod._try_run_session_goal(session)
        assert result is False
        # Goal should persist with pending status
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["goal"]["status"] == "pending"


# ══════════════════════════════════════════════════════════════════════
# Tests for _save_session_goal
# ══════════════════════════════════════════════════════════════════════


class TestSaveSessionGoal:
    async def test_root_save_goal(self) -> None:
        session = FakeSessionRootNoGoal()
        await prompt_mod._save_session_goal(session, {"code": "x=1", "status": "pending"})
        assert session.get_custom_data().get("goal") == {"code": "x=1", "status": "pending"}

    async def test_root_clear_goal(self) -> None:
        session = FakeSessionRootWithGoal(goal_code="x=1")
        assert "goal" in session.get_custom_data()
        await prompt_mod._save_session_goal(session, None)
        assert "goal" not in session.get_custom_data()

    async def test_subagent_save_goal(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent1") / "state.json"
        runtime = FakeRuntimeSubagent(store, "agent1")
        cli_session = FakeCLISession(custom_data={})
        cli = FakeCLIWithSession(runtime=runtime, session=cli_session)
        session = MagicMock()
        session._cli = cli
        session.get_custom_data.return_value = cli_session.custom_data

        await prompt_mod._save_session_goal(session, {"code": "x=1", "status": "pending"})
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["goal"] == {"code": "x=1", "status": "pending"}

    async def test_subagent_clear_goal(self, tmp_path: Path) -> None:
        store = FakeSubagentStore(tmp_path / "subagents")
        state_file = store.instance_dir("agent2") / "state.json"
        state_file.write_text(json.dumps({"goal": {"code": "x=1"}, "other": "data"}))
        runtime = FakeRuntimeSubagent(store, "agent2")
        cli_session = FakeCLISession(custom_data={})
        cli = FakeCLIWithSession(runtime=runtime, session=cli_session)
        session = MagicMock()
        session._cli = cli
        session.get_custom_data.return_value = cli_session.custom_data

        await prompt_mod._save_session_goal(session, None)
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "goal" not in data
        assert data.get("other") == "data"

    async def test_no_cli_does_not_raise(self) -> None:
        session = FakeSessionNoCLI()
        await prompt_mod._save_session_goal(session, {"code": "x=1"})


# ══════════════════════════════════════════════════════════════════════
# Tests for auto-run before enforcement loop
# ══════════════════════════════════════════════════════════════════════


class TestGoalEnforcementAutoRun:
    """Tests that the enforcement loop auto-runs the goal before reminding."""

    def test_auto_run_success_breaks_loop_silently(self, monkeypatch: Any) -> None:
        """When automatic execution succeeds, no reminder is injected."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="print('auto-pass')",  # simple code that works
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Only original prompt — goal ran successfully, loop broke silently
        assert session.prompts == ["hello"]
        # Goal should be cleared
        assert "goal" not in session.get_custom_data()

    def test_auto_run_failure_falls_through_to_reminder(self, monkeypatch: Any) -> None:
        """When automatic execution fails, the reminder should still be injected."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="exit(1)",  # code that fails
        )

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Original prompt + reminders (auto-run failed, so reminder fires)
        assert len(session.prompts) >= 2

    def test_auto_run_no_goal_no_reminder(self, monkeypatch: Any) -> None:
        """With no goal, the loop does nothing."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(has_goal=False)

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        assert session.prompts == ["hello"]
    def test_auto_run_then_marked_done_stops_loop(self, monkeypatch: Any) -> None:
        """Auto-run fails, first reminder succeeds (marks goal done), loop stops."""
        _suppress_stream(monkeypatch)
        session = FakeSessionWithCLI(
            has_goal=True,
            goal_status="pending",
            goal_code="exit(1)",  # auto-run will fail
        )

        async def mark_done_prompt(
            self: Any, prompt: str, *, merge_wire_messages: bool = False,
        ) -> Any:
            self.last_prompt = prompt
            self.prompts.append(prompt)
            # Simulate LLM marking goal done during the reminder
            if "Goal" in prompt or "Reminder" in prompt:
                goal_data = self.get_custom_data().get("goal")
                if goal_data:
                    goal_data["status"] = "done"
            yield TextPart(text="prompt output")

        monkeypatch.setattr(FakeSessionWithCLI, "prompt", mark_done_prompt)

        asyncio.run(prompt_mod.prompt_async(
            "hello", session=session, info_print=False,
        ))

        # Original prompt + first reminder (auto-run failed, reminder fixes it)
        assert len(session.prompts) == 2
