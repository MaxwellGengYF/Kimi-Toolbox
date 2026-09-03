"""Regression tests for the plan-session retry nudge provenance fix.

Background: ``prompt_plan_async`` (and both server session managers) verified
the plan file on disk with ``plan_file.exists() and size > 0``. On failure the
loop re-prompted the planner with an inline message ("The plan file was not
generated. Please generate the plan and save it using the WritePlan tool.")
that arrived as a user-role turn but was machine-generated. Planners read it
as the user confirming/reviewing their work and answered a question nobody
asked (hallucinated confirmation turns).

The fix routes all three call sites through ``build_plan_retry_reminder``,
which labels the nudge ``[system check]`` and states explicitly that it is not
a user message. These tests cover:

- builder wording (label, provenance, instruction, requirement embedding)
- prompt_plan_async retry: 2nd-attempt prompt is the labeled nudge
- prompt_plan_async exhausted retries: every retry is labeled, no raise
- drift guard: the old ambiguous literal is gone from all three sites, each
  site references the shared builder
"""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import kimix.base as base

# NB: ``import kimix.utils.prompt as prompt_mod`` would bind the *function*
# ``prompt`` re-exported by kimix.utils.__init__ (it shadows the submodule
# attribute); importlib reliably returns the module object.
prompt_mod = importlib.import_module("kimix.utils.prompt")
from kimix.utils.prompt import (
    PLAN_RETRY_CHECK_PREFIX,
    build_plan_retry_reminder,
    prompt_plan_async,
)

REQUIREMENT = "  Add --max-distance and --ao-gamma to the AO baker.  "

# The full old second sentence; must not be re-inlined at any call site.
OLD_NUDGE_SENTENCE = "Please generate the plan and save it using the WritePlan tool."


class _FakePlannerSession:
    """Minimal stand-in for the planner Session used by prompt_plan_async."""

    def __init__(self, prompts: list[str], write_on_attempt: int | None, plan_file: Path):
        self.prompts = prompts
        self._write_on_attempt = write_on_attempt
        self._plan_file = plan_file
        self._calls = 0
        self._cli = None  # disables the read-only runtime lock branch
        self._cancel_event = None

    def get_custom_data(self) -> dict:
        return {}

    def cancel(self) -> None:
        pass

    async def prompt(self, text: str, **kwargs):
        self._calls += 1
        self.prompts.append(text)
        if self._write_on_attempt is not None and self._calls == self._write_on_attempt:
            self._plan_file.write_text("# Plan\n\nstep one\n", encoding="utf-8")
        return
        yield  # pragma: no cover — makes this an async generator, yields nothing


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """Patch every collaborator of prompt_plan_async; return bookkeeping handles."""
    stream = MagicMock()
    monkeypatch.setattr(base, "_stream", stream)
    monkeypatch.setattr(base, "get_default_sub_provider", lambda role: None)
    monkeypatch.setattr(base, "_default_provider", {"type": "test"})

    from kimix.utils import _globals as session_globals

    monkeypatch.setattr(session_globals, "_default_session", None, raising=False)

    # _open_plan_file side effects must never touch the real system.
    import os as _os

    monkeypatch.setattr(_os, "startfile", lambda *a: None, raising=False)
    monkeypatch.setattr(prompt_mod.subprocess, "run", lambda *a, **k: None)

    close_mock = AsyncMock()
    monkeypatch.setattr(prompt_mod, "close_session_async", close_mock)

    return SimpleHandles(stream=stream, close_mock=close_mock)


class SimpleHandles:
    def __init__(self, stream, close_mock):
        self.stream = stream
        self.close_mock = close_mock

    def printed_texts(self) -> list[str]:
        texts = []
        for call in self.stream.colorful_print_word.call_args_list:
            if call.args:
                texts.append(str(call.args[0]))
        return texts


async def _drive(monkeypatch, patched, prompts: list[str], write_on_attempt: int | None, plan_file: Path, inputs: list[str]):
    fake = _FakePlannerSession(prompts, write_on_attempt, plan_file)
    monkeypatch.setattr(prompt_mod, "_create_session_async", AsyncMock(return_value=fake))
    it = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    await prompt_plan_async(REQUIREMENT, plan_file)
    return fake


class TestBuildPlanRetryReminder:
    def test_starts_with_system_check_label(self):
        text = build_plan_retry_reminder("do the thing")
        assert text.startswith(PLAN_RETRY_CHECK_PREFIX)

    def test_states_it_is_not_a_user_message(self):
        text = build_plan_retry_reminder("do the thing")
        assert "NOT a message from the user" in text
        assert "No human has read, confirmed, or reviewed anything yet" in text

    def test_contains_actionable_instruction_and_requirement(self):
        text = build_plan_retry_reminder("  spaced requirement \n")
        assert "Call WritePlan with the complete plan now" in text
        assert text.endswith("Requirement:\nspaced requirement")

    def test_old_ambiguous_phrasing_gone(self):
        # The builder must not reintroduce the confirmation-question tone.
        text = build_plan_retry_reminder("x").lower()
        assert "please generate the plan" not in text
        assert "do not ask the user questions" in text


class TestPromptPlanAsyncRetry:
    async def test_second_attempt_receives_labeled_nudge(self, monkeypatch, patched, tmp_path):
        """Attempt 1 never writes the file; attempt 2 writes it.

        The second planner prompt must be the [system check]-labeled nudge,
        and the first must remain the original human-authored reminder.
        """
        prompts: list[str] = []
        plan_file = tmp_path / "plan.md"
        await _drive(
            monkeypatch, patched, prompts,
            write_on_attempt=2, plan_file=plan_file,
            inputs=["n", "/quit"],  # review loop: not yet, then give up
        )

        assert len(prompts) == 2, prompts
        # First attempt: original reminder, no system-check label.
        assert not prompts[0].startswith(PLAN_RETRY_CHECK_PREFIX)
        assert "Requirement:" in prompts[0]
        # Second attempt: labeled automated nudge carrying the requirement.
        assert prompts[1].startswith(PLAN_RETRY_CHECK_PREFIX)
        assert "NOT a message from the user" in prompts[1]
        assert "Add --max-distance and --ao-gamma to the AO baker." in prompts[1]
        assert OLD_NUDGE_SENTENCE not in prompts[1]

        # The run must not have swallowed any exception on the way.
        assert not any("prompt_plan failed" in t for t in patched.printed_texts())
        # Review loop quit -> plan not executed, planner session closed.
        patched.close_mock.assert_awaited_once()

    async def test_all_retries_labeled_when_plan_never_written(self, monkeypatch, patched, tmp_path):
        """File never appears: 3 attempts, both retries labeled, graceful return."""
        prompts: list[str] = []
        plan_file = tmp_path / "plan.md"
        await _drive(
            monkeypatch, patched, prompts,
            write_on_attempt=None, plan_file=plan_file,
            inputs=[],  # review loop never reached
        )

        assert len(prompts) == 3, prompts
        assert not prompts[0].startswith(PLAN_RETRY_CHECK_PREFIX)
        assert prompts[1].startswith(PLAN_RETRY_CHECK_PREFIX)
        assert prompts[2].startswith(PLAN_RETRY_CHECK_PREFIX)
        assert any("Plan generation failed" in t for t in patched.printed_texts())
        assert not any("prompt_plan failed" in t for t in patched.printed_texts())
        patched.close_mock.assert_awaited_once()


class TestCallSiteDriftGuard:
    """The three nudge call sites must go through the shared builder."""

    SITES = [
        "src/kimix/utils/prompt.py",
        "src/kimix/server/session_manager.py",
        "src/kimix/server/dummy_session_manager.py",
    ]

    def test_old_literal_gone_and_builder_used(self):
        root = Path(__file__).resolve().parents[1]
        for rel in self.SITES:
            src = (root / rel).read_text(encoding="utf-8")
            assert OLD_NUDGE_SENTENCE not in src, f"{rel} re-inlines the ambiguous nudge"
            assert "build_plan_retry_reminder" in src, f"{rel} bypasses the shared builder"
