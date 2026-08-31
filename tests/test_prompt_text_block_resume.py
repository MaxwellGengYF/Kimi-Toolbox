from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from kosong.message import Message, TextPart, ThinkPart, ToolCall

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float = 0.0
    context_tokens: int = 0


def _tool_call(name: str = "grep") -> ToolCall:
    return ToolCall(id="call-1", function=ToolCall.FunctionBody(name=name, arguments="{}"))


class FakeSession:
    """A session with an inspectable ``_cli.soul.context.history``.

    ``on_resume`` is called before yielding output for the text-block resume
    prompt, letting tests simulate the model finishing with a text block.
    """

    def __init__(
        self,
        history: list[Message],
        on_resume: Any = None,
        current_prompt: str | None = None,
        work_dir: str | None = None,
    ) -> None:
        self.status = FakeStatus()
        self._cancel_event = None
        self.prompts: list[str] = []
        self.history = history
        self._on_resume = on_resume
        self._cli = type("FakeCLI", (), {})()
        self._cli.soul = SimpleNamespace(context=SimpleNamespace(history=history))
        self._cli._runtime = SimpleNamespace(current_prompt=current_prompt)
        self._cli.session = SimpleNamespace(work_dir=work_dir)

    async def prompt(self, prompt_str: str, *, merge_wire_messages: bool = False) -> Any:
        self.prompts.append(prompt_str)
        if "did not end with a plain text block" in prompt_str and self._on_resume is not None:
            self._on_resume()
        yield TextPart(text="ok")


async def _noop_print_agent_json(*args: Any, **kwargs: Any) -> None:
    """Async stand-in for ``kimix.ui.stream.print_agent_json`` (awaited)."""
    return None


def _suppress_output(monkeypatch: Any) -> None:
    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", lambda *a, **k: None)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *a, **k: None)
    monkeypatch.setattr(prompt_mod, "print_agent_json", _noop_print_agent_json)
    monkeypatch.setattr(prompt_mod, "print_agent_json_flush_text", lambda *a, **k: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *a, **k: None)


def _history(*assistant_parts: list[Any]) -> list[Message]:
    history: list[Message] = [Message(role="user", content="hello")]
    for parts in assistant_parts:
        history.append(Message(role="assistant", content=parts))
    return history


def _tool_call_history() -> list[Message]:
    """A history whose last assistant message is a tool call without text."""
    return [
        Message(role="user", content="hello"),
        Message(role="assistant", content=[], tool_calls=[_tool_call()]),
    ]


# ── _trailing_content_kind unit tests ────────────────────────────────────


def test_trailing_kind_tool_wins_even_with_text_preamble() -> None:
    # A message that issues tool calls is a tool turn, even if it also has
    # explanatory text. After the tool result, the session should resume for a
    # final text block instead of ending silently.
    msg = Message(role="assistant", content=[TextPart(text="done")], tool_calls=[_tool_call()])
    session = FakeSession(history=[msg])
    assert prompt_mod._trailing_content_kind(session) == "tool"


def test_trailing_kind_tool_when_no_text() -> None:
    msg = Message(role="assistant", content=[], tool_calls=[_tool_call()])
    session = FakeSession(history=[msg])
    assert prompt_mod._trailing_content_kind(session) == "tool"


def test_trailing_kind_think_when_only_reasoning() -> None:
    msg = Message(role="assistant", content=[ThinkPart(think="reasoning...")])
    session = FakeSession(history=[msg])
    assert prompt_mod._trailing_content_kind(session) == "think"


def test_trailing_kind_last_non_empty_assistant_wins() -> None:
    history = [
        Message(role="assistant", content=[], tool_calls=[_tool_call()]),
        Message(role="assistant", content=[]),  # empty — skipped
        Message(role="assistant", content=[TextPart(text="final")]),
    ]
    session = FakeSession(history=history)
    assert prompt_mod._trailing_content_kind(session) == "text"


def test_trailing_kind_none_without_cli_or_history() -> None:
    session = FakeStatus()  # no _cli at all
    assert prompt_mod._trailing_content_kind(session) is None


# ── _resume_for_text_block behavior ──────────────────────────────────────


def _tool_call_with_text_preamble_history() -> list[Message]:
    """A history whose last assistant message mixes a text preamble with a tool call."""
    return [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content=[TextPart(text="I will verify this now.")],
            tool_calls=[_tool_call()],
        ),
    ]


@pytest.mark.asyncio
async def test_resume_when_trailing_tool_call(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=_tool_call_history())

    def on_resume() -> None:
        session.history.append(Message(role="assistant", content=[TextPart(text="final answer")]))

    session._on_resume = on_resume

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == 1
    assert "did not end with a plain text block" in session.prompts[0]
    # The resume prompt does not copy the original request.
    assert "Original request" not in session.prompts[0]


@pytest.mark.asyncio
async def test_resume_when_trailing_tool_call_has_text_preamble(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=_tool_call_with_text_preamble_history())

    def on_resume() -> None:
        session.history.append(Message(role="assistant", content=[TextPart(text="final answer")]))

    session._on_resume = on_resume

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == 1
    assert "did not end with a plain text block" in session.prompts[0]


@pytest.mark.asyncio
async def test_resume_includes_workdir_but_not_original_request(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(
        history=_tool_call_history(),
        current_prompt="Implement the feature",
        work_dir="/tmp/proj",
    )

    def on_resume() -> None:
        session.history.append(Message(role="assistant", content=[TextPart(text="done")]))

    session._on_resume = on_resume

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == 1
    # The original request is intentionally NOT copied into the resume prompt.
    assert "Original request" not in session.prompts[0]
    assert "Implement the feature" not in session.prompts[0]
    assert "/tmp/proj" in session.prompts[0]


@pytest.mark.asyncio
async def test_no_resume_when_trailing_text(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=_history([TextPart(text="all done")]))

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert session.prompts == []


@pytest.mark.asyncio
async def test_no_resume_when_no_inspectable_history(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=[])
    session._cli.soul.context.history = None

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert session.prompts == []


@pytest.mark.asyncio
async def test_resume_when_trailing_think_only(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=_history([ThinkPart(think="just thinking")]))

    def on_resume() -> None:
        session.history.append(Message(role="assistant", content=[TextPart(text="done")]))

    session._on_resume = on_resume

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == 1
    assert "did not end with a plain text block" in session.prompts[0]


@pytest.mark.asyncio
async def test_stops_after_max_rounds_when_model_keeps_tool_calling(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    # The fake session never fixes the trailing tool call.
    session = FakeSession(history=_tool_call_history())

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == prompt_mod._MAX_TEXT_BLOCK_RESUME_ROUNDS == 2


@pytest.mark.asyncio
async def test_stops_resuming_once_text_appears(monkeypatch: Any) -> None:
    _suppress_output(monkeypatch)
    session = FakeSession(history=_tool_call_history())

    def on_resume() -> None:
        session.history.append(Message(role="assistant", content=[TextPart(text="final")]))

    session._on_resume = on_resume

    await prompt_mod._resume_for_text_block(session, None, None, False, False, False, None)

    assert len(session.prompts) == 1


def test_max_resume_rounds_constant() -> None:
    assert prompt_mod._MAX_TEXT_BLOCK_RESUME_ROUNDS == 2


# ── Dry-run test: backend-interrupt ``continue`` continuation ─────────────
# Scenario (mirrors the backend interruption flow, e.g. a sub-agent resumed
# with ``SubAgentParams(prompt="continue", ...)`` after the parent injected a
# response): the LLM answered with a tool call + reasoning and then the turn
# ended *before* a final text block. The text-block gate must resume the exact
# same session with the continuation prompt and only stop after the last
# assistant block is a plain text message.


class DryRunSession:
    """Deterministic dry-run session for ``prompt_async``.

    ``prompt`` hands each prompt to a caller-provided scripted handler. The
    handler may mutate ``self.history`` (simulating what a real LLM run would
    append) before returning the wire chunks the stream printer would receive.
    """

    def __init__(
        self,
        history: list[Message],
        handler: Any,
        *,
        current_prompt: str = "Implement the feature",
        work_dir: str = "C:\\proj\\feature",
    ) -> None:
        self.status = FakeStatus()
        self._cancel_event = None
        self.history = history
        self.prompts: list[str] = []
        self._handler = handler
        self._cli = SimpleNamespace(
            soul=SimpleNamespace(context=SimpleNamespace(history=history)),
            _runtime=SimpleNamespace(current_prompt=current_prompt),
            session=SimpleNamespace(work_dir=work_dir),
        )

    async def prompt(self, prompt_str: str, *, merge_wire_messages: bool = False) -> Any:
        self.prompts.append(prompt_str)
        for part in self._handler(prompt_str, self):
            yield part


@pytest.mark.asyncio
async def test_dry_run_continue_backend_interrupt_resumes_until_text(
    monkeypatch: Any,
) -> None:
    """Backend-interrupt ``continue``: the resumed turn ends on a tool call
    (no final text block), so ``prompt_async`` must resume the same session
    with a continuation prompt and stop as soon as a text block is produced."""
    _suppress_output(monkeypatch)

    # State after the backend interruption: the last assistant message is a
    # tool call with reasoning and no final text answer.
    history = [
        Message(role="user", content="continue"),
        Message(
            role="assistant",
            content=[ThinkPart(think="need to search")],
            tool_calls=[_tool_call("grep")],
        ),
    ]
    resume_sent: list[str] = []

    def handler(prompt: str, session: DryRunSession) -> list[Any]:
        if prompt == "continue":
            # The resumed run again ends on a tool call — still no text block.
            session.history.append(
                Message(
                    role="assistant",
                    content=[ThinkPart(think="searching")],
                    tool_calls=[_tool_call("write")],
                )
            )
            return [ThinkPart(think="searching"), _tool_call("write")]
        if "did not end with a plain text block" in prompt:
            resume_sent.append(prompt)
            # The continuation finally ends with a plain text block.
            session.history.append(
                Message(
                    role="assistant",
                    content=[TextPart(text="All done. Feature implemented.")],
                )
            )
            return [TextPart(text="All done. Feature implemented.")]
        return [TextPart(text="ok")]

    session = DryRunSession(
        history=history,
        handler=handler,
        current_prompt="continue",
        work_dir="C:\\proj\\feature",
    )

    await prompt_mod.prompt_async(
        "continue",
        session=session,
        info_print=False,
        merge_wire_messages=True,
        format_output=True,
    )

    # ── Dry-run transcript ──
    # 1. the original "continue" prompt (backend-interrupted resume)
    # 2. exactly one text-block resume (todo reminder: no todos -> skipped)
    assert len(session.prompts) == 2
    assert session.prompts[0] == "continue"
    assert "did not end with a plain text block" in session.prompts[1]
    assert "Original request" not in session.prompts[1]
    assert r"C:\proj\feature" in session.prompts[1]
    assert len(resume_sent) == 1
    # The session now ends on a text block — the gate is satisfied.
    assert prompt_mod._trailing_content_kind(session) == "text"
    assert isinstance(session.history[-1].content[-1], TextPart)


@pytest.mark.asyncio
async def test_dry_run_continue_resumes_until_text_across_rounds(
    monkeypatch: Any,
) -> None:
    """The gate keeps resuming while the backend keeps ending on tool calls
    and only stops once a text block appears (bounded by the max rounds)."""
    _suppress_output(monkeypatch)

    history = [
        Message(role="user", content="continue"),
        Message(role="assistant", content=[], tool_calls=[_tool_call("grep")]),
    ]
    resume_sent: list[str] = []

    def handler(prompt: str, session: DryRunSession) -> list[Any]:
        if "did not end with a plain text block" in prompt:
            resume_sent.append(prompt)
            if len(resume_sent) == 1:
                # Still working: another tool call, no text.
                session.history.append(
                    Message(role="assistant", content=[], tool_calls=[_tool_call("write")])
                )
                return [_tool_call("write")]
            # Final round: produce a text block.
            session.history.append(
                Message(role="assistant", content=[TextPart(text="Fully done.")])
            )
            return [TextPart(text="Fully done.")]
        # prompt == "continue" (original resumed run ends on a tool call)
        session.history.append(
            Message(role="assistant", content=[], tool_calls=[_tool_call("grep")])
        )
        return [_tool_call("grep")]

    session = DryRunSession(history=history, handler=handler, current_prompt="continue")

    await prompt_mod.prompt_async(
        "continue",
        session=session,
        info_print=False,
        merge_wire_messages=True,
        format_output=True,
    )

    assert session.prompts[0] == "continue"
    assert len(session.prompts) == 1 + 2  # original + two resume rounds
    assert len(resume_sent) == 2
    assert prompt_mod._trailing_content_kind(session) == "text"