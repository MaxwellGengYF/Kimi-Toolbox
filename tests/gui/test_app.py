from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton

from kimi_agent_sdk import (
    ApprovalRequest,
    BriefDisplayBlock,
    StatusUpdate,
    TextPart,
    TodoDisplayBlock,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
    TurnEnd,
)
from kimix_gui.app import KimixGuiApp
from kimix_gui.backend import SessionOptions
from kimix_gui.history import SessionHistory, Timeline
from kimix_gui.llm import KimixGuiConfigStore, resolved_provider_file
from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.composer import Composer
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.paint import layout_record
from kimix_gui.qt.request_dialogs import ApprovalDialog
from kimix_gui.transcript_data import (
    ActivityEntry,
    ActivityField,
    HistoryEntry,
    TextBlock,
    TextEntry,
    ToolActivity,
    ToolCallContent,
    ToolIdentity,
    entry_body_text,
    entry_kind,
    literal,
)

from .qtutil import find, launch_app, wait_chat_ready, wait_home, wait_idle, widget_text


def _fake_timeline(turn_count: int) -> Timeline:
    return Timeline.from_turn_entries(
        [
            [_text_entry("user", f"q{index}"), _text_entry("assistant", f"a{index}")]
            for index in range(turn_count)
        ]
    )


def _text_entry(kind: str, text: str) -> TextEntry:
    return TextEntry(
        key=f"test:{kind}:{text}",
        kind=kind,  # type: ignore[arg-type]
        blocks=(TextBlock(literal(text)),),
    )


def _row_kind(row: object) -> str:
    return entry_kind(row.entry)  # type: ignore[attr-defined]


def _row_text(row: object) -> str:
    return entry_body_text(row.entry)  # type: ignore[attr-defined]


def _config_store(tmp_path: Path) -> KimixGuiConfigStore:
    config_file = tmp_path / "provider.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 131_072,
                "url": "https://example.test/v1",
                "type": "openai_legacy",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    store = KimixGuiConfigStore(
        tmp_path / "kimix-gui.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )
    store.set_default(tmp_path, resolved_provider_file(config_file).selection)
    return store


class FakeSession:
    def __init__(self, messages: list[object], *, hang_prompt: bool = False) -> None:
        self.id = "fake-session"
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self._messages = messages
        self._hang_prompt = hang_prompt
        self._hang = asyncio.Event()
        self.prompt_started = asyncio.Event()
        self.prompts: list[str] = []
        self.cancelled = False
        self.closed = False

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        self.prompts.append(user_input)
        assert merge_wire_messages is False
        self.prompt_started.set()
        for message in self._messages:
            yield message
        if self._hang_prompt:
            await self._hang.wait()

    def cancel(self) -> None:
        self.cancelled = True
        self._hang.set()

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self._hang.set()
        self.closed = True


def _submit(qtbot, chat: ChatView, text: str) -> None:
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText(text)
    prompt.submitted.emit(text)


def test_keyboard_submit_streams_into_transcript(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="hello "), TextPart(text="world"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "hi")
    qtbot.waitUntil(lambda: session.prompts == ["hi"], timeout=10_000)
    wait_idle(qtbot, app)

    records = [(_row_kind(record), _row_text(record)) for record in chat.transcript.records]
    assert session.prompts == ["hi"]
    assert ("user", "hi") in records
    assert ("assistant", "hello world") in records


def test_keyboard_submit_sends_multiline_prompt(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText("hi\nthere")
    prompt.submitted.emit("hi\nthere")
    qtbot.waitUntil(lambda: session.prompts == ["hi\nthere"], timeout=10_000)
    wait_idle(qtbot, app)

    records = [(_row_kind(record), _row_text(record)) for record in chat.transcript.records]
    assert session.prompts == ["hi\nthere"]
    assert ("user", "hi\nthere") in records
    assert prompt.text == ""


def test_chat_prompt_stays_within_screen(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app, size=(800, 600))
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    assert prompt.x() >= 0
    assert prompt.geometry().right() <= chat.width()
    prompt.setFocus()
    for _ in range(5):
        qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert prompt.x() >= 0
    assert prompt.geometry().right() <= chat.width()


def test_send_button_submits_prompt(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    send = find(chat, "send-prompt", QPushButton)
    cancel = find(chat, "cancel-prompt", QPushButton)
    assert send.text() == "Send"
    assert cancel.text() == "Cancel"
    assert send.isVisible() is True
    assert send.isEnabled() is False
    assert cancel.isVisible() is False
    assert send.x() >= chat.prompt.geometry().right()
    assert chat.prompt.height() == Composer.MIN_HEIGHT
    assert send.height() == Composer.ACTION_HEIGHT
    assert "Ctrl+Enter" in chat.prompt.placeholderText()

    chat.prompt.setPlainText("hello from send")
    assert send.isEnabled() is True
    send.click()
    qtbot.waitUntil(lambda: session.prompts == ["hello from send"], timeout=10_000)
    wait_idle(qtbot, app)
    assert chat.prompt.text == ""
    assert send.isEnabled() is False


def test_expand_prompt_opens_pad_and_sends(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    find(chat.prompt, "expand-prompt", QPushButton).click()
    pad = find(chat, "composer-pad", QDialog)
    qtbot.waitUntil(pad.isVisible, timeout=5_000)
    editor = find(pad, "prompt-pad", QPlainTextEdit)
    long_text = "paste\n" * 40 + "done"
    editor.setPlainText(long_text)
    find(pad, "send-pad", QPushButton).click()
    qtbot.waitUntil(lambda: session.prompts == [long_text], timeout=10_000)
    wait_idle(qtbot, app)
    assert chat.prompt.text == ""
    assert pad.isVisible() is False


def test_expand_prompt_keeps_draft_on_close(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    find(chat.prompt, "expand-prompt", QPushButton).click()
    pad = find(chat, "composer-pad", QDialog)
    qtbot.waitUntil(pad.isVisible, timeout=5_000)
    find(pad, "prompt-pad", QPlainTextEdit).setPlainText("keep this draft")
    find(pad, "close-composer-pad", QPushButton).click()
    qtbot.waitUntil(lambda: not pad.isVisible(), timeout=5_000)
    assert chat.prompt.text == "keep this draft"
    assert session.prompts == []


def test_the_screen_reports_whatever_dialog_is_on_top(qtbot, tmp_path: Path) -> None:
    """``screen`` answers "what is the user looking at", including the composer pad.

    The window used to keep its own record of one modal, updated by every site that
    opened a dialog. The pad was not one of those sites, so it was invisible here.
    Qt tracks the modal stack, so nothing has to be told -- which is also what stops a
    site from forgetting to clear it and leaving ``screen`` naming a closed dialog.
    """

    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    assert app.screen is chat

    find(chat.prompt, "expand-prompt", QPushButton).click()
    pad = find(chat, "composer-pad", QDialog)
    qtbot.waitUntil(pad.isVisible, timeout=5_000)
    assert app.screen is pad

    find(pad, "close-composer-pad", QPushButton).click()
    qtbot.waitUntil(lambda: not pad.isVisible(), timeout=5_000)
    assert app.screen is chat


def test_cancel_button_stops_generation(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    cancel = find(chat, "cancel-prompt", QPushButton)
    qtbot.waitUntil(lambda: cancel.isVisible() and cancel.isEnabled(), timeout=10_000)
    assert find(chat, "send-prompt", QPushButton).isVisible() is False
    cancel.click()
    wait_idle(qtbot, app)
    assert session.cancelled is True
    assert chat.busy is False
    assert chat.prompt_enabled is True


def test_chat_live_stream_keeps_timeline_and_appends_at_tail(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        turns = [
            [_text_entry("user", f"q{index}"), _text_entry("assistant", f"a{index}")]
            for index in range(30)
        ]
        turns[0].insert(
            1,
            ActivityEntry(
                key="test:activity:read",
                activity=ToolActivity(
                    call_id="read-1",
                    identity=ToolIdentity("read", "read"),
                    call=ToolCallContent(
                        summary_parts=(literal("a.py"),),
                        fields=(
                            ActivityField("path", literal("a.py"), role="primary", hint="path"),
                        ),
                    ),
                ),
            ),
        )
        return Timeline.from_turn_entries(turns)

    monkeypatch.setattr("kimix_gui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([TextPart(text="hel"), TextPart(text="lo"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q0" not in texts
    assert "q29" in texts
    chat.transcript.jump_to_turn(0)
    assert chat.transcript.pinned_to_latest is False
    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q0" in texts
    assert "q29" not in texts
    assert any(
        _row_kind(record) == "tool" and "a.py" in _row_text(record)
        for record in chat.transcript.records
    )

    _submit(qtbot, chat, "hi")
    qtbot.waitUntil(lambda: session.prompts == ["hi"], timeout=10_000)
    wait_idle(qtbot, app)

    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q0" in texts
    assert "q29" not in texts
    assert _row_text(chat.transcript.records[-1]) == "hello"
    assert chat.transcript.pinned_to_latest is False
    assert chat.transcript.viewport_turn() == 0


def test_chat_shows_streamed_tool_call_and_detailed_result(qtbot, tmp_path: Path) -> None:
    session = FakeSession(
        [
            ToolCall(
                id="call-1",
                function=ToolCall.FunctionBody(name="read", arguments=""),
            ),
            ToolCallPart(arguments_part='{"path":'),
            ToolCallPart(arguments_part='"a.py"}'),
            ToolResult(
                tool_call_id="call-1",
                return_value=ToolReturnValue(
                    is_error=False,
                    output="file contents",
                    message="success",
                    display=[BriefDisplayBlock(text="read a.py")],
                    extras=None,
                ),
            ),
            TurnEnd(),
        ]
    )

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: session.prompts == ["go"], timeout=10_000)
    wait_idle(qtbot, app)
    # Call and result share one row; there is no separate tool_result record.
    tool = next(record for record in chat.transcript.records if _row_kind(record) == "tool")
    assert isinstance(tool.entry, ActivityEntry)
    assert tool.entry.activity.call_id == "call-1"
    assert tool.entry.activity.call is not None
    assert tool.entry.activity.result is not None
    assert tool.entry.activity.result.status == "ok"
    assert tool.entry.activity.call.fields[0].name == "path"
    assert "a.py" in _row_text(tool)
    assert "file contents" in _row_text(tool)
    layout = layout_record(tool.entry, width=120)
    assert layout.status == "ok"
    assert "read" in layout.header.lower()


def test_chat_keeps_detailed_status_after_turn_finishes(qtbot, tmp_path: Path) -> None:
    session = FakeSession(
        [
            StatusUpdate(
                context_tokens=2_000,
                max_context_tokens=20_000,
                context_usage=0.1,
            ),
            StatusUpdate(
                token_usage=TokenUsage(
                    input_other=100,
                    input_cache_read=800,
                    input_cache_creation=50,
                    output=75,
                ),
                message_id="msg-1",
            ),
            TurnEnd(),
        ]
    )

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: session.prompts == ["go"], timeout=10_000)
    wait_idle(qtbot, app)
    session_line = widget_text(chat, "status")
    context = widget_text(chat, "context")
    # Positive equality, not `"context" not in session_line`: the status line is
    # translated copy now, so a negative check against an English word would hold
    # for any localized text and stop proving that the two labels stay separate.
    assert session_line == "session fake-session"
    assert "context 2,000/20,000" in context
    assert "tokens in 950" in context
    assert "cache read 800" in context
    assert "out 75" in context
    # The message id itself, not the word "message": a value can never be
    # translated away, so this keeps proving that the id is not shown.
    assert "msg-1" not in context


def test_resumed_session_shows_recent_history(qtbot, tmp_path: Path) -> None:
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    async def history_loader(_work_dir: Path, session_id: str) -> SessionHistory:
        assert session_id == "fake-session"
        return SessionHistory(
            entries=(
                HistoryEntry(_text_entry("user", "fix login"), turn=0, ordinal=0),
                HistoryEntry(_text_entry("assistant", "Check the redirect."), turn=0, ordinal=1),
            ),
            omitted_turns=1,
        )

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        history_loader=history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    records = [(_row_kind(record), _row_text(record)) for record in chat.transcript.records]
    assert ("system", "Session: fake-session") in records
    # Singular gets its own whole sentence, so this no longer reads "last 1 turns".
    assert (
        "system",
        "Showing the last turn (1 earlier omitted)",
    ) in records
    assert ("user", "fix login") in records
    assert ("assistant", "Check the redirect.") in records


def test_chat_chrome_keeps_history_toolbar(qtbot, tmp_path: Path) -> None:
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    toolbar = find(chat, "history-toolbar")
    assert toolbar.isVisible()
    assert widget_text(chat, "history-info").startswith("History ·")
    assert find(chat, "open-settings", QPushButton).isVisible()
    assert find(chat, "leave-session", QPushButton).text() == "Home"
    assert find(chat, "load-older", QPushButton).text() == "←"
    status = find(chat, "status", QLabel)
    context = find(chat, "context", QLabel)
    assert status.parent().objectName() == "chat-toolbar"
    assert context.parent().objectName() == "composer-dock"
    assert "connecting" in status.text() or "session" in status.text().casefold()
    assert "context" in context.text() or "ready" in context.text().casefold()


def test_chat_history_toolbar_stays_visible_for_short_sessions(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(2)

    monkeypatch.setattr("kimix_gui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    assert find(chat, "history-toolbar").isVisible()
    assert widget_text(chat, "history-info").startswith("History · Turn 2 of 2")
    assert find(chat, "load-older", QPushButton).isEnabled()
    assert find(chat, "history-turn").isEnabled()


def test_chat_loads_older_history_pages_without_losing_latest_rows(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(130)

    monkeypatch.setattr("kimix_gui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app, size=(640, 700))
    chat = wait_chat_ready(qtbot, app)
    assert widget_text(chat, "history-info").startswith("History · Turn 130 of 130")
    turn_input = find(chat, "history-turn")
    assert turn_input.geometry().right() <= chat.width()
    assert turn_input.placeholderText() == "Turn 1-130"
    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q0" not in texts
    assert "q129" in texts
    assert chat.transcript.has_hidden_older_history is True

    chat.load_older_history()
    wait_idle(qtbot, app)
    qtbot.waitUntil(lambda: chat.transcript.pinned_to_latest is False, timeout=10_000)
    older_info = widget_text(chat, "history-info")
    assert older_info.startswith("History · Turn ")
    assert older_info.endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False
    assert any(_row_text(record) == "q128" for record in chat.transcript.records)
    assert any(_row_text(record) == "q129" for record in chat.transcript.records)

    qtbot.keyClick(window, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
    wait_idle(qtbot, app)
    assert widget_text(chat, "history-info").endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False

    qtbot.keyClick(window, Qt.Key.Key_F3)
    assert turn_input.hasFocus()
    turn_input.setText("5")
    turn_input.returnPressed.emit()
    wait_idle(qtbot, app)
    qtbot.waitUntil(
        lambda: any(_row_text(record) == "q4" for record in chat.transcript.records),
        timeout=10_000,
    )

    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q4" in texts
    assert "q129" not in texts
    jumped_info = widget_text(chat, "history-info")
    assert jumped_info.startswith("History · Turn ")
    assert jumped_info.endswith(" of 130")
    assert "q4" in chat.transcript.visible_text()

    find(chat, "load-newer", QPushButton).click()
    wait_idle(qtbot, app)
    newer_info = widget_text(chat, "history-info")
    assert newer_info.startswith("History · Turn ")
    assert newer_info.endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False

    find(chat, "jump-latest", QPushButton).click()
    wait_idle(qtbot, app)
    assert widget_text(chat, "history-info").startswith("History · Turn 130 of 130")
    assert chat.transcript.pinned_to_latest is True
    texts = [_row_text(record) for record in chat.transcript.records]
    assert "q129" in texts
    assert "q0" not in texts


def test_approval_is_resolved_from_keyboard(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TextPart(text="done"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    qtbot.keyClick(app.screen, Qt.Key.Key_A)
    wait_idle(qtbot, app)

    assert getattr(approval, "resolved", True)
    assert app.screen is chat
    assert any(_row_text(record) == "done" for record in chat.transcript.records)
    assert any(
        "Approval decision: approve" in _row_text(record) for record in chat.transcript.records
    )


def test_approval_is_resolved_by_clicking_approve(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    find(app.screen, "approve").click()
    wait_idle(qtbot, app)

    assert any(
        "Approval decision: approve" in _row_text(record) for record in chat.transcript.records
    )
    assert app.screen is chat


def test_escape_rejects_modal_without_leaving_chat(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    qtbot.keyClick(app.screen, Qt.Key.Key_Escape)
    wait_idle(qtbot, app)

    assert any(
        "Approval decision: reject" in _row_text(record) for record in chat.transcript.records
    )
    assert app.screen is chat
    assert chat.session_id == "fake-session"
    assert session.closed is False


def test_cancelling_prompt_unblocks_hung_generation(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    qtbot.keyClick(window, Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier)
    wait_idle(qtbot, app)

    assert session.prompts == ["cont"]
    assert session.cancelled is True
    assert chat.busy is False
    assert chat.prompt_enabled is True


def test_leave_during_running_prompt_returns_home_quietly(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    wait_home(qtbot, app)

    assert isinstance(app.screen, HomeView)
    assert window.chat is None
    assert chat.busy is False
    assert session.cancelled is True
    assert session.closed is True


def _todo_result(*items: dict) -> ToolResult:
    return ToolResult(
        tool_call_id="call-todo",
        return_value=ToolReturnValue(
            is_error=False,
            output="Current todo list:",
            message="Todo list updated.",
            display=[TodoDisplayBlock(items=list(items))],
            extras=None,
        ),
    )


def test_todo_panel_follows_todo_tool_results(qtbot, tmp_path: Path) -> None:
    session = FakeSession(
        [
            ToolCall(
                id="call-todo",
                function=ToolCall.FunctionBody(name="todo_write", arguments=""),
            ),
            _todo_result(
                {"title": "Explore repo", "status": "done"},
                {"title": "Wire the panel", "status": "in_progress", "notes": "qt overlay"},
                {"title": "Sub step", "status": "pending", "depth": 1},
                {"title": "Ship", "status": "pending"},
            ),
            TurnEnd(),
        ]
    )

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    assert chat.todo_panel.isVisible() is False

    _submit(qtbot, chat, "plan it")
    qtbot.waitUntil(lambda: session.prompts == ["plan it"], timeout=10_000)
    wait_idle(qtbot, app)
    qtbot.waitUntil(lambda: chat.todo_panel.isVisible(), timeout=10_000)

    snapshot = chat.todo_panel.snapshot
    assert [(entry.title, entry.status, entry.depth) for entry in snapshot.entries] == [
        ("Explore repo", "done", 0),
        ("Wire the panel", "in_progress", 0),
        ("Sub step", "pending", 1),
        ("Ship", "pending", 0),
    ]
    assert widget_text(chat, "todo-count") == "1/4"
    assert widget_text(chat, "todo-footer") == "1 in progress · 2 pending · 1 done"
    assert snapshot.active is not None
    assert snapshot.active.title == "Wire the panel"


def test_todo_panel_toggles_with_ctrl_t(qtbot, tmp_path: Path) -> None:
    session = FakeSession([_todo_result({"title": "one", "status": "pending"}), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "plan")
    qtbot.waitUntil(lambda: chat.todo_panel.isVisible(), timeout=10_000)
    assert chat.todo_panel.expanded is True

    qtbot.keyClick(window, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier)
    assert chat.todo_panel.expanded is False
    qtbot.keyClick(window, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier)
    assert chat.todo_panel.expanded is True


def test_todo_panel_loads_persisted_state_on_resume(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "sessions" / "fake-session" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        json.dumps(
            {
                "todos": [
                    {
                        "title": "Resumed parent",
                        "status": "in_progress",
                        "notes": "carried over",
                        "children": [{"title": "Resumed child", "status": "done"}],
                    }
                ],
                "archived_todos": [{"title": "old", "status": "done"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kimix_gui.todos.session_state_file",
        lambda _work_dir, session_id: tmp_path / "sessions" / session_id / "state.json",
    )
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    qtbot.waitUntil(lambda: chat.todo_panel.isVisible(), timeout=10_000)

    snapshot = chat.todo_panel.snapshot
    assert [(entry.title, entry.status, entry.depth) for entry in snapshot.entries] == [
        ("Resumed parent", "in_progress", 0),
        ("Resumed child", "done", 1),
    ]
    assert snapshot.archived == 1
    assert widget_text(chat, "todo-footer") == "1 in progress · 1 done · 1 archived"


def test_clear_command_empties_the_todo_panel(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "sessions" / "fake-session" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        json.dumps({"todos": [{"title": "before clear", "status": "pending"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kimix_gui.todos.session_state_file",
        lambda _work_dir, session_id: tmp_path / "sessions" / session_id / "state.json",
    )
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixGuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    qtbot.waitUntil(lambda: chat.todo_panel.isVisible(), timeout=10_000)

    state_file.unlink()
    chat.prompt.submitted.emit("/clear")
    wait_idle(qtbot, app)
    qtbot.waitUntil(lambda: not chat.todo_panel.isVisible(), timeout=10_000)
    assert chat.todo_panel.snapshot.is_empty is True
