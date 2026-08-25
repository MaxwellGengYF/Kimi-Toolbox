"""Tests for the conversational Agent system."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kaos.path import KaosPath

from kimix.base import MessageType
from kimix.tools.agent import (
    Agent,
    AgentClose,
    AgentCloseParams,
    AgentList,
    AgentListParams,
    AskAgent,
    AskAgentParams,
    SubAgentParams,
    _AgentConversationCollector,
    _drain_pending_messages,
    _format_pending_messages,
    _get_agent_session,
    _get_store,
    _pending_message_count,
    _queue_pending_message,
    _register_agent_session,
    _register_entry,
    _resolve_prompt,
    _unregister_agent_session,
    _unregister_entry,
)
from kimix.tools.agent.store import (
    AgentSessionEntry,
    AgentSessionStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    session.custom_config = {}
    return session


@pytest.fixture
def mock_sub_session() -> MagicMock:
    session = MagicMock()
    session.id = "sub-123"
    session.get_custom_config.return_value = {}
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# AgentSessionStore tests
# ---------------------------------------------------------------------------
async def test_store_get_put(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    assert store.get("nonexistent") is None

    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    assert store.get("s1") is entry


async def test_store_close(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    assert store.close("s1") is True
    assert store.get("s1") is None
    assert store.close("s1") is False


async def test_store_list_active(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=3,
    )
    store.put(entry)
    active = store.list_active()
    assert len(active) == 1
    assert active[0]["session_id"] == "s1"
    assert active[0]["total_turns"] == 3
    assert active[0]["state"] == "running"


async def test_store_lru_eviction(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    store.MAX_SESSIONS = 3

    for i in range(4):
        entry = AgentSessionEntry(
            session=mock_sub_session,
            session_id=f"s{i}",
            created_at=time.time(),
            last_accessed=time.time() + i,
            conversation_history=[],
            total_turns=0,
        )
        store.put(entry)

    assert len(store.entries) == 4
    with patch(
        "kimix.tools.agent.store.close_session_async", new_callable=AsyncMock
    ) as mock_close:
        await store.evict_lru_if_needed()
        assert mock_close.await_count == 2

    assert len(store.entries) == 2
    assert store.get("s0") is None  # oldest
    assert store.get("s1") is None
    assert store.get("s2") is not None
    assert store.get("s3") is not None


# ---------------------------------------------------------------------------
# _AgentConversationCollector tests
# ---------------------------------------------------------------------------
async def test_collector_text_only() -> None:
    col = _AgentConversationCollector()
    col.finalize_user_turn("hello")
    col.consume("world", MessageType.Text)
    text = col.finalize_assistant_turn()
    assert text == "world"
    assert len(col.turns) == 2
    assert col.turns[0].role == "user"
    assert col.turns[1].role == "assistant"
    assert col.turns[1].metadata == {"type": "text"}


async def test_collector_thinking_excluded_from_output() -> None:
    col = _AgentConversationCollector()
    col.consume("text1", MessageType.Text)
    col.consume("think1", MessageType.Thinking)
    col.consume("text2", MessageType.Text)
    text = col.finalize_assistant_turn()
    assert text == "text1text2"
    assert any(t.metadata == {"type": "thinking"} for t in col.turns)


async def test_collector_tool_call_and_result() -> None:
    col = _AgentConversationCollector()
    col.consume("ToolA args", MessageType.ToolCalling)
    col.consume("[ToolResult] ok", MessageType.ToolResult)
    text = col.finalize_assistant_turn()
    assert text == ""
    roles = [t.role for t in col.turns]
    assert roles == ["tool", "tool"]
    assert col.turns[0].metadata == {"type": "tool_call"}
    assert col.turns[1].metadata == {"type": "tool_result"}


async def test_collector_empty_output() -> None:
    col = _AgentConversationCollector()
    col.finalize_user_turn("prompt")
    text = col.finalize_assistant_turn()
    assert text == ""


# ---------------------------------------------------------------------------
# Agent tool tests
# ---------------------------------------------------------------------------
async def test_agent_recursive_guard(mock_session: MagicMock) -> None:
    mock_session.custom_config = {"is_sub_agent": True}
    agent = Agent(mock_session)
    result = await agent(SubAgentParams(prompt="test"))
    assert result.is_error
    assert "Recursive sub-agent call detected" in result.message


async def test_agent_new_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X"))

    assert not result.is_error
    assert result.extras is not None
    assert "session_id" in result.extras
    assert result.extras["status"] == "closed"
    assert result.extras["turn_count"] >= 1
    mock_create.assert_awaited_once()
    mock_prompt.assert_awaited_once()


async def test_agent_keep_alive_stores_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", close_session=False))

    assert not result.is_error
    assert result.extras["status"] == "continued"
    store = _get_store(mock_session)
    assert store.get(result.extras["session_id"]) is not None
    mock_close.assert_not_awaited()


async def test_agent_reuse_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="reuse-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=0,
        )
    )

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            agent = Agent(mock_session)
            result = await agent(
                SubAgentParams(prompt="follow up", session_id="reuse-id", close_session=False)
            )

    assert not result.is_error
    assert result.extras["session_id"] == "reuse-id"
    mock_create.assert_not_awaited()


async def test_agent_close_session_param(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="close-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=0,
        )
    )

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="do X", session_id="close-id", close_session=True
                    )
                )

    assert not result.is_error
    assert result.extras["status"] == "closed"
    assert store.get("close-id") is None
    mock_close.assert_awaited_once()


async def test_agent_return_history(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", return_history=True))

    assert not result.is_error
    assert "conversation_history" in result.extras
    history = result.extras["conversation_history"]
    assert isinstance(history, list)
    assert any(h["role"] == "user" for h in history)


async def test_agent_error_path(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = RuntimeError("boom")
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", close_session=False))

    assert result.is_error
    assert "boom" in result.message
    assert result.extras["status"] == "closed"
    mock_close.assert_awaited_once()


async def test_agent_error_saves_prompt_file(
    mock_session: MagicMock, mock_sub_session: MagicMock, tmp_path: Path
) -> None:
    """A failed sub-agent run saves the effective prompt for retry."""
    mock_session.custom_config = {"chat_provider": None}
    saved_files: list[Path] = []

    def fake_create_script_file(content: str, ext: str = ".md") -> str:
        target = tmp_path / f"saved_{len(saved_files)}{ext}"
        target.write_text(content, encoding="utf-8")
        saved_files.append(target)
        return str(target)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = RuntimeError("boom")
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                with patch(
                    "kimix.tools.agent._create_script_file",
                    side_effect=fake_create_script_file,
                ):
                    agent = Agent(mock_session)
                    result = await agent(
                        SubAgentParams(prompt="do X", close_session=False)
                    )

    assert result.is_error
    assert "boom" in result.message
    assert "[prompt saved to" in result.message
    assert "prompt=@" in result.message
    expected_display = str(tmp_path / "saved_0.md").replace("\\", "/")
    assert result.extras["prompt_file"] == expected_display
    # The saved file contains the exact prompt string sent to prompt_async.
    sent_prompt = mock_prompt.await_args.kwargs["prompt_str"]
    assert saved_files[0].read_text(encoding="utf-8") == sent_prompt
    mock_close.assert_awaited_once()


async def test_agent_prompt_from_file(
    mock_session: MagicMock, mock_sub_session: MagicMock, tmp_path: Path
) -> None:
    """prompt=@path reads the task text from the referenced file."""
    task_file = tmp_path / "task.md"
    task_file.write_text("do the file task", encoding="utf-8")
    mock_session.custom_config = {"chat_provider": None}
    mock_session.work_dir = KaosPath(str(tmp_path))

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(prompt="@task.md", close_session=False)
                )

    assert not result.is_error
    prompt_str = mock_prompt.await_args.kwargs["prompt_str"]
    assert "do the file task" in prompt_str
    assert "@task.md" not in prompt_str


async def test_agent_prompt_file_missing(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """prompt=@missing.md fails with a clear prompt-file error."""
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        agent = Agent(mock_session)
        result = await agent(SubAgentParams(prompt="@missing.md"))

    assert result.is_error
    assert "prompt file not found" in result.message


async def test_resolve_prompt_cwd_fallback(monkeypatch, tmp_path: Path) -> None:
    """Relative @path falls back to the process CWD when not under base_dir.

    The error-path retry hint points at the shared temp folder
    (``.kimix_cache/tmp_<pid>/<n>.md``), which is CWD-relative; the fallback
    keeps that retry working when the session work dir differs from CWD.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.md").write_text("from cwd", encoding="utf-8")

    assert _resolve_prompt("@task.md", work_dir) == "from cwd"
    with pytest.raises(FileNotFoundError):
        _resolve_prompt("@nope.md", work_dir)


async def test_agent_long_prompt_offloads_to_temp_file(
    mock_session: MagicMock, mock_sub_session: MagicMock, tmp_path: Path
) -> None:
    """Very long prompts are offloaded to a temp file the sub-agent reads."""
    mock_session.custom_config = {"chat_provider": None}
    prompt_text = "x" * (100 * 1024 + 1)
    calls: list[tuple[str, str]] = []
    fake_path = tmp_path / "saved.md"

    def fake_create_script_file(content: str, ext: str = ".md") -> str:
        calls.append((content, ext))
        return str(fake_path)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                with patch(
                    "kimix.tools.agent._create_script_file",
                    side_effect=fake_create_script_file,
                ):
                    agent = Agent(mock_session)
                    result = await agent(SubAgentParams(prompt=prompt_text))

    assert not result.is_error
    assert calls == [(prompt_text, ".md")]
    prompt_str = mock_prompt.await_args.kwargs["prompt_str"]
    assert "Please read the task from `" in prompt_str
    assert "` and execute it." in prompt_str


async def test_agent_lru_eviction(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.MAX_SESSIONS = 2

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.store.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                for i in range(3):
                    mock_sub = MagicMock()
                    mock_sub.id = f"sub-{i}"
                    mock_sub.get_custom_config.return_value = {}
                    mock_sub.close = AsyncMock()
                    mock_create.return_value = mock_sub
                    await agent(
                        SubAgentParams(prompt=f"task {i}", close_session=False)
                    )

    assert len(store.entries) == 2
    mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# inherit_context tests (mirrors CLI /store + /load session-copy logic)
# ---------------------------------------------------------------------------
async def test_agent_inherit_context_copies_parent_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """inherit_context=True copies the parent session dir into the new sub id."""
    mock_session.custom_config = {"chat_provider": None}
    mock_session.id = "parent-1"
    mock_session.work_dir = KaosPath(".")

    with patch(
        "kimix.tools.agent.Session.copy", new_callable=AsyncMock
    ) as mock_copy:
        with patch(
            "kimix.tools.agent._create_session_async", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_sub_session
            with patch(
                "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
            ):
                with patch(
                    "kimix.tools.agent.close_session_async", new_callable=AsyncMock
                ):
                    agent = Agent(mock_session)
                    result = await agent(
                        SubAgentParams(prompt="do X", inherit_context=True)
                    )

    assert not result.is_error
    sub_id = result.extras["session_id"]
    mock_copy.assert_awaited_once()
    copied_work_dir, copied_parent, copied_target = mock_copy.await_args.args
    assert copied_work_dir == KaosPath(".")
    assert copied_parent == "parent-1"
    assert copied_target == sub_id
    # The sub-agent session resumes from the copied session id (like /load).
    assert mock_create.await_args.kwargs["session_id"] == sub_id
    assert mock_create.await_args.kwargs["resume"] is True


async def test_agent_inherit_context_with_explicit_session_id(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """The parent context is copied into the explicitly requested session id."""
    mock_session.custom_config = {"chat_provider": None}
    mock_session.id = "parent-1"
    mock_session.work_dir = KaosPath(".")

    with patch(
        "kimix.tools.agent.Session.copy", new_callable=AsyncMock
    ) as mock_copy:
        with patch(
            "kimix.tools.agent._create_session_async", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_sub_session
            with patch(
                "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
            ):
                with patch(
                    "kimix.tools.agent.close_session_async", new_callable=AsyncMock
                ):
                    agent = Agent(mock_session)
                    result = await agent(
                        SubAgentParams(
                            prompt="do X",
                            session_id="fresh-sub",
                            inherit_context=True,
                        )
                    )

    assert not result.is_error
    assert result.extras["session_id"] == "fresh-sub"
    mock_copy.assert_awaited_once_with(
        KaosPath("."), "parent-1", "fresh-sub"
    )
    assert mock_create.await_args.kwargs["session_id"] == "fresh-sub"


async def test_agent_inherit_context_ignored_on_reuse(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """Reusing an active sub-agent session skips the context copy."""
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="reuse-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=0,
        )
    )

    with patch(
        "kimix.tools.agent.Session.copy", new_callable=AsyncMock
    ) as mock_copy:
        with patch(
            "kimix.tools.agent._create_session_async", new_callable=AsyncMock
        ) as mock_create:
            with patch(
                "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="follow up",
                        session_id="reuse-id",
                        inherit_context=True,
                        close_session=False,
                    )
                )

    assert not result.is_error
    assert result.extras["session_id"] == "reuse-id"
    mock_copy.assert_not_awaited()
    mock_create.assert_not_awaited()


async def test_agent_inherit_context_without_parent_id_errors(
    mock_session: MagicMock,
) -> None:
    """A parent with no resolvable session id cannot donate its context."""
    mock_session.custom_config = {"chat_provider": None}
    mock_session.id = ""
    mock_session._cli = None

    with patch(
        "kimix.tools.agent.Session.copy", new_callable=AsyncMock
    ) as mock_copy:
        agent = Agent(mock_session)
        result = await agent(SubAgentParams(prompt="do X", inherit_context=True))

    assert result.is_error
    assert "Cannot inherit parent context" in result.message
    mock_copy.assert_not_awaited()


async def test_agent_inherit_context_resets_system_prompt(
    mock_session: MagicMock,
) -> None:
    """The inherited parent system prompt is replaced with the sub-agent's."""
    agent = Agent(mock_session)

    sub_agent = MagicMock()
    sub_agent.get_system_prompt.return_value = "sub-agent prompt"
    sub_agent.system_prompt_cached = "parent prompt"
    context = MagicMock()
    context.write_system_prompt = AsyncMock()
    soul = MagicMock()
    soul.agent = sub_agent
    soul.context = context
    cli = MagicMock()
    cli.soul = soul
    sub_session = MagicMock()
    sub_session._cli = cli

    await agent._reset_inherited_system_prompt(sub_session)

    assert sub_agent.system_prompt_cached is None
    context.write_system_prompt.assert_awaited_once_with("sub-agent prompt")


async def test_agent_inherit_context_reset_skips_when_soul_missing(
    mock_session: MagicMock,
) -> None:
    """System prompt reset is best-effort: skips without soul internals."""
    agent = Agent(mock_session)
    sub_session = MagicMock()
    sub_session._cli = None
    await agent._reset_inherited_system_prompt(sub_session)


async def test_agent_inherit_context_copies_real_session_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Integration: the parent session dir is really copied (Session.copy)."""
    from kimi_cli.metadata import Metadata, save_metadata
    from kimi_cli.session import Session as CliSession
    from kimi_cli.soul.context_db import ContextDB
    from kosong.message import Message

    share_dir = tmp_path / "share"
    share_dir.mkdir()
    monkeypatch.setattr("kimi_cli.share.get_share_dir", lambda: share_dir)
    monkeypatch.setattr("kimi_cli.metadata.get_share_dir", lambda: share_dir)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    work = KaosPath(str(work_dir)).canonical()

    metadata = Metadata()
    wd_meta = metadata.new_work_dir_meta(work)
    save_metadata(metadata)

    parent_id = "parent-1"
    parent_dir = wd_meta.sessions_dir / parent_id
    parent_dir.mkdir(parents=True, exist_ok=True)

    db = ContextDB(parent_dir / "context.db")
    await db.initialize()
    await db.append_messages(
        [
            Message(role="user", content="hello parent"),
            Message(role="assistant", content="hi there"),
        ]
    )
    await db.close()

    parent = MagicMock()
    parent.id = parent_id
    parent.work_dir = work
    agent = Agent(parent)

    target_id = "sub-copy"
    await agent._inherit_parent_context(target_id)

    copied = await CliSession.find(work, target_id)
    assert copied is not None

    copied_db = ContextDB(parent_dir.parent / target_id / "context.db")
    await copied_db.initialize()
    try:
        messages = await copied_db.get_messages()
    finally:
        await copied_db.close()
    texts: list[str] = []
    for m in messages:
        if isinstance(m.content, str):
            texts.append(m.content)
        else:
            texts.extend(
                part.text for part in m.content
                if getattr(part, "type", None) == "text"
            )
    assert texts == ["hello parent", "hi there"]



# ---------------------------------------------------------------------------
# Companion tool tests
# ---------------------------------------------------------------------------
async def test_agent_list(mock_session: MagicMock, mock_sub_session: MagicMock) -> None:
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="list-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    agent_list = AgentList(mock_session)
    result = await agent_list(AgentListParams())
    assert not result.is_error
    assert "list-id" in result.output


async def test_agent_close(mock_session: MagicMock, mock_sub_session: MagicMock) -> None:
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="close-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    with patch(
        "kimix.tools.agent.close_session_async", new_callable=AsyncMock
    ) as mock_close:
        agent_close = AgentClose(mock_session)
        result = await agent_close(AgentCloseParams(session_id="close-id"))

    assert not result.is_error
    assert store.get("close-id") is None
    mock_close.assert_awaited_once()


async def test_agent_close_not_found(mock_session: MagicMock) -> None:
    agent_close = AgentClose(mock_session)
    result = await agent_close(AgentCloseParams(session_id="missing"))
    assert result.is_error
    assert "Session not found" in result.message


# ---------------------------------------------------------------------------
# AskAgent tests
# ---------------------------------------------------------------------------
class _FakeSteer:
    """Stand-in for ``kimi_cli.soul.steer.Steer`` recording pushed content."""

    def __init__(self, delivered: bool = True) -> None:
        self.delivered = delivered
        self.pushed: list[str] = []

    async def push(self, content: str) -> bool:
        self.pushed.append(content)
        return self.delivered


def _fake_sdk_session(session_id: str) -> MagicMock:
    """A stand-in SDK session exposing ``_cli.soul`` + ``_cli.session.id``."""
    sdk = MagicMock()
    cli = MagicMock()
    cli_session = MagicMock()
    cli_session.id = session_id
    cli.session = cli_session
    cli.soul = MagicMock()
    sdk._cli = cli
    return sdk


async def test_ask_agent_sub_agent_messages_parent(mock_sub_session: MagicMock) -> None:
    mock_sub_session.id = "sub-123"
    mock_sub_session.custom_config = {
        "is_sub_agent": True,
        "parent_session_id": "parent-1",
    }
    parent_sdk = _fake_sdk_session("parent-1")
    _register_agent_session("parent-1", parent_sdk)
    try:
        fake = _FakeSteer(delivered=True)
        with patch("kimi_cli.soul.steer.Steer.from_session", return_value=fake) as mock_from:
            ask_agent = AskAgent(mock_sub_session)
            result = await ask_agent(AskAgentParams(question="What is the color?"))
        assert not result.is_error
        assert "parent-1" in result.output
        assert fake.pushed == ["Message from agent 'sub-123':\nWhat is the color?"]
        mock_from.assert_called_once_with(parent_sdk)
    finally:
        _unregister_agent_session("parent-1")


async def test_ask_agent_sub_agent_ignores_id(mock_sub_session: MagicMock) -> None:
    """The ``id`` param is ignored for sub-agents: they always message the parent."""
    mock_sub_session.id = "sub-123"
    mock_sub_session.custom_config = {
        "is_sub_agent": True,
        "parent_session_id": "parent-1",
    }
    parent_sdk = _fake_sdk_session("parent-1")
    _register_agent_session("parent-1", parent_sdk)
    try:
        fake = _FakeSteer(delivered=True)
        with patch("kimi_cli.soul.steer.Steer.from_session", return_value=fake):
            ask_agent = AskAgent(mock_sub_session)
            result = await ask_agent(
                AskAgentParams(question="ignored id", id="some-other-agent")
            )
        assert not result.is_error
        assert fake.pushed == ["Message from agent 'sub-123':\nignored id"]
    finally:
        _unregister_agent_session("parent-1")


async def test_ask_agent_sub_agent_without_parent_id_errors(
    mock_sub_session: MagicMock,
) -> None:
    mock_sub_session.id = "sub-123"
    mock_sub_session.custom_config = {"is_sub_agent": True}
    ask_agent = AskAgent(mock_sub_session)
    result = await ask_agent(AskAgentParams(question="hi"))
    assert result.is_error
    assert "parent_session_id" in result.message


async def test_ask_agent_sub_agent_parent_not_registered_queues(
    mock_sub_session: MagicMock,
) -> None:
    """A message to an unregistered (closed) parent is queued, not an error."""
    mock_sub_session.id = "sub-123"
    mock_sub_session.custom_config = {
        "is_sub_agent": True,
        "parent_session_id": "missing-parent",
    }
    ask_agent = AskAgent(mock_sub_session)
    result = await ask_agent(AskAgentParams(question="hi"))
    assert not result.is_error
    assert "queued" in result.output
    assert "missing-parent" in result.output
    assert _pending_message_count("missing-parent") == 1
    assert _drain_pending_messages("missing-parent") == [
        "Message from agent 'sub-123':\nhi"
    ]


async def test_ask_agent_main_agent_targets_by_id(mock_session: MagicMock) -> None:
    mock_session.custom_config = {}
    mock_session.id = "main-1"
    target_sdk = _fake_sdk_session("target-1")
    _register_agent_session("target-1", target_sdk)
    try:
        fake = _FakeSteer(delivered=True)
        with patch("kimi_cli.soul.steer.Steer.from_session", return_value=fake):
            ask_agent = AskAgent(mock_session)
            result = await ask_agent(AskAgentParams(question="status?", id="target-1"))
        assert not result.is_error
        assert fake.pushed == ["Message from agent 'main-1':\nstatus?"]
    finally:
        _unregister_agent_session("target-1")


async def test_ask_agent_main_agent_defaults_to_recent_active(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {}
    store = _get_store(mock_session)
    old_sdk = _fake_sdk_session("old-1")
    recent_sdk = _fake_sdk_session("recent-1")
    store.put(
        AgentSessionEntry(
            session=old_sdk,
            session_id="old-1",
            created_at=time.time(),
            last_accessed=time.time() - 100,
            conversation_history=[],
            total_turns=1,
        )
    )
    store.put(
        AgentSessionEntry(
            session=recent_sdk,
            session_id="recent-1",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    fake = _FakeSteer(delivered=True)
    with patch("kimi_cli.soul.steer.Steer.from_session", return_value=fake) as mock_from:
        ask_agent = AskAgent(mock_session)
        result = await ask_agent(AskAgentParams(question="anyone there?"))
    assert not result.is_error
    assert "recent-1" in result.output
    mock_from.assert_called_once_with(recent_sdk)


async def test_ask_agent_main_agent_closed_session_queues(
    mock_session: MagicMock,
) -> None:
    """A message to a closed/unregistered session is queued, not an error."""
    mock_session.custom_config = {}
    mock_session.id = "main-1"
    ask_agent = AskAgent(mock_session)
    result = await ask_agent(AskAgentParams(question="ping?", id="ghost"))
    assert not result.is_error
    assert "queued" in result.output
    assert "ghost" in result.output
    assert _pending_message_count("ghost") == 1
    assert _drain_pending_messages("ghost") == ["Message from agent 'main-1':\nping?"]


async def test_ask_agent_queued_output_explains_resume(mock_session: MagicMock) -> None:
    """Queued output must say delivery happens only on subagent resume.

    Without this, an agent may believe a message to a closed session will be
    delivered automatically at some future prompt that never comes.
    """
    mock_session.custom_config = {}
    mock_session.id = "main-1"
    ask_agent = AskAgent(mock_session)
    result = await ask_agent(AskAgentParams(question="ping?", id="ghost"))
    assert not result.is_error
    assert "queued" in result.output
    assert "subagent(session_id='ghost'" in result.output
    assert "only if you resume" in result.output
    assert _pending_message_count("ghost") == 1
    _drain_pending_messages("ghost")


async def test_ask_agent_main_agent_no_active_sub_agents_errors(
    mock_session: MagicMock,
) -> None:
    mock_session.custom_config = {}
    ask_agent = AskAgent(mock_session)
    result = await ask_agent(AskAgentParams(question="hi"))
    assert result.is_error
    assert "no active sub-agents" in result.message


async def test_ask_agent_rejects_self_message(mock_session: MagicMock) -> None:
    mock_session.custom_config = {}
    mock_session.id = "self-1"
    target_sdk = _fake_sdk_session("self-1")
    _register_agent_session("self-1", target_sdk)
    try:
        ask_agent = AskAgent(mock_session)
        result = await ask_agent(AskAgentParams(question="hi", id="self-1"))
        assert result.is_error
        assert "yourself" in result.message
    finally:
        _unregister_agent_session("self-1")


async def test_ask_agent_queues_when_target_idle(mock_session: MagicMock) -> None:
    mock_session.custom_config = {}
    mock_session.id = "main-1"
    target_sdk = _fake_sdk_session("target-1")
    _register_agent_session("target-1", target_sdk)
    try:
        fake = _FakeSteer(delivered=False)
        with patch("kimi_cli.soul.steer.Steer.from_session", return_value=fake):
            ask_agent = AskAgent(mock_session)
            result = await ask_agent(AskAgentParams(question="hi", id="target-1"))
        assert not result.is_error
        assert "queued" in result.output
        # The message is persisted (not just left in the steer queue, which
        # would be discarded as stale at the next turn init).
        assert _pending_message_count("target-1") == 1
        assert _drain_pending_messages("target-1") == [
            "Message from agent 'main-1':\nhi"
        ]
    finally:
        _unregister_agent_session("target-1")
        _drain_pending_messages("target-1")


async def test_ask_agent_target_not_steerable_queues(mock_session: MagicMock) -> None:
    """A target without a steerable session is queued, not an error."""
    mock_session.custom_config = {}
    mock_session.id = "main-1"
    target_sdk = _fake_sdk_session("target-1")
    _register_agent_session("target-1", target_sdk)
    try:
        with patch("kimi_cli.soul.steer.Steer.from_session", return_value=None):
            ask_agent = AskAgent(mock_session)
            result = await ask_agent(AskAgentParams(question="hi", id="target-1"))
        assert not result.is_error
        assert "queued" in result.output
        assert _pending_message_count("target-1") == 1
        assert _drain_pending_messages("target-1") == [
            "Message from agent 'main-1':\nhi"
        ]
    finally:
        _unregister_agent_session("target-1")
        _drain_pending_messages("target-1")


def test_ask_agent_tool_name_is_report_canonical() -> None:
    """The tool registers under the report-canonical ``send_message`` name."""
    assert AskAgent.name == "send_message"


async def test_format_pending_messages() -> None:
    assert _format_pending_messages([]) == ""
    block = _format_pending_messages(["msg one", "msg two"])
    assert "<pending-messages>" in block
    assert "</pending-messages>" in block
    assert "1. msg one" in block
    assert "2. msg two" in block
    assert block.index("1. msg one") < block.index("2. msg two")


async def test_pending_message_queue_roundtrip() -> None:
    _queue_pending_message("s1", "first")
    _queue_pending_message("s1", "second")
    try:
        assert _pending_message_count("s1") == 2
        assert _drain_pending_messages("s1") == ["first", "second"]
        assert _pending_message_count("s1") == 0
        # Draining a session with no queued messages yields [].
        assert _drain_pending_messages("never-queued") == []
    finally:
        _drain_pending_messages("s1")


async def test_agent_resume_lists_pending_messages(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """Resuming an idle sub-agent lists messages queued by ``AskAgent`` at the
    top of its next prompt."""
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="idle-1",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    _queue_pending_message("idle-1", "Message from agent 'main-1':\nstatus?")
    captured_prompt = None

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_str
        if output_function:
            output_function("done", MessageType.Text)

    try:
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = _mock_prompt_async
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="resume", session_id="idle-1", close_session=False
                    )
                )

        assert not result.is_error
        assert result.extras["session_id"] == "idle-1"
        assert captured_prompt is not None
        assert "<pending-messages>" in captured_prompt
        assert "status?" in captured_prompt
        assert "resume" in captured_prompt
        # The queued message was consumed (listed once, not repeatedly).
        assert _pending_message_count("idle-1") == 0
    finally:
        _unregister_agent_session("idle-1")
        _drain_pending_messages("idle-1")


async def test_agent_new_session_lists_pending_messages_for_closed_id(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """Re-creating a closed session under the same id lists queued messages at
    the next prompt."""
    mock_session.custom_config = {"chat_provider": None}
    _queue_pending_message("closed-1", "Message from agent 'main-1':\ncome back")
    captured_prompt = None

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_str
        if output_function:
            output_function("ok", MessageType.Text)

    try:
        with patch(
            "kimix.tools.agent._create_session_async", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_sub_session
            with patch(
                "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
            ) as mock_prompt:
                mock_prompt.side_effect = _mock_prompt_async
                with patch(
                    "kimix.tools.agent.close_session_async", new_callable=AsyncMock
                ):
                    agent = Agent(mock_session)
                    result = await agent(
                        SubAgentParams(prompt="do X", session_id="closed-1")
                    )

        assert not result.is_error
        assert result.extras["session_id"] == "closed-1"
        assert captured_prompt is not None
        assert "<pending-messages>" in captured_prompt
        assert "come back" in captured_prompt
        assert _pending_message_count("closed-1") == 0
    finally:
        _unregister_agent_session("closed-1")
        _drain_pending_messages("closed-1")


async def test_agent_resolve_session_registers_parent_and_child(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    mock_session.id = "parent-1"

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", close_session=False))

    assert not result.is_error
    child_id = result.extras["session_id"]
    sub_config = mock_sub_session.get_custom_config()
    assert sub_config["is_sub_agent"] is True
    assert sub_config["parent_session_id"] == "parent-1"
    assert _get_agent_session(child_id) is mock_sub_session
    # Close-path cleanup removes the registration.
    _unregister_agent_session(child_id)
    assert _get_agent_session(child_id) is None


async def test_agent_awaiting_response_status(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="conv-id",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    _register_entry("conv-id", entry)

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        # Simulate sub-agent calling ask_parent during the turn
        entry.pending_question = "What format do you want?"
        entry.state = "awaiting_response"
        if output_function:
            output_function("I need clarification", MessageType.Text)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = _mock_prompt_async
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(prompt="do X", session_id="conv-id", close_session=False)
                )

    assert not result.is_error
    assert result.extras["status"] == "awaiting_response"
    assert result.extras["question"] == "What format do you want?"
    mock_close.assert_not_awaited()
    _unregister_entry("conv-id")


async def test_agent_response_injection(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="resp-id",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
        pending_question="What format?",
        state="awaiting_response",
    )
    store.put(entry)
    _register_entry("resp-id", entry)

    captured_prompt = None

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_str
        if output_function:
            output_function("OK", MessageType.Text)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = _mock_prompt_async
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="continue",
                        session_id="resp-id",
                        response="JSON format",
                        close_session=False,
                    )
                )

    assert not result.is_error
    assert captured_prompt is not None
    assert "JSON format" in captured_prompt
    assert "What format?" in captured_prompt
    assert entry.pending_question is None
    assert entry.state == "running"
    _unregister_entry("resp-id")


# ---------------------------------------------------------------------------
# Work-dir inheritance + context_files base (reflection fix)
# ---------------------------------------------------------------------------


async def test_agent_work_dir_inherited_by_sub_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """Sub-agent sessions must inherit the parent's working directory so the
    system prompt WORK DIR and relative paths resolve against the same repo."""
    from kaos.path import KaosPath

    mock_session.work_dir = KaosPath(str(Path.cwd()))
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                await agent(SubAgentParams(prompt="do X"))

    assert mock_create.await_args is not None
    _, kwargs = mock_create.await_args
    assert kwargs.get("work_dir") == KaosPath(str(Path.cwd()))


async def test_agent_work_dir_sdk_wrapped_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """SDK-wrapped sessions (session._cli.session.work_dir) resolve too."""
    from kaos.path import KaosPath

    inner = MagicMock()
    inner.work_dir = KaosPath(str(Path.cwd()))
    cli = MagicMock()
    cli.session = inner
    mock_session.work_dir = None
    mock_session._cli = cli
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                await agent(SubAgentParams(prompt="do X"))

    assert mock_create.await_args is not None
    _, kwargs = mock_create.await_args
    assert kwargs.get("work_dir") == KaosPath(str(Path.cwd()))


async def test_agent_context_files_resolve_against_work_dir(
    mock_session: MagicMock, mock_sub_session: MagicMock, tmp_path
) -> None:
    """context_files must resolve against the parent's work_dir, not the
    session cache dir (self._session.dir)."""
    from kaos.path import KaosPath

    marker = tmp_path / "marker.txt"
    marker.write_text("hello marker", encoding="utf-8")
    mock_session.work_dir = KaosPath(str(tmp_path))
    mock_session.custom_config = {"chat_provider": None}
    # Ensure the session dir differs from the work dir: a file that exists in
    # the session dir but not the work dir must NOT be picked up.
    session_dir = tmp_path / "session_cache"
    session_dir.mkdir(exist_ok=True)
    mock_session.dir = str(session_dir)

    captured: list[str] = []

    async def fake_prompt(prompt_str, **kwargs):
        captured.append(prompt_str)
        return None

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock, side_effect=fake_prompt
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                await agent(
                    SubAgentParams(prompt="do X", context_files=["marker.txt"])
                )

    assert captured, "prompt_async should have been called"
    assert "<file path='marker.txt'>\nhello marker\n</file>" in captured[0]


async def test_agent_work_dir_none_falls_back_to_cwd(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    """No resolvable work_dir on the parent must not crash the create call."""
    mock_session.work_dir = None
    mock_session._cli = None
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X"))

    assert not result.is_error
    assert mock_create.await_args is not None
    _, kwargs = mock_create.await_args
    assert kwargs.get("work_dir") is None
