"""Tests for the AgentSwarm tool."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kimix.base import MessageType
from kimix.tools.swarm import (
    AgentSwarm,
    AgentSwarmParams,
    SwarmSubagentResult,
    _expand_template,
    _render_results,
    _validate_uniqueness,
)


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {"is_swarm_session": True}
    session.custom_config = {"provider_dict": {"name": "mock"}}
    return session


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_params_accepts_prompt_prefix():
    """prompt_prefix + items as alternative to prompt_template."""
    params = AgentSwarmParams(
        description="test",
        prompt_prefix="Please fix: ",
        items=["file1.py", "file2.py"],
    )
    assert params.prompt_prefix == "Please fix: "
    expanded = _expand_template(None, ["file1.py"], prefix=params.prompt_prefix)
    assert expanded == ["Please fix: file1.py"]


# ---------------------------------------------------------------------------
# Template expansion and uniqueness
# ---------------------------------------------------------------------------
def test_expand_template():
    assert _expand_template("process {{item}}", ["a", "b"]) == [
        "process a",
        "process b",
    ]


def test_validate_uniqueness_ok():
    _validate_uniqueness(["p1", "p2"])


def test_validate_uniqueness_detects_duplicates():
    with pytest.raises(ValueError, match="must be unique"):
        _validate_uniqueness(["p1", "p1"])


# ---------------------------------------------------------------------------
# SkipThisTool / visibility
# ---------------------------------------------------------------------------
def test_agent_swarm_skipped_when_not_swarm_session(mock_session: MagicMock):
    mock_session.custom_data = {}
    from kimi_cli.tools import SkipThisTool

    with pytest.raises(SkipThisTool):
        AgentSwarm(mock_session)


def test_agent_swarm_available_in_swarm_session(mock_session: MagicMock):
    tool = AgentSwarm(mock_session)
    assert tool.name == "AgentSwarm"


# ---------------------------------------------------------------------------
# Recursive guard and exclusivity
# ---------------------------------------------------------------------------
async def test_recursive_sub_agent_call_rejected(mock_session: MagicMock):
    mock_session.custom_config["is_sub_agent"] = True
    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test",
        prompt_template="do {{item}}",
        items=["a", "b"],
    ))
    assert result.is_error is True
    assert "Recursive" in result.message


async def test_concurrent_call_rejected(mock_session: MagicMock):
    mock_session.custom_data["agent_swarm_in_flight"] = True
    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test",
        prompt_template="do {{item}}",
        items=["a", "b"],
    ))
    assert result.is_error is True
    assert "already in progress" in result.message


# ---------------------------------------------------------------------------
# XML rendering
# ---------------------------------------------------------------------------
def test_render_results():
    results = [
        SwarmSubagentResult(index=0, agent_id="a1", output="ok", success=True),
        SwarmSubagentResult(
            index=1, agent_id="a2", output="bad <value>", success=False, error="err"
        ),
    ]
    xml = _render_results(results, "my swarm")
    assert "<agent_swarm_result>" in xml
    assert "my swarm" in xml
    # Check new elapsed attribute
    assert 'elapsed="-"' in xml
    assert 'success="true"' in xml
    assert 'success="false"' in xml
    assert "bad &lt;value&gt;" in xml
    assert "err" in xml


# ---------------------------------------------------------------------------
# Execution with mocked prompt_async
# ---------------------------------------------------------------------------
async def test_successful_parallel_execution(mock_session: MagicMock, monkeypatch):
    async def fake_prompt_async(*, prompt_str, session, output_function, **kwargs):
        output_function(f"result for {prompt_str}", MessageType.Text)

    monkeypatch.setattr("kimix.tools.swarm.utils.prompt_async", fake_prompt_async)
    monkeypatch.setattr(
        "kimix.tools.swarm.utils.close_session_async", AsyncMock()
    )

    fake_sub = SimpleNamespace(id="sub-1", get_custom_config=lambda: {})
    created: list[dict[str, object]] = []

    async def fake_create_session_async(**kwargs):
        created.append(kwargs)
        return fake_sub

    monkeypatch.setattr(
        "kimix.tools.swarm.utils._create_session_async", fake_create_session_async
    )

    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test swarm",
        prompt_template="process {{item}}",
        items=["a", "b"],
    ))

    assert result.is_error is False
    assert "<succeeded>2</succeeded>" in result.output
    assert "result for process a" in result.output
    assert "result for process b" in result.output
    assert len(created) == 2


async def test_failed_subagent_returned_in_order(mock_session: MagicMock, monkeypatch):
    async def fake_prompt_async(*, prompt_str, session, output_function, **kwargs):
        if "a" in prompt_str:
            output_function("ok", MessageType.Text)
        else:
            raise RuntimeError("boom")

    monkeypatch.setattr("kimix.tools.swarm.utils.prompt_async", fake_prompt_async)
    monkeypatch.setattr(
        "kimix.tools.swarm.utils.close_session_async", AsyncMock()
    )

    fake_sub = SimpleNamespace(id="sub-1", get_custom_config=lambda: {})

    async def fake_create_session_async(**kwargs):
        return fake_sub

    monkeypatch.setattr(
        "kimix.tools.swarm.utils._create_session_async", fake_create_session_async
    )

    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test",
        prompt_template="process {{item}}",
        items=["a", "b"],
    ))

    assert "<succeeded>1</succeeded>" in result.output
    assert "<failed>1</failed>" in result.output
    # Verify order is preserved.
    assert result.output.index("index=\"0\"") < result.output.index("index=\"1\"")


# ---------------------------------------------------------------------------
# Resume via resume_agent_ids
# ---------------------------------------------------------------------------
async def test_rate_limit_retry(mock_session: MagicMock, monkeypatch):
    from kimi_agent_sdk import APIStatusError

    calls: list[int] = []

    async def fake_prompt_async(*, prompt_str, session, output_function, **kwargs):
        calls.append(len(calls))
        if len(calls) < 2:
            raise APIStatusError(
                status_code=429,
                message="rate limit exceeded",
            )
        output_function("ok after retry", MessageType.Text)

    monkeypatch.setattr("kimix.tools.swarm.utils.prompt_async", fake_prompt_async)
    monkeypatch.setattr(
        "kimix.tools.swarm.utils.close_session_async", AsyncMock()
    )

    fake_sub = SimpleNamespace(id="sub-1", get_custom_config=lambda: {})

    async def fake_create_session_async(**kwargs):
        return fake_sub

    monkeypatch.setattr(
        "kimix.tools.swarm.utils._create_session_async", fake_create_session_async
    )

    # Patch sleep to keep tests fast.
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test",
        prompt_template="do {{item}}",
        items=["a"],
        resume_agent_ids={"agent-old": "old prompt"},
    ))

    assert result.is_error is False
    # Two sub-agents were dispatched; the first one retried once, so we expect
    # three prompt_async calls in total.
    assert len(calls) == 3
    assert "ok after retry" in result.output


async def test_resume_agent_ids(mock_session: MagicMock, monkeypatch):
    async def fake_prompt_async(*, prompt_str, session, output_function, **kwargs):
        output_function(f"resumed {prompt_str}", MessageType.Text)

    monkeypatch.setattr("kimix.tools.swarm.utils.prompt_async", fake_prompt_async)
    monkeypatch.setattr(
        "kimix.tools.swarm.utils.close_session_async", AsyncMock()
    )

    fake_sub = SimpleNamespace(id="sub-1", get_custom_config=lambda: {})
    created: list[dict[str, object]] = []

    async def fake_create_session_async(**kwargs):
        created.append(kwargs)
        return fake_sub

    monkeypatch.setattr(
        "kimix.tools.swarm.utils._create_session_async", fake_create_session_async
    )

    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="resume test",
        prompt_template="process {{item}}",
        items=["new"],
        resume_agent_ids={"agent-old": "old prompt"},
    ))

    assert result.is_error is False
    assert "<total>2</total>" in result.output
    assert "resumed old prompt" in result.output
    resumed_kwargs = [c for c in created if c.get("session_id") == "agent-old"]
    assert len(resumed_kwargs) == 1
    assert resumed_kwargs[0].get("resume") is True


# ---------------------------------------------------------------------------
# parallel_sample mode (P7 best-of-N) validation
# ---------------------------------------------------------------------------


async def test_swarm_subagent_inherits_parent_work_dir(
    mock_session: MagicMock, monkeypatch
):
    """Swarm sub-agent sessions must inherit the parent session's work_dir so
    they resolve relative paths and AGENTS.md against the same repo."""
    from kaos.path import KaosPath

    fake_prompt = AsyncMock(side_effect=lambda *, prompt_str, session, output_function, **kwargs: output_function(f"ok {prompt_str}", MessageType.Text))
    monkeypatch.setattr("kimix.tools.swarm.utils.prompt_async", fake_prompt)
    monkeypatch.setattr("kimix.tools.swarm.utils.close_session_async", AsyncMock())
    fake_sub = SimpleNamespace(id="sub-wd", get_custom_config=lambda: {})
    created: list[dict[str, object]] = []

    async def fake_create_session_async(**kwargs):
        created.append(kwargs)
        return fake_sub

    monkeypatch.setattr(
        "kimix.tools.swarm.utils._create_session_async", fake_create_session_async
    )

    work_dir = KaosPath(str(Path.cwd()))
    mock_session.work_dir = work_dir
    tool = AgentSwarm(mock_session)
    result = await tool(AgentSwarmParams(
        description="test swarm wd",
        prompt_template="process {{item}}",
        items=["x", "y"],
    ))
    assert result.is_error is False
    assert created, "sub-agent session should have been created"
    assert all(c.get("work_dir") == work_dir for c in created)
