"""Tests for Kimix's managed OpenAI Codex provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import orjson
import pytest
from kosong.chat_provider.codex import OpenAICodex
from kosong.message import Message
from kosong.tooling import Tool

from kimi_cli.auth.codex import (
    CodexRequestAuth,
    CodexRuntimeCredentials,
)
from kimi_cli.codex_context import (
    CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS,
    CODEX_AUTO_COMPACT_PERCENT,
    CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
    CODEX_LOOP_CONTROL_KEYS,
    codex_loop_control,
    codex_trigger_point,
)
from kimi_cli.soul.compaction import should_auto_compact


@dataclass
class _CredentialFixture:
    calls: list[tuple[bool, str | None]]

    async def ensure_credentials(
        self,
        *,
        force_refresh: bool = False,
        rejected_credentials: CodexRuntimeCredentials | None = None,
    ) -> CodexRuntimeCredentials:
        rejected_token = (
            rejected_credentials.access_token
            if rejected_credentials is not None
            else None
        )
        self.calls.append((force_refresh, rejected_token))
        suffix = len(self.calls)
        return CodexRuntimeCredentials(
            f"token-{suffix}",
            f"account-{suffix}",
            None,
            f"generation-{suffix}",
        )

    async def invalidate_credentials(
        self,
        rejected_credentials: CodexRuntimeCredentials,
    ) -> None:
        self.invalidated_access_token = rejected_credentials.access_token


@pytest.mark.asyncio
async def test_http_auth_overrides_static_token_and_replays_only_one_401() -> None:
    seen: list[tuple[str, str, bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.headers["Authorization"],
                request.headers["ChatGPT-Account-ID"],
                request.content,
            )
        )
        return httpx.Response(401 if len(seen) == 1 else 200, content=b"ok")

    service = _CredentialFixture([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(
            service.ensure_credentials,
            service.invalidate_credentials,
        ),
        headers={"Authorization": "Bearer oauth-managed"},
    ) as client:
        response = await client.post(
            "https://chatgpt.com/backend-api/codex/responses", content=b"x"
        )

    assert response.status_code == 200
    assert not hasattr(service, "invalidated_access_token")
    assert seen == [
        ("Bearer token-1", "account-1", b"x"),
        ("Bearer token-2", "account-2", b"x"),
    ]
    assert service.calls == [(False, None), (True, "token-1")]


class _ProbeStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.read = False

    async def __aiter__(self):
        self.read = True
        yield b"event: response.output_text.delta\ndata: {}\n\n"


@pytest.mark.asyncio
async def test_successful_stream_is_not_pre_read_by_auth() -> None:
    stream = _ProbeStream()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    service = _CredentialFixture([])
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=CodexRequestAuth(
                service.ensure_credentials,
                service.invalidate_credentials,
            ),
        ) as client,
        client.stream("POST", "https://example.test", content=b"body") as response,
    ):
        assert response.status_code == 200
        assert stream.read is False
        await response.aread()

    assert stream.read is True


@pytest.mark.asyncio
async def test_second_401_is_returned_without_another_auth_replay() -> None:
    requests = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401)

    service = _CredentialFixture([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(
            service.ensure_credentials,
            service.invalidate_credentials,
        ),
    ) as client:
        response = await client.post("https://example.test", content=b"replayable")

    assert response.status_code == 401
    assert requests == 2
    assert len(service.calls) == 2
    assert service.invalidated_access_token == "token-2"


@pytest.mark.asyncio
async def test_each_request_resolves_fresh_credentials() -> None:
    authorizations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["Authorization"])
        return httpx.Response(200)

    service = _CredentialFixture([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(
            service.ensure_credentials,
            service.invalidate_credentials,
        ),
    ) as client:
        await client.get("https://example.test/one")
        await client.get("https://example.test/two")

    assert authorizations == ["Bearer token-1", "Bearer token-2"]
    assert service.calls == [(False, None), (False, None)]


@pytest.mark.asyncio
async def test_provider_uses_the_official_codex_responses_contract() -> None:
    bodies: list[dict[str, Any]] = []
    headers: list[httpx.Headers] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(orjson.loads(request.content))
        headers.append(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6-terra",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICodex(
        session_id="gui_session",
        model="gpt-5.6-terra",
        api_key="oauth-managed",
        base_url="https://chatgpt.com/backend-api/codex",
        http_client=http_client,
        stream=False,
    ).with_generation_kwargs(
        user="must-not-reach-codex",
        max_output_tokens=128_000,
        reasoning_effort="medium",
    )
    try:
        await provider.generate(
            "Follow the project instructions.",
            [
                Tool(
                    name="read_file",
                    description="Read one file.",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ],
            [Message(role="user", content="Hello")],
        )
    finally:
        await provider.shutdown()

    assert len(bodies) == 1
    assert bodies[0] == {
        "include": ["reasoning.encrypted_content"],
        "input": [
            {
                "content": [{"text": "Hello", "type": "input_text"}],
                "role": "user",
                "type": "message",
            }
        ],
        "instructions": "Follow the project instructions.",
        "model": "gpt-5.6-terra",
        "parallel_tool_calls": True,
        "prompt_cache_key": "gui_session",
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False,
        "stream": False,
        "tool_choice": "auto",
        "tools": [
            {
                "description": "Read one file.",
                "name": "read_file",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "type": "object",
                },
                "strict": False,
                "type": "function",
            }
        ],
    }
    assert headers[0]["session-id"] == "gui_session"
    assert headers[0]["thread-id"] == "gui_session"
    assert headers[0]["session_id"] == "gui_session"
    assert headers[0]["x-client-request-id"] == "gui_session"


@pytest.mark.asyncio
async def test_provider_maps_sequential_kimix_mode_to_official_parallel_flag() -> None:
    bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(orjson.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6-terra",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICodex(
        session_id="gui_session",
        model="gpt-5.6-terra",
        api_key="oauth-managed",
        base_url="https://chatgpt.com/backend-api/codex",
        http_client=http_client,
        stream=False,
    ).with_parallel_tool_calls(enabled=False)
    try:
        await provider.generate("Instructions", [], [Message(role="user", content="Hello")])
    finally:
        await provider.shutdown()

    assert bodies[0]["parallel_tool_calls"] is False
    assert "max_tool_calls" not in bodies[0]


@pytest.mark.asyncio
async def test_child_close_keeps_shared_transport_until_top_level_shutdown() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenAICodex(
        session_id="session-id",
        model="model",
        api_key="oauth-managed",
        http_client=http_client,
        own_http_client=False,
    )

    await provider.aclose()
    assert http_client.is_closed is False

    await provider.shutdown()
    assert http_client.is_closed is True


# ── Codex-parity context accounting ─────────────────────────────────────────


def _codex_trigger_point(max_context_size: int) -> int:
    """The token count at which openai/codex forces compaction.

    Recomputed here from the mirrored constants rather than calling
    ``codex_trigger_point``, so the shipped helper is checked against an
    independent transcription of ``codex-rs/core/src/session/context_window.rs``::

        token_limit_reached = scope_tokens >= auto_compact_token_limit + fallback_buffer
                           || active_tokens >= context_window * effective_percent / 100
    """
    usable = max_context_size * CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT // 100
    buffered = (
        max_context_size * CODEX_AUTO_COMPACT_PERCENT // 100
        + CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS
    )
    return min(usable, buffered)


def _kimix_trigger_point(max_context_size: int, loop_control: dict[str, Any], **kwargs: Any) -> int:
    """First token count for which ``should_auto_compact`` returns True."""
    call = dict(
        trigger_ratio=loop_control["compaction_trigger_ratio"],
        reserved_context_size=loop_control["reserved_context_size"],
        **kwargs,
    )
    low, high = 0, max_context_size
    while low < high:
        mid = (low + high) // 2
        if should_auto_compact(mid, max_context_size, **call):
            high = mid
        else:
            low = mid + 1
    return low


def test_codex_loop_control_matches_codex_thresholds() -> None:
    loop_control = codex_loop_control(272_000)

    # auto_compact_token_limit (244_800) + fallback buffer (16_384).
    assert loop_control["reserved_context_size"] == 261_184
    assert loop_control["compaction_trigger_ratio"] == 0.95
    # Codex reminds at 6_144 remaining tokens against the 90% limit.
    assert loop_control["compact_reminder_threshold"] == pytest.approx(238_656 / 272_000)


def test_codex_loop_control_neutralizes_the_output_reservation() -> None:
    """``max_tokens`` and the tool-output buffer must not move the trigger point.

    The Codex backend rejects explicit output-token limits, so reserving
    ``max_tokens`` would shrink the usable window to protect nothing.
    """
    max_context_size = 272_000
    loop_control = codex_loop_control(max_context_size)
    expected = _codex_trigger_point(max_context_size)
    assert expected == 258_400

    for max_tokens in (None, 16_000, 64_000, 128_000, 384_000):
        for tool_buffer in (0, 32_768, 100_000):
            assert (
                _kimix_trigger_point(
                    max_context_size,
                    loop_control,
                    max_tokens=max_tokens,
                    tool_call_buffer_tokens=tool_buffer,
                )
                == expected
            )


@pytest.mark.parametrize("max_context_size", [128_000, 272_000, 400_000, 1_000_000])
def test_codex_loop_control_parity_across_window_sizes(max_context_size: int) -> None:
    loop_control = codex_loop_control(max_context_size)
    assert (
        _kimix_trigger_point(
            max_context_size,
            loop_control,
            max_tokens=max_context_size // 4,
            tool_call_buffer_tokens=32_768,
        )
        == _codex_trigger_point(max_context_size)
    )


def test_codex_loop_control_is_a_valid_loop_control_payload() -> None:
    from kimi_cli.config import LoopControl

    resolved = LoopControl.model_validate(codex_loop_control(272_000))
    assert resolved.reserved_context_size == 261_184
    assert resolved.compaction_trigger_ratio == 0.95
    # Pruning must still run strictly before compaction.
    assert resolved.prune_trigger_ratio < resolved.compaction_trigger_ratio
    assert resolved.compact_reminder_threshold < resolved.compaction_trigger_ratio


def test_codex_loop_control_skipped_for_unknown_window() -> None:
    assert codex_loop_control(0) == {}
    assert codex_loop_control(-1) == {}


@pytest.mark.parametrize("max_context_size", [128_000, 272_000, 400_000, 1_000_000])
def test_codex_trigger_point_matches_the_reference_transcription(max_context_size: int) -> None:
    assert codex_trigger_point(max_context_size) == _codex_trigger_point(max_context_size)


def test_codex_loop_control_only_touches_declared_keys() -> None:
    """The advertised key set must match what the helper actually returns."""

    assert set(codex_loop_control(272_000)) == set(CODEX_LOOP_CONTROL_KEYS)
