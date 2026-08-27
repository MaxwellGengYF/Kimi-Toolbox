from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import httpx
import orjson
import pytest
from kosong.message import Message
from kosong.tooling import Tool

from kimix_gui.codex_auth import CodexModel, CodexModelCatalog, CodexRuntimeCredentials
from kimix_gui.codex_provider import (
    CodexProviderLease,
    CodexRequestAuth,
    ManagedOpenAICodex,
    create_codex_provider,
)


@dataclass
class _CredentialService:
    calls: list[tuple[bool, str | None]]

    async def ensure_credentials(
        self,
        *,
        force_refresh: bool = False,
        failed_access_token: str | None = None,
    ) -> CodexRuntimeCredentials:
        self.calls.append((force_refresh, failed_access_token))
        suffix = len(self.calls)
        return CodexRuntimeCredentials(f"token-{suffix}", f"account-{suffix}", None)


@dataclass
class _CatalogService(_CredentialService):
    model: CodexModel

    async def catalog(self) -> CodexModelCatalog:
        return CodexModelCatalog(0, (self.model,), False)


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

    service = _CredentialService([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(service),  # type: ignore[arg-type]
        headers={"Authorization": "Bearer oauth-managed"},
    ) as client:
        response = await client.post(
            "https://chatgpt.com/backend-api/codex/responses", content=b"x"
        )

    assert response.status_code == 200
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

    service = _CredentialService([])
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=CodexRequestAuth(service),  # type: ignore[arg-type]
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

    service = _CredentialService([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(service),  # type: ignore[arg-type]
    ) as client:
        response = await client.post("https://example.test", content=b"replayable")

    assert response.status_code == 401
    assert requests == 2
    assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_each_request_resolves_fresh_credentials() -> None:
    authorizations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["Authorization"])
        return httpx.Response(200)

    service = _CredentialService([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=CodexRequestAuth(service),  # type: ignore[arg-type]
    ) as client:
        await client.get("https://example.test/one")
        await client.get("https://example.test/two")

    assert authorizations == ["Bearer token-1", "Bearer token-2"]
    assert service.calls == [(False, None), (False, None)]


@pytest.mark.asyncio
async def test_provider_keeps_output_limit_as_metadata_and_uses_default_effort() -> None:
    model = CodexModel(
        "gpt-test",
        max_context_size=272_000,
        max_tokens=128_000,
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_reasoning_effort="medium",
    )
    service = _CatalogService([], model)

    runtime = await create_codex_provider(
        cast(Any, service),
        model_name=model.slug,
        session_id="session-id",
        thinking=False,
    )
    try:
        assert runtime.provider._generation_kwargs == {
            "reasoning_effort": "medium",
        }
        assert runtime.provider._session_id == "session-id"
        assert runtime.provider_dict["max_context_size"] == 272_000
        assert "max_tokens" not in runtime.provider_dict
        assert runtime.model.max_tokens == 128_000
        assert runtime.provider_dict["thinking_effort"] == "medium"
        assert runtime.provider_dict["supported_efforts"] == [
            "low",
            "medium",
            "high",
            "xhigh",
        ]
    finally:
        await runtime.lease.close()


@pytest.mark.asyncio
async def test_provider_maps_official_ultra_mode_to_max_wire_effort() -> None:
    model = CodexModel(
        "gpt-test",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max", "ultra"),
        default_reasoning_effort="low",
    )
    service = _CatalogService([], model)

    runtime = await create_codex_provider(
        cast(Any, service),
        model_name=model.slug,
        session_id="session-id",
        thinking=True,
    )
    try:
        assert runtime.provider._generation_kwargs["reasoning_effort"] == "max"
        assert runtime.provider_dict["thinking_effort"] == "max"
        assert runtime.provider_dict["supported_efforts"][-1] == "max"
        assert "ultra" not in runtime.provider_dict["supported_efforts"]
    finally:
        await runtime.lease.close()


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
    provider = ManagedOpenAICodex(
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
    provider = ManagedOpenAICodex(
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
async def test_provider_lease_closes_shared_transport_exactly_once() -> None:
    class Provider:
        closes = 0

        async def shutdown(self) -> None:
            self.closes += 1

    provider = Provider()
    lease = CodexProviderLease(cast(Any, provider))

    await lease.close()
    await lease.close()

    assert provider.closes == 1
    assert lease.closed is True


@pytest.mark.asyncio
async def test_child_close_keeps_shared_transport_until_top_level_shutdown() -> None:
    http_client = httpx.AsyncClient()
    provider = ManagedOpenAICodex(
        session_id="session-id",
        model="model",
        api_key="oauth-managed",
        http_client=http_client,
    )

    await provider.aclose()
    assert http_client.is_closed is False

    await provider.shutdown()
    assert http_client.is_closed is True
