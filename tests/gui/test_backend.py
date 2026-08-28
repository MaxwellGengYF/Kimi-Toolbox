from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, cast

import orjson
import pytest
from kimi_cli.auth.codex import CODEX_BASE_URL, CodexAuthService

from kimix_gui import backend
from kimix_gui.backend import (
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
    new_session_id,
)
from kimix_gui.llm import (
    ProviderFileTarget,
    chatgpt_selection,
    configured_selection,
    default_provider_file_path,
)


def test_session_options_are_standalone_values(tmp_path: Path) -> None:
    selection = configured_selection(ProviderFileTarget(tmp_path / "provider.json", "kimi"))
    options = SessionOptions(work_dir=tmp_path, llm_selection=selection)

    assert options.work_dir == tmp_path
    assert options.llm_selection == selection
    assert options.yolo is False


def test_new_session_id_is_compact_and_unique() -> None:
    first = new_session_id()
    second = new_session_id()

    assert first.startswith("gui_")
    assert len(first) == 16
    assert first != second


@pytest.mark.asyncio
async def test_session_factory_passes_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "secret",
            }
        ),
        encoding="utf-8",
    )
    created = object()
    received: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(backend, "create_session_async", fake_create)

    result = await create_sdk_session(
        SessionOptions(
            tmp_path,
            llm_selection=configured_selection(ProviderFileTarget(config_file, "override-model")),
        )
    )

    assert result is created
    provider_dict = received["provider_dict"]
    assert isinstance(provider_dict, dict)
    assert provider_dict["model"] == "override-model"
    assert provider_dict["url"] == "https://example.test/v1"
    assert received["resume"] is False
    assert received["model"] == "override-model"


@pytest.mark.asyncio
async def test_session_factory_resumes_through_kimix_worker_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = object()
    received: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(backend, "create_session_async", fake_create)

    result = await create_sdk_session(
        SessionOptions(
            tmp_path,
            session_id="existing-session",
            llm_selection=configured_selection(
                ProviderFileTarget(default_provider_file_path(), "model-override")
            ),
            yolo=True,
        )
    )

    assert result is created
    assert received["session_id"] == "existing-session"
    assert received["resume"] is True
    assert received["model"] == "model-override"
    assert "thinking" not in received
    assert received["yolo"] is True


@pytest.mark.asyncio
async def test_session_factory_loads_worker_execution_tools(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 131_072,
                "capabilities": [],
                "type": "openai_legacy",
                "url": "http://127.0.0.1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )

    session = await create_sdk_session(
        SessionOptions(
            tmp_path,
            llm_selection=configured_selection(ProviderFileTarget(config_file)),
        )
    )
    try:
        runtime_session = cast(Any, session)
        tool_names = {tool.name for tool in runtime_session._cli.soul.agent.toolset.tools}

        assert "python" in tool_names
        assert "job_output" in tool_names
        assert {"bash", "pwsh"} & tool_names
        assert session.status.yolo_enabled is False
    finally:
        await close_sdk_session(session)


@pytest.mark.asyncio
async def test_chatgpt_session_uses_managed_codex_provider_without_persisting_token(
    tmp_path: Path,
) -> None:
    payload = base64.urlsafe_b64encode(
        orjson.dumps(
            {
                "exp": int(time.time()) + 3_600,
                "https://api.openai.com/auth": {"chatgpt_account_id": "account-test"},
            }
        )
    ).rstrip(b"=")
    access_token = f"header.{payload.decode()}.signature"
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": access_token,
                "refresh_token": "refresh-secret",
                "expires_at": time.time() + 3_600,
                "account_id": "account-test",
                "credential_id": "credential-test",
                "models": [
                    {
                        "slug": "gpt-test-codex",
                        "max_context_size": 272_000,
                        "max_tokens": 128_000,
                        "input_modalities": ["text", "image"],
                        "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                        "default_reasoning_effort": "medium",
                    }
                ],
                "models_account_id": "account-test",
                "models_credential_id": "credential-test",
            }
        )
    )
    service = CodexAuthService(auth_file)

    session = await create_sdk_session(
        SessionOptions(
            tmp_path,
            llm_selection=chatgpt_selection("gpt-test-codex", "medium"),
        ),
        codex_service=service,
    )
    try:
        runtime_session = cast(Any, session)
        inner = runtime_session._session
        custom = inner.get_custom_config()
        provider = inner._cli.soul.runtime.llm.chat_provider

        assert provider.name == "openai-codex"
        assert str(provider.client.base_url).rstrip("/") == CODEX_BASE_URL
        assert "user" not in provider._generation_kwargs
        assert provider._session_id == session.id
        assert custom["provider_dict"]["api_key"] == "oauth-managed"
        assert custom["provider_dict"]["capabilities"] == ["thinking", "image_in"]
        assert custom["provider_dict"]["max_tokens"] == 128_000
        assert custom["provider_dict"]["thinking_effort"] == "medium"
        assert access_token not in repr(custom["provider_dict"])
        assert inner._cli.soul.runtime.llm.max_context_size == 272_000
        assert "max_output_tokens" not in provider._generation_kwargs
        assert provider._generation_kwargs["reasoning_effort"] == "medium"
    finally:
        await close_sdk_session(session)
        await service.aclose()
