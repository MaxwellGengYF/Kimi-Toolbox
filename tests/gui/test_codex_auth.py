from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import orjson
import pytest

from kimix_gui.codex_auth import (
    AUTH_CONNECTED,
    AUTH_LOGIN_REQUIRED,
    AUTH_RETRY_LATER,
    AUTHORIZATION_URL,
    BROWSER_CALLBACK_HOST,
    BROWSER_OAUTH_SCOPE,
    CODEX_CLIENT_ID,
    CODEX_MODELS_URL,
    DEFAULT_CODEX_MODELS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    PROBLEM_CALLBACK_UNAVAILABLE,
    PROBLEM_CANCELLED,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_RATE_LIMITED,
    PROBLEM_TIMEOUT,
    TOKEN_URL,
    CodexAuthError,
    CodexAuthService,
    CodexBrowserChallenge,
    CodexProblem,
    extract_chatgpt_account_id,
    fallback_catalog,
    parse_model_catalog,
)


def _jwt(*, exp: int = 10_000, account_id: str = "acct-1") -> str:
    payload = base64.urlsafe_b64encode(
        orjson.dumps(
            {
                "exp": exp,
                "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
            }
        )
    ).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def _write_tokens(path: Path, access: str, refresh: str = "refresh-1") -> None:
    path.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": access,
                "refresh_token": refresh,
                "expires_at": 100,
            }
        )
    )


async def _send_browser_callback(
    challenge: CodexBrowserChallenge,
    *,
    code: str | None = "authorization",
    state: str | None = None,
    error: str | None = None,
) -> int:
    authorization = urlsplit(challenge.authorization_url)
    authorization_query = parse_qs(authorization.query)
    redirect = urlsplit(authorization_query["redirect_uri"][0])
    callback_query = {
        "state": authorization_query["state"][0] if state is None else state,
    }
    if code is not None:
        callback_query["code"] = code
    if error is not None:
        callback_query["error"] = error
    reader, writer = await asyncio.open_connection(BROWSER_CALLBACK_HOST, redirect.port)
    target = f"{redirect.path}?{urlencode(callback_query)}"
    writer.write(
        (
            f"GET {target} HTTP/1.1\r\nHost: localhost:{redirect.port}\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
    )
    await writer.drain()
    status_line = await reader.readline()
    await reader.read()
    writer.close()
    await writer.wait_closed()
    return int(status_line.split()[1])


@pytest.mark.asyncio
async def test_browser_flow_uses_pkce_callback_exchanges_and_fetches_models(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, object]] = []
    access = _jwt()

    async def handler(request: httpx.Request) -> httpx.Response:
        body: object = request.content.decode()
        calls.append((request.method, str(request.url), body))
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": access, "refresh_token": "refresh-token"},
            )
        if str(request.url) == CODEX_MODELS_URL:
            assert request.headers["ChatGPT-Account-ID"] == "acct-1"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "gpt-visible",
                            "priority": 3,
                            "supported_in_api": False,
                            "context_window": 272_000,
                            "max_output_tokens": 128_000,
                            "default_reasoning_level": "high",
                            "supported_reasoning_levels": [
                                {"effort": "low"},
                                {"effort": "high"},
                            ],
                        }
                    ]
                },
            )
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
        callback_ports=(0,),
    )
    challenges: list[CodexBrowserChallenge] = []

    async def complete(challenge: CodexBrowserChallenge) -> None:
        challenges.append(challenge)
        assert await _send_browser_callback(challenge) == 200

    snapshot, catalog = await service.login(7, complete)
    await service.aclose()

    assert snapshot.state == AUTH_CONNECTED
    assert [model.slug for model in catalog.models] == ["gpt-visible"]
    assert catalog.models[0].max_context_size == 272_000
    assert catalog.models[0].max_tokens == 128_000
    assert catalog.models[0].reasoning_efforts == ("low", "high")
    assert catalog.models[0].default_reasoning_effort == "high"
    authorization = urlsplit(challenges[0].authorization_url)
    authorization_query = parse_qs(authorization.query)
    assert (
        f"{authorization.scheme}://{authorization.netloc}{authorization.path}" == AUTHORIZATION_URL
    )
    assert authorization_query["response_type"] == ["code"]
    assert authorization_query["client_id"] == [CODEX_CLIENT_ID]
    assert authorization_query["scope"] == [BROWSER_OAUTH_SCOPE]
    assert authorization_query["code_challenge_method"] == ["S256"]
    assert authorization_query["id_token_add_organizations"] == ["true"]
    assert authorization_query["codex_cli_simplified_flow"] == ["true"]
    assert authorization_query["originator"] == ["kimix-gui"]
    token_call = next(call for call in calls if call[1] == TOKEN_URL)
    token_form = parse_qs(str(token_call[2]))
    assert token_form["grant_type"] == ["authorization_code"]
    assert token_form["client_id"] == [CODEX_CLIENT_ID]
    assert token_form["code"] == ["authorization"]
    assert token_form["redirect_uri"] == authorization_query["redirect_uri"]
    verifier = token_form["code_verifier"][0]
    expected_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert authorization_query["code_challenge"] == [expected_challenge]
    assert "state=" not in repr(challenges[0])
    assert "code_challenge=" not in repr(challenges[0])
    persisted = (tmp_path / "auth.json").read_text(encoding="utf-8")
    assert "authorization" not in persisted
    assert verifier not in persisted
    assert authorization_query["state"][0] not in persisted


@pytest.mark.asyncio
async def test_browser_callback_rejects_wrong_state_then_accepts_expected_state(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": _jwt(), "refresh_token": "refresh"},
            )
        if str(request.url) == CODEX_MODELS_URL:
            return httpx.Response(200, json={"models": [{"slug": "model"}]})
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )

    async def complete(challenge: CodexBrowserChallenge) -> None:
        assert await _send_browser_callback(challenge, state="wrong-state") == 400
        assert await _send_browser_callback(challenge) == 200

    snapshot, _catalog = await service.login(8, complete)
    await service.aclose()

    assert snapshot.state == AUTH_CONNECTED


@pytest.mark.asyncio
async def test_browser_callback_access_denied_is_a_cancelled_login(tmp_path: Path) -> None:
    service = CodexAuthService(tmp_path / "auth.json", callback_ports=(0,))

    async def deny(challenge: CodexBrowserChallenge) -> None:
        assert await _send_browser_callback(challenge, code=None, error="access_denied") == 400

    with pytest.raises(CodexAuthError) as caught:
        await service.login(9, deny)
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CANCELLED
    assert not (tmp_path / "auth.json").exists()


@pytest.mark.asyncio
async def test_browser_flow_can_be_cancelled_from_challenge(tmp_path: Path) -> None:
    service = CodexAuthService(tmp_path / "auth.json", callback_ports=(0,))

    def cancel(challenge: CodexBrowserChallenge) -> None:
        service.cancel_login(challenge.operation_id)

    with pytest.raises(CodexAuthError) as caught:
        await service.login(4, cancel)
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CANCELLED


@pytest.mark.asyncio
async def test_browser_flow_cancel_wakes_an_active_callback_wait(tmp_path: Path) -> None:
    challenge_ready = asyncio.Event()
    service = CodexAuthService(tmp_path / "auth.json", callback_ports=(0,))

    def published(_challenge: CodexBrowserChallenge) -> None:
        challenge_ready.set()

    task = asyncio.create_task(service.login(10, published))
    await challenge_ready.wait()
    service.cancel_login(10)

    with pytest.raises(CodexAuthError) as caught:
        await asyncio.wait_for(task, timeout=1)
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CANCELLED


@pytest.mark.asyncio
async def test_cancel_during_model_refresh_removes_just_created_credentials(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    model_request_started = asyncio.Event()
    release_models = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": _jwt(), "refresh_token": "refresh"},
            )
        if str(request.url) == CODEX_MODELS_URL:
            model_request_started.set()
            await release_models.wait()
            return httpx.Response(200, json={"models": [{"slug": "model"}]})
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    task = asyncio.create_task(service.login(8, _send_browser_callback))
    await model_request_started.wait()
    service.cancel_login(8)
    release_models.set()

    with pytest.raises(CodexAuthError) as caught:
        await task
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CANCELLED
    assert not auth_file.exists()


@pytest.mark.asyncio
async def test_browser_flow_times_out_while_waiting_for_callback(tmp_path: Path) -> None:
    service = CodexAuthService(
        tmp_path / "auth.json",
        callback_ports=(0,),
        login_timeout=0.01,
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.login(5, lambda _challenge: None)
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_TIMEOUT


@pytest.mark.asyncio
async def test_browser_flow_reports_when_local_callback_ports_are_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attempted_ports: list[int] = []

    async def unavailable(_callback, _host: str, port: int, **_kwargs):
        attempted_ports.append(port)
        raise OSError("address in use")

    monkeypatch.setattr(asyncio, "start_server", unavailable)

    service = CodexAuthService(
        tmp_path / "auth.json",
        callback_ports=(1455, 1457),
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.login(6, lambda _challenge: None)
    snapshot = await service.snapshot()
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CALLBACK_UNAVAILABLE
    assert attempted_ports == [1455, 1457]
    assert snapshot.state == AUTH_RETRY_LATER


@pytest.mark.asyncio
async def test_token_exchange_401_requires_login_and_clears_old_tokens(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_tokens(auth_file, _jwt())

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(401, json={"error": "unauthorized"})
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.login(7, _send_browser_callback)
    snapshot = await service.snapshot()
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_LOGIN_REQUIRED
    assert snapshot.state == AUTH_LOGIN_REQUIRED
    state = orjson.loads(auth_file.read_bytes())
    assert "access_token" not in state
    assert "refresh_token" not in state


@pytest.mark.asyncio
async def test_refresh_rotates_token_and_a_second_service_adopts_it(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    old_access = _jwt(exp=100, account_id="old-account")
    new_access = _jwt(exp=20_000, account_id="new-account")
    _write_tokens(auth_file, old_access)
    refresh_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_calls
        assert str(request.url) == TOKEN_URL
        refresh_calls += 1
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={"access_token": new_access, "refresh_token": "refresh-2"},
        )

    transport = httpx.MockTransport(handler)
    first = CodexAuthService(auth_file, transport=transport, clock=lambda: 1_000)
    second = CodexAuthService(auth_file, transport=transport, clock=lambda: 1_000)
    credentials = await asyncio.gather(
        first.ensure_credentials(force_refresh=True, failed_access_token=old_access),
        second.ensure_credentials(force_refresh=True, failed_access_token=old_access),
    )
    await first.aclose()
    await second.aclose()

    assert refresh_calls == 1
    assert {item.account_id for item in credentials} == {"new-account"}
    state = orjson.loads(auth_file.read_bytes())
    assert state["refresh_token"] == "refresh-2"


@pytest.mark.asyncio
async def test_terminal_refresh_error_clears_tokens_but_retains_catalog(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_tokens(auth_file, _jwt(exp=100))
    state = orjson.loads(auth_file.read_bytes())
    state["models"] = [{"slug": "cached", "max_context_size": 200_000}]
    auth_file.write_bytes(orjson.dumps(state))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.ensure_credentials(force_refresh=True)
    snapshot = await service.snapshot()
    catalog = await service.catalog()
    await service.aclose()

    assert caught.value.problem.code == "invalid_grant"
    assert snapshot.state == AUTH_LOGIN_REQUIRED
    assert [model.slug for model in catalog.models] == ["cached"]
    persisted = auth_file.read_text(encoding="utf-8")
    assert "access_token" not in persisted
    assert "refresh_token" not in persisted


@pytest.mark.asyncio
async def test_temporary_refresh_error_keeps_credentials(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    access = _jwt(exp=100)
    _write_tokens(auth_file, access)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    with pytest.raises(CodexAuthError):
        await service.ensure_credentials(force_refresh=True)
    await service.aclose()

    state = orjson.loads(auth_file.read_bytes())
    assert state["access_token"] == access
    assert state["refresh_token"] == "refresh-1"


@pytest.mark.asyncio
async def test_refresh_rate_limit_keeps_credentials_and_retry_hint(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    access = _jwt(exp=100)
    _write_tokens(auth_file, access)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "9"})

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.ensure_credentials(force_refresh=True)
    await service.aclose()

    state = orjson.loads(auth_file.read_bytes())
    assert caught.value.problem.code == PROBLEM_RATE_LIMITED
    assert caught.value.problem.retry_after == 9
    assert state["access_token"] == access
    assert state["refresh_token"] == "refresh-1"


@pytest.mark.asyncio
async def test_model_network_failure_uses_cache_then_exact_fallback(tmp_path: Path) -> None:
    async def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    auth_file = tmp_path / "auth.json"
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": _jwt(),
                "refresh_token": "refresh",
                "expires_at": 10_000,
                "models": [{"slug": "cached-account-model", "priority": 1}],
            }
        )
    )
    cached_service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(offline),
        clock=lambda: 1_000,
    )
    cached = await cached_service.refresh_models(12)
    await cached_service.aclose()

    state = orjson.loads(auth_file.read_bytes())
    state.pop("models")
    auth_file.write_bytes(orjson.dumps(state))
    fallback_service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(offline),
        clock=lambda: 1_000,
    )
    fallback = await fallback_service.refresh_models(13)
    await fallback_service.aclose()

    assert cached.stale is True
    assert [model.slug for model in cached.models] == ["cached-account-model"]
    assert [model.slug for model in fallback.models] == list(DEFAULT_CODEX_MODELS)


@pytest.mark.asyncio
async def test_inflight_catalog_refresh_does_not_recreate_disconnected_store(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": _jwt(),
                "refresh_token": "refresh",
                "expires_at": 10_000,
            }
        )
    )
    request_started = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CODEX_MODELS_URL
        request_started.set()
        await release_response.wait()
        return httpx.Response(200, json={"models": [{"slug": "late-model"}]})

    refreshing = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    disconnecting = CodexAuthService(auth_file, clock=lambda: 1_000)
    task = asyncio.create_task(refreshing.refresh_models(14))
    await request_started.wait()

    await disconnecting.disconnect(15)
    release_response.set()
    catalog = await task
    await refreshing.aclose()
    await disconnecting.aclose()

    assert not auth_file.exists()
    assert catalog.stale is True
    assert [model.slug for model in catalog.models] == list(DEFAULT_CODEX_MODELS)


@pytest.mark.asyncio
async def test_login_required_problem_overrides_stored_access_token(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": _jwt(),
                "refresh_token": "refresh",
                "expires_at": 10_000,
                "last_error": {"code": PROBLEM_LOGIN_REQUIRED},
            }
        )
    )
    service = CodexAuthService(auth_file, clock=lambda: 1_000)

    snapshot = await service.snapshot()
    await service.disconnect()

    assert snapshot.state == AUTH_LOGIN_REQUIRED
    assert not auth_file.exists()


def test_model_catalog_filters_hidden_sorts_deduplicates_and_keeps_codex_only() -> None:
    models = parse_model_catalog(
        {
            "models": [
                {"slug": "z", "priority": 2, "supported_in_api": False},
                {"slug": "hidden", "visibility": "hidden", "priority": 0},
                {
                    "slug": "a",
                    "priority": 1,
                    "input_modalities": ["text", "image"],
                    "default_reasoning_level": "minimal",
                    "supported_reasoning_levels": [
                        {"effort": "minimal"},
                        {"effort": "xhigh"},
                        {"effort": "future-effort"},
                    ],
                },
                {"slug": "a", "priority": 9},
            ]
        }
    )

    assert [model.slug for model in models] == ["a", "z"]
    assert models[0].input_modalities == ("text", "image")
    assert models[0].reasoning_efforts == ("minimal", "xhigh", "future-effort")
    assert models[0].default_reasoning_effort == "minimal"
    assert models[1].max_context_size == DEFAULT_CONTEXT_WINDOW
    assert models[1].max_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert models[1].reasoning_efforts == ()
    assert models[1].default_reasoning_effort is None
    assert all("-900k" not in model.slug for model in models)


def test_fallback_catalog_uses_official_codex_runtime_profiles() -> None:
    models = {model.slug: model for model in fallback_catalog().models}

    assert models["gpt-5.6-sol"].max_context_size == 272_000
    assert models["gpt-5.6-sol"].max_tokens == 128_000
    assert models["gpt-5.6-sol"].default_reasoning_effort == "low"
    assert models["gpt-5.6-sol"].reasoning_efforts == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    )
    assert models["gpt-5.6-terra"].default_reasoning_effort == "medium"
    assert models["gpt-5.3-codex-spark"].default_reasoning_effort == "high"
    assert models["gpt-5.3-codex-spark"].input_modalities == ("text",)


@pytest.mark.asyncio
async def test_legacy_cached_profile_is_rehydrated_with_official_metadata(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "max_context_size": 200_000,
                        "reasoning_efforts": ["low", "medium", "high"],
                    }
                ],
            }
        )
    )
    service = CodexAuthService(auth_file)

    catalog = await service.catalog()
    await service.aclose()

    model = catalog.models[0]
    assert model.max_context_size == 272_000
    assert model.max_tokens == 128_000
    assert model.default_reasoning_effort == "low"
    assert model.reasoning_efforts[-2:] == ("max", "ultra")


def test_fallback_is_exact_and_secret_values_are_not_represented() -> None:
    problem = CodexProblem(PROBLEM_LOGIN_REQUIRED)
    service_error = CodexAuthError(problem)

    assert DEFAULT_CODEX_MODELS == (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
    )
    assert "token" not in repr(service_error).lower()
    assert extract_chatgpt_account_id(_jwt(account_id="account-x")) == "account-x"
