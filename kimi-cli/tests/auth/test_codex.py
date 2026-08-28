"""Kimix core ChatGPT Codex OAuth behavior."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import orjson
import pybase64
import pytest

from kimi_cli.auth.codex import (
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
    PROBLEM_CREDENTIAL_STORE_UNAVAILABLE,
    PROBLEM_INVALID_RESPONSE,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_LOGIN_SUPERSEDED,
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
    payload = pybase64.urlsafe_b64encode(
        orjson.dumps(
            {
                "exp": exp,
                "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
            }
        )
    ).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def _write_tokens(
    path: Path,
    access: str,
    refresh: str = "refresh-1",
    *,
    credential_id: str = "credential-1",
) -> None:
    path.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": access,
                "refresh_token": refresh,
                "expires_at": 100,
                "account_id": extract_chatgpt_account_id(access),
                "credential_id": credential_id,
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
    access = _jwt(account_id="access-token-account")

    async def handler(request: httpx.Request) -> httpx.Response:
        body: object = request.content.decode()
        calls.append((request.method, str(request.url), body))
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="acct-1"),
                    "access_token": access,
                    "refresh_token": "refresh-token",
                },
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
    assert authorization_query["originator"] == ["kimix"]
    token_call = next(call for call in calls if call[1] == TOKEN_URL)
    token_form = parse_qs(str(token_call[2]))
    assert token_form["grant_type"] == ["authorization_code"]
    assert token_form["client_id"] == [CODEX_CLIENT_ID]
    assert token_form["code"] == ["authorization"]
    assert token_form["redirect_uri"] == authorization_query["redirect_uri"]
    verifier = token_form["code_verifier"][0]
    expected_challenge = (
        pybase64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
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
                json={
                    "id_token": _jwt(),
                    "access_token": _jwt(),
                    "refresh_token": "refresh",
                },
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
        query = parse_qs(urlsplit(challenge.authorization_url).query)
        expected_state = query["state"][0]
        assert await _send_browser_callback(challenge, state="wrong-state") == 400
        assert await _send_browser_callback(challenge, state=f"{expected_state}.unexpected") == 400
        assert (
            await _send_browser_callback(
                challenge,
                state=f"{expected_state}.onboarding_entrypoint=life_sciences",
            )
            == 200
        )

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
                json={
                    "id_token": _jwt(),
                    "access_token": _jwt(),
                    "refresh_token": "refresh",
                },
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
async def test_failed_new_code_exchange_preserves_existing_credentials(tmp_path: Path) -> None:
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
    assert snapshot.state == AUTH_CONNECTED
    state = orjson.loads(auth_file.read_bytes())
    assert state["access_token"] == _jwt()
    assert state["refresh_token"] == "refresh-1"


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
        assert request.headers["Content-Type"] == "application/json"
        assert orjson.loads(request.content) == {
            "client_id": CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": "refresh-1",
        }
        refresh_calls += 1
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={
                "id_token": _jwt(exp=20_000, account_id="new-account"),
                "access_token": new_access,
                "refresh_token": "refresh-2",
            },
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
    assert state["credential_id"] != "credential-1"


@pytest.mark.asyncio
async def test_refresh_response_requires_a_new_access_token(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    old_access = _jwt(exp=100)
    _write_tokens(auth_file, old_access)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        return httpx.Response(
            200,
            json={"refresh_token": "rotated-refresh", "expires_in": 3_600},
        )

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.ensure_credentials(force_refresh=True)
    await service.aclose()

    state = orjson.loads(auth_file.read_bytes())
    assert caught.value.problem.code == PROBLEM_INVALID_RESPONSE
    assert state["access_token"] == old_access
    assert state["refresh_token"] == "refresh-1"


@pytest.mark.parametrize(
    "backend_code",
    [
        "invalid_grant",
        "refresh_token_expired",
        "REFRESH_TOKEN_INVALIDATED",
        "refresh_token_reused",
    ],
)
@pytest.mark.asyncio
async def test_terminal_refresh_error_clears_tokens_but_retains_catalog(
    tmp_path: Path,
    backend_code: str,
) -> None:
    auth_file = tmp_path / "auth.json"
    _write_tokens(auth_file, _jwt(exp=100))
    state = orjson.loads(auth_file.read_bytes())
    state["models"] = [{"slug": "cached", "max_context_size": 200_000}]
    auth_file.write_bytes(orjson.dumps(state))

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": backend_code}})

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

    assert caught.value.problem.code == backend_code.lower()
    assert snapshot.state == AUTH_LOGIN_REQUIRED
    assert [model.slug for model in catalog.models] == ["cached"]
    persisted = orjson.loads(auth_file.read_bytes())
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
                "account_id": "acct-1",
                "credential_id": "credential-cache",
                "models": [{"slug": "cached-account-model", "priority": 1}],
                "models_account_id": "acct-1",
                "models_credential_id": "credential-cache",
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
async def test_superseded_catalog_refresh_returns_current_catalog_state(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    old_access = _jwt(exp=10_000, account_id="shared-account")
    _write_tokens(auth_file, old_access, credential_id="old-credential")
    state = orjson.loads(auth_file.read_bytes())
    state["expires_at"] = 10_000
    auth_file.write_bytes(orjson.dumps(state))
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CODEX_MODELS_URL
        request_started.set()
        await release_request.wait()
        return httpx.Response(200, json={"models": [{"slug": "late-old-model"}]})

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        clock=lambda: 1_000,
    )
    task = asyncio.create_task(service.refresh_models(16))
    await request_started.wait()

    newer_access = _jwt(exp=20_000, account_id="shared-account")
    _write_tokens(auth_file, newer_access, credential_id="newer-credential")
    newer_state = orjson.loads(auth_file.read_bytes())
    newer_state.update(
        {
            "expires_at": 20_000,
            "models": [{"slug": "current-model"}],
            "models_account_id": "shared-account",
            "models_credential_id": "newer-credential",
            "models_updated_at": 2_000,
            "models_stale": False,
        }
    )
    auth_file.write_bytes(orjson.dumps(newer_state))
    release_request.set()
    catalog = await task
    await service.aclose()

    assert [model.slug for model in catalog.models] == ["current-model"]
    assert catalog.stale is False
    assert catalog.problem is None
    assert orjson.loads(auth_file.read_bytes()) == newer_state


@pytest.mark.asyncio
async def test_legacy_credentials_adopt_cached_model_ownership(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    access = _jwt(exp=10_000, account_id="legacy-account")
    auth_file.write_bytes(
        orjson.dumps(
            {
                "version": 1,
                "access_token": access,
                "refresh_token": "legacy-refresh",
                "expires_at": 10_000,
                "models": [{"slug": "legacy-model"}],
            }
        )
    )
    service = CodexAuthService(auth_file, clock=lambda: 1_000)

    before = await service.snapshot()
    credentials = await service.ensure_credentials()
    catalog = await service.catalog()
    await service.aclose()

    state = orjson.loads(auth_file.read_bytes())
    assert before.state == AUTH_CONNECTED
    assert before.model_count == 1
    assert credentials.account_id == "legacy-account"
    assert isinstance(state["credential_id"], str)
    assert state["models_credential_id"] == state["credential_id"]
    assert state["models_account_id"] == "legacy-account"
    assert [model.slug for model in catalog.models] == ["legacy-model"]


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
                "account_id": "acct-1",
                "credential_id": "credential-inflight",
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
    cached = service.cached_credentials()
    with pytest.raises(CodexAuthError) as caught:
        await service.ensure_credentials()
    await service.disconnect()

    assert snapshot.state == AUTH_LOGIN_REQUIRED
    assert cached is None
    assert caught.value.problem.code == PROBLEM_LOGIN_REQUIRED
    assert not auth_file.exists()


@pytest.mark.parametrize(
    "token_payload",
    [
        {"access_token": _jwt(), "refresh_token": "refresh"},
        {"id_token": _jwt(), "access_token": _jwt()},
        {"id_token": "opaque", "access_token": _jwt(), "refresh_token": "refresh"},
    ],
)
@pytest.mark.asyncio
async def test_initial_token_response_requires_complete_oidc_tokens(
    tmp_path: Path,
    token_payload: dict[str, str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        return httpx.Response(200, json=token_payload)

    service = CodexAuthService(
        tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )

    with pytest.raises(CodexAuthError) as caught:
        await service.login(20, _send_browser_callback)
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_INVALID_RESPONSE
    state = orjson.loads((tmp_path / "auth.json").read_bytes())
    assert "access_token" not in state
    assert "refresh_token" not in state


@pytest.mark.asyncio
async def test_terminal_catalog_failure_restores_previous_credentials(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    previous_access = _jwt(account_id="previous-account")
    _write_tokens(auth_file, previous_access, "previous-refresh", credential_id="previous-id")
    previous_state = orjson.loads(auth_file.read_bytes())
    previous_state.update(
        {
            "models": [{"slug": "previous-model"}],
            "models_account_id": "previous-account",
            "models_credential_id": "previous-id",
        }
    )
    auth_file.write_bytes(orjson.dumps(previous_state))

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="new-account"),
                    "access_token": _jwt(account_id="new-account"),
                    "refresh_token": "new-refresh",
                },
            )
        if str(request.url) == CODEX_MODELS_URL:
            return httpx.Response(403)
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    with pytest.raises(CodexAuthError) as caught:
        await service.login(20, _send_browser_callback)
    await service.aclose()

    persisted = orjson.loads(auth_file.read_bytes())
    assert caught.value.problem.code == PROBLEM_LOGIN_REQUIRED
    assert persisted == previous_state


@pytest.mark.asyncio
async def test_superseded_login_cannot_publish_or_overwrite_newer_credentials(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    model_request_started = asyncio.Event()
    release_model_request = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="shared-account"),
                    "access_token": _jwt(exp=20_000, account_id="shared-account"),
                    "refresh_token": "operation-refresh",
                },
            )
        if str(request.url) == CODEX_MODELS_URL:
            model_request_started.set()
            await release_model_request.wait()
            return httpx.Response(200, json={"models": [{"slug": "late-old-model"}]})
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    login_task = asyncio.create_task(service.login(21, _send_browser_callback))
    await model_request_started.wait()

    newer_access = _jwt(exp=30_000, account_id="shared-account")
    _write_tokens(
        auth_file,
        newer_access,
        "newer-refresh",
        credential_id="newer-credential",
    )
    newer_state = orjson.loads(auth_file.read_bytes())
    newer_state.update(
        {
            "models": [{"slug": "newer-model"}],
            "models_account_id": "shared-account",
            "models_credential_id": "newer-credential",
        }
    )
    auth_file.write_bytes(orjson.dumps(newer_state))
    release_model_request.set()

    with pytest.raises(CodexAuthError) as caught:
        await login_task
    await service.aclose()

    persisted = orjson.loads(auth_file.read_bytes())
    assert caught.value.problem.code == PROBLEM_LOGIN_SUPERSEDED
    assert persisted["access_token"] == newer_access
    assert persisted["credential_id"] == "newer-credential"
    assert persisted["models"] == [{"slug": "newer-model"}]


@pytest.mark.asyncio
async def test_account_changing_refresh_supersedes_inflight_login_and_catalog(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    model_request_started = asyncio.Event()
    release_model_request = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            if request.headers["Content-Type"].startswith("application/json"):
                return httpx.Response(
                    200,
                    json={
                        "id_token": _jwt(account_id="account-b"),
                        "access_token": _jwt(exp=30_000, account_id="account-b"),
                        "refresh_token": "refresh-b",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="account-a"),
                    "access_token": _jwt(exp=20_000, account_id="account-a"),
                    "refresh_token": "refresh-a",
                },
            )
        if str(request.url) == CODEX_MODELS_URL:
            model_request_started.set()
            await release_model_request.wait()
            return httpx.Response(200, json={"models": [{"slug": "account-a-only"}]})
        raise AssertionError(str(request.url))

    transport = httpx.MockTransport(handler)
    login_service = CodexAuthService(
        auth_file,
        transport=transport,
        callback_ports=(0,),
        clock=lambda: 1_000,
    )
    refresh_service = CodexAuthService(auth_file, transport=transport, clock=lambda: 1_000)
    login_task = asyncio.create_task(login_service.login(22, _send_browser_callback))
    await model_request_started.wait()
    login_generation = orjson.loads(auth_file.read_bytes())["credential_id"]

    refreshed = await refresh_service.ensure_credentials(force_refresh=True)
    release_model_request.set()
    with pytest.raises(CodexAuthError) as caught:
        await login_task
    await login_service.aclose()
    await refresh_service.aclose()

    persisted = orjson.loads(auth_file.read_bytes())
    assert caught.value.problem.code == PROBLEM_LOGIN_SUPERSEDED
    assert refreshed.account_id == "account-b"
    assert refreshed.credential_id != login_generation
    assert persisted["account_id"] == "account-b"
    assert persisted["credential_id"] == refreshed.credential_id
    assert "models" not in persisted


@pytest.mark.asyncio
async def test_new_account_never_inherits_previous_account_model_cache(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    account_a_access = _jwt(account_id="account-a")
    _write_tokens(auth_file, account_a_access, credential_id="credential-a")
    state = orjson.loads(auth_file.read_bytes())
    state.update(
        {
            "models": [{"slug": "account-a-only"}],
            "models_account_id": "account-a",
            "models_credential_id": "credential-a",
        }
    )
    auth_file.write_bytes(orjson.dumps(state))

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="account-b"),
                    "access_token": _jwt(account_id="account-b"),
                    "refresh_token": "refresh-b",
                },
            )
        if str(request.url) == CODEX_MODELS_URL:
            raise httpx.ConnectError("offline", request=request)
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    snapshot, catalog = await service.login(21, _send_browser_callback)
    await service.aclose()

    assert snapshot.state == AUTH_CONNECTED
    assert "account-a-only" not in {model.slug for model in catalog.models}
    persisted = orjson.loads(auth_file.read_bytes())
    assert persisted["account_id"] == "account-b"
    assert "models" not in persisted


@pytest.mark.asyncio
async def test_cancelled_login_does_not_delete_newer_process_credentials(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    model_request_started = asyncio.Event()
    release_models = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "id_token": _jwt(account_id="account-a"),
                    "access_token": _jwt(account_id="account-a"),
                    "refresh_token": "refresh-a",
                },
            )
        if str(request.url) == CODEX_MODELS_URL:
            model_request_started.set()
            await release_models.wait()
            return httpx.Response(200, json={"models": [{"slug": "model-a"}]})
        raise AssertionError(str(request.url))

    service = CodexAuthService(
        auth_file,
        transport=httpx.MockTransport(handler),
        callback_ports=(0,),
    )
    login_task = asyncio.create_task(service.login(22, _send_browser_callback))
    await model_request_started.wait()
    account_b_access = _jwt(account_id="account-b")
    _write_tokens(
        auth_file,
        account_b_access,
        "refresh-b",
        credential_id="credential-b",
    )
    service.cancel_login(22)
    release_models.set()

    with pytest.raises(CodexAuthError) as caught:
        await login_task
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CANCELLED
    persisted = orjson.loads(auth_file.read_bytes())
    assert persisted["access_token"] == account_b_access
    assert persisted["refresh_token"] == "refresh-b"
    assert persisted["credential_id"] == "credential-b"


@pytest.mark.asyncio
async def test_credential_store_failures_are_sanitized_domain_errors(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked", encoding="utf-8")
    service = CodexAuthService(blocked_parent / "auth.json")

    with pytest.raises(CodexAuthError) as caught:
        await service.disconnect()
    await service.aclose()

    assert caught.value.problem.code == PROBLEM_CREDENTIAL_STORE_UNAVAILABLE
    assert str(blocked_parent) not in str(caught.value)


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
