"""Tests for OAuth token refresh: retry with backoff and force refresh."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pydantic import SecretStr

from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    AUTH_LOGIN_REQUIRED,
    CODEX_BASE_URL,
    CODEX_OAUTH_KEY,
    PROBLEM_CREDENTIAL_STORE_UNAVAILABLE,
    PROBLEM_LOGIN_REQUIRED,
    CodexAuthError,
    CodexAuthSnapshot,
    CodexModel,
    CodexModelCatalog,
    CodexProblem,
)
from kimi_cli.auth.oauth import (
    _REJECTED_REFRESH_TOKENS,
    XAI_API_KEY_ENV,
    XAI_API_KEY_LEGACY_ENV,
    XAI_DEFAULT_BASE_URL,
    XAI_OAUTH_KEY,
    XAI_OAUTH_SCOPE,
    XAI_TOKEN_AUTH_HEADER,
    OAuthError,
    OAuthManager,
    OAuthToken,
    OAuthUnauthorized,
    _refresh_threshold,
    _save_to_file,
    has_xai_api_key_env,
    login_codex,
    login_xai,
    logout_codex,
    logout_xai,
    read_xai_api_key_env,
    refresh_token,
    register_xai_api_key,
)
from kimi_cli.config import Config, LLMModel, LLMProvider, OAuthRef, Services
from kimi_cli.llm import LLM

# ── helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_rejected_refresh_tokens():
    _REJECTED_REFRESH_TOKENS.clear()
    yield
    _REJECTED_REFRESH_TOKENS.clear()


def _make_token(
    *,
    expires_in: float = 900,
    access: str = "access-123",
    refresh: str = "refresh-123",
) -> OAuthToken:
    return OAuthToken(
        access_token=access,
        refresh_token=refresh,
        expires_at=time.time() + expires_in,
        scope="kimi-code",
        token_type="Bearer",
        expires_in=expires_in,
    )


def _make_config() -> Config:
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key="oauth/kimi-code"),
    )
    model = LLMModel(model="test-model", max_context_size=100_000)
    return Config(
        provider=provider,
        model=model,
        services=Services(),
    )


def _make_xai_config() -> Config:
    provider = LLMProvider(
        type="xai",
        base_url="https://api.x.ai/v1",
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key=XAI_OAUTH_KEY),
    )
    model = LLMModel(model="grok-3", max_context_size=131_072)
    return Config(
        provider=provider,
        model=model,
        services=Services(),
    )


def _make_manager(token: OAuthToken | None = None) -> OAuthManager:
    with patch("kimi_cli.auth.oauth.load_tokens", return_value=token):
        return OAuthManager(_make_config())


# ── refresh_token retry on network errors ──────────────────────


@pytest.mark.asyncio
async def test_refresh_token_retries_on_network_error():
    """refresh_token should retry up to max_retries on transient network errors."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 900,
            "scope": "kimi-code",
            "token_type": "Bearer",
        }
    )

    call_count = 0

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeContext()

    class FakeContext:
        async def __aenter__(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientError("Connection reset")
            return mock_response

        async def __aexit__(self, *args):
            pass

    with patch("kimi_cli.auth.oauth.new_client_session", return_value=FakeSession()):
        result = await refresh_token("old-refresh", max_retries=3)

    assert result.access_token == "new-access"
    assert call_count == 3  # Failed twice, succeeded third time


@pytest.mark.asyncio
async def test_refresh_token_does_not_retry_on_unauthorized():
    """OAuthUnauthorized should not be retried."""

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeContext()

    class FakeContext:
        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.status = 401
            mock_resp.json = AsyncMock(return_value={"error_description": "Token revoked"})
            return mock_resp

        async def __aexit__(self, *args):
            pass

    with (
        patch("kimi_cli.auth.oauth.new_client_session", return_value=FakeSession()),
        pytest.raises(OAuthUnauthorized, match="Token revoked"),
    ):
        await refresh_token("bad-refresh", max_retries=3)


@pytest.mark.asyncio
async def test_refresh_token_raises_after_all_retries_exhausted():
    """After max_retries network failures, should raise OAuthError."""

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeContext()

    class FakeContext:
        async def __aenter__(self):
            raise aiohttp.ClientError("Network down")

        async def __aexit__(self, *args):
            pass

    with (
        patch("kimi_cli.auth.oauth.new_client_session", return_value=FakeSession()),
        pytest.raises(OAuthError, match="after retries"),
    ):
        await refresh_token("some-refresh", max_retries=2)


@pytest.mark.asyncio
async def test_refresh_token_retries_on_5xx():
    """refresh_token should retry when the auth server returns 502/503."""
    ok_response = MagicMock()
    ok_response.status = 200
    ok_response.json = AsyncMock(
        return_value={
            "access_token": "recovered",
            "refresh_token": "new-refresh",
            "expires_in": 900,
            "scope": "kimi-code",
            "token_type": "Bearer",
        }
    )

    call_count = 0

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeContext()

    class FakeContext:
        async def __aenter__(self):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count < 3:
                resp.status = 502
                resp.json = AsyncMock(return_value={})
                return resp
            return ok_response

        async def __aexit__(self, *args):
            pass

    with patch("kimi_cli.auth.oauth.new_client_session", return_value=FakeSession()):
        result = await refresh_token("old-refresh", max_retries=3)

    assert result.access_token == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_refresh_token_does_not_retry_on_400():
    """Non-retryable HTTP errors (e.g. 400) should fail immediately."""

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeContext()

    class FakeContext:
        async def __aenter__(self):
            resp = MagicMock()
            resp.status = 400
            resp.json = AsyncMock(return_value={"error_description": "invalid_grant"})
            return resp

        async def __aexit__(self, *args):
            pass

    with (
        patch("kimi_cli.auth.oauth.new_client_session", return_value=FakeSession()),
        pytest.raises(OAuthError, match="invalid_grant"),
    ):
        await refresh_token("bad-refresh", max_retries=3)


# ── force refresh ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_fresh_force_bypasses_threshold():
    """force=True should refresh even when token has plenty of time left."""
    token = _make_token(expires_in=800)  # 13+ minutes remaining
    manager = _make_manager(token)

    mock_refresh = AsyncMock(return_value=_make_token())

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch("kimi_cli.auth.oauth.refresh_token", mock_refresh),
        patch("kimi_cli.auth.oauth.save_tokens"),
    ):
        await manager.ensure_fresh(force=True)

    mock_refresh.assert_called_once()


# ── dynamic threshold ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_fresh_uses_dynamic_threshold():
    """When expires_in is large, threshold should be expires_in * RATIO."""
    # Token with 1800s total lifetime; dynamic threshold = 1800 * 0.5 = 900.
    # Remaining 850s < 900 => should trigger refresh.
    token = _make_token(expires_in=1800)
    token.expires_at = time.time() + 850  # simulate time passing
    manager = _make_manager(token)

    mock_refresh = AsyncMock(return_value=_make_token())

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch("kimi_cli.auth.oauth.refresh_token", mock_refresh),
        patch("kimi_cli.auth.oauth.save_tokens"),
    ):
        await manager.ensure_fresh()

    mock_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_fresh_skips_when_plenty_of_time():
    """When remaining time exceeds the dynamic threshold, skip refresh."""
    # Token with 1800s total lifetime; dynamic threshold = 1800 * 0.5 = 900.
    # Remaining 1000s > 900 => should NOT trigger refresh.
    token = _make_token(expires_in=1800)
    token.expires_at = time.time() + 1000  # plenty of time
    manager = _make_manager(token)

    mock_refresh = AsyncMock(return_value=_make_token())

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch("kimi_cli.auth.oauth.refresh_token", mock_refresh),
        patch("kimi_cli.auth.oauth.save_tokens"),
    ):
        await manager.ensure_fresh()

    mock_refresh.assert_not_called()


# ── atomic save ────────────────────────────────────────────────


def test_save_to_file_is_atomic(tmp_path):
    """_save_to_file should write atomically via rename, not in-place."""
    key = "test-atomic"
    with patch("kimi_cli.auth.oauth._credentials_dir", return_value=tmp_path):
        token = _make_token()
        _save_to_file(key, token)
        path = tmp_path / f"{key}.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["access_token"] == "access-123"
        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


def test_save_to_file_expires_in_roundtrip(tmp_path):
    """expires_in should survive a save/load roundtrip."""
    key = "test-roundtrip"
    with patch("kimi_cli.auth.oauth._credentials_dir", return_value=tmp_path):
        token = _make_token(expires_in=7200)
        _save_to_file(key, token)
        path = tmp_path / f"{key}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        restored = OAuthToken.from_dict(data)
        assert restored.expires_in == 7200


# ── OAuthToken defaults ───────────────────────────────────────


def test_oauth_token_from_dict_defaults_expires_in():
    """from_dict should default expires_in to 0 when key is missing."""
    payload = {
        "access_token": "a",
        "refresh_token": "r",
        "expires_at": 123.0,
        "scope": "s",
        "token_type": "Bearer",
    }
    token = OAuthToken.from_dict(payload)
    assert token.expires_in == 0.0


# ── force refresh failure propagation ─────────────────────────


@pytest.mark.asyncio
async def test_ensure_fresh_force_raises_on_unauthorized():
    """force=True should propagate OAuthUnauthorized instead of swallowing it."""
    token = _make_token(expires_in=800)
    manager = _make_manager(token)

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch(
            "kimi_cli.auth.oauth.refresh_token", AsyncMock(side_effect=OAuthUnauthorized("revoked"))
        ),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
        pytest.raises(OAuthUnauthorized, match="revoked"),
    ):
        await manager.ensure_fresh(force=True)


@pytest.mark.asyncio
async def test_unauthorized_must_not_delete_credentials_file(tmp_path, monkeypatch):
    """A single 401 from the refresh endpoint must not delete the credentials
    file.  The load_tokens check above the deletion site is vulnerable to a
    TOCTOU race: a concurrent manager may write a freshly rotated token into
    the file between the check and the deletion, and wiping it would cause
    permanent auth loss even though a valid token is sitting on disk.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    _save_to_file("oauth/kimi-code", _make_token(refresh="R1", expires_in=100))
    cred = tmp_path / "credentials" / "kimi-code.json"
    assert cred.exists()

    manager = OAuthManager(_make_config())

    with (
        patch(
            "kimi_cli.auth.oauth.refresh_token",
            AsyncMock(side_effect=OAuthUnauthorized("invalid_grant")),
        ),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
        pytest.raises(OAuthUnauthorized),
    ):
        await manager.ensure_fresh(force=True)

    assert cred.exists(), (
        "credentials file was deleted on a single 401 — a concurrent "
        "manager may have just rotated the token in the TOCTOU window"
    )
    assert manager._access_tokens.get("oauth/kimi-code") is None, (
        "in-memory access token cache must still be cleared after 401"
    )


@pytest.mark.asyncio
async def test_unauthorized_non_force_must_not_delete_credentials_file(tmp_path, monkeypatch):
    """Same guarantee as the force=True case, but for the background-refresh
    path.  force=False swallows the exception rather than re-raising, but it
    still must not delete the credentials file on a single 401 — a concurrent
    manager may have just rotated the token.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    _save_to_file("oauth/kimi-code", _make_token(refresh="R1", expires_in=100))
    cred = tmp_path / "credentials" / "kimi-code.json"
    assert cred.exists()

    manager = OAuthManager(_make_config())

    with (
        patch(
            "kimi_cli.auth.oauth.refresh_token",
            AsyncMock(side_effect=OAuthUnauthorized("invalid_grant")),
        ),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
    ):
        # force=False: should NOT raise, just log warning and return
        await manager.ensure_fresh(force=False)

    assert cred.exists(), (
        "credentials file was deleted on a background-refresh 401 — "
        "same TOCTOU risk as the force=True case"
    )
    assert manager._access_tokens.get("oauth/kimi-code") is None, (
        "in-memory access token cache must still be cleared after 401"
    )


@pytest.mark.asyncio
async def test_rejected_refresh_token_cooldown_skips_background_retry(tmp_path, monkeypatch):
    """After a confirmed refresh 401, the same persisted refresh token should
    not be retried again immediately by the background-refresh path.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    _save_to_file("oauth/kimi-code", _make_token(refresh="R1", expires_in=100))

    manager = OAuthManager(_make_config())
    refresh = AsyncMock(side_effect=OAuthUnauthorized("invalid_grant"))

    with (
        patch("kimi_cli.auth.oauth.refresh_token", refresh),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(OAuthUnauthorized):
            await manager.ensure_fresh(force=True)

        await manager.ensure_fresh(force=False)

    assert refresh.await_count == 1, (
        "background refresh retried the same rejected refresh token without "
        "waiting for the cooldown"
    )


@pytest.mark.asyncio
async def test_rejected_tombstone_cleared_when_concurrent_instance_rotated(tmp_path, monkeypatch):
    """If another kimi-cli instance legitimately rotates the refresh token
    after we marked the old one rejected, the tombstone must clear and the
    new token must be picked up without going to the network.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    _save_to_file("oauth/kimi-code", _make_token(refresh="R1", expires_in=100))

    manager = OAuthManager(_make_config())
    refresh = AsyncMock(side_effect=OAuthUnauthorized("invalid_grant"))

    # Step 1: hit a 401 with R1 → marks R1 rejected
    with (
        patch("kimi_cli.auth.oauth.refresh_token", refresh),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
        pytest.raises(OAuthUnauthorized),
    ):
        await manager.ensure_fresh(force=True)

    assert _REJECTED_REFRESH_TOKENS.get("oauth/kimi-code") is not None

    # Step 2: simulate a concurrent instance writing a fresh token (R2) to disk
    _save_to_file(
        "oauth/kimi-code",
        _make_token(access="new-access", refresh="R2", expires_in=900),
    )

    # Step 3: next ensure_fresh should detect the rotation, clear the
    # tombstone, and NOT call refresh_token again
    with (
        patch("kimi_cli.auth.oauth.refresh_token", refresh),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
    ):
        await manager.ensure_fresh(force=False)

    assert refresh.await_count == 1, "should not retry refresh after rotation recovered"
    assert _REJECTED_REFRESH_TOKENS.get("oauth/kimi-code") is None, (
        "tombstone must be cleared once the on-disk refresh_token no longer matches"
    )
    assert manager._access_tokens.get("oauth/kimi-code") == "new-access", (
        "the new access token from R2 should be cached"
    )


@pytest.mark.asyncio
async def test_ensure_fresh_force_raises_on_network_error():
    """force=True should propagate network errors instead of swallowing them."""
    token = _make_token(expires_in=800)
    manager = _make_manager(token)

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch(
            "kimi_cli.auth.oauth.refresh_token", AsyncMock(side_effect=OAuthError("after retries"))
        ),
        pytest.raises(OAuthError, match="after retries"),
    ):
        await manager.ensure_fresh(force=True)


@pytest.mark.asyncio
async def test_ensure_fresh_non_force_swallows_errors():
    """Without force, refresh errors should be swallowed (background loop behavior)."""
    token = _make_token(expires_in=100)  # below threshold → triggers refresh
    token.expires_at = time.time() + 100
    manager = _make_manager(token)

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch("kimi_cli.auth.oauth.refresh_token", AsyncMock(side_effect=OAuthError("fail"))),
    ):
        # Should NOT raise — errors are swallowed in background mode
        await manager.ensure_fresh()


# ── _refresh_threshold helper ─────────────────────────────────


def test_refresh_threshold_uses_ratio_when_large():
    """When expires_in * RATIO > MIN, use the ratio-based threshold."""
    assert _refresh_threshold(1800) == 900.0  # 1800 * 0.5 = 900 > 300


def test_refresh_threshold_uses_minimum_when_small():
    """When expires_in * RATIO < MIN, use the minimum."""
    assert _refresh_threshold(500) == 300.0  # 500 * 0.5 = 250 < 300


def test_refresh_threshold_zero_expires_in():
    """When expires_in is 0, fall back to the minimum."""
    assert _refresh_threshold(0) == 300.0


# ── multi-provider dispatch ────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_fresh_dispatches_xai_refresh():
    """When the configured OAuth ref is xai, ensure_fresh must call refresh_xai_token."""
    token = _make_token(expires_in=100)
    manager = OAuthManager(_make_xai_config())
    refreshed = _make_token(access="xai-new-access", refresh="xai-new-refresh")

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=token),
        patch(
            "kimi_cli.auth.oauth.refresh_xai_token", AsyncMock(return_value=refreshed)
        ) as mock_refresh,
        patch("kimi_cli.auth.oauth.refresh_token") as mock_kimi_refresh,
        patch("kimi_cli.auth.oauth.save_tokens"),
    ):
        await manager.ensure_fresh(force=True)

    mock_refresh.assert_awaited_once_with(token.refresh_token)
    mock_kimi_refresh.assert_not_called()
    assert manager._access_tokens.get(XAI_OAUTH_KEY) == "xai-new-access"


def test_apply_access_token_xai():
    """_apply_access_token should update the XAI chat provider's api_key."""
    from kosong.chat_provider.xai import XAI

    config = _make_xai_config()
    chat_provider = XAI(model="grok-3", api_key="xai-fallback-key")
    llm = LLM(
        chat_provider=chat_provider,
        max_context_size=131_072,
        capabilities=set(),
        model_config=config.model,
        provider_config=config.provider,
    )
    runtime = SimpleNamespace(config=config, llm=llm)
    manager = OAuthManager(config)

    ref = OAuthRef(storage="file", key=XAI_OAUTH_KEY)
    manager._apply_access_token(ref, runtime, "xai-oauth-token")

    assert chat_provider.client.api_key == "xai-oauth-token"


def test_apply_access_token_xai_fallback_to_configured_key():
    """When access_token is empty, _apply_access_token should fall back to the configured api_key."""
    from kosong.chat_provider.xai import XAI

    config = _make_xai_config()
    config.provider.api_key = SecretStr("xai-configured-key")
    chat_provider = XAI(model="grok-3", api_key="xai-original-key")
    llm = LLM(
        chat_provider=chat_provider,
        max_context_size=131_072,
        capabilities=set(),
        model_config=config.model,
        provider_config=config.provider,
    )
    runtime = SimpleNamespace(config=config, llm=llm)
    manager = OAuthManager(config)

    ref = OAuthRef(storage="file", key=XAI_OAUTH_KEY)
    manager._apply_access_token(ref, runtime, "")

    assert chat_provider.client.api_key == "xai-configured-key"


# ── xAI API key env helpers ─────────────────────────────────────


def test_read_xai_api_key_env_prefers_xai_api_key(monkeypatch):
    monkeypatch.setenv(XAI_API_KEY_ENV, "xai-key")
    monkeypatch.setenv(XAI_API_KEY_LEGACY_ENV, "legacy-key")
    assert read_xai_api_key_env() == "xai-key"


def test_read_xai_api_key_env_falls_back_to_legacy(monkeypatch):
    monkeypatch.delenv(XAI_API_KEY_ENV, raising=False)
    monkeypatch.setenv(XAI_API_KEY_LEGACY_ENV, "legacy-key")
    assert read_xai_api_key_env() == "legacy-key"


def test_read_xai_api_key_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(XAI_API_KEY_ENV, raising=False)
    monkeypatch.delenv(XAI_API_KEY_LEGACY_ENV, raising=False)
    assert read_xai_api_key_env() is None


def test_has_xai_api_key_env(monkeypatch):
    monkeypatch.delenv(XAI_API_KEY_ENV, raising=False)
    monkeypatch.delenv(XAI_API_KEY_LEGACY_ENV, raising=False)
    assert has_xai_api_key_env() is False

    monkeypatch.setenv(XAI_API_KEY_ENV, "xai-key")
    assert has_xai_api_key_env() is True


# ── xAI login sets token auth header ──────────────────────────────


@pytest.mark.asyncio
async def test_login_xai_sets_token_auth_custom_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    config = _make_xai_config()
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True

    async def fake_device_authorization():
        from kimi_cli.auth.oauth import DeviceAuthorization

        return DeviceAuthorization(
            user_code="code",
            device_code="device",
            verification_uri="https://auth.x.ai/verify",
            verification_uri_complete="https://auth.x.ai/verify?code=code",
            expires_in=300,
            interval=1,
        )

    async def fake_device_token(auth):
        return 200, {
            "access_token": "xai-access",
            "refresh_token": "xai-refresh",
            "expires_in": 900,
            "scope": XAI_OAUTH_SCOPE,
            "token_type": "Bearer",
        }

    with (
        patch("kimi_cli.auth.oauth.request_xai_device_authorization", fake_device_authorization),
        patch("kimi_cli.auth.oauth._request_xai_device_token", fake_device_token),
        patch("kimi_cli.auth.oauth._list_xai_models", AsyncMock(return_value=["grok-3"])),
        patch(
            "kimi_cli.auth.oauth.save_tokens",
            return_value=OAuthRef(storage="file", key=XAI_OAUTH_KEY),
        ),
    ):
        events = [e async for e in login_xai(config, open_browser=False)]

    assert any(e.type == "success" for e in events)
    assert config.provider is not None
    assert config.provider.custom_headers == XAI_TOKEN_AUTH_HEADER
    assert config.provider.oauth is not None
    assert config.provider.oauth.key == XAI_OAUTH_KEY


# ── xAI API key registration ──────────────────────────────────────


@pytest.mark.asyncio
async def test_register_xai_api_key_stores_key_and_default_model(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    config = _make_xai_config()
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True

    with patch("kimi_cli.auth.oauth._list_xai_models", AsyncMock(return_value=["grok-3"])):
        events = [e async for e in register_xai_api_key(config, "xai-api-key-123")]

    assert any(e.type == "success" for e in events)
    assert config.provider is not None
    assert config.provider.type == "xai"
    assert config.provider.api_key.get_secret_value() == "xai-api-key-123"
    assert config.provider.oauth is None
    assert config.provider.custom_headers is None
    assert config.model is not None
    assert config.model.model == "grok-3"


@pytest.mark.asyncio
async def test_register_xai_api_key_requires_default_config_location():
    config = _make_xai_config()
    config.is_from_default_location = False

    events = [e async for e in register_xai_api_key(config, "xai-api-key-123")]
    assert any(e.type == "error" for e in events)


@pytest.mark.asyncio
async def test_register_xai_api_key_rejects_empty_key():
    config = _make_xai_config()

    events = [e async for e in register_xai_api_key(config, "   ")]
    assert any(e.type == "error" for e in events)


# ── xAI logout clears API key providers ─────────────────────────


@pytest.mark.asyncio
async def test_logout_xai_clears_api_key_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    config = _make_xai_config()
    config.provider = LLMProvider(
        type="xai",
        base_url=XAI_DEFAULT_BASE_URL,
        api_key=SecretStr("xai-api-key"),
    )
    config.model = LLMModel(model="grok-3", max_context_size=131_072)
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True

    with patch("kimi_cli.auth.oauth.save_config") as mock_save:
        events = [e async for e in logout_xai(config)]

    assert any(e.type == "success" for e in events)
    assert config.provider is None
    assert config.model is None
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_login_codex_configures_shared_oauth_provider(tmp_path):
    config = _make_config()
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True
    service = MagicMock()

    async def fake_login(operation_id, challenge_callback):
        await challenge_callback(
            SimpleNamespace(authorization_url="https://auth.openai.test/authorize")
        )
        model = CodexModel(
            slug="gpt-5.4",
            display_name="GPT-5.4",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="high",
        )
        return (
            CodexAuthSnapshot(operation_id, AUTH_CONNECTED),
            CodexModelCatalog(operation_id, (model,), False),
        )

    service.login = fake_login
    with (
        patch("kimi_cli.auth.oauth.default_codex_auth_service", return_value=service),
        patch("kimi_cli.auth.oauth.webbrowser.open", return_value=True),
        patch("kimi_cli.auth.oauth.save_config") as mock_save,
    ):
        events = [event async for event in login_codex(config)]

    assert [event.type for event in events] == ["verification_url", "waiting", "success"]
    assert config.provider is not None
    assert config.provider.type == "openai-codex"
    assert config.provider.base_url == CODEX_BASE_URL
    assert config.provider.oauth == OAuthRef(storage="file", key=CODEX_OAUTH_KEY)
    assert config.model is not None
    assert config.model.model == "gpt-5.4"
    assert config.model.supported_efforts == {"low", "high"}
    assert config.max_tokens == 128_000
    mock_save.assert_called_once_with(config)


@pytest.mark.asyncio
async def test_login_codex_does_not_report_success_for_terminal_snapshot(tmp_path):
    config = _make_config()
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True
    original_provider = config.provider
    original_model = config.model
    service = MagicMock()

    async def fake_login(operation_id, challenge_callback):
        await challenge_callback(
            SimpleNamespace(authorization_url="https://auth.openai.test/authorize")
        )
        problem = CodexProblem(PROBLEM_LOGIN_REQUIRED)
        model = CodexModel(slug="fallback")
        return (
            CodexAuthSnapshot(
                operation_id,
                AUTH_LOGIN_REQUIRED,
                problem=problem,
            ),
            CodexModelCatalog(operation_id, (model,), True, problem),
        )

    service.login = fake_login
    with (
        patch("kimi_cli.auth.oauth.default_codex_auth_service", return_value=service),
        patch("kimi_cli.auth.oauth.webbrowser.open", return_value=True),
        patch("kimi_cli.auth.oauth.save_config") as mock_save,
    ):
        events = [event async for event in login_codex(config)]

    assert [event.type for event in events] == ["verification_url", "waiting", "error"]
    assert config.provider is original_provider
    assert config.model is original_model
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_closing_login_codex_generator_cancels_background_login(tmp_path):
    config = _make_config()
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True
    cancelled = asyncio.Event()
    service = MagicMock()

    async def fake_login(_operation_id, challenge_callback):
        await challenge_callback(
            SimpleNamespace(authorization_url="https://auth.openai.test/authorize")
        )
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    service.login = fake_login
    generator = login_codex(config)
    with patch("kimi_cli.auth.oauth.default_codex_auth_service", return_value=service):
        event = await anext(generator)
        assert event.type == "verification_url"
        await generator.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    service.cancel_login.assert_called_once()


@pytest.mark.asyncio
async def test_logout_codex_clears_shared_provider(tmp_path):
    config = _make_config()
    config.provider = LLMProvider(
        type="openai-codex",
        base_url=CODEX_BASE_URL,
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key=CODEX_OAUTH_KEY),
    )
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True
    service = MagicMock()
    service.disconnect = AsyncMock()

    with (
        patch("kimi_cli.auth.oauth.default_codex_auth_service", return_value=service),
        patch("kimi_cli.auth.oauth.save_config") as mock_save,
    ):
        events = [event async for event in logout_codex(config)]

    assert [event.type for event in events] == ["success"]
    service.disconnect.assert_awaited_once_with()
    assert config.provider is None
    assert config.model is None
    mock_save.assert_called_once_with(config)


@pytest.mark.asyncio
async def test_logout_codex_preserves_config_when_credential_removal_fails(tmp_path):
    config = _make_config()
    config.provider = LLMProvider(
        type="openai-codex",
        base_url=CODEX_BASE_URL,
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key=CODEX_OAUTH_KEY),
    )
    config.source_file = tmp_path / "config.toml"
    config.is_from_default_location = True
    original_provider = config.provider
    service = MagicMock()
    service.disconnect = AsyncMock(
        side_effect=CodexAuthError(CodexProblem(PROBLEM_CREDENTIAL_STORE_UNAVAILABLE))
    )

    with (
        patch("kimi_cli.auth.oauth.default_codex_auth_service", return_value=service),
        patch("kimi_cli.auth.oauth.save_config") as mock_save,
    ):
        events = [event async for event in logout_codex(config)]

    assert [event.type for event in events] == ["error"]
    assert config.provider is original_provider
    mock_save.assert_not_called()
