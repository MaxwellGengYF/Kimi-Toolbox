"""ChatGPT Codex OAuth, credential storage, and model discovery."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import stat
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import httpx
import orjson
from kimi_cli.share import get_share_dir

from kimix_gui import __version__

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_AUTH_FILENAME = "kimix-gui-codex-auth.json"
AUTH_STORE_VERSION = 1
REFRESH_SKEW_SECONDS = 120
BROWSER_LOGIN_TIMEOUT_SECONDS = 15 * 60
BROWSER_CALLBACK_HOST = "127.0.0.1"
BROWSER_CALLBACK_PORTS = (1455, 1457)
BROWSER_CALLBACK_PATH = "/auth/callback"
BROWSER_OAUTH_SCOPE = "openid profile email offline_access"
# Codex deliberately uses a 272K active window for ChatGPT-backed sessions even
# when the public API model supports a larger long-context window.  This mirrors
# the bundled model metadata in the official Codex client, rather than the API
# product-page limit.
DEFAULT_CONTEXT_WINDOW = 272_000
DEFAULT_MAX_OUTPUT_TOKENS = 128_000
MAX_CACHED_MODELS = 256
CREDENTIAL_LOCK_TIMEOUT_SECONDS = 30.0

DEFAULT_CODEX_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
)

AUTH_DISCONNECTED = "disconnected"
AUTH_CONNECTING = "connecting"
AUTH_CONNECTED = "connected"
AUTH_LOGIN_REQUIRED = "login_required"
AUTH_RETRY_LATER = "retry_later"

PROBLEM_CANCELLED = "cancelled"
PROBLEM_TIMEOUT = "timeout"
PROBLEM_RATE_LIMITED = "rate_limited"
PROBLEM_NETWORK = "network"
PROBLEM_INVALID_RESPONSE = "invalid_response"
PROBLEM_LOGIN_REQUIRED = "login_required"
PROBLEM_MODEL_UNAVAILABLE = "model_unavailable"
PROBLEM_SERVER = "server_error"
PROBLEM_CALLBACK_UNAVAILABLE = "callback_unavailable"

_TERMINAL_TOKEN_ERRORS = {"invalid_grant", "invalid_token", "refresh_token_reused"}


@dataclass(frozen=True, slots=True)
class CodexProblem:
    """A secret-free problem code for the Qt translation boundary."""

    code: str
    retry_after: float | None = None
    http_status: int | None = None

    def __str__(self) -> str:
        if self.http_status is not None:
            return f"Codex authentication error {self.code} (HTTP {self.http_status})"
        return f"Codex authentication error: {self.code}"


class CodexAuthError(RuntimeError):
    """An OAuth or catalog failure that never exposes credential material."""

    def __init__(self, problem: CodexProblem) -> None:
        super().__init__(str(problem))
        self.problem = problem


@dataclass(frozen=True, slots=True)
class CodexBrowserChallenge:
    operation_id: int
    authorization_url: str = field(repr=False)
    expires_at: float


@dataclass(frozen=True, slots=True)
class CodexModel:
    slug: str
    display_name: str | None = None
    max_context_size: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS
    input_modalities: tuple[str, ...] = ("text", "image")
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    priority: int = 10_000


@dataclass(frozen=True, slots=True)
class _CodexModelProfile:
    display_name: str
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str
    input_modalities: tuple[str, ...] = ("text", "image")
    max_context_size: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


_STANDARD_CODEX_EFFORTS = ("low", "medium", "high", "xhigh")
_GPT_56_MAX_EFFORTS = (*_STANDARD_CODEX_EFFORTS, "max")
_GPT_56_ULTRA_EFFORTS = (*_GPT_56_MAX_EFFORTS, "ultra")
_CODEX_MODEL_PROFILES = {
    "gpt-5.6-sol": _CodexModelProfile(
        "GPT-5.6-Sol",
        _GPT_56_ULTRA_EFFORTS,
        "low",
    ),
    "gpt-5.6-terra": _CodexModelProfile(
        "GPT-5.6-Terra",
        _GPT_56_ULTRA_EFFORTS,
        "medium",
    ),
    "gpt-5.6-luna": _CodexModelProfile(
        "GPT-5.6-Luna",
        _GPT_56_MAX_EFFORTS,
        "medium",
    ),
    "gpt-5.5": _CodexModelProfile("GPT-5.5", _STANDARD_CODEX_EFFORTS, "medium"),
    "gpt-5.4-mini": _CodexModelProfile(
        "GPT-5.4-Mini",
        _STANDARD_CODEX_EFFORTS,
        "medium",
    ),
    "gpt-5.4": _CodexModelProfile("GPT-5.4", _STANDARD_CODEX_EFFORTS, "medium"),
    "gpt-5.3-codex": _CodexModelProfile(
        "GPT-5.3-Codex",
        _STANDARD_CODEX_EFFORTS,
        "medium",
    ),
    "gpt-5.3-codex-spark": _CodexModelProfile(
        "GPT-5.3-Codex-Spark",
        _STANDARD_CODEX_EFFORTS,
        "high",
        input_modalities=("text",),
    ),
}


@dataclass(frozen=True, slots=True)
class CodexModelCatalog:
    operation_id: int
    models: tuple[CodexModel, ...]
    stale: bool
    problem: CodexProblem | None = None


@dataclass(frozen=True, slots=True)
class CodexAuthSnapshot:
    operation_id: int
    state: str
    model_count: int = 0
    stale: bool = True
    expires_at: float | None = None
    problem: CodexProblem | None = None


@dataclass(frozen=True, slots=True)
class CodexRuntimeCredentials:
    access_token: str = field(repr=False)
    account_id: str | None
    expires_at: float | None


@dataclass(frozen=True, slots=True)
class _StoredTokens:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float | None


@dataclass(frozen=True, slots=True)
class _BrowserAuthorization:
    challenge: CodexBrowserChallenge
    redirect_uri: str
    code_verifier: str = field(repr=False)
    callback: asyncio.Future[str] = field(repr=False, compare=False)
    server: asyncio.AbstractServer = field(repr=False, compare=False)
    lifetime: float = BROWSER_LOGIN_TIMEOUT_SECONDS


ChallengeCallback = Callable[[CodexBrowserChallenge], Awaitable[None] | None]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


def default_codex_auth_file() -> Path:
    """Return Kimix GUI's private Codex credential file."""

    return get_share_dir() / CODEX_AUTH_FILENAME


def extract_chatgpt_account_id(access_token: str) -> str | None:
    """Decode the ChatGPT account id from a JWT without validating its signature."""

    claims = _jwt_claims(access_token)
    if not isinstance(claims, dict):
        return None
    namespace = claims.get("https://api.openai.com/auth")
    if isinstance(namespace, dict):
        account_id = namespace.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    dotted = claims.get("https://api.openai.com/auth.chatgpt_account_id")
    return dotted if isinstance(dotted, str) and dotted else None


def token_expiry(access_token: str) -> float | None:
    claims = _jwt_claims(access_token)
    expiry = claims.get("exp") if isinstance(claims, dict) else None
    if isinstance(expiry, int | float) and expiry > 0:
        return float(expiry)
    return None


def _jwt_claims(token: str) -> Mapping[str, Any] | None:
    try:
        pieces = token.split(".")
        if len(pieces) < 2:
            return None
        payload = pieces[1] + "=" * (-len(pieces[1]) % 4)
        decoded = orjson.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        return decoded if isinstance(decoded, dict) else None
    except ValueError, UnicodeError, orjson.JSONDecodeError:
        return None


def _profile_model(slug: str, *, priority: int = 10_000) -> CodexModel:
    profile = _CODEX_MODEL_PROFILES.get(slug)
    if profile is None:
        return CodexModel(slug=slug, priority=priority)
    return CodexModel(
        slug=slug,
        display_name=profile.display_name,
        max_context_size=profile.max_context_size,
        max_tokens=profile.max_tokens,
        input_modalities=profile.input_modalities,
        reasoning_efforts=profile.reasoning_efforts,
        default_reasoning_effort=profile.default_reasoning_effort,
        priority=priority,
    )


def fallback_catalog(
    operation_id: int = 0, problem: CodexProblem | None = None
) -> CodexModelCatalog:
    return CodexModelCatalog(
        operation_id=operation_id,
        models=tuple(_profile_model(model) for model in DEFAULT_CODEX_MODELS),
        stale=True,
        problem=problem,
    )


class _CrossProcessLock:
    """Small advisory lock that works on Windows and POSIX."""

    def __init__(self, path: Path, *, clock: Clock, sleep: Sleep) -> None:
        self._path = path
        self._clock = clock
        self._sleep = sleep
        self._handle: Any = None

    async def acquire(self, timeout: float = CREDENTIAL_LOCK_TIMEOUT_SECONDS) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = self._clock() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return
            except OSError as exc:
                if self._clock() >= deadline:
                    handle.close()
                    raise TimeoutError("Timed out waiting for Codex credential lock") from exc
                await self._sleep(0.05)

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class CodexAuthService:
    """Own the independent OAuth store and all ChatGPT Codex network operations."""

    def __init__(
        self,
        auth_file: Path | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.time,
        monotonic: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        callback_ports: tuple[int, ...] = BROWSER_CALLBACK_PORTS,
        login_timeout: float = BROWSER_LOGIN_TIMEOUT_SECONDS,
    ) -> None:
        if login_timeout <= 0:
            raise ValueError("login_timeout must be positive")
        self.auth_file = auth_file or default_codex_auth_file()
        self._transport = transport
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._callback_ports = callback_ports
        self._login_timeout = login_timeout
        self._refresh_lock = asyncio.Lock()
        self._login_lock = asyncio.Lock()
        self._cancelled_operations: set[int] = set()
        self._cancel_events: dict[int, asyncio.Event] = {}
        self._client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def snapshot(self, operation_id: int = 0) -> CodexAuthSnapshot:
        state = self._read_state()
        tokens = self._tokens_from_state(state)
        models = self._models_from_state(state)
        problem = self._problem_from_state(state)
        if problem is not None and problem.code in _TERMINAL_TOKEN_ERRORS | {
            PROBLEM_LOGIN_REQUIRED
        }:
            auth_state = AUTH_LOGIN_REQUIRED
        elif tokens is not None:
            auth_state = AUTH_CONNECTED
        elif problem is not None and problem.code in {
            PROBLEM_CALLBACK_UNAVAILABLE,
            PROBLEM_RATE_LIMITED,
            PROBLEM_NETWORK,
            PROBLEM_SERVER,
        }:
            auth_state = AUTH_RETRY_LATER
        else:
            auth_state = AUTH_DISCONNECTED
        return CodexAuthSnapshot(
            operation_id=operation_id,
            state=auth_state,
            model_count=len(models),
            stale=bool(state.get("models_stale", not bool(state.get("models_updated_at")))),
            expires_at=tokens.expires_at if tokens else None,
            problem=problem,
        )

    def cancel_login(self, operation_id: int) -> None:
        self._cancelled_operations.add(operation_id)
        event = self._cancel_events.get(operation_id)
        if event is not None:
            event.set()

    async def login(
        self,
        operation_id: int,
        challenge_callback: ChallengeCallback,
    ) -> tuple[CodexAuthSnapshot, CodexModelCatalog]:
        async with self._login_lock:
            self._raise_if_cancelled(operation_id)
            cancel_event = asyncio.Event()
            self._cancel_events[operation_id] = cancel_event
            authorization: _BrowserAuthorization | None = None
            try:
                authorization = await self._start_browser_authorization(operation_id)
                callback_result = challenge_callback(authorization.challenge)
                if callback_result is not None:
                    await callback_result
                self._raise_if_cancelled(operation_id)
                authorization_code = await self._wait_for_browser_callback(
                    authorization,
                    operation_id,
                    cancel_event,
                )
                self._raise_if_cancelled(operation_id)
                await self._close_browser_authorization(authorization)
                token_payload = await self._exchange_browser_code(
                    authorization_code,
                    authorization.code_verifier,
                    authorization.redirect_uri,
                )
                tokens = self._parse_token_payload(token_payload)
                await self._save_tokens(tokens, operation_id=operation_id)
                await self._abort_if_cancelled_after_credentials(operation_id)
                catalog = await self.refresh_models(operation_id)
                await self._abort_if_cancelled_after_credentials(operation_id)
                snapshot = await self.snapshot(operation_id)
                return snapshot, catalog
            except CodexAuthError as exc:
                if exc.problem.code != PROBLEM_CANCELLED:
                    await self._record_auth_problem(exc.problem)
                raise
            finally:
                self._cancel_events.pop(operation_id, None)
                self._cancelled_operations.discard(operation_id)
                if authorization is not None:
                    await self._close_browser_authorization(authorization)

    async def disconnect(self, operation_id: int = 0) -> CodexAuthSnapshot:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                try:
                    self.auth_file.unlink()
                except FileNotFoundError:
                    pass
            finally:
                lock.release()
        return CodexAuthSnapshot(operation_id=operation_id, state=AUTH_DISCONNECTED)

    async def ensure_credentials(
        self,
        *,
        force_refresh: bool = False,
        failed_access_token: str | None = None,
    ) -> CodexRuntimeCredentials:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                state = self._read_state()
                tokens = self._tokens_from_state(state)
                if tokens is None:
                    raise CodexAuthError(CodexProblem(PROBLEM_LOGIN_REQUIRED))
                if failed_access_token is not None and tokens.access_token != failed_access_token:
                    return self._runtime_credentials(tokens)
                should_refresh = force_refresh or self._token_needs_refresh(tokens)
                if not should_refresh:
                    return self._runtime_credentials(tokens)
                refreshed = await self._refresh_tokens(tokens, state)
                return self._runtime_credentials(refreshed)
            finally:
                lock.release()

    async def catalog(self, operation_id: int = 0) -> CodexModelCatalog:
        state = self._read_state()
        models = self._models_from_state(state)
        if models:
            return CodexModelCatalog(
                operation_id=operation_id,
                models=models,
                stale=bool(state.get("models_stale", not bool(state.get("models_updated_at")))),
                problem=self._problem_from_state(state),
            )
        return fallback_catalog(operation_id, self._problem_from_state(state))

    async def refresh_models(self, operation_id: int = 0) -> CodexModelCatalog:
        access_token: str | None = None
        try:
            credentials = await self.ensure_credentials()
            access_token = credentials.access_token
            headers = {"Authorization": f"Bearer {credentials.access_token}"}
            if credentials.account_id:
                headers["ChatGPT-Account-ID"] = credentials.account_id
            response = await self._http().get(
                CODEX_MODELS_URL,
                headers=headers,
                timeout=20.0,
            )
            if response.status_code == 401:
                credentials = await self.ensure_credentials(
                    force_refresh=True,
                    failed_access_token=credentials.access_token,
                )
                access_token = credentials.access_token
                headers["Authorization"] = f"Bearer {credentials.access_token}"
                if credentials.account_id:
                    headers["ChatGPT-Account-ID"] = credentials.account_id
                else:
                    headers.pop("ChatGPT-Account-ID", None)
                response = await self._http().get(
                    CODEX_MODELS_URL,
                    headers=headers,
                    timeout=20.0,
                )
            response.raise_for_status()
            payload = response.json()
            models = parse_model_catalog(payload)
            if not models:
                raise CodexAuthError(CodexProblem(PROBLEM_INVALID_RESPONSE))
            if not await self._save_model_cache(models, expected_access_token=access_token):
                raise CodexAuthError(CodexProblem(PROBLEM_LOGIN_REQUIRED))
            return CodexModelCatalog(operation_id, models, False)
        except CodexAuthError as exc:
            return await self._stale_catalog(
                operation_id,
                exc.problem,
                expected_access_token=access_token,
            )
        except httpx.HTTPStatusError as exc:
            problem = self._http_problem(exc.response)
            return await self._stale_catalog(
                operation_id,
                problem,
                expected_access_token=access_token,
            )
        except httpx.HTTPError, OSError, ValueError, orjson.JSONDecodeError:
            return await self._stale_catalog(
                operation_id,
                CodexProblem(PROBLEM_NETWORK),
                expected_access_token=access_token,
            )

    async def _start_browser_authorization(
        self,
        operation_id: int,
    ) -> _BrowserAuthorization:
        self._raise_if_cancelled(operation_id)
        loop = asyncio.get_running_loop()
        callback: asyncio.Future[str] = loop.create_future()
        code_verifier = _base64url(secrets.token_bytes(64))
        code_challenge = _base64url(hashlib.sha256(code_verifier.encode("ascii")).digest())
        state = _base64url(secrets.token_bytes(32))

        async def handle_callback(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await self._handle_browser_callback(reader, writer, state, callback)

        for port in self._callback_ports:
            try:
                server = await asyncio.start_server(
                    handle_callback,
                    BROWSER_CALLBACK_HOST,
                    port,
                    limit=16 * 1024,
                )
            except OSError:
                continue
            sockets = server.sockets or ()
            if not sockets:
                server.close()
                await server.wait_closed()
                continue
            actual_port = int(sockets[0].getsockname()[1])
            redirect_uri = f"http://localhost:{actual_port}{BROWSER_CALLBACK_PATH}"
            authorization_url = _build_authorization_url(
                redirect_uri=redirect_uri,
                code_challenge=code_challenge,
                state=state,
            )
            try:
                self._raise_if_cancelled(operation_id)
            except CodexAuthError:
                server.close()
                await server.wait_closed()
                raise
            return _BrowserAuthorization(
                challenge=CodexBrowserChallenge(
                    operation_id=operation_id,
                    authorization_url=authorization_url,
                    expires_at=self._clock() + self._login_timeout,
                ),
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                callback=callback,
                server=server,
                lifetime=self._login_timeout,
            )
        raise CodexAuthError(CodexProblem(PROBLEM_CALLBACK_UNAVAILABLE))

    async def _handle_browser_callback(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        expected_state: str,
        callback: asyncio.Future[str],
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not request_line or len(request_line) > 8 * 1024:
                await _write_callback_response(writer, 400)
                return
            try:
                method, target, _version = request_line.decode("ascii").rstrip("\r\n").split(" ", 2)
            except UnicodeError, ValueError:
                await _write_callback_response(writer, 400)
                return
            total_header_bytes = 0
            while True:
                header = await asyncio.wait_for(reader.readline(), timeout=10.0)
                total_header_bytes += len(header)
                if not header or header in {b"\r\n", b"\n"}:
                    break
                if total_header_bytes > 16 * 1024:
                    await _write_callback_response(writer, 431)
                    return
            if method != "GET":
                await _write_callback_response(writer, 405)
                return
            parsed = urlsplit(target)
            if parsed.path != BROWSER_CALLBACK_PATH:
                await _write_callback_response(writer, 404)
                return
            params = parse_qs(parsed.query, keep_blank_values=True)
            received_state = _single_query_value(params, "state")
            if received_state is None or not secrets.compare_digest(
                received_state,
                expected_state,
            ):
                await _write_callback_response(writer, 400)
                return
            if callback.done():
                await _write_callback_response(writer, 409)
                return
            error = _single_query_value(params, "error")
            if error is not None:
                problem = (
                    CodexProblem(PROBLEM_CANCELLED)
                    if error == "access_denied"
                    else CodexProblem(PROBLEM_INVALID_RESPONSE)
                )
                callback.set_exception(CodexAuthError(problem))
                await _write_callback_response(writer, 400)
                return
            code = _single_query_value(params, "code")
            if code is None or not code:
                callback.set_exception(CodexAuthError(CodexProblem(PROBLEM_INVALID_RESPONSE)))
                await _write_callback_response(writer, 400)
                return
            callback.set_result(code)
            await _write_callback_response(writer, 200, close_window=True)
        except TimeoutError, OSError, ValueError:
            with suppress(OSError):
                await _write_callback_response(writer, 400)
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    async def _wait_for_browser_callback(
        self,
        authorization: _BrowserAuthorization,
        operation_id: int,
        cancel_event: asyncio.Event,
    ) -> str:
        cancellation = asyncio.create_task(cancel_event.wait())
        try:
            done, _pending = await asyncio.wait(
                (authorization.callback, cancellation),
                timeout=authorization.lifetime,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation in done:
                self._raise_if_cancelled(operation_id)
                raise CodexAuthError(CodexProblem(PROBLEM_CANCELLED))
            if authorization.callback in done:
                return authorization.callback.result()
            raise CodexAuthError(CodexProblem(PROBLEM_TIMEOUT))
        finally:
            cancellation.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation

    @staticmethod
    async def _close_browser_authorization(
        authorization: _BrowserAuthorization,
    ) -> None:
        authorization.server.close()
        await authorization.server.wait_closed()
        if not authorization.callback.done():
            authorization.callback.cancel()

    async def _exchange_browser_code(
        self,
        authorization_code: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Mapping[str, Any]:
        try:
            response = await self._http().post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": CODEX_CLIENT_ID,
                    "code": authorization_code,
                    "code_verifier": code_verifier,
                    "redirect_uri": redirect_uri,
                },
                timeout=20.0,
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError(CodexProblem(PROBLEM_NETWORK)) from exc
        if not response.is_success:
            raise CodexAuthError(self._token_problem(response))
        return self._json_object(response)

    async def _save_tokens(self, tokens: _StoredTokens, *, operation_id: int) -> None:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                self._raise_if_cancelled(operation_id)
                state = self._read_state()
                state.update(
                    {
                        "version": AUTH_STORE_VERSION,
                        "access_token": tokens.access_token,
                        "refresh_token": tokens.refresh_token,
                        "expires_at": tokens.expires_at,
                        "last_refresh_at": self._clock(),
                    }
                )
                state.pop("last_error", None)
                self._write_state(state)
            finally:
                lock.release()

    async def _abort_if_cancelled_after_credentials(self, operation_id: int) -> None:
        if operation_id not in self._cancelled_operations:
            return
        self._cancelled_operations.discard(operation_id)
        await self.disconnect(operation_id)
        raise CodexAuthError(CodexProblem(PROBLEM_CANCELLED))

    async def _refresh_tokens(
        self,
        tokens: _StoredTokens,
        state: dict[str, Any],
    ) -> _StoredTokens:
        if not tokens.refresh_token:
            problem = CodexProblem(PROBLEM_LOGIN_REQUIRED)
            self._write_terminal_problem(state, problem)
            raise CodexAuthError(problem)
        try:
            response = await self._http().post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CODEX_CLIENT_ID,
                    "refresh_token": tokens.refresh_token,
                },
                timeout=20.0,
            )
        except httpx.HTTPError as exc:
            problem = CodexProblem(PROBLEM_NETWORK)
            self._write_temporary_problem(state, problem)
            raise CodexAuthError(problem) from exc
        if not response.is_success:
            problem = self._token_problem(response)
            if problem.code in _TERMINAL_TOKEN_ERRORS or response.status_code in (401, 403):
                self._write_terminal_problem(state, problem)
            else:
                self._write_temporary_problem(state, problem)
            raise CodexAuthError(problem)
        payload = self._json_object(response)
        refreshed = self._parse_token_payload(payload, previous=tokens)
        state.update(
            {
                "version": AUTH_STORE_VERSION,
                "access_token": refreshed.access_token,
                "refresh_token": refreshed.refresh_token,
                "expires_at": refreshed.expires_at,
                "last_refresh_at": self._clock(),
            }
        )
        state.pop("last_error", None)
        self._write_state(state)
        return refreshed

    def _parse_token_payload(
        self,
        payload: Mapping[str, Any],
        *,
        previous: _StoredTokens | None = None,
    ) -> _StoredTokens:
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        if not isinstance(access, str) or not access:
            raise CodexAuthError(CodexProblem(PROBLEM_INVALID_RESPONSE))
        if not isinstance(refresh, str) or not refresh:
            refresh = previous.refresh_token if previous is not None else ""
        expires_at = token_expiry(access)
        expires_in = _positive_number(payload.get("expires_in"))
        if expires_at is None and expires_in is not None:
            expires_at = self._clock() + expires_in
        return _StoredTokens(access, refresh, expires_at)

    def _token_needs_refresh(self, tokens: _StoredTokens) -> bool:
        return tokens.expires_at is not None and (
            tokens.expires_at - self._clock() <= REFRESH_SKEW_SECONDS
        )

    @staticmethod
    def _runtime_credentials(tokens: _StoredTokens) -> CodexRuntimeCredentials:
        return CodexRuntimeCredentials(
            tokens.access_token,
            extract_chatgpt_account_id(tokens.access_token),
            tokens.expires_at,
        )

    async def _stale_catalog(
        self,
        operation_id: int,
        problem: CodexProblem,
        *,
        expected_access_token: str | None,
    ) -> CodexModelCatalog:
        await self._record_catalog_problem(
            problem,
            expected_access_token=expected_access_token,
        )
        state = self._read_state()
        models = self._models_from_state(state)
        if not models:
            return fallback_catalog(operation_id, problem)
        return CodexModelCatalog(operation_id, models, True, problem)

    async def _save_model_cache(
        self,
        models: tuple[CodexModel, ...],
        *,
        expected_access_token: str,
    ) -> bool:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                state = self._read_state()
                tokens = self._tokens_from_state(state)
                if tokens is None or not _tokens_belong_to_same_account(
                    expected_access_token,
                    tokens.access_token,
                ):
                    return False
                state["models"] = [_model_to_data(model) for model in models[:MAX_CACHED_MODELS]]
                state["models_updated_at"] = self._clock()
                state["models_stale"] = False
                state.pop("last_error", None)
                self._write_state(state)
                return True
            finally:
                lock.release()

    async def _record_catalog_problem(
        self,
        problem: CodexProblem,
        *,
        expected_access_token: str | None,
    ) -> None:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                state = self._read_state()
                tokens = self._tokens_from_state(state)
                if tokens is None:
                    return
                if expected_access_token is not None and not _tokens_belong_to_same_account(
                    expected_access_token,
                    tokens.access_token,
                ):
                    return
                state["models_stale"] = True
                self._set_problem(state, problem)
                self._write_state(state)
            finally:
                lock.release()

    async def _record_auth_problem(self, problem: CodexProblem) -> None:
        async with self._refresh_lock:
            lock = self._file_lock()
            await lock.acquire()
            try:
                state = self._read_state()
                if problem.code in _TERMINAL_TOKEN_ERRORS | {PROBLEM_LOGIN_REQUIRED}:
                    self._write_terminal_problem(state, problem)
                else:
                    self._set_problem(state, problem)
                    self._write_state(state)
            finally:
                lock.release()

    def _read_state(self) -> dict[str, Any]:
        try:
            loaded = orjson.loads(self.auth_file.read_bytes())
        except OSError, orjson.JSONDecodeError:
            return {"version": AUTH_STORE_VERSION}
        if not isinstance(loaded, dict) or loaded.get("version") != AUTH_STORE_VERSION:
            return {"version": AUTH_STORE_VERSION}
        return dict(loaded)

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.auth_file.parent.mkdir(parents=True, exist_ok=True)
        self._limit_permissions(self.auth_file.parent, directory=True)
        temporary = self.auth_file.with_name(f".{self.auth_file.name}.{uuid4().hex}.tmp")
        payload = orjson.dumps(dict(state), option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.auth_file)
            self._limit_permissions(self.auth_file, directory=False)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _limit_permissions(path: Path, *, directory: bool) -> None:
        try:
            os.chmod(path, stat.S_IRWXU if directory else stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    @staticmethod
    def _tokens_from_state(state: Mapping[str, Any]) -> _StoredTokens | None:
        access = state.get("access_token")
        refresh = state.get("refresh_token")
        expiry = state.get("expires_at")
        if not isinstance(access, str) or not access:
            return None
        return _StoredTokens(
            access,
            refresh if isinstance(refresh, str) else "",
            float(expiry) if isinstance(expiry, int | float) else token_expiry(access),
        )

    @staticmethod
    def _models_from_state(state: Mapping[str, Any]) -> tuple[CodexModel, ...]:
        entries = state.get("models")
        if not isinstance(entries, list):
            return ()
        models = tuple(
            model
            for item in entries[:MAX_CACHED_MODELS]
            if isinstance(item, dict) and (model := _model_from_data(item)) is not None
        )
        return models

    @staticmethod
    def _problem_from_state(state: Mapping[str, Any]) -> CodexProblem | None:
        value = state.get("last_error")
        if not isinstance(value, dict):
            return None
        code = value.get("code")
        if not isinstance(code, str) or not code:
            return None
        retry = value.get("retry_after")
        status = value.get("http_status")
        return CodexProblem(
            code,
            float(retry) if isinstance(retry, int | float) else None,
            int(status) if isinstance(status, int) else None,
        )

    def _write_terminal_problem(self, state: dict[str, Any], problem: CodexProblem) -> None:
        state.pop("access_token", None)
        state.pop("refresh_token", None)
        state.pop("expires_at", None)
        self._set_problem(state, problem)
        self._write_state(state)

    def _write_temporary_problem(self, state: dict[str, Any], problem: CodexProblem) -> None:
        self._set_problem(state, problem)
        self._write_state(state)

    def _set_problem(self, state: dict[str, Any], problem: CodexProblem) -> None:
        value: dict[str, Any] = {"code": problem.code, "at": self._clock()}
        if problem.retry_after is not None:
            value["retry_after"] = problem.retry_after
        if problem.http_status is not None:
            value["http_status"] = problem.http_status
        state["last_error"] = value

    def _file_lock(self) -> _CrossProcessLock:
        return _CrossProcessLock(
            self.auth_file.with_suffix(self.auth_file.suffix + ".lock"),
            clock=self._monotonic,
            sleep=self._sleep,
        )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": f"kimix-gui/{__version__}"},
                transport=self._transport,
            )
        return self._client

    def _raise_if_cancelled(self, operation_id: int) -> None:
        if operation_id in self._cancelled_operations:
            self._cancelled_operations.discard(operation_id)
            raise CodexAuthError(CodexProblem(PROBLEM_CANCELLED))

    @staticmethod
    def _json_object(response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise CodexAuthError(CodexProblem(PROBLEM_INVALID_RESPONSE)) from exc
        if not isinstance(payload, dict):
            raise CodexAuthError(CodexProblem(PROBLEM_INVALID_RESPONSE))
        return payload

    def _token_problem(self, response: httpx.Response) -> CodexProblem:
        code = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                raw = payload.get("error")
                if isinstance(raw, dict):
                    raw = raw.get("code") or raw.get("type")
                if isinstance(raw, str):
                    code = raw
        except ValueError:
            pass
        if response.status_code in (401, 403) and code not in _TERMINAL_TOKEN_ERRORS:
            code = PROBLEM_LOGIN_REQUIRED
        if response.status_code == 429:
            code = PROBLEM_RATE_LIMITED
        if response.status_code >= 500:
            code = PROBLEM_SERVER
        return CodexProblem(
            code or PROBLEM_INVALID_RESPONSE,
            _parse_retry_after(response.headers.get("Retry-After")),
            response.status_code,
        )

    def _http_problem(self, response: httpx.Response) -> CodexProblem:
        if response.status_code in (401, 403):
            return CodexProblem(PROBLEM_LOGIN_REQUIRED, http_status=response.status_code)
        if response.status_code == 429:
            return CodexProblem(
                PROBLEM_RATE_LIMITED,
                _parse_retry_after(response.headers.get("Retry-After")),
                response.status_code,
            )
        if response.status_code >= 500:
            return CodexProblem(PROBLEM_SERVER, http_status=response.status_code)
        return CodexProblem(PROBLEM_INVALID_RESPONSE, http_status=response.status_code)


def _build_authorization_url(*, redirect_uri: str, code_challenge: str, state: str) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": CODEX_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": BROWSER_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "kimix-gui",
        }
    )
    return f"{AUTHORIZATION_URL}?{query}"


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _single_query_value(params: Mapping[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if values is None or len(values) != 1:
        return None
    return values[0]


async def _write_callback_response(
    writer: asyncio.StreamWriter,
    status: int,
    *,
    close_window: bool = False,
) -> None:
    reasons = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        431: "Request Header Fields Too Large",
    }
    body = (
        b"<!doctype html><meta charset=utf-8><title>Kimix</title><script>window.close()</script>"
        if close_window
        else b""
    )
    headers = [
        f"HTTP/1.1 {status} {reasons.get(status, 'Bad Request')}",
        "Content-Type: text/html; charset=utf-8",
        f"Content-Length: {len(body)}",
        "Cache-Control: no-store",
        "X-Content-Type-Options: nosniff",
        "Connection: close",
        "",
        "",
    ]
    writer.write("\r\n".join(headers).encode("ascii") + body)
    await writer.drain()


def parse_model_catalog(payload: object) -> tuple[CodexModel, ...]:
    """Normalize the live catalog while retaining Codex-only subscription models."""

    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    candidates: list[CodexModel] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        if item.get("hidden") is True:
            continue
        visibility = item.get("visibility")
        if isinstance(visibility, str) and visibility.strip().lower() in {
            "hide",
            "hidden",
            "none",
        }:
            continue
        priority_value = item.get("priority")
        priority = (
            int(priority_value)
            if isinstance(priority_value, int | float) and not isinstance(priority_value, bool)
            else 10_000
        )
        profile = _profile_model(slug, priority=priority)
        display = item.get("display_name") or item.get("displayName")
        context = (
            _first_positive_int(
                item,
                "context_window",
                "context_window_size",
                "max_context_size",
                "max_context_window",
                "contextWindow",
            )
            or profile.max_context_size
        )
        max_tokens = _first_positive_int(
            item,
            "max_output_tokens",
            "max_tokens",
            "maxOutputTokens",
        )
        if max_tokens is None:
            max_tokens = profile.max_tokens
        modalities_value = _first_present(
            item,
            "input_modalities",
            "supported_input_modalities",
            "inputModalities",
        )
        modalities = (
            _string_values(
                modalities_value if modalities_value is not _MISSING else profile.input_modalities
            )
            or profile.input_modalities
        )
        efforts_value = _first_present(
            item,
            "supported_reasoning_levels",
            "supported_reasoning_efforts",
            "supportedReasoningEfforts",
        )
        efforts = (
            profile.reasoning_efforts
            if efforts_value is _MISSING
            else _reasoning_efforts(efforts_value)
        )
        default_effort = _reasoning_effort(
            _first_present(
                item,
                "default_reasoning_level",
                "default_reasoning_effort",
                "defaultReasoningEffort",
            )
        )
        if default_effort is None:
            default_effort = profile.default_reasoning_effort
        candidates.append(
            CodexModel(
                slug=slug,
                display_name=display.strip()
                if isinstance(display, str) and display.strip()
                else profile.display_name,
                max_context_size=context,
                max_tokens=max_tokens,
                input_modalities=modalities,
                reasoning_efforts=efforts,
                default_reasoning_effort=default_effort,
                priority=priority,
            )
        )
    candidates.sort(key=lambda model: (model.priority, model.slug))
    unique: dict[str, CodexModel] = {}
    for model in candidates:
        unique.setdefault(model.slug, model)
        if len(unique) == MAX_CACHED_MODELS:
            break
    return tuple(unique.values())


def _reasoning_efforts(value: object) -> tuple[str, ...]:
    efforts: list[str] = []
    if isinstance(value, dict):
        entries: object = [key for key, enabled in value.items() if enabled]
    else:
        entries = value
    if isinstance(entries, list | tuple):
        for entry in entries:
            effort: object = entry
            if isinstance(entry, dict):
                effort = _first_present(
                    entry,
                    "effort",
                    "reasoning_effort",
                    "reasoningEffort",
                )
            normalized = _reasoning_effort(effort)
            if normalized is None or normalized in efforts:
                continue
            efforts.append(normalized)
            if len(efforts) == 32:
                break
    return tuple(efforts)


def _reasoning_effort(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    effort = value.strip()
    if not effort or len(effort) > 64 or any(character.isspace() for character in effort):
        return None
    return effort


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        values = [key for key, enabled in value.items() if enabled]
    elif isinstance(value, list | tuple):
        values = value
    else:
        return ()
    return tuple(
        dict.fromkeys(item.strip() for item in values if isinstance(item, str) and item.strip())
    )


def _first_positive_int(item: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


_MISSING = object()


def _first_present(item: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if key in item:
            return item[key]
    return _MISSING


def _model_to_data(model: CodexModel) -> dict[str, Any]:
    return {
        "slug": model.slug,
        "display_name": model.display_name,
        "max_context_size": model.max_context_size,
        "max_tokens": model.max_tokens,
        "input_modalities": list(model.input_modalities),
        "reasoning_efforts": list(model.reasoning_efforts),
        "default_reasoning_effort": model.default_reasoning_effort,
        "priority": model.priority,
    }


def _model_from_data(item: Mapping[str, Any]) -> CodexModel | None:
    slug = item.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    display = item.get("display_name")
    priority_value = item.get("priority")
    priority = (
        int(priority_value)
        if isinstance(priority_value, int | float) and not isinstance(priority_value, bool)
        else 10_000
    )
    profile = _profile_model(slug, priority=priority)
    legacy_cache = "default_reasoning_effort" not in item
    if legacy_cache and slug in _CODEX_MODEL_PROFILES:
        return CodexModel(
            slug=slug,
            display_name=display if isinstance(display, str) else profile.display_name,
            max_context_size=profile.max_context_size,
            max_tokens=profile.max_tokens,
            input_modalities=profile.input_modalities,
            reasoning_efforts=profile.reasoning_efforts,
            default_reasoning_effort=profile.default_reasoning_effort,
            priority=priority,
        )
    efforts = (
        _reasoning_efforts(item.get("reasoning_efforts"))
        if "reasoning_efforts" in item
        else profile.reasoning_efforts
    )
    return CodexModel(
        slug=slug,
        display_name=display if isinstance(display, str) else profile.display_name,
        max_context_size=(
            _first_positive_int(item, "max_context_size") or profile.max_context_size
        ),
        max_tokens=_first_positive_int(item, "max_tokens") or profile.max_tokens,
        input_modalities=(_string_values(item.get("input_modalities")) or profile.input_modalities),
        reasoning_efforts=efforts,
        default_reasoning_effort=(
            _reasoning_effort(item.get("default_reasoning_effort"))
            or profile.default_reasoning_effort
        ),
        priority=priority,
    )


def _positive_number(value: object) -> float | None:
    if isinstance(value, int | float) and value > 0:
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _tokens_belong_to_same_account(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    expected_account = extract_chatgpt_account_id(expected)
    return bool(expected_account and expected_account == extract_chatgpt_account_id(actual))


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(60.0, float(value)))
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except TypeError, ValueError, OverflowError:
            return None
        return max(0.0, min(60.0, delay))


_DEFAULT_SERVICE: CodexAuthService | None = None


def default_codex_auth_service() -> CodexAuthService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = CodexAuthService()
    return _DEFAULT_SERVICE
