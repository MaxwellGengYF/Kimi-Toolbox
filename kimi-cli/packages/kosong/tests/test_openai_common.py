import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import openai
import orjson
import pytest
from openai.types.chat import ChatCompletionChunk

from kosong.chat_provider import (
    DEFAULT_MAX_RETRIES,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
    openai_common,
)
from kosong.chat_provider.openai_common import (
    clamp_thinking_effort,
    convert_error,
    extract_reasoning_text,
    maybe_log_reasoning_content_error,
    reasoning_effort_to_thinking_effort,
    thinking_effort_to_reasoning_effort,
)
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy


class TestClampThinkingEffort:
    """Clamp thinking effort to a model's supported set."""

    @pytest.mark.parametrize(
        "effort,supported,expected",
        [
            ("off", None, "off"),
            ("off", {"low", "medium", "high"}, "off"),
            ("low", None, "low"),
            ("high", {"low", "medium", "high"}, "high"),
            ("max", {"low", "medium", "high"}, "high"),
            ("xhigh", {"low", "medium", "high"}, "high"),
            ("max", {"low", "medium", "high", "xhigh", "max"}, "max"),
            ("xhigh", {"low", "medium", "high", "xhigh", "max"}, "xhigh"),
            ("max", {"low", "medium", "high", "xhigh"}, "xhigh"),
            ("high", set(), "high"),
        ],
    )
    def test_clamp_thinking_effort(
        self, effort: str, supported: set[str] | None, expected: str
    ) -> None:
        assert (
            clamp_thinking_effort(
                effort,  # type: ignore[arg-type]
                supported,  # type: ignore[arg-type]
            )
            == expected
        )


class TestThinkingEffortMapping:
    """OpenAI's standard reasoning_effort accepts: none, minimal, low, medium,
    high, xhigh. Kosong's ThinkingEffort is: off, low, medium, high, xhigh, max.
    In this implementation OpenAI providers forward "max" verbatim (via
    extra_body) for backends that accept it, so the mapping preserves max and
    xhigh round-trips.
    """

    @pytest.mark.parametrize(
        "thinking_effort,expected_reasoning",
        [
            ("off", None),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            # xhigh must pass through — OpenAI supports it natively for
            # gpt-5.1-codex-max and later models.
            ("xhigh", "xhigh"),
            # max is forwarded as-is. The SDK's typed ReasoningEffort/Reasoning
            # models reject it, so providers send it through extra_body instead.
            ("max", "max"),
        ],
    )
    def test_thinking_to_reasoning(
        self, thinking_effort: str, expected_reasoning: str | None
    ) -> None:
        assert (
            thinking_effort_to_reasoning_effort(
                thinking_effort  # type: ignore[arg-type]
            )
            == expected_reasoning
        )

    @pytest.mark.parametrize(
        "reasoning_effort,expected_thinking",
        [
            (None, "off"),
            ("none", "off"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            # xhigh is a valid kosong level and a valid OpenAI level — round-trip.
            ("xhigh", "xhigh"),
        ],
    )
    def test_reasoning_to_thinking(
        self, reasoning_effort: str | None, expected_thinking: str
    ) -> None:
        assert (
            reasoning_effort_to_thinking_effort(
                reasoning_effort  # type: ignore[arg-type]
            )
            == expected_thinking
        )


def test_create_openai_client_does_not_inject_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_common, "AsyncOpenAI", FakeAsyncOpenAI)

    openai_common.create_openai_client(
        api_key="test-key",
        base_url="https://example.com/v1",
        client_kwargs={"timeout": 3},
    )

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["timeout"] == 3
    assert "max_retries" not in captured


def test_openai_legacy_applies_default_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_common, "AsyncOpenAI", FakeAsyncOpenAI)

    OpenAILegacy(model="gpt-4.1", api_key="test-key")

    assert captured["max_retries"] == DEFAULT_MAX_RETRIES


def test_openai_legacy_respects_explicit_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_common, "AsyncOpenAI", FakeAsyncOpenAI)

    OpenAILegacy(model="gpt-4.1", api_key="test-key", max_retries=7)

    assert captured["max_retries"] == 7


@pytest.mark.asyncio
async def test_retry_recovery_does_not_close_shared_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        http_client=http_client,
    )

    provider.on_retryable_error(APIConnectionError("Connection error."))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert provider.client._client is http_client  # type: ignore[reportPrivateUsage]
    assert http_client.is_closed is False
    await http_client.aclose()


# ---------------------------------------------------------------------------
# convert_error: openai.APIError (base class) handling
# ---------------------------------------------------------------------------

_DUMMY_REQUEST = httpx.Request("POST", "https://api.test")


class TestConvertErrorBaseAPIError:
    """openai.APIError (the base class, NOT APIConnectionError) must be
    correctly mapped when the error message indicates a network issue.

    This guards against the bug where streaming mid-flight disconnections
    raise ``openai.APIError("Network connection lost.")`` instead of
    ``openai.APIConnectionError``, and the converter falls through to
    the generic ``ChatProviderError`` — bypassing all retry/recovery logic.
    """

    @pytest.mark.parametrize(
        ("message", "expected_type"),
        [
            ("Network connection lost.", APIConnectionError),
            ("Connection error.", APIConnectionError),
            ("network error", APIConnectionError),
            ("disconnected from server", APIConnectionError),
            ("connection reset by peer", APIConnectionError),
            ("connection closed unexpectedly", APIConnectionError),
            ("Request timed out.", APITimeoutError),
            ("timed out", APITimeoutError),
            # Timeout must take priority over network when both patterns match.
            ("connection timed out", APITimeoutError),
            ("Something completely unrelated", ChatProviderError),
            ("Internal server error", ChatProviderError),
            # Bare "reset"/"closed" must NOT match — they are too broad
            # and could appear in non-network server messages.
            ("Your session has been reset", ChatProviderError),
            ("Stream closed by server due to policy violation", ChatProviderError),
        ],
        ids=[
            "network_connection_lost",
            "connection_error",
            "network_error",
            "disconnected",
            "connection_reset_by_peer",
            "connection_closed_unexpectedly",
            "request_timed_out",
            "timed_out",
            "connection_timed_out_timeout_priority",
            "unrelated_error",
            "internal_server_error",
            "bare_reset_no_match",
            "bare_closed_no_match",
        ],
    )
    def test_base_api_error_mapping(
        self, message: str, expected_type: type[ChatProviderError]
    ) -> None:
        err = openai.APIError(message=message, request=_DUMMY_REQUEST, body=None)
        result = convert_error(err)
        assert type(result) is expected_type, (
            f"Expected {expected_type.__name__} for message={message!r}, "
            f"got {type(result).__name__}"
        )

    def test_subclass_errors_still_match_first(self) -> None:
        """Existing specific error types must still be matched before
        the new base APIError branch."""
        # APIConnectionError should still match its own case
        conn_err = openai.APIConnectionError(request=_DUMMY_REQUEST)
        result = convert_error(conn_err)
        assert type(result) is APIConnectionError

        # APITimeoutError should still match its own case
        timeout_err = openai.APITimeoutError(request=_DUMMY_REQUEST)
        result = convert_error(timeout_err)
        assert type(result) is APITimeoutError

    def test_api_error_with_body_skips_heuristic(self) -> None:
        """SSE error events carry a body dict — they must NOT be
        heuristically reclassified, even if the message contains
        network keywords."""
        err = openai.APIError(
            message="Connection limit exceeded",
            request=_DUMMY_REQUEST,
            body={"error": {"message": "Connection limit exceeded", "type": "server_error"}},
        )
        result = convert_error(err)
        assert type(result) is ChatProviderError

    def test_api_response_validation_error_falls_through(self) -> None:
        """APIResponseValidationError has a body and must not be
        heuristically reclassified even if message contains keywords."""
        resp = httpx.Response(200, request=_DUMMY_REQUEST)
        err = openai.APIResponseValidationError(
            response=resp,
            body=None,
            message="connection field missing in response",
        )
        # APIResponseValidationError sets body from the response parsing,
        # but even with body=None the guard only applies to exact APIError;
        # however APIResponseValidationError IS an APIError subclass.
        # The key point: it should become ChatProviderError, not APIConnectionError.
        result = convert_error(err)
        assert type(result) is ChatProviderError

    def test_api_status_error_preserves_response_headers(self) -> None:
        """openai.APIStatusError must propagate response headers so the soul
        layer can read ``Retry-After``."""
        response = httpx.Response(
            429,
            request=_DUMMY_REQUEST,
            headers={"retry-after": "7", "x-request-id": "req-openai"},
        )
        err = openai.APIStatusError(
            response=response,
            body={"error": {"message": "rate limited"}},
            message="rate limited",
        )
        result = convert_error(err)
        assert type(result) is APIStatusError
        assert result.status_code == 429
        assert result.request_id == "req-openai"
        assert result.headers == response.headers
        assert result.retry_after == 7


# ---------------------------------------------------------------------------
# missing reasoning_content 400 logging
# ---------------------------------------------------------------------------


class TestMaybeLogReasoningContentError:
    """Debug logging for the Moonshot/Kimi 400 that occurs when thinking-mode
    reasoning_content is not passed back to the API.
    """

    _DUMMY_REQUEST = httpx.Request("POST", "https://api.test")

    @pytest.fixture
    def log_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _make_api_status_error(
        self,
        status_code: int = 400,
        message: str = "The reasoning_content in the thinking mode must be passed back to the API.",
        body: dict[str, Any] | None = None,
    ) -> openai.APIStatusError:
        response = httpx.Response(status_code, request=self._DUMMY_REQUEST)
        return openai.APIStatusError(
            response=response,
            body=body
            or {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "400001",
                }
            },
            message=message,
        )

    def test_logs_target_400_error(self, log_dir: Path) -> None:
        err = self._make_api_status_error()
        maybe_log_reasoning_content_error(
            err,
            provider_name="openai",
            model="gpt-test",
            messages=[{"role": "user", "content": "hi"}],
            generation_kwargs={"temperature": 0.5},
        )
        log_path = log_dir / "error.log"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = orjson.loads(lines[0])
        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-test"
        assert entry["error"]["status_code"] == 400
        assert entry["messages"] == [{"role": "user", "content": "hi"}]
        assert entry["generation_kwargs"] == {"temperature": 0.5}

    @pytest.mark.parametrize(
        "message",
        [
            "The reasoning_content in the thinking mode must be passed back to the API.",
            "The `reasoning_content` in the thinking mode must be passed back to the API.",
        ],
    )
    def test_matches_various_message_forms(self, log_dir: Path, message: str) -> None:
        err = self._make_api_status_error(message=message)
        maybe_log_reasoning_content_error(
            err,
            provider_name="openai",
            model="gpt-test",
            messages=[],
            generation_kwargs={},
        )
        assert (log_dir / "error.log").exists()

    def test_does_not_log_other_400_errors(self, log_dir: Path) -> None:
        err = self._make_api_status_error(
            message="Bad request",
            body={"error": {"message": "Bad request", "type": "invalid_request_error"}},
        )
        maybe_log_reasoning_content_error(
            err,
            provider_name="openai",
            model="gpt-test",
            messages=[],
            generation_kwargs={},
        )
        assert not (log_dir / "error.log").exists()

    def test_does_not_log_non_400_status(self, log_dir: Path) -> None:
        err = self._make_api_status_error(
            status_code=401,
            message="The reasoning_content in the thinking mode must be passed back to the API.",
        )
        maybe_log_reasoning_content_error(
            err,
            provider_name="openai",
            model="gpt-test",
            messages=[],
            generation_kwargs={},
        )
        assert not (log_dir / "error.log").exists()

    def test_does_not_log_non_api_status_error(self, log_dir: Path) -> None:
        err = openai.APIConnectionError(request=self._DUMMY_REQUEST)
        maybe_log_reasoning_content_error(
            err,
            provider_name="openai",
            model="gpt-test",
            messages=[],
            generation_kwargs={},
        )
        assert not (log_dir / "error.log").exists()


# ---------------------------------------------------------------------------
# Streaming error propagation (integration)
# ---------------------------------------------------------------------------


class TestOpenAIStreamingErrorPropagation:
    """When openai.APIError is raised during OpenAI stream consumption,
    _convert_stream_response must convert it to the correct kosong error type.

    This is the exact scenario from the bug: streaming for ~33 minutes,
    then the SSE connection drops and the SDK raises
    openai.APIError("Network connection lost.") — which must become
    APIConnectionError so that retry/recovery logic triggers.
    """

    async def test_base_api_error_becomes_connection_error(self) -> None:
        """openai.APIError("Network connection lost.") during streaming
        must surface as kosong APIConnectionError."""
        from kosong.contrib.chat_provider.openai_legacy import OpenAILegacyStreamedMessage

        async def _failing_stream() -> Any:
            raise openai.APIError(
                message="Network connection lost.",
                request=_DUMMY_REQUEST,
                body=None,
            )
            yield  # make this an async generator  # noqa: RUF027

        msg = OpenAILegacyStreamedMessage(_failing_stream(), reasoning_key=None)  # type: ignore[arg-type]
        with pytest.raises(APIConnectionError, match="Network connection lost"):
            async for _ in msg:
                pass


@pytest.mark.asyncio
async def test_openai_compatible_provider_aclose_closes_http_client() -> None:
    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    http_client = provider.client._client
    assert isinstance(http_client, httpx.AsyncClient)
    assert http_client.is_closed is False

    await provider.aclose()

    assert http_client.is_closed is True


@pytest.mark.asyncio
async def test_openai_compatible_provider_aclose_swallows_event_loop_closed() -> None:
    class FakeClient:
        async def close(self) -> None:
            raise RuntimeError("Event loop is closed")

    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    provider.client = FakeClient()  # type: ignore[assignment]

    # Should not raise.
    await provider.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_provider_aclose_swallows_cancelled_error() -> None:
    class FakeClient:
        async def close(self) -> None:
            raise asyncio.CancelledError()

    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    provider.client = FakeClient()  # type: ignore[assignment]

    # Should not raise.
    await provider.aclose()


# ---------------------------------------------------------------------------
# Tolerant SSE streaming: empty/invalid data events must not kill the stream
# ---------------------------------------------------------------------------
# Regression for: ``orjson.JSONDecodeError: Expecting value: line 1
# column 1 (char 0)`` raised inside ``openai._streaming.AsyncStream.__stream__``
# (``sse.json()``) when a backend (Moonshot/Kimi during long compaction
# requests) emits a keep-alive SSE event whose ``data:`` payload is empty.


def _sse_chunk(
    *, content: str | None = None, role: str | None = None, finish_reason: str | None = None
) -> str:
    """Build a chat.completion.chunk JSON payload for one SSE event."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return orjson.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    ).decode()


class _ChunkedResponse(httpx.Response):
    """httpx.Response that streams its body in the given byte chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(200, request=_DUMMY_REQUEST)
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _build_message(chunks: list[bytes]) -> Any:
    """Build a KimiStreamedMessage backed by a chunked SSE response."""
    from kosong.chat_provider.kimi import KimiStreamedMessage

    client = openai.AsyncOpenAI(api_key="test-key", base_url="https://api.test")
    response = _ChunkedResponse(chunks)
    stream = openai.AsyncStream(
        cast_to=ChatCompletionChunk,
        response=response,
        client=client,
    )
    return KimiStreamedMessage(stream), response


async def _collect_texts(message: Any) -> list[str]:
    from kosong.message import TextPart

    parts = [part async for part in message]
    return [p.text for p in parts if isinstance(p, TextPart)]


@pytest.mark.asyncio
async def test_tolerant_stream_skips_empty_data_events() -> None:
    """Keep-alive ``data:`` lines with no payload must be skipped."""
    body = (
        f"data: {_sse_chunk(role='assistant')}\n\n"
        "data:\n\n"  # <-- empty data event (the reported bug)
        "data: \n\n"  # <-- whitespace-only data event
        f"data: {_sse_chunk(content='Hello')}\n\n"
        "data: [DONE]\n\n"
    )
    # Split mid-line so events span arbitrary network chunk boundaries.
    mid = len(body) // 2
    message, _ = _build_message([body[:mid].encode(), body[mid:].encode()])
    assert await _collect_texts(message) == ["Hello"]


@pytest.mark.asyncio
async def test_tolerant_stream_skips_invalid_json_data() -> None:
    body = (
        f"data: {_sse_chunk(content='a')}\n\n"
        "data: this is not json\n\n"
        f"data: {_sse_chunk(content='b')}\n\n"
        "data: [DONE]\n\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["a", "b"]


@pytest.mark.asyncio
async def test_tolerant_stream_skips_non_object_payload() -> None:
    body = (
        f"data: {_sse_chunk(content='a')}\n\n"
        "data: [1, 2, 3]\n\n"
        f"data: {_sse_chunk(content='b')}\n\n"
        "data: [DONE]\n\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["a", "b"]


@pytest.mark.asyncio
async def test_tolerant_stream_stops_at_done() -> None:
    body = (
        f"data: {_sse_chunk(content='only')}\n\n"
        "data: [DONE]\n\n"
        f"data: {_sse_chunk(content='ignored')}\n\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["only"]


@pytest.mark.asyncio
async def test_tolerant_stream_raises_on_error_payload() -> None:
    """SSE error payloads must surface as ChatProviderError (via convert_error),
    not as a raw JSONDecodeError."""
    body = (
        f"data: {_sse_chunk(content='a')}\n\n"
        'data: {"error": {"message": "boom", "type": "server_error"}}\n\n'
    )
    message, _ = _build_message([body.encode()])
    with pytest.raises(ChatProviderError, match="boom"):
        async for _ in message:
            pass


@pytest.mark.asyncio
async def test_tolerant_stream_closes_response() -> None:
    body = f"data: {_sse_chunk(content='x')}\n\n" "data: [DONE]\n\n"
    message, response = _build_message([body.encode()])
    async for _ in message:
        pass
    assert response.closed is True


@pytest.mark.asyncio
async def test_tolerant_stream_handles_crlf_and_multiple_events_per_chunk() -> None:
    body = (
        f"data: {_sse_chunk(content='x')}\r\n\r\n"
        f"data: {_sse_chunk(content='y')}\r\n\r\n"
        "data: [DONE]\r\n\r\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["x", "y"]


@pytest.mark.asyncio
async def test_tolerant_stream_normalizes_unknown_finish_reason() -> None:
    """Non-standard finish_reason values such as ``unexpected_state`` must be
    treated as ``None`` so the stream continues instead of raising a pydantic
    validation error.
    """
    body = (
        f"data: {_sse_chunk(content='Hello', finish_reason='unexpected_state')}\n\n"
        "data: [DONE]\n\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["Hello"]


@pytest.mark.asyncio
async def test_tolerant_stream_concise_validation_error() -> None:
    """Validation errors that cannot be repaired must surface as a concise
    ``ChatProviderError`` rather than the full pydantic traceback.
    """
    body = (
        f"data: {_sse_chunk(content='Hello')}\n\n"
        'data: {"object": "chat.completion.chunk", "choices": '
        '[{"index": 0, "delta": "not-a-dict"}]}\n\n'
    )
    message, _ = _build_message([body.encode()])
    with pytest.raises(ChatProviderError, match="Backend returned an invalid chat stream chunk"):
        async for _ in message:
            pass


@pytest.mark.asyncio
async def test_kimi_streamed_message_survives_empty_sse_event() -> None:
    """End-to-end regression: an empty ``data:`` keep-alive event must not turn
    a valid compaction stream into a JSONDecodeError."""
    body = (
        f"data: {_sse_chunk(role='assistant')}\n\n"
        "data:\n\n"
        f"data: {_sse_chunk(content='Compaction summary')}\n\n"
        "data: [DONE]\n\n"
    )
    message, _ = _build_message([body.encode()])
    assert await _collect_texts(message) == ["Compaction summary"]


class TestExtractReasoningText:
    """Fallback extraction of reasoning text across backend-specific fields."""

    def test_prefers_configured_key(self) -> None:
        class Msg:
            reasoning_content = "deepseek thinking"
            reasoning = "commandcode thinking"

        assert extract_reasoning_text(Msg(), "reasoning_content") == "deepseek thinking"

    def test_falls_back_to_reasoning_field(self) -> None:
        class Msg:
            reasoning = "commandcode thinking"

        # reasoning_key defaults to reasoning_content in create_llm; the
        # Command Code API only sends `reasoning`, so it must fall back.
        assert extract_reasoning_text(Msg(), "reasoning_content") == "commandcode thinking"

    def test_falls_back_to_reasoning_details_blocks(self) -> None:
        class Msg:
            reasoning_details = [
                {"type": "reasoning.text", "text": "step one"},
                {"type": "reasoning.text", "text": " step two"},
            ]

        assert (
            extract_reasoning_text(Msg(), "reasoning_content")
            == "step one step two"
        )

    def test_reasoning_details_accepts_object_blocks(self) -> None:
        class Block:
            text = "obj block"

        class Msg:
            reasoning_details = [Block()]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "obj block"

    def test_empty_string_preserved(self) -> None:
        class Msg:
            reasoning_content = ""

        assert extract_reasoning_text(Msg(), "reasoning_content") == ""

    def test_disabled_when_key_is_empty_string(self) -> None:
        class Msg:
            reasoning = "thinking text"

        # An explicit empty reasoning_key disables reasoning round-tripping.
        assert extract_reasoning_text(Msg(), "") is None

    def test_none_when_no_field_present(self) -> None:
        class Msg:
            content = "hi"

        assert extract_reasoning_text(Msg(), "reasoning_content") is None

    def test_empty_configured_key_does_not_shadow_reasoning(self) -> None:
        """A present-but-empty higher-priority field must not hide real text.

        Some gateways (OpenRouter-style normalizers) emit
        ``reasoning_content: ""`` alongside a populated ``reasoning``; the
        empty string used to win and the real reasoning was silently dropped.
        """
        class Msg:
            reasoning_content = ""
            reasoning = "real thinking"

        assert extract_reasoning_text(Msg(), "reasoning_content") == "real thinking"

    def test_empty_reasoning_does_not_shadow_reasoning_details(self) -> None:
        class Msg:
            reasoning = ""
            reasoning_details = [{"type": "reasoning.text", "text": "details text"}]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "details text"

    def test_empty_string_still_returned_when_everything_empty(self) -> None:
        """Round-trip preservation: all fields empty -> empty string, not None."""
        class Msg:
            reasoning_content = ""
            reasoning = ""

        assert extract_reasoning_text(Msg(), "reasoning_content") == ""

    def test_reasoning_details_summary_blocks(self) -> None:
        """OpenRouter-style ``reasoning.summary`` blocks carry text under
        ``summary`` instead of ``text``."""
        class Msg:
            reasoning_details = [
                {"type": "reasoning.summary", "summary": "summary text"},
            ]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "summary text"

    def test_reasoning_details_encrypted_only_returns_none(self) -> None:
        """``reasoning.encrypted`` blocks carry a signature, not text — they
        must not surface as visible thinking."""
        class Msg:
            reasoning_details = [
                {"type": "reasoning.encrypted", "data": "signature-bytes"},
            ]

        assert extract_reasoning_text(Msg(), "reasoning_content") is None

    def test_reasoning_details_mixed_text_and_encrypted(self) -> None:
        class Msg:
            reasoning_details = [
                {"type": "reasoning.text", "text": "visible"},
                {"type": "reasoning.encrypted", "data": "signature-bytes"},
            ]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "visible"

    def test_reasoning_field_as_block_list(self) -> None:
        class Msg:
            reasoning = [{"text": "block "}, {"text": "list"}]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "block list"

    def test_object_block_with_summary_only(self) -> None:
        class Block:
            text = None
            summary = "obj summary"

        class Msg:
            reasoning_details = [Block()]

        assert extract_reasoning_text(Msg(), "reasoning_content") == "obj summary"

    def test_configured_key_matching_fallback_name_tried_once(self) -> None:
        """reasoning_key="reasoning" must not double-count the same field."""
        class Msg:
            reasoning = "single pass"

        assert extract_reasoning_text(Msg(), "reasoning") == "single pass"
