"""Tests for Anthropic provider error handling, especially httpx exception conversion.

These tests guard against httpx exceptions leaking through the Anthropic SDK
during streaming — the root cause of the evaluation zero-score bug where
httpx.ReadTimeout bypassed retry logic and crashed the process.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("anthropic", reason="Optional contrib dependency not installed")

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)

from kosong.chat_provider import (
    DEFAULT_MAX_RETRIES,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
    convert_httpx_error,
)
from kosong.contrib.chat_provider.anthropic import (
    Anthropic,
    AnthropicStreamedMessage,
    _convert_error,  # pyright: ignore[reportPrivateUsage]
)
from kosong.message import ImageURLPart, Message, TextPart, ThinkPart, ToolCall

# ---------------------------------------------------------------------------
# Shared convert_httpx_error (kosong.chat_provider)
# ---------------------------------------------------------------------------


class TestConvertHttpxError:
    """The shared convert_httpx_error utility must correctly map every httpx
    exception subclass to the corresponding kosong ChatProviderError."""

    @pytest.mark.parametrize(
        ("exc", "expected_type"),
        [
            (httpx.ReadTimeout("read timed out"), APITimeoutError),
            (httpx.ConnectTimeout("connect timed out"), APITimeoutError),
            (httpx.WriteTimeout("write timed out"), APITimeoutError),
            (httpx.PoolTimeout("pool timed out"), APITimeoutError),
            (httpx.NetworkError("connection reset"), APIConnectionError),
            (httpx.RemoteProtocolError("remote protocol error"), APIConnectionError),
            (httpx.LocalProtocolError("local protocol error"), ChatProviderError),
            (httpx.DecodingError("decode failed"), ChatProviderError),
        ],
        ids=[
            "ReadTimeout",
            "ConnectTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "NetworkError",
            "RemoteProtocolError",
            "LocalProtocolError",
            "DecodingError",
        ],
    )
    def test_httpx_error_mapping(
        self, exc: httpx.HTTPError, expected_type: type[ChatProviderError]
    ) -> None:
        assert isinstance(convert_httpx_error(exc), expected_type)

    def test_http_status_error(self) -> None:
        response = httpx.Response(
            502,
            request=httpx.Request("POST", "https://api.test"),
            headers={"retry-after": "10", "x-request-id": "req-httpx"},
        )
        exc = httpx.HTTPStatusError("bad gateway", request=response.request, response=response)
        err = convert_httpx_error(exc)
        assert isinstance(err, APIStatusError)
        assert err.status_code == 502
        assert err.headers == response.headers
        assert err.request_id == "req-httpx"
        assert err.retry_after == 10


# ---------------------------------------------------------------------------
# Anthropic-specific _convert_error
# ---------------------------------------------------------------------------


class TestAnthropicConvertError:
    """Anthropic's _convert_error must handle both AnthropicError and httpx.HTTPError,
    and must check APITimeoutError before APIConnectionError (inheritance order)."""

    def test_timeout_not_misclassified_as_connection(self) -> None:
        """AnthropicAPITimeoutError inherits from AnthropicAPIConnectionError.
        The conversion must check timeout FIRST to avoid misclassifying it."""
        err = _convert_error(AnthropicAPITimeoutError(request=None))  # pyright: ignore[reportArgumentType]
        assert type(err) is APITimeoutError

    def test_connection_error(self) -> None:
        err = _convert_error(AnthropicAPIConnectionError(request=None))  # pyright: ignore[reportArgumentType]
        assert isinstance(err, APIConnectionError)

    def test_delegates_httpx_to_shared_converter(self) -> None:
        """httpx errors should be delegated to convert_httpx_error."""
        err = _convert_error(httpx.ReadTimeout("stream timed out"))
        assert isinstance(err, APITimeoutError)

        err = _convert_error(httpx.NetworkError("connection reset"))
        assert isinstance(err, APIConnectionError)

    def test_api_status_error_preserves_response_headers(self) -> None:
        """Anthropic APIStatusError must propagate response headers so the soul
        layer can read ``Retry-After``."""
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(
            429,
            request=request,
            headers={"retry-after": "3", "x-request-id": "req-anthropic"},
        )
        err = AnthropicAPIStatusError(
            "rate limited", response=response, body=None
        )
        result = _convert_error(err)
        assert type(result) is APIStatusError
        assert result.status_code == 429
        assert result.headers == response.headers
        assert result.retry_after == 3


# ---------------------------------------------------------------------------
# Streaming error propagation (integration)
# ---------------------------------------------------------------------------


def _make_failing_stream(exc: Exception) -> AnthropicStreamedMessage:
    """Create an AnthropicStreamedMessage whose underlying async stream
    raises the given exception during iteration."""
    mock_stream = AsyncMock()

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise exc

    mock_manager = AsyncMock()
    mock_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_manager.__aexit__ = AsyncMock(return_value=False)
    mock_stream.__aiter__ = MagicMock(return_value=mock_stream)
    mock_stream.__anext__ = _raise
    return AnthropicStreamedMessage(mock_manager)


class TestStreamingErrorPropagation:
    """When httpx exceptions occur during stream consumption,
    AnthropicStreamedMessage._convert_stream_response must catch them
    and convert to kosong error types — not let them leak to the caller."""

    async def test_read_timeout(self) -> None:
        msg = _make_failing_stream(httpx.ReadTimeout("stream timed out after 600s"))
        with pytest.raises(APITimeoutError, match="stream timed out"):
            async for _ in msg:
                pass

    async def test_network_error(self) -> None:
        msg = _make_failing_stream(httpx.NetworkError("connection reset by peer"))
        with pytest.raises(APIConnectionError, match="connection reset"):
            async for _ in msg:
                pass

    async def test_connect_timeout(self) -> None:
        msg = _make_failing_stream(httpx.ConnectTimeout("connect timed out"))
        with pytest.raises(APITimeoutError, match="connect timed out"):
            async for _ in msg:
                pass


# ---------------------------------------------------------------------------
# Invalid JSON tool call arguments (graceful degradation)
# ---------------------------------------------------------------------------


class TestInvalidToolCallArguments:
    """When an assistant message contains a tool call with malformed JSON arguments,
    _convert_message must not crash.  It should emit an error text block and an
    empty tool_use input so the LLM sees the problem and can recover."""

    def test_invalid_json_returns_error_to_llm(self) -> None:
        provider = Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
        )
        message = Message(
            role="assistant",
            content=[TextPart(text="Let me call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_bad",
                    function=ToolCall.FunctionBody(
                        name="add", arguments='{"a": 1, "b": 2'  # missing closing }
                    ),
                )
            ],
        )
        result = provider._convert_message(message)
        assert result["role"] == "assistant"
        content = result["content"]
        # loads_relaxed repairs the broken JSON to a valid dict, so no error is emitted
        assert len(content) == 2  # text + tool_use
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me call a tool."
        assert content[1]["type"] == "tool_use"
        assert content[1]["input"] == {"a": 1, "b": 2}

    def test_non_dict_json_returns_error_to_llm(self) -> None:
        provider = Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
        )
        message = Message(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCall(
                    id="call_bad2",
                    function=ToolCall.FunctionBody(
                        name="add", arguments='[1, 2]'  # array, not object
                    ),
                )
            ],
        )
        result = provider._convert_message(message)
        content = result["content"]
        assert len(content) == 2  # error text + tool_use
        assert content[0]["type"] == "text"
        assert "must be a JSON object" in content[0]["text"]
        assert content[1]["type"] == "tool_use"
        assert content[1]["input"] == {}


class TestGracefulDataErrors:
    """Data validation errors during message conversion must return text blocks
    to the LLM instead of crashing the client."""

    def test_missing_tool_call_id_returns_error_to_llm(self) -> None:
        provider = Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
        )
        message = Message(
            role="tool",
            content=[TextPart(text="The result is 5")],
            tool_call_id=None,
        )
        result = provider._convert_message(message)
        assert result["role"] == "user"
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "missing `tool_call_id`" in content[0]["text"]
        assert "The result is 5" in content[0]["text"]

    def test_unsupported_part_in_tool_result_returns_error_to_llm(self) -> None:
        from kosong.contrib.chat_provider.anthropic import (
            _tool_result_message_to_block,  # pyright: ignore[reportPrivateUsage]
        )

        block = _tool_result_message_to_block(
            "call_123",
            [
                TextPart(text="Result text"),
                ThinkPart(think="thinking..."),  # unsupported in tool result
            ],
        )
        content = block["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Result text"
        assert content[1]["type"] == "text"
        assert "does not support" in content[1]["text"]

    def test_invalid_image_data_url_returns_error_to_llm(self) -> None:
        from kosong.contrib.chat_provider.anthropic import (
            _image_url_part_to_anthropic,  # pyright: ignore[reportPrivateUsage]
        )

        part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:invalid"))
        result = _image_url_part_to_anthropic(part)
        assert result["type"] == "text"
        assert "Invalid data URL for image" in result["text"]

    def test_unsupported_image_media_type_returns_error_to_llm(self) -> None:
        from kosong.contrib.chat_provider.anthropic import (
            _image_url_part_to_anthropic,  # pyright: ignore[reportPrivateUsage]
        )

        part = ImageURLPart(
            image_url=ImageURLPart.ImageURL(url="data:image/bmp;base64,QmFk")
        )
        result = _image_url_part_to_anthropic(part)
        assert result["type"] == "text"
        assert "Unsupported media type for base64 image" in result["text"]


# ---------------------------------------------------------------------------
# SDK retry policy defaults
# ---------------------------------------------------------------------------


class TestAnthropicRetryPolicy:
    """The Anthropic provider must configure a default SDK-level retry budget
    for transient errors such as 429 Rate Limit and 5xx server errors, while
    still allowing callers to override it explicitly."""

    def test_anthropic_applies_default_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class FakeAsyncAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "kosong.contrib.chat_provider.anthropic.AsyncAnthropic", FakeAsyncAnthropic
        )

        Anthropic(model="claude-sonnet-4-20250514", api_key="test-key", default_max_tokens=1024)

        assert captured["max_retries"] == DEFAULT_MAX_RETRIES

    def test_anthropic_respects_explicit_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class FakeAsyncAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "kosong.contrib.chat_provider.anthropic.AsyncAnthropic", FakeAsyncAnthropic
        )

        Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            max_retries=9,
        )

        assert captured["max_retries"] == 9
