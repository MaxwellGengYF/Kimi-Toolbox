"""Tests for the OpenAI Responses chat provider.

Reproduces the failure observed against an OpenAI-compatible gateway
(``llmproxy``-style) that intermittently rejects the very reasoning
``encrypted_content`` blobs it issued:

    400 invalid_encrypted_content
    "The encrypted content gAAA... could not be verified.
     Reason: Encrypted content could not be decrypted or parsed."

Because the offending blob sits in the conversation history, every retry
(and every session restart) fails at the same step. The provider must
recover by stripping the unverifiable blobs and retrying once.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

import kosong
from kosong.chat_provider import APIStatusError
from kosong.contrib.chat_provider.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesStreamedMessage,
    _is_invalid_encrypted_content_error,
    _strip_reasoning_encrypted_content,
)
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the weather of a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def _make_provider(**kwargs: Any) -> OpenAIResponses:
    return OpenAIResponses(model="gpt-5.6-sol", api_key="test-key", **kwargs)


def _bad_request_400(*, code: str, message: str) -> openai.BadRequestError:
    body = {"error": {"message": message, "type": "invalid_request_error", "code": code}}
    request = httpx.Request("POST", "http://proxy.test/v1/responses")
    return openai.BadRequestError(
        f"Error code: 400 - {body!r}",
        response=httpx.Response(400, request=request),
        body=body,
    )


def _invalid_encrypted_content_error() -> openai.BadRequestError:
    return _bad_request_400(
        code="invalid_encrypted_content",
        message=(
            "The encrypted content gAAA...H7jK could not be verified. "
            "Reason: Encrypted content could not be decrypted or parsed."
        ),
    )


# -- fake SSE event stream mirroring the observed gateway wire format --------


def _reasoning_item_added(item_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(id=item_id, type="reasoning", summary=[], encrypted_content=None),
    )


def _reasoning_summary_text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text)


def _reasoning_summary_part_added() -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_summary_part.added")


def _reasoning_item_done(item_id: str, encrypted: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(id=item_id, type="reasoning", summary=[], encrypted_content=encrypted),
    )


def _function_call_added(item_id: str, call_id: str, name: str) -> SimpleNamespace:
    # The gateway announces the item with empty arguments...
    return SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(
            id=item_id, type="function_call", call_id=call_id, name=name, arguments=""
        ),
    )


def _function_call_delta(item_id: str, full_arguments: str) -> SimpleNamespace:
    # ...then sends the COMPLETE arguments in a single delta event...
    return SimpleNamespace(
        type="response.function_call_arguments.delta", item_id=item_id, delta=full_arguments
    )


def _function_call_done(item_id: str, call_id: str, name: str, arguments: str) -> SimpleNamespace:
    # ...and repeats the complete item once more when done.
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(
            id=item_id, type="function_call", call_id=call_id, name=name, arguments=arguments
        ),
    )


def _completed(input_tokens: int = 100, output_tokens: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            )
        ),
    )


def _proxy_style_tool_call_stream() -> Any:
    """The exact event sequence observed from the gateway for a turn with
    visible reasoning + two parallel tool calls."""

    async def gen():
        yield _reasoning_item_added("rs_1")
        yield _reasoning_summary_part_added()
        yield _reasoning_summary_text_delta("Planning ")
        yield _reasoning_summary_text_delta("tool calls")
        yield _reasoning_item_done("rs_1", encrypted="gAAAAencrypted-blob")
        yield _function_call_added("fc_1", "call_1", "get_weather")
        yield _function_call_delta("fc_1", '{"city":"Beijing"}')
        yield _function_call_done("fc_1", "call_1", "get_weather", '{"city":"Beijing"}')
        yield _function_call_added("fc_2", "call_2", "get_weather")
        yield _function_call_delta("fc_2", '{"city":"Shanghai"}')
        yield _function_call_done("fc_2", "call_2", "get_weather", '{"city":"Shanghai"}')
        yield _completed()

    return gen()


def _history_with_encrypted_reasoning() -> list[Message]:
    assistant = Message(
        role="assistant",
        content=[ThinkPart(think="I should call the tool.", encrypted="gAAAAstale-blob")],
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolCall.FunctionBody(
                    name="get_weather", arguments='{"city":"Beijing"}'
                ),
            )
        ],
    )
    tool_result = Message(role="tool", tool_call_id="call_1", content="Sunny")
    return [
        Message(role="user", content="Weather in Beijing?"),
        assistant,
        tool_result,
        Message(role="user", content="And Shanghai?"),
    ]


def _reasoning_items(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in payload if isinstance(item, dict) and item.get("type") == "reasoning"]


# ---------------------------------------------------------------------------
# Stream conversion: proxy-style events must not duplicate/corrupt tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_merges_proxy_style_events_without_duplicate_tool_calls() -> None:
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_proxy_style_tool_call_stream())  # type: ignore[method-assign]

    result = await kosong.generate(
        provider,
        "You are helpful.",
        [_WEATHER_TOOL],
        [Message(role="user", content="Weather in Beijing and Shanghai?")],
    )

    # Exactly two tool calls, each with intact arguments — no duplicates and
    # no double-appended argument deltas.
    tool_calls = result.message.tool_calls or []
    assert [(tc.id, tc.function.name, tc.function.arguments) for tc in tool_calls] == [
        ("call_1", "get_weather", '{"city":"Beijing"}'),
        ("call_2", "get_weather", '{"city":"Shanghai"}'),
    ]

    # Reasoning summary deltas merge into a single ThinkPart carrying the
    # encrypted blob from the reasoning `output_item.done` event.
    think_parts = [p for p in result.message.content if isinstance(p, ThinkPart)]
    assert len(think_parts) == 1
    assert think_parts[0].think == "Planning tool calls"
    assert think_parts[0].encrypted == "gAAAAencrypted-blob"

    assert result.usage is not None
    assert result.usage.input == 100
    assert result.usage.output == 50


# ---------------------------------------------------------------------------
# invalid_encrypted_content 400 recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_retries_once_without_encrypted_content_on_400() -> None:
    """A 400 `invalid_encrypted_content` caused by a stale gateway blob must
    trigger exactly one retry with the blobs stripped (summaries kept)."""
    provider = _make_provider()

    # Snapshot each request payload at call time: the retry sanitizes the
    # shared `inputs` list in place, so `call_args_list` references alone
    # would show the mutated state for both calls.
    captured_inputs: list[list[dict[str, Any]]] = []
    outcomes: list[Any] = [_invalid_encrypted_content_error(), _proxy_style_tool_call_stream()]

    async def fake_create(**kwargs: Any) -> Any:
        import copy

        captured_inputs.append(copy.deepcopy(kwargs["input"]))
        outcome = outcomes[len(captured_inputs) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    provider.client.responses.create = fake_create  # type: ignore[method-assign]

    result = await kosong.generate(
        provider, "You are helpful.", [_WEATHER_TOOL], _history_with_encrypted_reasoning()
    )

    assert len(captured_inputs) == 2

    first_input, retry_input = captured_inputs

    # The first attempt passes the blob back verbatim (normal round-trip).
    first_reasoning = _reasoning_items(first_input)
    assert len(first_reasoning) == 1
    assert first_reasoning[0].get("encrypted_content") == "gAAAAstale-blob"

    # The retry strips the unverifiable blob but keeps the visible summary.
    retry_reasoning = _reasoning_items(retry_input)
    assert len(retry_reasoning) == 1
    assert "encrypted_content" not in retry_reasoning[0]
    assert retry_reasoning[0]["summary"] == [
        {"type": "summary_text", "text": "I should call the tool."}
    ]

    # The retried stream is consumed normally.
    assert [tc.function.name for tc in result.message.tool_calls or []] == [
        "get_weather",
        "get_weather",
    ]


@pytest.mark.asyncio
async def test_generate_does_not_retry_unrelated_400() -> None:
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=_bad_request_400(code="context_length_exceeded", message="too many tokens")
    )

    with pytest.raises(APIStatusError) as exc_info:
        await kosong.generate(
            provider, "You are helpful.", [_WEATHER_TOOL], _history_with_encrypted_reasoning()
        )

    assert exc_info.value.status_code == 400
    assert provider.client.responses.create.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_generate_does_not_retry_when_no_encrypted_content_to_strip() -> None:
    """If the 400 cannot be attributed to a blob in the request, do not retry."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=_invalid_encrypted_content_error()
    )
    history = [Message(role="user", content="Hello")]

    with pytest.raises(APIStatusError) as exc_info:
        await kosong.generate(provider, "You are helpful.", [_WEATHER_TOOL], history)

    assert exc_info.value.status_code == 400
    assert provider.client.responses.create.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_generate_raises_when_retry_also_fails() -> None:
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_invalid_encrypted_content_error(), _invalid_encrypted_content_error()]
    )

    with pytest.raises(APIStatusError) as exc_info:
        await kosong.generate(
            provider, "You are helpful.", [_WEATHER_TOOL], _history_with_encrypted_reasoning()
        )

    assert exc_info.value.status_code == 400
    assert provider.client.responses.create.call_count == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Error classification / payload sanitizing helpers
# ---------------------------------------------------------------------------


class TestInvalidEncryptedContentErrorDetection:
    def test_matches_structured_code(self) -> None:
        assert _is_invalid_encrypted_content_error(_invalid_encrypted_content_error())

    def test_matches_message_heuristic_without_code(self) -> None:
        error = _bad_request_400(
            code="invalid_request_error",
            message="The encrypted content could not be verified.",
        )
        assert _is_invalid_encrypted_content_error(error)

    def test_rejects_non_400(self) -> None:
        request = httpx.Request("POST", "http://proxy.test/v1/responses")
        error = openai.InternalServerError(
            "Error code: 500 - boom",
            response=httpx.Response(500, request=request),
            body={"error": {"code": "invalid_encrypted_content"}},
        )
        assert not _is_invalid_encrypted_content_error(error)

    def test_rejects_unrelated_400(self) -> None:
        error = _bad_request_400(code="context_length_exceeded", message="too many tokens")
        assert not _is_invalid_encrypted_content_error(error)


class TestStripReasoningEncryptedContent:
    def test_strips_blobs_and_keeps_summaries(self) -> None:
        inputs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi", "type": "message"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "blob",
            },
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
        ]
        assert _strip_reasoning_encrypted_content(inputs) is True  # type: ignore[arg-type]
        assert "encrypted_content" not in inputs[1]
        assert inputs[1]["summary"] == [{"type": "summary_text", "text": "plan"}]

    def test_returns_false_when_nothing_to_strip(self) -> None:
        inputs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi", "type": "message"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
        ]
        assert _strip_reasoning_encrypted_content(inputs) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reasoning item conversion: null encrypted_content must be omitted
# ---------------------------------------------------------------------------


class TestReasoningItemConversion:
    def test_omits_encrypted_content_key_when_none(self) -> None:
        provider = _make_provider()
        message = Message(
            role="assistant",
            content=[ThinkPart(think="just thinking", encrypted=None)],
        )
        items = provider._convert_message(message)
        reasoning = _reasoning_items(items)
        assert len(reasoning) == 1
        assert "encrypted_content" not in reasoning[0]

    def test_includes_encrypted_content_when_present(self) -> None:
        provider = _make_provider()
        message = Message(
            role="assistant",
            content=[ThinkPart(think="just thinking", encrypted="blob")],
        )
        items = provider._convert_message(message)
        reasoning = _reasoning_items(items)
        assert len(reasoning) == 1
        assert reasoning[0]["encrypted_content"] == "blob"


# ---------------------------------------------------------------------------
# DeepSeek Responses API: raw chain-of-thought (``reasoning_text``) events
# ---------------------------------------------------------------------------
#
# DeepSeek's Responses implementation does not generate OpenAI-style reasoning
# ``summary`` events.  It streams the raw chain of thought as
# ``response.reasoning_text.delta`` / ``response.reasoning_text.done`` and
# attaches the full text to the reasoning item's ``content`` (``reasoning_text``
# parts) with ``summary=[]`` and ``encrypted_content=None``.  These tests
# reproduce that wire format (see https://api-docs.deepseek.com/zh-cn/guides/responses_api).


def _deepseek_reasoning_delta(item_id: str, delta: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_text.delta", item_id=item_id, delta=delta)


def _deepseek_reasoning_done(item_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_text.done", item_id=item_id, text=text)


def _deepseek_reasoning_item_done(item_id: str, text: str = "full text") -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(
            id=item_id,
            type="reasoning",
            summary=[],
            content=[SimpleNamespace(type="reasoning_text", text=text)],
            encrypted_content=None,
        ),
    )


def _output_text_delta(delta: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


@pytest.mark.asyncio
async def test_stream_emits_deepseek_reasoning_text_deltas() -> None:
    """DeepSeek streams CoT via ``response.reasoning_text.delta`` events; the
    provider must surface them as ThinkParts (and not emit an empty trailing
    block from ``output_item.done``)."""
    provider = _make_provider()

    async def gen():
        yield _deepseek_reasoning_delta("rs_1", "Step one. ")
        yield _deepseek_reasoning_delta("rs_1", "Step two.")
        yield _deepseek_reasoning_done("rs_1", "Step one. Step two.")
        yield _deepseek_reasoning_item_done("rs_1")
        yield _output_text_delta("Answer")
        yield _completed()

    provider.client.responses.create = AsyncMock(return_value=gen())  # type: ignore[method-assign]

    result = await kosong.generate(
        provider, "You are helpful.", [], [Message(role="user", content="Hi")]
    )

    think_parts = [p for p in result.message.content if isinstance(p, ThinkPart)]
    assert len(think_parts) == 1
    assert think_parts[0].think == "Step one. Step two."
    assert think_parts[0].encrypted is None
    assert [p.text for p in result.message.content if isinstance(p, TextPart)] == ["Answer"]


@pytest.mark.asyncio
async def test_stream_reasoning_text_done_fallback_not_duplicated() -> None:
    """A backend that only emits ``response.reasoning_text.done`` (full text)
    must yield the text exactly once — the subsequent ``output_item.done`` must
    not duplicate it."""
    provider = _make_provider()

    async def gen():
        yield _deepseek_reasoning_done("rs_1", "Only full text")
        yield _deepseek_reasoning_item_done("rs_1", "Only full text")
        yield _output_text_delta("Answer")
        yield _completed()

    provider.client.responses.create = AsyncMock(return_value=gen())  # type: ignore[method-assign]

    result = await kosong.generate(
        provider, "You are helpful.", [], [Message(role="user", content="Hi")]
    )

    think_parts = [p for p in result.message.content if isinstance(p, ThinkPart)]
    assert len(think_parts) == 1
    assert think_parts[0].think == "Only full text"


@pytest.mark.asyncio
async def test_stream_reasoning_item_done_content_fallback() -> None:
    """Backends that attach the reasoning text only to the completed reasoning
    item (no delta events) still yield a ThinkPart from ``output_item.done``."""
    provider = _make_provider()

    async def gen():
        yield _deepseek_reasoning_item_done("rs_1", "Attached at done")
        yield _output_text_delta("Answer")
        yield _completed()

    provider.client.responses.create = AsyncMock(return_value=gen())  # type: ignore[method-assign]

    result = await kosong.generate(
        provider, "You are helpful.", [], [Message(role="user", content="Hi")]
    )

    think_parts = [p for p in result.message.content if isinstance(p, ThinkPart)]
    assert len(think_parts) == 1
    assert think_parts[0].think == "Attached at done"


def test_non_stream_reads_reasoning_content() -> None:
    """Non-streaming responses carry the CoT in ``content`` (``reasoning_text``)
    with an empty ``summary`` for DeepSeek; the provider must read it."""

    async def collect(agen: Any) -> list[Any]:
        return [part async for part in agen]

    msg = object.__new__(OpenAIResponsesStreamedMessage)
    fake_response = SimpleNamespace(
        id="resp_1",
        usage=None,
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[],
                content=[SimpleNamespace(type="reasoning_text", text="Chain of thought")],
                encrypted_content=None,
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Visible")],
            ),
        ],
    )
    parts = asyncio.run(collect(msg._convert_non_stream_response(fake_response)))
    thinks = [p for p in parts if isinstance(p, ThinkPart)]
    assert [t.think for t in thinks] == ["Chain of thought"]
    assert [p.text for p in parts if isinstance(p, TextPart)] == ["Visible"]


def test_deepseek_round_trip_uses_content_not_summary() -> None:
    """ThinkParts in history must round-trip as plaintext ``content``
    (``reasoning_text``) for DeepSeek — ``summary`` / ``encrypted_content`` are
    unsupported there and would be silently dropped."""
    provider = OpenAIResponses(model="deepseek-v4-flash", api_key="test-key")
    message = Message(
        role="assistant",
        content=[ThinkPart(think="hidden reasoning", encrypted="blob")],
    )
    items = provider._convert_message(message)
    reasoning = _reasoning_items(items)
    assert len(reasoning) == 1
    item = reasoning[0]
    assert "summary" not in item
    assert "encrypted_content" not in item
    assert item["content"] == [{"type": "reasoning_text", "text": "hidden reasoning"}]


def test_openai_round_trip_keeps_summary_and_encrypted() -> None:
    """Non-DeepSeek backends keep the summary + encrypted_content round-trip."""
    provider = _make_provider()
    message = Message(
        role="assistant",
        content=[ThinkPart(think="summary text", encrypted="blob")],
    )
    items = provider._convert_message(message)
    reasoning = _reasoning_items(items)
    assert len(reasoning) == 1
    assert reasoning[0]["summary"] == [{"type": "summary_text", "text": "summary text"}]
    assert reasoning[0]["encrypted_content"] == "blob"
    assert "content" not in reasoning[0]


def test_deepseek_backend_detection() -> None:
    assert OpenAIResponses(model="deepseek-v4-flash", api_key="k")._is_deepseek_backend() is True
    assert (
        OpenAIResponses(
            model="gpt-5-codex", api_key="k", base_url="https://api.deepseek.com"
        )._is_deepseek_backend()
        is True
    )
    assert OpenAIResponses(model="gpt-5-codex", api_key="k")._is_deepseek_backend() is False
