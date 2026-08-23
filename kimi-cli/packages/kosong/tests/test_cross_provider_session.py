"""Cross-provider session-replay tests.

A kimi-cli ``Session`` is shared across providers via client-side history
replay: the whole conversation is persisted and replayed on every turn, so any
provider must accept a history that was produced (persisted) under a different
provider. These tests feed one shared ``history`` through kimi /
openai_legacy / openai_responses / anthropic ``generate()`` and assert the wire
output shape per provider (sanitized tool-call ids, thinking round-trip, role
mapping, multimodal degradation, token-limit key portability).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import respx
from httpx import Response

from kosong.chat_provider.kimi import Kimi
from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import (
    AudioURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
    VideoURLPart,
)

# ``api_snapshot_tests.common`` lives in a sibling directory; make it importable
# regardless of pytest's import mode.
sys.path.insert(0, str(Path(__file__).parent / "api_snapshot_tests"))
from common import make_httpx2_client  # noqa: E402

LONG_TOOL_CALL_ID = "a" * 70
EXPECTED_LONG_TOOL_CALL_ID = "a" * 64  # truncated to the 64-char id budget


def _chat_response(model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _responses_response() -> dict[str, Any]:
    return {
        "id": "resp_test123",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "gpt-4.1",
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


async def _capture_kimi(
    history: list[Message], *, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=_chat_response("kimi-k2"))
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        if generation_kwargs:
            provider = provider.with_generation_kwargs(**generation_kwargs)
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        return json.loads(mock.calls.last.request.content.decode())


async def _capture_legacy(
    history: list[Message], *, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=_chat_response("gpt-4.1"))
        )
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        if generation_kwargs:
            provider = provider.with_generation_kwargs(**generation_kwargs)
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        return json.loads(mock.calls.last.request.content.decode())


async def _capture_responses(
    history: list[Message],
    *,
    model: str = "gpt-4.1",
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/responses").mock(return_value=Response(200, json=_responses_response()))
        provider = OpenAIResponses(model=model, api_key="test-key", stream=False)
        if generation_kwargs:
            provider = provider.with_generation_kwargs(**generation_kwargs)
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        return json.loads(mock.calls.last.request.content.decode())


async def _capture_anthropic(
    history: list[Message], *, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        if generation_kwargs:
            provider = provider.with_generation_kwargs(**generation_kwargs)
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        return json.loads(mock.calls.last.request.content.decode())


# ---------------------------------------------------------------------------
# 1. Tool-call id portability
# ---------------------------------------------------------------------------

TOOL_ID_HISTORY = [
    Message(role="user", content="Use the tools"),
    Message(
        role="assistant",
        content="I'll call them.",
        tool_calls=[
            ToolCall(
                id="Read:9", function=ToolCall.FunctionBody(name="read", arguments='{"path": "a"}')
            ),
            ToolCall(
                id="call_abc123",
                function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
            ),
            ToolCall(
                id=LONG_TOOL_CALL_ID,
                function=ToolCall.FunctionBody(name="mul", arguments='{"a": 4, "b": 5}'),
            ),
        ],
    ),
    Message(role="tool", content="r1", tool_call_id="Read:9"),
    Message(role="tool", content="r2", tool_call_id="call_abc123"),
    Message(role="tool", content="r3", tool_call_id=LONG_TOOL_CALL_ID),
]

EXPECTED_IDS = ["Read_9", "call_abc123", EXPECTED_LONG_TOOL_CALL_ID]


async def test_tool_id_portability_kimi():
    body = await _capture_kimi(TOOL_ID_HISTORY)
    messages = body["messages"]
    assistant = messages[1]
    assert [tc["id"] for tc in assistant["tool_calls"]] == EXPECTED_IDS
    # Every wire id is safe: [a-zA-Z0-9_-] and <= 64 chars.
    for tool_call in assistant["tool_calls"]:
        assert tool_call["id"] == tool_call["id"].translate(_safe_translate())
    assert [m["tool_call_id"] for m in messages[2:]] == EXPECTED_IDS


async def test_tool_id_portability_legacy():
    body = await _capture_legacy(TOOL_ID_HISTORY)
    messages = body["messages"]
    assistant = messages[1]
    assert [tc["id"] for tc in assistant["tool_calls"]] == EXPECTED_IDS
    assert [m["tool_call_id"] for m in messages[2:]] == EXPECTED_IDS


async def test_tool_id_portability_responses():
    body = await _capture_responses(TOOL_ID_HISTORY)
    input_items = body["input"]
    function_calls = [item for item in input_items if item.get("type") == "function_call"]
    outputs = [item for item in input_items if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in function_calls] == EXPECTED_IDS
    assert [item["call_id"] for item in outputs] == EXPECTED_IDS


async def test_tool_id_portability_anthropic():
    body = await _capture_anthropic(TOOL_ID_HISTORY)
    messages = body["messages"]
    assistant = messages[1]
    tool_use_ids = [b["id"] for b in assistant["content"] if b["type"] == "tool_use"]
    assert tool_use_ids == EXPECTED_IDS
    # Consecutive tool-result user messages are merged into a single user
    # message carrying all three tool_result blocks (Anthropic contract).
    user_messages = [m for m in messages if m["role"] == "user"][1:]
    assert len(user_messages) == 1
    tool_results = user_messages[0]["content"]
    assert all(b["type"] == "tool_result" for b in tool_results)
    assert [b["tool_use_id"] for b in tool_results] == EXPECTED_IDS


def test_tool_id_normalization_idempotent():
    """Replaying an already-normalized history must be a no-op: the original
    message objects are never mutated and ids stay byte-identical."""
    from kosong.contrib.chat_provider.common import normalize_tool_call_ids

    original = TOOL_ID_HISTORY
    once = normalize_tool_call_ids(original)
    twice = normalize_tool_call_ids(once)
    # Second pass returns the same sequence object (no rewrite needed).
    assert twice is once
    # The caller's history was not mutated.
    assert original[1].tool_calls[0].id == "Read:9"  # type: ignore[index]
    # Sanitized ids are stable across passes.
    ids_once = [tc.id for tc in once[1].tool_calls or []]
    ids_twice = [tc.id for tc in twice[1].tool_calls or []]
    assert ids_once == ids_twice == EXPECTED_IDS


def _safe_translate() -> dict[int, int]:
    return {ord(c): ord("_") for c in ":?/\\"}


# ---------------------------------------------------------------------------
# 2. Thinking (ThinkPart) round-trip
# ---------------------------------------------------------------------------


async def test_signed_thinkpart_round_trip():
    """Signed ThinkPart (anthropic origin): signature must be dropped by
    kimi/legacy (reasoning_content kept), kept as encrypted reasoning item by
    responses, kept as a `thinking` block by anthropic."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[ThinkPart(think="Let me think...", encrypted="sig_xyz"), TextPart(text="4.")],
        ),
        Message(role="user", content="Thanks!"),
    ]

    kimi_body = await _capture_kimi(history)
    kimi_assistant = kimi_body["messages"][1]
    assert kimi_assistant["reasoning_content"] == "Let me think..."
    assert "encrypted" not in kimi_assistant["content"][0]

    legacy_body = await _capture_legacy(history)
    legacy_assistant = legacy_body["messages"][1]
    assert legacy_assistant["reasoning_content"] == "Let me think..."

    responses_body = await _capture_responses(history)
    reasoning_items = [i for i in responses_body["input"] if i.get("type") == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"] == "sig_xyz"
    assert reasoning_items[0]["summary"] == [{"type": "summary_text", "text": "Let me think..."}]

    anthropic_body = await _capture_anthropic(history)
    anthropic_assistant = anthropic_body["messages"][1]
    thinking_blocks = [b for b in anthropic_assistant["content"] if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "Let me think..."
    assert thinking_blocks[0]["signature"] == "sig_xyz"


async def test_signatureless_thinkpart_round_trip():
    """Signature-less ThinkPart (kimi/openai origin): kept as
    reasoning_content by kimi/legacy, stripped silently by anthropic."""
    history = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content=[ThinkPart(think="Thinking..."), TextPart(text="Hello!")],
        ),
        Message(role="user", content="Bye"),
    ]

    kimi_body = await _capture_kimi(history)
    assert kimi_body["messages"][1]["reasoning_content"] == "Thinking..."

    legacy_body = await _capture_legacy(history)
    assert legacy_body["messages"][1]["reasoning_content"] == "Thinking..."

    anthropic_body = await _capture_anthropic(history)
    anthropic_assistant = anthropic_body["messages"][1]
    block_types = [b["type"] for b in anthropic_assistant["content"]]
    assert "thinking" not in block_types
    assert block_types == ["text"]


async def test_empty_thinkpart_preserved():
    """Empty ThinkPart: kimi/legacy preserve an empty reasoning_content
    (Moonshot 400 avoidance); responses emit an (empty) reasoning item; no
    provider crashes."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[ThinkPart(think=""), TextPart(text="4.")],
        ),
        Message(role="user", content="Thanks!"),
    ]

    kimi_body = await _capture_kimi(history)
    assert kimi_body["messages"][1]["reasoning_content"] == ""

    legacy_body = await _capture_legacy(history)
    assert legacy_body["messages"][1]["reasoning_content"] == ""

    responses_body = await _capture_responses(history)
    reasoning_items = [i for i in responses_body["input"] if i.get("type") == "reasoning"]
    assert len(reasoning_items) == 1

    anthropic_body = await _capture_anthropic(history)
    anthropic_assistant = anthropic_body["messages"][1]
    assert "thinking" not in [b["type"] for b in anthropic_assistant["content"]]


# ---------------------------------------------------------------------------
# 3. Role mapping
# ---------------------------------------------------------------------------


async def test_system_role_inside_history():
    """A `system`-role message inside the history must map per provider:
    anthropic `<system>` user wrapper; responses `developer` for OpenAI
    models; legacy/kimi keep `system`."""
    history = [
        Message(role="system", content="You are a system message inside history."),
        Message(role="user", content="Hi"),
    ]

    kimi_body = await _capture_kimi(history)
    assert kimi_body["messages"][0] == {
        "role": "system",
        "content": "You are a system message inside history.",
    }

    legacy_body = await _capture_legacy(history)
    assert legacy_body["messages"][0] == {
        "role": "system",
        "content": "You are a system message inside history.",
    }

    responses_body = await _capture_responses(history, model="gpt-4.1")
    assert responses_body["input"][0]["role"] == "developer"

    responses_body_custom = await _capture_responses(history, model="some-other-model")
    assert responses_body_custom["input"][0]["role"] == "system"

    anthropic_body = await _capture_anthropic(history)
    anthropic_user = anthropic_body["messages"][0]
    assert anthropic_user["role"] == "user"
    assert anthropic_user["content"][0]["type"] == "text"
    assert "<system>You are a system message inside history.</system>" in (
        anthropic_user["content"][0]["text"]
    )


# ---------------------------------------------------------------------------
# 4. Multimodal degradation
# ---------------------------------------------------------------------------


async def test_multimodal_parts_do_not_crash():
    """Video (ms://) / audio parts in a history persisted under kimi must
    degrade gracefully (skip or serialize) in every provider — never crash."""
    history = [
        Message(
            role="user",
            content=[
                TextPart(text="What's this?"),
                VideoURLPart(video_url=VideoURLPart.VideoURL(url="ms://video_abc123")),
                AudioURLPart(
                    audio_url=AudioURLPart.AudioURL(url="https://example.com/audio.mp3")
                ),
            ],
        ),
    ]

    # kimi: serialized as content parts (no crash)
    kimi_body = await _capture_kimi(history)
    kimi_content = kimi_body["messages"][0]["content"]
    assert any(part.get("type") == "video_url" for part in kimi_content)

    # legacy: serialized via model_dump (no crash)
    await _capture_legacy(history)

    # responses: video skipped, audio mapped to input_file (no crash)
    responses_body = await _capture_responses(history)
    responses_content = responses_body["input"][0]["content"]
    assert all(part["type"] != "video_url" for part in responses_content)

    # anthropic: unknown parts skipped (no crash)
    anthropic_body = await _capture_anthropic(history)
    anthropic_content = anthropic_body["messages"][0]["content"]
    assert all(part["type"] == "text" for part in anthropic_content)


async def test_legacy_auto_reasoning_effort_on_thinkpart_history():
    """Issue #1616: when the replayed history contains ThinkPart and a
    reasoning key is configured but no explicit effort, OpenAILegacy must
    auto-enable reasoning_effort=medium (cross-provider sessions from
    kimi/anthropic always carry ThinkParts when thinking was on)."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[ThinkPart(think="Thinking..."), TextPart(text="4.")],
        ),
        Message(role="user", content="Thanks!"),
    ]
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=_chat_response("gpt-4.1"))
        )
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# 5. Token-limit key portability
# ---------------------------------------------------------------------------


async def test_token_limit_key_portability():
    """The same token-limit kwargs replay per-provider: anthropic strips
    max_output_tokens/max_completion_tokens; legacy strips max_output_tokens;
    kimi converts max_tokens -> max_completion_tokens; responses keeps its
    native max_output_tokens."""

    anthropic_body = await _capture_anthropic(
        [Message(role="user", content="Hi")],
        generation_kwargs={
            "max_tokens": 5000,
            "max_output_tokens": 4096,
            "max_completion_tokens": 3000,
        },
    )
    assert anthropic_body["max_tokens"] == 5000
    assert "max_output_tokens" not in anthropic_body
    assert "max_completion_tokens" not in anthropic_body

    legacy_body = await _capture_legacy(
        [Message(role="user", content="Hi")],
        generation_kwargs={"max_tokens": 5000, "max_output_tokens": 4096},
    )
    assert legacy_body["max_tokens"] == 5000
    assert "max_output_tokens" not in legacy_body

    kimi_body = await _capture_kimi(
        [Message(role="user", content="Hi")],
        generation_kwargs={"max_tokens": 5000, "max_completion_tokens": 8000},
    )
    # max_completion_tokens wins when both are set; max_tokens is dropped.
    assert kimi_body["max_completion_tokens"] == 8000
    assert "max_tokens" not in kimi_body

    responses_body = await _capture_responses(
        [Message(role="user", content="Hi")],
        generation_kwargs={"max_output_tokens": 4096},
    )
    assert responses_body["max_output_tokens"] == 4096
