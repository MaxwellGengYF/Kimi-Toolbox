"""Tests for the Xiaomi MiMo chat provider (tool-message image flattening)."""

import respx
from common import capture_request, make_chat_completion_response
from httpx import Response

from kosong.chat_provider.compat import Xiaomi
from kosong.message import ImageURLPart, Message, TextPart, ToolCall


def make_provider(**kwargs):
    return Xiaomi(model="mimo-v2-omni", api_key="test", stream=False, **kwargs)


async def test_xiaomi_tool_message_image_flattened_to_text():
    with respx.mock(base_url="https://api.xiaomimimo.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("mimo-v2-omni"))
        )
        provider = make_provider()
        history = [
            Message(role="user", content="What's in this image?"),
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(name="view_image", arguments="{}"),
                    )
                ],
            ),
            Message(
                role="tool",
                content=[
                    TextPart(text="Here is the image:"),
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")
                    ),
                ],
                tool_call_id="call_abc123",
            ),
        ]
        body = await capture_request(mock, provider, "", [], history)
        tool_message = body["messages"][-1]
        assert tool_message["tool_call_id"] == "call_abc123"
        assert tool_message["content"] == "Here is the image:\n[image: https://example.com/image.png]"


async def test_xiaomi_user_image_preserved():
    with respx.mock(base_url="https://api.xiaomimimo.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("mimo-v2-omni"))
        )
        provider = make_provider()
        history = [
            Message(
                role="user",
                content=[
                    TextPart(text="Describe this:"),
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")
                    ),
                ],
            )
        ]
        body = await capture_request(mock, provider, "", [], history)
        content = body["messages"][-1]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/image.png"


async def test_xiaomi_plain_tool_message_unchanged():
    with respx.mock(base_url="https://api.xiaomimimo.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("mimo-v2-omni"))
        )
        provider = make_provider()
        history = [
            Message(role="user", content="Add 2 and 3"),
            Message(
                role="assistant",
                content="Sure.",
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                    )
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ]
        body = await capture_request(mock, provider, "", [], history)
        assert body["messages"][-1]["content"] == "5"


async def test_xiaomi_supports_vision_tool_messages_flag():
    assert Xiaomi.supports_vision_tool_messages is False
