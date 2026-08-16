"""Snapshot tests for Google GenAI (Gemini) chat provider."""

import json
from typing import Any

import pytest
import respx
from common import COMMON_CASES, Case, run_test_cases
from httpx import Response
from inline_snapshot import snapshot

pytest.importorskip("google.genai", reason="Optional contrib dependency not installed")

from google.genai import _api_client

from kosong.message import Message, TextPart, ToolCall

# Force google-genai to use httpx so respx can mock requests.
_api_client.has_aiohttp = False

from kosong.contrib.chat_provider.google_genai import (  # noqa: E402
    GoogleGenAI,
    GoogleGenAIStreamedMessage,
)


def make_response() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
        "modelVersion": "gemini-2.5-flash",
    }


TEST_CASES: dict[str, Case] = {
    # Google GenAI doesn't support image_url in the same way, use subset of common cases
    **{k: v for k, v in COMMON_CASES.items() if "image" not in k},
    "tool_call_with_thought_signature": {
        "history": [
            Message(role="user", content="Add 2 and 3"),
            Message(
                role="assistant",
                content=[TextPart(text="I'll add those.")],
                tool_calls=[
                    ToolCall(
                        id="add_call_sig",
                        function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                        extras={"thought_signature_b64": "dGhvdWdodF9zaWduYXR1cmVfZGF0YQ=="},
                    )
                ],
            ),
        ],
    },
}


async def test_google_genai_message_conversion():
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        provider = GoogleGenAI(model="gemini-2.5-flash", api_key="test-key", stream=False)
        results = await run_test_cases(
            mock, provider, TEST_CASES, ("contents", "systemInstruction", "tools")
        )

        assert results == snapshot(
            {
                "simple_user_message": {
                    "contents": [{"parts": [{"text": "Hello!"}], "role": "user"}],
                    "systemInstruction": {
                        "parts": [{"text": "You are helpful."}],
                        "role": "user",
                    },
                },
                "multi_turn_conversation": {
                    "contents": [
                        {"parts": [{"text": "What is 2+2?"}], "role": "user"},
                        {"parts": [{"text": "2+2 equals 4."}], "role": "model"},
                        {"parts": [{"text": "And 3+3?"}], "role": "user"},
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
                "multi_turn_with_system": {
                    "contents": [
                        {"parts": [{"text": "What is 2+2?"}], "role": "user"},
                        {"parts": [{"text": "2+2 equals 4."}], "role": "model"},
                        {"parts": [{"text": "And 3+3?"}], "role": "user"},
                    ],
                    "systemInstruction": {
                        "parts": [{"text": "You are a math tutor."}],
                        "role": "user",
                    },
                },
                "tool_definition": {
                    "contents": [{"parts": [{"text": "Add 2 and 3"}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "name": "add",
                                    "description": "Add two integers.",
                                    "parameters_json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "integer", "description": "First number"},
                                            "b": {
                                                "type": "integer",
                                                "description": "Second number",
                                            },
                                        },
                                        "required": ["a", "b"],
                                    },
                                },
                                {
                                    "description": "Multiply two integers.",
                                    "name": "multiply",
                                    "parameters_json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "integer", "description": "First number"},
                                            "b": {
                                                "type": "integer",
                                                "description": "Second number",
                                            },
                                        },
                                        "required": ["a", "b"],
                                    },
                                },
                            ]
                        }
                    ],
                },
                "tool_call": {
                    "contents": [
                        {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll add those numbers for you."},
                                {
                                    "functionCall": {
                                        "args": {"a": 2, "b": 3},
                                        "name": "add",
                                    }
                                },
                            ],
                            "role": "model",
                        },
                        {
                            "parts": [
                                {
                                    "functionResponse": {
                                        "parts": [],
                                        "name": "add",
                                        "response": {"output": "5"},
                                    }
                                }
                            ],
                            "role": "user",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
                "parallel_tool_calls": {
                    "contents": [
                        {"parts": [{"text": "Calculate 2+3 and 4*5"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll calculate both."},
                                {
                                    "functionCall": {
                                        "name": "add",
                                        "args": {"a": 2, "b": 3},
                                    }
                                },
                                {
                                    "functionCall": {
                                        "name": "multiply",
                                        "args": {"a": 4, "b": 5},
                                    }
                                },
                            ],
                            "role": "model",
                        },
                        {
                            "parts": [
                                {
                                    "functionResponse": {
                                        "parts": [],
                                        "name": "add",
                                        "response": {
                                            "output": "<system-reminder>This is a system reminder"
                                            "</system-reminder>5"
                                        },
                                    }
                                },
                                {
                                    "functionResponse": {
                                        "parts": [],
                                        "name": "multiply",
                                        "response": {
                                            "output": "<system-reminder>This is a system reminder"
                                            "</system-reminder>20"
                                        },
                                    }
                                },
                            ],
                            "role": "user",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "description": "Add two integers.",
                                    "name": "add",
                                    "parameters_json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "integer", "description": "First number"},
                                            "b": {
                                                "type": "integer",
                                                "description": "Second number",
                                            },
                                        },
                                        "required": ["a", "b"],
                                    },
                                },
                                {
                                    "description": "Multiply two integers.",
                                    "name": "multiply",
                                    "parameters_json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "integer", "description": "First number"},
                                            "b": {
                                                "type": "integer",
                                                "description": "Second number",
                                            },
                                        },
                                        "required": ["a", "b"],
                                    },
                                },
                            ]
                        }
                    ],
                },
                "tool_call_with_thought_signature": {
                    "contents": [
                        {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll add those."},
                                {
                                    "functionCall": {
                                        "name": "add",
                                        "args": {"a": 2, "b": 3},
                                    },
                                    "thoughtSignature": "dGhvdWdodF9zaWduYXR1cmVfZGF0YQ==",
                                },
                            ],
                            "role": "model",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
            }
        )


async def test_google_genai_vertexai_message_conversion():
    with respx.mock(base_url="https://aiplatform.googleapis.com") as mock:
        mock.route(
            method="POST",
            path__regex=r"/v1beta1/publishers/google/models/gemini-3-pro-preview:generateContent",
        ).mock(return_value=Response(200, json=make_response()))
        provider = GoogleGenAI(
            model="gemini-3-pro-preview",
            api_key="test-key",
            stream=False,
            vertexai=True,
        )
        results = await run_test_cases(
            mock, provider, TEST_CASES, ("contents", "systemInstruction", "tools")
        )
        assert results == snapshot(
            {
                "simple_user_message": {
                    "contents": [{"parts": [{"text": "Hello!"}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": "You are helpful."}], "role": "user"},
                },
                "multi_turn_conversation": {
                    "contents": [
                        {"parts": [{"text": "What is 2+2?"}], "role": "user"},
                        {"parts": [{"text": "2+2 equals 4."}], "role": "model"},
                        {"parts": [{"text": "And 3+3?"}], "role": "user"},
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
                "multi_turn_with_system": {
                    "contents": [
                        {"parts": [{"text": "What is 2+2?"}], "role": "user"},
                        {"parts": [{"text": "2+2 equals 4."}], "role": "model"},
                        {"parts": [{"text": "And 3+3?"}], "role": "user"},
                    ],
                    "systemInstruction": {
                        "parts": [{"text": "You are a math tutor."}],
                        "role": "user",
                    },
                },
                "tool_definition": {
                    "contents": [{"parts": [{"text": "Add 2 and 3"}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "description": "Add two integers.",
                                    "name": "add", "parameters_json_schema": {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "First number"},
        "b": {"type": "integer", "description": "Second number"},
    },
    "required": ["a", "b"],
}},
                                {
                                    "description": "Multiply two integers.",
                                    "name": "multiply", "parameters_json_schema": {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "First number"},
        "b": {"type": "integer", "description": "Second number"},
    },
    "required": ["a", "b"],
}},
                            ]
                        }
                    ],
                },
                "tool_call": {
                    "contents": [
                        {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll add those numbers for you."},
                                {"functionCall": {"args": {"a": 2, "b": 3}, "name": "add"}},
                            ],
                            "role": "model",
                        },
                        {
                            "parts": [
                                {
                                    "functionResponse": {
                                        "parts": [],
                                        "name": "add",
                                        "response": {"output": "5"},
                                    }
                                }
                            ],
                            "role": "user",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
                "parallel_tool_calls": {
                    "contents": [
                        {"parts": [{"text": "Calculate 2+3 and 4*5"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll calculate both."},
                                {"functionCall": {"args": {"a": 2, "b": 3}, "name": "add"}},
                                {"functionCall": {"args": {"a": 4, "b": 5}, "name": "multiply"}},
                            ],
                            "role": "model",
                        },
                        {
                            "parts": [
                                {"functionResponse": {
    "parts": [],
    "name": "add",
    "response": {
        "output": "<system-reminder>This is a system reminder</system-reminder>5"
    },
}},
                                {"functionResponse": {
    "parts": [],
    "name": "multiply",
    "response": {
        "output": "<system-reminder>This is a system reminder</system-reminder>20"
    },
}},
                            ],
                            "role": "user",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "description": "Add two integers.",
                                    "name": "add", "parameters_json_schema": {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "First number"},
        "b": {"type": "integer", "description": "Second number"},
    },
    "required": ["a", "b"],
}},
                                {
                                    "description": "Multiply two integers.",
                                    "name": "multiply", "parameters_json_schema": {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "First number"},
        "b": {"type": "integer", "description": "Second number"},
    },
    "required": ["a", "b"],
}},
                            ]
                        }
                    ],
                },
                "tool_call_with_thought_signature": {
                    "contents": [
                        {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
                        {
                            "parts": [
                                {"text": "I'll add those."},
                                {
                                    "functionCall": {"args": {"a": 2, "b": 3}, "name": "add"},
                                    "thoughtSignature": "dGhvdWdodF9zaWduYXR1cmVfZGF0YQ==",
                                },
                            ],
                            "role": "model",
                        },
                    ],
                    "systemInstruction": {"parts": [{"text": ""}], "role": "user"},
                },
            }
        )


async def test_google_genai_generation_kwargs():
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        provider = GoogleGenAI(
            model="gemini-2.5-flash", api_key="test-key", stream=False
        ).with_generation_kwargs(temperature=0.7, max_output_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        config = body.get("generationConfig", {})
        assert (config.get("temperature"), config.get("maxOutputTokens")) == snapshot((0.7, 2048))


async def test_google_genai_no_id_in_function_call_or_response():
    """Gemini API rejects 'id' in function_call/function_response parts.

    The google-genai SDK may accept ``id`` as a keyword argument for
    ``FunctionCall`` / ``FunctionResponse``, but the Gemini REST API returns
    HTTP 400 when `id` is present in the wire JSON.  Verify that our
    conversion never emits it.
    """
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content=[TextPart(text="Sure.")],
            tool_calls=[
                ToolCall(
                    id="call_xyz",
                    function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                ),
            ],
        ),
        Message(role="tool", content="5", tool_call_id="call_xyz"),
    ]

    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        provider = GoogleGenAI(model="gemini-2.5-flash", api_key="test-key", stream=False)
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())

    for content in body.get("contents", []):
        for part in content.get("parts", []):
            fc = part.get("functionCall") or part.get("function_call")
            if fc is not None:
                assert "id" not in fc, f"FunctionCall must not contain 'id', got: {fc}"
            fr = part.get("functionResponse") or part.get("function_response")
            if fr is not None:
                assert "id" not in fr, f"FunctionResponse must not contain 'id', got: {fr}"


async def test_google_genai_with_thinking():
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        provider = GoogleGenAI(
            model="gemini-2.5-flash", api_key="test-key", stream=False
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body.get("generationConfig", {}).get("thinkingConfig") == snapshot(
            {"include_thoughts": True, "thinking_budget": 32000}
        )


# -----------------------------------------------------------------------------
# Malformed tool-call arguments (defensive parsing)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "expected_error_substring"),
    [
        ('{"a": 1, "b": 2', ""),  # broken JSON — loads_relaxed repairs it
        ("<args><a>1</a></args>", "must be a JSON object, got str."),  # XML
        ("a: 1\nb: 2", "must be a JSON object, got str."),  # YAML
        ("{{a=1, b=2}}", "must be a JSON object, got list."),  # DSML-like
        ("not json at all", "must be a JSON object, got str."),  # garbage
        ("[1, 2, 3]", "must be a JSON object"),  # valid JSON, but array
    ],
    ids=["broken_json", "xml", "yaml", "dsml", "garbage", "json_array"],
)
async def test_google_genai_malformed_tool_call_arguments_in_request(
    arguments: str, expected_error_substring: str
):
    """Malformed tool-call arguments in history must be surfaced to the LLM, not crash."""
    from common import capture_request

    provider = GoogleGenAI(model="gemini-2.5-flash", api_key="test-key", stream=False)
    history = [
        Message(
            role="assistant",
            content=[TextPart(text="I'll call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_bad",
                    function=ToolCall.FunctionBody(name="add", arguments=arguments),
                )
            ],
        )
    ]

    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        body = await capture_request(mock, provider, "", [], history)

    contents = body["contents"]
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert parts[0]["text"] == "I'll call a tool."
    if not expected_error_substring:
        # loads_relaxed successfully repaired the JSON
        assert parts[1]["functionCall"]["args"] == {"a": 1, "b": 2}
        assert parts[1]["functionCall"]["name"] == "add"
    else:
        assert expected_error_substring in parts[1]["text"]
        assert parts[2]["functionCall"]["args"] == {}
        assert parts[2]["functionCall"]["name"] == "add"


async def test_google_genai_empty_tool_call_arguments_in_request():
    """Empty string arguments should produce empty functionCall args, not an error."""
    from common import capture_request

    provider = GoogleGenAI(model="gemini-2.5-flash", api_key="test-key", stream=False)
    history = [
        Message(
            role="assistant",
            content=[TextPart(text="I'll call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_empty",
                    function=ToolCall.FunctionBody(name="add", arguments=""),
                )
            ],
        )
    ]

    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_response())
        )
        body = await capture_request(mock, provider, "", [], history)

    contents = body["contents"]
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "I'll call a tool."
    assert parts[1]["functionCall"]["args"] == {}
    assert parts[1]["functionCall"]["name"] == "add"


async def test_google_genai_none_function_call_args_in_response():
    """Backend returning function_call with args=None must produce empty arguments."""
    from google.genai.types import (
        Candidate,
        Content,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )

    response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[Part(function_call=FunctionCall(name="add", args=None))],
                    role="model",
                ),
                finish_reason="STOP",
            )
        ]
    )
    stream = GoogleGenAIStreamedMessage(response)
    parts = [p async for p in stream]
    assert len(parts) == 1
    assert isinstance(parts[0], ToolCall)
    assert parts[0].function.arguments == "{}"


async def test_google_genai_empty_dict_function_call_args_in_response():
    """Backend returning function_call with args={} must produce empty arguments."""
    from google.genai.types import (
        Candidate,
        Content,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )

    response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[Part(function_call=FunctionCall(name="add", args={}))],
                    role="model",
                ),
                finish_reason="STOP",
            )
        ]
    )
    stream = GoogleGenAIStreamedMessage(response)
    parts = [p async for p in stream]
    assert len(parts) == 1
    assert isinstance(parts[0], ToolCall)
    assert parts[0].function.arguments == "{}"
