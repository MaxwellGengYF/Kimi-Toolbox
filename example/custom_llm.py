"""Simple custom ChatProvider demo (no network)."""
from __future__ import annotations

import asyncio
import orjson
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from kaos.path import KaosPath
from kosong.chat_provider import (
    ChatProvider,
    StreamedMessage,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling import Tool

from kimi_agent_sdk import Session, ToolResult
from kimix.utils import _create_session_async


class FixedStreamedMessage:
    """StreamedMessage that yields predefined parts."""

    def __init__(self, parts: list[StreamedMessagePart]) -> None:
        self._parts = parts
        self._msg_id = "fixed-msg-001"

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamedMessagePart]:
        for part in self._parts:
            yield part

    @property
    def id(self) -> str | None:
        return self._msg_id

    @property
    def usage(self) -> TokenUsage | None:
        return None


class FixedChatProvider:
    """ChatProvider that returns fixed responses."""

    name = "fixed"

    def __init__(self, responses: list[list[StreamedMessagePart]]) -> None:
        self._responses = responses
        self._index = 0

    @property
    def model_name(self) -> str:
        return "fixed-model"

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self, system_prompt: str, tools: Sequence[Tool], history: Sequence[Message]
    ) -> StreamedMessage:
        print('=' * 20 + ' system_prompt ' + '=' * 20)
        print(system_prompt)
        print('=' * 20 + ' tools ' + '=' * 20)
        s = ''
        for i, tool in enumerate(tools, 1):
            s += (f"  [{i}] {tool.name}") + '\n'
            s += (f"      description: {tool.description}") + '\n'
            s += (f"      parameters:  {orjson.dumps(tool.parameters, option=orjson.OPT_INDENT_2).decode("utf-8")}") + '\n'
        # Path('tools.md').write_text(s, encoding='utf-8', errors='replace')
        print('Exported ' + str(len(s)))
        print('=' * 20 + ' history ' + '=' * 20)
        for msg in history:
            print(msg)
        if self._index < len(self._responses):
            parts = self._responses[self._index]
        else:
            parts = [TextPart(text="Done")]
        self._index += 1
        return FixedStreamedMessage(parts)

    def with_thinking(self, effort: ThinkingEffort) -> FixedChatProvider:
        return self


async def demo_custom_llm_fixed_text_response() -> None:
    provider = FixedChatProvider([[TextPart(text="Hello from fixed LLM")]])

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = KaosPath.unsafe_from_local_path(Path(tmpdir))
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(
            """
[loop_control]
max_steps_per_turn = 5
max_retries_per_step = 1
""",
            encoding="utf-8",
        )

        session = await _create_session_async(
            work_dir=work_dir,
            chat_provider=provider,
        )
        try:
            text_parts: list[str] = []
            async for msg in session.prompt("Say hello"):
                if isinstance(msg, TextPart):
                    text_parts.append(msg.text)
            print(text_parts)
            result = " ".join(text_parts)
            if "Hello from fixed LLM" not in result:
                raise RuntimeError(
                    f"Expected greeting in result, got: {result}")
        finally:
            await session.close()


async def demo_custom_llm_fixed_tool_call() -> None:
    provider = FixedChatProvider(
        [
            [
                ToolCall(
                    id="call_001",
                    function=ToolCall.FunctionBody(
                        name="fake_tool",
                        arguments='{"arg": "value"}',
                    ),
                )
            ],
            [TextPart(text="Tool call completed")],
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = KaosPath.unsafe_from_local_path(Path(tmpdir))
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(
            """
[loop_control]
max_steps_per_turn = 5
max_retries_per_step = 1
""",
            encoding="utf-8",
        )

        session = await _create_session_async(
            work_dir=work_dir,
            chat_provider=provider,
        )
        try:
            tool_calls: list[ToolCall] = []
            text_parts: list[str] = []
            async for msg in session.prompt("Call a tool"):
                if isinstance(msg, ToolCall):
                    tool_calls.append(msg)
                elif isinstance(msg, TextPart):
                    text_parts.append(msg.text)
            print(tool_calls)
            if len(tool_calls) < 1:
                raise RuntimeError("Expected at least one tool call")
            if tool_calls[0].function.name != "fake_tool":
                raise RuntimeError(
                    f"Expected fake_tool, got: {tool_calls[0].function.name}")
            if not any("Tool call completed" in t for t in text_parts):
                raise RuntimeError("Expected completion text in result")
        finally:
            await session.close()


async def demo_custom_llm_read_file() -> None:
    test_file_path = str(Path(__file__).parent / "test_text.txt")
    provider = FixedChatProvider(
        [
            [
                ToolCall(
                    id="call_readfile",
                    function=ToolCall.FunctionBody(
                        name="read",
                        arguments=orjson.dumps({"file_path": test_file_path}).decode("utf-8"),
                    ),
                )
            ],
            [TextPart(text="114514 1919810")],
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = KaosPath.unsafe_from_local_path(Path(tmpdir))
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(
            """
[loop_control]
max_steps_per_turn = 5
max_retries_per_step = 1
""",
            encoding="utf-8",
        )

        session = await _create_session_async(
            work_dir=work_dir,
            chat_provider=provider,
        )
        try:
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []
            text_parts: list[str] = []
            async for msg in session.prompt("Read the test file"):
                if isinstance(msg, ToolCall):
                    tool_calls.append(msg)
                elif isinstance(msg, ToolResult):
                    tool_results.append(msg)
                elif isinstance(msg, TextPart):
                    text_parts.append(msg.text)

            if len(tool_calls) < 1:
                raise RuntimeError("Expected at least one tool call")
            if tool_calls[0].function.name != "read":
                raise RuntimeError(
                    f"Expected read, got: {tool_calls[0].function.name}")
            if len(tool_results) < 1:
                raise RuntimeError("Expected at least one tool result")
            result_output = str(tool_results[0].return_value.output)
            print(result_output)
            if not any("114514" in t and "1919810" in t for t in text_parts):
                raise RuntimeError("Expected specific text in result")
        finally:
            await session.close()


if __name__ == "__main__":
    asyncio.run(demo_custom_llm_fixed_text_response())
    asyncio.run(demo_custom_llm_fixed_tool_call())
    asyncio.run(demo_custom_llm_read_file())
