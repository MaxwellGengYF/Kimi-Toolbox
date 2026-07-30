import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, Unpack, cast

import httpx
from openai import AsyncStream, Omit, OpenAIError, omit
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from kosong.chat_provider import (
    ChatProvider,
    RetryableChatProvider,
    ThinkingEffort,
)
from kosong.chat_provider.openai_common import (
    CommonGenerationKwargs,
    OpenAICompatibleProviderMixin,
    OpenAICompatibleStreamedMessage,
    clamp_max_tokens,
    apply_generation_kwargs,
    clamp_thinking_effort,
    convert_error,
    extract_reasoning_from_content,
    is_effectively_empty_content_parts,
    maybe_log_reasoning_content_error,  # noqa: F401
    reasoning_effort_to_thinking_effort,
    thinking_effort_to_reasoning_effort,
    tool_to_openai,
)
from kosong.contrib.chat_provider.common import (
    ToolMessageConversion,
    check_tool_call_id,
    validate_tool_call_arguments,
)
from kosong.message import Message, TextPart, ThinkPart
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(openai_legacy: OpenAILegacy):
        _: ChatProvider = openai_legacy
        _: RetryableChatProvider = openai_legacy


def _reasoning_effort_to_extra_body_level(reasoning_effort: ReasoningEffort | Omit | None) -> str:
    """Map ReasoningEffort to the three-level effort string for extra_body.reasoning.effort.

    Standard ``ReasoningEffort`` values (``low``, ``medium``, ``high``) are
    preserved.  Non-standard values (``max``, ``xhigh``, …) are passed through
    as-is so that all ``extra_body`` fields that encode effort remain
    consistent with each other.
    """
    if reasoning_effort is None or reasoning_effort is omit:
        return "no_think"
    if reasoning_effort in ("low", "minimal"):
        return "low"
    if reasoning_effort == "medium":
        return "medium"
    if reasoning_effort == "high":
        return "high"
    # Non-standard values (max, xhigh, …) – pass through unchanged
    # so that extra_body.reasoning.effort, chat_template_kwargs.reasoning_effort,
    # and extra_body.reasoning_effort all carry the same value.
    return cast(str, reasoning_effort)


class OpenAILegacy(OpenAICompatibleProviderMixin):
    """
    A chat provider that uses the OpenAI Chat Completions API.

    >>> chat_provider = OpenAILegacy(model="gpt-5", api_key="sk-1234567890")
    >>> chat_provider.name
    'openai'
    >>> chat_provider.model_name
    'gpt-5'
    """

    name = "openai"

    class GenerationKwargs(CommonGenerationKwargs, extra_items=Any, total=False):
        """
        Generation kwargs for various kinds of OpenAI-compatible APIs.
        `extra_items=Any` is used to support any extra args.
        """

        n: int | None
        presence_penalty: float | None
        frequency_penalty: float | None
        stop: str | list[str] | None
        prompt_cache_key: str | None

    _DEFAULT_SUPPORTED_EFFORTS: frozenset[ThinkingEffort] = frozenset(
        {"low", "medium", "high", "xhigh", "max"}
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        reasoning_key: str | None = None,
        openai_settings: dict[str, Any] | None = None,
        tool_message_conversion: ToolMessageConversion | None = None,
        supported_efforts: set[ThinkingEffort] | None = None,
        **client_kwargs: Any,
    ):
        """
        Initialize the OpenAILegacy chat provider.

        To support OpenAI-compatible APIs that inject reasoning content in a extra field in
        the message, such as `{"reasoning": ...}`, `reasoning_key` can be set to the key name.

        ``openai_settings`` controls which auto-generated keys are included in the provider's
        ``extra_body`` when ``reasoning_key`` is configured. It should be a dict with boolean
        flags for ``thinking``, ``reasoning`` and ``chat_template_kwargs``. When omitted, all
        three keys are included for backward compatibility.

        ``supported_efforts`` restricts the thinking-effort levels this model accepts.
        When omitted, all levels (``low`` through ``max``) are allowed.
        """
        self._init_openai_client(api_key=api_key, base_url=base_url, client_kwargs=client_kwargs)
        """The underlying `AsyncOpenAI` client."""
        self.model = model
        self.stream = stream
        self._reasoning_effort: ReasoningEffort | Omit = omit
        self._reasoning_key = reasoning_key
        self._supported_efforts = (
            frozenset(supported_efforts)
            if supported_efforts is not None
            else self._DEFAULT_SUPPORTED_EFFORTS
        )
        # Moonshot's own API uses the standard OpenAI reasoning_effort/reasoning_content
        # wire format; the extra_body keys (thinking/reasoning/chat_template_kwargs) are
        # meant for other OpenAI-compatible backends and can confuse Moonshot.
        is_moonshot = model.startswith("kimi-") or (
            base_url is not None and "moonshot" in base_url.lower()
        )
        self._openai_settings: dict[str, Any] = {
            "thinking": not is_moonshot,
            "reasoning": not is_moonshot,
            "chat_template_kwargs": not is_moonshot,
        }
        if openai_settings is not None:
            self._openai_settings.update(openai_settings)
        self._tool_message_conversion: ToolMessageConversion | None = tool_message_conversion
        self._generation_kwargs: OpenAILegacy.GenerationKwargs = {}

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        if self._reasoning_effort is omit:
            return None
        return reasoning_effort_to_thinking_effort(self._reasoning_effort)

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> OpenAILegacyStreamedMessage:
        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(self._generation_kwargs)
        # Clamp max_tokens to a safe upper bound so that an over-large default
        # from the config layer (e.g. max_context_size = 1 M) doesn't trigger
        # a 400 from the API, which enforces its own per-model output limit.
        clamp_max_tokens(generation_kwargs)

        # The OpenAI Chat Completions API does not accept ``max_output_tokens``
        # (that is a Responses API parameter).  Some callers (e.g. the soul
        # retry logic in ``kimisoul.py``) set both ``max_tokens`` and
        # ``max_output_tokens``; the latter must be stripped here to avoid
        # ``TypeError: AsyncCompletions.create() got an unexpected keyword
        # argument 'max_output_tokens'``.
        generation_kwargs.pop("max_output_tokens", None)

        reasoning_effort = self._reasoning_effort
        # Auto-enable reasoning_effort when the history contains ThinkPart but reasoning
        # was not explicitly configured. This prevents server validation errors from APIs
        # (e.g. One API) that require reasoning_effort when messages contain reasoning_content.
        # See: https://github.com/MoonshotAI/kimi-cli/issues/1616
        if reasoning_effort is omit and self._reasoning_key:
            has_think_part = any(
                isinstance(part, ThinkPart) for message in history for part in message.content
            )
            if has_think_part:
                reasoning_effort = "medium"
        reasoning_enabled = reasoning_effort is not None and reasoning_effort is not omit

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            # `system` vs `developer`: see `message_to_openai` comments
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._convert_message(message) for message in history)

        if self._reasoning_key is not None:
            extra_body_level = _reasoning_effort_to_extra_body_level(reasoning_effort)
            extra_body: dict[str, Any] = {}
            if self._openai_settings.get("thinking", True):
                extra_body["thinking"] = {
                    "type": "enabled" if reasoning_enabled else "disabled",
                }
            if self._openai_settings.get("reasoning", True):
                extra_body["reasoning"] = {
                    "effort": extra_body_level,
                }
            if self._openai_settings.get("chat_template_kwargs", True):
                extra_body["chat_template_kwargs"] = {
                    "reasoning_effort": extra_body_level
                }
            if existing_extra_body := generation_kwargs.get("extra_body"):
                merged_extra_body: dict[str, Any] = {**extra_body, **existing_extra_body}
                for key in ("thinking", "reasoning", "chat_template_kwargs"):
                    auto_val = extra_body.get(key)
                    user_val = existing_extra_body.get(key)
                    if auto_val is not None and user_val is not None:
                        merged_extra_body[key] = {**auto_val, **user_val}
                extra_body = merged_extra_body
            if extra_body:
                generation_kwargs["extra_body"] = extra_body
        # Bypass SDK-side ReasoningEffort validation so non-standard effort values
        # such as "max" can be forwarded to OpenAI-compatible backends that accept
        # them. The value is still sent in the request body via extra_body.
        if reasoning_effort is not omit:
            extra_body = generation_kwargs.setdefault("extra_body", {})
            extra_body["reasoning_effort"] = reasoning_effort
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=(tool_to_openai(tool) for tool in tools),
                stream=self.stream,
                stream_options={"include_usage": True} if self.stream else omit,
                reasoning_effort=reasoning_effort,
                **generation_kwargs,
            )
            return OpenAILegacyStreamedMessage(response, self._reasoning_key)
        except (OpenAIError, httpx.HTTPError) as e:
            # Debug logging for the Moonshot/Kimi "reasoning_content must be passed back"
            # 400 is disabled by default. Uncomment the block below to enable it.
            # maybe_log_reasoning_content_error(
            #     e,
            #     provider_name=self.name,
            #     model=self.model,
            #     messages=messages,
            #     generation_kwargs=generation_kwargs,
            # )
            raise convert_error(e) from e

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        new_self._reasoning_effort = thinking_effort_to_reasoning_effort(
            clamp_thinking_effort(effort, self._supported_efforts)
        )
        return new_self

    def with_parallel_tool_calls(self, enabled: bool = True) -> Self:
        """Control whether the model may call multiple tools in parallel.

        Args:
            enabled: When ``True`` (the default), the model may emit multiple
                function/tool calls in a single turn. When ``False``, the model
                is restricted to at most one tool call per turn.
        """
        new_self = self.with_generation_kwargs()
        if enabled:
            new_self._generation_kwargs.pop("parallel_tool_calls", None)
        else:
            new_self._generation_kwargs["parallel_tool_calls"] = False
        return new_self

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> Self:
        """
        Copy the chat provider, updating the generation kwargs with the given values.

        Returns:
            Self: A new instance of the chat provider with updated generation kwargs.
        """
        return apply_generation_kwargs(self, **kwargs)

    @property
    def model_parameters(self) -> dict[str, Any]:
        """
        The parameters of the model to use.

        For tracing/logging purposes.
        """

        model_parameters: dict[str, Any] = {"base_url": str(self.client.base_url)}
        if self._reasoning_effort is not omit:
            model_parameters["reasoning_effort"] = self._reasoning_effort
        return model_parameters

    def _convert_message(self, message: Message) -> ChatCompletionMessageParam:
        """Convert a Kosong message to OpenAI message."""
        # Note: for openai, `developer` role is more standard, but `system` is still accepted.
        # And many openai-compatible models do not accept `developer` role.
        # So we use `system` role here. OpenAIResponses will use `developer` role.
        # See https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions
        # Only deep-copy when we might mutate (tool_calls with invalid JSON, or content
        # list modifications).  For simple user/text messages this avoids an expensive
        # recursive deep copy.
        needs_mutation = (
            (message.role == "assistant" and message.tool_calls)
            or (message.role == "tool" and self._tool_message_conversion == "extract_text")
        )
        message = message.model_copy(deep=needs_mutation)

        # Tool message without tool_call_id would cause a 400 from OpenAI.
        # Return the error to the LLM instead of crashing.
        if message.role == "tool" and (
            error_msg := check_tool_call_id(
                message.tool_call_id, message.extract_text(sep="\n")
            )
        ):
            return cast(
                ChatCompletionMessageParam,
                {"role": "user", "content": error_msg},
            )

        # Validate tool call arguments in assistant messages to avoid API 400s.
        if message.role == "assistant" and message.tool_calls:
            error_texts = validate_tool_call_arguments(message.tool_calls)
            if error_texts:
                message.content = [TextPart(text="\n".join(error_texts)), *message.content]

        reasoning_content, visible_content = extract_reasoning_from_content(message.content)
        # if tool message and `tool_result_conversion` is `extract_text`, patch all text parts into
        # one so that we can make use of the serialization process of `Message` to output string
        if message.role == "tool" and self._tool_message_conversion == "extract_text":
            message.content = [TextPart(text=message.extract_text(sep="\n"))]
        else:
            message.content = visible_content
        dumped_message = message.model_dump(exclude_none=True)
        if (
            message.role == "assistant"
            and message.tool_calls
            and is_effectively_empty_content_parts(visible_content)
        ):
            # OpenAI-compatible APIs allow assistant tool-call messages to omit
            # `content`, but some backends (e.g. Moonshot's) reject a content
            # list that contains an empty text part (observed: `content:
            # [{"type": "text", "text": ""}]` -> 400 "text content is empty").
            # Dropping `content` entirely is always accepted, so do that whenever
            # the visible content is effectively empty alongside a tool call.
            dumped_message.pop("content", None)
        # reasoning_content is required by several OpenAI-compatible APIs (DeepSeek/Kimi/
        # MiniMax/Qwen) whenever thinking mode is enabled. Pass it back on every assistant
        # message when a reasoning key is configured, so the backend sees a consistent
        # reasoning_content field across assistant turns. User/tool messages must not
        # carry the field.
        if self._reasoning_key and message.role == "assistant":
            dumped_message[self._reasoning_key] = reasoning_content
        return cast(ChatCompletionMessageParam, dumped_message)


class OpenAILegacyStreamedMessage(OpenAICompatibleStreamedMessage):
    def __init__(
        self, response: ChatCompletion | AsyncStream[ChatCompletionChunk], reasoning_key: str | None
    ):
        super().__init__(response, reasoning_key=reasoning_key)


if __name__ == "__main__":

    async def _dev_main():
        chat = OpenAILegacy(model="gpt-4o", stream=False)
        system_prompt = "You are a helpful assistant."
        history = [Message(role="user", content="Hello, how are you?")]
        async for part in await chat.generate(system_prompt, [], history):
            print(part.model_dump(exclude_none=True))

        tools = [
            Tool(
                name="get_weather",
                description="Get the weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather for.",
                        },
                    },
                },
            )
        ]
        history = [Message(role="user", content="What's the weather in Beijing?")]
        stream = await chat.generate(system_prompt, tools, history)
        async for part in stream:
            print(part.model_dump(exclude_none=True))
        print("usage:", stream.usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(_dev_main())
