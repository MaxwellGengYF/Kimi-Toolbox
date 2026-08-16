import uuid
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Self, Unpack, cast, get_args

import httpx
import openai
import regex as re
from openai import AsyncStream, OpenAIError
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseUsage,
    ToolParam,
)
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_input_file_content_param import (
    ResponseInputFileContentParam,
)
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.responses_model import ResponsesModel

from kosong.chat_provider import (
    ChatProvider,
    RetryableChatProvider,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.chat_provider.openai_common import (
    CommonGenerationKwargs,
    OpenAICompatibleProviderMixin,
    apply_generation_kwargs,
    clamp_max_tokens,
    clamp_thinking_effort,
    convert_error,
    maybe_log_reasoning_content_error,  # noqa: F401
    reasoning_effort_to_thinking_effort,
    thinking_effort_to_reasoning_effort,
)
from kosong.contrib.chat_provider.common import (
    BaseStreamedMessage,
    ToolMessageConversion,
    check_tool_call_id,
    normalize_tool_call_ids,
    validate_tool_call_arguments,
)
from kosong.message import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
)
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(openai_responses: OpenAIResponses):
        _: ChatProvider = openai_responses
        _: RetryableChatProvider = openai_responses


def get_openai_models_set() -> set[str]:
    """Return a set of all available OpenAI response models.

    This extracts all literal values from the ResponsesModel TypeAlias, which includes
    both ChatModel and additional response-specific models.
    """
    responses_model_args = get_args(ResponsesModel)
    # responses_model_args is (str, ChatModel, Literal[...])
    # Extract from ChatModel (index 1)
    chat_models = set(get_args(responses_model_args[1]))
    # Extract from the Literal part (index 2)
    response_models = set(get_args(responses_model_args[2]))

    return chat_models | response_models


_openai_models = get_openai_models_set()


def is_openai_model(model_name: str) -> bool:
    """Judge if the model name is an OpenAI model."""
    return model_name in _openai_models


class OpenAIResponses(OpenAICompatibleProviderMixin):
    """
    A chat provider that uses the OpenAI Responses API.

    Similar to `OpenAILegacy`, but uses `client.responses` under the hood.

    This provider always enables reasoning when generating responses.
    If you want to use a non-reasoning model, please use `OpenAILegacy` instead.

    >>> chat_provider = OpenAIResponses(model="gpt-5-codex", api_key="sk-1234567890")
    >>> chat_provider.name
    'openai-responses'
    >>> chat_provider.model_name
    'gpt-5-codex'
    """

    name = "openai-responses"

    class GenerationKwargs(CommonGenerationKwargs, total=False):
        max_output_tokens: int | None
        max_tool_calls: int | None
        reasoning_effort: ReasoningEffort | None
        temperature: float | None
        top_logprobs: float | None
        top_p: float | None
        user: str | None

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
        tool_message_conversion: ToolMessageConversion | None = None,
        supported_efforts: set[ThinkingEffort] | None = None,
        **client_kwargs: Any,
    ):
        self._init_openai_client(api_key=api_key, base_url=base_url, client_kwargs=client_kwargs)
        self._model = model
        self._stream = stream
        self._supported_efforts = (
            frozenset(supported_efforts)
            if supported_efforts is not None
            else self._DEFAULT_SUPPORTED_EFFORTS
        )
        self._tool_message_conversion: ToolMessageConversion | None = tool_message_conversion
        self._generation_kwargs: OpenAIResponses.GenerationKwargs = {}

    @property
    def model_name(self) -> str:
        return self._model

    def _is_deepseek_backend(self) -> bool:
        """Detect the DeepSeek Responses API backend.

        DeepSeek's implementation does not generate reasoning ``summary`` parts
        and does not support ``encrypted_content``: the chain of thought is
        delivered as plaintext ``content`` (``reasoning_text`` parts) instead,
        both in stream events and on reasoning items.
        """
        model = (self.model_name or "").lower()
        if model.startswith("deepseek"):
            return True
        base_url = (self._base_url or str(self.client.base_url)).lower()
        return "deepseek" in base_url

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        reasoning_effort = self._generation_kwargs.get("reasoning_effort")
        if reasoning_effort is None:
            return None
        return reasoning_effort_to_thinking_effort(reasoning_effort)

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> OpenAIResponsesStreamedMessage:
        inputs: ResponseInputParam = []
        if system_prompt:
            system_message: ResponseInputItemParam = {"role": "system", "content": system_prompt}
            if is_openai_model(self.model_name):
                system_message["role"] = "developer"
            inputs.append(system_message)
        # The `Message` type is OpenAI-compatible for Responses API `input` messages.

        # Normalize historical tool-call ids defensively (parity with the
        # anthropic/kimi providers) so sessions persisted under other providers
        # replay cleanly: ids like ``Read:9`` or >64 chars would 400 on strict
        # OpenAI-compatible backends. Input messages are never mutated.
        for message in normalize_tool_call_ids(history):
            inputs.extend(self._convert_message(message))

        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(self._generation_kwargs)
        clamp_max_tokens(generation_kwargs)
        reasoning_effort = generation_kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            # Bypass SDK-side Reasoning model validation so non-standard effort
            # values such as "max" can be forwarded to backends that accept them.
            extra_body = generation_kwargs.setdefault("extra_body", {})
            extra_body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
            generation_kwargs["include"] = ["reasoning.encrypted_content"]

        create_kwargs: dict[str, Any] = {
            "stream": self._stream,
            "model": self._model,
            "input": inputs,
            "tools": [_convert_tool(tool) for tool in tools],
            # Server-side sessions must stay disabled for cross-provider
            # session sharing: ``store=True``/``previous_response_id`` would
            # break replay of anthropic/kimi histories (they cannot continue
            # from an OpenAI response id) and leak provider-side state into
            # client-persisted sessions. See the session contract in
            # ``kimi_cli.llm.create_llm``. Identity is carried via ``user``.
            "store": False,
            **generation_kwargs,
        }
        try:
            response = await self.client.responses.create(**create_kwargs)
        except openai.APIStatusError as e:
            # Some OpenAI-compatible gateways (e.g. llmproxy front-ends) issue
            # reasoning ``encrypted_content`` blobs that they later fail to
            # verify (key rotation, multi-node deployments, ...). The request
            # then fails with a 400 ``invalid_encrypted_content`` on every
            # retry because the offending blob sits in the history. Recover by
            # stripping the unverifiable blobs (keeping the visible reasoning
            # summaries) and retrying once without them.
            if not (
                _is_invalid_encrypted_content_error(e)
                and _strip_reasoning_encrypted_content(inputs)
            ):
                raise convert_error(e) from e
            try:
                response = await self.client.responses.create(**create_kwargs)
            except (OpenAIError, httpx.HTTPError) as retry_error:
                raise convert_error(retry_error) from retry_error
        except (OpenAIError, httpx.HTTPError) as e:
            # Debug logging for the Moonshot/Kimi "reasoning_content must be passed back"
            # 400 is disabled by default. Uncomment the block below to enable it.
            # maybe_log_reasoning_content_error(
            #     e,
            #     provider_name=self.name,
            #     model=self._model,
            #     messages=inputs,
            #     generation_kwargs=generation_kwargs,
            # )
            raise convert_error(e) from e
        return OpenAIResponsesStreamedMessage(response)

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        reasoning_effort = thinking_effort_to_reasoning_effort(
            clamp_thinking_effort(effort, self._supported_efforts)
        )
        return self.with_generation_kwargs(reasoning_effort=reasoning_effort)

    def with_parallel_tool_calls(self, enabled: bool = True) -> Self:
        """Control whether the model may call multiple tools in parallel.

        For the OpenAI Responses API there is no explicit ``parallel_tool_calls``
        flag; parallel calling is controlled via ``max_tool_calls``. Setting
        ``enabled=False`` caps ``max_tool_calls`` at ``1``.

        Args:
            enabled: When ``True`` (the default), no artificial limit is placed
                on the number of simultaneous tool calls. When ``False``,
                ``max_tool_calls`` is set to ``1``.
        """
        new_self = self.with_generation_kwargs()
        if enabled:
            new_self._generation_kwargs.pop("max_tool_calls", None)
        else:
            new_self._generation_kwargs["max_tool_calls"] = 1
        return new_self

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> Self:
        """
        Copy the chat provider, updating the generation kwargs with the given values.

        Returns:
            Self: A new instance of the chat provider with updated generation kwargs.
        """
        return apply_generation_kwargs(self, **kwargs)

    def _convert_message(self, message: Message) -> list[ResponseInputItemParam]:
        """Convert a single message to OpenAI Responses input format.

        Rules:
        - role in {user, assistant}: map to EasyInputMessageParam with role kept
        role == system: map to role=developer for OpenAI models, otherwise kept
        content: str kept; list[ContentPart] mapped to ResponseInputMessageContentListParam
        - role == tool: map to FunctionCallOutput with call_id and output
        """
        # Only deep-copy when we might mutate (tool_calls with invalid JSON).
        needs_mutation = (
            message.role == "assistant" and message.tool_calls
        )
        message = message.model_copy(deep=needs_mutation)

        role = message.role
        if is_openai_model(self.model_name) and role == "system":
            role = "developer"

        # tool role → function_call_output (return value from a prior tool call)
        if role == "tool":
            # Tool message without tool_call_id would cause a 400 from OpenAI.
            # Return the error to the LLM instead of crashing.
            if error_msg := check_tool_call_id(
                message.tool_call_id, message.extract_text(sep="\n")
            ):
                return [
                    {
                        "role": "user",
                        "content": error_msg,
                        "type": "message",
                    }
                ]
            call_id = message.tool_call_id
            if self._tool_message_conversion == "extract_text":
                content = message.extract_text(sep="\n")
            else:
                content = message.content
            output = _message_content_to_function_output_items(content)

            return [
                {
                    "call_id": call_id,
                    "output": output,
                    "type": "function_call_output",
                }
            ]

        # Validate tool call arguments in assistant messages to avoid API 400s.
        if role == "assistant" and message.tool_calls:
            error_texts = validate_tool_call_arguments(message.tool_calls)
            if error_texts:
                message.content = [TextPart(text="\n".join(error_texts)), *message.content]

        result: list[ResponseInputItemParam] = []

        # user/system/assistant → message input item
        if len(message.content) > 0:
            # Split into two kinds of blocks: contiguous non-ThinkPart message blocks, and
            # contiguous ThinkPart groups (grouped by the same `encrypted` value)
            pending_parts: list[ContentPart] = []

            def flush_pending_parts() -> None:
                if not pending_parts:
                    return
                if role == "assistant":
                    # the "id" key is missing by purpose
                    result.append(
                        cast(
                            ResponseOutputMessageParam,
                            {
                                "content": _content_parts_to_output_items(pending_parts),
                                "role": role,
                                "type": "message",
                            },
                        )
                    )
                else:
                    result.append(
                        {
                            "content": _content_parts_to_input_items(pending_parts),
                            "role": role,
                            "type": "message",
                        }
                    )
                pending_parts.clear()

            i = 0
            n = len(message.content)
            while i < n:
                part = message.content[i]
                if isinstance(part, ThinkPart):
                    # Flush accumulated non-reasoning parts first
                    flush_pending_parts()
                    # Aggregate consecutive ThinkPart items with the same `encrypted` value
                    encrypted_value = part.encrypted
                    deepseek_backend = self._is_deepseek_backend()
                    texts = [part.think or ""]
                    i += 1
                    while i < n:
                        next_part = message.content[i]
                        if not isinstance(next_part, ThinkPart):
                            break
                        if next_part.encrypted != encrypted_value:
                            break
                        texts.append(next_part.think or "")
                        i += 1
                    reasoning_item: dict[str, Any] = {
                        "type": "reasoning",
                    }
                    if deepseek_backend:
                        # DeepSeek merges plaintext reasoning ``content`` into
                        # the adjacent assistant message; ``summary`` and
                        # ``encrypted_content`` are unsupported there.
                        reasoning_item["content"] = [
                            {"type": "reasoning_text", "text": text} for text in texts
                        ]
                    else:
                        reasoning_item["summary"] = [
                            {"type": "summary_text", "text": text} for text in texts
                        ]
                        # Omit the key entirely when no encrypted content is
                        # available: some backends reject an explicit null.
                        if encrypted_value is not None:
                            reasoning_item["encrypted_content"] = encrypted_value
                    result.append(cast(ResponseReasoningItemParam, reasoning_item))
                else:
                    pending_parts.append(part)
                    i += 1

            # Handle remaining trailing non-reasoning parts
            flush_pending_parts()

        for tool_call in message.tool_calls or []:
            result.append(
                {
                    "arguments": tool_call.function.arguments or "{}",
                    "call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "type": "function_call",
                }
            )

        return result


_INVALID_ENCRYPTED_CONTENT_MESSAGE_RE = re.compile(
    r"encrypted[ _]content", re.IGNORECASE
)


def _is_invalid_encrypted_content_error(error: openai.APIStatusError) -> bool:
    """Detect the 400 raised when a backend cannot verify a reasoning
    ``encrypted_content`` blob that was passed back in the request input.

    OpenAI-compatible gateways signal this with the
    ``invalid_encrypted_content`` error code (e.g. ``The encrypted content
    gAAA... could not be verified``); fall back to a message heuristic for
    backends that omit the structured code.
    """
    if error.status_code != 400:
        return False
    body = error.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("code") == "invalid_encrypted_content":
            return True
    message = error.message or ""
    return bool(_INVALID_ENCRYPTED_CONTENT_MESSAGE_RE.search(message)) and (
        "could not be verified" in message or "invalid" in message.lower()
    )


def _strip_reasoning_encrypted_content(inputs: ResponseInputParam) -> bool:
    """Remove ``encrypted_content`` from every reasoning item in *inputs*.

    The visible reasoning summaries are kept so the model retains its
    high-level chain of thought. Returns ``True`` when at least one blob was
    stripped (i.e. a retry without them is worth attempting).
    """
    stripped = False
    for item in inputs:
        if (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and item.pop("encrypted_content", None) is not None
        ):
            stripped = True
    return stripped


def _convert_tool(tool: Tool) -> ToolParam:
    """Convert a Kosong tool to an OpenAI Responses tool."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def _content_parts_to_input_items(parts: list[ContentPart]) -> ResponseInputMessageContentListParam:
    """Map internal ContentPart list → ResponseInputMessageContentListParam items."""
    items: ResponseInputMessageContentListParam = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                items.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImageURLPart):
            # default detail
            url = part.image_url.url
            items.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": url,
                }
            )
        elif isinstance(part, AudioURLPart):
            mapped = _map_audio_url_to_input_item(part.audio_url.url)
            if mapped is not None:
                items.append(mapped)
        else:
            # Unknown content – ignore
            continue
    return items


def _content_parts_to_output_items(parts: list[ContentPart]) -> list[ResponseOutputTextParam]:
    """Map internal ContentPart list → ResponseOutputTextParam list items."""
    items: list[ResponseOutputTextParam] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                items.append({"type": "output_text", "text": part.text, "annotations": []})
        else:
            # Unknown content – ignore
            continue
    return items


def _message_content_to_function_output_items(
    content: str | list[ContentPart],
) -> str | ResponseFunctionCallOutputItemListParam:
    """Map ContentPart list → ResponseFunctionCallOutputItemListParam items."""
    output: str | ResponseFunctionCallOutputItemListParam
    # If tool_result_process is `extract_text`, patch all text parts into one string
    if isinstance(content, str):
        output = content
    else:
        items: ResponseFunctionCallOutputItemListParam = []
        for part in content:
            if isinstance(part, TextPart):
                if part.text:
                    items.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImageURLPart):
                url = part.image_url.url
                items.append({"type": "input_image", "image_url": url})
            elif isinstance(part, AudioURLPart):
                mapped = _map_audio_url_to_file_content(part.audio_url.url)
                if mapped is not None:
                    items.append(mapped)
            else:
                continue
        output = items
    return output


def _parse_audio_url(url: str) -> tuple[str | None, str | None, str | None]:
    """Parse an audio URL into (file_url, file_data, ext).

    For ``data:audio/...`` URIs the *ext* is the inferred file extension
    (``"mp3"`` or ``"wav"``), or ``None`` for unsupported codecs.
    For ``http://`` / ``https://`` URLs the file_url is returned directly.
    """
    if url.startswith("data:audio/"):
        try:
            header, b64 = url.split(",", 1)
            subtype = header.split("/")[1].split(";")[0].lower()
            ext = "mp3" if subtype in {"mp3", "mpeg"} else ("wav" if subtype == "wav" else None)
            return None, b64, ext
        except Exception:
            return None, None, None
    if url.startswith("http://") or url.startswith("https://"):
        return url, None, None
    return None, None, None


def _map_audio_url_to_input_item(url: str) -> ResponseInputFileParam | None:
    """Map audio URL/data URI to an input content item (always an input_file).

    OpenAI Responses message content no longer accepts ``input_audio``, so both
    inline data and remote URLs are converted to ``input_file`` items instead.
    """
    file_url, file_data, ext = _parse_audio_url(url)
    if file_url is not None:
        return {"type": "input_file", "file_url": file_url}
    if file_data is not None:
        if ext is None:
            return None
        item: ResponseInputFileParam = {"type": "input_file", "file_data": file_data}
        item["filename"] = f"inline.{ext}"
        return item
    return None


def _map_audio_url_to_file_content(url: str) -> ResponseInputFileContentParam | None:
    """Map audio URL/data URI to a file content item for function_call_output."""
    file_url, file_data, _ = _parse_audio_url(url)
    if file_url is not None:
        return {"type": "input_file", "file_url": file_url}
    if file_data is not None:
        return {"type": "input_file", "file_data": file_data}
    return None


def _reasoning_item_text(item: Any) -> str:
    """Concatenate the visible reasoning text from a completed reasoning item.

    Prefers the raw ``content`` (``reasoning_text`` parts) used by backends
    that expose the full chain of thought (e.g. DeepSeek), falling back to
    the OpenAI-style ``summary`` parts.
    """
    content = getattr(item, "content", None) or []
    if content:
        return "".join(getattr(part, "text", "") or "" for part in content)
    summary = getattr(item, "summary", None) or []
    return "".join(getattr(part, "text", "") or "" for part in summary)


class OpenAIResponsesStreamedMessage(BaseStreamedMessage):
    def __init__(self, response: Response | AsyncStream[ResponseStreamEvent]):
        super().__init__()
        if isinstance(response, Response):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._usage: ResponseUsage | None = None
        # Reasoning item ids whose raw chain-of-thought text was already
        # streamed via ``response.reasoning_text.delta`` events, so the
        # ``.done`` / ``output_item.done`` events don't duplicate it.
        self._streamed_reasoning_item_ids: set[str] = set()

    @property
    def usage(self) -> TokenUsage | None:
        """Derive ``TokenUsage`` from the collected ``ResponseUsage``.

        Handles the standard OpenAI Responses caching schema
        (``input_tokens_details.cached_tokens``).
        """
        if self._usage:
            cached = 0
            total_input = self._usage.input_tokens
            if self._usage.input_tokens_details and self._usage.input_tokens_details.cached_tokens:
                cached = self._usage.input_tokens_details.cached_tokens
            return self._build_token_usage(
                input_other=total_input - cached,
                output=self._usage.output_tokens,
                input_cache_read=cached,
            )
        return None

    async def _convert_non_stream_response(
        self, response: Response
    ) -> AsyncIterator[StreamedMessagePart]:
        """Convert a non-streaming Responses API result into message parts."""
        self._id = response.id
        self._usage = response.usage
        for item in response.output:
            if item.type == "message":
                for content in item.content or []:
                    if content.type == "output_text":
                        yield TextPart(text=content.text)
            elif item.type == "function_call":
                yield ToolCall(
                    id=item.call_id or str(uuid.uuid4()),
                    function=ToolCall.FunctionBody(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
            elif item.type == "reasoning":
                # OpenAI delivers ``summary`` parts; some Responses-compatible
                # backends (e.g. DeepSeek) deliver the raw chain of thought as
                # ``content`` (``reasoning_text`` parts) instead.  Support both.
                reasoning_parts = [*(item.summary or []), *(item.content or [])]
                if reasoning_parts:
                    for reasoning_part in reasoning_parts:
                        yield ThinkPart(
                            think=reasoning_part.text,
                            encrypted=item.encrypted_content,
                        )
                elif item.encrypted_content is not None:
                    # Keep the encrypted blob for history round-tripping.
                    yield ThinkPart(think="", encrypted=item.encrypted_content)

    async def _convert_stream_response(
        self, response: AsyncStream[ResponseStreamEvent]
    ) -> AsyncIterator[StreamedMessagePart]:
        """Convert streaming Responses events into message parts."""
        try:
            async for chunk in response:
                if chunk.type == "response.output_text.delta":
                    yield TextPart(text=chunk.delta)
                elif chunk.type == "response.output_item.added":
                    item = chunk.item
                    self._id = item.id
                    if item.type == "function_call":
                        yield ToolCall(
                            id=item.call_id or str(uuid.uuid4()),
                            function=ToolCall.FunctionBody(
                                name=item.name,
                                arguments=item.arguments,
                            ),
                        )
                elif chunk.type == "response.output_item.done":
                    item = chunk.item
                    self._id = item.id
                    if item.type == "reasoning":
                        if item.encrypted_content is not None:
                            # Keep the encrypted blob for history round-tripping.
                            yield ThinkPart(think="", encrypted=item.encrypted_content)
                        elif item.id not in self._streamed_reasoning_item_ids:
                            # Fallback: backends that only attach the reasoning
                            # text to the completed item (no delta events).
                            text = _reasoning_item_text(item)
                            if text:
                                yield ThinkPart(think=text, encrypted=None)
                elif chunk.type == "response.function_call_arguments.delta":
                    yield ToolCallPart(arguments_part=chunk.delta)
                elif chunk.type == "response.reasoning_summary_part.added":
                    yield ThinkPart(think="")
                elif chunk.type == "response.reasoning_summary_text.delta":
                    yield ThinkPart(think=chunk.delta)
                elif chunk.type == "response.reasoning_text.delta":
                    # Raw chain-of-thought deltas (e.g. DeepSeek Responses API).
                    self._streamed_reasoning_item_ids.add(chunk.item_id)
                    yield ThinkPart(think=chunk.delta)
                elif chunk.type == "response.reasoning_text.done":
                    # Full reasoning text; normally the deltas already streamed
                    # it.  Only emit when this is the sole delivery channel.
                    if chunk.item_id not in self._streamed_reasoning_item_ids:
                        self._streamed_reasoning_item_ids.add(chunk.item_id)
                        yield ThinkPart(think=chunk.text)
                elif chunk.type == "response.completed":
                    self._usage = chunk.response.usage
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e


if __name__ == "__main__":

    async def _dev_main():
        # Non-streaming example
        chat = OpenAIResponses(model="gpt-5-codex", stream=True)
        system_prompt = "You are a helpful assistant."
        history = [Message(role="user", content="Hello, how are you?")]

        from kosong import generate

        result = await generate(chat, system_prompt, [], history)
        print(result.message)
        print(result.usage)
        history.append(result.message)

        # Streaming example with tools
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
        history.append(Message(role="user", content="What's the weather in Beijing?"))
        result = await generate(chat, system_prompt, tools, history)
        print(result.message)
        print(result.usage)
        history.append(result.message)
        for tool_call in result.message.tool_calls or []:
            assert tool_call.function.name == "get_weather"
            history.append(Message(role="tool", tool_call_id=tool_call.id, content="Sunny"))
        result = await generate(chat, system_prompt, tools, history)
        print(result.message)
        print(result.usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv(override=True)
    asyncio.run(_dev_main())
