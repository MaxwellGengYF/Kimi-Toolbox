try:
    from google import genai as _  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Google Gemini support requires the optional dependency 'google-genai'. "
        'Install with `pip install "kosong[contrib]"`.'
    ) from exc

import mimetypes
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast

import httpx
import orjson
import pybase64 as base64
from google import genai
from google.genai import client as genai_client
from google.genai import errors as genai_errors
from google.genai.types import (
    Content,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    FunctionResponsePart,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
    ThinkingConfig,
    ThinkingLevel,
    Tool,
    ToolConfig,
)

from kosong.chat_provider import (
    APIStatusError,
    APITimeoutError,
    ChatProvider,
    ChatProviderError,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
    convert_httpx_error,
)
from kosong.chat_provider.openai_common import apply_generation_kwargs
from kosong.contrib.chat_provider.common import (
    get_stream_iteration_timeout,
    validate_tool_call_arguments,
    with_stream_timeout,
)
from kosong.message import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
from kosong.tooling import Tool as KosongTool
from kosong.tooling import ToolReturnValue

if TYPE_CHECKING:

    def type_check(google_genai: GoogleGenAI):
        _: ChatProvider = google_genai


class GoogleGenAI:
    """
    Chat provider backed by Google's Gemini API.
    """

    name = "google_genai"

    class GenerationKwargs(TypedDict, total=False):
        max_output_tokens: int | None
        temperature: float | None
        top_k: int | None
        top_p: float | None
        # Thinking configuration for supported models
        thinking_config: ThinkingConfig | None
        # Tool configuration
        tool_config: ToolConfig | None
        # Extra headers
        http_options: HttpOptions | None

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        vertexai: bool | None = None,
        default_headers: dict[str, str] | None = None,
        **client_kwargs: Any,
    ):
        self._model = model
        self._stream = stream
        self._base_url = base_url
        http_options = HttpOptions(base_url=base_url, headers=default_headers)
        self._client: genai_client.Client = genai.Client(
            http_options=http_options,
            api_key=api_key,
            vertexai=vertexai,
            **client_kwargs,
        )
        self._generation_kwargs: GoogleGenAI.GenerationKwargs = {}

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        thinking_config = self._generation_kwargs.get("thinking_config")
        if thinking_config is None:
            return None

        # For gemini-3 models that use thinking_level
        thinking_level = thinking_config.thinking_level
        if thinking_level is not None:
            match thinking_level:
                case ThinkingLevel.LOW | ThinkingLevel.MINIMAL:
                    return "low"
                case ThinkingLevel.MEDIUM:
                    return "medium"
                case ThinkingLevel.HIGH:
                    return "high"
                case _:
                    return None

        # For other models that use thinking_budget
        thinking_budget = thinking_config.thinking_budget
        if thinking_budget is not None:
            if thinking_budget == 0:
                return "off"
            if thinking_budget <= 1024:
                return "low"
            if thinking_budget <= 4096:
                return "medium"
            return "high"
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[KosongTool],
        history: Sequence[Message],
    ) -> GoogleGenAIStreamedMessage:
        contents = messages_to_google_genai_contents(history)

        # Google GenAI only accepts ``max_output_tokens``.  Strip other token-limit
        # keys that may leak in from callers (e.g. the soul retry logic in
        # ``kimisoul.py``).
        generation_kwargs = dict(self._generation_kwargs)
        generation_kwargs.pop("max_tokens", None)
        generation_kwargs.pop("max_completion_tokens", None)
        config = GenerateContentConfig(**generation_kwargs)
        config.system_instruction = system_prompt
        config.tools = [tool_to_google_genai(tool) for tool in tools]

        try:
            if self._stream:
                stream_response = await self._client.aio.models.generate_content_stream(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,  # type: ignore[reportArgumentType]
                    config=config,
                )
                return GoogleGenAIStreamedMessage(stream_response)
            else:
                response = await self._client.aio.models.generate_content(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,  # type: ignore[reportArgumentType]
                    config=config,
                )
                return GoogleGenAIStreamedMessage(response)
        except Exception as e:  # genai_errors.APIError and others
            raise _convert_error(e) from e

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        thinking_config = ThinkingConfig(include_thoughts=True)

        # Map thinking effort to budget tokens
        if "gemini-3" in self._model:
            match effort:
                case "off":
                    # use default thinking config
                    pass
                case "low":
                    thinking_config.thinking_level = ThinkingLevel.LOW
                case "medium":
                    # FIXME: medium not supported yet, use high
                    thinking_config.thinking_level = ThinkingLevel.HIGH
                case "high" | "xhigh" | "max":
                    # xhigh/max are Anthropic-specific; degrade to HIGH for Gemini.
                    thinking_config.thinking_level = ThinkingLevel.HIGH
        else:
            match effort:
                case "off":
                    thinking_config.thinking_budget = 0
                    thinking_config.include_thoughts = False
                case "low":
                    thinking_config.thinking_budget = 1024
                    thinking_config.include_thoughts = True
                case "medium":
                    thinking_config.thinking_budget = 4096
                    thinking_config.include_thoughts = True
                case "high" | "xhigh" | "max":
                    # xhigh/max are Anthropic-specific; cap at Gemini's highest budget.
                    thinking_config.thinking_budget = 32_000
                    thinking_config.include_thoughts = True

        return self.with_generation_kwargs(thinking_config=thinking_config)

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
        return {
            "model": self._model,
            "base_url": self._base_url,
            **self._generation_kwargs,
        }


class GoogleGenAIStreamedMessage:
    def __init__(self, response: GenerateContentResponse | AsyncIterator[GenerateContentResponse]):
        if isinstance(response, GenerateContentResponse):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._id: str | None = None
        self._usage: GenerateContentResponseUsageMetadata | None = None
        self._timeout_iter: AsyncIterator[StreamedMessagePart] | None = None

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        if self._timeout_iter is None:
            self._timeout_iter = with_stream_timeout(
                self._iter, timeout=get_stream_iteration_timeout()
            )
        return self._timeout_iter

    async def __anext__(self) -> StreamedMessagePart:
        return await self.__aiter__().__anext__()

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage is None:
            return None
        return TokenUsage(
            input_other=self._usage.prompt_token_count or 0,
            output=self._usage.candidates_token_count or 0,
            input_cache_read=self._usage.cached_content_token_count or 0,
            input_cache_creation=0,
        )

    async def _convert_non_stream_response(
        self,
        response: GenerateContentResponse,
    ) -> AsyncIterator[StreamedMessagePart]:
        # Extract usage information
        if response.usage_metadata:
            self._usage = response.usage_metadata
        # Extract ID if available
        if response.response_id is not None:
            self._id = response.response_id

        # Process candidates
        for candidate in response.candidates or []:
            parts = candidate.content.parts if candidate.content else None
            if not parts:
                continue
            for part in parts:
                async for message_part in self._process_part_async(part):
                    yield message_part

    async def _convert_stream_response(
        self,
        response_stream: AsyncIterator[GenerateContentResponse],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async for response in response_stream:
                # Extract ID from first response
                if not self._id and response.response_id is not None:
                    self._id = response.response_id

                # Extract usage information
                if response.usage_metadata:
                    self._usage = response.usage_metadata

                # Process candidates
                for candidate in response.candidates or []:
                    parts = candidate.content.parts if candidate.content else None
                    if not parts:
                        continue
                    for part in parts:
                        async for message_part in self._process_part_async(part):
                            yield message_part
        except genai_errors.APIError as exc:
            raise _convert_error(exc) from exc
        except httpx.HTTPError as exc:
            raise convert_httpx_error(exc) from exc

    def _process_part(self, part: Part):
        """Process a single part and yield message components (synchronous generator).

        Handles different part types from Gemini API:
        - synthetic thinking parts (part.thought is True)
        - encrypted thinking parts (part.thought_signature is not None)
        - text parts
        - function calls
        """
        if part.thought:
            # Synthetic thinking part
            if part.text:
                yield ThinkPart(think=part.text)
        elif part.text:
            # Regular text part
            yield TextPart(text=part.text)
        elif part.function_call:
            func_call = part.function_call
            if func_call.name is None:
                # Skip function calls without a name
                return
            id_ = func_call.id if func_call.id is not None else f"{id(func_call)}"
            tool_call_id = f"{func_call.name}_{id_}"
            # Gemini uses thought_signature to store the encrypted thinking signature.
            # part.thought is synthetic
            # See: https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/thinking/intro_thought_signatures.ipynb
            thought_signature_b64 = (
                base64.b64encode(part.thought_signature).decode("ascii")
                if part.thought_signature
                else None
            )
            yield ToolCall(
                id=tool_call_id,
                function=ToolCall.FunctionBody(
                    name=func_call.name,
                    arguments=orjson.dumps(func_call.args).decode() if func_call.args else "{}",
                ),
                extras={
                    "thought_signature_b64": thought_signature_b64,
                }
                if thought_signature_b64
                else None,
            )

    async def _process_part_async(self, part: Part) -> AsyncIterator[StreamedMessagePart]:
        """Async wrapper for _process_part."""
        for message_part in self._process_part(part):
            yield message_part


def tool_to_google_genai(tool: KosongTool) -> Tool:
    """Convert a Kosong tool to GoogleGenAI tool format."""
    # Use parameters_json_schema instead of parameters to bypass the SDK's
    # Pydantic validation (extra='forbid') which rejects standard JSON Schema
    # metadata fields like $schema, $id, $comment, examples, etc.
    # This is the SDK's official way to pass raw JSON Schema directly to the API.
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.parameters,
            )
        ]
    )


def _image_url_part_to_google_genai(part: ImageURLPart) -> Part:
    """Convert an image URL part to GoogleGenAI format."""
    url = part.image_url.url

    # Handle data URLs
    if url.startswith("data:"):
        # data:[<media-type>][;base64],<data>
        res = url[5:].split(";base64,", 1)
        if len(res) != 2:
            return Part.from_text(
                text=f"Error: Invalid data URL for image: {url}",
            )

        media_type, data_b64 = res
        if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            return Part.from_text(
                text=f"Error: Unsupported media type for base64 image: {media_type}, url: {url}",
            )

        # Decode base64 string to bytes
        data_bytes = base64.b64decode(data_b64)
        return Part.from_bytes(data=data_bytes, mime_type=media_type)
    else:
        # For regular URLs, try to download the image and convert to bytes
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type or not mime_type.startswith("image/"):
            # Default to image/png if we can't detect or it's not an image type
            mime_type = "image/png"
        response = httpx.get(url).raise_for_status()
        data_bytes = response.content
        return Part.from_bytes(data=data_bytes, mime_type=mime_type)


def _audio_url_part_to_google_genai(part: AudioURLPart) -> Part:
    """Convert an audio URL part to GoogleGenAI format."""
    url = part.audio_url.url

    # Handle data URLs
    if url.startswith("data:"):
        # data:[<media-type>][;base64],<data>
        res = url[5:].split(";base64,", 1)
        if len(res) != 2:
            return Part.from_text(
                text=f"Error: Invalid data URL for audio: {url}",
            )

        media_type, data_b64 = res
        # Supported audio formats for GoogleGenAI
        supported_audio_types = (
            "audio/wav",
            "audio/mp3",
            "audio/aiff",
            "audio/aac",
            "audio/ogg",
            "audio/flac",
        )
        if media_type not in supported_audio_types:
            return Part.from_text(
                text=(
                    f"Error: Unsupported media type for base64 audio: {media_type}, url: {url}. "
                    f"Supported types: {supported_audio_types}"
                ),
            )

        # Decode base64 string to bytes
        data_bytes = base64.b64decode(data_b64)
        return Part.from_bytes(data=data_bytes, mime_type=media_type)
    else:
        # Fetch the audio and convert to bytes
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type or not mime_type.startswith("audio/"):
            # Default to audio/mp3 if we can't detect or it's not an audio type
            mime_type = "audio/mp3"
        response = httpx.get(url).raise_for_status()
        data_bytes = response.content
        return Part.from_bytes(data=data_bytes, mime_type=mime_type)


def _tool_result_to_response_and_parts(
    parts: list[ContentPart],
) -> tuple[dict[str, str], list[FunctionResponsePart]]:
    """Convert tool response content to Gemini function response format."""
    genai_parts: list[FunctionResponsePart] = []
    response: str = ""

    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                response += part.text
        elif isinstance(part, ImageURLPart):
            genai_parts.append(FunctionResponsePart.from_uri(file_uri=part.image_url.url))
        elif isinstance(part, AudioURLPart):
            genai_parts.append(FunctionResponsePart.from_uri(file_uri=part.audio_url.url))
        else:
            # Skip unsupported parts (like ThinkPart, etc.)
            continue

    return {"output": response}, genai_parts


def _tool_call_id_to_name(tool_call_id: str, tool_name_by_id: dict[str, str]) -> str:
    """Resolve Gemini `FunctionResponse.name` from a tool_call_id."""
    if tool_call_id in tool_name_by_id:
        return tool_name_by_id[tool_call_id]
    # Fallback for older ids of the form "{tool_name}_{id}".
    return tool_call_id.split("_", 1)[0]


def _tool_message_to_function_response_part(
    message: Message,
    *,
    tool_name_by_id: dict[str, str],
) -> Part:
    if message.role != "tool":  # pragma: no cover - defensive guard
        # Return the error to the LLM instead of crashing.
        return Part.from_text(
            text=(
                f"Error: Expected a tool message, got {message.role}. "
                f"Content: {message.extract_text(sep='\n')}"
            ),
        )
    if message.tool_call_id is None:
        # Return the error to the LLM instead of crashing.
        return Part.from_text(
            text=(
                "Error: Tool response is missing `tool_call_id`. "
                f"Content: {message.extract_text(sep='\n')}"
            ),
        )

    response_data, tool_result_parts = _tool_result_to_response_and_parts(message.content)
    return Part(
        function_response=FunctionResponse(
            name=_tool_call_id_to_name(message.tool_call_id, tool_name_by_id),
            response=response_data,
            parts=tool_result_parts,
        )
    )


def _tool_messages_to_google_genai_content(
    messages: Sequence[Message],
    *,
    tool_name_by_id: dict[str, str],
    expected_tool_call_ids: Sequence[str] | None = None,
    require_all_expected: bool = False,
) -> Content:
    """Pack one-or-more tool results into a single Gemini "user" turn.

    VertexAI-backed Gemini enforces that, for a tool-calling turn, the next
    turn contains the same number of `functionResponse` parts as the preceding
    `functionCall` parts. Packing multiple tool results into a single "user"
    Content keeps us compliant and avoids ordering issues from parallel tool
    execution.
    """
    if not messages:
        # Return the error to the LLM instead of crashing.
        return Content(
            role="user",
            parts=[Part.from_text(text="Error: Expected at least one tool message.")],
        )

    expected_index: dict[str, int] = (
        {tool_call_id: i for i, tool_call_id in enumerate(expected_tool_call_ids)}
        if expected_tool_call_ids is not None
        else {}
    )
    seen_tool_call_ids: set[str] = set()
    indexed_messages = list(enumerate(messages))
    indexed_messages.sort(
        key=lambda t: (expected_index.get(cast(str, t[1].tool_call_id), 10**9), t[0])
    )

    parts: list[Part] = []
    actual_tool_call_ids: list[str] = []
    for _, message in indexed_messages:
        if message.tool_call_id is None:
            # Return the error to the LLM instead of crashing.
            parts.append(
                Part.from_text(
                    text=(
                        "Error: Tool response is missing `tool_call_id`. "
                        f"Content: {message.extract_text(sep='\n')}"
                    ),
                )
            )
            continue
        if message.tool_call_id in seen_tool_call_ids:
            # Return the error to the LLM instead of crashing.
            parts.append(
                Part.from_text(
                    text=f"Error: Duplicate tool response for id: {message.tool_call_id}",
                )
            )
            continue
        seen_tool_call_ids.add(message.tool_call_id)
        actual_tool_call_ids.append(message.tool_call_id)
        parts.append(
            _tool_message_to_function_response_part(message, tool_name_by_id=tool_name_by_id)
        )

    if expected_tool_call_ids is not None and require_all_expected:
        expected_set = set(expected_tool_call_ids)
        missing = [
            tool_call_id
            for tool_call_id in expected_tool_call_ids
            if tool_call_id not in seen_tool_call_ids
        ]
        extra = [
            tool_call_id
            for tool_call_id in actual_tool_call_ids
            if tool_call_id not in expected_set
        ]
        if missing:
            parts.append(
                Part.from_text(
                    text=f"Error: Missing tool responses for ids: {missing}",
                )
            )
        if extra:
            parts.append(
                Part.from_text(
                    text=f"Error: Unexpected tool responses for ids: {extra}",
                )
            )

    return Content(role="user", parts=parts)


def messages_to_google_genai_contents(messages: Sequence[Message]) -> list[Content]:
    """Convert internal messages into a Gemini contents list.

    Tool results for a tool-calling turn are packed into a single "user" message
    with N `functionResponse` parts matching the preceding "model" message's
    N `functionCall` parts. This avoids ordering issues from parallel tool
    execution and satisfies VertexAI's stricter validation.
    """
    contents: list[Content] = []
    tool_name_by_id: dict[str, str] = {}

    i = 0
    while i < len(messages):
        message = messages[i]

        if message.role == "assistant" and message.tool_calls:
            contents.append(message_to_google_genai(message))
            expected_tool_call_ids: list[str] = []
            for tool_call in message.tool_calls:
                tool_name_by_id[tool_call.id] = tool_call.function.name
                expected_tool_call_ids.append(tool_call.id)

            # Collect consecutive tool messages that correspond to this turn.
            j = i + 1
            tool_messages: list[Message] = []
            while j < len(messages) and messages[j].role == "tool":
                tool_messages.append(messages[j])
                j += 1

            if tool_messages:
                contents.append(
                    _tool_messages_to_google_genai_content(
                        tool_messages,
                        tool_name_by_id=tool_name_by_id,
                        expected_tool_call_ids=expected_tool_call_ids,
                        require_all_expected=True,
                    )
                )
                i = j
                continue

            i += 1
            continue

        if message.role == "tool":
            # Tool message without an immediately preceding tool-calling assistant
            # message (e.g. truncated history). Convert it best-effort.
            contents.append(
                _tool_messages_to_google_genai_content([message], tool_name_by_id=tool_name_by_id)
            )
            i += 1
            continue

        contents.append(message_to_google_genai(message))
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name_by_id[tool_call.id] = tool_call.function.name
        i += 1

    return contents


def message_to_google_genai(message: Message) -> Content:
    """Convert a single internal message into GoogleGenAI wire format."""
    role = message.role

    if role == "tool":
        # Return the error to the LLM instead of crashing.
        return Content(
            role="user",
            parts=[
                Part.from_text(
                    text=(
                        "Error: Tool messages must be converted via "
                        "messages_to_google_genai_contents to preserve "
                        "tool-call ordering and tool-response packing."
                    )
                )
            ],
        )

    # GoogleGenAI uses: "user" and "model" (not "assistant")
    google_genai_role = "model" if role == "assistant" else role
    parts: list[Part] = []

    # Handle content parts
    for part in message.content:
        if isinstance(part, TextPart):
            parts.append(Part.from_text(text=part.text))
        elif isinstance(part, ImageURLPart):
            parts.append(_image_url_part_to_google_genai(part))
        elif isinstance(part, AudioURLPart):
            parts.append(_audio_url_part_to_google_genai(part))
        elif isinstance(part, ThinkPart):
            # Note: skip part.thought because it is synthetic
            continue
        else:
            # Skip unsupported parts
            continue

    # Handle tool calls for assistant messages
    # Validate tool call arguments before building parts.
    if message.tool_calls:
        error_texts = validate_tool_call_arguments(message.tool_calls)
        for err_text in error_texts:
            parts.append(Part.from_text(text=err_text))

    from kosong.utils.jsonx import loads_relaxed

    for tool_call in message.tool_calls or []:
        if tool_call.function.arguments:
            args = cast(dict[str, object], loads_relaxed(tool_call.function.arguments))
        else:
            args = {}

        function_call = FunctionCall(
            name=tool_call.function.name,
            args=args,
        )
        function_call_part = Part(function_call=function_call)
        # Add thought_signature back to function_call
        if tool_call.extras and "thought_signature_b64" in tool_call.extras:
            function_call_part.thought_signature = base64.b64decode(
                cast(str, tool_call.extras["thought_signature_b64"])
            )
        parts.append(function_call_part)

    return Content(role=google_genai_role, parts=parts)


def _response_headers(error: Exception) -> Mapping[str, str] | None:
    """Best-effort extraction of response headers from a Google GenAI error."""
    response = getattr(error, "response", None)
    return getattr(response, "headers", None) if response is not None else None


def _convert_error(error: Exception) -> ChatProviderError:
    """Convert a GoogleGenAI error to a Kosong chat provider error."""
    # Handle specific GoogleGenAI error types with detailed status code mapping
    if isinstance(error, genai_errors.ClientError):
        # 4xx client errors
        status_code = getattr(error, "code", 400)
        headers = _response_headers(error)
        if status_code == 401:
            return APIStatusError(
                401, f"Authentication failed: {error}", headers=headers
            )
        elif status_code == 403:
            return APIStatusError(
                403, f"Permission denied: {error}", headers=headers
            )
        elif status_code == 429:
            return APIStatusError(
                429, f"Rate limit exceeded: {error}", headers=headers
            )
        return APIStatusError(status_code, str(error), headers=headers)
    elif isinstance(error, genai_errors.ServerError):
        # 5xx server errors
        status_code = getattr(error, "code", 500)
        return APIStatusError(
            status_code, f"Server error: {error}", headers=_response_headers(error)
        )
    elif isinstance(error, genai_errors.APIError):
        # Generic API errors
        status_code = getattr(error, "code", 500)
        return APIStatusError(
            status_code, str(error), headers=_response_headers(error)
        )
    elif isinstance(error, TimeoutError):
        return APITimeoutError(f"Request timed out: {error}")
    else:
        # Fallback for unexpected errors
        return ChatProviderError(f"Unexpected GoogleGenAI error: {error}")


if __name__ == "__main__":

    async def main():
        import os
        from typing import override

        from pydantic import BaseModel

        import kosong
        from kosong.tooling import CallableTool2, ToolOk
        from kosong.tooling.simple import SimpleToolset

        chat = GoogleGenAI(
            model="gemini-3-pro-preview",
            vertexai=True,
            api_key=os.getenv("VERTEXAI_API_KEY"),
        ).with_thinking("high")
        system_prompt = "You are a helpful assistant."

        class GetWeatherParams(BaseModel):
            city: str

        class GetWeather(CallableTool2[GetWeatherParams]):
            name: str = "get_weather"
            description: str = "Get the weather of a city"
            params: type[GetWeatherParams] = GetWeatherParams

            @override
            async def __call__(self, params: GetWeatherParams) -> ToolReturnValue:
                return ToolOk(output="Sunny")

        toolset = SimpleToolset()
        toolset += GetWeather()
        history = [
            Message(
                role="user",
                content=(
                    "What's the weather like in Beijing and Shanghai? "
                    "Spawn parallel tool calls to get the answer."
                ),
            )
        ]
        result = await kosong.step(chat, system_prompt, toolset, history)
        tool_results = await result.tool_results()

        assistant_message = result.message
        tool_messages = [
            Message(role="tool", content=tr.return_value.output, tool_call_id=tr.tool_call_id)
            for tr in tool_results
        ]
        history.extend([assistant_message] + tool_messages)

        async for part in await chat.generate(system_prompt, toolset.tools, history):
            print(part.model_dump(exclude_none=True))

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
