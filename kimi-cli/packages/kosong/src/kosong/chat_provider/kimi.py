import copy
import mimetypes
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, Unpack, cast

import httpx
import regex as re
from openai import AsyncOpenAI, AsyncStream, BaseModel, OpenAIError, omit
from openai._types import RequestFiles, RequestOptions
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from typing_extensions import TypedDict

from kosong.chat_provider import (
    ChatProvider,
    ChatProviderError,
    RetryableChatProvider,
    ThinkingEffort,
    TokenUsage,
)
from kosong.chat_provider.openai_common import (
    CommonGenerationKwargs,
    OpenAICompatibleProviderMixin,
    OpenAICompatibleStreamedMessage,
    apply_generation_kwargs,
    clamp_max_tokens,
    convert_error,
    extract_reasoning_from_content,
    is_effectively_empty_content_parts,
    maybe_log_reasoning_content_error,  # noqa: F401
    tool_to_openai,
)
from kosong.message import (
    Message,
    ThinkPart,
    VideoURLPart,
)
from kosong.tooling import Tool
from kosong.utils.jsonschema import JsonDict, deref_json_schema, ensure_property_types

if TYPE_CHECKING:

    def type_check(kimi: Kimi):
        _: ChatProvider = kimi
        _: RetryableChatProvider = kimi


class ThinkingConfig(TypedDict, total=False):
    type: Literal["enabled", "disabled"]
    effort: str
    """Concrete thinking effort level (e.g. ``"low"``, ``"high"``, ``"max"``).
    Carried inside the ``thinking`` object; the API does not accept a top-level
    ``reasoning_effort`` field for this contract."""
    keep: Any
    """Moonshot-specific ``thinking.keep`` switch for preserved thinking.
    Forwarded verbatim to the API; callers are responsible for choosing a value
    the server accepts (e.g. ``"all"``)."""


class ExtraBody(TypedDict, total=False, extra_items=Any):
    thinking: ThinkingConfig


class Kimi(OpenAICompatibleProviderMixin):
    """
    A chat provider that uses the Kimi API.

    >>> chat_provider = Kimi(model="kimi-k2-turbo-preview", api_key="sk-1234567890")
    >>> chat_provider.name
    'kimi'
    >>> chat_provider.model_name
    'kimi-k2-turbo-preview'
    >>> chat_provider.with_generation_kwargs(temperature=0)._generation_kwargs
    {'temperature': 0}
    >>> chat_provider._generation_kwargs
    {}
    """

    name = "kimi"

    class GenerationKwargs(CommonGenerationKwargs, total=False):
        """
        See https://platform.moonshot.ai/docs/api/chat#request-body.
        """

        n: int | None
        presence_penalty: float | None
        frequency_penalty: float | None
        stop: str | list[str] | None
        prompt_cache_key: str | None
        reasoning_effort: str | None
        """Legacy thinking parameter, forwarded verbatim when set manually.
        Use ``with_thinking`` / `extra_body.thinking.effort` instead."""
        extra_body: ExtraBody | None

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = os.getenv("KIMI_API_KEY")
        if api_key is None:
            raise ChatProviderError(
                "The api_key client option or the KIMI_API_KEY environment variable is not set"
            )
        if base_url is None:
            base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")

        self._init_openai_client(api_key=api_key, base_url=base_url, client_kwargs=client_kwargs)
        """The underlying `AsyncOpenAI` client."""
        self.model: str = model
        """The name of the model to use."""
        self.stream: bool = stream
        """Whether to generate responses as a stream."""
        self._generation_kwargs: Kimi.GenerationKwargs = {}

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        extra_body = self._generation_kwargs.get("extra_body") or {}
        thinking = extra_body.get("thinking")
        if thinking is None:
            return None
        if thinking.get("type") == "disabled":
            return "off"
        effort = thinking.get("effort")
        if effort in ("low", "medium", "high", "xhigh", "max"):
            return cast(ThinkingEffort, effort)
        # Thinking enabled without a concrete effort (the boolean "on"
        # signal) is not representable in kosong's ThinkingEffort set.
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> KimiStreamedMessage:
        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(self._generation_kwargs)
        # Drop None-valued kwargs so they are never serialized as JSON null.
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        # Normalize the legacy ``max_tokens`` alias to Kimi's preferred
        # ``max_completion_tokens``. For reasoning models ``max_tokens`` shares
        # the budget with ``reasoning_content`` and a small value can cause a
        # 200 response with no ``content``. When both are set,
        # ``max_completion_tokens`` wins. When neither is set, send no cap.
        if generation_kwargs.get("max_completion_tokens") is None:
            max_tokens = generation_kwargs.get("max_tokens")
            if max_tokens is not None:
                generation_kwargs["max_completion_tokens"] = max_tokens
        generation_kwargs.pop("max_tokens", None)
        clamp_max_tokens(generation_kwargs)

        extra_body = cast(dict[str, Any], generation_kwargs.get("extra_body") or {})
        thinking = cast(dict[str, Any], extra_body.get("thinking") or {})
        # Preserved thinking (``thinking.keep == "all"``) requires a
        # ``reasoning_content`` field on every assistant message; without it
        # only messages that actually carry reasoning include the field.
        preserved_thinking_enabled = (
            thinking.get("keep") == "all" and thinking.get("type") != "disabled"
        )

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            _convert_message(message, preserved_thinking_enabled=preserved_thinking_enabled)
            for message in _normalize_tool_call_ids(history)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[_convert_tool(tool) for tool in tools] if tools else omit,
                stream=self.stream,
                stream_options={"include_usage": True} if self.stream else omit,
                **generation_kwargs,
            )
            return KimiStreamedMessage(response)
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
        thinking: ThinkingConfig
        if effort == "off":
            thinking = {"type": "disabled"}
        else:
            # Concrete effort strings pass through verbatim — model
            # compatibility and fallback are resolved above the provider
            # boundary.
            thinking = {"type": "enabled", "effort": effort}
        # Replace extra_body.thinking wholesale so a stale ``effort`` from a
        # previous with_thinking call can never linger on a disabled thinking
        # object — but carry over a ``keep`` set earlier via with_extra_body
        # (the KIMI_MODEL_THINKING_KEEP path applies keep after with_thinking
        # and merges on top, so it is unaffected either way).
        old_extra_body = self._generation_kwargs.get("extra_body") or {}
        old_thinking = old_extra_body.get("thinking") or {}
        keep = old_thinking.get("keep")
        if keep is not None:
            thinking["keep"] = keep
        return self.with_generation_kwargs(extra_body={**old_extra_body, "thinking": thinking})

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> Self:
        """
        Copy the chat provider, updating the generation kwargs with the given values.

        Returns:
            Self: A new instance of the chat provider with updated generation kwargs.
        """
        return apply_generation_kwargs(self, **kwargs)

    def with_extra_body(self, extra_body: ExtraBody) -> Self:
        """
        Copy the chat provider, updating the extra_body in generation kwargs.

        Top-level keys follow last-writer-wins semantics, except for the
        ``thinking`` key: its sub-dict is merged field-by-field so that a
        later call adding ``thinking.keep`` does not erase a ``thinking.type``
        installed by an earlier ``with_thinking`` call.

        Returns:
            Self: A new instance of the chat provider with updated extra_body.
        """
        new_self = copy.copy(self)
        new_self._generation_kwargs = copy.deepcopy(self._generation_kwargs)
        old_extra_body = new_self._generation_kwargs.get("extra_body") or {}
        new_extra_body: ExtraBody = {**old_extra_body, **extra_body}
        old_thinking = old_extra_body.get("thinking")
        new_thinking = extra_body.get("thinking")
        if old_thinking is not None and new_thinking is not None:
            new_extra_body["thinking"] = {**old_thinking, **new_thinking}
        new_self._generation_kwargs["extra_body"] = new_extra_body
        return new_self

    @property
    def files(self) -> KimiFiles:
        return KimiFiles(self.client)


class KimiFiles:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def upload_video(self, *, data: bytes, mime_type: str) -> VideoURLPart:
        """Upload a video to Kimi files API and return a video URL content part."""
        if not mime_type.startswith("video/"):
            raise ChatProviderError(f"Expected a video mime type, got {mime_type}")
        url = await self._upload_file(data=data, mime_type=mime_type, purpose="video")
        return VideoURLPart(video_url=VideoURLPart.VideoURL(url=url))

    async def _upload_file(self, *, data: bytes, mime_type: str, purpose: KimiFilePurpose) -> str:
        filename = _guess_filename(mime_type)
        files: RequestFiles = {"file": (filename, data, mime_type)}
        options: RequestOptions = {"headers": {"Content-Type": "multipart/form-data"}}
        try:
            response: KimiFileObject = await self._client.post(
                "/files",
                cast_to=KimiFileObject,
                body={"purpose": purpose},
                files=files,
                options=options,
            )
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e
        return f"ms://{response.id}"


class KimiFileObject(BaseModel):
    id: str


type KimiFilePurpose = Literal["video", "image"]


def _guess_filename(mime_type: str) -> str:
    extension = mimetypes.guess_extension(mime_type) or ".bin"
    return f"upload{extension}"


_EMPTY_TOOL_CALL_ID = "tool_call"
_TOOL_CALL_ID_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_CALL_ID_MAX_LENGTH = 64


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """Replace characters Moonshot rejects and truncate to its id budget."""
    sanitized = _TOOL_CALL_ID_SAFE_CHARS.sub("_", tool_call_id)
    return sanitized[:_TOOL_CALL_ID_MAX_LENGTH]


def _make_unique_tool_call_id(normalized: str, used: set[str]) -> str:
    base = normalized if normalized else _EMPTY_TOOL_CALL_ID
    candidate = base[:_TOOL_CALL_ID_MAX_LENGTH]
    if candidate not in used:
        return candidate
    index = 2
    while True:
        suffix = f"_{index}"
        candidate = base[: _TOOL_CALL_ID_MAX_LENGTH - len(suffix)] + suffix
        if candidate not in used:
            return candidate
        index += 1


def _normalize_tool_call_ids(history: Sequence[Message]) -> Sequence[Message]:
    """Rewrite invalid historical tool-call ids to Moonshot's accepted shape.

    Histories persisted from other providers (or older sessions) can contain
    tool-call ids with characters Moonshot rejects (e.g. ``Read:9``) or ids
    longer than 64 characters; sending them verbatim fails the whole request
    with a 400. Ids are sanitized to ``[a-zA-Z0-9_-]``, truncated, and made
    unique with ``_2``/``_3``... suffixes; assistant ``tool_calls`` entries
    and their matching ``tool`` messages are rewritten consistently.
    """
    raw_ids: list[str] = []
    seen: set[str] = set()
    for message in history:
        for tool_call in message.tool_calls or []:
            if tool_call.id not in seen:
                seen.add(tool_call.id)
                raw_ids.append(tool_call.id)
        if message.tool_call_id is not None and message.tool_call_id not in seen:
            seen.add(message.tool_call_id)
            raw_ids.append(message.tool_call_id)
    if not raw_ids:
        return history

    # Ids that already satisfy the contract keep their value (first pass), so
    # only genuinely invalid ids are rewritten (second pass).
    mapped: dict[str, str] = {}
    used: set[str] = set()
    for raw_id in raw_ids:
        normalized = _sanitize_tool_call_id(raw_id)
        if normalized == raw_id and normalized:
            mapped[raw_id] = normalized
            used.add(normalized)
    for raw_id in raw_ids:
        if raw_id in mapped:
            continue
        unique = _make_unique_tool_call_id(_sanitize_tool_call_id(raw_id), used)
        mapped[raw_id] = unique
        used.add(unique)

    if all(mapped[raw_id] == raw_id for raw_id in raw_ids):
        return history

    normalized_messages: list[Message] = []
    for message in history:
        changed = False
        new_tool_calls = message.tool_calls
        if message.tool_calls:
            new_tool_calls = []
            for tool_call in message.tool_calls:
                mapped_id = mapped[tool_call.id]
                if mapped_id == tool_call.id:
                    new_tool_calls.append(tool_call)
                else:
                    changed = True
                    new_tool_calls.append(tool_call.model_copy(update={"id": mapped_id}))
        new_tool_call_id = (
            mapped[message.tool_call_id]
            if message.tool_call_id is not None
            else message.tool_call_id
        )
        if new_tool_call_id != message.tool_call_id:
            changed = True
        if not changed:
            normalized_messages.append(message)
        else:
            normalized_messages.append(
                message.model_copy(
                    update={"tool_calls": new_tool_calls, "tool_call_id": new_tool_call_id}
                )
            )
    return normalized_messages

def _convert_message(
    message: Message, *, preserved_thinking_enabled: bool = False
) -> ChatCompletionMessageParam:
    message = message.model_copy(deep=True)
    has_reasoning_part = any(isinstance(part, ThinkPart) for part in message.content)
    reasoning_content, visible_content = extract_reasoning_from_content(message.content)
    message.content = visible_content
    dumped_message = message.model_dump(exclude_none=True)
    if (
        message.role == "assistant"
        and message.tool_calls
        and is_effectively_empty_content_parts(visible_content)
    ):
        # OpenAI-compatible APIs allow assistant tool-call messages to omit
        # `content`, but the Kimi-for-Coding compat layer rejects a content
        # list that contains an empty text part (observed: `content:
        # [{"type": "text", "text": ""}]` -> 400 "text content is empty").
        # Dropping `content` entirely is always accepted, so do that whenever
        # the visible content is effectively empty alongside a tool call.
        dumped_message.pop("content", None)
    # Moonshot/DeepSeek-compatible backends require reasoning_content to be
    # passed back on assistant messages that carried reasoning. When preserved
    # thinking is active (``thinking.keep == "all"`` and thinking is not
    # disabled), the field must additionally be present on every assistant
    # message, so backfill an empty string for messages without reasoning.
    if has_reasoning_part or (preserved_thinking_enabled and message.role == "assistant"):
        dumped_message["reasoning_content"] = reasoning_content
    return cast(ChatCompletionMessageParam, dumped_message)


def _convert_tool(tool: Tool) -> ChatCompletionToolParam:
    if tool.name.startswith("$"):
        # Kimi builtin functions start with `$`
        return cast(
            ChatCompletionToolParam,
            {
                "type": "builtin_function",
                "function": {
                    "name": tool.name,
                    # no need to set description and parameters
                },
            },
        )
    converted = tool_to_openai(tool)
    # Moonshot's API rejects parameter schemas whose nested properties omit
    # `type` (e.g. enum-only properties exposed by some MCP servers), and
    # schemas that keep draft-7 ``definitions``/``$ref`` indirections. Inline
    # local refs, then patch missing types locally so such tools keep working
    # against Kimi without requiring every MCP server author to tighten their
    # schemas.
    function = converted["function"]
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        normalized = ensure_property_types(deref_json_schema(cast(JsonDict, parameters)))
        function["parameters"] = cast(dict[str, object], normalized)
    return converted


class KimiStreamedMessage(OpenAICompatibleStreamedMessage):
    """The streamed message of the Kimi chat provider."""

    def __init__(self, response: ChatCompletion | AsyncStream[ChatCompletionChunk]):
        super().__init__(response, reasoning_key="reasoning_content")

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage:
            cached = 0
            total_input = self._usage.prompt_tokens
            if hasattr(self._usage, "cached_tokens"):
                # https://platform.moonshot.cn/docs/api/chat#%E8%BF%94%E5%9B%9E%E5%86%85%E5%AE%B9
                # TODO: delete this when Moonshot API becomes compatible with OpenAI API
                cached = getattr(self._usage, "cached_tokens") or 0  # noqa: B009
            elif (
                self._usage.prompt_tokens_details
                and self._usage.prompt_tokens_details.cached_tokens
            ):
                cached = self._usage.prompt_tokens_details.cached_tokens
            return self._build_token_usage(
                input_other=total_input - cached,
                output=self._usage.completion_tokens,
                input_cache_read=cached,
            )
        return None


if __name__ == "__main__":

    async def _dev_main():
        chat = Kimi(model="kimi-k2-turbo-preview", stream=False)
        system_prompt = ""
        history = [
            Message(role="user", content="Hello, who is Confucius?"),
        ]
        stream = await chat.with_generation_kwargs(
            temperature=0,
            max_tokens=1000,
        ).generate(system_prompt, [], history)
        async for part in stream:
            print(part.model_dump(exclude_none=True))
        print("id:", stream.id)
        print("usage:", stream.usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(_dev_main())
