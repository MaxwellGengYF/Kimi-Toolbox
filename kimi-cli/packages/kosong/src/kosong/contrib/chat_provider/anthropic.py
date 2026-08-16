try:
    import anthropic as _  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Anthropic support requires the optional dependency 'anthropic'. "
        'Install with `pip install "kosong[contrib]"`.'
    ) from exc

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, Unpack, cast

import httpx
import orjson
import regex as re
from anthropic import (
    AnthropicError,
    AsyncAnthropic,
    AsyncStream,
    omit,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    PermissionDeniedError as AnthropicPermissionDeniedError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.lib.streaming import MessageStopEvent
from anthropic.types import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageParam,
    MessageStartEvent,
    MetadataParam,
    OutputConfigParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageStreamEvent,
    TextBlockParam,
    ThinkingBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
    Usage,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types.tool_result_block_param import Content as ToolResultContent

from kosong.chat_provider import (
    DEFAULT_MAX_RETRIES,
    APIConnectionError,
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
    BaseStreamedMessage,
    ToolMessageConversion,
    check_tool_call_id,
    normalize_tool_call_ids,
    validate_tool_call_arguments,
)
from kosong.message import (
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

    def type_check(anthropic: Anthropic):
        _: ChatProvider = anthropic


type MessagePayload = tuple[str | None, list[MessageParam]]

type BetaFeatures = Literal["interleaved-thinking-2025-05-14"]


_ANTHROPIC_CLOSE_TASKS: set[asyncio.Task[None]] = set()


def _on_close_task_done(task: asyncio.Task[None]) -> None:
    _ANTHROPIC_CLOSE_TASKS.discard(task)
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


async def _drain_awaitable(awaitable: Awaitable[object]) -> None:
    """Drain an awaitable, ignoring harmless shutdown errors."""
    try:
        await asyncio.wait_for(awaitable, timeout=5.0)
    except TimeoutError:
        return
    except asyncio.CancelledError:
        # Outer task was cancelled (e.g. during event-loop shutdown).
        # Swallow silently — the OS will reclaim the sockets.
        return
    except RuntimeError as exc:
        # On Windows/Python 3.14, closing an httpx.AsyncClient whose
        # underlying transports were bound to a now-closed ProactorEventLoop
        # raises RuntimeError('Event loop is closed').  Swallow it silently —
        # the OS will reclaim the socket.
        if "Event loop is closed" in str(exc):
            return
        raise
    except Exception:
        return


# Models that accept adaptive thinking but don't expose a major.minor version
# in their identifier (e.g. Claude Mythos Preview).
_ADAPTIVE_MARKERS_NO_VERSION: tuple[str, ...] = ("mythos",)

# Matches the Claude family's major.minor version inside a model identifier.
# `\d{1,2}(?!\d)` prevents date suffixes like `sonnet-4-20250514` from being
# misread as minor=20: `20` would be followed by another digit, so the
# negative lookahead fails and the regex does not match at all.
_FAMILY_VERSION_RE = re.compile(r"(?:opus|sonnet|haiku)[.-](\d+)[.-](\d{1,2})(?!\d)")

# Adaptive thinking was introduced with Opus 4.6 / Sonnet 4.6.
_ADAPTIVE_MIN_VERSION: tuple[int, int] = (4, 6)

# Anthropic models currently top out at 128k output tokens.  The Python SDK
# additionally refuses non-streaming requests whose expected duration exceeds
# 10 minutes; that threshold is 128_000 * 10 / 60 = 21_333 tokens.
_MAX_OUTPUT_TOKENS_STREAMING = 128_000
_MAX_OUTPUT_TOKENS_NONSTREAMING = 21_333


def _clamp_max_tokens(kwargs: dict[str, Any], *, streaming: bool) -> None:
    """Clamp ``max_tokens`` to a safe upper bound for Anthropic models.

    The bound depends on whether streaming is enabled, because the SDK rejects
    large non-streaming requests upfront to keep them under 10 minutes.
    """
    raw = kwargs.get("max_tokens")
    if raw is None:
        return
    upper = _MAX_OUTPUT_TOKENS_STREAMING if streaming else _MAX_OUTPUT_TOKENS_NONSTREAMING
    if raw > upper:
        kwargs["max_tokens"] = upper


def _supports_adaptive_thinking(model: str) -> bool:
    """Whether the given model id accepts `thinking: {type: "adaptive"}`.

    Strategy: explicit marker for non-versioned models (e.g. Mythos), plus a
    strict regex over the Claude family that extrapolates to unknown future
    versions (>= 4.6) without code changes.
    """
    m = model.lower()
    if any(marker in m for marker in _ADAPTIVE_MARKERS_NO_VERSION):
        return True
    if match := _FAMILY_VERSION_RE.search(m):
        major, minor = int(match.group(1)), int(match.group(2))
        return (major, minor) >= _ADAPTIVE_MIN_VERSION
    return False


def _clamp_effort(
    effort: ThinkingEffort,
    supported_efforts: frozenset[ThinkingEffort],
) -> ThinkingEffort:
    """Clamp an effort level to the highest one supported by the model.

    ``off`` is passed through unchanged as it represents "disable thinking"
    rather than an effort rank. If the requested level is supported it passes
    through unchanged; otherwise the highest supported non-off level is chosen,
    falling back to ``high`` when nothing else is available.
    """
    if effort == "off":
        return effort
    if effort in supported_efforts:
        return effort
    # Fall back to the highest supported non-off level.
    for candidate in ("max", "xhigh", "high", "medium", "low"):
        if candidate in supported_efforts:
            return candidate  # type: ignore[return-value]
    return "high"


def _supports_effort_param(model: str) -> bool:
    """Whether the model accepts ``output_config.effort`` at all.

    Per Anthropic's effort docs, the parameter is explicitly supported on
    Claude Mythos Preview, Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet
    4.6, and Claude Opus 4.5. Adaptive-capable models all support it via the
    adaptive pathway. For the legacy (manual thinking) pathway, only Opus
    4.5 is explicitly listed.

    We gate ``output_config`` emission on this predicate to avoid sending
    effort to models that would reject it with a 400 (Claude 3.x, and
    conservatively Sonnet 4 / Sonnet 4.5 / Haiku 4.5 which are not in the
    explicit list). A false negative here means "effort not sent, no
    regression from pre-effort behaviour"; a false positive would mean
    "API 400 error", so we err on the side of silence.
    """
    if _supports_adaptive_thinking(model):
        return True
    # Opus 4.5 is the only legacy (non-adaptive) model that Anthropic docs
    # explicitly confirm supports the effort parameter.
    m = model.lower()
    return "opus-4-5" in m or "opus-4.5" in m


class Anthropic:
    """
    Chat provider backed by Anthropic's Messages API.
    """

    name = "anthropic"

    class GenerationKwargs(TypedDict, total=False):
        max_tokens: int
        temperature: float | None
        top_k: int | None
        top_p: float | None
        # e.g., {"type": "adaptive", "display": "summarized"}
        # or   {"type": "enabled", "budget_tokens": 1024}
        thinking: ThinkingConfigParam | None
        # e.g., {"effort": "high"} — soft guidance that applies to all output
        # tokens. Used in adaptive thinking requests, and in legacy requests
        # only when the model is on Anthropic's explicit effort-supporting
        # list (see ``_supports_effort_param``).
        output_config: OutputConfigParam | None
        # e.g., {"type": "auto", "disable_parallel_tool_use": True}
        tool_choice: ToolChoiceParam | None

        beta_features: list[BetaFeatures] | None
        extra_headers: Mapping[str, str] | None

    # ``xhigh`` is documented as Opus 4.7-only, so it is excluded from the
    # default set. Callers that know their model supports ``xhigh`` can opt in
    # by passing ``supported_efforts`` explicitly.
    _DEFAULT_SUPPORTED_EFFORTS: frozenset[ThinkingEffort] = frozenset(
        {"low", "medium", "high", "max"}
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        # which process should we apply on tool result
        tool_message_conversion: ToolMessageConversion | None = None,
        # Must provide a max_tokens. Can be overridden by .with_generation_kwargs()
        default_max_tokens: int,
        metadata: MetadataParam | None = None,
        supported_efforts: set[ThinkingEffort] | None = None,
        **client_kwargs: Any,
    ):
        self._model = model
        self._stream = stream
        self._supported_efforts = (
            frozenset(supported_efforts)
            if supported_efforts is not None
            else self._DEFAULT_SUPPORTED_EFFORTS
        )
        # Provide our own httpx.AsyncClient so the Anthropic SDK does not create
        # its default ``AsyncHttpxClientWrapper``. That wrapper's ``__del__``
        # schedules ``self.aclose()`` as a task on the running loop; on
        # Windows/Python 3.14, if the loop is torn down before the task finishes,
        # the unretrieved exception surfaces as a noisy
        # ``RuntimeError: Event loop is closed`` traceback.
        client_kwargs = dict(client_kwargs)
        # Apply a default SDK-level retry budget for transient errors such as
        # 429 Rate Limit and 5xx server errors.  Callers may override this by
        # passing ``max_retries`` explicitly in ``client_kwargs``.
        client_kwargs.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        if "http_client" not in client_kwargs:
            client_kwargs["http_client"] = httpx.AsyncClient()
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, **client_kwargs)
        self._tool_message_conversion: ToolMessageConversion | None = tool_message_conversion
        self._metadata = metadata
        self._generation_kwargs: Anthropic.GenerationKwargs = {
            "max_tokens": default_max_tokens,
            "beta_features": ["interleaved-thinking-2025-05-14"],
        }
        # Remember the event loop this provider was created on.  Closing the
        # underlying httpx client from a different (or already-closed) loop on
        # Windows/Python 3.14 raises a noisy RuntimeError; when the loop has
        # changed, we simply abandon the client and let the OS reclaim sockets.
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        thinking_config = self._generation_kwargs.get("thinking")
        if thinking_config is None:
            return None
        if thinking_config["type"] == "disabled":
            return "off"
        if thinking_config["type"] == "adaptive":
            output_config = self._generation_kwargs.get("output_config") or {}
            effort = output_config.get("effort")
            if effort in ("low", "medium", "high", "xhigh", "max"):
                return effort
            return "high"
        budget = thinking_config["budget_tokens"]
        if budget <= 1024:
            return "low"
        if budget <= 4096:
            return "medium"
        return "high"

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> AnthropicStreamedMessage:
        # https://docs.claude.com/en/api/messages#body-messages
        # Anthropic API does not support system roles, but just a system prompt.
        system = (
            [
                TextBlockParam(
                    text=system_prompt,
                    type="text",
                    cache_control=CacheControlEphemeralParam(type="ephemeral"),
                )
            ]
            if system_prompt
            else omit
        )
        # Normalize historical tool-call ids defensively: Anthropic validates
        # tool_use/tool_result ids against [a-zA-Z0-9_-] and 400s on ids
        # persisted from other providers (e.g. ``Read:9``). Assistant
        # tool_calls and their matching tool messages are rewritten
        # consistently; the caller's history objects are never mutated.
        messages: list[MessageParam] = []
        for message in normalize_tool_call_ids(history):
            converted = self._convert_message(message)
            # Per Anthropic spec, tool_result blocks for the same assistant turn
            # must live in a single user message. Internal Messages model one
            # tool call per entry, so merge consecutive tool-result-only user
            # messages here. Strict-compat backends (e.g. DeepSeek /anthropic)
            # 400 on the split form, and the official backend silently teaches
            # the model to stop calling tools in parallel.
            if (
                messages
                and converted["role"] == "user"
                and messages[-1]["role"] == "user"
                and _is_tool_result_only(messages[-1]["content"])
                and _is_tool_result_only(converted["content"])
            ):
                prev_content = cast(list[ContentBlockParam], messages[-1]["content"])
                new_content = cast(list[ContentBlockParam], converted["content"])
                messages[-1]["content"] = [*prev_content, *new_content]
            else:
                messages.append(converted)
        if messages:
            last_message = messages[-1]
            last_content = last_message["content"]

            # inject cache control in the last content.
            # https://docs.claude.com/en/docs/build-with-claude/prompt-caching
            if isinstance(last_content, list) and last_content:
                content_blocks = cast(list[ContentBlockParam], last_content)
                last_block = content_blocks[-1]
                match last_block["type"]:
                    case (
                        "text"
                        | "image"
                        | "document"
                        | "search_result"
                        | "tool_use"
                        | "tool_result"
                        | "server_tool_use"
                        | "web_search_tool_result"
                    ):
                        last_block["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
                    case "thinking" | "redacted_thinking":
                        pass
        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(self._generation_kwargs)

        # ``max_tokens`` is required by Anthropic and may be defaulted to the
        # model's full context size by upper layers. Clamp and validate it.
        if generation_kwargs.get("max_tokens") is None:
            raise ValueError("max_tokens must be provided for Anthropic")
        _clamp_max_tokens(generation_kwargs, streaming=self._stream)

        # The Anthropic Messages API only accepts ``max_tokens``.  Other token-limit
        # keys (``max_output_tokens``, ``max_completion_tokens``) may leak in from
        # callers (e.g. the soul retry logic in ``kimisoul.py``).  Strip them to
        # avoid ``TypeError`` from the SDK.
        generation_kwargs.pop("max_output_tokens", None)
        generation_kwargs.pop("max_completion_tokens", None)

        betas = generation_kwargs.pop("beta_features", []) or []
        extra_headers = generation_kwargs.pop("extra_headers", {}) or {}
        extra_headers = {
            **{"anthropic-beta": ",".join(str(e) for e in betas)},
            **extra_headers,
        }

        tools_ = [_convert_tool(tool) for tool in tools]
        if tools:
            tools_[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        try:
            response = await self._client.messages.create(
                model=self._model,
                messages=messages,
                system=system,
                tools=tools_,
                stream=self._stream,
                extra_headers=extra_headers,
                metadata=self._metadata if self._metadata is not None else omit,
                **generation_kwargs,
            )
            return AnthropicStreamedMessage(response)
        except (AnthropicError, httpx.HTTPError) as e:
            raise _convert_error(e) from e

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        if effort == "off":
            new = self.with_generation_kwargs(thinking={"type": "disabled"})
            # Clear any stale output_config from a prior adaptive configuration.
            new._generation_kwargs.pop("output_config", None)
            return new

        # Clamp to whatever the model actually accepts. xhigh/max fall back
        # to high on models that don't support them; low/medium/high pass
        # through; max passes through on 4.6-family and newer.
        effective = _clamp_effort(effort, self._supported_efforts)
        # SDK 0.78 OutputConfigParam TypedDict lists only low/medium/high/max.
        # `xhigh` is valid on Opus 4.7 per the API docs but not yet typed.
        output_config: OutputConfigParam = {"effort": effective}  # type: ignore[typeddict-item]

        if _supports_adaptive_thinking(self._model):
            # Opus 4.6+ / Sonnet 4.6+ / Mythos: adaptive thinking.
            # `display: "summarized"` is required on Opus 4.7+ (where the default
            # flipped to "omitted") and is a no-op on 4.6. Setting it
            # unconditionally keeps thinking content visible across versions.
            # SDK 0.78 TypedDict doesn't model `display` yet — thus the ignore.
            thinking_config: ThinkingConfigParam = {
                "type": "adaptive",
                "display": "summarized",
            }  # type: ignore[typeddict-item]
            new = self.with_generation_kwargs(
                thinking=thinking_config,
                output_config=output_config,
            )
            # Adaptive mode auto-enables interleaved thinking, so the beta
            # header is redundant. Drop it if still present from construction.
            if (
                beta_features := new._generation_kwargs.get("beta_features")
            ) and "interleaved-thinking-2025-05-14" in beta_features:
                beta_features.remove("interleaved-thinking-2025-05-14")
            return new

        # Pre-4.6 models: legacy budget-based thinking. After clamping,
        # `effective` is guaranteed to be one of low/medium/high here.
        # Only models that Anthropic's docs explicitly list as supporting the
        # effort parameter (e.g. Opus 4.5) get `output_config` emitted; other
        # pre-4.6 models (Sonnet 4, Sonnet 4.5, Haiku 4.5, Claude 3.x) omit it
        # to avoid 400 validation errors on models that don't accept it.
        budgets: dict[str, int] = {
            "low": 1024,
            "medium": 4096,
            "high": 32_000,
            "xhigh": 64_000,
            "max": 128_000,
        }
        kwargs: dict[str, Any] = {
            "thinking": {"type": "enabled", "budget_tokens": budgets[effective]},
        }
        if _supports_effort_param(self._model):
            kwargs["output_config"] = output_config
        return self.with_generation_kwargs(**kwargs)

    def with_parallel_tool_calls(self, enabled: bool = True) -> Self:
        """Control whether the model may call multiple tools in parallel.

        Args:
            enabled: When ``True`` (the default), the model may emit multiple
                tool_use blocks in a single turn. When ``False``, the model is
                restricted to at most one tool call per turn.
        """
        new_self = self.with_generation_kwargs()
        if enabled:
            new_self._generation_kwargs.pop("tool_choice", None)
        else:
            new_self._generation_kwargs["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }
        return new_self

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> Self:
        """
        Copy the chat provider, updating the generation kwargs with the given values.

        Returns:
            Self: A new instance of the chat provider with updated generation kwargs.
        """
        return apply_generation_kwargs(self, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying Anthropic HTTP client.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        running_loop = asyncio.get_running_loop()
        if running_loop.is_closed() or (
            self._loop is not None and running_loop is not self._loop
        ):
            # The client was bound to a different (or already-closed) event loop.
            # Abandon it silently; the OS will reclaim the sockets.
            return
        try:
            await self._client.close()
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                return
            raise
        except asyncio.CancelledError:
            # Cancelled during close (e.g. event-loop shutdown).
            # Swallow silently — the OS will reclaim the sockets.
            return

    def close(self) -> None:
        """Schedule an asynchronous close of the underlying HTTP client.

        This synchronous helper is useful when cleanup must be initiated from a
        non-async context (e.g. object finalization or a sync callback).  If an
        event loop is running the close is scheduled as a task; otherwise the
        client is abandoned because there is no loop left to drain it.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running.  The client's original loop is gone, so
            # creating a new loop to close it will fail for transports bound
            # to the old loop.  Abandon the client — the OS will clean up the
            # sockets on process exit.
            return
        if self._loop is not None and loop is not self._loop:
            # The running loop is not the one this client was bound to; closing
            # from here would raise RuntimeError('Event loop is closed').
            return
        try:
            result = self._client.close()
        except Exception:
            return
        try:
            task = loop.create_task(_drain_awaitable(cast(Awaitable[object], result)))
        except RuntimeError:
            # The loop is shutting down; abandon the close.
            return
        _ANTHROPIC_CLOSE_TASKS.add(task)
        task.add_done_callback(_on_close_task_done)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()

    @property
    def model_parameters(self) -> dict[str, Any]:
        """
        The parameters of the model to use.

        For tracing/logging purposes.
        """

        model_parameters: dict[str, Any] = {"base_url": str(self._client.base_url)}
        model_parameters.update(self._generation_kwargs)
        return model_parameters

    def _convert_message(self, message: Message) -> MessageParam:
        """Convert a single internal message into Anthropic wire format."""
        role = message.role

        if role == "system":
            # Anthropic does not support system messages in the conversation.
            # We map it to a special user message.
            return MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text", text=f"<system>{message.extract_text(sep='\n')}</system>"
                    )
                ],
            )
        elif role == "tool":
            if error_msg := check_tool_call_id(
                message.tool_call_id, message.extract_text(sep="\n")
            ):
                # Return the error to the LLM instead of crashing.
                return MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(type="text", text=error_msg)
                    ],
                )
            if self._tool_message_conversion == "extract_text":
                content = message.extract_text(sep="\n")
            else:
                content = message.content
            block = _tool_result_message_to_block(message.tool_call_id, content)
            return MessageParam(role="user", content=[block])

        assert role in ("user", "assistant")
        blocks: list[ContentBlockParam] = []
        for part in message.content:
            if isinstance(part, TextPart):
                blocks.append(TextBlockParam(type="text", text=part.text))
            elif isinstance(part, ImageURLPart):
                blocks.append(_image_url_part_to_anthropic(part))
            elif isinstance(part, ThinkPart):
                if part.encrypted is None:
                    # missing signature, strip this thinking block.
                    continue
                else:
                    blocks.append(
                        ThinkingBlockParam(
                            type="thinking", thinking=part.think, signature=part.encrypted
                        )
                    )
            else:
                continue
        # Validate tool call arguments before building blocks.
        if message.tool_calls:
            error_texts = validate_tool_call_arguments(message.tool_calls)
            for err_text in error_texts:
                blocks.append(TextBlockParam(type="text", text=err_text))

        from kosong.utils.jsonx import loads_relaxed

        for tool_call in message.tool_calls or []:
            if tool_call.function.arguments:
                tool_input = cast(dict[str, object], loads_relaxed(tool_call.function.arguments))
            else:
                tool_input = {}
            blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=tool_input,
                )
            )
        return MessageParam(role=role, content=blocks)


class AnthropicStreamedMessage(BaseStreamedMessage):
    def __init__(self, response: AnthropicMessage | AsyncStream[RawMessageStreamEvent]):
        if isinstance(response, AnthropicMessage):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._usage = Usage(input_tokens=0, output_tokens=0)

    @property
    def usage(self) -> TokenUsage | None:
        # https://docs.claude.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
        return self._build_token_usage(
            # Note: in some Anthropic-compatible APIs, input_tokens can be None.
            # Anthropic reports input_tokens as non-cached input already, so
            # it maps directly to input_other.
            input_other=self._usage.input_tokens or 0,
            output=self._usage.output_tokens,
            input_cache_read=self._usage.cache_read_input_tokens or 0,
            input_cache_creation=self._usage.cache_creation_input_tokens or 0,
        )

    def _update_usage(self, delta_usage: MessageDeltaUsage) -> None:
        if delta_usage.cache_creation_input_tokens is not None:
            self._usage.cache_creation_input_tokens = delta_usage.cache_creation_input_tokens
        if delta_usage.cache_read_input_tokens is not None:
            self._usage.cache_read_input_tokens = delta_usage.cache_read_input_tokens
        if delta_usage.input_tokens is not None:
            self._usage.input_tokens = delta_usage.input_tokens
        if delta_usage.output_tokens is not None:  # type: ignore
            self._usage.output_tokens = delta_usage.output_tokens

    async def _convert_non_stream_response(
        self,
        response: AnthropicMessage,
    ) -> AsyncIterator[StreamedMessagePart]:
        self._id = response.id
        self._usage = response.usage
        for block in response.content:
            match block.type:
                case "text":
                    yield TextPart(text=block.text)
                case "thinking":
                    yield ThinkPart(think=block.thinking, encrypted=block.signature)
                case "redacted_thinking":
                    yield ThinkPart(think="", encrypted=block.data)
                case "tool_use":
                    if isinstance(block.input, dict):
                        args = orjson.dumps(block.input).decode()
                    elif isinstance(block.input, str):
                        from kosong.utils.jsonx import loads_relaxed
                        try:
                            parsed = loads_relaxed(block.input)
                            if isinstance(parsed, dict):
                                args = block.input
                            else:
                                yield TextPart(
                                text=(
                                    f"Error: Tool call '{block.name}' returned non-object "
                                    f"input: {block.input}"
                                )
                                )
                                continue
                        except json.JSONDecodeError:
                            yield TextPart(
                                text=(
                                    f"Error: Tool call '{block.name}' returned invalid JSON "
                                    f"input: {block.input}"
                                )
                            )
                            continue
                    else:
                        yield TextPart(
                            text=(
                                f"Error: Tool call '{block.name}' returned unexpected input type "
                                f"{type(block.input).__name__}: {block.input}"
                            )
                        )
                        continue
                    yield ToolCall(
                        id=block.id,
                        function=ToolCall.FunctionBody(name=block.name, arguments=args),
                    )
                case _:
                    continue

    async def _convert_stream_response(
        self,
        manager: AsyncStream[RawMessageStreamEvent],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async with manager as stream:
                async for event in stream:
                    if isinstance(event, MessageStartEvent):
                        self._id = event.message.id
                        # Capture initial usage from start event
                        # (contains initial prompt/input token usage)
                        self._usage = event.message.usage
                    elif isinstance(event, RawContentBlockStartEvent):
                        block = event.content_block
                        match block.type:
                            case "text":
                                yield TextPart(text=block.text)
                            case "thinking":
                                yield ThinkPart(think=block.thinking)
                            case "redacted_thinking":
                                yield ThinkPart(think="", encrypted=block.data)
                            case "tool_use":
                                yield ToolCall(
                                    id=block.id,
                                    function=ToolCall.FunctionBody(name=block.name, arguments=""),
                                )
                            case "server_tool_use" | "web_search_tool_result":
                                # ignore
                                continue
                    elif isinstance(event, RawContentBlockDeltaEvent):
                        delta = event.delta
                        match delta.type:
                            case "text_delta":
                                yield TextPart(text=delta.text)
                            case "thinking_delta":
                                yield ThinkPart(think=delta.thinking)
                            case "input_json_delta":
                                yield ToolCallPart(arguments_part=delta.partial_json)
                            case "signature_delta":
                                yield ThinkPart(think="", encrypted=delta.signature)
                            case "citations_delta":
                                # ignore
                                continue
                    elif isinstance(event, MessageDeltaEvent):
                        if event.usage:
                            self._update_usage(event.usage)
                    elif isinstance(event, MessageStopEvent):
                        continue
        except (AnthropicError, httpx.HTTPError) as exc:
            raise _convert_error(exc) from exc


def _convert_tool(tool: Tool) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _is_tool_result_only(content: object) -> bool:
    """True iff ``content`` is a non-empty list of only tool_result blocks.

    Guards the parallel-tool-result merge in ``generate()``: we only collapse
    consecutive user messages when both sides carry pure tool results, never
    when a user message mixes in text, images, or anything else.
    """
    if not isinstance(content, list) or not content:
        return False
    blocks = cast(list[ContentBlockParam], content)
    return all(b["type"] == "tool_result" for b in blocks)


def _tool_result_message_to_block(
    tool_call_id: str, content: str | list[ContentPart]
) -> ToolResultBlockParam:
    block_content: str | list[ToolResultContent]
    # If tool_result_process is `extract_text`, we join all text parts into one string
    if isinstance(content, str):
        block_content = content
    else:
        # Otherwise, map parts to content blocks
        blocks: list[ToolResultContent] = []
        for part in content:
            if isinstance(part, TextPart):
                if part.text:
                    blocks.append(TextBlockParam(type="text", text=part.text))
            elif isinstance(part, ImageURLPart):
                blocks.append(_image_url_part_to_anthropic(part))
            else:
                # https://docs.claude.com/en/docs/build-with-claude/files#file-types-and-content-blocks
                # Anthropic API supports very limited file types
                blocks.append(
                    TextBlockParam(
                        type="text",
                        text=f"Error: Anthropic API does not support {type(part)} in tool result.",
                    )
                )
        block_content = blocks

    return ToolResultBlockParam(
        type="tool_result",
        tool_use_id=tool_call_id,
        content=block_content,
    )


def _image_url_part_to_anthropic(part: ImageURLPart) -> ToolResultContent:
    url = part.image_url.url
    # data:[<media-type>][;base64],<data>
    if url.startswith("data:"):
        res = url[5:].split(";base64,", 1)
        if len(res) != 2:
            return TextBlockParam(
                type="text",
                text=f"Error: Invalid data URL for image: {url}",
            )
        media_type, data = res
        if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            return TextBlockParam(
                type="text",
                text=f"Error: Unsupported media type for base64 image: {media_type}, url: {url}",
            )
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64",
                data=data,
                media_type=media_type,
            ),
        )
    else:
        return ImageBlockParam(
            type="image",
            source=URLImageSourceParam(type="url", url=url),
        )


def _response_headers(error: AnthropicError) -> Mapping[str, str] | None:
    response = getattr(error, "response", None)
    return getattr(response, "headers", None) if response is not None else None


def _convert_error(error: AnthropicError | httpx.HTTPError) -> ChatProviderError:
    # httpx errors may leak through the Anthropic SDK during streaming;
    # delegate to the shared converter.
    if isinstance(error, httpx.HTTPError):
        return convert_httpx_error(error)
    # Anthropic SDK errors — check subclasses before parents to avoid
    # misclassification (e.g. APITimeoutError inherits APIConnectionError).
    if isinstance(error, AnthropicAPIStatusError):
        return APIStatusError(
            error.status_code, str(error), headers=_response_headers(error)
        )
    if isinstance(error, AnthropicAuthenticationError):
        return APIStatusError(
            getattr(error, "status_code", 401), str(error), headers=_response_headers(error)
        )
    if isinstance(error, AnthropicPermissionDeniedError):
        return APIStatusError(
            getattr(error, "status_code", 403), str(error), headers=_response_headers(error)
        )
    if isinstance(error, AnthropicRateLimitError):
        return APIStatusError(
            getattr(error, "status_code", 429), str(error), headers=_response_headers(error)
        )
    if isinstance(error, AnthropicAPITimeoutError):
        return APITimeoutError(str(error))
    if isinstance(error, AnthropicAPIConnectionError):
        return APIConnectionError(str(error))
    return ChatProviderError(f"Anthropic error: {error}")
