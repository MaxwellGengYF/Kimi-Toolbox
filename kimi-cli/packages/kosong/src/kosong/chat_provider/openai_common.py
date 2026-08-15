import asyncio
import contextlib
import copy
import json
import regex as re
import ssl
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard, cast

import certifi
import httpx
import openai
from openai import AsyncOpenAI, AsyncStream, OpenAIError
from pydantic import ValidationError
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionToolParam,
)
from openai.types.completion_usage import CompletionUsage

from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
    DEFAULT_MAX_RETRIES,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
    convert_httpx_error,
)
from kosong.contrib.chat_provider.common import BaseStreamedMessage
from kosong.message import (
    ContentPart,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
)
from kosong.tooling import Tool

from typing_extensions import TypedDict


class CommonGenerationKwargs(TypedDict, total=False):
    """Shared generation kwargs for OpenAI-compatible chat providers.

    Provider-specific ``GenerationKwargs`` TypedDicts can extend this to
    inherit the common fields while adding their own proprietary ones.
    """

    max_tokens: int | None
    max_completion_tokens: int | None
    temperature: float | None
    top_p: float | None


# Safe upper bound for ``max_tokens`` / ``max_completion_tokens`` in
# OpenAI-compatible APIs.  Some backends (e.g. Moonshot) accept up to
# ~393k, while others have lower caps.  384k is generous enough to
# prevent "think-only" errors (token budget exhaustion during reasoning)
# while remaining well within the limits of every mainstream API.
_MAX_OUTPUT_TOKENS = 384_000


def clamp_max_tokens(kwargs: dict[str, Any]) -> None:
    """Clamp output-token budgets in *kwargs* to a safe upper bound.

    Covers ``max_tokens`` (the legacy field),
    ``max_completion_tokens`` (the modern field recommended for reasoning
    models), and ``max_output_tokens`` (used by the OpenAI Responses API).
    The ``llm.py`` layer may default these to the model's total
    context size (which can be 1 M+ tokens), but the API's output budget
    is typically capped far below that.  Sending an over-large value
    causes a ``400 BadRequest``.
    """
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        raw = kwargs.get(key)
        if raw is not None and raw > _MAX_OUTPUT_TOKENS:
            kwargs[key] = _MAX_OUTPUT_TOKENS


_SSL_CONTEXT: ssl.SSLContext | None = None


def _get_ssl_context() -> ssl.SSLContext:
    """Cached SSL context to avoid repeated CA bundle loading on client re-creation."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    return _SSL_CONTEXT


def create_openai_client(
    *,
    api_key: str | None,
    base_url: str | None,
    client_kwargs: Mapping[str, Any],
) -> AsyncOpenAI:
    kwargs = dict(client_kwargs)
    if "http_client" not in kwargs:
        kwargs["http_client"] = httpx.AsyncClient(verify=_get_ssl_context())
    return AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)


_CLIENT_CLOSE_TASKS: set[asyncio.Task[None]] = set()


def _on_close_task_done(task: asyncio.Task[None]) -> None:
    _CLIENT_CLOSE_TASKS.discard(task)
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


async def _drain_awaitable(awaitable: Awaitable[object]) -> None:
    try:
        await asyncio.wait_for(awaitable, timeout=5.0)
    except asyncio.TimeoutError:
        return
    except asyncio.CancelledError:
        # Outer task was cancelled (e.g. during event-loop shutdown).
        # Swallow silently — the OS will reclaim the sockets.
        return
    except RuntimeError as exc:
        # On Windows/Python 3.14, closing an httpx.AsyncClient whose
        # underlying transports were bound to a now-closed ProactorEventLoop
        # raises RuntimeError('Event loop is closed').  Swallow it — the OS
        # will reclaim the socket.
        if "Event loop is closed" in str(exc):
            return
        raise
    except Exception:
        return


def close_openai_client(client: AsyncOpenAI) -> None:
    """Schedule an async close of the given AsyncOpenAI client.

    ``AsyncOpenAI.close()`` is always a callable that returns an awaitable,
    so we skip the ``getattr`` / ``callable`` / ``isawaitable`` guards.
    """
    try:
        result = client.close()
    except Exception:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running.  The client's original loop is gone.
        # Creating a new loop to close it will fail for any transport
        # bound to the old loop (ProactorEventLoop on Windows raises
        # RuntimeError('Event loop is closed')).  Just abandon the
        # client — the OS will clean up the sockets on process exit.
        return
    # Loop is running but we're in a sync context (e.g. on_retryable_error).
    # Create a task and keep a strong reference so it can run to completion.
    task = loop.create_task(_drain_awaitable(cast(Awaitable[object], result)))
    _CLIENT_CLOSE_TASKS.add(task)
    task.add_done_callback(_on_close_task_done)


def close_replaced_openai_client(client: AsyncOpenAI, *, client_kwargs: Mapping[str, Any]) -> None:
    """
    Close a replaced OpenAI client unless it would close a shared external http client.

    When callers pass `http_client=...` to `AsyncOpenAI`, multiple wrappers may share the same
    `httpx.AsyncClient`. Closing the replaced wrapper would also close that shared client and
    break the new wrapper immediately.
    """
    shared_http_client = client_kwargs.get("http_client")
    if isinstance(shared_http_client, httpx.AsyncClient) and getattr(client, "_client", None) is (
        shared_http_client
    ):
        return
    close_openai_client(client)


def convert_error(error: OpenAIError | httpx.HTTPError) -> ChatProviderError:
    # httpx errors may leak through the OpenAI SDK during streaming;
    # delegate to the shared converter.
    if isinstance(error, httpx.HTTPError):
        return convert_httpx_error(error)
    # OpenAI SDK errors — check subclasses before parents to avoid
    # misclassification (e.g. APITimeoutError inherits APIConnectionError).
    match error:
        case openai.APIStatusError():
            response_headers = error.response.headers
            req_id = response_headers.get("x-request-id")
            return APIStatusError(
                error.status_code,
                error.message,
                request_id=req_id,
                headers=response_headers,
            )
        case openai.APITimeoutError():
            return APITimeoutError(error.message)
        case openai.APIConnectionError():
            return APIConnectionError(error.message)
        case openai.APIError() if type(error) is openai.APIError and error.body is None:
            # Base APIError with no body indicates a transport-layer failure
            # (e.g. "Network connection lost." during streaming).  SSE error
            # events from the server carry a body dict and should fall through
            # to the default case instead.
            return _classify_base_api_error(error.message)
        case _:
            return ChatProviderError(f"Error: {error}")


_MISSING_REASONING_CONTENT_RE = re.compile(
    r"reasoning_content\s+in\s+the\s+thinking\s+mode\s+must\s+be\s+passed\s+back\s+to\s+the\s+api",
    re.IGNORECASE,
)


def _is_missing_reasoning_content_error(error: openai.APIStatusError) -> bool:
    """Detect the Moonshot/Kimi 400 when thinking-mode reasoning_content is omitted."""
    if error.status_code != 400:
        return False
    # Backends vary in whether they quote `reasoning_content` with backticks.
    message = (error.message or "").replace("`", "")
    return bool(_MISSING_REASONING_CONTENT_RE.search(message))


def maybe_log_reasoning_content_error(
    error: OpenAIError | httpx.HTTPError,
    *,
    provider_name: str,
    model: str,
    messages: Sequence[Any],
    generation_kwargs: Mapping[str, Any],
) -> None:
    """Log details of a missing-reasoning_content 400 to ./error.log.

    This is a debugging aid for the specific Moonshot/Kimi-compatible backend
    error that occurs when thinking mode is enabled but a previous assistant
    reasoning block was not passed back in the request messages.
    """
    if not isinstance(error, openai.APIStatusError):
        return
    if not _is_missing_reasoning_content_error(error):
        return

    log_path = Path("error.log")
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": provider_name,
        "model": model,
        "error": {
            "status_code": error.status_code,
            "message": error.message,
            "code": getattr(error, "code", None),
            "type": getattr(error, "type", None),
            "body": error.body,
        },
        "messages": list(messages),
        "generation_kwargs": dict(generation_kwargs),
    }
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Logging is best-effort; never let it break the actual error handling.
        pass


_NETWORK_RE = re.compile(r"network|connection|connect|disconnect", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"timed?\s*out|timeout|deadline", re.IGNORECASE)


class _TolerantSSEDecoder:
    """Minimal SSE decoder for OpenAI-compatible chat streams.

    The openai SDK's ``SSEDecoder`` / ``ServerSentEvent.json()`` aborts the
    entire stream on the first event whose ``data:`` payload is empty or not
    valid JSON.  Some backends (observed with Moonshot/Kimi during long
    compaction requests) emit such keep-alive events mid-stream, which turns a
    healthy response into a ``json.JSONDecodeError``.  This decoder only
    extracts ``event`` / ``data`` fields and leaves payload interpretation to
    the caller, so unparsable events can be skipped instead of killing the
    response.
    """

    __slots__ = ("_data", "_event")

    def __init__(self) -> None:
        self._data: list[str] = []
        self._event: str | None = None

    def feed_block(self, block: bytes) -> Iterator[tuple[str | None, str]]:
        """Decode one blank-line-terminated SSE block into ``(event, data)`` pairs."""
        for raw_line in block.splitlines():
            line = raw_line.decode("utf-8")
            if not line:
                # Blank line: flush the accumulated event (if any).
                if not self._data and self._event is None:
                    continue
                event = self._event
                payload = "\n".join(self._data)
                self._data = []
                self._event = None
                yield event, payload
                continue
            if line.startswith(":"):
                # SSE comments (e.g. keep-alives) carry no payload.
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                self._data.append(value)
            elif field == "event":
                self._event = value
            # ``id`` / ``retry`` are not needed for chat completions; ignored.


async def _iter_sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str | None, str]]:
    """Iterate SSE ``(event, data)`` pairs from an httpx streaming response.

    Handles events split across arbitrary network chunks and multiple events
    per chunk, mirroring the openai SDK's ``SSEDecoder`` behavior without ever
    raising on empty/whitespace ``data:`` payloads.
    """
    decoder = _TolerantSSEDecoder()
    buffer = b""
    async for chunk in response.aiter_bytes():
        for line in chunk.splitlines(keepends=True):
            buffer += line
            if buffer.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                for event in decoder.feed_block(buffer):
                    yield event
                buffer = b""
    if buffer:
        for event in decoder.feed_block(buffer):
            yield event


def _is_object_mapping(data: object) -> TypeGuard[Mapping[str, object]]:
    """Return True if *data* is a JSON object (dict with object values)."""
    return isinstance(data, dict)


# OpenAI's ChatCompletionChunk.Choice only accepts these finish_reason values.
# Some OpenAI-compatible backends (observed with Moonshot/Kimi during tool calls)
# emit non-standard values such as ``unexpected_state``; normalize those to
# ``None`` so the stream can continue instead of raising a pydantic validation
# error.
_VALID_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


def _normalize_unknown_finish_reason(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Replace unknown ``finish_reason`` strings in *payload* with ``None``.

    The returned mapping is a shallow copy only when mutation is required;
    otherwise the original mapping is returned unchanged.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    mutated = False
    new_choices: list[object] = []
    for choice in choices:
        if not isinstance(choice, dict):
            new_choices.append(choice)
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason not in _VALID_FINISH_REASONS:
            if not mutated:
                # First mutation: copy the payload and previous choices.
                new_payload = dict(payload)
                new_payload["choices"] = new_choices[:]
                payload = new_payload
                mutated = True
            cast("dict[str, object]", choice)["finish_reason"] = None
        new_choices.append(choice)
    if mutated:
        cast("dict[str, object]", payload)["choices"] = new_choices
    return payload


async def _iter_tolerant_chunks(
    stream: AsyncStream[Any],
) -> AsyncIterator[ChatCompletionChunk]:
    """Iterate an OpenAI ``AsyncStream``, skipping SSE events with invalid JSON.

    Some OpenAI-compatible backends (observed with Moonshot/Kimi during long
    compaction requests) emit keep-alive SSE events whose ``data:`` payload is
    empty or otherwise not valid JSON.  The openai SDK's own
    ``AsyncStream.__stream__`` calls ``json.loads`` on every such event and
    raises ``json.JSONDecodeError``, aborting the whole response.  This
    re-implements the same chunk-processing loop (including ``[DONE]`` and
    error-payload handling) over the underlying httpx response with a tolerant
    JSON step, so malformed keep-alive events are skipped instead of killing
    the stream.
    """
    response = stream.response
    try:
        async for _event, data in _iter_sse_events(response):
            if data.startswith("[DONE]"):
                break
            try:
                payload: object = json.loads(data)
            except json.JSONDecodeError:
                # Keep-alive / empty data event — nothing to parse, skip it.
                continue
            if not _is_object_mapping(payload):
                # Malformed payload — skip rather than abort the whole stream.
                continue
            if payload.get("error"):
                error = payload.get("error")
                message: object | None = None
                if _is_object_mapping(error):
                    message = error.get("message")
                if not isinstance(message, str):
                    message = "An error occurred during streaming"
                raise openai.APIError(
                    message=message,
                    request=response.request,
                    body=payload["error"],
                )
            payload = _normalize_unknown_finish_reason(cast(Mapping[str, object], payload))
            try:
                yield ChatCompletionChunk.model_validate(payload)
            except ValidationError as exc:
                # Provide a concise backend-facing error instead of the full
                # pydantic traceback. Re-raise only the first issue to keep the
                # message short and actionable.
                first_error = exc.errors()[0] if exc.errors() else {"loc": ()}
                loc = ".".join(str(part) for part in first_error.get("loc", ()))
                msg = first_error.get("msg", "invalid response chunk")
                if loc:
                    raise ChatProviderError(f"Backend returned an invalid chat stream chunk at {loc}: {msg}")
                raise ChatProviderError(f"Backend returned an invalid chat stream chunk: {msg}")
    finally:
        await response.aclose()


def _classify_base_api_error(message: str) -> ChatProviderError:
    """Heuristically map an ``openai.APIError`` message to a retryable error type.

    Timeout patterns are checked first because a message like
    "connection timed out" should be classified as a timeout, not a
    connection error.
    """
    if _TIMEOUT_RE.search(message):
        return APITimeoutError(message)
    if _NETWORK_RE.search(message):
        return APIConnectionError(message)
    return ChatProviderError(f"Error: {message}")


def clamp_thinking_effort(
    effort: ThinkingEffort,
    supported: set[ThinkingEffort] | None,
) -> ThinkingEffort:
    """Clamp a thinking effort to the highest level accepted by the model.

    ``off`` is always returned unchanged because it disables thinking rather
    than selecting an effort rank. When ``supported`` is ``None`` or includes
    the requested effort, the effort passes through unchanged. Otherwise the
    highest supported non-off level is chosen, falling back to ``high`` when
    nothing else is available.
    """
    if effort == "off":
        return effort
    if supported is None:
        return effort
    if effort in supported:
        return effort
    # Fall back to the highest supported non-off level.
    for candidate in ("max", "xhigh", "high", "medium", "low"):
        if candidate in supported:
            return candidate  # type: ignore[return-value]
    return "high"


def thinking_effort_to_reasoning_effort(effort: ThinkingEffort) -> ReasoningEffort:
    if effort == 'off':
        return None
    return str(effort)


def reasoning_effort_to_thinking_effort(effort: ReasoningEffort) -> ThinkingEffort:
    match effort:
        case "low" | "minimal":
            return "low"
        case "medium":
            return "medium"
        case "high":
            return "high"
        case "xhigh":
            return "xhigh"
        case "max":
            return "max"
        case "none" | None:
            return "off"


def tool_to_openai(tool: Tool) -> ChatCompletionToolParam:
    """Convert a single tool to OpenAI tool format."""
    # simply `model_dump` because the `Tool` type is OpenAI-compatible
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def apply_generation_kwargs(self: Any, *, attr: str = "_generation_kwargs", **kwargs: Any) -> Any:
    """Copy *self* with updated generation kwargs.

    This is the shared implementation of the ``with_generation_kwargs``
    pattern used across all chat providers.  Returns a shallow copy of
    *self* whose *attr* dict is a deep copy of the original, merged with
    *kwargs*.
    """
    new_self = copy.copy(self)
    new_kwargs = copy.deepcopy(getattr(self, attr))
    new_kwargs.update(kwargs)
    setattr(new_self, attr, new_kwargs)
    return new_self


class OpenAICompatibleProviderMixin:
    """Mix-in for any chat provider backed by an ``AsyncOpenAI`` client.

    Provides canonical implementations of :meth:`on_retryable_error` and
    :meth:`model_parameters`, plus a helper to initialise the client during
    ``__init__``.

    Subclasses must store generation kwargs in ``self._generation_kwargs``
    (the standard pattern across all providers).
    """

    def _init_openai_client(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        client_kwargs: Mapping[str, Any],
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client_kwargs: dict[str, Any] = dict(client_kwargs)
        # Apply a default SDK-level retry budget for transient errors such as
        # 429 Rate Limit and 5xx server errors.  Callers may override this by
        # passing ``max_retries`` explicitly in ``client_kwargs``.
        self._client_kwargs.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        self.client: AsyncOpenAI = create_openai_client(
            api_key=self._api_key,
            base_url=self._base_url,
            client_kwargs=self._client_kwargs,
        )

    def on_retryable_error(self, error: BaseException) -> bool:
        old_client = self.client
        # Read api_key from the live client (not self._api_key) so that
        # OAuth token refreshes applied via client.api_key are preserved.
        current_api_key = old_client.api_key
        self.client = create_openai_client(
            api_key=current_api_key,
            base_url=self._base_url,
            client_kwargs=self._client_kwargs,
        )
        self._api_key = current_api_key
        close_replaced_openai_client(old_client, client_kwargs=self._client_kwargs)
        return True

    @property
    def model_parameters(self) -> dict[str, Any]:
        """Parameters of the underlying model (for tracing / logging)."""
        model_parameters: dict[str, Any] = {"base_url": str(self.client.base_url)}
        model_parameters.update(self._generation_kwargs)
        return model_parameters

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI HTTP client.

        Safe to call multiple times; subsequent calls are no-ops. Errors during
        close (including the harmless ``RuntimeError: Event loop is closed``
        that can occur on Windows/Python 3.14 during shutdown) are swallowed.
        """
        try:
            await self.client.close()
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise
        except asyncio.CancelledError:
            return


def extract_reasoning_from_content(
    content: list[ContentPart],
) -> tuple[str, list[ContentPart]]:
    """Separate ThinkPart content from visible content parts.

    Returns a ``(reasoning_text, visible_parts)`` tuple where *reasoning_text*
    is the concatenated text of all ``ThinkPart`` items and *visible_parts*
    contains every non-``ThinkPart`` entry in its original order.
    """
    reasoning = ""
    visible: list[ContentPart] = []
    for part in content:
        if isinstance(part, ThinkPart):
            reasoning += part.think
        else:
            visible.append(part)
    return reasoning, visible


def is_effectively_empty_content_parts(content: Sequence[ContentPart]) -> bool:
    """Return True if *content* contains no visible text.

    A message whose visible content is empty or consists only of whitespace
    text parts is treated as having no visible content. This is used when
    deciding whether to omit the ``content`` field for assistant tool-call
    messages, which some backends reject when paired with an empty text part.
    """
    for part in content:
        if not isinstance(part, TextPart):
            return False
        if part.text.strip():
            return False
    return True


class BufferedChatCompletionToolCall(TypedDict, total=False):
    """Per-stream-index buffering state for streamed tool-call deltas."""

    id: str | None
    arguments: str
    emitted: bool


def convert_chat_completion_stream_tool_call(
    tool_call: Any,
    buffered_by_index: dict[int, BufferedChatCompletionToolCall],
) -> list[StreamedMessagePart]:
    """Convert an OpenAI Chat Completions-style streamed tool-call delta into
    the normalized kosong stream-part protocol.

    OpenAI-compatible providers may emit argument chunks before the function
    name for a stream index. Buffer those early argument chunks until the
    first named header arrives, then emit the header with the buffered
    arguments prepended so no argument text is lost. After the header,
    subsequent argument chunks are emitted as ``ToolCallPart`` deltas.
    """
    function = tool_call.function
    if function is None:
        return []

    stream_index = getattr(tool_call, "index", None)
    name = function.name
    arguments = function.arguments
    has_name = isinstance(name, str) and len(name) > 0
    has_arguments = isinstance(arguments, str) and len(arguments) > 0

    if stream_index is None:
        if has_name:
            return [
                ToolCall(
                    id=tool_call.id or str(uuid.uuid4()),
                    function=ToolCall.FunctionBody(name=name, arguments=arguments),
                )
            ]
        if has_arguments:
            return [ToolCallPart(arguments_part=arguments)]
        return []

    buffered = buffered_by_index.get(stream_index)
    if buffered is None:
        buffered = BufferedChatCompletionToolCall(id=None, arguments="", emitted=False)
    if tool_call.id is not None:
        buffered["id"] = tool_call.id

    if not buffered["emitted"]:
        if not has_name:
            # Argument chunks arriving before the function name — buffer them
            # so they can be prepended to the header once the name arrives.
            if has_arguments:
                buffered["arguments"] = buffered["arguments"] + arguments
            buffered_by_index[stream_index] = buffered
            return []
        buffered["emitted"] = True
        initial_arguments: str | None
        if buffered["arguments"]:
            initial_arguments = buffered["arguments"] + (arguments or "")
        else:
            initial_arguments = arguments or None
        buffered["arguments"] = ""
        buffered_by_index[stream_index] = buffered
        return [
            ToolCall(
                id=buffered["id"] or tool_call.id or str(uuid.uuid4()),
                function=ToolCall.FunctionBody(name=name, arguments=initial_arguments),
            )
        ]

    if not has_arguments:
        return []
    return [ToolCallPart(arguments_part=arguments)]


def extract_usage_from_chunk(chunk: ChatCompletionChunk) -> CompletionUsage | None:
    """Extract token usage from a streaming ``ChatCompletionChunk``.

    OpenAI-compatible APIs may place usage info at the top-level ``usage``
    field (standard) or nest it inside the first choice's model dump (some
    compatibility layers).  This helper handles both formats.
    """
    if chunk.usage:
        return chunk.usage
    if not chunk.choices:
        return None
    choice_dump: dict[str, object] = chunk.choices[0].model_dump()
    raw_usage = choice_dump.get("usage")
    if isinstance(raw_usage, CompletionUsage):
        return raw_usage
    if isinstance(raw_usage, dict):
        return CompletionUsage.model_validate(raw_usage)
    return None


class OpenAICompatibleStreamedMessage(BaseStreamedMessage):
    """Base class for streamed messages using the OpenAI Chat Completions wire format.

    Handles both streaming and non-streaming responses, text / reasoning /
    tool-call delta processing, and usage extraction.  Subclasses only need
    to supply *reasoning_key* (e.g. ``"reasoning_content"`` for Kimi) and,
    optionally, override :meth:`usage` for provider-specific cache-token
    extraction.
    """

    def __init__(
        self,
        response: ChatCompletion | AsyncStream[ChatCompletionChunk],
        *,
        reasoning_key: str | None = None,
    ):
        super().__init__()
        self._reasoning_key: str | None = reasoning_key
        if isinstance(response, ChatCompletion):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._usage: CompletionUsage | None = None

    # -- usage (OpenAI-standard CompletionUsage → TokenUsage) ------------------

    @property
    def usage(self) -> TokenUsage | None:
        """Derive ``TokenUsage`` from the collected ``CompletionUsage``.

        The default implementation handles the standard OpenAI caching
        schema (``prompt_tokens_details.cached_tokens``).  Providers whose
        models surface caching via non-standard attributes (e.g. Moonshot's
        legacy ``cached_tokens`` field) should override this property.
        """
        if self._usage:
            cached = 0
            total_input = self._usage.prompt_tokens
            if (
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

    # -- non-streaming conversion ---------------------------------------------

    async def _convert_non_stream_response(
        self,
        response: ChatCompletion,
    ) -> AsyncIterator[StreamedMessagePart]:
        self._id = response.id
        self._usage = response.usage
        message = response.choices[0].message
        reasoning_key = self._reasoning_key
        reasoning_content = getattr(message, reasoning_key, None) if reasoning_key else None
        # Yield on presence, not truthiness: an explicitly empty
        # reasoning_content must round-trip as an empty ThinkPart so the
        # next request passes the field back (Moonshot rejects thinking-mode
        # histories whose assistant messages lost their reasoning field).
        if reasoning_content is not None:
            assert isinstance(reasoning_content, str)
            yield ThinkPart(think=reasoning_content)
        if message.content:
            yield TextPart(text=message.content)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    yield ToolCall(
                        id=tool_call.id or str(uuid.uuid4()),
                        function=ToolCall.FunctionBody(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )

    # -- streaming conversion -------------------------------------------------

    async def _convert_stream_response(
        self,
        response: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[StreamedMessagePart]:
        buffered_tool_calls: dict[int, BufferedChatCompletionToolCall] = {}
        try:
            if isinstance(response, AsyncStream):
                # The openai SDK's AsyncStream aborts the whole response on the
                # first SSE event with an empty/invalid ``data:`` payload (some
                # backends emit such keep-alive events during long requests);
                # iterate through the tolerant wrapper instead.
                chunk_iter: AsyncIterator[ChatCompletionChunk] = _iter_tolerant_chunks(response)
            else:
                chunk_iter = response
            async for chunk in chunk_iter:
                if chunk.id:
                    self._id = chunk.id
                if usage := extract_usage_from_chunk(chunk):
                    self._usage = usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # extract reasoning / thinking content
                reasoning_key = self._reasoning_key
                reasoning_content = getattr(delta, reasoning_key, None) if reasoning_key else None
                # Same presence-vs-truthiness rule as the non-stream path:
                # preserve an explicitly empty reasoning_content delta.
                if reasoning_content is not None:
                    assert isinstance(reasoning_content, str)
                    yield ThinkPart(think=reasoning_content)

                # extract text content
                if delta.content:
                    yield TextPart(text=delta.content)

                # extract tool-call deltas
                for tool_call in delta.tool_calls or []:
                    for part in convert_chat_completion_stream_tool_call(
                        tool_call, buffered_tool_calls
                    ):
                        yield part
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e
