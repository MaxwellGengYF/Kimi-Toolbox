from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar

import regex as re

from kosong.chat_provider import APITimeoutError, StreamedMessagePart, TokenUsage
from kosong.message import Message

if TYPE_CHECKING:
    from kosong.message import ToolCall

type ToolMessageConversion = Literal["extract_text"]


def validate_tool_call_arguments(tool_calls: list[ToolCall]) -> list[str]:
    """Validate and sanitize tool call arguments. Returns error messages.

    Each tool call with non-empty arguments is parsed as JSON.  Invalid
    JSON and non-object values are both reset to ``"{}"`` and reported
    via the returned error list.
    """
    from kosong.utils.jsonx import loads_relaxed

    errors: list[str] = []
    for tc in tool_calls:
        if not tc.function.arguments:
            continue
        try:
            parsed = loads_relaxed(tc.function.arguments)
        except json.JSONDecodeError as exc:
            errors.append(
                f"Error: Tool call '{tc.function.name}' has invalid JSON arguments: {exc}"
            )
            tc.function.arguments = "{}"
            continue
        if not isinstance(parsed, dict):
            errors.append(
                f"Error: Tool call '{tc.function.name}' arguments must be a JSON object, "
                f"got {type(parsed).__name__}."
            )
            tc.function.arguments = "{}"
    return errors


def check_tool_call_id(
    tool_call_id: str | None, message_content: str
) -> str | None:
    """Return an error message if *tool_call_id* is missing, otherwise ``None``.

    This is a shared helper for the ``tool_call_id is None`` guard used across
    multiple providers.
    """
    if tool_call_id is None:
        return (
            f"Error: Tool message is missing `tool_call_id`. "
            f"Content: {message_content}"
        )
    return None


_EMPTY_TOOL_CALL_ID = "tool_call"
_TOOL_CALL_ID_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_CALL_ID_MAX_LENGTH = 64


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """Replace characters strict backends reject and truncate to the id budget."""
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


def normalize_tool_call_ids(history: Sequence[Message]) -> Sequence[Message]:
    """Rewrite invalid historical tool-call ids to a safe, portable shape.

    Histories persisted from other providers (or older sessions) can contain
    tool-call ids with characters strict backends reject (e.g. Moonshot 400s
    on ``Read:9``) or ids longer than 64 characters; sending them verbatim
    fails the whole request. Ids are sanitized to ``[a-zA-Z0-9_-]``,
    truncated, and made unique with ``_2``/``_3``... suffixes; assistant
    ``tool_calls`` entries and their matching ``tool`` messages are rewritten
    consistently.

    Providers whose own backends generate well-formed ids (OpenAI, Anthropic)
    apply the same normalization defensively so cross-provider histories can
    be replayed anywhere. Input messages are never mutated; when every id is
    already valid the original sequence object is returned unchanged.
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


T = TypeVar("T")

_STREAM_ITERATION_TIMEOUT_DEFAULT = 60.0
_STREAM_ITERATION_TIMEOUT_ENV = "KOSONG_STREAM_ITERATION_TIMEOUT"


def get_stream_iteration_timeout() -> float:
    """Return the per-chunk timeout for chat provider streaming iterators.

    Reads ``KOSONG_STREAM_ITERATION_TIMEOUT`` from the environment; falls back
    to a 60-second default. Unparseable values are ignored.
    """
    raw = os.environ.get(_STREAM_ITERATION_TIMEOUT_ENV)
    if raw is None:
        return _STREAM_ITERATION_TIMEOUT_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return _STREAM_ITERATION_TIMEOUT_DEFAULT


class _StreamTimeoutIterator(AsyncIterator[T]):
    """Wrap an async iterator so each ``__anext__`` has a per-item timeout.

    This prevents a stalled network stream from blocking the caller forever
    when the underlying SDK stream waits indefinitely for the next chunk.
    """

    __slots__ = ("_iterator", "_timeout")

    def __init__(self, iterator: AsyncIterator[T], timeout: float) -> None:
        self._iterator = iterator
        self._timeout = timeout

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await asyncio.wait_for(self._iterator.__anext__(), timeout=self._timeout)
        except TimeoutError as exc:
            raise APITimeoutError(
                f"Stream stalled: no data received for {self._timeout:.3g}s"
            ) from exc


def with_stream_timeout[T](
    iterator: AsyncIterator[T], timeout: float | None = None
) -> AsyncIterator[T]:
    """Wrap *iterator* with a per-item timeout.

    Args:
        iterator: The async iterator to wrap.
        timeout: Per-item timeout in seconds. If ``None``, uses
            :func:`get_stream_iteration_timeout`.
    """
    if timeout is None:
        timeout = get_stream_iteration_timeout()
    return _StreamTimeoutIterator(iterator, timeout=timeout)


class BaseStreamedMessage:
    """Mixin / base class for provider-specific streamed messages.

    Provides the common ``__aiter__`` / ``__anext__`` / ``id`` boilerplate.
    Subclasses must set ``self._iter`` in ``__init__``.

    The underlying iterator is wrapped with a per-item timeout so that a stalled
    network stream (e.g. the remote server stops sending SSE chunks) is surfaced
    as :class:`~kosong.chat_provider.APITimeoutError` instead of hanging the
    caller forever.
    """

    _iter: AsyncIterator[StreamedMessagePart]
    _id: str | None = None
    _timeout_iter: AsyncIterator[StreamedMessagePart] | None = None

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

    @staticmethod
    def _build_token_usage(
        *,
        input_other: int,
        output: int,
        input_cache_read: int = 0,
        input_cache_creation: int = 0,
    ) -> TokenUsage:
        """Canonical factory for ``TokenUsage``.

        Subclass ``usage`` properties call this with provider-specific
        extraction logic so that edge-case handling (None-vs-0, negation,
        etc.) stays consistent across providers.
        """
        return TokenUsage(
            input_other=input_other,
            output=output,
            input_cache_read=input_cache_read,
            input_cache_creation=input_cache_creation,
        )
