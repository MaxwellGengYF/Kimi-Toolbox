"""Thin adapter around the public :mod:`kimi_agent_sdk` session API."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from kimix import close_session_async

try:
    from kimix import create_session_async  # type: ignore[attr-defined]
except ImportError:  # Local Kimix-CLI-X revisions export the worker factory privately.
    from kimix import base as _kimix_base
    from kimix.utils import _create_session_async

    async def create_session_async(**kwargs: Any) -> Any:
        kwargs.pop("model", None)
        thinking = bool(kwargs.pop("thinking", False))
        yolo = bool(kwargs.pop("yolo", False))
        previous_thinking = _kimix_base._default_thinking
        previous_yolo = _kimix_base._default_yolo
        _kimix_base._default_thinking = thinking
        _kimix_base._default_yolo = yolo
        try:
            return await _create_session_async(**kwargs)
        finally:
            _kimix_base._default_thinking = previous_thinking
            _kimix_base._default_yolo = previous_yolo


from kimi_cli.auth.codex import CodexAuthService, default_codex_auth_service

from kimix_gui.kimi_workdir import resolve_kimi_work_dir
from kimix_gui.llm import (
    LLMSelection,
    ProviderFileTarget,
    RuntimeOverrides,
    configured_selection,
    default_provider_file_path,
)
from kimix_gui.llm.providers.base import RuntimeLease, SessionRuntime
from kimix_gui.llm.registry import ProviderRegistry, default_provider_registry


class SdkSession(Protocol):
    """The public subset of ``kimi_agent_sdk.Session`` used by the UI."""

    @property
    def id(self) -> str: ...

    @property
    def status(self) -> Any: ...

    def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]: ...

    def cancel(self) -> None: ...

    async def clear(self, **custom_arguments: Any) -> None: ...

    async def compact(self, *, custom_instruction: str = "") -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionOptions:
    """Configuration for creating or resuming a public SDK session."""

    work_dir: Path
    session_id: str | None = None
    llm_selection: LLMSelection | None = None
    llm_runtime: RuntimeOverrides | None = None
    yolo: bool = False


class ManagedSdkSession:
    """Top-level SDK session that owns an external provider lease."""

    def __init__(self, session: SdkSession, lease: RuntimeLease) -> None:
        self._session = session
        self._lease = lease
        self._closed = False

    @property
    def id(self) -> str:
        return self._session.id

    @property
    def status(self) -> Any:
        return self._session.status

    @property
    def closed(self) -> bool:
        return self._closed

    def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        return self._session.prompt(user_input, merge_wire_messages=merge_wire_messages)

    def cancel(self) -> None:
        self._session.cancel()

    async def clear(self, **custom_arguments: Any) -> None:
        await self._session.clear(**custom_arguments)

    async def compact(self, *, custom_instruction: str = "") -> None:
        await self._session.compact(custom_instruction=custom_instruction)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await close_session_async(cast(Any, self._session))
        finally:
            await self._lease.close()


def new_session_id() -> str:
    """Return a compact persistent session id owned by this GUI."""

    return f"gui_{uuid4().hex[:12]}"


async def create_sdk_session(
    options: SessionOptions,
    *,
    codex_service: CodexAuthService | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> SdkSession:
    """Create or resume a session through one provider-neutral runtime path."""

    service = codex_service or default_codex_auth_service()
    registry = provider_registry or default_provider_registry(service)
    work_dir = resolve_kimi_work_dir(options.work_dir)
    selection = options.llm_selection or configured_selection(
        ProviderFileTarget(default_provider_file_path())
    )
    if options.llm_runtime is None and selection.parameters.entries:
        raise ValueError(
            "Selections with parameters require resolved llm_runtime overrides"
        )
    overrides = options.llm_runtime or RuntimeOverrides()
    session_id = options.session_id or new_session_id()
    runtime = await registry.create_runtime(
        selection.target,
        session_id=session_id,
        overrides=overrides,
    )
    try:
        session_kwargs: dict[str, Any] = {
            "work_dir": work_dir,
            "session_id": session_id,
            "resume": options.session_id is not None,
            "provider_dict": runtime.provider_dict,
            "model": runtime.model,
            "yolo": options.yolo,
        }
        if runtime.provider is not None:
            session_kwargs["chat_provider"] = runtime.provider
        session = await create_session_async(**session_kwargs)
    except BaseException:
        try:
            await _close_failed_runtime(runtime)
        except BaseException:
            pass
        raise
    sdk_session = cast(SdkSession, session)
    if runtime.lease is not None:
        return ManagedSdkSession(sdk_session, runtime.lease)
    return sdk_session


async def _close_failed_runtime(runtime: SessionRuntime) -> None:
    if runtime.lease is not None:
        await runtime.lease.close()
        return
    close = getattr(runtime.provider, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def close_sdk_session(session: SdkSession) -> None:
    """Close a Kimix-created session and remove it from the live-session registry."""

    if isinstance(session, ManagedSdkSession):
        await session.close()
        return
    await close_session_async(cast(Any, session))
