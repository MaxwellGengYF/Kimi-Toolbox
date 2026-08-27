"""Thin adapter around the public :mod:`kimi_agent_sdk` session API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from kimix import close_session_async

try:
    from kimix import create_session_async
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
from kimi_cli.llm_codex import CodexProviderLease, create_codex_provider

from kimix_gui.kimi_workdir import resolve_kimi_work_dir
from kimix_gui.llm_config import (
    ChatGPTSource,
    ConfigFileSource,
    LLMSource,
    default_config_path,
    load_kimix_provider_dict,
)


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


@dataclass(frozen=True, slots=True, init=False)
class SessionOptions:
    """Configuration for creating or resuming a public SDK session."""

    work_dir: Path
    session_id: str | None = None
    llm_source: LLMSource | None = None
    thinking: bool = False
    yolo: bool = False

    def __init__(
        self,
        work_dir: Path,
        session_id: str | None = None,
        llm_source: LLMSource | None = None,
        thinking: bool = False,
        yolo: bool = False,
        *,
        config_file: Path | None = None,
        model: str | None = None,
    ) -> None:
        """Normalize legacy constructor arguments at the application boundary.

        ``config_file`` and ``model`` remain accepted for callers using the public
        CLI-era shape, but are never stored as a second source of truth.
        """

        if llm_source is not None and (config_file is not None or model is not None):
            raise ValueError("llm_source cannot be combined with config_file or model")
        source = llm_source
        if source is None and (config_file is not None or model is not None):
            source = ConfigFileSource(config_file or default_config_path(), model)
        object.__setattr__(self, "work_dir", work_dir)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "llm_source", source)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "yolo", yolo)

    @property
    def config_file(self) -> Path | None:
        source = self.llm_source
        return source.path if isinstance(source, ConfigFileSource) else None

    @property
    def model(self) -> str | None:
        source = self.llm_source
        if isinstance(source, ConfigFileSource):
            return source.model_override
        if isinstance(source, ChatGPTSource):
            return source.model
        return None


class ManagedSdkSession:
    """Top-level SDK session that owns the final close of a shared Codex provider."""

    def __init__(self, session: SdkSession, lease: CodexProviderLease) -> None:
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
            await close_session_async(self._session)
        finally:
            await self._lease.close()


def new_session_id() -> str:
    """Return a compact persistent session id owned by this GUI."""

    return f"gui_{uuid4().hex[:12]}"


async def create_sdk_session(
    options: SessionOptions,
    *,
    codex_service: CodexAuthService | None = None,
) -> SdkSession:
    """Create or resume a session through Kimix's public Worker factory."""

    work_dir = resolve_kimi_work_dir(options.work_dir)
    source = options.llm_source
    session_id = options.session_id or new_session_id()
    if isinstance(source, ChatGPTSource):
        runtime = await create_codex_provider(
            codex_service or default_codex_auth_service(),
            model_name=source.model,
            session_id=session_id,
            thinking=options.thinking,
        )
        try:
            session = await create_session_async(
                work_dir=work_dir,
                session_id=session_id,
                resume=options.session_id is not None,
                provider_dict=runtime.provider_dict,
                model=source.model,
                thinking=options.thinking,
                yolo=options.yolo,
                chat_provider=runtime.provider,
            )
        except BaseException:
            await runtime.lease.close()
            raise
        return ManagedSdkSession(session, runtime.lease)

    provider_dict = None
    if source is not None and (
        source.path.is_file() or source.path.resolve(strict=False) != default_config_path()
    ):
        provider_dict = load_kimix_provider_dict(source.path)
    model = source.model_override if isinstance(source, ConfigFileSource) else None
    if provider_dict is not None and model is not None:
        provider_dict["model"] = model
    common: dict[str, Any] = {
        "provider_dict": provider_dict,
        "model": model,
        "thinking": options.thinking,
        "yolo": options.yolo,
    }

    return await create_session_async(
        work_dir=work_dir,
        session_id=session_id,
        resume=options.session_id is not None,
        **common,
    )


async def close_sdk_session(session: SdkSession) -> None:
    """Close a Kimix-created session and remove it from the live-session registry."""

    if isinstance(session, ManagedSdkSession):
        await session.close()
        return
    await close_session_async(session)
