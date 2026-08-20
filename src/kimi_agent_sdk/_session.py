from __future__ import annotations

import asyncio
import enum
import inspect
import logging
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from kaos.path import KaosPath
from kimi_cli.app import KimiCLI
from kimi_cli.config import Config
from kimi_cli.llm import LLM
from kimi_cli.safety_check import sanitize_for_tokenizer
from kimi_cli.session import Session as CliSession
from kimi_cli.soul import SessionRestartRequired, StatusSnapshot
from kimi_cli.wire.types import ContentPart, TextPart, ThinkPart, WireMessage
from kosong.chat_provider import ChatProvider

from kimi_agent_sdk._exception import SessionStateError

logger = logging.getLogger(__name__)

_prompt_semaphore = asyncio.Semaphore(5)

if TYPE_CHECKING:
    from kimi_agent_sdk import MCPConfig


class ExportFormat(enum.Enum):
    """Format for session export."""

    Markdown = "markdown"
    """Export as a human-readable Markdown file."""

    Jsonl = "jsonl"
    """Export as a JSON Lines file (machine-readable)."""


def _ensure_type(name: str, value: object, expected: type) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}, got {type(value).__name__}")


def _resolve_skills_dirs(
    skills_dir: KaosPath | None,
    skills_dirs: list[KaosPath] | None,
) -> list[KaosPath] | None:
    resolved: list[KaosPath] = []

    if skills_dir is not None:
        _ensure_type("skills_dir", skills_dir, KaosPath)
        resolved.append(skills_dir)

    if skills_dirs is not None:
        _ensure_type("skills_dirs", skills_dirs, list)
        for idx, item in enumerate(skills_dirs):
            _ensure_type(f"skills_dirs[{idx}]", item, KaosPath)
        resolved.extend(skills_dirs)

    return resolved or None


async def _load_config_json(work_dir: KaosPath) -> dict[str, Any]:
    """Load custom config from ``.kimix/config.json`` and wrap it under ``config_json``."""
    config_path = work_dir / ".kimix" / "config.json"
    config_json: dict[str, Any] = {}
    try:
        raw = await config_path.read_bytes()
        loaded = orjson.loads(raw)
        if isinstance(loaded, dict):
            config_json = loaded
    except (OSError, orjson.JSONDecodeError, ValueError):
        pass
    return {"config_json": config_json}


from kimi_cli.soul.context_records import ExportedContext  # noqa: E402, F401


class Session:
    """
    Kimi Agent session with low-level control.

    Use this class when you need full access to Wire messages, manual approval
    handling, or session persistence across prompts.
    """

    def __init__(self, cli: KimiCLI) -> None:
        self._cli = cli
        self._cancel_event: asyncio.Event | None = None
        self._closed = False
        self._create_kwargs: dict[str, Any] = {}
        self._tmp_data: dict[str, Any] = {}
        self._anonymous = False
        self._shutdown_cleanup: Any = None

    def _register_shutdown_cleanup(self) -> None:
        """Register process-shutdown cleanup for an anonymous session.

        When a process is killed by ``KeyboardInterrupt`` (Ctrl+C) while a
        prompt is running, the event-loop task that owns this session is left
        suspended, so ``Session.__del__`` may never run.  The soul's aiosqlite
        worker thread then keeps ``context.db`` locked and — being a
        non-daemon thread blocked on its work queue — also blocks
        ``threading._shutdown()`` at interpreter exit, hanging the process.

        ``threading._register_atexit`` callbacks are the only hooks that run
        *before* non-daemon threads are joined, so the registered callback
        stops the aiosqlite worker thread there (releasing the SQLite file
        handles) and then deletes the session directory.
        """
        if not self._anonymous or self._shutdown_cleanup is not None:
            return
        register = getattr(threading, "_register_atexit", None)
        if register is None:
            # Very old Python without threading._register_atexit: fall back to
            # the best-effort __del__ path only.
            return
        cleanup: Any = _build_shutdown_cleanup(self)
        self._shutdown_cleanup = cleanup
        try:
            register(cleanup)
        except Exception:
            self._shutdown_cleanup = None

    def _delete_sync_best_effort(self) -> None:
        """Synchronously close the soul's storage backend and delete the session dir.

        Never raises: this runs from ``__del__`` and the process-shutdown
        callback where an exception would only be printed and ignored anyway.
        Closing the aiosqlite worker thread synchronously is required on
        Windows — the open ``context.db`` handle otherwise makes
        ``shutil.rmtree`` fail silently and the anonymous session directory
        survives a ``KeyboardInterrupt``-killed process.
        """
        cli = getattr(self, "_cli", None)
        if cli is None:
            return
        soul = getattr(cli, "soul", None)
        if soul is not None:
            context = getattr(soul, "_context", None)
            close_sync = getattr(context, "close_sync", None)
            if close_sync is not None:
                try:
                    close_sync()
                except Exception:
                    pass
            # Close the FTS5 history index (apsw connection) so history.db is
            # not locked during the synchronous session-dir removal.
            history_index = getattr(soul, "_history_index", None)
            close_index = getattr(history_index, "close", None)
            if close_index is not None:
                try:
                    close_index()
                except Exception:
                    pass
        try:
            cli.session.delete_sync()
        except Exception:
            pass

    async def clear(self, **custom_arguments) -> None:
        """Clear the session by removing the context file and re-creating the CLI.

        This cancels any ongoing prompt, cleans up tool resources, deletes the
        session's context file (``context.db`` or ``context.jsonl``), and
        re-creates the underlying CLI with the same session ID and original
        creation parameters.

        Raises:
            SessionStateError: When the session is closed.
        """
        self._tmp_data.clear()
        if self._closed:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        await self._cleanup_tools()
        await self._close_chat_provider()
        # Close the KimiSoul's context storage (aiosqlite worker thread)
        # before deleting files. On Windows the SQLite database file stays
        # locked while the connection is open, causing PermissionError.
        soul = getattr(self._cli, "soul", None)
        if soul is not None:
            try:
                await soul.close()
            except Exception:
                pass

        # Close the session's ContextDB before deleting files
        await self._cli.session.close_context_db()

        work_dir = self._cli.session.work_dir
        session_id = self._cli.session.id
        context_file = self._cli.session.context_file
        if context_file.exists():
            context_file.unlink()
        # Clean up SQLite WAL/SHM companion files
        if context_file.suffix == ".db":
            for companion_suffix in (".db-wal", ".db-shm"):
                companion = context_file.with_suffix(companion_suffix)
                if companion.exists():
                    companion.unlink()

        # Clear persisted tool state (e.g. todos) and wire history
        session_dir = self._cli.session.dir
        state_file = session_dir / "state.json"
        if state_file.exists():
            state_file.unlink()
        wire_file = self._cli.session.wire_file.path
        if wire_file.exists():
            wire_file.unlink()

        # Clear custom data from the old session
        self._cli.session.custom_data.clear()

        cli_session = await CliSession.create(work_dir, session_id)
        kwargs = self._create_kwargs.copy()
        kwargs.pop("resumed", None)
        kwargs.update(custom_arguments)
        self._cli = await KimiCLI.create(cli_session, **kwargs)
        self._cancel_event = None
        self._closed = False

    async def _restart(self, **custom_arguments) -> None:
        """Restart the session while preserving context, state and wire history.

        This is used by :meth:`prompt` when a transient error (e.g. a persistent
        5xx or connection failure) forces a session restart. Unlike
        :meth:`clear`, it does not delete the context file, ``state.json`` or
        ``wire.jsonl``; instead it reloads the existing session so the new
        KimiCLI can restore the previous conversation and resume from where the
        failure occurred.
        """
        self._tmp_data.clear()
        if self._closed:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        await self._cleanup_tools()
        await self._close_chat_provider()
        # Close the KimiSoul's context storage (aiosqlite worker thread)
        # before recreating the CLI. On Windows the SQLite database file stays
        # locked while the connection is open, causing PermissionError.
        soul = getattr(self._cli, "soul", None)
        if soul is not None:
            try:
                await soul.close()
            except Exception:
                pass

        # Close the session's ContextDB before reloading the session object.
        await self._cli.session.close_context_db()

        work_dir = self._cli.session.work_dir
        session_id = self._cli.session.id
        old_custom_data = self._cli.session.custom_data.copy()
        old_custom_config = self._cli.session.custom_config.copy()

        # Reload the existing session so context/state/wire history are kept.
        cli_session = await CliSession.find(work_dir, session_id)
        if cli_session is None:
            cli_session = await CliSession.create(work_dir, session_id)

        # Preserve provider_dict/chat_provider overrides if the config file
        # does not already contain them (mirrors the logic in :meth:`rename`).
        custom_config = await _load_config_json(work_dir)
        for key in ("provider_dict", "chat_provider"):
            if key in old_custom_config and key not in custom_config:
                custom_config[key] = old_custom_config[key]
        cli_session.custom_config = custom_config

        kwargs = self._create_kwargs.copy()
        kwargs.pop("resumed", None)
        kwargs.update(custom_arguments)
        self._cli = await KimiCLI.create(cli_session, resumed=True, **kwargs)
        self._cli.session.custom_data.update(old_custom_data)
        self._cancel_event = None
        self._closed = False

    async def rename(self, new_session_id: str) -> None:
        """Rename the session to a new session ID.

        This cancels any ongoing prompt, cleans up tool resources, renames the
        session directory, and resumes the session with the new session ID.
        If the session is closed or cannot be renamed, a new session is created
        with the given session ID.

        Args:
            new_session_id: The new session ID to rename to.
        """
        self._tmp_data.clear()
        work_dir = self._cli.session.work_dir
        old_session_id: str | None = None

        if not self._closed:
            if self._cancel_event is not None:
                self._cancel_event.set()
            await self._cleanup_tools()
            await self._close_chat_provider()
            # Close the KimiSoul's context storage (and any other resources it
            # holds) before renaming the session directory.  On Windows the
            # aiosqlite worker thread keeps the SQLite database file locked
            # while the connection is open, which would otherwise make
            # os.rename() fail with PermissionError.
            soul = getattr(self._cli, "soul", None)
            if soul is not None:
                try:
                    await soul.close()
                except Exception:
                    pass
            # Also close the CLI session's own cached ContextDB, if any.
            await self._cli.session.close_context_db()

            old_session_id = self._cli.session.id
            cli_session = await CliSession.rename(work_dir, old_session_id, new_session_id)
        else:
            cli_session = None

        if cli_session is None:
            cli_session = await CliSession.create(work_dir, new_session_id)

        # Preserve provider_dict from old session's custom_config for sub-agent spawning
        old_custom_config = self._cli.session.custom_config
        custom_config = await _load_config_json(work_dir)
        if "provider_dict" in old_custom_config and "provider_dict" not in custom_config:
            custom_config["provider_dict"] = old_custom_config["provider_dict"]
        if "chat_provider" in old_custom_config and "chat_provider" not in custom_config:
            custom_config["chat_provider"] = old_custom_config["chat_provider"]
        cli_session.custom_config = custom_config

        kwargs = self._create_kwargs.copy()
        kwargs.pop("resumed", None)
        try:
            self._cli = await KimiCLI.create(cli_session, **kwargs)
        except Exception:
            # Rollback: attempt to rename the session directory back
            if old_session_id:
                try:
                    await CliSession.rename(work_dir, new_session_id, old_session_id)
                except Exception:
                    pass
            raise
        self._cancel_event = None
        self._closed = False
        self._anonymous = new_session_id is None
        self._register_shutdown_cleanup()

    async def compact(self, *, custom_instruction: str = "") -> None:
        """Compact the session context.

        This summarizes older conversation history into a condensed form,
        reducing token usage while preserving recent messages and essential
        context.

        Args:
            custom_instruction: Optional user instruction to guide the
                compaction focus.

        Raises:
            SessionStateError: When the session is closed or already running.
            LLMNotSet: When the LLM is not set.
            ChatProviderError: When the chat provider returns an error.
        """
        self._tmp_data.clear()
        if self._closed:
            raise SessionStateError("Session is closed")
        if self._cancel_event is not None:
            raise SessionStateError("Session is already running")

        from kimi_cli.soul import _current_wire
        from kimi_cli.wire import Wire

        wire = Wire()
        token = _current_wire.set(wire)
        try:
            await self._cli.soul.compact_context(custom_instruction=custom_instruction)
        finally:
            _current_wire.reset(token)
            wire.shutdown()

    async def _cleanup_tools(self) -> None:
        """Clean up tool resources without marking the session closed."""
        toolset = getattr(self._cli.soul.agent, "toolset", None)
        cleanup = getattr(toolset, "cleanup", None)
        if cleanup is None:
            return
        result = cleanup()
        if inspect.isawaitable(result):
            await result

    async def _close_chat_provider(self) -> None:
        """Close the underlying LLM chat provider's HTTP client if available.

        This prevents the Anthropic SDK's ``AsyncHttpxClientWrapper.__del__``
        from scheduling an ``aclose()`` task after the event loop has already
        been torn down, which on Windows/Python 3.14 surfaces as a noisy
        ``RuntimeError: Event loop is closed`` task exception.
        """
        try:
            soul = getattr(self._cli, "soul", None)
            if soul is None:
                return
            runtime = getattr(soul, "_runtime", None)
            if runtime is None:
                return
            llm = getattr(runtime, "llm", None)
            if llm is None:
                return
            chat_provider = getattr(llm, "chat_provider", None)
            if chat_provider is None:
                return
            aclose = getattr(chat_provider, "aclose", None)
            if aclose is None:
                return
            await aclose()
        except RuntimeError as exc:
            # Transports bound to a now-closed ProactorEventLoop (Windows
            # Python 3.14) raise RuntimeError('Event loop is closed').  The OS
            # will reclaim the socket, so swallow it silently.
            if "Event loop is closed" not in str(exc):
                raise
        except Exception:
            # Best-effort cleanup; never let provider close failures escape.
            pass

    @staticmethod
    async def create(
        work_dir: KaosPath | None = None,
        *,
        # Basic configuration
        session_id: str | None = None,
        config: Config | Path | None = None,
        model: str | None = None,
        thinking: bool = False,
        # Run mode
        yolo: bool = False,
        plan_mode: bool = False,
        # Extensions
        agent_file: Path | None = None,
        mcp_configs: list[MCPConfig] | list[dict[str, Any]] | None = None,
        skills_dir: KaosPath | None = None,
        skills_dirs: list[KaosPath] | None = None,
        anonymous: bool = False,
        # Loop control
        max_steps_per_turn: int | None = None,
        max_retries_per_step: int | None = None,
        max_ralph_iterations: int | None = None,
        **custom_arguments,  # Add by maxwell
    ) -> Session:
        """
        Create a new Session instance.

        Args:
            work_dir: Working directory (KaosPath). Defaults to current directory.
            session_id: Custom session ID (optional).
            config: Configuration object or path to a config file.
            model: Model name, e.g. "kimi".
            thinking: Whether to enable thinking mode (requires model support).
            yolo: Automatically approve all approval requests.
            agent_file: Agent specification file path.
            mcp_configs: MCP server configurations. Each entry is a ``fastmcp.mcp_config.MCPConfig``
                or an equivalent dict with an ``mcpServers`` mapping. Supports stdio, HTTP, and
                OAuth-enabled servers. Tools are loaded into the agent toolset; resources and
                prompts are discovered for status reporting.
            skills_dir: Single skills directory (KaosPath). Preserved for SDK compatibility.
            skills_dirs: Multiple skills directories (KaosPath list) for newer kimi-cli.
            max_steps_per_turn: Maximum number of steps in one turn.
            max_retries_per_step: Maximum number of retries per step.
            max_ralph_iterations: Extra iterations in Ralph mode (-1 for unlimited).

        Returns:
            Session: A new Session instance.

        Raises:
            FileNotFoundError: When the agent file is not found.
            ConfigError(KimiCLIException, ValueError): When the configuration is invalid.
            AgentSpecError(KimiCLIException, ValueError): When the agent specification is invalid.
            InvalidToolError(KimiCLIException, ValueError): When any tool cannot be loaded.
            MCPConfigError(KimiCLIException, ValueError): When any MCP configuration is invalid.
            MCPRuntimeError(KimiCLIException, RuntimeError): When any MCP server cannot be
                connected.
        """
        if work_dir is None:
            work_dir_path = KaosPath.cwd()
        else:
            _ensure_type("work_dir", work_dir, KaosPath)
            work_dir_path = work_dir
        resolved_skills_dirs = _resolve_skills_dirs(skills_dir, skills_dirs)
        cli_session = await CliSession.create(work_dir_path, session_id)
        custom_config = await _load_config_json(work_dir_path)
        cli_session.custom_config = custom_config
        llm: LLM | None = None
        chat_provider: ChatProvider | None = custom_arguments.pop("chat_provider", None)
        if chat_provider is not None:
            llm = LLM(chat_provider, 0, set())
        cli = await KimiCLI.create(
            cli_session,
            config=config,
            model_name=model,
            thinking=thinking,
            llm=llm,
            yolo=yolo,
            plan_mode=plan_mode,
            agent_file=agent_file,
            mcp_configs=mcp_configs,
            skills_dirs=resolved_skills_dirs,
            max_steps_per_turn=max_steps_per_turn,
            max_retries_per_step=max_retries_per_step,
            max_ralph_iterations=max_ralph_iterations,
            **custom_arguments,
        )
        session = Session(cli)
        session._anonymous = anonymous if anonymous else session_id is None
        session._register_shutdown_cleanup()
        session_dir = cli.session.dir
        state_file = session_dir / "state.json"
        if state_file.exists():
            state_file.unlink()
        session._create_kwargs = {
            "config": config,
            "model_name": model,
            "thinking": thinking,
            "llm": llm,
            "yolo": yolo,
            "plan_mode": plan_mode,
            "agent_file": agent_file,
            "mcp_configs": mcp_configs,
            "skills_dirs": resolved_skills_dirs,
            "max_steps_per_turn": max_steps_per_turn,
            "max_retries_per_step": max_retries_per_step,
            "max_ralph_iterations": max_ralph_iterations,
        }
        session._create_kwargs.update(custom_arguments)
        return session

    @staticmethod
    async def resume(
        work_dir: KaosPath,
        session_id: str | None = None,
        *,
        # Basic configuration
        config: Config | Path | None = None,
        model: str | None = None,
        thinking: bool = False,
        # Run mode
        yolo: bool = False,
        plan_mode: bool = False,
        # Extensions
        agent_file: Path | None = None,
        mcp_configs: list[MCPConfig] | list[dict[str, Any]] | None = None,
        skills_dir: KaosPath | None = None,
        skills_dirs: list[KaosPath] | None = None,
        anonymous: bool = False,
        # Loop control
        max_steps_per_turn: int | None = None,
        max_retries_per_step: int | None = None,
        max_ralph_iterations: int | None = None,
        **custom_arguments,  # Add by maxwell
    ) -> Session | None:
        """
        Resume an existing session.

        Args:
            work_dir: Working directory to resume from (KaosPath).
            session_id: Session ID to resume. If None, resumes the most recent session.
            config: Configuration object or path to a config file.
            model: Model name, e.g. "kimi".
            thinking: Whether to enable thinking mode (requires model support).
            yolo: Automatically approve all approval requests.
            agent_file: Agent specification file path.
            mcp_configs: MCP server configurations. Each entry is a ``fastmcp.mcp_config.MCPConfig``
                or an equivalent dict with an ``mcpServers`` mapping. Supports stdio, HTTP, and
                OAuth-enabled servers. Tools are loaded into the agent toolset; resources and
                prompts are discovered for status reporting.
            skills_dirs: Skills directories (KaosPath or list of KaosPath).
            skills_dir: Single skills directory (KaosPath). Preserved for SDK compatibility.
            skills_dirs: Multiple skills directories (KaosPath list) for newer kimi-cli.
            max_steps_per_turn: Maximum number of steps in one turn.
            max_retries_per_step: Maximum number of retries per step.
            max_ralph_iterations: Extra iterations in Ralph mode (-1 for unlimited).

        Returns:
            Session | None: The resumed session, or None if not found.

        Raises:
            FileNotFoundError: When the agent file is not found.
            ConfigError(KimiCLIException, ValueError): When the configuration is invalid.
            AgentSpecError(KimiCLIException, ValueError): When the agent specification is invalid.
            InvalidToolError(KimiCLIException, ValueError): When any tool cannot be loaded.
            MCPConfigError(KimiCLIException, ValueError): When any MCP configuration is invalid.
            MCPRuntimeError(KimiCLIException, RuntimeError): When any MCP server cannot be
                connected.
        """
        _ensure_type("work_dir", work_dir, KaosPath)
        resolved_skills_dirs = _resolve_skills_dirs(skills_dir, skills_dirs)
        if session_id is None:
            cli_session = await CliSession.continue_(work_dir)
        else:
            cli_session = await CliSession.find(work_dir, session_id)
        if cli_session is None:
            return None
        custom_config = await _load_config_json(work_dir)
        cli_session.custom_config = custom_config
        llm: LLM | None = None
        chat_provider: ChatProvider | None = custom_arguments.pop("chat_provider", None)
        if chat_provider is not None:
            llm = LLM(chat_provider, 0, set())
        cli = await KimiCLI.create(
            cli_session,
            config=config,
            model_name=model,
            thinking=thinking,
            llm=llm,
            yolo=yolo,
            plan_mode=plan_mode,
            agent_file=agent_file,
            mcp_configs=mcp_configs,
            skills_dirs=resolved_skills_dirs,
            max_steps_per_turn=max_steps_per_turn,
            max_retries_per_step=max_retries_per_step,
            max_ralph_iterations=max_ralph_iterations,
            **custom_arguments,
        )
        session = Session(cli)
        session._anonymous = anonymous if anonymous else session_id is None
        session._register_shutdown_cleanup()
        session._create_kwargs = {
            "config": config,
            "model_name": model,
            "thinking": thinking,
            "llm": llm,
            "yolo": yolo,
            "plan_mode": plan_mode,
            "agent_file": agent_file,
            "mcp_configs": mcp_configs,
            "skills_dirs": resolved_skills_dirs,
            "max_steps_per_turn": max_steps_per_turn,
            "max_retries_per_step": max_retries_per_step,
            "max_ralph_iterations": max_ralph_iterations,
        }
        session._create_kwargs.update(custom_arguments)
        return session

    @property
    def id(self) -> str:
        """Session ID."""
        return self._cli.session.id

    @property
    def model_name(self) -> str:
        """Name of the current model."""
        return self._cli.soul.model_name

    @property
    def status(self) -> StatusSnapshot:
        """Current status snapshot (context usage, yolo state, etc.)."""
        return self._cli.soul.status

    def get_custom_data(self) -> dict[str, Any] | None:
        # Return the custom data dictionary from the underlying CLI session. Always reset in 'clear'
        if self._cli is not None and self._cli.session is not None:
            return self._cli.session.custom_data
        return None

    def get_custom_config(self) -> dict[str, Any] | None:
        # Return the custom data dictionary from the underlying CLI session.
        if self._cli is not None and self._cli.session is not None:
            return self._cli.session.custom_config
        return None

    async def export(
        self,
        output_path: str | Path | None = None,
        format: ExportFormat = ExportFormat.Markdown,
    ) -> tuple[Path, int]:
        """Export current session context to a file.

        Args:
            output_path: Optional output file or directory path. If a directory,
                a default filename is generated. If not provided, the file is
                written to the session's work directory.
            format: Export format — ``ExportFormat.Markdown`` (default) for a
                human-readable markdown file, or ``ExportFormat.Jsonl`` for
                a JSON Lines file.  Internal messages (checkpoints, system
                reminders, notifications) are excluded in both formats.

        Returns:
            tuple[Path, int]: The output file path and the number of messages exported.

        Raises:
            SessionStateError: When the session is closed.
            ValueError: When there are no messages to export or writing fails.
        """
        self._tmp_data.clear()
        if self._closed:
            raise SessionStateError("Session is closed")

        import aiofiles
        import pendulum
        from kimi_cli.utils.export import build_export_jsonl, build_export_markdown

        soul = self._cli.soul
        session = self._cli.session

        history = list(soul.context.history)
        if not history:
            raise ValueError("No messages to export.")

        now = pendulum.now()
        short_id = session.id[:8]
        extension = "." + format.value
        default_name = f"kimi-export-{short_id}-{now.strftime('%Y%m%d-%H%M%S')}{extension}"

        # Resolve output path
        default_dir = Path(str(session.work_dir))
        if output_path:
            output = Path(output_path).expanduser()
            if not output.is_absolute():
                output = default_dir / output
            # If path ends with a separator or is an existing directory, treat as directory
            path_str = str(output_path).rstrip("/\\")
            if output_path != path_str or output.is_dir():
                output = output / default_name
        else:
            output = default_dir / default_name

        if format is ExportFormat.Jsonl:
            content = build_export_jsonl(
                session_id=session.id,
                work_dir=str(session.work_dir),
                history=history,
                token_count=soul.context.token_count,
                now=now,
            )
        else:
            content = build_export_markdown(
                session_id=session.id,
                work_dir=str(session.work_dir),
                history=history,
                token_count=soul.context.token_count,
                now=now,
            )

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(output, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            raise ValueError(f"Failed to write export file: {e}")

        return (output, len(history))

    async def prompt(
        self,
        user_input: str | list[ContentPart],
        *,
        merge_wire_messages: bool = False,
        max_restarts: int = 3,
    ) -> AsyncGenerator[WireMessage, None]:
        """
        Send a prompt and get a WireMessage stream.

        Args:
            user_input: User input, can be plain text or a list of content parts.
            merge_wire_messages: Whether to merge consecutive Wire messages.
            max_restarts: Maximum number of automatic session restarts when step
                retries are exhausted (e.g. persistent 5xx, connection failures).
                Set to 0 to disable auto-restart. Default is 3.

        Yields:
            WireMessage: Wire messages, including ApprovalRequest.

        Raises:
            LLMNotSet: When the LLM is not set.
            LLMNotSupported: When the LLM does not have required capabilities.
            ChatProviderError: When the LLM provider returns an error.
            MaxStepsReached: When the maximum number of steps is reached.
            RunCancelled: When the run is cancelled by the cancel event.
            SessionStateError: When the session is closed or already running.

        Note:
            Callers must handle ApprovalRequest manually unless yolo=True.
        """
        self._tmp_data.clear()
        if isinstance(user_input, str):
            user_input = sanitize_for_tokenizer(user_input).strip()
            if not user_input:
                return
        elif isinstance(user_input, list):
            sanitized_parts: list[ContentPart] = []
            for part in user_input:
                if isinstance(part, TextPart):
                    cleaned = sanitize_for_tokenizer(part.text).strip()
                    if cleaned:
                        part.text = cleaned
                        sanitized_parts.append(part)
                elif isinstance(part, ThinkPart):
                    cleaned = sanitize_for_tokenizer(part.think).strip()
                    if cleaned:
                        part.think = cleaned
                        sanitized_parts.append(part)
                else:
                    sanitized_parts.append(part)
            user_input = sanitized_parts
            if not user_input:
                return
        if self._closed:
            raise SessionStateError("Session is closed")
        if self._cancel_event is not None:
            raise SessionStateError("Session is already running")

        # Read max_restarts from LoopControl config if available
        loop_control = getattr(
            getattr(getattr(self._cli, 'soul', None), '_loop_control', None),
            'max_session_restarts',
            None,
        )
        if loop_control is not None:
            max_restarts = loop_control
        if max_restarts < 0:
            max_restarts = 0

        restart_count = 0
        current_user_input = user_input

        while True:
            cancel_event = asyncio.Event()
            self._cancel_event = cancel_event
            try:
                async with _prompt_semaphore:
                    async for msg in self._cli.run(
                        current_user_input,
                        cancel_event,
                        merge_wire_messages=merge_wire_messages,
                    ):
                        yield msg
                break  # success — exit the restart loop
            except SessionRestartRequired as e:
                restart_count += 1
                if restart_count > max_restarts:
                    logger.error(
                        "Session restart limit reached (%d/%d): %s",
                        restart_count - 1,
                        max_restarts,
                        e,
                    )
                    if e.original_error:
                        raise e.original_error from e
                    raise
                logger.warning(
                    "Auto-restarting session (%d/%d): %s",
                    restart_count,
                    max_restarts,
                    e,
                )
                # Notify user via Wire
                yield TextPart(
                    text=(
                        f"\n⚠️ Connection lost ({type(e.original_error).__name__ if e.original_error else 'unknown error'}). "
                        f"Restarting session (attempt {restart_count}/{max_restarts})...\n"
                    )
                )
                # Restart while preserving the existing session context. This
                # cancels any ongoing prompt, cleans up broken tools/providers,
                # and recreates the CLI from the same session so the next turn
                # resumes the prior conversation.
                await self._restart()
                # After _restart(), self._cancel_event is None,
                # self._closed is False, and self._cli resumes the same session.
                # current_user_input is preserved from the outer scope.
            finally:
                if self._cancel_event is cancel_event:
                    self._cancel_event = None

    def cancel(self) -> None:
        """
        Cancel the current prompt operation.

        This sets the cancel event used by the underlying KimiCLI.run call and
        results in RunCancelled being raised from the active prompt coroutine.
        """
        if self._cancel_event is not None:
            self._cancel_event.set()

    async def close(self) -> None:
        """
        Close the Session and release resources.

        This cancels any ongoing prompt and cleans up tool resources.
        For anonymous sessions (created or resumed without a session_id),
        this also deletes the session's context file (context.db or context.jsonl)
        and state.json files.
        """
        if self._closed:
            return
        self._closed = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        await self._cleanup_tools()
        await self._close_chat_provider()
        # Close the underlying KimiSoul so its context storage backend
        # (aiosqlite worker thread) is shut down before process exit.
        soul = getattr(self._cli, "soul", None)
        if soul is not None:
            try:
                await soul.close()
            except Exception:
                pass
        if getattr(self, "_anonymous", False):
            await self._cli.session.delete()

    def __del__(self):
        if getattr(self, "_closed", False):
            return
        if not getattr(self, "_anonymous", False):
            return
        try:
            self._delete_sync_best_effort()
        except Exception:
            # Never raise from __del__; the directory removal is best-effort.
            pass

    async def __aenter__(self) -> Session:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
        await self._cli.session.delete()


def _build_shutdown_cleanup(session: Session) -> Any:
    """Build the process-shutdown callback for an anonymous session.

    The closure intentionally keeps a strong reference to the session so the
    soul's aiosqlite storage stays alive (and can be stopped synchronously)
    until ``threading._shutdown`` runs the callback.  The callback is a no-op
    once the session has been renamed to a named session, and ``delete_sync``
    tolerates a missing directory, so it is safe after a normal ``close()``.
    """

    def _cleanup() -> None:
        try:
            if not getattr(session, "_anonymous", False):
                return
            session._delete_sync_best_effort()
        except Exception:
            # Process-shutdown cleanup must never raise.
            pass

    return _cleanup
