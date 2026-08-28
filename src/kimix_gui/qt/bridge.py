"""Kimix worker thread: asyncio loop, coalesced transcript deltas, request futures."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any

from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    AUTH_CONNECTING,
    AUTH_DISCONNECTED,
    AUTH_LOGIN_REQUIRED,
    AUTH_RETRY_LATER,
    PROBLEM_CALLBACK_UNAVAILABLE,
    PROBLEM_CANCELLED,
    PROBLEM_CREDENTIAL_STORE_BUSY,
    PROBLEM_CREDENTIAL_STORE_UNAVAILABLE,
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_LOGIN_SUPERSEDED,
    PROBLEM_NETWORK,
    PROBLEM_RATE_LIMITED,
    PROBLEM_SERVER,
    CodexAuthError,
    CodexAuthService,
    CodexAuthSnapshot,
    CodexBrowserChallenge,
    CodexProblem,
)
from PySide6.QtCore import QObject, Signal

from kimi_agent_sdk import ApprovalRequest, RunCancelled, ToolError, is_request
from kimix_gui.backend import (
    SdkSession,
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
)
from kimix_gui.history import HistoryLoader, Timeline, create_timeline
from kimix_gui.llm import ChatGPTTarget
from kimix_gui.qt.status_line import format_status_line
from kimix_gui.rendering import StatusValues, WireNormalizer, status_values
from kimix_gui.session_index import (
    SessionDeleter,
    SessionLoader,
    delete_sessions,
    list_session_summaries,
)
from kimix_gui.todos import (
    EMPTY_SNAPSHOT,
    TodoSnapshot,
    load_snapshot,
    snapshot_from_wire_message,
)
from kimix_gui.transcript_data import (
    APP_SOURCE,
    ActivityField,
    AppendText,
    ClearTranscript,
    FieldListBlock,
    HistoryEntry,
    LiteralText,
    LocalizedText,
    NoticeEntry,
    ReplaceEntry,
    StartEntry,
    TextBlock,
    TextEntry,
    TranscriptMutation,
    TranscriptUpdate,
    literal,
    localized,
)

SessionFactory = Callable[[SessionOptions], Awaitable[SdkSession]]
SessionOpenedCallback = Callable[[str], None]

_COALESCE_SECONDS = 0.016


@dataclass(frozen=True, slots=True)
class HistoryPage:
    entries: tuple[HistoryEntry, ...]
    total_turns: int
    target_turn: int | None
    pin_latest: bool
    epoch: int


@dataclass(frozen=True, slots=True)
class StatusLineUpdate:
    values: StatusValues
    epoch: int


@dataclass(frozen=True, slots=True)
class TodoUpdate:
    """A todo snapshot published to the UI, tagged with its session epoch."""

    snapshot: TodoSnapshot
    epoch: int


@dataclass(frozen=True, slots=True)
class ApprovalAsk:
    title: str
    description: str
    token: int
    epoch: int


@dataclass(frozen=True, slots=True)
class QuestionAsk:
    prompt: str
    body: str
    token: int
    epoch: int


class StreamCoalescer:
    """Coalesce typed text fragments or full activity snapshots for about 16ms."""

    def __init__(
        self, emit: Callable[[TranscriptUpdate], None], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._emit = emit
        self._loop = loop
        self._mutation: AppendText | ReplaceEntry | None = None
        self._epoch = 0
        self._handle: asyncio.TimerHandle | None = None

    @property
    def pending(self) -> bool:
        return self._mutation is not None

    def feed(self, mutation: TranscriptMutation, epoch: int) -> None:
        if isinstance(mutation, AppendText):
            pending = self._mutation
            if (
                isinstance(pending, AppendText)
                and pending.key == mutation.key
                and pending.block == mutation.block
                and pending.kind == mutation.kind
            ):
                self._mutation = AppendText(
                    key=pending.key,
                    kind=pending.kind,
                    source=pending.source,
                    block=pending.block,
                    fragment=pending.fragment + mutation.fragment,
                    format=pending.format,
                    tone=pending.tone,
                )
            else:
                self.flush()
                self._mutation = mutation
            self._epoch = epoch
            self._schedule()
            return
        if isinstance(mutation, ReplaceEntry):
            pending = self._mutation
            if isinstance(pending, ReplaceEntry) and pending.entry.key == mutation.entry.key:
                self._mutation = mutation
            else:
                self.flush()
                self._mutation = mutation
            self._epoch = epoch
            self._schedule()
            return
        self.flush()
        self._emit(TranscriptUpdate(epoch=epoch, mutation=mutation))

    def flush(self) -> None:
        self._cancel_timer()
        mutation = self._mutation
        if mutation is None:
            return
        self._emit(TranscriptUpdate(epoch=self._epoch, mutation=mutation))
        self._mutation = None

    def reset(self) -> None:
        self._cancel_timer()
        self._mutation = None

    def _schedule(self) -> None:
        if self._handle is not None:
            return
        self._handle = self._loop.call_later(_COALESCE_SECONDS, self.flush)

    def _cancel_timer(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


class KimixBridge(QObject):
    """Owns a private asyncio loop. SDK objects never leave the worker thread."""

    session_opened = Signal(str, object, int)
    session_failed = Signal(str, int)
    session_closed = Signal(int)
    transcript_updated = Signal(object)
    status_changed = Signal(object)
    history_page = Signal(object)
    history_loading = Signal(bool, int)
    todos_changed = Signal(object)
    generation_started = Signal(int)
    generation_finished = Signal(int)
    approval_asked = Signal(object)
    question_asked = Signal(object)
    notify = Signal(str, str, str)
    sessions_listed = Signal(object)
    sessions_list_failed = Signal(str)
    sessions_deleted = Signal(object)
    input_enabled = Signal(bool, int)
    codex_auth_changed = Signal(object)
    codex_browser_challenge = Signal(object)
    codex_catalog_changed = Signal(object)

    def __init__(
        self,
        *,
        session_factory: SessionFactory = create_sdk_session,
        history_loader: HistoryLoader | None = None,
        session_loader: SessionLoader | None = None,
        session_deleter: SessionDeleter | None = None,
        on_session_opened: SessionOpenedCallback | None = None,
        codex_service: CodexAuthService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._codex_service = codex_service or CodexAuthService()
        self._session_factory = (
            partial(create_sdk_session, codex_service=self._codex_service)
            if session_factory is create_sdk_session
            else session_factory
        )
        self._history_loader = history_loader
        self._session_loader = session_loader
        self._session_deleter = session_deleter
        self._on_session_opened = on_session_opened
        self._codex_operation = 0
        self._codex_account_lock = asyncio.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._busy = False
        self._epoch = 0
        self._session: SdkSession | None = None
        self._session_id: str | None = None
        self._options: SessionOptions | None = None
        self._timeline: Timeline | None = None
        self._history_total = 0
        self._history_loading = False
        # Live keys must never collide with the independently normalized history
        # snapshot installed in the same model.
        self._normalizer = WireNormalizer(scope="live")
        self._last_wire_status: str | None = None
        self._todos = EMPTY_SNAPSHOT
        self._coalescer: StreamCoalescer | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._tokens = count(1)
        self._entry_keys = count(1)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="kimix-bridge", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        operation_id = self._next_codex_operation()
        self.submit(self._initialize_codex(operation_id))

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        self.submit(self._shutdown())
        if self._thread is not None:
            self._thread.join(timeout=8)
        self._thread = None
        self._loop = None

    def is_idle(self) -> bool:
        with self._lock:
            pending = self._coalescer.pending if self._coalescer is not None else False
            in_flight = self._submitted != self._completed
            return not in_flight and not self._busy and not pending

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def submit(self, coro: Awaitable[object]) -> None:
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            self._submitted += 1
        asyncio.run_coroutine_threadsafe(self._tracked(coro), loop)

    def open_session(
        self,
        options: SessionOptions,
        *,
        on_session_opened: SessionOpenedCallback | None = None,
    ) -> None:
        if on_session_opened is not None:
            self._on_session_opened = on_session_opened
        self.submit(self._open_session(options))

    def close_session(self) -> None:
        self.submit(self._close_session())

    def run_prompt(self, text: str) -> None:
        self.submit(self._run_prompt(text))

    def run_command(self, command: str) -> None:
        self.submit(self._run_command(command))

    def cancel_prompt(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._cancel_prompt)

    def load_sessions(self, work_dir: Path) -> None:
        self.submit(self._load_sessions(work_dir))

    def delete_sessions(self, work_dir: Path, session_ids: list[str]) -> None:
        self.submit(self._delete_sessions(work_dir, session_ids))

    def jump_to_turn(self, turn: int) -> None:
        self.submit(self._jump_to_turn(turn))

    def load_older(self, current_turn: int) -> None:
        self.submit(self._load_older(current_turn))

    def load_newer(self, current_turn: int) -> None:
        self.submit(self._load_newer(current_turn))

    def prefetch_older(self) -> None:
        self.submit(self._prefetch_older())

    def prefetch_newer(self) -> None:
        self.submit(self._prefetch_newer())

    def jump_to_latest(self) -> None:
        self.submit(self._jump_to_latest())

    def refresh_todos(self) -> None:
        """Re-read the session's persisted todo state and republish it."""

        self.submit(self._refresh_todos(self.epoch))

    def connect_chatgpt(self) -> int:
        operation_id = self._next_codex_operation()
        self.codex_auth_changed.emit(
            CodexAuthSnapshot(operation_id=operation_id, state=AUTH_CONNECTING)
        )
        self.submit(self._connect_chatgpt(operation_id))
        return operation_id

    def cancel_chatgpt_login(self, operation_id: int) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._codex_service.cancel_login, operation_id)

    def refresh_codex_models(self) -> int:
        operation_id = self._next_codex_operation()
        self.submit(self._refresh_codex_models(operation_id))
        return operation_id

    def disconnect_chatgpt(self, *, close_active_session: bool = False) -> int:
        operation_id = self._next_codex_operation()
        self.submit(self._disconnect_chatgpt(operation_id, close_active_session))
        return operation_id

    @property
    def uses_chatgpt(self) -> bool:
        with self._lock:
            options = self._options
            session = self._session
        return (
            session is not None
            and options is not None
            and options.llm_selection is not None
            and isinstance(options.llm_selection.target, ChatGPTTarget)
        )

    def resolve_request(self, token: int, epoch: int, value: object) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._resolve_request, token, epoch, value)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._coalescer = StreamCoalescer(self._emit_update, loop)
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        with suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    async def _tracked(self, coro: Awaitable[object]) -> None:
        try:
            await coro
        finally:
            with self._lock:
                self._completed += 1

    async def _shutdown(self) -> None:
        await self._release_session()
        await self._codex_service.aclose()
        loop = self._loop
        if loop is not None:
            loop.stop()

    def _emit_update(self, update: TranscriptUpdate) -> None:
        self.transcript_updated.emit(update)

    async def _initialize_codex(self, operation_id: int) -> None:
        async with self._codex_account_lock:
            await self._initialize_codex_locked(operation_id)

    async def _initialize_codex_locked(self, operation_id: int) -> None:
        if not self._codex_operation_current(operation_id):
            return
        try:
            snapshot = await self._codex_service.snapshot(operation_id)
            if snapshot.state == AUTH_CONNECTED:
                catalog = await self._codex_service.refresh_models(operation_id)
                snapshot = await self._codex_service.snapshot(operation_id)
            else:
                catalog = await self._codex_service.catalog(operation_id)
        except CodexAuthError as exc:
            if self._codex_operation_current(operation_id):
                self.codex_auth_changed.emit(
                    CodexAuthSnapshot(
                        operation_id=operation_id,
                        state=AUTH_RETRY_LATER,
                        problem=exc.problem,
                    )
                )
            return
        if not self._codex_operation_current(operation_id):
            return
        self.codex_auth_changed.emit(snapshot)
        self.codex_catalog_changed.emit(catalog)

    async def _connect_chatgpt(self, operation_id: int) -> None:
        async with self._codex_account_lock:
            await self._connect_chatgpt_locked(operation_id)

    async def _connect_chatgpt_locked(self, operation_id: int) -> None:
        if not self._codex_operation_current(operation_id):
            return

        async def publish_challenge(challenge: CodexBrowserChallenge) -> None:
            if self._codex_operation_current(operation_id):
                self.codex_browser_challenge.emit(challenge)

        try:
            snapshot, catalog = await self._codex_service.login(
                operation_id,
                publish_challenge,
            )
        except CodexAuthError as exc:
            if not self._codex_operation_current(operation_id):
                return
            try:
                stored = await self._codex_service.snapshot(operation_id)
            except CodexAuthError:
                stored = CodexAuthSnapshot(
                    operation_id=operation_id,
                    state=AUTH_RETRY_LATER,
                )
            state = stored.state
            if exc.problem.code == PROBLEM_CANCELLED and state != AUTH_CONNECTED:
                state = AUTH_DISCONNECTED
            elif exc.problem.code in {PROBLEM_LOGIN_REQUIRED}:
                state = AUTH_LOGIN_REQUIRED
            elif exc.problem.code in {
                PROBLEM_CALLBACK_UNAVAILABLE,
                PROBLEM_CREDENTIAL_STORE_BUSY,
                PROBLEM_CREDENTIAL_STORE_UNAVAILABLE,
                PROBLEM_LOGIN_SUPERSEDED,
                PROBLEM_RATE_LIMITED,
                PROBLEM_NETWORK,
                PROBLEM_SERVER,
            }:
                state = AUTH_RETRY_LATER
            self.codex_auth_changed.emit(
                CodexAuthSnapshot(
                    operation_id=operation_id,
                    state=state,
                    model_count=stored.model_count,
                    stale=stored.stale,
                    expires_at=stored.expires_at,
                    problem=exc.problem,
                )
            )
            return
        if not self._codex_operation_current(operation_id):
            return
        self.codex_catalog_changed.emit(catalog)
        self.codex_auth_changed.emit(snapshot)

    async def _refresh_codex_models(self, operation_id: int) -> None:
        async with self._codex_account_lock:
            await self._refresh_codex_models_locked(operation_id)

    async def _refresh_codex_models_locked(self, operation_id: int) -> None:
        if not self._codex_operation_current(operation_id):
            return
        try:
            catalog = await self._codex_service.refresh_models(operation_id)
            snapshot = await self._codex_service.snapshot(operation_id)
        except CodexAuthError as exc:
            if self._codex_operation_current(operation_id):
                self.codex_auth_changed.emit(
                    CodexAuthSnapshot(
                        operation_id=operation_id,
                        state=AUTH_RETRY_LATER,
                        problem=exc.problem,
                    )
                )
            return
        if not self._codex_operation_current(operation_id):
            return
        self.codex_catalog_changed.emit(catalog)
        self.codex_auth_changed.emit(snapshot)

    async def _disconnect_chatgpt(
        self,
        operation_id: int,
        close_active_session: bool,
    ) -> None:
        async with self._codex_account_lock:
            await self._disconnect_chatgpt_locked(operation_id, close_active_session)

    async def _disconnect_chatgpt_locked(
        self,
        operation_id: int,
        close_active_session: bool,
    ) -> None:
        if not self._codex_operation_current(operation_id):
            return
        options = self._options
        active = (
            self._session is not None
            and options is not None
            and options.llm_selection is not None
            and isinstance(options.llm_selection.target, ChatGPTTarget)
        )
        if active and not close_active_session:
            self.codex_auth_changed.emit(
                CodexAuthSnapshot(
                    operation_id=operation_id,
                    state=AUTH_CONNECTED,
                    problem=CodexProblem(PROBLEM_LOGIN_REQUIRED),
                )
            )
            return
        if active:
            epoch = await self._release_session()
            self.session_closed.emit(epoch)
        try:
            snapshot = await self._codex_service.disconnect(operation_id)
        except CodexAuthError as exc:
            snapshot = CodexAuthSnapshot(
                operation_id=operation_id,
                state=AUTH_RETRY_LATER,
                problem=exc.problem,
            )
        if self._codex_operation_current(operation_id):
            self.codex_auth_changed.emit(snapshot)

    def _next_codex_operation(self) -> int:
        with self._lock:
            self._codex_operation += 1
            return self._codex_operation

    def _codex_operation_current(self, operation_id: int) -> bool:
        with self._lock:
            return operation_id == self._codex_operation

    def _emit_mutation(self, mutation: TranscriptMutation, epoch: int) -> None:
        self._emit_update(TranscriptUpdate(epoch=epoch, mutation=mutation))

    def _notice_mutation(
        self,
        kind: str,
        text: LiteralText | LocalizedText,
        *,
        fields: tuple[ActivityField, ...] = (),
    ) -> StartEntry:
        blocks = [TextBlock(text)]
        if fields:
            blocks.append(FieldListBlock(fields))
        return StartEntry(
            NoticeEntry(
                key=f"app:{kind}:{next(self._entry_keys)}",
                kind=kind,  # type: ignore[arg-type]
                blocks=tuple(blocks),
                source=APP_SOURCE,
            )
        )

    def _emit_notice(
        self,
        kind: str,
        text: LiteralText | LocalizedText,
        epoch: int,
        *,
        fields: tuple[ActivityField, ...] = (),
    ) -> None:
        self._emit_mutation(self._notice_mutation(kind, text, fields=fields), epoch)

    def _cancel_prompt(self) -> None:
        session = self._session
        if session is not None and self._busy:
            with suppress(Exception):
                session.cancel()

    def _resolve_request(self, token: int, epoch: int, value: object) -> None:
        future = self._pending.pop(token, None)
        if future is None or future.done():
            return
        with self._lock:
            current = self._epoch
        if epoch != current:
            future.cancel()
            return
        future.set_result(value)

    async def _open_session(self, options: SessionOptions) -> None:
        epoch = self._bump_epoch()
        self._options = options
        self._normalizer = WireNormalizer(scope="live")
        self._last_wire_status = None
        self._timeline = None
        self._history_total = 0
        self._publish_todos(EMPTY_SNAPSHOT, epoch)
        try:
            session = await self._session_factory(options)
        except Exception as exc:  # noqa: BLE001
            if epoch != self.epoch:
                return
            self.session_failed.emit(
                self.tr("Failed to open session: {reason}").format(reason=exc), epoch
            )
            return
        if epoch != self.epoch:
            await self._close_sdk(session)
            return
        with self._lock:
            self._session = session
            self._session_id = session.id
        if self._on_session_opened is not None:
            try:
                self._on_session_opened(session.id)
            except Exception as exc:  # noqa: BLE001
                self._emit_notice(
                    "error",
                    localized(
                        "Failed to save session configuration metadata: {reason}", reason=str(exc)
                    ),
                    epoch,
                )
        values = status_values(session.status)
        self.session_opened.emit(session.id, values, epoch)
        self._emit_notice("system", localized("Session: {id}", id=session.id), epoch)
        await self._refresh_todos(epoch)
        try:
            await self._replay_history(session, epoch)
        finally:
            if epoch == self.epoch and self._session is session:
                self.input_enabled.emit(True, epoch)

    async def _replay_history(self, session: SdkSession, epoch: int) -> None:
        if self._history_loader is None:
            self._history_loading = True
            self.history_loading.emit(True, epoch)
        try:
            options = self._options
            if options is None:
                # A session cannot start without options, so this is a programming error;
                # the handler below turns it into a visible notification either way.
                raise RuntimeError("history replay ran without session options")
            if self._history_loader is None:
                timeline = await create_timeline(options.work_dir, session.id)
                if timeline is None:
                    self._history_loading = False
                    self.history_loading.emit(False, epoch)
                    return
                self._timeline = timeline
                await self._publish_history(pin_latest=True, epoch=epoch)
                return
            history = await self._history_loader(options.work_dir, session.id)
        except Exception as exc:  # noqa: BLE001
            self._history_loading = False
            self.history_loading.emit(False, epoch)
            if epoch != self.epoch or self._session is not session:
                return
            self._emit_notice(
                "error",
                localized("Failed to load history: {reason}", reason=str(exc)),
                epoch,
            )
            return
        if epoch != self.epoch or self._session is not session:
            return
        if self._history_loader is not None:
            if history.omitted_turns:
                shown_turns = len({entry.turn for entry in history.entries})
                # Singular and plural as two whole strings, never ``%n``: a clone with
                # no compiled catalog renders the msgid, and "(s)" is not English.
                msgid = (
                    "Showing the last turn ({omitted} earlier omitted)"
                    if shown_turns == 1
                    else "Showing the last {shown} turns ({omitted} earlier omitted)"
                )
                self._emit_notice(
                    "system",
                    localized(msgid, shown=shown_turns, omitted=history.omitted_turns),
                    epoch,
                )
            self.history_page.emit(
                HistoryPage(
                    entries=history.entries,
                    total_turns=0,
                    target_turn=None,
                    pin_latest=True,
                    epoch=epoch,
                )
            )
            self._history_loading = False
            self.history_loading.emit(False, epoch)

    async def _publish_history(
        self,
        *,
        target: int | None = None,
        pin_latest: bool = False,
        epoch: int,
    ) -> None:
        timeline = self._timeline
        if timeline is None:
            self._history_loading = False
            self.history_loading.emit(False, epoch)
            return
        entries = timeline.history_entries()
        self._history_total = timeline.total_turns
        self._history_loading = False
        self.history_page.emit(
            HistoryPage(
                entries=entries,
                total_turns=timeline.total_turns,
                target_turn=target,
                pin_latest=pin_latest,
                epoch=epoch,
            )
        )
        self.history_loading.emit(False, epoch)

    async def _seek(
        self, target: int, *, pin_latest: bool, epoch: int, session: SdkSession
    ) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        await timeline.slide_to(target)
        if epoch != self.epoch or self._session is not session:
            return
        await self._publish_history(target=target, pin_latest=pin_latest, epoch=epoch)

    async def _jump_to_turn(self, turn: int) -> None:
        session = self._session
        epoch = self.epoch
        timeline = self._timeline
        total = self._history_total
        if timeline is None or session is None or self._history_loading or not 1 <= turn <= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(turn - 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not jump to turn {turn}: {reason}").format(
                        turn=turn, reason=exc
                    ),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _load_older(self, current_turn: int) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, current_turn - 1)
        first = timeline.first_materialized_turn()
        if current <= 0:
            if first <= 0:
                return
            target = first - 1
        else:
            target = current - 1
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(target, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not load older history: {reason}").format(reason=exc),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _load_newer(self, current_turn: int) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        total = self._history_total
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, current_turn - 1)
        if current + 1 >= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(current + 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not load newer history: {reason}").format(reason=exc),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _prefetch_older(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            return
        first = timeline.first_materialized_turn()
        if first <= 0:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(first - 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not load older history: {reason}").format(reason=exc),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _prefetch_newer(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        total = self._history_total
        if timeline is None or session is None or self._history_loading:
            return
        last = timeline.last_materialized_turn()
        if last + 1 >= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(last + 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not load newer history: {reason}").format(reason=exc),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _jump_to_latest(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            self.history_page.emit(
                HistoryPage(
                    entries=(),
                    total_turns=self._history_total,
                    target_turn=None,
                    pin_latest=True,
                    epoch=epoch,
                )
            )
            return
        total = self._history_total
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(max(0, total - 1), pin_latest=True, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(
                    self.tr("Could not jump to latest history: {reason}").format(reason=exc),
                    "error",
                    "",
                )
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _run_prompt(self, text: str) -> None:
        session = self._session
        epoch = self.epoch
        if session is None:
            return
        self._emit_mutation(
            StartEntry(
                TextEntry(
                    key=f"live:user:{next(self._entry_keys)}",
                    kind="user",
                    blocks=(TextBlock(literal(text)),),
                )
            ),
            epoch,
        )
        with self._lock:
            self._busy = True
        self.generation_started.emit(epoch)
        self.input_enabled.emit(False, epoch)
        coalescer = self._coalescer
        try:
            async for message in session.prompt(text, merge_wire_messages=False):
                if self._session is not session or epoch != self.epoch:
                    return
                if isinstance(message, ApprovalRequest):
                    await self._render_message(message, epoch)
                    await self._handle_approval(message, epoch)
                    continue
                if is_request(message):
                    await self._render_message(message, epoch)
                    await self._handle_other_request(message, epoch)
                    continue
                await self._render_message(message, epoch)
        except RunCancelled:
            if self._session is session and epoch == self.epoch:
                if coalescer:
                    coalescer.flush()
                self._emit_notice("system", localized("Generation cancelled"), epoch)
        except Exception as exc:  # noqa: BLE001
            if self._session is session and epoch == self.epoch:
                if coalescer:
                    coalescer.flush()
                self._emit_notice("error", literal(f"{type(exc).__name__}: {exc}"), epoch)
        finally:
            if coalescer:
                coalescer.flush()
            with self._lock:
                self._busy = False
            self.generation_finished.emit(epoch)
            if self._session is session and epoch == self.epoch:
                await self._refresh_todos(epoch)
                self.input_enabled.emit(True, epoch)

    def _publish_todos(self, snapshot: TodoSnapshot, epoch: int) -> None:
        if epoch != self.epoch:
            return
        if snapshot == self._todos:
            return
        self._todos = snapshot
        self.todos_changed.emit(TodoUpdate(snapshot=snapshot, epoch=epoch))

    async def _refresh_todos(self, epoch: int) -> None:
        """Publish the authoritative on-disk todo state for the open session."""

        options = self._options
        session_id = self.session_id
        if options is None or session_id is None:
            self._publish_todos(EMPTY_SNAPSHOT, epoch)
            return
        try:
            snapshot = await load_snapshot(options.work_dir, session_id)
        except Exception:  # noqa: BLE001 - todo state is decoration, never fatal
            return
        if snapshot is None:
            return
        self._publish_todos(snapshot, epoch)

    def _observe_todos(self, message: object, epoch: int) -> None:
        """Adopt the todo tree a root-agent tool result just wrote."""

        snapshot = snapshot_from_wire_message(message)
        if snapshot is None:
            return
        self._publish_todos(snapshot.with_archived(self._todos.archived), epoch)

    async def _render_message(self, message: object, epoch: int) -> None:
        self._observe_todos(message, epoch)
        # The prompt is inserted before awaiting the SDK so input feels immediate.
        # Its matching TurnBegin remains the persisted history boundary, not a second row.
        if type(message).__name__ == "TurnBegin":
            return
        normalized = self._normalizer.normalize(message)
        coalescer = self._coalescer
        if coalescer is None:
            return
        if normalized.status is not None:
            self._last_wire_status = format_status_line(normalized.status)
            self.status_changed.emit(StatusLineUpdate(normalized.status, epoch))
        for mutation in normalized.mutations:
            coalescer.feed(mutation, epoch)

    async def _handle_approval(self, request: ApprovalRequest, epoch: int) -> None:
        if self._coalescer:
            self._coalescer.flush()
        decision = await self._ask(
            ApprovalAsk(
                title=self.tr("Approve {action}?").format(action=request.action),
                description=request.description,
                token=next(self._tokens),
                epoch=epoch,
            )
        )
        if decision is None or epoch != self.epoch:
            return
        request.resolve(decision)  # type: ignore[arg-type]
        # ``decision`` is an SDK protocol value (``approve`` / ``approve_for_session``
        # / ``reject``) and stays in its wire spelling, like the metadata line below.
        self._emit_notice(
            "system",
            localized("Approval decision: {decision}", decision=str(decision)),
            epoch,
            fields=(ActivityField("Request ID", literal(request.id), role="secondary"),),
        )

    async def _handle_other_request(self, request: object, epoch: int) -> None:
        request_name = type(request).__name__
        if request_name == "QuestionRequest":
            answers: dict[str, str] = {}
            for question in getattr(request, "questions", []):
                prompt = str(getattr(question, "question", "") or self.tr("Question"))
                options = getattr(question, "options", [])
                option_lines = [
                    f"- {getattr(option, 'label', option)}"
                    + (
                        f": {getattr(option, 'description', '')}"
                        if getattr(option, "description", "")
                        else ""
                    )
                    for option in options
                ]
                answer = await self._ask(
                    QuestionAsk(
                        prompt=prompt,
                        body="\n".join(option_lines),
                        token=next(self._tokens),
                        epoch=epoch,
                    )
                )
                if answer is None:
                    set_exception = getattr(request, "set_exception", None)
                    if callable(set_exception):
                        set_exception(RuntimeError("Question cancelled by user"))
                    request_id = getattr(request, "id", "")
                    self._emit_notice(
                        "error",
                        localized("Question cancelled"),
                        epoch,
                        fields=(
                            ActivityField("Request ID", literal(request_id), role="secondary"),
                        ),
                    )
                    return
                answers[prompt] = str(answer)
            self._resolve_sdk_request(request, answers)
            self._emit_notice(
                "system",
                localized("Question response"),
                epoch,
                fields=tuple(
                    ActivityField(question, literal(answer), role="primary")
                    for question, answer in answers.items()
                ),
            )
            return

        if request_name == "HookRequest":
            decision = await self._ask(
                ApprovalAsk(
                    title=self.tr("Allow hook {event}?").format(
                        event=getattr(request, "event", "")
                    ),
                    description=str(getattr(request, "target", "")),
                    token=next(self._tokens),
                    epoch=epoch,
                )
            )
            if decision is None or epoch != self.epoch:
                return
            action = "allow" if decision != "reject" else "block"
            self._resolve_sdk_request(request, action, "")
            request_id = getattr(request, "id", "")
            # ``allow`` / ``block`` are the values handed back to the SDK, so the two
            # outcomes are spelled as whole sentences rather than a translated word
            # pasted into a stem.
            self._emit_notice(
                "system" if action == "allow" else "error",
                localized("Hook decision: allow" if action == "allow" else "Hook decision: block"),
                epoch,
                fields=(ActivityField("Request ID", literal(request_id), role="secondary"),),
            )
            return

        if request_name == "ToolCallRequest":
            # The ``ToolError`` text travels back to the SDK / the model, so it stays
            # English; only the transcript line the user reads is translated.
            error = ToolError(
                message="External client-side tools are not supported by this GUI prototype",
                brief="Unsupported external tool",
            )
            self._resolve_sdk_request(request, error)
            self._emit_notice(
                "error",
                localized("External client-side tools are not supported yet"),
                epoch,
            )
            return

        self._emit_notice(
            "error",
            localized("Unsupported SDK request: {request}", request=request_name),
            epoch,
        )
        if self._session is not None:
            self._session.cancel()

    async def _ask(self, ask: ApprovalAsk | QuestionAsk) -> object | None:
        # ``_ask`` is awaited on the worker loop, so the running loop *is* ``self._loop``
        # and there is no "not started yet" case left to assert about.
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[ask.token] = future
        if isinstance(ask, ApprovalAsk):
            self.approval_asked.emit(ask)
        else:
            self.question_asked.emit(ask)
        try:
            return await future
        except asyncio.CancelledError:
            return None

    @staticmethod
    def _resolve_sdk_request(request: object, *args: object) -> None:
        resolver = getattr(request, "resolve", None)
        if not callable(resolver):
            raise TypeError(f"SDK request {type(request).__name__} has no resolver")
        resolver(*args)

    async def _run_command(self, command: str) -> None:
        session = self._session
        epoch = self.epoch
        if session is None:
            return
        name, _, argument = command.partition(" ")
        if name == "/quit":
            await self._close_session()
            return
        if name == "/help":
            # The command names are input syntax and must not be translated; only the
            # parenthetical gloss is copy.
            self._emit_notice(
                "system",
                localized("/help  /status  /clear  /compact [instruction]  /quit (back to home)"),
                epoch,
            )
            return
        if name == "/status":
            self._emit_notice(
                "system",
                literal(
                    self._last_wire_status or format_status_line(status_values(session.status))
                ),
                epoch,
            )
            return
        with self._lock:
            self._busy = True
        self.input_enabled.emit(False, epoch)
        try:
            if name == "/clear":
                await session.clear()
                self._timeline = None
                self._history_total = 0
                self._emit_mutation(ClearTranscript(), epoch)
                self._emit_notice("system", localized("Session context cleared"), epoch)
            elif name == "/compact":
                # Deliberately untranslated: the SDK emits its own CompactionBegin /
                # CompactionEnd records around these two lines, and those carry the
                # untranslated diagnostic field block (``Trigger``, ``Shadowed
                # tokens``, ``Compaction ID``). Translating only the acknowledgement
                # would put Chinese and English headlines on adjacent rows of the same
                # block. See the note on ``rendering.py`` in ``docs/gui/i18n.md``.
                self._emit_notice("system", localized("Compacting context…"), epoch)
                await session.compact(custom_instruction=argument.strip())
                self._emit_notice("system", localized("Context compacted"), epoch)
            else:
                self._emit_notice(
                    "error", localized("Unknown command: {command}", command=name), epoch
                )
        except Exception as exc:  # noqa: BLE001
            self._emit_notice("error", literal(f"{type(exc).__name__}: {exc}"), epoch)
        finally:
            with self._lock:
                self._busy = False
            if self._session is session and epoch == self.epoch:
                await self._refresh_todos(epoch)
                self.input_enabled.emit(True, epoch)

    async def _close_session(self) -> None:
        epoch = await self._release_session()
        self.session_closed.emit(epoch)

    async def _release_session(self) -> int:
        epoch = self._bump_epoch()
        session = self._session
        with self._lock:
            self._session = None
            self._session_id = None
            self._busy = False
        self._timeline = None
        self._history_total = 0
        self._todos = EMPTY_SNAPSHOT
        self.todos_changed.emit(TodoUpdate(snapshot=EMPTY_SNAPSHOT, epoch=epoch))
        if self._coalescer:
            self._coalescer.reset()
        self._normalizer.reset()
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()
        await self._close_sdk(session)
        return epoch

    def _bump_epoch(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    @staticmethod
    async def _close_sdk(session: SdkSession | None) -> None:
        if session is None:
            return
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            with suppress(Exception):
                cancel()
        with suppress(Exception):
            await close_sdk_session(session)
        close = getattr(session, "close", None)
        if callable(close) and not getattr(session, "closed", False):
            with suppress(Exception):
                await close()

    async def _load_sessions(self, work_dir: Path) -> None:
        try:
            loader = self._session_loader or list_session_summaries
            summaries = sorted(
                await loader(work_dir),
                key=lambda summary: summary.updated_at,
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.sessions_list_failed.emit(str(exc))
            return
        self.sessions_listed.emit(summaries)

    async def _delete_sessions(self, work_dir: Path, session_ids: list[str]) -> None:
        try:
            await (self._session_deleter or delete_sessions)(work_dir, session_ids)
        except Exception as exc:  # noqa: BLE001
            self.notify.emit(
                self.tr("Failed to delete sessions: {reason}").format(reason=exc), "error", ""
            )
            return
        self.sessions_deleted.emit(session_ids)
