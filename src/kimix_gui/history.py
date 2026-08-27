"""Session timeline: wire.jsonl to a bounded window of structured entries."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from kaos.path import KaosPath

from kimix_gui.rendering import WireNormalizer
from kimix_gui.transcript_data import (
    ClearTranscript,
    HistoryEntry,
    NoticeEntry,
    RawBlock,
    StartEntry,
    TranscriptEntry,
    TranscriptReducer,
    entry_body_text,
    entry_cost,
    entry_kind,
    is_dialogue_entry,
    literal,
)

MAX_HISTORY_TURNS = 4
MAX_HISTORY_BLOCKS = 32
HISTORY_PAGE_TURNS = MAX_HISTORY_TURNS

HYDRATED_BODY_BUDGET = 6 * 1024 * 1024
HYDRATED_RECORD_BUDGET = 512
WINDOW_RADIUS = 3
UNMATERIALIZED_TURN_LINES = 6
_TURN_BEGIN_RECORD = re.compile(
    rb'"message"\s*:\s*\{\s*"type"\s*:\s*"TurnBegin"\s*,\s*"payload"\s*:'
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """A bounded sequence of structured history entries ready for the GUI."""

    entries: tuple[HistoryEntry, ...]
    omitted_turns: int = 0
    total_turns: int = 0
    start_turn: int = 0
    end_turn: int = 0
    has_older: bool = False


HistoryLoader = Callable[[Path, str], Awaitable[SessionHistory]]


@dataclass(frozen=True, slots=True)
class WireHistoryIndex:
    """Byte locators and conservative costs for one wire-log snapshot."""

    path: Path
    turn_offsets: tuple[int, ...]
    file_size: int
    turn_record_counts: tuple[int, ...] = ()

    @property
    def total_turns(self) -> int:
        return len(self.turn_offsets)


TurnIndex = WireHistoryIndex


def estimate_entry_lines(entry: TranscriptEntry, *, width: int = 80) -> int:
    """Conservative virtual-scroll estimate without caching a rendered body."""

    kind = entry_kind(entry)
    if kind not in {"user", "assistant", "thinking"}:
        return 1
    text = entry_body_text(entry)
    wrap_width = max(8, width - 2)
    body_lines = sum(
        max(1, (max(1, len(line)) + wrap_width - 1) // wrap_width)
        for line in (text.split("\n") if text else [""])
    )
    return 1 + body_lines + 1


@dataclass(slots=True)
class Timeline:
    """Sliding decoded window; loaded turns contain only consumable AST entries."""

    index: WireHistoryIndex | None = None
    hydrated_budget: int = HYDRATED_BODY_BUDGET
    hydrated_record_budget: int = HYDRATED_RECORD_BUDGET
    wrap_width: int = 80
    _turns: list[tuple[HistoryEntry, ...] | None] = field(default_factory=list)
    _source_turns: tuple[tuple[HistoryEntry, ...], ...] | None = None

    @classmethod
    def from_turn_entries(cls, turns: Sequence[Sequence[TranscriptEntry]]) -> Timeline:
        """Build an in-memory timeline while reusing the supplied AST objects."""

        source = tuple(
            tuple(
                HistoryEntry(entry=entry, turn=turn, ordinal=ordinal)
                for ordinal, entry in enumerate(entries)
            )
            for turn, entries in enumerate(turns)
        )
        return cls(_turns=list(source), _source_turns=source)

    @property
    def total_turns(self) -> int:
        if self.index is not None:
            return self.index.total_turns
        return len(self._turns)

    @property
    def materialized_turn_count(self) -> int:
        return sum(turn is not None for turn in self._turns)

    @property
    def materialized_record_count(self) -> int:
        return sum(len(turn) for turn in self._turns if turn is not None)

    def has_older_unmaterialized(self, first_materialized: int) -> bool:
        return first_materialized > 0

    def turn_range(self, turn: int) -> tuple[int, int]:
        index = self.index
        if index is None or not index.turn_offsets:
            return 0, 0
        turn = max(0, min(turn, index.total_turns - 1))
        start = index.turn_offsets[turn]
        end = index.file_size if turn + 1 >= index.total_turns else index.turn_offsets[turn + 1]
        return start, end

    def entries_for_turn(self, turn: int) -> tuple[HistoryEntry, ...] | None:
        if not 0 <= turn < len(self._turns):
            return None
        return self._turns[turn]

    def history_entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(item for turn in self._turns if turn is not None for item in turn)

    def first_materialized_turn(self) -> int:
        return next((index for index, turn in enumerate(self._turns) if turn is not None), 0)

    def last_materialized_turn(self) -> int:
        return next(
            (
                index
                for index in range(len(self._turns) - 1, -1, -1)
                if self._turns[index] is not None
            ),
            -1,
        )

    def virtual_lines(self) -> int:
        total = 0
        for turn_index in range(self.total_turns):
            turn = self._turns[turn_index] if turn_index < len(self._turns) else None
            total += (
                UNMATERIALIZED_TURN_LINES
                if turn is None
                else sum(estimate_entry_lines(item.entry, width=self.wrap_width) for item in turn)
            )
        return total

    def first_line_of_turn(self, turn: int) -> int:
        turn = max(0, min(turn, max(0, self.total_turns - 1)))
        total = 0
        for turn_index in range(turn):
            entries = self._turns[turn_index] if turn_index < len(self._turns) else None
            total += (
                UNMATERIALIZED_TURN_LINES
                if entries is None
                else sum(
                    estimate_entry_lines(item.entry, width=self.wrap_width) for item in entries
                )
            )
        return total

    def turn_at_line(self, line: int) -> int:
        if self.total_turns <= 0:
            return 0
        remaining = max(0, line)
        for turn_index in range(self.total_turns):
            entries = self._turns[turn_index] if turn_index < len(self._turns) else None
            height = (
                UNMATERIALIZED_TURN_LINES
                if entries is None
                else sum(
                    estimate_entry_lines(item.entry, width=self.wrap_width) for item in entries
                )
            )
            if remaining < height:
                return turn_index
            remaining -= height
        return self.total_turns - 1

    def hydrated_chars(self) -> int:
        return sum(entry_cost(item.entry) for item in self.history_entries())

    async def open(self) -> None:
        total = self.total_turns
        if total <= 0:
            if self.index is not None and self.index.file_size > 0:
                await self._materialize_byte_range(0, self.index.file_size, start_turn=0)
            else:
                self._turns = []
            return
        self._ensure_turn_slots(total)
        await self.slide_to(total - 1)

    async def materialize_turns(self, start: int, end: int) -> int:
        """Decode missing turns into complete ``HistoryEntry`` tuples."""

        total = self.total_turns
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        self._ensure_turn_slots(total)
        missing = [turn for turn in range(start, end) if self._turns[turn] is None]
        if not missing:
            return 0
        if self._source_turns is not None:
            for turn in missing:
                self._turns[turn] = self._source_turns[turn]
            return len(missing)
        if self.index is None:
            return 0
        run_start = missing[0]
        run_end = missing[-1] + 1
        start_offset, _ = self.turn_range(run_start)
        _, end_offset = self.turn_range(run_end - 1)
        await self._materialize_byte_range(start_offset, end_offset, start_turn=run_start)
        return sum(self._turns[turn] is not None for turn in missing)

    def window_bounds(self, keep_turn: int, radius: int | None = None) -> tuple[int, int]:
        total = self.total_turns
        if total <= 0:
            return 0, 0
        keep = max(0, min(keep_turn, total - 1))
        if radius is not None:
            fixed = max(0, radius)
            return max(0, keep - fixed), min(total, keep + fixed + 1)

        start = keep
        end = keep + 1
        used_chars, used_records = self._turn_cost_hint(keep)
        left = keep - 1
        right = keep + 1
        left_blocked = False
        right_blocked = False
        while not (left_blocked and right_blocked):
            progressed = False
            for side in ("right", "left"):
                candidate = right if side == "right" else left
                if not 0 <= candidate < total:
                    if side == "right":
                        right_blocked = True
                    else:
                        left_blocked = True
                    continue
                chars, records = self._turn_cost_hint(candidate)
                if not self._fits_budget(used_chars + chars, used_records + records):
                    if side == "right":
                        right_blocked = True
                    else:
                        left_blocked = True
                    continue
                used_chars += chars
                used_records += records
                progressed = True
                if side == "right":
                    end = candidate + 1
                    right += 1
                else:
                    start = candidate
                    left -= 1
            if not progressed and (left_blocked or left < 0) and (right_blocked or right >= total):
                break
        return start, end

    def _turn_cost_hint(self, turn: int) -> tuple[int, int]:
        entries = self._turns[turn] if turn < len(self._turns) else None
        if entries is not None:
            return max(1, sum(entry_cost(item.entry) for item in entries)), max(1, len(entries))
        if self._source_turns is not None:
            source = self._source_turns[turn]
            return max(1, sum(entry_cost(item.entry) for item in source)), max(1, len(source))
        if self.index is not None:
            start, end = self.turn_range(turn)
            counts = self.index.turn_record_counts
            records = counts[turn] if turn < len(counts) else 1
            return max(1, end - start), max(1, records)
        return 1, 1

    def _fits_budget(self, chars: int, records: int) -> bool:
        chars_fit = self.hydrated_budget <= 0 or chars <= self.hydrated_budget
        records_fit = self.hydrated_record_budget <= 0 or records <= self.hydrated_record_budget
        return chars_fit and records_fit

    def drop_outside_window(self, *, keep_turn: int, radius: int | None = None) -> None:
        if self.total_turns <= 0:
            return
        self._drop_outside_bounds(*self.window_bounds(keep_turn, radius))

    def _drop_outside_bounds(self, keep_start: int, keep_end: int) -> None:
        for turn, entries in enumerate(self._turns):
            if entries is not None and not keep_start <= turn < keep_end:
                self._turns[turn] = None

    async def slide_to(self, turn: int, *, radius: int | None = None) -> None:
        total = self.total_turns
        if total <= 0:
            return
        turn = max(0, min(turn, total - 1))
        start, end = self.window_bounds(turn, radius)
        self._drop_outside_bounds(start, end)
        await self.materialize_turns(start, end)
        if radius is None:
            self._trim_to_budget(turn)

    async def ensure_turn(self, turn: int, *, radius: int | None = None) -> None:
        await self.slide_to(turn, radius=radius)

    async def rehydrate_turns(self, start: int, end: int) -> None:
        """Materialize dropped turns; no partial row hydration exists."""

        await self.materialize_turns(start, end)

    def unload_distant(self, *, keep_turn: int, radius: int = WINDOW_RADIUS) -> None:
        self.drop_outside_window(keep_turn=keep_turn, radius=radius)

    def _trim_to_budget(self, keep_turn: int) -> None:
        while not self._fits_budget(self.hydrated_chars(), self.materialized_record_count):
            materialized = [index for index, rows in enumerate(self._turns) if rows is not None]
            if len(materialized) <= 1:
                return
            first = materialized[0]
            last = materialized[-1]
            if first == keep_turn:
                remove = last
            elif last == keep_turn or keep_turn - first >= last - keep_turn:
                remove = first
            else:
                remove = last
            self._turns[remove] = None

    def _ensure_turn_slots(self, total: int) -> None:
        if len(self._turns) < total:
            self._turns.extend([None] * (total - len(self._turns)))

    async def _materialize_byte_range(
        self, start_offset: int, end_offset: int, *, start_turn: int
    ) -> None:
        if self.index is None:
            return
        turns = await asyncio.to_thread(_read_wire_turns, self.index.path, start_offset, end_offset)
        self._ensure_turn_slots(max(self.total_turns, start_turn + len(turns)))
        for offset, entries in enumerate(turns):
            turn = start_turn + offset
            if turn >= len(self._turns):
                break
            self._turns[turn] = tuple(
                HistoryEntry(entry=entry, turn=turn, ordinal=ordinal)
                for ordinal, entry in enumerate(entries)
            )


@dataclass(slots=True)
class WireHistoryPager:
    index: WireHistoryIndex
    page_turns: int = HISTORY_PAGE_TURNS
    max_blocks: int = MAX_HISTORY_BLOCKS

    async def latest(self, *, page_turns: int | None = None) -> SessionHistory:
        return await load_wire_history_page(
            self.index,
            end_turn=self.index.total_turns,
            page_turns=page_turns or self.page_turns,
            max_blocks=self.max_blocks,
        )

    async def before(self, end_turn: int, *, page_turns: int | None = None) -> SessionHistory:
        return await load_wire_history_page(
            self.index,
            end_turn=end_turn,
            page_turns=page_turns or self.page_turns,
            max_blocks=self.max_blocks,
        )

    async def ending_at(self, end_turn: int, *, page_turns: int | None = None) -> SessionHistory:
        return await self.before(end_turn, page_turns=page_turns)


@dataclass(slots=True)
class HistoryAccumulator:
    """Apply the same typed mutations as the live transcript, grouped by turn."""

    max_turns: int = 0
    max_blocks: int = 0
    omitted_turns: int = 0
    _turns: deque[list[TranscriptEntry]] = field(default_factory=deque)
    _current: TranscriptReducer = field(default_factory=TranscriptReducer)
    _normalizer: WireNormalizer = field(default_factory=WireNormalizer)
    _block_count: int = 0
    _unknown_sequence: int = 0

    def feed(self, message: object) -> None:
        if type(message).__name__ == "TurnBegin":
            self._flush_turn()
        normalized = self._normalizer.normalize(message)
        for mutation in normalized.mutations:
            self._apply(mutation)

    def feed_unknown(self, name: str, payload: object) -> None:
        """Retain an otherwise valid unknown envelope as a bounded raw block."""

        self._unknown_sequence += 1
        entry = NoticeEntry(
            key=f"history:unknown:{self._unknown_sequence}",
            kind="system",
            blocks=(RawBlock(_json_text(payload), label=literal(name)),),
        )
        self._apply(StartEntry(entry))

    def finish(self) -> SessionHistory:
        for mutation in self._normalizer.finish():
            self._apply(mutation)
        self._flush_turn()
        entries = tuple(
            HistoryEntry(entry=entry, turn=turn, ordinal=ordinal)
            for turn, values in enumerate(self._turns)
            for ordinal, entry in enumerate(values)
        )
        return SessionHistory(entries=entries, omitted_turns=self.omitted_turns)

    def finish_turns(self) -> list[tuple[TranscriptEntry, ...]]:
        for mutation in self._normalizer.finish():
            self._apply(mutation)
        self._flush_turn()
        return [tuple(turn) for turn in self._turns]

    def _apply(self, mutation: object) -> None:
        if not isinstance(mutation, ClearTranscript):
            effect = self._current.apply(mutation)  # type: ignore[arg-type]
        else:
            effect = self._current.apply(mutation)
        if effect.action == "insert":
            self._block_count += 1
            self._trim_blocks()

    def _trim_blocks(self) -> None:
        if self.max_blocks <= 0:
            return
        while self._block_count > self.max_blocks:
            if not self._drop_oldest_auxiliary_entry():
                return

    def _drop_oldest_auxiliary_entry(self) -> bool:
        for turn_index, turn in enumerate(self._turns):
            for entry_index, entry in enumerate(turn):
                if is_dialogue_entry(entry):
                    continue
                del turn[entry_index]
                self._block_count -= 1
                if not turn:
                    del self._turns[turn_index]
                return True
        for entry_index, entry in enumerate(self._current.entries):
            if is_dialogue_entry(entry):
                continue
            del self._current.entries[entry_index]
            self._current.replace_all(self._current.entries)
            self._block_count -= 1
            return True
        return False

    def _flush_turn(self) -> None:
        entries = list(self._current.take_all())
        if not entries:
            return
        self._turns.append(entries)
        while self.max_turns > 0 and len(self._turns) > self.max_turns:
            self._block_count -= len(self._turns.popleft())
            self.omitted_turns += 1


def entries_from_wire_messages(messages: Sequence[object]) -> tuple[HistoryEntry, ...]:
    accumulator = HistoryAccumulator()
    for message in messages:
        accumulator.feed(message)
    return accumulator.finish().entries


def take_last_turns(
    entries: Sequence[HistoryEntry], *, max_turns: int = MAX_HISTORY_TURNS
) -> SessionHistory:
    if max_turns <= 0 or not entries:
        return SessionHistory(entries=tuple(entries))
    turns = sorted({item.turn for item in entries})
    if len(turns) <= max_turns:
        return SessionHistory(entries=tuple(entries))
    first = turns[-max_turns]
    return SessionHistory(
        entries=tuple(item for item in entries if item.turn >= first),
        omitted_turns=len(turns) - max_turns,
    )


def _scan_wire_history_index(path: Path) -> WireHistoryIndex:
    offsets: list[int] = []
    record_counts: list[int] = []
    file_size = 0
    try:
        with path.open("rb") as wire_file:
            for line in wire_file:
                if _TURN_BEGIN_RECORD.search(line):
                    offsets.append(file_size)
                    record_counts.append(1)
                elif record_counts:
                    record_counts[-1] += 1
                file_size += len(line)
    except OSError:
        return WireHistoryIndex(path=path, turn_offsets=(), file_size=0)
    return WireHistoryIndex(
        path=path,
        turn_offsets=tuple(offsets),
        file_size=file_size,
        turn_record_counts=tuple(record_counts),
    )


async def create_timeline(work_dir: Path, session_id: str) -> Timeline | None:
    from kimi_cli.session import Session as CliSession

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return None
    index = await asyncio.to_thread(_scan_wire_history_index, cli_session.wire_file.path)
    timeline = Timeline(
        index=index,
        hydrated_budget=HYDRATED_BODY_BUDGET,
        hydrated_record_budget=HYDRATED_RECORD_BUDGET,
    )
    await timeline.open()
    return timeline


async def create_history_pager(
    work_dir: Path,
    session_id: str,
    *,
    page_turns: int = HISTORY_PAGE_TURNS,
    max_blocks: int = MAX_HISTORY_BLOCKS,
) -> WireHistoryPager | None:
    from kimi_cli.session import Session as CliSession

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return None
    index = await asyncio.to_thread(_scan_wire_history_index, cli_session.wire_file.path)
    return WireHistoryPager(
        index=index,
        page_turns=max(1, page_turns),
        max_blocks=max(0, max_blocks),
    )


def _json_text(value: object) -> str:
    try:
        return orjson.dumps(
            value,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=lambda item: str(item),
        ).decode("utf-8")
    except TypeError, ValueError:
        return str(value)


def _unknown_envelope(data: object) -> tuple[str, object] | None:
    if not isinstance(data, dict) or data.get("type") != "message":
        return None
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    name = message.get("type")
    if not isinstance(name, str) or not name:
        return None
    return name, message.get("payload", {})


def _read_wire_turns(
    path: Path,
    start_offset: int,
    end_offset: int,
    *,
    max_blocks: int = 0,
) -> list[tuple[TranscriptEntry, ...]]:
    """Parse one byte range; malformed JSON is logged, valid unknown payloads survive."""

    from kimi_cli.wire.file import WireFileMetadata, WireMessageRecord
    from pydantic_core import from_json

    accumulator = HistoryAccumulator(max_blocks=max_blocks)
    try:
        with path.open("rb") as wire_file:
            wire_file.seek(max(0, start_offset))
            position = max(0, start_offset)
            first_line = start_offset == 0
            while position < end_offset:
                raw_line = wire_file.readline()
                if not raw_line:
                    break
                line_offset = position
                position += len(raw_line)
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = from_json(raw_line, cache_strings=False)
                except Exception as exc:  # noqa: BLE001 - one corrupt record must not hide history
                    _LOGGER.warning("Skipping malformed wire JSON at byte %d: %s", line_offset, exc)
                    continue
                try:
                    if first_line and isinstance(data, dict) and data.get("type") == "metadata":
                        parsed = WireFileMetadata.model_validate(data)
                    else:
                        parsed = WireMessageRecord.model_validate(data)
                    first_line = False
                    to_wire_message = getattr(parsed, "to_wire_message", None)
                    if callable(to_wire_message):
                        accumulator.feed(to_wire_message())
                except Exception as exc:  # noqa: BLE001 - preserve valid unknown envelopes
                    unknown = _unknown_envelope(data)
                    if unknown is not None:
                        accumulator.feed_unknown(*unknown)
                    else:
                        _LOGGER.warning(
                            "Skipping invalid wire envelope at byte %d: %s", line_offset, exc
                        )
    except OSError:
        return []
    return accumulator.finish_turns()


def _read_wire_history_range(
    path: Path,
    start_offset: int,
    end_offset: int,
    *,
    max_blocks: int,
    start_turn: int = 0,
) -> tuple[HistoryEntry, ...]:
    turns = _read_wire_turns(path, start_offset, end_offset, max_blocks=max_blocks)
    return tuple(
        HistoryEntry(entry=entry, turn=start_turn + turn, ordinal=ordinal)
        for turn, entries in enumerate(turns)
        for ordinal, entry in enumerate(entries)
    )


async def load_wire_history_page(
    index: WireHistoryIndex,
    *,
    end_turn: int | None = None,
    page_turns: int = HISTORY_PAGE_TURNS,
    max_blocks: int = MAX_HISTORY_BLOCKS,
) -> SessionHistory:
    total = index.total_turns
    if total == 0:
        entries = await asyncio.to_thread(
            _read_wire_history_range,
            index.path,
            0,
            index.file_size,
            max_blocks=max(0, max_blocks),
        )
        return SessionHistory(entries=entries)
    end = total if end_turn is None else min(total, max(0, end_turn))
    if end == 0:
        return SessionHistory(entries=(), total_turns=total)
    start = max(0, end - max(1, page_turns))
    start_offset = 0 if start == 0 else index.turn_offsets[start]
    end_offset = index.file_size if end == total else index.turn_offsets[end]
    entries = await asyncio.to_thread(
        _read_wire_history_range,
        index.path,
        start_offset,
        end_offset,
        max_blocks=max(0, max_blocks),
        start_turn=start,
    )
    return SessionHistory(
        entries=entries,
        omitted_turns=start,
        total_turns=total,
        start_turn=start,
        end_turn=end,
        has_older=start > 0,
    )


def _tail_wire_record_lines(path: Path, max_turns: int) -> tuple[list[bytes], int]:
    turns: deque[list[bytes]] = deque()
    current: list[bytes] = []
    omitted_turns = 0
    with path.open("rb") as wire_file:
        for line in wire_file:
            if _TURN_BEGIN_RECORD.search(line):
                if current:
                    turns.append(current)
                    if len(turns) > max_turns:
                        turns.popleft()
                        omitted_turns += 1
                current = [line]
            elif current:
                current.append(line)
    if current:
        turns.append(current)
        if len(turns) > max_turns:
            turns.popleft()
            omitted_turns += 1
    return [line for turn in turns for line in turn], omitted_turns


async def _feed_wire_log(work_dir: Path, session_id: str, accumulator: HistoryAccumulator) -> None:
    from kimi_cli.session import Session as CliSession
    from kimi_cli.wire.file import parse_wire_file_line

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return
    if accumulator.max_turns > 0:
        try:
            lines, omitted = await asyncio.to_thread(
                _tail_wire_record_lines, cli_session.wire_file.path, accumulator.max_turns
            )
        except OSError:
            return
        accumulator.omitted_turns += omitted
        for line in lines:
            try:
                record = parse_wire_file_line(line)
                to_wire_message = getattr(record, "to_wire_message", None)
                if callable(to_wire_message):
                    accumulator.feed(to_wire_message())
            except Exception as exc:  # noqa: BLE001 - skip one bad historical record
                _LOGGER.warning("Skipping unreadable historical record: %s", exc)
        return
    async for record in cli_session.wire_file.iter_records():
        try:
            accumulator.feed(record.to_wire_message())
        except Exception as exc:  # noqa: BLE001 - skip one bad historical record
            _LOGGER.warning("Skipping unreadable historical record: %s", exc)


async def load_session_history(
    work_dir: Path,
    session_id: str,
    *,
    max_turns: int = 0,
    max_blocks: int = 0,
    messages: Sequence[object] | None = None,
) -> SessionHistory:
    accumulator = HistoryAccumulator(max_turns=max_turns, max_blocks=max_blocks)
    if messages is None:
        await _feed_wire_log(work_dir, session_id, accumulator)
    else:
        for message in messages:
            accumulator.feed(message)
    return accumulator.finish()
