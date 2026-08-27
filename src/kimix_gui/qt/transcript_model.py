"""Qt row state over immutable transcript entries and a bounded history projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
from PySide6.QtWidgets import QWidget

from kimix_gui.qt.labels import translate_transcript_text
from kimix_gui.transcript_data import (
    AppendText,
    ClearTranscript,
    FinishEntry,
    HistoryEntry,
    TranscriptEntry,
    TranscriptMutation,
    TranscriptReducer,
    entry_body_text,
    entry_cost,
    is_dialogue_entry,
)
from kimix_gui.transcript_layout import accessibility_text, default_expanded

MAX_TRANSCRIPT_CHARS = 64 * 1024 * 1024
MAX_PRESENTATION_RECORDS = 192
MAX_PRESENTATION_CHARS = 2 * 1024 * 1024
DEFAULT_PRESENTATION_RECORDS = 64
_TRIM_TARGET_RATIO = 0.9
_RECORD_IDS = count()

type RecordId = tuple[str, int] | tuple[str, int, int]


@dataclass(slots=True)
class TranscriptRow:
    """Qt-only presentation state wrapped around one immutable semantic entry."""

    entry: TranscriptEntry
    expanded: bool
    turn: int | None = None
    record_id: RecordId = field(default_factory=lambda: ("live", next(_RECORD_IDS)))
    content_revision: int = 0

    @property
    def layout_chars(self) -> int:
        return entry_cost(self.entry)


def _new_row(
    entry: TranscriptEntry,
    *,
    turn: int | None = None,
    record_id: RecordId | None = None,
) -> TranscriptRow:
    identity = record_id or ("live", next(_RECORD_IDS))
    revision = hash(entry) if identity[0] == "history" else 0
    return TranscriptRow(
        entry=entry,
        expanded=default_expanded(entry),
        turn=turn,
        record_id=identity,
        content_revision=revision,
    )


def _row_from_history(item: HistoryEntry) -> TranscriptRow:
    return _new_row(
        item.entry,
        turn=item.turn,
        record_id=("history", item.turn, item.ordinal),
    )


@dataclass(slots=True)
class BodySelection:
    row: int
    anchor: int
    position: int

    @property
    def is_empty(self) -> bool:
        return self.anchor == self.position

    def normalized(self) -> tuple[int, int]:
        if self.anchor <= self.position:
            return self.anchor, self.position
        return self.position, self.anchor


class TranscriptModel(QAbstractListModel):
    """Store AST entries while Qt-only state remains mutable and local."""

    def __init__(
        self, parent: QWidget | None = None, *, max_chars: int = MAX_TRANSCRIPT_CHARS
    ) -> None:
        super().__init__(parent)
        self.records: list[TranscriptRow] = []
        self._max_chars = max(0, max_chars)
        self._record_chars = 0
        self._omitted_records = 0
        self._history_start: int | None = None
        self._history_end: int | None = None
        self._history_source: list[TranscriptRow] = []
        self._history_source_start = 0
        self._history_source_end = 0
        self._presentation_record_limit = min(
            DEFAULT_PRESENTATION_RECORDS, MAX_PRESENTATION_RECORDS
        )
        self._entry_positions: dict[str, int] = {}

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self.records)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or not 0 <= index.row() < len(self.records):
            return None
        row = self.records[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return entry_body_text(row.entry)
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return accessibility_text(row.entry, translate_transcript_text)
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None

    @property
    def omitted_records(self) -> int:
        return self._omitted_records

    @property
    def history_window(self) -> tuple[int, int] | None:
        if self._history_start is None or self._history_end is None:
            return None
        return self._history_start, self._history_end

    @property
    def has_hidden_older_history(self) -> bool:
        return self._history_source_start > 0

    @property
    def has_hidden_newer_history(self) -> bool:
        return self._history_source_end < len(self._history_source)

    @property
    def has_history_source(self) -> bool:
        return bool(self._history_source)

    @property
    def presentation_record_limit(self) -> int:
        """Maximum rows exposed to the view for its current viewport size."""

        return self._presentation_record_limit

    def set_presentation_record_limit(self, limit: int) -> bool:
        """Configure the exact-layout window without changing the current projection."""

        normalized = max(1, min(int(limit), MAX_PRESENTATION_RECORDS))
        if normalized == self._presentation_record_limit:
            return False
        self._presentation_record_limit = normalized
        return True

    def mark_history_window(self, start: int | None = None, end: int | None = None) -> int:
        position = len(self.records) if start is None else max(0, min(start, len(self.records)))
        self._history_start = position
        self._history_end = position if end is None else max(position, min(end, len(self.records)))
        return position

    def apply_mutation(self, mutation: TranscriptMutation) -> None:
        """Apply the canonical reducer semantics and mirror its one-row effect into Qt."""

        if isinstance(mutation, ClearTranscript):
            self.clear_messages()
            return
        key = mutation.key if isinstance(mutation, AppendText | FinishEntry) else mutation.entry.key
        position = self._entry_positions.get(key)
        reducer = TranscriptReducer(() if position is None else (self.records[position].entry,))
        effect = reducer.apply(mutation)
        if effect.action == "unchanged" or effect.entry is None:
            return
        if effect.action == "insert":
            self._insert_rows(len(self.records), (_new_row(effect.entry),))
            return
        if position is None:
            return
        row = self.records[position]
        old_cost = row.layout_chars
        row.entry = effect.entry
        row.content_revision += 1
        self._record_chars += row.layout_chars - old_cost
        self._trim_records()
        changed_position = self._entry_positions.get(key)
        if changed_position is not None:
            model_index = self.index(changed_position)
            self.dataChanged.emit(model_index, model_index)

    def apply_mutations(self, mutations: Sequence[TranscriptMutation]) -> None:
        for mutation in mutations:
            self.apply_mutation(mutation)

    def replace_history(
        self,
        entries: Sequence[HistoryEntry],
        *,
        target_turn: int | None = None,
        pin_latest: bool = False,
    ) -> None:
        """Replace the decoded AST source and expose one bounded presentation slice."""

        start = self._history_start
        if start is None:
            start = self.mark_history_window()
        end = self._history_end or start
        prefix = self.records[:start]
        suffix = self.records[end:]
        old_history = self._history_source or self.records[start:end]
        base_chars = self._record_chars - sum(row.layout_chars for row in old_history)
        replacement = [_row_from_history(item) for item in entries]
        if self._max_chars > 0:
            available = max(0, self._max_chars - base_chars)
            replacement_chars = sum(row.layout_chars for row in replacement)
            while replacement and replacement_chars > available:
                removable = next(
                    (
                        index
                        for index, row in enumerate(replacement)
                        if not is_dialogue_entry(row.entry)
                    ),
                    None,
                )
                if removable is None:
                    break
                replacement_chars -= replacement[removable].layout_chars
                del replacement[removable]
        source_start, source_end = self._initial_presentation_range(
            replacement,
            target_turn=target_turn,
            pin_latest=pin_latest,
        )
        self.beginResetModel()
        self._history_source = replacement
        self._history_source_start = source_start
        self._history_source_end = source_end
        self.records[:] = prefix + replacement[source_start:source_end] + suffix
        self._record_chars = base_chars + sum(row.layout_chars for row in replacement)
        self._history_start = start
        self._history_end = start + source_end - source_start
        self._reindex_entries()
        self.endResetModel()

    def reveal_older_history(self) -> bool:
        old_start = self._history_source_start
        if old_start <= 0:
            return False
        start, end = self._grow_presentation_range(
            old_start - 1,
            min(len(self._history_source), old_start + 1),
            prefer="older",
        )
        return self._install_history_slice(start, end)

    def reveal_newer_history(self) -> bool:
        old_end = self._history_source_end
        if old_end >= len(self._history_source):
            return False
        start, end = self._grow_presentation_range(max(0, old_end - 1), old_end + 1, prefer="newer")
        return self._install_history_slice(start, end)

    def reveal_turn(self, turn: int, *, pin_latest: bool = False) -> bool:
        if not self._history_source:
            return False
        if pin_latest:
            start, end = self._initial_presentation_range(
                self._history_source, target_turn=None, pin_latest=True
            )
        else:
            target = next(
                (index for index, row in enumerate(self._history_source) if row.turn == turn),
                None,
            )
            if target is None:
                return False
            start, end = self._grow_presentation_range(target, target + 1, prefer="target")
        if start == self._history_source_start and end == self._history_source_end:
            return True
        self._install_history_slice(start, end)
        return True

    def row_for_record(self, record_id: RecordId) -> int | None:
        return next(
            (index for index, row in enumerate(self.records) if row.record_id == record_id),
            None,
        )

    def _initial_presentation_range(
        self,
        records: Sequence[TranscriptRow],
        *,
        target_turn: int | None,
        pin_latest: bool,
    ) -> tuple[int, int]:
        if not records:
            return 0, 0
        if pin_latest or target_turn is None:
            last = len(records) - 1
            return self._grow_range(records, last, last + 1, prefer="latest")
        target = next(
            (index for index, row in enumerate(records) if row.turn == target_turn),
            len(records) - 1,
        )
        return self._grow_range(records, target, target + 1, prefer="target")

    def _grow_presentation_range(self, start: int, end: int, *, prefer: str) -> tuple[int, int]:
        return self._grow_range(self._history_source, start, end, prefer=prefer)

    def _grow_range(
        self,
        records: Sequence[TranscriptRow],
        start: int,
        end: int,
        *,
        prefer: str,
    ) -> tuple[int, int]:
        start = max(0, min(start, len(records)))
        end = max(start, min(end, len(records)))
        chars = sum(row.layout_chars for row in records[start:end])

        def can_add(index: int) -> bool:
            return (
                end - start < self._presentation_record_limit
                and chars + records[index].layout_chars <= MAX_PRESENTATION_CHARS
            )

        def prepend(limit: int | None = None) -> None:
            nonlocal start, chars
            added = 0
            while start > 0 and (limit is None or added < limit):
                candidate = start - 1
                if not can_add(candidate):
                    break
                start = candidate
                chars += records[candidate].layout_chars
                added += 1

        def append(limit: int | None = None) -> None:
            nonlocal end, chars
            added = 0
            while end < len(records) and (limit is None or added < limit):
                if not can_add(end):
                    break
                chars += records[end].layout_chars
                end += 1
                added += 1

        if prefer == "latest":
            prepend()
        elif prefer == "older":
            prepend(self._presentation_shift_records())
            append()
            prepend()
        elif prefer == "newer":
            append(self._presentation_shift_records())
            prepend()
            append()
        else:
            prepend(self._presentation_shift_records() // 2)
            append()
            prepend()
        return start, end

    def _presentation_shift_records(self) -> int:
        """Move about one viewport while retaining most currently measured rows."""

        return max(1, self._presentation_record_limit // 3)

    def _install_history_slice(self, source_start: int, source_end: int) -> bool:
        if source_start == self._history_source_start and source_end == self._history_source_end:
            return False
        history_start = self._history_start
        if history_start is None:
            return False
        old_start = self._history_source_start
        old_end = self._history_source_end
        overlap_start = max(old_start, source_start)
        overlap_end = min(old_end, source_end)
        if overlap_start >= overlap_end:
            self._remove_visible_history(history_start, old_end - old_start)
            self._insert_visible_history(
                history_start, self._history_source[source_start:source_end]
            )
        else:
            old_right = old_end - overlap_end
            self._remove_visible_history(
                history_start + overlap_end - old_start,
                old_right,
            )
            old_left = overlap_start - old_start
            self._remove_visible_history(history_start, old_left)

            new_left = self._history_source[source_start:overlap_start]
            self._insert_visible_history(history_start, new_left)
            new_right = self._history_source[overlap_end:source_end]
            self._insert_visible_history(
                history_start + len(new_left) + overlap_end - overlap_start,
                new_right,
            )
        self._history_source_start = source_start
        self._history_source_end = source_end
        self._history_end = history_start + source_end - source_start
        self._reindex_entries()
        return True

    def _remove_visible_history(self, start: int, count: int) -> None:
        if count <= 0:
            return
        self.beginRemoveRows(QModelIndex(), start, start + count - 1)
        del self.records[start : start + count]
        self.endRemoveRows()

    def _insert_visible_history(self, start: int, rows: Sequence[TranscriptRow]) -> None:
        if not rows:
            return
        self.beginInsertRows(QModelIndex(), start, start + len(rows) - 1)
        self.records[start:start] = rows
        self.endInsertRows()

    def clear_messages(self) -> None:
        self.beginResetModel()
        self.records.clear()
        self._history_source.clear()
        self._history_source_start = 0
        self._history_source_end = 0
        self._record_chars = 0
        self._history_start = None
        self._history_end = None
        self._entry_positions.clear()
        self.endResetModel()

    def toggle_expanded(self, row_index: int) -> None:
        if not 0 <= row_index < len(self.records):
            return
        row = self.records[row_index]
        if is_dialogue_entry(row.entry):
            return
        row.expanded = not row.expanded
        row.content_revision += 1
        index = self.index(row_index)
        self.dataChanged.emit(index, index)

    def _insert_rows(self, index: int, rows: Sequence[TranscriptRow]) -> None:
        if not rows:
            return
        index = max(0, min(index, len(self.records)))
        self.beginInsertRows(QModelIndex(), index, index + len(rows) - 1)
        self.records[index:index] = rows
        self._record_chars += sum(row.layout_chars for row in rows)
        self.endInsertRows()
        self._trim_records()
        self._reindex_entries()

    def _trim_records(self) -> None:
        if self._history_start is not None:
            return
        if self._max_chars <= 0 or self._record_chars <= self._max_chars:
            return
        target = max(1, int(self._max_chars * _TRIM_TARGET_RATIO))
        remove_count = 0
        removed_chars = 0
        while remove_count < len(self.records) - 1 and self._record_chars - removed_chars > target:
            removed_chars += self.records[remove_count].layout_chars
            remove_count += 1
        if not remove_count:
            return
        self.beginRemoveRows(QModelIndex(), 0, remove_count - 1)
        del self.records[:remove_count]
        self.endRemoveRows()
        self._record_chars -= removed_chars
        self._omitted_records += remove_count
        self._reindex_entries()

    def _reindex_entries(self) -> None:
        self._entry_positions = {row.entry.key: index for index, row in enumerate(self.records)}


# Existing code and tests use the generic word "record" for Qt row state. Keep a
# single type object, not a second data representation.
TranscriptRecord = TranscriptRow

__all__ = [
    "MAX_PRESENTATION_CHARS",
    "MAX_PRESENTATION_RECORDS",
    "MAX_TRANSCRIPT_CHARS",
    "BodySelection",
    "RecordId",
    "TranscriptModel",
    "TranscriptRecord",
    "TranscriptRow",
]
