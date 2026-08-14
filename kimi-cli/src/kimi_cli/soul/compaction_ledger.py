"""Durable, append-only JSONL ledger of compaction transactions.

Phase 3 (§5.1 of the compaction hardening plan): each compaction that makes an
LLM call records a transaction (``compaction_id``, shadowed range/tokens,
summary/preserved token estimates, shrink outcome). The ledger lives next to
the session export at ``<session_dir>/.kimix_cache/compaction_ledger.jsonl`` so
the durability story matches the deterministic export path.

Failure isolation is a hard guarantee: *no* public method raises. A broken or
unwritable path degrades to a ``logger.warning`` so a ledger problem can never
break ``SimpleCompaction.compact``.

The file format is one JSON object per line (JSONL), written with ``orjson``
per the AGENTS.md performance table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson

from kimi_cli.utils.logging import logger


@dataclass(frozen=True, slots=True)
class CompactionRecord:
    """One compaction transaction, as persisted to the ledger."""

    compaction_id: str                 # uuid4 hex
    trigger: Literal["auto", "manual", "overflow"]
    started_at: float                  # monotonic-ish epoch (time.time())
    shadowed_range: tuple[int, int]    # history indices [start, end) replaced
    shadowed_tokens: int               # estimated tokens of replaced region
    summary_tokens: int                # LLM usage.output when available, else estimate
    preserved_tokens: int              # estimated tokens of preserved tail
    shrank: bool                       # summary_tokens < shadowed_tokens
    error: str | None = None           # set on failure (end-of-transaction)


def _record_to_dict(record: CompactionRecord) -> dict:
    return {
        "compaction_id": record.compaction_id,
        "trigger": record.trigger,
        "started_at": record.started_at,
        "shadowed_range": [record.shadowed_range[0], record.shadowed_range[1]],
        "shadowed_tokens": record.shadowed_tokens,
        "summary_tokens": record.summary_tokens,
        "preserved_tokens": record.preserved_tokens,
        "shrank": record.shrank,
        "error": record.error,
    }


def _record_from_dict(data: dict) -> CompactionRecord:
    shadowed_range = data.get("shadowed_range") or (0, 0)
    return CompactionRecord(
        compaction_id=data["compaction_id"],
        trigger=data["trigger"],
        started_at=data["started_at"],
        shadowed_range=(shadowed_range[0], shadowed_range[1]),
        shadowed_tokens=data["shadowed_tokens"],
        summary_tokens=data["summary_tokens"],
        preserved_tokens=data["preserved_tokens"],
        shrank=data["shrank"],
        error=data.get("error"),
    )


class CompactionLedger:
    """Append-only JSONL ledger of compaction transactions.

    ``path=None`` creates a no-op ledger: every method silently does nothing
    (used when the ledger is disabled or its directory cannot be created).
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path

    @classmethod
    def for_session(cls, session_dir: Path, *, enabled: bool) -> "CompactionLedger":
        """Build the ledger for a session.

        Path: ``session_dir / ".kimix_cache" / "compaction_ledger.jsonl"``.
        When ``enabled=False`` (or the cache directory cannot be created)
        return a no-op ledger whose methods never touch the filesystem.
        """
        if not enabled:
            return cls(None)
        path = session_dir / ".kimix_cache" / "compaction_ledger.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Cannot create compaction ledger directory {dir}: {err}; "
                "disabling the ledger for this session",
                dir=path.parent,
                err=exc,
            )
            return cls(None)
        return cls(path)

    def record_start(self, record: CompactionRecord) -> None:
        """Append the start-of-transaction record. Never raises."""
        if self._path is None:
            return
        try:
            with self._path.open("ab") as f:
                f.write(orjson.dumps(_record_to_dict(record), option=orjson.OPT_APPEND_NEWLINE))
        except Exception as exc:  # noqa: BLE001 — failure isolation is the contract
            logger.warning(
                "Failed to record compaction start in ledger {path}: {err}",
                path=self._path,
                err=exc,
            )

    def record_end(
        self,
        compaction_id: str,
        *,
        error: str | None = None,
        summary_tokens: int | None = None,
        shrank: bool | None = None,
    ) -> None:
        """Finalize a transaction. Never raises.

        Design choice (documented): rewrite-in-place rather than append-a-second
        line. ``record_end`` reads the whole file, replaces the single line whose
        ``compaction_id`` matches, and rewrites the file — so the ledger stays
        exactly one line per transaction and ``latest()`` is the last *finished*
        transaction. On success ``error`` is cleared and (when provided)
        ``summary_tokens`` / ``shrank`` are updated with the real outcome; on
        failure ``error`` is set.

        If no matching line exists (e.g. the start was never persisted because
        of an earlier I/O failure) the file is left untouched and a warning is
        logged — we never invent data for a transaction we did not record.
        """
        if self._path is None:
            return
        try:
            records = self._read_records()
            found = False
            for rec in records:
                if rec.get("compaction_id") == compaction_id:
                    if error is not None:
                        rec["error"] = error
                    else:
                        rec.pop("error", None)
                    if summary_tokens is not None:
                        rec["summary_tokens"] = summary_tokens
                    if shrank is not None:
                        rec["shrank"] = shrank
                    found = True
                    break
            if not found:
                logger.warning(
                    "Cannot finalize compaction {cid}: no start record in ledger {path}",
                    cid=compaction_id,
                    path=self._path,
                )
                return
            self._write_records(records)
        except Exception as exc:  # noqa: BLE001 — failure isolation is the contract
            logger.warning(
                "Failed to finalize compaction {cid} in ledger {path}: {err}",
                cid=compaction_id,
                path=self._path,
                err=exc,
            )

    def latest(self) -> CompactionRecord | None:
        """Return the last record in the file, or None. Never raises."""
        if self._path is None:
            return None
        try:
            records = self._read_records()
            if not records:
                return None
            return _record_from_dict(records[-1])
        except Exception as exc:  # noqa: BLE001 — failure isolation is the contract
            logger.warning(
                "Failed to read compaction ledger {path}: {err}",
                path=self._path,
                err=exc,
            )
            return None

    def _read_records(self) -> list[dict]:
        """Read all records; malformed lines are skipped with a warning."""
        records: list[dict] = []
        if not self._path.exists():
            return records
        with self._path.open("rb") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(orjson.loads(line))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping malformed compaction ledger line {lineno} in {path}: {err}",
                        lineno=lineno,
                        path=self._path,
                        err=exc,
                    )
        return records

    def _write_records(self, records: list[dict]) -> None:
        """Rewrite the whole file from the parsed record list."""
        with self._path.open("wb") as f:
            for rec in records:
                f.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
