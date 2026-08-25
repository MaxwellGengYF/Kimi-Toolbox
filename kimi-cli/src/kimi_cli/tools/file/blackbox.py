"""Failure-isolated blackbox recorder for edit parse regressions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

from kimi_cli.session import Session
from kimi_cli.utils.logging import logger


@dataclass(frozen=True)
class AppliedEditSnapshot:
    """A committed file transition produced by an edit/write operation."""

    path: str
    prev: str
    next: str


class NoOpBlackboxRecorder:
    """Recorder that silently swallows all records."""

    async def record(
        self,
        snapshot: AppliedEditSnapshot,
        *,
        variant: str,
        arg: object,
    ) -> None:
        return None


class EditBlackboxRecorder:
    """Append-only JSONL recorder for introduced parse failures."""

    def __init__(self, log_path: Path, *, model: str, snapshot_max_bytes: int = 4 * 1024 * 1024) -> None:
        self._log_path = log_path
        self._model = model
        self._snapshot_max_bytes = snapshot_max_bytes

    async def record(
        self,
        snapshot: AppliedEditSnapshot,
        *,
        variant: str,
        arg: object,
    ) -> None:
        """Append a regression snapshot to the blackbox log.

        Failure-isolated: any exception is logged and swallowed.
        """
        try:
            if len(snapshot.prev) + len(snapshot.next) > self._snapshot_max_bytes:
                logger.debug(
                    "skipping blackbox record: snapshot exceeds size limit",
                    path=str(self._log_path),
                    snapshot_bytes=len(snapshot.prev) + len(snapshot.next),
                    limit=self._snapshot_max_bytes,
                )
                return

            payload = {
                "path": snapshot.path,
                "prev": snapshot.prev,
                "next": snapshot.next,
                "model": self._model,
                "variant": variant,
                "arg": _json_safe(arg),
                "ts": time.time(),
            }
            line = orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)

            def _append() -> None:
                parent = Path(str(self._log_path.parent))
                parent.mkdir(parents=True, exist_ok=True)
                with Path(str(self._log_path)).open("ab") as f:
                    f.write(line)

            import asyncio
            await asyncio.to_thread(_append)
        except Exception as e:
            logger.debug(
                "failed to record edit parse regression",
                path=str(self._log_path),
                error=str(e),
            )


def _json_safe(arg: object) -> Any:
    """Best-effort convert an arbitrary object into JSON-safe data."""
    try:
        orjson.dumps(arg)
        return arg
    except (TypeError, ValueError):
        try:
            return str(arg)
        except Exception:
            return None


def create_edit_blackbox_recorder(
    session: Session,
    variant: str,
    arg: object,
    *,
    snapshot_max_bytes: int = 4 * 1024 * 1024,
) -> EditBlackboxRecorder | NoOpBlackboxRecorder:
    """Create a recorder for this session, or a no-op recorder if disabled/failing."""
    try:
        config = session.custom_config.get("config_json", {}).get("edit", {})
        blackbox_cfg = config.get("blackbox", {})
        if not blackbox_cfg.get("enabled", True):
            return NoOpBlackboxRecorder()

        log_path = Path(str(session.work_dir)) / ".kimix_cache" / "edit-blackbox.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        model = session.custom_config.get("chat_provider", "unknown")
        return EditBlackboxRecorder(
            log_path,
            model=model,
            snapshot_max_bytes=snapshot_max_bytes,
        )
    except Exception as e:
        logger.debug(
            "failed to create edit blackbox recorder",
            error=str(e),
        )
        return NoOpBlackboxRecorder()
