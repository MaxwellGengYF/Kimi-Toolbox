"""Edit parse guard: blackbox detection + auto-repair orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path

from kimi_cli.session import Session
from kimi_cli.tools.display import DisplayBlock
from kimi_cli.tools.file.auto_repair import (
    CompleteFn,
    EditAutoRepairOutcome,
    attempt_edit_auto_repair,
)
from kimi_cli.tools.file.blackbox import (
    AppliedEditSnapshot,
    NoOpBlackboxRecorder,
    create_edit_blackbox_recorder,
)
from kimi_cli.tools.file.parse_check import introduced_parse_failure
from kimi_cli.tools.file.snapshot_store import record_file_snapshot
from kimi_cli.utils.logging import logger


class EditParseGuard:
    """Per-call observer for edit/write parse safety.

    Mirrors oh-my-pi's per-tool-call `parseFailures` map and post-execute repair
    loop. All failures are best-effort and never turn a committed edit into an
    error.
    """

    def __init__(
        self,
        session: Session,
        *,
        variant: str,
        arg: object,
        enabled_for_write: bool = False,
        complete: CompleteFn | None = None,
    ) -> None:
        self._session = session
        self._variant = variant
        self._arg = arg
        self._enabled_for_write = enabled_for_write
        self._complete = complete
        self._parse_failures: dict[str, AppliedEditSnapshot] = {}
        self._recorder = create_edit_blackbox_recorder(session, variant, arg)

    async def observe_applied(self, path: str, prev: str, next: str) -> None:
        """Record a committed transition and detect introduced parse failures."""
        try:
            snapshot = AppliedEditSnapshot(path=path, prev=prev, next=next)
            await record_file_snapshot(self._session, path)
            if not introduced_parse_failure(snapshot.prev, snapshot.next, snapshot.path):
                self._parse_failures.pop(path, None)
                return
            self._parse_failures[path] = snapshot
            if not isinstance(self._recorder, NoOpBlackboxRecorder):
                await self._recorder.record(
                    snapshot,
                    variant=self._variant,
                    arg=self._arg,
                )
        except Exception as e:
            logger.debug("EditParseGuard.observe_applied failed", error=str(e))

    async def finish(self) -> list[str]:
        """Attempt repairs for any stashed parse failures and return note strings."""
        notes: list[str] = []
        for path, snapshot in self._parse_failures.items():
            try:
                repair = await attempt_edit_auto_repair(
                    self._session,
                    snapshot.path,
                    snapshot.prev,
                    snapshot.next,
                    complete=self._complete,
                    enabled_for_write=self._enabled_for_write,
                )
                display = self._display_path(snapshot.path)
                if repair:
                    notes.append(
                        f"Note: {display} stopped parsing after this edit; an automatic syntax repair "
                        f"({repair.model}) was applied on top:\n{repair.diff}\n"
                        "Review the repaired region; adjust it if the repair guessed wrong."
                    )
                else:
                    notes.append(
                        f"Warning: {display} no longer parses after this edit. "
                        "The change was applied; re-read the edited region and fix the syntax, "
                        "or revert if unintended."
                    )
            except Exception as e:
                logger.debug("EditParseGuard.finish repair failed", error=str(e))
                display = self._display_path(path)
                notes.append(
                    f"Warning: {display} no longer parses after this edit. "
                    "The change was applied; re-read the edited region and fix the syntax, "
                    "or revert if unintended."
                )
        self._parse_failures.clear()
        return notes

    def _display_path(self, path: str) -> str:
        try:
            rel = Path(path).relative_to(Path(str(self._session.work_dir)))
            return str(rel).replace("\\", "/")
        except ValueError:
            return path.replace("\\", "/")


def create_edit_parse_guard(
    session: Session,
    *,
    variant: str,
    arg: object,
    enabled_for_write: bool = False,
    complete: CompleteFn | None = None,
) -> EditParseGuard:
    """Factory helper used by edit/write tools."""
    return EditParseGuard(
        session,
        variant=variant,
        arg=arg,
        enabled_for_write=enabled_for_write,
        complete=complete,
    )
