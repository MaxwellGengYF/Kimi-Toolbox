"""Target-level anti-loop detection (P1, gaps G1/G2).

The toolset's streak detection keys on ``(tool_name, canonical_args)`` —
it catches *identical* calls only. Two loop shapes escape it:

1. **File churn**: the same file is modified over and over through
   *different* tools (``WriteFile`` → ``EditFile`` → ``Powershell`` sed)
   or with different arguments each time.
2. **Error churn**: the same underlying error (identical modulo line
   numbers / paths) is hit repeatedly without its root cause being fixed.

This provider scans new history incrementally, counts normalized file
targets across all edit/shell tools, and fingerprints tool errors. It
throttles aggressively: at most one alert per file per turn, one strong
alert per file per turn, and a cooldown after any injection.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import orjson
import regex as re
import xxhash
from kosong.message import Message

from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider
from kimi_cli.soul.tool_taxonomy import COMMAND_PARAM_KEYS, EDIT_TOOLS, PATH_PARAM_KEYS, SHELL_TOOLS

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_TARGET_CHURN_TYPE = "target_churn"

# Shell write-target extraction (redirects, sed -i, tee).
_REDIRECT_RE = re.compile(r"(?<![<>])>>?\s*([^\s;&|<>]+)")
_SED_I_RE = re.compile(r"\bsed\s+(?:-\w+\s+)*-i(?:\.\S+)?(?:\s+['\"][^'\"]*['\"])+\s+([^\s;&|]+)")
_TEE_RE = re.compile(r"\btee\s+(?:-\w+\s+)*([^\s;&|]+)")

# Error-signature normalization.
_DIGITS_RE = re.compile(r"\d+")
_PATH_LIKE_RE = re.compile(r"(?:[A-Za-z]:\\[^\s'\"]+|/[^\s'\"]+)")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_ERROR_PREFIX = "<system>ERROR:"


def _normalize_path(raw: str) -> str:
    """Normalize a path for cross-tool comparison (Windows-aware)."""
    return os.path.normcase(os.path.normpath(raw.strip().strip("'\"")))


def _normalize_error(text: str) -> str:
    """Normalize an error message so line numbers/paths/values don't matter."""
    text = _QUOTED_RE.sub("<str>", text)
    text = _PATH_LIKE_RE.sub("<path>", text)
    text = _DIGITS_RE.sub("<n>", text)
    return " ".join(text.split())


class TargetChurnProvider(DynamicInjectionProvider):
    """Detects repeated edits to the same target and repeated identical errors."""

    def __init__(
        self,
        *,
        file_warn: int = 5,
        file_strong: int = 8,
        error_warn: int = 3,
        cooldown_steps: int = 6,
    ) -> None:
        self._file_warn = max(2, file_warn)
        self._file_strong = max(self._file_warn + 1, file_strong)
        self._error_warn = max(2, error_warn)
        self._cooldown_steps = max(0, cooldown_steps)

        self._cursor = 0  # next unprocessed history index
        self._file_counts: Counter[str] = Counter()
        self._error_last_fingerprint: str | None = None
        self._error_streak = 0

        # Per-turn alert dedup.
        self._turn_id: str = ""
        self._warned_files: set[str] = set()
        self._strong_warned_files: set[str] = set()
        self._error_warned_this_turn = False

        self._last_alert_step: int | None = None

    # ------------------------------------------------------------------
    # Incremental history scan
    # ------------------------------------------------------------------

    def _extract_tool_call_targets(self, tool_name: str, arguments: str | None) -> list[str]:
        if not arguments:
            return []
        try:
            args = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            return []
        if not isinstance(args, dict):
            return []
        if tool_name in EDIT_TOOLS:
            for key in PATH_PARAM_KEYS:
                value = args.get(key)
                if isinstance(value, str) and value:
                    return [_normalize_path(value)]
            return []
        if tool_name in SHELL_TOOLS:
            command = ""
            for key in COMMAND_PARAM_KEYS:
                value = args.get(key)
                if isinstance(value, str) and value:
                    command = value
                    break
            if not command:
                return []
            targets: list[str] = []
            targets.extend(_REDIRECT_RE.findall(command))
            targets.extend(_SED_I_RE.findall(command))
            targets.extend(_TEE_RE.findall(command))
            return [_normalize_path(t) for t in targets]
        return []

    def _process_message(self, message: Message) -> None:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                for target in self._extract_tool_call_targets(
                    tool_call.function.name, tool_call.function.arguments
                ):
                    self._file_counts[target] += 1
        elif message.role == "tool":
            text = message.extract_text(" ")
            if _ERROR_PREFIX in text:
                # Take the first error line (exception type + message).
                error_text = text.split("</system>", 1)[0].removeprefix(_ERROR_PREFIX).strip()
                first_line = error_text.splitlines()[0] if error_text else ""
                fingerprint = xxhash.xxh64(_normalize_error(first_line).encode("utf-8")).hexdigest()
                if fingerprint == self._error_last_fingerprint:
                    self._error_streak += 1
                else:
                    self._error_last_fingerprint = fingerprint
                    self._error_streak = 1
            else:
                # A successful tool result breaks the error streak.
                self._error_last_fingerprint = None
                self._error_streak = 0

    def _sync_turn(self, soul: KimiSoul) -> None:
        turn_id = soul._current_turn_id  # pyright: ignore[reportPrivateUsage]
        if turn_id and turn_id != self._turn_id:
            self._turn_id = turn_id
            self._warned_files = set()
            self._strong_warned_files = set()
            self._error_warned_this_turn = False

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        self._sync_turn(soul)

        # Defensive: history may have been rebuilt without a compaction
        # notification (e.g. D-Mail revert); never scan backwards blindly.
        if self._cursor > len(history):
            self._cursor = 0
            self._file_counts = Counter()
            self._error_last_fingerprint = None
            self._error_streak = 0

        for message in history[self._cursor :]:
            self._process_message(message)
        self._cursor = len(history)

        step_no = soul._current_step_no  # pyright: ignore[reportPrivateUsage]
        if self._last_alert_step is not None and (step_no - self._last_alert_step) < self._cooldown_steps:
            return []

        # Strong file-churn alert has the highest priority.
        for path, count in self._file_counts.most_common():
            if count >= self._file_strong and path not in self._strong_warned_files:
                self._strong_warned_files.add(path)
                self._warned_files.add(path)
                self._last_alert_step = step_no
                return [
                    DynamicInjection(
                        type=_TARGET_CHURN_TYPE,
                        content=(
                            f"You have modified the same file `{path}` {count} times. "
                            "Repeated patching of one file is a strong signal of a wrong approach.\n"
                            "Stop patching. Rewrite the file as a whole from your current "
                            "understanding, or switch to a fundamentally different approach. "
                            "If tests keep failing, re-read the error and fix the root cause "
                            "instead of iterating on the same spot."
                        ),
                    )
                ]

        # Error-signature streak.
        if self._error_streak >= self._error_warn and not self._error_warned_this_turn:
            self._error_warned_this_turn = True
            self._last_alert_step = step_no
            return [
                DynamicInjection(
                    type=_TARGET_CHURN_TYPE,
                    content=(
                        f"The same error has occurred {self._error_streak} times in a row "
                        "(identical modulo line numbers/paths). Retrying the same fix is not "
                        "working.\n"
                        "Analyze the root cause: read the full error output, inspect the exact "
                        "code involved, and form a new hypothesis before your next action."
                    ),
                )
            ]

        # Normal file-churn alert.
        for path, count in self._file_counts.most_common():
            if count >= self._file_warn and path not in self._warned_files:
                self._warned_files.add(path)
                self._last_alert_step = step_no
                return [
                    DynamicInjection(
                        type=_TARGET_CHURN_TYPE,
                        content=(
                            f"You have edited `{path}` {count} times. If you are iterating "
                            "without progress, pause and reconsider: verify your understanding "
                            "of the failure, and consider rewriting the file or choosing a "
                            "different approach instead of another small patch."
                        ),
                    )
                ]

        return []

    async def on_context_compacted(self) -> None:
        """Reset all counters — compacted history no longer shows the churn."""
        self._cursor = 0
        self._file_counts = Counter()
        self._error_last_fingerprint = None
        self._error_streak = 0
        self._warned_files = set()
        self._strong_warned_files = set()
        self._error_warned_this_turn = False
        self._last_alert_step = None
