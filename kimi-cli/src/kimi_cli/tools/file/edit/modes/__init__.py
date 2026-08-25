"""Mode executors for the multi-mode edit tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hashline import HashlineModeExecutor
from .patch import PatchModeExecutor
from .replace import ReplaceModeExecutor
from .sloppy import SloppyModeExecutor

if TYPE_CHECKING:
    from collections.abc import Callable

    from kimi_cli.tools.file.edit.params import EditMode

MODE_REGISTRY: dict[EditMode, type] = {
    "replace": ReplaceModeExecutor,
    "patch": PatchModeExecutor,
    "hashline": HashlineModeExecutor,
    "sloppy": SloppyModeExecutor,
}
