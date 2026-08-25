"""Backward-compatible shim for the literal-replace edit tool.

The implementation has moved to ``kimi_cli.tools.file.edit``. This module
re-exports the names that existing code and tests import from here.
"""

from __future__ import annotations

from kimi_cli.tools.file.edit import EditFile
from kimi_cli.tools.file.edit.params import EditParams as Params, ReplaceEditItem as Edit

__all__ = ["Edit", "EditFile", "Params"]
