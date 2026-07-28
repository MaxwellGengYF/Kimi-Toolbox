"""Compatibility shim for JSON/XML validation helpers (P9).

The canonical implementations now live in
:mod:`kimi_cli.tools.file.check_fmt`. This module re-exports them for
backward compatibility and will be removed in the next minor release
(see ``ChangeLog.md``).
"""

from __future__ import annotations

import warnings

from kimi_cli.tools.file.check_fmt import (
    check_json,
    check_json_str,
    check_xml,
    check_xml_str,
)

warnings.warn(
    "kimix.tools.check_fmt is deprecated; import from "
    "kimi_cli.tools.file.check_fmt instead. This shim will be removed "
    "in the next minor release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "check_json",
    "check_json_str",
    "check_xml",
    "check_xml_str",
]
