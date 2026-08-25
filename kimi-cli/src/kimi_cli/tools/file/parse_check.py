"""Per-language syntax checkers for edit/write blackbox detection."""

from __future__ import annotations

import ast
import contextlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kimi_cli.tools.file.check_fmt import (
    check_json_text,
    check_toml_text,
    check_xml_text,
    check_yaml_text,
)
from kimi_cli.utils.logging import logger


def source_parses(text: str, file_path: str | Path) -> bool:
    """Return True when `text` parses for the language implied by `file_path`.

    Unknown/unsupported languages return True so we never false-positive a
    parse failure for a file we cannot check.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    if suffix == ".json":
        return check_json_text(text) is None

    if suffix in (".yaml", ".yml"):
        return check_yaml_text(text) is None

    if suffix == ".toml":
        return check_toml_text(text) is None

    if suffix == ".xml":
        return check_xml_text(text) is None

    if suffix in (".js", ".mjs", ".cjs"):
        return _js_source_parses(text, Path(file_path))

    if suffix in (".ts", ".tsx"):
        # Optional TypeScript checking is deferred; treat as unknown.
        return True

    # Unknown / non-source files are not checked.
    return True


_node_available: bool | None = None


def _js_source_parses(text: str, file_path: Path) -> bool:
    """Optional `node --check` subprocess checker for JavaScript files.

    If `node` is not on PATH, the checker degrades to True (cannot prove failure).
    All subprocess failures are treated as "cannot prove a parse failure".
    """
    global _node_available  # noqa: PLW0603
    if _node_available is None:
        _node_available = shutil.which("node") is not None
    if not _node_available:
        return True

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=file_path.suffix,
            prefix="node_check_",
            delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
    except OSError:
        return True

    try:
        result = subprocess.run(
            ["node", "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return True
    finally:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)


def introduced_parse_failure(prev: str, next: str, file_path: str | Path) -> bool:
    """True when `prev` parses and `next` does not, for the given file path."""
    if source_parses(next, file_path):
        return False
    return source_parses(prev, file_path)
