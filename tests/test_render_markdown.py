"""Tests for kimix.cli_impl.utils.render_markdown inline emphasis handling."""
from __future__ import annotations

import regex as re

from kimix.cli_impl.utils import render_markdown

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_underscore_token_is_not_transformed() -> None:
    """``_TEXT_`` must stay literal: no underscores eaten, no color applied."""
    rendered = render_markdown("prefix _TEXT_ suffix")
    assert "_TEXT_" in rendered
    # A plain paragraph without any (remaining) inline markup carries no ANSI.
    assert rendered == _strip_ansi(rendered)


def test_snake_case_identifiers_are_preserved() -> None:
    rendered = render_markdown("call my_function now")
    assert "my_function" in _strip_ansi(rendered)


def test_asterisk_italic_still_rendered() -> None:
    """``*text*`` keeps the italic markdown transform."""
    rendered = render_markdown("prefix *text* suffix")
    plain = _strip_ansi(rendered)
    # Asterisks are consumed and the content is wrapped in ANSI styles.
    assert "*text*" not in plain
    assert "text" in plain
    assert rendered != plain
