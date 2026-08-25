"""Tests for parse_check syntax helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kimi_cli.tools.file.parse_check import introduced_parse_failure, source_parses


@pytest.mark.parametrize(
    "text, path, expected",
    [
        ("x = 1\n", "foo.py", True),
        ("x =\n", "foo.py", False),
        ("{\"a\": 1}", "foo.json", True),
        ("{\"a\": broken}", "foo.json", False),
        ("a: 1\n", "foo.yaml", True),
        ("a: [", "foo.yaml", False),
        ("[table]\nkey = 1\n", "foo.toml", True),
        ("[table\nkey = 1\n", "foo.toml", False),
        ("<root></root>", "foo.xml", True),
        ("<root><broken</root>", "foo.xml", False),
        ("anything", "foo.txt", True),
        ("", "foo.md", True),
    ],
)
def test_source_parses(text: str, path: str, expected: bool) -> None:
    assert source_parses(text, path) is expected


@pytest.mark.parametrize(
    "prev, next, path, expected",
    [
        ("x = 1\n", "x = 2\n", "foo.py", False),
        ("x = 1\n", "x =\n", "foo.py", True),
        ("x =\n", "x = 1\n", "foo.py", False),
        ("x =\n", "x =\n", "foo.py", False),
    ],
)
def test_introduced_parse_failure(prev: str, next: str, path: str, expected: bool) -> None:
    assert introduced_parse_failure(prev, next, path) is expected


def test_introduced_parse_failure_short_circuits_next_first() -> None:
    """When `next` parses, `prev` should not be re-parsed."""
    with patch("kimi_cli.tools.file.parse_check.source_parses") as mock:
        mock.return_value = True
        assert introduced_parse_failure("x = 1\n", "x = 2\n", "foo.py") is False
        # Should be called once with next, never with prev.
        mock.assert_called_once_with("x = 2\n", "foo.py")
