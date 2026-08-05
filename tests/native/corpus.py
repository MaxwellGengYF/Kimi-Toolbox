"""Shared input corpus for root-package native equivalence tests.

Covers empty input, ASCII, CJK/mixed Unicode, control chars, boundary sizes
and malformed inputs per kernel.
"""

from __future__ import annotations

# STREAM kernel corpus (filter_output / _dedup_output)
STREAM_CORPUS = [
    "",
    "plain text no escapes",
    "a" * 4096,
    "line1\nline2\nline3",
    "crlf\r\nline\r\nlone\rhere",
    "\x1b[31mred\x1b[0m text",
    "\x1b]0;title\x07OSC title\x1b\\ end",
    "repeat\nrepeat\nrepeat\nrepeat\nrepeat\nunique",
    "block\nblock\nblock\nunique",
    "mixed 中文 with \x1b[1mbold\x1b[0m ANSI",
    "emoji 🎉\r\nline two",
    "line with tab\tand \x00nul",
]

# TOOLS kernel corpus (find_in_file content scans)
FIND_STR_CORPUS = [
    ("", "x"),
    ("hello world", "hello"),
    ("hello hello hello", "hello"),
    ("Mixed Case Line", "mixed"),
    ("case\nSENSITIVE\ncase", "CASE"),
    ("alpha beta\ngamma delta\nbeta gamma", "beta"),
    ("日本語テキスト\n中文内容\nmixed text", "中文"),
    ("a\nab\nabc\nabcd", "ab"),
    ("tab\tseparated\tvalues", "\t"),
    ("line1\nline2\nline3\n", "line"),
    ("x" * 5000 + "\nneedle\n" + "y" * 5000, "needle"),
]

# DEDUP corpus
DEDUP_CORPUS = [
    ("", 3, 1),
    ("a\nb\nc", 3, 1),
    ("x\nx\nx\nx\nx", 3, 1),
    ("x\nx\nx\nx\nx\ny\nx\nx\nx\nx", 3, 1),
    ("a\na\na\nb\nb\nb\nc", 2, 1),
    ("repeat\nrepeat\nrepeat\nrepeat\nunique\nrepeat", 3, 1),
    ("blk1\nblk2\nblk1\nblk2\nblk1\nblk2\nblk1\nblk2", 3, 2),
    ("same\nsame\nsame\nsame", 4, 1),
    ("l1\nl1\nl1\nl1\nl1\nl1\nl2", 3, 3),
    ("trailing empty\n", 3, 1),
    ("x\n" * 20 + "end", 5, 1),
]
