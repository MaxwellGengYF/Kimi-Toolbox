"""Base parser class and data models for source code comment parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Comment:
    """Represents a single comment found in source code."""
    content: str
    line: int
    column: int
    kind: str  # "line", "block", "doc"


@dataclass
class ParseResult:
    """Result of parsing source code for comments."""
    language: str
    comments: list[Comment] = field(default_factory=list)
    code_without_comments: str = ""

    @property
    def total_comments(self) -> int:
        return len(self.comments)

    @property
    def comment_lines(self) -> int:
        return sum(1 for c in self.comments for _ in c.content.splitlines() if _)

    def get_comments_by_kind(self, kind: str) -> list[Comment]:
        return [c for c in self.comments if c.kind == kind]


def native_parse_result(
    lang: str,
    app_language: str,
    source_code: str,
) -> ParseResult | None:
    """Route a parser invocation to the native kernel (kimix_native.parse).

    Returns an app-shaped :class:`ParseResult` (comments converted to this
    module's :class:`Comment`) when the native path is active, else None so
    the caller runs its original pure-Python body unchanged.

    Args:
        lang: native language key ("c", "python", "shell", "sql",
            "html", "lisp", "pascal").
        app_language: the language label the app parsers put into
            ``ParseResult.language`` (e.g. "C", "Python").
        source_code: source text to parse.
    """
    try:
        from kimix.native_loader import get_module, use_native
    except Exception:
        # kimix.native_loader unavailable (e.g. kimix-base's isolated
        # reference-test env loads this file into a synthetic package):
        # run the pure-Python body unchanged.
        return None

    if not use_native("PARSE"):
        return None
    mod = get_module("parse")
    if mod is None:
        return None
    result = mod.parse(lang, source_code)
    return ParseResult(
        language=app_language,
        comments=[
            Comment(
                content=c.content,
                line=c.line,
                column=c.column,
                kind=c.kind,
            )
            for c in result.comments
        ],
        code_without_comments=result.code_without_comments,
    )


class BaseParser(ABC):
    """Abstract base class for all language parsers."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def parse(self, source_code: str) -> ParseResult:
        """Parse source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        ...

    def parse_file(self, file_path: str, encoding: str = "utf-8") -> ParseResult:
        """Parse a source file and extract comments.

        Args:
            file_path: Path to the source file.
            encoding: File encoding (default: utf-8).

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        with open(file_path, encoding=encoding) as f:
            source_code = f.read()
        return self.parse(source_code)

    def _build_result(self, language: str, source_code: str, comments: list[Comment]) -> ParseResult:
        """Build a ParseResult with code stripped of comments."""
        lines = source_code.splitlines(keepends=True)
        # Sort comments by line (and column) in reverse to avoid offset issues
        sorted_comments = sorted(comments, key=lambda c: (c.line, c.column), reverse=True)

        for comment in sorted_comments:
            if 1 <= comment.line <= len(lines):
                line = lines[comment.line - 1]
                # Replace the comment portion with whitespace
                col = comment.column - 1  # convert to 0-based
                end_col = min(col + len(comment.content), len(line))
                if col < len(line):
                    # Preserve indentation structure by replacing with spaces
                    replacement = " " * len(comment.content)
                    lines[comment.line - 1] = line[:col] + replacement + line[end_col:]

        code_without = "".join(lines)
        return ParseResult(
            language=language,
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without,
        )
