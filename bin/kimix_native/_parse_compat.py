"""Vendored pure-Python mirrors of the kimi-agent comment parsers.

Copied (with attribution) from C:/dev/kimi-agent/src/kimix/parser/ so the
shim has a bit-exact fallback when the reference checkout is not importable.
The classes are used unchanged (same algorithm, same quirks); only the import
of the base module is redirected to the definitions below.

Reference files: base.py, c_parser.py, py_parser.py, shell_parser.py,
sql_parser.py, html_parser.py, lisp_parser.py, pascal_parser.py (kimi-agent
project).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from enum import Enum, auto




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







class _CState(Enum):
    """States for the C-family comment parser state machine."""

    CODE = auto()
    LINE_COMMENT = auto()
    BLOCK_COMMENT = auto()
    DOC_COMMENT = auto()
    STRING_DOUBLE = auto()
    STRING_SINGLE = auto()
    BACKTICK_STRING = auto()
    REGEX_LITERAL = auto()


# Characters after which a '/' likely starts a regex literal (JavaScript/TypeScript)
_REGEX_PRECEDING_CHARS: set[str] = {
    "=",
    "(",
    "[",
    "!",
    "&",
    "|",
    ",",
    ";",
    "{",
    ":",
    "?",
    "~",
    "^",
    "*",
    "-",
    "+",
    "%",
    "<",
    ">",
    "/",
}

# Keywords after which a '/' likely starts a regex literal
_REGEX_KEYWORDS: set[str] = {
    "return",
    "case",
    "typeof",
    "instanceof",
    "void",
    "delete",
    "throw",
    "new",
    "in",
    "of",
    "yield",
    "await",
    "else",
}


class CParser(BaseParser):
    """Parse C-family source code and extract comments.

    Handles:
    - // line comments
    - /* */ block comments (multi-line)
    - /** */ doc comments (kind="doc")
    - String literals ("...", '...')
    - Template literals/backtick strings (`...`)
    - Regex literals (JavaScript /.../)
    - Raw string literals (Rust r#"..."#, Go `...`)
    - Escaped characters inside strings
    - Accurate line/column tracking
    """

    name = "C"
    description = (
        "Parse C-family source code and extract comments "
        "(// line, /* */ block, /** */ doc comments)."
    )

    def parse(self, source_code: str) -> ParseResult:  # noqa: C901
        """Parse C-family source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        comments: list[Comment] = []
        output_chars: list[str] = []

        state = _CState.CODE

        # Comment tracking
        comment_start_line = 0
        comment_start_col = 0
        comment_content: list[str] = []
        comment_kind: str = "line"

        # String literal tracking
        string_delimiter: str = '"'

        # Rust raw string tracking
        raw_hash_count = 0  # number of # in r#"..."#
        in_raw_string = False

        # Regex detection
        prev_non_whitespace: str | None = None
        word_buffer: list[str] = []

        # Position tracking
        line = 1
        col = 1

        i = 0
        n = len(source_code)

        def start_comment(kind: str) -> None:
            """Begin collecting a comment at the current position."""
            nonlocal comment_start_line, comment_start_col, comment_kind
            comment_start_line = line
            comment_start_col = col
            comment_kind = kind
            comment_content.clear()

        def finish_comment() -> None:
            """Finalize the current comment and add it to the list."""
            content = "".join(comment_content)
            comments.append(
                Comment(
                    content=content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind=comment_kind,
                )
            )

        def emit_char(ch: str) -> None:
            """Emit a character to the output (code without comments)."""
            output_chars.append(ch)

        def replace_comment_with_whitespace(end_line: int, end_col: int) -> None:
            """Emit spaces/newlines to preserve positions for a comment.

            Args:
                end_line: The line the comment ends on (1-based).
                end_col: The column the comment ends at (1-based).
            """
            # Emit spaces and newlines from comment start to comment end
            # to preserve line/column positions in code_without_comments
            current_line = comment_start_line
            current_col = comment_start_col

            while current_line < end_line:
                # Remainder of current line
                emit_char(" ")
                # Move to next line
                emit_char("\n")
                current_line += 1
                current_col = 1

            # On the final line, emit spaces up to end_col
            while current_col < end_col:
                emit_char(" ")
                current_col += 1

        def is_regex_start() -> bool:
            """Determine if a '/' at the current position starts a regex literal."""
            if prev_non_whitespace is None:
                return True
            word = "".join(word_buffer)
            if word in _REGEX_KEYWORDS:
                return True
            if prev_non_whitespace in _REGEX_PRECEDING_CHARS:
                return True
            return False

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else "\0"

            if state == _CState.CODE:
                # --- Check for raw string literals (Rust r#"..."#) ---
                if (
                    ch == "r"
                    and next_ch == "#"
                    and not in_raw_string
                ):
                    # Count the number of #
                    hash_count = 0
                    j = i + 1
                    while j < n and source_code[j] == "#":
                        hash_count += 1
                        j += 1
                    if j < n and source_code[j] == '"':
                        # This is a Rust raw string literal r#"..."#
                        raw_hash_count = hash_count
                        in_raw_string = True
                        state = _CState.STRING_DOUBLE
                        # Emit r and all # and opening "
                        for k in range(i, j + 1):
                            emit_char(source_code[k])
                        # Update col: advanced past r + hash_count #'s + "
                        col += (1 + hash_count + 1)
                        i = j + 1
                        prev_non_whitespace = '"'
                        word_buffer.clear()
                        continue

                # --- Line comment // ---
                if ch == "/" and next_ch == "/":
                    start_comment("line")
                    i += 2
                    col += 2
                    state = _CState.LINE_COMMENT
                    continue

                # --- Block comment /* or doc comment /** ---
                if ch == "/" and next_ch == "*":
                    if (
                        i + 2 < n
                        and source_code[i + 2] == "*"
                        and not (i + 3 < n and source_code[i + 3] == "/")
                    ):
                        # /** ... */  (doc comment, but not /**/)
                        start_comment("doc")
                        i += 3  # skip /**
                        col += 3
                        state = _CState.DOC_COMMENT
                    else:
                        # /* ... */  (block comment)
                        start_comment("block")
                        i += 2  # skip /*
                        col += 2
                        state = _CState.BLOCK_COMMENT
                    continue

                # --- String literals ---
                if ch == '"':
                    state = _CState.STRING_DOUBLE
                    string_delimiter = '"'
                    emit_char(ch)
                    prev_non_whitespace = '"'
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "'":
                    state = _CState.STRING_SINGLE
                    string_delimiter = "'"
                    emit_char(ch)
                    prev_non_whitespace = "'"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                # --- Template literals / backtick strings ---
                if ch == "`":
                    state = _CState.BACKTICK_STRING
                    emit_char(ch)
                    prev_non_whitespace = "`"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                # --- Regex literals (JavaScript/TypeScript) ---
                if ch == "/" and is_regex_start():
                    state = _CState.REGEX_LITERAL
                    emit_char(ch)
                    prev_non_whitespace = "/"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                # --- Regular code character ---
                if not ch.isspace():
                    prev_non_whitespace = ch
                    if ch.isalnum() or ch == "_":
                        word_buffer.append(ch)
                    else:
                        word_buffer.clear()
                emit_char(ch)

                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

            elif state == _CState.LINE_COMMENT:
                if ch == "\n":
                    finish_comment()
                    state = _CState.CODE
                    # Emit the newline to preserve line count
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    col += 1
                    i += 1

            elif state == _CState.BLOCK_COMMENT:
                if ch == "*" and next_ch == "/":
                    # End of block comment
                    finish_comment()
                    state = _CState.CODE
                    # Replace the closing */ with spaces
                    emit_char(" ")
                    emit_char(" ")
                    i += 2
                    col += 2
                elif ch == "\n":
                    comment_content.append(ch)
                    emit_char(ch)  # preserve newline in output
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    emit_char(" ")  # replace with space
                    col += 1
                    i += 1

            elif state == _CState.DOC_COMMENT:
                if ch == "*" and next_ch == "/":
                    finish_comment()
                    state = _CState.CODE
                    emit_char(" ")
                    emit_char(" ")
                    i += 2
                    col += 2
                elif ch == "\n":
                    comment_content.append(ch)
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    emit_char(" ")
                    col += 1
                    i += 1

            elif state == _CState.STRING_DOUBLE:
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                if ch == '"':
                    if in_raw_string:
                        # Check for closing Rust raw string: " followed by #*hash_count
                        j = i + 1
                        found_hashes = 0
                        while j < n and source_code[j] == "#":
                            found_hashes += 1
                            j += 1
                        if found_hashes == raw_hash_count:
                            # Close raw string
                            emit_char('"')
                            for _ in range(raw_hash_count):
                                emit_char("#")
                            in_raw_string = False
                            state = _CState.CODE
                            prev_non_whitespace = '"'
                            word_buffer.clear()
                            i = j
                            col += (1 + raw_hash_count)
                            continue
                        else:
                            # Not the closing delimiter, emit the " and continue
                            emit_char(ch)
                            col += 1
                            i += 1
                            continue
                    else:
                        emit_char(ch)
                        state = _CState.CODE
                        prev_non_whitespace = '"'
                        word_buffer.clear()
                        i += 1
                        col += 1
                        continue

                if ch == "\n":
                    # Unclosed string - handle gracefully
                    emit_char(ch)
                    if in_raw_string:
                        in_raw_string = False
                    state = _CState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _CState.STRING_SINGLE:
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                if ch == "'":
                    emit_char(ch)
                    state = _CState.CODE
                    prev_non_whitespace = "'"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "\n":
                    emit_char(ch)
                    state = _CState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _CState.BACKTICK_STRING:
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                if ch == "`":
                    emit_char(ch)
                    state = _CState.CODE
                    prev_non_whitespace = "`"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "\n":
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _CState.REGEX_LITERAL:
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                if ch == "/":
                    emit_char(ch)
                    state = _CState.CODE
                    prev_non_whitespace = "/"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "\n":
                    # Unclosed regex - likely not a regex, exit gracefully
                    emit_char(ch)
                    state = _CState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

        # Handle unclosed comments at end of file
        if state == _CState.LINE_COMMENT:
            finish_comment()
        elif state in (_CState.BLOCK_COMMENT, _CState.DOC_COMMENT):
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="C",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )


u"""Python source code comment parser using a state-machine approach.

Handles:
- # line comments (not inside strings or f-string expressions)
- triple-quoted \"\"\"...\"\"\" and '''...''' docstrings
- Single/double quoted strings with prefixes (r, b, f, rb, br, rf, fr)
- Escaped quotes inside strings
- f-string interpolation {..} (does not treat # inside them as comments)
"""




class PythonParser(BaseParser):
    """Parse Python source code and extract comments (# line comments, triple-quoted docstrings)."""

    name = "Python"
    description = (
        "Parse Python source code and extract comments "
        "(# line comments, triple-quoted docstrings)."
    )

    # --- State constants ---
    CODE = 0
    LINE_COMMENT = 1
    STRING_SINGLE = 2
    STRING_DOUBLE = 3
    STRING_TRIPLE_SINGLE = 4
    STRING_TRIPLE_DOUBLE = 5
    FSTRING_EXPR = 6

    _PREFIX_CHARS = frozenset("rRbBfF")
    _QUOTE_CHARS = frozenset("\"'")

    def parse(self, source_code: str) -> ParseResult:
        """Parse Python source and extract all comments and docstrings."""
        comments: list[Comment] = []

        state = self.CODE
        i = 0
        n = len(source_code)
        line = 1
        col = 1

        # Accumulators
        comment_start_line = 0
        comment_start_col = 0
        comment_chars: list[str] = []

        string_start_line = 0
        string_start_col = 0
        string_chars: list[str] = []
        string_quote_char: str = ""
        string_is_fstring: bool = False
        string_has_prefix: bool = False

        # Escape tracking for current string
        escape: bool = False

        # F-string expression brace depth
        fstring_depth: int = 0

        # For FSTRING_EXPR: re-use tracking
        fexpr_string_quote: str = ""
        fexpr_string_escape: bool = False
        fexpr_parent_state: int = 0  # state to return to after fstring expr

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else ""

            # ================================================================
            # STATE: CODE
            # ================================================================
            if state == self.CODE:
                if ch == "#":
                    state = self.LINE_COMMENT
                    comment_start_line = line
                    comment_start_col = col
                    comment_chars = ["#"]
                    i += 1
                    col += 1

                elif ch in self._PREFIX_CHARS:
                    # Scan for full prefix (e.g. "rb", "fr")
                    j = i + 1
                    while j < n and source_code[j] in self._PREFIX_CHARS:
                        j += 1
                    if j < n and source_code[j] in self._QUOTE_CHARS:
                        prefix = source_code[i:j]
                        quote = source_code[j]
                        is_f = "f" in prefix.lower()
                        # Check for triple quotes
                        if (
                            j + 2 < n
                            and source_code[j] == quote
                            and source_code[j + 1] == quote
                            and source_code[j + 2] == quote
                        ):
                            # Triple-quoted string with prefix
                            if quote == "'":
                                state = self.STRING_TRIPLE_SINGLE
                            else:
                                state = self.STRING_TRIPLE_DOUBLE
                            string_start_line = line
                            string_start_col = col
                            string_chars = [source_code[i : j + 3]]
                            string_quote_char = quote
                            string_is_fstring = is_f
                            string_has_prefix = True
                            consumed = j + 3 - i
                            i = j + 3
                            col += consumed
                        else:
                            # Single/double quoted string with prefix
                            if quote == "'":
                                state = self.STRING_SINGLE
                            else:
                                state = self.STRING_DOUBLE
                            string_start_line = line
                            string_start_col = col
                            string_chars = [source_code[i : j + 1]]
                            string_quote_char = quote
                            string_is_fstring = is_f
                            string_has_prefix = True
                            escape = False
                            consumed = j + 1 - i
                            i = j + 1
                            col += consumed
                    else:
                        # Not a string prefix, treat as regular code
                        col += 1
                        i += 1

                elif ch == "'":
                    # Check for triple single quotes
                    if (
                        i + 2 < n
                        and source_code[i + 1] == "'"
                        and source_code[i + 2] == "'"
                    ):
                        state = self.STRING_TRIPLE_SINGLE
                        string_start_line = line
                        string_start_col = col
                        string_chars = ["'''"]
                        string_quote_char = "'"
                        string_is_fstring = False
                        string_has_prefix = False
                        i += 3
                        col += 3
                    else:
                        state = self.STRING_SINGLE
                        string_start_line = line
                        string_start_col = col
                        string_chars = ["'"]
                        string_quote_char = "'"
                        string_is_fstring = False
                        string_has_prefix = False
                        escape = False
                        i += 1
                        col += 1

                elif ch == '"':
                    # Check for triple double quotes
                    if (
                        i + 2 < n
                        and source_code[i + 1] == '"'
                        and source_code[i + 2] == '"'
                    ):
                        state = self.STRING_TRIPLE_DOUBLE
                        string_start_line = line
                        string_start_col = col
                        string_chars = ['"""']
                        string_quote_char = '"'
                        string_is_fstring = False
                        string_has_prefix = False
                        i += 3
                        col += 3
                    else:
                        state = self.STRING_DOUBLE
                        string_start_line = line
                        string_start_col = col
                        string_chars = ['"']
                        string_quote_char = '"'
                        string_is_fstring = False
                        string_has_prefix = False
                        escape = False
                        i += 1
                        col += 1

                elif ch == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # ================================================================
            # STATE: LINE_COMMENT
            # ================================================================
            elif state == self.LINE_COMMENT:
                if ch == "\n":
                    comment_content = "".join(comment_chars)
                    comments.append(
                        Comment(
                            content=comment_content,
                            line=comment_start_line,
                            column=comment_start_col,
                            kind="line",
                        )
                    )
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_SINGLE
            # ================================================================
            elif state == self.STRING_SINGLE:
                if escape:
                    string_chars.append(ch)
                    escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_chars.append(ch)
                    escape = True
                    col += 1
                    i += 1
                elif ch == "'":
                    string_chars.append(ch)
                    state = self.CODE
                    col += 1
                    i += 1
                elif ch == "\n":
                    string_chars.append(ch)
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                elif ch == "{" and string_is_fstring:
                    if next_ch == "{":
                        string_chars.append("{{")
                        i += 2
                        col += 2
                    else:
                        string_chars.append(ch)
                        state = self.FSTRING_EXPR
                        fstring_depth = 0
                        fexpr_string_quote = ""
                        fexpr_string_escape = False
                        fexpr_parent_state = self.STRING_SINGLE
                        col += 1
                        i += 1
                else:
                    string_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_DOUBLE
            # ================================================================
            elif state == self.STRING_DOUBLE:
                if escape:
                    string_chars.append(ch)
                    escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_chars.append(ch)
                    escape = True
                    col += 1
                    i += 1
                elif ch == '"':
                    string_chars.append(ch)
                    state = self.CODE
                    col += 1
                    i += 1
                elif ch == "\n":
                    string_chars.append(ch)
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                elif ch == "{" and string_is_fstring:
                    if next_ch == "{":
                        string_chars.append("{{")
                        i += 2
                        col += 2
                    else:
                        string_chars.append(ch)
                        state = self.FSTRING_EXPR
                        fstring_depth = 0
                        fexpr_string_quote = ""
                        fexpr_string_escape = False
                        fexpr_parent_state = self.STRING_DOUBLE
                        col += 1
                        i += 1
                else:
                    string_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_TRIPLE_SINGLE
            # ================================================================
            elif state == self.STRING_TRIPLE_SINGLE:
                if escape:
                    string_chars.append(ch)
                    escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_chars.append(ch)
                    escape = True
                    col += 1
                    i += 1
                elif ch == "{" and string_is_fstring:
                    if next_ch == "{":
                        string_chars.append("{{")
                        i += 2
                        col += 2
                    else:
                        string_chars.append(ch)
                        state = self.FSTRING_EXPR
                        fstring_depth = 0
                        fexpr_string_quote = ""
                        fexpr_string_escape = False
                        fexpr_parent_state = self.STRING_TRIPLE_SINGLE
                        col += 1
                        i += 1
                elif ch == "'":
                    string_chars.append(ch)
                    if next_ch == "'" and i + 2 < n and source_code[i + 2] == "'":
                        string_chars.append("''")
                        if not string_has_prefix:
                            comment_content = "".join(string_chars)
                            comments.append(
                                Comment(
                                    content=comment_content,
                                    line=string_start_line,
                                    column=string_start_col,
                                    kind="doc",
                                )
                            )
                        state = self.CODE
                        i += 3
                        col += 3
                    else:
                        col += 1
                        i += 1
                elif ch == "\n":
                    string_chars.append(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    string_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_TRIPLE_DOUBLE
            # ================================================================
            elif state == self.STRING_TRIPLE_DOUBLE:
                if escape:
                    string_chars.append(ch)
                    escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_chars.append(ch)
                    escape = True
                    col += 1
                    i += 1
                elif ch == "{" and string_is_fstring:
                    if next_ch == "{":
                        string_chars.append("{{")
                        i += 2
                        col += 2
                    else:
                        string_chars.append(ch)
                        state = self.FSTRING_EXPR
                        fstring_depth = 0
                        fexpr_string_quote = ""
                        fexpr_string_escape = False
                        fexpr_parent_state = self.STRING_TRIPLE_DOUBLE
                        col += 1
                        i += 1
                elif ch == '"':
                    string_chars.append(ch)
                    if next_ch == '"' and i + 2 < n and source_code[i + 2] == '"':
                        string_chars.append('""')
                        if not string_has_prefix:
                            comment_content = "".join(string_chars)
                            comments.append(
                                Comment(
                                    content=comment_content,
                                    line=string_start_line,
                                    column=string_start_col,
                                    kind="doc",
                                )
                            )
                        state = self.CODE
                        i += 3
                        col += 3
                    else:
                        col += 1
                        i += 1
                elif ch == "\n":
                    string_chars.append(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    string_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: FSTRING_EXPR
            # ================================================================
            elif state == self.FSTRING_EXPR:
                if fexpr_string_escape:
                    fexpr_string_escape = False
                    col += 1
                    i += 1
                elif fexpr_string_quote:
                    if ch == "\\":
                        fexpr_string_escape = True
                        col += 1
                        i += 1
                    elif ch == fexpr_string_quote:
                        fexpr_string_quote = ""
                        col += 1
                        i += 1
                    elif ch == "\n":
                        line += 1
                        col = 1
                        i += 1
                    else:
                        col += 1
                        i += 1
                elif ch in self._PREFIX_CHARS:
                    # Could be string prefix inside fstring expr
                    j = i + 1
                    while j < n and source_code[j] in self._PREFIX_CHARS:
                        j += 1
                    if j < n and source_code[j] in self._QUOTE_CHARS:
                        quote = source_code[j]
                        if (
                            j + 2 < n
                            and source_code[j] == quote
                            and source_code[j + 1] == quote
                            and source_code[j + 2] == quote
                        ):
                            i = j + 3
                            col += 1
                        else:
                            fexpr_string_quote = quote
                            i = j + 1
                            col += 1
                    else:
                        col += 1
                        i += 1
                elif ch in self._QUOTE_CHARS:
                    if (
                        i + 2 < n
                        and source_code[i + 1] == ch
                        and source_code[i + 2] == ch
                    ):
                        i += 3
                        col += 3
                    else:
                        fexpr_string_quote = ch
                        i += 1
                        col += 1
                elif ch == "{":
                    fstring_depth += 1
                    col += 1
                    i += 1
                elif ch == "}":
                    if fstring_depth == 0:
                        state = fexpr_parent_state
                        string_chars.append(ch)
                        col += 1
                        i += 1
                    else:
                        fstring_depth -= 1
                        col += 1
                        i += 1
                elif ch == "#":
                    col += 1
                    i += 1
                elif ch == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

        # --- End of file handling ---

        if state == self.LINE_COMMENT:
            comment_content = "".join(comment_chars)
            comments.append(
                Comment(
                    content=comment_content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind="line",
                )
            )

        # Emit docstring for unclosed triple-quoted strings (only if no prefix)
        if state in (self.STRING_TRIPLE_SINGLE, self.STRING_TRIPLE_DOUBLE) and not string_has_prefix:
            comment_content = "".join(string_chars)
            comments.append(
                Comment(
                    content=comment_content,
                    line=string_start_line,
                    column=string_start_col,
                    kind="doc",
                )
            )

        return self._build_result("Python", source_code, comments)






class ShellParser(BaseParser):
    """Parse Shell/Bash source code and extract comments (# line comments)."""

    name = "Shell"
    description = "Parse shell script source code and extract comments (# line comments)."

    # --- State constants ---
    CODE = 0
    LINE_COMMENT = 1
    STRING_SINGLE = 2
    STRING_DOUBLE = 3
    HEREDOC = 4
    BACKTICK = 5
    DOLLAR_PAREN = 6

    def parse(self, source_code: str) -> ParseResult:
        """Parse Shell source and extract all comments."""
        comments: list[Comment] = []
        i = 0
        n = len(source_code)
        line = 1
        col = 1

        state = self.CODE

        # Line comment accumulators
        comment_start_line = 0
        comment_start_col = 0
        comment_chars: list[str] = []
        comment_kind: str = "line"

        # String escape tracking (used for STRING_DOUBLE and BACKTICK)
        string_escape: bool = False

        # Heredoc tracking
        heredoc_delimiter: str = ""
        heredoc_allow_tab: bool = False
        heredoc_line: str = ""
        heredoc_return_state: int = self.CODE

        # Dollar-paren tracking
        dp_depth: int = 0
        dp_return_state: int = self.CODE

        # Backtick return state
        bt_return_state: int = self.CODE

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else ""

            # ================================================================
            # STATE: CODE
            # ================================================================
            if state == self.CODE:
                if ch == "#":
                    state = self.LINE_COMMENT
                    comment_start_line = line
                    comment_start_col = col
                    if line == 1 and next_ch == "!":
                        comment_kind = "doc"
                    else:
                        comment_kind = "line"
                    comment_chars = [ch]
                    i += 1
                    col += 1

                elif ch == "'":
                    state = self.STRING_SINGLE
                    i += 1
                    col += 1

                elif ch == '"':
                    state = self.STRING_DOUBLE
                    string_escape = False
                    i += 1
                    col += 1

                elif ch == "`":
                    state = self.BACKTICK
                    bt_return_state = self.CODE
                    string_escape = False
                    i += 1
                    col += 1

                elif ch == "$" and next_ch == "(":
                    state = self.DOLLAR_PAREN
                    dp_depth = 1
                    dp_return_state = self.CODE
                    i += 2
                    col += 2

                elif ch == "<" and next_ch == "<":
                    # Heredoc start: <<EOF, <<'EOF', <<-EOF, <<-"EOF"
                    # Check for <<< (here-string, not heredoc)
                    if i + 2 < n and source_code[i + 2] == "<":
                        col += 3
                        i += 3
                    else:
                        j = i + 2
                        allow_tab = False
                        if j < n and source_code[j] == "-":
                            allow_tab = True
                            j += 1
                        # Skip whitespace
                        while j < n and source_code[j] in " \t":
                            j += 1
                        # Extract delimiter
                        delimiter = ""
                        if j < n and source_code[j] in ("'", '"'):
                            q = source_code[j]
                            j += 1
                            while j < n and source_code[j] != q:
                                delimiter += source_code[j]
                                j += 1
                            if j < n:
                                j += 1  # skip closing quote
                        else:
                            while j < n and source_code[j] not in " \t\n":
                                delimiter += source_code[j]
                                j += 1

                        if delimiter:
                            state = self.HEREDOC
                            heredoc_delimiter = delimiter
                            heredoc_allow_tab = allow_tab
                            heredoc_line = ""
                            heredoc_return_state = self.CODE
                            consumed = j - i
                            i = j
                            col += consumed
                        else:
                            col += 1
                            i += 1

                elif ch == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # ================================================================
            # STATE: LINE_COMMENT
            # ================================================================
            elif state == self.LINE_COMMENT:
                if ch == "\n":
                    comment_content = "".join(comment_chars)
                    comments.append(
                        Comment(
                            content=comment_content,
                            line=comment_start_line,
                            column=comment_start_col,
                            kind=comment_kind,
                        )
                    )
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_SINGLE
            # In single-quoted strings, everything is literal (no escape processing)
            # ================================================================
            elif state == self.STRING_SINGLE:
                if ch == "'":
                    state = self.CODE
                    col += 1
                    i += 1
                elif ch == "\n":
                    # Unclosed string — back to code
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING_DOUBLE
            # Double-quoted strings: escape sequences, $(), backticks
            # ================================================================
            elif state == self.STRING_DOUBLE:
                if string_escape:
                    string_escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_escape = True
                    col += 1
                    i += 1
                elif ch == '"':
                    state = self.CODE
                    col += 1
                    i += 1
                elif ch == "$" and next_ch == "(":
                    # Command substitution inside double quotes
                    state = self.DOLLAR_PAREN
                    dp_depth = 1
                    dp_return_state = self.STRING_DOUBLE
                    i += 2
                    col += 2
                elif ch == "`":
                    # Backtick inside double quotes
                    state = self.BACKTICK
                    bt_return_state = self.STRING_DOUBLE
                    string_escape = False
                    i += 1
                    col += 1
                elif ch == "\n":
                    # Unclosed string
                    state = self.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # ================================================================
            # STATE: HEREDOC
            # Inside heredoc: only look for the delimiter on its own line.
            # Content is NOT parsed for comments.
            # ================================================================
            elif state == self.HEREDOC:
                if ch == "\n":
                    # Check if the accumulated line matches the delimiter
                    stripped = heredoc_line
                    if heredoc_allow_tab:
                        stripped = stripped.lstrip("\t")
                    if stripped == heredoc_delimiter:
                        state = heredoc_return_state
                    heredoc_line = ""
                    line += 1
                    col = 1
                    i += 1
                else:
                    heredoc_line += ch
                    col += 1
                    i += 1

            # ================================================================
            # STATE: BACKTICK
            # Backtick command substitution: parse like CODE but exit on ``
            # Escaped characters: \\, \`, \$, \"
            # ================================================================
            elif state == self.BACKTICK:
                if string_escape:
                    string_escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_escape = True
                    col += 1
                    i += 1
                elif ch == "`":
                    state = bt_return_state
                    col += 1
                    i += 1
                elif ch == "#":
                    state = self.LINE_COMMENT
                    comment_start_line = line
                    comment_start_col = col
                    comment_kind = "line"
                    comment_chars = [ch]
                    i += 1
                    col += 1
                elif ch == "'":
                    state = self.STRING_SINGLE
                    i += 1
                    col += 1
                elif ch == '"':
                    state = self.STRING_DOUBLE
                    string_escape = False
                    i += 1
                    col += 1
                elif ch == "$" and next_ch == "(":
                    state = self.DOLLAR_PAREN
                    dp_depth = 1
                    dp_return_state = self.BACKTICK
                    i += 2
                    col += 2
                elif ch == "<" and next_ch == "<":
                    # Heredoc inside backtick
                    if i + 2 < n and source_code[i + 2] == "<":
                        col += 3
                        i += 3
                    else:
                        j = i + 2
                        allow_tab = False
                        if j < n and source_code[j] == "-":
                            allow_tab = True
                            j += 1
                        while j < n and source_code[j] in " \t":
                            j += 1
                        delimiter = ""
                        if j < n and source_code[j] in ("'", '"'):
                            q = source_code[j]
                            j += 1
                            while j < n and source_code[j] != q:
                                delimiter += source_code[j]
                                j += 1
                            if j < n:
                                j += 1
                        else:
                            while j < n and source_code[j] not in " \t\n":
                                delimiter += source_code[j]
                                j += 1
                        if delimiter:
                            state = self.HEREDOC
                            heredoc_delimiter = delimiter
                            heredoc_allow_tab = allow_tab
                            heredoc_line = ""
                            heredoc_return_state = self.BACKTICK
                            consumed = j - i
                            i = j
                            col += consumed
                        else:
                            col += 1
                            i += 1
                elif ch == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # ================================================================
            # STATE: DOLLAR_PAREN
            # $() command substitution: parse like CODE, track paren nesting.
            # ================================================================
            elif state == self.DOLLAR_PAREN:
                if ch == "(":
                    dp_depth += 1
                    col += 1
                    i += 1
                elif ch == ")":
                    dp_depth -= 1
                    if dp_depth == 0:
                        state = dp_return_state
                    col += 1
                    i += 1
                elif ch == "#":
                    state = self.LINE_COMMENT
                    comment_start_line = line
                    comment_start_col = col
                    comment_kind = "line"
                    comment_chars = [ch]
                    i += 1
                    col += 1
                elif ch == "'":
                    state = self.STRING_SINGLE
                    i += 1
                    col += 1
                elif ch == '"':
                    state = self.STRING_DOUBLE
                    string_escape = False
                    i += 1
                    col += 1
                elif ch == "`":
                    state = self.BACKTICK
                    bt_return_state = self.DOLLAR_PAREN
                    string_escape = False
                    i += 1
                    col += 1
                elif ch == "$" and next_ch == "(":
                    dp_depth += 1
                    i += 2
                    col += 2
                elif ch == "<" and next_ch == "<":
                    # Heredoc inside $()
                    if i + 2 < n and source_code[i + 2] == "<":
                        col += 3
                        i += 3
                    else:
                        j = i + 2
                        allow_tab = False
                        if j < n and source_code[j] == "-":
                            allow_tab = True
                            j += 1
                        while j < n and source_code[j] in " \t":
                            j += 1
                        delimiter = ""
                        if j < n and source_code[j] in ("'", '"'):
                            q = source_code[j]
                            j += 1
                            while j < n and source_code[j] != q:
                                delimiter += source_code[j]
                                j += 1
                            if j < n:
                                j += 1
                        else:
                            while j < n and source_code[j] not in " \t\n":
                                delimiter += source_code[j]
                                j += 1
                        if delimiter:
                            state = self.HEREDOC
                            heredoc_delimiter = delimiter
                            heredoc_allow_tab = allow_tab
                            heredoc_line = ""
                            heredoc_return_state = self.DOLLAR_PAREN
                            consumed = j - i
                            i = j
                            col += consumed
                        else:
                            col += 1
                            i += 1
                elif ch == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

            # end of state switch

        # --- End-of-file handling ---

        # If we ended in LINE_COMMENT state, emit the final comment
        if state == self.LINE_COMMENT:
            comment_content = "".join(comment_chars)
            comments.append(
                Comment(
                    content=comment_content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind=comment_kind,
                )
            )

        # If we ended in HEREDOC state with accumulated line matching delimiter,
        # that counts as closing the heredoc — but at EOF there's no trailing \n.
        # Check if the accumulated heredoc_line matches the delimiter.
        if state == self.HEREDOC:
            stripped = heredoc_line
            if heredoc_allow_tab:
                stripped = stripped.lstrip("\t")
            if stripped == heredoc_delimiter:
                # Delimiter found at EOF — properly closed
                pass  # state becomes irrelevant at EOF

        return self._build_result("Shell", source_code, comments)







class _SqlState(Enum):
    """States for the SQL comment parser state machine."""

    CODE = auto()
    LINE_COMMENT_DASH = auto()
    LINE_COMMENT_HASH = auto()
    BLOCK_COMMENT = auto()
    STRING_SINGLE = auto()
    ID_DOUBLE = auto()
    ID_BACKTICK = auto()


class SqlParser(BaseParser):
    """Parse SQL source code and extract comments.

    Handles:
    - ``--`` line comments (SQL standard, requires trailing space or newline)
    - ``#`` line comments (MySQL/TiDB style)
    - ``/* */`` block comments (multi-line, with nesting support)
    - String literals (``'...'``) with doubled single quote escapes
    - Backslash escapes inside strings (MySQL NO_BACKSLASH_ESCAPES mode)
    - Quoted identifiers (``"..."`` standard SQL and backtick ``\\`...\\``` MySQL)
    - Accurate line/column tracking
    """

    name = "SQL"
    description = (
        "Parse SQL source code and extract comments "
        "(-- line, /* */ block, # line comments)."
    )

    def parse(self, source_code: str) -> ParseResult:  # noqa: C901
        """Parse SQL source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        comments: list[Comment] = []
        output_chars: list[str] = []

        state = _SqlState.CODE

        # Comment tracking
        comment_start_line = 0
        comment_start_col = 0
        comment_content: list[str] = []
        comment_kind: str = "line"
        block_comment_depth = 0

        # Position tracking
        line = 1
        col = 1

        i = 0
        n = len(source_code)

        def start_comment(kind: str) -> None:
            """Begin collecting a comment at the current position."""
            nonlocal comment_start_line, comment_start_col, comment_kind
            comment_start_line = line
            comment_start_col = col
            comment_kind = kind
            comment_content.clear()

        def finish_comment() -> None:
            """Finalize the current comment and add it to the list."""
            content = "".join(comment_content)
            comments.append(
                Comment(
                    content=content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind=comment_kind,
                )
            )

        def emit_char(ch: str) -> None:
            """Emit a character to the output (code without comments)."""
            output_chars.append(ch)

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else "\0"

            if state == _SqlState.CODE:
                # --- Line comment -- (SQL standard: requires space or newline after --) ---
                if ch == "-" and next_ch == "-":
                    # Check if followed by space, newline, tab, or end of file
                    after = source_code[i + 2] if i + 2 < n else "\0"
                    if after in (" ", "\t", "\n", "\r", "\0"):
                        start_comment("line")
                        i += 2
                        col += 2
                        state = _SqlState.LINE_COMMENT_DASH
                        continue
                    # Otherwise, just two dashes in code — emit as regular chars

                # --- Line comment # (MySQL/TiDB style) ---
                if ch == "#":
                    start_comment("line")
                    i += 1
                    col += 1
                    state = _SqlState.LINE_COMMENT_HASH
                    continue

                # --- Block comment /* */ with optional nesting ---
                if ch == "/" and next_ch == "*":
                    start_comment("block")
                    block_comment_depth = 1
                    i += 2
                    col += 2
                    state = _SqlState.BLOCK_COMMENT
                    continue

                # --- String literal '...' ---
                if ch == "'":
                    emit_char(ch)
                    state = _SqlState.STRING_SINGLE
                    i += 1
                    col += 1
                    continue

                # --- Quoted identifier "..." (standard SQL) ---
                if ch == '"':
                    emit_char(ch)
                    state = _SqlState.ID_DOUBLE
                    i += 1
                    col += 1
                    continue

                # --- Quoted identifier `...` (MySQL) ---
                if ch == "`":
                    emit_char(ch)
                    state = _SqlState.ID_BACKTICK
                    i += 1
                    col += 1
                    continue

                # --- Regular code character ---
                emit_char(ch)
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

            elif state == _SqlState.LINE_COMMENT_DASH:
                # Collect everything until newline
                if ch == "\n":
                    finish_comment()
                    state = _SqlState.CODE
                    emit_char(ch)  # preserve newline positions
                    line += 1
                    col = 1
                    i += 1
                elif ch == "\r":
                    # Carriage return — could be \r\n on Windows
                    comment_content.append(ch)
                    i += 1
                    col += 1
                else:
                    comment_content.append(ch)
                    col += 1
                    i += 1

            elif state == _SqlState.LINE_COMMENT_HASH:
                # Collect everything until newline
                if ch == "\n":
                    finish_comment()
                    state = _SqlState.CODE
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                elif ch == "\r":
                    comment_content.append(ch)
                    i += 1
                    col += 1
                else:
                    comment_content.append(ch)
                    col += 1
                    i += 1

            elif state == _SqlState.BLOCK_COMMENT:
                # Handle nested block comments (PostgreSQL style)
                if ch == "/" and next_ch == "*":
                    block_comment_depth += 1
                    comment_content.append(ch)
                    comment_content.append(next_ch)
                    i += 2
                    col += 2
                    continue

                if ch == "*" and next_ch == "/":
                    block_comment_depth -= 1
                    if block_comment_depth == 0:
                        # End of top-level block comment
                        finish_comment()
                        state = _SqlState.CODE
                        emit_char(" ")  # replace closing */
                        emit_char(" ")
                        i += 2
                        col += 2
                    else:
                        # Inner nesting close — keep in comment content
                        comment_content.append(ch)
                        comment_content.append(next_ch)
                        i += 2
                        col += 2
                    continue

                if ch == "\n":
                    comment_content.append(ch)
                    emit_char(ch)  # preserve newline for line counting
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    emit_char(" ")  # replace with space to preserve position
                    col += 1
                    i += 1

            elif state == _SqlState.STRING_SINGLE:
                # Handle doubled single quote '' as escape
                if ch == "'" and next_ch == "'":
                    emit_char(ch)
                    emit_char(next_ch)
                    i += 2
                    col += 2
                    continue

                # Handle backslash escape (MySQL NO_BACKSLASH_ESCAPES mode)
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                # End of string literal
                if ch == "'":
                    emit_char(ch)
                    state = _SqlState.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed string at end of line — gracefully return
                if ch == "\n":
                    emit_char(ch)
                    state = _SqlState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _SqlState.ID_DOUBLE:
                # Handle doubled double-quote "" as escape
                if ch == '"' and next_ch == '"':
                    emit_char(ch)
                    emit_char(next_ch)
                    i += 2
                    col += 2
                    continue

                # Handle backslash escape
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                # End of quoted identifier
                if ch == '"':
                    emit_char(ch)
                    state = _SqlState.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed identifier at end of line
                if ch == "\n":
                    emit_char(ch)
                    state = _SqlState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _SqlState.ID_BACKTICK:
                # Handle doubled backtick `` as escape (MySQL)
                if ch == "`" and next_ch == "`":
                    emit_char(ch)
                    emit_char(next_ch)
                    i += 2
                    col += 2
                    continue

                # Handle backslash escape
                if ch == "\\":
                    emit_char(ch)
                    i += 1
                    col += 1
                    if i < n:
                        ch2 = source_code[i]
                        emit_char(ch2)
                        i += 1
                        col += 1
                    continue

                # End of backtick identifier
                if ch == "`":
                    emit_char(ch)
                    state = _SqlState.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed identifier at end of line
                if ch == "\n":
                    emit_char(ch)
                    state = _SqlState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

        # Handle unclosed comments at end of file
        if state in (_SqlState.LINE_COMMENT_DASH, _SqlState.LINE_COMMENT_HASH):
            finish_comment()
        elif state == _SqlState.BLOCK_COMMENT:
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="SQL",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )






class HtmlParser(BaseParser):
    """Parser for extracting comments from HTML/XML source code.

    Handles:
    - ``<!-- ... -->`` block comments (can span multiple lines)
    - ``<? ... ?>`` processing instructions (e.g., ``<?xml version="1.0"?>``)
    - ``<![CDATA[ ... ]]>`` sections (contents are *not* comments, skipped)
    - Quoted attribute values (``"..."`` and ``'...'``) — ``<!--`` inside
      them is *not* treated as a comment start.
    """

    name = "HTML"
    description = "Parse HTML/XML source code and extract comments (<!-- -->)."

    # Internal state identifiers
    _CODE = 0
    _COMMENT = 1
    _PI = 2
    _CDATA = 3
    _ATTR_DOUBLE = 4
    _ATTR_SINGLE = 5

    def parse(self, source_code: str) -> ParseResult:
        """Parse *source_code* and return extracted comments.

        Args:
            source_code: HTML/XML source string.

        Returns:
            A :class:`ParseResult` with discovered comments and code stripped
            of comments/processing-instructions.
        """
        comments: list[Comment] = []
        # Character-index ranges in the original source that should be
        # replaced with whitespace in *code_without_comments*.
        replace_ranges: list[tuple[int, int]] = []

        state = self._CODE

        line = 1
        col = 1

        # --- comment/PI tracking ---
        start_line = 0
        start_col = 0
        start_idx = 0          # index of '<' that started the construct
        content_chars: list[str] = []

        i = 0
        n = len(source_code)

        while i < n:
            ch = source_code[i]

            # ------------------------------------------------------------------
            #  CODE  (base state)
            # ------------------------------------------------------------------
            if state == self._CODE:

                # ----------  <!--  comment  ----------
                if ch == "<" and i + 3 < n and source_code[i : i + 4] == "<!--":
                    state = self._COMMENT
                    start_line = line
                    start_col = col
                    start_idx = i
                    content_chars = []
                    i += 4
                    col += 4
                    continue

                # ----------  <?  processing instruction  ----------
                if ch == "<" and i + 1 < n and source_code[i : i + 2] == "<?":
                    state = self._PI
                    start_line = line
                    start_col = col
                    start_idx = i
                    content_chars = []
                    i += 2
                    col += 2
                    continue

                # ----------  <![CDATA[  section  ----------
                if ch == "<" and i + 8 < n and source_code[i : i + 9] == "<![CDATA[":
                    state = self._CDATA
                    i += 9
                    col += 9
                    continue

                # ----------  double-quoted attribute  ----------
                if ch == '"':
                    state = self._ATTR_DOUBLE
                    i += 1
                    col += 1
                    continue

                # ----------  single-quoted attribute  ----------
                if ch == "'":
                    state = self._ATTR_SINGLE
                    i += 1
                    col += 1
                    continue

                # normal character
                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

            # ------------------------------------------------------------------
            #  COMMENT  (inside <!-- ... -->)
            # ------------------------------------------------------------------
            if state == self._COMMENT:
                if ch == "-" and i + 2 < n and source_code[i : i + 3] == "-->":
                    content = "".join(content_chars)
                    comments.append(
                        Comment(
                            content=content,
                            line=start_line,
                            column=start_col,
                            kind="block",
                        )
                    )
                    # Replace the whole <!-- ... --> with whitespace
                    replace_ranges.append((start_idx, i + 3))
                    state = self._CODE
                    i += 3
                    col += 3
                    continue

                content_chars.append(ch)
                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

            # ------------------------------------------------------------------
            #  PI  (processing instruction <? ... ?>)
            # ------------------------------------------------------------------
            if state == self._PI:
                if ch == "?" and i + 1 < n and source_code[i : i + 2] == "?>":
                    content = "".join(content_chars)
                    comments.append(
                        Comment(
                            content=content,
                            line=start_line,
                            column=start_col,
                            kind="doc",
                        )
                    )
                    replace_ranges.append((start_idx, i + 2))
                    state = self._CODE
                    i += 2
                    col += 2
                    continue

                content_chars.append(ch)
                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

            # ------------------------------------------------------------------
            #  CDATA  (inside <![CDATA[ ... ]]> — skip)
            # ------------------------------------------------------------------
            if state == self._CDATA:
                if ch == "]" and i + 2 < n and source_code[i : i + 3] == "]]>":
                    state = self._CODE
                    i += 3
                    col += 3
                    continue

                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

            # ------------------------------------------------------------------
            #  ATTR_DOUBLE  (inside "...")
            # ------------------------------------------------------------------
            if state == self._ATTR_DOUBLE:
                if ch == '"':
                    state = self._CODE
                    i += 1
                    col += 1
                    continue

                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

            # ------------------------------------------------------------------
            #  ATTR_SINGLE  (inside '...')
            # ------------------------------------------------------------------
            if state == self._ATTR_SINGLE:
                if ch == "'":
                    state = self._CODE
                    i += 1
                    col += 1
                    continue

                i += 1
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                continue

        # ------------------------------------------------------------------
        #  Build *code_without_comments*: replace comment/PI ranges with
        #  whitespace to preserve original line/column positions.
        # ------------------------------------------------------------------
        result_chars = list(source_code)
        for r_start, r_end in sorted(replace_ranges, reverse=True):
            for j in range(r_start, r_end):
                if result_chars[j] != "\n":
                    result_chars[j] = " "
        code_without = "".join(result_chars)

        return ParseResult(
            language=self.name,
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without,
        )







class _LispState(Enum):
    """States for the Lisp comment parser state machine."""

    CODE = auto()
    LINE_COMMENT = auto()
    BLOCK_COMMENT = auto()
    STRING = auto()


class LispParser(BaseParser):
    """Parse Lisp/Assembly source code and extract comments.

    Handles:
    - ; line comments (Lisp, Scheme, Clojure, assembly)
    - #| ... |# block comments (multi-line, Common Lisp standard)
    - String literals ("...") with escaped quotes
    - Character literals (#\\;)
    - Accurate line/column tracking for multi-line comments
    """

    name = "Lisp"
    description = (
        "Parse Lisp/Assembly source code and extract comments "
        "(; line comments, #| |# block comments)."
    )

    def parse(self, source_code: str) -> ParseResult:  # noqa: C901
        """Parse Lisp/Assembly source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        comments: list[Comment] = []
        i = 0
        n = len(source_code)
        line = 1
        col = 1

        state = _LispState.CODE

        # Comment tracking
        comment_start_line = 0
        comment_start_col = 0
        comment_chars: list[str] = []
        comment_kind: str = "line"

        # String escape tracking
        string_escape: bool = False

        def start_comment(kind: str) -> None:
            """Begin collecting a comment at the current position."""
            nonlocal comment_start_line, comment_start_col, comment_kind
            comment_start_line = line
            comment_start_col = col
            comment_kind = kind
            comment_chars.clear()

        def finish_comment() -> None:
            """Finalize the current comment and add it to the list."""
            content = "".join(comment_chars)
            comments.append(
                Comment(
                    content=content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind=comment_kind,
                )
            )

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else "\0"

            # ================================================================
            # STATE: CODE
            # ================================================================
            if state == _LispState.CODE:
                # --- Character literals (#\X) ---
                # #\; is a character literal for semicolon, NOT a comment start.
                # #\ followed by any character or name (e.g., #\Space, #\Newline).
                if ch == "#" and next_ch == "\\":
                    i += 2  # skip #\
                    col += 2
                    if i < n:
                        # Consume the character after #\
                        # For named chars like #\Space, consume the name
                        ch_after = source_code[i]
                        i += 1
                        col += 1
                        # If alphabetic, consume more for names like #\Space, #\Newline
                        if ch_after.isalpha():
                            while i < n and source_code[i].isalpha():
                                i += 1
                                col += 1
                    continue

                # --- Block comment start: #| ---
                if ch == "#" and next_ch == "|":
                    start_comment("block")
                    # Include #| in the comment content
                    comment_chars.append(ch)
                    comment_chars.append(next_ch)
                    i += 2
                    col += 2
                    state = _LispState.BLOCK_COMMENT
                    continue

                # --- Line comment start: ; ---
                if ch == ";":
                    start_comment("line")
                    comment_chars.append(ch)
                    i += 1
                    col += 1
                    state = _LispState.LINE_COMMENT
                    continue

                # --- String literal ---
                if ch == '"':
                    state = _LispState.STRING
                    string_escape = False
                    i += 1
                    col += 1
                    continue

                # --- Regular character ---
                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

            # ================================================================
            # STATE: LINE_COMMENT
            # ================================================================
            elif state == _LispState.LINE_COMMENT:
                if ch == "\n":
                    finish_comment()
                    state = _LispState.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: BLOCK_COMMENT
            # ================================================================
            elif state == _LispState.BLOCK_COMMENT:
                if ch == "|" and next_ch == "#":
                    # End of block comment: include |# in content
                    comment_chars.append(ch)
                    comment_chars.append(next_ch)
                    finish_comment()
                    state = _LispState.CODE
                    i += 2
                    col += 2
                elif ch == "\n":
                    comment_chars.append(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_chars.append(ch)
                    col += 1
                    i += 1

            # ================================================================
            # STATE: STRING
            # ================================================================
            elif state == _LispState.STRING:
                if string_escape:
                    # Consume escaped character
                    string_escape = False
                    col += 1
                    i += 1
                elif ch == "\\":
                    string_escape = True
                    col += 1
                    i += 1
                elif ch == '"':
                    # End of string
                    state = _LispState.CODE
                    col += 1
                    i += 1
                elif ch == "\n":
                    # Unclosed string -- back to code
                    state = _LispState.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

        # --- End-of-file handling ---
        # If we ended in a comment state, finalize it
        if state == _LispState.LINE_COMMENT:
            finish_comment()
        elif state == _LispState.BLOCK_COMMENT:
            finish_comment()

        return self._build_result(self.name, source_code, comments)







class _PascalState(Enum):
    """States for the Pascal comment parser state machine."""

    CODE = auto()
    BRACE_COMMENT = auto()
    PAREN_STAR_COMMENT = auto()
    LINE_COMMENT = auto()
    STRING = auto()


class PascalParser(BaseParser):
    """Parse Pascal/Delphi source code and extract comments.

    Handles:
    - { ... } block comments (Delphi/Pascal standard)
    - (* ... *) alternative block comments
    - // line comments (Delphi/Object Pascal extension)
    - String literals with single quotes '...'
    - Doubled single quotes inside strings as escapes ('' => literal ')
    - Nested { inside (* ... *) (non-standard but supported by some compilers)
    """

    name = "Pascal"
    description = (
        "Parse Pascal/Delphi source code and extract comments "
        "({ }, (* *), //)."
    )

    def parse(self, source_code: str) -> ParseResult:  # noqa: C901
        """Parse Pascal/Delphi source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        comments: list[Comment] = []
        output_chars: list[str] = []

        state = _PascalState.CODE

        # Comment tracking
        comment_start_line = 0
        comment_start_col = 0
        comment_content: list[str] = []
        comment_kind: str = "block"

        # Position tracking
        line = 1
        col = 1

        i = 0
        n = len(source_code)

        def start_comment(kind: str) -> None:
            """Begin collecting a comment at the current position."""
            nonlocal comment_start_line, comment_start_col, comment_kind
            comment_start_line = line
            comment_start_col = col
            comment_kind = kind
            comment_content.clear()

        def finish_comment() -> None:
            """Finalize the current comment and add it to the list."""
            content = "".join(comment_content)
            comments.append(
                Comment(
                    content=content,
                    line=comment_start_line,
                    column=comment_start_col,
                    kind=comment_kind,
                )
            )

        def emit_char(ch: str) -> None:
            """Emit a character to the output (code without comments)."""
            output_chars.append(ch)

        def replace_comment_area(start_line: int, start_col: int, end_line: int, end_col: int) -> None:
            """Emit whitespace to preserve positions from comment start to comment end.

            Args:
                start_line: The line the comment starts on (1-based).
                start_col: The column the comment starts at (1-based).
                end_line: The line the comment ends on (1-based).
                end_col: The column the comment ends at (1-based).
            """
            current_line = start_line
            current_col = start_col

            while current_line < end_line:
                emit_char(" ")
                emit_char("\n")
                current_line += 1
                current_col = 1

            while current_col < end_col:
                emit_char(" ")
                current_col += 1

        while i < n:
            ch = source_code[i]
            next_ch = source_code[i + 1] if i + 1 < n else "\0"

            if state == _PascalState.CODE:

                # --- String literal '...' ---
                if ch == "'":
                    state = _PascalState.STRING
                    emit_char(ch)
                    i += 1
                    col += 1
                    continue

                # --- Line comment // ---
                if ch == "/" and next_ch == "/":
                    start_comment("line")
                    i += 2
                    col += 2
                    state = _PascalState.LINE_COMMENT
                    continue

                # --- Alternative block comment (* ... *) ---
                if ch == "(" and next_ch == "*":
                    start_comment("block")
                    i += 2
                    col += 2
                    state = _PascalState.PAREN_STAR_COMMENT
                    continue

                # --- Brace comment { ... } ---
                if ch == "{":
                    start_comment("block")
                    i += 1
                    col += 1
                    state = _PascalState.BRACE_COMMENT
                    continue

                # --- Regular code character ---
                emit_char(ch)

                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

            elif state == _PascalState.BRACE_COMMENT:
                if ch == "}":
                    # End of brace comment
                    finish_comment()
                    # Replace the closing } with whitespace
                    replace_comment_area(comment_start_line, comment_start_col, line, col)
                    # Emit space for the closing } itself
                    emit_char(" ")
                    state = _PascalState.CODE
                    i += 1
                    col += 1
                elif ch == "\n":
                    comment_content.append(ch)
                    emit_char(ch)  # preserve newline in output
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    i += 1
                    col += 1

            elif state == _PascalState.PAREN_STAR_COMMENT:
                if ch == "*" and next_ch == ")":
                    # End of paren-star comment
                    finish_comment()
                    # Emit whitespace for the entire comment area including closing *)
                    replace_comment_area(comment_start_line, comment_start_col, line, col)
                    emit_char(" ")
                    emit_char(" ")
                    state = _PascalState.CODE
                    i += 2
                    col += 2
                elif ch == "\n":
                    comment_content.append(ch)
                    emit_char(ch)  # preserve newline in output
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    i += 1
                    col += 1

            elif state == _PascalState.LINE_COMMENT:
                if ch == "\n":
                    finish_comment()
                    state = _PascalState.CODE
                    # Emit the newline to preserve line count
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    i += 1
                    col += 1

            elif state == _PascalState.STRING:
                if ch == "'":
                    # Could be doubled '' (escape) or end of string
                    if next_ch == "'":
                        # Doubled single quote -- literal quote inside string
                        emit_char(ch)
                        emit_char(next_ch)
                        i += 2
                        col += 2
                        continue
                    else:
                        # End of string literal
                        emit_char(ch)
                        state = _PascalState.CODE
                        i += 1
                        col += 1
                        continue

                if ch == "\n":
                    # Unclosed string at end of line -- some Pascal dialects allow
                    # multi-line strings, but standard Pascal does not. Handle gracefully.
                    emit_char(ch)
                    state = _PascalState.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                i += 1
                col += 1

        # Handle unclosed comments at end of file
        if state in (_PascalState.BRACE_COMMENT, _PascalState.PAREN_STAR_COMMENT):
            finish_comment()
        elif state == _PascalState.LINE_COMMENT:
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="Pascal",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )


