"""SQL comment parser for MySQL, PostgreSQL, SQLite, TiDB, etc.

Handles:
- -- line comments (SQL standard, requires trailing space or newline)
- # line comments (MySQL/TiDB style)
- /* */ block comments (multi-line, with nesting support for PostgreSQL)
- String literals ('...') with doubled single quote escapes
- Backslash escapes inside strings (MySQL NO_BACKSLASH_ESCAPES mode)
- Quoted identifiers ("..." standard SQL and `...` MySQL backtick)
"""

from __future__ import annotations

from enum import Enum, auto

from kimix.parser.base import Comment, ParseResult, BaseParser


class _State(Enum):
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
        # Native acceleration: kimix_native.parse.parse (sql).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("sql", "SQL", source_code)
        if native is not None:
            return native
        
        """Parse SQL source code and extract comments.

        Args:
            source_code: The source code string to parse.

        Returns:
            ParseResult containing extracted comments and code without comments.
        """
        comments: list[Comment] = []
        output_chars: list[str] = []

        state = _State.CODE

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

            if state == _State.CODE:
                # --- Line comment -- (SQL standard: requires space or newline after --) ---
                if ch == "-" and next_ch == "-":
                    # Check if followed by space, newline, tab, or end of file
                    after = source_code[i + 2] if i + 2 < n else "\0"
                    if after in (" ", "\t", "\n", "\r", "\0"):
                        start_comment("line")
                        i += 2
                        col += 2
                        state = _State.LINE_COMMENT_DASH
                        continue
                    # Otherwise, just two dashes in code — emit as regular chars

                # --- Line comment # (MySQL/TiDB style) ---
                if ch == "#":
                    start_comment("line")
                    i += 1
                    col += 1
                    state = _State.LINE_COMMENT_HASH
                    continue

                # --- Block comment /* */ with optional nesting ---
                if ch == "/" and next_ch == "*":
                    start_comment("block")
                    block_comment_depth = 1
                    i += 2
                    col += 2
                    state = _State.BLOCK_COMMENT
                    continue

                # --- String literal '...' ---
                if ch == "'":
                    emit_char(ch)
                    state = _State.STRING_SINGLE
                    i += 1
                    col += 1
                    continue

                # --- Quoted identifier "..." (standard SQL) ---
                if ch == '"':
                    emit_char(ch)
                    state = _State.ID_DOUBLE
                    i += 1
                    col += 1
                    continue

                # --- Quoted identifier `...` (MySQL) ---
                if ch == "`":
                    emit_char(ch)
                    state = _State.ID_BACKTICK
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

            elif state == _State.LINE_COMMENT_DASH:
                # Collect everything until newline
                if ch == "\n":
                    finish_comment()
                    state = _State.CODE
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

            elif state == _State.LINE_COMMENT_HASH:
                # Collect everything until newline
                if ch == "\n":
                    finish_comment()
                    state = _State.CODE
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

            elif state == _State.BLOCK_COMMENT:
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
                        state = _State.CODE
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

            elif state == _State.STRING_SINGLE:
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
                    state = _State.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed string at end of line — gracefully return
                if ch == "\n":
                    emit_char(ch)
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _State.ID_DOUBLE:
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
                    state = _State.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed identifier at end of line
                if ch == "\n":
                    emit_char(ch)
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _State.ID_BACKTICK:
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
                    state = _State.CODE
                    i += 1
                    col += 1
                    continue

                # Unclosed identifier at end of line
                if ch == "\n":
                    emit_char(ch)
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

        # Handle unclosed comments at end of file
        if state in (_State.LINE_COMMENT_DASH, _State.LINE_COMMENT_HASH):
            finish_comment()
        elif state == _State.BLOCK_COMMENT:
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="SQL",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )
