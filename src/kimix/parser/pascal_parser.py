"""Pascal/Delphi/Ada comment parser.

Handles:
- { ... } block comments (Delphi/Pascal standard)
- (* ... *) alternative block comments
- // line comments (Delphi/Object Pascal extension)
- String literals with single quotes '...'
- Doubled single quotes '' as escape inside strings
- Accurate line/column tracking for multi-line comments
"""

from __future__ import annotations

from enum import Enum, auto

from kimix.parser.base import Comment, ParseResult, BaseParser


class _State(Enum):
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
        # Native acceleration: kimix_native.parse.parse (pascal).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("pascal", "Pascal", source_code)
        if native is not None:
            return native
        
        """Parse Pascal/Delphi source code and extract comments.

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

            if state == _State.CODE:

                # --- String literal '...' ---
                if ch == "'":
                    state = _State.STRING
                    emit_char(ch)
                    i += 1
                    col += 1
                    continue

                # --- Line comment // ---
                if ch == "/" and next_ch == "/":
                    start_comment("line")
                    i += 2
                    col += 2
                    state = _State.LINE_COMMENT
                    continue

                # --- Alternative block comment (* ... *) ---
                if ch == "(" and next_ch == "*":
                    start_comment("block")
                    i += 2
                    col += 2
                    state = _State.PAREN_STAR_COMMENT
                    continue

                # --- Brace comment { ... } ---
                if ch == "{":
                    start_comment("block")
                    i += 1
                    col += 1
                    state = _State.BRACE_COMMENT
                    continue

                # --- Regular code character ---
                emit_char(ch)

                if ch == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

            elif state == _State.BRACE_COMMENT:
                if ch == "}":
                    # End of brace comment
                    finish_comment()
                    # Replace the closing } with whitespace
                    replace_comment_area(comment_start_line, comment_start_col, line, col)
                    # Emit space for the closing } itself
                    emit_char(" ")
                    state = _State.CODE
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

            elif state == _State.PAREN_STAR_COMMENT:
                if ch == "*" and next_ch == ")":
                    # End of paren-star comment
                    finish_comment()
                    # Emit whitespace for the entire comment area including closing *)
                    replace_comment_area(comment_start_line, comment_start_col, line, col)
                    emit_char(" ")
                    emit_char(" ")
                    state = _State.CODE
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

            elif state == _State.LINE_COMMENT:
                if ch == "\n":
                    finish_comment()
                    state = _State.CODE
                    # Emit the newline to preserve line count
                    emit_char(ch)
                    line += 1
                    col = 1
                    i += 1
                else:
                    comment_content.append(ch)
                    i += 1
                    col += 1

            elif state == _State.STRING:
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
                        state = _State.CODE
                        i += 1
                        col += 1
                        continue

                if ch == "\n":
                    # Unclosed string at end of line -- some Pascal dialects allow
                    # multi-line strings, but standard Pascal does not. Handle gracefully.
                    emit_char(ch)
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                i += 1
                col += 1

        # Handle unclosed comments at end of file
        if state in (_State.BRACE_COMMENT, _State.PAREN_STAR_COMMENT):
            finish_comment()
        elif state == _State.LINE_COMMENT:
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="Pascal",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )
