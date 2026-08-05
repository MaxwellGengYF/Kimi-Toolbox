"""C-family comment parser for C, C++, Java, JavaScript, TypeScript, C#, Go, Rust, etc."""

from __future__ import annotations

from enum import Enum, auto

from kimix.parser.base import Comment, ParseResult, BaseParser


class _State(Enum):
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
        # Native acceleration: kimix_native.parse.parse (c).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("c", "C", source_code)
        if native is not None:
            return native
        
        """Parse C-family source code and extract comments.

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

            if state == _State.CODE:
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
                        state = _State.STRING_DOUBLE
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
                    state = _State.LINE_COMMENT
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
                        state = _State.DOC_COMMENT
                    else:
                        # /* ... */  (block comment)
                        start_comment("block")
                        i += 2  # skip /*
                        col += 2
                        state = _State.BLOCK_COMMENT
                    continue

                # --- String literals ---
                if ch == '"':
                    state = _State.STRING_DOUBLE
                    string_delimiter = '"'
                    emit_char(ch)
                    prev_non_whitespace = '"'
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "'":
                    state = _State.STRING_SINGLE
                    string_delimiter = "'"
                    emit_char(ch)
                    prev_non_whitespace = "'"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                # --- Template literals / backtick strings ---
                if ch == "`":
                    state = _State.BACKTICK_STRING
                    emit_char(ch)
                    prev_non_whitespace = "`"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                # --- Regex literals (JavaScript/TypeScript) ---
                if ch == "/" and is_regex_start():
                    state = _State.REGEX_LITERAL
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
                    col += 1
                    i += 1

            elif state == _State.BLOCK_COMMENT:
                if ch == "*" and next_ch == "/":
                    # End of block comment
                    finish_comment()
                    state = _State.CODE
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

            elif state == _State.DOC_COMMENT:
                if ch == "*" and next_ch == "/":
                    finish_comment()
                    state = _State.CODE
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

            elif state == _State.STRING_DOUBLE:
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
                            state = _State.CODE
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
                        state = _State.CODE
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
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                    continue

                emit_char(ch)
                col += 1
                i += 1

            elif state == _State.STRING_SINGLE:
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
                    state = _State.CODE
                    prev_non_whitespace = "'"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

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

            elif state == _State.BACKTICK_STRING:
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
                    state = _State.CODE
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

            elif state == _State.REGEX_LITERAL:
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
                    state = _State.CODE
                    prev_non_whitespace = "/"
                    word_buffer.clear()
                    i += 1
                    col += 1
                    continue

                if ch == "\n":
                    # Unclosed regex - likely not a regex, exit gracefully
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
        if state == _State.LINE_COMMENT:
            finish_comment()
        elif state in (_State.BLOCK_COMMENT, _State.DOC_COMMENT):
            finish_comment()

        code_without_comments = "".join(output_chars)
        return ParseResult(
            language="C",
            comments=sorted(comments, key=lambda c: (c.line, c.column)),
            code_without_comments=code_without_comments,
        )
