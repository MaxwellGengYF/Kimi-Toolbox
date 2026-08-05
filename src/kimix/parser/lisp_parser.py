"""Lisp/Assembly comment parser using a state-machine approach.

Handles:
- ; line comments (Lisp, Scheme, Clojure, some assembly languages)
- #| ... |# block comments (Common Lisp standard)
- "..." string literals with escaped quotes
- Character literals (#\\;) -- semicolons inside these don't start comments
- Accurate line/column tracking for multi-line comments
"""

from __future__ import annotations

from enum import Enum, auto

from kimix.parser.base import Comment, ParseResult, BaseParser


class _State(Enum):
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
        # Native acceleration: kimix_native.parse.parse (lisp).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("lisp", "Lisp", source_code)
        if native is not None:
            return native
        
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

        state = _State.CODE

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
            if state == _State.CODE:
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
                    state = _State.BLOCK_COMMENT
                    continue

                # --- Line comment start: ; ---
                if ch == ";":
                    start_comment("line")
                    comment_chars.append(ch)
                    i += 1
                    col += 1
                    state = _State.LINE_COMMENT
                    continue

                # --- String literal ---
                if ch == '"':
                    state = _State.STRING
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
            elif state == _State.LINE_COMMENT:
                if ch == "\n":
                    finish_comment()
                    state = _State.CODE
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
            elif state == _State.BLOCK_COMMENT:
                if ch == "|" and next_ch == "#":
                    # End of block comment: include |# in content
                    comment_chars.append(ch)
                    comment_chars.append(next_ch)
                    finish_comment()
                    state = _State.CODE
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
            elif state == _State.STRING:
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
                    state = _State.CODE
                    col += 1
                    i += 1
                elif ch == "\n":
                    # Unclosed string -- back to code
                    state = _State.CODE
                    line += 1
                    col = 1
                    i += 1
                else:
                    col += 1
                    i += 1

        # --- End-of-file handling ---
        # If we ended in a comment state, finalize it
        if state == _State.LINE_COMMENT:
            finish_comment()
        elif state == _State.BLOCK_COMMENT:
            finish_comment()

        return self._build_result(self.name, source_code, comments)
