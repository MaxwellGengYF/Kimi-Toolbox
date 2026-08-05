u"""Python source code comment parser using a state-machine approach.

Handles:
- # line comments (not inside strings or f-string expressions)
- triple-quoted \"\"\"...\"\"\" and '''...''' docstrings
- Single/double quoted strings with prefixes (r, b, f, rb, br, rf, fr)
- Escaped quotes inside strings
- f-string interpolation {..} (does not treat # inside them as comments)
"""

from __future__ import annotations

from kimix.parser.base import Comment, ParseResult, BaseParser


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
        # Native acceleration: kimix_native.parse.parse (python).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("python", "Python", source_code)
        if native is not None:
            return native
        
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
