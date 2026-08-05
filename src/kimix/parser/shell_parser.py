"""Shell/Bash source code comment parser using a state-machine approach.

Handles:
- # line comments
- Shebang (#!...) as doc comment
- String literals: '...', "..."
- Heredocs (<<EOF, <<'EOF', <<-EOF) — content not parsed for comments
- Backticks for command substitution
- $() for command substitution inside double-quoted strings
- Escaped characters inside double-quoted strings and backticks
"""

from __future__ import annotations

from kimix.parser.base import Comment, ParseResult, BaseParser


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
        # Native acceleration: kimix_native.parse.parse (shell).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("shell", "Shell", source_code)
        if native is not None:
            return native
        
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
