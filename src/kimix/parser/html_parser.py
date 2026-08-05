"""HTML/XML comment parser using a state-machine approach."""

from __future__ import annotations

from kimix.parser.base import Comment, ParseResult, BaseParser


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
        # Native acceleration: kimix_native.parse.parse (html).
        from kimix.parser.base import native_parse_result
        native = native_parse_result("html", "HTML", source_code)
        if native is not None:
            return native
        
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
