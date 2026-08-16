"""Tests for prompt_str utilities."""

import pytest
from unittest.mock import patch
from kimix.utils.prompt_str import (
    escape_file_paths,
    clean_text,
)


class TestEscapeFilePaths:
    """Tests for escape_file_paths with merged normalizations."""

    def test_no_paths_no_slashes(self):
        text = "hello world"
        assert escape_file_paths(text) == "hello world"

    def test_zero_width_chars_removed(self):
        text = "hello\u200bworld"
        assert escape_file_paths(text) == "helloworld"

    def test_zero_width_chars_removed_with_spaces(self):
        text = "hello \u200b world"
        # clean_text removes zwj, then remove_redundant_whitespace collapses spaces
        assert escape_file_paths(text) == "hello world"

    def test_surrogates_removed(self):
        text = "hello\ud800world"
        assert escape_file_paths(text) == "helloworld"

    def test_replacement_chars_removed(self):
        text = "hello\ufffdworld"
        assert escape_file_paths(text) == "helloworld"

    def test_pua_removed(self):
        text = "hello\ue000world"
        assert escape_file_paths(text) == "helloworld"

    def test_noncharacters_removed(self):
        text = "hello\ufdd0world"
        assert escape_file_paths(text) == "helloworld"

    def test_control_chars_removed(self):
        text = "hello\x00world"
        assert escape_file_paths(text) == "helloworld"

    def test_nfc_normalization(self):
        # e + combining acute -> é
        text = "hello\u0065\u0301world"
        assert escape_file_paths(text) == "hello\u00e9world"

    def test_dedupe_repeats(self):
        text = "A" * 200
        result = escape_file_paths(text, max_repeat=100)
        assert len(result) == 100
        assert result == "A" * 100

    def test_max_chars_truncate(self):
        text = "hello world"
        assert escape_file_paths(text, max_chars=5) == "hello"

    def test_max_chars_with_truncate_msg(self):
        text = "hello world"
        result = escape_file_paths(text, max_chars=10, truncate_msg="...")
        assert result == "hello w..."

    def test_strip_whitespace(self):
        text = "  hello world  \n"
        assert escape_file_paths(text) == "hello world"

    def test_non_string_input(self):
        assert escape_file_paths(123) == "123"

    @patch("kimix.utils.prompt_str.Path.exists", return_value=True)
    def test_escapes_real_path(self, mock_exists):
        text = "check src/kimix/utils.py for details"
        result = escape_file_paths(text)
        assert "`src/kimix/utils.py`" in result

    @patch("kimix.utils.prompt_str.Path.exists", return_value=False)
    def test_does_not_escape_nonexistent_path(self, mock_exists):
        text = "check /nonexistent/path.py for details"
        result = escape_file_paths(text)
        assert "`/nonexistent/path.py`" not in result
        assert result == "check /nonexistent/path.py for details"

    @patch("kimix.utils.prompt_str.Path.exists", return_value=True)
    def test_path_escaping_plus_sanitization(self, mock_exists):
        text = "check src/kimix/utils.py\u200b for details"
        result = escape_file_paths(text)
        assert "`src/kimix/utils.py`" in result
        assert "\u200b" not in result

    @patch("kimix.utils.prompt_str.Path.exists", return_value=True)
    def test_paths_in_quotes_unchanged(self, mock_exists):
        text = 'check "src/kimix/utils.py" for details'
        result = escape_file_paths(text)
        assert '`src/kimix/utils.py`' not in result
        assert '"src/kimix/utils.py"' in result

    def test_url_ignored(self):
        text = "visit https://example.com/path"
        assert escape_file_paths(text) == "visit https://example.com/path"

    def test_fraction_ignored(self):
        text = "the ratio is 3/4"
        assert escape_file_paths(text) == "the ratio is 3/4"

    def test_date_ignored(self):
        text = "today is 2024/01/15"
        assert escape_file_paths(text) == "today is 2024/01/15"

    def test_newlines_preserved(self):
        text = "line1\nline2"
        assert escape_file_paths(text) == "line1\nline2"

    def test_empty_string(self):
        assert escape_file_paths("") == ""

    def test_symbols_removed_by_default(self):
        text = "hello!!!??? 😀 world"
        result = escape_file_paths(text)
        assert "!!!" not in result
        assert "???" not in result
        assert "😀" not in result
        assert result == "hello! world"

    def test_encoding_normalized_by_default(self):
        text = "ＡＩ hello"
        result = escape_file_paths(text)
        assert result == "AI hello"

    def test_whitespace_collapsed_by_default(self):
        text = "hello    world\n\n\nfoo"
        result = escape_file_paths(text)
        assert result == "hello world\n\n\nfoo"

    def test_case_mode_lower(self):
        text = "Hello World"
        result = escape_file_paths(text, case_mode="lower")
        assert result == "hello world"

    def test_case_mode_title(self):
        text = "hello world"
        result = escape_file_paths(text, case_mode="title")
        assert result == "Hello World"

    def test_combined_normalization(self):
        text = "  Hello    World!!! 😀  \n\n  "
        result = escape_file_paths(text, case_mode="lower")
        assert result == "hello world!"

    @patch("kimix.utils.prompt_str.Path.exists", return_value=True)
    def test_path_escape_with_whitespace_collapsed(self, mock_exists):
        text = "1.check src/kimix/utils.py\n2.for details"
        result = escape_file_paths(text)
        assert "`src/kimix/utils.py`" in result
        assert "  " not in result

    def test_code_blocks_preserved(self):
        text = "```python\n  x = 1\n```\nhello    world"
        result = escape_file_paths(text)
        assert "```python\n  x = 1\n```" in result
        assert "hello world" in result

    def test_inline_code_preserved(self):
        text = "hello `  world  ` foo"
        result = escape_file_paths(text)
        assert result == "hello `  world  ` foo"

    def test_all_normalizations_together(self):
        text = "  HELLO    WORLD!!! 😀  \n\n  "
        result = escape_file_paths(text, case_mode="lower")
        assert result == "hello world!"

    def test_empty_string_with_case_mode(self):
        assert escape_file_paths("", case_mode="lower") == ""

    def test_non_string_with_case_mode(self):
        result = escape_file_paths(123, case_mode="title")
        assert result == "123"

    def test_fullwidth_digits_normalized(self):
        text = "price is １２３"
        result = escape_file_paths(text)
        assert result == "price is 123"

    def test_zero_width_removed_via_symbols(self):
        text = "hello\u200b\u200c world"
        result = escape_file_paths(text)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert result == "hello world"

    def test_tabs_collapsed(self):
        text = "a\t\t\tb"
        result = escape_file_paths(text)
        assert result == "a b"

    def test_case_mode_invalid(self):
        text = "Hello World"
        result = escape_file_paths(text, case_mode="invalid")
        assert result == "Hello World"

    def test_max_chars_with_merged_normalization(self):
        text = "  HELLO    WORLD  "
        result = escape_file_paths(text, case_mode="lower", max_chars=8)
        assert result == "hello wo"

    def test_max_chars_truncate_msg_with_merged(self):
        text = "  HELLO    WORLD  "
        result = escape_file_paths(
            text,
            case_mode="lower",
            max_chars=10,
            truncate_msg="...",
        )
        assert result == "hello w..."

    @patch("kimix.utils.prompt_str.Path.exists", return_value=True)
    def test_path_escape_with_all_normalizations(self, mock_exists):
        text = "Check  src/kimix/utils.py   for  details!!! 😀 "
        result = escape_file_paths(text, case_mode="lower")
        assert "`src/kimix/utils.py`" in result
        assert result == "check `src/kimix/utils.py` for details!"

    def test_code_block_preserved_with_all_normalizations(self):
        text = "```python\n  x = 1\n```\n  HELLO    WORLD!!! 😀  "
        result = escape_file_paths(text, case_mode="lower")
        assert "```python\n  x = 1\n```" in result
        assert "hello world!" in result

    def test_inline_code_preserved_with_all_normalizations(self):
        text = "`  HELLO  `   WORLD!!! 😀"
        result = escape_file_paths(text, case_mode="lower")
        assert "`  HELLO  `" in result
        assert "world!" in result

    def test_dedupe_repeats_with_whitespace_collapsed(self):
        text = "A" * 200 + "    " + "B" * 200
        result = escape_file_paths(text, max_repeat=50)
        assert result == "A" * 50 + " " + "B" * 50

    def test_fraction_ignored_after_normalization(self):
        text = "ratio 3/4 and 5/6"
        result = escape_file_paths(text)
        assert result == "ratio 3/4 and 5/6"

    def test_date_ignored_after_normalization(self):
        text = "date 2024/01/15 and 01/02/2024"
        result = escape_file_paths(text)
        assert result == "date 2024/01/15 and 01/02/2024"

    def test_chinese_chars_preserved(self):
        assert escape_file_paths("你好") == "你好"
        assert escape_file_paths("今天天气很好") == "今天天气很好"
        assert escape_file_paths("深度学习") == "深度学习"

    def test_chinese_with_emoji(self):
        # Regression: CJK chars must NOT be swallowed by _EMOJI_RE.
        text = "你好😀世界"
        result = escape_file_paths(text)
        assert "😀" not in result
        assert result == "你好世界"


class TestCleanText:
    def test_remove_zero_width(self):
        text = "a\u200bb\u200cc\u200dd\ufeffe"
        assert clean_text(text) == "abcde"

    def test_keep_newlines(self):
        text = "a\nb\tc"
        assert clean_text(text, keep_newlines=True) == "a\nb\tc"

    def test_remove_newlines(self):
        text = "a\nb\tc"
        assert clean_text(text, keep_newlines=False) == "abc"


class TestCodePlaceholders:
    """UUID-based placeholders survive normalization and restore verbatim."""

    def test_extract_restore_roundtrip(self):
        from kimix.utils.prompt_str import _extract_code, _restore_code

        text = "```python\n  x = 1\n```\n`inline` and plain"
        masked, pairs = _extract_code(text)
        assert "```" not in masked
        assert "`inline`" not in masked
        assert len(pairs) == 2
        # Tokens are unique, digit-only, and null-delimited (case-stable).
        tokens = [t for t, _ in pairs]
        assert len(set(tokens)) == len(tokens)
        assert all(t.startswith("\x00") and t.endswith("\x00") for t in tokens)
        assert all(t[1:-1].isdigit() for t in tokens)
        assert _restore_code(masked, pairs) == text

    def test_placeholders_survive_case_title(self):
        from kimix.utils.prompt_str import _extract_code, _restore_code

        text = "`hello` world"
        masked, pairs = _extract_code(text)
        titled = masked.title()
        assert _restore_code(titled, pairs) == "`hello` World"

    def test_placeholders_do_not_collide_with_counter_scheme(self):
        from kimix.utils.prompt_str import _extract_code, _restore_code

        # Multiple blocks produce distinct tokens even with the same content.
        text = "`same`\n`same`"
        masked, pairs = _extract_code(text)
        tokens = [t for t, _ in pairs]
        assert len(set(tokens)) == 2
        assert _restore_code(masked, pairs) == text

