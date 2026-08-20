"""Tests for kimi_cli.soul.fts5_search — sanitizer, CJK routing helpers, LIKE escaping."""

from __future__ import annotations

from kimi_cli.soul import fts5_search as fts


class TestSanitizeFts5Query:
    def test_preserves_quoted_phrases(self):
        q = fts.sanitize_fts5_query('"exact phrase" hello')
        assert '"exact phrase"' in q
        assert "hello" in q

    def test_unmatched_quote_becomes_space(self):
        q = fts.sanitize_fts5_query('hello "unclosed')
        assert '"' not in q
        assert "hello" in q

    def test_strips_special_chars(self):
        cases = [
            ("it's", "its"),
            ("gateway/run.py", "gateway run.py"),
            ("user@host", "user host"),
            ("a,b", "a b"),
            ("50%", "50"),
            ("TODO: fix", "TODO fix"),
        ]
        for raw, _ in cases:
            q = fts.sanitize_fts5_query(raw)
            assert q, raw
            for bad in ("'", "/", "@", ",", ":", "%"):
                assert bad not in q, (raw, q)

    def test_hyphen_and_dot_terms_wrapped_in_quotes(self):
        q = fts.sanitize_fts5_query("chat-send P2.2 my-app.config.ts")
        assert '"chat-send"' in q
        assert '"P2.2"' in q
        assert '"my-app.config.ts"' in q

    def test_cap_at_2048_chars(self):
        raw = "a" * 5000
        q = fts.sanitize_fts5_query(raw)
        assert len(q) <= 2048

    def test_empty_and_non_string(self):
        assert fts.sanitize_fts5_query("") == ""
        assert fts.sanitize_fts5_query("   ") == ""
        assert fts.sanitize_fts5_query(None) == ""  # type: ignore[arg-type]

    def test_dangling_boolean_operators_removed(self):
        q = fts.sanitize_fts5_query("hello AND")
        assert q == "hello"
        q2 = fts.sanitize_fts5_query("OR world")
        assert q2 == "world"
        q3 = fts.sanitize_fts5_query("foo NOT")
        assert q3 == "foo"


class TestCjkDetection:
    def test_contains_cjk(self):
        assert fts.contains_cjk("日本語")
        assert fts.contains_cjk("Hello 中文 world")
        assert fts.contains_cjk("한국어")
        assert not fts.contains_cjk("plain english text")
        assert not fts.contains_cjk("")

    def test_count_cjk(self):
        assert fts.count_cjk("日本語") == 3
        assert fts.count_cjk("Hello 中文") == 2
        assert fts.count_cjk("plain") == 0

    def test_has_lone_cjk_run(self):
        assert fts.has_lone_cjk_run("日")
        assert fts.has_lone_cjk_run("a日b")
        assert not fts.has_lone_cjk_run("日本語")
        assert not fts.has_lone_cjk_run("hello world")
        # A single-char run anywhere makes the query LIKE-eligible.
        assert fts.has_lone_cjk_run("日本語x中")

    def test_trigram_eligible_tokens(self):
        assert fts.trigram_eligible_tokens("日本語 テスト")
        assert fts.trigram_eligible_tokens("hello world")
        assert not fts.trigram_eligible_tokens("日")
        assert not fts.trigram_eligible_tokens("日本")
        assert not fts.trigram_eligible_tokens("hello AND 日")
        assert not fts.trigram_eligible_tokens("日本語 検索")  # 検索 is 2 chars
        assert not fts.trigram_eligible_tokens("")


class TestEscapeLike:
    def test_escapes_wildcards(self):
        assert fts.escape_like("100%") == "100\\%"
        assert fts.escape_like("a_b") == "a\\_b"
        assert fts.escape_like("a\\b") == "a\\\\b"
        assert fts.escape_like("plain") == "plain"

    def test_roundtrip_in_like(self):
        """Escaped tokens match literally when used with ESCAPE '\\'."""
        import apsw

        conn = apsw.Connection(":memory:")
        conn.execute("CREATE TABLE t (text TEXT)")
        conn.execute("INSERT INTO t VALUES ('50% done'), ('a_b'), ('other')")
        cursor = conn.execute(
            "SELECT text FROM t WHERE text LIKE ? ESCAPE '\\'",
            (f"%{fts.escape_like('50%')}%",),
        )
        rows = list(cursor)
        conn.close()
        assert rows == [("50% done",)]


class TestQuoteFtsTokens:
    def test_quotes_non_operator_tokens(self):
        q = fts.quote_fts_tokens("日本語 検索")
        assert q == '"日本語" "検索"'

    def test_preserves_boolean_operators(self):
        q = fts.quote_fts_tokens("日本 OR 中国")
        assert q == '"日本" OR "中国"'

    def test_escapes_embedded_quotes(self):
        q = fts.quote_fts_tokens('say "hi"')
        assert '""' in q  # embedded quotes doubled


class TestExtractText:
    def test_string_content(self):
        assert fts.extract_text_from_content("hello") == "hello"

    def test_text_part_object(self):
        from kimi_cli.wire.types import TextPart

        assert fts.extract_text_from_content([TextPart(text="hi"), TextPart(text="there")]) == "hi\nthere"

    def test_list_of_dicts(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert fts.extract_text_from_content(content) == "a\nb"

    def test_mixed_and_non_text_parts(self):
        content = ["plain", {"type": "image", "url": "x"}, {"type": "text", "text": "tail"}]
        assert fts.extract_text_from_content(content) == "plain\ntail"

    def test_empty(self):
        assert fts.extract_text_from_content(None) == ""
        assert fts.extract_text_from_content([]) == ""
        assert fts.extract_text_from_content({"type": "image", "url": "x"}) == ""

    def test_message_text(self):
        from kosong.message import Message

        msg = Message(role="user", content=[{"type": "text", "text": "question"}])
        assert fts.message_text(msg) == "question"
