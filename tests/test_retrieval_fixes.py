"""Comprehensive regression tests for kimix.retrieval fixes.

Covers all issues described in retrieval.md:
1. Doc-ID vs Array-Index mismatch
2. _dcg hard-coded 18 ranks
3. LambdaMART.fit coordinate ascent without baseline
4. stop_threshold=0.0 prunes all terms
5. BM25 IDF uses candidate-filtered DF
6. query_scope TypeError on unfinalized index
7. min_should_match truncation
8. Duplicate doc_id appends instead of overwriting
9. Incomplete CJK Unicode ranges
10. Native-endian serialization
11. Truncated forward-index chunk
12. Dead code _ensure_term_to_id removed
13. Redundant _posting_docs/_posting_tfs removed
14. Symmetric-Delete index not persisted
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kimix.retrieval import (
    BM25Scorer,
    InvertedIndex,
    LambdaMART,
    NgramTokenizer,
    QueryPerformancePredictor,
    Searcher,
    _dcg,
    _ndcg,
)

# ---------------------------------------------------------------------------
# Issue #1 – Doc-ID / array-index mismatch
# ---------------------------------------------------------------------------


class TestDocIdArrayIndexMismatch:
    def test_non_sequential_doc_ids(self) -> None:
        """BM25 scorer must not crash with non-sequential doc IDs."""
        idx = InvertedIndex()
        idx.add_document(5, ["ab", "bc", "cd"])
        idx.add_document(10, ["ab", "ef"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        scores = scorer.score(["ab"])
        assert 5 in scores
        assert 10 in scores
        assert scores[5] > 0.0
        assert scores[10] > 0.0

    def test_doc_lengths_arr_padded(self) -> None:
        """_doc_lengths_arr must be padded to max_doc_id + 1."""
        idx = InvertedIndex()
        idx.add_document(3, ["x", "y", "z"])
        idx.finalize()
        assert len(idx.doc_lengths_arr) == 4
        assert idx.doc_lengths_arr[3] == 3
        assert idx.doc_lengths_arr[0] == 0
        assert idx.doc_lengths_arr[1] == 0
        assert idx.doc_lengths_arr[2] == 0


# ---------------------------------------------------------------------------
# Issue #2 – _dcg crashes for lists > 18 elements
# ---------------------------------------------------------------------------


class TestDcgHardCodedRanks:
    def test_dcg_25_elements(self) -> None:
        """_dcg must handle arbitrary-length lists."""
        scores = [1.0] * 25
        result = _dcg(scores)
        assert result > 0.0
        assert math.isfinite(result)

    def test_dcg_100_elements(self) -> None:
        scores = list(range(100))
        result = _dcg(scores)
        assert result > 0.0
        assert math.isfinite(result)

    def test_ndcg_perfect_ranking(self) -> None:
        scores = [3, 2, 1, 0]
        assert _ndcg(scores) == 1.0


# ---------------------------------------------------------------------------
# Issue #3 – LambdaMART.fit coordinate ascent without baseline
# ---------------------------------------------------------------------------


class TestLambdaMARTBaseline:
    def test_fit_does_not_degrade_ndcg(self) -> None:
        """Coordinate ascent should only apply steps that improve NDCG."""
        # Two queries, each with 4 docs and 1 feature
        X = [
            [[1.0], [0.5], [0.2], [0.1]],
            [[0.9], [0.4], [0.3], [0.0]],
        ]
        y = [
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
        lm = LambdaMART(n_iterations=10, learning_rate=0.1)
        lm.fit(X, y)
        assert len(lm.weights) == 1

        # Compute NDCG before and after (weights start at 0)
        baseline_ndcg = 0.0
        trained_ndcg = 0.0
        for xi, yi in zip(X, y):
            baseline_ndcg += _ndcg(np.dot(xi, [0.0]))
            trained_ndcg += _ndcg(np.dot(xi, lm.weights))

        # Trained NDCG should not be worse than baseline
        assert trained_ndcg >= baseline_ndcg - 1e-9


# ---------------------------------------------------------------------------
# Issue #4 – stop_threshold=0.0 prunes all terms
# ---------------------------------------------------------------------------


class TestStopThresholdZero:
    def test_stop_threshold_zero_keeps_terms(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.finalize(stop_threshold=0.0)
        assert idx.has_term("aa")
        assert idx.has_term("bb")

    def test_single_doc_corpus_positive_threshold(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.finalize(stop_threshold=0.5)
        # Every term appears in 100% of docs, but with threshold=0.5 they are pruned.
        # With the fix, threshold=0.0 disables pruning, but 0.5 still prunes.
        # The key point is that 0.0 does NOT prune everything.
        assert not idx.has_term("aa") or not idx.has_term("bb")


# ---------------------------------------------------------------------------
# Issue #5 – BM25 IDF uses candidate-filtered DF
# ---------------------------------------------------------------------------


class TestBM25GlobalDf:
    def test_idf_uses_global_df(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["tok"] * 10)
        idx.add_document(1, ["tok"] * 10)
        idx.add_document(2, ["tok"] * 10)
        idx.add_document(3, ["other"] * 10)
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)

        # Score with candidate_docs restricting to doc 0 only.
        # The IDF should still use global df=3, not filtered df=1.
        scores_all = scorer.score(["tok"])
        scores_filtered = scorer.score(["tok"], candidate_docs={0})

        # Since all docs have identical tf, the relative score should be the same
        # (same IDF, same tf, same denom). The filtered score for doc 0 should
        # equal the unfiltered score for doc 0.
        assert scores_filtered[0] == pytest.approx(scores_all[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Issue #6 – query_scope TypeError on unfinalized index
# ---------------------------------------------------------------------------


class TestQueryScopeUnfinalized:
    def test_query_scope_no_typeerror(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["ab", "bc"])
        qpp = QueryPerformancePredictor(idx, BM25Scorer(idx))
        # Before the fix this raised TypeError: 'NoneType' object is not subscriptable
        scope = qpp.query_scope(["ab"])
        assert 0.0 <= scope <= 1.0


# ---------------------------------------------------------------------------
# Issue #7 – min_should_match truncates instead of ceiling
# ---------------------------------------------------------------------------


class TestMinShouldMatchCeiling:
    def test_min_should_match_ceil(self) -> None:
        idx = InvertedIndex()
        for i in range(5):
            idx.add_document(i, ["abc", "bcd", "cde"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, min_should_match=0.6, fuzziness=0)
        # Query "abcde" tokenizes (n=3) to ["abc", "bcd", "cde"]
        # With min_should_match=0.6 -> ceil(3*0.6)=2
        # All three tokens are in the index -> hits=3 >= 2
        results = searcher.search("abcde", top_k=10)
        assert len(results) > 0

        # Query "abcf" tokenizes to ["abc", "bcf"]
        # fuzziness=0, so "bcf" does not fuzzy-match anything.
        # min_match=ceil(2*0.6)=2, only "abc" matches -> no results
        results = searcher.search("abcf", top_k=10)
        assert results == []


# ---------------------------------------------------------------------------
# Issue #8 – Duplicate doc_id appends instead of overwriting
# ---------------------------------------------------------------------------


class TestDuplicateDocId:
    def test_duplicate_doc_id_overwrites(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.add_document(0, ["aa", "bb"])
        idx.add_document(0, ["cc", "dd"])
        idx.finalize(stop_threshold=1.0)
        assert idx.doc_freq("cc") == 1
        assert idx.doc_freq("dd") == 1
        scorer = BM25Scorer(idx)
        scores = scorer.score(["cc"])
        # With 1 doc, length=2, avgdl=2, idf=log(1+0.5/1.5)=0.28768
        # denom_base = 1.2 * (0.25 + 0.75/2 * 2) = 1.2
        # score = 1 * idf * 2.2 / (1 + 1.2) = idf
        assert 0 in scores
        assert scores[0] == pytest.approx(scorer._idf(1, 1), rel=1e-4)


# ---------------------------------------------------------------------------
# Issue #9 – Incomplete CJK Unicode ranges
# ---------------------------------------------------------------------------


class TestCJKRanges:
    def test_cjk_compatibility_ideographs(self) -> None:
        t = NgramTokenizer()
        # U+F900 is CJK Compatibility Ideograph
        assert t._is_cjk("\uF900")

    def test_cjk_compatibility_supplement(self) -> None:
        t = NgramTokenizer()
        # U+2F800 is CJK Compatibility Ideographs Supplement
        assert t._is_cjk("\U0002F800")

    def test_cjk_extension_g(self) -> None:
        t = NgramTokenizer()
        assert t._is_cjk("\U00030000")

    def test_hangul_jamo(self) -> None:
        t = NgramTokenizer()
        assert t._is_cjk("\u1100")

    def test_cjk_strokes(self) -> None:
        t = NgramTokenizer()
        assert t._is_cjk("\u31C0")


# ---------------------------------------------------------------------------
# Issue #10 – Native-endian serialization
# ---------------------------------------------------------------------------


class TestEndianness:
    def test_save_load_endianness(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb", "cc"])
        idx.add_document(1, ["aa", "dd"])
        idx.finalize()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = InvertedIndex()
            idx2.load(path)
            for term in idx.terms():
                docs1, tfs1 = idx.get_postings(term)
                docs2, tfs2 = idx2.get_postings(term)
                np.testing.assert_array_equal(docs1, docs2)
                np.testing.assert_array_equal(tfs1, tfs2)
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Issue #11 – Truncated forward-index chunk handled gracefully
# ---------------------------------------------------------------------------


class TestTruncatedForwardChunk:
    def test_truncated_forward_chunk_falls_back(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.add_document(1, ["aa", "cc"])
        idx.finalize(stop_threshold=1.0)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            idx.save(path, include_forward_index=True)
            # Truncate the file by a few bytes to corrupt the forward chunk
            data = Path(path).read_bytes()
            Path(path).write_bytes(data[:-5])
            idx2 = InvertedIndex()
            # Must not crash; falls back to rebuilding from postings
            idx2.load(path)
            assert idx2.has_term("aa")
            assert idx2.has_term("bb")
            assert idx2.has_term("cc")
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Issue #12 – Dead code _ensure_term_to_id removed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #13 – Redundant _posting_docs / _posting_tfs removed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #14 – Symmetric-Delete index persisted
# ---------------------------------------------------------------------------


class TestSymmetricDeletePersisted:
    def test_sd_index_roundtrip(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(1, ["hello", "test"])
        idx.finalize()
        idx._build_symmetric_delete_index()
        assert idx._symmetric_delete_index

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = InvertedIndex()
            idx2.load(path)
            assert idx2._symmetric_delete_index
            assert 1 in idx2._symmetric_delete_index
            assert 2 in idx2._symmetric_delete_index
        finally:
            Path(path).unlink(missing_ok=True)
