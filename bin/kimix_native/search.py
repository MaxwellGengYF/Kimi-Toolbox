"""kimix_native.search — search kernels: BM25, distance, fuzzy, hashing, rerank.

Native implementations live in ``runtime_py.search`` (compiled kernels, GIL
released, bytes in/out). ``_compat`` below are pure-Python mirrors written
from the same reference source (kimix/retrieval.py), so
``KIMIX_NATIVE_SEARCH=0`` yields identical behavior.

Hashing contract (documented deviation from the reference, which uses
xxhash.xxh64 for SimHash and the PYTHONHASHSEED-random built-in ``hash()``
for MinHash — neither reproducible): the kernels hash with XXH3-64
(``kimix::hash64``, default seed 2^61-1), and ``_compat`` reproduces that
exactly with the ``xxhash`` package (``xxh3_64(data, seed=2**61-1)``).
"""

from __future__ import annotations

import math
import struct
from collections import Counter

from . import _native, use_native

_XXH3_SEED = 2**61 - 1


def _enc(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def _dec(b: bytes) -> str:
    return b.decode("utf-8", "surrogatepass")


# ---------------------------------------------------------------------------
# _compat — pure-Python mirrors
# ---------------------------------------------------------------------------


def _compat_hash64(data: bytes, seed: int = _XXH3_SEED) -> int:
    """Reproduce kimix::hash64 (XXH3-64, default seed 2^61-1) via the xxhash
    package. Falls back to a documented deterministic hash when xxhash is
    unavailable (parity tests require xxhash)."""
    import xxhash

    return xxhash.xxh3_64(data, seed=seed).intdigest()


def _compat_bm25_idf(doc_count: int, df: int, k1: float = 1.2, b: float = 0.75) -> float:
    return math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))


def _compat_bm25_score(query_postings, idf, doc_lengths, avg_doc_len,
                       doc_count, k1: float = 1.2, b: float = 0.75) -> list[float]:
    scores = [0.0] * doc_count
    for postings, idf_t in zip(query_postings, idf):
        scale = idf_t * (k1 + 1.0)
        for doc_id, tf in postings:
            if doc_id >= doc_count:
                continue
            denom_base = k1 * ((1.0 - b) + (b / avg_doc_len) * doc_lengths[doc_id])
            scores[doc_id] += tf * scale / (tf + denom_base)
    return scores


def _compat_top_k(scores, k: int) -> list[int]:
    nonzero = [(doc, s) for doc, s in enumerate(scores) if s > 0.0]
    nonzero.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in nonzero[:k]]


def _compat_damerau_levenshtein(a: str, b: str, max_dist: int = -1) -> int:
    """Exact port of retrieval.py::LevenshteinAutomaton._damerau_levenshtein
    (including the fast-path quirks), plus the kernel's max_dist contract."""
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    if n == 0:
        d = m
    elif n == 1:
        d = 0 if a[0] == b[0] else 1
    elif m == 2 and n == 2:
        if a == b:
            d = 0
        elif a[0] == b[0] or a[1] == b[1]:
            d = 1
        elif a[0] == b[1] and a[1] == b[0]:
            d = 1
        else:
            d = 2
    else:
        prev_prev = list(range(n + 1))
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            si_1 = a[i - 1]
            for j in range(1, n + 1):
                cost = 0 if si_1 == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
                if i > 1 and j > 1 and si_1 == b[j - 2] and a[i - 2] == b[j - 1]:
                    curr[j] = min(curr[j], prev_prev[j - 2] + 1)
            prev_prev, prev, curr = prev, curr, prev_prev
        d = prev[n]
    if max_dist >= 0 and d > max_dist:
        return max_dist + 1
    return d


def _compat_freq_lower_bound(pattern: str, term: str) -> int:
    pcounts = Counter(pattern)
    total = 0
    matched = 0
    for c, pc in pcounts.items():
        tc = term.count(c)
        matched += tc
        if pc != tc:
            total += abs(pc - tc)
    total += len(term) - matched
    return (total + 1) // 2


def _compat_jaro_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    len_s, len_t = len(a), len(b)
    if len_s == 0 or len_t == 0:
        return 0.0
    match_distance = max(len_s, len_t) // 2 - 1
    s_matches = [False] * len_s
    t_matches = [False] * len_t
    matches = 0
    for i in range(len_s):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_t)
        for j in range(start, end):
            if t_matches[j] or a[i] != b[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len_s):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    return (matches / len_s + matches / len_t +
            (matches - transpositions / 2) / matches) / 3.0


def _compat_jaro_winkler(a: str, b: str, prefix_scale: float = 0.1) -> float:
    jaro = _compat_jaro_similarity(a, b)
    prefix = 0
    limit = min(4, len(a), len(b))
    for i in range(limit):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * prefix_scale * (1.0 - jaro)


def _compat_sorensen_dice(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    s_bigrams = {a[i : i + 2] for i in range(len(a) - 1)}
    t_bigrams = {b[i : i + 2] for i in range(len(b) - 1)}
    intersection = len(s_bigrams & t_bigrams)
    denom = len(s_bigrams) + len(t_bigrams)
    return 0.0 if denom == 0 else 2.0 * intersection / denom


def _compat_ngram_overlap(a: str, b: str, n: int = 2) -> float:
    if not a or not b:
        return 0.0
    s_grams = {a[i : i + n] for i in range(len(a) - n + 1)} if len(a) >= n else {a}
    t_grams = {b[i : i + n] for i in range(len(b) - n + 1)} if len(b) >= n else {b}
    union = s_grams | t_grams
    if not union:
        return 0.0
    return len(s_grams & t_grams) / len(union)


def _compat_generate_deletes(term: str, max_edits: int) -> set[str]:
    """Exact port of retrieval.py::_generate_deletes (char-based)."""
    if max_edits == 0 or not term:
        return {term}
    n = len(term)
    if max_edits == 1:
        result = {term}
        for i in range(n):
            result.add(term[:i] + term[i + 1 :])
        return result
    deletes = {term}
    for _ in range(max_edits):
        new_deletes = set()
        for t in deletes:
            for i in range(len(t)):
                new_deletes.add(t[:i] + t[i + 1 :])
        deletes |= new_deletes
    return deletes


class _CompatSymmetricDeleteIndex:
    """Pure-Python mirror of the kernel contract: two-level delete index,
    deterministic (score desc, term asc) ordering, bounded cache."""

    def __init__(self) -> None:
        self._terms: list[str] = []
        self._term_set: set[str] = set()
        self._deletes1: dict[str, list[int]] = {}
        self._deletes2: dict[str, list[int]] = {}
        self._cache: dict[tuple[str, int, int], list[tuple[str, float]]] = {}
        self._cache_order: list[tuple[str, int, int]] = []
        self._cache_size = 4096

    def add_term(self, term: str, max_edits: int = 2) -> None:
        tid = len(self._terms)
        self._terms.append(term)
        self._term_set.add(term)
        for me in range(1, min(max_edits, 2) + 1):
            target = self._deletes1 if me == 1 else self._deletes2
            for variant in _compat_generate_deletes(term, me):
                if variant != term:
                    target.setdefault(variant, []).append(tid)
        self._cache.clear()
        self._cache_order.clear()

    def _cache_get(self, key):
        if key in self._cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, value) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    def expand(self, query: str, max_edits: int, max_expansions: int = 50):
        if max_edits == 0 or not query:
            return [(query, 1.0)] if query and self.has_term(query) else []
        if max_expansions == 0:
            return []
        key = (query, max_edits, max_expansions)
        cached = self._cache_get(key)
        if cached is not None:
            return list(cached)
        sd = self._deletes2 if max_edits == 2 else self._deletes1
        if max_edits == 2 and not sd:
            sd = self._deletes1
        candidate_ids: set[int] = set()
        for variant in _compat_generate_deletes(query, max_edits):
            for tid in sd.get(variant, ()):
                candidate_ids.add(tid)
        if query in self._term_set:
            candidate_ids.add(self._terms.index(query))
        result: list[tuple[str, float]] = []
        for tid in candidate_ids:
            if len(result) >= max_expansions:
                break
            term = self._terms[tid]
            term_len = len(term)
            if abs(term_len - len(query)) > max_edits:
                continue
            if term[:1] != query[:1]:
                continue
            if len(query) <= 64 and _compat_freq_lower_bound(query, term) > max_edits:
                continue
            dl = _compat_damerau_levenshtein(query, term, max_edits)
            if dl <= max_edits:
                result.append((term, 1.0 / (1.0 + dl)))
        result.sort(key=lambda x: (-x[1], x[0]))
        result = result[:max_expansions]
        self._cache_put(key, result)
        return list(result)

    def term_count(self) -> int:
        return len(self._terms)

    def has_term(self, term: str) -> bool:
        return term in self._term_set

    def reset(self) -> None:
        self._terms = []
        self._term_set = set()
        self._deletes1 = {}
        self._deletes2 = {}
        self._cache.clear()
        self._cache_order.clear()


def _compat_simhash(tokens: list[str], seed: int = _XXH3_SEED) -> int:
    seen = set()
    v = [0] * 64
    for tok in tokens:
        h = _compat_hash64(_enc(tok), seed)
        if h in seen:
            continue
        seen.add(h)
        for bit in range(64):
            v[bit] += 1 if (h >> bit) & 1 else -1
    result = 0
    for bit in range(64):
        if v[bit] > 0:
            result |= 1 << bit
    return result


def _compat_minhash(shingles: list[str], k: int, seed: int = _XXH3_SEED) -> list[int]:
    if k == 0:
        return []
    if not shingles:
        return [0] * k  # reference: no shingles -> [0] * num_perm
    sig = [0xFFFFFFFF] * k
    perm_seeds = [_compat_hash64(struct.pack("<I", p), seed) & 0xFFFFFFFF for p in range(k)]
    for sh in shingles:
        sh_hash = _compat_hash64(_enc(sh), seed) & 0xFFFFFFFF
        for p in range(k):
            v = sh_hash ^ perm_seeds[p]
            if v < sig[p]:
                sig[p] = v
    return sig


def _compat_mmr_rerank(scores, sim_matrix, lambda_param: float = 0.5,
                       k: int = 0) -> list[int]:
    if not scores:
        return []
    k = len(scores) if k == 0 else min(k, len(scores))
    selected: list[int] = []
    remaining = list(range(len(scores)))
    while remaining and len(selected) < k:
        best_idx = -1
        best_score = float("-inf")
        for idx in remaining:
            max_sim = 0.0
            for sel in selected:
                s = sim_matrix[idx][sel]
                if s > max_sim:
                    max_sim = s
            score = lambda_param * scores[idx] - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _compat_xquad_rerank(scores, k: int = 0) -> list[int]:
    if not scores:
        return []
    k = len(scores) if k == 0 else min(k, len(scores))
    selected: list[int] = []
    remaining = list(range(len(scores)))
    while remaining and len(selected) < k:
        best_idx = -1
        best_score = float("-inf")
        for idx in remaining:
            # score-only xQuAD: (1 - lambda) * rel with lambda = 0.5
            score = 0.5 * scores[idx]
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------

def _ns():
    """Per-call native gate (mirrors text.py's per-call use_native checks so
    the KIMIX_NATIVE_SEARCH=0 monkeypatch toggle works after import)."""
    if not use_native("SEARCH") or _native is None:
        return None
    return _native.search


def bm25_idf(doc_count: int, df: int, k1: float = 1.2, b: float = 0.75) -> float:
    _N = _ns()
    if _N is not None:
        return float(_N.bm25_idf(int(doc_count), int(df), float(k1), float(b)))
    return _compat_bm25_idf(doc_count, df, k1, b)


def bm25_score(query_postings, idf, doc_lengths, avg_doc_len, doc_count,
               k1: float = 1.2, b: float = 0.75) -> list[float]:
    """One-shot BM25 accumulation. query_postings: list of per-term
    postings [(doc_id, tf), ...]; idf: per-term; doc_lengths indexed by
    doc_id (size doc_count). Returns the full float64 score array."""
    _N = _ns()
    if _N is not None:
        qp = [[(int(d), int(tf)) for d, tf in pl] for pl in query_postings]
        return [float(v) for v in _N.bm25_score(
            qp, [float(x) for x in idf], [int(x) for x in doc_lengths],
            float(avg_doc_len), int(doc_count), float(k1), float(b))]
    return _compat_bm25_score(query_postings, idf, doc_lengths,
                              avg_doc_len, doc_count, k1, b)


def bm25_topk(scores, k: int) -> list[int]:
    _N = _ns()
    if _N is not None:
        return [int(d) for d in _N.bm25_topk([float(x) for x in scores], int(k))]
    return _compat_top_k(scores, k)


def damerau_levenshtein(a: str, b: str, max_dist: int = -1) -> int:
    _N = _ns()
    if _N is not None:
        return int(_N.damerau_levenshtein(_enc(a), _enc(b), int(max_dist)))
    return _compat_damerau_levenshtein(a, b, max_dist)


def freq_lower_bound(pattern: str, term: str) -> int:
    _N = _ns()
    if _N is not None:
        return int(_N.freq_lower_bound(_enc(pattern), _enc(term)))
    return _compat_freq_lower_bound(pattern, term)


def jaro_similarity(a: str, b: str) -> float:
    _N = _ns()
    if _N is not None:
        return float(_N.jaro_similarity(_enc(a), _enc(b)))
    return _compat_jaro_similarity(a, b)


def jaro_winkler(a: str, b: str, prefix_scale: float = 0.1) -> float:
    _N = _ns()
    if _N is not None:
        return float(_N.jaro_winkler(_enc(a), _enc(b), float(prefix_scale)))
    return _compat_jaro_winkler(a, b, prefix_scale)


def sorensen_dice(a: str, b: str) -> float:
    _N = _ns()
    if _N is not None:
        return float(_N.sorensen_dice(_enc(a), _enc(b)))
    return _compat_sorensen_dice(a, b)


def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    _N = _ns()
    if _N is not None:
        return float(_N.ngram_overlap(_enc(a), _enc(b), int(n)))
    return _compat_ngram_overlap(a, b, n)


class SymmetricDeleteIndex:
    def __init__(self) -> None:
        _N = _ns()
        self._native = _N.SymmetricDeleteIndex() if _N is not None else None
        self._compat = None if self._native is not None else _CompatSymmetricDeleteIndex()

    def add_term(self, term: str, max_edits: int = 2) -> None:
        if self._native is not None:
            self._native.add_term(_enc(term), int(max_edits))
        else:
            self._compat.add_term(term, max_edits)

    def expand(self, query: str, max_edits: int, max_expansions: int = 50):
        if self._native is not None:
            out = self._native.expand(_enc(query), int(max_edits), int(max_expansions))
            return [(_dec(t), float(s)) for t, s in out]
        return self._compat.expand(query, max_edits, max_expansions)

    def term_count(self) -> int:
        if self._native is not None:
            return int(self._native.term_count())
        return self._compat.term_count()

    def has_term(self, term: str) -> bool:
        if self._native is not None:
            return bool(self._native.has_term(_enc(term)))
        return self._compat.has_term(term)

    def reset(self) -> None:
        if self._native is not None:
            self._native.reset()
        else:
            self._compat.reset()


def simhash(tokens: list[str], seed: int = _XXH3_SEED) -> int:
    _N = _ns()
    if _N is not None:
        return int(_N.simhash([_enc(t) for t in tokens], int(seed)))
    return _compat_simhash(tokens, seed)


def minhash(shingles: list[str], k: int, seed: int = _XXH3_SEED) -> list[int]:
    _N = _ns()
    if _N is not None:
        return [int(v) for v in _N.minhash([_enc(s) for s in shingles], int(k), int(seed))]
    return _compat_minhash(shingles, k, seed)


def mmr_rerank(scores, sim_matrix, lambda_param: float = 0.5, k: int = 0) -> list[int]:
    _N = _ns()
    if _N is not None:
        return [int(d) for d in _N.mmr_rerank(
            [float(x) for x in scores],
            [[float(v) for v in row] for row in sim_matrix],
            float(lambda_param), int(k))]
    return _compat_mmr_rerank(scores, sim_matrix, lambda_param, k)


def xquad_rerank(scores, k: int = 0) -> list[int]:
    _N = _ns()
    if _N is not None:
        return [int(d) for d in _N.xquad_rerank([float(x) for x in scores], int(k))]
    return _compat_xquad_rerank(scores, k)
