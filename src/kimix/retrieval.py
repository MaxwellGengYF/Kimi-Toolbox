"""BM25-based retrieval with n-gram tokenization, fuzzy matching, and hybrid scoring."""

from __future__ import annotations

import functools
import heapq
import math
import operator
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import msgspec
import numpy as np
import xxhash
from numpy.typing import NDArray

try:  # native acceleration (optional; failure-tolerant for isolated envs)
    from kimi_cli.native_loader import (
        get_compat as _native_get_compat,
        get_module as _native_get_module,
        use_native as _native_use_native,
    )
except Exception:  # pragma: no cover
    _native_get_compat = lambda name: None
    _native_get_module = lambda name: None
    _native_use_native = lambda kernel: False

# Resolve the native submodules once at import time. The runtime environment
# is stable (env toggles are fixed at process start), so
# ``_native_get_module("index")`` / ``("search")`` can never change — this
# turns the per-item hot path into a single gate check + attribute access
# instead of two function calls and two dict lookups.
_NATIVE_INDEX = _native_get_module("index")
_NATIVE_SEARCH = _native_get_module("search")
# Pure-Python reference implementation (canonical copy lives in the shim);
# resolved lazily to avoid an import-time dependency on the shim package.
_COMPAT_SEARCH = None


def _compat_search():
    global _COMPAT_SEARCH
    if _COMPAT_SEARCH is None:
        _COMPAT_SEARCH = _native_get_compat("search")
    return _COMPAT_SEARCH

_EMPTY_DOCS: NDArray[np.int32] = np.array([], dtype=np.int32)
_EMPTY_TFS: NDArray[np.float64] = np.array([], dtype=np.float64)


class NgramTokenizer:
    """Overlapping n-gram generator with text normalization."""

    __slots__ = ("n",)

    def __init__(self, n: int = 2) -> None:
        self.n = n

    @staticmethod
    def normalize(text: str) -> str:
        """Lower-case and apply Unicode NFKC normalization."""
        # Native acceleration: kimix_native.index.NgramTokenizer.normalize.
        if _native_use_native("INDEX") and _NATIVE_INDEX is not None:
            return _NATIVE_INDEX.NgramTokenizer(2).normalize(text)
        lowered = text.lower()
        if lowered.isascii():
            return lowered
        return unicodedata.normalize("NFKC", lowered)

    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def _is_cjk(char: str) -> bool:
        cp = ord(char)
        return (
            (0x4E00 <= cp <= 0x9FFF)          # CJK Unified Ideographs
            or (0xAC00 <= cp <= 0xD7AF)       # Hangul Syllables
            or (0x3040 <= cp <= 0x309F)       # Hiragana
            or (0x30A0 <= cp <= 0x30FF)       # Katakana
            or (0x3400 <= cp <= 0x4DBF)       # Extension A
            or (0x20000 <= cp <= 0x2EBEF)     # Extensions B-F
            or (0xF900 <= cp <= 0xFAFF)       # CJK Compatibility Ideographs
            or (0x2F800 <= cp <= 0x2FA1F)     # CJK Compatibility Ideographs Supplement
            or (0x30000 <= cp <= 0x3134F)     # Extension G
            or (0x31350 <= cp <= 0x323AF)     # Extension H
            or (0x2EBF0 <= cp <= 0x2EE5F)     # Extension I
            or (0x1100 <= cp <= 0x11FF)       # Hangul Jamo
            or (0xA960 <= cp <= 0xA97F)       # Hangul Jamo Extended-A
            or (0xD7B0 <= cp <= 0xD7FF)       # Hangul Jamo Extended-B
            or (0x31C0 <= cp <= 0x31EF)       # CJK Strokes
            or (0x3200 <= cp <= 0x32FF)       # Enclosed CJK Letters and Months
        )

    def _detect_n(self, text: str) -> int:
        """Auto-detect n-gram size: bigram for CJK, trigram for mixed/code."""
        # Native acceleration: kimix_native.index.NgramTokenizer.detect_n.
        if _native_use_native("INDEX") and _NATIVE_INDEX is not None:
            return _NATIVE_INDEX.NgramTokenizer(self.n).detect_n(text)
        if not text:
            return self.n
        # Fast path: ASCII text cannot contain CJK, so skip the scan
        if text.isascii():
            return 3 if self.n < 3 else self.n
        cjk_count = 0
        threshold = len(text) * 3 // 10
        is_cjk = self._is_cjk
        for c in text:
            if is_cjk(c):
                cjk_count += 1
                if cjk_count > threshold:
                    return 2
        return 3 if self.n < 3 else self.n

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _tokenize_impl(text: str, n: int) -> tuple[str, ...]:
        if len(text) < n:
            return (text,)
        return tuple(text[i : i + n] for i in range(len(text) - n + 1))

    def tokenize(self, text: str, n: int | None = None) -> list[str]:
        """Generate overlapping character n-grams from *text*."""
        # Native acceleration: kimix_native.index.NgramTokenizer.tokenize.
        if _native_use_native("INDEX") and _NATIVE_INDEX is not None:
            return _NATIVE_INDEX.NgramTokenizer(self.n).tokenize(text, n)
        text = self.normalize(text).strip()
        if not text:
            return []
        use_n = n if n is not None else self._detect_n(text)
        return list(self._tokenize_impl(text, use_n))


class _PostingList:
    """Lightweight mutable pair of parallel doc-id and tf lists."""

    __slots__ = ("doc_ids", "tfs")

    def __init__(self) -> None:
        self.doc_ids: list[int] = []
        self.tfs: list[int] = []


class InvertedIndex:
    """Inverted index: build, persist, and load."""

    __slots__ = (
        "_term_to_id",
        "_temp_postings",
        "_doc_lengths",
        "_doc_lengths_arr",
        "_sum_doc_lengths",
        "_N",
        "_avgdl",
        "_max_doc_id",
        "_doc_id_to_idx",
        "_idx_to_doc_id",
        "_finalized",
        "_terms_by_length",
        "_terms_by_length_prefix",
        "_symmetric_delete_index",
        "_doc_term_freqs",
        "_doc_token_strs",
        "_collection_lm_cache",
        "_posting_docs_data",
        "_posting_docs_ptr",
        "_posting_tfs_data",
        "_posting_tfs_ptr",
        "_term_collection_freqs",
        "_postings_sorted",
        "_last_doc_id",
        "_term_id_dirty",
    )

    _MAGIC = b"KIMX"
    _VERSION = 3

    # msgspec Structs for index metadata serialization
    class _IndexMeta(msgspec.Struct):
        """Serializable index metadata, replacing struct-based header."""
        version: int
        N: int
        num_docs: int
        avgdl: float
        terms: list[str]
        doc_lengths: list[int]
        symmetric_delete: dict[int, dict[str, list[str]]]

    class _ForwardIndex(msgspec.Struct):
        """Forward index (document -> term frequencies)."""
        doc_term_freqs: list[dict[str, int]]
        doc_token_strs: list[list[str]]

    def __init__(self) -> None:
        self._term_to_id: dict[str, int] = {}
        self._temp_postings: dict[str, _PostingList] = {}
        self._doc_lengths: list[int] = []
        self._doc_lengths_arr: NDArray[np.int32] = np.array([], dtype=np.int32)
        self._sum_doc_lengths: int = 0
        self._N: int = 0
        self._avgdl: float = 0.0
        self._max_doc_id: int = -1
        self._doc_id_to_idx: dict[int, int] = {}
        self._idx_to_doc_id: list[int] = []
        self._finalized: bool = False
        self._terms_by_length: dict[int, tuple[str, ...]] = {}
        self._terms_by_length_prefix: dict[tuple[int, str], tuple[str, ...]] = {}
        self._symmetric_delete_index: dict[int, dict[str, tuple[str, ...]]] = {}
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_token_strs: list[set[str]] = []
        self._collection_lm_cache: dict[str, float] | None = None
        self._posting_docs_data: NDArray[np.int32] = np.array([], dtype=np.int32)
        self._posting_docs_ptr: NDArray[np.int32] = np.array([0], dtype=np.int32)
        self._posting_tfs_data: NDArray[np.uint16] = np.array([], dtype=np.uint16)
        self._posting_tfs_ptr: NDArray[np.int32] = np.array([0], dtype=np.int32)
        self._term_collection_freqs: list[int] = []
        self._postings_sorted: bool = True
        self._last_doc_id: int = -1
        self._term_id_dirty: bool = True

    @property
    def N(self) -> int:
        return self._N

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @property
    def doc_lengths(self) -> list[int]:
        return self._doc_lengths

    @property
    def doc_lengths_arr(self) -> NDArray[np.int32]:
        return self._doc_lengths_arr

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Add a document's tokens to the index."""
        if self._finalized:
            raise RuntimeError("Cannot add documents after finalize().")
        counter = Counter(tokens)
        if doc_id in self._doc_id_to_idx:
            idx = self._doc_id_to_idx[doc_id]
            old_len = self._doc_lengths[idx]
            self._sum_doc_lengths += len(tokens) - old_len
            self._doc_lengths[idx] = len(tokens)
            self._doc_term_freqs[idx] = counter
        else:
            idx = len(self._doc_lengths)
            self._doc_id_to_idx[doc_id] = idx
            self._idx_to_doc_id.append(doc_id)
            self._doc_lengths.append(len(tokens))
            self._sum_doc_lengths += len(tokens)
            self._doc_term_freqs.append(counter)
            self._doc_token_strs.append(set())
        self._max_doc_id = max(self._max_doc_id, doc_id)
        self._N = self._max_doc_id + 1
        temp_postings = self._temp_postings
        for token, freq in counter.items():
            pl = temp_postings.setdefault(token, _PostingList())
            pl.doc_ids.append(doc_id)
            pl.tfs.append(freq)
        self._last_doc_id = doc_id
        self._term_id_dirty = True

    def _is_stop_ngram(self, token: str, df: int, threshold: float = 0.5) -> bool:
        """Drop n-grams appearing in >*threshold* fraction of docs or pure punctuation."""
        if not token:
            return True
        if threshold > 0 and df > self._N * threshold:
            return True
        if token.isalpha():
            return False
        for c in token:
            if not unicodedata.category(c).startswith("P"):
                return False
        return True

    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def _generate_deletes(term: str, max_edits: int) -> frozenset[str]:
        """Generate all unique strings obtainable by deleting up to max_edits chars."""
        if max_edits == 0 or not term:
            return frozenset({term})
        n = len(term)
        # Fast path for the overwhelmingly-common max_edits == 1 case
        if max_edits == 1:
            result = {term}
            for i in range(n):
                result.add(term[:i] + term[i + 1 :])
            return frozenset(result)
        deletes: set[str] = {term}
        for _ in range(max_edits):
            new_deletes: set[str] = set()
            for t in deletes:
                for i in range(len(t)):
                    new_deletes.add(t[:i] + t[i + 1 :])
            deletes |= new_deletes
        return frozenset(deletes)

    def _build_symmetric_delete_index(self, sd_max_len: int = 8) -> None:
        """Build Symmetric Delete indices for max_edits 1 and 2."""
        if self._symmetric_delete_index:
            return
        if not self._term_to_id:
            self._symmetric_delete_index = {1: {}, 2: {}}
            return
        sd1: dict[str, list[str]] = defaultdict(list)
        sd2: dict[str, list[str]] = defaultdict(list)
        for term in self._term_to_id:
            if len(term) <= sd_max_len:
                for variant in self._generate_deletes(term, 1):
                    if variant != term:
                        sd1[variant].append(term)
                for variant in self._generate_deletes(term, 2):
                    if variant != term:
                        sd2[variant].append(term)
        self._symmetric_delete_index = {
            1: {k: tuple(v) for k, v in sd1.items()},
            2: {k: tuple(v) for k, v in sd2.items()},
        }

    def finalize(self, stop_threshold: float = 0.5, prune_df: int | None = None) -> None:
        """Convert temporary postings to compact numpy arrays."""
        if self._finalized:
            return

        # Rebuild temp_postings from scratch to handle overwrites and ensure sorted order
        temp_postings: dict[str, _PostingList] = {}
        for idx, doc_id in enumerate(self._idx_to_doc_id):
            counter = self._doc_term_freqs[idx]
            for token, freq in counter.items():
                pl = temp_postings.setdefault(token, _PostingList())
                pl.doc_ids.append(doc_id)
                pl.tfs.append(freq)
        for pl in temp_postings.values():
            if len(pl.doc_ids) > 1:
                postings = list(zip(pl.doc_ids, pl.tfs))
                postings.sort(key=operator.itemgetter(0))
                pl.doc_ids[:], pl.tfs[:] = zip(*postings) if postings else ([], [])
        self._temp_postings = temp_postings

        kept_terms: dict[str, int] = {}
        kept_doc_ids: list[list[int]] = []
        kept_tfs: list[list[int]] = []
        for token, pl in self._temp_postings.items():
            df = len(pl.doc_ids)
            if self._is_stop_ngram(token, df, stop_threshold):
                continue
            if prune_df is not None and df > prune_df:
                continue
            tid = len(kept_terms)
            kept_terms[token] = tid
            kept_doc_ids.append(pl.doc_ids)
            kept_tfs.append(pl.tfs)

        # Pre-allocate flat arrays
        total_postings = sum(len(p) for p in kept_doc_ids)
        docs_data = np.empty(total_postings, dtype=np.int32)
        tfs_data = np.empty(total_postings, dtype=np.uint16)
        docs_ptr = np.empty(len(kept_doc_ids) + 1, dtype=np.int32)
        tfs_ptr = np.empty(len(kept_tfs) + 1, dtype=np.int32)
        docs_ptr[0] = 0
        tfs_ptr[0] = 0

        idx = 0
        for i, (doc_ids, tfs) in enumerate(zip(kept_doc_ids, kept_tfs)):
            n = len(doc_ids)
            if n == 1:
                docs_data[idx] = doc_ids[0]
                tfs_data[idx] = tfs[0]
            else:
                docs_data[idx : idx + n] = doc_ids  # type: ignore[call-overload]
                tfs_data[idx : idx + n] = tfs  # type: ignore[call-overload]
            idx += n
            docs_ptr[i + 1] = idx
            tfs_ptr[i + 1] = idx

        self._term_to_id = kept_terms
        self._posting_docs_data = docs_data
        self._posting_tfs_data = tfs_data
        self._posting_docs_ptr = docs_ptr
        self._posting_tfs_ptr = tfs_ptr

        by_len: dict[int, list[str]] = defaultdict(list)
        by_len_prefix: dict[tuple[int, str], list[str]] = defaultdict(list)
        for term in kept_terms:
            length = len(term)
            by_len[length].append(term)
            by_len_prefix[(length, term[:1])].append(term)
        self._terms_by_length = {length: tuple(terms) for length, terms in by_len.items()}
        self._terms_by_length_prefix = {key: tuple(terms) for key, terms in by_len_prefix.items()}
        if self._doc_lengths:
            self._avgdl = self._sum_doc_lengths / len(self._doc_lengths)
            arr = np.zeros(self._N, dtype=np.int32)
            for idx, doc_id in enumerate(self._idx_to_doc_id):
                arr[doc_id] = self._doc_lengths[idx]
            self._doc_lengths_arr = arr

        # Filter forward index only if terms were actually pruned
        if len(kept_terms) < len(self._temp_postings):
            self._doc_term_freqs = [
                {t: f for t, f in doc_freqs.items() if t in kept_terms}
                for doc_freqs in self._doc_term_freqs
            ]
        # _doc_token_strs is built on demand in _doc_token_set

        # Pre-compute collection LM and term collection frequencies
        self._term_collection_freqs = [0] * len(kept_terms)
        if self._sum_doc_lengths > 0:
            self._collection_lm_cache = {}
            for term, tid in kept_terms.items():
                start = self._posting_tfs_ptr[tid]
                end = self._posting_tfs_ptr[tid + 1]
                cf = int(self._posting_tfs_data[start:end].sum())
                self._term_collection_freqs[tid] = cf
                self._collection_lm_cache[term] = cf / self._sum_doc_lengths
        else:
            self._collection_lm_cache = {}

        self._term_id_dirty = False
        self._temp_postings.clear()
        self._finalized = True

    def get_postings(
        self, term: str
    ) -> tuple[NDArray[np.int32], NDArray[np.uint16]] | None:
        """Return (doc_ids, term_frequencies) for *term*, or ``None``."""
        if not self._finalized:
            self.finalize()
        tid = self._term_to_id.get(term)
        if tid is None:
            return None
        start = self._posting_docs_ptr[tid]
        end = self._posting_docs_ptr[tid + 1]
        docs = self._posting_docs_data[start:end]
        tfs = self._posting_tfs_data[start:end]
        return docs, tfs

    def doc_freq(self, term: str) -> int:
        """Document frequency of *term*."""
        postings = self.get_postings(term)
        if postings is None:
            return 0
        return len(postings[0])

    def has_term(self, term: str) -> bool:
        if not self._finalized:
            return term in self._temp_postings
        return term in self._term_to_id

    def terms(self) -> Iterable[str]:
        if not self._finalized:
            return self._temp_postings.keys()
        return self._term_to_id.keys()

    def save(self, path: str | Path, include_forward_index: bool = False) -> None:
        """Persist the index to disk using msgspec for metadata."""
        if not self._finalized:
            self.finalize()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        terms = list(self._term_to_id.keys())
        num_docs = len(self._doc_lengths)

        # Build symmetric delete index (convert tuples to lists for msgspec)
        sym_delete: dict[int, dict[str, list[str]]] = {}
        for k, v in self._symmetric_delete_index.items():
            sym_delete[k] = {vk: list(vv) for vk, vv in v.items()}

        # Serialize metadata with msgspec
        meta = self._IndexMeta(
            version=self._VERSION,
            N=self._N,
            num_docs=num_docs,
            avgdl=self._avgdl,
            terms=terms,
            doc_lengths=self._doc_lengths,
            symmetric_delete=sym_delete,
        )
        meta_bytes = msgspec.msgpack.encode(meta)

        # Build numpy posting data (efficient binary)
        num_terms = len(terms)
        docs_data_list: list[NDArray[np.int32]] = []
        tfs_data_list: list[NDArray[np.uint16]] = []
        df_list: list[int] = []
        for tid in range(num_terms):
            start = self._posting_docs_ptr[tid]
            end = self._posting_docs_ptr[tid + 1]
            docs = self._posting_docs_data[start:end]
            tfs = self._posting_tfs_data[start:end]
            docs_data_list.append(docs)
            tfs_data_list.append(tfs)
            df_list.append(len(docs))

        all_docs = np.concatenate(docs_data_list) if docs_data_list else np.array([], dtype=np.int32)
        all_tfs = np.concatenate(tfs_data_list) if tfs_data_list else np.array([], dtype=np.uint16)

        with open(path, "wb") as f:
            # Magic
            f.write(self._MAGIC)
            # msgspec-metadata length prefix (for fast skip during load)
            f.write(len(meta_bytes).to_bytes(4, 'big'))
            f.write(meta_bytes)
            # Posting list metadata
            f.write(np.array(df_list, dtype='<i4').tobytes())
            # Posting docs and tfs
            f.write(all_docs.astype('<i4').tobytes())
            f.write(all_tfs.astype('<u2').tobytes())

            # Optional forward index
            if include_forward_index and self._doc_term_freqs:
                fi = self._ForwardIndex(
                    doc_term_freqs=[dict(d) for d in self._doc_term_freqs],
                    doc_token_strs=[list(s) for s in self._doc_token_strs],
                )
                fi_bytes = msgspec.msgpack.encode(fi)
                f.write(b"\x01")
                f.write(len(fi_bytes).to_bytes(4, 'big'))
                f.write(fi_bytes)
            else:
                f.write(b"\x00")

    def load(self, path: str | Path) -> None:
        """Load a persisted index from disk using msgspec for metadata."""
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != self._MAGIC:
                raise ValueError(f"Invalid file format: expected {self._MAGIC!r}, got {magic!r}")

            # Read msgspec-metadata
            meta_len = int.from_bytes(f.read(4), 'big')
            meta_bytes = f.read(meta_len)
            meta: InvertedIndex._IndexMeta = msgspec.msgpack.decode(meta_bytes, type=self._IndexMeta)

            self._N = meta.N
            self._avgdl = meta.avgdl
            num_docs = meta.num_docs
            num_terms = len(meta.terms)

            self._term_to_id = {term: i for i, term in enumerate(meta.terms)}
            self._doc_lengths = meta.doc_lengths
            self._sum_doc_lengths = sum(meta.doc_lengths)
            self._doc_lengths_arr = np.array(meta.doc_lengths, dtype=np.int32)

            # Read df list (num_terms * 4 bytes)
            df_data = f.read(num_terms * 4)
            df_list = list(np.frombuffer(df_data, dtype='<i4'))

            total_postings = sum(df_list)
            # Read posting docs (total_postings * 4 bytes)
            docs_data = f.read(total_postings * 4) if total_postings else b''
            # Read posting tfs (total_postings * 2 bytes)
            tfs_data = f.read(total_postings * 2) if total_postings else b''

            # Build posting arrays
            docs_ptr = [0]
            tfs_ptr = [0]
            docs_data_list: list[NDArray[np.int32]] = []
            tfs_data_list: list[NDArray[np.uint16]] = []
            offset_docs = 0
            offset_tfs = 0
            for df in df_list:
                df_int = int(df)
                if df_int:
                    chunk_docs = np.frombuffer(docs_data[offset_docs:offset_docs + df_int * 4], dtype='<i4').copy()
                    chunk_tfs = np.frombuffer(tfs_data[offset_tfs:offset_tfs + df_int * 2], dtype='<u2').copy()
                    offset_docs += df_int * 4
                    offset_tfs += df_int * 2
                else:
                    chunk_docs = np.array([], dtype=np.int32)
                    chunk_tfs = np.array([], dtype=np.uint16)
                docs_data_list.append(chunk_docs)
                tfs_data_list.append(chunk_tfs)
                docs_ptr.append(docs_ptr[-1] + df_int)
                tfs_ptr.append(tfs_ptr[-1] + df_int)

            self._posting_docs_data = np.concatenate(docs_data_list) if docs_data_list else np.array([], dtype=np.int32)
            self._posting_tfs_data = np.concatenate(tfs_data_list) if tfs_data_list else np.array([], dtype=np.uint16)
            self._posting_docs_ptr = np.array(docs_ptr, dtype=np.int32)
            self._posting_tfs_ptr = np.array(tfs_ptr, dtype=np.int32)

            # Forward index flag
            flag_byte = f.read(1)
            has_forward_chunk = bool(flag_byte[0]) if flag_byte else False

            doc_token_strs_chunk: list[set[str]] = []
            doc_term_freqs_chunk: list[dict[str, int]] = []
            if has_forward_chunk:
                try:
                    fi_len = int.from_bytes(f.read(4), 'big')
                    fi_bytes = f.read(fi_len)
                    fi: InvertedIndex._ForwardIndex = msgspec.msgpack.decode(fi_bytes, type=self._ForwardIndex)
                    doc_term_freqs_chunk = fi.doc_term_freqs
                    doc_token_strs_chunk = [set(s) for s in fi.doc_token_strs]
                except (msgspec.DecodeError, ValueError):
                    has_forward_chunk = False
                    doc_token_strs_chunk = []
                    doc_term_freqs_chunk = []

            # Rebuild _IndexMeta.symmetric_delete (convert lists back to tuples)
            self._symmetric_delete_index = {}
            for k, v in meta.symmetric_delete.items():
                self._symmetric_delete_index[k] = {vk: tuple(vv) for vk, vv in v.items()}

            by_len: dict[int, list[str]] = defaultdict(list)
            by_len_prefix: dict[tuple[int, str], list[str]] = defaultdict(list)
            for term in self._term_to_id:
                length = len(term)
                by_len[length].append(term)
                by_len_prefix[(length, term[:1])].append(term)
            self._terms_by_length = {length: tuple(terms) for length, terms in by_len.items()}
            self._terms_by_length_prefix = {key: tuple(terms) for key, terms in by_len_prefix.items()}

            n_docs = len(self._doc_lengths)
            self._term_collection_freqs = [0] * num_terms
            total_tokens = self._sum_doc_lengths
            self._collection_lm_cache = {} if total_tokens > 0 else {}

            self._doc_id_to_idx = {i: i for i in range(n_docs)}
            self._idx_to_doc_id = list(range(n_docs))

            if has_forward_chunk:
                self._doc_token_strs = doc_token_strs_chunk
                self._doc_term_freqs = doc_term_freqs_chunk
                for term, tid in self._term_to_id.items():
                    start = self._posting_tfs_ptr[tid]
                    end = self._posting_tfs_ptr[tid + 1]
                    tfs_arr = self._posting_tfs_data[start:end]
                    cf = int(tfs_arr.sum()) if len(tfs_arr) else 0
                    self._term_collection_freqs[tid] = cf
                    if total_tokens > 0:
                        self._collection_lm_cache[term] = cf / total_tokens
            else:
                self._doc_token_strs = [set() for _ in range(n_docs)]
                self._doc_term_freqs = [dict() for _ in range(n_docs)]
                for term, tid in self._term_to_id.items():
                    start = self._posting_docs_ptr[tid]
                    end = self._posting_docs_ptr[tid + 1]
                    docs_arr = self._posting_docs_data[start:end]
                    tfs_arr = self._posting_tfs_data[start:end]
                    cf = int(tfs_arr.sum()) if len(tfs_arr) else 0
                    self._term_collection_freqs[tid] = cf
                    if total_tokens > 0:
                        self._collection_lm_cache[term] = cf / total_tokens
                    docs_list = docs_arr.tolist()
                    tfs_list = tfs_arr.tolist()
                    for d_int, tf_int in zip(docs_list, tfs_list):
                        if d_int < n_docs:
                            self._doc_token_strs[d_int].add(term)
                            self._doc_term_freqs[d_int][term] = tf_int

            self._finalized = True


class BM25Scorer:
    """BM25 relevance scorer over an :class:`InvertedIndex`."""

    __slots__ = ("index", "k1", "b", "_denom_base", "_tfs_buf", "_denom_buf")

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self._denom_base: NDArray[np.float64] | None = None
        self._tfs_buf: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._denom_buf: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._build_denom_base()

    def _build_denom_base(self) -> None:
        avgdl = self.index.avgdl
        if avgdl == 0:
            return
        k1 = self.k1
        b = self.b
        self._denom_base = k1 * ((1.0 - b) + (b / avgdl) * self.index.doc_lengths_arr)

    @staticmethod
    def _idf(df: int, N: int) -> float:
        """IDF = ln(1 + (N - df + 0.5) / (df + 0.5)) - Lucene BM25 variant."""
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    def _ensure_buffers(self, size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(self._tfs_buf) < size:
            new_size = max(size * 2, 256)
            self._tfs_buf = np.empty(new_size, dtype=np.float64)
            self._denom_buf = np.empty(new_size, dtype=np.float64)
        return self._tfs_buf, self._denom_buf

    def _prepare_candidates(self, candidate_docs):
        if candidate_docs is None:
            return None
        if isinstance(candidate_docs, np.ndarray):
            return candidate_docs
        return np.array(sorted(candidate_docs), dtype=np.int32)

    def _token_scores(
        self,
        token: str,
        q_weight: float,
        candidate_docs: NDArray[np.int32] | None,
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Return (doc_ids, score_deltas) for a single query token."""
        postings = self.index.get_postings(token)
        if postings is None:
            return _EMPTY_DOCS, _EMPTY_TFS
        docs, tfs = postings
        global_df = len(docs)
        if candidate_docs is not None:
            if len(candidate_docs) <= 256:
                idx = np.searchsorted(candidate_docs, docs)
                np.minimum(idx, len(candidate_docs) - 1, out=idx)
                valid = candidate_docs[idx] == docs
            else:
                valid = np.isin(docs, candidate_docs)
            if not np.any(valid):
                return _EMPTY_DOCS, _EMPTY_TFS
            docs = docs[valid]
            tfs = tfs[valid]
        df = len(docs)
        if df == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        idf = self._idf(global_df, self.index.N)
        n = len(docs)
        tfs_buf, denom_buf = self._ensure_buffers(n)
        tfs_f = tfs_buf[:n]
        denom = denom_buf[:n]
        tfs_f[:] = tfs
        np.add(tfs_f, self._denom_base[docs], out=denom)
        tfs_f *= idf * (self.k1 + 1.0) * q_weight
        np.divide(tfs_f, denom, out=tfs_f)
        return docs, tfs_f

    def _accumulate(
        self,
        query_tokens: list[str],
        candidate_docs: set[int] | None,
    ) -> np.ndarray:
        N = self.index.N
        scores_arr = np.zeros(N, dtype=np.float64)
        if N == 0 or self._denom_base is None:
            return scores_arr
        if candidate_docs is not None and len(candidate_docs) == 0:
            return scores_arr

        cand_sorted = self._prepare_candidates(candidate_docs)
        token_counts = Counter(query_tokens)
        for token, q_weight in token_counts.items():
            docs, scores = self._token_scores(token, q_weight, cand_sorted)
            if len(docs):
                np.add.at(scores_arr, docs, scores)
        return scores_arr

    def _accumulate_sparse(
        self,
        query_tokens: list[str],
        candidate_docs: set[int] | None,
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        N = self.index.N
        if N == 0 or self._denom_base is None:
            return scores
        if candidate_docs is not None and len(candidate_docs) == 0:
            return scores

        cand_sorted = self._prepare_candidates(candidate_docs)
        token_counts = Counter(query_tokens)
        for token, q_weight in token_counts.items():
            docs, tfs_f = self._token_scores(token, q_weight, cand_sorted)
            scores_get = scores.get
            for d, score in zip(docs.tolist(), tfs_f.tolist()):
                scores[d] = scores_get(d, 0.0) + score
        return scores

    def score(
        self,
        query_tokens: list[str],
        candidate_docs: set[int] | None = None,
    ) -> dict[int, float]:
        """Accumulate BM25 score per candidate document.

        ``candidate_docs`` restricts scoring to a subset of docs; ``None``
        scores every document that has at least one query token.

        Duplicate tokens in *query_tokens* are scored multiple times; callers
        should deduplicate if needed.
        """
        N = self.index.N
        # Use sparse accumulation for large indices to avoid allocating huge zero arrays
        if N > 50000:
            return self._accumulate_sparse(query_tokens, candidate_docs)
        scores_arr = self._accumulate(query_tokens, candidate_docs)
        nonzero = np.flatnonzero(scores_arr).tolist()
        return {int(i): float(scores_arr[i]) for i in nonzero}

    def score_topk(
        self,
        query_tokens: list[str],
        top_k: int,
        candidate_docs: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Accumulate BM25 scores and return the top-*k* results.

        This is significantly faster than :meth:`score` followed by manual
        top-*k* selection when the number of documents is large, because it
        avoids materialising a full ``dict`` of all nonzero scores.
        """
        if self.index.N == 0 or self._denom_base is None or top_k <= 0:
            return []

        N = self.index.N
        cand_size = len(candidate_docs) if candidate_docs is not None else N
        use_sparse = (
            (N > 50000 and top_k < N // 20)
            or (cand_size < 5000 and N > 100000)
            or (N > 5000 and top_k < N // 50)
        )

        if use_sparse:
            scores = self._accumulate_sparse(query_tokens, candidate_docs)
            if not scores:
                return []
            if top_k >= len(scores):
                return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            return heapq.nlargest(top_k, scores.items(), key=lambda x: (x[1], -x[0]))

        scores_arr = self._accumulate(query_tokens, candidate_docs)
        if top_k >= N:
            nonzero = np.flatnonzero(scores_arr).tolist()
            return [(int(i), float(scores_arr[i])) for i in nonzero]

        partitioned = np.argpartition(scores_arr, -top_k)[-top_k:]
        mask = scores_arr[partitioned] > 0
        top_indices = partitioned[mask]
        top_scores = scores_arr[top_indices]
        order = np.argsort(-top_scores).tolist()
        return [(int(top_indices[i]), float(top_scores[i])) for i in order]


class LevenshteinAutomaton:
    """Damerau-Levenshtein automaton for fuzzy term expansion."""

    __slots__ = ("pattern", "max_edits", "prefix_length", "_pattern_counts", "_pattern_counts_items", "_pattern_deletes")

    def __init__(
        self,
        pattern: str,
        max_edits: int,
        prefix_length: int = 1,
    ) -> None:
        self.pattern = pattern
        self.max_edits = max_edits
        self.prefix_length = prefix_length
        # Pre-compute character frequencies for cheap lower-bound rejection
        pc: dict[str, int] = {}
        for c in pattern:
            pc[c] = pc.get(c, 0) + 1
        self._pattern_counts = pc
        self._pattern_counts_items = list(pc.items())
        self._pattern_deletes: frozenset[str] = InvertedIndex._generate_deletes(pattern, max_edits)

    @staticmethod
    def auto_fuzziness(term: str) -> int:
        """AUTO mode: 0-2 chars -> 0, 3-5 -> 1, >5 -> 2."""
        length = len(term)
        if length <= 2:
            return 0
        if length <= 5:
            return 1
        return 2

    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def _damerau_levenshtein(s: str, t: str) -> int:
        """Compute Damerau-Levenshtein distance between *s* and *t*."""
        # Native acceleration: kimix_native.search.damerau_levenshtein.
        if _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
            return _NATIVE_SEARCH.damerau_levenshtein(s, t)
        if len(s) < len(t):
            s, t = t, s
        m, n = len(s), len(t)
        if n == 0:
            return m

        # Fast paths for very short strings (common for n-grams)
        if n == 1:
            return 0 if s[0] == t[0] else 1
        if m == 2 and n == 2:
            if s == t:
                return 0
            if s[0] == t[0] or s[1] == t[1]:
                return 1
            if s[0] == t[1] and s[1] == t[0]:
                return 1
            return 2

        prev_prev = list(range(n + 1))
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            si_1 = s[i - 1]
            for j in range(1, n + 1):
                cost = 0 if si_1 == t[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,      # insertion
                    prev[j] + 1,          # deletion
                    prev[j - 1] + cost,   # substitution
                )
                if (
                    i > 1
                    and j > 1
                    and si_1 == t[j - 2]
                    and s[i - 2] == t[j - 1]
                ):
                    curr[j] = min(curr[j], prev_prev[j - 2] + 1)  # transposition
            prev_prev, prev, curr = prev, curr, prev_prev
        return prev[n]

    def _freq_lower_bound(self, term: str) -> int:
        """Lower bound on edit distance based on character frequencies."""
        # Native acceleration: kimix_native.search.freq_lower_bound.
        if _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
            return _NATIVE_SEARCH.freq_lower_bound(self.pattern, term)
        total = 0
        matched = 0
        term_len = len(term)
        if term_len <= 32:
            for c, pc in self._pattern_counts_items:
                tc_c = term.count(c)
                matched += tc_c
                if pc != tc_c:
                    total += abs(pc - tc_c)
        else:
            tc = Counter(term)
            for c, pc in self._pattern_counts_items:
                tc_c = tc.get(c, 0)
                matched += tc_c
                if pc != tc_c:
                    total += abs(pc - tc_c)
        total += term_len - matched
        return (total + 1) // 2

    def match(self, dictionary: Iterable[str], max_expansions: int = 50) -> list[str]:
        """Walk *dictionary* and collect up to *max_expansions* matches."""
        results: list[str] = []
        pattern_len = len(self.pattern)
        max_edits = self.max_edits
        prefix_length = self.prefix_length
        prefix = self.pattern[:prefix_length] if prefix_length > 0 else ""
        dl = self._damerau_levenshtein
        has_freq_filter = len(self.pattern) <= 64

        # Use Symmetric Delete index for prefix_length == 1 when available
        if prefix_length == 1 and max_edits > 0 and hasattr(dictionary, "_symmetric_delete_index"):
            if not dictionary._symmetric_delete_index:  # type: ignore[attr-defined]
                dictionary._build_symmetric_delete_index()  # type: ignore[attr-defined]
            sd_index = dictionary._symmetric_delete_index  # type: ignore[attr-defined]
            sd = sd_index.get(max_edits) or sd_index.get(1, {})
            if sd:
                candidates: set[str] = set()
                for variant in self._pattern_deletes:
                    if variant in sd:
                        candidates.update(sd[variant])
                if dictionary.has_term(self.pattern):  # type: ignore[attr-defined]
                    candidates.add(self.pattern)
                for term in candidates:
                    if len(results) >= max_expansions:
                        return results
                    term_len = len(term)
                    if abs(term_len - pattern_len) > max_edits:
                        continue
                    if prefix and term[:1] != prefix:
                        continue
                    if has_freq_filter and self._freq_lower_bound(term) > max_edits:
                        continue
                    if dl(self.pattern, term) <= max_edits:
                        results.append(term)
                return results

        if hasattr(dictionary, "_terms_by_length_prefix"):
            terms_by_prefix = dictionary._terms_by_length_prefix  # type: ignore[attr-defined]
            for length in range(
                max(pattern_len - max_edits, prefix_length), pattern_len + max_edits + 1
            ):
                bucket = terms_by_prefix.get((length, prefix), ()) if prefix_length > 0 else ()
                if not bucket and prefix_length > 0:
                    continue
                candidates = bucket or dictionary._terms_by_length.get(length, ())  # type: ignore[attr-defined]
                if prefix_length == 1:
                    for term in candidates:
                        if len(results) >= max_expansions:
                            return results
                        if has_freq_filter and self._freq_lower_bound(term) > max_edits:
                            continue
                        if dl(self.pattern, term) <= max_edits:
                            results.append(term)
                else:
                    for term in candidates:
                        if len(results) >= max_expansions:
                            return results
                        if prefix_length > 0 and term[:prefix_length] != prefix:
                            continue
                        if has_freq_filter and self._freq_lower_bound(term) > max_edits:
                            continue
                        if dl(self.pattern, term) <= max_edits:
                            results.append(term)
                if len(results) >= max_expansions:
                    return results
            return results

        if hasattr(dictionary, "_terms_by_length"):
            terms_by_length = dictionary._terms_by_length  # type: ignore[attr-defined]
            for length in range(
                max(pattern_len - max_edits, prefix_length), pattern_len + max_edits + 1
            ):
                for term in terms_by_length.get(length, ()):
                    if len(results) >= max_expansions:
                        return results
                    if prefix_length == 1:
                        if term[0] != prefix[0]:
                            continue
                    elif prefix_length > 0 and term[:prefix_length] != prefix:
                        continue
                    if has_freq_filter and self._freq_lower_bound(term) > max_edits:
                        continue
                    if dl(self.pattern, term) <= max_edits:
                        results.append(term)
                if len(results) >= max_expansions:
                    return results
            return results

        for term in dictionary:
            if len(results) >= max_expansions:
                break
            term_len = len(term)
            if abs(term_len - pattern_len) > max_edits:
                continue
            if prefix_length == 1:
                if term_len >= 1 and term[0] != prefix[0]:
                    continue
            elif prefix_length > 0:
                if term_len >= prefix_length and term[:prefix_length] != prefix:
                    continue
            if has_freq_filter and self._freq_lower_bound(term) > max_edits:
                continue
            if dl(self.pattern, term) <= max_edits:
                results.append(term)
        return results


class Searcher:
    """Query pipeline orchestrator: normalize -> tokenize -> score -> rank."""

    __slots__ = (
        "index",
        "tokenizer",
        "scorer",
        "k1",
        "b",
        "min_should_match",
        "fuzziness",
        "max_expansions",
        "prefix_length",
        "_expand_cache",
    )

    def __init__(
        self,
        index: InvertedIndex,
        tokenizer: NgramTokenizer | None = None,
        scorer: BM25Scorer | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        min_should_match: float = 0.5,
        fuzziness: str | int = "AUTO",
        max_expansions: int = 50,
        prefix_length: int = 1,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer or NgramTokenizer()
        self.scorer = scorer or BM25Scorer(index, k1=k1, b=b)
        self.k1 = k1
        self.b = b
        self.min_should_match = min_should_match
        self.fuzziness = fuzziness
        self.max_expansions = max_expansions
        self.prefix_length = prefix_length
        self._expand_cache: dict[tuple[str, str | int, int, int], list[str]] = {}
        if self.fuzziness != 0:
            self.index._build_symmetric_delete_index()

    @staticmethod
    def _is_latin_token(token: str) -> bool:
        """Heuristic: token is primarily Latin/ASCII."""
        return bool(token) and token.isascii()

    def _expand_token(self, token: str) -> list[str]:
        """Fuzzy-expand a Latin token; CJK tokens are returned verbatim if present."""
        if not self._is_latin_token(token):
            return [token] if self.index.has_term(token) else []

        max_edits = (
            LevenshteinAutomaton.auto_fuzziness(token)
            if self.fuzziness == "AUTO"
            else int(self.fuzziness)
        )
        if max_edits == 0:
            return [token] if self.index.has_term(token) else []

        key = (token, self.fuzziness, self.prefix_length, self.max_expansions)
        if key in self._expand_cache:
            return self._expand_cache[key]

        automaton = LevenshteinAutomaton(
            token, max_edits=max_edits, prefix_length=self.prefix_length
        )
        matches = automaton.match(
            self.index, max_expansions=self.max_expansions
        )
        result = matches if matches else ([token] if self.index.has_term(token) else [])
        self._expand_cache[key] = result
        return result

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Run the full query pipeline and return top-k *(doc_id, score)* pairs."""
        if self.index.N == 0:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        unique_query = list(dict.fromkeys(query_tokens))
        min_match = max(1, math.ceil(len(unique_query) * self.min_should_match))

        token_expansions: list[list[str]] = []
        hits = 0
        for token in unique_query:
            expanded = self._expand_token(token)
            if expanded:
                hits += 1
            token_expansions.append(expanded)

        if hits < min_match:
            return []

        # Pre-compute candidate_docs from posting intersections to avoid full scan
        candidate_docs: set[int] | None = None
        if min_match > 1 or (self.index.N > 50000 and min_match >= 1):
            doc_token_counts: dict[int, int] = {}
            for expanded in token_expansions:
                if not expanded:
                    continue
                all_docs = []
                for t in expanded:
                    postings = self.index.get_postings(t)
                    if postings is not None:
                        all_docs.append(postings[0])
                if not all_docs:
                    continue
                total_len = sum(len(d) for d in all_docs)
                if total_len > 512 or len(expanded) > 2:
                    concat = np.concatenate(all_docs)
                    unique_docs = np.unique(concat)
                    for d_int in unique_docs.tolist():
                        doc_token_counts[d_int] = doc_token_counts.get(d_int, 0) + 1
                else:
                    seen_docs: set[int] = set()
                    seen_docs_add = seen_docs.add
                    doc_token_counts_get = doc_token_counts.get
                    for docs in all_docs:
                        for d_int in docs.tolist():
                            if d_int not in seen_docs:
                                seen_docs_add(d_int)
                                doc_token_counts[d_int] = doc_token_counts_get(d_int, 0) + 1
            if doc_token_counts:
                candidate_docs = {
                    doc_id for doc_id, count in doc_token_counts.items()
                    if count >= min_match
                }
            else:
                candidate_docs = set()

        expanded_tokens: list[str] = []
        for expanded in token_expansions:
            expanded_tokens.extend(expanded)

        if not expanded_tokens:
            return []

        if candidate_docs is not None and len(candidate_docs) == 0:
            return []

        expanded_tokens = list(dict.fromkeys(expanded_tokens))
        return self.scorer.score_topk(expanded_tokens, top_k=top_k, candidate_docs=candidate_docs)


# ---------------------------------------------------------------------------
# String similarity helpers (Jaro-Winkler, Sørensen-Dice)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=65536)
def jaro_similarity(s: str, t: str) -> float:
    """Return Jaro similarity between *s* and *t* (0.0–1.0)."""
    # Native acceleration: kimix_native.search.jaro_similarity.
    if _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
        return _NATIVE_SEARCH.jaro_similarity(s, t)
    return _compat_search()._compat_jaro_similarity(s, t)


@functools.lru_cache(maxsize=65536)
def jaro_winkler_similarity(s: str, t: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Return Jaro–Winkler similarity, boosting prefix matches."""
    # Native acceleration: kimix_native.search.jaro_winkler (kernel fixes the
    # prefix window at 4, matching this function's default; non-default
    # max_prefix keeps the pure-Python path).
    if max_prefix == 4 and _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
        return _NATIVE_SEARCH.jaro_winkler(s, t, p)
    jaro = jaro_similarity(s, t)
    prefix = 0
    limit = min(max_prefix, len(s), len(t))
    for i in range(limit):
        if s[i] == t[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


@functools.lru_cache(maxsize=65536)
def sorensen_dice_coefficient(s: str, t: str) -> float:
    """Return Sørensen–Dice coefficient based on bigram overlap."""
    # Native acceleration: kimix_native.search.sorensen_dice.
    if _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
        return _NATIVE_SEARCH.sorensen_dice(s, t)
    return _compat_search()._compat_sorensen_dice(s, t)


def ngram_overlap(s: str, t: str, n: int = 2) -> float:
    """Return n-gram overlap ratio (intersection / union)."""
    # Native acceleration: kimix_native.search.ngram_overlap.
    if _native_use_native("SEARCH") and _NATIVE_SEARCH is not None:
        return _NATIVE_SEARCH.ngram_overlap(s, t, n)
    return _compat_search()._compat_ngram_overlap(s, t, n)
    return len(s_grams & t_grams) / len(union)


def jaccard_similarity_tokens(a: set[str], b: set[str]) -> float:
    """Jaccard similarity for token sets."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# SimHash – near-duplicate detection
# ---------------------------------------------------------------------------


class SimHash:
    """Locality-sensitive hash for near-duplicate text detection."""

    __slots__ = ("hashbits", "value")

    def __init__(self, text: str = "", hashbits: int = 64) -> None:
        self.hashbits = hashbits
        self.value = self._compute(text)

    @staticmethod
    def _hash_token(token: str) -> int:
        """Stable 64-bit hash for a token."""
        return xxhash.xxh64(token.encode("utf-8")).intdigest()

    def _compute(self, text: str) -> int:
        v = [0] * self.hashbits
        for token in set(text.split()):
            h = self._hash_token(token)
            for i in range(self.hashbits):
                v[i] += 1 if (h >> i) & 1 else -1
        result = 0
        for i in range(self.hashbits):
            if v[i] > 0:
                result |= 1 << i
        return result

    def distance(self, other: SimHash) -> int:
        """Hamming distance between two SimHash values."""
        return (self.value ^ other.value).bit_count()

    def is_near_duplicate(self, other: SimHash, threshold: int = 3) -> bool:
        return self.distance(other) <= threshold


class SimHashLSH:
    """Locality-sensitive hashing for SimHash to speed up near-duplicate checks."""

    __slots__ = ("hashbits", "band_bits", "num_bands", "buckets", "hashes")

    def __init__(self, hashbits: int = 64, band_bits: int = 4) -> None:
        self.hashbits = hashbits
        self.band_bits = band_bits
        self.num_bands = hashbits // band_bits
        self.buckets: dict[tuple[int, int], set[int]] = {}
        self.hashes: dict[int, SimHash] = {}

    def add(self, doc_id: int, simhash: SimHash) -> None:
        self.hashes[doc_id] = simhash
        mask = (1 << self.band_bits) - 1
        for band in range(self.num_bands):
            val = (simhash.value >> (band * self.band_bits)) & mask
            key = (band, val)
            self.buckets.setdefault(key, set()).add(doc_id)

    def candidates(self, simhash: SimHash) -> set[int]:
        candidates: set[int] = set()
        mask = (1 << self.band_bits) - 1
        for band in range(self.num_bands):
            val = (simhash.value >> (band * self.band_bits)) & mask
            key = (band, val)
            if key in self.buckets:
                candidates.update(self.buckets[key])
        return candidates

    def remove(self, doc_id: int) -> None:
        h = self.hashes.pop(doc_id, None)
        if h is None:
            return
        mask = (1 << self.band_bits) - 1
        for band in range(self.num_bands):
            val = (h.value >> (band * self.band_bits)) & mask
            key = (band, val)
            bucket = self.buckets.get(key)
            if bucket is not None:
                bucket.discard(doc_id)
                if not bucket:
                    del self.buckets[key]


# ---------------------------------------------------------------------------
# MMR – Maximal Marginal Relevance
# ---------------------------------------------------------------------------


def _doc_token_set(index: InvertedIndex, doc_id: int) -> set[str]:
    """Recover the token set for *doc_id* from the forward index."""
    idx = index._doc_id_to_idx.get(doc_id)
    if idx is None:
        return set()
    if idx < len(index._doc_token_strs) and index._doc_token_strs[idx]:
        return index._doc_token_strs[idx].copy()
    if idx < len(index._doc_term_freqs):
        return set(index._doc_term_freqs[idx])
    return set()


def mmr_rerank(
    results: list[tuple[int, float]],
    index: InvertedIndex,
    lambda_param: float = 0.5,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Greedy MMR re-ranking over BM25 *results*.

    ``lambda_param`` balances relevance (1.0 = pure relevance,
    0.0 = pure diversity).  Token-set Jaccard is used as the
    document–document similarity proxy.
    """
    if not results:
        return []
    if top_k is None:
        top_k = len(results)

    # Pre-compute token sets for all result docs
    doc_tokens: dict[int, set[str]] = {}
    for doc_id, _ in results:
        doc_tokens[doc_id] = _doc_token_set(index, doc_id)

    selected: list[tuple[int, float]] = []
    remaining = list(results)

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_doc: tuple[int, float] | None = None
        best_score = -float("inf")
        for idx, (doc_id, relevance) in enumerate(remaining):
            max_sim = 0.0
            for sel_id, _ in selected:
                sim = jaccard_similarity_tokens(doc_tokens[doc_id], doc_tokens[sel_id])
                if sim > max_sim:
                    max_sim = sim
            score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_doc = (doc_id, relevance)
                best_idx = idx
        if best_doc is None:
            break
        selected.append(best_doc)
        remaining.pop(best_idx)
    return selected


# ---------------------------------------------------------------------------
# RM3 – Pseudo-Relevance Feedback query expansion
# ---------------------------------------------------------------------------


class RM3Expander:
    """RM3 pseudo-relevance feedback using top-k BM25 results."""

    __slots__ = ("index", "scorer", "fb_docs", "fb_terms", "alpha")

    def __init__(
        self,
        index: InvertedIndex,
        scorer: BM25Scorer,
        fb_docs: int = 3,
        fb_terms: int = 10,
        alpha: float = 0.5,
    ) -> None:
        self.index = index
        self.scorer = scorer
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.alpha = alpha

    def expand(self, query_tokens: list[str], top_k: int = 10) -> list[str]:
        """Return an expanded token list mixing original query and feedback terms."""
        if not query_tokens:
            return []

        results = self.scorer.score_topk(query_tokens, top_k=self.fb_docs)
        if not results:
            return query_tokens

        doc_ids = {doc_id for doc_id, _ in results}

        # Estimate term probabilities in feedback documents via forward index
        term_scores: dict[str, float] = {}
        total_tokens = 0
        for doc_id in doc_ids:
            idx = self.index._doc_id_to_idx.get(doc_id)
            if idx is not None and idx < len(self.index._doc_term_freqs):
                for term, tf in self.index._doc_term_freqs[idx].items():
                    term_scores[term] = term_scores.get(term, 0.0) + float(tf)
                    total_tokens += tf

        if total_tokens == 0:
            return query_tokens

        for term in term_scores:
            term_scores[term] /= total_tokens

        # Original query model
        expanded: dict[str, float] = {}
        for token in query_tokens:
            expanded[token] = expanded.get(token, 0.0) + self.alpha / len(query_tokens)

        # Feedback model
        top_terms = heapq.nlargest(
            self.fb_terms, term_scores.items(), key=lambda x: x[1]
        )
        for term, score in top_terms:
            expanded[term] = expanded.get(term, 0.0) + (1.0 - self.alpha) * score

        # Flatten to token list weighted by score (repetition ≈ weight)
        result: list[str] = []
        for term, weight in expanded.items():
            count = max(1, int(weight * 10))
            result.extend([term] * count)
        return result


# ---------------------------------------------------------------------------
# LambdaMART – simplified listwise LTR (coordinate-ascent style)
# ---------------------------------------------------------------------------


def _dcg(scores: list[float] | np.ndarray) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        s += (2.0 ** scores[i] - 1.0) / math.log2(i + 2)
    return s


def _ideal_dcg(scores: list[float] | np.ndarray) -> float:
    return _dcg(sorted(scores, reverse=True))


def _ndcg(scores: list[float] | np.ndarray) -> float:
    ideal = _ideal_dcg(scores)
    return 0.0 if ideal == 0 else _dcg(scores) / ideal


class LambdaMART:
    """Simplified LambdaMART / Coordinate-Ascent rank learner.

    Trains a linear weight vector over hand-crafted features to maximise
    mean NDCG.  Inference is a single dot-product per document.
    """

    __slots__ = ("n_iterations", "learning_rate", "weights")

    def __init__(self, n_iterations: int = 50, learning_rate: float = 0.05) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights: list[float] = []

    def fit(self, X: list[list[list[float]]], y: list[list[float]]) -> None:
        """Fit on training data.

        *X* – list of queries, each query is a list of doc-feature vectors.
        *y* – list of queries, each query is a list of relevance grades.
        """
        if not X or not X[0]:
            return
        n_features = len(X[0][0])
        self.weights = [0.0] * n_features
        X_np = [np.asarray(xi, dtype=np.float64) for xi in X]
        w_arr = np.zeros(n_features, dtype=np.float64)
        for _ in range(self.n_iterations):
            best_dim = -1
            best_delta = 0.0
            best_step = 0.0
            baseline_ndcg = 0.0
            w_current = np.asarray(self.weights, dtype=np.float64)
            for xi_np, yi in zip(X_np, y):
                scores = xi_np @ w_current
                baseline_ndcg += _ndcg(scores)
            for dim in range(n_features):
                for step in (-self.learning_rate, self.learning_rate):
                    w_arr[dim] = self.weights[dim] + step
                    ndcg_sum = 0.0
                    for xi_np, yi in zip(X_np, y):
                        scores = xi_np @ w_arr
                        ndcg_sum += _ndcg(scores)
                    improvement = ndcg_sum - baseline_ndcg
                    if improvement > best_delta:
                        best_delta = improvement
                        best_dim = dim
                        best_step = step
                    w_arr[dim] = self.weights[dim]
            if best_dim < 0 or best_delta <= 0:
                break
            self.weights[best_dim] += best_step

    def predict(self, X: list[list[float]]) -> list[float]:
        """Score a list of document-feature vectors."""
        if not self.weights:
            return [0.0] * len(X)
        w_arr = np.asarray(self.weights, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr.dot(w_arr).tolist()

    def rank(
        self, doc_features: list[tuple[int, list[float]]]
    ) -> list[tuple[int, float]]:
        """Return *(doc_id, score)* pairs sorted by descending score."""
        ids, feats = zip(*doc_features) if doc_features else ([], [])
        scores = self.predict(list(feats))
        ranked = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))
        return list(ranked)


# ---------------------------------------------------------------------------
# Query Performance Prediction (QPP)
# ---------------------------------------------------------------------------


class QueryPerformancePredictor:
    """Lightweight QPP heuristics for dynamic strategy selection."""

    __slots__ = ("index", "scorer")

    def __init__(self, index: InvertedIndex, scorer: BM25Scorer) -> None:
        self.index = index
        self.scorer = scorer

    def _batch_doc_freq(self, tokens: set[str]) -> dict[str, int]:
        """Batch document-frequency lookup without repeated state checks."""
        result: dict[str, int] = {}
        for token in tokens:
            tid = self.index._term_to_id.get(token)
            if tid is None:
                continue
            start = self.index._posting_docs_ptr[tid]
            end = self.index._posting_docs_ptr[tid + 1]
            result[token] = int(end - start)
        return result

    def avg_idf(self, query_tokens: list[str]) -> float:
        """Average IDF of query tokens; lower = harder query."""
        values = []
        N = self.index.N
        freqs = self._batch_doc_freq(set(query_tokens))
        for token, df in freqs.items():
            if df:
                values.append(self.scorer._idf(df, N))
        return sum(values) / len(values) if values else 0.0

    def max_idf(self, query_tokens: list[str]) -> float:
        values = []
        N = self.index.N
        freqs = self._batch_doc_freq(set(query_tokens))
        for token, df in freqs.items():
            if df:
                values.append(self.scorer._idf(df, N))
        return max(values) if values else 0.0

    def query_scope(self, query_tokens: list[str]) -> float:
        """Fraction of documents containing at least one query token."""
        if self.index.N == 0:
            return 0.0
        all_docs = []
        for t in set(query_tokens):
            postings = self.index.get_postings(t)
            if postings is not None:
                all_docs.append(postings[0])
        if not all_docs:
            return 0.0
        concat = np.concatenate(all_docs)
        unique = np.unique(concat)
        return len(unique) / self.index.N

    def is_hard_query(self, query_tokens: list[str], avg_idf_threshold: float = 2.0) -> bool:
        """Heuristic: query is "hard" if avg IDF is below threshold."""
        return self.avg_idf(query_tokens) < avg_idf_threshold


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------


def hamming_distance(s: str, t: str) -> int:
    """Hamming distance between two equal-length strings."""
    if len(s) != len(t):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(1 for a, b in zip(s, t) if a != b)


# ---------------------------------------------------------------------------
# Cosine similarity (TF-IDF vectors)
# ---------------------------------------------------------------------------


def cosine_similarity_tfidf(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors (dict: term -> weight)."""
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term, w in vec_a.items():
        norm_a += w * w
        if term in vec_b:
            dot += w * vec_b[term]
    for w in vec_b.values():
        norm_b += w * w
    denom = (norm_a ** 0.5) * (norm_b ** 0.5)
    return 0.0 if denom == 0 else dot / denom


# ---------------------------------------------------------------------------
# Soundex
# ---------------------------------------------------------------------------


def soundex(word: str) -> str:
    """Classic Soundex phonetic encoding (4-char code)."""
    word = word.upper()
    if not word:
        return ""
    first = word[0]
    # Map letters to digits
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2",
        "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    digits = [first]
    prev = mapping.get(first, "")
    for ch in word[1:]:
        d = mapping.get(ch, "")
        if d:
            if d != prev:
                digits.append(d)
            prev = d
        else:
            prev = ""
    # Remove vowels/H/W after first letter (already handled by mapping)
    result = "".join(digits)
    result = result[0] + result[1:].replace("0", "")
    result = result + "000"
    return result[:4]


# ---------------------------------------------------------------------------
# Metaphone
# ---------------------------------------------------------------------------


def metaphone(word: str) -> str:
    """Basic Metaphone phonetic encoding for English words."""
    word = word.upper()
    if not word:
        return ""
    # Simplified rules
    result: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        next_ch = word[i + 1] if i + 1 < n else ""
        next2 = word[i + 2] if i + 2 < n else ""
        prev = word[i - 1] if i > 0 else ""

        if ch in "AEIOU":
            if i == 0:
                result.append(ch)
            i += 1
            continue
        if ch == "B":
            if not (i == n - 1 and prev == "M"):
                result.append("B")
            i += 1
        elif ch == "C":
            if next_ch == "H" and prev != "S":
                result.append("X")
                i += 2
            elif next_ch == "I" and next2 == "A":
                result.append("X")
                i += 3
            elif next_ch in "IEY":
                result.append("S")
                i += 2
            else:
                result.append("K")
                i += 1
        elif ch == "D":
            if next_ch == "G" and next2 in "IEY":
                result.append("J")
                i += 3
            else:
                result.append("T")
                i += 1
        elif ch == "F":
            result.append("F")
            i += 1
        elif ch == "G":
            if next_ch == "H":
                if i > 0 and word[i - 1] not in "AEIOU":
                    result.append("K")
                i += 2
            elif next_ch == "N":
                if i == n - 2:
                    pass
                else:
                    result.append("N")
                i += 2
            elif next_ch == "E" and next2 == "L":
                result.append("K")
                i += 3
            elif next_ch == "I" and next2 == "O":
                result.append("J")
                i += 3
            elif next_ch in "IEY":
                result.append("J")
                i += 2
            else:
                result.append("K")
                i += 1
        elif ch == "H":
            if prev in "AEIOU" or next_ch in "AEIOU":
                result.append("H")
            i += 1
        elif ch == "J":
            result.append("J")
            i += 1
        elif ch == "K":
            if prev != "C":
                result.append("K")
            i += 1
        elif ch == "L":
            result.append("L")
            i += 1
        elif ch == "M":
            result.append("M")
            i += 1
        elif ch == "N":
            result.append("N")
            i += 1
        elif ch == "P":
            if next_ch == "H":
                result.append("F")
                i += 2
            else:
                result.append("P")
                i += 1
        elif ch == "Q":
            result.append("K")
            i += 1
        elif ch == "R":
            result.append("R")
            i += 1
        elif ch == "S":
            if next_ch == "H" or (next_ch == "I" and next2 in "OA"):
                result.append("X")
                i += 2 if next_ch == "H" else 3
            else:
                result.append("S")
                i += 1
        elif ch == "T":
            if next_ch == "I" and next2 in "OA":
                result.append("X")
                i += 3
            elif next_ch == "H":
                result.append("0")
                i += 2
            elif next_ch == "C" and next2 == "H":
                result.append("")
                i += 1
            else:
                result.append("T")
                i += 1
        elif ch == "V":
            result.append("F")
            i += 1
        elif ch == "W":
            if next_ch in "AEIOU":
                result.append("W")
            i += 1
        elif ch == "X":
            result.append("KS")
            i += 1
        elif ch == "Y":
            if next_ch in "AEIOU":
                result.append("Y")
            i += 1
        elif ch == "Z":
            result.append("S")
            i += 1
        else:
            i += 1

    # Remove duplicate adjacent letters
    final = []
    for c in result:
        for sub in c:
            if not final or sub != final[-1]:
                final.append(sub)
    return "".join(final)


# ---------------------------------------------------------------------------
# Porter Stemmer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=65536)
def porter_stem(word: str) -> str:
    """Simplified Porter Stemmer for English words."""
    word = word.lower()
    if len(word) <= 2:
        return word

    def _is_vowel(ch: str, prev: str = "") -> bool:
        if ch in "aeiou":
            return True
        return ch == "y" and prev not in "aeiou"

    def _measure(stem: str) -> int:
        n = len(stem)
        if n == 0:
            return 0
        # Replace y at start with vowel marker
        seq = []
        for i, ch in enumerate(stem):
            prev = stem[i - 1] if i > 0 else ""
            seq.append("V" if _is_vowel(ch, prev) else "C")
        # Count VC transitions
        m = 0
        prev = seq[0]
        for s in seq[1:]:
            if prev == "C" and s == "V":
                m += 1
            prev = s
        return m

    def _ends_with(stem: str, suffix: str) -> bool:
        return stem.endswith(suffix)

    def _replace_suffix(stem: str, suffix: str, repl: str) -> str:
        return stem[: -len(suffix)] + repl

    # Step 1a
    if _ends_with(word, "sses"):
        word = _replace_suffix(word, "sses", "ss")
    elif _ends_with(word, "ies"):
        word = _replace_suffix(word, "ies", "i")
    elif _ends_with(word, "ss"):
        pass
    elif _ends_with(word, "s"):
        word = word[:-1]

    # Step 1b
    step1b_done = False
    if _ends_with(word, "eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            word = stem + "ee"
    elif _ends_with(word, "ed"):
        stem = word[:-2]
        if any(_is_vowel(c, stem[i - 1] if i > 0 else "") for i, c in enumerate(stem)):
            word = stem
            step1b_done = True
    elif _ends_with(word, "ing"):
        stem = word[:-3]
        if any(_is_vowel(c, stem[i - 1] if i > 0 else "") for i, c in enumerate(stem)):
            word = stem
            step1b_done = True

    if step1b_done:
        if _ends_with(word, "at"):
            word = word + "e"
        elif _ends_with(word, "bl"):
            word = word + "e"
        elif _ends_with(word, "iz"):
            word = word + "e"
        elif (
            len(word) >= 2
            and word[-1] not in "aeiou"
            and word[-1] == word[-2]
            and word[-1] not in "lsz"
        ):
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word = word + "e"

    # Step 1c
    if word.endswith("y"):
        stem = word[:-1]
        if any(_is_vowel(c, stem[i - 1] if i > 0 else "") for i, c in enumerate(stem)):
            word = stem + "i"

    # Step 2
    step2_map = {
        "ational": "ate",
        "tional": "tion",
        "enci": "ence",
        "anci": "ance",
        "izer": "ize",
        "abli": "able",
        "alli": "al",
        "entli": "ent",
        "eli": "e",
        "ousli": "ous",
        "ization": "ize",
        "ation": "ate",
        "ator": "ate",
        "alism": "al",
        "iveness": "ive",
        "fulness": "ful",
        "ousness": "ous",
        "aliti": "al",
        "iviti": "ive",
        "biliti": "ble",
    }
    for suffix, repl in step2_map.items():
        if _ends_with(word, suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                word = stem + repl
            break

    # Step 3
    step3_map = {
        "icate": "ic",
        "ative": "",
        "alize": "al",
        "iciti": "ic",
        "ical": "ic",
        "ful": "",
        "ness": "",
    }
    for suffix, repl in step3_map.items():
        if _ends_with(word, suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                word = stem + repl
            break

    # Step 4
    step4_suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize",
    ]
    for suffix in step4_suffixes:
        if _ends_with(word, suffix):
            if suffix == "ion" and len(word) > 3 and word[-4] in "st":
                stem = word[:-3]
                if _measure(stem) > 1:
                    word = stem
                break
            stem = word[: -len(suffix)]
            if _measure(stem) > 1:
                word = stem
            break

    # Step 5a
    if _ends_with(word, "e"):
        stem = word[:-1]
        if _measure(stem) > 1:
            word = stem
        elif _measure(stem) == 1 and not _ends_cvc(stem):
            word = stem

    # Step 5b
    if (
        _measure(word) > 1
        and len(word) >= 2
        and word[-1] == word[-2]
        and word[-1] == "l"
    ):
        word = word[:-1]

    return word


def _ends_cvc(word: str) -> bool:
    """Check if word ends with consonant-vowel-consonant (with special cases)."""
    if len(word) < 3:
        return False
    a, b, c = word[-3], word[-2], word[-1]
    if c in "wxy":
        return False
    vowels = "aeiou"
    return a not in vowels and b in vowels and c not in vowels


# ---------------------------------------------------------------------------
# MinHash – near-duplicate detection via shingling
# ---------------------------------------------------------------------------


class MinHash:
    """MinHash signature for Jaccard similarity estimation."""

    __slots__ = ("num_perm", "signature")
    _SEEDS_CACHE: dict[int, np.ndarray] = {}

    def __init__(self, text: str = "", num_perm: int = 128, k: int = 3) -> None:
        self.num_perm = num_perm
        self.signature = self._compute(text, k)

    @classmethod
    def _get_seeds(cls, n: int) -> np.ndarray:
        if n not in cls._SEEDS_CACHE:
            cls._SEEDS_CACHE[n] = np.array([hash(i) & 0xFFFFFFFF for i in range(n)], dtype=np.uint32)
        return cls._SEEDS_CACHE[n]

    @staticmethod
    def _shingles(text: str, k: int) -> set[str]:
        """Generate k-character shingles."""
        text = text.lower()
        if len(text) < k:
            return {text} if text else set()
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _compute(self, text: str, k: int) -> list[int]:
        shingles = self._shingles(text, k)
        if not shingles:
            return [0] * self.num_perm
        shingle_hashes = np.fromiter(
            (hash(s) & 0xFFFFFFFF for s in shingles),
            dtype=np.uint32,
            count=len(shingles),
        )
        seeds = self._get_seeds(self.num_perm)
        # Broadcast XOR: (num_perm, 1) ^ (len(shingles),) -> (num_perm, len(shingles))
        sig = np.min(shingle_hashes ^ seeds[:, None], axis=1).tolist()
        return sig

    def jaccard(self, other: MinHash) -> float:
        """Estimated Jaccard similarity between two MinHash signatures."""
        if self.num_perm != other.num_perm:
            raise ValueError("MinHash signatures must have same num_perm")
        matches = sum(1 for a, b in zip(self.signature, other.signature) if a == b)
        return matches / self.num_perm


# ---------------------------------------------------------------------------
# Query Performance Prediction extensions
# ---------------------------------------------------------------------------


def _query_lm(query_tokens: list[str]) -> dict[str, float]:
    """Build a simple language model from query tokens."""
    if not query_tokens:
        return {}
    total = len(query_tokens)
    lm: dict[str, float] = {}
    for t in query_tokens:
        lm[t] = lm.get(t, 0.0) + 1.0 / total
    return lm


def _collection_lm(index: InvertedIndex) -> dict[str, float]:
    """Estimate collection language model from index statistics."""
    if index._collection_lm_cache is not None:
        return index._collection_lm_cache.copy()
    total_tokens = sum(index.doc_lengths)
    if total_tokens == 0:
        return {}
    clm: dict[str, float] = {}
    for term in index.terms():
        postings = index.get_postings(term)
        if postings is None:
            continue
        _, tfs = postings
        freq = int(tfs.sum())
        clm[term] = freq / total_tokens
    return clm


def clarity_score(index: InvertedIndex, query_tokens: list[str]) -> float:
    """Query clarity: KL divergence between query LM and collection LM.

    Higher clarity = query is more focused / easier.
    """
    if not query_tokens or index.N == 0:
        return 0.0
    q_lm = _query_lm(query_tokens)
    c_lm = _collection_lm(index)
    if not c_lm:
        return 0.0
    score = 0.0
    for term, p in q_lm.items():
        q = c_lm.get(term, 1e-12)
        if q > 0 and p > 0:
            score += p * math.log(p / q)
    return score


# ---------------------------------------------------------------------------
# RankSVM – pairwise ranking with linear SVM
# ---------------------------------------------------------------------------


class RankSVM:
    """Simplified RankSVM using pairwise transformations and SGD."""

    __slots__ = ("learning_rate", "n_iterations", "weights", "margin")

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, margin: float = 1.0) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: list[float] = []
        self.margin = margin

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Fit on document-feature vectors with relevance grades."""
        if not X:
            return
        n_features = len(X[0])
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0.0, 0.01, n_features).tolist()
        lr = self.learning_rate
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        # Build pairwise indices for docs with different grades
        pair_i: list[int] = []
        pair_j: list[int] = []
        pair_label: list[int] = []
        n = len(X)
        for i in range(n):
            for j in range(i + 1, n):
                if y_arr[i] != y_arr[j]:
                    pair_i.append(i)
                    pair_j.append(j)
                    pair_label.append(1 if y_arr[i] > y_arr[j] else -1)
        if not pair_i:
            return
        pair_i_arr = np.asarray(pair_i, dtype=np.int64)
        pair_j_arr = np.asarray(pair_j, dtype=np.int64)
        pair_label_arr = np.asarray(pair_label, dtype=np.int64)
        w_arr = np.asarray(self.weights, dtype=np.float64)
        for _ in range(self.n_iterations):
            order = rng.permutation(len(pair_i_arr))
            for idx in order:
                i = pair_i_arr[idx]
                j = pair_j_arr[idx]
                label = pair_label_arr[idx]
                diff = X_arr[i] - X_arr[j]
                margin = float(label) * np.dot(diff, w_arr)
                if margin < self.margin:
                    w_arr += lr * float(label) * diff
        self.weights = w_arr.tolist()

    def predict(self, X: list[list[float]]) -> list[float]:
        if not self.weights:
            return [0.0] * len(X)
        w_arr = np.asarray(self.weights, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr.dot(w_arr).tolist()

    def rank(self, doc_features: list[tuple[int, list[float]]]) -> list[tuple[int, float]]:
        ids, feats = zip(*doc_features) if doc_features else ([], [])
        scores = self.predict(list(feats))
        return sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# Coordinate Ascent – listwise LTR
# ---------------------------------------------------------------------------


CoordinateAscent = LambdaMART




class NoisyChannelSpeller:
    """Simple noisy-channel spell corrector using edit distance + unigram frequency."""

    __slots__ = ("dictionary", "max_edits", "_dict_keys", "_correct_cache")

    def __init__(self, dictionary: dict[str, int], max_edits: int = 2) -> None:
        self.dictionary = dictionary
        self.max_edits = max_edits
        self._dict_keys = set(dictionary)
        self._correct_cache: dict[str, str] = {}

    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def _edits1(word: str) -> frozenset[str]:
        letters = "abcdefghijklmnopqrstuvwxyz"
        edits: set[str] = {word}
        wlen = len(word)
        for i in range(wlen + 1):
            a = word[:i]
            b = word[i:]
            blen = len(b)
            if blen:
                # delete
                edits.add(a + b[1:])
                if blen > 1:
                    # transpose
                    edits.add(a + b[1] + b[0] + b[2:])
                # replace
                for c in letters:
                    edits.add(a + c + b[1:])
            # insert
            for c in letters:
                edits.add(a + c + b)
        return frozenset(edits)

    def _candidates(self, word: str) -> frozenset[str]:
        candidates = self._edits1(word)
        if self.max_edits >= 2:
            candidates = candidates | frozenset(
                e2 for w in candidates for e2 in self._edits1(w)
            )
        return candidates & self._dict_keys or candidates

    def correct(self, word: str) -> str:
        """Return the most likely correction for *word*."""
        word = word.lower()
        cached = self._correct_cache.get(word)
        if cached is not None:
            return cached
        if word in self.dictionary:
            self._correct_cache[word] = word
            return word
        candidates = self._candidates(word) & self._dict_keys
        if not candidates:
            self._correct_cache[word] = word
            return word
        result = max(candidates, key=lambda w: self.dictionary[w])
        self._correct_cache[word] = result
        return result


# ---------------------------------------------------------------------------
# Query Performance Prediction extensions
# ---------------------------------------------------------------------------


def scq(index: InvertedIndex, query_tokens: list[str]) -> float:
    """Collection Query Similarity (SCQ) – average of IDF * TF over query terms."""
    if not query_tokens or index.N == 0:
        return 0.0
    total = 0.0
    count = 0
    for token in set(query_tokens):
        postings = index.get_postings(token)
        if postings is None:
            continue
        docs, tfs = postings
        df = len(docs)
        cf = int(tfs.sum())
        idf = BM25Scorer._idf(df, index.N)
        total += idf * math.log(1 + cf)
        count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Rocchio Relevance Feedback
# ---------------------------------------------------------------------------


class RocchioExpander:
    """Rocchio pseudo-relevance feedback query expansion."""

    __slots__ = ("index", "scorer", "alpha", "beta", "gamma", "fb_docs", "fb_terms")

    def __init__(
        self,
        index: InvertedIndex,
        scorer: BM25Scorer,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
        fb_docs: int = 3,
        fb_terms: int = 10,
    ) -> None:
        self.index = index
        self.scorer = scorer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms

    def expand(self, query_tokens: list[str], non_rel_docs: set[int] | None = None) -> list[str]:
        """Return expanded query tokens using Rocchio formula."""
        if not query_tokens:
            return []
        results = self.scorer.score_topk(query_tokens, top_k=self.fb_docs)
        if not results:
            return query_tokens
        rel_docs = {doc_id for doc_id, _ in results}
        non_rel = non_rel_docs or set()
        # Build centroid vectors via forward index
        term_scores: dict[str, float] = {}
        for doc_id in rel_docs:
            if doc_id < len(self.index._doc_term_freqs):
                for term, tf in self.index._doc_term_freqs[doc_id].items():
                    term_scores[term] = term_scores.get(term, 0.0) + self.beta * tf / max(len(rel_docs), 1)
        for doc_id in non_rel:
            if doc_id < len(self.index._doc_term_freqs):
                for term, tf in self.index._doc_term_freqs[doc_id].items():
                    term_scores[term] = term_scores.get(term, 0.0) - self.gamma * tf / max(len(non_rel), 1)
        # Original query component
        expanded: dict[str, float] = {}
        for token in query_tokens:
            expanded[token] = expanded.get(token, 0.0) + self.alpha
        # Add top feedback terms
        top_terms = heapq.nlargest(self.fb_terms, term_scores.items(), key=lambda x: x[1])
        for term, score in top_terms:
            if score > 0:
                expanded[term] = expanded.get(term, 0.0) + score
        # Flatten by weight
        result: list[str] = []
        for term, weight in expanded.items():
            count = max(1, int(weight * 10))
            result.extend([term] * count)
        return result


# ---------------------------------------------------------------------------
# RankBoost – pairwise AdaBoost for ranking
# ---------------------------------------------------------------------------


class RankBoost:
    """Simplified RankBoost using pairwise preferences."""

    __slots__ = ("n_iterations", "weights", "weak_learners")

    def __init__(self, n_iterations: int = 100) -> None:
        self.n_iterations = n_iterations
        self.weights: list[float] = []
        self.weak_learners: list[tuple[int, float, float]] = []

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        if not X or not X[0]:
            return
        n = len(X)
        n_features = len(X[0])
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        # Pairwise distribution
        pairs_i: list[int] = []
        pairs_j: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                if y_arr[i] != y_arr[j]:
                    if y_arr[i] > y_arr[j]:
                        pairs_i.append(i)
                        pairs_j.append(j)
                    else:
                        pairs_i.append(j)
                        pairs_j.append(i)
        if not pairs_i:
            return
        pair_i = np.asarray(pairs_i, dtype=np.int64)
        pair_j = np.asarray(pairs_j, dtype=np.int64)
        m = len(pair_i)
        D_arr = np.full(m, 1.0 / m, dtype=np.float64)
        for _ in range(self.n_iterations):
            best_dim = -1
            best_err = float("inf")
            best_threshold = 0.0
            best_direction = 1.0
            # Sample candidate thresholds
            for dim in range(n_features):
                vals = np.unique(X_arr[:, dim])
                if len(vals) <= 1:
                    thresholds = [0.0]
                else:
                    thresholds = (vals[:-1] + vals[1:]) / 2.0
                for thresh in thresholds.tolist():
                    for direction in (1.0, -1.0):
                        if direction == 1.0:
                            wrong = (X_arr[pair_i, dim] < thresh) | (X_arr[pair_j, dim] >= thresh)
                        else:
                            wrong = (X_arr[pair_i, dim] > thresh) | (X_arr[pair_j, dim] <= thresh)
                        err = float(D_arr.dot(wrong))
                        if err < best_err:
                            best_err = err
                            best_dim = dim
                            best_threshold = thresh
                            best_direction = direction
            if best_err >= 0.5 or best_err == 0:
                break
            alpha_t = 0.5 * math.log((1.0 - best_err) / max(best_err, 1e-12))
            # Update distribution
            if best_direction == 1.0:
                correct_mask = (X_arr[pair_i, best_dim] >= best_threshold) & (X_arr[pair_j, best_dim] < best_threshold)
            else:
                correct_mask = (X_arr[pair_i, best_dim] <= best_threshold) & (X_arr[pair_j, best_dim] > best_threshold)
            correct = np.where(correct_mask, 1.0, -1.0)
            w = D_arr * np.exp(-alpha_t * correct)
            Z = float(w.sum())
            if Z == 0:
                break
            D_arr = w / Z
            self.weak_learners.append((best_dim, best_threshold, best_direction * alpha_t))

    def predict(self, X: list[list[float]]) -> list[float]:
        if not self.weak_learners:
            return [0.0] * len(X)
        X_arr = np.asarray(X, dtype=np.float64)
        scores = np.zeros(len(X), dtype=np.float64)
        for dim, thresh, alpha in self.weak_learners:
            scores += alpha * np.where(X_arr[:, dim] >= thresh, 1.0, -1.0)
        return scores.tolist()

    def rank(self, doc_features: list[tuple[int, list[float]]]) -> list[tuple[int, float]]:
        ids, feats = zip(*doc_features) if doc_features else ([], [])
        scores = self.predict(list(feats))
        return sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# xQuAD – explicit query aspect diversification
# ---------------------------------------------------------------------------


def xquad_rerank(
    results: list[tuple[int, float]],
    aspects: dict[int, set[str]],
    lambda_param: float = 0.5,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Greedy xQuAD re-ranking.

    *aspects* maps doc_id -> set of aspect labels.
    """
    if not results:
        return []
    if top_k is None:
        top_k = len(results)

    covered: set[str] = set()
    selected: list[tuple[int, float]] = []
    remaining = list(results)

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_doc: tuple[int, float] | None = None
        best_score = -float("inf")
        for idx, (doc_id, relevance) in enumerate(remaining):
            doc_aspects = aspects.get(doc_id, set())
            new_aspects = doc_aspects - covered
            diversity = len(new_aspects) / max(len(doc_aspects), 1) if doc_aspects else 0.0
            score = (1.0 - lambda_param) * relevance + lambda_param * diversity
            if score > best_score:
                best_score = score
                best_doc = (doc_id, relevance)
                best_idx = idx
        if best_doc is None:
            break
        selected.append(best_doc)
        covered |= aspects.get(best_doc[0], set())
        remaining.pop(best_idx)
    return selected


# ---------------------------------------------------------------------------
# I-Match – lexical fingerprint near-duplicate detection
# ---------------------------------------------------------------------------


def i_match_fingerprint(tokens: list[str], stopwords: set[str] | None = None) -> str:
    """Return an I-Match fingerprint (sorted unique non-stop tokens joined)."""
    stop = stopwords or set()
    unique = sorted({t for t in tokens if t not in stop})
    return " ".join(unique)
