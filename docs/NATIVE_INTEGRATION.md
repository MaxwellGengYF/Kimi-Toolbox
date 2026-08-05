# Native Integration (kimix-base `runtime_py` → kimi-agent)

The performance-critical pure-Python code in this project can be accelerated
by the native C++ runtime compiled in the
[kimix-base](https://github.com/…) repo (a sibling of this checkout by
default, or wherever `KIMIX_BASE` points) (`runtime_py.pyd` + `runtime.dll`,
pybind11 extension with submodules `text` / `index` / `search` / `parse` /
`soul` / `tools` / `stream` / `codec` / `json` / `concurrency`).

**The native library is an OPTIONAL acceleration path.** Every integrated
kernel keeps its original pure-Python body: when the binaries are missing (or
`KIMIX_NATIVE=0`), the existing Python code executes unchanged with
bit-identical outputs.

## How it works

1. **Build** the native library in kimix-base (``<kimix-base>`` is the repo
   root — the `KIMIX_BASE` env var when set, otherwise the ``kimix-base``
   sibling of this checkout):

   ```bash
   cd <kimix-base>
   python bootstrap.py            # release build (artifacts in bin\release)
   # python bootstrap.py --debug  # debug build (bin\debug)
   ```

2. **Stage** the binaries into this project's work-dir (``<repo>`` = this
   checkout's root):

   ```bash
   cd <repo>
   python tools\sync_native.py              # copies from kimix-base\bin\release
   python tools\sync_native.py --mode debug # force a build mode
   python tools\sync_native.py --mode auto  # newest valid build
   ```

   `sync_native.py` copies `runtime_py.pyd` + `runtime.dll` (plus any runtime
   DLL deps) into `<repo>\bin\` — the default native path. The
   `kimix_native` shim package is tracked by git (not ignored), so it is never
   synced. Idempotent; run it before every test/bench run.

3. **Load**: `kimix/native_loader.py` (re-exported by
   `kimi_cli/native_loader.py`) resolves the shim in this order:

   | priority | location |
   |---|---|
   | 1 | `KIMIX_NATIVE_PATH` env var (explicit override, only dir tried) |
   | 2 | **`<repo>\bin`** — the sync destination (default) |
   | 3 | already importable on `sys.path` (pip-installed `kimix-native`) |
   | 4 | dev-only last resort: `<kimix-base>\bin\{release,releasedbg,debug}` — where `<kimix-base>` is `$KIMIX_BASE` or the `kimix-base` sibling of this repo |

   The first usable directory is inserted on `sys.path` and the `kimix_native`
   shim is imported once (cached). No absolute cross-repo path is baked into
   the running project for the default path.

## Gate hot-path optimization

`use_native(kernel)` and `get_module(name)` are called on every hot-path
invocation (each token count / sanitize / ANSI-strip / parser call). Both
loaders cache their resolutions so the steady-state cost is a single dict
lookup per call:

- `kimix/native_loader.py` + `kimi-cli/native_loader.py`: per-kernel gate
  decisions (`_kernel_cache`) and per-module imports (`_module_cache`) are
  resolved once and cached; `kernel_module(kernel)` is a combined one-lookup
  accessor for new hot code.
- Env toggles are fixed at process start; tests toggle them via fresh
  subprocesses (mode-matrix / per-kernel-toggle suites) or by monkeypatching
  the loader functions in-process — the caches are per-process and never
  consulted after a monkeypatch replaces the function.
- The shim (`kimix_native`) itself stays uncached on purpose: kimix-base's
  parity tests flip `KIMIX_NATIVE` **in-process**, which would go stale.

Measured on this machine (release build, 200k iterations):
`use_native` 0.81 → 0.10 µs, `get_module` 0.47 → 0.07 µs, gate+module pair
1.28 → 0.16 µs per call (~8×).

## Env toggles (mode matrix)

| env | behavior |
|---|---|
| `KIMIX_NATIVE=0` | pure Python everywhere (never imports the .pyd) |
| `KIMIX_NATIVE=1` | **require** native; `ImportError` if the .pyd is missing |
| `KIMIX_NATIVE=auto` (default) | native when importable, Python fallback otherwise |
| `KIMIX_NATIVE_<KERNEL>=0` | disable one kernel (TEXT/INDEX/SEARCH/PARSE/SOUL/TOOLS/STREAM/CODEC/JSON/CONCURRENCY) |
| `KIMIX_BASE=<dir>` | kimix-base repo root for the dev-only fallback & `sync_native.py` (default: the `kimix-base` sibling of this checkout) |

Per-kernel toggles make every kernel switchable and reversible without code
changes — each kernel has a bit-identical Python fallback.

## Integration pattern

Each hot function keeps its old body and gains a lazy gate:

```python
from kimix.native_loader import get_module as _native_get_module, use_native as _native_use_native

def count_tokens(text, model=None):
    if _native_use_native("TEXT"):
        _mod = _native_get_module("text")
        if _mod is not None:
            return _mod.count_tokens(...)
    <existing pure-Python body unchanged>
```

## Integrated kernels (equivalence-tested)

| kernel | app function(s) | native shim | benchmark |
|---|---|---|---|
| TEXT | `kimi_cli/utils/tokens.py` `_estimate_chars_tokens`/`_is_cjk_text`, `kimi_cli/safety_check.py` `clean_text`/`sanitize_for_tokenizer` | `text.*` | sanitize 1 MB **27.3x** |
| STREAM | `src/kimix/tools/common.py` `filter_output`/`_dedup_output` | `stream.filter_output` / `LineProcessor` | 1 MB ANSI strip **5.8x** |
| TOOLS | `src/kimix/tools/file/find_str.py` `find_in_file`; `grep_local.py` `_search_content_single` (scan_lines_cb); `utils/export.py` `build_export_markdown` (dict bridge) | `tools.find_in_file`/`scan_lines_cb`/`build_export_markdown` | — |
| PARSE | all 7 comment parsers via `parser/base.py::native_parse_result`; `bash_fix.fix_bash_command`; `bash_tool._process_unquoted`; `pwsh_fix.fix_pwsh_command` | `parse.parse`/`fix_bash_command`/`_process_unquoted`/`fix_pwsh_command` | parse C 1 MB **2.1x** |
| INDEX | `src/kimix/retrieval.py` `NgramTokenizer` (normalize/detect_n/tokenize) | `index.NgramTokenizer` | — |
| SEARCH | `retrieval.py` `jaro_similarity`/`jaro_winkler_similarity`/`sorensen_dice_coefficient`/`ngram_overlap`/`LevenshteinAutomaton._damerau_levenshtein`/`_freq_lower_bound` | `search.*` | — |
| JSON | `kimi_cli/acp/session.py` `_ToolCallState` incremental tool-args lexer | `json.IncrementalJsonLexer` | — |

Equivalence tests: `tests/native/` (104 tests) + `kimi-cli/tests/native/`
(159 tests) — every kernel is run through the SAME corpus with the gate
forced on vs off and outputs asserted identical (return values, bytes,
errors, determinism, thread-safety smoke).

## Deliberately NOT integrated (and why)

| app code | native shim | reason |
|---|---|---|
| `InvertedIndex` internals (`retrieval.py`) | `index.InvertedIndex` | the app's `finalize()` prunes stop-ngrams (threshold-based) by default; the native index has no pruning; `save`/`load` formats (file vs KNIDX1 blob) and forward-index APIs differ |
| `kimi_cli/soul/history_index.py` | `index.HistoryIndex` | depends on the app InvertedIndex + Searcher (pruning) and persists JSON turn metadata; native persists a KNHIX1 blob |
| `BM25Scorer` internals | `search.bm25_*` | already numpy-vectorized; function contracts (postings arrays vs lists) differ; negligible gain |
| `SimHash._compute`/`MinHash._compute` | `search.simhash`/`minhash` | the native kernels use a fixed XXH3-64 seed contract; the app uses `xxhash.xxh64` — outputs would CHANGE (behavior break) |
| `mmr_rerank`/`xquad_rerank` | `search.mmr_rerank`/`xquad_rerank` | the native API consumes a precomputed sim_matrix and returns indices; the app derives Jaccard sets from the forward index (contract mismatch) |
| `hash_line.compute_line_hash` | `tools.line_hash` | per-line calls lose to the Python→C boundary (0.5x vs `xxhash.xxh32`, itself a C extension). The bulk kernels `tools.line_hashes`/`compute_line_hashes` remain available for whole-file hashing |
| wire merge/serde (`wire/__init__.py`, `serde.py`, `file.py`) | `codec.WireMergeBuffer`/envelopes/`JsonlRecorder` | the wire layer is pydantic-model-centric (`MergeableMixin` deep-copy semantics, envelope `model_dump`/`model_validate`); the codec is bytes-in/bytes-out — mapping would change object semantics |
| TCP recv (`network/tcp_*.py`) | `codec.RecvBuffer` | `_recv_all` is a socket read loop; `RecvBuffer` is a byte accumulator for a different data flow |
| SSE (`server/bus.py`) | `codec.build_sse_frame` / `concurrency.IdGenerator` | opencode ids are `evt_<hex>` strings; the native frame builder / id generator use int ids (contract mismatch) |
| SOUL kernels (`context_pruning.prune_scan`, `compaction.SimpleCompaction.prepare`, `message.strip_system_reminders`, `dynamic_injection.normalize_history`, `kosong/chat_provider/kimi.py` payload) | `soul.*` | the shim operates on plain-dict plans/bytes and covers only default option sets (balanced mode, no custom sections); the app flows operate on pydantic `Message` objects with option-dependent behavior and must keep their part-structure — keep Python as source of truth until per-case parity is proven (plan P4/P5, optional) |
| `process_pwsh.pwsh_transform` | `parse.pwsh_transform` | native returns **empty warnings** by design; the app surfaces warning strings — behavioral difference (documented in the shim) |

## Compatibility caveats

- **DLL shipping**: `runtime.dll` must sit next to `runtime_py.pyd`.
  `tools\sync_native.py` always copies both together, so the staged copy is
  consistent by construction. `KIMIX_NATIVE=1` + missing DLL → `ImportError`
  (documented contract); `auto` → silent Python fallback.
- **Python/arch mismatch** (e.g. 3.13 pyd on 3.14): `ImportError` → fallback.
- **Hash determinism**: native hash kernels use a fixed XXH3-64 seed (not
  `PYTHONHASHSEED`). `SimHash`/`MinHash` are not wired (see above) — no
  output change in this repo.
- **Benchmark tradeoff**: small per-call kernels can be *slower* native due to
  the Python→C boundary (see the NOT integrated table). Bulk kernels win big.
- `bin\` is gitignored except the tracked `bin\kimix_native\` shim; re-run
  `tools\sync_native.py` after every kimix-base rebuild to refresh the
  binaries.

## Adding a new kernel

1. Wire the gate in the app function (pattern above).
2. Add equivalence cases to `tests/native/test_behavior_equivalence.py` (and
   `kimi-cli/tests/native/…`) with adversarial corpus inputs.
3. Run `python tools\sync_native.py` then the native suites in both
   `KIMIX_NATIVE=0` and `auto` modes.
4. A kernel whose equivalence test is red or absent MUST default to
   pure Python (gate off) — do not merge a routing change without its
   equivalence test.
