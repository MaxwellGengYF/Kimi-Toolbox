# Native Integration (kimix-base `runtime_py` → kimi-agent)

The performance-critical pure-Python code in this project can be accelerated
by the native C++ runtime compiled in the
[kimix-base](https://github.com/…) repo (a sibling of this checkout by
default, or wherever `KIMIX_BASE` points) (`runtime_py.pyd`,
pybind11 extension with submodules `text` / `index` / `search` / `parse` /
`soul` / `tools` / `stream` / `codec` / `json` / `concurrency` / `diff` /
`glob` / `image` / `todo` / `workspace`).

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

   `sync_native.py` copies `runtime_py.pyd` (plus any runtime DLL deps) into
   `<repo>\bin\` — the default native path. The
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
| `KIMIX_NATIVE_<KERNEL>=0` | disable one kernel (TEXT / INDEX / SEARCH / PARSE / SOUL / TOOLS / STREAM / CODEC / JSON / CONCURRENCY / DIFF / GLOB / IMAGE / TODO / WORKSPACE) |
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

## Wired kernels (equivalence-tested)

These kernels are routed from app code to `kimix_native` and have passing
behavior-equivalence tests:

| kernel | app function(s) | native shim | benchmark |
|---|---|---|---|
| TEXT | `kimi_cli/utils/tokens.py` `_estimate_chars_tokens`/`_is_cjk_text`/`count_tokens`, `kimi_cli/safety_check.py` `clean_text`/`sanitize_for_tokenizer` | `text.*` | sanitize 1 MB **27.3x** |
| STREAM | `src/kimix/tools/common.py` `filter_output`/`_dedup_output` | `stream.filter_output` / `LineProcessor` | 1 MB ANSI strip **5.8x** |
| TOOLS | `src/kimix/tools/file/find_str.py` `find_in_file`; `kimi_cli/tools/file/grep_local.py` `scan_lines_cb`; `kimi_cli/utils/export.py` `build_export_markdown` | `tools.find_in_file`/`scan_lines_cb`/`build_export_markdown` | — |
| PARSE | all 7 comment parsers via `parser/base.py::native_parse_result`; `bash_fix.fix_bash_command`; `bash_tool._process_unquoted`; `pwsh_fix.fix_pwsh_command` | `parse.parse`/`fix_bash_command`/`_process_unquoted`/`fix_pwsh_command` | parse C 1 MB **2.1x** |
| INDEX | `src/kimix/retrieval.py` `NgramTokenizer` (normalize/detect_n/tokenize) | `index.NgramTokenizer` | — |
| SEARCH | `src/kimix/retrieval.py` `jaro_similarity`/`jaro_winkler_similarity`/`sorensen_dice_coefficient`/`ngram_overlap`/`LevenshteinAutomaton._damerau_levenshtein`/`_freq_lower_bound` | `search.*` | — |
| JSON | `kimi_cli/acp/session.py` `_ToolCallState` incremental tool-args lexer | `json.IncrementalJsonLexer` | — |
| WORKSPACE | `src/kimix/tools/swarm/best_of_n.py` workspace snapshot/diff/changed-files | `workspace.snapshot`/`diff_snapshots`/`changed_files` | — |
| GLOB | `kimi_cli/tools/file/glob.py` gitignore parsing / single-source-dir matching | `glob.parse_gitignore` / `glob.is_ignored` | — |

Equivalence tests: `tests/native/` (102 tests) + `kimi-cli/tests/native/`
(177 tests) — every wired kernel is run through the SAME corpus with the gate
forced on vs off and outputs asserted identical (return values, bytes,
errors, determinism, thread-safety smoke).

## Additional shim modules (not yet wired to app code)

These modules exist in `bin\kimix_native\` and are usable directly (e.g.
`from kimix_native import codec` or `native_loader.get_module("codec")`), but
the main app paths still use their existing Python implementations:

| module | public API | why it is shim-only |
|---|---|---|
| `codec` | `serialize_envelope`/`deserialize_envelope`, `canonicalize_payload`, `WireMergeBuffer`, `ArgsBuffer`, `JsonRpcFrameWriter`, `JsonlRecorder`, `RecvBuffer`, `build_sse_frame` | the app wire/server layers are pydantic-model-centric; the codec is bytes-in/bytes-out and not yet plumbed in |
| `concurrency` | `MpscEventBus`, `IdGenerator` | the app SSE / bus layers use string ids (`evt_<hex>`) and different event semantics; the native primitives are available but not yet adopted |
| `diff` | `unified_diff`, `diff_hunks`, `inline_diff_ranges` | no app hot path currently consumes these; available for future UI/diff features |
| `image` | `sniff_dimensions`, `read_exif_orientation`, `is_animated_webp`, `format_byte_size` | available for future image-pipeline acceleration |
| `soul` | `build_payload`, `normalize_tool_call_ids`, `prune_scan`, `count_leading_reminders`, `build_normalize_plan`, `build_compaction_prompt`, `apply_normalize_plan`, `apply_id_fixes` | the soul flows operate on pydantic `Message` objects with option-dependent behavior; the shim works on plain-dict plans and is kept for future parity work |
| `todo` | `merge`, `status_counts`, `format_summary` | available for future todo-tool acceleration |

## Deliberately NOT integrated (and why)

| app code | native shim | reason |
|---|---|---|
| `InvertedIndex` internals (`retrieval.py`) | `index.InvertedIndex` | the app's `finalize()` prunes stop-ngrams (threshold-based) by default; the native index has no pruning; `save`/`load` formats (file vs KNIDX1 blob) and forward-index APIs differ. The shim class is exposed but not wired. |
| `kimi_cli/soul/history_index.py` | `index.HistoryIndex` | depends on the app InvertedIndex + Searcher (pruning) and persists JSON turn metadata; native persists a KNHIX1 blob. The shim class is exposed but not wired. |
| `BM25Scorer` internals | `search.bm25_*` | already numpy-vectorized; function contracts (postings arrays vs lists) differ; negligible gain. The shim functions are exposed but not wired. |
| `SimHash._compute`/`MinHash._compute` | `search.simhash`/`minhash` | the native kernels use a fixed XXH3-64 seed contract; the app uses `xxhash.xxh64` — outputs would CHANGE (behavior break). The shim functions are exposed but not wired. |
| `mmr_rerank`/`xquad_rerank` | `search.mmr_rerank`/`xquad_rerank` | the native API consumes a precomputed sim_matrix and returns indices; the app derives Jaccard sets from the forward index (contract mismatch). The shim functions are exposed but not wired. |
| `hash_line.compute_line_hash` | `tools.line_hash` | per-line calls lose to the Python→C boundary (0.5x vs `xxhash.xxh32`, itself a C extension). The bulk kernels `tools.line_hashes`/`compute_line_hashes` are exposed by the shim but not wired to app hot paths. |
| `process_pwsh.pwsh_transform` | `parse.pwsh_transform` | native returns **empty warnings** by design; the app surfaces warning strings — behavioral difference. The shim function is exposed but not wired. |

## Compatibility caveats

- **DLL shipping**: any runtime DLL dependencies must sit next to
  `runtime_py.pyd`. `tools\sync_native.py` copies the `.pyd` plus any sibling
  `.dll` files, so the staged copy is consistent by construction.
  `KIMIX_NATIVE=1` + missing `.pyd` → `ImportError` (documented contract);
  `auto` → silent Python fallback.
- **Python/arch mismatch** (e.g. 3.13 pyd on 3.14): `ImportError` → fallback.
- **Hash determinism**: native hash kernels use a fixed XXH3-64 seed (not
  `PYTHONHASHSEED`). `SimHash`/`MinHash` are not wired (see above) — no
  output change in this repo.
- **Benchmark tradeoff**: small per-call kernels can be *slower* native due to
  the Python→C boundary (see the NOT integrated table). Bulk kernels win big.
- `tools.scan_lines_cb` requires the native extension; it has no pure-Python
  fallback because the regex matcher is supplied by Python caller code and the
  native side only computes line offsets.
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
