# Native Integration (kimix-base `runtime_py` → kimi-agent)

The performance-critical pure-Python code in this project can be accelerated
by the native C++ runtime compiled in the
[kimix-base](https://github.com/…) repo (a sibling of this checkout by
default, or wherever `KIMIX_BASE` points) (`runtime_py.pyd` on Windows /
`runtime_py.so` on Linux & macOS — the platform-dependent suffix of the same
pybind11 extension, with submodules `text` / `index` / `search` / `parse` /
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

   `sync_native.py` copies the platform's compiled extension —
   `runtime_py.pyd` on Windows, `runtime_py.so` on Linux & macOS — (plus any
   runtime DLL deps) into `<repo>\bin\` — the default native path. The
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
invocation (each token count / sanitize / ANSI-strip / parser call). The
runtime environment is stable (env toggles are fixed at process start, the
shim import is cached), so **every per-kernel gate decision and per-module
resolution is computed once at loader import time** and stored in
precomputed tables. The steady-state hot path is a single dict lookup with
no per-call `.upper()` allocation and no repeated availability checks:

- `kimix/native_loader.py`: `_KERNEL_TABLE` (each known kernel under its
  upper/lower/title spellings) and `_MODULE_TABLE` (each known submodule
  resolved to its module object or `None`) are built once by
  `_build_kernel_table()` / `_build_module_table()`; unknown kernel/module
  names fall back to the shim / `importlib` and are memoized into the same
  tables. `kernel_module(kernel)` is a combined one-lookup accessor for new
  hot code.
- `kimi-cli/native_loader.py`: delegates straight to the shared kimix
  loader (its own per-call cache layer was removed — the precomputed tables
  make it redundant); `kernel_module` is exposed eagerly.
- Hot call sites hoist the module resolution to import time
  (`_NATIVE_TEXT = _native_get_module("text")` once per module) and check
  `if _native_use_native("TEXT") and _NATIVE_TEXT is not None:` per item —
  one gate lookup + one attribute check per call instead of two function
  calls.
- Env toggles are fixed at process start; tests toggle them via fresh
  subprocesses (mode-matrix / per-kernel-toggle suites) or by monkeypatching
  the consuming modules' `_native_use_native` binding in-process — the gate
  is still consulted per call, so a monkeypatched gate keeps working.
- The shim (`kimix_native`) itself stays uncached on purpose: kimix-base's
  parity tests flip `KIMIX_NATIVE` **in-process**, which would go stale.

Measured on this machine (release build, 300k iterations,
`tools/bench_native_gate.py`): `use_native` ≈ 0.09 µs and `get_module`
≈ 0.10 µs per call; the old two-call gate+module pattern ≈ 0.35 µs vs the
new hoisted call-site pattern ≈ 0.15 µs (~2×).

## Env toggles (mode matrix)

| env | behavior |
|---|---|
| `KIMIX_NATIVE=0` | pure Python everywhere (never imports the native extension) |
| `KIMIX_NATIVE=1` | **require** native; `ImportError` if the extension (`.pyd`/`.so`) is missing |
| `KIMIX_NATIVE=auto` (default) | native when importable, Python fallback otherwise |
| `KIMIX_NATIVE_<KERNEL>=0` | disable one kernel (TEXT / INDEX / SEARCH / PARSE / SOUL / TOOLS / STREAM / CODEC / JSON / CONCURRENCY / DIFF / GLOB / IMAGE / TODO / WORKSPACE) |
| `KIMIX_BASE=<dir>` | kimix-base repo root for the dev-only fallback & `sync_native.py` (default: the `kimix-base` sibling of this checkout) |

Per-kernel toggles make every kernel switchable and reversible without code
changes — each kernel has a bit-identical Python fallback.

## Integration pattern

Each hot function keeps its old body and gains a lazy gate. New code should
hoist the module resolution to import time and keep the per-item gate check:

```python
from kimix.native_loader import get_module as _native_get_module, use_native as _native_use_native

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TEXT = _native_get_module("text")

def count_tokens(text, model=None):
    if _native_use_native("TEXT") and _NATIVE_TEXT is not None:
        return _NATIVE_TEXT.count_tokens(...)
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
| DIFF | `kimi_cli/utils/diff.py` `format_unified_diff`/`_build_diff_blocks_sync` | `diff.unified_diff`/`diff_hunks` | — |
| IMAGE | `kimi_cli/utils/image_compress.py` `format_byte_size`/`sniff_image_dimensions`/`_is_animated_webp` | `image.format_byte_size`/`sniff_dimensions`/`is_animated_webp` | — |
| CODEC | `kimi_cli/wire/server.py` `_frame_jsonrpc` (JSON-RPC framing); `kimi_cli/wire/file.py` `_dump_line` (jsonl record) | `codec.JsonRpcFrameWriter`/`JsonlRecorder` | — |
| TODO | `kimi_cli/tools/todo/__init__.py` `TodoList._status_counts` | `todo.status_counts` | — |
| SOUL | `kimi_cli/soul/context_pruning.py` `ContextPruner.prune` native fast path | `soul.prune_history` | — |

Equivalence tests: `tests/native/` (130 tests) + `kimi-cli/tests/native/`
(186 tests, incl. `test_additional_kernels_equivalence.py`) — every wired kernel
is run through the SAME corpus with the gate forced on vs off and outputs
asserted identical (return values, bytes, errors, determinism, thread-safety
smoke). kimix-base `python/tests/` adds per-kernel parity tests (incl.
`test_diff.py` / `test_image.py` / `test_todo.py`) and the C++ kernels are
tested by `tests/unit/native/test_{diff,image,todo}.cpp`.

## Additional shim modules (exposed + parity-tested, not wired to app code)

These modules are usable directly (e.g. `from kimix_native import codec` or
`native_loader.get_module("codec")`) and are covered by native↔Python parity
tests, but the main app paths still use their existing Python implementations
because the native contracts do not match the app's data model bit-for-bit:

| module | public API | why it is shim-only |
|---|---|---|
| `concurrency` | `MpscEventBus`, `IdGenerator` | the app SSE / bus layers (`src/kimix/server/bus.py`) use string ids (`evt_<hex>`) and `asyncio.Queue` semantics; `MpscEventBus` is a bounded DROP_OLDEST ring with offset-based subscribers and `IdGenerator` yields ints — wiring them would change the wire format, so they stay shim-only (parity-tested in kimix-base `python/tests/test_concurrency.py`) |
| `codec` | `serialize_envelope`/`deserialize_envelope`, `canonicalize_payload`, `WireMergeBuffer`, `ArgsBuffer`, `RecvBuffer`, `build_sse_frame` | the app's envelope/tool-call paths are pydantic-model-centric; the codec is bytes-in/bytes-out. The JSON-RPC framing (`JsonRpcFrameWriter`) and jsonl record (`JsonlRecorder`) kernels ARE wired (see the table above); the envelope/buffer kernels are exposed for future server work |
| `soul` | `build_payload`, `normalize_tool_call_ids`, `prune_scan`, `count_leading_reminders`, `build_normalize_plan`, `build_compaction_prompt`, `apply_normalize_plan`, `apply_id_fixes` | `prune_history` is wired in `context_pruning.py`; the remaining functions have no matching app call site (the soul flows operate on pydantic `Message` objects with option-dependent behavior; the shim works on plain-dict plans) and stay shim-only for future parity work |
| `diff` | `inline_diff_ranges` | `unified_diff`/`diff_hunks` are wired in `utils/diff.py`; `inline_diff_ranges` targets the rich diff renderer's rendered-text offsets (`_build_offset_map(raw, rendered, tab_size)`) whose contract differs from the shim's plain-string variant — kept shim-only |

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

- **Binary shipping**: any runtime DLL dependencies must sit next to the
  compiled extension (`runtime_py.pyd` on Windows / `runtime_py.so` on
  Linux & macOS). `tools\sync_native.py` copies the extension plus any sibling
  `.dll` files, so the staged copy is consistent by construction.
  `KIMIX_NATIVE=1` + missing extension → `ImportError` (documented contract);
  `auto` → silent Python fallback.
- **Python/arch mismatch** (e.g. 3.13 extension on 3.14): `ImportError` → fallback.
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
