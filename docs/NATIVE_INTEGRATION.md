# Native Integration (kimix-base `runtime_py` → kimi-agent)

The performance-critical pure-Python code in this project can be accelerated
by the native C++ runtime compiled in the
[kimix-base](https://github.com/…) repo (a sibling of this checkout by
default, or wherever `KIMIX_BASE` points) (`runtime_py.pyd` on Windows /
`runtime_py.so` on Linux & macOS — the platform-dependent suffix of the same
pybind11 extension, with submodules `text` / `index` / `search` / `parse` /
`tools` / `stream` / `codec` / `diff` / `glob`).

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

3. **Load**: kimi_cli/native_loader.py (the single home of the loader logic;
   kimix consumers import it directly) resolves the shim in this order:

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

- kimi_cli/native_loader.py: _KERNEL_TABLE (each known kernel under its
  upper/lower/title spellings) and _MODULE_TABLE (each known submodule
  resolved to its module object or None) are built once by
  _build_kernel_table() / _build_module_table(); unknown kernel/module
  names fall back to the shim / importlib and are memoized into the same
  tables. kernel_module(kernel) is a combined one-lookup accessor for new
  hot code. The loader is self-contained (standard library only) and never
  imports the kimix package, so kimix and kimi_cli consumers can both
  import it without a circular import.
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
| `KIMIX_NATIVE_<KERNEL>=0` | disable one kernel (TEXT / INDEX / SEARCH / PARSE / TOOLS / STREAM / CODEC / DIFF / GLOB) |
| `KIMIX_BASE=<dir>` | kimix-base repo root for the dev-only fallback & `sync_native.py` (default: the `kimix-base` sibling of this checkout) |

Per-kernel toggles make every kernel switchable and reversible without code
changes — each kernel has a bit-identical Python fallback.

## Integration pattern

Each hot function keeps its old body and gains a lazy gate. New code should
hoist the module resolution to import time and keep the per-item gate check:

```python
from kimi_cli.native_loader import get_module as _native_get_module, use_native as _native_use_native

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
| TOOLS | `src/kimix/tools/file/find_str.py` `find_in_file`; `kimi_cli/tools/file/grep_local.py` `scan_lines_cb`; `kimi_cli/utils/export.py` `build_export_markdown`; `kimi_cli/tools/file/micro_compress.py` `collapse_whitespace` / `intra_line_dedup` / `renumber_lines` / `strip_control_noise` | `tools.find_in_file`/`scan_lines_cb`/`build_export_markdown`/`compress_collapse_whitespace`/`compress_intra_line_dedup`/`compress_renumber_lines`/`compress_strip_control_noise` | 10 MB repeating log **8.8x**; 6 MB repeating unit **10.4x** |
| PARSE | all 7 comment parsers via `parser/base.py::native_parse_result`; `bash_fix.fix_bash_command`; `bash_tool._process_unquoted`; `pwsh_fix.fix_pwsh_command` | `parse.parse`/`fix_bash_command`/`_process_unquoted`/`fix_pwsh_command` | parse C 1 MB **2.1x** |
| INDEX | `src/kimix/retrieval.py` `NgramTokenizer` (normalize/detect_n/tokenize) | `index.NgramTokenizer` | — |
| SEARCH | `src/kimix/retrieval.py` `jaro_similarity`/`jaro_winkler_similarity`/`sorensen_dice_coefficient`/`ngram_overlap`/`LevenshteinAutomaton._damerau_levenshtein`/`_freq_lower_bound` | `search.*` | — |
| GLOB | `kimi_cli/tools/file/glob.py` gitignore parsing / single-source-dir matching | `glob.parse_gitignore` / `glob.is_ignored` | — |
| DIFF | `kimi_cli/utils/diff.py` `format_unified_diff`/`_build_diff_blocks_sync` | `diff.unified_diff`/`diff_hunks` | — |
| CODEC | `kimi_cli/wire/server.py` `_frame_jsonrpc` (JSON-RPC framing); `kimi_cli/wire/file.py` `_dump_line` (jsonl record) | `codec.JsonRpcFrameWriter`/`JsonlRecorder` | — |

### TOOLS micro-compression notes

The four native compress kernels mirror only the pure-string stages of
`kimi_cli/tools/file/micro_compress.py`:

- `compress_strip_control_noise` — Stage 2 (lossless)
- `compress_collapse_whitespace` — Stage 3 (lossless-or-annotated)
- `compress_renumber_lines` — Stage 5 (lossless)
- `compress_intra_line_dedup` — Stage 7 (annotated)

All four kernels are wired on **ASCII-only input** (`str.isascii()`); non-ASCII
text routes to the original Python body, matching the existing TOOLS kernel
policy.  Stages that depend on third-party libraries are intentionally
**not** ported because their Python paths already call C extensions and a C++
port is unlikely to beat them by the required 2× margin: Stage 1
`normalize_encoding` (unicodedata NFC), Stage 8 `near_duplicate_collapse`
(rapidfuzz), and Stage 9 `elide_low_value_content`.  The excluded stages
continue to run in Python before/after the native stages, so the overall
pipeline output is unchanged.

Equivalence tests: tests/native/ (318 tests, incl. test_compress_equivalence.py) + kimi-cli/tests/native/
(186 tests, incl. test_additional_kernels_equivalence.py) — every wired kernel
is run through the SAME corpus with the gate forced on vs off and outputs
asserted identical (return values, bytes, errors, determinism, thread-safety
smoke). kimix-base `python/tests/` adds per-kernel parity tests (incl.
`test_diff.py`) and the C++ kernels are tested by
`tests/unit/native/test_diff.cpp` and `tests/unit/tools/test_compress.cpp`.

> **Removed kernels:** json, image, concurrency, todo, workspace, soul were
> **deleted** (C++ sources, py bindings, shims, and tests) because their native
> kernels measured <2× faster (or slower) than the pure-Python fallback (see
> `NATIVE_BENCHMARK_REPORT.md`). All call sites now use the pure-Python
> implementations only; the loader no longer lists those kernels.

## Additional shim modules (exposed + parity-tested, not wired to app code)

These modules are usable directly (e.g. `from kimix_native import codec` or
`native_loader.get_module("codec")`) and are covered by native↔Python parity
tests, but the main app paths still use their existing Python implementations
because the native contracts do not match the app's data model bit-for-bit:

| module | public API | why it is shim-only |
|---|---|---|
| `codec` | `serialize_envelope`/`deserialize_envelope`, `canonicalize_payload`, `WireMergeBuffer`, `ArgsBuffer`, `RecvBuffer`, `build_sse_frame` | the app's envelope/tool-call paths are pydantic-model-centric; the codec is bytes-in/bytes-out. The JSON-RPC framing (`JsonRpcFrameWriter`) and jsonl record (`JsonlRecorder`) kernels ARE wired (see the table above); the envelope/buffer kernels are exposed for future server work |
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

## Windows file-lock note when re-staging `runtime_py.pyd`

On Windows, `runtime_py.pyd` is locked by any running Python process that has
already imported it. When you copy a freshly-built `.pyd` into `bin\`, a plain
`copy`/`shutil.copy2` fails with:

```text
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process.
```

The safe manual workflow (and what `tools\sync_native.py` does internally) is:

1. Rename the existing `bin\runtime_py.pyd` to `bin\runtime_py.pyd.old`.
   Windows permits renaming a mapped DLL; the old handle stays valid via the
   renamed path.
2. Copy the new `runtime_py.pyd` into `bin\` under the original name.
3. New processes load the new file; already-running processes continue using
   the `.old` copy until they exit.
4. Delete `bin\runtime_py.pyd.old` once nothing holds it. You can find holders
   with:

   ```cmd
   tasklist /m runtime_py.pyd
   ```

   Then either stop those processes or wait and remove the file after a shell
   restart.

This only applies to Windows. On Linux/macOS the extension is a `.so` and can
usually be overwritten in place even while mapped, so no `.old` dance is needed.

## Adding a new kernel

1. Wire the gate in the app function (pattern above).
2. Add equivalence cases to `tests/native/test_behavior_equivalence.py` (and
   `kimi-cli/tests/native/…`) with adversarial corpus inputs.
3. Run `python tools\sync_native.py` then the native suites in both
   `KIMIX_NATIVE=0` and `auto` modes.
4. A kernel whose equivalence test is red or absent MUST default to
   pure Python (gate off) — do not merge a routing change without its
   equivalence test.

## Updating the native version (kimix-base → kimi-agent)

A version bump is a **two-repo, three-file** change. Make it minimal: only the
version config files are touched — never hard-code the version anywhere else.

### Where the version lives

| repo | file | role |
|---|---|---|
| kimix-base | `version.txt` (root) | **single source of truth** (`X.Y.Z`). `publish.py`, `bootstrap.py` and the Python shim all read it; `src/xmake.lua` generates the C++ `kimix_version.h` header from it at build time, so `runtime_py.version()` reports `kimix-runtime <version>`. |
| kimi-agent | KIMIX_NATIVE_VERSION (root) | fallback marker read by native_loader._fallback_version() (kimi_cli/native_loader.py) and synced by install.py. Keep the file **without a trailing newline** (matches git history). |
| kimi-agent | `install.py` → `KIMIX_BASE_VERSION` | used for the GitHub release download URL and binary verification; `_sync_kimix_native_version(KIMIX_BASE_VERSION)` rewrites `KIMIX_NATIVE_VERSION` during install. |

### Minimal-change workflow

That is the whole diff. Do **not** edit bin/kimix_native/init.py (it reads
version.txt at runtime) or kimi-cli/src/kimi_cli/native_loader.py (it reads
KIMIX_NATIVE_VERSION).

### Rebuild + re-stage (the version is baked into the binary)

Because the C++ build embeds the version at compile time, bumping the config is
not enough — the staged binary must be rebuilt and re-synced or the consistency
tests fail (`test_verify_native_binaries_repo_bin` compares the staged
`runtime_py` version against `KIMIX_BASE_VERSION`):

```bash
cd <kimix-base>
xmake f -p windows -a x64 --toolchain=msvc -m release -c -y
xmake build -y runtime_py            # note: `-y` before the target name

cd <kimi-agent>
python tools/sync_native.py          # copies the fresh runtime_py.pyd into bin\
```

### Pitfall: `bin/release/runtime_py.pyd` is shared across platforms

The `runtime_py` xmake target uses `set_extension(".pyd")` on **all** platforms,
and Linux needs the file importable as `runtime_py.so`, so the WSL/Linux build
writes the module **and overwrites `bin/release/runtime_py.pyd`** with a Linux
ELF. `publish.py` builds Windows first, then Linux — so after a full publish the
on-disk `.pyd` is a Linux binary. If you then run `tools/sync_native.py` on
Windows, `bin/runtime_py.pyd` is an ELF and importing it fails with
`ImportError: DLL load failed ... %1 is not a valid Win32 application`.

**Fix:** always rebuild the Windows target *after* any Linux/WSL build and
before `sync_native.py` (the sequence above). Sanity-check the file is a real
Windows PE (`MZ`/`PE\x00\x00` magic, x64) or simply:

```bash
python -c "import sys; sys.path.insert(0, r'<repo>\bin'); import runtime_py; print(runtime_py.version())"
# expect: kimix-runtime <version>
```

### Verification

```bash
cd <kimi-agent>
python -m pytest tests/test_install_kimix_native.py tests/native/test_loader.py -q
# 39 passed, 2 skipped
```

`KIMIX_NATIVE=0` fallback also picks up the new version automatically:
`native_loader.version()` → `kimix-native <version> (python fallback)`.
