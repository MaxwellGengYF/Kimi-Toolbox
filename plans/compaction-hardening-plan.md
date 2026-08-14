# Compaction Hardening Plan — Port DSH Gaps into kimi-agent

**Status:** Proposed
**Owner:** agent
**Related study:** `plans/` — comparison of kimi-agent vs `C:\dev\deepseek-harness` compaction strategy (commit `14fa69c` "Fix compact strategy" + DSH `packages/compaction/*`).

---

## 0. Goal and Gap Analysis

kimi-agent already has a solid compaction core:
- Trigger: `should_auto_compact()` (ratio OR reserved-output boundary) — last commit `14fa69c` made reservations `max()`-based instead of summed.
- Pre-compaction pruning alternative (`ContextPruner`, 3 tiers) that can skip compaction.
- Summarization via `SimpleCompaction` (sliding-window preserve, first-message preservation, modes, todo re-injection).
- Durability: pre-compaction export to `.kimix_cache/context_compacted.md`, history-index marking, wire `CompactionBegin`/`CompactionEnd`.

The comparison against DSH found **four missing capabilities** that this plan ports over:

| # | Missing feature (DSH has it) | DSH reference | kimi gap today |
|---|---|---|---|
| 1 | **Tool-pairing-balanced cuts** | `packages/compaction/compaction/src/tool-pairing.ts` | `SimpleCompaction.prepare` only counts `user`/`assistant` messages; a preserved-boundary can split an assistant `ToolCall` from its `tool` result message |
| 2 | **KV-cache-aligned summarization** | `compaction-basic/src/summarizer.ts` | The compaction LLM call uses a generic system prompt + `EmptyToolset` + a flattened role-labeled text message; it does not reuse the conversation's system prompt/tools/prefix, so it invalidates the provider KV cache |
| 3 | **Durable transactional compaction** | `compaction/src/*`, `region.ts` | No `compactionId`, no shadowed-range accounting, no stability check across the async summarize gap, no classified manual errors, no shrink check |
| 4 | **Context-overflow recovery loop** | `compaction-basic/src/index.ts` (`agent/request-error` + `{kind:'retry'}`) | `classify_api_error` already detects `"context_overflow"`, but `_step` treats it like any other 4xx — retries then raises `SessionRestartRequired`; it never force-compacts and retries the request |

**Non-goals (out of scope):**
- Replacing kimi's message-count preserve strategy with DSH's token-budget tail retention (retainRatio). Kimi's `reserved_context_size` + adaptive depth already work; porting DSH's `retainRatio=0.16` would be a behavior change with no clear win. (Optional stretch, see §9.)
- Porting DSH's plugin/config-validation strictness wholesale.
- Moving kimi to an event-log session model (`compaction/*` events). kimi's durability model is `Context` + wire events + export; we mirror the *semantics* (id, provenance, shadow accounting), not the log format.

---

## 1. Architecture Overview of Changes

New/modified modules:

```
kimi-cli/src/kimi_cli/soul/
├── compaction.py            # MODIFIED — balanced cuts, KV-aligned input, tx id, shrink check
├── compaction_ledger.py     # NEW — durable compaction transaction records + shadow accounting
├── tool_pairing.py          # NEW — balanced-cut detection over Message history (port of DSH tool-pairing)
├── context_overflow.py      # NEW — overflow classification helper + recovery loop state
├── kimisoul.py              # MODIFIED — overflow recovery in _step; wire events carry compaction_id
├── slash.py                 # MODIFIED — /compact passes through tx options, reports classified errors
kimi-cli/src/kimi_cli/
├── config.py                # MODIFIED — LoopControl: overflow-retry + tx fields
├── wire/types.py            # MODIFIED — CompactionBegin/End carry compaction_id + shadowed stats
└── acp/session.py           # MODIFIED — surface new wire fields (compat read)
kimi-cli/src/kimi_cli/prompts/
├── compact.md               # MODIFIED (minor) — note the KV-aligned invocation; no content change
kimi-cli/tests/core/
├── test_tool_pairing.py     # NEW
├── test_compaction_ledger.py# NEW
├── test_context_overflow.py # NEW
└── test_simple_compaction.py# MODIFIED — new balanced-cut + tx + shrink tests
```

Dependency direction: `tool_pairing.py` and `compaction_ledger.py` are leaf modules (no kimi_cli.soul imports beyond types); `compaction.py` imports them; `kimisoul.py` imports `compaction.py` (already does).

---

## 2. Phase 0 — Config & Wire Foundations

### 2.1 `config.py` — new `LoopControl` fields

Add after `compaction_trigger_ratio`/`reserved_context_size` (lines ~138-157):

```python
# ── Context-overflow recovery (DSH port) ──────────────────────────────
context_overflow_retries: int = Field(default=1, ge=0, le=5)
"""Max number of force-compact-and-retry cycles after a provider-confirmed
context-window-exceeded error, per step. 0 disables recovery."""

context_overflow_preserve_depth: int = Field(default=1, ge=0, le=4)
"""Preserve depth used for the forced overflow compaction (bypasses the normal
adaptive depth). Lower = more aggressive reduction."""

context_overflow_force_threshold: bool = Field(default=True)
"""When true, overflow compaction bypasses should_auto_compact entirely
(DSh context-overflow semantics: 'force one useful balanced reduction')."""

# ── Durable compaction transaction ────────────────────────────────────
compaction_ledger_enabled: bool = Field(default=True)
"""Persist compaction transactions (compaction_id, shadowed seqs/tokens) to
the session ledger file."""
```

Model validation note: add to the existing `LoopControl` validator (near line 416) a check that `context_overflow_preserve_depth <= max_preserved`-style sanity is not required (it is independent), but `context_overflow_retries` is non-negative.

### 2.2 `wire/types.py` — extend compaction events

`CompactionBegin` / `CompactionEnd` (lines 99-116) currently carry no payload. Extend both (backward compatible — all fields optional):

```python
class CompactionBegin(BaseModel):
    compaction_id: str | None = None
    trigger: Literal["auto", "manual", "overflow"] | None = None
    shadowed_tokens: int | None = None   # estimated tokens being replaced (best-effort)

class CompactionEnd(BaseModel):
    compaction_id: str | None = None
    trigger: Literal["auto", "manual", "overflow"] | None = None
    shadowed_tokens: int | None = None
    estimated_token_count: int | None = None   # post-compaction estimate
    error: str | None = None                   # set when compaction failed after Begin
```

Update `acp/session.py` case statements (lines ~180, ~281) to tolerate/echo the new optional fields (no behavior change required since fields are optional).

---

## 3. Phase 1 — Tool-Pairing-Balanced Compaction Cuts

### 3.1 New module `kimi-cli/src/kimi_cli/soul/tool_pairing.py`

Port DSH `tool-pairing.ts` semantics to kimi's `Message` model:

```python
from collections.abc import Sequence
from kosong.message import Message, ToolCallPart

def message_tool_call_delta(msg: Message) -> int:
    """+N for an assistant message with N tool-call parts; -1 for a tool result; 0 otherwise."""
    if msg.role == "assistant":
        return sum(1 for part in msg.content if isinstance(part, ToolCallPart))
    if msg.role == "tool":
        return -1
    return 0

def balanced_cut_indices(messages: Sequence[Message]) -> set[int]:
    """Return the set of *cut indices* (0..len) where no unanswered tool call
    crosses the cut. cut i = boundary between messages[i-1] and messages[i].
    A fold over the history: maintain in_progress; a cut is balanced iff
    in_progress == 0 after processing messages[:i]. Raises ValueError on
    unbalanced history (tool result with no matching call)."""

def nearest_balanced_cut_before(messages: Sequence[Message], index: int) -> int:
    """Largest balanced cut <= index (never splits a call/result pair).
    index==len means 'after the last message'."""
```

Implementation notes:
- Mirror DSH: an assistant message with N `ToolCallPart`s increments by N; a `tool` message decrements by 1; throw `ValueError` if the count goes negative (corrupt history) — kimi's `normalize_history` guarantees pairing in practice.
- `ToolCallPart` is the streamed-arguments part; after generation kimi's assistant messages carry the merged `ToolCall` parts? Verify: `kosong.message.ToolCall` is the completed call. Check which part type appears in `_grow_context`-persisted assistant messages — use **both** `ToolCall` and `ToolCallPart` in the delta scan (count `ToolCall` instances and `ToolCallPart` instances; a merged message has `ToolCall`, a streamed one may have `ToolCallPart`). Implement `is_tool_call_part(part)` covering both.

### 3.2 Modify `SimpleCompaction.prepare` in `soul/compaction.py`

Current behavior (lines 368-439): scans from the tail counting `user`/`assistant` messages to find `preserve_start_index`, then (Phase 6) re-inserts `history[0]`.

Change:
1. After computing the raw `preserve_start_index`, snap it left to `nearest_balanced_cut_before(history, preserve_start_index)` — i.e. walk back until the cut is balanced. This guarantees the preserved tail never starts mid-pair.
2. When Phase 6 re-inserts `history[0]` as preserved, also ensure the split between `to_compact` and `to_preserve` is a balanced cut *after* the re-insertion: if inserting the first message at the front makes the boundary unbalanced (e.g. first message is a tool result — impossible in practice; but guard anyway), keep it in `to_compact` instead of forcing balance violations.
3. Add a `balanced: bool = True` constructor kwarg (`SimpleCompaction(..., balanced_cuts: bool = True)`) so callers can opt out; default on.
4. When balanced_cuts is on and no balanced cut exists below the preserve point (pathological), fall back to `preserve_start_index = 1` with a `logger.warning`.

Tests (`tests/core/test_tool_pairing.py`):
- delta counting for assistant-with-2-calls, tool result, plain user, system.
- balanced cut set on a mixed history.
- `prepare` never produces a preserved tail starting with a `tool` message whose call is inside `to_compact`.
- first-message re-insertion keeps boundaries balanced.
- opt-out flag restores legacy behavior exactly (snapshot tests).

---

## 4. Phase 2 — KV-Cache-Aligned Summarization

### 4.1 Problem

Today `SimpleCompaction.compact` calls `kosong.step(chat_provider, "You are a helpful assistant that compacts conversation context.", EmptyToolset(), [compact_message])` where `compact_message` is a **flattened text dump** (`## Message N\nRole: …`). This is a completely different request shape from the main loop → provider prefix cache is invalidated on the next step.

### 4.2 Design (mirror DSH `summarizer.ts`)

Replace the flattened-message compaction input with a **cache-aligned prefix call**:

1. `prepare()` still computes `to_compact` / `to_preserve` (a contiguous prefix + preserved tail) and the instruction text.
2. Instead of building `compact_message`, build a `SummarizationInput`:
   ```python
   @dataclass(frozen=True, slots=True)
   class SummarizationInput:
       system_prompt: str | None          # None → omit (generic fallback)
       tools: Sequence[Tool]              # real schemas, for prefix alignment
       messages: Sequence[Message]        # the contiguous to_compact region (original messages)
       instruction: Message               # final user message with the compaction prompt
   ```
3. `SimpleCompaction.compact` gains optional kwargs to receive the aligned context (from `KimiSoul`):
   ```python
   async def compact(self, messages, llm, *, custom_instruction="", options=None,
                     recorder=None, todos_loader=None, todos_stack_loader=None,
                     aligned_system_prompt: str | None = None,
                     aligned_tools: Sequence[Tool] | None = None) -> CompactionResult
   ```
   - When `aligned_system_prompt is not None` → call `kosong.generate(chat_provider, aligned_system_prompt, aligned_tools or [], [*input.messages, input.instruction])` (generate, not step — no tool dispatch; the instruction forbids tool calls).
   - Else → legacy flattened path (kept for tests/back-compat).
4. `KimiSoul.compact_context` passes `aligned_system_prompt=self._agent.get_system_prompt()` and `aligned_tools=self._agent.toolset.tools` when the toolset is a `KimiToolset` (fall back to legacy otherwise).
5. Instruction text: append the existing `prompts.COMPACT` (+ mode guidance + custom instruction + decision sections) as the **final user message** after the region, exactly like DSH appends `COMPACTION_INSTRUCTION` after the replayed prefix. Add one sentence to `compact.md` ("Treat the conversation above as established context…") — optional; the existing text already says "Compact the above agent conversation context".

### 4.3 Why this aligns the KV cache

The provider caches the request prefix keyed by (system, tools, leading messages). The region `to_compact` is the *contiguous head* of the real conversation; replaying system+tools+head verbatim and appending only the instruction means the cacheable prefix is identical to the previous step's request (up to the compaction point). DSH proves this works; kimi's `compact_export_path` determinism (cache-05) already follows the same philosophy.

### 4.4 Tests

`tests/core/test_simple_compaction.py`:
- `prepare` returns the new `SummarizationInput` (region messages = to_compact, instruction = final user message) when aligned args are passed.
- KV-aligned path calls `kosong.generate` with exactly `[region..., instruction]` (fake provider records calls).
- Legacy path unchanged (existing snapshots stay green).
- Instruction message is `role="user"`, single `TextPart`, contains `prompts.COMPACT`.

---

## 5. Phase 3 — Durable Transactional Compaction

### 5.1 New module `kimi-cli/src/kimi_cli/soul/compaction_ledger.py`

A durable, append-only ledger of compaction transactions, stored next to the session export (deterministic path, like cache-05):

```python
@dataclass(frozen=True, slots=True)
class CompactionRecord:
    compaction_id: str                 # uuid4 hex
    trigger: Literal["auto", "manual", "overflow"]
    started_at: float                  # monotonic-ish epoch
    shadowed_range: tuple[int, int]    # history indices [start, end) replaced
    shadowed_tokens: int               # estimated tokens of replaced region
    summary_tokens: int                # LLM usage.output when available, else estimate
    preserved_tokens: int              # estimated tokens of preserved tail
    shrank: bool                       # summary_tokens < shadowed_tokens
    error: str | None = None           # set on failure (end-of-transaction)

class CompactionLedger:
    def __init__(self, path: Path | None): ...
    def record_start(self, record: CompactionRecord) -> None
    def record_end(self, compaction_id: str, *, error: str | None = None) -> None
    def latest(self) -> CompactionRecord | None
    @classmethod
    def for_session(cls, session_dir: Path, *, enabled: bool) -> "CompactionLedger"
```

Path: `<session workdir>/.kimix_cache/compaction_ledger.jsonl` (append-only JSONL; `orjson` per AGENTS.md). Failure-isolated: ledger write errors must never break compaction (wrap in try/except + `logger.warning`).

### 5.2 Transaction semantics in `SimpleCompaction.compact`

Current `compact()` is one shot: prepare → kosong.step → wrap. Add the transaction envelope:

1. `compact()` generates `compaction_id = uuid4().hex` up front.
2. Before the LLM call: snapshot `SurfaceFingerprint`:
   ```python
   @dataclass(frozen=True, slots=True)
   class SurfaceFingerprint:
       history_len: int
       token_count: int
       last_message_text: str | None     # cheap content fingerprint of last message
   ```
3. After the LLM call, before applying: **stability check** — recompute the fingerprint of the same `messages` argument (compaction is called with `self._context.history`; re-verify `len(messages) == history_len` and the tail is unchanged). If changed → raise `SurfaceChangedError("conversation changed during compaction")`; `compact_context` then re-runs `prepare` once (retry) instead of applying a stale summary.
4. **Shrink check** (DSH `region.ts`): `summary_tokens >= shadowed_tokens` → raise `CompactionShrinkError`. Compute `shadowed_tokens = count_message_tokens(to_compact)`, `summary_tokens = result.usage.output` when available else `count_message_tokens([compacted_msg])`.
5. On success: `compaction_ledger.record_end(...)` with shrank=True.
6. Return `CompactionResult` extended with `compaction_id: str` and `shadowed_tokens: int` (NamedTuple — add fields with defaults to stay back-compat).

### 5.3 Classified manual errors (port `ManualCompactionError`)

Add to `soul/compaction.py`:

```python
class ManualCompactionError(Exception):
    code: Literal["busy", "cancelled", "changed", "summary", "commit", "persistence"]
```

`slash.py /compact` wraps `soul.compact_context(manual=True)` and maps:
- `changed` → "history changed during compaction; try again"
- `summary` → "summary was not smaller than the compacted content"
- `commit`/`persistence` → "compaction did not commit cleanly"
- `busy` → "compaction already in progress"

### 5.4 Wire the ledger into `KimiSoul.compact_context`

- Create the ledger at session init (`_runtime.session.dir`); pass `ledger` into `_compaction.compact(...)`.
- `wire_send(CompactionBegin(compaction_id=…, trigger=…, shadowed_tokens=…))` and `CompactionEnd(…, estimated_token_count=estimated_token_count)` (replace bare calls at lines 1911/2002).
- Keep existing export + history-index marking.

### 5.5 Tests

`tests/core/test_compaction_ledger.py`:
- record_start/end round-trip; latest() ordering; error field; JSONL parse with orjson.
- failure isolation: a broken ledger path does not raise out of `compact`.

`tests/core/test_simple_compaction.py` (extend):
- shrink check raises when summary >= shadowed.
- stability check raises `SurfaceChangedError` when messages mutated mid-flight (simulate with a fake provider that mutates the list).
- `CompactionResult.compaction_id` propagates.
- `ManualCompactionError` codes map in slash path.

---

## 6. Phase 4 — Context-Overflow Recovery Loop

### 6.1 New module `kimi-cli/src/kimi_cli/soul/context_overflow.py`

```python
from kosong.chat_provider import ChatProviderError

CONTEXT_OVERFLOW_MARKERS = ("context length", "context_length", "max tokens",
                            "maximum context", "too many tokens")

def is_context_overflow_error(exc: BaseException) -> bool:
    """True for APIStatusError 4xx whose message matches the markers,
    mirroring classify_api_error's 'context_overflow' branch (kimisoul.py:186-195)."""

class OverflowRecoveryState:
    """Per-step overflow retry budget (reset on each step begin / success)."""
    def __init__(self, max_retries: int): ...
    def can_retry(self) -> bool
    def consumed(self) -> None
    def reset(self) -> None
```

Note: `classify_api_error` already returns `"context_overflow"` — reuse it instead of duplicating the marker list where possible (import from `kimisoul` would create a cycle; keep the markers in `context_overflow.py` and have `classify_api_error` delegate to `is_context_overflow_error`).

### 6.2 Modify `_step` in `kimisoul.py`

In the retry-exhausted exception handling (lines 1676-1702), add an overflow branch **before** the generic `APIStatusError` → `SessionRestartRequired` branch:

```python
except (APIStatusError, APIConnectionError, APITimeoutError) as e:
    if (is_context_overflow_error(e)
            and self._loop_control.context_overflow_retries > 0
            and overflow_state.can_retry()):
        logger.warning("Context window exceeded at step {step}; force-compacting and retrying", ...)
        try:
            await self.compact_context(
                manual=False,
                mode=CompactMode.AGGRESSIVE,          # aggressive reduction
                options=CompactionOptions(preserve_depth_override=
                    self._loop_control.context_overflow_preserve_depth),
            )
        except Exception as compact_err:
            logger.error("Overflow compaction failed: {err}; preserving original error", ...)
            raise SessionRestartRequired(...) from e
        overflow_state.consumed()
        return await self._step()      # re-run the same step on the compacted context
    # existing generic handling
    raise SessionRestartRequired(...)
```

Implementation details:
- `overflow_state = OverflowRecoveryState(max_retries)` created at the top of `_step`; reset after a successful `_kosong_step_with_retry` (post line 1675) — mirrors DSH resetting on `agent/status idle` and `assistant/message`.
- **Guard against infinite recursion**: the re-entrant `return await self._step()` is bounded by `max_retries` (default 1). Because `compact_context` rebuilds history, the next attempt's input is strictly smaller; also the step guard `max_steps_per_turn` applies. Add a `RecursionError`-style safety: cap overflow re-entries per step via `overflow_state`.
- When `context_overflow_force_threshold=True`, bypass `should_auto_compact` for this forced compaction (call `compact_context` directly).
- `compact_context` needs a `preserve_depth_override` knob: extend `CompactionOptions` with `preserve_depth_override: int | None = None`, thread through `SimpleCompaction.compact` → `prepare` (`_resolve_preserve_depth` returns the override first).
- Wire the trigger as `"overflow"` in `CompactionBegin/End` + ledger.

### 6.3 Tests

`tests/core/test_context_overflow.py`:
- `is_context_overflow_error` matches the marker strings (reuse the 5 markers) and rejects 429/500/network.
- Overflow branch: fake provider raises `APIStatusError(400, "maximum context length exceeded")` once then succeeds; assert `compact_context` called with AGGRESSIVE + override depth, step retried, no `SessionRestartRequired`.
- Retry budget exhausted: provider always overflows → after `max_retries+1` raises `SessionRestartRequired`.
- `context_overflow_retries=0` → immediately `SessionRestartRequired` (recovery disabled).
- `classify_api_error` still returns `"context_overflow"` (regression, via delegation).

---

## 7. Integration Checklist (KimiSoul wiring)

- [ ] `KimiSoul.__init__`: build `CompactionLedger.for_session(...)`; store `self._compaction_ledger`; pass `aligned_system_prompt`/`aligned_tools` and `ledger` into `SimpleCompaction` constructor or per-call kwargs.
- [ ] `compact_context` (lines 1814-2017): generate `compaction_id`; emit enriched `CompactionBegin`; catch `SurfaceChangedError` → one re-prepare retry; catch `CompactionShrinkError` → raise `ManualCompactionError('summary', …)` when manual; emit enriched `CompactionEnd` with `estimated_token_count`; ledger record.
- [ ] `_step` (lines 1459+): overflow recovery branch (Phase 4); reset `overflow_state` after success.
- [ ] `slash.py /compact`: wrap errors into `ManualCompactionError` display; pass `custom_instruction` unchanged.
- [ ] `config.py`: new fields + docs; run `uv run tools/syntax_check.py` on all touched `.py`.
- [ ] `AGENTS.md` note: compaction ledger + overflow recovery are part of the compaction pipeline.

---

## 8. Test Matrix & Verification

| Test file | Covers |
|---|---|
| `tests/core/test_tool_pairing.py` (new) | delta counting, balanced cut set, nearest-balanced-before, prepare boundary guarantees, opt-out flag |
| `tests/core/test_simple_compaction.py` (extend) | SummarizationInput shape, KV-aligned generate call, legacy path snapshots, shrink check, stability check, `compaction_id` propagation, `CompactionOptions.preserve_depth_override` |
| `tests/core/test_compaction_ledger.py` (new) | JSONL round-trip, ordering, error field, failure isolation |
| `tests/core/test_context_overflow.py` (new) | marker detection, recovery loop, budget exhaustion, disabled mode, `classify_api_error` regression |
| `tests/core/test_kimisoul_context_prune.py` | unchanged behavior (regression) |
| `tests/core/test_compact_reminder.py`, `test_todo_compact_injection.py` | regression — must stay green |

Verification gate (per AGENTS.md):
1. `uv run tools/syntax_check.py <every touched .py>`
2. `uv run pytest kimi-cli/tests/core/test_tool_pairing.py kimi-cli/tests/core/test_simple_compaction.py kimi-cli/tests/core/test_compaction_ledger.py kimi-cli/tests/core/test_context_overflow.py`
3. Full `uv run pytest kimi-cli/tests/core` for regressions.
4. `uv run tools/git_diff.py <files>` review before commit.
5. Manual smoke: run a long conversation, force `/compact`, verify `CompactionEnd` carries `compaction_id`; simulate overflow with a small `max_context_size` override and observe force-compact + retry (no `SessionRestartRequired`).

---

## 9. Stretch / Follow-ups (explicitly deferred)

- **Token-budget tail retention** (`retainRatio`) as an alternative to message-count preserve — needs a per-message token estimate already available via `count_message_tokens`; can be a follow-up config option `compaction_retention: "messages" | "tokens"`.
- **Per-model policy overrides** (`modelPolicies` table) — kimi already has per-provider config; a `model_compaction_overrides` table could reuse `resolveTargetPolicy` logic later.
- **Prune-then-compact chaining inside the overflow path** — run `ContextPruner` first on overflow (DSH does prune unconditionally on overflow before selecting a range); cheap add-on to Phase 4.
- **`compaction/end` error-chain logging** parity — kimi's ledger `error` field covers this at a coarser granularity.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| KV-aligned summarization changes compaction output quality | Keep the legacy flattened path as fallback; add snapshot tests; the instruction text is unchanged, only the transport |
| Balanced cuts change preserve boundaries → different summaries | All changes are behind `balanced_cuts=True` default with opt-out; extensive snapshot tests |
| Overflow re-entry could loop | Budget via `OverflowRecoveryState` (default 1) + `max_steps_per_turn`; never re-enter more than `context_overflow_retries` times |
| Ledger I/O slows compaction | Append-only JSONL, failure-isolated, `orjson`; measured overhead << LLM call |
| Wire event schema change breaks ACP/UI | New fields optional; `acp/session.py` matched on structure, not fields |
| `kosong.generate` signature lacks `max_tokens`/usage envelope | Check `_generate.py` (line 17-78): it wraps `chat_provider.generate`; if usage is missing, estimate via `count_message_tokens` (current behavior already estimates) — record as known limitation |

---

## 11. Suggested Commit Sequence

1. `feat(compaction): balanced tool-pairing cuts` (Phase 1) — tool_pairing.py + prepare changes + tests.
2. `feat(compaction): KV-cache-aligned summarization input` (Phase 2) — SummarizationInput + generate path + tests.
3. `feat(compaction): durable transaction ledger + classified errors` (Phase 3) — ledger, tx envelope, wire events, slash mapping + tests.
4. `feat(compaction): context-overflow force-compact and retry` (Phase 4) — overflow module, _step branch, config + tests.
5. `chore(compaction): config docs + AGENTS.md note + full regression run`.

Each commit passes the verification gate independently; commit 4 depends on 3 (needs `compaction_id`/ledger + AGGRESSIVE mode), 3 depends on 1-2 only loosely (independent), so phases can land in any order after 1.
