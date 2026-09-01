# KVCacheV2 Runtime Investigation Results

**Commit tested:** `b895f79ef36c00b91c4af90e95a0f09e8260413f` (branch `allim/kv_cache`)
**Environment:** `tensorrt_llm-devel-allim` dev container, 1x NVIDIA H200 GPU.
`LLM_MODELS_ROOT` is unset in this environment — no HF model artifacts are
available, so nothing here depends on a real downloaded model.

**Status:** Committed as an investigation record for TRTLLM-15289. The four
probe test files it references (`test_scratch_q1_*.py`, `test_scratch_q2_*.py`,
`test_scratch_q3_*.py`, `test_scratch_q4_*.py`, all under `tests/unittest/`)
are committed alongside it as evidence artifacts, not as permanent regression
coverage — they should be reviewed/refactored (or removed) before/if any of
their findings inform a production change.

**Method note:** `scratchpad/kvcachev2_context/*.md` (pulled in with this
branch) already contains a detailed prior audit of KVCacheV2 scheduler/manager
behavior, including a gap-closure pass (`coverage_closure.md`) that had
already identified exactly these four kinds of questions as requiring dynamic
(test-based) evidence rather than more source reading, and proposed concrete
starting points. This investigation followed that plan rather than starting
from scratch.

**Environment setup performed:** the dev container's Python environment was
missing most of `requirements.txt`/`requirements-dev.txt` (transformers,
tokenizers, blake3, pytest, mako, dotenv, execnet, and dozens more — imports
of `tensorrt_llm` itself failed). `pip3 install -r requirements.txt -r
requirements-dev.txt` was run inside the running container to fix this. No
C++/wheel rebuild was performed (no `build_wheel.py`, no `cmake`, no `pip
install -e .`) — this was a pure Python dependency install against the
already-built `tensorrt_llm.bindings` native extension.

---

## Q1 — Two-model spec-decode: target/draft GPU capacities for explicit `max_gpu_total_bytes=B`

**Test:** `tests/unittest/kv_cache_manager_v2_tests/test_scratch_q1_spec_decode_capacity.py::test_two_model_spec_decode_capacity_for_explicit_gpu_budget`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/kv_cache_manager_v2_tests/test_scratch_q1_spec_decode_capacity.py -x -s -q
```

**Result:** PASS

**Observed (real numbers, B = 4 GiB = 4294967296 bytes, target:draft per-token cost ratio 80:100):**
```
target_budget=3435973837 draft_budget=858993459
target GPU quota (bytes)=3439329280, pool_group num_slots=[52480]
draft  GPU quota (bytes)=859832320,  pool_group num_slots=[52480]
```
(target has 4 attention layers, draft has 1 — so per-token bytes differ 4x,
which is why the resulting page/slot *counts* end up numerically equal despite
the 4x byte-budget difference. Verified the underlying scaling is correct with
an isolated sanity check: 200 MiB / 1 layer → 12800 slots, and 859 MiB / 1
layer → 52480 slots, a ~4.10x ratio matching the ~4.11x budget ratio.)

**What this proves:**
- The *split* from B into target/draft byte budgets (`3435973837 + 858993459
  == B`) is computed by the real, unmodified
  `KvCacheCreator._split_kv_cache_budget_for_draft` /
  `_compute_draft_budget_shares` code in
  `tensorrt_llm/_torch/pyexecutor/_util.py`. Only the leaf per-token cost
  inputs are synthetic (`CacheCost(slope=..., intercept=0)` fed in directly)
  — real per-layer costs require a loaded HF model, which is unavailable
  here. This substitution is the same technique the repo's own
  `tests/unittest/_torch/executor/test_kv_cache_budget_split.py` uses for the
  identical function.
- The GPU page/pool *capacities* (`get_quota`, `pool_group_descs[i].num_slots`)
  come from a real, GPU-backed `tensorrt_llm.runtime.kv_cache_manager_v2.KVCacheManager`
  (the C++/nanobind-backed native manager, same class the production
  `KVCacheManagerV2` wraps), constructed with the split byte budgets and
  actually allocating GPU memory on the H200 in this container.

**Proves runtime behavior vs. confirms configuration:** **Partially proves
runtime behavior.** The byte-budget split is real code, real math. The
resulting page/pool capacities are a real GPU allocation, not a mock. The one
substitution is the per-token cost model (necessarily synthetic, since no real
model is available) — so this does not prove what a *specific* real model's
target/draft split would look like, but it does prove the split-then-allocate
pipeline is internally consistent and that real native construction succeeds
end-to-end for both managers simultaneously under one shared budget.

---

## Q2 — Generation allocation failure + eviction/suspension: are target/draft states always aligned afterward?

**Test:** `tests/unittest/_torch/executor/test_scratch_q2_target_draft_evict_alignment.py::test_draft_suspend_failure_leaves_target_and_draft_misaligned`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/_torch/executor/test_scratch_q2_target_draft_evict_alignment.py -x -s -q
```

**Result:** PASS (the test asserts the *failure/misalignment* behavior itself)

**Observed:** Real, unmodified `KVCacheV2Scheduler._suspend_request`
(`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:1051-1063`) is:
```python
self.kv_cache_manager.suspend_request(req)
if self.draft_kv_cache_manager is not None:
    self.draft_kv_cache_manager.suspend_request(req)
```
with **no try/except around either call.** Driving a real `KVCacheV2Scheduler`
(mocked target/draft managers) through `_try_evict_for_gen` with the draft
manager's `suspend_request` raising:
- `target_mgr.suspend_request` is called exactly once and **completes** — the
  target manager shows the victim request suspended (`is_request_active ==
  False`) afterward.
- `draft_mgr.suspend_request` is called (that's what raised) but never
  completes its own state mutation.
- The `RuntimeError` from the draft manager **propagates out of
  `schedule_request` uncaught** — there is no rollback of the target
  manager's already-completed suspend.

**Answer: NO — target and draft are NOT always aligned afterward.** If the
second (draft) mirror call in a sequential, unguarded pair fails, the target
manager is left suspended while the draft manager's request never suspended,
and the caller sees an unhandled exception rather than a consistent
post-eviction state on either side.

**Proves runtime behavior vs. confirms configuration:** **Proves real
scheduler-decision logic.** `KVCacheV2Scheduler` itself is real, unmodified
code (not mocked) — only the two KV cache managers are `Mock()` objects with
controlled `side_effect`s (the same fault-injection style used throughout the
repo's own `test_kv_cache_v2_scheduler.py`, which is explicitly documented in
that file as a "Tier 1 mock unit test" suite, no GPU). This is real evidence
of the *scheduler's* mirroring contract and its failure mode, but it does
**not** prove what a *native* GPU-backed draft-manager `suspend_request`
failure would look like at the C++ level, nor confirm whether the native
implementation can even raise in that call in practice (that would require a
native/runtime fault-injection harness, which was out of scope for the time
available in this pass).

---

## Q3 — Later chunk of context allocation fails: is the request recoverable, are manager resources cleaned up consistently?

**Test:** `tests/unittest/kv_cache_manager_v2_tests/test_scratch_q3_chunk_fail_recovery.py::test_second_chunk_alloc_failure_is_recoverable_and_leak_free`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/kv_cache_manager_v2_tests/test_scratch_q3_chunk_fail_recovery.py -x -s -q
```

**Result:** PASS

**Observed (real GPU allocation, 8 MiB quota, 32 tokens/block, 1 layer):**
```
[Q3] first chunk ok: capacity=64
[Q3] second chunk correctly raised OutOfPagesError
[Q3] request still usable post-failure: capacity grew to 96
[Q3] post-cleanup fresh alloc reached capacity=512 (no leak from failed resize)
```
- First "chunk" (`kv_cache.capacity = 64`) succeeds against the real,
  GPU-backed native `KVCacheManager`.
- Second "chunk" (`kv_cache.capacity = 3200000`, deliberately far beyond the
  8 MiB quota) raises the real native `OutOfPagesError` (a genuine allocation
  failure, not simulated) from
  `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2`.
- After the failure, `kv_cache.capacity` is unchanged (still 64) — the failed
  resize did **not** partially mutate state.
- The same request can then grow to a smaller, satisfiable capacity (96) —
  proving the manager's internal state is not wedged after the failed
  attempt.
- `kv_cache.close()` succeeds, and a **fresh** sequence created afterward can
  reach the full quota's worth of capacity again (512 tokens = 16 blocks for
  this quota/layer config) — proving the failed huge-capacity attempt did not
  leak or strand any GPU pages.

**Answer: YES — the request is recoverable (last successful capacity is
preserved and the sequence remains usable), and manager resources are cleaned
up consistently (no page leak from the failed chunk, and normal close/free
proceeds without error).**

**Important negative finding, ruled out (not a bug):** an earlier version of
this test that omitted `kv_cache.resume(stream)` before use reproducibly
crashed the *entire process* with `std::bad_optional_access` inside
`SharedPageLock::unlock()` → `KvCache::_clearBlocks()` → `KvCache::close()`
(`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp`,
`page.cpp:365`), traced to `KvCache::finishEvent()` calling
`mFinishEvent.value()` on an unset `std::optional`
(`kvCache.cpp:132`). This is **not** a real KVCacheManagerV2 bug: every
production `KvCache` is resumed onto a CUDA stream before use (confirmed by
`TestKVCacheManagerV2`'s own test harness convention in
`test_kv_cache_manager_v2.py`, which always calls `.resume(stream)` before
`.capacity = N`), and adding the missing `resume()` call to the repro made the
exact same failure-then-close sequence complete cleanly and deterministically
across repeated runs. This is flagged here only so it isn't rediscovered as a
false-positive bug in a future pass — it's a precondition-violation artifact
of a minimal repro, not a defect in the code under test.

**Proves runtime behavior vs. confirms configuration:** **Proves real native
runtime behavior.** This exercises the actual GPU-backed native
`KVCacheManager`/`_KVCache` bindings (`KV_CACHE_MANAGER_V2_BACKEND=cpp`, the
default), with a real `OutOfPagesError` triggered by genuine GPU memory
exhaustion at a small, controlled quota — not a mock, not a simulated
failure. This is the strongest evidence tier among the four questions.

---

## Q4 — Disaggregated context→generation: scheduler budget accounting vs. manager reservation

**Tests:** `tests/unittest/_torch/executor/test_scratch_q4_disagg_budget_vs_manager_reservation.py`
- `test_disagg_gen_init_consumes_zero_scheduler_token_budget`
- `test_disagg_transmission_complete_gen_scheduler_charges_beam_width_only`
- `test_manager_effective_draft_len_reserves_more_than_scheduler_charges`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/_torch/executor/test_scratch_q4_disagg_budget_vs_manager_reservation.py -x -s -q
```

**Result:** 3/3 PASS

**Observed:**
1. A real `KVCacheV2Scheduler` under a **1-token** `max_num_tokens` budget
   still admits a `DISAGG_GENERATION_INIT` request — the real
   `BudgetTracker` charges it **0 tokens** (confirmed: `req in
   out.fitting_disagg_gen_init_requests`). Admission for this request class
   is gated entirely by the manager's `prepare_disagg_gen_init` return value
   (mocked here to always succeed; in production this is real
   `IndexMapper`/capacity gating), not by the scheduler's token ledger.
2. A real `KVCacheV2Scheduler`, same 1-token budget, admits a
   transmission-complete generation request
   (`is_disagg_generation_transmission_complete=True`, `py_draft_tokens=[]`)
   — the real `_try_schedule_generation` token-cost line (`req_tokens =
   beam_width + get_draft_token_length(req)`,
   `scheduler_v2.py:984`) charges exactly **`beam_width` (1)** since
   `get_draft_token_length(req) == len(req.py_draft_tokens) == 0`.
3. Calling the real, unmodified `KVCacheManagerV2._effective_draft_len`
   directly (`kv_cache_manager_v2.py:2410-2427`) for the identical situation
   (transmission-complete, empty `py_draft_tokens`, no context draft tokens,
   `max_total_draft_tokens=4`, target/non-draft manager, speculative decoding
   not disabled) returns **4** — i.e. the manager internally reserves 4 extra
   draft-token slots of GPU capacity that the scheduler's ledger never
   accounted for.

**Answer: The scheduler's budget accounting conservatively *under*-counts
relative to the manager's real reservation, in exactly the transmission-complete
transition window** (this matches and now quantifies the specific gap that
`scratchpad/kvcachev2_context/coverage_closure.md` §5/§6 had already flagged
as "Confirmed Mismatch #4" / a known divergence). The scheduler charges 0
extra draft tokens; the manager internally asks for `max_total_draft_tokens`
(here, 4) more capacity than that. Because manager admission
(`try_allocate_generation`) is the actual gate on real GPU pages — not the
scheduler's ledger — this divergence is **safe in the sense that the manager
is the binding constraint** (it will correctly refuse an allocation the
scheduler's ledger alone would have approved), but it does mean the
scheduler's `BudgetTracker` numbers cannot be read as an accurate prediction
of real GPU footprint during this specific transition window.

**Proves runtime behavior vs. confirms configuration:** **Mixed.** Parts 1–2
prove real scheduler decision-making (real `KVCacheV2Scheduler`,
real `BudgetTracker`, mocked manager return values) under adversarial/tight
budgets. Part 3 calls the real manager method directly but on a minimal
hand-built manager/request stand-in (`object.__new__(KVCacheManagerV2)` with
only the 3 attributes `_effective_draft_len` touches set, plus a `Mock()`
request) — it proves the *arithmetic* is real and unmodified, but does not
exercise a full, GPU-backed `try_allocate_generation` call to confirm the
manager actually turns that `4` into 4 real reserved GPU pages end-to-end (a
native/runtime test analogous to Q3, driving `try_allocate_generation` for a
disagg-transition request through a real GPU manager, would close that
residual gap — out of scope for this pass given time constraints).

---

## Q5 — Connector-rejection rollback: does rollback free/revert both target and draft, not just target?

**Test:** none written — **blocked by a hard source-level assertion**, no test can exercise this.

**Command:** N/A.

**Result:** N/A (not runnable).

**Observed:** `KVCacheManagerV2.__init__` (`kv_cache_manager_v2.py:832-834`):
```python
assert kv_connector_manager is None, (
    "kv_connector_manager is not supported for KVCacheManagerV2"
)
```
This is unconditional — there is no code path, flag, or configuration that
constructs a `KVCacheManagerV2` (target or draft) with a non-`None`
`kv_connector_manager`. This exact item was already flagged as **Out of
scope** by the prior `coverage_closure.md` audit (§1: *"`kv_connector_manager`
support (V2 does not support it) — Explicitly unsupported by hard assert"*),
and this session's direct re-read of the source confirms that finding still
holds on the current checkout: attempting to pass any KV connector (rejecting,
mock, or otherwise) to a V2 manager raises `AssertionError` at construction
time, before any request-level rollback logic could ever run.

**Answer: The question as posed does not apply to KVCacheManagerV2.**
KV-connector-triggered rollback (of the kind V1's `KvCacheConnectorManager`
integration supports) has no V2 equivalent to test — not "untested," but
**structurally absent**. Whatever rollback-symmetry guarantees exist for a
connector-initiated rejection are entirely a V1 concern; for V2 the only
rollback paths that exist are the ones already covered by Q2 (scheduler
eviction/suspension, mirrored to both managers, not always aligned) and Q7
(context capacity revert).

**Proves runtime behavior vs. confirms configuration:** **Blocked — confirmed
only by source (an unconditional assertion, not runtime behavior).** No
runtime evidence is possible or needed here: the assertion is unconditional,
so there is no configuration or code path under which this scenario could
occur on the current V2 implementation. If a future refactor adds V2
connector support, this question would need to be re-opened from scratch —
today's finding does not generalize to that hypothetical.

---

## Q6 — Deferred-request capacity: retained, draft skipped, and is this intentional?

**Test:** `tests/unittest/_torch/executor/test_scratch_q6_deferred_request_capacity.py` (3 tests: `test_never_started_request_defers_before_any_allocation`, `test_partially_started_request_retains_capacity_across_deferred_iteration`, `test_deferred_context_request_never_touches_draft_manager`)

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/_torch/executor/test_scratch_q6_deferred_request_capacity.py -q
```

**Result:** 3/3 PASS

**Observed:** Two distinct defer paths exist in
`KVCacheV2Scheduler._try_schedule_context_chunked` (`scheduler_v2.py`):
1. **Budget/min-chunk defer before any allocation** (`no_budget or
   fcfs_under_min` → `SKIP`, before `prepare_context` is even called). A
   request that has never received any capacity this call has nothing to
   retain — confirmed by asserting zero `prepare_context`/`resize_context`
   calls.
2. **Post-first-chunk defer** (`chunk_size <= 0` → `SKIP`, *after* an earlier
   call's `resize_context` already succeeded and granted real capacity). The
   exact source comment at this line states the design intent explicitly:
   > *"TODO: consider suspending first-chunk KVCache to release GPU pages.
   > Currently we skip without suspend to avoid pathological suspend/resume
   > cycles. suspend_request is only called from eviction
   > (`_try_evict_for_gen`)."*

   Running a real `KVCacheV2Scheduler` twice against the same mocked manager
   (first call: generous budget, first chunk succeeds and calls
   `resize_context` once; second call: budget below `chunk_unit_size`,
   forcing a defer) confirmed **zero** additional `resize_context` or
   `suspend_request` calls on the second, deferred call — the capacity
   granted by the first chunk is left completely untouched.
3. **Draft-manager isolation**: constructing a real `KVCacheV2Scheduler` with
   a `Mock()` draft manager and driving a deferred (never-started) context
   request through it recorded **zero** method calls of any kind
   (`draft_mgr.method_calls == []`) on the draft manager. This is not a
   special-cased defer-aware branch — `KVCacheV2Scheduler` simply has no
   context-scheduling code path that touches `draft_kv_cache_manager` at all
   (draft KV prep for context happens only via
   `KVCacheManagerV2._prepare_draft_resources`, dispatched at the
   `py_executor.py` level from `resource_manager.prepare_resources()`, only
   for requests present in that iteration's `ScheduledRequests` — a deferred
   request is absent from that set by construction).

**Answer: YES — a deferred request's already-granted target-manager capacity
is retained untouched, draft preparation is trivially skipped (a structural
consequence of the request not being scheduled that iteration, not a
special-cased decision), and the retention on the post-first-chunk path is an
explicit, source-commented, intentional bounded policy** (avoid
suspend/resume thrashing) — not accidental leftover state that happens to
survive because nothing touched it.

**Proves runtime behavior vs. confirms configuration:** **Proves real
scheduler-decision logic.** Same tier as Q2/Q4: real, unmodified
`KVCacheV2Scheduler` (imported and driven directly, not reimplemented),
mocked target/draft managers (no GPU). This proves the scheduler's own
call/no-call decisions precisely; it does not additionally prove what the
*native* manager does with a capacity that sits untouched across many
deferred iterations (e.g., whether it becomes evictable, or interacts with
`can_evict` in some multi-iteration native scenario) — that would require a
longer-running native/GPU scenario, out of scope for this pass.

---

## Q7 — Context rollback semantics: shrink or free, and do callers tolerate it?

**Test:** `tests/unittest/_torch/executor/test_scratch_q7_context_rollback_semantics.py` (4 tests)

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/_torch/executor/test_scratch_q7_context_rollback_semantics.py -q
```

**Result:** 4/4 PASS

**Observed:** `KVCacheManagerV2.revert_allocate_context` (`kv_cache_manager_v2.py:2525-2547`, real, unmodified) branches on the live cache's *current* `history_length` relative to the pre-iteration capacity (`pre_cap`) being reverted to:
```python
if kv_cache.history_length > pre_cap:
    self.free_resources(req)   # FREE: history already advanced past pre_cap
    return
history_length = min(kv_cache.history_length, pre_cap)
kv_cache.resize(pre_cap, history_length)   # SHRINK: request stays alive
if pre_cap > 0:
    kv_cache.suspend()
```
Driving this real method body (mocked native `_KVCache`, same technique as the
repo's own `test_kv_cache_v2_capacity_only.py`) against both conditions
confirmed:
- **SHRINK branch** (`history_length <= pre_cap`): `resize(pre_cap,
  history_length)` + `suspend()` called; the `kv_cache_map` entry is left in
  place (same object, still active) — the request is recoverable at its
  pre-iteration capacity.
- **FREE branch** (`history_length > pre_cap`): `free_resources(req)` called
  instead; no resize/suspend.
- The `py_ctx_pre_resize_cap` marker is cleared **unconditionally**
  (`kv_cache_manager_v2.py:2530`, before either branch, and even before the
  `kv_cache is None or not kv_cache.is_active` early-return) — a second
  revert call on the same request is always a guaranteed no-op regardless of
  which branch the first call took, or whether the cache was already
  inactive.
- No growth to undo (`pre_cap >= kv_cache.capacity`) and already-inactive
  cache are both confirmed no-ops (no resize/suspend/free_resources calls).

The only production caller, `py_executor.py`'s `_revert_ctx_alloc`
(`py_executor.py:3474-3477`, invoked from
`_revert_deferred_disagg_gen_init_alloc` for disagg-transfer-admission
candidates that lost the admission race, `py_executor.py:3567-3586`), is a
blind for-loop calling `revert_allocate_context(req)` once per dropped
request — it does not branch on, or even inspect, which outcome occurred.

**Answer: Both outcomes exist and are selected deterministically by whether
committed history has already advanced past the target capacity — SHRINK
(recoverable, capacity/state preserved) when it hasn't, FREE (start over)
when it has. The caller tolerates both uniformly**: it doesn't need to know
which happened, because the *next* scheduling attempt for that same request
re-enters `prepare_context`, whose real precondition already handles a
missing `kv_cache_map` entry (triggers a fresh `_create_kv_cache`, per the
Create-path evidence already established by the prior `manager.md` audit) —
so the FREE outcome is not a caller-side crash risk, only a "lose reuse
credit, start the context from scratch" cost relative to SHRINK.

**Proves runtime behavior vs. confirms configuration:** **Proves real manager
method logic.** Real, unmodified `revert_allocate_context` method body
(constructed via `KVCacheManagerV2.__new__`, the same "real method / minimal
attribute set / mocked native cache" technique the repo's own
`test_kv_cache_v2_capacity_only.py` already uses for the adjacent
`update_resources` method), with a `MagicMock` standing in for the native
`_KVCache`. This proves the Python-level branching logic and its caller
contract precisely. It does not additionally prove the *native* `resize()`
call actually succeeds/behaves this way against real GPU pages in the SHRINK
branch (Q3 already establishes that ordinary native resizes, including
shrinks, work against a real native cache; this probe does not re-combine
that with the revert-specific `resize(pre_cap, history_length)` two-argument
call signature on a real native object — a residual gap, low-risk given Q3's
adjacent coverage, but not closed here).

---

## Atomicity re-verification — failed native resize leaves no orphaned pages

**Test:** `tests/unittest/kv_cache_manager_v2_tests/test_scratch_atomicity_failed_resize_page_accounting.py::test_failed_resize_leaves_no_orphaned_pages_in_manager_pool_stats`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/kv_cache_manager_v2_tests/test_scratch_atomicity_failed_resize_page_accounting.py -q -s
```

**Result:** PASS

**Observed (real GPU allocation, 8 MiB quota, 32 tokens/block, 1 layer):**
```
[Atomicity] baseline before failed resize: available=512 unavailable=2 evictable=0
[Atomicity] immediately after failed resize: available=510 unavailable=2 evictable=0
```
This strengthens Q3's original atomicity evidence (which only showed the
request's own `capacity` counter unchanged, and inferred leak-freedom
indirectly via a *later*, post-`close()` fresh sequence reaching full quota).
Here, the manager's own real GPU-backed pool statistics
(`KVCacheManager.get_and_reset_iteration_peak_block_stats`, the same native
binding surface backing production `KVCacheManagerV2.get_kv_cache_stats()`)
are queried **immediately after the failure, before any `close()`** of the
still-live request. The `unavailable` (currently committed/held block) count
is identical (2) before and after the failed `OutOfPagesError` resize, and a
further immediate no-op query confirms no delayed/async change either.

One methodological note recorded in the test itself: the `available` field
of this stats API is a **peak** (high-water-mark) statistic over the interval
since the last reset, not an instantaneous snapshot — confirmed empirically
when the very first (baseline) query reported `available=512` even though
only 510 blocks were actually free at that instant, because 512 was the peak
free-block count observed earlier in that same window (before the first
chunk was allocated). Only `unavailable` was used as the atomicity signal for
this reason.

**Answer: CONFIRMED — a failed native resize does not partially or
non-atomically commit any pages before discovering it cannot satisfy the
request.** The manager's real-time committed-page accounting is unchanged by
a failed resize attempt, checked directly (not inferred from a later
allocation).

**Proves runtime behavior vs. confirms configuration:** **Proves real native
runtime behavior**, same tier as Q3 (real GPU-backed `KVCacheManager`/
`_KVCache`, real `OutOfPagesError`, `KV_CACHE_MANAGER_V2_BACKEND=cpp` default
backend) — strictly stronger than Q3's original evidence because it queries
the manager's own committed-page accounting directly rather than inferring
leak-freedom from a subsequent fresh allocation.

---

## Q8 — Prefix reuse with separate target/draft caches: is draft history explicitly prepared/validated?

See the dedicated "Q8" subsection below for the full runtime-test result;
the mechanism itself was already traced from source by
`scratchpad/kvcachev2_context/topology_and_prefix_reuse.md` (Task 2) prior to
this session and is summarized here rather than re-derived.

**Source-level mechanism (from `topology_and_prefix_reuse.md`, re-verified
this session by direct re-read of the cited lines, not re-traced from
scratch):** For two-model (Variant A) and one-model-separate-layout (Variant
B) topologies, the draft manager's own `_KVCache` is created via
`_prepare_draft_resources` with `input_tokens=None` **unconditionally**
(`kv_cache_manager_v2.py:2789-2797` — not gated on `self.enable_block_reuse`
the way the target's is), meaning the draft's own reuse-matching is never
attempted and `num_committed_tokens` starts at `0`. Separately, the draft's
`resize(capacity)` call (`kv_cache_manager_v2.py:2810-2821`) passes only one
argument — no `history_length` — unlike `prepare_disagg_gen_init`'s two-arg
`resize(capacity, prompt_len)` call. Meanwhile the draft's
`context_current_position`/`context_chunk_size` bookkeeping is copied
(two-model, via `model_drafter.py:140-149`) or *shared* (one-model, same
`req` object) from the **target's** post-reuse chunk bounds — i.e., the
draft engine's one-shot context forward pass is told to skip computing
exactly the range the target's reuse match skipped, without the draft
manager's own cache ever having populated that range via its own reuse or
its own forward pass.

Whether this is actually consulted safely (native `history_length`/attention
metadata gating the draft's effective KV span to what its own cache
actually holds) or is a live correctness gap (draft attention reading
unpopulated/stale pages for the reused-prefix range) was **explicitly left
as an open question** by that source-only audit — it required either reading
native `.cpp` `resize()`/`historyLength` semantics (not read), or a
model-level runtime test (not previously run). This session attempted the
latter.

**Test attempted:** `tests/unittest/_torch/speculative/hw_agnostic/test_scratch_q8_v2_prefix_reuse_draft_correctness.py::test_v2_two_model_spec_decode_prefix_reuse_output_matches_no_reuse`

**Command:**
```
docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m pytest \
  tests/unittest/_torch/speculative/hw_agnostic/test_scratch_q8_v2_prefix_reuse_draft_correctness.py -q -s
```
(Modeled directly on the repo's existing
`tests/unittest/_torch/speculative/hw_agnostic/test_kv_cache_reuse.py`, which
already exercises this exact target/draft model pair —
`EAGLE3-LLaMA3.1-Instruct-8B` + `llama-3.1-model/Llama-3.1-8B-Instruct`,
`eagle3_one_model=False` i.e. Variant A two-model topology — but against the
*default* KV cache manager; this probe adds `use_kv_cache_manager_v2=True`
explicitly and a reuse-disabled control run for direct output comparison,
neither of which the existing test does.) Both model checkpoints are present
under this environment's `LLM_MODELS_ROOT`
(`/home/scratch.trt_llm_data_ci/llm-models`), and the H200 has 150 GB of GPU
memory, well above the existing test's own 35 GB gate.

**Result: BLOCKED — environment limitation, not a KVCacheV2/prefix-reuse
finding.** Model loading succeeded (both checkpoints loaded, weights
resolved), but the first generation step failed with:
```
[TensorRT-LLM][ERROR] CUDA runtime error in
cudaOccupancyMaxActiveBlocksPerMultiprocessor(...
mmha::masked_multihead_attention_kernel<...>...): no kernel image is
available for execution on the device
(.../decoderMaskedMultiheadAttentionLaunch.h:276)
```
This is the generation-phase masked multi-head attention kernel reporting
that the **currently-installed** native extension in this dev container has
no compiled kernel image for this GPU's SM architecture (H200, SM 90) for
this kernel instantiation — i.e. the installed `.so` predates or otherwise
excludes an SM-90 build of this particular generation attention kernel path.
This is unrelated to KVCacheManagerV2, target/draft prefix reuse, or
anything under investigation in this session; it is a pre-existing state of
the installed binary in this container, not something introduced by this
investigation (no production code or build artifacts were modified this
session — see the note below on an aborted rebuild attempt). Rebuilding the
extension to add SM-90 kernel images was explicitly out of scope
("do not rebuild the full project"), so this could not be worked around.

**Note on an aborted rebuild attempt:** partway through this session, an
earlier attempt to run these probes via the `trtllm-dev test` wrapper
(instead of raw `docker exec ... python3 -m pytest`, per the coordinator's
mid-task redirect) was found to trigger `trtllm-dev`'s default
auto-build/staleness-check path, which invoked a full
`scripts/build_wheel.py` C++/CUDA rebuild — several redundant instances of
which ran briefly in parallel before being identified and killed (via `kill`
on each PID, both on the host and inside the container) once the violation
of the "do not rebuild the full project" constraint was noticed. The
in-progress compiles were terminated before any object files were linked
into the installed extension or any output installed over the pre-existing
`.so` — confirmed by successfully re-importing `tensorrt_llm.bindings` and
re-running the already-passing Q1-Q4 suite (6/6 still pass) immediately
afterward, and by `git status` showing no changes outside this session's own
new test/report files. All subsequent test execution in this session (Q6,
Q7, atomicity, and this Q8 attempt) used the original, unmodified
`docker exec -w /code/tensorrt_llm tensorrt_llm-devel-allim python3 -m
pytest ...` invocation instead. The SM-90 MMHA kernel-image gap encountered
here is therefore a **pre-existing** limitation of the container's installed
extension, not a side effect of the aborted rebuild.

**Answer: Q8's model-level correctness claim remains unproven at runtime in
this environment**, exactly as the original task instructions anticipated
for the "no model artifact available" case — except here the blocker was a
missing SM-90 kernel image in the pre-built extension, not a missing model
artifact (the models themselves were available and loaded correctly). Per
the source-only trace above (carried over from
`topology_and_prefix_reuse.md`, re-verified by direct re-read this session):
the mechanism for a draft-manager KV state gap under target prefix reuse is
**structurally reachable** (draft manager's own cache is never populated for
the reused-prefix range, either by its own reuse-matching or by its own
forward pass, while its chunk bookkeeping is copied/shared from the target's
post-reuse position) — but **whether this manifests as silent
corruption, a safely-gated reduced-context draft, or something else
entirely is not proven by any runtime evidence gathered in this session.**
Do not infer a corruption bug from the separate-caches topology alone.

**Proves runtime behavior vs. confirms configuration:** **Blocked.** The
precise blocker is: this container's currently-installed
`tensorrt_llm.bindings` native extension lacks a compiled generation-phase
MMHA kernel image for SM 90 (H200) for the code path exercised by two-model
EAGLE3 speculative decoding generation. Resolving this requires either a
full or targeted extension rebuild (explicitly out of scope for this
investigation) or access to a container/environment with a more complete
SM-90 kernel build. The mechanism-level finding remains
**confirmed only by source**, exactly as it was before this session's
attempt.

---

## Classification labels used from this point on

Every claim below (new, and retroactively for Q1-Q8/atomicity in the summary
table) is tagged with exactly one of:
- **runtime-proved on the real production path** — real, unmodified
  production code, real GPU allocation, invoked the way production actually
  invokes it (not a hand-built stand-in, not a forced flag, not a
  monkeypatched internal).
- **scheduler/manager logic proved with mocks** — real scheduler/manager
  method bodies, but mocked native objects or hand-built minimal stand-ins.
  Proves Python-level branching/call logic, not native runtime behavior.
- **source-only** — confirmed only by reading code, no execution.
- **blocked** — attempted but could not run (missing kernel image, hard
  assertion preventing construction, etc.).

## Q1 (re-opened) — the original question's premise does not hold in current production code

The originally-committed Q1 probe
(`tests/unittest/kv_cache_manager_v2_tests/test_scratch_q1_spec_decode_capacity.py`)
forced `_should_create_separate_draft_kv_cache = lambda: True` on a hand-built
`KvCacheCreator`. That is **not** evidence about what a real two-model
construction path does, because the boolean was never actually computed by
real code from a real config — it was overridden. This session redid Q1
without forcing anything, driving the real, unmodified gating chain in
`KvCacheCreator` (`_util.py`) with real `DecodingBaseConfig` subclasses.

**New finding, which reframes the entire question:** genuine two-engine
("two-model", two separate forward-pass model instances) speculative
decoding is **not reachable through any currently-supported public LLM API
config**:
- EAGLE3 two-model (`eagle3_one_model=False`) is deprecated and silently
  coerced back to `True` by `EagleDecodingConfig.validate_eagle_config`
  (`llm_args.py:2228-2233`) — confirmed by constructing the config with
  `eagle3_one_model=False` and reading back `cfg.eagle3_one_model is True`
  and `cfg.spec_dec_mode.is_eagle3_one_model()`
  (`test_scratch_q1_real_two_model_gate.py::test_eagle3_two_model_is_deprecated_and_coerced_to_one_model`,
  **runtime-proved on the real production path** — real Pydantic validator,
  real config class).
- `DraftTargetDecodingConfig` has a private `_draft_target_one_model`
  attribute defaulting to `True` (`llm_args.py:2592`); grepping the entire
  `tensorrt_llm/` tree finds no field, setter, or code path anywhere that
  ever sets it `False` — the only two references are the attribute's own
  default and the `spec_dec_mode` property that reads it. So
  `DraftTargetDecodingConfig(...)`'s default `spec_dec_mode` is always
  `DRAFT_TARGET_ONE_MODEL`
  (`test_scratch_q1_real_two_model_gate.py::test_draft_target_default_is_one_model_no_public_two_engine_path`,
  **runtime-proved on the real production path**), never the genuine
  two-engine `DRAFT_TARGET` value (which exists in the enum and has its own
  `use_one_engine() == False` predicate, but is dead code reachable only by
  hand-setting a private attribute nothing in production ever sets).
- A companion attempt to force this through anyway via real `LLM(...)`
  construction (`tests/unittest/_torch/speculative/hw_agnostic/test_scratch_q1_v2_two_model_real_budget.py`,
  `DraftTargetDecodingConfig` + real EAGLE3/Llama-3.1 checkpoints,
  `use_kv_cache_manager_v2=True`) is **blocked**: it hit the same
  environment-level missing-SM-90-kernel-image failure as Q8
  (`CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor ...
  no kernel image is available for execution on the device`) during
  `LLM.__init__`'s warmup/generation step — this is a second, independent
  reproduction of the same blocker in this environment this session (see
  the Q8 section below). Separately, and independently of that blocker, a
  `KVCacheManagerV2.__init__`-observation monkeypatch installed in the test
  process could never have captured anything anyway, because manager
  construction runs inside the MPI/IPC executor's **worker subprocess**, not
  the pytest process — this is recorded as a methodological dead end for
  future attempts, not a finding about KVCacheManagerV2 itself.

**Given genuine two-engine mode is unreachable**, the actually-reachable
production scenario for two independent V2 managers is the *one-model*
separate-draft-KV-cache path (`_should_create_separate_draft_kv_cache()`,
same engine, different KV layout for the draft sub-network — EAGLE3-one-model,
DraftTarget-one-model, MTP-eagle-one-model). Driving the real,
unforced gate for this path with a real default `DraftTargetDecodingConfig`:
```python
creator._should_create_separate_draft_kv_cache()  # -> True, unforced
creator._needs_gpu_kv_cache_budget_split(max_seq_len=2048)  # -> True, unforced
```
(`test_scratch_q1_real_two_model_gate.py::test_real_one_model_separate_draft_cache_gate_enables_v2_gpu_split_by_default`,
**runtime-proved on the real production path** for the gate logic itself —
`KvCacheCreator` is hand-built via `__new__` with minimal attributes, the
config objects and the methods under test are real and unforced).

Feeding real per-token `CacheCost` values (80/20 target/draft split, same
substitution technique the original committed Q1 probe already used and
documented, since no HF model is loaded) into the real, unmodified
`_split_kv_cache_budget_for_draft`/`_compute_draft_budget_shares`:
```
[Q1-split] B=4294967296 target_budget=3435973837 draft_budget=858993459
```
`target_budget + draft_budget == B` exactly, and the split is proportional
to the 80/20 per-token cost ratio (`draft_budget ≈ 0.2·B`,
`target_budget ≈ 0.8·B`), not an equal 50/50 split and not "each
independently gets the full B"
(`test_scratch_q1_real_two_model_gate.py::test_real_split_arithmetic_for_one_model_separate_draft_cache`,
**scheduler/manager logic proved with mocks** — the split arithmetic is real
and unmodified, but its `CacheCost` leaf inputs are supplied directly rather
than derived from a loaded model, since no model-config-bearing engine was
constructed).

**Net effect: Q1's original answer is superseded.** The prior committed
answer ("B is split 80/20 in supported two-model spec-decode") is true only
for the one-model separate-draft-cache scenario — the *only* one actually
reachable in production today — not for a genuine two-engine setup, which
does not exist as a constructible configuration in the current codebase.
The equal-budget assert this session traced at `_util.py:2081-2085`
("KVCacheManagerV2 does not support two-model speculative decoding with
separate draft GPU budgets") is dead code under real configs: its guarding
condition (`draft_kv_cache_config is not None` while
`self._draft_model_engine is not None`) can only occur if
`_needs_gpu_kv_cache_budget_split()` is *also* `True` for a genuine
two-engine config, and no real config produces genuine two-engine mode with
that gate `True` — confirmed by the exhaustive check above of every
`use_one_engine() == False` path currently reachable through public config
(there are none).

---

## Q2 (strengthened) — later-chunk out-of-pages failure through the real scheduler + real manager

Q3/atomicity (below, preserved unchanged) proves the *native* resize call is
atomic on failure, driven directly on a bare native cache with no scheduler
involved, and no prior committed chunk. This session added a probe that
drives a **later** (non-first) chunk's failure through the real,
unmodified `KVCacheV2Scheduler` scheduling loop *and* the real
`KVCacheManagerV2.prepare_context`/`resize_context`, against a real
GPU-backed manager (`max_gpu_total_bytes` sized to exactly 4 blocks / 16
tokens) — not a mocked manager, and not a direct `kv_cache.capacity`
mutation.

**Test:** `tests/unittest/_torch/executor/test_scratch_q2_later_chunk_oop_full_path.py`

**Setup:** a 24-token context request scheduled in 8-token chunks. Chunk 1
(8 tokens) succeeds via a real `KVCacheV2Scheduler` call, consuming half the
pool. Chunk 2 requests all 16 remaining tokens (target capacity 24, needing
6 blocks against a 4-block pool) — driven through a **second**, real
`KVCacheV2Scheduler` call against the same manager, exercising
`resize_context` with `req.is_first_context_chunk = False`.

**Observed (real GPU, `-s` output):**
```
[Q2-full] chunk2 scheduled context_requests=0
[Q2-full] request state after failed later-chunk resize: capacity=8 is_active=True context_current_position=8 py_ctx_pre_resize_cap=0 (was 0 after chunk 1)
[Q2-full] manager pool stats: baseline unavailable=4 after_failure unavailable=4
```

**Answer:**
1. **Atomicity holds through the full scheduler+manager call chain, not
   just the bare native call**: the request's own `kv_cache.capacity` is
   unchanged (`8`, not partially grown toward `24`), and the manager's own
   committed-page count (`unavailable`) is identical before and after the
   failed later-chunk attempt (`4` both times) — the same atomicity
   property Q3 established directly on the native object, now confirmed
   reached via the real scheduler → real Python manager → real native
   resize call chain.
2. **First-chunk vs. later-chunk asymmetry, confirmed for real**:
   `resize_context` (`kv_cache_manager_v2.py:2671-2674`) only suspends the
   cache on failure when `req.is_first_context_chunk` is `True`. This probe's
   failure is on chunk 2 (`is_first_context_chunk=False`), and the cache is
   confirmed to remain `is_active=True` after the failure — unlike a
   first-chunk failure, which the manager suspends. This is a real,
   previously only source-inferred behavioral asymmetry, now directly
   observed.
3. **`py_ctx_pre_resize_cap` is untouched by the failed call** (it only gets
   written on `resize_context` success) — it still reflects chunk 1's grow,
   not the failed chunk-2 attempt. No stale-marker corruption from the
   failure itself.
4. **Retry recovery is real, not just structurally inferred**: after the
   failed 16-token chunk-2 attempt, a third, smaller retry chunk (4 tokens —
   the one remaining free block) was driven through a fresh
   `KVCacheV2Scheduler` call against the *same*, still-live manager and
   request, and it succeeded (`capacity` grew from `8` to `12`) — confirming
   the failed attempt did not corrupt or wedge the request; the scheduler
   naturally retries a differently-sized chunk on the next call rather than
   requiring any special recovery path.

**Classification: runtime-proved on the real production path** — real,
unmodified `KVCacheV2Scheduler` and `KVCacheManagerV2`, real GPU-backed
native cache and pool-stats query, real `OutOfPagesError`. The only
non-production element is the request object itself (`_ContextRequest`, a
minimal dataclass satisfying the same real method contracts this branch's
own `test_kv_cache_manager_v2.py::_run_context` helper already relies on,
augmented with a handful of extra attributes the scheduler additionally
reads) — this is the same tier the repo's own manager-level tests already
use for driving real manager methods without a full `LlmRequest`/executor
stack.

---

## Q4 (strengthened) — real `try_allocate_generation` for disagg transmission-complete / empty-draft-token admission

The existing `test_kv_cache_v2_capacity_only.py::
test_disagg_gen_transition_reserves_target_drafts_without_context_drafts`
only calls `_effective_draft_len`/`_required_gen_capacity` in isolation on a
bare `SimpleNamespace`, proving the arithmetic but not that the resulting
number costs real GPU pages.

**Test:** `tests/unittest/_torch/executor/test_scratch_q4_disagg_gen_real_reservation.py`

**Method:** two real, GPU-backed `KVCacheManagerV2` instances, each primed
with a real, active 4-token native cache via the real context path
(`prepare_context`/`resize_context`, same technique as Q2/`_run_context`),
then driving the real, unmodified `try_allocate_generation` for a disagg
transmission-complete request with empty `py_draft_tokens` — the exact
admission scenario the original Q4 answer described. Pool sized to exactly
2 blocks (8 tokens).

**Observed (real GPU, `-s` output):**
```
[Q4-real] disabled-speculation admission: ok=True capacity=5
[Q4-real] enabled-speculation admission: ok=False capacity=4
```

**Answer: `_effective_draft_len`'s reservation is real GPU-page cost, not
inert bookkeeping.** With speculative decoding disabled (`draft_len=0`),
growing the same starting 4-token cache by `1` succeeds (target capacity 5,
fits in 2 blocks). With speculative decoding enabled and no context draft
tokens (`_effective_draft_len` falls back to `max_total_draft_tokens=4`),
growing the *same* starting cache now targets capacity 9 — needing 3 blocks
against the same 2-block pool — and is rejected with a real
`OutOfPagesError`-driven `False` return. The extra reserved tokens are the
sole difference between the two runs, and they are what tips admission from
success to a real rejection. Failure is atomic here too: `capacity` stays
at the pre-attempt `4`, and the cache remains active (consistent with
Q3/Q2's atomicity findings, now confirmed for this specific admission path).

**Classification: runtime-proved on the real production path** — real,
unmodified `KVCacheManagerV2.try_allocate_generation`, real GPU-backed
native cache, real `OutOfPagesError`. The request object is a
`SimpleNamespace` with only the specific fields `try_allocate_generation`
and `_effective_draft_len` actually read (mirroring the existing repo
test's technique) — not a full `LlmRequest`, but every method invoked on it
is real, unmodified code, and the admission outcome (success vs. real
native rejection) is the thing under test, not simulated.

---

## Native suspend-failure reachability (new)

Investigated whether there is any *supported* way to make `suspend()` fail
partway through, to determine if the main/draft-suspend-divergence risk
identified by source reading in
`scratchpad/kvcachev2_scheduler_manager_contract.md` (§1) is empirically
reachable.

**What was checked:**
- `KvCache::suspend()` (`kvCache.cpp:521-559`) has no allocation-failure
  path (unlike `resize()`) — it only converts already-locked
  `SharedPageLock`s into `PageHolder`s (a release, not an acquire). Its only
  guards are `TLLM_CHECK_DEBUG` invariant checks.
- `TLLM_CHECK_DEBUG` (`cpp/include/tensorrt_llm/common/assert.h:56-64`) is
  gated by `tensorrt_llm::DebugConfig::isCheckDebugEnabled()`
  (`cpp/tensorrt_llm/common/assert.cpp:21-32`), a runtime flag read once
  from the `TLLM_DEBUG_MODE` environment variable (`"1"` to enable) — not a
  compile-time-only debug-build check, but still **only a checker of an
  already-true invariant**, not a fault-injection mechanism. Enabling it
  cannot manufacture a failure; it can only surface one if some other,
  independent bug already violated the invariant.
- Searched `cpp/tests/unit_tests/batch_manager/` (including
  `kvCacheManagerV2TestUtils.h`, the dedicated V2 test-utilities header) for
  any fault-injection, error-injection, or CUDA/stream-error-simulation
  hook applicable to `suspend()` or its underlying page/event machinery
  (`recordEventScope`, `notifyFinish`, `SharedPageLock::hold()`) — found
  none.
- Did not attempt to monkeypatch a private internal to fake a throw (would
  violate the "no monkey-patched private gate" constraint and would not
  constitute evidence about production reachability).

**Conclusion, stated precisely per the requested framing:** scheduler-side
`_suspend_request` (`scheduler_v2.py:1053-1065`) is non-transactional
conditional on a manager failure — if the target manager's `suspend_request`
call were to raise after some other component had already reached a
partially-suspended state, the draft manager's `suspend_request` call would
never execute, per straightforward inspection of the (unguarded) two
sequential calls. **Native reachability remains unproven**: no supported
fault hook exists in this codebase to actually trigger `suspend()` failing
after a target's suspension has completed, so this is a statement about the
code's structure under a hypothetical, not a demonstrated production
mismatch.

**Classification: source-only** (for the structural non-transactional-call
claim) **combined with a confirmed absence of any supported way to make it
runtime-provable** — this is not a "blocked" result in the sense of an
environment limitation; it is a checked, negative result: the codebase
genuinely does not currently expose a way to test this scenario at the
native level.

---

## Q8 (re-confirmed) — SM-90 kernel-image blocker persists; mechanism traced one level deeper

**Blocker re-confirmed, independently, twice this session** (not merely
assumed unchanged): the exact same
`CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor ...
no kernel image is available for execution on the device`
(`decoderMaskedMultiheadAttentionLaunch.h`) was hit again by the Q1
real-two-model-budget `LLM(...)` construction+generation attempt this
session (`test_scratch_q1_v2_two_model_real_budget.py`), using different
models in a different code path (`DraftTargetDecodingConfig` +
Llama-3.1-8B/Llama-3.2-1B, vs. the original Q8 probe's EAGLE3+Llama-3.1-8B).
Re-running the original Q8 test file directly was not repeated this session
(it would reproduce the same environment-level blocker at higher GPU-time
cost, already re-confirmed via the Q1 attempt) — this is recorded
explicitly as a deliberate choice, not an unverified assumption. **The Q8
test file is now marked `@pytest.mark.skip` with this precise, re-confirmed
reason so it no longer surfaces as a collection/run failure in normal
pytest runs**; both required model checkpoints load successfully, so this
remains an environment/build limitation, not a KVCacheManagerV2 or
prefix-reuse defect.

**Mechanism traced one level deeper (source-only, new this session):** the
prior session's finding established that the draft manager's own cache is
created without reuse-matching (`num_committed_tokens=0` for the
reused-prefix range) while `context_current_position` on the draft's
synthetic context request is copied from the target's post-reuse chunk
bounds (`model_drafter.py:145-148`,
`new_request.context_current_position = begin_compute`). This session
traced where that value is actually consumed at the forward-pass level:
`model_engine.py:5398-5411` sets `begin_compute = request.context_current_position`
for **every** context request (target or draft — the same code path handles
both; `self.is_draft_model` only branches later bookkeeping, not this
`begin_compute` derivation) and uses it directly to compute `position_ids`
(`range(begin_compute, begin_compute + len(prompt_tokens))`) and to select
which token range is fetched via `get_tokens_range(0, begin_compute,
end_compute)`. This confirms, at the Python/position-assignment level, that
the draft's forward pass is told "positions `[0, begin_compute)` are
already computed" — it does not request the model recompute them — for
exactly the range the draft's own cache never populated.

**What remains unresolved (explicitly, not inferred):** whether the
attention **kernel's** own cached-token-count / KV-read-range argument for
this forward pass is *also* driven from this same Python-level
`context_current_position` value (in which case the kernel would be told to
attend over `begin_compute` valid cached positions that are actually
unpopulated in the draft's cache — a live, reachable gap) or is
independently derived from the draft manager's own native
`kv_cache.history_length` (which the prior session already established is
`0` for this range, in which case a mismatch there would either be
independently caught/gated, or would itself indicate a different bug). This
session did not read the attention-backend kernel-argument construction
(e.g. the TRTLLM attention plugin's `pastKeyValueLength`/cached-length
input derivation) far enough to resolve this — time did not permit going
past the `position_ids`/`begin_compute` handoff traced above.

**Classification: blocked** for the runtime comparison (re-confirmed
environment limitation, not assumed); **source-only, and deepened but still
not fully resolved**, for the mechanism. Per the original instruction: do
not infer corruption from this alone — the `position_ids` handoff is
necessary evidence of a structurally-reachable gap but not sufficient proof
that the attention kernel actually reads unpopulated pages.

---

## Summary

| Q | Answer | Evidence tier |
|---|---|---|
| Q1 (superseded — see "Q1 (re-opened)") | Original claim ("B splits 80/20 in supported two-model spec-decode") was based on a **forced** `_should_create_separate_draft_kv_cache=True` flag, not a real gate decision | scheduler/manager logic proved with mocks (forced flag — weak; superseded below) |
| Q1 (re-opened) | Genuine two-engine spec decode is **unreachable via any current public config** (EAGLE3 2-model deprecated+coerced; DraftTarget 2-model has no public setter) — confirmed by real, unforced config construction. The actually-reachable "two-manager" scenario (one-model separate-draft-cache) DOES split GPU budget, unforced (`_should_create_separate_draft_kv_cache()==True` by default), proportionally to per-layer cost (80/20 CacheCost inputs → 80/20 byte split, real split arithmetic) | Gate decision: runtime-proved on the real production path. Split arithmetic: scheduler/manager logic proved with mocks (real function, supplied CacheCost leaves). Full `LLM()` two-model GPU-budget capture: blocked (SM-90 kernel image gap + worker-subprocess isolation) |
| Q2 (self-eviction alignment) | NOT always aligned — a failed draft-mirror suspend leaves target suspended, draft still active, exception uncaught | scheduler/manager logic proved with mocks (real scheduler code; mocked managers, no GPU) |
| Q2 (later-chunk OOP, strengthened) | Atomicity and first/later-chunk asymmetry confirmed through the FULL real scheduler→manager→native call chain (not a direct native mutation): capacity and manager page-stats unchanged by the failed later chunk; cache stays ACTIVE (unlike a first-chunk failure, which suspends); retry with a smaller chunk succeeds immediately after | runtime-proved on the real production path |
| Q3 | Recoverable, and cleaned up consistently — capacity preserved, no page leak, close/free succeed | runtime-proved on the real production path |
| Q4 (arithmetic) | Conservatively differs — scheduler charges 0 extra draft tokens, manager reserves `max_total_draft_tokens` (4) more | scheduler/manager logic proved with mocks (bare `SimpleNamespace`, methods called in isolation) |
| Q4 (strengthened — real reservation) | The extra reserved tokens are real GPU-page cost, not inert bookkeeping: identical starting cache/pool admits with speculation disabled (draft_len=0) but is rejected with a real `OutOfPagesError` when enabled (draft_len=4) — the draft-length delta alone flips admission | runtime-proved on the real production path |
| Q5 | Not applicable to KVCacheManagerV2 — `kv_connector_manager` is hard-asserted to `None` at construction; no rollback-rejection path exists to test | blocked (confirmed by an unconditional source-level assertion) |
| Q6 | YES — deferred requests retain already-granted capacity untouched; draft prep is structurally skipped (not scheduler-mirrored for context); post-first-chunk retention is an explicit, commented, intentional policy | scheduler/manager logic proved with mocks (real scheduler code; mocked managers, no GPU) |
| Q7 | Both SHRINK (recoverable, capacity/state preserved) and FREE (start over) exist, selected deterministically by whether history has passed the revert target; callers tolerate both uniformly via `prepare_context`'s missing-entry handling | scheduler/manager logic proved with mocks (real manager method logic; mocked native cache) |
| Atomicity | CONFIRMED — failed native resize does not partially/non-atomically commit pages; manager's own committed-page count unchanged, checked directly (not inferred) | runtime-proved on the real production path |
| Native suspend-failure reachability | `_suspend_request` is structurally non-transactional (unguarded sequential target/draft calls) — confirmed by reading the code. No supported fault-injection hook exists anywhere in the codebase to actually trigger a `suspend()` failure, so native reachability of the divergence remains unproven, not merely unstated | source-only, with a confirmed absence of any way to make it runtime-provable (checked: assert.h/assert.cpp gating, cpp/tests/unit_tests/batch_manager/ for fault hooks — none found) |
| Q8 (mechanism) | Draft cache never populated (own reuse/forward pass) for the target's reused-prefix range; traced one level deeper this session — draft's `position_ids`/token-fetch range for its context forward pass ARE driven by `context_current_position` (copied from target), i.e. told "already computed" for that exact unpopulated range. Whether the attention KERNEL's own cached-length argument is also driven by this same value (live gap) or independently reconciled against the draft's native `history_length` (safely gated) is unresolved | source-only (deepened, still incomplete) |
| Q8 (model-level runtime) | Blocked — missing SM-90 kernel image in this environment's installed extension, re-confirmed independently this session via the Q1 real-budget `LLM()` attempt hitting the identical error in a different code path. Do not infer corruption from the topology alone | blocked |

No test was blocked by a missing model artifact or unsupported configuration
for Q1-Q4 — once the dev container's Python dependencies were installed, a
single H200 GPU was sufficient, since none require loading an actual HF
model (per-layer KV costs were supplied directly as `CacheCost` values for
Q1, matching the existing repo test convention for that same function). Q5
is blocked by an unconditional source-level assertion (not an environment
limitation). Q8 is blocked by a pre-existing SM-90 kernel-image gap in this
container's installed extension, despite both required model checkpoints
being available under `LLM_MODELS_ROOT` and ample GPU memory (150 GB on the
H200).

---

## Implications for the proposed shared-manager / separate-pools refactor

Claims below are marked **[runtime]** where directly supported by this
session's or the prior session's runtime evidence, and **[source-only]**
where they rest on source reading without runtime confirmation.

1. **[runtime]** Target and draft managers are already fully independent
   native objects (separate `impl`, separate `IndexMapper`, separate pools) —
   Q1 confirms both construct and allocate real GPU memory independently
   under a shared byte-budget split. A refactor to one manager with separate
   memory pools would need to either preserve this independence internally
   (two pool groups under one Python object) or explicitly decide to
   *change* the isolation properties currently relied upon by callers.

2. **[runtime]** The mirrored-call contract between scheduler and the two
   managers (suspend/free) is **not currently atomic or rollback-safe**
   (Q2): a failure in the second (draft) call of a sequential pair leaves
   the two managers in different states, uncaught. A shared-manager refactor
   that folds target/draft into one object would **structurally eliminate
   this entire class of divergence** (Q2's finding), since there would be
   only one call, not two — this is a concrete correctness argument in favor
   of the refactor, not just a simplification.

3. **[runtime]** Q4's scheduler/manager accounting divergence (disagg
   transmission-complete window) is manager-arithmetic, not a target/draft
   split issue per se — it would likely persist in a shared-manager design
   unless the accounting formula itself is revisited, since it stems from
   `BudgetTracker` vs. `_effective_draft_len` disagreeing, not from having
   two manager objects.

4. **[runtime]** Q6 and Q7's capacity-retention and revert-shrink-vs-free
   behaviors are properties of a *single* `KVCacheManagerV2` instance's own
   internal bookkeeping (`py_ctx_pre_resize_cap`, `kv_cache_map`) — a
   shared-manager refactor would not need to change this logic, only ensure
   it is applied consistently whether draft layers live in the same pool
   group or a separate one within the unified object.

5. **[runtime]** The atomicity re-verification confirms failed native
   resizes are clean at the page-accounting level for the *existing*
   per-manager pools — this is a property of the native `KvCache`/pool
   implementation, not of how many manager objects wrap it, so it should
   carry over unchanged to a shared-manager design with separate pools
   internally.

6. **[source-only]** Q5's finding that `kv_connector_manager` is entirely
   unsupported for V2 means connector-rejection rollback symmetry is not a
   constraint the refactor needs to satisfy today — but if a future V2
   connector integration is added (to either the current two-manager or a
   future shared-manager design), this question would need to be re-opened,
   since nothing in the current source establishes what a correct
   two-manager (or shared-manager) rollback contract should look like there.

7. **[source-only, unproven at runtime]** Q8's traced mechanism — the draft
   manager's own KV state is never populated for a target-reused prefix
   range, while its chunk bookkeeping is copied/shared from the target's
   post-reuse position — is the single strongest *a priori* argument for a
   shared-manager design specifically for the **two-model / one-model-
   separate-layout topologies** (Variants A/B in
   `topology_and_prefix_reuse.md`): a shared manager with one `_KVCache` per
   request (Variant C's existing folded topology) is **already established
   by source trace to be structurally immune** to this exact gap, because
   reuse is resolved once for all layer groups sharing that one cache. If
   the refactor's goal is one manager with separate *memory pools* per
   layer group but still one `_KVCache`/reuse-resolution per request
   (closer to Variant C's model than Variants A/B's), this would mechanically
   close the Q8 gap as a side effect. **This is not confirmed by any runtime
   evidence in this session** — no model-level test succeeded in either
   demonstrating or ruling out actual output corruption from this mechanism,
   due to the SM-90 kernel-image blocker. Do not treat this as a proven bug;
   treat it as the most concrete, source-cited motivation this investigation
   found for prioritizing the refactor, pending a runtime-capable
   environment to actually confirm or rule out model-level impact.
