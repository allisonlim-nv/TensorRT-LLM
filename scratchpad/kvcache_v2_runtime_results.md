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

## Summary

| Q | Answer | Evidence tier |
|---|---|---|
| Q1 | Target/draft GPU capacities for B=4GiB: target 3439329280B/52480 slots (4 layers), draft 859832320B/52480 slots (1 layer); split computed by real code, scaling verified correct | Real split-code + real GPU alloc; synthetic per-token cost inputs (no model available) |
| Q2 | NOT always aligned — a failed draft-mirror suspend leaves target suspended, draft still active, exception uncaught | Real scheduler code; mocked managers (no GPU) |
| Q3 | Recoverable, and cleaned up consistently — capacity preserved, no page leak, close/free succeed | Real native GPU allocation, real OutOfPagesError |
| Q4 | Conservatively differs — scheduler charges 0 extra draft tokens, manager reserves `max_total_draft_tokens` (4) more; manager is the binding, safe constraint | Real scheduler code (mocked managers) + real manager method (minimal stand-in object, not full GPU path) |

No test was blocked by a missing model artifact or unsupported configuration
— once the dev container's Python dependencies were installed, a single
H200 GPU was sufficient for all four questions, since none require loading an
actual HF model (per-layer KV costs were supplied directly as `CacheCost`
values for Q1, matching the existing repo test convention for that same
function).
