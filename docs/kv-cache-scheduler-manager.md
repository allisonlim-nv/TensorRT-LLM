# KVCacheV2Scheduler ↔ KVCacheManagerV2: Boundary and Failure Modes

> Read-only audit. August 2026. Covers `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`
> (1085 lines) and `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` (3751 lines), plus the
> lower-level `_KVCache`/`KVCacheManager` implementation in
> `tensorrt_llm/runtime/kv_cache_manager_v2/_core/` (see that directory's own `AGENTS.md` for its
> internal architecture).

## Overview / responsibility boundary

Three layers are in play:

- **`KVCacheV2Scheduler`** (`scheduler_v2.py:136`) — decides *which* requests run this iteration
  and, unlike the V1 scheduler, performs KV cache allocation inline as part of scheduling
  (`_try_schedule_context_full`/`_try_schedule_context_chunked` call `resize_context` directly;
  `_try_schedule_generation` calls `try_allocate_generation` directly). V2's `prepare_resources`
  is a no-op for the primary manager (`kv_cache_manager_v2.py:2474-2481`) — allocation has already
  happened in the scheduling loop, not in a later prepare-resources phase as in V1. The scheduler
  owns: per-iteration admission decisions, request ordering/sorting, eviction-victim selection,
  and the transient `BudgetTracker`.
- **`BudgetTracker`** (`scheduler_v2.py:43-133`) — a plain Python object, recreated fresh at the
  top of every `_schedule_loop` call (`scheduler_v2.py:261-265`) and discarded when
  `schedule_request` returns. It owns three independent counters: `num_tokens` (token budget),
  `num_requests` (batch-size budget), and `_claimed_peft_pages`/`_seen_peft_task_ids` (PEFT
  device-page budget). It has no relationship to GPU memory itself — it is a bookkeeping cache
  of decisions already made by `KVCacheManagerV2`/`PeftCacheManager` this iteration, used only to
  decide whether to *keep trying* more requests within the same call.
- **`KVCacheManagerV2`** (`kv_cache_manager_v2.py:742`) — owns the actual per-request KV cache
  objects (`kv_cache_map: dict[request_id, _KVCache]`), page/block allocation, prefix-tree reuse,
  suspend/resume (GPU ↔ host tier), and eviction. `_KVCache` and the underlying `KVCacheManager`
  (nanobind name clash — the *runtime* one) live in
  `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py` and `_kv_cache_manager.py`. That
  package is **pure Python, mypyc-compiled** (its own `AGENTS.md`, read in full for this audit,
  states this explicitly) — it is not a nanobind/C++ black box, so its control flow is directly
  readable, unlike most other "C++ core" components in this repo. `IndexMapper` and
  `copy_batch_block_offsets_to_device` (`kv_cache_manager_v2_utils`, imported at
  `kv_cache_manager_v2.py:36`) *are* real nanobind bindings and were not traced in this pass — see
  Open Questions.

**What the scheduler assumes stays true across calls into the manager:** that `kv_cache_map` state
for a given `request_id` — cache existence, `is_active`, `capacity`, `history_length` — is exactly
what the manager's own preceding calls (`prepare_context`, `resize_context`,
`try_allocate_generation`, `suspend_request`) left it as, with no external mutation between
scheduler calls. The scheduler never re-reads authoritative capacity/allocation state from the
manager except through the boolean return values of these methods and `is_request_active()`
(`kv_cache_manager_v2.py:2162-2165`). It does not query free-page counts, pool utilization, or
eviction feasibility ahead of time — `can_schedule()` (`scheduler_v2.py:1076-1085`) is a stub that
"Does NOT allocate" and always returns `True`, because "V2's try-and-see model lacks a free-blocks
query API" (its own TODO comment). The whole design is optimistic: try the allocation, and only on
failure fall back to eviction/suspension.

## Compact state model

**Request state** (scheduler-visible): `LlmRequestState` is a nanobind-exposed C++ enum (see
`tensorrt_llm/bindings/__init__.pyi`); the scheduler only compares `req.state_value` against
cached integer thresholds (`scheduler_v2.py:208-213`) — `CONTEXT_INIT`, `ENCODER_INIT`,
`DISAGG_GENERATION_INIT`, `GENERATION_TO_COMPLETE` — plus request-local scheduler-owned fields:
`req.py_batch_idx` (cleared on suspend, `scheduler_v2.py:1002-1003`), `req.context_chunk_size`,
`req.context_current_position`, `req.py_ctx_pre_resize_cap` (manager-owned scratch field recording
pre-resize capacity for revert, `kv_cache_manager_v2.py:2382,2411,2245-2248`).

**Python budgets** (`BudgetTracker`, per-iteration, ephemeral):
- `num_tokens` / `max_num_tokens` — token budget, incremented only in `commit()`
  (`scheduler_v2.py:91-96`), which is only called after a request is fully `SCHEDULED`.
- `num_requests` / `max_num_requests` — batch-size budget, same commit point. Disagg-gen-init
  requests are explicitly excluded (`scheduler_v2.py:346-354`).
- `_claimed_peft_pages` / `_seen_peft_task_ids` — PEFT device-page budget, committed either via
  `commit()` (folds in `commit_peft`) or standalone `commit_peft()` for disagg requests
  (`scheduler_v2.py:98-106`), and separately via `pre_claim_peft()` for
  `GENERATION_TO_COMPLETE` requests still holding a live adapter (`scheduler_v2.py:122-133,
  300-310`).

**C++/native page-cache state** (`KVCacheManagerV2` + `_KVCache`, persistent across iterations):
- `kv_cache_map[request_id] -> _KVCache` — presence/absence of an entry.
- `_KVCache.status` — `ACTIVE` / `SUSPENDED` (`_core/_kv_cache.py`; `resize()` asserts `ACTIVE`
  at entry, line 719; `suspend()` asserts `ACTIVE` at entry, line 1124; `resume()` asserts
  `SUSPENDED` at entry, line 1149).
- `_KVCache.capacity`, `_KVCache.history_length` — logical sizing, mutated by `resize()`.
- Underlying page/block ownership inside `_storage`/`_block_radix_tree`/`_eviction_controller`
  (`tensorrt_llm/runtime/kv_cache_manager_v2/_storage*.py`, `_block_radix_tree.py`) — not audited
  in this pass beyond what `resize()`/`suspend()`/`resume()` expose.
- `KVCacheManagerV2._allocated_draft_lens[request_id]` — Python-side dict recording the draft
  length used by the last `try_allocate_generation()` call, consumed by `revert_allocate_generation`
  and `extend_capacity_for_tokens` (`kv_cache_manager_v2.py:2211,2230-2232,2437`).

## Scheduler assumptions & invariants

- Context requests are always deferred to a second phase (`pending_ctx`, `scheduler_v2.py:298,
  401-423`) so generation PEFT budget settles first — the comment states this explicitly prevents
  "PEFT adapter eviction failures when gen requests hold adapters that can't be evicted
  mid-iteration" (`scheduler_v2.py:292-297`).
- Eviction victims are always drawn from indices `>= req_it` (not yet processed this loop), so the
  scheduler assumes victims were never added to `scheduled_ctx`/`scheduled_gen` and therefore never
  need a `BudgetTracker` reclaim (`scheduler_v2.py:1022-1024`) — this is true only because
  `_try_evict_for_gen` bounds its search to `range(req_it_end - 1, req_it, -1)` (`:1032`).
  A request already committed to `budget` earlier in phase 1 can never become an eviction victim.
- `is_request_active()` is used as the sole gate for "already suspended, skip as eviction victim"
  (`_is_evictable`, `scheduler_v2.py:1005-1013`) and for "worth self-evicting"
  (`scheduler_v2.py:967`). It only inspects `self.kv_cache_manager` (the *main* manager) — it does
  not consult `draft_kv_cache_manager` or `cross_kv_cache_manager` state. The scheduler assumes
  main-manager activity is sufficient to characterize whether a request still holds GPU pages
  anywhere.
- `suspend_request()` is assumed to be idempotent/safe to call unconditionally: `_suspend_request`
  (`scheduler_v2.py:988-1000`) calls it on both `kv_cache_manager` and (if present)
  `draft_kv_cache_manager`, without checking `is_request_active` first, and without checking a
  return value (`suspend_request` returns `None`, `kv_cache_manager_v2.py:2454-2458`). This holds
  because the manager-side implementation no-ops when `kv_cache is None or not kv_cache.is_active`
  — verified at `kv_cache_manager_v2.py:2456-2458` and, one level down, by the `assert self.status
  == ACTIVE` guard inside `_KVCache.suspend()` (`_core/_kv_cache.py:1124`) which the manager-level
  `is_active` check protects against triggering.
- PEFT resources are explicitly **not** released on suspend — the docstring on `_suspend_request`
  says so directly: "TODO: Also release PEFT resources (mark_request_done) for the suspended
  request... Currently only KV cache is freed; the adapter remains 'active' on device"
  (`scheduler_v2.py:991-995`). This is a known, self-documented gap, not something this audit
  discovered.
- `resize_context`/`prepare_disagg_gen_init` failure handling assumes suspending the *first chunk*
  is sufficient recovery — both call `kv_cache.suspend()` only `if req.is_first_context_chunk`
  (`kv_cache_manager_v2.py:2378-2381, 2407-2410`); a resize failure on a non-first chunk leaves the
  cache active with its pre-failure capacity, relying on the caller (scheduler) to retry or drop
  the request through some other path.

## Audited methods

### `prepare_context` (`kv_cache_manager_v2.py:2297-2358`)

Creates the `_KVCache` on first chunk (with optional block-reuse token lookup), or verifies/reactivates
an existing cache on later chunks (`_prepare_context_impl`, `:2310-2358`). Delegates suspended-cache
reactivation to `_resume_and_restore` (`:2285-2295`), which itself calls `_KVCache.resume()` and, on
success, re-binds host page-index buffers (`_restore_page_index_bufs`, `:2267-2283` — required because
`suspend()` clears those pointers, `:1127-1130`). Asserts the request is not in
`is_disagg_generation_init_state` — disagg init has a separate entry point. Returns `False` on any
failure (cache creation failure, or `resume()` refused due to `max_util_for_resume`); the scheduler
treats `False` as `SKIP`/`STOP` depending on call site (`scheduler_v2.py:543,592`), never as a
partial-success state requiring cleanup, because nothing was resized yet at this point.

### `resize_context` (`kv_cache_manager_v2.py:2360-2384`)

Grows `_KVCache.capacity` to `context_current_position + num_tokens + num_extra_kv_tokens` (never
shrinks — `capacity = max(kv_cache.capacity, target)`, `:2374`). Records the pre-resize capacity into
`req.py_ctx_pre_resize_cap` **only when capacity actually grew** (`:2382`, `None` otherwise) — this is
the field `revert_allocate_context` later reads to undo the growth. On `kv_cache.resize()` failure,
suspends the cache **only if this is the first context chunk** (`:2379-2380`); non-first-chunk
resize failures leave the cache active and un-reverted. The scheduler calls this from both
`_try_schedule_context_full` (`scheduler_v2.py:558`) and `_try_schedule_context_chunked`
(`scheduler_v2.py:660`), always as the step after `prepare_context` succeeded.

### `try_allocate_generation` (`kv_cache_manager_v2.py:2195-2213`)

Resumes the cache if suspended (returns `False` if resume is refused), records the draft length used
into `_allocated_draft_lens[request_id]` (`:2211`, unconditionally — even before knowing whether the
subsequent `resize()` will succeed), then grows capacity by `1 + draft_len`
(`_required_gen_capacity`, `:2188-2193`). Returns `False` if `kv_cache_map` has no entry for the
request, or if resume/resize fails. Note the recorded `_allocated_draft_lens` entry is written even
on ultimate resize failure — see Investigation 3 below.

### `suspend_request` (`kv_cache_manager_v2.py:2454-2458`)

Three lines: look up the cache, and if it exists and `is_active`, call `.suspend()`. No exception
handling, no return value — success/no-op are indistinguishable to the caller. This is the primitive
both `_suspend_request` (scheduler-level, wraps main + draft) and the cross-context failure path
(`scheduler_v2.py:563,665`) rely on.

### `prepare_disagg_gen_init` (`kv_cache_manager_v2.py:2385-2412`)

Combines `_prepare_context_impl` (same cache-creation/reuse path as `prepare_context`) with an
immediate resize to the **full prompt length** (`req.prompt_len + draft_len + num_extra_kv_tokens`,
`:2402`) and sets `history_length` to `req.prompt_len` in the same `resize()` call (`:2406` — passes
`history_length` positionally, unlike `resize_context` which never touches history_length). Same
first-chunk-only suspend-on-failure pattern as `resize_context` (`:2408-2409`). Called from
`_try_schedule_disagg_gen_init` (`scheduler_v2.py:488-505`), which treats `False` as `SKIP` (not
`STOP`) — a disagg request that can't yet be admitted is retried next iteration rather than blocking
the whole scheduling pass, per the comment "Capacity is gated by IndexMapper slot availability...
the request is skipped and retried next iteration" (`scheduler_v2.py:328-333`).

## Investigation 1: generation allocation failure → eviction → suspension

Call chain: `_try_schedule_generation` (`:927-975`) → on `try_allocate_generation` failure →
`_try_evict_for_gen` (`:1015-1052`) → per victim, `_suspend_request` (`:988-1000`) → on victim
exhaustion, self-eviction also calls `_suspend_request` on the requesting request itself
(`:967-973`).

`_suspend_request` mutates two independent managers sequentially with no atomicity between them:

```python
def _suspend_request(self, req: LlmRequest) -> None:
    self._clear_request_runtime_state(req)
    self.kv_cache_manager.suspend_request(req)
    if self.draft_kv_cache_manager is not None:
        self.draft_kv_cache_manager.suspend_request(req)
```

| Scheduler assumption | Evidence | Partial-failure / divergence scenario | Current recovery/rollback | Verdict + severity | Remaining uncertainty |
|---|---|---|---|---|---|
| Suspending main and draft managers for one request is effectively atomic — both end up suspended together. | `scheduler_v2.py:988-1000`; `suspend_request` no-op guard `kv_cache_manager_v2.py:2456-2458`; `_KVCache.suspend()` asserts `status == ACTIVE` at entry (`_core/_kv_cache.py:1124`). | If `self.kv_cache_manager.suspend_request(req)` raises (e.g. an `assert` inside `_KVCache.suspend()` firing on a state invariant not otherwise checked, such as `assert self._finish_event is None`, `:1126`), the exception propagates out of `_suspend_request` before the draft manager's `suspend_request` call is ever reached. Main cache may be left suspended (if the assert fires after `_status = SUSPENDED` is set, unlikely given it's the last line) or fully active (if it fires earlier, more likely) while the draft cache is untouched either way — no divergence in the common failure shape, but *no exception handler exists anywhere in this call chain* to confirm which. | None. No `try`/`except` around either `suspend_request` call in `_suspend_request` or its callers up to `_schedule_loop`. An uncaught exception here propagates out of `schedule_request()` into whatever calls the scheduler (`py_executor.py`), which was not traced in this pass. | **Risk, not confirmed bug.** Severity: low likelihood (requires hitting an internal `_KVCache` assertion under normal call patterns, which the manager-level `is_active` guard is specifically designed to prevent) but high blast radius if it does happen — an uncaught exception mid-eviction would abort the whole scheduling pass, potentially with some victims already suspended and others not, and the loop never returns a `SchedulerOutput`. | Whether any assertion inside `_KVCache.suspend()`/`resume()`/`resize()` is reachable given the exact call sequences the scheduler uses was not exhaustively verified — would require tracing every `_storage`/`_block_radix_tree` invariant these methods touch, which is out of scope for this pass (see that subsystem's own `AGENTS.md`). |
| Main-manager `is_active` fully characterizes whether a request still holds pages anywhere. | `_is_evictable` (`scheduler_v2.py:1005-1013`) and self-eviction gate (`scheduler_v2.py:967`) query only `self.kv_cache_manager.is_request_active(...)`. | **Confirmed reachable, but only inside a bounded, non-scheduling window — not a standing condition the scheduler ever has to reason about.** `PyExecutor._maybe_rebalance_kv_pools` (`py_executor.py:4413-4444`) suspends every active request's **main-only** cache to run the V2 KV-pool auto-tuner (`mgr.impl.adjust()`), then resumes them — `mgr = self.kv_cache_manager` (`:4425`), and `PyExecutor` holds no `draft_kv_cache_manager` reference at all (grepped `self.draft_kv_cache_manager` and `self.kv_cache_manager =` across `py_executor.py`: only the main-manager assignments at `:418,642` exist). For the duration of this call, every active request has `main.is_request_active(req) == False` while its draft cache (if any) is untouched and stays `is_request_active(req) == True`. Crucially, `_maybe_rebalance_kv_pools` runs as its own step in the `PyExecutor` loop, not inside `KVCacheV2Scheduler.schedule_request()` — the scheduler's `_is_evictable`/self-eviction checks never execute concurrently with this window, so the divergence never actually reaches the code that assumes main-only activity is sufficient. | `resume_request` is called on the same `paused` list right after `adjust()` (`:4443-4444`); a resume failure is explicitly tolerated and left suspended by design — the docstring states "Resume failures stay suspended; the scheduler reactivates them through prepare_context / try_allocate_generation on the next iteration, the same path it uses today after eviction" (`:4419-4423`). No corresponding action is taken on the draft manager at all — it is not paused, not queried, and not adjusted. | **Intentional, confirmed — scoped to pool tuning, not a scheduler-wide guarantee.** `_maybe_rebalance_kv_pools` is a self-contained, temporary state during main-pool auto-tuning: it deliberately rebalances only the main manager's GPU/host pool ratio, treats the draft manager's pool as out of scope, and closes the window (via `resume_request`) before control returns to the executor loop that calls the scheduler next. This does **not** mean "main and draft managers can diverge at any time the scheduler might observe" — the scheduler's assumption that main-manager activity is sufficient still holds for every call it actually makes, because `schedule_request()` and `_maybe_rebalance_kv_pools()` never interleave. Severity: none for the scheduler's own invariant; whether the auto-tuner *should* also rebalance/consider the draft pool is a separate product-design question, out of scope for this document. | None remaining on reachability — confirmed via `py_executor.py:4413-4444`. Still open: whether draft-pool sizing has its own, separate rebalancing mechanism elsewhere (not found in this pass; the draft manager's `need_adjustment`/`adjust()` were not checked), and whether `_maybe_rebalance_kv_pools` and `schedule_request()` share any lock/ordering guarantee that makes their non-interleaving structural rather than incidental (not traced in this pass). |
| Eviction is worth attempting even if it ultimately fails, and any victims suspended along the way are legitimately freed (not wasted work needing undo). | Docstring: "Returns (new_req_it_end, success)... new_req_it_end is always updated to reflect evicted victims (even on failure)" (`scheduler_v2.py:1026-1028`). | If `_try_evict_for_gen` suspends N victims and still fails to fit the requesting request, those N victims stay suspended — this is explicitly intentional (comment says so), not a bug: the requesting request itself is then self-evicted too (`scheduler_v2.py:963-973`) and the whole set appears in `evicted`/`paused_requests`. | By design: victims are never un-suspended. Confirmed by `test_gen_alloc_fails_evict_insufficient`, `test_multiple_evictions_needed` in `tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py`. | **Intentional behavior.** No rollback needed because a paused/suspended request is a valid scheduler-visible outcome (`SchedulerOutput.paused_requests`), not an error. | None — this is well-tested and documented. |

Test coverage: `tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py` has extensive coverage
of the eviction/self-eviction paths (`test_gen_alloc_fails_evict_succeeds`,
`test_gen_alloc_fails_evict_insufficient`, `test_multiple_evictions_needed`,
`test_suspended_request_not_evictable`, `test_evict_gen_from_tail`, `test_multiple_evictions_order`,
`test_req_it_end_shrinks_after_eviction`, `test_self_eviction_on_alloc_fail`, etc. — line ~458-825 and
~1977-2053). **No test in this file constructs a scheduler with `draft_kv_cache_manager` set**
(grepped for `draft_kv_cache_manager`/`draft_mgr` — zero hits), so the main/draft suspend-divergence
question above is entirely untested at the scheduler level.

## Investigation 2: `resize_context` / `_try_schedule_context_chunked` mutation-then-failure

`_try_schedule_context_chunked` (`scheduler_v2.py:568-670`) mutation points, in order:

1. `self.kv_cache_manager.prepare_context(req)` (`:590`) — may create the `_KVCache` and mutate
   `req.context_current_position`/`req.set_prepopulated_prompt_len(...)` as a block-reuse side
   effect (`kv_cache_manager_v2.py:2340-2343`). Failure here → `SKIP`, no further mutation, no
   rollback needed (nothing capacity-related happened yet).
2. `req.context_chunk_size = ...` (`:647`) — Python-only field write, no manager state.
3. `self.kv_cache_manager.resize_context(req, resize_tokens)` (`:660`) — grows `_KVCache.capacity`
   and sets `req.py_ctx_pre_resize_cap`. **This is the mutation of interest.**
4. `self._try_schedule_cross_context(req)` (`:663`) — may create/resize the **cross**-attention
   `_KVCache` in a *separate* manager (`cross_kv_cache_manager`).

If step 4 fails after step 3 succeeded:

```python
cross_action = self._try_schedule_cross_context(req)
if cross_action is not ScheduleAction.SCHEDULED:
    self._suspend_request(req)
    return cross_action, 0, False
```

| Scheduler assumption | Evidence | Partial-failure / divergence scenario | Current recovery/rollback | Verdict + severity | Remaining uncertainty |
|---|---|---|---|---|---|
| Suspending the (self-)`_KVCache` after a resize is a sufficient undo for a same-iteration cross-context failure. | `scheduler_v2.py:563-564,665-666`; contrast with the *actual* revert primitive, `revert_allocate_context` (`kv_cache_manager_v2.py:2243-2265`), which restores `capacity` back to `py_ctx_pre_resize_cap` (not just suspend) and is called only from `py_executor.py:3373-3376,3465-3487` for a **different** scenario (deferred disagg-transfer-admission requests, `_revert_deferred_disagg_gen_init_alloc`). Full trace of `_KVCache.resize()` (`_core/_kv_cache.py:718-911`, read start to finish) and `suspend()` (`:1121-1145`, read start to finish). | The chunked/full-context resize path never calls `revert_allocate_context`. Capacity growth (`kv_cache.capacity` now at `target`, `py_ctx_pre_resize_cap` left set to the old value) survives the suspend. Next time this request is scheduled, `resize_context` runs again with `capacity = max(kv_cache.capacity, target)` (`:2374`) — the elevated capacity becomes the new floor, never shrunk. | `suspend()` never touches `self._capacity` — verified by reading the full method body (`_core/_kv_cache.py:1121-1145`): it releases active GPU pages/scratch slots and flips `_status` to `SUSPENDED`, nothing else. `resize()`'s growth path only mutates `self._capacity` at its very last line, after every allocation step has already succeeded (`:907`, guarded by `assert NDEBUG or self._check_sanity()` at `:910`); its one failure branch, `OutOfPagesError` from `storage.new_gpu_slots(...)` (`:812-819`), is caught and explicitly rolled back before returning `False` — `self._recover_excess_scratch_slots(excess_scratch_slots)` and `self._lock_held_blocks(backup_holders)` (`:820-821`) undo the `_unlock_stale_blocks`/scratch-slot bookkeeping done earlier in the same call, and `self._capacity` is left completely untouched (still the pre-call value) because line 907 is never reached. The shrink branch (`new_num_blocks < old_num_blocks`, `:760-774`) has no failure path — it runs unconditionally before any allocation and cannot raise `OutOfPagesError`. So `resize()` is atomic with respect to `capacity`: it either fully applies (all mutations plus the capacity bump) or fully rolls back the mutations it made and returns `False` with capacity unchanged. | **Intentional.** `resize_context`'s own framing — `capacity = max(kv_cache.capacity, target)` (`kv_cache_manager_v2.py:2374`) — explicitly treats current capacity as a floor to `max()` against, not a value ever meant to shrink outside `resize()`'s own (unused-by-this-path) shrink branch. `_KVCache.suspend()`'s docstring states the design directly: "Suspend, allow the KV cache manager to evict buffers from GPU, but don't drop them. suspend+resume allows us to implement dynamic batch size" (`_core/_kv_cache.py:1121-1122`) — capacity is a logical reservation independent of physical GPU residency, and preserving it across suspend is the entire point: pages are freed (no leak), but the reservation survives so a later `resize()` to the same target hits the `_shortcut_set_capacity` fast path (`:747-753`) instead of re-deriving/re-allocating from scratch. There is no logical inconsistency: nothing downstream (`try_allocate_generation`, `resize_context`, budget accounting) reads `capacity` as "currently resident," only as "currently reserved," and both call sites of this pattern (`resize_context:2378-2381`, `prepare_disagg_gen_init:2407-2410`) rely on exactly this contract. Severity: none — this is working as designed, not a risk. | None remaining — `resize()` and `suspend()` were both read in full for this pass. |
| `resize_context`'s `py_ctx_pre_resize_cap` bookkeeping is the single source of truth for "how much did this iteration's resize grow capacity by." | `kv_cache_manager_v2.py:2382,2411`; consumed only by `revert_allocate_context` (`:2243-2263`). | The chunked-context loop can call `resize_context` more than once for the same request across chunks (once per chunk, `scheduler_v2.py:660` inside `_try_schedule_context_chunked`, itself called once per `schedule_request` invocation per pending request). Each call overwrites `py_ctx_pre_resize_cap` with *that call's* pre-resize capacity (or clears it to `None` if this call didn't grow capacity). If a chunk resize succeeds (sets the field) and a **later, unrelated** revert call fires (e.g. `_revert_ctx_alloc` from `py_executor.py` for a disagg-admission-deferred reason unrelated to this specific resize), it would revert to the *most recent* chunk's pre-resize capacity, not the cumulative pre-chunking baseline — this is very likely intentional (each `resize_context` call is its own logical grow-step) but was not confirmed against a concrete multi-chunk + revert test. | N/A — no test in `test_kv_cache_v2_scheduler.py` exercises `py_ctx_pre_resize_cap` combined with multi-chunk context requests. | **Intentional behavior, low confidence.** The field name and single-slot storage strongly suggest "last grow amount," consistent with `revert_allocate_context`'s docstring "Undo the capacity growth from *this iteration's* context resize" (`kv_cache_manager_v2.py:2243-2244`, emphasis on "this iteration"). | Not independently verified against a running multi-chunk scenario; inferred from reading, not execution. |

## Investigation 3: `BudgetTracker` vs C++/native manager state drift

`BudgetTracker` is transient (one instance per `_schedule_loop` call, discarded on return,
`scheduler_v2.py:261-265`), so it cannot itself accumulate drift *across* scheduler invocations —
but within a single invocation, and across the boundary into the next invocation via
manager-persisted state, several points let the two diverge:

| Scheduler assumption | Evidence | Partial-failure / divergence scenario | Current recovery/rollback | Verdict + severity | Remaining uncertainty |
|---|---|---|---|---|---|
| `budget.commit(...)` is only ever called once a request's manager-side allocation has fully succeeded, so `budget.num_tokens`/`num_requests` always track real committed KV allocation. | Every `commit()` call site follows a successful `resize_context`/`try_allocate_generation`/`prepare_disagg_gen_init` return (`scheduler_v2.py:397,414,423,352`). | Holds for `num_tokens`/`num_requests`/`_claimed_peft_pages` *within this loop*. But the manager-side allocation this budget reflects can later be **reverted** by `py_executor.py` (`_revert_gen_alloc`/`_revert_ctx_alloc`, called after `schedule_request()` returns, e.g. on attention-DP `can_queue=False`, `py_executor.py:3321-3337,3373-3376`) — at that point the `BudgetTracker` instance that recorded the commit is already gone (new one created next iteration), so there's no live Python counter left to "un-drift." This is *not* a bug given the ephemeral design, but it does mean the budget accounting for one scheduling pass can end up describing allocations that are later fully undone, with nothing in this file recording that fact. | None needed by design — budget is discarded per-iteration; the manager alone carries state across iterations. | **Intentional design, verified consistent.** No drift within `BudgetTracker`'s own lifetime. | None. |
| `try_allocate_generation`'s `_allocated_draft_lens[request_id]` write and its actual `kv_cache.resize()` outcome are consistent with each other. | `kv_cache_manager_v2.py:2210-2212`: `self._allocated_draft_lens[req.py_request_id] = draft_len` is written **unconditionally**, then `return kv_cache.resize(...)` — the dict write happens whether or not the resize that follows succeeds. | If `resize()` fails (returns `False`), `try_allocate_generation` returns `False` up to the scheduler, which — per Investigation 1 — self-evicts or evicts a victim and retries. But `_allocated_draft_lens[request_id]` now holds a draft length for an allocation that never happened. If this same request is later scheduled again (e.g. next iteration, or after `_try_evict_for_gen` frees room and the *same* call retries — note: the retry in `_try_evict_for_gen`, `scheduler_v2.py:1049`, calls `try_allocate_generation` again, which overwrites `_allocated_draft_lens[request_id]` again before its own resize — no accumulation within one scheduling pass), the stale value would only matter if something reads `_allocated_draft_lens` **between** a failed `try_allocate_generation` and a subsequent read (`revert_allocate_generation` or `extend_capacity_for_tokens`) without an intervening successful `try_allocate_generation` call for the same request. | `revert_allocate_generation` (`:2214-2241`) pops the value and computes `reverted_cap = kv_cache.capacity - 1 - draft_len`, guarding `if reverted_cap < 0: return` — so a stale/wrong value would silently produce a wrong (or silently-skipped) capacity revert rather than crash. `extend_capacity_for_tokens` (`:2426-2452`) similarly pops it and computes a delta to `resize()`, raising `ValueError` only if that resize itself fails — a wrong stale draft length would produce a wrong delta without any error signal. | **Risk — plausible but not confirmed reachable.** Needs a concrete trace of whether `revert_allocate_generation`/`extend_capacity_for_tokens` can be called for a request whose *most recent* `try_allocate_generation` call failed (rather than succeeded) — this depends on `py_executor.py` call ordering relative to `SchedulerOutput.generation_requests` membership (a request whose allocation failed should not appear in `generation_requests` in the first place, per `scheduler_v2.py:960-961,975`, which would make this unreachable in practice). Severity if reachable: silent wrong capacity accounting, not a crash. | Whether `py_executor.py` ever calls `revert_allocate_generation`/`extend_capacity_for_tokens` for a request outside `scheduled_batch.generation_requests` was not checked — this is the key fact that would resolve the question. |
| PEFT page accounting (`_claimed_peft_pages`) always matches what `PeftCacheManager` actually has resident on device. | `BudgetTracker.commit_peft`/`pre_claim_peft` (`scheduler_v2.py:98-133`) increment `_claimed_peft_pages` purely in Python, based on `peft_cache_manager.determine_num_pages(req)` (`:117,131`) — a **query**, not an allocation call. | `BudgetTracker` never calls any PEFT allocation/commit method on `self.peft_cache_manager` — it only asks "how many pages would this need" and locally tracks a running total to gate admission. The actual PEFT page allocation presumably happens elsewhere (not in this file — not traced in this pass). If that real allocation can fail *after* the scheduler already committed the request into `scheduled_ctx`/`scheduled_gen`/`disagg_candidates` based on this budget's optimistic count, the KV-side commit (`budget.commit`) and the real PEFT allocation outcome could diverge — but this scheduler file has no visibility into or control over that downstream step. | Unknown — the actual PEFT allocation call site is outside `scheduler_v2.py` and `kv_cache_manager_v2.py`, not traced in this pass. | **Unconfirmed — flagged as unknown**, not assessed as bug/risk/intentional since the relevant code wasn't located in this audit. | Where the real PEFT page commit (as opposed to this dry-run `determine_num_pages` query) happens, and whether it can fail after `BudgetTracker` already optimistically counted the pages as claimed, was not investigated — would need to locate `peft_cache_manager`'s allocation entry point (likely in `_torch/peft/` per `docs/codebase-map.md`) and trace its call site in `py_executor.py`'s resource-preparation phase. |

## Open questions / unknowns

- **Resolved: `_KVCache.resize()` internal partial-mutation behavior on growth failure.** Read in
  full (`_core/_kv_cache.py:718-911`). The only failure branch is `OutOfPagesError` from
  `storage.new_gpu_slots(...)` (`:812-819`); it is caught and explicitly rolled back
  (`_recover_excess_scratch_slots`, `_lock_held_blocks`, `:820-821`) before returning `False`, and
  `self._capacity` is only ever mutated at the method's last line (`:907`), which a failure never
  reaches. `resize()` is atomic with respect to capacity — see the revised Investigation 2 verdict
  above (now **Intentional**).
- **Resolved: draft-manager independent suspend/resume paths.** Repo-wide search (all `.py` files
  under `tensorrt_llm/_torch/pyexecutor/`, `_torch/speculative/`) for
  `.suspend_request(`/`.resume_request(`/`draft_kv_cache_manager` found exactly one real independent
  path outside `KVCacheV2Scheduler._suspend_request`: `PyExecutor._maybe_rebalance_kv_pools`
  (`py_executor.py:4413-4444`), which suspends/resumes only `self.kv_cache_manager` (main) for KV
  pool auto-tuning — `PyExecutor` has no `draft_kv_cache_manager` reference at all. This is a
  confirmed, structural main-only suspend window, but it is **temporary and scoped to the
  auto-tuner's own step in the executor loop** — it never overlaps with `schedule_request()`, so it
  does not weaken the scheduler's own main-only-activity assumption during scheduling. See the
  revised Investigation 1 verdict above (**Intentional, confirmed — scoped to pool tuning**). All
  other `draft_kv_cache_manager` call sites found in this search (`kv_cache_manager_v2.py:3064-3181`,
  warmup/dummy-request creation) mirror main and draft resize calls in lockstep with shared
  failure/rollback (`release_resources`), and are not scheduling-time suspend/resume paths.
- **Real PEFT page allocation/commit call site** — not located in this pass; needed to assess
  whether `BudgetTracker`'s PEFT accounting can diverge from actual device state (Investigation 3,
  third row).
- **`IndexMapper` / `copy_batch_block_offsets_to_device`** (`kv_cache_manager_v2_utils`, nanobind
  binding, `kv_cache_manager_v2.py:36-39`) — these are genuine C++ bindings and were not traced at
  all. Any atomicity question involving index-slot allocation (as opposed to `_KVCache` capacity)
  would need the C++ source under `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` (or
  wherever `kv_cache_manager_v2_utils` is implemented — not confirmed in this pass) to resolve.
- **Whether `revert_allocate_generation`/`extend_capacity_for_tokens` can ever be called for a
  request whose most recent `try_allocate_generation` failed** — would resolve the severity of the
  `_allocated_draft_lens` staleness question in Investigation 3.
- No exception-handling audit was performed for `kv_cache.resize()`/`suspend()`/`resume()`
  themselves raising (as opposed to returning `False`) under conditions this pass didn't enumerate
  — the assumption throughout is "these return bool and don't raise under scheduler-reachable
  inputs," which held for every call site actually read, but wasn't proven exhaustively.
