# Executor Iteration Ordering: `scheduler.schedule_request` → `prepare_resources`/`update_resources` → revert → forward

commit 4716843cee6e7a6c08bf4d8be29fae25321a9344, branch feat/native-kv-events-clean, date 2026-08-31.

Scope: read-only trace of `tensorrt_llm/_torch/pyexecutor/py_executor.py` (8688 lines) and
`tensorrt_llm/_torch/pyexecutor/resource_manager.py` (3187 lines) on this commit. Three main
iteration loops exist: `_executor_loop` (non-overlap, non-PP), `_executor_loop_overlap`
(overlap scheduler, non-PP), `_executor_loop_pp` (pipeline-parallel). All three funnel
scheduling through the shared helper `_prepare_and_schedule_batch` (`py_executor.py:3733`),
except PP which inlines an equivalent sequence via `_pp_schedule_and_propagate` /
`_pp_retry_until_can_schedule`.

---

## 1. Main iteration loop(s) and call-site line numbers on this commit

Prior-session line numbers have drifted. Re-verified locations:

- `_executor_loop_pp`: `py_executor.py:2587`
  - PP-follower local re-run of `scheduler.schedule_request`: `py_executor.py:2644` (was ~2644, unchanged)
  - `can_queue` check: `py_executor.py:2682`; revert: `py_executor.py:2690`
  - `resource_manager.prepare_resources(scheduled_batch)`: `py_executor.py:2715`
  - `kv_cache_manager.update_context_resources(...)`: `py_executor.py:3256`
  - `resource_manager.update_resources(...)`: `py_executor.py:3275`
- `_executor_loop` (non-overlap): `py_executor.py:4176`
  - `can_queue, _ = self._can_queue(scheduled_batch)`: `py_executor.py:4248`
  - `resource_manager.prepare_resources(scheduled_batch)`: `py_executor.py:4261`
  - second `can_queue` re-check (kv-connector only): `py_executor.py:4271`
  - `_revert_gen_alloc(scheduled_batch)`: `py_executor.py:4274`
  - forward pass `self._forward_step(scheduled_batch)`: `py_executor.py:4332`
  - `kv_cache_manager.update_context_resources(...)`: `py_executor.py:4381`
  - `resource_manager.update_resources(...)`: `py_executor.py:4399`
- `_executor_loop_overlap`: `py_executor.py:5006`
  - `can_queue` check: `py_executor.py:5067`
  - `resource_manager.prepare_resources(scheduled_batch)`: `py_executor.py:5094`
  - second `can_queue` re-check (kv-connector only): `py_executor.py:5104`
  - `_revert_gen_alloc(scheduled_batch)`: `py_executor.py:5108`
  - forward pass `self._forward_step(...)`: `py_executor.py:5185`
  - `kv_cache_manager.update_context_resources(...)`: `py_executor.py:5263`
  - `resource_manager.update_resources(...)` (on the **previous** iteration's batch): `py_executor.py:5424`, inside `_process_previous_batch` (`py_executor.py:5414-5428`)
- `_schedule()` (wraps `scheduler.schedule_request`): `py_executor.py:6271`, call at `py_executor.py:6276`
- `_prepare_disagg_gen_init`: `py_executor.py:7207-7234` (was cited as 7206-7234; off by one line, otherwise matches)

**Confidence: Verified current behavior** for every line cited above (directly read).

---

## 2. Ordering of `prepare_resources`/`update_resources` relative to `schedule_request`, and registry composition

### 2a. Is `prepare_resources` called once per iteration, immediately after `schedule_request`, over the same `ScheduledRequests`?

**Mostly yes, with one important caveat.** `_schedule()` (`py_executor.py:6271-6333`) calls
`scheduler.schedule_request` once (`py_executor.py:6276`) and builds a **new**
`ScheduledRequests` object (`py_executor.py:6324-6332`) populated from `scheduler_output`'s
fields. That object — not `scheduler_output` itself — is what flows forward as
`scheduled_batch` into `prepare_resources`.

Caveat: between `schedule_request` returning and `_schedule()` returning, several filters can
**drop entries from `context_requests`** without reverting their already-grown V2 KV capacity:
- `_balance_adp_requests` (`py_executor.py:6291`, defined at `5951`)
- `_waiting_requests` (`py_executor.py:6305`, defined at `6064-6082`) — batch-waiting: if the
  iteration's token count is below a ratio threshold, it returns `[]` for `context_requests`
  entirely, deferring **all** context requests to a later iteration while **not** calling
  `revert_allocate_context`. The docstring/comment at `py_executor.py:6303-6304` states this
  explicitly: *"With KV cache manager V2, scheduling has already grown context request KV cache
  capacity. Requests dropped for batch waiting still occupy KV cache and may reduce the batch
  size available for generation requests."*
- `_cap_context_by_total_kv_len` (`py_executor.py:6321`, defined at `6249-6268`) — trims the
  tail of `context_requests` past an fp8-context-MLA KV-length cap; docstring at
  `py_executor.py:6253-6254` states deferred requests "stay active and retry next iteration,
  mirroring `_waiting_requests`" — again no revert call.

So the `scheduled_batch` handed to `prepare_resources` can be a **strict subset** of what the
V2 scheduler granted capacity for inside `schedule_request`. The dropped requests' capacity
growth is *not* rolled back and is *not* mirrored onto `draft_kv_cache_manager` this iteration
(mirroring only touches requests actually present in `scheduled_batch`, per
`_prepare_draft_resources`). This is a genuine "stale relative to scheduler's true decision"
window, but it is a **subtraction only** (fewer requests seen by `prepare_resources` than the
scheduler internally admitted), not a case of a *different* request set or reordering.

Distinct disagg-only path: `_apply_disagg_transfer_admission` (`py_executor.py:3541-3571`) can
similarly defer `fitting_disagg_gen_init_requests` under a transfer-budget controller, but this
path **does** call `_revert_ctx_alloc` on the deferred subset (`py_executor.py:3566-3568`,
`3587-3588`) before `_prepare_disagg_gen_init` ever runs on the admitted remainder
(`py_executor.py:3844`, function at `7207`) — so disagg-admission deferrals are properly
reverted; the batch-waiting/kv-len-cap deferrals inside `_schedule()` (2a above) are not.

**Confidence: Verified current behavior.**

### 2b. Is `DRAFT_KV_CACHE_MANAGER` in the registry unconditionally, whenever configured?

`ResourceManager.prepare_resources`/`update_resources`/`free_resources`
(`resource_manager.py:3001-3037`) iterate `self.resource_managers.items()` (an `OrderedDict`)
unconditionally and call the hook if `hasattr(resource_manager, "prepare_resources")` /
`"update_resources"` — **no per-type filtering, no conditional skip by `ResourceManagerType`.**
There is a `reorder_pipeline` method (`resource_manager.py:3039-3043`) that can reorder the
dict via `move_to_end`, but `grep` across `tensorrt_llm/_torch/pyexecutor/*.py` finds **no
caller of `reorder_pipeline` on this commit** — it is dead/unused, so registration order is the
effective iteration order.

Registration order: `_util.py:2104-2108` inserts
`ResourceManagerType.KV_CACHE_MANAGER` (target) **before**
`ResourceManagerType.DRAFT_KV_CACHE_MANAGER` (draft, possibly `None` if not configured) before
`ResourceManagerType.CROSS_KV_CACHE_MANAGER`. So within a single `prepare_resources`/
`update_resources` call, the target manager's (no-op) call always executes before the draft
manager's (real, mirroring) call, in the same synchronous Python call stack — never
interleaved with anything else.

If `draft_kv_cache_manager` is `None` (no speculative decoding, or spec decoding without a
separate draft cache), the dict entry exists but the value is `None`;
`hasattr(None, "prepare_resources")` is `False`, so the loop silently skips it
(`resource_manager.py:3004`, `3025`). So "unconditionally included whenever one is configured"
is correct; when not configured, the dict slot is a no-op via the `hasattr` guard, not via
special-casing the `ResourceManagerType`.

**Confidence: Verified current behavior.**

---

## 3. Rollback ordering — does reverting the target manager's capacity also revert the draft manager's mirrored growth?

**Verified: no, and there is a real (not merely theoretical) window where this matters.**

Every `revert_allocate_generation`/`revert_allocate_context` call site in `py_executor.py`
targets `self.kv_cache_manager` only — confirmed by exhaustive grep, 4 call sites total:
- `py_executor.py:3438` — inside `_revert_gen_alloc` (`3418-3438`)
- `py_executor.py:3477` — inside `_revert_ctx_alloc` (`3474-3477`)
- `py_executor.py:4217` — `_check_benchmark_disagg_gate` retry path in `_executor_loop`
- `py_executor.py:5048` — same retry path in `_executor_loop_overlap`

None of these ever call `self.draft_kv_cache_manager.revert_allocate_*`. This matches and
reconfirms the prior audit's finding on this commit.

**The actual sequencing race in `_executor_loop` and `_executor_loop_overlap`:**

In `_executor_loop` (`py_executor.py:4248-4275`) and `_executor_loop_overlap`
(`py_executor.py:5067-5109`), the sequence is:

```
1. can_queue, _ = self._can_queue(scheduled_batch)          # first vote
2. if can_queue:
       ...
       self.resource_manager.prepare_resources(scheduled_batch)   # <-- draft mirroring happens HERE
3. if self.kv_connector_manager:
       self.kv_connector_manager.handle_metadata()
4. if can_queue:
       self._kv_connector_start_batch(scheduled_batch)
5. if self.kv_connector_manager:                              # SECOND, independent vote
       can_queue, _ = self._can_queue(scheduled_batch)
6. if not can_queue:
       self._revert_gen_alloc(scheduled_batch)                # <-- reverts ONLY self.kv_cache_manager
```
(`_executor_loop`: steps at lines 4248, 4261, 4263, 4266, 4270-4271, 4273-4274.
`_executor_loop_overlap`: steps at lines 5067, 5094, 5096, 5099, 5103-5104, 5107-5108. Identical
structure, same comment text "if using a kv connector, we need to call can_queue again since
scheduled_batch might have changed" at `4269`/`5102`.)

So: when `self.kv_connector_manager` is configured (a real, supported feature —
`self.kv_connector_manager` is assigned once, unconditionally from a constructor argument, at
`py_executor.py:985`, i.e. it is present whenever a KV connector is configured, not a
test-only path), `prepare_resources` — including the draft manager's real mirroring work in
`_prepare_draft_resources` (`kv_cache_manager_v2.py:2779`) — has **already executed and mutated
draft-manager state** by the time the *second* `can_queue` re-check runs. If that second check
flips `can_queue` from `True` to `False` (e.g., the KV connector changed `scheduled_batch`'s
composition in a way that empties it on this or another attention-DP rank), `_revert_gen_alloc`
fires and reverts only `self.kv_cache_manager`'s generation-capacity growth for
`scheduled_batch.generation_requests`. The draft manager's already-mirrored capacity growth for
the exact same request set is **never rolled back** in this code path.

This is a genuine, source-confirmed asymmetry: the scheduler-level contract ("scheduler only
ever mirrors `suspend_request`/`free_resources`, never calls capacity methods on
`draft_kv_cache_manager`") holds, but the *executor*-level `prepare_resources` dispatch does
call capacity-growing code on the draft manager (via the generic registry), and the executor's
only revert hook is blind to that manager. Whether this causes actual leakage/overflow depends
on whether draft-manager capacity growth via `_prepare_draft_resources` is idempotent/self-
correcting on a subsequent iteration when the same (still-active) request is rescheduled with a
now-larger already-mirrored draft capacity — that determination requires reading
`_prepare_draft_resources` and the V2 manager's grow/shrink semantics in detail, which is
**out of scope for this pass** (this pass is executor-ordering only, not manager-internals).

`_executor_loop_pp` (`py_executor.py:2587`) does **not** have this race: its single `can_queue`
check (`py_executor.py:2682`) and matching `_revert_gen_alloc` (`py_executor.py:2690`) both
occur strictly **before** `resource_manager.prepare_resources` is called
(`py_executor.py:2715`). No second `can_queue` re-check exists later in the PP loop body
(confirmed by grep — `can_queue` appears at `2682`, `2691`, `2822`, `2825`, `2855`, `2927`, none
of which reassign `can_queue` from a fresh `_can_queue()` call after `prepare_resources`). So in
PP mode this specific draft-manager-not-reverted scenario is structurally unreachable: revert
always happens before any capacity mirroring could occur.

**Confidence:**
- Non-overlap/overlap loops with a KV connector configured: **Verified current behavior** — the
  code path exists exactly as described and executes in this order whenever
  `self.kv_connector_manager` is truthy and the second `_can_queue` call returns `False`.
  Whether this is benign (self-correcting next iteration) or a real resource leak requires
  reading `KVCacheManagerV2._prepare_draft_resources` and its shrink/grow invariants —
  **Source-inconclusive at the executor-ordering level; requires a targeted read of
  `kv_cache_manager_v2.py`'s draft-mirroring internals (out of this pass's scope) or a
  fault-injection test** that forces the kv-connector's second `can_queue` vote to flip after
  `prepare_resources` has run, then inspects `draft_kv_cache_manager`'s page/capacity counters
  before and after `_revert_gen_alloc`.
- PP loop: **Verified current behavior / unreachable** — no second `can_queue` re-check exists
  after `prepare_resources` in `_executor_loop_pp`, so this specific race cannot occur there.

---

## 4. Forward-pass ordering relative to both `prepare_resources` calls

**Verified current behavior:** in all three loops, the forward pass is placed strictly after the
single `resource_manager.prepare_resources(scheduled_batch)` call for that iteration's
`scheduled_batch`, and only runs `if can_queue:` (the final, post-second-check value in the
kv-connector case).

- `_executor_loop`: `prepare_resources` at `4261`; forward at `4332`
  (`self._forward_step(scheduled_batch)`), inside the `if can_queue:` block opened at `4280`,
  i.e. after both `can_queue` checks and after `_revert_gen_alloc`/`_finalize_adp_dummy_allocation`
  at `4273-4275`.
- `_executor_loop_overlap`: `prepare_resources` at `5094`; forward at `5185`
  (`self._forward_step(scheduled_batch, ...)`), inside the `if can_queue:` block opened at
  `5118`, i.e. also after both checks (`5107-5109`).
- `_executor_loop_pp`: `prepare_resources` at `2715`, inside the `else:` branch of the single
  `can_queue` check (`2691-2696`); forward happens later in that same `else` branch (not fully
  traced line-by-line in this pass, but structurally the forward step for PP microbatches is
  reached only through this branch, so it is downstream of `prepare_resources` by construction).

Since `prepare_resources` dispatches synchronously through the `ResourceManager` registry
(`resource_manager.py:3002-3005`) with target-then-draft ordering (see §2b), and the forward
pass is a separate statement strictly later in the same synchronous function body, there is
**no interleaving window** where the draft model's forward pass could start before its own
`prepare_resources` has run for the current iteration's request set, in any of the three loops,
for the *current* iteration's `scheduled_batch`.

(Note: this addresses forward-pass-vs-`prepare_resources` ordering only. §5 below addresses the
separate, intentional one-iteration skew for `update_resources`/post-forward bookkeeping in the
overlap loop, which is unrelated to this ordering guarantee.)

**Confidence: Verified current behavior**, for all three loops, for the capacity-growth/
mirroring call (`prepare_resources`). The draft *model*'s own forward pass (as opposed to the
draft *KV cache manager's* resource prep) is driven separately by `self.drafter.prepare_draft_tokens`
(`_executor_loop`: `4297`) / `self._handle_speculative_decoding` (`_executor_loop_overlap`:
`5153`), both of which occur after `resource_manager.prepare_resources` at `4261`/`5094`
respectively, inside the same `if can_queue:` block — so the draft model's forward is also
downstream of its own KV manager's `prepare_resources` mirroring within the same iteration.

---

## 5. Multi-iteration / overlap-executor skew

**Verified current behavior — `prepare_resources` (capacity growth/mirroring) is NOT skewed;
`update_resources` (post-forward bookkeeping) IS intentionally skewed by exactly one iteration,
uniformly across all registered managers including the draft one.**

In `_executor_loop_overlap`:
- `resource_manager.prepare_resources(scheduled_batch)` (`py_executor.py:5094`) operates on the
  **current** iteration's `scheduled_batch`, freshly returned from
  `_prepare_and_schedule_batch()` → `_schedule()` → `scheduler.schedule_request(...)` at the top
  of the same loop iteration (`5036` → `6276`). There is no deferred/queued version of this
  call — draft-manager capacity mirroring for iteration N always corresponds to exactly the
  request set the scheduler committed to in iteration N's own `schedule_request` call. This
  directly answers the concern in the task prompt: there is **no off-by-one skew** between when
  the scheduler commits to a request set and when the draft manager's `prepare_resources` sees
  that same request set — they are the same synchronous call stack, same iteration.
- `resource_manager.update_resources(scheduled_requests, ...)` (`py_executor.py:5424`), by
  contrast, is called from `_process_previous_batch()` (`5414-5428`) on
  `self.previous_batch.scheduled_requests` (`5420`) — i.e., the **N-1** iteration's
  `scheduled_batch`, captured into `self.previous_batch` at the end of the N-1 iteration
  (`5300-5309`, `BatchState(scheduled_requests=scheduled_batch, ...)`). This is the standard
  overlap-executor pattern: iteration N launches its own forward asynchronously, then finalizes
  N-1's already-completed sample state (`_update_requests(self.previous_batch.sample_state)` at
  `5192`) and only then calls `update_resources` on N-1's batch. This skew is applied uniformly
  — the same `ResourceManager.update_resources` registry dispatch
  (`resource_manager.py:3017-3031`) that includes the draft manager is called with the same
  `scheduled_requests` argument for every registered manager type, so there is **no additional
  skew between the target and draft managers within `update_resources`** — both see the same
  (N-1) `scheduled_requests` object at the same call.

So: the "GENERATION_TO_COMPLETE ... `mark_request_done` one iteration after `prepare_resources`"
characterization in the task background is more precisely: `update_resources`/`free_resources`
bookkeeping (which is where `PeftCacheManager.free_resources` → `self.impl.mark_request_done`
lives, `resource_manager.py:3174-3175`, unrelated to KV capacity) is one-iteration-skewed by
design in the overlap loop; `prepare_resources` (KV capacity growth/mirroring, the thing this
research pass is about) is not.

`GENERATION_TO_COMPLETE` (`llm_request.py:33-34`, set at `py_executor.py:7776` and `7805`) is a
request-state marker for deferring "will this request's last-generation-logits be excluded"
bookkeeping by one iteration under the overlap scheduler — it does not gate or delay
`prepare_resources`/draft-manager mirroring; it is orthogonal (affects
`set_exclude_last_generation_logits` and downstream sampling/logits-exclusion, not KV capacity).
Traced via `_update_generation_requests_that_will_complete_next_iteration`
(`py_executor.py:7765-7776`) and `_update_request_states_tp`
(`py_executor.py:7778-7807`); grep confirms no reference to `GENERATION_TO_COMPLETE` in
`resource_manager.py` other than an unrelated docstring reference at `resource_manager.py:2903`
(inside `BaseKVCompressionManager`'s `update_resources` docstring, explaining why it uses
`context_requests_last_chunk` instead of state transitions — again a different manager, KV
compression, not draft-KV mirroring).

**Confidence: Verified current behavior** for: (a) `prepare_resources`/draft mirroring is same-
iteration, no skew; (b) `update_resources` is one-iteration-skewed by design, uniformly across
all registered manager types including draft; (c) `GENERATION_TO_COMPLETE` is unrelated to KV
capacity/draft mirroring.

One residual point is **Source-inconclusive, requires unit/fault-injection test**: whether the
one-iteration skew in `update_resources` (which is where token/page *release* for finished or
partially-generated tokens happens) can, combined with the §3 kv-connector revert gap, compound
into a multi-iteration drift in the draft manager's capacity accounting under attention-DP with
a KV connector configured. Static reading establishes the two mechanisms independently
(§3's un-reverted mirror, §5's one-iteration-skewed release) but does not establish whether
`update_resources`/`free_resources` on a later iteration naturally reconciles the un-reverted
mirror from §3 (e.g., because the same request, still active, gets its draft capacity re-
computed idempotently next time `_prepare_draft_resources` runs) or whether it compounds. This
would need either (a) reading `KVCacheManagerV2._prepare_draft_resources` and its shrink/grow
call graph in detail, or (b) a fault-injection test that runs several iterations with a KV
connector forcing intermittent `can_queue` flips after `prepare_resources`, then asserts the
draft manager's block/page counters return to the expected steady-state.

---

## Open Questions

1. Does `KVCacheManagerV2._prepare_draft_resources` (`kv_cache_manager_v2.py:2779-2785` and
   beyond, not read in this pass) recompute/overwrite the draft manager's per-request capacity
   from scratch each time it mirrors a still-active request, or does it incrementally grow
   without ever shrinking absent an explicit revert/suspend call? This determines whether the
   §3 un-reverted-draft-mirror scenario is a genuine leak or self-heals on the next iteration
   the request is (re-)scheduled.
2. Is there a scenario where the same request appears in `scheduled_batch.generation_requests`
   across two consecutive iterations with `prepare_resources` mirroring capacity both times, but
   the *target* manager's capacity was reverted after the first mirror (§3) — could this cause
   draft capacity to grow strictly faster than target capacity over many iterations under
   sustained kv-connector-triggered `can_queue` flips? Static analysis alone cannot bound this;
   would need instrumented multi-iteration fault injection.
3. This pass did not trace `_pp_schedule_and_propagate` / `_pp_retry_until_can_schedule`
   internals in the PP loop in the same depth as the non-PP loops (only confirmed the ordering
   of the cited line numbers). If a future refactor touches PP-specific scheduling, that helper
   pair should be read in full before relying on "PP loop has no revert-after-prepare_resources
   race" as a universal invariant — it was confirmed only for the specific `can_queue`/
   `prepare_resources`/revert call sites cited in §3, not for every PP branch (e.g. drain-for-
   rebalance at `2683-2688`, or `_pp_ring_is_drained()` handling at `2927`).
4. The batch-waiting / fp8-context-MLA-cap deferral-without-revert behavior noted in §2a
   (`_waiting_requests`, `_cap_context_by_total_kv_len`) was flagged by the code's own comments
   as an accepted trade-off ("mirroring `_waiting_requests`"), not investigated further here
   since it is a scheduler-internal capacity-accounting question, not an executor-ordering
   question — but it is adjacent enough to the draft-mirroring correctness question in
   Open Question 1 that a future pass on `kv_cache_manager_v2.py` internals should account for
   it: dropped-but-not-reverted context requests never reach `prepare_resources` this iteration,
   so the draft manager never even attempts to mirror them until a later iteration when the
   scheduler re-admits them.
