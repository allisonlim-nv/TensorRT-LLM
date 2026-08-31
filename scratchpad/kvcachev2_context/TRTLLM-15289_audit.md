# TRTLLM-15289 — Audit: Scheduler / KV Cache Manager Contracts (KVCacheV2Scheduler ↔ KVCacheManagerV2)

**Repo:** `/Users/allim/TensorRT-LLM`
**Commit:** `4716843cee6e7a6c08bf4d8be29fae25321a9344`
**Branch:** `feat/native-kv-events-clean`
**Date:** 2026-08-31

**Inputs (already-completed, read-only audits reused as evidence base — no broad rediscovery performed):**
- `scratchpad/kvcachev2_context/scheduler.md`
- `scratchpad/kvcachev2_context/manager.md`
- `scratchpad/kvcachev2_context/interface_map.md`

All `path:line` citations below are carried from those three documents (themselves independently re-verified against `scheduler_v2.py`, `kv_cache_manager_v2.py`, `_util.py`, `py_executor.py`, and native headers under `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`). No production code, tests, or config were modified to produce this document, and no benchmarks were run.

**Scope note:** This document does not propose code fixes. Every item below is reported as a behavior of the current code, evidence-cited, and labeled Verified / Inferred / Unresolved. Nothing is called a "bug" or "leak" unless the cited evidence directly proves it.

---

## Executive Summary (for JIRA description)

KVCacheV2Scheduler and KVCacheManagerV2 communicate almost entirely through a **try/fail boolean API** (`prepare_context`, `resize_context`, `try_allocate_generation`, `prepare_disagg_gen_init`) with no capacity-query API (`can_schedule` is a stub returning `True` unconditionally — `scheduler_v2.py:1250-1259`). This audit traced every scheduler→manager call site and reconciled the two sides' documented contracts. Findings, scoped to the four JIRA questions:

1. **Generation self-eviction.** On allocation failure the scheduler tries eviction (`_try_evict_for_gen`) then recompute-pause (`_try_recompute_pause_for_gen`); if both are exhausted it "self-evicts" the requesting request itself via `_suspend_request`, which suspends the primary manager and (if present) mirrors the suspend onto the draft manager, then returns `STOP` for the whole scheduling phase — not a per-request retry. Both suspend calls are unconditional, unguarded void calls with no observed exception handling between them; if the draft mirror call were to fail after the primary succeeded, nothing in the scheduler code would detect or repair the resulting divergence. No such failure was observed in the read code paths — this is a structural risk, not a demonstrated defect (**Unresolved**, needs native-exception evidence).

2. **Context allocation/resizing.** `prepare_context` can mutate `req.context_remaining_length` (block-reuse trimming) and create/resume the native cache *before* the later `resize_context` call is even attempted; if `resize_context` then fails, the manager's documented behavior is to leave the cache **suspended** (not freed) on the **first** chunk only — non-first-chunk failure suspension is undocumented in the code read (**Unresolved**). A cross-context reservation failure unwinds already-grown primary capacity via `_suspend_request` (suspend, not free). Downstream, `revert_allocate_context` (called only by `py_executor.py`, never the scheduler) can itself escalate to a full `free_resources` teardown instead of a simple resize-back, when committed history has advanced past the revert point (`kv_cache_manager_v2.py:2536-2538`) — a real, evidence-backed divergence from a "just revert" mental model, not a bug in the strict sense (no incorrect final state was shown).

3. **BudgetTracker vs. manager capacity.** These are two structurally independent accounting systems — Python per-iteration token/request/PEFT counters vs. native KV-page counters — with **no reconciliation API** between them; the scheduler never queries manager capacity, it only observes call success/failure. One concrete, scoped divergence was confirmed: during the disaggregated context→generation transmission-complete transition window, the manager's internal `_effective_draft_len` can substitute a larger draft-token reservation than the scheduler's `BudgetTracker` accounted for via `get_draft_token_length` (`kv_cache_manager_v2.py:2418-2429`). This causes the manager to reserve **more** capacity than the scheduler budgeted, not less — no under-allocation/limit-violation risk was found in the evidence. In steady-state aggregate speculative decoding, the two draft-length computations are proven identical.

4. **Speculative decoding (target/draft).** The scheduler drives only the primary (target) manager's capacity (`try_allocate_generation`/`prepare_context`/`resize_context`); it only ever mirrors `suspend_request` and `free_resources` onto `draft_kv_cache_manager`. Draft-manager capacity growth/resize/resume is entirely self-driven inside the manager's own `prepare_resources`→`_prepare_draft_resources`/`update_resources`, invoked once per iteration by `py_executor.py` for each manager independently — this ordering relationship between the scheduler's target-side decisions and the executor's per-manager dispatch to the draft manager was not verified end-to-end (**Unresolved**). No confirmed case of target/draft state actually diverging (e.g., target active while draft freed) was found in the evidence; the divergence risk identified is structural (unguarded sequential mirror calls in `_suspend_request`), not observed.

No current-code defect was proven strongly enough to warrant a code-fix recommendation. Four items are labeled **Confirmed** behavioral divergence-from-naive-expectation (carried from `interface_map.md`'s Confirmed Mismatches section) but none is shown to produce an incorrect end state; ten items remain **Unresolved**, requiring native C++ source (`kvCache.cpp`, `kvCacheManager.cpp`, binding glue) or targeted runtime tests to close. A prioritized test plan is given at the end.

---

## 1. Generation Self-Eviction — `_try_schedule_generation` / `_try_evict_for_gen` / `_suspend_request`

### Call chain and state trace

`_try_schedule_generation(req, budget, ...)` (`scheduler_v2.py:965-1040`), called from Phase 1 of `_schedule_loop` (`scheduler_v2.py:406-421`).

**Before any manager call (Verified fact):** `req_tokens = beam_width + get_draft_token_length(req)` (`scheduler_v2.py:983-984`); `budget.can_fit_tokens(req_tokens)` gates entry — `False` → `STOP` immediately, no manager call at all (`:986-987`). Beam-width consistency is enforced across the iteration's batch of generation requests (`:989-992`).

**Manager call and state (Verified fact):** `success = self.kv_cache_manager.try_allocate_generation(req)` (`scheduler_v2.py:994`). Manager side (`kv_cache_manager_v2.py:2465-2491`):
- *Before* attempting resize: computes `draft_len` and writes `self._allocated_draft_lens[req_id] = draft_len` (`:2480-2481`) — **this Python-side dict entry is written even if the subsequent resize fails**; it is only cleared later by `pop()` in `revert_allocate_generation`/`free_resources`/a later successful `extend_capacity_for_tokens` call, not rolled back inline on failure.
- Native mutation: `kv_cache.resize(current_capacity + 1 + draft_len)` (`:2485`).
- Helix-specific: `req.py_helix_decode_group_index` is only incremented **after** a successful resize (`:2483-2490`, "commit only on success so a same-pass retry recomputes the same step instead of skipping one" — explicit at-most-once-on-success invariant for that counter).
- Returns `False` on resume-or-resize failure; no exception raised for ordinary capacity exhaustion.

**Manager state before/after `try_allocate_generation` failure (Verified fact + Inference):** Before: cache in whatever state it was (active or suspended-then-resumed). After a failed call: cache is unchanged (resize failed, no capacity growth applied), but `self._allocated_draft_lens[req_id]` has been overwritten to the newly-computed `draft_len` regardless of outcome — a Python-side bookkeeping write survives a failed native call. No other partial native mutation was observed in the read range of `try_allocate_generation` (manager.md §9).

### On failure: Helix vs. ordinary path

- **Helix (`self.has_cp_helix`) (Verified fact):** raises `RuntimeError` immediately — "No-evict stance: every validated helix run used GUARANTEED_NO_EVICT semantics; eviction is disabled under helix" (`scheduler_v2.py:997-1008`). No retry, no eviction, no self-suspend. Corroborated by test `test_kv_cache_manager_v2_helix_superblock.py:342` (`RuntimeError` matching "eviction is disabled under helix").
- **Ordinary path (Verified fact):** tries `_try_evict_for_gen` (`scheduler_v2.py:1009-1011`), then, if still failing, `_try_recompute_pause_for_gen` (`:1013-1023`).

### `_try_evict_for_gen` (`scheduler_v2.py:1104-1143`)

Searches backward from `req_it_end` (exclusive of the current, not-yet-processed range) for the first *evictable* victim — `_is_evictable` requires the victim not be in-flight, be a "started" request, and have `kv_cache_manager.is_request_active(...) == True` (already-suspended requests are skipped as useless victims, `scheduler_v2.py:1070-1080`). Victim is suspended via `_suspend_request(victim)` (mirroring onto draft, see below), `req_it_end` shrinks to the victim's index, and `try_allocate_generation(req)` is retried. Loops until success or no evictable victim remains. **Invariant assumed and stated in-code (Verified fact, `scheduler_v2.py:1112-1116`):** "Victims are always at indices ≥ req_it (not yet processed by the main loop), so they are never in `scheduled_ctx`/`scheduled_gen` and no token budget reclaim is needed" — i.e. eviction victims never had `BudgetTracker` state committed for them, so eviction does not require budget rollback.

### `_try_recompute_pause_for_gen` (`scheduler_v2.py:1082-1226`)

A more destructive fallback (no-op if `not self.enable_recompute_pause`, which is `False` on disaggregated generation workers, `_util.py:3168`). Victim selection additionally excludes `GENERATION_TO_COMPLETE`-state and certain multimodal-replay-incompatible requests (`:1082-1096`). Victim teardown is via `_recompute_pause_request(victim)` → `kv_cache_manager.free_resources(req)` (full teardown, not suspend) and, if present, `draft_kv_cache_manager.free_resources(req)` (`:1098-1102`). Gated additionally by `kv_cache_manager.can_evict` (a static post-construction boolean, `= len(config.cache_tiers) > 1`, `kv_cache_manager_v2.py:1241`) for reuse of already-evicted/recompute-paused victims and for a compound retry that re-invokes `_try_evict_for_gen` after a recompute-pause.

### If eviction/recompute-pause is exhausted ("self-eviction")

**Verified fact (`scheduler_v2.py:1028-1040`):**
```
if self.kv_cache_manager.is_request_active(req.py_request_id):
    self._suspend_request(req)
    evicted.append(req)
return STOP
```
So on total exhaustion, the requesting request's own cache — if still active — is suspended (not freed), appended to the `evicted` list for bookkeeping, and `STOP` is returned. `STOP` halts the **entire Phase-1 loop for this scheduling iteration**, not just this one request; the request will be reconsidered on the next `schedule_request` call.

### `_suspend_request` (`scheduler_v2.py:1054-1068`) — target/draft mirroring, and what happens if either suspend fails

**Verified fact:**
```
def _suspend_request(self, req):
    self._clear_request_runtime_state(req)      # req.py_batch_idx = None
    self.kv_cache_manager.suspend_request(req)   # primary
    if self.draft_kv_cache_manager is not None:
        self.draft_kv_cache_manager.suspend_request(req)  # draft mirror
```
(line numbers per scheduler.md §8a: `:1063` primary, `:1064-1065` draft mirror). A `TODO` comment in the same block (`:1056-1061`) states PEFT resources are explicitly **not** released here — the code itself flags this as a known gap ("could cause `ensure_batch` to fail if it needs to load a different adapter into a full cache"), separate from the KV-cache question.

**Manager-side `suspend_request(req) -> None` contract (Verified fact, `kv_cache_manager_v2.py:2750-2754`, manager.md §9):** no-op if cache absent or already inactive; otherwise calls native `kv_cache.suspend()`. No return value, no documented exception surface in the header comment read.

**Direct answers:**

- **"What happens if target suspension fails?"** — `Unresolved`. `suspend_request` is documented (in the header comments read) as a no-op-safe void call with no failure return channel. If the underlying native `kv_cache.suspend()` were to raise (no `terminateOnException` coverage was verified — manager.md §5, Open Q7), the exception would propagate up through `_suspend_request` and `_try_schedule_generation` uncaught in the code paths read; there is no scheduler-side `try/except` around this call. No test or code path demonstrates this actually happening.
- **"What happens if draft suspension fails?"** — `Unresolved`, same reasoning. The draft mirror call (`:1064-1065`) executes **after** the primary succeeds, with no exception handling between the two calls visible in the read code. A failure here would leave the primary manager already suspended while the draft mirror call raised — an uncaught exception, not a caught-and-compensated failure.
- **"Can main and draft states disagree?"** — **structurally possible, not demonstrated.** Because the two `suspend_request` calls are sequential, unguarded, and void-returning, a failure in the second call (draft) after the first (primary) succeeds would leave primary=suspended, draft=unchanged, with no code path visible that re-synchronizes them. No evidence was found that this occurs in practice (no test exercises it), and the manager's own `suspend_request` is documented as tolerant of "no cache" / "already inactive" states, which reduces (but does not eliminate, absent a demonstrated native-exception case) the practical likelihood.
- **"What cleanup or retry behavior exists?"** — At the `_try_schedule_generation` level: none beyond returning `STOP` (phase-level retry deferral to the next `schedule_request` call). At the `_suspend_request` level: none — it is a straight-line sequence of two calls with no compensating rollback logic if either raises.

**Confidence:** Verified for the happy path (both suspends succeed, mirroring is intentional and consistently applied at every call site — `_try_evict_for_gen`, `_suspend_request` itself, `_recompute_pause_request`). Unresolved for the failure-of-either-suspend-call path, which requires native/binding-layer evidence (`terminateOnException` coverage) to close.

**Minimal test to resolve:** A manager-level unit test that forces `KvCache::suspend()` (or its Python-backend equivalent) to raise on the **second** of two calls in a `_suspend_request`-style primary-then-draft sequence, asserting whether the exception propagates cleanly (recoverable) or the process aborts (`std::terminate`), and whether the primary manager's cache is left correctly suspended regardless. Starting point: `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py`'s existing suspend/resume lifecycle test (`:2365-2395`), extended with a draft-manager pair and an injected failure.

---

## 2. Context Allocation / Resizing — `prepare_context` / `resize_context` / chunked / cross-context / revert

### `prepare_context(req) -> bool` (`kv_cache_manager_v2.py:2579-2645`)

Call sites: `scheduler_v2.py:579` (full), `:628` (chunked), and directly for the cross-attention pool at `:918`.

- **Precondition (Verified fact):** `assert not req.is_disagg_generation_init_state` (`:2587-2589`).
- **Explicit non-guarantee (Verified fact, docstring):** "Create `_KVCache`, handle block reuse, and resume. Does NOT resize" (`:2580`) — capacity growth is deferred entirely to the caller's subsequent `resize_context` call.
- **State before call:** for the first chunk, no cache may exist yet. **State after success (Verified fact):** cache created (if absent) via `_create_kv_cache`; for block-reuse-enabled configurations, `req.context_current_position`/`req.set_prepopulated_prompt_len` are set (`:2623-2629`), and — critically for the scheduler's subsequent budget check — **`req.context_remaining_length` is mutated as a side effect of block-reuse trimming**, read immediately by the scheduler at `scheduler_v2.py:583-592` ("Prepare first so block reuse updates `context_remaining_length` before budget check," comment `:577-578`). For non-first chunks: asserts the cache already exists (`:2641-2643`) and only resumes it.
- **Failure (Verified fact):** returns `False` if `_create_kv_cache` returns `None` (IndexMapper saturated — an explicitly documented, expected, retryable condition per `kv_cache_manager_v2.py:4149-4158`, "Skipping KV cache creation; request will retry next iteration") or if resume fails. Scheduler reaction: `_try_schedule_context_full` returns `STOP` (not SKIP) on `prepare_context` failure (`scheduler_v2.py:580-581`) — halts the phase, not just this request.

### `resize_context(req, num_tokens) -> bool` (`kv_cache_manager_v2.py:2646-2676`)

Call sites: `scheduler_v2.py:596` (full), `:698` (chunked), and `:924` for the cross-attention pool.

- **Preconditions (Verified fact):** `assert not req.is_disagg_generation_init_state` (`:2652-2654`); raises `ValueError` if `self._has_cp_helix and not req.is_dummy_request` — helix requests must never take the context path (`:2655-2661`).
- **Postcondition on success (Verified fact):** resizes to `max(current_capacity, context_current_position + num_tokens + num_extra_kv_tokens)` (`:2666-2670`); sets `req.py_ctx_pre_resize_cap` (Python-side, used later by `revert_allocate_context`) (`:2675`).
- **State before/after failure (Verified fact for first chunk; Unresolved for non-first):** on resize failure, **if this is the first context chunk**, the manager calls `kv_cache.suspend()` before returning `False` (`:2672-2674`) — the cache the `prepare_context` call just created/resumed is left registered in `kv_cache_map` but inactive, not destroyed. For **non-first** chunks, neither the docstring nor the code path read documents an equivalent suspend-on-failure — this is silent, not contradicted (`interface_map.md` §3 table, row 2; manager.md §9 `resize_context`).
- **Scheduler reaction (Verified fact):** `SKIP` (not `STOP`) — `scheduler_v2.py:596-597,699` — the request may become schedulable again in a future iteration once more capacity frees up. The scheduler does **not** itself call any explicit rollback here; it relies entirely on the manager's documented (first-chunk-only) suspend-on-failure behavior.

**Manager mutations that occur before a later check can fail:** Concretely, for a single context-scheduling attempt: (1) `prepare_context` may create the native cache and mutate `req.context_remaining_length` — both survive even if the subsequent `resize_context` fails; (2) `resize_context`, on success, sets `req.py_ctx_pre_resize_cap` and grows native capacity — both survive even if the subsequent `_try_schedule_cross_context` step fails.

### `_try_schedule_context_chunked` (`scheduler_v2.py:606-708`)

Same `prepare_context`→`resize_context` sequence as the non-chunked path, with an added dependency: `chunk_size` is computed from `req.context_remaining_length` **cached once** right after `prepare_context` (`scheduler_v2.py:634`), then used through several chunk-boundary computation steps (`_align_chunk_to_mm_block`, forced-chunk-boundary helpers) before the later `resize_context` call. **Invariant assumed (Inference):** `context_remaining_length` is stable between the `prepare_context` call and the `resize_context` call within one scheduler invocation — nothing re-queries the manager in between. No manager-side contract (positive or negative) confirms this stability (`interface_map.md` §1 table, row 3 — Unresolved).

### Cross-context reservation (`_try_schedule_cross_context` / `_try_schedule_cross_context_v2`, `scheduler_v2.py:880-963`)

Called immediately after `resize_context` succeeds, for encoder-decoder cross-attention KV. Two paths:
- **Public-API path** (non-`KVCacheManagerV2` cross manager): `cross_kv_cache_manager.prepare_context(req)` then `.resize_context(req, req_tokens)` — same contract as above (`:918-926`).
- **V2 fast path** (`_try_schedule_cross_context_v2`, static method, `:928-963`) — **Verified fact, re-verified directly against source (`interface_map.md` line 13):** this method reaches into `cross_kv_cache_manager.kv_cache_map`, `._create_kv_cache(...)`, `._resume_and_restore(...)`, and per-`_KVCache` methods (`.resize`, `.suspend`, `.stop_committing`, `.cuda_stream`) directly, rather than going through the public `prepare_context`/`resize_context` API used by every other scheduler call site. This is a **Confirmed** structural fact (both scheduler.md and manager.md, plus a direct re-read, agree): the manager's own documentation of `_create_kv_cache`'s intended callers (`prepare_context`/`prepare_disagg_gen_init`/`add_dummy_requests`/`_prepare_draft_resources`) does not list the scheduler. This is reported as a **layering observation**, not a proven defect — no incorrect behavior was demonstrated, only that the scheduler is coupled to manager-internal names.

**Failure/rollback (Verified fact):** if the cross-context step doesn't return `SCHEDULED`, both context-scheduling callers call `self._suspend_request(req)` (`scheduler_v2.py:600-602,701-704`) — this suspends **both** managers (primary and draft, if present) for `req`, even though only the primary context resize had actually succeeded at that point. So a cross-KV failure unwinds the primary allocation via **suspend**, not free — capacity is retained (parked) for a future resume, not released back to the pool immediately.

### `revert_allocate_context(req) -> None` (`kv_cache_manager_v2.py:2525-2547`) — orchestration-caller-only rollback

Call site: `py_executor.py:3477` only — **never called by the scheduler itself.** Driven by `req.py_ctx_pre_resize_cap`, used when a downstream admission-control step (attention-DP `can_queue=False`, disagg admission rejection) discards a request the scheduler already admitted.

- **No-op conditions (Verified fact):** `py_ctx_pre_resize_cap` unset, cache absent, cache inactive, or `pre_cap >= current capacity` (`:2527-2535`).
- **Normal path (Verified fact):** `kv_cache.resize(pre_cap, min(history_length, pre_cap))`, then `kv_cache.suspend()` if `pre_cap > 0` (`:2539-2547`); raises `RuntimeError` on resize failure (`:2541-2545`).
- **Escalation case (Verified fact, `interface_map.md` Confirmed Mismatch #2):** if `kv_cache.history_length > pre_cap` at revert time — i.e. committed history has already advanced past the point being reverted to — the method calls **`self.free_resources(req)`** instead of resizing (`:2536-2538`). This is a materially different, more destructive outcome (full cache teardown + IndexMapper slot release) than a simple capacity shrink-back. This is reported as a real, evidence-backed divergence from a "revert = shrink" mental model; it is not shown to produce an incorrect end state (a fully-freed cache is a valid, consistent state for a request that is being discarded), so it is **not** labeled a bug.

### Summary — is capacity retained for retry, reverted, suspended, or unaccounted for, per failure path?

| Failure point | Outcome |
|---|---|
| `prepare_context` fails (IndexMapper saturated / resume fails) | No cache created (saturation case) → nothing to account for; or existing-cache resume failure → **Unresolved** what state the pre-existing cache is left in (not documented in the read code) |
| `resize_context` fails, first chunk | **Suspended** (Verified) — capacity parked, cache inactive, retryable via resume |
| `resize_context` fails, non-first chunk | **Unresolved** — no suspend-on-failure documented for this case in the code read |
| Cross-context reservation fails | Primary capacity **suspended** via `_suspend_request` (Verified) — not freed, not left active |
| Downstream admission rejects an already-scheduled context request | `revert_allocate_context`: normally **reverted** (resize-back + suspend); **escalates to full free** if committed history has advanced past the revert point (Verified, `interface_map.md` Confirmed Mismatch #2) |

**Confidence:** Verified for first-chunk resize failure and cross-context failure unwind; Unresolved for non-first-chunk resize failure and for `prepare_context` failing against a pre-existing (non-first-chunk) cache.

**Minimal test to resolve:** Extend `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py` with a multi-chunk context request where `resize_context` is forced to fail on the **second** chunk (not the first); assert `kv_cache.is_active`/status afterward, to determine whether non-first-chunk failure suspends the cache the same way first-chunk failure does.

---

## 3. `BudgetTracker` vs. Manager/Native Page Accounting

`BudgetTracker` (`scheduler_v2.py:51-141`) is a **pure-Python, per-`_schedule_loop`-call** counter with no notion of KV pages, tiers, or native capacity — it is freshly constructed every scheduling iteration (`:284-288`) and discarded at the end. It gates admission via `can_fit_tokens`/`requests_full`/`peft_pages_needed`, and is updated via `commit`/`commit_peft`/`pre_claim_peft` only on the scheduler's own say-so, never by querying the manager.

Every possible divergence point identified, classified per the task's rubric:

| # | Divergence point | Classification | Evidence |
|---|---|---|---|
| 1 | Token/request counters (`num_tokens`, `num_requests`) vs. native KV pages | **Intentional different accounting domain.** BudgetTracker counts scheduling slots and forward-pass token budget; the manager counts physical KV pages/blocks per tier. There is no shared unit and no reconciliation API — the scheduler gates by manager call success/failure only (`can_schedule` is a stub, `scheduler_v2.py:1250-1259`). | scheduler.md §2; interface_map.md §4 |
| 2 | PEFT pages (`_claimed_peft_pages`, tied to `peft_cache_manager`) vs. KV pages | **Intentional different accounting domain.** PEFT adapter cache pages are a wholly separate resource pool from KV cache pages, tracked by a different manager (`peft_cache_manager`) entirely. | scheduler.md §2 (`scheduler_v2.py:74-76`) |
| 3 | Scheduler's `get_draft_token_length(req)` vs. manager's `_effective_draft_len(req)` — **aggregate (non-disagg) speculative decoding** | **Verified: no divergence.** `_effective_draft_len` returns `get_draft_token_length(req)` unmodified unless the disagg-transition guard (below) is true; that guard is never true outside disagg. Re-verified directly at `kv_cache_manager_v2.py:2418-2429`. | interface_map.md §2 (re-verified against `kv_cache_manager_v2.py:2400-2492`) |
| 4 | Same pair — **disaggregated context→generation transmission-complete transition** | **Verified mismatch (scoped).** When `draft_len == 0` and `req.is_disagg_generation_transmission_complete and req.context_phase_params is not None`, `_effective_draft_len` substitutes `context_phase_params.draft_tokens` length or, if also empty, `self.max_total_draft_tokens` (`kv_cache_manager_v2.py:2419-2428`) — while the scheduler's `BudgetTracker.commit` for that same request used `get_draft_token_length(req)`, which is `0` in this exact window since `req.py_draft_tokens` is still empty at scheduling time. The manager reserves **more** KV-page capacity than the scheduler's Python token budget accounted for during this transition; no under-allocation or limit-violation scenario was found — the direction of the mismatch is conservative (manager over-reserves), not unsafe. | interface_map.md §2, §Confirmed Mismatches #4 |
| 5 | Manager's draft-capacity reserve slack (`_kv_reserve_draft_tokens` padding in `_prepare_draft_resources`, and its reclaim in `update_resources` for draft managers) vs. scheduler's budget | **Intentional different accounting domain.** This is manager-internal padding for dynamic-tree draft managers, applied and reclaimed entirely inside the manager's self-driven draft lifecycle (`kv_cache_manager_v2.py` §7 per manager.md, corroborated with concrete numbers — 201 vs. 230 capacity — by `test_kv_cache_v2_capacity_only.py:124-138`). The scheduler never queries or budgets against this quantity. | manager.md §7, §12.5 |
| 6 | `GENERATION_TO_COMPLETE`-state PEFT pre-claim (`BudgetTracker.pre_claim_peft`) vs. an analogous KV-page-level carve-out | **Plausible but unproven (unresolved).** `pre_claim_peft` exists specifically because PEFT adapters for `GENERATION_TO_COMPLETE` requests are "not yet released" and would otherwise be invisible to the budget (`scheduler_v2.py:130-141`). A grep of `kv_cache_manager_v2.py` for `GENERATION_TO_COMPLETE`/`is_generation_to_complete` returns **zero matches** (re-verified, interface_map.md line 15) — the manager has no analogous carve-out for this specific state. `update_resources`'s completion short-circuit only special-cases `GENERATION_COMPLETE`/`CONTEXT_INIT` (`kv_cache_manager_v2.py:4070-4074`), not `GENERATION_TO_COMPLETE`. Whether this absence is safe-by-construction (because `GENERATION_TO_COMPLETE` requests are otherwise excluded from KV-affecting scheduler calls) or a genuine unhandled interaction was **not resolved** by either input audit. | interface_map.md §4, Unresolved item #10 |

**No item above is classified as a proven "leak" or "limit violation."** The one verified mismatch (#4) is conservative in direction (manager reserves more, not less, than the scheduler's budget implies), and item #6 is explicitly unresolved rather than asserted.

**Minimal test to resolve item #6 (the only open item in this section):** Construct a request in `GENERATION_TO_COMPLETE` state and drive it through `try_allocate_generation`/`update_resources` in the same scheduling iteration as a competing context request needing the budget that PEFT pre-claim would have reserved; assert no double-allocation/crash. Starting point: `test_kv_cache_v2_capacity_only.py`'s `GENERATION_COMPLETE` short-circuit test (`:141-149`), extended to `GENERATION_TO_COMPLETE`.

---

## 4. Speculative Decoding — Target/Draft Lifecycle

### Aggregate (two-model and one-model), non-disaggregated

**Manager construction constraints (Verified fact, `_util.py:2077-2093`):**
- **Two-model** (separate draft model engine exists): if `KVCacheManagerV2` and a `draft_kv_cache_config` is given, a hard `assert` requires `draft_kv_cache_config.max_gpu_total_bytes == self_kv_cache_config.max_gpu_total_bytes` — "KVCacheManagerV2 does not support two-model speculative decoding with separate draft GPU budgets" (`_util.py:2079-2087`). Offload-tier (host/disk) budgets **can** be split per-manager independently of this GPU constraint (`_util.py:2038-2044`).
- **One-model** (separate-layout draft cache, e.g. Eagle3): built via `_create_one_model_draft_kv_cache_manager`, disabled under attention-DP or DeepSeek-V4-sparse-with-`pp_size>1` (`_util.py:1432-1455`).
- Each `KVCacheManagerV2` instance (target and draft) constructs its **own independent native manager object** (`self.impl = KVCacheManagerPy(config, ...)`) and its own `IndexMapper` (`kv_cache_manager_v2.py:1174-1240,1360`) — target and draft are fully separate native objects, linked only by the scheduler/executor calling both and by shared request IDs (**Inference**, drawn from construction code; not stated verbatim in a single docstring).

**Which lifecycle transitions are mirrored to draft, and which are not (Verified fact):**

| Transition | Mirrored to draft? | Evidence |
|---|---|---|
| Suspend (eviction, cross-context failure unwind, self-eviction) | **Yes** — `_suspend_request` calls `draft_kv_cache_manager.suspend_request(req)` immediately after the primary call, if `draft_kv_cache_manager is not None` | `scheduler_v2.py:1063-1065` |
| Full teardown (recompute-pause) | **Yes** — `_recompute_pause_request` calls `draft_kv_cache_manager.free_resources(req)` after the primary | `scheduler_v2.py:1098-1102` |
| Generation allocation (`try_allocate_generation`) | **No** — scheduler never calls this on `draft_kv_cache_manager`; draft capacity is driven entirely inside the draft manager's own `prepare_resources` → `_prepare_draft_resources`, invoked once per iteration by `py_executor.py` for *each* manager separately | manager.md §7 (`kv_cache_manager_v2.py:2771-2785`); scheduler.md §11 |
| Context prepare/resize | **No**, same reasoning — draft-token capacity for the draft manager is folded into its own self-driven resource-preparation path, not into scheduler calls | manager.md §7 |
| Resume | **No** direct scheduler call to either manager's `resume_request` — resume is invoked from within `try_allocate_generation`/`prepare_context`/`_prepare_draft_resources` themselves, or from `py_executor.py` warmup paths, not the scheduler directly | scheduler.md §7,§9 (no `resume_request` call site in `scheduler_v2.py`) |

**Draft capacity reserve mechanics (Verified fact, manager.md §7):** `_prepare_draft_resources` pads generation-request resize to `self._kv_reserve_draft_tokens` beyond `_required_gen_capacity`; `update_resources` reclaims unused dynamic-tree reserve slack for draft managers only ("target managers do not allocate this reserve slack"), with concrete numeric corroboration (201 vs. 230 capacity) from `test_kv_cache_v2_capacity_only.py:124-138`.

**Draft-length accounting in the aggregate case (Verified, re-derived above in §3):** `_effective_draft_len(req) == get_draft_token_length(req)` always, in the non-disagg case — **the scheduler's Python budget math and the manager's internal draft-capacity reservation for the target manager agree exactly on the draft-token count in aggregate spec decoding.** (This is about the *count* used to size the target manager's own resize call; the draft manager's *own* pool sizing includes additional self-driven reserve slack per the paragraph above, which the scheduler never sees or needs to reconcile against.)

**Which invariants prevent — or fail to prevent — target/draft divergence:**
- **Prevents:** every scheduler-initiated suspend/free is unconditionally mirrored onto the draft manager when one is configured (`scheduler_v2.py:1063-1065,1098-1102`) — this is the scheduler's sole mechanism for keeping draft lifecycle state in lockstep with the primary for the transitions it controls.
- **Does not prevent (Unresolved):** the two mirror calls are sequential and unguarded (see §1 above) — a failure in the second call is not detected or compensated.
- **Does not prevent / not verified (Unresolved):** the draft manager's *own* self-driven `prepare_resources`/`update_resources` calls happen once per iteration from `py_executor.py`, on a code path outside the scheduler entirely. Whether there is any ordering guarantee ensuring the draft manager's per-iteration resource prep always reflects the *same* scheduling decision (same request set) the scheduler just made for the primary manager in that same iteration was **not verified** by either input audit — this is the most significant open question for target/draft alignment, because it is the one transition (allocation, not suspend/free) that is *not* mirrored by the scheduler and instead relies on independent, executor-driven synchronization.

**Can target and draft actually diverge? (Direct answer)** No demonstrated case of divergence was found in the evidence gathered. Two structural avenues exist where divergence is *possible but not proven*:
1. An exception in the second of the two sequential, unguarded mirror calls inside `_suspend_request`/`_recompute_pause_request` (§1).
2. An unverified ordering gap between the scheduler's target-manager scheduling decision and the executor's separate, once-per-iteration `prepare_resources` dispatch to the draft manager.

Neither is demonstrated by a failing test or an observed inconsistent state in the code paths read; both are reported as **Unresolved**, not as confirmed bugs.

### Disaggregated path

**Verified fact (`scheduler.md` §11; `py_executor.py:7206-7234`):** `_prepare_disagg_gen_init` in `py_executor.py` — the orchestration caller, not the scheduler class — iterates `(KV_CACHE_MANAGER, SPEC_RESOURCE_MANAGER, DRAFT_KV_CACHE_MANAGER)` and calls `prepare_resources` on each, for `fitting_disagg_gen_init_requests` (i.e., only requests the scheduler's `_try_schedule_disagg_gen_init` already admitted via `prepare_disagg_gen_init` on the primary manager, §2/§4 of scheduler.md). This is the disagg-specific analog of the aggregate path's draft mirroring — still executor-driven, not scheduler-driven, consistent with the aggregate case's pattern.

**The one confirmed accounting divergence found in this audit is scoped to this exact transition (Verified, re-derived in §3 item #4):** during the disagg context→generation transmission-complete window, with `req.py_draft_tokens` still empty, the (primary) manager's `_effective_draft_len` substitutes a larger value than the scheduler's `get_draft_token_length`-based budget commit used, when `try_allocate_generation` is later called on the primary manager for that request's first decode step. This is a **target-manager-vs-scheduler-budget** divergence, not a target-vs-draft-manager divergence per se — the evidence does not show the draft manager's own capacity diverging from the target's in this window, only that the *scheduler's Python bookkeeping* under-counts relative to what the (primary) manager actually reserves.

**Confidence:** Verified for the construction-time constraints (two-model GPU-budget parity, one-model gating conditions) and for the suspend/free mirroring pattern. Unresolved for cross-iteration ordering between scheduler decisions and the executor's separate draft-manager `prepare_resources` dispatch.

**Minimal test to resolve:** A disagg-specific test driving a request through `is_disagg_generation_transmission_complete=True` with empty `py_draft_tokens`, asserting the scheduler's committed `BudgetTracker` token count for that request is strictly less than the KV pages `try_allocate_generation` actually reserves via `_effective_draft_len`'s fallback path — quantifying the gap (already identified as non-hazardous in direction) so it is tracked going forward. A second, separate test should assert that the executor's per-iteration `prepare_resources` dispatch to `draft_kv_cache_manager` always covers exactly the request set the scheduler admitted for the primary manager in the same iteration, to close the ordering-guarantee open question.

---

## Contract Table

| Scheduler operation | Manager guarantee | Failure/rollback behavior | Target/draft status | Evidence | Confidence |
|---|---|---|---|---|---|
| `try_allocate_generation` (generation admission) | Resumes-if-needed then resizes by `1 + draft_len`; records `_allocated_draft_lens` **before** attempting resize (survives failure) | Returns `False`, no exception, no native rollback observed; scheduler retries via eviction/recompute-pause, then self-suspends the request itself if exhausted | Never called on draft manager directly; draft self-driven separately | `kv_cache_manager_v2.py:2465-2491`; `scheduler_v2.py:965-1040` | Verified |
| `_suspend_request` (eviction victim / self-eviction / cross-context unwind) | `suspend_request` no-op-safe if cache absent/inactive | No compensating action if either sequential call raises; no return-value failure signal | Primary suspended, draft mirrored **if** `draft_kv_cache_manager is not None` — sequential, unguarded | `scheduler_v2.py:1054-1068`; `kv_cache_manager_v2.py:2750-2754` | Verified (happy path) / Unresolved (failure-of-mirror path) |
| `_recompute_pause_request` (destructive fallback) | `free_resources` fully tears down cache + IndexMapper slot | No compensating action if either sequential call raises | Primary freed, draft mirrored if present — sequential, unguarded | `scheduler_v2.py:1098-1102`; `kv_cache_manager_v2.py:3691-3706` | Verified (happy path) / Unresolved (failure-of-mirror path) |
| `prepare_context` (context admission) | Creates cache + resumes + block-reuse trims `context_remaining_length`; explicitly does **not** resize | Returns `False` on IndexMapper saturation (documented, retryable) or resume failure; scheduler `STOP`s the phase | N/A (primary-only call) | `kv_cache_manager_v2.py:2579-2645`; `scheduler_v2.py:557-604,606-708` | Verified |
| `resize_context` (context capacity growth) | Grows to `max(current, position + tokens + extra)`; sets `py_ctx_pre_resize_cap` | On failure, **first chunk**: suspends cache before returning `False` (Verified). **Non-first chunk**: undocumented in code read (Unresolved) | N/A (primary-only call) | `kv_cache_manager_v2.py:2646-2676` | Verified (first chunk) / Unresolved (non-first chunk) |
| Cross-context (`_try_schedule_cross_context_v2`) | Reaches into manager private members (`kv_cache_map`, `_create_kv_cache`, `_resume_and_restore`) instead of the public API | Failure → `_suspend_request` unwinds already-grown primary context capacity via suspend (not free) | Suspends both primary and draft (via `_suspend_request`) | `scheduler_v2.py:928-963,600-602,701-704` | Verified (structural fact); not shown to cause incorrect state |
| `prepare_disagg_gen_init` | Allocates full-prompt (+draft) capacity in one call; on first-chunk resize failure, suspends (not destroys) | `False` → scheduler simply skips-and-retries next iteration, no explicit scheduler-side rollback call | Disagg draft/spec resource prep is executor-driven (`_prepare_disagg_gen_init` in `py_executor.py`), not scheduler-driven | `kv_cache_manager_v2.py:2678-2708`; `scheduler_v2.py:362-382,526-543` | Verified |
| `revert_allocate_generation`/`revert_allocate_context` (post-scheduling rollback) | Undoes prior successful growth; `revert_allocate_context` **escalates to full `free_resources`** if committed history advanced past the revert point | Silently no-ops if reverted capacity would be negative; raises `RuntimeError` on resize failure (unlike the bool-returning family) | N/A — called only by `py_executor.py`, never the scheduler | `kv_cache_manager_v2.py:2493-2523,2525-2547`; `py_executor.py:3438,3477,4217,5048` | Verified |
| `can_evict` (attribute read, gates recompute-pause escalation) | `= len(config.cache_tiers) > 1`, static after construction; can silently become `False` fleet-wide if any single rank's host-tier construction fails | No scheduler-visible signal distinguishes "host tier never configured" from "host tier lost to a construction-time fallback on another rank" | N/A | `kv_cache_manager_v2.py:1241,1178-1236` | Verified |
| BudgetTracker token/PEFT accounting vs. manager KV pages | No reconciliation API; independent accounting domains | N/A — scheduler never queries manager capacity, only observes call success/failure | Draft-token count agrees exactly with target's `_effective_draft_len` in aggregate spec decoding; diverges (manager reserves more) only in the disagg context→generation transition window | `scheduler_v2.py:51-141`; `kv_cache_manager_v2.py:2418-2429` | Verified |

---

## Prioritized Test Plan

1. **[High]** Native/manager-level test: force a failure in the **second** of two sequential `suspend_request`/`free_resources` calls in a primary-then-draft mirror sequence (`_suspend_request`, `_recompute_pause_request`); assert whether the exception is catchable in Python or aborts the process, and what state the primary is left in. Resolves JIRA Q1's core uncertainty. Starting point: `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py:2365-2395`.
2. **[High]** `resize_context` non-first-chunk failure: extend the same file with a multi-chunk request where the **second** chunk's resize fails; assert cache status. Resolves JIRA Q2's largest open item.
3. **[Medium]** Draft-length disagg-transition gap: a disagg test asserting `BudgetTracker`'s committed token count vs. `try_allocate_generation`'s actual manager-side reservation during the `is_disagg_generation_transmission_complete` window with empty `py_draft_tokens`; quantify (not just confirm) the gap. Resolves JIRA Q3/Q4's one confirmed divergence.
4. **[Medium]** Ordering guarantee between scheduler target-manager admission and executor-driven draft-manager `prepare_resources` dispatch within the same iteration — assert the draft manager's per-iteration resource prep always covers exactly the request set the scheduler admitted for the primary manager. Resolves the most significant open item in JIRA Q4.
5. **[Medium]** `GENERATION_TO_COMPLETE` KV-layer accounting: construct a request in that state and drive it through `try_allocate_generation`/`update_resources` alongside a competing context request; assert no double-allocation. Resolves JIRA Q3 item #6. Starting point: `test_kv_cache_v2_capacity_only.py:141-149` (extend the existing `GENERATION_COMPLETE` short-circuit test).
6. **[Low]** `revert_allocate_context` escalation-to-`free_resources` path: construct a request, grow via `resize_context`, advance `history_length` past `pre_cap` via commit, call `revert_allocate_context`, and assert the resulting state (full teardown) is handled correctly by both call sites in `py_executor.py`. Confirms — does not currently dispute — the documented escalation behavior from §2.
7. **[Low]** Fleet-wide `can_evict` degradation visibility: extend the existing `USE_NO_HOST` fallback tests with a scheduler-level assertion that `_try_evict_for_gen`/`_try_recompute_pause_for_gen` behave correctly (no crash, correct deadlock/no-deadlock outcome) when constructed against a manager already in the post-fallback (`can_evict=False`) state.

No code changes are proposed. Items 1, 2, and 4 are the highest-value follow-ups because they are the only three items in this audit where the *absence* of evidence (rather than a demonstrated safe behavior) is the reason a JIRA question remains open.
