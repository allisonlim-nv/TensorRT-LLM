# Coverage Closure Pass — KVCacheV2Scheduler ↔ KVCacheManagerV2

**Repo:** `/Users/allim/TensorRT-LLM`
**Commit:** `4716843cee6e7a6c08bf4d8be29fae25321a9344`
**Branch:** `feat/native-kv-events-clean`
**Date:** 2026-08-31

**Purpose:** Determine whether the four existing artifacts fully explain current KVCacheV2 scheduler/manager behavior, without repeating the underlying audit or reading additional source. This is a meta-pass over already-produced evidence.

**Inputs (read in full, no new source reading performed for this pass):**
- `scratchpad/kvcachev2_context/scheduler.md`
- `scratchpad/kvcachev2_context/manager.md`
- `scratchpad/kvcachev2_context/interface_map.md`
- `scratchpad/kvcachev2_context/TRTLLM-15289_audit.md`

**Status legend:**
- **Covered** — a source-cited `path:line` claim exists in the artifacts and is sufficient to explain the behavior.
- **Out of scope** — deliberately excluded by the original audits' scoping boundaries (stated explicitly in the artifacts), not a gap.
- **Missing trace** — the behavior is referenced by the artifacts but the underlying source (Python, C++ header, or `.cpp`/binding file) was not read; closable by further **source inspection**, no new test needed.
- **Needs unit/fault-injection test** — source inspection alone cannot establish the behavior (e.g., what happens when a call is made to fail, or two sequential calls interact under failure) — a Python-level test with a mocked/injected failure would resolve it.
- **Needs native/runtime test** — resolution requires exercising the compiled C++ backend or an actual multi-rank/multi-iteration runtime scenario; source-level fault injection in Python is insufficient.

---

## 1. Manager Construction and Topology

| Item | Status | Evidence |
|---|---|---|
| Two-model draft manager creation (separate engine, GPU-budget-parity assert) | **Covered** | manager.md §7 (`_util.py:2077-2088`); audit §4 |
| One-model draft manager creation (`_should_create_separate_draft_kv_cache`, gating) | **Covered** | manager.md §7 (`_util.py:1432-1455,1516-1583`) |
| GPU tier quota resolution (`max_gpu_total_bytes`/`max_tokens` min, multi-rank allreduce) | **Covered** | manager.md §6.1 (`kv_cache_manager_v2.py:1058-1104`) |
| Host tier — explicit sizing | **Covered** | manager.md §6.2 (`:1105-1106`) |
| Host tier — auto sizing (`_compute_auto_host_tier_quota`, rank-local division, fleet-min sync) | **Covered** | manager.md §6.2 (`:242-304,1108-1147`); test-corroborated (`test_kvv2_host_tier_sizing.py`) |
| Host tier — construction-failure fallback (`USE_NO_HOST`/`ABORT`, fleet-wide rank sync) | **Covered** | manager.md §6.4 (`:1178-1236`); test-corroborated (`test_kv_cache_manager_v2.py:356-451`) |
| Disk tier (explicit-only, no auto policy) | **Covered** | manager.md §6.3 (`:1151-1160`) |
| `IndexMapper` — capacity formula (`max_num_sequences * (2 if disagg else 1) + reserved`), saturation check (`num_free_slots()==0` in `_create_kv_cache`), Python-side role (request_id → zero-copy slot) | **Covered** | manager.md §6.6 (`:1339-1353`), §8 (`:1281-1298,1360-1362`), §9 `_create_kv_cache` (`:4149-4158`) |
| `IndexMapper` — native/binding-internal implementation (how slots are actually allocated/freed inside the bound `IndexMapper` type) | **Missing trace** | Not read by any artifact; only its Python-visible contract (capacity math, `num_free_slots`, `add_new_sequence`, `remove_sequence`) was traced, not its own source |
| `can_evict` — definition, static-after-construction semantics | **Covered** | manager.md §10 (`:1241`) |
| Fleet-wide `can_evict` silent degradation on rank host-tier failure | **Covered** (as a documented risk, not a bug) | manager.md §6.4/§11.10; interface_map.md Confirmed Mismatch #3; audit exec summary |
| Estimation-phase manager sizing (throwaway probes, dropped offload-tier budgets) | **Covered** | manager.md §6.5 |
| Dual cpp/python backend selection and symbol parity | **Covered** (selection mechanism) / **Missing trace** (whether cpp-only fallback symbols actually exist in the built extension) | manager.md §1 (`__init__.py:23,122-236`) — explicit open question on the fallback symbols |
| Native `KvCacheManager` full C++ implementation (`resize`/`adjust`/eviction-victim-selection internals, `.cpp` bodies) | **Missing trace** | manager.md §2 Open question — only header (`kvCacheManager.h`) comments read, `.cpp` not read |
| `config.cpp` validation bodies (tier-ordering enforcement beyond `quota>0`, `max_gpu_total_bytes`-adjacent checks) | **Missing trace** | manager.md §3 Open question |
| `terminateOnException` call-site coverage (which manager/KvCache methods abort the process vs. raise catchably) | **Missing trace**, closes via source read; residual behavioral confirmation **needs native/runtime test** | manager.md §5 Open Q7 |
| `kv_connector_manager` support (V2 does not support it) | **Out of scope** | Explicitly unsupported by hard `assert` (`kv_cache_manager_v2.py:832-834`) — not a V2 feature to audit further |
| `max_beam_width > 1` support | **Out of scope** | Explicitly unsupported (`:835`) |
| Streaming KV events on the default cpp backend | **Out of scope** (documented as unsupported, not investigated further) | manager.md §6.6 (`validate_streaming_support`, `:971-980`) |

---

## 2. Scheduler Lifecycle

| Item | Status | Evidence |
|---|---|---|
| `prepare_disagg_gen_init` entry (`_try_schedule_disagg_gen_init`) — call chain, budget interaction, failure handling | **Covered** | scheduler.md §4 (`scheduler_v2.py:351-382,526-543`); audit §2 |
| Full (non-chunked) context (`_try_schedule_context_full`) | **Covered** | scheduler.md §5a; audit §2 |
| Chunked context (`_try_schedule_context_chunked`), including `_align_chunk_to_mm_block` | **Covered** | scheduler.md §5b,§9; audit §2 |
| `context_remaining_length` stability between `prepare_context` and `resize_context` within one chunked call | **Needs unit/fault-injection test** | scheduler.md §5b Open Q3; interface_map.md §1 row 3 (Unresolved) — no manager contract found either way from source |
| Generation admission (`_try_schedule_generation`) | **Covered** | scheduler.md §7; audit §1 |
| Eviction (`_try_evict_for_gen`) | **Covered** | scheduler.md §8b; audit §1 |
| Recompute-pause (`_try_recompute_pause_for_gen`, victim search, frontier bookkeeping) | **Covered** | scheduler.md §8c; audit §1 |
| Self-suspend / self-eviction on exhaustion | **Covered** (happy path) / **Needs native/runtime test** (mirror-call failure path) | scheduler.md §7-8a; audit §1 — sequential unguarded primary→draft suspend calls, no exception handling observed |
| Rollback paths (`revert_allocate_generation`/`revert_allocate_context`) as consumed by `py_executor.py` | **Covered** at call-site level | manager.md §9; audit §2; scheduler.md §4,§7 (call sites `py_executor.py:3438,3477,4217,5048`) |
| Deeper `py_executor.py` orchestration logic surrounding rollback (admission-control decision logic itself, PP-follower re-run conditions, `_check_benchmark_disagg_gate` retry logic) | **Out of scope** | Original scheduler audit's stated scope: "KVCacheV2Scheduler and its immediate orchestration callers only" (scheduler.md header) — deeper `py_executor.py` control flow beyond call-site inventory was a deliberate boundary, not an omission |
| `can_schedule` PP dry-run stub | **Covered** (as an explicitly incomplete, self-documented stub — PP unsupported for V2) | scheduler.md §10 (`scheduler_v2.py:1250-1259`) — "flagged as an explicit known-incomplete area by the code itself" |
| Deadlock detection (`RuntimeError` in `_schedule_loop`) | **Covered** (trigger condition) / **Needs native/runtime test** (false-positive conditions under transient manager states, e.g. mid-resume) | scheduler.md §3 (`:456-483`), Open Q9 |
| `BudgetTracker` mechanics (token/request/PEFT, `pre_claim_peft`) | **Covered** | scheduler.md §2 |
| Two-phase scheduling design and PEFT-adapter-eviction-avoidance rationale | **Covered** | scheduler.md §3 (`:316-334`) |
| Environment-gated reordering (`TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN`) | **Covered** | scheduler.md §3 (`:233-235,305-310`) |

---

## 3. Manager Lifecycle

| Item | Status | Evidence |
|---|---|---|
| Create (`_create_kv_cache`) — precondition, saturation handling, native mutation | **Covered** | manager.md §9 (`:4135-4181`) |
| Prepare (`prepare_context`, `prepare_disagg_gen_init`) | **Covered** | manager.md §9 (`:2579-2645,2678-2708`); audit §2 |
| Resize (`resize_context`, `try_allocate_generation`'s internal resize) | **Covered** (Python-wrapper-layer semantics) / **Missing trace** (native `KvCache::resize()`'s exact return-value semantics for the non-no-op-success case) | manager.md §4 (`kvCache.h:200-202`, ambiguous doc comment), §9 |
| Suspend (`suspend_request`) | **Covered** | manager.md §9 (`:2750-2754`) |
| Resume (`resume_request`) — documented failure mode (`max_util_for_resume`) | **Covered** (top-level contract) | manager.md §9 (`:2756-2766`) |
| Resume — per-pool-group rejection rule (does ANY over-threshold pool group block the whole resume, or only touched ones?) | **Missing trace** (a test already exists but its body was not read: `test_resume_rejects_if_any_pool_group_exceeds_threshold`, `test_kv_cache_manager_v2.py:747`) | manager.md §12.4, Open Q11 — this is closable by reading existing source/test, not by writing a new test |
| Free (`free_resources`) — postconditions, idempotency, IndexMapper interaction | **Covered** | manager.md §9 (`:3691-3706`) |
| `pin_on_release` parameter (confirmed unused in method body) | **Covered** (the fact it's unused) / **Missing trace** (whether any caller anywhere in the repo passes `pin_on_release=True` expecting an effect — only two files were grepped) | manager.md §9,§11.5, Open Q2 |
| Revert (`revert_allocate_generation`, `revert_allocate_context`, including the escalation-to-`free_resources` case) | **Covered** | manager.md §9 (`:2493-2547`); audit §2, interface_map.md Confirmed Mismatch #2 |
| Update/commit (`update_resources`, `update_context_resources`, `try_commit_blocks`) — race-tolerant suspended-cache skip, completion short-circuit, draft-reserve reclaim | **Covered** | manager.md §9 (`:3992-4088`), test-corroborated by `test_kv_cache_v2_capacity_only.py` |
| `release_index_slot` (early IndexMapper release for NIXL/UCX handoff) | **Covered** | manager.md §9 (`:3675-3690`) |
| Stats — `get_kv_cache_stats()` (GPU-tier-only scope) | **Covered** | manager.md §10 (`:3314-3343`) |
| Stats — `get_iteration_stats()` (get-and-reset semantics, suspend/resume counts confirmed) | **Covered** (top-level) / **Missing trace** (full field-by-field catalogue of `KVCacheV2IterationStatsReport`/`KVCacheV2PoolGroupIterationStats`, specifically whether host/disk byte/block occupancy is exposed anywhere in it) | manager.md §9b (`:3364-3463`), Open Q10 |
| `shutdown` (teardown ordering, event-manager-not-nulled invariant) | **Covered** | manager.md §9b (`:3935-3945`) |
| `add_dummy_requests` (all-or-nothing rollback behavior) | **Covered** (behavior) / **Missing trace** (actual callers — not found in the two files grepped; likely CUDA-graph warmup code) | manager.md §9 (`:3479-3649`), Open Q5 |
| `probe_prefix_match_length`, `prefetch_for_context_tokens`, `reset_reuse_state` | **Covered** | manager.md §9b (`:4183-4237`) |
| `copy_batch_block_offsets`, `get_batch_cache_indices*` | **Covered** | manager.md §9b (`:3714-3809,4090-4130`) |

---

## 4. Scheduler ↔ Manager Contract

This area is comprehensively addressed by `interface_map.md §1`'s 14-row call-site table, cross-checked against `TRTLLM-15289_audit.md`'s Contract Table. Every scheduler-initiated call into a `KVCacheManagerV2` instance identified across `scheduler_v2.py` is accounted for.

| Item | Status | Evidence |
|---|---|---|
| `prepare_disagg_gen_init` — args, return semantics, state mutation | **Covered** | interface_map.md §1 row 1 |
| `prepare_context`/`resize_context` (both chunked and non-chunked) — args, return semantics, state mutation, retry/rollback | **Covered** (with two named unresolved sub-items: `context_remaining_length` stability, non-first-chunk suspend-on-failure) | interface_map.md §1 rows 2-4; audit §2 |
| Cross-context public-API path | **Covered** | interface_map.md §1 row 5 |
| Cross-context V2 private-member path (`_try_schedule_cross_context_v2`) | **Covered** (as a structural fact — direct source re-read confirms) | interface_map.md §1 row 6, Confirmed Mismatch #1 |
| `try_allocate_generation` — args, return, idempotency-across-retries | **Covered** (Python-wrapper layer) / **Missing trace** (native-level idempotency of a failed `KvCache::resize`) | interface_map.md §1 row 7; manager.md §9b Open Q6 |
| `suspend_request` (+ draft mirror) | **Covered** | interface_map.md §1 row 8 |
| `resume_request` (not scheduler-invoked directly — confirmed via grep) | **Covered** | interface_map.md §1 row 9 |
| `free_resources` (+ draft mirror) | **Covered** | interface_map.md §1 row 10 |
| `is_request_active` | **Covered** | interface_map.md §1 row 11 |
| `revert_allocate_generation`/`revert_allocate_context` | **Covered** | interface_map.md §1 rows 12-13 |
| `can_evict` attribute read | **Covered** | interface_map.md §1 row 14 |
| Exception-vs-bool-return convention across the whole API (which methods raise vs. return `False`) | **Covered** — explicitly tabulated: `revert_allocate_generation`/`revert_allocate_context`/`update_resources`/`update_context_resources` raise on failure; `prepare_context`/`resize_context`/`prepare_disagg_gen_init`/`try_allocate_generation`/`resume_request` return bool | manager.md §9,§9b (assembled across individual method entries) |
| Whether any scheduler call site passes non-default kwargs not otherwise documented (e.g. `pin_on_release`) | **Missing trace** | Only `scheduler_v2.py`/`py_executor.py` were grepped for call sites; a repo-wide grep for specific kwargs was not performed |

---

## 5. Target/Draft Ownership

| Item | Status | Evidence |
|---|---|---|
| Scheduler-mirrored transitions (suspend, free) — exactly which two calls, and only those two | **Covered** | scheduler.md §11; audit §4 (table of mirrored vs. not-mirrored transitions) |
| Executor-driven draft preparation (`prepare_resources` → `_prepare_draft_resources`, `update_resources` draft-reserve reclaim) | **Covered** | manager.md §7 (`:2771-2785,4062-4069`); test-corroborated (`test_kv_cache_v2_capacity_only.py:124-138`) |
| Aggregate path — draft-length numeric equivalence (`get_draft_token_length` vs. `_effective_draft_len`) | **Covered — proven identical** | interface_map.md §2 (re-verified `kv_cache_manager_v2.py:2418-2429`); audit §3 item 3 |
| Disaggregated path — draft-length divergence in the transmission-complete transition window | **Covered — confirmed, scoped** | interface_map.md §2, Confirmed Mismatch #4; audit §3 item 4, §4 |
| Ordering guarantee between the scheduler's per-iteration target-manager admission and the executor's separate, once-per-iteration `prepare_resources` dispatch to `draft_kv_cache_manager` | **Needs native/runtime test** | audit §4 — explicitly named as the most significant open item; no artifact traces `py_executor.py`'s iteration-loop ordering of these two dispatches against each other (would require reading/exercising `py_executor.py`'s main loop in depth, which all three prior audits deliberately scoped out) |
| Native-level independence of target/draft manager objects (separate `impl`, separate `IndexMapper` — confirmed at construction-code level; whether any hidden native coupling exists, e.g. shared CUDA pools/event sink) | **Covered** (construction-code-level independence) / **Missing trace** (native `kvCacheManager.cpp` not read to confirm no hidden coupling) | manager.md §7 closing Open question |
| Sequential, unguarded mirror-call failure (target succeeds, draft raises, or vice versa) | **Needs native/runtime test** | audit §1,§4 — same item as Area 2's self-suspend gap, restated here for the target/draft lens |
| Two-model GPU-budget-parity constraint enforcement | **Covered** | manager.md §7 (`_util.py:2079-2087`) |
| One-model gating conditions (attention-DP, DeepSeek-V4-sparse+pp>1) | **Covered** | manager.md §7 (`_util.py:1432-1455`) |

---

## 6. Budget and Capacity

| Item | Status | Evidence |
|---|---|---|
| `BudgetTracker` token/request accounting | **Covered** | scheduler.md §2 |
| `BudgetTracker` PEFT accounting (`peft_pages_needed`, `commit_peft`, `pre_claim_peft`) | **Covered** | scheduler.md §2 |
| Manager/native page accounting — general mechanism (try/fail, no query API) | **Covered** | interface_map.md §4; audit §3 |
| Draft reserve slack (`_kv_reserve_draft_tokens` padding and reclaim) | **Covered**, with concrete numeric test corroboration | manager.md §7,§12.5 (`test_kv_cache_v2_capacity_only.py:124-138`) |
| GPU tier sizing | **Covered** | manager.md §6.1 |
| Host tier sizing (explicit + auto + fleet-sync) | **Covered** | manager.md §6.2 |
| Disk tier sizing | **Covered** | manager.md §6.3 |
| Six-point divergence enumeration between `BudgetTracker` and manager accounting (token/PEFT domains, draft-length aggregate vs. disagg, draft-reserve slack, `GENERATION_TO_COMPLETE` carve-out) | **Covered** (5 of 6 classified conclusively) | audit §3 table |
| `GENERATION_TO_COMPLETE`-state KV-layer accounting gap (item 6 of the above) | **Needs unit/fault-injection test** | audit §3 item 6 — confirmed via grep that no manager code path special-cases this state; whether that absence is safe was not resolved by source reading alone |
| `get_kv_cache_stats()`/`get_iteration_stats()` host/disk occupancy field coverage | **Missing trace** | manager.md §10, §9b Open Q10 — closable by reading the stats-struct field definitions, not by a new test |
| Whether scheduler admission decisions ever indirectly depend on stats methods (`get_kv_cache_stats`/`get_iteration_stats`) | **Covered — confirmed no dependency** | interface_map.md §4 — no citation in scheduler.md references either stats method from `scheduler_v2.py`; scheduling is purely try/fail-gated |

---

## Cross-Area Summary

| Status | Count of distinct items |
|---|---|
| Covered (fully) | ~55 |
| Covered with a named residual (partial) | 8 |
| Out of scope (deliberate) | 5 |
| Missing source trace (closable by reading, no new test) | 10 |
| Needs unit/fault-injection test | 3 |
| Needs native/runtime test | 6 |

**Conclusion:** The four existing artifacts explain the large majority of KVCacheV2 scheduler/manager behavior with direct `path:line` evidence. No area is uncovered in the sense of "never investigated" — every one of the six requested areas has a substantive, cited base. The remaining gaps cluster into two kinds:
1. **Cheap to close** — ten items are gaps only because a specific already-known source location (a `.cpp` file, an existing test body, a binding module) was not read by the prior passes; these do not require writing any new test.
2. **Structurally require dynamic evidence** — six items (sequential mirror-call failure semantics in both the self-suspend and target/draft contexts, non-first-chunk resize-failure suspension, native `resize()`'s exact return contract, `terminateOnException` coverage, per-pool-group resume rejection precision, and the scheduler/executor draft-dispatch ordering guarantee) cannot be resolved by reading more source alone, because the current source is either silent, ambiguous, or the guarantee in question is about *runtime interleaving* rather than a fixed code path.

---

## Smallest Ordered Test Plan to Close Remaining Gaps

Ordered to do the cheapest, highest-value closures first; source-reading closures (no test authored) precede test-authoring items, since some source-reading closures might make a planned test unnecessary or reshape it.

1. **[Source trace, no test]** Read `test_resume_rejects_if_any_pool_group_exceeds_threshold` (`tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py:747`) — resolves the per-pool-group resume-rejection rule (Area 3) from an existing test that was only named, not read.
2. **[Source trace, no test]** Read `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.cpp` and `kvCache.cpp` for: (a) `resize()`'s exact return-value contract for the non-no-op-success case (Area 3), (b) `terminateOnException` call-site enumeration (Area 1), (c) whether target/draft native objects share any hidden state (Area 5).
3. **[Source trace, no test]** Read `config.cpp`'s `KVCacheManagerConfig::validate()`/`DiskCacheTierConfig::assertValid()` bodies for tier-ordering and sizing validation beyond `quota>0` (Area 1).
4. **[Source trace, no test]** Repo-wide grep (not limited to `scheduler_v2.py`/`py_executor.py`) for `pin_on_release=` and `add_dummy_requests(` call sites (Area 3, Area 4) — resolves whether any caller relies on the currently-dead `pin_on_release` parameter, and identifies `add_dummy_requests`'s actual callers.
5. **[Source trace, no test]** Read the field definitions of `KVCacheV2IterationStatsReport`/`KVCacheV2PoolGroupIterationStats` to catalogue whether host/disk byte/block occupancy is exposed anywhere in the stats surface (Area 3, Area 6).
6. **[Unit/fault-injection test]** Force `resize_context` to fail on the **second** (non-first) chunk of a multi-chunk context request; assert cache status afterward — closes the largest named gap in Area 2/Area 4. Starting point: `test_kv_cache_manager_v2.py:2365-2395`.
7. **[Unit/fault-injection test]** Construct a request in `GENERATION_TO_COMPLETE` state and drive it through `try_allocate_generation`/`update_resources` alongside a competing context request needing the same budget; assert no double-allocation — closes Area 6 item 6. Starting point: `test_kv_cache_v2_capacity_only.py:141-149`.
8. **[Unit/fault-injection test]** Mock `cross_kv_cache_manager.prepare_context`/equivalent to mutate `req.context_remaining_length` mid-call in the chunked-context path; assert whether the scheduler's cached pre-loop value or a freshly-read value drives the subsequent `resize_context` call — closes Area 2's `context_remaining_length` stability item.
9. **[Native/runtime test]** Inject a failure into the **second** of two sequential `suspend_request`/`free_resources` calls in a primary-then-draft mirror pair (both the self-suspend path of Area 2 and the target/draft-divergence framing of Area 5 reduce to this same underlying test); assert whether the exception is catchable in Python or aborts the process (`std::terminate`), and what state the primary manager is left in.
10. **[Native/runtime test]** A disagg-specific runtime test driving a request through `is_disagg_generation_transmission_complete=True` with empty `py_draft_tokens`; quantify the gap between `BudgetTracker`'s committed token count and `try_allocate_generation`'s actual manager-side reservation (Area 5/Area 6) — the direction is already known to be conservative, this closes the magnitude.
11. **[Native/runtime test]** A multi-iteration scheduler+executor integration test asserting the executor's per-iteration `prepare_resources` dispatch to `draft_kv_cache_manager` always covers exactly the request set the scheduler admitted for the primary manager in the same iteration — closes the single highest-value open item in Area 5 (scheduler/executor draft-dispatch ordering guarantee). This is last because it is the most expensive to construct (requires a live scheduler+executor+two-manager harness) and its outcome may be shaped by findings from items 2 and 9.
12. **[Native/runtime test]** Simulate a manager state where `try_allocate_generation` fails while `is_request_active` reports a value that could look like "nothing evictable" from the scheduler's perspective (e.g., a transient mid-resume state); assert the `_schedule_loop` deadlock `RuntimeError` (Area 2) fires only for genuine deadlocks, not transient manager-side states.
