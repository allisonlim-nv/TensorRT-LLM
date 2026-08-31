# KVCacheV2Scheduler — Scheduler-Side Context Audit

**Repo:** /Users/allim/TensorRT-LLM
**Commit:** 4716843cee6e7a6c08bf4d8be29fae25321a9344
**Branch:** feat/native-kv-events-clean
**HEAD date:** 2026-08-31

**Files covered:**
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py` (primary; 1259 lines, read in full)
- `tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py` (read in full)
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` (`SchedulerOutput`, helper functions — read relevant sections)
- `tensorrt_llm/_torch/pyexecutor/_util.py` (construction/wiring — read relevant sections, ~lines 3100-3190)
- `tensorrt_llm/_torch/pyexecutor/py_executor.py` (immediate orchestration caller — read relevant sections)
- `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` — **only the scheduler-facing API surface** was read (method signatures/docstrings under the `# ---- Scheduling API (called by KVCacheV2Scheduler) ----` marker at line 2403, plus a few adjoining methods). Internals are explicitly out of scope per instructions.

Scope note: per task instructions, this document does NOT explain how `KVCacheManagerV2` implements anything internally — only what the scheduler calls, with what arguments, and what it does with return values/side effects it can observe.

---

## 1. Construction / Wiring (`_util.py`)

**Verified fact.** `KVCacheV2Scheduler` is constructed at `tensorrt_llm/_torch/pyexecutor/_util.py:3144-3169`, gated on `isinstance(kv_cache_manager, KVCacheManagerV2)` (line 3144). Constructor args passed: `max_batch_size`, `max_num_tokens`, `kv_cache_manager`, `scheduler_policy` (from `scheduler_config.capacity_scheduler_policy` or defaults to `MAX_UTILIZATION`, lines 3148-3150), `ctx_chunk_config`, `peft_cache_manager=peft_cache_manager.impl` (line 3160-3161), `scheduler_capacity=v2_scheduler_capacity` (line 3162), `draft_kv_cache_manager` (fetched from `resources.get(ResourceManagerType.DRAFT_KV_CACHE_MANAGER)`, lines 3146-3147), `cross_kv_cache_manager`, `no_schedule_until_state` (`ENCODER_INIT` if a cross KV manager is present else `CONTEXT_INIT`, lines 3128-3130), `enable_prefix_aware_scheduling` (from `scheduler_config.enable_prefix_aware_scheduling`, default `True`, lines 3151-3153), and `enable_recompute_pause=not is_disagg` (line 3168, comment: "A disaggregated generation worker must not replay context locally").

**Verified fact.** `v2_scheduler_capacity = max_batch_size` (`_util.py:3140`), with a +1 bump when `v2_scheduler_capacity == 1 and mapping.enable_attention_dp and kv_cache_manager` (lines 3141-3142). This differs from the V1 `scheduler_capacity = max_batch_size * mapping.pp_size` (line 3120) — comment at lines 3132-3139 states V2 merges CapacityScheduler+MicroBatchScheduler into one loop and PP is handled via `inflight_request_ids` filtering instead of an inflated capacity budget.

**Inference.** `KVCacheV2Scheduler` requires the primary `kv_cache_manager` to be `KVCacheManagerV2` (asserted in `scheduler_v2.py:173-175`); `draft_kv_cache_manager` and `cross_kv_cache_manager` are typed as `object | None` in the constructor (`scheduler_v2.py:162-163`) but are treated as `KVCacheManagerV2`-shaped by call sites that invoke `suspend_request`/`free_resources` on them, and `cross_kv_cache_manager` is explicitly `isinstance`-checked against `KVCacheManagerV2` at `scheduler_v2.py:911`.

---

## 2. `BudgetTracker` — token / request / PEFT accounting

Defined at `scheduler_v2.py:51-141`. One instance is created per `_schedule_loop` call (`scheduler_v2.py:284-288`), so it is fresh per scheduling iteration — no cross-iteration state.

**Token/request budget (Verified fact):**
- `max_num_tokens` and `max_num_requests` are set from the scheduler's own fields at construction (`scheduler_v2.py:67-68`); `max_num_requests` is `self.max_num_requests` = `scheduler_capacity` or `max_batch_size` (`scheduler_v2.py:167-170`).
- `requests_full` property: `num_requests >= max_num_requests` (`scheduler_v2.py:82-84`).
- `can_fit_tokens(num_tokens)`: `False` iff `max_num_tokens is not None and (num_tokens_so_far + num_tokens) > max_num_tokens` (`scheduler_v2.py:86-90`) — so `max_num_tokens=None` means unlimited.
- `remaining_tokens` property returns `None` if unlimited, else `max_num_tokens - num_tokens` (`scheduler_v2.py:92-97`).
- `commit(req, num_tokens, peft_pages)`: increments `num_tokens += num_tokens`, `num_requests += 1`, and if `peft_pages > 0` also calls `commit_peft` (`scheduler_v2.py:99-104`).

**PEFT accounting (Verified fact):**
- `_max_peft_pages = peft_cache_manager.max_device_pages if peft_cache_manager is not None else 0` (`scheduler_v2.py:74-76`); `_claimed_peft_pages` and `_seen_peft_task_ids` start at 0/empty (`scheduler_v2.py:77-78`).
- `peft_pages_needed(req)`: returns `0` if no `peft_cache_manager` configured, or if `req.lora_task_id is None` or already in `_seen_peft_task_ids` (dedup — a task only pays once per iteration) (`scheduler_v2.py:118-124`); otherwise calls `self._peft_cache_manager.determine_num_pages(req)` and returns `None` if `_claimed_peft_pages + required > _max_peft_pages` (budget exceeded), else the page count (`scheduler_v2.py:125-128`).
- `commit_peft(req, peft_pages)`: adds to `_claimed_peft_pages` and adds `lora_task_id` to `_seen_peft_task_ids`, WITHOUT touching `num_tokens`/`num_requests` (`scheduler_v2.py:106-114`) — explicitly documented (lines 107-111) as needed for disagg-gen-init requests, which need PEFT accounting but do not participate in the forward pass.
- `pre_claim_peft(req)`: same bookkeeping as `commit_peft` but computed directly via `determine_num_pages` rather than a precomputed `peft_pages` value; used to reserve pages for `GENERATION_TO_COMPLETE` requests whose adapters are still resident (see §3 Phase 1 pre-claim, `scheduler_v2.py:130-141`).

**Call convention (Verified fact):** every dispatch site in `_schedule_loop`/`_try_schedule_*` calls `budget.peft_pages_needed(req)` first; if it returns `None` the loop `break`s (stop scheduling this phase) — see `scheduler_v2.py:363-365` (disagg), `407-409` (generation), `437-439` (phase-2 pending_ctx, uses `continue` not `break`, line 438-439). Only on success is `budget.commit`/`commit_peft` invoked with the already-computed `peft_pages` (no re-query), so `peft_pages_needed` and the later commit must be logically consistent within one request's handling — **Inference**, not literally enforced by code beyond call ordering.

---

## 3. `schedule_request` / `_schedule_loop` — main orchestration entry point

**Purpose and call chain (Verified fact).** `schedule_request(active_requests, inflight_request_ids)` (`scheduler_v2.py:243-270`) is the public `RequestScheduler` entry point. It is called from `py_executor.py` at line 2644 (`self.scheduler.schedule_request(self.active_requests, self.inflight_req_ids)`, PP-follower re-run path) and line 6276 (`self.scheduler.schedule_request(...)`, main call site — not fully read but grepped). It first calls `drop_decoder_context_requests_waiting_for_encoder_output(active_requests)` (`scheduler_v2.py:246`, imported from `scheduler.py`), then `_schedule_loop`, then sorts outputs and returns a `SchedulerOutput` namedtuple (`scheduler.py:66-108`) with fields: `encoder_requests`, `context_requests`, `generation_requests`, `paused_requests`, `fitting_disagg_gen_init_requests`, `num_fitting_requests`, `scheduled_mm_encoder_items` (unused by V2 — defaults `None`), `recompute_paused_requests` (V2-only field, defaults `[]`).

**Inputs/outputs/mutated state (Verified fact).**
- Inputs: `active_requests` (iterable of `LlmRequest`), `inflight_request_ids: set[int]` (requests mid-flight, e.g. from a prior PP stage — these are always skipped, `scheduler_v2.py:345-347`, `1076-1077`, `1085-1086`).
- `_schedule_loop` builds a local `requests_list = list(active_requests)` (`scheduler_v2.py:299`) and iterates by index `req_it` up to a shrinkable `req_it_end` (`scheduler_v2.py:312-314`) so that eviction can shrink the iteration range from the tail without invalidating indices already visited.
- Two-phase design (**Verified fact**, comment `scheduler_v2.py:316-321`): Phase 1 handles disagg-gen-init and generation requests; context/encoder requests are deferred to `pending_ctx` (`scheduler_v2.py:402,444` wait — see line ~402) and processed in Phase 2 (`scheduler_v2.py:432-454`) "so that generation requests are fully accounted for in the budget before any context request competes for resources," explicitly to avoid PEFT adapter eviction failures when gen requests hold adapters that can't be evicted mid-iteration.
- Optional pre-pass (`scheduler_v2.py:332-334`): for every request in `GENERATION_TO_COMPLETE` state, `budget.pre_claim_peft(req)` is called before the main loop — comment (`scheduler_v2.py:324-331`) explains this is needed because in the overlap executor these requests are outside the schedulable range but their PEFT adapters are not yet released (`mark_request_done` runs after `prepare_resources` in the next iteration); without pre-claiming, the budget would look empty and a different-adapter context request could be admitted, crashing `ensure_batch`.
- Optional environment-gated reordering (`scheduler_v2.py:233-235, 305-310`): `TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN=1` stable-sorts `requests_list` to put just-arrived generation-only requests (`py_decoding_iter == 0`) first, to reduce TTFT under budget pressure on a disagg generation server. Default off.

**Return value construction (Verified fact).** `schedule_request` sorts `scheduled_encoder` by LoRA task id (`scheduler_v2.py:259`) and calls `self._sort_requests(scheduled_ctx, scheduled_gen, has_chunking)` (`scheduler_v2.py:260`) which, when chunking occurred, partitions context requests into not-last/last chunk groups, sorts each by LoRA key, and reassembles not-last-then-last (`scheduler_v2.py:1234-1246`) — ensures non-final chunks are scheduled ahead of final chunks within a batch.

**Deadlock detection (Verified fact, failure/rollback-adjacent).** At `scheduler_v2.py:456-483`: if no generation and no context requests were scheduled, the code counts generation candidates still eligible (`is_generation_in_progress_state and not is_generation_to_complete_state and not inflight`). If that count is `>0` and nothing was evicted, recompute-paused, or inflight, it raises `RuntimeError` — "V2 scheduler deadlock ... KV cache pool is likely exhausted with no secondary cache tier for suspend/resume offload," suggesting `kv_cache_config.host_cache_size`/`disk_cache_size`/`max_tokens` config changes. This is a hard crash, not a soft retry.

**Consumption by py_executor.py (Verified fact, immediate caller).** The `SchedulerOutput` is copied into a `ScheduledRequests` object: `scheduled_requests.paused_requests = scheduler_output.paused_requests` (`py_executor.py:6328`) and `scheduled_requests.recompute_paused_requests = scheduler_output.recompute_paused_requests` (`py_executor.py:6331`). Downstream, `paused_requests` feed `_pause_requests`/`_terminate_requests` (`py_executor.py:4238-4239, 5064, 5228`) and `recompute_paused_requests` feed `_terminate_recompute_paused_requests`/`_pause_recompute_paused_requests` (`py_executor.py:2659-2660, 4219-4221, 4235-4236, 5050-5052, 5062`), which call `req.pause(...)` / `req.reset_for_recompute(...)` (`py_executor.py:8556-8561`) or route through a disagg PP termination handler (`py_executor.py:8563-8577`).

---

## 4. `_try_schedule_disagg_gen_init` (Phase 1) — `prepare_disagg_gen_init` entry point

**Purpose and call chain (Verified fact).** Invoked from `_schedule_loop` Phase 1 when `req_state_value == self._disagg_gen_init_state_value` (`scheduler_v2.py:362-382`). Per the comment at `scheduler_v2.py:351-361`: disagg-gen-init requests bypass both state-range gating and `budget.requests_full`, matching the C++/V1 scheduler, but V2 owns inline KV allocation (V1 defers to `prepareResources`; V2's `prepare_resources` is a no-op for the primary manager) so allocation must happen here.

**Inputs/outputs (Verified fact).** `_try_schedule_disagg_gen_init(req, budget) -> (ScheduleAction, int)` (`scheduler_v2.py:526-543`). Calls `self.kv_cache_manager.prepare_disagg_gen_init(req)` (`scheduler_v2.py:540`); on `False`, logs a debug message and returns `(SKIP, 0)`; on `True` returns `(SCHEDULED, 0)` — tokens are always `0` since disagg requests don't enter the forward-pass token budget (comment `scheduler_v2.py:537-539`).

**Budget interaction (Verified fact, `scheduler_v2.py:362-382`).** `peft_pages = budget.peft_pages_needed(req)`; if `None`, `break` (stop Phase 1 entirely). On `SCHEDULED`, `req` is appended to `disagg_candidates`, and if `peft_pages > 0`, `budget.commit_peft(req, peft_pages)` is called — explicitly NOT `budget.commit` (comment `scheduler_v2.py:374-378`): disagg requests must not count toward `num_requests`/`num_tokens` because that would steal batch slots and delay KV transfer initiation; capacity is gated by "IndexMapper slot availability" via `prepare_context` returning `False` when no free slots remain (comment references `prepare_context`, but the actual call here is `prepare_disagg_gen_init`, which per the manager docstring internally calls `_prepare_context_impl` — **Inference**, based on comment naming and the manager's own docstring at `kv_cache_manager_v2.py:2678` "Allocates capacity for the full prompt (+ draft) and sets `kv_cache.history_length` to `prompt_len`. Returns True on success, False if preparation or resize failed (cache is suspended on resize failure)").

**Manager call surface used (Verified fact, scheduler-visible only).** `prepare_disagg_gen_init(req: LlmRequest) -> bool` — single call, no other manager interaction in this path from the scheduler.

**Invariant the scheduler assumes (Inference).** A `False` return from `prepare_disagg_gen_init` must be safely retryable next iteration without having partially mutated `req`'s allocation state — the scheduler treats it purely as SKIP-and-retry (`req_it += 1; continue`, `scheduler_v2.py:381-382`), with no rollback call of its own. If the manager left a half-allocated cache on failure, repeated retries could leak; **Open question** (below).

**Failure/rollback (Verified fact + Open question).** On failure the request is simply skipped for this iteration (loop continues) — no explicit scheduler-side cleanup call. Downstream, `py_executor.py:_prepare_disagg_gen_init` (line 7206-7234) drives `prepare_resources` for the KV/spec/draft-KV managers over `fitting_disagg_gen_init_requests` (i.e., only requests that DID succeed in `_try_schedule_disagg_gen_init`) and then `_recv_disagg_gen_cache(...)`. A companion path, `_revert_deferred_disagg_gen_init_alloc` (`py_executor.py:3573-3588`), calls `self._revert_ctx_alloc(deferred_requests)` → `self.kv_cache_manager.revert_allocate_context(req)` (`py_executor.py:3476-3477`) for disagg candidates that were scheduler-admitted but NOT ultimately admitted by a later admission-control step (`admission_result.admitted_requests`) — this reverts the *context-style* capacity growth (`py_ctx_pre_resize_cap`), confirming that `prepare_disagg_gen_init`'s allocation is undone via the same `revert_allocate_context` mechanism as ordinary context requests when a downstream admission gate rejects the request after the scheduler already admitted it.

---

## 5. `_try_schedule_context` → `_try_schedule_context_full` / `_try_schedule_context_chunked` — `prepare_context` / `resize_context` entry points

**Purpose and call chain (Verified fact).** Dispatched from Phase 2's `pending_ctx` loop (`scheduler_v2.py:434-454`) via `_try_schedule_context(req, budget)` (`scheduler_v2.py:545-555`), which routes to `_try_schedule_context_chunked` if `self.chunking_enabled` else `_try_schedule_context_full`.

### 5a. `_try_schedule_context_full` (`scheduler_v2.py:557-604`)

**Inputs/outputs (Verified fact).** Returns `(ScheduleAction, tokens, chunking_flag=False)`. Reads `req.context_remaining_length` and `get_draft_token_length(req)` (`scheduler_v2.py:564-565`).
- If `enable_prefix_aware_scheduling` is `False`: does a pre-check `can_fit_tokens(pre_prepare_context_tokens + draft_len)` before calling into the manager, returning `STOP` if it doesn't fit (`scheduler_v2.py:567-575`).
- Always calls `self.kv_cache_manager.prepare_context(req)` (`scheduler_v2.py:579`); on `False`, returns `STOP` (not SKIP) with a debug log (`scheduler_v2.py:580-581`).
- Comment at `scheduler_v2.py:577-578`: "Prepare first so block reuse updates `context_remaining_length` before budget check" — i.e. the scheduler explicitly relies on the manager's `prepare_context` call to MUTATE `req.context_remaining_length` as a side effect (block-reuse-aware trimming) when `enable_prefix_aware_scheduling` is `True` (re-read at line 583, re-checked against budget at 584-592).
- Calls `self.kv_cache_manager.resize_context(req, context_tokens + draft_len)` (`scheduler_v2.py:596`) — comment: "V2 resizes KV cache directly in the scheduler (no separate `prepareResources` for main cache), so include draft tokens." On `False`, returns `(SKIP, 0, False)` (not STOP — the request may become schedulable in a future iteration once more capacity frees up).
- Then calls `self._try_schedule_cross_context(req)` (`scheduler_v2.py:599`); if the result is not `SCHEDULED`, the scheduler calls `self._suspend_request(req)` (undoing what was just prepared/resized) and returns that action with `(0, False)` (`scheduler_v2.py:600-602`).

**Manager call surface used (Verified fact).** `prepare_context(req) -> bool`, `resize_context(req, num_tokens) -> bool`. Per the manager's own docstrings (read for API-surface purposes): `prepare_context` "Create `_KVCache`, handle block reuse, and resume. Does NOT resize... Returns True on success, False if preparation failed" (`kv_cache_manager_v2.py:2579-2585`); `resize_context` "Resize KV cache to cover `context_current_position + num_tokens`. Returns True on success, False if resize failed (first chunk is suspended on failure)" (`kv_cache_manager_v2.py:2646-2650`).

**Invariants the scheduler assumes (Inference, drawn from manager docstrings it reads/relies on but does not implement):**
1. After `prepare_context` succeeds, `req.context_remaining_length` reflects any block-reuse trimming so the subsequent budget check is accurate (`scheduler_v2.py:577-578, 583-592`).
2. `resize_context` failing on a first chunk leaves the cache suspended per the manager's own docstring (`kv_cache_manager_v2.py:2649-2650`, "first chunk is suspended on failure") — the scheduler does NOT itself call `suspend` in this failure branch (`scheduler_v2.py:596-597` just returns SKIP), so it is relying on the manager to have already put the cache into a consistent (suspended) state on `resize_context` failure. **This is an assumption the scheduler-side code does not verify — flagged as an open question below.**
3. `resize_context` sets `req.py_ctx_pre_resize_cap` when growth occurred (per manager docstring reference and its use in `revert_allocate_context`, `kv_cache_manager_v2.py:2665-2675` region) — this field is read later by `py_executor.py`'s `_revert_ctx_alloc` (`py_executor.py:3476-3477`), which the scheduler itself never calls directly; it's an orchestration-caller-side rollback for context requests dropped after scheduling but before the forward pass (e.g., PP-follower mismatch or attention-DP `can_queue=False`... **Inference**, not fully traced in this scheduler-only audit — see §9).

### 5b. `_try_schedule_context_chunked` (`scheduler_v2.py:606-708`)

**Purpose (Verified fact).** "FCFS interleaved chunking for a single context request." Docstring notes chunking uses implicit SKIP on failure (not STOP/break) so subsequent generation requests (needing far fewer tokens) can still be scheduled this iteration (`scheduler_v2.py:612-616`).

**Flow (Verified fact):**
1. Early-out on remaining budget: if `remaining_budget <= 0` (`no_budget`) or (non-force-chunk policy and `remaining_budget < chunk_unit_size`, i.e. `fcfs_under_min`), return `(SKIP, 0, False)` before touching the manager at all (`scheduler_v2.py:617-625`).
2. `self.kv_cache_manager.prepare_context(req)` (`scheduler_v2.py:628`) — same call as the non-chunked path, same block-reuse-mutates-`context_remaining_length` contract; on `False`, `SKIP` (`scheduler_v2.py:629-630`).
3. Computes `chunk_size` from remaining budget, `context_remaining` (post block-reuse), `max_context_length`, `chunk_unit_size`, and (if `FORCE_CHUNK` policy) forced-chunk-boundary helpers `_get_forced_context_chunk_size`/`_is_forced_context_chunk_boundary` imported from `scheduler.py` (`scheduler_v2.py:632-665`; these helper internals are outside V2-scheduler scope but are called from here).
4. `_align_chunk_to_mm_block(...)` (`scheduler_v2.py:674-676`, method at 710-871) further clips/snaps `chunk_size` to avoid splitting a bidirectional-multimodal block across a chunk boundary; can return `0` to defer (SKIP) or raise `ValueError` if a single MM block is unconditionally larger than `max_context_length` (livelock-prevention, `scheduler_v2.py:813-825`). This method reads `req.py_multimodal_data` fields (`mm_bidirectional_blocks`, `multimodal_embed_mask_cumsum`) — no manager interaction.
5. Sets `req.context_chunk_size = min(chunk_size, context_remaining)` (`scheduler_v2.py:685`), computes `chunk_tokens`/`resize_tokens`, adding `draft_len` only "for last chunk" (`scheduler_v2.py:687-694`).
6. Calls `self.kv_cache_manager.resize_context(req, resize_tokens)` (`scheduler_v2.py:698`); `False` → `SKIP` (`scheduler_v2.py:699`).
7. `_try_schedule_cross_context(req)` — same as non-chunked path; failure triggers `_suspend_request(req)` then returns that action (`scheduler_v2.py:701-704`).
8. `chunking_flag = req.context_chunk_size < req.context_remaining_length` (`scheduler_v2.py:706`) — signals to `_schedule_loop`/`_sort_requests` that this request is a non-final chunk.

**Mutated request state (Verified fact).** `req.context_chunk_size` is set directly by the scheduler (`scheduler_v2.py:685`) — this is scheduler-owned state, not manager-owned, consumed later by the forward pass / `is_last_context_chunk` checks.

**Manager call surface used (Verified fact).** Same two calls as the non-chunked path: `prepare_context(req)`, `resize_context(req, resize_tokens)`.

**Invariant (Inference).** The scheduler assumes `context_remaining_length` (post-`prepare_context`) is stable/consistent across the chunk-size computation steps between the `prepare_context` call (step 2) and the `resize_context` call (step 6) — nothing in the scheduler re-queries the manager between these; it relies purely on the value cached from step 2 (`context_remaining = req.context_remaining_length` at `scheduler_v2.py:634`).

**Open question.** Does `prepare_context` on a non-first chunk ever change `context_remaining_length` (e.g. due to concurrent state changes), or is it purely a resume/liveness check for chunk ≥ 2? The scheduler-side code path (`_try_schedule_context_chunked`) doesn't distinguish first vs. non-first chunk handling explicitly other than relying on `prepare_context`'s return value — see §10.

---

## 6. `_try_schedule_cross_context` / `_try_schedule_cross_context_v2` — cross-KV reservation

**Purpose (Verified fact).** Reserves cross-attention KV blocks for the first decoder-context step of encoder-decoder models (`scheduler_v2.py:893-926`). Called from both context paths (§5a/§5b) right after `resize_context` succeeds.

**Flow (Verified fact).**
- `_needs_cross_context_allocation(req)` (`scheduler_v2.py:886-891`) is `True` only if `_get_optional_encoder_output_len(req)` is not `None` and `req.py_skip_cross_kv_projection` is not `True`. If not needed, immediately returns `SCHEDULED` (no manager call) (`scheduler_v2.py:895-896`).
- If needed but `self.cross_kv_cache_manager is None`, logs a warning and returns `STOP` (`scheduler_v2.py:898-904`).
- If `cross_kv_cache_manager` is a `KVCacheManagerV2` instance (checked via local import, `scheduler_v2.py:909-916`), calls the scheduler's own helper `_try_schedule_cross_context_v2(cross_kv_cache_manager, req, req_tokens)` (static method, `scheduler_v2.py:928-963`) which — **this is scheduler-owned orchestration logic that reaches directly into manager internals** (`kv_cache_map`, `_create_kv_cache`, `_resume_and_restore`, `.resize(...)`, `.suspend()`, `.stop_committing()`, `num_extra_kv_tokens`) rather than calling `prepare_context`/`resize_context`. This is flagged explicitly because it breaks the "only call the public scheduling API" pattern used everywhere else in this file — **Verified fact** of what the code does, but its correctness relative to the manager's contract is an **Open question** (the manager audit should confirm whether these private members are an intended scheduler-manager contract or an internal leak).
- Otherwise (non-V2 cross manager, e.g. legacy V1-style object), calls the public `cross_kv_cache_manager.prepare_context(req)` then `.resize_context(req, req_tokens)` (`scheduler_v2.py:918-926`) — the "normal" pattern.

**Failure/rollback (Verified fact).** Both context-scheduling callers (`_try_schedule_context_full:600-602`, `_try_schedule_context_chunked:701-704`) react to a non-`SCHEDULED` cross-context result by calling `self._suspend_request(req)` (which suspends BOTH `kv_cache_manager` and `draft_kv_cache_manager`, see §8) even though only the *primary* context resize had already succeeded — i.e., a cross-KV reservation failure unwinds the primary-pool allocation too, via suspend (not full free).

---

## 7. `_try_schedule_generation` — `try_allocate_generation` entry point

**Purpose and call chain (Verified fact).** Called from Phase 1 for non-context/non-disagg/non-encoder requests (`scheduler_v2.py:406-421`). Signature: `_try_schedule_generation(req, budget, requests_list, req_it, req_it_end, recompute_pause_state, evicted, recompute_paused, inflight_request_ids, scheduled_beam_width) -> (ScheduleAction, tokens, scheduled_beam_width, req_it_end)` (`scheduler_v2.py:965-1040`).

**Flow (Verified fact):**
1. `beam_width = req.get_beam_width_by_iter(for_next_iteration=False)`; `req_tokens = beam_width + get_draft_token_length(req)` (`scheduler_v2.py:983-984`).
2. `budget.can_fit_tokens(req_tokens)` — `False` → `STOP` (`scheduler_v2.py:986-987`).
3. Beam-width consistency: the FIRST scheduled generation request in the iteration sets `scheduled_beam_width`; any subsequent request with a different beam width is `SKIP`ped (`scheduler_v2.py:989-992`) — i.e., **all generation requests scheduled in one iteration must share the same beam width**.
4. `success = self.kv_cache_manager.try_allocate_generation(req)` (`scheduler_v2.py:994`).
5. On failure, if `self.has_cp_helix` (Helix context-parallel mode): raises `RuntimeError` immediately — "No-evict stance: every validated helix run used GUARANTEED_NO_EVICT semantics; eviction is disabled under helix" (`scheduler_v2.py:997-1008`). No retry.
6. Otherwise, on failure, tries `_try_evict_for_gen` (§ below) (`scheduler_v2.py:1009-1011`), then if still failing, `_try_recompute_pause_for_gen` (`scheduler_v2.py:1013-1023`).
7. On eventual success: `SCHEDULED, req_tokens, scheduled_beam_width, req_it_end` (`scheduler_v2.py:1025-1026`).
8. On eventual failure: "self-eviction" — if `self.kv_cache_manager.is_request_active(req.py_request_id)`, calls `self._suspend_request(req)` and appends `req` to `evicted` (`scheduler_v2.py:1028-1039`), then returns `STOP` (`scheduler_v2.py:1040`) — this stops the ENTIRE Phase-1 loop (not just this request), since a self-eviction means the current request itself couldn't be housed even after evicting others.

**Manager call surface used (Verified fact).** `try_allocate_generation(req) -> bool` and `is_request_active(request_id) -> bool`. Per manager docstrings: `try_allocate_generation` "Try to allocate one additional KV cache slot for a generation request. Resumes from suspended state if needed, then resizes capacity by 1 (+ draft tokens). Returns True on success, False if allocation failed" (`kv_cache_manager_v2.py:2465-2470`); `is_request_active` "Return True if `request_id` has a live, non-suspended KV cache" (`kv_cache_manager_v2.py:2405-2408`).

**Invariant the scheduler assumes (Inference).** `try_allocate_generation` is idempotent/retryable — the scheduler calls it repeatedly (once initially, then again after each eviction/recompute-pause attempt in `_try_evict_for_gen`/`_try_recompute_pause_for_gen`) without any explicit "undo the previous failed attempt" call between retries. This implies the manager either leaves no partial state on a failed `resize`, or leaves state that a subsequent successful `resize` call simply supersedes. **Open question** for the manager-side audit.

**Failure/rollback (Verified fact).** Note `try_allocate_generation`'s growth (on success) is NOT rolled back by the scheduler itself even if the overall iteration later fails to build a non-empty batch (see `_schedule_loop` deadlock check, §3) — rollback of successful `try_allocate_generation` calls is handled entirely by the *orchestration caller* (`py_executor.py`) via `revert_allocate_generation`, called from `_revert_gen_alloc` (`py_executor.py:3417-3438`, used when attention-DP `can_queue=False`) and from two additional sites gated on `_check_benchmark_disagg_gate` retries (`py_executor.py:4212-4218`, `~5048`). The scheduler itself never calls `revert_allocate_generation`.

---

## 8. Eviction and recompute-pause — `_try_evict_for_gen`, `_try_recompute_pause_for_gen`, `_suspend_request`, `_recompute_pause_request`

### 8a. `_is_started_request` / `_suspend_request` / `_clear_request_runtime_state` (`scheduler_v2.py:1044-1101`)

**Verified fact.**
- `_is_started_request(req)`: `(req.is_context_init_state and not req.is_first_context_chunk) or req.is_generation_in_progress_state` (`scheduler_v2.py:1044-1052`) — comment: "Matches V1."
- `_suspend_request(req)`: calls `self._clear_request_runtime_state(req)` (sets `req.py_batch_idx = None`, `scheduler_v2.py:1067-1068`), then `self.kv_cache_manager.suspend_request(req)` (`scheduler_v2.py:1063`), then, IF `self.draft_kv_cache_manager is not None`, also `self.draft_kv_cache_manager.suspend_request(req)` (`scheduler_v2.py:1064-1065`). A `TODO` comment (`scheduler_v2.py:1056-1061`) states PEFT resources are NOT released here (`mark_request_done` not called), so the adapter remains "active" on device — flagged by the code itself as a known gap that "could cause `ensure_batch` to fail if it needs to load a different adapter into a full cache."
- `_is_evictable(req, inflight_request_ids)`: `False` if in-flight; `False` if not `_is_started_request`; else `self.kv_cache_manager.is_request_active(req.py_request_id)` (`scheduler_v2.py:1070-1080`) — comment: already-suspended requests are not useful victims since re-suspending frees no pages.

### 8b. `_try_evict_for_gen` (`scheduler_v2.py:1104-1143`)

**Purpose (Verified fact).** "Evict started requests from `active_requests` tail to make room." Searches backward from `req_it_end` (exclusive of `req_it`, i.e., only requests not yet processed this loop) for the first evictable index, suspends it, shrinks `req_it_end` to that victim's index, and retries `try_allocate_generation(req)`. Loops until either allocation succeeds or no evictable victim remains. Returns `(new_req_it_end, success)` — `req_it_end` is always updated even on eventual failure, "so the caller can skip already-evicted requests" (`scheduler_v2.py:1117-1119`).

**Manager call surface used (Verified fact).** `self.kv_cache_manager.is_request_active` (via `_is_evictable`), `self._suspend_request(victim)` → `kv_cache_manager.suspend_request` (+ draft mirror), `self.kv_cache_manager.try_allocate_generation(req)` (retry).

**Invariant assumed (Verified fact, stated in code comment `scheduler_v2.py:1112-1116`).** "Victims are always at indices >= req_it (not yet processed by the main loop), so they are never in scheduled_ctx/scheduled_gen and no token budget reclaim is needed" — i.e., the scheduler assumes suspending a not-yet-scheduled request never needs a budget rollback, because such a request never had budget committed for it in the first place.

### 8c. `_is_recompute_pause_candidate` / `_recompute_pause_request` / `_try_recompute_pause_for_gen` (`scheduler_v2.py:1082-1226`)

**Purpose (Verified fact).** A more "destructive" fallback used when ordinary suspend-eviction (§8b) is insufficient. Per docstring (`scheduler_v2.py:1156-1163`): "recompute frontier is independent from the ordinary-eviction frontier," keeps previously-suspended started requests visible even when ordinary eviction skips over them.

**Gating (Verified fact).** No-op (`return req_it_end, False`) if `not self.enable_recompute_pause` (`scheduler_v2.py:1164-1165`) — this flag is `False` on disaggregated generation workers per construction wiring (`_util.py:3168`, "A disaggregated generation worker must not replay context locally").

**Candidate filter `_is_recompute_pause_candidate` (Verified fact, `scheduler_v2.py:1082-1096`):** excludes in-flight requests; excludes `GENERATION_TO_COMPLETE` state (still finalizing); excludes requests with `py_multimodal_data is not None` while in a generation-in-progress state — comment: "Completed multimodal prefill deliberately releases the inputs and embedding needed for replay, leaving an empty or MRoPE-only dict as the durable marker. Partial-context requests still retain replay data" — i.e., recompute-pause cannot safely be applied to a request whose MM replay inputs have already been discarded. Otherwise requires `_is_started_request`.

**Victim search (Verified fact, `scheduler_v2.py:1167-1197`):** first searches `requests_list[req_it+1 : recompute_pause_state.frontier]` for an eligible, not-already-recompute-paused candidate (skips indices in `recompute_pause_state.victim_indices`), preferring candidates not already ordinarily-evicted UNLESS `self.kv_cache_manager.can_evict` is `True` (in which case even already-evicted requests are eligible again — apparently to double-tap them into full recompute teardown). If no fresh victim found and `can_evict` is `True`, falls back to re-examining the `evicted` list itself for a recompute-pause-eligible entry.

**Manager call surface used (Verified fact).** `self.kv_cache_manager.can_evict` (boolean attribute, read directly — NOT a method call) is checked repeatedly (`scheduler_v2.py:1175, 1180, 1218`) to gate both victim reuse and a compound retry (line 1218: after a recompute-pause, if `try_allocate_generation` still fails AND `can_evict`, it also calls `_try_evict_for_gen` again). `_recompute_pause_request(victim)` calls `self.kv_cache_manager.free_resources(req)` (full teardown, not suspend) and, if present, `self.draft_kv_cache_manager.free_resources(req)` (`scheduler_v2.py:1098-1102`).

**can_evict semantics (Open question — scheduler treats it as a manager-owned capability flag).** The scheduler reads `kv_cache_manager.can_evict` as a plain attribute (not called as a method), implying it is manager-side static/cheap-to-read state. From what the manager-visible surface shows (grepped, not deeply read): `self.can_evict = len(config.cache_tiers) > 1` (manager internals, out of scope) — but the scheduler-side code does not know this; it merely trusts the flag's truth value to mean "further destructive frees can reclaim capacity beyond ordinary suspend."

**Frontier bookkeeping (Verified fact, `scheduler_v2.py:1205-1211`).** After pausing a victim: pop it from `evicted` if present (`1199-1203`), append to `recompute_paused`, add its index to `recompute_pause_state.victim_indices`, and shrink `recompute_pause_state.frontier = min(frontier, victim_idx)`. Comment (`_RecomputePauseState` class, `scheduler_v2.py:43-48`) documents this as "Candidate frontier and exact victims for one scheduling iteration" — `victim_indices` prevents the main loop (`scheduler_v2.py:340-342`) from revisiting destructively-freed requests within the same `_schedule_loop` call.

**Retry (Verified fact, `scheduler_v2.py:1213-1224`).** After each recompute-pause, immediately retries `try_allocate_generation(req)`; if still failing and `can_evict`, also invokes `_try_evict_for_gen` (ordinary suspend) once more — comment explains this handles the case where "full teardown can make the allocation fit even for an already-suspended victim... use any secondary-tier capacity just released to ordinary-suspend another active victim." Loops (`while True`) until success or no further victim is found.

---

## 9. `_align_chunk_to_mm_block` — multimodal chunk boundary safety (context-chunking helper)

Documented in §5b step 4 above. **Verified fact, no manager interaction** — purely reads `req.py_multimodal_data` (a dict written by the input processor per the code comment `scheduler_v2.py:743-746`, `inputs/registry.py`) and computes chunk boundaries so a bidirectional MM block never straddles a chunk. Raises `ValueError` (config error, not silent failure) if a single MM block unconditionally exceeds `max_context_length` (`scheduler_v2.py:813-825`). Not part of the manager contract, but its output (`chunk_size`) directly feeds the `resize_context` call in `_try_schedule_context_chunked`, so its correctness is load-bearing for what capacity the manager is asked to reserve.

---

## 10. `can_schedule` — PP dry-run stub

**Verified fact (`scheduler_v2.py:1250-1259`).** Always returns `True`. Docstring: "V2's try-and-see model lacks a free-blocks query API. Implementing this requires exposing storage statistics from the V2 runtime... For now, always returns True — PP is not yet supported (asserted in KVCacheManagerV2 constructor via `kv_connector_manager=None`)." This means `_pp_retry_until_can_schedule` in `py_executor.py` (referenced at `py_executor.py:2638`, not deeply read — out of scope for scheduler-only audit) is currently a no-op gate for V2 users; **flagged as an explicit known-incomplete area by the code itself, not a defect found by this audit.**

---

## 11. Target/draft behavior in speculative decoding

**Verified fact — scheduler-visible surface only.**
- The scheduler consumes draft-token length uniformly via `get_draft_token_length(req)` (imported from `..llm_request`, `scheduler_v2.py:23`; defined at `llm_request.py:1693-1703`: returns `len(request.py_draft_tokens)` if not `None`, else `0`). This is added to the token-budget request in THREE places: non-chunked context (`scheduler_v2.py:565,567,585`), chunked context's last chunk only (`scheduler_v2.py:687-694`), and generation requests (`scheduler_v2.py:984`).
- A separate `draft_kv_cache_manager` (constructor param, comment: "KVCacheManagerV2 for MTP draft layers", `scheduler_v2.py:162`) is stored but the scheduler ONLY ever calls two methods on it, always mirroring a call already made on the primary `kv_cache_manager`:
  - `self.draft_kv_cache_manager.suspend_request(req)` inside `_suspend_request` (`scheduler_v2.py:1064-1065`), mirroring `kv_cache_manager.suspend_request(req)`.
  - `self.draft_kv_cache_manager.free_resources(req)` inside `_recompute_pause_request` (`scheduler_v2.py:1101-1102`), mirroring `kv_cache_manager.free_resources(req)`.
  - The scheduler NEVER calls `try_allocate_generation`, `prepare_context`, or `resize_context` on `draft_kv_cache_manager` directly — capacity/budget accounting for draft tokens is folded into the PRIMARY manager's calls via `get_draft_token_length` additions (see above), not via separate draft-manager scheduling calls.
- **Inference:** this implies the draft KV cache manager's own capacity growth/shrink (for MTP draft layers) is driven elsewhere (likely `resource_manager.prepare_resources` / the model engine, called at `py_executor.py:2715, 4261, 5094` — grepped but not read in this audit since it's outside scheduler scope) rather than by `KVCacheV2Scheduler` itself; the scheduler's role for the draft manager is limited to keeping its suspend/free lifecycle synchronized with the primary manager so the two never diverge (e.g., primary suspended but draft still resident).
- Disaggregated path: `_prepare_disagg_gen_init` in `py_executor.py` (§4 above, `py_executor.py:7206-7234`) explicitly iterates `(KV_CACHE_MANAGER, SPEC_RESOURCE_MANAGER, DRAFT_KV_CACHE_MANAGER)` resource types and calls `prepare_resources` on each for `fitting_disagg_gen_init_requests` — this is the disagg-specific analog of the aggregate path's draft-manager lifecycle, run entirely by the orchestration caller, not the scheduler class itself.
- `_effective_draft_len` (manager-side, `kv_cache_manager_v2.py:2409-2420` — read only for context, not analyzed in depth) is referenced in a scheduler-adjacent comment chain (`kv_cache_manager_v2.py:2465-2470` `try_allocate_generation` docstring: "Resumes from suspended state if needed, then resizes capacity by 1 (+ draft tokens)") confirming the manager, not the scheduler, decides the exact draft-token reservation size during `try_allocate_generation`; the scheduler's own `req_tokens = beam_width + get_draft_token_length(req)` (`scheduler_v2.py:984`) is used ONLY for the scheduler's own token-budget bookkeeping (`BudgetTracker`), a logically separate quantity from whatever the manager internally reserves in KV pages. **Open question:** whether these two draft-length computations (scheduler's `get_draft_token_length` vs. manager's `_effective_draft_len`) are always numerically identical, or can diverge (e.g., during the context→generation transition, the manager docstring at `kv_cache_manager_v2.py:2409-2415` — grepped snippet — mentions "During the context-to-generation transition, `py_draft_tokens` is still empty when the scheduler asks for capacity. Prefer transferred context draft tokens; if there are none, reserve the generation worker's configured draft capacity before its first forward pass" — i.e., the manager may reserve MORE than the scheduler's token-budget math accounts for in this specific transition window).

---

## Open Questions for Manager-Side Reconciliation

1. **(§4) `prepare_disagg_gen_init` failure state.** When `prepare_disagg_gen_init` returns `False`, does it leave the request's KV cache in a safe, retryable (or absent) state, or can repeated scheduler retries (simple SKIP-and-loop, no explicit rollback call) leak partial allocations across iterations?

2. **(§5a) `resize_context` failure on first chunk.** The scheduler relies on the manager's documented behavior ("first chunk is suspended on failure") without itself calling any explicit rollback in that branch. Confirm the manager always leaves a consistent (suspended, not partially-resized) state when `resize_context` fails on `req.is_first_context_chunk`. Does the same guarantee hold for non-first (subsequent) chunks, where the docstring is silent about suspension on failure?

3. **(§5b) `context_remaining_length` stability within one `_try_schedule_context_chunked` call.** Is `req.context_remaining_length` guaranteed stable between the `prepare_context` call (which may mutate it via block reuse) and the later `resize_context` call within the same scheduler invocation, with no other actor able to change it in between?

4. **(§6) `_try_schedule_cross_context_v2` reaching into manager internals.** This scheduler method directly touches `cross_kv_cache_manager.kv_cache_map`, `._create_kv_cache(...)`, `._resume_and_restore(...)`, `.num_extra_kv_tokens`, and per-`_KVCache` object methods (`.resize`, `.suspend`, `.stop_committing`, `.cuda_stream`) rather than going through the public `prepare_context`/`resize_context` API used elsewhere. Is this an intentional, stable, documented extension point of the manager's contract for cross-KV, or an internal-API leak that the manager audit should flag as fragile/should-be-refactored?

5. **(§7) `try_allocate_generation` retry idempotency.** The scheduler calls `try_allocate_generation` repeatedly (initial attempt, then again after each `_try_evict_for_gen` victim, then again after each `_try_recompute_pause_for_gen` victim) with no explicit "undo prior failed attempt" step between retries. Does a failed `resize` inside `try_allocate_generation` ever leave partial/inconsistent state that a subsequent call could compound, or is each call fully idempotent relative to the previous failed one?

6. **(§8c) `can_evict` semantics.** The scheduler treats `kv_cache_manager.can_evict` as a cheap boolean attribute read (not a method) that gates whether recompute-paused/evicted victims can be reused, and whether a compound (recompute-pause + ordinary-evict) retry is attempted. Confirm this attribute is safe to read at arbitrary points during scheduling (no staleness/consistency requirement) and confirm its intended meaning matches the scheduler's usage ("further destructive frees can reclaim capacity beyond ordinary suspend").

7. **(§7/§4) `revert_allocate_generation` / `revert_allocate_context` correctness contract.** These are called exclusively by `py_executor.py` (the orchestration caller), never by the scheduler itself, to undo capacity growth from `try_allocate_generation`/`resize_context`+`prepare_disagg_gen_init` when a batch is skipped post-scheduling (attention-DP `can_queue=False`, benchmark-disagg retry gate, or disagg admission-control rejection). Does the manager guarantee that `revert_allocate_generation`/`revert_allocate_context` exactly undoes the corresponding forward call regardless of how many scheduling iterations have elapsed in between (e.g., if `try_allocate_generation` succeeded, then several other iterations touched the same request before revert is called)? The scheduler-side code gives no indication of ordering guarantees here.

8. **(§11) Draft-length accounting divergence.** Does the scheduler's `get_draft_token_length(req)` (based on `len(req.py_draft_tokens)`) ever diverge from the manager's internal `_effective_draft_len(req)` (which, per its docstring, may substitute "the generation worker's configured draft capacity" during the context→generation transition when `py_draft_tokens` is still empty)? If so, could the scheduler's `BudgetTracker` under-reserve token budget relative to what the manager actually allocates in KV pages during that transition window?

9. **(§3) Deadlock-detection false positives.** The `RuntimeError` deadlock check in `_schedule_loop` (`scheduler_v2.py:456-483`) fires when no gen/ctx were scheduled, nothing evicted/recompute-paused, and no in-flight requests. Are there legitimate manager-side states (e.g., all capacity held by a secondary tier mid-resume, or an async manager operation in flight) that could transiently look like "no eviction possible" from the scheduler's point of view but are not actually a true deadlock?

10. **(§2/§4) PEFT pre-claim vs. real claim reconciliation with manager-side KV/adapter co-scheduling.** `pre_claim_peft` reserves budget for `GENERATION_TO_COMPLETE` requests whose adapters are "not yet released" — this is scheduler/PEFT-cache-manager bookkeeping only, unrelated to `KVCacheManagerV2` KV pages. Confirm with the manager-side report whether any KV-page-level accounting has an analogous "not yet released" transient state for `GENERATION_TO_COMPLETE` requests that the scheduler's KV allocation calls (`try_allocate_generation`, `is_request_active`) might not account for.
