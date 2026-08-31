# KVCacheV2 Refactor Design: Toward a Single-Manager / Multi-Pool Target-Draft Architecture

**Repo:** `/Users/allim/TensorRT-LLM`
**Commit:** `4716843cee6e7a6c08bf4d8be29fae25321a9344`
**Branch:** `feat/native-kv-events-clean`
**Date:** 2026-08-31

**Evidence base (all read-only, no code/test/config changes made to produce this document):**
- `scratchpad/kvcachev2_context/{scheduler,manager,interface_map}.md` — general contract audit
- `scratchpad/kvcachev2_context/TRTLLM-15289_audit.md` — JIRA contract audit
- `scratchpad/kvcachev2_context/coverage_closure.md` — coverage matrix over the above
- `scratchpad/kvcachev2_context/topology_and_prefix_reuse.md` — this session's new topology + prefix-reuse trace
- `scratchpad/kvcachev2_context/config_propagation.md` — this session's new configuration-propagation trace

Every claim below traces to `path:line` evidence in one of the above documents (which themselves cite the live source). No claim is carried forward from prior TensorRT-LLM knowledge or historical bug reports without re-verification against this commit — per instruction, where a historical concern ("target-prefix/draft-missing-page") could not be fully confirmed as a live defect, it is reported as **unresolved**, not assumed true.

---

## 1. Current-State Topology

Four topology variants coexist in `KVCacheManagerV2`/`KvCacheCreator` today, selected by disjoint conditions in `tensorrt_llm/_torch/pyexecutor/_util.py`:

| Variant | Selection condition | Draft owns its own manager instance? | Request-object sharing | Evidence |
|---|---|---|---|---|
| **A — Two-model** (separate draft engine) | `self._draft_model_engine is not None` (`_util.py:2032-2034`, checked first) | Yes — separate `KVCacheManagerV2`, own `self.impl`, own `IndexMapper`, own pools | Draft gets a **cloned** `LlmRequest` with the **same numeric `request_id`** (`model_drafter.py:106-123`) | topology_and_prefix_reuse.md Task 1 Variant A |
| **B — One-model, separate draft layout** (Eagle3/MTP-with-independent-layout) | `self._should_create_separate_draft_kv_cache()` (`_util.py:1432-1461`, requires no attention-DP, not DeepSeek-V4-sparse+pp>1, and `should_use_separate_draft_kv_cache(spec_config)`) | Yes — separate `KVCacheManagerV2` instance, but built via `_create_one_model_draft_kv_cache_manager` | **Same `LlmRequest` object** shared with target — no clone (topology_and_prefix_reuse.md §Variant B, Inference from absence of a clone-construction site) | topology_and_prefix_reuse.md Task 1 Variant B |
| **C — Folded/shared manager** (MTP shared-layout, DSpark-embedded, attention-DP forced, DeepSeek-V4-sparse+pp>1 forced) | Negation of A and B: `has_draft` false, or one of the two forced-fold conditions (`_util.py:1435-1461`) | **No** — one `KVCacheManagerV2` instance covers target and draft/MTP layer groups via `layer_mask=None` | Same object trivially — one manager, one `_KVCache` per request | topology_and_prefix_reuse.md Task 1 Variant C |
| **D — Aggregate vs. disaggregated** | `KvCacheCreator._is_disagg` (`_util.py:592`), threaded into every manager constructor call | Orthogonal to A/B/C — doubles `IndexMapper` capacity uniformly (`kv_cache_manager_v2.py:1339-1352`) on whichever manager(s) exist | N/A | topology_and_prefix_reuse.md Task 1 Variant D |

**Per-topology resource ownership (from topology_and_prefix_reuse.md, all Verified fact unless noted):**

- **Request IDs:** numerically shared across all variants; only Variant A uses a genuinely distinct `LlmRequest` Python object.
- **IndexMapper:** one instance per `KVCacheManagerV2` object (`kv_cache_manager_v2.py:1349-1360`) — Variants A/B therefore have two independent `IndexMapper`s per request; Variant C has one.
- **Pools:** one native `self.impl` (hence one set of GPU/host/disk pools) per `KVCacheManagerV2` object — same A/B-vs-C split. GPU budget is **required equal** between target/draft for Variant A (hard `assert`, `_util.py:2079-2084`) and **intentionally unequal** (affine split) for Variant B (`_util.py:2043-2047`, `1638-1740`).
- **Eviction:** `can_evict` and the radix-tree/suspend-resume state live per-`self.impl` — Variants A/B evict target and draft independently, with no cross-manager coordination found; Variant C evicts atomically (one manager, one decision).
- **Prefix reuse:** per-`self.impl` radix tree in Variants A/B, but — critically — **the draft manager's radix tree is never populated or queried** (§2 below). Variant C resolves reuse once for the single shared `_KVCache`, covering all layer groups atomically.

---

## 2. Verified Motivation

This section states only what this session's evidence directly supports. Three categories emerged: (a) a **verified reachable but not fully resolved correctness-mechanism concern**, (b) **verified-fragile invariants that are comment-only, not enforced**, and (c) **verified reachable configuration-divergence paths** whose practical impact is unresolved. None is asserted to be a proven, currently-occurring production bug; all are asserted to be real properties of the current code that a unified architecture would remove **by construction**, independent of whether each currently manifests as an observed failure.

### 2a. Prefix-reuse mechanism concern (verified reachable at the Python level; final consequence unresolved)

For Variants A and B, when `enable_block_reuse=True` on the target manager:
1. Target's `_prepare_context_impl` matches a reused prefix via `self.impl.create_kv_cache(scope, tokens, ...)`, setting `req.context_current_position = kv_cache.num_committed_tokens` (e.g. 800 of 1000 tokens) (`kv_cache_manager_v2.py:2607-2629`, cited in topology_and_prefix_reuse.md §Task 2.1).
2. The target's forward pass computes only the *unreused* remainder (`model_engine.py:5401-5411`), and this chunk boundary is recorded onto the request as `req.py_last_context_chunk` (`py_executor.py:7788-7792`).
3. **The draft manager's own cache creation always passes `input_tokens=None`** (`kv_cache_manager_v2.py:2789-2797`) — unconditionally, not gated on `self.enable_block_reuse` — so the draft's `_KVCache` never attempts reuse matching; `num_committed_tokens=0` for the draft.
4. Yet the draft's chunk bounds are seeded **from the target's post-reuse chunk boundary** — copied directly in Variant A (`model_drafter.py:140-149`), or literally identical in Variant B since the same `req` object is used — meaning the draft engine is told to skip forward-computing exactly the range `[0, 800)` that its own cache never populated by any mechanism (neither reuse nor its own forward pass).
5. Confirmed **no code guard** exists that disables block reuse when a separate draft manager topology is configured (grep across `_util.py`, `llm_args.py`, `speculative/*.py` per topology_and_prefix_reuse.md §Task 2.4) — so the trigger conditions are not mutually exclusive by construction.

**What is verified:** the mechanism described above is a directly-cited, mechanical fact of the current code — the draft manager's committed/history state for the reused-prefix range is never populated while its own bookkeeping (inherited/copied from the target) instructs it to skip computing that same range.

**What is unresolved:** whether this actually causes the draft's attention kernel to read uninitialized/garbage GPU memory (silent corruption, possibly bounded by the target's speculative-decoding acceptance/rejection step) or is safely gated by some native mechanism (e.g. attention metadata built from the draft `_KVCache`'s own `history_length`, which resize() may leave at 0, rather than from `req.context_current_position`). Resolving this requires reading `kvCache.cpp`'s `resize()`/`historyLength` semantics and the draft attention-metadata construction path (`speculative/interface.py`), neither of which was read this session (topology_and_prefix_reuse.md Open Questions #1-#2).

**Verified structural fact relevant to the redesign:** Variant C (folded/shared manager) is **immune to this mechanism by construction** — there is exactly one `_KVCache` per request, so reuse is resolved once for all layer groups atomically, and "draft never consults reuse while target does" cannot occur because there is no separate draft cache creation call at all (topology_and_prefix_reuse.md §Task 2.4a).

### 2b. Comment-only invariants, not enforced by code

- **`layer_mask`/`num_layers` synchronization** (config_propagation.md Dimension 5): the comment "must stay in sync with the num_layers passed to the draft KV cache manager constructor" (`_util.py:1478-1479`) is not backed by any assert. `_create_kv_cache_manager` gives `num_layers` unconditional priority over `sum(layer_mask)` when both are supplied (`_util.py:2327-2333`). **Verified fact: not a live bug today** (both values derive from the same `_get_num_draft_layers()` call within one `build_managers()` invocation), but a future edit to either call site could silently violate the invariant with no runtime error.

### 2c. Verified reachable configuration-divergence paths (impact unresolved)

- **Host-tier auto-sizing timing skew** (config_propagation.md Dimension 2): when `host_cache_size` is unset, each manager (target, then draft, sequentially) independently snapshots live `os.sysconf("SC_AVPHYS_PAGES")` inside its own `__init__` (`kv_cache_manager_v2.py:1105-1145`). Because the target manager's construction actually page-locks host memory before the draft manager constructs, the draft's later memory snapshot observes strictly less available memory — a **verified reachable, ordering-dependent divergence**. `_sync_host_tier_quota`'s cross-rank `allreduce(MIN)` synchronizes a *single* manager's quota across ranks; it does **not** synchronize target-vs-draft quotas within a rank, and no such cross-manager invariant/assert was found. Whether this produces an operational problem is unresolved from source alone (config_propagation.md Open Question #2).
- **Two-model `max_tokens`-driven quota divergence** (config_propagation.md Dimension 1 / Open Question #1): the two-model equality assert only constrains the `max_gpu_total_bytes` **config field**; if a user also sets `kv_cache_config.max_tokens`, each manager independently derives its final `quota` from its own per-layer byte cost, which can differ by architecture even when `max_gpu_total_bytes` is equal — a gap between "config field equal" and "derived device quota equal" that was not confirmed to be blocked elsewhere.
- **Two-model `pool_ratio` arity** (config_propagation.md Dimension 4): no Python-side normalization exists for two-model draft managers (unlike the one-model `[1.0]` reset, `_util.py:1539-1551`); a mismatched arity would fail fast at native construction (`storageManager.cpp:298-312`, `std::invalid_argument`) rather than silently corrupt, but whether any currently-supported config combination reaches this path is unresolved.
- **Two-model attention-window derivation** (config_propagation.md Dimension 3): no dedicated `_derive_draft_max_attention_window`-equivalent exists for the two-model path; window shaping falls back to a generic modulo-indexed pattern-reuse mechanism not specifically verified against real two-model VSWA-draft configurations.

### 2d. Structural duplication with no verified benefit for the coupled cases

For Variant B specifically, target and draft **already share the same `LlmRequest` object and the same `context_current_position`/chunk bookkeeping** (topology_and_prefix_reuse.md §Variant B) — the only thing kept separate is the physical cache/pool/`IndexMapper`/eviction state. This is the case where consolidation into Variant C's single-manager pattern has the clearest verified precedent (Variant C already exists and works for architecturally-equivalent MTP/DSpark-embedded cases) and the least behavioral distance to cover, since request-level coupling is already total.

---

## 3. Proposed Architecture: Single-Manager / Multiple-Pool

**Core idea:** extend today's already-working Variant C pattern (one `KVCacheManagerV2` instance, one native `self.impl`, multiple layer groups distinguished by `pool_ratio`/`initialPoolRatio` and layer masking) to cover the currently-separate-manager cases (Variants A and B), instead of inventing new native machinery. Variant C's existence is itself verified evidence that the native layer-grouping/pool-ratio mechanism (`poolGroupDescs`, `initialPoolRatio`, `layerGrouping`) can already host heterogeneous layer groups (target attention + MTP/embedded-draft layers) inside one manager and one `_KVCache` per request.

**Proposed shape:**
- One `KVCacheManagerV2` instance per (worker role × topology), owning one native `self.impl`, one `IndexMapper`, one set of GPU/host/disk pools, and one radix tree/reuse structure.
- Target and draft layers are expressed as **separate layer groups within that one manager**, each with its own `pool_ratio`/`initialPoolRatio` share (already a native-supported mechanism, per the arity-checked `initialPoolRatio` path in `storageManager.cpp:298-312`) and its own per-layer-group buffer configuration (`BufferConfig` role/size/dtype), rather than as two separate managers.
- One `_create_kv_cache` call per request creates the single `_KVCache` that backs **all** layer groups; reuse (`matchReuse`/`create_kv_cache` with `input_tokens`) is resolved exactly once, and its result — committed history, `context_current_position` — is authoritative for every layer group by construction, eliminating the two-writer inconsistency described in §2a.
- Suspend/resume/evict/free operate on the single `_KVCache`/manager atomically across target and draft layer groups — removing the sequential, unguarded two-call mirror pattern (`_suspend_request`'s primary-then-draft calls, `scheduler_v2.py:1063-1065`) that the JIRA audit flagged as a structural (unproven) divergence risk.
- Host/GPU/disk tier sizing is computed **once per manager**, eliminating the ordering-dependent double-snapshot divergence in §2c, and per-layer-group budget shares are expressed via `pool_ratio` rather than via two independently-sized managers.

**Explicitly scoped exclusion (not proposed to unify without further evidence):** Variant A's two-model case where the draft is a genuinely different model architecture (different `head_dim`, `num_kv_heads`, `dtype`, or `tokens_per_block`-adjacent buffer geometry) may require native layer-group support for heterogeneous per-group buffer roles beyond what Variant C exercises today (Variant C's existing folded cases — MTP, DSpark-embedded — share the target's architecture by construction, per `_get_effective_draft_config`'s fallback, `_util.py:1458-1473`). Whether the native layer-grouping mechanism already supports fully heterogeneous per-group buffer geometry, or would need extension, is an **open question** (§8), not assumed either way.

---

## 4. Invariants the Design Must Guarantee

1. **Single reuse resolution per request.** Exactly one `_create_kv_cache`/reuse-match call per request, whose result (committed history, `context_current_position`) is shared by construction across all layer groups — no code path may create a second, independently-reuse-resolved cache for the same request.
2. **Atomic lifecycle transitions.** Suspend, resume, free, and revert operate on the whole per-request `_KVCache` (all layer groups) in one native call, not as two sequential Python-level calls that could partially fail (removing the §2b-adjacent risk identified in the JIRA audit's self-suspend analysis).
3. **Single IndexMapper slot per request.** One slot, one capacity-doubling policy for disagg (`is_disagg` factor), not two independently-sized `IndexMapper`s that could saturate at different times.
4. **Single tier-quota computation per manager.** GPU/host/disk quotas computed once, with per-layer-group *shares* expressed via `pool_ratio`/`initialPoolRatio` (already arity-checked natively) rather than via duplicated, independently-timed sizing logic — eliminating the host-tier auto-sizing skew in §2c.
5. **Enforced (not comment-only) layer-count/layer-mask consistency.** Any remaining place where `num_layers` and `layer_mask` are both supplied must assert `num_layers == sum(layer_mask)` rather than silently prioritizing one (fixes §2b as a small, low-risk preliminary step — see §5).
6. **No silent loss of intentional per-group budget asymmetry.** The one-model affine GPU-budget split (`_compute_draft_budget_shares`, currently expressed as two managers with different `max_gpu_total_bytes`) must be reproducible losslessly via `pool_ratio`-based per-layer-group shares within the unified manager — this is a hard functional-parity requirement for migrating Variant B, not optional.
7. **`tokens_per_block` remains a single shared scalar** — already true today (config_propagation.md Dimension 7, "structurally guaranteed"); the unified design must not introduce a per-layer-group override, since no downstream consumer (`copy_batch_block_offsets`, index mapping) is designed to handle divergence.
8. **Disagg worker-role correctness.** The design must define, explicitly (not by omission), what a context-only vs. generation-only disagg worker does with draft layer groups — this is currently untraced (topology_and_prefix_reuse.md Open Question #3) and must not be left implicit in the unified design.

---

## 5. Migration Plan and Modes to Remove/Unify

**Step 0 (preliminary, low-risk, independent of the larger migration):** Add the missing `assert num_layers == sum(layer_mask)` (or equivalent) at the one-model draft call site / inside `_create_kv_cache_manager` when both are supplied (§2b, invariant 5). This closes a verified-fragile-but-not-currently-buggy gap with minimal blast radius and should land before or independently of the rest of this plan.

**Step 1 — Migrate Variant B (one-model, separate draft layout) into the unified Variant-C-style pattern first.** Rationale: target and draft already share the same `LlmRequest` object and chunk bookkeeping (§2d) — this is the smallest behavioral delta, and it directly closes the §2a prefix-reuse mechanism concern for MTP/Eagle3/DSpark-with-independent-layout configurations, which is plausibly the most common speculative-decoding deployment shape. Requires: reproducing the affine GPU-budget split via `pool_ratio` (invariant 6), reproducing the `[1.0]`-reset logic's *purpose* (avoiding arity mismatches) via correct per-group `pool_ratio` sizing from the start, and reproducing `_derive_draft_max_attention_window`'s per-group window logic within one manager's layer-group construction.

**Step 2 — Migrate Variant A (two-model, same-GPU-budget case) where the draft architecture is close enough to the target's to fit the existing layer-grouping mechanism** (e.g. same `tokens_per_block`, compatible buffer roles). This is lower-risk than the general two-model case because the GPU-budget-equality assert (`_util.py:2079-2084`) already establishes that today's two-model configurations using V2 do not rely on independent GPU sizing — consolidation does not remove a capability anyone is exercising for this subset.

**Step 3 — Evaluate genuinely-heterogeneous two-model cases** (different `head_dim`/`dtype`/architecture) against whatever native layer-group buffer-heterogeneity support exists or is added (§8 Open Question 1). If native support is insufficient, **retain a separate-manager mode explicitly for this case** rather than forcing unification — do not remove the separate-manager code path until this is resolved.

**Modes to remove/unify, summarized:**

| Current variant | Migration disposition |
|---|---|
| C (folded) | Unchanged — this is the target pattern |
| B (one-model separate layout) | Migrate to unified pattern (Step 1) |
| A, same-architecture/budget subset | Migrate to unified pattern (Step 2) |
| A, heterogeneous-architecture subset | **Retain** as an explicit separate-manager mode pending §8 Open Question 1 |
| D (aggregate/disagg) | Orthogonal — applies uniformly to whichever manager(s) exist post-migration; disagg worker-role semantics must be resolved (invariant 8) before Step 1 ships for disagg deployments |

---

## 6. Correctness Test Plan

1. **Baseline regression (pre-migration):** capture current behavior for all four variants — capacity accounting, suspend/resume, eviction, and (for A/B) the exact §2a mechanism — as characterization tests before any migration code lands, so post-migration equivalence (or intentional behavior change) can be verified against a known baseline.
2. **Prefix-reuse correctness (the primary target of this refactor):** construct a shared-prefix scenario (two requests, same long system prompt, `enable_block_reuse=True`) under the unified Variant-B-migrated manager; assert that the draft layer group's forward pass reads correctly-populated KV state for the reused range (this is the direct fix-verification for §2a) — contrast with an equivalent test on the pre-migration Variant B path, which should demonstrate the mechanism described in §2a (subject to resolving topology_and_prefix_reuse.md Open Questions #1-#2 first, so the "before" state is itself understood).
3. **Layer-count invariant enforcement test:** verify the Step-0 assert actually fires when `layer_mask` and `num_layers` are deliberately desynced in a test double.
4. **Budget-share parity test:** for a migrated Variant B configuration, assert the unified manager's per-layer-group effective capacity matches (within rounding) what the pre-migration affine-split two-manager configuration would have produced, for a representative set of `pool_ratio`/cost-model inputs.
5. **Host-tier sizing determinism test:** assert the migrated unified manager performs exactly one host-tier auto-sizing computation per construction (not two), and that its result no longer depends on construction ordering relative to any other manager.
6. **Atomic lifecycle test:** inject a failure partway through what was previously two sequential suspend/free calls (target-then-draft) and assert the unified single-call lifecycle transition is all-or-nothing — directly closes the JIRA audit's unresolved "can target and draft states disagree" question for migrated topologies.
7. **Disagg worker-role test:** once invariant 8 is resolved, add a test exercising a context-only and a generation-only disagg worker against the unified manager, asserting draft layer-group behavior matches the resolved design (this test cannot be written meaningfully until §8 Open Question 4 is answered).
8. **Native arity regression test:** confirm the migrated `pool_ratio` construction never trips the native `storageManager.cpp:298-312` arity check for any migrated configuration in the test matrix.

## 7. Performance / Trace Benchmark Plan

1. **Memory-efficiency comparison:** for representative MTP/Eagle3 (Variant B) and matched-architecture two-model (Variant A subset) configurations, compare peak GPU/host memory usage between pre- and post-migration topologies via `trtllm-bench` — the unified design removes one `IndexMapper`'s fixed overhead and one manager's per-instance bookkeeping per request, which should be measurable, if small.
2. **Host-overhead trace comparison (nsys):** compare per-iteration host-side scheduling overhead before/after migration — specifically the reduction from two sequential manager dispatch calls (target `try_allocate_generation`/`prepare_context`, then executor-driven draft `prepare_resources`) to one unified dispatch. Use `perf-nsight-systems`-style iteration/NVTX breakdown to isolate this delta from unrelated variance.
3. **Multi-rank collective count:** confirm the migration reduces host-tier auto-sizing from two independent per-rank `allreduce(MIN)` calls (one per manager) to one, and verify this does not change convergence behavior or introduce new collective-ordering hazards under TP/PP/DP.
4. **Speculative-decoding acceptance-rate delta:** for prefix-reuse-heavy workloads (e.g. shared system-prompt benchmarks), measure the draft acceptance rate before and after migration. Given §2a's unresolved status, this benchmark doubles as an **indirect correctness signal**: if pre-migration acceptance rate is measurably depressed on high-reuse workloads relative to low-reuse workloads (controlling for other factors) and post-migration this gap closes, that would be strong empirical corroboration that §2a was a live, silent correctness/quality issue rather than a purely theoretical one — this benchmark should be run and its result folded back into the motivation section before committing to the full migration.
5. **Throughput regression guard:** standard `trtllm-bench throughput` runs across the full migrated configuration matrix (Step 1 and Step 2 subsets) to guard against any unintended overhead from unifying scheduling paths.

## 8. Open Questions That Must Be Resolved Before Implementation

1. **Native layer-group buffer heterogeneity** — does the existing `poolGroupDescs`/`layerGrouping`/`initialPoolRatio` mechanism already support layer groups with genuinely different `head_dim`/`num_kv_heads`/`dtype`/buffer-role sets within one manager, or would this need native extension? This gates whether Step 3 (heterogeneous two-model) is ever feasible without native changes, and determines the final scope of the "retain separate-manager mode" carve-out in §5.
2. **Native `resize()`/`historyLength` semantics and draft attention-metadata construction** (topology_and_prefix_reuse.md Open Questions #1-#2) — required both to fully characterize §2a's current-state severity (garbage-read vs. safely-degraded) and to design the correctness test in §6 item 2 meaningfully.
3. **Disaggregated worker-role participation of draft managers** (topology_and_prefix_reuse.md Open Question #3) — untraced; must be resolved before the unified design can define invariant 8 concretely, and before Step 1 can ship for disaggregated deployments.
4. **Two-model `max_tokens`-driven quota divergence** (config_propagation.md Open Question #1) — is this combination (two-model V2 + explicit `max_tokens`) actually reachable through `TorchLlmArgs` validation today? If reachable, the unified design must decide how per-layer-group `max_tokens`-derived sizing is expressed (extends invariant 6).
5. **Customer-facing behavior change assessment** — does any documented/supported configuration rely on independently sizing the draft's GPU budget smaller than the target's for V2 two-model (today enforced-equal by assert) or one-model (today intentionally split) in a way that unification would need to explicitly preserve as a user-facing knob (`pool_ratio` override), rather than silently changing default behavior? Not assessed this session — requires a docs/config surface review, not just source tracing.
6. **Prioritization signal** — should the acceptance-rate benchmark in §7 item 4 run *before* committing engineering effort to the full migration, specifically to convert §2a from "unresolved mechanism concern" into either "confirmed live quality issue, migrate urgently" or "masked by an as-yet-unidentified native guard, migrate opportunistically for architecture simplification rather than correctness"? This session's evidence supports either framing; only running native-level tracing or the empirical benchmark can distinguish them.
7. **Host-tier auto-sizing divergence impact** (config_propagation.md Open Question #2) — does target/draft host-quota inequality under the current separate-manager topologies cause any observed scheduling or capacity problem today? If not, this motivator is "cleanup" rather than "fix," which affects §5's prioritization but not its correctness requirements.
