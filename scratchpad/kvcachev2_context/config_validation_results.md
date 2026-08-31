# KVCacheV2 Target/Draft Configuration — Validation Results (A/B/C/D)

**Repo:** `/Users/allim/TensorRT-LLM`
**Commit:** `4716843cee6e7a6c08bf4d8be29fae25321a9344`
**Branch:** `feat/native-kv-events-clean`
**Date:** 2026-08-31

**Execution mode:** This machine (macOS/arm64) has no local GPU and no local Slurm client (`nvidia-smi`/`squeue` both absent). Presented with the choice of routing to a remote GPU cluster, running static-only, or using an externally-reachable Docker host, **the user selected static-only analysis, no execution.** Accordingly, no code, test, or script was run at any point in this pass — "test command and result" below report either an existing test's *already-written* assertions (read as source, not executed) or an explicit "Not executed" with the analytical substitute used instead. No production code or tests were modified.

**Evidence base:** `scratchpad/kvcachev2_context/config_propagation.md` (prior deep-dive, not repeated) and `scratchpad/kvcachev2_context/config_static_findings.md` (this session's static source/test-reading pass, produced specifically to answer A/B/C). All citations below trace through those two documents to live `path:line` source.

---

## A. Two-Model Quota Semantics

**Question:** Is two-model V2 with both `max_gpu_total_bytes` and explicit `max_tokens` supported by public config validation? If so, can target/draft quotas diverge despite equal GPU-budget fields?

### Exact configuration

Two-model V2 speculative decoding (`self._draft_model_engine is not None`, `_util.py:2032-2034`). Target: 32 attention layers, `num_kv_heads=8`, `head_dim=128`, fp16. Draft: 4 attention layers, same head geometry, fp16. Both managers configured with `kv_cache_config.max_gpu_total_bytes = 40 GiB` (identical, satisfying the equality assert) and `kv_cache_config.max_tokens = 100,000` (identical, propagated to both via `_util.py:1091,1093,1313`).

### Source citations

- No validator anywhere in `TorchLlmArgs` (`llm_args.py:4297-4409` KvCacheConfig-scoped validators, `llm_args.py:5928-6013` `validate_speculative_config`) cross-checks `kv_cache_config.max_tokens`/`max_gpu_total_bytes` against `speculative_config` — confirmed by exhaustive grep across `llm_args.py`, `_torch/speculative/*.py`, `_torch/pyexecutor/_util.py`.
- Equality assert: `_util.py:2081-2085` (`draft_kv_cache_config.max_gpu_total_bytes == self_kv_cache_config.max_gpu_total_bytes`).
- Quota derivation, independently per manager: `kv_cache_manager_v2.py:1056-1102`.
- `_get_quota_from_max_tokens`/`_get_quota_from_max_tokens_impl`: `kv_cache_manager_v2.py:1646-1685`.
- `_get_runtime_cache_size_layer_components`: `kv_cache_manager_v2.py:1589-1600`.
- `get_layer_bytes_per_token`: `kv_cache_manager_v2.py:3837-3884` (formula: `kv_factor * num_kv_heads * head_dim * dtype_bytes`, `kv_factor=2` for K+V, `kv_cache_manager_v2.py:901,3862-3868`).
- `max_util_for_resume` default `0.95`: `llm_args.py:4174-4181`.

### Test command and result

**Not executed** (static-only mode). Substituted with a hand-worked numeric trace of the cited formulas (`config_static_findings.md` §A.2):

1. Per-layer bytes/token (identical head geometry for both): `2 × 8 × 128 × 2 = 4096 bytes/token/layer`.
2. `full_attn_size_per_token`: target `4096 × 32 = 131,072`; draft `4096 × 4 = 16,384`.
3. `_get_quota_from_max_tokens_impl(100,000)`: target `13,107,200,000` bytes; draft `1,638,400,000` bytes.
4. Divide by `max_util_for_resume=0.95`: target `≈12.85 GiB`; draft `≈1.61 GiB`.
5. `quota = min(max_gpu_total_bytes=40 GiB, quota_from_max_tokens)`: **target quota ≈ 12.85 GiB; draft quota ≈ 1.61 GiB.**

No existing test constructs this exact scenario (`test_kv_cache_budget_split.py`'s two-model tests never set `max_tokens`; `test_dual_pool_kv_cache.py` sets `max_tokens` but never for a two-model draft/target pair — `config_static_findings.md` §A.3).

### Observed target/draft quotas

Target ≈ 12.85 GiB, draft ≈ 1.61 GiB — **an 8× divergence**, derived analytically from source formulas, not empirically measured (no execution performed).

### Classification

**Neither a proven bug nor documented intentional policy — a reachable, unvalidated gap.** The divergence does not crash, does not corrupt state, and each manager's resulting quota is internally consistent with its own architecture (a 4-layer draft model legitimately needs less capacity per token than a 32-layer target). Per the instruction not to call a difference a bug without a concrete incorrect state: **no incorrect state was found** — each manager computes a mathematically correct quota for its own configuration. What *is* verified is that the equality assert's apparent intent ("GPU budgets are not split" per its own comment, `_util.py:2044-2045`) is **silently undermined** by the independent `max_tokens`-driven derivation the moment a user sets `max_tokens` — the assert constrains only the input config field, not the actual byte allocation each manager ends up making. This is **reachable current behavior**, not validated in either direction (no test, no doc, no error), and not classifiable as "intentional" absent a code comment saying so.

### Refactor requirement implied

If the target/draft topology is unified into a single manager with per-layer-group `pool_ratio` shares (as previously proposed), this ambiguity disappears by construction — there is one quota, split by `pool_ratio`, not two independently-derived quotas. If separate managers are retained, the refactor should either (a) extend the equality assert to also constrain (or explicitly document as unconstrained) the `max_tokens`-derived quota, or (b) add a construction-time warning when `max_tokens` is set alongside two-model V2, since today nothing surfaces this divergence to the user at all.

---

## B. Two-Model Layout/Config Compatibility

### B1 — Mismatched layer-group counts with explicit `pool_ratio`

**Exact configuration:** Two-model V2 draft manager whose own architecture has a different layer-group count than the target's `pool_ratio` list length (e.g. target is a hybrid/VSWA model with `pool_ratio=[0.7, 0.3]` for two layer groups; draft is a plain single-group attention model).

**Source citations — full call chain (`config_static_findings.md` §B.2):**
- `_util.py:2072-2089` — two-model draft config derivation; only the GPU-budget-equality assert exists here, nothing checks `pool_ratio`.
- `_util.py:1360-1405,2198-2224` — `pool_ratio` passed through unmodified.
- `kv_cache_manager_v2.py:2090-2107` — `_build_base_config` sets `initial_pool_ratio=kv_cache_config.pool_ratio` directly, no length check.
- `kvCacheManagerV2.cpp:1612-1665` (nanobind binding) — straight pass-through constructor, no length adjustment.
- `kvCacheManager.cpp:104-122` — native constructor calls `mConfig.validate()` (`config.cpp:40-88`, which does **not** check `initialPoolRatio`) before constructing `mStorage`.
- `storageManager.cpp:296-315` — the actual arity check: `if (initialPoolRatio->size() != toSizeT(numLifeCycles())) throw std::invalid_argument(...)`.
- Exception surfacing: `std::invalid_argument` has no custom nanobind translator registered (only `CuError`/`AssertionError` are, `kvCacheManagerV2.cpp:801-849`) — falls through to nanobind's default translator → plain Python `ValueError`.

**Test command and result:** **Not executed.** Static trace of the above chain confirms the failure mode without running code. No existing test constructs this two-manager scenario — the only existing `pool_ratio`-arity test (`test_kv_cache_manager_v2.py:3413-3428`, `test_invalid_initial_pool_ratio`) validates a *single* manager's config in isolation, and the only draft-specific `pool_ratio` test (`test_kv_cache_estimation.py:979-1033`) exercises the **one-model** `[1.0]` reset only, not the two-model unvalidated path.

**Observed behavior:** Construction-time `ValueError: initial_pool_ratio length must match number of layer groups`, raised from deep inside `StorageManager`'s constructor — not from any Python-level pre-check.

**Classification: Safe fail-fast validation.** The mismatch is rejected cleanly at construction time, before any request is processed — it does not proceed incorrectly, does not silently corrupt state, and does not crash the process (per the earlier native ground-truth pass, `std::invalid_argument` is a catchable Python exception, not a `terminateOnException` abort). It is a legitimate fail-fast, just with a diagnostics gap: the error surfaces from deep native code with no attribution back to "your two-model draft's layer-group layout doesn't match its `pool_ratio`," unlike the one-model path which normalizes proactively and logs why (`_util.py:1539-1551`).

**Refactor requirement implied:** Either extend the one-model-style `[1.0]`-normalization logic to the two-model path (so a mismatched two-model draft fails with a clear, attributable Python-level message or is auto-normalized when safe to do so), or explicitly document that two-model draft configurations with custom `pool_ratio` require manual layer-group-count matching. Under a unified single-manager design, this scenario is avoided differently: `pool_ratio` would be defined once per manager covering all layer groups (target's and draft's, in the same list), so the "two independently-configured `pool_ratio` lists" question doesn't arise.

### B2 — Per-layer attention-window vectors for differing VSWA/window layouts

**Exact configuration:** Two-model V2 with a VSWA target (e.g. `max_attention_window=[512, 4096, 512]` reflecting the target's own layer pattern) and a draft model with a different (typically smaller) layer count.

**Source citations (`config_static_findings.md` §B.3):**
- Indexing code, `kv_cache_manager_v2.py:2079-2088` and `:1589-1600`: `self.max_attention_window_vec[self.pp_layers[layer_id] % len(self.max_attention_window_vec)]`.
- `self.max_attention_window_vec` provenance (`kv_cache_manager_v2.py:935-952`): built directly from whatever `kv_cache_config.max_attention_window` was passed to *that* manager. For two-model draft, this is the target's config, budget-split only (`_split_kv_cache_budget_for_draft`, `_util.py:2044-2052`, which touches only byte-budget fields) — so **the draft manager's window vector is the target's raw, unmodified list.**
- `self.pp_layers` provenance (`kv_cache_manager_v2.py:838-843`, `resource_manager.py:193-213`): for two-model, built from the draft model's own (typically smaller) `num_hidden_layers`, with `layer_mask=None` — i.e. the draft's own correct layer index sequence.
- `_derive_draft_max_attention_window` (`_util.py:513-536`) — confirmed by repo-wide grep to have exactly one production caller, `_get_one_model_draft_kv_cache_config` (`_util.py:1499-1514`), itself only reachable from the **one-model** path (`_util.py:1535`). **The two-model branch never calls it.**

**Test command and result:** **Not executed.** No existing test constructs a two-model draft with a layer count different from its VSWA target and inspects the resulting per-layer `sliding_window_size` values — the only existing window-derivation tests (`test_eagle3.py:188-236`) unit-test `_derive_draft_max_attention_window` directly, which (per the citation above) is never invoked for two-model at all.

**Observed behavior (from static trace only):** The modulo indexing is memory-safe — `0 <= index < len(vec)` always holds, so this cannot raise `IndexError` or crash. But it takes the draft's own layer position and reduces it modulo the length of the **target's** window list, indexing into the **target's** pattern with no draft-architecture-aware derivation at all.

**Classification: Inconclusive** (not "reachable bug," per the instruction not to call a difference a bug without a concrete incorrect state). What is verified is the *mechanism* — a two-model draft with a different layer count gets no draft-specific window derivation and instead inherits a modulo-indexed slice of the target's raw pattern. Whether this actually assigns a *wrong* window to any draft layer for any currently-shipped two-model configuration was **not established** — that would require constructing a real VSWA target + differently-layered two-model draft pair and inspecting the resulting `sliding_window_size` per draft layer, which needs execution this pass explicitly did not perform.

**Refactor requirement implied:** Before any refactor decision, resolve this specific inconclusive item with either (a) a native/runtime test constructing a concrete VSWA-target + differently-layered two-model-draft pair and asserting on the resulting per-layer windows, or (b) a source-level extension of `_derive_draft_max_attention_window` (or an equivalent) to cover the two-model path regardless of refactor direction, since the current code path is unguarded and unintentional-looking (no comment justifies inheriting the target's raw pattern for two-model, unlike the one-model path's three explicitly-commented cases). Under a unified single-manager design, each layer group would need its own explicitly-derived window regardless of source model, which would force this gap to be closed as part of the migration rather than left implicit.

---

## C. Auto-Host Sizing Behavior

**Exact configuration:** Target `KVCacheManagerV2` constructed first with `host_cache_size=None` (auto-sizing), followed by draft `KVCacheManagerV2` constructed second, also with `host_cache_size=None`, in the same process/rank.

### Source citations

- Auto-sizing branch: `kv_cache_manager_v2.py:1105-1150`, `_compute_auto_host_tier_quota` (`:242-279`), `_sync_host_tier_quota` (`:282-304`).
- Sequential construction order (target before draft): `_util.py` `build_managers` (established in the prior config-propagation pass, target constructed before either draft branch).
- No cross-manager comparison exists anywhere in source: exhaustive grep of `kv_cache_manager_v2.py` and `scheduler/scheduler_v2.py` for `host_tier`/`quota` combined with `draft`/`target` manager references — **zero matches** in the scheduler; the only source references to `host_tier` are the two functions above plus their call site (`config_static_findings.md` §C.3).

### Test command and result

**Not executed.** Existing coverage read in full instead (`test_kvv2_host_tier_sizing.py`, 182 lines, 9 test functions — full table in `config_static_findings.md` §C.1). Summary: every test either (a) calls `_compute_auto_host_tier_quota` in isolation with hand-picked scalars for a single rank/manager, or (b) exercises `_sync_host_tier_quota`'s cross-**rank** MPI reconciliation (`test_multi_rank_syncs_to_fleet_min`, the only multi-instance test in the file) for **one manager type replicated across ranks** — not a target-vs-draft comparison. No test constructs two `KVCacheManagerV2` instances in one process and compares their host quotas.

### Observed target/draft host quotas

**Not observed — no execution performed.** The mechanism (independent, sequentially-timed `os.sysconf("SC_AVPHYS_PAGES")` snapshots, one per manager, target's snapshot preceding draft's) is confirmed by source reading (`config_propagation.md`, `config_static_findings.md` §C.3) but no actual numbers were captured.

### Whether unequal quotas affect suspend/resume or capacity admission

**Verified absent — no code path compares one manager's host-tier quota against another's.** `_compute_auto_host_tier_quota`/`_sync_host_tier_quota` are pure functions of scalar arguments with no manager-instance references; the call site in `__init__` reads only that manager's own `mapping`/OS-state; and the scheduler (`scheduler_v2.py`) contains zero references to `host_tier`/`quota` at all — the scheduler-visible `can_evict` boolean (established in the prior general audit) is a same-manager attribute, not a cross-manager comparison.

### Classification

**Reachable, no proven adverse effect — not classifiable as a bug.** The sequential-snapshot mechanism genuinely produces two independently-timed, independently-sized host quotas (this part is Verified current behavior). But since no code anywhere depends on target/draft host quotas being equal or related, an inequality between them cannot violate any invariant that exists in source — there is simply nothing for it to break. Whether the *magnitude* of this divergence ever meaningfully starves the draft manager's host tier under real memory pressure (e.g., if the target's construction consumes enough host memory that the draft's later snapshot sees a materially smaller available pool) is a question about runtime memory dynamics, not source logic — **Source-inconclusive, requiring native/runtime test** with actual memory-pressure conditions, since this cannot be resolved by reading code that computes each manager's quota in isolation.

### Refactor requirement implied

A unified single-manager design (per-request, single host-tier quota computed once) removes this divergence source entirely by construction, matching how it already removes the GPU-quota divergence from §A. If separate managers are retained, the refactor should either compute the auto-host-tier quota once and split it (mirroring the existing explicit-`host_cache_size` split path, `_split_kv_cache_budget_for_draft`), or accept the current per-manager independence as intentional and document it (currently it is neither validated nor documented either way).

---

## D. Synthesis Summary

| Question | Classification | Concrete incorrect state / crash proven? | Refactor implication |
|---|---|---|---|
| A — Two-model quota divergence via `max_tokens` | Reachable, unvalidated gap (not a bug) | No — each manager's quota is internally correct for its own architecture; only the assert's implied guarantee is undermined | Unification removes the ambiguity; if kept separate, extend validation or document explicitly |
| B1 — `pool_ratio` arity mismatch | Safe fail-fast validation | Construction-time `ValueError`, not silent corruption — confirmed | Extend one-model-style normalization/clearer error to two-model, or accept as-is; unification sidesteps the two-independent-lists question |
| B2 — Attention-window indexing for differing layer counts | Inconclusive | Mechanism confirmed reachable; no concrete wrong-value case proven without execution | Needs a targeted native/runtime test before/independent of any refactor decision; unification forces explicit per-group derivation regardless |
| C — Auto-host sizing divergence | Reachable, no proven adverse effect | No — no code compares cross-manager host quotas, so nothing is violated; magnitude/impact under real pressure unresolved | Unification removes the divergence source; if kept separate, either split a single computation or document the independence explicitly |

**Overall:** none of the four questions resolved to a "reachable bug" in the strict sense the task requires (a concrete incorrect state, crash, or proven violation) — the static evidence available in this session either shows safe fail-fast behavior (B1), internally-correct-but-divergent computations with no downstream dependency to violate (A, C), or a plausible-but-unproven mechanism requiring actual execution to confirm (B2). This is consistent with, and does not contradict, the higher-priority open item already identified in `refactor_ground_truth.md` (whether the draft's prefix-reuse mechanism causes actual data corruption) — that remains the single item in this whole investigation series with a live, unresolved correctness stake; today's four sub-questions are best characterized as configuration-robustness and validation-completeness gaps rather than correctness defects.
