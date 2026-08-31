# KV Cache V2 — Static Config/Validation Findings

Commit: `4716843cee6e7a6c08bf4d8be29fae25321a9344`
Branch: `feat/native-kv-events-clean`
Date: 2026-08-31

Method: static source reading only (no code execution, no pytest, no scripts run). Builds on
`scratchpad/kvcachev2_context/config_propagation.md` (not re-summarized here).

Note on path correction: the prior deep-dive referenced `tensorrt_llm/_torch/speculative/_util.py` and
`cpp/tensorrt_llm/batch_manager/storageManager.cpp`. Neither path exists. The actual files are
`tensorrt_llm/_torch/pyexecutor/_util.py` (same line-numbered content as described) and
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp` (same lines 296-315). All
citations below use the corrected paths.

---

## Question A: Is two-model V2 + explicit `max_tokens` blocked by public config validation?

### A.1 — Does any validator reject `max_tokens` + two-model V2 speculative decoding?

**Verified absent.**

- `tensorrt_llm/llmapi/llm_args.py`: 88 `@model_validator`/`@field_validator` occurrences total. All
  `KvCacheConfig`-scoped validators (`llm_args.py:4297-4409`) check only internal `kv_cache_config.*`
  invariants (dtype, `free_gpu_memory_fraction`, `max_gpu_total_bytes` ≥ 0, `mamba_state_config`,
  `max_attention_window`, `pool_ratio`) — none reference `speculative_config`.
- `TorchLlmArgs.validate_speculative_config` (`llm_args.py:5928-6013`+) is the only validator that
  inspects `speculative_config` in depth (backend support, rejection-sampling compatibility with
  `context_parallel_size`, `guided_decoding_backend`, `sa_config`, dynamic-tree topK). It never reads
  `self.kv_cache_config`.
- Every `kv_cache_config` reference in `llm_args.py` is either confined to the `KvCacheConfig` class body
  (~3596-4409) or to unrelated cross-references: `sync_quant_config_with_kv_cache_config_dtype`
  (`llm_args.py:6410`) and the HELIX `cp_config.tokens_per_block` check (`llm_args.py:6428`) — neither
  touches `max_tokens`/`max_gpu_total_bytes` vs. `speculative_config`.
- `grep -rln "draft_kv_cache_config" tensorrt_llm/` → only `tensorrt_llm/_torch/pyexecutor/_util.py`.
  The assert is at `_util.py:2081-2085`, inside `if` at `_util.py:2078-2079`.
- `grep -rn "kv_cache_config.*speculative_config\|speculative_config.*kv_cache_config"` across
  `llm_args.py`, `_torch/speculative/*.py`, `_torch/pyexecutor/_util.py` → no output.

**Conclusion:** nothing in the Pydantic validation layer blocks or restricts setting
`kv_cache_config.max_tokens` together with two-model V2 speculative decoding. Execution reaches
`KvCacheCreator.build_managers` → `KVCacheManagerV2.__init__` unblocked.

### A.2 — Worked numeric example of divergent quotas

**Verified current behavior**, formulas read in full from:
- Quota derivation: `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py:1056-1102`
- `_get_quota_from_max_tokens` / `_get_quota_from_max_tokens_impl`: `kv_cache_manager_v2.py:1646-1685`
- `_get_runtime_cache_size_layer_components`: `kv_cache_manager_v2.py:1589-1600`
- `get_layer_bytes_per_token`: `kv_cache_manager_v2.py:3837-3884`
- `kv_factor` assignment: `kv_cache_manager_v2.py:901` (`self.kv_factor = 1 if SELFKONLY else 2`)
- `_estimate_full_attn_size_per_token`: `kv_cache_manager_v2.py:232-239`
- `max_util_for_resume` default 0.95: `tensorrt_llm/llmapi/llm_args.py:4174-4181`
- Assert: `tensorrt_llm/_torch/pyexecutor/_util.py:2078-2085`

**Formula** (`kv_cache_manager_v2.py:1056-1071`):
```
quota = max_gpu_total_bytes                                       # if set (>0)
if max_tokens is not None:
    quota_from_max_tokens = ceil(_get_quota_from_max_tokens(max_tokens) / max_util_for_resume)
    quota = min(quota, quota_from_max_tokens)
```
For the non-VSWA case (`kv_cache_manager_v2.py:1656-1685`, all SWA terms 0), this collapses to:
```
_get_quota_from_max_tokens_impl(max_tokens) = max_tokens * full_attn_size_per_token
                                             = max_tokens * sum(layer_bytes_per_token for local layers)
```
Per-layer, non-quantized (`kv_cache_manager_v2.py:3862-3868`):
```
layer_bytes_per_token = kv_factor * num_kv_heads * head_dim * dtype_bytes,  kv_factor = 2 (K+V)
```

**Concrete example:** target = 32 layers, `num_kv_heads=8`, `head_dim=128`, fp16 (2 bytes/elem); draft =
4 layers, same head geometry, fp16; `kv_cache_config.max_tokens = 100,000` set identically for both
(propagated identically per `_util.py:1091/1093,1313`); `max_gpu_total_bytes = 40 GiB` for both (satisfies
the `_util.py:2081-2082` equality assert, and is large enough not to bind).

1. Per-layer bytes/token (identical for both, same head config):
   `2 * 8 * 128 * 2 = 4096 bytes/token/layer`
2. `full_attn_size_per_token` (sum over local layers, `kv_cache_manager_v2.py:232-239`):
   - Target: `4096 * 32 = 131,072 bytes/token`
   - Draft: `4096 * 4 = 16,384 bytes/token`
3. `_get_quota_from_max_tokens_impl(100,000)` (`kv_cache_manager_v2.py:1656-1685`):
   - Target: `100,000 * 131,072 = 13,107,200,000 bytes`
   - Draft: `100,000 * 16,384 = 1,638,400,000 bytes`
4. Apply `max_util_for_resume = 0.95` (`kv_cache_manager_v2.py:1064-1071`), `quota_from_max_tokens = ceil(raw / 0.95)`:
   - Target: `ceil(13,107,200,000 / 0.95) = 13,797,052,632 bytes ≈ 12.85 GiB`
   - Draft: `ceil(1,638,400,000 / 0.95) = 1,724,631,579 bytes ≈ 1.61 GiB`
5. Final `quota = min(max_gpu_total_bytes, quota_from_max_tokens)` (`kv_cache_manager_v2.py:1071`), with
   `max_gpu_total_bytes = 40 GiB` for both:
   - Target quota: `min(40 GiB, 12.85 GiB) = 12.85 GiB`
   - Draft quota: `min(40 GiB, 1.61 GiB) = 1.61 GiB`

**Result:** despite `max_gpu_total_bytes` being asserted identical, and `max_tokens` being numerically
identical for both managers, the resulting GPU byte `quota` values differ by exactly the ratio of
per-token byte costs (8× here, from the 32-vs-4 layer difference). The assert at `_util.py:2081-2085`
constrains only the input cap, not the derived `quota` each `KVCacheManagerV2` actually allocates once
`max_tokens` is also set.

### A.3 — Existing test coverage

**Verified absent.** Search patterns tried:
- `grep -n "draft_kv_cache_config.max_gpu_total_bytes ==" / "does not support two-model" / "separate draft GPU"` across `tests/unittest/` → no hits.
- `grep -rln "_get_quota_from_max_tokens\|quota_from_max_tokens" tests/unittest --include="*.py"` → hits
  only in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`
  (single-manager scratch/SWA quota test, no draft/target pair), plus `test_kv_cache_estimation.py` and
  `test_kv_cache_manager_v2_helix_superblock.py` (neither reference draft in that context).
- `grep -rln "_draft_model_engine" tests/unittest --include="*.py" | xargs grep -l "max_tokens"` →
  `tests/unittest/_torch/executor/test_kv_cache_budget_split.py` and
  `tests/unittest/_torch/executor/test_dual_pool_kv_cache.py`.
  - `test_kv_cache_budget_split.py` (`_make_creator` L36-77, `_make_two_model_creator` L625-632,
    `test_two_model_offload_budget_is_split` L635-652, `test_two_model_keeps_the_gpu_budget_whole`
    L654-663): the `KvCacheConfig` fixture (L55-60) never sets `max_tokens` (only
    `max_gpu_total_bytes`, `host_cache_size`, `disk_cache_size`/`disk_cache_path`).
    `test_two_model_keeps_the_gpu_budget_whole` asserts `config.max_gpu_total_bytes == 10 * GB` for both
    target and draft `kv_cache_config_override` — confirms two-model V2 never splits
    `max_gpu_total_bytes`, but says nothing about `max_tokens` or the derived `quota`.
  - `test_dual_pool_kv_cache.py` sets `max_tokens` (e.g. L259, 276, 420, 606) but only with
    `creator._should_create_separate_draft_kv_cache = Mock(return_value=False)` and
    `creator._draft_model_engine = None` (L175) — single-model/cross-attention (encoder-decoder)
    fixtures, not a two-model draft+target pair.

**No test constructs a two-model V2 draft/target pair with `max_tokens` explicitly set and asserts
anything about the resulting `quota` values.**

---

## Question B: Two-model layout/config compatibility — pool_ratio arity and attention-window vectors

### B.1 — Existing tests for two-model draft/target with different layer-group counts / `pool_ratio`

**Verified absent** for the two-model construction-time scenario. Searches: `grep -rn "pool_ratio"
tests/unittest/`; `grep -rln "draft_model_engine|two.model|two_model|is_draft_model|TwoModel"
tests/unittest/ | xargs grep -l "pool_ratio"` → only `tests/unittest/_torch/executor/test_kv_cache_estimation.py`.

All `pool_ratio` tests found are:
- **Single-manager config validation** (not draft/target pair):
  `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py:3413-3428`,
  `test_invalid_initial_pool_ratio`, parametrized `("wrong_length", [1.0], "initial_pool_ratio length")`
  etc., via `_make_config` (single-manager helper). Confirms Python-level `KvCacheConfig` construction
  validates `pool_ratio` sum/positivity/length against constraints at the *config* level, unrelated to a
  two-manager arity mismatch.
- **One-model normalization only:**
  `tests/unittest/_torch/executor/test_kv_cache_estimation.py:979-1033`,
  `test_separate_one_model_draft_normalizes_target_pool_ratio` — asserts (L1032)
  `draft_config.pool_ratio == [1.0]` and (L1033) `creator._kv_cache_config.pool_ratio == target_pool_ratio`
  (target's untouched, draft's reset to `[1.0]`). Exercises the one-model reset at `_util.py:1539-1551`
  only.
- `tests/unittest/_torch/executor/test_mamba_cache_manager.py:2024`
  (`test_v2_hybrid_pool_ratio_controls_allocated_memory`) and
  `tests/unittest/llmapi/test_llm_args.py:807-838,1216-1229` exercise `pool_ratio` for hybrid
  single-manager cases / Pydantic field validation — not two-model draft/target pairs.

**No test anywhere constructs two separate `KVCacheManagerV2` instances (draft + target) with different
layer-group counts or mismatched `pool_ratio` and checks the resulting behavior at construction.** This is
a genuine coverage gap, not a documented "expected clean error" scenario.

### B.2 — `pool_ratio` call chain to the native arity check; no earlier normalization

**Verified current behavior**, full hop-by-hop chain:

1. Two-model creation site, `tensorrt_llm/_torch/pyexecutor/_util.py:2072-2089`:
   ```python
   draft_build_kv_cache_config = (draft_kv_cache_config
                                  if draft_kv_cache_config is not None else
                                  self_kv_cache_config)
   ...
   if self._draft_model_engine is not None:
       if (self._is_kv_cache_manager_v2 and draft_kv_cache_config is not None):
           assert (draft_kv_cache_config.max_gpu_total_bytes ==
                   self_kv_cache_config.max_gpu_total_bytes), (...)
       draft_kv_cache_manager = self._create_kv_cache_manager(
           self._draft_model_engine, estimating_kv_cache,
           kv_cache_config_override=draft_build_kv_cache_config)
   ```
   The only assert here (L2081-2085) checks `max_gpu_total_bytes` equality; nothing checks `pool_ratio`.
2. `self._create_kv_cache_manager` (`_util.py:1360-1405`) passes `kv_cache_config` through unmodified
   (w.r.t. `pool_ratio`) to the module-level `_create_kv_cache_manager` free function
   (`_util.py:2198-2224`).
3. That function passes `kv_cache_config` (still carrying the untouched target/split `pool_ratio`) to
   `kv_cache_manager_cls(...)`, i.e. `KVCacheManagerV2.__init__`.
4. `KVCacheManagerV2._build_base_config` (`kv_cache_manager_v2.py:2090-2107`) sets
   `initial_pool_ratio=kv_cache_config.pool_ratio` (L2106) directly — no length check against
   `layer_configs`/layer groups.
5. This builds a `KVCacheManagerConfigPy` — nanobind class `kv::KVCacheManagerConfig` bound in
   `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:1612-1665`. The binding is a straight
   pass-through constructor (`initial_pool_ratio` arg → `cfg->initialPoolRatio`,
   `kvCacheManagerV2.cpp:1618,1637`), no length adjustment.
6. Native `KvCacheManager::KvCacheManager`
   (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.cpp:104-122`) member-initializes
   `mLifeCycles(config)`, then calls `mConfig.validate()` (L114) before constructing `mStorage`.
7. `KVCacheManagerConfig::validate()`
   (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/config.cpp:40-88`) checks cache-tier ordering,
   duplicate layer ids, `tokensPerBlockOverride` divisibility, SSM `commit_min_snapshot` — **does not
   check `initialPoolRatio` at all**.
8. `mStorage = std::make_shared<StorageManager>(mLifeCycles, storageConfig, ...,
   mConfig.initialPoolRatio, ...)` (`kvCacheManager.cpp:117-119`) reaches
   `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp:296-315`:
   ```cpp
   if (initialPoolRatio.has_value())
   {
       if (initialPoolRatio->size() != toSizeT(numLifeCycles()))
       {
           throw std::invalid_argument("initial_pool_ratio length must match number of layer groups");
       }
       ...
   }
   ```
9. `std::invalid_argument` is not custom-translated: the only two `nb::register_exception_translator`
   calls in `kvCacheManagerV2.cpp:801-830` and `835-849` handle `kv::CuError` and `kv::AssertionError`
   respectively. `std::invalid_argument` falls through to nanobind's default translator → plain Python
   `ValueError`.

**Conclusion:** no Python-side or native pre-check inspects, truncates, pads, or otherwise normalizes
`pool_ratio` for the two-model draft path before it reaches `StorageManager`'s constructor. A wrong-length
`pool_ratio` for a two-model draft manager reaches the native arity check as-is and surfaces as
`ValueError: initial_pool_ratio length must match number of layer groups` at draft-manager construction
time — a clean-ish (not silent/corrupted) failure, but attributed deep inside `StorageManager` rather than
with a friendlier Python-level message (unlike the one-model path, which normalizes to `[1.0]` before ever
reaching this code).

### B.3 — Per-layer attention-window indexing for two-model draft

**Verified current behavior.** Exact indexing code, `kv_cache_manager_v2.py:2079-2088` (inside
`_build_base_config`):
```python
layer_configs: List[AttentionLayerConfig] = []
for layer_id in typed_range(LayerId(self.num_local_layers)):
    buffers = [...]
    ...
    layer_configs.append(
        AttentionLayerConfig(
            layer_id=layer_id,
            buffers=buffers,
            sliding_window_size=self.max_attention_window_vec[
                self.pp_layers[layer_id] % len(self.max_attention_window_vec)
            ],
            num_sink_tokens=None,
        )
    )
```
Equivalent pattern also at `kv_cache_manager_v2.py:1589-1600`
(`_get_runtime_cache_size_layer_components`):
```python
pattern_len = len(self.max_attention_window_vec)
for local_layer_idx in range(self.num_local_layers):
    ...
    attention_windows.append(
        self.max_attention_window_vec[self.pp_layers[local_layer_idx] % pattern_len]
    )
```

**Operand provenance:**
- `self.max_attention_window_vec` (`kv_cache_manager_v2.py:935-952`) is built directly from whatever
  `kv_cache_config.max_attention_window` was passed to *this* manager's constructor:
  ```python
  if kv_cache_config.max_attention_window is not None:
      self.max_attention_window_vec = kv_cache_config.max_attention_window.copy()
      self.max_attention_window_vec = [min(max_seq_len, w) for w in self.max_attention_window_vec]
      self.max_attention_window_vec = [None if w == max_seq_len else w for w in self.max_attention_window_vec]
  else:
      self.max_attention_window_vec = [None]
  ```
  For two-model mode, the `kv_cache_config` fed to the draft manager
  (`draft_build_kv_cache_config`, `_util.py:2072-2089`) is either a budget-split copy or a direct alias
  of `self_kv_cache_config` (the target's config). The budget split
  (`_split_kv_cache_budget_for_draft`, `_util.py:2044-2052`) only touches byte-budget fields
  (`max_gpu_total_bytes`, offload tiers), not `max_attention_window`. So the draft manager's
  `max_attention_window_vec` is **the target's own raw window list, unmodified**.
- `self.pp_layers` (`kv_cache_manager_v2.py:838-843`) is built via `get_pp_layers(num_layers, mapping,
  spec_config=spec_config, layer_mask=layer_mask)`. For the draft manager, `num_layers` traces back
  (`_create_kv_cache_manager`, `_util.py:2242-2250` → `config.num_hidden_layers` at `_util.py:2335`) to
  the **draft model engine's own (typically smaller) `num_hidden_layers`**, with `layer_mask=None` for
  two-model (mask is only set for one-model separate-draft-cache at `_util.py:1378-1380`). So
  `self.pp_layers` for the draft manager is `[0, 1, ..., num_draft_layers-1]` (or PP-sharded subset) —
  the draft's own, correct layer index sequence (confirmed via `get_pp_layers`,
  `resource_manager.py:193-213`).

**Indexing as written:**
`self.max_attention_window_vec[self.pp_layers[layer_id] % len(self.max_attention_window_vec)]` — takes
the draft's own global layer index, reduces it modulo the length of the **target's** window vector, and
indexes into the **target's** window list.

**Safety assessment (read `kv_cache_manager_v2.py:2040-2107` generously):**
- **Safe by construction, never out-of-bounds:** the modulo against `len(self.max_attention_window_vec)`
  guarantees `0 <= index < len(vec)` regardless of the relationship between `self.pp_layers[layer_id]`
  and that length — cannot raise `IndexError`. No other guard exists or is needed for memory safety.
- **Semantically questionable / likely incorrect for two-model draft with a different layer count:** the
  target's `max_attention_window` list is designed (e.g. Gemma4 hybrid logic at `_util.py:2296-2313`, or
  user-provided VSWA config) with a length and per-index meaning tied to the **target's** layer
  structure/VSWA period. When the draft has a smaller/different layer count, `self.pp_layers[layer_id]`
  enumerates *draft*-relative layer positions (0..num_draft_layers-1), positionally mapped (mod
  target-list-length) into the *target's* window pattern. There is no architectural guarantee that "draft
  layer 2" corresponds to whatever pattern position `2 % len(target_vec)` represents in the target's VSWA
  layout. This can silently assign the wrong sliding-window size to a draft layer (e.g. full attention
  where sliding was intended, or vice versa) without any error or warning — **Verified current behavior**
  for the mechanism; **Source-inconclusive** on whether this actually manifests incorrectly for any
  currently-shipped model pairing (would require running an actual VSWA target + differently-layered
  draft combination and inspecting the resulting `sliding_window_size` per draft layer).

**Bypass of `_derive_draft_max_attention_window` for two-model — Verified current behavior.** Repo-wide
caller search (`grep -rn "_derive_draft_max_attention_window" tensorrt_llm/ tests/`): function defined at
`_util.py:513-536` (also cited in prior session as `495-536`); its **only production caller** is
`_util.py:1492`, inside `_get_one_model_draft_kv_cache_config` (`_util.py:1499-1514`), itself only called
from `_create_one_model_draft_kv_cache_manager` (`_util.py:1535`) — the **one-model** separate-draft-cache
path. The two-model branch (`_util.py:2077-2089`) never calls it; it passes
`draft_build_kv_cache_config` (target-derived, budget-split only, `_util.py:2072-2074`) straight to
`_create_kv_cache_manager`. **Two-model draft managers get no draft-specific attention-window derivation
at all** — they inherit the target's raw `max_attention_window` list verbatim and rely purely on the
safe-but-semantically-blind modulo indexing above.

### B.4 — Existing tests for draft attention-window correctness

**Verified absent** for the two-model, differing-layer-count scenario. Search:
`grep -rln "max_attention_window" tests/unittest/ | xargs grep -l "draft"` → 15 files. Only
`tests/unittest/_torch/speculative/test_eagle3.py` tests draft-window *derivation logic* directly:
- `test_eagle3_draft_kv_cache_uses_full_window_when_draft_has_no_swa` (L188-199)
- `test_eagle3_draft_kv_cache_uses_draft_layer_types_for_swa` (L202-216): asserts
  `max_attention_window == [512, 4096, 512]`
- `test_eagle3_draft_kv_cache_rejects_multiple_sliding_window_sizes` (L219-236)

All three call `_derive_draft_max_attention_window` directly as a **unit test of that function** — i.e.
they test the **one-model** derivation helper's output, not the per-layer `sliding_window_size` actually
assigned inside a constructed `KVCacheManagerV2`, and not the two-model path (which, per B.3, never calls
this function). `grep -n "def test_"` on
`tests/unittest/_torch/executor/test_kv_cache_manager_v2.py` and
`tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py`, filtered on
`window\|draft`, turned up only unrelated `is_draft`-flag plumbing tests, not window-correctness
assertions.

**No test constructs a two-model draft `KVCacheManagerV2` with a layer count different from its target
and asserts the resulting per-layer window values are correct or even sane.**

---

## Question C: Auto-host sizing — existing test coverage

### C.1 — Full test table for `tests/unittest/_torch/executor/test_kvv2_host_tier_sizing.py`

**Verified current behavior** (file read in full, 182 lines). The module docstring (L1-10) explicitly
frames scope as **per-rank / cross-rank** sizing, not cross-manager: *"the auto-provisioned host tier is
computed per rank ... `TestSyncHostTierQuota` covers the cross-rank `allreduce(MIN)`."*

| Test function | Inputs (exact) | Expected output | What it proves |
|---|---|---|---|
| `test_single_rank_uses_device_quota_when_memory_is_ample` (L24-34) | `quota=173*GiB, local_ranks=1, mem_available=440*GiB, memlock_limit=inf` | `== 173*GiB` | With 1 rank and ample memory, device quota (not the 220 GiB memory cap) wins. |
| `test_colocated_ranks_divide_node_memory_budget` (L36-44) | `quota=173*GiB, local_ranks=4, mem_available=440*GiB, memlock_limit=inf` | `== int(440*GiB/4*0.5)` = 55 GiB | 4 co-located ranks each get `mem_available/local_ranks*0.5`, not the full device quota. |
| `test_aggregate_across_ranks_stays_within_available_memory` (L46-55) | `local_ranks=4, mem_available=440*GiB` (same as above) | `per_rank * local_ranks <= mem_available` | Aggregate of 4 ranks' quotas stays within node memory — computed from one call's result × `local_ranks`, not from invoking the function 4× or across 2 different managers. |
| `test_memlock_limit_caps_quota` (L57-63) | `quota=173*GiB, local_ranks=1, mem_available=inf, memlock_limit=10*GiB` | `== int(10*GiB*0.8)` = 8 GiB | RLIMIT_MEMLOCK caps quota at 80% of soft limit. |
| `test_unknown_limits_fall_back_to_device_quota` (L65-74) | `quota=173*GiB, local_ranks=8, mem_available=inf, memlock_limit=inf` | `== 173*GiB` | Both limits `inf` → falls back to device quota regardless of `local_ranks`. |
| `test_non_positive_result_falls_back_to_device_quota` (L76-89, parametrized `memlock_limit=[0.0, 1.0]`) | `quota=173*GiB, local_ranks=4, mem_available=440*GiB, memlock_limit=0.0 or 1.0` | `== 173*GiB` | A ≤0-producing memlock limit falls back to device quota (avoids deadlocking the scheduler). |
| `test_result_is_always_positive` (L91-101) | `quota=173*GiB, local_ranks=4, mem_available=0.0, memlock_limit=inf` | `> 0` | Exhausted node memory reading never yields a non-positive tier. |
| `test_single_rank_is_a_noop` (L137-144) | `quota=173*GiB`, fake mapping `world_size=1` | `_sync_host_tier_quota(quota, mapping) == quota` | `world_size==1` skips the collective entirely. |
| `test_multi_rank_syncs_to_fleet_min` (L146-181, `@pytest.mark.cpu_only`, skipped without `ENABLE_MULTI_DEVICE`) | 2 real MPI ranks via `MpiPoolSession`; rank0 `mem_available=440 GiB` → local quota 220 GiB; rank1 `mem_available=880 GiB` → local quota 440 GiB | `local_quotas[0] != local_quotas[1]` (pre-sync); `len(set(synced_quotas))==1`; `synced_quotas[0]==min(local_quotas)`; `synced_quotas[0]==local_quotas[0]` | Only test running the sizing/sync pipeline across more than one instance — but the two instances are **two MPI ranks of the same logical tier** (regression test for PR #17380/TRTLLM-15179 hang), reconciled via `allreduce(MIN)`. Rank-to-rank sync within one manager type, not target-vs-draft-manager. |

### C.2 — Does any test simulate two sequential managers (target-then-draft) and compare quotas?

**Verified absent.** Search patterns used:
- `grep -rn "_compute_auto_host_tier_quota" tests/unittest/` → only in `test_kvv2_host_tier_sizing.py`
  (L15, 27, 39, 49, 58, 67, 82, 94, 114, 125).
- `grep -rn "_sync_host_tier_quota" tests/unittest/` → only in the same file (L16, 115, 132, 144, 155).
- `grep -rln "draft" tests/unittest/_torch/executor/` → many files, none reference either function
  (cross-checked against the two greps above — no overlap).
- `grep -rn "host_tier" tensorrt_llm/ tests/unittest/` → only `kv_cache_manager_v2.py` (source, L242,
  282, 285, 1144-1145) and the one test file; no other test references host-tier sizing at all.

Every test either (a) calls `_compute_auto_host_tier_quota` in isolation with hand-picked scalars for a
single rank/manager (`TestComputeAutoHostTierQuota`, L23-101), or (b) exercises
`_sync_host_tier_quota`'s cross-**rank** MPI reconciliation for one manager instance replicated across
ranks (`TestSyncHostTierQuota`, L136-181). **No test constructs two `KVCacheManagerV2` instances (one
simulating target, one draft) in the same process/rank and compares their `host_quota` values.** No test
asserts equality, bounded difference, or a sum constraint between a target-model manager's host quota and
a draft-model manager's host quota.

### C.3 — Does source code compare one manager's host-tier quota against another's?

**Verified absent.**

- `_compute_auto_host_tier_quota` (`kv_cache_manager_v2.py:242-279`) is a pure function of scalar args
  (`quota, local_ranks, mem_available, memlock_limit`) — no manager-instance references, let alone
  another manager's quota.
- `_sync_host_tier_quota` (`kv_cache_manager_v2.py:282-304`) takes only `(host_quota: int, mapping:
  Mapping)`; when `mapping.world_size > 1`, does
  `Distributed.get(mapping).allreduce(host_quota, op=ReduceOp.MIN)` (L302-303) — a cross-**rank**
  collective over the *same* logical tier, not a cross-manager (target vs. draft) comparison.
- Call site in `KVCacheManagerV2.__init__` (`kv_cache_manager_v2.py:1105-1150`, esp. L1124-1145):
  `local_ranks`, `mem_available = os.sysconf(...)` are read fresh from OS state at *this* manager's own
  construction time; `host_quota` is computed/synced using only this manager's own `mapping`/local
  state. No `self.other_manager`, no reading a previously-constructed manager's `host_quota`, no shared
  registry/coordinator object.
- `grep -rn "draft_kv_cache_manager\|target_kv_cache_manager\|self.draft.*manager\|self.target.*manager"
  tensorrt_llm/_torch/pyexecutor/*.py | grep -i "host\|quota"` → no matches.
- `grep -rln "KVCacheManagerV2(" tensorrt_llm/_torch/` → only `kv_cache_manager_v2.py` (class def) and
  `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py`; neither contains
  cross-manager quota comparison (the minimax file constructs its own manager independently).
- `find . -iname "scheduler_v2.py"` → `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`;
  `grep -n "host_tier\|host_quota\|quota"` → **zero matches**. The scheduler contains no references to
  host-tier quota at all.

**Bottom line:** each `KVCacheManagerV2` instance (target or draft) independently snapshots live host
memory state and computes/syncs its own host-tier quota purely via cross-rank MPI reduction. No code path
anywhere in searched source reads or compares one manager's host-tier quota against a *different* manager
instance's quota, and no test simulates or asserts on such a relationship.

---

## Summary Table

| Question | Answer | Classification | Evidence |
|---|---|---|---|
| A.1 — validator blocking `max_tokens` + two-model V2 | No such validator exists | Verified absent | `llm_args.py:4297-4409`, `5928-6013`, `6410`, `6428`; `_util.py:2078-2085` |
| A.2 — do target/draft quotas diverge despite equal `max_gpu_total_bytes`? | Yes, when `max_tokens` is also set and per-layer byte costs differ (worked example: 12.85 GiB vs 1.61 GiB, 8× divergence for 32 vs 4 layers) | Verified current behavior | `kv_cache_manager_v2.py:1056-1102,1589-1600,1646-1685,3837-3884,232-239,901`; `llm_args.py:4174-4181` |
| A.3 — existing test for two-model V2 + explicit `max_tokens` | None found | Verified absent | `test_kv_cache_budget_split.py:36-77,625-663`; `test_dual_pool_kv_cache.py:175,259,276,420,606` |
| B.1 — test for two-model draft/target with mismatched `pool_ratio`/layer counts | None found | Verified absent | `test_kv_cache_manager_v2.py:3413-3428` (single-manager only); `test_kv_cache_estimation.py:979-1033` (one-model only) |
| B.2 — does wrong-length `pool_ratio` reach native arity check uncaught? | Yes — no Python or native pre-check normalizes/truncates it; surfaces as `ValueError` from `StorageManager` ctor | Verified current behavior | `_util.py:2072-2089,1360-1405,2198-2224`; `kv_cache_manager_v2.py:2090-2107`; `kvCacheManagerV2.cpp:1612-1665,801-849`; `kvCacheManager.cpp:104-122`; `config.cpp:40-88`; `storageManager.cpp:296-315` |
| B.3 — is draft attention-window indexing safe/correct for differing layer counts? | Safe (never out-of-bounds, modulo-guarded) but semantically questionable — indexes draft's own layer position into the target's raw window pattern with no draft-specific derivation | Verified current behavior (mechanism); Source-inconclusive (whether it manifests wrong values for any shipped model pair — would need execution) | `kv_cache_manager_v2.py:2079-2088,1589-1600,935-952,838-843`; `_util.py:2072-2089,2242-2250,2335,513-536,1492,1499-1514,1535`; `resource_manager.py:193-213` |
| B.4 — test for draft attention-window correctness (two-model, differing layers) | None found; only one-model `_derive_draft_max_attention_window` unit tests exist | Verified absent | `test_eagle3.py:188-236` |
| C.1 — host-tier sizing test coverage | 9 tests, all single-rank-scalar or cross-**rank** MPI sync of one manager type; none simulate target+draft managers | Verified current behavior | `test_kvv2_host_tier_sizing.py:1-181` (full read) |
| C.2 — test comparing target vs. draft host quotas | None exists | Verified absent | grep of `_compute_auto_host_tier_quota`/`_sync_host_tier_quota` across `tests/unittest/` |
| C.3 — source code comparing one manager's host quota to another's | None exists | Verified absent | `kv_cache_manager_v2.py:242-279,282-304,1105-1150`; `scheduler/scheduler_v2.py` (zero `quota`/`host_tier` matches) |
