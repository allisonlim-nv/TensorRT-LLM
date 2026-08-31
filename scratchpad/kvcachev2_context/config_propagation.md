# KVCacheV2 Configuration/Memory Propagation Into Target vs. Draft Caches

Repo state: commit `4716843cee6e7a6c08bf4d8be29fae25321a9344`, branch
`feat/native-kv-events-clean`, 2026-08-31. Read-only audit; no code executed.

Files read this session:
- `tensorrt_llm/_torch/pyexecutor/_util.py` (full `KvCacheCreator` class, lines ~1-2160,
  1290-2160, 1620-1850)
- `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` (lines 220-560, 960-1180,
  1595-1660, 2000-2110)
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp` (grep only,
  lines ~298-312)

Scope note: this pass goes deep on the 7 dimensions below only; it does not repeat the
general manager-construction survey from prior artifacts.

## Summary Table

| # | Dimension | Target computation | Draft (one-model) | Draft (two-model) | Classification |
|---|-----------|--------------------|--------------------|--------------------|-----------------|
| 1 | GPU memory (`max_gpu_total_bytes`/quota) | `configure_kv_cache_capacity` always sets `max_gpu_total_bytes` (`_util.py:1355`); manager's own `__init__` independently computes `quota` via `_get_quota_from_max_tokens`+allreduce (`kv_cache_manager_v2.py:1056-1102`) | Budget is **affinely split** between target/draft via `_split_kv_cache_budget_for_draft("max_gpu_total_bytes", ...)` (`_util.py:2043-2047`, `1638-1740`); draft gets its own (generally smaller) `max_gpu_total_bytes` and independently runs the same quota/allreduce logic | Budget is **required equal** — hard `assert` at `_util.py:2081-2085`; GPU split is *not* applied (`_needs_gpu_kv_cache_budget_split` returns `_should_create_separate_draft_kv_cache()`, which is one-model-only, `_util.py:1963-1970`). Draft manager still independently computes its own `quota` inside its own `__init__` from the (equal) `max_gpu_total_bytes`, and independently from `max_tokens` if the user sets it | Intentional policy for two-model (explicit assert + comment); intentional-but-different policy for one-model (affine split). See "Open Questions" for a real divergence path via `max_tokens`. |
| 2 | Host/disk tier | Explicit `host_cache_size`/`disk_cache_size` split once in `_util.py` via same `_split_kv_cache_budget_for_draft` helper (unconditional loop over `_OFFLOAD_TIER_BUDGET_ATTRS` whenever `has_draft`, `_util.py:2048-2052`); if unset, each manager auto-sizes via `_compute_auto_host_tier_quota` inside its own `__init__` | Explicit sizes: split once (shared affine model). Auto sizing (host_cache_size=None): computed **independently per manager**, sequentially (target built first at `_util.py:2054`, draft at `2092`), each snapshotting its own `mem_available`/`local_ranks` (`kv_cache_manager_v2.py:1105-1145`) | Same as one-model for auto sizing: independently invoked per manager, sequentially, with potentially different rank-local mem snapshots | Explicit sizing: **intentional policy** (single split, documented rationale at `_util.py:1656-1668`). Auto sizing: **proven propagation risk** — two independent, time-separated invocations of `_compute_auto_host_tier_quota`/`_sync_host_tier_quota` reading live host memory state; see dimension detail. |
| 3 | Attention window (VSWA) | Derived from target's own pretrained config / `kv_cache_config.max_attention_window` | `_derive_draft_max_attention_window` (`_util.py:513-536`) derives per-layer windows from the draft's own HF config when it "defines its own attention layout" (`draft_config_defines_attention_layout`, `_util.py:495-510`), else falls back to inheriting the target's `max_attention_window` list (non-VSWA) or `None` (VSWA, target-defined) | Not separately modeled for two-model in `_util.py` beyond the general `_get_model_kv_cache_manager_cls`/config path — two-model draft uses its own `kv_cache_config` copy but no `_derive_draft_max_attention_window`-style call was found for the two-model branch | Intentional policy (explicit docstring at `_util.py:499-505` justifying divergence when draft config defines its own layout) |
| 4 | Layer-group / `pool_ratio` | `kv_cache_config.pool_ratio` (user-set list or `None`), consumed as `initial_pool_ratio` in `kv_cache_manager_v2.py:2106` | Explicit reset-to-`[1.0]` logic at `_util.py:1539-1551`, gated on `not uses_vswa AND pool_ratio is not None AND len(pool_ratio) != 1` | **No equivalent reset exists.** Two-model draft config is a `model_copy()` of the (possibly VSWA/hybrid) target config; `pool_ratio` passes through unchanged into the generic `_create_kv_cache_manager` path (`_util.py:2086-2089`) | One-model: intentional policy (explicit log message + comment). Two-model: **unresolved / needs runtime evidence** — native code (`storageManager.cpp:298-312`) throws `std::invalid_argument` if `initial_pool_ratio.size() != numLifeCycles()`, so an arity mismatch between the inherited target `pool_ratio` and the draft model's own layer-group count would hard-fail at construction, not silently corrupt state. Whether this is reachable depends on whether any two-model spec-decoding config combines a VSWA/hybrid target with a differently-grouped draft; not confirmed from source alone. |
| 5 | Layer count / `layer_mask` / `num_draft_layers` | N/A (target uses config-derived `num_hidden_layers`) | `_get_num_draft_layers()` (`_util.py:1475-1483`) computed twice per build: once inside `_get_one_model_draft_layer_mask()` (`_util.py:715-722`) to build the mask, once directly at `_util.py:1526` for the `num_layers=` kwarg passed to the constructor (`_util.py:1590`) | N/A (two-model draft uses its own model's native layer count, no mask) | **Verified fact, not a bug today**: the "must stay in sync" comment (`_util.py:1478-1479`) is *not* backed by any runtime assert. `_create_kv_cache_manager` (`_util.py:2330-2333`) gives `num_layers` (when provided) unconditional priority over `sum(layer_mask)` — if the two ever diverged, `num_layers` would silently win with no error. Currently both derive from the same `_get_num_draft_layers()` call, so no divergence is reachable in this commit; classified **intentional-but-fragile / needs no active bug report**, not a proven bug. |
| 6 | Buffer/role/dtype (`BufferConfig`, K/V/scale) | Computed per-manager from that manager's own `quant_config`/`kv_cache_dtype`/`head_dim_per_layer` in `_build_base_config` (`kv_cache_manager_v2.py:2040-2107`); extra NVFP4 scale buffers (`Role.KEY_BLOCK_SCALE`/`VALUE_BLOCK_SCALE`) added when `kv_cache_config.dtype == "nvfp4"` (`kv_cache_manager_v2.py:2043-2050`) | Fully independent: draft manager built from `effective_draft_config` (`_util.py:1531,1586-1588`), its own `dtype`/`quant_config`, so buffer roles/sizes reflect the draft model's own quantization, not the target's | Fully independent: draft engine has its own `model_config`/`quant_config` (`_util.py:745-750`) | **Intentional policy** — each `KVCacheManagerV2` instance derives buffers solely from its own model's `quant_config`/dtype; there is no shared/copied buffer config between managers by design. |
| 7 | `tokens_per_block` | Single `self._tokens_per_block` value stored once in `KvCacheCreator.__init__` (`_util.py:585`) | Same single value passed unmodified to every `_create_kv_cache_manager` call site: target (`_util.py:1388`), one-model draft (`_util.py:1573`), cross (`_util.py:1946`) | Same single value reused for the two-model draft build (goes through the generic `_create_kv_cache_manager` with the same `self._tokens_per_block`) | **Intentional policy / structurally guaranteed** — no code path in `_util.py` ever assigns a different `tokens_per_block` to a draft manager; it is a single scalar threaded through the whole `KvCacheCreator`. No cross-manager equality assert exists because none is needed (no divergence is possible from this source). |

## 1. GPU memory (`max_gpu_total_bytes` / quota)

### The hard equality assert

`_util.py:2076-2085` (read with full surrounding context 2003-2109):

```python
draft_kv_cache_manager = None
draft_build_kv_cache_config = (draft_kv_cache_config
                               if draft_kv_cache_config is not None else
                               self_kv_cache_config)

# Two-model speculative decoding: draft model has separate engine
if self._draft_model_engine is not None:
    if (self._is_kv_cache_manager_v2
            and draft_kv_cache_config is not None):
        # Offload budgets are divided per manager, GPU budgets are not.
        assert (draft_kv_cache_config.max_gpu_total_bytes ==
                self_kv_cache_config.max_gpu_total_bytes), (
                    "KVCacheManagerV2 does not support two-model "
                    "speculative decoding with separate draft GPU "
                    "budgets.")
    draft_kv_cache_manager = self._create_kv_cache_manager(
        self._draft_model_engine,
        estimating_kv_cache,
        kv_cache_config_override=draft_build_kv_cache_config)
# One-model speculative decoding with different KV layouts
elif self._should_create_separate_draft_kv_cache():
    draft_kv_cache_manager = self._create_one_model_draft_kv_cache_manager(
        original_max_seq_len,
        estimating_kv_cache,
        kv_cache_config_override=draft_build_kv_cache_config)
```

**Verified fact**: the equality assert exists **only** in the `self._draft_model_engine is
not None` branch, i.e. only for **two-model** speculative decoding. The one-model branch
(`elif self._should_create_separate_draft_kv_cache()`) has no such assert.

### Why: GPU-split gating differs by variant

`_util.py:1963-1979` (`_needs_gpu_kv_cache_budget_split`):

```python
def _needs_gpu_kv_cache_budget_split(
    self,
    max_seq_len: int,
    kv_cache_config: Optional[KvCacheConfig] = None,
) -> bool:
    """Whether max_gpu_total_bytes must be split per manager."""
    if self._is_kv_cache_manager_v2:
        return self._should_create_separate_draft_kv_cache()
    ...
```

`_should_create_separate_draft_kv_cache()` (`_util.py:1432-1456`) is the flag for
**one-model** separate-draft-cache mode (`should_use_separate_draft_kv_cache(spec_config)`);
it is unrelated to whether a separate `draft_model_engine` exists. Consequently, for V2:

- **Two-model**: `_needs_gpu_kv_cache_budget_split` returns `False` (the one-model flag is
  not set), so `build_managers` (`_util.py:2036-2047`) never calls
  `_split_kv_cache_budget_for_draft("max_gpu_total_bytes", ...)`. `draft_kv_cache_config`
  only gets created later (or not) by the *offload-tier* split loop
  (`_util.py:2048-2052`), which — per `_split_kv_cache_budget_for_draft`'s own doc
  (`_util.py:1656-1668`) — only ever mutates `host_cache_size`/`disk_cache_size` on the
  clone, leaving `max_gpu_total_bytes` copied verbatim from `self_kv_cache_config`. This is
  exactly why the equality assert always holds by construction today (**Verified fact**).
- **One-model** (separate draft KV cache, V2): `_needs_gpu_kv_cache_budget_split` returns
  `True`, so `build_managers` *does* call
  `_split_kv_cache_budget_for_draft("max_gpu_total_bytes", self_kv_cache_config, None)`
  (`_util.py:2043-2047`), which affinely divides the shared budget between target and
  draft using `_get_target_and_draft_cache_costs` (`_util.py:1594-1615`) and
  `_compute_draft_budget_shares` (`_util.py:1617-1636`). Target and draft therefore get
  **different, generally unequal** `max_gpu_total_bytes` values — this is the deliberate
  design for one-model mode (comment at `_util.py:1656-1668`: "Splits the value of
  `target_kv_cache_config.<budget_attr>` using the affine target/draft cache costs").

### Does each manager compute its own quota, or inherit a precomputed value?

**Verified fact**: every `KVCacheManagerV2` instance — target or draft, one-model or
two-model — independently runs the full quota-derivation block inside its own `__init__`
(`kv_cache_manager_v2.py:1056-1102`, read in full):

```python
quota = sys.maxsize
if (kv_cache_config.max_gpu_total_bytes is not None
        and kv_cache_config.max_gpu_total_bytes > 0):
    quota = int(kv_cache_config.max_gpu_total_bytes)
    ...
if kv_cache_config.max_tokens is not None:
    quota_from_max_tokens = int(math.ceil(
        self._get_quota_from_max_tokens(kv_cache_config.max_tokens)
        / max_util_for_resume))
    quota = min(quota, quota_from_max_tokens)
    ...
assert quota < sys.maxsize, (...)
if mapping.world_size > 1:
    ... resumable_quota = int(quota * max_util_for_resume)
    max_tokens = self._get_max_tokens_from_quota(resumable_quota)
    max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
    ...
```

`quota_from_max_tokens` is computed via `self._get_quota_from_max_tokens_impl`
(`kv_cache_manager_v2.py:1656` ff.), which reads `self._get_runtime_cache_size_layer_components()`
— i.e. **that manager's own** per-layer byte sizes. This is architecture-specific
(different `head_dim`, `num_kv_heads`, quant dtype, layer count between target and draft
models). So: the draft manager does **not** inherit a precomputed byte quota; it
independently re-derives `quota` (and separately performs its own cross-rank
`allreduce(MIN)`) from the config it was handed. `max_gpu_total_bytes` (the config field)
is what's kept equal by the assert for two-model — the *derived* `quota` inside each
manager can still diverge if `kv_cache_config.max_tokens` is also set (see Open
Questions).

**Classification**: Intentional policy for both variants regarding *which config field* is
shared vs. split (two-model: `max_gpu_total_bytes` equal by assert; one-model:
`max_gpu_total_bytes` affinely split — both are explicit, commented design choices). The
per-manager independent quota derivation is also intentional (each manager owns its
device-quota bookkeeping and cross-rank sync). See Open Questions for one reachable
divergence caveat under `max_tokens`.

## 2. Host tier and disk tier

### Explicit sizes: split once, shared result

`_util.py:2048-2052` inside `build_managers`:

```python
for budget_attr in self._OFFLOAD_TIER_BUDGET_ATTRS:
    self_kv_cache_config, draft_kv_cache_config = (
        self._split_kv_cache_budget_for_draft(
            budget_attr, self_kv_cache_config,
            draft_kv_cache_config))
```

`_OFFLOAD_TIER_BUDGET_ATTRS = ("host_cache_size", "disk_cache_size")` (`_util.py:546`).
This loop runs unconditionally whenever `has_draft` is true (`_util.py:2032-2034`,
covering both one-model separate-draft-cache and two-model), **regardless of** whether the
GPU budget itself was split. `_split_kv_cache_budget_for_draft` (full body read,
`_util.py:1638-1740`) is a no-op when the attribute is unset/zero (`total_budget <= 0`
early-return, `_util.py:1673-1675`), and otherwise computes a single affine split (same
`CacheCost` machinery as the GPU split) and returns two config clones. **This is a single
computation, applied once, in Python, before either manager is constructed** — not
recomputed per manager. Classified **intentional policy**: explicit docstring
(`_util.py:1656-1668`) documents "the intercept is dropped so the split stays
proportional to the per-token cost" for non-GPU budgets.

### Auto-sizing (host_cache_size unset): computed independently, twice, sequentially

When `host_cache_size` is `None` (the common case — no explicit split value flows through,
since the split loop above only mutates budgets that are `> 0`), each `KVCacheManagerV2`
instance falls into its own auto-sizing branch inside `__init__`
(`kv_cache_manager_v2.py:1105-1145`, full text read):

```python
if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size >= 0:
    host_quota = kv_cache_config.host_cache_size
else:
    ...
    local_ranks = max(1, Distributed.get(mapping).local_world_size)
    try:
        mem_available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        mem_available = float("inf")
    try:
        _soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        memlock_limit = _soft if _soft != resource.RLIM_INFINITY else float("inf")
    except (ValueError, OSError):
        memlock_limit = float("inf")
    host_quota = _compute_auto_host_tier_quota(
        quota, local_ranks, mem_available, memlock_limit)
    ...
    host_quota = _sync_host_tier_quota(host_quota, mapping)
```

`_compute_auto_host_tier_quota` (`kv_cache_manager_v2.py:242-279`, full body read) takes
`quota` (this manager's own GPU device quota, computed as in dimension 1), `local_ranks`,
and a **live snapshot** of `os.sysconf("SC_AVPHYS_PAGES")` (available physical memory) and
`RLIMIT_MEMLOCK` taken at the moment that specific manager's `__init__` runs.

**Verified fact — sequential construction order**: `build_managers` constructs the target
manager first (`kv_cache_manager = self._create_kv_cache_manager(self._model_engine, ...)`,
`_util.py:2054-2057`), and only afterward constructs the draft manager
(`_util.py:2086-2089` two-model, or `_util.py:2091-2095` one-model). Because the target
manager's construction actually **allocates and page-locks host memory** for its own auto
host tier (per the module comment at `_util.py:542-546`: "the host tier is prefaulted and
page-locked... Every live manager reserves its own"), the draft manager's later
`os.sysconf("SC_AVPHYS_PAGES")` read at `kv_cache_manager_v2.py:1127` will observe **less**
available host memory than the target manager did, purely due to ordering. This means
`_compute_auto_host_tier_quota` is invoked twice with different `mem_available` snapshots
(and potentially different `quota` inputs — equal for two-model, unequal by design for
one-model).

There is a partial mitigation: `_sync_host_tier_quota` (`kv_cache_manager_v2.py:282-304`,
full body read) performs an `allreduce(MIN)` across **ranks** for a *single* manager's
computed quota, explicitly to avoid **cross-rank** divergence ("Divergent host-tier
retention makes per-rank MAX_UTILIZATION schedulers disagree... wedges collectives").
This mechanism synchronizes the same manager's quota across TP/PP/DP ranks; it does **not**
synchronize target-vs-draft quotas within the same rank, since those are two different
manager instances constructed sequentially with genuinely different `quota` and
`mem_available` inputs, not two views of the same computation.

**Classification**: **Proven propagation/splitting-adjacent risk**, not a hard bug — this
is a real, reachable divergence: two independently-timed reads of live host memory state
feeding two independently-sized host-cache tiers for target and draft. Whether this
constitutes a functional problem depends on whether the two managers' resulting
`host_quota` values need to be equal for correctness; no equality invariant or assert was
found for host tier sizing between target and draft (unlike the GPU-quota case, which has
an explicit assert for two-model). The code's own rationale for `_sync_host_tier_quota`
(cross-rank sync avoids wedged collectives) does not extend to target/draft, suggesting
this asymmetry either (a) was not considered a problem because target/draft managers don't
participate in the same collective-scheduling decisions, or (b) is an unaddressed gap.
Source alone cannot determine which — see Open Questions.

Disk tier (`disk_cache_size`) has no auto-sizing branch at all (`kv_cache_manager_v2.py:1151-1160`
only handles the explicit `> 0` case), so it is not subject to this divergence — it is
either explicitly split once (dimension above) or absent.

## 3. Attention window (`max_attention_window`, VSWA)

### `_derive_draft_max_attention_window` — full logic

`_util.py:495-536` (full function bodies read):

```python
def draft_config_defines_attention_layout(
    draft_pretrained_config: object, ) -> bool:
    """Return whether the draft HF config explicitly defines its attention layout.

    A ``True`` result makes the draft settings authoritative, including an
    explicit full-attention layout. For example, a config with
    ``use_sliding_window=False`` and ``sliding_window=4096`` returns ``True``:
    its layers should attend to ``max_seq_len`` instead of inheriting the
    target model's window. A config that provides none of
    ``use_sliding_window``, ``sliding_window``, or ``layer_types`` returns
    ``False`` so the legacy uniform-target fallback can be used.
    """
    return (
        getattr(draft_pretrained_config, "use_sliding_window", None) is not None
        or getattr(draft_pretrained_config, "sliding_window", None) is not None
        or bool(getattr(draft_pretrained_config, "layer_types", None)))


def _derive_draft_max_attention_window(
    kv_cache_config: KvCacheConfig,
    draft_pretrained_config: object,
    max_seq_len: int,
    num_draft_layers: int,
) -> Optional[List[int]]:
    layer_windows = [
        get_layer_attention_window(draft_pretrained_config, layer_idx)
        for layer_idx in range(num_draft_layers)
    ]
    if draft_config_defines_attention_layout(draft_pretrained_config):
        draft_windows = [
            max_seq_len if window is None else window
            for window in layer_windows
        ]
        return _normalize_attention_windows(draft_windows, max_seq_len)

    if not uses_vswa_kv_cache_layout(kv_cache_config.max_attention_window):
        max_attention_window = kv_cache_config.max_attention_window
        if max_attention_window is None:
            return None
        return _normalize_attention_windows(max_attention_window, max_seq_len)

    return None
```

Three cases, all documented by the code's own comments:

1. **Draft HF config defines its own attention layout** (`use_sliding_window`,
   `sliding_window`, or `layer_types` present) → the draft's own per-layer windows are
   authoritative, independent of the target's window. This is **explicitly intentional**
   per the docstring at `_util.py:499-505` (example given: draft with
   `use_sliding_window=False` gets full-attention windows even if target is VSWA).
2. **Target is non-VSWA and draft doesn't define its own layout** → draft inherits the
   target's single (or list-but-non-VSWA) `max_attention_window` verbatim. Intentional
   "legacy uniform-target fallback" (docstring, `_util.py:503-505`).
3. **Target is VSWA and draft doesn't define its own layout** → returns `None`, meaning
   the caller falls back to no explicit override (the draft manager gets whatever its own
   default/config-derived window is, not the target's VSWA pattern). This is the case
   where "target and draft windows could legitimately differ" per the task's framing.

### Caller

`_get_draft_max_attention_window` (`_util.py:1485-1497`, full body read):

```python
def _get_draft_max_attention_window(
    self, max_seq_len: int, kv_cache_config: KvCacheConfig,
) -> Optional[List[int]]:
    """Derive the draft manager's per-layer attention windows."""
    effective_draft_config = self._get_effective_draft_config()
    return _derive_draft_max_attention_window(
        kv_cache_config,
        effective_draft_config.pretrained_config,
        max_seq_len,
        self._get_num_draft_layers(),
    )
```

`effective_draft_config` for MTP falls back to the **target's own** `model_config`
(`_get_effective_draft_config`, `_util.py:1458-1473`: "MTP layers reuse the target model
architecture, so the target model's config describes the correct KV cache layout for the
draft layers as well"). For MTP specifically, `draft_config_defines_attention_layout`
therefore inspects the *target's* HF config fields — so an MTP draft essentially always
inherits the target's windows or falls into the same-VSWA-`None` case, by construction
(**intentional**, since draft layers are literally part of the same architecture).

`_get_one_model_draft_kv_cache_config` (`_util.py:1499-1514`, full body read) calls this
only when **not** estimating (`estimating_kv_cache=False`), with a documented reason:
"Estimation uses a small max_tokens-sized temporary draft cache before the measured GPU
budget is available to split. Applying VSWA there would size every window pool from the
unsplit free-memory budget." During estimation the draft manager gets
`max_attention_window=None` explicitly.

This one-model-specific path (`_derive_draft_max_attention_window`,
`_get_draft_max_attention_window`, `_get_one_model_draft_kv_cache_config`) is used only
from `_create_one_model_draft_kv_cache_manager` (`_util.py:1516-1592`, one-model). For
**two-model**, no call site of `_derive_draft_max_attention_window` or
`_get_draft_max_attention_window` was found; the two-model draft build
(`_util.py:2086-2089`) passes `draft_build_kv_cache_config`, which — absent a GPU/window
split for two-model — is either `self_kv_cache_config` (target's config, including its
`max_attention_window`) or an offload-only clone of it. **Verified fact**: the two-model
draft manager's `max_attention_window` field, as handed to `_create_kv_cache_manager`, is
the *target's own* `kv_cache_config.max_attention_window` (untouched), not a
draft-architecture-derived value. Whatever the draft's actual per-layer windows end up
being is then whatever `_create_kv_cache_manager`'s generic layer-count/`config`-derived
logic computes from the draft model's own `model_config.pretrained_config` inside the
manager's own `_get_static_cache_size_layer_components`-style code
(`kv_cache_manager_v2.py:355-411`, which reads `kv_cache_config.max_attention_window`
combined with the draft's own layer count) — i.e. the *shape* of the window list is still
sourced from the target-oriented config, applied to the draft's own layer count via
modulo indexing (`kv_cache_manager_v2.py:399-408`: `window_pattern[layer_idx %
len(window_pattern)]`). This modulo-reuse is a generic mechanism used for all managers
(not draft-specific), so it is **not** unique inconsistency-inducing logic for draft vs.
target, but it does mean a two-model draft with a different natural window pattern than
the target has no dedicated derivation step analogous to `_derive_draft_max_attention_window`.

**Classification**: One-model divergence cases 1–3 above are **intentional policy**,
explicitly justified in code comments/docstrings. Two-model window handling is
**unresolved / needs further evidence** — no dedicated draft-window derivation function
was found for the two-model path in `_util.py`; window shaping for two-model draft relies
on the generic modulo-based window-pattern application inside `kv_cache_manager_v2.py`,
which was not exercised/verified against real two-model VSWA-draft scenarios in this pass.

## 4. Layer-group / `pool_ratio` configuration

### Target

`kv_cache_config.pool_ratio` — user-set list or `None` — flows straight into
`initial_pool_ratio=kv_cache_config.pool_ratio` in `_build_base_config`
(`kv_cache_manager_v2.py:2090-2107`, read in full above). During KV-cache **estimation**,
`pool_ratio` is forced to `None` for the shared config:
`_util.py:1077-1080` ("User-provided pool sizing can underprovision the temporary
estimation cache and cause warmup to hang or fail. Override it for estimation, then
restore it in `configure_kv_cache_capacity()`"), and restored at `_util.py:1265`
(`self._kv_cache_config.pool_ratio = self._pool_ratio_in`) inside
`configure_kv_cache_capacity`.

### One-model draft: the `[1.0]` reset — full logic

`_create_one_model_draft_kv_cache_manager` (`_util.py:1516-1592`, full body read),
specifically `_util.py:1535-1556`:

```python
draft_kv_config = self._get_one_model_draft_kv_cache_config(
    kv_cache_config,
    max_seq_len,
    estimating_kv_cache=estimating_kv_cache)
if (not uses_vswa_kv_cache_layout(draft_kv_config.max_attention_window)
        and draft_kv_config.pool_ratio is not None
        and len(draft_kv_config.pool_ratio) != 1):
    # pool_ratio describes one manager's layer-group layout. The
    # target hybrid manager may have separate recurrent-state and
    # attention layer groups, while a non-VSWA draft manager has one
    # attention layer group. Reusing the target's ratios fails its arity
    # check.
    logger.info(
        "Normalizing the separate one-model draft KV cache pool_ratio "
        f"from {draft_kv_config.pool_ratio} to [1.0] for its single "
        "layer group.")
    draft_kv_config.pool_ratio = [1.0]
if uses_vswa_kv_cache_layout(draft_kv_config.max_attention_window):
    logger.info(
        f"Derived draft KV cache max_attention_window for separate "
        f"draft manager: {draft_kv_config.max_attention_window}")
```

**Precise conditions for the reset to apply** (all three must hold):
1. `draft_kv_config.max_attention_window` is **not** VSWA-shaped (i.e. the draft ends up
   with a single uniform window, or no window override) — because if it *is* VSWA, the
   draft legitimately has multiple layer groups too and the target's `pool_ratio` list
   might (by coincidence or not) still be valid for it; the code does not reset in that
   case.
2. `draft_kv_config.pool_ratio is not None` — i.e. the target actually had an explicit
   `pool_ratio` (inherited via the `model_copy` in `_get_one_model_draft_kv_cache_config`,
   `_util.py:1499-1514`, which copies the full target `kv_cache_config` and only
   overrides `max_attention_window`).
3. `len(draft_kv_config.pool_ratio) != 1` — the inherited ratio list has more than one
   entry (implying the target has multiple layer groups, e.g. hybrid Mamba+attention or
   VSWA).

When all three hold, the ratio list is force-reset to `[1.0]` (single group, full
allocation). **What the code's own reasoning says would go wrong if skipped**: the comment
states "Reusing the target's ratios fails its arity check" — i.e. the draft manager (one
attention-only layer group) would receive a multi-element `pool_ratio` sized for the
target's (possibly hybrid) layer-group count, and (per the native check found at
`storageManager.cpp:298-312`, `if (initialPoolRatio->size() != toSizeT(numLifeCycles()))
throw std::invalid_argument("initial_pool_ratio length must match number of layer
groups")`) construction would throw. So skipping this reset would be a **construction-time
crash** for exactly the case it guards against (non-VSWA one-model draft + multi-group
target `pool_ratio`), not a silent correctness bug.

When it does **not** apply: VSWA draft (own multi-group layout, not overridden), or
`pool_ratio is None` (default, single implicit group, nothing to normalize), or
`len == 1` already (already correctly shaped).

### Two-model draft: no equivalent normalization

The two-model draft build goes through the **generic** `_create_kv_cache_manager`
(`_util.py:2086-2089`), not `_create_one_model_draft_kv_cache_manager`. There is no
pool_ratio-normalization step in the two-model code path — `draft_build_kv_cache_config`
(target's config, or an offload-tier-only clone of it) carries the target's `pool_ratio`
verbatim into the draft manager's `_build_base_config`/`initial_pool_ratio`.

**Classification**: One-model reset — **intentional policy**, code's own justification
present and specific ("fails its arity check"). Two-model — **unresolved / needs native or
runtime evidence**: the native arity check (`storageManager.cpp:298-312`) would reject a
mismatched `pool_ratio` at construction (fail-fast, not silent corruption), so this is not
a "proven propagation bug" in the sense of silently-wrong behavior, but it is a gap: no
Python-side normalization exists for two-model draft managers with a differently-grouped
architecture than the target. Whether this is reachable in practice depends on whether any
supported two-model configuration pairs a multi-group (hybrid/VSWA) target with a
single-group draft (or vice versa) and an explicit `pool_ratio`; this was not confirmed
from source alone in this pass.

## 5. Layer count / `layer_mask` / `num_draft_layers`

### `_get_num_draft_layers` — full logic

`_util.py:1475-1483` (full body read):

```python
def _get_num_draft_layers(self) -> int:
    """Return the actual number of draft KV cache layers.

    This must stay in sync with the num_layers passed to the draft KV
    cache manager constructor in _create_one_model_draft_kv_cache_manager.
    """
    if self._speculative_config.spec_dec_mode.is_external_drafter():
        return self._draft_config.pretrained_config.num_hidden_layers
    return get_num_spec_layers(self._speculative_config)
```

Two branches: external-drafter mode reads the draft HF config's own layer count directly;
all other one-model spec-dec modes (EAGLE3, MTP, draft-target) use
`get_num_spec_layers(spec_config)` (imported from `..speculative`, not further traced in
this pass — treated as authoritative for "how many draft layers exist at runtime",
matching the comment at `_util.py:753-755`: "the HF config may report a different layer
count than what is actually used at runtime (e.g. EAGLE3: config says 1, runtime uses 4)").

### `layer_mask` construction

`_get_one_model_draft_layer_mask` (`_util.py:715-722`, full body read):

```python
def _get_one_model_draft_layer_mask(self) -> List[bool]:
    """Return the same draft-only mask used by runtime construction."""
    num_draft_layers = self._get_num_draft_layers()
    if self._speculative_config.spec_dec_mode.is_external_drafter():
        return [True] * num_draft_layers
    target_num_layers = (self._model_engine.model.model_config.
                         pretrained_config.num_hidden_layers)
    return [False] * target_num_layers + [True] * num_draft_layers
```

Called from `_create_one_model_draft_kv_cache_manager` at `_util.py:1527`
(`spec_dec_layer_mask = self._get_one_model_draft_layer_mask()`), which **separately**
calls `_get_num_draft_layers()` again at `_util.py:1526`
(`num_draft_layers = self._get_num_draft_layers()`) to pass as the constructor's
`num_layers=` kwarg (`_util.py:1590`).

### Is the "must stay in sync" invariant enforced?

`_create_kv_cache_manager` (`_util.py:2198-2340`, region 2327-2335 read in full):

```python
# Use provided num_layers if available, otherwise use config.
# When layer_mask is set (e.g., KV sharing), num_layers for the cache
# manager must equal the number of enabled (True) layers in the mask.
if num_layers is not None:
    num_hidden_layers = num_layers
elif layer_mask is not None:
    num_hidden_layers = sum(layer_mask)
else:
    num_hidden_layers = config.num_hidden_layers
```

**Verified fact**: when **both** `num_layers` and `layer_mask` are provided (exactly the
one-model draft call, `_util.py:1589-1590`), `num_layers` **unconditionally wins** — there
is no comparison, no assert, no warning that `num_layers == sum(layer_mask)`. The comment
at `_util.py:2328-2329` states the invariant ("must equal") but the code does not check
it; only `num_layers`' branch is taken (`elif layer_mask is not None` is unreachable once
`num_layers is not None`). If `_get_one_model_draft_layer_mask()` and `_get_num_draft_layers()`
were ever independently modified to drift apart, `layer_mask` and the constructed
manager's declared layer count (`num_hidden_layers = num_layers`) would silently disagree
— the mask would still be applied elsewhere (e.g. wherever `layer_mask` gates which of the
*target* model's runtime layers feed which cache), but `num_hidden_layers` for the draft
cache's own internal sizing would come from a different source than `sum(layer_mask)`,
with no error surfaced at construction time.

**Currently not reachable as an actual bug**: both values are produced by the *same*
underlying call, `self._get_num_draft_layers()` (called once inside
`_get_one_model_draft_layer_mask` at `_util.py:717`, and once directly at
`_util.py:1526`), which is a pure function of `self._speculative_config`/`self._draft_config`
state that does not change between the two calls within a single `build_managers()`
invocation. So today, `num_layers == sum(layer_mask[-num_draft_layers:])` always holds by
construction — but this is a call-site coincidence, not an enforced contract.

**Classification**: **Verified fact, not a proven bug** — the invariant is comment-only,
not asserted; a future edit that changes either `_get_num_draft_layers()` call site (e.g.
inlining or caching one but not the other) or adds a third caller could silently violate it
with no runtime error. Recommend, but do not implement (read-only audit), adding an
`assert num_layers == sum(layer_mask)` at the one-model draft call site or inside
`_create_kv_cache_manager` when both are supplied.

## 6. Buffer/role/dtype configuration (`BufferConfig`)

`_build_base_config` (`kv_cache_manager_v2.py:2040-2107`, full body read above) is a method
on the manager instance itself (`self`), driven entirely by that instance's own
`kv_cache_config.dtype`, `self.head_dim_per_layer`, `self.kv_cache_type`,
`self.max_attention_window_vec`, `self.num_local_layers`/`self.pp_layers`:

```python
buffer_type = [Role.KEY]
if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
    buffer_type.append(Role.VALUE)
if kv_cache_config.dtype == "nvfp4":
    for layer_idx, hd in enumerate(self.head_dim_per_layer):
        assert hd % 2 == 0, (...)
    buffer_type.append(Role.KEY_BLOCK_SCALE)
    if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
        buffer_type.append(Role.VALUE_BLOCK_SCALE)
...
for layer_id in typed_range(LayerId(self.num_local_layers)):
    buffers = [
        BufferConfig(
            role=role,
            size=self.get_layer_bytes_per_token(local_layer_idx=layer_id, data_role=role)
            * tokens_per_block,
        )
        for role in buffer_type
    ]
    for extra in extra_buffers_per_layer.get(int(layer_id), ()):
        ...
        buffers.append(extra)
```

Since `KVCacheManagerV2` for the draft is a **separate instance**, constructed from the
draft's own `model_config`/`dtype`/`quant_config` (`_util.py:1531,1557-1560,1586-1588` for
one-model; `_util.py:744-750` for two-model — `draft_model_config =
self._draft_model_engine.model.model_config`), its `_build_base_config` call independently
re-derives `buffer_type` and per-layer `BufferConfig` sizes from the draft's own quant
mode. **Verified fact**: NVFP4 extra scale buffers (`Role.KEY_BLOCK_SCALE`/
`VALUE_BLOCK_SCALE`) are added **per manager based on that manager's own
`kv_cache_config.dtype`** — so a target model quantized to NVFP4 paired with a draft model
in a different (e.g. bf16) precision would get scale buffers on the target manager only,
and vice versa. There is no shared/copied `BufferConfig` list between target and draft
managers anywhere in `_util.py` or `kv_cache_manager_v2.py`'s construction path.

`tokensPerBlockOverride`-equivalent: `tokens_per_block` (the physical page granularity,
dimension 7 below) is passed into `_build_base_config` as a parameter and used both for
buffer `size=... * tokens_per_block` and for `KVCacheManagerConfigPy(tokens_per_block=
self._ledger_tokens_per_block, ...)` (`kv_cache_manager_v2.py:2093`) — this is the single
shared scalar from dimension 7, so buffer sizing uses a consistent block granularity across
target/draft even though the byte-per-token rate differs per manager.

**Classification**: **Intentional policy** — buffer/role/dtype configuration is, by
design, computed independently per manager from that manager's own model config; this
matches the general architecture-independence goal of separate target/draft KV cache
managers (they can have entirely different quantization, head_dim, KV cache dtype).

## 7. `tokens_per_block`

`KvCacheCreator.__init__` stores a single scalar: `self._tokens_per_block = tokens_per_block`
(`_util.py:558,585`). Grep of all `tokens_per_block=` occurrences in `_util.py` confirms
every `_create_kv_cache_manager`/`_create_one_model_draft_kv_cache_manager`/
`_create_cross_kv_cache_manager` call site passes exactly `self._tokens_per_block`
(`_util.py:708, 1388, 1573, 1844, 1946`) — there is no branch anywhere in `_util.py` that
computes or overrides a different `tokens_per_block` for a draft manager, one-model or
two-model. This value originates from a single constructor parameter
(`_util.py:558: tokens_per_block: int`) supplied by the caller of `KvCacheCreator`, i.e.
one value for the whole executor build, not per-model.

**Downstream assumption**: no explicit cross-manager assert comparing `tokens_per_block`
between target and draft managers was found (grep of `kv_cache_manager_v2.py`,
`mamba_cache_manager.py`, `resource_manager.py`, and `scheduler/scheduler_v2.py` for
`tokens_per_block`/`copy_batch_block_offsets`). `KVCacheV2Scheduler` reads
`self.tokens_per_block = kv_cache_manager.tokens_per_block`
(`scheduler/scheduler_v2.py:195`) from **one** manager (its own KV cache manager instance)
— it does not cross-check against a draft manager's `tokens_per_block`. Since divergence is
structurally impossible from the `_util.py` construction path (single shared scalar), the
absence of a cross-check assert is not evidence of a latent bug — there is simply nothing
to check.

**Classification**: **Intentional policy / structurally guaranteed identical** — a single
scalar is threaded through all manager constructions with no override path found for
draft-specific `tokens_per_block`.

## Open Questions

1. **Two-model `max_tokens`-driven quota divergence (dimension 1)**: the equality assert
   at `_util.py:2081-2085` only constrains the **config field** `max_gpu_total_bytes` to be
   equal between target and draft for two-model V2. But `KVCacheManagerV2.__init__`
   (`kv_cache_manager_v2.py:1064-1076`) additionally clamps `quota =
   min(quota, quota_from_max_tokens)` whenever `kv_cache_config.max_tokens` is set, and
   `quota_from_max_tokens` is derived from **that manager's own** per-layer byte costs
   (`_get_quota_from_max_tokens_impl`, `kv_cache_manager_v2.py:1656` ff., reading
   `self._get_runtime_cache_size_layer_components()`). `configure_kv_cache_capacity`
   restores the user's raw `max_tokens` value unchanged for V2 (`_util.py:1311-1313`:
   "KVCacheManagerV2 doesn't rely on max_tokens to control capacity, so restore user
   provided value") and this same (unsplit) value is present in both `self_kv_cache_config`
   and `draft_build_kv_cache_config` for two-model (since no GPU/`max_tokens` split exists
   for two-model). If a user explicitly sets `kv_cache_config.max_tokens` alongside
   two-model V2 speculative decoding, target and draft managers would apply the *same*
   `max_tokens` number to *different* per-token byte costs, potentially producing genuinely
   different final `quota` values on top of an equal `max_gpu_total_bytes` — a possible gap
   between the assert's guarantee ("config field equal") and the actual outcome ("derived
   device quota equal"). I could not confirm from source alone whether this combination
   (two-model V2 + explicit `max_tokens`) is blocked elsewhere (e.g. in `TorchLlmArgs`
   validators, not read in this pass) or is a live, reachable configuration. Needs
   validation-layer or runtime evidence.

2. **Host tier auto-sizing divergence (dimension 2)**: confirmed as a reachable code path
   (two sequential, independently-snapshotted `_compute_auto_host_tier_quota` calls), but
   whether resulting *unequal* host quotas between target and draft managers cause any
   observable correctness or scheduling problem (vs. simply being an intentional
   per-manager sizing choice, similar to how one-model GPU quotas are also intentionally
   unequal) was not determined from source alone. No comment in
   `kv_cache_manager_v2.py` or `_util.py` addresses target/draft host-tier parity the way
   `_sync_host_tier_quota`'s docstring addresses cross-rank parity. Needs runtime/native
   evidence or an explicit design statement.

3. **Two-model `pool_ratio` arity (dimension 4)**: the native `storageManager.cpp:298-312`
   arity check would reject a mismatched `pool_ratio` for a two-model draft manager whose
   layer-group count differs from the target's `pool_ratio` list length, with no
   Python-side normalization analogous to the one-model `[1.0]` reset. Whether any current
   supported two-model configuration (VSWA/hybrid target + differently-grouped draft
   architecture) actually reaches this path was not confirmed — would need either a
   concrete config combination traced through `TorchLlmArgs` validation, or a runtime
   reproduction.

4. **Two-model draft attention-window derivation (dimension 3)**: no dedicated
   `_derive_draft_max_attention_window`-equivalent call site was found for the two-model
   draft build in `_util.py`. Whether the generic modulo-indexed window-pattern reuse in
   `kv_cache_manager_v2.py:399-408` produces correct per-layer windows for a two-model
   draft with a VSWA pattern that differs in shape/period from the target's is not
   confirmed from source alone in this pass — would benefit from a dedicated follow-up
   trace through `get_kv_cache_manager_cls`/`_create_kv_cache_manager`'s window-building
   path specifically for the `self._draft_model_engine is not None` branch.
