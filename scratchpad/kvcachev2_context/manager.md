# KVCacheManagerV2 Context Audit — Manager / Native Bridge / Resource-Management Layer

**Repo**: `/Users/allim/TensorRT-LLM`
**Commit**: `4716843cee6e7a6c08bf4d8be29fae25321a9344`
**Branch**: `feat/native-kv-events-clean`
**HEAD date**: 2026-08-31

**Primary files covered**:
- `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` (4237 lines; class `KVCacheManagerV2(BaseResourceManager)` at line 792)
- `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py` (dual cpp/python backend selector)
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h`
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h`
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h`
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h` (native per-sequence `KvCache`, partial)
- `tensorrt_llm/_torch/pyexecutor/_util.py` (construction / target-draft linkage, `KvCacheCreator` class)
- Tests: `test_kvv2_host_tier_sizing.py`, `test_kv_cache_manager_v2_helix_superblock.py`, `test_kv_cache_v2_capacity_only.py`, `test_kv_cache_v2_extra_buffers.py`, `test_dual_pool_kv_cache.py` (used as behavioral corroboration only, not analyzed line-by-line)
- Scheduler call sites identified via grep in `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py` and `tensorrt_llm/_torch/pyexecutor/py_executor.py` (call-site inventory only; scheduler control flow itself is **out of scope**, per instructions — a separate audit covers it).

Out of scope: deep control-flow analysis of `KVCacheV2Scheduler` itself (only call-site enumeration is provided here).

---

## 1. Dual-backend architecture (cpp default, python alternative)

**Verified fact.** `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py:23` selects the backend via `TLLM_KV_CACHE_MANAGER_V2_BACKEND` env var, defaulting to `"cpp"` (`__init__.py:23`). `BACKEND` is exported (`__init__.py:27`) and consumed by the pyexecutor manager as `KV_CACHE_MANAGER_V2_BACKEND` (`kv_cache_manager_v2.py:78`).

**Verified fact.** When `_BACKEND == "python"`, all core types (`KVCacheManager`, `_KVCache`, `KVCacheManagerConfig`, tier configs, etc.) come from a **pure-Python, mypyc-compilable** implementation under `tensorrt_llm/runtime/kv_cache_manager_v2/_core/`, `_storage/`, `_page.py`, `_block_radix_tree.py`, etc. (`__init__.py:29-104`; corroborated by `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`, "Architecture" section).

**Verified fact.** When `_BACKEND != "python"` (the default), all symbols are pulled from a compiled C++ extension module `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2` (`__init__.py:122-136,139-236`). The C++ source for that extension lives under `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`.

**Inference.** Because both backends expose the identical Python-level symbol names (`KVCacheManager`, `KVCacheManagerConfig`, `_KVCache`, etc.), `kv_cache_manager_v2.py` (the pyexecutor-level manager, hereafter "the Python manager wrapper" or "KVCacheManagerV2") is backend-agnostic at the source level; which backend actually executes is decided once at process startup by the env var.

**Open question.** Some symbols exist only on the C++ side and are given Python-only fallbacks with a `TODO(kvCacheManagerV2-cpp)` comment (`__init__.py:227-236`): `AttnLifeCycle`, `OutOfMemoryError` (falls back to builtin `MemoryError`), `PageIndexConverter`, `ReuseScope`, `ScratchDesc`, `SwaScratchReuseConfig`. Whether these gaps affect behavior when running the cpp backend (the default) is unclear without checking whether `getattr(_cpp, ...)` succeeds in the currently-built extension — not determinable from source alone.

---

## 2. Native KvCacheManager class API surface (C++, default backend)

Source: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h`.

**Verified fact.** Constructor: `KvCacheManager(KVCacheManagerConfig const& config, std::shared_ptr<EventSink> eventSink = nullptr, std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec = nullptr)` (`kvCacheManager.h:107-108`). Comment states the cold-page codec "is consumed when construction is invoked, including when construction throws" (`kvCacheManager.h:105-106`).

**Verified fact.** Lifecycle: `shutdown()` (`:116`), `clearReusableBlocks()` — clears all reusable (committed) blocks from the radix tree (`:118-119`).

**Verified fact.** KvCache creation: `createKvCache(ReuseScope, TokenSpan inputTokens, optional<RequestIdType> id, PriorityCb, optional<int> expectedPromptLength, optional<bool> textOnly, bool enableRequestStats)` (`:136-139`). Doc comment: "Returned cache is SUSPENDED; call activate() with a stream" (`:123`, though the actual activation method is `resume()`, see §4 below — the comment appears stale relative to the actual method name). `inputTokens` is documented as "a non-owning view; the caller must keep the underlying buffer alive for the duration of the call" (`:134-135`).

**Verified fact.** Reuse probing: `matchReuse(ReuseScope, TokenSpan, bool knownNoDigest=false)` and `probeReuse(...)` (`:143-145`), with the note that `knownNoDigest` "defaults false (safe: the scanning path is taken)" (`:141-142`).

**Verified fact.** Memory-pool queries: `getMemPoolBaseAddress`, `getPageStride`, `getPageIndexUpperBound`, `getPageIndexScale`, `getPageIndexConverter`, `getAggregatedPages`, `poolGroupDescs` (`:151-167`).

**Verified fact.** Query/info: `tokensPerBlock()`, `enablePartialMatch()`, `commitMinSnapshot()`, `textOnly()`, `isSwaScratchReuseEnabled()`, `supportsIndexMode(PageIndexMode)` (returns `optional<bool>`, "true/false for a definitive answer, nullopt for per-instance check", `:191-192`), `allowSeqRebasing()` (always returns `true`, `:194-197`), `numLayers()`, `layerIds()`, `getLayerGroupId(LayerId)`, `layerGrouping()`, `allBufferIds()`, `cacheTierList()`, `clampMaxSeqLenForMem(batchSize, tokenNumUpperBound)` (`:171-219`).

**Verified fact / explicit non-guarantee.** `layerGrouping()` doc: "the iteration order of the layer lists (and of the groups) is NOT part of the API contract and may differ across backends/runs. Do not rely on it... query `poolGroupDescs()`... for that" (`kvCacheManager.h:205-208`). This is directly relevant to the Python wrapper's caching of `_pool_layer_ids_by_role` from `pool_group_descs` rather than from `layer_grouping()` iteration order (`kv_cache_manager_v2.py:1253-1265`).

**Verified fact.** Resize: `bool resize(CacheLevel level, size_t quota, bool bestEfforts = false)`, `size_t getQuota(CacheLevel level)` (`:223-224`). This is the tier-level resize (not per-sequence — per-sequence resize is on `KvCache`, see §4).

**Verified fact.** Statistics: `commitStats`, `getCommittedStats`, `getAndResetIterationStats`, `getAndResetIterationPeakBlockStats`, `commitSsmSnapshotIterationStats`, `getAndResetSsmSnapshotIterationStats`, `recordRequestSuspended()`/`recordRequestResumed()` with the comment: "Both counters track the same population, so the running (suspended - resumed) total is the number of requests still parked in the SUSPENDED state" (`:243-247`), `markStatsDirty`/`clearStatsDirty`/`getDirtyStatsKvCacheIds`, `markStatsExcluded`/`clearStatsExcluded`/`isStatsExcluded` (`:228-254`).

**Verified fact.** `needAdjustment()` / `adjust()`: "Mirrors Python's need_adjustment property and adjust() method. All KvCaches must be suspended before calling adjust()" (`:256-259`) — an explicit precondition.

**Verified fact.** `registerKvCache(KvCache*)` / `unregisterKvCache(KvCache*)` are called by the `KvCache` constructor/destructor (`:288-290`) — the manager tracks a `std::set<KvCache*> mLivingKvCaches` (`:345`) of weak (non-owning) raw pointers.

**Open question.** The full implementation of `resize`, `adjust`, eviction selection, and stats aggregation is in `kvCacheManager.cpp`, which was not read as part of this audit (out of the primary file set given the time budget); behavioral details of *how* eviction victims are chosen, or exactly what `bestEfforts` changes, are not verified beyond the header comments.

---

## 3. Native KVCacheManagerConfig / tier config structs

Source: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h`.

**Verified fact.** `GpuCacheTierConfig { size_t quota = 0; }` with `assertValid()` throwing `std::invalid_argument("GpuCacheTierConfig: quota must be > 0")` if `quota == 0` (`config.h:37-51`). No `max_gpu_total_bytes`-named field exists at the native config level — the native tier config only ever takes a raw byte `quota`; the semantics of `max_gpu_total_bytes` (a `KvCacheConfig` field) are resolved entirely in the **Python wrapper**, not in the native config (see §5).

**Verified fact.** `HostCacheTierConfig { size_t quota = 0; }`, same `quota > 0` validation (`config.h:53-67`).

**Verified fact.** `DiskCacheTierConfig { size_t quota = 0; std::string path; }`; `assertValid()` is declared but its body is only in `config.cpp` (not read) (`config.h:69-80`).

**Verified fact.** `CacheTierConfig = std::variant<GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig>` (`:83`). Top-level `KVCacheManagerConfig::cacheTiers` comment: "Ordered from warm (GPU) to cold (disk). First must be GPU memory" (`:265`) — this is an **implicit ordering precondition**, not enforced by a visible assertion in this header (validation body is in `config.cpp`, not read).

**Verified fact.** `KVCacheManagerConfig` fields (`config.h:261-310`): `tokensPerBlock`, `cacheTiers`, `layers` (attention or SSM, "Layer IDs must be unique" comment, `:268`), `maxUtilForResume = 0.97f` ("Suspend/resume threshold: if utilization > this, resuming will fail", `:271-272`), `enablePartialReuse = true`, `constraints` (batches that must always be supportable), `typicalStep` (typical step for initial ratio computation), `initialPoolRatio` (per-layer-group normalized hot-tier byte-quota weight override, `:280-282`), `swaScratchReuse` (optional `SwaScratchReuseConfig{int maxRewindLen}`), `commitMinSnapshot` (bool, "Required when SSM layers are present", `:292`), `enableStats = true`, `textOnly = false` ("Deployment-level guarantee that no request carries multi-modal content... A per-KvCache text_only override may only tighten this", `:298-302`).

**Verified fact.** `BufferConfig { DataRole role; size_t size; optional<int> tokensPerBlockOverride; }` — `tokensPerBlockOverride` "Must be a divisor of KVCacheManagerConfig::tokensPerBlock" (`config.h:100-108`, comment-only, not shown enforced in this header).

**Verified fact.** `SsmLayerConfig::validate()` throws `std::invalid_argument("tokensPerBlockOverride not supported for SSM layers")` if any buffer has that override set (`config.h:174-182`).

**Verified fact.** `KVCacheDesc { int capacity; int historyLength; }` with `validate()` using `TLLM_CHECK_DEBUG(0 <= historyLength && historyLength <= capacity)` (`config.h:191-199`) — a **debug-only** check (not enforced in release builds, based on the `_DEBUG` naming convention typical in this codebase; not independently confirmed by reading `TLLM_CHECK_DEBUG`'s macro definition).

**Open question.** Whether `max_gpu_total_bytes`/host/disk sizing validation (beyond the trivial `quota > 0` checks shown here) happens anywhere in `config.cpp`'s `KVCacheManagerConfig::validate()` was not verified — `config.cpp` was not read.

---

## 4. Native per-sequence `KvCache` state machine (partial)

Source: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h:154-330` (read only; full file not exhaustively covered).

**Verified fact.** `Status` enum: `ACTIVE`, `SUSPENDED`, `CLOSED` (`kvCache.h:158-162`). `CommitState` enum: `ALLOWED`, `VIRTUAL_STOP`, `USER_STOP` (`:163-167`).

**Verified fact.** `bool resume(optional<CUstream> stream = nullopt)` — "check utilization and lock all pages to GPU... Returns false if utilization too high or out of memory" (`:183-185`). This is the actual state-machine transition invoked by the Python wrapper's `kv_cache.resume(...)` calls (e.g. `kv_cache_manager_v2.py:2476`, `:2574`).

**Verified fact.** `void suspend()` — "detach from CUDA stream, unlock pages → PageHolder" (`:187-188`).

**Verified fact.** `void close()` — "release all blocks back to KvCacheManager" (`:189-190`).

**Verified fact.** `bool resize(optional<int> capacity, optional<int> historyLength = nullopt)` — "Resize capacity and/or history_length. Returns true if the resize was a no-op shortcut" (`:200-202`). **Note**: the doc phrasing is ambiguous — it says "true if... a no-op shortcut", not explicitly "true on success generally"; the Python wrapper (§6 below) uniformly treats a `False`/falsy return as failure and a truthy return as success, consistent with typical usage, but the exact semantics of the return value in the non-no-op success case are not spelled out in this comment.

**Verified fact.** `commit(TokenSpan tokens, bool isEnd = false)` — "Commit tokens: finalises the oldest uncommitted block and makes it available for reuse... tokens must contain exactly tokensPerBlock tokens per call (until the last)... This is a terminal-memory contract: callers must not perform later writes to this KvCache's memory [after isEnd=true]. The final live pages may be moved into the radix tree instead of copied" (`:216-224`).

**Verified fact.** `stopCommitting()` — "Stop committing (called by close() automatically)" (`:226`). Cross-referenced with the Python-implementation `AGENTS.md` gotcha: "`stopCommitting()` must NOT call `commit()` — it would double-append tokens to the block" (`tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`, Gotchas section) — this is documented for the pure-Python backend specifically, but is presumably an equivalent invariant on the C++ side (**Inference**, not directly verified in `kvCache.h`).

**Verified fact.** `planCommittedBlockDrop()` — "Must be called after stopCommitting(). Returns nullptr without creating a plan if any required SWA page is unavailable" (`:290-296`) — matches the Python wrapper's `ConversationManager.save_drop_plan` which calls `kv_cache.plan_committed_block_drop()` and treats a `None` return as "blocks have been dropped" (`kv_cache_manager_v2.py:183-192`).

**Open question.** The rest of `kvCache.h` (page-locking internals, beam-fork behavior, scratch-desc mechanics beyond the signature) was not read in full; deep semantics of `getBasePageIndices`/`setBasePageIndexBuf` zero-copy wiring are only inferred from the Python wrapper's usage (§6), not independently verified against the C++ implementation body.

---

## 5. Exception hierarchy

Source: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h`.

**Verified fact.** Hierarchy: `OutOfMemoryError : std::runtime_error` (`:56-63`) → `HostOOMError`, `DiskOOMError`, `CuOOMError` (`:65-90`, all `: OutOfMemoryError`). `LogicError : std::logic_error` — "Indicates a bug in the KV cache manager code" (`:93-100`). `AssertionError : std::logic_error` — "the binding layer translates this to a Python AssertionError so shared tests observe the same exception type as the pure-Python backend" (`:105-112`). `CuError : std::runtime_error` wraps a `CUresult` (`:115-135`). `ResourceBusyError` — "A resource (e.g., a page lock) is still in use" (`:138-145`). `OutOfPagesError : std::runtime_error` — "Not enough free pages to satisfy an allocation request" (`:148-155`).

**Verified fact.** `cuCheck(CUresult)` helper: throws `CuOOMError` specifically for `CUDA_ERROR_OUT_OF_MEMORY`, else generic `CuError` (`:173-184`).

**Verified fact.** `unwrap<T>(WeakPtr<T> const&)` throws `LogicError("Dereferencing a dangling weak_ptr")` if the weak pointer is expired (`:161-168`) — "Mirrors Python's unwrap_rawref" comment ties this to the pure-Python backend's `rawref` mechanism.

**Verified fact — Python-side exception surface.** In the cpp-backend branch of `runtime/kv_cache_manager_v2/__init__.py`: `OutOfPagesError = _cpp.OutOfPagesError` (`:213`), `CuError = _cpp.CuError` (`:225`), `OutOfMemoryError = getattr(_cpp, "OutOfMemoryError", MemoryError)` (`:230`, fallback only if the binding doesn't expose it). The pyexecutor manager imports `OutOfMemoryError` as `KVCacheOutOfMemoryError` (`kv_cache_manager_v2.py:81`) and `CuError` directly (`:55`), and catches both specifically during host-tier fallback construction (`kv_cache_manager_v2.py:1183`, see §7).

**Open question.** `terminateOnException` (`exceptions.h:33-50`) calls `std::terminate()` on any exception — it's unclear from this header alone which call sites (if any in the manager/KvCache methods) are wrapped with it vs. which propagate exceptions normally to Python. This determines whether certain failure paths are recoverable in Python or crash the process; not resolvable without reading the `.cpp` files and/or the nanobind binding glue.

---

## 6. Manager construction (`KVCacheManagerV2.__init__`, `kv_cache_manager_v2.py:797-1373`)

### 6.1 GPU quota resolution

**Verified fact.** `max_gpu_total_bytes` and `max_tokens` are mutually combinable, taking the **minimum** implied quota:
- If `kv_cache_config.max_gpu_total_bytes` is set and `> 0`: `quota = int(max_gpu_total_bytes)` (`kv_cache_manager_v2.py:1058-1063`).
- If `kv_cache_config.max_tokens` is set: converts tokens→bytes via `_get_quota_from_max_tokens`, divides by `max_util_for_resume`, and takes `quota = min(quota, quota_from_max_tokens)` (`:1064-1076`).
- **Verified fact / hard precondition.** `assert quota < sys.maxsize, "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"` (`:1078-1080`) — **at least one of `max_gpu_total_bytes` or `max_tokens` must be set**, or construction raises `AssertionError`.

**Verified fact.** With `mapping.world_size > 1`, the quota is synchronized across ranks by converting to a token count, `allreduce(MIN)`, and converting back, with an explicit clamp comment: "allreduce(MIN) must never increase the local quota... clamp to guard against a bogus inflation (nvbugs/6418103)" (`kv_cache_manager_v2.py:1085-1100`).

**Verified fact.** `cache_tiers = [GpuCacheTierConfig(quota=int(quota))]` is always first (`:1104`) — matches the native config's documented tier ordering (§3).

### 6.2 Host tier sizing (explicit vs. auto)

**Verified fact.** If `kv_cache_config.host_cache_size is not None and >= 0`: `host_quota = kv_cache_config.host_cache_size` (explicit) (`:1105-1106`).

**Verified fact.** Otherwise, **auto-provisioning** kicks in, with the rationale documented in-line: "The V2 MAX_UTILIZATION scheduler relies on suspend/resume to evict and later restore KV cache pages. Without a secondary tier, suspended held pages cannot migrate out of GPU, so suspension cannot free capacity and scheduling can deadlock" (`:1108-1112`). The auto host tier defaults to matching the GPU quota, "Cap at available host memory and pinnable memory limit to avoid allocation failures" (`:1113-1115`).

**Verified fact.** Rank-aware: `local_ranks = max(1, Distributed.get(mapping).local_world_size)` divides the per-node memory budget among co-located ranks, "without this, N co-located ranks each reserving a device-quota-sized block can OOM the host (observed on GB300 NVL72 with 4 ranks/node and ~170GiB device quota each on a 975GiB node)" (`:1118-1124`).

**Verified fact.** `_compute_auto_host_tier_quota(quota, local_ranks, mem_available, memlock_limit)` (`:242-279`) computes `candidates = [quota]`, appends `int(mem_available / local_ranks * 0.5)` if `mem_available != inf`, appends `int(memlock_limit * 0.8)` if `memlock_limit != inf`, and returns `min(candidates)`. If the result is `<= 0`, it logs a warning and **falls back to the raw device quota** (`:266-279`) — i.e. auto-sizing never produces a non-positive host quota; worst case it equals the GPU quota unconstrained.

**Verified fact.** `mem_available` is read via `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")`, falling back to `float("inf")` on `(ValueError, OSError)` (`:1126-1129`). `memlock_limit` is read via `resource.getrlimit(resource.RLIMIT_MEMLOCK)`'s soft limit (`RLIM_INFINITY` → `inf`), same exception fallback (`:1130-1134`).

**Verified fact.** The rank-local auto quota is then reduced to the fleet minimum via `_sync_host_tier_quota` (`:1145`, function body `:282-304`): "co-scheduled ranks can arrive at divergent host quotas (observed up to 10x). Divergent host-tier retention makes per-rank MAX_UTILIZATION schedulers disagree about which suspended requests can resume, which wedges collectives on non-attention-DP TP" (`:285-291`). Uses `Distributed.get(mapping).allreduce(host_quota, op=ReduceOp.MIN)` only if `mapping.world_size > 1` (`:302-304`); single-rank jobs skip the collective.

**Verified fact.** `if host_quota > 0: cache_tiers.append(HostCacheTierConfig(quota=int(host_quota)))` (`:1146-1147`) — i.e. host tier is **only added if positive**; given §6.2's fallback logic, explicit `host_cache_size=0` is the only way to end up with **no** host tier via this path (auto-sizing always yields `> 0`).

**Test corroboration.** `tests/unittest/_torch/executor/test_kvv2_host_tier_sizing.py` directly unit-tests `_compute_auto_host_tier_quota` (class `TestComputeAutoHostTierQuota`, line 23) and `_sync_host_tier_quota` (class `TestSyncHostTierQuota`, line 136), confirming these are treated as independently-testable, load-bearing policy functions.

### 6.3 Disk tier sizing

**Verified fact.** `disk_cache_size = kv_cache_config.disk_cache_size`; if set and `> 0`, `disk_cache_path` is asserted non-`None` and `DiskCacheTierConfig(quota=int(disk_cache_size), path=disk_cache_path)` is appended (`:1151-1160`). **No auto-sizing exists for disk** — it is purely explicit opt-in, consistent with the `_drop_explicit_offload_tier_budgets` docstring in `_util.py` ("The disk tier is V2 only and has no auto policy, so dropping its budget leaves the tier out", `_util.py` around the `_drop_explicit_offload_tier_budgets` classmethod).

### 6.4 Construction retry / fallback on host-tier failure

**Verified fact.** If a host tier is configured, construction is wrapped in a try/except that specifically catches `(CuError, KVCacheOutOfMemoryError)` and sets a local status `USE_NO_HOST`; any other exception sets status `ABORT` and the original exception is re-raised (stripped of traceback via `.with_traceback(None)`) (`kv_cache_manager_v2.py:1178-1187`).

**Verified fact.** The **worst status across all ranks** is computed via `_sync_kv_cache_manager_init_status` (an `allreduce(MAX)` over the `_KVCacheManagerInitStatus` IntEnum, whose numeric ordering is explicitly part of the protocol: "more severe outcomes must have larger values" — `KEEP_HOST=0 < USE_NO_HOST=1 < ABORT=2`, `:307-323`). This means **one rank's host-tier allocation failure aborts host tier construction on all ranks**, keeping ranks consistent.

**Verified fact.** On `ABORT`: any partially-constructed candidate is shut down and the original error (or a generic `RuntimeError("KV cache manager initialization failed on another rank")` if this rank had no local error) is raised (`:1191-1196`).

**Verified fact.** On `USE_NO_HOST`: logs a warning ("At least one rank could not use the KV cache manager host tier (cuMemHostRegister may have failed). Rebuilding without the host cache tier on all ranks", `:1199-1203`), shuts down the failed candidate, strips `HostCacheTierConfig` from `config.cache_tiers`, and retries construction **without any host tier at all** — even on ranks whose host tier construction would have succeeded (`:1204-1236`). This retry is itself synchronized (another `allreduce(MAX)`) and can itself `ABORT` (`:1218-1236`).

**Verified fact.** `self.can_evict = len(config.cache_tiers) > 1` (`:1241`) — **eviction to a secondary tier is only possible if at least one non-GPU tier survived construction.** If the host-tier fallback triggers, `can_evict` becomes `False`, and (per the in-line rationale in §6.2) the MAX_UTILIZATION scheduler's suspend/resume-based eviction path is effectively disabled for that run.

### 6.6 Constructor signature and misc construction-time asserts

**Verified fact.** `KVCacheManagerV2.__init__` signature (`kv_cache_manager_v2.py:797-827`) key kwargs: `kv_cache_config: KvCacheConfig`, `kv_cache_type`, `num_layers`, `num_kv_heads`, `head_dim`, `tokens_per_block`, `max_seq_len`, `max_batch_size`, `mapping`, `dtype`, `spec_config=None`, `layer_mask`, `vocab_size`, `max_num_tokens=8192`, `model_config`, `max_beam_width=1`, `is_draft: bool = False`, `kv_connector_manager`, `execution_stream`, `is_disagg: bool = False`, `enable_stats: bool = False`, `num_reserved_index_slots: int = 1`, `kv_events_config`, `is_estimating_kv_cache: bool = False`.

**Verified fact / explicit non-support.** `kv_connector_manager` must be `None` — `assert kv_connector_manager is None` (`:832-834`), i.e. **V2 does not support the KV-connector integration** that V1 `KVCacheManager` supports. `max_beam_width` must be `1` (`:835`) — beam search width `>1` is unsupported at construction time (in addition to the separate `assert beam_width == 1` inside `copy_batch_block_offsets`, §9).

**Verified fact.** `IndexMapper` capacity: `index_mapper_capacity = max_num_sequences * (2 if is_disagg else 1) + num_reserved_index_slots` (`:1339-1353`) — the disagg 2x factor is explained in-line: active-generation slots and in-flight-NIXL/UCX-transfer slots must coexist for the same logical request during a hand-off window.

**Verified fact.** `max_seq_len` can be **silently clamped down**: `max_num_tokens = self.get_num_available_tokens(token_num_upper_bound=max_seq_len)`; if `max_seq_len > max_num_tokens`, `self.max_seq_len` is reduced to `int(max_num_tokens)` with only a warning log, not an exception (`:1300-1311`) — a caller requesting more sequence length than the configured quota can support does not get an error, it silently gets less.

**Verified fact.** `max_blocks_per_seq` is derived from `max_seq_len + num_extra_kv_tokens + _kv_reserve_draft_tokens + 1`, then padded up to a multiple of 4 to satisfy a `copy_block_offsets` kernel alignment requirement (`:1313-1323`).

**Verified fact — streaming KV events require the Python backend.** `validate_streaming_support(..., backend=KV_CACHE_MANAGER_V2_BACKEND)` is called at construction (`:980`) and (per the surrounding comment, `:971-972`) rejects streaming KV-cache-event configuration when the backend is not `"python"` — i.e. **streaming KV events are not supported on the default `"cpp"` backend** as of this commit.

**Verified fact.** The streaming event manager is started **last** in `__init__`, after every other construction step that can fail, specifically so a failure earlier (including a rank-coordinated abort where this rank raises because a peer failed) leaves no bound socket needing cleanup (`:1366-1372`).

### 6.5 Estimation-phase managers

**Verified fact.** `is_estimating_kv_cache: bool = False` constructor kwarg (`:825`) is stored verbatim so "consumers (e.g. CUDAGraphRunner.preallocate_padding_dummies) can distinguish the throwaway estimation-phase managers from the final ones: the estimation cache is sized with no headroom for retained dummy requests" (`:846-850`).

**Verified fact (from `_util.py`).** During estimation, explicit offload-tier budgets (host/disk) are dropped before manager construction: "Estimation managers are throwaway probes whose pools only hold dummy requests, so an explicit offload tier would reserve capacity the probe cannot fill" (`_util.py`, `build_managers`, comment above `estimating_kv_cache` branch, ~line 2013-2018 per the earlier read at `_util.py:1900-2100` region). This means **estimation-phase managers rely on auto host-tier sizing (or no host tier) even when the final run has explicit host/disk sizing.**

---

## 7. Target/draft manager creation and linkage (speculative decoding)

Source: `tensorrt_llm/_torch/pyexecutor/_util.py`, class with method `build_managers` (~line 1995 onward) and helpers `_should_create_separate_draft_kv_cache`, `_create_one_model_draft_kv_cache_manager`, `_create_kv_cache_manager`.

**Verified fact.** `KvCacheCreator.build_managers` (`_util.py`, `build_managers` def around line 1995) determines `has_draft = (self._draft_model_engine is not None) or self._should_create_separate_draft_kv_cache()` — the **first disjunct is two-model spec decoding** (a distinct draft model engine exists), the **second is one-model spec decoding with a separate-layout draft cache** (e.g. Eagle3/draft-target one-model mode) (`_util.py`, comment "Two-model speculative decoding: draft model has separate engine" and "One-model speculative decoding with different KV layouts" bracketing the two branches, read at `_util.py:2077-2093`).

**Verified fact — two-model path.** `_util.py:2077-2088`: if `self._draft_model_engine is not None`:
```
if (self._is_kv_cache_manager_v2 and draft_kv_cache_config is not None):
    assert (draft_kv_cache_config.max_gpu_total_bytes ==
            self_kv_cache_config.max_gpu_total_bytes), (
        "KVCacheManagerV2 does not support two-model "
        "speculative decoding with separate draft GPU "
        "budgets.")
draft_kv_cache_manager = self._create_kv_cache_manager(
    self._draft_model_engine, estimating_kv_cache,
    kv_cache_config_override=draft_build_kv_cache_config)
```
**This is an explicit, hard constraint**: for `KVCacheManagerV2`, the target and draft managers in two-model spec decoding **must share the same `max_gpu_total_bytes`** — a per-manager GPU budget split is not supported (only offload-tier budgets, i.e. host/disk, are split per-manager; see `_util.py:2038-2044`, "The GPU split applies when each manager sizes its pools from max_gpu_total_bytes (V2 and V1 VSWA)... For budget_attr in self._OFFLOAD_TIER_BUDGET_ATTRS: split per draft").

**Verified fact — one-model path.** `elif self._should_create_separate_draft_kv_cache(): draft_kv_cache_manager = self._create_one_model_draft_kv_cache_manager(...)` (`_util.py:2091-2093`). `_should_create_separate_draft_kv_cache` (`_util.py:1432-1455`) returns `False` unconditionally if `mapping.enable_attention_dp` ("Attention DP is enabled, separate draft KV cache is not supported", logged) or if using DeepSeek-V4 sparse attention with `pp_size > 1` ("folding draft layers into the unified manager"); otherwise delegates to `should_use_separate_draft_kv_cache(self._speculative_config)` (external helper, not read).

**Verified fact.** `_create_one_model_draft_kv_cache_manager` (`_util.py:1516-1583`) builds the draft manager with: `num_layers=num_draft_layers` (from `_get_num_draft_layers`, `_util.py:1473-1479`, "must stay in sync with the num_layers passed to the draft KV cache manager constructor" — a documented but not statically-enforced invariant), `is_draft=True`, `layer_mask=spec_dec_layer_mask`, and a derived `KvCacheConfig` clone whose `max_attention_window` is recomputed via `_get_draft_max_attention_window` (VSWA-aware) unless estimating (`_util.py:1509-1515`).

**Verified fact — pool_ratio normalization for one-model draft.** If the draft manager's window layout is non-VSWA and the target's `pool_ratio` has more than one entry, it's forcibly reset to `[1.0]`: "pool_ratio describes one manager's layer-group layout. The target hybrid manager may have separate recurrent-state and attention layer groups, while a non-VSWA draft manager has one attention layer group. Reusing the target's ratios fails its arity check" (`_util.py:1541-1551`).

**Verified fact — Python-manager side draft flag.** The manager's own `is_draft` constructor kwarg (`kv_cache_manager_v2.py:818`) changes several code paths: `enable_swa_scratch_reuse` is forced off for draft managers (`kv_cache_manager_v2.py:854-856`, `... and not self.is_draft`), `enable_conversation_manager` requires `not self.is_draft` (`:1328-1331`), `commit_scheduled_kv_cache_stats` is a no-op for draft managers (`:2382-2383`), and `try_commit_blocks` skips block-reuse commit for draft managers (`:3652-3656`, `not self.is_draft`).

**Verified fact — draft manager mirrors scheduling, doesn't schedule itself.** `prepare_resources` (`:2771-2777`): `if self.is_draft: self._prepare_draft_resources(scheduled_batch); return` — for a non-draft (target) manager, `prepare_resources` is a no-op. `_prepare_draft_resources` docstring: "The main V2 scheduler only manages the primary KV cache manager. The draft manager must mirror context/generation allocations so that its IndexMapper contains the correct request IDs for copy_batch_block_offsets()" (`:2780-2785`). This confirms **`KVCacheV2Scheduler` (out of scope) drives only the target manager**; the draft manager's per-request lifecycle (create/resize/resume) is self-driven from within `prepare_resources`/`update_resources`, called once per iteration by the executor for *each* manager (target and draft) — see call sites in §9.

**Verified fact — draft capacity reserve.** In `_prepare_draft_resources`, generation-request resize pads to `self._kv_reserve_draft_tokens` (reserve draft slack) beyond `_required_gen_capacity` (`:2833-2843`); and in `update_resources` for a draft manager (`:4062-4069`), extra rewind is applied to reclaim unused dynamic-tree reserve slack: "Dynamic-tree draft managers reserve K * max_draft_len slots, which can exceed the tree's runtime draft width. Reclaim that reserve slack together with rejected draft tokens; otherwise it accumulates in the draft KV cache after every generation step. Target managers do not allocate this reserve slack."

**Verified fact — Helix (CP) rejects draft managers.** `if self._has_cp_helix: ... if is_draft: raise ValueError("KVCacheManagerV2 does not support a draft cache manager with helix context parallelism.")` (`kv_cache_manager_v2.py:885-889`) — an explicit unsupported combination.

**Open question.** Whether/how the target and draft `KVCacheManagerV2` instances share (or don't share) the native `KvCacheManager` C++ object, the event manager, or the `IndexMapper` was not directly verified — from the constructor code read, **each `KVCacheManagerV2` Python wrapper builds its own independent native `self.impl` (a separate `KVCacheManagerPy(config, ...)` call) and its own `self.index_mapper`** (`kv_cache_manager_v2.py:1174-1240`, `:1360`), so target and draft managers are fully independent native objects, linked only via the scheduler/executor calling both and via shared request IDs. This is an **Inference** from the construction code, not an explicit statement in a docstring.

---

## 8. Python-side vs. native-side state

**Verified fact.** The Python wrapper's per-request Python-side state (`kv_cache_manager_v2.py:1281-1298,1360-1362`):
- `self.kv_cache_map: dict[int, _KVCache]` — request_id → native `_KVCache` handle (Python wrapper's only strong reference to each per-sequence native object).
- `self._request_stats_enabled_ids: set[int]`.
- `self._allocated_draft_lens: dict[int, int]` — "Tracks the draft length allocated by try_allocate_generation per request. Used by extend_capacity_for_tokens to compute the exact padding delta instead of blindly extending" (`:1284-1287`).
- `self._gpu_max_tokens` — GPU-only cap for `get_num_available_tokens` when `max_tokens` is explicit (`:1290-1298`).
- `self.index_mapper: IndexMapper` (native `IndexMapper` binding from `kv_cache_manager_v2_utils`, `kv_cache_manager_v2.py:38,1360`) — separate from the KvCacheManager's own internal bookkeeping; maps request IDs to fixed-capacity zero-copy buffer slots.
- `self._early_freed_index_requests: set[int]` — tracks requests whose IndexMapper slot was released early via `release_index_slot` so `free_resources` does not double-release it (`:1361`, `:3675-3706`).

**Verified fact — potential desync risk (explicit comment).** `_restore_page_index_bufs` docstring: "`suspend()` clears the `base_page_index_buf` pointers (sets them to None) so the KV cache stops writing page indices to the host buffer. After `resume()`, the KV cache has re-locked pages but `copy_batch_block_offsets` still reads from the host buffer, so we must re-connect the buffers to avoid stale/zero page indices that would cause illegal memory accesses" (`:2549-2558`). This shows the native `KvCache` object's page-index buffer wiring and the Python-owned `self.host_kv_cache_block_offsets` tensor are **two separate pieces of state that must be manually re-synchronized on every resume** — a caller who resumes a cache without going through `_resume_and_restore`/`try_allocate_generation`/`resume_request` (all of which call `_restore_page_index_bufs`) would produce stale page indices.

**Verified fact.** Native-side state is authoritative for: block/page allocation, radix-tree reuse matches, per-tier quotas, committed/dirty/excluded stats sets (`self.impl.get_dirty_stats_kv_cache_ids()`, `self.impl.clear_stats_excluded(...)`, `self.impl.mark_stats_excluded(...)` — `:2384`, `:3698`, `:3702`, `:4172`), and the `_KVCache` object's own `Status`/`CommitState`/capacity/history_length (all queried, never independently cached in Python beyond the `kv_cache_map` handle itself).

**Verified fact.** `get_num_free_blocks()` has an explicit precondition comment and assertion: `assert len(self.kv_cache_map) == 0, "get_num_free_blocks is only used when the kv cache manager is empty"` (`:2370-2372`) plus an explicit caveat: "This method is used to get the number of blocks in the primary pool not the FREE blocks. However, since we only use this function when the kv cache manager is empty, so it is safe to do so" (`:2368-2369`) — i.e. **this method's name is misleading**; it returns total pool capacity, and its correctness as "free blocks" depends entirely on the precondition that no requests are currently registered.

---

## 9. Public scheduler-facing / resource-manager-facing API — per-method contracts

Call sites confirmed via grep of `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py` and `tensorrt_llm/_torch/pyexecutor/py_executor.py`.

### `prepare_disagg_gen_init(req) -> bool` (`kv_cache_manager_v2.py:2678-2708`)
Call site: `scheduler_v2.py:540`.
- **Preconditions**: none enforced beyond `req` being in disagg-generation-init state (the caller decides when to call it; the sibling `prepare_context`/`resize_context` explicitly assert `not req.is_disagg_generation_init_state`, `:2587-2589,2652-2654`, implying this method is the disagg-init-specific counterpart).
- **Postconditions**: on success, allocates capacity for the full prompt (+ draft, + extra KV tokens) and sets `kv_cache.history_length = prompt_len` via `kv_cache.resize(capacity, prompt_len)` (`:2702`).
- **Python-side mutation**: `req.py_ctx_pre_resize_cap` set if capacity grew (`:2707`), used later by `revert_allocate_context`.
- **Native mutation**: calls `_prepare_context_impl` (creates `_KVCache` via `self.impl.create_kv_cache`, see `_create_kv_cache` below) then `kv_cache.resize(...)`.
- **Errors**: returns `False` (not an exception) on preparation or resize failure; on resize failure for the first context chunk, calls `kv_cache.suspend()` before returning `False` (`:2703-2706`) — i.e. **failure leaves the cache suspended, not destroyed**.
- **Not guaranteed**: does not guarantee the cache remains active after a failed resize (explicitly suspends it); does not free the cache on failure (caller must call `free_resources` if it wants full cleanup).

### `prepare_context(req) -> bool` (`:2579-2645`)
Call sites: `scheduler_v2.py:579,628`; also directly for the cross-attention pool at `scheduler_v2.py:918` (`cross_kv_cache_manager.prepare_context`).
- **Precondition**: `assert not req.is_disagg_generation_init_state` (`:2587-2589`).
- **Postcondition — explicit non-guarantee in docstring**: "Create `_KVCache`, handle block reuse, and resume. Does NOT resize" (`:2580`) — capacity growth is the caller's (scheduler's) responsibility via a separate `resize_context` call.
- For the first chunk: creates the cache (if absent) via `_create_kv_cache`, optionally augments tokens for block reuse, sets `req.context_current_position`/`req.set_prepopulated_prompt_len` when block reuse is enabled (`:2623-2629`), and resumes it. For non-first chunks: asserts the cache already exists (`:2641-2643`) and just resumes.
- **Native mutation**: `self.impl.create_kv_cache(...)` (only on first chunk, cache absent), `kv_cache.resume(...)`.
- **Errors**: returns `False` if `_create_kv_cache` returns `None` (IndexMapper saturated) or if resume fails; raises `AssertionError` if called on a non-first chunk with no existing cache (a genuine bug condition, not a runtime/scheduling failure).

### `resize_context(req, num_tokens) -> bool` (`:2646-2676`)
Call sites: `scheduler_v2.py:596,698`; also `cross_kv_cache_manager.resize_context` at `scheduler_v2.py:924`.
- **Precondition**: `assert not req.is_disagg_generation_init_state` (`:2652-2654`); **raises `ValueError`** if `self._has_cp_helix and not req.is_dummy_request` — "`resize_context` is not helix-aware... Helix requests are disagg-generation-only and must never take the context path" (`:2655-2661`).
- **Postcondition**: resizes to `max(current_capacity, context_current_position + num_tokens + num_extra_kv_tokens)` (`:2666-2670`).
- **Python-side mutation**: `req.py_ctx_pre_resize_cap` set (used by `revert_allocate_context`) (`:2675`).
- **Errors**: returns `False` on resize failure; if this is the first context chunk, **suspends** the cache before returning `False` (`:2672-2674`) — same partial-failure pattern as `prepare_disagg_gen_init`.

### `try_allocate_generation(req) -> bool` (`:2465-2491`)
Call sites: `scheduler_v2.py:994,1140,1217`.
- **Preconditions**: none explicit; returns `False` immediately if no cache exists for the request (`:2472-2473`).
- **Behavior**: resumes from suspended state if needed (restoring page-index buffers via `_restore_page_index_bufs`, `:2475-2478`); computes `draft_len` and records it in `self._allocated_draft_lens[req_id]` (Python-side state, `:2480-2481`) *before* attempting the resize — this means **the recorded draft length is written even if the subsequent resize fails** (only reset by a later `pop()` in `extend_capacity_for_tokens`/`revert_allocate_generation`/`free_resources`, not rolled back here on failure).
- **Native mutation**: `kv_cache.resize(current_capacity + 1 + draft_len)` (`:2485`).
- **Helix-specific side effect**: if this is a real (non-dummy) helix request, calls `_set_helix_rank_fields(req)` *before* resize, and only commits `req.py_helix_decode_group_index += 1` **after** a successful resize (`:2483-2490`) — "Commit only on success so a same-pass retry recomputes the same step instead of skipping one" (`:2488-2489`), an explicit at-most-once-on-success guarantee for the helix step counter.
- **Errors**: returns `False` on resume or resize failure; no exception raised.
- **Not guaranteed**: does not roll back the just-recorded `_allocated_draft_lens` entry on resize failure (caller relies on `revert_allocate_generation`/`free_resources` to clean it up later).

### `revert_allocate_generation(req) -> None` (`:2493-2523`)
Call sites: `py_executor.py:3438,4217,5048`.
- **Purpose** (docstring): "When attention DP causes can_queue=False after scheduling, the forward pass is skipped but the scheduler already grew each generation request's KV cache capacity by 1 (+draft tokens). This method shrinks capacity back to undo that spurious growth" (`:2495-2499`).
- **Preconditions**: no-op if cache absent or not active (`:2506-2508`).
- **Postcondition**: shrinks capacity by `1 + draft_len` (draft_len read from `_allocated_draft_lens.pop(...)`, falling back to `_effective_draft_len(req)` if not recorded, `:2512-2514`); no-op if the reverted capacity would be negative (`:2515-2517`, silently returns without resizing — an **explicit non-guarantee**: capacity is not always actually reverted).
- **Errors**: **raises `RuntimeError`** if the resize itself fails (`:2518-2523`) — this is the one lifecycle method that raises on native resize failure rather than returning a bool, presumably because a revert failing indicates a genuine invariant violation rather than an expected capacity-pressure outcome.

### `revert_allocate_context(req) -> None` (`:2525-2547`)
Call site: `py_executor.py:3477`.
- **Precondition/no-op guard**: relies on `req.py_ctx_pre_resize_cap` being set by a prior `resize_context`/`prepare_disagg_gen_init` call; no-op if unset, cache absent, not active, or `pre_cap >= current capacity` (`:2527-2535`).
- **Special case**: if `kv_cache.history_length > pre_cap` (i.e. committed history has already advanced past the point being reverted to), calls **`self.free_resources(req)`** instead of resizing (`:2536-2538`) — this is a significant behavior: reverting can escalate to a full free of the request's cache.
- **Postcondition (normal path)**: `kv_cache.resize(pre_cap, min(history_length, pre_cap))`, then `kv_cache.suspend()` if `pre_cap > 0` (`:2539-2547`).
- **Errors**: raises `RuntimeError` on resize failure (`:2541-2545`), matching `revert_allocate_generation`'s pattern.

### `suspend_request(req) -> None` (`:2750-2754`)
Call site: `scheduler_v2.py:1063` (and `self.draft_kv_cache_manager.suspend_request(req)` at `:1065`, confirming the scheduler suspends both target and draft managers for a given request).
- No-op if cache absent or already inactive; otherwise calls `kv_cache.suspend()`. No return value, no exception surface documented beyond whatever `suspend()` itself might raise (not observed to raise in the header).

### `resume_request(req) -> bool` (`:2756-2766`)
Call sites: `py_executor.py:4834,4965` (note: **not** called from `scheduler_v2.py` directly per the grep — called from the executor, e.g. warmup/dummy-request resume paths).
- Docstring: "Returns True if the cache is (or becomes) active on GPU, False if resume was refused (e.g. GPU pressure above max_util_for_resume) or no cache exists for the request" (`:2758-2761`) — this is the most complete documented failure-mode explanation of any resume-family method: **explicitly cites `max_util_for_resume`** as a rejection cause (matches `KVCacheManagerConfig::maxUtilForResume`, config.h:271-272).

### `is_request_active(request_id) -> bool` (`:2405-2408`)
Call sites: `scheduler_v2.py:1032,1080`.
- Pure query: `kv_cache is not None and kv_cache.is_active`. No mutation.

### `free_resources(request, pin_on_release=False) -> None` (`:3691-3706`)
Call sites: `scheduler_v2.py:1100` (and `self.draft_kv_cache_manager.free_resources(req)` at `:1102`); also internally from `add_dummy_requests`' `release_resources` closure, and from `revert_allocate_context`.
- **Postconditions**: finishes conversation-manager bookkeeping, drops `_allocated_draft_lens`/`_request_stats_enabled_ids` entries, pops `kv_cache_map[request_id]`. If a cache existed: discards pending stats, calls `kv_cache.close()`, clears native stats-excluded flag, and removes the IndexMapper slot **unless** the request was already early-freed via `release_index_slot` (checked against `self._early_freed_index_requests`, `:3703-3706`).
- **Idempotency note (inference)**: if no cache is found (`kv_cache is None`), it still calls `self.impl.clear_stats_excluded(request.py_request_id)` and returns (`:3697-3699`) — i.e. calling `free_resources` twice, or on a request that never had a cache, is safe with respect to the native stats-excluded flag, though the second call's `index_mapper.remove_sequence` is skipped entirely since `kv_cache_map.pop` already returned `None`.
- **`pin_on_release` parameter is accepted but never used in the method body** (`:3691`, grep of the body `:3691-3706` shows no reference to it) — **Verified fact**: this looks like a currently-inert parameter (possibly a stub for a feature not yet wired up); flagged as a discrepancy worth reconciling with any scheduler-side assumption that pinning-on-release actually happens.

### `release_index_slot(request_id) -> None` (`:3675-3690`)
Call site: `py_executor.py:7511`.
- Docstring: "Release IndexMapper slot early while keeping KV cache blocks allocated. After prefill completes on a context-only worker, the IndexMapper slot... is no longer needed. Releasing it early allows new requests to be scheduled while the KV cache blocks are still being transferred via NIXL/UCX" (`:3676-3681`).
- **Postconditions**: detaches all `base_page_index_buf` pointers (`kv_cache.set_base_page_index_buf(i, pool_idx, None)` for every beam/pool) (`:3685-3687`), calls `self.index_mapper.remove_sequence(request_id)`, and adds `request_id` to `self._early_freed_index_requests` so a later `free_resources` call knows not to double-remove the slot (`:3688-3689`).
- **Not guaranteed**: KV cache blocks themselves remain allocated — this method only releases the fixed-capacity IndexMapper slot, not cache memory. Caller must still eventually call `free_resources` to release the blocks.

### `try_commit_blocks(request) -> None` (`:3651-3673`)
Call site: only internal (called from `update_context_resources`, `:4029`, not directly from `scheduler_v2.py`) — used as a helper inside the resource-manager-facing `update_context_resources`.
- No-op if `not (enable_block_reuse and not is_draft and not request.is_dummy_request)`, or if no cache exists (`:3652-3660`).
- Commits newly-advanced tokens (`context_current_position > kv_cache.num_committed_tokens`) via `kv_cache.commit(tokens)`, then calls `kv_cache.stop_committing()` once `request.context_remaining_length == 0` (`:3662-3673`).

### `update_context_resources(scheduled_batch) -> None` (`:3992-4036`)
Call sites: `py_executor.py:3256,4381,5263`.
- Docstring: "separated from `update_resources`... because the overlap executor needs context KV cache updates to happen before next batch scheduling. Otherwise, the scheduler would under-estimate available KV cache for sliding-window attention layer" (`:3993-4000`) — an explicit ordering requirement on the caller.
- **Per-request skip condition**: `if not kv_cache.is_active: continue` with comment "iteration N+1's eviction may suspend a ctx request's KV cache while iteration N's update still needs to process it. Skip the resize — the request will be resumed by the scheduler on the next iteration" (`:4006-4012`) — this is a **documented race-tolerant skip**, an explicit non-guarantee that every context request in the batch is actually updated every call.
- **Postconditions**: conditionally resizes history length (`kv_cache.resize(None, context_current_position)`), conditionally commits blocks (`try_commit_blocks`), and on `context_remaining_length == 0`, saves the conversation drop plan and disables `enable_swa_scratch_reuse` for the cache (context/gen boundary) (`:4020-4036`).
- **Errors**: raises `ValueError` if the history-length resize fails (`:4022-4027`) — unlike the try_allocate/prepare family, this raises rather than returning bool, presumably because by this point in the pipeline a resize failure is unexpected/unrecoverable within the current step.

### `update_resources(scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None) -> None` (`:4038-4088`)
Call sites: `py_executor.py:3275,4399,5424` (via `self.resource_manager.update_resources(...)`, a generic resource-manager dispatch, not manager-specific in the grep — confirmed the manager implements this as part of the `BaseResourceManager` interface).
- For non-draft managers, first calls `_update_kv_cache_draft_token_location(...)` (module-level helper, `:677-741`, not analyzed in depth — out of the "public API" focus but flagged as a side effect).
- Iterates only `scheduled_batch.generation_requests` (context requests are explicitly handled by the separate `update_context_resources`, per its own docstring).
- Same **documented race-tolerant skip** for inactive caches as `update_context_resources` (`:4059`, identical rationale comment).
- **Draft-specific reserve reclaim**: for draft managers, `rewind_len` is inflated by `max(_kv_reserve_draft_tokens - runtime_draft_len, 0)` to reclaim unused dynamic-tree reserve slack (`:4062-4069`, see §7).
- **Completion short-circuit**: `new_capacity = None if req.state in (GENERATION_COMPLETE, CONTEXT_INIT) else kv_cache.capacity - rewind_len` (`:4070-4074`) — passing `None` capacity to `resize` presumably means "don't touch capacity", only history length is updated for completed/reinit requests (inferred from `_KVCache.resize`'s `Optional[int] capacity` signature in `kvCache.h:200-202`, which treats `nullopt` as "leave unchanged" per the Python-side convention — **Inference**, the exact native "leave-unchanged" semantics for `nullopt` are documented at the C++ signature level as `optional<int>` but the "no-op means leave field unchanged" reading is not spelled out verbatim in `kvCache.h`).
- **History length under helix/compression**: `history_length = None if (self.kv_compression_manages_history or self._has_cp_helix) else req.max_beam_num_tokens - 1` (`:4075-4081`) — "Reuse (history's consumer) is disabled under helix, and max_beam_num_tokens mixes rank-local and global counts" (`:4077-4078`).
- **Errors**: raises `ValueError` on resize failure (`:4082-4088`), same pattern as `update_context_resources`.

### `add_dummy_requests(...)` (`:3479-3649`)
Call sites: not found via the scheduler_v2/py_executor grep used (likely called from warmup/CUDA-graph-capture code elsewhere in `py_executor.py` or `model_engine`); flagged as scheduler-adjacent infra used for CUDA graph capture / dummy padding requests, not a per-iteration scheduling call. Not exhaustively traced to its callers given time budget — **Open question**: exact callers outside the two grepped files were not verified.
- Builds synthetic `LlmRequest` objects, optionally creating/resizing real KV cache resources for each (`prepare_resource=True` default) via `_create_kv_cache` + `resize`.
- **Partial-failure/rollback behavior**: defines a local `release_resources` closure that frees *all already-created* dummy requests (`requests` list) plus the current one (and the draft-manager mirror, if any) on any failure partway through the loop (`:3512-3523`, invoked at `:3578,3583,3593,3602,3606,3611,3637,3642`) — i.e. **`add_dummy_requests` is all-or-nothing**: any single dummy request's cache-creation/resize/resume failure causes the whole batch's already-allocated resources to be released and the function returns `None` for the whole batch, not a partial list.
- Returns `None` on any such failure, or the full `requests` list on success.

### `_create_kv_cache(...)` (private, but load-bearing for `prepare_context`/`prepare_disagg_gen_init`/`add_dummy_requests`/`_prepare_draft_resources`, and **called directly by the scheduler** for the cross-attention pool at `scheduler_v2.py:946`) (`:4135-4181`)
- **Precondition**: `assert request_id not in self.kv_cache_map` (`:4146-4148`) — creating a cache for an already-tracked request is a hard bug, not a soft failure.
- **Capacity/IndexMapper check**: if `self.index_mapper.num_free_slots() == 0`, logs a warning and **returns `None`** rather than raising — "Skipping KV cache creation; request will retry next iteration" (`:4149-4158`). This is the **documented saturation-handling contract**: IndexMapper exhaustion is an expected, retryable condition, not an error.
- **Native mutation**: `self.impl.create_kv_cache(ReuseScope(...), input_tokens, id=request_id, enable_request_stats=..., expected_prompt_length=...)` (`:4161-4167`).
- **Python-side mutation**: registers `self.kv_cache_map[request_id] = kv_cache`; conditionally tracks `_request_stats_enabled_ids`; for dummy requests, marks native stats-excluded and discards pending stats (`:4171-4173`); calls `self.index_mapper.add_new_sequence(request_id)` and wires up `base_page_index_buf` for every beam/pool from `self.host_kv_cache_block_offsets` (`:4174-4180`).
- **Not guaranteed**: does not resume the cache (native `createKvCache` returns a cache that per the C++ doc comment is "SUSPENDED" — `kvCacheManager.h:123`); callers must separately call `resume`/`_resume_and_restore`.

---

## 9b. Additional public methods (event streaming, stats, teardown, reuse-probe, prefetch) — lines 3300-4237

These were audited by a separate pass over `kv_cache_manager_v2.py:3100-4237`; citations are to that file.

### `get_iteration_stats(self) -> Optional[KVCacheV2IterationStatsReport]` (`:3364-~3463`)
- **Precondition/opt-out**: returns `None` immediately if `self.enable_stats` is `False` (`:3365-3366`).
- **Native mutation — consume-and-reset semantics**: calls `self.impl.get_and_reset_iteration_stats()` (`:3371`), `get_and_reset_ssm_snapshot_iteration_stats()` (`:3372`), `get_and_reset_iteration_suspend_resume_stats()` (`:3373-3375`), `_get_and_reset_iteration_peak_block_stats(...)` (`:3376,3379`) — all four are "get-and-reset" native calls, so **this method is not idempotent/re-entrant**: calling it twice in a row returns real data the first time and an empty/zeroed delta the second (**Inference** from the `get_and_reset_*` naming; not stated in a comment in this file). This directly resolves part of Open Question 10 below: the manager DOES expose suspend/resume counts (`recordRequestSuspended`/`recordRequestResumed` native counters, surfaced via `get_and_reset_iteration_suspend_resume_stats()`), aggregated into `suspended_requests`/`resumed_requests` fields on the returned `KVCacheV2IterationStatsReport` (`:3450-3463`) — but this is a **separate call from `get_kv_cache_stats()`** (§10), which remains GPU-tier-block-count-only.
- **Non-guarantee (inference)**: callers must call this exactly once per reporting interval or lose stats to the reset; no dedup/buffering exists in this method.

### `flush_iteration_events(self)` / `get_latest_events(self, timeout_ms=None)` (`:3345-3363`)
- **Explicit non-guarantee (comment)**: in streaming-publish mode, `get_latest_events` returns `[]` unconditionally rather than raising or blocking (`:3351-3354`) — buffered/pull-based event retrieval degrades silently to empty when streaming mode is active.
- Snapshots `self.event_manager` into a local variable specifically to avoid a concurrent-shutdown race turning it into `None` mid-call (documented thread-safety comment, `:3352-3354`); `None` event_manager also returns `[]` silently (`:3357`).

### `shutdown(self)` (`:3935-3945`) — full manager teardown
- **Postconditions**: iterates `self.kv_cache_map.values()` calling `kv_cache.close()` for each (`:3936-3937`), clears `kv_cache_map` (`:3938`) and `_request_stats_enabled_ids` (`:3939`), then calls native `self.impl.shutdown()` (`:3940`).
- **Explicit ordering guarantee (comment)**: the streaming event manager is shut down **last**, after `impl.shutdown()`, so that removal events emitted during cache/impl teardown are still flushed before the publisher stops (`:3941-3945`).
- **Explicit non-null-out invariant (comment)**: `self.event_manager` is deliberately **not** set to `None` after shutdown, so any in-flight `get_latest_events`/flush call can still safely snapshot and operate on a closed manager (`:3941-3945`) — callers must not assume `manager.event_manager is None` post-shutdown as a liveness check.

### `probe_prefix_match_length(self, ...) -> int` (`:4183-4203`)
- Returns `0` immediately if block reuse is disabled or `input_tokens` is empty (`:4195-4198`).
- **Explicit non-guarantee (docstring)**: this is a pure query — it does **not** acquire page ownership (`:4184-4194`). It also explicitly warns that `cache_salt`/`lora_task_id` must match exactly what `_create_kv_cache` will later use, or the returned match length will be silently wrong (no validation exists to catch a mismatch) — a documented correctness hazard, not an exception.

### `prefetch_for_context_tokens(self, requests) -> bool` (`:4205-4232`)
- Returns `False` immediately if block reuse disabled (`:4210-4211`). Aggregate return is `True` **iff all** requests' prefetches succeeded, but the loop does not early-exit on a per-request failure (`:4216,4229-4230`).
- **Mechanism (comment)**: creates a *transient* `_KVCache` via `self.impl.create_kv_cache(...)` that is deliberately **not** registered in `kv_cache_map`/IndexMapper, calls `kv_cache.prefetch(CACHE_LEVEL1)`, then closes it — used purely to trigger tier-prefetch as a side effect without going through the normal request-registration lifecycle (`:4212-4231`).
- **Explicit non-guarantee**: prefetch is a "best-effort hint" (comment, `:4226-4227`); failures are logged per-request and folded into the aggregate boolean — no exception path.

### `get_batch_cache_indices` / `_get_batch_cache_indices_by_pool_id` / `get_batch_cache_indices_flat` (`:3714,3734,3774`)
- Read-only accessors converting native per-request `base_page_indices` into block-table tensors/lists for the framework. **No explicit guard** on `self.kv_cache_map[req_id]` lookups — a missing request id raises a raw, undocumented `KeyError` (`:3755,3809`), not a domain-specific exception.
- `get_batch_cache_indices_flat`'s docstring states the returned CPU tensor is "ready for an async H2D copy" but the method does **not** itself perform the copy or guarantee stream ordering — that is the caller's responsibility (`:3780-3791`).

### `copy_batch_block_offsets(...)` (`:4090-4130`)
- **Hard precondition, `assert`-only (not exception)**: `assert beam_width == 1` (`:4101`) — beam width `>1` unsupported, and since this is a plain Python `assert` it is stripped under `-O` (**Inference**).
- Delegates to the native/bound `copy_batch_block_offsets_to_device` (imported from `kv_cache_manager_v2_utils`) unless `self.enable_swa_scratch_reuse` is set, in which case a different `_copy_batch_block_offsets_per_layer` path is used (`:4106-4110`).
- **Documented compatibility shim**: the `max_blocks` argument is accepted "for signature parity with KVCacheManager" (the V1 manager) but is not required for V2 correctness — silently ignored for that purpose (`:4099-4100`).

### `reset_reuse_state(self)` (`:4234-4237`, last method in the file)
- Calls native `self.impl.clear_reusable_blocks()` and, if present, `self.conversation_manager.clear()`. **Open question**: no comment addresses whether in-flight/active (non-reusable) blocks are affected by this call — name-only evidence.

---

## 10. Capacity/accounting query methods

### `get_num_available_tokens(*, token_num_upper_bound, batch_size=1, max_num_draft_tokens=0) -> int` (`:2339-2365`)
- Delegates to native `self.impl.clamp_max_seq_len_for_mem(batch_size, token_num_upper_bound + extra_tokens) - extra_tokens`, where `extra_tokens = num_extra_kv_tokens + max_num_draft_tokens` (`:2353-2359`).
- **Explicit GPU-only cap**: "clamp_max_seq_len_for_mem considers all tiers (GPU + host). When max_tokens is explicitly set, cap by GPU-only capacity so callers (e.g. CUDA graph warmup) don't exceed the GPU pool" — clamps to `self._gpu_max_tokens - extra_tokens` if `_gpu_max_tokens` was set (`:2360-2364`). **This means the return value's tier semantics differ depending on whether `max_tokens` was explicitly configured** — with explicit `max_tokens`, the result is GPU-only capacity; without it, the result may reflect combined GPU+host capacity (per the native method's own documented behavior).
- **Helix unit note**: "under helix the backend runs on the ledger tokens_per_block, so the returned capacity... is in GLOBAL ledger tokens" (`:2344-2346`).

### `get_cache_bytes_per_token()`, `get_layer_bytes_per_token(...)`, `get_cache_size_per_token(...)` — static/instance helpers for pre-construction capacity planning (signatures at `:3822,3837,3967`); `get_cache_size_per_token` is a `@staticmethod` used before any manager instance exists, for capacity estimation during config resolution (used by `_util.py`'s `KvCacheCreator.configure_kv_cache_capacity`, confirmed by `_util.py` grep hit "KVCacheManagerV2 divides the quota derived from max_tokens across its..." around line 996).

### `get_kv_cache_stats()` (`:3314-3343`)
- Populates a `KvCacheStats` struct (bindings type) with: `max_num_blocks`/`free_num_blocks`/`used_num_blocks` (derived from summing `pool_group_stats[...].total`/`.available` at `GPU_LEVEL` only — **GPU tier only**, not host/disk, per `self._get_storage_statistics(GPU_LEVEL)` at `:3316`), `tokens_per_block`, `alloc_total_blocks`/`alloc_new_blocks`/`reused_blocks`/`missed_blocks` (from `self.impl.get_committed_stats()`), `cache_hit_rate` (0.0 guarded against div-by-zero, `:3330-3334`), `num_free_blocks_per_window_size` (per-SWA-window breakdown), `allocated_bytes = self.impl.get_quota(GPU_LEVEL)`.
- **Explicit scope limitation (verified by code, not comment)**: this method reports **only GPU-tier** stats; no host/disk free/used block counts are surfaced through `KvCacheStats`. **Open question for scheduler reconciliation**: if the scheduler needs host/disk tier occupancy to make suspend/resume decisions, it must get that from elsewhere (`get_iteration_stats()`, not analyzed in per-field depth here, is a stronger candidate — it builds `KVCacheV2IterationStatsReport`/`KVCacheV2PoolGroupIterationStats`/etc., `kv_cache_manager_v2.py:3364` onward, not read in this pass — **flagged but not verified**).

### `can_evict` (instance attribute, not a method) (`:1241`)
- `= len(config.cache_tiers) > 1` — set once at construction time and never updated afterward (no setter found in the read ranges). Consumed directly by the scheduler as a plain boolean gate (`scheduler_v2.py:1175,1180,1218`).

---

## 11. Explicit non-guarantees collected (cross-reference)

1. `prepare_context` explicitly does **not** resize capacity (`:2580`).
2. `resize_context` / `prepare_disagg_gen_init` on failure **suspend** (don't free) the first-chunk cache — cache remains registered in `kv_cache_map` but inactive (`:2672-2674`, `:2703-2706`).
3. `revert_allocate_generation` silently no-ops (does not revert) if the computed reverted capacity would be negative (`:2515-2517`).
4. `update_context_resources` / `update_resources` silently **skip** (do not update, do not error) any request whose cache is currently suspended due to a race with concurrent eviction — documented as intentional, resolved by the next iteration's resume (`:4006-4012`, `:4054-4059`).
5. `free_resources`' `pin_on_release` parameter is accepted but appears unused in the method body (`:3691-3706`) — possible dead/stub parameter.
6. `get_num_free_blocks` is misleadingly named (returns total pool capacity, not free capacity) and is only correct under the caller-enforced precondition that `kv_cache_map` is empty (`:2367-2372`).
7. `get_kv_cache_stats()` reports GPU-tier-only occupancy; no host/disk occupancy in that struct (`:3314-3343`, inferred from code, no explicit comment).
8. `layerGrouping()` iteration order is explicitly NOT part of the native API contract (`kvCacheManager.h:205-208`).
9. Native `createKvCache` returns a cache in **SUSPENDED** status; callers must explicitly resume it (`kvCacheManager.h:123`).
10. Host-tier construction fallback (`USE_NO_HOST`) can silently disable secondary-tier eviction (`can_evict = False`) for ranks whose host tier construction would have succeeded, because the fallback is fleet-wide once any rank fails (`kv_cache_manager_v2.py:1189,1198-1236`).
11. `KVCacheManagerV2` does not support two-model speculative decoding with per-manager-different `max_gpu_total_bytes` for target vs. draft (hard `assert`, `_util.py:2083-2087`).
12. `KVCacheManagerV2` does not support draft managers, block reuse, or attention-DP together with helix context parallelism (`kv_cache_manager_v2.py:877-897`).
13. `resize_context` explicitly refuses helix (non-dummy) requests via `ValueError` — helix requests must go through the disagg-generation path only (`:2655-2661`).

---

## 12. Test-derived evidence (behavior implied by test assertions, not manager source directly)

All citations below are to test files, not to `kv_cache_manager_v2.py`; each is labeled per the audit's evidence classes.

### 12.1 Host tier auto-sizing (`tests/unittest/_torch/executor/test_kvv2_host_tier_sizing.py`)
- **Test assertion**: 1 rank, ample memory → `_compute_auto_host_tier_quota` returns the device `quota` unchanged (`:24-34`). N co-located ranks → capped at `mem_available/local_ranks*0.5` (`:36-44`), and per-rank×local_ranks stays `≤ mem_available` (`:46-55`). `RLIMIT_MEMLOCK` caps at `memlock_limit*0.8` (`:57-63`). Unknown limits (`inf`) fall back to device quota (`:65-74`). Non-positive result falls back to device quota (`:76-89`); result is always positive (`:91-101`).
- **Test assertion**: `_sync_host_tier_quota` is a no-op for `world_size==1` (`:137-144`); for multi-rank, does `allreduce(MIN)` — docstring/comment cites this as a regression guard for a hang tracked as PR #17380/TRTLLM-15179 (`:104-181`).

### 12.2 Helix capacity sizing and manager-combination constraints (`tests/unittest/_torch/executor/test_kv_cache_manager_v2_helix_superblock.py`)
- **Test assertion**: if `max_gpu_total_bytes` is already set, `_configure_helix_kv_cache_capacity` is a no-op (`:258-261`); explicit-but-non-positive raises `ValueError` matching "must be positive" (`:262-265`); if unset, computed as `free*fraction` and written to `max_gpu_total_bytes` (not `max_tokens`, to avoid double-inflation by `1/max_util_for_resume`) (`:266-271`); zero free memory raises `ValueError` matching "free memory" (`:272-275`).
- **Test assertion**: V1 `KVCacheManager` + Helix CP raises `NotImplementedError` matching "V2 KV cache manager" — V1 interprets `max_tokens` as rank-local while V2 emits super-block-ledger coordinates (`:278-290`).
- **Test assertion**: under Helix, when `try_allocate_generation` returns `False`, the scheduler does not fall through to its normal eviction path — it raises `RuntimeError` matching "eviction is disabled under helix" instead, described as a "precedent-consistent no-evict stance" (`:324-355`), implying eviction-on-allocation-failure **is** the normal (non-Helix) scheduler behavior and is specifically disabled under Helix.

### 12.3 Disk/host tier init-failure semantics (`tests/unittest/_torch/executor/test_kv_cache_manager_v2.py`)
- **Test assertion**: host tier with `host_cache_size=0` + disk tier still yields `manager.can_evict == True` and `cache_tiers == [GpuCacheTierConfig, DiskCacheTierConfig]` (`:335-353`) — corroborates §6.2/§6.3 (explicit zero host quota drops the host tier, but a disk tier alone still enables eviction).
- **Test assertion**: disk tier init failure does **not** fall back to host — raises `_CacheTierInitError` matching "disk tier init failed" (`:356-366`), unlike host-tier failure which has the two-phase fallback protocol (§6.4). If a peer rank's fallback fails while the local rank's succeeds, `RuntimeError` matching "failed on another rank" is raised and **both** the initial and fallback native `impl` objects are explicitly `.shutdown()`'d on the local rank (`:427-451`) — confirms §6.4's abort path does not leak the fallback candidate either.
- **Test assertion**: `avg_seq_len` config exceeding `max_seq_len` raises `ValueError` matching "avg_seq_len" (`:328`) — a construction-time input-validation error not otherwise covered above.

### 12.4 Suspend/resume contract at the per-sequence `_KVCache` level (`tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py`)
- **Test assertion**: lifecycle `resume(stream)` → `stop_committing()` → `suspend()` → status `SUSPENDED` → `resume(stream)` returns `True` and status returns to `ACTIVE` (`:2365-2395`). For SSM layers, the SSM block base index (slot) is identical before and after a suspend/resume cycle — i.e. resume is guaranteed to restore the **same physical slot**, not just logical continuity (`:2381-2394`).
- **Test name only (not read in depth by the sub-audit)**: `test_resume_rejects_if_any_pool_group_exceeds_threshold` (`:747`) — corroborates the `resume_request` docstring's "refused... e.g. GPU pressure above max_util_for_resume" behavior (§9), but the exact rejection condition per pool-group was not independently verified against this test body. **Flagged as needing a closer read if resume-rejection semantics matter for scheduler reconciliation.**
- **Test assertion**: `OutOfPagesError` is raised on GPU page exhaustion during allocation — `self.assertRaises(OutOfPagesError, lambda: self.run_naive(seq_len + 1, 1, False))` (`:743-745`), with additional raise sites at `:429,447,2241`.

### 12.5 Capacity-only mode and per-manager `kv_compression_manages_history` scoping (`tests/unittest/_torch/executor/test_kv_cache_v2_capacity_only.py`)
- **Test assertion**: default (non-capacity-only) generation resize for `rewind=3`, `max_beam_num_tokens=201` → `cache.resize.assert_called_once_with(253, 200)` (`:96-104`).
- **Test assertion — resolves part of §7's target/draft independence open question**: `kv_compression_manages_history` is a **per-manager-instance flag**, not global — with `target.kv_compression_manages_history=True` and `draft.kv_compression_manages_history=False` on the *same request*, the draft cache gets `resize(253, 200)` (history tracked) while the target cache gets `resize(253, None)` (history suppressed) (`test_capacity_only_is_scoped_to_target_manager`, `:107-121`). This directly confirms target and draft managers are independently configurable and independently invoked, consistent with §7's "Inference" that they are fully separate native objects.
- **Test assertion**: dynamic-tree draft-token reserve reclaim differs by `is_draft`: with `_kv_reserve_draft_tokens=60`, `is_draft=True` → expected capacity 201 ("draft reclaims reserve"); `is_draft=False` → expected capacity 230 ("target has no reserve") (`:124-138`) — corroborates §7's `update_resources` draft-reserve-reclaim description with concrete numbers.
- **Test assertion**: on `GENERATION_COMPLETE`, `resize.assert_called_once_with(None, None)` — both capacity and history left at manager default (`:141-149`), corroborating §9's "completion short-circuit" entry (capacity `None`) and additionally showing history is `None` too in this specific completion case.
- **Test assertion**: a suspended cache (`cache.is_active=False`) is skipped entirely by `update_resources` — `cache.resize.assert_not_called()` (`:152-160`) — corroborates §11.4's documented race-tolerant skip.

### 12.6 Extra-buffers-per-layer / sparse Index-K hook (`tests/unittest/_torch/executor/test_kv_cache_v2_extra_buffers.py`)
- **Test assertion**: default `_extra_buffers_per_layer` hook returns nothing; standard layers get exactly `[Role.KEY, Role.VALUE]` (plus `KEY_BLOCK_SCALE`/`VALUE_BLOCK_SCALE` for NVFP4), never `Role.INDEX_KEY`, unless a subclass overrides the hook (`:122-158,187-217`).
- **Test assertion**: registering a duplicate/colliding role (e.g. `Role.KEY`) via the hook raises `AssertionError`, both from the hook's own assertion and from `AttentionLayerConfig.__post_init__` (`:267-272`) — corroborates `config.h`'s "no duplicate buffer roles" `validate()` check (§3) at the Python-binding level too.
- **Test assertion**: `get_index_k_buffer(layer_idx, ...)` returns `None` for any layer without the role registered, **including universally on the plain base `KVCacheManagerV2`** — explicitly documented in the test as an intentional "lets generic callers probe the accessor without breaking dense-only models" contract (`:304-325,526-543`). On a registered sparse layer it returns a zero-copy torch view aliasing the same underlying pool as the K/V buffers for that layer (verified via matching `data_ptr()`/pool base address, `:353-390,448-473`); rejects unsupported `kv_layout` with `ValueError` matching "Unsupported kv_layout" (`:511-524`) and mismatched `head_dim`/`dtype` with `AssertionError` (`:475-509`).

### 12.7 Dual-pool (encoder/decoder cross-attention) manager pairing — NOT draft/target (`tests/unittest/_torch/executor/test_dual_pool_kv_cache.py`)
- **Clarification**: despite the name, this file covers a *second independent manager instance* pattern for **encoder-decoder cross-attention** (self-pool + cross-pool), not speculative-decoding draft/target pairing — draft/target linkage evidence is in §7/§12.5 instead.
- **Test assertion**: `cross_kv_cache_fraction` is required for encoder-decoder models — `ValueError` matching "cross_kv_cache_fraction" if unset (`:247-254`). Budget (`max_gpu_total_bytes` and, separately, `host_cache_size`) is split proportionally between `self_config`/`cross_config` by that fraction, summing back exactly to the original total; the base `config` object is never mutated in place (`:208-342`). When no explicit `max_gpu_total_bytes` is set, the split instead divides `free_gpu_memory_fraction` (`:255-287`).
- **Test assertion**: the cross manager is constructed using **encoder** geometry (num_layers/num_kv_heads/head_dim/max_seq_len from the encoder side of the HF config), registered under `ResourceManagerType.CROSS_KV_CACHE_MANAGER`, and works with either `KVCacheManagerV2` or the V1 `KVCacheManager` class interchangeably via an injected `kv_cache_manager_cls` (`:353-482,578-596`).

### 12.8 Error/exception assertion table (test-file evidence, consolidated)

| Exception | Trigger | Test citation |
|---|---|---|
| `ValueError` ("must be positive") | Helix: explicit `max_gpu_total_bytes=0` | `test_kv_cache_manager_v2_helix_superblock.py:264` |
| `ValueError` ("free memory") | Helix: `torch.cuda.mem_get_info()` reports 0 free | `:274` |
| `NotImplementedError` ("V2 KV cache manager") | V1 manager + Helix CP combo | `:289` |
| `RuntimeError` ("eviction is disabled under helix") | `try_allocate_generation` fails under Helix | `:342` |
| `ValueError` ("cross_kv_cache_fraction") | enc-dec dual-pool, fraction unset | `test_dual_pool_kv_cache.py:252` |
| `ValueError` ("avg_seq_len") | `avg_seq_len` config exceeds `max_seq_len` | `test_kv_cache_manager_v2.py:328` |
| `_CacheTierInitError` ("disk tier init failed") | disk tier native init raises; no host-style fallback | `:357` |
| `RuntimeError` ("fallback init failed") | host tier init fails AND host-fallback also fails | `:408` |
| `RuntimeError` ("failed on another rank") | local fallback succeeds but a peer rank's fails | `:440` |
| `OutOfPagesError` | GPU page exhaustion on allocation | `kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py:745` (assertion), `:429,447,2241` (raise sites) |
| `AssertionError` | duplicate `Role.KEY` registered via extra-buffers hook | `test_kv_cache_v2_extra_buffers.py:271` |
| `AssertionError` | `get_index_k_buffer` called with wrong `head_dim`/`dtype` | `:483,500` |
| `ValueError` ("Unsupported kv_layout") | `get_index_k_buffer(..., kv_layout="HDN")` | `:514` |

---

## Open Questions for Scheduler-Side Reconciliation

1. **`get_kv_cache_stats()` GPU-only scope** — does the scheduler (esp. the MAX_UTILIZATION path that relies on suspend/resume) ever assume this reports host/disk occupancy too? It doesn't (§10). Needs checking against `get_iteration_stats()` (not analyzed here) as the likely correct source for host/disk numbers.
2. **`free_resources(..., pin_on_release=...)`** — parameter appears unused in the current method body (§9, §11.5). Does the scheduler pass `pin_on_release=True` anywhere expecting an effect that doesn't currently happen? (Requires grepping scheduler call sites for this kwarg specifically — not done here since the call site inventory only captured the bare method name via a broad grep pattern that would also match this.)
3. **Suspended-cache skip races in `update_resources`/`update_context_resources`** (§11.4) — does the scheduler's own bookkeeping (e.g. counts of "active" requests) stay consistent with a request silently not being updated for one iteration? This requires scheduler-side control-flow analysis, explicitly out of scope here.
4. **`can_evict` fleet-wide degradation** (§11.10, §6.4) — if one rank's host tier allocation fails, ALL ranks lose the host tier and `can_evict` becomes `False` everywhere. Does the scheduler have any way to detect/log this transition at runtime, or does it just silently behave as if no host tier was ever configured? Not verifiable from manager-side code alone.
5. **`add_dummy_requests` callers** — not found in the two files grepped (`scheduler_v2.py`, `py_executor.py`); likely called from CUDA-graph warmup code elsewhere. Exact caller(s) and how they handle the `None`-on-failure return were not traced.
6. **Native `resize()` return-value semantics** — the C++ doc comment ("Returns true if the resize was a no-op shortcut") is ambiguous about the success-but-not-no-op case (§4). The Python wrapper treats any falsy return uniformly as failure; whether there's a distinct "no-op true" vs "resized true" signal that any caller could exploit (or that scheduler code assumes) is unclear without reading `kvCache.cpp`.
7. **`terminateOnException` usage sites** (§5) — unclear which (if any) of the public `KvCacheManager`/`KvCache` methods are wrapped such that an internal C++ exception causes `std::terminate()` (process abort) rather than propagating to Python as a catchable exception. This materially affects what "error handling" the scheduler can actually rely on for e.g. OOM conditions — requires reading the `.cpp` files and/or the nanobind glue, not done here.
8. **Independence of target/draft native objects** (§7 closing open question) — confirmed each `KVCacheManagerV2` wraps its own native manager instance, but whether there's any hidden cross-manager coupling at the native level (e.g. shared CUDA memory pools, shared event sink) beyond what's visible in the Python constructor code was not verified against `kvCacheManager.cpp`.
9. **`DiskCacheTierConfig::assertValid()` and `KVCacheManagerConfig::validate()` bodies** — only declared in `config.h`; not read from `config.cpp`, so the actual validation rules (e.g. tier-ordering enforcement, `max_gpu_total_bytes`-adjacent checks) beyond the trivial `quota > 0` checks are unverified.
10. **[Partially resolved, see §9b]** `get_iteration_stats()` exposes `suspended_requests`/`resumed_requests` counts (native `recordRequestSuspended`/`recordRequestResumed` counters) via `KVCacheV2IterationStatsReport`, and is a "get-and-reset" call (not idempotent/re-callable without losing data — **Inference**). Still open: whether it also surfaces host/disk *block/byte* occupancy (as opposed to request-suspend counts) was not confirmed — `get_kv_cache_stats()` (§10) is confirmed GPU-tier-only, but the finer-grained pool-group/lifecycle breakdown fields of `KVCacheV2IterationStatsReport`/`KVCacheV2PoolGroupIterationStats` were not catalogued field-by-field.
11. **Per-pool-group resume-rejection threshold** — `test_resume_rejects_if_any_pool_group_exceeds_threshold` (`tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py:747`) confirms `resume`/`resume_request` can be rejected on a **per-pool-group** utilization basis (matching `maxUtilForResume` in the native config), but the exact rejection rule (e.g. whether ANY over-threshold pool group blocks the whole resume, or only pool groups touched by that request) was not read in this audit — needs a closer read of that test body plus `KvCache::resume`'s C++ implementation.
12. **`free_resources`'s unused `pin_on_release` parameter** — confirmed dead in the current method body across two independent read passes (§9, §9b, §11.5); worth flagging to code owners regardless of scheduler-side impact, since it may indicate an in-progress or reverted feature.
13. **`_extra_buffers_per_layer` / `Role.INDEX_KEY` interaction with capacity accounting** — the extra-buffers tests (§12.6) confirm the buffers are registered and zero-copy-accessible, but this audit did not verify whether `get_kv_cache_stats()`/`get_iteration_stats()` capacity accounting includes or excludes these extra per-layer buffer bytes when reporting GPU-tier used/free blocks.
