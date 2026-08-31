# KVCacheV2: Topology Enumeration + Target/Draft Prefix-Reuse Correctness

Commit: `4716843cee6e7a6c08bf4d8be29fae25321a9344`, branch `feat/native-kv-events-clean`, 2026-08-31.

Read-only pass. Does not repeat the general scheduler/manager contract audit in
`{scheduler,manager,interface_map}.md` / `TRTLLM-15289_audit.md`. Focus: (1) full
target/draft topology enumeration with resource-ownership detail, (2) prefix-reuse
correctness between target and draft managers.

Confidence labels: **Verified fact** (read directly this session), **Inference**
(reasonable conclusion from what was read, not stated verbatim), **Open question**
(unresolved / needs native `.cpp` or runtime evidence).

---

## Task 1: Target/Draft Topology Enumeration

### Variant A — Two-model speculative decoding (separate draft *engine*)

**Selection condition (Verified fact).**
`KvCacheCreator.build_managers`:
```
has_draft = (
    self._draft_model_engine is not None  # two-model
    or self._should_create_separate_draft_kv_cache())  # one-model
...
if self._draft_model_engine is not None:
    draft_kv_cache_manager = self._create_kv_cache_manager(
        self._draft_model_engine, estimating_kv_cache,
        kv_cache_config_override=draft_build_kv_cache_config)
elif self._should_create_separate_draft_kv_cache():
    draft_kv_cache_manager = self._create_one_model_draft_kv_cache_manager(...)
```
`tensorrt_llm/_torch/pyexecutor/_util.py:2032-2094`. The `self._draft_model_engine is not None` branch is checked first and wins whenever a distinct draft `ModelEngine` object exists (classic two-model / EAGLE with external drafter / draft-target speculative decoding).

A `KVCacheManagerV2` for this branch is built by `_create_kv_cache_manager` (the `KvCacheCreator` instance method at `_util.py:1360`), which forwards to `model_engine.is_draft_model` to set `is_draft` (`_util.py:2198-2253`, `is_draft = model_engine.is_draft_model` at the module-level `_create_kv_cache_manager`, line ~2253).

**Request-ID ownership (Verified fact).**
The draft engine does **not** reuse the target's `LlmRequest` object; `ModelDrafter._create_draft_request` builds a brand-new `LlmRequest` but explicitly passes the **same numeric ID**:
```python
return LlmRequest(
    input_tokens=input_tokens,
    request_id=request.py_request_id,   # <-- same integer ID as target
    ...
    is_draft=True, ...)
```
`tensorrt_llm/_torch/speculative/model_drafter.py:106-123`. So target and draft managers key their `kv_cache_map`/`IndexMapper` by the *same* `request_id` integer, but the `LlmRequest` Python object instances differ and carry independently-managed fields (`context_current_position`, `context_chunk_size`, etc.) — see Task 2 below for why this matters.

**IndexMapper ownership (Verified fact).**
Each `KVCacheManagerV2.__init__` call constructs its own `self.index_mapper = IndexMapper(index_mapper_capacity, max_beam_width)` (`kv_cache_manager_v2.py:1349-1360`). Since the draft manager is a fully separate `KVCacheManagerV2` Python/native object (built by a separate `_create_kv_cache_manager` call), it gets its own `IndexMapper` instance — not shared with the target's.

Disagg capacity doubling: `index_mapper_capacity = max_num_sequences * (2 if is_disagg else 1) + num_reserved_index_slots` (`kv_cache_manager_v2.py:1351-1352`), with the comment explaining the 2x is to let generation-active requests (up to `max_num_sequences`) and requests still in `TRANS_IN_PROGRESS` KV transfer (another up to `max_num_sequences`) hold slots simultaneously (`kv_cache_manager_v2.py:1339-1348`). `is_disagg` is threaded uniformly from `KvCacheCreator._is_disagg` into **both** the target and draft manager constructors (`_util.py:1401` target, `_util.py:1591` one-model draft) — **Verified fact** that both managers get the same doubling behavior when disagg is enabled; whether context-only vs generation-only *worker processes* each instantiate their own target+draft manager pair, or whether the draft manager is even created on a context-only worker, was **not traced this session** (Open question — would require reading the disagg worker-role wiring, e.g. `py_executor.py`'s disagg setup, not covered here).

**Pool ownership (Verified fact).**
Each `KVCacheManagerV2` builds its own native `self.impl = KVCacheManagerPy(config, ...)` (`kv_cache_manager_v2.py:1217`, `1240`), i.e. its own pools, its own `cache_tiers`, its own `can_evict` (`self.can_evict = len(config.cache_tiers) > 1`, `kv_cache_manager_v2.py:1241`). Two-model draft therefore has physically separate GPU/host/disk pools from the target.

GPU budget is **not** split between target and draft for `KVCacheManagerV2` in the two-model case — enforced by an assertion:
```python
assert (draft_kv_cache_config.max_gpu_total_bytes ==
        self_kv_cache_config.max_gpu_total_bytes), (
            "KVCacheManagerV2 does not support two-model "
            "speculative decoding with separate draft GPU "
            "budgets.")
```
`_util.py:2079-2084`. Offload-tier (host/disk) budgets **are** split per manager via `_split_kv_cache_budget_for_draft` (`_util.py:2038-2049`).

**Eviction ownership (Verified fact).**
`can_evict` and the native eviction/suspend-resume/radix-tree state all live inside each manager's own `self.impl`, so eviction is scoped per-manager independently — there is no code path found that coordinates eviction decisions between target and draft `KVCacheManagerV2` instances.

**Prefix-reuse ownership (Verified fact, detailed in Task 2).**
Draft manager has its own native reuse structure (own `self.impl`, hence own radix tree) but the Python wrapper **never invokes reuse matching for the draft manager**: `_prepare_draft_resources` always calls `self._create_kv_cache(req.py_request_id, req.lora_task_id, None, ...)` with `input_tokens=None` (`kv_cache_manager_v2.py:2791-2797`), unconditionally (not gated on `self.enable_block_reuse`). See Task 2.

---

### Variant B — One-model speculative decoding, separate draft-layout manager (Eagle3/MTP/DSpark with `use_separate_draft_kv_cache`)

**Selection condition (Verified fact).**
```python
def _should_create_separate_draft_kv_cache(self) -> bool:
    if self._mapping.enable_attention_dp:
        return False
    sparse_cfg = self._sparse_attention_config
    if (sparse_cfg is not None
            and getattr(sparse_cfg, "algorithm", None) == "deepseek_v4"
            and self._mapping.pp_size > 1):
        return False
    return should_use_separate_draft_kv_cache(self._speculative_config)
```
`_util.py:1432-1461`. It requires *no* attention-DP, *not* (DeepSeek-V4-sparse AND pp_size>1), and `should_use_separate_draft_kv_cache(spec_config)` to be `True`:
```python
def should_use_separate_draft_kv_cache(spec_config) -> bool:
    if spec_config is None: return False
    if not spec_config.spec_dec_mode.use_one_engine(): return False
    if spec_config._use_shared_kv_cache: return False
    if (spec_config.spec_dec_mode.is_dspark()
            and spec_config.draft_is_embedded_in_target):
        return False
    return spec_config._allow_separate_draft_kv_cache
```
`tensorrt_llm/_torch/speculative/interface.py:110-128`. This is the "one engine, but draft layers have a different KV layout (head_dim/num_kv_heads/etc.), so they need their own pool" case — e.g. classic EAGLE3 where the draft layer's hidden size differs from the target's.

Manager construction: `_create_one_model_draft_kv_cache_manager` (`_util.py:1519-1596`) builds a second `KVCacheManagerV2` with `is_draft=True`, `layer_mask=self._get_one_model_draft_layer_mask()`, `num_layers=self._get_num_draft_layers()` (`_util.py:1590-1594`), using the effective draft `ModelConfig` (explicit draft config, or falls back to the target's config for MTP since MTP layers share the target's architecture — `_util.py:1465-1479`).

The *target* manager, when this variant is active, is built with an explicit layer mask that **excludes** the draft layers:
```python
spec_dec_layer_mask = None
if self._should_create_separate_draft_kv_cache():
    num_target_layers = model_engine.model.model_config.pretrained_config.num_hidden_layers
    spec_dec_layer_mask = [True] * num_target_layers
```
`_util.py:1372-1379` — comment: *"use layer_mask to include only target layers. The draft layers should only be in the separate draft KV cache manager."*

**Request-ID / request-object ownership (Verified fact + Inference).**
Unlike the two-model case, there is **one** model engine and **one** forward call per step; `resource_manager.prepare_resources()` is invoked on all resource managers (target `KVCacheManagerV2` and draft `KVCacheManagerV2`) against the **same** `ScheduledRequests`/`LlmRequest` objects (Inference — no separate `LlmRequest` clone construction analogous to `ModelDrafter._create_draft_request` was found for this variant; `_prepare_draft_resources` reads `req.context_current_position`, `req.context_chunk_size` directly off the same `req` passed to `prepare_resources`, `kv_cache_manager_v2.py:2779-2821`). So request ID *and* the `context_current_position` field are literally shared with the target for this variant (not just numerically equal as in Variant A) — this is a stronger form of coupling than Variant A and is the basis of the Task 2 "same forward call, same chunk boundary" analysis below.

**IndexMapper / Pool / Eviction ownership.** Same as Variant A: separate `KVCacheManagerV2` instance ⇒ own `IndexMapper` (`kv_cache_manager_v2.py:1360`), own `self.impl` pools/eviction (`kv_cache_manager_v2.py:1240-1241`). **Verified fact.**

**Prefix-reuse ownership.** Same unconditional `input_tokens=None` in `_prepare_draft_resources` (`kv_cache_manager_v2.py:2791-2797`) applies — this code path is shared between Variant A and Variant B (`prepare_resources` dispatches to `_prepare_draft_resources` whenever `self.is_draft` is `True`, `kv_cache_manager_v2.py:2770-2777`, regardless of whether the draft manager belongs to a two-model or one-model-separate topology). **Verified fact.**

---

### Variant C — Shared-manager / folded topology (draft layers inside the unified target `KVCacheManagerV2`)

**Selection condition (Verified fact).**
This is simply the negation of Variant B's condition when there is no separate draft *engine* either: `has_draft` in `build_managers` is `False` when `self._draft_model_engine is None` and `self._should_create_separate_draft_kv_cache()` is `False` (`_util.py:2032-2034`). In that case `_create_kv_cache_manager` (the `KvCacheCreator` method) computes `spec_dec_layer_mask = None` (the `if self._should_create_separate_draft_kv_cache():` guard at `_util.py:1373` is skipped), so the manager is constructed with `layer_mask=None` at `_util.py:~1397` — **Inference**: `layer_mask=None` means "no restriction," i.e. *all* layers (target attention layers **and** MTP/embedded draft layers) live in the one `KVCacheManagerV2` instance and its native `self.impl`. This is the MTP-with-shared-layout case (`spec_config._use_shared_kv_cache == True` → `should_use_separate_draft_kv_cache` returns `False`), the DSpark-embedded-in-target case (`spec_config.draft_is_embedded_in_target`), the attention-DP case (forced fold regardless of config, `_util.py:1435-1441`), and the DeepSeek-V4-sparse-with-pp_size>1 case (forced fold, `_util.py:1447-1461`, with an explicit log message: *"DeepSeek-V4 separate draft KV cache is only supported for PP=1; folding draft layers into the unified manager for pp_size=%d."*).

**Request-ID / IndexMapper / Pool / Eviction / Reuse ownership (Inference, follows trivially from "one instance").**
There is exactly one `KVCacheManagerV2` object, one `self.impl`, one `IndexMapper`, one set of native pools, one eviction/radix-tree structure. `resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]` is set to `None` in this variant (`draft_kv_cache_manager = None` unless one of Variant A/B fired, `_util.py:2074-2094`, then `resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] = draft_kv_cache_manager` at `_util.py:2100-2101`). Because target and draft layers share the *same* `_KVCache` object per request (one `create_kv_cache` call, one reuse match), a reused prefix is available to *all* layer groups in that one manager identically — **this variant is structurally immune to the Task 2 concern** (see Task 2 §4a).

---

### Variant D — Aggregate vs. disaggregated execution

**Verified fact.** `is_disagg` is a single boolean (`KvCacheCreator._is_disagg`, set at `_util.py:592`) threaded uniformly into every manager constructor call this session found: the main target manager (`_util.py:1401`), the one-model separate draft manager (`_util.py:1591`), and `get_kv_cache_manager_cls(... is_disagg=self._is_disagg)` for class selection (`_util.py:1557-1558`). Both target and draft managers (when a separate draft manager exists) get the same `2x` `IndexMapper` capacity multiplier when `is_disagg=True` (`kv_cache_manager_v2.py:1351-1352`).

**Open question.** Whether a context-only disagg worker process constructs (or needs) a draft `KVCacheManagerV2` at all — and whether/how `draft_kv_cache_manager` participates during the context→generation handoff (`TRANS_IN_PROGRESS`) — was not traced this session; it would require reading the disagg role-split / worker-startup code (not in `_util.py`/`kv_cache_manager_v2.py`) and is out of scope for what was read. What **is** verified is that the capacity-doubling exists specifically to let slots for "actively generating" and "in KV transfer" requests coexist (`kv_cache_manager_v2.py:1343-1348` comment), implying that during a handoff a request's `IndexMapper` slot on a given manager instance is held (not released) until the transfer completes — but this comment is manager-generic, not draft-specific, and it was not confirmed whether the draft manager's own slots participate in the same handoff protocol on a disagg worker.

---

## Task 2: Prefix-Reuse Correctness Between Target and Draft

### 1. Block/page selection for a reused prompt — target path

`_prepare_context_impl` (target manager, `is_draft=False`), first context chunk:
```python
if self.enable_block_reuse:
    tokens = self._augment_tokens_for_block_reuse(all_tokens, req, end=len(all_tokens) - 1)
else:
    tokens = None
kv_cache = self._create_kv_cache(req.py_request_id, req.lora_task_id, tokens, ...)
...
req.context_current_position = kv_cache.num_committed_tokens
req.set_prepopulated_prompt_len(kv_cache.num_committed_tokens, self.tokens_per_block)
```
`kv_cache_manager_v2.py:2592-2629`. `_create_kv_cache` forwards `tokens` as `input_tokens` to the single native call that resolves reuse:
```python
kv_cache = self.impl.create_kv_cache(
    ReuseScope(lora_id=lora_task_id, salt=salt_int),
    input_tokens, id=request_id, ...)
```
`kv_cache_manager_v2.py:4161-4167`. The native header documents `createKvCache`'s `inputTokens` param as *"optional sequence to match against existing cached blocks"* (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h:118-136`), and a separate `matchReuse`/`probeReuse` pair exists for pure lookups (`kvCacheManager.h:143-145`). **Verified fact**: this reuse-matching call happens once, on the manager's own `self.impl`, i.e. per-manager, not shared/broadcast to any other manager instance.

### 2. Do target and draft use the same logical block mapping?

**No — verified fact that the draft manager never independently attempts reuse for the same prompt.** `_prepare_draft_resources` (called whenever `self.is_draft` is `True`, `kv_cache_manager_v2.py:2770-2777`) creates the draft's `_KVCache` with `input_tokens=None` unconditionally:
```python
kv_cache = self._create_kv_cache(
    req.py_request_id, req.lora_task_id, None,
    cache_salt=req.cache_salt, is_dummy=req.is_dummy)
```
`kv_cache_manager_v2.py:2789-2797` (the third positional arg — `input_tokens` — is a hard-coded `None`, not `self.enable_block_reuse`-gated the way the target's is at `kv_cache_manager_v2.py:2601-2606`). Passing `input_tokens=None` (empty `TokenSpan`) means the native call performs **no** prefix match — `num_committed_tokens` for the draft's fresh `_KVCache` is `0`.

The draft manager instead sizes its cache purely from **capacity/count** fields already resolved by the target/scheduler side, not from any reuse outcome of its own:
```python
draft_len = get_draft_token_length(req)
capacity = (req.context_current_position + req.context_chunk_size
            + draft_len + self.num_extra_kv_tokens)
if not kv_cache.resize(capacity):
    raise RuntimeError(...)
```
`kv_cache_manager_v2.py:2810-2821`. Note `kv_cache.resize(capacity)` is called with **one** argument here — no `historyLength` argument is passed (contrast with `prepare_disagg_gen_init`'s `kv_cache.resize(capacity, prompt_len)` at `kv_cache_manager_v2.py:2702`, and native `resize(capacity, historyLength=nullopt)` signature at `cpp/.../kvCache.h:206`). So the draft's `_KVCache.history_length` is **not** explicitly advanced to match `req.context_current_position`; it is left to whatever the native default is when `historyLength` is omitted (**Open question** — not traced in native `.cpp`, only the header declaration was read).

### 3. Is draft reuse explicitly consulted, or implicitly assumed?

**Explicitly not consulted — verified fact.** The draft's `_create_kv_cache` call always passes `input_tokens=None`; `matchReuse`/`probeReuse` are never called from `_prepare_draft_resources`, and `probe_prefix_match_length` (the wrapper around `self.impl.probe_reuse`, `kv_cache_manager_v2.py:4183-4203`) is not invoked anywhere in `_prepare_draft_resources`. Combined with the previously-known guard in `try_commit_blocks`:
```python
should_block_reuse = (self.enable_block_reuse and not self.is_draft
                       and not request.is_dummy_request)
```
`kv_cache_manager_v2.py:3651-3656` — the draft manager neither **reads** (matches) nor **writes** (commits) reusable blocks. Its radix tree (it has its own, per Task 1) is functionally inert: nothing is ever committed into it (so nothing to be reused across requests), and nothing is ever matched out of it for a given request's own prefix.

### 4. Is the reported "target prefix reuse / draft missing page" scenario reachable?

**(a) Folded/shared-manager topology (Variant C): not reachable.**
When target and draft layers share one `KVCacheManagerV2` (`layer_mask=None`, one `self.impl`, one `_KVCache` per request), the single `_create_kv_cache` call at `kv_cache_manager_v2.py:2607-2618` resolves reuse **once** for all layer groups (both target and MTP/embedded-draft layer groups) in that one physical `_KVCache`. There is no second, independent "draft" cache creation — `_prepare_draft_resources` is dead code for this topology (it is a method on the draft `KVCacheManagerV2` instance, which does not exist in Variant C; `resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]` is `None`, `_util.py:2074-2101`). So the reused prefix (whatever committed length the radix tree matched) is, by construction, the *same* physical blocks backing both target and draft layer groups — no mismatch is possible. **Verified/Inference** (verified the single-instance construction; inference that this rules out the described mismatch, since there is nothing to diverge from).

**(b) Separate-manager topologies (Variant A, two-model; Variant B, one-model-separate-layout): mechanism for divergence is reachable at the Python level; final data-corruption consequence needs native confirmation.**

Concrete trigger sequence:
1. A prompt shares a long prefix with a previously-served request (e.g. a common system prompt), and `enable_block_reuse=True` on the target manager.
2. Target's `_prepare_context_impl` matches, say, 800/1000 prefix tokens via `self.impl.create_kv_cache(scope, tokens, ...)` (`kv_cache_manager_v2.py:2607-2618`); `req.context_current_position = kv_cache.num_committed_tokens = 800` (`kv_cache_manager_v2.py:2626`).
3. Target computes the forward pass only for the *remaining* chunk `[800, 1000)` — `model_engine.py:5401-5411`: `begin_compute = request.context_current_position; end_compute = begin_compute + request.context_chunk_size; prompt_tokens = request.get_tokens_range(0, begin_compute, end_compute)`.
4. Per-iteration bookkeeping records this chunk's bounds onto the request: `request.py_last_context_chunk = (request.context_current_position, request.context_current_position + request.context_chunk_size)` (`py_executor.py:7788-7792`), i.e. `(800, 1000)` for this example.
5. **Two-model case (Variant A):** once the target produces its first generated token, `ModelDrafter._create_draft_request_for_request` builds the draft's context request and seeds its chunk bounds *from the target's last chunk only*:
   ```python
   begin_compute, end_compute = request.py_last_context_chunk
   if begin_compute is not None:
       new_request.context_current_position = begin_compute
       new_request.context_chunk_size = end_compute - begin_compute
   ```
   `model_drafter.py:140-149`. With the example numbers, the draft's own new `LlmRequest` gets `context_current_position=800`, `context_chunk_size=200` — i.e. the draft engine is told to skip computing positions `[0, 800)` too, mirroring the target's reuse-derived skip.
   **One-model case (Variant B):** since target and draft share the *same* `req` object and the *same* single forward call (Task 1, Variant B), the draft layer group is driven by the identical `req.context_current_position=800` with no separate seeding step at all — an even more direct coupling.
6. But the draft `KVCacheManagerV2`'s own `_KVCache` for this request was created via `_prepare_draft_resources` with `input_tokens=None` (step 3 above) — `num_committed_tokens=0`, and no forward computation ever ran for positions `[0, 800)` on the draft's layer group (the one-shot draft context step only computes `[800, 1000)`, per the `begin_compute`/`end_compute` slice — Variant A: `model_engine.py:5401-5411` run against the draft engine's own `scheduled_requests.context_requests`, i.e. against `new_request` with `context_current_position=800`; Variant B: the single forward call slices identically for the draft's layer group since it uses the same `req.context_current_position`).
7. Consequently the draft manager believes (via `context_current_position`/chunk bookkeeping copied from the target) that positions `[0, 800)` are already resident and valid, while its own `_KVCache` for those positions was **never populated** — neither by reuse-matching (explicitly skipped, step in §2/§3) nor by its own forward computation (skipped per steps 5-6).

**Determination: Verified reachable at the level of "draft manager's committed/history state for the reused-prefix range is never populated, while the chunk bookkeeping shared with (Variant B) or derived from (Variant A) the target instructs the draft forward pass to skip computing that same range."** This is a directly-cited, mechanical fact from this session's reading.

Whether this actually manifests as reading garbage/uninitialized GPU memory during the draft's attention over `[0, 800)` — as opposed to some other native-level guard (e.g. `historyLength` defaulting to `0` and the attention-metadata/kernel path refusing to attend past `history_length`, causing a correctness-safe but silently-degraded draft, or an assertion/error) — was **not resolved this session**. The two candidate outcomes are:
   - **Silent corruption**: draft attention reads uninitialized pages for `[0, 800)` → garbage-influenced draft logits/tokens (a correctness bug, though possibly bounded by verification/rejection in the target's speculative-decoding acceptance step, which would degrade acceptance rate rather than corrupt final output).
   - **Guarded/no-op**: some native mechanism (attention metadata built from `kv_cache.history_length`, not from `req.context_current_position`) actually gates the draft's attention span to what the draft cache's own `history_length` says (which would be `0` after `_prepare_draft_resources`, since no `historyLength` arg is passed to `resize`), in which case the draft model would effectively see only the current chunk's tokens as context (still wrong relative to full-prefix attention, but not reading garbage memory).

Distinguishing these two requires reading the native `resize()` / `historyLength` semantics in `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp` (not `.h`) and the attention-metadata construction path (e.g. `prepare_attn_metadata_for_draft_replay`, `tensorrt_llm/_torch/speculative/interface.py:129+`) that consumes `history_length` vs `context_current_position` when building the KV page/position span fed to the attention kernel for the draft engine — none of which was read this session. **This final step is an Open question**, not a "verified reachable" data-corruption claim.

No code guard was found (searched `enable_block_reuse` combined with `spec`/`draft` terms across `_util.py`, `llm_args.py`, and `speculative/*.py`) that disables block reuse when a separate two-model or one-model draft KV cache manager is configured — so the trigger conditions (block reuse enabled + separate draft manager topology) are not mutually exclusive by construction.

---

## Open Questions

1. **Native `resize()`/`historyLength` semantics** (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp`, not read this session): does omitting `historyLength` in `_prepare_draft_resources`'s `kv_cache.resize(capacity)` call (`kv_cache_manager_v2.py:2817`) leave `history_length` at `0`, or does it silently track `capacity`/some other default? This determines whether the draft attention path would read stale/garbage memory or be safely gated to the freshly-computed chunk only.
2. **How the draft engine's attention metadata is built** relative to `history_length` vs `context_current_position` — specifically whether `prepare_attn_metadata_for_draft_replay` (`tensorrt_llm/_torch/speculative/interface.py:129+`, not read this session) or the draft's attention-backend metadata prep uses the draft `_KVCache`'s own committed/history state (safe) or blindly trusts `req.context_current_position`/`py_last_context_chunk` copied from the target (potentially unsafe). This is the crux of resolving Task 2 §4's "Open question."
3. **Disaggregated-worker role split**: does a context-only disagg worker instantiate a draft `KVCacheManagerV2` at all, and how does `draft_kv_cache_manager` participate (if at all) during `TRANS_IN_PROGRESS` handoff to a generation-only worker? Not traced — would need the disagg worker startup/role-assignment code outside `_util.py`/`kv_cache_manager_v2.py`.
4. **Whether the draft-model architectures in practice need full-prefix self-attention** (i.e., whether EAGLE3/MTP/DSpark draft layers actually attend over the full `[0, context_current_position)` range, or only over a short local/sliding window, or consume target hidden states instead of needing their own KV history for that range) — this affects the practical *severity* of the Task 2 finding even if the mechanism is confirmed reachable. Not traced this session (would require reading the draft model architectures under `tensorrt_llm/_torch/models/` or `_torch/speculative/eagle3.py`/`mtp.py`).
5. **Whether block reuse + separate draft-manager topology is a tested/supported combination** in CI (`tests/unittest/...`) — not checked this session; if it is untested, that would raise the practical likelihood that the Task 2 mechanism is a live gap rather than something masked by an untraced safety net.
