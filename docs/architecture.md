# Architecture

> Synthesized from AGENTS.md, `.github/tava_architecture_diagram.md`, `docs/source/developer-guide/overview.md`, and `docs/source/torch/arch_overview.md`. August 2026.

## High-Level Structure

TensorRT-LLM has three independent inference backends that share a common Python `LLM` API and a common C++ scheduling/decoding core:

```
HuggingFace Checkpoint
        │
        ▼
   tensorrt_llm.LLM          ← user entry point (llmapi/llm.py)
        │
        ├── PyTorch Backend   (default)
        ├── AutoDeploy Backend (beta, torch.export + graph transforms)
        └── VisualGen          (DiT image/video — separate API surface)

Shared C++ core (via Nanobind):
  Scheduler → BatchManager (in-flight batching) → KVCacheManager
  Decoder → Sampling
```

## PyTorch Backend (Default)

**Entry path:** `LLM` → `TorchLlmArgs` → `_TorchLLM` → `PyExecutor`

`PyExecutor` (`_torch/pyexecutor/py_executor.py:504`) runs a continuous background loop:

1. **Fetch** new requests from `executor_request_queue.py`
2. **Schedule** — `CapacityScheduler` (C++ via Nanobind) checks resource availability; `MicroBatchScheduler` selects the batch
3. **Prepare resources** — `ResourceManager` allocates KV cache slots (via `KVCacheManagerV2`) and PEFT adapters
4. **Forward** — `PyTorchModelEngine.forward()` runs the model on GPU
5. **Decode** — C++ `Decoder` produces output tokens (greedy / sampling)
6. **Process outputs** — finalize finished requests, return results

**Overlap Scheduler** (default on): step N+1 GPU work launches before CPU finishes step N results, hiding host-side latency.

**CUDA Graphs**: batches are padded to cached graph sizes to maximize hit rate (~22% throughput gain on some configs). Controlled by `DecodeCudaGraphConfig` / `EncodeCudaGraphConfig` in `llm_args.py`.

### Model Layer

All PyTorch models live in `_torch/models/modeling_<name>.py` (~70 models). Each file provides:
- A `Config` class inheriting `PretrainedConfig`
- A `ForCausalLM` class inheriting `PretrainedModel`
- Auto-registered in `models/automodel.py:AutoModelForCausalLM` via the HF `architectures` field

### Attention Subsystem

`_torch/modules/attention.py` dispatches to pluggable backends:
- `TRTLLM` — default custom kernel
- `FlashInfer` — `_torch/attention_backend/flashinfer.py`
- `FlashAttention` — `_torch/attention_backend/fmha/`
- `Triton` prefill — `_torch/attention_backend/triton_prefill.py`
- MLA (Multi-Latent Attention) — `_torch/modules/mla.py`

Backend selected via `TorchLlmArgs.attn_backend`.

### MoE Subsystem

`_torch/modules/fused_moe/` provides multiple dispatch backends:
- Triton (`fused_moe_triton.py`), CuTe DSL (`fused_moe_cute_dsl.py`), DeepGEMM (`fused_moe_deepgemm.py`), CUTLASS (`fused_moe_cutlass.py`), TRT-LLM gen (`fused_moe_trtllm_gen.py`), Dense GEMM fallback
- Expert parallelism via DeepEP or NCCL
- `MoeLoadBalancerConfig` in `llm_args.py` controls expert load balancing

### Speculative Decoding

Controlled by `speculative_config` on `LLM`. Available algorithms (all in `_torch/speculative/`):

| Algorithm | Config class | Notes |
|-----------|-------------|-------|
| Draft/Target | `DraftTargetDecodingConfig` | Separate draft model |
| EAGLE 3 | `Eagle3DecodingConfig` | With optional dynamic tree |
| MTP | `MTPDecodingConfig` | Multi-Token Prediction |
| NGram | `NGramDecodingConfig` | Lookup-based |
| PARD | `PARDDecodingConfig` | Parallel autoregressive draft |
| DFlash | `DFlashDecodingConfig` | Flash-based draft |
| DSpark | `DSparkDecodingConfig` | |
| Medusa | `MedusaDecodingConfig` | |
| Lookahead | `LookaheadDecodingConfig` | |
| SA (Suffix Automaton) | `SADecodingConfig` | Enhances EAGLE 3 on repetitive content |

Draft tokens are verified in a single target-model forward pass. Acceptance-length metrics are logged per step.

## AutoDeploy Backend (Beta)

**Entry path:** `LLM` → `_torch/auto_deploy/llm_args.py` → `ADExecutor` → `ADEngine`

AutoDeploy captures the model via `torch.export`, then applies a pipeline of graph transforms before execution. It reuses the same C++ scheduler and decoder as PyTorch backend.

**Transform pipeline** (`_torch/auto_deploy/transform/library/`, applied in order via `optimizer.py`):
- Export & cleanup: `export_to_gm.py`, `cleanup_*.py`, `eliminate_redundant_transposes.py`
- Fusion: `fuse_silu_mul.py`, `fuse_swiglu.py`, `fuse_rmsnorm_quant_*.py`, `fuse_rope_into_trtllm_attention.py`, `fused_moe.py`, etc.
- Quantization: `quantization.py`, `fuse_quant.py`, `quantize_moe.py`
- Sharding: `sharding.py`, `sharding_ir.py`
- KV cache injection: `kvcache.py`, `kvcache_transformers.py`
- Optional MLIR elementwise fusion

Config: `_torch/auto_deploy/config/default.yaml` (which transforms are enabled, sharding, quant options).

## Serving (`trtllm-serve`)

`tensorrt_llm/serve/` provides an OpenAI-compatible HTTP server:
- `openai_server.py` / `openai_service.py` — main server + request routing
- `openai_protocol.py` — request/response Pydantic models
- `disagg_coordinator.py` / `openai_disagg_server.py` — disaggregated serving coordinator
- `router.py` — load balancing across multiple server instances

**Disaggregated serving** separates prefill (context) and decode (generation) GPUs. KV cache is transferred via NIXL (default), UCX, or MPI. See `_torch/disaggregation/` and `docs/source/features/disagg-serving.md`.

## KV Cache

Paged block pool. Key properties:
- Blocks hold KV state for a fixed token count (must be a power of 2 > 1)
- Radix search tree for prefix reuse across requests
- Prioritized LRU eviction; blocks can offload to secondary (CPU) memory
- Per-request `KvCacheRetentionConfig` assigns eviction priority by token range
- Multi-pool for models with varying attention window sizes or GQA

Controlled by `KvCacheConfig` in `llm_args.py`. Implementation: `_torch/pyexecutor/kv_cache_manager_v2.py`.
For the `KVCacheV2Scheduler` ↔ `KVCacheManagerV2` boundary, state model, and audited allocation/eviction/suspension failure modes, see `docs/kv-cache-scheduler-manager.md`.

## Configuration Hierarchy

```
BaseLlmArgs (llm_args.py:4341)
└── TorchLlmArgs (llm_args.py:5048)
    ├── KvCacheConfig
    ├── DecodeCudaGraphConfig / EncodeCudaGraphConfig
    ├── MoeConfig / MoeLoadBalancerConfig
    ├── *DecodingConfig  (speculative decoding)
    ├── MultimodalConfig
    ├── GuidedDecodingConfig
    ├── CacheTransceiverConfig  (disaggregated KV transfer)
    ├── ExtendedRuntimePerfKnobConfig
    └── ... (many more)
```

All config classes are Pydantic `StrictBaseModel`. Changes to `BaseLlmArgs` or nested configs require running `scripts/generate_llm_args_golden_manifest.py` and committing `tensorrt_llm/usage/llm_args_golden_manifest.json`.

## Parallelism

`Mapping` class (`mapping.py:453`) encodes the parallelism topology:
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Expert Parallelism (EP) — for MoE
- Context Parallelism (CP)

Multiple execution backends: MPI (default), Ray, RPC. Controlled by `_ParallelConfig` in `llm_args.py`.

## VisualGen

Separate from LLM inference. Entry: `from tensorrt_llm import VisualGen`.

```
VisualGen API (visual_gen/) → DiffusionRemoteClient → DiffusionExecutor
    → PipelineLoader → BasePipeline (WAN / DiT) → MediaOutput
```

Shares attention, quantization, and parallelism kernels with PyTorch backend but has its own engine, args, and outputs. See `_torch/visual_gen/ENGINEERING_CRITERIA.md` before modifying.

## C++ Core (via Nanobind)

Both PyTorch and AutoDeploy backends bind to the same C++ components:

| Component | Header location | Role |
|-----------|----------------|------|
| `Scheduler` | `cpp/include/tensorrt_llm/batch_manager/` | Request scheduling |
| `BatchManager` | same | In-flight batching |
| `KVCacheManager` | same | KV block allocation |
| `Executor` | `cpp/include/tensorrt_llm/executor/` | C++ executor (legacy/TRT path) |
| `Decoder` | `cpp/include/tensorrt_llm/layers/` | Token generation |
| Custom kernels | `cpp/include/tensorrt_llm/kernels/` | Attention, quantization, etc. |

## Telemetry

Anonymous usage data sent to `NvTelemetry` endpoint via a background reporter thread. Fields controlled by `TelemetryConfig` / `TelemetryField`. New telemetry fields require CODEOWNER approval (privacy review). See `docs/source/developer-guide/telemetry.md`.

## External Dependencies

| Dependency | Role |
|-----------|------|
| PyTorch | Model execution, tensor ops |
| HuggingFace Transformers | Model loading, tokenization |
| NIXL | KV cache transfer in disaggregated serving (default) |
| FlashInfer | Optional attention kernel |
| DeepGEMM / DeepEP | MoE matrix kernels / expert parallelism |
| ModelOpt (NVIDIA) | Quantization (PTQ, QAT) |
| Ray | Multi-node distributed execution (optional) |
| Triton Inference Server | Optional serving backend (`triton_backend/`) |

## `cpp/` C++ Build Graph

`cpp/CMakeLists.txt` (project `tensorrt_llm`, C/CXX/CUDA, C++17) is the top-level entry. Key options: `BUILD_PYT` (PyTorch mode, default ON), `BUILD_TESTS` (gtest, ON), `BUILD_DEEP_EP` / `BUILD_DEEP_GEMM` / `BUILD_FLASH_MLA` (ON), `ENABLE_MULTI_DEVICE` (NCCL+MPI, ON), `ENABLE_UCX` (ON), `TRTLLM_ABI_NAMESPACE`. It `FetchContent`s nanobind, cutlass, cxxopts, json, xgrammar, ucxx (built via a separate `ucxx-src/build.sh` script, not plain `add_subdirectory`, to avoid a cudart-linking issue), and deepgemm, then descends into `add_subdirectory(tensorrt_llm)` and `tests`.

`cpp/tensorrt_llm/CMakeLists.txt` is the target graph hub. It builds per-area static "src" libraries (`common_src`, `kernels_src`, `layers_src`, `runtime_src`, plus many kernel-family libs gated by `USING_OSS_CUTLASS_*` options) and links them all `PUBLIC` into one shared library target `tensorrt_llm`. `batch_manager` (`tensorrt_llm_batch_manager_static`) and `executor` (`tensorrt_llm_executor_static`) are linked into `tensorrt_llm` with `WHOLE_ARCHIVE`, but each also declares an `INTERFACE` dependency back on `tensorrt_llm` — an intentional cyclic dependency (documented in a CMake comment). Internal cutlass kernels come either from a prebuilt tarball (`STATIC IMPORTED`) or are built from source if `INTERNAL_CUTLASS_KERNELS_PATH` is set.

**Python bindings**: `cpp/tensorrt_llm/nanobind/CMakeLists.txt` builds the `bindings` extension module (there is no `pybind/` — Nanobind is the only binding layer for the core API) from sources under `batch_manager/`, `executor/`, `runtime/`, `process_group/`, `thop/`, etc. It links `PUBLIC` against the already-built `tensorrt_llm` shared lib, `TORCH_LIBRARIES`, `torch_python`, and NVSHMEM libs if enabled — a thin wrapper compiled after and linked against the core.

Build is driven by `scripts/build_wheel.py` (~line 527–765), which shells out to `cmake -S <src> -D...` then `cmake --build . --parallel <n>`. Tests live under `cpp/tests/unit_tests/{batch_manager,common,executor,kernels,layers,multi_gpu,runtime,thop}`, using a `FetchContent`-pulled googletest and an `add_gtest()` helper.

*Needs verification: exact FetchContent version pins in `cpp/3rdparty/CMakeLists.txt`, and full contents of `thop/CMakeLists.txt` (TorchScript custom-op registration).*

## Scaffolding (`tensorrt_llm/scaffolding/`)

Test-time-compute / inference-scaling framework (chain-of-thought, best-of-N, majority vote, MCTS, tool-using agents) — see `scaffolding/README.md`. It separates *method* (`Controller`) from *execution backend* (`Worker`), glued by `ScaffoldingLlm` (`scaffolding_llm.py:26`).

- **`Controller`** (`controller.py:17`) is an ABC whose `process(tasks, **kwargs)` is implemented as a **Python generator** — a cooperative state machine. It `yield`s either a `List[Task]` (dispatched to workers and awaited) or a `ParallelProcess` (concurrent sub-generators), then resumes with results filled into the same `Task` objects by reference. Example: `BestOfNController.process` (`controller.py:377`) yields a `ParallelProcess` of N generation sub-controllers, scores them via a reward controller, and picks the max.
- **`Task`** (`task.py:34`) is a dataclass carrying a `worker_tag`; subclasses include `GenerationTask`, `ChatTask`, `RewardTask`, `MCPCallTask`.
- **`Worker`** (`worker.py:51`) is an ABC with `async run_task(task)`, dispatch-by-type via a per-subclass `task_handlers` map. `TRTLLMWorker` wraps a real `tensorrt_llm.LLM` (same `LLM`/`KvCacheConfig`/`SamplingParams` as the normal API, not a separate lightweight path); `OpenaiWorker`/`MCPWorker` hit external HTTP/MCP endpoints instead — the LLM call is pluggable per worker.

**Request flow**: `ScaffoldingLlm.generate_async(prompt)` clones `prototype_controller` and enqueues a request; a background asyncio event loop (own thread if the caller isn't already async) schedules up to `max_parallel_requests` concurrently, driving each controller's generator. Yielded task lists are dispatched via `asyncio.gather`; yielded `ParallelProcess`es recurse into concurrent sub-generators, each under its own `ExecutionScope`. When the top-level generator returns, the result is attached to the request's `ScaffoldingResult`.

*Needs verification: `execution_scope.py` internals (child scopes), and the `trace_replay/` subsystem (`ReplayEngine`) — appears to replay recorded executions but wasn't traced in depth.*

## NIXL and Disaggregated Serving: Request Lifecycle

Two transceiver implementations exist, selected via `CacheTransceiverConfig.transceiver_runtime`:
- **Python (v2)**: `_torch/disaggregation/transceiver.py:KvCacheTransceiverV2` (subclass of `KvCacheTransceiver` in `_torch/pyexecutor/kv_cache_transceiver.py`), driving per-request `TxSession`/`RxSession` objects through a `TransferWorker` (`_torch/disaggregation/native/transfer.py`).
- **C++ (`CPP` runtime)**: `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` + `dataTransceiver.cpp` + `cacheFormatter.cpp`/`mlaCacheFormatter.cpp`, with the actual NIXL SDK calls in `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` and connection handling in `cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp`.

The NIXL agent wrapper (`_torch/disaggregation/nixl/agent.py`) picks `BindingsNixlTransferAgent` (C++ `nixl_bindings` module) or `NixlTransferAgent` (PyPI `nixl` package) — both implement `BaseTransferAgent` (`register_memory`, `load_remote_agent`, `submit_transfer_requests`, `notify_sync_message`, `shutdown`). The two runtimes are independent implementations of the same protocol, not two views of shared code — the lifecycle below is traced separately for each.

### Setup

**Python**: memory registration and NIXL agent creation happen once per worker at `TransferWorker._setup_transfer_engine()` (`native/transfer.py:2308-2340`), inside `KvCacheTransceiverV2.__init__` (`transceiver.py:77-131`). Rank 0 broadcasts an instance UUID (`_broadcast_instance_name`) and its `RankInfoServer` endpoint (`_broadcast_context_endpoint`); all ranks then allgather `TransferWorker.sender_endpoint` plus per-PP-stage layer counts (`_exchange_rank_info`). Per-peer registration is lazy but happens on the **receiver's first request** to a sender endpoint, not on "first transfer" generically: `Receiver._get_sender_info` (`native/transfer.py:1668-1689`) ZMQ-dials the sender for its `RankInfo`, then sends `REGISTER_RANK_INFO` back — and only the **sender side**, on receiving that message, calls `load_remote_agent(...)` (`_register_peer_rank`, `transfer.py:1007-1027`). The receiver itself never calls `load_remote_agent`. This asymmetric first-contact handshake is the documented cause of bandwidth fluctuation on the first few requests.

**C++**: memory registration is **eager** — `AgentConnectionManager`'s constructor registers all KV-cache send/recv buffers with the NIXL agent up front (`connection.cpp:395-418`). Peer connection is lazy, on first `connect()` to a remote agent name (`connection.cpp:697-750`): local `AgentState{agentName, connectionInfo}` is MPI-allgathered at startup (`connection.cpp:420-434`), then `loadRemoteAgent(...)` is called when a request's `CommState` first arrives (`connection.cpp:719,730` → `transferAgent.cpp:568-591`/`704-738`). The IP:port variant busy-polls on `NIXL_ERR_NOT_FOUND` while waiting for the peer's metadata to appear (`transferAgent.cpp:725-736`) — this is connection-setup polling, not failure retry.

### State machine

**Python**: `SessionStatus` (`base/transfer.py:70-90`): `INIT → READY → TRANSFERRING → KV_TRANSFERRED → FULLY_TRANSFERRED`, with `ERROR`/`CANCELLED` terminal branches; per-task `TaskStatus` (`native/transfer.py:167-171`): `INIT → TRANSFERRING → TRANSFERRED/ERROR`. `TxSession.status`/`RxSession.status` are *computed properties* (`native/transfer.py:1237-1245`, `1853-1863`) derived from the underlying KV/aux task statuses each time they're read, not a stored field — a `_terminal_status` override latches `ERROR` (once any task hits `TaskStatus.ERROR`) or `CANCELLED` (only for tasks still `INIT`; tasks already `TRANSFERRING` are left to resolve async).

**C++**: low-level `TransferState{kIN_PROGRESS, kSUCCESS, kFAILURE}` (`cpp/include/tensorrt_llm/executor/transferAgent.h:353-357`) from `NixlTransferStatus::queryStatus()/wait()`. Request-level `LlmRequestState` transitions `kDISAGG_CONTEXT_TRANS_IN_PROGRESS → kDISAGG_GENERATION_TRANS_IN_PROGRESS → kDISAGG_GENERATION_TRANS_COMPLETE/kDISAGG_CONTEXT_COMPLETE`, or `kDISAGG_TRANS_ERROR` on failure (`cacheTransceiver.cpp:868,895,918,924,938,1354,1371,1612,1629`). A cross-rank **consensus** layer (`TransferConsensusState`) additionally reduces per-rank pass/fail/timeout votes before a request's final state is committed — a distributed-agreement step with no equivalent in the Python path.

### Timeout / error propagation

**Python**: `submit_transfer_requests` raises `RuntimeError("NIXL transfer failed: op=..., remote=...")` on an underlying call error; `NixlTransferStatus.wait()` returns `False` and logs on `ERROR` state or timeout. These failure strings are classified `"severe"` (not `"immediate_fatal"`) in `_torch/pyexecutor/error_classification.py`, but — see below — that classification only matters for the async path; the sync path never reaches it. Two structurally different call paths exist, selected by `_uses_async_disagg_gen_transfer` (`py_executor.py:3428-3431`, async unless `_is_disagg_gen_only_no_context_benchmark()` or the debug env var `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` — **not** tied to the overlap-scheduler on/off choice, which only selects `_executor_loop` vs `_executor_loop_overlap`; both can reach either transfer path):

- **Async path** (the default): `_check_cache_transfer_errors` (`py_executor.py:6260-6273`) polls for requests already in `DISAGG_TRANS_ERROR` and calls `PyExecutor._handle_errors(..., charge_budget=False)` (`py_executor.py:6580-6634`) — the error budget is *not* charged, and only the affected requests are failed back to the client. This is the graceful, per-request failure path.
- **Sync path** (only exercised when `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1`, or briefly in the gen-only-no-context benchmark before it short-circuits, `py_executor.py:6161-6164`): `request_and_receive_sync` (`transceiver.py:611-644`) sets `req.state = DISAGG_TRANS_ERROR` then **re-raises**, uncaught through the entire call chain (`_recv_disagg_gen_cache` → `_prepare_disagg_gen_init` → `_prepare_and_schedule_batch`/loop variants, none of which have a `try`/`except` around this call), up to `_event_loop_wrapper` (`py_executor.py:1192-1214`). **It never reaches `_handle_errors` or the error-budget machinery — those apply only to the async path.** See "Sync-path exception handling" below for the full trace and its consequences.

Timeouts are config-driven: `kv_transfer_timeout_ms` (60000ms default), `kv_transfer_sender_future_timeout_ms` (1000ms), `kv_transfer_poll_interval_ms` (5000ms).

**C++**: `AgentConnection::send()` throws (`TLLM_CHECK_WITH_INFO`, `connection.cpp:219`) on any non-`kSUCCESS` transfer state; the exception propagates through `dataTransceiver.cpp`'s async send/receive workers (`std::async`/`std::promise`, `dataTransceiver.cpp:717-822`), then `CacheTransceiver::checkContextTransferStatus`/`checkGenTransferStatus` catch it and set `kDISAGG_TRANS_ERROR` (`cacheTransceiver.cpp:914-926,1254-1260,1544-1554`). Elapsed time is separately checked against `kv_transfer_timeout_ms`; expiry triggers cancellation, not retry (`cacheTransceiver.cpp:1172-1327,1495-1528`).

### Sync-path exception handling — **resolved**: crashes the event loop, does not go through the error budget

Tracing the re-raised sync-path exception to its actual catch site (previously an open gap) shows it behaves fundamentally differently from the async path, not just "the same failure without budget accounting":

- `_event_loop_wrapper` (`py_executor.py:1192-1214`) is the only frame that catches it. Its `except` block deliberately does **not** call `_handle_errors`/`_enqueue_responses` — a code comment explains why: those helpers trigger `tp_gather`/allgather collectives that would deadlock if only this rank crashed while peer ranks are still waiting on the collective. Instead it logs the error, stores it in `self._event_loop_error` (line 1211), and re-raises; the `finally` block runs `_executor_loop_cleanup()` (line 1214 → `py_executor.py:2412-2435`), which sets `self.is_shutdown = True` and notifies `response_cv`. **The `ErrorBudget`/`charge_budget` machinery in `_handle_errors` is never invoked on this path.**
- No per-request error response is constructed at the point of failure. `active_requests` is not filtered or cleaned up directly; instead, every caller blocked in `_await_single_response` (`py_executor.py:7063-7094`) wakes on `is_shutdown`, finds no response queued for its request id, reads `self._event_loop_error`, and raises a generic `RuntimeError(f"Event loop terminated with error: {error}")` (line 7084-7085) — **not** a NIXL-specific message; every in-flight request on that rank fails identically, regardless of whether it was the one whose transfer actually errored. `executor/base_worker.py:1127-1158` (`_broadcast_event_loop_error`) further propagates the error to other ranks/consumers.
- Net effect: an async-path NIXL failure fails one request gracefully; a sync-path NIXL failure crashes the entire executor event-loop thread for that rank, failing *all* in-flight requests with a generic error. This is a materially worse failure mode, and it is only reachable via the debug env var `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` in normal operation (the gen-only-no-context benchmark path bypasses the transfer call before it can fail this way).
- A `grep -rni restart tensorrt_llm/_torch/pyexecutor/*.py` finds exactly one hit — an operator-facing log string ("...process restart is required") inside `_handle_disagg_cache_errors_synced` (`py_executor.py:3912-3990`), used when the *async*-path poisoned-transfer-buffer consensus check declares a fatal error; it is not code that triggers a restart. What actually happens after this crash — and whether anything outside `_torch/pyexecutor/` detects or recovers from it — is traced in "Recovery after event-loop crash" below.

### Recovery after event-loop crash

Following the sync-path crash from the previous section outward through the process/serving stack — what detects it, and what (if anything) restarts:

- **Thread level** (in-repo, verified): `PyExecutor.start_worker()` runs `_event_loop_wrapper` on a `daemon=True` `threading.Thread` (`py_executor.py:1276-1278`). No custom `threading.excepthook` exists anywhere in the repo (searched, none found), so an uncaught exception from `event_loop()` follows plain CPython default behavior: the traceback prints to stderr, the thread dies, and **the process itself does not exit or crash on its own** — it just has one permanently-dead thread. Nothing else is interrupted unless something else explicitly polls `_event_loop_error`/`is_shutdown`.
- **Local detection, no restart** (in-repo, verified): `BaseWorker.AwaitResponseHelper.__call__` (`executor/base_worker.py:1117-1156`) polls `engine._event_loop_error` and calls `_broadcast_event_loop_error` to inject an `ErrorResponse`/wake pending `GenerationResult` queues — detect-and-report, not restart; no code anywhere re-spawns the dead event-loop thread. For MPI-launched multi-process workers, `executor/proxy.py` runs a genuine watchdog thread (`_error_monitor_thread`/`_error_monitor_loop`, `proxy.py:215-219,392`) that polls worker process/future liveness (`_check_mpi_workers`/`_check_mpi_futures`, `proxy.py:230-262`) and, on death, calls `_set_fatal_error` + `pre_shutdown()` (`proxy.py:245-247,255-260`) — an orderly shutdown of the surviving process, not a respawn. `check_health()` (`proxy.py:289-309`) exposes this state to callers. (Separately, `ray_executor.py:366` calls `ray.kill(no_restart=True)` — this confirms Ray-actor auto-restart is *deliberately disabled*, and only appears during intentional teardown, not crash recovery.)
- **`trtllm-serve` layer** (in-repo, verified): the `/health` handler (`serve/openai_server.py:1167-1190`) checks `_fatal_error`; if set and the server isn't already shutting down, it calls `signal.raise_signal(signal.SIGINT)` (line 1185) to trigger uvicorn's graceful shutdown, and returns HTTP 503 either way. This turns a crashed backend into a clean server-process exit rather than a server that silently hangs while returning errors — but it is **self-termination, not restart**; nothing in this repo brings the process back up afterward. `/health_generate` additionally does an end-to-end generation probe.
- **Disaggregated worker-pool orchestration** (in-repo, partially verified / partially external): `serve/router.py`'s `check_servers_health` (lines 634-669, driven by a polling loop around 528-563) actively health-checks context/generation worker URLs and calls `self._metadata_server.remove(key)` to drop a dead worker from the routing pool — real, repo-internal detection-and-eviction. It does **not** spawn a replacement worker. `docs/source/features/disagg-serving.md:227-229` points to SLURM launch scripts (`examples/disaggregated/slurm`) for cluster launch, and (line ~126) names Dynamo — an external NVIDIA project — for Kubernetes-based deployment, monitoring, and dynamic scaling. **Replacement/respawn of a dead worker is explicitly deferred to external orchestration (SLURM job scripts or Dynamo+Kubernetes); this repo's own code only detects and evicts.**
- **MPI-wide abort semantics**: no `MPI.COMM_WORLD.Abort()` call was found in `_torch/pyexecutor/` or `executor/`. Whether an uncaught exception that kills one rank's process also terminates the entire `mpirun` job (or hangs the others waiting on a collective) is standard MPI/`mpi4py` runtime behavior, not something this repo's code controls or overrides — **unverified from this codebase alone; would need to be checked against the specific MPI implementation/launcher in use.**
- **Operator-visible symptoms** (in-repo, verified where cited): `logger.error(f"Error in event loop: {e}")` plus traceback in worker logs (`py_executor.py:1202-1203`); in attached/single-process mode, the caller receives `RuntimeError: Event loop terminated with error: ...`; in proxy/MPI mode, an `EngineDeadError` propagates to pending requests and `/health` starts returning 503, followed by the SIGINT-triggered self-shutdown described above. **Not verified**: the process's actual exit code after that SIGINT-triggered uvicorn shutdown, and how cleanly the remaining MPI ranks tear down — both would require running the failure rather than static reading.

**Bottom line**: recovery is layered — thread death and process/rank-death detection are handled inside this repository (daemon-thread isolation, `proxy.py` watchdog, router-level eviction, `/health`-triggered self-termination) — but nothing in TensorRT-LLM itself **restarts** a crashed executor, worker process, or rank. Bringing a replacement worker back into service is explicitly the job of external infrastructure: SLURM job scripts (`examples/disaggregated/slurm`) or Dynamo/Kubernetes, per `docs/source/features/disagg-serving.md`. Check those external configurations directly for restart policy, backoff, and request availability during recovery — none of that is implemented here.

### Retry / reconnect — **resolved**

**Neither path retries a failed transfer or automatically reconnects a peer.** This was previously an open unknown ("C++-path retry/reconnect behavior not verified"); it is now confirmed closed on both sides:
- **C++**: a case-insensitive grep across `cpp/tensorrt_llm/batch_manager/` and `cpp/tensorrt_llm/executor/cache_transmission/` for retry/reconnect/backoff/re-register finds exactly two "retry" hits, both about re-sending a *cancellation control message* if a peer hasn't yet acknowledged it (`cacheTransceiver.cpp:1343,1601`) — unrelated to data-transfer retry. No reconnect/backoff/re-register logic exists anywhere in these trees.
- **Python**: the same grep across `_torch/disaggregation/` turns up only `invalidate_remote_agent`'s definition and its **one** production call site, inside `Sender.shutdown()` (`transfer.py:1160`) — i.e. remote-agent invalidation happens only at teardown, never in response to a failed transfer. `load_remote_agent` is called exactly once per peer, at first contact; it is never called again for that peer, including after failure.
- In both implementations, a failed transfer ends in the request being marked `DISAGG_TRANS_ERROR`/`kDISAGG_TRANS_ERROR` — never retried. The async path's escalation mechanism (error-budget exhaustion → `is_fatal` shutdown) and the sync path's crash behavior are covered above under "Timeout / error propagation" and "Sync-path exception handling".

### Teardown

**Python**: per-request `.close()` is called on every terminal branch (`transceiver.py:720-730,784,811`, and `request_and_receive_sync`'s `finally`) — this frees the session object but does not deregister NIXL memory or invalidate the remote agent (those are process-lifetime resources). Transceiver-level `shutdown()` (`transceiver.py:214-226`) is idempotent, closes every live session without draining in-flight transfers (abandons anything still `TRANSFERRING`), then calls `TransferWorker.shutdown()` (`native/transfer.py:2378-2420`), which itself is idempotent and order-sequenced: stop `RankInfoServer` → `Sender.shutdown()` (stops ZMQ listener, joins worker threads with a 5s timeout, then `invalidate_remote_agent()` per peer) → `Receiver.shutdown()` → close bounce buffer → deregister memory → `agent.shutdown()`. No SIGTERM/SIGINT handler was found in these files — an unclean process kill skips all of this and relies on OS-level fd/GPU cleanup.

**C++**: `AgentConnectionManager`'s destructor deregisters all locally registered memory (`connection.cpp:878-882`); `NixlTransferAgent::shutdown()`/destructor invalidate remote-agent metadata and destroy the underlying `nixlAgent`, wrapped in a try/catch that only logs warnings (`transferAgent.cpp:761-803`). On a per-request failure, connections/registrations are **not** proactively invalidated — they persist for reuse by other requests; only the failing request itself is torn down.

### Observability

**Python**: `get_status_dump()` (`transceiver.py:172-212`) counts TX/RX sessions per `SessionStatus` plus a `waiting_for_peer_info` count. `logger.error`/`.warning` cover malformed/unknown ZMQ messages, shutdown-path failures (all non-fatal warnings), and timeouts (`"TxSession ... timed out"`, `"TxSession ... failed"`, `"Disagg gen transfer FAILED rank=..."`). Per-transfer size/bandwidth is **only** logged when explicitly enabled via env vars (`TRTLLM_KVCACHE_TIME_OUTPUT_PATH`, `TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO` + `TLLM_KV_TRANSFER_PERF_LOG_FILE`) — the `PerfTimer`/`PerfLogManager` machinery in `native/perf_logger.py` tracks `transfer_size_bytes`, `avg_segment_size_bytes`, `throughput_mbs` per peer rank, written as CSV or `logger.info`; without those env vars, no per-transfer size/bandwidth appears in logs.

**C++**: `CacheTransceiver::getStatusDump()` (`cacheTransceiver.h:315`, impl `cacheTransceiver.cpp:~750-800`) — directly analogous to Python's, returning active/timed-out/canceling/completed/failed sender and requester counts plus `hasPoisonedTransferBuffer()`. Extensive `TLLM_LOG_DEBUG/WARNING/ERROR` include rank, agent name, and request ID; a gen-side CSV transfer summary is written unconditionally (`cacheTransceiver.cpp:353, writeGenTransferSummary`) — unlike Python, this isn't gated behind extra env vars.

**Documented but not code-verified** (`docs/source/features/disagg-serving.md`): NIXL backend selection via `TRTLLM_NIXL_KVCACHE_BACKEND`; `UCX_CUDA_IPC_ENABLE_MNNVL=n` recommended to reduce cross-NVLink-domain UCX timeout errors (doc states these "don't necessarily cause your trtllm-serve to fail").

**Config**: `CacheTransceiverConfig` (`llmapi/llm_args.py` ~line 4196) — `backend` (`DEFAULT`/`UCX`/`NIXL`/`MOONCAKE`/`MPI`), `transceiver_runtime` (`CPP`/`PYTHON`/`auto`), `kv_transfer_timeout_ms`, `kv_transfer_sender_future_timeout_ms`, `kv_transfer_poll_interval_ms`, `kv_cache_bounce_size_mb` (Python v2 only — coalesced multi-rail NIXL write buffer).

### Remaining NIXL-specific gaps

- Worker-disconnect / process-crash liveness detection (heartbeat) for a peer *mid-transfer* — no such mechanism was found in either transfer path itself; both rely on the transfer-level timeout (`kv_transfer_timeout_ms`) rather than an explicit peer-liveness check. (Process-level liveness detection *does* exist one layer up, in `executor/proxy.py` and `serve/router.py` — see "Recovery after event-loop crash" — but that's process supervision, not a transfer-level peer heartbeat.)
- **Resolved** (see "Recovery after event-loop crash"): nothing in this repo restarts a crashed rank/process — TensorRT-LLM only detects and evicts/self-terminates. Actual restart is external (SLURM job scripts or Dynamo/Kubernetes); check those configs directly for restart policy, backoff, and request availability during recovery.
- Whether an uncaught exception on one MPI rank terminates the whole `mpirun` job or hangs peer ranks on a collective — this is standard MPI/`mpi4py` runtime behavior, not controlled by this repo's code (no `MPI.COMM_WORLD.Abort()` call was found); would need verification against the specific MPI implementation/launcher in use, not from source alone.
- Exact process exit code after the `/health`-triggered SIGINT self-shutdown, and how cleanly remaining MPI ranks tear down afterward — would require running the failure, not static reading.
- Whether `disable_overlap_scheduler` or other configuration besides `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` can route production (non-benchmark) traffic onto the sync path — confirmed the two are independent switches (overlap scheduler picks the executor loop variant, the env var picks sync vs async transfer), but not exhaustively checked for other code paths that might set the env var or otherwise force sync mode.
- Full behavior of the C++ cross-rank `TransferConsensusState` vote mechanics (exact quorum/threshold rules) — its existence and role (reduce pass/fail/timeout votes before committing a request's final state) is confirmed, but the vote logic itself wasn't traced line-by-line.
- Whether `TRTLLM_NIXL_KVCACHE_BACKEND` (UCX/LIBFABRIC selection) and the `UCX_CUDA_IPC_ENABLE_MNNVL` workaround, both from `disagg-serving.md`, are reachable/observable in the code paths traced here — not cross-referenced against the source in this pass.

## `_torch/compilation/` — torch.compile Integration

`PyTorchModelEngine.__init__` (`_torch/pyexecutor/model_engine.py`, ~line 543–622) reads `TorchLlmArgs.torch_compile_config`, builds a `Backend` instance (`_torch/compilation/backend.py`), and applies `torch.compile(self.model.model, backend=self._torch_compile_backend, fullgraph=...)` for `DecoderModelForCausalLM` (fullgraph defaults to `True`; the codebase relies on Dynamo fullgraph mode for piecewise CUDA graph capture to work, so untraceable ops must be wrapped as custom ops with fake kernels).

`Backend.__call__` (`backend.py:193`) is the Dynamo entry point: runs `recover_pass` (un-fuses `flashinfer_fused_add_rmsnorm` so pattern-matching sees a consistent unfused form), then `aot_module_simplified(..., fw_compiler=self.optimize)`. `optimize()` (`backend.py:142`) runs custom fusion passes (`patterns/residual_add_norm.py`, `patterns/ar_residual_norm.py` — add+RMSNorm and allreduce+add+norm fusion) and `remove_copy_pass.py` (drops redundant `copy_` nodes from `auto_functionalize`), then dispatches to one of three tails: `piecewise_optimizer.py` (if `enable_piecewise_cuda_graph`), Inductor's `compile_fx_inner` (if `enable_inductor`), or the graph module returned as-is (pure Dynamo/AOTAutograd).

`piecewise_optimizer.py` splits the traced graph at attention/mamba custom-op boundaries (`attn_custom_op_inplace`, `mla_custom_op_inplace`, `mla_dsa_attn_inplace`, `gdn_custom_op_inplace`, …), runs those submodules eager, and captures/replays CUDA graphs for the rest via `PiecewiseInterpreter`/`PiecewiseRunner`.

**Config** (`TorchCompileConfig`, `llmapi/llm_args.py:5004`): `enable_fullgraph` (default `True`), `enable_inductor` (default `False`), `enable_piecewise_cuda_graph` (default `False`), `capture_num_tokens`, `enable_userbuffers` (default `True`), `max_num_streams` (default `1`).

**Known limitations** (`docs/source/features/torch_compile_and_piecewise_cuda_graph.md`): incompatible with two-model speculative decoding (`mtp_eagle_one_model: False`, `eagle3_one_model: False`); limited multimodal support; one open workaround (`DISABLE_LAMPORT_REDUCE_NORM_FUSION=1` env var, `backend.py` lines 92–95) for a multi-rank allreduce+norm fusion bug with the Lamport kernel when `world_size > 1`.

*Needs verification: `apply_llm_torch_compile` model coverage, and `_filter_piecewise_capture_num_tokens` interaction with `cuda_graph_config.batch_sizes`.*

## `_torch/memory/` — GPU Memory Service (not pinned-weight staging)

Contrary to the name, this directory is **not** a pinned-host-memory staging pool or a general virtual-memory abstraction. It's a thin adapter (`gpu_memory_backend.py`, 610 lines; `__init__.py` re-exports) around an *external* library, GPU Memory Service (GMS, from `ai-dynamo/dynamo`), used for **cross-process GPU weight sharing** on the same node. The actual CUDA VMM calls (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`) and FD-passing live in the external `gpu_memory_service` package, imported lazily — not in this repo.

`GPUMemoryBackend` (a `Protocol`) defines the interface consumed by the model loader; `GMSBackend` is the concrete implementation with two modes:
- **RW (writer)**: the first worker loads weights normally, with allocations routed into a GMS-managed CUDA pool via `mem_pool_scope()`; `finalize_write()` commits the tensors and the pool flips the writer to RO in place.
- **RO (reader)**: other workers zero-copy import the committed weights via `materialize_module()`, rebinding meta-tensor params to GMS-backed CUDA pointers — no disk I/O, no data copy.

Invoked from `_torch/pyexecutor/model_loader.py` (~line 739–968) when `LoadFormat.GMS` is selected. **Config**: `GmsConfig` (`llmapi/llm_args.py` ~line 4928: `socket_path`, `mode` — `auto`/`rw`/`ro`, `tag`), exposed as `TorchLlmArgs.gms_config`. A validator rejects `LoadFormat.GMS` combined with `moe_config.load_balancer` (MoE-balancer allocations land outside the GMS pool after it closes).

No coupling to KV cache management (`kv_cache_manager_v2.py`) or CUDA graph memory pools was found — separate concerns.

*Needs verification: exact call order of `move_untracked_params()` relative to `post_load_weights` in `model_loader.py`; whether any backend besides `GMSBackend` implements `GPUMemoryBackend` in practice.*
