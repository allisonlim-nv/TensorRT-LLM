# Codebase Map

> Cross-references AGENTS.md and docs/architecture.md. Updated August 2026.

## Repo Root

| Path | Role |
|------|------|
| `tensorrt_llm/` | Primary Python package |
| `cpp/` | C++ runtime, batch manager, executor bindings |
| `tests/` | Unit (`unittest/`) and integration (`integration/`) tests |
| `examples/` | End-to-end usage scripts and reference configs |
| `scripts/` | Dev utilities (manifest generation, CI helpers) |
| `docs/source/` | Sphinx docs; `developer-guide/`, `features/`, `torch/` are most useful |
| `triton_backend/` | Triton Inference Server backend |
| `triton_kernels/` | Hand-written Triton GPU kernels |
| `docker/` | Dockerfiles for dev and release containers |

## `tensorrt_llm/` Top-Level Modules

| Module / Package | Role |
|-----------------|------|
| `llmapi/llm.py` | Public `LLM` class (`BaseLLM → _TorchLLM → LLM`); entry point for all inference |
| `llmapi/llm_args.py` | All Pydantic config classes; `BaseLlmArgs` (line 4341) → `TorchLlmArgs` (line 5048) |
| `llmapi/llm_utils.py` | Model loading helpers, architecture-specific default overrides |
| `sampling_params.py` | `SamplingParams`, `GuidedDecodingParams`, `LogprobParams` |
| `mapping.py` | `Mapping` class — TP/PP/EP/CP parallelism topology descriptor |
| `executor/executor.py` | `GenerationExecutor` ABC; bridges Python and C++ executors |
| `models/automodel.py` | `AutoModelForCausalLM` — HF-style auto-discovery by `architectures` field |
| `models/modeling_utils.py` | `PretrainedConfig`, `PretrainedModel` base classes; `QuantConfig`, `QuantAlgo` |
| `quantization/` | Quantization configs and ModelOpt integration |
| `serve/` | OpenAI-compatible REST/gRPC server; disaggregated coordinator |
| `scaffolding/` | Multi-step / agentic scaffolding (`ScaffoldingLlm`, task controllers). Entry: `scaffolding_llm.py:26`; controller base: `controller.py:17`; `README.md` explains the method-vs-worker split |
| `visual_gen/` | **Public** VisualGen API (`VisualGen`, `VisualGenArgs`, `VisualGenParams`) — treat as user-facing surface |
| `bindings/` | Nanobind Python bindings to C++ executor, scheduler, and batch manager |
| `deep_ep/`, `deep_gemm/`, `flash_mla/` | Pre-compiled binary extensions for DeepEP, DeepGEMM, FlashMLA |

## `tensorrt_llm/_torch/` — PyTorch Backend

| Sub-path | Role |
|----------|------|
| `pyexecutor/py_executor.py` | `PyExecutor` — main inference loop (scheduling → forward → decode) |
| `pyexecutor/model_engine.py` | `PyTorchModelEngine` — wraps model forward pass |
| `pyexecutor/scheduler/` | Python-side scheduling logic (wraps C++ `CapacityScheduler`) |
| `pyexecutor/resource_manager.py` | `ResourceManager` — allocates KV cache, PEFT, and other per-request resources |
| `pyexecutor/sampler/` | Token sampling (greedy, top-k/p, beam search) |
| `models/` | ~70 per-model `modeling_<name>.py` files; each provides a `ForCausalLM` class |
| `modules/attention.py` | Multi-head attention; read `ATTENTION_DEVELOPER_GUIDE.md` before modifying |
| `modules/mla.py` | Multi-Latent Attention (MLA, used by DeepSeek variants) |
| `modules/fused_moe/` | MoE dispatch/combine; multiple backends (Triton, CuTe, DeepGEMM, CUTLASS) |
| `attention_backend/` | Pluggable attention kernels: `trtllm.py`, `flashinfer.py`, `fmha/`, `triton_prefill.py` |
| `speculative/` | Speculative decoding algorithms: EAGLE3, MTP, NGram, PARD, DFlash, DSpark |
| `auto_deploy/` | AutoDeploy beta backend (see below) |
| `visual_gen/` | VisualGen internal implementation (DiT pipelines) |
| `disaggregation/` | Disaggregated serving: NIXL/UCX/MPI KV-cache transceiver. Python v2: `transceiver.py:KvCacheTransceiverV2`, `native/transfer.py:TransferWorker`, `nixl/agent.py`. C++: `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp`, `dataTransceiver.cpp`, `executor/cache_transmission/nixl_utils/transferAgent.cpp`. Verified lifecycle/failure/recovery behavior: `docs/architecture.md` § NIXL and Disaggregated Serving: Request Lifecycle |
| `compilation/` | `torch.compile` integration: `backend.py:Backend` (Dynamo/AOT backend), `piecewise_optimizer.py` (piecewise CUDA graph split), `patterns/` (Inductor fusion passes) |
| `distributed/` | Tensor/expert parallelism collectives |
| `peft/` | LoRA and adapter management |
| `memory/` | **Not** pinned-host staging — thin adapter (`gpu_memory_backend.py:GMSBackend`) around the external GPU Memory Service (GMS) library for cross-process GPU weight sharing; gated by `LoadFormat.GMS` |
| `configs/` | Per-model runtime config defaults |

## `tensorrt_llm/_torch/auto_deploy/` — AutoDeploy Backend

| Sub-path | Role |
|----------|------|
| `shim/ad_executor.py` | `ADExecutor` — adapts `PyExecutor` for AutoDeploy |
| `config/default.yaml` | Default AutoDeploy config (transforms, sharding, quantization) |
| `transform/library/` | ~50 graph transforms (fusion, sharding, quant, cleanup, MoE, etc.) |
| `transform/optimizer.py` | Applies transforms in order |
| `export/` | `torch.export` wrappers |
| `models/` | Model factory + patches (HF-compatible, custom, EAGLE, Nemotron) |
| `compile/` | Model compilation pipeline |
| `mlir/` | Optional MLIR elementwise fusion path |

## `cpp/` — C++ Core

| Sub-path | Role |
|----------|------|
| `include/tensorrt_llm/batch_manager/` | Batch manager public API |
| `include/tensorrt_llm/executor/` | `Executor` C++ API |
| `include/tensorrt_llm/runtime/` | Runtime types (buffers, tensors) |
| `include/tensorrt_llm/kernels/` | Custom CUDA kernels |
| `include/tensorrt_llm/layers/` | Layer implementations |
| `CMakeLists.txt` | Top-level build config: `BUILD_PYT`/`BUILD_TESTS`/`ENABLE_MULTI_DEVICE`/`ENABLE_UCX` options, `FetchContent` for nanobind/cutlass/ucxx |
| `tensorrt_llm/CMakeLists.txt` | Target graph hub — builds per-area `*_src` static libs, links into shared `tensorrt_llm` target; see `docs/architecture.md` § C++ Build Graph |
| `tensorrt_llm/nanobind/` | `bindings` extension module (the only Python binding layer — no `pybind/`); source for `tensorrt_llm.bindings` |
| `tests/unit_tests/` | gtest suites: `batch_manager`, `common`, `executor`, `kernels`, `layers`, `multi_gpu`, `runtime`, `thop` |

## `tests/`

| Path | Notes |
|------|-------|
| `tests/unittest/` | Fast; run without GPUs where possible |
| `tests/unittest/api_stability/` | Guards public API signatures — changes here need code-owner approval |
| `tests/integration/defs/` | Require GPU + `LLM_MODELS_ROOT` |
| `tests/integration/test_lists/test-db/` | Per-GPU YAML files (`l0_h100.yml`, `l0_b200.yml`, etc.) |

## Where to Start by Task

| Task | Start here |
|------|-----------|
| Change inference API | `tensorrt_llm/llmapi/llm.py:273` (`BaseLLM`) |
| Add/change a config option | `tensorrt_llm/llmapi/llm_args.py:4341` (`BaseLlmArgs`) → regenerate manifest |
| Add a new model (PyTorch backend) | `tensorrt_llm/_torch/models/` + `docs/source/torch/adding_new_model.md` |
| Add a new model (AutoDeploy) | `tensorrt_llm/_torch/auto_deploy/models/` + `config/default.yaml` |
| Modify attention kernel | Read `_torch/modules/ATTENTION_DEVELOPER_GUIDE.md` first |
| Modify MoE | Read `_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md` first |
| Add a speculative decoding algorithm | `tensorrt_llm/_torch/speculative/` + new `*DecodingConfig` in `llm_args.py` |
| Change serving behavior | `tensorrt_llm/serve/openai_server.py` / `openai_service.py` |
| Debug KV cache | `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` + `docs/source/features/kvcache.md` |
| Debug KV cache scheduler/manager boundary or partial-failure behavior | `docs/kv-cache-scheduler-manager.md` — `KVCacheV2Scheduler`/`BudgetTracker`/`KVCacheManagerV2` responsibility boundary, state model, and audited failure modes |
| Add an AutoDeploy graph transform | `tensorrt_llm/_torch/auto_deploy/transform/library/` |
| Change VisualGen | Read `tensorrt_llm/_torch/visual_gen/ENGINEERING_CRITERIA.md` first |
| Modify the C++ CMake build | Start at `cpp/tensorrt_llm/CMakeLists.txt`; see `docs/architecture.md` § C++ Build Graph |
| Add a scaffolding controller/task | `tensorrt_llm/scaffolding/controller.py` + `task.py`; read `scaffolding/README.md` first |
| Debug disaggregated KV transfer / NIXL failures | `_torch/disaggregation/transceiver.py` (Python v2) or `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (C++) — see `docs/architecture.md` § NIXL and Disaggregated Serving: Request Lifecycle for verified setup/timeout/retry/teardown behavior |
| Debug crash/restart behavior after an executor event-loop failure | `executor/proxy.py`, `serve/openai_server.py` (`/health`), `serve/router.py` (`check_servers_health`) — see `docs/architecture.md` § Recovery after event-loop crash. Detection is in-repo; actual process restart is external (SLURM or Dynamo/Kubernetes) |
| Debug torch.compile / piecewise CUDA graph | `_torch/compilation/backend.py`, `docs/source/features/torch_compile_and_piecewise_cuda_graph.md` |
| Debug GMS cross-process weight sharing | `_torch/memory/gpu_memory_backend.py`, `_torch/pyexecutor/model_loader.py` |
