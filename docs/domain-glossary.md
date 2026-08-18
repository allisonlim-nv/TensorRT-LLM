# Domain Glossary

> Key terms used across TensorRT-LLM code, docs, and CI. August 2026.

## Inference Fundamentals

**Prefill (context phase):** The initial forward pass that processes all input (prompt) tokens in parallel. Compute-bound. Produces the KV cache for the prompt.

**Decode (generation phase):** Autoregressive token generation, one token per step. Memory-bandwidth-bound. Each step is a forward pass over a single new token per sequence.

**In-flight batching (continuous batching):** Requests join and leave a running batch dynamically rather than waiting for a fixed batch to complete. Managed by `BatchManager` (C++).

**ISL / OSL:** Input Sequence Length / Output Sequence Length. Used in benchmark configs and perf comparisons.

**Throughput:** Tokens generated per second across all concurrent requests.

**Latency:** Time-to-first-token (TTFT) and inter-token latency (ITL). Disaggregated serving optimizes these separately.

## KV Cache

**KV cache:** Stores pre-computed attention key/value tensors so decode steps don't recompute the prompt. Allocated in paged blocks.

**Block:** Fixed-size unit of KV storage. Token count per block must be a power of 2 > 1. Multiple attention layers are packed into a single block.

**Radix tree:** Search structure used for prefix reuse — new requests that share a prompt prefix reuse already-computed KV blocks.

**Eviction:** Freeing blocks when memory pressure requires it. LRU-based with per-block priority; lower-priority blocks evicted first. Evicted blocks may offload to CPU secondary memory.

**KvCacheRetentionConfig:** Per-request policy assigning eviction priority to token ranges. Higher priority = kept longer.

**GQA / MQA / MHA:** Grouped Query Attention / Multi-Query Attention / Multi-Head Attention. Affect how many KV heads are stored; GQA and MQA reduce KV cache memory.

## Speculative Decoding

**Speculative decoding:** A drafter proposes candidate tokens; the target model verifies all candidates in one forward pass. Accepted tokens are kept. Reduces sequential steps at low batch sizes.

**Draft model:** A smaller, faster model (or heuristic) that proposes candidate tokens.

**Target model:** The main, full-size model that verifies candidates.

**max_draft_len:** Maximum number of candidate tokens per step.

**Acceptance rate / acceptance length:** Fraction of draft tokens accepted per step. Higher = more speedup.

**EAGLE 3:** A learned draft head attached to the target model's hidden states. Supports linear and dynamic-tree drafting. See `_torch/speculative/eagle3.py`.

**MTP (Multi-Token Prediction):** Draft head that predicts multiple tokens simultaneously.

**NGram:** Drafts by matching recent output tokens against the prompt (no separate model needed).

**Dynamic tree:** Tree-structured draft where the drafter expands multiple candidates at each layer. More compute per step but higher acceptance rates than a linear chain.

## Parallelism

**TP (Tensor Parallelism):** Splits individual weight matrices across GPUs. Applied within a single model layer.

**PP (Pipeline Parallelism):** Splits model layers across GPUs; each GPU holds a pipeline stage.

**EP (Expert Parallelism):** For MoE models; distributes experts across GPUs.

**CP (Context Parallelism):** Splits long sequences across GPUs during prefill.

**Mapping:** `tensorrt_llm.mapping.Mapping` — encodes TP/PP/EP/CP degrees and GPU rank assignment. Every model forward is parameterized by a `Mapping` instance.

**World size:** Total number of GPUs = TP × PP (× EP for MoE).

## Quantization

**PTQ / QAT:** Post-Training Quantization / Quantization-Aware Training.

**FP8 / INT8 / INT4 / NVFP4 / MXFP8:** Reduced-precision numeric formats. NVFP4 and MXFP8 are NVIDIA micro-scaling formats.

**QuantAlgo:** Enum in `tensorrt_llm/models/modeling_utils.py` listing all supported quantization algorithms.

**QuantConfig:** Pydantic config class binding a `QuantAlgo` to per-layer settings. Passed to `BaseLlmArgs`.

**ModelOpt:** NVIDIA's model optimization toolkit (external); used for PTQ calibration and QAT.

## MoE (Mixture of Experts)

**Expert:** A separate FFN (feed-forward network) sub-module. Only a subset is active per token.

**Router / gating:** Selects which experts process each token. Controlled by `MoeConfig`.

**Top-K routing:** Each token is sent to the K highest-scoring experts.

**DeepEP:** NVIDIA library for efficient all-to-all expert dispatch in EP configurations.

**Load balancer:** Redistributes expert assignment to avoid hot spots. `MoeLoadBalancerConfig` in `llm_args.py`.

## Attention Backends

**TRTLLM:** Default custom CUDA attention kernel (fused, optimized for TRT-LLM batch formats).

**FlashInfer:** Third-party high-performance attention kernel; optional via `attn_backend="FlashInfer"`.

**FlashAttention (FMHA):** Flash attention CUDA kernels; selected via `attn_backend="FlashAttention"`.

**MLA (Multi-Latent Attention):** DeepSeek's KV-cache-compressed attention variant. Implemented in `_torch/modules/mla.py`.

**Sparse attention:** Limiting each token's attention window. Multiple configs: `SeqLenAwareSparseAttentionConfig`, `DeepSeekSparseAttentionConfig`, `RocketSparseAttentionConfig`, etc.

## Backends & Execution Paths

**PyTorch backend:** Default path. Runs the model as standard PyTorch `nn.Module` with custom ops and CUDA kernels. Entry: `PyExecutor`.

**AutoDeploy (AD):** Beta backend. Uses `torch.export` to capture the model graph and applies graph transforms (fusion, sharding, quant). Lower model-porting effort. Entry: `ADExecutor`.

**TensorRT backend (legacy):** Compiles the model into a TensorRT engine. Mentioned in architecture diagrams; the primary new development path is now PyTorch.

**VisualGen:** Diffusion-Transformer (DiT) based image/video generation. Not an LLM backend. Separate `VisualGen` API.

## Scaffolding (Test-Time Compute)

**Scaffolding:** Framework (`tensorrt_llm/scaffolding/`) for test-time-compute methods — chain-of-thought, best-of-N, majority vote, MCTS, tool-using agents — separating the *method* from the *execution backend*.

**Controller:** A `scaffolding` abstraction (`controller.py:Controller`) implementing a strategy as a Python generator: it yields `Task`s or a `ParallelProcess` to be executed, then resumes with results filled in by reference.

**Worker:** A `scaffolding` abstraction (`worker.py:Worker`) that executes `Task`s — e.g. `TRTLLMWorker` wraps a real `tensorrt_llm.LLM`, `OpenaiWorker`/`MCPWorker` hit external HTTP/MCP endpoints.

**ParallelProcess:** A bundle of concurrent sub-generators (sub-controllers) run together under `ScaffoldingLlm`'s asyncio event loop, each in its own `ExecutionScope`.

## Serving

**trtllm-serve:** CLI that starts an OpenAI-compatible HTTP+gRPC server backed by any TensorRT-LLM backend.

**Disaggregated serving:** Separates prefill and decode onto different GPU pools. Allows independent scaling and latency optimization for each phase.

**NIXL:** Default KV cache transfer library for disaggregated serving (NVIDIA). Two integration paths exist: a Python transceiver (`_torch/disaggregation/`) and a C++ transceiver (`cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp`), selected via `CacheTransceiverConfig.transceiver_runtime`. Failed transfers mark the request `DISAGG_TRANS_ERROR` rather than retrying automatically.

**Transceiver:** The component that sends/receives KV cache blocks between context (prefill) and generation (decode) workers in disaggregated serving. `KvCacheTransceiverV2` (Python) or the C++ `CacheTransceiver`.

**GMS (GPU Memory Service):** External library (`ai-dynamo/dynamo`) for cross-process GPU weight sharing without a copy — one process loads and commits weights (RW), others zero-copy import them (RO) via CUDA VMM. Integrated in `tensorrt_llm/_torch/memory/gpu_memory_backend.py`, selected via `LoadFormat.GMS`. Unrelated to KV cache or pinned host memory.

**Overlap scheduler:** PyExecutor optimization that launches GPU work for step N+1 while the CPU is processing step N results. Enabled by default.

**CUDA graph:** Captures a sequence of GPU operations as a graph for single-call replay. Reduces CPU kernel-launch overhead. Batches are padded to cached graph sizes.

**Piecewise CUDA graph:** A `torch.compile`-integrated variant that splits the traced graph at attention/mamba custom-op boundaries, runs those unsupported submodules eager, and captures/replays CUDA graphs for the rest. Implemented in `_torch/compilation/piecewise_optimizer.py`; enabled via `TorchCompileConfig.enable_piecewise_cuda_graph`.

**Dynamo fullgraph mode:** `torch.compile` mode requiring the entire traced region to be representable without graph breaks. TRT-LLM's compile backend (`_torch/compilation/backend.py`) defaults to this (`enable_fullgraph=True`) so piecewise CUDA graph capture works; ops Dynamo can't trace must be wrapped as custom ops with a fake kernel for shape/dtype inference.

**Guided decoding:** Constrains output to a JSON schema or regex. `GuidedDecodingConfig` in `llm_args.py`.

## Config & API

**BaseLlmArgs:** Base Pydantic config for all backends (`llm_args.py:4341`). All user-facing LLM options live here or in nested configs.

**TorchLlmArgs:** Extends `BaseLlmArgs` with PyTorch-backend-specific options (`llm_args.py:5048`).

**SamplingParams:** Per-request generation parameters (temperature, top-k, top-p, stop criteria, etc.).

**StrictBaseModel:** Project-local Pydantic base class; forbids extra fields. All config classes must inherit from it, not `BaseModel` directly.

**PybindMirror:** Mixin for config classes that mirror C++ Nanobind-bound structs (e.g., `KvCacheConfig`, `LookaheadDecodingConfig`). Handles field name mapping between Python and C++.

**llm_args_golden_manifest.json:** Auto-generated JSON snapshot of all `BaseLlmArgs` fields and types. Used in CI to detect unintended API changes. Must be regenerated after any field change.

## Testing & CI

**L0 tests:** Tier-0 CI tests (fastest, most critical); listed in `tests/integration/test_lists/test-db/l0_*.yml`.

**Waives:** Known-failing tests excluded from CI with NVBug tracking links. See `tests/integration/test_lists/waives.txt`.

**API stability tests:** `tests/unittest/api_stability/` — protect the public Python API signature. Changes that break these tests require code-owner approval.

**DCO:** Developer Certificate of Origin. All commits must be signed with `git commit -s`.

**Bot commands:** CI is triggered by PR comments (`/bot run`, `/bot run --extra-stage "..."`, etc.).
