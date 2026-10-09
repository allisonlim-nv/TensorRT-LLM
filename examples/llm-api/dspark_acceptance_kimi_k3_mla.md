<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Kimi K3 with the vLLM/Inferact MLA DSpark head

These presets run **TensorRT-LLM** with the standalone
`Inferact/Kimi-K3-DSpark` MLA head. Here, "vLLM DSpark" identifies the drafter's
lineage. The inference engine, speculative decoding implementation, serving
processes, and acceptance counters are TensorRT-LLM. The target is Kimi K3
NVFP4, whose layers combine KDA and MLA attention.

| Mode | Runner | Preset | GPUs |
|------|--------|--------|------|
| Aggregate | `measure_dspark_acceptance.py` | [Aggregate YAML](dspark_acceptance_kimi_k3_mla.yaml) | 8 |
| Disaggregated | `measure_dspark_acceptance_disagg.py` | [Disaggregated YAML](dspark_acceptance_kimi_k3_mla_disagg.yaml) | 16, split into two TP8 workers |

The presets use TP8, MoE TP8/EP1, seven draft tokens, batch size 1, an 8,192-token
sequence limit, and a 1,024-token chunked prefill budget. Both modes enable
decode CUDA graphs, overlap, and cache reuse. They use CUTLASS MoE for NVFP4,
FP8 MLA KV, FP32 KDA state, and the V2 hybrid cache manager. Disaggregation adds
the Python NIXL transceiver and enables the same head on both workers.
These are acceptance measurement configurations; they require GPU validation.

This branch transfers the hybrid target cache between workers, but does not
transfer the standalone drafter's prompt history. The generation-side MLA
drafter therefore starts without that history. Expect this to affect acceptance
relative to aggregate inference even with matching settings; enabling DSpark on
both workers does not provide drafter-history transfer. The runner measures
that difference rather than assuming AL parity.

## Prerequisites

Run inside a container built from this branch with its TensorRT-LLM serving
dependencies and MLA DSpark kernels. The disaggregated runner also needs NIXL
and `prometheus_client`. `PyYAML` is needed to read the presets, and `datasets`
is needed when using the default GSM8K corpus. The scripts do not build a
container or download model weights.

Mount the complete NVFP4 target and Inferact MLA drafter checkpoints at these
paths, or copy both YAMLs and change their `model` and `drafter` fields:

```text
/models/Kimi-K3-NVFP4/config.json
/models/Kimi-K3-DSpark-MLA/config.json
```

Each directory also needs its weights and associated model files. These are
local aliases; `Kimi-K3-DSpark-MLA` must point to the Inferact MLA checkpoint.
Use the same target and drafter revisions for both runs. The RadixArk/SGLang
GQA head is a different checkpoint.

Plan for GPUs with GB300-class memory; GPU count alone does not guarantee model
fit. The disaggregated launcher requires all 16 GPUs to be visible on **one
host**. It does not combine GPUs across Slurm nodes. A four-GPU-per-node
allocation needs a separate cluster serving launcher. The aggregate runner
can use its existing `--launcher` option for a multi-node allocation; see the
[aggregate launcher instructions](dspark_acceptance.md).

The MLPerf agentic reproduction setup prepares a cluster workspace and client
environment, and expects a separately built TRT-LLM image. Its serving YAMLs
have a different schema from these acceptance recipes. Its trajectory dataset
also cannot be passed directly as `--prompt-file`, which expects one
`{"prompt": "..."}` record per line.

## Run a matched pair

From the repository root, preview the configurations without loading models:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --config examples/llm-api/dspark_acceptance_kimi_k3_mla.yaml \
  --models-root /models --dry-run

python3 examples/llm-api/measure_dspark_acceptance_disagg.py \
  --config examples/llm-api/dspark_acceptance_kimi_k3_mla_disagg.yaml \
  --models-root /models \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --dry-run
```

An explicit device list in a dry run checks the recipe's GPU count; it does not
check whether those GPUs exist or whether the models fit. Start the aggregate
run on eight idle GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 examples/llm-api/measure_dspark_acceptance.py \
  --config examples/llm-api/dspark_acceptance_kimi_k3_mla.yaml \
  --models-root /models --output-dir results/k3-mla-agg \
  --num-prompts 64 --max-tokens 256
```

After it succeeds, run disaggregation on sixteen idle GPUs using the frozen
corpus and aggregate reference:

```bash
python3 examples/llm-api/measure_dspark_acceptance_disagg.py \
  --config examples/llm-api/dspark_acceptance_kimi_k3_mla_disagg.yaml \
  --models-root /models \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --output-dir results/k3-mla-disagg \
  --prompt-file results/k3-mla-agg/prompts.jsonl \
  --aggregate-results results/k3-mla-agg \
  --num-prompts 64 --max-tokens 256 --concurrency 1
```

Use fresh output directories each time. Both runs use greedy decoding and
exclude two short warmup requests. For a smaller smoke run, set `--num-prompts
2 --max-tokens 32` in both commands. If you change other engine or head settings,
update both YAMLs and regenerate the aggregate reference.

Both presets explicitly set `KIMI_K3_AUX_ATTN_RES_STREAM: "0"` through the
recipe's `environment` mapping. This selects the prefix-sum hidden-state capture
convention associated with Inferact's vLLM lineage. The runner applies these
strings before TensorRT-LLM imports and overrides inherited values. The mapping
is recorded in the recipe and checked during aggregate comparison. It is a
checkpoint convention, not a performance tuning knob; it does not establish
answer accuracy.

Results include per-case logs, JSON, CSV, and a frozen prompt corpus. The
disaggregated AL comes from generation-side completed-request counters:

```text
AL = 1 + accepted draft tokens / verification steps
```

Comparison checks the corpus, tokenized prompts, head, declared environment,
and engine settings. It reports the AL difference; pass `--max-al-difference`
only when you have chosen an appropriate tolerance. See the
[disaggregated measurement documentation](dspark_acceptance_disagg.md) for
counter details and output artifacts. No GPU acceptance results are bundled.
