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

# DSpark acceptance length on disaggregated inference

`measure_dspark_acceptance_disagg.py` is a separate counterpart of the aggregate
measurement script. It starts one context server, one generation server, and a
disaggregated proxy for each selected case. Context and generation use disjoint
GPU sets, with DSpark enabled on the generation server only. Neither the
aggregate script nor its configuration needs to be modified.

The defaults match the aggregate measurement: the first 64 GSM8K test questions,
chat formatting with thinking disabled, greedy generation, up to 256 output
tokens, and two short warmup prompts excluded from the measured counters. Cases
run sequentially. This is an AL measurement, not an answer-accuracy test or an
AL-threshold gate.

## Hardware and configurations

The presets preserve the aggregate recipe's TP/EP settings separately for each
role. Disaggregation therefore doubles the total GPU count:

| Case | Context TP/EP | Generation TP/EP | Total GPUs | Attention DP | Maximum draft length |
|------|---------------|------------------|------------|--------------|----------------------|
| `qwen3-8b` | 1/1 | 1/1 | 2 | Off | 7 |
| `deepseek-v4-flash-nvfp4` | 4/4 | 4/4 | 8 | On | 5 |

This matrix targets an eight-B200 node and runs at most eight GPUs at a time.
Both DeepSeek-V4-Pro variants are excluded because preserving TP8/EP8 for each
role would require 16 GPUs. The runner does not reduce TP to make a case fit.
GPU count alone is not a model-fit guarantee. This launcher is single-node only.

All cases use the same target/drafter paths and draft lengths as
[`dspark_acceptance.yaml`](dspark_acceptance.yaml). DeepSeek targets contain their
embedded drafter weights in the same checkpoint; Qwen has a separate drafter
checkpoint. Common engine settings are batch size 8, sequence limit 8,192,
token budget 8,192, and disabled CUDA graphs, overlap scheduling, chunked prefill,
and KV block reuse. KV transfer uses NIXL with the Python transceiver, required
by the DeepSeek-V4 cache implementation.

These are new, unvalidated disaggregated measurement presets. Flash is derived
from `TestDeepSeekV4FlashDSpark.test_gsm8k_1p1d_dep4`, with settings adjusted to
match the aggregate measurement. Qwen is an experimental disaggregated
adaptation, not a qualified accuracy recipe.

## Running

Run inside the built container, with TensorRT-LLM, its serving dependencies,
NIXL, and model checkpoints available. Use a fresh output directory for each
actual run. A dry run prints resolved configurations without loading weights or
starting servers:

```bash
cd /code/tensorrt_llm

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="/code/tensorrt_llm${PYTHONPATH:+:$PYTHONPATH}" \
/code/tensorrt_llm/.venv-3.12/bin/python3 \
examples/llm-api/measure_dspark_acceptance_disagg.py \
  --models-root /home/scratch.trt_llm_data_ci/llm-models \
  --output-dir results/dspark-disagg-preview \
  --dry-run
```

Start with the two-GPU Qwen smoke measurement:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="/code/tensorrt_llm${PYTHONPATH:+:$PYTHONPATH}" \
/code/tensorrt_llm/.venv-3.12/bin/python3 \
examples/llm-api/measure_dspark_acceptance_disagg.py \
  --models-root /home/scratch.trt_llm_data_ci/llm-models \
  --case qwen3-8b \
  --output-dir results/dspark-disagg-qwen-1 \
  --num-prompts 64 \
  --max-tokens 256
```

After validating the smoke runs, use the aggregate reference's frozen corpus
for a comparable workload on the two cases that fit the eight-GPU node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="/code/tensorrt_llm${PYTHONPATH:+:$PYTHONPATH}" \
/code/tensorrt_llm/.venv-3.12/bin/python3 \
examples/llm-api/measure_dspark_acceptance_disagg.py \
  --models-root /home/scratch.trt_llm_data_ci/llm-models \
  --case qwen3-8b \
  --case deepseek-v4-flash-nvfp4 \
  --output-dir results/dspark-disagg-reference-1 \
  --prompt-file results/dspark-agg-reference-1/prompts.jsonl \
  --num-prompts 1319 \
  --max-tokens 1024
```

The prompt file must contain at least the requested number of records. Each
record has a `prompt` string. Omit `--prompt-file` to load GSM8K instead. Use
`--prompt-format raw` for already formatted prompts; the default is `chat`.

Omitting `--case` selects both presets. If fewer GPUs are visible than a case
requires, it is recorded as skipped. A run with successful cases and only
hardware skips may exit successfully, but its summary explicitly identifies
the incomplete coverage. Failed cases make the run fail; selecting only cases
that are skipped also exits nonzero. Do not treat a zero exit code alone as
evidence that every selected pair was measured.

`--models-root` defaults to `LLM_MODELS_ROOT`. `--devices` accepts an explicit
comma-separated GPU list; otherwise the runner uses `CUDA_VISIBLE_DEVICES`, or
discovers GPUs with `nvidia-smi` when it is unset. Select only idle GPUs. The
runner's server processes use distinct device subsets, and the runner shuts
down the processes it started after each case.

Other controls are `--concurrency 8`, `--warmup-prompts 2` (zero disables
warmup), `--startup-timeout 3600`, and `--request-timeout 1800`, with timeouts in
seconds. Concurrency limits simultaneous HTTP requests and is recorded in
`manifest.json`. Although the engine settings match the aggregate presets,
HTTP admission makes this a different queued workload from the aggregate
script's bulk submission. Keep concurrency fixed across disaggregated runs.
Server initialization can include kernel downloads, compilation, and model
loading; inspect `<case>/router.log`, `<case>/context.log`, and
`<case>/generation.log` under the output directory if startup fails.

Copy [`dspark_acceptance_disagg.yaml`](dspark_acceptance_disagg.yaml) and pass
`--config /path/to/cases.yaml` to customize recipes. `common_options` merges
recursively with `defaults.common_options` for both roles; optional
`context_options` and `generation_options` override their respective roles.
`spec_options` configures DSpark on generation only. Changes to role topology,
batching, cache settings, or the prompt protocol define a different comparison.

## Measurement and comparison with aggregate inference

The runner takes snapshots of the generation server's request-derived
Prometheus counters at `/prometheus/metrics` after warmup and after the measured
requests. From their differences it calculates:

```text
verification_steps = drafted tokens at position 0
accepted_draft_tokens = sum of accepted tokens across draft positions
AL = 1 + accepted_draft_tokens / verification_steps
```

The counters aggregate completed generation requests across attention-DP ranks.
They avoid the iteration-statistics `/metrics` path and its attention-DP
speculative-counter reporting issue. Each server has its own Prometheus
multi-process directory so context and generation statistics are not mixed.
Warmup is excluded by subtracting the pre-measurement snapshot, and missing or
inconsistent counters fail the measurement instead of producing a guessed AL.

This counts verification work before final EOS/output-length truncation, not
emitted tokens per step. Disaggregated handoff without context-side drafting
also initializes draft slots for the first generation verification; those
bootstrap verification counters are included.

Identical prompts and settings do not guarantee identical aggregate and
disaggregated AL. In particular, the standalone Qwen drafter does not receive
the target's prefill hidden-state history across the disaggregation boundary.
The generation-side drafter's starting state and bootstrap work can therefore
differ from aggregate inference. Establish a separate disaggregated baseline;
do not reuse an aggregate AL threshold automatically.

The output directory contains `manifest.json`, frozen `prompts.jsonl`,
`summary.json`, `summary.csv`, and `<case>.json` for each successful case. Each
`<case>/` directory contains server logs, tokenized prompts, server configs,
and raw `context-before.prom`, `context-after.prom`, `generation-before.prom`,
and `generation-after.prom` snapshots. Keep these artifacts for comparison.
The generated `router.yaml`, `context.yaml`, and `generation.yaml` configs have
mode `0600` and contain a private handoff authentication key; do not publish
them without redacting the key.

Validation so far is CPU-only local testing; no GPU run has validated this
runner, and no CI test lists are changed. No measured GPU AL values or
regression thresholds are bundled with these presets.
