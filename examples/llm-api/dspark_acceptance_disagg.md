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
measurement script. By default it starts one context server, one generation
server, and a disaggregated proxy for each selected case. Context and generation use disjoint
GPU sets, with the same DSpark configuration enabled on both servers. Neither the
aggregate script nor its configuration needs to be modified.

The defaults match the aggregate measurement: the first 64 GSM8K test questions,
chat formatting with thinking disabled, greedy generation, up to 256 output
tokens, and two short warmup prompts excluded from the measured counters. Cases
run sequentially. This measures AL, not answer accuracy. Optional aggregate
comparison and an explicit AL-difference tolerance are supported.

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
GPU count alone is not a model-fit guarantee. Both roles now load the drafter,
so context-side memory usage can increase compared with generation-only
speculation. The built-in server launcher is single-node only; the
[external-server mode](#externally-managed-servers) can measure a separately
launched multi-node deployment.

For the vLLM/Inferact MLA DSpark head on Kimi K3 NVFP4, use the dedicated
[Kimi K3 presets and commands](dspark_acceptance_kimi_k3_mla.md). That pair uses
TP8 per worker and requires 16 GPUs total. The built-in launcher requires all
16 GPUs to be locally visible.

All cases use the corresponding target/drafter paths and draft lengths from
[`dspark_acceptance.yaml`](dspark_acceptance.yaml). DeepSeek targets contain their
embedded drafter weights in the same checkpoint; Qwen has a separate drafter
checkpoint. Common engine settings are batch size 1, sequence limit 8,192,
token budget 128, enabled decode CUDA graphs for batch size 1 on both roles,
enabled overlap scheduling and chunked prefill, and enabled KV block reuse.
DSV4 retains attention DP in all cases and SWA scratch except in its explicitly
named scratch-off control.
Prompts must exceed 128 tokens after formatting
to exercise multiple prefill chunks. Per-iteration logging is enabled on both
workers to inspect the context token counts and cached prefix. KV transfer uses NIXL with the Python
transceiver, required by the DeepSeek-V4 cache implementation.

These are new, unvalidated disaggregated measurement presets. Flash is derived
from `TestDeepSeekV4FlashDSpark.test_gsm8k_1p1d_dep4`, with settings adjusted to
match the aggregate measurement and speculation also enabled on context.
Qwen is an experimental disaggregated adaptation, not a qualified accuracy recipe.

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

Omitting `--case` selects all five cases. If fewer GPUs are visible than a case
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

Other controls are `--concurrency 1`, `--warmup-prompts 2` (zero disables
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
`spec_options` configures DSpark identically on both context and generation,
using the same drafter model and draft settings. There is no separate enable
flag. This enables the requested configuration; it does not establish that
draft KV-cache transfer works or that prefill drafter state reaches generation.
Changes to role topology, batching, cache settings, or the prompt protocol
define a different comparison.

## Externally managed servers

For a deployment launched separately, supply all three server origins and
select exactly one case. Each URL must use HTTP or HTTPS with no credentials,
path, query, or fragment. The following example uses placeholder hostnames;
replace them with the origins of your dedicated deployment:

```bash
python3 examples/llm-api/measure_dspark_acceptance_disagg.py \
  --config examples/llm-api/dspark_acceptance_disagg.yaml \
  --models-root /models --case qwen3-8b \
  --router-url http://router.example:8000 \
  --context-url http://context.example:8001 \
  --generation-url http://generation.example:8002 \
  --output-dir results/dspark-disagg-external \
  --prompt-file results/dspark-agg-reference/prompts.jsonl \
  --aggregate-results results/dspark-agg-reference \
  --num-prompts 64 --max-tokens 256 --concurrency 1
```

The caller must configure both workers to match the selected recipe, including
its target, drafter, DSpark settings, parallelism, cache settings, and
`environment` values. Both workers need `return_perf_metrics: true`,
`num_postprocess_workers: 0`, and `num_serve_frontends: 1`. Keep their
Prometheus multi-process directories separate. Use a dedicated deployment with
no other request traffic during warmup and measurement. Aggregate comparison
checks the declared recipe and corpus; it does not inspect the deployed engine
configuration.

The client still needs TensorRT-LLM's tokenizer dependencies and access to the
declared checkpoint paths. It skips local GPU discovery and GPU-count checks,
waits for all three `/health` endpoints and both worker metrics endpoints, and
then runs the same warmup, counter measurement, and aggregate comparison. It
never starts or stops these servers, including on failure; their owner handles
cleanup. External URLs are recorded in `manifest.json` and
`<case>/external_servers.json`. Server configs and logs remain with the external
deployment.

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
Context counters are also validated and saved separately, but are not added
to the generation counters when calculating AL. Zero context verification
counters are valid even with DSpark configured there.
Warmup is excluded by subtracting the pre-measurement snapshot, and missing or
inconsistent counters fail the measurement instead of producing a guessed AL.

This counts verification work before final EOS/output-length truncation, not
emitted tokens per step. The first generation verification, including
disaggregated bootstrap work, remains included.

Identical prompts and settings do not guarantee identical aggregate and
disaggregated AL. Handoff, drafter initialization, and HTTP admission differ
from aggregate inference. Enabling context-side speculation alone does not
prove drafter-state continuity or aggregate/disaggregated AL parity.
Use a fresh output directory and establish a new disaggregated baseline;
do not reuse an aggregate or generation-only
disaggregated AL threshold automatically.

The output directory contains `manifest.json`, frozen `prompts.jsonl`,
`summary.json`, `summary.csv`, and `<case>.json` for each successful case. Each
`<case>/` directory contains tokenized prompts and raw
`context-before.prom`, `context-after.prom`, `generation-before.prom`,
and `generation-after.prom` snapshots. Keep these artifacts for comparison.
Locally managed runs also save server logs and configs. The generated
`router.yaml`, `context.yaml`, and `generation.yaml` configs have
mode `0600` and contain a private handoff authentication key; do not publish
them without redacting the key.

The added coverage recipes require GPU validation. No CI test lists are changed,
and no measured GPU AL values or regression thresholds are bundled with these presets.

## Expanded matrix

The [aggregate coverage matrix](dspark_acceptance.md#expanded-coverage-and-matched-comparisons)
describes the five CP1 cases shared by both runners:

| Case | Context | Generation | GPUs |
|------|---------|------------|------|
| `qwen3-8b` | TP1 | TP1 | 2 |
| `qwen3-8b-explicit-pools` | TP1 | TP1 | 2 |
| `deepseek-v4-flash-nvfp4` | TP4/EP4 | TP4/EP4 | 8 |
| `deepseek-v4-flash-nvfp4-explicit-pools` | TP4/EP4 | TP4/EP4 | 8 |
| `deepseek-v4-flash-nvfp4-explicit-pools-no-scratch` | TP4/EP4 | TP4/EP4 | 8 |

All cases retain graphs, overlap, chunked prefill, and block reuse. DSV4 retains
attention DP. Its explicit-pool cases use `[0.2, 0.7, 0.1]` with target scratch
on/off; Qwen uses `[1.0]`, the only valid ratio for its single target group.
Helix CP2 is excluded because the current attention implementations do not
support these DSpark combinations.

Use `--aggregate-results <fresh-aggregate-output>` to report paired AL values and
differences. The runner selects `aggregate_case` from each recipe (defaulting to
the disagg case name), rejects mismatched inputs/configurations, and records the
comparison in JSON and CSV. `--max-al-difference VALUE` optionally makes excessive
absolute differences fail the case. Old results without tokenized-corpus hashes
must be regathered.
