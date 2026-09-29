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

# DSpark acceptance length on aggregate inference

`measure_dspark_acceptance.py` runs each selected target/drafter pair in a fresh
child process and reports corpus acceptance length (AL). Each engine performs
both prefill and decode. The runner does not start disaggregated servers or
collect disaggregated results.

By default it measures the first 64 GSM8K test questions with greedy decoding,
up to 256 output tokens each. Two short warmup requests are excluded from AL;
use `--warmup-prompts 0` to disable warmup.

Run from the repository root in an installed TensorRT-LLM environment with the
checkpoints mounted locally, once the container build has finished. The default
matrix targets a single node with eight B200 GPUs. Cases run sequentially, so
the aggregate sweep uses at most four GPUs at a time:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al --dry-run

python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al \
  --num-prompts 64 --max-tokens 256
```

`--models-root` defaults to `LLM_MODELS_ROOT`. Paths in
[`dspark_acceptance.yaml`](dspark_acceptance.yaml) are relative to that root;
copy the YAML and pass `--config /path/to/cases.yaml` to change paths, draft
lengths, or hardware settings. Case options merge with the defaults, including
nested mappings such as `kv_cache_config`.
Use a new `--output-dir` for each actual run; the script rejects existing
directories to avoid mixing results. A dry run creates no output files.

| Case | Target | Drafter | GPUs | Maximum draft length |
|------|--------|---------|------|----------------------|
| `qwen3-8b` | Qwen3-8B | `dspark_qwen3_8b_block7` | 1 | 7 |
| `deepseek-v4-flash-nvfp4` | DeepSeek-V4-Flash-nvfp4-DSpark | Embedded | 4 | 5 |

Select a subset with repeatable `--case` arguments:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al \
  --case qwen3-8b --case deepseek-v4-flash-nvfp4
```

Kimi recipes are excluded from this matrix because their existing configurations
require GB300 memory or 16 GPUs in one NVL72 domain. For a custom configuration
that spans nodes, `--launcher` accepts a command prefix with `{gpus}` expanded
to the case's GPU count. The Python environment, script, config, checkpoints,
and output directory must be accessible at the same paths on the launched nodes.

For example, inside an existing Slurm allocation:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al --case deepseek-v4-flash-nvfp4 \
  --launcher 'srun --ntasks={gpus} --mpi=pmix trtllm-llmapi-launch'
```

The NVFP4 Flash settings are aggregate adaptations. All presets enable decode CUDA graphs for batch size 1, overlap
scheduling, and chunked prefill with a 128-token budget for this AL measurement. Prompts
must exceed 128 tokens after formatting to exercise multiple chunks. Per-iteration
logging (`print_iter_log: true`) is enabled to inspect the context token counts and
cached prefix. The aggregate runner defaults `TLLM_LOG_LEVEL` to `info` before loading
TensorRT-LLM, and its workers inherit that setting; an explicitly set environment
value takes precedence. Batch size remains 1 and the sequence limit remains 8,192.
These are not throughput benchmark recipes. The presets still require GPU validation.

Use `--prompt-file prompts.jsonl` to evaluate a fixed corpus. Each JSONL record
has a `prompt` string, for example:

```json
{"prompt": "Explain why the sky is blue in three sentences."}
{"prompt": "Write a Python function that merges two sorted lists."}
```

The default `--prompt-format chat` applies the target tokenizer's chat template
with `enable_thinking: false`. Set `--prompt-format raw` for already formatted
prompts. AL depends on the corpus, formatting, generation length, and draft
length; keep these fixed when comparing drafters. In particular, the Qwen drafter
expects chat prompts, and completion-style prompts can substantially lower AL.

The corpus metric includes the bonus token from each verification step:

```text
AL = 1 + sum(accepted draft tokens) / sum(verification steps)
```

Verification steps come from `per_pos_drafted[0]` for completed requests across
all attention-DP ranks. This weights each verification step equally, rather than
averaging per-request AL values. These are verification counters before final
EOS/output-length truncation, not emitted tokens per step. Requests that finish
before drafting contribute no verification steps. The output directory contains JSON and CSV
summaries plus per-case logs and result files. Missing checkpoints and failed
cases appear with statuses and make the run exit nonzero, so partial results
cannot be mistaken for a complete sweep.
The frozen `prompts.jsonl` and its SHA-256 in `manifest.json` identify the exact
corpus shared by all cases.

No GPU AL numbers are bundled with this script; running it produces the measured
results for the selected checkpoints and prompts.

## Expanded coverage and matched comparisons

The original two cases retain graphs, overlap, chunked prefill, and block reuse;
DSV4 also retains attention DP and target SWA scratch reuse. Three additional
aggregate cases extend coverage without replacing those cases:

| Case suffix | Models | Settings |
|-------------|--------|----------|
| `-explicit-pools` | Both | Original features plus explicit target `pool_ratio` |
| `-explicit-pools-no-scratch` | DSV4 | Explicit pools with target scratch disabled as a control |

The full aggregate matrix has five cases, all at CP1. Qwen uses `pool_ratio: [1.0]`:
its single target group requires that value because ratios must sum to one.
This tests the explicit configuration path; a different target-pool distribution
is only meaningful for a model with multiple target groups.
DSV4 uses `[0.2, 0.7, 0.1]` for its SWA, full compressed-history, and short
compressor-state target groups in both scratch variants. This manually assigns
20%/70%/10% of the target share instead of deriving ratios from the default
cache-sizing heuristic. These are coverage settings, not tuned serving ratios.
The unified manager reserves draft storage separately and scales the
target ratios within the remaining quota; do not append a draft ratio yourself.

The checked-in `results/dspark-agg-reference-11/prompts.jsonl` contains the
shared 1,319-prompt corpus. Run all five aggregate cases, then all five
disaggregated cases against the same frozen corpus. Set `LLM_MODELS_ROOT`, use fresh output directories, and keep
the counts the same in both commands:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --output-dir results/dspark-agg-coverage \
  --prompt-file results/dspark-agg-reference-11/prompts.jsonl \
  --num-prompts 1319 --max-tokens 1024 --warmup-prompts 2

python3 examples/llm-api/measure_dspark_acceptance_disagg.py \
  --output-dir results/dspark-disagg-coverage --devices 0,1,2,3,4,5,6,7 \
  --prompt-file results/dspark-agg-coverage/prompts.jsonl \
  --aggregate-results results/dspark-agg-coverage \
  --num-prompts 1319 --max-tokens 1024 --warmup-prompts 2 --concurrency 1
```

Each disagg case uses the aggregate result with the same case name. Comparison checks the corpus,
tokenized prompts, generation limits, warmup, checkpoint paths, speculation
settings, and engine options before
launching servers. JSON/CSV
summaries include aggregate AL and the signed disagg-minus-aggregate difference.
Optionally pass `--max-al-difference VALUE` to enforce an absolute tolerance;
there is no default tolerance. Aggregate bulk submission and disagg HTTP
admission still differ.

Helix CP2 is excluded from these presets: the current Qwen attention path rejects
multi-token speculative Helix verification, and DeepSeek-V4 sparse attention
rejects Helix post-processing. GPU validation of the added disagg pool/scratch
combinations is still required. Regather explicit-pool baselines whenever their
ratios change; the comparison rejects results with different engine settings.
