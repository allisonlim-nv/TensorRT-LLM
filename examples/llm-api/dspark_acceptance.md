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
the full sweep uses at most eight GPUs at a time:

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
| `deepseek-v4-pro` | DeepSeek-V4-Pro-DSpark | Embedded | 8 | 5 |
| `deepseek-v4-pro-nvfp4` | DeepSeek-V4-Pro-nvfp4-DSpark | Embedded | 8 | 3 |
| `deepseek-v4-flash-nvfp4` | DeepSeek-V4-Flash-nvfp4-DSpark | Embedded | 4 | 5 |

Select a subset with repeatable `--case` arguments:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al \
  --case qwen3-8b --case deepseek-v4-pro
```

Kimi recipes are excluded from this matrix because their existing configurations
require GB300 memory or 16 GPUs in one NVL72 domain. For a custom configuration
that spans nodes, `--launcher` accepts a command prefix with `{gpus}` expanded
to the case's GPU count. The Python environment, script, config, checkpoints,
and output directory must be accessible at the same paths on the launched nodes.

For example, inside an existing Slurm allocation:

```bash
python3 examples/llm-api/measure_dspark_acceptance.py \
  --models-root /models --output-dir dspark-al --case deepseek-v4-pro \
  --launcher 'srun --ntasks={gpus} --mpi=pmix trtllm-llmapi-launch'
```

The NVFP4 Pro and Flash settings are aggregate adaptations; the Pro NVFP4 preset
is experimental. All presets use eager execution, disabled overlap scheduling,
and whole-sequence prefill for this AL measurement. These are not throughput
benchmark recipes. The presets still require GPU validation.

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
