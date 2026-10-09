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

# vLLM Kimi K3 MLA DSpark acceptance

The aggregate and disaggregated runners measure corpus acceptance length (AL)
for `moonshotai/Kimi-K3` with the **MLA drafter** `Inferact/Kimi-K3-DSpark`.
Kimi K3 itself combines KDA and MLA attention. The default draft length is seven;
the drafter uses `FLASHINFER_MLA` and greedy drafting with temperature zero,
following its [model card](https://huggingface.co/Inferact/Kimi-K3-DSpark).

Run inside a vLLM environment supporting Kimi K3, MLA DSpark, and, for disaggregation,
NIXL transfer of the hybrid target and drafter caches. See the
[vLLM Kimi K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)
for compatible builds. These scripts call the installed `vllm` executable;
they do not install vLLM or TensorRT-LLM, build images, or submit Slurm jobs.
The client requires `prometheus-client`; `datasets` is needed only when no
`--prompt-file` is supplied. `fastsafetensors` must be available in the server
environment. `--vllm-executable` can select another installation.

The default topology is TP8 for each server: eight GPUs for aggregate inference,
and sixteen for one prefill plus one decode server. The K3 published recipe uses
B300-class memory. The scripts do not establish that a checkpoint fits a given
GPU, and these measurement presets still need GPU validation. Local launch
requires all selected GPUs to be visible on the current host; use existing
servers for multi-node deployments.

## Aggregate

Preview the resolved launch without loading models, discovering GPUs, downloading
prompts, or writing files. With no `--devices` or `CUDA_VISIBLE_DEVICES`, a dry run
uses illustrative indices `0..TP-1` (twice as many for disaggregation).

```bash
python3 examples/llm-api/measure_vllm_dspark_acceptance.py --dry-run

python3 examples/llm-api/measure_vllm_dspark_acceptance.py \
  --model /models/Kimi-K3 \
  --drafter /models/Kimi-K3-DSpark-MLA \
  --devices 0,1,2,3,4,5,6,7 \
  --output-dir results/vllm-dspark-agg
```

`--model` and `--drafter` accept local paths or Hugging Face IDs. Defaults use
the IDs above. Select idle GPUs; child servers receive distinct subsets of
`--devices`, or of `CUDA_VISIBLE_DEVICES` when it is set. Actual runs otherwise
discover local indices with `nvidia-smi`. Each output directory must be new.

## Disaggregated

Both servers load the same target and MLA drafter with the same TP and sampling
settings. NIXL uses producer/consumer roles, separate side-channel ports, and
`VLLM_SSM_CONV_STATE_LAYOUT=DS`. Transfer failures use the `fail` policy.

```bash
python3 examples/llm-api/measure_vllm_dspark_acceptance_disagg.py --dry-run

python3 examples/llm-api/measure_vllm_dspark_acceptance_disagg.py \
  --model /models/Kimi-K3 \
  --drafter /models/Kimi-K3-DSpark-MLA \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --prompt-file results/vllm-dspark-agg/prompts.jsonl \
  --aggregate-results results/vllm-dspark-agg \
  --output-dir results/vllm-dspark-disagg
```

The client implements the
[upstream K3 prefill/decode handoff](https://github.com/vllm-project/vllm/blob/main/tests/models/kimi_k3/test_prefix_cache.py):
prefill receives the tokenized prompt and `max_tokens=1`; decode receives the
same original prompt, the full requested output limit, and prefill's
`kv_transfer_params`. No proxy process is required. Missing handoff metadata
fails the run. Prefill's one-token response is not appended to the prompt or
counted in decode AL. Prefill counters are saved separately for diagnostics.

The local HTTP ports default to `8000` (aggregate/decode) and `8001` (prefill),
with NIXL side-channel ports `5557` and `5657`. Change these with `--port`,
`--prefill-port`, `--prefill-side-channel-port`, and `--decode-side-channel-port`.
The local launcher binds HTTP to loopback and shuts down its own process groups
on completion, errors, Ctrl-C, or SIGTERM.

For servers already launched across nodes, pass their origins instead of local
GPU lists:

```bash
python3 examples/llm-api/measure_vllm_dspark_acceptance.py \
  --server-url http://aggregate-host:8000 \
  --output-dir results/vllm-dspark-agg-remote

python3 examples/llm-api/measure_vllm_dspark_acceptance_disagg.py \
  --prefill-url http://prefill-host:8000 \
  --decode-url http://decode-host:8000 \
  --prompt-file results/vllm-dspark-agg-remote/prompts.jsonl \
  --aggregate-results results/vllm-dspark-agg-remote \
  --output-dir results/vllm-dspark-disagg-remote
```

External servers must be dedicated to this measurement, expose `/health`,
`/tokenize`, `/v1/completions`, and `/metrics`, and serve the `--model` name.
Configure the same target, drafter, TP, limits, prefix caching, chunked prefill,
and speculation settings shown by `--dry-run`. Pass corresponding runner
arguments when changing those settings. Both workers need network access to
each other's NIXL endpoints; use the
[NIXL deployment guide](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_usage.md).
The runner neither reconfigures nor stops external servers. Their configuration
must be verified by the operator; the recorded CLI recipe is not an attestation
of a remote server's configuration.

## Workload and results

Defaults match the existing acceptance corpus: first 64 GSM8K test questions,
chat formatting with thinking disabled, two short warmup requests, up to 256
output tokens, and concurrency one. Flags include `--num-prompts`, `--max-tokens`,
`--warmup-prompts`, `--concurrency`, `--seed`, and `--prompt-format raw`.
For sampled generation use `--temperature` and `--top-p`. Drafting defaults
to greedy at temperature zero and probabilistic otherwise; override it with
`--draft-sample-method greedy|probabilistic`. `--rejection-sample-method standard|block`
selects the rejection algorithm explicitly (otherwise vLLM supplies its default).
Use `--num-speculative-tokens` to change draft length within the checkpoint's
supported range. For example, pass the same flags to both runners when comparing
four-token speculation with sampled targets and greedy drafts:

```bash
python3 examples/llm-api/measure_vllm_dspark_acceptance.py \
  --num-speculative-tokens 4 --temperature 1 --top-p 1 \
  --draft-sample-method greedy --rejection-sample-method standard \
  --kv-cache-dtype fp8 --dry-run
```

`--prompt-file` accepts JSONL records of the form `{"prompt": "..."}` and must
contain at least `--num-prompts` records. The frozen corpus and token IDs are
saved; both disaggregated tokenizers must produce identical tokens.

The server defaults enable prefix caching and chunked prefill, with a 2,048-token
batch budget and an 8,192-token sequence limit. `--max-num-batched-tokens` and
`--max-model-len` override those limits. `--kv-cache-dtype` selects the target
KV cache dtype (`auto`, `fp8`, or `fp8_e4m3`). The FP8 modes also configure
FlashInfer MLA prefill and quantized prefill queries, as required by K3.
Set the token budget lower than your prompt
lengths to exercise multiple prefill chunks. These are AL measurement settings,
not a throughput-tuned recipe. No attention-DP, expert-parallel, pipeline-parallel,
or decode-context-parallel topology is implied by TP8.

Warmup is excluded using before/after Prometheus snapshots. The runner waits
for completed-request counters before taking snapshots, then calculates the
[vLLM metric](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py):

```text
AL = 1 + delta(spec_decode_num_accepted_tokens) / delta(spec_decode_num_drafts)
AR = delta(spec_decode_num_accepted_tokens) / delta(spec_decode_num_draft_tokens)
```

Disaggregated AL uses **decode counters only**. The bonus token is included by
convention; this is not emitted output tokens per iteration. Missing counters,
counter resets, invalid totals, zero draft verification, failed requests, and
unexpected completion counts fail the measurement. Keep server statistics
enabled. Per-position accepted counters, when present, are validated too.

Each run saves `manifest.json`, `prompts.jsonl`, `prompt_token_ids.json`, raw
Prometheus snapshots, `summary.json`, and `summary.csv`; locally launched servers
also have per-role logs. Failed runs return nonzero and write a failed summary.
`--aggregate-results` requires a successful vLLM aggregate run with matching
tokenized prompts, declared recipe, generation, warmup, and concurrency settings.
It adds aggregate AL and signed disagg-minus-agg AL to the summary. An optional
`--max-al-difference VALUE` makes an absolute difference above that value fail;
there is no default tolerance or claim of aggregate/disaggregated parity.

These scripts are independent of the TensorRT-LLM runners and do not accept
their YAML configurations or result directories as matched references.

## Local validation

The CPU regression suite checks metrics, handoff payloads, launch cleanup, and
mocked aggregate/disaggregated measurements. Run it directly in an environment
with `prometheus-client`, without TensorRT-LLM or a GPU:

```bash
python3 tests/unittest/llmapi/test_vllm_dspark_acceptance_script.py
```

This is local runner QA, following the existing acceptance-script tests. It is
not registered in GPU CI and does not validate real model loading or KV transfer.
