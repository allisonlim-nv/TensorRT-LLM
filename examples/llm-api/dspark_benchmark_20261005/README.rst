.. Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Qwen3-8B DSpark B200 benchmark bundle
===================================

This directory packages the tools, workload, and evidence used for the October 5,
2026 comparison. ``summary.csv`` contains the full-run results; ``comparison.json``
contains the methodology, exact revisions, consistency checks, and limitations.
The publication branch starts at fork main
``234ee16411a0ced12cbeb242f2fd1eb5390e4d61``. That packaging base is not one of the
three measured source revisions:

* Current: ``ff47bcd68987adb4d148b7110a39cd7797f6ee51``.
* NVIDIA main, resolved once: ``d27e97f6dece43028c40fb602cb77bb5ab28132d``.
* Comparison: ``ec4ef2666d05ef60012c834a43dca62c75509aec``.

Every full run completed 1,319 requests with zero measured failures and 345,680
output tokens. Current delivered 531.76 output tokens/s/GPU, main 473.60, and
``ec4ef266`` 424.50. AL was 6.334, 5.275, and 6.334, respectively. These are single
trials of whole revisions, not an isolated one-patch causal experiment.

Contents
--------

``artifacts.tar.gz`` is tracked with Git LFS. ``manifest.json`` records its SHA256,
every included file's original and published hashes, and every excluded file.
The archive restores the original relative paths under
``results/dspark-endpoints-current-20261005T183845Z`` and the frozen collector at
``examples/llm-api/measure_dspark_endpoints.py``. It includes:

* The exact collector, recovered aggregate/disaggregated runners and YAML recipes,
  original 1,319-prompt JSONL, generated tokenized datasets, and client configs.
* All three smoke/full runs' raw client reports, request events, timing/token
  audits, generation/prefill counter snapshots, resolved configs, and server logs.
* The failed initial warmup attempt, retained separately from reported results.
* Native build logs/configuration, dependency pins, binary hashes, worker import
  provenance, model/tokenizer metadata, source identities, and empty source diffs.
* The pinned Endpoints metric/schema source excerpts and image/source checks.
* Original launch, audit, replay, native-install, runtime-verification, and
  checkpoint-restoration scripts.

The readable sibling runner files and ``audit_reports.py`` are provided for
inspection. The archive preserves the exact measured script bytes even if code
formatting changes in the readable copies.

Only generated per-run router authentication keys are redacted (21 occurrences).
All timing/token reports, counters, and prompt content retain their original
bytes. The original unredacted results remain on the node.

External dependencies
---------------------

Model weights, the existing development container/venv, compiled native binaries,
and the 1.3 GB local native recovery archive are not vendored. Their paths,
versions, and available hashes are recorded. The Git index and ephemeral
Python/Prometheus cache files are also excluded; exported counters are included.

The actual environment was:

* Host checkout: ``/home/scratch.allim_coreai/src/TensorRT-LLM``.
* Container: ``tensorrt_llm-devel-allim``, checkout ``/code/tensorrt_llm``,
  Python ``/code/tensorrt_llm/.venv-3.12/bin/python3``.
* Models root: ``/home/scratch.trt_llm_data_ci/llm-models``.
* Target: ``Qwen3/Qwen3-8B``; drafter: ``dspark/dspark_qwen3_8b_block7``.
* Separate client image: ``ghcr.io/mlcommons/endpoints:89904fb``;
  digest ``sha256:ba85b060b9c1de0d3b44d52f6fce8369e81f88ac31c460807cdc1e61441249bf``.

The server container and compatible libraries must already exist. Source revision
selection and native rebuilds must be done explicitly; the collector does not
switch source revisions or install dependencies. Main and ``ec4ef266`` required
native rebuilds during this comparison; current used the existing compatible
libraries. All original source/native state was restored after measurement.

Restore the frozen tools and reports
-----------------------------------

From this publication branch, retrieve the LFS object::

    git lfs pull --include="examples/llm-api/dspark_benchmark_20261005/artifacts.tar.gz"

Verify and unpack into the checkout mounted by the existing server container::

    python3 examples/llm-api/dspark_benchmark_20261005/unpack.py \
        /home/scratch.allim_coreai/src/TensorRT-LLM --check-only
    python3 examples/llm-api/dspark_benchmark_20261005/unpack.py \
        /home/scratch.allim_coreai/src/TensorRT-LLM

The unpacker accepts existing files matching either their original or redacted
published hashes, skips them, and refuses differing files before writing anything.
It never changes source revisions, libraries, dependencies, or running servers.
Use the archived frozen collector for exact hash identity. If a readable source
copy has different formatting, unpack into the intended implementation checkout
where that file is absent, rather than overwriting local work.

Run the same workload
---------------------

After selecting the intended implementation and verifying compatible native
libraries, run from the actual host checkout::

    cd /home/scratch.allim_coreai/src/TensorRT-LLM
    bench_base=results/dspark-endpoints-current-20261005T183845Z
    bash "$bench_base/rerun.sh" repeat-current

This creates a new timestamped output directory. It reuses the recovered server
launcher and executes smoke then full measurements, each with fresh servers and
2 separate warmup requests. It refuses occupied GPUs and stops only its own
servers. The equivalent explicit command is::

    python3 examples/llm-api/measure_dspark_endpoints.py run \
        --runner "$bench_base/recovered/measure_dspark_acceptance_disagg.py" \
        --config "$bench_base/recovered/dspark_acceptance_disagg.yaml" \
        --prompts "$bench_base/prompts.jsonl" \
        --output "$bench_base/repeat-$(date -u +%Y%m%dT%H%M%SZ)" \
        --devices 0,1

The client command generated for each phase is
``inference-endpoint benchmark from-config --config <phase>-client.yaml`` inside
the pinned image, with the server container's network. The resolved YAML/JSON,
full Docker command, and actual source SHA/diff are saved for every invocation.

On the original node only, after unpacking and with its existing recovery archive
still present, the checkpoint-specific three-revision replay is::

    bash "$bench_base/comparison/replay.sh"

That script validates its original branch, source, collector, native checksums,
and client image before starting. It rebuilds/stages compatible native targets
for main and ``ec4ef266``, runs them sequentially, and restores the original
checkpoint. It is not a general checkout manager. On another node, create a fresh
checkpoint and use the recorded per-revision build/launch commands; the original
node's native recovery archive is deliberately not published. Do not replace a
newer local source state with this historical checkpoint.

Matched settings and metric accounting
--------------------------------------

GPU 0 handled prefill and GPU 1 generation, TP=1 each. Both had DSpark enabled and
explicit KVCache V2, with the default C++ manager verified. NIXL used its Python
transfer runtime, independently of the C++ cache manager. Client concurrency was
4 while server batch size remained 1. CUDA graphs used batch [1], overlap and
block reuse stayed enabled, draft length was 7, and streaming interval was 1.

Requests streamed through the disaggregation router with temperature 0,
max output 1024, min output 0, and EOS respected. Qwen chat templating disabled
thinking. The longest input was 200 tokens; 200 + 1024 fit the 8192-token context.
The recovered reference-11 corpus was identical across runs; equivalence to the
previously deleted reference-1 file cannot be verified.

AL = 1 + accepted generation draft tokens / generation verification steps.
The position-zero drafted counter was validated to increment once per actual
verification step. No fixed-draft-length division, prefill counters, duplicated
ranks, or synthetic acceptance was used. Counter snapshots bracket the same
measured requests as the client report, after warmup.

TTFT and TPOT use the pinned client's implementation. TPOT counts tokens in text
after the first content chunk, not OSL minus one. E2E interactivity is total output
tokens divided by summed request E2E seconds. Output TPS/GPU uses the client's
measurement duration and divides by both serving GPUs. Mean OSL uses the client's
Qwen retokenization. ``audit_reports.py`` checks these semantics against the raw
request events using the pinned image's tokenizer and output types.
