# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Investigation probe for TRTLLM-15289 (KVCacheV2 target/draft runtime behavior).

Answers Q8: for two-model speculative decoding on the KVCacheManagerV2
backend, with prefix/block reuse enabled, does a repeated prompt (which
causes the *target* manager to reuse a cached prefix, per
tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py's
_prepare_context_impl) still produce correct/matching output relative to a
reuse-disabled run and a cold (first) run -- i.e. is there any observable
sign of the draft engine reading unpopulated/garbage KV state for the
reused-prefix range described in
scratchpad/kvcachev2_context/topology_and_prefix_reuse.md Task 2 (a
source-only, not-yet-runtime-confirmed concern)?

This is model-output-level correctness (not corruption vs. no-corruption).
The topology_and_prefix_reuse.md audit already traced, from source, that the
draft manager's own KVCache is created with input_tokens=None (no reuse
match, num_committed_tokens=0) while context_current_position/
context_chunk_size for the draft's one-shot context step are copied from the
target's post-reuse chunk bounds (model_drafter.py's
_create_draft_request_for_request) -- i.e. the draft manager is told to skip
computing the reused-prefix range's KV without ever having populated it via
its own reuse or its own forward pass. That source trace is not repeated
here; this probe only adds the piece the source trace explicitly could not
resolve: does this observably corrupt output at the model level, run twice
with a real two-model EAGLE3 checkpoint pair on H200 GPU + KVCacheManagerV2
explicitly enabled (kv_cache_config.use_kv_cache_manager_v2=True), following
the same test pattern as the existing
tests/unittest/_torch/speculative/hw_agnostic/test_kv_cache_reuse.py (which
exercises the *default* KV cache manager, not V2 explicitly).

Mirrors test_kv_cache_reuse.py's structure/models/config almost exactly,
adding only `use_kv_cache_manager_v2=True` and running an explicit
reuse-disabled control in a separate LLM instance for direct output
comparison (test_kv_cache_reuse.py only compares cold vs. warm within one
enable_block_reuse=True LLM instance; it does not have a reuse-disabled
control run).

KNOWN ENVIRONMENT LIMITATION (as of this writing): in the
tensorrt_llm-devel-allim dev container used to author/run this probe, the
currently-installed tensorrt_llm.bindings native extension has no compiled
generation-phase masked-multi-head-attention kernel image for this
container's GPU (H200, SM 90) for the kernel instantiation this two-model
EAGLE3 generation path hits -- model loading succeeds, but the first
generation step fails with a native "no kernel image is available for
execution on the device" CUDA error
(decoderMaskedMultiheadAttentionLaunch.h:276). This is a pre-existing state
of the installed extension in that environment, unrelated to
KVCacheManagerV2 or anything under test here, and could not be worked
around without rebuilding the extension (out of scope for that
investigation). See scratchpad/kvcache_v2_runtime_results.md's Q8 section
for the full writeup. This test is left in place because the mechanism it
targets (target prefix reuse vs. an unpopulated draft cache) remains
untested at the model level anywhere in the repo; it should pass once run
in an environment with a complete SM-90 (or matching-architecture) kernel
build.
"""

import gc
import sys

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import Eagle3DecodingConfig, KvCacheConfig

pytestmark = [pytest.mark.gpu, pytest.mark.high_cuda_memory]


def _run(models_path: str, enable_block_reuse: bool, prompt: str,
         sampling_params: SamplingParams) -> tuple[str, str]:
    """Runs the same prompt twice inside one LLM instance; returns
    (cold_text, warm_text). With reuse enabled, the second call's context
    for this exact repeated prompt hits the target manager's prefix cache;
    with reuse disabled, it does not."""
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=enable_block_reuse,
        max_tokens=8192,
        use_kv_cache_manager_v2=True,
    )
    spec_config = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=eagle_model_dir,
        eagle3_one_model=False,
    )
    llm_spec = LLM(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        max_batch_size=1,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
        speculative_config=spec_config,
    )
    try:
        cold = llm_spec.generate(prompt, sampling_params).outputs[0].text
        warm = llm_spec.generate(prompt, sampling_params).outputs[0].text
    finally:
        llm_spec.shutdown()
        del llm_spec
        gc.collect()
        torch.cuda.empty_cache()
    return cold, warm


@pytest.mark.skip(
    reason="Blocked (re-confirmed twice this environment): this container's "
    "installed tensorrt_llm.bindings extension has no compiled generation-"
    "phase MMHA kernel image for SM 90 (H200) -- "
    "'CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor "
    "... no kernel image is available for execution on the device' "
    "(decoderMaskedMultiheadAttentionLaunch.h). First hit running this exact "
    "test; independently re-hit this session by a real two-model V2 LLM() "
    "construction+generate() attempt for Q1 "
    "(test_scratch_q1_v2_two_model_real_budget.py). Not a KVCacheManagerV2 "
    "or prefix-reuse issue -- both required model checkpoints load "
    "successfully. Remove this skip once the environment's extension "
    "includes an SM-90 build of this kernel path.")
def test_v2_two_model_spec_decode_prefix_reuse_output_matches_no_reuse():
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    prompt = "The future of AI is"
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    reuse_cold, reuse_warm = _run(models_path, True, prompt, sampling_params)
    print(f"[Q8] reuse=True  cold={reuse_cold!r} warm={reuse_warm!r}", file=sys.stderr)
    assert reuse_cold == reuse_warm, (
        "reuse-enabled run: cold vs. warm (prefix-cache-hit) output diverged "
        "-- this alone would already be runtime evidence of a target/draft "
        "prefix-reuse correctness problem for two-model V2 spec decode")

    noreuse_cold, noreuse_warm = _run(models_path, False, prompt, sampling_params)
    print(f"[Q8] reuse=False cold={noreuse_cold!r} warm={noreuse_warm!r}", file=sys.stderr)
    assert noreuse_cold == noreuse_warm

    assert reuse_warm == noreuse_warm, (
        "reuse-enabled warm output differs from reuse-disabled warm output "
        "for the identical repeated prompt -- runtime evidence that target "
        "prefix reuse changes generation output under two-model V2 spec "
        "decode (the topology_and_prefix_reuse.md source-trace concern "
        "manifesting at the model level)")


if __name__ == "__main__":
    test_v2_two_model_spec_decode_prefix_reuse_output_matches_no_reuse()
