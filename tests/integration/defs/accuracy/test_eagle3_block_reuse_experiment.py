# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                  KvCacheConfig, SamplingParams)

from ..conftest import llm_models_root, skip_pre_hopper

_PROMPT = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?")


def _run_eagle3_arm(enable_block_reuse):
    # Startup logs show two final target/draft KV managers per LLM
    # (plus a temporary estimation pair).
    root = llm_models_root()
    spec_config = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=f"{root}/EAGLE3-LLaMA3.1-Instruct-8B",
        eagle3_one_model=True,
    )
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                    free_gpu_memory_fraction=0.8)
    with LLM(
            model=f"{root}/llama-3.1-model/Llama-3.1-8B-Instruct",
            max_batch_size=1,
            sampler_force_async_worker=False,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=1,
                                              enable_padding=True),
            kv_cache_config=kv_cache_config,
            speculative_config=spec_config,
            max_stats_len=-1,
            enable_iter_perf_stats=True,
    ) as llm:
        params = SamplingParams(temperature=0, max_tokens=128)
        all_iters = []
        for i in range(10):
            llm.generate(_PROMPT, sampling_params=params, use_tqdm=False)
            raw = llm.get_stats(timeout=3)
            req_iters = [
                s['specDecodingStats'] for s in raw
                if s.get('specDecodingStats')
                and s['specDecodingStats']['numDraftTokens'] > 0
            ]
            all_iters.extend(req_iters)
            accepted = sum(s['numAcceptedTokens'] for s in req_iters)
            reqs = sum(s['numRequestsWithDraftTokens'] for s in req_iters)
            al = (accepted + reqs) / reqs if reqs else math.nan
            print(f"  req {i:2d}: drafted={sum(s['numDraftTokens'] for s in req_iters):4d}"
                  f"  accepted={accepted:4d}  AL={al:.3f}")

    assert all_iters, "No speculative decoding stats collected"
    accepted_total = sum(s['numAcceptedTokens'] for s in all_iters)
    reqs_total = sum(s['numRequestsWithDraftTokens'] for s in all_iters)
    overall = (accepted_total + reqs_total) / reqs_total
    print(f"  overall AL (enable_block_reuse={enable_block_reuse}): {overall:.4f}")
    return overall


@skip_pre_hopper
def test_eagle3_block_reuse_acceptance():
    for enable_block_reuse in [False, True]:
        print(f"\n=== enable_block_reuse={enable_block_reuse} ===")
        _run_eagle3_arm(enable_block_reuse)
