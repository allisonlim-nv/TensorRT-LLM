# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TRTLLM-15289 Q1 (strengthened): real two-model V2 GPU-budget capture.

Drives the actual, unmodified LLM construction path for DraftTarget
(genuine two-engine, two-model) speculative decoding with
KVCacheManagerV2, and observes (does not fake) the KvCacheConfig object
each manager constructor actually receives. No flag is forced -- the
scheduler/creator decides `_should_create_separate_draft_kv_cache()` and
`_needs_gpu_kv_cache_budget_split()` on its own from a real
DraftTargetDecodingConfig (spec_dec_mode.use_one_engine() is False for
DraftTarget, so this is a genuine two-engine path, unlike EAGLE3 2-model
which is deprecated and silently coerced to one-model -- see
`llm_args.py::EagleDecodingConfig.validate_eagle_config`).
"""

import os
import sys

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


@pytest.mark.high_cuda_memory
def test_v2_two_model_draft_target_real_budget_not_split():
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 60:
        pytest.skip("Not enough memory to load target model")

    models_path = llm_models_root()
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    draft_model_dir = f"{models_path}/llama-3.2-models/Llama-3.2-1B-Instruct"

    budget_bytes = 4 * 1024**3  # B = 4 GiB, small enough to make a split visible if one occurred

    captured = []  # (is_draft, id(kv_cache_config), max_gpu_total_bytes)
    original_init = KVCacheManagerV2.__init__

    def observing_init(self, kv_cache_config, *args, is_draft=False, **kwargs):
        captured.append(
            (is_draft, id(kv_cache_config), kv_cache_config.max_gpu_total_bytes)
        )
        return original_init(self, kv_cache_config, *args, is_draft=is_draft, **kwargs)

    KVCacheManagerV2.__init__ = observing_init
    try:
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            use_kv_cache_manager_v2=True,
            max_gpu_total_bytes=budget_bytes,
        )
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=2,
            speculative_model=draft_model_dir,
        )
        llm = LLM(
            model=target_model_dir,
            backend="pytorch",
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
            max_batch_size=1,
            max_num_tokens=512,
            kv_cache_config=kv_cache_config,
            speculative_config=spec_config,
        )
        try:
            # Sanity: the real path actually constructed a two-model V2 run.
            resource_managers = llm._executor.engine.resource_manager.resource_managers
            from tensorrt_llm._torch.pyexecutor.resource_manager import (
                ResourceManagerType,
            )

            target_mgr = resource_managers[ResourceManagerType.KV_CACHE_MANAGER]
            draft_mgr = resource_managers[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]
            assert isinstance(target_mgr, KVCacheManagerV2)
            assert isinstance(draft_mgr, KVCacheManagerV2)
            assert target_mgr is not draft_mgr

            # A short generation to confirm the two-model run actually executes,
            # not just constructs.
            out = llm.generate(
                ["The capital of France is"],
                SamplingParams(max_tokens=8, temperature=0.0),
            )
            assert len(out[0].outputs[0].text) > 0
        finally:
            llm.shutdown()
    finally:
        KVCacheManagerV2.__init__ = original_init

    non_draft_calls = [c for c in captured if not c[0]]
    draft_calls = [c for c in captured if c[0]]
    assert len(non_draft_calls) >= 1, "target KVCacheManagerV2 was never constructed"
    assert len(draft_calls) >= 1, "draft KVCacheManagerV2 was never constructed"

    target_bytes = non_draft_calls[0][2]
    draft_bytes = draft_calls[0][2]

    print(f"[Q1-real] target max_gpu_total_bytes={target_bytes}")
    print(f"[Q1-real] draft  max_gpu_total_bytes={draft_bytes}")

    # The real, unforced creator path for two-model V2: budget is NOT split.
    # Both managers are constructed from a KvCacheConfig carrying the *same*
    # user-specified B; each manager independently sizes its own pool against
    # that same ceiling (further constrained by actual per-layer memory and
    # whatever GPU memory remains after the other manager has allocated).
    assert target_bytes == budget_bytes
    assert draft_bytes == budget_bytes
    assert target_bytes == draft_bytes
