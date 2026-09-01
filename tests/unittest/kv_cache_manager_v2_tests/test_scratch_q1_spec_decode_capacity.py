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

Answers: In supported two-model speculative decoding, after real manager
construction, what are the target and draft GPU page/pool capacities and
actual GPU allocations for an explicit max_gpu_total_bytes=B?

Method:
 1. Use the real `KvCacheCreator._split_kv_cache_budget_for_draft` /
    `_compute_draft_budget_shares` code (unmodified, imported from
    tensorrt_llm._torch.pyexecutor._util) to split B into target/draft byte
    budgets. The leaf per-token cost inputs are supplied directly (as
    CacheCost) rather than derived from a real HF model, because no model
    artifact / LLM_MODELS_ROOT is available in this environment. This is the
    same substitution technique already used by the repo's own
    tests/unittest/_torch/executor/test_kv_cache_budget_split.py.
 2. Feed the resulting byte budgets into the real native
    tensorrt_llm.runtime.kv_cache_manager_v2.KVCacheManager (the same
    class/bindings used by tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py's
    TestKVCacheManagerV2 harness) to construct two independent, real,
    GPU-backed managers and read back their actual allocated capacities.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_manager_v2 import create_config  # noqa: E402

from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once  # noqa: E402

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402

GB = 1 << 30
TOKENS_PER_BLOCK = 32
KV_BUF_SIZE = 8192  # bytes per (layer, key/value) buffer entry, matches create_config default


def _make_creator(max_gpu_total_bytes: int, total_kv_per_token: int,
                   target_kv_per_token: int) -> KvCacheCreator:
    c = object.__new__(KvCacheCreator)
    c._kv_cache_config = KvCacheConfig(max_gpu_total_bytes=max_gpu_total_bytes)
    c._tokens_per_block = TOKENS_PER_BLOCK
    c._get_kv_size_per_token = lambda cfg: CacheCost(slope=total_kv_per_token, intercept=0)
    c._should_create_separate_draft_kv_cache = lambda: True
    c._per_manager_cache_cost = (
        lambda cls, model_config, cfg, use_separate_draft_kv_cache: CacheCost(
            slope=target_kv_per_token, intercept=0))
    c._kv_cache_manager_cls = Mock()
    c._model_engine = Mock()
    return c


@pytest.mark.gpu
def test_two_model_spec_decode_capacity_for_explicit_gpu_budget():
    B = 4 * GB
    # Target consumes 80% of the per-token bytes -> proportional split.
    creator = _make_creator(max_gpu_total_bytes=B, total_kv_per_token=100,
                             target_kv_per_token=80)

    target_cfg, draft_cfg = creator._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
    assert draft_cfg is not None, "split produced no draft config (unexpected for this input)"

    target_budget = target_cfg.max_gpu_total_bytes
    draft_budget = draft_cfg.max_gpu_total_bytes
    assert target_budget + draft_budget == B

    init_cuda_once()
    kv_buf_size = KV_BUF_SIZE

    target_manager = None
    draft_manager = None
    try:
        target_cfg_native = create_config(
            TOKENS_PER_BLOCK, target_budget, 0, 0, num_layers=4, window_size=None,
            sink_tokens=0, kv_buf_size=kv_buf_size)
        target_manager = KVCacheManager(target_cfg_native)

        draft_cfg_native = create_config(
            TOKENS_PER_BLOCK, draft_budget, 0, 0, num_layers=1, window_size=None,
            sink_tokens=0, kv_buf_size=kv_buf_size)
        draft_manager = KVCacheManager(draft_cfg_native)

        target_quota = target_manager.get_quota(GPU_LEVEL)
        draft_quota = draft_manager.get_quota(GPU_LEVEL)

        target_pool_groups = list(target_manager.pool_group_descs)
        draft_pool_groups = list(draft_manager.pool_group_descs)

        print(f"[Q1] B={B} target_budget={target_budget} draft_budget={draft_budget}")
        print(f"[Q1] target GPU quota (bytes)={target_quota}, "
              f"pool_group num_slots={[pg.num_slots for pg in target_pool_groups]}")
        print(f"[Q1] draft GPU quota (bytes)={draft_quota}, "
              f"pool_group num_slots={[pg.num_slots for pg in draft_pool_groups]}")

        assert target_quota > 0
        assert draft_quota > 0
        assert all(pg.num_slots > 0 for pg in target_pool_groups)
        assert all(pg.num_slots > 0 for pg in draft_pool_groups)
    finally:
        if target_manager is not None:
            target_manager.shutdown()
        if draft_manager is not None:
            draft_manager.shutdown()
