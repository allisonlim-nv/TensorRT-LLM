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
"""TRTLLM-15289 Q4 (strengthened): real `try_allocate_generation` for the
disagg transmission-complete / empty-draft-token admission case.

The existing `test_kv_cache_v2_capacity_only.py::
test_disagg_gen_transition_reserves_target_drafts_without_context_drafts`
only calls `_effective_draft_len`/`_required_gen_capacity` in isolation on a
bare `SimpleNamespace` request -- it proves the *arithmetic*, not that the
resulting number actually costs real GPU pages via `try_allocate_generation`.

This probe drives the real, unmodified `KVCacheManagerV2.try_allocate_generation`
against a real GPU-backed native cache, and empirically demonstrates that
`_effective_draft_len`'s reservation is not inert bookkeeping: with a small,
fixed real page pool, admission that would succeed with speculative
decoding disabled (draft_len=0) is REJECTED with a real OutOfPagesError when
enabled (draft_len=max_total_draft_tokens=4) for the exact same request and
pool -- because the extra reserved tokens push the resize target across a
real block boundary.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_manager_v2 import _ContextRequest  # noqa: E402

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2  # noqa: E402
from tensorrt_llm.bindings import DataType  # noqa: E402
from tensorrt_llm.bindings.internal.batch_manager import CacheType  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once  # noqa: E402

TOKENS_PER_BLOCK = 4
MAX_SEQ_LEN = 32
# Exactly 2 blocks (8 tokens) of real GPU pages.
GPU_QUOTA_BYTES = 8 << 20


def _make_manager() -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=GPU_QUOTA_BYTES,
            max_attention_window=[MAX_SEQ_LEN, TOKENS_PER_BLOCK],
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        enable_stats=False,
    )


def _prime_active_context_cache(manager: KVCacheManagerV2, req_id: int, capacity: int):
    """Create a real, active native cache at `capacity` tokens via the real
    context path (prepare_context + resize_context), the same technique
    `_run_context` in test_kv_cache_manager_v2.py already uses."""
    ctx_req = _ContextRequest(req_id, list(range(capacity)), capacity, f"conv-{req_id}",
                              use_conversation_params=False)
    assert manager.prepare_context(ctx_req)
    assert manager.resize_context(ctx_req, num_tokens=capacity)
    return manager.kv_cache_map[req_id]


def _gen_request(req_id: int, *, disable_speculative: bool):
    return SimpleNamespace(
        py_request_id=req_id,
        py_draft_tokens=[],  # empty -- draft_len must come from _effective_draft_len's fallback
        is_disagg_generation_transmission_complete=True,
        context_phase_params=SimpleNamespace(draft_tokens=None),
        py_disable_speculative_decoding=disable_speculative,
        is_dummy_request=False,
    )


@pytest.mark.gpu
def test_disagg_transmission_complete_draft_reservation_costs_real_pages():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    init_cuda_once()

    # --- Case A: speculative decoding disabled -> draft_len=0. Growing a
    # 4-token (1-block) cache by 1 needs capacity 5 -> still fits in 2 blocks.
    manager_a = _make_manager()
    try:
        manager_a.is_draft = False
        manager_a.max_total_draft_tokens = 4
        manager_a._has_cp_helix = False
        manager_a._allocated_draft_lens = {}
        cache_a = _prime_active_context_cache(manager_a, 1, capacity=4)
        req_a = _gen_request(1, disable_speculative=True)

        assert manager_a._effective_draft_len(req_a) == 0
        ok = manager_a.try_allocate_generation(req_a)
        print(f"[Q4-real] disabled-speculation admission: ok={ok} capacity={cache_a.capacity}")
        assert ok is True
        assert cache_a.capacity == 5  # 4 + 1 + 0
    finally:
        manager_a.free_resources(_ContextRequest(1, [], 0, "conv-1", use_conversation_params=False))
        manager_a.shutdown()

    # --- Case B: speculative decoding enabled, no context draft tokens ->
    # _effective_draft_len falls back to max_total_draft_tokens=4. Growing
    # the SAME 4-token (1-block) cache now targets capacity 9 -> needs 3
    # blocks, exceeding the 2-block pool -> real OutOfPagesError rejection.
    manager_b = _make_manager()
    try:
        manager_b.is_draft = False
        manager_b.max_total_draft_tokens = 4
        manager_b._has_cp_helix = False
        manager_b._allocated_draft_lens = {}
        cache_b = _prime_active_context_cache(manager_b, 2, capacity=4)
        req_b = _gen_request(2, disable_speculative=False)

        assert manager_b._effective_draft_len(req_b) == 4
        ok = manager_b.try_allocate_generation(req_b)
        print(f"[Q4-real] enabled-speculation admission: ok={ok} capacity={cache_b.capacity}")

        # The draft reservation is real, not inert: it turns an admission
        # that would otherwise succeed (case A, same starting capacity,
        # same pool) into a real rejection.
        assert ok is False
        # Failure is atomic (consistent with Q3): capacity is unchanged,
        # still the pre-attempt 4 tokens, not partially grown toward 9.
        assert cache_b.capacity == 4
        assert cache_b.is_active is True
    finally:
        manager_b.free_resources(_ContextRequest(2, [], 0, "conv-2", use_conversation_params=False))
        manager_b.shutdown()
