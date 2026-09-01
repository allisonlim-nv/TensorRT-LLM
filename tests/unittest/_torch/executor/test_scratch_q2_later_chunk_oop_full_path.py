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
"""TRTLLM-15289 Q2 (strengthened): a later-chunk (non-first-chunk) real
OutOfPagesError driven through the real `KVCacheV2Scheduler` scheduling
loop AND the real `KVCacheManagerV2.prepare_context`/`resize_context`
against a real, GPU-backed native cache -- not a mocked manager, and not a
direct `kv_cache.capacity` mutation on a bare native object (that is the
existing, separately-preserved Q3/atomicity probe, which is a first-chunk-
style *direct* native resize failure with no scheduler involved).

This complements, not replaces, Q3/atomicity
(test_scratch_atomicity_failed_resize_page_accounting.py): that probe
proves the *native* resize call itself is atomic on failure. This probe
proves what the *scheduler + Python manager* do around a failure that
happens on a *second* (non-first) chunk of a real multi-chunk context
request already holding committed capacity from an earlier, successful
chunk.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_manager_v2 import _ContextRequest  # noqa: E402

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import GPU_LEVEL, KVCacheManagerV2  # noqa: E402
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState  # noqa: E402
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler  # noqa: E402
from tensorrt_llm.bindings import DataType  # noqa: E402
from tensorrt_llm.bindings.internal.batch_manager import CacheType  # noqa: E402
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once  # noqa: E402

TOKENS_PER_BLOCK = 4
# Small quota: exactly 4 blocks (16 tokens) worth of real GPU pages.
GPU_QUOTA_BYTES = 16 << 20
MAX_SEQ_LEN = 64  # generous vs. the pool, so the failure is page exhaustion, not a seq-len cap


def _augment_for_scheduler(req: _ContextRequest) -> _ContextRequest:
    """`_ContextRequest` already satisfies the real manager's
    prepare_context/resize_context (proven by this branch's own
    `_run_context` helper). Add the extra attributes the real *scheduler*
    reads (`_try_schedule_context_chunked`, `_try_schedule_cross_context`),
    mirroring `test_kv_cache_v2_scheduler.make_ctx_request`.
    """
    req.state_value = LlmRequestState.CONTEXT_INIT.value
    req.expect_snapshot_points = []
    req.num_draft_tokens = 0
    req.has_draft_tokens = False
    req.py_draft_tokens = []
    req.context_chunk_size = 0
    req.is_context_init_state = True
    req.is_generation_in_progress_state = False
    req.encoder_output_len = None
    req.py_encoder_output_ready_event = None
    req.py_skip_cross_kv_projection = False
    req.request_id = req.py_request_id
    return req


@pytest.mark.gpu
def test_later_chunk_out_of_pages_through_real_scheduler_and_manager():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    init_cuda_once()

    manager = KVCacheManagerV2(
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
    try:
        req = _augment_for_scheduler(
            _ContextRequest(1, list(range(24)), 24, "conv-q2", use_conversation_params=False)
        )

        # --- Chunk 1: budget=8, chunk_unit=8 -> exactly 8 tokens, fits well
        # inside the 16-token (4-block) pool. Real scheduler + real manager.
        sched1 = KVCacheV2Scheduler(
            max_batch_size=10,
            max_num_tokens=8,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(None, 8),
        )
        req.is_first_context_chunk = True
        req.is_last_context_chunk = False
        out1 = sched1.schedule_request([req], set())
        assert len(out1.context_requests) == 1, "chunk 1 must schedule successfully"
        assert manager.kv_cache_map[req.py_request_id].capacity == 8
        # The scheduler itself does not advance context_current_position --
        # that is production's post-forward-pass executor bookkeeping
        # (mirrors this branch's own `_run_context` helper in
        # test_kv_cache_manager_v2.py). Apply it manually here, exactly as
        # that helper does, so the *second* prepare_context call sees a
        # correct is_first_context_chunk=False state.
        req.context_current_position = 8
        req.context_remaining_length = 16

        pre_cap_after_chunk1 = req.py_ctx_pre_resize_cap
        baseline_stats = manager._get_and_reset_iteration_peak_block_stats(GPU_LEVEL)[0]

        # --- Chunk 2: request all 16 remaining tokens. Combined with chunk
        # 1's 8, this targets capacity 24 -- 6 blocks -- against a pool that
        # only has 4 blocks total. This must fail with a REAL native
        # OutOfPagesError surfacing through resize_context(), reached via
        # the real scheduler's chunked context path (not a direct
        # kv_cache.capacity mutation).
        req.is_first_context_chunk = False
        sched2 = KVCacheV2Scheduler(
            max_batch_size=10,
            max_num_tokens=16,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(None, 16),
        )
        out2 = sched2.schedule_request([req], set())

        print(f"[Q2-full] chunk2 scheduled context_requests={len(out2.context_requests)}")
        assert len(out2.context_requests) == 0, (
            "chunk 2 must NOT schedule -- it should fail via a real "
            "OutOfPagesError inside resize_context()"
        )

        after_failure_stats = manager._get_and_reset_iteration_peak_block_stats(GPU_LEVEL)[0]
        kv_cache = manager.kv_cache_map[req.py_request_id]

        print(
            f"[Q2-full] request state after failed later-chunk resize: "
            f"capacity={kv_cache.capacity} is_active={kv_cache.is_active} "
            f"context_current_position={req.context_current_position} "
            f"py_ctx_pre_resize_cap={req.py_ctx_pre_resize_cap} "
            f"(was {pre_cap_after_chunk1} after chunk 1)"
        )
        print(
            f"[Q2-full] manager pool stats: baseline unavailable="
            f"{baseline_stats.unavailable} after_failure unavailable="
            f"{after_failure_stats.unavailable}"
        )

        # 1. The request's own committed capacity is unchanged by the
        #    failed later-chunk resize -- same atomicity property Q3
        #    established directly on the native object, now confirmed
        #    reached via the real scheduler + Python manager call chain.
        assert kv_cache.capacity == 8

        # 2. The manager's real committed-page count is unchanged too (no
        #    partial commit leaked by the failed later-chunk attempt).
        assert after_failure_stats.unavailable == baseline_stats.unavailable

        # 3. Non-first-chunk asymmetry: `resize_context`
        #    (kv_cache_manager_v2.py:2671-2674) only suspends the cache on
        #    failure when `req.is_first_context_chunk` is True. This is
        #    chunk 2 (is_first_context_chunk=False), so the cache must
        #    remain ACTIVE after the failure -- unlike a first-chunk
        #    failure, which suspends. This is a real behavioral asymmetry
        #    between first-chunk and later-chunk resize failures.
        assert kv_cache.is_active is True

        # 4. py_ctx_pre_resize_cap is untouched by the failed call (resize_context
        #    only writes it on success); it still reflects chunk 1's grow.
        assert req.py_ctx_pre_resize_cap == pre_cap_after_chunk1

        # 5. Retry behavior: the request is still schedulable at its
        #    pre-failure (chunk-1) capacity/position -- nothing about the
        #    failed attempt corrupted state such that a subsequent, smaller
        #    chunk request can't proceed. Retry with a chunk that actually
        #    fits in the remaining pool (1 more block = 4 tokens).
        sched3 = KVCacheV2Scheduler(
            max_batch_size=10,
            max_num_tokens=4,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(None, 4),
        )
        out3 = sched3.schedule_request([req], set())
        assert len(out3.context_requests) == 1, (
            "a smaller retry chunk that fits the remaining pool must "
            "succeed after the earlier failed larger chunk -- confirms the "
            "failed attempt did not corrupt or wedge the request"
        )
        assert kv_cache.capacity == 12
    finally:
        if req.py_request_id in manager.kv_cache_map:
            manager.free_resources(req)
        manager.shutdown()
