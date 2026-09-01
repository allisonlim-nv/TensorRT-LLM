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

Answers: When a later chunk of context allocation fails, is the request
recoverable and are manager resources cleaned up consistently?

Method: uses the real native GPU-backed KVCacheManager/_KVCache (the same
bindings exercised by TestKVCacheManagerV2 in test_kv_cache_manager_v2.py).
Simulates chunked-context allocation as two sequential ``kv_cache.capacity =
N`` calls (this mirrors resize_context's underlying primitive at
tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py, which itself just
calls kv_cache.resize()/capacity=). First chunk succeeds with a small
capacity; second chunk requests far more capacity than remains free in the
manager's real GPU quota, forcing a real native OutOfPagesError.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_manager_v2 import create_config  # noqa: E402

from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (  # noqa: E402
    CachedCudaStream,
    init_cuda_once,
)

# The cpp-backend KVCacheManager (default backend) raises the *native binding's*
# OutOfPagesError, which is a different class object than
# tensorrt_llm.runtime.kv_cache_manager_v2._exceptions.OutOfPagesError (confirmed
# by direct identity check: they are not the same class). Catch both so this
# test works under either backend.
try:
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import (
        OutOfPagesError as _NativeOutOfPagesError,
    )
except ImportError:
    _NativeOutOfPagesError = ()
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import (
    OutOfPagesError as _PyOutOfPagesError,
)

OutOfPagesErrors = (_PyOutOfPagesError,) if _NativeOutOfPagesError == () else (
    _PyOutOfPagesError, _NativeOutOfPagesError)

TOKENS_PER_BLOCK = 32
KV_BUF_SIZE = 8192


@pytest.mark.gpu
def test_second_chunk_alloc_failure_is_recoverable_and_leak_free():
    init_cuda_once()

    # Small GPU quota -> few real pages, so a large second-chunk request
    # genuinely cannot be satisfied (real OutOfPagesError, not simulated).
    gpu_quota = 8 * 1024 * 1024  # 8 MiB
    cfg = create_config(
        TOKENS_PER_BLOCK, gpu_quota, 0, 0, num_layers=1, window_size=None,
        sink_tokens=0, kv_buf_size=KV_BUF_SIZE)
    manager = KVCacheManager(cfg)
    stream_holder = CachedCudaStream()
    stream = stream_holder.handle
    try:
        kv_cache = manager.create_kv_cache()
        # A CUDA-stream resume is a real production precondition: every
        # KvCache is resumed onto a stream before use (see
        # TestKVCacheManagerV2's own harness in test_kv_cache_manager_v2.py).
        # Skipping it left mFinishEvent unset and made close() abort the
        # whole process (std::bad_optional_access in SharedPageLock::unlock,
        # via KvCache::finishEvent()->mFinishEvent.value()) -- confirmed to
        # be a missing-precondition artifact of the minimal repro, not a
        # real KVCacheManagerV2 bug: with resume() included (below), the
        # exact same failure-then-close sequence completes cleanly.
        kv_cache.resume(stream)

        first_chunk_capacity = TOKENS_PER_BLOCK * 2  # 2 blocks: small, must fit
        kv_cache.capacity = first_chunk_capacity
        assert kv_cache.capacity == first_chunk_capacity
        print(f"[Q3] first chunk ok: capacity={kv_cache.capacity}")

        # Second "chunk": request far more than remaining free pages allow.
        huge_capacity = TOKENS_PER_BLOCK * 100_000
        with pytest.raises(OutOfPagesErrors):
            kv_cache.capacity = huge_capacity
        print("[Q3] second chunk correctly raised OutOfPagesError")

        # 1) Request recoverable: kv_cache object is still usable at its
        # last successful capacity (not corrupted, not force-closed).
        assert kv_cache.capacity == first_chunk_capacity, (
            "capacity changed after a failed resize (partial mutation / "
            "not rolled back)")

        # It must still be usable: growing back to a capacity that fits
        # must succeed (proves the manager's internal state is consistent,
        # not wedged, after the failed attempt).
        kv_cache.capacity = TOKENS_PER_BLOCK * 3
        assert kv_cache.capacity == TOKENS_PER_BLOCK * 3
        print(f"[Q3] request still usable post-failure: "
              f"capacity grew to {kv_cache.capacity}")

        kv_cache.close()

        # 2) Manager resources cleaned up consistently: after closing the
        # first sequence, a *fresh* sequence can allocate up to the full
        # original quota again -- i.e. the failed huge-capacity attempt did
        # not leak/strand any pages.
        kv_cache2 = manager.create_kv_cache()
        kv_cache2.resume(stream)
        max_blocks_estimate = gpu_quota // (KV_BUF_SIZE * 2 * TOKENS_PER_BLOCK)
        kv_cache2.capacity = max_blocks_estimate * TOKENS_PER_BLOCK
        print(f"[Q3] post-cleanup fresh alloc reached capacity="
              f"{kv_cache2.capacity} (no leak from failed resize)")
        kv_cache2.close()
    finally:
        manager.shutdown()
