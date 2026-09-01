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

Strengthens the Q3 atomicity claim in scratchpad/kvcache_v2_runtime_results.md:
Q3 showed that after a failed native resize (real OutOfPagesError), the
request's own ``kv_cache.capacity`` counter is unchanged, and that a later
*fresh* sequence (after ``close()``) can still reach the manager's full
quota -- i.e. no leak, established only indirectly (by re-probing capacity
through a brand-new sequence after the original was torn down).

This probe adds a direct manager-level page/pool-accounting assertion,
queried immediately after the failed resize and BEFORE any close() of the
still-live request -- i.e. it inspects the manager's own view of how many
pages are currently held (``unavailable``) rather than inferring
leak-freedom indirectly through a subsequent fresh allocation. Uses
``KVCacheManager.get_and_reset_iteration_peak_block_stats(cache_level)``
(the same native binding surface backing
``KVCacheManagerV2.get_kv_cache_stats()`` in production), which reports
real, GPU-backed pool statistics (``available``/``unavailable``/
``evictable`` block counts), not simulated counters.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_manager_v2 import create_config  # noqa: E402

from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (  # noqa: E402
    CachedCudaStream,
    init_cuda_once,
)

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
def test_failed_resize_leaves_no_orphaned_pages_in_manager_pool_stats():
    init_cuda_once()

    gpu_quota = 8 * 1024 * 1024  # 8 MiB -> a small, real, finite native pool
    cfg = create_config(
        TOKENS_PER_BLOCK, gpu_quota, 0, 0, num_layers=1, window_size=None,
        sink_tokens=0, kv_buf_size=KV_BUF_SIZE)
    manager = KVCacheManager(cfg)
    stream_holder = CachedCudaStream()
    stream = stream_holder.handle
    kv_cache = None
    try:
        kv_cache = manager.create_kv_cache()
        kv_cache.resume(stream)

        first_chunk_capacity = TOKENS_PER_BLOCK * 2
        kv_cache.capacity = first_chunk_capacity

        # Baseline: query the manager's own committed-page view right
        # before the failed attempt (reset the peak-tracking window here so
        # the *next* query reflects exactly the interval spanning the
        # failed resize, and nothing before it).
        baseline = manager.get_and_reset_iteration_peak_block_stats(GPU_LEVEL)[0]
        print(f"[Atomicity] baseline before failed resize: "
              f"available={baseline.available} unavailable={baseline.unavailable} "
              f"evictable={baseline.evictable}")

        huge_capacity = TOKENS_PER_BLOCK * 100_000
        with pytest.raises(OutOfPagesErrors):
            kv_cache.capacity = huge_capacity

        # Query immediately after the failure -- kv_cache is still live
        # (not closed), so this is the manager's real-time page-accounting
        # view of a request whose most recent resize attempt failed.
        after_failure = manager.get_and_reset_iteration_peak_block_stats(GPU_LEVEL)[0]
        print(f"[Atomicity] immediately after failed resize: "
              f"available={after_failure.available} "
              f"unavailable={after_failure.unavailable} "
              f"evictable={after_failure.evictable}")

        # The manager's "unavailable" (currently held/committed) block count
        # must be identical before and after the failed resize -- if the
        # failed huge-capacity attempt had permanently grabbed any pages
        # before discovering it couldn't satisfy the full request (a
        # non-atomic partial commit), unavailable would have increased here.
        assert after_failure.unavailable == baseline.unavailable, (
            f"failed resize changed the manager's committed-page count "
            f"from {baseline.unavailable} to {after_failure.unavailable} "
            f"-- this would mean the failed OutOfPagesError resize "
            f"partially/non-atomically committed pages before failing")

        # A second, immediate no-op query (no intervening manager
        # operation) must report the same "unavailable" (committed) count --
        # rules out a delayed/async release that hadn't settled yet by the
        # first post-failure query.
        #
        # Note: "available" is a *peak* statistic over the interval since
        # the last reset (a high-water-mark of free blocks seen during that
        # window), not an instantaneous snapshot -- confirmed empirically:
        # the very first query (baseline, spanning manager construction
        # through the first successful chunk resize) reported available=512
        # even though only 510 blocks were actually free at that instant
        # (512 total - 2 held), because 512 was the peak free-block count
        # observed earlier in that same window, before the first chunk was
        # allocated. It is therefore not a reliable per-call "current free
        # count" signal and is deliberately not asserted on here; only
        # "unavailable" (currently committed/held blocks) is used as the
        # atomicity signal, since it tracks the manager's real-time
        # committed-block count.
        steady = manager.get_and_reset_iteration_peak_block_stats(GPU_LEVEL)[0]
        assert steady.unavailable == baseline.unavailable

        # Cross-check against Q3's indirect signal: the request is still
        # usable at its pre-failure capacity (no partial mutation of the
        # request's own state either).
        assert kv_cache.capacity == first_chunk_capacity

        kv_cache.close()
        kv_cache = None
    finally:
        if kv_cache is not None:
            kv_cache.close()
        manager.shutdown()
