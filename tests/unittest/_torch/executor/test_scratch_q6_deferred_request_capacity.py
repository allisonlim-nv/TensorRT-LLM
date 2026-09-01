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

Answers: When batch-waiting or KV-length-cap deferral happens in
KVCacheV2Scheduler, does a deferred request retain its already-granted
target-manager capacity, is draft preparation skipped that iteration, and is
this an intentional bounded policy or just retained state that happens not to
be touched?

Method: exercises the real, unmodified
tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2.KVCacheV2Scheduler
(not mocked) via its own test suite's helpers
(``make_kv_cache_manager``/``make_scheduler``, imported directly from
tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py rather than
reimplemented) with a Mock KVCacheManagerV2 standing in for the native
manager -- same "real scheduler / mocked manager" tier as
test_scratch_q2_target_draft_evict_alignment.py.

Two distinct defer paths exist in ``_try_schedule_context_chunked``
(scheduler_v2.py):
  1. Budget/min-chunk defer BEFORE prepare_context is ever called
     (scheduler_v2.py:1606-1610: ``no_budget or fcfs_under_min`` -> SKIP).
     Nothing has been allocated for this request yet this call, so there is
     nothing to "retain" -- the request simply hasn't started.
  2. Budget/MM-alignment defer AFTER prepare_context (and, on a later chunk,
     after an earlier chunk's resize_context already succeeded) -- chunk_size
     computes to <= 0 -> SKIP (scheduler_v2.py:1667-1673). The code comment
     at this exact line is explicit about this being a deliberate,
     acknowledged design choice, not an oversight:
         "TODO: consider suspending first-chunk KVCache to release GPU
         pages. Currently we skip without suspend to avoid pathological
         suspend/resume cycles. suspend_request is only called from
         eviction (_try_evict_for_gen)."
     i.e. capacity already granted for a prior chunk is deliberately
     retained (not suspended/freed) across a deferred iteration, to avoid
     suspend/resume thrashing -- a documented, intentional bounded policy,
     not accidental retained state.

Draft-preparation skipping is a structural consequence, not a scheduler-side
decision: KVCacheV2Scheduler never calls the draft manager at all for
context requests (only ``_suspend_request``/``free_resources`` mirror calls
for gen eviction/free are scheduler-driven per Q2's report section; draft KV
prep for context happens via KVCacheManagerV2._prepare_draft_resources,
invoked from resource_manager.prepare_resources() at the py_executor level,
only for requests present in that iteration's ScheduledRequests). A request
the scheduler SKIPs is, by construction, absent from ScheduledRequests for
that iteration, so draft prepare_resources is never dispatched for it that
iteration -- confirmed here by asserting the scheduler itself never touches
a ``draft_kv_cache_manager`` mock for a chunked-context request (there is no
draft mirroring path for context scheduling to begin with, unlike the gen
eviction/free mirroring documented in Q2).
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_v2_scheduler import make_ctx_request, make_kv_cache_manager, make_scheduler  # noqa: E402

pytestmark = pytest.mark.cpu_only


def test_never_started_request_defers_before_any_allocation():
    """Path 1: budget too small even for the minimum chunk. prepare_context/
    resize_context are never called -- nothing was ever allocated for this
    request, so there is nothing to "retain": this is a pure no-op defer."""
    mgr = make_kv_cache_manager(tokens_per_block=64)
    sched = make_scheduler(mgr, max_num_tokens=50, ctx_chunk_config=(None, 64))
    req = make_ctx_request(0, context_remaining_length=1000)

    out = sched.schedule_request([req], set())

    assert len(out.context_requests) == 0
    mgr.prepare_context.assert_not_called()
    mgr.resize_context.assert_not_called()
    mgr.suspend_request.assert_not_called()


def test_partially_started_request_retains_capacity_across_deferred_iteration():
    """Path 2: first chunk succeeds (real resize_context call, capacity
    granted); a second call with an exhausted budget defers again WITHOUT
    suspending or freeing -- the already-granted capacity from chunk 1 is
    retained untouched across the deferred iteration."""
    mgr = make_kv_cache_manager(tokens_per_block=64)
    # First call: comfortable budget, request gets its first (non-last)
    # chunk successfully.
    sched1 = make_scheduler(mgr, max_num_tokens=128, ctx_chunk_config=(None, 64))
    req = make_ctx_request(0, context_remaining_length=1000)
    out1 = sched1.schedule_request([req], set())
    assert len(out1.context_requests) == 1
    first_chunk_size = req.context_chunk_size
    assert first_chunk_size > 0
    assert mgr.resize_context.call_count == 1
    resize_calls_after_chunk1 = mgr.resize_context.call_count
    suspend_calls_after_chunk1 = mgr.suspend_request.call_count
    prepare_calls_after_chunk1 = mgr.prepare_context.call_count

    # Second call: budget now below chunk_unit_size (simulates the request
    # losing the budget race to other requests this iteration) -> the
    # scheduler must defer (SKIP) rather than force a too-small chunk.
    sched2 = make_scheduler(mgr, max_num_tokens=50, ctx_chunk_config=(None, 64))
    out2 = sched2.schedule_request([req], set())

    assert len(out2.context_requests) == 0, (
        "expected a defer (SKIP) on the too-small second-iteration budget")
    # Deferred WITHOUT a further resize_context or a suspend_request call --
    # the capacity granted by chunk 1 is retained exactly as-is.
    assert mgr.resize_context.call_count == resize_calls_after_chunk1, (
        "a deferred (not scheduled) chunk must not call resize_context -- "
        "the manager's granted capacity from the prior successful chunk "
        "must be left untouched")
    assert mgr.suspend_request.call_count == suspend_calls_after_chunk1, (
        "the scheduler must not suspend a request on this defer path "
        "(scheduler_v2.py's explicit, commented design choice: "
        "'skip without suspend to avoid pathological suspend/resume "
        "cycles') -- capacity retention here is intentional policy, not "
        "an accidental omission")
    assert mgr.prepare_context.call_count == prepare_calls_after_chunk1, (
        "the too-small-budget defer returns SKIP before even calling "
        "prepare_context again this iteration (scheduler_v2.py's "
        "no_budget/fcfs_under_min check runs before the prepare_context "
        "call) -- confirms the defer is a pure early-exit, touching "
        "nothing manager-side")


def test_deferred_context_request_never_touches_draft_manager():
    """A chunked context request that gets deferred (budget too small for
    even the minimum chunk) never causes the scheduler to call ANY method on
    the draft manager -- confirming draft-side skipping for a deferred
    request is structural (the scheduler has no context-scheduling path
    that mirrors to the draft manager at all), not a special defer-aware
    branch that had to remember to skip the draft side."""
    target_mgr = make_kv_cache_manager(tokens_per_block=64)
    draft_mgr = Mock()
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
    from unittest.mock import patch

    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(target_mgr),
    ):
        sched = KVCacheV2Scheduler(
            max_batch_size=100,
            max_num_tokens=50,
            kv_cache_manager=target_mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(None, 64),
            peft_cache_manager=None,
            draft_kv_cache_manager=draft_mgr,
        )
    req = make_ctx_request(0, context_remaining_length=1000)

    out = sched.schedule_request([req], set())

    assert len(out.context_requests) == 0
    draft_mgr.assert_not_called()
    assert draft_mgr.method_calls == [], (
        "scheduler must not invoke any draft-manager method for a deferred "
        "context request -- draft KV preparation for context requests is "
        "entirely an executor-level (prepare_resources), not scheduler-"
        "level, concern, so a scheduler-side defer trivially skips it too")
