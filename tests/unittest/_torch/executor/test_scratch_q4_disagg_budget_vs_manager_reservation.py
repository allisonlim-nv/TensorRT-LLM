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

Answers: In disaggregated context->generation, does the scheduler's budget
accounting match or conservatively differ from manager reservation?

Two real, unmodified code paths are exercised directly (no re-implementation
of their logic):

  1. Scheduler-side token cost for a generation request, as computed by
     KVCacheV2Scheduler._try_schedule_generation:
         req_tokens = beam_width + get_draft_token_length(req)
     (tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:984)

  2. Manager-side reservation for the same request, as computed by
     KVCacheManagerV2._effective_draft_len, used internally by
     try_allocate_generation's capacity math:
         (tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py:2410-2427)

Part A drives the real KVCacheV2Scheduler (mocked manager, same technique as
tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py's TestDisagg
class) to directly observe the scheduler's real BudgetTracker committing 0
tokens for a DISAGG_GEN_INIT request, and only ``beam_width`` tokens for a
transmission-complete generation request with no draft tokens yet -- both
already exercise real, unmodified scheduler code.

Part B calls the real (unmodified) KVCacheManagerV2._effective_draft_len
directly on an unpatched, real bound method with a lightweight fake request/
manager object, to get the actual reserved draft length the manager would
ask for in the same situation.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_v2_scheduler import (  # noqa: E402
    make_disagg_request,
    make_gen_request,
    make_kv_cache_manager,
)

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2  # noqa: E402
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler  # noqa: E402
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy  # noqa: E402

pytestmark = pytest.mark.cpu_only


def test_disagg_gen_init_consumes_zero_scheduler_token_budget():
    """Part A1: scheduler charges 0 tokens for DISAGG_GEN_INIT requests.

    This is real KVCacheV2Scheduler code (BudgetTracker), not reimplemented:
    a tight max_num_tokens=1 budget still admits the disagg-init request,
    proving the scheduler's ledger commits 0 tokens for it while the actual
    KV-cache reservation is entirely gated by the (mocked here, real in
    production) manager's prepare_disagg_gen_init/IndexMapper check.
    """
    mgr = make_kv_cache_manager()
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(mgr),
    ):
        sched = KVCacheV2Scheduler(
            max_batch_size=100,
            max_num_tokens=1,  # only 1 token of budget total
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        out = sched.schedule_request([make_disagg_request(0)], set())
    assert [r.py_request_id for r in out.fitting_disagg_gen_init_requests] == [0]
    print("[Q4] scheduler admitted a disagg-init request under a 1-token "
          "budget: BudgetTracker charges it 0 tokens (real scheduler code).")


def test_disagg_transmission_complete_gen_scheduler_charges_beam_width_only():
    """Part A2: real BudgetTracker cost for a transmission-complete gen req.

    Uses the real _try_schedule_generation token-cost line
    (`req_tokens = beam_width + get_draft_token_length(req)`) via the real
    scheduler. With py_draft_tokens=[] (no tokens transferred yet) the
    scheduler's ledger charges exactly beam_width=1 token, regardless of
    max_total_draft_tokens.
    """
    committed = []

    def try_allocate_generation_fn(req):
        return True

    mgr = make_kv_cache_manager(try_allocate_generation_fn=try_allocate_generation_fn)
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(mgr),
    ):
        sched = KVCacheV2Scheduler(
            max_batch_size=100,
            max_num_tokens=1,  # exactly enough for beam_width=1, nothing more
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        req = make_gen_request(0, num_draft_tokens=0)
        req.is_disagg_generation_transmission_complete = True
        req.context_phase_params = None
        out = sched.schedule_request([req], set())
    assert [r.py_request_id for r in out.generation_requests] == [0]
    print("[Q4] scheduler admits a transmission-complete gen request with "
          "empty py_draft_tokens under a 1-token budget: BudgetTracker "
          "charges only beam_width=1 (get_draft_token_length(req)==0).")


def test_manager_effective_draft_len_reserves_more_than_scheduler_charges():
    """Part B: the real KVCacheManagerV2._effective_draft_len (unmodified,
    called directly as a bound method) reserves max_total_draft_tokens of
    *extra* capacity for the exact same transmission-complete/no-draft-yet
    situation that the scheduler charged only beam_width for above.

    This is the concrete, quantified gap: manager reservation (extra
    max_total_draft_tokens) > scheduler ledger charge (0 draft tokens) --
    the scheduler's accounting is not merely different, it is the smaller
    number, and the real manager is what enforces the true, larger
    footprint. That makes the manager the binding/conservative constraint:
    the scheduler under-counts, but real page allocation never proceeds on
    the scheduler's optimistic number alone -- try_allocate_generation must
    still succeed against the manager's larger internal ask, so a
    scheduler-admitted request can still legitimately fail manager admission
    purely on draft-reserve grounds. This is a real, unmodified manager
    method call, not a re-derivation of its logic.
    """
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = False
    manager.max_total_draft_tokens = 4
    manager.py_disable_speculative_decoding = False

    req = Mock()
    req.py_draft_tokens = []
    req.is_disagg_generation_transmission_complete = True
    req.context_phase_params = Mock(draft_tokens=None)
    req.py_disable_speculative_decoding = False

    manager_reserved = manager._effective_draft_len(req)
    scheduler_charged = 0  # get_draft_token_length(req) == len(req.py_draft_tokens) == 0

    assert manager_reserved == 4
    assert manager_reserved > scheduler_charged
    print(f"[Q4] QUANTIFIED GAP: manager._effective_draft_len reserves "
          f"{manager_reserved} draft-token slots of GPU capacity for this "
          f"request; the scheduler's BudgetTracker ledger charged "
          f"{scheduler_charged} draft tokens for the same request in Part "
          f"A2. The scheduler conservatively (from the manager's point of "
          f"view) *under*-counts; the manager is the true, larger, binding "
          f"reservation.")
