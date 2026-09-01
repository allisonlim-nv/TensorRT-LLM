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

Answers: When generation allocation fails and scheduler eviction/suspension
occurs, are target and draft request states always aligned afterward?

Method: exercises the real, unmodified tensorrt_llm._torch.pyexecutor.
scheduler.scheduler_v2.KVCacheV2Scheduler (not mocked) with two Mock
KVCacheManagerV2-like managers standing in for the target and draft
managers -- the same fault-injection style already used by
tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py (which is
itself a Tier-1, mocked-manager, no-GPU suite explicitly documented as such
in that file's own docstring). This is a real-scheduler /
mocked-native-backend "unit/fault-injection" test per the coverage_closure.md
taxonomy in scratchpad/kvcachev2_context/, not a native/GPU proof.

scheduler_v2.py's _suspend_request (used by both self-eviction and
_try_evict_for_gen) does:
    self.kv_cache_manager.suspend_request(req)
    if self.draft_kv_cache_manager is not None:
        self.draft_kv_cache_manager.suspend_request(req)
sequentially and unguarded -- no try/except around either call. This test
makes the *draft* manager's suspend_request raise, and inspects what state
the *target* manager (and the request) are left in.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kv_cache_v2_scheduler import make_gen_request  # noqa: E402

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState  # noqa: E402
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler  # noqa: E402
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _KVCacheMap(dict):
    def __missing__(self, key):
        entry = SimpleNamespace(is_active=True)
        self[key] = entry
        return entry


def _make_manager(try_allocate_generation_fn, suspend_side_effect=None):
    mgr = Mock()
    mgr.tokens_per_block = 64
    mgr.can_evict = True
    mgr._has_cp_helix = False
    mgr.kv_cache_map = _KVCacheMap()
    mgr.prepare_context.side_effect = lambda req: True
    mgr.resize_context.side_effect = lambda req, n: True
    mgr.prepare_disagg_gen_init.side_effect = lambda req: True
    mgr.try_allocate_generation.side_effect = try_allocate_generation_fn

    def suspend_request(req):
        if suspend_side_effect is not None:
            suspend_side_effect(req)
        mgr.kv_cache_map[req.py_request_id].is_active = False

    mgr.suspend_request.side_effect = suspend_request
    mgr.is_request_active.side_effect = lambda req_id: mgr.kv_cache_map[req_id].is_active
    return mgr


def test_draft_suspend_failure_leaves_target_and_draft_misaligned():
    """Target suspends successfully; the mirrored draft suspend raises.

    Confirms the two managers are NOT always aligned afterward: this is a
    real, unmodified scheduler code path (scheduler_v2.py's _suspend_request,
    called from _try_evict_for_gen) with no exception handling around the
    draft mirror call, so the target manager ends up suspended while the
    draft manager's suspend never completed -- and the exception propagates
    out of schedule_request uncaught.
    """
    call_count = [0]

    def alloc_fn(req):
        call_count[0] += 1
        return call_count[0] != 1  # first alloc fails (triggers eviction), retry succeeds

    target_mgr = _make_manager(alloc_fn)

    def draft_suspend_raises(req):
        raise RuntimeError(f"draft suspend_request failed for req {req.py_request_id}")

    draft_mgr = _make_manager(lambda req: True, suspend_side_effect=draft_suspend_raises)

    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(target_mgr),
    ):
        sched = KVCacheV2Scheduler(
            max_batch_size=100,
            max_num_tokens=100,
            kv_cache_manager=target_mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            peft_cache_manager=None,
            draft_kv_cache_manager=draft_mgr,
        )

    victim = make_gen_request(99)  # gen in progress -> evictable
    reqs = [make_gen_request(0), victim]

    with pytest.raises(RuntimeError, match="draft suspend_request failed"):
        sched.schedule_request(reqs, set())

    # Target manager's suspend_request *did* complete for the victim (called
    # first, before the draft mirror raised) -- state divergence confirmed:
    target_mgr.suspend_request.assert_called_once_with(victim)
    assert target_mgr.is_request_active(victim.py_request_id) is False, (
        "target manager should show the victim suspended (its call ran "
        "and returned before the draft mirror call raised)")

    # Draft manager's suspend_request was invoked (that's what raised) but
    # its state-mutation line (`is_active = False`) never executed because
    # the side effect callback raises *before* reaching it in our test
    # manager -- mirroring a real native suspend_request() that raises
    # mid-operation and leaves the draft KV cache's active status
    # unresolved/unchanged rather than suspended.
    draft_mgr.suspend_request.assert_called_once_with(victim)
    assert draft_mgr.is_request_active(victim.py_request_id) is True, (
        "draft manager never actually completed suspension for the victim")

    print("[Q2] CONFIRMED: after a failed mirrored suspend, target manager "
          "shows the request SUSPENDED while draft manager still shows it "
          "ACTIVE -- states are misaligned, and the RuntimeError propagates "
          "uncaught out of schedule_request (no rollback of the target "
          "manager's already-completed suspend).")
