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

Answers: When context-allocation revert (KVCacheManagerV2.revert_allocate_context)
is exercised after context history has already advanced, does it shrink the
request back to its prior capacity, or free the request completely -- and do
callers correctly tolerate whichever happens?

Method: exercises the real, unmodified
tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2.revert_allocate_context
method body (not reimplemented/duplicated) against a MagicMock ``_KVCache``
standing in for the native cache object -- the same "real manager
method, minimal-attribute manager instance via __new__, mocked native cache"
technique already used by
tests/unittest/_torch/executor/test_kv_cache_v2_capacity_only.py's
``_manager``/``_cache`` helpers (reused directly, not reimplemented) for the
adjacent ``update_resources`` method on this same class.

revert_allocate_context's actual (real, source-read) branching, from
kv_cache_manager_v2.py:2525-2547:
    pre_cap = req.py_ctx_pre_resize_cap  (None -> no-op, already reverted/never grown)
    if pre_cap >= kv_cache.capacity: return  (no growth to undo)
    if kv_cache.history_length > pre_cap:
        self.free_resources(req)   # <-- FREE branch: committed history has
                                    #     already advanced past the shrink
                                    #     target, so shrinking would leave
                                    #     committed/history state inconsistent
                                    #     with capacity -- the manager frees
                                    #     the whole request instead.
        return
    history_length = min(kv_cache.history_length, pre_cap)
    kv_cache.resize(pre_cap, history_length)   # <-- SHRINK branch
    if pre_cap > 0:
        kv_cache.suspend()

py_executor.py's only caller, ``_revert_ctx_alloc`` (py_executor.py:3474-3477,
called from ``_revert_deferred_disagg_gen_init_alloc`` at py_executor.py:3588
for candidates dropped by disagg-transfer admission), does not branch on
which of the two outcomes occurred and does not re-check kv_cache_map
membership afterward -- it is a blind for-loop calling
``revert_allocate_context(req)`` once per dropped request. Whether that
caller "correctly tolerates" the FREE branch is therefore a structural
question about what happens on the *next* scheduling attempt for that same
request, not something the caller itself inspects: the request is re-offered
to the scheduler next iteration in CONTEXT_INIT state either way, and
KVCacheManagerV2.prepare_context's real precondition
(kv_cache_map.get(...) is None triggers fresh _create_kv_cache, per
_prepare_context_impl / manager.md's Create section) already tolerates a
missing kv_cache_map entry -- so a fully-freed request is not a caller-side
crash risk, only a "start over from scratch, no reuse credit retained"
outcome relative to the SHRINK branch, which explicitly preserves the
kv_cache_map entry (still active, at pre_cap capacity, suspended).
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def _manager() -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = False
    manager.kv_cache_map = {}
    return manager


def _cache(*, capacity: int, history_length: int, active: bool = True) -> MagicMock:
    cache = MagicMock()
    cache.capacity = capacity
    cache.history_length = history_length
    cache.is_active = active
    cache.resize.return_value = True
    return cache


class _Req:
    def __init__(self, request_id: int, pre_resize_cap):
        self.py_request_id = request_id
        self.py_ctx_pre_resize_cap = pre_resize_cap


def test_revert_no_growth_to_undo_is_a_no_op():
    """pre_cap >= current capacity: nothing grew this iteration, no-op."""
    manager = _manager()
    manager.free_resources = MagicMock()
    cache = _cache(capacity=100, history_length=100)
    manager.kv_cache_map[1] = cache
    req = _Req(1, pre_resize_cap=100)

    manager.revert_allocate_context(req)

    cache.resize.assert_not_called()
    cache.suspend.assert_not_called()
    manager.free_resources.assert_not_called()
    assert manager.kv_cache_map[1] is cache, "no-op case must not evict the map entry"


def test_revert_shrinks_when_history_has_not_passed_pre_cap():
    """history_length <= pre_cap: SHRINK branch. Request stays alive, at
    pre_cap capacity, suspended -- kv_cache_map entry is retained
    unchanged."""
    manager = _manager()
    manager.free_resources = MagicMock()
    # Grew from pre_cap=64 to 128 this iteration; history only advanced to 50
    # (<= pre_cap=64), so shrinking back to 64 does not truncate committed
    # history.
    cache = _cache(capacity=128, history_length=50)
    manager.kv_cache_map[1] = cache
    req = _Req(1, pre_resize_cap=64)

    manager.revert_allocate_context(req)

    cache.resize.assert_called_once_with(64, 50)
    cache.suspend.assert_called_once()
    manager.free_resources.assert_not_called()
    assert manager.kv_cache_map[1] is cache, (
        "SHRINK branch must leave the request's kv_cache_map entry in place "
        "(same object, still active) -- the request is recoverable, not freed")
    assert req.py_ctx_pre_resize_cap is None, (
        "revert must clear the pre-resize marker so a second revert call "
        "(e.g. a duplicate/retry) is a no-op, not a double-shrink")


def test_revert_frees_completely_when_history_has_passed_pre_cap():
    """history_length > pre_cap: committed history already advanced past
    where we'd need to shrink to -- shrinking would leave capacity < history
    (an inconsistent state), so the manager frees the whole request instead
    of a partial shrink."""
    manager = _manager()
    manager.free_resources = MagicMock()
    # Grew from pre_cap=64 to 128; history advanced to 90 (> pre_cap=64) --
    # e.g. a chunked-context request whose committed/history position moved
    # past the pre-iteration capacity before the revert was requested.
    cache = _cache(capacity=128, history_length=90)
    manager.kv_cache_map[1] = cache
    req = _Req(1, pre_resize_cap=64)

    manager.revert_allocate_context(req)

    cache.resize.assert_not_called()
    cache.suspend.assert_not_called()
    manager.free_resources.assert_called_once_with(req)
    assert req.py_ctx_pre_resize_cap is None


def test_revert_on_already_inactive_cache_is_a_no_op():
    """kv_cache.is_active is False (already suspended/freed by something
    else this iteration): revert_allocate_context must not touch the cache --
    no resize/suspend/free_resources call. The pre-resize marker IS still
    cleared (source: kv_cache_manager_v2.py:2530, `req.py_ctx_pre_resize_cap
    = None` runs unconditionally as soon as pre_cap is not None, *before*
    the `kv_cache is None or not kv_cache.is_active` early-return at
    :2531-2533) -- so a second revert call on the same request is a
    guaranteed no-op regardless of which branch the first call took."""
    manager = _manager()
    manager.free_resources = MagicMock()
    cache = _cache(capacity=128, history_length=50, active=False)
    manager.kv_cache_map[1] = cache
    req = _Req(1, pre_resize_cap=64)

    manager.revert_allocate_context(req)

    cache.resize.assert_not_called()
    cache.suspend.assert_not_called()
    manager.free_resources.assert_not_called()
    assert req.py_ctx_pre_resize_cap is None
