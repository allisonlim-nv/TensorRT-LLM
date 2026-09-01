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
"""TRTLLM-15289 Q1 (strengthened, no forced flag): drives the REAL,
unmodified V2 GPU-budget gating chain in ``KvCacheCreator`` (``_util.py``)
with real, default speculative-decoding configs -- not a forced boolean.

Central finding this probe establishes, which reframes the original Q1
question: **genuine two-engine ("two-model", two separate forward-pass
engines) speculative decoding is not reachable through any currently
supported public LLM API config**:

  - EAGLE3 two-model (`eagle3_one_model=False`) is deprecated and silently
    coerced back to one-model by
    `EagleDecodingConfig.validate_eagle_config` (llm_args.py:2228-2233) --
    confirmed below by constructing the config and reading back
    `spec_dec_mode`.
  - `DraftTargetDecodingConfig` has a private `_draft_target_one_model`
    attribute defaulting to `True` (llm_args.py:2592) with no field/setter
    anywhere in this codebase that ever sets it `False` (grepped: the only
    two references are the attribute's own default and the property that
    reads it) -- so `DraftTargetDecodingConfig(...)`'s default
    `spec_dec_mode` is always `DRAFT_TARGET_ONE_MODEL`, not the genuine
    two-engine `DRAFT_TARGET`.
  - A companion GPU-runtime attempt
    (`test_scratch_q1_v2_two_model_real_budget.py`, in
    tests/unittest/_torch/speculative/hw_agnostic/) tried to force this
    through anyway via `LLM(...)` construction; it hit the same
    environment-level missing-SM-90-kernel-image blocker as Q8 during
    warmup/generation, and separately could not observe manager
    construction args at all because `KVCacheManagerV2.__init__` runs
    inside the MPI/IPC executor's worker subprocess, not the test process
    -- both are documented there as a **blocked** runtime attempt, kept
    distinct from this file's real gating-logic result.

Given genuine two-engine mode is unreachable, the actually-reachable
"two-manager" production scenario for V2 is the *one-model* separate-draft-
KV-cache path (`_should_create_separate_draft_kv_cache()`, same engine,
different KV layout for the draft sub-network, e.g. EAGLE3-one-model,
DraftTarget-one-model, MTP-eagle-one-model). This probe confirms that gate
is `True` **by default** (no forcing) for a real `DraftTargetDecodingConfig`,
and that it, in turn, makes `_needs_gpu_kv_cache_budget_split()` return
`True` for V2 -- i.e. in the actually-reachable path, V2 GPU budget for
target/draft IS split (proportional to per-layer cache cost), which is the
opposite of the (unreachable) two-engine case's dead-code equal-budget
assert at _util.py:2081-2085.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm.llmapi.llm_args import (
    DraftTargetDecodingConfig,
    EagleDecodingConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

GB = 1 << 30


def _creator(*, is_kv_cache_manager_v2: bool, speculative_config, kv_cache_config) -> KvCacheCreator:
    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._mapping = Mapping()
    creator._sparse_attention_config = None
    creator._speculative_config = speculative_config
    creator._is_kv_cache_manager_v2 = is_kv_cache_manager_v2
    creator._kv_cache_config = kv_cache_config
    creator._draft_model_engine = None  # one-model: no separate engine, by construction
    return creator


def test_eagle3_two_model_is_deprecated_and_coerced_to_one_model():
    cfg = EagleDecodingConfig(
        max_draft_len=3, eagle3_one_model=False, speculative_model="unused/for-gating-only"
    )
    assert cfg.eagle3_one_model is True  # coerced despite the explicit False
    assert cfg.spec_dec_mode.is_eagle3_one_model()
    assert cfg.spec_dec_mode.use_one_engine()


def test_draft_target_default_is_one_model_no_public_two_engine_path():
    cfg = DraftTargetDecodingConfig(max_draft_len=4, speculative_model="unused/for-gating-only")
    assert cfg.spec_dec_mode.is_draft_target_one_model()
    assert cfg.spec_dec_mode.use_one_engine()
    # The genuine two-engine value exists in the enum and has its own
    # predicate, but nothing in this codebase ever sets
    # `_draft_target_one_model=False` to reach it via public config.
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

    assert not SpeculativeDecodingMode.DRAFT_TARGET.use_one_engine()


def test_real_one_model_separate_draft_cache_gate_enables_v2_gpu_split_by_default():
    """The real, unforced gate for the actually-reachable "two-manager" V2
    scenario: a default (unmodified) one-model DraftTarget config."""
    spec_config = DraftTargetDecodingConfig(
        max_draft_len=4, speculative_model="unused/for-gating-only"
    )
    kv_cache_config = KvCacheConfig(max_gpu_total_bytes=4 * GB)
    creator = _creator(
        is_kv_cache_manager_v2=True,
        speculative_config=spec_config,
        kv_cache_config=kv_cache_config,
    )

    # Real method, real default DraftTarget(one-model) config, not forced.
    assert creator._should_create_separate_draft_kv_cache() is True
    # Real method: for V2 this directly gates the GPU split.
    assert creator._needs_gpu_kv_cache_budget_split(max_seq_len=2048) is True


def test_real_split_arithmetic_for_one_model_separate_draft_cache():
    """Drives the real, unmodified `_split_kv_cache_budget_for_draft` /
    `_compute_draft_budget_shares` (not reimplemented) with directly-supplied
    per-token `CacheCost` leaves -- the same substitution the already-
    committed `test_scratch_q1_spec_decode_capacity.py` uses, justified
    identically: no HF model artifact is loaded here, so the *inputs* to the
    real split arithmetic are supplied rather than derived, but the split
    *computation itself* (the part actually in question -- proportional vs.
    fixed-ratio vs. equal split) is real, unmodified code.

    Unlike the existing committed probe, this one does not need to force
    `_should_create_separate_draft_kv_cache`: the gate test above already
    established it is `True` by default for this real config, so calling
    the split function here is exercising the branch the real gate actually
    selects, not calling it out-of-band.
    """
    B = 4 * GB
    spec_config = DraftTargetDecodingConfig(
        max_draft_len=4, speculative_model="unused/for-gating-only"
    )
    kv_cache_config = KvCacheConfig(max_gpu_total_bytes=B)
    creator = _creator(
        is_kv_cache_manager_v2=True,
        speculative_config=spec_config,
        kv_cache_config=kv_cache_config,
    )
    assert creator._should_create_separate_draft_kv_cache() is True

    # Target consumes 80% of the combined per-token bytes (e.g. more target
    # layers/heads than draft) -> split must be proportional, not 50/50 and
    # not "each gets B" (that equal-budget behavior is what the *unreachable*
    # two-engine path's dead assert would otherwise imply).
    creator._get_target_and_draft_cache_costs = lambda kv_cache_config: (
        CacheCost(slope=80, intercept=0),
        CacheCost(slope=20, intercept=0),
    )
    creator._kv_cache_manager_cls = Mock()
    creator._model_engine = Mock()

    target_cfg, draft_cfg = creator._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
    assert draft_cfg is not None

    target_budget = target_cfg.max_gpu_total_bytes
    draft_budget = draft_cfg.max_gpu_total_bytes
    print(f"[Q1-split] B={B} target_budget={target_budget} draft_budget={draft_budget}")

    assert target_budget + draft_budget == B
    # Proportional to the 80/20 per-token cost split, not equal halves and
    # not "each independently gets the full B".
    assert draft_budget == pytest.approx(0.2 * B, rel=0.01)
    assert target_budget == pytest.approx(0.8 * B, rel=0.01)
    assert target_budget != draft_budget
    assert target_budget != B and draft_budget != B
