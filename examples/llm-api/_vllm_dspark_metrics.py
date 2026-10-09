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
"""Validate vLLM speculative decoding counters for an isolated workload."""

from __future__ import annotations

import json
import math
from typing import TypedDict

__all__ = ["parse_metrics", "summarize_delta"]

_COUNTERS = {
    "verification_steps": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_draft_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "completed_requests": "vllm:request_success_total",
}
_PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


class _MetricsSnapshot(TypedDict):
    verification_steps: int | None
    draft_tokens: int | None
    accepted_draft_tokens: int | None
    completed_requests: int | None
    accepted_per_position: dict[str, int]
    completed_by_reason: dict[str, int]
    _series: dict[str, dict[str, int]]


class _AcceptanceSummary(TypedDict):
    verification_steps: int
    draft_tokens: int
    accepted_draft_tokens: int
    completed_requests: int
    num_requests: int
    accepted_per_position: dict[str, int]
    completed_by_reason: dict[str, int]
    acceptance_length: float
    acceptance_rate: float


def _group_by_label(series: dict[str, int], label_name: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for identity, value in series.items():
        labels = json.loads(identity)
        if label_name not in labels:
            raise ValueError(f"Missing {label_name!r} label in metrics series {identity}")
        label = labels[label_name]
        if label_name == "position" and (not label.isdecimal() or str(int(label)) != label):
            raise ValueError(f"Invalid draft position {label!r}")
        totals[label] = totals.get(label, 0) + value
    return totals


def parse_metrics(body: str) -> _MetricsSnapshot:
    """Parse one Prometheus scrape without importing vLLM or a GPU runtime.

    Args:
        body: The text response from a vLLM server's /metrics endpoint.

    Returns:
        Counter totals across engines, per-position and finish-reason totals,
        and JSON-safe per-series values used to detect counter resets. A missing
        scalar counter is None; a present zero remains zero. This also supports
        diagnostic scrapes from a prefill server without speculative decoding.

    Raises:
        ValueError: A tracked counter is invalid or has duplicate series.
    """
    from prometheus_client.parser import text_string_to_metric_families

    series: dict[str, dict[str, int]] = {name: {} for name in _COUNTERS.values()}
    series[_PER_POSITION] = {}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name not in series:
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f"Invalid counter {sample.name}: {value!r}")
            identity = json.dumps(sample.labels, sort_keys=True, separators=(",", ":"))
            if identity in series[sample.name]:
                raise ValueError(f"Duplicate counter series {sample.name}{identity}")
            series[sample.name][identity] = int(value)

    def total(key: str) -> int | None:
        values = series[_COUNTERS[key]]
        return sum(values.values()) if values else None

    return {
        "verification_steps": total("verification_steps"),
        "draft_tokens": total("draft_tokens"),
        "accepted_draft_tokens": total("accepted_draft_tokens"),
        "completed_requests": total("completed_requests"),
        "accepted_per_position": _group_by_label(series[_PER_POSITION], "position"),
        "completed_by_reason": _group_by_label(
            series[_COUNTERS["completed_requests"]], "finished_reason"
        ),
        "_series": series,
    }


def _series_delta(name: str, before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    missing = before.keys() - after.keys()
    if missing:
        raise ValueError(f"Counter series disappeared for {name}: {sorted(missing)}")
    delta = {}
    for identity, value in after.items():
        difference = value - before.get(identity, 0)
        if difference < 0:
            raise ValueError(f"Counter reset for {name}{identity}")
        delta[identity] = difference
    return delta


def summarize_delta(
    before: _MetricsSnapshot,
    after: _MetricsSnapshot,
    num_requests: int,
    draft_len: int,
) -> _AcceptanceSummary:
    """Calculate acceptance from decode-server counters after excluding warmup.

    Args:
        before: A parsed scrape taken after all warmup requests completed.
        after: A parsed scrape taken after the measured requests completed.
        num_requests: Expected number of completed measured requests.
        draft_len: Configured maximum number of speculative draft tokens.

    Returns:
        Measured counter deltas, acceptance length (including the bonus token),
        and acceptance rate as a fraction between zero and one.

    Raises:
        ValueError: Counters are missing, reset, incomplete, or inconsistent
            with the isolated workload. Zero verification steps are invalid.
    """
    if type(num_requests) is not int or num_requests <= 0:
        raise ValueError("num_requests must be a positive integer")
    if type(draft_len) is not int or draft_len <= 0:
        raise ValueError("draft_len must be a positive integer")

    for snapshot_name, snapshot in (("before", before), ("after", after)):
        for name in _COUNTERS.values():
            if not snapshot["_series"].get(name):
                raise ValueError(f"Missing required counter {name} in {snapshot_name} scrape")

    deltas = {
        name: _series_delta(name, before["_series"][name], after["_series"][name])
        for name in (*_COUNTERS.values(), _PER_POSITION)
    }
    totals = {key: sum(deltas[name].values()) for key, name in _COUNTERS.items()}
    reasons = _group_by_label(deltas[_COUNTERS["completed_requests"]], "finished_reason")
    invalid_reasons = {
        reason: count
        for reason, count in reasons.items()
        if reason not in ("stop", "length") and count
    }
    if invalid_reasons:
        raise ValueError(f"Requests ended with invalid finish reasons: {invalid_reasons}")
    if totals["completed_requests"] != num_requests:
        raise ValueError(
            f"Expected {num_requests} measured completions, got {totals['completed_requests']}; "
            "wait for metrics to flush and use dedicated servers without other traffic"
        )

    steps = totals["verification_steps"]
    drafted = totals["draft_tokens"]
    accepted = totals["accepted_draft_tokens"]
    if steps == 0 or drafted == 0:
        raise ValueError("No speculative drafts were measured; verify DSpark is enabled on decode")
    if accepted > drafted or drafted > steps * draft_len:
        raise ValueError(
            f"Inconsistent speculative counters: {steps} verification steps, "
            f"{drafted} draft tokens, {accepted} accepted tokens, draft_len={draft_len}"
        )

    positions = _group_by_label(deltas[_PER_POSITION], "position")
    if positions:
        if set(positions) != {str(index) for index in range(draft_len)}:
            raise ValueError("Per-position counters do not match the configured draft length")
        counts = [positions[str(index)] for index in range(draft_len)]
        if sum(counts) != accepted or any(
            later > earlier for earlier, later in zip([steps, *counts], counts)
        ):
            raise ValueError("Per-position accepted counts disagree with speculative totals")

    return {
        "verification_steps": steps,
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "completed_requests": totals["completed_requests"],
        "num_requests": num_requests,
        "accepted_per_position": positions,
        "completed_by_reason": reasons,
        "acceptance_length": 1 + accepted / steps,
        "acceptance_rate": accepted / drafted,
    }
