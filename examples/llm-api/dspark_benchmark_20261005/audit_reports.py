# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audit pinned client reports using its own tokenizer and chunk representation."""

import csv
import json
import math
import sys
from pathlib import Path

import msgspec
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    load_reference_tokenizer,
)
from inference_endpoint.core.types import TextModelOutput

root = Path(sys.argv[1])
tokenizer = load_reference_tokenizer("/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-8B")
for phase, expected in [("smoke", 8), ("full", 1319)]:
    directory = root / phase
    report_dir = directory / "measured-report"
    report = json.loads((report_dir / "performance/result_summary.json").read_text())
    events = [json.loads(line) for line in (report_dir / "events.jsonl").read_text().splitlines()]
    mapping = json.loads((report_dir / "sample_idx_map.json").read_text())["performance"]
    assert sorted(mapping.values()) == list(range(expected))
    data = [json.loads(line) for line in (directory / "measured.jsonl").read_text().splitlines()]
    issued = {e["sample_uuid"]: e for e in events if e["event_type"] == "sample.issued"}
    first = {e["sample_uuid"]: e for e in events if e["event_type"] == "sample.recv_first"}
    completed = {e["sample_uuid"]: e for e in events if e["event_type"] == "sample.complete"}
    assert len(issued) == len(first) == len(completed) == expected
    assert not any(e["event_type"].startswith("error.") for e in events)
    rows = []
    for uid, start in issued.items():
        index = mapping[uid]
        assert start["data"][2] == data[index]["input_tokens"]
        result = msgspec.json.decode(
            json.dumps(completed[uid]["data"]).encode(), type=TextModelOutput
        )
        assert not result.reasoning and not result.tool_calls
        n_output = len(tokenizer.encode(str(result), add_special_tokens=False))
        n_tail = len(tokenizer.encode(result.text_after_first_chunk(), add_special_tokens=False))
        assert n_tail > 0
        e2e = completed[uid]["timestamp_ns"] - start["timestamp_ns"]
        ttft = first[uid]["timestamp_ns"] - start["timestamp_ns"]
        rows.append(
            {
                "sample_uuid": uid,
                "prompt_index": index,
                "output_tokens": n_output,
                "post_first_chunk_tokens": n_tail,
                "ttft_ns": ttft,
                "e2e_ns": e2e,
                "tpot_ns": (e2e - ttft) / n_tail,
            }
        )
    tracking = [
        e["timestamp_ns"] for e in events if e["event_type"] == "session.start_performance_tracking"
    ]
    assert len(tracking) == 1
    assert report["duration_ns"] == max(e["timestamp_ns"] for e in completed.values()) - tracking[0]
    for column, key in [
        ("output_tokens", "output_sequence_lengths"),
        ("ttft_ns", "ttft"),
        ("e2e_ns", "latency"),
        ("tpot_ns", "tpot"),
    ]:
        assert math.isclose(sum(r[column] for r in rows), report[key]["total"], rel_tol=1e-10)
    counts = json.loads((report_dir / "metrics/final_snapshot.json").read_text())["metrics"]
    series_counts = {s["name"]: s["count"] for s in counts if s["type"] == "series"}
    assert all(
        series_counts[k] == expected for k in ["ttft_ns", "tpot_ns", "osl", "sample_latency_ns"]
    )
    assert math.isclose(
        report["e2e_avg_interactivity"],
        sum(r["output_tokens"] for r in rows) / (sum(r["e2e_ns"] for r in rows) / 1e9),
    )
    with (directory / "request-audit.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "requests": expected,
        "unique_prompt_coverage": True,
        "wire_token_ids_match": True,
        "TTFT_total_matches": True,
        "TPOT_total_matches_pinned_chunk_semantics": True,
        "OSL_total_matches": True,
        "E2E_total_matches": True,
        "measurement_duration_matches": True,
        "series_counts": series_counts,
    }
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(phase, audit)
