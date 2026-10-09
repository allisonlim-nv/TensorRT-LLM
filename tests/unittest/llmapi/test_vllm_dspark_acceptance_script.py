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
"""CPU-only local QA; run this file directly to avoid GPU pytest fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "llm-api"


def _load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = _load_module("_vllm_dspark_metrics")
with patch.dict(sys.modules, {"_vllm_dspark_metrics": _METRICS}):
    _RUNNER = _load_module("_vllm_dspark_acceptance")

_URLS = {"prefill": "http://prefill:8001", "decode": "http://decode:8000"}
_TRANSFER = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_engine_id": "prefill-engine",
    "remote_request_id": "prefill-request",
    "remote_block_ids": [[1, 2]],
    "remote_host": "prefill",
    "remote_port": 5557,
}
_COMPLETION = {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 12}}
_NAMES = [
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:request_success_total",
]


def _scrape(engines: list[tuple], positions: bool = True) -> str:
    lines = []
    for engine, values in enumerate(engines):
        for index, (name, value) in enumerate(zip(_NAMES, values)):
            labels = f'engine="{engine}"'
            if index == 3:
                labels += ',finished_reason="length"'
            lines.append(f"{name}{{{labels}}} {value}")
        if positions:
            for position, value in enumerate(values[4]):
                lines.append(
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total"
                    f'{{engine="{engine}",position="{position}"}} {value}'
                )
    return "\n".join(lines) + "\n"


@unittest.skipUnless(importlib.util.find_spec("prometheus_client"), "requires prometheus_client")
class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.before_text = _scrape([(10, 30, 16, 1, [8, 5, 3]), (20, 60, 22, 2, [12, 8, 2])])
        self.after_text = _scrape([(30, 90, 46, 3, [24, 15, 7]), (40, 120, 46, 4, [24, 16, 6])])
        self.before = _METRICS.parse_metrics(self.before_text)
        self.after = _METRICS.parse_metrics(self.after_text)

    def test_engine_aggregation_and_warmup_exclusion(self) -> None:
        result = _METRICS.summarize_delta(self.before, self.after, 4, 3)
        self.assertEqual(json.loads(json.dumps(self.before)), self.before)
        self.assertEqual(result["verification_steps"], 40)
        self.assertEqual(result["draft_tokens"], 120)
        self.assertEqual(result["accepted_draft_tokens"], 54)
        self.assertEqual(result["accepted_per_position"], {"0": 28, "1": 18, "2": 8})
        self.assertAlmostEqual(result["acceptance_length"], 2.35)
        self.assertAlmostEqual(result["acceptance_rate"], 0.45)

    def test_invalid_missing_or_duplicate_counters(self) -> None:
        for value in ("-1", "NaN", "+Inf", "-Inf", "1.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _METRICS.parse_metrics(f"{_NAMES[0]} {value}\n")
        for name in _NAMES:
            text = "\n".join(line for line in self.after_text.splitlines() if name not in line)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing required"):
                _METRICS.summarize_delta(self.before, _METRICS.parse_metrics(text), 4, 3)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _METRICS.parse_metrics(f"{_NAMES[0]} 1\n{_NAMES[0]} 2\n")
        diagnostic = _METRICS.parse_metrics(f'{_NAMES[3]}{{finished_reason="length"}} 0\n')
        self.assertIsNone(diagnostic["verification_steps"])
        self.assertEqual(diagnostic["completed_requests"], 0)

    def test_per_engine_reset_cannot_hide_in_rising_aggregate(self) -> None:
        snapshots = [
            _scrape([(9, 90, 46, 3, [24, 15, 7]), (61, 120, 46, 4, [24, 16, 6])]),
            _scrape([(70, 210, 92, 7, [48, 31, 13])]),
        ]
        for text in snapshots:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "reset|disappeared"):
                _METRICS.summarize_delta(self.before, _METRICS.parse_metrics(text), 4, 3)

    def test_inconsistent_measurements_fail(self) -> None:
        before = _METRICS.parse_metrics(_scrape([(0, 0, 0, 0, [0, 0, 0])]))
        invalid = [
            (0, 0, 0, 4, [0, 0, 0]),
            (10, 31, 14, 4, [8, 4, 2]),
            (10, 30, 31, 4, [10, 10, 11]),
            (10, 30, 14, 3, [8, 4, 2]),
            (10, 30, 14, 4, [4, 8, 2]),
            (10, 30, 14, 4, [8, 4, 3]),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _METRICS.summarize_delta(before, _METRICS.parse_metrics(_scrape([values])), 4, 3)
        for reason in ("error", "abort", "repetition"):
            extra = f'{_NAMES[3]}{{finished_reason="{reason}"}} 1\n'
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "finish reasons"):
                _METRICS.summarize_delta(
                    self.before, _METRICS.parse_metrics(self.after_text + extra), 4, 3
                )


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        socket_patch = patch.object(_RUNNER.socket, "socket")
        self.socket = socket_patch.start()
        self.addCleanup(socket_patch.stop)

    def arguments(self, mode: str = "disagg", extras: tuple | list = ()) -> argparse.Namespace:
        return _RUNNER._arguments(
            mode,
            [
                "--tensor-parallel-size",
                "1",
                "--num-speculative-tokens",
                "3",
                "--max-tokens",
                "12",
                "--output-dir",
                str(self.root / "output"),
                *extras,
            ],
        )

    def test_handoff_preserves_prompt_budget_sampling_and_request_id(self) -> None:
        args = self.arguments(extras=["--temperature", "0.7", "--top-p", "0.8"])
        responses = [json.dumps({"kv_transfer_params": _TRANSFER}), json.dumps(_COMPLETION)]
        with patch.object(_RUNNER, "_http", side_effect=responses) as http:
            _RUNNER._complete(args, _URLS, [17, 23, 45], 12)
        prefill, decode = [call.args for call in http.call_args_list]
        self.assertEqual(prefill[2]["max_tokens"], 1)
        self.assertEqual(decode[2]["max_tokens"], 12)
        self.assertEqual(decode[2]["kv_transfer_params"], _TRANSFER)
        self.assertEqual(prefill[3], decode[3])
        for request in (prefill, decode):
            self.assertEqual(request[2]["prompt"], [17, 23, 45])
            self.assertEqual(request[2]["temperature"], 0.7)
            self.assertEqual(request[2]["top_p"], 0.8)

    def test_incomplete_handoff_never_calls_decode(self) -> None:
        for key in (
            "do_remote_prefill",
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
            "remote_block_ids",
        ):
            transfer = {name: value for name, value in _TRANSFER.items() if name != key}
            with (
                self.subTest(missing=key),
                patch.object(
                    _RUNNER, "_http", return_value=json.dumps({"kv_transfer_params": transfer})
                ) as http,
            ):
                with self.assertRaisesRegex(RuntimeError, "NIXL handoff"):
                    _RUNNER._complete(self.arguments(), _URLS, [1, 2], 12)
                self.assertEqual(http.call_count, 1)

    def test_gpu_and_port_allocation(self) -> None:
        for devices in ("0", "0,0", "0,00", "0,GPU-abcd", "0,", "-1,0", ""):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                _RUNNER._launch_plan(self.arguments(extras=["--devices=" + devices]))
        for flags in (
            ["--port", "65536"],
            ["--prefill-port", "8000"],
            ["--decode-side-channel-port", "5557"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                _RUNNER._launch_plan(self.arguments(extras=["--devices", "0,1", *flags]))
        plan = _RUNNER._launch_plan(self.arguments(extras=["--devices", "3,7"]))
        self.assertEqual([item["env"]["CUDA_VISIBLE_DEVICES"] for item in plan], ["3", "7"])

    def test_sampling_recipe_matches_launch_and_comparison(self) -> None:
        args = self.arguments(
            extras=[
                "--devices",
                "0,1",
                "--temperature",
                "0.7",
                "--top-p",
                "0.8",
                "--rejection-sample-method",
                "block",
                "--kv-cache-dtype",
                "fp8",
            ]
        )
        comparison = _RUNNER._comparison(args, [[1, 2]])
        self.assertEqual(comparison["temperature"], 0.7)
        self.assertEqual(comparison["top_p"], 0.8)
        spec = comparison["recipe"]["speculative_config"]
        self.assertEqual(spec["draft_sample_method"], "probabilistic")
        self.assertEqual(spec["rejection_sample_method"], "block")
        self.assertEqual(spec["attention_backend"], "FLASHINFER_MLA")
        for server in _RUNNER._launch_plan(args):
            command = server["command"]
            self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]), spec)
            self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "fp8")
            attention = json.loads(command[command.index("--attention-config") + 1])
            self.assertEqual(
                attention,
                {
                    "mla_prefill_backend": "FLASHINFER",
                    "use_prefill_query_quantization": True,
                },
            )
            self.assertEqual(attention, comparison["recipe"]["attention_config"])
        self.assertEqual(_RUNNER._recipe(self.arguments())["attention_config"], {})
        self.assertEqual(self.arguments().draft_sample_method, "greedy")
        self.assertEqual(
            self.arguments(
                extras=["--temperature", "0.7", "--draft-sample-method", "greedy"]
            ).draft_sample_method,
            "greedy",
        )

    def test_invalid_sampling_and_external_urls(self) -> None:
        flags = [
            ["--temperature", "nan"],
            ["--top-p", "0"],
            ["--top-p", "1.1"],
            ["--prefill-url", _URLS["prefill"]],
        ]
        flags += [
            ["--prefill-url", url, "--decode-url", _URLS["decode"]]
            for url in (
                "http://prefill?x=1",
                "http://prefill#fragment",
                "http://user:pass@prefill",
                "http://prefill/v1",
            )
        ]
        for extras in flags:
            with (
                self.subTest(extras=extras),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.arguments(extras=extras)

    def test_setup_and_workload_failures_clean_started_processes(self) -> None:
        args = self.arguments(extras=["--devices", "0,1"])
        args.output_dir.mkdir()
        process = Mock(pid=43210)
        process.poll.return_value = None
        for setup_failure in (True, False):
            launches = [process, OSError("launch failed")] if setup_failure else [process, process]
            with (
                self.subTest(setup_failure=setup_failure),
                patch.object(_RUNNER.subprocess, "Popen", side_effect=launches),
                patch.object(_RUNNER, "_http", return_value="ok"),
                patch.object(_RUNNER, "_stop_processes") as stop,
            ):
                with self.assertRaises((OSError, RuntimeError)):
                    with _RUNNER._servers(args, _RUNNER._launch_plan(args)):
                        raise RuntimeError("workload failed")
                self.assertEqual(
                    stop.call_args.args[0], [process] if setup_failure else [process, process]
                )

    def test_occupied_port_rejected_before_launch_or_health_probe(self) -> None:
        args = self.arguments(extras=["--devices", "0,1"])
        self.socket.return_value.__enter__.return_value.bind.side_effect = OSError("in use")
        with (
            patch.object(_RUNNER.subprocess, "Popen") as launch,
            patch.object(_RUNNER, "_http") as http,
        ):
            with self.assertRaisesRegex(OSError, "in use"):
                with _RUNNER._servers(args, _RUNNER._launch_plan(args)):
                    self.fail("Occupied port must not yield a ready server")
            launch.assert_not_called()
            http.assert_not_called()

    def test_cleanup_escalates_unresponsive_child(self) -> None:
        process = Mock(pid=43210)
        process.wait.side_effect = [subprocess.TimeoutExpired("vllm", 5), 0]
        with patch.object(_RUNNER.os, "killpg") as kill:
            _RUNNER._stop_processes([process])
        self.assertEqual(
            [call.args for call in kill.call_args_list],
            [(43210, signal.SIGTERM), (43210, signal.SIGKILL)],
        )

    def test_dry_run_avoids_downloads_launches_http_and_writes(self) -> None:
        for mode in ("agg", "disagg"):
            with (
                self.subTest(mode=mode),
                patch.object(_RUNNER, "_measure", side_effect=AssertionError("measurement")),
                patch.object(_RUNNER.subprocess, "run", side_effect=AssertionError("GPU probe")),
                patch.dict(os.environ, {}, clear=True),
                redirect_stdout(io.StringIO()),
            ):
                destination = self.root / mode
                self.assertEqual(
                    _RUNNER.main(mode, ["--dry-run", "--output-dir", str(destination)]), 0
                )
                self.assertFalse(destination.exists())
                self.socket.assert_not_called()

    @unittest.skipUnless(
        importlib.util.find_spec("prometheus_client"), "requires prometheus_client"
    )
    def test_snapshot_waits_for_warmup_and_rejects_other_traffic(self) -> None:
        args = self.arguments("agg")
        args.output_dir.mkdir()
        bodies = [_scrape([(0, 0, 0, completed, [0, 0, 0])]) for completed in (1, 2, 3, 4)]
        urls = {"aggregate": "http://agg"}
        with (
            patch.object(_RUNNER, "_http", side_effect=bodies[:3]) as http,
            patch.object(_RUNNER.time, "sleep"),
        ):
            snapshot = _RUNNER._snapshot(args, urls, {"aggregate": 3}, "before")
        self.assertEqual(http.call_count, 3)
        self.assertEqual(snapshot["aggregate"]["completed_requests"], 3)
        with (
            patch.object(_RUNNER, "_http", return_value=bodies[3]),
            self.assertRaisesRegex(ValueError, "concurrent traffic"),
        ):
            _RUNNER._snapshot(args, urls, {"aggregate": 3}, "after")
        with (
            patch.object(_RUNNER, "_http", return_value=bodies[0]),
            patch.object(_RUNNER.time, "monotonic", side_effect=[0, 31]),
            self.assertRaisesRegex(TimeoutError, "catch up"),
        ):
            _RUNNER._snapshot(args, urls, {"aggregate": 3}, "before")

    @unittest.skipUnless(
        importlib.util.find_spec("prometheus_client"), "requires prometheus_client"
    )
    def test_full_aggregate_disaggregate_and_comparison_smoke(self) -> None:
        prompt_file = self.root / "prompts.jsonl"
        prompt_file.write_text('{"prompt":"one"}\n{"prompt":"two"}\n')
        for mode in ("agg", "disagg"):
            counts = {"aggregate": 0} if mode == "agg" else {"prefill": 0, "decode": 0}
            requests = []

            def http(
                url: str, timeout: int, payload: dict | None = None, request_id: str | None = None
            ) -> str:
                role = next(role for role in counts if f"{role}:" in url)
                if url.endswith("/health"):
                    return "ok"
                if url.endswith("/tokenize"):
                    return json.dumps({"tokens": [1, 2, 3], "max_model_len": 8192})
                if url.endswith("/metrics"):
                    count = counts[role]
                    if role == "prefill":
                        return f'{_NAMES[3]}{{finished_reason="length"}} {count}\n'
                    return _scrape(
                        [(count * 2, count * 6, count * 4, count, [count * 2, count * 2, 0])]
                    )
                counts[role] += 1
                requests.append((role, payload))
                return json.dumps(
                    {"kv_transfer_params": _TRANSFER} if role == "prefill" else _COMPLETION
                )

            destination = self.root / mode
            argv = [
                "--output-dir",
                str(destination),
                "--prompt-file",
                str(prompt_file),
                "--num-prompts",
                "2",
                "--warmup-prompts",
                "1",
                "--num-speculative-tokens",
                "3",
                "--max-tokens",
                "12",
            ]
            if mode == "agg":
                argv += ["--server-url", "http://aggregate:8000"]
            else:
                argv += [
                    "--prefill-url",
                    _URLS["prefill"],
                    "--decode-url",
                    _URLS["decode"],
                    "--aggregate-results",
                    str(self.root / "agg"),
                    "--max-al-difference",
                    "0",
                ]
            with (
                self.subTest(mode=mode),
                patch.object(_RUNNER, "_http", side_effect=http),
                redirect_stdout(io.StringIO()),
                patch.object(
                    _RUNNER.subprocess, "Popen", side_effect=AssertionError("external launch")
                ),
            ):
                self.assertEqual(_RUNNER.main(mode, argv), 0)
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["verification_steps"], 4)
            self.assertEqual(summary["accepted_draft_tokens"], 8)
            self.assertEqual(summary["completed_requests"], 2)
            self.assertEqual(summary["acceptance_length"], 3.0)
            self.assertAlmostEqual(summary["acceptance_rate"], 2 / 3)
            role = "aggregate" if mode == "agg" else "decode"
            self.assertEqual(
                [body["max_tokens"] for target, body in requests if target == role], [8, 12, 12]
            )
            for name in ("summary.csv", "manifest.json", f"{role}-before.prom"):
                self.assertTrue((destination / name).exists())
        self.assertEqual(summary["al_difference"], 0)


if __name__ == "__main__":
    unittest.main()
