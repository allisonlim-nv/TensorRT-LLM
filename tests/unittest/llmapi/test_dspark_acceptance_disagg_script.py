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
"""CPU-only local tests; run this file directly to avoid GPU pytest fixtures."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "llm-api"
_SPEC = importlib.util.spec_from_file_location(
    "dspark_acceptance_disagg_script", _EXAMPLES / "measure_dspark_acceptance_disagg.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def _counters(
    drafted: dict[str, int] | None = None,
    accepted: dict[str, int] | None = None,
    completed: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    return {"drafted": drafted or {}, "accepted": accepted or {}, "completed": completed or {}}


def _arguments(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=_EXAMPLES / "dspark_acceptance_disagg.yaml",
        models_root=output_dir.parent / "models",
        output_dir=output_dir,
        case=None,
        devices=None,
        num_prompts=2,
        max_tokens=256,
        prompt_file=None,
        prompt_format="chat",
        warmup_prompts=2,
        concurrency=8,
        startup_timeout=60,
        request_timeout=60,
        dry_run=False,
    )


@unittest.skipUnless(importlib.util.find_spec("prometheus_client"), "requires prometheus_client")
class TestPrometheusCounters(unittest.TestCase):
    def test_completed_request_metrics_use_real_labels_and_sum_workers(self) -> None:
        metrics = """
# TYPE trtllm_spec_decode_drafted_tokens counter
trtllm_spec_decode_drafted_tokens_total{token_position="0",worker="0"} 10
trtllm_spec_decode_drafted_tokens_total{token_position="0",worker="1"} 20
trtllm_spec_decode_drafted_tokens_total{token_position="1",worker="0"} 10
trtllm_spec_decode_drafted_tokens_total{token_position="1",worker="1"} 20
trtllm_spec_decode_drafted_tokens_total{token_position="2",worker="0"} 10
trtllm_spec_decode_drafted_tokens_total{token_position="2",worker="1"} 20
# TYPE trtllm_spec_decode_accepted_tokens counter
trtllm_spec_decode_accepted_tokens_total{token_position="0",worker="0"} 10
trtllm_spec_decode_accepted_tokens_total{token_position="0",worker="1"} 20
trtllm_spec_decode_accepted_tokens_total{token_position="1",worker="1"} 20
trtllm_spec_decode_accepted_tokens_total{token_position="2",worker="1"} 10
# TYPE trtllm_request_success counter
trtllm_request_success_total{finished_reason="stop",worker="0"} 1
trtllm_request_success_total{finished_reason="length",worker="1"} 1
trtllm_spec_decode_drafted_tokens_created{token_position="0"} 999
trtllm_iteration_num_accepted_draft_tokens{worker="0"} 999
"""
        actual = _RUNNER._parse_metrics(metrics)
        self.assertEqual(
            actual,
            _counters(
                {"0": 30, "1": 30, "2": 30},
                {"0": 30, "1": 20, "2": 10},
                {"stop": 1, "length": 1},
            ),
        )
        summary = _RUNNER._summarize_counters(
            _counters(completed={"not_finished": 2}), actual, 2, 3
        )
        self.assertEqual(summary["acceptance_length"], 3.0)
        self.assertNotEqual(summary["acceptance_length"], (2.0 + 3.5) / 2)

    def test_unobserved_accepted_positions_are_zero(self) -> None:
        actual = _RUNNER._parse_metrics("""
trtllm_spec_decode_drafted_tokens_total{token_position="0"} 5
trtllm_spec_decode_drafted_tokens_total{token_position="1"} 5
trtllm_request_success_total{finished_reason="stop"} 1
""")
        self.assertEqual(actual["accepted"], {})
        summary = _RUNNER._summarize_counters(
            _counters(completed={"not_finished": 1}), actual, 1, 2
        )
        self.assertEqual(summary["acceptance_length"], 1.0)
        self.assertEqual(summary["acceptance_rate"], 0.0)

    def test_invalid_counter_values_fail_closed(self) -> None:
        for value in ("-1", "1.5", "NaN", "+Inf", "-Inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _RUNNER._parse_metrics(
                    'trtllm_spec_decode_drafted_tokens_total{token_position="0"} ' + value
                )

    def test_invalid_position_fails_closed(self) -> None:
        for label in ("-1", "all", "0.5"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                _RUNNER._parse_metrics(
                    f'trtllm_spec_decode_drafted_tokens_total{{token_position="{label}"}} 1'
                )

    def test_missing_finished_reason_is_not_treated_as_success(self) -> None:
        with self.assertRaises((KeyError, ValueError)):
            _RUNNER._parse_metrics('trtllm_request_success_total{finish_reason="stop"} 1')


class TestCounterMeasurement(unittest.TestCase):
    def test_counter_delta_excludes_warmup_and_preserves_new_labels(self) -> None:
        before = _counters({"0": 2, "1": 2}, {"0": 1}, {"stop": 2})
        after = _counters({"0": 12, "1": 10}, {"0": 9, "1": 6}, {"stop": 5, "length": 1})
        before_copy, after_copy = copy.deepcopy(before), copy.deepcopy(after)
        self.assertEqual(
            _RUNNER._counter_delta(before, after),
            _counters({"0": 10, "1": 8}, {"0": 8, "1": 6}, {"stop": 3, "length": 1}),
        )
        self.assertEqual(before, before_copy)
        self.assertEqual(after, after_copy)

    def test_reset_or_disappearing_nonzero_counter_fails(self) -> None:
        before = _counters({"0": 2})
        for after in (_counters({"0": 1}), _counters()):
            with self.subTest(after=after), self.assertRaisesRegex(RuntimeError, "reset"):
                _RUNNER._counter_delta(before, after)

    def test_context_early_eos_is_excluded_from_generation_handoffs(self) -> None:
        context = _counters(completed={"stop": 1, "length": 1, "not_finished": 2})
        generation = _counters({"0": 10, "1": 8}, {"0": 8, "1": 6}, {"stop": 2, "length": 1})
        actual = _RUNNER._summarize_counters(context, generation, 4, 2)
        self.assertEqual(actual["acceptance_length"], 2.4)
        self.assertAlmostEqual(actual["acceptance_rate"], 14 / 18)
        self.assertEqual(actual["verification_steps"], 10)
        self.assertEqual(actual["accepted_draft_tokens"], 14)
        self.assertEqual(actual["draft_tokens"], 18)
        self.assertEqual(actual["num_requests"], 4)
        self.assertEqual(actual["generation_requests"], 3)
        self.assertEqual(actual["context_only_requests"], 1)

    def test_incomplete_or_unexpected_completions_fail(self) -> None:
        for context, generation in (
            ({"not_finished": 1}, {"stop": 2}),
            ({"not_finished": 2}, {"stop": 1}),
            ({"cancelled": 2}, {}),
            ({"not_finished": 2}, {"cancelled": 2}),
        ):
            with (
                self.subTest(context=context, generation=generation),
                self.assertRaises(RuntimeError),
            ):
                _RUNNER._summarize_counters(
                    _counters(completed=context),
                    _counters({"0": 4, "1": 4}, {"0": 2}, generation),
                    2,
                    2,
                )

    def test_invalid_speculative_counts_fail(self) -> None:
        for drafted, accepted in (
            ({"0": 4, "1": 4}, {"0": -1}),
            ({"0": 4, "1": 4}, {"0": 5}),
            ({"0": 4, "1": 4, "2": 1}, {}),
            ({"0": 4, "1": 5}, {}),
        ):
            with self.subTest(drafted=drafted, accepted=accepted), self.assertRaises(RuntimeError):
                _RUNNER._summarize_counters(
                    _counters(completed={"not_finished": 1}),
                    _counters(drafted, accepted, {"stop": 1}),
                    1,
                    2,
                )

    def test_no_drafts_cannot_produce_an_al(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No draft"):
            _RUNNER._summarize_counters(_counters(completed={"stop": 1}), _counters(), 1, 2)

    def test_position_zero_only_fallback_is_rejected_for_multi_token_drafts(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "multi-position"):
            _RUNNER._summarize_counters(
                _counters(completed={"not_finished": 1}),
                _counters({"0": 10}, {"0": 5}, {"stop": 1}),
                1,
                2,
            )

    def test_position_zero_is_sufficient_for_single_token_drafts(self) -> None:
        result = _RUNNER._summarize_counters(
            _counters(completed={"not_finished": 1}),
            _counters({"0": 10}, {"0": 5}, {"stop": 1}),
            1,
            1,
        )
        self.assertEqual(result["acceptance_length"], 1.5)


class TestRecipesAndPrompts(unittest.TestCase):
    def test_recipes_keep_per_worker_parallelism_and_generation_only_dspark(self) -> None:
        cases = _RUNNER._load_cases(_arguments(Path("/tmp/dspark-test-unused-output")))
        expected = {
            "qwen3-8b": (2, 1, 7),
            "deepseek-v4-flash-nvfp4": (8, 4, 5),
        }
        self.assertEqual(set(cases), set(expected))
        for name, (total, per_worker, draft_len) in expected.items():
            with self.subTest(case=name):
                case = cases[name]
                self.assertEqual(case["required_gpus"], total)
                self.assertEqual(case["max_draft_len"], draft_len)
                self.assertTrue(Path(case["model"]).is_absolute())
                self.assertTrue(Path(case["drafter"]).is_absolute())
                self.assertNotIn("speculative_config", case["context_options"])
                self.assertEqual(
                    case["generation_options"]["speculative_config"]["decoding_type"], "DSpark"
                )
                for role in ("context", "generation"):
                    options = case[f"{role}_options"]
                    self.assertEqual(options["tensor_parallel_size"], per_worker)
                    self.assertEqual(options["moe_expert_parallel_size"], per_worker)
                    self.assertEqual(options["num_postprocess_workers"], 0)
                    self.assertEqual(options["num_serve_frontends"], 1)
                    self.assertTrue(options["return_perf_metrics"])
                    self.assertEqual(options["max_batch_size"], 8)
                    self.assertEqual(options["cache_transceiver_config"]["backend"], "NIXL")
                    self.assertEqual(
                        options["cache_transceiver_config"]["transceiver_runtime"], "PYTHON"
                    )

    def test_case_selection_rejects_unknown_duplicate_and_mixed_all(self) -> None:
        for selected in (["missing"], ["qwen3-8b", "qwen3-8b"], ["all", "qwen3-8b"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                args = _arguments(Path("/tmp/dspark-test-unused-output"))
                args.case = selected
                _RUNNER._load_cases(args)

    def test_chat_prompt_is_tokenized_once_with_thinking_disabled(self) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = [12, 34]
        result = _RUNNER._format_prompts(
            tokenizer,
            ["question"],
            {"system_prompt": "system", "chat_template_kwargs": {"enable_thinking": False}},
            "chat",
        )
        self.assertEqual(result, [[12, 34]])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenizer.encode.assert_not_called()

    def test_raw_prompt_does_not_add_special_tokens(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [12, 34]
        self.assertEqual(_RUNNER._format_prompts(tokenizer, ["question"], {}, "raw"), [[12, 34]])
        tokenizer.encode.assert_called_once_with("question", add_special_tokens=False)
        tokenizer.apply_chat_template.assert_not_called()

    def test_explicit_devices_override_environment_without_gpu_queries(self) -> None:
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}),
            patch.object(_RUNNER.subprocess, "run") as subprocess_run,
        ):
            self.assertEqual(_RUNNER._get_devices("0,1"), ["0", "1"])
            self.assertEqual(_RUNNER._get_devices(None), ["4", "5"])
            self.assertEqual(_RUNNER._get_devices("-1"), [])
            subprocess_run.assert_not_called()

    def test_device_aliases_are_normalized_and_duplicates_rejected(self) -> None:
        gpu_uuid = "GPU-ABCDEF01-1234-5678-9ABC-DEF012345678"
        self.assertEqual(_RUNNER._get_devices("00,01"), ["0", "1"])
        self.assertEqual(_RUNNER._get_devices(gpu_uuid), ["GPU-" + gpu_uuid[4:].lower()])
        for devices in ("0,00", f"{gpu_uuid},{'GPU-' + gpu_uuid[4:].lower()}"):
            with self.subTest(devices=devices), self.assertRaisesRegex(ValueError, "distinct"):
                _RUNNER._get_devices(devices)

    def test_mixed_gpu_aliases_and_mig_are_rejected(self) -> None:
        for devices in (
            "0,GPU-abcdef01-1234-5678-9abc-def012345678",
            "MIG-abcdef01-1234-5678-9abc-def012345678",
            "GPU-abcdef01",
            "0,",
            "-2",
        ):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                _RUNNER._get_devices(devices)


class TestRunnerOrchestration(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dspark-disagg-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = _arguments(self.directory / "output")
        self.cases = _RUNNER._load_cases(self.args)

    def test_eight_gpu_sweep_runs_all_configured_cases_without_skips(self) -> None:
        def result(
            args: argparse.Namespace, name: str, case: dict, devices: list, prompts: list
        ) -> dict:
            return {
                "case": name,
                "status": "ok",
                "acceptance_length": 2.0,
                "num_requests": len(prompts),
            }

        with (
            patch.object(_RUNNER, "_load_prompts", return_value=["one", "two"]),
            patch.object(_RUNNER, "_run_case", side_effect=result) as run_case,
            patch.object(Path, "is_file", return_value=True),
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, self.cases, list(map(str, range(8))))
        self.assertEqual(code, 0)
        self.assertEqual(
            [call.args[1] for call in run_case.call_args_list],
            ["qwen3-8b", "deepseek-v4-flash-nvfp4"],
        )
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(
            [row["status"] for row in summary["results"]],
            ["ok", "ok"],
        )
        self.assertEqual(len((self.args.output_dir / "prompts.jsonl").read_text().splitlines()), 2)
        manifest = json.loads((self.args.output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["mode"], "disagg")
        self.assertEqual(manifest["num_prompts"], 2)
        self.assertTrue((self.args.output_dir / "summary.csv").is_file())

    def test_two_visible_gpus_run_qwen_and_report_flash_skip(self) -> None:
        with (
            patch.object(_RUNNER, "_load_prompts", return_value=["one", "two"]),
            patch.object(
                _RUNNER, "_run_case", return_value={"status": "ok", "acceptance_length": 2.0}
            ) as run_case,
            patch.object(Path, "is_file", return_value=True),
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, self.cases, ["0", "1"])
        self.assertEqual(code, 0)
        run_case.assert_called_once()
        self.assertEqual(run_case.call_args.args[1], "qwen3-8b")
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(
            [row["status"] for row in summary["results"]], ["ok", "skipped_insufficient_gpus"]
        )
        skipped = summary["results"][1]
        self.assertEqual(skipped["case"], "deepseek-v4-flash-nvfp4")
        self.assertEqual(skipped["required_gpus"], 8)
        self.assertNotIn("acceptance_length", skipped)

    def test_all_hardware_skips_do_not_load_dataset_or_start_workers(self) -> None:
        with (
            patch.object(_RUNNER, "_load_prompts") as load_prompts,
            patch.object(_RUNNER, "_run_case") as run_case,
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, self.cases, [])
        self.assertNotEqual(code, 0)
        load_prompts.assert_not_called()
        run_case.assert_not_called()
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertTrue(
            all(row["status"] == "skipped_insufficient_gpus" for row in summary["results"])
        )

    def test_missing_checkpoint_is_failure_not_hardware_skip(self) -> None:
        with (
            patch.object(_RUNNER, "_load_prompts", return_value=["one", "two"]),
            patch.object(_RUNNER, "_run_case") as run_case,
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, {"qwen3-8b": self.cases["qwen3-8b"]}, ["0", "1"])
        self.assertNotEqual(code, 0)
        run_case.assert_not_called()
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertEqual(summary["results"][0]["status"], "missing_checkpoint")

    def test_worker_failure_is_recorded_and_other_cases_continue(self) -> None:
        with (
            patch.object(_RUNNER, "_load_prompts", return_value=["one", "two"]),
            patch.object(
                _RUNNER,
                "_run_case",
                side_effect=[
                    RuntimeError("worker failed"),
                    {"status": "ok", "acceptance_length": 2.0},
                ],
            ) as run_case,
            patch.object(Path, "is_file", return_value=True),
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, self.cases, list(map(str, range(8))))
        self.assertNotEqual(code, 0)
        self.assertEqual(run_case.call_count, 2)
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertEqual(summary["results"][0]["status"], "failed")
        self.assertIn("worker failed", summary["results"][0]["error"])
        self.assertEqual(summary["results"][-1]["status"], "ok")

    def test_existing_output_directory_is_not_reused(self) -> None:
        self.args.output_dir.mkdir()
        with (
            patch.object(_RUNNER, "_load_prompts") as load_prompts,
            patch.object(_RUNNER, "_run_case") as run_case,
            redirect_stdout(io.StringIO()),
            self.assertRaises(FileExistsError),
        ):
            _RUNNER._run_cases(self.args, self.cases, list(map(str, range(8))))
        load_prompts.assert_not_called()
        run_case.assert_not_called()

    def test_dry_run_does_not_create_outputs_or_start_workers(self) -> None:
        self.args.dry_run = True
        with (
            patch.object(_RUNNER, "_load_prompts") as load_prompts,
            patch.object(_RUNNER, "_run_case") as run_case,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(_RUNNER._run_cases(self.args, self.cases, list(map(str, range(8)))), 0)
        self.assertFalse(self.args.output_dir.exists())
        load_prompts.assert_not_called()
        run_case.assert_not_called()

    def test_launcher_uses_disjoint_gpu_pools_and_cleans_up_owned_processes(self) -> None:
        case_dir = self.directory / "qwen"
        case_dir.mkdir()
        (case_dir / "router.addr").write_text("127.0.0.1:12340")
        cluster = {
            "is_ready": True,
            "current_workers": {
                "context_servers": [{"host": "127.0.0.1", "port": 12341}],
                "generation_servers": [{"host": "127.0.0.1", "port": 12342}],
            },
        }
        sockets = [MagicMock(), MagicMock()]
        for index, rendezvous in enumerate(sockets):
            rendezvous.__enter__.return_value = rendezvous
            rendezvous.getsockname.return_value = ("127.0.0.1", 12400 + index)
        processes = [MagicMock(pid=20000 + index) for index in range(3)]
        for process in processes:
            process.poll.return_value = None
        with (
            patch.object(_RUNNER.subprocess, "Popen", side_effect=processes) as popen,
            patch.object(_RUNNER.socket, "socket", side_effect=sockets),
            patch.object(_RUNNER, "_http", return_value=json.dumps(cluster)),
            patch.object(_RUNNER, "_stop_processes") as stop,
            patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "4", "MASTER_PORT": "9999"}),
        ):
            with _RUNNER._launch_servers(
                self.args, self.cases["qwen3-8b"], ["0", "1"], case_dir
            ) as launched:
                self.assertEqual(
                    launched[:3],
                    ("http://127.0.0.1:12340", "http://127.0.0.1:12341", "http://127.0.0.1:12342"),
                )
                self.assertEqual(launched[3], processes)
                stop.assert_not_called()
            stop.assert_called_once_with(processes)
        self.assertEqual(popen.call_count, 3)
        self.assertEqual(
            [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list],
            ["", "0", "1"],
        )
        for call in popen.call_args_list:
            self.assertTrue(call.kwargs["start_new_session"])
            self.assertNotIn("OMPI_COMM_WORLD_RANK", call.kwargs["env"])
        self.assertNotEqual(
            popen.call_args_list[1].kwargs["env"]["MASTER_PORT"],
            popen.call_args_list[2].kwargs["env"]["MASTER_PORT"],
        )
        for role in ("router", "context", "generation"):
            self.assertEqual((case_dir / f"{role}.yaml").stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
