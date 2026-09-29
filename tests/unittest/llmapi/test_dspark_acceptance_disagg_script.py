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
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
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

    def test_context_speculative_counters_do_not_change_generation_al(self) -> None:
        generation = _counters({"0": 10, "1": 8}, {"0": 8, "1": 6}, {"stop": 2, "length": 1})
        for drafted, accepted in (
            ({"0": 100, "1": 80}, {"0": 90, "1": 70}),
            ({"0": 100}, {"0": 90}),
            ({"0": 0, "1": 0}, {"0": 0, "1": 0}),
        ):
            with self.subTest(drafted=drafted, accepted=accepted):
                context = _counters(drafted, accepted, {"stop": 1, "not_finished": 3})
                actual = _RUNNER._summarize_counters(context, generation, 4, 2)
                self.assertEqual(actual["acceptance_length"], 2.4)
                self.assertAlmostEqual(actual["acceptance_rate"], 14 / 18)
                self.assertEqual(actual["verification_steps"], 10)
                self.assertEqual(actual["accepted_draft_tokens"], 14)
                self.assertEqual(actual["draft_tokens"], 18)
                self.assertEqual(actual["generation_requests"], 3)
                self.assertEqual(actual["context_only_requests"], 1)
                self.assertEqual(actual["context_counters"], context)

    def test_malformed_context_speculative_counters_are_rejected(self) -> None:
        generation = _counters({"0": 10, "1": 8}, {"0": 8, "1": 6}, {"stop": 1})
        for drafted, accepted in (
            ({"0": -1}, {}),
            ({"0": 4, "1": 4}, {"0": -1}),
            ({"0": 4, "1": 4}, {"0": 5}),
            ({"0": 4, "1": 4, "2": 1}, {}),
            ({"0": 4, "1": 5}, {}),
            ({"-1": 1}, {}),
            ({"bad": 1}, {}),
        ):
            with (
                self.subTest(drafted=drafted, accepted=accepted),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                _RUNNER._summarize_counters(
                    _counters(drafted, accepted, {"not_finished": 1}), generation, 1, 2
                )

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
    def test_expanded_matrix_preserves_features_and_has_matched_aggregate_references(self) -> None:
        args = _arguments(Path("/tmp/dspark-test-unused-output"))
        cases = _RUNNER._load_cases(args)
        spec = importlib.util.spec_from_file_location(
            "aggregate_runner", _EXAMPLES / "measure_dspark_acceptance.py"
        )
        aggregate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aggregate)
        aggregate_args = copy.copy(args)
        aggregate_args.config = _EXAMPLES / "dspark_acceptance.yaml"
        references = aggregate._load_cases(aggregate_args)
        expected_cases = {
            "qwen3-8b",
            "qwen3-8b-explicit-pools",
            "deepseek-v4-flash-nvfp4",
            "deepseek-v4-flash-nvfp4-explicit-pools",
            "deepseek-v4-flash-nvfp4-explicit-pools-no-scratch",
        }
        self.assertEqual(set(cases), expected_cases)
        self.assertEqual(set(references), expected_cases)
        for name, case in cases.items():
            with self.subTest(case=name):
                reference = references[case["aggregate_case"]]
                self.assertEqual(reference["model"], case["model"])
                self.assertLessEqual(case["required_gpus"], 8)
                for role in ("context", "generation"):
                    options = case[f"{role}_options"]
                    self.assertTrue(options["cuda_graph_config"])
                    self.assertTrue(options["enable_chunked_prefill"])
                    self.assertFalse(options["disable_overlap_scheduler"])
                    self.assertEqual(options["max_num_tokens"], 128)
                    cache = options["kv_cache_config"]
                    self.assertTrue(cache["enable_block_reuse"])
                    expected_ratio = None
                    if "explicit-pools" in name:
                        expected_ratio = [0.2, 0.7, 0.1] if name.startswith("deepseek") else [1.0]
                    self.assertEqual(cache.get("pool_ratio"), expected_ratio)
                    self.assertEqual(
                        reference["llm_options"]["kv_cache_config"].get("pool_ratio"),
                        expected_ratio,
                    )
                    if name.startswith("deepseek"):
                        self.assertTrue(options["enable_attention_dp"])
                        self.assertEqual(
                            cache["enable_swa_scratch_reuse"], "no-scratch" not in name
                        )
                    cp = options.get("context_parallel_size", 1)
                    self.assertEqual(cp, 1)
                    self.assertEqual(
                        options["tensor_parallel_size"] * cp,
                        reference["llm_options"]["tensor_parallel_size"],
                    )

    def test_aggregate_comparison_rejects_different_corpus_tokens_and_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _arguments(Path(temporary) / "output")
            args.aggregate_results = Path(temporary)
            case = _RUNNER._load_cases(args)["qwen3-8b-explicit-pools"]
            prompts, tokens = ["one", "two"], [[1], [2]]
            corpus = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
            manifest = {
                "mode": "agg",
                "num_prompts": 2,
                "prompt_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
                "prompt_format": args.prompt_format,
                "max_tokens": args.max_tokens,
                "warmup_prompts": args.warmup_prompts,
            }
            recipe = copy.deepcopy(case)
            recipe["llm_options"] = copy.deepcopy(case["context_options"])
            recipe["spec_options"] = recipe["llm_options"].pop("speculative_config")
            for key in ("cache_transceiver_config", "num_serve_frontends", "return_perf_metrics"):
                recipe["llm_options"].pop(key)
            result = {
                "status": "ok",
                "acceptance_length": 5.0,
                "recipe": recipe,
                "prompt_token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            }
            (args.aggregate_results / "manifest.json").write_text(json.dumps(manifest))
            path = args.aggregate_results / f"{case['aggregate_case']}.json"
            path.write_text(json.dumps(result))
            self.assertEqual(_RUNNER._aggregate_reference(args, case, prompts, tokens), result)
            with self.assertRaisesRegex(ValueError, "prompt_sha256"):
                _RUNNER._aggregate_reference(args, case, ["changed", "two"], tokens)
            with self.assertRaisesRegex(ValueError, "tokenized prompts"):
                _RUNNER._aggregate_reference(args, case, prompts, [[9], [2]])
            result["recipe"]["llm_options"]["kv_cache_config"]["pool_ratio"] = None
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "engine settings"):
                _RUNNER._aggregate_reference(args, case, prompts, tokens)
            result["recipe"]["llm_options"]["kv_cache_config"]["pool_ratio"] = [1.0]
            path.write_text(json.dumps(result))
            case["generation_options"]["kv_cache_config"]["enable_block_reuse"] = False
            with self.assertRaisesRegex(ValueError, "engine settings"):
                _RUNNER._aggregate_reference(args, case, prompts, tokens)

    def test_recipes_keep_per_worker_parallelism_and_dspark_on_both_workers(self) -> None:
        cases = _RUNNER._load_cases(_arguments(Path("/tmp/dspark-test-unused-output")))
        expected = {
            "qwen3-8b": (2, 1, 7),
            "deepseek-v4-flash-nvfp4": (8, 4, 5),
        }
        self.assertTrue(set(expected).issubset(cases))
        for name, (total, per_worker, draft_len) in expected.items():
            with self.subTest(case=name):
                case = cases[name]
                self.assertEqual(case["required_gpus"], total)
                self.assertEqual(case["max_draft_len"], draft_len)
                self.assertTrue(Path(case["model"]).is_absolute())
                self.assertTrue(Path(case["drafter"]).is_absolute())
                expected_spec = {
                    **case.get("spec_options", {}),
                    "decoding_type": "DSpark",
                    "speculative_model": case["drafter"],
                    "max_draft_len": draft_len,
                }
                for role in ("context", "generation"):
                    options = case[f"{role}_options"]
                    self.assertEqual(options["speculative_config"], expected_spec)
                    self.assertEqual(options["tensor_parallel_size"], per_worker)
                    self.assertEqual(options["moe_expert_parallel_size"], per_worker)
                    self.assertEqual(options["num_postprocess_workers"], 0)
                    self.assertEqual(options["num_serve_frontends"], 1)
                    self.assertTrue(options["return_perf_metrics"])
                    self.assertEqual(options["max_batch_size"], 1)
                    self.assertFalse(options["disable_overlap_scheduler"])
                    self.assertTrue(options["enable_chunked_prefill"])
                    self.assertEqual(options["max_num_tokens"], 128)
                    self.assertTrue(options["print_iter_log"])
                    self.assertEqual(options["max_seq_len"], 8192)
                    self.assertEqual(options["cache_transceiver_config"]["backend"], "NIXL")
                    self.assertEqual(
                        options["cache_transceiver_config"]["transceiver_runtime"], "PYTHON"
                    )

    def test_token_budget_only_limits_unchunked_prompts(self) -> None:
        args = _arguments(Path("/tmp/dspark-test-unused-output"))
        tokenizer_module = MagicMock()
        tokens = list(range(257))
        tokenizer_module.load_hf_tokenizer.return_value.apply_chat_template.return_value = tokens
        case = _RUNNER._load_cases(args)["qwen3-8b"]
        with patch.dict("sys.modules", {"tensorrt_llm.tokenizer": tokenizer_module}):
            self.assertEqual(_RUNNER._tokenize(args, case, ["long prompt"]), [tokens])
            for role in ("context", "generation"):
                with self.subTest(role=role):
                    unchunked = copy.deepcopy(case)
                    unchunked[f"{role}_options"]["enable_chunked_prefill"] = False
                    with self.assertRaisesRegex(ValueError, f"{role} unchunked prefill budget"):
                        _RUNNER._tokenize(args, unchunked, ["long prompt"])
                    too_short = copy.deepcopy(case)
                    too_short[f"{role}_options"]["max_seq_len"] = len(tokens) + args.max_tokens - 1
                    with self.assertRaisesRegex(ValueError, f"{role} sequence budget"):
                        _RUNNER._tokenize(args, too_short, ["long prompt"])

    def test_context_and_generation_speculative_configs_are_independent(self) -> None:
        cases = _RUNNER._load_cases(_arguments(Path("/tmp/dspark-test-unused-output")))
        for name, case in cases.items():
            with self.subTest(case=name):
                context = case["context_options"]["speculative_config"]
                generation = case["generation_options"]["speculative_config"]
                source = copy.deepcopy(case.get("spec_options", {}))
                original_generation = copy.deepcopy(generation)
                self.assertIsNot(context, generation)
                context["max_draft_len"] = 1
                self.assertEqual(generation, original_generation)
                self.assertEqual(case.get("spec_options", {}), source)

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

    def test_run_case_excludes_warmup_and_context_from_reported_generation_al(self) -> None:
        self.args.output_dir.mkdir()
        self.args.warmup_prompts = 1
        tokens = [[11], [22]]
        measured_requests = [
            {"index": 0, "finish_reason": "stop"},
            {"index": 1, "finish_reason": "stop"},
        ]
        snapshots = iter(
            [
                _counters({"0": 100, "1": 80}, {"0": 80, "1": 60}, {"not_finished": 1}),
                _counters({"0": 20, "1": 16}, {"0": 2, "1": 0}, {"stop": 1}),
                _counters({"0": 120, "1": 96}, {"0": 99, "1": 73}, {"not_finished": 3}),
                _counters({"0": 30, "1": 24}, {"0": 10, "1": 6}, {"stop": 3}),
            ]
        )
        events = []

        def workload(
            args: argparse.Namespace,
            url: str,
            model: str,
            prompts: list[list[int]],
            max_tokens: int,
            processes: list,
        ) -> list[dict]:
            events.append(("workload", max_tokens, len(prompts)))
            return [{"warmup": True}] if max_tokens == 8 else measured_requests

        def snapshot(url: str, path: Path) -> dict:
            events.append(("snapshot", path.name))
            return next(snapshots)

        with (
            patch.object(_RUNNER, "_tokenize", return_value=tokens),
            patch.object(
                _RUNNER,
                "_launch_servers",
                return_value=nullcontext(("router", "context", "generation", [])),
            ),
            patch.object(_RUNNER, "_workload", side_effect=workload),
            patch.object(_RUNNER, "_snapshot", side_effect=snapshot),
        ):
            result = _RUNNER._run_case(
                self.args, "qwen3-8b", self.cases["qwen3-8b"], ["0", "1"], ["one", "two"]
            )

        self.assertEqual(
            events,
            [
                ("workload", 8, 1),
                ("snapshot", "context-before.prom"),
                ("snapshot", "generation-before.prom"),
                ("workload", 256, 2),
                ("snapshot", "context-after.prom"),
                ("snapshot", "generation-after.prom"),
            ],
        )
        self.assertEqual(result["acceptance_length"], 2.4)
        self.assertAlmostEqual(result["acceptance_rate"], 14 / 18)
        self.assertEqual(result["verification_steps"], 10)
        self.assertEqual(result["accepted_draft_tokens"], 14)
        self.assertEqual(result["draft_tokens"], 18)
        self.assertEqual(result["warmup_requests"], 1)
        self.assertEqual(result["num_requests"], 2)
        self.assertEqual(result["requests"], measured_requests)
        self.assertEqual(
            result["context_counters"],
            _counters({"0": 20, "1": 16}, {"0": 19, "1": 13}, {"not_finished": 2}),
        )
        self.assertEqual(
            result["generation_counters"],
            _counters({"0": 10, "1": 8}, {"0": 8, "1": 6}, {"stop": 2}),
        )
        self.assertEqual(
            result["metric_source"], "generation completed-request per-position Prometheus counters"
        )
        self.assertEqual(
            result["al_definition"],
            "1 + sum(accepted_draft_tokens) / verification_steps; includes disagg bootstrap",
        )
        self.assertEqual(json.loads((self.args.output_dir / "qwen3-8b.json").read_text()), result)

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
            list(self.cases),
        )
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(
            [row["status"] for row in summary["results"]],
            ["ok"] * len(self.cases),
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
        self.assertEqual(
            [call.args[1] for call in run_case.call_args_list],
            [name for name, case in self.cases.items() if case["required_gpus"] <= 2],
        )
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(
            [row["status"] for row in summary["results"]],
            [
                "ok" if case["required_gpus"] <= 2 else "skipped_insufficient_gpus"
                for case in self.cases.values()
            ],
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
                ]
                + [{"status": "ok", "acceptance_length": 2.0}] * (len(self.cases) - 1),
            ) as run_case,
            patch.object(Path, "is_file", return_value=True),
            redirect_stdout(io.StringIO()),
        ):
            code = _RUNNER._run_cases(self.args, self.cases, list(map(str, range(8))))
        self.assertNotEqual(code, 0)
        self.assertEqual(run_case.call_count, len(self.cases))
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
        import yaml

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
        for role in ("context", "generation"):
            with self.subTest(role=role):
                worker_config = yaml.safe_load((case_dir / f"{role}.yaml").read_text())
                expected_spec = self.cases["qwen3-8b"][f"{role}_options"]["speculative_config"]
                self.assertEqual(worker_config["speculative_config"], expected_spec)
                self.assertEqual(worker_config["speculative_config"]["decoding_type"], "DSpark")


if __name__ == "__main__":
    unittest.main()
