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
"""Measure corpus acceptance length for DSpark drafters using aggregated LLM inference.

Run all recipes in dspark_acceptance.yaml, or select recipes with --case.
Each recipe runs in a fresh process and produces a log and JSON result. See
dspark_acceptance.md for hardware requirements and multi-node launch examples.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi.llm import RequestOutput


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("dspark_acceptance.yaml")
    )
    parser.add_argument("--models-root", type=Path, default=os.environ.get("LLM_MODELS_ROOT"))
    parser.add_argument(
        "--case", action="append", help="Recipe to run; repeat to select several (default: all)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dspark-al"), help="New directory for this run."
    )
    parser.add_argument("--num-prompts", type=_positive_int, default=64)
    parser.add_argument("--max-tokens", type=_positive_int, default=256)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSONL with one nonempty 'prompt' string per line; default: GSM8K test.",
    )
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=2,
        help="Warmup requests excluded from AL (0 disables).",
    )
    parser.add_argument(
        "--launcher",
        default="",
        help="Command prefix, e.g. 'srun --ntasks={gpus} --mpi=pmix trtllm-llmapi-launch'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved recipes/commands without loading models or datasets.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.models_root is None:
        parser.error("set LLM_MODELS_ROOT or pass --models-root")
    if args.warmup_prompts < 0:
        parser.error("--warmup-prompts must be nonnegative")
    args.models_root = args.models_root.expanduser().resolve()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.prompt_file is not None:
        args.prompt_file = args.prompt_file.resolve()
    return args


def _merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_cases(args: argparse.Namespace) -> dict[str, dict]:
    import yaml

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    names = list(config["cases"]) if not args.case or args.case == ["all"] else args.case
    if len(names) != len(set(names)):
        raise ValueError("A recipe may only be selected once")
    cases = {}
    for name in names:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name not in config["cases"]:
            raise ValueError(
                f"Unknown or invalid case {name!r}; choices: {', '.join(config['cases'])}"
            )
        case = _merge(config.get("defaults", {}), config["cases"][name])
        for key in ("model", "drafter"):
            case[key] = str((args.models_root / case[key]).resolve())
        options = case.setdefault("llm_options", {})
        if options.get("cache_transceiver_config") is not None:
            raise ValueError(f"{name}: cache_transceiver_config is not allowed in an aggregate run")
        if options.get("num_postprocess_workers", 0) != 0:
            raise ValueError(
                f"{name}: request acceptance counters require num_postprocess_workers=0"
            )
        if any(key in options for key in ("model", "speculative_config")):
            raise ValueError(f"{name}: set model/drafter/max_draft_len at the recipe level")
        if options.get("backend", "pytorch") != "pytorch":
            raise ValueError(f"{name}: DSpark requires the pytorch backend")
        options.update(backend="pytorch", num_postprocess_workers=0)
        case["spec_options"] = _merge(
            case.get("spec_options", {}),
            {
                "decoding_type": "DSpark",
                "speculative_model": case["drafter"],
                "max_draft_len": case["max_draft_len"],
            },
        )
        cases[name] = case
    return cases


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file is not None:
        prompts = []
        with args.prompt_file.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                prompts.append(json.loads(line)["prompt"])
                if len(prompts) == args.num_prompts:
                    break
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [
            row["question"] for row in dataset.select(range(min(args.num_prompts, len(dataset))))
        ]
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("The dataset must contain nonempty prompt strings")
    return prompts


def _format_prompts(
    llm: LLM, prompts: list[str], case: dict, prompt_format: str
) -> list[list[int]]:
    tokenizer = llm.tokenizer
    if tokenizer is None:
        raise RuntimeError("A tokenizer is required to prepare the prompt corpus")
    tokenized = []
    for prompt in prompts:
        if prompt_format == "raw":
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            messages = []
            if case.get("system_prompt"):
                messages.append({"role": "system", "content": case["system_prompt"]})
            messages.append({"role": "user", "content": prompt})
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **case.get("chat_template_kwargs", {}),
            )
        tokenized.append(tokens)
    return tokenized


def _summarize_requests(outputs: list[RequestOutput]) -> dict:
    """Aggregate final request counters, including requests on other attention-DP ranks.

    Position zero counts actual draft verification steps. Prefill-only requests
    contribute zero; the warmup results must not be included in ``outputs``.
    Counts describe verification before EOS/output-length truncation.
    """
    requests = []
    for index, output in enumerate(outputs):
        if not output.finished:
            raise RuntimeError(f"Request {index} has not finished")
        positions = output.per_pos_drafted
        if positions is None or len(positions) == 0:
            raise RuntimeError(f"Request {index} is missing speculative verification counters")
        steps = int(positions[0])
        accepted, drafted = output.spec_dec_totals or (0, 0)
        accepted, drafted = int(accepted), int(drafted)
        if steps < 0 or not 0 <= accepted <= drafted or (drafted == 0) != (steps == 0):
            raise RuntimeError(f"Request {index} has inconsistent speculative counters")
        requests.append(
            {
                "index": index,
                "accepted_draft_tokens": accepted,
                "draft_tokens": drafted,
                "verification_steps": steps,
                "acceptance_length": 1 + accepted / steps if steps else None,
                "output_tokens": len(output.outputs[0].token_ids),
            }
        )
    accepted = sum(row["accepted_draft_tokens"] for row in requests)
    drafted = sum(row["draft_tokens"] for row in requests)
    steps = sum(row["verification_steps"] for row in requests)
    if steps == 0:
        raise RuntimeError(
            "No draft verification occurred; increase --max-tokens or inspect the drafter"
        )
    al = 1 + accepted / steps
    if not math.isfinite(al):
        raise RuntimeError("Non-finite acceptance length")
    return {
        "acceptance_length": al,
        "acceptance_rate": accepted / drafted,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "verification_steps": steps,
        "num_requests": len(requests),
        "requests_without_drafts": sum(row["verification_steps"] == 0 for row in requests),
        "total_output_tokens": sum(row["output_tokens"] for row in requests),
        "requests": requests,
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _run_worker(args: argparse.Namespace, name: str, case: dict) -> None:
    from tensorrt_llm import LLM, SamplingParams

    prompts = _load_prompts(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    with LLM(
        model=case["model"], speculative_config=case["spec_options"], **case["llm_options"]
    ) as llm:
        tokenized = _format_prompts(llm, prompts, case, args.prompt_format)
        longest = max(map(len, tokenized))
        if longest + args.max_tokens > llm.args.max_seq_len:
            raise ValueError(
                f"Prompt length {longest} + --max-tokens exceeds max_seq_len={llm.args.max_seq_len}"
            )
        if not llm.args.enable_chunked_prefill and longest > llm.args.max_num_tokens:
            raise ValueError(f"Prompt length {longest} exceeds the unchunked prefill token budget")
        if args.warmup_prompts:
            llm.generate(
                tokenized[: args.warmup_prompts],
                SamplingParams(temperature=0.0, max_tokens=min(8, args.max_tokens)),
                use_tqdm=False,
            )
        start = time.monotonic()
        outputs = llm.generate(tokenized, sampling_params, use_tqdm=True)
        elapsed = time.monotonic() - start
        if len(outputs) != len(prompts):
            raise RuntimeError("Generation did not return the entire prompt corpus")
        summary = _summarize_requests(outputs)
    result = {
        "case": name,
        "status": "ok",
        "mode": "agg",
        "recipe": case,
        "prompt_format": args.prompt_format,
        "max_tokens": args.max_tokens,
        "warmup_requests": min(args.warmup_prompts, len(prompts)),
        "elapsed_seconds": elapsed,
        "al_definition": "1 + sum(accepted_draft_tokens) / sum(verification_steps)",
        **summary,
    }
    _write_json(args.output_dir / f"{name}.json", result)
    print(
        f"{name}: AL={summary['acceptance_length']:.4f}, AR={summary['acceptance_rate']:.4f}",
        flush=True,
    )


def _command(args: argparse.Namespace, name: str, case: dict) -> list[str]:
    options = case["llm_options"]
    gpus = options.get("tensor_parallel_size", 1) * options.get("pipeline_parallel_size", 1)
    # Parse before substituting so quoted paths containing spaces remain intact.
    command = [part.replace("{gpus}", str(gpus)) for part in shlex.split(args.launcher)]
    command += [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        name,
        "--config",
        str(args.config),
        "--models-root",
        str(args.models_root),
        "--output-dir",
        str(args.output_dir),
        "--prompt-file",
        str(args.output_dir / "prompts.jsonl"),
        "--num-prompts",
        str(args.num_prompts),
        "--max-tokens",
        str(args.max_tokens),
        "--prompt-format",
        args.prompt_format,
        "--warmup-prompts",
        str(args.warmup_prompts),
    ]
    return command


def _write_summary(output_dir: Path, rows: list[dict]) -> None:
    _write_json(output_dir / "summary.json", {"mode": "agg", "results": rows})
    columns = [
        "case",
        "status",
        "acceptance_length",
        "acceptance_rate",
        "verification_steps",
        "accepted_draft_tokens",
        "draft_tokens",
        "num_requests",
        "error",
        "log",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_cases(args: argparse.Namespace, cases: dict[str, dict]) -> int:
    if args.dry_run:
        for name, case in cases.items():
            print(json.dumps({"case": name, "mode": "agg", "recipe": case}, indent=2))
            print(shlex.join(_command(args, name, case)))
        return 0
    # A new directory prevents stale results from being mistaken for this run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompts = _load_prompts(args)
    corpus = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
    (args.output_dir / "prompts.jsonl").write_text(corpus, encoding="utf-8")
    _write_json(
        args.output_dir / "manifest.json",
        {
            "mode": "agg",
            "cases": cases,
            "num_prompts": len(prompts),
            "prompt_source": str(args.prompt_file)
            if args.prompt_file
            else "openai/gsm8k:main:test",
            "prompt_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
            "prompt_format": args.prompt_format,
            "max_tokens": args.max_tokens,
            "launcher": args.launcher,
        },
    )
    rows = []
    child_env = dict(os.environ, LLM_MODELS_ROOT=str(args.models_root))
    for name, case in cases.items():
        log = args.output_dir / f"{name}.log"
        row = {"case": name, "status": "failed", "log": str(log)}
        missing = [
            case[key]
            for key in ("model", "drafter")
            if not (Path(case[key]) / "config.json").is_file()
        ]
        if missing:
            row.update(
                status="missing_checkpoint",
                error=f"Missing config.json: {', '.join(dict.fromkeys(missing))}",
            )
        else:
            command = _command(args, name, case)
            print(f"Running {name}; log: {log}\n{shlex.join(command)}", flush=True)
            with log.open("w") as handle:
                try:
                    completed = subprocess.run(
                        command, env=child_env, stdout=handle, stderr=subprocess.STDOUT, check=False
                    )
                except OSError as error:
                    row["error"] = str(error)
                else:
                    result_path = args.output_dir / f"{name}.json"
                    if completed.returncode:
                        row["error"] = f"Worker exited with code {completed.returncode}; see log"
                    elif not result_path.is_file():
                        row["error"] = "Worker exited without producing results; see log"
                    else:
                        with result_path.open() as result_file:
                            result = json.load(result_file)
                        row.update(
                            {
                                key: value
                                for key, value in result.items()
                                if key not in ("requests", "recipe")
                            }
                        )
        rows.append(row)
        _write_summary(args.output_dir, rows)
        print(
            f"{name}: {row['status']}  AL={row.get('acceptance_length', 'n/a')} {row.get('error', '')}",
            flush=True,
        )
    print(f"Results: {args.output_dir / 'summary.csv'}", flush=True)
    return int(any(row["status"] != "ok" for row in rows))


def _main() -> int:
    args = _parse_arguments()
    cases = _load_cases(args)
    if args.worker:
        if len(cases) != 1 or args.dry_run:
            raise ValueError("A worker must run exactly one case")
        name, case = next(iter(cases.items()))
        _run_worker(args, name, case)
        return 0
    return _run_cases(args, cases)


if __name__ == "__main__":
    sys.exit(_main())
