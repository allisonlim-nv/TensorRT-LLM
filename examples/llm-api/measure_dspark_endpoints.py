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
"""Run pinned Endpoints measurements using the existing DSpark disagg launcher.

Run on the host with Python's standard library. The serve subcommand runs inside
the existing TRT-LLM container; the client runs in a separate pinned image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMAGE = "ghcr.io/mlcommons/endpoints:89904fb"
CONTAINER_ROOT = Path("/code/tensorrt_llm")
PYTHON = "/code/tensorrt_llm/.venv-3.12/bin/python3"
MODEL_ROOT = "/home/scratch.trt_llm_data_ci/llm-models"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("dspark_disagg_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serve(args: argparse.Namespace) -> None:
    """Launch owned workers and service counter-snapshot requests via files."""
    runner = load_runner(args.runner)
    options = argparse.Namespace(
        config=args.config,
        models_root=Path(args.models_root),
        case=["qwen3-8b"],
        prompt_format="chat",
        max_tokens=1024,
        startup_timeout=1200,
        request_timeout=1800,
    )
    case = runner._load_cases(options)["qwen3-8b"]
    if case["required_gpus"] != 2:
        raise ValueError("This collector requires one TP1 worker per role")
    for role in ("context", "generation"):
        worker = case[f"{role}_options"]
        worker["kv_cache_config"]["use_kv_cache_manager_v2"] = True
        worker.setdefault("stream_interval", 1)
        if worker["speculative_config"]["decoding_type"] != "DSpark":
            raise ValueError("Both workers must enable DSpark")
    write_json(args.output / "recipe.json", case)
    prompts = [json.loads(line)["prompt"] for line in args.prompts.read_text().splitlines()]
    if len(prompts) != 1319:
        raise ValueError(f"Expected the complete 1319-prompt corpus, got {len(prompts)}")
    tokens = runner._tokenize(options, case, prompts)
    from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2 as cpp
    from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime

    if runtime.KVCacheManager is not cpp.KVCacheManager:
        raise RuntimeError("KVCache V2 did not resolve to the default C++ binding")
    write_json(
        args.output / "validation.json",
        {
            "kv_cache_v2_class": str(runtime.KVCacheManager),
            "cpp_class_identity": True,
            "prompt_count": len(tokens),
            "longest_prompt_tokens": max(map(len, tokens)),
            "max_output_tokens": 1024,
            "context_limits": {
                role: case[f"{role}_options"]["max_seq_len"] for role in ("context", "generation")
            },
            "chat_template_kwargs": case["chat_template_kwargs"],
            "environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("TRTLLM_", "TLLM_", "CUDA_VISIBLE", "UCX_"))
            },
        },
    )
    for name, count in (("warmup", 2), ("measured", args.count)):
        with (args.output / f"{name}.jsonl").open("x") as handle:
            for prompt, ids in zip(prompts[:count], tokens[:count]):
                handle.write(json.dumps({"prompt": prompt, "input_tokens": ids}) + "\n")
    server_dir = args.output / "server"
    server_dir.mkdir()
    with runner._launch_servers(options, case, args.devices.split(","), server_dir) as (
        router,
        context,
        generation,
        processes,
    ):
        write_json(
            args.output / "ready.json",
            {
                "router": router,
                "context": context,
                "generation": generation,
                "model": case["model"],
                "pids": [p.pid for p in processes],
            },
        )
        snapshots = {}
        while not (args.output / "STOP").exists():
            runner._check_processes(processes)
            for boundary in ("before", "after", "diagnostic"):
                if boundary in snapshots or not (args.output / f"{boundary}.request").exists():
                    continue
                snapshots[boundary] = {}
                for role, url in (("context", context), ("generation", generation)):
                    path = args.output / f"{role}-{boundary}.prom"
                    counters = runner._snapshot(url, path)
                    # One rank and one frontend: reject duplicated logical counters.
                    keys = []
                    for line in path.read_text().splitlines():
                        match = re.match(
                            r"(trtllm_spec_decode_(?:drafted|accepted)_tokens_total|"
                            r"trtllm_request_success_total)\{([^}]+)\} ",
                            line,
                        )
                        if match:
                            label = re.search(
                                r'(?:token_position|finished_reason)="([^"]+)"', match[2]
                            )
                            keys.append((match[1], label[1]))
                    if len(keys) != len(set(keys)):
                        raise RuntimeError("Duplicate rank/frontend counters; refusing to sum")
                    snapshots[boundary][role] = counters
                if boundary == "after":
                    summary = runner._summarize_counters(
                        runner._counter_delta(
                            snapshots["before"]["context"], snapshots["after"]["context"]
                        ),
                        runner._counter_delta(
                            snapshots["before"]["generation"], snapshots["after"]["generation"]
                        ),
                        args.count,
                        case["max_draft_len"],
                    )
                    write_json(args.output / "counters.json", summary)
                write_json(args.output / f"{boundary}.done", snapshots[boundary])
            time.sleep(0.2)


def wait_file(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Server launcher exited ({process.returncode}); see launcher.log")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(1)


def client_config(directory: Path, ready: dict, count: int, phase: str) -> dict:
    return {
        "name": f"qwen3-8b-dspark-{phase}",
        "version": "1.0",
        "type": "online",
        "model_params": {
            "name": ready["model"],
            "tokenizer_name": ready["model"],
            "temperature": 0.0,
            "max_new_tokens": 1024,
            "min_new_tokens": 0,
            "streaming": "on",
        },
        "datasets": [
            {
                "name": "qwen-local",
                "type": "performance",
                "path": str(directory / f"{phase}.jsonl"),
                "samples": count,
            }
        ],
        "settings": {
            "runtime": {
                "n_samples_to_issue": count,
                "dataloader_random_seed": 42,
                "scheduler_random_seed": 42,
            },
            "load_pattern": {"type": "concurrency", "target_concurrency": 4},
            "client": {
                "num_workers": 1,
                "warmup_connections": 4,
                "max_connections": 4,
                "min_required_connections": 4,
            },
            "warmup": {"enabled": False},
            "early_stopping": {"enabled": False},
            "timeouts": {"run_timeout_s": 7200},
        },
        "endpoint_config": {"endpoints": [ready["router"]], "api_type": "openai_completions"},
        "report_dir": str(directory / f"{phase}-report"),
        "enable_cpu_affinity": False,
    }


def summarize(directory: Path, expected: int) -> dict:
    report = json.loads((directory / "measured-report/performance/result_summary.json").read_text())
    counters = json.loads((directory / "counters.json").read_text())
    if (
        not report["complete"]
        or report["n_samples_failed"]
        or report["n_samples_completed"] != expected
    ):
        raise RuntimeError(
            "Incomplete/failed workload; raw reports retained, no valid AL comparison"
        )
    duration = report["duration_ns"] / 1e9
    osl = report["output_sequence_lengths"]
    row = {
        "phase": directory.name,
        "AL": counters["acceptance_length"],
        "TTFT_ms": report["ttft"]["avg"] / 1e6,
        "TPOT_ms": report["tpot"]["avg"] / 1e6,
        "E2E_interactivity": report["e2e_avg_interactivity"],
        "Output_TPS_per_GPU": osl["total"] / duration / 2,
        "Mean_OSL": osl["avg"],
        "output_tokens": osl["total"],
        "completed_requests": report["n_samples_completed"],
        "failures": report["n_samples_failed"],
        "measurement_duration_s": duration,
        "accepted_draft_tokens": counters["accepted_draft_tokens"],
        "verification_steps": counters["verification_steps"],
    }
    for metric in ("ttft", "tpot"):
        for percentile in (50, 95):
            values = report[metric]["percentiles"]
            value = values.get(str(percentile), values.get(f"{percentile}.0"))
            if value is None:
                raise ValueError(f"Client report lacks {metric} p{percentile}")
            row[f"{metric.upper()}_p{percentile}_ms"] = value / 1e6
    write_json(directory / "summary.json", row)
    return row


def run(args: argparse.Namespace) -> None:
    """Run smoke and full phases with fresh owned servers and identical warmup."""
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def mapped(path: Path) -> Path:
        return CONTAINER_ROOT / path.resolve().relative_to(root)

    runner = args.runner.resolve()
    config = args.config.resolve()
    prompts = args.prompts.resolve()
    for path in (runner, config, prompts):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in (
        ("runner.py", runner),
        ("server-recipe.yaml", config),
        ("prompts.jsonl", prompts),
        ("collector.py", Path(__file__)),
    ):
        shutil.copyfile(path, output / name)
    identity = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prompt_sha256": sha256(prompts),
        "runner_sha256": sha256(runner),
        "config_sha256": sha256(config),
        "collector_sha256": sha256(Path(__file__)),
        "image": IMAGE,
        "container": args.container,
        "devices": args.devices,
        "model_root": args.models_root,
        "command": sys.argv,
        "image_inspect": json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE])),
    }
    for name, command in (
        ("source.patch", ["git", "diff", "HEAD", "--binary"]),
        ("git-status.txt", ["git", "status", "--short"]),
        ("gpu-before.txt", ["nvidia-smi"]),
    ):
        (output / name).write_bytes(subprocess.check_output(command))
    identity["source_diff_sha256"] = sha256(output / "source.patch")
    write_json(output / "identity.json", identity)
    rows = []
    for name, count in (("smoke", 8), ("full", 1319)):
        if name == "full" and args.smoke_only:
            break
        busy = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.devices}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if busy:
            raise RuntimeError(f"Selected GPUs are busy; existing workloads untouched: {busy}")
        directory = output / name
        directory.mkdir()
        command = [
            "docker",
            "exec",
            "-w",
            str(CONTAINER_ROOT),
            args.container,
            PYTHON,
            str(mapped(Path(__file__))),
            "serve",
            "--runner",
            str(mapped(runner)),
            "--config",
            str(mapped(config)),
            "--prompts",
            str(mapped(prompts)),
            "--output",
            str(mapped(directory)),
            "--count",
            str(count),
            "--devices",
            args.devices,
            "--models-root",
            args.models_root,
        ]
        print(f"Starting {name}: {count} requests; logs: {directory}", flush=True)
        with (directory / "launcher.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_file(directory / "ready.json", process, 1500)
                ready = json.loads((directory / "ready.json").read_text())
                for phase, n_requests in (("warmup", 2), ("measured", count)):
                    if phase == "measured":
                        (directory / "before.request").touch()
                        wait_file(directory / "before.done", process, 60)
                    cfg = client_config(mapped(directory), ready, n_requests, phase)
                    cfg_path = directory / f"{phase}-client.yaml"
                    write_json(cfg_path, cfg)
                    client_command = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        f"container:{args.container}",
                        "--user",
                        f"{os.getuid()}:{os.getgid()}",
                        "-e",
                        "HF_HOME=/tmp/hf",
                        "-e",
                        "HOME=/tmp",
                        "-v",
                        f"{root}:{CONTAINER_ROOT}",
                        "-v",
                        f"{args.models_root}:{args.models_root}:ro",
                        IMAGE,
                        "inference-endpoint",
                        "benchmark",
                        "from-config",
                        "--config",
                        str(mapped(cfg_path)),
                    ]
                    write_json(directory / f"{phase}-command.json", client_command)
                    with (directory / f"{phase}-client.log").open("w") as client_log:
                        subprocess.run(
                            client_command,
                            stdout=client_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=7500,
                        )
                    report_path = directory / f"{phase}-report/performance/result_summary.json"
                    report = json.loads(report_path.read_text())
                    if (
                        not report["complete"]
                        or report["n_samples_failed"]
                        or report["n_samples_completed"] != n_requests
                    ):
                        raise RuntimeError(f"{phase} failed; see {report_path}")
                (directory / "after.request").touch()
                wait_file(directory / "after.done", process, 60)
                rows.append(summarize(directory, count))
                with (output / "summary.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(rows[-1]), flush=True)
            finally:
                if (directory / "ready.json").exists() and process.poll() is None:
                    (directory / "diagnostic.request").touch()
                    try:
                        wait_file(directory / "diagnostic.done", process, 40)
                    except (TimeoutError, RuntimeError):
                        print("Final diagnostic snapshot unavailable; see server logs", flush=True)
                (directory / "STOP").touch()
                process.wait(timeout=1500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "serve"))
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--models-root", default=MODEL_ROOT)
    parser.add_argument("--container", default="tensorrt_llm-devel-allim")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.devices.split(","))) != 2:
        parser.error("Specify exactly two distinct GPU IDs")
    if args.mode == "serve":
        serve(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
