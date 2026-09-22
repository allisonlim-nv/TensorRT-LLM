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
"""Measure DSpark acceptance on separate, local prefill and decode GPU pools.

Launches a private disaggregated router and two workers per case. Acceptance
comes from completed-request, per-position Prometheus counters, not iteration
statistics. See dspark_acceptance_disagg.md. The aggregate runner is independent.
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
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

_DRAFTED = "trtllm_spec_decode_drafted_tokens_total"
_ACCEPTED = "trtllm_spec_decode_accepted_tokens_total"
_SUCCESS = "trtllm_request_success_total"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("dspark_acceptance_disagg.yaml")
    )
    parser.add_argument("--models-root", type=Path, default=os.environ.get("LLM_MODELS_ROOT"))
    parser.add_argument("--output-dir", type=Path, default=Path("dspark-al-disagg"))
    parser.add_argument("--case", action="append", help="Repeat to select cases; default: all.")
    parser.add_argument(
        "--devices", help="Local GPU IDs/UUIDs; defaults to CUDA_VISIBLE_DEVICES or nvidia-smi."
    )
    parser.add_argument("--num-prompts", type=_positive_int, default=64)
    parser.add_argument("--max-tokens", type=_positive_int, default=256)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSONL records containing a prompt string; default: GSM8K test.",
    )
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--concurrency", type=_positive_int, default=8)
    parser.add_argument("--startup-timeout", type=_positive_int, default=3600)
    parser.add_argument("--request-timeout", type=_positive_int, default=1800)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved recipes and GPU allocation; do not start servers.",
    )
    args = parser.parse_args()
    if args.models_root is None:
        parser.error("set LLM_MODELS_ROOT or pass --models-root")
    if args.warmup_prompts < 0:
        parser.error("--warmup-prompts must be nonnegative")
    for field in ("config", "models_root", "output_dir", "prompt_file"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    return args


def _merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def _gpu_count(options: dict) -> int:
    count = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
        value = options.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid {key}: {value}")
        count *= value
    return count


def _load_cases(args: argparse.Namespace) -> dict[str, dict]:
    import yaml

    config = yaml.safe_load(args.config.read_text())
    names = list(config["cases"]) if not args.case or args.case == ["all"] else args.case
    if len(names) != len(set(names)):
        raise ValueError("A case may only be selected once")
    cases = {}
    for name in names:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name not in config["cases"]:
            raise ValueError(f"Unknown case {name}; choices: {', '.join(config['cases'])}")
        case = _merge(config.get("defaults", {}), config["cases"][name])
        for key in ("model", "drafter"):
            case[key] = str((args.models_root / case[key]).resolve())
        draft_len = case["max_draft_len"]
        if type(draft_len) is not int or not 1 <= draft_len <= 16:
            raise ValueError("max_draft_len must be in [1, 16] for exact per-position counters")
        for role in ("context", "generation"):
            options = _merge(case.get("common_options", {}), case.get(f"{role}_options", {}))
            forbidden = {
                "model",
                "speculative_config",
                "disagg_cluster",
                "internal_request_auth_key",
                "post_processor_hook",
            }
            if forbidden.intersection(options):
                raise ValueError(f"{name}: {role}_options contain runner-managed fields")
            if options.get("backend", "pytorch") != "pytorch":
                raise ValueError("DSpark requires backend=pytorch")
            for key, required in (("num_postprocess_workers", 0), ("num_serve_frontends", 1)):
                if options.get(key, required) != required:
                    raise ValueError(
                        f"{name}: requires num_postprocess_workers=0 and num_serve_frontends=1"
                    )
            transfer = options.get("cache_transceiver_config", {})
            if transfer.get("backend") != "NIXL" or transfer.get("transceiver_runtime") != "PYTHON":
                raise ValueError("This runner requires the NIXL Python transceiver on both workers")
            options.update(
                backend="pytorch",
                return_perf_metrics=True,
                num_postprocess_workers=0,
                num_serve_frontends=1,
            )
            options["speculative_config"] = _merge(
                case.get("spec_options", {}),
                {
                    "decoding_type": "DSpark",
                    "speculative_model": case["drafter"],
                    "max_draft_len": draft_len,
                },
            )
            case[f"{role}_options"] = options
        if (
            case["context_options"]["cache_transceiver_config"]
            != case["generation_options"]["cache_transceiver_config"]
        ):
            raise ValueError("Context and generation transceiver settings must match")
        case["required_gpus"] = sum(
            _gpu_count(case[f"{role}_options"]) for role in ("context", "generation")
        )
        cases[name] = case
    return cases


def _get_devices(explicit: str | None) -> list[str]:
    value = explicit if explicit is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        value = ",".join(result.stdout.split())
    if value.strip() in ("", "-1"):
        return []
    devices = [part.strip() for part in value.split(",")]
    if all(re.fullmatch(r"[0-9]+", device) for device in devices):
        devices = [str(int(device)) for device in devices]
    elif all(
        re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device)
        for device in devices
    ):
        devices = ["GPU-" + device[4:].lower() for device in devices]
    else:
        raise ValueError(
            "Use only local GPU indices or only full GPU UUIDs; mixed aliases and MIG are not supported"
        )
    if len(devices) != len(set(devices)):
        raise ValueError("Provide distinct GPUs; prefill and decode pools must not overlap")
    return devices


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = []
        with args.prompt_file.open() as handle:
            for line in handle:
                if line.strip():
                    prompts.append(json.loads(line)["prompt"])
                if len(prompts) == args.num_prompts:
                    break
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [
            row["question"] for row in dataset.select(range(min(args.num_prompts, len(dataset))))
        ]
    if len(prompts) != args.num_prompts or any(
        not isinstance(p, str) or not p.strip() for p in prompts
    ):
        raise ValueError(f"Need exactly {args.num_prompts} nonempty prompts; got {len(prompts)}")
    return prompts


def _format_prompts(
    tokenizer, prompts: list[str], case: dict, prompt_format: str
) -> list[list[int]]:
    result = []
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
        result.append(tokens)
    return result


def _tokenize(args: argparse.Namespace, case: dict, prompts: list[str]) -> list[list[int]]:
    from tensorrt_llm.tokenizer import load_custom_tokenizer, load_hf_tokenizer

    options = case["context_options"]
    kwargs = {
        "trust_remote_code": options.get("trust_remote_code", False),
        "use_fast": options.get("tokenizer_mode", "auto") != "slow",
    }
    if options.get("custom_tokenizer"):
        tokenizer = load_custom_tokenizer(options["custom_tokenizer"], case["model"], **kwargs)
    else:
        tokenizer = load_hf_tokenizer(case["model"], **kwargs)
    if tokenizer is None:
        raise RuntimeError("Failed to load the target tokenizer")
    tokens = _format_prompts(tokenizer, prompts, case, args.prompt_format)
    longest = max(map(len, tokens))
    for role in ("context", "generation"):
        limits = case[f"{role}_options"]
        if longest + args.max_tokens > limits["max_seq_len"] or longest > limits["max_num_tokens"]:
            raise ValueError(
                f"Prompt/output exceeds the {role} sequence or unchunked prefill budget"
            )
    return tokens


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _http(url: str, timeout: int, payload: dict | None = None) -> str:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        # All endpoints belong to this local benchmark; never send them through
        # an inherited HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code} from {url}: {error.read().decode()[:2000]}"
        ) from error


def _parse_metrics(body: str) -> dict[str, dict[str, int]]:
    from prometheus_client.parser import text_string_to_metric_families

    result: dict[str, dict[str, int]] = {"drafted": {}, "accepted": {}, "completed": {}}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name not in (_DRAFTED, _ACCEPTED, _SUCCESS):
                continue
            if not math.isfinite(sample.value) or sample.value < 0 or not sample.value.is_integer():
                raise ValueError(f"Invalid counter {sample.name}: {sample.value}")
            if sample.name == _SUCCESS:
                kind, label = "completed", sample.labels.get("finished_reason")
            else:
                kind = "drafted" if sample.name == _DRAFTED else "accepted"
                label = sample.labels.get("token_position")
                if label is None or not label.isdigit():
                    raise ValueError(f"Invalid token_position {label}")
            if not label:
                raise ValueError(f"Missing required labels for {sample.name}")
            result[kind][label] = result[kind].get(label, 0) + int(sample.value)
    return result


def _counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for kind in ("drafted", "accepted", "completed"):
        result[kind] = {}
        for label in before[kind].keys() | after[kind].keys():
            delta = after[kind].get(label, 0) - before[kind].get(label, 0)
            if delta < 0:
                raise RuntimeError("Worker counters reset/disappeared during measurement")
            result[kind][label] = delta
    return result


def _validate_speculative_counters(counters: dict, draft_len: int) -> None:
    drafted, accepted = counters["drafted"], counters["accepted"]
    for pos in drafted.keys() | accepted.keys():
        if not 0 <= int(pos) < draft_len or not 0 <= accepted.get(pos, 0) <= drafted.get(pos, 0):
            raise RuntimeError("Invalid per-position speculative counters")
    if any(drafted.get(str(pos), 0) > drafted.get(str(pos - 1), 0) for pos in range(1, draft_len)):
        raise RuntimeError("Drafted-position counters must be non-increasing")


def _summarize_counters(context: dict, generation: dict, num_requests: int, draft_len: int) -> dict:
    ctx = context["completed"]
    gen = generation["completed"]
    if sum(ctx.values()) != num_requests:
        raise RuntimeError("Context completion count does not match the measured corpus")
    if any(
        value and reason not in ("stop", "length", "not_finished") for reason, value in ctx.items()
    ):
        raise RuntimeError(f"Unexpected context finish reasons: {ctx}")
    # Both workers enable DSpark, but AL remains generation-only. Context
    # counters are retained for diagnostics, never added to generation AL.
    _validate_speculative_counters(context, draft_len)
    expected_gen = ctx.get("length", 0) + ctx.get("not_finished", 0)
    if sum(gen.values()) != expected_gen or any(
        value and reason not in ("stop", "length") for reason, value in gen.items()
    ):
        raise RuntimeError(
            f"Generation completion count/reasons do not match handoffs: {gen}; expected {expected_gen}"
        )
    drafted, accepted = generation["drafted"], generation["accepted"]
    _validate_speculative_counters(generation, draft_len)
    steps = drafted.get("0", 0)
    if not steps:
        raise RuntimeError("No draft verification occurred; no DSpark AL can be measured")
    if draft_len > 1 and not any(value > 0 and int(pos) > 0 for pos, value in drafted.items()):
        raise RuntimeError(
            "Missing multi-position counters; refusing a possibly aggregate-only metric"
        )
    accepted_total, draft_total = sum(accepted.values()), sum(drafted.values())
    return {
        "acceptance_length": 1 + accepted_total / steps,
        "acceptance_rate": accepted_total / draft_total,
        "verification_steps": steps,
        "accepted_draft_tokens": accepted_total,
        "draft_tokens": draft_total,
        "num_requests": num_requests,
        "generation_requests": expected_gen,
        "context_only_requests": ctx.get("stop", 0),
        "context_counters": context,
        "generation_counters": generation,
    }


def _check_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server process {process.pid} exited with code {process.returncode}; see server logs"
            )


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    # Each process is launched in its own session. Signal only those owned groups,
    # including MPI descendants, never unrelated servers or the caller's shell.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in reversed(processes):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


@contextmanager
def _launch_servers(
    args: argparse.Namespace, case: dict, devices: list[str], directory: Path
) -> Iterator[tuple[str, str, str, list[subprocess.Popen]]]:
    import yaml

    processes: list[subprocess.Popen] = []
    command = [sys.executable, "-m", "tensorrt_llm.commands.serve"]
    cluster = {
        "cluster_name": f"dspark-{secrets.token_hex(8)}",
        "cluster_uri": "http://127.0.0.1:0",
        "heartbeat_interval_sec": 5,
        "inactive_timeout_sec": 30,
        "minimal_instances": {"context_servers": 1, "generation_servers": 1},
    }
    auth_key = secrets.token_hex(32)
    router = {
        "hostname": "127.0.0.1",
        "port": 0,
        "backend": "pytorch",
        "context_servers": {"num_instances": 1},
        "generation_servers": {"num_instances": 1},
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
    }
    deadline = time.monotonic() + args.startup_timeout
    with ExitStack() as stack:

        def write_config(name: str, config: dict) -> Path:
            path = directory / f"{name}.yaml"
            # Worker handoff keys are local benchmark secrets, not public artifacts.
            with open(
                path, "x", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)
            ) as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            return path

        def launch(
            role: str, argv: list[str], gpu_ids: list[str], master_port: int | None = None
        ) -> None:
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("OMPI_", "PMI_", "PMIX_"))
                and key
                not in (
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                )
            }
            env.update(
                CUDA_VISIBLE_DEVICES=",".join(gpu_ids),
                LLM_MODELS_ROOT=str(args.models_root),
                PYTHONUNBUFFERED="1",
            )
            cache_dir = directory / f"{role}-prometheus"
            cache_dir.mkdir()
            env["PROMETHEUS_MULTIPROC_DIR"] = str(cache_dir)
            env.setdefault("UCX_TLS", "^ib")
            env.setdefault("UCX_MM_ERROR_HANDLING", "y")
            if master_port is not None:
                env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port))
            log = stack.enter_context((directory / f"{role}.log").open("w"))
            processes.append(
                subprocess.Popen(
                    command + argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )

        try:
            router_path = write_config("router", router)
            address_path = directory / "router.addr"
            launch(
                "router",
                [
                    "disaggregated",
                    "--config",
                    str(router_path),
                    "--server_start_timeout",
                    str(args.startup_timeout),
                    "--request_timeout",
                    str(args.request_timeout),
                    "--schedule_style",
                    "context_first",
                    "--report_addr",
                    str(address_path),
                ],
                [],
            )
            while not address_path.is_file():
                _check_processes(processes)
                if time.monotonic() > deadline:
                    raise TimeoutError("Router did not publish its address before startup timeout")
                time.sleep(1)
            address = address_path.read_text().strip()
            host, port = address.rsplit(":", 1)
            if host not in ("127.0.0.1", "localhost") or not port.isdigit():
                raise RuntimeError(f"Unexpected router address: {address}")
            router_url = f"http://{address}"
            cluster["cluster_uri"] = router_url
            # Reserve both rendezvous ports concurrently, ensuring they differ.
            with socket.socket() as ctx_socket, socket.socket() as gen_socket:
                ctx_socket.bind(("127.0.0.1", 0))
                gen_socket.bind(("127.0.0.1", 0))
                master_ports = [ctx_socket.getsockname()[1], gen_socket.getsockname()[1]]
            offset = 0
            for role, master_port in zip(("context", "generation"), master_ports):
                options = case[f"{role}_options"]
                count = _gpu_count(options)
                worker_path = write_config(
                    role,
                    {**options, "disagg_cluster": cluster, "internal_request_auth_key": auth_key},
                )
                argv = [
                    "serve",
                    case["model"],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--backend",
                    "pytorch",
                    "--config",
                    str(worker_path),
                    "--server_role",
                    role,
                ]
                for flag, key in (
                    ("--tp_size", "tensor_parallel_size"),
                    ("--pp_size", "pipeline_parallel_size"),
                    ("--cp_size", "context_parallel_size"),
                ):
                    argv.extend([flag, str(options.get(key, 1))])
                launch(role, argv, devices[offset : offset + count], master_port)
                offset += count
            last_update = 0.0
            while time.monotonic() < deadline:
                _check_processes(processes)
                try:
                    info = json.loads(_http(f"{router_url}/cluster_info", 10))
                except (urllib.error.URLError, TimeoutError):
                    info = {}
                if info.get("is_ready"):
                    workers = info["current_workers"]
                    urls = []
                    for role in ("context_servers", "generation_servers"):
                        if len(workers[role]) != 1:
                            raise RuntimeError(
                                "Expected exactly one context and one generation worker"
                            )
                        worker = workers[role][0]
                        urls.append(f"http://{worker['host']}:{worker['port']}")
                    _write_json(directory / "cluster.json", info)
                    yield router_url, urls[0], urls[1], processes
                    return
                if time.monotonic() - last_update > 30:
                    print(f"Waiting for context/generation startup; logs: {directory}", flush=True)
                    last_update = time.monotonic()
                time.sleep(2)
            raise TimeoutError("Disaggregated workers did not become ready before startup timeout")
        finally:
            _stop_processes(processes)


def _workload(
    args: argparse.Namespace,
    url: str,
    model: str,
    tokens: list[list[int]],
    max_tokens: int,
    processes: list[subprocess.Popen],
) -> list[dict]:
    def complete(index: int, prompt: list[int]) -> dict:
        response = json.loads(
            _http(
                f"{url}/v1/completions",
                args.request_timeout,
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                    "n": 1,
                    "add_special_tokens": False,
                },
            )
        )
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") not in ("stop", "length"):
            raise RuntimeError(f"Request {index} did not finish successfully: {response}")
        return {
            "index": index,
            "finish_reason": choices[0]["finish_reason"],
            "usage": response.get("usage", {}),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(complete, index, prompt) for index, prompt in enumerate(tokens)]
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                _check_processes(processes)
                if len(rows) % 32 == 0 or len(rows) == len(tokens):
                    print(f"Completed {len(rows)}/{len(tokens)} disagg requests", flush=True)
        except (RuntimeError, ValueError, OSError, KeyboardInterrupt):
            for future in futures:
                future.cancel()
            _stop_processes(processes)
            raise
    return sorted(rows, key=lambda row: row["index"])


def _snapshot(url: str, path: Path) -> dict:
    text = _http(f"{url}/prometheus/metrics", 30)
    path.write_text(text, encoding="utf-8")
    return _parse_metrics(text)


def _run_case(
    args: argparse.Namespace, name: str, case: dict, devices: list[str], prompts: list[str]
) -> dict:
    directory = args.output_dir / name
    directory.mkdir()
    tokens = _tokenize(args, case, prompts)
    _write_json(directory / "prompt_token_ids.json", tokens)
    with _launch_servers(args, case, devices, directory) as (
        router,
        context,
        generation,
        processes,
    ):
        if args.warmup_prompts:
            _workload(
                args,
                router,
                case["model"],
                tokens[: args.warmup_prompts],
                min(8, args.max_tokens),
                processes,
            )
        before_ctx = _snapshot(context, directory / "context-before.prom")
        before_gen = _snapshot(generation, directory / "generation-before.prom")
        start = time.monotonic()
        requests = _workload(args, router, case["model"], tokens, args.max_tokens, processes)
        elapsed = time.monotonic() - start
        after_ctx = _snapshot(context, directory / "context-after.prom")
        after_gen = _snapshot(generation, directory / "generation-after.prom")
        summary = _summarize_counters(
            _counter_delta(before_ctx, after_ctx),
            _counter_delta(before_gen, after_gen),
            len(prompts),
            case["max_draft_len"],
        )
        result = {
            "case": name,
            "status": "ok",
            "mode": "disagg",
            "recipe": case,
            "elapsed_seconds": elapsed,
            "max_tokens": args.max_tokens,
            "prompt_format": args.prompt_format,
            "warmup_requests": min(args.warmup_prompts, len(prompts)),
            "metric_source": "generation completed-request per-position Prometheus counters",
            "al_definition": "1 + sum(accepted_draft_tokens) / verification_steps; includes disagg bootstrap",
            "requests": requests,
            **summary,
        }
        _write_json(args.output_dir / f"{name}.json", result)
        return result


def _write_summary(output_dir: Path, rows: list[dict], total_cases: int) -> None:
    _write_json(
        output_dir / "summary.json",
        {
            "mode": "disagg",
            "complete": len(rows) == total_cases and all(row["status"] == "ok" for row in rows),
            "results": rows,
        },
    )
    columns = [
        "case",
        "status",
        "required_gpus",
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


def _run_cases(args: argparse.Namespace, cases: dict[str, dict], devices: list[str]) -> int:
    for name, case in cases.items():
        ctx_count = _gpu_count(case["context_options"])
        runnable = len(devices) >= case["required_gpus"]
        print(
            f"{name}: DSpark disagg, needs {case['required_gpus']} GPUs; "
            + (
                f"context={devices[:ctx_count]}, generation={devices[ctx_count : case['required_gpus']]}"
                if runnable
                else f"SKIP: only {len(devices)} GPUs visible"
            ),
            flush=True,
        )
        if args.dry_run:
            print(json.dumps(case, indent=2))
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runnable = any(len(devices) >= case["required_gpus"] for case in cases.values())
    prompts = _load_prompts(args) if runnable else []
    corpus = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
    (args.output_dir / "prompts.jsonl").write_text(corpus, encoding="utf-8")
    _write_json(
        args.output_dir / "manifest.json",
        {
            "mode": "disagg",
            "cases": cases,
            "devices": devices,
            "num_prompts": len(prompts),
            "prompt_source": str(args.prompt_file)
            if args.prompt_file
            else "openai/gsm8k:main:test",
            "prompt_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
            "prompt_format": args.prompt_format,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup_prompts": args.warmup_prompts,
        },
    )
    rows = []
    for name, case in cases.items():
        row = {
            "case": name,
            "status": "failed",
            "required_gpus": case["required_gpus"],
            "log": str(args.output_dir / name),
        }
        if len(devices) < case["required_gpus"]:
            row.update(
                status="skipped_insufficient_gpus",
                error=f"Requires {case['required_gpus']} local GPUs; only {len(devices)} visible",
            )
        else:
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
                print(f"Running {name}; logs: {row['log']}", flush=True)
                try:
                    result = _run_case(args, name, case, devices, prompts)
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                else:
                    row.update(
                        {
                            key: value
                            for key, value in result.items()
                            if key
                            not in ("requests", "recipe", "context_counters", "generation_counters")
                        }
                    )
        rows.append(row)
        _write_summary(args.output_dir, rows, len(cases))
        print(
            f"{name}: {row['status']} AL={row.get('acceptance_length', 'n/a')} {row.get('error', '')}",
            flush=True,
        )
    skipped = sum(row["status"] == "skipped_insufficient_gpus" for row in rows)
    print(
        f"Results: {args.output_dir / 'summary.csv'}; {skipped} hardware skips. A skipped case has NO disagg result.",
        flush=True,
    )
    return int(
        not any(row["status"] == "ok" for row in rows)
        or any(row["status"] not in ("ok", "skipped_insufficient_gpus") for row in rows)
    )


def _handle_termination(signum: int, frame) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def _main() -> int:
    args = _parse_arguments()
    signal.signal(signal.SIGTERM, _handle_termination)
    return _run_cases(args, _load_cases(args), _get_devices(args.devices))


if __name__ == "__main__":
    sys.exit(_main())
