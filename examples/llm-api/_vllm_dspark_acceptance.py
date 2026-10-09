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
"""Shared serving and measurement code for the vLLM Kimi K3 MLA DSpark runners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator, Literal

from _vllm_dspark_metrics import parse_metrics, summarize_delta


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _arguments(mode: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Measure vLLM Kimi K3 MLA DSpark acceptance ({mode})."
    )
    parser.add_argument("--model", default="moonshotai/Kimi-K3", help="HF ID or local target path.")
    parser.add_argument("--drafter", default="Inferact/Kimi-K3-DSpark", help="MLA drafter ID/path.")
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=8)
    parser.add_argument("--num-speculative-tokens", type=_positive_int, default=7)
    parser.add_argument("--max-model-len", type=_positive_int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8", "fp8_e4m3"),
        default="auto",
        help="vLLM target KV cache dtype; FP8 also enables quantized MLA prefill queries.",
    )
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"))
    parser.add_argument("--rejection-sample-method", choices=("standard", "block"))
    parser.add_argument(
        "--devices", help="Local GPU indices/UUIDs; otherwise CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument(
        "--port", type=_positive_int, default=8000, help="Aggregate/decode HTTP port."
    )
    if mode == "agg":
        parser.add_argument("--server-url", help="Use an existing dedicated aggregate server.")
    else:
        parser.add_argument("--prefill-port", type=_positive_int, default=8001)
        parser.add_argument("--prefill-side-channel-port", type=_positive_int, default=5557)
        parser.add_argument("--decode-side-channel-port", type=_positive_int, default=5657)
        parser.add_argument("--prefill-url", help="Existing dedicated NIXL prefill server.")
        parser.add_argument("--decode-url", help="Existing dedicated NIXL decode server.")
        parser.add_argument(
            "--aggregate-results", type=Path, help="Matching aggregate output directory."
        )
        parser.add_argument(
            "--max-al-difference", type=float, help="Optional absolute AL tolerance."
        )
    parser.add_argument("--output-dir", type=Path, default=Path(f"vllm-dspark-al-{mode}"))
    parser.add_argument("--prompt-file", type=Path, help="JSONL with a nonempty prompt per record.")
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--num-prompts", type=_positive_int, default=64)
    parser.add_argument("--max-tokens", type=_positive_int, default=256)
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=_positive_int, default=3600)
    parser.add_argument("--request-timeout", type=_positive_int, default=1800)
    parser.add_argument("--metrics-timeout", type=_positive_int, default=30)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print launch plan; no downloads or writes."
    )
    args = parser.parse_args(argv)
    args.mode = mode
    if args.warmup_prompts < 0:
        parser.error("--warmup-prompts must be nonnegative")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must leave room for prompts in --max-model-len")
    if args.max_num_batched_tokens < args.concurrency:
        parser.error("--max-num-batched-tokens must be at least --concurrency")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be finite and nonnegative; --top-p must be in (0, 1]")
    if args.draft_sample_method is None:
        args.draft_sample_method = "greedy" if args.temperature == 0 else "probabilistic"
    if mode == "disagg":
        if bool(args.prefill_url) != bool(args.decode_url):
            parser.error("provide both --prefill-url and --decode-url")
        if args.prefill_url and args.prefill_url.rstrip("/") == args.decode_url.rstrip("/"):
            parser.error("prefill and decode must be distinct servers")
        if args.max_al_difference is not None and (
            not args.aggregate_results
            or not math.isfinite(args.max_al_difference)
            or args.max_al_difference < 0
        ):
            parser.error(
                "--max-al-difference requires --aggregate-results and a finite nonnegative value"
            )
    urls = (
        {"aggregate": args.server_url}
        if mode == "agg"
        else {"prefill": args.prefill_url, "decode": args.decode_url}
    )
    args.external = all(urls.values())
    for url in urls.values():
        if url:
            parsed = urllib.parse.urlsplit(url)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.netloc
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                parser.error("server URLs must be HTTP(S) origins without /v1 or other paths")
    args.urls = {role: url.rstrip("/") for role, url in urls.items() if url}
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _recipe(args: argparse.Namespace) -> dict:
    speculation = {
        "model": args.drafter,
        "method": "dspark",
        "num_speculative_tokens": args.num_speculative_tokens,
        "attention_backend": "FLASHINFER_MLA",
        "draft_sample_method": args.draft_sample_method,
    }
    if args.rejection_sample_method:
        speculation["rejection_sample_method"] = args.rejection_sample_method
    return {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.concurrency,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "attention_config": {
            "mla_prefill_backend": "FLASHINFER",
            "use_prefill_query_quantization": True,
        }
        if args.kv_cache_dtype != "auto"
        else {},
        "speculative_config": speculation,
    }


def _launch_plan(args: argparse.Namespace) -> list[dict]:
    if args.external:
        return []
    roles = ["aggregate"] if args.mode == "agg" else ["prefill", "decode"]
    count = args.tensor_parallel_size * len(roles)
    visible = args.devices if args.devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if args.dry_run:
            visible = ",".join(map(str, range(count)))
        else:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            visible = ",".join(result.stdout.split())
    devices = [device.strip() for device in visible.split(",")]
    if all(re.fullmatch(r"[0-9]+", device) for device in devices):
        devices = [str(int(device)) for device in devices]
    elif all(
        re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device)
        for device in devices
    ):
        devices = ["GPU-" + device[4:].lower() for device in devices]
    else:
        raise ValueError("Provide distinct GPU indices or full GPU UUIDs, without mixed aliases")
    if len(set(devices)) != len(devices):
        raise ValueError("Provide distinct GPUs; prefill and decode must not share a GPU")
    if len(devices) < count:
        raise ValueError(
            f"Need {count} GPUs ({args.tensor_parallel_size} per server); got {len(devices)}"
        )
    ports = [args.port]
    if args.mode == "disagg":
        ports += [args.prefill_port, args.prefill_side_channel_port, args.decode_side_channel_port]
    if len(set(ports)) != len(ports) or any(port > 65535 for port in ports):
        raise ValueError("HTTP and NIXL side-channel ports must be distinct and in [1, 65535]")
    plan = []
    for index, role in enumerate(roles):
        port = args.prefill_port if role == "prefill" else args.port
        command = [
            args.vllm_executable,
            "serve",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--trust-remote-code",
            "--language-model-only",
            "--load-format",
            "fastsafetensors",
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-num-seqs",
            str(args.concurrency),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--seed",
            str(args.seed),
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--speculative-config",
            json.dumps(_recipe(args)["speculative_config"]),
        ]
        if _recipe(args)["attention_config"]:
            command += ["--attention-config", json.dumps(_recipe(args)["attention_config"])]
        start = index * args.tensor_parallel_size
        env = {"CUDA_VISIBLE_DEVICES": ",".join(devices[start : start + args.tensor_parallel_size])}
        if role != "aggregate":
            command += [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "NixlConnector",
                        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                        "kv_load_failure_policy": "fail",
                    }
                ),
            ]
            env.update(
                VLLM_SSM_CONV_STATE_LAYOUT="DS",
                VLLM_NIXL_SIDE_CHANNEL_PORT=str(
                    args.prefill_side_channel_port
                    if role == "prefill"
                    else args.decode_side_channel_port
                ),
            )
        plan.append(
            {"role": role, "url": f"http://127.0.0.1:{port}", "command": command, "env": env}
        )
    return plan


def _http(
    url: str, timeout: int, payload: dict | None = None, request_id: str | None = None
) -> str:
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code} from {url}: {error.read().decode()[:2000]}"
        ) from error


def _check_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server {process.pid} exited with code {process.returncode}; see logs"
            )


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in reversed(processes):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


@contextmanager
def _servers(args: argparse.Namespace, plan: list[dict]) -> Iterator[tuple[dict, list]]:
    processes: list[subprocess.Popen] = []
    urls = dict(args.urls)
    with ExitStack() as stack:
        try:
            # Reject an occupied port before health probes can see an unrelated server.
            for server in plan:
                port = urllib.parse.urlsplit(server["url"]).port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", port))
            for server in plan:
                role = server["role"]
                log = stack.enter_context((args.output_dir / f"{role}.log").open("w"))
                env = dict(os.environ)
                # Each API server owns its metric registry and engine children.
                env.pop("PROMETHEUS_MULTIPROC_DIR", None)
                env.update(server["env"])
                processes.append(
                    subprocess.Popen(
                        server["command"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                urls[role] = server["url"]
            deadline = time.monotonic() + args.startup_timeout
            pending = dict(urls)
            while pending:
                _check_processes(processes)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server startup timed out: {list(pending)}")
                for role, url in list(pending.items()):
                    try:
                        _http(f"{url}/health", 5)
                    except (OSError, RuntimeError):
                        continue
                    del pending[role]
                if pending:
                    time.sleep(1)
            yield urls, processes
        finally:
            _stop_processes(processes)


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


def _tokenize(args: argparse.Namespace, urls: dict, prompts: list[str]) -> list[list[int]]:
    all_tokens = []
    for url in urls.values():
        tokenized = []
        for prompt in prompts:
            payload = {"model": args.model, "add_special_tokens": False}
            if args.prompt_format == "raw":
                payload["prompt"] = prompt
            else:
                payload.update(
                    messages=[{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            response = json.loads(_http(f"{url}/tokenize", args.request_timeout, payload))
            tokens = response["tokens"]
            if not tokens or any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("Tokenizer returned invalid token IDs")
            if len(tokens) + args.max_tokens > min(args.max_model_len, response["max_model_len"]):
                raise ValueError("Prompt plus output exceeds the model's sequence limit")
            tokenized.append(tokens)
        all_tokens.append(tokenized)
    if any(tokens != all_tokens[0] for tokens in all_tokens[1:]):
        raise ValueError("Prefill/decode tokenizers do not agree on the prompt corpus")
    return all_tokens[0]


def _complete(args: argparse.Namespace, urls: dict, prompt: list[int], max_tokens: int) -> dict:
    request_id = str(uuid.uuid4())
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": False,
        "n": 1,
        "add_special_tokens": False,
    }
    if args.mode == "disagg":
        prefill_payload = dict(
            payload,
            max_tokens=1,
            kv_transfer_params={
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        )
        prefill = json.loads(
            _http(
                f"{urls['prefill']}/v1/completions",
                args.request_timeout,
                prefill_payload,
                request_id,
            )
        )
        transfer = prefill.get("kv_transfer_params")
        if (
            not isinstance(transfer, dict)
            or transfer.get("do_remote_prefill") is not True
            or any(
                not transfer.get(key)
                for key in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                    "remote_block_ids",
                )
            )
        ):
            raise RuntimeError(
                "Prefill did not return a NIXL handoff; refusing local decode fallback"
            )
        payload["kv_transfer_params"] = transfer
        url = urls["decode"]
    else:
        url = urls["aggregate"]
    response = json.loads(_http(f"{url}/v1/completions", args.request_timeout, payload, request_id))
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") not in ("stop", "length"):
        raise RuntimeError(f"Completion failed: {response}")
    return {"finish_reason": choices[0]["finish_reason"], "usage": response.get("usage", {})}


def _workload(
    args: argparse.Namespace, urls: dict, tokens: list[list[int]], max_tokens: int
) -> list[dict]:
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        return list(pool.map(lambda prompt: _complete(args, urls, prompt, max_tokens), tokens))
    finally:
        # Cancel queued requests before server cleanup. In-flight requests finish
        # or fail when managed servers stop (external requests retain their timeout).
        pool.shutdown(wait=False, cancel_futures=True)


def _snapshot(
    args: argparse.Namespace, urls: dict, expected: dict[str, int] | None, label: str
) -> dict:
    deadline = time.monotonic() + args.metrics_timeout
    while True:
        snapshots = {}
        ready = True
        for role, url in urls.items():
            body = _http(f"{url}/metrics", 30)
            metrics = parse_metrics(body)
            (args.output_dir / f"{role}-{label}.prom").write_text(body, encoding="utf-8")
            completed = metrics["completed_requests"]
            if completed is None:
                raise ValueError("Missing vLLM request counters; enable server statistics")
            if expected is not None:
                if completed > expected[role]:
                    raise ValueError("Unexpected concurrent traffic on the measurement server")
                ready &= completed == expected[role]
            snapshots[role] = metrics
        if ready:
            return snapshots
        if time.monotonic() >= deadline:
            raise TimeoutError("Completed-request metrics did not catch up with the workload")
        time.sleep(0.1)


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _comparison(args: argparse.Namespace, tokens: list[list[int]]) -> dict:
    return {
        "recipe": _recipe(args),
        "prompt_format": args.prompt_format,
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "num_prompts": len(tokens),
        "warmup_prompts": min(args.warmup_prompts, len(tokens)),
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }


def _measure(args: argparse.Namespace, plan: list[dict]) -> dict:
    prompts = _load_prompts(args)
    corpus = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
    (args.output_dir / "prompts.jsonl").write_text(corpus, encoding="utf-8")
    manifest = {
        "backend": "vllm",
        "mode": args.mode,
        "recipe": _recipe(args),
        "launch_plan": plan,
        "external_servers": args.urls,
        "prompt_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    with _servers(args, plan) as (urls, processes):
        tokens = _tokenize(args, urls, prompts)
        _write_json(args.output_dir / "prompt_token_ids.json", tokens)
        comparison = _comparison(args, tokens)
        manifest["comparison"] = comparison
        _write_json(args.output_dir / "manifest.json", manifest)
        reference = None
        if args.mode == "disagg" and args.aggregate_results:
            reference = json.loads((args.aggregate_results / "summary.json").read_text())
            if (
                reference.get("status") != "ok"
                or reference.get("mode") != "agg"
                or reference.get("comparison") != comparison
            ):
                raise ValueError(
                    "Aggregate reference must be a successful run with matching recipe and workload"
                )
        initial = _snapshot(args, urls, None, "initial")
        warmup = min(args.warmup_prompts, len(tokens))
        _workload(args, urls, tokens[:warmup], min(8, args.max_tokens))
        expected = {
            role: counters["completed_requests"] + warmup for role, counters in initial.items()
        }
        before = _snapshot(args, urls, expected, "before")
        _check_processes(processes)
        start = time.monotonic()
        requests = _workload(args, urls, tokens, args.max_tokens)
        elapsed = time.monotonic() - start
        _check_processes(processes)
        after = _snapshot(
            args, urls, {role: count + len(tokens) for role, count in expected.items()}, "after"
        )
        role = "aggregate" if args.mode == "agg" else "decode"
        metrics = summarize_delta(
            before[role], after[role], len(tokens), args.num_speculative_tokens
        )
        result = {
            "backend": "vllm",
            "mode": args.mode,
            "case": "kimi-k3-mla",
            "status": "ok",
            "comparison": comparison,
            "elapsed_seconds": elapsed,
            "requests": requests,
            "metric_source": f"{role} vLLM Prometheus speculative counters, warmup excluded",
            "al_definition": "1 + accepted_draft_tokens / verification_steps",
            **metrics,
        }
        if reference is not None:
            difference = result["acceptance_length"] - reference["acceptance_length"]
            result.update(aggregate_al=reference["acceptance_length"], al_difference=difference)
            if args.max_al_difference is not None and abs(difference) > args.max_al_difference:
                result.update(
                    status="failed",
                    error=f"Absolute AL difference {abs(difference)} exceeds {args.max_al_difference}",
                )
        return result


def main(mode: Literal["agg", "disagg"], argv: list[str] | None = None) -> int:
    """Run one Kimi K3 MLA measurement, or print its launch plan without side effects."""
    args = _arguments(mode, argv)
    plan = _launch_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {"mode": mode, "recipe": _recipe(args), "servers": plan or args.urls}, indent=2
            )
        )
        for server in plan:
            print(
                shlex.join(
                    ["env", *(f"{k}={v}" for k, v in server["env"].items()), *server["command"]]
                )
            )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        result = _measure(args, plan)
    except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as error:
        result = {
            "backend": "vllm",
            "mode": mode,
            "status": "failed",
            "error": str(error) or "Interrupted",
        }
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    _write_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        columns = [
            "case",
            "status",
            "acceptance_length",
            "acceptance_rate",
            "verification_steps",
            "accepted_draft_tokens",
            "draft_tokens",
            "aggregate_al",
            "al_difference",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    print(
        f"{mode}: {result['status']} AL={result.get('acceptance_length', 'n/a')} {result.get('error', '')}"
    )
    print(f"Results: {args.output_dir / 'summary.json'}")
    return int(result["status"] != "ok")


def _terminate(signum: int, frame) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")
