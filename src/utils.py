"""Shared helpers: config loading, device-aware timing, result serialization."""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/benchmark.yaml") -> dict:
    """Load the benchmark configuration."""
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


def synchronize(device: str) -> None:
    """Block until all queued GPU work has completed.

    GPU calls are asynchronous: control returns to Python before the
    computation finishes. Without this barrier a timer measures dispatch
    latency, not execution time, and reports numbers that are far too low.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_inference(fn, device: str, warmup: int, iterations: int) -> dict:
    """Time a callable and return the full latency distribution in milliseconds.

    Warmup runs are discarded: the first calls include weight transfer,
    kernel compilation, and cache population, which are one-time costs
    and not representative of steady-state inference.
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "std_ms": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[int(len(samples) * 0.95)], 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        "fps_p50": round(1000.0 / samples[len(samples) // 2], 2),
        "warmup_runs": warmup,
        "iterations": iterations,
    }


def model_size_mb(path: str | Path) -> float:
    """On-disk size in MB. Directories (CoreML .mlpackage) are summed."""
    p = Path(path)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        total = p.stat().st_size
    return round(total / (1024 * 1024), 2)


def environment() -> dict:
    """Capture the machine and library versions that produced a result."""
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
    return env


def save_result(name: str, payload: dict, config: dict) -> Path:
    """Write one variant's results to JSON, with settings and environment attached.

    A latency number without the configuration that produced it is not a
    result, it is an anecdote. Config and environment travel with the data.
    """
    out_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "variant": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config["benchmark"],
        "environment": environment(),
        **payload,
    }

    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
