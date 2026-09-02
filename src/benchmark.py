"""Benchmark a PyTorch YOLO checkpoint: latency distribution and model size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


def fixed_input(imgsz: int, seed: int = 0) -> np.ndarray:
    """A deterministic synthetic frame.

    Identical input across every variant, so any latency difference is
    attributable to the model or export format rather than scene content.
    Synthetic rather than real because this measures throughput, not accuracy;
    accuracy is measured separately in evaluate.py against the real val set.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)


def run(weights: str, variant: str, config: dict) -> dict:
    bench = config["benchmark"]
    device = bench["device"]
    imgsz = bench["imgsz"]

    print(f"loading {weights} on {device}")
    model = YOLO(weights)
    model.to(device)

    frame = fixed_input(imgsz)

    def infer():
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)

    print(f"warmup {bench['warmup_runs']} / timing {bench['iterations']}")
    latency = utils.time_inference(
        infer,
        device=device,
        warmup=bench["warmup_runs"],
        iterations=bench["iterations"],
    )

    payload = {
        "weights": str(weights),
        "format": "pytorch",
        "size_mb": utils.model_size_mb(weights),
        "latency": latency,
    }

    path = utils.save_result(variant, payload, config)
    print(f"\nvariant   {variant}")
    print(f"size      {payload['size_mb']} MB")
    print(f"p50       {latency['p50_ms']} ms   ({latency['fps_p50']} FPS)")
    print(f"p95       {latency['p95_ms']} ms")
    print(f"mean/std  {latency['mean_ms']} / {latency['std_ms']} ms")
    print(f"written   {path.relative_to(utils.PROJECT_ROOT)}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint")
    ap.add_argument("--variant", required=True, help="result name, e.g. yolo26s_pytorch")
    ap.add_argument("--config", default="configs/benchmark.yaml")
    args = ap.parse_args()

    config = utils.load_config(args.config)
    run(args.weights, args.variant, config)


if __name__ == "__main__":
    main()
