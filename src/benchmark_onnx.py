"""Benchmark an exported ONNX model under ONNX Runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


def preprocess(imgsz: int, image: str | None, seed: int = 0) -> np.ndarray:
    """Produce the NCHW float32 tensor the exported graph expects.

    Ultralytics normally handles this. Exporting the graph exports only the
    network, so preprocessing becomes the caller's responsibility: resize,
    BGR to RGB, HWC to CHW, scale to [0,1], add batch dimension.
    """
    if image:
        import cv2
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
        img = cv2.resize(img, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(x, 0)


def run(model_path: str, variant: str, config: dict,
        image: str | None, providers: list[str]) -> dict:
    bench = config["benchmark"]
    imgsz = bench["imgsz"]

    print(f"loading {model_path}")
    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()
    print(f"providers requested {providers} -> active {active}")

    inp = sess.get_inputs()[0].name
    x = preprocess(imgsz, image)
    print(f"input {inp} shape {x.shape} source {'real' if image else 'synthetic'}")

    def infer():
        sess.run(None, {inp: x})

    latency = utils.time_inference(
        infer, device="cpu",
        warmup=bench["warmup_runs"], iterations=bench["iterations"],
    )

    payload = {
        "weights": str(model_path),
        "format": "onnx",
        "providers_requested": providers,
        "providers_active": active,
        "input_source": f"real:{image}" if image else "synthetic",
        "size_mb": utils.model_size_mb(model_path),
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--image", default=None)
    ap.add_argument("--providers", default="CPUExecutionProvider",
                    help="comma-separated, e.g. CoreMLExecutionProvider,CPUExecutionProvider")
    args = ap.parse_args()

    config = utils.load_config(args.config)
    run(args.model, args.variant, config, args.image,
        [p.strip() for p in args.providers.split(",")])


if __name__ == "__main__":
    main()
