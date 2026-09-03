"""TensorRT export, benchmark, and COCO validation on NVIDIA.

Run in Google Colab with a GPU runtime. Ultralytics must be installed with
--no-deps: a plain `pip install ultralytics` pulls a torchvision build whose
ABI does not match the preinstalled torch, and `torchvision::nms` fails to
register, breaking validation.

    !pip install -q ultralytics --no-deps
    !pip install -q pyyaml requests pillow scipy matplotlib pandas psutil \
        py-cpuinfo tqdm ultralytics-thop ultralytics-platform

Usage:

    !python scripts/tensorrt_colab.py --device-tag a100 --workspace 8 --validate
    !python scripts/tensorrt_colab.py --device-tag t4   --workspace 4

Accuracy is device-independent, so --validate is only needed on the first
card. Repeating it downloads 18 GB of COCO for no new information.
"""

import argparse, json, os, platform, shutil, time
from datetime import datetime, timezone

import numpy as np
import torch
from ultralytics import YOLO

IMGSZ, WARMUP, ITERS = 640, 50, 300

# TensorRT engines are compiled per GPU and are not portable. DEVICE_TAG
# labels outputs so results from different cards do not overwrite one another.


def build_engines(tag_prefix, workspace, base="yolo26s.pt"):
    m = YOLO(base)
    specs = [
        ("fp32", dict(half=False, int8=False)),
        ("fp16", dict(half=True, int8=False)),
        ("int8", dict(int8=True, data="coco128.yaml")),
    ]
    for tag, kw in specs:
        out = f"yolo26s_{tag_prefix}_{tag}.engine"
        if os.path.exists(out):
            print(f"{tag}: exists, skipping")
            continue
        t = time.time()
        p = m.export(format="engine", imgsz=IMGSZ, batch=1, device=0,
                     workspace=workspace, **kw)
        shutil.move(str(p), out)
        print(f"{tag}: built in {time.time() - t:.0f}s")


def benchmark(tag_prefix, tag):
    """Latency with explicit CUDA synchronization.

    Without the barrier the timer measures kernel dispatch rather than
    execution, since GPU calls return before the work completes.
    """
    m = YOLO(f"yolo26s_{tag_prefix}_{tag}.engine", task="detect")
    img = np.random.default_rng(0).integers(0, 256, (IMGSZ, IMGSZ, 3)).astype(np.uint8)

    for _ in range(WARMUP):
        m.predict(img, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()

    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        m.predict(img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()

    return {
        "mean_ms": round(sum(ts) / len(ts), 3),
        "p50_ms": round(ts[len(ts) // 2], 3),
        "p95_ms": round(ts[int(len(ts) * 0.95)], 3),
        "min_ms": round(ts[0], 3),
        "max_ms": round(ts[-1], 3),
        "fps_p50": round(1000 / ts[len(ts) // 2], 2),
        "warmup_runs": WARMUP,
        "iterations": ITERS,
    }


def validate(tag_prefix, tag):
    m = YOLO(f"yolo26s_{tag_prefix}_{tag}.engine", task="detect")
    r = m.val(data="coco.yaml", imgsz=IMGSZ, batch=1, device=0)
    return {
        "mAP_50_95": round(float(r.box.map), 4),
        "mAP_50": round(float(r.box.map50), 4),
        "mAP_75": round(float(r.box.map75), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-tag", required=True,
                    help="short label for this GPU, e.g. a100 or t4")
    ap.add_argument("--workspace", type=int, default=8,
                    help="TensorRT scratch memory in GB; 8 for A100, 4 for T4")
    ap.add_argument("--validate", action="store_true",
                    help="run COCO validation; skip on repeat hardware since mAP is device-independent")
    args = ap.parse_args()

    prefix = args.device_tag
    build_engines(prefix, args.workspace)

    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_available": True,
    }
    props = torch.cuda.get_device_properties(0)
    env["memory_mib"] = props.total_memory // (1024 ** 2)

    for tag in ["fp32", "fp16", "int8"]:
        payload = {
            "variant": f"yolo26s_{prefix}_tensorrt_{tag}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": "tensorrt",
            "precision": tag,
            "device_tag": prefix,
            "environment": env,
            "config": {"imgsz": IMGSZ, "batch": 1, "warmup_runs": WARMUP,
                       "iterations": ITERS, "device": "cuda",
                       "trt_workspace_gb": args.workspace},
            "size_mb": round(os.path.getsize(f"yolo26s_{prefix}_{tag}.engine") / 1024**2, 2),
            "latency": benchmark(prefix, tag),
        }
        if args.validate and tag in ("fp32", "int8"):
            payload["accuracy"] = validate(prefix, tag)

        name = f"yolo26s_{prefix}_tensorrt_{tag}.json"
        with open(name, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {name}")


if __name__ == "__main__":
    main()
