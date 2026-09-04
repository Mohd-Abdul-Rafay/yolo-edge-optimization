"""Video tracking pipeline with per-stage timing.

Reports end-to-end throughput broken down by stage. Per-frame inference
latency is the number usually published; it is one of five stages here and
frequently not the largest. Decode and preprocess scale with source
resolution, which inference does not.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


class Stage:
    """Accumulates per-stage timings across frames."""

    def __init__(self):
        self.t = defaultdict(list)

    def record(self, name, dt_ms):
        self.t[name].append(dt_ms)

    def summary(self):
        out = {}
        for name, samples in self.t.items():
            s = sorted(samples)
            out[name] = {
                "mean_ms": round(statistics.mean(s), 3),
                "p50_ms": round(s[len(s) // 2], 3),
                "p95_ms": round(s[int(len(s) * 0.95)], 3),
                "total_ms": round(sum(s), 1),
                "share_pct": None,
            }
        grand = sum(v["total_ms"] for v in out.values())
        for v in out.values():
            v["share_pct"] = round(100 * v["total_ms"] / grand, 1)
        return out


def run(weights, video, tracker, config, device, render, limit):
    imgsz = config["benchmark"]["imgsz"]
    model = YOLO(weights)
    if not str(weights).endswith(".engine"):
        model.to(device)

    cap = cv2.VideoCapture(video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if limit:
        n_frames = min(n_frames, limit)

    print(f"{video}  {src_w}x{src_h}  {n_frames} frames  tracker={tracker}")

    writer = None
    if render:
        out_path = str(utils.PROJECT_ROOT / "figures" / f"tracked_{tracker.split('.')[0]}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (src_w, src_h))

    st = Stage()
    ids_seen = set()
    per_frame_ids = []
    wall_start = time.perf_counter()
    i = 0

    while i < n_frames:
        t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        st.record("decode", (time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        resized = cv2.resize(frame, (imgsz, imgsz))
        st.record("preprocess", (time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        res = model.track(resized, imgsz=imgsz, tracker=tracker,
                          persist=True, verbose=False, device=device)[0]
        if device in ("mps", "cuda"):
            utils.synchronize(device)
        st.record("infer_track", (time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        boxes = res.boxes
        frame_ids = []
        if boxes is not None and boxes.id is not None:
            frame_ids = [int(x) for x in boxes.id.tolist()]
            ids_seen.update(frame_ids)
        per_frame_ids.append(frame_ids)
        st.record("postprocess", (time.perf_counter() - t) * 1000)

        if writer is not None:
            t = time.perf_counter()
            annotated = res.plot()
            writer.write(cv2.resize(annotated, (src_w, src_h)))
            st.record("render", (time.perf_counter() - t) * 1000)

        i += 1

    wall = time.perf_counter() - wall_start
    cap.release()
    if writer is not None:
        writer.release()

    # track continuity: how often an ID present in frame k is absent in k+1
    fragments = 0
    for a, b in zip(per_frame_ids, per_frame_ids[1:]):
        fragments += len(set(a) - set(b))

    stages = st.summary()
    payload = {
        "weights": str(weights),
        "video": video,
        "source_resolution": f"{src_w}x{src_h}",
        "model_input": imgsz,
        "tracker": tracker,
        "device": device,
        "frames": i,
        "wall_clock_s": round(wall, 2),
        "end_to_end_fps": round(i / wall, 2),
        "stages": stages,
        "tracking": {
            "unique_ids": len(ids_seen),
            "track_terminations": fragments,
            "mean_ids_per_frame": round(
                sum(len(f) for f in per_frame_ids) / max(len(per_frame_ids), 1), 2),
        },
    }

    src_tag = "webcam" if str(video).isdigit() else Path(video).stem
    name = f"track_{Path(weights).stem}_{tracker.split('.')[0]}_{src_tag}_{device}"
    path = utils.save_result(name, payload, config)

    print(f"\nend-to-end   {payload['end_to_end_fps']} FPS over {i} frames "
          f"({wall:.1f}s wall clock)")
    print(f"unique IDs   {len(ids_seen)}   terminations {fragments}   "
          f"mean/frame {payload['tracking']['mean_ids_per_frame']}")
    print("\nstage           p50 ms    share")
    for k, v in stages.items():
        print(f"  {k:13s} {v['p50_ms']:7.3f}   {v['share_pct']:5.1f}%")
    print(f"\nwritten {path.relative_to(utils.PROJECT_ROOT)}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracker", default="botsort.yaml",
                    choices=["botsort.yaml", "bytetrack.yaml"])
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--render", action="store_true", help="write annotated video")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    config = utils.load_config(args.config)
    device = args.device or config["benchmark"]["device"]
    run(args.weights, args.video, args.tracker, config, device, args.render, args.limit)


if __name__ == "__main__":
    main()
