"""Class-agnostic MOT evaluation on VisDrone-MOT sequences.

Classes are ignored deliberately. The detector uses COCO-pretrained weights
on aerial drone imagery, where class accuracy is poor for reasons unrelated
to tracking. Evaluating association without class matching isolates the
tracker comparison from domain mismatch in the detector.

Annotation format (VisDrone-MOT):
  frame, id, x, y, w, h, score, category, truncation, occlusion
Rows with score == 0 are ignore regions and are excluded from ground truth.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# motmetrics 1.4 calls np.asfarray, removed in NumPy 2.0. Restore it before
# import rather than patching site-packages, so the venv stays reproducible
# from requirements.txt.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils

ROOT = utils.PROJECT_ROOT / "data" / "visdrone-mot" / "VisDrone2019-MOT-val"


def load_gt(seq: str) -> dict[int, list]:
    """frame -> [(id, x, y, w, h)], excluding ignore regions."""
    gt = defaultdict(list)
    path = ROOT / "annotations" / f"{seq}.txt"
    for line in path.read_text().strip().split("\n"):
        f = line.split(",")
        frame, tid = int(f[0]), int(f[1])
        x, y, w, h = map(float, f[2:6])
        score = int(f[6])
        if score == 0:
            continue
        gt[frame].append((tid, x, y, w, h))
    return gt


def run_sequence(model, seq: str, tracker: str, imgsz: int, device: str,
                 conf: float, limit: int | None):
    frames_dir = ROOT / "sequences" / seq
    files = sorted(frames_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]

    gt = load_gt(seq)
    acc = mm.MOTAccumulator(auto_id=False)
    model.predictor = None   # reset tracker state between sequences

    for idx, fp in enumerate(files, start=1):
        img = cv2.imread(str(fp))
        H, W = img.shape[:2]

        res = model.track(img, imgsz=imgsz, tracker=tracker, persist=True,
                          conf=conf, verbose=False, device=device)[0]

        hyp_ids, hyp_boxes = [], []
        b = res.boxes
        if b is not None and b.id is not None:
            for (x1, y1, x2, y2), tid in zip(b.xyxy.tolist(), b.id.tolist()):
                hyp_ids.append(int(tid))
                hyp_boxes.append([x1, y1, x2 - x1, y2 - y1])

        g = gt.get(idx, [])
        gt_ids = [t[0] for t in g]
        gt_boxes = [[t[1], t[2], t[3], t[4]] for t in g]

        dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5) \
            if gt_boxes and hyp_boxes else np.empty((len(gt_boxes), len(hyp_boxes)))
        acc.update(gt_ids, hyp_ids, dist, frameid=idx)

    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["botsort.yaml", "bytetrack.yaml"])
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sequences", type=int, default=None,
                    help="evaluate only the first N sequences")
    ap.add_argument("--limit", type=int, default=None,
                    help="frames per sequence")
    args = ap.parse_args()

    config = utils.load_config(args.config)
    device = args.device or config["benchmark"]["device"]
    imgsz = config["benchmark"]["imgsz"]

    seqs = sorted(p.name for p in (ROOT / "sequences").iterdir() if p.is_dir())
    if args.sequences:
        seqs = seqs[:args.sequences]

    model = YOLO(args.weights)
    if not str(args.weights).endswith(".engine"):
        model.to(device)

    accs, names = [], []
    for s in seqs:
        n = len(list((ROOT / "sequences" / s).glob("*.jpg")))
        print(f"{s}  ({args.limit or n} frames)")
        accs.append(run_sequence(model, s, args.tracker, imgsz, device,
                                 args.conf, args.limit))
        names.append(s)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names,
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches",
                 "num_fragmentations", "num_misses", "num_false_positives",
                 "mostly_tracked", "mostly_lost"],
        generate_overall=True,
    )
    print("\n" + mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names))

    o = summary.loc["OVERALL"]
    payload = {
        "weights": str(args.weights),
        "tracker": args.tracker,
        "device": device,
        "conf_threshold": args.conf,
        "class_agnostic": True,
        "sequences": names,
        "frames": int(o["num_frames"]),
        "metrics": {
            "MOTA": round(float(o["mota"]), 4),
            "MOTP": round(float(o["motp"]), 4),
            "IDF1": round(float(o["idf1"]), 4),
            "ID_switches": int(o["num_switches"]),
            "fragmentations": int(o["num_fragmentations"]),
            "misses": int(o["num_misses"]),
            "false_positives": int(o["num_false_positives"]),
            "mostly_tracked": int(o["mostly_tracked"]),
            "mostly_lost": int(o["mostly_lost"]),
        },
    }
    name = f"mot_{Path(args.weights).stem}_{args.tracker.split('.')[0]}"
    path = utils.save_result(name, payload, config)
    print(f"\nwritten {path.relative_to(utils.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
