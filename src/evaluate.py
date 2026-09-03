"""Measure mAP for an ONNX detection model against COCO val2017.

The exported graph contains only the network. Letterboxing, coordinate
recovery, class-ID mapping, and JSON formatting are reimplemented here to
match the Ultralytics pipeline, because a mismatch anywhere shows up as an
accuracy gap indistinguishable from quantization damage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils

# COCO's 80 contiguous class indices map to non-contiguous category IDs.
COCO91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,
          28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,
          53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,
          79,80,81,82,84,85,86,87,88,89,90]


def letterbox(img: np.ndarray, size: int = 640):
    """Resize preserving aspect ratio, pad to square with grey.

    Returns the padded image plus the scale and offsets needed to map
    predictions back to original image coordinates.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess(path: str, size: int):
    img = cv2.imread(path)
    padded, r, dx, dy = letterbox(img, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0), r, dx, dy


def decode(out: np.ndarray, r: float, dx: int, dy: int, conf_thr: float):
    """Convert model output to COCO-format detections in original coordinates."""
    preds = out[0]                      # (300, 6) for NMS-free heads
    keep = preds[:, 4] >= conf_thr
    preds = preds[keep]

    dets = []
    for x1, y1, x2, y2, score, cls in preds:
        x1 = (x1 - dx) / r
        y1 = (y1 - dy) / r
        x2 = (x2 - dx) / r
        y2 = (y2 - dy) / r
        dets.append({
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
            "category_id": COCO91[int(cls)],
        })
    return dets


def run(model_path: str, variant: str, config: dict,
        limit: int | None, conf_thr: float, providers: list[str]) -> dict:
    imgsz = config["benchmark"]["imgsz"]
    root = utils.PROJECT_ROOT / "data" / "coco"
    ann_file = root / "annotations" / "instances_val2017.json"

    coco = COCO(str(ann_file))
    img_ids = sorted(coco.getImgIds())
    if limit:
        img_ids = img_ids[:limit]

    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0].name
    print(f"providers active: {sess.get_providers()}")
    print(f"evaluating {len(img_ids)} images\n")

    results = []
    for i, iid in enumerate(img_ids):
        meta = coco.loadImgs(iid)[0]
        path = root / "val2017" / meta["file_name"]
        x, r, dx, dy = preprocess(str(path), imgsz)
        out = sess.run(None, {inp: x})[0]
        for d in decode(out, r, dx, dy, conf_thr):
            results.append({"image_id": iid, **d})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(img_ids)}")

    if not results:
        raise RuntimeError("no detections produced; check output shape and threshold")

    pred_file = utils.PROJECT_ROOT / "results" / f"{variant}_predictions.json"
    pred_file.write_text(json.dumps(results))

    coco_dt = coco.loadRes(str(pred_file))
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    payload = {
        "weights": str(model_path),
        "images_evaluated": len(img_ids),
        "conf_threshold": conf_thr,
        "detections": len(results),
        "accuracy": {
            "mAP_50_95": round(float(ev.stats[0]), 4),
            "mAP_50": round(float(ev.stats[1]), 4),
            "mAP_75": round(float(ev.stats[2]), 4),
            "mAP_small": round(float(ev.stats[3]), 4),
            "mAP_medium": round(float(ev.stats[4]), 4),
            "mAP_large": round(float(ev.stats[5]), 4),
        },
        "size_mb": utils.model_size_mb(model_path),
    }
    path = utils.save_result(f"{variant}_accuracy", payload, config)
    print(f"\nwritten {path.relative_to(utils.PROJECT_ROOT)}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--limit", type=int, default=None, help="subset size for quick checks")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low threshold: mAP integrates over the full PR curve")
    ap.add_argument("--providers", default="CPUExecutionProvider")
    args = ap.parse_args()

    config = utils.load_config(args.config)
    run(args.model, args.variant, config, args.limit, args.conf,
        [p.strip() for p in args.providers.split(",")])


if __name__ == "__main__":
    main()
