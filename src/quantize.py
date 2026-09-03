"""Static INT8 quantization of an ONNX detection model.

Static quantization fixes both weight and activation scales ahead of time.
Activation ranges cannot be known from the graph alone, so a calibration
pass runs representative images through the model and records observed
minima and maxima. Calibration images are held out from evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_static,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils
from evaluate import preprocess


class Reader(CalibrationDataReader):
    """Yields preprocessed tensors using the exact evaluation preprocessing.

    Any mismatch between calibration and inference preprocessing produces
    wrong activation ranges, and the resulting accuracy loss would be
    misattributed to quantization itself.
    """

    def __init__(self, files: list[str], input_name: str, imgsz: int):
        self.input_name = input_name
        self.imgsz = imgsz
        self.files = iter(files)
        self.n = len(files)
        self.done = 0

    def get_next(self):
        try:
            path = next(self.files)
        except StopIteration:
            return None
        x, _, _, _ = preprocess(path, self.imgsz)
        self.done += 1
        if self.done % 25 == 0:
            print(f"  calibrated {self.done}/{self.n}")
        return {self.input_name: x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib", default="data/calibration_files.json")
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--exclude-head", action="store_true",
                    help="leave detection head nodes in FP32")
    args = ap.parse_args()

    config = utils.load_config(args.config)
    imgsz = config["benchmark"]["imgsz"]

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    files = json.loads(Path(args.calib).read_text())
    print(f"calibrating on {len(files)} held-out images")

    exclude = []
    if args.exclude_head:
        import onnx
        m = onnx.load(args.model)
        exclude = [n.name for n in m.graph.node if n.name.startswith("/model.23/")]
        print(f"excluding {len(exclude)} detection-head nodes from quantization")

    quantize_static(
        model_input=args.model,
        model_output=args.output,
        calibration_data_reader=Reader(files, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=exclude,
    )

    before = utils.model_size_mb(args.model)
    after = utils.model_size_mb(args.output)
    print(f"\n{args.model}  {before} MB")
    print(f"{args.output}  {after} MB")
    print(f"reduction {before / after:.2f}x")


if __name__ == "__main__":
    main()
