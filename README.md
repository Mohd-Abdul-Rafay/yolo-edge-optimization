# YOLO Edge Optimization

Taking a detection model from research checkpoint to deployment, and measuring what every optimization actually costs.

The question is not "how fast is YOLO." It is **whether published latency numbers mean anything**, and what happens to the same weights under a different runtime, on different silicon, at different precision.

**Status:** PyTorch, ONNX, and INT8 complete on Apple Silicon. TensorRT and video tracking in progress.

---

## Headline finding

**One architectural decision produces three independent deployment failures, all traceable to the same 95 nodes.**

YOLO26 replaces Non-Maximum Suppression with a one-to-one detection head, emitting final boxes without a suppression stage. In PyTorch this is a win. Everywhere else it is a liability:

| Failure | Consequence |
|---|---|
| Head introduces 10 tensor-indexing ops absent in YOLO11 | CoreML graph splits into 8 partitions vs 5 — ONNX runs **1.69× slower** than native PyTorch |
| Head's confidence logits exceed INT8 dynamic range | Naive quantization yields **0.000 mAP** — every detection collapses to zero confidence |
| Excluding the head fixes accuracy, but QDQ nodes remain | Partitions explode to **191**, CoreML runs **11.6× slower** than FP32 |

And the general result: **size, latency, and accuracy do not move together, and the direction depends on the runtime.** The identical INT8 file is 1.38× faster on CPU and 11.6× slower on CoreML.

---

## Full results

Apple M4 Max, batch 1, 640×640. Latency: 50 warmup runs discarded, 300 timed iterations, real image input. Accuracy: COCO val2017, 5000 images, conf 0.001.

| Variant | Size | mAP@.5:.95 | CPU p50 | CoreML p50 | MPS p50 | Partitions |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 19.48 MB | — | 33.295 ms | — | **6.126 ms** | — |
| ONNX FP32 | 36.52 MB | **0.477** | 23.241 ms | 10.335 ms | — | 8 |
| INT8 naive | 9.88 MB | **0.000** | — | — | — | — |
| INT8 head-FP32 | 11.24 MB | **0.464** | **16.805 ms** | 120.186 ms | — | **191** |

YOLO11s comparison, same conditions:

| Variant | Size | MPS p50 | CoreML p50 | Partitions | Indexing ops |
|---|---:|---:|---:|---:|---:|
| YOLO26s | 19.48 MB | **6.126 ms** | 10.335 ms | 8 | 10 |
| YOLO11s | 18.42 MB | 6.484 ms | **8.452 ms** | 5 | **0** |

---

## Pipeline validation

The exported graph contains only the network. Letterboxing, coordinate recovery, class-ID mapping, and COCO JSON formatting are reimplemented here to match the Ultralytics pipeline.

**Measured FP32 mAP@0.5:0.95 = 0.477 against Ultralytics' published 0.478.** A 0.001 gap.

Any accuracy change reported below is therefore attributable to the optimization under test, not to a preprocessing mismatch. This validation ran before any quantization work.

---

## Finding 1 — The runtime inversion

### Operator difference

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Graph nodes | 384 | 320 |
| Indexing ops (`TopK`, `GatherElements`, `Mod`, `Expand`, `ReduceMax`, `Cast`) | **10** | **0** |

Not fewer — zero. The operator family is entirely absent from the conventional head.

### Location

All ten cluster in the final 8% of the graph, inside `/model.23/`:

```
idx 367   ReduceMax        idx 373   TopK_1
idx 368   TopK             idx 375   Mod
idx 370   Expand           idx 378   GatherElements_1
idx 371   GatherElements   idx 379   Cast
idx 372   Flatten          idx 381   Expand_1
                           idx 382   GatherElements_2
```

That is a top-k selection with index arithmetic. `TopK` picks highest-scoring candidates, `Mod` converts flat indices to per-class indices, `GatherElements` retrieves the corresponding boxes — how a one-to-one head produces detections without a suppression loop.

### Cost on CoreML

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Nodes supported | 359 / 384 (93.5%) | 314 / 320 (98.1%) |
| **Partitions** | **8** | **5** |

93.5% coverage sounds excellent, but the 25 unsupported nodes are scattered rather than contiguous, splitting execution into 8 segments. Each boundary is a Neural Engine ↔ CPU handoff: tensors copied between memory contexts, engines synchronized.

**The 6.5% of unsupported operators cost more than the 93.5% of supported ones saved.**

Removing NMS eliminates a post-processing loop and replaces it with tensor indexing. On this hardware the replacement costs more than the thing it replaced. Vendor benchmarks report the PyTorch number.

---

## Finding 2 — INT8 collapses the detection head

Static INT8 quantization, per-channel weights, QDQ format, calibrated on 100 images held out from the evaluation set.

### Naive quantization produces a dead model

3.70× smaller and **0.000 mAP across all 5000 images.** Not degraded — dead. Diagnostic output on a single image:

```
FP32   conf min/max  0.001926 / 0.932771     conf > 0.25:  14
INT8   conf min/max  0.000000 / 0.000000     conf > 0.25:   0
INT8   first box     [0, 0, 2.707, 2.707]
```

Every confidence is exactly zero. Box coordinates repeat the same value, meaning regression outputs also snapped to a single quantization level.

**Mechanism.** The confidence path ends in a sigmoid. Its input logits are strongly negative for the vast majority of 300 candidates and positive for the few real detections. Per-tensor INT8 offers 256 levels across the calibrated range; when large-magnitude negatives dominate that range, positive logits collapse into the same bucket. Sigmoid of a large negative is zero, so every candidate reads zero confidence.

### Excluding the head recovers the model

95 nodes under `/model.23/` left in FP32:

| | Naive INT8 | Head-excluded INT8 |
|---|---:|---:|
| Size | 9.88 MB (3.70×) | 11.24 MB (3.25×) |
| mAP@.5:.95 | 0.000 | **0.464** |
| CPU p50 | — | 16.805 ms |

0.45× of the available compression buys back a working model. 1.3 mAP points lost against FP32.

### Accuracy loss by object size

| | FP32 | INT8 | Δ | Relative |
|---|---:|---:|---:|---:|
| Small | 0.302 | 0.282 | −0.020 | **−6.6%** |
| Medium | 0.524 | 0.507 | −0.017 | −3.2% |
| Large | 0.642 | 0.625 | −0.017 | −2.6% |

Absolute loss is nearly uniform, but relative damage to small objects is **2.5× that of large ones**. Small objects carry less signal, so identical quantization noise costs proportionally more. Relevant to any aerial, surveillance, or inspection domain where most targets are small.

---

## Finding 3 — INT8 is fast on CPU and unusable on CoreML

The same file, two runtimes:

| Runtime | FP32 | INT8 | Change |
|---|---:|---:|---|
| CPU | 23.241 ms | **16.805 ms** | **1.38× faster** |
| CoreML | 10.335 ms | **120.186 ms** | **11.6× slower** |

QDQ quantization inserts Quantize/DeQuantize node pairs throughout the graph:

| | FP32 ONNX | INT8 ONNX |
|---|---:|---:|
| Graph nodes | 384 | **1262** |
| Supported by CoreML | 359 (93.5%) | **273 (21.6%)** |
| **Partitions** | **8** | **191** |

191 partitions means 191 handoffs between the Neural Engine and CPU per inference. The model is a third the size and runs at 8.32 FPS.

**A single INT8 export is the right choice for CPU deployment and the wrong choice for Neural Engine deployment.** Model compression and inference latency are decoupled, and their relationship inverts across backends.

---

## Finding 4 — The benchmark itself was misleading

Before the export work, five rounds of PyTorch measurement produced three successive interpretations, two of them wrong. The sequence is kept because the wrong turns are the useful part.

**Round 1, synthetic input.** YOLO26s 5.779 ms, YOLO11s 5.585 ms. Apparent conclusion: NMS-free provides no benefit. *Wrong* — random pixels produce almost no detections, so post-processing was never exercised.

**Round 2, real image.** YOLO26s 6.126 ms (+0.347), YOLO11s 6.484 ms (+0.899). Ranking inverted. Apparent conclusion: NMS is the cause and cost should scale with detection count.

**Round 3, forcing detections.** Confidence lowered to 0.01. YOLO26s moved +0.001 ms between 6 and 19 detections; YOLO11s moved +0.023 ms between 5 and 25. *Neither model scales with detection count.* Hypothesis failed.

**Round 4, the resolution.** Running the same comparison under ONNX Runtime:

| Runtime | Synthetic | Real | Δ |
|---|---:|---:|---:|
| PyTorch MPS | 5.779 ms | 6.126 ms | **+0.347 ms** |
| ONNX CoreML | 10.319 ms | 10.335 ms | **+0.016 ms** |

Identical model, input, and hardware. The real-image penalty is 0.347 ms in PyTorch and 0.016 ms under ONNX — inside the noise floor.

**The effect is Ultralytics runtime overhead, not a model property.** An exported graph runs identical operations regardless of content; the Python inference path constructs result objects, moves detections off-GPU, and formats output, and that work scales with what was found.

A framework overhead was nearly published as a model characteristic.

### Device validation

| Device | p50 | FPS | Speedup |
|---|---:|---:|---:|
| CPU | 33.295 ms | 30.0 | 1.0× |
| MPS | 5.779 ms | 173.0 | **5.76×** |

Run to confirm MPS was engaged rather than silently falling back. CPU-only inference lands at 30.03 FPS — precisely at the real-time video threshold, with no headroom for decoding, tracking, or rendering.

---

## Practical implications

1. **Benchmark on data resembling deployment.** Synthetic tensors reversed a model ranking here.
2. **Detection count is not the variable to control.** Near-zero effect across a 5× range; input realism mattered instead.
3. **Measure on the runtime you will ship.** Latency ordering is not preserved across PyTorch MPS, ONNX CPU, and CoreML.
4. **Check partition counts, not node coverage.** 93.5% CoreML support performed worse than 98.1%; scatter matters more than percentage.
5. **Quantize the head separately, or not at all.** Detection heads carry wide-dynamic-range logits that per-tensor INT8 cannot represent.
6. **Compression does not imply speed.** The same 3.25×-smaller file was 1.38× faster on CPU and 11.6× slower on CoreML.
7. **Report accuracy by object size.** Aggregate mAP hid a 2.5× difference in relative quantization damage.

---

## Method

**GPU synchronization.** GPU calls are asynchronous — control returns to Python before computation completes. Without an explicit barrier (`torch.mps.synchronize()` / `torch.cuda.synchronize()`), a timer measures dispatch latency rather than execution and reports numbers several times too low. Every timed call is bracketed.

**Warmup.** First 50 runs discarded — weight transfer, kernel compilation, and cache population are one-time costs.

**Distribution over mean.** 300 iterations, reported as p50, p95, min, max, std. A 52 ms mean is compatible with a stable model and with one that spikes to 200 ms. Only one ships.

**Fixed input.** Same frame across every variant within a condition.

**Batch size 1.** Batching flatters throughput; a camera delivers one frame at a time.

**Calibration/evaluation split.** INT8 calibrated on the last 100 COCO val images, held out from the 4900 used for evaluation. Calibrating and evaluating on the same images would be optimistic.

**Provenance.** Every result JSON embeds the configuration and full environment that produced it.

---

## Environment

| | |
|---|---|
| Hardware | Apple M4 Max, 64 GB unified memory, 40-core GPU |
| OS | macOS 26.6.2 (arm64) |
| Python | 3.14.2 |
| PyTorch | 2.14.0 |
| Ultralytics | 8.4.138 |
| ONNX / Runtime | 1.22.0 / 1.29.0 |
| Providers | CoreML, CPU |

Versions pinned in `requirements.txt`. ONNX Runtime fuses operations differently between releases; cross-version latency comparison is not valid.

---

## Layout

```
configs/benchmark.yaml     measurement parameters, shared by all scripts
src/utils.py               timing with device-aware GPU synchronization
src/benchmark.py           PyTorch checkpoint latency
src/benchmark_onnx.py      ONNX Runtime latency, provider-configurable
src/quantize.py            static INT8 with optional head exclusion
src/evaluate.py            COCO mAP with reimplemented pre/postprocessing
results/*.json             one per variant, config and environment embedded
```

Prediction dumps (`results/*_predictions.json`) are regenerated by `evaluate.py` and not tracked; only summary metrics are committed.

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pycocotools

# COCO val2017
mkdir -p data/coco && cd data/coco
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && cd ../..

# Export
python -c "from ultralytics import YOLO; YOLO('weights/yolo26s.pt').export(
    format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)"

# Baseline accuracy — must reproduce ~0.477 before trusting anything downstream
python src/evaluate.py --model weights/yolo26s.onnx --variant yolo26s_fp32

# INT8, head excluded
python src/quantize.py --model weights/yolo26s.onnx \
  --output weights/yolo26s_int8_head_fp32.onnx --exclude-head
python src/evaluate.py --model weights/yolo26s_int8_head_fp32.onnx --variant yolo26s_int8

# Latency
python src/benchmark_onnx.py --model weights/yolo26s_int8_head_fp32.onnx \
  --variant int8_coreml --image data/samples/dense.jpg \
  --providers CoreMLExecutionProvider,CPUExecutionProvider
```

Weights download on first use. Changing any value in `configs/benchmark.yaml` invalidates comparison with existing results.

---

## In progress

- **TensorRT on NVIDIA** — does partition fragmentation exist outside Apple silicon, or is it CoreML-specific?
- **Fine-tuning on VisDrone** — dense small-object domain, where the 6.6% relative quantization loss matters most
- **Video pipeline** — BoT-SORT and ByteTrack, end-to-end FPS by stage rather than per-frame inference

## Limitations

- Single hardware platform. The partition behaviour may be CoreML-specific; the TensorRT comparison addresses this.
- Detection counts tested span 0 to 25. VisDrone frames carry dozens to hundreds.
- Pretrained COCO weights, not domain-specific.
- Calibration used 100 images. Larger sets may narrow the INT8 accuracy gap; this was not swept.
- ONNX Runtime emits a shape-inference preprocessing warning during quantization that was not acted on. It may affect which nodes quantize.
- Partition counts are reported by ONNX Runtime. The specific rejected nodes were inferred from operator-support patterns rather than read from the runtime.
- Per-tensor versus per-channel activation quantization was not compared; only per-channel weights were tested.

---

**Abdul Rafay Mohd** · M.S. Artificial Intelligence, University of North Texas
[github.com/Mohd-Abdul-Rafay](https://github.com/Mohd-Abdul-Rafay) · [linkedin.com/in/mohd-abdul-rafay](https://linkedin.com/in/mohd-abdul-rafay)
