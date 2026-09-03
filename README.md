# YOLO Edge Optimization

Taking a detection model from research checkpoint to deployment across two vendors' accelerators, and measuring what every optimization actually costs.

The question is not "how fast is YOLO." It is **whether published latency numbers transfer**, and what happens to the same weights under a different runtime, on different silicon, at different precision.

**Status:** Apple Silicon and NVIDIA (A100 + T4) complete — PyTorch, ONNX Runtime, CoreML, TensorRT, FP16, INT8. Video tracking and VisDrone fine-tuning in progress.

---

## Headline finding

**Export helps by 3.5× on NVIDIA and hurts by 1.7× on Apple. Quantization helps by 1.24× on NVIDIA and hurts by 11.6× on Apple. Same model, same graph, same export settings.**

| | Apple M4 Max | NVIDIA A100 (400 W) |
|---|---:|---:|
| Native PyTorch | 6.126 ms | 12.269 ms |
| Best exported FP32 | 10.335 ms (ONNX CoreML) | **3.479 ms** (TensorRT) |
| **Export effect** | **1.69× slower** | **3.50× faster** |
| INT8 | 120.186 ms | **2.806 ms** |
| **INT8 effect** | **11.6× slower** | **1.24× faster** |

A 43× divergence in the direction and magnitude of the same optimization.

---

## Complete results

Batch 1, 640×640, 50 warmup runs discarded, 300 timed iterations. Accuracy: COCO val2017, 5000 images.

### Apple M4 Max

| Variant | Runtime | mAP@.5:.95 | p50 | Size | Partitions |
|---|---|---:|---:|---:|---:|
| YOLO26s FP32 | PyTorch MPS | 0.477 † | **6.126 ms** | 19.5 MB | — |
| YOLO26s FP32 | ONNX CoreML | **0.477** | 10.335 ms | 36.5 MB | 8 |
| YOLO26s FP32 | ONNX CPU | 0.477 | 23.241 ms | 36.5 MB | — |
| YOLO26s FP32 | PyTorch CPU | 0.477 † | 33.295 ms | 19.5 MB | — |
| YOLO26s INT8 (naive) | ONNX | **0.000** | not measured ‡ | 9.9 MB | — |
| YOLO26s INT8 (head FP32) | ONNX CPU | **0.464** | 16.805 ms | 11.2 MB | — |
| YOLO26s INT8 (head FP32) | ONNX CoreML | 0.464 | 120.186 ms | 11.2 MB | **191** |
| YOLO11s FP32 | PyTorch MPS | not measured | 6.484 ms | 18.4 MB | — |
| YOLO11s FP32 | ONNX CoreML | not measured | **8.452 ms** | 36.3 MB | **5** |

### NVIDIA A100

| Variant | Runtime | mAP@.5:.95 | p50 | Size |
|---|---|---:|---:|---:|
| YOLO26s FP32 | PyTorch CUDA | 0.477 † | 12.269 ms | 19.5 MB |
| YOLO26s FP32 | TensorRT | **0.477** | 3.479 ms | 114.5 MB |
| YOLO26s FP16 | TensorRT | not measured | 2.988 ms | 79.3 MB |
| YOLO26s INT8 | TensorRT | **0.444** | **2.806 ms** | **30.4 MB** |

### NVIDIA T4 (70 W, edge-representative)

Accuracy not re-measured; mAP is device-independent and the A100 figures apply.

| Variant | Runtime | p50 | p95 | FPS | Size |
|---|---|---:|---:|---:|---:|
| YOLO26s FP32 | TensorRT | 9.956 ms | 10.943 ms | 100.4 | 116.0 MB |
| YOLO26s FP16 | TensorRT | 4.734 ms | 5.197 ms | 211.2 | 101.3 MB |
| YOLO26s INT8 | TensorRT | **3.952 ms** | 4.347 ms | **253.1** | **33.1 MB** |

† FP32 PyTorch, ONNX, and TensorRT are numerically equivalent; 0.477 was measured on the ONNX and TensorRT paths and applies to all FP32 variants.
‡ Latency not benchmarked — the model produced zero detections, so timing it would measure a non-functional graph.

**FP32 accuracy matches to three decimals across both platforms** (0.477 / 0.477), against Ultralytics' published 0.478. Two independent evaluation paths agreeing to 0.001 validates both.

---

## Finding 1 — Apple's Neural Engine cannot compile NMS-free detection heads

YOLO26 replaces Non-Maximum Suppression with a one-to-one head, emitting final boxes without a suppression stage. That eliminates a post-processing loop and replaces it with tensor indexing.

### The operator difference

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Graph nodes | 384 | 320 |
| Indexing ops (`TopK`, `GatherElements`, `Mod`, `Expand`, `ReduceMax`, `Cast`) | **10** | **0** |

Not fewer — zero. The operator family is entirely absent from the conventional head.

All ten cluster in the final 8% of the graph, inside `/model.23/`:

```
idx 367  ReduceMax    idx 373  TopK_1
idx 368  TopK         idx 375  Mod
idx 370  Expand       idx 378  GatherElements_1
idx 371  GatherElements  idx 379  Cast
idx 372  Flatten      idx 381  Expand_1
                      idx 382  GatherElements_2
```

`TopK` selects highest-scoring candidates, `Mod` converts flat indices to per-class indices, `GatherElements` retrieves the boxes.

### The cost

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Nodes supported by CoreML | 359 / 384 (93.5%) | 314 / 320 (98.1%) |
| **Partitions** | **8** | **5** |
| ONNX CoreML p50 | 10.335 ms | **8.452 ms** |

93.5% coverage sounds excellent, but the 25 unsupported nodes are scattered rather than contiguous, splitting execution into 8 segments. Each boundary is a Neural Engine ↔ CPU handoff.

**The 6.5% of unsupported operators cost more than the 93.5% of supported ones saved.**

### Is this architectural?

**No.** TensorRT compiled the identical 384-node graph — `TopK`, `GatherElements`, `Mod` and all — into an engine **3.50× faster** than eager PyTorch CUDA. NVIDIA absorbed exactly the operators Apple rejected.

This is a vendor coverage gap, not a property of NMS-free detection heads. Establishing that required measuring on both.

---

## Finding 2 — INT8 collapse is a quantizer limitation, not an architectural one

### On ONNX Runtime, naive INT8 produces a dead model

3.70× smaller and **0.000 mAP across all 5000 images.** Not degraded — dead:

```
FP32   conf min/max  0.001926 / 0.932771     conf > 0.25:  14
INT8   conf min/max  0.000000 / 0.000000     conf > 0.25:   0
INT8   first box     [0, 0, 2.707, 2.707]
```

Every confidence exactly zero. Box coordinates repeat one value, so regression outputs also snapped to a single quantization level.

**Mechanism.** The confidence path ends in a sigmoid. Its input logits are strongly negative for ~294 of 300 candidates and positive for the few real detections. Per-tensor INT8 offers 256 levels across the calibrated range; when large-magnitude negatives dominate that range, positive logits fall into the same bucket. Sigmoid of a large negative is zero.

Excluding 95 `/model.23/` nodes from quantization recovers the model: **0.464 mAP**, at 3.25× compression instead of 3.70×.

### On TensorRT, the same head quantizes without difficulty

**0.444 mAP, 2.806 ms, 30.4 MB — no exclusion required.** Full-graph INT8, head included.

So the collapse was **ONNX Runtime's per-tensor activation calibration**, not the architecture. A second hypothesis that measuring on a second vendor overturned.

### But TensorRT loses more accuracy

| | ONNX (head excluded) | TensorRT (full graph) |
|---|---:|---:|
| mAP@.5:.95 | **0.464** | 0.444 |
| Loss vs FP32 | −0.013 (−2.7%) | −0.033 (−6.9%) |
| Size reduction | 3.25× | **3.77×** |

This inverts on inspection. ONNX preserved accuracy *because it failed* — 95 nodes stayed FP32. TensorRT quantized everything, so it got more compression, more speed, and more accuracy loss.

**Neither tool is better. They made different tradeoffs on the same problem, and only measuring both reveals that the tradeoff exists.**

---

## Finding 3 — Quantization damages small objects disproportionately on ONNX, but not on TensorRT

| Platform | | FP32 | INT8 | Relative loss |
|---|---|---:|---:|---:|
| ONNX | small | 0.302 | 0.282 | **−6.6%** |
| ONNX | medium | 0.524 | 0.507 | −3.2% |
| ONNX | large | 0.642 | 0.625 | −2.6% |
| TensorRT | small | 0.303 | 0.276 | **−8.9%** |
| TensorRT | medium | 0.524 | 0.477 | −9.0% |
| TensorRT | large | 0.642 | 0.601 | −6.4% |

**On ONNX the pattern is clean.** Absolute loss is nearly uniform across sizes (0.020 / 0.017 / 0.017), so relative loss scales inversely with baseline accuracy: 6.6% for small against 2.6% for large, a 2.5× difference. Small objects carry less signal, so identical quantization noise costs proportionally more.

**On TensorRT it does not hold.** Medium objects lost 9.0% relative, marginally more than small at 8.9%, and lost more in absolute terms than any other bucket (0.047 against 0.027 for small). The ordering small > medium > large breaks.

Two candidate explanations, neither tested here:

The quantizers touched different parts of the model. ONNX left 95 detection-head nodes in FP32; TensorRT quantized the full graph. If the head disproportionately affects small-object localization, excluding it would preserve small-object accuracy specifically — making the clean ONNX pattern an artifact of that exclusion rather than a property of quantization.

TensorRT's larger overall loss (−0.033 against −0.013) may simply dominate any size-dependent structure.

**The honest claim is narrower than it first appeared:** quantization damaged small objects disproportionately under one quantizer and did not under the other. Whether size-dependent damage is a general property of INT8 is not established by this data. Settling it needs a controlled comparison — same exclusion policy, both quantizers — which is left as future work.

The practical implication survives regardless: aggregate mAP hides substantial variation across object scales, and any deployment where most targets are small should evaluate by size rather than on the headline number.

---

## Finding 4 — Precision gains scale inversely with hardware headroom

The same INT8 and FP16 engines were built and benchmarked on two NVIDIA cards: an A100-SXM4 (400 W, 40 GB) and a T4 (70 W, 16 GB).

| Precision | A100 p50 | speedup | T4 p50 | speedup |
|---|---:|---:|---:|---:|
| FP32 | 3.479 ms | — | 9.956 ms | — |
| FP16 | 2.988 ms | 1.16× | 4.734 ms | **2.10×** |
| INT8 | 2.806 ms | **1.24×** | 3.952 ms | **2.52×** |

**Reduced precision is roughly twice as valuable on the weaker card.**

The mechanism is bottleneck location. Reduced precision accelerates arithmetic, and only helps when arithmetic is the constraint. At batch 1 with a 21-GFLOP model, the A100 finishes the math and waits — for memory transfers, kernel launches, and host dispatch. The T4, with roughly 40× less compute, is genuinely saturated.

Two pieces of supporting evidence:

**The A100 is only 2.86× faster than the T4 at FP32**, despite far more than 2.86× the peak throughput. Most of its capability is unused at this batch size.

**PyTorch CUDA on the A100 (12.269 ms) was slower than PyTorch MPS on an M4 Max laptop (6.126 ms).** A datacenter GPU losing to a laptop only happens when the GPU is not the constraint. TensorRT's 3.50× gain over eager PyTorch on the same card is the size of the framework overhead being removed.

A secondary observation: the T4 FP16 engine is **101.3 MB against the A100's 79.3 MB**, and shrank only 1.15× from FP32 versus the A100's 1.44×. TensorRT compiles hardware-specific kernels, and less of the graph appears to execute in half precision on Turing than on Ampere.

**The general principle: an optimization only pays where the thing it optimizes is the bottleneck.** The same reasoning explains the CoreML result — Apple's Neural Engine is fast at convolution, but partition handoffs were the constraint, so faster convolution did not help.

---

## Finding 5 — The benchmark itself was misleading

Five rounds of PyTorch measurement produced three successive interpretations, two of them wrong.

**Round 1, synthetic input.** YOLO26s 5.779 ms, YOLO11s 5.585 ms. Apparent conclusion: NMS-free provides no benefit. *Wrong* — random pixels produce almost no detections, so post-processing was never exercised.

**Round 2, real image.** YOLO26s 6.126 ms (+0.347), YOLO11s 6.484 ms (+0.899). Ranking inverted.

**Round 3, forcing detections.** At conf 0.01, YOLO26s moved +0.001 ms between 6 and 19 detections; YOLO11s moved +0.023 ms between 5 and 25. **Neither model scales with detection count.** Second hypothesis failed.

**Round 4, resolution.** Under ONNX Runtime:

| Runtime | Synthetic | Real | Δ |
|---|---:|---:|---:|
| PyTorch MPS | 5.779 ms | 6.126 ms | **+0.347 ms** |
| ONNX CoreML | 10.319 ms | 10.335 ms | **+0.016 ms** |

Identical model, input, and hardware. **The effect is Ultralytics runtime overhead, not a model property.** An exported graph runs identical operations regardless of content; the Python path constructs result objects and moves detections off-GPU, and that scales with what was found.

A framework overhead was nearly published as a model characteristic.

### Device validation

| Device | p50 | FPS | Speedup |
|---|---:|---:|---:|
| CPU | 33.295 ms | 30.0 | 1.0× |
| MPS | 5.779 ms | 173.0 | **5.76×** |

Confirms MPS was engaged rather than silently falling back. CPU-only inference lands at 30.03 FPS — precisely at the real-time threshold, with no headroom for decoding or tracking.

---

## Practical implications

1. **Latency ordering does not transfer between vendors.** Export was a 3.5× win on NVIDIA and a 1.7× loss on Apple.
2. **Benchmark on data resembling deployment.** Synthetic tensors reversed a model ranking.
3. **Detection count is not the variable to control.** Near-zero effect across a 5× range.
4. **Check partition counts, not node coverage.** 93.5% CoreML support performed worse than 98.1%.
5. **Compression does not imply speed.** The same 3.25×-smaller file was 1.38× faster on CPU and 11.6× slower on CoreML.
6. **Quantizer choice is an accuracy/compression tradeoff, not a quality ranking.** ONNX preserved 2 mAP points by failing to quantize the head; TensorRT gained 0.5× compression by succeeding.
7. **Report accuracy by object size.** Aggregate mAP hid variation of 6.6% to 9.0% relative across scale buckets, and the ordering differed between quantizers.
8. **NVIDIA eager PyTorch was slower than Apple eager PyTorch** (12.269 vs 6.126 ms). At this model size, framework dispatch dominates and the A100 is idle. Compilation is what unlocks it.
9. **Quantization is worth more on weaker hardware.** INT8 bought 2.52× on a T4 and 1.24× on an A100. Benchmarking optimizations on the most powerful available GPU understates their value at the edge.

---

## Method

**GPU synchronization.** GPU calls are asynchronous — control returns to Python before computation completes. Without an explicit barrier (`torch.mps.synchronize()` / `torch.cuda.synchronize()`), a timer measures dispatch latency and reports numbers several times too low. Every timed call is bracketed.

**Warmup.** First 50 runs discarded — weight transfer, kernel compilation, and cache population are one-time costs.

**Distribution over mean.** 300 iterations, reported as p50, p95, min, max, std. A 52 ms mean is compatible with a stable model and with one that spikes to 200 ms.

**Fixed input.** Same frame across every variant within a condition.

**Batch size 1.** Batching flatters throughput; a camera delivers one frame at a time.

**Calibration/evaluation split.** ONNX INT8 calibrated on the last 100 COCO val images, held out from the 4900 used for evaluation. TensorRT INT8 calibrated on COCO128.

**Provenance.** Every result JSON embeds the configuration and full environment that produced it.

---

## Environments

| | Apple | NVIDIA A100 | NVIDIA T4 |
|---|---|---|---|
| Hardware | M4 Max, 64 GB, 40-core GPU | A100-SXM4-40GB, 400 W | Tesla T4, 15 GB, 70 W |
| OS | macOS 26.6.2 (arm64) | Colab, driver 580.82.07 | Colab |
| Python | 3.14.2 | 3.13.15 | 3.13.15 |
| PyTorch | 2.14.0 | 2.14.0+cu130 | 2.11.0+cu128 |
| Ultralytics | 8.4.138 | 8.4.138 | 8.4.138 |
| Runtime | ONNX Runtime 1.29.0, CoreML | TensorRT | TensorRT |

The two NVIDIA sessions ran different Colab images and therefore different torch versions. TensorRT engines execute compiled kernels rather than framework operations, so this affects host-side overhead only; it is noted rather than controlled for.

Versions pinned in `requirements.txt`. ONNX Runtime fuses differently between releases; cross-version latency comparison is not valid. TensorRT engines are GPU-specific and not portable.

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

Prediction dumps (`results/*_predictions.json`) are regenerated by `evaluate.py` and not tracked.

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pycocotools

mkdir -p data/coco && cd data/coco
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && cd ../..

python -c "from ultralytics import YOLO; YOLO('weights/yolo26s.pt').export(
    format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)"

# Baseline — must reproduce ~0.477 before trusting anything downstream
python src/evaluate.py --model weights/yolo26s.onnx --variant yolo26s_fp32

python src/quantize.py --model weights/yolo26s.onnx \
  --output weights/yolo26s_int8_head_fp32.onnx --exclude-head
python src/evaluate.py --model weights/yolo26s_int8_head_fp32.onnx --variant yolo26s_int8
```

TensorRT builds require an NVIDIA GPU. Install `ultralytics` with `--no-deps` in Colab to avoid a torch/torchvision ABI conflict that breaks `torchvision::nms`.

---

## In progress

- **VisDrone fine-tuning** — dense small-object domain, where the quantization penalty matters most
- **Video pipeline** — BoT-SORT and ByteTrack, end-to-end FPS by stage rather than per-frame inference

## Limitations

- T4 accuracy was not re-evaluated; mAP is hardware-independent and the A100 figures apply. T4 results are latency and size only.
- TensorRT FP16 accuracy was not evaluated on full COCO, on either card.
- The two NVIDIA sessions used different torch versions (2.14.0+cu130 and 2.11.0+cu128), a consequence of differing Colab images.
- Detection counts tested span 0 to 25. VisDrone frames carry dozens to hundreds.
- YOLO11s accuracy was not measured; it appears only as a latency and graph-structure comparison. Published mAP is 0.470.
- Pretrained COCO weights, not domain-specific.
- ONNX calibration used 100 images, TensorRT used COCO128. Calibration set size was not swept on either.
- The two INT8 variants are not directly comparable: ONNX excluded the detection head, TensorRT did not. Size-dependent accuracy differences may reflect that exclusion rather than quantizer behaviour.
- ONNX Runtime emits a shape-inference preprocessing warning during quantization that was not acted on.
- Partition counts are reported by ONNX Runtime. Specific rejected nodes were inferred from operator-support patterns rather than read from the runtime.
- TensorRT engine layer inspection failed due to a version mismatch between the pip-installed `tensorrt` package and the runtime Ultralytics used; fusion behaviour was inferred from latency rather than read directly.

---

**Abdul Rafay Mohd** · M.S. Artificial Intelligence, University of North Texas
[github.com/Mohd-Abdul-Rafay](https://github.com/Mohd-Abdul-Rafay) · [linkedin.com/in/mohd-abdul-rafay](https://linkedin.com/in/mohd-abdul-rafay)
