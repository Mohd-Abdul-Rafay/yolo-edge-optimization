# YOLO Edge Optimization

Taking a detection model from research checkpoint to real-time deployment, and measuring what every optimization actually costs.

The question is not "how fast is YOLO." It is **whether published latency numbers mean anything**, and what changes when the same weights run under a different runtime, on different silicon, at different precision.

**Status:** Phases 1 and 3 (partial) complete. INT8, TensorRT, and video tracking in progress.

---

## Headline finding

**An architectural choice that optimizes one runtime pessimizes another, and the ranking inverts.**

YOLO26 replaces Non-Maximum Suppression with a one-to-one detection head, producing final boxes without a separate suppression stage. In PyTorch this is a win. Exported to ONNX and dispatched to Apple's Neural Engine, it is a loss — and for the same reason.

| | YOLO26s | YOLO11s | Faster |
|---|---:|---:|---|
| PyTorch MPS | **6.126 ms** | 6.484 ms | YOLO26s by 5.5% |
| ONNX + CoreML | 10.335 ms | **8.452 ms** | YOLO11s by 22.3% |

The mechanism is traceable end to end. It is documented below.

---

## Full results

Apple M4 Max, batch 1, 640×640, 50 warmup runs discarded, 300 timed iterations, real image input.

| Variant | Runtime | p50 | p95 | FPS | Size |
|---|---|---:|---:|---:|---:|
| YOLO26s | PyTorch MPS | **6.126 ms** | 6.231 ms | 163.2 | 19.48 MB |
| YOLO11s | PyTorch MPS | 6.484 ms | 6.590 ms | 154.2 | 18.42 MB |
| YOLO11s | ONNX CoreML | 8.452 ms | 8.494 ms | 118.3 | 36.29 MB |
| YOLO26s | ONNX CoreML | 10.335 ms | 10.440 ms | 96.8 | 36.52 MB |
| YOLO26s | ONNX CPU | 23.241 ms | 24.616 ms | 43.0 | 36.52 MB |
| YOLO26s | PyTorch CPU | 33.295 ms | 33.967 ms | 30.0 | 19.48 MB |

Two things worth flagging before the analysis:

**ONNX export did not improve latency on this hardware.** Native PyTorch MPS beat every exported variant. Graph optimization is real — ONNX CPU beat PyTorch CPU by 1.43× — but it could not overcome Metal.

**Export nearly doubled file size**, 19.48 MB to 36.52 MB. ONNX stores weights as uncompressed tensors; PyTorch checkpoints are zip-compressed.

---

## Why the ranking inverts

### 1. The operator difference

Exporting both models and counting operator types:

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Total graph nodes | 384 | 320 |
| Indexing ops (`TopK`, `GatherElements`, `Mod`, `Expand`, `ReduceMax`, `Cast`) | **10** | **0** |

Not fewer — zero. The operator family is entirely absent from YOLO11's conventional head.

### 2. Where they sit

All ten cluster in the final 8% of YOLO26's graph, inside `/model.23/` — the detection head:

```
idx 367   ReduceMax        /model.23/ReduceMax
idx 368   TopK             /model.23/TopK
idx 370   Expand           /model.23/Expand
idx 371   GatherElements   /model.23/GatherElements
idx 372   Flatten          /model.23/Flatten
idx 373   TopK             /model.23/TopK_1
idx 375   Mod              /model.23/Mod
idx 378   GatherElements   /model.23/GatherElements_1
idx 379   Cast             /model.23/Cast
idx 381   Expand           /model.23/Expand_1
idx 382   GatherElements   /model.23/GatherElements_2
```

That sequence is a top-k selection with index arithmetic: `TopK` picks the highest-scoring candidates, `Mod` converts flat indices to per-class indices, `GatherElements` retrieves the corresponding boxes. It is how a one-to-one head produces detections without a suppression loop.

### 3. What it costs on CoreML

| | YOLO26s | YOLO11s |
|---|---:|---:|
| Nodes supported by CoreML | 359 / 384 (93.5%) | 314 / 320 (98.1%) |
| **Graph partitions** | **8** | **5** |

93.5% coverage sounds excellent. But the 25 unsupported nodes are scattered rather than contiguous, splitting execution into 8 segments. Each boundary is a handoff between the Neural Engine and the CPU: tensors copied between memory contexts, execution engines synchronized.

**The 6.5% of unsupported operators cost more than the 93.5% of supported ones saved.**

Convolution and elementwise arithmetic — 280 of 384 nodes — map cleanly to the Neural Engine. Dynamic gather-and-index does not.

### 4. The tradeoff, stated plainly

Removing NMS eliminates a post-processing loop. It replaces that loop with tensor-indexing operators. Those operators are poorly supported on Apple's Neural Engine. On this hardware, the replacement costs more than the thing it replaced.

Vendor benchmarks report the PyTorch number.

---

## A second finding: the benchmark itself was misleading

Before the export work, five rounds of measurement on PyTorch produced three successive interpretations, two of which were wrong. The sequence is documented because the wrong turns are the useful part.

### Round 1 — Synthetic input

Random 640×640 uint8 frames.

| Variant | p50 | mean ± std |
|---|---:|---:|
| YOLO26s | 5.779 ms | 5.792 ± 0.065 ms |
| YOLO11s | 5.585 ms | 5.604 ± 0.058 ms |

Standard deviation ~1% of the mean — stable measurements, not noise.

**Apparent conclusion:** NMS-free provides no benefit on Apple Silicon.

**Wrong.** Random pixels produce almost no confident detections, so post-processing was never exercised.

### Round 2 — Real image

| Variant | Detections | p50 | Δ vs synthetic |
|---|---:|---:|---:|
| YOLO26s | 6 | 6.126 ms | +0.347 ms |
| YOLO11s | 5 | 6.484 ms | +0.899 ms |

The ranking inverted. YOLO11s slowed 2.6× more, and its standard deviation rose from 0.058 to 0.082 ms while YOLO26s stayed flat.

**Apparent conclusion:** NMS is the cause; cost should scale with detection count.

### Round 3 — Forcing more detections

Confidence threshold lowered to 0.01.

| Variant | Detections | p50 | Δ vs conf 0.25 |
|---|---:|---:|---:|
| YOLO26s | 19 | 6.127 ms | +0.001 ms |
| YOLO11s | 25 | 6.507 ms | +0.023 ms |

YOLO26s moved by one microsecond between 6 and 19 detections — exactly what NMS-free predicts. But YOLO11s moved only 0.023 ms between 5 and 25.

**Neither model's cost scales with detection count.** The scaling hypothesis failed.

### Round 4 — The resolution

Running the same comparison under ONNX Runtime:

| Runtime | Synthetic | Real | Δ |
|---|---:|---:|---:|
| PyTorch MPS | 5.779 ms | 6.126 ms | **+0.347 ms** |
| ONNX CoreML | 10.319 ms | 10.335 ms | **+0.016 ms** |

Identical model, identical input, identical hardware. The real-image penalty is 0.347 ms in PyTorch and 0.016 ms under ONNX — inside the noise floor.

**The effect is Ultralytics runtime overhead, not a model property.** An exported graph runs the same operations regardless of input content. The Python inference path does not: it constructs result objects, moves detections off-GPU, and formats output, and that work scales with what was found.

A framework overhead was nearly published as a model characteristic.

### Device validation

| Device | p50 | FPS | Speedup |
|---|---:|---:|---:|
| CPU | 33.295 ms | 30.0 | 1.0× |
| MPS | 5.779 ms | 173.0 | **5.76×** |

Run to confirm MPS was engaged rather than silently falling back. CPU-only inference lands at 30.03 FPS — precisely at the real-time video threshold, with no headroom for decoding, tracking, or rendering.

---

## What this means in practice

1. **Benchmark on data resembling deployment.** Synthetic tensors are convenient and reversed a model ranking here.
2. **Detection count is not the variable to control.** It had almost no effect across a 5× range. Input realism did.
3. **Measure on the runtime you will ship.** Latency ordering is not preserved across PyTorch, ONNX CPU, and CoreML.
4. **Check partition counts, not node coverage.** 93.5% CoreML support produced worse performance than 98.1%, because scatter matters more than percentage.
5. **Report distributions.** p95 determines whether a real-time pipeline stutters; the mean does not.

---

## Method

**GPU synchronization.** GPU calls are asynchronous — control returns to Python before computation completes. Without an explicit barrier (`torch.mps.synchronize()` / `torch.cuda.synchronize()`), a timer measures dispatch latency rather than execution and reports numbers several times too low. Every timed call is bracketed.

**Warmup.** First 50 runs discarded. Initial calls include weight transfer, kernel compilation, and cache population — one-time costs, not steady state.

**Distribution over mean.** 300 iterations reported as p50, p95, min, max, std. A 52 ms mean is compatible with a stable 51–53 ms model and with a 45 ms model that spikes to 200 ms. Only one ships.

**Fixed input.** Same frame for every variant within a condition, so differences are attributable to model or runtime rather than scene content.

**Batch size 1.** Batching amortizes overhead and flatters throughput. A camera delivers one frame at a time.

**Provenance.** Every result JSON embeds the configuration and full environment that produced it. A latency number without its settings is an anecdote.

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
results/*.json             one per variant, config and environment embedded
figures/                   charts generated from results
```

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# PyTorch
python src/benchmark.py --weights weights/yolo26s.pt --variant yolo26s_pytorch_mps
python src/benchmark.py --weights weights/yolo26s.pt --variant yolo26s_real --image data/samples/dense.jpg

# Export and ONNX
python -c "from ultralytics import YOLO; YOLO('weights/yolo26s.pt').export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)"
python src/benchmark_onnx.py --model weights/yolo26s.onnx --variant yolo26s_onnx_coreml \
  --image data/samples/dense.jpg --providers CoreMLExecutionProvider,CPUExecutionProvider

# Operator inspection
python -c "
import onnx; from collections import Counter
m = onnx.load('weights/yolo26s.onnx')
print(Counter(n.op_type for n in m.graph.node))"
```

Weights download on first use. Changing any value in `configs/benchmark.yaml` invalidates comparison with existing results.

---

## In progress

- **INT8 quantization** — the first optimization that can cost accuracy; mAP measured against COCO val
- **TensorRT on NVIDIA** — does the partition problem exist outside Apple silicon?
- **Fine-tuning on VisDrone** — domain-specific weights, dense small-object scenes
- **Video pipeline** — BoT-SORT and ByteTrack, end-to-end FPS by stage rather than per-frame inference

## Limitations

- Single hardware platform. Apple Silicon partition behaviour may not generalize; the TensorRT comparison addresses this.
- Detection counts tested span 0 to 25. VisDrone frames carry dozens to hundreds; scaling behaviour may differ at that density.
- Pretrained COCO weights, not domain-specific.
- Accuracy not yet measured. Every result so far is latency only — the FP32 variants are numerically equivalent, but that changes with INT8.
- The 8-vs-5 partition count is reported by ONNX Runtime; the specific nodes CoreML rejected were inferred from operator support patterns rather than read from the runtime directly.

---

**Abdul Rafay Mohd** · M.S. Artificial Intelligence, University of North Texas
[github.com/Mohd-Abdul-Rafay](https://github.com/Mohd-Abdul-Rafay) · [linkedin.com/in/mohd-abdul-rafay](https://linkedin.com/in/mohd-abdul-rafay)
