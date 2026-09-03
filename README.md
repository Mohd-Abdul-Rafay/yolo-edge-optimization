# YOLO Edge Optimization

Taking a detection model from research checkpoint to real-time deployment, and measuring what every optimization actually costs.

The question this repo answers is not "how fast is YOLO." It is **whether the latency numbers people publish mean anything**, and what changes when you export, quantize, and deploy the same model across different runtimes and hardware.

**Status:** Phase 1 complete (baseline measurement). Phases 2–5 in progress.

---

## Headline finding so far

**Synthetic-input latency benchmarks do not predict real-input latency, and the direction of the error is architecture-dependent.**

Benchmarking YOLO26s against YOLO11s on Apple M4 Max (MPS), the ranking inverts depending on what you feed the model:

| Input condition | YOLO26s | YOLO11s | Faster model |
|---|---:|---:|---|
| Synthetic noise, conf 0.25 | 5.779 ms | 5.585 ms | YOLO11s by 3.5% |
| Real image, conf 0.25 | 6.126 ms | 6.484 ms | **YOLO26s by 5.5%** |

A 9-point swing, driven entirely by input content. Anyone benchmarking on synthetic tensors — a common shortcut, since it needs no dataset — would report the opposite conclusion from someone benchmarking on photographs.

---

## The investigation

This took five measurement rounds, and each one overturned the previous interpretation. The sequence is documented here because the wrong turns are the useful part.

### Round 1 — Baseline on synthetic input

Random 640×640 uint8 frames, batch 1, 50 warmup runs discarded, 300 timed iterations.

| Variant | p50 | p95 | mean ± std | Size |
|---|---:|---:|---:|---:|
| YOLO26s PyTorch MPS | 5.779 ms | 5.913 ms | 5.792 ± 0.065 ms | 19.48 MB |
| YOLO11s PyTorch MPS | 5.585 ms | 5.686 ms | 5.604 ± 0.058 ms | 18.42 MB |

Standard deviation is ~1% of the mean, so these are stable measurements, not noise.

**Apparent conclusion:** YOLO26's NMS-free architecture provides no latency benefit on Apple Silicon.

**This conclusion was wrong.** Random pixels produce almost no confident detections, so any post-processing stage was never exercised.

### Round 2 — Real image

Same models, same settings, a real photograph instead of noise.

| Variant | Detections | p50 | Δ vs synthetic |
|---|---:|---:|---:|
| YOLO26s | 6 | 6.126 ms | +0.347 ms |
| YOLO11s | 5 | 6.484 ms | +0.899 ms |

Both models slowed, but YOLO11s slowed **2.6× more**. Its standard deviation also rose from 0.058 to 0.082 ms, while YOLO26s stayed flat — the variance signature of input-dependent post-processing.

**Apparent conclusion:** NMS is the cause, and the cost should scale with detection count.

### Round 3 — Forcing more detections

Confidence threshold lowered to 0.01 on the same real image, pushing detection counts up.

| Variant | Detections | p50 | Δ vs conf 0.25 |
|---|---:|---:|---:|
| YOLO26s | 19 | 6.127 ms | +0.001 ms |
| YOLO11s | 25 | 6.507 ms | +0.023 ms |

YOLO26s moved by **one microsecond** between 6 and 19 detections — exactly what NMS-free predicts.

But YOLO11s moved only 0.023 ms between 5 and 25 detections. **Neither model's cost scales with detection count in this range.** The gap appears when there is real content and then stays constant.

**The scaling hypothesis failed.**

### Round 4 — Synthetic input at low confidence

The decisive test: force many candidate boxes into post-processing on an input with no real structure.

| Variant | Detections | p50 |
|---|---:|---:|
| YOLO26s | 1 | 6.117 ms |
| YOLO11s | 0 | 5.645 ms |

YOLO11s at conf 0.01 on noise produced **zero** detections and stayed at 5.645 ms — unchanged from conf 0.25. NMS never ran, so this round says nothing about NMS.

YOLO26s, on identical synthetic input, slowed by 0.34 ms purely from the threshold change, and emitted 1 detection instead of 0.

### What the data supports

| Condition | YOLO26s | YOLO11s |
|---|---:|---:|
| Synthetic, conf 0.25 | 5.779 ms | 5.585 ms |
| Synthetic, conf 0.01 | 6.117 ms | 5.645 ms |
| Real, conf 0.25 | 6.126 ms | 6.484 ms |
| Real, conf 0.01 | 6.127 ms | 6.507 ms |

Both models exhibit a **fixed step change**, not a per-detection cost. Their triggers differ: YOLO26s switches state when it emits any output at all; YOLO11s switches state on real image content regardless of detection count.

**The NMS explanation is not supported by this data.** The 0.38 ms real-image gap is more plausibly a difference in how each architecture's feature maps behave on structured versus random input — activation sparsity, memory access patterns, cache behaviour — rather than post-processing cost.

This is recorded as a negative result. The intuitive explanation was tested three ways and did not survive.

### Secondary result — device validation

| Device | p50 | FPS | Speedup |
|---|---:|---:|---:|
| CPU | 33.295 ms | 30.0 | 1.0× |
| MPS | 5.779 ms | 173.0 | **5.76×** |

Run as a sanity check that MPS was actually engaged rather than silently falling back. Worth noting that CPU-only inference lands at 30.03 FPS — precisely at the real-time video threshold, with zero headroom left for decoding, tracking, or rendering.

---

## Why this matters

Most published YOLO latency numbers do not state what input produced them. Based on the above, that omission is enough to reverse a model ranking.

Three practical consequences:

1. **Benchmark on data resembling deployment.** Synthetic tensors are convenient and misleading.
2. **Detection count is not the variable to control for.** It had almost no effect across a 5× range. Input realism did.
3. **Report the distribution, not the mean.** p95 determines whether a real-time pipeline stutters; the mean does not.

---

## Method

Every measurement uses the same harness, and every result JSON embeds the configuration and environment that produced it.

**Timing.** GPU calls are asynchronous — control returns to Python before computation finishes. Without an explicit synchronization barrier (`torch.mps.synchronize()` / `torch.cuda.synchronize()`), a timer measures dispatch latency rather than execution, and reports numbers several times too low. Every timed call is bracketed by synchronization.

**Warmup.** The first 50 runs are discarded. Initial calls include weight transfer, kernel compilation, and cache population — one-time costs that are not representative of steady-state inference.

**Distribution over mean.** 300 timed iterations, reported as p50, p95, min, max, and standard deviation. A mean of 52 ms is compatible with a stable 51–53 ms model and with a 45 ms model that occasionally spikes to 200 ms. Only one of those ships.

**Fixed input.** The same frame for every variant within a condition, so observed differences are attributable to the model or runtime rather than to scene content.

**Batch size 1.** Batching amortizes overhead and produces flattering throughput figures, but a camera delivers one frame at a time. Batch 1 is the deployment reality.

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

Exact package versions are pinned in `requirements.txt`. ONNX Runtime fuses operations differently between releases, so latency comparisons across versions are not valid.

---

## Repository layout

```
configs/benchmark.yaml    measurement parameters, shared by all scripts
src/utils.py              timing with device-aware GPU synchronization
src/benchmark.py          latency measurement for PyTorch checkpoints
results/*.json            one file per variant, config and environment embedded
figures/                  charts generated from results
```

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/benchmark.py --weights weights/yolo26s.pt --variant yolo26s_pytorch_mps
python src/benchmark.py --weights weights/yolo26s.pt --variant yolo26s_real --image data/samples/dense.jpg
```

Weights download automatically on first use. Changing any value in `configs/benchmark.yaml` invalidates comparison with existing results.

---

## Planned

- **Phase 2** — Fine-tune YOLO26s on VisDrone (Colab, NVIDIA)
- **Phase 3** — Export matrix: ONNX FP32, ONNX INT8, CoreML, TensorRT; latency and mAP for each
- **Phase 4** — Video pipeline with BoT-SORT and ByteTrack; end-to-end FPS by stage, not per-frame inference
- **Phase 5** — Accuracy-versus-latency tradeoff curve across all variants

## Limitations

- Single hardware platform so far. Apple Silicon results may not generalize to NVIDIA or embedded targets; the Phase 3 TensorRT comparison addresses this.
- Detection counts tested range from 0 to 25. Denser scenes — VisDrone frames carry dozens to hundreds of objects — may reveal scaling behaviour absent here.
- Pretrained COCO weights, not domain-specific. Phase 2 introduces fine-tuned weights.
- The mechanism behind the real-versus-synthetic gap is not established, only that NMS does not explain it.

---

**Abdul Rafay Mohd** · M.S. Artificial Intelligence, University of North Texas
[github.com/Mohd-Abdul-Rafay](https://github.com/Mohd-Abdul-Rafay) · [linkedin.com/in/mohd-abdul-rafay](https://linkedin.com/in/mohd-abdul-rafay)
