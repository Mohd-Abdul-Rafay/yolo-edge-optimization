"""
Regenerate the write-up figures from the measured results/*.json files.

    python scripts/make_figures.py

    figures/int8_two_backends.{png,svg}
    figures/tensorrt_precision_two_gpus.{png,svg}

Nothing is typed in: every latency, size and ratio is read from the JSON
the benchmark wrote. Styling matches the portfolio (abdulrafaymohd.com):
cream paper, Inter Tight labels, JetBrains Mono numbers, and the four
semantic colours used only for what they mean --

    blue = result held    rust = regression
    amber = unresolved    sage = supporting / reference

Text is always ink or muted, never a semantic colour.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

# ---------------------------------------------------------------- style

PAPER = "#F9F6EE"
INK = "#0F172A"
MUTED = "#64706B"
FAINT = "#9AA29E"
RULE = "#D6D5D1"
BLUE, RUST, AMBER, SAGE = "#2563EB", "#D3542A", "#F4C430", "#6B8F73"

MISSING = []


def pick_font(want, fallback):
    if want in {f.name for f in font_manager.fontManager.ttflist}:
        return want
    MISSING.append(want)
    return fallback


SANS = pick_font("Inter Tight", "DejaVu Sans")
MONO = pick_font("JetBrains Mono", "DejaVu Sans Mono")

plt.rcParams.update({
    "font.family": SANS,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "yolo-edge-optimization",
})

W = 8.0
DPI = 240


def warn_missing_fonts():
    if MISSING:
        bar = "!" * 72
        print(f"{bar}\n  WARNING: FIGURE FONTS MISSING: {', '.join(MISSING)}\n"
              f"  Output will differ from the committed figures.\n{bar}", file=sys.stderr)


def load(variant):
    return json.loads((RESULTS / f"{variant}.json").read_text())


def p50(variant):
    return load(variant)["latency"]["p50_ms"]


def ratio_text(r):
    """1.38x, 2.52x -- but 11.6x: two significant decimals reads as false precision."""
    return f"{r:.1f}×" if r >= 10 else f"{r:.2f}×"


def gpu_name(env, fallback):
    return env.get("gpu") or env.get("cuda_device") or env.get("device_name") or fallback


def header(fig, title, subtitle):
    fig.text(0.045, 0.955, title, color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.045, 0.885, subtitle, color=MUTED, fontsize=9.5, va="top")


def footer(fig, text):
    fig.text(0.045, 0.03, text, color=FAINT, fontsize=8, va="bottom")


def legend(fig, y, items):
    x = 0.045
    for colour, label in items:
        fig.patches.append(Rectangle((x, y), 0.016, 0.028, transform=fig.transFigure,
                                     figure=fig, facecolor=colour, edgecolor="none"))
        t = fig.text(x + 0.024, y + 0.014, label, color=MUTED, fontsize=9, va="center")
        fig.canvas.draw()
        x = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.03


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    fig.savefig(OUT / f"{name}.svg", format="svg", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote figures/{name}.png + .svg")


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(length=0, labelsize=8.5)
    for label in ax.get_yticklabels():
        label.set_family(MONO)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------- figure 1: two backends

def int8_two_backends():
    backends = [
        ("ONNX Runtime\nCPU", "yolo26s_onnx_cpu", "yolo26s_int8_cpu"),
        ("ONNX Runtime\nCoreML (Neural Engine)", "yolo26s_onnx_coreml", "yolo26s_int8_coreml"),
    ]
    fp32_mb = load("yolo26s_onnx_cpu")["size_mb"]
    int8_mb = load("yolo26s_int8_cpu")["size_mb"]

    fig = plt.figure(figsize=(W, 5.2), dpi=DPI)
    header(fig, "The same quantized file is faster on one backend\nand unusable on the other", "")
    fig.texts[-1].set_text("YOLO26s  ·  INT8 static quantization  ·  Apple M4 Max  ·  "
                           "batch 1, 640×640  ·  p50 latency, log scale")
    fig.texts[-1].set_y(0.835)
    legend(fig, 0.745, [(SAGE, f"FP32  ·  {fp32_mb:.1f} MB  (reference)"),
                        (BLUE, f"INT8  ·  {int8_mb:.1f} MB, faster"),
                        (RUST, "INT8, slower")])

    ax = fig.add_axes([0.1, 0.15, 0.86, 0.56])
    width = 0.34
    for i, (label, fp32_v, int8_v) in enumerate(backends):
        fp32, int8 = p50(fp32_v), p50(int8_v)
        faster = int8 < fp32
        int8_colour = BLUE if faster else RUST
        for x, value, colour in [(i - width / 2, fp32, SAGE), (i + width / 2, int8, int8_colour)]:
            ax.bar(x, value, width * 0.92, color=colour)
            ax.text(x, value * 1.07, f"{value:.1f} ms", ha="center", va="bottom",
                    fontsize=9, family=MONO, color=INK)
        ratio = fp32 / int8 if faster else int8 / fp32
        ax.text(i, max(fp32, int8) * 1.9, f"{ratio_text(ratio)} {'faster' if faster else 'slower'}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)

    ax.set_yscale("log")
    ax.set_ylim(3, 400)
    ax.minorticks_off()
    ax.set_yticks([5, 10, 25, 50, 100, 200])
    ax.set_yticklabels([str(t) for t in [5, 10, 25, 50, 100, 200]])
    ax.set_xticks(range(len(backends)))
    ax.set_xticklabels([b[0] for b in backends], fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, len(backends) - 0.4)
    style_axis(ax, "p50 latency (ms)")

    footer(fig, "Generated by scripts/make_figures.py from results/yolo26s_{onnx,int8}_{cpu,coreml}.json")
    save(fig, "int8_two_backends")


# --------------------------------------------- figure 2: two NVIDIA cards

def tensorrt_precision_two_gpus():
    precisions = ["fp32", "fp16", "int8"]
    gpus = [
        ("A100", "yolo26s_tensorrt_{}", SAGE),        # reference card
        ("T4", "yolo26s_t4_tensorrt_{}", BLUE),       # where the result holds
    ]
    lat = {g: [p50(pattern.format(p)) for p in precisions] for g, pattern, _ in gpus}
    name = {g: gpu_name(load(pattern.format("fp32"))["environment"], g) for g, pattern, _ in gpus}

    fig = plt.figure(figsize=(W, 5.0), dpi=DPI)
    header(fig, "Reduced precision is worth twice as much on the weaker GPU",
           "YOLO26s  ·  TensorRT  ·  batch 1, 640×640  ·  identical engines, two NVIDIA cards")
    legend(fig, 0.79, [(colour, f"{g}  ·  {name[g]}") for g, _, colour in gpus])

    # left: latency
    ax = fig.add_axes([0.08, 0.14, 0.42, 0.58])
    width = 0.36
    for k, (g, _, colour) in enumerate(gpus):
        for i, value in enumerate(lat[g]):
            x = i + (k - 0.5) * width
            ax.bar(x, value, width * 0.92, color=colour)
            ax.text(x, value + 0.15, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=8, family=MONO, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p.upper() for p in precisions], fontsize=9.5, color=INK)
    ax.set_ylim(0, max(max(v) for v in lat.values()) * 1.18)
    style_axis(ax, "p50 latency (ms)")
    ax.set_title("Latency", loc="left", fontsize=10, color=INK, pad=10)

    # right: speedup over each card's own FP32
    ax2 = fig.add_axes([0.6, 0.14, 0.36, 0.58])
    for g, _, colour in gpus:
        speedup = [lat[g][0] / v for v in lat[g]]
        ax2.plot(range(3), speedup, color=colour, linewidth=2.2, marker="o", markersize=6)
        for i, s in enumerate(speedup):
            if i == 0:
                continue  # 1.00x by definition; labelled once below
            ax2.text(i, s + 0.08, ratio_text(s), ha="center", va="bottom",
                     fontsize=8.5, family=MONO, color=INK)
        ax2.text(2.12, speedup[-1], g, va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax2.text(0, 0.9, "1.00×", ha="center", va="top", fontsize=8.5, family=MONO, color=MUTED)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([p.upper() for p in precisions], fontsize=9.5, color=INK)
    ax2.set_xlim(-0.25, 2.45)
    ax2.set_ylim(0.8, 2.9)
    style_axis(ax2, "speedup over own FP32")
    ax2.set_title("Speedup", loc="left", fontsize=10, color=INK, pad=10)

    footer(fig, "Generated by scripts/make_figures.py from results/yolo26s{,_t4}_tensorrt_{fp32,fp16,int8}.json")
    save(fig, "tensorrt_precision_two_gpus")


if __name__ == "__main__":
    warn_missing_fonts()
    int8_two_backends()
    tensorrt_precision_two_gpus()
    warn_missing_fonts()
