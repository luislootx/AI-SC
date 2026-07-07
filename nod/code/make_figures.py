"""Generate all figures for the workshop deck.

Outputs PNG files into slides/figures/. Run once after the pipeline finishes:
    python make_figures.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Wedge, ConnectionPatch, RegularPolygon, PathPatch)
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "slides", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# TAMU + duotone-template-derived palette
TAMU_MAROON = "#500000"
DUO = {
    "maroon": "#500000",
    "deep_blue": "#1F497D",
    "blue": "#4F81BD",
    "red": "#C0504D",
    "green": "#9BBB59",
    "purple": "#8064A2",
    "cyan": "#4BACC6",
    "orange": "#F79646",
    "cream": "#EEECE1",
    "ink": "#1A1A1A",
    "ink_soft": "#3F3F3F",
}

PARADIGM_COLOR = {
    "fno": DUO["blue"],
    "deeponet": DUO["red"],
    "transformer": DUO["purple"],
    "wavelet": DUO["green"],
    "hybrid_fno_attn": DUO["orange"],
    "random": DUO["cyan"],
    "fno_spawn": DUO["blue"],
    "deeponet_spawn": DUO["red"],
    "transformer_spawn": DUO["purple"],
    "wavelet_spawn": DUO["green"],
}
BLOCK_COLOR = {
    "fourier":       DUO["blue"],
    "attention":     DUO["purple"],
    "branch_trunk":  DUO["red"],
    "wavelet":       DUO["green"],
    "residual_conv": DUO["orange"],
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": DUO["ink_soft"],
    "axes.labelcolor": DUO["ink"],
    "xtick.color": DUO["ink_soft"],
    "ytick.color": DUO["ink_soft"],
    "axes.titleweight": "bold",
    "axes.titlecolor": TAMU_MAROON,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})


def load(name: str):
    path = os.path.join(RESULTS, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Figure 1 — THE OVERVIEW DIAGRAM (the "chingón" one)
# ---------------------------------------------------------------------------

def fig_overview():
    """Radial scientific-community diagram. 8 labs orbiting a global registry,
    each lab a 3-sector node (planning / worker / evaluation), with peer-review
    arcs and PSO velocity arrows. LLM-driven sectors marked with a brain glyph.
    """
    demo = load("results_agentic-demo.json")
    last_iter = demo["iteration_history"][-1]
    labs = last_iter["lab_summaries"]

    fig, ax = plt.subplots(figsize=(14.0, 9.0))
    ax.set_xlim(-7.8, 7.8)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(labs)
    R = 2.75         # satellite ring radius
    r_lab = 0.55     # lab disk radius
    r_hub = 0.95
    cx, cy = 0.0, 0.0

    # 1. Background — soft radial wash
    for i, alpha in enumerate(np.linspace(0.0, 0.20, 60)):
        c = Circle((cx, cy), 7.5 - i * 0.1, color=DUO["cream"], alpha=alpha,
                   linewidth=0, zorder=0)
        ax.add_patch(c)

    # 2. Peer-review arcs (curved cross-talk between non-adjacent pairs)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    pts = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in angles]
    for i in range(n):
        for j in range(i + 1, n):
            if j - i not in (1, 2):
                continue
            (x1, y1), (x2, y2) = pts[i], pts[j]
            arc = FancyArrowPatch((x1, y1), (x2, y2),
                                  connectionstyle=f"arc3,rad={'0.30' if j-i==1 else '0.50'}",
                                  arrowstyle="-",
                                  color=DUO["maroon"], alpha=0.16, lw=1.2,
                                  zorder=1)
            ax.add_patch(arc)

    # 3. PSO influence rays — global best -> each lab
    for (px, py), lab in zip(pts, labs):
        dx, dy = px - cx, py - cy
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        x0, y0 = cx + r_hub * ux, cy + r_hub * uy
        x1, y1 = px - (r_lab + 0.06) * ux, py - (r_lab + 0.06) * uy
        ray = FancyArrowPatch((x0, y0), (x1, y1),
                              arrowstyle="->,head_length=9,head_width=6",
                              color=DUO["deep_blue"],
                              alpha=0.45 + 0.45 * lab.get("trust", 0.5),
                              lw=1.8, zorder=2)
        ax.add_patch(ray)

    # 4. Lab nodes
    role_colors = {"P": DUO["orange"], "W": DUO["cyan"], "E": DUO["green"]}
    for (px, py), lab in zip(pts, labs):
        paradigm = lab["paradigm"]
        ring_color = PARADIGM_COLOR.get(paradigm, DUO["ink_soft"])
        # Outer paradigm ring
        ax.add_patch(Circle((px, py), r_lab + 0.10, color=ring_color,
                            alpha=0.95, zorder=3))
        # Inner white disk
        ax.add_patch(Circle((px, py), r_lab, color="white", zorder=4))
        # 3 wedges (planning / worker / evaluation)
        sectors = [(60, 180, "P"), (300, 60, "W"), (180, 300, "E")]
        for t1, t2, role in sectors:
            ax.add_patch(Wedge((px, py), r_lab - 0.04, t1, t2,
                               width=0.22, facecolor=role_colors[role],
                               edgecolor="white", linewidth=1.5,
                               alpha=0.92, zorder=5))
        for theta_deg, role in [(120, "P"), (240, "E"), (0, "W")]:
            tx = px + 0.37 * np.cos(np.deg2rad(theta_deg))
            ty = py + 0.37 * np.sin(np.deg2rad(theta_deg))
            ax.text(tx, ty, role, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=6)
        # Lab id
        ax.text(px, py, f"L{lab['lab_id']}", ha="center", va="center",
                fontsize=15, color=TAMU_MAROON, fontweight="bold", zorder=7)
        # Paradigm label & block strip — placed by quadrant for readability
        idx = lab["lab_id"]
        # (label_dx, label_dy, strip_dy)
        layout = {
            0: (0, 1.00, None),     # TOP — label above, no strip (would clip title)
            1: (1.00, 0.55, -0.55), # upper right
            2: (1.10, 0,    -0.55), # right
            3: (1.00, -0.55, 0.55), # lower right (strip above)
            4: (0, -1.00, None),    # BOTTOM — label below, no strip (would clip legend)
            5: (-1.00, -0.55, 0.55),
            6: (-1.10, 0,    -0.55),
            7: (-1.00, 0.55, -0.55),
        }
        ldx, ldy, sdy = layout.get(idx, (0, 1.0, -0.55))
        clean = paradigm.replace("_spawn", "*").replace("hybrid_fno_attn", "FNO+ATTN")
        ax.text(px + ldx, py + ldy, clean.upper(),
                ha="center", va="center",
                fontsize=10, color=DUO["ink"], fontweight="bold", zorder=8)
        if sdy is not None:
            blocks = lab.get("blocks", [])[:4]
            strip_w = 0.30
            bx0 = px - (len(blocks) - 1) * strip_w / 2
            for k, b in enumerate(blocks):
                bx = bx0 + k * strip_w
                by = py + sdy
                ax.add_patch(FancyBboxPatch((bx - 0.13, by - 0.10), 0.26, 0.20,
                                            boxstyle="round,pad=0.02,rounding_size=0.05",
                                            facecolor=BLOCK_COLOR.get(b, "#888"),
                                            edgecolor="white", linewidth=1.0,
                                            alpha=0.95, zorder=6))
                ax.text(bx, by, b[:3].upper(), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=7)

    # 5. Central hub
    hub = RegularPolygon((cx, cy), numVertices=6, radius=r_hub,
                         orientation=np.pi / 6,
                         facecolor=TAMU_MAROON, edgecolor="white",
                         linewidth=2.8, alpha=0.97, zorder=9)
    ax.add_patch(hub)
    ax.text(cx, cy + 0.22, "GLOBAL", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", zorder=10)
    ax.text(cx, cy + 0.02, "BEST", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", zorder=10)
    ax.text(cx, cy - 0.30, "REGISTRY", ha="center", va="center",
            fontsize=9, color=DUO["cream"], zorder=10)

    # 6. Title (top, comfortably above ring)
    ax.text(0, 4.55, "AI SCIENTIFIC COMMUNITY",
            ha="center", va="center", fontsize=24,
            color=TAMU_MAROON, fontweight="bold")
    ax.text(0, 4.15, "8 virtual labs · planning · worker · evaluation · Gemma 3 12B + PyTorch",
            ha="center", va="center", fontsize=11.5, color=DUO["ink_soft"],
            style="italic")

    # 7. Legend bar (bottom)
    band = FancyBboxPatch((-7.2, -4.85), 14.4, 0.95,
                          boxstyle="round,pad=0.04,rounding_size=0.10",
                          facecolor="white", edgecolor=DUO["ink_soft"],
                          linewidth=0.8, alpha=0.95, zorder=10)
    ax.add_patch(band)
    legend_y = -4.40
    legend_items = [
        ("Planning agent (LLM)",  DUO["orange"]),
        ("Worker (PyTorch GPU)",  DUO["cyan"]),
        ("Evaluator (LLM)",       DUO["green"]),
        ("Peer-review citation",  DUO["maroon"]),
        ("PSO pull",              DUO["deep_blue"]),
    ]
    xs = np.linspace(-6.4, 6.0, 5)
    for x, (label, color) in zip(xs, legend_items):
        ax.add_patch(Circle((x, legend_y), 0.16, color=color, zorder=11))
        ax.text(x + 0.32, legend_y, label, ha="left", va="center",
                fontsize=10.5, color=DUO["ink"], zorder=11)

    save(fig, "01_swarm_overview.png")


# ---------------------------------------------------------------------------
# Figure 2 — Block-usage evolution
# ---------------------------------------------------------------------------

def fig_block_evolution():
    demo = load("results_agentic-demo.json")
    iters = demo["iteration_history"]
    block_pct: Dict[str, List[float]] = {b: [] for b in BLOCK_COLOR}
    for snap in iters:
        counts = defaultdict(int)
        total = 0
        for lab in snap["lab_summaries"]:
            for b in lab["blocks"]:
                counts[b] += 1
                total += 1
        for b in BLOCK_COLOR:
            block_pct[b].append(100 * counts.get(b, 0) / max(total, 1))

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(1, len(iters) + 1)
    order = ["fourier", "attention", "wavelet", "branch_trunk", "residual_conv"]
    ax.stackplot(x,
                 *[block_pct[b] for b in order],
                 labels=[b.replace("_", " ") for b in order],
                 colors=[BLOCK_COLOR[b] for b in order],
                 alpha=0.92)
    ax.set_xlabel("Swarm iteration", fontsize=12, color=DUO["ink"])
    ax.set_ylabel("Block usage (%)", fontsize=12, color=DUO["ink"])
    ax.set_title("Block usage evolution — DeepONet (branch-trunk) is abandoned;\n"
                 "Fourier + Wavelet emerge as essential primitives",
                 fontsize=14)
    ax.set_xlim(1, len(iters))
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.legend(loc="lower right", framealpha=0.95, ncol=5, fontsize=10,
              bbox_to_anchor=(1.0, -0.22))
    for b in ("fourier", "wavelet", "branch_trunk"):
        first = block_pct[b][0]
        last = block_pct[b][-1]
        delta = last - first
        ax.annotate(f"{b}: {first:.0f}% → {last:.0f}%  ({delta:+.0f}%)",
                    xy=(len(iters), sum_until(block_pct, order, b, -1)),
                    xytext=(len(iters) + 0.25, sum_until(block_pct, order, b, -1)),
                    fontsize=9, color=DUO["ink_soft"])
    save(fig, "02_block_evolution.png")


def sum_until(d, order, target, idx):
    s = 0
    for b in order:
        s += d[b][idx]
        if b == target:
            return s - d[b][idx] / 2
    return s


# ---------------------------------------------------------------------------
# Figure 3 — Validation comparison (rel L2 vs params, log-log)
# ---------------------------------------------------------------------------

def fig_validation():
    val = load("validation_baselines.json")
    rows = sorted(val["results"], key=lambda r: r["params"])
    names = [r["name"].split(" (")[0] for r in rows]
    params = np.array([r["params"] for r in rows])
    l2 = np.array([r["rel_l2_clean"] for r in rows])
    l2_n = np.array([r["rel_l2_noisy"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.3, 5.2),
                                   gridspec_kw={"width_ratios": [1.2, 1.0]})

    # LEFT — bar plot: clean vs noisy rel L2 (linear)
    y = np.arange(len(names))
    bw = 0.36
    bars_clean = ax1.barh(y - bw / 2, l2, height=bw, color=DUO["deep_blue"],
                          alpha=0.92, label="Clean rel L²")
    bars_noisy = ax1.barh(y + bw / 2, l2_n, height=bw, color=DUO["orange"],
                          alpha=0.92, label="Noisy rel L²")
    for bars, vals in [(bars_clean, l2), (bars_noisy, l2_n)]:
        for b, v in zip(bars, vals):
            ax1.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=9, color=DUO["ink"])
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Relative L² error  (lower is better)", fontsize=11)
    ax1.set_title("Honest validation — same data, same epochs", fontsize=13)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # RIGHT — Pareto: params vs L2 (log-log) with manual offsets to avoid overlap
    colors = [DUO["blue"] if "FNO" in n else
              DUO["red"] if "DeepONet" in n else
              DUO["purple"] if "Transformer" in n else
              DUO["green"] if "Smoke" in n else
              DUO["orange"] for n in names]
    # Manual offsets per arch (in points, dx, dy, ha)
    offsets = {
        "Pure FNO":         (0,  -16, "center"),
        "Pure DeepONet":    (15, 8,   "left"),
        "Pure Transformer": (-15, 8,  "right"),
        "Smoke Hybrid":     (0,  10,  "center"),
        "Demo Hybrid":      (0,  -16, "center"),
    }
    for x, y_, n, c in zip(params, l2, names, colors):
        ax2.scatter(x, y_, s=180, color=c, edgecolor="white", linewidth=1.8,
                    zorder=3)
        dx, dy, ha = offsets.get(n, (0, 8, "center"))
        ax2.annotate(n, (x, y_), xytext=(dx, dy),
                     textcoords="offset points",
                     ha=ha, fontsize=10, color=DUO["ink"], fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Parameters", fontsize=11)
    ax2.set_ylabel("Relative L² (clean)", fontsize=11)
    ax2.set_title("Pareto front: accuracy vs cost", fontsize=13)
    ax2.grid(True, which="both", alpha=0.3, linestyle=":")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # Highlight the demo hybrid with a halo
    for x, y_, n in zip(params, l2, names):
        if "Demo" in n:
            from matplotlib.patches import Ellipse
            halo = Ellipse((x, y_), x * 0.65, y_ * 1.6,
                           fill=False, edgecolor=DUO["orange"],
                           linewidth=2.5, alpha=0.85, zorder=2)
            ax2.add_patch(halo)
    fig.suptitle("Discovered hybrid: 5× fewer params, 1.11× FNO noisy error",
                 fontsize=14, fontweight="bold", color=TAMU_MAROON, y=1.03)
    save(fig, "03_validation_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4 — Discovered architecture diagram
# ---------------------------------------------------------------------------

def fig_arch_diagram():
    """Visualize the discovered hybrid: input -> fourier -> attention ->
    wavelet -> attention -> output, with annotations describing each block."""
    blocks = [
        ("input",     "1ch  vorticity\n(32×32)",  "white",          DUO["ink"]),
        ("fourier",   "Spectral conv\nFFT modes=10", BLOCK_COLOR["fourier"], "white"),
        ("attention", "Self-attn\nnonlocal coupling", BLOCK_COLOR["attention"], "white"),
        ("wavelet",   "Multi-scale\nlow+high pass", BLOCK_COLOR["wavelet"], "white"),
        ("attention", "Self-attn\nrefinement",     BLOCK_COLOR["attention"], "white"),
        ("output",    "1ch  vorticity\nat t+0.5",  "white",          DUO["ink"]),
    ]
    fig, ax = plt.subplots(figsize=(13.3, 5.0))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(-0.5, 5.6)
    ax.axis("off")
    n = len(blocks)
    x_step = 13.3 / n
    cy = 2.0
    for i, (name, desc, fc, tc) in enumerate(blocks):
        cx = (i + 0.5) * x_step
        if name in ("input", "output"):
            box = mpatches.Rectangle((cx - 0.85, cy - 0.55), 1.7, 1.1,
                                     facecolor=fc, edgecolor=DUO["ink"],
                                     linewidth=2.0, zorder=3)
        else:
            box = mpatches.Rectangle((cx - 0.85, cy - 0.65), 1.7, 1.3,
                                     facecolor=fc, edgecolor="white",
                                     linewidth=2.5, alpha=0.96, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy + 0.18, name.upper(), ha="center", va="center",
                fontsize=13, fontweight="bold", color=tc, zorder=4)
        ax.text(cx, cy - 0.30, desc, ha="center", va="center",
                fontsize=8.5, color=tc, zorder=4)
        if i < n - 1:
            ax.annotate("", xy=((i + 1) * x_step + 0.05, cy),
                        xytext=((i + 0.5) * x_step + 0.85, cy),
                        arrowprops=dict(arrowstyle="->,head_length=10,head_width=6",
                                        color=DUO["ink_soft"], lw=2.0),
                        zorder=2)
    # Skip-connection arc — placed BELOW title, ABOVE blocks
    ax.annotate("", xy=(12.6, 3.5), xytext=(0.7, 3.5),
                arrowprops=dict(arrowstyle="-",
                                color=DUO["maroon"], lw=1.8,
                                connectionstyle="arc3,rad=-0.20",
                                alpha=0.55))
    ax.text(6.65, 4.32, "global skip connection",
            ha="center", fontsize=10, color=DUO["maroon"], style="italic")

    ax.text(6.65, 0.20, "977 K parameters · trained 15 epochs · 256 NS samples",
            ha="center", va="center", fontsize=11, color=DUO["ink_soft"])
    ax.text(6.65, 5.20, "DISCOVERED HYBRID — fitness 0.9822 / rel L² 0.0022",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TAMU_MAROON)
    save(fig, "04_discovered_architecture.png")


# ---------------------------------------------------------------------------
# Figure 5 — Per-lab fitness evolution
# ---------------------------------------------------------------------------

def fig_fitness_evolution():
    demo = load("results_agentic-demo.json")
    iters = demo["iteration_history"]
    n_iters = len(iters)
    # Collect per-lab composite over iterations
    per_lab: Dict[int, List[float]] = defaultdict(lambda: [None] * n_iters)
    paradigms: Dict[int, str] = {}
    for snap in iters:
        for lab in snap["lab_summaries"]:
            per_lab[lab["lab_id"]][snap["iteration"]] = lab["composite"]
            paradigms[lab["lab_id"]] = lab["paradigm"]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(1, n_iters + 1)
    for lid, vals in sorted(per_lab.items()):
        clean = [v if v is not None else np.nan for v in vals]
        c = PARADIGM_COLOR.get(paradigms[lid], DUO["ink_soft"])
        ax.plot(x, clean, marker="o", lw=2.0, color=c, alpha=0.85,
                label=f"Lab {lid} ({paradigms[lid]})")

    # Global best line
    gb = [s["global_best_fitness"] for s in iters]
    ax.plot(x, gb, color=TAMU_MAROON, lw=3.5, linestyle="--",
            marker="*", markersize=13, label="Global best",
            zorder=10)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Composite fitness", fontsize=12)
    ax.set_title("Per-lab composite fitness — Lab 7 hits ceiling at iter 1\n"
                 "and survives 8 iterations of community challenge",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "05_fitness_evolution.png")


# ---------------------------------------------------------------------------
# Figure 6 — Pipeline detail: PSO vs Agentic
# ---------------------------------------------------------------------------

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.3, 6.0))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    title = "Pipeline — what runs where"
    ax.text(6.65, 5.65, title, ha="center", fontsize=18,
            fontweight="bold", color=TAMU_MAROON)

    rows = [
        ("PLANNING agent",   "Gemma 3 12B (Ollama, local)",
         "JSON: {action, rationale, new_genome}",
         DUO["orange"]),
        ("WORKER agent",     "PyTorch (CUDA, RTX 4080)",
         "ConfigurableNeuralOperator + AdamW + 15 epochs on NS data",
         DUO["cyan"]),
        ("EVALUATION agent", "Gemma 3 12B (Ollama, local)",
         "JSON: {scores: {lab_id: weight, …}, rationale}",
         DUO["green"]),
    ]
    for i, (role, runtime, contract, color) in enumerate(rows):
        y = 4.4 - i * 1.45
        # role pill
        pill = FancyBboxPatch((0.5, y - 0.45), 2.5, 0.9,
                              boxstyle="round,pad=0.05,rounding_size=0.18",
                              facecolor=color, edgecolor="white",
                              linewidth=2.0, alpha=0.96)
        ax.add_patch(pill)
        ax.text(1.75, y, role, ha="center", va="center", fontsize=13,
                color="white", fontweight="bold")
        # runtime
        ax.text(3.4, y + 0.22, runtime, fontsize=12,
                color=DUO["ink"], fontweight="bold")
        # contract
        ax.text(3.4, y - 0.18, contract, fontsize=10,
                color=DUO["ink_soft"], family="monospace")
    # Connecting arrows on the left
    for i in range(2):
        y0 = 4.4 - i * 1.45 - 0.45
        y1 = 4.4 - (i + 1) * 1.45 + 0.45
        ax.annotate("", xy=(1.75, y1), xytext=(1.75, y0),
                    arrowprops=dict(arrowstyle="->,head_length=8,head_width=5",
                                    color=DUO["ink_soft"], lw=1.4))

    # Feedback loop label on right
    box = FancyBboxPatch((9.6, 1.0), 3.5, 4.0,
                         boxstyle="round,pad=0.05,rounding_size=0.18",
                         facecolor=DUO["cream"], edgecolor=TAMU_MAROON,
                         linewidth=2.0, alpha=0.65)
    ax.add_patch(box)
    ax.text(11.35, 4.6, "PSO velocity update", ha="center",
            fontsize=12, color=TAMU_MAROON, fontweight="bold")
    ax.text(11.35, 3.7, "v ← w·v + c1·r1·(pb − x)\n        + c2·r2·(gb − x)",
            ha="center", fontsize=11, color=DUO["ink"], family="monospace")
    ax.text(11.35, 2.7, "+ block-sequence\ncrossover from\nGlobal Best",
            ha="center", fontsize=10, color=DUO["ink_soft"])
    ax.text(11.35, 1.5, "(used as fallback\nwhen LLM fails)",
            ha="center", fontsize=9, color=DUO["ink_soft"], style="italic")
    save(fig, "06_pipeline.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(f"Generating figures in: {FIG_DIR}")
    fig_overview()
    fig_block_evolution()
    fig_validation()
    fig_arch_diagram()
    fig_fitness_evolution()
    fig_pipeline()
    print("Done.")


if __name__ == "__main__":
    main()
