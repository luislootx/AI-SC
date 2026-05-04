"""Master results figure: ALL FOUR experiments (pwreg, advec, burgers, NS)
on one canvas, designed as the headline visual on the ICERM poster.

For each PDE we show:
  - PDE name + setup line (epochs, iterations, train samples, resolution)
  - Convergence: composite fitness curve + best rel-L2 (log) on twin axis
  - Discovered architecture as a colored block pipeline
  - Best-ever rel-L2 callout (big number)
  - Honest comparison vs pure-family baselines (mini bar)

Design notes:
  - 4 rows (one per PDE), 5 columns
  - All 4 PDEs share the same colour scheme for blocks
  - Highly readable when printed at poster scale (column width ~12")
  - 300 dpi PNG + vector PDF
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- paths

NOD_RESULTS = r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Neural Operator Discovery/results"
SWARM_DIR = os.path.join(NOD_RESULTS, "swarm_runs")
VAL_DIR = NOD_RESULTS
OUT_DIR = r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Poster/figures"
os.makedirs(OUT_DIR, exist_ok=True)

PDES = [
    {
        "key": "pwreg",
        "label": "Piecewise Regression  (1D)",
        "tag": "paper_pwreg_seed42",
        "val": "validation_baselines_1d_pwreg.json",
        "setup": "16 labs $\\times$ 20 iter $\\times$ 40 epochs $\\times$ 256 train  -  res 128",
        "color": "#8e44ad",
    },
    {
        "key": "advec",
        "label": "Linear Advection  (1D)",
        "tag": "paper_advec_seed42",
        "val": "validation_baselines_1d_advec.json",
        "setup": "16 labs $\\times$ 20 iter $\\times$ 40 epochs $\\times$ 256 train  -  res 128",
        "color": "#16a085",
    },
    {
        "key": "burgers",
        "label": "Burgers  (1D)",
        "tag": "paper_burgers_seed42",
        "val": "validation_baselines_1d_burgers.json",
        "setup": "16 labs $\\times$ 20 iter $\\times$ 40 epochs $\\times$ 256 train  -  res 128",
        "color": "#d35400",
    },
    {
        "key": "ns",
        "label": "Navier--Stokes  (2D)",
        "tag": "paper_ns_seed42",
        "val": None,
        "setup": "16 labs $\\times$ 20 iter $\\times$ 50 epochs $\\times$ 512 train  -  res 32$\\times$32",
        "color": "#1f4e79",
    },
]

BLOCK_COLORS = {
    "fourier":       "#e74c3c",
    "attention":     "#9b59b6",
    "wavelet":       "#27ae60",
    "residual_conv": "#f39c12",
    "branch_trunk":  "#3498db",
}
BLOCK_SHORT = {
    "fourier":       "F",
    "attention":     "A",
    "wavelet":       "W",
    "residual_conv": "R",
    "branch_trunk":  "T",
}
BLOCK_LABEL = {
    "fourier":       "Fourier",
    "attention":     "Attention",
    "wavelet":       "Wavelet",
    "residual_conv": "ResConv",
    "branch_trunk":  "Branch-Trunk",
}

# Hard-coded NS baseline numbers for the rightmost mini-bar (taken from
# validate_baselines_v3 / EXP-4 v3 results in memory):
NS_BASELINES = {
    "FNO h64 m12 (4.7M)":   0.0002,
    "POD-DeepONet (142K)":  0.0033,
    "DeepONet (603K)":      0.1613,
    "Transformer (105K)":   0.2624,
}
# Best-ever paper-grade NS number (lab 2 iter 16, 1.5M params)
NS_BEST_EVER_REL = 2.59e-4

# ---------------------------------------------------------------- loaders

def load_run(tag: str):
    base = os.path.join(SWARM_DIR, tag)
    if not os.path.isdir(base):
        return None
    iter_files = sorted(glob.glob(os.path.join(base, "iter_*.json")))
    if not iter_files:
        return None
    iters = [json.load(open(f)) for f in iter_files]
    final_p = os.path.join(base, "FINAL.json")
    final = json.load(open(final_p)) if os.path.exists(final_p) else None
    return {"iters": iters, "final": final}


def load_val(name: str):
    if name is None:
        return None
    p = os.path.join(VAL_DIR, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def fitness_and_rel(iters):
    fits, rels = [], []
    for d in iters:
        fits.append(d["global_best_fitness"])
        leaderboard = d.get("leaderboard", [])
        rels.append(min((l["rel_l2_clean"] for l in leaderboard),
                        default=np.nan) if leaderboard else np.nan)
    return np.array(fits), np.array(rels)


def best_ever(iters):
    best = None
    for d in iters:
        for lab in d.get("leaderboard", []):
            r = lab.get("rel_l2_clean")
            if r is None:
                continue
            if best is None or r < best["rel_l2_clean"]:
                best = {**lab, "iter": d["iteration"]}
    return best


# ---------------------------------------------------------------- drawing

def draw_arch_pipeline(ax, blocks, label_top="", label_bottom="", *,
                       y0=0.55, h=0.30, x0=0.02, x1=0.98):
    """Draw the architecture as a row of colored boxes inside `ax`."""
    n = max(len(blocks), 1)
    pad = (x1 - x0) / n
    bw = pad * 0.78
    for i, blk in enumerate(blocks):
        cx = x0 + (i + 0.5) * pad
        rect = FancyBboxPatch(
            (cx - bw / 2, y0), bw, h,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            facecolor=BLOCK_COLORS.get(blk, "#888888"),
            edgecolor="white", lw=1.0,
            transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(cx, y0 + h / 2, BLOCK_SHORT.get(blk, blk[:1].upper()),
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                transform=ax.transAxes)
        if i < n - 1:
            nxt = x0 + (i + 1.5) * pad
            ax.add_patch(FancyArrowPatch(
                (cx + bw / 2, y0 + h / 2),
                (nxt - bw / 2, y0 + h / 2),
                arrowstyle="->", mutation_scale=8,
                color="#666", lw=0.8,
                transform=ax.transAxes, clip_on=False))
    if label_top:
        ax.text(0.5, y0 + h + 0.10, label_top,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222",
                transform=ax.transAxes)
    if label_bottom:
        ax.text(0.5, y0 - 0.10, label_bottom,
                ha="center", va="top",
                fontsize=8, color="#666", style="italic",
                transform=ax.transAxes)


def draw_metric_callout(ax, value_str, sublabel,
                         color="#1f4e79", bg="#f4f4f4"):
    ax.add_patch(Rectangle((0, 0), 1, 1,
                            facecolor=bg, edgecolor="#dcdcdc", lw=0.8,
                            transform=ax.transAxes))
    ax.text(0.5, 0.62, value_str, ha="center", va="center",
            fontsize=24, fontweight="bold", color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.22, sublabel, ha="center", va="center",
            fontsize=8.5, color="#444",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_bar_compare(ax, labels, values, highlight_idx, *,
                      log=True, xlabel="rel L2"):
    y = np.arange(len(labels))
    bars = ax.barh(y, values,
                   color=["#c0392b" if i == highlight_idx else "#bdc3c7"
                          for i in range(len(labels))],
                   edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", alpha=0.25, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, v in enumerate(values):
        ax.text(v * 1.15 if log else v + 0.005, i,
                f"{v:.4f}", va="center", fontsize=7.5,
                color="#222")


# ---------------------------------------------------------------- main figure

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Pull data per PDE
    rows = []
    for cfg in PDES:
        run = load_run(cfg["tag"])
        val = load_val(cfg["val"])
        rows.append({"cfg": cfg, "run": run, "val": val})

    n_rows = len(rows)

    fig = plt.figure(figsize=(20, 13), constrained_layout=False)
    # Use gridspec with 5 cols + a small left "label" column
    # Columns:  PDE | conv. curve | architecture | best-ever | vs baselines
    gs = fig.add_gridspec(
        n_rows + 2, 5,
        height_ratios=[0.6] + [1.0] * n_rows + [0.35],
        width_ratios=[1.4, 1.6, 2.4, 1.0, 1.7],
        hspace=0.55, wspace=0.30,
        left=0.04, right=0.98, top=0.96, bottom=0.04,
    )

    # ----- Header banner row (spans all cols)
    axH = fig.add_subplot(gs[0, :])
    axH.axis("off")
    axH.text(0.0, 0.7,
             "Agentic Discovery of Neural Operator Architectures",
             ha="left", va="center",
             fontsize=24, fontweight="bold", color="#500000",
             transform=axH.transAxes)
    axH.text(0.0, 0.15,
             "AI Scientific Community swarm  -  16 virtual labs, citation-based economy of influence,"
             "  PSO coordinator  -  paper-grade run, seed 42, RTX 4080",
             ha="left", va="center",
             fontsize=12, color="#444",
             transform=axH.transAxes)
    axH.text(1.0, 0.7,
             "ICERM 2026  -  Hot Topics: Agentic SciML",
             ha="right", va="center",
             fontsize=11, color="#444", style="italic",
             transform=axH.transAxes)
    axH.text(1.0, 0.15,
             "L. F. Loo, U. Braga-Neto - Texas A&M University",
             ha="right", va="center",
             fontsize=10, color="#666",
             transform=axH.transAxes)

    # ----- Per-PDE rows
    for r, item in enumerate(rows):
        cfg = item["cfg"]
        run = item["run"]
        val = item["val"]
        gs_row = r + 1

        # Left column: PDE label + setup
        axL = fig.add_subplot(gs[gs_row, 0])
        axL.axis("off")
        axL.add_patch(Rectangle((0, 0), 0.05, 1, facecolor=cfg["color"],
                                 transform=axL.transAxes, clip_on=False))
        axL.text(0.10, 0.88, cfg["label"],
                 ha="left", va="top",
                 fontsize=15, fontweight="bold", color="#222",
                 transform=axL.transAxes)
        axL.text(0.10, 0.62, cfg["setup"],
                 ha="left", va="top",
                 fontsize=9, color="#555",
                 transform=axL.transAxes, wrap=True)
        if run and run.get("final"):
            g = run["final"]["global_best_genome"]
            axL.text(0.10, 0.35,
                     f"Composite fitness: {run['final']['global_best_fitness']:.4f}",
                     ha="left", va="top",
                     fontsize=10, fontweight="bold", color=cfg["color"],
                     transform=axL.transAxes)
            axL.text(0.10, 0.20,
                     f"hidden ch.\\,= {g['hidden_channels']}, modes = {g['fourier_modes']}, act = {g['activation']}",
                     ha="left", va="top",
                     fontsize=8.5, color="#444",
                     transform=axL.transAxes)

        # Middle 1: convergence curve (twin axis)
        axC = fig.add_subplot(gs[gs_row, 1])
        if run:
            fits, rels = fitness_and_rel(run["iters"])
            xs = np.arange(1, len(fits) + 1)
            axC.plot(xs, fits, "-o", color="#1f4e79", lw=1.8, ms=4,
                     label="comp. fitness")
            ax2 = axC.twinx()
            ax2.plot(xs, rels, "-s", color="#c0392b", lw=1.4, ms=3,
                     alpha=0.85, label="best rel L2")
            ax2.set_yscale("log")
            ax2.tick_params(axis="y", labelsize=7.5, colors="#c0392b")
            ax2.set_ylabel("rel L2 [log]", fontsize=8.5, color="#c0392b")
            for s in ("top",):
                ax2.spines[s].set_visible(False)
            axC.set_xlabel("iter", fontsize=8.5)
            axC.set_ylabel("fitness", fontsize=8.5, color="#1f4e79")
            axC.tick_params(axis="y", labelsize=7.5, colors="#1f4e79")
            axC.tick_params(axis="x", labelsize=7.5)
            axC.set_xlim(0.5, len(fits) + 0.5)
            axC.grid(True, alpha=0.25)
        axC.set_title("Convergence", fontsize=9.5)

        # Middle 2: architecture
        axA = fig.add_subplot(gs[gs_row, 2])
        axA.axis("off")
        axA.set_title("Discovered architecture", fontsize=9.5,
                      loc="left", x=0.02, y=0.98)
        if run and run.get("final"):
            blocks = run["final"]["global_best_genome"]["block_sequence"]
            # composite winner
            draw_arch_pipeline(
                axA, blocks,
                label_top=f"composite winner  -  {len(blocks)} blocks",
                label_bottom="",
                y0=0.58, h=0.27, x0=0.02, x1=0.98)
            # best-ever (only if different from composite winner)
            be = best_ever(run["iters"])
            if be is not None:
                be_blocks = be["blocks"]
                same = be_blocks == blocks
                if not same:
                    draw_arch_pipeline(
                        axA, be_blocks,
                        label_top=f"best-ever rel L2 architecture  -  {len(be_blocks)} blocks",
                        label_bottom="",
                        y0=0.10, h=0.25, x0=0.02, x1=0.98)
                else:
                    axA.text(0.5, 0.20,
                             "(composite winner = best-ever architecture)",
                             ha="center", va="top", fontsize=8,
                             color="#666", style="italic",
                             transform=axA.transAxes)

        # Right 1: best-ever rel-L2 callout
        axM = fig.add_subplot(gs[gs_row, 3])
        if cfg["key"] == "ns":
            val_text = f"{NS_BEST_EVER_REL:.2e}"
            sub = "best-ever rel L2\n(lab 2, iter 16, 1.5M params)"
        else:
            be = best_ever(run["iters"]) if run else None
            if be is not None:
                val_text = f"{be['rel_l2_clean']:.4f}"
                sub = (f"best-ever rel L2\n"
                       f"(lab {be['lab_id']} {be['paradigm']}, "
                       f"iter {be['iter']+1}, {be.get('params','?'):,} params)")
            else:
                val_text = "-"
                sub = "(no data)"
        draw_metric_callout(axM, val_text, sub,
                             color=cfg["color"], bg="#fdfdf6")

        # Right 2: vs baselines bar
        axB = fig.add_subplot(gs[gs_row, 4])
        if cfg["key"] == "ns":
            labels = list(NS_BASELINES.keys()) + ["Discovered Hybrid (1.5M)"]
            values = list(NS_BASELINES.values()) + [NS_BEST_EVER_REL]
            order = sorted(range(len(values)), key=lambda i: values[i])
            labels = [labels[i] for i in order]
            values = [values[i] for i in order]
            highlight = labels.index("Discovered Hybrid (1.5M)")
        elif val is not None:
            res = sorted(val["results"], key=lambda r: r["rel_l2_clean"])
            labels = []
            values = []
            highlight = -1
            for k, r in enumerate(res):
                short = (r["name"]
                         .replace("Pure ", "")
                         .replace(" (4xfourier)", "")
                         .replace(" (3xattention)", "")
                         .replace(" (4xbranch_trunk)", "")
                         .replace(" (4xwavelet)", "")
                         .replace(f" ({cfg['tag']})", ""))
                labels.append(short)
                values.append(r["rel_l2_clean"])
                if "Discovered Hybrid" in r["name"]:
                    highlight = k
        else:
            labels, values, highlight = [], [], -1
        if labels:
            draw_bar_compare(axB, labels, values, highlight,
                             log=True, xlabel="rel L2 (clean)")
        axB.set_title("Vs.\\ pure-family baselines",
                      fontsize=9.5, loc="left", x=0.0)

    # ----- Footer (block legend)
    axF = fig.add_subplot(gs[-1, :])
    axF.axis("off")
    legend_items = [
        ("fourier", "Fourier (FNO)"),
        ("attention", "Attention (Transformer / GNOT)"),
        ("wavelet", "Wavelet (MWT)"),
        ("residual_conv", "Residual Conv (UNet-style)"),
        ("branch_trunk", "Branch-Trunk (DeepONet)"),
    ]
    n_items = len(legend_items)
    box_w = 0.020
    pad_x = 0.18
    start_x = 0.5 - (n_items - 1) * pad_x / 2 - box_w
    for i, (b, lbl) in enumerate(legend_items):
        cx = start_x + i * pad_x
        axF.add_patch(FancyBboxPatch(
            (cx, 0.45), box_w, 0.30,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            facecolor=BLOCK_COLORS[b], edgecolor="white", lw=0.5,
            transform=axF.transAxes, clip_on=False))
        axF.text(cx + box_w + 0.005, 0.60, lbl,
                 ha="left", va="center", fontsize=10,
                 color="#222", transform=axF.transAxes)
    axF.text(0.5, 0.05,
             "Block library used by every lab; the 5-letter pipeline "
             "above each PDE row is the genome that the swarm converged on.",
             ha="center", va="center", fontsize=8.5, color="#666",
             style="italic", transform=axF.transAxes)

    out_png = os.path.join(OUT_DIR, "master_results.png")
    out_pdf = os.path.join(OUT_DIR, "master_results.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                 facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
