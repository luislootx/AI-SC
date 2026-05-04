"""1D-validation figures for the ICERM poster.

Reads the per-iteration JSON snapshots written by `run_swarm_1d.py` for the
three abstract-mandated benchmarks (piecewise regression, linear advection,
Burgers) and produces:

  results/figures/1d_<pde>.png        -- one figure per problem (for the
                                         GitHub README): fitness curve, best
                                         arch diagram, paradigm breakdown.
  results/figures/1d_combined.png     -- one combined figure: fitness curves
                                         + block-frequency evolution per PDE
                                         in a 4-row layout.

Run after the 3 swarm runs finish:
  python make_1d_figures.py

Inputs hardcoded (paths assume the repo layout from `Neural Operator
Discovery/`); override via env-vars NOD_RESULTS / OUT_DIR.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- config

NOD_RESULTS = os.environ.get(
    "NOD_RESULTS",
    r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Neural Operator Discovery/results/swarm_runs",
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Poster/figures",
)
os.makedirs(OUT_DIR, exist_ok=True)

PDE_RUNS = {
    "pwreg":   ("paper_pwreg_seed42",   "Piecewise Regression (1D)"),
    "advec":   ("paper_advec_seed42",   "Linear Advection (1D)"),
    "burgers": ("paper_burgers_seed42", "Burgers (1D)"),
}

BLOCK_COLORS = {
    "fourier":       "#e74c3c",
    "attention":     "#9b59b6",
    "wavelet":       "#27ae60",
    "residual_conv": "#f39c12",
    "branch_trunk":  "#3498db",
}
BLOCK_LABELS = {
    "fourier":       "Fourier",
    "attention":     "Attention",
    "wavelet":       "Wavelet",
    "residual_conv": "ResConv",
    "branch_trunk":  "Branch-Trunk",
}
ALL_BLOCKS = list(BLOCK_COLORS.keys())


# ---------------------------------------------------------------- loaders

def load_run(tag: str) -> dict | None:
    """Return None if the run is missing / incomplete."""
    run_dir = os.path.join(NOD_RESULTS, tag)
    if not os.path.isdir(run_dir):
        return None
    iter_files = sorted(glob.glob(os.path.join(run_dir, "iter_*.json")))
    if not iter_files:
        return None
    iters = [json.load(open(f)) for f in iter_files]
    final = None
    final_path = os.path.join(run_dir, "FINAL.json")
    if os.path.exists(final_path):
        final = json.load(open(final_path))
    return {"tag": tag, "run_dir": run_dir, "iters": iters, "final": final}


def best_per_iter(iters):
    """Return (xs, fits, rels, gb_blocks) per iteration."""
    xs, fits, rels, gbs = [], [], [], []
    for d in iters:
        xs.append(d["iteration"])
        fits.append(d["global_best_fitness"])
        leaderboard = d.get("leaderboard", [])
        rels.append(min((l["rel_l2_clean"] for l in leaderboard),
                        default=np.nan) if leaderboard else np.nan)
        gbs.append(d.get("global_best_genome", {}).get("block_sequence", []))
    return np.array(xs), np.array(fits), np.array(rels), gbs


def block_freq_at(iter_d):
    c = Counter()
    for lab in iter_d.get("leaderboard", []):
        c.update(lab["blocks"])
    total = sum(c.values()) or 1
    return [100 * c.get(b, 0) / total for b in ALL_BLOCKS]


def best_ever(iters):
    best = None
    for d in iters:
        for lab in d.get("leaderboard", []):
            r = lab.get("rel_l2_clean")
            if r is None:
                continue
            if best is None or r < best["rel_l2_clean"]:
                best = {**lab, "iteration": d["iteration"]}
    return best


# ---------------------------------------------------------------- single-PDE figure

def make_per_problem_figure(run, pde_key, label):
    iters = run["iters"]
    final = run["final"] or {}
    xs, fits, rels, gbs = best_per_iter(iters)
    if not iters:
        return None

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.titleweight": "bold"})

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.7],
                          hspace=0.5, wspace=0.3,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    fig.suptitle(f"Agentic Discovery on {label}  -  paper-grade run, seed 42",
                 fontsize=13, fontweight="bold")

    # Panel A: fitness + rel-L2 curves
    axA = fig.add_subplot(gs[0, 0])
    ax2 = axA.twinx()
    axA.plot(xs, fits, "-o", color="#1f4e79", lw=2, label="composite fitness")
    if not np.all(np.isnan(rels)):
        ax2.plot(xs, rels, "-s", color="#c0392b", lw=1.5, alpha=0.85,
                 label="best rel L2 (clean)")
        ax2.set_yscale("log")
    axA.set_xlabel("Swarm iteration")
    axA.set_ylabel("Composite fitness", color="#1f4e79")
    ax2.set_ylabel("Best rel L2 [log]", color="#c0392b")
    axA.tick_params(axis="y", colors="#1f4e79")
    ax2.tick_params(axis="y", colors="#c0392b")
    axA.grid(True, alpha=0.25)
    axA.set_title("A.  Convergence")
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    # Panel B: block-frequency shift, iter 1 vs final
    axB = fig.add_subplot(gs[0, 1])
    f1 = block_freq_at(iters[0])
    fN = block_freq_at(iters[-1])
    x = np.arange(len(ALL_BLOCKS))
    w = 0.38
    axB.bar(x - w / 2, f1, w, color="#bdc3c7", edgecolor="#7f8c8d",
            label=f"Iter {iters[0]['iteration']+1}")
    axB.bar(x + w / 2, fN, w, color="#2980b9", edgecolor="#1f4e79",
            label=f"Iter {iters[-1]['iteration']+1}")
    axB.set_xticks(x)
    axB.set_xticklabels([BLOCK_LABELS[b] for b in ALL_BLOCKS],
                        rotation=15, fontsize=9)
    axB.set_ylabel("Share across active labs (%)")
    axB.set_title("B.  Block-usage shift")
    axB.legend(fontsize=8)
    axB.grid(axis="y", alpha=0.25)

    # Panel C: discovered architecture
    axC = fig.add_subplot(gs[1, :])
    axC.set_xlim(0, 100)
    axC.set_ylim(0, 10)
    axC.axis("off")
    gb_genome = (final.get("global_best_genome")
                 or iters[-1].get("global_best_genome", {}))
    gb_blocks = gb_genome.get("block_sequence", [])
    if not gb_blocks:
        axC.text(50, 5, "(no global best yet)", ha="center", fontsize=12)
    else:
        n = len(gb_blocks)
        margin = 4
        pad = (100 - 2 * margin) / max(n, 1)
        bx_w = pad * 0.78
        bx_h = 4.0
        y0 = 3.0
        axC.text(margin - 1.5, y0 + bx_h / 2, "input\nfield",
                 ha="right", va="center", fontsize=9, color="#333")
        for i, blk in enumerate(gb_blocks):
            cx = margin + (i + 0.5) * pad
            rect = FancyBboxPatch(
                (cx - bx_w / 2, y0), bx_w, bx_h,
                boxstyle="round,pad=0.02,rounding_size=0.4",
                facecolor=BLOCK_COLORS.get(blk, "#888"),
                edgecolor="white", lw=1.4, alpha=0.95)
            axC.add_patch(rect)
            axC.text(cx, y0 + bx_h / 2, BLOCK_LABELS.get(blk, blk),
                     ha="center", va="center", fontsize=8.5,
                     color="white", fontweight="bold")
            if i < n - 1:
                nxt = margin + (i + 1.5) * pad
                axC.add_patch(FancyArrowPatch(
                    (cx + bx_w / 2, y0 + bx_h / 2),
                    (nxt - bx_w / 2, y0 + bx_h / 2),
                    arrowstyle="->", mutation_scale=12,
                    color="#555", lw=1.2))
        axC.annotate("output\nfield",
                     xy=(100 - margin + 0.5, y0 + bx_h / 2),
                     xytext=(100 - margin + 2.0, y0 + bx_h / 2),
                     ha="left", va="center", fontsize=9, color="#333",
                     arrowprops=dict(arrowstyle="<-", color="#555", lw=1.2))
        be = best_ever(iters) or {}
        axC.text(50, y0 + bx_h + 1.5,
                 f"Composite winner ({n} blocks, ch={gb_genome.get('hidden_channels')}, "
                 f"modes={gb_genome.get('fourier_modes')}, {gb_genome.get('activation')})",
                 ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
        if be.get("rel_l2_clean") is not None:
            axC.text(50, y0 - 1.2,
                     f"Best-ever rel L2 = {be['rel_l2_clean']:.4f} "
                     f"(lab {be.get('lab_id')}, iter {be.get('iteration')+1})",
                     ha="center", fontsize=9, color="#444", style="italic")
    axC.set_title("C.  Discovered architecture", loc="left", fontweight="bold")

    out = os.path.join(OUT_DIR, f"1d_{pde_key}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- combined figure

def make_combined_figure(runs):
    """Combined figure: 3 rows (one per PDE) x 3 panels (fitness, rel-L2, block freq)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.titleweight": "bold"})

    rows = [(k, v) for k, v in PDE_RUNS.items() if k in runs and runs[k] is not None]
    n_rows = len(rows)
    if n_rows == 0:
        print("No runs available; skipping combined figure")
        return None

    fig = plt.figure(figsize=(13, 3.3 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.55, wspace=0.32,
                          left=0.06, right=0.97, top=0.93, bottom=0.07,
                          width_ratios=[1.05, 1.05, 1.0])

    fig.suptitle("Agentic Neural Operator Discovery on the 1D ICERM Benchmarks "
                 "(paper-grade, seed 42)",
                 fontsize=14, fontweight="bold")

    for r, (pde_key, (tag, label)) in enumerate(rows):
        run = runs[pde_key]
        iters = run["iters"]
        xs, fits, rels, _ = best_per_iter(iters)

        # Col 1: fitness
        axL = fig.add_subplot(gs[r, 0])
        axL.plot(xs, fits, "-o", color="#1f4e79", lw=2, ms=4)
        axL.set_xlabel("Iter")
        axL.set_ylabel("Composite fitness", color="#1f4e79")
        axL.set_title(f"{label}  -  fitness")
        axL.grid(True, alpha=0.25)

        # Col 2: rel-L2
        axM = fig.add_subplot(gs[r, 1])
        axM.plot(xs, rels, "-s", color="#c0392b", lw=1.6, ms=4)
        axM.set_xlabel("Iter")
        axM.set_ylabel("Best rel L2 [log]", color="#c0392b")
        axM.set_yscale("log")
        axM.set_title(f"{label}  -  best rel L2")
        axM.grid(True, alpha=0.25, which="both")

        # Col 3: stacked block-frequency over iterations
        axR = fig.add_subplot(gs[r, 2])
        freqs = np.array([block_freq_at(d) for d in iters])  # (T, B)
        bottom = np.zeros(len(iters))
        for j, b in enumerate(ALL_BLOCKS):
            axR.bar(xs + 1, freqs[:, j], bottom=bottom,
                    color=BLOCK_COLORS[b], width=0.85,
                    label=BLOCK_LABELS[b] if r == 0 else None)
            bottom += freqs[:, j]
        axR.set_xlabel("Iter")
        axR.set_ylabel("Block share (%)")
        axR.set_title(f"{label}  -  block evolution")
        axR.set_xlim(0.5, max(xs) + 1.5)
        axR.set_ylim(0, 100)
        if r == 0:
            axR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                       ncol=5, fontsize=8, frameon=False)

    out = os.path.join(OUT_DIR, "1d_combined.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- main

def main():
    runs = {k: load_run(tag) for k, (tag, _) in PDE_RUNS.items()}
    available = [k for k, r in runs.items() if r is not None]
    if not available:
        print("No 1D runs found. Run code/run_swarm_1d.py first.")
        return
    print(f"Found runs: {available}")

    for pde_key, (tag, label) in PDE_RUNS.items():
        if runs[pde_key] is None:
            print(f"  SKIP {pde_key} (no data)")
            continue
        out = make_per_problem_figure(runs[pde_key], pde_key, label)
        print(f"  wrote {out}")

    out = make_combined_figure(runs)
    if out:
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
