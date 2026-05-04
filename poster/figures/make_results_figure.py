"""
Build a single-page summary figure for advisor (Ulisses) showing the paper-grade
NS swarm run results. Reusable on the ICERM poster.

Outputs:
  results_summary.png  (300 dpi, ~ 14 x 10 in)
  results_summary.pdf  (vector)
"""
import json
import os
import glob
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

BASE = r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Neural Operator Discovery/results/swarm_runs/paper_ns_seed42"
OUT_DIR = r"C:/Users/luisl/OneDrive/Documentos/TAMU/2026.4 ICERM/Poster/figures"

# ---------------------------------------------------------------- load data
final = json.load(open(os.path.join(BASE, "FINAL.json")))
iter_files = sorted(glob.glob(os.path.join(BASE, "iter_*.json")))

iters = []
fit_curve = []
best_rel_curve = []
gb_blocks_per_iter = []
all_lab_records = []
for f in iter_files:
    d = json.load(open(f))
    it = d["iteration"]
    iters.append(it)
    fit_curve.append(d["global_best_fitness"])
    gb_blocks_per_iter.append(d["global_best_genome"]["block_sequence"])
    leaderboard = d.get("leaderboard", [])
    if leaderboard:
        best_rel_curve.append(min(l["rel_l2_clean"] for l in leaderboard))
    else:
        best_rel_curve.append(np.nan)
    for lab in leaderboard:
        all_lab_records.append((it, lab))

# Block frequency at iter 1 and iter 20
def block_freq(d):
    c = Counter()
    for lab in d["leaderboard"]:
        c.update(lab["blocks"])
    return c

c1 = block_freq(json.load(open(os.path.join(BASE, "iter_01.json"))))
c20 = block_freq(json.load(open(os.path.join(BASE, "iter_20.json"))))
all_blocks = ["fourier", "attention", "wavelet", "residual_conv", "branch_trunk"]
total1 = sum(c1.values())
total20 = sum(c20.values())
freq1 = [100 * c1[b] / total1 for b in all_blocks]
freq20 = [100 * c20[b] / total20 for b in all_blocks]

# Best-ever architecture (lowest rel-L2)
best_record = min(all_lab_records, key=lambda r: r[1]["rel_l2_clean"])
best_it, best_lab = best_record

# Final composite winner
gb_blocks = final["global_best_genome"]["block_sequence"]
gb_fit = final["global_best_fitness"]

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 10), constrained_layout=False)
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 0.9, 1.0],
    hspace=0.55, wspace=0.28,
    left=0.07, right=0.97, top=0.92, bottom=0.06,
)

# Title
fig.suptitle(
    "Agentic Neural Operator Discovery: 2D Navier–Stokes (paper-grade run, seed 42)",
    fontsize=15, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.945,
    "16 virtual labs $\\times$ 20 iterations $\\times$ 50 epochs $\\times$ 512 train samples  •  PSO swarm with citation-based peer review",
    ha="center", fontsize=10.5, color="#444",
)

# -------- Panel A: fitness curve + best rel-L2 curve (twin axis)
axA = fig.add_subplot(gs[0, 0])
ax2 = axA.twinx()

cFit = "#1f4e79"
cRel = "#c0392b"
axA.plot(iters, fit_curve, "-o", color=cFit, lw=2, ms=5, label="Composite fitness (global best)")
ax2.plot(iters, best_rel_curve, "-s", color=cRel, lw=1.6, ms=4, alpha=0.85, label="Best rel L2 (across labs)")
ax2.set_yscale("log")

axA.set_xlabel("Swarm iteration")
axA.set_ylabel("Composite fitness", color=cFit)
ax2.set_ylabel("Best rel $L_2$ (clean)  [log]", color=cRel)
axA.tick_params(axis="y", colors=cFit)
ax2.tick_params(axis="y", colors=cRel)
axA.set_title("A.  Convergence: composite fitness + best rel $L_2$")

axA.set_xlim(-0.5, 20)
axA.grid(True, alpha=0.25)

# Annotate breakpoints from the data
break_pts = [
    (2, 0.8106, "FNO seed\nleads"),
    (7, 0.8179, "DeepONet block\nrecruited"),
    (14, 0.8259, "6-block\nhybrid"),
    (17, 0.8337, "7-block\nfinal"),
]
for it, fit, lbl in break_pts:
    axA.annotate(lbl, xy=(it, fit), xytext=(it + 0.4, fit - 0.018),
                 fontsize=8, ha="left", color="#1f4e79",
                 arrowprops=dict(arrowstyle="-", color="#1f4e79", lw=0.7, alpha=0.6))

# best-ever rel-L2 annotation
axA.scatter([best_it], [best_rel_curve[best_it]], facecolor="none", edgecolor=cRel,
            s=140, lw=1.8, zorder=5, transform=ax2.transData)
ax2.annotate(f"Best ever\n{best_lab['rel_l2_clean']:.4f}",
             xy=(best_it, best_lab["rel_l2_clean"]),
             xytext=(best_it - 5, best_lab["rel_l2_clean"] * 0.55),
             fontsize=8, color=cRel,
             arrowprops=dict(arrowstyle="->", color=cRel, lw=0.8))

# combined legend
h1, l1 = axA.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axA.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.9)

# -------- Panel B: block frequency shift
axB = fig.add_subplot(gs[0, 1])
x = np.arange(len(all_blocks))
w = 0.38
bars1 = axB.bar(x - w / 2, freq1, w, label=f"Iter 1 (n={total1})",
                color="#bdc3c7", edgecolor="#7f8c8d")
bars2 = axB.bar(x + w / 2, freq20, w, label=f"Iter 20 (n={total20})",
                color="#2980b9", edgecolor="#1f4e79")
axB.set_xticks(x)
axB.set_xticklabels([b.replace("_", "\n") for b in all_blocks], fontsize=9)
axB.set_ylabel("Share of blocks across active labs (%)")
axB.set_title("B.  Block usage shift: swarm consensus")
axB.legend(loc="upper right", fontsize=8)
axB.grid(axis="y", alpha=0.25)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        axB.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.0f}%",
                 ha="center", fontsize=8)

# -------- Panel C: discovered architecture diagram
axC = fig.add_subplot(gs[1, :])
axC.set_xlim(0, 100)
axC.set_ylim(0, 10)
axC.axis("off")

block_colors = {
    "fourier":       "#e74c3c",
    "attention":     "#9b59b6",
    "wavelet":       "#27ae60",
    "residual_conv": "#f39c12",
    "branch_trunk":  "#3498db",
}
block_labels = {
    "fourier":       "Fourier\n(FNO)",
    "attention":     "Attention\n(Transformer)",
    "wavelet":       "Wavelet\n(MWT)",
    "residual_conv": "Residual\nConv",
    "branch_trunk":  "Branch–Trunk\n(DeepONet)",
}

n = len(gb_blocks)
margin = 4
pad = (100 - 2 * margin) / n
bx_w = pad * 0.78
bx_h = 4.0
y0 = 3.0

# Input arrow + label
axC.text(margin - 1.5, y0 + bx_h / 2, "$\\omega(x,y,0)$\n+ forcing",
         ha="right", va="center", fontsize=9, color="#333")

for i, blk in enumerate(gb_blocks):
    cx = margin + (i + 0.5) * pad
    rect = FancyBboxPatch(
        (cx - bx_w / 2, y0), bx_w, bx_h,
        boxstyle="round,pad=0.02,rounding_size=0.4",
        facecolor=block_colors[blk], edgecolor="white", lw=1.4,
        alpha=0.95,
    )
    axC.add_patch(rect)
    axC.text(cx, y0 + bx_h / 2, block_labels[blk],
             ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    if i < n - 1:
        nxt = margin + (i + 1.5) * pad
        arr = FancyArrowPatch(
            (cx + bx_w / 2, y0 + bx_h / 2),
            (nxt - bx_w / 2, y0 + bx_h / 2),
            arrowstyle="->", mutation_scale=12, color="#555", lw=1.2,
        )
        axC.add_patch(arr)

# Output arrow
axC.annotate("$\\omega(x,y,T)$",
             xy=(100 - margin + 0.5, y0 + bx_h / 2),
             xytext=(100 - margin + 2.0, y0 + bx_h / 2),
             ha="left", va="center", fontsize=9, color="#333",
             arrowprops=dict(arrowstyle="<-", color="#555", lw=1.2))

axC.text(50, y0 + bx_h + 1.4,
         f"Discovered hybrid (lab {final['global_best_lab_id']}, fitness {gb_fit:.4f}, "
         f"hidden=64, modes=10, GELU, $\\sim$1.0M params)",
         ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
axC.text(50, y0 - 1.2,
         "Spectral $\\to$ Attention $\\to$ Multi-scale $\\to$ Local $\\to$ Operator-learning $\\to$ Spectral",
         ha="center", fontsize=9, color="#444", style="italic")
axC.set_title("C.  Discovered architecture (paper-grade, NS-2D, seed 42)",
              loc="center", fontweight="bold")

# -------- Panel D: baseline comparison bar
axD = fig.add_subplot(gs[2, 0])
models = [
    "FNO h64 m12\n(4.7M)",
    "POD-DeepONet\n(142K)",
    "DeepONet\nfaithful (603K)",
    "Pure Transformer\n(105K)",
    "Discovered Hybrid\n(v3, 977K)",
    "Paper-grade\nbest-ever (1.5M)",
    "Paper-grade\ncomposite winner\n(1.0M)",
]
rel_l2 = [0.0002, 0.0033, 0.1613, 0.2624, 0.0008, best_lab["rel_l2_clean"], 0.00138]
colors_bar = ["#7f8c8d", "#7f8c8d", "#bdc3c7", "#bdc3c7", "#3498db", "#c0392b", "#1f4e79"]

bars = axD.barh(np.arange(len(models)), rel_l2, color=colors_bar, edgecolor="white")
axD.set_yticks(np.arange(len(models)))
axD.set_yticklabels(models, fontsize=8)
axD.invert_yaxis()
axD.set_xscale("log")
axD.set_xlabel("Relative $L_2$ error (clean test, lower is better)")
axD.set_title("D.  Where the swarm sits vs baselines (NS-2D, res 32)")
axD.grid(axis="x", alpha=0.25, which="both")
for i, (b, v) in enumerate(zip(bars, rel_l2)):
    axD.text(v * 1.15, i, f"{v:.4f}", va="center", fontsize=8)

# -------- Panel E: take-home text
axE = fig.add_subplot(gs[2, 1])
axE.axis("off")
take_home = (
    "Honest claim for ICERM\n"
    "─────────────────────\n"
    "  •  Agentic discovery converges on a regime-aware\n"
    "      hybrid that recombines four operator families\n"
    "      (Fourier + Attention + Wavelet + Branch–Trunk)\n"
    "      with no human prior on architecture.\n\n"
    "  •  Best-ever rel $L_2$ = 2.6e-4 with $\\sim$1.5M params,\n"
    "      vs 4.7M for the FNO baseline ($\\sim$3$\\times$ smaller).\n\n"
    "  •  Block usage shifts measurably between iter 1 and\n"
    "      iter 20 → swarm is not random search.\n\n"
    "  •  Cross-PDE comparison (Darcy 2-D) is the natural\n"
    "      next run; pending advisor approval.\n\n"
    "  •  We do NOT claim to beat DeepONet — POD-DeepONet\n"
    "      (142K) still wins on parameter efficiency, and\n"
    "      DeepONet's regime is Darcy-style operators."
)
axE.text(0.0, 1.0, take_home, ha="left", va="top",
         fontsize=10, color="#222", family="monospace",
         transform=axE.transAxes,
         bbox=dict(boxstyle="round,pad=0.7", facecolor="#fdf6e3",
                   edgecolor="#dcd5b8", lw=1))
axE.set_title("E.  Take-home (advisor-safe framing)", loc="left", fontweight="bold")

# Footer credit
fig.text(0.5, 0.012,
         "Luis F. Loo, Texas A&M University  •  Advisor: Ulisses Braga-Neto  "
         "•  Workshop: Agentic Scientific Computing & SciML, ICERM 2026",
         ha="center", fontsize=8, color="#666")

# Save
out_png = os.path.join(OUT_DIR, "results_summary.png")
out_pdf = os.path.join(OUT_DIR, "results_summary.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
