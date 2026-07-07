"""Publication-quality figures for the v2 campaign + exp4v3 baselines.

Resilient: each figure is independent and skipped (with a note) if its data is
not yet present, so it runs on a partial campaign. PDF+PNG to paper/figures/.

  fig_baselines    : cross-method rel L2 on NS vs Darcy (no universal winner)
  fig_ablation     : per-PDE discovered rel L2, LLM (Gemma 3) vs PSO
  fig_convergence  : best rel L2 vs iteration, LLM vs PSO, mean +/- band (multi-PDE)
  fig_pareto       : params vs rel L2 for all swarm labs + baselines (NS, Darcy)
  fig_blockevo     : block-type usage fraction across iterations (emergent recipe)
  fig_timing       : per-lab-eval wall time, LLM vs PSO (cost of agency)
  fig_archdiagram  : discovered block sequence per PDE, as colored chips
"""
from __future__ import annotations
import os, json, glob
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- publication styling (NeurIPS-friendly) ----
plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "swarm_runs")
EXP4 = os.path.join(ROOT, "results", "exp4v3")
FIG = os.path.join(ROOT, "paper", "figures")
SEEDS = [42, 137, 2024]
PDES = ["pwreg", "advec", "burgers", "ns", "darcy"]
PDE_LABEL = {"pwreg": "Piecewise Reg.", "advec": "Advection",
             "burgers": "Burgers", "ns": "Navier-Stokes", "darcy": "Darcy"}
# block-type palette (consistent across all figures)
BLOCKS = ["fourier", "attention", "wavelet", "residual_conv", "branch_trunk"]
BCOL = {"fourier": "#1f77b4", "attention": "#d62728", "wavelet": "#2ca02c",
        "residual_conv": "#ff7f0e", "branch_trunk": "#9467bd"}
BABBR = {"fourier": "F", "attention": "A", "wavelet": "W",
         "residual_conv": "R", "branch_trunk": "T"}


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}")


def _base_paradigm(p):
    return p.split("_spawn")[0] if p else p


def _iters(tag):
    return sorted(glob.glob(os.path.join(RUNS, tag, "iter_*.json")))


# ------------------------------------------------------------------ baselines
def fig_baselines():
    agg = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(EXP4, "*.json")):
        obj = _load(f)
        for d in (obj if isinstance(obj, list) else [obj] if obj else []):
            if isinstance(d, dict) and "rel_l2_clean" in d:
                agg[d.get("pde")][d.get("model")].append(d["rel_l2_clean"])
    if not agg.get("ns2d") or not agg.get("darcy2d"):
        print("  skip fig_baselines (no exp4v3)"); return
    models = sorted({m for pde in ("ns2d", "darcy2d") for m in agg[pde]},
                    key=lambda m: mean(agg["ns2d"].get(m, [9])))
    ns = [mean(agg["ns2d"].get(m, [np.nan])) for m in models]
    dc = [mean(agg["darcy2d"].get(m, [np.nan])) for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, ns, w, label="Navier-Stokes 2D", color="#1f77b4")
    ax.bar(x + w/2, dc, w, label="Darcy 2D", color="#ff7f0e")
    ax.set_yscale("log"); ax.set_ylabel("relative $L_2$ (clean)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("No universal winner: optimal operator is PDE-dependent")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3, which="both")
    _save(fig, "fig_baselines")


# ------------------------------------------------------------------ ablation
def fig_ablation():
    agg = _load(os.path.join(ROOT, "results", "aggregate_v2.json"))
    disc = (agg or {}).get("discovery", {})
    labs, llm, pso = [], [], []
    for pde in PDES:
        d = disc.get(pde)
        if not d:
            continue
        l = d["llm"]["rel_l2_clean"]; s = d["pso"]["rel_l2_clean"]
        if l or s:
            labs.append(PDE_LABEL[pde])
            llm.append(l["mean"] if l else 0); pso.append(s["mean"] if s else 0)
    if not labs:
        print("  skip fig_ablation (no v2 runs yet)"); return
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, llm, w, label="LLM agents (Gemma 3)", color="#d62728")
    ax.bar(x + w/2, pso, w, label="PSO (rules)", color="#7f7f7f")
    ax.set_yscale("log"); ax.set_ylabel("discovered rel $L_2$ (clean)")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_title("Ablation: LLM-agent vs.\\ rule-based coordination")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3, which="both")
    _save(fig, "fig_ablation")


# ------------------------------------------------------------------ convergence
def _best_rel_curve(tag):
    ys = []
    for it in _iters(tag):
        d = _load(it) or {}
        lb = [e["rel_l2_clean"] for e in d.get("leaderboard", [])
              if e.get("rel_l2_clean") is not None]
        if lb:
            ys.append(min(lb))
    return ys


def fig_convergence():
    pdes = [p for p in PDES if any(_iters(f"v2_llm_{p}_seed{s}") for s in SEEDS)]
    if not pdes:
        print("  skip fig_convergence (no runs yet)"); return
    n = len(pdes)
    fig, axes = plt.subplots(1, n, figsize=(2.7*n, 3), squeeze=False)
    for ax, pde in zip(axes[0], pdes):
        for cond, col in (("llm", "#d62728"), ("pso", "#7f7f7f")):
            curves = [_best_rel_curve(f"v2_{cond}_{pde}_seed{s}") for s in SEEDS]
            curves = [c for c in curves if c]
            if not curves:
                continue
            m = min(len(c) for c in curves)
            arr = np.array([c[:m] for c in curves])
            xs = np.arange(1, m+1)
            mu = arr.mean(0)
            ax.plot(xs, mu, color=col, label=cond.upper(), lw=1.8)
            if arr.shape[0] > 1:
                sd = arr.std(0)
                ax.fill_between(xs, mu-sd, mu+sd, color=col, alpha=0.18)
        ax.set_yscale("log"); ax.set_title(PDE_LABEL[pde])
        ax.set_xlabel("iteration"); ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("best rel $L_2$")
    axes[0][-1].legend(frameon=False)
    fig.suptitle("Swarm convergence: LLM agents vs.\\ PSO (mean $\\pm$ s.d.\\ over seeds)",
                 y=1.04)
    _save(fig, "fig_convergence")


# ------------------------------------------------------------------ Pareto
def _baseline_points(pde2d):
    pts = []
    for f in glob.glob(os.path.join(EXP4, f"{pde2d}__*.json")):
        obj = _load(f)
        for d in (obj if isinstance(obj, list) else [obj] if obj else []):
            if isinstance(d, dict) and "rel_l2_clean" in d and d.get("params"):
                pts.append((d["model"], d["params"], d["rel_l2_clean"]))
    # average per model
    agg = defaultdict(lambda: [[], []])
    for m, p, r in pts:
        agg[m][0].append(p); agg[m][1].append(r)
    return [(m, mean(v[0]), mean(v[1])) for m, v in agg.items()]


def fig_pareto():
    mapping = {"ns": "ns2d", "darcy": "darcy2d"}
    have = [(p, q) for p, q in mapping.items()
            if any(_iters(f"v2_llm_{p}_seed{s}") for s in SEEDS)
            and glob.glob(os.path.join(EXP4, f"{q}__*.json"))]
    if not have:
        print("  skip fig_pareto (need 2D swarm + baselines)"); return
    fig, axes = plt.subplots(1, len(have), figsize=(5*len(have), 4), squeeze=False)
    for ax, (pde, pde2d) in zip(axes[0], have):
        # all swarm labs from final iteration, all seeds
        seen_par = set()
        for s in SEEDS:
            its = _iters(f"v2_llm_{pde}_seed{s}")
            if not its:
                continue
            for e in (_load(its[-1]) or {}).get("leaderboard", []):
                if e.get("params") and e.get("rel_l2_clean"):
                    bp = _base_paradigm(e["paradigm"])
                    ax.scatter(e["params"], e["rel_l2_clean"], s=28, alpha=0.55,
                               color={"fno": "#1f77b4", "deeponet": "#9467bd",
                                      "transformer": "#d62728", "wavelet": "#2ca02c"}
                               .get(bp, "#7f7f7f"),
                               label=bp if bp not in seen_par else None,
                               edgecolors="none")
                    seen_par.add(bp)
        # baselines as black stars
        for m, p, r in _baseline_points(pde2d):
            ax.scatter(p, r, marker="*", s=140, color="black", zorder=5)
            ax.annotate(m, (p, r), fontsize=6.5, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("parameters"); ax.set_title(f"{PDE_LABEL[pde]} 2D")
        ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, title="swarm labs", fontsize=7)
    axes[0][0].set_ylabel("relative $L_2$ (clean)")
    fig.suptitle("Accuracy-parameter trade-off: swarm labs ($\\circ$) vs.\\ baselines ($\\star$)",
                 y=1.02)
    _save(fig, "fig_pareto")


# ------------------------------------------------------------------ block evolution
def fig_blockevo():
    pdes = [p for p in PDES if _iters(f"v2_llm_{p}_seed42")]
    if not pdes:
        print("  skip fig_blockevo (no runs yet)"); return
    n = len(pdes)
    fig, axes = plt.subplots(1, n, figsize=(2.7*n, 3), squeeze=False)
    for ax, pde in zip(axes[0], pdes):
        # aggregate block fractions per iteration across seeds
        per_iter_frac = defaultdict(lambda: defaultdict(list))
        for s in SEEDS:
            for it in _iters(f"v2_llm_{pde}_seed{s}"):
                d = _load(it) or {}
                i = d.get("iteration", 0)
                cnt = defaultdict(int); tot = 0
                for e in d.get("leaderboard", []):
                    for b in e.get("blocks", []):
                        cnt[b] += 1; tot += 1
                if tot:
                    for b in BLOCKS:
                        per_iter_frac[i][b].append(100*cnt[b]/tot)
        if not per_iter_frac:
            continue
        its = sorted(per_iter_frac)
        xs = [i+1 for i in its]
        series = {b: [mean(per_iter_frac[i][b]) if per_iter_frac[i][b] else 0
                      for i in its] for b in BLOCKS}
        ax.stackplot(xs, *[series[b] for b in BLOCKS],
                     colors=[BCOL[b] for b in BLOCKS],
                     labels=[BABBR[b] for b in BLOCKS], alpha=0.9)
        ax.set_title(PDE_LABEL[pde]); ax.set_xlabel("iteration")
        ax.set_xlim(min(xs), max(xs)); ax.set_ylim(0, 100)
        ax.margins(0)
    axes[0][0].set_ylabel("block usage (%)")
    axes[0][-1].legend(frameon=False, ncol=1, loc="center left",
                       bbox_to_anchor=(1.0, 0.5), title="block")
    fig.suptitle("Maintained diversity: population block-usage stays mixed "
                 "(no groupthink collapse)", y=1.04)
    _save(fig, "fig_blockevo")


# ------------------------------------------------------------------ timing
def fig_timing():
    agg = _load(os.path.join(ROOT, "results", "aggregate_v2.json"))
    tim = (agg or {}).get("timing", {})
    labs, llm, pso = [], [], []
    for pde in PDES:
        t = tim.get(pde)
        if not t:
            continue
        li = t["llm"]["iter_s"]; pi = t["pso"]["iter_s"]
        labs.append(PDE_LABEL[pde])
        llm.append(li["mean"]/16 if li else 0)
        pso.append(pi["mean"]/16 if pi else 0)
    if not labs:
        print("  skip fig_timing (no timing yet)"); return
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, llm, w, label="LLM agents (Gemma 3)", color="#d62728")
    ax.bar(x + w/2, pso, w, label="PSO (rules)", color="#7f7f7f")
    ax.set_ylabel("wall time per lab-evaluation (s)")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_title("Cost of agency: LLM planning+review overhead per lab-evaluation")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_timing")


# ------------------------------------------------------------------ arch diagram
def fig_archdiagram():
    agg = _load(os.path.join(ROOT, "results", "aggregate_v2.json"))
    disc = (agg or {}).get("discovery", {})
    rows = [(pde, disc[pde]["llm"].get("example_arch"))
            for pde in PDES if disc.get(pde) and disc[pde]["llm"].get("example_arch")]
    if not rows:
        print("  skip fig_archdiagram (no discovered arch yet)"); return
    fig, ax = plt.subplots(figsize=(8, 0.7*len(rows)+1))
    maxlen = max(len(a) for _, a in rows)
    for r, (pde, arch) in enumerate(rows):
        y = len(rows)-1-r
        for c, b in enumerate(arch):
            ax.add_patch(plt.Rectangle((c, y-0.35), 0.92, 0.7,
                         color=BCOL.get(b, "#ccc"), alpha=0.9))
            ax.text(c+0.46, y, BABBR.get(b, "?"), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
        ax.text(-0.3, y, PDE_LABEL[pde], ha="right", va="center", fontsize=10)
    ax.set_xlim(-3, maxlen); ax.set_ylim(-0.6, len(rows)-0.4)
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BCOL[b]) for b in BLOCKS]
    ax.legend(handles, [f"{BABBR[b]} = {b}" for b in BLOCKS],
              frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 0.0), fontsize=8)
    ax.set_title("Discovered architecture per PDE (LLM agents): different recipe per regime")
    _save(fig, "fig_archdiagram")


def fig_method():
    """Publication schematic, two panels: (a) the three role agents inside one
    lab for one iteration, with the LLM/PSO ablation as a first-class element;
    (b) the community: peer-review citations + lifecycle across iterations.
    Uses the same block palette as the rest of the figure family."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

    LLM_C, WORK_C, LIFE_C = "#b03a3a", "#2d5f8a", "#3a7d44"
    EDGE, INK, MUT = "#9a9a9a", "#222222", "#666666"

    # Physically compact (8.2in wide) so fonts stay >= ~6pt effective when the
    # figure is scaled to \linewidth in the paper.
    fig, ax = plt.subplots(figsize=(8.2, 3.30))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 5.0); ax.axis("off")

    def card(x, y, w, h, title, body, hc, fs_t=9.5, fs_b=8.6):
        """White card with a colored header bar (squared, inset)."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                     facecolor="white", edgecolor=EDGE, lw=1.0))
        hh = 0.40
        ax.add_patch(plt.Rectangle((x + 0.015, y + h - hh), w - 0.03, hh,
                     facecolor=hc, edgecolor="none"))
        ax.text(x + w/2, y + h - hh/2, title, ha="center", va="center",
                fontsize=fs_t, color="white", fontweight="bold")
        ax.text(x + w/2, y + (h - hh)/2 + 0.02, body, ha="center", va="center",
                fontsize=fs_b, color=INK, linespacing=1.5)

    def arrow(p1, p2, color=MUT, rad=0.0, lw=1.2, ms=11, ls="-"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                     lw=lw, color=color, linestyle=ls,
                     connectionstyle=f"arc3,rad={rad}", zorder=3))

    def chip(x, y, b, s=0.30, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0.015",
                     facecolor=BCOL[b], edgecolor="none", zorder=4))
        ax.text(x + s/2, y + s/2, BABBR[b], ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=5)

    # ---------- title ----------
    ax.text(6.3, 4.74, "The AI Scientific Community for neural operator discovery",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # ================= panel (a): one lab, one iteration =================
    ax.text(0.30, 4.22, "(a) Inside one lab, one iteration",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(4.30, 3.88, "one of $N = 16$ labs", ha="center",
            fontsize=7.4, color=MUT, style="italic")

    cy, ch, cw, gapx = 2.35, 1.30, 2.42, 0.43
    x1, x2, x3 = 0.30, 0.30 + cw + gapx, 0.30 + 2 * (cw + gapx)
    card(x1, cy, cw, ch, "Planner $\\cdot$ LLM",
         "", LLM_C, fs_t=8.4)
    ax.text(x1 + cw/2, cy + 0.66, "proposes genome $g_i$",
            ha="center", va="center", fontsize=8.0, color=INK)
    gxc = x1 + cw/2 - (4 * 0.30 + 3 * 0.05) / 2
    for i, b in enumerate(["fourier", "attention", "wavelet", "branch_trunk"]):
        chip(gxc + i * 0.35, cy + 0.14, b)
    card(x2, cy, cw, ch, "Worker $\\cdot$ numerical",
         "trains (PyTorch);\nmeasures rel. $L_2$", WORK_C, fs_t=8.4, fs_b=8.0)
    card(x3, cy, cw, ch, "Reviewer $\\cdot$ LLM",
         "votes on peer genomes\n$\\to$ citations, trust", LLM_C, fs_t=8.4, fs_b=8.0)

    arrow((x1 + cw, cy + ch/2), (x2, cy + ch/2))
    arrow((x2 + cw, cy + ch/2), (x3, cy + ch/2))
    # feedback arc reviewer -> planner (negative rad = bulge DOWN for a
    # right-to-left arc; positive would cut through the cards)
    arrow((x3 + cw/2, cy - 0.08), (x1 + cw/2, cy - 0.08), rad=-0.22)
    ax.text(4.30, 1.26,
            "votes update trust; history + global best inform the next plan",
            ha="center", fontsize=7.6, color=MUT, style="italic")

    # ablation ribbon under the two LLM roles
    ax.add_patch(FancyBboxPatch((0.55, 0.42), 7.60, 0.45,
                 boxstyle="round,pad=0.03", facecolor="#fdf2f2",
                 edgecolor=LLM_C, lw=0.9))
    ax.text(4.35, 0.65, "ablation: planner + reviewer = LLM (gemma3:12b) "
            "$\\leftrightarrow$ PSO rules",
            ha="center", va="center", fontsize=7.6, color=LLM_C)
    for xx in (x1 + cw/2, x3 + cw/2):        # ties from the two LLM cards
        ax.plot([xx, xx], [0.89, 1.55], color=LLM_C, lw=0.8, ls=(0, (2, 2)),
                zorder=1, alpha=0.55)

    # ================= panel (b): the community =================
    ax.plot([8.75, 8.75], [0.26, 4.40], color="#d9d9d9", lw=1.0)
    ax.text(8.98, 4.22, "(b) The community,  $t = 1 \\dots 20$",
            fontsize=9.5, fontweight="bold", color=INK)

    # 4x4 grid of labs, colored by seed paradigm
    PARA = ["fourier", "branch_trunk", "attention", "wavelet"]
    gx, gy, s, gap = 9.14, 1.50, 0.50, 0.22
    pos = {}
    for k in range(16):
        r, c = divmod(k, 4)
        x = gx + c * (s + gap); y = gy + (3 - r) * (s + gap)
        pos[k] = (x + s/2, y + s/2)
        fc = BCOL[PARA[(r + c) % 4]]
        culled = (k == 13)
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0.02",
                     facecolor=("#dddddd" if culled else fc),
                     edgecolor="none", alpha=(0.45 if culled else 0.80)))
        if culled:
            ax.text(x + s/2, y + s/2, "$\\times$", ha="center", va="center",
                    fontsize=9, color="#888888")
    # global best: star tucked at the top-right corner of lab 6's cell
    bx, by = pos[6]
    ax.text(bx + 0.20, by + 0.21, "$\\bigstar$", fontsize=9, color="#c9a227",
            ha="center", va="center", zorder=6)
    # breeding badge at the same cell's bottom-left corner
    ax.add_patch(Circle((bx - 0.20, by - 0.20), 0.085, facecolor=LIFE_C,
                 edgecolor="white", lw=0.6, zorder=6))
    ax.text(bx - 0.20, by - 0.205, "+", ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=7)
    # citation arrows (peer review votes)
    for a, b_, rd in [(0, 6, 0.25), (11, 6, -0.25), (13, 2, 0.30), (4, 9, -0.25)]:
        arrow(pos[a], pos[b_], color="#8a8a8a", rad=rd, lw=0.85, ms=7)

    lx = gx + 2 * (s + gap) - gap / 2      # legend centered under the grid
    ax.text(lx, 1.10, "arrows = citation votes;  $\\bigstar$ = global best",
            ha="center", fontsize=7.2, color=MUT)
    ax.text(lx, 0.84, "influential labs breed ($+$);  uncited labs",
            ha="center", fontsize=7.2, color=MUT)
    ax.text(lx, 0.58, "are culled ($\\times$) and replaced",
            ha="center", fontsize=7.2, color=MUT)
    ax.text(lx, 0.30, "cell color = seed paradigm",
            ha="center", fontsize=6.9, color=MUT, style="italic")

    _save(fig, "fig_method")


def main():
    os.makedirs(FIG, exist_ok=True)
    for fn in (fig_method, fig_baselines, fig_ablation, fig_convergence,
               fig_pareto, fig_blockevo, fig_timing, fig_archdiagram):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
    print("[figures] done")


if __name__ == "__main__":
    main()
