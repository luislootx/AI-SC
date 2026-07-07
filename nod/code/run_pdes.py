"""EXP-3 driver — multi-PDE generalization.

Re-runs the agentic-demo swarm on each PDE in {ns, darcy, heat} for the same
seeds, then reports whether the discovered architectures differ across physics.

Usage:
    python run_pdes.py --pdes ns darcy heat --seeds 42 137 2024 --tag exp3
    python run_pdes.py --skip-runs --tag exp3       # only aggregate

Per-run JSON lands at:
    results/results_agentic-demo_exp3_{pde}_seed{N}.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
os.makedirs(LOGS, exist_ok=True)

PYTHON = sys.executable
MODE = "agentic-demo"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_one(pde: str, seed: int, tag: str) -> tuple[int, str]:
    sub_tag = f"{tag}_{pde}_seed{seed}"
    log_name = f"seed_{sub_tag}.log"
    log_path = os.path.join(LOGS, log_name)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [PYTHON, "-u", "main.py", MODE, "--seed", str(seed),
           "--pde", pde, "--tag", sub_tag]
    print(f"[{ts()}] ▶ {pde} seed={seed}  cmd: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=HERE, env=env, stdout=f,
                              stderr=subprocess.STDOUT, text=True)
    dt = (time.time() - t0) / 60
    print(f"[{ts()}] ◀ {pde} seed={seed}  rc={proc.returncode}  {dt:.1f} min")
    return proc.returncode, log_path


def collect(pde: str, seeds: list[int], tag: str) -> list[dict]:
    out = []
    for seed in seeds:
        path = os.path.join(RESULTS,
                            f"results_{MODE}_{tag}_{pde}_seed{seed}.json")
        if not os.path.exists(path):
            print(f"  ⚠ missing: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def winner_stats(results: list[dict]) -> dict:
    if not results:
        return {}
    composite = []
    rel_l2_clean = []
    rel_l2_noisy = []
    block_pcts: list[dict] = []
    winner_blocks: list[list[str]] = []
    winner_paradigm: list[str] = []
    for r in results:
        composite.append(r["global_best"]["fitness"])
        hist = r.get("iteration_history") or []
        if not hist:
            continue
        last = hist[-1]
        best_id = r["global_best"]["lab_id"]
        counts: dict[str, int] = defaultdict(int)
        total = 0
        best_lab = None
        for lab in last["lab_summaries"]:
            for b in lab.get("blocks", []):
                counts[b] += 1
                total += 1
            if lab["lab_id"] == best_id:
                best_lab = lab
        block_pcts.append({b: 100 * c / max(total, 1) for b, c in counts.items()})
        if best_lab:
            rel_l2_clean.append(best_lab.get("rel_l2_clean", -1))
            rel_l2_noisy.append(best_lab.get("rel_l2_noisy", -1))
            winner_blocks.append(best_lab.get("blocks", []))
            winner_paradigm.append(best_lab.get("paradigm", "?"))

    def stat(vs):
        clean = [v for v in vs if v is not None and v >= 0]
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(mean(clean), 4),
            "std": round(stdev(clean), 4) if len(clean) > 1 else 0.0,
            "n": len(clean),
            "values": [round(v, 4) for v in clean],
        }

    all_blocks = set()
    for d in block_pcts:
        all_blocks.update(d.keys())
    block_stats = {b: stat([d.get(b, 0.0) for d in block_pcts])
                   for b in sorted(all_blocks)}
    return {
        "n_seeds": len(results),
        "composite": stat(composite),
        "rel_l2_clean": stat(rel_l2_clean),
        "rel_l2_noisy": stat(rel_l2_noisy),
        "final_block_usage_pct": block_stats,
        "winner_paradigms": winner_paradigm,
        "winner_blocks": winner_blocks,
    }


def report(per_pde: dict[str, dict]):
    print("\n" + "=" * 92)
    print("  EXP-3 — MULTI-PDE GENERALIZATION")
    print("=" * 92)
    print(f"\n  {'PDE':<8} {'comp μ ± σ':>14} {'rel L² clean':>16} "
          f"{'rel L² noisy':>16}  winners")
    for pde, agg in per_pde.items():
        if not agg:
            print(f"  {pde:<8}  (no data)")
            continue
        c = f"{agg['composite']['mean']:.4f} ± {agg['composite']['std']:.4f}"
        l2 = (f"{agg['rel_l2_clean']['mean']:.4f} ± {agg['rel_l2_clean']['std']:.4f}"
              if agg['rel_l2_clean']['n'] else "n/a")
        l2n = (f"{agg['rel_l2_noisy']['mean']:.4f} ± {agg['rel_l2_noisy']['std']:.4f}"
               if agg['rel_l2_noisy']['n'] else "n/a")
        winners = "; ".join(["+".join(b) for b in agg['winner_blocks']])
        print(f"  {pde:<8} {c:>14} {l2:>16} {l2n:>16}  {winners}")
    print("\n  Block usage % (μ across seeds):")
    blocks = set()
    for agg in per_pde.values():
        if agg:
            blocks.update(agg.get("final_block_usage_pct", {}).keys())
    for b in sorted(blocks):
        row = "    " + f"{b:18s}"
        for pde, agg in per_pde.items():
            if not agg:
                continue
            stat = agg.get("final_block_usage_pct", {}).get(b, {"mean": 0})
            row += f" {pde:>5}={stat['mean']:5.1f} %"
        print(row)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdes", type=str, nargs="+",
                   default=["ns", "darcy", "heat"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    p.add_argument("--tag", type=str, default="exp3")
    p.add_argument("--skip-runs", action="store_true",
                   help="only aggregate already-saved results")
    args = p.parse_args()

    print(f"[{ts()}] EXP-3 multi-PDE: pdes={args.pdes} seeds={args.seeds} tag={args.tag}")

    if not args.skip_runs:
        for pde in args.pdes:
            for seed in args.seeds:
                rc, _ = run_one(pde, seed, args.tag)
                if rc != 0:
                    print(f"  ⚠ {pde} seed {seed} returned rc={rc}")

    per_pde = {pde: winner_stats(collect(pde, args.seeds, args.tag))
               for pde in args.pdes}

    report(per_pde)
    out_path = os.path.join(RESULTS, f"aggregate_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_pde, f, indent=2, default=str)
    print(f"  Saved aggregate: {out_path}")


if __name__ == "__main__":
    main()
