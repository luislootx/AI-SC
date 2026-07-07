"""Multi-seed wrapper. Runs `main.py <mode>` for each seed and aggregates the
final global_best + lab summaries with mean ± std.

Usage:
    python run_seeds.py agentic-demo --seeds 42 137 2024 --tag exp1
    python run_seeds.py large-demo   --seeds 42 137 2024 --tag exp2
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


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_one(mode: str, seed: int, tag: str) -> tuple[int, str]:
    log_name = f"seed_{tag}_{seed}.log"
    log_path = os.path.join(LOGS, log_name)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [PYTHON, "-u", "main.py", mode, "--seed", str(seed),
           "--tag", f"{tag}_seed{seed}"]
    print(f"[{ts()}] ▶ seed={seed}  cmd: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=HERE, env=env, stdout=f,
                              stderr=subprocess.STDOUT, text=True)
    dt = (time.time() - t0) / 60
    print(f"[{ts()}] ◀ seed={seed}  rc={proc.returncode}  {dt:.1f} min")
    return proc.returncode, log_path


def collect_results(mode: str, tag: str, seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        path = os.path.join(RESULTS, f"results_{mode}_{tag}_seed{seed}.json")
        if not os.path.exists(path):
            print(f"  ⚠ missing: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    # Final-iteration block usage % across all labs (per seed)
    final_block_pct_per_seed = []
    composite_per_seed = []
    rel_l2_clean_per_seed = []
    rel_l2_noisy_per_seed = []
    diversity_per_seed = []
    winner_paradigm_per_seed = []
    winner_blocks_per_seed = []

    for r in results:
        hist = r.get("iteration_history") or []
        if not hist:
            continue
        last = hist[-1]
        # Block usage
        counts: dict[str, int] = defaultdict(int)
        total = 0
        best_lab_id = r["global_best"]["lab_id"]
        best_lab_summary = None
        for lab in last["lab_summaries"]:
            for b in lab.get("blocks", []):
                counts[b] += 1
                total += 1
            if lab["lab_id"] == best_lab_id:
                best_lab_summary = lab
        pct = {b: 100 * c / max(total, 1) for b, c in counts.items()}
        final_block_pct_per_seed.append(pct)
        composite_per_seed.append(r["global_best"]["fitness"])
        if best_lab_summary:
            rel_l2_clean_per_seed.append(best_lab_summary.get("rel_l2_clean", -1))
            rel_l2_noisy_per_seed.append(best_lab_summary.get("rel_l2_noisy", -1))
            winner_paradigm_per_seed.append(best_lab_summary.get("paradigm"))
            winner_blocks_per_seed.append(best_lab_summary.get("blocks"))

    def stat(values):
        clean = [v for v in values if v is not None and v >= 0]
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(mean(clean), 4),
            "std": round(stdev(clean), 4) if len(clean) > 1 else 0.0,
            "n": len(clean),
            "values": [round(v, 4) for v in clean],
        }

    # Aggregate block percentages per type
    all_blocks = set()
    for d in final_block_pct_per_seed:
        all_blocks.update(d.keys())
    block_stats = {}
    for b in sorted(all_blocks):
        vals = [d.get(b, 0.0) for d in final_block_pct_per_seed]
        block_stats[b] = stat(vals)

    return {
        "n_seeds": len(results),
        "seeds": [r.get("seed") for r in results],
        "composite_global_best":  stat(composite_per_seed),
        "rel_l2_clean_global_best": stat(rel_l2_clean_per_seed),
        "rel_l2_noisy_global_best": stat(rel_l2_noisy_per_seed),
        "final_block_usage_pct":  block_stats,
        "winner_paradigms":       winner_paradigm_per_seed,
        "winner_block_sequences": winner_blocks_per_seed,
    }


def print_report(name: str, agg: dict):
    print("\n" + "=" * 78)
    print(f"  AGGREGATE — {name}  ({agg['n_seeds']} seeds: {agg['seeds']})")
    print("=" * 78)

    def show(label, s):
        if s["n"] == 0:
            print(f"  {label:30s} (no data)")
            return
        sd = f" ± {s['std']:.4f}" if s["n"] > 1 else ""
        print(f"  {label:30s} {s['mean']:>8.4f}{sd}   "
              f"values={s['values']}")
    show("composite (global best)",   agg["composite_global_best"])
    show("rel L² clean (global best)", agg["rel_l2_clean_global_best"])
    show("rel L² noisy (global best)", agg["rel_l2_noisy_global_best"])

    print("\n  Final block usage % (μ ± σ across seeds):")
    for b, s in sorted(agg["final_block_usage_pct"].items(),
                       key=lambda kv: -kv[1]["mean"]):
        sd = f" ± {s['std']:5.1f}" if s["n"] > 1 else ""
        print(f"    {b:18s} {s['mean']:6.1f} %{sd}")

    print("\n  Winners per seed:")
    for paradigm, blocks in zip(agg["winner_paradigms"], agg["winner_block_sequences"]):
        print(f"    seed → {paradigm:18s} :: [{'+'.join(blocks)}]")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", help="agentic-smoke / agentic-demo / large-demo / ...")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    p.add_argument("--tag", type=str, required=True,
                   help="experiment tag (e.g. exp1, exp2)")
    p.add_argument("--skip-runs", action="store_true",
                   help="only aggregate already-saved results")
    args = p.parse_args()

    print(f"[{ts()}] Multi-seed run: mode={args.mode}  seeds={args.seeds}  tag={args.tag}")

    if not args.skip_runs:
        for seed in args.seeds:
            rc, _ = run_one(args.mode, seed, args.tag)
            if rc != 0:
                print(f"  ⚠ seed {seed} returned rc={rc}")

    results = collect_results(args.mode, args.tag, args.seeds)
    agg = aggregate(results)

    name = f"{args.mode}_{args.tag}"
    print_report(name, agg)

    out_path = os.path.join(RESULTS, f"aggregate_{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"  Saved aggregate: {out_path}")


if __name__ == "__main__":
    main()
