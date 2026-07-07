"""Chain NS then Darcy paper-grade swarm runs, both resume-safe.

If NS is already complete (FINAL.json present), skips to Darcy.
If Darcy is already complete, exits.
If interrupted by PC reboot, re-running this script picks up exactly where
it left off (each underlying run_swarm_resumable.py call is resumable).

Usage:
  python code/chain_ns_then_darcy.py [--seed 42]
"""
from __future__ import annotations
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(pde: str, tag: str, seed: int) -> int:
    final = os.path.join(ROOT, "results", "swarm_runs", tag, "FINAL.json")
    if os.path.exists(final):
        print(f"[chain] {tag} already complete, skipping.", flush=True)
        return 0
    cmd = [sys.executable, os.path.join("code", "run_swarm_resumable.py"),
           "paper-grade", "--pde", pde, "--tag", tag, "--seed", str(seed)]
    print(f"[chain] launching: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rc = run("ns", f"paper_ns_seed{args.seed}", args.seed)
    if rc != 0:
        print(f"[chain] NS failed with rc={rc}, aborting before Darcy.", flush=True)
        sys.exit(rc)

    rc = run("darcy", f"paper_darcy_seed{args.seed}", args.seed)
    sys.exit(rc)
