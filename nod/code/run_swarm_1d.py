"""1D variant of `run_swarm_resumable.py`.

Runs the AI Scientific Community swarm against the 1D benchmarks committed
to in the ICERM abstract: piecewise-regression, linear advection, Burgers.
Re-uses all checkpoint / resume / per-iter snapshot logic from the 2D
runner — only the community class and PDE choices differ.

Usage:
  PY="C:/Users/luisl/anaconda3/envs/jax-env-3.11/python.exe"  # or `python`
  $PY code/run_swarm_1d.py paper-grade-1d --pde pwreg   --tag paper_pwreg_seed42  --seed 42
  $PY code/run_swarm_1d.py paper-grade-1d --pde advec   --tag paper_advec_seed42  --seed 42
  $PY code/run_swarm_1d.py paper-grade-1d --pde burgers --tag paper_burgers_seed42 --seed 42

Re-running the same command after a crash / reboot picks up where it left
off (≤1 iteration of work lost, identical to the 2D path).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from config import SwarmConfig, small_smoke_config
from community_1d import AIScientificCommunity1D
from run_swarm_resumable import run_resumable


# ----------------------------------------------------------------- presets

def smoke_1d_config() -> SwarmConfig:
    """Tiny end-to-end sanity check (~1 min on CPU)."""
    return SwarmConfig(
        num_labs=4,
        num_iterations=2,
        resolution=64,
        num_train_samples=32,
        num_test_samples=8,
        train_epochs_per_iteration=3,
        batch_size=8,
        use_llm_planner=False,
        use_llm_reviewer=False,
        unbounded_fitness=True,
        accuracy_floor=0.0,
    )


def medium_1d_config() -> SwarmConfig:
    """Mid-size 1D run (~10 min on RTX 4080)."""
    return SwarmConfig(
        num_labs=8,
        num_iterations=10,
        resolution=128,
        num_train_samples=256,
        num_test_samples=64,
        train_epochs_per_iteration=20,
        batch_size=16,
        use_llm_planner=False,
        use_llm_reviewer=False,
        unbounded_fitness=True,
        accuracy_floor=0.0,
    )


def paper_grade_1d_config() -> SwarmConfig:
    """Paper-grade 1D run (~30-45 min per problem on RTX 4080).

    Same swarm size + iter count as 2D paper-grade, but trained on 1D
    (256 train / 64 test) so each candidate finishes much faster. Agentic
    layer OFF for reproducibility (the 1D demo aims to be a rule-based
    swarm result; LLM-augmented runs are a separate experiment).
    """
    return SwarmConfig(
        num_labs=16,
        num_iterations=20,
        resolution=128,
        num_train_samples=256,
        num_test_samples=64,
        train_epochs_per_iteration=40,
        batch_size=16,
        initial_exploration_rate=0.85,
        min_exploration_rate=0.15,
        exploration_decay=0.95,
        use_llm_planner=False,
        use_llm_reviewer=False,
        unbounded_fitness=True,
        accuracy_floor=0.0,
    )


PRESETS_1D = {
    "smoke-1d":        smoke_1d_config,
    "medium-1d":       medium_1d_config,
    "paper-grade-1d":  paper_grade_1d_config,
}


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=list(PRESETS_1D.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pde", required=True,
                   choices=["pwreg", "advec", "burgers"])
    p.add_argument("--tag", required=True,
                   help="run tag (= subdirectory under results/swarm_runs/)")
    args = p.parse_args()

    cfg = PRESETS_1D[args.preset]()
    cfg.pde_name = args.pde

    run_dir = os.path.join(ROOT, "results", "swarm_runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    data_path = os.path.join(run_dir, "data.pt")
    cfg_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "preset": args.preset, "seed": args.seed, "pde": args.pde,
                "spatial_dim": 1,
                "num_labs": cfg.num_labs, "num_iterations": cfg.num_iterations,
                "resolution": cfg.resolution,
                "num_train_samples": cfg.num_train_samples,
                "num_test_samples":  cfg.num_test_samples,
                "epochs_per_iter":   cfg.train_epochs_per_iteration,
                "use_llm_planner":   cfg.use_llm_planner,
                "use_llm_reviewer":  cfg.use_llm_reviewer,
                "llm_backend":       cfg.llm_backend,
                "started_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    print("╔" + "═" * 78 + "╗")
    print("║" + f"  1D SWARM (resume-safe)  preset={args.preset}".center(78) + "║")
    print("║" + f"  pde={args.pde}  tag={args.tag}  seed={args.seed}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    agentic = "ON" if (cfg.use_llm_planner or cfg.use_llm_reviewer) else "OFF"
    print(f"  Config: {cfg.num_labs} labs x {cfg.num_iterations} iter, "
          f"res={cfg.resolution}, train={cfg.num_train_samples}, "
          f"epochs/iter={cfg.train_epochs_per_iteration}, device={cfg.device}")
    print(f"  Agentic: {agentic}")
    print(f"  Run dir: {run_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    community = AIScientificCommunity1D(cfg)
    rc = run_resumable(community, run_dir=run_dir, data_path=data_path)
    if rc != 0:
        print(f"\n  exited with rc={rc}")
        sys.exit(rc)

    final = {
        "preset": args.preset, "seed": args.seed, "pde": args.pde,
        "spatial_dim": 1,
        "global_best_fitness": community.registry.global_best_fitness,
        "global_best_lab_id":  community.registry.global_best_lab_id,
        "global_best_genome":  (community.registry.global_best_genome.to_dict()
                                if community.registry.global_best_genome else None),
        "iteration_history":   community.registry.iteration_history,
        "completed_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "FINAL.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  ✓ FINAL.json written. Done.")


if __name__ == "__main__":
    main()
