"""1D analogue of validate_baselines.py.

Trains classical baselines + the swarm's discovered hybrid on the three 1D
benchmarks (piecewise regression, linear advection, Burgers), with
identical data and training budget per architecture, and reports honest
relative-L2 comparisons.

Discovered-hybrid genomes are loaded from `results/swarm_runs/<tag>/FINAL.json`.

Usage:
    python code/validate_baselines_1d.py --pde pwreg
    python code/validate_baselines_1d.py --pde advec   --epochs 30 --samples 256
    python code/validate_baselines_1d.py --pde burgers --epochs 40

Writes results to `results/validation_baselines_1d_<pde>.json`.
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
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from config import SwarmConfig
from genome import ArchitectureGenome
from genome_1d import ConfigurableNeuralOperator1D
from fitness import evaluate_model
from pdes_1d import make_generator_1d


# --- 1D pure baselines (matched to validate_baselines.py philosophy) ----

PURE_FNO_1D = ArchitectureGenome(
    block_sequence=["fourier"] * 4,
    hidden_channels=64, fourier_modes=12, activation="gelu",
    use_gating=False, use_skip_connections=True, dropout_rate=0.0,
    num_blocks=4, learning_rate=1e-3, weight_decay=1e-4,
)
PURE_DEEPONET_1D = ArchitectureGenome(
    block_sequence=["branch_trunk"] * 4,
    hidden_channels=64, fourier_modes=8, activation="relu",
    use_gating=False, use_skip_connections=False, dropout_rate=0.0,
    num_blocks=4, learning_rate=1e-3, weight_decay=1e-4,
)
PURE_TRANSFORMER_1D = ArchitectureGenome(
    block_sequence=["attention"] * 3,
    hidden_channels=64, fourier_modes=8, activation="gelu",
    use_gating=False, use_skip_connections=True, dropout_rate=0.0,
    num_blocks=3, learning_rate=1e-3, weight_decay=1e-4,
)
PURE_WAVELET_1D = ArchitectureGenome(
    block_sequence=["wavelet"] * 4,
    hidden_channels=48, fourier_modes=8, activation="silu",
    use_gating=True, use_skip_connections=True, dropout_rate=0.0,
    num_blocks=4, learning_rate=1e-3, weight_decay=1e-4,
)


def load_discovered_hybrid(tag: str) -> ArchitectureGenome | None:
    p = os.path.join(ROOT, "results", "swarm_runs", tag, "FINAL.json")
    if not os.path.exists(p):
        return None
    final = json.load(open(p))
    g = final.get("global_best_genome")
    if not g:
        return None
    return ArchitectureGenome(**g)


def run_single(name: str, genome: ArchitectureGenome, train_x, train_y,
               test_x, test_y, config: SwarmConfig) -> dict:
    print(f"\n──── {name} ────")
    print(f"  Blocks: {' → '.join(genome.block_sequence)}  "
          f"(ch={genome.hidden_channels}, modes={genome.fourier_modes})")
    model = ConfigurableNeuralOperator1D(genome).to(config.device)
    n_params = model.count_parameters()
    print(f"  Params: {n_params:,}")
    t0 = time.time()
    fit = evaluate_model(model, train_x, train_y, test_x, test_y, config)
    dt = time.time() - t0
    print(f"  Train+eval time: {dt:.1f} s")
    print(f"  rel L2 (clean): {fit.rel_l2_clean:.4f}")
    print(f"  rel L2 (noisy): {fit.rel_l2_noisy:.4f}")
    print(f"  Efficiency:     {fit.efficiency:.4f}")
    return {
        "name": name,
        "blocks": list(genome.block_sequence),
        "params": n_params,
        "rel_l2_clean": fit.rel_l2_clean,
        "rel_l2_noisy": fit.rel_l2_noisy,
        "efficiency": fit.efficiency,
        "time_seconds": dt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pde", required=True,
                   choices=["pwreg", "advec", "burgers"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--test-samples", type=int, default=64)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hybrid-tag", default=None,
                   help="run tag whose FINAL.json supplies the discovered hybrid genome "
                        "(default: paper_<pde>_seed42)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = SwarmConfig(
        resolution=args.resolution,
        num_train_samples=args.samples,
        num_test_samples=args.test_samples,
        train_epochs_per_iteration=args.epochs,
        batch_size=16,
        unbounded_fitness=True,
        accuracy_floor=0.0,
    )

    print("=" * 78)
    print(f"  HONEST 1D BASELINE VALIDATION  pde={args.pde}")
    print("=" * 78)
    print(f"  Device: {cfg.device}")
    print(f"  Train/test/epochs: {args.samples}/{args.test_samples}/{args.epochs}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Seed: {args.seed}")

    print(f"\n  Generating {args.pde} data...")
    t0 = time.time()
    gen = make_generator_1d(args.pde, resolution=args.resolution)
    train_x, train_y = gen.generate(args.samples)
    test_x, test_y = gen.generate(args.test_samples)
    print(f"  Done in {time.time() - t0:.1f}s. "
          f"Train: {tuple(train_x.shape)}, Test: {tuple(test_x.shape)}")

    hybrid_tag = args.hybrid_tag or f"paper_{args.pde}_seed42"
    hybrid = load_discovered_hybrid(hybrid_tag)

    archs = [
        ("Pure FNO (4xfourier)", PURE_FNO_1D),
        ("Pure DeepONet (4xbranch_trunk)", PURE_DEEPONET_1D),
        ("Pure Transformer (3xattention)", PURE_TRANSFORMER_1D),
        ("Pure Wavelet (4xwavelet)", PURE_WAVELET_1D),
    ]
    if hybrid is not None:
        archs.append((f"Discovered Hybrid ({hybrid_tag})", hybrid))
    else:
        print(f"\n  WARN: no FINAL.json found for {hybrid_tag}; "
              f"running baselines only.")

    results = []
    for name, g in archs:
        torch.manual_seed(args.seed)
        results.append(run_single(name, g, train_x, train_y,
                                  test_x, test_y, cfg))

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    results.sort(key=lambda r: r["rel_l2_clean"])
    print(f"\n  {'Architecture':<55} {'params':>10} {'rel L2':>10} {'noisy':>10}")
    print("  " + "-" * 86)
    for r in results:
        print(f"  {r['name']:<55} {r['params']:>10,} "
              f"{r['rel_l2_clean']:>10.4f} {r['rel_l2_noisy']:>10.4f}")
    print(f"\n  -> Best: {results[0]['name']}")

    out = os.path.join(ROOT, "results",
                        f"validation_baselines_1d_{args.pde}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "pde": args.pde,
            "config": {"epochs": args.epochs, "samples": args.samples,
                       "resolution": args.resolution, "seed": args.seed,
                       "hybrid_tag": hybrid_tag},
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {out}\n")


if __name__ == "__main__":
    main()
