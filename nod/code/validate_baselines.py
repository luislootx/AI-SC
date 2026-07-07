"""Honest validation: train classical baselines + the discovered hybrid on
identical data with identical training budget, then compare relative L2.

This answers the question: did the swarm's 'global best' actually beat the
seed paradigms, or did the multi-objective composite reward something
trivial (like a tiny model gaming efficiency)?

Usage:
    python validate_baselines.py
    python validate_baselines.py --epochs 15 --samples 128   # heavier run
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import SwarmConfig
from data import NavierStokesGenerator
from genome import ArchitectureGenome, ConfigurableNeuralOperator
from fitness import evaluate_model


# Architecture under test: agentic-smoke run global-best (legacy)
DISCOVERED_HYBRID_SMOKE = ArchitectureGenome(
    block_sequence=["wavelet", "attention", "residual_conv"],
    hidden_channels=64,
    fourier_modes=10,
    activation="gelu",
    use_gating=False,
    use_skip_connections=True,
    dropout_rate=0.2,
    num_blocks=3,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# Agentic-demo run global-best (Apr 28, with re-tuned fitness)
DISCOVERED_HYBRID_DEMO = ArchitectureGenome(
    block_sequence=["fourier", "attention", "wavelet", "attention"],
    hidden_channels=64,
    fourier_modes=10,
    activation="gelu",
    use_gating=False,
    use_skip_connections=True,
    dropout_rate=0.0,
    num_blocks=4,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# Classical baselines
PURE_FNO = ArchitectureGenome(
    block_sequence=["fourier"] * 4,
    hidden_channels=64,
    fourier_modes=12,
    activation="gelu",
    use_gating=False,
    use_skip_connections=True,
    dropout_rate=0.0,
    num_blocks=4,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

PURE_DEEPONET = ArchitectureGenome(
    block_sequence=["branch_trunk"] * 4,
    hidden_channels=64,
    fourier_modes=8,
    activation="relu",
    use_gating=False,
    use_skip_connections=False,
    dropout_rate=0.0,
    num_blocks=4,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

PURE_TRANSFORMER = ArchitectureGenome(
    block_sequence=["attention"] * 3,
    hidden_channels=64,
    fourier_modes=8,
    activation="gelu",
    use_gating=False,
    use_skip_connections=True,
    dropout_rate=0.0,
    num_blocks=3,
    learning_rate=1e-3,
    weight_decay=1e-4,
)


def run_single(name: str, genome: ArchitectureGenome, train_x, train_y,
               test_x, test_y, config: SwarmConfig) -> dict:
    print(f"\n──── {name} ────")
    print(f"  Blocks: {' → '.join(genome.block_sequence)}  "
          f"(ch={genome.hidden_channels}, modes={genome.fourier_modes})")
    model = ConfigurableNeuralOperator(genome).to(config.device)
    n_params = model.count_parameters()
    print(f"  Params: {n_params:,}")
    t0 = time.time()
    fit = evaluate_model(model, train_x, train_y, test_x, test_y, config)
    dt = time.time() - t0
    rel_l2 = 1.0 - fit.accuracy
    rel_l2_noisy = 1.0 - fit.generalization
    print(f"  Train+eval time: {dt:.1f} s")
    print(f"  rel L2 (clean): {rel_l2:.4f}   acc={fit.accuracy:.4f}")
    print(f"  rel L2 (noisy): {rel_l2_noisy:.4f}   gen={fit.generalization:.4f}")
    print(f"  Efficiency:    {fit.efficiency:.4f}   (param_score+time_score / 2)")
    return {
        "name": name,
        "blocks": list(genome.block_sequence),
        "params": n_params,
        "rel_l2_clean": rel_l2,
        "rel_l2_noisy": rel_l2_noisy,
        "accuracy": fit.accuracy,
        "generalization": fit.generalization,
        "efficiency": fit.efficiency,
        "time_seconds": dt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5,
                   help="train epochs per architecture (default 5, matches agentic-smoke)")
    p.add_argument("--samples", type=int, default=64,
                   help="num train samples (default 64, matches agentic-smoke)")
    p.add_argument("--test-samples", type=int, default=16)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = SwarmConfig(
        resolution=args.resolution,
        num_train_samples=args.samples,
        num_test_samples=args.test_samples,
        train_epochs_per_iteration=args.epochs,
        batch_size=8,
    )

    print("=" * 78)
    print("  HONEST BASELINE VALIDATION")
    print("=" * 78)
    print(f"  Device: {config.device}")
    print(f"  Train/test/epochs: {args.samples}/{args.test_samples}/{args.epochs}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Seed: {args.seed}  (data is regenerated identically across archs)")

    print("\n  Generating Navier-Stokes data...")
    t0 = time.time()
    gen = NavierStokesGenerator(resolution=config.resolution, device=config.device)
    train_x, train_y = gen.generate(config.num_train_samples)
    test_x, test_y = gen.generate(config.num_test_samples)
    print(f"  Done in {time.time() - t0:.1f}s. "
          f"Train: {tuple(train_x.shape)}, Test: {tuple(test_x.shape)}")

    archs = [
        ("Pure FNO (4×fourier)", PURE_FNO),
        ("Pure DeepONet (4×branch_trunk)", PURE_DEEPONET),
        ("Pure Transformer (3×attention)", PURE_TRANSFORMER),
        ("Smoke Hybrid (wavelet→attention→residual_conv)", DISCOVERED_HYBRID_SMOKE),
        ("Demo Hybrid (fourier→attention→wavelet→attention)", DISCOVERED_HYBRID_DEMO),
    ]

    results = []
    for name, g in archs:
        # Re-seed for fair training across architectures
        torch.manual_seed(args.seed)
        results.append(run_single(name, g, train_x, train_y, test_x, test_y, config))

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"\n  {'Architecture':<55} {'params':>10} {'rel L2':>10} {'noisy':>10}")
    print("  " + "-" * 86)
    results.sort(key=lambda r: r["rel_l2_clean"])
    for r in results:
        print(f"  {r['name']:<55} {r['params']:>10,} "
              f"{r['rel_l2_clean']:>10.4f} {r['rel_l2_noisy']:>10.4f}")

    best = results[0]
    print(f"\n  → Lowest clean rel L2: {best['name']}  ({best['rel_l2_clean']:.4f})")
    fno = next(r for r in results if r["name"].startswith("Pure FNO"))
    for r in results:
        if "Hybrid" not in r["name"]:
            continue
        delta = fno["rel_l2_clean"] - r["rel_l2_clean"]
        ratio = r["rel_l2_clean"] / fno["rel_l2_clean"] if fno["rel_l2_clean"] > 0 else float("inf")
        if delta > 0:
            print(f"  → {r['name']} beats FNO by {delta:.4f} L2")
        else:
            print(f"  → {r['name']} loses to FNO by {-delta:.4f} L2 ({ratio:.1f}× gap)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "validation_baselines.json")
    with open(out, "w") as f:
        json.dump({
            "config": {
                "epochs": args.epochs, "samples": args.samples,
                "resolution": args.resolution, "seed": args.seed,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {out}\n")


if __name__ == "__main__":
    main()
