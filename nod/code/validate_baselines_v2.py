"""EXP-4 — honest baseline validation.

Trains:
  • Pure FNO sweep (hidden ∈ {32, 64, 128} × modes ∈ {8, 12, 16})
  • Faithful DeepONet (branch CNN + trunk MLP over query coords)
  • POD-DeepONet (POD basis trunk)
  • Pure Transformer (3× spatial attention)
  • Discovered hybrid (best from agentic-demo run)

All on the same Navier-Stokes data with the same training budget. Reports
mean ± std over N seeds and saves a JSON summary.

Usage:
    python validate_baselines_v2.py --seeds 42 137 2024
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import math
import random
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from data import NavierStokesGenerator
from baselines import PureFNO, DeepONetFaithful, PODDeepONet
from genome import ArchitectureGenome, ConfigurableNeuralOperator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Discovered hybrid (best from latest agentic-demo)
DISCOVERED_HYBRID = ArchitectureGenome(
    block_sequence=["fourier", "attention", "wavelet", "attention"],
    hidden_channels=64, fourier_modes=10, activation="gelu",
    use_gating=False, use_skip_connections=True, dropout_rate=0.0,
    num_blocks=4, learning_rate=1e-3, weight_decay=1e-4,
)
PURE_TRANSFORMER = ArchitectureGenome(
    block_sequence=["attention"] * 3,
    hidden_channels=64, fourier_modes=8, activation="gelu",
    use_gating=False, use_skip_connections=True, dropout_rate=0.0,
    num_blocks=3, learning_rate=1e-3, weight_decay=1e-4,
)


def train_and_eval(model, train_x, train_y, test_x, test_y, *,
                   epochs: int, batch_size: int, lr: float = 1e-3,
                   wd: float = 1e-4, label: str = "?") -> dict:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = train_x.shape[0]
    model.train()
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = train_x[idx].to(DEVICE)
            by = train_y[idx].to(DEVICE)
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        ty = test_y.to(DEVICE)
        pred = model(test_x.to(DEVICE))
        rel_l2 = (torch.norm(pred - ty) / (torch.norm(ty) + 1e-8)).item()
        noisy = test_x + 0.1 * torch.randn_like(test_x)
        pred_n = model(noisy.to(DEVICE))
        rel_l2_n = (torch.norm(pred_n - ty) / (torch.norm(ty) + 1e-8)).item()

    return {
        "name": label,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "rel_l2_clean": rel_l2,
        "rel_l2_noisy": rel_l2_n,
        "train_time_s": round(elapsed, 1),
    }


def run_one_seed(*, seed: int, train_samples: int, test_samples: int,
                 epochs: int, batch_size: int, resolution: int) -> list[dict]:
    print(f"\n  ── seed {seed} ──")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = NavierStokesGenerator(resolution=resolution, device=DEVICE)
    train_x, train_y = gen.generate(train_samples)
    test_x, test_y = gen.generate(test_samples)
    print(f"  data: train={tuple(train_x.shape)}, test={tuple(test_x.shape)}")

    rows: list[dict] = []
    # Pure FNO sweep
    for h in (32, 64, 128):
        for m in (8, 12, 16):
            torch.manual_seed(seed)
            mdl = PureFNO(in_ch=1, out_ch=1, hidden=h, modes=m, depth=4)
            r = train_and_eval(mdl, train_x, train_y, test_x, test_y,
                               epochs=epochs, batch_size=batch_size,
                               label=f"FNO h{h} m{m}")
            print(f"   {r['name']:>14}  params={r['params']:>9,}  "
                  f"rel L²={r['rel_l2_clean']:.4f} (noisy {r['rel_l2_noisy']:.4f})")
            rows.append(r)

    # Faithful DeepONet
    torch.manual_seed(seed)
    mdl = DeepONetFaithful(in_resolution=resolution, latent=128,
                           branch_hidden=128, trunk_hidden=128, trunk_depth=4)
    r = train_and_eval(mdl, train_x, train_y, test_x, test_y,
                       epochs=epochs, batch_size=batch_size,
                       label="DeepONet faithful")
    print(f"   {r['name']:>20}  params={r['params']:>9,}  "
          f"rel L²={r['rel_l2_clean']:.4f} (noisy {r['rel_l2_noisy']:.4f})")
    rows.append(r)

    # POD-DeepONet
    torch.manual_seed(seed)
    mdl = PODDeepONet(in_resolution=resolution, latent=64, branch_hidden=256)
    mdl.fit_pod_basis(train_y)
    r = train_and_eval(mdl, train_x, train_y, test_x, test_y,
                       epochs=epochs, batch_size=batch_size,
                       label="POD-DeepONet")
    print(f"   {r['name']:>20}  params={r['params']:>9,}  "
          f"rel L²={r['rel_l2_clean']:.4f} (noisy {r['rel_l2_noisy']:.4f})")
    rows.append(r)

    # Pure Transformer
    torch.manual_seed(seed)
    mdl = ConfigurableNeuralOperator(PURE_TRANSFORMER)
    r = train_and_eval(mdl, train_x, train_y, test_x, test_y,
                       epochs=epochs, batch_size=batch_size,
                       label="Pure Transformer")
    print(f"   {r['name']:>20}  params={r['params']:>9,}  "
          f"rel L²={r['rel_l2_clean']:.4f} (noisy {r['rel_l2_noisy']:.4f})")
    rows.append(r)

    # Discovered hybrid
    torch.manual_seed(seed)
    mdl = ConfigurableNeuralOperator(DISCOVERED_HYBRID)
    r = train_and_eval(mdl, train_x, train_y, test_x, test_y,
                       epochs=epochs, batch_size=batch_size,
                       label="Discovered Hybrid")
    print(f"   {r['name']:>20}  params={r['params']:>9,}  "
          f"rel L²={r['rel_l2_clean']:.4f} (noisy {r['rel_l2_noisy']:.4f})")
    rows.append(r)

    for r in rows:
        r["seed"] = seed
    return rows


def aggregate(all_rows: list[dict]) -> list[dict]:
    by_name = defaultdict(list)
    for r in all_rows:
        by_name[r["name"]].append(r)
    out = []
    for name, rs in by_name.items():
        l2 = [x["rel_l2_clean"] for x in rs]
        l2n = [x["rel_l2_noisy"] for x in rs]
        params = rs[0]["params"]
        out.append({
            "name": name,
            "params": params,
            "rel_l2_clean_mean": round(mean(l2), 4),
            "rel_l2_clean_std":  round(stdev(l2), 4) if len(l2) > 1 else 0.0,
            "rel_l2_noisy_mean": round(mean(l2n), 4),
            "rel_l2_noisy_std":  round(stdev(l2n), 4) if len(l2n) > 1 else 0.0,
            "n_seeds": len(rs),
            "values_clean": [round(v, 4) for v in l2],
            "values_noisy": [round(v, 4) for v in l2n],
        })
    out.sort(key=lambda r: r["rel_l2_clean_mean"])
    return out


def print_table(rows: list[dict]):
    print("\n" + "=" * 92)
    print(f"  {'Architecture':<28} {'params':>10}  {'rel L² clean (μ ± σ)':>22}  {'rel L² noisy (μ ± σ)':>22}")
    print("-" * 92)
    for r in rows:
        c = f"{r['rel_l2_clean_mean']:.4f} ± {r['rel_l2_clean_std']:.4f}"
        n = f"{r['rel_l2_noisy_mean']:.4f} ± {r['rel_l2_noisy_std']:.4f}"
        print(f"  {r['name']:<28} {r['params']:>10,}  {c:>22}  {n:>22}")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--test-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resolution", type=int, default=32)
    args = p.parse_args()

    print("=" * 92)
    print("  EXP-4 — HONEST BASELINE VALIDATION  (faithful DeepONet, POD-DeepONet, FNO sweep)")
    print("=" * 92)
    print(f"  Device: {DEVICE}  |  seeds: {args.seeds}")
    print(f"  Train/test/epochs/batch: {args.samples}/{args.test_samples}/"
          f"{args.epochs}/{args.batch_size}")

    all_rows: list[dict] = []
    for s in args.seeds:
        all_rows.extend(run_one_seed(
            seed=s, train_samples=args.samples,
            test_samples=args.test_samples, epochs=args.epochs,
            batch_size=args.batch_size, resolution=args.resolution))

    agg = aggregate(all_rows)
    print_table(agg)

    out = {
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs, "samples": args.samples,
            "test_samples": args.test_samples, "batch_size": args.batch_size,
            "resolution": args.resolution,
        },
        "raw": all_rows,
        "aggregate": agg,
    }
    out_path = os.path.join(RESULTS, "exp4_baselines.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
