"""DIAGNOSE — why is DeepONet faithful failing at rel L2 = 0.95?

Hypotheses (in priority order):
  H1. Training budget too small (15 epochs / 256 samples). DeepONet is data
      and step hungry — POD-DeepONet works because POD basis is fixed/free.
  H2. No I/O normalization. DeepONet sensitive to scale.
  H3. Tanh trunk saturates / trains slowly.
  H4. Latent / trunk too small for 1024-pixel NS vorticity output.
  H5. Branch CNN architecture has the AdaptiveAvgPool2d(1) bottleneck.

We test by sweeping: (epochs ∈ {15, 100, 400}) × (norm ∈ {off, on}) × (act
∈ {tanh, gelu}) × (latent ∈ {128, 256}). Single seed. Print compact table.

Run:
    "C:/Users/luisl/anaconda3/envs/jax-env-3.11/python.exe" code/diagnose_deeponet.py
"""
from __future__ import annotations
import os, sys, time, math, json, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from data import NavierStokesGenerator
from baselines import DeepONetFaithful

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TRAIN_N = 512   # bumped up (was 256)
TEST_N = 64
RES = 32


def make_deeponet(latent: int, trunk_act: str) -> DeepONetFaithful:
    return DeepONetFaithful(
        in_resolution=RES, latent=latent,
        branch_hidden=max(128, latent), trunk_hidden=max(128, latent),
        trunk_depth=4, trunk_activation=trunk_act,
    )


def train_eval(model, train_x, train_y, test_x, test_y, *,
               epochs: int, lr: float = 1e-3, wd: float = 1e-4,
               normalize: bool = False, batch_size: int = 16) -> dict:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if normalize:
        x_mean = train_x.mean(); x_std = train_x.std() + 1e-6
        y_mean = train_y.mean(); y_std = train_y.std() + 1e-6
        tx = (train_x - x_mean) / x_std
        ty = (train_y - y_mean) / y_std
        ex = (test_x - x_mean) / x_std
        ey_raw = test_y           # eval in raw space
    else:
        tx, ty = train_x, train_y
        ex = test_x; ey_raw = test_y
        x_mean = y_mean = torch.tensor(0.0); x_std = y_std = torch.tensor(1.0)

    n = tx.shape[0]
    model.train()
    t0 = time.time()
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = tx[idx].to(DEVICE)
            by = ty[idx].to(DEVICE)
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        losses.append(ep_loss / max(1, nb))
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred_n = model(ex.to(DEVICE))
        if normalize:
            pred_n = pred_n.cpu() * y_std + y_mean
        else:
            pred_n = pred_n.cpu()
        rel_l2 = (torch.norm(pred_n - ey_raw) / (torch.norm(ey_raw) + 1e-8)).item()

    return {
        "rel_l2": rel_l2,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_time_s": round(elapsed, 1),
        "first_loss": round(losses[0], 5),
        "last_loss": round(losses[-1], 5),
    }


def main():
    print("="*84)
    print("  DEEPONET FAITHFUL — diagnostic sweep")
    print("="*84)
    print(f"  device: {DEVICE} | seed: {SEED} | train/test/res: {TRAIN_N}/{TEST_N}/{RES}")

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    gen = NavierStokesGenerator(resolution=RES, device=DEVICE)
    print("\n  generating data...")
    train_x, train_y = gen.generate(TRAIN_N)
    test_x, test_y = gen.generate(TEST_N)
    print(f"  data: x.std={train_x.std():.3f}, y.std={train_y.std():.3f}, "
          f"x.range=[{train_x.min():.2f},{train_x.max():.2f}], "
          f"y.range=[{train_y.min():.2f},{train_y.max():.2f}]")

    rows = []
    configs = [
        # (epochs, normalize, latent, trunk_act, label)
        (15,   False, 128, "tanh", "baseline (current)"),
        (15,   True,  128, "tanh", "+norm"),
        (15,   True,  128, "gelu", "+norm +gelu"),
        (100,  True,  128, "gelu", "+norm +gelu +100ep"),
        (100,  True,  256, "gelu", "+norm +gelu +256lat"),
        (400,  True,  256, "gelu", "+norm +gelu +400ep"),
    ]

    for (ep, norm, lat, act, label) in configs:
        torch.manual_seed(SEED)
        model = make_deeponet(latent=lat, trunk_act=act)
        r = train_eval(model, train_x, train_y, test_x, test_y,
                       epochs=ep, normalize=norm)
        r["label"] = label
        r["epochs"] = ep
        r["normalize"] = norm
        r["latent"] = lat
        r["trunk_act"] = act
        print(f"   {label:<26}  rel L2={r['rel_l2']:.4f}  "
              f"params={r['params']:>8,}  ep={ep:>3}  "
              f"loss {r['first_loss']:.4f}->{r['last_loss']:.5f}  "
              f"{r['train_time_s']}s")
        rows.append(r)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    out = os.path.join(ROOT, "results", "diagnose_deeponet.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "train_n": TRAIN_N, "test_n": TEST_N, "res": RES,
            "rows": rows,
        }, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
