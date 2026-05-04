"""Resume-safe honest baseline validation (2D PDEs)\.

Improvements over v2:
  • Per-(seed, model, pde) results checkpointed to disk *immediately* —
    a PC reboot loses at most one in-flight run.
  • Per-model training budget (DeepONet gets 200+ epochs, FNO/POD get 50;
    v2 gave everyone 15, which is broken for DeepONet).
  • Per-model I/O normalization toggle (DeepONet faithful needs it).
  • Multi-PDE: 2D Navier-Stokes + 2D Darcy (Burgers 1D registered but
    skipped — 1D requires 1D-versioned blocks; future work).
  • CUDA cache + gc cleanup between runs.
  • Heartbeat log file flushed every epoch.

Resume: every result lands at
  results/exp4v3/{pde}__{model}__seed{seed}.json
On startup we read the directory and skip any (pde, model, seed) already
present. To force re-run a slot, delete its JSON.

Aggregation: a separate function reads all per-result files and computes
mean ± std, then writes results/exp4v3/AGGREGATE.json + a Markdown table.

Run:
  PY="C:/Users/luisl/anaconda3/envs/jax-env-3.11/python.exe"
  $PY code/validate_baselines_v3.py
  $PY code/validate_baselines_v3.py --pdes ns2d darcy2d --seeds 42 137 2024
  $PY code/validate_baselines_v3.py --aggregate-only
"""
from __future__ import annotations
import argparse, gc, json, math, os, random, sys, time
from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import mean, stdev
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "exp4v3")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "_run.log")

sys.path.insert(0, os.path.join(ROOT, "code"))
from baselines import PureFNO, DeepONetFaithful, PODDeepONet
from genome import ArchitectureGenome, ConfigurableNeuralOperator
from pdes_extra import make_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- model registry -----------------------------------------------

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


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[int], nn.Module]   # builder(resolution) -> Module
    epochs: int
    normalize_io: bool
    lr: float = 1e-3
    wd: float = 1e-4
    needs_pod_fit: bool = False           # POD-DeepONet only


def build_models(resolution: int) -> dict[str, ModelSpec]:
    return {
        # Honest budgets per architecture:
        # - DeepONet faithful: 200 ep + I/O norm — its known requirement
        # - POD-DeepONet: 80 ep
        # - Pure FNO sweep: 50 ep (already converges fast)
        # - Pure Transformer: 80 ep
        # - Discovered Hybrid: 80 ep (matches what swarm trained with)
        "FNO h64 m12":          ModelSpec("FNO h64 m12",
            lambda r: PureFNO(in_ch=1, out_ch=1, hidden=64, modes=12, depth=4),
            epochs=50, normalize_io=False),
        "FNO h32 m12":          ModelSpec("FNO h32 m12",
            lambda r: PureFNO(in_ch=1, out_ch=1, hidden=32, modes=12, depth=4),
            epochs=50, normalize_io=False),
        "FNO h128 m16":         ModelSpec("FNO h128 m16",
            lambda r: PureFNO(in_ch=1, out_ch=1, hidden=128, modes=16, depth=4),
            epochs=50, normalize_io=False),
        # Diagnostic 2026-05-02: DeepONet faithful needs (norm + gelu trunk +
        # latent=256 + 400+ epochs) to escape the 0.92 plateau. We give it
        # 600 here; loss curve was still descending at 400 ep.
        "DeepONet faithful":    ModelSpec("DeepONet faithful",
            lambda r: DeepONetFaithful(
                in_resolution=r, latent=256, branch_hidden=256,
                trunk_hidden=256, trunk_depth=5, trunk_activation="gelu"),
            epochs=600, normalize_io=True, lr=5e-4),
        "POD-DeepONet":         ModelSpec("POD-DeepONet",
            lambda r: PODDeepONet(in_resolution=r, latent=64, branch_hidden=256),
            epochs=80, normalize_io=False, needs_pod_fit=True),
        "Pure Transformer":     ModelSpec("Pure Transformer",
            lambda r: ConfigurableNeuralOperator(PURE_TRANSFORMER),
            epochs=80, normalize_io=False),
        "Discovered Hybrid":    ModelSpec("Discovered Hybrid",
            lambda r: ConfigurableNeuralOperator(DISCOVERED_HYBRID),
            epochs=80, normalize_io=False),
    }


# ---------- helpers ------------------------------------------------------

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def slot_path(pde: str, model: str, seed: int) -> str:
    safe = model.replace(" ", "_").replace("/", "_")
    return os.path.join(RESULTS_DIR, f"{pde}__{safe}__seed{seed}.json")


def load_completed_slots() -> set[tuple[str, str, int]]:
    done = set()
    for fn in os.listdir(RESULTS_DIR):
        if not fn.endswith(".json") or fn == "AGGREGATE.json":
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
                r = json.load(f)
            if "rel_l2_clean" in r and r.get("rel_l2_clean") is not None:
                done.add((r["pde"], r["model"], r["seed"]))
        except Exception:
            continue
    return done


# ---------- training -----------------------------------------------------

def train_and_eval(model: nn.Module, train_x, train_y, test_x, test_y,
                   *, spec: ModelSpec) -> dict:
    model = model.to(DEVICE)
    x_mean = train_x.mean(); x_std = train_x.std() + 1e-6
    y_mean = train_y.mean(); y_std = train_y.std() + 1e-6
    if spec.normalize_io:
        tx = (train_x - x_mean) / x_std
        ty = (train_y - y_mean) / y_std
        ex_clean = (test_x - x_mean) / x_std
    else:
        tx, ty, ex_clean = train_x, train_y, test_x
    ey_raw = test_y

    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)
    n = tx.shape[0]
    bs = 16
    model.train()
    t0 = time.time()
    losses = []
    for ep in range(spec.epochs):
        perm = torch.randperm(n)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
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
        if (ep + 1) % max(1, spec.epochs // 5) == 0 or ep == spec.epochs - 1:
            log(f"      ep {ep+1:>3}/{spec.epochs}  loss={losses[-1]:.5f}")
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred_clean = model(ex_clean.to(DEVICE)).cpu()
        if spec.normalize_io:
            pred_clean = pred_clean * y_std + y_mean
        rel_l2_clean = (torch.norm(pred_clean - ey_raw)
                        / (torch.norm(ey_raw) + 1e-8)).item()

        noisy_x_raw = test_x + 0.1 * torch.randn_like(test_x)
        if spec.normalize_io:
            noisy_in = (noisy_x_raw - x_mean) / x_std
        else:
            noisy_in = noisy_x_raw
        pred_noisy = model(noisy_in.to(DEVICE)).cpu()
        if spec.normalize_io:
            pred_noisy = pred_noisy * y_std + y_mean
        rel_l2_noisy = (torch.norm(pred_noisy - ey_raw)
                        / (torch.norm(ey_raw) + 1e-8)).item()

    return {
        "rel_l2_clean": rel_l2_clean,
        "rel_l2_noisy": rel_l2_noisy,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": spec.epochs,
        "lr": spec.lr,
        "wd": spec.wd,
        "normalize_io": spec.normalize_io,
        "train_time_s": round(elapsed, 1),
        "first_loss": round(losses[0], 5),
        "last_loss": round(losses[-1], 5),
    }


# ---------- per-PDE driver -----------------------------------------------

PDE_CONFIGS = {
    # name: (resolution, train_n, test_n)
    "ns2d":   (32, 256, 64),
    "darcy2d":(32, 256, 64),
    # "burgers1d": (128, 512, 64),  # SKIPPED (would need 1D versions of blocks)
}


def make_data(pde: str, seed: int):
    res, train_n, test_n = PDE_CONFIGS[pde]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    log(f"    generating {pde} data (res={res}, n_train={train_n}, n_test={test_n})...")
    t0 = time.time()
    gen = make_generator(pde, resolution=res, device=DEVICE)
    train_x, train_y = gen.generate(train_n)
    test_x, test_y = gen.generate(test_n)
    log(f"    data ready in {time.time()-t0:.1f}s; "
        f"x.std={train_x.std():.3f} y.std={train_y.std():.3f}")
    return train_x, train_y, test_x, test_y, res


def run_slot(pde: str, model_name: str, seed: int, models: dict[str, ModelSpec],
             cached_data: Optional[tuple] = None) -> dict:
    spec = models[model_name]
    if cached_data is None:
        train_x, train_y, test_x, test_y, res = make_data(pde, seed)
    else:
        train_x, train_y, test_x, test_y, res = cached_data

    log(f"  -> {pde} | {model_name} | seed {seed} (epochs={spec.epochs}, "
        f"norm_io={spec.normalize_io})")
    torch.manual_seed(seed)
    model = spec.builder(res)
    if spec.needs_pod_fit:
        model.fit_pod_basis(train_y)

    r = train_and_eval(model, train_x, train_y, test_x, test_y, spec=spec)
    r.update({"pde": pde, "model": model_name, "seed": seed,
              "resolution": res, "train_n": train_x.shape[0],
              "test_n": test_x.shape[0],
              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    out_path = slot_path(pde, model_name, seed)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    log(f"     saved {os.path.basename(out_path)}  rel L2 clean={r['rel_l2_clean']:.4f} "
        f"noisy={r['rel_l2_noisy']:.4f}  ({r['train_time_s']}s)")
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r


# ---------- aggregator ---------------------------------------------------

def aggregate():
    by_key = defaultdict(list)
    for fn in os.listdir(RESULTS_DIR):
        if not fn.endswith(".json") or fn == "AGGREGATE.json":
            continue
        with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
            r = json.load(f)
        by_key[(r["pde"], r["model"])].append(r)

    table = []
    for (pde, model), rs in by_key.items():
        cs = [r["rel_l2_clean"] for r in rs]
        ns = [r["rel_l2_noisy"] for r in rs]
        params = rs[0]["params"]
        table.append({
            "pde": pde, "model": model, "n_seeds": len(rs), "params": params,
            "rel_l2_clean_mean": round(mean(cs), 5),
            "rel_l2_clean_std":  round(stdev(cs), 5) if len(cs) > 1 else 0.0,
            "rel_l2_noisy_mean": round(mean(ns), 5),
            "rel_l2_noisy_std":  round(stdev(ns), 5) if len(ns) > 1 else 0.0,
            "values_clean": [round(v, 5) for v in cs],
            "values_noisy": [round(v, 5) for v in ns],
        })
    table.sort(key=lambda r: (r["pde"], r["rel_l2_clean_mean"]))

    out = os.path.join(RESULTS_DIR, "AGGREGATE.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    log(f"\n  wrote {out}")

    md_lines = ["# Baseline validation aggregate", ""]
    cur_pde = None
    for row in table:
        if row["pde"] != cur_pde:
            cur_pde = row["pde"]
            md_lines += ["", f"## {cur_pde}", "",
                         "| Model | params | rel L2 clean (μ ± σ) | rel L2 noisy (μ ± σ) | n_seeds |",
                         "| --- | ---: | ---: | ---: | ---: |"]
        md_lines.append(
            f"| {row['model']} | {row['params']:,} | "
            f"{row['rel_l2_clean_mean']:.4f} ± {row['rel_l2_clean_std']:.4f} | "
            f"{row['rel_l2_noisy_mean']:.4f} ± {row['rel_l2_noisy_std']:.4f} | "
            f"{row['n_seeds']} |")
    md_path = os.path.join(RESULTS_DIR, "AGGREGATE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    log(f"  wrote {md_path}")
    return table


# ---------- main ---------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    p.add_argument("--pdes",  nargs="+", default=["ns2d", "darcy2d"],
                   choices=list(PDE_CONFIGS.keys()))
    p.add_argument("--models", nargs="+", default=None,
                   help="restrict to these model names (default: all)")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    log("=" * 84)
    log(f"Baseline validation (resume-safe)  device={DEVICE}  seeds={args.seeds}  pdes={args.pdes}")
    log("=" * 84)

    # Build model spec for each resolution we'll use; cache per-PDE resolution.
    res_to_models = {}
    for pde in args.pdes:
        res = PDE_CONFIGS[pde][0]
        if res not in res_to_models:
            res_to_models[res] = build_models(res)

    done = load_completed_slots()
    log(f"  resumed: {len(done)} slots already complete")

    # Plan: iterate seed -> pde -> model. Generate data once per (seed, pde),
    # then loop models. This avoids regenerating NS data per-model.
    for seed in args.seeds:
        for pde in args.pdes:
            res = PDE_CONFIGS[pde][0]
            models = res_to_models[res]
            wanted = list(args.models) if args.models else list(models.keys())

            # Skip the entire (seed, pde) data gen if all models for it are done
            pending_for_block = [m for m in wanted
                                 if (pde, m, seed) not in done]
            if not pending_for_block:
                log(f"  ({pde}, seed {seed}) all done, skipping")
                continue

            cached = make_data(pde, seed)
            for m in wanted:
                if (pde, m, seed) in done:
                    log(f"  skip {pde} | {m} | seed {seed} (already done)")
                    continue
                try:
                    run_slot(pde, m, seed, models, cached_data=cached)
                    done.add((pde, m, seed))
                except Exception as e:
                    log(f"  FAILED {pde}/{m}/seed{seed}: {type(e).__name__}: {e}")
                    err_path = slot_path(pde, m, seed) + ".error"
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(f"{type(e).__name__}: {e}\n")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()

            # Free cached data before next PDE
            del cached
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    aggregate()


if __name__ == "__main__":
    main()
