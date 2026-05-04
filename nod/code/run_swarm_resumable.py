"""Resume-safe wrapper around AIScientificCommunity.

Saves a checkpoint after every iteration:
  results/swarm_runs/{tag}/state.pkl     — registry + lab states + iter idx
  results/swarm_runs/{tag}/data.pt       — train/test tensors (saved once)
  results/swarm_runs/{tag}/config.json   — config snapshot
  results/swarm_runs/{tag}/iter_{N:02d}.json — per-iteration leaderboard snapshot

On startup, if state.pkl exists, the runner:
  1. Restores RNG, registry, lab genomes/fitness/velocity/exploration_rate
  2. Reuses cached data
  3. Resumes from iteration (last_completed + 1)

This means a PC reboot loses at most ONE in-flight iteration (the one that
was running when power died); all prior progress is preserved.

Usage:
  PY="C:/Users/luisl/anaconda3/envs/jax-env-3.11/python.exe"
  $PY code/run_swarm_resumable.py paper-grade --pde ns    --tag paperv1_ns    --seed 42
  $PY code/run_swarm_resumable.py paper-grade --pde darcy --tag paperv1_darcy --seed 42

Re-run the same command after a crash → it picks up where it left off.
"""
from __future__ import annotations
import argparse, json, os, pickle, random, sys, time
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from config import (SwarmConfig, small_smoke_config, medium_config,
                    agentic_smoke_config, agentic_demo_config,
                    large_demo_config, paper_grade_config)
from community import AIScientificCommunity
from novelty import compute_novelty_scores


PRESETS = {
    "smoke":          small_smoke_config,
    "medium":         medium_config,
    "agentic-smoke":  agentic_smoke_config,
    "agentic-demo":   agentic_demo_config,
    "large-demo":     large_demo_config,
    "paper-grade":    paper_grade_config,
    "full":           SwarmConfig,
}


# ---------- lab (de)serialization ---------------------------------------

def _serialize_lab(lab) -> dict:
    """Pull the resumable state out of a VirtualLab. NN modules are NOT
    saved (they are rebuilt+retrained each iteration anyway)."""
    return {
        "lab_id":           lab.lab_id,
        "paradigm":         lab.paradigm,
        "current_genome":   (lab.current_genome.to_dict()
                             if lab.current_genome else None),
        "best_genome":      (lab.best_genome.to_dict()
                             if lab.best_genome else None),
        "best_fitness":     lab.best_fitness,
        "velocity":         dict(lab.velocity),
        "exploration_rate": lab.exploration_rate,
        "trust_score":      lab.trust_score,
        "is_active":        lab.is_active,
        "history":          list(lab.history),
    }


def _restore_lab(lab, snap: dict):
    from genome import ArchitectureGenome
    lab.lab_id            = snap["lab_id"]
    lab.paradigm          = snap["paradigm"]
    lab.current_genome    = (ArchitectureGenome(**snap["current_genome"])
                             if snap["current_genome"] else None)
    lab.best_genome       = (ArchitectureGenome(**snap["best_genome"])
                             if snap["best_genome"] else None)
    lab.best_fitness      = snap["best_fitness"]
    lab.velocity          = dict(snap["velocity"])
    lab.exploration_rate  = snap["exploration_rate"]
    lab.trust_score       = snap["trust_score"]
    lab.is_active         = snap["is_active"]
    lab.history           = list(snap["history"])


def _serialize_registry(reg) -> dict:
    return {
        "global_best_fitness":  reg.global_best_fitness,
        "global_best_lab_id":   reg.global_best_lab_id,
        "global_best_genome":   (reg.global_best_genome.to_dict()
                                 if reg.global_best_genome else None),
        "iteration_history":    list(reg.iteration_history),
        # records dict is rebuilt each iteration in update_lab; safe to drop
    }


def _restore_registry(reg, snap: dict):
    from genome import ArchitectureGenome
    reg.global_best_fitness  = snap["global_best_fitness"]
    reg.global_best_lab_id   = snap["global_best_lab_id"]
    reg.global_best_genome   = (ArchitectureGenome(**snap["global_best_genome"])
                                if snap["global_best_genome"] else None)
    reg.iteration_history    = list(snap["iteration_history"])


# ---------- checkpoint I/O -----------------------------------------------

def save_checkpoint(run_dir: str, community, last_completed_iter: int):
    state = {
        "last_completed_iter": last_completed_iter,
        "registry":            _serialize_registry(community.registry),
        "labs":                [_serialize_lab(lab) for lab in community.labs],
        "rng": {
            "torch_cpu":  torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy":      np.random.get_state(),
            "python":     random.getstate(),
        },
        "saved_at":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = os.path.join(run_dir, "state.pkl.tmp")
    final = os.path.join(run_dir, "state.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, final)   # atomic on Windows + Unix


def load_checkpoint(run_dir: str) -> dict | None:
    p = os.path.join(run_dir, "state.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_iter_snapshot(run_dir: str, it: int, registry, labs):
    snap = {
        "iteration": it,
        "global_best_fitness": registry.global_best_fitness,
        "global_best_lab_id":  registry.global_best_lab_id,
        "global_best_genome":  (registry.global_best_genome.to_dict()
                                if registry.global_best_genome else None),
        "leaderboard": [],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for lab in sorted(
            [l for l in labs if l.is_active and l.current_fitness],
            key=lambda l: l.current_fitness.composite, reverse=True):
        snap["leaderboard"].append({
            "lab_id": lab.lab_id, "paradigm": lab.paradigm,
            "blocks": lab.current_genome.block_sequence,
            "composite": round(lab.current_fitness.composite, 5),
            "rel_l2_clean": round(lab.current_fitness.rel_l2_clean, 6),
            "rel_l2_noisy": round(lab.current_fitness.rel_l2_noisy, 6),
            "trust": round(lab.trust_score, 3),
            "params": (lab.current_model.count_parameters()
                       if lab.current_model else None),
        })
    with open(os.path.join(run_dir, f"iter_{it+1:02d}.json"), "w",
              encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)


# ---------- resumable run loop -------------------------------------------

def run_resumable(community: AIScientificCommunity, *, run_dir: str,
                  data_path: str) -> int:
    cfg = community.config

    # ---- DATA: cache or generate ----
    if os.path.exists(data_path):
        print(f"  Loading cached data from {data_path}")
        d = torch.load(data_path)
        community.train_x = d["train_x"]; community.train_y = d["train_y"]
        community.test_x  = d["test_x"];  community.test_y  = d["test_y"]
    else:
        community._generate_data()
        torch.save({"train_x": community.train_x, "train_y": community.train_y,
                    "test_x":  community.test_x,  "test_y":  community.test_y},
                   data_path)
        print(f"  Cached data to {data_path}")

    # ---- CHECKPOINT: restore or initialize ----
    state = load_checkpoint(run_dir)
    if state is None:
        print("  No checkpoint found — initializing fresh swarm")
        community._initialize_labs()
        start_iter = 0
    else:
        print(f"  Restoring checkpoint from {state['saved_at']} "
              f"(last completed iter={state['last_completed_iter']+1})")
        # Init labs first to create them (paradigm + initial structure),
        # then overwrite their state from the snapshot
        community._initialize_labs()
        if len(state["labs"]) != len(community.labs):
            print(f"  WARN: lab count mismatch (saved {len(state['labs'])} vs "
                  f"current {len(community.labs)}); aborting")
            return -1
        for lab, snap in zip(community.labs, state["labs"]):
            _restore_lab(lab, snap)
        _restore_registry(community.registry, state["registry"])
        torch.set_rng_state(state["rng"]["torch_cpu"])
        if torch.cuda.is_available() and state["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        start_iter = state["last_completed_iter"] + 1
        print(f"  Resuming at iteration {start_iter+1}/{cfg.num_iterations}")

    # ---- ITERATION LOOP ----
    print("=" * 80)
    print("  BEGINNING SWARM ITERATIONS")
    print("=" * 80)
    for it in range(start_iter, cfg.num_iterations):
        print(f"\n{'─' * 80}")
        print(f"  ITERATION {it + 1}/{cfg.num_iterations}")
        print(f"{'─' * 80}")
        active = [lab for lab in community.labs if lab.is_active]
        print(f"  Active labs: {len(active)}/{len(community.labs)}")

        print("\n  Phase 1: Planning...")
        for lab in active:
            lab.plan_next_iteration(
                community.registry.global_best_genome,
                llm_planner=community.llm_planner,
                peers=active, iteration=it)

        print("\n  Phase 2: Building and Training...")
        for lab in active:
            lab.build_and_train(community.train_x, community.train_y,
                                community.test_x,  community.test_y)

        novelty = compute_novelty_scores(active)

        print("\n  Phase 3: Peer Review...")
        votes = community.peer_review.conduct_review(
            active, community.test_x, community.test_y)

        print("\n  Phase 4: Composite Fitness...")
        for lab in active:
            if lab.current_fitness:
                lab.current_fitness.novelty = novelty.get(lab.lab_id, 0.0)
                v = votes.get(lab.lab_id, 0.0)
                lab.current_fitness.compute_composite(cfg, v)
                lab.trust_score = community.peer_review.get_trust_score(lab.lab_id)
                if community.registry.update_global_best(lab):
                    print(f"    ★ NEW GLOBAL BEST: Lab {lab.lab_id} "
                          f"(composite={lab.current_fitness.composite:.4f})")
                community.registry.update_lab(lab, v)
                lab.record_iteration(it, v)

        community.labs = community.lifecycle.manage(
            community.labs, community.registry, it)
        community.registry.record_iteration(it, community.labs)
        community.registry.print_leaderboard(community.labs)
        community._print_diversity_metrics(active)

        # ---- checkpoint immediately after iteration ----
        save_iter_snapshot(run_dir, it, community.registry, community.labs)
        save_checkpoint(run_dir, community, last_completed_iter=it)
        print(f"  ✓ checkpoint saved (iteration {it+1}/{cfg.num_iterations})")

    community._print_final_report()
    return 0


# ---------- entry point --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=list(PRESETS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pde",  default="ns", choices=["ns", "darcy", "heat"])
    p.add_argument("--tag",  required=True,
                   help="run tag (= subdirectory under results/swarm_runs/)")
    args = p.parse_args()

    cfg = PRESETS[args.preset]()
    cfg.pde_name = args.pde

    run_dir = os.path.join(ROOT, "results", "swarm_runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    data_path = os.path.join(run_dir, "data.pt")
    cfg_path  = os.path.join(run_dir, "config.json")

    # Save config snapshot (informational)
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "preset": args.preset, "seed": args.seed, "pde": args.pde,
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
    print("║" + f"  PAPER-GRADE SWARM (resume-safe)  preset={args.preset}".center(78) + "║")
    print("║" + f"  pde={args.pde}  tag={args.tag}  seed={args.seed}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    agentic = "ON" if (cfg.use_llm_planner or cfg.use_llm_reviewer) else "OFF"
    print(f"  Config: {cfg.num_labs} labs × {cfg.num_iterations} iter, "
          f"res={cfg.resolution}, train={cfg.num_train_samples}, "
          f"epochs/iter={cfg.train_epochs_per_iteration}, device={cfg.device}")
    print(f"  Agentic: {agentic} (planner={cfg.use_llm_planner}, "
          f"reviewer={cfg.use_llm_reviewer}, backend={cfg.llm_backend})")
    print(f"  Run dir: {run_dir}")

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    community = AIScientificCommunity(cfg)
    rc = run_resumable(community, run_dir=run_dir, data_path=data_path)
    if rc != 0:
        print(f"\n  exited with rc={rc}")
        sys.exit(rc)

    final = {
        "preset": args.preset, "seed": args.seed, "pde": args.pde,
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
