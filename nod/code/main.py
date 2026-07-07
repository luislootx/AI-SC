"""Entry point. Usage:
    python main.py                          # full SwarmConfig() defaults (PSO only)
    python main.py smoke                    # tiny PSO-only sanity check
    python main.py medium                   # mid PSO-only run
    python main.py agentic-smoke            # tiny run with LLM planner + reviewer
    python main.py agentic-demo             # mid-size run with LLM planner + reviewer
    python main.py large-demo               # 16 labs × 25 iter, unbounded fitness (EXP-2)
    python main.py agentic-demo --seed 137  # override seed (for multi-seed runs)
Env: set LLM_BACKEND=ollama (default) or openai in .env
"""
import argparse
import json
import os
import sys
import random
import numpy as np
import torch

# Force UTF-8 on Windows consoles (default cp1252 chokes on box-drawing chars)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (SwarmConfig, small_smoke_config, medium_config,
                    agentic_smoke_config, agentic_demo_config, large_demo_config)
from community import AIScientificCommunity


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", nargs="?", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for the results JSON (so multi-seed runs don't overwrite)")
    parser.add_argument("--pde", type=str, default="ns",
                        choices=["ns", "darcy", "heat"],
                        help="PDE benchmark to discover an operator for (default: ns)")
    args = parser.parse_args()
    mode = args.mode

    if mode == "smoke":
        config = small_smoke_config()
    elif mode == "medium":
        config = medium_config()
    elif mode == "agentic-smoke":
        config = agentic_smoke_config()
    elif mode == "agentic-demo":
        config = agentic_demo_config()
    elif mode == "large-demo":
        config = large_demo_config()
    else:
        config = SwarmConfig()
    config.pde_name = args.pde

    pde_titles = {"ns": "2D Navier-Stokes (vorticity)",
                  "darcy": "2D Darcy Flow", "heat": "2D Heat Diffusion"}
    title = pde_titles.get(args.pde, args.pde)
    print("╔" + "═" * 78 + "╗")
    print("║" + " AI SCIENTIFIC COMMUNITY FOR NEURAL OPERATOR DISCOVERY ".center(78) + "║")
    print("║" + " {} | Mode: {:>10s} ".format(title, mode).center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    agentic = "ON" if (config.use_llm_planner or config.use_llm_reviewer) else "OFF"
    print(f"\n  Config: {config.num_labs} labs × {config.num_iterations} iter, "
          f"res={config.resolution}, train={config.num_train_samples}, "
          f"epochs/iter={config.train_epochs_per_iteration}, device={config.device}")
    print(f"  Agentic layer: {agentic}  (planner={config.use_llm_planner}, "
          f"reviewer={config.use_llm_reviewer}, backend={config.llm_backend})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    community = AIScientificCommunity(config)
    registry = community.run()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(out_dir, f"results_{mode}{suffix}.json")

    results = {
        "mode": mode,
        "seed": args.seed,
        "pde": args.pde,
        "config": {
            "num_labs": config.num_labs,
            "num_iterations": config.num_iterations,
            "resolution": config.resolution,
            "train_samples": config.num_train_samples,
            "epochs_per_iter": config.train_epochs_per_iteration,
            "device": config.device,
            "unbounded_fitness": config.unbounded_fitness,
            "pde": args.pde,
        },
        "global_best": {
            "fitness": registry.global_best_fitness,
            "lab_id": registry.global_best_lab_id,
            "genome": (registry.global_best_genome.to_dict()
                       if registry.global_best_genome else None),
        },
        "iteration_history": registry.iteration_history,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
