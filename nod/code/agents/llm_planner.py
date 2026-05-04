"""LLM-backed planner: replaces VirtualLab._explore/_exploit with a
language-model-driven proposal. On any failure, falls back to the analytic
planner so a swarm run never crashes on transient LLM errors.
"""
from typing import Any, Dict, List, Optional
import random

from genome import ArchitectureGenome
from agents.llm_backend import LLMBackend
from agents.prompts import (
    PLANNER_SYSTEM, planner_user_prompt,
    ALLOWED_BLOCKS, ALLOWED_ACTIVATIONS, ALLOWED_HIDDEN,
    ALLOWED_MODES, ALLOWED_LR, ALLOWED_WD,
)


def _nearest(value, options):
    return min(options, key=lambda x: abs(x - value))


def _coerce_genome(d: Dict[str, Any], fallback: ArchitectureGenome) -> ArchitectureGenome:
    """Map an LLM-proposed JSON to a valid genome. Snap to allowed values."""
    try:
        seq = d.get("block_sequence") or fallback.block_sequence
        seq = [b for b in seq if b in ALLOWED_BLOCKS]
        if not seq:
            seq = list(fallback.block_sequence)
        seq = seq[:8]
        if len(seq) < 2:
            seq = seq + [random.choice(ALLOWED_BLOCKS) for _ in range(2 - len(seq))]
        hc = int(d.get("hidden_channels", fallback.hidden_channels))
        hc = _nearest(hc, ALLOWED_HIDDEN)
        fm = int(d.get("fourier_modes", fallback.fourier_modes))
        fm = _nearest(fm, ALLOWED_MODES)
        act = d.get("activation", fallback.activation)
        if act not in ALLOWED_ACTIVATIONS:
            act = fallback.activation
        gate = bool(d.get("use_gating", fallback.use_gating))
        skip = bool(d.get("use_skip_connections", fallback.use_skip_connections))
        drop = float(d.get("dropout_rate", fallback.dropout_rate))
        drop = max(0.0, min(0.3, drop))
        lr = float(d.get("learning_rate", fallback.learning_rate))
        lr = _nearest(lr, ALLOWED_LR)
        wd = float(d.get("weight_decay", fallback.weight_decay))
        wd = _nearest(wd, ALLOWED_WD)
        return ArchitectureGenome(
            block_sequence=seq, hidden_channels=hc, fourier_modes=fm,
            activation=act, use_gating=gate, use_skip_connections=skip,
            dropout_rate=round(drop, 3), num_blocks=len(seq),
            learning_rate=lr, weight_decay=wd)
    except Exception:
        return fallback.copy()


class LLMPlanner:
    def __init__(self, backend: LLMBackend, log_calls: bool = True):
        self.backend = backend
        self.log_calls = log_calls
        self.last_action: Optional[str] = None
        self.last_rationale: Optional[str] = None

    def propose(self, *, lab, global_best_genome: Optional[ArchitectureGenome],
                iteration: int, peers: List[Any]) -> ArchitectureGenome:
        peers_summary = []
        for p in peers:
            if p.lab_id == lab.lab_id or p.current_genome is None:
                continue
            entry = {
                "lab_id": p.lab_id,
                "paradigm": p.paradigm,
                "blocks": list(p.current_genome.block_sequence),
                "hidden_channels": p.current_genome.hidden_channels,
                "fourier_modes": p.current_genome.fourier_modes,
            }
            if p.current_fitness:
                entry["composite"] = round(p.current_fitness.composite, 4)
                entry["accuracy"] = round(p.current_fitness.accuracy, 4)
            peers_summary.append(entry)

        own_hist = []
        for r in lab.history[-3:]:
            row = {"iteration": r["iteration"]}
            if r.get("fitness"):
                row.update(r["fitness"])
            own_hist.append(row)

        user = planner_user_prompt(
            lab_id=lab.lab_id,
            paradigm=lab.paradigm,
            iteration=iteration,
            exploration_rate=lab.exploration_rate,
            current_genome=lab.current_genome.to_dict(),
            own_history=own_hist,
            global_best=(global_best_genome.to_dict() if global_best_genome else None),
            peers_summary=peers_summary,
        )
        try:
            data = self.backend.chat_json(
                PLANNER_SYSTEM, user, max_tokens=900, temperature=0.8)
            new_genome = _coerce_genome(data.get("new_genome", {}), lab.current_genome)
            self.last_action = data.get("action")
            self.last_rationale = data.get("rationale")
            if self.log_calls:
                print(f"    Lab {lab.lab_id} planner [{self.backend.name}/{self.backend.model}] "
                      f"action={self.last_action!s:>10} :: "
                      f"{(self.last_rationale or '')[:90]}")
            return new_genome
        except Exception as e:
            print(f"    Lab {lab.lab_id} planner FAILED ({type(e).__name__}: {e}); "
                  f"falling back to PSO planner")
            self.last_action = "fallback"
            self.last_rationale = f"{type(e).__name__}: {e}"
            # Fall back to the analytic planner via lab's own methods
            if random.random() < lab.exploration_rate:
                lab._explore()
            else:
                lab._exploit(global_best_genome)
            return lab.current_genome
