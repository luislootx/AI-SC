"""Global swarm registry + lab lifecycle manager (deactivate/respawn)."""
from typing import Dict, List, Optional
import random
from genome import ArchitectureGenome
from config import SwarmConfig


class SwarmRegistry:
    def __init__(self):
        self.records: Dict[int, Dict] = {}
        self.global_best_genome: Optional[ArchitectureGenome] = None
        self.global_best_fitness: float = -float("inf")
        self.global_best_lab_id: Optional[int] = None
        self.iteration_history: List[Dict] = []

    def update_lab(self, lab, peer_votes: float):
        self.records[lab.lab_id] = {
            "paradigm": lab.paradigm,
            "genome": lab.current_genome.to_dict() if lab.current_genome else None,
            "fitness": lab.current_fitness,
            "peer_votes": peer_votes,
            "trust_score": lab.trust_score,
            "is_active": lab.is_active,
            "param_count": lab.current_model.count_parameters() if lab.current_model else 0,
        }

    def update_global_best(self, lab) -> bool:
        if lab.current_fitness and lab.current_fitness.composite > self.global_best_fitness:
            self.global_best_fitness = lab.current_fitness.composite
            self.global_best_genome = lab.current_genome.copy()
            self.global_best_lab_id = lab.lab_id
            return True
        return False

    def record_iteration(self, iteration: int, labs):
        snap = {
            "iteration": iteration,
            "global_best_fitness": round(self.global_best_fitness, 4),
            "global_best_lab_id": self.global_best_lab_id,
            "global_best_blocks": (self.global_best_genome.block_sequence
                                   if self.global_best_genome else []),
            "num_active_labs": sum(1 for lab in labs if lab.is_active),
            "lab_summaries": [],
        }
        for lab in labs:
            if lab.is_active and lab.current_fitness:
                snap["lab_summaries"].append({
                    "lab_id": lab.lab_id,
                    "paradigm": lab.paradigm,
                    "blocks": lab.current_genome.block_sequence if lab.current_genome else [],
                    "composite": round(lab.current_fitness.composite, 4),
                    "rel_l2_clean": round(lab.current_fitness.rel_l2_clean, 5),
                    "rel_l2_noisy": round(lab.current_fitness.rel_l2_noisy, 5),
                    "trust": round(lab.trust_score, 3),
                })
        self.iteration_history.append(snap)

    def print_leaderboard(self, labs, top_k: int = 5):
        active = [(lab, lab.current_fitness.composite) for lab in labs
                  if lab.is_active and lab.current_fitness]
        active.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "=" * 80)
        print("  SWARM LEADERBOARD")
        print("=" * 80)
        for rank, (lab, score) in enumerate(active[:top_k], 1):
            blocks = "+".join(lab.current_genome.block_sequence)
            print(f"  #{rank}  Lab {lab.lab_id:2d} ({lab.paradigm:18s}) | "
                  f"composite={score:.4f} | trust={lab.trust_score:.3f} | [{blocks}]")
        print(f"\n  Global best: Lab {self.global_best_lab_id} "
              f"(fitness={self.global_best_fitness:.4f})")
        if self.global_best_genome:
            g = self.global_best_genome
            print(f"  Best architecture: {'+'.join(g.block_sequence)} | "
                  f"ch={g.hidden_channels} | modes={g.fourier_modes} | act={g.activation}")
        print("=" * 80 + "\n")


class LabLifecycleManager:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.min_active_labs = max(4, config.num_labs // 3)
        self.deactivation_threshold = 0.15

    def manage(self, labs, registry, iteration: int):
        if iteration < 3:
            return labs
        active = [lab for lab in labs if lab.is_active and lab.current_fitness]
        if len(active) <= self.min_active_labs:
            return labs
        active.sort(key=lambda l: l.current_fitness.composite)
        n_deact = max(1, int(len(active) * self.deactivation_threshold))
        n_deact = min(n_deact, len(active) - self.min_active_labs)
        deactivated = []
        for i in range(n_deact):
            lab = active[i]
            if lab.trust_score < 0.3:
                lab.is_active = False
                deactivated.append(lab.lab_id)
        top = active[-3:]
        for did in deactivated:
            dl = next(l for l in labs if l.lab_id == did)
            parent = random.choice(top)
            dl.is_active = True
            dl.paradigm = parent.paradigm + "_spawn"
            dl.current_genome = (parent.best_genome.copy() if parent.best_genome
                                 else parent.current_genome.copy())
            dl.exploration_rate = self.config.initial_exploration_rate * 0.7
            dl.trust_score = 0.4
            dl._explore()
            print(f"    Lab {did} respawned from Lab {parent.lab_id} "
                  f"(paradigm: {dl.paradigm})")
        return labs
