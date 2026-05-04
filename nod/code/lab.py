"""VirtualLab: a single swarm particle. Holds planning / worker / evaluation
roles, a position (genome), velocity, personal best, and history."""
from typing import Dict, List, Optional
import random
import numpy as np

from config import SwarmConfig
from blocks import BLOCK_TYPE_OPTIONS, ACTIVATION_OPTIONS
from genome import ArchitectureGenome, ConfigurableNeuralOperator, random_genome
from fitness import FitnessScore, evaluate_model


class VirtualLab:
    def __init__(self, lab_id: int, paradigm: str, config: SwarmConfig):
        self.lab_id = lab_id
        self.paradigm = paradigm
        self.config = config
        self.current_genome: Optional[ArchitectureGenome] = None
        self.current_model: Optional[ConfigurableNeuralOperator] = None
        self.current_fitness: Optional[FitnessScore] = None
        self.best_genome: Optional[ArchitectureGenome] = None
        self.best_fitness: float = -float("inf")
        self.velocity: Dict[str, float] = {
            "hidden_channels": 0.0, "fourier_modes": 0.0,
            "dropout_rate": 0.0, "learning_rate": 0.0, "num_blocks": 0.0,
        }
        self.exploration_rate = config.initial_exploration_rate
        self.trust_score: float = 0.5
        self.is_active: bool = True
        self.history: List[Dict] = []

    def initialize(self):
        self.current_genome = random_genome(self.paradigm)
        self.best_genome = self.current_genome.copy()
        print(f"  Lab {self.lab_id} ({self.paradigm}): "
              f"blocks={self.current_genome.block_sequence}, "
              f"ch={self.current_genome.hidden_channels}, "
              f"modes={self.current_genome.fourier_modes}")

    def plan_next_iteration(self, global_best_genome, *,
                            llm_planner=None, peers=None, iteration: int = 0):
        if self.current_genome is None:
            self.initialize()
            return
        if llm_planner is not None:
            try:
                proposed = llm_planner.propose(
                    lab=self,
                    global_best_genome=global_best_genome,
                    iteration=iteration,
                    peers=peers or [],
                )
                if proposed is not None:
                    self.current_genome = proposed
            except Exception as e:
                print(f"    Lab {self.lab_id}: LLM planner threw {type(e).__name__}; "
                      f"falling back to PSO planner")
                if random.random() < self.exploration_rate:
                    self._explore()
                else:
                    self._exploit(global_best_genome)
        else:
            if random.random() < self.exploration_rate:
                self._explore()
            else:
                self._exploit(global_best_genome)
        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay)

    def _explore(self):
        g = self.current_genome.copy()
        muts = random.sample([
            "swap_block", "add_block", "remove_block",
            "change_channels", "change_modes", "change_activation",
            "toggle_gating", "toggle_skip", "change_dropout",
            "change_lr", "inject_novel_block",
        ], k=random.randint(1, 3))
        for m in muts:
            if m == "swap_block" and g.block_sequence:
                i = random.randint(0, len(g.block_sequence) - 1)
                g.block_sequence[i] = random.choice(BLOCK_TYPE_OPTIONS)
            elif m == "add_block" and len(g.block_sequence) < 8:
                pos = random.randint(0, len(g.block_sequence))
                g.block_sequence.insert(pos, random.choice(BLOCK_TYPE_OPTIONS))
            elif m == "remove_block" and len(g.block_sequence) > 2:
                i = random.randint(0, len(g.block_sequence) - 1)
                g.block_sequence.pop(i)
            elif m == "change_channels":
                g.hidden_channels = random.choice([32, 48, 64, 96, 128])
            elif m == "change_modes":
                g.fourier_modes = random.choice([4, 6, 8, 10, 12, 16])
            elif m == "change_activation":
                g.activation = random.choice(ACTIVATION_OPTIONS)
            elif m == "toggle_gating":
                g.use_gating = not g.use_gating
            elif m == "toggle_skip":
                g.use_skip_connections = not g.use_skip_connections
            elif m == "change_dropout":
                g.dropout_rate = round(random.uniform(0.0, 0.2), 3)
            elif m == "change_lr":
                g.learning_rate = random.choice([1e-4, 5e-4, 1e-3, 2e-3])
            elif m == "inject_novel_block" and g.block_sequence:
                i = random.randint(0, len(g.block_sequence) - 1)
                g.block_sequence[i] = random.choice(BLOCK_TYPE_OPTIONS)
        g.num_blocks = len(g.block_sequence)
        self.current_genome = g

    def _exploit(self, global_best_genome):
        g = self.current_genome.copy()
        cfg = self.config
        if self.best_genome is None and global_best_genome is None:
            self._explore()
            return
        r1, r2 = random.random(), random.random()
        for p in self.velocity:
            cur = getattr(g, p, 0)
            pb = getattr(self.best_genome, p, cur) if self.best_genome else cur
            gb = getattr(global_best_genome, p, cur) if global_best_genome else cur
            self.velocity[p] = (cfg.inertia_weight * self.velocity[p]
                                + cfg.cognitive_coeff * r1 * (pb - cur)
                                + cfg.social_coeff * r2 * (gb - cur))
        g.hidden_channels = int(np.clip(g.hidden_channels + self.velocity["hidden_channels"], 16, 128))
        g.fourier_modes = int(np.clip(g.fourier_modes + self.velocity["fourier_modes"], 2, 16))
        g.dropout_rate = float(np.clip(g.dropout_rate + self.velocity["dropout_rate"] * 0.01, 0.0, 0.3))
        g.learning_rate = float(np.clip(g.learning_rate + self.velocity["learning_rate"] * 0.0001, 1e-5, 0.01))
        if global_best_genome and random.random() < 0.3 and global_best_genome.block_sequence:
            donor = random.choice(global_best_genome.block_sequence)
            if g.block_sequence:
                i = random.randint(0, len(g.block_sequence) - 1)
                g.block_sequence[i] = donor
        if self.best_genome and random.random() < 0.3 and self.best_genome.block_sequence:
            donor = random.choice(self.best_genome.block_sequence)
            if g.block_sequence:
                i = random.randint(0, len(g.block_sequence) - 1)
                g.block_sequence[i] = donor
        if global_best_genome and random.random() < 0.2:
            g.activation = global_best_genome.activation
        if global_best_genome and random.random() < 0.2:
            g.use_gating = global_best_genome.use_gating
        if global_best_genome and random.random() < 0.2:
            g.use_skip_connections = global_best_genome.use_skip_connections
        g.num_blocks = len(g.block_sequence)
        self.current_genome = g

    def build_and_train(self, train_x, train_y, test_x, test_y):
        if self.current_genome is None:
            return None
        try:
            self.current_model = ConfigurableNeuralOperator(self.current_genome)
            n_params = self.current_model.count_parameters()
            print(f"    Lab {self.lab_id}: {n_params:,} params, "
                  f"blocks={self.current_genome.block_sequence}")
            self.current_fitness = evaluate_model(
                self.current_model, train_x, train_y, test_x, test_y, self.config)
            if self.current_fitness.accuracy > self.best_fitness:
                self.best_fitness = self.current_fitness.accuracy
                self.best_genome = self.current_genome.copy()
            return self.current_fitness
        except Exception as e:
            print(f"    Lab {self.lab_id}: BUILD/TRAIN FAILED - {type(e).__name__}: {e}")
            self.current_fitness = FitnessScore()
            self.current_model = None
            return self.current_fitness

    def record_iteration(self, iteration: int, peer_votes: float):
        rec = {
            "iteration": iteration,
            "genome": self.current_genome.to_dict() if self.current_genome else None,
            "fitness": None,
            "peer_votes": peer_votes,
            "exploration_rate": self.exploration_rate,
            "trust_score": self.trust_score,
        }
        if self.current_fitness:
            f = self.current_fitness
            rec["fitness"] = {
                "accuracy": round(f.accuracy, 4),
                "generalization": round(f.generalization, 4),
                "efficiency": round(f.efficiency, 4),
                "novelty": round(f.novelty, 4),
                "composite": round(f.composite, 4),
            }
        self.history.append(rec)

    def summary(self) -> str:
        if self.current_genome is None:
            return f"Lab {self.lab_id} ({self.paradigm}): not initialized"
        blocks = "+".join(self.current_genome.block_sequence)
        f = self.current_fitness
        if f:
            fit = (f"acc={f.accuracy:.4f}, gen={f.generalization:.4f}, "
                   f"eff={f.efficiency:.4f}, nov={f.novelty:.4f}, comp={f.composite:.4f}")
        else:
            fit = "N/A"
        return (f"Lab {self.lab_id} ({self.paradigm}): [{blocks}] "
                f"ch={self.current_genome.hidden_channels} "
                f"modes={self.current_genome.fourier_modes} | {fit} | "
                f"trust={self.trust_score:.3f}")
