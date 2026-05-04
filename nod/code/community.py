"""Top-level orchestrator: runs the AI Scientific Community swarm loop."""
from collections import defaultdict
from typing import Dict, List
import random
import numpy as np

from config import SwarmConfig
from data import NavierStokesGenerator
from lab import VirtualLab
from peer_review import PeerReviewSystem
from novelty import compute_novelty_scores
from registry import SwarmRegistry, LabLifecycleManager


PDE_LABELS = {"ns": "Navier-Stokes", "darcy": "Darcy", "heat": "Heat"}


class AIScientificCommunity:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.registry = SwarmRegistry()
        self.lifecycle = LabLifecycleManager(config)
        self.labs: List[VirtualLab] = []
        pde = getattr(config, "pde_name", "ns").lower()
        if pde in ("ns", "navier_stokes", "navier-stokes"):
            self.data_generator = NavierStokesGenerator(
                resolution=config.resolution, device=config.device)
        else:
            from pdes import get_generator
            self.data_generator = get_generator(
                pde, resolution=config.resolution, device=config.device)
        self.pde_label = PDE_LABELS.get(pde, pde)

        self.llm_planner = None
        self.llm_reviewer = None
        if config.use_llm_planner or config.use_llm_reviewer:
            from agents import build_backend, LLMPlanner, LLMReviewer
            backend = build_backend(config.llm_backend)
            print(f"  LLM backend ready: {backend.name} / {backend.model}")
            if config.use_llm_planner:
                self.llm_planner = LLMPlanner(backend)
            if config.use_llm_reviewer:
                self.llm_reviewer = LLMReviewer(backend, config)

        self.peer_review = (self.llm_reviewer
                            if self.llm_reviewer is not None
                            else PeerReviewSystem(config))

    def _initialize_labs(self):
        paradigms = [
            "fno", "fno", "deeponet", "deeponet",
            "transformer", "transformer", "wavelet",
            "hybrid_fno_attn", "random", "random", "random", "random",
        ]
        while len(paradigms) < self.config.num_labs:
            paradigms.append(random.choice(
                ["fno", "deeponet", "transformer", "wavelet", "random"]))
        paradigms = paradigms[:self.config.num_labs]

        print("\n" + "=" * 80)
        print("  INITIALIZING AI SCIENTIFIC COMMUNITY")
        print("=" * 80)
        for i, p in enumerate(paradigms):
            lab = VirtualLab(lab_id=i, paradigm=p, config=self.config)
            lab.initialize()
            self.labs.append(lab)
        print(f"\n  Initialized {len(self.labs)} virtual labs across "
              f"{len(set(paradigms))} paradigms\n")

    def _generate_data(self):
        print(f"  Generating {self.pde_label} data...")
        self.train_x, self.train_y = self.data_generator.generate(self.config.num_train_samples)
        self.test_x, self.test_y = self.data_generator.generate(self.config.num_test_samples)
        print(f"  Train: {tuple(self.train_x.shape)}, Test: {tuple(self.test_x.shape)}\n")

    def run(self) -> SwarmRegistry:
        self._initialize_labs()
        self._generate_data()
        print("=" * 80)
        print("  BEGINNING SWARM ITERATIONS")
        print("=" * 80 + "\n")

        for it in range(self.config.num_iterations):
            print(f"\n{'─' * 80}")
            print(f"  ITERATION {it + 1}/{self.config.num_iterations}")
            print(f"{'─' * 80}")
            active = [lab for lab in self.labs if lab.is_active]
            print(f"  Active labs: {len(active)}/{len(self.labs)}")

            print("\n  Phase 1: Planning...")
            for lab in active:
                lab.plan_next_iteration(
                    self.registry.global_best_genome,
                    llm_planner=self.llm_planner,
                    peers=active,
                    iteration=it,
                )
                mode = "EXPLORE" if lab.exploration_rate > 0.5 else "EXPLOIT"
                if self.llm_planner is None:
                    print(f"    Lab {lab.lab_id}: {mode} (rate={lab.exploration_rate:.2f})")

            print("\n  Phase 2: Building and Training...")
            for lab in active:
                lab.build_and_train(self.train_x, self.train_y,
                                    self.test_x, self.test_y)

            novelty = compute_novelty_scores(active)

            print("\n  Phase 3: Peer Review...")
            votes = self.peer_review.conduct_review(active, self.test_x, self.test_y)

            print("\n  Phase 4: Composite Fitness...")
            for lab in active:
                if lab.current_fitness:
                    lab.current_fitness.novelty = novelty.get(lab.lab_id, 0.0)
                    v = votes.get(lab.lab_id, 0.0)
                    lab.current_fitness.compute_composite(self.config, v)
                    lab.trust_score = self.peer_review.get_trust_score(lab.lab_id)
                    if self.registry.update_global_best(lab):
                        print(f"    ★ NEW GLOBAL BEST: Lab {lab.lab_id} "
                              f"(composite={lab.current_fitness.composite:.4f})")
                    self.registry.update_lab(lab, v)
                    lab.record_iteration(it, v)

            self.labs = self.lifecycle.manage(self.labs, self.registry, it)
            self.registry.record_iteration(it, self.labs)
            self.registry.print_leaderboard(self.labs)
            self._print_diversity_metrics(active)

        self._print_final_report()
        return self.registry

    def _print_diversity_metrics(self, labs):
        block_usage = defaultdict(int)
        paradigm_counts = defaultdict(int)
        active_count = 0
        for lab in labs:
            if lab.is_active and lab.current_genome:
                active_count += 1
                paradigm_counts[lab.paradigm] += 1
                for b in lab.current_genome.block_sequence:
                    block_usage[b] += 1
        if active_count == 0:
            return
        print("\n  ── Diversity Metrics ──")
        print("  Active paradigms: ", end="")
        for p, c in sorted(paradigm_counts.items()):
            print(f"{p}({c}) ", end="")
        print()
        total = sum(block_usage.values())
        print("  Block usage: ", end="")
        for b, c in sorted(block_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"{b}={100 * c / (total + 1e-8):.0f}% ", end="")
        print()
        active = [l for l in labs if l.is_active and l.current_genome]
        if len(active) >= 2:
            ds = []
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ds.append(active[i].current_genome.distance(active[j].current_genome))
            print(f"  Architectural diversity index: {np.mean(ds):.4f} "
                  f"(0=homogeneous, 1=maximally diverse)")
        camps = self._detect_rival_camps(active)
        if camps:
            print(f"  Detected {len(camps)} rival camp(s):")
            for name, mems in camps.items():
                ids = [l.lab_id for l in mems]
                avg = np.mean([l.current_fitness.composite for l in mems
                               if l.current_fitness]) if mems else 0
                print(f"    Camp '{name}': Labs {ids}, avg composite={avg:.4f}")

    def _detect_rival_camps(self, labs) -> Dict[str, List]:
        camps = defaultdict(list)
        for lab in labs:
            if lab.current_genome is None:
                continue
            counts = defaultdict(int)
            for b in lab.current_genome.block_sequence:
                counts[b] += 1
            if counts:
                dom = max(counts, key=counts.get)
                if counts[dom] / len(lab.current_genome.block_sequence) > 0.5:
                    camps[f"{dom}-dominant"].append(lab)
                else:
                    camps["hybrid-mixed"].append(lab)
        return {k: v for k, v in camps.items() if len(v) >= 2}

    def _print_final_report(self):
        print("\n" + "=" * 80)
        print("  FINAL REPORT: AI SCIENTIFIC COMMUNITY RESULTS")
        print("=" * 80)
        reg = self.registry
        print("\n  ★ GLOBAL BEST ARCHITECTURE ★")
        print(f"  Discovered by: Lab {reg.global_best_lab_id}")
        print(f"  Composite fitness: {reg.global_best_fitness:.4f}")
        if reg.global_best_genome:
            g = reg.global_best_genome
            print(f"  Block sequence: {' → '.join(g.block_sequence)}")
            print(f"  Hidden channels: {g.hidden_channels}")
            print(f"  Fourier modes: {g.fourier_modes}")
            print(f"  Activation: {g.activation}")
            print(f"  Gating: {g.use_gating} | Skip: {g.use_skip_connections} | "
                  f"Dropout: {g.dropout_rate}")
            print(f"  LR: {g.learning_rate} | WD: {g.weight_decay}")
            self._classify_architecture(g)
        print("\n  ── All Labs Final State ──")
        for lab in sorted(self.labs,
                          key=lambda l: (l.current_fitness.composite if l.current_fitness else -1),
                          reverse=True):
            status = "ACTIVE" if lab.is_active else "INACTIVE"
            print(f"  [{status}] {lab.summary()}")
        print("\n  ── Global Best Evolution ──")
        for snap in reg.iteration_history:
            print(f"  Iter {snap['iteration'] + 1:2d}: "
                  f"fitness={snap['global_best_fitness']:.4f} | "
                  f"lab={snap['global_best_lab_id']} | "
                  f"blocks={'+'.join(snap['global_best_blocks'])} | "
                  f"active={snap['num_active_labs']}")

    def _classify_architecture(self, g):
        block_set = set(g.block_sequence)
        counts = defaultdict(int)
        for b in g.block_sequence:
            counts[b] += 1
        print("\n  ── Architecture Classification ──")
        if len(block_set) == 1:
            only = list(block_set)[0]
            print(f"  Type: PURE {only.upper()}")
        else:
            print("  Type: HYBRID")
            comps = [f"{b}({100 * c / len(g.block_sequence):.0f}%)"
                     for b, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
            print(f"  Components: {', '.join(comps)}")
            if "fourier" in block_set and "attention" in block_set:
                print("  Notable: Spectral-Attention hybrid")
            if "fourier" in block_set and "branch_trunk" in block_set:
                print("  Notable: Spectral-DeepONet hybrid")
            if "attention" in block_set and "branch_trunk" in block_set:
                print("  Notable: Transformer-DeepONet hybrid")
            if "wavelet" in block_set and "fourier" in block_set:
                print("  Notable: Multi-scale spectral hybrid")
