"""1D variant of AIScientificCommunity.

Wires the 1D PDE generators (`pdes_1d`) and the 1D-aware VirtualLab
(`lab_1d`). Everything else (peer review, novelty, registry, lifecycle)
is re-used unchanged from the 2D community.
"""
from __future__ import annotations
import random

from community import AIScientificCommunity, PDE_LABELS  # noqa: F401
from lab_1d import VirtualLab1D
from pdes_1d import make_generator_1d


PDE_LABELS_1D = {
    "pwreg":   "PiecewiseRegression-1D",
    "advec":   "LinearAdvection-1D",
    "burgers": "Burgers-1D",
}


class AIScientificCommunity1D(AIScientificCommunity):
    """Same swarm logic, but data is 1D and labs build 1D operators."""

    def __init__(self, config):
        # Bypass parent __init__'s 2D data-generator branch by setting the
        # generator before invoking the rest of parent setup.
        self.config = config
        from registry import SwarmRegistry, LabLifecycleManager
        self.registry = SwarmRegistry()
        self.lifecycle = LabLifecycleManager(config)
        self.labs = []

        pde = getattr(config, "pde_name", "burgers").lower()
        self.data_generator = make_generator_1d(
            pde, resolution=config.resolution, device=config.device)
        self.pde_label = PDE_LABELS_1D.get(pde, pde)

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

        from peer_review import PeerReviewSystem
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
        print("  INITIALIZING AI SCIENTIFIC COMMUNITY (1D)")
        print("=" * 80)
        for i, p in enumerate(paradigms):
            lab = VirtualLab1D(lab_id=i, paradigm=p, config=self.config)
            lab.initialize()
            self.labs.append(lab)
        print(f"\n  Initialized {len(self.labs)} virtual 1D labs across "
              f"{len(set(paradigms))} paradigms\n")
