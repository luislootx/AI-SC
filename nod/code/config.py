"""Master configuration for the AI Scientific Community swarm."""
from dataclasses import dataclass
import torch


@dataclass
class SwarmConfig:
    # Swarm parameters
    num_labs: int = 12
    num_iterations: int = 15
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5
    social_coeff: float = 1.5
    initial_exploration_rate: float = 0.8
    min_exploration_rate: float = 0.2
    exploration_decay: float = 0.93

    # Neural operator parameters
    resolution: int = 32
    num_train_samples: int = 200
    num_test_samples: int = 50
    train_epochs_per_iteration: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3

    # Fitness weights (multi-objective).
    # Tuned so accuracy dominates; small efficiency / novelty terms prevent
    # parameter-saturated or trivial models from gaming the composite.
    weight_accuracy: float = 0.70
    weight_generalization: float = 0.20
    weight_efficiency: float = 0.05
    weight_novelty: float = 0.05
    # Accuracy floor: composite is zeroed for labs below this threshold,
    # preventing trivial models from winning via novelty+efficiency only.
    accuracy_floor: float = 0.50

    # Peer review
    num_reviewers_per_lab: int = 3
    vote_influence_weight: float = 0.3

    # Agentic layer
    use_llm_planner: bool = False
    use_llm_reviewer: bool = False
    llm_backend: str = "ollama"   # "ollama" or "openai" (read by agents.build_backend if None)

    # Fitness variants
    unbounded_fitness: bool = False  # if True, accuracy = -log10(rel_l2)/4 (more headroom)

    # PDE selector. "ns" = NavierStokes (default); also "darcy", "heat".
    pde_name: str = "ns"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def small_smoke_config() -> SwarmConfig:
    """Tiny config for a fast end-to-end sanity check (~minutes on GPU)."""
    return SwarmConfig(
        num_labs=4,
        num_iterations=3,
        resolution=32,
        num_train_samples=64,
        num_test_samples=16,
        train_epochs_per_iteration=5,
        batch_size=8,
    )


def medium_config() -> SwarmConfig:
    """Mid-size run for the workshop demo (~30-60 min on RTX 4080)."""
    return SwarmConfig(
        num_labs=8,
        num_iterations=8,
        resolution=32,
        num_train_samples=256,
        num_test_samples=64,
        train_epochs_per_iteration=15,
        batch_size=16,
    )


def agentic_smoke_config() -> SwarmConfig:
    """Smoke test with the LLM agentic layer enabled (planner + reviewer)."""
    cfg = small_smoke_config()
    cfg.use_llm_planner = True
    cfg.use_llm_reviewer = True
    return cfg


def agentic_demo_config() -> SwarmConfig:
    """Mid-size demo with full agentic layer for the workshop."""
    cfg = medium_config()
    cfg.use_llm_planner = True
    cfg.use_llm_reviewer = True
    return cfg


def paper_grade_config() -> SwarmConfig:
    """Paper-grade discovery run (~2-3 h on RTX 4080).

    Above the "minimum defensible" baseline. Used for ICERM poster + paper:
      • 16 labs to broaden architectural diversity
      • 20 iterations for swarm dynamics to stabilize
      • 50 epochs/candidate so each fitness measurement is well-converged
      • 512 train / 128 test — matches DeepONet-friendly data regime so the
        swarm can find Darcy-style winners too
      • Full agentic layer (LLM planner + reviewer)
      • Slower exploration decay → keeps diverse paradigms alive longer
    """
    return SwarmConfig(
        num_labs=16,
        num_iterations=20,
        resolution=32,
        num_train_samples=512,
        num_test_samples=128,
        train_epochs_per_iteration=50,
        batch_size=16,
        initial_exploration_rate=0.85,
        min_exploration_rate=0.15,
        exploration_decay=0.95,
        use_llm_planner=True,
        use_llm_reviewer=True,
        unbounded_fitness=True,
    )


def large_demo_config() -> SwarmConfig:
    """Scale-up config: 16 labs, 25 iter, 512 samples, unbounded fitness."""
    return SwarmConfig(
        num_labs=16,
        num_iterations=25,
        resolution=32,
        num_train_samples=512,
        num_test_samples=128,
        train_epochs_per_iteration=20,
        batch_size=16,
        initial_exploration_rate=0.85,
        min_exploration_rate=0.15,
        exploration_decay=0.95,    # slower decay so exploration lasts longer
        use_llm_planner=True,
        use_llm_reviewer=True,
        unbounded_fitness=True,
    )
