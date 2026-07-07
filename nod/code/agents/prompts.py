"""Prompt templates for the agentic layer.

The schema returned by each agent is fixed and parsed back into Python on the
caller side. We constrain the search space explicitly so the LLM can only
propose values the swarm machinery understands.
"""
from typing import Any, Dict, List

ALLOWED_BLOCKS = ["fourier", "attention", "branch_trunk", "wavelet", "residual_conv"]
ALLOWED_ACTIVATIONS = ["relu", "gelu", "silu", "tanh"]
ALLOWED_HIDDEN = [16, 32, 48, 64, 96, 128]
ALLOWED_MODES = [2, 4, 6, 8, 10, 12, 16]
ALLOWED_LR = [1e-4, 5e-4, 1e-3, 2e-3]
ALLOWED_WD = [0.0, 1e-5, 1e-4]


PLANNER_SYSTEM = """You are the planning agent of a virtual research lab in a
swarm of labs collectively discovering neural operator architectures for 2D
Navier-Stokes. Each lab is one PSO particle; your job is to propose the next
architecture (genome) for your lab to train and evaluate next iteration.

You must balance EXPLORATION (novel hybrids, unexplored block combinations)
with EXPLOITATION (move toward the global best architecture observed in the
swarm). Use the lab's current exploration_rate as a guide: high → favor
exploration, low → favor exploitation. Consider what other labs are doing —
if the swarm is converging, propose something different to maintain diversity
and avoid groupthink.

Output STRICT JSON ONLY (no prose, no fences) matching this schema:
{
  "action": "explore" | "exploit" | "hybridize",
  "rationale": "<1-2 sentences explaining your choice>",
  "new_genome": {
    "block_sequence": ["<block_name>", ...],   // 2-8 entries from ALLOWED_BLOCKS
    "hidden_channels": <int from ALLOWED_HIDDEN>,
    "fourier_modes":   <int from ALLOWED_MODES>,
    "activation":      "<one of ALLOWED_ACTIVATIONS>",
    "use_gating":      <bool>,
    "use_skip_connections": <bool>,
    "dropout_rate":    <float in [0.0, 0.3]>,
    "learning_rate":   <float from ALLOWED_LR>,
    "weight_decay":    <float from ALLOWED_WD>
  }
}

Constraints:
- ALLOWED_BLOCKS = ["fourier","attention","branch_trunk","wavelet","residual_conv"]
- ALLOWED_ACTIVATIONS = ["relu","gelu","silu","tanh"]
- ALLOWED_HIDDEN = [16,32,48,64,96,128]
- ALLOWED_MODES = [2,4,6,8,10,12,16]
- ALLOWED_LR = [1e-4,5e-4,1e-3,2e-3]
- ALLOWED_WD = [0.0,1e-5,1e-4]
"""


def planner_user_prompt(*, lab_id: int, paradigm: str, iteration: int,
                        exploration_rate: float, current_genome: dict,
                        own_history: List[dict],
                        global_best: dict | None,
                        peers_summary: List[dict]) -> str:
    import json as _json
    return f"""Lab id: {lab_id}  | seed paradigm: {paradigm}
Iteration: {iteration}
Exploration rate (1.0=explore, 0.0=exploit): {exploration_rate:.2f}

Your current genome:
{_json.dumps(current_genome, indent=2)}

Your last 3 fitness records (newest last):
{_json.dumps(own_history[-3:] if own_history else [], indent=2)}

Global best in the swarm so far:
{_json.dumps(global_best, indent=2) if global_best else "(none yet)"}

What other ACTIVE labs are doing right now (anonymized):
{_json.dumps(peers_summary, indent=2)}

Propose your next genome. Respond with JSON only.
"""


REVIEWER_SYSTEM = """You are an anonymous peer reviewer evaluating neural
operator architectures proposed by other virtual labs in a research swarm.

You must rank the candidates on overall scientific merit for solving 2D
Navier-Stokes operator learning. Reward: high accuracy, good generalization
under noise, parameter efficiency, architectural novelty (unusual block
combinations), and theoretical soundness (e.g. fourier blocks for periodic
domains, attention for nonlocal coupling). Penalize: groupthink (identical to
global best), pathological setups (too few or too many blocks, mismatched
modes), or poor measured fitness.

Output STRICT JSON ONLY:
{
  "scores":   { "<lab_id>": <float in [0,1]>, ... },   // one entry per candidate
  "rationale": "<1-2 sentences total, not per lab>"
}
Scores must sum to roughly 1.0 across the candidates (soft-vote distribution).
"""


def reviewer_user_prompt(*, reviewer_id: int, reviewer_paradigm: str,
                         candidates: List[dict]) -> str:
    import json as _json
    return f"""You are reviewer Lab {reviewer_id} (seed paradigm: {reviewer_paradigm}).

Candidates to review (each has its genome and measured fitness):
{_json.dumps(candidates, indent=2)}

Distribute your votes across the candidates. Respond with JSON only.
"""
