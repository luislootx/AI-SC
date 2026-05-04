"""Novelty scoring: each lab's average architectural distance from peers."""
from typing import Dict
import numpy as np


def compute_novelty_scores(labs) -> Dict[int, float]:
    novelty = {}
    items = [(lab.lab_id, lab.current_genome) for lab in labs
             if lab.current_genome is not None]
    for i, (lid_i, g_i) in enumerate(items):
        dists = [g_i.distance(g_j) for j, (lid_j, g_j) in enumerate(items) if i != j]
        novelty[lid_i] = float(np.mean(dists)) if dists else 0.5
    if novelty:
        m = max(novelty.values()) + 1e-8
        novelty = {k: v / m for k, v in novelty.items()}
    return novelty
