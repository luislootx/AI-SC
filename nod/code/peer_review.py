"""Cross-lab peer review: evaluation agents vote on each other's models."""
from collections import defaultdict
from typing import Dict, List
import random
import torch

from config import SwarmConfig


class PeerReviewSystem:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.vote_history: Dict[int, List[float]] = defaultdict(list)

    def conduct_review(self, labs, test_x, test_y):
        device = self.config.device
        if len(labs) < 2:
            return {lab.lab_id: 0.5 for lab in labs}

        vote_counts = defaultdict(float)
        for reviewer in labs:
            candidates = [lab for lab in labs if lab.lab_id != reviewer.lab_id]
            n = min(self.config.num_reviewers_per_lab, len(candidates))
            sample = random.sample(candidates, n)
            scores = {}
            for cand in sample:
                if cand.current_model is None:
                    scores[cand.lab_id] = 0.0
                    continue
                try:
                    cand.current_model.eval()
                    with torch.no_grad():
                        sub = min(10, test_x.shape[0])
                        idx = torch.randperm(test_x.shape[0])[:sub]
                        pred = cand.current_model(test_x[idx].to(device))
                        target = test_y[idx].to(device)
                        rel = (torch.norm(pred - target) /
                               (torch.norm(target) + 1e-8)).item()
                        scores[cand.lab_id] = max(0.0, 1.0 - rel)
                except Exception:
                    scores[cand.lab_id] = 0.0
            if scores:
                total = sum(scores.values()) + 1e-8
                for lid, s in scores.items():
                    vote_counts[lid] += s / total

        max_v = max(vote_counts.values()) if vote_counts else 1.0
        normalized = {}
        for lab in labs:
            normalized[lab.lab_id] = vote_counts.get(lab.lab_id, 0.0) / (max_v + 1e-8)
            self.vote_history[lab.lab_id].append(normalized[lab.lab_id])
        return normalized

    def get_trust_score(self, lab_id: int) -> float:
        h = self.vote_history.get(lab_id, [])
        if not h:
            return 0.5
        weights = [0.7 ** i for i in range(len(h) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, h)) / (sum(weights) + 1e-8)
