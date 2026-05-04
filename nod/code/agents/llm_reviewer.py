"""LLM-backed peer reviewer. The reviewer reads candidate genomes + measured
fitness and returns soft votes. Falls back to fitness-proportional voting on
any LLM failure so the swarm never blocks on transient errors.
"""
from collections import defaultdict
from typing import Dict, List
import random
import torch

from config import SwarmConfig
from agents.llm_backend import LLMBackend
from agents.prompts import REVIEWER_SYSTEM, reviewer_user_prompt


class LLMReviewer:
    """Drop-in replacement for PeerReviewSystem that uses an LLM for voting."""

    def __init__(self, backend: LLMBackend, config: SwarmConfig,
                 log_calls: bool = True):
        self.backend = backend
        self.config = config
        self.log_calls = log_calls
        self.vote_history: Dict[int, List[float]] = defaultdict(list)

    def conduct_review(self, labs, test_x, test_y) -> Dict[int, float]:
        if len(labs) < 2:
            return {lab.lab_id: 0.5 for lab in labs}

        device = self.config.device
        vote_counts: Dict[int, float] = defaultdict(float)

        for reviewer in labs:
            cand_pool = [l for l in labs if l.lab_id != reviewer.lab_id]
            n = min(self.config.num_reviewers_per_lab, len(cand_pool))
            sample = random.sample(cand_pool, n)
            payload = []
            for cand in sample:
                if cand.current_genome is None:
                    continue
                entry = {
                    "lab_id": cand.lab_id,
                    "paradigm": cand.paradigm,
                    "genome": cand.current_genome.to_dict(),
                }
                if cand.current_fitness:
                    entry["fitness"] = {
                        "accuracy": round(cand.current_fitness.accuracy, 4),
                        "generalization": round(cand.current_fitness.generalization, 4),
                        "efficiency": round(cand.current_fitness.efficiency, 4),
                    }
                if cand.current_model is not None:
                    try:
                        cand.current_model.eval()
                        with torch.no_grad():
                            sub = min(8, test_x.shape[0])
                            idx = torch.randperm(test_x.shape[0])[:sub]
                            pred = cand.current_model(test_x[idx].to(device))
                            tgt = test_y[idx].to(device)
                            rel = (torch.norm(pred - tgt) /
                                   (torch.norm(tgt) + 1e-8)).item()
                            entry["measured_rel_l2"] = round(rel, 4)
                    except Exception:
                        pass
                payload.append(entry)

            if not payload:
                continue

            scores = self._llm_vote(reviewer.lab_id, reviewer.paradigm, payload)
            if not scores:
                # Fallback: distribute by accuracy
                total = sum(c.current_fitness.accuracy for c in sample
                            if c.current_fitness) + 1e-8
                scores = {c.lab_id: (c.current_fitness.accuracy / total)
                          for c in sample if c.current_fitness}
            for lid, s in scores.items():
                vote_counts[lid] += float(s)

        max_v = max(vote_counts.values()) if vote_counts else 1.0
        normalized = {}
        for lab in labs:
            normalized[lab.lab_id] = vote_counts.get(lab.lab_id, 0.0) / (max_v + 1e-8)
            self.vote_history[lab.lab_id].append(normalized[lab.lab_id])
        return normalized

    def _llm_vote(self, reviewer_id: int, reviewer_paradigm: str,
                  candidates: List[dict]) -> Dict[int, float]:
        try:
            user = reviewer_user_prompt(
                reviewer_id=reviewer_id,
                reviewer_paradigm=reviewer_paradigm,
                candidates=candidates,
            )
            data = self.backend.chat_json(
                REVIEWER_SYSTEM, user, max_tokens=600, temperature=0.5)
            raw = data.get("scores") or {}
            scores: Dict[int, float] = {}
            for k, v in raw.items():
                try:
                    scores[int(k)] = max(0.0, float(v))
                except Exception:
                    continue
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
            if self.log_calls:
                rationale = (data.get("rationale") or "")[:90]
                print(f"    Reviewer Lab {reviewer_id} "
                      f"[{self.backend.name}/{self.backend.model}] :: {rationale}")
            return scores
        except Exception as e:
            print(f"    Reviewer Lab {reviewer_id} FAILED ({type(e).__name__}: {e}); "
                  f"falling back to accuracy-proportional voting")
            return {}

    def get_trust_score(self, lab_id: int) -> float:
        h = self.vote_history.get(lab_id, [])
        if not h:
            return 0.5
        weights = [0.7 ** i for i in range(len(h) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, h)) / (sum(weights) + 1e-8)
