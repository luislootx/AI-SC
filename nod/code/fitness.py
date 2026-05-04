"""Multi-objective fitness evaluation: train + evaluate a model and score it
on accuracy, generalization, efficiency, and (later) novelty + peer votes."""
import time
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from config import SwarmConfig
from genome import ConfigurableNeuralOperator


@dataclass
class FitnessScore:
    accuracy: float = 0.0
    generalization: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    peer_votes: float = 0.0
    composite: float = 0.0
    # Raw rel_l2 measurements, kept for honest reporting (independent of fitness shaping).
    rel_l2_clean: float = 1.0
    rel_l2_noisy: float = 1.0

    def compute_composite(self, config: SwarmConfig, peer_votes: float = 0.0):
        self.peer_votes = peer_votes
        base = (config.weight_accuracy * self.accuracy
                + config.weight_generalization * self.generalization
                + config.weight_efficiency * self.efficiency
                + config.weight_novelty * self.novelty)
        composite = ((1 - config.vote_influence_weight) * base
                     + config.vote_influence_weight * self.peer_votes)
        # Accuracy floor: trivial models cannot win via novelty + efficiency
        if self.accuracy < getattr(config, "accuracy_floor", 0.0):
            composite *= 0.1  # penalty: keep a tiny gradient signal but rule it out
        self.composite = composite


def evaluate_model(model: ConfigurableNeuralOperator,
                   train_x, train_y, test_x, test_y,
                   config: SwarmConfig) -> FitnessScore:
    device = config.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=model.genome.learning_rate,
                                  weight_decay=model.genome.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_epochs_per_iteration)

    n_train = train_x.shape[0]
    model.train()
    for _ in range(config.train_epochs_per_iteration):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, config.batch_size):
            idx = perm[i:i + config.batch_size]
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    fit = FitnessScore()
    with torch.no_grad():
        test_y_dev = test_y.to(device)
        pred = model(test_x.to(device))
        rel_l2 = (torch.norm(pred - test_y_dev) / (torch.norm(test_y_dev) + 1e-8)).item()
        # rel_l2_raw is stored on the score so reporting layers can recover it
        fit.rel_l2_clean = rel_l2

        noisy = test_x + 0.1 * torch.randn_like(test_x)
        pred_n = model(noisy.to(device))
        rel_l2_n = (torch.norm(pred_n - test_y_dev) / (torch.norm(test_y_dev) + 1e-8)).item()
        fit.rel_l2_noisy = rel_l2_n

        if getattr(config, "unbounded_fitness", False):
            # Map rel_l2 in [1e-5, 1] → accuracy in ~[0, 1.25] via -log10
            # This avoids the 0.99 ceiling that lets Lab 7 saturate at iter 1.
            import math
            fit.accuracy = max(0.0, -math.log10(rel_l2 + 1e-8) / 4.0)
            fit.generalization = max(0.0, -math.log10(rel_l2_n + 1e-8) / 4.0)
        else:
            fit.accuracy = max(0.0, 1.0 - rel_l2)
            fit.generalization = max(0.0, 1.0 - rel_l2_n)

        params = model.count_parameters()
        param_score = max(0.0, 1.0 - params / 5_000_000)
        start = time.time()
        for _ in range(10):
            _ = model(test_x[:4].to(device))
        elapsed = (time.time() - start) / 10.0
        time_score = max(0.0, 1.0 - elapsed / 1.0)
        fit.efficiency = 0.5 * param_score + 0.5 * time_score
    return fit
