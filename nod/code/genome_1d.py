"""1D variant of ConfigurableNeuralOperator.

Reuses ArchitectureGenome / random_genome / PARADIGM_TEMPLATES from
genome.py — only the model class changes (1D ops). Same block names, so the
existing PSO planner / mutation operators / paradigm seeding all work
unchanged.
"""
from __future__ import annotations
import torch.nn as nn

from genome import ArchitectureGenome  # noqa: F401  (re-exported)
from blocks_1d import BLOCK_REGISTRY_1D, GatingBlock1D


class ConfigurableNeuralOperator1D(nn.Module):
    """1D analogue of ConfigurableNeuralOperator.

    Input/output convention: (B, in_channels, L) tensors, where L is the 1D
    spatial resolution. Block sequence + lift/project all use Conv1d.
    """

    def __init__(self, genome: ArchitectureGenome,
                 in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.genome = genome
        hc = genome.hidden_channels
        self.lift = nn.Sequential(nn.Conv1d(in_channels, hc, 1), nn.GELU())
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        for block_name in genome.block_sequence:
            ctor = BLOCK_REGISTRY_1D.get(block_name,
                                          BLOCK_REGISTRY_1D["residual_conv"])
            self.blocks.append(ctor(hc, genome.fourier_modes, genome.activation))
            if genome.use_gating and len(self.blocks) > 1:
                self.gates.append(GatingBlock1D(hc))
            else:
                self.gates.append(nn.Identity())
        self.dropout = (nn.Dropout1d(genome.dropout_rate)
                        if genome.dropout_rate > 0 else nn.Identity())
        self.project = nn.Sequential(
            nn.Conv1d(hc, hc, 1), nn.GELU(), nn.Conv1d(hc, out_channels, 1))

    def forward(self, x):
        x = self.lift(x)
        x_input = x
        for block, gate in zip(self.blocks, self.gates):
            x_prev = x
            x = block(x)
            x = gate(x)
            x = self.dropout(x)
            if self.genome.use_skip_connections and x.shape == x_prev.shape:
                x = x + x_prev
        if self.genome.use_skip_connections and x.shape == x_input.shape:
            x = x + x_input
        return self.project(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
