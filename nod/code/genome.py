"""Architecture genome (the "DNA" of a neural operator) and the
configurable model that builds itself from a genome."""
from dataclasses import dataclass
from typing import List
import random
import torch
import torch.nn as nn

from blocks import BLOCK_REGISTRY, BLOCK_TYPE_OPTIONS, ACTIVATION_OPTIONS, GatingBlock


@dataclass
class ArchitectureGenome:
    block_sequence: List[str]
    hidden_channels: int
    fourier_modes: int
    activation: str
    use_gating: bool
    use_skip_connections: bool
    dropout_rate: float
    num_blocks: int
    learning_rate: float
    weight_decay: float

    def to_dict(self) -> dict:
        return {
            "block_sequence": list(self.block_sequence),
            "hidden_channels": self.hidden_channels,
            "fourier_modes": self.fourier_modes,
            "activation": self.activation,
            "use_gating": self.use_gating,
            "use_skip_connections": self.use_skip_connections,
            "dropout_rate": self.dropout_rate,
            "num_blocks": self.num_blocks,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def distance(self, other: "ArchitectureGenome") -> float:
        s1, s2 = set(self.block_sequence), set(other.block_sequence)
        d = 0.0
        if s1 | s2:
            d += 1.0 - len(s1 & s2) / len(s1 | s2)
        d += abs(self.hidden_channels - other.hidden_channels) / 128.0
        d += abs(self.fourier_modes - other.fourier_modes) / 16.0
        d += abs(self.num_blocks - other.num_blocks) / 8.0
        d += abs(self.dropout_rate - other.dropout_rate)
        d += abs(self.learning_rate - other.learning_rate) / 0.01
        d += 0.0 if self.activation == other.activation else 0.5
        d += 0.0 if self.use_gating == other.use_gating else 0.3
        d += 0.0 if self.use_skip_connections == other.use_skip_connections else 0.3
        return d / 5.0

    def copy(self) -> "ArchitectureGenome":
        return ArchitectureGenome(
            block_sequence=list(self.block_sequence),
            hidden_channels=self.hidden_channels,
            fourier_modes=self.fourier_modes,
            activation=self.activation,
            use_gating=self.use_gating,
            use_skip_connections=self.use_skip_connections,
            dropout_rate=self.dropout_rate,
            num_blocks=self.num_blocks,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )


PARADIGM_TEMPLATES = {
    "fno": dict(block_sequence=["fourier"] * 4, hidden_channels=64, fourier_modes=12,
                activation="gelu", use_gating=False, use_skip_connections=True),
    "deeponet": dict(block_sequence=["branch_trunk"] * 4, hidden_channels=64, fourier_modes=8,
                     activation="relu", use_gating=False, use_skip_connections=False),
    "transformer": dict(block_sequence=["attention"] * 3, hidden_channels=64, fourier_modes=8,
                        activation="gelu", use_gating=False, use_skip_connections=True),
    "wavelet": dict(block_sequence=["wavelet"] * 4, hidden_channels=48, fourier_modes=8,
                    activation="silu", use_gating=True, use_skip_connections=True),
    "hybrid_fno_attn": dict(block_sequence=["fourier", "attention", "fourier", "attention"],
                            hidden_channels=64, fourier_modes=10, activation="gelu",
                            use_gating=True, use_skip_connections=True),
}


def random_genome(paradigm: str = "random") -> ArchitectureGenome:
    if paradigm in PARADIGM_TEMPLATES:
        t = PARADIGM_TEMPLATES[paradigm]
        blocks, hc, fm = list(t["block_sequence"]), t["hidden_channels"], t["fourier_modes"]
        act, gate, skip = t["activation"], t["use_gating"], t["use_skip_connections"]
    else:
        n = random.randint(2, 6)
        blocks = [random.choice(BLOCK_TYPE_OPTIONS) for _ in range(n)]
        hc = random.choice([32, 48, 64, 96])
        fm = random.choice([4, 6, 8, 10, 12])
        act = random.choice(ACTIVATION_OPTIONS)
        gate = random.choice([True, False])
        skip = random.choice([True, False])
    return ArchitectureGenome(
        block_sequence=blocks,
        hidden_channels=hc,
        fourier_modes=fm,
        activation=act,
        use_gating=gate,
        use_skip_connections=skip,
        dropout_rate=round(random.uniform(0.0, 0.15), 3),
        num_blocks=len(blocks),
        learning_rate=random.choice([5e-4, 1e-3, 2e-3]),
        weight_decay=random.choice([0.0, 1e-5, 1e-4]),
    )


class ConfigurableNeuralOperator(nn.Module):
    def __init__(self, genome: ArchitectureGenome, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.genome = genome
        hc = genome.hidden_channels
        self.lift = nn.Sequential(nn.Conv2d(in_channels, hc, 1), nn.GELU())
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        for block_name in genome.block_sequence:
            ctor = BLOCK_REGISTRY.get(block_name, BLOCK_REGISTRY["residual_conv"])
            self.blocks.append(ctor(hc, genome.fourier_modes, genome.activation))
            if genome.use_gating and len(self.blocks) > 1:
                self.gates.append(GatingBlock(hc))
            else:
                self.gates.append(nn.Identity())
        self.dropout = nn.Dropout2d(genome.dropout_rate) if genome.dropout_rate > 0 else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hc, hc, 1), nn.GELU(), nn.Conv2d(hc, out_channels, 1))

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
