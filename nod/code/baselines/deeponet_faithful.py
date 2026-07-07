"""Faithful DeepONet — Lu et al. 2021.

G[u](y) = sum_k  b_k(u) * t_k(y)  +  bias

where:
  u        — input function (here: vorticity field, sampled on a 32×32 grid)
  y        — query coordinate (x, y) ∈ [0, 2π]²
  b_k      — branch network output (encodes the input function)
  t_k      — trunk network output (encodes the query location)

This is the key difference from the simplified `branch_trunk` block used in
the swarm registry: that one operates only on grid features, this one takes
the query coordinate as a separate input. This is DeepONet's structural
advantage and what we want to give back to it for an honest comparison.
"""
from typing import Optional
import math
import torch
import torch.nn as nn


class DeepONetFaithful(nn.Module):
    def __init__(
        self,
        in_resolution: int = 32,
        in_channels: int = 1,
        latent: int = 128,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        trunk_depth: int = 4,
        trunk_activation: str = "tanh",
    ):
        super().__init__()
        self.in_resolution = in_resolution
        self.latent = latent

        # Branch: CNN over the input field → flatten → MLP → latent vector
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),    # 16×16
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),   # 8×8
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.ReLU(),  # 4×4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, branch_hidden), nn.ReLU(),
            nn.Linear(branch_hidden, latent),
        )

        # Trunk: MLP over query coordinates (R² → R^latent)
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(),
               "gelu": nn.GELU(), "silu": nn.SiLU()}[trunk_activation]
        layers = [nn.Linear(2, trunk_hidden), act]
        for _ in range(trunk_depth - 2):
            layers += [nn.Linear(trunk_hidden, trunk_hidden), act]
        layers += [nn.Linear(trunk_hidden, latent)]
        self.trunk = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(1))

        # Pre-compute the regular query grid (same for every sample)
        with torch.no_grad():
            xs = torch.linspace(0, 2 * math.pi, in_resolution + 1)[:-1]
            ys = torch.linspace(0, 2 * math.pi, in_resolution + 1)[:-1]
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self.register_buffer(
                "query_grid",
                torch.stack([xx, yy], dim=-1).reshape(-1, 2))  # (HW, 2)

    def forward(self, x_in, query_coords: Optional[torch.Tensor] = None):
        """
        x_in:           (B, C, H, W)
        query_coords:   (Nq, 2) — defaults to the input grid
        returns:        (B, 1, H_out, W_out)  reshaped if query is the grid,
                        or (B, Nq) if a custom query is passed
        """
        B = x_in.shape[0]
        b = self.branch(x_in)                         # (B, latent)
        q = self.query_grid if query_coords is None else query_coords
        t = self.trunk(q)                              # (Nq, latent)
        out = torch.einsum("bl,nl->bn", b, t) + self.bias  # (B, Nq)
        if query_coords is None:
            H = self.in_resolution
            return out.reshape(B, 1, H, H)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
