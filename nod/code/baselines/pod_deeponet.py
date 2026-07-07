"""POD-DeepONet — Lu et al. 2022.

Replaces the trunk MLP with a fixed POD basis computed from the training
output fields. This gives the operator a strong prior that often
outperforms vanilla DeepONet on smooth-solution PDEs.

Workflow:
  1. Compute POD basis from training outputs (SVD).
  2. Trunk = first `latent` POD modes (fixed, no learnable params).
  3. Branch network → coefficients projecting input onto POD basis.
  4. Output = sum_k branch_k(u) · pod_basis_k(y) + bias.
"""
from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn


class PODDeepONet(nn.Module):
    def __init__(self, in_resolution: int = 32, in_channels: int = 1,
                 latent: int = 64, branch_hidden: int = 256):
        super().__init__()
        self.in_resolution = in_resolution
        self.latent = latent

        # Branch: input field → coefficients
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, branch_hidden), nn.ReLU(),
            nn.Linear(branch_hidden, latent),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        # POD basis is fitted later via fit_pod_basis(); register a
        # placeholder of the right shape now.
        H = in_resolution
        self.register_buffer("pod_basis",
                             torch.zeros(latent, H, H))
        self.register_buffer("pod_mean",
                             torch.zeros(H, H))

    def fit_pod_basis(self, train_y: torch.Tensor):
        """Fit POD basis from training outputs (B, 1, H, W) → (latent, H, W)."""
        with torch.no_grad():
            B, C, H, W = train_y.shape
            assert C == 1
            flat = train_y.reshape(B, H * W)              # (B, HW)
            mean = flat.mean(dim=0)                        # (HW,)
            centered = flat - mean
            # SVD: centered = U S V^T,  V (HW × k) holds the spatial modes
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            modes = Vt[: self.latent].reshape(self.latent, H, W)
            self.pod_basis.copy_(modes)
            self.pod_mean.copy_(mean.reshape(H, W))
        return S[: self.latent]

    def forward(self, x_in, query_coords: Optional[torch.Tensor] = None):
        """x_in: (B, C, H, W). Returns (B, 1, H, W)."""
        B = x_in.shape[0]
        coeffs = self.branch(x_in)                        # (B, latent)
        # Reconstruct: sum over modes
        # pod_basis: (latent, H, W); coeffs: (B, latent)
        out = torch.einsum("bk,khw->bhw", coeffs, self.pod_basis)
        out = out + self.pod_mean + self.bias              # (B, H, W)
        return out.unsqueeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
