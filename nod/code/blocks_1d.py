"""1D versions of the neural operator building blocks.

Mirrors `blocks.py` (2D) so the genome can mix and match 1D blocks for the
piecewise-regression / linear-advection / Burgers benchmarks committed to in
the ICERM abstract. Same block names so genome + paradigm logic is reused.

Tensor convention: (B, C, L) where L = spatial length.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1DLayer(nn.Module):
    """FNO-style 1D spectral convolution (real FFT, lowest `modes` retained)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes,
                                dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = x.shape[-1]
        x_ft = torch.fft.rfft(x, norm="ortho")
        L_freq = L // 2 + 1
        m = min(self.modes, L_freq)
        out_ft = torch.zeros(B, self.out_channels, L_freq,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum(
            "bil,iol->bol", x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=L, norm="ortho")


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(),
            "tanh": nn.Tanh()}.get(name, nn.GELU())


class FourierBlock1D(nn.Module):
    def __init__(self, channels: int, modes: int, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv1DLayer(channels, channels, modes)
        self.bypass = nn.Conv1d(channels, channels, 1)
        self.norm = nn.InstanceNorm1d(channels)
        self.act = _activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class SpatialAttentionBlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, activation: str = "gelu"):
        super().__init__()
        # Round num_heads down so that channels % num_heads == 0
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.act = _activation(activation)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.act,
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, L = x.shape
        x_seq = x.permute(0, 2, 1).contiguous()                  # (B, L, C)
        x_norm = self.norm(x_seq)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads,
                                        C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, h, L, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (C // self.num_heads) ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, C)
        x_seq = x_seq + self.proj(out)
        x_seq = x_seq + self.ff(self.norm2(x_seq))
        return x_seq.permute(0, 2, 1).contiguous()               # (B, C, L)


class BranchTrunkBlock1D(nn.Module):
    """DeepONet-inspired branch-trunk on 1D grid data."""

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu"):
        super().__init__()
        self.rank = rank
        self.act = _activation(activation)
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            self.act,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, rank),
        )
        self.trunk_basis = nn.Parameter(torch.randn(rank, channels))
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x):
        B, C, L = x.shape
        branch_out = self.branch(x)                              # (B, rank)
        combined = torch.einsum("br,rc->bc", branch_out, self.trunk_basis)
        combined = combined.unsqueeze(-1).expand(B, C, L)
        return self.norm(self.proj(x + combined))


class WaveletBlock1D(nn.Module):
    def __init__(self, channels: int, activation: str = "gelu"):
        super().__init__()
        self.act = _activation(activation)
        self.low_pass = nn.AvgPool1d(2)
        self.high_pass_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.low_pass_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.merge = nn.Conv1d(channels * 2, channels, 1)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x):
        B, C, L = x.shape
        low = self.low_pass(x)
        low = self.low_pass_conv(low)
        low = F.interpolate(low, size=L, mode="linear", align_corners=False)
        high = x - F.interpolate(self.low_pass(x), size=L,
                                 mode="linear", align_corners=False)
        high = self.high_pass_conv(high)
        return self.act(self.norm(self.merge(torch.cat([low, high], dim=1))))


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels: int, activation: str = "gelu"):
        super().__init__()
        self.act = _activation(activation)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.norm2 = nn.InstanceNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class GatingBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.Sigmoid())
        self.transform = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        g = self.gate(x)
        return g * self.transform(x) + (1 - g) * x


BLOCK_REGISTRY_1D = {
    "fourier":       lambda ch, modes, act: FourierBlock1D(ch, modes, act),
    "attention":     lambda ch, modes, act: SpatialAttentionBlock1D(ch, num_heads=4, activation=act),
    "branch_trunk":  lambda ch, modes, act: BranchTrunkBlock1D(ch, rank=16, activation=act),
    "wavelet":       lambda ch, modes, act: WaveletBlock1D(ch, act),
    "residual_conv": lambda ch, modes, act: ResidualConvBlock1D(ch, act),
}
