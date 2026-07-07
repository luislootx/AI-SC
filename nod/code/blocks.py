"""Neural operator building blocks used by the configurable architecture.

Each block is registered in BLOCK_REGISTRY so the genome can mix and match
them freely across virtual labs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvLayer(nn.Module):
    """FNO-style spectral convolution (real FFT, two mode quadrants)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m = self.modes
        H = x.size(-2)
        W_freq = x.size(-1) // 2 + 1
        m_h = min(m, H)
        m_w = min(m, W_freq)
        out_ft = torch.zeros(
            B, self.out_channels, H, W_freq, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m_h, :m_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m_h, :m_w],
            self.weights1[:, :, :m_h, :m_w])
        if H >= 2 * m_h:
            out_ft[:, :, -m_h:, :m_w] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -m_h:, :m_w],
                self.weights2[:, :, :m_h, :m_w])
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(),
            "tanh": nn.Tanh()}.get(name, nn.GELU())


class FourierBlock(nn.Module):
    def __init__(self, channels: int, modes: int, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConvLayer(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = _activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, activation: str = "gelu"):
        super().__init__()
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
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_seq)
        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (C // self.num_heads) ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, H * W, C)
        x_seq = x_seq + self.proj(out)
        x_seq = x_seq + self.ff(self.norm2(x_seq))
        return x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class BranchTrunkBlock(nn.Module):
    """DeepONet-inspired branch-trunk on grid data."""

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu"):
        super().__init__()
        self.rank = rank
        self.act = _activation(activation)
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            self.act,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, rank),
        )
        self.trunk_basis = nn.Parameter(torch.randn(rank, channels))
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        branch_out = self.branch(x)
        combined = torch.einsum("br,rc->bc", branch_out, self.trunk_basis)
        combined = combined.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        return self.norm(self.proj(x + combined))


class WaveletBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "gelu"):
        super().__init__()
        self.act = _activation(activation)
        self.low_pass = nn.AvgPool2d(2)
        self.high_pass_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.low_pass_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.merge = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        low = self.low_pass(x)
        low = self.low_pass_conv(low)
        low = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        high = x - F.interpolate(self.low_pass(x), size=(H, W),
                                 mode="bilinear", align_corners=False)
        high = self.high_pass_conv(high)
        return self.act(self.norm(self.merge(torch.cat([low, high], dim=1))))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "gelu"):
        super().__init__()
        self.act = _activation(activation)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class GatingBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.transform = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        g = self.gate(x)
        return g * self.transform(x) + (1 - g) * x


BLOCK_REGISTRY = {
    "fourier":       lambda ch, modes, act: FourierBlock(ch, modes, act),
    "attention":     lambda ch, modes, act: SpatialAttentionBlock(ch, num_heads=4, activation=act),
    "branch_trunk":  lambda ch, modes, act: BranchTrunkBlock(ch, rank=16, activation=act),
    "wavelet":       lambda ch, modes, act: WaveletBlock(ch, act),
    "residual_conv": lambda ch, modes, act: ResidualConvBlock(ch, act),
}

ACTIVATION_OPTIONS = ["relu", "gelu", "silu", "tanh"]
BLOCK_TYPE_OPTIONS = list(BLOCK_REGISTRY.keys())
