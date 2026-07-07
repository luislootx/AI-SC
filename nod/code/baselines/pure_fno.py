"""Pure FNO baseline — Li et al. 2021 (Fourier Neural Operator).

Standard 2D FNO with lift → 4× SpectralConv blocks → projection.
Allows width and modes sweep for honest comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes,
                                                   dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes,
                                                   dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        H, Wf = x.size(-2), x.size(-1) // 2 + 1
        m = self.modes
        m_h = min(m, H)
        m_w = min(m, Wf)
        out = torch.zeros(B, self.out_ch, H, Wf, dtype=torch.cfloat,
                          device=x.device)
        out[:, :, :m_h, :m_w] = torch.einsum("bixy,ioxy->boxy",
            x_ft[:, :, :m_h, :m_w], self.w1[:, :, :m_h, :m_w])
        if H >= 2 * m_h:
            out[:, :, -m_h:, :m_w] = torch.einsum("bixy,ioxy->boxy",
                x_ft[:, :, -m_h:, :m_w], self.w2[:, :, :m_h, :m_w])
        return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, ch: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes)
        self.bypass = nn.Conv2d(ch, ch, 1)
        self.norm = nn.InstanceNorm2d(ch)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class PureFNO(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 hidden: int = 64, modes: int = 12, depth: int = 4):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU())
        self.blocks = nn.ModuleList(
            [FNOBlock(hidden, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1))

    def forward(self, x):
        x = self.lift(x)
        x_in = x
        for b in self.blocks:
            x = b(x)
        return self.project(x + x_in)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
