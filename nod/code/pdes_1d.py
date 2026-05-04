"""1D PDE / regression generators for the ICERM-abstract benchmarks.

Three problems committed to in the abstract:
  - PiecewiseRegression1D : 4-jump piecewise constant + smooth perturbation.
  - LinearAdvection1D     : u_t + v u_x = 0 with smooth random IC.
  - Burgers1D             : u_t + u u_x = nu u_xx (re-uses pdes_extra logic).

Each generator mirrors the `NavierStokesGenerator` interface used by
`AIScientificCommunity` / `lab.evaluate_model`:

    gen = SomeGen1D(resolution=128, device=device)
    x, y = gen.generate(num_samples)        # (N, 1, L), (N, 1, L)

The abstract-mandated framing is "operator learning over 1D fields", so we
wrap the regression problem as a 1-sample-per-batch operator (input field =
target field with no perturbation; the swarm still has to recover it from a
single-channel signal).
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import torch


# ====================================================================
# 1.  Piecewise-constant regression
# ====================================================================

class PiecewiseRegression1D:
    """4-jump piecewise constant target + smooth sinusoidal perturbation.

    Each sample randomises (a) the discontinuity locations (b) the plateau
    heights (c) the perturbation amplitude/frequency. Input = the noisy
    field; target = the clean field. This makes it an operator-learning
    problem (denoising / signal recovery), not pointwise regression.
    """
    name = "PiecewiseRegression1D"

    def __init__(self, resolution: int = 128, x_min: float = -5.0,
                 x_max: float = 5.0, noise_std: float = 0.15,
                 device: str = "cpu"):
        self.res = resolution
        self.x_min, self.x_max = x_min, x_max
        self.noise_std = noise_std
        self.device = device
        self.x = torch.linspace(x_min, x_max, resolution, device=device)

    def _random_target(self) -> torch.Tensor:
        n_jumps = 4
        # jumps drawn uniformly inside the domain (sorted)
        jump_positions = sorted(
            float(j) for j in
            torch.empty(n_jumps).uniform_(self.x_min + 0.5, self.x_max - 0.5).tolist())
        # plateau heights in [-3, 3]
        heights = torch.empty(n_jumps + 1).uniform_(-3.0, 3.0).tolist()
        y = torch.zeros_like(self.x)
        boundaries = [self.x_min - 1.0] + jump_positions + [self.x_max + 1.0]
        for k in range(n_jumps + 1):
            mask = (self.x >= boundaries[k]) & (self.x < boundaries[k + 1])
            y = torch.where(mask, torch.full_like(y, heights[k]), y)
        # smooth perturbation
        amp = float(torch.empty(()).uniform_(0.1, 0.4).item())
        freq = float(torch.empty(()).uniform_(1.0, 3.0).item())
        y = y + amp * torch.sin(freq * self.x)
        return y

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ins, outs = [], []
        for _ in range(num_samples):
            target = self._random_target()
            noisy = target + self.noise_std * torch.randn_like(target)
            ins.append(noisy.unsqueeze(0))      # (1, L)
            outs.append(target.unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# ====================================================================
# 2.  Linear advection u_t + v u_x = 0 with periodic IC
# ====================================================================

class LinearAdvection1D:
    """1D linear advection on [0, 1] with periodic BC.

    Input  = initial condition u(x, 0).
    Output = u(x, T) = u_0(x - v T) (exact characteristic shift).
    Random ICs: smooth Fourier-mode mixtures.
    """
    name = "LinearAdvection1D"

    def __init__(self, resolution: int = 128, velocity: float = 1.0,
                 T: float = 0.5, ic_modes: int = 4, device: str = "cpu"):
        self.res = resolution
        self.v = velocity
        self.T = T
        self.ic_modes = ic_modes
        self.device = device
        # endpoint excluded so periodicity is exact
        self.x = torch.linspace(0.0, 1.0, resolution + 1,
                                device=device)[:-1]

    def _random_ic(self) -> torch.Tensor:
        u = torch.zeros_like(self.x)
        for m in range(1, self.ic_modes + 1):
            a = float(torch.empty(()).normal_().item()) / m
            b = float(torch.empty(()).normal_().item()) / m
            u = u + a * torch.sin(2 * math.pi * m * self.x)
            u = u + b * torch.cos(2 * math.pi * m * self.x)
        return u

    def _shift(self, u: torch.Tensor) -> torch.Tensor:
        """Spectral periodic shift by v*T."""
        L = u.shape[-1]
        u_hat = torch.fft.fft(u)
        k = torch.fft.fftfreq(L, d=1.0 / L).to(u.device)
        phase = torch.exp(-1j * 2 * math.pi * k * (self.v * self.T))
        return torch.fft.ifft(u_hat * phase).real

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ins, outs = [], []
        for _ in range(num_samples):
            u0 = self._random_ic()
            uT = self._shift(u0)
            ins.append(u0.unsqueeze(0))
            outs.append(uT.unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# ====================================================================
# 3.  Burgers' equation (viscous, periodic)
# ====================================================================

class Burgers1D:
    """Viscous Burgers u_t + u u_x = nu u_xx on [0, 1] with periodic BC.

    Input  = initial condition u(x, 0).
    Output = u(x, T).  Pseudo-spectral solver (CN on diffusion, explicit
    advection). Mirrors `pdes_extra.Burgers1DGenerator` but returns
    3-D tensors (B, 1, L) to match the 1D operator pipeline.
    """
    name = "Burgers1D"

    def __init__(self, resolution: int = 128, viscosity: float = 0.01,
                 T: float = 1.0, dt: float = 5e-4, ic_modes: int = 5,
                 device: str = "cpu"):
        self.res = resolution
        self.nu = viscosity
        self.T = T
        self.dt = dt
        self.ic_modes = ic_modes
        self.device = device
        k = torch.fft.fftfreq(resolution, d=1.0 / resolution).to(device)
        self.k = 2 * math.pi * k
        self.k_sq = self.k ** 2

    def _random_ic(self) -> torch.Tensor:
        x = torch.linspace(0, 1, self.res + 1, device=self.device)[:-1]
        u = torch.zeros_like(x)
        for m in range(1, self.ic_modes + 1):
            a = float(torch.empty(()).normal_().item()) * (1.0 / m)
            b = float(torch.empty(()).normal_().item()) * (1.0 / m)
            u = u + a * torch.sin(2 * math.pi * m * x)
            u = u + b * torch.cos(2 * math.pi * m * x)
        return u

    def _step(self, u: torch.Tensor) -> torch.Tensor:
        u_hat = torch.fft.fft(u)
        ux = torch.fft.ifft(1j * self.k * u_hat).real
        rhs = u - self.dt * u * ux
        rhs_hat = torch.fft.fft(rhs)
        u_new_hat = rhs_hat / (1 + self.dt * self.nu * self.k_sq)
        return torch.fft.ifft(u_new_hat).real

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nsteps = int(round(self.T / self.dt))
        ins, outs = [], []
        for _ in range(num_samples):
            u0 = self._random_ic()
            u = u0.clone()
            for _ in range(nsteps):
                u = self._step(u)
            ins.append(u0.unsqueeze(0))
            outs.append(u.unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# ====================================================================
# Registry
# ====================================================================

def make_generator_1d(name: str, resolution: int = 128, device: str = "cpu"):
    name = name.lower()
    if name in ("pwreg", "piecewise", "piecewise_regression"):
        return PiecewiseRegression1D(resolution=resolution, device=device)
    if name in ("advec", "advection", "linear_advection"):
        return LinearAdvection1D(resolution=resolution, device=device)
    if name in ("burgers", "burgers1d"):
        return Burgers1D(resolution=resolution, device=device)
    raise ValueError(f"unknown 1D PDE: {name}")
