"""Extra PDE generators for honest cross-PDE evaluation.

Adds 1D Burgers and 2D Darcy to complement the existing 2D Navier-Stokes
generator in `data.py`. Same interface as `NavierStokesGenerator.generate`:
returns (inputs, outputs) tensors of shape (N, 1, ...).

Why each:
  - Burgers 1D: classic DeepONet benchmark (its home turf). If the swarm
    hybrid loses here too, the comparison is more honest.
  - Darcy 2D: smooth elliptic. FNO doesn't have a periodic / spectral
    advantage like it does on NS; both should be competitive.
  - Navier-Stokes 2D: kept in `data.py` (chaotic, periodic — favours FNO).
"""
from __future__ import annotations
from typing import Tuple
import math
import numpy as np
import torch


# ---------- 1D Burgers ----------------------------------------------------

class Burgers1DGenerator:
    """1D Burgers: u_t + u u_x = nu u_xx, periodic on [0, 1].

    Map from initial condition u(x, 0) to solution at T (default 1.0).
    Uses an explicit pseudo-spectral solver with small dt.
    Resolution = number of grid points.
    """
    def __init__(self, resolution: int = 128, viscosity: float = 0.01,
                 T: float = 1.0, dt: float = 5e-4, device: str = "cpu",
                 ic_modes: int = 5):
        self.res = resolution
        self.nu = viscosity
        self.T = T
        self.dt = dt
        self.device = device
        self.ic_modes = ic_modes

        k = torch.fft.fftfreq(resolution, d=1.0 / resolution).to(device)
        # 2*pi factor for [0,1] domain so wavenumbers match math convention
        self.k = 2 * math.pi * k
        self.k_sq = self.k ** 2

    def _random_ic(self) -> torch.Tensor:
        # Fixed-energy random IC with limited modes (smooth)
        x = torch.linspace(0, 1, self.res + 1, device=self.device)[:-1]
        u = torch.zeros_like(x)
        for m in range(1, self.ic_modes + 1):
            a = torch.randn(()) * (1.0 / m)
            b = torch.randn(()) * (1.0 / m)
            u = u + a * torch.sin(2 * math.pi * m * x) + b * torch.cos(2 * math.pi * m * x)
        return u

    def _step(self, u: torch.Tensor) -> torch.Tensor:
        u_hat = torch.fft.fft(u)
        ux = torch.fft.ifft(1j * self.k * u_hat).real
        # Crank-Nicolson on diffusion, explicit on advection
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
            # Reshape to (1, 1, N) — treat as 1D "image" with H=1
            ins.append(u0.unsqueeze(0).unsqueeze(0))
            outs.append(u.unsqueeze(0).unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# ---------- 2D Darcy ------------------------------------------------------

class Darcy2DGenerator:
    """2D Darcy flow: -div(a(x) grad u) = f on [0,1]^2 with u = 0 on bdry.

    Input: random log-permeability field a(x), output: pressure u(x).
    Uses 5-point finite-difference + sparse direct solve. Resolution
    keeps interior nodes only at output (zero Dirichlet).

    For honest cross-PDE comparison; neither FNO nor DeepONet has an
    architectural prior that fits Darcy better than the other.
    """
    def __init__(self, resolution: int = 32, force: float = 1.0,
                 length_scale: float = 0.2, device: str = "cpu",
                 modes: int = 6):
        self.res = resolution
        self.force = force
        self.ls = length_scale
        self.device = device
        self.modes = modes
        # Pre-compute spatial grid
        h = 1.0 / (resolution - 1)
        self.h = h
        x = torch.linspace(0, 1, resolution)
        Y, X = torch.meshgrid(x, x, indexing="ij")
        self.X, self.Y = X, Y

    def _random_a(self) -> torch.Tensor:
        """Smooth random log-permeability using truncated Fourier basis."""
        a = torch.zeros(self.res, self.res)
        for m in range(1, self.modes + 1):
            for n in range(1, self.modes + 1):
                w = (m * m + n * n) ** -1.0
                c1 = torch.randn(()) * w
                c2 = torch.randn(()) * w
                a = a + c1 * torch.sin(math.pi * m * self.X) * torch.sin(math.pi * n * self.Y)
                a = a + c2 * torch.cos(math.pi * m * self.X) * torch.cos(math.pi * n * self.Y)
        # Map to positive permeability ∈ [~0.1, ~10] via exp
        return torch.exp(a)

    def _solve_darcy(self, a: torch.Tensor) -> torch.Tensor:
        """5-point stencil with harmonic averaging. Pure-torch dense solve
        (avoids scipy/CUDA-MKL segfault we hit on Windows)."""
        N = self.res
        h = self.h
        nint = N - 2
        # Vectorized harmonic-mean face coefficients
        a_in = a[1:-1, 1:-1]                          # (nint, nint)
        a_e  = 2 * a_in * a[1:-1, 2:]   / (a_in + a[1:-1, 2:]   + 1e-9)
        a_w  = 2 * a_in * a[1:-1, :-2]  / (a_in + a[1:-1, :-2]  + 1e-9)
        a_n  = 2 * a_in * a[:-2, 1:-1]  / (a_in + a[:-2, 1:-1]  + 1e-9)
        a_s  = 2 * a_in * a[2:, 1:-1]   / (a_in + a[2:, 1:-1]   + 1e-9)
        # Build dense (nint*nint, nint*nint) matrix on CPU (cheap for nint<=64)
        n2 = nint * nint
        A = torch.zeros(n2, n2)
        diag = (a_e + a_w + a_n + a_s).reshape(-1)
        A[torch.arange(n2), torch.arange(n2)] = diag
        # East: (i, j) <-> (i, j+1)
        for i in range(nint):
            for j in range(nint - 1):
                k = i * nint + j
                A[k, k + 1] = -a_e[i, j]
                A[k + 1, k] = -a_w[i, j + 1]
        # North/South
        for i in range(nint - 1):
            for j in range(nint):
                k  = i * nint + j
                kn = (i + 1) * nint + j
                A[k, kn] = -a_s[i, j]
                A[kn, k] = -a_n[i + 1, j]
        rhs = torch.full((n2,), self.force * h * h)
        u_int = torch.linalg.solve(A, rhs)
        u = torch.zeros(N, N)
        u[1:-1, 1:-1] = u_int.reshape(nint, nint)
        return u

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ins, outs = [], []
        for _ in range(num_samples):
            a = self._random_a()
            u = self._solve_darcy(a)
            # Use log(a) as input field (standard practice)
            ins.append(torch.log(a + 1e-6).unsqueeze(0))
            outs.append(u.unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# ---------- registry ------------------------------------------------------

def make_generator(name: str, resolution: int, device: str = "cpu"):
    """Returns a generator with `.generate(n) -> (x, y)`."""
    if name == "ns2d":
        from data import NavierStokesGenerator
        return NavierStokesGenerator(resolution=resolution, device=device)
    if name == "burgers1d":
        return Burgers1DGenerator(resolution=resolution, device=device)
    if name == "darcy2d":
        return Darcy2DGenerator(resolution=resolution, device=device)
    raise ValueError(f"unknown PDE: {name}")
