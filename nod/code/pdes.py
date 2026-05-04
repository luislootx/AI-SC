"""Multi-PDE generators (Darcy 2D, heat 2D)\.

All generators produce 2D scalar fields shaped (B, 1, H, W) so the existing
ConfigurableNeuralOperator can ingest them without changes. Each generator
encodes a different inductive bias / physics regime:

  - NavierStokesGenerator (in data.py)  — turbulent, time-evolved, periodic
  - DarcyGenerator                       — static elliptic, log-permeability
  - HeatGenerator                        — diffusion-only, smooth
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import torch


# --- Darcy 2D ---------------------------------------------------------------

class DarcyGenerator:
    """Static elliptic problem on [0, 1]² with Dirichlet boundaries.

        -∇ · (a(x, y) ∇u) = f,    u = 0 on ∂Ω

    Input:  log a(x, y)  (random Gaussian field, smoothed)
    Output: u(x, y)      (solved by 5-point Laplacian + sparse solve)
    """
    def __init__(self, resolution: int = 32, length_scale: float = 0.15,
                 forcing: float = 1.0, device: str = "cpu"):
        self.res = resolution
        self.length_scale = length_scale
        self.forcing = forcing
        self.device = device

    def _smooth_random_field(self, n: int) -> np.ndarray:
        """Generate a smooth Gaussian random field on [0, 1]² via FFT."""
        kx = np.fft.fftfreq(self.res, d=1.0 / self.res)
        ky = np.fft.fftfreq(self.res, d=1.0 / self.res)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        # Gaussian spectral envelope, length_scale controls smoothness
        spectrum = np.exp(-0.5 * K2 * self.length_scale ** 2)
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        out = np.empty((n, self.res, self.res), dtype=np.float32)
        for i in range(n):
            noise = rng.standard_normal((self.res, self.res)) \
                  + 1j * rng.standard_normal((self.res, self.res))
            field = np.real(np.fft.ifft2(noise * spectrum))
            # normalize
            field = (field - field.mean()) / (field.std() + 1e-8)
            out[i] = field
        return out

    def _solve_darcy(self, a: np.ndarray) -> np.ndarray:
        """Solve -div(a grad u) = f on a regular grid with zero Dirichlet."""
        from scipy.sparse import diags, csr_matrix
        from scipy.sparse.linalg import spsolve
        H = self.res
        h = 1.0 / (H - 1)
        N = H * H
        # Average a at edges for FV-like stencil
        # Build interior stencil with Dirichlet BCs
        rows, cols, vals = [], [], []
        rhs = np.zeros(N)
        for j in range(H):
            for i in range(H):
                k = j * H + i
                if i == 0 or i == H - 1 or j == 0 or j == H - 1:
                    rows.append(k); cols.append(k); vals.append(1.0)
                    rhs[k] = 0.0
                else:
                    a_e = 0.5 * (a[j, i] + a[j, i + 1])
                    a_w = 0.5 * (a[j, i] + a[j, i - 1])
                    a_n = 0.5 * (a[j, i] + a[j + 1, i])
                    a_s = 0.5 * (a[j, i] + a[j - 1, i])
                    diag = (a_e + a_w + a_n + a_s) / (h * h)
                    rows.append(k); cols.append(k); vals.append(diag)
                    rows.append(k); cols.append(k + 1);   vals.append(-a_e / (h * h))
                    rows.append(k); cols.append(k - 1);   vals.append(-a_w / (h * h))
                    rows.append(k); cols.append(k + H);   vals.append(-a_n / (h * h))
                    rows.append(k); cols.append(k - H);   vals.append(-a_s / (h * h))
                    rhs[k] = self.forcing
        A = csr_matrix((vals, (rows, cols)), shape=(N, N))
        u = spsolve(A, rhs).reshape(H, H)
        return u.astype(np.float32)

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        log_a = self._smooth_random_field(num_samples)        # (n, H, H)
        a_fields = np.exp(0.5 * log_a)                         # ensure a > 0
        outputs = np.zeros_like(log_a)
        for i in range(num_samples):
            outputs[i] = self._solve_darcy(a_fields[i])
        # Normalize outputs for trainable scale
        for i in range(num_samples):
            s = outputs[i].std() + 1e-8
            outputs[i] /= s
        x = torch.from_numpy(log_a).unsqueeze(1)               # (n, 1, H, H)
        y = torch.from_numpy(outputs).unsqueeze(1)
        return x, y


# --- Heat 2D ----------------------------------------------------------------

class HeatGenerator:
    """Heat / diffusion on [0, 2π]² periodic.

        ∂u/∂t = κ ∇²u

    Input:  random initial condition (smooth)
    Output: u at time T
    Solved exactly in spectral space: u_hat(t) = u_hat(0) · exp(-κ k² t)
    """
    def __init__(self, resolution: int = 32, kappa: float = 0.1,
                 T: float = 2.0, device: str = "cpu"):
        self.res = resolution
        self.kappa = kappa
        self.T = T
        self.device = device
        k = torch.fft.fftfreq(resolution, d=1.0 / resolution).to(device)
        kx = k.unsqueeze(1).expand(resolution, resolution)
        ky = k.unsqueeze(0).expand(resolution, resolution)
        self.k_sq = kx ** 2 + ky ** 2
        self.k_sq_safe = self.k_sq.clone()
        self.k_sq_safe[0, 0] = 1.0
        self.decay = torch.exp(-kappa * self.k_sq * T)

    def _random_initial(self) -> torch.Tensor:
        amp = torch.randn(self.res, self.res, dtype=torch.cfloat,
                          device=self.device)
        spectrum = self.k_sq_safe ** (-1.5)
        spectrum[0, 0] = 0
        u_hat = amp * spectrum
        return torch.fft.ifft2(u_hat).real

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ins, outs = [], []
        for _ in range(num_samples):
            u0 = self._random_initial()
            u_hat0 = torch.fft.fft2(u0)
            uT = torch.fft.ifft2(u_hat0 * self.decay).real
            scale = u0.std() + 1e-8
            ins.append((u0 / scale).unsqueeze(0))
            outs.append((uT / scale).unsqueeze(0))
        return torch.stack(ins).cpu(), torch.stack(outs).cpu()


# --- Registry --------------------------------------------------------------

def get_generator(name: str, resolution: int = 32, device: str = "cpu"):
    name = name.lower()
    if name in ("ns", "navier_stokes", "navier-stokes"):
        from data import NavierStokesGenerator
        return NavierStokesGenerator(resolution=resolution, device=device)
    if name == "darcy":
        return DarcyGenerator(resolution=resolution, device=device)
    if name == "heat":
        return HeatGenerator(resolution=resolution, device=device)
    raise ValueError(f"Unknown PDE: {name}")
