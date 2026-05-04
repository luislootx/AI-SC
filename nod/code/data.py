"""Pseudo-spectral 2D Navier-Stokes (vorticity) data generator.

Solves dw/dt + u . grad(w) = nu * laplacian(w) + f on a 2*pi periodic
square domain, with Crank-Nicolson diffusion + Adams-Bashforth advection
and a fixed Kolmogorov-like forcing.
"""
from typing import Optional, Tuple
import numpy as np
import torch


class NavierStokesGenerator:
    def __init__(
        self,
        resolution: int = 32,
        viscosity: float = 1e-3,
        dt: float = 0.01,
        T_burn: float = 1.0,
        T_evolve: float = 0.5,
        device: str = "cpu",
    ):
        self.res = resolution
        self.nu = viscosity
        self.dt = dt
        self.T_burn = T_burn
        self.T_evolve = T_evolve
        self.device = device

        k = torch.fft.fftfreq(resolution, d=1.0 / resolution).to(device)
        self.kx = k.unsqueeze(1).expand(resolution, resolution)
        self.ky = k.unsqueeze(0).expand(resolution, resolution)
        self.k_sq = self.kx ** 2 + self.ky ** 2
        self.k_sq_safe = self.k_sq.clone()
        self.k_sq_safe[0, 0] = 1.0

        x = torch.linspace(0, 2 * np.pi, resolution + 1, device=device)[:-1]
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.forcing = 4 * torch.sin(4 * Y) - 2 * torch.cos(3 * X)

    def _random_vorticity(self) -> torch.Tensor:
        amp = torch.randn(self.res, self.res, dtype=torch.cfloat, device=self.device)
        spectrum = self.k_sq_safe ** (-1.5)
        spectrum[0, 0] = 0
        amp = amp * spectrum
        return torch.fft.ifft2(amp).real

    def _velocity_from_vorticity(self, w_hat):
        psi_hat = -w_hat / self.k_sq_safe
        psi_hat[0, 0] = 0
        u = torch.fft.ifft2(1j * self.ky * psi_hat).real
        v = torch.fft.ifft2(-1j * self.kx * psi_hat).real
        return u, v

    def _advection_term(self, w_hat):
        u, v = self._velocity_from_vorticity(w_hat)
        dwdx = torch.fft.ifft2(1j * self.kx * w_hat).real
        dwdy = torch.fft.ifft2(1j * self.ky * w_hat).real
        return torch.fft.fft2(u * dwdx + v * dwdy)

    def _step(self, w_hat, adv_prev: Optional[torch.Tensor]):
        adv_curr = self._advection_term(w_hat)
        f_hat = torch.fft.fft2(self.forcing)
        if adv_prev is None:
            rhs = w_hat + self.dt * (-adv_curr + f_hat)
        else:
            rhs = w_hat + self.dt * (-1.5 * adv_curr + 0.5 * adv_prev + f_hat)
        w_hat_new = rhs / (1 + 0.5 * self.nu * self.dt * self.k_sq)
        return w_hat_new, adv_curr

    def generate(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        burn_steps = int(self.T_burn / self.dt)
        evolve_steps = int(self.T_evolve / self.dt)
        inputs, outputs = [], []
        for _ in range(num_samples):
            w_hat = torch.fft.fft2(self._random_vorticity())
            adv_prev = None
            for _ in range(burn_steps):
                w_hat, adv_prev = self._step(w_hat, adv_prev)
            w_input = torch.fft.ifft2(w_hat).real.clone()
            for _ in range(evolve_steps):
                w_hat, adv_prev = self._step(w_hat, adv_prev)
            w_output = torch.fft.ifft2(w_hat).real.clone()
            inputs.append(w_input.unsqueeze(0))
            outputs.append(w_output.unsqueeze(0))
        return torch.stack(inputs).cpu(), torch.stack(outputs).cpu()
