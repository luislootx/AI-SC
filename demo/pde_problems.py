"""PDE problem definitions with initial conditions and exact solutions.

Each problem defines:
- training data (initial condition fit)
- validation data (solution at t_final for PDE, or dense grid for regression)
- exact solution for comparison
- characteristic shift for PDE solve (linear advection)
"""

import numpy as np


class PiecewiseRegression:
    """Piecewise constant function with smooth perturbation. 4 jumps to discover."""

    name = "Piecewise Regression"
    description = "4 discontinuities in [-5, 5] with smooth background"
    mode = "regression"

    def __init__(self):
        self.x_min, self.x_max = -5.0, 5.0
        self.jumps = [-3.0, -1.0, 1.0, 3.0]
        self.t_final = 0.0  # no time evolution

    def target_function(self, x):
        y = np.zeros_like(x, dtype=float)
        y[x < -3] = -2.0
        y[(x >= -3) & (x < -1)] = 1.0
        y[(x >= -1) & (x < 1)] = -1.0
        y[(x >= 1) & (x < 3)] = 3.0
        y[x >= 3] = 0.0
        # add smooth perturbation
        y += 0.3 * np.sin(2.0 * x)
        return y

    def training_data(self, n_points=500):
        x = np.linspace(self.x_min, self.x_max, n_points)
        return x, self.target_function(x)

    def validation_data(self, n_points=1000):
        x = np.linspace(self.x_min, self.x_max, n_points)
        y = self.target_function(x)
        return x, y

    def exact_solution(self, x, t=0.0):
        return self.target_function(x)


class LinearAdvection:
    """Linear advection u_t + v * u_x = 0 with step initial condition.

    Exact solution: u(x,t) = u_0(x - v*t).
    The characteristic transform maps the PDE to an IC evaluation.
    """

    name = "Linear Advection"
    description = "Step IC propagating at v=1.0, demonstrates characteristic shift"
    mode = "pde"

    def __init__(self, v=1.0, u_L=5.0, u_R=1.0, x_disc=1.0):
        self.v = v
        self.u_L = u_L
        self.u_R = u_R
        self.x_disc = x_disc
        self.x_min, self.x_max = 0.0, 2.0
        self.t_final = 0.5

    def initial_condition(self, x):
        return np.where(x < self.x_disc, self.u_L, self.u_R)

    def training_data(self, n_points=500):
        x = np.linspace(self.x_min, self.x_max, n_points)
        return x, self.initial_condition(x)

    def validation_data(self, n_points=1000):
        x = np.linspace(self.x_min, self.x_max, n_points)
        y = self.exact_solution(x, self.t_final)
        return x, y

    def exact_solution(self, x, t):
        return self.initial_condition(x - self.v * t)

    def characteristic_shift(self, x, t):
        """For linear advection, the characteristic coordinate is x - v*t."""
        return x - self.v * t


class BurgersShock:
    """Inviscid Burgers u_t + u * u_x = 0 with Riemann IC (shock).

    Shock speed from Rankine-Hugoniot: s = (u_L + u_R) / 2.
    Demonstrates step neurons tracking shock position.
    """

    name = "Burgers Equation (Shock)"
    description = "Riemann problem with shock, demonstrates shock tracking"
    mode = "pde"

    def __init__(self, u_L=2.0, u_R=0.5, x_disc=0.3):
        self.u_L = u_L
        self.u_R = u_R
        self.x_disc = x_disc
        self.shock_speed = (u_L + u_R) / 2.0
        self.x_min, self.x_max = 0.0, 1.5
        self.t_final = 0.3

    def initial_condition(self, x):
        return np.where(x < self.x_disc, self.u_L, self.u_R)

    def training_data(self, n_points=500):
        x = np.linspace(self.x_min, self.x_max, n_points)
        return x, self.initial_condition(x)

    def validation_data(self, n_points=1000):
        x = np.linspace(self.x_min, self.x_max, n_points)
        y = self.exact_solution(x, self.t_final)
        return x, y

    def exact_solution(self, x, t):
        x_shock = self.x_disc + self.shock_speed * t
        return np.where(x < x_shock, self.u_L, self.u_R)

    def characteristic_shift(self, x, t):
        """Approximate: use shock speed for the shift (valid for Riemann)."""
        return x - self.shock_speed * t


PROBLEMS = {
    "Piecewise Regression": PiecewiseRegression,
    "Linear Advection": LinearAdvection,
    "Burgers Shock": BurgersShock,
}
