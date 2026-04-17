"""Core Step-CINN functions: ELM basis, step neurons, ridge regression.

This module implements the fundamental building blocks of the Step-CINN method:
- Tanh hidden neurons with fixed random weights (ELM philosophy)
- Step neurons (sigmoid) with learnable positions (GA-optimized)
- Ridge regression for analytical output-weight solving
"""

import numpy as np


def generate_tanh_weights(n_tanh, seed=7, scale=2.5):
    """Generate fixed random input weights for tanh neurons (ELM-style)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale, scale, size=n_tanh)
    return W, b


def tanh_features(x, W, b):
    """Compute tanh hidden-layer activations. Returns (N, n_tanh)."""
    return np.tanh(np.outer(x, W) + b)


def step_features(x, positions, kappa=500.0):
    """Compute step neuron activations: sigma(kappa * (x - x_step)).

    Each step neuron is a sigmoid centered at a learnable position.
    The kappa parameter controls steepness (larger = sharper step).
    Returns (N, n_steps).
    """
    if len(positions) == 0:
        return np.empty((len(x), 0))
    z = kappa * (x[:, None] - np.array(positions)[None, :])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def build_feature_matrix(x, W, b, positions, kappa):
    """Build combined [tanh | step] feature matrix."""
    H_tanh = tanh_features(x, W, b)
    if len(positions) > 0:
        H_step = step_features(x, positions, kappa)
        return np.hstack([H_tanh, H_step])
    return H_tanh


def solve_ridge(H, y, lam=1e-6):
    """Solve output weights analytically: beta = (H'H + lam*I)^{-1} H'y."""
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


def predict(x, W, b, positions, kappa, beta):
    """Predict y = H @ beta for given input x."""
    H = build_feature_matrix(x, W, b, positions, kappa)
    return H @ beta


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def l2_relative(y_true, y_pred):
    """Relative L2 error: ||y_true - y_pred|| / ||y_true||."""
    denom = np.linalg.norm(y_true)
    if denom < 1e-15:
        return np.linalg.norm(y_true - y_pred)
    return np.linalg.norm(y_true - y_pred) / denom
