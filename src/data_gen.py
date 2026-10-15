"""
Synthetic training-data generation.

For a given SVJ benchmark scenario, sample strikes over a (possibly
limited) log-moneyness window, price them under the true SVJ model, and
add noise in implied-volatility space (a simple stand-in for bid-ask
spread / quote discreteness / microstructure noise) before converting
back to call prices.  This mirrors the "limited moneyness" of real option
markets: the network only ever sees
market-like quotes inside a bounded moneyness window, and must be
trusted to extrapolate sensibly (or not) outside it.
"""
from __future__ import annotations

import numpy as np

from .svj_model import SVJParams, true_density_and_grid, call_price_from_density, bs_implied_vol, bs_call_price


def make_training_data(
    p: SVJParams,
    moneyness_range: tuple[float, float] = (-0.2, 0.2),
    n_strikes: int = 25,
    iv_noise_std: float = 0.0025,
    seed: int = 0,
    x_grid=None,
    f_grid=None,
):
    """Return a dict of noisy training quotes over the given moneyness window."""
    rng = np.random.default_rng(seed)
    x = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
    K = p.S0 * np.exp(x)

    if x_grid is None or f_grid is None:
        x_grid, f_grid = true_density_and_grid(p)
    C_true = call_price_from_density(K, p, x_grid, f_grid)
    iv_true = bs_implied_vol(C_true, p.S0, K, p.r, p.q, p.T)

    iv_obs = iv_true + rng.normal(0.0, iv_noise_std, size=iv_true.shape)
    iv_obs = np.clip(iv_obs, 1e-4, None)
    C_obs = bs_call_price(p.S0, K, p.r, p.q, iv_obs, p.T)

    return {
        "x": x, "K": K,
        "C_true": C_true, "C_obs": C_obs,
        "iv_true": iv_true, "iv_obs": iv_obs,
    }


def make_eval_grid(p: SVJParams, full_range: tuple[float, float] = (-0.5, 0.5), n_points: int = 401):
    """Dense evaluation/collocation grid spanning a wider domain than training,
    used for (a) physics-loss collocation points during PINN training and
    (b) out-of-sample arbitrage / density evaluation for both models."""
    x_grid, f_grid = true_density_and_grid(p)
    x_eval = np.linspace(full_range[0], full_range[1], n_points)
    f_true_eval = np.interp(x_eval, x_grid, f_grid)
    return x_eval, f_true_eval, x_grid, f_grid
