"""BL+MJD risk-neutral density construction: nonparametric
Breeden-Litzenberger extraction (finite differences) in the central
moneyness region, spliced to exponentially-tilted Merton jump-diffusion
(MJD) tails at the boundary, with a final global exponential tilt
enforcing total-probability and martingale restrictions.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .mjd_model import MJDParams, calibrate_mjd, mjd_call_price, mjd_log_return_density
from .svj_model import bs_implied_vol


def bl_finite_difference(K: np.ndarray, C: np.ndarray, r: float, T: float):
    """Discrete Breeden-Litzenberger density at interior strikes via a
    three-point finite difference on a non-uniform grid; K, C must be sorted ascending by strike."""
    N = len(K)
    psi = np.full(N, np.nan)
    for i in range(1, N - 1):
        h_im1 = K[i] - K[i - 1]
        h_i = K[i + 1] - K[i]
        d2C = (2.0 / (h_im1 + h_i)) * ((C[i + 1] - C[i]) / h_i - (C[i] - C[i - 1]) / h_im1)
        psi[i] = np.exp(r * T) * d2C
    return psi  # NaN at the two endpoints (eq. 43's I_BL excludes them)


def bl_central_density(K: np.ndarray, C: np.ndarray, S0: float, r: float, T: float):
    """Interior (x_i, f_BL(x_i)) pairs."""
    psi = bl_finite_difference(K, C, r, T)
    interior = ~np.isnan(psi)
    K_i = K[interior]
    x_i = np.log(K_i / S0)
    f_i = K_i * psi[interior]
    return x_i, f_i


def _tail_tilt(x_splice, f_bl_val, f_bl_slope, r, q, T, mjd_p: MJDParams, floor: float = 1e-8):
    """Value/slope-matching tail tilt, returned as
    (log_alpha, beta) so the tail can be evaluated in log space.

    As in the original construction, beta is NOT clipped, however
    large the noisy finite-difference slope makes it (|beta| reaches the
    hundreds or thousands under quote noise -- that is the method's own
    behaviour and part of what the paper measures). The only guard is a
    floor on the splice-point density: the construction is
    undefined when the raw BL estimate there is <= 0, which happens on
    densely quoted real chains; the floor keeps it defined instead of
    raising, and only ever activates in that degenerate case."""
    f_m, f_m_prime = mjd_log_return_density(np.array([x_splice]), r, q, T, mjd_p)
    f_m, f_m_prime = max(float(f_m[0]), 1e-300), float(f_m_prime[0])
    f_bl_safe = max(f_bl_val, floor)
    beta = f_bl_slope / f_bl_safe - f_m_prime / f_m
    log_alpha = np.log(f_bl_safe) - beta * x_splice - np.log(f_m)
    return log_alpha, beta


def _tilted_tail(log_alpha, beta, x, f_m):
    """alpha * exp(beta x) * f_M(x), computed in log space (mathematically
    identical to the direct product; avoids overflow when |beta| is large)."""
    with np.errstate(divide="ignore"):
        log_f = log_alpha + beta * x + np.log(f_m)
    return np.exp(np.clip(log_f, -745.0, 700.0))


def _global_tilt(x_grid: np.ndarray, g: np.ndarray, r: float, q: float, T: float):
    target = np.exp((r - q) * T)

    def h(beta):
        num = np.trapezoid(np.exp((beta + 1) * x_grid) * g, x_grid)
        den = np.trapezoid(np.exp(beta * x_grid) * g, x_grid)
        return num / den - target

    lo, hi = -8.0, 8.0
    flo, fhi = h(lo), h(hi)
    tries = 0
    while flo * fhi > 0 and tries < 8:
        lo *= 1.6
        hi *= 1.6
        flo, fhi = h(lo), h(hi)
        tries += 1
    if flo * fhi > 0:
        beta = 0.0
    else:
        beta = brentq(h, lo, hi, xtol=1e-10, maxiter=200)
    alpha = 1.0 / np.trapezoid(np.exp(beta * x_grid) * g, x_grid)
    return alpha, beta


def fit_bl_mjd(p, data: dict, eval_range: tuple[float, float], n_grid: int = 401):
    """Fit the BL+MJD construction to `data` (same
    (x, K, C_obs, iv_obs) format used for the PINN/NN) and evaluate the
    resulting RND on a dense grid over `eval_range`.

    Returns a dict with the calibrated MJD parameters, the reconstructed
    density on the evaluation grid, and price/IV fit diagnostics on the
    training quotes using the calibrated MJD smile (MAPE in implied volatility).
    """
    K, C, iv_obs = data["K"], data["C_obs"], data["iv_obs"]
    order = np.argsort(K)
    K, C, iv_obs = K[order], C[order], iv_obs[order]

    mjd_p = calibrate_mjd(K, iv_obs, p.S0, p.r, p.q, p.T)

    x_i, f_i = bl_central_density(K, C, p.S0, p.r, p.T)
    f_bl_interp = PchipInterpolator(x_i, f_i, extrapolate=False)
    f_bl_deriv = f_bl_interp.derivative()
    x_min, x_max = x_i[0], x_i[-1]

    x_grid = np.linspace(eval_range[0], eval_range[1], n_grid)
    f_M_grid, _ = mjd_log_return_density(x_grid, p.r, p.q, p.T, mjd_p)

    f_bl_min, f_bl_min_slope = float(f_bl_interp(x_min)), float(f_bl_deriv(x_min))
    f_bl_max, f_bl_max_slope = float(f_bl_interp(x_max)), float(f_bl_deriv(x_max))
    la_m, beta_m = _tail_tilt(x_min, f_bl_min, f_bl_min_slope, p.r, p.q, p.T, mjd_p)
    la_p, beta_p = _tail_tilt(x_max, f_bl_max, f_bl_max_slope, p.r, p.q, p.T, mjd_p)

    g = np.empty_like(x_grid)
    left = x_grid < x_min
    right = x_grid > x_max
    mid = ~left & ~right
    g[left] = _tilted_tail(la_m, beta_m, x_grid[left], f_M_grid[left])
    g[right] = _tilted_tail(la_p, beta_p, x_grid[right], f_M_grid[right])
    g[mid] = f_bl_interp(x_grid[mid])

    alpha, beta = _global_tilt(x_grid, g, p.r, p.q, p.T)
    f_final = alpha * np.exp(beta * x_grid) * g

    mjd_price = mjd_call_price(K, p.S0, p.r, p.q, p.T, mjd_p)
    mjd_iv = bs_implied_vol(np.clip(mjd_price, 1e-10, None), p.S0, K, p.r, p.q, p.T)

    return {
        "mjd_params": mjd_p,
        "x_eval": x_grid, "f_eval": f_final,
        "x_min": x_min, "x_max": x_max,
        "mjd_price_on_data": mjd_price, "mjd_iv_on_data": mjd_iv,
        "price_mape": float(np.mean(np.abs(mjd_price - C) / np.abs(C))),
        "iv_mape_%": float(np.mean(np.abs(mjd_iv - iv_obs) / np.abs(iv_obs)) * 100.0),
    }
