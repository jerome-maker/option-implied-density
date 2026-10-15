"""Merton (1976) jump-diffusion (MJD) model: pricing, log-return density,
and inverse-vega-weighted calibration to an observed implied-volatility
smile, used as the calibration/tail-completion model in the BL+MJD
construction (src/bl_mjd.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from .svj_model import bs_call_price, bs_implied_vol, bs_vega


@dataclass
class MJDParams:
    sigma_M: float
    lam_M: float
    gamma_M: float
    delta_M: float


def mjd_jump_compensator(p: MJDParams) -> float:
    return np.expm1(p.gamma_M + 0.5 * p.delta_M ** 2)


def mjd_call_price(K, S0: float, r: float, q: float, T: float, p: MJDParams, n_max: int = 50) -> np.ndarray:
    """Poisson mixture of Black-Scholes prices."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    zeta = mjd_jump_compensator(p)
    lamT = p.lam_M * T
    price = np.zeros_like(K)
    # Poisson weights p_n(lamT); truncate once the remaining tail mass is negligible.
    log_pn = -lamT
    pn = np.exp(log_pn)
    cum = 0.0
    for n in range(n_max):
        if n > 0:
            pn = pn * lamT / n
        cum += pn
        sigma_n = np.sqrt(p.sigma_M ** 2 + n * p.delta_M ** 2 / T)
        r_n = r - p.lam_M * zeta + n * (p.gamma_M + 0.5 * p.delta_M ** 2) / T
        c_bs = bs_call_price(S0, K, r_n, q, sigma_n, T)
        price += pn * np.exp(-(r - r_n) * T) * c_bs
        if cum > 1 - 1e-14 and n > 5:
            break
    return price


def mjd_log_return_density(x, r: float, q: float, T: float, p: MJDParams, n_max: int = 50):
    """Closed-form MJD log-return density f_M(x) as a Poisson-weighted
    mixture of Normal densities (conditional on the jump count), and its
    first derivative -- both in closed form, used for the tail-splice
    value/slope matching."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    zeta = mjd_jump_compensator(p)
    lamT = p.lam_M * T
    drift0 = (r - q - p.lam_M * zeta - 0.5 * p.sigma_M ** 2) * T
    f = np.zeros_like(x)
    fprime = np.zeros_like(x)
    log_pn = -lamT
    pn = np.exp(log_pn)
    cum = 0.0
    for n in range(n_max):
        if n > 0:
            pn = pn * lamT / n
        cum += pn
        mean_n = drift0 + n * p.gamma_M
        var_n = p.sigma_M ** 2 * T + n * p.delta_M ** 2
        sd_n = np.sqrt(var_n)
        phi = norm.pdf(x, loc=mean_n, scale=sd_n)
        f += pn * phi
        fprime += pn * phi * (-(x - mean_n) / var_n)
        if cum > 1 - 1e-14 and n > 5:
            break
    return f, fprime


def calibrate_mjd(K, iv_target, S0: float, r: float, q: float, T: float, n_restarts: int = 6, seed: int = 0) -> MJDParams:
    """Constrained nonlinear least-squares calibration in IV space with
    inverse-vega weighting, MultiStart over
    `n_restarts` random initial guesses; the lowest-weighted-SSE local
    solution is kept."""
    K = np.asarray(K, dtype=float)
    iv_target = np.asarray(iv_target, dtype=float)
    weights = 1.0 / np.maximum(bs_vega(S0, K, r, q, iv_target, T), 1e-8)

    lo = np.array([0.01, 1e-6, -1.0, 1e-3])
    hi = np.array([2.0, 50.0, 1.0, 2.0])

    def resid(theta):
        p = MJDParams(sigma_M=theta[0], lam_M=theta[1], gamma_M=theta[2], delta_M=theta[3])
        price = mjd_call_price(K, S0, r, q, T, p)
        iv = bs_implied_vol(np.clip(price, 1e-10, None), S0, K, r, q, T)
        return weights * (iv - iv_target)

    rng = np.random.default_rng(seed)
    best = None
    best_cost = np.inf
    x0_flat = np.array([0.20, 1.0, -0.05, 0.10])
    starts = [x0_flat] + [rng.uniform(lo * 1.2 + 1e-3, hi * 0.4) for _ in range(n_restarts - 1)]
    for x0 in starts:
        x0c = np.clip(x0, lo + 1e-6, hi - 1e-6)
        try:
            sol = least_squares(resid, x0c, bounds=(lo, hi), max_nfev=400)
        except Exception:
            continue
        cost = float(np.sum(sol.fun ** 2))
        if cost < best_cost:
            best_cost = cost
            best = sol.x
    if best is None:
        best = x0_flat
    return MJDParams(sigma_M=best[0], lam_M=best[1], gamma_M=best[2], delta_M=best[3])
