"""Truncated moments of a risk-neutral density, computed on the central p-probability window.

For a central probability level p in (0, 1], let a_p = Q((1-p)/2) and
b_p = Q((1+p)/2) be the corresponding quantiles (Q = inverse CDF). The
truncated density is f_p(x) = f(x) * 1{a_p <= x <= b_p}, and the
truncated mean/variance/skewness/kurtosis are computed by integrating
against f_p directly (i.e. NOT renormalized to integrate to one over the
truncated window) -- so that
p = 100% recovers the ordinary moments of f.
"""
from __future__ import annotations

import numpy as np


def cdf_from_density(x_grid: np.ndarray, f_grid: np.ndarray):
    """Monotone CDF on x_grid from a (possibly slightly negative, e.g. from
    an unconstrained NN) density -- clipped at zero and renormalised so
    that a well-defined quantile function exists."""
    f_clip = np.clip(f_grid, 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f_clip[1:] + f_clip[:-1]) * np.diff(x_grid))])
    total = cdf[-1]
    if total <= 0:
        raise ValueError("Density integrates to a non-positive mass; cannot form a CDF.")
    return f_clip, cdf / total


def quantile(x_grid: np.ndarray, cdf: np.ndarray, u) -> np.ndarray:
    return np.interp(u, cdf, x_grid)


def truncated_moments(x_grid: np.ndarray, f_grid: np.ndarray, p: float) -> dict:
    """Return {mean, variance, skew, kurt} truncated to the central-p region."""
    f_clip, cdf = cdf_from_density(x_grid, f_grid)
    lo_q, hi_q = (1 - p) / 2, (1 + p) / 2
    a_p, b_p = quantile(x_grid, cdf, [lo_q, hi_q])

    mask = (x_grid >= a_p) & (x_grid <= b_p)
    f_trunc = np.where(mask, f_clip, 0.0)

    mu = np.trapezoid(x_grid * f_trunc, x_grid)
    var = np.trapezoid((x_grid - mu) ** 2 * f_trunc, x_grid)
    sd = np.sqrt(max(var, 1e-14))
    skew = np.trapezoid((x_grid - mu) ** 3 * f_trunc, x_grid) / sd ** 3
    kurt = np.trapezoid((x_grid - mu) ** 4 * f_trunc, x_grid) / sd ** 4
    return {"mean": mu, "variance": var, "skew": skew, "kurt": kurt, "a_p": a_p, "b_p": b_p}


P_LEVELS = (1.00, 0.995, 0.99, 0.95, 0.90)


def moment_table(x_grid: np.ndarray, f_grid: np.ndarray, p_levels=P_LEVELS) -> dict:
    return {p: truncated_moments(x_grid, f_grid, p) for p in p_levels}
