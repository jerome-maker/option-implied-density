"""Theory: noise amplification of the finite-difference Breeden-Litzenberger
density estimator.

Setting. Observed call prices C_j = C(K_j) + e_j at strikes K_1 < ... < K_N, with
independent errors of standard deviation s_j. Quoting noise is in implied
volatility (s_IV), so by the delta method s_j ~= vega_j * s_IV.

Estimator. At an interior strike K_i, with h_{i-1} = K_i - K_{i-1}, h_i = K_{i+1} - K_i,

    psi_hat(K_i) = e^{rT} * sum_j a_j C_j,
    a_{i-1} = 2 / ((h_{i-1}+h_i) h_{i-1}),  a_i = -2 / (h_{i-1} h_i),  a_{i+1} = 2 / ((h_{i-1}+h_i) h_i),

and the log-return density estimate is f_hat(x_i) = K_i psi_hat(K_i).

Proposition (noise and bias).
  (i)  sd[f_hat(x_i)] = K_i e^{rT} sqrt( sum_j a_j^2 s_j^2 ).
       With equal spacing h and locally constant s: sqrt(6) K_i e^{rT} s / h^2.
  (ii) bias[f_hat(x_i)] = K_i e^{rT} (h^2/12) C''''(K_i) + O(h^3)   (equal spacing).
  (iii) The MSE-optimal spacing is h* = (864 s^2 / C''''(K)^2)^{1/8}, and the
       signal-to-noise ratio f / sd falls below one once
       h < h_c = ( sqrt(6) K e^{rT} s / f(x) )^{1/2}.
  The noise term grows as h^{-2}: halving the strike spacing quadruples the
  noise standard deviation of every density estimate built this way.
"""
from __future__ import annotations

import numpy as np

from .svj_model import bs_vega


def fd_weights(K: np.ndarray):
    """Three-point non-uniform second-difference weights at interior strikes:
    returns array A of shape (N-2, 3) with columns (a_{i-1}, a_i, a_{i+1})."""
    h_m = K[1:-1] - K[:-2]
    h_p = K[2:] - K[1:-1]
    a_m = 2.0 / ((h_m + h_p) * h_m)
    a_0 = -2.0 / (h_m * h_p)
    a_p = 2.0 / ((h_m + h_p) * h_p)
    return np.stack([a_m, a_0, a_p], axis=1)


def bl_fd_noise_sd(p, K: np.ndarray, iv: np.ndarray, sigma_iv: float) -> np.ndarray:
    """Proposition (i): sd of the finite-difference log-return density at the
    interior strikes K[1:-1], under IV quote noise of sd sigma_iv."""
    s = bs_vega(p.S0, K, p.r, p.q, iv, p.T) * sigma_iv
    A = fd_weights(K)
    s_stack = np.stack([s[:-2], s[1:-1], s[2:]], axis=1)
    return K[1:-1] * np.exp(p.r * p.T) * np.sqrt(np.sum(A ** 2 * s_stack ** 2, axis=1))


def bl_fd_noise_sd_uniform(p, K_i, h, vega_i, sigma_iv):
    """Equal-spacing form of Proposition (i): sqrt(6) K e^{rT} vega s_IV / h^2."""
    return np.sqrt(6.0) * K_i * np.exp(p.r * p.T) * vega_i * sigma_iv / h ** 2


def critical_spacing(p, K_i, f_i, vega_i, sigma_iv):
    """Proposition (iii): spacing below which noise sd exceeds the density itself."""
    return np.sqrt(np.sqrt(6.0) * K_i * np.exp(p.r * p.T) * vega_i * sigma_iv / f_i)


def mse_optimal_spacing(vega_i, sigma_iv, c4):
    """Proposition (iii): h* = (864 s^2 / C''''^2)^{1/8}."""
    s = vega_i * sigma_iv
    return (864.0 * s ** 2 / c4 ** 2) ** 0.125
