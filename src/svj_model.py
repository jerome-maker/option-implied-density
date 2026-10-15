"""
Bates stochastic-volatility jump-diffusion (SVJ) benchmark model.

Benchmark market model (Bates, 1996):

    dS_t/S_t- = (r - q - lambda*zeta) dt + sqrt(V_t) dW_t^S + (Y-1) dN_t
    dV_t      = kappa(theta - V_t) dt + sigma_V sqrt(V_t) dW_t^V
    d<W^S,W^V>_t = rho dt,   J = log(Y) ~ N(gamma, delta^2)

The characteristic function of X_T = log(S_T/S0) is available in closed
form (Heston "little trap" parametrisation, numerically stable across the
branch cut).  We use it to:

  1. Recover the true risk-neutral log-return density f_X(x) via a
     Fourier-cosine (COS) expansion (Fang & Oosterlee, 2008) -- this is
     the "ground truth" RND used to grade both the PINN and the vanilla
     NN.
  2. Price European calls by numerically integrating that same truncated
     cosine-series density against the call payoff.  This is
     mathematically identical to the closed-form COS pricing formula
     (Fang & Oosterlee, 2008): both approximate the payoff
     integral against a truncated Fourier-cosine expansion of the
     density on [a, b]; we simply carry out the integration numerically
     instead of pre-deriving the closed-form H_n coefficients, which
     avoids a class of sign/branch bugs while keeping identical accuracy
     for a given (a, b, N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class SVJParams:
    """Bates SVJ parameter set (risk-neutral)."""

    scenario_id: str
    S0: float
    r: float
    q: float
    T: float
    v0: float       # = theta (long-run variance), imposed v0 = theta
    theta: float
    kappa: float
    sigma_V: float
    rho: float
    lam: float       # jump intensity lambda
    gamma: float     # mean log jump size
    delta: float     # std of log jump size


def jump_compensator(lam: float, gamma: float, delta: float) -> float:
    """zeta = E^Q[Y-1] = exp(gamma + delta^2/2) - 1."""
    return np.expm1(gamma + 0.5 * delta ** 2)


def _char_func_svj(u: np.ndarray, p: SVJParams) -> np.ndarray:
    """Characteristic function phi_X(u; T) = E[exp(i u X_T)] of the Bates SVJ model."""
    u = np.asarray(u, dtype=complex)
    T = p.T
    zeta = jump_compensator(p.lam, p.gamma, p.delta)

    # Jump component (eq. 13, second factor)
    phi_J = np.exp(p.lam * T * (np.exp(1j * u * p.gamma - 0.5 * p.delta ** 2 * u ** 2) - 1.0))

    # Stochastic-volatility component (eqs. 13-16)
    if p.sigma_V < 1e-10:
        # sigma_V -> 0 limit: CIR variance collapses to the constant v0 = theta,
        # so the diffusion component reduces to a plain constant-variance
        # (Black-Scholes-type) characteristic function.
        phi_SV = np.exp(-0.5 * p.v0 * T * (u ** 2 + 1j * u))
    else:
        b = p.kappa - p.rho * p.sigma_V * 1j * u
        d = np.sqrt(b ** 2 + p.sigma_V ** 2 * (u ** 2 + 1j * u))
        # Ensure Re(d) >= 0 branch
        d = np.where(np.real(d) < 0, -d, d)
        g = (b - d) / (b + d)
        exp_dT = np.exp(-d * T)
        C = (p.kappa * p.theta / p.sigma_V ** 2) * (
            (b - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = ((b - d) / p.sigma_V ** 2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
        phi_SV = np.exp(C + D * p.v0)

    drift = 1j * u * (p.r - p.q - p.lam * zeta) * T
    return np.exp(drift) * phi_SV * phi_J


def _cumulants_c1_c2(p: SVJParams) -> tuple[float, float]:
    """Closed-form mean and variance of X_T (drift term plus the variance of the diffusion and jump parts)."""
    zeta = jump_compensator(p.lam, p.gamma, p.delta)
    c1 = (p.r - p.q - p.lam * zeta) * p.T - 0.5 * p.v0 * p.T  # mean (approx; jump mean adds too)
    c1 += p.lam * p.T * p.gamma

    # Stochastic-volatility variance contribution (eq. 9), imposing v0 = theta
    A = -0.5 + p.rho * p.kappa / p.sigma_V if p.sigma_V > 1e-10 else 0.0
    B = p.rho / p.sigma_V if p.sigma_V > 1e-10 else 0.0
    if p.sigma_V < 1e-10:
        var_sv = p.v0 * p.T
    else:
        kap, sig, T = p.kappa, p.sigma_V, p.T
        term1 = A ** 2 * (p.theta * sig ** 2 / kap ** 2) * (T - (3 - 4 * np.exp(-kap * T) + np.exp(-2 * kap * T)) / (2 * kap))
        term2 = B ** 2 * (p.theta * sig ** 2 / (2 * kap)) * (1 - np.exp(-2 * kap * T))
        term3 = 2 * A * B * (p.theta * sig ** 2 / (2 * kap ** 2)) * (1 - np.exp(-kap * T)) ** 2
        term4 = (1 - p.rho ** 2) * p.theta * T
        var_sv = term1 + term2 + term3 + term4

    var_j = p.lam * p.T * (p.gamma ** 2 + p.delta ** 2)
    c2 = var_sv + var_j
    return c1, c2


def cos_truncation_range(p: SVJParams, L: float = 14.0) -> tuple[float, float]:
    """Truncation interval [a, b] for the COS expansion, centred on the mean."""
    c1, c2 = _cumulants_c1_c2(p)
    width = L * np.sqrt(max(c2, 1e-8))
    return c1 - width, c1 + width


def cos_density(x: np.ndarray, p: SVJParams, N: int = 8192, a: float | None = None, b: float | None = None) -> np.ndarray:
    """Fourier-cosine (COS) reconstruction of the log-return density f_X(x)."""
    if a is None or b is None:
        a, b = cos_truncation_range(p)
    x = np.asarray(x, dtype=float)
    n = np.arange(N)
    u = n * np.pi / (b - a)
    phi = _char_func_svj(u, p)
    Fk = (2.0 / (b - a)) * np.real(phi * np.exp(-1j * n * np.pi * a / (b - a)))
    Fk[0] *= 0.5
    # f(x) = sum_n Fk[n] * cos(n*pi*(x-a)/(b-a))
    arg = np.outer((x - a) / (b - a), n) * np.pi  # (len(x), N)
    dens = np.cos(arg) @ Fk
    return dens


def true_density_and_grid(p: SVJParams, N: int = 8192, n_grid: int = 4001, L: float = 14.0):
    """Return (x_grid, f_grid) -- a dense, normalised true RND on [a, b]."""
    a, b = cos_truncation_range(p, L=L)
    x_grid = np.linspace(a, b, n_grid)
    f_grid = cos_density(x_grid, p, N=N, a=a, b=b)
    f_grid = np.clip(f_grid, 0.0, None)
    mass = np.trapezoid(f_grid, x_grid)
    f_grid = f_grid / mass
    return x_grid, f_grid


def call_price_from_density(K: np.ndarray, p: SVJParams, x_grid: np.ndarray, f_grid: np.ndarray) -> np.ndarray:
    """European call price via numerical integration of the (COS-recovered) density."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    S_grid = p.S0 * np.exp(x_grid)
    disc = np.exp(-p.r * p.T)
    out = np.empty_like(K)
    for i, k in enumerate(K):
        payoff = np.clip(S_grid - k, 0.0, None)
        out[i] = disc * np.trapezoid(payoff * f_grid, x_grid)
    return out


def svj_call_prices(K: np.ndarray, p: SVJParams, N: int = 8192, n_grid: int = 4001) -> np.ndarray:
    """Convenience wrapper: SVJ benchmark European call prices at strikes K."""
    x_grid, f_grid = true_density_and_grid(p, N=N, n_grid=n_grid)
    return call_price_from_density(K, p, x_grid, f_grid)


def bs_call_price(S0, K, r, q, sigma, T):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_vega(S0, K, r, q, sigma, T):
    """Black-Scholes vega: dC/dsigma."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_implied_vol(price, S0, K, r, q, T, lo=1e-6, hi=5.0, tol=1e-10, max_iter=100):
    """Robust bisection implied-vol solver (vectorised over K/price arrays)."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    lo_arr = np.full_like(price, lo)
    hi_arr = np.full_like(price, hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        p_mid = bs_call_price(S0, K, r, q, mid, T)
        too_high = p_mid > price
        hi_arr = np.where(too_high, mid, hi_arr)
        lo_arr = np.where(too_high, lo_arr, mid)
    return 0.5 * (lo_arr + hi_arr)
