"""
The 12 SVJ benchmark scenarios (2 volatility levels x {No,Weak,Strong} SV x
{Frequent-Minor, Rare-Significant} jump patterns).

Common settings across all scenarios: T = 1/12, rho = -0.6.
Risk-free rate / dividend yield are simulation conveniences (not fitted to
market data, since the empirical purpose of this paper is purely the
PINN-vs-NN methodological comparison on controlled synthetic benchmarks).
"""
from __future__ import annotations

from .svj_model import SVJParams

S0 = 100.0
R = 0.03
Q = 0.01
T = 1.0 / 12.0

# Scenario ID -> (sigma_V, rho, lambda, gamma, delta, v0=theta, kappa)
_TABLE_2 = {
    "30-No-FM":     (0.00, -0.60, 3.00, -0.03, 0.0900, 0.0630, 1.50),
    "30-No-RS":     (0.00, -0.60, 0.50, -0.10, 0.2098, 0.0630, 1.50),
    "30-Weak-FM":   (0.25, -0.60, 3.00, -0.03, 0.0900, 0.0626, 1.50),
    "30-Weak-RS":   (0.25, -0.60, 0.50, -0.10, 0.2098, 0.0626, 1.50),
    "30-Strong-FM": (0.70, -0.60, 3.00, -0.03, 0.0900, 0.0619, 1.50),
    "30-Strong-RS": (0.70, -0.60, 0.50, -0.10, 0.2098, 0.0619, 1.50),
    "60-No-FM":     (0.00, -0.60, 8.00, -0.06, 0.0995, 0.2520, 1.20),
    "60-No-RS":     (0.00, -0.60, 1.00, -0.25, 0.2133, 0.2520, 1.20),
    "60-Weak-FM":   (0.45, -0.60, 8.00, -0.06, 0.0995, 0.2493, 1.20),
    "60-Weak-RS":   (0.45, -0.60, 1.00, -0.25, 0.2133, 0.2493, 1.20),
    "60-Strong-FM": (1.20, -0.60, 8.00, -0.06, 0.0995, 0.2447, 1.20),
    "60-Strong-RS": (1.20, -0.60, 1.00, -0.25, 0.2133, 0.2447, 1.20),
}


def get_scenario(scenario_id: str) -> SVJParams:
    sigma_V, rho, lam, gamma, delta, v0, kappa = _TABLE_2[scenario_id]
    return SVJParams(
        scenario_id=scenario_id,
        S0=S0, r=R, q=Q, T=T,
        v0=v0, theta=v0, kappa=kappa,
        sigma_V=sigma_V, rho=rho,
        lam=lam, gamma=gamma, delta=delta,
    )


def all_scenario_ids() -> list[str]:
    return list(_TABLE_2.keys())
