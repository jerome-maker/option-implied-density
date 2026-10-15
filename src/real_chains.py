"""Loader: turn a Yahoo Finance option-chain snapshot (one expiry) into an
out-of-the-money (OTM) call-equivalent quote set with bid/ask, as is standard
for American-style equity/ETF options in the RND literature (e.g. Figlewski
2010): OTM calls for K >= F, OTM puts for K < F converted to call-equivalent
prices by put-call parity. OTM options carry little early-exercise premium,
so parity introduces the least distortion on this subset.

Quote-noise proxy: a quote's price noise sd is taken as half-spread / sqrt(3)
(true value uniform within the spread).
"""
from __future__ import annotations

import json
import numpy as np

from .svj_model import SVJParams, bs_implied_vol


def load_otm_chain(path: str, r: float, q: float, window_sd: float = 4.0,
                   window_bounds=(0.05, 0.35), max_rel_spread: float = 0.5):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    res = j["optionChain"]["result"][0]
    quote = res["quote"]
    S0 = float(quote["regularMarketPrice"])
    t_quote = float(quote["regularMarketTime"])
    expiry = float(res["options"][0]["expirationDate"])
    # Expiry timestamp is 00:00 UTC of the expiry date; options settle at the
    # US close (~20:00 UTC), so add 20h.
    T = max((expiry + 20 * 3600 - t_quote) / (365.0 * 86400.0), 1.0 / 365.0)
    F = S0 * np.exp((r - q) * T)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)

    rows = []
    for side, opts in (("C", res["options"][0]["calls"]), ("P", res["options"][0]["puts"])):
        for o in opts:
            K = float(o["strike"])
            bid, ask = float(o.get("bid") or 0.0), float(o.get("ask") or 0.0)
            if not (bid > 0 and ask > bid):
                continue
            if (ask - bid) / (0.5 * (ask + bid)) > max_rel_spread:
                continue
            if side == "C" and K < F:
                continue
            if side == "P" and K >= F:
                continue
            shift = 0.0 if side == "C" else S0 * disc_q - K * disc_r   # put -> call-equivalent
            rows.append((K, bid + shift, ask + shift, side))
    rows.sort()
    K = np.array([r_[0] for r_ in rows])
    bid = np.array([r_[1] for r_ in rows])
    ask = np.array([r_[2] for r_ in rows])
    side = np.array([r_[3] for r_ in rows])
    mid = 0.5 * (bid + ask)
    x = np.log(K / S0)

    # Window scaled by ATM volatility so different maturities are comparable.
    iv_all = bs_implied_vol(np.clip(mid, 1e-8, None), S0, K, r, q, T)
    atm = float(iv_all[np.argmin(np.abs(np.log(K / F)))])
    half = float(np.clip(window_sd * atm * np.sqrt(T), *window_bounds))
    keep = np.abs(x) <= half
    p = SVJParams(scenario_id=path, S0=S0, r=r, q=q, T=T, v0=0, theta=0, kappa=0, sigma_V=0,
                  rho=0, lam=0, gamma=0, delta=0)
    data = {"x": x[keep], "K": K[keep], "C_obs": mid[keep], "bid": bid[keep], "ask": ask[keep],
            "side": side[keep], "iv_obs": iv_all[keep], "iv_true": iv_all[keep]}
    meta = {"S0": S0, "T": T, "F": F, "atm_iv": atm, "half_window": half, "n_quotes": int(keep.sum())}
    return p, data, meta


def price_from_density(p, K, x_grid, f_grid):
    """Discounted call prices implied by a log-return density (any method)."""
    f = np.clip(f_grid, 0.0, None)
    S = p.S0 * np.exp(x_grid)
    payoff = np.clip(S[None, :] - np.asarray(K)[:, None], 0.0, None)
    return np.exp(-p.r * p.T) * np.trapezoid(payoff * f[None, :], x_grid, axis=1)
