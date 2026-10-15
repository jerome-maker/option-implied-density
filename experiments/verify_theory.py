"""Monte Carlo check of Proposition 1: the standard deviation of the
finite-difference BL density under implied-volatility quote noise.

For four scenarios, two strike counts and two noise levels, draws 2,000
noisy quote sets, computes the finite-difference density at each interior
strike, and compares its sample standard deviation with the closed form in
src/theory.py. Also reports the median predicted signal-to-noise ratio.

Output: results_wp3/tables/theory_verification.csv
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.scenarios import get_scenario
from src.svj_model import true_density_and_grid, call_price_from_density, bs_implied_vol, bs_call_price
from src.bl_mjd import bl_central_density
from src.theory import bl_fd_noise_sd

OUT = os.path.join(BASE, "results_wp3", "tables")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for sid in ("30-No-FM", "30-Strong-RS", "60-No-FM", "60-Strong-RS"):
        p = get_scenario(sid)
        xg, fg = true_density_and_grid(p)
        for n in (25, 97):
            x = np.linspace(-0.2, 0.2, n)
            K = p.S0 * np.exp(x)
            C = call_price_from_density(K, p, xg, fg)
            iv = bs_implied_vol(C, p.S0, K, p.r, p.q, p.T)
            for s in (0.001, 0.0025):
                F = np.array([bl_central_density(K, bs_call_price(p.S0, K, p.r, p.q, iv + rng.normal(0, s, n), p.T),
                                                 p.S0, p.r, p.T)[1] for _ in range(2000)])
                sd_theory = bl_fd_noise_sd(p, K, iv, s)
                ratio = F.std(0) / sd_theory
                rows.append(dict(scenario=sid, n_strikes=n, sigma_iv=s,
                                 ratio_median=np.median(ratio), ratio_min=ratio.min(), ratio_max=ratio.max(),
                                 snr_median=np.median(np.interp(x[1:-1], xg, fg) / sd_theory)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "theory_verification.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
