"""SVI-anchored neural residual on the same 12 scenarios x {noisy, noiseless}
quotes as the horse race (identical data: seed=0), plus an ablation with the
physics penalties switched off (same SVI base and residual, lambda1=lambda2=
lambda3=0). Appends rows comparable to results_horserace/tables/horse_race_metrics.csv.

    python experiments/run_hybrid.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scenarios import get_scenario, all_scenario_ids
from src.data_gen import make_training_data, make_eval_grid
from src.baselines import fit_svi
from src.hybrid import SVIResidualPINN
from src.train import train_model, predict_prices
from src.evaluate import (model_density_on_grid, density_l1, arbitrage_violation_rate,
                          excess_tail_mass, _eval_range_for)
from src.moments import moment_table, P_LEVELS
from src.svj_model import bs_implied_vol
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(BASE, "results_horserace", "tables")
os.makedirs(TABLES, exist_ok=True)

WINDOW = (-0.2, 0.2)
LAMBDA_R = 0.1
VARIANTS = {
    "SVI+PINN": dict(lambda1=2.0, lambda2=2.0, lambda3=1.0),
    "SVI+NN (no physics)": dict(lambda1=0.0, lambda2=0.0, lambda3=0.0),
}


def main():
    t_start = time.time()
    records, moments = [], []
    for noise in (0.0025, 0.0):
        print(f"=== iv_noise_std={noise} ===", flush=True)
        for sid in all_scenario_ids():
            p = get_scenario(sid)
            er = _eval_range_for(sid)
            data = make_training_data(p, moneyness_range=WINDOW, n_strikes=25, iv_noise_std=noise, seed=0)
            xe, ft, _, _ = make_eval_grid(p, full_range=er, n_points=401)
            true_mom = moment_table(xe, ft)
            tail_true = excess_tail_mass(xe, ft, WINDOW)
            svi = fit_svi(p, data, er, n_grid=len(xe))
            line = []
            for name, lam in VARIANTS.items():
                t0 = time.time()
                m = SVIResidualPINN(svi["params"], p.r, p.q, p.T)
                train_model(m, p, data, colloc_range=er, lambda_r=LAMBDA_R, epochs=3000, lr=2e-3,
                            n_colloc=150, n_quad=200, seed=0, **lam)
                f = model_density_on_grid(m, xe, p.r, p.T)
                mom = moment_table(xe, f)
                price = predict_prices(m, p, data["x"])
                iv = bs_implied_vol(np.clip(price, 1e-10, None), p.S0, data["K"], p.r, p.q, p.T)
                records.append({
                    "scenario": sid, "noise": noise, "model": name,
                    "iv_mape_%": float(np.mean(np.abs(iv - data["iv_true"]) / data["iv_true"]) * 100),
                    "violation_%": arbitrage_violation_rate(f) * 100,
                    "density_L1": density_l1(xe, f, ft),
                    "tail_mass": excess_tail_mass(xe, f, WINDOW), "tail_mass_true": tail_true,
                    "var99_ratio": mom[0.99]["variance"] / true_mom[0.99]["variance"],
                    "skew99": mom[0.99]["skew"], "skew99_true": true_mom[0.99]["skew"],
                    "kurt99": mom[0.99]["kurt"], "kurt99_true": true_mom[0.99]["kurt"],
                    "train_seconds": time.time() - t0,
                })
                for p_level in P_LEVELS:
                    mm = mom[p_level]
                    moments.append({"scenario": sid, "noise": noise, "model": name, "p_level": p_level,
                                    **{k: mm[k] for k in ("mean", "variance", "skew", "kurt")}})
                line.append(f"{name}={records[-1]['density_L1']:.4f}")
            pd.DataFrame(records).to_csv(os.path.join(TABLES, "hybrid_metrics.csv"), index=False)
            pd.DataFrame(moments).to_csv(os.path.join(TABLES, "hybrid_moments.csv"), index=False)
            print(f"  {sid:14s} L1: " + "  ".join(line), flush=True)
    print(f"\nAll done in {time.time()-t_start:.1f}s (CPU)", flush=True)


if __name__ == "__main__":
    main()
