"""Horse race: six RND estimators x 12 SVJ scenarios x {noisy, noiseless}
quotes, narrow [-0.2, 0.2] window. Writes to results_horserace/ only --
nothing the manuscript currently cites is touched.

Run:
    python experiments/run_horse_race.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scenarios import get_scenario, all_scenario_ids
from src.evaluate import run_horse_race

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("RND_HORSERACE_DIR", os.path.join(BASE, "results_horserace"))
TABLES, FIGS = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

NOISE_LEVELS = (0.0025, 0.0)
REPRESENTATIVE = {"30-No-FM", "30-Strong-RS", "60-No-FM", "60-Strong-RS"}
STYLE = {
    "PINN": dict(color="#1b6ca8", lw=1.5),
    "Vanilla NN": dict(color="#c0392b", lw=1.1, ls="--"),
    "BL+MJD": dict(color="#27803a", lw=1.1, ls=":"),
    "SVI": dict(color="#8e44ad", lw=1.2),
    "PCA": dict(color="#d68910", lw=1.2, ls="-."),
    "Spline+GEV": dict(color="#5d6d7e", lw=1.2, ls=(0, (5, 1))),
}


def plot_all(res, path, log=True):
    base = res["base"]
    x, ft = base["x_eval"], base["f_true_eval"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axvspan(*base["moneyness_range"], color="0.92", zorder=0, label="calibration window")
    ax.plot(x, np.clip(ft, 1e-8, None) if log else ft, color="k", lw=1.8, label="True SVJ")
    for name, f in res["densities"].items():
        ax.plot(x, np.clip(f, 1e-8, None) if log else f, label=name, **STYLE[name])
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-6, None)
    ax.set_xlabel("x = log(S_T/S_0)")
    ax.set_ylabel("Risk-neutral density" + (" (log scale)" if log else ""))
    ax.set_title(base["scenario_id"])
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    t_start = time.time()
    records, moments = [], []
    for noise in NOISE_LEVELS:
        tag = "noisy" if noise > 0 else "noiseless"
        print(f"=== iv_noise_std={noise} ({tag}) ===", flush=True)
        for sid in all_scenario_ids():
            t0 = time.time()
            res = run_horse_race(get_scenario(sid), iv_noise_std=noise, seed=0)
            records.extend(res["records"])
            moments.extend(res["moment_records"])
            pd.DataFrame(records).to_csv(os.path.join(TABLES, "horse_race_metrics.csv"), index=False)
            pd.DataFrame(moments).to_csv(os.path.join(TABLES, "horse_race_moments.csv"), index=False)
            l1 = {r["model"]: r["density_L1"] for r in res["records"]}
            print(f"  {sid:14s} {time.time()-t0:5.1f}s  L1: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in l1.items()), flush=True)
            if sid in REPRESENTATIVE:
                plot_all(res, os.path.join(FIGS, f"horse_log_{tag}_{sid}.pdf"), log=True)
                plot_all(res, os.path.join(FIGS, f"horse_lin_{tag}_{sid}.pdf"), log=False)
    print(f"\nAll done in {time.time()-t_start:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
