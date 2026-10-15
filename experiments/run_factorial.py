"""Factorial experiment: noise level x strike count x estimator.

  scenarios : 30-No-FM, 30-Strong-RS, 60-No-FM, 60-Strong-RS
  sigma_IV  : 0, 0.0005, 0.001, 0.0025, 0.005
  n strikes : 13, 25, 49, 97 (equally spaced in log-moneyness over [-0.2, 0.2])
  seeds     : 0..4
  methods   : PINN, Vanilla NN (GPU batched, src/batched.py);
              BL+MJD, SVI, PCA, Spline+GEV (CPU, process pool).

Each cell also records the theory-predicted median signal-to-noise ratio of
the finite-difference BL density (src/theory.py, Proposition (i)).

    python experiments/run_factorial.py gpu
    python experiments/run_factorial.py classical
Outputs: results_wp3/tables/factorial_{gpu,classical}.csv
"""
from __future__ import annotations

import os
import sys
import time
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scenarios import get_scenario
from src.data_gen import make_training_data, make_eval_grid
from src.evaluate import density_l1, arbitrage_violation_rate, _eval_range_for
from src.moments import truncated_moments
from src.svj_model import bs_vega
from src.theory import bl_fd_noise_sd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results_wp3", "tables")
os.makedirs(OUT, exist_ok=True)

SCENARIOS = ["30-No-FM", "30-Strong-RS", "60-No-FM", "60-Strong-RS"]
NOISES = [0.0, 0.0005, 0.001, 0.0025, 0.005]
NSTRIKES = [13, 25, 49, 97]
SEEDS = [0, 1, 2, 3, 4]
WINDOW = (-0.2, 0.2)


def cells():
    return list(itertools.product(SCENARIOS, NOISES, NSTRIKES, SEEDS))


def _truth(sid):
    p = get_scenario(sid)
    er = _eval_range_for(sid)
    xe, ft, xg, fg = make_eval_grid(p, full_range=er, n_points=401)
    return p, er, xe, ft, xg, fg


def snr_median(p, data, xg, fg, noise):
    if noise == 0:
        return np.inf
    sd = bl_fd_noise_sd(p, data["K"], data["iv_true"], noise)
    f_i = np.interp(data["x"][1:-1], xg, fg)
    return float(np.median(f_i / sd))


def score(xe, f, ft, true99):
    m = truncated_moments(xe, f, 0.99)
    return {"density_L1": density_l1(xe, f, ft), "violation_%": arbitrage_violation_rate(f) * 100,
            "var99_ratio": m["variance"] / true99["variance"],
            "skew_ok": bool(np.sign(m["skew"]) == np.sign(true99["skew"]))}


# ---------------------------------------------------------------- GPU part
def run_gpu():
    import torch
    from src.batched import train_batched, batched_density
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("預期要用 GPU 卻取到 CPU — 先檢查 torch 是不是 +cpu 建置")
    rows, t_all = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    groups = {}
    for sid, noise, n, seed in cells():
        groups.setdefault((_eval_range_for(sid), n), []).append((sid, noise, n, seed))
    for (er, n), members in groups.items():
        t0 = time.time()
        jobs = [(c, kind) for c in members for kind in ("PINN", "Vanilla NN")]
        X, Cn, lam, meta = [], [], [], []
        for (sid, noise, n_, seed), kind in jobs:
            p = get_scenario(sid)
            d = make_training_data(p, WINDOW, n_, iv_noise_std=noise, seed=seed)
            X.append(d["x"]); Cn.append(d["C_obs"] / p.S0)
            lam.append(1.0 if kind == "PINN" else 0.0)
            meta.append((sid, noise, n_, seed, kind, d))
        lam = np.array(lam)
        p0 = get_scenario(members[0][0])
        model = train_batched(np.array(X), np.array(Cn), 2.0 * lam, 2.0 * lam, lam, p0.r, p0.T, er,
                              seeds=range(len(jobs)), epochs=4000, device="cuda")
        truth = {}
        xe = np.linspace(er[0], er[1], 401)
        F = batched_density(model, xe, p0.r, p0.T)
        for b, (sid, noise, n_, seed, kind, d) in enumerate(meta):
            if sid not in truth:
                truth[sid] = _truth(sid)
            p, _, xe_, ft, xg, fg = truth[sid]
            true99 = truncated_moments(xe_, ft, 0.99)
            rows.append({"scenario": sid, "noise": noise, "n_strikes": n_, "seed": seed, "method": kind,
                         "snr_bl_median": snr_median(p, d, xg, fg, noise), **score(xe_, F[b], ft, true99)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "factorial_gpu.csv"), index=False)
        print(f"  group eval={er} n={n}: {len(jobs)} models in {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"GPU part done in {time.time()-t_all:.0f}s; peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


# ---------------------------------------------------------------- CPU part
def _classical_cell(cell):
    from src.bl_mjd import fit_bl_mjd
    from src.baselines import fit_svi, fit_pca, fit_spline_gev
    sid, noise, n, seed = cell
    p, er, xe, ft, xg, fg = _truth(sid)
    true99 = truncated_moments(xe, ft, 0.99)
    d = make_training_data(p, WINDOW, n, iv_noise_std=noise, seed=seed)
    snr = snr_median(p, d, xg, fg, noise)
    out = []
    for name, fn in (("BL+MJD", fit_bl_mjd), ("SVI", fit_svi), ("PCA", fit_pca), ("Spline+GEV", fit_spline_gev)):
        try:
            f = fn(p, d, er, n_grid=len(xe))["f_eval"]
            out.append({"scenario": sid, "noise": noise, "n_strikes": n, "seed": seed, "method": name,
                        "snr_bl_median": snr, **score(xe, f, ft, true99)})
        except Exception as e:  # record failures rather than silently dropping cells
            out.append({"scenario": sid, "noise": noise, "n_strikes": n, "seed": seed, "method": name,
                        "snr_bl_median": snr, "error": repr(e)})
    return out


def run_classical(workers):
    rows, t0 = [], time.time()
    todo = cells()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_classical_cell, todo, chunksize=2), 1):
            rows.extend(res)
            if i % 20 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "factorial_classical.csv"), index=False)
                print(f"  {i}/{len(todo)} cells, {time.time()-t0:.0f}s", flush=True)
    print(f"CPU part done in {time.time()-t0:.0f}s with {workers} worker processes", flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    if part == "gpu":
        run_gpu()
    else:
        run_classical(workers=max(1, min(6, (os.cpu_count() or 2) - 2)))
