"""Real-data evaluation on 12 live option chains (SPY, QQQ, IWM x 15/29/64/92
days), six estimators, no ground truth.

Out-of-sample test: strikes are split into two interleaved folds; each method is
fit on one fold and the held-out OTM call-equivalent quotes are priced by
integrating that method's OWN density (src/real_chains.price_from_density).
Reported: share of held-out quotes priced inside [bid, ask] and the median
absolute error in half-spreads. Full-sample diagnostics: violation rate and
size, mass, 99%-truncated skew/kurtosis, mass outside the quote window, and the
theory-predicted median signal-to-noise ratio of the finite-difference BL
density (quote noise = half-spread / sqrt(3); signal = SVI density, a proxy).

    python experiments/run_wp4.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.real_chains import load_otm_chain, price_from_density
from src.bl_mjd import fit_bl_mjd
from src.baselines import fit_svi, fit_pca, fit_spline_gev
from src.moments import truncated_moments
from src.theory import fd_weights

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sorted(glob.glob(os.path.join(BASE, "data", "wp4_*")))[-1]
OUT = os.path.join(BASE, "results_wp4")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
DIV = {"SPY": 0.0073570693, "QQQ": 0.00242815, "IWM": 0.0074326564}   # trailing yields from the same snapshot
CLASSICAL = {"BL+MJD": fit_bl_mjd, "SVI": fit_svi, "PCA": fit_pca, "Spline+GEV": fit_spline_gev}
G = 601


def subset(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def oos_metrics(p, test, xg, f):
    pred = price_from_density(p, test["K"], xg, f)
    half = 0.5 * (test["ask"] - test["bid"])
    inside = (pred >= test["bid"] - 1e-9) & (pred <= test["ask"] + 1e-9)
    return float(inside.mean()), float(np.median(np.abs(pred - test["C_obs"]) / half))


def full_metrics(p, data, meta, xg, f):
    neg = f < -1e-8
    peak = max(f.max(), 1e-12)
    m = truncated_moments(xg, f, 0.99)
    out = np.abs(xg) > meta["half_window"]
    pred = price_from_density(p, data["K"], xg, f)
    inside = (pred >= data["bid"] - 1e-9) & (pred <= data["ask"] + 1e-9)
    return {"violation_%": float(neg.mean() * 100),
            "worst_neg_rel_peak": float(-f.min() / peak) if neg.any() else 0.0,
            "mass": float(np.trapezoid(np.clip(f, 0, None), xg)),
            "skew99": m["skew"], "kurt99": m["kurt"],
            "mass_outside_window": float(np.trapezoid(np.where(out, np.clip(f, 0, None), 0.0), xg)),
            "insample_inside_%": float(inside.mean() * 100)}


def main():
    import torch
    from src.batched import train_batched_ragged, batched_density_ragged
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("預期要用 GPU 卻取到 CPU — 先檢查 torch 是不是 +cpu 建置")
    r = json.load(open(os.path.join(DATA, "rates.json")))["irx_pct"] / 100.0
    chains = []
    for fn in sorted(glob.glob(os.path.join(DATA, "*_*d.json"))):
        tic = os.path.basename(fn)[:3]
        p, data, meta = load_otm_chain(fn, r, DIV[tic])
        half_eval = float(min(0.8, max(0.3, 8 * meta["atm_iv"] * np.sqrt(p.T))))
        idx = np.arange(meta["n_quotes"])
        folds = {"full": idx, "A": idx[0::2], "B": idx[1::2]}
        chains.append(dict(name=os.path.basename(fn)[:-5], p=p, data=data, meta=meta,
                           er=(-half_eval, half_eval), xg=np.linspace(-half_eval, half_eval, G), folds=folds))
    print(f"{len(chains)} chains from {DATA}; r={r:.4f}", flush=True)

    # ---- GPU: PINN and vanilla NN for every chain x fold, one ragged batch ----
    t0 = time.time()
    jobs = [(c, fold, kind) for c in chains for fold in ("full", "A", "B") for kind in ("PINN", "Vanilla NN")]
    torch.cuda.reset_peak_memory_stats()
    model = train_batched_ragged(
        [c["data"]["x"][c["folds"][fold]] for c, fold, _ in jobs],
        [c["data"]["C_obs"][c["folds"][fold]] / c["p"].S0 for c, fold, _ in jobs],
        [1.0 if kind == "PINN" else 0.0 for _, _, kind in jobs],
        [c["er"] for c, _, _ in jobs], r, [c["p"].T for c, _, _ in jobs],
        seeds=range(len(jobs)), epochs=4000, device="cuda")
    F = batched_density_ragged(model, np.stack([c["xg"] for c, _, _ in jobs]), r, [c["p"].T for c, _, _ in jobs])
    gpu_s, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 20
    del model
    torch.cuda.empty_cache()
    dens = {(c["name"], fold, kind): F[b] for b, (c, fold, kind) in enumerate(jobs)}
    print(f"GPU: {len(jobs)} networks in {gpu_s:.0f}s, peak VRAM {peak:.0f} MiB", flush=True)

    # ---- CPU: classical estimators (sequential optimisers) ----
    t1 = time.time()
    fails = []
    for c in chains:
        for fold in ("full", "A", "B"):
            sub = subset(c["data"], c["folds"][fold])
            for name, fn in CLASSICAL.items():
                try:
                    dens[(c["name"], fold, name)] = fn(c["p"], sub, c["er"], n_grid=G)["f_eval"]
                except Exception as e:
                    fails.append((c["name"], fold, name, repr(e)[:80]))
        print(f"  {c['name']} classical done ({time.time()-t1:.0f}s)", flush=True)
    cpu_s = time.time() - t1

    # ---- metrics ----
    rows = []
    methods = ["PINN", "Vanilla NN"] + list(CLASSICAL)
    for c in chains:
        p, data, meta, xg = c["p"], c["data"], c["meta"], c["xg"]
        # theory SNR of finite-difference BL on this chain (noise = half-spread/sqrt(3); signal = SVI proxy)
        s = 0.5 * (data["ask"] - data["bid"]) / np.sqrt(3.0)
        A = fd_weights(data["K"])
        sd = data["K"][1:-1] * np.exp(p.r * p.T) * np.sqrt(np.sum(A ** 2 * np.stack([s[:-2], s[1:-1], s[2:]], 1) ** 2, 1))
        snr = float(np.median(np.interp(data["x"][1:-1], xg, dens[(c["name"], "full", "SVI")]) / sd)) \
            if (c["name"], "full", "SVI") in dens else np.nan
        for m in methods:
            row = {"chain": c["name"], "ticker": c["name"][:3], "days": round(p.T * 365, 1), "method": m,
                   "n_quotes": meta["n_quotes"], "atm_iv": meta["atm_iv"], "bl_snr_proxy": snr}
            if (c["name"], "full", m) in dens:
                row.update(full_metrics(p, data, meta, xg, dens[(c["name"], "full", m)]))
            ins, err = [], []
            for fit_fold, test_fold in (("A", "B"), ("B", "A")):
                if (c["name"], fit_fold, m) in dens:
                    a, e = oos_metrics(p, subset(data, c["folds"][test_fold]), xg, dens[(c["name"], fit_fold, m)])
                    ins.append(a); err.append(e)
            row["oos_inside_%"] = float(np.mean(ins) * 100) if ins else np.nan
            row["oos_err_halfspreads"] = float(np.mean(err)) if err else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tables", "wp4_metrics.csv"), index=False)
    pd.DataFrame(fails, columns=["chain", "fold", "method", "error"]).to_csv(os.path.join(OUT, "tables", "wp4_failures.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "tables", "wp4_densities.npz"),
                        **{f"{k[0]}|{k[1]}|{k[2]}": v for k, v in dens.items()},
                        **{f"grid|{c['name']}": c["xg"] for c in chains})
    print(f"CPU classical: {cpu_s:.0f}s; failures: {len(fails)}", flush=True)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
