"""Phase diagrams from results_wp3/tables/factorial_{gpu,classical}.csv.

For each volatility group (30% / 60%) and each estimator: a heat map of the
median density L1 error over (sigma_IV x number of strikes), cells where the
theory-predicted median BL signal-to-noise ratio is below one hatched. Also a
"best method" map and a tidy summary table.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(BASE, "results_wp3", "tables")
FIG = os.path.join(BASE, "results_wp3", "figures")
os.makedirs(FIG, exist_ok=True)
METHODS = ["BL+MJD", "SVI", "PCA", "Spline+GEV", "PINN", "Vanilla NN"]


def load():
    parts = [pd.read_csv(os.path.join(TAB, f)) for f in ("factorial_gpu.csv", "factorial_classical.csv")
             if os.path.exists(os.path.join(TAB, f))]
    d = pd.concat(parts, ignore_index=True)
    d["group"] = np.where(d.scenario.str.startswith("30-"), "30% vol", "60% vol")
    return d


def main():
    d = load()
    d["failed"] = d["error"].notna() if "error" in d.columns else False
    fail = d.groupby(["group", "method", "noise", "n_strikes"]).failed.mean().reset_index()
    print(f"{int(d.failed.sum())} failed fits (excluded from medians, shown as fail-% in the BL+MJD panel)")
    d = d[~d.failed]
    noises = sorted(d.noise.unique())
    ns = sorted(d.n_strikes.unique())
    summary = (d.groupby(["group", "method", "noise", "n_strikes"])
                 .agg(L1_median=("density_L1", "median"), viol_mean=("violation_%", "mean"),
                      var_ratio_median=("var99_ratio", "median"), skew_ok_share=("skew_ok", "mean"),
                      snr_bl_median=("snr_bl_median", "median"))
                 .reset_index())
    summary.to_csv(os.path.join(TAB, "factorial_summary.csv"), index=False)

    for grp in summary.group.unique():
        s = summary[summary.group == grp]
        vmin = max(s.L1_median.min(), 1e-3)
        vmax = s.L1_median.max()
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
        snr = s[s.method == "BL+MJD"].pivot(index="noise", columns="n_strikes", values="snr_bl_median").loc[noises, ns]
        for ax, m in zip(axes.ravel(), METHODS):
            g = s[s.method == m].pivot(index="noise", columns="n_strikes", values="L1_median").reindex(index=noises, columns=ns)
            im = ax.imshow(g.values, origin="lower", cmap="viridis_r", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
            for i in range(len(noises)):
                for j in range(len(ns)):
                    v = g.values[i, j]
                    fs = fail[(fail.group == grp) & (fail.method == m) & (fail.noise == noises[i]) & (fail.n_strikes == ns[j])].failed
                    fs = float(fs.iloc[0]) if len(fs) else 0.0
                    if np.isfinite(v):
                        label = f"{v:.3f}" + (f"\nfail {fs*100:.0f}%" if fs > 0 else "")
                        ax.text(j, i, label, ha="center", va="center", fontsize=6.5 if fs > 0 else 7,
                                color="white" if v > np.sqrt(vmin * vmax) else "black")
                    if np.isfinite(snr.values[i, j]) and snr.values[i, j] < 1:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, ec="red", alpha=0.6))
            ax.set_title(m, fontsize=10)
            ax.set_xticks(range(len(ns)), [str(n) for n in ns])
            ax.set_yticks(range(len(noises)), [f"{v:g}" for v in noises])
        for ax in axes[-1]:
            ax.set_xlabel("number of strikes in [-0.2, 0.2]")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"IV noise $\sigma_{IV}$")
        fig.colorbar(im, ax=axes, shrink=0.8, label="median density L1 error (log scale)")
        fig.suptitle(f"Phase diagram, {grp}: median L1 error over successful fits (fail % = share where the estimator returned no density); red hatching = theory-predicted BL SNR < 1", fontsize=9.5)
        fig.savefig(os.path.join(FIG, f"phase_L1_{grp.split('%')[0]}.pdf"), bbox_inches="tight")
        plt.close(fig)

        best = s.loc[s.groupby(["noise", "n_strikes"]).L1_median.idxmin()]
        bmap = best.pivot(index="noise", columns="n_strikes", values="method").reindex(index=noises, columns=ns)
        codes = {m: i for i, m in enumerate(METHODS)}
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.imshow(bmap.replace(codes).astype(float).values, origin="lower", cmap="tab10", vmin=0, vmax=9, aspect="auto")
        for i in range(len(noises)):
            for j in range(len(ns)):
                ax.text(j, i, bmap.values[i, j], ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns]); ax.set_yticks(range(len(noises)), [f"{v:g}" for v in noises])
        ax.set_xlabel("number of strikes"); ax.set_ylabel(r"$\sigma_{IV}$")
        ax.set_title(f"Lowest median L1 error, {grp}", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"best_method_{grp.split('%')[0]}.pdf")); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
