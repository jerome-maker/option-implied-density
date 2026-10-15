"""Build every table (.tex) and copy every figure used by the manuscript
from the result files, so no number in the paper is hand-copied.

Inputs : results_horserace/, results_wp3/, results_wp4/
Outputs: paper_assets/tables/*.tex, paper_assets/figures/*.pdf
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(BASE, "paper_assets", "tables")
FIG = os.path.join(BASE, "paper_assets", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SIM8 = ["BL+MJD", "SVI", "PCA", "Spline+GEV", "Vanilla NN", "PINN", "SVI+NN (no physics)", "SVI+PINN"]
LABEL = {"SVI+NN (no physics)": "SVI+NN", "SVI+PINN": "SVI+PINN"}
REAL6 = ["BL+MJD", "SVI", "PCA", "Spline+GEV", "Vanilla NN", "PINN"]


def lab(m):
    return LABEL.get(m, m)


def write(name, lines):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


def f3(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def f2(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


# ------------------------------------------------------------------ simulation horse race
def horse():
    h = pd.read_csv(os.path.join(BASE, "results_horserace", "tables", "horse_race_metrics.csv"))
    y = pd.read_csv(os.path.join(BASE, "results_horserace", "tables", "hybrid_metrics.csv")).drop(columns="train_seconds")
    d = pd.concat([h, y], ignore_index=True)
    d["skew_ok"] = np.sign(d.skew99) == np.sign(d.skew99_true)
    d["var_err"] = (d.var99_ratio - 1).abs()
    d["kurt_err"] = (d.kurt99 - d.kurt99_true).abs()
    return d


def table_horse(d):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Simulation horse race: eight estimators on the twelve SVJ scenarios (narrow $[-0.2,0.2]$ window, 25 strikes), "
         r"with and without implied-volatility quote noise ($\sigma_{IV}=0.0025$). L1 is the integrated absolute density error "
         r"(maximum 2); violation is the share of a dense grid with negative density; variance and kurtosis errors are for the "
         r"99\%-truncated moments, medians over scenarios; skew sign counts scenarios with the correct sign out of 12.}",
         r"\label{tab:horse}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccc|ccc}", r"\toprule",
         r" & \multicolumn{5}{c|}{Noisy quotes ($\sigma_{IV}=0.0025$)} & \multicolumn{3}{c}{Noiseless quotes} \\",
         r"Estimator & Mean L1 & Violation (\%) & $|\mathrm{Var}/\mathrm{Var}^*-1|$ & $|\mathrm{Kurt}-\mathrm{Kurt}^*|$ & Skew sign & Mean L1 & Violation (\%) & $|\mathrm{Var}/\mathrm{Var}^*-1|$ \\",
         r"\midrule"]
    for m in SIM8:
        a = d[(d.model == m) & (d.noise > 0)]
        b = d[(d.model == m) & (d.noise == 0)]
        L.append(f"{lab(m)} & {f3(a.density_L1.mean())} & {f2(a['violation_%'].mean())} & {f3(a.var_err.median())} & "
                 f"{f2(a.kurt_err.median())} & {int(a.skew_ok.sum())}/12 & {f3(b.density_L1.mean())} & "
                 f"{f2(b['violation_%'].mean())} & {f3(b.var_err.median())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("horse.tex", L)


def table_horse_by_scenario(d):
    cols = SIM8
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Integrated absolute density error (L1) by scenario, noisy quotes ($\sigma_{IV}=0.0025$). Lowest value per row in bold.}",
         r"\label{tab:horse_scen}", r"\small", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Scenario & " + " & ".join(lab(c) for c in cols) + r" \\", r"\midrule"]
    p = d[d.noise > 0].pivot(index="scenario", columns="model", values="density_L1")[cols]
    for sid, row in p.iterrows():
        best = row.min()
        cells = [(rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}") for v in row.values]
        L.append(f"{sid} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("horse_by_scenario.tex", L)


# ------------------------------------------------------------------ theory
def table_theory():
    t = pd.read_csv(os.path.join(BASE, "results_wp3", "tables", "theory_verification.csv"))
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Monte Carlo check of Proposition~\ref{prop:noise}(i): ratio of the empirical standard deviation of the "
         r"finite-difference Breeden--Litzenberger density (2{,}000 noise draws) to the closed-form value, over interior strikes; "
         r"and the implied median signal-to-noise ratio $f/\mathrm{sd}$.}",
         r"\label{tab:theory}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Scenario & Strikes & $\sigma_{IV}$ & Ratio: median [min, max] & Median SNR \\", r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"{r.scenario} & {int(r.n_strikes)} & {r.sigma_iv:g} & {r.ratio_median:.3f} [{r.ratio_min:.3f}, {r.ratio_max:.3f}] & {r.snr_median:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("theory.tex", L)


# ------------------------------------------------------------------ factorial
def factorial():
    g = pd.read_csv(os.path.join(BASE, "results_wp3", "tables", "factorial_gpu.csv"))
    c = pd.read_csv(os.path.join(BASE, "results_wp3", "tables", "factorial_classical.csv"))
    d = pd.concat([g, c], ignore_index=True)
    d["failed"] = d["error"].notna() if "error" in d.columns else False
    return d


def table_snr(d):
    b = d[(d.method == "BL+MJD") & (d.noise > 0)]
    bins = [(0, 1, r"$<1$"), (1, 2, r"$[1,2)$"), (2, 5, r"$[2,5)$"), (5, np.inf, r"$\ge 5$")]
    rho = np.log(b[~b.failed].snr_bl_median).corr(np.log(b[~b.failed].density_L1), method="spearman")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{BL+MJD in the factorial experiment, grouped by the theory-predicted median signal-to-noise ratio of the "
         r"finite-difference density (Proposition~\ref{prop:noise}). Noisy cells only (4 scenarios $\times$ 4 noise levels "
         rf"$\times$ 4 strike counts $\times$ 5 seeds). Spearman correlation of log SNR with log L1 over successful fits: {rho:.2f}.}}",
         r"\label{tab:snr}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
         r"Predicted SNR & Cells & Failed to return a density (\%) & Median L1 of successful fits \\", r"\midrule"]
    for lo, hi, name in bins:
        s = b[(b.snr_bl_median >= lo) & (b.snr_bl_median < hi)]
        L.append(f"{name} & {len(s)} & {s.failed.mean()*100:.1f} & {f3(s[~s.failed].density_L1.median())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("snr.tex", L)


def table_noise_sensitivity(d):
    ok = d[~d.failed]
    noises = sorted(d.noise.unique())
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Factorial experiment: median density L1 error by estimator and quote-noise level, pooled over four scenarios, "
         r"four strike counts (13--97) and five seeds; BL+MJD medians exclude fits that failed to return a density "
         r"(Table~\ref{tab:snr}).}",
         r"\label{tab:noise}", r"\small", r"\begin{tabular}{l" + "c" * len(noises) + "}", r"\toprule",
         r"Estimator & " + " & ".join(rf"$\sigma_{{IV}}={v:g}$" for v in noises) + r" \\", r"\midrule"]
    for m in REAL6:
        s = ok[ok.method == m]
        L.append(m + " & " + " & ".join(f3(s[s.noise == v].density_L1.median()) for v in noises) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("noise.tex", L)


# ------------------------------------------------------------------ penalties
def table_penalty(d):
    def row(m, sub):
        a = d[(d.model == m) & (d.noise > 0)]
        b = d[(d.model == m) & (d.noise == 0)]
        rs = d[(d.model == m) & d.scenario.isin(["30-No-RS", "30-Weak-RS"])]
        return (f"{sub} & {lab(m)} & {f3(a.density_L1.mean())} & {f3(b.density_L1.mean())} & "
                f"{f2(pd.concat([a, b])['violation_%'].mean())} & {f2(pd.concat([a, b])['violation_%'].max())} & "
                f"{f3(rs.density_L1.mean())} \\\\")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{When do no-arbitrage penalties help? Same data as Table~\ref{tab:horse}. Panel A: a base that does not "
         r"enforce no-arbitrage by construction. Panel B: an arbitrage-free SVI base with a neural residual on total variance, "
         r"with and without the penalties. Last column: mean L1 over the two scenarios where SVI is most misspecified "
         r"(30-No-RS, 30-Weak-RS), both noise levels.}",
         r"\label{tab:penalty}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Panel & Estimator & Mean L1 (noisy) & Mean L1 (noiseless) & Mean violation (\%) & Max violation (\%) & L1, SVI-misspecified \\",
         r"\midrule", row("Vanilla NN", "A"), row("PINN", "A"), r"\midrule",
         row("SVI", "B"), row("SVI+NN (no physics)", "B"), row("SVI+PINN", "B"),
         r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("penalty.tex", L)


# ------------------------------------------------------------------ real data
def table_real():
    r = pd.read_csv(os.path.join(BASE, "results_wp4", "tables", "wp4_metrics.csv"))
    fails = pd.read_csv(os.path.join(BASE, "results_wp4", "tables", "wp4_failures.csv"))
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Real data: twelve live option chains (SPY, QQQ, IWM; 15, 29, 64 and 92 days; snapshot 15 October 2026, "
         r"15:23 UTC). Out-of-sample: strikes split into two interleaved folds; each estimator is fit on one fold and the "
         r"held-out OTM call-equivalent quotes are priced by integrating that estimator's own density. Averages over chains "
         r"unless stated.}",
         r"\label{tab:real}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Estimator & OOS inside spread (\%) & OOS error (half-spreads, median) & In-sample inside (\%) & Violation (\%) & "
         r"Worst negative / peak & Mass (median) & Negative skew (\% chains) & Failed fits \\", r"\midrule"]
    for m in REAL6:
        s = r[r.method == m]
        L.append(f"{m} & {s['oos_inside_%'].mean():.1f} & {s.oos_err_halfspreads.median():.2f} & {s['insample_inside_%'].mean():.1f} & "
                 f"{s['violation_%'].mean():.2f} & {s.worst_neg_rel_peak.max():.3f} & {s['mass'].median():.3f} & "
                 f"{(s.skew99 < 0).mean()*100:.0f} & {int((fails.method == m).sum())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("real.tex", L)

    ch = r.drop_duplicates("chain").set_index("chain")
    piv = r.pivot(index="chain", columns="method", values="oos_inside_%")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Real data by chain: number of OTM quotes in the window, ATM implied volatility, the theory-predicted median "
         r"BL signal-to-noise ratio (quote noise = half-spread$/\sqrt{3}$; signal = SVI density, a proxy), and the out-of-sample "
         r"share of held-out quotes priced inside the spread (\%).}",
         r"\label{tab:real_chain}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccc" + "c" * len(REAL6) + "}",
         r"\toprule", r"Chain & Days & Quotes & ATM IV & BL SNR & " + " & ".join(REAL6) + r" \\", r"\midrule"]
    for name, row in ch.iterrows():
        L.append(f"{name.replace('_', ' ')} & {row.days:.0f} & {int(row.n_quotes)} & {row.atm_iv:.3f} & {row.bl_snr_proxy:.2f} & " +
                 " & ".join(f"{piv.loc[name, m]:.1f}" for m in REAL6) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("real_by_chain.tex", L)


def figure_real(chain="SPY_029d"):
    z = np.load(os.path.join(BASE, "results_wp4", "tables", "wp4_densities.npz"))
    x = z[f"grid|{chain}"]
    style = {"PINN": dict(color="#1b6ca8", lw=1.5), "Vanilla NN": dict(color="#c0392b", lw=1.1, ls="--"),
             "BL+MJD": dict(color="#27803a", lw=0.9, ls=":"), "SVI": dict(color="#8e44ad", lw=1.2),
             "PCA": dict(color="#d68910", lw=1.3, ls="-."), "Spline+GEV": dict(color="#5d6d7e", lw=1.2, ls=(0, (5, 1)))}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m in REAL6:
        k = f"{chain}|full|{m}"
        if k in z:
            ax.plot(x, np.clip(z[k], 1e-6, None), label=m, **style[m])
    ax.set_yscale("log"); ax.set_ylim(1e-5, None)
    ax.set_xlabel(r"$x=\log(S_T/S_0)$"); ax.set_ylabel("Risk-neutral density (log scale)")
    ax.set_title(f"{chain.replace('_', ' ')}: full-sample fits", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"real_{chain}.pdf")); plt.close(fig)


def copy_figures():
    pairs = [("results_horserace/figures/horse_log_noisy_30-Strong-RS.pdf", "horse_log_30-Strong-RS.pdf"),
             ("results_horserace/figures/horse_lin_noisy_60-Strong-RS.pdf", "horse_lin_60-Strong-RS.pdf"),
             ("results_wp3/figures/phase_L1_30.pdf", "phase_30.pdf"),
             ("results_wp3/figures/phase_L1_60.pdf", "phase_60.pdf")]
    for src, dst in pairs:
        shutil.copy(os.path.join(BASE, src), os.path.join(FIG, dst))


if __name__ == "__main__":
    d = horse(); table_horse(d); table_horse_by_scenario(d); table_penalty(d)
    table_theory()
    f = factorial(); table_snr(f); table_noise_sensitivity(f)
    table_real(); figure_real("SPY_029d")
    copy_figures()
