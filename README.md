# When can you trust an option-implied density?

Code and results for

> Chou, J. C.-L., Miao, D. W.-C., Lin, X. C.-S. and Lee, Y.-I. (2026).
> *When Can You Trust an Option-Implied Density? Noise Amplification,
> Architecture and the Limits of No-Arbitrage Penalties.* Manuscript under review.

The paper compares eight risk-neutral density (RND) estimators on quotes with
known true densities and on live option chains:

| Estimator | Module |
|---|---|
| BL+MJD: finite-difference Breeden–Litzenberger interior with calibrated Merton jump-diffusion tails | `src/bl_mjd.py`, `src/mjd_model.py` |
| SVI with Gatheral–Jacquier no-butterfly-arbitrage conditions | `src/baselines.py` |
| Positive convolution approximation (Bondarenko, 2003) | `src/baselines.py` |
| Smoothing spline with GEV tails (Figlewski, 2010) | `src/baselines.py` |
| Unconstrained neural network | `src/pinn.py`, `src/losses.py` |
| Physics-informed neural network (no-arbitrage penalties) | `src/pinn.py`, `src/losses.py` |
| SVI + neural residual, with and without penalties | `src/hybrid.py` |

The noise-amplification result for the finite-difference density
(Proposition 1 in the paper) is implemented in `src/theory.py`.

## Layout

```
src/                  estimators, Bates SVJ benchmark (COS density), theory
experiments/          one script per part of the paper
results_horserace/    simulation horse race (Tables 3, 6, 8; Figure 1)
results_wp3/          theory check and factorial experiment (Tables 1, 4, 5; Figures 2, 4)
results_wp4/          live option chains (Tables 7, 9; Figure 3)
paper_assets/         the .tex tables and .pdf figures exactly as used in the paper
data/                 fingerprints of the live option-chain snapshot (see data/README.md)
```

## Reproducing

Python 3.11; package versions in `requirements.txt`. A CUDA GPU is used where
available (the factorial experiment trains its networks in batches on the GPU).

| Part of the paper | Command | Output |
|---|---|---|
| Simulation horse race | `python experiments/run_horse_race.py` | `results_horserace/` |
| SVI + neural residual | `python experiments/run_hybrid.py` | `results_horserace/tables/hybrid_*.csv` |
| Proposition 1 Monte Carlo check | `python experiments/verify_theory.py` | `results_wp3/tables/theory_verification.csv` |
| Factorial experiment | `python experiments/run_factorial.py gpu` and `... classical` | `results_wp3/tables/factorial_*.csv` |
| Phase diagrams | `python experiments/plot_factorial.py` | `results_wp3/figures/` |
| Live option chains | `python experiments/run_wp4.py` (needs the snapshot, see below) | `results_wp4/` |
| Paper tables and figures | `python experiments/make_paper_assets.py` | `paper_assets/` |

All simulations use fixed seeds. On the authors' machine (GTX 1060 3 GB) the
factorial experiment takes about 35 minutes on the GPU plus 70 minutes for the
classical estimators on the CPU; the horse race takes about 75 minutes on the CPU.

## Live option-chain data

The real-data section uses one intraday snapshot (15 October 2026, 15:23 UTC)
of SPY, QQQ and IWM option chains at four maturities, taken from Yahoo Finance,
with the 13-week Treasury bill yield (^IRX, 4.08%) as the risk-free rate. The
raw quotes are not redistributed here because of the data provider's terms.
`data/README.md` lists the SHA-256 fingerprint of every file so that a copy
can be verified, and describes the expected format. The derived results
(`results_wp4/`) are included in full.

## License

Code: MIT (see `LICENSE`).
