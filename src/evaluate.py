"""Fit PINN + vanilla-NN baseline on one scenario/moneyness window and
compute all evaluation metrics: price MAPE, arbitrage-violation rate,
excess extrapolated tail mass, and truncated-moment tables.
"""
from __future__ import annotations

import numpy as np
import torch

from .data_gen import make_training_data, make_eval_grid
from .pinn import CallPriceNet, implied_density
from .baseline_nn import build_baseline_model
from .train import train_model, predict_prices
from .moments import moment_table, P_LEVELS
from .svj_model import bs_implied_vol
from .bl_mjd import fit_bl_mjd
from .baselines import fit_svi, fit_pca, fit_spline_gev

DEFAULT_HPARAMS = dict(hidden_dim=64, n_hidden=3, activation="softplus")
DEFAULT_TRAIN = dict(epochs=4000, lr=2e-3, n_colloc=150, n_quad=200)


def _eval_range_for(scenario_id: str) -> tuple[float, float]:
    return (-1.0, 1.0) if scenario_id.startswith("60-") else (-0.5, 0.5)


def model_density_on_grid(model, x_grid: np.ndarray, r: float, T: float) -> np.ndarray:
    x_t = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
    f = implied_density(model, x_t, r, T).detach().numpy().ravel()
    return f


def arbitrage_violation_rate(f_grid: np.ndarray) -> float:
    return float(np.mean(f_grid < -1e-8))


def excess_tail_mass(x_grid, f_grid, moneyness_range) -> float:
    """Probability mass placed *outside* the training moneyness window --
    compared against the true model's own out-of-window mass, this
    quantifies whether a model's extrapolated tail is a realistic size
    or spuriously heavy."""
    f_clip = np.clip(f_grid, 0.0, None)
    lo, hi = moneyness_range
    mask = (x_grid < lo) | (x_grid > hi)
    return float(np.trapezoid(np.where(mask, f_clip, 0.0), x_grid))


def price_mape(model, p, x, C_true) -> float:
    pred = predict_prices(model, p, x)
    return float(np.mean(np.abs(pred - C_true) / np.abs(C_true)))


def iv_mape(model, p, x, iv_true) -> float:
    pred_price = predict_prices(model, p, x)
    pred_price = np.clip(pred_price, 1e-8, None)
    K = p.S0 * np.exp(x)
    iv_pred = bs_implied_vol(pred_price, p.S0, K, p.r, p.q, p.T)
    return float(np.mean(np.abs(iv_pred - iv_true) / np.abs(iv_true)) * 100.0)


def run_scenario(
    p,
    moneyness_range=(-0.2, 0.2),
    n_strikes=25,
    iv_noise_std=0.0025,
    lambda1=2.0,
    lambda2=2.0,
    lambda3=1.0,
    seed=0,
    hparams=None,
    train_kwargs=None,
    eval_range=None,
    verbose=False,
):
    hparams = {**DEFAULT_HPARAMS, **(hparams or {})}
    train_kwargs = {**DEFAULT_TRAIN, **(train_kwargs or {})}
    eval_range = eval_range or _eval_range_for(p.scenario_id)

    data = make_training_data(p, moneyness_range=moneyness_range, n_strikes=n_strikes,
                               iv_noise_std=iv_noise_std, seed=seed)
    x_eval, f_true_eval, xg, fg = make_eval_grid(p, full_range=eval_range, n_points=401)

    pinn = CallPriceNet(**hparams)
    hist_pinn = train_model(pinn, p, data, colloc_range=eval_range,
                             lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                             seed=seed, verbose=verbose, **train_kwargs)

    nn_base = build_baseline_model(**hparams)
    hist_nn = train_model(nn_base, p, data, colloc_range=eval_range,
                           lambda1=0.0, lambda2=0.0, lambda3=0.0,
                           seed=seed, verbose=False, **train_kwargs)

    f_pinn = model_density_on_grid(pinn, x_eval, p.r, p.T)
    f_nn = model_density_on_grid(nn_base, x_eval, p.r, p.T)

    blmjd = fit_bl_mjd(p, data, eval_range=eval_range, n_grid=len(x_eval))
    f_blmjd = blmjd["f_eval"]

    result = {
        "scenario_id": p.scenario_id,
        "moneyness_range": moneyness_range,
        "eval_range": eval_range,
        "p": p,
        "data": data,
        "x_eval": x_eval, "f_true_eval": f_true_eval,
        "f_pinn": f_pinn, "f_nn": f_nn, "f_blmjd": f_blmjd,
        "pinn": pinn, "nn": nn_base, "blmjd": blmjd,
        "hist_pinn": hist_pinn, "hist_nn": hist_nn,
        "metrics": {
            "price_mape_pinn": price_mape(pinn, p, data["x"], data["C_true"]),
            "price_mape_nn": price_mape(nn_base, p, data["x"], data["C_true"]),
            "price_mape_blmjd": blmjd["price_mape"],
            "iv_mape_pinn": iv_mape(pinn, p, data["x"], data["iv_true"]),
            "iv_mape_nn": iv_mape(nn_base, p, data["x"], data["iv_true"]),
            "iv_mape_blmjd": blmjd["iv_mape_%"],
            "violation_rate_pinn": arbitrage_violation_rate(f_pinn),
            "violation_rate_nn": arbitrage_violation_rate(f_nn),
            "violation_rate_blmjd": arbitrage_violation_rate(f_blmjd),
            "tail_mass_true": excess_tail_mass(x_eval, f_true_eval, moneyness_range),
            "tail_mass_pinn": excess_tail_mass(x_eval, f_pinn, moneyness_range),
            "tail_mass_nn": excess_tail_mass(x_eval, f_nn, moneyness_range),
            "tail_mass_blmjd": excess_tail_mass(x_eval, f_blmjd, moneyness_range),
        },
        "moments_true": moment_table(x_eval, f_true_eval),
        "moments_pinn": moment_table(x_eval, f_pinn),
        "moments_nn": moment_table(x_eval, f_nn),
        "moments_blmjd": moment_table(x_eval, f_blmjd),
    }
    return result


def density_l1(x_grid, f_model, f_true) -> float:
    """Integrated absolute density error over the evaluation grid."""
    return float(np.trapezoid(np.abs(f_model - f_true), x_grid))


def run_horse_race(p, moneyness_range=(-0.2, 0.2), iv_noise_std=0.0025, seed=0, eval_range=None):
    """All six estimators on identical quotes: PINN, vanilla NN, BL+MJD
    SVI (Gatheral-Jacquier), PCA (Bondarenko)
    and smoothing spline + GEV tails (Figlewski). Returns one tidy record
    per estimator plus truncated-moment rows."""
    eval_range = eval_range or _eval_range_for(p.scenario_id)
    base = run_scenario(p, moneyness_range=moneyness_range, iv_noise_std=iv_noise_std,
                        seed=seed, eval_range=eval_range)
    data, x_eval, f_true = base["data"], base["x_eval"], base["f_true_eval"]
    n = len(x_eval)

    fits = {
        "SVI": fit_svi(p, data, eval_range, n_grid=n),
        "PCA": fit_pca(p, data, eval_range, n_grid=n),
        "Spline+GEV": fit_spline_gev(p, data, eval_range, n_grid=n),
    }
    m = base["metrics"]
    densities = {"PINN": base["f_pinn"], "Vanilla NN": base["f_nn"], "BL+MJD": base["f_blmjd"]}
    iv_mape = {"PINN": m["iv_mape_pinn"], "Vanilla NN": m["iv_mape_nn"], "BL+MJD": m["iv_mape_blmjd"]}
    for name, fit in fits.items():
        densities[name] = fit["f_eval"]
        iv_mape[name] = fit["iv_mape_%"]

    true_mom = moment_table(x_eval, f_true)
    records, moment_records = [], []
    for name, f in densities.items():
        mom = moment_table(x_eval, f)
        records.append({
            "scenario": p.scenario_id, "noise": iv_noise_std, "model": name,
            "iv_mape_%": iv_mape[name],
            "violation_%": arbitrage_violation_rate(f) * 100,
            "density_L1": density_l1(x_eval, f, f_true),
            "tail_mass": excess_tail_mass(x_eval, f, moneyness_range),
            "tail_mass_true": m["tail_mass_true"],
            "var99_ratio": mom[0.99]["variance"] / true_mom[0.99]["variance"],
            "skew99": mom[0.99]["skew"], "skew99_true": true_mom[0.99]["skew"],
            "kurt99": mom[0.99]["kurt"], "kurt99_true": true_mom[0.99]["kurt"],
        })
        for p_level in P_LEVELS:
            mm = mom[p_level]
            moment_records.append({"scenario": p.scenario_id, "noise": iv_noise_std, "model": name,
                                   "p_level": p_level, **{k: mm[k] for k in ("mean", "variance", "skew", "kurt")}})
    for p_level in P_LEVELS:
        mm = true_mom[p_level]
        moment_records.append({"scenario": p.scenario_id, "noise": iv_noise_std, "model": "SVJ (true)",
                               "p_level": p_level, **{k: mm[k] for k in ("mean", "variance", "skew", "kurt")}})
    return {"base": base, "fits": fits, "densities": densities, "records": records, "moment_records": moment_records}
