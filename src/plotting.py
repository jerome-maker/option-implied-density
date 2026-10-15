"""Figures: IV-smile fit, density comparison (linear + log scale), and
training-loss convergence curves. Saved as vector PDFs for \\includegraphics.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .svj_model import bs_implied_vol
from .train import predict_prices

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def plot_iv_fit(result, path):
    p = result["p"]
    data = result["data"]
    x_eval = result["x_eval"]

    c_pinn = predict_prices(result["pinn"], p, x_eval)
    c_nn = predict_prices(result["nn"], p, x_eval)
    K_eval = p.S0 * np.exp(x_eval)
    iv_pinn = bs_implied_vol(np.clip(c_pinn, 1e-9, None), p.S0, K_eval, p.r, p.q, p.T)
    iv_nn = bs_implied_vol(np.clip(c_nn, 1e-9, None), p.S0, K_eval, p.r, p.q, p.T)
    from .mjd_model import mjd_call_price
    c_blmjd = mjd_call_price(K_eval, p.S0, p.r, p.q, p.T, result["blmjd"]["mjd_params"])
    iv_blmjd = bs_implied_vol(np.clip(c_blmjd, 1e-9, None), p.S0, K_eval, p.r, p.q, p.T)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lo, hi = result["moneyness_range"]
    ax.axvspan(lo, hi, color="0.85", zorder=0, label="calibration window")
    ax.plot(x_eval, iv_pinn, color="#1b6ca8", lw=1.6, label="PINN")
    ax.plot(x_eval, iv_nn, color="#c0392b", lw=1.4, ls="--", label="Vanilla NN")
    ax.plot(x_eval, iv_blmjd, color="#27803a", lw=1.2, ls=":", label="BL+MJD")
    ax.scatter(data["x"], data["iv_obs"], s=14, color="k", zorder=5, label="Noisy IV quotes")
    ax.set_xlabel(r"$\ln(K/S_t)$")
    ax.set_ylabel("Implied volatility")
    ax.set_title(result["scenario_id"])
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_density_linear(result, path):
    x_eval = result["x_eval"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lo, hi = result["moneyness_range"]
    ax.axvspan(lo, hi, color="0.9", zorder=0, label="calibration window")
    ax.plot(x_eval, result["f_true_eval"], color="k", lw=1.6, label="True SVJ")
    ax.plot(x_eval, result["f_pinn"], color="#1b6ca8", lw=1.4, label="PINN")
    ax.plot(x_eval, result["f_nn"], color="#c0392b", lw=1.2, ls="--", label="Vanilla NN")
    ax.plot(x_eval, result["f_blmjd"], color="#27803a", lw=1.2, ls=":", label="BL+MJD")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("x = log(S_T/S_0)")
    ax.set_ylabel("Risk-neutral density")
    ax.set_title(result["scenario_id"])
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_density_log(result, path):
    x_eval = result["x_eval"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lo, hi = result["moneyness_range"]
    ax.axvspan(lo, hi, color="0.9", zorder=0, label="calibration window")
    ax.plot(x_eval, np.clip(result["f_true_eval"], 1e-8, None), color="k", lw=1.6, label="True SVJ")
    ax.plot(x_eval, np.clip(result["f_pinn"], 1e-8, None), color="#1b6ca8", lw=1.4, label="PINN")
    ax.plot(x_eval, np.clip(result["f_nn"], 1e-8, None), color="#c0392b", lw=1.2, ls="--", label="Vanilla NN")
    ax.plot(x_eval, np.clip(result["f_blmjd"], 1e-8, None), color="#27803a", lw=1.2, ls=":", label="BL+MJD")
    ax.set_yscale("log")
    ax.set_xlabel("x = log(S_T/S_0)")
    ax.set_ylabel("Risk-neutral density (log scale)")
    ax.set_title(result["scenario_id"])
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_loss(result, path):
    hist = result["hist_pinn"]
    epochs = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(epochs, [h["data"] for h in hist], label="Data loss", lw=1.3)
    ax.plot(epochs, [h["mono"] for h in hist], label="Monotonicity penalty", lw=1.1)
    ax.plot(epochs, [h["conv"] for h in hist], label="Convexity penalty", lw=1.1)
    ax.plot(epochs, [h["mass"] for h in hist], label="Mass-conservation penalty", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"PINN training convergence: {result['scenario_id']}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
