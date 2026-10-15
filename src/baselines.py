"""Strong classical baselines for risk-neutral density (RND) estimation
each fit to the same (x, K, C_obs, iv_obs) quotes as the PINN,
vanilla NN and BL+MJD, and evaluated on the same dense log-return grid:

  * fit_svi        -- Gatheral & Jacquier (2014): raw-SVI total variance
                      calibrated subject to their butterfly no-arbitrage
                      condition g(k) >= 0 and Roger Lee's moment bound; the
                      density follows in closed form from g(k).
  * fit_pca        -- Bondarenko (2003) positive convolution approximation:
                      the density is a mixture of normals in log-return
                      space with non-negative weights summing to one, so
                      positivity, convexity and monotonicity of the implied
                      call-price function hold by construction; the
                      martingale condition is an equality constraint.
  * fit_spline_gev -- Figlewski (2010): smoothing spline on the implied-
                      volatility smile, Breeden-Litzenberger on the
                      smoothed prices in the interior, and generalized
                      extreme value (GEV) tails attached at both ends.

All return the same dict shape as bl_mjd.fit_bl_mjd: x_eval, f_eval,
price_mape, iv_mape_% (in-sample, on the training quotes).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize, nnls, least_squares
from scipy.stats import norm, genextreme

from .svj_model import bs_call_price, bs_implied_vol


def _fit_metrics(price_model, iv_model, C, iv_obs):
    return {
        "price_mape": float(np.mean(np.abs(price_model - C) / np.abs(C))),
        "iv_mape_%": float(np.mean(np.abs(iv_model - iv_obs) / np.abs(iv_obs)) * 100.0),
    }


# --------------------------------------------------------------------------
# SVI (Gatheral & Jacquier 2014)
# --------------------------------------------------------------------------

def _svi_w(k, a, b, rho, m, s):
    d = k - m
    root = np.sqrt(d * d + s * s)
    w = a + b * (rho * d + root)
    w1 = b * (rho + d / root)
    w2 = b * s * s / root ** 3
    return w, w1, w2


def _svi_g(k, params):
    """Butterfly-arbitrage function g(k) of Gatheral & Jacquier (2014, eq. 2.1)."""
    w, w1, w2 = _svi_w(k, *params)
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_svi(p, data, eval_range, n_grid=401, n_restarts=8, seed=0):
    x, K, C, iv_obs = data["x"], data["K"], data["C_obs"], data["iv_obs"]
    fwd_shift = (p.r - p.q) * p.T  # k = log(K/F) = x - (r-q)T
    k_obs = x - fwd_shift
    w_obs = iv_obs ** 2 * p.T
    k_chk = np.linspace(min(eval_range[0], k_obs.min()) - 0.5, max(eval_range[1], k_obs.max()) + 0.5, 400)

    def objective(theta):
        a, b, rho, m, s = theta
        w, _, _ = _svi_w(k_obs, a, b, rho, m, s)
        iv_fit = np.sqrt(np.maximum(w, 1e-12) / p.T)
        loss = np.mean((iv_fit - iv_obs) ** 2)
        g = _svi_g(k_chk, theta)
        pen = np.mean(np.minimum(g, 0.0) ** 2)
        lee = max(b * (1 + abs(rho)) - 4.0, 0.0) ** 2
        wmin = max(-(a + b * s * np.sqrt(1 - rho ** 2)), 0.0) ** 2
        return loss + 1e3 * (pen + lee + wmin)

    bounds = [(-1.0, 1.0), (1e-6, 4.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 2.0)]
    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    x0s = [np.array([w_obs.min() * 0.5, 0.1, -0.5, 0.0, 0.1])]
    x0s += [np.array([rng.uniform(lo, hi) for lo, hi in [(-0.05, 0.05), (0.01, 1.0), (-0.9, 0.5), (-0.3, 0.3), (0.01, 0.5)]])
            for _ in range(n_restarts - 1)]
    for x0 in x0s:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    params = tuple(best)

    x_grid = np.linspace(eval_range[0], eval_range[1], n_grid)
    k_grid = x_grid - fwd_shift
    w, _, _ = _svi_w(k_grid, *params)
    w = np.maximum(w, 1e-12)
    dm = -k_grid / np.sqrt(w) - np.sqrt(w) / 2
    f = _svi_g(k_grid, params) / np.sqrt(2 * np.pi * w) * np.exp(-dm ** 2 / 2)

    w_d, _, _ = _svi_w(k_obs, *params)
    iv_fit = np.sqrt(np.maximum(w_d, 1e-12) / p.T)
    price_fit = bs_call_price(p.S0, K, p.r, p.q, iv_fit, p.T)
    out = {"params": params, "x_eval": x_grid, "f_eval": f,
           "min_g": float(_svi_g(k_chk, params).min())}
    out.update(_fit_metrics(price_fit, iv_fit, C, iv_obs))
    return out


def svi_iv_curve(fit, p, x_grid):
    k = x_grid - (p.r - p.q) * p.T
    w, _, _ = _svi_w(k, *fit["params"])
    return np.sqrt(np.maximum(w, 1e-12) / p.T)


# --------------------------------------------------------------------------
# Positive convolution approximation (Bondarenko 2003)
# --------------------------------------------------------------------------

def _pca_design(p, K, centers, h):
    """Discounted call price of each N(c_j, h^2) log-return component."""
    lnk = np.log(K / p.S0)[:, None]
    c = centers[None, :]
    d2 = (c - lnk) / h
    d1 = d2 + h
    return np.exp(-p.r * p.T) * (p.S0 * np.exp(c + 0.5 * h * h) * norm.cdf(d1) - K[:, None] * norm.cdf(d2))


def fit_pca(p, data, eval_range, n_grid=401, n_centers=121, bw_mult=2.0, smooth=1e-3, eq_weight=1e3):
    x, K, C, iv_obs = data["x"], data["K"], data["C_obs"], data["iv_obs"]
    centers = np.linspace(eval_range[0], eval_range[1], n_centers)
    h = bw_mult * (centers[1] - centers[0])

    A = _pca_design(p, K, centers, h)
    scale = 1.0 / (C + 1e-3 * p.S0)          # relative-error rows
    A_fit, b_fit = A * scale[:, None], C * scale

    ones = np.ones((1, n_centers))
    mart = np.exp(centers + 0.5 * h * h)[None, :]
    D2 = np.diff(np.eye(n_centers), n=2, axis=0)  # smoothness on adjacent weights
    A_aug = np.vstack([A_fit, eq_weight * ones, eq_weight * mart, smooth * D2])
    b_aug = np.concatenate([b_fit, [eq_weight * 1.0], [eq_weight * np.exp((p.r - p.q) * p.T)], np.zeros(D2.shape[0])])
    wts, _ = nnls(A_aug, b_aug, maxiter=50 * n_centers)

    x_grid = np.linspace(eval_range[0], eval_range[1], n_grid)
    f = (wts[None, :] * norm.pdf(x_grid[:, None], loc=centers[None, :], scale=h)).sum(axis=1)

    price_fit = A @ wts
    iv_fit = bs_implied_vol(np.clip(price_fit, 1e-10, None), p.S0, K, p.r, p.q, p.T)
    out = {"weights": wts, "centers": centers, "h": h, "x_eval": x_grid, "f_eval": f,
           "mass": float(wts.sum()), "martingale": float((mart @ wts)[0])}
    out.update(_fit_metrics(price_fit, iv_fit, C, iv_obs))
    return out


def pca_prices(fit, p, K):
    return _pca_design(p, np.atleast_1d(K), fit["centers"], fit["h"]) @ fit["weights"]


# --------------------------------------------------------------------------
# Smoothing spline + GEV tails (Figlewski 2010)
# --------------------------------------------------------------------------

def _gev_logpdf_cdf(x, c, loc, scale):
    """Closed-form GEV log-pdf and cdf in scipy's genextreme convention
    (shape c; t = 1 - c z, z = (x - loc)/scale). Identical to
    scipy.stats.genextreme, without its per-call overhead."""
    z = (x - loc) / scale
    if abs(c) < 1e-8:
        log_pdf = -np.log(scale) - z - np.exp(-z)
        return log_pdf, np.exp(-np.exp(-z))
    t = 1.0 - c * z
    if t <= 0:
        # outside the support: cdf is 0 (c<0, lower end) or 1 (c>0, upper end)
        return -np.inf, (1.0 if c > 0 else 0.0)
    tp = t ** (1.0 / c)
    log_pdf = -np.log(scale) + (1.0 / c - 1.0) * np.log(t) - tp
    return log_pdf, np.exp(-tp)


def _gev_tail_params(x_a, x_b, f_a, f_b, F_b, tol=1e-14):
    """Right-tail GEV (on variable x) matching pdf at x_a, x_b and cdf at x_b."""
    def resid(theta):
        c, loc, log_scale = theta
        sc = np.exp(log_scale)
        lpa, _ = _gev_logpdf_cdf(x_a, c, loc, sc)
        lpb, Fb = _gev_logpdf_cdf(x_b, c, loc, sc)
        return [max(lpa, -690.0) - np.log(f_a),
                max(lpb, -690.0) - np.log(f_b),
                10.0 * (Fb - F_b)]

    best, best_cost = None, np.inf
    for c0 in (-0.3, -0.1, 0.0, 0.1, 0.3):
        for loc0 in (x_a, x_b, 0.5 * (x_a + x_b)):
            try:
                sol = least_squares(resid, [c0, loc0, np.log(max(x_b - x_a, 1e-3))], max_nfev=2000)
            except Exception:
                continue
            if sol.cost < best_cost:
                best, best_cost = sol.x, sol.cost
            if best_cost < tol:          # three equations solved exactly; no better start exists
                break
        if best_cost < tol:
            break
    c, loc, log_scale = best
    return c, loc, np.exp(log_scale)


def fit_spline_gev(p, data, eval_range, n_grid=401, n_fine=3001, attach_frac=0.15):
    x, K, C, iv_obs = data["x"], data["K"], data["C_obs"], data["iv_obs"]
    order = np.argsort(x)
    x, K, C, iv_obs = x[order], K[order], C[order], iv_obs[order]

    spl = make_smoothing_spline(x, iv_obs)  # GCV-chosen smoothing

    # Interior: BL on the smoothed smile over the observed strike range.
    xf = np.linspace(x[0], x[-1], n_fine)
    Kf = p.S0 * np.exp(xf)
    Cf = bs_call_price(p.S0, Kf, p.r, p.q, np.maximum(spl(xf), 1e-4), p.T)
    dC = np.gradient(Cf, Kf)
    d2C = np.gradient(dC, Kf)
    f_in = np.exp(p.r * p.T) * Kf * d2C            # log-return density
    F_in = 1.0 + np.exp(p.r * p.T) * dC            # risk-neutral CDF in K (= in x)

    # Attachment points a little inside each edge (Figlewski 2010).
    n_edge = max(int(attach_frac * n_fine), 3)
    iR_b, iR_a = n_fine - 2, n_fine - 2 - n_edge
    iL_b, iL_a = 1, 1 + n_edge
    fR = np.maximum(f_in, 1e-12)

    cR, locR, scR = _gev_tail_params(xf[iR_a], xf[iR_b], fR[iR_a], fR[iR_b], np.clip(F_in[iR_b], 1e-6, 1 - 1e-6))
    # Left tail: GEV on -x; survival of -x at -x_b equals F(x_b).
    cL, locL, scL = _gev_tail_params(-xf[iL_a], -xf[iL_b], fR[iL_a], fR[iL_b], np.clip(1 - F_in[iL_b], 1e-6, 1 - 1e-6))

    x_grid = np.linspace(eval_range[0], eval_range[1], n_grid)
    f = np.interp(x_grid, xf, f_in)
    right = x_grid > xf[iR_b]
    left = x_grid < xf[iL_b]
    f[right] = genextreme.pdf(x_grid[right], cR, loc=locR, scale=scR)
    f[left] = genextreme.pdf(-x_grid[left], cL, loc=locL, scale=scL)

    iv_fit = spl(x)
    price_fit = bs_call_price(p.S0, K, p.r, p.q, np.maximum(iv_fit, 1e-4), p.T)
    out = {"spline": spl, "x_eval": x_grid, "f_eval": f,
           "gev_right": (cR, locR, scR), "gev_left": (cL, locL, scL),
           "mass": float(np.trapezoid(np.clip(f, 0, None), x_grid))}
    out.update(_fit_metrics(price_fit, iv_fit, C, iv_obs))
    return out
