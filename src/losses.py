"""Loss functions for the PINN and the vanilla-NN baseline.

    Loss_Total = Loss_Data
               + lambda1 * Loss_Monotonicity
               + lambda2 * Loss_Convexity
               + lambda3 * Loss_MassConservation

Loss_Data is a scale-robust (log-price) fit error on observed (noisy)
call prices. For the vanilla-NN baseline, lambda1 = lambda2 = lambda3 = 0,
i.e. it is trained on Loss_Data alone -- this isolates exactly the
contribution of the physics penalties, holding architecture, optimizer,
and data fixed.

Loss_MassConservation is a third physics term beyond the pointwise
Breeden-Litzenberger monotonicity/convexity conditions: it enforces the
*global* conservation law that a risk-neutral density must integrate to
one, ``\\int f_X(x) dx = 1``, evaluated on a fixed quadrature grid spanning
the full domain (including the extrapolated region outside the observed
strikes). Pointwise monotonicity/convexity alone are necessary but not
sufficient for a *sensible* tail: infinitely many arbitrage-free
extrapolations exist (e.g. an unrealistically slowly-decaying tail), and
without this term the network has no incentive to prefer thin, realistic
tails over needlessly heavy ones. Total-probability conservation is the
natural additional physical law that disciplines the tail shape while
remaining completely model-free (no parametric tail model is assumed, in
contrast to the calibrated-MJD tail extension of BL+MJD).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pinn import arbitrage_residuals, implied_density

mse = nn.MSELoss()


def data_loss(model, x_data: torch.Tensor, c_data: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """Log-price squared error.

    Raw price MSE is dominated by near-the-money quotes (large price
    scale) and effectively ignores deep-OTM quotes (tiny price scale) --
    exactly the scale problem that BL+MJD addresses with
    inverse-vega weighting in implied-volatility space (src/mjd_model.py). A
    direct inverse-price-squared weight on the raw residual is unstable
    early in training (huge weight x huge initial residual for the
    smallest quotes), so instead we compare log(price + floor):  this
    gives every strike, including the deep wings, comparable relative
    weight in the loss while remaining numerically well-behaved even when
    the network's initial predictions are far from the tiny tail prices.
    """
    pred = model(x_data)
    return mse(torch.log(pred + floor), torch.log(c_data + floor))


def physics_losses(model, x_colloc: torch.Tensor):
    mono_violation, conv_violation = arbitrage_residuals(model, x_colloc)
    loss_mono = (mono_violation ** 2).mean()
    loss_conv = (conv_violation ** 2).mean()
    return loss_mono, loss_conv


def mass_conservation_loss(model, x_quad: torch.Tensor, r: float, T: float) -> torch.Tensor:
    """(\\int f_X(x) dx - 1)^2 on a fixed quadrature grid (trapezoidal rule)."""
    f = implied_density(model, x_quad, r, T).view(-1)
    x_flat = x_quad.detach().view(-1)
    mass = torch.trapezoid(torch.clamp(f, min=0.0), x_flat)
    return (mass - 1.0) ** 2


def total_loss(model, x_data, c_data, x_colloc, x_quad, r, T, lambda1: float, lambda2: float,
               lambda3: float = 0.0, lambda_r: float = 0.0):
    """lambda_r > 0 adds model.regularization(x_colloc) -- used by the
    SVI-anchored hybrid (src/hybrid.py) to pull the neural residual toward
    zero where quotes carry no information."""
    l_data = data_loss(model, x_data, c_data)
    l_reg = model.regularization(x_colloc) if lambda_r > 0.0 else torch.zeros_like(l_data)
    if lambda1 == 0.0 and lambda2 == 0.0 and lambda3 == 0.0:
        total = l_data + lambda_r * l_reg
        return total, {"data": l_data.item(), "mono": 0.0, "conv": 0.0, "mass": 0.0, "reg": l_reg.item()}
    l_mono, l_conv = physics_losses(model, x_colloc)
    l_mass = mass_conservation_loss(model, x_quad, r, T) if lambda3 > 0.0 else torch.zeros_like(l_data)
    total = l_data + lambda1 * l_mono + lambda2 * l_conv + lambda3 * l_mass + lambda_r * l_reg
    return total, {"data": l_data.item(), "mono": l_mono.item(), "conv": l_conv.item(),
                   "mass": l_mass.item(), "reg": l_reg.item()}
