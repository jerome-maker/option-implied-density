"""Training loop shared by the PINN and the vanilla-NN baseline.

Collocation points for the physics penalties are re-sampled every epoch
from a domain that is *wider* than the training moneyness window (e.g.
training quotes only in [-0.2, 0.2] but collocation points drawn from
[-0.6, 0.6]).  This is what forces the PINN's no-arbitrage guarantee to
hold in the extrapolated tails, not just inside the observed data range
-- the analogue of the separate MJD tail-splicing step in BL+MJD.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses import total_loss


def train_model(
    model,
    p,
    data: dict,
    colloc_range: tuple[float, float] = (-0.6, 0.6),
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    lambda_r: float = 0.0,
    epochs: int = 4000,
    lr: float = 2e-3,
    n_colloc: int = 256,
    n_quad: int = 400,
    seed: int = 0,
    log_every: int = 500,
    verbose: bool = False,
):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    x_data = torch.tensor(data["x"], dtype=torch.float32).view(-1, 1)
    c_data = torch.tensor(data["C_obs"] / p.S0, dtype=torch.float32).view(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    lo, hi = colloc_range
    x_quad = torch.linspace(lo, hi, n_quad).view(-1, 1)
    for epoch in range(epochs):
        opt.zero_grad()
        x_colloc = torch.rand((n_colloc, 1), generator=rng) * (hi - lo) + lo
        loss, parts = total_loss(model, x_data, c_data, x_colloc, x_quad, p.r, p.T, lambda1, lambda2, lambda3, lambda_r)
        loss.backward()
        opt.step()
        sched.step()
        parts["total"] = loss.item()
        parts["epoch"] = epoch
        history.append(parts)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:5d}  total={parts['total']:.3e}  "
                  f"data={parts['data']:.3e}  mono={parts['mono']:.3e}  conv={parts['conv']:.3e}  mass={parts['mass']:.3e}")
    return history


def predict_prices(model, p, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        c = model(xt).view(-1).numpy()
    return c * p.S0
