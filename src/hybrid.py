"""SVI-anchored PINN: an arbitrage-free SVI smile (Gatheral &
Jacquier 2014) as the base, plus a neural residual on total implied
variance,

    w(k) = w_SVI(k) * exp(r_theta(x)),   k = x - (r - q) T,

trained with the same data + monotonicity + convexity + mass-conservation
losses as the plain PINN, plus a residual penalty lambda_r * E[r_theta^2]
over the collocation domain so the model falls back to the SVI wings where
quotes carry no information. The residual network's last layer starts at
zero, so training starts exactly at the SVI fit.

The forward pass returns the normalised call price c(x) = C/S0 through a
differentiable Black-Scholes formula, so the RND still comes from autograd
via pinn.implied_density -- no finite differences anywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _norm_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class SVIResidualPINN(nn.Module):
    def __init__(self, svi_params, r: float, q: float, T: float,
                 hidden_dim: int = 32, n_hidden: int = 2, activation: str = "softplus"):
        super().__init__()
        a, b, rho, m, s = (float(v) for v in svi_params)
        self.register_buffer("svi", torch.tensor([a, b, rho, m, s], dtype=torch.float32))
        self.r, self.q, self.T = r, q, T
        act = {"softplus": nn.Softplus, "silu": nn.SiLU}[activation]
        layers = [nn.Linear(1, hidden_dim), act()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        last = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.res = nn.Sequential(*layers)

    def w_svi(self, k: torch.Tensor) -> torch.Tensor:
        a, b, rho, m, s = self.svi
        d = k - m
        return a + b * (rho * d + torch.sqrt(d * d + s * s))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x)

    def total_variance(self, x: torch.Tensor) -> torch.Tensor:
        k = x - (self.r - self.q) * self.T
        return torch.clamp(self.w_svi(k), min=1e-10) * torch.exp(self.residual(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised call price c(x) = C(S0 e^x)/S0 under Black-Scholes with total variance w(x)."""
        w = self.total_variance(x)
        sw = torch.sqrt(w)
        k = x - (self.r - self.q) * self.T          # log(K/F)
        d1 = -k / sw + 0.5 * sw
        d2 = d1 - sw
        disc_f = math.exp(-self.q * self.T)          # e^{-rT} F / S0
        disc_k = math.exp(-self.r * self.T)
        return disc_f * _norm_cdf(d1) - disc_k * torch.exp(x) * _norm_cdf(d2)

    def regularization(self, x_colloc: torch.Tensor) -> torch.Tensor:
        return (self.residual(x_colloc) ** 2).mean()
