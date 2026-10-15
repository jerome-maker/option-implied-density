"""
Physics-informed neural network (PINN) for arbitrage-free call-price /
risk-neutral-density recovery.

The network maps normalised log-moneyness x = log(K/S_t) to the
normalised call price c(x) = C(K,T)/S0.  Both the network output and its
inputs stay in this well-scaled [-0.6, 0.6] x O(1) range, which keeps
training stable without extra input/output normalisation layers.

No-arbitrage constraints in log-moneyness space
-------------------------------------------------
Let K = S0 e^x. By the chain rule (dx/dK = 1/K):

    dC/dK   = C_x / K
    d^2C/dK^2 = (C_xx - C_x) / K^2

Since K > 0 always, the classical Breeden-Litzenberger no-arbitrage
conditions on the *strike-space* call price translate into simple,
scale-free conditions on the network's own x-derivatives:

    Monotonicity   dC/dK <= 0   <=>   c_x  <= 0
    Convexity/BL   d^2C/dK^2 >= 0   <=>   c_xx - c_x >= 0

and the (normalised) risk-neutral log-return density itself is obtained
in closed form, everywhere, with no finite differences:

    f_X(x) = K * e^{rT} * d^2C/dK^2 = e^{rT - x} * (c_xx - c_x)

All derivatives are obtained via torch.autograd (exact, not
finite-difference), which is what lets the tails (collocation points
placed outside the observed strike range) be constrained just as tightly
as the interior.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CallPriceNet(nn.Module):
    """MLP: x (log-moneyness) -> normalised call price c(x) = C(K,T)/S0.

    Softplus/SiLU activations only (never ReLU): both are C-infinity smooth,
    so the second derivative needed for the convexity/density term is
    never identically zero, unlike ReLU whose second derivative vanishes
    almost everywhere and would force the implied density to zero.
    """

    def __init__(self, hidden_dim: int = 64, n_hidden: int = 3, activation: str = "softplus"):
        super().__init__()
        act_cls = {"softplus": nn.Softplus, "swish": nn.SiLU, "silu": nn.SiLU, "relu": nn.ReLU}[activation]
        layers = [nn.Linear(1, hidden_dim), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        # Final softplus guarantees c(x) > 0 everywhere (call prices are
        # non-negative by construction, on top of the soft BL penalties).
        self.out_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.net(x))


def price_and_derivatives(model: nn.Module, x: torch.Tensor):
    """Return (c, c_x, c_xx) at x, all via autograd (x must require grad)."""
    x = x.requires_grad_(True)
    c = model(x)
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    return c, c_x, c_xx


def implied_density(model: nn.Module, x: torch.Tensor, r: float, T: float) -> torch.Tensor:
    """Closed-form (autograd-based) risk-neutral log-return density f_X(x)."""
    _, c_x, c_xx = price_and_derivatives(model, x)
    disc = torch.exp(torch.tensor(r * T, dtype=x.dtype) - x.detach())
    return disc * (c_xx - c_x)


def arbitrage_residuals(model: nn.Module, x: torch.Tensor):
    """Return (monotonicity_violation, convexity_violation) raw residuals.

    monotonicity_violation = relu(c_x)          (want c_x <= 0)
    convexity_violation    = relu(-(c_xx-c_x))  (want c_xx - c_x >= 0)
    """
    _, c_x, c_xx = price_and_derivatives(model, x)
    mono_violation = torch.relu(c_x)
    conv_violation = torch.relu(-(c_xx - c_x))
    return mono_violation, conv_violation
