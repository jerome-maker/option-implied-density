"""Vanilla-NN baseline: identical architecture and optimizer to the PINN,
trained on data MSE only (lambda1 = lambda2 = 0).  This is the ablation
requested for the paper: "no physics constraints" vs. "PINN".

Reuses `CallPriceNet` from pinn.py so the only difference between the two
models is the loss function used during training, not the architecture.
"""
from __future__ import annotations

from .pinn import CallPriceNet


def build_baseline_model(hidden_dim: int = 64, n_hidden: int = 3, activation: str = "softplus") -> CallPriceNet:
    """Same architecture as the PINN; kept as a separate constructor so
    the training script can build both models from clearly-named
    factories, and so a future variant (e.g. a ReLU baseline, to
    illustrate the zero-second-derivative failure mode) can be added
    without touching the PINN's own module."""
    return CallPriceNet(hidden_dim=hidden_dim, n_hidden=n_hidden, activation=activation)


def build_relu_baseline_model(hidden_dim: int = 64, n_hidden: int = 3) -> CallPriceNet:
    """Additional ablation: ReLU activation, data-loss only.  ReLU's
    second derivative is zero almost everywhere, so any BL-implied
    density computed from this network's autograd second derivative is
    (near-)degenerate -- included to make that failure mode concrete
    rather than just asserted."""
    return CallPriceNet(hidden_dim=hidden_dim, n_hidden=n_hidden, activation="relu")
