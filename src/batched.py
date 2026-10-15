"""GPU batched training of many independent call-price networks (PINN and
vanilla NN) at once, for the factorial experiment.

B models with stacked weights run through batched matrix multiplies. The
loss is the SUM over models of each model's own loss, so each model's
gradient is exactly what it would get trained alone, and Adam (element-wise)
then updates each model as if trained separately. Per-model lambda vectors
let PINN (lambda > 0) and vanilla-NN (lambda = 0) models share one batch.

Architecture, initialisation, losses, optimiser and schedule mirror
src/pinn.CallPriceNet + src/losses + src/train.train_model:
softplus MLP (1 -> H -> H -> H -> 1) with softplus output; log-price data
loss; squared-ReLU monotonicity/convexity penalties on c_x and c_xx - c_x at
random collocation points; trapezoid mass-conservation penalty on a fixed
grid; Adam + cosine annealing.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _uniform(shape, bound, gen, device):
    return (torch.rand(shape, generator=gen, device="cpu") * 2 - 1).mul_(bound).to(device)


class BatchedCallPriceNet(torch.nn.Module):
    def __init__(self, B: int, hidden: int = 64, n_hidden: int = 3, seeds=None, device="cuda"):
        super().__init__()
        dims = [1] + [hidden] * n_hidden + [1]
        self.Ws, self.bs = torch.nn.ParameterList(), torch.nn.ParameterList()
        seeds = list(seeds) if seeds is not None else list(range(B))
        Ws = [[] for _ in dims[:-1]]
        bs = [[] for _ in dims[:-1]]
        for s in seeds:
            # Same scheme as nn.Linear's default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
            g = torch.Generator().manual_seed(int(s))
            for l, (fi, fo) in enumerate(zip(dims[:-1], dims[1:])):
                bound = 1.0 / math.sqrt(fi)
                Ws[l].append(_uniform((fi, fo), bound, g, "cpu"))
                bs[l].append(_uniform((1, fo), bound, g, "cpu"))
        for l in range(len(dims) - 1):
            self.Ws.append(torch.nn.Parameter(torch.stack(Ws[l]).to(device)))
            self.bs.append(torch.nn.Parameter(torch.stack(bs[l]).to(device)))

    def load_single(self, b: int, net):
        """Copy the weights of a src.pinn.CallPriceNet into slot b (for validation)."""
        lin = [m for m in net.net if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for l, m in enumerate(lin):
                self.Ws[l][b].copy_(m.weight.T)
                self.bs[l][b].copy_(m.bias.view(1, -1))

    def forward(self, x):                     # x: (B, N, 1)
        h = x
        L = len(self.Ws)
        for l in range(L):
            h = torch.baddbmm(self.bs[l], h, self.Ws[l])
            if l < L - 1:
                h = F.softplus(h)
        return F.softplus(h)                  # (B, N, 1) normalised price c(x)


def _derivs(model, x):
    x = x.requires_grad_(True)
    c = model(x)
    c_x = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
    c_xx = torch.autograd.grad(c_x.sum(), x, create_graph=True)[0]
    return c, c_x, c_xx


def train_batched(x_data, c_data, lam1, lam2, lam3, r, T, colloc_range, seeds, epochs=4000, lr=2e-3,
                  n_colloc=150, n_quad=200, hidden=64, n_hidden=3, device="cuda", init_from=None, colloc_seed=0):
    """x_data, c_data: (B, n) numpy arrays (c normalised by S0). lam*: length-B arrays.
    Collocation points are drawn from a CPU generator seeded with colloc_seed and shared
    across the batch, matching src.train.train_model's sequence for that seed."""
    B = x_data.shape[0]
    xd = torch.tensor(x_data, dtype=torch.float32, device=device).unsqueeze(-1)
    cd = torch.tensor(c_data, dtype=torch.float32, device=device).unsqueeze(-1)
    l1, l2, l3 = (torch.tensor(np.asarray(v, dtype=np.float32), device=device) for v in (lam1, lam2, lam3))
    model = BatchedCallPriceNet(B, hidden, n_hidden, seeds=seeds, device=device)
    if init_from is not None:
        for b, net in init_from.items():
            model.load_single(b, net)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    torch.manual_seed(colloc_seed)
    rng = torch.Generator().manual_seed(colloc_seed)
    lo, hi = colloc_range
    xq = torch.linspace(lo, hi, n_quad).view(1, -1, 1).to(device).expand(B, -1, -1).contiguous()
    phys = (l1 > 0) | (l2 > 0) | (l3 > 0)
    floor = 1e-4
    for _ in range(epochs):
        opt.zero_grad()
        xc = (torch.rand((n_colloc, 1), generator=rng) * (hi - lo) + lo).to(device)
        xc = xc.view(1, -1, 1).expand(B, -1, -1).contiguous()
        pred = model(xd)
        l_data = ((torch.log(pred + floor) - torch.log(cd + floor)) ** 2).mean(dim=(1, 2))
        loss = l_data
        if bool(phys.any()):
            _, cx, cxx = _derivs(model, xc)
            l_mono = (F.relu(cx) ** 2).mean(dim=(1, 2))
            l_conv = (F.relu(-(cxx - cx)) ** 2).mean(dim=(1, 2))
            _, qx, qxx = _derivs(model, xq.clone())
            f = torch.exp(r * T - xq.detach()) * (qxx - qx)
            mass = torch.trapezoid(torch.clamp(f, min=0.0).squeeze(-1), xq[0, :, 0].detach(), dim=1)
            l_mass = (mass - 1.0) ** 2
            loss = loss + l1 * l_mono + l2 * l_conv + l3 * l_mass
        loss.sum().backward()
        opt.step()
        sched.step()
    return model


def batched_density(model, x_grid, r, T, device="cuda"):
    B = model.Ws[0].shape[0]
    x = torch.tensor(x_grid, dtype=torch.float32, device=device).view(1, -1, 1).expand(B, -1, -1).contiguous()
    _, cx, cxx = _derivs(model, x)
    f = torch.exp(r * T - x.detach()) * (cxx - cx)
    return f.detach().squeeze(-1).cpu().numpy()          # (B, len(x_grid))


def batched_prices(model, x, S0, device="cuda"):
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1)
        return model(xt).squeeze(-1).cpu().numpy() * S0


def train_batched_ragged(x_list, c_list, lam, ranges, r, T_list, seeds, epochs=4000, lr=2e-3,
                         n_colloc=150, n_quad=200, hidden=64, n_hidden=3, device="cuda", colloc_seed=0):
    """Like train_batched, but each model may have its own number of quotes
    (padded + masked in the data loss), its own collocation/quadrature range
    (lo_b, hi_b) and its own maturity T_b. Collocation points are one shared
    uniform draw u in [0,1] per epoch, mapped to lo_b + u (hi_b - lo_b)."""
    B = len(x_list)
    nmax = max(len(v) for v in x_list)
    X = np.zeros((B, nmax), np.float32); C = np.ones((B, nmax), np.float32); M = np.zeros((B, nmax), np.float32)
    for b, (xv, cv) in enumerate(zip(x_list, c_list)):
        n = len(xv); X[b, :n] = xv; C[b, :n] = cv; M[b, :n] = 1.0
        X[b, n:] = xv[-1]; C[b, n:] = cv[-1]
    xd = torch.tensor(X, device=device).unsqueeze(-1)
    cd = torch.tensor(C, device=device).unsqueeze(-1)
    md = torch.tensor(M, device=device).unsqueeze(-1)
    lam = torch.tensor(np.asarray(lam, np.float32), device=device)
    lo = torch.tensor([a for a, _ in ranges], dtype=torch.float32, device=device).view(B, 1, 1)
    hi = torch.tensor([b_ for _, b_ in ranges], dtype=torch.float32, device=device).view(B, 1, 1)
    Tb = torch.tensor(np.asarray(T_list, np.float32), device=device).view(B, 1, 1)
    model = BatchedCallPriceNet(B, hidden, n_hidden, seeds=seeds, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = torch.Generator().manual_seed(colloc_seed)
    uq = torch.linspace(0, 1, n_quad, device=device).view(1, -1, 1)
    xq = (lo + uq * (hi - lo)).contiguous()
    floor = 1e-4
    for _ in range(epochs):
        opt.zero_grad()
        u = torch.rand((1, n_colloc, 1), generator=rng).to(device)
        xc = (lo + u * (hi - lo)).contiguous()
        pred = model(xd)
        l_data = (md * (torch.log(pred + floor) - torch.log(cd + floor)) ** 2).sum(dim=(1, 2)) / md.sum(dim=(1, 2))
        _, cx, cxx = _derivs(model, xc)
        l_mono = (F.relu(cx) ** 2).mean(dim=(1, 2))
        l_conv = (F.relu(-(cxx - cx)) ** 2).mean(dim=(1, 2))
        _, qx, qxx = _derivs(model, xq.clone())
        f = torch.exp(r * Tb - xq.detach()) * (qxx - qx)
        mass = torch.trapezoid(torch.clamp(f, min=0.0).squeeze(-1), xq.detach().squeeze(-1), dim=1)
        loss = l_data + lam * (2.0 * l_mono + 2.0 * l_conv + (mass - 1.0) ** 2)
        loss.sum().backward()
        opt.step()
        sched.step()
    return model


def batched_density_ragged(model, grids, r, T_list, device="cuda"):
    """grids: (B, G) numpy array, one evaluation grid per model."""
    x = torch.tensor(np.asarray(grids, np.float32), device=device).unsqueeze(-1)
    Tb = torch.tensor(np.asarray(T_list, np.float32), device=device).view(-1, 1, 1)
    _, cx, cxx = _derivs(model, x)
    return (torch.exp(r * Tb - x.detach()) * (cxx - cx)).detach().squeeze(-1).cpu().numpy()
