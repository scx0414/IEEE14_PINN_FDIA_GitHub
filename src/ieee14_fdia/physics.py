"""Full AC injection map; theta1 is fixed to zero, all14 Vm are estimated."""

import numpy as np
from .io import read_json


def load_network(path):
    if str(path).endswith(".json"):
        n = read_json(path)
    else:
        n = np.load(path, allow_pickle=False)
    G, B = np.asarray(n["G"], dtype=np.float64), np.asarray(n["B"], dtype=np.float64)
    if (
        G.shape != (14, 14)
        or B.shape != (14, 14)
        or not (np.isfinite(G).all() and np.isfinite(B).all())
    ):
        raise ValueError("G and B must be finite14x14 matrices")
    return G, B


def ac_measurements(states, G, B):
    x = np.atleast_2d(np.asarray(states, dtype=np.float64))
    if x.shape[1] != 27:
        raise ValueError("State order:14 Vm pu,13 nonreference angles rad")
    theta = np.c_[np.zeros(len(x)), x[:, 14:]]
    v = x[:, :14] * np.exp(1j * theta)
    # einsum avoids mixing Torch's OpenMP runtime with a NumPy BLAS matrix product.
    current = np.einsum("ij,bj->bi", G + 1j * B, v, optimize=False)
    s = v * np.conj(current)
    return np.c_[s.real, s.imag]


def node_residual_scores(states, observed, G, B):
    r = np.asarray(observed, dtype=np.float64) - ac_measurements(states, G, B)
    return r[:, :14] ** 2 + r[:, 14:] ** 2


def torch_measurements(states, G, B):
    import torch

    vm = states[:, :14]
    theta = torch.cat([torch.zeros_like(vm[:, :1]), states[:, 14:]], dim=1)
    vr, vi = vm * theta.cos(), vm * theta.sin()
    ir, ii = vr @ G.T - vi @ B.T, vr @ B.T + vi @ G.T
    return torch.cat([vr * ir + vi * ii, vi * ir - vr * ii], dim=1)


def pinn_losses(prediction, truth, measured, G, B):
    """Original per-batch type-group normalization, not covariance whitening."""
    scales = [
        v.std(correction=0) + 1e-8
        for v in (truth[:, :14], truth[:, 14:], measured[:, :14], measured[:, 14:])
    ]
    data = ((prediction[:, :14] - truth[:, :14]) / scales[0]).square().mean()
    data = data + ((prediction[:, 14:] - truth[:, 14:]) / scales[1]).square().mean()
    power = torch_measurements(prediction, G, B)
    physical = ((power[:, :14] - measured[:, :14]) / scales[2]).square().mean()
    physical = (
        physical + ((power[:, 14:] - measured[:, 14:]) / scales[3]).square().mean()
    )
    return data, physical
