import numpy as np
import torch
from scipy.stats import weibull_min
from ieee14_fdia.physics import ac_measurements, torch_measurements
from ieee14_fdia.detector import fit_weibull, classify, event_metrics


def test_numpy_torch_ac_agree_and_gradient():
    rng = np.random.default_rng(7)
    G, B = rng.normal(size=(14, 14)), rng.normal(size=(14, 14))
    state = np.c_[1 + rng.normal(0, 0.01, (8, 14)), rng.normal(0, 0.05, (8, 13))]
    x = torch.tensor(state, requires_grad=True)
    power = torch_measurements(x, torch.tensor(G), torch.tensor(B))
    np.testing.assert_allclose(
        power.detach().numpy(), ac_measurements(state, G, B), atol=1e-12
    )
    power.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_system_threshold_controls_system_statistic():
    system = 0.2 * np.random.default_rng(4).weibull(1.7, 10000)
    node = np.repeat(system[:, None], 14, axis=1)
    fit = fit_weibull(node, 0.05)
    np.testing.assert_allclose(
        weibull_min.sf(fit["threshold"], fit["shape"], scale=fit["scale"]), 0.05
    )
    assert 0.03 < classify(node, fit["threshold"]).mean() < 0.07


def test_event_delay_does_not_count_misses_as_zero():
    es = event_metrics(
        np.array([False, True, False, False, False]), np.array([1, 1, 0, 2, 2]), 5
    )
    assert es[0] == dict(event_id=1, detected=True, delay_minutes=5)
    assert es[1] == dict(event_id=2, detected=False, delay_minutes=None)
