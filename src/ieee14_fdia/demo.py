"""Synthetic end-to-end smoke example, intentionally separate from published results."""

from pathlib import Path
import numpy as np
from .physics import load_network, ac_measurements
from .io import write_json


def run_demo(network_path, models_path, output, device="cpu", epochs=2):
    from .training import train
    from .attacks import generate
    from .pipeline import calibrate, evaluate

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    state = np.c_[1 + rng.normal(0, 0.002, (384, 14)), rng.normal(0, 0.005, (384, 13))]
    network = load_network(network_path)
    z = ac_measurements(state, *network) + rng.normal(0, 0.001, (384, 28))
    # A mock noisy estimate exercises the schema; it is NOT a WLS claim.
    estimate = state + rng.normal(0, 1e-4, state.shape)
    data = output / "synthetic_normal.npz"
    np.savez_compressed(
        data, z_pu=z, state_true_pu_rad=state, normal_estimated_state=estimate
    )
    split = output / "split.json"
    write_json(split, dict(train_end=200, selection_end=260, calibration_end=320))
    config = output / "attack_config.json"
    write_json(
        config,
        dict(
            test_start=320,
            test_end=384,
            duration_steps=5,
            normal_gap_steps=7,
            replay_delay=12,
            seed=42,
            sampling_minutes=5,
        ),
    )
    artifacts = output / "artifacts"
    train(data, network, models_path, split, "LSTM", 42, artifacts, device, epochs)
    calibrate(data, artifacts, network, split, 0.05, device)
    generate(data, network, config, output / "attacks")
    evaluate(output / "attacks", artifacts, network, output / "evaluation", device)
    write_json(
        output / "DEMO_ONLY.json",
        dict(synthetic=True, mock_estimates_not_WLS=True, benchmark_results=False),
    )
    print("Synthetic smoke workflow complete; these metrics are NOT research results.")
