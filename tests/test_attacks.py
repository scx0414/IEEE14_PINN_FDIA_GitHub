import json
import numpy as np
from ieee14_fdia.attacks import generate
from ieee14_fdia.data import make_windows
from ieee14_fdia.physics import ac_measurements


def test_replay_scaling_and_continuous_history(tmp_path):
    rng = np.random.default_rng(3)
    G, B = np.eye(14) + 0.01 * np.ones((14, 14)), -np.eye(14) + 0.02 * np.ones((14, 14))
    state = np.c_[1 + rng.normal(0, 0.01, (160, 14)), rng.normal(0, 0.02, (160, 13))]
    z = ac_measurements(state, G, B) + rng.normal(0, 0.001, (160, 28))
    source = tmp_path / "normal.npz"
    np.savez(
        source, z_pu=z, state_true_pu_rad=state, normal_estimated_state=state.copy()
    )
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            dict(
                test_start=64,
                test_end=160,
                duration_steps=15,
                normal_gap_steps=48,
                replay_delay=12,
                seed=42,
            )
        )
    )
    folder = tmp_path / "attacks"
    generate(source, (G, B), cfg, folder)
    d = np.load(folder / "cases/replay_delay12_n3.npz")
    assert d["attack_label"].sum() == 30
    mask = d["target_node_mask"][0]
    channels = np.flatnonzero(np.r_[mask, mask])
    np.testing.assert_array_equal(d["z_pu"][:15, channels], z[52:67, channels])
    np.testing.assert_array_equal(d["state_true_pu_rad"], state[64:])
    windows = make_windows(d["z_pu"], 24, z[41:64])
    np.testing.assert_array_equal(windows[15, -16:-1], d["z_pu"][:15])
    scale = np.load(folder / "cases/scale_minus10_n3.npz")
    np.testing.assert_array_equal(
        scale["z_pu"][:15, channels], 0.9 * z[64:79, channels]
    )
