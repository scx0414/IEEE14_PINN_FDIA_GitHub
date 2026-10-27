"""Continuous replay, scaling and AC angle FDIA; all randomness is local."""

from pathlib import Path
import random
import numpy as np
from .io import load_normal, write_json, read_json, digest
from .physics import ac_measurements


def generate(data_path, network, config_path, output):
    d, cfg = load_normal(data_path), read_json(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    lo, hi = cfg["test_start"], cfg.get("test_end", len(d["z_pu"]))
    duration, gap = cfg["duration_steps"], cfg["normal_gap_steps"]
    if not 24 <= lo < hi <= len(d["z_pu"]) or duration < 1 or gap < 0:
        raise ValueError("Invalid test interval/duration/gap")
    z, x = d["z_pu"][lo:hi], d["state_true_pu_rad"][lo:hi]
    starts = np.arange(0, len(z) - duration + 1, duration + gap)
    if not len(starts):
        raise ValueError("No complete attack episode fits the test interval")
    label, event_id = np.zeros(len(z), np.uint8), np.zeros(len(z), np.int32)
    for e, start in enumerate(starts, 1):
        label[start : start + duration], event_id[start : start + duration] = 1, e
    np.savez_compressed(
        out / "normal.npz",
        z_pu=z,
        state_true_pu_rad=x,
        original_index=np.arange(lo, hi),
        past_z_pu=d["z_pu"][lo - 23 : lo],
    )
    rng = random.Random(cfg["seed"])
    common = [sorted(rng.sample(range(1, 14), 3)) for _ in starts]
    groups = [
        dict(
            id=f"replay_delay{cfg.get('replay_delay', 12)}_n3",
            kind="replay",
            nodes=3,
            delay=cfg.get("replay_delay", 12),
        ),
        dict(id="scale_plus10_n3", kind="scale", nodes=3, scale=0.1),
        dict(id="scale_minus10_n3", kind="scale", nodes=3, scale=-0.1),
    ]
    for count in [1, 2, 3]:
        for strength in [0.2, 0.3]:
            groups.append(
                dict(
                    id=f"fdia_angle_n{count}_s{int(strength * 100)}",
                    kind="FDIA",
                    nodes=count,
                    strength=strength,
                )
            )
    if "normal_estimated_state" not in d:
        raise ValueError(
            "FDIA requires normal_estimated_state from normal AC-WLS; do not substitute true labels"
        )
    base = d["normal_estimated_state"][lo:hi]
    if base.shape != x.shape or not np.isfinite(base).all():
        raise ValueError("Invalid normal WLS state array")
    for gi, g in enumerate(groups):
        rng, nr = random.Random(cfg["seed"] + 1 + gi), np.random.RandomState(
            cfg["seed"] + 1 + gi
        )
        attacked, targets, ratios = (
            z.copy(),
            np.zeros((len(z), 14), bool),
            np.zeros((len(z), 14)),
        )
        for e, start in enumerate(starts):
            ii = np.arange(start, start + duration)
            nodes = (
                common[e]
                if g["kind"] != "FDIA"
                else sorted(rng.sample(range(1, 14), g["nodes"]))
            )
            channels = nodes + [j + 14 for j in nodes]
            targets[np.ix_(ii, nodes)] = True
            if g["kind"] == "replay":
                if lo + start - g["delay"] < 0:
                    raise ValueError("Replay source predates available data")
                attacked[np.ix_(ii, channels)] = d["z_pu"][
                    np.ix_(lo + ii - g["delay"], channels)
                ]
            elif g["kind"] == "scale":
                attacked[np.ix_(ii, channels)] = z[np.ix_(ii, channels)] * (
                    1 + g["scale"]
                )
            else:
                ratio = (g["strength"] - 0.1 + 0.1 * nr.rand(g["nodes"])) * (
                    -1 + 2 * nr.randint(2, size=g["nodes"])
                )
                fake = base[ii].copy()
                fake[:, np.array(nodes) + 13] *= 1 + ratio
                ratios[np.ix_(ii, nodes)] = ratio
                attacked[ii] = (
                    z[ii]
                    + ac_measurements(fake, *network)
                    - ac_measurements(base[ii], *network)
                )
        if not np.isfinite(attacked).all():
            raise ValueError("Attack construction produced nonfinite measurements")
        if not np.all(np.any(attacked[label == 1] != z[label == 1], axis=1)):
            raise ValueError(
                "An attack has no measurement effect; check target states/network"
            )
        name = f"cases/{g['id']}.npz"
        np.savez_compressed(
            out / name,
            z_pu=attacked,
            state_true_pu_rad=x,
            attack_label=label,
            event_id=event_id,
            target_node_mask=targets,
            state_ratio=ratios,
            original_index=np.arange(lo, hi),
        )
        g.update(path=name, sha256=digest(out / name))
    write_json(
        out / "manifest.json",
        dict(
            groups=groups,
            configuration=cfg,
            test_count=len(z),
            attack_points=int(label.sum()),
            normal_points=int((label == 0).sum()),
            events=len(starts),
            normal_sha256=digest(out / "normal.npz"),
            source_sha256=digest(data_path),
            bdd="Not computed by Python generator; optional MATLAB checker in matlab/",
            interpretation="15-step continuous attacks; no attack history reset; parameters fixed per episode; paper-parameter adaptation, not exact replication",
        ),
    )
    return out / "manifest.json"
