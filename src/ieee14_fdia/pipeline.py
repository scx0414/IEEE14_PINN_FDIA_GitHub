from pathlib import Path
import numpy as np
from .io import load_normal, read_json, write_json, write_csv, digest
from .data import make_windows
from .physics import node_residual_scores
from .detector import fit_weibull, classify, confusion, event_metrics


def checkpoint_files(artifacts):
    files = sorted(
        p for p in Path(artifacts).glob("*.pt") if not p.stem.endswith("_parent")
    )
    if not files:
        raise ValueError("No final checkpoints found")
    return files


def calibrate(data_path, artifacts, network, split_path, alpha=0.05, device="auto"):
    from .inference import Predictor

    d, split = load_normal(data_path), read_json(split_path)
    lo, hi = split["selection_end"], split["calibration_end"]
    if not 24 <= lo < hi <= len(d["z_pu"]):
        raise ValueError("Invalid normal calibration interval")
    artifacts = Path(artifacts)
    scalers = np.load(artifacts / "preprocessing.npz")
    results = []
    for ck in checkpoint_files(artifacts):
        predictor = Predictor(ck, scalers, device)
        window = predictor.config["window"]
        x = make_windows(d["z_pu"][lo:hi], window, d["z_pu"][:lo])
        score = node_residual_scores(predictor.predict(x), d["z_pu"][lo:hi], *network)
        entry = fit_weibull(score, alpha)
        # Current file names encode seed; checkpoint carries definitive metadata.
        import torch

        metadata = torch.load(ck, map_location="cpu", weights_only=True)
        entry.update(
            model=metadata["family"],
            seed=metadata["seed"],
            checkpoint=ck.name,
            checkpoint_sha256=digest(ck),
        )
        results.append(entry)
    write_json(artifacts / "thresholds.json", results)
    write_json(
        artifacts / "provenance.json",
        dict(
            normal_data_sha256=digest(data_path),
            calibration_interval=[lo, hi],
            preprocessing_sha256=digest(artifacts / "preprocessing.npz"),
            mode="normal-only calibration",
        ),
    )
    return results


def evaluate(
    attacks, artifacts, network, output, device="auto", save_predictions=False
):
    from .inference import Predictor

    attacks, artifacts, output = Path(attacks), Path(artifacts), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(attacks / "manifest.json")
    normal = np.load(attacks / "normal.npz")
    if digest(attacks / "normal.npz") != manifest["normal_sha256"]:
        raise ValueError("Normal file hash mismatch")
    threshold_file = artifacts / "thresholds.json"
    frozen_hash = digest(threshold_file)
    thresholds = read_json(threshold_file)
    scalers = np.load(artifacts / "preprocessing.npz")
    provenance = read_json(artifacts / "provenance.json")
    if provenance["preprocessing_sha256"] != digest(artifacts / "preprocessing.npz"):
        raise ValueError("Preprocessing does not match calibrated artifacts")
    rows, normals, events = [], [], []
    for t in thresholds:
        ck = artifacts / t["checkpoint"]
        if digest(ck) != t["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after calibration/import")
        predictor = Predictor(ck, scalers, device)
        history = predictor.config["window"]
        x = make_windows(normal["z_pu"], history, normal["past_z_pu"])
        node = node_residual_scores(predictor.predict(x), normal["z_pu"], *network)
        normals.append(
            dict(
                model=t["model"],
                seed=t["seed"],
                clean_test_system_fpr=float(classify(node, t["threshold"]).mean()),
                threshold=t["threshold"],
            )
        )
        for group in manifest["groups"]:
            path = attacks / group["path"]
            if digest(path) != group["sha256"]:
                raise ValueError("Attack file hash mismatch")
            d = np.load(path)
            np.testing.assert_array_equal(d["original_index"], normal["original_index"])
            # Use actual preceding attacked observations; never repair history from normal.
            x = make_windows(d["z_pu"], history, normal["past_z_pu"])
            predicted = predictor.predict(x)
            node = node_residual_scores(predicted, d["z_pu"], *network)
            alarm = classify(node, t["threshold"])
            es = event_metrics(
                alarm,
                d["event_id"],
                manifest["configuration"].get("sampling_minutes", 5),
            )
            detected = [e["delay_minutes"] for e in es if e["detected"]]
            row = dict(
                model=t["model"],
                seed=t["seed"],
                scenario=group["id"],
                kind=group["kind"],
                nodes=group["nodes"],
                **confusion(alarm, d["attack_label"]),
                event_detection_rate=float(np.mean([e["detected"] for e in es])),
                detected_median_delay_minutes=(
                    float(np.median(detected)) if detected else None
                ),
            )
            rows.append(row)
            events.extend(
                dict(model=t["model"], seed=t["seed"], scenario=group["id"], **e)
                for e in es
            )
            if save_predictions:
                np.savez_compressed(
                    output / f"{ck.stem}_{group['id']}.npz",
                    estimated_states=predicted,
                    node_scores=node,
                    alarm=alarm,
                    threshold=t["threshold"],
                )
            print(
                f"{t['model']} seed={t['seed']} {group['id']}: recall={100 * row['recall']:.2f}% FPR={100 * row['fpr']:.2f}%",
                flush=True,
            )
    if digest(threshold_file) != frozen_hash:
        raise RuntimeError("Threshold file changed during evaluation")
    summaries = []
    for model in sorted(set(r["model"] for r in rows)):
        for scenario in (g["id"] for g in manifest["groups"]):
            selected = [
                r for r in rows if r["model"] == model and r["scenario"] == scenario
            ]
            s = dict(model=model, scenario=scenario)
            for key in [
                "recall",
                "fpr",
                "precision",
                "f1",
                "accuracy",
                "balanced_accuracy",
                "event_detection_rate",
            ]:
                values = [r[key] for r in selected]
                s[key + "_mean"] = float(np.mean(values))
                s[key + "_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            summaries.append(s)
    for name, values in [
        ("metrics", rows),
        ("normal_metrics", normals),
        ("events", events),
        ("summary", summaries),
    ]:
        write_json(output / f"{name}.json", values)
        write_csv(output / f"{name}.csv", values)
    write_json(
        output / "protocol.json",
        dict(
            manifest_sha256=digest(attacks / "manifest.json"),
            threshold_sha256=frozen_hash,
            frozen_thresholds=True,
            retraining=False,
            history_reset=False,
            normal_counterfactual_used_only_for_diagnostic=True,
        ),
    )
    return rows
