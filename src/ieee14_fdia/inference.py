from pathlib import Path
import numpy as np
import torch
from .models import build_model
from .data import normalize
from .io import read_json, write_json, digest


def resolve_device(request="auto"):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return request


class Predictor:
    def __init__(self, checkpoint, scalers, device="auto", batch_size=256):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.device, self.batch_size = resolve_device(device), batch_size
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.family, self.config = ck["family"], ck["config"]
        self.model = build_model(self.family, self.config).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.scalers = scalers
        self.mean = torch.as_tensor(
            scalers["state_mean"], dtype=torch.float32, device=self.device
        )
        self.scale = torch.as_tensor(
            scalers["state_scale"], dtype=torch.float32, device=self.device
        )

    def predict(self, windows):
        if windows.shape[1] < self.config["window"]:
            raise ValueError(
                f"Checkpoint needs{self.config['window']} steps, got{windows.shape[1]}"
            )
        z = normalize(windows, self.scalers)
        values = []
        with torch.inference_mode():
            for start in range(0, len(z), self.batch_size):
                batch = torch.as_tensor(
                    z[start : start + self.batch_size].copy(), device=self.device
                )
                values.append(
                    (self.mean + self.model(batch) * self.scale).cpu().numpy()
                )
        return np.concatenate(values).astype(np.float64)


def import_legacy(checkpoints, preprocessing, thresholds, output):
    """Explicit opt-in local import; weights/data remain ignored by Git."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prep = np.load(preprocessing, allow_pickle=False)
    np.savez(
        output / "preprocessing.npz",
        **{
            k: prep[k]
            for k in ("input_mean", "input_scale", "state_mean", "state_scale")
        },
    )
    old = read_json(thresholds)
    imported = []
    for entry in old:
        name = f"{entry['model']}_{entry['seed']}.pt"
        source = Path(checkpoints) / name
        ck = torch.load(source, map_location="cpu", weights_only=True)
        if ck["family"] != entry["model"] or ck["seed"] != entry["seed"]:
            raise ValueError("Checkpoint family/seed mismatch")
        # Remove experiment-specific provenance fields; tensor values/configuration unchanged.
        torch.save(
            {k: ck[k] for k in ("family", "seed", "config", "state_dict")},
            output / name,
        )
        imported.append(
            dict(
                model=entry["model"],
                seed=entry["seed"],
                shape=entry["shape"],
                scale=entry["scale"],
                threshold=entry["threshold"],
                alpha=0.05,
                checkpoint=name,
                checkpoint_sha256=digest(output / name),
                calibration_system_fpr=entry["calibration_system_fpr"],
            )
        )
    write_json(output / "thresholds.json", imported)
    write_json(
        output / "provenance.json",
        dict(
            source_thresholds_sha256=digest(thresholds),
            preprocessing_sha256=digest(output / "preprocessing.npz"),
            checkpoint_names=[r["checkpoint"] for r in imported],
            mode="imported frozen weights/scalers/system5percent Weibull thresholds; no fitting",
        ),
    )
    return len(imported)
