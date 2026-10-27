"""Portable file I/O; user paths are arguments, never import-time constants."""

from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_normal(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    z, x = d["z_pu"], d["state_true_pu_rad"]
    if z.ndim != 2 or z.shape[1] != 28 or x.shape != (len(z), 27):
        raise ValueError("Expected z_pu[N,28] and state_true_pu_rad[N,27]")
    if not np.isfinite(z).all() or not np.isfinite(x).all():
        raise ValueError("Normal input/state arrays must be finite")
    if "attack_label" in d and np.any(d["attack_label"]):
        raise ValueError("Normal source must not contain attacks")
    return d
