"""Portable eager training; same loss/model recipe, not bitwise CUDA Graph replay."""

from pathlib import Path
import copy
import math
import random
import numpy as np
import torch
from .models import build_model
from .data import fit_scalers, make_windows, normalize
from .physics import pinn_losses
from .io import load_normal, read_json, write_json, digest
from .inference import resolve_device


def train(
    data_path,
    network,
    config_path,
    split_path,
    family,
    seed,
    output,
    device="auto",
    epochs=None,
):
    d, plans, split = (
        load_normal(data_path),
        read_json(config_path),
        read_json(split_path),
    )
    cfgs = copy.deepcopy(plans[family])
    if epochs is not None:
        for cfg in cfgs.values():
            cfg["epochs"] = epochs
    lo, hi = split["train_end"], split["selection_end"]
    if not 24 < lo < hi <= len(d["z_pu"]):
        raise ValueError("Invalid training/selection split")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    prep = fit_scalers(d["z_pu"], d["state_true_pu_rad"], lo)
    prepfile = out / "preprocessing.npz"
    if prepfile.exists():
        existing = np.load(prepfile)
        for k, v in prep.items():
            np.testing.assert_array_equal(
                existing[k], v, err_msg="Output folder has incompatible scalers"
            )
    else:
        np.savez(prepfile, **prep)
    window = cfgs["base"]["window"]
    windows = make_windows(d["z_pu"][:hi], window)
    z = torch.tensor(normalize(windows, prep), device=device)
    truth = torch.tensor(
        d["state_true_pu_rad"][:hi], dtype=torch.float32, device=device
    )
    pq = torch.tensor(d["z_pu"][:hi], dtype=torch.float32, device=device)
    ym = torch.tensor(prep["state_mean"], dtype=torch.float32, device=device)
    ys = torch.tensor(prep["state_scale"], dtype=torch.float32, device=device)
    G, B = [torch.tensor(n, dtype=torch.float32, device=device) for n in network]
    train_ids, val_ids = torch.arange(23, lo, device=device), torch.arange(
        lo, hi, device=device
    )
    model = build_model(family, cfgs["base"]).to(device)
    history = []

    def validation():
        model.eval()
        with torch.inference_mode():
            predicted = torch.cat(
                [ym + model(z[j - window + 1]) * ys for j in val_ids.split(256)]
            )
        return float((predicted - truth[val_ids]).square().mean())

    for stage, cfg in cfgs.items():
        if cfg["window"] != window:
            raise ValueError("Base/refine architecture must match")
        peak = cfg.get("lr", cfg.get("learning_rate"))
        optimizer = torch.optim.Adam(model.parameters(), lr=peak)
        best = float("inf") if stage == "base" else validation()
        weights, best_epoch = copy.deepcopy(model.state_dict()), 0
        for ep in range(1, cfg["epochs"] + 1):
            if cfg["schedule"] == "fixed":
                ratio = 1.0
            else:
                ratio = min(1.0, ep / 5) * (
                    0.02
                    + 0.98
                    * 0.5
                    * (
                        1
                        + math.cos(math.pi * max(0, ep - 5) / max(1, cfg["epochs"] - 5))
                    )
                )
            for group in optimizer.param_groups:
                group["lr"] = peak * ratio
            order = train_ids[torch.randperm(len(train_ids), device=device)]
            model.train()
            for j in order.split(cfg["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                pred = ym + model(z[j - window + 1]) * ys
                ld, lp = pinn_losses(pred, truth[j], pq[j], G, B)
                loss = (1 - cfg["physics_weight"]) * ld + cfg["physics_weight"] * lp
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
            if ep == 1 or ep % 5 == 0 or ep == cfg["epochs"]:
                score = validation()
                history.append(dict(stage=stage, epoch=ep, validation_state_mse=score))
                if score < best:
                    best, weights, best_epoch = (
                        score,
                        copy.deepcopy(model.state_dict()),
                        ep,
                    )
                print(
                    f"{family} seed={seed} {stage} epoch={ep} valMSE={score:.8g}",
                    flush=True,
                )
            if ep >= 80 and ep - best_epoch >= 50:
                break
        model.load_state_dict(weights)
    target = out / f"{family}_{seed}.pt"
    torch.save(
        dict(
            family=family,
            seed=seed,
            config=cfgs["refine"],
            state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        ),
        target,
    )
    write_json(out / f"{family}_{seed}_history.json", history)
    write_json(
        out / f"{family}_{seed}_training.json",
        dict(
            data_sha256=digest(data_path),
            split=split,
            plans=cfgs,
            task="current-input AC state estimation",
            selection="normal validation state MSE",
            implementation="portable eager; not bitwise historical CUDA-Graph training",
        ),
    )
    return target
