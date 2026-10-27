"""Command line entry point; every private/local path is supplied explicitly."""

import argparse
from pathlib import Path
from .io import read_json, write_json
from .physics import load_network


def main(argv=None):
    p = argparse.ArgumentParser(
        description="AC-PINN state estimation and FDIA detection"
    )
    subs = p.add_subparsers(dest="command", required=True)
    prep = subs.add_parser(
        "prepare",
        help="Convert an existing MATPOWER normal_cache.mat; no new power-flow simulation",
    )
    prep.add_argument("--cache", required=True)
    prep.add_argument("--output", required=True)
    imp = subs.add_parser(
        "import-legacy",
        help="Explicitly import existing final checkpoints/scalers/frozen5%% thresholds",
    )
    for name in ["checkpoints", "preprocessing", "thresholds", "output"]:
        imp.add_argument("--" + name, required=True)
    train = subs.add_parser("train")
    for name in ["data", "network", "config", "split", "output"]:
        train.add_argument("--" + name, required=True)
    train.add_argument(
        "--family", choices=["LSTM", "P_sLSTM", "xLSTM_Mixer"], required=True
    )
    train.add_argument("--seed", type=int, default=2024)
    train.add_argument(
        "--epochs",
        type=int,
        help="Override epochs in BOTH base/refine stages, for smoke runs",
    )
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    attack = subs.add_parser("generate-attacks")
    for name in ["data", "network", "config", "output"]:
        attack.add_argument("--" + name, required=True)
    export = subs.add_parser(
        "export-matlab",
        help="Export generated scenarios for optional blind AC-WLS checking",
    )
    for name in ["attacks", "network", "output"]:
        export.add_argument("--" + name, required=True)
    cal = subs.add_parser("calibrate")
    for name in ["data", "network", "artifacts", "split"]:
        cal.add_argument("--" + name, required=True)
    cal.add_argument("--alpha", type=float, default=0.05)
    cal.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate = subs.add_parser("evaluate")
    for name in ["attacks", "network", "artifacts", "output"]:
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--save-predictions", action="store_true")
    demo = subs.add_parser(
        "demo", help="Synthetic smoke workflow; NOT a benchmark result"
    )
    demo.add_argument("--network", required=True)
    demo.add_argument("--config", required=True)
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    demo.add_argument("--epochs", type=int, default=2)
    args = p.parse_args(argv)
    if args.command == "prepare":
        import numpy as np
        from scipy.io import loadmat

        n = loadmat(args.cache, simplify_cells=True)
        if not n["normal_wls_converged"].all():
            raise ValueError("Normal cache includes unconverged WLS estimates")
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            z_pu=n["z_normal"],
            state_true_pu_rad=n["state_true"],
            normal_estimated_state=n["normal_wls_state"],
        )
        return
    if args.command == "export-matlab":
        import numpy as np
        from scipy.io import savemat

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        G, B = load_network(args.network)
        savemat(out / "network.mat", dict(G=G, B=B))
        manifest = read_json(Path(args.attacks) / "manifest.json")
        for group in manifest["groups"]:
            d = np.load(Path(args.attacks) / group["path"])
            savemat(
                out / f"{group['id']}.mat",
                {k: d[k] for k in d.files},
                do_compression=True,
                oned_as="column",
            )
        return
    if args.command in ("train", "evaluate", "calibrate", "demo", "import-legacy"):
        import torch

        torch.set_num_threads(4)
    if args.command == "import-legacy":
        from .inference import import_legacy

        print(
            "Imported checkpoints:",
            import_legacy(
                args.checkpoints, args.preprocessing, args.thresholds, args.output
            ),
        )
    elif args.command == "train":
        from .training import train

        print(
            train(
                args.data,
                load_network(args.network),
                args.config,
                args.split,
                args.family,
                args.seed,
                args.output,
                args.device,
                args.epochs,
            )
        )
    elif args.command == "generate-attacks":
        from .attacks import generate

        print(generate(args.data, load_network(args.network), args.config, args.output))
    elif args.command == "calibrate":
        from .pipeline import calibrate

        calibrate(
            args.data,
            args.artifacts,
            load_network(args.network),
            args.split,
            args.alpha,
            args.device,
        )
    elif args.command == "evaluate":
        from .pipeline import evaluate

        evaluate(
            args.attacks,
            args.artifacts,
            load_network(args.network),
            args.output,
            args.device,
            args.save_predictions,
        )
    elif args.command == "demo":
        from .demo import run_demo

        run_demo(args.network, args.config, args.output, args.device, args.epochs)
