"""Denoise one XYZ file or the fixed noisy clouds of a benchmark dataset."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import jittor as jt
import numpy as np

from flow2surf.dataset import load_dataset_spec, load_points, test_cloud_paths
from flow2surf.inference import denoise_cloud
from flow2surf.models import build_model
from flow2surf.runtime import configure_logging, format_duration, load_config, seed_runtime
from flow2surf.training.checkpoint import load_model_weights

log = logging.getLogger("flow2surf.predict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="One XYZ file")
    inputs.add_argument("--dataset", help="Benchmark descriptor, e.g. datasets/PCNet.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, help="Integration steps per pass; default from config")
    parser.add_argument("--passes", type=int, default=1, help="Complete denoising passes; default 1")
    parser.add_argument("--noise", type=float, help="Select a benchmark noise level in percent")
    parser.add_argument("--out", required=True, help="XYZ file for --input, directory for --dataset")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        if args.steps < 1:
            parser.error("--steps must be positive")
        cfg["num_steps"] = args.steps
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.noise is not None and args.input:
        parser.error("--noise requires --dataset")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    if args.input:
        noisy = load_points(args.input)
    else:
        spec = load_dataset_spec(args.dataset)
        inputs = list(test_cloud_paths(spec, noise=args.noise))
    configure_logging()
    seed_runtime(cfg["inference_seed"])
    jt.flags.use_cuda = 1
    model = build_model(cfg)
    load_model_weights(model, args.model)
    model.eval()
    log.info("Model: %s", args.model)
    log.info("Config: %s", args.config)
    if args.dataset:
        log.info("Dataset: %s (%s) | %d shapes", args.dataset, spec.name, len(inputs))
    else:
        log.info("Input: %s | %d points", args.input, len(noisy))
    log.info(
        f"Inference: steps={cfg['num_steps']} "
        f"passes={args.passes} seed_k={cfg['seed_k']} "
        f"powers={cfg['alpha_clean_power']:g}/{cfg['alpha_noisy_power']:g} "
        f"beta={cfg['patch_agg_beta']:g}"
    )
    if args.input:
        jt.sync_all()
        start = perf_counter()
        pred = denoise_cloud(model, noisy, cfg, (Path(args.input).stem,),
                             num_passes=args.passes)
        jt.sync_all()
        predict_seconds = perf_counter() - start
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, pred, fmt="%.9g")
        log.info("Prediction time: %.2fs", predict_seconds)
        log.info("Saved: %s", out)
        return
    predict_seconds = 0.0
    for index, (condition, noisy_path) in enumerate(inputs, 1):
        noisy = load_points(noisy_path)
        identity = (spec.name, condition, noisy_path.stem)
        jt.sync_all()
        start = perf_counter()
        pred = denoise_cloud(model, noisy, cfg, identity,
                             num_passes=args.passes)
        jt.sync_all()
        predict_seconds += perf_counter() - start
        directory = out / condition
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / noisy_path.name, pred, fmt="%.9g")
        if index % 10 == 0 or index == len(inputs):
            log.info("[%d/%d] %s/%s", index, len(inputs), condition, noisy_path.name)
    log.info("Prediction time: %s total | %.2fs/shape",
             format_duration(predict_seconds), predict_seconds / len(inputs))
    log.info("Saved: %s | %d shapes", out, len(inputs))


if __name__ == "__main__":
    main()
