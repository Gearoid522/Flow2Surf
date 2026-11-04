"""Evaluate fixed PU-Net/PCNet noisy clouds against released references."""

import argparse
import logging
from pathlib import Path

import jittor as jt

from flow2surf.dataset import load_dataset_spec
from flow2surf.evaluation import evaluate_benchmark, write_evaluation
from flow2surf.models import build_model
from flow2surf.runtime import configure_logging, format_duration, load_config, seed_runtime, timestamp
from flow2surf.training.checkpoint import load_model_weights

log = logging.getLogger("flow2surf.eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="datasets/PUNet.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, help="Integration steps per pass; default from config")
    parser.add_argument("--passes", type=int, default=1, help="Complete denoising passes; default 1")
    parser.add_argument("--noise", type=float, help="Select a test noise level in percent, e.g. 2.5")
    parser.add_argument(
        "--meshes", type=int, help="Maximum shapes per condition; default all"
    )
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        if args.steps < 1:
            parser.error("--steps must be positive")
        cfg["num_steps"] = args.steps
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.meshes is not None and args.meshes < 1:
        parser.error("--meshes must be positive")
    spec = load_dataset_spec(args.dataset)
    out = Path(args.out or f"results/{spec.name.lower()}_{timestamp()}.json")
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    configure_logging()
    seed_runtime(cfg["inference_seed"])
    jt.flags.use_cuda = 1
    model = build_model(cfg)
    load_model_weights(model, args.model)
    model.eval()
    log.info("Model: %s", args.model)
    log.info("Config: %s", args.config)
    log.info(f"Dataset: {args.dataset} ({spec.name})")
    log.info(
        f"Inference: steps={cfg['num_steps']} "
        f"passes={args.passes} seed_k={cfg['seed_k']} "
        f"powers={cfg['alpha_clean_power']:g}/{cfg['alpha_noisy_power']:g} "
        f"beta={cfg['patch_agg_beta']:g}"
    )
    rows = evaluate_benchmark(model, spec, cfg, args.meshes,
                              num_passes=args.passes, noise=args.noise)
    result = write_evaluation(out, spec, args.dataset, args.model, cfg, rows,
                              num_passes=args.passes, noise=args.noise)
    log.info("Results | CD/P2M x1e4 | prediction seconds per shape")
    log.info("%-14s %6s %10s %10s %12s", "Condition", "Shapes", "CD", "P2M", "Predict (s)")
    for condition, metrics in result["conditions"].items():
        log.info(
            "%-14s %6d %10.4f %10.4f %12.2f",
            condition,
            metrics["num_shapes"],
            metrics["cd"] * 1e4,
            metrics["p2m"] * 1e4,
            metrics["predict_seconds"],
        )
    predict_seconds = sum(row["predict_seconds"] for row in rows)
    log.info("Prediction time: %s total | %.2fs/shape",
             format_duration(predict_seconds), predict_seconds / len(rows))
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()
