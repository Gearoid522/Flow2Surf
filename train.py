#!/usr/bin/env python
"""Train the Flow2Surf conditional-flow point-cloud denoiser."""

import argparse
import logging
from pathlib import Path
import time

import jittor as jt
import numpy as np
from jittor import nn

from flow2surf.dataset import build_dataset, load_dataset_spec
from flow2surf.models import build_model
from flow2surf.runtime import (
    INFERENCE_FIELDS,
    configure_logging,
    format_duration,
    git_revision,
    load_config,
    timestamp,
    seed_runtime,
)
from flow2surf.training.checkpoint import CheckpointManager, read_resume
from flow2surf.training.losses import patch_loss
from flow2surf.training.optimization import ModelEMA, clip_grad_norm, get_lr

log = logging.getLogger("flow2surf.train")


def train_epoch(model, loader, optimizer, ema, cfg, epoch):
    """Optimize one epoch over patch batches."""
    model.train()
    frozen = [name for name, param in model.named_parameters() if param.is_stop_grad()]
    if frozen:
        names = ", ".join(frozen[:3])
        raise RuntimeError(f"Model parameters are frozen before training: {names}")
    total_loss = total_n = 0
    total_velocity = total_geometry = 0.0
    step_loss = step_vel = step_geometry = 0.0
    interval_n = 0
    interval_steps = 0
    geometry_weight = cfg["geometry_weight"]
    scale_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], epoch, 0, 2]))
    grad_clip = cfg["grad_clip"]
    clipping = grad_clip is not None and grad_clip > 0
    max_grad_norm = jt.float32(0.0).stop_grad() if clipping else None
    t0 = step_t0 = time.time()
    total_steps = len(loader)

    for i, batch in enumerate(loader):
        n = len(batch["noisy"])
        scale = scale_rng.uniform(*cfg["scale_range"], size=n).astype(np.float32)
        vel, geometry = patch_loss(model, batch, cfg, scale)
        loss = vel + geometry_weight * geometry
        loss_v = np.asarray(loss.numpy())
        vel_v = np.asarray(vel.numpy())
        geometry_v = np.asarray(geometry.numpy())
        batch_loss = float(loss_v.mean())
        batch_vel = float(vel_v.mean())
        batch_geometry = float(geometry_v.mean())
        optimizer.backward(loss.sum() / n)

        if clipping:
            grad_norm = clip_grad_norm(optimizer, grad_clip)
            max_grad_norm.update(jt.maximum(max_grad_norm, grad_norm))
        optimizer.step()
        if ema is not None:
            ema.update()

        total_loss += batch_loss * n
        total_velocity += batch_vel * n
        total_geometry += batch_geometry * n
        total_n += n
        step_loss += batch_loss * n
        step_vel += batch_vel * n
        step_geometry += batch_geometry * n
        interval_n += n
        interval_steps += 1

        if (i + 1) % cfg["log_interval"] == 0 or i + 1 == total_steps:
            inv = 1.0 / interval_n
            avg_loss = step_loss * inv
            avg_vel = step_vel * inv
            avg_geometry = step_geometry * inv
            step_loss = step_vel = step_geometry = 0.0
            interval_n = 0
            elapsed = time.time() - t0
            eta_s = (
                (time.time() - step_t0) / interval_steps * (total_steps - i - 1)
            )
            step_t0 = time.time()
            interval_steps = 0

            gn = f" | grad_norm_max {float(max_grad_norm):.4f}" if clipping else ""
            if clipping:
                max_grad_norm.update(0.0)
            log.info(
                f"  [{epoch}] step {i+1:4d}/{total_steps} | "
                f"loss {avg_loss:.6f} "
                f"(vel {avg_vel:.6f}, dcd {avg_geometry:.6f}) | "
                f"elapsed {format_duration(elapsed)} | eta {format_duration(eta_s)}{gn}"
            )

    return {"loss": total_loss / total_n, "velocity": total_velocity / total_n,
            "dcd": total_geometry / total_n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="datasets/PUNet.yaml")
    parser.add_argument("--resume", help="Full periodic checkpoint to resume")
    args = parser.parse_args()
    cfg = {
        key: value for key, value in load_config(args.config).items()
        if key not in INFERENCE_FIELDS
    }
    scale_range = np.asarray(cfg["scale_range"], dtype=np.float64)
    if (scale_range.shape != (2,) or not np.isfinite(scale_range).all()
            or scale_range[0] <= 0 or scale_range[1] < scale_range[0]):
        raise ValueError("scale_range must be a positive ordered pair")
    if min(cfg["geometry_weight"], cfg["dcd_alpha"]) <= 0:
        raise ValueError("Geometry weights and dcd_alpha must be positive")
    if cfg["epochs"] < 1 or cfg["checkpoint_interval"] < 1 or cfg["log_interval"] < 1:
        raise ValueError("Epoch, checkpoint and log intervals must be positive")
    if (type(cfg["pretrain_epochs"]) is not int
            or not cfg["warmup_epochs"] < cfg["pretrain_epochs"] <= cfg["epochs"]):
        raise ValueError("pretrain_epochs must be greater than warmup_epochs and at most epochs")
    if not 0 < cfg["lr_min"] <= cfg["refine_lr"] <= cfg["lr"]:
        raise ValueError("Require 0 < lr_min <= refine_lr <= lr")
    final_epoch = cfg["epochs"]
    checkpoint_interval = int(cfg["checkpoint_interval"])
    payload = read_resume(args.resume) if args.resume else None
    spec = load_dataset_spec(args.dataset)
    dataset_info = {"name": spec.name, "root": str(spec.root), "train": spec.train}
    if payload is not None:
        previous = payload["state"]
        if previous["config"] != cfg or previous["dataset"] != dataset_info:
            raise ValueError("Resume config and dataset must match the checkpoint")
        if previous["epoch"] >= final_epoch:
            raise ValueError("Resume checkpoint has already reached the requested final epoch")
    if payload is not None:
        output_dir = Path(args.resume).resolve().parent.parent
    else:
        output_dir = Path("outputs") / f"{spec.name.lower()}_{timestamp()}"
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    seed_runtime(cfg["seed"])
    jt.flags.use_cuda = 1
    start_epoch = payload["state"]["epoch"] + 1 if payload is not None else 1
    train_ds = build_dataset(cfg, spec, start_epoch)
    configure_logging(str(output_dir / "train.log"))
    log.info("Flow2Surf training")
    repo = str(Path(__file__).resolve().parent)
    log.info("Output: %s", output_dir)
    log.info("Config: %s | git %s", args.config, git_revision(repo))
    log.info("Dataset: %s (%s) | root %s", args.dataset, spec.name, spec.root)
    log.info("Train clouds: %d | batch size: %d | steps/epoch: %d",
             len(train_ds.paths), cfg["batch_size"], len(train_ds))
    log.info("Sampling: %s std=%s (%s) | alpha=U(0,1) | scale=%s",
             cfg["noise_type"], cfg["noise_std_range"], cfg["noise_std_sampling"],
             cfg["scale_range"])
    log.info("Training: epochs=%d | seed=%d | checkpoint interval=%d + final",
             final_epoch, cfg["seed"], checkpoint_interval)
    log.info("Loss: velocity + %g * DCD | dcd_alpha %g",
             cfg["geometry_weight"], cfg["dcd_alpha"])

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %.3fM parameters | EMA %s", n_params / 1e6,
             cfg["ema_decay"] if cfg["ema_decay"] else "off")

    optimizer = nn.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    decay = float(cfg["ema_decay"])
    if not 0 <= decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    ema = ModelEMA(model, decay) if decay else None
    manager = CheckpointManager(output_dir)
    state = {"epoch": 0, "checkpoint": None, "checkpoint_epoch": 0, "history": [],
             "config": cfg, "dataset": dataset_info}
    if payload is not None:
        state = manager.restore(model, optimizer, ema, payload)
        state["config"] = cfg
        log.info("Resume: %s (completed epoch %d)", args.resume, state["epoch"])
    manager.write_state(state)
    epoch_times = []
    try:
        for epoch in range(state["epoch"] + 1, final_epoch + 1):
            lr = get_lr(cfg, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            phase = "pretrain" if epoch <= cfg["pretrain_epochs"] else "refine"
            log.info("Epoch %d/%d | %s | lr %.2e", epoch, final_epoch, phase, lr)
            t0 = time.time()
            metrics = train_epoch(model, train_ds, optimizer, ema, cfg, epoch)
            state["epoch"] = epoch
            history = {"epoch": epoch, "lr": float(lr),
                       **{f"train_{key}": value for key, value in metrics.items()}}
            state["history"].append(history)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            eta = (sum(epoch_times[-10:]) / len(epoch_times[-10:])) * (
                final_epoch - epoch
            )

            log.info(
                f"  train | loss {metrics['loss']:.6f} "
                f"(vel {metrics['velocity']:.6f}, "
                f"dcd {metrics['dcd']:.6f}) | "
                f"epoch {format_duration(elapsed)} | "
                f"eta {format_duration(eta)}"
            )
            if epoch % checkpoint_interval == 0 or epoch == final_epoch:
                manager.save_checkpoint(model, optimizer, ema, state)
            manager.write_state(state)
    finally:
        train_ds.terminate()
    manager.save_final(ema.model if ema is not None else model)
    log.info("Training complete | epoch %d/%d", state["epoch"], final_epoch)


if __name__ == "__main__":
    main()
