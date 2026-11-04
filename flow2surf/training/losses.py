"""Patch-level objectives for conditional-flow training."""

import jittor as jt

from ..flow import flow_state, flow_velocity, reconstruct_clean
from ..ops import nearest_squared_distances


def dcd_loss(pred, target, alpha):
    """Return density-aware Chamfer loss per point set.

    Reciprocal nearest-neighbor multiplicity penalizes many-to-one matches; an
    exponential kernel maps squared distances to a loss in [0, 1] for equal-size
    sets. Point-count ratios follow the original DCD definition for unequal sets.
    """
    batch_size, pred_count, _ = pred.shape
    target_count = target.shape[1]
    pred_distance, target_idx, target_distance, pred_idx = nearest_squared_distances(
        pred, target
    )
    target_multiplicity = jt.zeros((batch_size, target_count)).scatter(
        1,
        target_idx,
        jt.ones((batch_size, pred_count)),
        reduce="add",
    )
    pred_multiplicity = jt.zeros((batch_size, pred_count)).scatter(
        1,
        pred_idx,
        jt.ones((batch_size, target_count)),
        reduce="add",
    )
    # Every gathered multiplicity is positive: the querying point contributes one.
    pred_weight = (pred_count / target_count) / jt.gather(
        target_multiplicity, 1, target_idx
    )
    target_weight = (target_count / pred_count) / jt.gather(
        pred_multiplicity, 1, pred_idx
    )
    pred_term = (1.0 - pred_weight * jt.exp(-alpha * pred_distance)).mean(dim=1)
    target_term = (1.0 - target_weight * jt.exp(-alpha * target_distance)).mean(dim=1)
    return 0.5 * (pred_term + target_term)


def velocity_loss(pred_velocity, target_velocity):
    """Return mean squared L2 velocity error for each patch in the batch."""
    error = pred_velocity - target_velocity
    return (error * error).sum(dim=-1).mean(dim=1)


def patch_loss(model, batch, cfg, scale):
    """Return velocity and geometry losses on the known linear path."""
    noisy = jt.array(batch["noisy"])
    clean = jt.array(batch["clean"])
    target_velocity = flow_velocity(clean, noisy)
    seed_clean = jt.array(batch["seed_clean"]).unsqueeze(1)
    alpha = jt.array(batch["alpha"])
    alpha_view = alpha.reshape(-1, 1, 1)
    center = (1.0 - alpha_view) * seed_clean
    x_alpha = flow_state(clean, noisy, alpha_view) - center
    clean = clean - center
    # Scale inputs and restore predicted velocity to reference coordinate units.
    scale = jt.array(scale).reshape(-1, 1, 1)
    velocity = model(x_alpha.permute(0, 2, 1) * scale, alpha) / scale
    vel = velocity_loss(velocity, target_velocity)
    pred_clean = reconstruct_clean(x_alpha, velocity, alpha_view)
    return vel, dcd_loss(pred_clean, clean, cfg["dcd_alpha"])
