"""Deterministic patch-wise reverse-flow inference."""

import hashlib

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from .dataset import normalize_points
from .flow import alpha_schedule


def make_rng(seed, *parts):
    """Create a deterministic NumPy generator from a seed and namespace parts."""
    key = "::".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little") % (2**32))


def _farthest_point_indices(points, count, rng):
    """Select count farthest-point indices from points shaped (N, 3)."""
    num_points = points.shape[0]
    selected = np.empty(count, dtype=np.int64)
    selected[0] = rng.integers(num_points)
    min_distance = np.full(num_points, np.inf)
    distance = np.empty(num_points, dtype=points.dtype)
    workspace = np.empty_like(distance)

    for i in range(1, count):
        center = points[selected[i - 1]]
        np.subtract(points[:, 0], center[0], out=distance)
        np.square(distance, out=distance)
        np.subtract(points[:, 1], center[1], out=workspace)
        np.square(workspace, out=workspace)
        np.add(distance, workspace, out=distance)
        np.subtract(points[:, 2], center[2], out=workspace)
        np.square(workspace, out=workspace)
        np.add(distance, workspace, out=distance)
        np.minimum(min_distance, distance, out=min_distance)
        selected[i] = min_distance.argmax()

    return selected


def patch_seed_count(num_points, patch_size, seed_k):
    """Return the patch-center count for one point cloud."""
    num_points = int(num_points)
    patch_size = int(patch_size)
    seed_k = int(seed_k)
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if seed_k < 1:
        raise ValueError(f"seed_k must be >= 1, got {seed_k}")
    effective_patch_size = min(patch_size, num_points)
    return min(
        num_points,
        max(1, int(seed_k * num_points / effective_patch_size)),
    )


def patch_seed_indices(points, patch_size, seed_k, rng):
    """Select deterministic farthest-point patch centers from an (N, 3) cloud."""
    num_patches = patch_seed_count(len(points), patch_size, seed_k)
    return _farthest_point_indices(points, num_patches, rng)


class PatchField:
    """Fixed noisy-input neighborhoods and center-weighted velocity fusion."""

    def __init__(self, points, patch_size, seed_k, *, patch_agg_beta,
                 rng=None, patch_seed_idx=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.num_points = len(points)
        self.patch_size = min(patch_size, self.num_points)
        num_patches = patch_seed_count(self.num_points, patch_size, seed_k)
        if patch_seed_idx is None:
            self.seed_idx = patch_seed_indices(points, patch_size, seed_k, rng)
        else:
            self.seed_idx = np.asarray(patch_seed_idx, dtype=np.int64)
            if self.seed_idx.shape != (num_patches,):
                raise ValueError(f"patch_seed_idx must have shape ({num_patches},)")
            if np.any((self.seed_idx < 0) | (self.seed_idx >= self.num_points)):
                raise ValueError("patch_seed_idx contains an out-of-range point index")

        distances, self.neighbor_idx = cKDTree(points).query(
            points[self.seed_idx], k=self.patch_size, workers=-1
        )
        if self.patch_size == 1:
            distances = distances[:, None]
            self.neighbor_idx = self.neighbor_idx[:, None]
        distances = distances / (distances[:, -1:] + 1e-8)
        self.point_idx = self.neighbor_idx.reshape(-1)
        radial_position = distances.reshape(-1)
        self.weights = np.exp(-patch_agg_beta * radial_position * radial_position)
        self.weight_sum = np.bincount(
            self.point_idx, weights=self.weights, minlength=self.num_points
        )

    def predict(self, model, points, alpha, num_chunks):
        """Evaluate the fused field without retaining a gradient graph."""
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        num_patches = len(self.seed_idx)
        chunk_size = (num_patches + num_chunks - 1) // num_chunks
        patches = points[self.neighbor_idx]
        patch_velocity = np.zeros_like(patches)

        with jt.no_grad():
            for start in range(0, num_patches, chunk_size):
                batch = patches[start : start + chunk_size]
                batch_size = batch.shape[0]
                centers = points[self.seed_idx[start : start + batch_size]][:, None, :]
                model_input = jt.array(batch - centers).permute(0, 2, 1)
                alpha_input = jt.full((batch_size,), float(alpha))
                velocity = model(model_input, alpha_input).numpy()
                patch_velocity[start : start + batch_size] = velocity

        flat_votes = patch_velocity.reshape(-1, 3)
        velocity_sum = np.stack(
            [
                np.bincount(
                    self.point_idx,
                    weights=self.weights * flat_votes[:, channel],
                    minlength=self.num_points,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        return (velocity_sum / (self.weight_sum[:, None] + 1e-8)).astype(np.float32)


def patch_denoise(
    model,
    pc_noisy,
    patch_size,
    seed_k,
    num_steps,
    num_chunks,
    *,
    patch_agg_beta,
    alpha_clean_power,
    alpha_noisy_power,
    rng=None,
    patch_seed_idx=None,
):
    """Integrate alpha 1 to 0 with noisy-input patches fixed within the pass."""
    field = PatchField(pc_noisy, patch_size, seed_k,
                       patch_agg_beta=patch_agg_beta, rng=rng,
                       patch_seed_idx=patch_seed_idx)

    alphas = alpha_schedule(
        num_steps,
        alpha_clean_power,
        alpha_noisy_power,
    )
    current = pc_noisy.copy()
    for step in range(num_steps - 1, -1, -1):
        alpha_current = float(alphas[step])
        alpha_previous = float(alphas[step - 1]) if step > 0 else 0.0
        alpha_step = alpha_current - alpha_previous

        velocity = field.predict(model, current, alpha_current, num_chunks)
        current = current - alpha_step * velocity

    return current.astype(np.float32)


def denoise_passes(model, noisy, cfg, identity, *, num_passes, patch_seed_idx=None):
    """Yield complete passes, rebuilding patches in one fixed coordinate frame.

    Each pass integrates alpha from 1 to 0. FPS uses the same random starting
    point on the updated geometry; supplied seeds apply only to the first pass.
    """
    if type(num_passes) is not int or num_passes < 1:
        raise ValueError("num_passes must be a positive integer")
    normalized, center, scale = normalize_points(noisy)
    model.eval()
    for pass_index in range(num_passes):
        normalized = patch_denoise(
            model,
            normalized,
            patch_size=cfg["patch_size"],
            seed_k=cfg["seed_k"],
            num_steps=cfg["num_steps"],
            num_chunks=cfg["inference_chunks"],
            rng=make_rng(cfg["inference_seed"], *identity),
            patch_seed_idx=patch_seed_idx if pass_index == 0 else None,
            patch_agg_beta=cfg["patch_agg_beta"],
            alpha_clean_power=cfg["alpha_clean_power"],
            alpha_noisy_power=cfg["alpha_noisy_power"],
        )
        yield (normalized * scale + center).astype(np.float32)


@jt.no_grad()
def denoise_cloud(model, noisy, cfg, identity, *, num_passes, patch_seed_idx=None):
    """Return the final pass in source coordinates."""
    for pred in denoise_passes(model, noisy, cfg, identity,
                              num_passes=num_passes, patch_seed_idx=patch_seed_idx):
        pass
    return pred
