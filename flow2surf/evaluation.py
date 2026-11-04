"""Full-cloud CD and P2M metrics for fixed benchmark inputs."""

import json
import logging
from pathlib import Path
from time import perf_counter

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from .dataset import (
    benchmark_records,
    load_mesh,
    load_points,
    normalize_points,
)
from .inference import denoise_cloud
from .ops import point_mesh_squared_distances
from .runtime import INFERENCE_FIELDS, git_revision

log = logging.getLogger(__name__)


class _ChamferReference:
    """Reuse exact nearest-neighbor search on a fixed CPU reference cloud."""

    def __init__(self, target):
        self.tree = cKDTree(target)

    def __call__(self, pred):
        pred_tree = cKDTree(pred)
        pred_to_target = self.tree.query(pred_tree.data, workers=4)[0]
        target_to_pred = pred_tree.query(self.tree.data, workers=4)[0]
        return float((pred_to_target**2).mean() + (target_to_pred**2).mean())


def chamfer_distance(pred, target):
    """Sum of two mean squared nearest-neighbor distances."""
    return _ChamferReference(target)(pred)


@jt.no_grad()
def point_to_mesh(points, vertices, faces):
    """Mesh-normalized sum of mean squared point/face and face/point distances.

    The reference bounding-box center and maximum radius define the coordinate
    frame. FP64 CUDA evaluates closed triangles without an area cutoff or
    additive epsilon. The reverse term weights every face equally.
    """
    if not jt.flags.use_cuda:
        raise RuntimeError("P2M evaluation requires CUDA")
    vertices, center, scale = normalize_points(vertices)
    points = jt.array(
        (np.asarray(points, dtype=np.float32) - center) / scale, dtype="float64"
    )
    triangles = jt.array(vertices[faces], dtype="float64")
    point_distance, face_distance = point_mesh_squared_distances(points, triangles)
    return float(point_distance.mean() + face_distance.mean())


def summarize_conditions(rows):
    """Keep every density/noise condition separate; average equally by shape."""
    if not rows:
        raise ValueError("No evaluation samples")
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    summary = {}
    for condition, samples in groups.items():
        summary[condition] = {"num_shapes": len(samples)}
        for metric in ("cd", "p2m", "predict_seconds"):
            summary[condition][metric] = float(
                np.mean([sample[metric] for sample in samples])
            )
    return summary


@jt.no_grad()
def evaluate_benchmark(model, spec, cfg, meshes=None, *, num_passes, noise=None):
    """Evaluate one inference setup, averaging full-cloud errors by shape."""
    model.eval()
    records = list(benchmark_records(spec, meshes, noise=noise))
    log.info("Evaluating %d shapes | CD/P2M x1e4 | prediction time excludes I/O and metrics",
             len(records))
    rows = []
    for index, record in enumerate(records, 1):
        noisy = load_points(record["noisy_path"])
        clean, center, scale = normalize_points(load_points(record["cloud_path"]))
        vertices, faces = load_mesh(record["mesh_path"])
        identity = (spec.name, record["condition"], record["shape_id"])
        jt.sync_all()
        start = perf_counter()
        pred = denoise_cloud(
            model, noisy, cfg, identity, num_passes=num_passes
        )
        jt.sync_all()
        predict_seconds = perf_counter() - start
        row = {"shape_id": record["shape_id"], "condition": record["condition"],
               "cd": chamfer_distance((pred - center) / scale, clean),
               "p2m": point_to_mesh(pred, vertices, faces),
               "predict_seconds": predict_seconds}
        rows.append(row)
        log.info("[%d/%d] %s/%s | CD %.4f P2M %.4f | predict %.2fs",
                 index, len(records), *identity[1:], row["cd"] * 1e4,
                 row["p2m"] * 1e4, predict_seconds)
    return rows


def write_evaluation(path, spec, dataset_path, model_path, cfg, rows, *, num_passes, noise=None):
    """Persist effective settings, source paths and results."""
    result = {
        "dataset": spec.name,
        "dataset_path": str(Path(dataset_path).resolve()),
        "model": str(Path(model_path).resolve()),
        "git_revision": git_revision(),
        "model_config": {key: cfg[key] for key in (
            "feat_dim", "encoder_dims", "k_neighbors", "num_blocks",
            "alpha_dim", "ffn_ratio", "decoder_dims",
        )},
        "inference": {**{key: cfg[key] for key in INFERENCE_FIELDS},
                      "num_passes": num_passes, "patch_size": cfg["patch_size"]},
        "noise_percent": noise,
        "conditions": summarize_conditions(rows),
        "samples": rows,
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    return result
