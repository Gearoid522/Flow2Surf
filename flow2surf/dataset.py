"""Clean clouds, stochastic flow patches, and fixed benchmark inputs."""

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from scipy.spatial.distance import cdist
import yaml

_NOISE_TYPES = {"gaussian", "laplace"}


@dataclass(frozen=True)
class DatasetSpec:
    """Paths declared by a dataset YAML, relative to its root."""

    name: str
    root: Path
    train: list
    test: dict


def load_dataset_spec(path):
    """Read explicit clean directories and named benchmark conditions."""
    path = Path(path).resolve()
    with path.open() as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict) or set(values) != {
        "name",
        "root",
        "train",
        "test",
    }:
        raise ValueError("Dataset requires name, root, train and test")
    name = values["name"]
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
    ):
        raise ValueError("Dataset name must be a directory-safe label")
    data_root = (path.parent / values["root"]).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not isinstance(values["train"], list) or any(
        not isinstance(directory, str) or not directory for directory in values["train"]
    ):
        raise ValueError("train must list clean-cloud directories")
    if not isinstance(values["test"], dict):
        raise ValueError(
            "test must map condition names to input/clean/mesh directories"
        )
    for name, condition in values["test"].items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid condition name: {name}")
        if not isinstance(condition, dict) or set(condition) != {
            "input",
            "clean",
            "mesh",
        }:
            raise ValueError(f"Condition {name} requires input, clean and mesh")
    return DatasetSpec(
        values["name"], data_root, values["train"], values["test"]
    )


def _xyz_files(directory):
    paths = sorted(directory.glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"No XYZ clouds: {directory}")
    return paths


def load_points(path):
    """Read one XYZ cloud and reject malformed/non-finite input."""
    points = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Expected a non-empty (N, 3) XYZ cloud: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite coordinates: {path}")
    return points


def _test_conditions(spec, noise):
    """Select declared test conditions by coordinate-noise percentage."""
    conditions = spec.test
    if noise is not None:
        suffix = f"_{float(noise)}pct"
        conditions = {name: value for name, value in conditions.items() if name.endswith(suffix)}
    if not conditions:
        raise ValueError(f"No test conditions declared for noise={noise}")
    return conditions.items()


def benchmark_records(spec, limit=None, *, noise=None):
    """Pair fixed noisy inputs with references, limiting shapes per condition."""
    if limit is not None and limit < 1:
        raise ValueError("Shape limit must be positive")
    for name, condition in _test_conditions(spec, noise):
        inputs = _xyz_files(spec.root / condition["input"])
        for path in inputs[:limit]:
            clean = spec.root / condition["clean"] / path.name
            mesh = spec.root / condition["mesh"] / f"{path.stem}.off"
            if not clean.is_file():
                raise FileNotFoundError(clean)
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            yield {
                "condition": name,
                "shape_id": path.stem,
                "noisy_path": path,
                "cloud_path": clean,
                "mesh_path": mesh,
            }


def test_cloud_paths(spec, *, noise=None):
    """Yield declared noisy inputs without loading evaluation references."""
    for condition, directories in _test_conditions(spec, noise):
        for path in _xyz_files(spec.root / directories["input"]):
            yield condition, path


def load_mesh(path):
    """Load mesh vertices and triangular faces as NumPy arrays."""
    import trimesh

    mesh = trimesh.load(
        path,
        process=False,
        force="mesh",
        skip_materials=True,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Mesh has invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Mesh has no triangular faces: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has non-finite vertices: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh has out-of-range face indices: {path}")
    return vertices, faces


def normalize_points(points):
    """Return (points - center) / radius, center, and maximum Euclidean radius."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(
            f"Expected non-empty points with shape (N, 3), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")

    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    centered = points - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Points must have positive finite radius")
    return centered / radius, center, radius


def sample_noise_std(cfg, rng):
    """Sample a coordinate standard deviation under the configured measure."""
    std_min, std_max = cfg["noise_std_range"]
    sampling = cfg["noise_std_sampling"]
    if sampling == "linear":
        return float(rng.uniform(std_min, std_max))
    if sampling == "log":
        return float(np.exp(rng.uniform(np.log(std_min), np.log(std_max))))
    raise ValueError(f"Unknown noise_std_sampling: {sampling}")


def _sample_unit_directions(count, rng):
    """Sample `count` independent directions uniformly on the unit sphere."""
    directions = rng.normal(size=(count, 3))
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    zero = lengths[:, 0] == 0
    while zero.any():
        directions[zero] = rng.normal(size=(int(zero.sum()), 3))
        lengths[zero] = np.linalg.norm(directions[zero], axis=1, keepdims=True)
        zero = lengths[:, 0] == 0
    return directions / lengths


def add_noise(points, noise_std, noise_type, rng):
    """Add pointwise spherical noise with covariance `noise_std^2 I`."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    if noise_type not in _NOISE_TYPES:
        valid = ", ".join(sorted(_NOISE_TYPES))
        raise ValueError(f"Unknown noise_type: {noise_type}. Valid values: {valid}")
    noise_std = float(noise_std)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError(f"noise_std must be finite and non-negative, got {noise_std}")
    if noise_type == "gaussian":
        noise = rng.normal(0, noise_std, points.shape).astype(np.float32)
    elif noise_type == "laplace":
        direction = _sample_unit_directions(len(points), rng)
        radius = rng.gamma(3.0, noise_std / 2.0, size=(len(points), 1))
        noise = (radius * direction).astype(np.float32)
    return points + noise


def _nearest_indices(points, queries, k):
    """Return k nearest point indices in ascending squared-distance order."""
    points = np.asarray(points)
    queries = np.asarray(queries)
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1:] != queries.shape[1:]:
        raise ValueError(
            f"Expected points and queries with matching (N, D) shapes, got "
            f"{points.shape} and {queries.shape}"
        )
    if not 1 <= k <= len(points):
        raise ValueError(f"k must be in [1, {len(points)}], got {k}")
    sq_dist = cdist(queries, points, metric="sqeuclidean")
    indices = np.argpartition(sq_dist, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(sq_dist, indices, axis=1)
    order = np.argsort(selected, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)


def _extract_patches(clean, noisy, patch_size, seed_idx):
    """Return clean/noisy patches and clean seeds in the noisy-seed frame.

    Coordinates are relative to the noisy seed. Training moves
    the origin to its interpolated seed.
    """
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    if clean.shape != noisy.shape or clean.ndim != 2 or clean.shape[1] != 3:
        raise ValueError(
            f"Expected matching clean/noisy shapes (N, 3), got "
            f"{clean.shape} and {noisy.shape}"
        )
    num_points = len(noisy)
    seed_idx = np.asarray(seed_idx, dtype=np.int64)
    if seed_idx.ndim != 1 or not len(seed_idx):
        raise ValueError(
            f"Expected non-empty seed_idx with shape (P,), got {seed_idx.shape}"
        )
    if seed_idx.min() < 0 or seed_idx.max() >= num_points:
        raise ValueError("seed_idx contains an out-of-range point index")
    nn_idx = _nearest_indices(noisy, noisy[seed_idx], patch_size)  # (P, M)
    # Keep the chosen seed when distance ties select an equivalent point.
    missing = ~(nn_idx == seed_idx[:, None]).any(axis=1)
    nn_idx[missing, 0] = seed_idx[missing]
    seed_noisy = noisy[seed_idx, None, :]
    return (
        (noisy[nn_idx] - seed_noisy).astype(np.float32),
        (clean[nn_idx] - seed_noisy).astype(np.float32),
        (clean[seed_idx] - noisy[seed_idx]).astype(np.float32),
    )


class _EpochSampler:
    """Encode epoch and sample position in Jittor's existing worker index queue."""

    def __init__(self, count, start_epoch):
        self.count = count
        self.epoch = start_epoch

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        return iter(range(epoch * self.count, (epoch + 1) * self.count))


class FlowPatchDataset(Dataset):
    """Epoch/item-seeded patches with persistent, prefetching workers."""

    def __init__(self, paths, cfg, start_epoch=1):
        super().__init__(endless=cfg["num_workers"] > 0)
        self.paths = paths
        self.cfg = cfg
        self._iterator = None
        self._order_epoch = None
        self._order = None
        self.clean_clouds = []
        for path in paths:
            clean, _, _ = normalize_points(load_points(path))
            clean.setflags(write=False)
            self.clean_clouds.append(clean)
        count = cfg["patches_per_cloud"]
        if min(len(clean) for clean in self.clean_clouds) < cfg["patch_size"]:
            raise ValueError("patch_size exceeds the smallest clean cloud")
        self.set_attrs(
            total_len=len(paths) * count,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            keep_numpy_array=True,
            drop_last=False,
        )
        self.sampler = _EpochSampler(self.total_len, start_epoch)

    def __iter__(self):
        if not self.num_workers:
            return super().__iter__()
        # Jittor prefetches across epoch boundaries. Reuse one endless iterator
        # so each queued epoch is consumed once, without restarting workers.
        if self._iterator is None:
            self._iterator = self._ordered_batches(super().__iter__())
        return islice(self._iterator, len(self))

    def _ordered_batches(self, batches):
        """Restore sampler order when workers enqueue batches out of order."""
        next_index = self.sampler.epoch * self.total_len
        pending = {}
        while True:
            while next_index not in pending:
                batch = next(batches)
                pending[int(batch["sample_index"][0])] = batch
            batch = pending.pop(next_index)
            next_index += len(batch["noisy"])
            yield batch

    def __getitem__(self, index):
        epoch, position = divmod(int(index), self.total_len)
        if epoch != self._order_epoch:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], epoch]))
            self._order = rng.permutation(self.total_len)
            self._order_epoch = epoch
        item = int(self._order[position])
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], epoch, item]))
        alpha = np.asarray(rng.uniform(), dtype=np.float32)
        clean = self.clean_clouds[item % len(self.paths)]
        noisy = add_noise(
            clean, sample_noise_std(self.cfg, rng), self.cfg["noise_type"], rng
        )
        centers = rng.integers(len(clean), size=1)
        noisy, clean, seed_clean = _extract_patches(
            clean, noisy, self.cfg["patch_size"], centers
        )
        return {
            "noisy": noisy[0],
            "clean": clean[0],
            "seed_clean": seed_clean[0],
            "alpha": alpha,
            "sample_index": int(index),
        }


def validate_data_config(cfg):
    """Validate the explicit sampling fields."""
    for field in ["patches_per_cloud", "patch_size", "batch_size"]:
        if type(cfg[field]) is not int or cfg[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(cfg["num_workers"]) is not int or cfg["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if cfg["noise_type"] not in _NOISE_TYPES:
        raise ValueError(f"Unknown noise_type: {cfg['noise_type']}")
    lo, hi = cfg["noise_std_range"]
    if not np.isfinite([lo, hi]).all() or lo < 0 or hi < lo:
        raise ValueError("Invalid noise_std_range")
    if cfg["noise_std_sampling"] not in {"linear", "log"}:
        raise ValueError("Unknown noise_std_sampling")
    if cfg["noise_std_sampling"] == "log" and lo <= 0:
        raise ValueError("Log noise sampling requires a positive minimum")


def build_dataset(cfg, spec, start_epoch=1):
    """Build patches from the declared training directories."""
    validate_data_config(cfg)
    paths = [
        path.resolve()
        for directory in spec.train
        for path in _xyz_files(spec.root / directory)
    ]
    if not paths:
        raise ValueError("No training clouds declared")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate training clouds")
    return FlowPatchDataset(paths, cfg, start_epoch)
