"""Tensor operators for nearest-point and point/triangle squared distances."""

import jittor as jt
from jittor import Function

from ._sources import load_cuda

_CUDA_NEAREST_FORWARD = load_cuda("nearest_distance_forward.cu")
_CUDA_NEAREST_BACKWARD = load_cuda("nearest_distance_backward.cu")
_CUDA_MESH_HEADER = load_cuda("point_triangle.cuh")
_CUDA_MESH_FORWARD = load_cuda("point_mesh_forward.cu")


class _NearestDistanceFunction(Function):
    """Bidirectional nearest squared distances with an exact coordinate gradient."""

    def execute(self, first, second):
        batch, first_count, _ = first.shape
        second_count = second.shape[1]
        first_distance, first_index, second_distance, second_index = jt.code(
            [
                (batch, first_count),
                (batch, first_count),
                (batch, second_count),
                (batch, second_count),
            ],
            [first.dtype, "int32", second.dtype, "int32"],
            [first, second],
            cuda_header="#include <math.h>",
            cuda_src=_CUDA_NEAREST_FORWARD,
        )
        first_index.stop_grad()
        second_index.stop_grad()
        self.saved_tensors = first, second, first_index, second_index
        return first_distance, first_index, second_distance, second_index

    def grad(
        self,
        grad_first_distance,
        grad_first_index,
        grad_second_distance,
        grad_second_index,
    ):
        first, second, first_index, second_index = self.saved_tensors
        if grad_first_distance is None:
            grad_first_distance = jt.zeros(first_index.shape, dtype=first.dtype)
        if grad_second_distance is None:
            grad_second_distance = jt.zeros(second_index.shape, dtype=second.dtype)
        grad_first, grad_second = jt.code(
            [first.shape, second.shape],
            [first.dtype, second.dtype],
            [
                first,
                second,
                first_index,
                second_index,
                grad_first_distance,
                grad_second_distance,
            ],
            cuda_src=_CUDA_NEAREST_BACKWARD,
        )
        return grad_first, grad_second


def nearest_squared_distances(first, second):
    """Return distances and indices from each 3D point set to the other.

    Inputs are float32 (B, M, 3) and (B, N, 3). Returns
    (first_distance, second_index, second_distance, first_index), shaped
    (B, M), (B, M), (B, N), (B, N). Ties select the lowest reference index.
    Coordinate gradients are supported; indices are not differentiable.
    CUDA uses shared-memory tiles; the CPU reference materializes pairwise
    distances and is intended for small sets. Full-cloud evaluation uses KD-trees.
    """
    if len(first.shape) != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected first shaped (B, M, 3), got {first.shape}")
    if len(second.shape) != 3 or second.shape[2] != 3:
        raise ValueError(f"Expected second shaped (B, N, 3), got {second.shape}")
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"Batch sizes differ: {first.shape[0]} and {second.shape[0]}")
    if first.shape[0] < 1 or first.shape[1] < 1 or second.shape[1] < 1:
        raise ValueError("Point sets must be nonempty")
    if first.dtype != second.dtype:
        raise ValueError(f"Dtypes differ: {first.dtype} and {second.dtype}")
    if str(first.dtype) != "float32":
        raise ValueError("Differentiable nearest-point distances require float32")
    if jt.flags.use_cuda:
        return _NearestDistanceFunction()(first, second)

    distance = ((first.unsqueeze(2) - second.unsqueeze(1)) ** 2).sum(dim=-1)
    second_index, first_distance = jt.argmin(distance, dim=2)
    first_index, second_distance = jt.argmin(distance, dim=1)
    return first_distance, second_index, second_distance, first_index


def chamfer_distance(first, second):
    """Sum bidirectional mean squared distances, then average over the batch.

    Inputs follow nearest_squared_distances. This scalar tensor supports
    coordinate gradients. Coordinates are used as supplied, without normalization.
    """
    first_distance, _, second_distance, _ = nearest_squared_distances(first, second)
    return (first_distance.mean(dim=1) + second_distance.mean(dim=1)).mean()


@jt.no_grad()
def point_mesh_squared_distances(points, triangles):
    """Return exact-search point-to-face and face-to-point squared distances.

    Inputs are FP64 CUDA tensors shaped (N, 3) and (F, 3, 3); outputs are
    (N,) and (F,). Each face is a closed triangle, including its interior,
    edges and vertices. Degenerate faces use their segments or points.
    No area cutoff, additive epsilon, normalization, or spatial index is used.
    This evaluation operator has no backward. FP64 is not exact arithmetic.
    """
    if not jt.flags.use_cuda:
        raise RuntimeError("Point/mesh distances require CUDA")
    if len(points.shape) != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("points must have shape (N, 3), N > 0")
    if (
        len(triangles.shape) != 3
        or tuple(triangles.shape[1:]) != (3, 3)
        or triangles.shape[0] < 1
    ):
        raise ValueError("triangles must have shape (F, 3, 3), F > 0")
    if str(points.dtype) != "float64" or str(triangles.dtype) != "float64":
        raise ValueError("Point/mesh distances require float64 coordinates")
    distances = jt.code(
        [(points.shape[0],), (triangles.shape[0],)],
        ["float64", "float64"],
        [points, triangles],
        cuda_header=_CUDA_MESH_HEADER,
        cuda_src=_CUDA_MESH_FORWARD,
    )
    return tuple(distances)
