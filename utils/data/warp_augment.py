"""Smooth global position warping for training-time augmentation.

Each epoch samples a low-resolution displacement field on the normalised
coordinate domain (typically ``[-0.5, 0.5]^2``). Displacements are
bilinearly interpolated, magnitude-clamped, and added to cell positions.
The field is shared across all cells and slices for that epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class SmoothWarpField2D:
    """Bilinear displacement field on a regular 2-D grid."""

    x_coords: np.ndarray  # [Gx]
    y_coords: np.ndarray  # [Gy]
    disp_x: np.ndarray    # [Gy, Gx]
    disp_y: np.ndarray    # [Gy, Gx]
    max_displacement: float

    def displacement_at(self, xy: np.ndarray) -> np.ndarray:
        """Return displacement vectors ``[N, 2]`` for query points ``[N, 2]``."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected xy shape [N, 2], got {xy.shape}.")
        dx = _bilinear_sample(self.x_coords, self.y_coords, self.disp_x, xy)
        dy = _bilinear_sample(self.x_coords, self.y_coords, self.disp_y, xy)
        disp = np.stack([dx, dy], axis=-1)
        norms = np.linalg.norm(disp, axis=-1, keepdims=True)
        scale = np.minimum(
            1.0,
            self.max_displacement / np.maximum(norms, 1e-12),
        )
        return disp * scale

    def warp(self, xy: ArrayLike) -> np.ndarray:
        """Apply the field to ``xy`` and return warped coordinates as numpy."""
        if isinstance(xy, torch.Tensor):
            xy_np = xy.detach().cpu().numpy()
        else:
            xy_np = np.asarray(xy, dtype=np.float64)
        return xy_np + self.displacement_at(xy_np)


def sample_smooth_warp_field(
    seed: int,
    *,
    grid_size: int = 8,
    domain: Tuple[float, float] = (-0.5, 0.5),
    max_displacement: float = 0.01,
    max_angle_span: float = math.pi / 2,
) -> SmoothWarpField2D:
    """Sample a smooth displacement field for one training epoch.

    Grid angles lie in a window of width ``max_angle_span`` so the warp
    direction varies by at most ``max_angle_span`` over the domain. Grid
    magnitudes are uniform in ``[0, max_displacement]``; interpolation
    is followed by a per-point magnitude clamp.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")
    if max_displacement <= 0.0:
        raise ValueError("max_displacement must be > 0.")
    if max_angle_span <= 0.0:
        raise ValueError("max_angle_span must be > 0.")

    rng = np.random.default_rng(int(seed))
    lo, hi = float(domain[0]), float(domain[1])
    x_coords = np.linspace(lo, hi, int(grid_size), dtype=np.float64)
    y_coords = np.linspace(lo, hi, int(grid_size), dtype=np.float64)

    theta0 = float(rng.uniform(0.0, 2.0 * math.pi))
    half = float(max_angle_span) * 0.5
    thetas = rng.uniform(theta0 - half, theta0 + half, size=(grid_size, grid_size))
    mags = rng.uniform(0.0, float(max_displacement), size=(grid_size, grid_size))
    disp_x = mags * np.cos(thetas)
    disp_y = mags * np.sin(thetas)

    return SmoothWarpField2D(
        x_coords=x_coords,
        y_coords=y_coords,
        disp_x=disp_x,
        disp_y=disp_y,
        max_displacement=float(max_displacement),
    )


def apply_warp_field(xy: ArrayLike, field: SmoothWarpField2D) -> torch.Tensor:
    """Warp ``xy`` ``[N, 2]`` and return a float32 torch tensor."""
    warped = field.warp(xy)
    return torch.from_numpy(warped.astype(np.float32))


def warp_displacement_grid(
    field: SmoothWarpField2D,
    resolution: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the field on a dense grid for quiver / magnitude plots.

    Returns ``xs, ys, dx, dy`` with ``xs, ys`` 1-D axes and ``dx, dy`` shaped
    ``[resolution, resolution]``.
    """
    lo_x, hi_x = float(field.x_coords[0]), float(field.x_coords[-1])
    lo_y, hi_y = float(field.y_coords[0]), float(field.y_coords[-1])
    xs = np.linspace(lo_x, hi_x, int(resolution), dtype=np.float64)
    ys = np.linspace(lo_y, hi_y, int(resolution), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    queries = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    disp = field.displacement_at(queries)
    dx = disp[:, 0].reshape(grid_y.shape)
    dy = disp[:, 1].reshape(grid_y.shape)
    return xs, ys, dx, dy


def _bilinear_sample(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Bilinear sample ``values[Gy, Gx]`` at ``queries[N, 2]``."""
    x0, x1 = float(x_coords[0]), float(x_coords[-1])
    y0, y1 = float(y_coords[0]), float(y_coords[-1])
    gx = len(x_coords) - 1
    gy = len(y_coords) - 1

    qx = queries[:, 0]
    qy = queries[:, 1]
    u = (qx - x0) / max(x1 - x0, 1e-12) * gx
    v = (qy - y0) / max(y1 - y0, 1e-12) * gy
    u = np.clip(u, 0.0, float(gx))
    v = np.clip(v, 0.0, float(gy))

    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    i1 = np.minimum(i0 + 1, gx)
    j1 = np.minimum(j0 + 1, gy)
    tu = (u - i0).astype(np.float64)
    tv = (v - j0).astype(np.float64)

    v00 = values[j0, i0]
    v10 = values[j0, i1]
    v01 = values[j1, i0]
    v11 = values[j1, i1]
    return (
        (1.0 - tu) * (1.0 - tv) * v00
        + tu * (1.0 - tv) * v10
        + (1.0 - tu) * tv * v01
        + tu * tv * v11
    )
