"""Cahn-Hilliard energy losses for spatial cell-type distributions.

This module is a standalone (testing-side) port of the differentiable
Cahn-Hilliard energy losses used during training. It contains two losses:

* :class:`CahnHilliardEnergyCurveLoss` — for every cell type, builds a
  continuous landscape ``phi(x; r) in [-1, 1]`` from the cell positions at
  each radius of a radius sweep, evaluates the Cahn-Hilliard energy
  ``E(r) = sum [(phi^2-1)^2 + kappa * |grad phi|^2] dx dy`` on a shared grid
  for both the ground-truth and predicted positions, and compares the two
  energy curves vector-wise (per radius) with the bounded, scale-invariant
  form ``1 - exp(-|E_pred - E_gt| / (|E_gt| + eps))``.

* :class:`VoronoiPhasePairEnergyLoss` — for every unordered pair of cell
  types ``(A, B)``, builds the Voronoi phase-separation landscape
  ``phi_AB(x) = tanh((dist_A(x) - dist_B(x)) / w)`` (where ``dist_t`` is the
  distance to the nearest cell of type ``t``), evaluates its Cahn-Hilliard
  energy for both GT and prediction, and compares them with the same
  bounded relative form. Per-cell-type distance fields are computed once
  per side and reused across all pairs.

The two helper functions at the bottom
(:func:`ch_energy_curve_loss_per_sample` and
:func:`voronoi_pair_energy_loss_per_sample`) are what the testing pipeline
calls: they take a single sample's predicted positions, the GT positions
and the per-cell integer class ids, and return the per-type / per-pair
float losses plus the matched energy values so the pipeline can average
over denoising samples and report per-cell-type and per-cell-type-pair
summaries.

The differentiable core (bump functions, shared grid, landscape builder,
CH energy, distance fields, Voronoi field) is kept byte-for-byte close to
the training-side reference so the numerical values are directly
comparable. The only things removed here are the WandB-logging /
visualization hooks (the testing pipeline does not use WandB); the
``forward`` methods still work with ``DataHolder`` objects.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Bump functions
# ─────────────────────────────────────────────────────────────────────────────

BumpFn = Callable[..., torch.Tensor]


def bump_gaussian(r: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaussian bump with scale = ``radius`` and ``bump(0) = 1``."""
    radius = max(float(radius), 1e-12)
    return torch.exp(-(r * r) / (2.0 * radius * radius))


def sigmoid_bump(
    r: torch.Tensor,
    cell_radius: float,
    decay_rate: Optional[float] = None,
    shift: float = 0.0,
) -> torch.Tensor:
    """Sigmoid bump: ``1 / (1 + exp(decay_rate * (r - cell_radius) - shift))``.

    Default ``decay_rate = 4 / cell_radius``; ``shift`` defaults to 0 (the
    notebook's ``128 * radius`` is only meaningful in integer pixel units).
    """
    cell_radius = max(float(cell_radius), 1e-12)
    if decay_rate is None:
        decay_rate = 4.0 / cell_radius
    decay_rate = float(decay_rate)
    return torch.sigmoid(-(decay_rate * (r - cell_radius) - shift))


# ─────────────────────────────────────────────────────────────────────────────
# Shared grid
# ─────────────────────────────────────────────────────────────────────────────


def shared_square_grid(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
    grid_resolution: int,
    margin: float = 0.0,
    square: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Axis-aligned grid covering the union bbox of two point sets.

    When ``square`` is True the bbox is expanded to a square so x and y
    share the same physical scale (matches the reference
    ``build_continuous_landscape_from_points`` convention).
    """
    all_pts = torch.cat([pts_a, pts_b], dim=0).detach()
    lo = all_pts.min(dim=0).values
    hi = all_pts.max(dim=0).values

    if square:
        center = 0.5 * (lo + hi)
        side = (hi - lo).max().clamp(min=1e-6)
        half = 0.5 * side
        lo = center - half
        hi = center + half

    extent = (hi - lo).clamp(min=1e-6)
    lo = lo - margin * extent
    hi = hi + margin * extent

    device = pts_a.device
    dtype = pts_a.dtype
    grid_x = torch.linspace(
        lo[0].item(), hi[0].item(), grid_resolution,
        device=device, dtype=dtype,
    )
    grid_y = torch.linspace(
        lo[1].item(), hi[1].item(), grid_resolution,
        device=device, dtype=dtype,
    )
    denom = max(grid_resolution - 1, 1)
    dx = (hi[0].item() - lo[0].item()) / denom
    dy = (hi[1].item() - lo[1].item()) / denom
    return grid_x, grid_y, dx, dy


# ─────────────────────────────────────────────────────────────────────────────
# Continuous landscape from points (single radius)
# ─────────────────────────────────────────────────────────────────────────────


def build_continuous_landscape_from_points(
    positions: torch.Tensor,                 # [N, 2]
    valid_mask: torch.Tensor,                # [N] bool
    grid_x: torch.Tensor,                    # [nx]
    grid_y: torch.Tensor,                    # [ny]
    *,
    radius: float,
    bump_fn: BumpFn = sigmoid_bump,
    bump_kwargs: Optional[Dict] = None,
    combine: Literal["soft_max", "hard_max"] = "soft_max",
    soft_max_beta: float = 16.0,
    support_factor: float = 10.0,
    chunk: int = 256,
) -> torch.Tensor:
    """Build a continuous 2D scalar field ``phi(x) in [-1, 1]`` on a grid.

    The field is ``-1`` everywhere by default; each point contributes a
    bump so that ``candidate = -1 + 2 * bump_i(r)`` reaches ``+1`` at the
    point. Bumps are combined by pointwise maximum (``hard_max``) or its
    smooth log-sum-exp approximation (``soft_max``, default).
    """
    device = positions.device
    dtype = positions.dtype
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]

    if valid_mask.sum() == 0:
        return torch.full((ny, nx), -1.0, device=device, dtype=dtype)

    pts = positions[valid_mask]                 # [M, 2]

    support = float(support_factor) * float(radius)
    bump_kwargs = dict(bump_kwargs or {})
    chunk = max(1, int(chunk))

    dx_grid = abs(float((grid_x[1] - grid_x[0]).detach().item())) if nx > 1 else 1.0
    dy_grid = abs(float((grid_y[1] - grid_y[0]).detach().item())) if ny > 1 else 1.0
    half_x = min(nx - 1, max(0, math.ceil(support / max(dx_grid, 1e-12)) + 1))
    half_y = min(ny - 1, max(0, math.ceil(support / max(dy_grid, 1e-12)) + 1))
    offset_x = torch.arange(-half_x, half_x + 1, device=device)
    offset_y = torch.arange(-half_y, half_y + 1, device=device)

    def local_patch_values(pts_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centers_x = torch.round(
            (pts_chunk[:, 0].detach() - grid_x[0]) / max(dx_grid, 1e-12)
        ).long().clamp(0, nx - 1)
        centers_y = torch.round(
            (pts_chunk[:, 1].detach() - grid_y[0]) / max(dy_grid, 1e-12)
        ).long().clamp(0, ny - 1)

        x_idx = centers_x[:, None, None] + offset_x[None, None, :]
        y_idx = centers_y[:, None, None] + offset_y[None, :, None]
        valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
        x_idx = x_idx.clamp(0, nx - 1)
        y_idx = y_idx.clamp(0, ny - 1)

        gx = grid_x[x_idx]
        gy = grid_y[y_idx]
        dx = gx - pts_chunk[:, 0, None, None]
        dy = gy - pts_chunk[:, 1, None, None]
        r = torch.sqrt(dx * dx + dy * dy)
        support_mask = valid & (r <= support)
        bump = bump_fn(r, radius, **bump_kwargs)
        flat_idx = y_idx * nx + x_idx
        return flat_idx, bump * support_mask.to(dtype)

    if combine == "soft_max":
        beta = float(soft_max_beta)
        acc = torch.ones(ny * nx, device=device, dtype=dtype)
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_values(pts[start:start + chunk])
            values = torch.exp(beta * bump) * (bump > 0).to(dtype)
            acc.scatter_add_(0, flat_idx.reshape(-1), values.reshape(-1))
        soft_max_bump = torch.log(acc.reshape(ny, nx)) / beta
        phi = -1.0 + 2.0 * soft_max_bump
    elif combine == "hard_max":
        best = torch.zeros(ny * nx, device=device, dtype=dtype)
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_values(pts[start:start + chunk])
            best.scatter_reduce_(
                0,
                flat_idx.reshape(-1),
                bump.reshape(-1),
                reduce="amax",
                include_self=True,
            )
        best = best.reshape(ny, nx)
        phi = -1.0 + 2.0 * best
    else:
        raise ValueError(f"unknown combine='{combine}'")

    phi = phi.clamp(-1.0, 1.0)
    return phi


# ─────────────────────────────────────────────────────────────────────────────
# Continuous landscape from points (multi-radius, single pass)
# ─────────────────────────────────────────────────────────────────────────────


def build_continuous_landscapes_multi_radius(
    positions: torch.Tensor,                 # [N, 2]
    valid_mask: torch.Tensor,                # [N] bool
    grid_x: torch.Tensor,                    # [nx]
    grid_y: torch.Tensor,                    # [ny]
    radii: Sequence[float],
    *,
    bump_fn: BumpFn = sigmoid_bump,
    bump_kwargs: Optional[Dict] = None,
    combine: Literal["soft_max", "hard_max"] = "soft_max",
    soft_max_beta: float = 16.0,
    support_factor: float = 10.0,
    chunk: int = 256,
) -> torch.Tensor:
    """Build CH landscapes for all ``radii`` in one pass over cell chunks.

    Returns ``phi`` of shape ``[R, ny, nx]``.
    """
    device = positions.device
    dtype = positions.dtype
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    radii_f = [float(r) for r in radii]
    n_radii = len(radii_f)
    if n_radii < 1:
        raise ValueError("Need at least one radius.")

    if valid_mask.sum() == 0:
        return torch.full((n_radii, ny, nx), -1.0, device=device, dtype=dtype)

    pts = positions[valid_mask]
    bump_kwargs = dict(bump_kwargs or {})
    chunk = max(1, int(chunk))

    max_radius = max(radii_f)
    support_max = float(support_factor) * max_radius
    dx_grid = abs(float((grid_x[1] - grid_x[0]).detach().item())) if nx > 1 else 1.0
    dy_grid = abs(float((grid_y[1] - grid_y[0]).detach().item())) if ny > 1 else 1.0
    half_x = min(nx - 1, max(0, math.ceil(support_max / max(dx_grid, 1e-12)) + 1))
    half_y = min(ny - 1, max(0, math.ceil(support_max / max(dy_grid, 1e-12)) + 1))
    offset_x = torch.arange(-half_x, half_x + 1, device=device)
    offset_y = torch.arange(-half_y, half_y + 1, device=device)

    radii_t = torch.tensor(radii_f, device=device, dtype=dtype)
    supports = float(support_factor) * radii_t
    grid_size = ny * nx

    def local_patch_bumps(pts_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centers_x = torch.round(
            (pts_chunk[:, 0].detach() - grid_x[0]) / max(dx_grid, 1e-12)
        ).long().clamp(0, nx - 1)
        centers_y = torch.round(
            (pts_chunk[:, 1].detach() - grid_y[0]) / max(dy_grid, 1e-12)
        ).long().clamp(0, ny - 1)

        x_idx = centers_x[:, None, None] + offset_x[None, None, :]
        y_idx = centers_y[:, None, None] + offset_y[None, :, None]
        valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
        x_idx = x_idx.clamp(0, nx - 1)
        y_idx = y_idx.clamp(0, ny - 1)

        gx = grid_x[x_idx]
        gy = grid_y[y_idx]
        dx = gx - pts_chunk[:, 0, None, None]
        dy = gy - pts_chunk[:, 1, None, None]
        r = torch.sqrt(dx * dx + dy * dy)

        r_exp = r.unsqueeze(1)
        rad = radii_t.view(1, n_radii, 1, 1)
        decay = 4.0 / rad
        shift = 128.0 * rad
        if bump_fn is sigmoid_bump and not bump_kwargs:
            bump = torch.sigmoid(-(decay * (r_exp - rad) - shift))
        else:
            bumps = []
            for rad_k in radii_f:
                bump_k = bump_fn(
                    r,
                    rad_k,
                    decay_rate=4.0 / rad_k,
                    shift=128.0 * rad_k,
                    **bump_kwargs,
                )
                bumps.append(bump_k)
            bump = torch.stack(bumps, dim=1)
        support_mask = valid.unsqueeze(1) & (r_exp <= supports.view(1, n_radii, 1, 1))
        bump = bump * support_mask.to(dtype)

        flat_idx = y_idx * nx + x_idx
        return flat_idx, bump

    if combine == "soft_max":
        beta = float(soft_max_beta)
        acc = torch.ones(n_radii, grid_size, device=device, dtype=dtype)
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_bumps(pts[start:start + chunk])
            values = torch.exp(beta * bump) * (bump > 0).to(dtype)
            idx = flat_idx.unsqueeze(1).expand(-1, n_radii, -1, -1)
            idx = idx.permute(1, 0, 2, 3).reshape(n_radii, -1)
            values = values.permute(1, 0, 2, 3).reshape(n_radii, -1)
            acc.scatter_add_(1, idx, values)
        soft_max_bump = torch.log(acc.reshape(n_radii, ny, nx)) / beta
        phi = -1.0 + 2.0 * soft_max_bump
    elif combine == "hard_max":
        best = torch.zeros(n_radii, grid_size, device=device, dtype=dtype)
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_bumps(pts[start:start + chunk])
            idx = flat_idx.unsqueeze(1).expand(-1, n_radii, -1, -1)
            idx = idx.permute(1, 0, 2, 3).reshape(n_radii, -1)
            bump_flat = bump.permute(1, 0, 2, 3).reshape(n_radii, -1)
            best.scatter_reduce_(
                1,
                idx,
                bump_flat,
                reduce="amax",
                include_self=True,
            )
        phi = -1.0 + 2.0 * best.reshape(n_radii, ny, nx)
    else:
        raise ValueError(f"unknown combine='{combine}'")

    return phi.clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cahn-Hilliard energy
# ─────────────────────────────────────────────────────────────────────────────


def _grad_xy_edge_order_1(
    phi: torch.Tensor, dx: float, dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradient with ``edge_order=1`` boundaries."""
    inner_x = (phi[..., :, 2:] - phi[..., :, :-2]) / (2.0 * dx)
    left_x = ((phi[..., :, 1] - phi[..., :, 0]) / dx).unsqueeze(-1)
    right_x = ((phi[..., :, -1] - phi[..., :, -2]) / dx).unsqueeze(-1)
    dphi_dx = torch.cat([left_x, inner_x, right_x], dim=-1)

    inner_y = (phi[..., 2:, :] - phi[..., :-2, :]) / (2.0 * dy)
    top_y = ((phi[..., 1, :] - phi[..., 0, :]) / dy).unsqueeze(-2)
    bot_y = ((phi[..., -1, :] - phi[..., -2, :]) / dy).unsqueeze(-2)
    dphi_dy = torch.cat([top_y, inner_y, bot_y], dim=-2)

    return dphi_dy, dphi_dx


def cahn_hilliard_energy_density(
    phi: torch.Tensor,                    # [..., ny, nx] with values in [-1, 1]
    dx: float,
    dy: float,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Discrete energy density ``e = (phi^2 - 1)^2 + kappa * |grad phi|^2``."""
    if kappa < 0:
        raise ValueError("kappa must be >= 0")

    well = (phi * phi - 1.0).pow(2)

    dphi_dy, dphi_dx = _grad_xy_edge_order_1(phi, dx, dy)
    grad_sq = dphi_dx * dphi_dx + dphi_dy * dphi_dy

    return well + kappa * grad_sq


def cahn_hilliard_energy(
    phi: torch.Tensor,
    dx: float,
    dy: float,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Integrated Cahn-Hilliard energy (Riemann sum).

    For ``phi`` of shape ``[..., ny, nx]`` returns a tensor of shape
    ``[...]``.
    """
    density = cahn_hilliard_energy_density(phi, dx, dy, kappa=kappa)
    return density.sum(dim=(-2, -1)) * dx * dy


def compute_ch_energy_curve_from_points(
    positions: torch.Tensor,                 # [N, >=2]
    valid_mask: torch.Tensor,                # [N] bool
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    dx: float,
    dy: float,
    radii: Sequence[float],
    *,
    kappa: float = 1.0,
    bump_fn: BumpFn = sigmoid_bump,
    bump_kwargs: Optional[Dict] = None,
    combine: Literal["soft_max", "hard_max"] = "soft_max",
    soft_max_beta: float = 16.0,
    support_factor: float = 10.0,
    chunk: int = 256,
) -> torch.Tensor:
    """Return CH energies ``E(r)`` as a ``[len(radii)]`` tensor."""
    positions_xy = positions[..., :2]
    phi = build_continuous_landscapes_multi_radius(
        positions_xy,
        valid_mask,
        grid_x,
        grid_y,
        radii,
        bump_fn=bump_fn,
        bump_kwargs=bump_kwargs,
        combine=combine,
        soft_max_beta=soft_max_beta,
        support_factor=support_factor,
        chunk=chunk,
    )
    return cahn_hilliard_energy(phi, dx, dy, kappa)


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi phase-separation landscape (per cell-type pair)
# ─────────────────────────────────────────────────────────────────────────────


def soft_nearest_distance(
    grid_xy: torch.Tensor,    # [G, 2]
    cell_xy: torch.Tensor,    # [N, 2]
    soft_beta: Optional[float] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Distance from each grid point to the nearest cell (hard or soft min)."""
    n_cells = cell_xy.shape[0]
    if n_cells == 0:
        return torch.full(
            (grid_xy.shape[0],),
            float("inf"),
            device=grid_xy.device,
            dtype=grid_xy.dtype,
        )

    G = grid_xy.shape[0]
    chunk = max(1, int(chunk))
    out_chunks = []
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        d = torch.cdist(grid_xy[s:e], cell_xy, p=2)
        if soft_beta is None:
            out_chunks.append(d.min(dim=-1).values)
        else:
            beta = float(soft_beta)
            out_chunks.append(-torch.logsumexp(-beta * d, dim=-1) / beta)
    return torch.cat(out_chunks, dim=0)


def compute_distance_field_per_type(
    positions_xy: torch.Tensor,                  # [N, 2]
    type_masks: Dict[int, torch.Tensor],         # type_id -> [N] bool
    grid_x: torch.Tensor,                        # [nx]
    grid_y: torch.Tensor,                        # [ny]
    soft_beta: Optional[float] = None,
    chunk: int = 4096,
) -> Dict[int, torch.Tensor]:
    """Per-cell-type "distance to nearest cell of that type" fields."""
    ny, nx = grid_y.shape[0], grid_x.shape[0]
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [G, 2]

    out: Dict[int, torch.Tensor] = {}
    for type_id, mask in type_masks.items():
        cells = positions_xy[mask]
        if cells.shape[0] == 0:
            continue
        d_flat = soft_nearest_distance(
            grid_xy, cells, soft_beta=soft_beta, chunk=chunk
        )
        out[type_id] = d_flat.reshape(ny, nx)
    return out


def voronoi_phase_field(
    dist_neg: torch.Tensor,        # [..., ny, nx]
    dist_pos: torch.Tensor,        # [..., ny, nx]
    transition_width: float,
) -> torch.Tensor:
    """Smooth two-phase Voronoi landscape from precomputed distance fields.

    ``phi(x) = tanh((dist_neg(x) - dist_pos(x)) / transition_width)``.
    """
    w = float(transition_width)
    if w <= 0:
        raise ValueError("transition_width must be > 0")
    return torch.tanh((dist_neg - dist_pos) / w)


def compute_distance_fields_per_type_fused(
    grid_xy: torch.Tensor,            # [G, 2]
    grid_norm_sq: torch.Tensor,       # [G] precomputed ||g||^2
    cells_xy: torch.Tensor,           # [N, 2] cells of *all* valid types stacked
    cell_type_idx: torch.Tensor,      # [N] in [0, T)
    num_types: int,
    soft_beta: Optional[float] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Fused per-type (soft-)distance fields in a single pass over cells.

    Returns ``[T, G]``. See the training-side reference for the matmul
    expansion of the squared distance and the per-type scatter reduction.
    """
    G = grid_xy.shape[0]
    N = cells_xy.shape[0]
    device = grid_xy.device
    dtype = grid_xy.dtype

    if N == 0 or num_types == 0:
        return torch.full((max(num_types, 1), G), float("inf"),
                          device=device, dtype=dtype)

    cell_norm_sq = (cells_xy * cells_xy).sum(-1)  # [N]
    out = torch.empty((num_types, G), device=device, dtype=dtype)

    chunk = max(1, int(chunk))
    type_idx_row = cell_type_idx.unsqueeze(0)  # [1, N]

    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        Gc = e - s

        g = grid_xy[s:e]                         # [Gc, 2]
        gn2 = grid_norm_sq[s:e].unsqueeze(-1)    # [Gc, 1]

        dot = g @ cells_xy.t()                   # [Gc, N]
        d2 = gn2 + cell_norm_sq.unsqueeze(0) - 2.0 * dot
        eps_d2 = 1e-12
        d2 = d2.clamp_min(eps_d2)

        type_idx_chunk = type_idx_row.expand(Gc, N)

        if soft_beta is None:
            min_d2 = torch.full(
                (Gc, num_types), float("inf"),
                device=device, dtype=dtype,
            )
            min_d2.scatter_reduce_(
                dim=1, index=type_idx_chunk, src=d2,
                reduce="amin", include_self=False,
            )
            out[:, s:e] = min_d2.sqrt().t()
        else:
            beta = float(soft_beta)
            d = d2.sqrt()                        # [Gc, N]
            neg_beta_d = -beta * d               # [Gc, N]
            max_per_t = torch.full(
                (Gc, num_types), float("-inf"),
                device=device, dtype=dtype,
            )
            max_per_t.scatter_reduce_(
                dim=1, index=type_idx_chunk, src=neg_beta_d,
                reduce="amax", include_self=False,
            )
            shifted = neg_beta_d - max_per_t.gather(1, type_idx_chunk)
            sums = torch.zeros(
                (Gc, num_types), device=device, dtype=dtype,
            )
            sums.scatter_add_(
                dim=1, index=type_idx_chunk, src=shifted.exp(),
            )
            lse = max_per_t + sums.clamp_min(1e-30).log()
            out[:, s:e] = (-lse / beta).t()

    return out  # [T, G]


# ─────────────────────────────────────────────────────────────────────────────
# Loss modules (forward works with DataHolder; WandB hooks stripped)
# ─────────────────────────────────────────────────────────────────────────────


class CahnHilliardEnergyCurveLoss(nn.Module):
    """Per-cell-type Cahn-Hilliard energy-curve loss (vector-wise).

    For every sample and every cell type:

        1. Build a shared (square) grid covering the union bbox of the GT
           and predicted points of that type.
        2. For each radius ``r`` build the landscape ``phi(x; r)`` for both
           GT and prediction and evaluate the CH energy.
        3. Compare the two energy vectors per-radius with
           ``1 - exp(-|E_pred(r) - E_gt(r)| / (|E_gt(r)| + eps))`` and
           average over radii.

    Sample loss = mean over cell types; batch loss = mean over samples.
    """

    def __init__(
        self,
        radii: Sequence[float] = (0.005, 0.01, 0.02, 0.04, 0.08),
        grid_resolution: int = 64,
        kappa: float = 1.0,
        bump_fn: BumpFn = sigmoid_bump,
        bump_kwargs: Optional[Dict] = None,
        combine: Literal["soft_max", "hard_max"] = "soft_max",
        soft_max_beta: float = 16.0,
        support_factor: float = 10.0,
        landscape_chunk_size: int = 128,
        square_bbox: bool = True,
        margin: float = 0.05,
        eps: float = 1e-6,
        min_cells_per_type: int = 2,
        viz_every_n_epochs: int = 0,
        viz_num_figures: int = 1,
    ) -> None:
        super().__init__()
        if len(radii) < 1:
            raise ValueError("Need at least one radius for the energy curve.")
        self.radii = tuple(float(r) for r in sorted(radii))
        self.grid_resolution = int(grid_resolution)
        self.kappa = float(kappa)
        self.bump_fn = bump_fn
        self.bump_kwargs = dict(bump_kwargs or {})
        self.combine = combine
        self.soft_max_beta = float(soft_max_beta)
        self.support_factor = float(support_factor)
        self.landscape_chunk_size = int(landscape_chunk_size)
        self.square_bbox = bool(square_bbox)
        self.margin = float(margin)
        self.eps = float(eps)
        self.min_cells_per_type = int(min_cells_per_type)
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        self.viz_num_figures = int(viz_num_figures)
        self.current_epoch: int = 0

    def _normalized_exp_diff(
        self,
        e_pred: torch.Tensor,
        e_gt: torch.Tensor,
    ) -> torch.Tensor:
        rel = (e_pred - e_gt).abs() / (e_gt.detach().abs() + self.eps)
        return 1.0 - torch.exp(-rel)

    def _landscape(
        self,
        positions_xy: torch.Tensor,
        type_mask: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        return build_continuous_landscape_from_points(
            positions_xy, type_mask, grid_x, grid_y,
            radius=radius,
            bump_fn=self.bump_fn,
            bump_kwargs={"decay_rate": 4 / radius, "shift": 128 * radius},
            combine=self.combine,
            soft_max_beta=self.soft_max_beta,
            support_factor=self.support_factor,
            chunk=self.landscape_chunk_size,
        )

    def _sample_loss(
        self,
        pred_pos: torch.Tensor,     # [N, >=2]
        true_pos: torch.Tensor,     # [N, >=2]
        mask: torch.Tensor,         # [N]
        cell_class: torch.Tensor,   # [N] integer ids (padding marked with <0)
    ) -> Tuple[torch.Tensor, int]:
        device = pred_pos.device
        dtype = pred_pos.dtype

        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        if cell_class.dim() == 2 and cell_class.shape[-1] == 1:
            cell_class = cell_class.squeeze(-1)
        elif cell_class.dim() != 1:
            raise ValueError(
                f"CahnHilliardEnergyCurveLoss expected cell_class shape [N] or [N, 1], "
                f"got {tuple(cell_class.shape)}."
            )

        if mask_b.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        unique_types = torch.unique(cell_class[mask_b])

        total = torch.zeros((), device=device, dtype=dtype)
        n_types = 0

        for ct in unique_types.tolist():
            if ct < 0:
                continue
            type_mask = mask_b & (cell_class == ct)
            n_cells = int(type_mask.sum().item())
            if n_cells < self.min_cells_per_type:
                continue

            pts_true_type = true_xy[type_mask]
            pts_pred_type = pred_xy[type_mask]

            grid_x, grid_y, dx, dy = shared_square_grid(
                pts_true_type, pts_pred_type,
                grid_resolution=self.grid_resolution,
                margin=self.margin,
                square=self.square_bbox,
            )

            ch_kwargs = dict(
                kappa=self.kappa,
                bump_fn=self.bump_fn,
                bump_kwargs=self.bump_kwargs,
                combine=self.combine,
                soft_max_beta=self.soft_max_beta,
                support_factor=self.support_factor,
                chunk=self.landscape_chunk_size,
            )
            with torch.no_grad():
                e_gt_t = compute_ch_energy_curve_from_points(
                    true_xy, type_mask, grid_x, grid_y, dx, dy, self.radii,
                    **ch_kwargs,
                )
            e_pred_t = compute_ch_energy_curve_from_points(
                pred_xy, type_mask, grid_x, grid_y, dx, dy, self.radii,
                **ch_kwargs,
            )

            per_radius_loss = self._normalized_exp_diff(e_pred_t, e_gt_t)
            total = total + per_radius_loss.mean()
            n_types += 1

        if n_types == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        return total / float(n_types), n_types

    def forward(
        self,
        masked_pred: Any,
        masked_true: Any,
        train_stage: bool = True,
        log: bool = False,
        cell_class: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute the CH energy-curve loss for a batch.

        ``masked_pred`` / ``masked_true`` are ``DataHolder``-like objects
        with ``.positions`` ``[B, N, >=2]``, ``.node_mask`` ``[B, N]`` and
        (on the true side) ``.cell_class`` ``[B, N]``.
        """
        if cell_class is None:
            cell_class = masked_true.cell_class
        if cell_class is None:
            raise ValueError(
                "CahnHilliardEnergyCurveLoss requires cell_class; none was provided."
            )

        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask

        B = pred_positions.shape[0]
        losses = []
        type_counts = []
        for b in range(B):
            loss_b, n_types_b = self._sample_loss(
                pred_positions[b],
                true_positions[b],
                node_mask[b],
                cell_class[b],
            )
            losses.append(loss_b)
            type_counts.append(n_types_b)

        stacked = torch.stack(losses)
        loss = stacked.mean()

        to_log = None
        if log:
            key = (
                "train_loss/ch_energy_curve"
                if train_stage
                else "val_loss/ch_energy_curve"
            )
            to_log = {
                key: loss.item(),
                f"{key}/avg_types_per_sample": (
                    float(sum(type_counts)) / max(len(type_counts), 1)
                ),
            }

        return loss, to_log

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def reset(self) -> None:
        pass


class VoronoiPhasePairEnergyLoss(nn.Module):
    """Cahn-Hilliard energy loss over Voronoi phase landscapes of all
    unordered pairs of cell types.

    For every sample:
      1. Identify cell types with >= ``min_cells_per_type`` cells.
      2. Build a single shared grid covering the union bbox of those cells.
      3. Compute, once per type, the (soft-)distance fields to the nearest
         cell of that type, for both GT and pred.
      4. For each unordered pair ``(A, B)`` build
         ``phi_AB = tanh((dist_A - dist_B) / transition_width)`` and
         evaluate its CH energy for both sides.
      5. Per-pair loss
         ``L_AB = 1 - exp(-|E_pred - E_gt| / (|E_gt| + eps))``; sample loss
         = mean over pairs; batch loss = mean over samples.
    """

    def __init__(
        self,
        transition_width: float = 0.01,
        grid_resolution: int = 64,
        kappa: float = 1.0,
        soft_beta: Optional[float] = None,
        square_bbox: bool = True,
        margin: float = 0.05,
        eps: float = 1e-6,
        min_cells_per_type: int = 2,
        chunk: int = 4096,
        cache_gt: bool = False,
        cache_key_mode: Literal["content", "slot"] = "content",
        viz_every_n_epochs: int = 0,
        viz_num_figures: int = 4,
    ) -> None:
        super().__init__()
        self.transition_width = float(transition_width)
        self.grid_resolution = int(grid_resolution)
        self.kappa = float(kappa)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.square_bbox = bool(square_bbox)
        self.margin = float(margin)
        self.eps = float(eps)
        self.min_cells_per_type = int(min_cells_per_type)
        self.chunk = int(chunk)
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        self.viz_num_figures = int(viz_num_figures)
        self.current_epoch: int = 0
        self.cache_gt = bool(cache_gt)
        if cache_key_mode not in ("content", "slot"):
            raise ValueError(
                f"cache_key_mode must be 'content' or 'slot', got {cache_key_mode!r}."
            )
        self.cache_key_mode = cache_key_mode
        self._gt_cache: Dict = {}

    def clear_gt_cache(self) -> None:
        self._gt_cache.clear()

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _normalized_exp_diff(
        self,
        e_pred: torch.Tensor,
        e_gt: torch.Tensor,
    ) -> torch.Tensor:
        rel = (e_pred - e_gt).abs() / (e_gt.detach().abs() + self.eps)
        return 1.0 - torch.exp(-rel)

    def _sample_loss(
        self,
        pred_pos: torch.Tensor,     # [N, >=2]
        true_pos: torch.Tensor,     # [N, >=2]
        mask: torch.Tensor,         # [N]
        cell_class: torch.Tensor,   # [N] integer ids (padding marked with <0)
        *,
        batch_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        cell_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        device = pred_pos.device
        dtype = pred_pos.dtype

        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        if cell_class.dim() == 2 and cell_class.shape[-1] == 1:
            cell_class = cell_class.squeeze(-1)
        elif cell_class.dim() != 1:
            raise ValueError(
                f"VoronoiPhasePairEnergyLoss expected cell_class shape [N] or [N, 1], "
                f"got {tuple(cell_class.shape)}."
            )

        if mask_b.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        cache_key = None
        if self.cache_gt:
            if self.cache_key_mode == "content" and cell_id is not None:
                cache_key = (
                    "cid",
                    cell_id.detach().to(torch.int64).cpu().contiguous().numpy().tobytes(),
                )
            elif self.cache_key_mode == "slot" \
                    and batch_idx is not None and sample_idx is not None:
                cache_key = ("slot", int(batch_idx), int(sample_idx))

        cached: Optional[Dict] = None
        if cache_key is not None:
            cached = self._gt_cache.get(cache_key)
            if cached is not None and cell_id is not None:
                cached_cid = cached.get("cell_id")
                if cached_cid is None or cached_cid.shape != cell_id.shape \
                        or not torch.equal(cached_cid, cell_id.detach()):
                    cached = None

        if cached is None:
            masked_classes = cell_class[mask_b]
            masked_classes = masked_classes[masked_classes >= 0]
            if masked_classes.numel() == 0:
                if cache_key is not None:
                    self._gt_cache[cache_key] = {
                        "cell_id": cell_id.detach() if cell_id is not None else None,
                        "valid_types": [],
                    }
                return torch.zeros((), device=device, dtype=dtype), 0

            uniq, counts = torch.unique(masked_classes, return_counts=True)
            keep = counts >= self.min_cells_per_type
            valid_types_t = uniq[keep]
            num_types = int(valid_types_t.numel())
            if num_types < 2:
                if cache_key is not None:
                    self._gt_cache[cache_key] = {
                        "cell_id": cell_id.detach() if cell_id is not None else None,
                        "valid_types": [],
                    }
                return torch.zeros((), device=device, dtype=dtype), 0

            participates = mask_b & torch.isin(cell_class, valid_types_t)
            classes_kept = cell_class[participates]
            cell_type_idx = torch.searchsorted(valid_types_t, classes_kept)

            valid_types = valid_types_t.tolist()

            gt_bbox_pts = true_xy[mask_b]
            grid_x, grid_y, dx, dy = shared_square_grid(
                gt_bbox_pts, gt_bbox_pts,
                grid_resolution=self.grid_resolution,
                margin=self.margin,
                square=self.square_bbox,
            )

            ny, nx = grid_y.shape[0], grid_x.shape[0]
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
            grid_norm_sq = (grid_xy * grid_xy).sum(-1)

            pair_idx = torch.triu_indices(
                num_types, num_types, offset=1, device=device,
            )
            pair_idx_a = pair_idx[0]
            pair_idx_b = pair_idx[1]

            with torch.no_grad():
                gt_cells_xy = true_xy[participates]
                gt_dists = compute_distance_fields_per_type_fused(
                    grid_xy=grid_xy,
                    grid_norm_sq=grid_norm_sq,
                    cells_xy=gt_cells_xy,
                    cell_type_idx=cell_type_idx,
                    num_types=num_types,
                    soft_beta=self.soft_beta,
                    chunk=self.chunk,
                ).view(num_types, ny, nx)

                phi_gt = voronoi_phase_field(
                    gt_dists[pair_idx_a],
                    gt_dists[pair_idx_b],
                    self.transition_width,
                )
                e_gt_vec = cahn_hilliard_energy(
                    phi_gt, dx, dy, self.kappa,
                )

                del gt_dists, phi_gt

            cached = {
                "cell_id": cell_id.detach() if cell_id is not None else None,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_xy": grid_xy,
                "grid_norm_sq": grid_norm_sq,
                "dx": float(dx),
                "dy": float(dy),
                "ny": ny,
                "nx": nx,
                "valid_types": valid_types,
                "valid_types_t": valid_types_t,
                "num_types": num_types,
                "participates": participates,
                "cell_type_idx": cell_type_idx,
                "pair_idx_a": pair_idx_a,
                "pair_idx_b": pair_idx_b,
                "e_gt": e_gt_vec,
            }
            if cache_key is not None:
                self._gt_cache[cache_key] = cached

        if len(cached.get("valid_types", [])) < 2:
            return torch.zeros((), device=device, dtype=dtype), 0

        grid_xy = cached["grid_xy"]
        grid_norm_sq = cached["grid_norm_sq"]
        dx = cached["dx"]
        dy = cached["dy"]
        ny = cached["ny"]
        nx = cached["nx"]
        num_types = cached["num_types"]
        participates = cached["participates"]
        cell_type_idx = cached["cell_type_idx"]
        pair_idx_a = cached["pair_idx_a"]
        pair_idx_b = cached["pair_idx_b"]
        e_gt_vec = cached["e_gt"]

        pred_cells_xy = pred_xy[participates]
        pred_dists = compute_distance_fields_per_type_fused(
            grid_xy=grid_xy,
            grid_norm_sq=grid_norm_sq,
            cells_xy=pred_cells_xy,
            cell_type_idx=cell_type_idx,
            num_types=num_types,
            soft_beta=self.soft_beta,
            chunk=self.chunk,
        ).view(num_types, ny, nx)

        phi_pred = voronoi_phase_field(
            pred_dists[pair_idx_a],
            pred_dists[pair_idx_b],
            self.transition_width,
        )
        e_pred_vec = cahn_hilliard_energy(phi_pred, dx, dy, self.kappa)

        loss_per_pair = self._normalized_exp_diff(e_pred_vec, e_gt_vec)
        n_pairs = int(pair_idx_a.numel())

        if n_pairs == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        return loss_per_pair.sum() / float(n_pairs), n_pairs

    def forward(
        self,
        masked_pred: Any,
        masked_true: Any,
        train_stage: bool = True,
        log: bool = False,
        cell_class: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute the Voronoi phase-pair CH energy loss for a batch."""
        if cell_class is None:
            cell_class = masked_true.cell_class
        if cell_class is None:
            raise ValueError(
                "VoronoiPhasePairEnergyLoss requires cell_class; none was provided."
            )

        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask

        cell_id_full = getattr(masked_true, "cell_ID", None)
        if cell_id_full is not None and cell_id_full.dim() == 3 \
                and cell_id_full.shape[-1] == 1:
            cell_id_full = cell_id_full.squeeze(-1)

        B = pred_positions.shape[0]
        losses = []
        pair_counts = []
        for b in range(B):
            cell_id_b = None if cell_id_full is None else cell_id_full[b]
            loss_b, n_pairs_b = self._sample_loss(
                pred_positions[b],
                true_positions[b],
                node_mask[b],
                cell_class[b],
                batch_idx=batch_idx,
                sample_idx=b,
                cell_id=cell_id_b,
            )
            losses.append(loss_b)
            pair_counts.append(n_pairs_b)

        stacked = torch.stack(losses)
        loss = stacked.mean()

        to_log = None
        if log:
            key = (
                "train_loss/voronoi_phase_pair_ch_energy"
                if train_stage
                else "val_loss/voronoi_phase_pair_ch_energy"
            )
            to_log = {
                key: loss.item(),
                f"{key}/avg_pairs_per_sample": (
                    float(sum(pair_counts)) / max(len(pair_counts), 1)
                ),
            }

        return loss, to_log

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample helpers for the testing pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _to_tensor_positions(
    pred_positions: np.ndarray,   # [N, 2]
    gt_positions: np.ndarray,     # [N, 2]
    cell_class_int: np.ndarray,   # [N] integer class ids
    device: Any,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one sample's numpy arrays into the tensors the losses expect.

    Returns ``(pred_pos, true_pos, node_mask, cell_class)`` where
    ``pred_pos`` / ``true_pos`` are ``[N, 2]``, ``node_mask`` is ``[N]`` of
    ones, and ``cell_class`` is ``[N]`` integer ids. Everything sits on
    ``device``.
    """
    pred = torch.as_tensor(np.asarray(pred_positions), dtype=dtype, device=device)
    true = torch.as_tensor(np.asarray(gt_positions), dtype=dtype, device=device)
    cls = torch.as_tensor(np.asarray(cell_class_int), dtype=torch.long, device=device)
    n = pred.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=device)
    return pred, true, mask, cls


def ch_energy_curve_loss_per_sample(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    cell_class_int: np.ndarray,
    loss_module: CahnHilliardEnergyCurveLoss,
    device: Any,
) -> Tuple[float, int]:
    """Run :class:`CahnHilliardEnergyCurveLoss` on a single sample.

    Returns ``(loss_value, n_types)`` where ``loss_value`` is the mean-over-
    cell-types CH energy-curve loss in ``[0, 1)`` and ``n_types`` is the
    number of cell types that contributed.
    """
    pred, true, mask, cls = _to_tensor_positions(
        pred_positions, gt_positions, cell_class_int, device
    )
    with torch.no_grad():
        loss_t, n_types = loss_module._sample_loss(pred, true, mask, cls)
    return float(loss_t.item()), n_types


def voronoi_pair_energy_loss_per_sample(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    cell_class_int: np.ndarray,
    loss_module: VoronoiPhasePairEnergyLoss,
    device: Any,
) -> Tuple[float, int]:
    """Run :class:`VoronoiPhasePairEnergyLoss` on a single sample.

    Returns ``(loss_value, n_pairs)`` where ``loss_value`` is the mean-over-
    pairs Voronoi phase-pair CH energy loss in ``[0, 1)`` and ``n_pairs`` is
    the number of unordered cell-type pairs that contributed.
    """
    pred, true, mask, cls = _to_tensor_positions(
        pred_positions, gt_positions, cell_class_int, device
    )
    with torch.no_grad():
        loss_t, n_pairs = loss_module._sample_loss(pred, true, mask, cls)
    return float(loss_t.item()), n_pairs
