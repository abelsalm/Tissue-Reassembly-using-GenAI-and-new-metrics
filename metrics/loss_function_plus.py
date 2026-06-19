import math
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder


BumpFn = Callable[..., torch.Tensor]


def sigmoid_bump(
    r: torch.Tensor,
    cell_radius: float,
    decay_rate: Optional[float] = None,
    shift: float = 0.0,
) -> torch.Tensor:
    """Sigmoid bump: ``1 / (1 + exp(decay_rate * (r - cell_radius) - shift))``.

    Mirrors ``cahn_hilliard.sigmoid_bump``. Default ``decay_rate = 4 /
    cell_radius`` (matches the notebook convention ``d = 4 / radius``); the
    notebook also uses ``shift = 128 * radius`` -- that value is only
    meaningful when distances are expressed in integer pixel units, so we
    leave ``shift = 0`` by default and let the caller override.
    """
    cell_radius = max(float(cell_radius), 1e-12)
    if decay_rate is None:
        decay_rate = 4.0 / cell_radius
    decay_rate = float(decay_rate)
    return torch.sigmoid(-(decay_rate * (r - cell_radius) - shift))


# ---------- shared grid (mirror common_match_key) --------------------------


def shared_square_grid(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
    grid_resolution: int,
    margin: float = 0.0,
    square: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Axis-aligned grid covering the union bbox of two point sets.

    Replaces the numpy-side ``ContinuousLandscape2D.global_key`` +
    ``common_match_key`` pair. When ``square`` is True we expand the bbox
    to a square (same convention as ``build_continuous_landscape_from_points``)
    so that x and y have the same physical scale.
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

    Returns ``phi`` of shape ``[R, ny, nx]``. Distances from each cell to
    its local grid patch are computed **once** per chunk and reused for
    every radius; bump parameters and support masks are broadcast over the
    radius axis. Energy integration can then call
    :func:`cahn_hilliard_energy` once on the stacked field.
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

    # Size local patches for the largest support so every radius fits.
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

        # Broadcast distances over radii; bump params vary per radius.
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
            # flat_idx: [m, py, px]; bump: [m, R, py, px] -> scatter into [R, G].
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


# ---------- Cahn-Hilliard energy (mirror ContinuousLandscape2D energy) -----


def _grad_xy_edge_order_1(
    phi: torch.Tensor, dx: float, dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradient with ``edge_order=1`` boundaries.

    Numerically identical to ``torch.gradient(phi, spacing=(dy, dx),
    dim=(-2, -1))`` (and to ``np.gradient(..., edge_order=1)``), but
    inlined as slice + ``cat`` so it skips the Python wrapper and works
    naturally on any leading batch shape ``[..., ny, nx]``. Note: a
    direct ``F.conv2d`` with ``[-1, 0, 1]`` doesn't reproduce the
    ``edge_order=1`` boundary scaling without per-edge fix-ups, which
    would cancel the cuDNN win -- the slice+cat path below is faster in
    practice for our grid sizes.

    Returns ``(d_phi_dy, d_phi_dx)``, each shaped like ``phi``.
    """
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
    """Discrete energy density ``e = (phi^2 - 1)^2 + kappa * |grad phi|^2``.

    Mirrors :meth:`ContinuousLandscape2D.cahn_hilliard_energy_density`.
    Accepts any leading batch dims (e.g. ``[P, ny, nx]`` for stacked
    pair fields); the gradient is taken along the last two axes so a
    bare ``[ny, nx]`` input still works unchanged.
    """
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
    ``[...]`` (so a 2D phi yields a scalar; a stacked ``[P, ny, nx]``
    yields ``[P]`` energies in one fused kernel).
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


def compute_neighborhood_avg_and_density_multi_radius(
    positions: torch.Tensor,        # [B, N, D]
    features: torch.Tensor,         # [B, N, F]
    mask: torch.Tensor,             # [B, N] (bool / 0-1)
    radii: Sequence[float],
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
    include_self: bool = True,
    cached_dists: Optional[torch.Tensor] = None,
    return_neighborhood_sum: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-shell neighborhood averages, log-density, and optional sums.

    ``radii`` may be given in any order; they are sorted ascending internally.
    Output axis ``R`` indexes annuli: shell ``0`` is the ball of radius
    ``r_0``; shell ``k > 0`` is the ring between ``r_{k-1}`` and ``r_k``
    (cumulative mask at ``r_k`` minus cumulative mask at ``r_{k-1}``).

    Returns:
        avg:         ``[B, R, N, F]`` -- features averaged over neighbors
                     in each shell only. Padded rows are zeroed out.
        log_density: ``[B, R, N]`` -- ``log(sum_j w^shell_ij + eps)`` per
                     shell (soft count in that annulus only).
        neighborhood_sum: ``[B, R, N, F]`` or ``None`` -- weighted feature
                     sum in each shell before dividing by the soft count.

    Sharing across shells:
      * ``cdist`` is computed **once** (or reused via ``cached_dists``)
        -- this is the only ``O(B N^2 D)`` op.
      * ``valid_j`` (padding mask broadcast across ``i``) and the
        self-exclusion ``eye`` are built once.
      * Per listed radius we build one cumulative ball mask, subtract the
        previous cumulative mask to get the shell, then one matmul.

    Memory:
      Peak memory stays the same as the single-radius helper
      (``O(B N^2 + B N F)``). We deliberately *do not* materialise a
      ``[B, R, N, N]`` stacked-weights tensor (which would be ``R x``
      more memory) -- the python loop over typically ``R <= 8`` radii
      is negligible compared to the matmul cost.
    """
    if len(radii) < 1:
        raise ValueError("Need at least one radius.")
    B, N, _ = positions.shape
    dtype = features.dtype
    device = positions.device

    # ONE cdist per side, shared by every radius. We allow the caller to
    # pass a cached one (e.g. GT-side reused across epochs when positions
    # are frozen) but most callers leave it at ``None`` -- the GT side is
    # already wrapped in ``torch.no_grad`` upstream, so there is no
    # autograd graph to worry about.
    if cached_dists is None:
        dists = torch.cdist(positions, positions, p=2)  # [B, N, N]
    else:
        dists = cached_dists

    # Padding mask for neighbors j -- broadcast over the i axis.
    valid_j = mask.to(dtype).unsqueeze(1)  # [B, 1, N]

    # Pre-build self-exclusion if requested; identical for every radius.
    eye: Optional[torch.Tensor] = None
    if not include_self:
        eye = torch.eye(N, device=device, dtype=dtype)  # [N, N], broadcasts over B

    def _cumulative_ball_weights(radius: float) -> torch.Tensor:
        if soft_beta is None:
            w_cum = (dists <= float(radius)).to(dtype)
        else:
            w_cum = torch.sigmoid(float(soft_beta) * (float(radius) - dists))
        w_cum = w_cum * valid_j
        if eye is not None:
            w_cum = w_cum * (1.0 - eye)
        return w_cum

    avgs = []
    log_densities = []
    neighborhood_sums: list = []
    w_cum_prev: Optional[torch.Tensor] = None
    for r in sorted(float(x) for x in radii):
        w_cum = _cumulative_ball_weights(r)
        if w_cum_prev is None:
            w = w_cum
        else:
            # Annulus: membership in (r_prev, r], not the full ball(r).
            w = (w_cum - w_cum_prev).clamp_min(0.0)
        w_cum_prev = w_cum

        # Soft cell-count = sum of shell weights along the neighbor axis.
        sum_w = w.sum(dim=-1)            # [B, N]
        log_d = torch.log(sum_w + eps)   # [B, N]

        # Averaged features: same single-matmul trick as the single-radius
        # helper. Reusing ``sum_w`` avoids re-summing inside ``denom``.
        sum_f = torch.matmul(w, features)            # [B, N, F]
        denom = sum_w.unsqueeze(-1).clamp_min(eps)   # [B, N, 1]
        avg = sum_f / denom                          # [B, N, F]

        if return_neighborhood_sum:
            neighborhood_sums.append(sum_f)

        avgs.append(avg)
        log_densities.append(log_d)

    avg_t = torch.stack(avgs, dim=1)            # [B, R, N, F]
    log_density_t = torch.stack(log_densities, dim=1)  # [B, R, N]
    neighborhood_sum_t = (
        torch.stack(neighborhood_sums, dim=1)
        if return_neighborhood_sum
        else None
    )

    # Zero out rows for padded cells i so they cannot contaminate the
    # later mean. ``log_density`` is left untouched here; the loss class
    # multiplies it by the same valid_i mask before aggregating.
    valid_i = mask.to(dtype).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
    avg_t = avg_t * valid_i
    if neighborhood_sum_t is not None:
        neighborhood_sum_t = neighborhood_sum_t * valid_i
    return avg_t, log_density_t, neighborhood_sum_t


def precompute_gt_neighborhood_multi_radius(
    true_positions: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
    radii: Sequence[float],
    *,
    save_path: Optional[str] = None,
    soft_beta: Optional[float] = None,
    include_self: bool = True,
    eps: float = 1e-6,
    return_neighborhood_sum: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Multi-radius analogue of :func:`precompute_gt_neighborhood_features`.

    Computes the GT-side ``(avg, log_density[, neighborhood_sum])`` tensors
    once under ``torch.no_grad`` so they can be looked up at every training
    step instead of being recomputed.

    If ``save_path`` is provided we write the tensors to disk via
    ``torch.save`` so they can be reloaded across runs.
    """
    with torch.no_grad():
        avg, log_density, neighborhood_sum = (
            compute_neighborhood_avg_and_density_multi_radius(
                true_positions, node_features, node_mask,
                radii=radii,
                soft_beta=soft_beta,
                eps=eps,
                include_self=include_self,
                return_neighborhood_sum=return_neighborhood_sum,
            )
        )
    if save_path is not None:
        payload = {
            "avg": avg.detach().cpu(),
            "log_density": log_density.detach().cpu(),
        }
        if neighborhood_sum is not None:
            payload["neighborhood_sum"] = neighborhood_sum.detach().cpu()
        torch.save(payload, save_path)
    return avg, log_density, neighborhood_sum


def _align_radii_and_transcriptome_tolerances(
    radii: Sequence[float],
    tolerance: Union[float, Sequence[float]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Sort radii ascending and permute tolerances to stay index-aligned.

    ``tolerance[i]`` in the input pairs with ``radii[i]`` before sorting.
    """
    r_list = [float(r) for r in radii]
    if isinstance(tolerance, (float, int)):
        t_list = [float(tolerance)] * len(r_list)
    else:
        t_list = [float(t) for t in tolerance]
        if len(t_list) != len(r_list):
            raise ValueError(
                "transcriptome_tolerance must have the same length as radii "
                f"({len(t_list)} != {len(r_list)})."
            )
    pairs = sorted(zip(r_list, t_list), key=lambda pair: pair[0])
    sorted_radii = tuple(r for r, _ in pairs)
    sorted_tols = tuple(t for _, t in pairs)
    return sorted_radii, sorted_tols


def _gt_relative_band_squared_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tolerance: Union[float, torch.Tensor],
    *,
    gate_beta: Optional[float] = None,
    forgiveness: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-gene squared error with a GT-relative tolerance band.

    Applied element-wise on ``[..., F]``: for each cell, shell, and gene
    ``f``, the half-width is ``tolerance * |gt[..., f]|``. ``tolerance``
    may be a scalar or a per-shell tensor broadcastable as ``[1, R, 1, 1]``.
    ``gate`` is 0 inside the band and 1 outside (sigmoid when ``gate_beta``
    is set). Forgiveness ``f in [0, 1]`` linearly blends between plain
    ``(pred-gt)^2`` (``f=0``) and ``gate * (pred-gt)^2`` (``f=1``)::

        sq_err = (pred-gt)^2 * (1 - f * (1 - gate))

    So at epoch ``k`` of an ``n``-epoch ramp (``f = k/n``), in-band loss is
    scaled by ``(n-k)/n``. Scalar ``tolerance <= 0`` or ``f <= 0`` returns
    plain ``(pred-gt)^2``.
    """
    diff = pred - gt
    sq = diff.pow(2)
    if forgiveness <= 0.0:
        return sq
    if isinstance(tolerance, torch.Tensor):
        tol = tolerance.to(device=sq.device, dtype=sq.dtype)
    else:
        if tolerance <= 0.0:
            return sq
        tol = tolerance
    half_width = tol * gt.abs().clamp_min(eps)
    excess = diff.abs() - half_width
    if gate_beta is None:
        gate = torch.where(excess > 0, torch.ones_like(sq), torch.zeros_like(sq))
    else:
        gate = torch.sigmoid(float(gate_beta) * excess)
    forgiveness_t = torch.tensor(
        float(forgiveness), device=sq.device, dtype=sq.dtype
    )
    return sq * (1.0 - forgiveness_t * (1.0 - gate))


class MultiRadiusNeighborhoodLoss(nn.Module):
    """Multi-radius neighborhood loss: transcriptome RMSE + log-density diff.

    Pipeline (per side: ``pred``, ``gt``):
      1. Sort ``radii`` ascending and build **annulus** masks (each shell
         is ``ball(r_k) \\ ball(r_{k-1})``). Compute ``avg``, ``log_density``,
         and optional neighborhood sums per shell via
         :func:`compute_neighborhood_avg_and_density_multi_radius`.
      2. Per-cell per-shell per-**gene** transcriptome term (averaged vs averaged).
         For each gene ``f`` and shell ``k``, a GT-relative band
         ``gt_f +/- tolerance[k] * |gt_f|`` softly zeros in-band error
         (sigmoid gate). ``tolerance[k]`` aligns with ``radii[k]`` in the
         config list (re-sorted with radii internally). Forgiveness ramps over the first
         ``transcriptome_tolerance_warmup_epochs`` epochs: epoch ``0`` is
         plain RMSE; epoch ``k`` keeps ``(n-k)/n`` of in-band loss; epoch
         ``n+`` applies the full band::

             rmse(i, r) = sqrt( mean_f gated_sq_err_f + eps )

      3. Per-cell per-radius density term::

             dens(i, r) = | log_density_pred(i, r) - log_density_gt(i, r) |

      4. Per-cell per-radius *global* transcriptome term (pred vs GT
         neighborhood **without** averaging over neighbor count)::

             global_rmse(i, r) = sqrt( mean_f gated_sq_err_f + eps )

         with the same GT-relative band as the transcriptome term.
         ``sum_side = sum_j w_ij * features[j]`` (no division by count).

      5. Scale only the global term by ``loss_radius_scale * r`` (default
         ``512 * r``) before aggregation.
      6. Aggregate: mean over valid cells, then mean over radii, then::

             L = transcriptome_term
               + density_weight * density_term
               + global_transcriptome_weight * global_term

    Differentiability is governed by ``soft_beta`` exactly like the
    single-radius variant (set it to ``None`` for a non-differentiable
    diagnostic, set it to a positive float for training).

    The GT side does **not** depend on the model and can be precomputed
    via :meth:`precompute_gt` and passed as ``cached_gt_avg`` /
    ``cached_gt_log_density`` / ``cached_gt_neighborhood_sum`` to skip the
    GT recomputation every step.
    """

    def __init__(
        self,
        radii: Sequence[float] = (0.005, 0.01, 0.05, 0.1),
        density_weight: float = 1.0,
        global_transcriptome_weight: float = 0.0,
        loss_radius_scale: float = 512.0,
        transcriptome_tolerance: Union[float, Sequence[float]] = 0.05,
        transcriptome_tolerance_gate_beta: Optional[float] = 256.0,
        transcriptome_tolerance_warmup_epochs: int = 100,
        soft_beta: Optional[float] = None,
        eps: float = 1e-6,
        include_self: bool = True,
    ) -> None:
        super().__init__()
        if len(radii) < 1:
            raise ValueError("Need at least one radius.")
        # Sorted ascending: index k is the annulus with outer radius r_k;
        # per-shell tolerances are permuted to stay aligned with radii.
        self.radii, self.transcriptome_tolerance_per_shell = (
            _align_radii_and_transcriptome_tolerances(radii, transcriptome_tolerance)
        )
        self.density_weight = float(density_weight)
        self.global_transcriptome_weight = float(global_transcriptome_weight)
        self.loss_radius_scale = float(loss_radius_scale)
        self.transcriptome_tolerance_gate_beta = (
            None
            if transcriptome_tolerance_gate_beta is None
            else float(transcriptome_tolerance_gate_beta)
        )
        self.transcriptome_tolerance_warmup_epochs = int(
            transcriptome_tolerance_warmup_epochs
        )
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.eps = float(eps)
        self.include_self = bool(include_self)
        self.current_epoch: int = 0
        # Per-step caches surfaced through ``log_epoch_metrics``.
        self._last_loss: float = -1.0
        self._last_transcriptome: float = -1.0
        self._last_density: float = -1.0
        self._last_global_transcriptome: float = -1.0

    def set_current_epoch(self, epoch: int) -> None:
        """Track trainer epoch for tolerance-band warmup (called each epoch)."""
        self.current_epoch = int(epoch)

    def _tolerance_forgiveness(self) -> float:
        """Fraction of in-band forgiveness active at ``current_epoch``.

        Returns ``min(epoch, n) / n`` when ``n > 0``, else ``1.0``.
        Epoch ``0`` -> ``0`` (no band); epoch ``n+`` -> ``1`` (full band).
        """
        n = self.transcriptome_tolerance_warmup_epochs
        if n <= 0:
            return 1.0
        return min(max(self.current_epoch, 0), n) / float(n)

    def _shell_tolerance_tensor(
        self, *, device: torch.device, dtype: torch.dtype, n_shells: int
    ) -> torch.Tensor:
        """Per-shell tolerance as ``[1, R, 1, 1]`` for band gating."""
        if len(self.transcriptome_tolerance_per_shell) != n_shells:
            raise ValueError(
                "Shell count mismatch between radii and transcriptome tolerances."
            )
        return torch.tensor(
            self.transcriptome_tolerance_per_shell,
            device=device,
            dtype=dtype,
        ).view(1, n_shells, 1, 1)

    # ---------------- GT precompute / caching ----------------

    def precompute_gt(
        self,
        true_positions: torch.Tensor,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Convenience wrapper around
        :func:`precompute_gt_neighborhood_multi_radius`."""
        return precompute_gt_neighborhood_multi_radius(
            true_positions, node_features, node_mask,
            radii=self.radii,
            save_path=save_path,
            soft_beta=self.soft_beta,
            include_self=self.include_self,
            eps=self.eps,
            return_neighborhood_sum=self.global_transcriptome_weight > 0.0,
        )

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        cached_gt_avg: Optional[torch.Tensor] = None,
        cached_gt_log_density: Optional[torch.Tensor] = None,
        cached_gt_neighborhood_sum: Optional[torch.Tensor] = None,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,  # accept ``batch_idx`` etc. for drop-in compat.
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        node_mask = masked_true.node_mask
        node_features = masked_true.node_features
        true_xy = masked_true.positions[..., :2]
        pred_xy = masked_pred.positions[..., :2]
        use_global = self.global_transcriptome_weight > 0.0

        # ---- GT side: either reuse the cache or recompute under no_grad.
        gt_provided = (
            cached_gt_avg is not None and cached_gt_log_density is not None
        )
        if not gt_provided:
            with torch.no_grad():
                gt_avg, gt_logd, gt_sum = (
                    compute_neighborhood_avg_and_density_multi_radius(
                        true_xy, node_features, node_mask,
                        radii=self.radii,
                        soft_beta=self.soft_beta,
                        eps=self.eps,
                        include_self=self.include_self,
                        return_neighborhood_sum=use_global,
                    )
                )
        else:
            gt_avg = cached_gt_avg.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )
            gt_logd = cached_gt_log_density.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )
            if use_global:
                if cached_gt_neighborhood_sum is None:
                    # Recover sum from avg and soft count: sum_f = avg * sum_w.
                    sum_w = torch.exp(gt_logd) - self.eps
                    gt_sum = gt_avg * sum_w.unsqueeze(-1).clamp_min(0.0)
                else:
                    gt_sum = cached_gt_neighborhood_sum.detach().to(
                        device=node_features.device, dtype=node_features.dtype
                    )
            else:
                gt_sum = None

        # ---- Pred side: features come from GT (we're learning positions,
        # not gene expressions), neighborhoods come from predicted xys.
        pred_avg, pred_logd, pred_sum = (
            compute_neighborhood_avg_and_density_multi_radius(
                pred_xy, node_features, node_mask,
                radii=self.radii,
                soft_beta=self.soft_beta,
                eps=self.eps,
                include_self=self.include_self,
                return_neighborhood_sum=use_global,
            )
        )

        # ---- Transcriptome term: per-cell RMSE over the feature axis,
        # for each (sample, radius). GT-relative band zeros loss inside
        # ``gt +/- tolerance * |gt|`` per gene; outside uses ``(pred-gt)^2``.
        tolerance_forgiveness = self._tolerance_forgiveness()
        shell_tolerance = self._shell_tolerance_tensor(
            device=pred_avg.device,
            dtype=pred_avg.dtype,
            n_shells=pred_avg.shape[1],
        )
        sq_err = _gt_relative_band_squared_error(
            pred_avg,
            gt_avg,
            shell_tolerance,
            gate_beta=self.transcriptome_tolerance_gate_beta,
            forgiveness=tolerance_forgiveness,
            eps=self.eps,
        )                                              # [B, R, N, F]
        per_cell_rmse = torch.sqrt(sq_err.mean(dim=-1) + self.eps)

        # ---- Density term: |log d_pred - log d_gt|, per cell per radius.
        per_cell_dens = (pred_logd - gt_logd).abs()    # [B, R, N]

        # ---- Global transcriptome term: pred vs GT neighborhood weighted
        # sums (same comparison as transcriptome, but without / sum_w).
        if use_global:
            assert pred_sum is not None and gt_sum is not None
            sq_err_global = _gt_relative_band_squared_error(
                pred_sum,
                gt_sum,
                shell_tolerance,
                gate_beta=self.transcriptome_tolerance_gate_beta,
                forgiveness=tolerance_forgiveness,
                eps=self.eps,
            )                                              # [B, R, N, F]
            per_cell_global = torch.sqrt(
                sq_err_global.mean(dim=-1) + self.eps
            )                                              # [B, R, N]
            radius_div = torch.tensor(
                [self.loss_radius_scale * r for r in self.radii],
                device=per_cell_global.device,
                dtype=per_cell_global.dtype,
            ).view(1, -1, 1)
            per_cell_global = per_cell_global / radius_div
        else:
            per_cell_global = torch.zeros_like(per_cell_dens)

        # ---- Aggregation: mean over valid cells -> mean over radii ->
        # mean over batch. Doing it in two steps (cells, then radii)
        # means a sample with many cells does not dominate per-radius
        # statistics.
        valid_i = node_mask.to(per_cell_rmse.dtype).unsqueeze(1)   # [B, 1, N]
        n_valid = valid_i.sum(dim=-1).clamp_min(1.0)               # [B, 1]
        rmse_per_r = (per_cell_rmse * valid_i).sum(dim=-1) / n_valid  # [B, R]
        dens_per_r = (per_cell_dens * valid_i).sum(dim=-1) / n_valid  # [B, R]
        global_per_r = (per_cell_global * valid_i).sum(dim=-1) / n_valid  # [B, R]

        transcriptome_term = rmse_per_r.mean()
        density_term = dens_per_r.mean()
        global_term = global_per_r.mean()
        loss = (
            transcriptome_term
            + self.density_weight * density_term
            + self.global_transcriptome_weight * global_term
        )

        # Cache scalars for log_epoch_metrics (Lightning aggregates these
        # into a single per-epoch metric via on_epoch=True).
        if train_stage:
            self._last_transcriptome = float(transcriptome_term.detach().item())
            self._last_density = float(density_term.detach().item())
            self._last_global_transcriptome = float(global_term.detach().item())
            self._last_loss = float(loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/neighborhood_multi_radius": float(loss.detach().item()),
                f"{prefix}/neighborhood_multi_radius_transcriptome": float(
                    transcriptome_term.detach().item()
                ),
                #f"{prefix}/neighborhood_multi_radius_density": float(
                #    density_term.detach().item()
                #),
                f"{prefix}/neighborhood_multi_radius_global_transcriptome": float(
                    global_term.detach().item()
                ),
                f"{prefix}/tolerance_forgiveness": float(tolerance_forgiveness),
            }
            # Also expose per-radius diagnostics (one number per (B, r) pair
            # averaged over the batch). Useful for tuning the radius set:
            # if the smallest radius dominates ``rmse`` you may want to
            # add a larger one and vice-versa.
            rmse_per_r_mean = rmse_per_r.mean(dim=0).detach()
            dens_per_r_mean = dens_per_r.mean(dim=0).detach()
            for ri, r in enumerate(self.radii):
                to_log[f"{prefix}/neighborhood_multi_radius_rmse_r{r:g}"] = float(
                    rmse_per_r_mean[ri].item()
                )
                #to_log[f"{prefix}/neighborhood_multi_radius_density_r{r:g}"] = float(
                #    dens_per_r_mean[ri].item()
                #)
                to_log[
                    f"{prefix}/neighborhood_multi_radius_global_transcriptome_r{r:g}"
                ] = float(global_per_r.mean(dim=0).detach()[ri].item())
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """No running state; the last-loss caches are intentionally
        carried across the epoch boundary (overwritten by next forward)."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Expose the last-step components under ``train_epoch/...`` keys.

        Per-step call from ``training_step_func`` + ``on_epoch=True`` in
        ``log_dict`` makes Lightning average each key over the epoch.
        Four keys: the combined loss plus the transcriptome / density /
        global-transcriptome breakdown -- useful for monitoring which side
        dominates.
        """
        to_log = {
            "train_epoch/neighborhood_multi_radius": float(self._last_loss),
            "train_epoch/neighborhood_multi_radius_transcriptome": float(
                self._last_transcriptome
            ),
            "train_epoch/neighborhood_multi_radius_density": float(
                self._last_density
            ),
            "train_epoch/neighborhood_multi_radius_global_transcriptome": float(
                self._last_global_transcriptome
            ),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class SlidePointCloudMetricLoss(nn.Module):
    """Whole-slide point-cloud loss: CH AUC + PCA shape descriptors.

    Four weighted sub-terms (any weight ``0`` skips that branch)::

        L = ch_auc_weight        * L_ch_auc
          + anisotropy_weight    * |aniso_pred - aniso_gt|
          + omnivariance_weight  * |omni_pred  - omni_gt|
          + linearity_weight     * |lin_pred   - lin_gt|
    """

    def __init__(
        self,
        ch_auc_weight: float = 1.0,
        anisotropy_weight: float = 1.0,
        omnivariance_weight: float = 1.0,
        linearity_weight: float = 1.0,
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
        min_cells: int = 10,
        cache_gt: bool = True,
    ) -> None:
        super().__init__()
        if len(radii) < 2:
            raise ValueError("Need at least two radii for CH AUC integration.")
        self.ch_auc_weight = float(ch_auc_weight)
        self.anisotropy_weight = float(anisotropy_weight)
        self.omnivariance_weight = float(omnivariance_weight)
        self.linearity_weight = float(linearity_weight)
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
        self.min_cells = int(min_cells)
        self.cache_gt = bool(cache_gt)
        self._gt_cache: Dict = {}
        self._last_loss: float = -1.0
        self._last_ch_auc: float = -1.0
        self._last_anisotropy: float = -1.0
        self._last_omnivariance: float = -1.0
        self._last_linearity: float = -1.0

    def clear_gt_cache(self) -> None:
        """Drop cached GT CH curves (e.g. after rechunk / re-shuffle)."""
        self._gt_cache.clear()

    def _ch_curve_kwargs(self) -> Dict:
        return dict(
            kappa=self.kappa,
            bump_fn=self.bump_fn,
            bump_kwargs=self.bump_kwargs,
            combine=self.combine,
            soft_max_beta=self.soft_max_beta,
            support_factor=self.support_factor,
            chunk=self.landscape_chunk_size,
        )

    def _gt_cache_key(
        self, cell_id: Optional[torch.Tensor]
    ) -> Optional[Tuple[str, bytes]]:
        if not self.cache_gt or cell_id is None:
            return None
        return (
            "cid",
            cell_id.detach().to(torch.int64).cpu().contiguous().numpy().tobytes(),
        )

    def _normalized_exp_diff(
        self, pred_val: torch.Tensor, gt_val: torch.Tensor
    ) -> torch.Tensor:
        rel = (pred_val - gt_val).abs() / (gt_val.detach().abs() + self.eps)
        return 1.0 - torch.exp(-rel)

    def _ch_auc_term(
        self,
        pred_xy: torch.Tensor,
        true_xy: torch.Tensor,
        mask_b: torch.Tensor,
        cell_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_cells = int(mask_b.sum().item())
        if n_cells < self.min_cells:
            return pred_xy.sum() * 0.0

        pts_true = true_xy[mask_b]
        pts_pred = pred_xy[mask_b]
        cache_key = self._gt_cache_key(cell_id)
        cached = self._gt_cache.get(cache_key) if cache_key is not None else None
        if cached is not None and cell_id is not None:
            cached_cid = cached.get("cell_id")
            if cached_cid is None or cached_cid.shape != cell_id.shape \
                    or not torch.equal(cached_cid, cell_id.detach()):
                cached = None

        if cached is None:
            if self.cache_gt:
                grid_x, grid_y, dx, dy = shared_square_grid(
                    pts_true,
                    pts_true,
                    grid_resolution=self.grid_resolution,
                    margin=self.margin,
                    square=self.square_bbox,
                )
            else:
                grid_x, grid_y, dx, dy = shared_square_grid(
                    pts_true,
                    pts_pred,
                    grid_resolution=self.grid_resolution,
                    margin=self.margin,
                    square=self.square_bbox,
                )
            with torch.no_grad():
                e_gt_curve = compute_ch_energy_curve_from_points(
                    true_xy,
                    mask_b,
                    grid_x,
                    grid_y,
                    dx,
                    dy,
                    self.radii,
                    **self._ch_curve_kwargs(),
                )
            radii_t = torch.tensor(
                self.radii, device=pred_xy.device, dtype=pred_xy.dtype
            )
            auc_gt = torch.trapezoid(e_gt_curve, radii_t)
            if cache_key is not None:
                self._gt_cache[cache_key] = {
                    "cell_id": cell_id.detach() if cell_id is not None else None,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "dx": dx,
                    "dy": dy,
                    "auc_gt": auc_gt,
                }
        else:
            grid_x = cached["grid_x"]
            grid_y = cached["grid_y"]
            dx = cached["dx"]
            dy = cached["dy"]
            auc_gt = cached["auc_gt"]

        e_pred_curve = compute_ch_energy_curve_from_points(
            pred_xy,
            mask_b,
            grid_x,
            grid_y,
            dx,
            dy,
            self.radii,
            **self._ch_curve_kwargs(),
        )
        radii_t = torch.tensor(
            self.radii, device=pred_xy.device, dtype=pred_xy.dtype
        )
        auc_pred = torch.trapezoid(e_pred_curve, radii_t)
        return self._normalized_exp_diff(auc_pred, auc_gt)

    def _sample_loss(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        mask: torch.Tensor,
        cell_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = pred_pos.sum() * 0.0
        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        if self.ch_auc_weight != 0.0:
            ch_term = self._ch_auc_term(pred_xy, true_xy, mask_b, cell_id=cell_id)
        else:
            ch_term = zero

        from metrics.cell_types_metrics import compute_slide_pointcloud_pca_descriptors

        gt_pca = compute_slide_pointcloud_pca_descriptors(
            true_pos, mask, min_cells=self.min_cells, eps=self.eps
        )
        pred_pca = compute_slide_pointcloud_pca_descriptors(
            pred_pos, mask, min_cells=self.min_cells, eps=self.eps
        )

        if gt_pca is None or pred_pca is None:
            aniso_term = zero
            omni_term = zero
            lin_term = zero
        else:
            gt_aniso, gt_omni, gt_lin = gt_pca
            pred_aniso, pred_omni, pred_lin = pred_pca
            aniso_term = (pred_aniso - gt_aniso.detach()).abs()
            omni_term = (pred_omni - gt_omni.detach()).abs()
            lin_term = (pred_lin - gt_lin.detach()).abs()

        total = (
            self.ch_auc_weight * ch_term
            + self.anisotropy_weight * aniso_term
            + self.omnivariance_weight * omni_term
            + self.linearity_weight * lin_term
        )
        return total, ch_term, aniso_term, omni_term, lin_term

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        batch_idx: Optional[int] = None,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask
        cell_id = masked_true.cell_ID

        losses = []
        ch_terms, aniso_terms, omni_terms, lin_terms = [], [], [], []
        for b in range(pred_positions.shape[0]):
            cid_b = cell_id[b] if cell_id is not None else None
            total, ch_t, an_t, om_t, li_t = self._sample_loss(
                pred_positions[b],
                true_positions[b],
                node_mask[b],
                cell_id=cid_b,
            )
            losses.append(total)
            ch_terms.append(ch_t)
            aniso_terms.append(an_t)
            omni_terms.append(om_t)
            lin_terms.append(li_t)

        if not losses:
            zero = pred_positions.sum() * 0.0
            return zero, None

        loss = torch.stack(losses).mean()
        if train_stage:
            self._last_loss = float(loss.detach().item())
            self._last_ch_auc = float(torch.stack(ch_terms).mean().detach().item())
            self._last_anisotropy = float(
                torch.stack(aniso_terms).mean().detach().item()
            )
            self._last_omnivariance = float(
                torch.stack(omni_terms).mean().detach().item()
            )
            self._last_linearity = float(
                torch.stack(lin_terms).mean().detach().item()
            )

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/slide_pointcloud": float(loss.detach().item()),
                f"{prefix}/slide_pointcloud_ch_auc": self._last_ch_auc,
                f"{prefix}/slide_pointcloud_anisotropy": self._last_anisotropy,
                f"{prefix}/slide_pointcloud_omnivariance": self._last_omnivariance,
                f"{prefix}/slide_pointcloud_linearity": self._last_linearity,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.clear_gt_cache()

    def log_epoch_metrics(self) -> Dict[str, float]:
        return {
            "train_epoch/slide_pointcloud": float(self._last_loss),
            "train_epoch/slide_pointcloud_ch_auc": float(self._last_ch_auc),
            "train_epoch/slide_pointcloud_anisotropy": float(self._last_anisotropy),
            "train_epoch/slide_pointcloud_omnivariance": float(self._last_omnivariance),
            "train_epoch/slide_pointcloud_linearity": float(self._last_linearity),
        }


class MultiRadiusSlideCombinedLoss(nn.Module):
    """``MultiRadiusNeighborhoodLoss`` + ``SlidePointCloudMetricLoss``.

    ``L = neighborhood_weight * L_neighborhood + L_slide`` where ``L_slide``
    is the internally weighted sum of the four slide sub-terms.
    """

    def __init__(
        self,
        neighborhood: MultiRadiusNeighborhoodLoss,
        slide: SlidePointCloudMetricLoss,
        neighborhood_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.neighborhood = neighborhood
        self.slide = slide
        self.neighborhood_weight = float(neighborhood_weight)
        self._last_loss: float = -1.0
        self._last_neighborhood: float = -1.0
        self._last_slide: float = -1.0

    def set_current_epoch(self, epoch: int) -> None:
        if hasattr(self.neighborhood, "set_current_epoch"):
            self.neighborhood.set_current_epoch(epoch)

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        batch_idx: Optional[int] = None,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        n_loss, n_log = self.neighborhood(
            masked_pred, masked_true, train_stage=train_stage, log=False
        )
        s_loss, s_log = self.slide(
            masked_pred,
            masked_true,
            train_stage=train_stage,
            log=False,
            batch_idx=batch_idx,
        )
        loss = self.neighborhood_weight * n_loss + s_loss

        if train_stage:
            self._last_loss = float(loss.detach().item())
            self._last_neighborhood = float(n_loss.detach().item())
            self._last_slide = float(s_loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/neighborhood_multi_radius_slide": float(
                    loss.detach().item()
                ),
                f"{prefix}/neighborhood_multi_radius_slide_neighborhood": float(
                    n_loss.detach().item()
                ),
                f"{prefix}/neighborhood_multi_radius_slide_slide": float(
                    s_loss.detach().item()
                ),
            }
            if n_log:
                to_log.update(n_log)
            if s_log:
                to_log.update(s_log)
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.neighborhood.reset()
        self.slide.reset()

    def clear_gt_cache(self) -> None:
        if hasattr(self.slide, "clear_gt_cache"):
            self.slide.clear_gt_cache()

    def log_epoch_metrics(self) -> Dict[str, float]:
        to_log = {
            "train_epoch/neighborhood_multi_radius_slide": float(self._last_loss),
            "train_epoch/neighborhood_multi_radius_slide_neighborhood": float(
                self._last_neighborhood
            ),
            "train_epoch/neighborhood_multi_radius_slide_slide": float(
                self._last_slide
            ),
        }
        to_log.update(self.neighborhood.log_epoch_metrics())
        to_log.update(self.slide.log_epoch_metrics())
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
