import math
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder


class LossFunction(nn.Module):
    """TrainLoss class for computing and logging training metrics.

    Attributes:
        train_position_mse (MeanSquaredError): Mean squared error for position predictions.

    Methods:
        __init__()
        forward(masked_pred: utils.Placeholder, masked_true: utils.Placeholder, log: bool) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]
        reset() -> None
        log_epoch_metrics() -> Dict[str, float]
    """

    def __init__(self) -> None:
        """
        Constructor to initialize the TrainLoss instance.

        Returns:
            None
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.true_positions = None
        self.pred_positions = None
        self.node_mask = None

    def masked_euclidean_distance(self, true_pos, pred_pos, mask):
        # Compute pairwise distance matrices
        true_dist_matrix = torch.cdist(true_pos[mask], true_pos[mask], p=2)
        pred_dist_matrix = torch.cdist(pred_pos[mask], pred_pos[mask], p=2)

        # Compute the square error between the distance matrices
        dist = self.mse(pred_dist_matrix, true_dist_matrix)

        return dist

    def compute_loss(self):
        # Compute MSE loss over all graphs
        losses = []
        for true_pos, pred_pos, mask in zip(
            self.true_positions, self.pred_positions, self.node_mask
        ):
            dist = self.masked_euclidean_distance(true_pos, pred_pos, mask)
            losses.append(dist)

        stacked_losses = torch.stack(losses)
        mse_loss = torch.mean(stacked_losses)
        return mse_loss

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,  # Default value set to True
        log: bool = True,  # Default value set to True
        **_unused: object,  # accept (and ignore) e.g. ``batch_idx`` from
        # the training loop so this loss stays drop-in compatible with
        # the richer loss APIs (``CombinedLossFunction``, etc.).
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        self.node_mask = masked_true.node_mask

        self.true_positions = masked_true.positions
        self.pred_positions = masked_pred.positions

        # Compute loss
        loss = self.compute_loss()

        # Log the loss
        to_log = None
        if log:
            loss_key = (
                "train_loss/position_mse" if train_stage else "val_loss/position_mse"
            )
            to_log = {loss_key: loss.item()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """Reset the training loss."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Log epoch-level metrics for training loss.

        Returns:
            Dict[str, float]: Dictionary of epoch-level metrics.
        """
        loss = self.compute_loss()
        epoch_position_loss = loss.item() if loss > 0 else -1.0

        to_log = {
            "train_epoch/position_mse": epoch_position_loss,
        }

        # Log epoch-level metrics if using WandB
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log


# ---------------------------------------------------------------------------
# Cahn-Hilliard energy-curve loss (vector comparison across radii)
# ---------------------------------------------------------------------------
#
# This is a differentiable (PyTorch) port of the notebook-side analysis
# implemented in
#     LUNA_perturb/Spatial-Transcriptomics-and-Perturbations-Modeling/
#         data_splits/cahn_hilliard.py
# (see ``ContinuousLandscape2D``, ``build_continuous_landscape_from_points``
#  and the ``plot_ch_energy_curves`` helper). The conventions are kept as
# close as possible to that reference so numerical values are directly
# comparable:
#
#   * The field ``phi`` lives in ``[-1, 1]``:
#         background is ``-1``, each point lifts the field towards ``+1``
#         via ``candidate = -1 + 2 * bump_i(r)``, and bumps are combined with
#         a (soft) pointwise ``max``. We then clip to ``[-1, 1]``.
#
#   * The Cahn-Hilliard energy uses the standard symmetric double-well
#         ``f(c) = (c^2 - 1)^2``
#     so the minima are at c = +/-1, matching the landscape range. The
#     integrated energy is
#         E = sum_ij [ f(c_ij) + kappa * |grad c|_ij^2 ] * dx * dy
#     (note: no ``1/2`` in front of the gradient term, to match the
#      reference implementation exactly).
#
#   * The ground-truth and predicted landscapes are built on a *shared*
#     grid obtained as the union of their (square) bounding boxes -- this
#     mirrors ``common_match_key`` from the reference.
#
#   * A radius sweep gives an energy curve ``E(r)``; the loss compares
#     the per-radius energies directly (no integration / AUC step).
#
# Per cell-type loss:
#     L_t = mean_r [ 1 - exp( - |E_pred(r) - E_gt(r)| / (|E_gt(r)| + eps) ) ]
# (each per-radius term is bounded in ``[0, 1)``, zero iff the energies
#  match, smooth everywhere; the denominator is detached so the
#  normalisation doesn't leak gradients; the mean over radii is also in
#  ``[0, 1)``).
#
# Sample loss is the sum over cell types; batch loss is the mean over
# samples. The whole pipeline is differentiable w.r.t. ``pred_positions``
# (only the grid corners are detached, since they are a discrete property
# of the bounding-box union).
# ---------------------------------------------------------------------------


# ---------- bump functions (mirror cahn_hilliard.py signatures) ------------


BumpFn = Callable[..., torch.Tensor]


def bump_gaussian(r: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaussian bump with scale = ``radius`` and ``bump(0) = 1``.

    Mirrors ``cahn_hilliard.bump_gaussian``.
    """
    radius = max(float(radius), 1e-12)
    return torch.exp(-(r * r) / (2.0 * radius * radius))


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


# ---------- continuous landscape (mirror build_continuous_landscape_from_points) ---


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
    """Torch port of ``cahn_hilliard.build_continuous_landscape_from_points``.

    Build a continuous 2D scalar field ``phi(x) in [-1, 1]`` on a regular
    grid from a set of points. The field is ``-1`` everywhere by default;
    each point contributes a bump so that ``candidate = -1 + 2 * bump_i(r)``
    reaches ``+1`` at the point. Bumps are combined by pointwise maximum
    (``combine='hard_max'``) or its smooth log-sum-exp approximation
    (``combine='soft_max'``, default, gives non-zero gradient to every
    contributing point).

    Points further than ``support_factor * radius`` from a grid cell are
    dropped from that cell's combination, matching the reference's
    per-point support window.

    Args:
        positions: ``[N, 2]`` tensor of 2D points (must support autograd for
            entries you want gradients w.r.t.).
        valid_mask: ``[N]`` bool tensor selecting the points to use.
        grid_x, grid_y: 1D grids defining the regular ``(nx, ny)`` lattice.
        radius: bump scale passed to ``bump_fn``.
        bump_fn: callable ``(r, radius, **bump_kwargs) -> tensor``, returning
            values in ``[0, 1]`` with ``bump(0) = 1``. Default is
            :func:`sigmoid_bump`; :func:`bump_gaussian` is also provided.
        combine: how bumps are aggregated -- ``'soft_max'`` (differentiable)
            or ``'hard_max'`` (exact max, gradient flows only to the argmax).
        soft_max_beta: inverse-temperature for the soft-max; larger values
            approach a hard max but with smaller gradient on non-dominant
            points.
        support_factor: per-point support cutoff in units of ``radius``
            (default 10, matching the reference default).
        chunk: number of points processed at once. Each chunk is evaluated on
            fixed-size local patches, which keeps memory bounded while giving
            the GPU enough work per kernel.

    Returns:
        ``phi`` tensor of shape ``(ny, nx)`` with values in ``[-1, 1]``
        (``field`` convention from :class:`ContinuousLandscape2D`, so
        dimension 0 is y and dimension 1 is x).
    """
    device = positions.device
    dtype = positions.dtype
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]

    if valid_mask.sum() == 0:
        # No cells of this type: empty field stays at -1 everywhere.
        return torch.full((ny, nx), -1.0, device=device, dtype=dtype)

    pts = positions[valid_mask]                 # [M, 2]

    # The support is expressed in radius units. Keeping it local is what
    # prevents materialising an [M, ny, nx] bump tensor for large cell types.
    support = float(support_factor) * float(radius)
    bump_kwargs = dict(bump_kwargs or {})
    chunk = max(1, int(chunk))

    # The grid comes from torch.linspace, so index-space patches can be
    # computed with simple arithmetic. This avoids per-cell searchsorted/item()
    # calls, which would synchronize CPU and GPU thousands of times.
    dx_grid = abs(float((grid_x[1] - grid_x[0]).detach().item())) if nx > 1 else 1.0
    dy_grid = abs(float((grid_y[1] - grid_y[0]).detach().item())) if ny > 1 else 1.0
    half_x = min(nx - 1, max(0, math.ceil(support / max(dx_grid, 1e-12)) + 1))
    half_y = min(ny - 1, max(0, math.ceil(support / max(dy_grid, 1e-12)) + 1))
    offset_x = torch.arange(-half_x, half_x + 1, device=device)
    offset_y = torch.arange(-half_y, half_y + 1, device=device)

    def local_patch_values(pts_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorise bump evaluation over a chunk of local point patches.

        The output tensors are shaped [m, patch_y, patch_x]. This is the
        compromise between the original full-grid broadcast [m, ny, nx]
        (fast but huge) and the point-by-point loop (small but slow).
        """
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
        # soft-max over bumps: (1/beta) * logsumexp(beta * b_i).
        # Init accumulator at exp(beta * 0) = 1 so a grid cell with no
        # contributing point degrades to soft-max value 0 -> phi = -1.
        beta = float(soft_max_beta)
        acc = torch.ones(ny * nx, device=device, dtype=dtype)        # baseline exp(beta * 0)
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_values(pts[start:start + chunk])
            # Only supported patch entries contribute; unsupported entries
            # have value 0 and therefore add nothing to the sparse-like sum.
            values = torch.exp(beta * bump) * (bump > 0).to(dtype)
            acc = acc.scatter_add(0, flat_idx.reshape(-1), values.reshape(-1))
        soft_max_bump = torch.log(acc.reshape(ny, nx)) / beta         # ~ max_i bumps_i
        phi = -1.0 + 2.0 * soft_max_bump
    elif combine == "hard_max":
        # Exact pointwise max over local patches. scatter_reduce keeps this
        # vectorized over chunks without allocating a full [M, ny, nx] tensor.
        best = torch.zeros(ny * nx, device=device, dtype=dtype)       # max bump so far; 0 == no point
        for start in range(0, pts.shape[0], chunk):
            flat_idx, bump = local_patch_values(pts[start:start + chunk])
            best = best.scatter_reduce(
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

    # Clip to [-1, 1] in case numerical slop pushes past the bounds.
    phi = phi.clamp(-1.0, 1.0)
    return phi


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


# ---------------------------------------------------------------------------
# Visualization helper (off the training hot path; called every N epochs)
# ---------------------------------------------------------------------------


def _build_phi_energy_figure(
    phi_pred: np.ndarray,
    phi_gt: np.ndarray,
    ed_pred: np.ndarray,
    ed_gt: np.ndarray,
    extent: Tuple[float, float, float, float],
    e_pred: float,
    e_gt: float,
    title: str,
):
    """Build a 2x2 matplotlib figure: phi (landscape) above energy density,
    pred on the left and GT on the right.

    Inputs are detached numpy arrays of shape ``[ny, nx]``. Returns the
    ``matplotlib.figure.Figure`` (caller is responsible for ``plt.close``).
    Imports matplotlib lazily so the loss module stays import-cheap.
    """
    # Force a non-interactive backend (training nodes have no display).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top row: phi field (the "landscape"). Bounded in [-1, 1] by tanh
    # (Voronoi) or by the soft_max combine clip (CH-curve), so we fix the
    # color range to make pred vs GT visually comparable.
    im00 = axes[0, 0].imshow(
        phi_pred, origin="lower", extent=extent,
        vmin=-1.0, vmax=1.0, cmap="RdBu_r",
    )
    axes[0, 0].set_title("phi_pred (landscape)")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(
        phi_gt, origin="lower", extent=extent,
        vmin=-1.0, vmax=1.0, cmap="RdBu_r",
    )
    axes[0, 1].set_title("phi_gt (landscape)")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom row: energy density (>= 0). Share the color range across
    # pred and GT so they're visually comparable.
    ed_max = float(max(ed_pred.max(), ed_gt.max(), 1e-12))
    im10 = axes[1, 0].imshow(
        ed_pred, origin="lower", extent=extent,
        vmin=0.0, vmax=ed_max, cmap="viridis",
    )
    axes[1, 0].set_title(f"energy density pred  (E={e_pred:.3e})")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im11 = axes[1, 1].imshow(
        ed_gt, origin="lower", extent=extent,
        vmin=0.0, vmax=ed_max, cmap="viridis",
    )
    axes[1, 1].set_title(f"energy density gt    (E={e_gt:.3e})")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------- Loss module ----------------------------------------------------


class CahnHilliardEnergyCurveLoss(nn.Module):
    """Per-cell-type Cahn-Hilliard energy-curve loss (vector-wise).

    For every sample in the batch and every cell type present in its mask:

        1. Build a shared (square) grid covering the union bbox of the
           ground-truth and predicted points of that type -- torch
           equivalent of the ``common_match_key`` workflow from
           ``cahn_hilliard.py``.
        2. For each radius ``r`` in ``self.radii``, build the continuous
           landscape ``phi(x; r)`` (values in ``[-1, 1]``) on that grid
           for both GT and prediction, then evaluate the CH energy.
        3. Compare the two energy vectors ``E_pred = [E_pred(r_1), ...]``
           and ``E_gt = [E_gt(r_1), ...]`` *vector-wise*, with the same
           scale-invariant exponential form as :class:`VoronoiPhasePairEnergyLoss`
           applied **per radius** and then averaged:

               ``L_t = mean_r [ 1 - exp(-|E_pred(r) - E_gt(r)|
                                          / (|E_gt(r)| + eps)) ]``.

           No trapezoidal integration / AUC step is involved.

    The sample loss is the mean over cell types; the batch loss is the
    mean over samples. Fully differentiable w.r.t. the predicted positions.

    Args:
        radii: sequence of bump radii (in position coordinate units) defining
            the energy curve.
        grid_resolution: number of grid points per axis.
        kappa: Cahn-Hilliard gradient-energy coefficient.
        bump_fn: bump function; defaults to :func:`sigmoid_bump`.
        bump_kwargs: extra kwargs forwarded to ``bump_fn`` normally computed using the radius with good proportions
        combine: ``'soft_max'`` (default, differentiable) or ``'hard_max'``.
        soft_max_beta: temperature for the soft-max bump combination.
        support_factor: per-point support cutoff in units of ``radius``.
        landscape_chunk_size: number of cells evaluated together when building
            local landscape patches. Larger values improve GPU utilization but
            use more memory.
        square_bbox: build the shared grid on the square bbox of GT ∪ pred
            (matches reference default).
        margin: fractional padding added to the (square) bbox per axis.
        eps: numerical floor used everywhere we divide.
        min_cells_per_type: minimum number of cells of a given type (in the
            masked sample) for the contribution to be included.
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
        # Periodic visualization (every N epochs); see comments on the
        # Voronoi loss for the same mechanism.
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        self.viz_num_figures = int(viz_num_figures)
        self.current_epoch: int = 0

    # ---------------- internals ----------------

    def _normalized_exp_diff(
        self,
        e_pred: torch.Tensor,
        e_gt: torch.Tensor,
    ) -> torch.Tensor:
        """``1 - exp(-|e_pred - e_gt| / (|e_gt| + eps))``.

        Operates element-wise so it returns a tensor with the same shape
        as its inputs (a vector when comparing the two ``[len(radii)]``
        energy curves, a scalar when comparing two scalars). The
        denominator is ``detached`` so the relative normalisation
        doesn't leak gradient into the GT side.
        """
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
            bump_kwargs={"decay_rate": 4/radius, "shift": 128*radius}, # here for now function of the radius
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

        # Only the first two spatial dims are used for the landscape.
        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        unique_types = torch.unique(cell_class[mask_b])

        total = torch.zeros((), device=device, dtype=dtype)
        n_types = 0

        for ct in unique_types.tolist():
            if ct < 0:  # padding / invalid class id
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

            e_gt_curve = []
            e_pred_curve = []
            for r in self.radii:
                # The GT curve is a constant target for this batch. Avoid
                # building an autograd graph for it; only the predicted
                # landscape must carry gradients back to pred positions.
                with torch.no_grad():
                    phi_gt = self._landscape(true_xy, type_mask, grid_x, grid_y, r)
                    e_gt_curve.append(cahn_hilliard_energy(phi_gt, dx, dy, self.kappa))
                phi_pred = self._landscape(pred_xy, type_mask, grid_x, grid_y, r)
                e_pred_curve.append(cahn_hilliard_energy(phi_pred, dx, dy, self.kappa))

            # Stack into [len(radii)] vectors and compare *vector-wise*
            # (no trapezoidal integration / AUC). The exponential
            # normalisation is applied per radius and then averaged --
            # equivalent to ``_normalized_exp_diff(e_pred_t, e_gt_t).mean()``
            # but spelt out to make the per-radius nature explicit.
            e_gt_t = torch.stack(e_gt_curve)
            e_pred_t = torch.stack(e_pred_curve)
            per_radius_loss = self._normalized_exp_diff(e_pred_t, e_gt_t)
            total = total + per_radius_loss.mean()
            n_types += 1

        if n_types == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        # Per-sample loss = mean of per-type (mean-over-radii) losses.
        return total / float(n_types), n_types

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = False,
        cell_class: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute the CH energy-curve loss (vector-wise, no AUC).

        Args:
            masked_pred: DataHolder with predicted ``.positions`` of shape
                ``[B, N, >=2]`` and ``.node_mask`` of shape ``[B, N]``.
            masked_true: DataHolder with ground-truth ``.positions`` and
                ``.cell_class`` of shape ``[B, N]`` (integer ids). If
                ``masked_true.cell_class`` is ``None`` you must pass
                ``cell_class`` explicitly.
            train_stage: True for training logs, False for validation logs.
            log: whether to emit a WandB log line.
            cell_class: optional override for the per-cell integer class
                tensor, shape ``[B, N]``.
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
            if wandb.run:
                wandb.log(to_log, commit=True)

        # Periodic visualization (training only, batch_idx == 0 of an
        # epoch divisible by ``viz_every_n_epochs``).
        if train_stage and self._should_log_viz(batch_idx):
            self._log_viz_to_wandb(
                pred_positions=pred_positions,
                true_positions=true_positions,
                node_mask=node_mask,
                cell_class=cell_class,
            )

        return loss, to_log

    def set_current_epoch(self, epoch: int) -> None:
        """Track the trainer's current epoch so viz can be gated."""
        self.current_epoch = int(epoch)

    def _should_log_viz(self, batch_idx: Optional[int]) -> bool:
        if self.viz_every_n_epochs <= 0:
            return False
        if batch_idx is None or int(batch_idx) != 0:
            return False
        return (self.current_epoch % self.viz_every_n_epochs) == 0

    def _compute_type_radius_viz_data(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        mask: torch.Tensor,
        cell_class_b: torch.Tensor,
        sample_idx: int,
        rng: np.random.Generator,
    ) -> Optional[Dict]:
        """Rebuild phi_pred / phi_gt / energy densities for one random
        (sample, type, radius). Returns ``None`` if the sample has no
        type passing the ``min_cells_per_type`` filter.
        """
        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        if cell_class_b.dim() == 2 and cell_class_b.shape[-1] == 1:
            cell_class_b = cell_class_b.squeeze(-1)
        if mask_b.sum() == 0:
            return None

        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        # Identify valid types (vectorised, single sync) -- mirrors the
        # filter used by ``_sample_loss``.
        masked_classes = cell_class_b[mask_b]
        masked_classes = masked_classes[masked_classes >= 0]
        if masked_classes.numel() == 0:
            return None
        uniq, counts = torch.unique(masked_classes, return_counts=True)
        keep = counts >= self.min_cells_per_type
        valid_types_t = uniq[keep]
        if int(valid_types_t.numel()) == 0:
            return None

        # Random type + random radius.
        ti = int(rng.integers(0, int(valid_types_t.numel())))
        ct = int(valid_types_t[ti].item())
        type_mask = mask_b & (cell_class_b == ct)

        radius = float(rng.choice(np.asarray(self.radii)))

        grid_x, grid_y, dx, dy = shared_square_grid(
            true_xy[type_mask], pred_xy[type_mask],
            grid_resolution=self.grid_resolution,
            margin=self.margin, square=self.square_bbox,
        )

        with torch.no_grad():
            phi_gt = self._landscape(true_xy, type_mask, grid_x, grid_y, radius)
            phi_pred = self._landscape(pred_xy, type_mask, grid_x, grid_y, radius)
            ed_gt = cahn_hilliard_energy_density(phi_gt, dx, dy, self.kappa)
            ed_pred = cahn_hilliard_energy_density(phi_pred, dx, dy, self.kappa)
            e_gt = float((ed_gt.sum() * dx * dy).item())
            e_pred = float((ed_pred.sum() * dx * dy).item())

        return {
            "phi_pred": phi_pred.detach().cpu().numpy(),
            "phi_gt": phi_gt.detach().cpu().numpy(),
            "ed_pred": ed_pred.detach().cpu().numpy(),
            "ed_gt": ed_gt.detach().cpu().numpy(),
            "extent": (
                float(grid_x[0].item()), float(grid_x[-1].item()),
                float(grid_y[0].item()), float(grid_y[-1].item()),
            ),
            "ct": ct, "radius": radius,
            "e_pred": e_pred, "e_gt": e_gt,
            "sample_idx": int(sample_idx),
        }

    def _log_viz_to_wandb(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        node_mask: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> None:
        """Render ``viz_num_figures`` random (sample, type, radius)
        panels and push them to WandB.
        """
        if not wandb.run:
            return
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return

        B = int(pred_positions.shape[0])
        rng = np.random.default_rng(seed=int(self.current_epoch))

        log_dict: Dict[str, "wandb.Image"] = {}
        max_attempts = max(self.viz_num_figures * 4, 8)
        figs_logged = 0
        attempt = 0
        while figs_logged < self.viz_num_figures and attempt < max_attempts:
            attempt += 1
            b = int(rng.integers(0, B))
            viz = self._compute_type_radius_viz_data(
                pred_pos=pred_positions[b],
                true_pos=true_positions[b],
                mask=node_mask[b],
                cell_class_b=cell_class[b],
                sample_idx=b,
                rng=rng,
            )
            if viz is None:
                continue
            title = (
                f"CH-curve viz | epoch {self.current_epoch} | "
                f"sample {viz['sample_idx']} | type {viz['ct']} | "
                f"radius {viz['radius']:.4f}"
            )
            fig = _build_phi_energy_figure(
                phi_pred=viz["phi_pred"], phi_gt=viz["phi_gt"],
                ed_pred=viz["ed_pred"], ed_gt=viz["ed_gt"],
                extent=viz["extent"],
                e_pred=viz["e_pred"], e_gt=viz["e_gt"],
                title=title,
            )
            log_dict[f"viz/ch_curve/fig_{figs_logged}"] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figs_logged += 1

        if log_dict:
            wandb.log(log_dict, commit=False)

    def reset(self) -> None:
        """No running state to reset."""
        pass


# ---------------------------------------------------------------------------
# Voronoi phase-separation energy loss (per cell-type pair)
# ---------------------------------------------------------------------------
#
# Differentiable torch port of
#     Celullar-Tissue-Spatial-Metrics-/cahn-hilliard-energy/cahn_hilliard.py
#         build_voronoi_phase_landscape_from_cell_types
# (see also the "comparing phase separation between two cell types" section
# of ch_energy_tests.ipynb).
#
# For an unordered pair of cell types ``(A, B)`` the landscape is
#
#       phi_AB(x) = tanh( ( dist_A(x) - dist_B(x) ) / w )
#
# where ``dist_t(x)`` is the distance from the grid cell ``x`` to the
# nearest cell of type ``t``. The field is naturally bounded in ``[-1, 1]``:
#   * ``phi ~ -1`` where the closest cell is of type A
#   * ``phi ~ +1`` where the closest cell is of type B
#   * smooth sigmoid-like transition across the Voronoi frontier between
#     the two cell types, with width ``transition_width``.
#
# The Cahn-Hilliard energy ``E_AB`` is then evaluated on this landscape on
# both the ground-truth and the predicted positions. Per-pair loss:
#
#     L_AB = 1 - exp( -|E_pred - E_gt| / (|E_gt| + eps) )
#
# This mirrors the bounded, scale-invariant form already used by
# :class:`CahnHilliardEnergyCurveLoss` (denominator is detached so the
# normalisation does not leak gradients). The sample loss is the mean over
# pairs; the batch loss is the mean over samples.
#
# Performance notes (this is where the speed-up vs. a naive per-pair
# implementation comes from):
#   * The bbox / shared grid is built **once per sample**, not per pair.
#   * For each cell type ``t`` the field ``dist_t`` is computed **once**
#     per side (GT/pred) and reused across all pairs that involve ``t``.
#     The cost therefore scales as ``O(K * G * N_t)`` (over all types) and
#     the per-pair work is just a tanh + a CH energy eval, both
#     ``O(G)``. With K cell types we save a factor ``~K`` over the naive
#     "recompute everything per pair" approach.
#   * GT side runs under ``torch.no_grad`` to skip autograd graph
#     construction; only the predicted side carries gradients back to
#     ``pred_positions``.
#   * Distance evaluation chunks the grid axis of ``cdist`` to keep peak
#     memory bounded for samples with many cells.
#
# Differentiability:
#   * ``soft_beta=None`` -> hard ``min``; sub-differentiable but exact.
#     Gradient flows only through the closest cell (sparse but unbiased).
#   * ``soft_beta>0`` -> ``-logsumexp(-beta * d) / beta``; a smooth
#     soft-min that is differentiable through every contributing cell.
#     Larger ``beta`` -> sharper, closer to the exact min.
# ---------------------------------------------------------------------------


def soft_nearest_distance(
    grid_xy: torch.Tensor,    # [G, 2] -- flattened grid points (x, y)
    cell_xy: torch.Tensor,    # [N, 2] -- cell centers (x, y)
    soft_beta: Optional[float] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Distance from each grid point to the nearest cell.

    Computes ``min_i ||grid_xy[g] - cell_xy[i]||`` (or its soft
    approximation) without materialising the full ``[G, N]`` distance
    matrix, by chunking the ``G`` axis of ``cdist``. Differentiable w.r.t.
    ``cell_xy`` (sparse gradient for the hard ``min``, dense gradient for
    the soft ``min``).

    Args:
        grid_xy: ``[G, 2]`` grid sample locations.
        cell_xy: ``[N, 2]`` cell positions.
        soft_beta: ``None`` for an exact (hard) ``min`` over ``i``;
            a positive float to use the smooth soft-min
            ``-logsumexp(-beta * d) / beta`` (matches min as ``beta -> inf``).
        chunk: number of grid points per ``cdist`` block; controls peak
            memory of the temporary ``[chunk, N]`` distance matrix.

    Returns:
        ``[G]`` tensor of (soft-)nearest distances.
    """
    n_cells = cell_xy.shape[0]
    if n_cells == 0:
        # Convention used downstream: an "empty type" never beats any other
        # type in the (dist_A - dist_B) comparison; +inf is a safe sentinel.
        return torch.full(
            (grid_xy.shape[0],),
            float("inf"),
            device=grid_xy.device,
            dtype=grid_xy.dtype,
        )

    G = grid_xy.shape[0]
    chunk = max(1, int(chunk))
    out_chunks = []
    # Chunk along G (the grid axis) so we never materialise the full
    # [G, N] distance matrix; this keeps memory bounded by O(chunk * N).
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        # [chunk, N] pairwise Euclidean distances.
        d = torch.cdist(grid_xy[s:e], cell_xy, p=2)
        if soft_beta is None:
            out_chunks.append(d.min(dim=-1).values)
        else:
            beta = float(soft_beta)
            # softmin(d_1, ..., d_N) = -logsumexp(-beta*d_i) / beta.
            # As beta -> inf this approaches min_i d_i. logsumexp is stable.
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
    """Per-cell-type "distance to nearest cell of that type" fields.

    For each entry ``(type_id, mask)`` of ``type_masks`` we compute a
    ``[ny, nx]`` field giving, at every grid location, the (soft-)distance
    to the nearest cell of that type. These fields are the building block
    of :func:`voronoi_phase_field`: they are computed **once per type**
    per side (GT/pred), and any pair ``(A, B)`` is then constructed in
    O(G) time as ``tanh((dist_A - dist_B) / w)``.

    Empty types (no cells passing the mask) are dropped from the output.
    """
    ny, nx = grid_y.shape[0], grid_x.shape[0]
    # Build the [G, 2] flattened grid once and reuse across types. Indexing
    # convention matches `cahn_hilliard_energy_density`: dim 0 = y, dim 1 = x.
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

    ``phi(x) = tanh((dist_neg(x) - dist_pos(x)) / transition_width)`` -- the
    same formula as ``build_voronoi_phase_landscape_from_cell_types`` from
    ``cahn_hilliard.py``. ``tanh`` already returns in ``(-1, 1)`` by
    construction, so no further clamp is applied (one fewer kernel).
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
    """Per-type "(soft-)distance to nearest cell of that type" fields.

    Two fused changes compared to :func:`compute_distance_field_per_type`:

    1. **Single pass over all cells** (instead of one ``cdist`` per type):
       we compute the squared-distance matrix ``[Gc, N]`` once per grid
       chunk, then reduce per-type via ``scatter_reduce`` along the cell
       axis. With ``T`` types this trades ``T`` cdist calls for one
       matmul + one scatter, removing ``T-1`` full grid passes.
    2. **Matmul expansion of the squared distance**:
       ``||g - c||^2 = ||g||^2 + ||c||^2 - 2 g.c``. The dot product is
       a plain ``matmul`` (cuBLAS / tensor cores), much faster than
       ``cdist(p=2)`` whose general kernel doesn't take advantage of
       tensor-core matmul. The ``sqrt`` is then applied only on the
       reduced ``[Gc, T]`` tensor for the hard-min branch -- one
       ``sqrt`` over ``Gc * T`` values instead of ``Gc * N``.

    Soft-min branch: we still need ``sqrt(d2)`` over the full ``[Gc, N]``
    tensor (because ``-logsumexp(-beta * d)/beta`` is defined on
    distances, not squared distances), but the squared-distance matmul
    itself is still cheaper than ``cdist``.

    Args:
        grid_xy: ``[G, 2]`` flattened grid points.
        grid_norm_sq: ``[G]`` precomputed ``(grid_xy ** 2).sum(-1)``;
            cached to avoid recomputing each step.
        cells_xy: ``[N, 2]`` positions of cells across **all** valid
            types (concatenated -- not one tensor per type).
        cell_type_idx: ``[N]`` integer in ``[0, num_types)`` indicating
            which valid-type each cell belongs to.
        num_types: ``T``, the number of valid types.
        soft_beta: ``None`` for hard min (sqrt-at-the-end), positive
            float for the smooth ``-logsumexp(-beta * d)/beta`` softmin.
        chunk: how many grid rows to process at once. Controls the peak
            ``[chunk, N]`` working-set memory.

    Returns:
        ``[T, G]`` tensor of (soft-)distances. The caller reshapes to
        ``[T, ny, nx]`` outside.
    """
    G = grid_xy.shape[0]
    N = cells_xy.shape[0]
    device = grid_xy.device
    dtype = grid_xy.dtype

    if N == 0 or num_types == 0:
        # No cells of any valid type -> +inf everywhere (sentinel used
        # downstream to mean "this type loses every distance comparison").
        return torch.full((max(num_types, 1), G), float("inf"),
                          device=device, dtype=dtype)

    cell_norm_sq = (cells_xy * cells_xy).sum(-1)  # [N]
    out = torch.empty((num_types, G), device=device, dtype=dtype)

    chunk = max(1, int(chunk))
    # Index tensor used by the scatter_reduce/scatter_add ops; stays the
    # same across chunks (only the row dim Gc changes).
    type_idx_row = cell_type_idx.unsqueeze(0)  # [1, N]

    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        Gc = e - s

        g = grid_xy[s:e]                         # [Gc, 2]
        gn2 = grid_norm_sq[s:e].unsqueeze(-1)    # [Gc, 1]

        # Squared-distance matrix via matmul expansion.
        dot = g @ cells_xy.t()                   # [Gc, N]  (tensor cores)
        d2 = gn2 + cell_norm_sq.unsqueeze(0) - 2.0 * dot
        # Float32 roundoff in the expansion can push d2 slightly below
        # zero (or to exactly zero) when the actual distance is tiny.
        # Clamping at a small *positive* epsilon keeps both forward and
        # backward well-behaved: the forward bias is at most ``sqrt(eps)``
        # (~1e-6 in coord units, far below any meaningful spatial scale),
        # but ``sqrt`` then has a finite derivative everywhere. This is
        # the safety net that ``torch.cdist`` provides internally and
        # that a plain ``matmul`` does not -- without it, a cell sitting
        # exactly on a grid point makes ``d/dx sqrt(x)|_0 = inf`` and
        # NaNs the backward.
        eps_d2 = 1e-12
        d2 = d2.clamp_min(eps_d2)

        type_idx_chunk = type_idx_row.expand(Gc, N)

        if soft_beta is None:
            # Hard min: scatter-amin over the cell axis, grouped by type.
            min_d2 = torch.full(
                (Gc, num_types), float("inf"),
                device=device, dtype=dtype,
            )
            min_d2.scatter_reduce_(
                dim=1, index=type_idx_chunk, src=d2,
                reduce="amin", include_self=False,
            )
            # One sqrt on the reduced tensor [Gc, T] (instead of [Gc, N]).
            out[:, s:e] = min_d2.sqrt().t()
        else:
            beta = float(soft_beta)
            # -logsumexp trick: subtract per-type max for numerical stability.
            d = d2.sqrt()                        # [Gc, N]
            neg_beta_d = -beta * d               # [Gc, N]
            # Per-(grid, type) max of -beta*d.
            max_per_t = torch.full(
                (Gc, num_types), float("-inf"),
                device=device, dtype=dtype,
            )
            max_per_t.scatter_reduce_(
                dim=1, index=type_idx_chunk, src=neg_beta_d,
                reduce="amax", include_self=False,
            )
            # Broadcast each cell's per-type max back to its position.
            shifted = neg_beta_d - max_per_t.gather(1, type_idx_chunk)
            # Sum exp(shifted) per type.
            sums = torch.zeros(
                (Gc, num_types), device=device, dtype=dtype,
            )
            sums.scatter_add_(
                dim=1, index=type_idx_chunk, src=shifted.exp(),
            )
            # log-sum-exp: max + log(sum). clamp_min guards types with
            # zero cells in this batch (sums == 0 -> log(0) = -inf,
            # which yields softmin = +inf -- the correct sentinel).
            lse = max_per_t + sums.clamp_min(1e-30).log()
            out[:, s:e] = (-lse / beta).t()

    return out  # [T, G]


class VoronoiPhasePairEnergyLoss(nn.Module):
    """Cahn-Hilliard energy loss over Voronoi phase-separation landscapes
    of all unordered pairs of cell types.

    For every sample in the batch:
      1. Identify the cell types present in the (masked) sample with at
         least ``min_cells_per_type`` cells.
      2. Build a single shared grid covering the union bbox of all those
         cells (GT and pred), optionally squarified.
      3. Compute, **once per type**, the distance fields
            ``dist_t(x) = (soft-)distance to nearest cell of type t``
         on that grid -- once for the GT positions and once for the
         predicted ones.
      4. For each unordered pair ``(A, B)`` of those types, build
            ``phi_AB = tanh((dist_A - dist_B) / transition_width)``
         (cheap O(G) operation, reuses the precomputed dist fields), and
         evaluate the Cahn-Hilliard energy
            ``E_AB = sum_ij [(phi^2 - 1)^2 + kappa * |grad phi|^2] dx dy``.
      5. Accumulate the per-pair loss
            ``L_AB = 1 - exp(-|E_pred - E_gt| / (|E_gt| + eps))``,
         then average over pairs (sample loss). Batch loss is the mean
         over samples.

    Note on symmetry: swapping ``(A, B) -> (B, A)`` flips the sign of
    ``phi_AB``; the CH energy is even in ``phi`` (well: ``(c^2-1)^2``;
    gradient term: ``|grad c|^2``), so ``E_BA = E_AB`` and we only
    iterate over unordered pairs.

    Args:
        transition_width: width of the Voronoi frontier transition (coord
            units). Smaller -> sharper boundary; larger -> broader.
        grid_resolution: number of grid points per axis (``nx = ny``).
        kappa: Cahn-Hilliard gradient-energy coefficient.
        soft_beta: ``None`` for an exact ``min`` (sparse gradient through
            the closest cell only, matches the reference numerics) or a
            positive float for a smooth soft-min (dense, fully
            differentiable through every cell). Recommended for training.
        square_bbox: build the shared grid on the square bbox of GT u pred
            so that x/y are on the same physical scale (matches the
            reference helper).
        margin: fractional padding around the (square) bbox per axis.
        eps: numerical floor for divisions.
        min_cells_per_type: a type contributes only if it has this many
            cells in the masked sample.
        chunk: ``cdist`` chunk size along the grid axis (controls peak
            memory of the per-type distance computation).
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
        # Visualization: every N epochs (> 0 enables; 0 disables) we
        # detach a small slice of the loss's intermediate tensors,
        # render them to a matplotlib figure and push it to WandB so
        # you can verify the loss is doing what you think it's doing.
        # The work happens only on ``batch_idx == 0`` of the chosen
        # epoch, so it costs at most one extra forward of the loss
        # every N epochs.
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        self.viz_num_figures = int(viz_num_figures)
        # Tracked by the training loop via ``set_current_epoch`` and
        # gated by ``viz_every_n_epochs``. Default 0 means "no viz
        # until the trainer tells us otherwise".
        self.current_epoch: int = 0
        # When ``cache_gt`` is enabled, the grid is built from GT cells
        # *only* (instead of the GT u pred union), which makes the grid --
        # and therefore the GT distance fields and pair energies -- depend
        # on the input batch only, never on the model's predictions. This
        # is what makes caching across training steps correct.
        self.cache_gt = bool(cache_gt)
        # Two ways to key the cache:
        #   * "content" (default, shuffle-safe): hash the cell_ID tensor
        #     of the sample. The same chunk of cells maps to the same
        #     cache entry regardless of which (batch_idx, sample_idx)
        #     slot it lands in -- so the cache survives DataLoader
        #     ``shuffle=True``, which is the standard training setup.
        #   * "slot": use ``(batch_idx, sample_idx)``. Faster lookup but
        #     only useful when sample-to-batch assignment is stable
        #     across epochs (no shuffling, no rechunking). Kept as an
        #     escape hatch; not recommended for typical training.
        if cache_key_mode not in ("content", "slot"):
            raise ValueError(
                f"cache_key_mode must be 'content' or 'slot', got {cache_key_mode!r}."
            )
        self.cache_key_mode = cache_key_mode
        # Maps cache_key -> dict with the cached GT data:
        #   {"cell_id", "grid_x", "grid_y", "dx", "dy",
        #    "valid_types", "type_masks", "pair_energies"}
        # The key type depends on ``cache_key_mode``: ``bytes`` for
        # "content", ``Tuple[int, int]`` for "slot". Tensors are kept on
        # the device they were computed on (typically CUDA); per-entry
        # footprint is small (a 1D grid of ``grid_resolution`` floats
        # per axis, a few ``[N]`` boolean type masks, and one scalar per
        # cell-type pair).
        self._gt_cache: Dict = {}

    def clear_gt_cache(self) -> None:
        """Drop all cached GT data.

        Call after an event that changes the GT cells assigned to a given
        ``(batch_idx, sample_idx)`` slot -- e.g. dataset rechunking or
        re-shuffling. Note that the cache is *also* automatically
        invalidated per-entry when the ``cell_ID`` fingerprint of a sample
        changes, so calling this manually is usually only a safety net.
        """
        self._gt_cache.clear()

    def set_current_epoch(self, epoch: int) -> None:
        """Track the trainer's current epoch so viz can be gated."""
        self.current_epoch = int(epoch)

    def _should_log_viz(self, batch_idx: Optional[int]) -> bool:
        """True iff this is a viz-logging step.

        Triggers on ``batch_idx == 0`` of any epoch divisible by
        ``viz_every_n_epochs`` (and only when viz is enabled). Restricting
        to ``batch_idx == 0`` makes "every N epochs" mean exactly that --
        one set of figures per epoch, regardless of the number of batches.
        """
        if self.viz_every_n_epochs <= 0:
            return False
        if batch_idx is None or int(batch_idx) != 0:
            return False
        return (self.current_epoch % self.viz_every_n_epochs) == 0

    def _compute_pair_viz_data(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        mask: torch.Tensor,
        cell_class_b: torch.Tensor,
        sample_idx: int,
        rng: np.random.Generator,
    ) -> Optional[Dict]:
        """Rebuild phi_pred / phi_gt / energy densities for one random
        (sample, type pair). Returns ``None`` if the sample has < 2 valid
        types. Runs under ``torch.no_grad`` -- viz must not influence the
        training graph.
        """
        device = pred_pos.device

        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        if cell_class_b.dim() == 2 and cell_class_b.shape[-1] == 1:
            cell_class_b = cell_class_b.squeeze(-1)
        if mask_b.sum() == 0:
            return None

        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        masked_classes = cell_class_b[mask_b]
        masked_classes = masked_classes[masked_classes >= 0]
        if masked_classes.numel() == 0:
            return None

        uniq, counts = torch.unique(masked_classes, return_counts=True)
        keep = counts >= self.min_cells_per_type
        valid_types_t = uniq[keep]
        num_types = int(valid_types_t.numel())
        if num_types < 2:
            return None

        # Pick a random *valid* pair of types.
        pair = rng.choice(num_types, size=2, replace=False)
        i_a, i_b = int(min(pair)), int(max(pair))
        ta = int(valid_types_t[i_a].item())
        tb = int(valid_types_t[i_b].item())

        participates = mask_b & torch.isin(cell_class_b, valid_types_t)
        cell_type_idx = torch.searchsorted(valid_types_t, cell_class_b[participates])

        # Same grid the loss itself uses (bbox over *all* real GT cells).
        gt_bbox_pts = true_xy[mask_b]
        grid_x, grid_y, dx, dy = shared_square_grid(
            gt_bbox_pts, gt_bbox_pts,
            grid_resolution=self.grid_resolution,
            margin=self.margin, square=self.square_bbox,
        )
        ny, nx = grid_y.shape[0], grid_x.shape[0]
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        grid_norm_sq = (grid_xy * grid_xy).sum(-1)

        with torch.no_grad():
            gt_cells = true_xy[participates]
            pred_cells = pred_xy[participates]
            gt_dists = compute_distance_fields_per_type_fused(
                grid_xy=grid_xy, grid_norm_sq=grid_norm_sq,
                cells_xy=gt_cells, cell_type_idx=cell_type_idx,
                num_types=num_types, soft_beta=self.soft_beta,
                chunk=self.chunk,
            ).view(num_types, ny, nx)
            pred_dists = compute_distance_fields_per_type_fused(
                grid_xy=grid_xy, grid_norm_sq=grid_norm_sq,
                cells_xy=pred_cells, cell_type_idx=cell_type_idx,
                num_types=num_types, soft_beta=self.soft_beta,
                chunk=self.chunk,
            ).view(num_types, ny, nx)

            phi_gt = voronoi_phase_field(
                gt_dists[i_a], gt_dists[i_b], self.transition_width
            )
            phi_pred = voronoi_phase_field(
                pred_dists[i_a], pred_dists[i_b], self.transition_width
            )
            ed_gt = cahn_hilliard_energy_density(phi_gt, dx, dy, self.kappa)
            ed_pred = cahn_hilliard_energy_density(phi_pred, dx, dy, self.kappa)
            e_gt = float((ed_gt.sum() * dx * dy).item())
            e_pred = float((ed_pred.sum() * dx * dy).item())

        # Squeeze tensors to numpy on CPU. ``detach`` is redundant under
        # ``no_grad`` but cheap and makes intent explicit.
        return {
            "phi_pred": phi_pred.detach().cpu().numpy(),
            "phi_gt": phi_gt.detach().cpu().numpy(),
            "ed_pred": ed_pred.detach().cpu().numpy(),
            "ed_gt": ed_gt.detach().cpu().numpy(),
            "extent": (
                float(grid_x[0].item()), float(grid_x[-1].item()),
                float(grid_y[0].item()), float(grid_y[-1].item()),
            ),
            "ta": ta, "tb": tb,
            "e_pred": e_pred, "e_gt": e_gt,
            "sample_idx": int(sample_idx),
        }

    def _log_viz_to_wandb(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        node_mask: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> None:
        """Render ``viz_num_figures`` random (sample, pair) panels and
        push them to WandB. No-op if WandB isn't initialised.
        """
        if not wandb.run:
            return
        try:
            import matplotlib  # noqa: F401  (figure builder will import)
        except ImportError:
            return  # matplotlib not installed -> silently skip viz

        B = int(pred_positions.shape[0])
        # Seed by epoch so multiple runs that line up step-for-step
        # produce comparable diagnostic panels.
        rng = np.random.default_rng(seed=int(self.current_epoch))

        log_dict: Dict[str, "wandb.Image"] = {}
        # We may not get a successful viz on every try (e.g. a sample
        # without two valid types), so we bound the number of attempts.
        max_attempts = max(self.viz_num_figures * 4, 8)
        figs_logged = 0
        attempt = 0
        while figs_logged < self.viz_num_figures and attempt < max_attempts:
            attempt += 1
            b = int(rng.integers(0, B))
            viz = self._compute_pair_viz_data(
                pred_pos=pred_positions[b],
                true_pos=true_positions[b],
                mask=node_mask[b],
                cell_class_b=cell_class[b],
                sample_idx=b,
                rng=rng,
            )
            if viz is None:
                continue
            title = (
                f"Voronoi pair viz | epoch {self.current_epoch} | "
                f"sample {viz['sample_idx']} | types ({viz['ta']}, {viz['tb']})"
            )
            fig = _build_phi_energy_figure(
                phi_pred=viz["phi_pred"], phi_gt=viz["phi_gt"],
                ed_pred=viz["ed_pred"], ed_gt=viz["ed_gt"],
                extent=viz["extent"],
                e_pred=viz["e_pred"], e_gt=viz["e_gt"],
                title=title,
            )
            # Fixed-key images so WandB groups them as a time series
            # rather than creating a new key per epoch.
            log_dict[f"viz/voronoi/fig_{figs_logged}"] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figs_logged += 1

        if log_dict:
            wandb.log(log_dict, commit=False)

    # ---------------- internals ----------------

    def _normalized_exp_diff(
        self,
        e_pred: torch.Tensor,
        e_gt: torch.Tensor,
    ) -> torch.Tensor:
        """``1 - exp(-|e_pred - e_gt| / |e_gt|)``; denominator is detached.

        Same bounded, scale-invariant form as
        :class:`CahnHilliardEnergyCurveLoss._normalized_exp_diff` -- so
        per-pair contributions remain comparable across pairs with very
        different absolute energies, and the loss is bounded in ``[0, 1)``.
        """
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

        # Only the spatial dims drive the landscape.
        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        # ---- (Optional) cache lookup ---------------------------------
        # Try to reuse the GT-side data computed on a previous call.
        # The key depends on ``cache_key_mode``:
        #   * "content": hash of the sample's ``cell_id`` -> survives
        #     DataLoader shuffling, because the same chunk of cells maps
        #     to the same key regardless of where it lands in the epoch.
        #   * "slot": ``(batch_idx, sample_idx)`` -> only correct when
        #     sample-to-slot is stable (no shuffling, no rechunking).
        # In both modes ``cell_id`` is *also* stored as a fingerprint and
        # checked on hit, so a stale entry is rebuilt rather than reused.
        cache_key = None
        if self.cache_gt:
            if self.cache_key_mode == "content" and cell_id is not None:
                # Hash via tobytes(): O(N) and fully deterministic.
                # ``int64`` cast guards against future dtype drift (the
                # bytes representation must be stable for the same IDs).
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
                # Detach + same-device compare; cheap for typical sample
                # sizes (a few thousand ints). Acts as a defensive
                # fingerprint check (collision guard for "content" mode,
                # rechunk guard for "slot" mode).
                if cached_cid is None or cached_cid.shape != cell_id.shape \
                        or not torch.equal(cached_cid, cell_id.detach()):
                    cached = None  # stale entry: rebuild below

        # ---- Compute the GT side (only if not cached) ----------------
        if cached is None:
            # 1) Identify valid types in a single fused pass (avenue 11):
            # one ``torch.unique(..., return_counts=True)`` replaces the
            # old per-type Python loop with its many ``.item()`` syncs.
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
            valid_types_t = uniq[keep]              # sorted, [T]
            num_types = int(valid_types_t.numel())  # one sync
            if num_types < 2:
                if cache_key is not None:
                    self._gt_cache[cache_key] = {
                        "cell_id": cell_id.detach() if cell_id is not None else None,
                        "valid_types": [],
                    }
                return torch.zeros((), device=device, dtype=dtype), 0

            # ``participates[c]`` <=> cell c is masked AND in a valid type.
            participates = mask_b & torch.isin(cell_class, valid_types_t)
            classes_kept = cell_class[participates]
            # ``valid_types_t`` is sorted (output of ``torch.unique``), so
            # ``searchsorted`` gives each cell its valid-type index in [0, T).
            cell_type_idx = torch.searchsorted(valid_types_t, classes_kept)

            # Single ``tolist()`` so the cache stores Python-friendly type
            # ids too (one sync, used only for logging/introspection).
            valid_types = valid_types_t.tolist()

            # 2) GT-only grid (decoupled from pred so the cache stays valid).
            # The bbox spans **all real GT cells** of the sample (every
            # cell with ``mask_b == True``), regardless of whether their
            # type passed the ``min_cells_per_type`` filter. This makes
            # the grid extent independent of the type-validity threshold:
            # the same square is used for every (a, b) pair on this
            # sample, and for both GT and pred sides. Cells of
            # non-valid types still don't contribute to any distance
            # field (see ``cell_type_idx`` below) -- they only widen the
            # bbox so the grid covers the actual spatial extent of the
            # sample.
            gt_bbox_pts = true_xy[mask_b]
            grid_x, grid_y, dx, dy = shared_square_grid(
                gt_bbox_pts, gt_bbox_pts,
                grid_resolution=self.grid_resolution,
                margin=self.margin,
                square=self.square_bbox,
            )

            # 3) Pre-build the flat grid + its squared norms once and cache
            # them (avenue 9). These are reused on every pred step.
            ny, nx = grid_y.shape[0], grid_x.shape[0]
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [G, 2]
            grid_norm_sq = (grid_xy * grid_xy).sum(-1)                   # [G]

            # 4) Pair indices, computed once on the (cached) grid.
            pair_idx = torch.triu_indices(
                num_types, num_types, offset=1, device=device,
            )  # [2, P]
            pair_idx_a = pair_idx[0]
            pair_idx_b = pair_idx[1]

            # 5) GT distance fields + GT pair energies (avenues 1+2+3),
            # all under no_grad: the target side carries no autograd graph.
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

                # Stack the pair fields in one shot, then the CH energy
                # is a single fused kernel returning [P] energies. This
                # replaces the Python (i, j) loop entirely.
                phi_gt = voronoi_phase_field(
                    gt_dists[pair_idx_a],
                    gt_dists[pair_idx_b],
                    self.transition_width,
                )  # [P, ny, nx]
                e_gt_vec = cahn_hilliard_energy(
                    phi_gt, dx, dy, self.kappa,
                )  # [P]

                # Free the [T, ny, nx] GT distance fields and the [P,
                # ny, nx] phi tensor before pred-side work allocates.
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

        # ---- Use cached GT data --------------------------------------
        if len(cached.get("valid_types", [])) < 2:
            return torch.zeros((), device=device, dtype=dtype), 0

        # Bind locals for clarity / avoiding repeated dict lookups in the
        # hot path.
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

        # ---- Pred side: fused distance fields + stacked pair energies
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
        )  # [P, ny, nx]
        e_pred_vec = cahn_hilliard_energy(phi_pred, dx, dy, self.kappa)  # [P]

        # Per-pair normalised loss; one element-wise call over [P].
        loss_per_pair = self._normalized_exp_diff(e_pred_vec, e_gt_vec)
        n_pairs = int(pair_idx_a.numel())

        if n_pairs == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        return loss_per_pair.sum() / float(n_pairs), n_pairs

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        cell_class: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute the Voronoi phase-pair CH energy loss.

        Args:
            masked_pred: DataHolder with predicted ``.positions`` of shape
                ``[B, N, >=2]`` and ``.node_mask`` of shape ``[B, N]``.
            masked_true: DataHolder with ground-truth ``.positions`` and
                ``.cell_class`` of shape ``[B, N]`` (integer ids).
            train_stage: True for training logs, False for validation logs.
            log: whether to emit a WandB log line.
            cell_class: optional override for the per-cell integer class
                tensor, shape ``[B, N]``.
            batch_idx: index of the current batch within the epoch. When
                provided together with ``cache_gt=True`` (set in
                ``__init__``), GT-side data are cached per
                ``(batch_idx, sample_idx)`` and reused on subsequent
                calls -- skipping the dominant cost (GT distance fields
                + GT pair energies). Pass ``None`` (the default) to
                disable caching for this call (e.g. validation).
        """
        if cell_class is None:
            cell_class = masked_true.cell_class
        if cell_class is None:
            raise ValueError(
                "VoronoiPhasePairEnergyLoss requires cell_class; none was provided."
            )

        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask

        # Optional fingerprint used to invalidate cache entries when the
        # GT cells change (e.g. dataset rechunking). ``cell_ID`` may be
        # ``[B, N]`` or ``[B, N, 1]`` depending on the data path; we
        # squeeze the trailing dim for the per-sample compare.
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

        # Batch loss = mean over samples (samples with no valid pair
        # contribute a 0 from _sample_loss, which is reasonable since they
        # carry no signal -- they're effectively a no-op for this loss).
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
            if wandb.run:
                wandb.log(to_log, commit=True)

        # Periodic visualization (training only, batch_idx == 0 of an
        # epoch divisible by ``viz_every_n_epochs``). Push a small
        # number of (sample, pair) landscape + energy-density figures
        # to WandB so you can eyeball whether the loss is computing
        # what you think it's computing.
        if train_stage and self._should_log_viz(batch_idx):
            self._log_viz_to_wandb(
                pred_positions=pred_positions,
                true_positions=true_positions,
                node_mask=node_mask,
                cell_class=cell_class,
            )

        return loss, to_log

    def reset(self) -> None:
        """No running state to reset."""
        pass


# ---------------------------------------------------------------------------
# Neighborhood-averaged transcriptome RMSE loss
# ---------------------------------------------------------------------------
#
# For every cell ``i`` we collect the cells ``j`` that lie within ``radius``
# of ``i`` (Euclidean distance on the spatial coordinates) and average their
# transcriptomic profile to obtain ``avg(i)``.  The loss is the RMSE between
# the predicted-side and the GT-side averages, then averaged over cells.
#
# The transcriptomic features themselves are GT data: the model only
# predicts positions, so the predicted-side average reuses the same
# ``node_features`` but uses the predicted positions to define the
# neighborhood.
#
# The GT-side averages do not depend on the model and can be computed once
# (cf. :func:`precompute_gt_neighborhood_features`) and looked up at every
# step instead of being recomputed.
# ---------------------------------------------------------------------------


def compute_neighborhood_average_features(
    positions: torch.Tensor,        # [B, N, D]
    features: torch.Tensor,         # [B, N, F]
    mask: torch.Tensor,             # [B, N] (bool / 0-1)
    radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
    include_self: bool = True,
) -> torch.Tensor:
    """Vectorised per-cell neighborhood average of ``features``.

    For each cell ``i``::

        avg(i) = (sum_j w_ij * features[j]) / (sum_j w_ij)

    where the sum runs over valid cells ``j`` (``mask[j] == 1``) and:

      * ``w_ij = 1`` if ``dist(i, j) <= radius``, else ``0``  (default,
        ``soft_beta=None``)  -- exact membership but **not differentiable**
        w.r.t. the positions, since the mask is boolean.
      * ``w_ij = sigmoid(soft_beta * (radius - dist(i, j)))``  -- a smooth
        approximation of the step at ``dist == radius`` that is fully
        differentiable.  Use this for the predicted side if you want the
        loss to back-propagate through ``positions``.

    The full computation is one ``cdist`` + one batched matmul:
        ``[B, N, N] @ [B, N, F] = [B, N, F]``
    which keeps the per-cell loop completely on the GPU.

    Padding rows (``mask==0``) get zero contribution as neighbors and the
    output rows for padded cells are zeroed out.
    """
    # [B, N, N] pairwise Euclidean distances. Same shape regardless of D.
    dists = torch.cdist(positions, positions, p=2)

    if soft_beta is None:
        # Hard cutoff: boolean adjacency cast to float.
        weights = (dists <= float(radius)).to(features.dtype)
    else:
        # Soft cutoff that is smooth around the boundary -> differentiable.
        weights = torch.sigmoid(float(soft_beta) * (float(radius) - dists))

    # Exclude invalid neighbors j: mask[:, None, :] broadcasts along the i axis.
    valid_j = mask.to(weights.dtype).unsqueeze(1)
    weights = weights * valid_j

    if not include_self:
        n = positions.shape[-2]
        eye = torch.eye(n, device=positions.device, dtype=weights.dtype)
        weights = weights * (1.0 - eye)

    # Single batched matmul replaces a per-cell python loop.
    sum_features = torch.matmul(weights, features)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    avg = sum_features / denom

    # Zero out rows for padded cells i so they cannot contaminate later means.
    valid_i = mask.to(avg.dtype).unsqueeze(-1)
    return avg * valid_i


def precompute_gt_neighborhood_features(
    true_positions: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
    radius: float,
    *,
    save_path: Optional[str] = None,
    soft_beta: Optional[float] = None,
    include_self: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """One-shot helper to compute the GT neighborhood-averaged transcriptome.

    Returns a tensor of shape ``[B, N, F]``. If ``save_path`` is given, the
    detached tensor is written to disk via ``torch.save`` so it can be
    reloaded across runs without recomputation.

    The GT side does not need gradients, so the computation runs under
    ``torch.no_grad`` to avoid building an autograd graph.
    """
    with torch.no_grad():
        avg = compute_neighborhood_average_features(
            true_positions, node_features, node_mask,
            radius=radius,
            soft_beta=soft_beta,
            eps=eps,
            include_self=include_self,
        )
    if save_path is not None:
        torch.save(avg.detach().cpu(), save_path)
    return avg


class NeighborhoodTranscriptomeRMSELoss(nn.Module):
    """RMSE between predicted and GT neighborhood-averaged transcriptomes.

    Pipeline:
      1. For each cell ``i`` and side (pred / GT), compute the average
         transcriptomic profile of cells ``j`` within ``radius`` of ``i``
         (cf. :func:`compute_neighborhood_average_features`).
      2. ``per_cell_rmse(i) = sqrt(mean_f (avg_pred(i, f) - avg_gt(i, f))^2)``.
      3. ``loss = mean_i per_cell_rmse(i)``  over valid (unmasked) cells.

    The GT-side average is independent of the model. Either pass a
    precomputed tensor as ``cached_gt_avg`` (recommended; see
    :func:`precompute_gt_neighborhood_features`) or let the loss recompute
    it under ``no_grad`` on every call.

    Differentiability:
      * ``soft_beta=None`` (default) uses a hard cutoff. The loss is
        well-defined but its gradient w.r.t. predicted positions is zero
        almost everywhere (the neighborhood is boolean).  Use this only as
        a diagnostic / metric.
      * ``soft_beta>0`` uses a sigmoid-soft membership and is differentiable
        through the predicted positions; recommended for training.
    """

    def __init__(
        self,
        radius: float,
        soft_beta: Optional[float] = None,
        eps: float = 1e-6,
        include_self: bool = True,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.eps = float(eps)
        self.include_self = bool(include_self)
        # Cached scalar of the last forward pass; consumed by
        # :meth:`log_epoch_metrics` so the trainer can emit a single
        # ``train_epoch/...`` key that PyTorch-Lightning will aggregate
        # across the epoch.
        self._last_loss: float = -1.0

    # ---------------- GT precompute / caching ----------------

    def precompute_gt(
        self,
        true_positions: torch.Tensor,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Convenience wrapper around :func:`precompute_gt_neighborhood_features`."""
        return precompute_gt_neighborhood_features(
            true_positions, node_features, node_mask,
            radius=self.radius,
            save_path=save_path,
            soft_beta=self.soft_beta,
            include_self=self.include_self,
            eps=self.eps,
        )

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        cached_gt_avg: Optional[torch.Tensor] = None,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,  # accept (and ignore) ``batch_idx`` etc. so this
        # loss is drop-in compatible with the richer Combined/CH APIs the
        # training loop dispatches to.
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        node_mask = masked_true.node_mask
        node_features = masked_true.node_features
        true_xy = masked_true.positions[..., :2]
        pred_xy = masked_pred.positions[..., :2]

        # GT side: either look up the cached average or recompute under
        # no_grad. We never want autograd state on the GT.
        if cached_gt_avg is None:
            with torch.no_grad():
                gt_avg = compute_neighborhood_average_features(
                    true_xy, node_features, node_mask,
                    radius=self.radius,
                    soft_beta=self.soft_beta,
                    eps=self.eps,
                    include_self=self.include_self,
                )
        else:
            gt_avg = cached_gt_avg.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )

        # Predicted side: features are GT data, neighborhoods come from the
        # predicted positions. Same one-matmul vectorised computation.
        pred_avg = compute_neighborhood_average_features(
            pred_xy, node_features, node_mask,
            radius=self.radius,
            soft_beta=self.soft_beta,
            eps=self.eps,
            include_self=self.include_self,
        )

        # Per-cell RMSE over the feature dimension.
        sq_err = (pred_avg - gt_avg).pow(2)
        per_cell_mse = sq_err.mean(dim=-1)
        per_cell_rmse = torch.sqrt(per_cell_mse + self.eps)

        # Mean over valid (unmasked) cells only.
        valid_i = node_mask.to(per_cell_rmse.dtype)
        n_valid = valid_i.sum().clamp_min(1.0)
        loss = (per_cell_rmse * valid_i).sum() / n_valid

        # Always update the cached scalar (one CPU sync per step). This
        # is what :meth:`log_epoch_metrics` and the epoch-end print in
        # ``utils/diffusion_model/train/train.py`` consume.
        loss_val = float(loss.detach().item())
        if train_stage:
            self._last_loss = loss_val

        to_log: Optional[Dict[str, float]] = None
        if log:
            key = (
                "train_loss/neighborhood_transcriptome_rmse"
                if train_stage
                else "val_loss/neighborhood_transcriptome_rmse"
            )
            to_log = {key: loss_val}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        """No running state to reset (the last-loss cache is intentionally
        carried across the epoch boundary; it is overwritten by the next
        ``forward`` call)."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Expose the last-step loss under a ``train_epoch/...`` key.

        Returned every step from ``training_step_func`` and forwarded to
        ``self.log_dict(..., on_step=False, on_epoch=True)``; Lightning
        therefore averages the per-step values into a single
        end-of-epoch metric that downstream code (epoch-end print, early
        stopping, checkpointing) can consume.
        """
        to_log = {
            "train_epoch/neighborhood_transcriptome_rmse": float(self._last_loss),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


# ---------------------------------------------------------------------------
# Multi-radius neighborhood loss (transcriptome RMSE + log-density diff)
# ---------------------------------------------------------------------------
#
# Generalisation of :class:`NeighborhoodTranscriptomeRMSELoss` along two
# axes:
#
#   1. Radii are sorted ascending; each index defines an **annulus** (ring),
#      not a cumulative ball. For sorted ``[r_0, r_1, ..., r_{R-1}]``::
#
#        shell_0: ball(r_0)
#        shell_k: ball(r_k) \\ ball(r_{k-1})   (mask_k = mask_cum_k - mask_cum_{k-1})
#
#      Transcriptome / density / global-sum terms use only neighbors in
#      that shell, so scales do not double-count smaller neighborhoods.
#   2. Per shell we compare log (soft) cell count and averaged / summed
#      transcriptomes between pred and GT.
#   3. Optionally, pred vs. GT neighborhood *weighted sums* (no / count)
#      in the same shells -- the ``global`` counterpart to the average.
#
# Each shell reuses one cumulative ball mask minus the previous shell's
# cumulative mask. ``cdist`` is once per side; one sigmoid (or cutoff) and
# one matmul per listed radius.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Neighborhood weighted-spatial-covariance loss (trace + det per (cell, gene))
# ---------------------------------------------------------------------------
#
# For every cell ``c`` and every gene ``g`` we build the 2x2 *spatial*
# covariance matrix of the (x, y) positions of cells in the neighborhood
# of ``c``, weighted by ``exp(features[i, g])`` for each neighbor ``i``::
#
#     w_i = exp(F[i, g])
#     S_α = Σ_i a_{c,i} w_i · α_i             (α ∈ {1, x, y, xx, xy, yy})
#     μ_x = S_x / (S_1 + eps),   μ_y = S_y / (S_1 + eps)
#     C_xx = S_xx / (S_1 + eps) - μ_x^2
#     C_yy = S_yy / (S_1 + eps) - μ_y^2
#     C_xy = S_xy / (S_1 + eps) - μ_x μ_y
#     trace(c, g) = C_xx + C_yy
#     det(c, g)   = C_xx · C_yy - C_xy^2
#
# Reasoning behind ``exp(F)`` rather than ``F`` directly:
#   * ``exp`` is strictly positive, so ``S_1 = Σ a w`` cannot vanish for
#     any cell that has at least one (soft) neighbor -- the
#     "no zero-division" property the user asked for.
#   * Multiplicatively, ``exp(F)`` amplifies the contribution of
#     high-expressing cells, so the covariance summarises the spatial
#     spread of *expressing* cells in the neighborhood.
#   * ``exp(F)`` is evaluated in a numerically stable, max-shifted form
#     (subtract the per-(sample, gene) max before exponentiating) so raw
#     un-normalised counts cannot overflow float32 to ``+inf``. Since the
#     covariance is a weighted average, this rescale is mathematically
#     transparent -- see ``compute_neighborhood_weighted_covariance_trace_det``.
#
# Broadcasting / speed trick: instead of six separate matmuls (one per
# moment 1, x, y, xx, xy, yy) we pack the moment weights along a new
# axis and run ONE big ``[B, N, N] @ [B, N, 6F]`` matmul that computes
# all six moments in a single CUDA kernel. ``cdist`` and the membership
# matrix ``A`` are computed once per side.
#
# Numerical clamps: ``C_xx``, ``C_yy`` and ``det`` are mathematically
# non-negative (Cauchy-Schwarz on the weighted samples) but the
# bias-variance identity ``E[X^2] - E[X]^2`` can produce tiny negative
# values from float roundoff. We ``clamp_min(0)`` after the identity to
# protect the downstream ``sqrt`` in the RMSE term.
# ---------------------------------------------------------------------------


def compute_neighborhood_weighted_covariance_trace_det(
    positions: torch.Tensor,        # [B, N, >=2]
    features: torch.Tensor,         # [B, N, F]
    mask: torch.Tensor,             # [B, N]
    radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
    include_self: bool = True,
    cached_dists: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell-per-gene 2x2 spatial-covariance trace + determinant.

    Returns:
        trace: ``[B, N, F]`` -- ``C_xx + C_yy`` for every (cell, gene).
                Rows for padded cells are zeroed out.
        det:   ``[B, N, F]`` -- ``C_xx * C_yy - C_xy^2`` for every
                (cell, gene). Rows for padded cells are zeroed out.

    Args:
        positions: ``[B, N, D]``; only the first two dims (x, y) are used.
        features:  ``[B, N, F]`` per-cell gene-expression matrix.
        mask:      ``[B, N]`` per-cell validity (1 = real cell, 0 = padding).
        radius:    neighborhood radius (position-coord units).
        soft_beta: sigmoid sharpness; ``None`` = hard cutoff (gradient
            through positions is zero almost everywhere -- diagnostic
            only). For training use a positive float (e.g. 256).
        eps: numerical floor for the divisions and the moment denominators.
        include_self: include cell ``c`` itself in its own neighborhood.
        cached_dists: optional precomputed ``[B, N, N]`` distance matrix.
            Skips the ``cdist`` call (useful when the same matrix is
            already needed by another sub-loss).

    Performance:
        Per side: one ``cdist`` ``O(B N^2 D)``, one membership matrix
        ``[B, N, N]``, one fused matmul ``[B, N, N] @ [B, N, 6F]`` whose
        cost is ``6x`` the transcriptome-average matmul. The "six
        moments stacked along an axis" packing means we pay this 6x
        cost as a *single* GPU kernel launch, not six.
    """
    B, N, _ = positions.shape
    F = features.shape[-1]
    dtype = features.dtype
    device = positions.device

    # ---- Neighborhood membership matrix A (shared by all genes).
    if cached_dists is None:
        dists = torch.cdist(positions[..., :2], positions[..., :2], p=2)
    else:
        dists = cached_dists

    if soft_beta is None:
        A = (dists <= float(radius)).to(dtype)
    else:
        A = torch.sigmoid(float(soft_beta) * (float(radius) - dists))

    valid_j = mask.to(dtype).unsqueeze(1)  # [B, 1, N], broadcasts over i
    A = A * valid_j

    if not include_self:
        eye = torch.eye(N, device=device, dtype=dtype)
        A = A * (1.0 - eye)

    # ---- Per-cell-per-gene weights ``w = exp(F)``. ``exp`` is strictly
    # positive, so the moment denominator S_1 cannot vanish when at
    # least one (soft) neighbor exists.
    #
    # Numerical stability: raw gene counts (the real pipeline does NOT
    # log-normalise ``node_features``) can exceed ~88, where ``exp``
    # overflows float32 to ``+inf`` and the weighted means collapse to
    # ``inf/inf = NaN``. We therefore subtract the per-(sample, gene)
    # maximum over valid cells before exponentiating. Because every
    # quantity below is a *weighted average* (numerator and denominator
    # both scale linearly in the weights), multiplying all of a gene's
    # weights by the same positive constant ``exp(-max)`` cancels
    # exactly -- so this rescale leaves trace/det unchanged in the
    # well-behaved regime while guaranteeing the largest weight is
    # ``exp(0) = 1``. The ``clamp_min`` floors the shifted exponent so
    # very-low-expression neighbors cannot underflow to exactly zero,
    # keeping ``S_1`` strictly positive. The same shift is applied to
    # GT and prediction (both use the GT ``node_features``), so the two
    # sides stay directly comparable.
    neg_inf = torch.finfo(features.dtype).min
    valid_feat = features.masked_fill(~mask.bool().unsqueeze(-1), neg_inf)
    f_max = valid_feat.amax(dim=1, keepdim=True)  # [B, 1, F]
    # Guard an all-padding sample (amax would be -inf -> use 0 shift).
    f_max = torch.where(torch.isfinite(f_max), f_max, torch.zeros_like(f_max))
    W = torch.exp((features - f_max).clamp_min(-50.0))  # [B, N, F], in (0, 1]

    # ---- Build the six moment-weighted tensors all at once.
    # Shapes are kept explicit for readability; PyTorch broadcasts
    # ``[B, N, 1]`` * ``[B, N, F]`` into ``[B, N, F]``.
    x = positions[..., 0:1]              # [B, N, 1]
    y = positions[..., 1:2]              # [B, N, 1]
    Wx  = W * x                          # [B, N, F]
    Wy  = W * y
    Wxx = W * (x * x)
    Wxy = W * (x * y)
    Wyy = W * (y * y)

    # Stack along a new "moment" axis and flatten (moment, gene) so we
    # can run ONE batched matmul instead of six. The order here
    # (1, x, y, xx, xy, yy) is preserved when we unpack below.
    V = torch.stack([W, Wx, Wy, Wxx, Wxy, Wyy], dim=2)  # [B, N, 6, F]
    V_flat = V.reshape(B, N, 6 * F)

    # ---- Single fused matmul: A @ V_flat gives all six moments per (c, g).
    S = torch.matmul(A, V_flat).reshape(B, N, 6, F)  # [B, N, 6, F]
    S0  = S[:, :, 0]
    Sx  = S[:, :, 1]
    Sy  = S[:, :, 2]
    Sxx = S[:, :, 3]
    Sxy = S[:, :, 4]
    Syy = S[:, :, 5]

    # ---- Weighted means.
    denom = S0 + eps                              # [B, N, F]
    mu_x = Sx / denom
    mu_y = Sy / denom

    # ---- 2x2 weighted covariance entries, using the user-supplied
    # *exact* formula (numerator form). This is algebraically equal to
    # the bias-variance identity ``S_αβ/(S_0+ε) - μ_α μ_β`` only up to
    # an O(ε)-sized term, so we use the explicit numerator
    # ``S_αβ - μ_α S_β - μ_β S_α + μ_α μ_β S_0`` (which simplifies to
    # ``S_αα - 2 μ_α S_α + μ_α^2 S_0`` on the diagonal) and then divide
    # by ``S_0 + ε``. This guarantees the loss matches the formula in
    # the docstring bit-for-bit.
    num_xx = Sxx - 2.0 * mu_x * Sx + mu_x * mu_x * S0
    num_yy = Syy - 2.0 * mu_y * Sy + mu_y * mu_y * S0
    num_xy = Sxy - mu_x * Sy - mu_y * Sx + mu_x * mu_y * S0

    C_xx = (num_xx / denom).clamp_min(0.0)
    C_yy = (num_yy / denom).clamp_min(0.0)
    C_xy = num_xy / denom

    trace = C_xx + C_yy
    det = (C_xx * C_yy - C_xy * C_xy).clamp_min(0.0)

    # ---- Zero out rows for padded cells i.
    valid_i = mask.to(dtype).unsqueeze(-1)  # [B, N, 1]
    trace = trace * valid_i
    det = det * valid_i
    return trace, det


def precompute_gt_neighborhood_weighted_covariance(
    true_positions: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
    radius: float,
    *,
    save_path: Optional[str] = None,
    soft_beta: Optional[float] = None,
    include_self: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot helper to precompute the GT-side (trace, det) tensors.

    Runs under ``torch.no_grad`` so no autograd graph is built. Mirrors
    :func:`precompute_gt_neighborhood_features` for the single-radius
    transcriptome loss; the GT trace/det are constant per batch and can
    therefore be looked up at every step instead of being recomputed.
    """
    with torch.no_grad():
        tr, det = compute_neighborhood_weighted_covariance_trace_det(
            true_positions, node_features, node_mask,
            radius=radius,
            soft_beta=soft_beta,
            eps=eps,
            include_self=include_self,
        )
    if save_path is not None:
        torch.save(
            {"trace": tr.detach().cpu(), "det": det.detach().cpu()},
            save_path,
        )
    return tr, det


class NeighborhoodWeightedCovarianceLoss(nn.Module):
    """RMSE between predicted and GT (trace, det) of the weighted-spatial
    covariance matrix per (cell, gene).

    For every (cell ``c``, gene ``g``) we build the 2x2 covariance matrix
    of cell positions in the neighborhood of ``c`` weighted by
    ``exp(feature[i, g])`` (see
    :func:`compute_neighborhood_weighted_covariance_trace_det` for the
    exact formula). The loss compares the trace and determinant tensors
    between predicted and GT positions with a per-cell RMSE over genes.

    Per-cell-per-side aggregation:
        rmse_trace(c) = sqrt( mean_g (trace_pred(c, g) - trace_gt(c, g))^2 + eps )
        rmse_det(c)   = sqrt( mean_g (det_pred(c, g)   - det_gt(c, g))^2   + eps )

    Then::

        L = trace_weight * mean_c rmse_trace(c)
          + det_weight   * mean_c rmse_det(c)

    Differentiability is governed by ``soft_beta`` exactly like the
    other neighborhood losses.

    The GT side does not depend on the model and can be precomputed
    once via :meth:`precompute_gt` and supplied through
    ``cached_gt_trace`` / ``cached_gt_det`` to skip the GT
    recomputation every step.
    """

    def __init__(
        self,
        radius: float,
        soft_beta: Optional[float] = None,
        eps: float = 1e-6,
        include_self: bool = True,
        trace_weight: float = 1.0,
        det_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.eps = float(eps)
        self.include_self = bool(include_self)
        self.trace_weight = float(trace_weight)
        self.det_weight = float(det_weight)
        # Per-step caches surfaced through ``log_epoch_metrics``.
        self._last_loss: float = -1.0
        self._last_trace: float = -1.0
        self._last_det: float = -1.0

    # ---------------- GT precompute / caching ----------------

    def precompute_gt(
        self,
        true_positions: torch.Tensor,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around
        :func:`precompute_gt_neighborhood_weighted_covariance`."""
        return precompute_gt_neighborhood_weighted_covariance(
            true_positions, node_features, node_mask,
            radius=self.radius,
            save_path=save_path,
            soft_beta=self.soft_beta,
            include_self=self.include_self,
            eps=self.eps,
        )

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        cached_gt_trace: Optional[torch.Tensor] = None,
        cached_gt_det: Optional[torch.Tensor] = None,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,  # swallow ``batch_idx`` etc. for drop-in compat
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        node_mask = masked_true.node_mask
        node_features = masked_true.node_features
        pred_xy = masked_pred.positions[..., :2]
        true_xy = masked_true.positions[..., :2]

        # ---- GT side: either reuse the cache or recompute under no_grad.
        gt_provided = (
            cached_gt_trace is not None and cached_gt_det is not None
        )
        if not gt_provided:
            with torch.no_grad():
                gt_trace, gt_det = compute_neighborhood_weighted_covariance_trace_det(
                    true_xy, node_features, node_mask,
                    radius=self.radius,
                    soft_beta=self.soft_beta,
                    eps=self.eps,
                    include_self=self.include_self,
                )
        else:
            gt_trace = cached_gt_trace.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )
            gt_det = cached_gt_det.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )

        # ---- Pred side (with gradient through positions).
        pred_trace, pred_det = compute_neighborhood_weighted_covariance_trace_det(
            pred_xy, node_features, node_mask,
            radius=self.radius,
            soft_beta=self.soft_beta,
            eps=self.eps,
            include_self=self.include_self,
        )

        # ---- Per-cell RMSE over the gene axis (mirrors the transcriptome
        # RMSE aggregation in ``NeighborhoodTranscriptomeRMSELoss``).
        sq_err_trace = (pred_trace - gt_trace).pow(2)    # [B, N, F]
        sq_err_det = (pred_det - gt_det).pow(2)          # [B, N, F]
        per_cell_mse_trace = sq_err_trace.mean(dim=-1)   # [B, N]
        per_cell_mse_det = sq_err_det.mean(dim=-1)       # [B, N]
        # ``+ eps`` keeps ``sqrt`` differentiable at the optimum.
        per_cell_rmse_trace = torch.sqrt(per_cell_mse_trace + self.eps)
        per_cell_rmse_det = torch.sqrt(per_cell_mse_det + self.eps)

        # ---- Mean over valid cells.
        valid_i = node_mask.to(per_cell_rmse_trace.dtype)
        n_valid = valid_i.sum().clamp_min(1.0)
        trace_term = (per_cell_rmse_trace * valid_i).sum() / n_valid
        det_term = (per_cell_rmse_det * valid_i).sum() / n_valid

        loss = self.trace_weight * trace_term + self.det_weight * det_term

        # Cache scalars for log_epoch_metrics.
        if train_stage:
            self._last_trace = float(trace_term.detach().item())
            self._last_det = float(det_term.detach().item())
            self._last_loss = float(loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/neighborhood_cov": float(loss.detach().item()),
                f"{prefix}/neighborhood_cov_trace": float(
                    trace_term.detach().item()
                ),
                f"{prefix}/neighborhood_cov_det": float(det_term.detach().item()),
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        """No running state to reset (last-step caches are intentionally
        carried across the epoch boundary; overwritten by next forward)."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Expose the last-step loss + trace/det breakdown under
        ``train_epoch/...`` keys for Lightning epoch-aggregation."""
        to_log = {
            "train_epoch/neighborhood_cov": float(self._last_loss),
            "train_epoch/neighborhood_cov_trace": float(self._last_trace),
            "train_epoch/neighborhood_cov_det": float(self._last_det),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class NeighborhoodTranscriptomeAndCovarianceLoss(nn.Module):
    """Combined loss: transcriptome-RMSE + weighted-spatial-covariance.

    Wraps two existing single-radius neighborhood sub-losses (both
    sharing the **same** radius, ``soft_beta``, ``eps`` and
    ``include_self``):

      * :class:`NeighborhoodTranscriptomeRMSELoss` -- per-cell RMSE
        between predicted and GT neighborhood-averaged gene profiles.
      * :class:`NeighborhoodWeightedCovarianceLoss` -- per-cell RMSE
        between predicted and GT (trace, det) of the
        ``exp(expression)``-weighted 2x2 spatial-covariance matrix.

    The total is::

        L = transcriptome_weight * L_transcriptome
          + cov_weight           * L_covariance

    where ``L_covariance = trace_weight * trace_rmse + det_weight * det_rmse``
    (see :class:`NeighborhoodWeightedCovarianceLoss`).
    """

    def __init__(
        self,
        radius: float,
        soft_beta: Optional[float] = None,
        eps: float = 1e-6,
        include_self: bool = True,
        transcriptome_weight: float = 1.0,
        cov_weight: float = 1.0,
        trace_weight: float = 16.0,
        det_weight: float = 32.0,
    ) -> None:
        super().__init__()
        self.transcriptome_weight = float(transcriptome_weight)
        self.cov_weight = float(cov_weight)
        self.transcriptome = NeighborhoodTranscriptomeRMSELoss(
            radius=radius,
            soft_beta=soft_beta,
            eps=eps,
            include_self=include_self,
        )
        self.covariance = NeighborhoodWeightedCovarianceLoss(
            radius=radius*4,
            soft_beta=soft_beta,
            eps=eps,
            include_self=include_self,
            trace_weight=trace_weight,
            det_weight=det_weight,
        )
        # Caches for log_epoch_metrics. The sub-losses also expose their
        # own caches so we keep separate fields here to log the combined
        # total cleanly.
        self._last_loss: float = -1.0
        self._last_transcriptome: float = -1.0
        self._last_covariance: float = -1.0

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,  # accept ``batch_idx`` etc. for drop-in compat
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        # The sub-losses each compute their own ``cdist`` (one per side
        # for transcriptome, one per side for covariance). We *could*
        # share them but it would require routing a cached-dists matrix
        # through the transcriptome path too; the 2x cdist overhead is
        # ~10% of total loss cost on typical GPU workloads so we keep
        # the cleaner composition.
        tx_loss, tx_log = self.transcriptome(
            masked_pred, masked_true,
            train_stage=train_stage, log=log,
        )
        cov_loss, cov_log = self.covariance(
            masked_pred, masked_true,
            train_stage=train_stage, log=log,
        )

        loss = (
            self.transcriptome_weight * tx_loss
            + self.cov_weight * cov_loss
        )

        if train_stage:
            self._last_transcriptome = float(tx_loss.detach().item())
            self._last_covariance = float(cov_loss.detach().item())
            self._last_loss = float(loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/neighborhood_transcriptome_cov": float(
                    loss.detach().item()
                ),
                f"{prefix}/neighborhood_transcriptome_cov_transcriptome": float(
                    tx_loss.detach().item()
                ),
                f"{prefix}/neighborhood_transcriptome_cov_covariance": float(
                    cov_loss.detach().item()
                ),
            }
            # Surface the sub-losses' detailed keys too (already emitted
            # to WandB by their own ``wandb.log`` calls; we add them to
            # the dict so the Lightning ``log_dict`` path also sees them).
            if tx_log:
                to_log.update(tx_log)
            if cov_log:
                to_log.update(cov_log)
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.transcriptome.reset()
        self.covariance.reset()

    def log_epoch_metrics(self) -> Dict[str, float]:
        to_log = {
            "train_epoch/neighborhood_transcriptome_cov": float(self._last_loss),
            "train_epoch/neighborhood_transcriptome_cov_transcriptome": float(
                self._last_transcriptome
            ),
            "train_epoch/neighborhood_transcriptome_cov_covariance": float(
                self._last_covariance
            ),
        }
        # Forward to sub-losses so their own ``train_epoch/...`` keys
        # (e.g. ``train_epoch/neighborhood_cov_trace``) get aggregated by
        # Lightning too.
        to_log.update(self.transcriptome.log_epoch_metrics())
        to_log.update(self.covariance.log_epoch_metrics())
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class CombinedLossFunction(nn.Module):
    """Convenience wrapper combining:
       * the pairwise-distance MSE (``LossFunction``),
       * the Cahn-Hilliard energy-curve loss
         (``CahnHilliardEnergyCurveLoss``, per-radius vector comparison,
         no AUC integration),
       * the Voronoi phase-pair CH-energy loss
         (``VoronoiPhasePairEnergyLoss``).

           L = mse_weight     * L_mse
             + ch_weight      * L_ch_curve
             + voronoi_weight * L_voronoi_pair

    Any component can be turned off by setting its weight to ``0`` (we
    short-circuit those branches so we don't pay their cost).

    Parameters are split by component using a ``voronoi_*`` prefix to
    avoid colliding with the existing CH-curve parameters (some of which
    have similar names, e.g. ``grid_resolution``).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        ch_weight: float = 1.0,
        # --- CH energy-curve parameters -----------------------------------
        radii: Sequence[float] = np.linspace(0.0004, 0.01, 25),
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
        # --- Voronoi phase-pair parameters --------------------------------
        # Disabled by default (weight=0) so the existing behaviour of this
        # combined loss is unchanged unless the user opts in.
        voronoi_weight: float = None,
        voronoi_transition_width: float = 0.01,
        voronoi_grid_resolution: int = 64,
        voronoi_kappa: float = 1.0,
        voronoi_soft_beta: Optional[float] = None,
        voronoi_square_bbox: bool = True,
        voronoi_margin: float = 0.05,
        voronoi_min_cells_per_type: int = 2,
        voronoi_chunk: int = 4096,
        voronoi_cache_gt: bool = False,
        voronoi_cache_key_mode: Literal["content", "slot"] = "content",
        # --- Visualization (every N epochs, both sub-losses) --------------
        viz_every_n_epochs: int = 0,
        ch_viz_num_figures: int = 1,
        voronoi_viz_num_figures: int = 4,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.ch_weight = float(ch_weight)
        self.voronoi_weight = float(voronoi_weight) if voronoi_weight != -1 else float(ch_weight)

        self.mse_loss = LossFunction()
        self.ch_loss = CahnHilliardEnergyCurveLoss(
            radii=radii,
            grid_resolution=grid_resolution,
            kappa=kappa,
            bump_fn=bump_fn,
            bump_kwargs=bump_kwargs,
            combine=combine,
            soft_max_beta=soft_max_beta,
            support_factor=support_factor,
            landscape_chunk_size=landscape_chunk_size,
            square_bbox=square_bbox,
            margin=margin,
            eps=eps,
            min_cells_per_type=min_cells_per_type,
            viz_every_n_epochs=viz_every_n_epochs,
            viz_num_figures=ch_viz_num_figures,
        )
        # We always instantiate the Voronoi loss so users can introspect /
        # toggle it without recreating the combined module; if the weight
        # is 0 we just skip its forward pass (see ``forward`` below).
        self.voronoi_loss = VoronoiPhasePairEnergyLoss(
            transition_width=voronoi_transition_width,
            grid_resolution=voronoi_grid_resolution,
            kappa=voronoi_kappa,
            soft_beta=voronoi_soft_beta,
            square_bbox=voronoi_square_bbox,
            margin=voronoi_margin,
            eps=eps,
            min_cells_per_type=voronoi_min_cells_per_type,
            chunk=voronoi_chunk,
            cache_gt=voronoi_cache_gt,
            cache_key_mode=voronoi_cache_key_mode,
            viz_every_n_epochs=viz_every_n_epochs,
            viz_num_figures=voronoi_viz_num_figures,
        )

    def set_current_epoch(self, epoch: int) -> None:
        """Forward the trainer's current epoch to the sub-losses that
        gate visualization on it.
        """
        if hasattr(self.ch_loss, "set_current_epoch"):
            self.ch_loss.set_current_epoch(epoch)
        if hasattr(self.voronoi_loss, "set_current_epoch"):
            self.voronoi_loss.set_current_epoch(epoch)

    def clear_voronoi_gt_cache(self) -> None:
        """Drop the cached GT-side data of the Voronoi sub-loss.

        Useful as a manual safety net after dataset rechunking; with
        ``voronoi_cache_gt=True`` the cache also auto-invalidates per
        entry whenever a sample's ``cell_ID`` fingerprint changes.
        """
        self.voronoi_loss.clear_gt_cache()

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        verbose: bool = False,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:

        if verbose:
            print(f"MSE weight: {self.mse_weight}")
            print(f"CH weight: {self.ch_weight}")
            print(f"Voronoi weight: {self.voronoi_weight}")

        # MSE is essentially free, always compute it (and weight it).
        mse_val, mse_log = self.mse_loss(
            masked_pred, masked_true, train_stage=train_stage, log=log
        )
        loss = self.mse_weight * mse_val

        

        # CH-curve: skip entirely if disabled (it's the most expensive of
        # the three; no point running it just to multiply by zero).
        ch_log: Optional[Dict[str, float]] = None
        if self.ch_weight != 0.0:
            ch_val, ch_log = self.ch_loss(
                masked_pred, masked_true,
                train_stage=train_stage, log=log,
                batch_idx=batch_idx,
            )
            loss = loss + self.ch_weight * ch_val

        # Voronoi phase-pair: same short-circuit pattern. ``batch_idx`` is
        # forwarded so the sub-loss can cache GT-side energies per batch
        # (only effective when ``voronoi_cache_gt=True`` was passed at
        # construction time).
        voronoi_log: Optional[Dict[str, float]] = None
        if self.voronoi_weight != 0.0:
            voronoi_val, voronoi_log = self.voronoi_loss(
                masked_pred, masked_true,
                train_stage=train_stage, log=log,
                batch_idx=batch_idx,
            )
            loss = loss + self.voronoi_weight * voronoi_val

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {f"{prefix}/combined": loss.item()}
            if mse_log is not None:
                to_log.update(mse_log)
            if ch_log is not None:
                to_log.update(ch_log)
            if voronoi_log is not None:
                to_log.update(voronoi_log)
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.mse_loss.reset()
        self.ch_loss.reset()
        self.voronoi_loss.reset()

    def log_epoch_metrics(self) -> Dict[str, float]:
        return self.mse_loss.log_epoch_metrics()
