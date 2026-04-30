from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

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
        log: bool = False,  # Default value set to False
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
# Cahn-Hilliard energy-curve AUC loss
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
#   * A radius sweep gives an energy curve ``E(r)``, and the loss builds on
#     its trapezoidal AUC.
#
# Per cell-type loss:
#     L_t = 1 - exp( - |AUC_pred - AUC_gt| / (|AUC_gt| + eps) )
# (bounded in ``[0, 1)``, zero iff the AUCs match, smooth everywhere; the
#  denominator is detached so the normalisation doesn't leak gradients).
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
        chunk: number of points processed at once to bound peak memory.

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
    M = pts.shape[0]

    # Grid of shape (ny, nx, 2), y varying on dim 0, x on dim 1 (matches
    # the reference numpy landscape convention).
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)        # [ny, nx, 2]

    support = float(support_factor) * float(radius)
    bump_kwargs = dict(bump_kwargs or {})

    if combine == "soft_max":
        # soft-max over bumps: (1/beta) * logsumexp(beta * b_i).
        # Init accumulator at exp(beta * 0) = 1 so a grid cell with no
        # contributing point degrades to soft-max value 0 -> phi = -1.
        beta = float(soft_max_beta)
        acc = torch.zeros(ny, nx, device=device, dtype=dtype)        # accumulates exp(beta * b)
        acc = acc + 1.0                                               # baseline b=0
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            pts_chunk = pts[start:end]                                # [m, 2]
            diff = grid.unsqueeze(0) - pts_chunk[:, None, None, :]    # [m, ny, nx, 2]
            r = torch.norm(diff, dim=-1)                              # [m, ny, nx]
            bumps = bump_fn(r, radius, **bump_kwargs)                 # [m, ny, nx]
            # Zero out contributions beyond support window.
            bumps = bumps * (r <= support).to(bumps.dtype)
            acc = acc + torch.exp(beta * bumps).sum(dim=0)
        soft_max_bump = torch.log(acc) / beta                         # ~ max_i bumps_i
        phi = -1.0 + 2.0 * soft_max_bump
    elif combine == "hard_max":
        # exact pointwise max, running over chunks.
        best = torch.zeros(ny, nx, device=device, dtype=dtype)        # max bump so far; 0 == no point
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            pts_chunk = pts[start:end]
            diff = grid.unsqueeze(0) - pts_chunk[:, None, None, :]
            r = torch.norm(diff, dim=-1)
            bumps = bump_fn(r, radius, **bump_kwargs)
            bumps = bumps * (r <= support).to(bumps.dtype)
            chunk_max = bumps.max(dim=0).values
            best = torch.maximum(best, chunk_max)
        phi = -1.0 + 2.0 * best
    else:
        raise ValueError(f"unknown combine='{combine}'")

    # Clip to [-1, 1] in case numerical slop pushes past the bounds.
    phi = phi.clamp(-1.0, 1.0)
    return phi


# ---------- Cahn-Hilliard energy (mirror ContinuousLandscape2D energy) -----


def cahn_hilliard_energy_density(
    phi: torch.Tensor,                    # [ny, nx] with values in [-1, 1]
    dx: float,
    dy: float,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Discrete energy density ``e = (phi^2 - 1)^2 + kappa * |grad phi|^2``.

    Mirrors :meth:`ContinuousLandscape2D.cahn_hilliard_energy_density`. Uses
    ``torch.gradient`` (central differences with ``edge_order=1`` boundary
    handling), the same scheme as ``np.gradient(c, dy, dx, edge_order=1)``.
    """
    if kappa < 0:
        raise ValueError("kappa must be >= 0")

    well = (phi * phi - 1.0).pow(2)

    # torch.gradient defaults to edge_order=1 (one-sided on borders, central
    # inside) and returns gradients in the order the spacing is given,
    # matching np.gradient conventions.
    dphi_dy, dphi_dx = torch.gradient(phi, spacing=(float(dy), float(dx)))
    grad_sq = dphi_dx * dphi_dx + dphi_dy * dphi_dy

    return well + kappa * grad_sq


def cahn_hilliard_energy(
    phi: torch.Tensor,
    dx: float,
    dy: float,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Integrated Cahn-Hilliard energy (Riemann sum)."""
    return cahn_hilliard_energy_density(phi, dx, dy, kappa=kappa).sum() * dx * dy


# ---------- Loss module ----------------------------------------------------


class CahnHilliardEnergyAUCLoss(nn.Module):
    """Per-cell-type Cahn-Hilliard energy-curve AUC loss.

    For every sample in the batch and every cell type present in its mask:

        1. Build a shared (square) grid covering the union bbox of the
           ground-truth and predicted points of that type -- torch
           equivalent of the ``common_match_key`` workflow from
           ``cahn_hilliard.py``.
        2. For each radius ``r`` in ``self.radii``, build the continuous
           landscape ``phi(x; r)`` (values in ``[-1, 1]``) on that grid
           for both GT and prediction, then evaluate the CH energy.
        3. Integrate the resulting ``E(r)`` curve via the trapezoidal rule
           to get ``AUC_gt`` and ``AUC_pred``.
        4. Per-type loss:
               ``L_t = 1 - exp( -|AUC_pred - AUC_gt| / (|AUC_gt| + eps) )``.

    The sample loss is the sum over cell types; the batch loss is the mean
    over samples. Fully differentiable w.r.t. the predicted positions.

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
        square_bbox: bool = True,
        margin: float = 0.05,
        eps: float = 1e-6,
        min_cells_per_type: int = 2,
    ) -> None:
        super().__init__()
        if len(radii) < 2:
            raise ValueError("Need at least two radii to compute an AUC.")
        self.radii = tuple(float(r) for r in sorted(radii))
        self.grid_resolution = int(grid_resolution)
        self.kappa = float(kappa)
        self.bump_fn = bump_fn
        self.bump_kwargs = dict(bump_kwargs or {})
        self.combine = combine
        self.soft_max_beta = float(soft_max_beta)
        self.support_factor = float(support_factor)
        self.square_bbox = bool(square_bbox)
        self.margin = float(margin)
        self.eps = float(eps)
        self.min_cells_per_type = int(min_cells_per_type)

    # ---------------- internals ----------------

    def _normalized_exp_diff(
        self,
        auc_pred: torch.Tensor,
        auc_gt: torch.Tensor,
    ) -> torch.Tensor:
        """``1 - exp(-|diff| / |auc_gt|)``; denominator is detached."""
        rel = (auc_pred - auc_gt).abs() / (auc_gt.detach().abs() + self.eps)
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
                f"CahnHilliardEnergyAUCLoss expected cell_class shape [N] or [N, 1], "
                f"got {tuple(cell_class.shape)}."
            )

        if mask_b.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype), 0

        # Only the first two spatial dims are used for the landscape.
        pred_xy = pred_pos[..., :2]
        true_xy = true_pos[..., :2]

        unique_types = torch.unique(cell_class[mask_b])
        radii_t = torch.tensor(self.radii, device=device, dtype=dtype)

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
                phi_gt = self._landscape(true_xy, type_mask, grid_x, grid_y, r)
                phi_pred = self._landscape(pred_xy, type_mask, grid_x, grid_y, r)
                e_gt_curve.append(cahn_hilliard_energy(phi_gt, dx, dy, self.kappa))
                e_pred_curve.append(cahn_hilliard_energy(phi_pred, dx, dy, self.kappa))

            e_gt_t = torch.stack(e_gt_curve)
            e_pred_t = torch.stack(e_pred_curve)

            auc_gt = torch.trapezoid(e_gt_t, radii_t)
            auc_pred = torch.trapezoid(e_pred_t, radii_t)

            total = total + self._normalized_exp_diff(auc_pred, auc_gt)
            n_types += 1

        return total, n_types

    # ---------------- forward ----------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = False,
        cell_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute the CH energy-AUC loss.

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
                "CahnHilliardEnergyAUCLoss requires cell_class; none was provided."
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
                "train_loss/ch_energy_auc"
                if train_stage
                else "val_loss/ch_energy_auc"
            )
            to_log = {
                key: loss.item(),
                f"{key}/avg_types_per_sample": (
                    float(sum(type_counts)) / max(len(type_counts), 1)
                ),
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """No running state to reset."""
        pass


class CombinedLossFunction(nn.Module):
    """Convenience wrapper combining the pairwise-distance MSE with the
    Cahn-Hilliard energy-AUC loss.

        L = mse_weight * L_mse + ch_weight * L_ch

    Either component can be turned off by setting its weight to 0.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        ch_weight: float = 1.0,
        radii: Sequence[float] = np.linspace(0.0004, 0.01, 25),
        grid_resolution: int = 64,
        kappa: float = 1.0,
        bump_fn: BumpFn = sigmoid_bump,
        bump_kwargs: Optional[Dict] = None,
        combine: Literal["soft_max", "hard_max"] = "soft_max",
        soft_max_beta: float = 16.0,
        support_factor: float = 10.0,
        square_bbox: bool = True,
        margin: float = 0.05,
        eps: float = 1e-6,
        min_cells_per_type: int = 2,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.ch_weight = float(ch_weight)
        self.mse_loss = LossFunction()
        self.ch_loss = CahnHilliardEnergyAUCLoss(
            radii=radii,
            grid_resolution=grid_resolution,
            kappa=kappa,
            bump_fn=bump_fn,
            bump_kwargs=bump_kwargs,
            combine=combine,
            soft_max_beta=soft_max_beta,
            support_factor=support_factor,
            square_bbox=square_bbox,
            margin=margin,
            eps=eps,
            min_cells_per_type=min_cells_per_type,
        )

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        mse_val, mse_log = self.mse_loss(
            masked_pred, masked_true, train_stage=train_stage, log=log
        )

        if self.ch_weight == 0.0:
            return self.mse_weight * mse_val, mse_log

        ch_val, ch_log = self.ch_loss(
            masked_pred, masked_true, train_stage=train_stage, log=log
        )
        loss = self.mse_weight * mse_val + self.ch_weight * ch_val

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {f"{prefix}/combined": loss.item()}
            if mse_log is not None:
                to_log.update(mse_log)
            if ch_log is not None:
                to_log.update(ch_log)
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.mse_loss.reset()
        self.ch_loss.reset()

    def log_epoch_metrics(self) -> Dict[str, float]:
        return self.mse_loss.log_epoch_metrics()
