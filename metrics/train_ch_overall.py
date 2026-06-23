## Loss function for Cahn-Hilliard overall (whole-slide CH AUC)

import math
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

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

    Default ``decay_rate = 4 / cell_radius``; ``shift = 0`` by default.
    """
    cell_radius = max(float(cell_radius), 1e-12)
    if decay_rate is None:
        decay_rate = 4.0 / cell_radius
    decay_rate = float(decay_rate)
    return torch.sigmoid(-(decay_rate * (r - cell_radius) - shift))


def shared_square_grid(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
    grid_resolution: int,
    margin: float = 0.0,
    square: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Axis-aligned grid covering the union bbox of two point sets."""
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
    """Build CH landscapes for all ``radii`` over cell chunks.

    Returns ``phi`` of shape ``[R, ny, nx]``. Each radius uses its own
    local grid patch sized to that radius' support (``support_factor * r``).
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

    dx_grid = abs(float((grid_x[1] - grid_x[0]).detach().item())) if nx > 1 else 1.0
    dy_grid = abs(float((grid_y[1] - grid_y[0]).detach().item())) if ny > 1 else 1.0
    grid_size = ny * nx
    beta = float(soft_max_beta)
    use_fast_sigmoid = (bump_fn is sigmoid_bump) and not bump_kwargs

    def patch_bumps(
        pts_chunk: torch.Tensor, rad_k: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support_k = float(support_factor) * rad_k
        half_x = min(nx - 1, max(0, math.ceil(support_k / max(dx_grid, 1e-12)) + 1))
        half_y = min(ny - 1, max(0, math.ceil(support_k / max(dy_grid, 1e-12)) + 1))
        offset_x = torch.arange(-half_x, half_x + 1, device=device)
        offset_y = torch.arange(-half_y, half_y + 1, device=device)

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

        if use_fast_sigmoid:
            bump = torch.sigmoid(-((4.0 / rad_k) * (r - rad_k) - 128.0 * rad_k))
        else:
            bump = bump_fn(
                r,
                rad_k,
                decay_rate=4.0 / rad_k,
                shift=128.0 * rad_k,
                **bump_kwargs,
            )
        support_mask = valid & (r <= support_k)
        bump = bump * support_mask.to(dtype)

        flat_idx = y_idx * nx + x_idx
        return flat_idx, bump

    rows = []
    for rad_k in radii_f:
        if combine == "soft_max":
            acc = torch.ones(grid_size, device=device, dtype=dtype)
            for start in range(0, pts.shape[0], chunk):
                flat_idx, bump = patch_bumps(pts[start:start + chunk], rad_k)
                values = torch.exp(beta * bump) * (bump > 0).to(dtype)
                acc.scatter_add_(0, flat_idx.reshape(-1), values.reshape(-1))
            rows.append(torch.log(acc.reshape(ny, nx)) / beta)
        elif combine == "hard_max":
            best = torch.zeros(grid_size, device=device, dtype=dtype)
            for start in range(0, pts.shape[0], chunk):
                flat_idx, bump = patch_bumps(pts[start:start + chunk], rad_k)
                best.scatter_reduce_(
                    0,
                    flat_idx.reshape(-1),
                    bump.reshape(-1),
                    reduce="amax",
                    include_self=True,
                )
            rows.append(best.reshape(ny, nx))
        else:
            raise ValueError(f"unknown combine='{combine}'")

    phi = -1.0 + 2.0 * torch.stack(rows, dim=0)
    return phi.clamp(-1.0, 1.0)


def _grad_xy_edge_order_1(
    phi: torch.Tensor, dx: float, dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradient with ``edge_order=1`` boundaries.

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
    phi: torch.Tensor,
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

    For ``phi`` of shape ``[..., ny, nx]`` returns a tensor of shape ``[...]``.
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


class SlideCHLoss(nn.Module):
    """Whole-slide Cahn-Hilliard AUC loss.

    Compares the area under the CH energy-vs-radius curve between predicted
    and ground-truth cell positions. The AUC difference is normalised via
    ``1 - exp(-|pred - gt| / (|gt| + eps))``.
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
        min_cells: int = 10,
        cache_gt: bool = True,
    ) -> None:
        super().__init__()
        if len(radii) < 2:
            raise ValueError("Need at least two radii for CH AUC integration.")
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
    ) -> Optional[Tuple]:
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

        ch_terms = []
        for b in range(pred_positions.shape[0]):
            mask_b = node_mask[b]
            mask_b = mask_b.bool() if mask_b.dtype != torch.bool else mask_b
            pred_xy = pred_positions[b, ..., :2]
            true_xy = true_positions[b, ..., :2]
            cid_b = cell_id[b] if cell_id is not None else None
            ch_t = self._ch_auc_term(pred_xy, true_xy, mask_b, cell_id=cid_b)
            ch_terms.append(ch_t)

        if not ch_terms:
            zero = pred_positions.sum() * 0.0
            return zero, None

        loss = torch.stack(ch_terms).mean()
        self._last_loss = float(loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/slide_ch_auc": float(loss.detach().item()),
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.clear_gt_cache()

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        return {
            f"{epoch_prefix}/slide_ch_auc": float(self._last_loss),
        }
