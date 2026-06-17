"""Per-cell-type spatial PCA loss using ``cell_class`` labels.

For each cell type present in a sample we:

  1. Compute the type centroid in (x, y).
  2. Build a soft radial mask that down-weights the ``alpha`` fraction of
     cells farthest from that centroid (sigmoid cutoff at the
     ``(1 - alpha)`` distance quantile).
  3. Run a weighted 2D PCA on the masked cluster.
  4. Extract anisotropy, omnivariance, and the PCA-1 direction (mod ``pi``).

The training loss compares predicted vs. ground-truth anisotropy and
omnivariance per type, and the pairwise modulo-``pi`` angles between
PCA-1 directions of different types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder


def _normalize_cell_class(cell_class: torch.Tensor) -> torch.Tensor:
    """Return ``[N]`` or ``[B, N]`` integer class ids."""
    if cell_class.dim() == 3 and cell_class.shape[-1] == 1:
        cell_class = cell_class.squeeze(-1)
    if cell_class.dim() == 2 and cell_class.shape[-1] == 1:
        cell_class = cell_class.squeeze(-1)
    return cell_class


def _soft_tail_mask_from_centroid(
    positions_xy: torch.Tensor,   # [M, 2]
    alpha: float,
    soft_beta: Optional[float],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft weights that taper the ``alpha`` tail farthest from the centroid.

    The cutoff radius is the ``(1 - alpha)`` quantile of distances to the
    (unweighted) centroid. Weights are ``sigmoid(soft_beta * (r_cut - d))``,
    so cells inside the core stay near 1 and the farthest ``alpha`` fraction
    are pushed toward 0.

    Returns:
        weights: ``[M]``
        centroid: ``[2]``
    """
    M = positions_xy.shape[0]
    device = positions_xy.device
    dtype = positions_xy.dtype

    if M == 0:
        return torch.zeros(0, device=device, dtype=dtype), torch.zeros(
            2, device=device, dtype=dtype
        )

    centroid = positions_xy.mean(dim=0)
    dists = torch.linalg.vector_norm(positions_xy - centroid.unsqueeze(0), dim=-1)

    alpha = float(alpha)
    if alpha <= 0.0:
        return torch.ones(M, device=device, dtype=dtype), centroid

    q = max(min(1.0 - alpha, 1.0), 0.0)
    if M == 1:
        r_cut = dists[0]
    else:
        r_cut = torch.quantile(dists.detach(), q)

    if soft_beta is None:
        weights = (dists <= r_cut).to(dtype)
    else:
        weights = torch.sigmoid(float(soft_beta) * (r_cut - dists))

    return weights.clamp_min(eps), centroid


def _weighted_pca_eigensystem(
    positions_xy: torch.Tensor,   # [M, 2]
    weights: torch.Tensor,        # [M]
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted 2D PCA eigensystem.

    Returns:
        lambda1: largest eigenvalue (>= lambda2)
        lambda2: smallest eigenvalue
        pca1_dir: unit vector ``[2]`` along PCA-1 (mod ``pi``)
    """
    w_sum = weights.sum().clamp_min(eps)
    mu = (weights.unsqueeze(-1) * positions_xy).sum(dim=0) / w_sum

    dx = positions_xy - mu.unsqueeze(0)
    dx_x = dx[:, 0]
    dx_y = dx[:, 1]

    cxx = (weights * dx_x * dx_x).sum() / w_sum
    cyy = (weights * dx_y * dx_y).sum() / w_sum
    cxy = (weights * dx_x * dx_y).sum() / w_sum

    trace = cxx + cyy
    det = (cxx * cyy - cxy * cxy).clamp_min(0.0)
    half = trace * 0.5
    disc = torch.sqrt((half * half - det).clamp_min(0.0))
    lambda1 = half + disc
    lambda2 = (half - disc).clamp_min(eps)

    theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
    pca1_dir = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    spread = trace.clamp_min(0.0)
    valid = (w_sum > eps) & (spread > eps)
    if not bool(valid.item()):
        lambda1 = torch.zeros((), device=positions_xy.device, dtype=positions_xy.dtype)
        lambda2 = torch.zeros((), device=positions_xy.device, dtype=positions_xy.dtype)
        pca1_dir = torch.zeros(2, device=positions_xy.device, dtype=positions_xy.dtype)

    return lambda1, lambda2, pca1_dir


def pca_anisotropy(lambda1: torch.Tensor, lambda2: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalized anisotropy in ``[0, 1)``: ``(l1 - l2) / (l1 + l2)``."""
    return (lambda1 - lambda2) / (lambda1 + lambda2 + eps)


def pca_omnivariance(lambda1: torch.Tensor, lambda2: torch.Tensor, eps: float) -> torch.Tensor:
    """Omnivariance: ``sqrt(l1 * l2)`` (geometric mean axis spread)."""
    return torch.sqrt(lambda1 * lambda2 + eps)


def pca_linearity(lambda1: torch.Tensor, lambda2: torch.Tensor, eps: float) -> torch.Tensor:
    """Linearity index: ``(l1 - l2) / (l1 + eps)`` in ``[0, 1)`` when ``l1 >= l2``."""
    return (lambda1 - lambda2) / (lambda1 + eps)


def compute_slide_pointcloud_pca_descriptors(
    positions: torch.Tensor,        # [N, >=2]
    mask: torch.Tensor,             # [N]
    *,
    min_cells: int = 4,
    eps: float = 1e-6,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Unweighted 2D PCA shape descriptors for one slide (all cells together).

    Returns ``(anisotropy, omnivariance, linearity)`` scalars, or ``None`` if
    too few valid cells.
    """
    mask_b = mask.bool() if mask.dtype != torch.bool else mask
    if int(mask_b.sum().item()) < min_cells:
        return None

    pts = positions[..., :2][mask_b]
    weights = torch.ones(pts.shape[0], device=pts.device, dtype=pts.dtype)
    lam1, lam2, _ = _weighted_pca_eigensystem(pts, weights, eps=eps)
    if (lam1 + lam2) <= eps:
        return None

    aniso = pca_anisotropy(lam1, lam2, eps)
    omni = pca_omnivariance(lam1, lam2, eps)
    linear = pca_linearity(lam1, lam2, eps)
    return aniso, omni, linear


def pairwise_mod_pi_angles(unit_dirs: torch.Tensor) -> torch.Tensor:
    """Pairwise undirected angles in ``[0, pi/2]``, shape ``[K, K]``."""
    dot = unit_dirs @ unit_dirs.transpose(-1, -2)
    cos_angle = dot.abs().clamp(0.0, 1.0)
    return torch.acos(cos_angle)


@dataclass
class SampleCellTypePCAFeatures:
    """PCA-derived descriptors for all qualifying types in one sample."""

    type_ids: torch.Tensor       # [K] int64
    anisotropy: torch.Tensor     # [K]
    omnivariance: torch.Tensor   # [K]
    pca1_dirs: torch.Tensor      # [K, 2]
    pair_angles: torch.Tensor    # [K, K]


def compute_sample_cell_type_pca_features(
    positions: torch.Tensor,      # [N, >=2]
    cell_class: torch.Tensor,       # [N]
    mask: torch.Tensor,             # [N] bool / 0-1
    *,
    alpha: float = 0.1,
    soft_beta: Optional[float] = 256.0,
    min_cells_per_type: int = 4,
    min_weight_sum: float = 1e-3,
    eps: float = 1e-6,
) -> Optional[SampleCellTypePCAFeatures]:
    """Compute per-type PCA descriptors for one sample (one batch row)."""
    mask_b = mask.bool() if mask.dtype != torch.bool else mask
    cell_class = _normalize_cell_class(cell_class)

    if mask_b.sum() == 0:
        return None

    xy = positions[..., :2]
    valid = mask_b & (cell_class >= 0)
    if valid.sum() == 0:
        return None

    unique_types = torch.unique(cell_class[valid])
    type_ids: List[torch.Tensor] = []
    anisotropies: List[torch.Tensor] = []
    omnivariances: List[torch.Tensor] = []
    directions: List[torch.Tensor] = []

    for type_id in unique_types.tolist():
        if type_id < 0:
            continue
        type_mask = valid & (cell_class == type_id)
        n_cells = int(type_mask.sum().item())
        if n_cells < min_cells_per_type:
            continue

        pts = xy[type_mask]
        tail_w, _ = _soft_tail_mask_from_centroid(
            pts, alpha=alpha, soft_beta=soft_beta, eps=eps,
        )
        if tail_w.sum() < min_weight_sum:
            continue

        lam1, lam2, pca1 = _weighted_pca_eigensystem(pts, tail_w, eps=eps)
        if (lam1 + lam2) <= eps:
            continue

        type_ids.append(torch.tensor(type_id, device=positions.device, dtype=torch.long))
        anisotropies.append(pca_anisotropy(lam1, lam2, eps))
        omnivariances.append(pca_omnivariance(lam1, lam2, eps))
        directions.append(pca1)

    if not type_ids:
        return None

    ids = torch.stack(type_ids)
    order = torch.argsort(ids)
    ids = ids[order]
    aniso = torch.stack(anisotropies)[order]
    omni = torch.stack(omnivariances)[order]
    dirs = torch.stack(directions)[order]
    pair_angles = pairwise_mod_pi_angles(dirs)

    return SampleCellTypePCAFeatures(
        type_ids=ids,
        anisotropy=aniso,
        omnivariance=omni,
        pca1_dirs=dirs,
        pair_angles=pair_angles,
    )


def _upper_triangle_mask(K: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)


def _align_features(
    gt: SampleCellTypePCAFeatures,
    pred: SampleCellTypePCAFeatures,
) -> Optional[Tuple[SampleCellTypePCAFeatures, SampleCellTypePCAFeatures]]:
    """Keep only types present in both GT and pred, in identical order."""
    gt_set = {int(t.item()) for t in gt.type_ids}
    pred_set = {int(t.item()) for t in pred.type_ids}
    common = sorted(gt_set & pred_set)
    if not common:
        return None

    gt_idx = [int((gt.type_ids == t).nonzero(as_tuple=True)[0].item()) for t in common]
    pred_idx = [int((pred.type_ids == t).nonzero(as_tuple=True)[0].item()) for t in common]

    gt_idx_t = torch.tensor(gt_idx, device=gt.type_ids.device, dtype=torch.long)
    pred_idx_t = torch.tensor(pred_idx, device=pred.type_ids.device, dtype=torch.long)

    gt_aligned = SampleCellTypePCAFeatures(
        type_ids=gt.type_ids[gt_idx_t],
        anisotropy=gt.anisotropy[gt_idx_t],
        omnivariance=gt.omnivariance[gt_idx_t],
        pca1_dirs=gt.pca1_dirs[gt_idx_t],
        pair_angles=pairwise_mod_pi_angles(gt.pca1_dirs[gt_idx_t]),
    )
    pred_aligned = SampleCellTypePCAFeatures(
        type_ids=pred.type_ids[pred_idx_t],
        anisotropy=pred.anisotropy[pred_idx_t],
        omnivariance=pred.omnivariance[pred_idx_t],
        pca1_dirs=pred.pca1_dirs[pred_idx_t],
        pair_angles=pairwise_mod_pi_angles(pred.pca1_dirs[pred_idx_t]),
    )
    return gt_aligned, pred_aligned


def _sample_loss(
    gt: SampleCellTypePCAFeatures,
    pred: SampleCellTypePCAFeatures,
    *,
    anisotropy_weight: float,
    omnivariance_weight: float,
    pairwise_weight: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Scalar losses for one aligned sample."""
    device = pred.anisotropy.device
    dtype = pred.anisotropy.dtype
    K = pred.anisotropy.shape[0]

    aniso_loss = ((pred.anisotropy - gt.anisotropy.detach()) ** 2).mean()
    omni_loss = ((pred.omnivariance - gt.omnivariance.detach()) ** 2).mean()

    if K < 2 or pairwise_weight == 0.0:
        pair_loss = torch.zeros((), device=device, dtype=dtype)
        n_pairs = 0
    else:
        tri = _upper_triangle_mask(K, device)
        sq_err = (pred.pair_angles - gt.pair_angles.detach()).pow(2)
        pair_loss = sq_err[tri].mean()
        n_pairs = int(tri.sum().item())

    total = (
        anisotropy_weight * aniso_loss
        + omnivariance_weight * omni_loss
        + pairwise_weight * pair_loss
    )
    return total, aniso_loss, omni_loss, pair_loss, K, n_pairs


class CellTypesMetricLoss(nn.Module):
    """Match per-type PCA shape statistics and inter-type orientations."""

    def __init__(
        self,
        alpha: float = 0.1,
        soft_beta: Optional[float] = 256.0,
        min_cells_per_type: int = 4,
        min_weight_sum: float = 1e-3,
        eps: float = 1e-6,
        anisotropy_weight: float = 1.0,
        omnivariance_weight: float = 1.0,
        pairwise_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.min_cells_per_type = int(min_cells_per_type)
        self.min_weight_sum = float(min_weight_sum)
        self.eps = float(eps)
        self.anisotropy_weight = float(anisotropy_weight)
        self.omnivariance_weight = float(omnivariance_weight)
        self.pairwise_weight = float(pairwise_weight)
        self._last_loss: float = -1.0
        self._last_anisotropy: float = -1.0
        self._last_omnivariance: float = -1.0
        self._last_pairwise: float = -1.0

    def _compute_side(
        self,
        positions: torch.Tensor,
        cell_class: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[SampleCellTypePCAFeatures]:
        return compute_sample_cell_type_pca_features(
            positions,
            cell_class,
            mask,
            alpha=self.alpha,
            soft_beta=self.soft_beta,
            min_cells_per_type=self.min_cells_per_type,
            min_weight_sum=self.min_weight_sum,
            eps=self.eps,
        )

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        cell_class: Optional[torch.Tensor] = None,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        if cell_class is None:
            cell_class = masked_true.cell_class
        if cell_class is None:
            raise ValueError(
                "CellTypesMetricLoss requires cell_class; none was provided."
            )

        cell_class = _normalize_cell_class(cell_class)
        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask

        B = pred_positions.shape[0]
        device = pred_positions.device
        dtype = pred_positions.dtype

        losses: List[torch.Tensor] = []
        aniso_terms: List[torch.Tensor] = []
        omni_terms: List[torch.Tensor] = []
        pair_terms: List[torch.Tensor] = []

        for b in range(B):
            gt_feats = self._compute_side(
                true_positions[b], cell_class[b], node_mask[b],
            )
            if gt_feats is None:
                continue

            pred_feats = self._compute_side(
                pred_positions[b], cell_class[b], node_mask[b],
            )
            if pred_feats is None:
                continue

            aligned = _align_features(gt_feats, pred_feats)
            if aligned is None:
                continue

            gt_a, pred_a = aligned
            total, aniso, omni, pair, _, _ = _sample_loss(
                gt_a,
                pred_a,
                anisotropy_weight=self.anisotropy_weight,
                omnivariance_weight=self.omnivariance_weight,
                pairwise_weight=self.pairwise_weight,
                eps=self.eps,
            )
            losses.append(total)
            aniso_terms.append(aniso)
            omni_terms.append(omni)
            if pred_a.anisotropy.shape[0] >= 2:
                pair_terms.append(pair)

        if not losses:
            zero = pred_positions.sum() * 0.0
            return zero, None

        loss = torch.stack(losses).mean()
        loss_val = float(loss.detach().item())
        if train_stage:
            self._last_loss = loss_val
            self._last_anisotropy = float(torch.stack(aniso_terms).mean().detach().item())
            self._last_omnivariance = float(torch.stack(omni_terms).mean().detach().item())
            if pair_terms:
                self._last_pairwise = float(torch.stack(pair_terms).mean().detach().item())
            else:
                self._last_pairwise = 0.0

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/cell_types_metric": loss_val,
                f"{prefix}/cell_types_anisotropy": self._last_anisotropy,
                f"{prefix}/cell_types_omnivariance": self._last_omnivariance,
                f"{prefix}/cell_types_pairwise": self._last_pairwise,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        to_log = {
            "train_epoch/cell_types_metric": float(self._last_loss),
            "train_epoch/cell_types_anisotropy": float(self._last_anisotropy),
            "train_epoch/cell_types_omnivariance": float(self._last_omnivariance),
            "train_epoch/cell_types_pairwise": float(self._last_pairwise),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
