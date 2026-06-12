"""Directional coherence loss from spatially weighted, transcriptome-gated PCA.

For each sample we randomly subsample ``n_target`` cells. For every target
cell we build a soft spatial neighborhood (sigmoid on distance), gate it by
the inverse transcriptome distance ``(1 - cos_sim) * RMSE`` to all cells,
run a spatial weighted PCA, and take the PCA-1 direction modulo ``pi`` as a
unit vector. A second (coherence) radius averages those unit vectors among
target cells only; the mean-vector length is the coherence score.

The training loss matches predicted vs. ground-truth coherence per target
cell and the pairwise modulo-``pi`` angle differences between target cells.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder


def _spatial_soft_weights(
    dists: torch.Tensor,
    radius: float,
    soft_beta: Optional[float],
) -> torch.Tensor:
    """Sigmoid (or hard) spatial membership, shape preserved."""
    if soft_beta is None:
        return (dists <= float(radius)).to(dists.dtype)
    return torch.sigmoid(float(soft_beta) * (float(radius) - dists))


def _transcriptome_distance(
    features: torch.Tensor,          # [B, N, F]
    target_idx: torch.Tensor,        # [B, T]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise transcriptome distance from each target to all cells.

    ``dist = (1 - cosine_similarity) * RMSE`` with shapes ``[B, T, N]``.
    """
    B, _, F = features.shape
    T = target_idx.shape[1]
    idx = target_idx.unsqueeze(-1).expand(B, T, F)
    target_feat = torch.gather(features, 1, idx)  # [B, T, F]

    diff = target_feat.unsqueeze(2) - features.unsqueeze(1)  # [B, T, N, F]
    rmse = torch.sqrt(diff.pow(2).mean(dim=-1).clamp_min(eps))

    t_norm = target_feat.norm(dim=-1, keepdim=True).clamp_min(eps)
    n_norm = features.norm(dim=-1).clamp_min(eps)
    cos_sim = (target_feat.unsqueeze(2) * features.unsqueeze(1)).sum(dim=-1)
    cos_sim = cos_sim / (t_norm * n_norm.unsqueeze(1))
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    return (1.0 - cos_sim) * rmse


def sample_target_indices(
    mask: torch.Tensor,
    n_target: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample up to ``n_target`` valid cells per batch row.

    Returns:
        target_idx: ``[B, T]`` long indices into the ``N`` axis.
        target_valid: ``[B, T]`` bool mask (``False`` when a slot is padding
            because the sample had fewer than ``T`` valid cells).
    """
    B, N = mask.shape
    device = mask.device
    valid = mask.bool()
    n_valid = valid.sum(dim=1)
    n_sample = n_valid.clamp(max=n_target).long()

    scores = torch.rand(B, N, device=device)
    scores = scores.masked_fill(~valid, -1.0)
    target_idx = scores.topk(n_target, dim=1).indices

    slot = torch.arange(n_target, device=device).unsqueeze(0)
    target_valid = slot < n_sample.unsqueeze(1)
    return target_idx, target_valid


def _gather_positions(
    positions: torch.Tensor,   # [B, N, D]
    target_idx: torch.Tensor,  # [B, T]
) -> torch.Tensor:
    """Gather ``[B, T, D]`` positions at ``target_idx``."""
    B, T = target_idx.shape
    D = positions.shape[-1]
    idx = target_idx.unsqueeze(-1).expand(B, T, D)
    return torch.gather(positions, 1, idx)


def compute_weighted_pca_directions(
    positions_xy: torch.Tensor,    # [B, N, 2]
    features: torch.Tensor,        # [B, N, F]
    mask: torch.Tensor,            # [B, N]
    target_idx: torch.Tensor,      # [B, T]
    target_valid: torch.Tensor,    # [B, T]
    pca_radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCA-1 unit directions (modulo ``pi``) for each target cell.

    Returns:
        unit_dirs: ``[B, T, 2]`` -- ``(cos theta, sin theta)``.
        pca_valid: ``[B, T]`` bool -- ``True`` when the weighted neighborhood
            has enough mass to define a direction.
    """
    target_pos = _gather_positions(positions_xy, target_idx)
    spatial_dists = torch.cdist(target_pos, positions_xy, p=2)

    spatial_w = _spatial_soft_weights(spatial_dists, pca_radius, soft_beta)
    trans_dist = _transcriptome_distance(features, target_idx, eps=eps)
    weights = spatial_w / (trans_dist + eps)
    weights = weights * mask.to(weights.dtype).unsqueeze(1)

    w_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)

    pos = positions_xy.unsqueeze(1)
    mu = (weights.unsqueeze(-1) * pos).sum(dim=2) / w_sum

    dx = pos - mu.unsqueeze(2)
    dx_x = dx[..., 0]
    dx_y = dx[..., 1]

    cxx = (weights * dx_x * dx_x).sum(dim=-1) / w_sum.squeeze(-1)
    cyy = (weights * dx_y * dx_y).sum(dim=-1) / w_sum.squeeze(-1)
    cxy = (weights * dx_x * dx_y).sum(dim=-1) / w_sum.squeeze(-1)

    theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
    unit_dirs = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    spread = (cxx + cyy).clamp_min(0.0)
    pca_valid = target_valid & (w_sum.squeeze(-1) > eps) & (spread > eps)
    unit_dirs = unit_dirs * pca_valid.unsqueeze(-1).to(unit_dirs.dtype)
    return unit_dirs, pca_valid


def compute_coherence(
    target_positions: torch.Tensor,  # [B, T, 2]
    unit_dirs: torch.Tensor,         # [B, T, 2]
    target_valid: torch.Tensor,      # [B, T]
    coherence_radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-vector length of neighboring target directions.

    Returns ``[B, T]`` coherence in ``[0, 1]`` (clipped for stability).
    """
    dists = torch.cdist(target_positions, target_positions, p=2)
    coh_w = _spatial_soft_weights(dists, coherence_radius, soft_beta)

    valid_pair = target_valid.unsqueeze(2) & target_valid.unsqueeze(1)
    coh_w = coh_w * valid_pair.to(coh_w.dtype)

    denom = coh_w.sum(dim=-1, keepdim=True).clamp_min(eps)
    avg_vec = torch.matmul(coh_w, unit_dirs) / denom
    coherence = avg_vec.norm(dim=-1).clamp(0.0, 1.0)
    return coherence * target_valid.to(coherence.dtype)


def pairwise_mod_pi_angles(unit_dirs: torch.Tensor) -> torch.Tensor:
    """Pairwise undirected angles in ``[0, pi/2]``, shape ``[B, T, T]``."""
    dot = torch.matmul(unit_dirs, unit_dirs.transpose(-1, -2))
    cos_angle = dot.abs().clamp(0.0, 1.0)
    return torch.acos(cos_angle)


def compute_directional_features(
    positions: torch.Tensor,         # [B, N, >=2]
    features: torch.Tensor,          # [B, N, F]
    mask: torch.Tensor,              # [B, N]
    target_idx: torch.Tensor,        # [B, T]
    target_valid: torch.Tensor,      # [B, T]
    pca_radius: float,
    coherence_radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full directional pipeline for one side (pred or GT).

    Returns:
        coherence: ``[B, T]``
        pair_angles: ``[B, T, T]`` modulo-``pi`` pairwise angles
        feature_valid: ``[B, T]`` bool mask for targets with valid PCA
    """
    xy = positions[..., :2]
    unit_dirs, pca_valid = compute_weighted_pca_directions(
        xy, features, mask, target_idx, target_valid,
        pca_radius, soft_beta=soft_beta, eps=eps,
    )
    target_pos = _gather_positions(xy, target_idx)
    coherence = compute_coherence(
        target_pos, unit_dirs, pca_valid,
        coherence_radius, soft_beta=soft_beta, eps=eps,
    )
    pair_angles = pairwise_mod_pi_angles(unit_dirs)
    return coherence, pair_angles, pca_valid


def _upper_triangle_mask(T: int, device: torch.device) -> torch.Tensor:
    """``[T, T]`` bool mask with ``True`` on ``i < j``."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def _masked_mean(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(eps)


class DirectionalMetricLoss(nn.Module):
    """Match directional coherence and pairwise angle structure pred vs. GT.

    Randomly subsamples ``n_target`` cells per sample, builds transcriptome-
    gated spatial PCA directions, aggregates coherence among targets, and
    penalises mismatches in per-cell coherence and all pairwise modulo-``pi``
    angle differences.
    """

    def __init__(
        self,
        n_target: int = 32,
        pca_radius: float = 0.01,
        coherence_radius: float = 0.02,
        soft_beta: Optional[float] = 256.0,
        eps: float = 1e-6,
        coherence_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        min_valid_targets: int = 2,
    ) -> None:
        super().__init__()
        self.n_target = int(n_target)
        self.pca_radius = float(pca_radius)
        self.coherence_radius = float(coherence_radius)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.eps = float(eps)
        self.coherence_weight = float(coherence_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.min_valid_targets = int(min_valid_targets)
        self._last_loss: float = -1.0
        self._last_coherence: float = -1.0
        self._last_pairwise: float = -1.0

    def _pairwise_loss(
        self,
        pred_angles: torch.Tensor,
        gt_angles: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        T = pred_angles.shape[1]
        tri = _upper_triangle_mask(T, pred_angles.device)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        pair_valid = pair_valid & tri.unsqueeze(0)
        sq_err = (pred_angles - gt_angles.detach()).pow(2)
        return _masked_mean(sq_err, pair_valid, eps=self.eps)

    def _coherence_loss(
        self,
        pred_coh: torch.Tensor,
        gt_coh: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        sq_err = (pred_coh - gt_coh.detach()).pow(2)
        return _masked_mean(sq_err, valid, eps=self.eps)

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        node_mask = masked_true.node_mask
        node_features = masked_true.node_features
        true_xy = masked_true.positions
        pred_xy = masked_pred.positions

        target_idx, target_valid = sample_target_indices(node_mask, self.n_target)

        with torch.no_grad():
            gt_coh, gt_pairs, gt_valid = compute_directional_features(
                true_xy, node_features, node_mask, target_idx, target_valid,
                self.pca_radius, self.coherence_radius,
                soft_beta=self.soft_beta, eps=self.eps,
            )

        pred_coh, pred_pairs, pred_valid = compute_directional_features(
            pred_xy, node_features, node_mask, target_idx, target_valid,
            self.pca_radius, self.coherence_radius,
            soft_beta=self.soft_beta, eps=self.eps,
        )

        valid = gt_valid & pred_valid & target_valid
        n_valid = valid.sum()

        if n_valid < self.min_valid_targets:
            zero = pred_xy.sum() * 0.0
            return zero, None

        coh_loss = self._coherence_loss(pred_coh, gt_coh, valid)
        pair_loss = self._pairwise_loss(pred_pairs, gt_pairs, valid)
        loss = self.coherence_weight * coh_loss + self.pairwise_weight * pair_loss

        loss_val = float(loss.detach().item())
        if train_stage:
            self._last_loss = loss_val
            self._last_coherence = float(coh_loss.detach().item())
            self._last_pairwise = float(pair_loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/directional_metric": loss_val,
                f"{prefix}/directional_coherence": self._last_coherence,
                f"{prefix}/directional_pairwise": self._last_pairwise,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        to_log = {
            "train_epoch/directional_metric": float(self._last_loss),
            "train_epoch/directional_coherence": float(self._last_coherence),
            "train_epoch/directional_pairwise": float(self._last_pairwise),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
