## Loss function for directional coherence of the spatial cells based on transcriptome distance and spatial direction

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder

"""Directional coherence loss from transcriptome-weighted orientation averaging.

For each sample we randomly subsample ``n_target`` cells. For every target
cell we look at its spatial neighbors (soft sigmoid membership inside
``neighbor_radius``) and build a *local orientation axis* using circular
statistics on the **double-angle** representation:

  1. For each neighbor ``j`` of target ``i`` we take the unit vector
     ``u_ij = (p_j - p_i) / ||p_j - p_i||`` with angle ``theta_ij``.
  2. We map it to the double-angle (axis) space
     ``z_ij = (cos 2 theta_ij, sin 2 theta_ij)``. This identifies opposite
     directions (``theta`` and ``theta + pi``) so we measure an *axis*, not a
     signed direction.
  3. We average ``z_ij`` with weights ``w_ij = membership_ij *
     exp(-beta * d_trans_ij)`` where ``d_trans_ij = (1 - cos_sim) * RMSE`` is
     the transcriptome distance. The resultant ``Z_i = sum_j w_ij z_ij /
     sum_j w_ij`` has norm ``R_i in [0, 1]`` (orientation concentration) and
     argument ``2 phi_i`` (twice the mean axis angle). We keep ``R_i``,
     normalise to recover the axis, send it back to spatial coordinates
     (mod ``pi``) and re-scale by ``R_i`` -- which is exactly ``Z_i`` again.

A second radius (``coherence_radius``) smooths these axes among target cells
(double-angle average again, so cells with a more coherent axis -- larger
``R`` -- dominate). The final per-cell *length* is the norm of the smoothed
double-angle vector, and the per-cell *axis* is its (half-angle) argument.

The training loss matches predicted vs. ground-truth per-cell length and the
pairwise modulo-``pi`` axis-angle differences between target cells. Each
pairwise term is weighted by the product of the two cells' ground-truth
lengths.
"""

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
    k = min(int(n_target), int(N))
    target_idx = scores.topk(k, dim=1).indices
    if k < n_target:
        pad = torch.zeros(B, n_target - k, device=device, dtype=target_idx.dtype)
        target_idx = torch.cat([target_idx, pad], dim=1)

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


def compute_orientation_axes(
    positions_xy: torch.Tensor,    # [B, N, 2]
    features: torch.Tensor,        # [B, N, F]
    mask: torch.Tensor,            # [B, N]
    target_idx: torch.Tensor,      # [B, T]
    target_valid: torch.Tensor,    # [B, T]
    neighbor_radius: float,
    *,
    trans_beta: float = 1.0,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """First-radius local orientation axis per target cell (double-angle).

    For target ``i`` and neighbor ``j`` let ``(dx, dy) = p_j - p_i``. The unit
    vector ``u_ij`` has angle ``theta_ij`` and its double-angle map is

        z_ij = (cos 2 theta_ij, sin 2 theta_ij)
             = ( (dx^2 - dy^2) / (dx^2 + dy^2),  2 dx dy / (dx^2 + dy^2) ).

    With weights ``w_ij = membership(d_ij; neighbor_radius)
    * exp(-trans_beta * d_trans_ij) * mask_j`` (self excluded) the resultant

        Z_i = sum_j w_ij z_ij / sum_j w_ij

    has norm ``R_i = ||Z_i|| in [0, 1]`` (orientation concentration) and
    argument ``2 phi_i``. Note that ``Z_i`` already equals
    ``R_i * (cos 2 phi_i, sin 2 phi_i)`` -- i.e. the spatial axis
    ``(cos phi_i, sin phi_i)`` re-scaled by ``R_i`` and re-encoded in
    double-angle space -- so we return ``Z_i`` directly for the second stage.

    Returns:
        axis_double: ``[B, T, 2]`` -- the double-angle resultant ``Z_i``.
        valid: ``[B, T]`` bool -- targets with non-degenerate neighborhoods.
    """
    B, N, _ = positions_xy.shape
    target_pos = _gather_positions(positions_xy, target_idx)        # [B, T, 2]

    # Neighbor-minus-target offsets and their double-angle directions.
    delta = positions_xy.unsqueeze(1) - target_pos.unsqueeze(2)     # [B, T, N, 2]
    dx = delta[..., 0]
    dy = delta[..., 1]
    r2 = (dx * dx + dy * dy).clamp_min(eps)                         # [B, T, N]
    dist = torch.sqrt(r2)
    cos2 = (dx * dx - dy * dy) / r2
    sin2 = (2.0 * dx * dy) / r2
    z = torch.stack((cos2, sin2), dim=-1)                          # [B, T, N, 2]

    # Weights: spatial membership * transcriptome proximity * padding mask.
    spatial_w = _spatial_soft_weights(dist, neighbor_radius, soft_beta)
    trans_dist = _transcriptome_distance(features, target_idx, eps=eps)
    trans_w = torch.exp(-float(trans_beta) * trans_dist)
    weights = spatial_w * trans_w * mask.to(spatial_w.dtype).unsqueeze(1)

    # Exclude the target itself (zero offset -> undefined direction).
    self_oh = torch.zeros(B, target_idx.shape[1], N, device=weights.device, dtype=weights.dtype)
    self_oh.scatter_(2, target_idx.unsqueeze(-1), 1.0)
    weights = weights * (1.0 - self_oh)

    w_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)       # [B, T, 1]
    axis_double = (weights.unsqueeze(-1) * z).sum(dim=2) / w_sum    # [B, T, 2]

    valid = target_valid & (w_sum.squeeze(-1) > eps)
    axis_double = axis_double * valid.unsqueeze(-1).to(axis_double.dtype)
    return axis_double, valid


def aggregate_axes_second_radius(
    target_positions: torch.Tensor,  # [B, T, 2]
    axis_double: torch.Tensor,       # [B, T, 2]
    valid: torch.Tensor,             # [B, T]
    coherence_radius: float,
    *,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Second-radius smoothing of the axes among target cells.

    Spatial membership ``a_ik`` (sigmoid inside ``coherence_radius``, self
    included) averages the double-angle axes:

        Z'_i = sum_k a_ik Z_k / sum_k a_ik.

    Because each ``Z_k`` carries its own magnitude ``R_k``, this is a
    magnitude-weighted axis average: cells whose first-radius axis is more
    coherent contribute more. The final length is ``||Z'_i||`` and the final
    axis is ``arg(Z'_i) / 2`` (mod ``pi``).

    Returns:
        axis_double2: ``[B, T, 2]`` -- smoothed double-angle vector ``Z'_i``.
        valid2: ``[B, T]`` bool.
    """
    dists = torch.cdist(target_positions, target_positions, p=2)    # [B, T, T]
    a = _spatial_soft_weights(dists, coherence_radius, soft_beta)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    a = a * valid_pair.to(a.dtype)

    denom = a.sum(dim=-1, keepdim=True).clamp_min(eps)
    axis_double2 = torch.matmul(a, axis_double) / denom            # [B, T, 2]

    valid2 = valid & (denom.squeeze(-1) > eps)
    axis_double2 = axis_double2 * valid2.unsqueeze(-1).to(axis_double2.dtype)
    return axis_double2, valid2


def pairwise_axis_angles(
    axis_double: torch.Tensor,  # [B, T, 2] double-angle vectors
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise undirected axis angles in ``[0, pi/2]``, shape ``[B, T, T]``.

    The angle between two double-angle unit vectors is ``2 * delta`` where
    ``delta`` is the axis (mod ``pi``) angle difference, so we halve the
    full angle. Vector magnitudes are normalised out, so the per-cell length
    does not affect the angle.

    Uses ``atan2`` instead of ``acos`` so gradients stay finite when axes are
    nearly parallel (``dot ~= +/-1``). Zero-norm axes (degenerate targets) are
    masked out so ``atan2(0, 0)`` never runs.
    """
    raw_norms = axis_double.norm(dim=-1, keepdim=True)
    valid = raw_norms.squeeze(-1) > eps
    norms = raw_norms.clamp_min(eps)
    hat = axis_double / norms
    hat = hat * valid.unsqueeze(-1).to(hat.dtype)

    dot = torch.matmul(hat, hat.transpose(-1, -2)).clamp(-1.0, 1.0)
    x = hat[..., 0]
    y = hat[..., 1]
    cross = x.unsqueeze(2) * y.unsqueeze(1) - y.unsqueeze(2) * x.unsqueeze(1)
    angles = 0.5 * torch.atan2(cross.abs(), dot.clamp_min(eps))

    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    return torch.where(pair_valid, angles, torch.zeros_like(angles))


def compute_directional_features(
    positions: torch.Tensor,         # [B, N, >=2]
    features: torch.Tensor,          # [B, N, F]
    mask: torch.Tensor,              # [B, N]
    target_idx: torch.Tensor,        # [B, T]
    target_valid: torch.Tensor,      # [B, T]
    neighbor_radius: float,
    coherence_radius: float,
    *,
    trans_beta: float = 1.0,
    soft_beta: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full directional pipeline for one side (pred or GT).

    Returns:
        lengths: ``[B, T]`` -- final per-cell axis length (orientation
            concentration after the second-radius smoothing).
        pair_angles: ``[B, T, T]`` -- modulo-``pi`` pairwise axis angles.
        valid: ``[B, T]`` bool mask for targets with a valid axis.
    """
    xy = positions[..., :2]
    axis1, valid1 = compute_orientation_axes(
        xy, features, mask, target_idx, target_valid,
        neighbor_radius,
        trans_beta=trans_beta,
        soft_beta=soft_beta,
        eps=eps,
    )
    target_pos = _gather_positions(xy, target_idx)
    axis2, valid2 = aggregate_axes_second_radius(
        target_pos, axis1, valid1,
        coherence_radius,
        soft_beta=soft_beta,
        eps=eps,
    )
    lengths = axis2.norm(dim=-1)
    pair_angles = pairwise_axis_angles(axis2, eps=eps)
    return lengths, pair_angles, valid2


def _upper_triangle_mask(T: int, device: torch.device) -> torch.Tensor:
    """``[T, T]`` bool mask with ``True`` on ``i < j``."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def _masked_mean(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(eps)


class DirectionalMetricLoss(nn.Module):
    """Match per-cell axis length and pairwise axis-angle structure pred vs. GT.

    Randomly subsamples ``n_target`` cells per sample, builds transcriptome-
    weighted local orientation axes via double-angle circular averaging,
    smooths them over a second radius, and penalises mismatches in per-cell
    axis length and all pairwise modulo-``pi`` axis-angle differences. Each
    pairwise term is weighted by the product of the two cells' GT lengths.
    """

    def __init__(
        self,
        n_target: int = 32,
        neighbor_radius: float = 0.01,
        coherence_radius: float = 0.02,
        trans_beta: float = 1.0,
        soft_beta: Optional[float] = 256.0,
        eps: float = 1e-6,
        length_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        min_valid_targets: int = 2,
    ) -> None:
        super().__init__()
        self.n_target = int(n_target)
        self.neighbor_radius = float(neighbor_radius)
        self.coherence_radius = float(coherence_radius)
        self.trans_beta = float(trans_beta)
        self.soft_beta = None if soft_beta is None else float(soft_beta)
        self.eps = float(eps)
        self.length_weight = float(length_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.min_valid_targets = int(min_valid_targets)
        self._last_loss: float = -1.0
        self._last_length: float = -1.0
        self._last_pairwise: float = -1.0

    def _pairwise_loss(
        self,
        pred_angles: torch.Tensor,
        gt_angles: torch.Tensor,
        gt_lengths: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Squared axis-angle error, weighted by the product of GT lengths."""
        T = pred_angles.shape[1]
        tri = _upper_triangle_mask(T, pred_angles.device)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        pair_valid = pair_valid & tri.unsqueeze(0)

        # Per-pair weight = gt_length_i * gt_length_j (detached, in [0, 1]).
        length_prod = (
            gt_lengths.unsqueeze(2) * gt_lengths.unsqueeze(1)
        ).detach()
        weight = pair_valid.to(pred_angles.dtype) * length_prod

        sq_err = (pred_angles - gt_angles.detach()).pow(2)
        return (sq_err * weight).sum() / weight.sum().clamp_min(self.eps)

    def _length_loss(
        self,
        pred_len: torch.Tensor,
        gt_len: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        sq_err = (pred_len - gt_len.detach()).pow(2)
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
            gt_len, gt_pairs, gt_valid = compute_directional_features(
                true_xy, node_features, node_mask, target_idx, target_valid,
                self.neighbor_radius, self.coherence_radius,
                trans_beta=self.trans_beta,
                soft_beta=self.soft_beta,
                eps=self.eps,
            )

        pred_len, pred_pairs, pred_valid = compute_directional_features(
            pred_xy, node_features, node_mask, target_idx, target_valid,
            self.neighbor_radius, self.coherence_radius,
            trans_beta=self.trans_beta,
            soft_beta=self.soft_beta,
            eps=self.eps,
        )

        valid = gt_valid & pred_valid & target_valid
        n_valid = valid.sum()

        if n_valid < self.min_valid_targets:
            zero = pred_xy.sum() * 0.0
            return zero, None

        length_loss = self._length_loss(pred_len, gt_len, valid)
        pair_loss = self._pairwise_loss(pred_pairs, gt_pairs, gt_len, valid)
        loss = self.length_weight * length_loss + self.pairwise_weight * pair_loss

        loss_val = float(loss.detach().item())
        self._last_loss = loss_val
        self._last_length = float(length_loss.detach().item())
        self._last_pairwise = float(pair_loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/directional_metric": loss_val,
                f"{prefix}/directional_length": self._last_length,
                f"{prefix}/directional_pairwise": self._last_pairwise,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        pass

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        to_log = {
            f"{epoch_prefix}/directional_metric": float(self._last_loss),
            f"{epoch_prefix}/directional_length": float(self._last_length),
            f"{epoch_prefix}/directional_pairwise": float(self._last_pairwise),
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
