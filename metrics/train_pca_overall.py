## Loss function for overall PCA of the spatial cells distributions regardless of transcriptome or types

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder
from metrics.gt_cache import cache_key_matches, gt_batch_cache_key, gt_row_cache_key


def compute_slide_pointcloud_pca_descriptors(
    positions: torch.Tensor,   # [N, >=2]
    mask: torch.Tensor,        # [N] (bool / 0-1)
    *,
    min_cells: int = 10,
    eps: float = 1e-6,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Anisotropy / omnivariance / linearity of the masked point cloud.

    For the 2x2 covariance of the (x, y) positions with eigenvalues
    ``lambda_1 >= lambda_2 >= 0``:

        anisotropy   = (lambda_1 - lambda_2) / (lambda_1 + eps)      in [0, 1)
        omnivariance = sqrt(lambda_1 * lambda_2)                     (area-like scale)
        linearity    = lambda_1 / (lambda_1 + lambda_2 + eps)        in [0.5, 1)

    A line-like cloud gives ``anisotropy -> 1``, ``linearity -> 1`` and
    ``omnivariance -> 0``; an isotropic blob gives ``anisotropy -> 0``,
    ``linearity -> 0.5`` and ``omnivariance -> variance``.

    The eigenvalues are computed in closed form so the descriptors are fully
    differentiable w.r.t. the (predicted) positions.

    Returns a tuple of three 0-dim tensors, or ``None`` when fewer than
    ``min_cells`` cells are present.
    """
    mask_b = mask.bool() if mask.dtype != torch.bool else mask
    pts = positions[mask_b][..., :2]                       # [M, 2]
    if pts.shape[0] < int(min_cells):
        return None

    mu = pts.mean(dim=0)
    d = pts - mu
    cxx = (d[:, 0] * d[:, 0]).mean()
    cyy = (d[:, 1] * d[:, 1]).mean()
    cxy = (d[:, 0] * d[:, 1]).mean()

    half = 0.5 * (cxx + cyy)
    det = cxx * cyy - cxy * cxy
    disc = torch.sqrt((half * half - det).clamp_min(eps))
    lambda1 = half + disc
    lambda2 = (half - disc).clamp_min(0.0)

    anisotropy = (lambda1 - lambda2) / (lambda1 + eps)
    omnivariance = torch.sqrt((lambda1 * lambda2).clamp_min(0.0) + eps)
    linearity = lambda1 / (lambda1 + lambda2 + eps)
    return anisotropy, omnivariance, linearity


class SlidePCALoss(nn.Module):
    """Whole-slide PCA shape descriptor loss.

    Treats all masked cells of a slide as a single 2D point cloud and
    penalises differences in three closed-form eigenvalue descriptors::

        L = anisotropy_weight    * |aniso_pred - aniso_gt|
          + omnivariance_weight  * |omni_pred  - omni_gt|
          + linearity_weight     * |lin_pred   - lin_gt|
    """

    def __init__(
        self,
        anisotropy_weight: float = 1.0,
        omnivariance_weight: float = 1.0,
        linearity_weight: float = 1.0,
        min_cells: int = 10,
        eps: float = 1e-6,
        cache_gt: bool = False,
    ) -> None:
        super().__init__()
        self.anisotropy_weight = float(anisotropy_weight)
        self.omnivariance_weight = float(omnivariance_weight)
        self.linearity_weight = float(linearity_weight)
        self.min_cells = int(min_cells)
        self.eps = float(eps)
        self.cache_gt = bool(cache_gt)
        self._gt_cache: Dict = {}
        self._last_loss: float = -1.0
        self._last_anisotropy: float = -1.0
        self._last_omnivariance: float = -1.0
        self._last_linearity: float = -1.0

    def clear_gt_cache(self) -> None:
        self._gt_cache.clear()

    def _cached_gt_descriptors(
        self,
        cell_id: Optional[torch.Tensor],
        batch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.cache_gt:
            return None
        cache_key = gt_row_cache_key(cell_id, batch_idx)
        if cache_key is None:
            return None
        cached = self._gt_cache.get(cache_key)
        if cached is None:
            return None
        cid_row = cell_id[batch_idx] if cell_id is not None and cell_id.dim() >= 2 else cell_id
        if not cache_key_matches(cid_row, cached.get("cell_id")):
            return None
        return (
            cached["anisotropy"].to(device=device, dtype=dtype),
            cached["omnivariance"].to(device=device, dtype=dtype),
            cached["linearity"].to(device=device, dtype=dtype),
        )

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask

        losses, aniso_terms, omni_terms, lin_terms = [], [], [], []
        for b in range(pred_positions.shape[0]):
            mask_b = node_mask[b]
            zero = pred_positions[b].sum() * 0.0

            gt_pca = self._cached_gt_descriptors(
                masked_true.cell_ID, b, pred_positions.device, pred_positions.dtype
            )
            if gt_pca is None:
                gt_pca = compute_slide_pointcloud_pca_descriptors(
                    true_positions[b], mask_b,
                    min_cells=self.min_cells, eps=self.eps,
                )
                if self.cache_gt and gt_pca is not None:
                    cache_key = gt_row_cache_key(masked_true.cell_ID, b)
                    if cache_key is not None:
                        cid_row = (
                            masked_true.cell_ID[b].detach()
                            if masked_true.cell_ID is not None
                            else None
                        )
                        gt_aniso, gt_omni, gt_lin = gt_pca
                        self._gt_cache[cache_key] = {
                            "cell_id": cid_row,
                            "anisotropy": gt_aniso.detach(),
                            "omnivariance": gt_omni.detach(),
                            "linearity": gt_lin.detach(),
                        }
            pred_pca = compute_slide_pointcloud_pca_descriptors(
                pred_positions[b], mask_b,
                min_cells=self.min_cells, eps=self.eps,
            )

            if gt_pca is None or pred_pca is None:
                losses.append(zero)
                aniso_terms.append(zero)
                omni_terms.append(zero)
                lin_terms.append(zero)
            else:
                gt_aniso, gt_omni, gt_lin = gt_pca
                pred_aniso, pred_omni, pred_lin = pred_pca
                aniso_t = (pred_aniso - gt_aniso.detach()).abs()
                omni_t = (pred_omni - gt_omni.detach()).abs()
                lin_t = (pred_lin - gt_lin.detach()).abs()
                total = (
                    self.anisotropy_weight * aniso_t
                    + self.omnivariance_weight * omni_t
                    + self.linearity_weight * lin_t
                )
                losses.append(total)
                aniso_terms.append(aniso_t)
                omni_terms.append(omni_t)
                lin_terms.append(lin_t)

        if not losses:
            zero = pred_positions.sum() * 0.0
            return zero, None

        loss = torch.stack(losses).mean()
        self._last_loss = float(loss.detach().item())
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
                f"{prefix}/slide_pca": float(loss.detach().item()),
                f"{prefix}/slide_pca_anisotropy": self._last_anisotropy,
                f"{prefix}/slide_pca_omnivariance": self._last_omnivariance,
                f"{prefix}/slide_pca_linearity": self._last_linearity,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    def reset(self) -> None:
        self.clear_gt_cache()

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        return {
            f"{epoch_prefix}/slide_pca": float(self._last_loss),
            f"{epoch_prefix}/slide_pca_anisotropy": float(self._last_anisotropy),
            f"{epoch_prefix}/slide_pca_omnivariance": float(self._last_omnivariance),
            f"{epoch_prefix}/slide_pca_linearity": float(self._last_linearity),
        }
