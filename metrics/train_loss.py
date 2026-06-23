## Master combined training loss
##
## Imports all modular sub-losses and combines them based on hyperparameters.
## Any sub-loss whose weight(s) are all zero is never instantiated or called.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder
from metrics.train_spatial_transcriptomics import MultiRadiusNeighborhoodLoss
from metrics.train_ch_overall import SlideCHLoss
from metrics.train_pca_overall import SlidePCALoss
from metrics.train_directional_metric import DirectionalMetricLoss


class CombinedTrainLoss(nn.Module):
    """Master loss: weighted sum of active sub-losses driven by config weights.

    Sub-losses:
        * **Neighborhood** (``neighborhood_weight``): multi-radius transcriptome
          RMSE + log-density across spatial annuli (``MultiRadiusNeighborhoodLoss``).
        * **CH AUC** (``ch_weight``): whole-slide Cahn-Hilliard energy-curve AUC
          (``SlideCHLoss``).
        * **PCA** (``pca_anisotropy_weight`` / ``pca_omnivariance_weight`` /
          ``pca_linearity_weight``): whole-slide eigenvalue shape descriptors
          (``SlidePCALoss``).
        * **Directional** (``directional_weight``): transcriptome-weighted local
          orientation coherence via double-angle statistics (``DirectionalMetricLoss``).

    Any sub-loss with all its weights equal to zero is never instantiated or
    called, saving both memory and compute. Config keys accept both the new
    names and the legacy ``slide_*`` prefixed names for backward compatibility.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        def _get(key: str, legacy: Optional[str] = None, default=0.0):
            v = getattr(cfg, key, None)
            if v is None and legacy is not None:
                v = getattr(cfg, legacy, None)
            return v if v is not None else default

        # ------------------------------------------------------------------ #
        # Neighborhood sub-loss
        # ------------------------------------------------------------------ #
        self.neighborhood_weight = float(_get(
            "neighborhood_weight", "slide_neighborhood_weight", 1.0
        ))
        if self.neighborhood_weight != 0.0:
            self.neighborhood: Optional[MultiRadiusNeighborhoodLoss] = (
                MultiRadiusNeighborhoodLoss(
                    radii=list(_get("multi_radius_radii", default=[0.02, 0.04, 0.08, 0.16])),
                    density_weight=float(_get("multi_radius_density_weight", default=1.0)),
                    global_transcriptome_weight=float(
                        _get("multi_radius_global_transcriptome_weight", default=0.0)
                    ),
                    loss_radius_scale=float(_get("multi_radius_loss_radius_scale", default=512.0)),
                    transcriptome_tolerance=_get("multi_radius_transcriptome_tolerance", default=0.05),
                    transcriptome_tolerance_gate_beta=_get(
                        "multi_radius_transcriptome_tolerance_soft_beta", default=256.0
                    ),
                    transcriptome_tolerance_warmup_epochs=int(
                        _get("multi_radius_transcriptome_tolerance_warmup_epochs", default=100)
                    ),
                    soft_beta=_get("multi_radius_soft_beta", default=None),
                    eps=float(_get("multi_radius_eps", default=1e-6)),
                    include_self=bool(_get("multi_radius_include_self", default=True)),
                )
            )
        else:
            self.neighborhood = None

        # ------------------------------------------------------------------ #
        # CH AUC sub-loss
        # ------------------------------------------------------------------ #
        self.ch_weight = float(_get("ch_weight", "slide_ch_auc_weight", 0.0))
        if self.ch_weight != 0.0:
            self.ch: Optional[SlideCHLoss] = SlideCHLoss(
                radii=list(_get("ch_radii", "slide_ch_radii",
                                [0.001, 0.0022857143, 0.0035714286, 0.0048571429,
                                 0.0061428571, 0.0074285714, 0.0087142857, 0.01])),
                grid_resolution=int(_get("ch_grid_resolution", "slide_ch_grid_resolution", 512)),
                kappa=float(_get("ch_kappa", "slide_ch_kappa", 0.5)),
                soft_max_beta=float(_get("ch_soft_max_beta", "slide_ch_soft_max_beta", 32.0)),
                support_factor=float(_get("ch_support_factor", "slide_ch_support_factor", 8)),
                landscape_chunk_size=int(
                    _get("ch_landscape_chunk_size", "slide_ch_landscape_chunk_size", 128)
                ),
                square_bbox=bool(_get("ch_square_bbox", "slide_ch_square_bbox", True)),
                margin=float(_get("ch_margin", "slide_ch_margin", 0.01)),
                eps=float(_get("ch_eps", "slide_ch_eps", 1e-6)),
                min_cells=int(_get("ch_min_cells", "slide_min_cells", 10)),
                cache_gt=bool(_get("ch_cache_gt", "slide_ch_cache_gt", True)),
            )
        else:
            self.ch = None

        # ------------------------------------------------------------------ #
        # PCA sub-loss
        # ------------------------------------------------------------------ #
        pca_aniso = float(_get("pca_anisotropy_weight", "slide_pca_anisotropy_weight", 0.0))
        pca_omni = float(_get("pca_omnivariance_weight", "slide_pca_omnivariance_weight", 0.0))
        pca_lin = float(_get("pca_linearity_weight", "slide_pca_linearity_weight", 0.0))
        if pca_aniso != 0.0 or pca_omni != 0.0 or pca_lin != 0.0:
            self.pca: Optional[SlidePCALoss] = SlidePCALoss(
                anisotropy_weight=pca_aniso,
                omnivariance_weight=pca_omni,
                linearity_weight=pca_lin,
                min_cells=int(_get("pca_min_cells", "slide_min_cells", 10)),
                eps=float(_get("pca_eps", "slide_ch_eps", 1e-6)),
            )
        else:
            self.pca = None

        # ------------------------------------------------------------------ #
        # Directional sub-loss
        # ------------------------------------------------------------------ #
        self.directional_weight = float(_get("directional_weight", default=0.0))
        if self.directional_weight != 0.0:
            self.directional: Optional[DirectionalMetricLoss] = DirectionalMetricLoss(
                n_target=int(_get("directional_n_target", default=32)),
                neighbor_radius=float(_get("directional_neighbor_radius", default=0.01)),
                coherence_radius=float(_get("directional_coherence_radius", default=0.02)),
                trans_beta=float(_get("directional_trans_beta", default=1.0)),
                soft_beta=_get("directional_soft_beta", default=256.0),
                length_weight=float(_get("directional_length_weight", default=1.0)),
                pairwise_weight=float(_get("directional_pairwise_weight", default=1.0)),
                min_valid_targets=int(_get("directional_min_valid_targets", default=2)),
            )
        else:
            self.directional = None

        # Shared caches for logging
        self._last_loss: float = -1.0
        self._last_neighborhood: float = -1.0
        self._last_ch: float = -1.0
        self._last_pca: float = -1.0
        self._last_directional: float = -1.0

    # ------------------------------------------------------------------ #
    # Epoch tracking (forwarded to neighborhood for tolerance warmup)
    # ------------------------------------------------------------------ #

    def set_current_epoch(self, epoch: int) -> None:
        if self.neighborhood is not None and hasattr(self.neighborhood, "set_current_epoch"):
            self.neighborhood.set_current_epoch(epoch)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        batch_idx: Optional[int] = None,
        **_unused: object,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:

        # We need a zero tensor to initialise the accumulator on the right
        # device without any forward pass committed yet.
        device = masked_pred.positions.device
        dtype = masked_pred.positions.dtype
        loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        to_log: Dict[str, float] = {}

        # ---- Neighborhood ----
        if self.neighborhood is not None and self.neighborhood_weight != 0.0:
            n_loss, n_log = self.neighborhood(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            loss = loss + self.neighborhood_weight * n_loss
            if train_stage:
                self._last_neighborhood = float(n_loss.detach().item())
            if log and n_log:
                to_log.update(n_log)

        # ---- CH AUC ----
        if self.ch is not None and self.ch_weight != 0.0:
            c_loss, c_log = self.ch(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
                batch_idx=batch_idx,
            )
            loss = loss + self.ch_weight * c_loss
            if train_stage:
                self._last_ch = float(c_loss.detach().item())
            if log and c_log:
                to_log.update(c_log)

        # ---- PCA ----
        if self.pca is not None:
            p_loss, p_log = self.pca(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            loss = loss + p_loss
            if train_stage:
                self._last_pca = float(p_loss.detach().item())
            if log and p_log:
                to_log.update(p_log)

        # ---- Directional ----
        if self.directional is not None and self.directional_weight != 0.0:
            d_loss, d_log = self.directional(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            loss = loss + self.directional_weight * d_loss
            if train_stage:
                self._last_directional = float(d_loss.detach().item())
            if log and d_log:
                to_log.update(d_log)

        if train_stage:
            self._last_loss = float(loss.detach().item())

        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log[f"{prefix}/combined"] = float(loss.detach().item())
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log if log else None

    # ------------------------------------------------------------------ #
    # Reset / epoch logging
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        if self.neighborhood is not None:
            self.neighborhood.reset()
        if self.ch is not None:
            self.ch.reset()
        if self.pca is not None:
            self.pca.reset()
        if self.directional is not None and hasattr(self.directional, "reset"):
            self.directional.reset()

    def log_epoch_metrics(self) -> Dict[str, float]:
        to_log: Dict[str, float] = {
            "train_epoch/combined": float(self._last_loss),
        }
        if self.neighborhood is not None:
            to_log.update(self.neighborhood.log_epoch_metrics())
        if self.ch is not None:
            to_log.update(self.ch.log_epoch_metrics())
        if self.pca is not None:
            to_log.update(self.pca.log_epoch_metrics())
        if self.directional is not None and hasattr(self.directional, "log_epoch_metrics"):
            to_log.update(self.directional.log_epoch_metrics())
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    def clear_gt_cache(self) -> None:
        """Clear GT caches in sub-losses that cache them (e.g. CH AUC)."""
        if self.ch is not None:
            self.ch.clear_gt_cache()
