## Master combined training loss
##
## Imports all modular sub-losses and combines them based on hyperparameters.
## Any sub-loss whose global weight is zero is never instantiated or called.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder
from metrics.train_spatial_transcriptomics import MultiRadiusNeighborhoodLoss
from metrics.train_ch_overall import SlideCHLoss
from metrics.train_pca_overall import SlidePCALoss
from metrics.train_directional_metric import DirectionalMetricLoss
from metrics.train_mmds import SlideMMDLoss


class CombinedTrainLoss(nn.Module):
    """Master loss: weighted sum of active sub-losses driven by config weights.

    Global weights (all in ``cfg`` under ``train``):
        * **transcriptome_multi_radius_weight** — multi-radius neighborhood
          transcriptome RMSE + log-density (``MultiRadiusNeighborhoodLoss``).
        * **ch_weight** — whole-slide Cahn-Hilliard energy-curve AUC
          (``SlideCHLoss``).
        * **pca_weight** — whole-slide PCA shape descriptors (``SlidePCALoss``).
        * **directional_weight** — transcriptome-weighted local orientation
          coherence (``DirectionalMetricLoss``).
        * **mmd_weight** — per-class local-GT kernel MMD on cell positions
          (``SlideMMDLoss``).

    **other_trigger** — CH, PCA, directional, and MMD terms are forced to zero
    (no forward compute) until ``current_epoch >= other_trigger``; configured
    weights apply only from that epoch onward. ``0`` = active from epoch 0.

    A global weight of ``0`` disables that term entirely (no module init, no
    forward compute). Legacy ``slide_*`` / ``neighborhood_weight`` keys are
    still accepted as fallbacks.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        def _get(key: str, legacy: Optional[str] = None, default=0.0):
            v = getattr(cfg, key, None)
            if v is None and legacy is not None:
                v = getattr(cfg, legacy, None)
            return v if v is not None else default

        def _get_first(*keys: str, default=0.0):
            for key in keys:
                v = getattr(cfg, key, None)
                if v is not None:
                    return v
            return default

        # ------------------------------------------------------------------ #
        # Transcriptome multi-radius neighborhood sub-loss
        # ------------------------------------------------------------------ #
        self.transcriptome_multi_radius_weight = float(
            _get_first(
                "transcriptome_multi_radius_weight",
                "neighborhood_weight",
                "slide_neighborhood_weight",
                default=1.0,
            )
        )
        if self.transcriptome_multi_radius_weight != 0.0:
            self.neighborhood: Optional[MultiRadiusNeighborhoodLoss] = (
                MultiRadiusNeighborhoodLoss(
                    radii=list(_get("multi_radius_radii", default=[0.02, 0.04, 0.08, 0.16])),
                    avg_transcriptome_weight=float(
                        _get("multi_radius_avg_transcriptome_weight", default=1.0)
                    ),
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
                    cache_gt=bool(_get("neighborhood_cache_gt", default=True)),
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
        self.pca_weight = float(_get("pca_weight", default=0.0))
        if self.pca_weight != 0.0:
            self.pca: Optional[SlidePCALoss] = SlidePCALoss(
                anisotropy_weight=float(
                    _get("pca_anisotropy_weight", "slide_pca_anisotropy_weight", 1.0)
                ),
                omnivariance_weight=float(
                    _get("pca_omnivariance_weight", "slide_pca_omnivariance_weight", 1.0)
                ),
                linearity_weight=float(
                    _get("pca_linearity_weight", "slide_pca_linearity_weight", 1.0)
                ),
                min_cells=int(_get("pca_min_cells", "slide_min_cells", 10)),
                eps=float(_get("pca_eps", "slide_ch_eps", 1e-6)),
                cache_gt=bool(_get("pca_cache_gt", default=False)),
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
                density_length_gate=bool(_get("directional_density_length_gate", default=False)),
                density_length_beta=float(_get("directional_density_length_beta", default=4.0)),
                density_radius_gate=bool(_get("directional_density_radius_gate", default=False)),
                density_radius_beta=float(_get("directional_density_radius_beta", default=4.0)),
            )
        else:
            self.directional = None

        # ------------------------------------------------------------------ #
        # MMD sub-loss (per-class local-GT kernel MMD)
        # ------------------------------------------------------------------ #
        self.mmd_weight = float(_get("mmd_weight", default=0.0))
        if self.mmd_weight != 0.0:
            self.mmd: Optional[SlideMMDLoss] = SlideMMDLoss(cfg)
        else:
            self.mmd = None

        self.other_trigger = int(_get("other_trigger", default=0))
        self._current_epoch = 0

        # Shared caches for logging (raw unweighted sub-loss values)
        self._last_loss: float = -1.0
        self._last_transcriptome_multi_radius: float = -1.0
        self._last_ch: float = -1.0
        self._last_pca: float = -1.0
        self._last_directional: float = -1.0
        self._last_mmd: float = -1.0

    # ------------------------------------------------------------------ #
    # Epoch tracking (forwarded to neighborhood for tolerance warmup)
    # ------------------------------------------------------------------ #

    def _other_losses_active(self) -> bool:
        return self._current_epoch >= self.other_trigger

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
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

        device = masked_pred.positions.device
        dtype = masked_pred.positions.dtype
        # Seed from pred positions so the combined loss always carries a grad_fn,
        # even when every sub-loss is disabled / gated off (e.g. by other_trigger).
        # Otherwise loss.backward() raises "element 0 does not require grad".
        loss = masked_pred.positions.sum() * 0.0
        to_log: Dict[str, float] = {}
        prefix = "train_loss" if train_stage else "val_loss"

        # ---- Transcriptome multi-radius ----
        if self.neighborhood is not None and self.transcriptome_multi_radius_weight != 0.0:
            n_loss, n_log = self.neighborhood(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            weighted = self.transcriptome_multi_radius_weight * n_loss
            loss = loss + weighted
            self._last_transcriptome_multi_radius = float(n_loss.detach().item())
            if log:
                to_log[f"{prefix}/transcriptome_multi_radius_weighted"] = float(
                    weighted.detach().item()
                )
            if log and n_log:
                to_log.update(n_log)

        # ---- CH AUC ----
        if self.ch is not None and self.ch_weight != 0.0 and self._other_losses_active():
            c_loss, c_log = self.ch(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
                batch_idx=batch_idx,
            )
            weighted = self.ch_weight * c_loss
            loss = loss + weighted
            self._last_ch = float(c_loss.detach().item())
            if log:
                to_log[f"{prefix}/ch_auc_weighted"] = float(weighted.detach().item())
            if log and c_log:
                to_log.update(c_log)

        # ---- PCA ----
        if self.pca is not None and self.pca_weight != 0.0 and self._other_losses_active():
            p_loss, p_log = self.pca(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            weighted = self.pca_weight * p_loss
            loss = loss + weighted
            self._last_pca = float(p_loss.detach().item())
            if log:
                to_log[f"{prefix}/pca_weighted"] = float(weighted.detach().item())
            if log and p_log:
                to_log.update(p_log)

        # ---- Directional ----
        if (
            self.directional is not None
            and self.directional_weight != 0.0
            and self._other_losses_active()
        ):
            d_loss, d_log = self.directional(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            weighted = self.directional_weight * d_loss
            loss = loss + weighted
            self._last_directional = float(d_loss.detach().item())
            if log:
                to_log[f"{prefix}/directional_weighted"] = float(
                    weighted.detach().item()
                )
            if log and d_log:
                to_log.update(d_log)

        # ---- MMD ----
        if self.mmd is not None and self.mmd_weight != 0.0 and self._other_losses_active():
            m_loss, m_log = self.mmd(
                masked_pred, masked_true,
                train_stage=train_stage,
                log=log,
            )
            weighted = self.mmd_weight * m_loss
            loss = loss + weighted
            self._last_mmd = float(m_loss.detach().item())
            if log:
                to_log[f"{prefix}/mmd_weighted"] = float(weighted.detach().item())
            if log and m_log:
                to_log.update(m_log)

        self._last_loss = float(loss.detach().item())

        if log:
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
        if self.mmd is not None:
            self.mmd.reset()
        self.clear_gt_cache()

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        to_log: Dict[str, float] = {
            f"{epoch_prefix}/combined": float(self._last_loss),
        }
        if self.neighborhood is not None and self.transcriptome_multi_radius_weight != 0.0:
            to_log[f"{epoch_prefix}/transcriptome_multi_radius"] = float(
                self._last_transcriptome_multi_radius
            )
            to_log[f"{epoch_prefix}/transcriptome_multi_radius_weighted"] = float(
                self._last_transcriptome_multi_radius * self.transcriptome_multi_radius_weight
            )
            to_log.update(self.neighborhood.log_epoch_metrics(train_stage=train_stage))
        if self.ch is not None and self.ch_weight != 0.0:
            if self._other_losses_active():
                to_log[f"{epoch_prefix}/ch_auc"] = float(self._last_ch)
                to_log[f"{epoch_prefix}/ch_auc_weighted"] = float(
                    self._last_ch * self.ch_weight
                )
                to_log.update(self.ch.log_epoch_metrics(train_stage=train_stage))
            else:
                to_log[f"{epoch_prefix}/ch_auc_weighted"] = 0.0
        if self.pca is not None and self.pca_weight != 0.0:
            if self._other_losses_active():
                to_log[f"{epoch_prefix}/pca"] = float(self._last_pca)
                to_log[f"{epoch_prefix}/pca_weighted"] = float(
                    self._last_pca * self.pca_weight
                )
                to_log.update(self.pca.log_epoch_metrics(train_stage=train_stage))
            else:
                to_log[f"{epoch_prefix}/pca_weighted"] = 0.0
        if self.directional is not None and self.directional_weight != 0.0:
            if self._other_losses_active() and hasattr(
                self.directional, "log_epoch_metrics"
            ):
                to_log[f"{epoch_prefix}/directional"] = float(self._last_directional)
                to_log[f"{epoch_prefix}/directional_weighted"] = float(
                    self._last_directional * self.directional_weight
                )
                to_log.update(
                    self.directional.log_epoch_metrics(train_stage=train_stage)
                )
            else:
                to_log[f"{epoch_prefix}/directional_weighted"] = 0.0
        if self.mmd is not None and self.mmd_weight != 0.0:
            if self._other_losses_active():
                to_log[f"{epoch_prefix}/mmd"] = float(self._last_mmd)
                to_log[f"{epoch_prefix}/mmd_weighted"] = float(
                    self._last_mmd * self.mmd_weight
                )
                to_log.update(self.mmd.log_epoch_metrics(train_stage=train_stage))
            else:
                to_log[f"{epoch_prefix}/mmd_weighted"] = 0.0
        # Do not wandb.log here: this is called every training/val step so
        # Lightning can average with ``on_epoch=True``. WandB is updated once
        # per epoch from ``on_*_epoch_end``.
        return to_log

    def clear_gt_cache(self) -> None:
        """Clear cached GT values in all sub-losses (e.g. after warp / rechunk)."""
        if self.neighborhood is not None:
            self.neighborhood.clear_gt_cache()
        if self.ch is not None:
            self.ch.clear_gt_cache()
        if self.pca is not None:
            self.pca.clear_gt_cache()
        if self.mmd is not None:
            self.mmd.clear_gt_cache()
