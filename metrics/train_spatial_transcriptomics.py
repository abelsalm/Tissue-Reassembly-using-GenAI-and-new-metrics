## Loss function for spatial transcriptomics based on multiple neighborhoods around each cell

import math
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder


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