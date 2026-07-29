## Loss function for spatial transcriptomics based on multiple neighborhoods around each cell

import math
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder
from metrics.gt_cache import cache_key_matches, gt_batch_cache_key


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


def soft_ranks(
    x: torch.Tensor,
    tau: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Differentiable soft ranks along the last axis (pairwise sigmoid).

    For each vector ``x[..., :]`` of length ``F``::

        r_i = sum_j sigmoid( (x_i - x_j) / tau )

    Self-comparisons contribute ``0.5``, so ranks lie in ``(0.5, F-0.5)`` and
    approach the usual midrank encoding as ``tau -> 0``.

    ``x`` may be ``[..., F]`` with arbitrary leading dims. When the leading
    product is large (many cells × shells), ranks are computed in chunks of
    ``chunk_size`` flattened leading rows so a full ``[..., F, F]`` tensor is
    never materialised for the whole batch at once.
    """
    if tau <= 0.0:
        raise ValueError(f"soft-rank temperature tau must be > 0, got {tau}")
    *lead, F = x.shape
    if F < 2:
        return x.new_zeros(x.shape)

    flat = x.reshape(-1, F)
    n_rows = flat.shape[0]
    inv_tau = 1.0 / float(tau)
    chunks = []
    for start in range(0, n_rows, int(chunk_size)):
        chunk = flat[start : start + chunk_size]          # [C, F]
        diff = chunk.unsqueeze(-1) - chunk.unsqueeze(-2)  # [C, F, F]
        chunks.append(torch.sigmoid(diff * inv_tau).sum(dim=-1))
    return torch.cat(chunks, dim=0).reshape(*lead, F)


def pearson_corr_last(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pearson correlation along the last axis; returns ``[...,]``."""
    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)
    num = (a_c * b_c).sum(dim=-1)
    den = (a_c.norm(dim=-1) * b_c.norm(dim=-1)).clamp_min(eps)
    return (num / den).clamp(-1.0, 1.0)


def soft_spearman_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float,
    *,
    eps: float = 1e-6,
    chunk_size: int = 64,
) -> torch.Tensor:
    """``1 - soft_Spearman(pred, gt)`` along the gene axis.

    Soft-ranks both profiles, then Pearson on those ranks. GT ranks are
    computed under ``no_grad`` (fixed profiles); gradients flow through the
    pred soft-ranks only. Shape ``pred``/``gt``: ``[..., F]`` → ``[...]``.
    """
    with torch.no_grad():
        gt_ranks = soft_ranks(gt, tau, chunk_size=chunk_size)
    pred_ranks = soft_ranks(pred, tau, chunk_size=chunk_size)
    rho = pearson_corr_last(pred_ranks, gt_ranks, eps=eps)
    return 1.0 - rho


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
    """Multi-radius neighborhood loss: transcriptome RMSE / soft-Spearman + density.

    Shared neighborhood pipeline (radii, soft_beta, include_self), then optional
    comparison heads gated by weight (0 = skip compute for that head):

      L = avg_transcriptome_weight * avg_rmse
        + avg_spearman_weight * avg_soft_spearman
        + density_weight * density_term
        + global_transcriptome_weight * global_rmse
        + global_spearman_weight * global_soft_spearman

    Soft Spearman is ``1 - Pearson(soft_rank(pred), soft_rank(gt))`` along genes
    (temperature ``spearman_tau``). Global heads are divided by
    ``loss_radius_scale * r``. RMSE heads use the GT-relative tolerance band.
    """

    def __init__(
        self,
        radii: Sequence[float] = (0.005, 0.01, 0.05, 0.1),
        avg_transcriptome_weight: float = 1.0,
        density_weight: float = 1.0,
        global_transcriptome_weight: float = 0.0,
        avg_spearman_weight: float = 0.0,
        global_spearman_weight: float = 0.0,
        spearman_tau: float = 0.1,
        spearman_chunk_size: int = 64,
        loss_radius_scale: float = 512.0,
        transcriptome_tolerance: Union[float, Sequence[float]] = 0.05,
        transcriptome_tolerance_gate_beta: Optional[float] = 256.0,
        transcriptome_tolerance_warmup_epochs: int = 100,
        soft_beta: Optional[float] = None,
        eps: float = 1e-6,
        include_self: bool = True,
        cache_gt: bool = False,
    ) -> None:
        super().__init__()
        if len(radii) < 1:
            raise ValueError("Need at least one radius.")
        # Sorted ascending: index k is the annulus with outer radius r_k;
        # per-shell tolerances are permuted to stay aligned with radii.
        self.radii, self.transcriptome_tolerance_per_shell = (
            _align_radii_and_transcriptome_tolerances(radii, transcriptome_tolerance)
        )
        self.avg_transcriptome_weight = float(avg_transcriptome_weight)
        self.density_weight = float(density_weight)
        self.global_transcriptome_weight = float(global_transcriptome_weight)
        self.avg_spearman_weight = float(avg_spearman_weight)
        self.global_spearman_weight = float(global_spearman_weight)
        self.spearman_tau = float(spearman_tau)
        self.spearman_chunk_size = int(spearman_chunk_size)
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
        self.cache_gt = bool(cache_gt)
        self._gt_cache: Dict = {}
        self.current_epoch: int = 0
        # Per-step caches surfaced through ``log_epoch_metrics``.
        self._last_loss: float = -1.0
        self._last_transcriptome: float = -1.0
        self._last_density: float = -1.0
        self._last_global_transcriptome: float = -1.0
        self._last_avg_spearman: float = -1.0
        self._last_global_spearman: float = -1.0

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
            return_neighborhood_sum=self._needs_neighborhood_sum(),
        )

    def _needs_neighborhood_sum(self) -> bool:
        return (
            self.global_transcriptome_weight > 0.0
            or self.global_spearman_weight > 0.0
        )

    def clear_gt_cache(self) -> None:
        self._gt_cache.clear()

    def _get_cached_gt(
        self,
        masked_true: DataHolder,
        device: torch.device,
        dtype: torch.dtype,
        use_global: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        if not self.cache_gt:
            return None
        cache_key = gt_batch_cache_key(masked_true.cell_ID)
        if cache_key is None:
            return None
        cached = self._gt_cache.get(cache_key)
        if cached is None:
            return None
        if not cache_key_matches(masked_true.cell_ID, cached.get("cell_id")):
            return None
        gt_avg = cached["avg"].to(device=device, dtype=dtype)
        gt_logd = cached["log_density"].to(device=device, dtype=dtype)
        gt_sum = cached.get("neighborhood_sum")
        if gt_sum is not None:
            gt_sum = gt_sum.to(device=device, dtype=dtype)
        elif use_global:
            sum_w = torch.exp(gt_logd) - self.eps
            gt_sum = gt_avg * sum_w.unsqueeze(-1).clamp_min(0.0)
        return gt_avg, gt_logd, gt_sum

    def _store_gt_cache(
        self,
        masked_true: DataHolder,
        gt_avg: torch.Tensor,
        gt_logd: torch.Tensor,
        gt_sum: Optional[torch.Tensor],
    ) -> None:
        cache_key = gt_batch_cache_key(masked_true.cell_ID)
        if cache_key is None:
            return
        entry = {
            "cell_id": masked_true.cell_ID.detach()
            if masked_true.cell_ID is not None
            else None,
            "avg": gt_avg.detach(),
            "log_density": gt_logd.detach(),
        }
        if gt_sum is not None:
            entry["neighborhood_sum"] = gt_sum.detach()
        self._gt_cache[cache_key] = entry

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

        use_avg_rmse = self.avg_transcriptome_weight > 0.0
        use_global_rmse = self.global_transcriptome_weight > 0.0
        use_avg_spearman = self.avg_spearman_weight > 0.0
        use_global_spearman = self.global_spearman_weight > 0.0
        use_global = self._needs_neighborhood_sum()
        use_any_rmse = use_avg_rmse or use_global_rmse

        cached_gt = self._get_cached_gt(
            masked_true, node_features.device, node_features.dtype, use_global
        )
        if cached_gt is not None:
            gt_avg, gt_logd, gt_sum = cached_gt
        elif (
            cached_gt_avg is not None and cached_gt_log_density is not None
        ):
            gt_avg = cached_gt_avg.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )
            gt_logd = cached_gt_log_density.detach().to(
                device=node_features.device, dtype=node_features.dtype
            )
            if use_global:
                if cached_gt_neighborhood_sum is None:
                    sum_w = torch.exp(gt_logd) - self.eps
                    gt_sum = gt_avg * sum_w.unsqueeze(-1).clamp_min(0.0)
                else:
                    gt_sum = cached_gt_neighborhood_sum.detach().to(
                        device=node_features.device, dtype=node_features.dtype
                    )
            else:
                gt_sum = None
        else:
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
            if self.cache_gt:
                self._store_gt_cache(masked_true, gt_avg, gt_logd, gt_sum)

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

        # ---- Density term: |log d_pred - log d_gt|, per cell per radius.
        per_cell_dens = (pred_logd - gt_logd).abs()    # [B, R, N]
        zero_bn = pred_avg.new_zeros(pred_avg.shape[:3])  # [B, R, N]

        # ---- RMSE heads (skipped entirely when both RMSE weights are 0).
        tolerance_forgiveness = self._tolerance_forgiveness()
        if use_any_rmse:
            shell_tolerance = self._shell_tolerance_tensor(
                device=pred_avg.device,
                dtype=pred_avg.dtype,
                n_shells=pred_avg.shape[1],
            )
        else:
            shell_tolerance = None

        if use_avg_rmse:
            assert shell_tolerance is not None
            sq_err = _gt_relative_band_squared_error(
                pred_avg,
                gt_avg,
                shell_tolerance,
                gate_beta=self.transcriptome_tolerance_gate_beta,
                forgiveness=tolerance_forgiveness,
                eps=self.eps,
            )                                              # [B, R, N, F]
            per_cell_rmse = torch.sqrt(sq_err.mean(dim=-1) + self.eps)
        else:
            per_cell_rmse = zero_bn

        if use_global_rmse:
            assert pred_sum is not None and gt_sum is not None
            assert shell_tolerance is not None
            sq_err_global = _gt_relative_band_squared_error(
                pred_sum,
                gt_sum,
                shell_tolerance,
                gate_beta=self.transcriptome_tolerance_gate_beta,
                forgiveness=tolerance_forgiveness,
                eps=self.eps,
            )                                              # [B, R, N, F]
            per_cell_global_rmse = torch.sqrt(
                sq_err_global.mean(dim=-1) + self.eps
            )                                              # [B, R, N]
        else:
            per_cell_global_rmse = zero_bn

        # ---- Soft-Spearman heads (same profiles; separate comparison).
        if use_avg_spearman:
            per_cell_avg_spearman = soft_spearman_distance(
                pred_avg,
                gt_avg,
                self.spearman_tau,
                eps=self.eps,
                chunk_size=self.spearman_chunk_size,
            )
        else:
            per_cell_avg_spearman = zero_bn

        if use_global_spearman:
            assert pred_sum is not None and gt_sum is not None
            per_cell_global_spearman = soft_spearman_distance(
                pred_sum,
                gt_sum,
                self.spearman_tau,
                eps=self.eps,
                chunk_size=self.spearman_chunk_size,
            )
        else:
            per_cell_global_spearman = zero_bn

        # Scale only global (sum) heads by loss_radius_scale * r.
        if use_global_rmse or use_global_spearman:
            radius_div = torch.tensor(
                [self.loss_radius_scale * r for r in self.radii],
                device=zero_bn.device,
                dtype=zero_bn.dtype,
            ).view(1, -1, 1)
            if use_global_rmse:
                per_cell_global_rmse = per_cell_global_rmse / radius_div
            if use_global_spearman:
                per_cell_global_spearman = per_cell_global_spearman / radius_div

        # ---- Aggregation: mean over valid cells -> mean over radii ->
        # mean over batch. Doing it in two steps (cells, then radii)
        # means a sample with many cells does not dominate per-radius
        # statistics.
        valid_i = node_mask.to(zero_bn.dtype).unsqueeze(1)   # [B, 1, N]
        n_valid = valid_i.sum(dim=-1).clamp_min(1.0)         # [B, 1]

        def _mean_per_r(per_cell: torch.Tensor) -> torch.Tensor:
            return (per_cell * valid_i).sum(dim=-1) / n_valid  # [B, R]

        rmse_per_r = _mean_per_r(per_cell_rmse)
        dens_per_r = _mean_per_r(per_cell_dens)
        global_rmse_per_r = _mean_per_r(per_cell_global_rmse)
        avg_spearman_per_r = _mean_per_r(per_cell_avg_spearman)
        global_spearman_per_r = _mean_per_r(per_cell_global_spearman)

        transcriptome_term = rmse_per_r.mean()
        density_term = dens_per_r.mean()
        global_term = global_rmse_per_r.mean()
        avg_spearman_term = avg_spearman_per_r.mean()
        global_spearman_term = global_spearman_per_r.mean()
        loss = (
            self.avg_transcriptome_weight * transcriptome_term
            + self.avg_spearman_weight * avg_spearman_term
            + self.density_weight * density_term
            + self.global_transcriptome_weight * global_term
            + self.global_spearman_weight * global_spearman_term
        )

        # Cache scalars for log_epoch_metrics (Lightning aggregates these
        # into a single per-epoch metric via on_epoch=True).
        self._last_transcriptome = float(transcriptome_term.detach().item())
        self._last_density = float(density_term.detach().item())
        self._last_global_transcriptome = float(global_term.detach().item())
        self._last_avg_spearman = float(avg_spearman_term.detach().item())
        self._last_global_spearman = float(global_spearman_term.detach().item())
        self._last_loss = float(loss.detach().item())

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/neighborhood_multi_radius": float(loss.detach().item()),
                f"{prefix}/neighborhood_multi_radius_transcriptome": float(
                    transcriptome_term.detach().item()
                ),
                f"{prefix}/neighborhood_multi_radius_global_transcriptome": float(
                    global_term.detach().item()
                ),
                f"{prefix}/neighborhood_multi_radius_avg_spearman": float(
                    avg_spearman_term.detach().item()
                ),
                f"{prefix}/neighborhood_multi_radius_global_spearman": float(
                    global_spearman_term.detach().item()
                ),
                f"{prefix}/tolerance_forgiveness": float(tolerance_forgiveness),
            }
            rmse_per_r_mean = rmse_per_r.mean(dim=0).detach()
            for ri, r in enumerate(self.radii):
                to_log[f"{prefix}/neighborhood_multi_radius_rmse_r{r:g}"] = float(
                    rmse_per_r_mean[ri].item()
                )
                to_log[
                    f"{prefix}/neighborhood_multi_radius_global_transcriptome_r{r:g}"
                ] = float(global_rmse_per_r.mean(dim=0).detach()[ri].item())
                if use_avg_spearman or use_global_spearman:
                    to_log[
                        f"{prefix}/neighborhood_multi_radius_avg_spearman_r{r:g}"
                    ] = float(avg_spearman_per_r.mean(dim=0).detach()[ri].item())
                    to_log[
                        f"{prefix}/neighborhood_multi_radius_global_spearman_r{r:g}"
                    ] = float(global_spearman_per_r.mean(dim=0).detach()[ri].item())
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """Clear GT cache; last-loss scalars are overwritten on next forward."""
        self.clear_gt_cache()

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        """Expose the last-step components under ``train_epoch/...`` or ``val_epoch/...`` keys.

        Per-step call from ``training_step_func`` + ``on_epoch=True`` in
        ``log_dict`` makes Lightning average each key over the epoch.
        Four keys: the combined loss plus the transcriptome / density /
        global-transcriptome breakdown -- useful for monitoring which side
        dominates.
        """
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        to_log = {
            f"{epoch_prefix}/whole_neighborhood_multi_radius": float(self._last_loss),
            f"{epoch_prefix}/neighborhood_multi_radius_avg": float(
                self._last_transcriptome
            ),
            f"{epoch_prefix}/neighborhood_multi_radius_global": float(
                self._last_global_transcriptome
            ),
            f"{epoch_prefix}/neighborhood_multi_radius_avg_spearman": float(
                self._last_avg_spearman
            ),
            f"{epoch_prefix}/neighborhood_multi_radius_global_spearman": float(
                self._last_global_spearman
            ),
        }
        # No per-step wandb.log — Lightning averages these at epoch end.
        return to_log