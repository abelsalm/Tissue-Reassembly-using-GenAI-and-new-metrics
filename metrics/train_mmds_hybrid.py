"""Hybrid, ragged training-time MMD.

The legacy implementation remains in :mod:`metrics.train_mmds` and is used as
the numerical reference.  This module provides an independently selectable
implementation with three goals:

* remove avoidable quadratic work and host synchronisations;
* batch exact local MMDs in narrow size buckets and reduce them by tiles;
* use incomplete U-statistics for large classes and whole-slide terms.

The local GT frame definition, IMQ kernels, sigma-squared band scaling and
slide/class reductions intentionally match the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from metrics.gt_cache import gt_batch_cache_key
from metrics.train_mmds import (
    COORD_HI,
    COORD_LO,
    ClassMMGTCache,
    LocalGTCache,
    MMDConfig,
    SlideMMDLoss,
    WholeSliceMMDCache,
    _aniso_axis_scales,
    _imq_from_sqdist,
    _mmd_config_from_cfg,
    _safe_unit_imq,
    box_penalty,
    class_anisotropy,
    class_top_eigvec,
    min_dist_repulsion_loss,
    pair_dist_match_loss,
    support_penalty,
)
from utils.data.dataholder import DataHolder


@dataclass(frozen=True)
class HybridMMDOptions:
    exact_cutoff: int = 1024
    exact_buckets: Tuple[int, ...] = (
        64, 96, 128, 192, 256, 384, 512, 768, 1024,
    )
    tile_size: int = 128
    bucket_max_classes: int = 16
    # Off by default: class-batch size K and bucket capacity both vary, so
    # torch.compile(dynamic=False) recompiles until the limit and can then
    # trigger CUDA device-side asserts on later steps.
    compile_exact: bool = False
    local_pairs: int = 32_768
    local_pairs_per_cell: int = 64
    local_pairs_max: int = 524_288
    gt_kernel_pairs: int = 131_072
    whole_slice_pairs: int = 131_072
    pair_dist_pairs: int = 8_192
    pair_dist_samples: int = 2_000
    statistic_pairs: int = 65_536
    support_queries: int = 256
    cache_gt: bool = False
    seed: int = 17


def _hybrid_options_from_cfg(cfg) -> HybridMMDOptions:
    def get(name, default):
        value = getattr(cfg, name, None)
        return default if value is None else value

    buckets = tuple(
        int(x)
        for x in get(
            "mmd_hybrid_exact_buckets",
            (64, 96, 128, 192, 256, 384, 512, 768, 1024),
        )
    )
    cutoff = int(get("mmd_hybrid_exact_cutoff", 1024))
    buckets = tuple(sorted({x for x in buckets if 2 <= x <= cutoff}))
    if not buckets or buckets[-1] < cutoff:
        buckets = (*buckets, cutoff)
    return HybridMMDOptions(
        exact_cutoff=cutoff,
        exact_buckets=buckets,
        tile_size=int(get("mmd_hybrid_tile_size", 128)),
        bucket_max_classes=int(get("mmd_hybrid_bucket_max_classes", 16)),
        compile_exact=bool(get("mmd_hybrid_compile", False)),
        local_pairs=int(get("mmd_hybrid_local_pairs", 32_768)),
        local_pairs_per_cell=int(
            get("mmd_hybrid_local_pairs_per_cell", 64)
        ),
        local_pairs_max=int(get("mmd_hybrid_local_pairs_max", 524_288)),
        gt_kernel_pairs=int(get("mmd_hybrid_gt_kernel_pairs", 131_072)),
        whole_slice_pairs=int(get("mmd_hybrid_whole_slice_pairs", 131_072)),
        pair_dist_pairs=int(get("mmd_hybrid_pair_dist_pairs", 8_192)),
        pair_dist_samples=int(get("mmd_hybrid_pair_dist_samples", 2_000)),
        statistic_pairs=int(get("mmd_hybrid_statistic_pairs", 65_536)),
        support_queries=int(get("mmd_hybrid_support_queries", 256)),
        cache_gt=bool(get("mmd_hybrid_cache_gt", False)),
        seed=int(get("mmd_hybrid_seed", 17)),
    )


def _sample_ordered_distinct_pairs(
    n: int,
    count: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform ordered pairs ``(i,j)`` with ``i != j`` in O(count)."""
    if n < 2 or count <= 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    i = torch.randint(n, (count,), device=device, generator=generator)
    j = torch.randint(n - 1, (count,), device=device, generator=generator)
    j = j + (j >= i).to(j.dtype)
    return i, j


def sample_pair_distances(
    points: torch.Tensor,
    max_samples: int,
    generator: Optional[torch.Generator] = None,
    exact_when_small: bool = True,
) -> torch.Tensor:
    """Sample the pair-distance distribution without constructing ``n×n``.

    When all unordered pairs fit under ``max_samples``, the result is exactly
    the legacy upper triangle.  Otherwise, ordered distinct pairs are sampled
    with replacement; symmetry of Euclidean distance makes their distribution
    identical to uniform unordered pairs.
    """
    n = int(points.shape[0])
    if n < 2 or max_samples <= 0:
        return points.new_zeros((0,))
    total = n * (n - 1) // 2
    if exact_when_small and total <= max_samples:
        ij = torch.triu_indices(n, n, offset=1, device=points.device)
        return torch.linalg.vector_norm(points[ij[0]] - points[ij[1]], dim=-1)
    i, j = _sample_ordered_distinct_pairs(
        n, min(max_samples, total) if exact_when_small else max_samples,
        points.device, generator,
    )
    return torch.linalg.vector_norm(points[i] - points[j], dim=-1)


def _stratified_skip_self_indices(
    unit_draw: torch.Tensor,
    source: torch.Tensor,
    n_targets: torch.Tensor,
) -> torch.Tensor:
    """Map a ``[0, 1)``-style draw onto ``{0..n-1}\\{source}``.

    ``unit_draw`` should be ``(strata + jitter) / r`` in ``[0, 1]`` (may
    land on 1 due to float32).  Without clamping, ``floor(u * (n-1))`` can
    become ``n-1`` and the skip-self bump then yields index ``n``.
    """
    n_m1 = (n_targets - 1).clamp_min(0)
    raw_max = (n_targets - 2).clamp_min(0)  # so raw+1 ≤ n-1
    raw = torch.floor(unit_draw * n_m1.to(unit_draw.dtype)).long()
    raw = torch.minimum(raw, raw_max.long())
    # If n < 2, keep raw (0); the bump would be out of range.
    bumped = raw + (raw >= source).long()
    return torch.where(n_targets.long() <= 1, raw, bumped)


def _stratified_inclusive_indices(
    unit_draw: torch.Tensor,
    n_targets: torch.Tensor,
) -> torch.Tensor:
    """Map a unit draw onto ``{0..n-1}`` with float-safe clamping."""
    n = n_targets.clamp_min(1)
    idx = torch.floor(unit_draw * n.to(unit_draw.dtype)).long()
    return torch.minimum(idx, (n - 1).long())


def _sample_pair_distances_stratified(
    points: torch.Tensor,
    target_samples: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Pair-distance samples with every point represented as a source."""
    n = int(points.shape[0])
    total = n * (n - 1) // 2
    if total <= target_samples:
        return sample_pair_distances(points, target_samples, generator)
    r = max(1, (target_samples + n - 1) // n)
    strata = torch.arange(
        r, device=points.device, dtype=torch.float32,
    )[None, :]
    jitter = torch.rand(
        (n, 1), device=points.device, generator=generator,
    )
    source = torch.arange(n, device=points.device)[:, None]
    unit = (strata + jitter) / float(r)
    n_t = torch.full((), n, device=points.device)
    target = _stratified_skip_self_indices(unit, source, n_t)
    return torch.linalg.vector_norm(
        points[:, None] - points[target], dim=-1,
    ).reshape(-1)


def _sampled_imq_selfterm_1d(
    values: torch.Tensor,
    sigmas: torch.Tensor,
    pair_count: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_zeros(())
    i, j = _sample_ordered_distinct_pairs(
        int(values.numel()), pair_count, values.device, generator,
    )
    d2 = (values[i] - values[j]).pow(2)
    kernels = _imq_from_sqdist(d2[:, None], sigmas).squeeze(-1)
    return (kernels.mean(-1) * sigmas.pow(2).clamp_min(1e-12)).mean()


def _sampled_imq_crossterm_1d(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sigmas: torch.Tensor,
    pair_count: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.new_zeros(())
    ip = torch.randint(
        pred.numel(), (pair_count,), device=pred.device, generator=generator,
    )
    ig = torch.randint(
        gt.numel(), (pair_count,), device=gt.device, generator=generator,
    )
    d2 = (pred[ip] - gt[ig]).pow(2)
    kernels = _imq_from_sqdist(d2[:, None], sigmas).squeeze(-1)
    return (kernels.mean(-1) * sigmas.pow(2).clamp_min(1e-12)).mean()


def sampled_pair_distance_mmd(
    X: torch.Tensor,
    gt_samples: torch.Tensor,
    gt_self_term: torch.Tensor,
    sigmas: torch.Tensor,
    distance_samples: int,
    kernel_pairs: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    pred = _sample_pair_distances_stratified(
        X, distance_samples, generator,
    )
    if pred.numel() < 2:
        return X.new_zeros(())
    pred_self = _sampled_imq_selfterm_1d(
        pred, sigmas, kernel_pairs, generator,
    )
    cross = _sampled_imq_crossterm_1d(
        pred, gt_samples, sigmas, kernel_pairs, generator,
    )
    return pred_self + gt_self_term - 2.0 * cross


def _class_moments_scatter(
    positions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class IDs, counts, barycenters and covariances without class loops."""
    pts = positions[mask]
    labs = labels[mask].long()
    classes, inverse, counts = torch.unique(
        labs, sorted=True, return_inverse=True, return_counts=True,
    )
    k = classes.numel()
    sums = positions.new_zeros((k, 2))
    sums.index_add_(0, inverse, pts)
    barycenters = sums / counts.to(positions.dtype).unsqueeze(-1)
    centered = pts - barycenters[inverse]
    outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)
    cov_sum = positions.new_zeros((k, 2, 2))
    cov_sum.index_add_(0, inverse, outer)
    cov = cov_sum / counts.to(positions.dtype).view(-1, 1, 1)
    return classes, counts, barycenters, cov


def _axis_landmarks_vectorized(
    barycenters: torch.Tensor,
    covariances: torch.Tensor,
    ref_dirs: torch.Tensor,
    counts: torch.Tensor,
    length_mult: float,
) -> torch.Tensor:
    """Vectorized equivalent of the legacy variable-length axis landmarks."""
    if barycenters.shape[0] == 0:
        return barycenters.new_zeros((0, 2))
    axis, eigenvalue = class_top_eigvec(covariances)
    sign = torch.sign((axis * ref_dirs).sum(-1, keepdim=True))
    axis = axis * torch.where(sign == 0, torch.ones_like(sign), sign)
    length = length_mult * torch.sqrt(eigenvalue.clamp_min(0.0))
    repeats = torch.sqrt(counts.float()).round().long().clamp_min(2)
    group = torch.repeat_interleave(
        torch.arange(repeats.numel(), device=repeats.device), repeats,
    )
    starts = torch.cumsum(repeats, dim=0) - repeats
    offset = torch.arange(group.numel(), device=group.device) - starts[group]
    t = -1.0 + 2.0 * offset.to(barycenters.dtype) / (repeats[group] - 1).to(
        barycenters.dtype
    )
    return (
        barycenters[group]
        + t.unsqueeze(-1)
        * length[group].unsqueeze(-1)
        * axis[group]
    )


def _procrustes_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    with_scale: bool,
    weights: Optional[torch.Tensor] = None,
    allow_reflection: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy Procrustes math while keeping scale on-device as a tensor."""
    X = source.detach().double()
    Y = target.detach().double()
    if weights is None:
        w = torch.ones(X.shape[0], dtype=torch.float64, device=X.device)
    else:
        w = weights.detach().double()
    w = w / w.sum().clamp_min(1e-12)
    mu_x = (w[:, None] * X).sum(0)
    mu_y = (w[:, None] * Y).sum(0)
    Xc, Yc = X - mu_x, Y - mu_y
    H = Xc.T @ (w[:, None] * Yc)
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    if not allow_reflection:
        reflected = torch.det(R) < 0
        correction = torch.eye(2, dtype=R.dtype, device=R.device)
        correction[-1, -1] = torch.where(
            reflected, correction.new_tensor(-1.0), correction.new_tensor(1.0),
        )
        R = Vh.T @ correction @ U.T
    if with_scale:
        denom = (w * Xc.pow(2).sum(-1)).sum().clamp_min(1e-12)
        scale = (w * (Yc * (Xc @ R.T)).sum(-1)).sum() / denom
    else:
        scale = X.new_ones(())
    translation = mu_y - scale * (R @ mu_x)
    dtype = source.dtype
    return R.to(dtype), translation.to(dtype), scale.to(dtype)


def _exact_local_bucket_tiled(
    X: torch.Tensor,
    G: torch.Tensor,
    mask: torch.Tensor,
    sigmas: torch.Tensor,
    ky_offdiag: torch.Tensor,
    V: torch.Tensor,
    stretch: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Exact local MMD for a padded size bucket, reduced tile by tile.

    Shapes: X/G [K,N,2], mask [K,N], sigmas/ky [K,S],
    V [K,S,N,2,2], stretch [K,S,N].
    """
    k, n, _ = X.shape
    s = sigmas.shape[1]
    sum_xx = X.new_zeros((k, s))
    sum_xg = X.new_zeros((k, s))
    all_idx = torch.arange(n, device=X.device)
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, :, None, None]
    scales = sigmas[:, :, None, None] * torch.stack(
        [stretch, stretch.clamp_min(1e-12).reciprocal()], dim=-1,
    )
    scales = torch.nan_to_num(
        scales, nan=1e-3, posinf=1.0, neginf=1e-3,
    ).clamp_min(1e-6)

    for start in range(0, n, tile_size):
        stop = min(start + tile_size, n)
        Xi = X[:, start:stop]
        row_mask = mask[:, start:stop]

        diff_xx = Xi[:, :, None, :] - X[:, None, :, :]
        d2_xx = diff_xx.pow(2).sum(-1)
        kxx = _safe_unit_imq(d2_xx[:, None] / c2)
        valid_xx = row_mask[:, :, None] & mask[:, None, :]
        valid_xx = valid_xx & (
            all_idx[start:stop, None] != all_idx[None, :]
        ).unsqueeze(0)
        sum_xx = sum_xx + (
            kxx * valid_xx[:, None].to(kxx.dtype)
        ).sum(dim=(2, 3))

        diff_xg = Xi[:, :, None, :] - G[:, None, :, :]
        proj = torch.einsum("ktnc,ksncd->kstnd", diff_xg, V)
        proj = proj / scales[:, :, None, :, :]
        d2_xg = proj.pow(2).sum(-1)
        kxg = _safe_unit_imq(d2_xg)
        valid_xg = row_mask[:, :, None] & mask[:, None, :]
        sum_xg = sum_xg + (
            kxg * valid_xg[:, None].to(kxg.dtype)
        ).sum(dim=(2, 3))

    counts = mask.sum(-1).to(X.dtype)
    denom_xx = (counts * (counts - 1)).clamp_min(1.0)[:, None]
    denom_xg = counts.pow(2).clamp_min(1.0)[:, None]
    per_band = sum_xx / denom_xx + ky_offdiag - 2.0 * sum_xg / denom_xg
    return (per_band * sigmas.pow(2).clamp_min(1e-12)).mean(-1)


def _sampled_local_batch(
    X: torch.Tensor,
    G: torch.Tensor,
    mask: torch.Tensor,
    sigmas: torch.Tensor,
    ky_offdiag: torch.Tensor,
    V: torch.Tensor,
    stretch: torch.Tensor,
    pairs_per_source: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stratified incomplete U-statistic for ragged large classes.

    Every active prediction is used as a source and receives the same number
    of sampled XX/XG interactions.  This remains unbiased while avoiding the
    sparse, high-variance gradients produced by globally sampled pairs.
    """
    k, n, _ = X.shape
    counts = mask.sum(-1).long()
    r = int(pairs_per_source)
    # One jittered target per index stratum is markedly lower-variance than
    # iid sampling for the narrow spatial bands used by MERFISH.
    jitter = torch.rand(
        (k, 2, n, 1),
        device=X.device,
        dtype=torch.float32,
        generator=generator,
    )
    strata = torch.arange(r, device=X.device, dtype=torch.float32)[None, None, None, :]
    source = torch.arange(n, device=X.device)[None, :, None]
    unit_x = (strata[:, 0] + jitter[:, 0]) / float(r)
    unit_g = (strata[:, 0] + jitter[:, 1]) / float(r)
    counts_b = counts[:, None, None]
    jx = _stratified_skip_self_indices(unit_x, source, counts_b)
    jg = _stratified_inclusive_indices(unit_g, counts_b)
    batch = torch.arange(k, device=X.device)[:, None, None]

    x_i = X[:, :, None, :]
    x_j = X[batch, jx]
    g_j = G[batch, jg]
    d2_xx = (x_i - x_j).pow(2).sum(-1)
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, :, None, None]
    source_mask = mask[:, None, :, None].to(X.dtype)
    denom = (counts * r).clamp_min(1).to(X.dtype)[:, None]
    kxx = (
        _safe_unit_imq(d2_xx[:, None] / c2) * source_mask
    ).sum(dim=(2, 3)) / denom

    diff = x_i - g_j
    s = sigmas.shape[1]
    batch_s = torch.arange(k, device=X.device)[:, None, None, None]
    band = torch.arange(s, device=X.device)[None, :, None, None]
    target = jg[:, None].expand(k, s, n, r)
    Vj = V[batch_s, band, target]
    proj = torch.einsum("knrc,ksnrcd->ksnrd", diff, Vj)
    stretch_j = stretch[batch_s, band, target]
    scales = sigmas[:, :, None, None, None] * torch.stack(
        [stretch_j, stretch_j.clamp_min(1e-12).reciprocal()], dim=-1,
    )
    kxg = (
        _safe_unit_imq(
            (proj / scales.clamp_min(1e-6)).pow(2).sum(-1)
        )
        * source_mask
    ).sum(dim=(2, 3)) / denom
    per_band = kxx + ky_offdiag - 2.0 * kxg
    return (per_band * sigmas.pow(2).clamp_min(1e-12)).mean(-1)


def _sampled_local_gt_offdiag(
    G: torch.Tensor,
    sigmas: torch.Tensor,
    V: torch.Tensor,
    stretch: torch.Tensor,
    pair_count: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Unbiased one-sided GT kernel mean; symmetrisation has the same mean."""
    i, j = _sample_ordered_distinct_pairs(
        int(G.shape[0]), pair_count, G.device, generator,
    )
    diff = G[i] - G[j]
    Vj = V[:, j]
    proj = torch.einsum("qc,sqcd->sqd", diff, Vj)
    scales = _aniso_axis_scales(sigmas, stretch[:, j])
    return _safe_unit_imq((proj / scales).pow(2).sum(-1)).mean(-1)


def _exact_local_gt_offdiag_tiled(
    G: torch.Tensor,
    sigmas: torch.Tensor,
    V: torch.Tensor,
    stretch: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    n = G.shape[0]
    total = G.new_zeros((sigmas.numel(),))
    scales = _aniso_axis_scales(sigmas, stretch)
    all_idx = torch.arange(n, device=G.device)
    for start in range(0, n, tile_size):
        stop = min(start + tile_size, n)
        diff = G[start:stop, None, :] - G[None, :, :]
        proj = torch.einsum("tnc,sncd->stnd", diff, V)
        kernel = _safe_unit_imq(
            (proj / scales[:, None, :, :]).pow(2).sum(-1)
        )
        valid = all_idx[start:stop, None] != all_idx[None, :]
        total = total + (kernel * valid[None].to(kernel.dtype)).sum((1, 2))
    return total / max(n * (n - 1), 1)


def _local_gt_metrics_vectorized(
    G: torch.Tensor,
    sigmas: torch.Tensor,
    cfg: MMDConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all sigma-band local PCA frames from one shared distance matrix."""
    D = torch.cdist(G, G)
    scaled = sigmas / cfg.pca_scale_factor
    tau = cfg.sigmoid_tau_frac * scaled
    W = torch.sigmoid(
        (scaled[:, None, None] - D[None])
        / (tau[:, None, None] + 1e-8)
    )
    sums = W.sum(-1)
    degenerate = sums < 1e-6
    weights = W / sums.clamp_min(1e-6).unsqueeze(-1)
    mu = torch.einsum("sij,jc->sic", weights, G)
    # Use centered second moments rather than E[xx]-E[x]E[x].  The latter
    # suffers severe cancellation for the tiny local covariances and changes
    # anisotropy ratios enough to diverge from the baseline.
    diff = G[None, None, :, :] - mu[:, :, None, :]
    cov = torch.einsum("sij,sijc,sijd->sicd", weights, diff, diff)
    out_dtype = G.dtype
    eigenvalues, eigenvectors = torch.linalg.eigh(cov.float())
    lam1 = eigenvalues[..., 1].clamp_min(1e-8)
    lam2 = eigenvalues[..., 0].clamp_min(1e-8)
    ratio = torch.sqrt(lam1 / lam2).clamp(1.0, 10.0).to(out_dtype)
    V = eigenvectors[..., :, [1, 0]].to(out_dtype)
    bad = (
        degenerate
        | (~torch.isfinite(V).all(dim=(-2, -1)))
        | (~torch.isfinite(ratio))
    )
    eye = torch.eye(2, device=G.device, dtype=out_dtype).expand_as(V)
    V = torch.where(bad[..., None, None], eye, V)
    ratio = torch.where(bad, torch.ones_like(ratio), ratio)
    return D, V, ratio.pow(cfg.local_aniso_stretch_power)


def _build_hybrid_class_cache(
    G: torch.Tensor,
    cfg: MMDConfig,
    options: HybridMMDOptions,
    generator: Optional[torch.Generator],
) -> Optional[ClassMMGTCache]:
    """Build one GT cache while sharing pairwise work across sigma bands."""
    n = int(G.shape[0])
    if n < 2:
        return None
    with torch.no_grad():
        stat_dists = sample_pair_distances(
            G, options.statistic_pairs, generator, exact_when_small=True,
        )
        median = stat_dists.median() if stat_dists.numel() else G.new_tensor(1e-2)
        median = torch.nan_to_num(median, nan=1e-2, posinf=1e-2).clamp_min(1e-6)
        mults = torch.as_tensor(
            cfg.spatial_band_mults, dtype=G.dtype, device=G.device,
        )
        sigmas = mults * median
        D, V, stretch = _local_gt_metrics_vectorized(G, sigmas, cfg)
        if n <= options.exact_cutoff:
            ky = _exact_local_gt_offdiag_tiled(
                G, sigmas, V, stretch, options.tile_size,
            )
        else:
            ky = _sampled_local_gt_offdiag(
                G, sigmas, V, stretch, options.gt_kernel_pairs, generator,
            )
        local = LocalGTCache(
            sigmas=sigmas.detach(),
            ky_offdiag=ky.detach(),
            V=V.detach(),
            stretch=stretch.detach(),
        )

        D_nn = D.clone()
        D_nn.fill_diagonal_(float("inf"))
        support_margin = (
            cfg.support_margin_mult * D_nn.min(-1).values.median()
        ).detach()
        if cfg.min_dist_weight:
            if cfg.min_dist_thresh is None:
                finite = D_nn[torch.isfinite(D_nn)]
                md_thresh = torch.quantile(
                    finite.float(), cfg.min_dist_quantile,
                ).to(G.dtype)
            else:
                md_thresh = G.new_tensor(cfg.min_dist_thresh)
        else:
            md_thresh = G.new_zeros(())

        if cfg.pair_dist_mmd_weight:
            gt_samples = sample_pair_distances(
                G, options.pair_dist_samples, generator, exact_when_small=True,
            ).detach()
            med_dist = stat_dists.median() if stat_dists.numel() else median
            dist_sigmas = torch.as_tensor(
                cfg.pair_dist_mmd_band_mults,
                dtype=G.dtype,
                device=G.device,
            ) * med_dist
            gt_self = _sampled_imq_selfterm_1d(
                gt_samples, dist_sigmas, options.pair_dist_pairs, generator,
            ).detach()
        else:
            gt_samples = gt_self = dist_sigmas = None

        gt_mean = stat_dists.mean().detach() if cfg.pair_dist_weight else G.new_zeros(())
    return ClassMMGTCache(
        G=G.detach(),
        gt_cache=local,
        gt_mean_pd=gt_mean,
        md_thresh=md_thresh.detach(),
        support_margin=support_margin,
        dist_sigmas=dist_sigmas,
        pdm_gt_samples=gt_samples,
        pdm_gt_self=gt_self,
    )


def _exact_isotropic_offdiag_tiled(
    G: torch.Tensor,
    sigmas: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    n = G.shape[0]
    total = G.new_zeros((sigmas.numel(),))
    all_idx = torch.arange(n, device=G.device)
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, None, None]
    for start in range(0, n, tile_size):
        stop = min(start + tile_size, n)
        d2 = (G[start:stop, None] - G[None]).pow(2).sum(-1)
        kernel = _safe_unit_imq(d2[None] / c2)
        valid = all_idx[start:stop, None] != all_idx[None]
        total = total + (kernel * valid[None].to(kernel.dtype)).sum((1, 2))
    return total / max(n * (n - 1), 1)


def _sampled_isotropic_offdiag(
    G: torch.Tensor,
    sigmas: torch.Tensor,
    pair_count: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    i, j = _sample_ordered_distinct_pairs(
        int(G.shape[0]), pair_count, G.device, generator,
    )
    d2 = (G[i] - G[j]).pow(2).sum(-1)
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, None]
    return _safe_unit_imq(d2[None] / c2).mean(-1)


def _build_hybrid_whole_cache(
    G: torch.Tensor,
    band_mults: Sequence[float],
    options: HybridMMDOptions,
    generator: Optional[torch.Generator],
) -> Optional[WholeSliceMMDCache]:
    if G.shape[0] < 2:
        return None
    distances = sample_pair_distances(
        G, options.statistic_pairs, generator, exact_when_small=True,
    )
    median = distances.median() if distances.numel() else G.new_tensor(1e-2)
    median = torch.nan_to_num(median, nan=1e-2, posinf=1e-2).clamp_min(1e-6)
    sigmas = torch.as_tensor(
        band_mults, dtype=G.dtype, device=G.device,
    ) * median
    if G.shape[0] <= options.exact_cutoff:
        ky = _exact_isotropic_offdiag_tiled(G, sigmas, options.tile_size)
    else:
        ky = _sampled_isotropic_offdiag(
            G, sigmas, options.whole_slice_pairs, generator,
        )
    return WholeSliceMMDCache(
        G=G.detach(), sigmas=sigmas.detach(), ky_offdiag=ky.detach(),
    )


def _sampled_isotropic_mmd(
    X: torch.Tensor,
    cache: WholeSliceMMDCache,
    pair_count: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    n, m = X.shape[0], cache.G.shape[0]
    if n < 2 or m < 1:
        return X.new_zeros(())
    r = max(1, (pair_count + n - 1) // n)
    strata = torch.arange(r, device=X.device, dtype=torch.float32)[None, :]
    jitter = torch.rand(
        (2, n, 1), device=X.device, generator=generator,
    )
    source = torch.arange(n, device=X.device)[:, None]
    unit_x = (strata + jitter[0]) / float(r)
    unit_g = (strata + jitter[1]) / float(r)
    n_t = torch.full((), n, device=X.device)
    m_t = torch.full((), m, device=X.device)
    j = _stratified_skip_self_indices(unit_x, source, n_t)
    jg = _stratified_inclusive_indices(unit_g, m_t)
    d2_xx = (X[:, None] - X[j]).pow(2).sum(-1)
    d2_xg = (X[:, None] - cache.G[jg]).pow(2).sum(-1)
    c2 = cache.sigmas.pow(2).clamp_min(1e-12)[:, None, None]
    kxx = _safe_unit_imq(d2_xx[None] / c2).mean(dim=(1, 2))
    kxg = _safe_unit_imq(d2_xg[None] / c2).mean(dim=(1, 2))
    per_band = kxx + cache.ky_offdiag - 2.0 * kxg
    return (per_band * cache.sigmas.pow(2).clamp_min(1e-12)).mean()


def _pad_first_dimension(
    tensors: Sequence[torch.Tensor],
    capacity: int,
) -> torch.Tensor:
    padded = pad_sequence(list(tensors), batch_first=True)
    if padded.shape[1] > capacity:
        raise ValueError(f"bucket capacity {capacity} < tensor size {padded.shape[1]}")
    pad_shape = [0, 0] * (padded.ndim - 2) + [0, capacity - padded.shape[1]]
    return F.pad(padded, tuple(pad_shape))


@dataclass
class _ClassSegment:
    slide: int
    class_id: int
    X: torch.Tensor
    cache: ClassMMGTCache
    seed_offset: int

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


class SlideMMDLossHybrid(SlideMMDLoss):
    """Drop-in hybrid implementation; the legacy class remains untouched."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = _mmd_config_from_cfg(cfg)
        self.options = _hybrid_options_from_cfg(cfg)
        self.cache_gt = self.options.cache_gt
        self._current_epoch = 0
        self._forward_index = 0
        self._last_metric_tensors: Dict[str, torch.Tensor] = {}
        self._hybrid_gt_cache: Dict[Tuple, object] = {}
        self._compiled_exact = None
        if self.options.compile_exact and hasattr(torch, "compile"):
            # dynamic=True: K (classes/chunk) and N (bucket capacity) both vary.
            # dynamic=False previously hit dynamo's recompile_limit and then
            # raised CUDA device-side asserts mid-training.
            self._compiled_exact = torch.compile(
                _exact_local_bucket_tiled,
                dynamic=True,
                fullgraph=False,
            )

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        self._forward_index = 0

    def reset(self) -> None:
        super().reset()
        self._forward_index = 0
        self._last_metric_tensors.clear()

    def clear_gt_cache(self) -> None:
        super().clear_gt_cache()
        if hasattr(self, "_hybrid_gt_cache"):
            self._hybrid_gt_cache.clear()

    def _sync_last_metrics(self) -> None:
        """Materialize logging scalars only at an explicit logging boundary."""
        if not self._last_metric_tensors:
            return
        values = {
            name: float(value.item())
            for name, value in self._last_metric_tensors.items()
        }
        self._last_loss = values["loss"]
        self._last_local = values["local"]
        self._last_pair_dist = values["pair_dist"]
        self._last_min_dist = values["min_dist"]
        self._last_pair_dist_mmd = values["pair_dist_mmd"]
        self._last_whole_slice = values["whole_slice"]

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        self._sync_last_metrics()
        return super().log_epoch_metrics(train_stage=train_stage)

    def _generator(
        self,
        device: torch.device,
        slide: int,
        class_id: int,
        term: int,
    ) -> torch.Generator:
        seed = (
            self.options.seed
            + 1_000_003 * self._current_epoch
            + 10_007 * self._forward_index
            + 101 * slide
            + 17 * class_id
            + term
        ) % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _aligned_pred_xy_vectorized(
        self,
        pred_xy: torch.Tensor,
        true_xy: torch.Tensor,
        labels: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.procrustes_align or labels is None:
            return pred_xy
        classes, counts, P_bc, P_cov = _class_moments_scatter(
            pred_xy, labels, mask,
        )
        _, gt_counts, G_bc, G_cov = _class_moments_scatter(
            true_xy, labels, mask,
        )
        if classes.numel() < 2:
            return pred_xy
        if not torch.equal(counts, gt_counts):
            raise RuntimeError("pred/GT class counts differ in hybrid Procrustes")

        P_all, G_all, weights = P_bc, G_bc, None
        if self.cfg.procrustes_axis_align:
            gate = counts >= self.cfg.procrustes_axis_min_cells
            gate = gate & (
                class_anisotropy(G_cov)
                >= self.cfg.procrustes_axis_min_anisotropy
            )
            if torch.any(gate):
                R0, _, _ = _procrustes_tensor(
                    P_bc, G_bc, with_scale=False,
                    allow_reflection=self.cfg.procrustes_allow_reflection,
                )
                gt_axis, _ = class_top_eigvec(G_cov)
                ref_dirs = gt_axis @ R0
                P_axis = _axis_landmarks_vectorized(
                    P_bc[gate], P_cov[gate], ref_dirs[gate], counts[gate],
                    self.cfg.procrustes_axis_length_mult,
                )
                G_axis = _axis_landmarks_vectorized(
                    G_bc[gate], G_cov[gate], gt_axis[gate], counts[gate],
                    self.cfg.procrustes_axis_length_mult,
                )
                P_all = torch.cat([P_bc, P_axis])
                G_all = torch.cat([G_bc, G_axis])
                weights = torch.cat(
                    [
                        torch.ones(P_bc.shape[0], device=pred_xy.device),
                        torch.full(
                            (P_axis.shape[0],),
                            self.cfg.procrustes_axis_weight,
                            device=pred_xy.device,
                        ),
                    ]
                )
        R, t, scale = _procrustes_tensor(
            P_all, G_all, self.cfg.procrustes_with_scale, weights,
            allow_reflection=self.cfg.procrustes_allow_reflection,
        )
        return scale * (pred_xy @ R.T) + t

    def _bucket_capacity(self, size: int) -> int:
        for capacity in self.options.exact_buckets:
            if size <= capacity:
                return capacity
        return self.options.exact_cutoff

    def _exact_bucket_losses(
        self,
        segments: Sequence[_ClassSegment],
    ) -> Dict[int, torch.Tensor]:
        grouped: Dict[int, List[Tuple[int, _ClassSegment]]] = {}
        for index, segment in enumerate(segments):
            if segment.size <= self.options.exact_cutoff:
                grouped.setdefault(self._bucket_capacity(segment.size), []).append(
                    (index, segment)
                )
        result: Dict[int, torch.Tensor] = {}
        kernel = self._compiled_exact or _exact_local_bucket_tiled
        max_k = self.options.bucket_max_classes
        for capacity, entries in grouped.items():
            for begin in range(0, len(entries), max_k):
                chunk = entries[begin:begin + max_k]
                indices = [entry[0] for entry in chunk]
                items = [entry[1] for entry in chunk]
                counts = torch.tensor(
                    [item.size for item in items],
                    device=items[0].X.device,
                    dtype=torch.long,
                )
                mask = (
                    torch.arange(capacity, device=counts.device)[None]
                    < counts[:, None]
                )
                X = _pad_first_dimension([item.X for item in items], capacity)
                G = _pad_first_dimension(
                    [item.cache.G for item in items], capacity,
                )
                V = _pad_first_dimension(
                    [
                        item.cache.gt_cache.V.permute(1, 0, 2, 3)
                        for item in items
                    ],
                    capacity,
                ).permute(0, 2, 1, 3, 4)
                stretch = _pad_first_dimension(
                    [
                        item.cache.gt_cache.stretch.T
                        for item in items
                    ],
                    capacity,
                ).permute(0, 2, 1)
                sigmas = torch.stack(
                    [item.cache.gt_cache.sigmas for item in items],
                )
                ky = torch.stack(
                    [item.cache.gt_cache.ky_offdiag for item in items],
                )
                losses = kernel(
                    X, G, mask, sigmas, ky, V, stretch,
                    self.options.tile_size,
                )
                result.update(
                    {index: losses[pos] for pos, index in enumerate(indices)}
                )
        return result

    def _sampled_large_losses(
        self,
        segments: Sequence[_ClassSegment],
    ) -> Dict[int, torch.Tensor]:
        entries = [
            (index, segment)
            for index, segment in enumerate(segments)
            if segment.size > self.options.exact_cutoff
        ]
        if not entries:
            return {}
        result: Dict[int, torch.Tensor] = {}
        max_k = self.options.bucket_max_classes
        for begin in range(0, len(entries), max_k):
            chunk = entries[begin:begin + max_k]
            indices = [entry[0] for entry in chunk]
            items = [entry[1] for entry in chunk]
            capacity = max(item.size for item in items)
            counts = torch.tensor(
                [item.size for item in items],
                device=items[0].X.device,
                dtype=torch.long,
            )
            mask = (
                torch.arange(capacity, device=counts.device)[None]
                < counts[:, None]
            )
            X = _pad_first_dimension([item.X for item in items], capacity)
            G = _pad_first_dimension([item.cache.G for item in items], capacity)
            V = _pad_first_dimension(
                [
                    item.cache.gt_cache.V.permute(1, 0, 2, 3)
                    for item in items
                ],
                capacity,
            ).permute(0, 2, 1, 3, 4)
            stretch = _pad_first_dimension(
                [item.cache.gt_cache.stretch.T for item in items],
                capacity,
            ).permute(0, 2, 1)
            sigmas = torch.stack(
                [item.cache.gt_cache.sigmas for item in items],
            )
            ky = torch.stack(
                [item.cache.gt_cache.ky_offdiag for item in items],
            )
            minimum_per_source = (
                self.options.local_pairs + capacity - 1
            ) // capacity
            maximum_per_source = max(
                1, self.options.local_pairs_max // capacity,
            )
            pairs_per_source = min(
                max(
                    self.options.local_pairs_per_cell,
                    minimum_per_source,
                ),
                maximum_per_source,
            )
            losses = _sampled_local_batch(
                X, G, mask, sigmas, ky, V, stretch,
                pairs_per_source,
                self._generator(
                    X.device,
                    items[0].slide,
                    items[0].class_id,
                    300 + begin,
                ),
            )
            result.update(
                {index: losses[pos] for pos, index in enumerate(indices)}
            )
        return result

    def _auxiliary_loss(
        self,
        segment: _ClassSegment,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X, cc = segment.X, segment.cache
        generator = self._generator(
            X.device, segment.slide, segment.class_id, 500 + segment.seed_offset,
        )
        zero = X.new_zeros(())
        if self.cfg.pair_dist_weight:
            if segment.size <= self.options.exact_cutoff:
                pd_loss = pair_dist_match_loss(X, cc.gt_mean_pd)
            else:
                pred_dist = sample_pair_distances(
                    X, self.options.statistic_pairs, generator,
                    exact_when_small=False,
                )
                pd_loss = (pred_dist.mean() - cc.gt_mean_pd).pow(2)
        else:
            pd_loss = zero

        if self.cfg.min_dist_weight:
            md_loss = min_dist_repulsion_loss(X, cc.md_thresh)
        else:
            md_loss = zero

        if self.cfg.pair_dist_mmd_weight:
            pdm_loss = sampled_pair_distance_mmd(
                X,
                cc.pdm_gt_samples,
                cc.pdm_gt_self,
                cc.dist_sigmas,
                self.options.pair_dist_samples,
                self.options.pair_dist_pairs,
                generator,
            )
        else:
            pdm_loss = zero

        loss = (
            self.cfg.pair_dist_weight * pd_loss
            + self.cfg.min_dist_weight * md_loss
            + self.cfg.pair_dist_mmd_weight * pdm_loss
        )
        if self.cfg.mmd_support:
            if segment.size <= self.options.exact_cutoff:
                support = support_penalty(X, cc.G, cc.support_margin)
            else:
                q = min(self.options.support_queries, segment.size)
                idx = torch.randint(
                    segment.size, (q,), device=X.device, generator=generator,
                )
                support = support_penalty(X[idx], cc.G, cc.support_margin)
            loss = loss + self.cfg.mmd_support * support
        if self.cfg.mmd_box:
            loss = loss + self.cfg.mmd_box * box_penalty(X)
        return loss, pd_loss, md_loss, pdm_loss

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = True,
        **_unused: object,
    ):
        pred_positions = masked_pred.positions
        true_positions = masked_true.positions
        node_mask = masked_true.node_mask.bool()
        cell_class = masked_true.cell_class
        cell_id = masked_true.cell_ID
        zero = pred_positions.sum() * 0.0
        if not torch.isfinite(pred_positions[node_mask]).all():
            raise FloatingPointError("hybrid MMD received non-finite predictions")

        segments: List[_ClassSegment] = []
        whole_inputs: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        slide_keys: List[Optional[bytes]] = [None] * pred_positions.shape[0]

        for slide in range(pred_positions.shape[0]):
            mask = node_mask[slide]
            pred_xy = pred_positions[slide, ..., :2]
            true_xy = true_positions[slide, ..., :2]
            if cell_class is None or cell_class.dim() < 2:
                labels = torch.zeros_like(mask, dtype=torch.long)
            else:
                labels_raw = cell_class[slide]
                labels = (
                    labels_raw.squeeze(-1)
                    if labels_raw.dim() >= 2 and labels_raw.shape[-1] == 1
                    else labels_raw
                ).long()
            pred_xy = self._aligned_pred_xy_vectorized(
                pred_xy, true_xy, labels, mask,
            )
            slide_key = None
            if self.options.cache_gt and cell_id is not None:
                # One host transfer per slide, rather than one per class.
                slide_key = gt_batch_cache_key(cell_id[slide])
                slide_keys[slide] = slide_key
            classes, counts = torch.unique(
                labels[mask], sorted=True, return_counts=True,
            )
            class_ids = classes.detach().cpu().tolist()
            class_counts = counts.detach().cpu().tolist()
            eligible_x: List[torch.Tensor] = []
            for class_id, count in zip(class_ids, class_counts):
                if count < self.min_cells:
                    continue
                class_mask = (labels == class_id) & mask
                X = pred_xy[class_mask]
                G = true_xy[class_mask].detach()
                generator = self._generator(
                    G.device, slide, class_id, 100,
                )
                cache_key = (
                    ("class", slide_key, class_id, count)
                    if slide_key is not None else None
                )
                cache = (
                    self._hybrid_gt_cache.get(cache_key)
                    if cache_key is not None else None
                )
                if cache is None:
                    cache = _build_hybrid_class_cache(
                        G, self.cfg, self.options, generator,
                    )
                    if cache_key is not None and cache is not None:
                        self._hybrid_gt_cache[cache_key] = cache
                if cache is None:
                    continue
                segments.append(
                    _ClassSegment(
                        slide=slide,
                        class_id=class_id,
                        X=X,
                        cache=cache,
                        seed_offset=len(segments),
                    )
                )
                eligible_x.append(X)
            if eligible_x and self.cfg.whole_slice_MMD_weight:
                whole_inputs.append(
                    (slide, torch.cat(eligible_x), true_xy[mask].detach())
                )

        if not segments and not whole_inputs:
            return zero, None

        local_by_index = self._exact_bucket_losses(segments)
        local_by_index.update(self._sampled_large_losses(segments))
        class_losses: List[torch.Tensor] = []
        pd_terms: List[torch.Tensor] = []
        md_terms: List[torch.Tensor] = []
        pdm_terms: List[torch.Tensor] = []
        slides = []
        local_terms = []
        for index, segment in enumerate(segments):
            local = local_by_index[index]
            auxiliary, pd, md, pdm = self._auxiliary_loss(segment)
            class_losses.append(local + auxiliary)
            local_terms.append(local.detach())
            pd_terms.append(pd.detach())
            md_terms.append(md.detach())
            pdm_terms.append(pdm.detach())
            slides.append(segment.slide)

        slide_loss = pred_positions.new_zeros((pred_positions.shape[0],))
        slide_count = pred_positions.new_zeros((pred_positions.shape[0],))
        if class_losses:
            losses = torch.stack(class_losses)
            slide_index = torch.tensor(
                slides, device=losses.device, dtype=torch.long,
            )
            slide_loss = slide_loss.index_add(0, slide_index, losses)
            slide_count = slide_count.index_add(
                0, slide_index, torch.ones_like(losses),
            )
            if self.cfg.mmd_average_over_cell_types:
                slide_loss = slide_loss / slide_count.clamp_min(1.0)

        ws_terms: List[torch.Tensor] = []
        for slide, X_all, G_all in whole_inputs:
            generator = self._generator(X_all.device, slide, -1, 900)
            whole_key = None
            if slide_keys[slide] is not None:
                whole_key = ("whole", slide_keys[slide])
            cache = (
                self._hybrid_gt_cache.get(whole_key)
                if whole_key is not None else None
            )
            if cache is None:
                cache = _build_hybrid_whole_cache(
                    G_all,
                    self.cfg.whole_slice_MMD_band_mults,
                    self.options,
                    generator,
                )
                if whole_key is not None and cache is not None:
                    self._hybrid_gt_cache[whole_key] = cache
            if cache is None:
                continue
            ws = _sampled_isotropic_mmd(
                X_all, cache, self.options.whole_slice_pairs, generator,
            )
            slide_loss[slide] = (
                slide_loss[slide] + self.cfg.whole_slice_MMD_weight * ws
            )
            ws_terms.append(ws.detach())

        valid_slide = slide_count > 0
        loss = slide_loss[valid_slide].mean() if torch.any(valid_slide) else zero
        detached_zero = zero.detach()
        self._last_metric_tensors = {
            "loss": loss.detach(),
            "local": (
                torch.stack(local_terms).mean()
                if local_terms else detached_zero
            ),
            "pair_dist": (
                torch.stack(pd_terms).mean()
                if pd_terms else detached_zero
            ),
            "min_dist": (
                torch.stack(md_terms).mean()
                if md_terms else detached_zero
            ),
            "pair_dist_mmd": (
                torch.stack(pdm_terms).mean()
                if pdm_terms else detached_zero
            ),
            "whole_slice": (
                torch.stack(ws_terms).mean()
                if ws_terms else detached_zero
            ),
        }
        self._forward_index += 1

        to_log = None
        if log:
            self._sync_last_metrics()
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/slide_mmd": self._last_loss,
                f"{prefix}/slide_mmd_local": self._last_local,
                f"{prefix}/slide_mmd_pair_dist": self._last_pair_dist,
                f"{prefix}/slide_mmd_min_dist": self._last_min_dist,
                f"{prefix}/slide_mmd_pair_dist_mmd": self._last_pair_dist_mmd,
                f"{prefix}/slide_mmd_whole_slice": self._last_whole_slice,
            }
        return loss, to_log


__all__ = [
    "HybridMMDOptions",
    "SlideMMDLossHybrid",
    "_class_moments_scatter",
    "_exact_local_bucket_tiled",
    "_hybrid_options_from_cfg",
    "_sampled_local_batch",
    "sample_pair_distances",
]
