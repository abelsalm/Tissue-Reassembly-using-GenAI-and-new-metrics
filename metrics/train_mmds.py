## Training-time whole-slide per-class local-GT kernel MMD loss.
##
## Ports the optimization-based MMD alignment of ``script_mmd.py`` into the
## DDPM training framework. The inner Adam loop on coordinates is replaced by
## the model's own gradient descent: this module just evaluates the same MMD
## objective (per-class local anisotropic GT IMQ-MMD + optional whole-slice
## isotropic MMD + the same auxiliary terms) once per forward pass, so that
## gradients flow back into the predicted positions.
##
## Procrustes pre-alignment and the inner coordinate optimizer from the script
## are intentionally dropped: training learns positions directly, so there is
## no fixed starting point to align to or to anchor against.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb

from utils.data.dataholder import DataHolder
from metrics.gt_cache import cache_key_matches, gt_row_cache_key


COORD_LO, COORD_HI = -0.52, 0.52
PAIR_DIST_MMD_MAX_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Geometry helpers (verbatim ports from script_mmd.py)
# ---------------------------------------------------------------------------

def pairwise_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.cdist(A, B, p=2)


def nn_spacing(G: torch.Tensor) -> torch.Tensor:
    if G.shape[0] < 2:
        return torch.tensor(1e-2, device=G.device)
    D = pairwise_dist(G, G)
    D.fill_diagonal_(float("inf"))
    return D.min(dim=1).values.median()


def median_pairwise(G: torch.Tensor) -> torch.Tensor:
    if G.shape[0] < 2:
        return torch.tensor(1e-2, device=G.device)
    D = pairwise_dist(G, G)
    iu = torch.triu_indices(D.shape[0], D.shape[1], offset=1)
    return D[iu[0], iu[1]].median()


def _off_diag_mean(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[-1]
    if n < 2:
        return K.new_zeros(K.shape[:-2])
    trace = K.diagonal(dim1=-2, dim2=-1).sum(-1)
    total = K.sum(dim=(-2, -1))
    return (total - trace) / (n * (n - 1))


def subsample_1d(vec: torch.Tensor, max_samples: int) -> torch.Tensor:
    n = vec.shape[0]
    if max_samples is None or n <= max_samples:
        return vec
    idx = torch.randperm(n, device=vec.device)[:max_samples]
    return vec[idx]


def _sigmas_tensor(sigmas, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(sigmas):
        return sigmas.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(sigmas, dtype=like.dtype, device=like.device)


def imq_selfterm_1d(d: torch.Tensor, sigmas) -> torch.Tensor:
    if d.numel() < 2:
        return d.new_zeros(())
    sig = _sigmas_tensor(sigmas, d)
    d2 = (d[:, None] - d[None, :]).pow(2)
    per_band = _off_diag_mean(_imq_from_sqdist(d2, sig))
    # σ²-scale each band so grads stay O(1) for small bandwidths.
    return (per_band * sig.pow(2).clamp_min(1e-12)).mean()


def imq_crossterm_1d(a: torch.Tensor, b: torch.Tensor, sigmas) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros(())
    sig = _sigmas_tensor(sigmas, a)
    d2 = (a[:, None] - b[None, :]).pow(2)
    per_band = _imq_from_sqdist(d2, sig).mean(dim=(1, 2))
    return (per_band * sig.pow(2).clamp_min(1e-12)).mean()


def support_penalty(X, G, margin):
    d_nn = pairwise_dist(X, G).min(dim=1).values
    return torch.relu(d_nn - margin).pow(2).mean()


def box_penalty(X):
    below = torch.relu(COORD_LO - X).pow(2)
    above = torch.relu(X - COORD_HI).pow(2)
    return (below + above).mean()


# ---------------------------------------------------------------------------
# Procrustes similarity alignment (ported from script_mmd.py)
# ---------------------------------------------------------------------------

def class_barycenters(positions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                      classes: torch.Tensor) -> torch.Tensor:
    """Stack the per-class barycenters of ``positions[mask]`` in ``classes`` order."""
    pts = positions[mask]
    labs = labels[mask]
    return torch.stack([pts[labs == int(c)].mean(dim=0) for c in classes])


def class_covariances(positions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                       classes: torch.Tensor) -> torch.Tensor:
    """Stack the per-class 2x2 (biased) covariance matrices of ``positions[mask]``.

    Same class order as ``class_barycenters``. Classes with a single cell get
    an all-zero covariance (no orientation information, handled by the
    caller's gating logic rather than here).
    """
    pts = positions[mask]
    labs = labels[mask]
    covs = []
    for c in classes:
        p = pts[labs == int(c)]
        d = p - p.mean(dim=0, keepdim=True)
        cov = (d.unsqueeze(-1) * d.unsqueeze(-2)).mean(dim=0) if d.shape[0] > 0 else torch.zeros(
            positions.shape[-1], positions.shape[-1], device=positions.device, dtype=positions.dtype
        )
        covs.append(cov)
    return torch.stack(covs)


def class_anisotropy(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Closed-form eigenvalue-ratio anisotropy ``(lambda1 - lambda2) / lambda1`` for a
    batch of 2x2 covariance matrices, computed from trace/det (no ``eigh`` call
    needed for this scalar summary). The exact value is in ``[0, 1]``; clamp
    round-off outside that interval before it is used for axis gating."""
    a, d = cov[..., 0, 0], cov[..., 1, 1]
    b = cov[..., 0, 1]
    half = 0.5 * (a + d)
    det = (a * d - b * b).clamp_min(0.0)
    disc = torch.sqrt((half * half - det).clamp_min(0.0))
    lam1 = half + disc
    return ((2.0 * disc) / (lam1 + eps)).clamp(0.0, 1.0)


def class_top_eigvec(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top (largest-eigenvalue) unit eigenvector and eigenvalue of a batch of
    2x2 symmetric PSD matrices. Sign of the eigenvector is arbitrary (as for
    any eigendecomposition) and must be resolved by the caller."""
    evals, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues
    return evecs[..., :, -1], evals[..., -1]


def class_axis_landmarks(
    barycenters: torch.Tensor,
    cov: torch.Tensor,
    ref_dirs: torch.Tensor,
    counts: torch.Tensor,
    length_mult: float = 1.0,
) -> torch.Tensor:
    """Per-class PCA1 landmarks evenly spaced on ``[bc - L v, bc + L v]``.

    For each class, builds ``n ≈ round(sqrt(n_cells))`` (at least 2) points
    along the principal axis through the barycenter, from
    ``barycenter - length * axis`` to ``barycenter + length * axis``.
    ``length`` is ``length_mult * sqrt(top eigenvalue)``.

    The eigenvector sign is resolved against ``ref_dirs`` (unit vectors, one
    per class) by flipping to a non-negative dot product -- this makes the
    landmarks well-defined despite eigenvector sign ambiguity (see
    ``_aligned_pred_xy`` for how ``ref_dirs`` is derived from a reference
    rotation so pred/GT axes are paired consistently).

    Returns a concatenated ``[sum_c n_c, D]`` tensor (classes may contribute
    different ``n``). Empty input returns an empty ``[0, D]`` tensor.
    """
    if barycenters.shape[0] == 0:
        return barycenters.new_zeros((0, barycenters.shape[-1]))

    v, lam = class_top_eigvec(cov)
    sign = torch.sign((v * ref_dirs).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    v = v * sign
    length = length_mult * torch.sqrt(lam.clamp_min(0.0)).unsqueeze(-1)  # [C, 1]

    landmarks = []
    for i in range(barycenters.shape[0]):
        n_cells = int(counts[i].item())
        n = max(2, int(round(n_cells ** 0.5)))
        # Evenly spaced scalars in [-1, +1] (includes both extrema).
        t = torch.linspace(-1.0, 1.0, n, device=barycenters.device, dtype=barycenters.dtype)
        # [n, D] = bc + t[:, None] * (length * v)
        landmarks.append(barycenters[i] + t.unsqueeze(-1) * (length[i] * v[i]))
    return torch.cat(landmarks, dim=0)


def procrustes_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    with_scale: bool = True,
    weights: Optional[torch.Tensor] = None,
):
    """Best-fit weighted similarity (R, t, s) aligning ``source`` to ``target``.

    Returns ``(R, t, s)`` as float tensors on the input device. ``s`` is forced
    to ``1`` when ``with_scale`` is False (rotation + translation only). The
    SVD runs in float64 for numerical stability; the result is cast back to the
    input dtype. Intended to be called on **detached** inputs so the fitted
    transform is a stop-gradient constant (keeps the loss rotation-invariant
    without backprop through the SVD).

    ``weights``, if given, is a non-negative per-row weight (one per landmark
    point) used for the mean, cross-covariance, and scale terms -- standard
    weighted orthogonal Procrustes. Defaults to uniform weights (equivalent to
    the original unweighted fit).
    """
    X = source.detach().to(torch.float64)
    Y = target.detach().to(torch.float64)

    if weights is None:
        w = torch.ones(X.shape[0], dtype=torch.float64, device=X.device)
    else:
        w = weights.detach().to(torch.float64)
    w = w / w.sum().clamp_min(1e-12)

    mu_x = (w[:, None] * X).sum(0)
    mu_y = (w[:, None] * Y).sum(0)
    Xc, Yc = X - mu_x, Y - mu_y

    H = Xc.T @ (w[:, None] * Yc)

    U, _, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        R = Vh.T @ U.T

    if with_scale:
        s = (w * (Yc * (Xc @ R.T)).sum(-1)).sum() / (w * Xc.pow(2).sum(-1)).sum()
    else:
        s = torch.tensor(1.0, dtype=torch.float64, device=source.device)
    t = mu_y - s * R @ mu_x

    out_dtype = source.dtype if source.is_floating_point() else torch.float32
    return R.to(dtype=out_dtype), t.to(dtype=out_dtype), float(s)


def apply_similarity(pts: torch.Tensor, R: torch.Tensor, t: torch.Tensor, s: float) -> torch.Tensor:
    """Apply ``s * (pts @ R.T) + t`` (preserves gradients through ``pts``)."""
    return s * (pts @ R.T) + t


# ---------------------------------------------------------------------------
# Local GT kernel MMD (ported from script_mmd.py)
# ---------------------------------------------------------------------------

@dataclass
class LocalGTCache:
    sigmas: torch.Tensor       # (S,)
    ky_offdiag: torch.Tensor   # (S,)
    V: torch.Tensor            # (S, n, 2, 2)
    stretch: torch.Tensor      # (S, n)


def local_cell_metrics_batch(points, sigma, tau_frac, pca_scale_factor, eps=1e-8, max_ratio=10.0):
    n = points.shape[0]
    sigma_scaled = sigma / pca_scale_factor
    D = pairwise_dist(points, points)
    tau = tau_frac * sigma_scaled
    W = torch.sigmoid((sigma_scaled - D) / (tau + 1e-8))

    wsum = W.sum(dim=1)
    degenerate = wsum < 1e-6
    w = W / wsum.clamp_min(1e-6)[:, None]
    mu = w @ points
    diff = points.unsqueeze(0) - mu.unsqueeze(1)
    cov = torch.einsum("ij,ijc,ijd->icd", w, diff, diff)

    evals, evecs = torch.linalg.eigh(cov)
    lam1 = evals[:, 1].clamp_min(eps)
    lam2 = evals[:, 0].clamp_min(eps)
    ratio = torch.sqrt(lam1 / lam2).clamp(1.0, max_ratio)
    V = evecs[:, :, [1, 0]]

    if degenerate.any():
        eye = torch.eye(2, device=points.device, dtype=points.dtype).expand(n, 2, 2)
        V = torch.where(degenerate[:, None, None], eye, V)
        ratio = torch.where(degenerate, torch.ones_like(ratio), ratio)
    return V, ratio


def symmetrized_local_kernel_matrix(points, sigma, tau_frac, stretch_power, pca_scale_factor):
    V, ratio = local_cell_metrics_batch(points, sigma, tau_frac, pca_scale_factor)
    stretch = ratio.pow(stretch_power)
    # Put bandwidth into the metric once: d2 = ||(V^T diff) / (σ · shape)||²,
    # then IMQ is K = 1/(1+d2). Equivalent to σ²/(σ² + ||diff_aniso||²), but
    # keeps d2 O(1) for typical neighbors and avoids DivBackward overflow when
    # predictions are far from GT (raw c2/(c2+d2) with unscaled d2).
    sigma_t = torch.as_tensor(sigma, dtype=points.dtype, device=points.device).clamp_min(
        1e-6
    )
    sc = sigma_t * torch.stack([stretch, stretch.reciprocal()], dim=-1)

    diff = points.unsqueeze(0) - points.unsqueeze(1)
    proj = torch.einsum("ijc,icd->ijd", diff, V) / sc.unsqueeze(1)
    d2 = proj.pow(2).sum(-1).clamp_max(1e8)
    K = 1.0 / (1.0 + d2)
    K = 0.5 * (K + K.T)
    return K, V, stretch


def precompute_local_gt_cache(G, sigmas, tau_frac, stretch_power, pca_scale_factor) -> LocalGTCache:
    ky_list, V_list, stretch_list = [], [], []
    with torch.no_grad():
        for s in sigmas:
            Ky, V, stretch = symmetrized_local_kernel_matrix(
                G, s, tau_frac, stretch_power, pca_scale_factor
            )
            ky_list.append(_off_diag_mean(Ky))
            V_list.append(V)
            stretch_list.append(stretch)

    return LocalGTCache(
        sigmas=torch.as_tensor(sigmas, dtype=G.dtype, device=G.device),
        ky_offdiag=torch.stack(ky_list),
        V=torch.stack(V_list),
        stretch=torch.stack(stretch_list),
    )


def cross_kernel_gt_local(X, G, cache: LocalGTCache) -> torch.Tensor:
    diff = X.unsqueeze(1) - G.unsqueeze(0)
    proj = torch.einsum("imc,smcd->simd", diff, cache.V)
    # Same metric as the GT self-kernel: σ · [stretch, 1/stretch], K = 1/(1+d2).
    sig = cache.sigmas.clamp_min(1e-6)[:, None]
    sc = sig.unsqueeze(-1) * torch.stack(
        [cache.stretch, cache.stretch.reciprocal()],
        dim=-1,
    )
    proj = proj / sc.unsqueeze(1)
    d2 = proj.pow(2).sum(-1).clamp_max(1e8)
    K = 1.0 / (1.0 + d2)
    return K.mean(dim=(1, 2))


def upper_pairwise_dists(points):
    n = points.shape[0]
    if n < 2:
        return points.new_zeros((0,))
    D = pairwise_dist(points, points)
    iu = torch.triu_indices(n, n, offset=1)
    return D[iu[0], iu[1]]


def mean_pairwise_dist(points):
    d = upper_pairwise_dists(points)
    return d.mean() if d.numel() else points.new_tensor(0.0)


def gt_min_dist_threshold(G, quantile):
    d = upper_pairwise_dists(G)
    if d.numel() == 0:
        return G.new_tensor(1e-2)
    return torch.quantile(d, quantile).detach()


def pair_dist_match_loss(X, gt_mean_pd):
    return (mean_pairwise_dist(X) - gt_mean_pd).pow(2)


def pair_dist_mmd_sigmas(G, band_mults) -> torch.Tensor:
    d = upper_pairwise_dists(G)
    med = d.median() if d.numel() else G.new_tensor(1e-2)
    mults = torch.as_tensor(band_mults, dtype=G.dtype, device=G.device)
    return (mults * med).detach()


def precompute_pair_dist_mmd_gt(G, dist_sigmas, max_samples):
    gt_samples = subsample_1d(upper_pairwise_dists(G).detach(), max_samples)
    gt_self_term = imq_selfterm_1d(gt_samples, dist_sigmas).detach()
    return gt_samples, gt_self_term


def pair_dist_mmd_loss(X, gt_samples, gt_self_term, dist_sigmas, max_samples):
    dp = subsample_1d(upper_pairwise_dists(X), max_samples)
    if dp.numel() < 2:
        return X.new_zeros(())
    pred_self = imq_selfterm_1d(dp, dist_sigmas)
    cross = imq_crossterm_1d(dp, gt_samples, dist_sigmas)
    return pred_self + gt_self_term - 2.0 * cross


def min_dist_repulsion_loss(X, thresh):
    d = upper_pairwise_dists(X)
    if d.numel() == 0:
        return X.new_zeros(())
    return torch.relu(thresh - d).pow(2).mean()


def _imq_from_sqdist(d2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Stable IMQ ``σ²/(σ²+d²)`` as ``1/(1 + d²/σ²)``.

    ``d2`` is ``(n, m)`` pairwise squared distances; result is ``(S, n, m)``.
    """
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, None, None]
    if d2.ndim == 2:
        d2 = d2.unsqueeze(0)
    # Cap the scaled distance so far-away pairs stay in the flat IMQ regime
    # without float overflow in the backward of the division.
    return 1.0 / (1.0 + (d2 / c2).clamp_max(1e8))


def mmd2_imq_iso(X, G, sigmas: torch.Tensor, ky_offdiag: torch.Tensor) -> torch.Tensor:
    dxx = pairwise_dist(X, X).pow(2)
    dxy = pairwise_dist(X, G).pow(2)
    kxx_offdiag = _off_diag_mean(_imq_from_sqdist(dxx, sigmas))
    kxy_mean = _imq_from_sqdist(dxy, sigmas).mean(dim=(1, 2))
    per_band = kxx_offdiag + ky_offdiag - 2.0 * kxy_mean
    # Per-band σ² scaling: IMQ grads are O(1/σ²); this keeps them O(1).
    return (per_band * sigmas.pow(2).clamp_min(1e-12)).mean()


@dataclass
class WholeSliceMMDCache:
    G: torch.Tensor
    sigmas: torch.Tensor
    ky_offdiag: torch.Tensor


def precompute_whole_slice_mmd_cache(G, band_mults) -> Optional[WholeSliceMMDCache]:
    if G.shape[0] < 2:
        return None
    med = median_pairwise(G)
    mults = torch.as_tensor(band_mults, dtype=G.dtype, device=G.device)
    sigmas = mults * med
    dyy = pairwise_dist(G, G).pow(2)
    ky_offdiag = _off_diag_mean(_imq_from_sqdist(dyy, sigmas)).detach()
    return WholeSliceMMDCache(G=G, sigmas=sigmas.detach(), ky_offdiag=ky_offdiag)


def mmd2_local_gt_iso_pred(X, G, gt_cache: LocalGTCache) -> torch.Tensor:
    dxx = pairwise_dist(X, X).pow(2)
    kxx_offdiag = _off_diag_mean(_imq_from_sqdist(dxx, gt_cache.sigmas))
    kxy_mean = cross_kernel_gt_local(X, G, gt_cache)
    per_band = kxx_offdiag + gt_cache.ky_offdiag - 2.0 * kxy_mean
    # Per-band σ² scaling: IMQ grads are O(1/σ²); this keeps them O(1).
    return (per_band * gt_cache.sigmas.pow(2).clamp_min(1e-12)).mean()


# ---------------------------------------------------------------------------
# Per-class GT cache (fixed during training; keyed by slide row + class)
# ---------------------------------------------------------------------------

@dataclass
class ClassMMGTCache:
    """Everything about one (slide, class) that is fixed while pred moves."""
    G: torch.Tensor
    gt_cache: LocalGTCache
    gt_mean_pd: torch.Tensor
    md_thresh: torch.Tensor
    support_margin: torch.Tensor
    dist_sigmas: Optional[torch.Tensor]
    pdm_gt_samples: Optional[torch.Tensor]
    pdm_gt_self: Optional[torch.Tensor]


def build_class_gt_cache(
    G: torch.Tensor,
    cfg: "MMDConfig",
) -> Optional[ClassMMGTCache]:
    """Precompute every fixed GT quantity for one class (no gradients)."""
    if G.shape[0] < 2:
        return None

    med = median_pairwise(G).item()
    sigmas = [m * med for m in cfg.spatial_band_mults]
    gt_cache = precompute_local_gt_cache(
        G, sigmas, cfg.sigmoid_tau_frac, cfg.local_aniso_stretch_power, cfg.pca_scale_factor
    )
    md_thresh = (
        gt_min_dist_threshold(G, cfg.min_dist_quantile)
        if cfg.min_dist_thresh is None
        else G.new_tensor(float(cfg.min_dist_thresh)).detach()
    )
    support_margin = (cfg.support_margin_mult * nn_spacing(G)).detach()
    if cfg.pair_dist_mmd_weight:
        dist_sigmas = pair_dist_mmd_sigmas(G, cfg.pair_dist_mmd_band_mults)
        pdm_gt_samples, pdm_gt_self = precompute_pair_dist_mmd_gt(
            G, dist_sigmas, PAIR_DIST_MMD_MAX_SAMPLES
        )
    else:
        dist_sigmas = None
        pdm_gt_samples = pdm_gt_self = None

    return ClassMMGTCache(
        G=G,
        gt_cache=gt_cache,
        gt_mean_pd=mean_pairwise_dist(G).detach(),
        md_thresh=md_thresh,
        support_margin=support_margin,
        dist_sigmas=dist_sigmas,
        pdm_gt_samples=pdm_gt_samples,
        pdm_gt_self=pdm_gt_self,
    )


def class_mmd_loss(
    X: torch.Tensor,
    cc: ClassMMGTCache,
    cfg: "MMDConfig",
    X0: Optional[torch.Tensor] = None,
):
    """Scalar loss for one (slide, class); mirrors ``class_mmd_loss`` in the script.

    ``X0`` is the optional anchor reference. In the optimization script it is
    the Procrustes-aligned starting point; in training there is no separate
    initialization, so the anchor is a no-op unless a reference is supplied.
    """
    loss = mmd2_local_gt_iso_pred(X, cc.G, cc.gt_cache)
    pd_loss = pair_dist_match_loss(X, cc.gt_mean_pd)
    loss = loss + cfg.pair_dist_weight * pd_loss

    if cfg.min_dist_weight:
        md_loss = min_dist_repulsion_loss(X, cc.md_thresh)
        loss = loss + cfg.min_dist_weight * md_loss
    else:
        md_loss = X.new_zeros(())

    if cfg.pair_dist_mmd_weight:
        pdm_loss = pair_dist_mmd_loss(
            X, cc.pdm_gt_samples, cc.pdm_gt_self, cc.dist_sigmas,
            PAIR_DIST_MMD_MAX_SAMPLES,
        )
        loss = loss + cfg.pair_dist_mmd_weight * pdm_loss
    else:
        pdm_loss = X.new_zeros(())

    if cfg.mmd_anchor and X0 is not None:
        loss = loss + cfg.mmd_anchor * (X - X0).pow(2).mean()
    loss = loss + cfg.mmd_support * support_penalty(X, cc.G, cc.support_margin)
    loss = loss + cfg.mmd_box * box_penalty(X)
    return loss, pd_loss, md_loss, pdm_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MMDConfig:
    mmd_average_over_cell_types: bool = True
    sigmoid_tau_frac: float = 10e-3
    pca_scale_factor: float = 1.0
    pair_dist_weight: float = 0.0
    min_dist_weight: float = 0.1
    min_dist_thresh: Optional[float] = None
    min_dist_quantile: float = 0.05
    pair_dist_mmd_weight: float = 1.0
    pair_dist_mmd_band_mults: tuple = (0.1, 0.25, 0.5, 1.0)
    local_aniso_stretch_power: float = 0.2
    spatial_band_mults: tuple = (1.0, 2.0, 4.0, 8.0, 16.0)
    whole_slice_MMD_weight: float = 0.0
    whole_slice_MMD_band_mults: tuple = (1.0, 2.0, 4.0, 8.0)
    # Procrustes pre-alignment (rotation / translation / scale invariance):
    procrustes_align: bool = True
    procrustes_with_scale: bool = True
    # Optional per-class shape/orientation term added to the Procrustes fit
    # (matches per-class covariance "ellipses" via their matrix square roots,
    # which is sign-ambiguity-free unlike raw eigenvector matching). Helps
    # stabilize the fitted rotation when barycenter geometry alone is nearly
    # degenerate (e.g. few classes, collinear or symmetric centroids).
    procrustes_axis_align: bool = False
    procrustes_axis_weight: float = 1.0
    procrustes_axis_min_cells: int = 5
    procrustes_axis_min_anisotropy: float = 0.2
    procrustes_axis_length_mult: float = 1.0
    # The following mirror alignment_config.json but are unused in training
    # (no inner coordinate optimizer / Procrustes starting point here):
    mmd_steps: int = 200
    mmd_lr: float = 1e-2
    mmd_cosine_lr: bool = False
    mmd_lr_min: float = 1e-4
    mmd_lr_max: Optional[float] = None
    mmd_anchor: float = 1e-2     # no-op in training (no separate init point)
    mmd_support: float = 0.5
    mmd_box: float = 1e-1
    mmd_max_grad: float = 1.0    # unused in training (grad clipping handled by the trainer)
    support_margin_mult: float = 3.0
    min_cells: int = 2          # skip classes with fewer GT cells


def _mmd_config_from_cfg(cfg) -> MMDConfig:
    def g(key, default):
        v = getattr(cfg, key, None)
        return v if v is not None else default

    def tup(key, default):
        v = getattr(cfg, key, None)
        if v is None:
            return tuple(default)
        return tuple(float(x) for x in v)

    return MMDConfig(
        mmd_average_over_cell_types=bool(
            g("mmd_average_over_cell_types", True)
        ),
        sigmoid_tau_frac=float(g("sigmoid_tau_frac", 10e-3)),
        pca_scale_factor=float(g("pca_scale_factor", 1.0)),
        pair_dist_weight=float(g("pair_dist_weight", 0.0)),
        min_dist_weight=float(g("min_dist_weight", 0.1)),
        min_dist_thresh=g("min_dist_thresh", None),
        min_dist_quantile=float(g("min_dist_quantile", 0.05)),
        pair_dist_mmd_weight=float(g("pair_dist_mmd_weight", 1.0)),
        pair_dist_mmd_band_mults=tup("pair_dist_mmd_band_mults", (0.1, 0.25, 0.5, 1.0)),
        local_aniso_stretch_power=float(g("local_aniso_stretch_power", 0.2)),
        spatial_band_mults=tup("spatial_band_mults", (1.0, 2.0, 4.0, 8.0, 16.0)),
        whole_slice_MMD_weight=float(g("whole_slice_MMD_weight", 0.0)),
        whole_slice_MMD_band_mults=tup("whole_slice_MMD_band_mults", (1.0, 2.0, 4.0, 8.0)),
        procrustes_align=bool(g("procrustes_align", True)),
        procrustes_with_scale=bool(g("procrustes_with_scale", True)),
        procrustes_axis_align=bool(g("procrustes_axis_align", False)),
        procrustes_axis_weight=float(g("procrustes_axis_weight", 1.0)),
        procrustes_axis_min_cells=int(g("procrustes_axis_min_cells", 5)),
        procrustes_axis_min_anisotropy=float(g("procrustes_axis_min_anisotropy", 0.2)),
        procrustes_axis_length_mult=float(g("procrustes_axis_length_mult", 1.0)),
        mmd_steps=int(g("mmd_steps", 200)),
        mmd_lr=float(g("mmd_lr", 1e-2)),
        mmd_cosine_lr=bool(g("mmd_cosine_lr", False)),
        mmd_lr_min=float(g("mmd_lr_min", 1e-4)),
        mmd_lr_max=g("mmd_lr_max", None),
        mmd_anchor=float(g("mmd_anchor", 1e-2)),
        mmd_support=float(g("mmd_support", 0.5)),
        mmd_box=float(g("mmd_box", 1e-1)),
        mmd_max_grad=float(g("mmd_max_grad", 1.0)),
        support_margin_mult=float(g("support_margin_mult", 3.0)),
        min_cells=int(g("mmd_min_cells", 2)),
    )


# ---------------------------------------------------------------------------
# Slide-level loss module (training framework interface)
# ---------------------------------------------------------------------------

class SlideMMDLoss(nn.Module):
    """Whole-slide per-class local-GT kernel MMD loss for DDPM training.

    For every slide in the batch and every cell class present, computes the
    same per-class objective as ``script_mmd.py`` (local anisotropic GT IMQ-MMD
    + pair-distance match + min-distance repulsion + pair-distance MMD +
    support / box penalties), reduces the per-class losses by configurable
    mean or sum, and optionally adds a whole-slice isotropic MMD coherence term
    coupling all predicted cells against the full GT point cloud.

    All GT-side quantities are cached per (slide, class) and reused across
    steps; the cache is keyed by the slide's ``cell_ID`` row so it survives
    gradient accumulation within an epoch and is cleared on
    ``reset`` / ``clear_gt_cache`` (e.g. after a position warp or rechunk).

    The inner Adam coordinate optimizer of the original script is replaced by
    the model's own training: this module only evaluates the objective once
    per forward pass so gradients flow into ``masked_pred.positions``.

    Rotation invariance: a single Procrustes similarity transform is fit per
    slide on the per-class barycenters (pred vs GT) and applied to all
    predicted cells before the MMD is evaluated, exactly like
    ``run_procrustes`` in the script. The transform is a stop-gradient
    constant (SVD on detached inputs, recomputed every forward), so the loss
    value is rotation / translation / scale invariant while gradients remain
    stable (no backprop through the SVD). Disable with ``procrustes_align``
    or drop the scale term with ``procrustes_with_scale``.

    Optionally (``procrustes_axis_align``), the rotation fit is additionally
    informed by per-class shape/orientation (covariance) matching on top of
    the barycenters, which stabilizes the estimated rotation when the
    barycenter geometry alone is weak (few classes, near-collinear or
    near-symmetric centroids). See ``class_shape_alignment_term``.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg: MMDConfig = _mmd_config_from_cfg(cfg)
        self.min_cells = int(getattr(cfg, "mmd_min_cells", 2))
        self.cache_gt = bool(getattr(cfg, "mmd_cache_gt", True))
        self._gt_cache: Dict = {}
        self._last_loss: float = -1.0
        self._last_local: float = -1.0
        self._last_pair_dist: float = -1.0
        self._last_min_dist: float = -1.0
        self._last_pair_dist_mmd: float = -1.0
        self._last_whole_slice: float = -1.0

    # ------------------------------------------------------------------ #
    # GT cache
    # ------------------------------------------------------------------ #

    def clear_gt_cache(self) -> None:
        self._gt_cache.clear()

    def reset(self) -> None:
        self.clear_gt_cache()

    def _class_cache_key(self, cell_id_row, class_int, mask_count):
        if not self.cache_gt or cell_id_row is None:
            return None
        row_key = gt_row_cache_key(cell_id_row, 0)
        if row_key is None:
            return None
        return (row_key, int(class_int), int(mask_count))

    def _get_class_cache(
        self,
        G: torch.Tensor,
        class_int: int,
        mask_count: int,
        cell_id_row,
        device,
        dtype,
    ) -> Optional[ClassMMGTCache]:
        if not self.cache_gt:
            return build_class_gt_cache(G, self.cfg)

        key = self._class_cache_key(cell_id_row, class_int, mask_count)
        if key is None:
            return build_class_gt_cache(G, self.cfg)
        cached = self._gt_cache.get(key)
        if cached is not None:
            if cell_id_row is not None:
                cached_cid = cached.get("cell_id")
                if cached_cid is None or not cache_key_matches(cell_id_row, cached_cid):
                    cached = None
        if cached is None:
            cc = build_class_gt_cache(G, self.cfg)
            if cc is not None:
                self._gt_cache[key] = {
                    "cell_id": cell_id_row.detach() if cell_id_row is not None else None,
                    "cc": cc,
                }
            return cc
        return cached["cc"]

    def _get_whole_slice_cache(
        self,
        G_all: torch.Tensor,
        cell_id_row,
        device,
        dtype,
    ) -> Optional[WholeSliceMMDCache]:
        if not self.cfg.whole_slice_MMD_weight:
            return None
        if not self.cache_gt or cell_id_row is None:
            return precompute_whole_slice_mmd_cache(G_all, self.cfg.whole_slice_MMD_band_mults)

        row_key = gt_row_cache_key(cell_id_row, 0)
        if row_key is None:
            return precompute_whole_slice_mmd_cache(G_all, self.cfg.whole_slice_MMD_band_mults)
        key = ("ws", row_key)
        cached = self._gt_cache.get(key)
        if cached is not None:
            if not cache_key_matches(cell_id_row, cached.get("cell_id")):
                cached = None
        if cached is None:
            ws = precompute_whole_slice_mmd_cache(G_all, self.cfg.whole_slice_MMD_band_mults)
            if ws is not None:
                self._gt_cache[key] = {
                    "cell_id": cell_id_row.detach(),
                    "ws": ws,
                }
            return ws
        return cached["ws"]

    # ------------------------------------------------------------------ #
    # Procrustes alignment (detached transform, recomputed each forward)
    # ------------------------------------------------------------------ #

    def _aligned_pred_xy(
        self,
        pred_xy: torch.Tensor,
        true_xy: torch.Tensor,
        labels: Optional[torch.Tensor],
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """Procrustes-align ``pred_xy`` to ``true_xy`` via per-class barycenters.

        Mirrors ``run_procrustes`` in ``script_mmd.py``: one global similarity
        transform is fit on the per-class barycenters (pred vs GT) and applied
        to every predicted cell of the slide. The fitted ``(R, t, s)`` is a
        stop-gradient constant (SVD in float64 on detached inputs), so the
        loss value is rotation / translation / scale invariant while gradients
        flow through ``pred_xy`` only via the linear application
        ``s * (X @ R.T) + t``. Returns ``pred_xy`` unchanged when alignment is
        disabled or there are too few usable barycenters (< 2 classes).

        When ``procrustes_axis_align`` is set, classes with enough GT cells
        (``procrustes_axis_min_cells``) and a sufficiently anisotropic GT
        shape (``procrustes_axis_min_anisotropy``) also contribute
        ``n ≈ sqrt(n_cells)`` principal-axis pseudo-landmarks per class
        (evenly spaced on ``[bc - L·PCA1, bc + L·PCA1]``), weighted by
        ``procrustes_axis_weight`` relative to the (unit-weight) barycenter
        landmarks. Gating on the *GT*-side anisotropy (fixed target, not the
        still-training prediction) keeps the set of classes used for
        orientation matching stable across training steps.

        Axis eigenvectors are only defined up to sign, so a cheap
        barycenter-only rotation fit is computed first and used purely as a
        reference to pick each class's pred-axis sign consistently with its
        GT-axis pairing (see ``class_axis_landmarks``); the final fit then
        re-solves the rotation using both barycenters and axis landmarks
        together.
        """
        if not self.cfg.procrustes_align:
            return pred_xy
        if labels is None:
            return pred_xy

        usable = []
        for cval in torch.unique(labels[mask_b]):
            cint = int(cval.item())
            cmask = (labels == cint) & mask_b
            if int(cmask.sum().item()) < 1:
                continue
            usable.append(cint)
        if len(usable) < 2:
            return pred_xy  # need >= 2 barycenters for a non-degenerate rotation

        classes_t = torch.tensor(usable, device=pred_xy.device, dtype=labels.dtype)
        P_bc = class_barycenters(pred_xy, labels, mask_b, classes_t)
        G_bc = class_barycenters(true_xy, labels, mask_b, classes_t)

        P_all, G_all, weights = P_bc, G_bc, None
        if self.cfg.procrustes_axis_align:
            counts = torch.tensor(
                [int(((labels == c) & mask_b).sum().item()) for c in usable],
                device=pred_xy.device,
            )
            axis_gate = counts >= self.cfg.procrustes_axis_min_cells
            if bool(axis_gate.any()):
                G_cov = class_covariances(true_xy, labels, mask_b, classes_t)
                gt_aniso = class_anisotropy(G_cov)
                axis_gate = axis_gate & (gt_aniso >= self.cfg.procrustes_axis_min_anisotropy)
                if bool(axis_gate.any()):
                    P_cov = class_covariances(pred_xy, labels, mask_b, classes_t)
                    R0, _, _ = procrustes_similarity(P_bc, G_bc, with_scale=False)
                    G_axis_dir, _ = class_top_eigvec(G_cov)
                    # Reference direction for the pred-side axis, expressed in
                    # the pred frame (inverse of R0 applied to the GT axis),
                    # so the sign flip in ``class_axis_landmarks`` pairs each
                    # pred axis with the GT axis it is closest to under R0.
                    ref_dirs = G_axis_dir @ R0
                    P_axis = class_axis_landmarks(
                        P_bc[axis_gate], P_cov[axis_gate], ref_dirs[axis_gate],
                        counts[axis_gate], self.cfg.procrustes_axis_length_mult,
                    )
                    G_axis = class_axis_landmarks(
                        G_bc[axis_gate], G_cov[axis_gate], G_axis_dir[axis_gate],
                        counts[axis_gate], self.cfg.procrustes_axis_length_mult,
                    )
                    P_all = torch.cat([P_bc, P_axis], dim=0)
                    G_all = torch.cat([G_bc, G_axis], dim=0)
                    weights = torch.cat([
                        torch.ones(P_bc.shape[0], device=pred_xy.device),
                        torch.full(
                            (P_axis.shape[0],), self.cfg.procrustes_axis_weight,
                            device=pred_xy.device,
                        ),
                    ])

        R, t, s = procrustes_similarity(
            P_all, G_all, with_scale=self.cfg.procrustes_with_scale, weights=weights,
        )
        return apply_similarity(pred_xy, R, t, s)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

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
        cell_class = masked_true.cell_class
        cell_id = masked_true.cell_ID

        active_pred = pred_positions[node_mask.bool()]
        if not torch.isfinite(active_pred).all():
            nonfinite = int((~torch.isfinite(active_pred)).sum().item())
            raise FloatingPointError(
                f"MMD received {nonfinite} non-finite active predicted coordinates"
            )

        device = pred_positions.device
        dtype = pred_positions.dtype
        zero = pred_positions.sum() * 0.0

        loss_terms = []
        local_terms = []
        pd_terms = []
        md_terms = []
        pdm_terms = []
        ws_terms = []

        for b in range(pred_positions.shape[0]):
            mask_b = node_mask[b]
            mask_b = mask_b.bool() if mask_b.dtype != torch.bool else mask_b
            pred_xy = pred_positions[b, ..., :2]
            true_xy = true_positions[b, ..., :2]
            cid_b = cell_id[b] if cell_id is not None else None

            if cell_class is None or cell_class.dim() < 2:
                labels = None
                classes_present = [(0, mask_b)]
            else:
                cc_b = cell_class[b]
                if cc_b.dim() >= 2 and cc_b.shape[-1] == 1:
                    labels = cc_b.squeeze(-1)
                else:
                    labels = cc_b if cc_b.dim() == 1 else cc_b.argmax(-1)
                classes_present = []
                for cval in torch.unique(labels[mask_b]):
                    cint = int(cval.item())
                    cmask = (labels == cint) & mask_b
                    classes_present.append((cint, cmask))

            pred_xy = self._aligned_pred_xy(pred_xy, true_xy, labels, mask_b)
            if not torch.isfinite(pred_xy[mask_b]).all():
                nonfinite = int((~torch.isfinite(pred_xy[mask_b])).sum().item())
                raise FloatingPointError(
                    f"MMD Procrustes alignment produced {nonfinite} non-finite "
                    f"predicted coordinates"
                )

            per_class_losses = []
            per_class_xy = []
            for cint, cmask in classes_present:
                if int(cmask.sum().item()) < self.min_cells:
                    continue
                G = true_xy[cmask]
                cc = self._get_class_cache(
                    G.detach(), cint, int(cmask.sum().item()), cid_b, device, dtype,
                )
                if cc is None:
                    continue
                X = pred_xy[cmask]
                c_loss, pd_loss, md_loss, pdm_loss = class_mmd_loss(
                    X, cc, self.cfg, X0=None,
                )
                per_class_losses.append(c_loss)
                per_class_xy.append(X)
                local_terms.append(c_loss.detach())
                pd_terms.append(pd_loss.detach())
                md_terms.append(md_loss.detach())
                pdm_terms.append(pdm_loss.detach())

            ws_loss = zero
            if self.cfg.whole_slice_MMD_weight and mask_b.sum() >= 2:
                G_all = true_xy[mask_b]
                ws_cache = self._get_whole_slice_cache(G_all.detach(), cid_b, device, dtype)
                if ws_cache is not None and per_class_xy:
                    X_all = torch.cat(per_class_xy, dim=0)
                    if X_all.shape[0] >= 2:
                        ws_loss = mmd2_imq_iso(
                            X_all, ws_cache.G, ws_cache.sigmas, ws_cache.ky_offdiag,
                        )
                        ws_terms.append(ws_loss.detach())

            if per_class_losses:
                per_class_stack = torch.stack(per_class_losses)
                if self.cfg.mmd_average_over_cell_types:
                    # Keep the slide scale independent of how many cell types
                    # pass ``mmd_min_cells``.
                    per_class_loss = per_class_stack.mean()
                else:
                    per_class_loss = per_class_stack.sum()
                slide_loss = per_class_loss + self.cfg.whole_slice_MMD_weight * ws_loss
            else:
                slide_loss = zero + self.cfg.whole_slice_MMD_weight * ws_loss
            loss_terms.append(slide_loss)

        if not loss_terms:
            return zero, None

        loss = torch.stack(loss_terms).mean()
        self._last_loss = float(loss.detach().item())
        self._last_local = float(torch.stack(local_terms).mean().item()) if local_terms else 0.0
        self._last_pair_dist = float(torch.stack(pd_terms).mean().item()) if pd_terms else 0.0
        self._last_min_dist = float(torch.stack(md_terms).mean().item()) if md_terms else 0.0
        self._last_pair_dist_mmd = float(torch.stack(pdm_terms).mean().item()) if pdm_terms else 0.0
        self._last_whole_slice = float(torch.stack(ws_terms).mean().item()) if ws_terms else 0.0

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {
                f"{prefix}/slide_mmd": self._last_loss,
                f"{prefix}/slide_mmd_local": self._last_local,
                f"{prefix}/slide_mmd_pair_dist": self._last_pair_dist,
                f"{prefix}/slide_mmd_min_dist": self._last_min_dist,
                f"{prefix}/slide_mmd_pair_dist_mmd": self._last_pair_dist_mmd,
                f"{prefix}/slide_mmd_whole_slice": self._last_whole_slice,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss, to_log

    # ------------------------------------------------------------------ #
    # Epoch logging
    # ------------------------------------------------------------------ #

    def log_epoch_metrics(self, train_stage: bool = True) -> Dict[str, float]:
        epoch_prefix = "train_epoch" if train_stage else "val_epoch"
        return {
            f"{epoch_prefix}/slide_mmd": float(self._last_loss),
            f"{epoch_prefix}/slide_mmd_local": float(self._last_local),
            f"{epoch_prefix}/slide_mmd_pair_dist": float(self._last_pair_dist),
            f"{epoch_prefix}/slide_mmd_min_dist": float(self._last_min_dist),
            f"{epoch_prefix}/slide_mmd_pair_dist_mmd": float(self._last_pair_dist_mmd),
            f"{epoch_prefix}/slide_mmd_whole_slice": float(self._last_whole_slice),
        }
