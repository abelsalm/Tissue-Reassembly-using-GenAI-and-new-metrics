"""Numerical and invariance tests for the hybrid MMD path."""

from types import SimpleNamespace

import torch

from metrics.train_mmds import (
    SlideMMDLoss,
    _mmd_config_from_cfg,
    class_barycenters,
    class_covariances,
    mmd2_local_gt_iso_pred,
)
from metrics.train_mmds_hybrid import (
    HybridMMDOptions,
    SlideMMDLossHybrid,
    _build_hybrid_class_cache,
    _class_moments_scatter,
    _exact_local_bucket_tiled,
    _sampled_local_batch,
    sample_pair_distances,
)
from utils.data.dataholder import DataHolder


def _cfg(**overrides):
    values = dict(
        sigmoid_tau_frac=0.24,
        pca_scale_factor=8.0,
        pair_dist_weight=0.0,
        min_dist_weight=0.0,
        min_dist_thresh=None,
        min_dist_quantile=0.1,
        pair_dist_mmd_weight=0.0,
        pair_dist_mmd_band_mults=(0.08, 0.16, 0.32),
        local_aniso_stretch_power=0.5,
        spatial_band_mults=(0.08, 0.16, 0.64),
        whole_slice_MMD_weight=0.0,
        whole_slice_MMD_band_mults=(0.08, 0.16),
        procrustes_align=False,
        procrustes_with_scale=False,
        procrustes_axis_align=False,
        mmd_anchor=0.0,
        mmd_support=0.0,
        mmd_box=0.0,
        support_margin_mult=3.0,
        mmd_min_cells=2,
        mmd_average_over_cell_types=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _cache(G, exact_cutoff=128):
    options = HybridMMDOptions(
        exact_cutoff=exact_cutoff,
        compile_exact=False,
        statistic_pairs=4096,
        gt_kernel_pairs=16_384,
    )
    generator = torch.Generator(device=G.device).manual_seed(7)
    return _build_hybrid_class_cache(
        G, _mmd_config_from_cfg(_cfg()), options, generator,
    )


def test_exact_bucket_matches_reference_loss_and_gradient():
    torch.manual_seed(1)
    n = 37
    G = torch.rand(n, 2)
    cache = _cache(G)
    X_ref = (G + 0.03 * torch.randn_like(G)).requires_grad_(True)
    X_new = X_ref.detach().clone().requires_grad_(True)

    reference = mmd2_local_gt_iso_pred(X_ref, G, cache.gt_cache)
    mask = torch.ones((1, n), dtype=torch.bool)
    hybrid = _exact_local_bucket_tiled(
        X_new[None],
        G[None],
        mask,
        cache.gt_cache.sigmas[None],
        cache.gt_cache.ky_offdiag[None],
        cache.gt_cache.V[None],
        cache.gt_cache.stretch[None],
        tile_size=11,
    ).sum()
    reference.backward()
    hybrid.backward()

    torch.testing.assert_close(hybrid, reference, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(X_new.grad, X_ref.grad, rtol=2e-4, atol=2e-6)
    cosine = torch.nn.functional.cosine_similarity(
        X_new.grad.flatten(), X_ref.grad.flatten(), dim=0,
    )
    assert cosine > 0.9999


def test_exact_bucket_ignores_padding_and_batches_classes():
    torch.manual_seed(2)
    clouds = [torch.rand(19, 2), torch.rand(31, 2)]
    caches = [_cache(cloud) for cloud in clouds]
    capacity = 40
    X = torch.zeros((2, capacity, 2))
    G = torch.zeros_like(X)
    mask = torch.zeros((2, capacity), dtype=torch.bool)
    for index, cloud in enumerate(clouds):
        n = cloud.shape[0]
        X[index, :n] = cloud + 0.01 * torch.randn_like(cloud)
        G[index, :n] = cloud
        mask[index, :n] = True
    sigmas = torch.stack([cache.gt_cache.sigmas for cache in caches])
    ky = torch.stack([cache.gt_cache.ky_offdiag for cache in caches])
    bands = sigmas.shape[1]
    V = torch.zeros((2, bands, capacity, 2, 2))
    stretch = torch.ones((2, bands, capacity))
    for index, cache in enumerate(caches):
        n = clouds[index].shape[0]
        V[index, :, :n] = cache.gt_cache.V
        stretch[index, :, :n] = cache.gt_cache.stretch
    losses = _exact_local_bucket_tiled(
        X, G, mask, sigmas, ky, V, stretch, tile_size=13,
    )
    references = torch.stack(
        [
            mmd2_local_gt_iso_pred(
                X[i, :cloud.shape[0]], cloud, caches[i].gt_cache,
            )
            for i, cloud in enumerate(clouds)
        ]
    )
    torch.testing.assert_close(losses, references, rtol=2e-5, atol=2e-7)


def test_scatter_class_moments_match_legacy_helpers():
    torch.manual_seed(3)
    positions = torch.randn(71, 2)
    labels = torch.randint(0, 6, (71,))
    mask = torch.rand(71) > 0.1
    classes, counts, barycenters, covariances = _class_moments_scatter(
        positions, labels, mask,
    )
    legacy_barycenters = class_barycenters(
        positions, labels, mask, classes,
    )
    legacy_covariances = class_covariances(
        positions, labels, mask, classes,
    )
    assert torch.all(counts > 0)
    torch.testing.assert_close(barycenters, legacy_barycenters)
    torch.testing.assert_close(covariances, legacy_covariances)


def test_direct_pair_sampling_is_exact_when_triangle_fits():
    points = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    distances = sample_pair_distances(points, max_samples=3)
    torch.testing.assert_close(
        distances.sort().values,
        torch.tensor([3.0, 4.0, 5.0]),
    )


def test_sampled_large_estimator_has_reference_gradient_direction():
    torch.manual_seed(4)
    n = 80
    G = torch.rand(n, 2)
    cache = _cache(G, exact_cutoff=32)
    X_ref = (G + 0.02 * torch.randn_like(G)).requires_grad_(True)
    reference = mmd2_local_gt_iso_pred(X_ref, G, cache.gt_cache)
    reference.backward()

    X = X_ref.detach().clone().requires_grad_(True)
    mask = torch.ones((1, n), dtype=torch.bool)
    torch.manual_seed(5)
    sampled = _sampled_local_batch(
        X[None],
        G[None],
        mask,
        cache.gt_cache.sigmas[None],
        cache.gt_cache.ky_offdiag[None],
        cache.gt_cache.V[None],
        cache.gt_cache.stretch[None],
        pairs_per_source=2048,
    ).sum()
    sampled.backward()
    cosine = torch.nn.functional.cosine_similarity(
        X.grad.flatten(), X_ref.grad.flatten(), dim=0,
    )
    assert cosine > 0.97


def test_full_exact_hybrid_matches_baseline_with_axis_procrustes():
    torch.manual_seed(8)
    n = 60
    true_positions = torch.rand(1, n, 2) - 0.5
    pred_base = (
        true_positions + 0.02 * torch.randn_like(true_positions)
    ).requires_grad_(True)
    pred_hybrid = pred_base.detach().clone().requires_grad_(True)
    labels = (torch.arange(n) % 3).view(1, n, 1)
    node_mask = torch.ones((1, n), dtype=torch.bool)
    cell_ids = torch.arange(n).view(1, n, 1)
    features = torch.zeros((1, n, 1))
    cfg = _cfg(
        mmd_cache_gt=False,
        mmd_min_cells=2,
        procrustes_align=True,
        procrustes_axis_align=True,
        procrustes_axis_min_cells=4,
        procrustes_axis_min_anisotropy=0.0,
        procrustes_axis_weight=1.0,
        mmd_hybrid_compile=False,
        mmd_hybrid_exact_cutoff=128,
        mmd_hybrid_exact_buckets=(32, 64, 128),
        mmd_hybrid_tile_size=16,
        mmd_hybrid_bucket_max_classes=8,
        mmd_hybrid_cache_gt=False,
    )
    truth = DataHolder(
        true_positions, features, None, cell_ID=cell_ids,
        cell_class=labels, node_mask=node_mask,
    )
    base_holder = DataHolder(
        pred_base, features, None, cell_ID=cell_ids,
        cell_class=labels, node_mask=node_mask,
    )
    hybrid_holder = DataHolder(
        pred_hybrid, features, None, cell_ID=cell_ids,
        cell_class=labels, node_mask=node_mask,
    )
    base_loss, _ = SlideMMDLoss(cfg)(base_holder, truth, log=False)
    hybrid_loss, _ = SlideMMDLossHybrid(cfg)(
        hybrid_holder, truth, log=False,
    )
    base_loss.backward()
    hybrid_loss.backward()
    torch.testing.assert_close(
        hybrid_loss, base_loss, rtol=5e-4, atol=2e-7,
    )
    cosine = torch.nn.functional.cosine_similarity(
        pred_hybrid.grad.flatten(), pred_base.grad.flatten(), dim=0,
    )
    assert cosine > 0.9999
