"""CUDA micro-benchmark for baseline versus hybrid training MMD.

Run from the repository root, for example:

    python -m metrics.benchmark_train_mmds_hybrid --sizes 256 512 1000 5000

The default sizes approximate the dominant MERFISH_ABC classes in a 5000-cell
chunk.  Results are emitted as JSON so multiple configurations can be compared
without parsing the progress-bar output of a full training run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Callable, Dict, List

import torch

from metrics.train_mmds import (
    build_class_gt_cache,
    mmd2_imq_iso,
    mmd2_local_gt_iso_pred,
    pair_dist_mmd_loss,
    precompute_whole_slice_mmd_cache,
)
from metrics.train_mmds_hybrid import (
    HybridMMDOptions,
    _build_hybrid_class_cache,
    _build_hybrid_whole_cache,
    _class_moments_scatter,
    _exact_local_bucket_tiled,
    _sampled_local_batch,
    _sampled_isotropic_mmd,
    sampled_pair_distance_mmd,
)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        sigmoid_tau_frac=0.24,
        pca_scale_factor=8.0,
        pair_dist_weight=0.0,
        min_dist_weight=0.0,
        min_dist_thresh=None,
        min_dist_quantile=0.1,
        pair_dist_mmd_weight=16.0,
        pair_dist_mmd_band_mults=(0.08, 0.16, 0.32),
        local_aniso_stretch_power=0.5,
        spatial_band_mults=(0.08, 0.16, 0.64),
        whole_slice_MMD_weight=64.0,
        whole_slice_MMD_band_mults=(0.08, 0.16),
        procrustes_align=True,
        procrustes_with_scale=False,
        procrustes_axis_align=True,
        procrustes_axis_weight=1.0,
        procrustes_axis_min_cells=64,
        procrustes_axis_min_anisotropy=0.4,
        procrustes_axis_length_mult=1.0,
        mmd_anchor=0.0,
        mmd_support=0.5,
        mmd_box=0.1,
        support_margin_mult=3.0,
        mmd_min_cells=10,
        mmd_average_over_cell_types=True,
    )


def _mmd_cfg():
    from metrics.train_mmds import _mmd_config_from_cfg

    return _mmd_config_from_cfg(_cfg())


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _measure(
    name: str,
    closure: Callable[[], torch.Tensor],
    repeats: int,
    warmup: int,
    backward: bool,
) -> Dict[str, float]:
    for _ in range(warmup):
        value = closure()
        if backward:
            value.backward()
    torch.cuda.synchronize()
    samples: List[float] = []
    peaks: List[float] = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        value = closure()
        if backward:
            value.backward()
        stop.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        peaks.append(torch.cuda.max_memory_allocated() / 2**20)
    return {
        "name": name,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "peak_allocated_mib": max(peaks),
    }


def _one_size(
    n: int,
    device: torch.device,
    repeats: int,
    warmup: int,
    include_large_baseline: bool,
    trace: bool,
    exact_cutoff: int,
) -> List[Dict[str, float]]:
    cfg = _mmd_cfg()
    options = HybridMMDOptions(
        exact_cutoff=exact_cutoff,
        compile_exact=False,
        local_pairs=32_768,
        gt_kernel_pairs=131_072,
        whole_slice_pairs=131_072,
        pair_dist_pairs=8_192,
        pair_dist_samples=2_000,
        statistic_pairs=65_536,
    )
    generator = _generator(device, 1000 + n)
    G = (torch.rand((n, 2), device=device, generator=generator) - 0.5).float()
    results: List[Dict[str, float]] = []
    baseline_allowed = n <= 1536 or include_large_baseline

    hybrid_cache = None

    def hybrid_cache_build():
        nonlocal hybrid_cache
        hybrid_cache = _build_hybrid_class_cache(
            G, cfg, options, _generator(device, 2000 + n),
        )
        return hybrid_cache.gt_cache.ky_offdiag.sum()

    results.append(
        _measure(
            "hybrid_gt_cache", hybrid_cache_build, repeats, warmup, False,
        )
    )
    assert hybrid_cache is not None

    baseline_cache = None
    if baseline_allowed:
        def baseline_cache_build():
            nonlocal baseline_cache
            baseline_cache = build_class_gt_cache(G, cfg)
            return baseline_cache.gt_cache.ky_offdiag.sum()

        results.append(
            _measure(
                "baseline_gt_cache", baseline_cache_build,
                repeats, warmup, False,
            )
        )
        assert baseline_cache is not None

    def hybrid_local():
        X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
        cache = hybrid_cache.gt_cache
        mask = torch.ones((1, n), device=device, dtype=torch.bool)
        return _exact_local_bucket_tiled(
            X[None],
            G[None],
            mask,
            cache.sigmas[None],
            cache.ky_offdiag[None],
            cache.V[None],
            cache.stretch[None],
            128,
        ).sum()

    if n <= options.exact_cutoff:
        results.append(
            _measure(
                "hybrid_exact_local_forward_backward",
                hybrid_local,
                repeats,
                warmup,
                True,
            )
        )
    else:
        def hybrid_sampled_local():
            X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
            cache = hybrid_cache.gt_cache
            mask = torch.ones((1, n), device=device, dtype=torch.bool)
            return _sampled_local_batch(
                X[None],
                G[None],
                mask,
                cache.sigmas[None],
                cache.ky_offdiag[None],
                cache.V[None],
                cache.stretch[None],
                min(
                    max(
                        options.local_pairs_per_cell,
                        (options.local_pairs + n - 1) // n,
                    ),
                    max(1, options.local_pairs_max // n),
                ),
            ).sum()

        results.append(
            _measure(
                "hybrid_sampled_local_forward_backward",
                hybrid_sampled_local,
                repeats,
                warmup,
                True,
            )
        )

    if baseline_allowed:
        def baseline_local():
            X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
            return mmd2_local_gt_iso_pred(
                X, baseline_cache.G, baseline_cache.gt_cache,
            )

        results.append(
            _measure(
                "baseline_local_forward_backward",
                baseline_local,
                repeats,
                warmup,
                True,
            )
        )

    def hybrid_pair_dist():
        X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
        return sampled_pair_distance_mmd(
            X,
            hybrid_cache.pdm_gt_samples,
            hybrid_cache.pdm_gt_self,
            hybrid_cache.dist_sigmas,
            options.pair_dist_samples,
            options.pair_dist_pairs,
            _generator(device, 3000 + n),
        )

    results.append(
        _measure(
            "hybrid_pair_distance_forward_backward",
            hybrid_pair_dist,
            repeats,
            warmup,
            True,
        )
    )

    if baseline_allowed:
        def baseline_pair_dist():
            X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
            return pair_dist_mmd_loss(
                X,
                baseline_cache.pdm_gt_samples,
                baseline_cache.pdm_gt_self,
                baseline_cache.dist_sigmas,
                2000,
            )

        results.append(
            _measure(
                "baseline_pair_distance_forward_backward",
                baseline_pair_dist,
                repeats,
                warmup,
                True,
            )
        )

    hybrid_whole = _build_hybrid_whole_cache(
        G,
        cfg.whole_slice_MMD_band_mults,
        options,
        _generator(device, 4000 + n),
    )

    def hybrid_whole_loss():
        X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
        return _sampled_isotropic_mmd(
            X,
            hybrid_whole,
            options.whole_slice_pairs,
            _generator(device, 5000 + n),
        )

    results.append(
        _measure(
            "hybrid_whole_slice_forward_backward",
            hybrid_whole_loss,
            repeats,
            warmup,
            True,
        )
    )

    if baseline_allowed:
        baseline_whole = precompute_whole_slice_mmd_cache(
            G, cfg.whole_slice_MMD_band_mults,
        )

        def baseline_whole_loss():
            X = (G + 0.02 * torch.randn_like(G)).detach().requires_grad_(True)
            return mmd2_imq_iso(
                X,
                baseline_whole.G,
                baseline_whole.sigmas,
                baseline_whole.ky_offdiag,
            )

        results.append(
            _measure(
                "baseline_whole_slice_forward_backward",
                baseline_whole_loss,
                repeats,
                warmup,
                True,
            )
        )

    labels = torch.arange(n, device=device) % min(14, max(2, n // 10))
    mask = torch.ones(n, device=device, dtype=torch.bool)

    def scatter_metadata():
        _, _, barycenters, cov = _class_moments_scatter(G, labels, mask)
        return barycenters.sum() + cov.sum()

    results.append(
        _measure(
            "hybrid_scatter_class_metadata",
            scatter_metadata,
            repeats,
            warmup,
            False,
        )
    )

    if trace:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            context = (
                torch.autograd.profiler.record_function(f"hybrid_n_{n}")
                if trace else nullcontext()
            )
            with context:
                hybrid_pair_dist().backward()
        profiler.export_chrome_trace(f"mmd_hybrid_n{n}.json")

    for row in results:
        row["n"] = n
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 730, 850, 1000, 5000],
        help="Class sizes; defaults reflect common MERFISH_ABC chunks.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--exact-cutoff", type=int, default=1024)
    parser.add_argument("--include-large-baseline", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    rows: List[Dict[str, float]] = []
    for size in args.sizes:
        rows.extend(
            _one_size(
                size,
                device,
                args.repeats,
                args.warmup,
                args.include_large_baseline,
                args.trace,
                args.exact_cutoff,
            )
        )
    payload = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
