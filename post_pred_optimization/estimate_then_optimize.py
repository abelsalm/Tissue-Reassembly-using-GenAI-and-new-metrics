#!/usr/bin/env python3
"""Post-prediction optimization matching per-class triplet-area distributions.

For each slice in ``gt_pred_new`` (paired ``*_metadata_true.csv`` /
``*_metadata_pred.csv``):

1. Per ``cell_class``, subsample up to ``n_subsample`` cells and estimate the
   distribution of triangle areas over all (or capped) triplets.
2. Optimize predicted coordinates so those area distributions match the GT
   reference via multi-bandwidth 1-D IMQ-MMD. Optimizer mode from config:
     - ``"adam"``: gradient descent with Adam
     - ``"metropolis_hastings"`` / ``"mh"``: randomly move 1–16 points with
       Gaussian noise; accept only if overall loss improves
3. Save a per-class area-distribution figure and a 3-panel scatter
   (GT | raw pred | optimized pred) in the style of
   ``metrics/test_evaluation_plot.py``.

Config: ``post_pred_optimization/cfg.json``. Set ``"slice"`` to a stem
(e.g. ``"mouse2_slice1_0"``) to run a single slice, or ``null`` for all.

Usage::

    conda activate LUNA
    python post_pred_optimization/estimate_then_optimize.py
    python post_pred_optimization/estimate_then_optimize.py --cfg post_pred_optimization/cfg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.train_mmds import (  # noqa: E402
    COORD_HI,
    COORD_LO,
    box_penalty,
    subsample_1d,
)

DEFAULT_CFG = Path(__file__).resolve().parent / "cfg.json"


# ---------------------------------------------------------------------------
# Config / IO
# ---------------------------------------------------------------------------


def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def list_slice_pairs(csv_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for true_path in sorted(csv_dir.glob("*_metadata_true.csv")):
        stem = true_path.name[: -len("_metadata_true.csv")]
        pred_path = csv_dir / f"{stem}_metadata_pred.csv"
        if not pred_path.is_file():
            print(f"[skip] missing pred for {stem}")
            continue
        pairs.append((stem, true_path, pred_path))
    return pairs


def resolve_slices(
    csv_dir: Path, slice_cfg: Any
) -> List[Tuple[str, Path, Path]]:
    pairs = list_slice_pairs(csv_dir)
    if slice_cfg is None or slice_cfg == "" or slice_cfg is False:
        return pairs
    wanted = [slice_cfg] if isinstance(slice_cfg, str) else list(slice_cfg)
    wanted_set = set(wanted)
    selected = [p for p in pairs if p[0] in wanted_set]
    missing = wanted_set - {p[0] for p in selected}
    if missing:
        raise FileNotFoundError(f"Requested slice(s) not found in {csv_dir}: {sorted(missing)}")
    return selected


# ---------------------------------------------------------------------------
# Geometry (fully vectorized on GPU)
# ---------------------------------------------------------------------------


_COMBO_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def combinations_3(n: int, device: torch.device) -> torch.Tensor:
    """All index triples ``i < j < k`` for ``n`` points. Shape ``(T, 3)``."""
    key = (n, str(device))
    cached = _COMBO_CACHE.get(key)
    if cached is not None:
        return cached
    if n < 3:
        out = torch.empty((0, 3), dtype=torch.long, device=device)
        _COMBO_CACHE[key] = out
        return out
    idx = torch.arange(n, device=device)
    i = idx.view(-1, 1, 1).expand(n, n, n)
    j = idx.view(1, -1, 1).expand(n, n, n)
    k = idx.view(1, 1, -1).expand(n, n, n)
    mask = (i < j) & (j < k)
    out = torch.stack((i[mask], j[mask], k[mask]), dim=1)
    _COMBO_CACHE[key] = out
    return out


def select_triplet_indices(
    n: int,
    device: torch.device,
    max_triplets: Optional[int],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Return triplet index rows ``(T, 3)`` with ``i < j < k``.

    If ``max_triplets`` is set and ``C(n,3)`` is larger, sample unique triples
    without materializing the full combination table (important for n=256).
    """
    total = n * (n - 1) * (n - 2) // 6 if n >= 3 else 0
    if total == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)
    if max_triplets is None or total <= int(max_triplets):
        return combinations_3(n, device)

    m = int(max_triplets)
    # Rejection sample sorted distinct triples without building C(n,3).
    # Use a CPU generator for portability, then move indices to ``device``.
    cpu_gen = generator
    if cpu_gen is None or (hasattr(cpu_gen, "device") and cpu_gen.device.type != "cpu"):
        cpu_gen = torch.Generator()
        cpu_gen.manual_seed(0 if generator is None else int(generator.initial_seed()))
    chunks: List[torch.Tensor] = []
    have = 0
    while have < m:
        need = max(m - have, 4096)
        raw = torch.randint(0, n, (need * 4, 3), generator=cpu_gen)
        raw, _ = torch.sort(raw, dim=1)
        ok = (raw[:, 0] < raw[:, 1]) & (raw[:, 1] < raw[:, 2])
        raw = raw[ok]
        if raw.numel() == 0:
            continue
        chunks.append(raw)
        have += int(raw.shape[0])
        if have >= m * 2:
            break
    pooled = torch.unique(torch.cat(chunks, dim=0), dim=0)
    if pooled.shape[0] > m:
        perm = torch.randperm(pooled.shape[0], generator=cpu_gen)[:m]
        pooled = pooled[perm]
    return pooled.to(device)


def triplet_areas(xy: torch.Tensor, triplet_idx: torch.Tensor) -> torch.Tensor:
    """Signed-abs triangle areas. ``xy``: ``(N, 2)``, ``triplet_idx``: ``(T, 3)``."""
    if triplet_idx.numel() == 0:
        return xy.new_zeros((0,))
    p0 = xy.index_select(0, triplet_idx[:, 0])
    p1 = xy.index_select(0, triplet_idx[:, 1])
    p2 = xy.index_select(0, triplet_idx[:, 2])
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p2[:, 0] - p0[:, 0]
    ) * (p1[:, 1] - p0[:, 1])
    return 0.5 * cross.abs()


def _off_diag_mean(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[-1]
    if n < 2:
        return K.new_zeros(K.shape[:-2])
    trace = K.diagonal(dim1=-2, dim2=-1).sum(-1)
    total = K.sum(dim=(-2, -1))
    return (total - trace) / (n * (n - 1))


def _imq_from_sqdist(d2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """IMQ kernel ``1 / (1 + d²/σ²)``. ``d2`` is ``(n, m)`` → ``(S, n, m)``."""
    c2 = sigmas.pow(2).clamp_min(1e-12)[:, None, None]
    if d2.ndim == 2:
        d2 = d2.unsqueeze(0)
    return 1.0 / (1.0 + (d2 / c2).clamp_max(1e8))


def imq_selfterm_1d_unscaled(d: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Mean multi-bandwidth IMQ self-term (no σ² reweighting)."""
    if d.numel() < 2:
        return d.new_zeros(())
    d2 = (d[:, None] - d[None, :]).pow(2)
    return _off_diag_mean(_imq_from_sqdist(d2, sigmas)).mean()


def imq_crossterm_1d_unscaled(
    a: torch.Tensor, b: torch.Tensor, sigmas: torch.Tensor
) -> torch.Tensor:
    """Mean multi-bandwidth IMQ cross-term (no σ² reweighting)."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros(())
    d2 = (a[:, None] - b[None, :]).pow(2)
    return _imq_from_sqdist(d2, sigmas).mean(dim=(1, 2)).mean()


def resolve_sigmas(
    band_values: torch.Tensor,
    median_area: torch.Tensor,
    sigma_mode: str,
) -> torch.Tensor:
    """Map config bandwidths to IMQ σ on the area axis.

    - ``absolute``: use ``band_values`` directly as σ (area units, ~0–0.3).
    - ``median_mult``: σ = band_values × median GT area.
    """
    mode = str(sigma_mode).strip().lower()
    if mode in ("absolute", "abs", "fixed"):
        return band_values.clamp_min(1e-8).detach()
    if mode in ("median_mult", "median", "relative", "mult"):
        return (band_values * median_area.clamp_min(1e-8)).detach()
    raise ValueError(
        f"Unknown sigma_mode '{sigma_mode}'. Use 'absolute' or 'median_mult'."
    )


def area_mmd(
    pred_areas: torch.Tensor,
    gt_samples: torch.Tensor,
    gt_self: torch.Tensor,
    sigmas: torch.Tensor,
    max_mmd_samples: int,
    mmd_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mmd_idx is not None and mmd_idx.numel() > 0:
        dp = pred_areas.index_select(0, mmd_idx)
    else:
        dp = subsample_1d(pred_areas, max_mmd_samples)
    if dp.numel() < 2 or gt_samples.numel() < 2:
        return pred_areas.new_zeros(())
    pred_self = imq_selfterm_1d_unscaled(dp, sigmas)
    cross = imq_crossterm_1d_unscaled(dp, gt_samples, sigmas)
    return pred_self + gt_self - 2.0 * cross


# ---------------------------------------------------------------------------
# Per-class GT caches
# ---------------------------------------------------------------------------


@dataclass
class ClassAreaCache:
    name: str
    global_idx: torch.Tensor  # (n,) indices into full slice
    triplet_idx: torch.Tensor  # (T, 3) into the subsampled cloud
    gt_areas_full: torch.Tensor  # (T,) detached, for plotting
    gt_samples: torch.Tensor  # subsampled for MMD
    gt_self: torch.Tensor
    sigmas: torch.Tensor
    mmd_idx: torch.Tensor  # fixed indices into areas for deterministic MMD
    median_area: float


def build_class_caches(
    gt_xy: torch.Tensor,
    labels: np.ndarray,
    *,
    n_subsample: int,
    min_cells: int,
    max_triplets: Optional[int],
    max_mmd_samples: int,
    band_mults: Sequence[float],
    sigma_mode: str,
    device: torch.device,
    generator: torch.Generator,
) -> List[ClassAreaCache]:
    caches: List[ClassAreaCache] = []
    classes = sorted(set(str(c) for c in labels))
    band = torch.as_tensor(list(band_mults), dtype=gt_xy.dtype, device=device)

    for name in classes:
        mask = np.asarray([str(c) == name for c in labels])
        idxs = np.flatnonzero(mask)
        if idxs.size < min_cells:
            continue
        if idxs.size > n_subsample:
            # Deterministic subset from global seed + class name.
            class_seed = (int(torch.initial_seed()) ^ (hash(name) & 0xFFFFFFFF)) % (2**31 - 1)
            rng = np.random.default_rng(class_seed)
            idxs = rng.choice(idxs, size=n_subsample, replace=False)
            idxs.sort()
        global_idx = torch.as_tensor(idxs, dtype=torch.long, device=device)
        n = int(global_idx.numel())
        trip = select_triplet_indices(n, device, max_triplets, generator)
        if trip.shape[0] < 2:
            continue
        gt_cloud = gt_xy.index_select(0, global_idx)
        gt_areas = triplet_areas(gt_cloud, trip).detach()
        med = gt_areas.median() if gt_areas.numel() else gt_xy.new_tensor(1e-4)
        sigmas = resolve_sigmas(band, med, sigma_mode)
        t = int(gt_areas.numel())
        n_mmd = min(t, int(max_mmd_samples))
        if n_mmd < t:
            mmd_idx = torch.randperm(t, generator=generator)[:n_mmd].to(device)
        else:
            mmd_idx = torch.arange(t, device=device)
        gt_samples = gt_areas.index_select(0, mmd_idx).detach()
        gt_self = imq_selfterm_1d_unscaled(gt_samples, sigmas).detach()
        caches.append(
            ClassAreaCache(
                name=name,
                global_idx=global_idx,
                triplet_idx=trip,
                gt_areas_full=gt_areas,
                gt_samples=gt_samples,
                gt_self=gt_self,
                sigmas=sigmas,
                mmd_idx=mmd_idx,
                median_area=float(med.item()),
            )
        )
    return caches


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def compute_total_loss(
    x: torch.Tensor,
    pred_xy0: torch.Tensor,
    caches: List[ClassAreaCache],
    *,
    anchor_weight: float,
    box_penalty_weight: float,
    max_mmd_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total_loss, mmd_loss)`` over all class caches at once."""
    class_losses = []
    for cache in caches:
        cloud = x.index_select(0, cache.global_idx)
        areas = triplet_areas(cloud, cache.triplet_idx)
        class_losses.append(
            area_mmd(
                areas,
                cache.gt_samples,
                cache.gt_self,
                cache.sigmas,
                max_mmd_samples,
                mmd_idx=cache.mmd_idx,
            )
        )
    mmd_loss = torch.stack(class_losses).mean() if class_losses else x.new_zeros(())
    loss = mmd_loss
    if anchor_weight > 0:
        loss = loss + float(anchor_weight) * (x - pred_xy0).pow(2).mean()
    if box_penalty_weight > 0:
        loss = loss + float(box_penalty_weight) * box_penalty(x)
    return loss, mmd_loss


def optimize_slice_adam(
    pred_xy0: torch.Tensor,
    caches: List[ClassAreaCache],
    *,
    steps: int,
    lr: float,
    adam_betas: Sequence[float],
    anchor_weight: float,
    box_penalty_weight: float,
    max_mmd_samples: int,
    log_every: int,
) -> Tuple[torch.Tensor, List[float]]:
    x = pred_xy0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, betas=tuple(adam_betas))
    history: List[float] = []

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, mmd_loss = compute_total_loss(
            x,
            pred_xy0,
            caches,
            anchor_weight=anchor_weight,
            box_penalty_weight=box_penalty_weight,
            max_mmd_samples=max_mmd_samples,
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(COORD_LO, COORD_HI)

        val = float(loss.detach().item())
        history.append(val)
        if step == 1 or step == steps or step % max(1, int(log_every)) == 0:
            print(
                f"  [adam] step {step:4d}/{steps}  loss={val:.6g}  "
                f"mmd={float(mmd_loss.detach().item()):.6g}"
            )

    return x.detach(), history


def optimize_slice_metropolis_hastings(
    pred_xy0: torch.Tensor,
    caches: List[ClassAreaCache],
    *,
    steps: int,
    mh_n_points_min: int,
    mh_n_points_max: int,
    mh_move_sigma: float,
    anchor_weight: float,
    box_penalty_weight: float,
    max_mmd_samples: int,
    log_every: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Greedy MH: move k~U{min..max} points with N(0, σ²); accept iff loss drops."""
    x = pred_xy0.detach().clone()
    # Only move cells that enter the MMD (subsampled per class).
    if caches:
        movable = torch.unique(torch.cat([c.global_idx for c in caches], dim=0))
    else:
        movable = torch.arange(x.shape[0], device=x.device)
    n_movable = int(movable.numel())
    k_min = max(1, int(mh_n_points_min))
    k_max = min(n_movable, max(k_min, int(mh_n_points_max)))
    sigma = float(mh_move_sigma)

    with torch.no_grad():
        cur_loss_t, cur_mmd_t = compute_total_loss(
            x,
            pred_xy0,
            caches,
            anchor_weight=anchor_weight,
            box_penalty_weight=box_penalty_weight,
            max_mmd_samples=max_mmd_samples,
        )
    cur_loss = float(cur_loss_t.item())
    cur_mmd = float(cur_mmd_t.item())
    history: List[float] = [cur_loss]
    n_accept = 0

    print(
        f"  [mh] init loss={cur_loss:.6g}  mmd={cur_mmd:.6g}  "
        f"move k∈[{k_min},{k_max}] of {n_movable} cells  σ={sigma:g}"
    )

    for step in range(1, steps + 1):
        # CPU RNG for proposal indices/noise, then move onto ``x.device``.
        k = int(torch.randint(k_min, k_max + 1, (1,), generator=generator).item())
        pick = torch.randperm(n_movable, generator=generator)[:k]
        move_idx = movable[pick.to(movable.device)]
        noise = (
            torch.randn((k, x.shape[1]), dtype=x.dtype, generator=generator).to(x.device)
            * sigma
        )

        proposal = x.clone()
        proposal[move_idx] = (proposal[move_idx] + noise).clamp(COORD_LO, COORD_HI)

        with torch.no_grad():
            prop_loss_t, prop_mmd_t = compute_total_loss(
                proposal,
                pred_xy0,
                caches,
                anchor_weight=anchor_weight,
                box_penalty_weight=box_penalty_weight,
                max_mmd_samples=max_mmd_samples,
            )
        prop_loss = float(prop_loss_t.item())
        prop_mmd = float(prop_mmd_t.item())

        if prop_loss < cur_loss:
            x = proposal
            cur_loss, cur_mmd = prop_loss, prop_mmd
            n_accept += 1

        history.append(cur_loss)
        if step == 1 or step == steps or step % max(1, int(log_every)) == 0:
            acc = n_accept / step
            print(
                f"  [mh] step {step:4d}/{steps}  loss={cur_loss:.6g}  "
                f"mmd={cur_mmd:.6g}  accept={acc:.2%}  last_k={k}"
            )

    return x, history


def optimize_slice(
    pred_xy0: torch.Tensor,
    caches: List[ClassAreaCache],
    cfg: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[float]]:
    mode = str(cfg.get("optimizer", "adam")).strip().lower()
    common = dict(
        steps=int(cfg["steps"]),
        anchor_weight=float(cfg.get("anchor_weight", 0.0)),
        box_penalty_weight=float(cfg.get("box_penalty_weight", 0.0)),
        max_mmd_samples=int(cfg["max_mmd_samples"]),
        log_every=int(cfg.get("log_every", 50)),
    )
    if mode in ("metropolis_hastings", "mh", "metropolis"):
        return optimize_slice_metropolis_hastings(
            pred_xy0,
            caches,
            mh_n_points_min=int(cfg.get("mh_n_points_min", 1)),
            mh_n_points_max=int(cfg.get("mh_n_points_max", 16)),
            mh_move_sigma=float(cfg.get("mh_move_sigma", 0.01)),
            generator=generator,
            **common,
        )
    if mode != "adam":
        raise ValueError(
            f"Unknown optimizer '{mode}'. Use 'adam' or 'metropolis_hastings'."
        )
    return optimize_slice_adam(
        pred_xy0,
        caches,
        lr=float(cfg["lr"]),
        adam_betas=cfg.get("adam_betas", [0.9, 0.999]),
        **common,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_area_distributions(
    caches: List[ClassAreaCache],
    pred_raw: torch.Tensor,
    pred_opt: torch.Tensor,
    save_path: Path,
    dpi: int = 200,
) -> None:
    n = len(caches)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for i, cache in enumerate(caches):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        with torch.no_grad():
            raw_areas = triplet_areas(
                pred_raw.index_select(0, cache.global_idx), cache.triplet_idx
            ).cpu().numpy()
            opt_areas = triplet_areas(
                pred_opt.index_select(0, cache.global_idx), cache.triplet_idx
            ).cpu().numpy()
            gt_areas = cache.gt_areas_full.cpu().numpy()

        # Subsample for KDE speed if huge.
        def _cap(a: np.ndarray, m: int = 20000) -> np.ndarray:
            if a.size <= m:
                return a
            rng = np.random.default_rng(0)
            return rng.choice(a, size=m, replace=False)

        gt_p, raw_p, opt_p = _cap(gt_areas), _cap(raw_areas), _cap(opt_areas)
        sns.kdeplot(gt_p, ax=ax, label="GT", color="#1b9e77", linewidth=2)
        sns.kdeplot(raw_p, ax=ax, label="Raw pred", color="#d95f02", linewidth=1.5)
        sns.kdeplot(opt_p, ax=ax, label="Optimized", color="#7570b3", linewidth=1.5)
        ax.set_title(cache.name, fontsize=11)
        ax.set_xlabel("Triplet area")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Per-class triplet-area distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_scatter_three_panel(
    metadata_true: pd.DataFrame,
    metadata_pred_raw: pd.DataFrame,
    metadata_pred_opt: pd.DataFrame,
    uniques: Sequence[str],
    save_path: Path,
    dpi: int = 200,
) -> None:
    """GT | raw prediction | optimized prediction (glasbey, like test_evaluation_plot)."""
    fig, axarr = plt.subplots(1, 3, figsize=(22, 6))
    pl_palette = sns.color_palette(cc.glasbey, n_colors=len(uniques))
    palette_dict = dict(zip(uniques, pl_palette))

    panels = [
        (axarr[0], metadata_true, "Groundtruth"),
        (axarr[1], metadata_pred_raw, "Raw prediction"),
        (axarr[2], metadata_pred_opt, "Optimized prediction"),
    ]
    for ax, data, title in panels:
        ax.set_title(title, fontsize=16)
        g = sns.scatterplot(
            data=data,
            x="coord_X",
            y="coord_Y",
            hue="cell_class",
            s=15,
            ax=ax,
            palette=palette_dict,
            hue_order=list(uniques),
            legend=False,
        )
        g.set_xlabel("X", fontsize=14)
        g.set_ylabel("Y", fontsize=14)
        ax.set_aspect("equal", adjustable="datalim")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cat,
            markerfacecolor=palette_dict[cat],
            markersize=10,
        )
        for cat in uniques
    ]
    ncol = max(1, len(uniques) // 4)
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.02),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Per-slice driver
# ---------------------------------------------------------------------------


def run_slice(
    stem: str,
    true_path: Path,
    pred_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    print(f"\n=== {stem} ===")
    true_df = pd.read_csv(true_path, index_col=0)
    pred_df = pd.read_csv(pred_path, index_col=0)
    if not true_df.index.equals(pred_df.index):
        raise ValueError(f"{stem}: GT/pred cell ID index mismatch")
    if not (true_df["cell_class"].astype(str) == pred_df["cell_class"].astype(str)).all():
        raise ValueError(f"{stem}: GT/pred cell_class mismatch")

    labels = true_df["cell_class"].astype(str).to_numpy()
    gt_xy = torch.as_tensor(
        true_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32), device=device
    )
    pred_xy0 = torch.as_tensor(
        pred_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32), device=device
    )

    gen = torch.Generator()  # CPU generator; samples moved to ``device`` as needed
    gen.manual_seed(int(cfg.get("seed", 0)))

    caches = build_class_caches(
        gt_xy,
        labels,
        n_subsample=int(cfg["n_subsample"]),
        min_cells=int(cfg.get("min_cells", 3)),
        max_triplets=cfg.get("max_triplets", None),
        max_mmd_samples=int(cfg["max_mmd_samples"]),
        band_mults=cfg["band_mults"],
        sigma_mode=str(cfg.get("sigma_mode", "absolute")),
        device=device,
        generator=gen,
    )
    print(f"  eligible cell classes: {len(caches)} / {len(set(labels))}")
    print(f"  sigma_mode={cfg.get('sigma_mode', 'absolute')}  band_mults={cfg['band_mults']}")
    with torch.no_grad():
        for c in caches:
            raw_areas = triplet_areas(
                pred_xy0.index_select(0, c.global_idx), c.triplet_idx
            )
            init_mmd = area_mmd(
                raw_areas,
                c.gt_samples,
                c.gt_self,
                c.sigmas,
                int(cfg["max_mmd_samples"]),
                mmd_idx=c.mmd_idx,
            )
            sig_str = ", ".join(f"{s:.4g}" for s in c.sigmas.tolist())
            print(
                f"    {c.name}: n={c.global_idx.numel()}  "
                f"triplets={c.triplet_idx.shape[0]}  "
                f"med_area={c.median_area:.4g}  "
                f"σ=[{sig_str}]  "
                f"init_mmd={float(init_mmd.item()):.6g}  "
                f"area_range=[{float(c.gt_areas_full.min()):.4g}, "
                f"{float(c.gt_areas_full.max()):.4g}]"
            )
    if not caches:
        print("  no eligible classes; skipping")
        return

    pred_opt, history = optimize_slice(pred_xy0, caches, cfg, generator=gen)

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    slice_dir = out_dir / stem
    slice_dir.mkdir(parents=True, exist_ok=True)

    opt_df = pred_df.copy()
    opt_np = pred_opt.cpu().numpy()
    opt_df["coord_X"] = opt_np[:, 0]
    opt_df["coord_Y"] = opt_np[:, 1]
    opt_df.to_csv(slice_dir / f"{stem}_metadata_pred_optimized.csv")
    np.save(slice_dir / "loss_history.npy", np.asarray(history, dtype=np.float64))

    dpi = int(cfg.get("dpi", 200))
    plot_area_distributions(
        caches,
        pred_xy0,
        pred_opt,
        slice_dir / f"{stem}_triplet_area_distributions.png",
        dpi=dpi,
    )
    uniques = sorted(true_df["cell_class"].astype(str).unique().tolist())
    plot_scatter_three_panel(
        true_df,
        pred_df,
        opt_df,
        uniques,
        slice_dir / f"{stem}_class_scatter_gt_raw_opt.png",
        dpi=dpi,
    )


def pick_device(device_str: str) -> torch.device:
    """Resolve ``device``; bare ``cuda`` picks the freest visible GPU."""
    if not str(device_str).startswith("cuda"):
        return torch.device(device_str)
    if not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    if device_str not in ("cuda", "cuda:auto"):
        return torch.device(device_str)
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_i = free, i
    print(
        f"[device] auto-selected cuda:{best_i} "
        f"({best_free / (1024 ** 3):.1f} GiB free)"
    )
    return torch.device(f"cuda:{best_i}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG,
        help="Path to JSON config (default: post_pred_optimization/cfg.json)",
    )
    args = parser.parse_args(argv)
    cfg = load_cfg(args.cfg)

    csv_dir = Path(cfg["csv_dir"])
    if not csv_dir.is_absolute():
        csv_dir = REPO_ROOT / csv_dir

    device = pick_device(str(cfg.get("device", "cuda")))

    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    pairs = resolve_slices(csv_dir, cfg.get("slice", None))
    print(f"Running {len(pairs)} slice(s) on {device}")
    for stem, true_path, pred_path in pairs:
        run_slice(stem, true_path, pred_path, cfg, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
