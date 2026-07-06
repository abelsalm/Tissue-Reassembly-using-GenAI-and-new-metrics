"""Statistical testing pipeline for LUNA checkpoints.

For each checkpoint listed in ``configs/test/default.yaml`` the pipeline:

1. Runs ``n`` denoising samples (different seeds) on every ``cell_section`` of
   the test split. Sampling is organised **seed-first**: for each seed, all
   sections are inferred in batches of ``test.batch_size`` graphs; analysis
   runs only after every seed/section pair has been collected.
2. Computes **precise per-cell-type distributional distances** between the
   predicted spatial distributions and the ground-truth slice.

   For every cell type we end up with **two Wasserstein distances**:

   * ``wasserstein_whole_slice_alignment`` — every denoising sample's *full*
     point cloud is rotation-aligned to the GT slice via Procrustes, the
     cells of the given type are turned into a 2-D continuous distribution
     (KDE), discretized on a shared grid, averaged over samples, and compared
     to the GT type sample (also KDE'd on the same grid). The distance is the
     Sinkhorn approximation of the Wasserstein-1 distance between the two
     discretized distributions.
   * ``wasserstein_per_class_alignment`` — instead of aligning the whole slice,
     we align (Procrustes, rotation only) each sample's *cell-type-only* point
     cloud to the GT cell-type sample, then build the per-sample KDE on the
     same shared grid, average over samples, and again compute the Sinkhorn
     distance to the GT type KDE.

   The continuous→discrete step is shared by both alignments: a 2-D Gaussian
   KDE is built from the point set, evaluated on a regular grid, and mass-
   normalised so each distribution sums to 1 on the grid. The Sinkhorn
   iteration runs on these two histograms with the ground-cost matrix given
   by the Euclidean distance between grid cell centres.

3. Computes **Cahn-Hilliard energy comparison metrics** (see
   ``metrics/test_ch_and_voronoi.py``) per denoising sample, then averages
   over samples. Two losses are reported per section and aggregated across
   sections:

   * ``ch_energy_curve_loss`` — for each cell type, a continuous landscape
     ``phi(x; r)`` is built from the cell positions at each radius of a
     sweep, the Cahn-Hilliard energy ``E(r)`` is integrated on a shared
     grid for both GT and prediction, and the two energy curves are
     compared per-radius with the bounded relative form
     ``1 - exp(-|E_pred - E_gt| / (|E_gt| + eps))`` (sample value = mean
     over cell types of the mean over radii).
   * ``voronoi_phase_pair_ch_energy_loss`` — for each unordered pair of cell
     types ``(A, B)`` the Voronoi phase-separation landscape
     ``phi_AB = tanh((dist_A - dist_B) / w)`` is built, its CH energy is
     evaluated for both GT and prediction, and the same bounded relative
     form is applied per pair (sample value = mean over pairs).

   Both are averaged over the denoising samples of a section, then averaged
   across sections in the final summary.

4. Computes a **cross-sample positional spread** (no GT comparison): for
   each cell, the average pairwise Euclidean distance between that cell's
   positions across the denoising samples (after rotation-only Procrustes
   alignment of every sample onto a reference sample), averaged over cells
   and then over sections. The alignment and pairwise-distance computation
   are vectorised on GPU (batched SVD + a broadcast ``(S, S, N)`` distance
   tensor).

Usage
-----
    python metrics/test_testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        general.mode=test_only

    python metrics/test_testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        test.checkpoint_path=/path/to/epoch=19999.ckpt \\
        test.pipeline.num_samples=64
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yaml import safe_load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.test_evaluation_statistics import align_point_clouds  # noqa: E402
from utils.testing.diffusion2spatial_probs import (  # noqa: E402
    REPO_ROOT as _DIFFUSION_REPO_ROOT,
    SliceBatch,
    _resolve_device,
    _set_seed,
    build_section_batch,
    load_model,
    save_predictions_csv,
)
from utils.diffusion_model.sample.sample import sample_from_single_graph  # noqa: E402
from metrics.test_ch_and_voronoi import (  # noqa: E402
    CahnHilliardEnergyCurveLoss,
    VoronoiPhasePairEnergyLoss,
    ch_energy_curve_loss_per_sample,
    voronoi_pair_energy_loss_per_sample,
)
from metrics.test_vanilla_loss import LossFunction as PositionMSELoss  # noqa: E402
from metrics.train_directional_metric import DirectionalMetricLoss  # noqa: E402
from utils.data.dataholder import DataHolder  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Config / checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_section_dirname(cell_section: str, max_length: int = 120) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cell_section).strip()).strip("._")
    return (name or "cell_section")[:max_length]


def _checkpoint_stem(checkpoint_path: Path) -> str:
    name = checkpoint_path.stem
    if name.startswith("epoch="):
        return name
    return checkpoint_path.name.replace(".ckpt", "")


def resolve_checkpoints(cfg: DictConfig) -> list[Path]:
    """Resolve one or many checkpoint paths from ``cfg.test``."""
    if getattr(cfg.test, "checkpoint_path", None):
        ckpt = Path(str(cfg.test.checkpoint_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [ckpt]

    parent = Path(str(cfg.test.checkpoints_parent_dir))
    if not parent.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {parent}")

    name_list = cfg.test.checkpoints_name_list
    if name_list == "all":
        names = sorted(n for n in os.listdir(parent) if n.endswith(".ckpt"))
    else:
        names = list(name_list)

    paths = [parent / name for name in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(str(p) for p in missing)
        )
    return paths


def load_model_config_from_checkpoint(cfg: DictConfig, checkpoint_path: Path) -> None:
    """Restore ``cfg.model`` from the training run that produced the checkpoint."""
    config_file = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    if not config_file.exists():
        print(
            f"[testing_pipeline] Warning: no saved config at {config_file}; "
            "using current model config."
        )
        return
    loading_model_cfg = safe_load(config_file.open())
    cfg.model = loading_model_cfg["model"]


def list_cell_sections(test_dataset) -> list[str]:
    """Return sorted unique ``cell_section`` labels in the test split."""
    sections = np.asarray(test_dataset._cell_sections_clean)
    return sorted({str(s) for s in np.unique(sections)})


def pipeline_output_dir(cfg: DictConfig, checkpoint_path: Path) -> Path:
    """Directory where all artefacts for one checkpoint are written."""
    base = cfg.test.save_dir
    if base is None:
        base = str(checkpoint_path.parent)
    return Path(base) / cfg.general.name / "testing_pipeline" / _checkpoint_stem(checkpoint_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling (seed-outer, section-batch-inner)
# ─────────────────────────────────────────────────────────────────────────────


def prepare_section_batches(
    datamodule: Any,
    dataset_infos: Any,
    cell_sections: list[str],
) -> dict[str, SliceBatch]:
    """Build and cache one ``SliceBatch`` per ``cell_section``."""
    section_batches: dict[str, SliceBatch] = {}
    for cell_section in tqdm(
        cell_sections, desc="Building section batches", unit="section",
        disable=False, mininterval=0.5,
    ):
        section_batches[cell_section] = build_section_batch(
            datamodule, dataset_infos, cell_section
        )
    return section_batches


def build_batched_holder(slice_batches: list[SliceBatch], device: Any):
    """Pad variable-size section graphs into one dense ``DataHolder`` batch."""
    from utils.data.dataholder import DataHolder

    holders = [section_batch.holder for section_batch in slice_batches]
    batch_size = len(holders)
    max_nodes = max(int(holder.positions.shape[1]) for holder in holders)
    feat_dim = int(holders[0].node_features.shape[-1])

    positions = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float32)
    node_features = torch.zeros(batch_size, max_nodes, feat_dim, dtype=torch.float32)
    cell_class = torch.zeros(batch_size, max_nodes, 1, dtype=torch.long)
    cell_ID = torch.zeros(batch_size, max_nodes, 1, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)

    for graph_idx, holder in enumerate(holders):
        num_nodes = holder.positions.shape[1]
        positions[graph_idx, :num_nodes] = holder.positions[0]
        node_features[graph_idx, :num_nodes] = holder.node_features[0]
        cell_class[graph_idx, :num_nodes] = holder.cell_class[0]
        cell_ID[graph_idx, :num_nodes] = holder.cell_ID[0]
        node_mask[graph_idx, :num_nodes] = True

    return DataHolder(
        positions=positions.to(device),
        node_features=node_features.to(device),
        cell_class=cell_class.to(device),
        cell_ID=cell_ID.to(device),
        node_mask=node_mask.to(device),
        diffusion_time=None,
    ).mask()


def split_batched_positions(
    positions: torch.Tensor,
    node_mask: torch.Tensor,
    slice_batches: list[SliceBatch],
) -> dict[str, np.ndarray]:
    """Extract per-section predicted coordinates from a padded batch."""
    section_preds: dict[str, np.ndarray] = {}
    for graph_idx, section_batch in enumerate(slice_batches):
        num_nodes = int(node_mask[graph_idx].sum().item())
        section_preds[section_batch.cell_section] = (
            positions[graph_idx, :num_nodes].detach().cpu().numpy().astype(np.float32)
        )
    return section_preds


def run_all_sampling(
    model: Any,
    section_batches: dict[str, SliceBatch],
    cell_sections: list[str],
    num_samples: int,
    seed_start: int,
    batch_size: int,
    device: Any,
) -> dict[str, np.ndarray]:
    """Sample every section for each seed, batching sections within a seed.

    Returns
    -------
    dict mapping ``cell_section`` -> ``(num_samples, N, 2)`` predicted positions.
    """
    section_pred_lists: dict[str, list[np.ndarray]] = {
        cell_section: [] for cell_section in cell_sections
    }
    section_chunks = [
        cell_sections[chunk_start : chunk_start + batch_size]
        for chunk_start in range(0, len(cell_sections), batch_size)
    ]

    seeds = range(seed_start, seed_start + num_samples)
    for seed in tqdm(
        seeds, desc="Seeds", unit="seed",
        disable=False, mininterval=0.5,
    ):
        for chunk in section_chunks:
            slice_batch_list = [section_batches[cell_section] for cell_section in chunk]
            holder = build_batched_holder(slice_batch_list, device)
            _set_seed(seed)
            positions = sample_from_single_graph(model, test=True, batch=holder)
            split_preds = split_batched_positions(
                positions, holder.node_mask, slice_batch_list
            )
            for cell_section, pred in split_preds.items():
                section_pred_lists[cell_section].append(pred)

    stacked_preds: dict[str, np.ndarray] = {}
    for cell_section, pred_list in section_pred_lists.items():
        stacked_preds[cell_section] = np.stack(pred_list, axis=0)
    return stacked_preds


def predictions_to_gt_dataframe(batch: SliceBatch) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_section": batch.cell_section,
            "cell_ID": batch.cell_ids,
            "cell_class": batch.cell_class_labels,
            "cell_class_int": batch.cell_class_int,
            "coord_X": batch.gt_positions[:, 0],
            "coord_Y": batch.gt_positions[:, 1],
        }
    )


def cell_classes_with_min_cells(
    gt_df: pd.DataFrame,
    min_cells: int,
) -> list[str]:
    counts = gt_df["cell_class"].astype(str).value_counts()
    return sorted(counts[counts >= min_cells].index.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Continuous distributions (KDE) + grid discretization
# ─────────────────────────────────────────────────────────────────────────────


def build_common_grid(
    points_list: Iterable[np.ndarray],
    grid_resolution: int,
    grid_margin: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Build a shared regular grid covering the bounding box of every point set.

    ``points_list`` should include both the GT and (a representative sample of)
    the predicted point clouds so the grid contains every distribution we will
    discretize. Returns ``(grid_x, grid_y, extent)`` where ``extent`` follows
    matplotlib's ``(xmin, xmax, ymin, ymax)`` convention.
    """
    stacked = np.concatenate(
        [np.asarray(pts, dtype=np.float64) for pts in points_list if pts.size],
        axis=0,
    )
    if stacked.size == 0:
        raise ValueError("Cannot build a grid from an empty point set.")
    xmin, ymin = stacked.min(axis=0) - grid_margin
    xmax, ymax = stacked.max(axis=0) + grid_margin
    grid_x = np.linspace(xmin, xmax, grid_resolution, dtype=np.float64)
    grid_y = np.linspace(ymin, ymax, grid_resolution, dtype=np.float64)
    extent = (float(xmin), float(xmax), float(ymin), float(ymax))
    return grid_x, grid_y, extent


def _kde_eval_on_grid(
    points: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Evaluate an isotropic 2-D Gaussian KDE at ``points`` over the grid.

    Returns a ``(len(grid_y), len(grid_x))`` density array. Uses
    ``scipy.stats.gaussian_kde`` with a fixed bandwidth (Scott's factor is
    overridden via ``bw_method = bandwidth / std`` per axis-covariance so the
    resulting kernel std is exactly ``bandwidth`` in both x and y). When fewer
    than two points are provided, falls back to a single Gaussian placed at the
    (one) point so the function stays total.
    """
    from scipy.stats import gaussian_kde

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((grid_y.size, grid_x.size), dtype=np.float64)
    if n < 2:
        # Place a single isotropic Gaussian at the lone point.
        dx = grid_x[None, :] - pts[0, 0]
        dy = grid_y[None, :] - pts[0, 1]
        g = np.exp(-(dx**2 + dy**2) / (2.0 * bandwidth**2))
        g /= 2.0 * np.pi * bandwidth**2
        return g.astype(np.float64)

    # gaussian_kde scales the kernel by bw_method * data_covariance. To get a
    # kernel whose std is exactly ``bandwidth`` regardless of the spread of the
    # data, we first standardise the points and set bw_method so the resulting
    # std on the standardised scale equals bandwidth / per_axis_std. Working on
    # the standardised data keeps the covariance isotropic.
    pts_T = pts.T  # (2, n)
    kde = gaussian_kde(pts_T, bw_method=bandwidth)
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")  # both (H, W)
    coords = np.vstack([X.ravel(), Y.ravel()])
    density = kde(coords).reshape(X.shape).astype(np.float64)
    return density


def discretize_to_grid(
    points: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build a normalized probability matrix from ``points`` on the grid.

    A 2-D Gaussian KDE is built from ``points`` and evaluated at every grid
    cell centre; the resulting density is then mass-normalised to sum to 1
    (i.e. a discrete probability distribution over grid cells). Returns an
    array of shape ``(len(grid_y), len(grid_x))``.
    """
    density = _kde_eval_on_grid(points, grid_x, grid_y, bandwidth)
    total = density.sum()
    if total <= 0.0:
        out = np.full_like(density, eps)
        return out / out.sum()
    density = density / total
    # Avoid exact zeros so the Sinkhorn iterations stay numerically stable.
    density = np.where(density > eps, density, eps)
    density /= density.sum()
    return density


# ─────────────────────────────────────────────────────────────────────────────
# Sinkhorn approximation of the Wasserstein-1 distance on a grid
# ─────────────────────────────────────────────────────────────────────────────


def _grid_cost_matrix(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Euclidean ground cost between every pair of grid cell centres.

    Returns a matrix of shape ``(nx*ny, nx*ny)`` where cell ``(ix, iy)`` is
    flattened in row-major order over y then x (i.e. index = iy * nx + ix).
    """
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")  # both (ny, nx)
    centres = np.stack([X.ravel(), Y.ravel()], axis=1)  # (nx*ny, 2)
    diff = centres[:, None, :] - centres[None, :, :]
    cost = np.sqrt((diff**2).sum(axis=-1))
    return cost.astype(np.float64)


def sinkhorn_wasserstein_grid(
    P: np.ndarray,
    Q: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    reg: float,
    n_iters: int = 500,
    tol: float = 1e-9,
    cost: np.ndarray | None = None,
) -> float:
    """Approximate the Wasserstein-1 distance between two grid distributions.

    Both ``P`` and ``Q`` must be probability matrices of shape
    ``(len(grid_y), len(grid_x))`` summing to 1. Sinkhorn's algorithm is run
    with entropic regularization ``reg`` (the larger ``reg``, the smoother /
    more biased the estimate). The returned scalar is the regularised Sinkhorn
    transport cost — an upper-bound proxy for the true Wasserstein-1 distance.
    """
    P = np.asarray(P, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64).ravel()
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must share shape; got {P.shape} and {Q.shape}."
        )
    if P.sum() <= 0 or Q.sum() <= 0:
        raise ValueError("P and Q must be non-empty probability distributions.")

    P = P / P.sum()
    Q = Q / Q.sum()
    if cost is None:
        cost = _grid_cost_matrix(grid_x, grid_y)

    K = np.exp(-cost / reg)
    u = np.ones_like(P)
    v = np.ones_like(Q)
    for _ in range(n_iters):
        u_prev = u
        # K v
        kv = K @ v
        u = P / np.where(kv > 0, kv, 1e-300)
        ku = K.T @ u
        v = Q / np.where(ku > 0, ku, 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    # Regularised transport cost = sum_ij T_ij * cost_ij with T = diag(u) K diag(v).
    transport = u[:, None] * K * v[None, :]
    wdist = float((transport * cost).sum())
    return wdist


# ─────────────────────────────────────────────────────────────────────────────
# Procrustes alignment helpers
# ─────────────────────────────────────────────────────────────────────────────


def align_samples_to_reference(
    samples: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Rotation-only Procrustes align every sample onto ``reference``.

    ``samples`` has shape ``(S, N, 2)`` and ``reference`` shape ``(N, 2)``;
    each sample is aligned independently using ``align_point_clouds``.
    """
    samples = np.asarray(samples, dtype=np.float32)
    aligned = np.empty_like(samples)
    for i in range(samples.shape[0]):
        aligned[i] = align_point_clouds(reference, samples[i])
    return aligned


def align_samples_to_reference_torch(
    samples: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Batched rotation-only Procrustes alignment of every sample onto ``reference``.

    Vectorised GPU implementation of :func:`align_samples_to_reference` using
    batched SVD (matches ``align_point_clouds`` exactly: rotation only, no
    scaling, re-centered at the reference mean).

    Args:
        samples: ``(S, N, 2)`` point clouds (one per denoising sample).
        reference: ``(N, 2)`` reference point cloud (typically ``samples[0]``
            or the GT slice).

    Returns:
        ``(S, N, 2)`` tensor with each sample rotation-aligned to
        ``reference``.
    """
    samples = samples.to(dtype=torch.float32)
    reference = reference.to(dtype=torch.float32)
    s, n, _ = samples.shape

    ref_mean = reference.mean(dim=0, keepdim=True)               # (1, 2)
    ref_centered = reference - ref_mean                          # (N, 2)

    samp_mean = samples.mean(dim=1, keepdim=True)                # (S, 1, 2)
    samp_centered = samples - samp_mean                          # (S, N, 2)

    # M = samp_centered.T @ ref_centered  -> (S, 2, 2)
    M = torch.bmm(
        samp_centered.transpose(1, 2),                # (S, 2, N)
        ref_centered.unsqueeze(0).expand(s, -1, -1),  # (S, N, 2)
    )
    U, _, Vt = torch.linalg.svd(M)                               # each (S, 2, 2)
    R = torch.bmm(U, Vt)                                         # (S, 2, 2)

    aligned = torch.bmm(samp_centered, R) + ref_mean.unsqueeze(0)  # (S, N, 2)
    # If one of the samples *is* the reference cloud, force its aligned row to
    # be exactly the reference (avoids tiny numerical drift on the identity
    # alignment). Only do this when the reference actually equals a sample row,
    # never when the reference is an external cloud such as the GT slice.
    ref_broadcast = reference.unsqueeze(0).expand_as(samples)
    is_ref = ((samples - ref_broadcast).abs().max(dim=-1).values.max(dim=-1).values < 1e-6)
    if is_ref.any():
        aligned[is_ref] = reference
    return aligned


def compute_cross_sample_position_spread(
    preds: np.ndarray,
    gt_positions: np.ndarray,
    device: Any | None = None,
) -> tuple[float, np.ndarray]:
    """Mean pairwise position difference across samples, per cell.

    For each cell, computes the average Euclidean distance between that
    cell's positions across the different denoising samples (after rotation-
    only Procrustes alignment of **every sample directly onto the GT
    slice**), then averages over cells. No comparison to ground truth is
    involved in the distance itself — the GT is only used as the alignment
    reference frame, which is more robust than aligning onto one of the
    (noisy) denoising samples.

    Fully broadcast on GPU: the per-cell pairwise distance matrix is built
    as ``||(aligned[i] - aligned[j])||`` over the ``S`` samples for every
    cell at once (shape ``(S, S, N)``); the SVD-based alignment is batched.

    Args:
        preds: ``(S, N, 2)`` predicted positions for one section.
        gt_positions: ``(N, 2)`` ground-truth positions used as the common
            alignment reference frame.
        device: torch device; defaults to CUDA if available else CPU.

    Returns:
        ``(overall_mean, per_cell_spreads)`` where ``per_cell_spreads`` is a
        ``(N,)`` numpy array and ``overall_mean`` is the mean over cells.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.as_tensor(np.asarray(preds), dtype=torch.float32, device=device)
    reference = torch.as_tensor(
        np.asarray(gt_positions), dtype=torch.float32, device=device
    )
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"preds must have shape (S, N, 2); got {samples.shape}")
    if reference.ndim != 2 or reference.shape[-1] != 2:
        raise ValueError(
            f"gt_positions must have shape (N, 2); got {reference.shape}"
        )
    if reference.shape[0] != samples.shape[1]:
        raise ValueError(
            f"gt_positions has {reference.shape[0]} cells but preds has "
            f"{samples.shape[1]}; they must match."
        )
    s, n, _ = samples.shape
    if s < 2:
        zeros = np.zeros(n, dtype=np.float32)
        return 0.0, zeros

    aligned = align_samples_to_reference_torch(samples, reference)   # (S, N, 2)

    # Pairwise distances over the sample axis for every cell at once.
    # diff[i, j, c] = aligned[i, c] - aligned[j, c]  -> (S, S, N, 2)
    diff = aligned.unsqueeze(0) - aligned.unsqueeze(1)
    pair_dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-20)           # (S, S, N)
    # Exclude the diagonal (self-pairs, distance 0) and average.
    iu = torch.triu_indices(s, s, offset=1, device=device)
    pair_vals = pair_dist[iu[0], iu[1]]                              # (n_pairs, N)
    per_cell = pair_vals.mean(dim=0)                                 # (N,)
    overall = float(per_cell.mean().item())
    return overall, per_cell.detach().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell-type distributional distances
# ─────────────────────────────────────────────────────────────────────────────


def _class_mask_for_sample(batch: SliceBatch, cell_class: str) -> np.ndarray:
    return np.asarray([str(c) == str(cell_class) for c in batch.cell_class_labels])


def compute_wasserstein_whole_slice_alignment(
    preds: np.ndarray,
    batch: SliceBatch,
    cell_classes: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float,
    sinkhorn_reg: float,
    sinkhorn_iters: int,
) -> pd.DataFrame:
    """Wasserstein distances after whole-slice Procrustes alignment.

    Every sample's full point cloud is rotation-aligned to the GT slice; for
    each cell type the aligned class points are turned into a KDE on the shared
    grid, averaged over samples, and compared to the GT class KDE via Sinkhorn.
    """
    gt_positions = batch.gt_positions.astype(np.float32)
    aligned_preds = align_samples_to_reference(preds, gt_positions)

    cost = _grid_cost_matrix(grid_x, grid_y)
    rows = []
    for cell_class in cell_classes:
        mask = _class_mask_for_sample(batch, cell_class)
        gt_class_points = gt_positions[mask]
        if gt_class_points.shape[0] < 2:
            continue

        gt_dist = discretize_to_grid(gt_class_points, grid_x, grid_y, bandwidth)

        sample_dists = []
        for s in range(aligned_preds.shape[0]):
            class_pts = aligned_preds[s][mask]
            if class_pts.shape[0] == 0:
                continue
            sample_dists.append(
                discretize_to_grid(class_pts, grid_x, grid_y, bandwidth)
            )
        if not sample_dists:
            continue
        pred_dist = np.mean(np.stack(sample_dists, axis=0), axis=0)
        pred_dist = pred_dist / pred_dist.sum()

        wdist = sinkhorn_wasserstein_grid(
            pred_dist, gt_dist, grid_x, grid_y,
            reg=sinkhorn_reg, n_iters=sinkhorn_iters, cost=cost,
        )
        rows.append({
            "cell_class": str(cell_class),
            "wasserstein_whole_slice_alignment": wdist,
            "n_class_cells": int(mask.sum()),
            "n_samples_used": len(sample_dists),
        })
    return pd.DataFrame(rows)


def compute_wasserstein_per_class_alignment(
    preds: np.ndarray,
    batch: SliceBatch,
    cell_classes: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float,
    sinkhorn_reg: float,
    sinkhorn_iters: int,
) -> pd.DataFrame:
    """Wasserstein distances after per-cell-type Procrustes alignment.

    For each cell type, every sample's class-only point cloud is rotation-
    aligned to the GT class sample, KDE'd on the shared grid, averaged over
    samples, and compared to the GT class KDE via Sinkhorn.
    """
    gt_positions = batch.gt_positions.astype(np.float32)
    cost = _grid_cost_matrix(grid_x, grid_y)
    rows = []
    for cell_class in cell_classes:
        mask = _class_mask_for_sample(batch, cell_class)
        gt_class_points = gt_positions[mask]
        if gt_class_points.shape[0] < 2:
            continue

        gt_dist = discretize_to_grid(gt_class_points, grid_x, grid_y, bandwidth)

        sample_dists = []
        for s in range(preds.shape[0]):
            class_pts = preds[s][mask]
            if class_pts.shape[0] < 2:
                continue
            aligned_pts = align_point_clouds(gt_class_points, class_pts)
            sample_dists.append(
                discretize_to_grid(aligned_pts, grid_x, grid_y, bandwidth)
            )
        if not sample_dists:
            continue
        pred_dist = np.mean(np.stack(sample_dists, axis=0), axis=0)
        pred_dist = pred_dist / pred_dist.sum()

        wdist = sinkhorn_wasserstein_grid(
            pred_dist, gt_dist, grid_x, grid_y,
            reg=sinkhorn_reg, n_iters=sinkhorn_iters, cost=cost,
        )
        rows.append({
            "cell_class": str(cell_class),
            "wasserstein_per_class_alignment": wdist,
            "n_class_cells": int(mask.sum()),
            "n_samples_used": len(sample_dists),
        })
    return pd.DataFrame(rows)


def compute_section_wasserstein_metrics(
    preds: np.ndarray,
    batch: SliceBatch,
    cell_classes: list[str],
    wasserstein_cfg: Any,
) -> pd.DataFrame:
    """Compute both Wasserstein distances for every cell type of one section."""
    grid_resolution = int(wasserstein_cfg.grid_resolution)
    grid_margin = float(wasserstein_cfg.grid_margin)
    bandwidth = float(wasserstein_cfg.kde_bandwidth)
    sinkhorn_reg = float(wasserstein_cfg.sinkhorn_reg)
    sinkhorn_iters = int(wasserstein_cfg.sinkhorn_iters)

    gt_positions = batch.gt_positions.astype(np.float32)
    grid_x, grid_y, _ = build_common_grid(
        [gt_positions] + [preds[s] for s in range(preds.shape[0])],
        grid_resolution=grid_resolution,
        grid_margin=grid_margin,
    )

    whole_df = compute_wasserstein_whole_slice_alignment(
        preds, batch, cell_classes, grid_x, grid_y,
        bandwidth, sinkhorn_reg, sinkhorn_iters,
    )
    per_class_df = compute_wasserstein_per_class_alignment(
        preds, batch, cell_classes, grid_x, grid_y,
        bandwidth, sinkhorn_reg, sinkhorn_iters,
    )

    if whole_df.empty and per_class_df.empty:
        return pd.DataFrame()
    merged = pd.merge(
        whole_df, per_class_df,
        on=["cell_class", "n_class_cells", "n_samples_used"],
        how="outer",
    )
    return merged


def aggregate_wasserstein_across_sections(
    section_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average the two Wasserstein distances per cell type across sections."""
    if not section_tables:
        return pd.DataFrame()
    combined = pd.concat(section_tables, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    grouped = (
        combined.groupby("cell_class", as_index=False)
        .agg(
            mean_wasserstein_whole_slice_alignment=(
                "wasserstein_whole_slice_alignment", "mean"
            ),
            std_wasserstein_whole_slice_alignment=(
                "wasserstein_whole_slice_alignment", "std"
            ),
            mean_wasserstein_per_class_alignment=(
                "wasserstein_per_class_alignment", "mean"
            ),
            std_wasserstein_per_class_alignment=(
                "wasserstein_per_class_alignment", "std"
            ),
            n_sections=("cell_class", "size"),
            total_cells=("n_class_cells", "sum"),
        )
        .sort_values("cell_class")
    )
    for col in (
        "std_wasserstein_whole_slice_alignment",
        "std_wasserstein_per_class_alignment",
    ):
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def plot_wasserstein_summary(
    summary_df: pd.DataFrame,
    save_path: Path,
    n_sections_total: int | None = None,
) -> None:
    """Grouped bar chart of the two per-class Wasserstein distances."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No Wasserstein data to plot.")
        return

    save_path = Path(save_path)
    plot_df = summary_df.sort_values("mean_wasserstein_whole_slice_alignment")
    classes = plot_df["cell_class"].to_numpy()
    y = np.arange(len(classes))
    bar_h = 0.4

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * len(classes))))
    ax.barh(
        y - bar_h / 2,
        plot_df["mean_wasserstein_whole_slice_alignment"],
        xerr=plot_df["std_wasserstein_whole_slice_alignment"],
        height=bar_h,
        color="tab:blue",
        alpha=0.85,
        capsize=3,
        label="Whole-slice alignment",
    )
    ax.barh(
        y + bar_h / 2,
        plot_df["mean_wasserstein_per_class_alignment"],
        xerr=plot_df["std_wasserstein_per_class_alignment"],
        height=bar_h,
        color="tab:orange",
        alpha=0.85,
        capsize=3,
        label="Per-class alignment",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("Sinkhorn-approximated Wasserstein distance to GT")
    ax.set_ylabel("Cell class")
    sections_note = (
        f", {n_sections_total} sections" if n_sections_total is not None else ""
    )
    ax.set_title(
        "Per-cell-type Wasserstein distance to GT spatial distribution\n"
        f"(KDE on grid + Sinkhorn{sections_note}; "
        f"{int(plot_df['total_cells'].sum())} cells total)"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved Wasserstein summary plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cahn-Hilliard & Voronoi phase-pair energy losses (per sample, averaged)
# ─────────────────────────────────────────────────────────────────────────────


def _build_ch_loss(ch_cfg: Any) -> CahnHilliardEnergyCurveLoss:
    return CahnHilliardEnergyCurveLoss(
        radii=list(ch_cfg.radii),
        grid_resolution=int(ch_cfg.grid_resolution),
        kappa=float(ch_cfg.kappa),
        combine=str(ch_cfg.combine),
        soft_max_beta=float(ch_cfg.soft_max_beta),
        support_factor=float(ch_cfg.support_factor),
        landscape_chunk_size=int(ch_cfg.chunk),
        square_bbox=bool(ch_cfg.square_bbox),
        margin=float(ch_cfg.margin),
        eps=float(ch_cfg.eps),
        min_cells_per_type=int(ch_cfg.min_cells_per_type),
    )


def _build_voronoi_loss(vor_cfg: Any) -> VoronoiPhasePairEnergyLoss:
    soft_beta = getattr(vor_cfg, "soft_beta", None)
    soft_beta = None if soft_beta is None else float(soft_beta)
    return VoronoiPhasePairEnergyLoss(
        transition_width=float(vor_cfg.transition_width),
        grid_resolution=int(vor_cfg.grid_resolution),
        kappa=float(vor_cfg.kappa),
        soft_beta=soft_beta,
        square_bbox=bool(vor_cfg.square_bbox),
        margin=float(vor_cfg.margin),
        eps=float(vor_cfg.eps),
        min_cells_per_type=int(vor_cfg.min_cells_per_type),
        chunk=int(vor_cfg.chunk),
    )


def _build_directional_loss(dir_cfg: Any) -> DirectionalMetricLoss:
    """Instantiate the directional loss module from config."""
    soft_beta = getattr(dir_cfg, "soft_beta", None)
    soft_beta = None if soft_beta is None else float(soft_beta)
    return DirectionalMetricLoss(
        n_target=int(dir_cfg.n_target),
        neighbor_radius=float(dir_cfg.neighbor_radius),
        coherence_radius=float(dir_cfg.coherence_radius),
        trans_beta=float(dir_cfg.trans_beta),
        soft_beta=soft_beta,
        eps=float(getattr(dir_cfg, "eps", 1e-6)),
        length_weight=float(getattr(dir_cfg, "length_weight", 1.0)),
        pairwise_weight=float(getattr(dir_cfg, "pairwise_weight", 1.0)),
        min_valid_targets=int(getattr(dir_cfg, "min_valid_targets", 2)),
        density_length_gate=bool(getattr(dir_cfg, "density_length_gate", False)),
        density_length_beta=float(getattr(dir_cfg, "density_length_beta", 4.0)),
        density_radius_gate=bool(getattr(dir_cfg, "density_radius_gate", False)),
        density_radius_beta=float(getattr(dir_cfg, "density_radius_beta", 4.0)),
    )


def compute_section_ch_voronoi_losses(
    preds: np.ndarray,                       # [S, N, 2]
    batch: SliceBatch,
    ch_loss: CahnHilliardEnergyCurveLoss,
    voronoi_loss: VoronoiPhasePairEnergyLoss,
    device: Any,
) -> pd.DataFrame:
    """Per-sample CH energy-curve and Voronoi pair energy losses for a section.

    Returns one row per denoising sample with columns ``sample_index``,
    ``ch_energy_curve_loss``, ``ch_n_types``, ``voronoi_phase_pair_ch_energy_loss``
    and ``voronoi_n_pairs``.
    """
    gt_positions = batch.gt_positions.astype(np.float32)
    cell_class_int = np.asarray(batch.cell_class_int)

    rows = []
    for s in range(preds.shape[0]):
        ch_val, n_types = ch_energy_curve_loss_per_sample(
            preds[s], gt_positions, cell_class_int, ch_loss, device
        )
        vor_val, n_pairs = voronoi_pair_energy_loss_per_sample(
            preds[s], gt_positions, cell_class_int, voronoi_loss, device
        )
        rows.append({
            "sample_index": s,
            "ch_energy_curve_loss": ch_val,
            "ch_n_types": n_types,
            "voronoi_phase_pair_ch_energy_loss": vor_val,
            "voronoi_n_pairs": n_pairs,
        })
    return pd.DataFrame(rows)


def aggregate_ch_voronoi_across_sections(
    section_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average the per-sample CH/Voronoi losses across sections."""
    if not section_tables:
        return pd.DataFrame()
    combined = pd.concat(section_tables, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    metric_cols = [
        "ch_energy_curve_loss",
        "voronoi_phase_pair_ch_energy_loss",
    ]
    grouped = (
        combined.groupby("cell_section", as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    if "index" in grouped.columns:
        grouped = grouped.drop(columns=["index"])

    overall = {"cell_section": "ALL"}
    for col in metric_cols:
        overall[f"{col}_mean"] = float(combined[col].mean())
        overall[f"{col}_std"] = float(combined[col].std(ddof=0))
    overall_row = pd.DataFrame([overall])
    return pd.concat([grouped, overall_row], ignore_index=True)


def plot_ch_voronoi_summary(
    summary_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Bar chart of the mean CH and Voronoi pair losses (ALL + per section)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No CH/Voronoi data to plot.")
        return

    save_path = Path(save_path)
    plot_df = summary_df.copy()
    labels = plot_df["cell_section"].tolist()
    ch_means = plot_df["ch_energy_curve_loss_mean"].to_numpy()
    ch_stds = plot_df["ch_energy_curve_loss_std"].to_numpy()
    vor_means = plot_df["voronoi_phase_pair_ch_energy_loss_mean"].to_numpy()
    vor_stds = plot_df["voronoi_phase_pair_ch_energy_loss_std"].to_numpy()

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 5))
    ax.bar(x - w / 2, ch_means, w, yerr=ch_stds, capsize=3,
           color="tab:red", alpha=0.85, label="CH energy-curve loss")
    ax.bar(x + w / 2, vor_means, w, yerr=vor_stds, capsize=3,
           color="tab:purple", alpha=0.85, label="Voronoi pair CH energy loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Loss (mean over samples, in [0, 1))")
    ax.set_title("Cahn-Hilliard energy comparison per section (avg over samples)")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved CH/Voronoi summary plot → {save_path}")


def plot_cross_sample_spread_summary(
    spread_summary: pd.DataFrame,
    overall_mean: float,
    save_path: Path,
) -> None:
    """Bar chart of mean per-cell cross-sample spread per section + overall avg."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if spread_summary.empty:
        print("[testing_pipeline] No cross-sample spread data to plot.")
        return

    save_path = Path(save_path)
    plot_df = spread_summary.copy()
    labels = plot_df["cell_section"].tolist()
    means = plot_df["mean_cross_sample_spread"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color="tab:cyan", alpha=0.85, label="Per section (mean over cells)")
    ax.axhline(
        overall_mean, color="black", linestyle="--", linewidth=1.5,
        label=f"Overall avg ({overall_mean:.4g})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean pairwise position difference across samples")
    ax.set_title(
        "Cross-sample positional spread per cell\n"
        "(after Procrustes alignment; averaged over cells)"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved cross-sample spread plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Position-MSE + directional (pairwise / length) scalar losses
# ─────────────────────────────────────────────────────────────────────────────


def _make_minimal_holder(
    positions: np.ndarray,        # (N, 2)
    node_features: np.ndarray,    # (N, F)
    mask: np.ndarray,             # (N,) bool
    device: Any,
) -> DataHolder:
    """Build a minimal 1-batch ``DataHolder`` for the scalar losses.

    The vanilla / directional loss modules only read ``.positions``,
    ``.node_mask`` and ``.node_features`` (no ``.mask()`` call), so we just
    stack a batch dimension and move to ``device``.
    """
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=device)
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)                                  # (1, N, 2)
    feat = torch.as_tensor(np.asarray(node_features), dtype=torch.float32, device=device)
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)                                # (1, N, F)
    m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=device)
    if m.ndim == 1:
        m = m.unsqueeze(0)                                      # (1, N)
    return DataHolder(
        positions=pos,
        node_features=feat,
        diffusion_time=0,
        node_mask=m,
    )


def _valid_cells_mask(cell_class_int: np.ndarray) -> np.ndarray:
    """All-ones mask (these scalar losses are global, not per-class)."""
    return np.ones(int(np.asarray(cell_class_int).shape[0]), dtype=bool)


def _slice_position_mse(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    cell_class_int: np.ndarray,
    device: Any,
) -> float:
    """Vanilla position MSE for one (pred, GT) pair, mirroring ``test_vanilla_loss``.

    Computes the MSE between the masked pairwise distance matrices
    ``cdist(gt[mask], gt[mask])`` and ``cdist(pred[mask], pred[mask])`` (the
    training loss is rotation/translation-invariant). A single scalar per
    sample is returned.
    """
    mask = _valid_cells_mask(cell_class_int)
    if mask.sum() < 2:
        return float("nan")

    dummy_feat = np.zeros((int(mask.sum()), 1), dtype=np.float32)
    pos_mse = PositionMSELoss()
    pred_h = _make_minimal_holder(pred_positions, dummy_feat, mask, device)
    true_h = _make_minimal_holder(gt_positions, dummy_feat, mask, device)
    with torch.no_grad():
        loss, _ = pos_mse(true_h, pred_h, train_stage=False, log=False)
    return float(loss.item())


def _slice_directional_losses(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    node_features: np.ndarray,
    cell_class_int: np.ndarray,
    device: Any,
    *,
    directional_module: DirectionalMetricLoss,
    sample_seed: int,
) -> tuple[float, float]:
    """Directional length & pairwise losses for one (pred, GT) pair.

    Uses the GT ``node_features`` (transcriptome) for both sides, mirroring
    ``DirectionalMetricLoss.forward``. The target subsampling is seeded
    deterministically per sample so the reported scalars are reproducible.
    """
    mask = _valid_cells_mask(cell_class_int)
    if mask.sum() < directional_module.min_valid_targets:
        return float("nan"), float("nan")

    feat = np.asarray(node_features, dtype=np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    true_h = _make_minimal_holder(gt_positions, feat, mask, device)
    pred_h = _make_minimal_holder(pred_positions, feat, mask, device)

    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if str(device) != "cpu" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(int(sample_seed))
    try:
        with torch.no_grad():
            directional_module(pred_h, true_h, train_stage=False, log=False)
        length_val = float(directional_module._last_length)
        pairwise_val = float(directional_module._last_pairwise)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

    if length_val < 0 or pairwise_val < 0:
        return float("nan"), float("nan")
    return length_val, pairwise_val


def compute_section_scalar_losses(
    batch: "SliceBatch",
    pred_positions: np.ndarray,           # (S, N, 2)
    gt_positions: np.ndarray,             # (N, 2)
    cell_class_int: np.ndarray,           # (N,)
    device: Any,
    *,
    compute_position_mse: bool,
    compute_directional: bool,
    directional_module: Optional[DirectionalMetricLoss],
    directional_base_seed: int,
) -> pd.DataFrame:
    """Per-sample scalar losses (position MSE, directional length/pairwise).

    Returns a long DataFrame with one row per (sample, metric_name).
    """
    records: list[dict[str, Any]] = []
    S = int(pred_positions.shape[0])

    node_features = None
    if compute_directional:
        node_features = np.asarray(batch.holder.node_features, dtype=np.float32)
        if node_features.ndim == 3 and node_features.shape[0] == 1:
            node_features = node_features[0]

    for s in range(S):
        pred = pred_positions[s]
        if compute_position_mse:
            pmse = _slice_position_mse(pred, gt_positions, cell_class_int, device)
            records.append({"sample_index": s, "metric": "position_mse", "value": pmse})
        if compute_directional and directional_module is not None:
            length_val, pairwise_val = _slice_directional_losses(
                pred, gt_positions, node_features, cell_class_int, device,
                directional_module=directional_module,
                sample_seed=int(directional_base_seed) + s,
            )
            records.append({"sample_index": s, "metric": "directional_length", "value": length_val})
            records.append({"sample_index": s, "metric": "directional_pairwise", "value": pairwise_val})

    return pd.DataFrame.from_records(records, columns=["sample_index", "metric", "value"])


def aggregate_scalar_losses_across_sections(
    per_section: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-section scalar-loss records into a section × metric summary.

    Output columns: section, metric, mean, std, n_samples.
    """
    frames = []
    for section_id, df in per_section.items():
        if df.empty:
            continue
        agg = (
            df.groupby("metric")["value"]
            .agg(mean="mean", std="std", n_samples="count")
            .reset_index()
            .assign(section=int(section_id))
        )
        frames.append(agg[["section", "metric", "mean", "std", "n_samples"]])
    if not frames:
        return pd.DataFrame(columns=["section", "metric", "mean", "std", "n_samples"])
    return pd.concat(frames, ignore_index=True)


def plot_scalar_losses_summary(
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    save_path: Path,
) -> None:
    """Grouped bar chart of mean scalar losses per section, per metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        print("[testing_pipeline] No scalar-loss data to plot.")
        return

    save_path = Path(save_path)
    metrics = list(overall["metric"]) if not overall.empty else list(summary["metric"].unique())
    sections = sorted(summary["section"].unique())
    x = np.arange(len(sections))
    width = 0.8 / max(1, len(metrics))

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(sections)), 5))
    for i, metric in enumerate(metrics):
        vals = []
        for sec in sections:
            row = summary[(summary["section"] == sec) & (summary["metric"] == metric)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + i * width, vals, width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([str(s) for s in sections])
    ax.set_xlabel("Section")
    ax.set_ylabel("Loss (mean over samples)")
    ax.set_title("Scalar losses per section (avg over samples)")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved scalar-losses summary plot → {save_path}")



# ─────────────────────────────────────────────────────────────────────────────
# Per-checkpoint orchestration
# ─────────────────────────────────────────────────────────────────────────────


def run_checkpoint_pipeline(
    cfg: DictConfig,
    checkpoint_path: Path,
    datamodule: Any,
    dataset_infos: Any,
) -> Path:
    """Run the full statistical testing pipeline for one checkpoint."""
    pipe_cfg = cfg.test.pipeline
    wasserstein_cfg = pipe_cfg.wasserstein_analysis
    ch_voronoi_cfg = pipe_cfg.ch_voronoi_analysis
    scalar_cfg = pipe_cfg.scalar_losses_analysis
    num_samples = int(pipe_cfg.num_samples)
    seed_start = int(pipe_cfg.seed_start)
    min_cells = int(pipe_cfg.min_cells_per_class)
    batch_size = int(cfg.test.batch_size)

    output_dir = pipeline_output_dir(cfg, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "effective_config.yaml")

    device = _resolve_device(str(pipe_cfg.device))
    ch_loss = _build_ch_loss(ch_voronoi_cfg.cahn_hilliard)
    voronoi_loss = _build_voronoi_loss(ch_voronoi_cfg.voronoi)

    compute_position_mse = bool(getattr(scalar_cfg, "compute_position_mse", True))
    compute_directional = bool(getattr(scalar_cfg, "compute_directional", True))
    directional_module: Optional[DirectionalMetricLoss] = None
    if compute_directional:
        directional_module = _build_directional_loss(scalar_cfg.directional)
    directional_base_seed = int(getattr(scalar_cfg.directional, "base_seed", 0))
    print("=" * 78)
    print(f"[testing_pipeline] Checkpoint : {checkpoint_path}")
    print(f"[testing_pipeline] Output dir  : {output_dir}")
    print(f"[testing_pipeline] num_samples : {num_samples}")
    print(f"[testing_pipeline] batch_size  : {batch_size}")
    print(f"[testing_pipeline] device      : {device}")
    print("=" * 78)

    load_model_config_from_checkpoint(cfg, checkpoint_path)
    model = load_model(cfg, dataset_infos, str(checkpoint_path), device)

    cell_sections = list_cell_sections(datamodule.test_dataset)
    print(f"[testing_pipeline] Found {len(cell_sections)} cell sections in test split.")

    section_batches = prepare_section_batches(datamodule, dataset_infos, cell_sections)

    print("[testing_pipeline] Phase 1/2: batched sampling (seed-outer, sections-inner)…")
    all_section_preds = run_all_sampling(
        model=model,
        section_batches=section_batches,
        cell_sections=cell_sections,
        num_samples=num_samples,
        seed_start=seed_start,
        batch_size=batch_size,
        device=device,
    )

    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("[testing_pipeline] Phase 2/2: analysis (Wasserstein + CH/Voronoi + spread)…")
    section_wasserstein_tables: list[pd.DataFrame] = []
    section_ch_voronoi_tables: list[pd.DataFrame] = []
    section_spread_rows: list[dict] = []
    section_spread_per_cell_tables: list[pd.DataFrame] = []
    section_scalar_losses_tables: list[pd.DataFrame] = []

    for cell_section in tqdm(
        cell_sections, desc="Analysis", unit="section",
        disable=False, mininterval=0.5,
    ):
        section_dir = output_dir / "sections" / _safe_section_dirname(cell_section)
        section_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = section_dir / "predictions.csv"
        gt_csv = section_dir / "ground_truth.csv"

        batch = section_batches[cell_section]
        preds = all_section_preds[cell_section]
        save_predictions_csv(
            preds=preds,
            batch=batch,
            seed_start=seed_start,
            output_csv=predictions_csv,
            gt_csv=gt_csv,
        )

        gt_df = predictions_to_gt_dataframe(batch)
        cell_classes = cell_classes_with_min_cells(gt_df, min_cells)
        if not cell_classes:
            print(
                f"[testing_pipeline] Skipping Wasserstein for {cell_section!r}: "
                f"no cell class with >= {min_cells} cells."
            )
        else:
            section_wasserstein = compute_section_wasserstein_metrics(
                preds=preds,
                batch=batch,
                cell_classes=cell_classes,
                wasserstein_cfg=wasserstein_cfg,
            )
            if not section_wasserstein.empty:
                section_wasserstein.insert(0, "cell_section", str(cell_section))
                section_wasserstein.to_csv(
                    section_dir / "wasserstein_per_class.csv", index=False
                )
                section_wasserstein_tables.append(section_wasserstein)

        section_ch_voronoi = compute_section_ch_voronoi_losses(
            preds=preds,
            batch=batch,
            ch_loss=ch_loss,
            voronoi_loss=voronoi_loss,
            device=device,
        )
        if not section_ch_voronoi.empty:
            section_ch_voronoi.insert(0, "cell_section", str(cell_section))
            section_ch_voronoi.to_csv(
                section_dir / "ch_voronoi_losses_per_sample.csv", index=False
            )
            section_ch_voronoi_tables.append(section_ch_voronoi)

        spread_mean, spread_per_cell = compute_cross_sample_position_spread(
            preds=preds,
            gt_positions=batch.gt_positions,
            device=device,
        )
        section_spread_rows.append({
            "cell_section": str(cell_section),
            "mean_cross_sample_spread": spread_mean,
            "n_cells": int(spread_per_cell.shape[0]),
            "n_samples": int(preds.shape[0]),
        })
        spread_per_cell_df = pd.DataFrame({
            "cell_section": str(cell_section),
            "cell_ID": batch.cell_ids,
            "cell_class": [str(c) for c in batch.cell_class_labels],
            "mean_cross_sample_position_spread": spread_per_cell,
        })
        spread_per_cell_df.to_csv(
            section_dir / "cross_sample_spread_per_cell.csv", index=False
        )
        section_spread_per_cell_tables.append(spread_per_cell_df)

        if compute_position_mse or compute_directional:
            section_scalar = compute_section_scalar_losses(
                batch=batch,
                pred_positions=preds,
                gt_positions=batch.gt_positions.astype(np.float32),
                cell_class_int=np.asarray(batch.cell_class_int),
                device=device,
                compute_position_mse=compute_position_mse,
                compute_directional=compute_directional,
                directional_module=directional_module,
                directional_base_seed=directional_base_seed,
            )
            if not section_scalar.empty:
                section_scalar.insert(0, "cell_section", str(cell_section))
                section_scalar.to_csv(
                    section_dir / "scalar_losses_per_sample.csv", index=False
                )
                section_scalar_losses_tables.append(section_scalar)

    wasserstein_summary = aggregate_wasserstein_across_sections(
        section_wasserstein_tables
    )
    wasserstein_csv = output_dir / "wasserstein_summary.csv"
    wasserstein_plot = output_dir / "wasserstein_summary_by_cell_class.png"
    if not wasserstein_summary.empty:
        wasserstein_summary.to_csv(wasserstein_csv, index=False)
        plot_wasserstein_summary(
            wasserstein_summary,
            wasserstein_plot,
            n_sections_total=len(cell_sections),
        )
    else:
        print("[testing_pipeline] No Wasserstein summaries produced.")

    ch_voronoi_summary = aggregate_ch_voronoi_across_sections(
        section_ch_voronoi_tables
    )
    ch_voronoi_csv = output_dir / "ch_voronoi_summary.csv"
    ch_voronoi_plot = output_dir / "ch_voronoi_summary.png"
    if not ch_voronoi_summary.empty:
        ch_voronoi_summary.to_csv(ch_voronoi_csv, index=False)
        plot_ch_voronoi_summary(ch_voronoi_summary, ch_voronoi_plot)
    else:
        print("[testing_pipeline] No CH/Voronoi summaries produced.")

    if section_spread_rows:
        spread_summary = pd.DataFrame(section_spread_rows)
        spread_summary.to_csv(output_dir / "cross_sample_spread_summary.csv", index=False)
        overall_spread = float(spread_summary["mean_cross_sample_spread"].mean())
        print(
            f"[testing_pipeline] Mean cross-sample position spread "
            f"(avg over cells & sections): {overall_spread:.6g}"
        )
        plot_cross_sample_spread_summary(
            spread_summary, overall_spread,
            output_dir / "cross_sample_spread_summary.png",
        )
        if section_spread_per_cell_tables:
            all_per_cell_spread = pd.concat(
                section_spread_per_cell_tables, ignore_index=True
            )
            all_per_cell_spread.to_csv(
                output_dir / "cross_sample_spread_per_cell_all_sections.csv",
                index=False,
            )
    else:
        print("[testing_pipeline] No cross-sample spread produced.")

    scalar_summary = aggregate_scalar_losses_across_sections(
        {i: df for i, df in enumerate(section_scalar_losses_tables)}
    )
    scalar_csv = output_dir / "scalar_losses_summary.csv"
    scalar_plot = output_dir / "scalar_losses_summary.png"
    if not scalar_summary.empty:
        scalar_summary.to_csv(scalar_csv, index=False)
        overall_scalar = (
            scalar_summary.groupby("metric")["mean"]
            .agg(overall_mean="mean")
            .reset_index()
        )
        for _, r in overall_scalar.iterrows():
            print(
                f"[testing_pipeline] {r['metric']} (avg over samples & sections): "
                f"{r['overall_mean']:.6g}"
            )
        plot_scalar_losses_summary(scalar_summary, overall_scalar, scalar_plot)
    else:
        print("[testing_pipeline] No scalar-loss summaries produced.")

    print(f"[testing_pipeline] Finished checkpoint {checkpoint_path.name}.")
    return output_dir


def run_testing_pipeline(cfg: DictConfig) -> list[Path]:
    """Entry point: run the pipeline for every resolved checkpoint.

    After computing the per-checkpoint summaries, if more than one checkpoint
    was processed it also writes a combined comparison PNG
    (``checkpoint_comparison.png``) and a combined CSV
    (``checkpoint_comparison.csv``) into the shared ``testing_pipeline``
    parent directory, comparing every metric across checkpoints.
    """
    from datasets.data_module import DataModule, Infos

    cfg.general.mode = "test_only"
    cfg.validation.if_validate = False

    checkpoint_paths = resolve_checkpoints(cfg)
    datamodule = DataModule(cfg)
    dataset_infos = Infos(datamodule, cfg)

    output_dirs = []
    for checkpoint_path in checkpoint_paths:
        output_dirs.append(
            run_checkpoint_pipeline(cfg, checkpoint_path, datamodule, dataset_infos)
        )

    if len(output_dirs) > 1:
        try:
            compare_checkpoints(cfg, checkpoint_paths, output_dirs)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[testing_pipeline] Checkpoint comparison failed: "
                f"{type(exc).__name__}: {exc}"
            )
    return output_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Cross-checkpoint comparison
# ─────────────────────────────────────────────────────────────────────────────


def _checkpoint_label(checkpoint_path: Path) -> str:
    return _checkpoint_stem(checkpoint_path)


def _load_checkpoint_summaries(output_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Load the per-checkpoint summary CSVs produced by ``run_checkpoint_pipeline``."""
    summaries = {
        "wasserstein": output_dir / "wasserstein_summary.csv",
        "ch_voronoi": output_dir / "ch_voronoi_summary.csv",
        "spread": output_dir / "cross_sample_spread_summary.csv",
        "scalar_losses": output_dir / "scalar_losses_summary.csv",
    }
    return {
        key: (pd.read_csv(path) if path.exists() else None)
        for key, path in summaries.items()
    }


def _collect_checkpoint_comparison(
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
) -> pd.DataFrame:
    """Collapse every metric to one scalar per checkpoint into a long table.

    Returns a DataFrame with columns ``checkpoint``, ``metric``, ``value`` where
    each row is a single benchmark number for one checkpoint (no cell-type / no
    per-section split). Metrics (in fixed display order):

      * ``position_mse``
      * ``directional_length``
      * ``directional_pairwise``
      * ``ch_energy_curve_loss``
      * ``voronoi_phase_pair_ch_energy_loss``
      * ``wasserstein_whole_slice_alignment``
      * ``wasserstein_per_class_alignment``
      * ``cross_sample_spread``
    """
    order = [
        "position_mse",
        "directional_length",
        "directional_pairwise",
        "ch_energy_curve_loss",
        "voronoi_phase_pair_ch_energy_loss",
        "wasserstein_whole_slice_alignment",
        "wasserstein_per_class_alignment",
        "cross_sample_spread",
    ]

    rows: list[dict[str, Any]] = []
    for ckpt, out_dir in zip(checkpoint_paths, output_dirs):
        label = _checkpoint_label(ckpt)
        loaded = _load_checkpoint_summaries(out_dir)

        w = loaded.get("wasserstein")
        if w is not None and not w.empty:
            rows.append({"checkpoint": label, "metric": "wasserstein_whole_slice_alignment",
                         "value": float(w["mean_wasserstein_whole_slice_alignment"].mean())})
            rows.append({"checkpoint": label, "metric": "wasserstein_per_class_alignment",
                         "value": float(w["mean_wasserstein_per_class_alignment"].mean())})

        cv = loaded.get("ch_voronoi")
        if cv is not None and not cv.empty:
            if "cell_section" in cv.columns:
                all_rows = cv[cv["cell_section"] == "ALL"]
            else:
                all_rows = cv
            if all_rows.empty:
                all_rows = cv
            rows.append({"checkpoint": label, "metric": "ch_energy_curve_loss",
                         "value": float(all_rows["ch_energy_curve_loss_mean"].mean())})
            rows.append({"checkpoint": label, "metric": "voronoi_phase_pair_ch_energy_loss",
                         "value": float(all_rows["voronoi_phase_pair_ch_energy_loss_mean"].mean())})

        sp = loaded.get("spread")
        if sp is not None and not sp.empty:
            rows.append({"checkpoint": label, "metric": "cross_sample_spread",
                         "value": float(sp["mean_cross_sample_spread"].mean())})

        sl = loaded.get("scalar_losses")
        if sl is not None and not sl.empty:
            for metric, sub in sl.groupby("metric"):
                rows.append({"checkpoint": label, "metric": str(metric),
                             "value": float(sub["mean"].mean())})

    df = pd.DataFrame.from_records(rows, columns=["checkpoint", "metric", "value"])
    if df.empty:
        return df
    # Drop any metrics not in the canonical order, then sort by it.
    df = df[df["metric"].isin(order)].copy()
    df["_order"] = df["metric"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values(["_order", "checkpoint"]).drop(columns=["_order"]).reset_index(drop=True)
    return df


def _save_combined_comparison_csv(
    comparison: pd.DataFrame,
    save_path: Path,
) -> None:
    """Write the unified long checkpoint-comparison table to CSV.

    Columns: checkpoint, metric, value, normalized (per-metric min-max across
    checkpoints so the CSV mirrors the bar plot).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        with save_path.open("w") as fh:
            fh.write("# checkpoint comparison\n# (no data)\n")
        print(f"[testing_pipeline] Saved checkpoint comparison CSV → {save_path}")
        return

    df = comparison.copy()
    df["normalized_score"] = np.nan
    for metric, sub in df.groupby("metric"):
        vals = sub["value"].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = hi - lo
        # All benchmarked metrics are "lower is better": invert so 1 = best (min).
        if not np.isfinite(denom) or denom <= 0:
            df.loc[sub.index, "normalized_score"] = 1.0
        else:
            df.loc[sub.index, "normalized_score"] = (hi - vals) / denom
    df.to_csv(save_path, index=False)
    print(f"[testing_pipeline] Saved checkpoint comparison CSV → {save_path}")


def plot_checkpoint_comparison(
    comparison: pd.DataFrame,
    save_path: Path,
) -> None:
    """One single bar plot comparing every metric across checkpoints.

    Each metric is collapsed to one scalar per checkpoint. Because the metrics
    live on very different scales, each metric is min-max normalised across
    checkpoints (so the worst checkpoint = 0, the best = 1 for that metric)
    before plotting — this lets every metric share one bar plot. The raw value
    is annotated on top of each bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    if comparison.empty:
        print("[testing_pipeline] No checkpoint comparison data to plot.")
        return

    metrics = list(dict.fromkeys(comparison["metric"].tolist()))
    checkpoints = list(dict.fromkeys(comparison["checkpoint"].tolist()))
    n_metrics = len(metrics)
    n_ckpt = len(checkpoints)
    bar_w = 0.8 / max(n_ckpt, 1)
    x = np.arange(n_metrics)

    # Per-metric min-max normalisation across checkpoints (the plot scale).
    # All benchmarked metrics are "lower is better", so we invert: 1 = best (min),
    # 0 = worst (max). This makes taller bars = better models.
    norm_vals: dict[tuple[str, str], float] = {}
    raw_vals: dict[tuple[str, str], float] = {}
    for metric in metrics:
        sub = comparison[comparison["metric"] == metric].set_index("checkpoint")
        sub = sub.reindex(checkpoints)
        vals = sub["value"].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = hi - lo
        for ckpt, v in zip(checkpoints, vals):
            raw_vals[(ckpt, metric)] = float(v) if np.isfinite(v) else np.nan
            if not np.isfinite(denom) or denom <= 0:
                norm_vals[(ckpt, metric)] = 1.0
            else:
                norm_vals[(ckpt, metric)] = float((hi - v) / denom)

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * n_metrics), 6))
    cmap = plt.get_cmap("tab10")
    for ci, ckpt in enumerate(checkpoints):
        heights = np.array([norm_vals[(ckpt, m)] for m in metrics], dtype=float)
        offsets = x + (ci - (n_ckpt - 1) / 2) * bar_w
        ax.bar(offsets, heights, bar_w, label=ckpt, color=cmap(ci % 10), alpha=0.88,
               edgecolor="black", linewidth=0.4)
        for off, m, h in zip(offsets, metrics, heights):
            v = raw_vals[(ckpt, m)]
            if np.isfinite(v):
                ax.text(off, h + 0.01, f"{v:.2g}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Per-metric normalised score (1 = best, 0 = worst across checkpoints; all metrics lower-is-better)")
    ax.set_title("Model benchmark — metrics comparison across checkpoints (taller = better; raw values labelled)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85, title="checkpoint")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved checkpoint comparison plot → {save_path}")


def compare_checkpoints(
    cfg: DictConfig,
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
) -> Path:
    """Load every checkpoint's summaries and write a combined CSV + comparison PNG.

    Artefacts are written into the shared ``testing_pipeline`` parent
    directory (the common ancestor of all per-checkpoint output dirs).
    """
    base = cfg.test.save_dir
    if base is None:
        base = str(checkpoint_paths[0].parent)
    parent_dir = Path(base) / cfg.general.name / "testing_pipeline"
    parent_dir.mkdir(parents=True, exist_ok=True)

    combined = _collect_checkpoint_comparison(checkpoint_paths, output_dirs)
    _save_combined_comparison_csv(combined, parent_dir / "checkpoint_comparison.csv")
    plot_checkpoint_comparison(combined, parent_dir / "checkpoint_comparison.png")
    return parent_dir


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    assert _DIFFUSION_REPO_ROOT == REPO_ROOT
    run_testing_pipeline(cfg)


if __name__ == "__main__":
    main()
