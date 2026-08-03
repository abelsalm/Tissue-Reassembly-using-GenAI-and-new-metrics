"""Statistical testing pipeline for LUNA checkpoints.

For each checkpoint listed in ``configs/test/default.yaml`` the pipeline:

1. Runs ``n`` denoising samples (different seeds) on every ``cell_section`` of
   the test split. Sampling is organised **seed-first**: for each seed, all
   sections are inferred in batches of ``test.batch_size`` graphs; analysis
   runs only after every seed/section pair has been collected.
2. Computes **isotropic IMQ-MMD distances** (from ``metrics/train_mmds.py``,
   no anisotropy) between predicted and GT point clouds, under two ICP
   registrations:

   * **global ICP** — align the full predicted cloud to the GT slice once.
   * **per-class ICP** — independently ICP-align each cell-type cloud to its
     GT class cloud (then stitch class clouds for the whole-slice term).

   For each registration we report three scalars (averaged over denoising
   samples, then over sections):

   * ``mmd_whole_slice`` — isotropic IMQ-MMD on the full point cloud
     (bandwidths = median pairwise × ``whole_slice_band_mults``).
   * ``mmd_pair_dist`` — IMQ-MMD on pairwise-distance distributions
     (bandwidths = median pair-dist × ``pair_dist_band_mults``), averaged
     over cell types.
   * ``mmd_spatial`` — isotropic IMQ-MMD on per-class spatial point clouds
     (bandwidths = median pairwise × ``spatial_band_mults``), averaged
     over cell types.

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

5. Computes **grid transcript-count metrics** after similarity Procrustes
   alignment (rotation + translation + scale) of each predicted sample onto
   the GT — the same barycenter-based fit used in ``metrics/train_mmds.py``.
   All ``S`` samples are Procrustes-aligned in one batched SVD, bin counts
   use a single GPU ``scatter_add`` per grid size, and soft Spearman is
   evaluated on the full ``(S, F, n_squares)`` tensor (chunked over genes).
   The aligned cloud and the GT are each binned into square grids
   (``8×8``, ``16×16``, … from config). For every grid cell the per-gene
   transcript counts are the sum of ``node_features`` over cells that fall
   in that square. Two scalars are reported per sample (then averaged over
   samples / sections / grid sizes):

   * ``grid_transcript_diff`` — mean absolute difference (MAE) of gene
     counts across grid squares and genes.
   * ``grid_transcript_soft_spearman`` — soft Spearman correlation of the
     per-square count vectors between pred and GT, computed **per gene**
     (spatial pattern match), then averaged over genes. Soft ranks reuse
     ``soft_spearman_distance`` from ``train_spatial_transcriptomics``
     (reported here as the correlation ``1 - distance``).

Usage
-----
    python metrics/test_testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        general.mode=test_only

    python metrics/test_testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        test.checkpoint_path=/path/to/epoch=19999.ckpt \\
        test.pipeline.num_samples=64

    python metrics/test_testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        'test.checkpoint_paths=[/path/a/epoch=7999.ckpt,/path/b/epoch=12000.ckpt]'
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

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
from metrics.train_spatial_transcriptomics import (  # noqa: E402
    MultiRadiusNeighborhoodLoss,
    soft_spearman_distance,
)
from metrics.train_mmds import (  # noqa: E402
    PAIR_DIST_MMD_MAX_SAMPLES,
    mmd2_imq_iso,
    pair_dist_mmd_loss,
    pair_dist_mmd_sigmas,
    precompute_pair_dist_mmd_gt,
    precompute_whole_slice_mmd_cache,
)
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


def _checkpoint_output_id(checkpoint_path: Path) -> str:
    """Unique artefact folder name for a checkpoint (avoids cross-session collisions)."""
    stem = _checkpoint_stem(checkpoint_path)
    parent = checkpoint_path.parent
    if parent.name == "checkpoints":
        run_label = parent.parent.name
    else:
        run_label = parent.name
    return f"{run_label}_{stem}"


def resolve_checkpoints(cfg: DictConfig) -> list[Path]:
    """Resolve one or many checkpoint paths from ``cfg.test``.

    Precedence (same as ``main.test_model``):
    1) ``test.checkpoint_paths`` — full paths, may span different training sessions
    2) ``test.checkpoint_path`` — single checkpoint
    3) ``test.checkpoints_parent_dir`` + ``test.checkpoints_name_list``
    """
    explicit_paths = getattr(cfg.test, "checkpoint_paths", None)
    if explicit_paths:
        if isinstance(explicit_paths, str):
            explicit_paths = [explicit_paths]
        paths = [Path(str(p)) for p in explicit_paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint(s): " + ", ".join(str(p) for p in missing)
            )
        return paths

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
    return Path(base) / cfg.general.name / "testing_pipeline" / _checkpoint_output_id(checkpoint_path)


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
# ICP-based rigid alignment helpers
# ─────────────────────────────────────────────────────────────────────────────


def _symmetric_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """Mean bidirectional nearest-neighbour distance between two point sets."""
    from scipy.spatial import cKDTree

    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("inf")
    tree_b = cKDTree(points_b)
    tree_a = cKDTree(points_a)
    ab = tree_b.query(points_a, k=1)[0]
    ba = tree_a.query(points_b, k=1)[0]
    return float(ab.mean() + ba.mean())


def _procrustes_rigid(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigidly align ``target`` onto ``base`` (rotation + translation, no scaling).

    Closed-form via SVD on the centred clouds. Cell order is used as the
    correspondence, but for ICP that correspondence is the *nearest-neighbour*
    pairing of the previous iteration, not the original cell index.
    """
    from scipy.linalg import svd

    base_mean = base.mean(axis=0)
    target_mean = target.mean(axis=0)
    base_c = base - base_mean
    target_c = target - target_mean
    U, _, Vt = svd(target_c.T @ base_c)
    R = U @ Vt
    return target_c @ R + base_mean 


def align_point_clouds_icp(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    with_reflection: bool = True,
    n_init_rotations: int = 12,
) -> np.ndarray:
    """Rigidly align ``target`` to ``reference`` via global Iterative Closest Point.

    ICP does not assume a known cell-to-cell correspondence: each iteration
    matches every point to its nearest neighbour in the reference cloud, then
    solves the optimal rigid transform (rotation + translation) on those
    pairings via Procrustes/SVD. Translation is optimised jointly with rotation
    (the cloud is re-centred onto the matched reference points every step), so
    the sample is genuinely moved, not just rotated.

    Plain ICP is sensitive to its initialisation (it easily falls into a local
    minimum when the target starts far from the reference). To make the search
    global we run ICP from several seeds — a coarse grid of initial rotations
    (``n_init_rotations`` angles on ``[0, 2π)``), each combined with the
    optional x/y reflection — and keep the run with the smallest final
    symmetric Chamfer distance. This recovers mirror-symmetric and
    large-rotation alignments that single-start ICP misses.
    """
    from scipy.spatial import cKDTree

    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape[0] == 0:
        return np.asarray(target, dtype=np.float32)
    if target.shape[0] == 0:
        return np.asarray(target, dtype=np.float32)

    ref_tree = cKDTree(reference)
    ref_mean = reference.mean(axis=0)
    tgt_centered = target - target.mean(axis=0)

    def _run_icp(seed_points: np.ndarray) -> tuple[np.ndarray, float]:
        current = np.asarray(seed_points, dtype=np.float64).copy()
        prev_error = float("inf")
        for _ in range(max_iter):
            dists, idxs = ref_tree.query(current, k=1)
            matched = reference[idxs]
            current = _procrustes_rigid(matched, current)
            error = float(dists.mean())
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        final_err = _symmetric_chamfer_distance(current, reference)
        return current, final_err

    # Build multi-start seeds: coarse rotation grid (centred at ref mean) x reflection.
    flip_axes: list[int | None] = [None] if not with_reflection else [None, 0, 1]
    angles = np.linspace(0.0, 2.0 * np.pi, n_init_rotations, endpoint=False)

    seeds: list[np.ndarray] = []
    for flip_axis in flip_axes:
        candidate = tgt_centered
        if flip_axis is not None:
            candidate = candidate.copy()
            candidate[:, flip_axis] *= -1.0
        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
            seeds.append(candidate @ rot.T + ref_mean)

    best_aligned: np.ndarray | None = None
    best_error = float("inf")
    for seed in seeds:
        aligned, err = _run_icp(seed)
        if err < best_error:
            best_error = err
            best_aligned = aligned

    assert best_aligned is not None
    return np.asarray(best_aligned, dtype=np.float32)


def align_samples_to_reference(
    samples: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """ICP-align every sample onto ``reference`` (no per-cell correspondence).

    ``samples`` has shape ``(S, N, 2)`` and ``reference`` shape ``(N, 2)``;
    each sample is aligned independently using
    :func:`align_point_clouds_icp`.
    """
    samples = np.asarray(samples, dtype=np.float32)
    aligned = np.empty_like(samples)
    for i in range(samples.shape[0]):
        aligned[i] = align_point_clouds_icp(reference, samples[i])
    return aligned


def align_samples_to_reference_torch(
    samples: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Batched rotation-only Procrustes alignment of every sample onto ``reference``.

    Vectorised GPU implementation of rotation-only Procrustes (no axis-reflection
    search). Used for cross-sample spread metrics.

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
# Isotropic IMQ-MMD distances (ICP global + ICP per-class)
# ─────────────────────────────────────────────────────────────────────────────


def _class_mask_for_sample(batch: SliceBatch, cell_class: str) -> np.ndarray:
    return np.asarray([str(c) == str(cell_class) for c in batch.cell_class_labels])


def _to_torch_xy(points: np.ndarray, device: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(points, dtype=np.float32), dtype=torch.float32, device=device)


def _iso_mmd(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    band_mults: tuple[float, ...],
) -> float:
    """Isotropic IMQ-MMD between two 2-D point clouds (no anisotropy)."""
    if pred_xy.shape[0] < 2 or gt_xy.shape[0] < 2:
        return float("nan")
    cache = precompute_whole_slice_mmd_cache(gt_xy, band_mults)
    if cache is None:
        return float("nan")
    return float(mmd2_imq_iso(pred_xy, cache.G, cache.sigmas, cache.ky_offdiag).item())


def _pair_dist_mmd(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    band_mults: tuple[float, ...],
    max_samples: int,
) -> float:
    """IMQ-MMD between pairwise-distance distributions of two point clouds."""
    if pred_xy.shape[0] < 2 or gt_xy.shape[0] < 2:
        return float("nan")
    dist_sigmas = pair_dist_mmd_sigmas(gt_xy, band_mults)
    gt_samples, gt_self = precompute_pair_dist_mmd_gt(gt_xy, dist_sigmas, max_samples)
    val = pair_dist_mmd_loss(pred_xy, gt_samples, gt_self, dist_sigmas, max_samples)
    return float(val.item())


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _mmd_triplet_on_clouds(
    *,
    pred_full: torch.Tensor,
    gt_full: torch.Tensor,
    class_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    whole_slice_band_mults: tuple[float, ...],
    spatial_band_mults: tuple[float, ...],
    pair_dist_band_mults: tuple[float, ...],
    pair_dist_max_samples: int,
) -> dict[str, float]:
    """Compute whole-slice / spatial / pair-dist MMD for one aligned sample.

    ``class_pairs`` is a list of ``(pred_class_xy, gt_class_xy)`` tensors used
    for the per-class spatial and pair-distance terms. ``pred_full`` /
    ``gt_full`` are the clouds used for the whole-slice term (globally
    aligned full clouds, or the concatenation of independently aligned
    class clouds).
    """
    whole = _iso_mmd(pred_full, gt_full, whole_slice_band_mults)

    spatial_vals: list[float] = []
    pair_vals: list[float] = []
    for X, G in class_pairs:
        spatial_vals.append(_iso_mmd(X, G, spatial_band_mults))
        pair_vals.append(
            _pair_dist_mmd(X, G, pair_dist_band_mults, pair_dist_max_samples)
        )
    return {
        "mmd_whole_slice": whole,
        "mmd_spatial": _mean_finite(spatial_vals),
        "mmd_pair_dist": _mean_finite(pair_vals),
    }


def compute_section_mmd_metrics(
    preds: np.ndarray,
    batch: SliceBatch,
    cell_classes: list[str],
    mmd_cfg: Any,
    device: Any,
) -> pd.DataFrame:
    """Per-sample isotropic MMDs under global ICP and per-class ICP.

    Returns one row per denoising sample with columns:
    ``sample_index``,
    ``mmd_whole_slice_global``, ``mmd_pair_dist_global``, ``mmd_spatial_global``,
    ``mmd_whole_slice_per_class``, ``mmd_pair_dist_per_class``, ``mmd_spatial_per_class``.
    """
    whole_slice_band_mults = tuple(float(x) for x in mmd_cfg.whole_slice_band_mults)
    spatial_band_mults = tuple(float(x) for x in mmd_cfg.spatial_band_mults)
    pair_dist_band_mults = tuple(float(x) for x in mmd_cfg.pair_dist_band_mults)
    pair_dist_max_samples = int(
        getattr(mmd_cfg, "pair_dist_max_samples", PAIR_DIST_MMD_MAX_SAMPLES)
    )

    gt_np = batch.gt_positions.astype(np.float32)
    gt_full = _to_torch_xy(gt_np, device)
    class_masks = [_class_mask_for_sample(batch, c) for c in cell_classes]
    # Drop empty / tiny classes for per-class terms.
    kept = []
    for mask in class_masks:
        if int(mask.sum()) >= 2:
            kept.append(mask)
    class_masks = kept
    if not class_masks:
        return pd.DataFrame()

    rows = []
    for s in range(preds.shape[0]):
        pred_s = preds[s].astype(np.float32)

        # --- Global ICP: one rigid transform on the full cloud ---
        aligned_global_np = align_point_clouds_icp(gt_np, pred_s)
        aligned_global = _to_torch_xy(aligned_global_np, device)
        global_pairs = [
            (aligned_global[mask], gt_full[mask]) for mask in class_masks
        ]
        global_vals = _mmd_triplet_on_clouds(
            pred_full=aligned_global,
            gt_full=gt_full,
            class_pairs=global_pairs,
            whole_slice_band_mults=whole_slice_band_mults,
            spatial_band_mults=spatial_band_mults,
            pair_dist_band_mults=pair_dist_band_mults,
            pair_dist_max_samples=pair_dist_max_samples,
        )

        # --- Per-class ICP: independent rigid transform per cell type ---
        aligned_classes: list[torch.Tensor] = []
        gt_classes: list[torch.Tensor] = []
        for mask in class_masks:
            class_pred = pred_s[mask]
            class_gt_np = gt_np[mask]
            aligned_c = align_point_clouds_icp(class_gt_np, class_pred)
            aligned_classes.append(_to_torch_xy(aligned_c, device))
            gt_classes.append(gt_full[mask])
        stitched = torch.cat(aligned_classes, dim=0)
        gt_stitched = torch.cat(gt_classes, dim=0)
        per_class_pairs = list(zip(aligned_classes, gt_classes))
        per_class_vals = _mmd_triplet_on_clouds(
            pred_full=stitched,
            gt_full=gt_stitched,
            class_pairs=per_class_pairs,
            whole_slice_band_mults=whole_slice_band_mults,
            spatial_band_mults=spatial_band_mults,
            pair_dist_band_mults=pair_dist_band_mults,
            pair_dist_max_samples=pair_dist_max_samples,
        )

        rows.append({
            "sample_index": s,
            "mmd_whole_slice_global": global_vals["mmd_whole_slice"],
            "mmd_pair_dist_global": global_vals["mmd_pair_dist"],
            "mmd_spatial_global": global_vals["mmd_spatial"],
            "mmd_whole_slice_per_class": per_class_vals["mmd_whole_slice"],
            "mmd_pair_dist_per_class": per_class_vals["mmd_pair_dist"],
            "mmd_spatial_per_class": per_class_vals["mmd_spatial"],
        })
    return pd.DataFrame(rows)


_MMD_METRIC_PAIRS: list[tuple[str, str, str]] = [
    ("mmd_whole_slice", "mmd_whole_slice_global", "mmd_whole_slice_per_class"),
    ("mmd_pair_dist", "mmd_pair_dist_global", "mmd_pair_dist_per_class"),
    ("mmd_spatial", "mmd_spatial_global", "mmd_spatial_per_class"),
]


def aggregate_mmd_across_sections(
    section_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average per-sample MMD metrics across sections (one row per section + ALL)."""
    if not section_tables:
        return pd.DataFrame()
    combined = pd.concat(section_tables, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    metric_cols = [
        "mmd_whole_slice_global",
        "mmd_pair_dist_global",
        "mmd_spatial_global",
        "mmd_whole_slice_per_class",
        "mmd_pair_dist_per_class",
        "mmd_spatial_per_class",
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
    out = pd.concat([grouped, pd.DataFrame([overall])], ignore_index=True)
    for col in metric_cols:
        std_col = f"{col}_std"
        if std_col in out.columns:
            out[std_col] = out[std_col].fillna(0.0)
    return out


def plot_mmd_summary(
    summary_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Three-subplot bar chart: whole-slice / pair-dist / spatial MMD.

    Each subplot shows global ICP vs per-class ICP (two bars per section).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No MMD data to plot.")
        return

    save_path = Path(save_path)
    plot_df = summary_df[summary_df["cell_section"] != "ALL"].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()
    labels = plot_df["cell_section"].tolist()
    x = np.arange(len(labels))
    w = 0.36

    titles = (
        "Whole-slice MMD\n(isotropic IMQ, multi-bandwidth)",
        "Pair-distance MMD\n(distance-distribution IMQ, multi-bandwidth)",
        "Spatial MMD\n(per-class isotropic IMQ, multi-bandwidth)",
    )
    pairs = _MMD_METRIC_PAIRS

    fig, axes = plt.subplots(1, 3, figsize=(max(12.0, 1.1 * len(labels) * 3), 5.2))
    for ax, title, (_, global_key, per_class_key) in zip(axes, titles, pairs):
        g_means = plot_df[f"{global_key}_mean"].to_numpy(dtype=float)
        g_stds = plot_df[f"{global_key}_std"].to_numpy(dtype=float)
        p_means = plot_df[f"{per_class_key}_mean"].to_numpy(dtype=float)
        p_stds = plot_df[f"{per_class_key}_std"].to_numpy(dtype=float)
        ax.bar(
            x - w / 2, g_means, w, yerr=g_stds, capsize=3,
            color="tab:blue", alpha=0.85, label="Global ICP",
        )
        ax.bar(
            x + w / 2, p_means, w, yerr=p_stds, capsize=3,
            color="tab:orange", alpha=0.85, label="Per-class ICP",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("MMD² (mean over samples)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Isotropic IMQ-MMD to GT under two ICP registrations",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved MMD summary plot → {save_path}")


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


def _build_multi_radius_loss(mr_cfg: Any) -> MultiRadiusNeighborhoodLoss:
    """Instantiate the multi-radius neighborhood loss from config (train defaults)."""
    soft_beta = getattr(mr_cfg, "soft_beta", None)
    soft_beta = None if soft_beta is None else float(soft_beta)
    tol_beta = getattr(mr_cfg, "transcriptome_tolerance_soft_beta", None)
    tol_beta = None if tol_beta is None else float(tol_beta)
    tol = getattr(mr_cfg, "transcriptome_tolerance", 0.05)
    if hasattr(tol, "__iter__") and not isinstance(tol, (str, bytes)):
        tol = [float(x) for x in tol]
    else:
        tol = float(tol)
    return MultiRadiusNeighborhoodLoss(
        radii=[float(r) for r in mr_cfg.radii],
        avg_transcriptome_weight=float(
            getattr(mr_cfg, "avg_transcriptome_weight", 1.0)
        ),
        density_weight=float(getattr(mr_cfg, "density_weight", 0.0)),
        global_transcriptome_weight=float(
            getattr(mr_cfg, "global_transcriptome_weight", 1.0)
        ),
        loss_radius_scale=float(getattr(mr_cfg, "loss_radius_scale", 512.0)),
        transcriptome_tolerance=tol,
        transcriptome_tolerance_gate_beta=tol_beta,
        transcriptome_tolerance_warmup_epochs=int(
            getattr(mr_cfg, "transcriptome_tolerance_warmup_epochs", 0)
        ),
        soft_beta=soft_beta,
        eps=float(getattr(mr_cfg, "eps", 1e-6)),
        include_self=bool(getattr(mr_cfg, "include_self", True)),
        cache_gt=False,
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


def _slice_multi_radius_loss(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    node_features: np.ndarray,
    cell_class_int: np.ndarray,
    device: Any,
    *,
    multi_radius_module: MultiRadiusNeighborhoodLoss,
) -> float:
    """Multi-radius neighborhood transcriptome loss for one (pred, GT) pair.

    Uses GT ``node_features`` on both sides (positions differ), mirroring
    ``MultiRadiusNeighborhoodLoss.forward``.
    """
    mask = _valid_cells_mask(cell_class_int)
    if mask.sum() < 2:
        return float("nan")

    feat = np.asarray(node_features, dtype=np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    true_h = _make_minimal_holder(gt_positions, feat, mask, device)
    pred_h = _make_minimal_holder(pred_positions, feat, mask, device)
    with torch.no_grad():
        loss, _ = multi_radius_module(pred_h, true_h, train_stage=False, log=False)
    return float(loss.item())


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
    multi_radius_module: MultiRadiusNeighborhoodLoss,
) -> pd.DataFrame:
    """Per-sample scalar losses (position MSE, directional, multi-radius).

    Returns a long DataFrame with one row per (sample, metric_name).
    ``transcriptome_multi_radius`` is always computed.
    """
    records: list[dict[str, Any]] = []
    S = int(pred_positions.shape[0])

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

        mr_val = _slice_multi_radius_loss(
            pred, gt_positions, node_features, cell_class_int, device,
            multi_radius_module=multi_radius_module,
        )
        records.append({
            "sample_index": s,
            "metric": "transcriptome_multi_radius",
            "value": mr_val,
        })

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
# Grid transcript-count metrics (Procrustes+scale → spatial bins → gene counts)
# Fully batched on GPU over denoising samples (and genes for soft Spearman).
# ─────────────────────────────────────────────────────────────────────────────


def _class_barycenters_batched(
    positions: torch.Tensor,
    labels: torch.Tensor,
    classes: torch.Tensor,
) -> torch.Tensor:
    """Per-sample per-class means.

    ``positions``: ``(S, N, 2)`` or ``(N, 2)``; ``labels``: ``(N,)``;
    ``classes``: ``(C,)``. Returns ``(S, C, 2)`` (``S=1`` if positions is 2-D).
    """
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    s, n, d = positions.shape
    c = int(classes.shape[0])
    # one-hot (N, C) → broadcast over samples
    one_hot = (labels.unsqueeze(1) == classes.unsqueeze(0)).to(positions.dtype)  # (N, C)
    counts = one_hot.sum(dim=0).clamp_min(1.0)  # (C,)
    # (S, N, 2) x (N, C) → (S, C, 2)
    sums = torch.einsum("snd,nc->scd", positions, one_hot)
    return sums / counts.view(1, c, 1)


def _procrustes_similarity_batched(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    with_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched uniform-weight Procrustes (same math as ``procrustes_similarity``).

    ``source``: ``(S, M, 2)``, ``target``: ``(M, 2)`` or ``(S, M, 2)``.
    Returns ``R (S, 2, 2)``, ``t (S, 2)``, ``s (S,)`` in ``source`` dtype.
    """
    X = source.detach().to(torch.float64)
    Y = target.detach().to(torch.float64)
    if Y.ndim == 2:
        Y = Y.unsqueeze(0).expand(X.shape[0], -1, -1)

    mu_x = X.mean(dim=1, keepdim=True)  # (S, 1, 2)
    mu_y = Y.mean(dim=1, keepdim=True)
    Xc, Yc = X - mu_x, Y - mu_y

    # H = Xc^T @ Yc  → (S, 2, 2)
    H = torch.matmul(Xc.transpose(1, 2), Yc)
    U, _, Vh = torch.linalg.svd(H)
    R = torch.matmul(Vh.transpose(1, 2), U.transpose(1, 2))
    # Reflection fix per sample
    det = torch.det(R)
    flip = det < 0
    if bool(flip.any()):
        Vh = Vh.clone()
        Vh[flip, -1, :] *= -1
        R = torch.matmul(Vh.transpose(1, 2), U.transpose(1, 2))

    if with_scale:
        num = (Yc * torch.matmul(Xc, R.transpose(1, 2))).sum(dim=(1, 2))
        den = Xc.pow(2).sum(dim=(1, 2)).clamp_min(1e-12)
        s = num / den
    else:
        s = torch.ones(X.shape[0], dtype=torch.float64, device=X.device)

    t = mu_y.squeeze(1) - s.unsqueeze(-1) * torch.matmul(
        R, mu_x.squeeze(1).unsqueeze(-1)
    ).squeeze(-1)

    out_dtype = source.dtype if source.is_floating_point() else torch.float32
    return R.to(dtype=out_dtype), t.to(dtype=out_dtype), s.to(dtype=out_dtype)


def align_samples_to_gt_procrustes_torch(
    samples: torch.Tensor,
    gt: torch.Tensor,
    labels: torch.Tensor,
    *,
    with_scale: bool = True,
) -> torch.Tensor:
    """Batched similarity-align every sample onto GT (``train_mmds`` style).

    ``samples``: ``(S, N, 2)``, ``gt``: ``(N, 2)``, ``labels``: ``(N,)``.
    Fits one Procrustes ``(R, t, s)`` per sample on per-class barycenters and
    applies it with a single batched matmul. Returns ``(S, N, 2)``.
    """
    samples = samples.to(dtype=torch.float32)
    gt = gt.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"samples must be (S, N, 2); got {tuple(samples.shape)}")
    if gt.shape != samples.shape[1:]:
        raise ValueError(
            f"gt shape {tuple(gt.shape)} must match samples[1:] {tuple(samples.shape[1:])}"
        )

    classes = torch.unique(labels)
    # Drop empty (shouldn't happen); need >= 2 classes for a rotation.
    usable = []
    for cval in classes:
        if int((labels == cval).sum().item()) >= 1:
            usable.append(int(cval.item()))
    if len(usable) < 2:
        return samples

    classes_t = torch.tensor(usable, device=samples.device, dtype=labels.dtype)
    P_bc = _class_barycenters_batched(samples, labels, classes_t)  # (S, C, 2)
    G_bc = _class_barycenters_batched(gt, labels, classes_t).squeeze(0)  # (C, 2)
    R, t, s = _procrustes_similarity_batched(P_bc, G_bc, with_scale=with_scale)
    # aligned = s * (pts @ R.T) + t   → (S, N, 2)
    return s.view(-1, 1, 1) * torch.matmul(samples, R.transpose(1, 2)) + t.unsqueeze(1)


def align_pred_to_gt_procrustes(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    cell_class_int: np.ndarray,
    *,
    with_scale: bool = True,
    device: Any | None = None,
) -> np.ndarray:
    """Single-sample wrapper around :func:`align_samples_to_gt_procrustes_torch`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.as_tensor(
        np.asarray(pred_xy, dtype=np.float32), device=device
    ).unsqueeze(0)
    gt = torch.as_tensor(np.asarray(gt_xy, dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(cell_class_int), dtype=torch.long, device=device)
    aligned = align_samples_to_gt_procrustes_torch(
        samples, gt, labels, with_scale=with_scale
    )
    return aligned[0].detach().cpu().numpy().astype(np.float32)


def _gt_square_bbox_torch(
    gt_xy: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Square bbox around GT with fractional ``margin``; returns scalars on device."""
    x0y0 = gt_xy.amin(dim=0)
    x1y1 = gt_xy.amax(dim=0)
    span = (x1y1 - x0y0).clamp_min(1e-8)
    side = span.amax()
    center = 0.5 * (x0y0 + x1y1)
    half = 0.5 * side * (1.0 + 2.0 * float(margin))
    lo = center - half
    hi = center + half
    return lo[0], hi[0], lo[1], hi[1]


def _grid_bin_indices_torch(
    xy: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Map coords to flat grid indices.

    ``xy``: ``(N, 2)`` or ``(S, N, 2)`` → same leading shape ``(...,)`` of
    long indices in ``[0, grid_size²)``.
    """
    n = int(grid_size)
    span_x = (x1 - x0).clamp_min(1e-8)
    span_y = (y1 - y0).clamp_min(1e-8)
    ix = torch.floor((xy[..., 0] - x0) / span_x * n).long().clamp(0, n - 1)
    iy = torch.floor((xy[..., 1] - y0) / span_y * n).long().clamp(0, n - 1)
    return iy * n + ix


def _transcript_counts_on_grid_torch(
    xy: torch.Tensor,
    features: torch.Tensor,
    *,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Batched scatter-sum of gene features into grid squares.

    ``xy``: ``(S, N, 2)`` or ``(N, 2)``; ``features``: ``(N, F)``.
    Returns ``(S, grid_size², F)`` (``S=1`` if ``xy`` is 2-D).
    """
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
    s, n_cells, _ = xy.shape
    n_genes = int(features.shape[-1])
    n_bins = int(grid_size) ** 2
    bins = _grid_bin_indices_torch(xy, x0, x1, y0, y1, grid_size)  # (S, N)

    # scatter_add over the bin axis: expand features to (S, N, F)
    feat = features.unsqueeze(0).expand(s, -1, -1).contiguous()
    index = bins.unsqueeze(-1).expand(-1, -1, n_genes)
    counts = xy.new_zeros(s, n_bins, n_genes)
    return counts.scatter_add(1, index, feat)


def _grid_transcript_metrics_batched(
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
    *,
    spearman_tau: float,
    spearman_chunk_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MAE and mean soft-Spearman per sample, fully batched on device.

    ``pred_counts``: ``(S, n_squares, F)``, ``gt_counts``: ``(n_squares, F)``
    or ``(S, n_squares, F)``. Soft Spearman ranks along the square axis for
    each gene, then averages over genes → ``(S,)`` correlation.
    """
    if gt_counts.ndim == 2:
        gt_counts = gt_counts.unsqueeze(0).expand_as(pred_counts)

    # MAE over squares × genes, per sample
    mae = (pred_counts - gt_counts).abs().mean(dim=(1, 2))  # (S,)

    # (S, n_squares, F) → (S, F, n_squares) for per-gene spatial Spearman
    pred_g = pred_counts.transpose(1, 2)
    gt_g = gt_counts.transpose(1, 2)
    if pred_g.shape[-1] < 2:
        return mae, torch.full_like(mae, float("nan"))

    dist = soft_spearman_distance(
        pred_g,
        gt_g,
        float(spearman_tau),
        eps=float(eps),
        chunk_size=int(spearman_chunk_size),
    )  # (S, F)
    corr = (1.0 - dist).nanmean(dim=-1)  # (S,)
    return mae, corr


def compute_section_grid_transcript_metrics(
    preds: np.ndarray,
    batch: "SliceBatch",
    grid_cfg: Any,
    device: Any,
) -> pd.DataFrame:
    """Per-sample grid transcript MAE + soft Spearman after Procrustes+scale.

    Fully vectorised on ``device``:
      * one batched Procrustes+scale for all ``S`` samples,
      * one ``scatter_add`` binning pass per grid size for all samples,
      * one soft-Spearman pass over ``(S, F, n_squares)`` (chunked over genes).

    Returns one row per sample with ``grid_transcript_diff`` /
    ``grid_transcript_soft_spearman`` (averaged over configured grid sizes)
    plus per-resolution columns ``grid_transcript_diff_{n}`` /
    ``grid_transcript_soft_spearman_{n}``.
    """
    grid_sizes = [int(g) for g in grid_cfg.grid_sizes]
    margin = float(getattr(grid_cfg, "margin", 0.05))
    spearman_tau = float(getattr(grid_cfg, "spearman_tau", 4.0))
    spearman_chunk_size = int(getattr(grid_cfg, "spearman_chunk_size", 512))
    with_scale = bool(getattr(grid_cfg, "procrustes_with_scale", True))
    eps = float(getattr(grid_cfg, "eps", 1e-6))

    samples = torch.as_tensor(
        np.asarray(preds, dtype=np.float32), dtype=torch.float32, device=device
    )
    gt = torch.as_tensor(
        np.asarray(batch.gt_positions, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.as_tensor(
        np.asarray(batch.cell_class_int), dtype=torch.long, device=device
    )
    features = torch.as_tensor(
        np.asarray(batch.holder.node_features, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"node_features must be (N, F); got {tuple(features.shape)}")

    s = int(samples.shape[0])
    x0, x1, y0, y1 = _gt_square_bbox_torch(gt, margin)

    with torch.no_grad():
        aligned = align_samples_to_gt_procrustes_torch(
            samples, gt, labels, with_scale=with_scale
        )  # (S, N, 2)

        per_size_mae: dict[int, torch.Tensor] = {}
        per_size_corr: dict[int, torch.Tensor] = {}
        for n in grid_sizes:
            gt_counts = _transcript_counts_on_grid_torch(
                gt, features, x0=x0, x1=x1, y0=y0, y1=y1, grid_size=n
            )[0]  # (n_bins, F)
            pred_counts = _transcript_counts_on_grid_torch(
                aligned, features, x0=x0, x1=x1, y0=y0, y1=y1, grid_size=n
            )  # (S, n_bins, F)
            mae_s, corr_s = _grid_transcript_metrics_batched(
                pred_counts,
                gt_counts,
                spearman_tau=spearman_tau,
                spearman_chunk_size=spearman_chunk_size,
                eps=eps,
            )
            per_size_mae[n] = mae_s
            per_size_corr[n] = corr_s

        # Stack resolutions → (S, R) then mean over R
        mae_stack = torch.stack([per_size_mae[n] for n in grid_sizes], dim=1)
        corr_stack = torch.stack([per_size_corr[n] for n in grid_sizes], dim=1)
        mae_mean = mae_stack.nanmean(dim=1)
        corr_mean = corr_stack.nanmean(dim=1)

        mae_np = {n: per_size_mae[n].detach().cpu().numpy() for n in grid_sizes}
        corr_np = {n: per_size_corr[n].detach().cpu().numpy() for n in grid_sizes}
        mae_mean_np = mae_mean.detach().cpu().numpy()
        corr_mean_np = corr_mean.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for i in range(s):
        row: dict[str, Any] = {
            "sample_index": i,
            "grid_transcript_diff": float(mae_mean_np[i]),
            "grid_transcript_soft_spearman": float(corr_mean_np[i]),
        }
        for n in grid_sizes:
            row[f"grid_transcript_diff_{n}"] = float(mae_np[n][i])
            row[f"grid_transcript_soft_spearman_{n}"] = float(corr_np[n][i])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_grid_transcript_across_sections(
    section_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average per-sample grid transcript metrics across sections."""
    if not section_tables:
        return pd.DataFrame()
    combined = pd.concat(section_tables, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in combined.columns
        if c.startswith("grid_transcript_")
    ]
    if not metric_cols:
        return pd.DataFrame()

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
    out = pd.concat([grouped, pd.DataFrame([overall])], ignore_index=True)
    for col in metric_cols:
        std_col = f"{col}_std"
        if std_col in out.columns:
            out[std_col] = out[std_col].fillna(0.0)
    return out


def plot_grid_transcript_summary(
    summary_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Single subplot: count MAE and soft Spearman side-by-side (twin y-axes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No grid-transcript data to plot.")
        return

    save_path = Path(save_path)
    plot_df = summary_df.copy()
    labels = plot_df["cell_section"].tolist()
    x = np.arange(len(labels))
    w = 0.36

    diff_means = plot_df["grid_transcript_diff_mean"].to_numpy(dtype=float)
    diff_stds = plot_df["grid_transcript_diff_std"].to_numpy(dtype=float)
    corr_means = plot_df["grid_transcript_soft_spearman_mean"].to_numpy(dtype=float)
    corr_stds = plot_df["grid_transcript_soft_spearman_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.95 * len(labels)), 5.2))
    ax_r = ax.twinx()

    bars_mae = ax.bar(
        x - w / 2, diff_means, w, yerr=diff_stds, capsize=3,
        color="tab:brown", alpha=0.85, label="Count MAE",
    )
    bars_spr = ax_r.bar(
        x + w / 2, corr_means, w, yerr=corr_stds, capsize=3,
        color="tab:olive", alpha=0.85, label="Soft Spearman",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean |Δ transcript counts| (MAE)", color="tab:brown")
    ax_r.set_ylabel("Soft Spearman correlation", color="tab:olive")
    ax.tick_params(axis="y", labelcolor="tab:brown")
    ax_r.tick_params(axis="y", labelcolor="tab:olive")
    ax.set_title(
        "Spatial transcript grid metrics (after Procrustes+scale to GT)\n"
        "MAE + soft Spearman averaged over squares, genes, grid sizes"
    )
    ax.legend(
        [bars_mae, bars_spr],
        ["Count MAE", "Soft Spearman"],
        loc="best",
        fontsize=8,
        framealpha=0.85,
    )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved grid-transcript summary plot → {save_path}")


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
    mmd_cfg = pipe_cfg.mmd_analysis
    ch_voronoi_cfg = pipe_cfg.ch_voronoi_analysis
    scalar_cfg = pipe_cfg.scalar_losses_analysis
    grid_tx_cfg = pipe_cfg.grid_transcript_analysis
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
    directional_base_seed = 0
    if compute_directional:
        directional_module = _build_directional_loss(scalar_cfg.directional)
        directional_base_seed = int(getattr(scalar_cfg.directional, "base_seed", 0))
    multi_radius_module = _build_multi_radius_loss(pipe_cfg.multi_radius_analysis)
    print("=" * 78)
    print(f"[testing_pipeline] Checkpoint : {checkpoint_path}")
    print(f"[testing_pipeline] Output dir  : {output_dir}")
    print(f"[testing_pipeline] num_samples : {num_samples}")
    print(f"[testing_pipeline] batch_size  : {batch_size}")
    print(f"[testing_pipeline] device      : {device}")
    print(f"[testing_pipeline] position_mse: {compute_position_mse}")
    print(f"[testing_pipeline] directional : {compute_directional}")
    print(f"[testing_pipeline] multi_radius: always on")
    print(
        f"[testing_pipeline] grid_tx   : sizes={list(grid_tx_cfg.grid_sizes)} "
        f"procrustes_scale={bool(grid_tx_cfg.procrustes_with_scale)}"
    )
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

    print(
        "[testing_pipeline] Phase 2/2: analysis "
        "(MMD + CH/Voronoi + spread + grid transcripts)…"
    )
    section_mmd_tables: list[pd.DataFrame] = []
    section_ch_voronoi_tables: list[pd.DataFrame] = []
    section_spread_rows: list[dict] = []
    section_spread_per_cell_tables: list[pd.DataFrame] = []
    section_scalar_losses_tables: list[pd.DataFrame] = []
    section_grid_tx_tables: list[pd.DataFrame] = []

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
                f"[testing_pipeline] Skipping MMD for {cell_section!r}: "
                f"no cell class with >= {min_cells} cells."
            )
        else:
            section_mmd = compute_section_mmd_metrics(
                preds=preds,
                batch=batch,
                cell_classes=cell_classes,
                mmd_cfg=mmd_cfg,
                device=device,
            )
            if not section_mmd.empty:
                section_mmd.insert(0, "cell_section", str(cell_section))
                section_mmd.to_csv(
                    section_dir / "mmd_per_sample.csv", index=False
                )
                section_mmd_tables.append(section_mmd)

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
            multi_radius_module=multi_radius_module,
        )
        if not section_scalar.empty:
            section_scalar.insert(0, "cell_section", str(cell_section))
            section_scalar.to_csv(
                section_dir / "scalar_losses_per_sample.csv", index=False
            )
            section_scalar_losses_tables.append(section_scalar)

        section_grid_tx = compute_section_grid_transcript_metrics(
            preds=preds,
            batch=batch,
            grid_cfg=grid_tx_cfg,
            device=device,
        )
        if not section_grid_tx.empty:
            section_grid_tx.insert(0, "cell_section", str(cell_section))
            section_grid_tx.to_csv(
                section_dir / "grid_transcript_per_sample.csv", index=False
            )
            section_grid_tx_tables.append(section_grid_tx)

    mmd_summary = aggregate_mmd_across_sections(section_mmd_tables)
    mmd_csv = output_dir / "mmd_summary.csv"
    mmd_plot = output_dir / "mmd_summary.png"
    if not mmd_summary.empty:
        mmd_summary.to_csv(mmd_csv, index=False)
        plot_mmd_summary(mmd_summary, mmd_plot)
        all_row = mmd_summary[mmd_summary["cell_section"] == "ALL"]
        if not all_row.empty:
            for label, key in (
                ("whole_slice/global", "mmd_whole_slice_global_mean"),
                ("whole_slice/per_class", "mmd_whole_slice_per_class_mean"),
                ("pair_dist/global", "mmd_pair_dist_global_mean"),
                ("pair_dist/per_class", "mmd_pair_dist_per_class_mean"),
                ("spatial/global", "mmd_spatial_global_mean"),
                ("spatial/per_class", "mmd_spatial_per_class_mean"),
            ):
                print(
                    f"[testing_pipeline] MMD {label}: "
                    f"{float(all_row[key].iloc[0]):.6g}"
                )
    else:
        print("[testing_pipeline] No MMD summaries produced.")

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

    grid_tx_summary = aggregate_grid_transcript_across_sections(section_grid_tx_tables)
    grid_tx_csv = output_dir / "grid_transcript_summary.csv"
    grid_tx_plot = output_dir / "grid_transcript_summary.png"
    if not grid_tx_summary.empty:
        grid_tx_summary.to_csv(grid_tx_csv, index=False)
        plot_grid_transcript_summary(grid_tx_summary, grid_tx_plot)
        all_row = grid_tx_summary[grid_tx_summary["cell_section"] == "ALL"]
        if not all_row.empty:
            print(
                f"[testing_pipeline] grid_transcript_diff "
                f"(avg over samples & sections): "
                f"{float(all_row['grid_transcript_diff_mean'].iloc[0]):.6g}"
            )
            print(
                f"[testing_pipeline] grid_transcript_soft_spearman "
                f"(avg over samples & sections): "
                f"{float(all_row['grid_transcript_soft_spearman_mean'].iloc[0]):.6g}"
            )
    else:
        print("[testing_pipeline] No grid-transcript summaries produced.")

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
    return _checkpoint_output_id(checkpoint_path)


def _load_checkpoint_summaries(output_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Load the per-checkpoint summary CSVs produced by ``run_checkpoint_pipeline``."""
    summaries = {
        "mmd": output_dir / "mmd_summary.csv",
        "ch_voronoi": output_dir / "ch_voronoi_summary.csv",
        "spread": output_dir / "cross_sample_spread_summary.csv",
        "scalar_losses": output_dir / "scalar_losses_summary.csv",
        "grid_transcript": output_dir / "grid_transcript_summary.csv",
    }
    return {
        key: (pd.read_csv(path) if path.exists() else None)
        for key, path in summaries.items()
    }


def _collect_checkpoint_comparison(
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
    *,
    include_directional: bool = True,
    include_position_mse: bool = True,
) -> pd.DataFrame:
    """Collapse every metric to one scalar per checkpoint into a long table.

    Returns a DataFrame with columns ``checkpoint``, ``metric``, ``value`` where
    each row is a single benchmark number for one checkpoint (no cell-type / no
    per-section split). Metrics (in fixed display order):

      * ``position_mse`` (if enabled)
      * ``directional_length`` / ``directional_pairwise`` (if enabled)
      * ``transcriptome_multi_radius`` (always)
      * ``ch_energy_curve_loss``
      * ``voronoi_phase_pair_ch_energy_loss``
      * ``mmd_whole_slice_global`` / ``mmd_whole_slice_per_class``
      * ``mmd_pair_dist_global`` / ``mmd_pair_dist_per_class``
      * ``mmd_spatial_global`` / ``mmd_spatial_per_class``
      * ``cross_sample_spread``
      * ``grid_transcript_diff`` / ``grid_transcript_soft_spearman``
    """
    order = []
    if include_position_mse:
        order.append("position_mse")
    if include_directional:
        order.extend(["directional_length", "directional_pairwise"])
    order.append("transcriptome_multi_radius")
    order.extend([
        "ch_energy_curve_loss",
        "voronoi_phase_pair_ch_energy_loss",
        "mmd_whole_slice_global",
        "mmd_whole_slice_per_class",
        "mmd_pair_dist_global",
        "mmd_pair_dist_per_class",
        "mmd_spatial_global",
        "mmd_spatial_per_class",
        "cross_sample_spread",
        "grid_transcript_diff",
        "grid_transcript_soft_spearman",
    ])

    allowed_scalar = {"transcriptome_multi_radius"}
    if include_position_mse:
        allowed_scalar.add("position_mse")
    if include_directional:
        allowed_scalar.update({"directional_length", "directional_pairwise"})

    rows: list[dict[str, Any]] = []
    for ckpt, out_dir in zip(checkpoint_paths, output_dirs):
        label = _checkpoint_label(ckpt)
        loaded = _load_checkpoint_summaries(out_dir)

        m = loaded.get("mmd")
        if m is not None and not m.empty:
            if "cell_section" in m.columns:
                all_rows = m[m["cell_section"] == "ALL"]
            else:
                all_rows = m
            if all_rows.empty:
                all_rows = m
            for metric in (
                "mmd_whole_slice_global",
                "mmd_whole_slice_per_class",
                "mmd_pair_dist_global",
                "mmd_pair_dist_per_class",
                "mmd_spatial_global",
                "mmd_spatial_per_class",
            ):
                col = f"{metric}_mean"
                if col in all_rows.columns:
                    rows.append({
                        "checkpoint": label,
                        "metric": metric,
                        "value": float(all_rows[col].mean()),
                    })

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
                metric_name = str(metric)
                if metric_name not in allowed_scalar:
                    continue
                rows.append({"checkpoint": label, "metric": metric_name,
                             "value": float(sub["mean"].mean())})

        gt_tx = loaded.get("grid_transcript")
        if gt_tx is not None and not gt_tx.empty:
            if "cell_section" in gt_tx.columns:
                all_rows = gt_tx[gt_tx["cell_section"] == "ALL"]
            else:
                all_rows = gt_tx
            if all_rows.empty:
                all_rows = gt_tx
            for metric in ("grid_transcript_diff", "grid_transcript_soft_spearman"):
                col = f"{metric}_mean"
                if col in all_rows.columns:
                    rows.append({
                        "checkpoint": label,
                        "metric": metric,
                        "value": float(all_rows[col].mean()),
                    })

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
    checkpoints; kept for convenience in the CSV, not used by the PNG plot).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        with save_path.open("w") as fh:
            fh.write("# checkpoint comparison\n# (no data)\n")
        print(f"[testing_pipeline] Saved checkpoint comparison CSV → {save_path}")
        return

    df = comparison.copy()
    df["normalized_score"] = np.nan
    # Soft Spearman correlation is higher-is-better; everything else is lower-is-better.
    higher_is_better = {"grid_transcript_soft_spearman"}
    for metric, sub in df.groupby("metric"):
        vals = sub["value"].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = hi - lo
        if not np.isfinite(denom) or denom <= 0:
            df.loc[sub.index, "normalized_score"] = 1.0
        elif str(metric) in higher_is_better:
            df.loc[sub.index, "normalized_score"] = (vals - lo) / denom
        else:
            df.loc[sub.index, "normalized_score"] = (hi - vals) / denom
    df.to_csv(save_path, index=False)
    print(f"[testing_pipeline] Saved checkpoint comparison CSV → {save_path}")


def _comparison_panels(
    *,
    include_directional: bool = True,
    include_position_mse: bool = True,
) -> list[tuple[str, list[str]]]:
    """Build checkpoint-comparison subplot specs from enabled scalar flags."""
    panels: list[tuple[str, list[str]]] = []
    if include_position_mse:
        panels.append(("Position MSE", ["position_mse"]))
    if include_directional:
        panels.append(("Directional length", ["directional_length"]))
        panels.append(("Directional pairwise", ["directional_pairwise"]))
    panels.append(("Transcriptome multi-radius", ["transcriptome_multi_radius"]))
    panels.extend([
        (
            "MMD whole-slice",
            ["mmd_whole_slice_global", "mmd_whole_slice_per_class"],
        ),
        (
            "MMD pair-dist",
            ["mmd_pair_dist_global", "mmd_pair_dist_per_class"],
        ),
        (
            "MMD spatial",
            ["mmd_spatial_global", "mmd_spatial_per_class"],
        ),
        (
            "Cahn-Hilliard",
            ["ch_energy_curve_loss", "voronoi_phase_pair_ch_energy_loss"],
        ),
        ("Cross-sample spread", ["cross_sample_spread"]),
        (
            "Grid transcript",
            ["grid_transcript_diff", "grid_transcript_soft_spearman"],
        ),
    ])
    return panels


_COMPARISON_METRIC_LABELS: dict[str, str] = {
    "mmd_whole_slice_global": "global ICP",
    "mmd_whole_slice_per_class": "per-class ICP",
    "mmd_pair_dist_global": "global ICP",
    "mmd_pair_dist_per_class": "per-class ICP",
    "mmd_spatial_global": "global ICP",
    "mmd_spatial_per_class": "per-class ICP",
    "ch_energy_curve_loss": "CH energy",
    "voronoi_phase_pair_ch_energy_loss": "Voronoi phase",
    "grid_transcript_diff": "count MAE",
    "grid_transcript_soft_spearman": "soft Spearman",
}


def plot_checkpoint_comparison(
    comparison: pd.DataFrame,
    save_path: Path,
    *,
    include_directional: bool = True,
    include_position_mse: bool = True,
) -> None:
    """Bar plot comparing metrics across checkpoints (raw scale).

    Subplot count follows the enabled scalar flags: position MSE and/or both
    directional metrics, always-on transcriptome multi-radius, then three MMD
    panels, Cahn-Hilliard, cross-sample spread, and grid transcript metrics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#dddddd",
        "figure.facecolor": "#f8f9fb",
        "axes.facecolor": "#ffffff",
    })

    save_path = Path(save_path)
    if comparison.empty:
        print("[testing_pipeline] No checkpoint comparison data to plot.")
        return

    panels = _comparison_panels(
        include_directional=include_directional,
        include_position_mse=include_position_mse,
    )
    if not panels:
        print("[testing_pipeline] No checkpoint comparison panels to plot.")
        return

    checkpoints = list(dict.fromkeys(comparison["checkpoint"].tolist()))
    palette = sns.color_palette("deep", n_colors=max(len(checkpoints), 3))
    ckpt_colors = {ckpt: palette[i % len(palette)] for i, ckpt in enumerate(checkpoints)}

    raw_vals: dict[tuple[str, str], float] = {}
    for _, row in comparison.iterrows():
        raw_vals[(str(row["checkpoint"]), str(row["metric"]))] = float(row["value"])

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(12.0, 3.2 * len(panels)), 6.2),
        facecolor=plt.rcParams["figure.facecolor"],
    )
    if len(panels) == 1:
        axes = [axes]

    metric_alphas = (0.95, 0.55)

    def _style_axis(ax: plt.Axes, *, show_ylabel: bool) -> None:
        sns.despine(ax=ax, left=False, bottom=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="-", alpha=0.7)
        ax.xaxis.grid(False)
        ax.tick_params(axis="both", labelsize=8, colors="#444444")
        if show_ylabel:
            ax.set_ylabel("Value", fontsize=9, color="#555555")
        else:
            ax.set_ylabel("")

    def _annotate_bars(
        ax: plt.Axes,
        positions: np.ndarray,
        heights: np.ndarray,
        *,
        fontsize: float,
    ) -> float:
        max_height = 0.0
        for pos, height in zip(positions, heights):
            if not np.isfinite(height):
                continue
            max_height = max(max_height, float(height))
            ax.text(
                pos,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#333333",
                fontweight="medium",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
        return max_height

    for ax, (title, metrics) in zip(axes, panels):
        available_metrics = [
            m for m in metrics
            if any((ckpt, m) in raw_vals for ckpt in checkpoints)
        ]
        if not available_metrics:
            ax.set_title(title, fontsize=11, pad=10)
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10, color="#888888",
            )
            ax.set_axis_off()
            continue

        n_metrics = len(available_metrics)
        n_ckpt = len(checkpoints)
        max_height = 0.0

        if n_metrics == 1:
            metric = available_metrics[0]
            x = np.arange(n_ckpt)
            heights = np.array(
                [raw_vals.get((ckpt, metric), np.nan) for ckpt in checkpoints],
                dtype=float,
            )
            positions = x.astype(float)
            ax.bar(
                positions,
                heights,
                width=0.62,
                color=[ckpt_colors[ckpt] for ckpt in checkpoints],
                alpha=0.95,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(checkpoints, rotation=28, ha="right")
            max_height = _annotate_bars(ax, positions, heights, fontsize=7.5)
        else:
            group_width = 0.78
            bar_w = group_width / n_metrics
            x = np.arange(n_ckpt)
            for mi, metric in enumerate(available_metrics):
                offsets = x + (mi - (n_metrics - 1) / 2) * bar_w
                heights = np.array(
                    [raw_vals.get((ckpt, metric), np.nan) for ckpt in checkpoints],
                    dtype=float,
                )
                ax.bar(
                    offsets,
                    heights,
                    bar_w,
                    color=[ckpt_colors[ckpt] for ckpt in checkpoints],
                    alpha=metric_alphas[mi % len(metric_alphas)],
                    edgecolor="white",
                    linewidth=1.0,
                    label=_COMPARISON_METRIC_LABELS.get(metric, metric),
                    zorder=3,
                )
                max_height = max(
                    max_height,
                    _annotate_bars(ax, offsets, heights, fontsize=6.8),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(checkpoints, rotation=28, ha="right")
            ax.legend(
                fontsize=7,
                loc="upper right",
                title="Metric",
                title_fontsize=7,
                frameon=True,
                handlelength=1.2,
                handleheight=0.9,
            )

        ax.set_title(title, fontsize=11, pad=10)
        _style_axis(ax, show_ylabel=(ax is axes[0]))
        if max_height > 0:
            ax.set_ylim(0, max_height * 1.22)

    handles = [
        Patch(facecolor=ckpt_colors[ckpt], edgecolor="white", linewidth=0.8, label=ckpt)
        for ckpt in checkpoints
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(checkpoints), 6),
        bbox_to_anchor=(0.5, 1.03),
        fontsize=9,
        title="Checkpoint",
        title_fontsize=9,
        frameon=True,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.suptitle(
        "Model benchmark — metrics comparison across checkpoints",
        y=1.10,
        fontsize=14,
        fontweight="bold",
        color="#222222",
    )
    fig.subplots_adjust(top=0.80, wspace=0.30, left=0.04, right=0.99, bottom=0.18)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=220, facecolor=fig.get_facecolor())
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

    scalar_cfg = cfg.test.pipeline.scalar_losses_analysis
    include_directional = bool(getattr(scalar_cfg, "compute_directional", True))
    include_position_mse = bool(getattr(scalar_cfg, "compute_position_mse", True))

    combined = _collect_checkpoint_comparison(
        checkpoint_paths,
        output_dirs,
        include_directional=include_directional,
        include_position_mse=include_position_mse,
    )
    _save_combined_comparison_csv(combined, parent_dir / "checkpoint_comparison.csv")
    plot_checkpoint_comparison(
        combined,
        parent_dir / "checkpoint_comparison.png",
        include_directional=include_directional,
        include_position_mse=include_position_mse,
    )
    return parent_dir


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    assert _DIFFUSION_REPO_ROOT == REPO_ROOT
    run_testing_pipeline(cfg)


if __name__ == "__main__":
    main()
