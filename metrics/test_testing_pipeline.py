"""Statistical testing pipeline for LUNA checkpoints.

For each checkpoint listed in ``configs/test/default.yaml`` the pipeline:

1. Runs ``n`` denoising samples (different seeds) on every ``cell_section`` of
   the test split. Sampling is organised **seed-first**: for each seed, all
   sections are inferred in batches of ``test.batch_size`` graphs; analysis
   runs only after every seed/section pair has been collected.
2. Computes **isotropic IMQ-MMD distances** (from ``metrics/train_mmds.py``,
   no anisotropy) between predicted and GT point clouds after the **same
   Procrustes similarity** used by the training MMD loss
   (``test.pipeline.procrustes``): one global ``(R, t, s)`` fit on per-class
   barycenters **plus** PCA1 axis landmarks
   (``n ≈ √n_cells`` points on ``[bc ± L·PCA1]``), then applied to every
   predicted cell. Three scalars are reported (averaged over denoising
   samples, then over sections):

   * ``mmd_whole_slice`` — isotropic IMQ-MMD on the full point cloud
     (bandwidths = median pairwise × ``whole_slice_band_mults``).
   * ``mmd_pair_dist`` — IMQ-MMD on pairwise-distance distributions
     (bandwidths = median pair-dist × ``pair_dist_band_mults``), averaged
     over cell types.
   * ``mmd_spatial`` — isotropic IMQ-MMD on per-class spatial point clouds
     (bandwidths = median pairwise × ``spatial_band_mults``), averaged
     over cell types.

   All three use the **same whole-slice Procrustes** (no per-class ICP /
   per-class registration onto GT).

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

4. Computes a **cross-sample positional spread** on the same shared
   train-MMD Procrustes-aligned predictions used by the other GT metrics:
   for each cell, the average pairwise Euclidean distance between that
   cell's positions across the denoising samples, then averaged over cells
   and sections. The pairwise-distance computation is vectorised on GPU
   (broadcast ``(S, S, N)`` distance tensor).

5. Computes **soft grid-intersection transcript metrics** on the same
   train-MMD Procrustes-aligned predictions used for MMD / CH / scalar
   losses. For each configured lattice size ``n`` (``8``, ``16``, …) an
   ``n×n`` square tiling of the GT bbox defines ``(n+1)²`` grid-line
   **intersections**. At each intersection a soft circular spatial weight
   is built with the same sigmoid membership used by the multi-radius
   transcriptome loss:

       ``w_j = sigmoid(soft_beta · (½·spacing − ‖x_j − c‖))``

   where ``spacing`` is the grid step (side / ``n``), the soft radius is
   half that spacing, and ``c`` is the intersection. Per intersection the
   **weighted sum** of cell
   transcriptomes is compared between Procrustes-aligned pred positions
   and GT positions (features are shared). Soft Spearman is evaluated on
   the full ``(S, F, n_intersections)`` tensor (chunked over genes).
   Two scalars are reported per sample (then averaged over samples /
   sections / grid sizes):

   * ``grid_transcript_diff`` — mean absolute difference (MAE) of the
     soft weighted transcriptome sums across intersections and genes.
   * ``grid_transcript_soft_spearman`` — soft Spearman correlation of the
     per-intersection weighted-sum vectors between pred and GT, computed
     **per gene** (spatial pattern match), then averaged over genes.
     Soft ranks reuse ``soft_spearman_distance`` from
     ``train_spatial_transcriptomics`` (reported as ``1 - distance``).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

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
    apply_similarity,
    class_anisotropy,
    class_axis_landmarks,
    class_barycenters,
    class_covariances,
    class_top_eigvec,
    mmd2_imq_iso,
    pair_dist_mmd_loss,
    pair_dist_mmd_sigmas,
    precompute_pair_dist_mmd_gt,
    precompute_whole_slice_mmd_cache,
    procrustes_similarity,
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


def resolve_inference_devices(cfg: DictConfig, pipe_device: str) -> list[Any]:
    """Devices used for Phase-1 sampling (analysis stays on the first one).

    Uses ``distribute.gpus_per_node`` (same as train) capped by visible CUDA
    devices when ``pipeline.device`` is CUDA. CPU / single-GPU configs return
    one device.
    """
    resolved = _resolve_device(str(pipe_device))
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return [resolved]

    requested = 1
    distribute = getattr(cfg, "distribute", None)
    if distribute is not None:
        requested = max(1, int(getattr(distribute, "gpus_per_node", 1)))
    n_gpus = min(requested, int(torch.cuda.device_count()))
    return [torch.device(f"cuda:{i}") for i in range(n_gpus)]


def _set_sampling_seed(seed: int, device: Any) -> None:
    """Seed Python / NumPy / Torch RNGs for one sampling draw.

    Uses per-device ``cuda.manual_seed`` (not ``manual_seed_all``) so concurrent
    multi-GPU inference threads do not overwrite each other's CUDA RNG state.
    """
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.manual_seed(int(seed))


def run_all_sampling(
    model: Any,
    section_batches: dict[str, SliceBatch],
    cell_sections: list[str],
    batch_size: int,
    device: Any,
    seeds: Sequence[int],
    progress_desc: str = "Seeds",
    progress_position: int = 0,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Sample every section for each seed, batching sections within a seed.

    Returns
    -------
    dict mapping ``cell_section`` -> ``(len(seeds), N, 2)`` predicted positions,
    stacked in the same order as ``seeds``.
    """
    section_pred_lists: dict[str, list[np.ndarray]] = {
        cell_section: [] for cell_section in cell_sections
    }
    section_chunks = [
        cell_sections[chunk_start : chunk_start + batch_size]
        for chunk_start in range(0, len(cell_sections), batch_size)
    ]

    for seed in tqdm(
        seeds,
        desc=progress_desc,
        unit="seed",
        position=progress_position,
        leave=True,
        disable=not show_progress,
        mininterval=0.5,
    ):
        for chunk in section_chunks:
            slice_batch_list = [section_batches[cell_section] for cell_section in chunk]
            holder = build_batched_holder(slice_batch_list, device)
            _set_sampling_seed(int(seed), device)
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


def _merge_seed_sharded_preds(
    cell_sections: list[str],
    all_seeds: Sequence[int],
    shards: list[tuple[Sequence[int], dict[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    """Reassemble per-GPU seed shards into global ``(num_samples, N, 2)`` stacks."""
    seed_to_local: dict[int, tuple[int, int]] = {}
    for shard_idx, (shard_seeds, _) in enumerate(shards):
        for local_idx, seed in enumerate(shard_seeds):
            seed_to_local[int(seed)] = (shard_idx, local_idx)

    merged: dict[str, np.ndarray] = {}
    for cell_section in cell_sections:
        rows = []
        for seed in all_seeds:
            shard_idx, local_idx = seed_to_local[int(seed)]
            rows.append(shards[shard_idx][1][cell_section][local_idx])
        merged[cell_section] = np.stack(rows, axis=0)
    return merged


def run_sampling_phase(
    cfg: DictConfig,
    checkpoint_path: Path,
    dataset_infos: Any,
    section_batches: dict[str, SliceBatch],
    cell_sections: list[str],
    num_samples: int,
    seed_start: int,
    batch_size: int,
    devices: Sequence[Any],
) -> dict[str, np.ndarray]:
    """Phase-1 inference: shard seeds across GPUs when more than one is available.

    Each device loads its own model copy, runs a disjoint seed subset, then
    results are merged in seed order. Analysis remains single-device afterward.
    """
    all_seeds = list(range(seed_start, seed_start + num_samples))
    device_list = list(devices)
    if not device_list:
        raise ValueError("No inference devices resolved.")

    if len(device_list) == 1 or num_samples <= 1:
        device = device_list[0]
        print(f"[testing_pipeline] Sampling on single device: {device}")
        model = load_model(cfg, dataset_infos, str(checkpoint_path), device)
        try:
            return run_all_sampling(
                model=model,
                section_batches=section_batches,
                cell_sections=cell_sections,
                batch_size=batch_size,
                device=device,
                seeds=all_seeds,
            )
        finally:
            del model
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    n_workers = min(len(device_list), num_samples)
    device_list = device_list[:n_workers]
    shards = [all_seeds[i::n_workers] for i in range(n_workers)]
    print(
        f"[testing_pipeline] Sampling on {n_workers} GPUs "
        f"(seed-sharded; devices={[str(d) for d in device_list]})"
    )

    def _worker(worker_idx: int) -> tuple[Sequence[int], dict[str, np.ndarray]]:
        device = device_list[worker_idx]
        seeds = shards[worker_idx]
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = load_model(cfg, dataset_infos, str(checkpoint_path), device)
        try:
            preds = run_all_sampling(
                model=model,
                section_batches=section_batches,
                cell_sections=cell_sections,
                batch_size=batch_size,
                device=device,
                seeds=seeds,
                progress_desc=f"Seeds[{device}]",
                progress_position=worker_idx,
                show_progress=True,
            )
            return seeds, preds
        finally:
            del model
            try:
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    shard_results: list[tuple[Sequence[int], dict[str, np.ndarray]] | None] = [
        None
    ] * n_workers
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker, worker_idx): worker_idx
            for worker_idx in range(n_workers)
        }
        for fut in as_completed(futures):
            worker_idx = futures[fut]
            shard_results[worker_idx] = fut.result()

    assert all(r is not None for r in shard_results)
    return _merge_seed_sharded_preds(
        cell_sections, all_seeds, shard_results  # type: ignore[arg-type]
    )


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
    device: Any | None = None,
) -> tuple[float, np.ndarray]:
    """Mean pairwise position difference across samples, per cell.

    ``preds`` must already be aligned with the shared train-MMD Procrustes
    (barycenters + PCA1 landmarks → GT), same registration as MMD / CH /
    grid transcript / scalar losses. For each cell, averages Euclidean
    distance between that cell's positions across denoising samples, then
    averages over cells.

    Fully broadcast on GPU: ``||aligned[i] - aligned[j]||`` over the ``S``
    samples for every cell at once (shape ``(S, S, N)``).

    Args:
        preds: ``(S, N, 2)`` Procrustes-aligned predicted positions.
        device: torch device; defaults to CUDA if available else CPU.

    Returns:
        ``(overall_mean, per_cell_spreads)`` where ``per_cell_spreads`` is a
        ``(N,)`` numpy array and ``overall_mean`` is the mean over cells.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.as_tensor(np.asarray(preds), dtype=torch.float32, device=device)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"preds must have shape (S, N, 2); got {samples.shape}")
    s, n, _ = samples.shape
    if s < 2:
        zeros = np.zeros(n, dtype=np.float32)
        return 0.0, zeros

    # Pairwise distances over the sample axis for every cell at once.
    # diff[i, j, c] = samples[i, c] - samples[j, c]  -> (S, S, N, 2)
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)
    pair_dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-20)           # (S, S, N)
    # Exclude the diagonal (self-pairs, distance 0) and average.
    iu = torch.triu_indices(s, s, offset=1, device=device)
    pair_vals = pair_dist[iu[0], iu[1]]                              # (n_pairs, N)
    per_cell = pair_vals.mean(dim=0)                                 # (N,)
    overall = float(per_cell.mean().item())
    return overall, per_cell.detach().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Isotropic IMQ-MMD distances (after shared train-MMD Procrustes)
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


def _mmd_metrics_on_clouds(
    *,
    pred_full: torch.Tensor,
    gt_full: torch.Tensor,
    class_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    whole_slice_band_mults: tuple[float, ...],
    spatial_band_mults: tuple[float, ...],
    pair_dist_band_mults: tuple[float, ...],
    pair_dist_max_samples: int,
) -> dict[str, float]:
    """Whole-slice + per-class spatial/pair-dist MMD on one aligned sample.

    ``class_pairs`` are class subsets of the **same** whole-slice Procrustes
    alignment (not independently ICP-aligned onto each GT class cloud).
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
    """Per-sample isotropic MMDs on already Procrustes-aligned predictions.

    ``preds`` must already be aligned with the shared train-MMD Procrustes
    (barycenters + PCA1 landmarks). Returns one row per sample with
    ``mmd_whole_slice``, ``mmd_pair_dist``, ``mmd_spatial``.
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
    kept = [mask for mask in class_masks if int(mask.sum()) >= 2]
    class_masks = kept
    if not class_masks:
        return pd.DataFrame()

    rows = []
    for s in range(preds.shape[0]):
        aligned = _to_torch_xy(preds[s].astype(np.float32), device)
        class_pairs = [(aligned[mask], gt_full[mask]) for mask in class_masks]
        vals = _mmd_metrics_on_clouds(
            pred_full=aligned,
            gt_full=gt_full,
            class_pairs=class_pairs,
            whole_slice_band_mults=whole_slice_band_mults,
            spatial_band_mults=spatial_band_mults,
            pair_dist_band_mults=pair_dist_band_mults,
            pair_dist_max_samples=pair_dist_max_samples,
        )
        rows.append({
            "sample_index": s,
            "mmd_whole_slice": vals["mmd_whole_slice"],
            "mmd_pair_dist": vals["mmd_pair_dist"],
            "mmd_spatial": vals["mmd_spatial"],
        })
    return pd.DataFrame(rows)


_MMD_METRIC_COLS: list[str] = [
    "mmd_whole_slice",
    "mmd_pair_dist",
    "mmd_spatial",
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

    metric_cols = list(_MMD_METRIC_COLS)
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
    """Three-subplot bar chart: whole-slice / pair-dist / spatial MMD."""
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

    titles = (
        ("mmd_whole_slice", "Whole-slice MMD\n(isotropic IMQ, multi-bandwidth)"),
        ("mmd_pair_dist", "Pair-distance MMD\n(distance-distribution IMQ)"),
        ("mmd_spatial", "Spatial MMD\n(per-class, whole-slice Procrustes)"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(max(12.0, 1.1 * len(labels) * 3), 5.2))
    for ax, (key, title) in zip(axes, titles):
        means = plot_df[f"{key}_mean"].to_numpy(dtype=float)
        stds = plot_df[f"{key}_std"].to_numpy(dtype=float)
        ax.bar(x, means, 0.7, yerr=stds, capsize=3, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("MMD² (mean over samples)")
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        "Isotropic IMQ-MMD to GT after train-MMD Procrustes (barycenters + PCA1)",
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
    """Instantiate the multi-radius neighborhood loss from ``test.pipeline.multi_radius_analysis``."""
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
        avg_spearman_weight=float(getattr(mr_cfg, "avg_spearman_weight", 0.0)),
        global_spearman_weight=float(getattr(mr_cfg, "global_spearman_weight", 0.0)),
        spearman_tau=float(getattr(mr_cfg, "spearman_tau", 4.0)),
        spearman_chunk_size=int(getattr(mr_cfg, "spearman_chunk_size", 512)),
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
# Shared train-MMD Procrustes (barycenters + PCA1 axis landmarks)
# Used for every GT-comparison metric in the testing pipeline.
# ─────────────────────────────────────────────────────────────────────────────


def align_pred_to_gt_mmd_procrustes(
    pred_xy: torch.Tensor,
    true_xy: torch.Tensor,
    labels: torch.Tensor,
    *,
    with_scale: bool = False,
    allow_reflection: bool = True,
    axis_align: bool = True,
    axis_weight: float = 1.0,
    axis_min_cells: int = 64,
    axis_min_anisotropy: float = 0.4,
    axis_length_mult: float = 1.0,
) -> torch.Tensor:
    """One-sample Procrustes matching ``SlideMMDLoss._aligned_pred_xy``.

    Fits a global similarity on per-class barycenters, optionally augmented
    with PCA1 axis landmarks (``n ≈ √n_cells`` points on ``[bc ± L·PCA1]``
    for classes that pass the GT count / anisotropy gates), then applies
    ``s * (X @ R.T) + t`` to every predicted cell.
    """
    mask_b = torch.ones(pred_xy.shape[0], dtype=torch.bool, device=pred_xy.device)
    usable = []
    for cval in torch.unique(labels[mask_b]):
        cint = int(cval.item())
        if int(((labels == cint) & mask_b).sum().item()) < 1:
            continue
        usable.append(cint)
    if len(usable) < 2:
        return pred_xy

    classes_t = torch.tensor(usable, device=pred_xy.device, dtype=labels.dtype)
    P_bc = class_barycenters(pred_xy, labels, mask_b, classes_t)
    G_bc = class_barycenters(true_xy, labels, mask_b, classes_t)

    P_all, G_all, weights = P_bc, G_bc, None
    if axis_align:
        counts = torch.tensor(
            [int(((labels == c) & mask_b).sum().item()) for c in usable],
            device=pred_xy.device,
        )
        axis_gate = counts >= int(axis_min_cells)
        if bool(axis_gate.any()):
            G_cov = class_covariances(true_xy, labels, mask_b, classes_t)
            gt_aniso = class_anisotropy(G_cov)
            axis_gate = axis_gate & (gt_aniso >= float(axis_min_anisotropy))
            if bool(axis_gate.any()):
                P_cov = class_covariances(pred_xy, labels, mask_b, classes_t)
                R0, _, _ = procrustes_similarity(
                    P_bc, G_bc, with_scale=False,
                    allow_reflection=bool(allow_reflection),
                )
                G_axis_dir, _ = class_top_eigvec(G_cov)
                ref_dirs = G_axis_dir @ R0
                P_axis = class_axis_landmarks(
                    P_bc[axis_gate], P_cov[axis_gate], ref_dirs[axis_gate],
                    counts[axis_gate], float(axis_length_mult),
                )
                G_axis = class_axis_landmarks(
                    G_bc[axis_gate], G_cov[axis_gate], G_axis_dir[axis_gate],
                    counts[axis_gate], float(axis_length_mult),
                )
                P_all = torch.cat([P_bc, P_axis], dim=0)
                G_all = torch.cat([G_bc, G_axis], dim=0)
                weights = torch.cat([
                    torch.ones(P_bc.shape[0], device=pred_xy.device),
                    torch.full(
                        (P_axis.shape[0],), float(axis_weight),
                        device=pred_xy.device,
                    ),
                ])

    R, t, s = procrustes_similarity(
        P_all, G_all, with_scale=bool(with_scale),
        weights=weights,
        allow_reflection=bool(allow_reflection),
    )
    return apply_similarity(pred_xy, R, t, s)


def align_samples_to_gt_procrustes_torch(
    samples: torch.Tensor,
    gt: torch.Tensor,
    labels: torch.Tensor,
    *,
    procrustes_cfg: Any | None = None,
    with_scale: bool | None = None,
    allow_reflection: bool | None = None,
    axis_align: bool | None = None,
    axis_weight: float | None = None,
    axis_min_cells: int | None = None,
    axis_min_anisotropy: float | None = None,
    axis_length_mult: float | None = None,
) -> torch.Tensor:
    """Align every sample onto GT with the train-MMD Procrustes (PCA landmarks).

    ``samples``: ``(S, N, 2)``, ``gt``: ``(N, 2)``, ``labels``: ``(N,)``.
    Knobs default from ``procrustes_cfg`` (``test.pipeline.procrustes``),
    matching ``configs/train`` Procrustes keys.
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

    def _cfg(name: str, default, override):
        if override is not None:
            return override
        if procrustes_cfg is not None and hasattr(procrustes_cfg, name):
            return getattr(procrustes_cfg, name)
        return default

    if not bool(_cfg("align", True, None)):
        return samples

    kwargs = dict(
        with_scale=bool(_cfg("with_scale", False, with_scale)),
        allow_reflection=bool(_cfg("allow_reflection", True, allow_reflection)),
        axis_align=bool(_cfg("axis_align", True, axis_align)),
        axis_weight=float(_cfg("axis_weight", 1.0, axis_weight)),
        axis_min_cells=int(_cfg("axis_min_cells", 64, axis_min_cells)),
        axis_min_anisotropy=float(
            _cfg("axis_min_anisotropy", 0.4, axis_min_anisotropy)
        ),
        axis_length_mult=float(_cfg("axis_length_mult", 1.0, axis_length_mult)),
    )

    aligned = [
        align_pred_to_gt_mmd_procrustes(samples[i], gt, labels, **kwargs)
        for i in range(int(samples.shape[0]))
    ]
    return torch.stack(aligned, dim=0)


def align_section_preds_to_gt(
    preds: np.ndarray,
    batch: "SliceBatch",
    procrustes_cfg: Any,
    device: Any,
) -> np.ndarray:
    """Numpy convenience wrapper: ``(S, N, 2)`` → train-MMD Procrustes-aligned."""
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
    with torch.no_grad():
        aligned = align_samples_to_gt_procrustes_torch(
            samples, gt, labels, procrustes_cfg=procrustes_cfg
        )
    return aligned.detach().cpu().numpy().astype(np.float32)


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


def _grid_intersection_centers_torch(
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corners of an ``n×n`` square tiling → ``(n+1)²`` intersections + spacing.

    Returns ``centers (K, 2)`` with ``K = (grid_size + 1)²`` and scalar
    ``spacing = side / grid_size`` (bbox is square by construction).
    """
    n = int(grid_size)
    if n < 1:
        raise ValueError(f"grid_size must be >= 1; got {grid_size}")
    xs = torch.linspace(x0, x1, n + 1, device=x0.device, dtype=x0.dtype)
    ys = torch.linspace(y0, y1, n + 1, device=y0.device, dtype=y0.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    spacing = ((x1 - x0) / float(n)).clamp_min(1e-8)
    return centers, spacing


def _soft_intersection_transcript_sums_torch(
    xy: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    soft_beta: float,
) -> torch.Tensor:
    """Soft circular weighted transcriptome sums at grid intersections.

    At each intersection ``c`` every cell ``j`` gets membership
    ``w_j = sigmoid(soft_beta · (radius − ‖x_j − c‖))`` (same form as the
    multi-radius neighborhood soft ball). Returns the **weighted sum**
    ``Σ_j w_j · features_j`` per intersection.

    ``xy``: ``(S, N, 2)`` or ``(N, 2)``; ``features``: ``(N, F)``;
    ``centers``: ``(K, 2)``. Returns ``(S, K, F)`` (``S=1`` if ``xy`` is 2-D).
    """
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
    s = int(xy.shape[0])
    # (S, K, N) pairwise distances from each intersection to each cell
    centers_b = centers.unsqueeze(0).expand(s, -1, -1)
    dists = torch.cdist(centers_b, xy)
    w = torch.sigmoid(float(soft_beta) * (float(radius) - dists))  # (S, K, N)
    # (S, K, N) @ (N, F) → (S, K, F)
    return torch.matmul(w, features)


def _grid_transcript_metrics_batched(
    pred_sums: torch.Tensor,
    gt_sums: torch.Tensor,
    *,
    spearman_tau: float,
    spearman_chunk_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MAE and mean soft-Spearman per sample, fully batched on device.

    ``pred_sums``: ``(S, K, F)``, ``gt_sums``: ``(K, F)`` or ``(S, K, F)``
    where ``K`` is the number of soft grid intersections. Soft Spearman
    ranks along the intersection axis for each gene, then averages over
    genes → ``(S,)`` correlation.
    """
    if gt_sums.ndim == 2:
        gt_sums = gt_sums.unsqueeze(0).expand_as(pred_sums)

    # MAE over intersections × genes, per sample
    mae = (pred_sums - gt_sums).abs().mean(dim=(1, 2))  # (S,)

    # (S, K, F) → (S, F, K) for per-gene spatial Spearman
    pred_g = pred_sums.transpose(1, 2)
    gt_g = gt_sums.transpose(1, 2)
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
    """Per-sample soft grid-intersection MAE + soft Spearman.

    ``preds`` must already be aligned with the shared train-MMD Procrustes.
    Fully vectorised on ``device``:
      * soft circular weighted transcriptome sums at ``(n+1)²`` intersections
        per lattice size (sigmoid membership, radius = half grid spacing),
      * one soft-Spearman pass over ``(S, F, K)`` (chunked over genes).

    Returns one row per sample with ``grid_transcript_diff`` /
    ``grid_transcript_soft_spearman`` (averaged over configured grid sizes)
    plus per-resolution columns ``grid_transcript_diff_{n}`` /
    ``grid_transcript_soft_spearman_{n}``.
    """
    grid_sizes = [int(g) for g in grid_cfg.grid_sizes]
    margin = float(getattr(grid_cfg, "margin", 0.05))
    soft_beta = float(getattr(grid_cfg, "soft_beta", 128.0))
    spearman_tau = float(getattr(grid_cfg, "spearman_tau", 4.0))
    spearman_chunk_size = int(getattr(grid_cfg, "spearman_chunk_size", 512))
    eps = float(getattr(grid_cfg, "eps", 1e-6))

    aligned = torch.as_tensor(
        np.asarray(preds, dtype=np.float32), dtype=torch.float32, device=device
    )
    gt = torch.as_tensor(
        np.asarray(batch.gt_positions, dtype=np.float32),
        dtype=torch.float32,
        device=device,
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

    s = int(aligned.shape[0])
    x0, x1, y0, y1 = _gt_square_bbox_torch(gt, margin)

    with torch.no_grad():
        per_size_mae: dict[int, torch.Tensor] = {}
        per_size_corr: dict[int, torch.Tensor] = {}
        for n in grid_sizes:
            centers, spacing = _grid_intersection_centers_torch(
                x0, x1, y0, y1, n
            )
            radius = 0.5 * spacing
            gt_sums = _soft_intersection_transcript_sums_torch(
                gt,
                features,
                centers,
                radius=radius,
                soft_beta=soft_beta,
            )[0]  # (K, F)
            pred_sums = _soft_intersection_transcript_sums_torch(
                aligned,
                features,
                centers,
                radius=radius,
                soft_beta=soft_beta,
            )  # (S, K, F)
            mae_s, corr_s = _grid_transcript_metrics_batched(
                pred_sums,
                gt_sums,
                spearman_tau=spearman_tau,
                spearman_chunk_size=spearman_chunk_size,
                eps=eps,
            )
            per_size_mae[n] = mae_s
            per_size_corr[n] = corr_s

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
    """Soft intersection MAE and soft Spearman side-by-side (twin y-axes)."""
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
        color="tab:brown", alpha=0.85, label="Weighted-sum MAE",
    )
    bars_spr = ax_r.bar(
        x + w / 2, corr_means, w, yerr=corr_stds, capsize=3,
        color="tab:olive", alpha=0.85, label="Soft Spearman",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean |Δ soft weighted transcriptome sums|", color="tab:brown")
    ax_r.set_ylabel("Soft Spearman correlation", color="tab:olive")
    ax.tick_params(axis="y", labelcolor="tab:brown")
    ax_r.tick_params(axis="y", labelcolor="tab:olive")
    ax.set_title(
        "Soft grid-intersection transcript metrics "
        "(after train-MMD Procrustes to GT)\n"
        "MAE + soft Spearman over circular spots, genes, grid sizes"
    )
    ax.legend(
        [bars_mae, bars_spr],
        ["Weighted-sum MAE", "Soft Spearman"],
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
    procrustes_cfg = getattr(pipe_cfg, "procrustes", None)
    num_samples = int(pipe_cfg.num_samples)
    seed_start = int(pipe_cfg.seed_start)
    min_cells = int(pipe_cfg.min_cells_per_class)
    batch_size = int(cfg.test.batch_size)

    output_dir = pipeline_output_dir(cfg, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "effective_config.yaml")

    inference_devices = resolve_inference_devices(cfg, str(pipe_cfg.device))
    device = inference_devices[0]  # analysis stays on the first device
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
    print(f"[testing_pipeline] analysis    : {device}")
    print(
        f"[testing_pipeline] inference   : "
        f"{len(inference_devices)} device(s) "
        f"{[str(d) for d in inference_devices]}"
    )
    print(f"[testing_pipeline] position_mse: {compute_position_mse}")
    print(f"[testing_pipeline] directional : {compute_directional}")
    print(f"[testing_pipeline] multi_radius: always on")
    if procrustes_cfg is not None:
        print(
            f"[testing_pipeline] procrustes : align={bool(getattr(procrustes_cfg, 'align', True))} "
            f"scale={bool(getattr(procrustes_cfg, 'with_scale', False))} "
            f"axis={bool(getattr(procrustes_cfg, 'axis_align', True))} "
            f"reflect={bool(getattr(procrustes_cfg, 'allow_reflection', True))}"
        )
    print(
        f"[testing_pipeline] grid_tx   : sizes={list(grid_tx_cfg.grid_sizes)} "
        f"soft_beta={float(getattr(grid_tx_cfg, 'soft_beta', 128.0))}"
    )
    print("=" * 78)

    load_model_config_from_checkpoint(cfg, checkpoint_path)

    cell_sections = list_cell_sections(datamodule.test_dataset)
    print(f"[testing_pipeline] Found {len(cell_sections)} cell sections in test split.")

    section_batches = prepare_section_batches(datamodule, dataset_infos, cell_sections)

    print("[testing_pipeline] Phase 1/2: batched sampling (seed-outer, sections-inner)…")
    all_section_preds = run_sampling_phase(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_infos=dataset_infos,
        section_batches=section_batches,
        cell_sections=cell_sections,
        num_samples=num_samples,
        seed_start=seed_start,
        batch_size=batch_size,
        devices=inference_devices,
    )

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

        # Shared train-MMD Procrustes for every GT-comparison metric.
        aligned_preds = align_section_preds_to_gt(
            preds, batch, procrustes_cfg, device
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
                preds=aligned_preds,
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
            preds=aligned_preds,
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

        # Same shared Procrustes as every other GT-frame metric.
        spread_mean, spread_per_cell = compute_cross_sample_position_spread(
            preds=aligned_preds,
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
            pred_positions=aligned_preds,
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
            preds=aligned_preds,
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
                ("whole_slice", "mmd_whole_slice_mean"),
                ("pair_dist", "mmd_pair_dist_mean"),
                ("spatial", "mmd_spatial_mean"),
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

    try:
        plot_best_predictions_for_models(cfg, checkpoint_paths, output_dirs)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[testing_pipeline] Best-prediction plots failed: "
            f"{type(exc).__name__}: {exc}"
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
# Best-seed prediction scatter plots (selected slices)
# ─────────────────────────────────────────────────────────────────────────────


def _testing_pipeline_parent_dir(
    cfg: DictConfig,
    checkpoint_paths: Sequence[Path],
) -> Path:
    base = cfg.test.save_dir
    if base is None:
        base = str(checkpoint_paths[0].parent)
    return Path(base) / cfg.general.name / "testing_pipeline"


def _loss_weights_from_cfg(best_cfg: Any) -> dict[str, float]:
    raw = getattr(best_cfg, "loss_weights", None)
    if raw is None:
        return {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    return {str(k): float(v) for k, v in dict(raw).items()}


def _load_section_per_sample_metrics(section_dir: Path) -> pd.DataFrame:
    """Merge per-sample metric CSVs for one section into a wide table."""
    frames: list[pd.DataFrame] = []

    mmd_path = section_dir / "mmd_per_sample.csv"
    if mmd_path.exists():
        frames.append(pd.read_csv(mmd_path))

    ch_path = section_dir / "ch_voronoi_losses_per_sample.csv"
    if ch_path.exists():
        frames.append(pd.read_csv(ch_path))

    grid_path = section_dir / "grid_transcript_per_sample.csv"
    if grid_path.exists():
        frames.append(pd.read_csv(grid_path))

    scalar_path = section_dir / "scalar_losses_per_sample.csv"
    if scalar_path.exists():
        long_df = pd.read_csv(scalar_path)
        if not long_df.empty and {"sample_index", "metric", "value"}.issubset(
            long_df.columns
        ):
            wide = (
                long_df.pivot_table(
                    index="sample_index",
                    columns="metric",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
            )
            wide.columns.name = None
            frames.append(wide)

    if not frames:
        return pd.DataFrame()

    out = frames[0].copy()
    for frame in frames[1:]:
        overlap = [c for c in frame.columns if c in out.columns and c != "sample_index"]
        frame = frame.drop(columns=overlap, errors="ignore")
        out = out.merge(frame, on="sample_index", how="outer")
    return out.sort_values("sample_index").reset_index(drop=True)


def _weighted_seed_scores(
    metrics_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Score each sample; lower is better. Higher-is-better metrics are negated."""
    active: list[tuple[str, float]] = []
    for metric, weight in weights.items():
        w = float(weight)
        if abs(w) <= 0.0:
            continue
        if metric not in metrics_df.columns:
            print(
                f"[testing_pipeline] best-prediction weight for {metric!r} "
                "ignored (metric missing in per-sample CSVs)."
            )
            continue
        active.append((metric, w))

    rows: list[dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        sample_index = int(row["sample_index"])
        score = 0.0
        ok = True
        breakdown: dict[str, float] = {}
        for metric, weight in active:
            val = float(row[metric])
            breakdown[metric] = val
            if not np.isfinite(val):
                ok = False
                continue
            if metric in _HIGHER_IS_BETTER_METRICS:
                score += -weight * val
            else:
                score += weight * val
        rows.append({
            "sample_index": sample_index,
            "weighted_score": float(score) if ok else float("inf"),
            "score_valid": bool(ok and bool(active)),
            **{f"metric__{k}": v for k, v in breakdown.items()},
        })
    return pd.DataFrame(rows)


def _pick_best_sample_row(score_df: pd.DataFrame) -> Optional[pd.Series]:
    if score_df.empty:
        return None
    valid = score_df[score_df["score_valid"].astype(bool)]
    if valid.empty:
        return None
    return valid.loc[valid["weighted_score"].idxmin()]


def _align_single_prediction_np(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    cell_class_int: np.ndarray,
    procrustes_cfg: Any,
    device: Any,
) -> np.ndarray:
    pred_t = torch.as_tensor(
        np.asarray(pred_xy, dtype=np.float32), dtype=torch.float32, device=device
    )
    gt_t = torch.as_tensor(
        np.asarray(gt_xy, dtype=np.float32), dtype=torch.float32, device=device
    )
    labels_t = torch.as_tensor(
        np.asarray(cell_class_int), dtype=torch.long, device=device
    )
    with torch.no_grad():
        aligned = align_samples_to_gt_procrustes_torch(
            pred_t.unsqueeze(0),
            gt_t,
            labels_t,
            procrustes_cfg=procrustes_cfg,
        )
    return aligned[0].detach().cpu().numpy().astype(np.float32)


def plot_slice_best_predictions_scatter(
    gt_df: pd.DataFrame,
    model_panels: list[tuple[str, str, pd.DataFrame]],
    save_path: Path,
    *,
    title: str,
) -> None:
    """One figure per slice: GT once, then each model's best prediction.

    ``model_panels`` entries are ``(panel_title, panel_subtitle, pred_df)``.
    Uses the same glasbey class palette as ``metrics.test_evaluation_plot``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from metrics.test_evaluation_plot import COLOR_PALETTE

    if not model_panels:
        raise ValueError("model_panels must be non-empty")

    all_class_vals = list(gt_df["cell_class"].astype(str))
    for _, _, pred_df in model_panels:
        all_class_vals.extend(pred_df["cell_class"].astype(str).tolist())
    classes = sorted(set(all_class_vals))
    pl_palette = sns.color_palette(COLOR_PALETTE, n_colors=max(len(classes), 1))
    palette_dict = dict(zip(classes, pl_palette))

    n_panels = 1 + len(model_panels)
    # Prefer a single row when few models; wrap otherwise.
    n_cols = min(n_panels, 4)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_w = 4.2 * n_cols
    fig_h = 4.4 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    panels: list[tuple[str, str, pd.DataFrame]] = [
        ("Ground truth", "", gt_df),
        *model_panels,
    ]

    xs_parts = [gt_df["coord_X"].to_numpy()]
    ys_parts = [gt_df["coord_Y"].to_numpy()]
    for _, _, pred_df in model_panels:
        xs_parts.append(pred_df["coord_X"].to_numpy())
        ys_parts.append(pred_df["coord_Y"].to_numpy())
    xs = np.concatenate(xs_parts)
    ys = np.concatenate(ys_parts)
    pad_x = 0.02 * max(float(xs.max() - xs.min()), 1e-6)
    pad_y = 0.02 * max(float(ys.max() - ys.min()), 1e-6)
    xlim = (float(xs.min()) - pad_x, float(xs.max()) + pad_x)
    ylim = (float(ys.min()) - pad_y, float(ys.max()) + pad_y)

    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= n_panels:
            ax.axis("off")
            continue
        panel_title, panel_subtitle, data = panels[ax_idx]
        sns.scatterplot(
            data=data,
            x="coord_X",
            y="coord_Y",
            hue="cell_class",
            hue_order=classes,
            palette=palette_dict,
            s=15,
            linewidth=0,
            ax=ax,
            legend=False,
        )
        full_title = panel_title if not panel_subtitle else f"{panel_title}\n{panel_subtitle}"
        ax.set_title(full_title, fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette_dict[c],
            markersize=7,
            label=c,
        )
        for c in classes
    ]
    ncol = max(1, min(6, (len(classes) + 3) // 4))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=1.02)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _load_best_aligned_prediction_for_section(
    *,
    section_dir: Path,
    sample_index: int,
    procrustes_cfg: Any,
    device: Any,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Return ``(gt_plot_df, aligned_pred_plot_df)`` for one sample, or ``(None, None)``."""
    pred_csv = section_dir / "predictions.csv"
    gt_csv = section_dir / "ground_truth.csv"
    if not pred_csv.exists() or not gt_csv.exists():
        return None, None

    pred_all = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    pred_s = pred_all[pred_all["sample_index"] == sample_index].copy()
    if pred_s.empty:
        return None, None

    if "cell_ID" in gt_df.columns and "cell_ID" in pred_s.columns:
        gt_ids = gt_df["cell_ID"].astype(str)
        pred_s["cell_ID"] = pred_s["cell_ID"].astype(str)
        pred_s = pred_s.set_index("cell_ID").reindex(gt_ids).reset_index()
        if pred_s[["coord_X", "coord_Y"]].isna().any().any():
            return None, None

    pred_xy = pred_s[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    gt_xy = gt_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    labels = gt_df["cell_class_int"].to_numpy()
    classes = gt_df["cell_class"].astype(str).tolist()
    aligned_xy = _align_single_prediction_np(
        pred_xy, gt_xy, labels, procrustes_cfg, device
    )
    pred_plot = pd.DataFrame(
        {
            "coord_X": aligned_xy[:, 0],
            "coord_Y": aligned_xy[:, 1],
            "cell_class": classes,
        }
    )
    gt_plot = gt_df[["coord_X", "coord_Y", "cell_class"]].copy()
    gt_plot["cell_class"] = gt_plot["cell_class"].astype(str)
    return gt_plot, pred_plot


def plot_best_predictions_for_models(
    cfg: DictConfig,
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
) -> Optional[Path]:
    """Select best seed per model×slice; one PNG per slice (GT once + models).

    Writes under ``{testing_pipeline}/best_predictions/``, next to the
    checkpoint comparison artefacts. No-op when disabled or no slices set.
    """
    pipe_cfg = cfg.test.pipeline
    best_cfg = getattr(pipe_cfg, "best_prediction_plots", None)
    if best_cfg is None or not bool(getattr(best_cfg, "enabled", False)):
        return None

    sections_raw = getattr(best_cfg, "cell_sections", None)
    if not sections_raw:
        print("[testing_pipeline] best_prediction_plots enabled but no cell_sections.")
        return None
    cell_sections = [str(s) for s in list(sections_raw)]
    weights = _loss_weights_from_cfg(best_cfg)
    if not any(abs(w) > 0.0 for w in weights.values()):
        print("[testing_pipeline] best_prediction_plots: all loss_weights are 0; skip.")
        return None

    parent_dir = _testing_pipeline_parent_dir(cfg, checkpoint_paths)
    out_dir = parent_dir / "best_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    display_names = resolve_checkpoint_display_names(cfg, checkpoint_paths)
    procrustes_cfg = getattr(pipe_cfg, "procrustes", None)
    device = resolve_inference_devices(cfg, str(pipe_cfg.device))[0]
    seed_start = int(pipe_cfg.seed_start)

    selection_rows: list[dict[str, Any]] = []
    print(
        f"[testing_pipeline] Best-prediction plots → {out_dir} "
        f"(sections={cell_sections})"
    )

    for cell_section in cell_sections:
        gt_plot: Optional[pd.DataFrame] = None
        model_panels: list[tuple[str, str, pd.DataFrame]] = []

        for checkpoint_path, output_dir, model_name in zip(
            checkpoint_paths, output_dirs, display_names
        ):
            section_dir = (
                Path(output_dir) / "sections" / _safe_section_dirname(cell_section)
            )
            metrics_df = _load_section_per_sample_metrics(section_dir)
            if metrics_df.empty:
                print(
                    f"[testing_pipeline] No per-sample metrics for "
                    f"{model_name!r} / {cell_section!r}; skip."
                )
                continue

            score_df = _weighted_seed_scores(metrics_df, weights)
            best_row = _pick_best_sample_row(score_df)
            if best_row is None:
                print(
                    f"[testing_pipeline] No valid weighted score for "
                    f"{model_name!r} / {cell_section!r}; skip."
                )
                continue

            sample_index = int(best_row["sample_index"])
            score = float(best_row["weighted_score"])
            seed = seed_start + sample_index

            gt_i, pred_plot = _load_best_aligned_prediction_for_section(
                section_dir=section_dir,
                sample_index=sample_index,
                procrustes_cfg=procrustes_cfg,
                device=device,
            )
            if gt_i is None or pred_plot is None:
                print(
                    f"[testing_pipeline] Could not load/align best sample for "
                    f"{model_name!r} / {cell_section!r}; skip."
                )
                continue

            if gt_plot is None:
                gt_plot = gt_i

            panel_subtitle = f"seed={seed}  score={score:.4g}"
            model_panels.append((model_name, panel_subtitle, pred_plot))

            selection_rows.append({
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "cell_section": cell_section,
                "sample_index": sample_index,
                "seed": seed,
                "weighted_score": score,
                "png": str(out_dir / f"{_safe_section_dirname(cell_section)}.png"),
                **{
                    m: float(best_row[f"metric__{m}"])
                    if f"metric__{m}" in best_row.index
                    else float("nan")
                    for m, w in weights.items()
                    if abs(float(w)) > 0.0
                },
            })

        if gt_plot is None or not model_panels:
            print(
                f"[testing_pipeline] No models available for slice "
                f"{cell_section!r}; skip plot."
            )
            continue

        save_path = out_dir / f"{_safe_section_dirname(cell_section)}.png"
        plot_slice_best_predictions_scatter(
            gt_plot,
            model_panels,
            save_path,
            title=f"Best predictions  ·  {cell_section}",
        )
        print(f"[testing_pipeline] Saved best-prediction plot → {save_path}")

    if selection_rows:
        sel_csv = out_dir / "best_prediction_selection.csv"
        pd.DataFrame(selection_rows).to_csv(sel_csv, index=False)
        print(f"[testing_pipeline] Saved best-prediction selection → {sel_csv}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Cross-checkpoint comparison
# ─────────────────────────────────────────────────────────────────────────────


def _checkpoint_label(checkpoint_path: Path) -> str:
    return _checkpoint_output_id(checkpoint_path)


def resolve_checkpoint_display_names(
    cfg: DictConfig,
    checkpoint_paths: list[Path],
) -> list[str]:
    """Pretty model names for comparison plots (parallel to ``checkpoint_paths``).

    Uses ``test.checkpoint_display_names`` when provided (same length/order as
    the resolved checkpoint list). Falls back to a short auto label built from
    the run folder + checkpoint stem.
    """
    raw = getattr(cfg.test, "checkpoint_display_names", None)
    if raw is not None:
        names = [str(x) for x in list(raw)]
        if len(names) != len(checkpoint_paths):
            raise ValueError(
                f"test.checkpoint_display_names has {len(names)} entries but "
                f"{len(checkpoint_paths)} checkpoints were resolved; "
                "lists must match in order and length."
            )
        return names
    return [_checkpoint_label(p) for p in checkpoint_paths]


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


def _section_rows(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Drop the aggregate ``ALL`` row from a per-section summary table."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "cell_section" not in df.columns:
        return df.copy()
    return df[df["cell_section"].astype(str) != "ALL"].copy()


def _collect_checkpoint_comparison(
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
    display_names: list[str],
    *,
    include_directional: bool = True,
    include_position_mse: bool = True,
) -> pd.DataFrame:
    """Long table of per-slice metric values for each model.

    Each row is one ``(model, metric, cell_section)`` after averaging over
    denoising seeds within that slice — **not** collapsed across slices.
    Columns: ``model``, ``checkpoint``, ``cell_section``, ``metric``, ``value``.
    """
    if len(display_names) != len(checkpoint_paths):
        raise ValueError("display_names must match checkpoint_paths length")

    order = []
    if include_position_mse:
        order.append("position_mse")
    if include_directional:
        order.extend(["directional_length", "directional_pairwise"])
    order.append("transcriptome_multi_radius")
    order.extend([
        "ch_energy_curve_loss",
        "voronoi_phase_pair_ch_energy_loss",
        "mmd_whole_slice",
        "mmd_pair_dist",
        "mmd_spatial",
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
    for ckpt, out_dir, model in zip(checkpoint_paths, output_dirs, display_names):
        ckpt_id = _checkpoint_label(ckpt)
        loaded = _load_checkpoint_summaries(out_dir)

        m = _section_rows(loaded.get("mmd"))
        if not m.empty:
            for _, r in m.iterrows():
                section = str(r.get("cell_section", "unknown"))
                for metric in ("mmd_whole_slice", "mmd_pair_dist", "mmd_spatial"):
                    col = f"{metric}_mean"
                    if col in m.columns and pd.notna(r.get(col)):
                        rows.append({
                            "model": model,
                            "checkpoint": ckpt_id,
                            "cell_section": section,
                            "metric": metric,
                            "value": float(r[col]),
                        })

        cv = _section_rows(loaded.get("ch_voronoi"))
        if not cv.empty:
            for _, r in cv.iterrows():
                section = str(r.get("cell_section", "unknown"))
                for metric in (
                    "ch_energy_curve_loss",
                    "voronoi_phase_pair_ch_energy_loss",
                ):
                    col = f"{metric}_mean"
                    if col in cv.columns and pd.notna(r.get(col)):
                        rows.append({
                            "model": model,
                            "checkpoint": ckpt_id,
                            "cell_section": section,
                            "metric": metric,
                            "value": float(r[col]),
                        })

        sp = loaded.get("spread")
        if sp is not None and not sp.empty:
            for _, r in sp.iterrows():
                section = str(r.get("cell_section", "unknown"))
                if "mean_cross_sample_spread" in sp.columns and pd.notna(
                    r.get("mean_cross_sample_spread")
                ):
                    rows.append({
                        "model": model,
                        "checkpoint": ckpt_id,
                        "cell_section": section,
                        "metric": "cross_sample_spread",
                        "value": float(r["mean_cross_sample_spread"]),
                    })

        sl = loaded.get("scalar_losses")
        if sl is not None and not sl.empty:
            section_col = (
                "cell_section" if "cell_section" in sl.columns
                else ("section" if "section" in sl.columns else None)
            )
            for _, r in sl.iterrows():
                metric_name = str(r["metric"])
                if metric_name not in allowed_scalar:
                    continue
                section = "unknown" if section_col is None else str(r[section_col])
                if pd.notna(r.get("mean")):
                    rows.append({
                        "model": model,
                        "checkpoint": ckpt_id,
                        "cell_section": section,
                        "metric": metric_name,
                        "value": float(r["mean"]),
                    })

        gt_tx = _section_rows(loaded.get("grid_transcript"))
        if not gt_tx.empty:
            for _, r in gt_tx.iterrows():
                section = str(r.get("cell_section", "unknown"))
                for metric in (
                    "grid_transcript_diff",
                    "grid_transcript_soft_spearman",
                ):
                    col = f"{metric}_mean"
                    if col in gt_tx.columns and pd.notna(r.get(col)):
                        rows.append({
                            "model": model,
                            "checkpoint": ckpt_id,
                            "cell_section": section,
                            "metric": metric,
                            "value": float(r[col]),
                        })

    df = pd.DataFrame.from_records(
        rows,
        columns=["model", "checkpoint", "cell_section", "metric", "value"],
    )
    if df.empty:
        return df
    df = df[df["metric"].isin(order)].copy()
    df["_order"] = df["metric"].map({m: i for i, m in enumerate(order)})
    model_order = {name: i for i, name in enumerate(display_names)}
    df["_model_order"] = df["model"].map(model_order)
    df = (
        df.sort_values(["_order", "_model_order", "cell_section"])
        .drop(columns=["_order", "_model_order"])
        .reset_index(drop=True)
    )
    return df


# Metrics where larger values are better; everything else is minimize.
_HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({
    "grid_transcript_soft_spearman",
})


def _save_combined_comparison_csv(
    comparison: pd.DataFrame,
    save_path: Path,
) -> None:
    """Write the per-slice checkpoint-comparison table to CSV."""
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
        if not np.isfinite(denom) or denom <= 0:
            df.loc[sub.index, "normalized_score"] = 1.0
        elif str(metric) in _HIGHER_IS_BETTER_METRICS:
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
        ("MMD whole-slice", ["mmd_whole_slice"]),
        ("MMD pair-dist", ["mmd_pair_dist"]),
        ("MMD spatial (per-class)", ["mmd_spatial"]),
        (
            "Cahn-Hilliard",
            ["ch_energy_curve_loss", "voronoi_phase_pair_ch_energy_loss"],
        ),
        ("Cross-sample spread", ["cross_sample_spread"]),
        ("Grid transcript MAE", ["grid_transcript_diff"]),
        ("Grid transcript Spearman", ["grid_transcript_soft_spearman"]),
    ])
    return panels


_COMPARISON_METRIC_LABELS: dict[str, str] = {
    "mmd_whole_slice": "whole-slice",
    "mmd_pair_dist": "pair-dist",
    "mmd_spatial": "spatial (per-class)",
    "ch_energy_curve_loss": "CH energy",
    "voronoi_phase_pair_ch_energy_loss": "Voronoi phase",
    "grid_transcript_diff": "soft weighted-sum MAE",
    "grid_transcript_soft_spearman": "soft Spearman",
}

def _metric_direction_arrow(metric: str) -> str:
    """↑ = maximize, ↓ = minimize."""
    return "↑" if str(metric) in _HIGHER_IS_BETTER_METRICS else "↓"


def _panel_title_with_arrow(title: str, metrics: list[str]) -> str:
    """Append minimize/maximize arrow(s) to a panel title."""
    arrows = [_metric_direction_arrow(m) for m in metrics]
    if len(set(arrows)) == 1:
        return f"{title}  {arrows[0]}"
    # Mixed panel: annotate each metric short-label with its own arrow.
    parts = [
        f"{_COMPARISON_METRIC_LABELS.get(m, m)} {_metric_direction_arrow(m)}"
        for m in metrics
    ]
    return f"{title}  ({' · '.join(parts)})"


def plot_checkpoint_comparison(
    comparison: pd.DataFrame,
    save_path: Path,
    *,
    include_directional: bool = True,
    include_position_mse: bool = True,
) -> None:
    """Large 3-column seaborn violin grid across models (distribution over slices).

    Each violin is the distribution of per-slice values for one model (seeds
    already averaged within each slice). Panel titles include ↑ (maximize) or
    ↓ (minimize). Display names come from the ``model`` column
    (``test.checkpoint_display_names``).
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)
    plt.rcParams.update({
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#dddddd",
        "figure.facecolor": "#f7f8fb",
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

    models = list(dict.fromkeys(comparison["model"].tolist()))
    plot_df = comparison.copy()
    plot_df["metric_label"] = plot_df["metric"].map(
        lambda m: (
            f"{_COMPARISON_METRIC_LABELS.get(str(m), str(m))} "
            f"{_metric_direction_arrow(str(m))}"
        )
    )
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=models, ordered=True)

    n_panels = len(panels)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    # Very large canvas so each violin has room; 3×3 ≈ 30×28 inches.
    fig_w = 10.0 * ncols
    fig_h = 9.0 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        facecolor=plt.rcParams["figure.facecolor"],
        squeeze=False,
    )
    axes_flat = axes.ravel()

    def _style_axis(ax, *, show_ylabel: bool) -> None:
        sns.despine(ax=ax, left=False, bottom=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="-", alpha=0.65)
        ax.xaxis.grid(False)
        ax.tick_params(axis="y", labelsize=12, colors="#444444")
        ax.tick_params(axis="x", labelsize=11, colors="#333333")
        ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel("Value (per slice)", fontsize=13, color="#555555")
        else:
            ax.set_ylabel("")

    def _intense_shade(rgba, factor: float = 0.55):
        """Darker shade of a face color for edges / mean ticks."""
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(rgba[:3])
        a = float(rgba[3]) if len(rgba) > 3 else 1.0
        return (r * factor, g * factor, b * factor, a)

    def _lighten_rgb(rgb, amount: float = 0.38):
        """Mix with white so two hues of the same model stay distinguishable."""
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(rgb[:3])
        return (
            r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount,
        )

    def _style_violin_bodies(ax, *, edge_factor: float = 0.52, edge_lw: float = 2.4):
        """Recolor violin outlines to a deeper shade of each fill (not black)."""
        from matplotlib.collections import PolyCollection

        body_colors: list = []
        for coll in ax.collections:
            if not isinstance(coll, PolyCollection):
                continue
            fcs = coll.get_facecolors()
            if fcs is None or len(fcs) == 0:
                continue
            edges = [_intense_shade(fc, edge_factor) for fc in fcs]
            coll.set_edgecolors(edges)
            coll.set_linewidth(edge_lw)
            coll.set_zorder(2)
            body_colors.append(_intense_shade(fcs[0], edge_factor))
        return body_colors

    def _recolor_paired_metric_violins(
        ax,
        models: list,
        *,
        shade_amount: float = 0.38,
        face_alpha: float = 0.92,
    ) -> None:
        """Use deep per-model colors; second metric = slightly lighter shade.

        Assumes seaborn drew PolyCollections in (model, hue) order — one body
        per (model × metric) for the dodged Cahn–Hilliard panel.
        """
        from matplotlib.collections import PolyCollection

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        deep = sns.color_palette("deep", n_colors=max(len(models), 1))
        expected = len(models) * 2
        if len(polys) < expected:
            return
        idx = 0
        for i, _model in enumerate(models):
            base = deep[i % len(deep)]
            for shade in (base, _lighten_rgb(base, shade_amount)):
                polys[idx].set_facecolor((*shade, face_alpha))
                idx += 1

    def _draw_mean_ticks(
        ax,
        sub: pd.DataFrame,
        *,
        models: list,
        body_colors: list,
        hue_col: str | None = None,
        hue_order: list | None = None,
        mean_lw: float = 4.0,
        tick_half_width: float = 0.18,
    ) -> None:
        """Thick horizontal mean tick per violin, colored like its outline."""
        if hue_col is None or not hue_order or len(hue_order) <= 1:
            for i, model in enumerate(models):
                vals = sub.loc[sub["model"] == model, "value"]
                if vals.empty:
                    continue
                m = float(vals.mean())
                color = body_colors[i] if i < len(body_colors) else "#333333"
                ax.hlines(
                    m,
                    i - tick_half_width,
                    i + tick_half_width,
                    colors=[color],
                    linewidth=mean_lw,
                    zorder=5,
                )
            return

        n_hue = len(hue_order)
        width = 0.8
        offsets = np.linspace(-(n_hue - 1) / 2, (n_hue - 1) / 2, n_hue) * (
            width / max(n_hue, 1)
        )
        half = 0.5 * (width / n_hue) * 0.75
        color_idx = 0
        for i, model in enumerate(models):
            for j, hue in enumerate(hue_order):
                vals = sub.loc[
                    (sub["model"] == model) & (sub[hue_col] == hue), "value"
                ]
                if vals.empty:
                    continue
                m = float(vals.mean())
                color = (
                    body_colors[color_idx]
                    if color_idx < len(body_colors)
                    else "#333333"
                )
                color_idx += 1
                x = float(i + offsets[j])
                ax.hlines(
                    m,
                    x - half,
                    x + half,
                    colors=[color],
                    linewidth=mean_lw,
                    zorder=5,
                )

    violin_kwargs = dict(
        inner=None,  # draw our own thick mean ticks instead of thin quartile lines
        cut=0,
        linewidth=0.0,  # edges restyled after draw
    )
    try:
        import inspect
        sig = inspect.signature(sns.violinplot)
        if "density_norm" in sig.parameters:
            violin_kwargs["density_norm"] = "width"
        else:
            violin_kwargs["scale"] = "width"
    except Exception:
        violin_kwargs["scale"] = "width"

    for idx, ax in enumerate(axes_flat):
        if idx >= n_panels:
            ax.set_visible(False)
            continue

        title, metrics = panels[idx]
        panel_title = _panel_title_with_arrow(title, metrics)
        sub = plot_df[plot_df["metric"].isin(metrics)].copy()
        if sub.empty:
            ax.set_title(panel_title, fontsize=16, pad=14)
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=14, color="#888888",
            )
            ax.set_axis_off()
            continue

        n_metrics = len([m for m in metrics if m in set(sub["metric"])])
        row_i, col_i = divmod(idx, ncols)
        if n_metrics == 1:
            sns.violinplot(
                data=sub,
                x="model",
                y="value",
                hue="model",
                palette="deep",
                legend=False,
                ax=ax,
                **violin_kwargs,
            )
            body_colors = _style_violin_bodies(ax)
            _draw_mean_ticks(
                ax, sub, models=models, body_colors=body_colors, mean_lw=4.2
            )
            sns.stripplot(
                data=sub,
                x="model",
                y="value",
                color="#222222",
                alpha=0.40,
                size=5.0,
                jitter=0.15,
                ax=ax,
                zorder=3,
            )
        else:
            hue_order = [
                (
                    f"{_COMPARISON_METRIC_LABELS.get(str(m), str(m))} "
                    f"{_metric_direction_arrow(str(m))}"
                )
                for m in metrics
                if m in set(sub["metric"].astype(str))
            ]
            # Draw with any 2-color palette first; faces are recolored to deep
            # per-model colors (second metric = lighter shade of the same).
            sns.violinplot(
                data=sub,
                x="model",
                y="value",
                hue="metric_label",
                hue_order=hue_order,
                palette="deep",
                dodge=True,
                ax=ax,
                **violin_kwargs,
            )
            _recolor_paired_metric_violins(ax, models)
            body_colors = _style_violin_bodies(ax)
            _draw_mean_ticks(
                ax,
                sub,
                models=models,
                body_colors=body_colors,
                hue_col="metric_label",
                hue_order=hue_order,
                mean_lw=4.0,
            )
            sns.stripplot(
                data=sub,
                x="model",
                y="value",
                hue="metric_label",
                hue_order=hue_order,
                dodge=True,
                palette="dark:#222222",
                alpha=0.40,
                size=4.5,
                jitter=0.10,
                ax=ax,
                legend=False,
                zorder=3,
            )
            from matplotlib.patches import Patch

            legend_base = sns.color_palette("deep", n_colors=1)[0]
            legend_handles = [
                Patch(
                    facecolor=legend_base,
                    edgecolor=_intense_shade(legend_base)[:3],
                    linewidth=1.5,
                    label=hue_order[0],
                ),
                Patch(
                    facecolor=_lighten_rgb(legend_base),
                    edgecolor=_intense_shade(_lighten_rgb(legend_base))[:3],
                    linewidth=1.5,
                    label=hue_order[1] if len(hue_order) > 1 else "alt",
                ),
            ]
            ax.legend(
                handles=legend_handles[:n_metrics],
                fontsize=11,
                loc="best",
                title="Metric",
                title_fontsize=11,
                frameon=True,
            )

        ax.set_title(panel_title, fontsize=16, pad=14, color="#1a1a1a")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        _style_axis(ax, show_ylabel=(col_i == 0))

    fig.suptitle(
        "Model benchmark — per-slice metric distributions\n"
        "↑ higher is better   ·   ↓ lower is better",
        y=0.995,
        fontsize=22,
        fontweight="bold",
        color="#222222",
        linespacing=1.35,
    )
    fig.subplots_adjust(
        top=0.92,
        bottom=0.07,
        left=0.06,
        right=0.98,
        wspace=0.28,
        hspace=0.38,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[testing_pipeline] Saved checkpoint comparison plot → {save_path}")


def compare_checkpoints(
    cfg: DictConfig,
    checkpoint_paths: list[Path],
    output_dirs: list[Path],
) -> Path:
    """Load every checkpoint's summaries and write a combined CSV + violin PNG.

    Artefacts are written into the shared ``testing_pipeline`` parent
    directory (the common ancestor of all per-checkpoint output dirs).
    """
    parent_dir = _testing_pipeline_parent_dir(cfg, checkpoint_paths)
    parent_dir.mkdir(parents=True, exist_ok=True)

    scalar_cfg = cfg.test.pipeline.scalar_losses_analysis
    include_directional = bool(getattr(scalar_cfg, "compute_directional", True))
    include_position_mse = bool(getattr(scalar_cfg, "compute_position_mse", True))
    display_names = resolve_checkpoint_display_names(cfg, checkpoint_paths)

    combined = _collect_checkpoint_comparison(
        checkpoint_paths,
        output_dirs,
        display_names,
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
