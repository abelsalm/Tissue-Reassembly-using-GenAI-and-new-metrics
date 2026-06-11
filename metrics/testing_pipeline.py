"""Statistical testing pipeline for LUNA checkpoints.

For each checkpoint listed in ``configs/test/default.yaml`` the pipeline:

1. Runs ``n`` denoising samples (different seeds) on every ``cell_section`` of
   the test split. Sampling is organised **seed-first**: for each seed, all
   sections are inferred in batches of ``test.batch_size`` graphs; analysis
   runs only after every seed/section pair has been collected.
2. Builds spatial probability maps per cell class
   (``utils/testing/diffusion2spatial_probs.py``) and scores ground-truth
   positions under those maps (``utils/testing/sampling_proximity.py``).
   For each ``cell_section``, also saves a multi-panel figure with one
   subplot per cell class (density heatmap + aligned GT scatter).
3. Computes standard evaluation metrics
   (``metrics/evaluation_statistics.py``) on every sample and aggregates
   results across seeds and sections.
4. After Procrustes alignment (same as spatial-probability maps), measures
   per-cell mean pairwise distance across denoising samples, aggregates by
   cell type, and plots type-wise and overall averages.

Usage
-----
    python metrics/testing_pipeline.py \\
        experiment=MERFISH_small_transcripts \\
        general.mode=test_only

    python metrics/testing_pipeline.py \\
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
from typing import Any, Iterable

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

from metrics.evaluation_statistics import (  # noqa: E402
    compute_contact,
    compute_RSSD,
    compute_spearman_correlation,
)
from utils.data.load import compute_distance, position_normalize, to_dataframe  # noqa: E402
from utils.testing.diffusion2spatial_probs import (  # noqa: E402
    REPO_ROOT as _DIFFUSION_REPO_ROOT,
    SliceBatch,
    _align_samples,
    _resolve_device,
    _set_seed,
    build_section_batch,
    compute_spatial_probability_map,
    load_model,
    save_predictions_csv,
)
from utils.diffusion_model.sample.sample import sample_from_single_graph  # noqa: E402
from utils.testing.sampling_proximity import (  # noqa: E402
    _align_real_positions,
    _evaluate_density,
    _safe_plot_filename,
)

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
    for cell_section in tqdm(cell_sections, desc="Building section batches", unit="section"):
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
    for seed in tqdm(seeds, desc="Seeds", unit="seed"):
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
# Spatial probability maps & GT likelihood scoring
# ─────────────────────────────────────────────────────────────────────────────


def _result_to_distribution(result: dict, label: str) -> dict:
    return {
        "density": result["density"],
        "grid_x": result["grid_x"],
        "grid_y": result["grid_y"],
        "cell_class": str(result["cell_class"]),
        "label": label,
        "reference_all_points": result["reference_all_points"],
        "reference_cell_ids": np.asarray(result["reference_cell_ids"], dtype=str),
    }


def save_density_npz(result: dict, save_path: Path, label: str, source_csv: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        density=result["density"],
        grid_x=result["grid_x"],
        grid_y=result["grid_y"],
        extent=np.asarray(result["extent"]),
        reference_points=result["reference_points"],
        reference_all_points=result["reference_all_points"],
        reference_cell_ids=np.asarray(result["reference_cell_ids"], dtype=str),
        reference_cell_classes=np.asarray(result["reference_cell_classes"], dtype=str),
        align_reference=result["align_reference"],
        cell_class=str(result["cell_class"]),
        gaussian_sigma=result["gaussian_sigma"],
        num_samples=result["num_samples"],
        num_class_cells=result["num_class_cells"],
        source_csv=str(source_csv),
        label=str(label),
    )


def build_section_densities(
    predictions_csv: Path,
    gt_df: pd.DataFrame,
    cell_classes: Iterable[str],
    spatial_cfg: Any,
    density_dir: Path,
    section_label: str,
) -> dict[str, dict]:
    """Compute and save one spatial density per cell class for a section.

    Returns a mapping ``cell_class -> {"result": ..., "dist": ...}`` where
    ``result`` is the full output of ``compute_spatial_probability_map`` and
    ``dist`` is the slim dict used for likelihood scoring.
    """
    section_densities: dict[str, dict] = {}
    for cell_class in cell_classes:
        result = compute_spatial_probability_map(
            predictions_csv=predictions_csv,
            cell_class=str(cell_class),
            gaussian_sigma=float(spatial_cfg.gaussian_sigma),
            grid_resolution=int(spatial_cfg.grid_resolution),
            grid_margin=float(spatial_cfg.grid_margin),
            align_reference=int(spatial_cfg.align_reference),
        )
        safe_name = _safe_plot_filename(str(cell_class))
        npz_path = density_dir / f"{safe_name}.npz"
        save_density_npz(result, npz_path, label=section_label, source_csv=predictions_csv)
        section_densities[str(cell_class)] = {
            "result": result,
            "dist": _result_to_distribution(result, label=section_label),
        }
    return section_densities


def _subplot_grid_shape(n_panels: int, max_cols: int = 4) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def plot_section_density_grid(
    gt_df: pd.DataFrame,
    section_densities: dict[str, dict],
    cell_section: str,
    save_path: Path,
    overlay_size: float = 6.0,
    max_cols: int = 4,
    cmap: str = "magma",
) -> None:
    """Save one figure with a subplot per cell class: density + aligned GT cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not section_densities:
        return

    gt_df = gt_df.copy()
    gt_df["cell_class"] = gt_df["cell_class"].astype(str)
    align_args = argparse.Namespace(
        cell_id_column="cell_ID",
        x_column="coord_X",
        y_column="coord_Y",
    )

    cell_classes = sorted(section_densities.keys())
    first_dist = section_densities[cell_classes[0]]["dist"]
    aligned_all, matched, ref_total = _align_real_positions(gt_df, first_dist, align_args)

    nrows, ncols = _subplot_grid_shape(len(cell_classes), max_cols=max_cols)
    fig_w = 4.5 * ncols
    fig_h = 4.0 * nrows + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, cell_class in enumerate(cell_classes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        entry = section_densities[cell_class]
        result = entry["result"]

        im = ax.imshow(
            result["density"],
            origin="lower",
            extent=result["extent"],
            cmap=cmap,
            aspect="equal",
            interpolation="bilinear",
        )
        class_mask = gt_df["cell_class"] == cell_class
        aligned_gt = aligned_all[class_mask.to_numpy()]
        if aligned_gt.shape[0] > 0:
            ax.scatter(
                aligned_gt[:, 0],
                aligned_gt[:, 1],
                s=overlay_size,
                color="cyan",
                marker="o",
                linewidths=0,
                label=f"GT ({int(class_mask.sum())} cells)",
            )
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        ax.set_title(
            f"{cell_class}\n"
            f"{result['num_samples']} samples × {result['num_class_cells']} cells",
            fontsize=9,
        )
        ax.set_xlabel("X (aligned)", fontsize=8)
        ax.set_ylabel("Y (aligned)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(cell_classes), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle(
        f"Spatial probability maps with GT overlay — {cell_section!r}\n"
        f"aligned with {matched}/{ref_total} reference cells, "
        f"sigma={section_densities[cell_classes[0]]['result']['gaussian_sigma']:g}",
        fontsize=11,
        y=1.02,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved section density grid → {save_path}")


def score_gt_under_distribution(
    gt_df: pd.DataFrame,
    dist: dict,
    cell_class: str,
    epsilon: float,
) -> dict | None:
    """Return summary statistics for GT cells of ``cell_class`` under ``dist``."""
    gt_df = gt_df.copy()
    gt_df["cell_class"] = gt_df["cell_class"].astype(str)
    class_mask = gt_df["cell_class"] == str(cell_class)
    if not class_mask.any():
        return None

    align_args = argparse.Namespace(
        cell_id_column="cell_ID",
        x_column="coord_X",
        y_column="coord_Y",
    )
    aligned_all, matched, ref_total = _align_real_positions(gt_df, dist, align_args)
    aligned_class = aligned_all[class_mask.to_numpy()]

    likelihood, outside = _evaluate_density(
        dist["density"],
        dist["grid_x"],
        dist["grid_y"],
        aligned_class,
        epsilon,
    )
    log_lik = np.log(likelihood)
    return {
        "cell_class": str(cell_class),
        "mean_log_likelihood": float(log_lik.mean()),
        "median_log_likelihood": float(np.median(log_lik)),
        "mean_likelihood": float(likelihood.mean()),
        "n_cells": int(class_mask.sum()),
        "outside_grid_fraction": float(outside.mean()),
        "alignment_matched_cells": int(matched),
        "alignment_reference_cells": int(ref_total),
    }


def score_section_likelihoods(
    gt_df: pd.DataFrame,
    section_densities: dict[str, dict],
    cell_section: str,
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for cell_class, entry in section_densities.items():
        summary = score_gt_under_distribution(
            gt_df, entry["dist"], cell_class, epsilon
        )
        if summary is None:
            continue
        summary["cell_section"] = cell_section
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_likelihoods_across_sections(section_scores: list[pd.DataFrame]) -> pd.DataFrame:
    if not section_scores:
        return pd.DataFrame()
    combined = pd.concat(section_scores, ignore_index=True)
    grouped = (
        combined.groupby("cell_class", as_index=False)
        .agg(
            mean_log_likelihood=("mean_log_likelihood", "mean"),
            std_log_likelihood=("mean_log_likelihood", "std"),
            median_log_likelihood=("median_log_likelihood", "mean"),
            mean_likelihood=("mean_likelihood", "mean"),
            n_sections=("cell_section", "nunique"),
            total_cells=("n_cells", "sum"),
            mean_outside_grid_fraction=("outside_grid_fraction", "mean"),
        )
        .sort_values("cell_class")
    )
    grouped["std_log_likelihood"] = grouped["std_log_likelihood"].fillna(0.0)
    return grouped


def plot_likelihood_summary(
    summary_df: pd.DataFrame,
    save_path: Path,
    n_sections_total: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No likelihood data to plot.")
        return

    plot_df = summary_df.sort_values("mean_log_likelihood")
    height = max(4.0, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, height))
    yerr = plot_df["std_log_likelihood"].to_numpy()
    ax.barh(
        plot_df["cell_class"],
        plot_df["mean_log_likelihood"],
        xerr=yerr,
        color="tab:blue",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel("Mean log likelihood (averaged across cell sections)")
    ax.set_ylabel("Cell class")
    if n_sections_total is None:
        n_sections_total = int(plot_df["n_sections"].max())
    ax.set_title(
        "Ground-truth likelihood under predicted spatial distributions\n"
        f"({n_sections_total} cell sections, "
        f"{int(plot_df['total_cells'].sum())} cells total)"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved likelihood plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (Spearman, RSSD, contact)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_single_sample(
    batch: SliceBatch,
    pred_positions: np.ndarray,
    sample_index: int,
    seed: int,
) -> dict:
    """Compute evaluation metrics for one denoising sample."""
    cell_class = batch.cell_class_labels
    positions_true = batch.gt_positions
    cell_id = batch.cell_ids

    metadata_true = to_dataframe(cell_class, positions_true, index=cell_id)
    metadata_pred_raw = to_dataframe(cell_class, pred_positions, index=cell_id)
    metadata_pred = position_normalize(metadata_pred_raw.copy())

    metadata_true = metadata_true.fillna(0)
    metadata_pred = metadata_pred.fillna(0)

    _, _, spr_avg, spr_median = compute_spearman_correlation(
        metadata_true, metadata_pred
    )
    sum_rssd, mean_rssd, absolute_rssd = compute_RSSD(metadata_true, metadata_pred)

    distances_true = compute_distance(metadata_true)
    distances_pred = compute_distance(metadata_pred)
    precision, f1 = compute_contact(distances_true, distances_pred, percentile=0.1)

    return {
        "cell_section": batch.cell_section,
        "sample_index": sample_index,
        "seed": seed,
        "precision": precision,
        "f1": f1,
        "spearman_avg": spr_avg,
        "spearman_median": spr_median,
        "rssd_absolute": absolute_rssd,
        "mean_rssd": mean_rssd,
        "sum_rssd": sum_rssd,
        "num_cells": len(metadata_true),
    }


def evaluate_section_samples(
    batch: SliceBatch,
    preds: np.ndarray,
    seed_start: int,
) -> pd.DataFrame:
    rows = []
    for i in range(preds.shape[0]):
        rows.append(
            evaluate_single_sample(batch, preds[i], sample_index=i, seed=seed_start + i)
        )
    return pd.DataFrame(rows)


def aggregate_evaluation_metrics(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    if per_sample_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "precision",
        "f1",
        "spearman_avg",
        "spearman_median",
        "rssd_absolute",
        "mean_rssd",
        "sum_rssd",
    ]
    section_stats = (
        per_sample_df.groupby("cell_section")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    section_stats.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in section_stats.columns
    ]

    overall = {"cell_section": "ALL"}
    for col in metric_cols:
        overall[f"{col}_mean"] = float(per_sample_df[col].mean())
        overall[f"{col}_std"] = float(per_sample_df[col].std(ddof=0))
    overall_row = pd.DataFrame([overall])
    return pd.concat([section_stats, overall_row], ignore_index=True)


def plot_evaluation_metrics(summary_df: pd.DataFrame, save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("[testing_pipeline] No evaluation metrics to plot.")
        return

    overall = summary_df[summary_df["cell_section"] == "ALL"]
    if overall.empty:
        print("[testing_pipeline] No overall evaluation summary row found.")
        return

    row = overall.iloc[0]
    labels = [
        "Precision",
        "F1",
        "Spearman (avg)",
        "Spearman (median)",
        "RSSD (absolute)",
        "Mean RSSD",
        "Sum RSSD",
    ]
    keys = [
        "precision",
        "f1",
        "spearman_avg",
        "spearman_median",
        "rssd_absolute",
        "mean_rssd",
        "sum_rssd",
    ]
    means = [float(row[f"{k}_mean"]) for k in keys]
    stds = [float(row[f"{k}_std"]) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:green", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Evaluation metrics (mean ± std over all samples & sections)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved evaluation metrics plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell cross-sample position spread (after Procrustes alignment)
# ─────────────────────────────────────────────────────────────────────────────


def compute_per_cell_sample_spread(
    preds: np.ndarray,
    batch: SliceBatch,
    align_reference: int = 0,
) -> pd.DataFrame:
    """Mean pairwise distance per cell across Procrustes-aligned sample positions.

    Each denoising sample's full point cloud is rotation-aligned to
    ``preds[align_reference]`` (same convention as spatial probability maps).
    For cell *j*, the metric is the average Euclidean distance over all pairs
    of that cell's positions in the different samples.
    """
    from scipy.spatial.distance import pdist

    aligned = _align_samples(preds.astype(np.float32), reference_idx=align_reference)
    num_samples, num_cells, _ = aligned.shape
    spreads = np.zeros(num_cells, dtype=np.float64)
    if num_samples >= 2:
        for cell_idx in range(num_cells):
            spreads[cell_idx] = float(pdist(aligned[:, cell_idx, :]).mean())
    return pd.DataFrame(
        {
            "cell_ID": batch.cell_ids,
            "cell_class": [str(c) for c in batch.cell_class_labels],
            "mean_sample_position_spread": spreads,
        }
    )


def aggregate_spread_by_cell_class(per_cell_df: pd.DataFrame) -> pd.DataFrame:
    per_cell_df = per_cell_df.copy()
    per_cell_df["cell_class"] = per_cell_df["cell_class"].astype(str)
    grouped = (
        per_cell_df.groupby("cell_class", as_index=False)
        .agg(
            mean_spread=("mean_sample_position_spread", "mean"),
            std_spread=("mean_sample_position_spread", "std"),
            n_cells=("mean_sample_position_spread", "count"),
        )
        .sort_values("cell_class")
    )
    grouped["std_spread"] = grouped["std_spread"].fillna(0.0)
    return grouped


def aggregate_spread_across_sections(
    section_per_cell_tables: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return per-cell table, per-class summary, and overall mean spread."""
    if not section_per_cell_tables:
        return pd.DataFrame(), pd.DataFrame(), float("nan")
    combined = pd.concat(section_per_cell_tables, ignore_index=True)
    class_summary = aggregate_spread_by_cell_class(combined)
    overall_mean = float(combined["mean_sample_position_spread"].mean())
    return combined, class_summary, overall_mean


def plot_sample_spread_summary(
    class_summary: pd.DataFrame,
    overall_mean: float,
    save_path: Path,
    n_sections_total: int | None = None,
) -> None:
    """Bar chart of mean per-cell spread by cell type plus overall average."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if class_summary.empty:
        print("[testing_pipeline] No sample-spread data to plot.")
        return

    plot_df = class_summary.sort_values("mean_spread")
    height = max(4.0, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(
        plot_df["cell_class"],
        plot_df["mean_spread"],
        xerr=plot_df["std_spread"],
        color="tab:purple",
        alpha=0.85,
        capsize=3,
        label="Per cell type",
    )
    ax.axvline(
        overall_mean,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall avg ({overall_mean:.4g})",
    )
    ax.set_xlabel(
        "Mean pairwise distance across samples (after Procrustes alignment)"
    )
    ax.set_ylabel("Cell class")
    sections_note = (
        f", {n_sections_total} sections" if n_sections_total is not None else ""
    )
    ax.set_title(
        "Cross-sample positional spread per cell\n"
        f"(averaged over cells{sections_note}; "
        f"{int(plot_df['n_cells'].sum())} cells total)"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[testing_pipeline] Saved sample-spread plot → {save_path}")


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
    spatial_cfg = pipe_cfg.spatial_analysis
    num_samples = int(pipe_cfg.num_samples)
    seed_start = int(pipe_cfg.seed_start)
    min_cells = int(pipe_cfg.min_cells_per_class)
    batch_size = int(cfg.test.batch_size)
    epsilon = float(spatial_cfg.epsilon)

    output_dir = pipeline_output_dir(cfg, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "effective_config.yaml")

    device = _resolve_device(str(pipe_cfg.device))
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

    print("[testing_pipeline] Phase 2/2: analysis…")
    section_likelihood_tables: list[pd.DataFrame] = []
    section_spread_tables: list[pd.DataFrame] = []
    all_eval_rows: list[pd.DataFrame] = []
    align_reference = int(spatial_cfg.align_reference)

    for cell_section in tqdm(cell_sections, desc="Analysis", unit="section"):
        section_dir = output_dir / "sections" / _safe_section_dirname(cell_section)
        section_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = section_dir / "predictions.csv"
        gt_csv = section_dir / "ground_truth.csv"
        density_dir = section_dir / "densities"

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
        per_cell_spread = compute_per_cell_sample_spread(
            preds=preds,
            batch=batch,
            align_reference=align_reference,
        )
        per_cell_spread["cell_section"] = str(cell_section)
        per_cell_spread.to_csv(section_dir / "per_cell_sample_spread.csv", index=False)
        section_spread_tables.append(per_cell_spread)

        cell_classes = cell_classes_with_min_cells(gt_df, min_cells)
        if not cell_classes:
            print(
                f"[testing_pipeline] Skipping likelihoods for {cell_section!r}: "
                f"no cell class with >= {min_cells} cells."
            )
        else:
            section_densities = build_section_densities(
                predictions_csv=predictions_csv,
                gt_df=gt_df,
                cell_classes=cell_classes,
                spatial_cfg=spatial_cfg,
                density_dir=density_dir,
                section_label=str(cell_section),
            )
            plot_section_density_grid(
                gt_df=gt_df,
                section_densities=section_densities,
                cell_section=str(cell_section),
                save_path=section_dir / "spatial_density_grid_with_gt.png",
                overlay_size=float(getattr(spatial_cfg, "overlay_gt_scatter_size", 6.0)),
                max_cols=int(getattr(spatial_cfg, "section_density_plot_cols", 4)),
            )
            section_scores = score_section_likelihoods(
                gt_df=gt_df,
                section_densities=section_densities,
                cell_section=str(cell_section),
                epsilon=epsilon,
            )
            if not section_scores.empty:
                section_scores.to_csv(
                    section_dir / "likelihood_summary.csv", index=False
                )
                section_likelihood_tables.append(section_scores)

        eval_df = evaluate_section_samples(batch, preds, seed_start)
        eval_df.to_csv(section_dir / "evaluation_metrics_per_sample.csv", index=False)
        all_eval_rows.append(eval_df)

    likelihood_summary = aggregate_likelihoods_across_sections(section_likelihood_tables)
    likelihood_csv = output_dir / "likelihood_summary.csv"
    likelihood_plot = output_dir / "mean_log_likelihood_by_cell_class.png"
    if not likelihood_summary.empty:
        likelihood_summary.to_csv(likelihood_csv, index=False)
        plot_likelihood_summary(
            likelihood_summary,
            likelihood_plot,
            n_sections_total=len(cell_sections),
        )
    else:
        print("[testing_pipeline] No likelihood summaries produced.")

    spread_per_cell, spread_by_class, spread_overall = aggregate_spread_across_sections(
        section_spread_tables
    )
    if not spread_by_class.empty:
        spread_per_cell.to_csv(
            output_dir / "per_cell_sample_spread_all_sections.csv", index=False
        )
        spread_summary = spread_by_class.copy()
        spread_summary["overall_mean_spread"] = spread_overall
        spread_summary.to_csv(
            output_dir / "mean_sample_spread_by_cell_class.csv", index=False
        )
        plot_sample_spread_summary(
            spread_by_class,
            spread_overall,
            output_dir / "mean_sample_spread_by_cell_class.png",
            n_sections_total=len(cell_sections),
        )
    else:
        print("[testing_pipeline] No sample-spread summaries produced.")

    per_sample_eval = pd.concat(all_eval_rows, ignore_index=True)
    per_sample_eval.to_csv(output_dir / "evaluation_metrics_per_sample.csv", index=False)
    eval_summary = aggregate_evaluation_metrics(per_sample_eval)
    eval_summary.to_csv(output_dir / "evaluation_metrics_summary.csv", index=False)
    plot_evaluation_metrics(eval_summary, output_dir / "evaluation_metrics.png")

    print(f"[testing_pipeline] Finished checkpoint {checkpoint_path.name}.")
    return output_dir


def run_testing_pipeline(cfg: DictConfig) -> list[Path]:
    """Entry point: run the pipeline for every resolved checkpoint."""
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
    return output_dirs


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    assert _DIFFUSION_REPO_ROOT == REPO_ROOT
    run_testing_pipeline(cfg)


if __name__ == "__main__":
    main()
