"""Evaluate and visualize DirectionalMetricLoss on LiVAE comparison CSV slices.

Loads slice CSVs (gene columns, GT/pred coordinates, ``cell_section``,
``cell_class``). For each slice:

  1. Builds tensors and runs ``DirectionalMetricLoss`` (``n_target`` subsample).
  2. Visualizes **only** the loss subsample: the first-radius local
     orientation axes and the second-radius smoothed axes from
     ``train_directional_metric.py`` (not the full-slice viz in
     ``build_plot_with_pred.py``).
  3. Writes a summary PNG and per-slice arrow PNGs.

Usage::

    # GPU timing (request a GPU in Slurm):
    srun --gres=gpu:1 --mem=32G python metrics/test_loss_direcrtional.py \\
        --gpu --csv /path/to/test_results.csv

    python metrics/test_loss_direcrtional.py \\
        --csv /path/to/not_normalized_test_results.csv \\
        --output metrics/directional_loss_report.png
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import colorcet as cc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LIVAE_ROOT = Path("/data-master/code/Tissue-Reassembly-using-GenAI-and-new-metrics/results/")
DEFAULT_CSV = DEFAULT_LIVAE_ROOT / "data_test_observations" / "not_normalized_test_results.csv"

TRUE_X_COL = "coord_X"
TRUE_Y_COL = "coord_Y"
PRED_X_COL = "coord_X_test"
PRED_Y_COL = "coord_Y_test"
SECTION_COL = "cell_section"

N_TARGET = 1024
NEIGHBOR_RADIUS = 0.16
COHERENCE_RADIUS = 0.08
TRANS_BETA = 32.0
SOFT_BETA = 256.0
EPS = 1e-6
DENSITY_LENGTH_GATE = True  # Shorten pre-smoothing axis length for less-crowded-than-average neighborhoods.
DENSITY_LENGTH_BETA = 8.0    # Sigmoid sharpness for the density-length gate.
DENSITY_RADIUS_GATE = True  # Resize the first-radius neighborhood per cell (smaller radius for isolated cells).
DENSITY_RADIUS_BETA = 32.0    # Sigmoid sharpness for the density-radius gate.
TARGET_MEAN_ARROW_LENGTH = 0.04
TISSUE_SCATTER_SIZE = 6
LOSS_SEED = 0
DTYPE = torch.float32


@dataclass
class SliceResult:
    slice_name: str
    n_cells: int
    n_features: int
    loss_total: float
    loss_length: float
    loss_pairwise: float
    metric_seconds: float
    viz_seconds: float


def _setup_import_paths(livae_root: Path) -> None:
    for path in (REPO_ROOT, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    arrow_plot_dir = livae_root
    if str(arrow_plot_dir) not in sys.path:
        sys.path.insert(0, str(arrow_plot_dir))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(device_str: str | None, *, use_gpu: bool) -> torch.device:
    """Pick compute device; ``use_gpu=True`` requires CUDA."""
    if device_str is not None:
        return torch.device(device_str)

    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--gpu) but torch.cuda.is_available() is False. "
                "Request a GPU in your job (e.g. srun --gres=gpu:1 ...) or pass --device cpu."
            )
        return torch.device("cuda:0")

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    mem_gb = props.total_memory / (1024 ** 3)
    return f"cuda:{index} ({props.name}, {mem_gb:.1f} GiB)"


def _timed_call(device: torch.device, fn, *args, **kwargs) -> tuple[object, float]:
    _sync_device(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync_device(device)
    return result, time.perf_counter() - start


def list_slices(csv_path: Path, section_col: str = SECTION_COL) -> list[str]:
    from build_plot_with_pred import _read_comparison_csv

    header = _read_comparison_csv(csv_path, index_col="auto")
    if section_col not in header.columns:
        raise ValueError(f"Missing slice column {section_col!r} in {csv_path}")
    return sorted(header[section_col].astype(str).unique())


def load_slice_tensors(
    csv_path: Path,
    slice_name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from build_plot_with_pred import load_slice, normalize_coords

    slice_df = load_slice(
        csv_path,
        slice_name,
        normalize=False,
        x_col=TRUE_X_COL,
        y_col=TRUE_Y_COL,
        section_col=SECTION_COL,
    )
    feature_cols = slice_df.attrs["feature_cols"]

    required = {"cell_class", TRUE_X_COL, TRUE_Y_COL, PRED_X_COL, PRED_Y_COL}
    missing = required - set(slice_df.columns)
    if missing:
        raise ValueError(f"Slice {slice_name!r} is missing columns: {sorted(missing)}")

    true_plot_df = normalize_coords(slice_df, x_col=TRUE_X_COL, y_col=TRUE_Y_COL)
    pred_plot_df = normalize_coords(slice_df, x_col=PRED_X_COL, y_col=PRED_Y_COL)
    true_plot_df["cell_class"] = slice_df["cell_class"].astype(str).values
    pred_plot_df["cell_class"] = slice_df["cell_class"].astype(str).values

    feature_matrix = slice_df[feature_cols].to_numpy(dtype=np.float32)
    n_cells = len(slice_df)

    true_positions = torch.as_tensor(
        true_plot_df[[TRUE_X_COL, TRUE_Y_COL]].to_numpy(dtype=np.float32),
        device=device,
        dtype=DTYPE,
    ).unsqueeze(0)
    pred_positions = torch.as_tensor(
        pred_plot_df[[PRED_X_COL, PRED_Y_COL]].to_numpy(dtype=np.float32),
        device=device,
        dtype=DTYPE,
    ).unsqueeze(0)
    node_features = torch.as_tensor(feature_matrix, device=device, dtype=DTYPE).unsqueeze(0)
    node_mask = torch.ones((1, n_cells), device=device, dtype=torch.bool)

    return (
        slice_df,
        true_plot_df,
        pred_plot_df,
        feature_cols,
        true_positions,
        pred_positions,
        node_features,
        node_mask,
    )


def make_holders(
    true_positions: torch.Tensor,
    pred_positions: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
):
    from utils.data.dataholder import DataHolder

    true_holder = DataHolder(
        positions=true_positions,
        node_features=node_features,
        diffusion_time=None,
        node_mask=node_mask,
    )
    pred_holder = DataHolder(
        positions=pred_positions,
        node_features=node_features,
        diffusion_time=None,
        node_mask=node_mask,
    )
    return pred_holder, true_holder


def _double_angle_to_spatial_axis(axis_double: torch.Tensor) -> torch.Tensor:
    """Map double-angle vectors ``Z=(cos 2phi, sin 2phi)*R`` to spatial axes.

    Returns a ``[..., 2]`` vector ``R * (cos phi, sin phi)`` whose direction is
    the (mod-pi) axis ``phi = 0.5 * atan2(sin 2phi, cos 2phi)`` and whose length
    is the orientation concentration ``R = ||Z||``. This is the representation
    we draw as an arrow in real space.
    """
    length = axis_double.norm(dim=-1)
    phi = 0.5 * torch.atan2(axis_double[..., 1], axis_double[..., 0])
    return torch.stack((length * torch.cos(phi), length * torch.sin(phi)), dim=-1)


def compute_loss_target_viz_fields(
    positions: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    neighbor_radius: float,
    coherence_radius: float,
    trans_beta: float,
    soft_beta: float | None,
    density_length_gate: bool = False,
    density_length_beta: float = 4.0,
    density_radius_gate: bool = False,
    density_radius_beta: float = 4.0,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Viz arrays for the exact ``n_target`` subsample used by the loss."""
    from train_directional_metric import (
        aggregate_axes_second_radius,
        compute_orientation_axes,
        _gather_positions,
        _neighborhood_counts,
        _density_length_scale,
        _density_radius_scale,
    )

    xy = positions[..., :2]
    target_pos = _gather_positions(xy, target_idx)

    # Stage 1: first-radius local orientation axis (double-angle resultant).
    if density_radius_gate:
        counts = _neighborhood_counts(
            xy, target_pos, mask, target_idx,
            neighbor_radius, soft_beta,
            include_self=False, eps=eps,
        )
        per_target_r = _density_radius_scale(
            counts, target_valid, base_radius=neighbor_radius,
            beta=density_radius_beta, eps=eps,
        )
        axis1_double, valid1 = compute_orientation_axes(
            xy,
            features,
            mask,
            target_idx,
            target_valid,
            neighbor_radius,
            trans_beta=trans_beta,
            soft_beta=soft_beta,
            per_target_radius=per_target_r,
            eps=eps,
        )
    else:
        axis1_double, valid1 = compute_orientation_axes(
            xy,
            features,
            mask,
            target_idx,
            target_valid,
            neighbor_radius,
            trans_beta=trans_beta,
            soft_beta=soft_beta,
            eps=eps,
        )

    if density_length_gate:
        counts = _neighborhood_counts(
            xy, target_pos, mask, target_idx,
            neighbor_radius, soft_beta,
            include_self=False, eps=eps,
        )
        scale = _density_length_scale(
            counts, valid1, beta=density_length_beta, eps=eps,
        )
        axis1_double = axis1_double * scale.unsqueeze(-1).to(axis1_double.dtype)

    # Stage 2: second-radius smoothing among target cells.
    axis2_double, valid2 = aggregate_axes_second_radius(
        target_pos,
        axis1_double,
        valid1,
        coherence_radius,
        soft_beta=soft_beta,
        eps=eps,
    )

    # Convert both to spatial axis vectors for plotting; the per-cell length is
    # the norm of the smoothed double-angle vector.
    axis1_spatial = _double_angle_to_spatial_axis(axis1_double)
    axis2_spatial = _double_angle_to_spatial_axis(axis2_double)
    length = axis2_double.norm(dim=-1)

    valid = valid2[0]
    idx = target_idx[0][valid]

    return {
        "cell_idx": idx.detach().cpu().numpy(),
        "x": xy[0, idx, 0].detach().cpu().numpy(),
        "y": xy[0, idx, 1].detach().cpu().numpy(),
        "pc1_x": axis1_spatial[0, valid, 0].detach().cpu().numpy(),
        "pc1_y": axis1_spatial[0, valid, 1].detach().cpu().numpy(),
        "avg_dir_x": axis2_spatial[0, valid, 0].detach().cpu().numpy(),
        "avg_dir_y": axis2_spatial[0, valid, 1].detach().cpu().numpy(),
        "coherence": length[0, valid].detach().cpu().numpy(),
    }


def _arrow_scale_for_unit_vectors(
    vec_x: np.ndarray,
    vec_y: np.ndarray,
    target_mean_length: float,
) -> float:
    lengths = np.hypot(vec_x, vec_y)
    mean_length = float(np.nanmean(lengths))
    if mean_length <= 0.0 or not np.isfinite(mean_length):
        return target_mean_length
    return target_mean_length / mean_length


def make_loss_arrow_plot_df(
    plot_df: pd.DataFrame,
    fields: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build arrow dataframe for loss subsample cells only."""
    cell_idx = fields["cell_idx"].astype(int)
    out = plot_df.iloc[cell_idx].copy().reset_index(drop=True)
    out["plot_x"] = fields["x"]
    out["plot_y"] = fields["y"]
    out["pc1_x"] = fields["pc1_x"]
    out["pc1_y"] = fields["pc1_y"]
    out["avg_dir_x"] = fields["avg_dir_x"]
    out["avg_dir_y"] = fields["avg_dir_y"]
    out["coherence"] = fields["coherence"]
    out["line_length"] = out["coherence"]
    out["_arrow_u"] = out["pc1_x"]
    out["_arrow_v"] = out["pc1_y"]
    return out


def _axis_double_angle_components(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit double-angle direction and smoothed-axis length from spatial axes."""
    length = np.hypot(u, v)
    safe = np.maximum(length, EPS)
    cos2 = (u * u - v * v) / (safe * safe)
    sin2 = (2.0 * u * v) / (safe * safe)
    return cos2, sin2, length


def weighted_circular_resultant(
    cos2: np.ndarray,
    sin2: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted mean resultant length in [0, 1] (1 = aligned, 0 = dispersed)."""
    weight_sum = float(weights.sum())
    if weight_sum <= EPS:
        return 0.0
    w = weights / weight_sum
    return float(np.hypot(np.sum(w * cos2), np.sum(w * sin2)))


def _set_ylim_from_values(ax: plt.Axes, values: np.ndarray, *, pad_fraction: float = 0.1) -> None:
    """Set y limits from data range with a small margin (not forced to [0, 1])."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    ymin = float(vals.min())
    ymax = float(vals.max())
    if ymax <= ymin:
        pad = max(abs(ymax), 1e-6) * pad_fraction
        ax.set_ylim(ymin - pad, ymax + pad)
        return
    pad = (ymax - ymin) * pad_fraction
    ax.set_ylim(ymin - pad, ymax + pad)


def slice_axis_alignment(arrow_df: pd.DataFrame) -> float:
    """Slice-level axis alignment from smoothed axes (coherence-weighted)."""
    cos2, sin2, length = _axis_double_angle_components(
        arrow_df["avg_dir_x"].to_numpy(dtype=np.float64),
        arrow_df["avg_dir_y"].to_numpy(dtype=np.float64),
    )
    return weighted_circular_resultant(cos2, sin2, length)


def collect_slice_analysis_records(
    slice_name: str,
    true_arrow_df: pd.DataFrame,
    pred_arrow_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Build per-cell and per-slice-per-cell-type analysis records for one slice."""
    per_cell: list[dict] = []
    per_slice: list[dict] = []

    for source, arrow_df in (("gt", true_arrow_df), ("pred", pred_arrow_df)):
        for cell_class, class_df in arrow_df.groupby("cell_class", sort=False):
            per_slice.append({
                "slice_name": slice_name,
                "source": source,
                "cell_class": str(cell_class),
                "mean_line_length": float(class_df["line_length"].mean()),
                "axis_alignment": slice_axis_alignment(class_df),
                "n_cells": int(len(class_df)),
            })
            for row in class_df.itertuples(index=False):
                per_cell.append({
                    "slice_name": slice_name,
                    "source": source,
                    "cell_class": str(cell_class),
                    "line_length": float(row.line_length),
                })

    return per_cell, per_slice


def _grouped_gt_pred_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    ylabel: str,
    title: str,
    error_col: str | None = None,
) -> None:
    """Grouped GT/pred bar chart indexed by cell type."""
    for source in ("gt", "pred"):
        if source not in summary.columns:
            summary[source] = np.nan
    summary = summary.sort_index()

    x = np.arange(len(summary))
    bar_width = 0.35
    gt_vals = summary["gt"].to_numpy(dtype=np.float64)
    pred_vals = summary["pred"].to_numpy(dtype=np.float64)
    gt_err = (
        summary[f"gt_{error_col}"].to_numpy(dtype=np.float64)
        if error_col and f"gt_{error_col}" in summary.columns
        else None
    )
    pred_err = (
        summary[f"pred_{error_col}"].to_numpy(dtype=np.float64)
        if error_col and f"pred_{error_col}" in summary.columns
        else None
    )

    ax.bar(
        x - bar_width / 2,
        gt_vals,
        width=bar_width,
        yerr=gt_err,
        capsize=4 if gt_err is not None else 0,
        label="Ground truth",
        color="tab:blue",
        alpha=0.85,
    )
    ax.bar(
        x + bar_width / 2,
        pred_vals,
        width=bar_width,
        yerr=pred_err,
        capsize=4 if pred_err is not None else 0,
        label="Prediction",
        color="tab:orange",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    ymax_candidates = [gt_vals, pred_vals]
    if gt_err is not None:
        ymax_candidates.append(gt_vals + np.nan_to_num(gt_err))
    if pred_err is not None:
        ymax_candidates.append(pred_vals + np.nan_to_num(pred_err))
    _set_ylim_from_values(ax, np.concatenate(ymax_candidates))


def save_analysis_png(
    per_cell_df: pd.DataFrame,
    per_slice_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Dataset-level analysis: line length and axis alignment by cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    length_by_class = (
        per_cell_df.groupby(["cell_class", "source"], as_index=False)["line_length"]
        .mean()
        .pivot(index="cell_class", columns="source", values="line_length")
    )
    _grouped_gt_pred_bars(
        axes[0],
        length_by_class,
        ylabel="Mean smoothed axis length",
        title="Mean line length by cell type\n(averaged across all slices)",
    )

    align_stats = (
        per_slice_df.groupby(["cell_class", "source"])["axis_alignment"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    align_stats["sem"] = align_stats["std"] / np.sqrt(align_stats["count"].clip(lower=1))
    align_mean = align_stats.pivot(index="cell_class", columns="source", values="mean")
    align_sem = align_stats.pivot(index="cell_class", columns="source", values="sem")
    align_summary = align_mean.copy()
    for source in ("gt", "pred"):
        sem_col = f"{source}_sem"
        if source in align_sem.columns:
            align_summary[sem_col] = align_sem[source]
        else:
            align_summary[sem_col] = np.nan
    _grouped_gt_pred_bars(
        axes[1],
        align_summary,
        ylabel="Axis alignment",
        title=(
            "Axis alignment by cell type\n"
            "per-slice weighted circular resultant, mean ± SEM across slices"
        ),
        error_col="sem",
    )

    n_slices = per_slice_df["slice_name"].nunique()
    fig.suptitle(
        f"Directional analysis — {n_slices} slices — "
        f"smoothed axis (r={COHERENCE_RADIUS:.3g})",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def draw_class_axis_bars(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    u_col: str,
    v_col: str,
    scale: float,
    palette_dict: dict[str, tuple],
    uniques: list[str],
    linewidth: float = 1.2,
) -> None:
    """Draw short axis line segments centered on each cell (no arrow heads)."""
    from matplotlib.collections import LineCollection

    for cell_class in uniques:
        sub = plot_df[plot_df["cell_class"] == cell_class]
        if sub.empty:
            continue
        u = sub[u_col].to_numpy(dtype=np.float64) * scale
        v = sub[v_col].to_numpy(dtype=np.float64) * scale
        x = sub["plot_x"].to_numpy(dtype=np.float64)
        y = sub["plot_y"].to_numpy(dtype=np.float64)
        segments = np.stack(
            [
                np.stack([x - 0.5 * u, y - 0.5 * v], axis=-1),
                np.stack([x + 0.5 * u, y + 0.5 * v], axis=-1),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=palette_dict[cell_class],
                linewidths=linewidth,
                zorder=2,
            )
        )


def draw_class_scatter(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    palette_dict: dict[str, tuple],
    marker_size: float = TISSUE_SCATTER_SIZE,
) -> None:
    """Scatter plot of cell positions colored by class."""
    sns.scatterplot(
        data=plot_df,
        x="plot_x",
        y="plot_y",
        hue="cell_class",
        s=marker_size,
        ax=ax,
        palette=palette_dict,
        legend=False,
        linewidth=0,
    )


def evaluate_slice(
    csv_path: Path,
    slice_name: str,
    device: torch.device,
    loss_fn,
    *,
    compute_viz: bool,
    warmup_iters: int = 0,
) -> tuple[SliceResult, dict | None]:
    (
        _slice_df,
        true_plot_df,
        pred_plot_df,
        feature_cols,
        true_positions,
        pred_positions,
        node_features,
        node_mask,
    ) = load_slice_tensors(csv_path, slice_name, device)

    pred_holder, true_holder = make_holders(
        true_positions, pred_positions, node_features, node_mask,
    )

    torch.manual_seed(LOSS_SEED)

    def _forward():
        return loss_fn(
            masked_pred=pred_holder,
            masked_true=true_holder,
            train_stage=True,
            log=False,
        )

    for _ in range(max(int(warmup_iters), 0)):
        _forward()
    _sync_device(device)

    (loss, _), metric_seconds = _timed_call(device, _forward)
    loss_total = float(loss.detach().item())
    loss_length = float(loss_fn._last_length)
    loss_pairwise = float(loss_fn._last_pairwise)

    viz_out = None
    viz_seconds = 0.0
    if compute_viz:
        from train_directional_metric import sample_target_indices

        def _viz():
            torch.manual_seed(LOSS_SEED)
            target_idx, target_valid = sample_target_indices(
                node_mask, loss_fn.n_target,
            )
            true_fields = compute_loss_target_viz_fields(
                true_positions,
                node_features,
                node_mask,
                target_idx,
                target_valid,
                neighbor_radius=NEIGHBOR_RADIUS,
                coherence_radius=COHERENCE_RADIUS,
                trans_beta=TRANS_BETA,
                soft_beta=SOFT_BETA,
                density_length_gate=loss_fn.density_length_gate,
                density_length_beta=loss_fn.density_length_beta,
                density_radius_gate=loss_fn.density_radius_gate,
                density_radius_beta=loss_fn.density_radius_beta,
            )
            pred_fields = compute_loss_target_viz_fields(
                pred_positions,
                node_features,
                node_mask,
                target_idx,
                target_valid,
                neighbor_radius=NEIGHBOR_RADIUS,
                coherence_radius=COHERENCE_RADIUS,
                trans_beta=TRANS_BETA,
                soft_beta=SOFT_BETA,
                density_length_gate=loss_fn.density_length_gate,
                density_length_beta=loss_fn.density_length_beta,
                density_radius_gate=loss_fn.density_radius_gate,
                density_radius_beta=loss_fn.density_radius_beta,
            )
            return target_idx, true_fields, pred_fields

        viz_out, viz_seconds = _timed_call(device, _viz)

    result = SliceResult(
        slice_name=slice_name,
        n_cells=true_positions.shape[1],
        n_features=len(feature_cols),
        loss_total=loss_total,
        loss_length=loss_length,
        loss_pairwise=loss_pairwise,
        metric_seconds=metric_seconds,
        viz_seconds=viz_seconds,
    )

    if viz_out is None:
        return result, None

    target_idx, true_fields, pred_fields = viz_out
    n_viz = int(true_fields["x"].shape[0])
    return result, {
        "n_target_viz": n_viz,
        "true_tissue_df": true_plot_df[[TRUE_X_COL, TRUE_Y_COL, "cell_class"]].rename(
            columns={TRUE_X_COL: "plot_x", TRUE_Y_COL: "plot_y"},
        ),
        "pred_tissue_df": pred_plot_df[[PRED_X_COL, PRED_Y_COL, "cell_class"]].rename(
            columns={PRED_X_COL: "plot_x", PRED_Y_COL: "plot_y"},
        ),
        "true_arrow_df": make_loss_arrow_plot_df(
            true_plot_df[[TRUE_X_COL, TRUE_Y_COL, "cell_class"]],
            true_fields,
        ),
        "pred_arrow_df": make_loss_arrow_plot_df(
            pred_plot_df[[PRED_X_COL, PRED_Y_COL, "cell_class"]],
            pred_fields,
        ),
        "loss_total": loss_total,
        "loss_length": loss_length,
        "loss_pairwise": loss_pairwise,
    }


def save_summary_png(results: list[SliceResult], output_path: Path, device: torch.device) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values("loss_total", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["loss_total"], width=width, label="total", color="tab:blue")
    ax.bar(x, df["loss_length"], width=width, label="length", color="tab:orange")
    ax.bar(x + width, df["loss_pairwise"], width=width, label="pairwise", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(df["slice_name"], rotation=45, ha="right")
    ax.set_ylabel("DirectionalMetricLoss")
    ax.set_title("Loss components per slice")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(df["slice_name"], df["metric_seconds"], color="tab:purple", alpha=0.85)
    ax.set_xticklabels(df["slice_name"], rotation=45, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title(f"Loss forward time ({device.type}) per slice")

    ax = axes[1, 0]
    ax.bar(df["slice_name"], df["n_cells"], color="tab:gray", alpha=0.85)
    ax.set_xticklabels(df["slice_name"], rotation=45, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Cells per slice")

    ax = axes[1, 1]
    ax.axis("off")
    table_df = df[
        ["slice_name", "n_cells", "loss_total", "loss_length", "loss_pairwise", "metric_seconds"]
    ].copy()
    table_df["metric_seconds"] = table_df["metric_seconds"].map(lambda v: f"{v:.4f}")
    for col in ("loss_total", "loss_length", "loss_pairwise"):
        table_df[col] = table_df[col].map(lambda v: f"{v:.6f}")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    ax.set_title("Per-slice summary", pad=20)

    fig.suptitle(
        f"Directional metric report — {len(df)} slices — device={device}",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_slice_arrow_png(
    slice_name: str,
    viz_data: dict,
    output_path: Path,
    *,
    left_panel_tissue_dots: bool = False,
) -> None:
    true_arrow_df = viz_data["true_arrow_df"]
    pred_arrow_df = viz_data["pred_arrow_df"]
    true_tissue_df = viz_data["true_tissue_df"]
    pred_tissue_df = viz_data["pred_tissue_df"]
    n_target_viz = viz_data["n_target_viz"]

    class_source = pd.concat(
        [true_tissue_df["cell_class"], pred_tissue_df["cell_class"]],
        ignore_index=True,
    ) if left_panel_tissue_dots else true_arrow_df["cell_class"]
    uniques = sorted(class_source.unique())
    palette_dict = dict(zip(uniques, sns.color_palette(cc.glasbey, n_colors=len(uniques))))

    axis1_arrow_scale = _arrow_scale_for_unit_vectors(
        true_arrow_df["pc1_x"].to_numpy(),
        true_arrow_df["pc1_y"].to_numpy(),
        TARGET_MEAN_ARROW_LENGTH,
    )
    axis2_arrow_scale = _arrow_scale_for_unit_vectors(
        true_arrow_df["avg_dir_x"].to_numpy(),
        true_arrow_df["avg_dir_y"].to_numpy(),
        TARGET_MEAN_ARROW_LENGTH,
    )

    xlim = (
        min(true_tissue_df["plot_x"].min(), pred_tissue_df["plot_x"].min()),
        max(true_tissue_df["plot_x"].max(), pred_tissue_df["plot_x"].max()),
    )
    ylim = (
        min(true_tissue_df["plot_y"].min(), pred_tissue_df["plot_y"].min()),
        max(true_tissue_df["plot_y"].max(), pred_tissue_df["plot_y"].max()),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    row_specs = [
        (true_tissue_df, true_arrow_df, "Ground truth"),
        (pred_tissue_df, pred_arrow_df, "Prediction"),
    ]

    for row_idx, (tissue_df, arrow_df, row_label) in enumerate(row_specs):
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")

            if left_panel_tissue_dots and col_idx == 0:
                ax.set_title(f"{row_label} — Tissue layout", fontsize=13)
                draw_class_scatter(ax, tissue_df, palette_dict=palette_dict)
            elif col_idx == 0:
                ax.set_title(
                    f"{row_label} — Local axis (r={NEIGHBOR_RADIUS:.3g})",
                    fontsize=13,
                )
                draw_class_axis_bars(
                    ax,
                    arrow_df,
                    u_col="_arrow_u",
                    v_col="_arrow_v",
                    scale=axis1_arrow_scale,
                    palette_dict=palette_dict,
                    uniques=uniques,
                )
            else:
                ax.set_title(
                    f"{row_label} — Smoothed axis (r={COHERENCE_RADIUS:.3g})",
                    fontsize=13,
                )
                draw_class_axis_bars(
                    ax,
                    arrow_df,
                    u_col="avg_dir_x",
                    v_col="avg_dir_y",
                    scale=axis2_arrow_scale,
                    palette_dict=palette_dict,
                    uniques=uniques,
                )

    loss_text = (
        f"total={viz_data['loss_total']:.4f}, "
        f"length={viz_data['loss_length']:.4f}, "
        f"pairwise={viz_data['loss_pairwise']:.4f}"
    )
    fig.suptitle(
        f"{slice_name} — loss subsample n={n_target_viz} — {loss_text}",
        fontsize=14,
        y=1.01,
    )

    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=cat,
            markerfacecolor=palette_dict[cat], markersize=7,
        )
        for cat in uniques
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=max(1, len(uniques) // 6),
        bbox_to_anchor=(0.5, -0.02),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DirectionalMetricLoss on LiVAE comparison CSV slices.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to not_normalized_test_results.csv",
    )
    parser.add_argument(
        "--livae-root",
        type=Path,
        default=DEFAULT_LIVAE_ROOT,
        help="LiVAE_tests root (for build_plot_with_pred imports)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "directional_loss_summary.png",
        help="Summary PNG path (per-slice arrow PNGs use the same stem)",
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=None,
        help="Slice names to evaluate (default: all slices in CSV)",
    )
    parser.add_argument(
        "--viz-slices",
        nargs="*",
        default=None,
        help="Slices for arrow PNGs (default: same as --slices)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda:0 or cpu (default: cuda:0 if available else cpu)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require CUDA (cuda:0). Exits with an error if no GPU is visible to PyTorch.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Loss forward passes before timing (default: 2 on GPU, 0 on CPU)",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=N_TARGET,
        help="Target cells subsampled by DirectionalMetricLoss (loss + viz use the same subsample)",
    )
    parser.add_argument(
        "--viz-tissue-dots",
        action="store_true",
        help=(
            "Replace left-column panels (local axis) with full-slice tissue dot plots; "
            "right-column panels still show smoothed axis bars on the loss subsample"
        ),
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "Compute viz fields on all evaluated slices and write dataset analysis PNG "
            "(mean line length by cell type; slice axis alignment GT vs pred)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_import_paths(args.livae_root)

    from train_directional_metric import DirectionalMetricLoss

    if not args.csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {args.csv}")

    device = resolve_device(args.device, use_gpu=args.gpu)
    warmup_iters = (
        args.warmup
        if args.warmup is not None
        else (2 if device.type == "cuda" else 0)
    )

    slice_names = args.slices or list_slices(args.csv)
    if not slice_names:
        raise ValueError(f"No slices found in {args.csv}")

    viz_slices = set(args.viz_slices if args.viz_slices is not None else slice_names)

    loss_fn = DirectionalMetricLoss(
        n_target=args.n_target,
        neighbor_radius=NEIGHBOR_RADIUS,
        coherence_radius=COHERENCE_RADIUS,
        trans_beta=TRANS_BETA,
        soft_beta=SOFT_BETA,
        eps=EPS,
        length_weight=1.0,
        pairwise_weight=1.0,
        density_length_gate=DENSITY_LENGTH_GATE,
        density_length_beta=DENSITY_LENGTH_BETA,
        density_radius_gate=DENSITY_RADIUS_GATE,
        density_radius_beta=DENSITY_RADIUS_BETA,
    ).to(device)

    print(f"CSV: {args.csv}")
    print(f"Device: {describe_device(device)}")
    print(f"Positions tensor device: {device} (checked per slice after load)")
    print(f"Timing warmup iterations: {warmup_iters}")
    print(f"Graph batch size: 1 slice per forward pass")
    print(f"Loss + viz n_target: {args.n_target} (seed={LOSS_SEED})")
    print(f"Neighbor radius: {NEIGHBOR_RADIUS}  Coherence radius: {COHERENCE_RADIUS}  trans_beta: {TRANS_BETA}")
    print(
        f"Density length gate: {DENSITY_LENGTH_GATE}  "
        f"density_length_beta: {DENSITY_LENGTH_BETA}"
    )
    print(
        f"Density radius gate: {DENSITY_RADIUS_GATE}  "
        f"density_radius_beta: {DENSITY_RADIUS_BETA}"
    )
    print(f"Slices to evaluate: {len(slice_names)}")
    print(f"Slices with arrow PNGs: {len(viz_slices)}")
    print(f"Left panel tissue dots: {args.viz_tissue_dots}")
    print(f"Dataset analysis: {args.analysis}")

    results: list[SliceResult] = []
    analysis_cell_records: list[dict] = []
    analysis_slice_records: list[dict] = []
    output_stem = args.output.with_suffix("")

    for slice_name in slice_names:
        print(f"\n=== {slice_name} ===")
        need_viz = slice_name in viz_slices or args.analysis
        result, viz_data = evaluate_slice(
            args.csv,
            slice_name,
            device,
            loss_fn,
            compute_viz=need_viz,
            warmup_iters=warmup_iters,
        )
        results.append(result)

        print(f"Cells: {result.n_cells:,} | Features: {result.n_features}")
        print(f"Tensor device: {device}")
        print(
            f"DirectionalMetricLoss: {result.loss_total:.6f} "
            f"(length={result.loss_length:.6f}, "
            f"pairwise={result.loss_pairwise:.6f})"
        )
        print(f"Loss forward time ({device.type}): {result.metric_seconds:.4f} s")
        if result.viz_seconds > 0.0:
            print(f"Viz field time: {result.viz_seconds:.4f} s")

        if viz_data is not None and args.analysis:
            cell_records, slice_records = collect_slice_analysis_records(
                slice_name,
                viz_data["true_arrow_df"],
                viz_data["pred_arrow_df"],
            )
            analysis_cell_records.extend(cell_records)
            analysis_slice_records.extend(slice_records)

        if viz_data is not None and slice_name in viz_slices:
            arrow_path = Path(f"{output_stem}_{slice_name}_arrows.png")
            save_slice_arrow_png(
                slice_name,
                viz_data,
                arrow_path,
                left_panel_tissue_dots=args.viz_tissue_dots,
            )
            print(f"Saved arrows: {arrow_path}")

        del viz_data

    summary_path = args.output
    save_summary_png(results, summary_path, device)
    print(f"\nSaved summary: {summary_path}")

    if args.analysis:
        if not analysis_cell_records:
            print("Analysis requested but no viz data was collected.")
        else:
            per_cell_df = pd.DataFrame(analysis_cell_records)
            per_slice_df = pd.DataFrame(analysis_slice_records)
            analysis_path = Path(f"{output_stem}_analysis.png")
            save_analysis_png(per_cell_df, per_slice_df, analysis_path)
            print(f"Saved analysis: {analysis_path}")


if __name__ == "__main__":
    main()
