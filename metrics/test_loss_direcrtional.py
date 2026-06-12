"""Evaluate and visualize DirectionalMetricLoss on LiVAE comparison CSV slices.

Loads the same input format as ``LiVAE_tests/arrow_plot/build_plot_with_pred.py``
(``not_normalized_test_results.csv``: gene columns, ``coord_X``/``coord_Y``,
``coord_X_test``/``coord_Y_test``, ``cell_section``, ``cell_class``).

For each slice the script:
  1. Builds batched tensors on GPU/CPU.
  2. Times the broadcast directional-metric forward pass.
  3. Optionally renders PCA / coherence arrow panels.
  4. Writes a summary PNG (loss + timing per slice) and per-slice arrow PNGs.

Usage::

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
DEFAULT_LIVAE_ROOT = Path("/home/asalmona/Documents/Ricci/code/LiVAE_tests")
DEFAULT_CSV = DEFAULT_LIVAE_ROOT / "data_test_observations" / "not_normalized_test_results.csv"

TRUE_X_COL = "coord_X"
TRUE_Y_COL = "coord_Y"
PRED_X_COL = "coord_X_test"
PRED_Y_COL = "coord_Y_test"
SECTION_COL = "cell_section"

N_TARGET = 128
PCA_RADIUS = 0.16
COHERENCE_RADIUS = 0.08
SOFT_BETA = 256.0
EPS = 1e-6
TARGET_MEAN_ARROW_LENGTH = 0.03
LOSS_SEED = 0
DTYPE = torch.float32


@dataclass
class SliceResult:
    slice_name: str
    n_cells: int
    n_features: int
    loss_total: float
    loss_coherence: float
    loss_pairwise: float
    metric_seconds: float
    viz_seconds: float


def _setup_import_paths(livae_root: Path) -> None:
    for path in (REPO_ROOT, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    arrow_plot_dir = livae_root / "arrow_plot"
    if str(arrow_plot_dir) not in sys.path:
        sys.path.insert(0, str(arrow_plot_dir))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def compute_directional_viz_fields(
    positions: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    pca_radius: float,
    coherence_radius: float,
    soft_beta: float | None,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    from directional_metric import (
        _gather_positions,
        _spatial_soft_weights,
        _transcriptome_distance,
        compute_weighted_pca_directions,
    )

    n_cells = positions.shape[1]
    target_idx = torch.arange(n_cells, device=positions.device).unsqueeze(0)
    target_valid = mask.clone()

    unit_dirs, pca_valid = compute_weighted_pca_directions(
        positions,
        features,
        mask,
        target_idx,
        target_valid,
        pca_radius,
        soft_beta=soft_beta,
        eps=eps,
    )

    target_pos = _gather_positions(positions, target_idx)
    spatial_dists = torch.cdist(target_pos, positions, p=2)
    spatial_w = _spatial_soft_weights(spatial_dists, pca_radius, soft_beta)
    trans_dist = _transcriptome_distance(features, target_idx, eps=eps)
    weights = spatial_w / (trans_dist + eps)
    weights = weights * mask.to(weights.dtype).unsqueeze(1)
    w_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)

    pos = positions.unsqueeze(1)
    mu = (weights.unsqueeze(-1) * pos).sum(dim=2) / w_sum
    dx = pos - mu.unsqueeze(2)
    dx_x = dx[..., 0]
    dx_y = dx[..., 1]
    cxx = (weights * dx_x * dx_x).sum(dim=-1) / w_sum.squeeze(-1)
    cyy = (weights * dx_y * dx_y).sum(dim=-1) / w_sum.squeeze(-1)
    cxy = (weights * dx_x * dx_y).sum(dim=-1) / w_sum.squeeze(-1)
    spread = (cxx + cyy).clamp_min(0.0)
    disc = torch.sqrt(torch.clamp((cxx - cyy).pow(2) + 4.0 * cxy.pow(2), min=0.0))
    lam1 = 0.5 * (spread + disc)
    lam2 = 0.5 * (spread - disc)
    anisotropy = ((lam1 - lam2) / (lam1 + lam2 + eps)).clamp(0.0, 1.0)

    dists = torch.cdist(target_pos, target_pos, p=2)
    coh_w = _spatial_soft_weights(dists, coherence_radius, soft_beta)
    valid_pair = pca_valid.unsqueeze(2) & pca_valid.unsqueeze(1)
    coh_w = coh_w * valid_pair.to(coh_w.dtype)
    denom = coh_w.sum(dim=-1, keepdim=True).clamp_min(eps)
    avg_vec = torch.matmul(coh_w, unit_dirs) / denom
    coherence = avg_vec.norm(dim=-1).clamp(0.0, 1.0)

    valid = pca_valid[0].detach().cpu().numpy()
    return {
        "x": positions[0, :, 0].detach().cpu().numpy(),
        "y": positions[0, :, 1].detach().cpu().numpy(),
        "pc1_x": unit_dirs[0, :, 0].detach().cpu().numpy(),
        "pc1_y": unit_dirs[0, :, 1].detach().cpu().numpy(),
        "anisotropy": anisotropy[0].detach().cpu().numpy(),
        "avg_dir_x": avg_vec[0, :, 0].detach().cpu().numpy(),
        "avg_dir_y": avg_vec[0, :, 1].detach().cpu().numpy(),
        "coherence": coherence[0].detach().cpu().numpy(),
        "valid": valid,
    }


def _arrow_scale_for_mean_length(anisotropy: np.ndarray, target_mean_length: float) -> float:
    mean_anisotropy = float(np.nanmean(anisotropy))
    if mean_anisotropy <= 0.0 or not np.isfinite(mean_anisotropy):
        return target_mean_length
    return target_mean_length / mean_anisotropy


def _arrow_scale_for_vector_length(
    vec_x: np.ndarray,
    vec_y: np.ndarray,
    target_mean_length: float,
) -> float:
    lengths = np.hypot(vec_x, vec_y)
    mean_length = float(np.nanmean(lengths))
    if mean_length <= 0.0 or not np.isfinite(mean_length):
        return target_mean_length
    return target_mean_length / mean_length


def make_arrow_plot_df(plot_df: pd.DataFrame, fields: dict[str, np.ndarray]) -> pd.DataFrame:
    out = plot_df.copy()
    for key, values in fields.items():
        out[key] = values
    out = out.loc[fields["valid"]].copy()
    out["plot_x"] = out["x"]
    out["plot_y"] = out["y"]
    out["_arrow_u"] = out["pc1_x"] * out["anisotropy"]
    out["_arrow_v"] = out["pc1_y"] * out["anisotropy"]
    return out


def draw_class_arrows(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    u_col: str,
    v_col: str,
    scale: float,
    palette_dict: dict[str, tuple],
    uniques: list[str],
) -> None:
    for cell_class in uniques:
        sub = plot_df[plot_df["cell_class"] == cell_class]
        if sub.empty:
            continue
        ax.quiver(
            sub["plot_x"],
            sub["plot_y"],
            sub[u_col].to_numpy() * scale,
            sub[v_col].to_numpy() * scale,
            color=palette_dict[cell_class],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            headwidth=2.8,
            headlength=4.2,
            headaxislength=3.4,
            minshaft=2.0,
            zorder=2,
        )


def evaluate_slice(
    csv_path: Path,
    slice_name: str,
    device: torch.device,
    loss_fn,
    *,
    compute_viz: bool,
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

    (loss, _), metric_seconds = _timed_call(device, _forward)
    loss_total = float(loss.detach().item())
    loss_coherence = float(loss_fn._last_coherence)
    loss_pairwise = float(loss_fn._last_pairwise)

    viz_bundle = None
    viz_seconds = 0.0
    if compute_viz:
        def _viz():
            true_fields = compute_directional_viz_fields(
                true_positions,
                node_features,
                node_mask,
                pca_radius=PCA_RADIUS,
                coherence_radius=COHERENCE_RADIUS,
                soft_beta=SOFT_BETA,
            )
            pred_fields = compute_directional_viz_fields(
                pred_positions,
                node_features,
                node_mask,
                pca_radius=PCA_RADIUS,
                coherence_radius=COHERENCE_RADIUS,
                soft_beta=SOFT_BETA,
            )
            return true_plot_df, pred_plot_df, true_fields, pred_fields

        viz_bundle, viz_seconds = _timed_call(device, _viz)

    result = SliceResult(
        slice_name=slice_name,
        n_cells=true_positions.shape[1],
        n_features=len(feature_cols),
        loss_total=loss_total,
        loss_coherence=loss_coherence,
        loss_pairwise=loss_pairwise,
        metric_seconds=metric_seconds,
        viz_seconds=viz_seconds,
    )

    if viz_bundle is None:
        return result, None

    true_plot_df, pred_plot_df, true_fields, pred_fields = viz_bundle
    return result, {
        "true_plot_df": true_plot_df,
        "pred_plot_df": pred_plot_df,
        "true_arrow_df": make_arrow_plot_df(
            true_plot_df[[TRUE_X_COL, TRUE_Y_COL, "cell_class"]],
            true_fields,
        ),
        "pred_arrow_df": make_arrow_plot_df(
            pred_plot_df[[PRED_X_COL, PRED_Y_COL, "cell_class"]],
            pred_fields,
        ),
        "loss_total": loss_total,
        "loss_coherence": loss_coherence,
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
    ax.bar(x, df["loss_coherence"], width=width, label="coherence", color="tab:orange")
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
    ax.set_title("Broadcast directional-metric time per sample")

    ax = axes[1, 0]
    ax.bar(df["slice_name"], df["n_cells"], color="tab:gray", alpha=0.85)
    ax.set_xticklabels(df["slice_name"], rotation=45, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Cells per slice")

    ax = axes[1, 1]
    ax.axis("off")
    table_df = df[
        ["slice_name", "n_cells", "loss_total", "loss_coherence", "loss_pairwise", "metric_seconds"]
    ].copy()
    table_df["metric_seconds"] = table_df["metric_seconds"].map(lambda v: f"{v:.4f}")
    for col in ("loss_total", "loss_coherence", "loss_pairwise"):
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
) -> None:
    true_layout_df = viz_data["true_plot_df"]
    pred_layout_df = viz_data["pred_plot_df"]
    true_arrow_df = viz_data["true_arrow_df"]
    pred_arrow_df = viz_data["pred_arrow_df"]

    uniques = sorted(true_arrow_df["cell_class"].unique())
    palette_dict = dict(zip(uniques, sns.color_palette(cc.glasbey, n_colors=len(uniques))))

    pca_arrow_scale = _arrow_scale_for_mean_length(
        true_arrow_df["anisotropy"].to_numpy(),
        TARGET_MEAN_ARROW_LENGTH,
    )
    dir_arrow_scale = _arrow_scale_for_vector_length(
        true_arrow_df["avg_dir_x"].to_numpy(),
        true_arrow_df["avg_dir_y"].to_numpy(),
        TARGET_MEAN_ARROW_LENGTH,
    )

    xlim = (true_layout_df[TRUE_X_COL].min(), true_layout_df[TRUE_X_COL].max())
    ylim = (true_layout_df[TRUE_Y_COL].min(), true_layout_df[TRUE_Y_COL].max())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_specs = [
        (true_layout_df, true_arrow_df, "Ground truth", TRUE_X_COL, TRUE_Y_COL),
        (pred_layout_df, pred_arrow_df, "Prediction", PRED_X_COL, PRED_Y_COL),
    ]

    for row_idx, (layout_df, arrow_df, row_label, x_col, y_col) in enumerate(panel_specs):
        ax_scatter, ax_pca, ax_coh = axes[row_idx]

        for ax, title in (
            (ax_scatter, f"{row_label} — tissue layout"),
            (ax_pca, f"Per-cell PCA (r={PCA_RADIUS:.3g}) — {row_label.lower()}"),
            (ax_coh, f"Neighborhood mean (r={COHERENCE_RADIUS:.3g}) — {row_label.lower()}"),
        ):
            ax.set_title(title, fontsize=13)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")

        sns.scatterplot(
            data=layout_df,
            x=x_col,
            y=y_col,
            hue="cell_class",
            s=8,
            ax=ax_scatter,
            palette=palette_dict,
            legend=False,
        )
        draw_class_arrows(
            ax_pca,
            arrow_df,
            u_col="_arrow_u",
            v_col="_arrow_v",
            scale=pca_arrow_scale,
            palette_dict=palette_dict,
            uniques=uniques,
        )
        draw_class_arrows(
            ax_coh,
            arrow_df,
            u_col="avg_dir_x",
            v_col="avg_dir_y",
            scale=dir_arrow_scale,
            palette_dict=palette_dict,
            uniques=uniques,
        )

    loss_text = (
        f"total={viz_data['loss_total']:.4f}, "
        f"coherence={viz_data['loss_coherence']:.4f}, "
        f"pairwise={viz_data['loss_pairwise']:.4f}"
    )
    fig.suptitle(f"{slice_name} — {loss_text}", fontsize=15, y=1.01)

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
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=N_TARGET,
        help="Target cells subsampled per forward pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_import_paths(args.livae_root)

    from directional_metric import DirectionalMetricLoss

    if not args.csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {args.csv}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    slice_names = args.slices or list_slices(args.csv)
    if not slice_names:
        raise ValueError(f"No slices found in {args.csv}")

    viz_slices = set(args.viz_slices if args.viz_slices is not None else slice_names)

    loss_fn = DirectionalMetricLoss(
        n_target=args.n_target,
        pca_radius=PCA_RADIUS,
        coherence_radius=COHERENCE_RADIUS,
        soft_beta=SOFT_BETA,
        eps=EPS,
        coherence_weight=1.0,
        pairwise_weight=1.0,
    ).to(device)

    print(f"CSV: {args.csv}")
    print(f"Device: {device}")
    print(f"Slices to evaluate: {len(slice_names)}")
    print(f"Slices with arrow PNGs: {len(viz_slices)}")

    results: list[SliceResult] = []
    output_stem = args.output.with_suffix("")

    for slice_name in slice_names:
        print(f"\n=== {slice_name} ===")
        result, viz_data = evaluate_slice(
            args.csv,
            slice_name,
            device,
            loss_fn,
            compute_viz=slice_name in viz_slices,
        )
        results.append(result)

        print(f"Cells: {result.n_cells:,} | Features: {result.n_features}")
        print(
            f"DirectionalMetricLoss: {result.loss_total:.6f} "
            f"(coherence={result.loss_coherence:.6f}, "
            f"pairwise={result.loss_pairwise:.6f})"
        )
        print(f"Broadcast metric time: {result.metric_seconds:.4f} s")
        if result.viz_seconds > 0.0:
            print(f"Viz field time: {result.viz_seconds:.4f} s")

        if viz_data is not None:
            arrow_path = Path(f"{output_stem}_{slice_name}_arrows.png")
            save_slice_arrow_png(slice_name, viz_data, arrow_path)
            print(f"Saved arrows: {arrow_path}")

        del viz_data

    summary_path = args.output
    save_summary_png(results, summary_path, device)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
