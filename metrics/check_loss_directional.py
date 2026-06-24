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
NEIGHBOR_RADIUS = 0.12
COHERENCE_RADIUS = 0.1
TRANS_BETA = 32.0
SOFT_BETA = 256.0
EPS = 1e-6
TARGET_MEAN_ARROW_LENGTH = 0.04
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
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Viz arrays for the exact ``n_target`` subsample used by the loss."""
    from train_directional_metric import (
        aggregate_axes_second_radius,
        compute_orientation_axes,
        _gather_positions,
    )

    xy = positions[..., :2]

    # Stage 1: first-radius local orientation axis (double-angle resultant).
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

    # Stage 2: second-radius smoothing among target cells.
    target_pos = _gather_positions(xy, target_idx)
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
    out["_arrow_u"] = out["pc1_x"]
    out["_arrow_v"] = out["pc1_y"]
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
) -> None:
    true_arrow_df = viz_data["true_arrow_df"]
    pred_arrow_df = viz_data["pred_arrow_df"]
    n_target_viz = viz_data["n_target_viz"]

    uniques = sorted(true_arrow_df["cell_class"].unique())
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
        min(true_arrow_df["plot_x"].min(), pred_arrow_df["plot_x"].min()),
        max(true_arrow_df["plot_x"].max(), pred_arrow_df["plot_x"].max()),
    )
    ylim = (
        min(true_arrow_df["plot_y"].min(), pred_arrow_df["plot_y"].min()),
        max(true_arrow_df["plot_y"].max(), pred_arrow_df["plot_y"].max()),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_specs = [
        (true_arrow_df, "Ground truth", f"Local axis (r={NEIGHBOR_RADIUS:.3g})"),
        (true_arrow_df, "Ground truth", f"Smoothed axis (r={COHERENCE_RADIUS:.3g})"),
        (pred_arrow_df, "Prediction", f"Local axis (r={NEIGHBOR_RADIUS:.3g})"),
        (pred_arrow_df, "Prediction", f"Smoothed axis (r={COHERENCE_RADIUS:.3g})"),
    ]
    arrow_specs = [
        ("_arrow_u", "_arrow_v", axis1_arrow_scale),
        ("avg_dir_x", "avg_dir_y", axis2_arrow_scale),
        ("_arrow_u", "_arrow_v", axis1_arrow_scale),
        ("avg_dir_x", "avg_dir_y", axis2_arrow_scale),
    ]

    for ax, (arrow_df, row_label, panel_title), (u_col, v_col, scale) in zip(
        axes.flat, panel_specs, arrow_specs,
    ):
        ax.set_title(f"{row_label} — {panel_title}", fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        sns.scatterplot(
            data=arrow_df,
            x="plot_x",
            y="plot_y",
            hue="cell_class",
            s=6,
            ax=ax,
            palette=palette_dict,
            legend=False,
        )
        draw_class_arrows(
            ax,
            arrow_df,
            u_col=u_col,
            v_col=v_col,
            scale=scale,
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
    ).to(device)

    print(f"CSV: {args.csv}")
    print(f"Device: {describe_device(device)}")
    print(f"Positions tensor device: {device} (checked per slice after load)")
    print(f"Timing warmup iterations: {warmup_iters}")
    print(f"Graph batch size: 1 slice per forward pass")
    print(f"Loss + viz n_target: {args.n_target} (seed={LOSS_SEED})")
    print(f"Neighbor radius: {NEIGHBOR_RADIUS}  Coherence radius: {COHERENCE_RADIUS}  trans_beta: {TRANS_BETA}")
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
