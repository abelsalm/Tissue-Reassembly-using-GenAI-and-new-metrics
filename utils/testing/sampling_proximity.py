"""Score real cell positions against saved spatial probability distributions.

Use this after ``compare_distributions.py`` has saved ``.npz`` density grids and
``distribution_manifest.csv``. For each cell type, this script first aligns the
real full point cloud to each distribution's saved reference sample, then asks
how likely the aligned real positions are under distribution A versus
distribution B.

Example
-------
    python utils/testing/sampling_proximity.py \
        --manifest outputs/distribution_comparison/distribution_manifest.csv \
        --real-csv path/to/real_positions.csv \
        --out-dir outputs/sampling_proximity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="distribution_manifest.csv written by compare_distributions.py.",
    )
    parser.add_argument(
        "--real-csv",
        required=True,
        type=Path,
        help="CSV containing real cell positions.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/sampling_proximity"),
        type=Path,
        help="Directory where likelihood tables and plots will be written.",
    )
    parser.add_argument(
        "--cell-class-column",
        default="cell_class",
        help="Column in the real CSV containing cell type labels.",
    )
    parser.add_argument(
        "--x-column",
        default="coord_X",
        help="Column in the real CSV containing X coordinates.",
    )
    parser.add_argument(
        "--y-column",
        default="coord_Y",
        help="Column in the real CSV containing Y coordinates.",
    )
    parser.add_argument(
        "--cell-id-column",
        default="cell_ID",
        help="Optional cell ID column to preserve in the per-cell output.",
    )
    parser.add_argument(
        "--epsilon",
        default=1e-12,
        type=float,
        help="Minimum density used before taking logs.",
    )
    parser.add_argument(
        "--summary-filename",
        default="likelihood_summary.csv",
        help="Filename for the per-cell-type summary table.",
    )
    parser.add_argument(
        "--per-cell-filename",
        default="per_cell_likelihoods.csv",
        help="Filename for the per-real-cell likelihood table.",
    )
    parser.add_argument(
        "--plot-filename",
        default="mean_log_likelihood_ratio.png",
        help="Filename for the summary plot.",
    )
    return parser.parse_args()


def _require_columns(df: Any, columns: set[str], source: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def _npz_scalar(npz: Any, key: str, default: str | None = None) -> str:
    if key not in npz:
        if default is None:
            raise KeyError(f"Missing key {key!r} in distribution npz.")
        return default
    value = npz[key]
    return str(value.item() if value.shape == () else value)


def _load_distribution(npz_path: Path) -> dict:
    import numpy as np

    with np.load(npz_path) as npz:
        required = {
            "density",
            "grid_x",
            "grid_y",
            "cell_class",
            "reference_all_points",
            "reference_cell_ids",
        }
        missing = required - set(npz.files)
        if missing:
            raise ValueError(
                f"{npz_path} is missing required arrays: {sorted(missing)}. "
                "Re-run compare_distributions.py so the saved densities include "
                "alignment reference metadata."
            )
        return {
            "density": npz["density"].astype(float),
            "grid_x": npz["grid_x"].astype(float),
            "grid_y": npz["grid_y"].astype(float),
            "cell_class": _npz_scalar(npz, "cell_class"),
            "label": _npz_scalar(npz, "label", npz_path.stem),
            "reference_all_points": npz["reference_all_points"].astype(float),
            "reference_cell_ids": npz["reference_cell_ids"].astype(str),
        }


def _fit_alignment_transform(base_points, target_points) -> tuple:
    """Fit the same rotation-only Procrustes transform used for sample alignment."""
    import numpy as np

    base_mean = np.mean(base_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    base_centered = base_points - base_mean
    target_centered = target_points - target_mean
    u, _, vt = np.linalg.svd(np.dot(target_centered.T, base_centered))
    rotation = np.dot(u, vt)
    return rotation, target_mean, base_mean


def _apply_alignment(points, rotation, target_mean, base_mean):
    import numpy as np

    return np.dot(points - target_mean, rotation) + base_mean


def _align_real_positions(real_df: Any, dist: dict, args: argparse.Namespace) -> tuple:
    """Align all real positions to one distribution's saved reference frame."""
    if args.cell_id_column not in real_df.columns:
        raise ValueError(
            f"Real CSV must contain {args.cell_id_column!r} so true cells can be "
            "matched to the saved reference sample before alignment."
        )

    real = real_df.copy()
    real["_match_cell_id"] = real[args.cell_id_column].astype(str)
    real = real.drop_duplicates(subset=["_match_cell_id"], keep="first")
    real_by_id = real.set_index("_match_cell_id", drop=False)

    ref_ids = dist["reference_cell_ids"].astype(str)
    ref_points = dist["reference_all_points"]
    common_ref_indices = [i for i, cell_id in enumerate(ref_ids) if cell_id in real_by_id.index]
    common_ids = [ref_ids[i] for i in common_ref_indices]

    if len(common_ids) < 2:
        raise ValueError(
            f"Need at least 2 shared cell IDs to align real positions to "
            f"{dist['label']!r}; found {len(common_ids)}."
        )

    base = ref_points[common_ref_indices]
    target = real_by_id.loc[common_ids, [args.x_column, args.y_column]].to_numpy(dtype=float)
    rotation, target_mean, base_mean = _fit_alignment_transform(base, target)

    all_points = real_df[[args.x_column, args.y_column]].to_numpy(dtype=float)
    aligned_points = _apply_alignment(all_points, rotation, target_mean, base_mean)
    return aligned_points, len(common_ids), len(ref_ids)


def _evaluate_density(
    density,
    grid_x,
    grid_y,
    points,
    epsilon: float,
) -> tuple:
    """Bilinearly interpolate density at points; outside-grid points get epsilon."""
    import numpy as np

    if density.shape != (grid_y.size, grid_x.size):
        raise ValueError(
            "Density shape does not match grid sizes: "
            f"density={density.shape}, grid_y={grid_y.size}, grid_x={grid_x.size}"
        )
    if grid_x.size < 2 or grid_y.size < 2:
        raise ValueError("Density grids need at least two x and y coordinates.")

    x = points[:, 0]
    y = points[:, 1]
    inside = (
        (x >= grid_x[0])
        & (x <= grid_x[-1])
        & (y >= grid_y[0])
        & (y <= grid_y[-1])
    )

    values = np.full(points.shape[0], epsilon, dtype=float)
    if not inside.any():
        return values, ~inside

    xi = np.searchsorted(grid_x, x[inside], side="right") - 1
    yi = np.searchsorted(grid_y, y[inside], side="right") - 1
    xi = np.clip(xi, 0, grid_x.size - 2)
    yi = np.clip(yi, 0, grid_y.size - 2)

    x0 = grid_x[xi]
    x1 = grid_x[xi + 1]
    y0 = grid_y[yi]
    y1 = grid_y[yi + 1]
    tx = np.divide(x[inside] - x0, x1 - x0, out=np.zeros_like(x0), where=x1 != x0)
    ty = np.divide(y[inside] - y0, y1 - y0, out=np.zeros_like(y0), where=y1 != y0)

    d00 = density[yi, xi]
    d10 = density[yi, xi + 1]
    d01 = density[yi + 1, xi]
    d11 = density[yi + 1, xi + 1]
    interp = (
        (1.0 - tx) * (1.0 - ty) * d00
        + tx * (1.0 - ty) * d10
        + (1.0 - tx) * ty * d01
        + tx * ty * d11
    )
    values[inside] = np.maximum(interp, epsilon)
    return values, ~inside


def _score_cell_class(
    real_cells,
    dist_a: dict,
    dist_b: dict,
    aligned_points_a,
    aligned_points_b,
    args: argparse.Namespace,
) -> tuple:
    import numpy as np
    import pandas as pd

    likelihood_a, outside_a = _evaluate_density(
        dist_a["density"],
        dist_a["grid_x"],
        dist_a["grid_y"],
        aligned_points_a,
        args.epsilon,
    )
    likelihood_b, outside_b = _evaluate_density(
        dist_b["density"],
        dist_b["grid_x"],
        dist_b["grid_y"],
        aligned_points_b,
        args.epsilon,
    )

    log_a = np.log(likelihood_a)
    log_b = np.log(likelihood_b)
    log_ratio = log_a - log_b
    cell_class = str(dist_a["cell_class"])

    per_cell = pd.DataFrame(
        {
            "cell_class": cell_class,
            "coord_X": real_cells[args.x_column].to_numpy(dtype=float),
            "coord_Y": real_cells[args.y_column].to_numpy(dtype=float),
            "aligned_coord_X_a": aligned_points_a[:, 0],
            "aligned_coord_Y_a": aligned_points_a[:, 1],
            "aligned_coord_X_b": aligned_points_b[:, 0],
            "aligned_coord_Y_b": aligned_points_b[:, 1],
            "likelihood_a": likelihood_a,
            "likelihood_b": likelihood_b,
            "log_likelihood_a": log_a,
            "log_likelihood_b": log_b,
            "log_likelihood_ratio_a_over_b": log_ratio,
            "outside_grid_a": outside_a,
            "outside_grid_b": outside_b,
        }
    )
    if args.cell_id_column in real_cells.columns:
        per_cell.insert(1, args.cell_id_column, real_cells[args.cell_id_column].to_numpy())

    mean_log_a = float(log_a.mean())
    mean_log_b = float(log_b.mean())
    summary = {
        "cell_class": cell_class,
        "n_real_cells": int(aligned_points_a.shape[0]),
        "label_a": dist_a["label"],
        "label_b": dist_b["label"],
        "mean_likelihood_a": float(likelihood_a.mean()),
        "mean_likelihood_b": float(likelihood_b.mean()),
        "median_likelihood_a": float(np.median(likelihood_a)),
        "median_likelihood_b": float(np.median(likelihood_b)),
        "total_log_likelihood_a": float(log_a.sum()),
        "total_log_likelihood_b": float(log_b.sum()),
        "mean_log_likelihood_a": mean_log_a,
        "mean_log_likelihood_b": mean_log_b,
        "mean_log_likelihood_ratio_a_over_b": float(log_ratio.mean()),
        "total_log_likelihood_ratio_a_over_b": float(log_ratio.sum()),
        "outside_grid_fraction_a": float(outside_a.mean()),
        "outside_grid_fraction_b": float(outside_b.mean()),
        "preferred_distribution": dist_a["label"] if mean_log_a >= mean_log_b else dist_b["label"],
    }
    return per_cell, summary


def _plot_summary(summary_df: Any, save_path: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = summary_df.sort_values("mean_log_likelihood_ratio_a_over_b")
    colors = np.where(plot_df["mean_log_likelihood_ratio_a_over_b"] >= 0, "tab:blue", "tab:orange")

    height = max(4.0, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(
        plot_df["cell_class"],
        plot_df["mean_log_likelihood_ratio_a_over_b"],
        color=colors,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    label_a = str(plot_df["label_a"].iloc[0])
    label_b = str(plot_df["label_b"].iloc[0])
    ax.set_xlabel(f"Mean log likelihood ratio ({label_a} - {label_b})")
    ax.set_ylabel("Cell class")
    ax.set_title("Real-position likelihood under saved spatial distributions")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[sampling_proximity] Saved {save_path}")


def _safe_plot_filename(value: str, max_length: int = 120) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return (name or "cell_class")[:max_length]


def _find_predictions_csv(testing_dir: Path) -> Path | None:
    candidates = [
        testing_dir / "predictions.csv",
        testing_dir / "predicted_positions.csv",
    ]
    candidates.extend(sorted(testing_dir.glob("*pred*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _reference_from_predictions_csv(predictions_csv: Path, align_target: int) -> dict:
    import pandas as pd

    df = pd.read_csv(predictions_csv)
    _require_columns(
        df,
        {"sample_index", "cell_ID", "cell_class", "coord_X", "coord_Y"},
        predictions_csv,
    )
    sample_indices = sorted(df["sample_index"].unique())
    if not (0 <= align_target < len(sample_indices)):
        raise ValueError(
            f"align_target={align_target} is out of range for {predictions_csv}; "
            f"found {len(sample_indices)} samples."
        )

    sample_value = sample_indices[align_target]
    ref = df[df["sample_index"] == sample_value].sort_values("cell_ID", kind="stable")
    return {
        "reference_all_points": ref[["coord_X", "coord_Y"]].to_numpy(dtype=float),
        "reference_cell_ids": ref["cell_ID"].astype(str).to_numpy(),
        "reference_cell_classes": ref["cell_class"].astype(str).to_numpy(),
    }


def _load_distribution_for_gt_overlay(
    npz_path: Path,
    testing_dir: Path,
    align_target: int,
) -> dict:
    import numpy as np

    with np.load(npz_path) as npz:
        required = {"density", "grid_x", "grid_y", "cell_class"}
        missing = required - set(npz.files)
        if missing:
            raise ValueError(f"{npz_path} is missing required arrays: {sorted(missing)}")

        dist = {
            "density": npz["density"].astype(float),
            "grid_x": npz["grid_x"].astype(float),
            "grid_y": npz["grid_y"].astype(float),
            "extent": npz["extent"].astype(float) if "extent" in npz else None,
            "cell_class": _npz_scalar(npz, "cell_class"),
            "label": _npz_scalar(npz, "label", npz_path.stem),
        }
        if {"reference_all_points", "reference_cell_ids"} <= set(npz.files):
            dist["reference_all_points"] = npz["reference_all_points"].astype(float)
            dist["reference_cell_ids"] = npz["reference_cell_ids"].astype(str)
            return dist

    predictions_csv = _find_predictions_csv(testing_dir)
    if predictions_csv is None:
        raise ValueError(
            f"{npz_path} does not contain alignment reference metadata, and no "
            f"predictions CSV was found in {testing_dir}. Re-run the density "
            "generation with the current scripts or keep predictions.csv in the folder."
        )
    dist.update(_reference_from_predictions_csv(predictions_csv, align_target))
    return dist


def plot_gt_over_probability_npz(
    gt_csv: str | Path,
    testing_dir: str | Path,
    output_dir: str | Path | None = None,
    align_target: int = 0,
    cell_class_column: str = "cell_class",
    x_column: str = "coord_X",
    y_column: str = "coord_Y",
    cell_id_column: str = "cell_ID",
    min_cells: int = 1,
    overlay_size: float = 6.0,
    cmap: str = "magma",
) -> list[Path]:
    """Recreate density plots from saved ``.npz`` files with aligned GT dots.

    Parameters
    ----------
    gt_csv:
        Ground-truth/real-cell CSV containing cell IDs, classes and coordinates.
    testing_dir:
        Folder containing ``.npz`` density files produced by the testing scripts.
        Files are discovered recursively, so both ``testing_dir/*.npz`` and
        ``testing_dir/densities/*.npz`` work.
    output_dir:
        Destination for overlay plots. Defaults to ``testing_dir/gt_overlays``.
    align_target:
        Reference sample index to use when an ``.npz`` lacks saved reference
        metadata and the function has to reconstruct the reference from
        ``predictions.csv``.

    Returns
    -------
    list[Path]
        Paths of the saved overlay figures.
    """
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_csv = Path(gt_csv)
    testing_dir = Path(testing_dir)
    output_dir = Path(output_dir) if output_dir is not None else testing_dir / "gt_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(gt_csv)
    _require_columns(gt_df, {cell_class_column, x_column, y_column, cell_id_column}, gt_csv)
    gt_df = gt_df.dropna(subset=[cell_class_column, x_column, y_column, cell_id_column]).copy()
    gt_df[cell_class_column] = gt_df[cell_class_column].astype(str)

    npz_files = sorted(testing_dir.rglob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz density files found under {testing_dir}.")

    saved_paths = []
    for npz_path in npz_files:
        dist = _load_distribution_for_gt_overlay(npz_path, testing_dir, align_target)
        cell_class = str(dist["cell_class"])
        class_mask = gt_df[cell_class_column] == cell_class
        gt_cells = gt_df[class_mask]
        if len(gt_cells) < min_cells:
            print(
                f"[sampling_proximity] Skipping {cell_class!r} from {npz_path}: "
                f"found {len(gt_cells)} GT cells, need at least {min_cells}."
            )
            continue

        args = argparse.Namespace(
            cell_id_column=cell_id_column,
            x_column=x_column,
            y_column=y_column,
        )
        aligned_all, matched, ref_total = _align_real_positions(gt_df, dist, args)
        aligned_gt = aligned_all[class_mask.to_numpy()]

        density = dist["density"]
        extent = dist["extent"]
        if extent is None:
            extent = (
                float(dist["grid_x"][0]),
                float(dist["grid_x"][-1]),
                float(dist["grid_y"][0]),
                float(dist["grid_y"][-1]),
            )

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            density,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.scatter(
            aligned_gt[:, 0],
            aligned_gt[:, 1],
            s=overlay_size,
            color="cyan",
            marker="o",
            linewidths=0,
            label=f"aligned GT ({len(gt_cells)} cells)",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("probability density (per unit area)")
        ax.set_title(
            f"GT overlay - {cell_class!r}\n"
            f"{dist['label']}, aligned with {matched}/{ref_total} reference cells"
        )
        ax.set_xlabel("X (aligned)")
        ax.set_ylabel("Y (aligned)")

        save_name = f"{_safe_plot_filename(npz_path.stem)}__gt_overlay.png"
        save_path = output_dir / save_name
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        saved_paths.append(save_path)
        print(f"[sampling_proximity] Saved {save_path}")

    if not saved_paths:
        raise ValueError("No GT overlay plots were produced.")
    return saved_paths


def main() -> None:
    '''args = _parse_args()
    import pandas as pd

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    _require_columns(manifest, {"cell_class", "npz_a", "npz_b"}, args.manifest)

    real_df = pd.read_csv(args.real_csv)
    _require_columns(
        real_df,
        {args.cell_class_column, args.x_column, args.y_column},
        args.real_csv,
    )
    real_df = real_df.dropna(
        subset=[args.cell_class_column, args.x_column, args.y_column]
    ).copy()
    real_df[args.cell_class_column] = real_df[args.cell_class_column].astype(str)

    # Remove cell classes that have fewer than 10 cells
    cell_class_counts = real_df[args.cell_class_column].value_counts()
    cell_classes_to_keep = cell_class_counts[cell_class_counts >= 10].index
    real_df = real_df[real_df[args.cell_class_column].isin(cell_classes_to_keep)].copy()

    per_cell_tables = []
    summaries = []
    for row in manifest.itertuples(index=False):
        cell_class = str(row.cell_class)
        if cell_class not in cell_classes_to_keep:
            print(f"[sampling_proximity] Skipping {cell_class!r}: fewer than 10 real cells.")
            continue
        class_mask = real_df[args.cell_class_column] == cell_class
        real_cells = real_df[class_mask]
        if real_cells.empty:
            print(f"[sampling_proximity] Skipping {cell_class!r}: no real cells found.")
            continue

        npz_a = _resolve_manifest_path(args.manifest, str(row.npz_a))
        npz_b = _resolve_manifest_path(args.manifest, str(row.npz_b))
        print(
            f"[sampling_proximity] Scoring {len(real_cells)} real cells for "
            f"{cell_class!r}"
        )
        dist_a = _load_distribution(npz_a)
        dist_b = _load_distribution(npz_b)
        if dist_a["cell_class"] != cell_class or dist_b["cell_class"] != cell_class:
            raise ValueError(
                f"Manifest row {cell_class!r} does not match npz cell classes "
                f"{dist_a['cell_class']!r} and {dist_b['cell_class']!r}."
            )

        aligned_all_a, matched_a, ref_total_a = _align_real_positions(real_df, dist_a, args)
        aligned_all_b, matched_b, ref_total_b = _align_real_positions(real_df, dist_b, args)
        class_mask_array = class_mask.to_numpy()
        per_cell, summary = _score_cell_class(
            real_cells,
            dist_a,
            dist_b,
            aligned_all_a[class_mask_array],
            aligned_all_b[class_mask_array],
            args,
        )
        summary["alignment_matched_cells_a"] = matched_a
        summary["alignment_reference_cells_a"] = ref_total_a
        summary["alignment_matched_cells_b"] = matched_b
        summary["alignment_reference_cells_b"] = ref_total_b
        per_cell_tables.append(per_cell)
        summaries.append(summary)

    if not summaries:
        raise ValueError("No manifest cell classes had matching real cells to score.")

    per_cell_df = pd.concat(per_cell_tables, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("cell_class")

    per_cell_path = args.out_dir / args.per_cell_filename
    summary_path = args.out_dir / args.summary_filename
    plot_path = args.out_dir / args.plot_filename

    per_cell_df.to_csv(per_cell_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[sampling_proximity] Saved {per_cell_path}")
    print(f"[sampling_proximity] Saved {summary_path}")
    _plot_summary(summary_df, plot_path)'''

    plot_gt_over_probability_npz(
        gt_csv="/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_10-54-06-MERFISH_epoch_249/mouse2_slice300_0/metadata_true.csv",
        testing_dir="/home/asalmona/Documents/Ricci/code/__some_results/testing/comparing_ch_and_null_r0.05",
        align_target=0,
    )


if __name__ == "__main__":
    main()
