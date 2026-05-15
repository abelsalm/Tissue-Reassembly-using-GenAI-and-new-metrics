"""Compare spatial probability maps from two LUNA prediction CSVs.

The input CSVs should be the long-format files produced by
``utils/testing/diffusion2spatial_probs.py``. For every shared cell class, this
script independently computes the spatial probability map for each CSV and
saves a side-by-side comparison plot.

Example
-------
    python utils/testing/compare_distributions.py \
        --csv-a outputs/model_a/predictions.csv \
        --csv-b outputs/model_b/predictions.csv \
        --label-a model_a \
        --label-b model_b \
        --out-dir outputs/distribution_comparison
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv-a", required=True, type=Path, help="First predictions CSV.")
    parser.add_argument("--csv-b", required=True, type=Path, help="Second predictions CSV.")
    parser.add_argument(
        "--label-a",
        default="CSV A",
        help="Display label for the first CSV/model.",
    )
    parser.add_argument(
        "--label-b",
        default="CSV B",
        help="Display label for the second CSV/model.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/distribution_comparison"),
        type=Path,
        help="Directory where comparison plots will be written.",
    )
    parser.add_argument(
        "--cell-class",
        action="append",
        dest="cell_classes",
        help=(
            "Cell class to plot. Can be passed multiple times. "
            "By default, all classes shared by both CSVs are plotted."
        ),
    )
    parser.add_argument(
        "--gaussian-sigma",
        default=0.05,
        type=float,
        help="Gaussian sigma passed to compute_spatial_probability_map.",
    )
    parser.add_argument(
        "--grid-resolution",
        default=256,
        type=int,
        help="Grid resolution passed to compute_spatial_probability_map.",
    )
    parser.add_argument(
        "--grid-margin",
        default=0.05,
        type=float,
        help="Grid margin passed to compute_spatial_probability_map.",
    )
    parser.add_argument(
        "--align-reference",
        default=0,
        type=int,
        help="Sample index used as the alignment reference in each CSV.",
    )
    parser.add_argument(
        "--overlay-reference-scatter",
        action="store_true",
        help="Overlay reference-sample points for the plotted cell class.",
    )
    parser.add_argument(
        "--separate-color-scales",
        action="store_true",
        help="Use an independent color scale for each subplot.",
    )
    return parser.parse_args()


def _cell_classes(csv_path: Path) -> set[str]:
    from diffusion2spatial_probs import load_predictions_csv

    df = load_predictions_csv(csv_path)
    return set(df["cell_class"].dropna().astype(str).unique())


def _safe_filename(value: str, max_length: int = 120) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = name.strip("._")
    if not name:
        name = "cell_class"
    return name[:max_length]


def _select_cell_classes(
    csv_a: Path,
    csv_b: Path,
    requested: Iterable[str] | None,
) -> list[str]:
    classes_a = _cell_classes(csv_a)
    classes_b = _cell_classes(csv_b)

    if requested:
        missing_a = sorted(set(requested) - classes_a)
        missing_b = sorted(set(requested) - classes_b)
        if missing_a or missing_b:
            parts = []
            if missing_a:
                parts.append(f"missing from csv-a: {missing_a}")
            if missing_b:
                parts.append(f"missing from csv-b: {missing_b}")
            raise ValueError("; ".join(parts))
        return sorted(set(requested))

    shared = sorted(classes_a & classes_b)
    only_a = sorted(classes_a - classes_b)
    only_b = sorted(classes_b - classes_a)
    if only_a:
        print(f"[compare_distributions] Skipping classes only in csv-a: {only_a}")
    if only_b:
        print(f"[compare_distributions] Skipping classes only in csv-b: {only_b}")
    if not shared:
        raise ValueError("No shared cell_class values found between the two CSVs.")
    # remove cell type if under 10 cells
    shared = [cell_type for cell_type in shared if df[df["cell_class"] == cell_type]["cell_ID"].nunique() >= 10]
    return shared


def _plot_comparison(
    result_a: dict,
    result_b: dict,
    label_a: str,
    label_b: str,
    save_path: Path,
    overlay_reference_scatter: bool,
    separate_color_scales: bool,
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_class = str(result_a["cell_class"])
    density_a = result_a["density"]
    density_b = result_b["density"]

    if separate_color_scales:
        vmin = vmax = None
    else:
        vmin = 0.0
        vmax = float(max(np.nanmax(density_a), np.nanmax(density_b)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, result, label, density in (
        (axes[0], result_a, label_a, density_a),
        (axes[1], result_b, label_b, density_b),
    ):
        im = ax.imshow(
            density,
            origin="lower",
            extent=result["extent"],
            cmap="magma",
            aspect="equal",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        if overlay_reference_scatter:
            ref_pts = result["reference_points"]
            if ref_pts.shape[0] > 0:
                ax.scatter(
                    ref_pts[:, 0],
                    ref_pts[:, 1],
                    s=4,
                    color="cyan",
                    marker="o",
                    label="reference sample",
                )
                ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        ax.set_title(
            f"{label}\n"
            f"{result['num_samples']} samples x {result['num_class_cells']} cells"
        )
        ax.set_xlabel("X (normalised)")
        ax.set_ylabel("Y (normalised)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("probability density (per unit area)")

    fig.suptitle(
        f"Spatial probability comparison - {cell_class!r}\n"
        f"sigma={result_a['gaussian_sigma']:g}"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[compare_distributions] Saved {save_path}")


def _save_density_npz(
    result: dict,
    save_path: Path,
    source_csv: Path,
    label: str,
) -> None:
    import numpy as np

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
    print(f"[compare_distributions] Saved {save_path}")


def main() -> None:
    args = _parse_args()
    from diffusion2spatial_probs import compute_spatial_probability_map

    cell_classes = _select_cell_classes(args.csv_a, args.csv_b, args.cell_classes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    density_dir = args.out_dir / "densities"
    manifest_path = args.out_dir / "distribution_manifest.csv"

    print(
        f"[compare_distributions] Comparing {len(cell_classes)} cell classes "
        f"from {args.csv_a} and {args.csv_b}"
    )

    manifest_rows = []
    for cell_class in cell_classes:
        print(f"[compare_distributions] Computing maps for {cell_class!r}")
        result_a = compute_spatial_probability_map(
            predictions_csv=args.csv_a,
            cell_class=cell_class,
            gaussian_sigma=args.gaussian_sigma,
            grid_resolution=args.grid_resolution,
            grid_margin=args.grid_margin,
            align_reference=args.align_reference,
        )
        result_b = compute_spatial_probability_map(
            predictions_csv=args.csv_b,
            cell_class=cell_class,
            gaussian_sigma=args.gaussian_sigma,
            grid_resolution=args.grid_resolution,
            grid_margin=args.grid_margin,
            align_reference=args.align_reference,
        )

        safe_cell_class = _safe_filename(cell_class)
        save_path = args.out_dir / f"{safe_cell_class}.png"
        npz_a = density_dir / f"{safe_cell_class}__a.npz"
        npz_b = density_dir / f"{safe_cell_class}__b.npz"
        _save_density_npz(result_a, npz_a, args.csv_a, args.label_a)
        _save_density_npz(result_b, npz_b, args.csv_b, args.label_b)
        _plot_comparison(
            result_a=result_a,
            result_b=result_b,
            label_a=args.label_a,
            label_b=args.label_b,
            save_path=save_path,
            overlay_reference_scatter=args.overlay_reference_scatter,
            separate_color_scales=args.separate_color_scales,
        )
        manifest_rows.append(
            {
                "cell_class": cell_class,
                "plot_path": str(save_path),
                "npz_a": str(npz_a),
                "npz_b": str(npz_b),
                "label_a": args.label_a,
                "label_b": args.label_b,
                "csv_a": str(args.csv_a),
                "csv_b": str(args.csv_b),
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[compare_distributions] Saved manifest {manifest_path}")


if __name__ == "__main__":
    main()
