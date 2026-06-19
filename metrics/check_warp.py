"""Visualise the smooth global position-warp augmentation on one slice.

Loads a CSV slice, normalises coordinates, samples the epoch warp field, and
writes a side-by-side scatter plot (original vs warped).

Usage::

    python metrics/test_warp.py \\
        --csv /path/to/MERFISH_mouse_cortex_train.csv \\
        --cell-section mouse1.V1 \\
        --seed 7 \\
        --output results/test_warp_slice.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data.load import position_normalize, standardise_dataframe_colnames
from utils.data.warp_augment import sample_smooth_warp_field, warp_displacement_grid


def _load_slice_positions(csv_path: Path, cell_section: str) -> np.ndarray:
    df = pd.read_csv(csv_path, index_col=0)
    df = standardise_dataframe_colnames(df)
    if "cell_section" not in df.columns:
        raise ValueError("CSV must contain a cell_section column.")
    section_df = df.loc[df["cell_section"] == cell_section].copy()
    if section_df.empty:
        raise ValueError(f"No rows for cell_section={cell_section!r}.")
    section_df = position_normalize(section_df)
    return section_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)


def _plot_warp_comparison(
    xy: np.ndarray,
    warped: np.ndarray,
    field,
    *,
    output: Path,
    title: str,
) -> None:
    disp = warped - xy
    disp_norm = np.linalg.norm(disp, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    axes[0].scatter(xy[:, 0], xy[:, 1], s=3, c="#355C7D", alpha=0.55, linewidths=0)
    axes[0].set_title("Original coordinates")
    axes[0].set_xlabel("coord_X")
    axes[0].set_ylabel("coord_Y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.2)

    sc = axes[1].scatter(
        warped[:, 0],
        warped[:, 1],
        s=3,
        c=disp_norm,
        cmap="viridis",
        alpha=0.65,
        linewidths=0,
    )
    axes[1].set_title("Warped coordinates")
    axes[1].set_xlabel("coord_X")
    axes[1].set_ylabel("coord_Y")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("|displacement|")

    xs, ys, dx, dy = warp_displacement_grid(field, resolution=24)
    step = max(1, len(xs) // 12)
    axes[1].quiver(
        xs[::step],
        ys[::step],
        dx[::step, ::step],
        dy[::step, ::step],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="crimson",
        alpha=0.8,
        width=0.004,
    )

    fig.suptitle(title, fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise smooth position warp.")
    parser.add_argument(
        "--csv",
        type=Path,
        default="/data-master/code/Tissue-Reassembly-using-GenAI-and-new-metrics/results/test_results.csv",
        help="Training CSV with coord_X/coord_Y and cell_section.",
    )
    parser.add_argument(
        "--cell-section",
        type=str,
        default="mouse2_slice300",
        help="Slice label in cell_section (default: first section in CSV).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Warp field RNG seed.")
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=0.05,
        help="Max displacement in normalised coordinates.",
    )
    parser.add_argument(
        "--max-angle-span",
        type=float,
        default=float(np.pi / 2),
        help="Max direction variation (radians) across the domain.",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "test_warp_slice.png",
    )
    args = parser.parse_args()

    df_head = pd.read_csv(args.csv, index_col=0, nrows=1)
    df_head = standardise_dataframe_colnames(df_head)
    cell_section = args.cell_section
    if cell_section is None:
        full = pd.read_csv(args.csv, index_col=0, usecols=["cell_section"])
        full = standardise_dataframe_colnames(full)
        cell_section = str(full["cell_section"].iloc[0])

    xy = _load_slice_positions(args.csv, cell_section)
    field = sample_smooth_warp_field(
        args.seed,
        grid_size=args.grid_size,
        max_displacement=args.max_displacement,
        max_angle_span=args.max_angle_span,
    )
    warped = field.warp(xy)

    title = (
        f"Smooth warp seed={args.seed}  section={cell_section}  "
        f"max_disp={args.max_displacement:g}  max_angle={args.max_angle_span:.3g} rad"
    )
    _plot_warp_comparison(
        xy,
        warped,
        field,
        output=args.output,
        title=title,
    )
    print(f"Wrote {args.output}")
    print(
        f"Cells={len(xy):,}  |disp| mean={np.linalg.norm(warped - xy, axis=1).mean():.5f}  "
        f"max={np.linalg.norm(warped - xy, axis=1).max():.5f}"
    )


if __name__ == "__main__":
    main()
