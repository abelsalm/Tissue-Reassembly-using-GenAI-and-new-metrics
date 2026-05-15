"""LUNA spatial probability-map experiment.

Run many LUNA inferences on a single slice of the test dataset (selected by
``cell_section``) and aggregate the predicted positions of a chosen cell class
into a spatial probability map.

Usage
-----
    python utils/testing/diffusion2spatial_probs.py \
        cell_section=mouse2.AUDp \
        analysis.cell_class="L2/3 IT CTX Glut" \
        num_samples=128

CLI overrides use Hydra-style ``key=value`` dot-list syntax and are applied
to ``utils/testing/config.yaml`` after it has been loaded with OmegaConf.

The main LUNA config (``configs/config.yaml``) is composed automatically via
``hydra.compose`` so dataset / model paths stay in sync with the rest of the
codebase. Inference-only overrides defined in ``luna_overrides`` are layered
on top.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.evaluation_statistics import align_point_clouds  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_omegaconf():
    """Import OmegaConf only for the config-driven CLI workflow."""
    from omegaconf import OmegaConf

    # Hydra registers ``${now:...}`` lazily when ``@hydra.main`` runs; we call
    # ``OmegaConf.resolve`` directly so we register it here ourselves.
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda fmt: datetime.now().strftime(fmt), replace=False
        )
    return OmegaConf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the testing YAML config (default: utils/testing/config.yaml).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, e.g. cell_section=foo analysis.gaussian_sigma=0.03",
    )
    return parser.parse_args()


def load_testing_config(config_path: Path, overrides: list[str]) -> Any:
    """Load the testing YAML config and apply CLI dot-list overrides."""
    OmegaConf = _load_omegaconf()
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    # Resolve interpolations like ${now:...}; this materialises the experiment_name.
    OmegaConf.resolve(cfg)
    return cfg


def compose_luna_config(overrides: list[str]) -> Any:
    """Compose the main LUNA config (``configs/config.yaml``) programmatically."""
    from hydra import compose, initialize_config_dir

    configs_dir = (REPO_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Slice extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SliceBatch:
    """Holds the dense, single-graph batch fed to LUNA plus bookkeeping."""

    holder: Any
    cell_class_int: np.ndarray  # (N,) integer codes
    cell_class_labels: list[str]  # (N,) decoded strings
    cell_ids: np.ndarray  # (N,)
    gt_positions: np.ndarray  # (N, 2) ground-truth (normalised) coords
    cell_section: str


def _find_section_slice(test_dataset, cell_section) -> tuple[int, int, str]:
    """Locate ``(start, end)`` indices of the requested cell section.

    The dataset is sorted by ``cell_section`` during processing and (when
    ``maximum_graph_size is None``) every section is a single contiguous slice.
    We honour the slice boundaries to avoid spanning chunks created by the
    rechunk-aware loader.
    """
    sections = np.asarray(test_dataset._cell_sections_clean)
    boundaries = test_dataset.slices["positions"].tolist()

    available = sorted({str(s) for s in np.unique(sections)})

    if cell_section is None:
        if not available:
            raise RuntimeError("Test dataset is empty: no sections to choose from.")
        target = sections[boundaries[0]]
        print(
            f"[diffusion2spatial_probs] cell_section not provided — "
            f"falling back to first available section: {target!r}"
        )
        return int(boundaries[0]), int(boundaries[1]), str(target)

    target_str = str(cell_section)
    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        if start >= end:
            continue
        if str(sections[start]) == target_str:
            return start, end, target_str

    raise ValueError(
        f"cell_section={cell_section!r} not found in test split. "
        f"Available sections (first 20): {available[:20]}"
    )


def build_section_batch(
    datamodule: Any,
    dataset_infos: Any,
    cell_section,
) -> SliceBatch:
    """Extract a dense, padded batch for a single ``cell_section`` of the test split."""
    import torch

    from utils.data.dataholder import DataHolder

    test_dataset = datamodule.test_dataset
    start, end, section_str = _find_section_slice(test_dataset, cell_section)

    data = test_dataset._data
    positions = data.positions[start:end].float().unsqueeze(0)  # (1, N, 2)
    node_features = data.node_features[start:end].float().unsqueeze(0)  # (1, N, F)
    cell_class = data.cell_class[start:end].long().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    cell_id_tensor = data.cell_ID[start:end].long().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    n_cells = positions.shape[1]
    node_mask = torch.ones(1, n_cells, dtype=torch.bool)

    holder = DataHolder(
        positions=positions,
        node_features=node_features,
        cell_class=cell_class,
        cell_ID=cell_id_tensor,
        node_mask=node_mask,
        diffusion_time=None,
    ).mask()

    # Pull the decoded labels (string class names) once, in the same order as
    # the tensor rows.
    cc_int = cell_class.squeeze(0).squeeze(-1).cpu().numpy()
    cc_labels = [dataset_infos.cell_class_decoder[int(i)] for i in cc_int]

    # Re-fetch the *original* (string-or-int) cell IDs for reporting. The
    # Dataset stores integer codes plus the original labels on the side; for
    # string IDs the labels live in a ``pd.factorize`` uniques array indexed
    # by the integer codes, while for numeric IDs they are already aligned
    # to the per-row order.
    ids_original_arr = np.asarray(test_dataset._cell_ids_original)
    if ids_original_arr.dtype.kind in ("U", "S", "O"):
        codes = data.cell_ID[start:end].cpu().numpy()
        original_ids = ids_original_arr[codes]
    else:
        original_ids = ids_original_arr[start:end]

    # Ground truth uses the masked-and-centered positions (matches what LUNA
    # is conditioned on).
    gt_positions = holder.positions.squeeze(0).cpu().numpy()

    print(
        f"[diffusion2spatial_probs] Selected section {section_str!r} "
        f"with {n_cells} cells."
    )
    return SliceBatch(
        holder=holder,
        cell_class_int=cc_int,
        cell_class_labels=cc_labels,
        cell_ids=original_ids,
        gt_positions=gt_positions,
        cell_section=section_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading and sampling
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(
    luna_cfg: Any,
    dataset_infos: Any,
    checkpoint_path: str,
    device: Any,
) -> Any:
    """Load a trained LUNA model and move it to ``device`` in eval mode."""
    from diffusion_model import FullDenoisingDiffusion

    model = FullDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        dataset_infos=dataset_infos,
        cfg=luna_cfg,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inferences(
    model: Any,
    batch: SliceBatch,
    num_samples: int,
    seed_start: int,
    device: Any,
) -> np.ndarray:
    """Run LUNA ``num_samples`` times on the same input batch.

    Returns
    -------
    np.ndarray of shape (num_samples, N, 2) — predicted, masked positions.
    """
    from utils.data.dataholder import DataHolder
    from utils.diffusion_model.sample.sample import sample_from_single_graph

    holder = batch.holder
    holder_dev = DataHolder(
        positions=holder.positions.to(device),
        node_features=holder.node_features.to(device),
        cell_class=holder.cell_class.to(device),
        cell_ID=holder.cell_ID.to(device),
        node_mask=holder.node_mask.to(device),
        diffusion_time=None,
    )

    n_cells = holder.positions.shape[1]
    preds = np.empty((num_samples, n_cells, 2), dtype=np.float32)

    for i in tqdm(range(num_samples), desc="LUNA inferences", unit="sample"):
        _set_seed(seed_start + i)
        sample = sample_from_single_graph(model, test=True, batch=holder_dev)
        # ``sample`` has shape (1, N, 2). Squeeze and move to CPU.
        preds[i] = sample.squeeze(0).detach().cpu().numpy()

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────


def save_predictions_csv(
    preds: np.ndarray,
    batch: SliceBatch,
    seed_start: int,
    output_csv: Path,
    gt_csv: Optional[Path],
) -> None:
    """Long-format CSV: one row per (sample_index, cell)."""
    num_samples, n_cells, _ = preds.shape
    n_total = num_samples * n_cells

    sample_idx = np.repeat(np.arange(num_samples), n_cells)
    seeds = seed_start + sample_idx
    cell_ids = np.tile(batch.cell_ids, num_samples)
    cell_class_labels = np.tile(np.asarray(batch.cell_class_labels), num_samples)
    cell_class_ints = np.tile(batch.cell_class_int, num_samples)
    flat_positions = preds.reshape(n_total, 2)

    df = pd.DataFrame(
        {
            "sample_index": sample_idx,
            "seed": seeds,
            "cell_section": batch.cell_section,
            "cell_ID": cell_ids,
            "cell_class": cell_class_labels,
            "cell_class_int": cell_class_ints,
            "coord_X": flat_positions[:, 0],
            "coord_Y": flat_positions[:, 1],
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"[diffusion2spatial_probs] Saved {len(df):,} predicted positions "
        f"({num_samples} samples × {n_cells} cells) → {output_csv}"
    )

    if gt_csv is not None:
        gt_df = pd.DataFrame(
            {
                "cell_section": batch.cell_section,
                "cell_ID": batch.cell_ids,
                "cell_class": batch.cell_class_labels,
                "cell_class_int": batch.cell_class_int,
                "coord_X": batch.gt_positions[:, 0],
                "coord_Y": batch.gt_positions[:, 1],
            }
        )
        gt_csv.parent.mkdir(parents=True, exist_ok=True)
        gt_df.to_csv(gt_csv, index=False)
        print(f"[diffusion2spatial_probs] Saved ground truth → {gt_csv}")


def load_predictions_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sample_index", "cell_ID", "cell_class", "coord_X", "coord_Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing columns: {missing}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Spatial probability map
# ─────────────────────────────────────────────────────────────────────────────


def _predictions_to_array(df: pd.DataFrame) -> np.ndarray:
    """Pivot a long-format predictions DataFrame back into a ``(S, N, 2)`` tensor.

    Assumes every sample has the same number of cells in the same order (which
    is exactly what ``run_inferences`` produces).
    """
    df_sorted = df.sort_values(["sample_index", "cell_ID"], kind="stable")
    sample_indices = sorted(df_sorted["sample_index"].unique())
    arrays = []
    n_cells = None
    for s in sample_indices:
        sub = df_sorted[df_sorted["sample_index"] == s][["coord_X", "coord_Y"]].to_numpy()
        if n_cells is None:
            n_cells = sub.shape[0]
        elif sub.shape[0] != n_cells:
            raise ValueError(
                f"Sample {s} has {sub.shape[0]} cells but expected {n_cells}; "
                "spatial-probability aggregation requires the same cells in every sample."
            )
        arrays.append(sub)
    return np.stack(arrays, axis=0).astype(np.float32)


def _align_samples(
    samples: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    """Rotate every sample onto ``samples[reference_idx]`` via Procrustes (rotation only).

    Operates on the *full* point cloud of each sample (i.e. all cells, all
    classes) — this is what the user asked for: align predictions globally,
    *then* analyse a specific cell class.
    """
    ref = samples[reference_idx]
    aligned = np.empty_like(samples)
    aligned[reference_idx] = ref
    for i in range(samples.shape[0]):
        if i == reference_idx:
            continue
        aligned[i] = align_point_clouds(ref, samples[i])
    return aligned


def _gaussian_density(
    points: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma: float,
    point_chunk: int = 256,
) -> np.ndarray:
    """Sum normalised 2-D isotropic Gaussians at ``points`` over the grid.

    Returns an array of shape ``(len(grid_y), len(grid_x))`` whose integral
    over ``(x, y)`` equals ``len(points)`` (i.e. each Gaussian integrates to 1).
    """
    if points.size == 0:
        return np.zeros((grid_y.size, grid_x.size), dtype=np.float32)

    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")  # both (H, W)
    out = np.zeros_like(X, dtype=np.float32)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    norm = 1.0 / (2.0 * np.pi * sigma * sigma)

    for start in range(0, points.shape[0], point_chunk):
        chunk = points[start : start + point_chunk]  # (K, 2)
        dx = X[..., None] - chunk[:, 0]  # (H, W, K)
        dy = Y[..., None] - chunk[:, 1]
        g = np.exp(-(dx * dx + dy * dy) * inv_two_sigma_sq)
        out += g.sum(axis=-1).astype(np.float32)

    out *= norm
    return out


def compute_spatial_probability_map(
    predictions_csv: Path,
    cell_class: str,
    gaussian_sigma: float = 0.02,
    grid_resolution: int = 256,
    grid_margin: float = 0.05,
    align_reference: int = 0,
) -> dict:
    """Return the mean spatial probability map for ``cell_class``.

    Pipeline:
      1. Load every sample's predicted positions.
      2. Align every sample to ``samples[align_reference]`` using an optimal
         rotation (full point cloud, no scaling/translation beyond centering).
      3. Per sample, take the cells of ``cell_class``, place a Gaussian of
         std ``gaussian_sigma`` at each, sum them and divide by the number
         of cells of that class (so each sample's distribution integrates to 1).
      4. Average across samples → final probability map.

    Returns a dict with keys ``density``, ``grid_x``, ``grid_y``, ``extent``,
    ``reference_points``, and ``num_samples`` ready to feed into ``plot_density``.
    """
    df = load_predictions_csv(predictions_csv)
    samples = _predictions_to_array(df)
    first_sample_df = (
        df[df["sample_index"] == df["sample_index"].min()]
        .sort_values("cell_ID", kind="stable")
    )
    cell_class_per_row = (
        df.sort_values(["sample_index", "cell_ID"], kind="stable")
        .drop_duplicates(subset=["cell_ID"])
        .sort_values("cell_ID", kind="stable")["cell_class"]
        .to_numpy()
    )
    # ``_predictions_to_array`` already enforces consistent ordering, so the
    # row-wise class array above (taken from the first sample) is valid for
    # every sample. Recompute the mask explicitly from the first sample to be
    # safe against any reordering surprises.
    first_sample_classes = first_sample_df["cell_class"].to_numpy()
    class_mask = first_sample_classes == cell_class
    n_class_cells = int(class_mask.sum())
    if n_class_cells == 0:
        unique = sorted(set(cell_class_per_row))
        raise ValueError(
            f"Cell class {cell_class!r} not found in predictions. "
            f"Available (first 30): {unique[:30]}"
        )

    if not (0 <= align_reference < samples.shape[0]):
        raise ValueError(
            f"align_reference={align_reference} is out of range "
            f"[0, {samples.shape[0]})."
        )

    aligned = _align_samples(samples, reference_idx=align_reference)

    # Bounding box from the reference sample's class cells (with margin),
    # falling back to the full reference cloud when there is only one cell.
    ref_class_points = aligned[align_reference][class_mask]
    if ref_class_points.shape[0] >= 2:
        ref_for_bbox = ref_class_points
    else:
        ref_for_bbox = aligned[align_reference]

    xmin, ymin = ref_for_bbox.min(axis=0) - grid_margin
    xmax, ymax = ref_for_bbox.max(axis=0) + grid_margin
    grid_x = np.linspace(xmin, xmax, grid_resolution, dtype=np.float32)
    grid_y = np.linspace(ymin, ymax, grid_resolution, dtype=np.float32)

    density = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
    n_samples = aligned.shape[0]
    for s in tqdm(range(n_samples), desc="Density aggregation", unit="sample"):
        pts = aligned[s][class_mask]
        if pts.shape[0] == 0:
            continue
        per_sample = _gaussian_density(pts, grid_x, grid_y, gaussian_sigma)
        per_sample /= float(pts.shape[0])  # normalise by # cells in class
        density += per_sample
    density /= float(n_samples)  # mean over samples

    extent = (float(xmin), float(xmax), float(ymin), float(ymax))
    return {
        "density": density,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "extent": extent,
        "reference_points": ref_class_points,
        "reference_all_points": aligned[align_reference],
        "reference_cell_ids": first_sample_df["cell_ID"].astype(str).to_numpy(),
        "reference_cell_classes": first_sample_classes.astype(str),
        "align_reference": align_reference,
        "num_samples": n_samples,
        "num_class_cells": n_class_cells,
        "cell_class": cell_class,
        "gaussian_sigma": gaussian_sigma,
    }


def plot_density(
    result: dict,
    save_path: Path,
    overlay_reference_scatter: bool = False,
    title: Optional[str] = None,
) -> None:
    """Save the mean spatial probability map produced by ``compute_spatial_probability_map``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density = result["density"]
    extent = result["extent"]
    cell_class = result["cell_class"]
    n_samples = result["num_samples"]
    n_class_cells = result["num_class_cells"]
    sigma = result["gaussian_sigma"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        density,
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
        interpolation="bilinear",
    )
    if overlay_reference_scatter:
        ref_pts = result["reference_points"]
        if ref_pts.shape[0] > 0:
            ax.scatter(
                ref_pts[:, 0],
                ref_pts[:, 1],
                s=4,  # size of the dots; decrease for smaller points
                color="cyan",  # use filled dots, not line circles
                marker="o",    # full circle marker
                label="reference sample (class cells)",
            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
      

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability density (per unit area)")

    if title is None:
        title = (
            f"Spatial probability map — class {cell_class!r}\n"
            f"{n_samples} samples × {n_class_cells} cells, sigma={sigma:g}"
        )
    ax.set_title(title)
    ax.set_xlabel("X (normalised)")
    ax.set_ylabel("Y (normalised)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[diffusion2spatial_probs] Saved spatial probability map → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_path(p: str | os.PathLike, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def main() -> None:
    args = _parse_args()
    cfg = load_testing_config(Path(args.config), args.overrides)
    OmegaConf = _load_omegaconf()

    # Resolve filesystem paths up front, relative to the repo root.
    output_dir = _resolve_path(cfg.output_dir, REPO_ROOT) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = _resolve_path(cfg.predictions_csv, output_dir)
    gt_csv = _resolve_path(cfg.ground_truth_csv, output_dir)
    plot_path = _resolve_path(cfg.analysis.plot_filename, output_dir)
    density_npy = (
        _resolve_path(cfg.analysis.density_npy_filename, output_dir)
        if cfg.analysis.density_npy_filename
        else None
    )

    print("=" * 78)
    print("LUNA spatial probability-map experiment")
    print(f"  experiment dir : {output_dir}")
    print(f"  cell_section   : {cfg.cell_section!r}")
    print(f"  num_samples    : {cfg.num_samples}")
    print(f"  seed_start     : {cfg.seed_start}")
    print(f"  cell_class     : {cfg.analysis.cell_class!r}")
    print("=" * 78)

    # Persist the merged effective config for reproducibility.
    OmegaConf.save(cfg, output_dir / "effective_config.yaml")

    if cfg.run_sampling:
        import torch

        from datasets.data_module import DataModule, Infos

        if not cfg.checkpoint_path:
            raise ValueError("checkpoint_path must be set when run_sampling=true.")

        luna_cfg = compose_luna_config(list(cfg.luna_overrides))
        luna_cfg.test.checkpoint_path = cfg.checkpoint_path
        luna_cfg.general.local_saved_path = str(output_dir)

        # Allow the testing config to override dataset paths so the user does
        # not have to edit the LUNA experiment file just to swap an input CSV.
        ds_overrides = cfg.get("dataset", None)
        if ds_overrides is not None:
            for key, value in OmegaConf.to_container(ds_overrides, resolve=True).items():
                if value is None:
                    continue
                if key not in luna_cfg.dataset:
                    raise KeyError(
                        f"dataset.{key} from testing config is not a known field of "
                        f"the LUNA dataset config. Known fields: {list(luna_cfg.dataset.keys())}"
                    )
                print(
                    f"[diffusion2spatial_probs] Overriding dataset.{key} = {value!r} "
                    "(from utils/testing/config.yaml)"
                )
                luna_cfg.dataset[key] = value

        print(f"[diffusion2spatial_probs] Using test data CSV: {luna_cfg.dataset.test_data_path}")

        device = _resolve_device(str(cfg.device))
        print(f"[diffusion2spatial_probs] Device: {device}")

        print("[diffusion2spatial_probs] Building data module…")
        datamodule = DataModule(luna_cfg)
        dataset_infos = Infos(datamodule, luna_cfg)

        slice_batch = build_section_batch(datamodule, dataset_infos, cfg.cell_section)

        print(
            f"[diffusion2spatial_probs] Loading checkpoint from {cfg.checkpoint_path}"
        )
        model = load_model(luna_cfg, dataset_infos, cfg.checkpoint_path, device)

        preds = run_inferences(
            model=model,
            batch=slice_batch,
            num_samples=int(cfg.num_samples),
            seed_start=int(cfg.seed_start),
            device=device,
        )

        save_predictions_csv(
            preds=preds,
            batch=slice_batch,
            seed_start=int(cfg.seed_start),
            output_csv=predictions_csv,
            gt_csv=gt_csv,
        )

        # Release the model & datamodule before the analysis phase to free GPU/CPU memory.
        del model, datamodule, dataset_infos, slice_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(
            "[diffusion2spatial_probs] run_sampling=false — using existing predictions "
            f"at {predictions_csv}"
        )

    if not cfg.run_analysis:
        print("[diffusion2spatial_probs] run_analysis=false — done.")
        return

    if not cfg.analysis.cell_class:
        raise ValueError("analysis.cell_class must be set when run_analysis=true.")

    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"Cannot run analysis: predictions CSV not found at {predictions_csv}."
        )

    result = compute_spatial_probability_map(
        predictions_csv=predictions_csv,
        cell_class=str(cfg.analysis.cell_class),
        gaussian_sigma=float(cfg.analysis.gaussian_sigma),
        grid_resolution=int(cfg.analysis.grid_resolution),
        grid_margin=float(cfg.analysis.grid_margin),
        align_reference=int(cfg.analysis.align_reference),
    )

    plot_density(
        result=result,
        save_path=plot_path,
        overlay_reference_scatter=bool(cfg.analysis.overlay_reference_scatter),
    )

    if density_npy is not None:
        np.savez(
            density_npy.with_suffix(".npz"),
            density=result["density"],
            grid_x=result["grid_x"],
            grid_y=result["grid_y"],
            extent=np.asarray(result["extent"]),
            reference_points=result["reference_points"],
            cell_class=result["cell_class"],
            gaussian_sigma=result["gaussian_sigma"],
            num_samples=result["num_samples"],
            num_class_cells=result["num_class_cells"],
        )
        print(
            f"[diffusion2spatial_probs] Saved raw density grid → "
            f"{density_npy.with_suffix('.npz')}"
        )


if __name__ == "__main__":
    main()
