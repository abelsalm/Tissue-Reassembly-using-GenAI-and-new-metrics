#!/usr/bin/env python3
"""Compute overall MMD and multi-radius transcriptome loss terms from pred/GT CSVs.

Uses the same hyperparameters as ``configs/train/default.yaml`` (via OmegaConf)
and the same modules as training:

  * ``metrics.train_mmds.SlideMMDLoss``
  * ``metrics.train_spatial_transcriptomics.MultiRadiusNeighborhoodLoss``

For each slice in ``gt_pred_vanilla/`` (paired ``*_metadata_pred.csv`` /
``*_metadata_true.csv``), builds ``DataHolder``s and evaluates both losses.
Gene features are looked up from the MERFISH CSVs by cell ID (CSV index),
matching ``datasets/data_module.py`` (sorted gene columns).

Aggregation matches WandB / Lightning epoch logging: mean over slices
(``on_epoch=True`` averages per-batch scalars).

Usage::

    conda activate LUNA
    python script_test_mmd_transcriptomics_from_csv.py
    python script_test_mmd_transcriptomics_from_csv.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.train_mmds import SlideMMDLoss
from metrics.train_spatial_transcriptomics import MultiRadiusNeighborhoodLoss
from utils.data.dataholder import DataHolder

DEFAULT_CSV_DIR = REPO_ROOT / "gt_pred_vanilla"
DEFAULT_TRAIN_CFG = REPO_ROOT / "configs" / "train" / "default.yaml"
DEFAULT_EXPERIMENT_CFG = (
    REPO_ROOT / "configs" / "experiment" / "MERFISH_small_transcripts.yaml"
)


def _get(cfg, key: str, default=None):
    v = OmegaConf.select(cfg, key, default=None)
    return default if v is None else v


def build_mmd_loss(train_cfg) -> SlideMMDLoss:
    return SlideMMDLoss(train_cfg)


def build_transcriptome_loss(train_cfg) -> MultiRadiusNeighborhoodLoss:
    return MultiRadiusNeighborhoodLoss(
        radii=list(_get(train_cfg, "multi_radius_radii", [0.02, 0.04, 0.08, 0.16])),
        avg_transcriptome_weight=float(
            _get(train_cfg, "multi_radius_avg_transcriptome_weight", 1.0)
        ),
        density_weight=float(_get(train_cfg, "multi_radius_density_weight", 1.0)),
        global_transcriptome_weight=float(
            _get(train_cfg, "multi_radius_global_transcriptome_weight", 0.0)
        ),
        loss_radius_scale=float(
            _get(train_cfg, "multi_radius_loss_radius_scale", 512.0)
        ),
        transcriptome_tolerance=_get(
            train_cfg, "multi_radius_transcriptome_tolerance", 0.05
        ),
        transcriptome_tolerance_gate_beta=_get(
            train_cfg, "multi_radius_transcriptome_tolerance_soft_beta", 256.0
        ),
        transcriptome_tolerance_warmup_epochs=int(
            _get(train_cfg, "multi_radius_transcriptome_tolerance_warmup_epochs", 100)
        ),
        soft_beta=_get(train_cfg, "multi_radius_soft_beta", None),
        eps=float(_get(train_cfg, "multi_radius_eps", 1e-6)),
        include_self=bool(_get(train_cfg, "multi_radius_include_self", True)),
        cache_gt=bool(_get(train_cfg, "neighborhood_cache_gt", True)),
    )


def list_slice_pairs(csv_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Return ``(slice_stem, pred_path, true_path)`` for complete pairs only."""
    pairs: List[Tuple[str, Path, Path]] = []
    for pred_path in sorted(csv_dir.glob("*_metadata_pred.csv")):
        stem = pred_path.name[: -len("_metadata_pred.csv")]
        true_path = csv_dir / f"{stem}_metadata_true.csv"
        if not true_path.is_file():
            print(f"[skip] missing GT for {stem}")
            continue
        pairs.append((stem, pred_path, true_path))
    for true_path in sorted(csv_dir.glob("*_metadata_true.csv")):
        stem = true_path.name[: -len("_metadata_true.csv")]
        pred_path = csv_dir / f"{stem}_metadata_pred.csv"
        if not pred_path.is_file():
            print(f"[skip] missing pred for {stem}")
    return pairs


def load_gene_table(
    data_paths: List[Path],
    gene_columns_start: int,
    gene_columns_end: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load MERFISH CSV(s) and return gene features indexed by cell ID.

    Gene column order matches ``AbstractDataset.filter_genes`` (slice then sort).
    """
    frames = []
    gene_names: Optional[List[str]] = None
    for path in data_paths:
        header = pd.read_csv(path, nrows=0, index_col=0)
        cols = list(header.columns)
        names = list(cols[gene_columns_start:gene_columns_end])
        names_sorted = sorted(names)
        if gene_names is None:
            gene_names = names_sorted
        elif gene_names != names_sorted:
            raise ValueError(f"Gene column mismatch in {path}")
        dtype_hints = {c: np.float32 for c in names}
        # Read only index + gene columns (avoid loading coord / metadata).
        df = pd.read_csv(path, index_col=0, dtype=dtype_hints)
        frames.append(df[names_sorted])
    assert gene_names is not None
    table = pd.concat(frames, axis=0)
    table = table[~table.index.duplicated(keep="first")]
    return table, gene_names


def load_slice_holders(
    pred_path: Path,
    true_path: Path,
    gene_table: pd.DataFrame,
    device: torch.device,
) -> Tuple[DataHolder, DataHolder, int]:
    pred_df = pd.read_csv(pred_path, index_col=0)
    true_df = pd.read_csv(true_path, index_col=0)

    # Align on shared cell IDs (pred/GT should match; keep GT order).
    shared = true_df.index.intersection(pred_df.index)
    if len(shared) == 0:
        raise ValueError(f"No shared cell IDs between {pred_path.name} and {true_path.name}")
    if len(shared) < len(true_df) or len(shared) < len(pred_df):
        print(
            f"  [warn] {true_path.stem}: using {len(shared)} shared IDs "
            f"(gt={len(true_df)}, pred={len(pred_df)})"
        )
    true_df = true_df.loc[shared]
    pred_df = pred_df.loc[shared]

    missing_genes = shared.difference(gene_table.index)
    if len(missing_genes) > 0:
        raise KeyError(
            f"{true_path.name}: {len(missing_genes)} cell IDs missing from gene table "
            f"(e.g. {list(missing_genes[:3])})"
        )
    feats = gene_table.loc[shared]

    n = len(true_df)
    true_xy = true_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    pred_xy = pred_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    feat_np = feats.to_numpy(dtype=np.float32)

    # Integer class labels (same encoding on pred/GT; MMD only needs equality).
    class_codes, _ = pd.factorize(true_df["cell_class"].astype(str), sort=True)
    cell_ids = np.asarray(shared, dtype=np.int64)

    positions_true = torch.as_tensor(true_xy, device=device).unsqueeze(0)
    positions_pred = torch.as_tensor(pred_xy, device=device).unsqueeze(0)
    node_features = torch.as_tensor(feat_np, device=device).unsqueeze(0)
    node_mask = torch.ones((1, n), device=device, dtype=torch.bool)
    cell_class = torch.as_tensor(class_codes, device=device, dtype=torch.long).view(1, n, 1)
    cell_ID = torch.as_tensor(cell_ids, device=device, dtype=torch.long).view(1, n, 1)

    true_holder = DataHolder(
        positions=positions_true,
        node_features=node_features,
        diffusion_time=None,
        cell_ID=cell_ID,
        cell_class=cell_class,
        node_mask=node_mask,
    ).mask()
    pred_holder = DataHolder(
        positions=positions_pred,
        node_features=node_features,
        diffusion_time=None,
        cell_ID=cell_ID,
        cell_class=cell_class,
        node_mask=node_mask,
    ).mask()
    return pred_holder, true_holder, n


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MMD + transcriptome loss terms on gt_pred CSV pairs."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Folder with *_metadata_pred.csv / *_metadata_true.csv pairs.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CFG,
        help="Train YAML (MMD + transcriptome hyperparameters).",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CFG,
        help="Experiment YAML (gene column range + MERFISH CSV paths).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-slice loss values.",
    )
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.train_config)
    exp_cfg = OmegaConf.load(args.experiment_config)

    mmd_weight = float(_get(train_cfg, "mmd_weight", 1.0))
    tx_weight = float(_get(train_cfg, "transcriptome_multi_radius_weight", 1.0))

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    gene_start = int(exp_cfg.dataset.gene_columns_start)
    gene_end = int(exp_cfg.dataset.gene_columns_end)
    data_paths = []
    for key in ("test_data_path", "train_data_path", "validation_data_path"):
        p = Path(str(exp_cfg.dataset[key]))
        if p.is_file() and p not in data_paths:
            data_paths.append(p)
    if not data_paths:
        raise FileNotFoundError("No MERFISH gene CSVs found from experiment config.")

    print(f"Train config: {args.train_config}")
    print(f"CSV dir:      {args.csv_dir}")
    print(f"Device:       {device}")
    print(f"Gene sources: {[str(p) for p in data_paths]}")
    print(
        f"Weights:      mmd_weight={mmd_weight}, "
        f"transcriptome_multi_radius_weight={tx_weight}"
    )

    print("Loading gene feature table …")
    gene_table, gene_names = load_gene_table(data_paths, gene_start, gene_end)
    print(f"  {len(gene_table):,} cells, {len(gene_names)} genes")

    mmd_loss = build_mmd_loss(train_cfg).to(device).eval()
    tx_loss = build_transcriptome_loss(train_cfg).to(device).eval()
    # warmup_epochs=0 in default.yaml → full tolerance band (forgiveness=1).
    tx_loss.set_current_epoch(0)

    pairs = list_slice_pairs(args.csv_dir)
    if not pairs:
        raise RuntimeError(f"No complete pred/GT pairs in {args.csv_dir}")

    mmd_values: List[float] = []
    tx_values: List[float] = []

    for stem, pred_path, true_path in pairs:
        pred_h, true_h, n_cells = load_slice_holders(
            pred_path, true_path, gene_table, device
        )
        m_val, _ = mmd_loss(pred_h, true_h, train_stage=False, log=False)
        t_val, _ = tx_loss(pred_h, true_h, train_stage=False, log=False)
        m_f = float(m_val.detach().item())
        t_f = float(t_val.detach().item())
        mmd_values.append(m_f)
        tx_values.append(t_f)
        if args.verbose:
            print(
                f"  {stem}: n={n_cells}  "
                f"mmd={m_f:.6f}  transcriptome={t_f:.6f}"
            )

    # Mean over slices == Lightning ``on_epoch=True`` aggregation of batch scalars.
    mean_mmd = float(np.mean(mmd_values))
    mean_tx = float(np.mean(tx_values))
    n = len(mmd_values)

    print()
    print(f"Slices evaluated: {n}")
    print(f"Overall MMD term (mean over slices):            {mean_mmd:.8f}")
    print(f"Overall transcriptome term (mean over slices):  {mean_tx:.8f}")
    if mmd_weight != 1.0 or tx_weight != 1.0:
        print(
            f"Weighted (as in combined loss): "
            f"mmd={mmd_weight * mean_mmd:.8f}, "
            f"transcriptome={tx_weight * mean_tx:.8f}"
        )


if __name__ == "__main__":
    main()
