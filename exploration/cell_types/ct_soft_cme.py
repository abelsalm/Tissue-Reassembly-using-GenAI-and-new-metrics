"""Precompute soft cell-microenvironment (CME) targets on full tissue slices.

For each cell ``i`` in a real ``cell_section`` (not training subgraph chunks):

    w_{ij} = exp( -||x_i - x_j||^2 / (2 σ^2) )   for neighbors j ≠ i
            (optionally truncated at ``cutoff_radius``)

    soft_cme[i, c] = Σ_{j: class(j)=c} w_{ij}  /  Σ_j w_{ij}

The result is a probability distribution over cell types describing the
Gaussian-weighted neighborhood composition. Arrays are keyed by the CSV
cell index so training can look them up after graph chunking/rechunking.

Save layout (default)::

    exploration/cell_types/outputs/ct_soft_cmes/
        meta_sigma{σ}.json
        train_sigma{σ}.npz
        validation_sigma{σ}.npz
        test_sigma{σ}.npz

Usage (from repo root)::

    python exploration/cell_types/ct_soft_cme.py \\
        --config exploration/cell_types/ct_config.json \\
        --sigma 50

    python exploration/cell_types/ct_soft_cme.py \\
        --csv /path/to/train.csv --split train --sigma 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.data.load import character_to_int, standardise_dataframe_colnames  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "ct_soft_cmes"


def _sigma_tag(sigma: float) -> str:
    """Filename-safe sigma tag (``50`` or ``50p5``)."""
    if float(sigma).is_integer():
        return str(int(sigma))
    return str(sigma).replace(".", "p")


def resolve_class_vocab(
    class_labels: Sequence[Any],
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[List[int], Dict[int, str], List[str]]:
    """Map string labels → ints with a fixed vocabulary."""
    if class_names is None:
        uniques = sorted({str(x) for x in class_labels})
    else:
        uniques = [str(x) for x in class_names]
    ints, int_to_class = character_to_int(list(class_labels), uniques)
    if any(v is None for v in ints):
        missing = sorted(
            {str(x) for x, v in zip(class_labels, ints) if v is None}
        )
        raise ValueError(
            f"Labels not in class vocabulary: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return [int(v) for v in ints], int_to_class, uniques


def soft_cme_gaussian_section(
    positions: np.ndarray,
    class_ids: np.ndarray,
    num_classes: int,
    sigma: float,
    *,
    exclude_self: bool = True,
    cutoff_radius: Optional[float] = None,
) -> np.ndarray:
    """Soft CME for all cells in one section.

    Args:
        positions: ``(N, 2)`` coordinates (same units as ``sigma``).
        class_ids: ``(N,)`` ints in ``[0, num_classes)``.
        num_classes: ``C``.
        sigma: Gaussian length-scale.
        exclude_self: If True, cell ``i`` is not used in its own CME.
        cutoff_radius: Neighbor radius; default ``3 * sigma``.

    Returns:
        ``(N, C)`` float32 rows that sum to 1 (uniform if no neighbors).
    """
    positions = np.asarray(positions, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2), got {positions.shape}")
    if class_ids.shape[0] != positions.shape[0]:
        raise ValueError("class_ids length must match positions")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    n = positions.shape[0]
    out = np.zeros((n, num_classes), dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    if cutoff_radius is None:
        cutoff_radius = float(3.0 * sigma)
    cutoff_radius = float(cutoff_radius)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    tree = cKDTree(positions)
    neighbor_lists = tree.query_ball_point(positions, r=cutoff_radius)

    for i, neigh in enumerate(neighbor_lists):
        if exclude_self:
            neigh = [j for j in neigh if j != i]
        if not neigh:
            out[i] = 1.0 / float(num_classes)
            continue

        neigh_arr = np.asarray(neigh, dtype=np.int64)
        delta = positions[neigh_arr] - positions[i]
        d2 = np.einsum("ij,ij->i", delta, delta)
        w = np.exp(-d2 * inv_two_sigma2)
        w_sum = float(w.sum())
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            out[i] = 1.0 / float(num_classes)
            continue

        for j_idx, weight in zip(neigh_arr, w):
            out[i, class_ids[j_idx]] += float(weight)
        out[i] /= w_sum

    return out.astype(np.float32)


def compute_soft_cme_dataframe(
    df: pd.DataFrame,
    sigma: float,
    *,
    class_names: Optional[Sequence[str]] = None,
    exclude_self: bool = True,
    cutoff_radius: Optional[float] = None,
    id_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute soft CME on every full ``cell_section`` in ``df``.

    Expects columns ``coord_X``, ``coord_Y``, ``cell_class``, ``cell_section``
    (raw / standardised). Uses the DataFrame index as ``cell_id`` unless
    ``id_column`` is provided.
    """
    data = standardise_dataframe_colnames(df.copy())
    required = ("coord_X", "coord_Y", "cell_class", "cell_section")
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns {missing}")

    cell_ids = (
        data[id_column].to_numpy()
        if id_column is not None
        else data.index.to_numpy()
    )

    class_ids, int_to_class, vocab = resolve_class_vocab(
        data["cell_class"].astype(str).tolist(), class_names=class_names
    )
    class_ids_arr = np.asarray(class_ids, dtype=np.int64)
    num_classes = len(vocab)

    soft = np.zeros((len(data), num_classes), dtype=np.float32)
    sections = data["cell_section"].astype(str).to_numpy()
    n_empty = 0

    for section in pd.unique(sections):
        mask = sections == section
        idx = np.flatnonzero(mask)
        pos = data.loc[mask, ["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)
        soft[idx] = soft_cme_gaussian_section(
            pos,
            class_ids_arr[idx],
            num_classes=num_classes,
            sigma=sigma,
            exclude_self=exclude_self,
            cutoff_radius=cutoff_radius,
        )
        # Uniform rows ⇒ no neighbors within cutoff (uninformative target).
        row_max = soft[idx].max(axis=1)
        n_empty_sec = int(np.isclose(row_max, 1.0 / num_classes).sum())
        n_empty += n_empty_sec
        print(
            f"  [soft_cme] section={section!r}  n={len(idx)}  sigma={sigma}  "
            f"empty_neigh={n_empty_sec}/{len(idx)}",
            flush=True,
        )

    frac_empty = n_empty / max(len(data), 1)
    if frac_empty > 0.5:
        cut = float(cutoff_radius if cutoff_radius is not None else 3.0 * sigma)
        print(
            f"[soft_cme] WARNING: {frac_empty:.1%} cells have empty neighborhoods "
            f"(fallback=uniform 1/C). Soft CME is uninformative — soft-CE floor "
            f"stays near log(C)≈{np.log(num_classes):.3f}. "
            f"Increase sigma (coords ~1e3–1e4, median NN≈15) or cutoff "
            f"(current cutoff={cut}).",
            flush=True,
        )

    return {
        "soft_cme": soft,
        "cell_id": cell_ids,
        "cell_section": sections,
        "class_id": class_ids_arr,
        "class_names": vocab,
        "int_to_class": {int(k): str(v) for k, v in int_to_class.items()},
        "sigma": float(sigma),
        "exclude_self": bool(exclude_self),
        "cutoff_radius": float(
            cutoff_radius if cutoff_radius is not None else 3.0 * sigma
        ),
        "num_classes": int(num_classes),
    }


def save_soft_cme(
    result: Dict[str, Any],
    out_dir: Union[str, Path],
    split: str,
) -> Tuple[Path, Path]:
    """Write ``{split}_sigma{σ}.npz`` and update ``meta_sigma{σ}.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _sigma_tag(float(result["sigma"]))
    npz_path = out_dir / f"{split}_sigma{tag}.npz"
    meta_path = out_dir / f"meta_sigma{tag}.json"

    np.savez_compressed(
        npz_path,
        soft_cme=np.asarray(result["soft_cme"], dtype=np.float32),
        cell_id=np.asarray(result["cell_id"]),
        cell_section=np.asarray(result["cell_section"]).astype(str),
        class_id=np.asarray(result["class_id"], dtype=np.int64),
    )

    meta = {
        "sigma": float(result["sigma"]),
        "exclude_self": bool(result["exclude_self"]),
        "cutoff_radius": float(result["cutoff_radius"]),
        "num_classes": int(result["num_classes"]),
        "class_names": list(result["class_names"]),
        "int_to_class": {
            str(k): v for k, v in dict(result["int_to_class"]).items()
        },
        "splits": {},
    }
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            if prev.get("class_names") == meta["class_names"]:
                meta["splits"] = dict(prev.get("splits") or {})
        except (json.JSONDecodeError, OSError):
            pass

    meta["splits"][split] = {
        "npz": npz_path.name,
        "n_cells": int(len(result["soft_cme"])),
        "n_sections": int(len(set(map(str, result["cell_section"])))),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[soft_cme] wrote {npz_path}  ({meta['splits'][split]['n_cells']} cells)")
    print(f"[soft_cme] wrote {meta_path}")
    return npz_path, meta_path


def load_soft_cme(
    out_dir: Union[str, Path],
    split: str,
    sigma: float,
) -> Dict[str, Any]:
    """Load precomputed soft CME for ``split`` / ``sigma``.

    Includes ``by_cell_id``: ``dict[cell_id -> (C,) float32]`` for training
    lookup after chunking.
    """
    out_dir = Path(out_dir)
    tag = _sigma_tag(float(sigma))
    npz_path = out_dir / f"{split}_sigma{tag}.npz"
    meta_path = out_dir / f"meta_sigma{tag}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing soft CME file: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    soft = data["soft_cme"]
    cell_id = data["cell_id"]
    if np.issubdtype(cell_id.dtype, np.integer):
        by_cell_id = {int(cell_id[i]): soft[i] for i in range(len(cell_id))}
    else:
        by_cell_id = {cell_id[i]: soft[i] for i in range(len(cell_id))}

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return {
        "soft_cme": soft,
        "cell_id": cell_id,
        "cell_section": data["cell_section"],
        "class_id": data["class_id"],
        "by_cell_id": by_cell_id,
        "meta": meta,
        "npz_path": npz_path,
        "sigma": float(sigma),
        "split": split,
    }


def load_csv_for_cme(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load spatial + label columns; keep CSV index as cell id."""
    csv_path = Path(csv_path)
    # Include the index column explicitly in ``usecols`` so pandas does not
    # promote ``coord_X`` (or another kept column) to the index.
    all_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if not all_cols:
        raise ValueError(f"Empty CSV: {csv_path}")
    id_col = all_cols[0]
    aliases = {
        "coord_X",
        "coord_Y",
        "x",
        "y",
        "cell_class",
        "subclass",
        "cell_section",
        "region",
    }
    keep = [id_col] + [c for c in all_cols[1:] if c in aliases]
    df = pd.read_csv(csv_path, usecols=keep, index_col=0)
    return standardise_dataframe_colnames(df)


def precompute_from_config(
    config_path: Union[str, Path],
    sigma: float,
    *,
    out_dir: Optional[Union[str, Path]] = None,
    splits: Optional[Sequence[str]] = None,
    exclude_self: bool = True,
    cutoff_radius: Optional[float] = None,
) -> Dict[str, Path]:
    """Precompute soft CME for splits listed in ``ct_config.json``."""
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    ds = cfg["dataset"]
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR

    split_to_key = {
        "train": "train_data_path",
        "validation": "validation_data_path",
        "val": "validation_data_path",
        "test": "test_data_path",
    }
    if splits is None:
        splits = ["train", "validation", "test"]

    train_df = load_csv_for_cme(ds["train_data_path"])
    _, _, vocab = resolve_class_vocab(train_df["cell_class"].astype(str).tolist())

    written: Dict[str, Path] = {}
    path_to_npz: Dict[str, Path] = {}

    for split in splits:
        key = split_to_key.get(split)
        if key is None or key not in ds or not ds[key]:
            print(f"[soft_cme] skip split={split} (no path)")
            continue
        split_name = "validation" if split == "val" else split
        csv_path = str(ds[key])

        if csv_path in path_to_npz:
            src = path_to_npz[csv_path]
            dst = out_dir / f"{split_name}_sigma{_sigma_tag(sigma)}.npz"
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            loaded = np.load(dst, allow_pickle=False)
            tag = _sigma_tag(sigma)
            meta_path = out_dir / f"meta_sigma{tag}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta.setdefault("splits", {})
            meta["splits"][split_name] = {
                "npz": dst.name,
                "n_cells": int(len(loaded["soft_cme"])),
                "n_sections": int(len(set(map(str, loaded["cell_section"])))),
                "copied_from": src.name,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            written[split_name] = dst
            print(f"[soft_cme] reused {src.name} -> {dst.name}")
            continue

        print(f"[soft_cme] computing split={split_name}  csv={csv_path}")
        df = (
            train_df
            if csv_path == str(ds["train_data_path"])
            else load_csv_for_cme(csv_path)
        )
        result = compute_soft_cme_dataframe(
            df,
            sigma,
            class_names=vocab,
            exclude_self=exclude_self,
            cutoff_radius=cutoff_radius,
        )
        npz_path, _ = save_soft_cme(result, out_dir, split_name)
        written[split_name] = npz_path
        path_to_npz[csv_path] = npz_path

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute soft Gaussian CME targets")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("ct_config.json")),
        help="ct_config.json (dataset paths)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Gaussian σ in coordinate units (median NN ≈ 15 on this MERFISH set)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for npz/meta",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,validation,test",
        help="Comma-separated splits to precompute",
    )
    p.add_argument(
        "--cutoff-radius",
        type=float,
        default=None,
        help="Neighbor radius (default: 3*sigma)",
    )
    p.add_argument(
        "--include-self",
        action="store_true",
        help="Include the cell itself in its CME (default: excluded)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional single CSV (overrides --config splits)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name when using --csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude_self = not bool(args.include_self)
    out_dir = Path(args.out_dir)

    if args.csv:
        print(
            f"[soft_cme] single csv={args.csv} split={args.split} "
            f"sigma={args.sigma}"
        )
        df = load_csv_for_cme(args.csv)
        result = compute_soft_cme_dataframe(
            df,
            float(args.sigma),
            exclude_self=exclude_self,
            cutoff_radius=args.cutoff_radius,
        )
        save_soft_cme(result, out_dir, args.split)
        return

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    precompute_from_config(
        args.config,
        float(args.sigma),
        out_dir=out_dir,
        splits=splits,
        exclude_self=exclude_self,
        cutoff_radius=args.cutoff_radius,
    )


if __name__ == "__main__":
    main()
