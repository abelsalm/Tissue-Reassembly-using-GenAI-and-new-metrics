"""Fast Gaussian kNN CME for MERFISH ABC (cell-type neighborhood mix).

Uses coordinates only to *precompute* features; the domain model never sees x/y.
Truncated Gaussian over K nearest neighbors (default K=64) is equivalent to the
folder's radius-Gaussian CME when K covers the 3σ ball.

Usage (repo root, LUNA)::

    python exploration/cell_types/experiments/precompute_abc_cme.py --sigma 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exploration.cell_types.ct_soft_cme import (  # noqa: E402
    presence_scale_from_mass,
    resolve_class_vocab,
    save_soft_cme,
    load_csv_for_cme,
)


def knn_gaussian_cme(
    positions: np.ndarray,
    class_ids: np.ndarray,
    num_classes: int,
    sigma: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = positions.shape[0]
    mass = np.zeros((n, num_classes), dtype=np.float64)
    soft = np.zeros((n, num_classes), dtype=np.float32)
    if n == 0:
        return soft, mass.astype(np.float32)
    kk = min(int(k) + 1, n)
    tree = cKDTree(positions)
    dists, idx = tree.query(positions, k=kk)
    if kk == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    # drop self (first neighbor)
    dists = np.asarray(dists[:, 1:], dtype=np.float64)
    idx = np.asarray(idx[:, 1:], dtype=np.int64)
    inv = 1.0 / (2.0 * sigma * sigma)
    w = np.exp(-(dists ** 2) * inv)
    neigh_cls = class_ids[idx]
    for c in range(num_classes):
        mass[:, c] = (w * (neigh_cls == c)).sum(axis=1)
    denom = mass.sum(axis=1, keepdims=True)
    empty = denom[:, 0] <= 0
    denom = np.clip(denom, 1e-12, None)
    soft = (mass / denom).astype(np.float32)
    if empty.any():
        soft[empty] = 1.0 / float(num_classes)
    return soft, mass.astype(np.float32)


def compute_split(df: pd.DataFrame, sigma: float, k: int, class_names):
    class_ids, int_to_class, vocab = resolve_class_vocab(
        df["cell_class"].astype(str).tolist(), class_names=class_names
    )
    class_ids_arr = np.asarray(class_ids, dtype=np.int64)
    num_classes = len(vocab)
    soft = np.zeros((len(df), num_classes), dtype=np.float32)
    mass = np.zeros((len(df), num_classes), dtype=np.float32)
    sections = df["cell_section"].astype(str).to_numpy()
    for section in pd.unique(sections):
        mask = sections == section
        idx = np.flatnonzero(mask)
        pos = df.loc[mask, ["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)
        s, m = knn_gaussian_cme(
            pos, class_ids_arr[idx], num_classes, sigma, k
        )
        soft[idx] = s
        mass[idx] = m
        print(f"  section={section!r} n={len(idx)}", flush=True)
    scale = presence_scale_from_mass(mass)
    return {
        "soft_cme": soft,
        "cme_mass": mass,
        "presence_scale": np.asarray(scale, dtype=np.float64),
        "cell_id": df.index.to_numpy(),
        "cell_section": sections,
        "class_id": class_ids_arr,
        "class_names": vocab,
        "int_to_class": {int(k): str(v) for k, v in int_to_class.items()},
        "sigma": float(sigma),
        "exclude_self": True,
        "cutoff_radius": float(3.0 * sigma),
        "num_classes": int(num_classes),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(
        Path(__file__).resolve().parents[1] / "ct_config_domain.json"
    ))
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--k", type=int, default=64)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "outputs" / "abc_soft_cmes"
        ),
    )
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    ds = cfg["dataset"]
    out_dir = Path(args.out_dir)
    train_df = load_csv_for_cme(ds["train_data_path"])
    _, _, vocab = resolve_class_vocab(train_df["cell_class"].astype(str).tolist())
    splits = [
        ("train", ds["train_data_path"]),
        ("validation", ds["validation_data_path"]),
        ("test", ds["test_data_path"]),
    ]
    seen = {}
    for split, path in splits:
        if path in seen:
            src = seen[path]
            dst = out_dir / src.name.replace(src.name.split("_")[0], split, 1)
            # reuse via save of loaded arrays is handled below by recompute skip
            import shutil

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"[abc_cme] copied {src.name} -> {dst.name}")
            continue
        print(f"[abc_cme] split={split} sigma={args.sigma} k={args.k}")
        df = train_df if path == ds["train_data_path"] else load_csv_for_cme(path)
        result = compute_split(df, args.sigma, args.k, vocab)
        npz, _ = save_soft_cme(result, out_dir, split)
        seen[path] = npz


if __name__ == "__main__":
    main()
