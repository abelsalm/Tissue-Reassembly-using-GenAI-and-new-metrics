"""Train a single-head ``CellTypeTransformer`` on MERFISH ABC spatial domains.

Labels come from ``dataset.label_column`` (default ``spatial_module_l1_complete``).
Each distinct value starting with ``shared`` is its own class; every other
label (ABC-only modules, NaNs, …) is mapped to one extra ``other`` class.

Usage (from repo root, LUNA env)::

    python exploration/cell_types/ct_train_domains.py
    python exploration/cell_types/ct_train_domains.py \\
        --config exploration/cell_types/ct_config_domain.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUTPUT_DIR = "exploration/cell_types/outputs"
DEFAULT_LOG_DIR = "exploration/cell_types/logs"
DEFAULT_LABEL_COLUMN = "spatial_module_l1_complete"
DEFAULT_SHARED_PREFIX = "shared"
DEFAULT_OTHER_NAME = "other"

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import (  # noqa: E402
    class_indices_from_batch,
    get_loss,
)
from exploration.cell_types.ct_transformer import CellTypeTransformer  # noqa: E402
from utils.data.misc import to_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train domain (spatial-module) classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("ct_config_domain.json")),
        help="Path to ct_config_domain.json",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def resolve_repo_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


class _TeeStream:
    """Mirror writes to several text streams (e.g. console + log file)."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.streams[0], "isatty", lambda: False)())


def setup_train_log_dir(cfg: Dict[str, Any]) -> Path:
    log_root = resolve_repo_path(cfg.get("log_dir", DEFAULT_LOG_DIR))
    run_name = str(cfg.get("run_name", "ct_transformer_abc_domains"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_run_dir = log_root / run_name / stamp
    log_run_dir.mkdir(parents=True, exist_ok=True)

    log_file = (log_run_dir / "train.log").open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.__stderr__, log_file)  # type: ignore[assignment]

    def _restore_stdio() -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

    atexit.register(_restore_stdio)
    return log_run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


class ModelEMA:
    """Exponential moving average of trainable parameters."""

    def __init__(self, model: CellTypeTransformer, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: CellTypeTransformer) -> None:
        decay = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    @torch.no_grad()
    def apply_shadow(self, model: CellTypeTransformer) -> None:
        self._backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: CellTypeTransformer) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self._backup[name])
        self._backup = {}


def build_scheduler(optimizer: AdamW, cfg: Dict[str, Any], n_epochs: int):
    tcfg = cfg.get("train") or {}
    scfg = tcfg.get("scheduler") or {}
    name = str(scfg.get("name", "none")).strip().lower()
    if name in ("", "none", "null", "off"):
        return None
    if name in ("cosine", "cosine_warmup"):
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        min_lr = float(scfg.get("min_lr", 1e-6))
        if scfg.get("t_max") is not None:
            cosine_epochs = max(1, int(scfg["t_max"]))
        else:
            cosine_epochs = max(1, n_epochs - warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=min_lr
        )
        if warmup_epochs <= 0:
            return cosine
        warmup = LinearLR(
            optimizer,
            start_factor=float(scfg.get("warmup_start_factor", 0.01)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return ReduceLROnPlateau(
            optimizer,
            mode=str(scfg.get("mode", "max")),
            factor=float(scfg.get("factor", 0.5)),
            patience=int(scfg.get("patience", 8)),
            min_lr=float(scfg.get("min_lr", 1e-6)),
            threshold=float(scfg.get("threshold", 0.001)),
        )
    raise ValueError(f"Unknown scheduler '{name}'")


def inverse_freq_weights(
    class_ids: torch.Tensor, num_classes: int, power: float = 1.0
) -> List[float]:
    counts = torch.bincount(class_ids.reshape(-1).long(), minlength=num_classes)
    counts = counts.float().clamp_min(1.0)
    w = (counts.sum() / (float(num_classes) * counts)).pow(float(power))
    w = w / w.mean()
    return [float(x) for x in w.tolist()]


def _id_index_size(cell_id: torch.Tensor) -> int:
    return int(cell_id.max().item()) + 1


def build_type_lookup(
    datamodule: DataModule,
) -> Tuple[Dict[str, torch.Tensor], int, List[str]]:
    """Map ``cell_ID`` codes → train cell-type ids (0 = UNK)."""
    train_ds = datamodule.train_dataset
    decoder = dict(train_ds.statistics.cell_class_decoder)
    names = [decoder[i] for i in range(len(decoder))]
    n_types = len(names)
    tables: Dict[str, torch.Tensor] = {}
    datasets = {
        "train": datamodule.train_dataset,
        "validation": getattr(datamodule, "validation_dataset", None),
        "test": getattr(datamodule, "test_dataset", None),
    }
    train_name_to_id = {str(n): i + 1 for i, n in enumerate(names)}
    for split, ds in datasets.items():
        if ds is None:
            continue
        codes = ds._data.cell_ID
        orig = ds._data.cell_class
        table = torch.zeros(_id_index_size(codes), dtype=torch.long)
        if split == "train":
            mapped = orig.long() + 1
        else:
            split_dec = dict(ds.statistics.cell_class_decoder)
            mapped = torch.zeros_like(orig)
            for src_i, name in split_dec.items():
                mapped[orig == int(src_i)] = int(train_name_to_id.get(str(name), 0))
        table[codes] = mapped
        tables[split] = table
    return tables, n_types, names


def build_cme_lookup(
    datamodule: DataModule,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Load precomputed Gaussian CME keyed by original cell id → ``cell_ID`` table."""
    acfg = cfg.get("aux_inputs") or {}
    out_dir = resolve_repo_path(
        acfg.get("cme_dir", "exploration/cell_types/outputs/abc_soft_cmes")
    )
    sigma_list = acfg.get("cme_sigmas")
    if sigma_list:
        sigmas = [float(s) for s in sigma_list]
    else:
        sigmas = [float(acfg.get("cme_sigma", 0.25))]
    from exploration.cell_types.ct_soft_cme import _sigma_tag

    split_files = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    datasets = {
        "train": datamodule.train_dataset,
        "validation": getattr(datamodule, "validation_dataset", None),
        "test": getattr(datamodule, "test_dataset", None),
    }
    tables: Dict[str, torch.Tensor] = {}
    cme_dim = 0
    loaded_by_path: Dict[str, Dict[str, Any]] = {}
    dcfg = cfg["dataset"]
    path_by_split = {
        "train": str(dcfg["train_data_path"]),
        "validation": str(dcfg.get("validation_data_path") or ""),
        "test": str(dcfg.get("test_data_path") or ""),
    }
    from exploration.cell_types.ct_soft_cme import _sigma_tag

    for split, ds in datasets.items():
        if ds is None:
            continue
        file_split = split_files[split]
        csv_path = path_by_split[split]
        feats = []
        orig_ids = np.asarray(ds._cell_ids_clean).astype(str)
        codes = ds._data.cell_ID
        for sigma in sigmas:
            cache_key = f"{csv_path}::{sigma}"
            if cache_key in loaded_by_path:
                loaded = loaded_by_path[cache_key]
            else:
                npz_path = out_dir / f"{file_split}_sigma{_sigma_tag(sigma)}.npz"
                if not npz_path.exists():
                    raise FileNotFoundError(
                        f"Missing ABC CME for split={file_split} sigma={sigma} "
                        f"in {out_dir}. Precompute with experiments/precompute_abc_cme.py"
                    )
                packed = np.load(npz_path, allow_pickle=True)
                loaded = {
                    "soft_cme": packed["soft_cme"],
                    "cell_id": packed["cell_id"],
                }
                loaded_by_path[cache_key] = loaded
            soft = np.asarray(loaded["soft_cme"], dtype=np.float32)
            cell_id = np.asarray(loaded["cell_id"])
            dim = int(soft.shape[1])
            by_id = pd.DataFrame(soft, index=pd.Index(cell_id.astype(str)))
            aligned = by_id.reindex(pd.Index(orig_ids))
            if aligned.isna().any().any():
                aligned = aligned.fillna(1.0 / max(dim, 1))
            feats.append(torch.from_numpy(aligned.to_numpy(dtype=np.float32)))
        feat = torch.cat(feats, dim=-1)
        cme_dim = int(feat.size(-1))
        table = torch.zeros(_id_index_size(codes), cme_dim, dtype=torch.float32)
        table[codes] = feat
        tables[split] = table
    print(f"[ct_domains] CME sigmas={sigmas} dim={cme_dim}", flush=True)
    return tables, cme_dim


def attach_aux_to_batch(
    batch,
    device: torch.device,
    type_table: Optional[torch.Tensor],
    cme_table: Optional[torch.Tensor],
) -> None:
    if batch.cell_ID is None:
        return
    ids = batch.cell_ID
    if ids.dim() == 3:
        ids = ids.squeeze(-1)
    ids = ids.long().clamp(min=0)
    if type_table is not None:
        t = type_table.to(device)
        ids_t = ids.clamp(max=t.size(0) - 1)
        batch.cell_type = t[ids_t]
    if cme_table is not None:
        c = cme_table.to(device)
        ids_c = ids.clamp(max=c.size(0) - 1)
        batch.cme_features = c[ids_c]


def compact_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    return {
        "loss": float(metrics["loss"]),
        "acc": float(metrics["acc"]),
        "n_cells": float(metrics["n_cells"]),
    }


def best_per_class_payload(
    *,
    epoch: int,
    split: str,
    class_names: List[str],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compact per-class table for the current best checkpoint (one write)."""
    accs = list(metrics.get("per_class_acc") or [])
    ns = list(metrics.get("per_class_n") or [])
    rows: List[Dict[str, Any]] = []
    defined: List[float] = []
    zeros: List[str] = []
    for i, name in enumerate(class_names):
        acc = accs[i] if i < len(accs) else None
        n = int(ns[i]) if i < len(ns) else 0
        rows.append({"class": name, "acc": acc, "n": n})
        if acc is None or n <= 0:
            continue
        defined.append(float(acc))
        if float(acc) <= 0.0:
            zeros.append(name)
    macro = (sum(defined) / len(defined)) if defined else None
    return {
        "epoch": int(epoch),
        "split": str(split),
        "micro_acc": float(metrics.get("acc", 0.0)),
        "macro_acc": macro,
        "n_classes": len(rows),
        "n_zero_acc": len(zeros),
        "zero_classes": zeros,
        "per_class": rows,
    }


def write_best_per_class(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def print_best_per_class_summary(payload: Dict[str, Any]) -> None:
    rows = list(payload.get("per_class") or [])
    zeros = list(payload.get("zero_classes") or [])
    macro = payload.get("macro_acc")
    macro_s = f"{macro:.4f}" if isinstance(macro, float) else "n/a"
    print(
        f"[ct_domains] best {payload.get('split', 'val')} per-class "
        f"(epoch={payload.get('epoch')}, micro={payload.get('micro_acc'):.4f}, "
        f"macro={macro_s}): {payload.get('n_zero_acc', 0)}/"
        f"{payload.get('n_classes', 0)} classes at 0",
        flush=True,
    )
    if zeros:
        print(
            f"[ct_domains] zero-acc classes: {', '.join(zeros)}",
            flush=True,
        )
    scored = [r for r in rows if r.get("acc") is not None and int(r.get("n") or 0) > 0]
    scored.sort(key=lambda r: float(r["acc"]))
    if scored:
        worst = ", ".join(
            f"{r['class']}={float(r['acc']):.3f}(n={int(r['n'])})"
            for r in scored[:8]
        )
        print(f"[ct_domains] hardest classes: {worst}", flush=True)


def build_datamodule_cfg(cfg: Dict[str, Any]):
    return OmegaConf.create(
        {
            "general": {
                "seed": int(cfg.get("seed", 0)),
                "name": cfg.get("run_name", "domains"),
            },
            "dataset": cfg["dataset"],
            "train": {"batch_size": int(cfg["train"]["batch_size"])},
            "validation": {
                "batch_size": int(
                    cfg.get("validation", {}).get(
                        "batch_size", cfg["train"]["batch_size"]
                    )
                )
            },
            "test": {
                "batch_size": int(
                    cfg.get("test", {}).get("batch_size", cfg["train"]["batch_size"])
                )
            },
        }
    )


def load_label_series(csv_path: Union[str, Path], label_column: str) -> pd.Series:
    """Index-aligned label column; CSV first column is the cell id."""
    csv_path = Path(csv_path)
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    if not header:
        raise ValueError(f"Empty CSV: {csv_path}")
    if label_column not in header:
        raise ValueError(
            f"Column {label_column!r} not in {csv_path} "
            f"(have {header[:12]}{'…' if len(header) > 12 else ''})"
        )
    id_col = header[0]
    df = pd.read_csv(csv_path, usecols=[id_col, label_column], index_col=id_col)
    return df[label_column]


def build_shared_plus_other_vocab(
    labels: pd.Series,
    *,
    prefix: str = DEFAULT_SHARED_PREFIX,
    other_name: str = DEFAULT_OTHER_NAME,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """``shared_*`` classes (sorted) + one catch-all ``other`` class."""
    raw = labels.astype(str)
    shared = sorted({v for v in raw.unique() if v.startswith(prefix)})
    if other_name in shared:
        raise ValueError(
            f"other_class_name {other_name!r} collides with a shared label"
        )
    class_names = shared + [other_name]
    name_to_id = {name: i for i, name in enumerate(class_names)}
    decoder = {i: name for i, name in enumerate(class_names)}
    return class_names, name_to_id, decoder


def labels_to_class_ids(
    labels: pd.Series,
    name_to_id: Dict[str, int],
    other_name: str,
) -> np.ndarray:
    other_id = int(name_to_id[other_name])
    raw = labels.astype(str).to_numpy()
    mapped = np.full(len(raw), other_id, dtype=np.int64)
    for name, idx in name_to_id.items():
        if name == other_name:
            continue
        mapped[raw == name] = int(idx)
    return mapped


def apply_domain_labels(
    datamodule: DataModule,
    cfg: Dict[str, Any],
) -> Tuple[int, List[str], Dict[int, str]]:
    """Overwrite ``cell_class`` with shared-prefix + other domain ids.

    Vocab is built from the **train** CSV so val/test extra modules collapse
    into ``other``.
    """
    dcfg = cfg["dataset"]
    label_col = str(dcfg.get("label_column", DEFAULT_LABEL_COLUMN))
    prefix = str(dcfg.get("shared_label_prefix", DEFAULT_SHARED_PREFIX))
    other_name = str(dcfg.get("other_class_name", DEFAULT_OTHER_NAME))

    series_by_path: Dict[str, pd.Series] = {}

    def series_for(path: str) -> pd.Series:
        key = str(Path(path).resolve())
        if key not in series_by_path:
            print(f"[ct_domains] loading labels {label_col} from {path}", flush=True)
            series_by_path[key] = load_label_series(path, label_col)
        return series_by_path[key]

    train_series = series_for(str(dcfg["train_data_path"]))
    class_names, name_to_id, decoder = build_shared_plus_other_vocab(
        train_series, prefix=prefix, other_name=other_name
    )
    num_classes = len(class_names)
    n_shared = num_classes - 1
    print(
        f"[ct_domains] domain vocab: {n_shared} '{prefix}*' classes + "
        f"1 '{other_name}'  (C={num_classes})",
        flush=True,
    )

    split_paths = {
        "train": str(dcfg["train_data_path"]),
        "validation": str(dcfg.get("validation_data_path") or ""),
        "test": str(dcfg.get("test_data_path") or ""),
    }
    datasets = {
        "train": datamodule.train_dataset,
        "validation": getattr(datamodule, "validation_dataset", None),
        "test": getattr(datamodule, "test_dataset", None),
    }

    for split, ds in datasets.items():
        if ds is None:
            continue
        path = split_paths.get(split) or ""
        if not path:
            raise ValueError(f"No CSV path for split={split}")
        series = series_for(path)
        cell_ids = np.asarray(ds._cell_ids_clean)
        lookup = series.copy()
        lookup.index = lookup.index.map(str)
        id_index = pd.Index(np.asarray(cell_ids).astype(str))
        aligned = lookup.reindex(id_index)
        in_csv = id_index.isin(lookup.index)
        n_unmatched = int((~in_csv).sum())
        n_nan_label = int((in_csv & aligned.isna()).sum())
        if n_unmatched:
            print(
                f"[ct_domains] split={split}: {n_unmatched} cell ids "
                f"not in CSV → '{other_name}'",
                flush=True,
            )
        if n_nan_label:
            print(
                f"[ct_domains] split={split}: {n_nan_label} NaN "
                f"{label_col} → '{other_name}'",
                flush=True,
            )
        aligned = aligned.fillna(other_name)

        class_ids = labels_to_class_ids(aligned, name_to_id, other_name)
        counts = np.bincount(class_ids, minlength=num_classes)
        n_other = int(counts[-1])
        print(
            f"[ct_domains] split={split}: n={len(class_ids)}  "
            f"other={n_other} ({n_other / max(len(class_ids), 1):.1%})",
            flush=True,
        )
        ds._data.cell_class = torch.from_numpy(class_ids)
        ds.num_cell_class = num_classes
        ds.statistics.num_cell_class = num_classes
        ds.statistics.cell_class_decoder = decoder

    datamodule.statistics["train"].num_cell_class = num_classes
    datamodule.statistics["train"].cell_class_decoder = decoder
    for split in ("validation", "test"):
        stats = datamodule.statistics.get(split)
        if stats is not None:
            stats.num_cell_class = num_classes
            stats.cell_class_decoder = decoder

    return num_classes, class_names, decoder


def build_model(
    cfg: Dict[str, Any],
    gene_dim: int,
    num_classes: int,
    *,
    num_cell_types: int = 0,
    cme_input_dim: int = 0,
) -> CellTypeTransformer:
    mcfg = cfg["model"]
    acfg = cfg.get("aux_inputs") or {}
    use_type = bool(acfg.get("cell_type", False))
    use_cme = bool(acfg.get("cme", False))
    return CellTypeTransformer(
        gene_dim=gene_dim,
        num_classes=num_classes,
        n_layers=int(mcfg.get("n_layers", 4)),
        hidden_mlp_dims=dict(mcfg["hidden_mlp_dims"]),
        hidden_dims=dict(mcfg["hidden_dims"]),
        dropout_cls=float(mcfg.get("dropout_cls", 0.1)),
        dropout_layer=float(mcfg.get("dropout_layer", 0.1)),
        gene_mlp_activation=mcfg.get("gene_mlp_activation", "relu"),
        gene_mlp_depth=int(mcfg.get("gene_mlp_depth", 2)),
        layer_activation=str(mcfg.get("layer_activation", "relu")),
        cls_activation=str(mcfg.get("cls_activation", "relu")),
        global_skip=mcfg.get("global_skip", None),
        subgraph_summary=bool(mcfg.get("subgraph_summary", False)),
        subgraph_pool=str(mcfg.get("subgraph_pool", "mean")),
        subgraph_pool_source=str(mcfg.get("subgraph_pool_source", "mixed")),
        attention_use_node_mask=bool(mcfg.get("attention_use_node_mask", False)),
        gene_knn_k=int(mcfg.get("gene_knn_k", 0) or 0),
        pre_norm=bool(mcfg.get("pre_norm", False)),
        predict_cme=False,
        feature_cross=False,
        use_cell_type_input=use_type,
        num_cell_types=int(num_cell_types if use_type else 0),
        cell_type_embed_dim=int(acfg.get("type_embed_dim", mcfg.get("type_embed_dim", 32))),
        use_cme_input=use_cme,
        cme_input_dim=int(cme_input_dim if use_cme else 0),
        aux_fusion=str(acfg.get("fusion", mcfg.get("aux_fusion", "add"))),
        aux_inject=str(acfg.get("inject", mcfg.get("aux_inject", "input"))),
        type_dropout=float(acfg.get("type_dropout", 0.0)),
        n_domain_queries=int(mcfg.get("n_domain_queries", acfg.get("n_domain_queries", 0)) or 0),
        bag_type_hist=bool(mcfg.get("bag_type_hist", acfg.get("bag_type_hist", False))),
        aux_token_mode=str(mcfg.get("aux_token_mode", acfg.get("aux_token_mode", "off"))),
        n_encoder_queries=int(mcfg.get("n_encoder_queries", acfg.get("n_encoder_queries", 0)) or 0),
        dual_stream=bool(mcfg.get("dual_stream", acfg.get("dual_stream", False))),
        attn_kind=str(mcfg.get("attn_kind", acfg.get("attn_kind", "linear"))),
        graph_k=int(mcfg.get("graph_k", acfg.get("graph_k", 32)) or 32),
        graph_on=str(mcfg.get("graph_on", acfg.get("graph_on", "cme"))),
        cme_n_scales=int(
            mcfg.get(
                "cme_n_scales",
                acfg.get(
                    "cme_n_scales",
                    len(acfg.get("cme_sigmas") or [acfg.get("cme_sigma", 0.25)]),
                ),
            )
            or 1
        ),
    )


def setup_wandb_run(
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    log_dir: Path,
    gene_dim: int,
    num_classes: int,
    n_params: int,
    class_names: List[str],
) -> bool:
    wcfg = cfg.get("wandb") or {}
    if not bool(wcfg.get("enabled", False)):
        print("[ct_domains] wandb disabled")
        return False

    mode = str(wcfg.get("mode", "online"))
    init_kwargs: Dict[str, Any] = {
        "project": str(wcfg.get("project", "cell_types")),
        "name": str(cfg.get("run_name", "ct_transformer_abc_domains")),
        "config": {
            **cfg,
            "resolved": {
                "gene_dim": gene_dim,
                "num_classes": num_classes,
                "n_params": n_params,
                "class_names": class_names,
                "run_dir": str(run_dir),
                "log_dir": str(log_dir),
            },
        },
        "dir": str(log_dir),
        "mode": mode,
        "reinit": True,
    }
    entity = wcfg.get("entity", None)
    if entity:
        init_kwargs["entity"] = str(entity)
    tags = wcfg.get("tags", None)
    if tags:
        init_kwargs["tags"] = list(tags)

    wandb.init(**init_kwargs)
    wandb.save(str(log_dir / "ct_config.snapshot.json"), policy="now")
    print(
        f"[ct_domains] wandb: project={init_kwargs['project']} "
        f"name={init_kwargs['name']} mode={mode}"
        + (f" url={wandb.run.url}" if wandb.run is not None else "")
    )
    return True


def wandb_log(payload: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if wandb.run is None:
        return
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def log_model_parameter_counts(model: CellTypeTransformer) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    print(
        f"[ct_domains] parameters: total={total:,} trainable={trainable:,}",
        flush=True,
    )
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count == 0:
            continue
        rows.append([name, count])
        print(f"  [params] {name}: {count:,}", flush=True)

    if wandb.run is None:
        return
    table = wandb.Table(data=rows, columns=["component", "parameters"])
    wandb_log(
        {
            "model/parameter_count": total,
            "model/trainable_parameter_count": trainable,
            "model/parameters_by_component": wandb.plot.bar(
                table,
                "component",
                "parameters",
                title="Model parameters by component",
            ),
        }
    )


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, batch) -> Tuple[float, int]:
    mask = batch.node_mask.bool()
    targets = class_indices_from_batch(batch)
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == targets[mask]).sum().item()
    total = int(mask.sum().item())
    if total == 0:
        return 0.0, 0
    return correct / total, total


@torch.no_grad()
def update_per_class_counts(
    logits: torch.Tensor,
    batch,
    correct_counts: torch.Tensor,
    total_counts: torch.Tensor,
) -> None:
    mask = batch.node_mask.bool()
    targets = class_indices_from_batch(batch)[mask]
    preds = logits.argmax(dim=-1)[mask]
    if targets.numel() == 0:
        return
    num_classes = correct_counts.numel()
    targets_cpu = targets.detach().cpu()
    correct_cpu = preds.detach().cpu() == targets_cpu
    total_counts += torch.bincount(targets_cpu, minlength=num_classes)
    if correct_cpu.any():
        correct_counts += torch.bincount(
            targets_cpu[correct_cpu], minlength=num_classes
        )


def per_class_accuracy_list(
    correct_counts: torch.Tensor, total_counts: torch.Tensor
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for c in range(correct_counts.numel()):
        n = int(total_counts[c].item())
        if n == 0:
            out.append(None)
        else:
            out.append(float(correct_counts[c].item()) / float(n))
    return out


def per_class_acc_wandb_payload(
    split: str,
    per_class_acc: List[Optional[float]],
    class_names: List[str],
) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for c, name in enumerate(class_names):
        if c < len(per_class_acc) and per_class_acc[c] is not None:
            safe = str(name).replace("/", "_")
            payload[f"{split}/acc_per_class/{safe}"] = float(per_class_acc[c])
    return payload


def run_epoch(
    model: CellTypeTransformer,
    loader,
    loss_fn,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[AdamW] = None,
    grad_clip: Optional[float] = None,
    log_every: int = 0,
    epoch: int = 0,
    split: str = "train",
    global_step_offset: int = 0,
    wandb_log_steps: bool = False,
    type_table: Optional[torch.Tensor] = None,
    cme_table: Optional[torch.Tensor] = None,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    n_batches = 0
    correct_counts = torch.zeros(num_classes, dtype=torch.long)
    total_counts = torch.zeros(num_classes, dtype=torch.long)

    for step, raw in enumerate(loader):
        batch = to_batch(raw, device=device)
        attach_aux_to_batch(batch, device, type_table, cme_table)
        outputs = model(batch)
        if not isinstance(outputs, dict):
            outputs = {"cls_logits": outputs}

        loss = loss_fn(outputs, batch)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        logits = outputs["cls_logits"].detach()
        acc, n = masked_accuracy(logits, batch)
        update_per_class_counts(logits, batch, correct_counts, total_counts)
        loss_val = float(loss.detach().item())
        total_loss += loss_val
        total_correct += int(acc * n)
        total_cells += n
        n_batches += 1

        if log_every and train and (step + 1) % log_every == 0:
            print(
                f"  [{split}] epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss_val:.4f} acc={acc:.4f}",
                flush=True,
            )
            if wandb_log_steps:
                wandb_log(
                    {
                        f"{split}/step_loss": loss_val,
                        f"{split}/step_acc": acc,
                        "epoch": epoch,
                    },
                    step=global_step_offset + step + 1,
                )

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_correct / max(total_cells, 1),
        "n_cells": float(total_cells),
        "per_class_acc": per_class_accuracy_list(correct_counts, total_counts),
        "per_class_n": [int(x) for x in total_counts.tolist()],
    }


def save_checkpoint(
    path: Path,
    model: CellTypeTransformer,
    optimizer: AdamW,
    epoch: int,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    class_names: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "gene_dim": model.gene_dim,
            "num_classes": model.num_classes,
            "class_names": class_names,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "cuda")))

    out_dir = resolve_repo_path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    run_dir = out_dir / str(cfg.get("run_name", "ct_transformer_abc_domains"))
    run_dir.mkdir(parents=True, exist_ok=True)

    log_run_dir = setup_train_log_dir(cfg)
    with (log_run_dir / "ct_config.snapshot.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (run_dir / "ct_config.snapshot.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[ct_domains] config={args.config}")
    print(f"[ct_domains] device={device}  checkpoints={run_dir}")
    print(f"[ct_domains] logs={log_run_dir}")

    dm_cfg = build_datamodule_cfg(cfg)
    print("[ct_domains] Loading data …")
    datamodule = DataModule(dm_cfg)

    acfg = cfg.get("aux_inputs") or {}
    use_type = bool(acfg.get("cell_type", False))
    use_cme = bool(acfg.get("cme", False))
    type_tables: Dict[str, torch.Tensor] = {}
    cme_tables: Dict[str, torch.Tensor] = {}
    num_cell_types = 0
    cme_dim = 0
    if use_type:
        type_tables, num_cell_types, type_names = build_type_lookup(datamodule)
        print(
            f"[ct_domains] cell-type input ON  T={num_cell_types} "
            f"(embed + inject={acfg.get('inject', 'input')}/"
            f"{acfg.get('fusion', 'add')})",
            flush=True,
        )
        print(f"[ct_domains] type vocab sample: {type_names[:6]}…", flush=True)

    num_classes, class_names, decoder = apply_domain_labels(datamodule, cfg)

    if use_cme:
        cme_tables, cme_dim = build_cme_lookup(datamodule, cfg)
        sigmas_print = acfg.get("cme_sigmas") or [acfg.get("cme_sigma", 0.25)]
        print(
            f"[ct_domains] CME input ON  dim={cme_dim}  sigmas={sigmas_print}",
            flush=True,
        )

    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(infos.num_genes)
    infos.num_cell_class = num_classes
    infos.cell_class_decoder = decoder
    print(
        f"[ct_domains] genes={gene_dim}  domain_classes={num_classes} "
        f"({class_names[-1]!r} is the catch-all)",
        flush=True,
    )

    model = build_model(
        cfg,
        gene_dim=gene_dim,
        num_classes=num_classes,
        num_cell_types=num_cell_types,
        cme_input_dim=cme_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ct_domains] single-head classifier  params={n_params:,}  "
        f"attn={getattr(model, 'attn_kind', 'linear')}  "
        f"graph_k={getattr(model, 'graph_k', 0)}  "
        f"graph_on={getattr(model, 'graph_on', 'cme')}  "
        f"cme_n_scales={getattr(model, 'cme_n_scales', 1)}",
        flush=True,
    )

    use_wandb = setup_wandb_run(
        cfg,
        run_dir=run_dir,
        log_dir=log_run_dir,
        gene_dim=gene_dim,
        num_classes=num_classes,
        n_params=n_params,
        class_names=class_names,
    )
    log_model_parameter_counts(model)
    wcfg = cfg.get("wandb") or {}
    wandb_log_steps = use_wandb and bool(wcfg.get("log_every_steps", False))
    log_per_class = use_wandb and bool(wcfg.get("log_per_class_acc", False))

    loss_cfg = dict(cfg.get("loss", {}))
    loss_name = str(loss_cfg.pop("name", "cross_entropy"))
    cw = loss_cfg.get("class_weight")
    if isinstance(cw, str):
        key = cw.lower().strip()
        power_by_name = {
            "inverse_freq": 1.0,
            "auto": 1.0,
            "balanced": 1.0,
            "inverse_sqrt": 0.5,
        }
        if key not in power_by_name:
            raise ValueError(
                f"Unknown class_weight {cw!r}. Use inverse_freq, inverse_sqrt, "
                "or a list of C floats."
            )
        w = inverse_freq_weights(
            datamodule.train_dataset._data.cell_class,
            num_classes,
            power=power_by_name[key],
        )
        loss_cfg["class_weight"] = w
        print(
            f"[ct_domains] class_weight={key} min={min(w):.3f} max={max(w):.3f}",
            flush=True,
        )
    loss_fn = get_loss(loss_name, **loss_cfg)
    tcfg = cfg["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, cfg, int(tcfg["n_epochs"]))
    if scheduler is not None:
        print(
            f"[ct_domains] scheduler={((tcfg.get('scheduler') or {}).get('name'))}",
            flush=True,
        )
    ema_cfg = tcfg.get("ema") or {}
    ema: Optional[ModelEMA] = None
    if bool(ema_cfg.get("enabled", False)):
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.98)))
        print(f"[ct_domains] EMA enabled decay={ema.decay}", flush=True)

    train_loader = datamodule.train_dataloader()
    val_loader = None
    if getattr(datamodule, "validation_dataset", None) is not None:
        val_loader = datamodule.validation_dataloader()

    type_train = type_tables.get("train")
    type_val = type_tables.get("validation")
    cme_train = cme_tables.get("train")
    cme_val = cme_tables.get("validation")

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    history: List[Dict[str, Any]] = []
    n_epochs = int(tcfg["n_epochs"])
    val_every = int(tcfg.get("val_every_n_epochs", 1))
    save_every = int(tcfg.get("save_every_n_epochs", 10))
    log_every = int(tcfg.get("log_every", 0))
    grad_clip = tcfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    rechunk_every = int(tcfg.get("rechunk_every_n_epochs", 0))
    es_cfg = tcfg.get("early_stopping") or {}
    early_stop = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 20))
    es_monitor = str(es_cfg.get("monitor", "val/acc"))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    global_step = 0
    epoch = 0
    train_metrics: Dict[str, Any] = {}
    val_metrics: Optional[Dict[str, Any]] = None

    try:
        for epoch in range(1, n_epochs + 1):
            if (
                rechunk_every > 0
                and hasattr(datamodule, "train_dataset")
                and hasattr(datamodule.train_dataset, "rechunk")
                and (epoch == 1 or (epoch - 1) % rechunk_every == 0)
            ):
                datamodule.train_dataset.rechunk(seed=epoch)
                print(
                    f"[ct_domains] rechunked train graphs (seed={epoch})",
                    flush=True,
                )

            print(f"\n=== Epoch {epoch}/{n_epochs} ===", flush=True)
            train_metrics = run_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                num_classes=num_classes,
                optimizer=optimizer,
                grad_clip=grad_clip,
                log_every=log_every,
                epoch=epoch,
                split="train",
                global_step_offset=global_step,
                wandb_log_steps=wandb_log_steps,
                type_table=type_train,
                cme_table=cme_train,
                ema=ema,
            )
            global_step += len(train_loader)
            print(
                f"[train] loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} "
                f"cells={int(train_metrics['n_cells'])}",
                flush=True,
            )

            val_metrics = None
            if val_loader is not None and (epoch % val_every == 0):
                if ema is not None:
                    ema.apply_shadow(model)
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    num_classes=num_classes,
                    optimizer=None,
                    epoch=epoch,
                    split="val",
                    type_table=type_val,
                    cme_table=cme_val,
                )
                if ema is not None:
                    ema.restore(model)
                print(
                    f"[val]   loss={val_metrics['loss']:.4f} "
                    f"acc={val_metrics['acc']:.4f} "
                    f"cells={int(val_metrics['n_cells'])}",
                    flush=True,
                )
                improved = False
                if val_metrics["acc"] > best_val_acc + (
                    es_min_delta if es_monitor == "val/acc" else 0.0
                ):
                    best_val_acc = val_metrics["acc"]
                    if es_monitor == "val/acc":
                        improved = True
                elif val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]
                if val_metrics["loss"] < best_val_loss - (
                    es_min_delta if es_monitor == "val/loss" else 0.0
                ):
                    best_val_loss = val_metrics["loss"]
                    if es_monitor == "val/loss":
                        improved = True
                elif val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]

                if val_metrics["acc"] >= best_val_acc:
                    if ema is not None:
                        ema.apply_shadow(model)
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        epoch,
                        cfg,
                        {
                            "train": compact_metrics(train_metrics),
                            "val": compact_metrics(val_metrics),
                        },
                        class_names,
                    )
                    if ema is not None:
                        ema.restore(model)
                    print(
                        f"[ct_domains] saved best checkpoint "
                        f"(val_acc={best_val_acc:.4f})",
                        flush=True,
                    )
                    best_pc = best_per_class_payload(
                        epoch=epoch,
                        split="val",
                        class_names=class_names,
                        metrics=val_metrics,
                    )
                    write_best_per_class(
                        log_run_dir / "best_per_class.json", best_pc
                    )
                    write_best_per_class(run_dir / "best_per_class.json", best_pc)
                    if use_wandb:
                        wandb_log(
                            {
                                "best_val_acc": best_val_acc,
                                "best_val_loss": best_val_loss,
                                "best_epoch": epoch,
                            },
                            step=global_step,
                        )

                if early_stop:
                    if improved:
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1
                        print(
                            f"[ct_domains] early_stopping: no improve on "
                            f"{es_monitor} for {epochs_without_improve}/"
                            f"{es_patience} val checks",
                            flush=True,
                        )

            epoch_log: Dict[str, Any] = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train/loss": train_metrics["loss"],
                "train/acc": train_metrics["acc"],
                "train/n_cells": train_metrics["n_cells"],
            }
            if val_metrics is not None:
                epoch_log.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/acc": val_metrics["acc"],
                        "val/n_cells": val_metrics["n_cells"],
                    }
                )
            if log_per_class:
                epoch_log.update(
                    per_class_acc_wandb_payload(
                        "train", train_metrics["per_class_acc"], class_names
                    )
                )
                if val_metrics is not None:
                    epoch_log.update(
                        per_class_acc_wandb_payload(
                            "val", val_metrics["per_class_acc"], class_names
                        )
                    )
            wandb_log(epoch_log, step=global_step)

            history.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "train": compact_metrics(train_metrics),
                    "val": compact_metrics(val_metrics),
                }
            )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    metric = (
                        val_metrics["acc"]
                        if val_metrics is not None
                        else train_metrics["acc"]
                    )
                    scheduler.step(metric)
                else:
                    scheduler.step()

            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    {
                        "train": compact_metrics(train_metrics),
                        "val": compact_metrics(val_metrics),
                    },
                    class_names,
                )

            with (log_run_dir / "history.json").open("w") as f:
                json.dump(history, f)

            if (
                early_stop
                and val_metrics is not None
                and epochs_without_improve >= es_patience
            ):
                print(
                    f"[ct_domains] early stopping at epoch {epoch} "
                    f"(patience={es_patience}, monitor={es_monitor})",
                    flush=True,
                )
                break

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            cfg,
            {
                "train": train_metrics,
                "val": val_metrics,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
            },
            class_names,
        )
        if use_wandb:
            wandb_log(
                {
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                step=global_step,
            )
            wandb.summary["best_val_acc"] = best_val_acc
            wandb.summary["best_val_loss"] = best_val_loss
        best_pc_path = log_run_dir / "best_per_class.json"
        if best_pc_path.exists():
            best_pc = json.loads(best_pc_path.read_text())
            print_best_per_class_summary(best_pc)
            if use_wandb:
                wandb.summary["best_macro_acc"] = best_pc.get("macro_acc")
                wandb.summary["best_n_zero_acc"] = best_pc.get("n_zero_acc")
                wandb.summary["best_zero_classes"] = best_pc.get("zero_classes")
                wandb_log(
                    per_class_acc_wandb_payload(
                        "best_val",
                        [r.get("acc") for r in best_pc.get("per_class") or []],
                        class_names,
                    ),
                    step=global_step,
                )
        print(
            f"\n[ct_domains] done. best_val_acc={best_val_acc:.4f}  "
            f"checkpoints={run_dir}  logs={log_run_dir}"
        )
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
