"""Train ``CellTypeTransformer`` for per-cell type classification.

Usage (from repo root, LUNA env)::

    python exploration/cell_types/ct_train.py
    python exploration/cell_types/ct_train.py --config exploration/cell_types/ct_config.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW

# Repo root on sys.path when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import (  # noqa: E402
    class_indices_from_batch,
    get_loss,
)
from exploration.cell_types.ct_soft_cme import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _sigma_tag,
    load_soft_cme,
    precompute_from_config,
)
from exploration.cell_types.ct_transformer import CellTypeTransformer  # noqa: E402
from utils.data.misc import to_batch  # noqa: E402


class SoftCMELookup:
    """Dense ``cell_id → soft_cme`` table for fast batch gathering."""

    def __init__(self, cell_ids: np.ndarray, soft_cme: np.ndarray):
        cell_ids = np.asarray(cell_ids)
        soft_cme = np.asarray(soft_cme, dtype=np.float32)
        if cell_ids.shape[0] != soft_cme.shape[0]:
            raise ValueError("cell_ids / soft_cme length mismatch")
        if not np.issubdtype(cell_ids.dtype, np.integer):
            raise ValueError("SoftCMELookup expects integer cell_ids")
        max_id = int(cell_ids.max()) if len(cell_ids) else 0
        self.num_classes = int(soft_cme.shape[1])
        self.table = np.zeros((max_id + 1, self.num_classes), dtype=np.float32)
        self.table[cell_ids.astype(np.int64)] = soft_cme
        self.max_id = max_id

    def gather(
        self, cell_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Map ``cell_ids`` ``(B, N)`` or ``(B, N, 1)`` → ``(B, N, C)``."""
        if cell_ids.dim() == 3:
            cell_ids = cell_ids.squeeze(-1)
        ids = cell_ids.detach().cpu().long().numpy()
        ids = np.clip(ids, 0, self.max_id)
        out = self.table[ids]  # (B, N, C)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)


def attach_soft_cme(
    batch, lookup: SoftCMELookup, device: torch.device
):
    """Set ``batch.soft_cme`` from ``batch.cell_ID`` via precomputed table."""
    if batch.cell_ID is None:
        raise ValueError("batch.cell_ID is required to attach soft CME targets")
    batch.soft_cme = lookup.gather(batch.cell_ID, device)
    return batch


def soft_cme_is_needed(cfg: Dict[str, Any]) -> bool:
    scfg = cfg.get("soft_cme") or {}
    mcfg = cfg.get("model") or {}
    lcfg = cfg.get("loss") or {}
    return (
        bool(scfg.get("enabled", False))
        or bool(mcfg.get("predict_cme", False))
        or str(lcfg.get("name", "")) == "combined"
        or float(lcfg.get("cme_weight", 0.0) or 0.0) != 0.0
    )


def soft_cme_out_dir(cfg: Dict[str, Any]) -> Path:
    scfg = cfg.get("soft_cme") or {}
    out_dir = Path(scfg.get("out_dir", DEFAULT_OUT_DIR))
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    return out_dir


def soft_cme_npz_path(out_dir: Path, split: str, sigma: float) -> Path:
    file_split = "validation" if split in ("val", "validation") else split
    return out_dir / f"{file_split}_sigma{_sigma_tag(float(sigma))}.npz"


def ensure_soft_cme_precomputed(
    cfg: Dict[str, Any], config_path: Union[str, Path]
) -> None:
    """If soft-CME npz files are missing, run ``precompute_from_config``.

    Controlled by ``soft_cme.auto_precompute`` (default ``true``).
    """
    if not soft_cme_is_needed(cfg):
        return
    scfg = cfg.get("soft_cme") or {}
    if not bool(scfg.get("auto_precompute", True)):
        return

    out_dir = soft_cme_out_dir(cfg)
    sigma = float(scfg["sigma"])
    splits = ["train", "validation", "test"]
    missing = [
        s for s in splits if not soft_cme_npz_path(out_dir, s, sigma).exists()
    ]
    if not missing:
        return

    exclude_self = bool(scfg.get("exclude_self", True))
    cutoff = scfg.get("cutoff_radius", None)
    cutoff = float(cutoff) if cutoff is not None else None
    print(
        f"[ct_train] soft CME missing for sigma={sigma}: {missing} "
        f"(out_dir={out_dir}) — auto-precomputing …",
        flush=True,
    )
    written = precompute_from_config(
        config_path,
        sigma,
        out_dir=out_dir,
        splits=missing,
        exclude_self=exclude_self,
        cutoff_radius=cutoff,
    )
    print(
        f"[ct_train] soft CME precompute done: "
        + ", ".join(f"{k}={v.name}" for k, v in written.items()),
        flush=True,
    )


def load_soft_cme_lookup(
    cfg: Dict[str, Any],
    split: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[SoftCMELookup]:
    if not soft_cme_is_needed(cfg):
        return None
    scfg = cfg.get("soft_cme") or {}
    out_dir = soft_cme_out_dir(cfg)
    sigma = float(scfg["sigma"])
    file_split = "validation" if split in ("val", "validation") else split
    npz_path = soft_cme_npz_path(out_dir, file_split, sigma)
    if not npz_path.exists():
        if config_path is None:
            raise FileNotFoundError(
                f"Missing soft CME file: {npz_path}. "
                "Pass config_path or enable soft_cme.auto_precompute."
            )
        ensure_soft_cme_precomputed(cfg, config_path)
    loaded = load_soft_cme(out_dir, file_split, sigma)
    print(
        f"[ct_train] soft CME loaded split={file_split} "
        f"sigma={sigma} n={len(loaded['cell_id'])} "
        f"C={loaded['soft_cme'].shape[1]}",
        flush=True,
    )
    return SoftCMELookup(loaded["cell_id"], loaded["soft_cme"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cell-type transformer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("ct_config.json")),
        help="Path to ct_config.json",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        cfg = json.load(f)
    return cfg


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


def build_datamodule_cfg(cfg: Dict[str, Any]):
    """Minimal OmegaConf tree expected by ``DataModule`` / ``AbstractDataModule``."""
    return OmegaConf.create(
        {
            "general": {
                "seed": int(cfg.get("seed", 0)),
                "name": cfg.get("run_name", "cell_types"),
            },
            "dataset": cfg["dataset"],
            "train": {"batch_size": int(cfg["train"]["batch_size"])},
            "validation": {
                "batch_size": int(cfg.get("validation", {}).get("batch_size", cfg["train"]["batch_size"]))
            },
            "test": {
                "batch_size": int(cfg.get("test", {}).get("batch_size", cfg["train"]["batch_size"]))
            },
        }
    )


def build_model(cfg: Dict[str, Any], gene_dim: int, num_classes: int) -> CellTypeTransformer:
    mcfg = cfg["model"]
    scfg = cfg.get("soft_cme") or {}
    predict_cme = bool(mcfg.get("predict_cme", scfg.get("enabled", False)))
    return CellTypeTransformer(
        gene_dim=gene_dim,
        num_classes=num_classes,
        n_layers=int(mcfg.get("n_layers", 4)),
        hidden_mlp_dims=dict(mcfg["hidden_mlp_dims"]),
        hidden_dims=dict(mcfg["hidden_dims"]),
        dropout_cls=float(mcfg.get("dropout_cls", 0.1)),
        dropout_layer=float(mcfg.get("dropout_layer", 0.1)),
        dropout_cme=float(mcfg.get("dropout_cme", mcfg.get("dropout_cls", 0.1))),
        dropout_fusion=float(mcfg.get("dropout_fusion", mcfg.get("dropout_cls", 0.1))),
        gene_mlp_activation=mcfg.get("gene_mlp_activation", "relu"),
        cls_activation=str(mcfg.get("cls_activation", "relu")),
        cme_activation=str(mcfg.get("cme_activation", mcfg.get("cls_activation", "relu"))),
        fusion_activation=mcfg.get("fusion_activation", None),
        global_skip=mcfg.get("global_skip", None),
        subgraph_summary=bool(mcfg.get("subgraph_summary", False)),
        predict_cme=predict_cme,
        feature_cross=bool(mcfg.get("feature_cross", False)),
        feature_cross_include_h=bool(mcfg.get("feature_cross_include_h", True)),
    )


def setup_wandb_run(
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    gene_dim: int,
    num_classes: int,
    n_params: int,
) -> bool:
    """Init wandb from ``cfg['wandb']``. Returns True if logging is active."""
    wcfg = cfg.get("wandb") or {}
    if not bool(wcfg.get("enabled", False)):
        print("[ct_train] wandb disabled")
        return False

    mode = str(wcfg.get("mode", "online"))
    init_kwargs: Dict[str, Any] = {
        "project": str(wcfg.get("project", "cell_types")),
        "name": str(cfg.get("run_name", "ct_transformer")),
        "config": {
            **cfg,
            "resolved": {
                "gene_dim": gene_dim,
                "num_classes": num_classes,
                "n_params": n_params,
                "run_dir": str(run_dir),
            },
        },
        "dir": str(run_dir),
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
    wandb.save(str(run_dir / "ct_config.snapshot.json"), policy="now")
    print(
        f"[ct_train] wandb: project={init_kwargs['project']} "
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
    """Print and log a one-time parameter breakdown by top-level module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []

    print(
        f"[ct_train] parameters: total={total:,} trainable={trainable:,}",
        flush=True,
    )
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
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
    wandb.run.summary["model/parameter_count"] = total
    wandb.run.summary["model/trainable_parameter_count"] = trainable


def class_names_from_decoder(
    decoder: Optional[dict], num_classes: int
) -> List[str]:
    """Map class index → display name (falls back to ``class_i``)."""
    decoder = decoder or {}
    names: List[str] = []
    for i in range(num_classes):
        label = decoder.get(i, decoder.get(str(i), f"class_{i}"))
        names.append(str(label))
    return names


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, batch) -> Tuple[float, int]:
    """Return (accuracy, n_cells) over non-padded cells (integer labels)."""
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
    """Accumulate correct / total cell counts per class index."""
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
    """Per-class accuracy; ``None`` if that class was absent this epoch."""
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
    """One wandb scalar per class → separate auto-plots in the UI.

    Keys like ``train/acc_per_class/<name>``; group by prefix in wandb if
    you want them side-by-side.
    """
    payload: Dict[str, float] = {}
    for c, name in enumerate(class_names):
        if c < len(per_class_acc) and per_class_acc[c] is not None:
            # Sanitize key fragments that break wandb path grouping.
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
    soft_cme_lookup: Optional[SoftCMELookup] = None,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_cme_loss = 0.0
    total_correct = 0
    total_cells = 0
    n_batches = 0
    correct_counts = torch.zeros(num_classes, dtype=torch.long)
    total_counts = torch.zeros(num_classes, dtype=torch.long)

    for step, raw in enumerate(loader):
        batch = to_batch(raw, device=device)
        if soft_cme_lookup is not None:
            attach_soft_cme(batch, soft_cme_lookup, device)

        outputs = model(batch)
        if not isinstance(outputs, dict):
            outputs = {"cls_logits": outputs}

        loss_out = loss_fn(outputs, batch)
        if isinstance(loss_out, tuple):
            loss, parts = loss_out
            cls_loss_val = float(parts["cls"].detach().item())
            cme_loss_val = float(parts.get("cme", parts["cls"] * 0).detach().item())
        else:
            # Hard-CE-only path still expects logits tensor historically.
            loss = loss_out
            cls_loss_val = float(loss.detach().item())
            cme_loss_val = 0.0

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        cls_logits = outputs["cls_logits"].detach()
        acc, n = masked_accuracy(cls_logits, batch)
        update_per_class_counts(cls_logits, batch, correct_counts, total_counts)
        loss_val = float(loss.detach().item())
        total_loss += loss_val
        total_cls_loss += cls_loss_val
        total_cme_loss += cme_loss_val
        total_correct += int(acc * n)
        total_cells += n
        n_batches += 1

        if log_every and train and (step + 1) % log_every == 0:
            print(
                f"  [{split}] epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss_val:.4f} cls={cls_loss_val:.4f} "
                f"cme={cme_loss_val:.4f} acc={acc:.4f}",
                flush=True,
            )
            if wandb_log_steps:
                wandb_log(
                    {
                        f"{split}/step_loss": loss_val,
                        f"{split}/step_cls_loss": cls_loss_val,
                        f"{split}/step_cme_loss": cme_loss_val,
                        f"{split}/step_acc": acc,
                        "epoch": epoch,
                    },
                    step=global_step_offset + step + 1,
                )

    mean_loss = total_loss / max(n_batches, 1)
    mean_acc = total_correct / max(total_cells, 1)
    return {
        "loss": mean_loss,
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "cme_loss": total_cme_loss / max(n_batches, 1),
        "acc": mean_acc,
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
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "cuda")))

    out_dir = Path(cfg.get("output_dir", "exploration/cell_types/outputs"))
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    run_dir = out_dir / str(cfg.get("run_name", "ct_transformer"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "ct_config.snapshot.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[ct_train] config={args.config}")
    print(f"[ct_train] device={device}  output={run_dir}")

    dm_cfg = build_datamodule_cfg(cfg)
    print("[ct_train] Loading data …")
    datamodule = DataModule(dm_cfg)
    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(infos.num_genes)
    num_classes = int(infos.num_cell_class)
    print(f"[ct_train] genes={gene_dim}  classes={num_classes}")

    model = build_model(cfg, gene_dim=gene_dim, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ct_train] model predict_cme={model.predict_cme}  "
        f"params={n_params:,}",
        flush=True,
    )

    ensure_soft_cme_precomputed(cfg, args.config)
    train_cme = load_soft_cme_lookup(cfg, "train", config_path=args.config)
    val_cme = load_soft_cme_lookup(cfg, "validation", config_path=args.config)
    if train_cme is not None and train_cme.num_classes != num_classes:
        raise ValueError(
            f"soft CME C={train_cme.num_classes} != dataset classes={num_classes}"
        )
    if model.predict_cme and train_cme is None:
        raise ValueError(
            "model.predict_cme=True but soft CME tables could not be loaded "
            "(check soft_cme.out_dir / sigma)"
        )

    use_wandb = setup_wandb_run(
        cfg,
        run_dir=run_dir,
        gene_dim=gene_dim,
        num_classes=num_classes,
        n_params=n_params,
    )
    log_model_parameter_counts(model)
    wcfg = cfg.get("wandb") or {}
    wandb_log_steps = use_wandb and bool(wcfg.get("log_every_steps", True))
    log_per_class = use_wandb and bool(wcfg.get("log_per_class_acc", True))
    class_names = class_names_from_decoder(
        getattr(datamodule.statistics["train"], "cell_class_decoder", None),
        num_classes,
    )

    loss_kwargs = {
        k: v for k, v in dict(cfg.get("loss", {})).items() if k != "name"
    }
    loss_fn = get_loss(str(cfg["loss"]["name"]), **loss_kwargs)
    tcfg = cfg["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    train_loader = datamodule.train_dataloader()
    val_loader = None
    if getattr(datamodule, "validation_dataset", None) is not None:
        val_loader = datamodule.validation_dataloader()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    history = []
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
    es_monitor = str(es_cfg.get("monitor", "val/acc"))  # val/acc | val/loss
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    global_step = 0

    try:
        for epoch in range(1, n_epochs + 1):
            if (
                rechunk_every > 0
                and hasattr(datamodule, "train_dataset")
                and hasattr(datamodule.train_dataset, "rechunk")
                and (epoch == 1 or (epoch - 1) % rechunk_every == 0)
            ):
                datamodule.train_dataset.rechunk(seed=epoch)
                print(f"[ct_train] rechunked train graphs (seed={epoch})", flush=True)

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
                soft_cme_lookup=train_cme,
            )
            global_step += len(train_loader)
            print(
                f"[train] loss={train_metrics['loss']:.4f} "
                f"cls={train_metrics['cls_loss']:.4f} "
                f"cme={train_metrics['cme_loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} "
                f"cells={int(train_metrics['n_cells'])}",
                flush=True,
            )

            val_metrics = None
            if val_loader is not None and (epoch % val_every == 0):
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    num_classes=num_classes,
                    optimizer=None,
                    epoch=epoch,
                    split="val",
                    soft_cme_lookup=val_cme,
                )
                print(
                    f"[val]   loss={val_metrics['loss']:.4f} "
                    f"cls={val_metrics['cls_loss']:.4f} "
                    f"cme={val_metrics['cme_loss']:.4f} "
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
                if val_metrics["loss"] < best_val_loss - (
                    es_min_delta if es_monitor == "val/loss" else 0.0
                ):
                    best_val_loss = val_metrics["loss"]
                    if es_monitor == "val/loss":
                        improved = True

                # Always checkpoint on best accuracy (primary metric).
                if val_metrics["acc"] >= best_val_acc:
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        epoch,
                        cfg,
                        {"train": train_metrics, "val": val_metrics},
                    )
                    print(
                        f"[ct_train] saved best checkpoint "
                        f"(val_acc={best_val_acc:.4f})"
                    )
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
                            f"[ct_train] early_stopping: no improve on "
                            f"{es_monitor} for {epochs_without_improve}/"
                            f"{es_patience} val checks",
                            flush=True,
                        )

            epoch_log: Dict[str, Any] = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train/loss": train_metrics["loss"],
                "train/cls_loss": train_metrics["cls_loss"],
                "train/cme_loss": train_metrics["cme_loss"],
                "train/acc": train_metrics["acc"],
                "train/n_cells": train_metrics["n_cells"],
            }
            if val_metrics is not None:
                epoch_log.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/cls_loss": val_metrics["cls_loss"],
                        "val/cme_loss": val_metrics["cme_loss"],
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

            # history.json: keep JSON-serializable fields only
            _metric_keys = (
                "loss",
                "cls_loss",
                "cme_loss",
                "acc",
                "n_cells",
                "per_class_acc",
                "per_class_n",
            )
            record = {
                "epoch": epoch,
                "train": {
                    k: v for k, v in train_metrics.items() if k in _metric_keys
                },
                "val": None
                if val_metrics is None
                else {k: v for k, v in val_metrics.items() if k in _metric_keys},
            }
            history.append(record)

            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    {"train": train_metrics, "val": val_metrics},
                )

            with (run_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)

            if (
                early_stop
                and val_metrics is not None
                and epochs_without_improve >= es_patience
            ):
                print(
                    f"[ct_train] early stopping at epoch {epoch} "
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
        print(f"\n[ct_train] done. best_val_acc={best_val_acc:.4f}  dir={run_dir}")
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

