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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
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
from exploration.cell_types.ct_transformer import CellTypeTransformer  # noqa: E402
from utils.data.misc import to_batch  # noqa: E402


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
    return CellTypeTransformer(
        gene_dim=gene_dim,
        num_classes=num_classes,
        n_layers=int(mcfg.get("n_layers", 4)),
        hidden_mlp_dims=dict(mcfg["hidden_mlp_dims"]),
        hidden_dims=dict(mcfg["hidden_dims"]),
        dropout_cls=float(mcfg.get("dropout_cls", 0.1)),
        dropout_layer=float(mcfg.get("dropout_layer", 0.1)),
        gene_mlp_activation=mcfg.get("gene_mlp_activation", "relu"),
        global_skip=mcfg.get("global_skip", None),
        subgraph_summary=bool(mcfg.get("subgraph_summary", False)),
    )


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


def run_epoch(
    model: CellTypeTransformer,
    loader,
    loss_fn,
    device: torch.device,
    optimizer: Optional[AdamW] = None,
    grad_clip: Optional[float] = None,
    log_every: int = 0,
    epoch: int = 0,
    split: str = "train",
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    n_batches = 0

    for step, raw in enumerate(loader):
        batch = to_batch(raw, device=device)
        logits = model(batch)
        loss = loss_fn(logits, batch)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        acc, n = masked_accuracy(logits.detach(), batch)
        total_loss += float(loss.detach().item())
        total_correct += int(acc * n)
        total_cells += n
        n_batches += 1

        if log_every and train and (step + 1) % log_every == 0:
            print(
                f"  [{split}] epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.4f} acc={acc:.4f}",
                flush=True,
            )

    mean_loss = total_loss / max(n_batches, 1)
    mean_acc = total_correct / max(total_cells, 1)
    return {"loss": mean_loss, "acc": mean_acc, "n_cells": float(total_cells)}


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
    print(f"[ct_train] model params={n_params:,}")

    loss_fn = get_loss(str(cfg["loss"]["name"]))
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
    history = []
    n_epochs = int(tcfg["n_epochs"])
    val_every = int(tcfg.get("val_every_n_epochs", 1))
    save_every = int(tcfg.get("save_every_n_epochs", 10))
    log_every = int(tcfg.get("log_every", 0))
    grad_clip = tcfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    for epoch in range(1, n_epochs + 1):
        print(f"\n=== Epoch {epoch}/{n_epochs} ===", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            log_every=log_every,
            epoch=epoch,
            split="train",
        )
        print(
            f"[train] loss={train_metrics['loss']:.4f} "
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
                optimizer=None,
                epoch=epoch,
                split="val",
            )
            print(
                f"[val]   loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} "
                f"cells={int(val_metrics['n_cells'])}",
                flush=True,
            )
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    {"train": train_metrics, "val": val_metrics},
                )
                print(f"[ct_train] saved best checkpoint (val_acc={best_val_acc:.4f})")

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
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

    save_checkpoint(
        run_dir / "last.pt",
        model,
        optimizer,
        n_epochs,
        cfg,
        {"train": train_metrics, "val": val_metrics, "best_val_acc": best_val_acc},
    )
    print(f"\n[ct_train] done. best_val_acc={best_val_acc:.4f}  dir={run_dir}")


if __name__ == "__main__":
    main()
