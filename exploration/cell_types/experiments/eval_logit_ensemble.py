"""Evaluate logit-average ensemble of EMA checkpoints on canonical val metric.

Usage (repo root, LUNA)::

    CUDA_VISIBLE_DEVICES=5 python exploration/cell_types/experiments/eval_logit_ensemble.py \
      --config exploration/cell_types/experiments/configs/exp089_ema98.json \
      --ckpts .../exp089_ema98/best.pt .../exp090_ema975/best.pt \
      --name ens_089_090
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import get_loss  # noqa: E402
from exploration.cell_types.ct_train import (  # noqa: E402
    attach_soft_cme,
    build_canonical_eval_loss,
    build_datamodule_cfg,
    build_model,
    load_config,
    load_soft_cme_lookup,
    masked_accuracy,
    resolve_device,
    resolve_repo_path,
)
from utils.data.misc import to_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--name", default="logit_ens")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_ema_into_model(model, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("ema_state_dict"):
        shadow = ckpt["ema_state_dict"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(
                        shadow[name].to(device=param.device, dtype=param.dtype)
                    )
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    ckpt_paths = [resolve_repo_path(p) for p in args.ckpts]

    dm_cfg = build_datamodule_cfg(cfg)
    print(f"[ens] loading data for {args.name} …", flush=True)
    datamodule = DataModule(dm_cfg)
    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(infos.num_genes)
    num_classes = int(infos.num_cell_class)

    models = []
    for p in ckpt_paths:
        m = build_model(cfg, gene_dim=gene_dim, num_classes=num_classes).to(device)
        load_ema_into_model(m, p, device)
        m.eval()
        models.append(m)
        print(f"[ens] loaded {p}", flush=True)

    val_cme = load_soft_cme_lookup(cfg, "validation", config_path=args.config)
    eval_fn = build_canonical_eval_loss(cfg)
    val_loader = datamodule.val_dataloader()

    total_eval = 0.0
    total_cls = 0.0
    total_cme = 0.0
    total_correct = 0
    total_cells = 0
    n_batches = 0

    with torch.no_grad():
        for raw in val_loader:
            batch = to_batch(raw, device=device)
            if val_cme is not None:
                attach_soft_cme(batch, val_cme, device)

            cls_stack = []
            cme_stack = []
            for m in models:
                out = m(batch)
                cls_stack.append(out["cls_logits"])
                cme_stack.append(out["cme_logits"])
            # Average logits (equivalent to geometric mean of softmax for CE)
            ens = {
                "cls_logits": torch.stack(cls_stack, dim=0).mean(dim=0),
                "cme_logits": torch.stack(cme_stack, dim=0).mean(dim=0),
            }
            _, parts = eval_fn(ens, batch)
            cls_v = float(parts["cls"].detach().item())
            cme_v = float(parts["cme"].detach().item())
            total_cls += cls_v
            total_cme += cme_v
            total_eval += cls_v + cme_v
            acc, n = masked_accuracy(ens["cls_logits"], batch)
            total_correct += int(acc * n)
            total_cells += n
            n_batches += 1

    metrics = {
        "eval_loss": total_eval / max(n_batches, 1),
        "cls_loss": total_cls / max(n_batches, 1),
        "cme_loss": total_cme / max(n_batches, 1),
        "acc": total_correct / max(total_cells, 1),
    }
    print(
        f"[ens] {args.name}: eval={metrics['eval_loss']:.6f} "
        f"cls={metrics['cls_loss']:.6f} cme={metrics['cme_loss']:.6f} "
        f"acc={metrics['acc']:.6f}",
        flush=True,
    )
    out_path = (
        resolve_repo_path("exploration/cell_types/experiments")
        / f"{args.name}_metrics.json"
    )
    out_path.write_text(
        json.dumps(
            {"name": args.name, "ckpts": [str(p) for p in ckpt_paths], "metrics": metrics},
            indent=2,
        )
        + "\n"
    )
    print(f"[ens] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
