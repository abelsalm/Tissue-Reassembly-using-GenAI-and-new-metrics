"""Average EMA shadows from several best.pt checkpoints and eval on val.

Usage (repo root, LUNA, CUDA_VISIBLE_DEVICES=X)::

    python exploration/cell_types/experiments/eval_soup.py \\
      --config exploration/cell_types/experiments/configs/exp089_ema98.json \\
      --ckpts outputs/exp089_ema98/best.pt outputs/exp090_ema975/best.pt \\
      --name soup_089_090
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import get_loss  # noqa: E402
from exploration.cell_types.ct_train import (  # noqa: E402
    build_canonical_eval_loss,
    build_datamodule_cfg,
    build_model,
    load_config,
    load_soft_cme_lookup,
    resolve_device,
    resolve_repo_path,
    run_epoch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--name", default="soup")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_ema_shadow(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        shadow = ckpt["ema_state_dict"]["shadow"]
        return {k: v.detach().float().clone() for k, v in shadow.items()}
    # Fallback: raw model weights
    return {
        k: v.detach().float().clone()
        for k, v in ckpt["model_state_dict"].items()
        if not k.endswith("num_batches_tracked")
    }


def average_shadows(paths: List[Path]) -> Dict[str, torch.Tensor]:
    shadows = [load_ema_shadow(p) for p in paths]
    keys = shadows[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for s in shadows:
            if k not in s:
                raise KeyError(f"missing key {k} in one checkpoint")
            acc = s[k] if acc is None else acc + s[k]
        out[k] = acc / float(len(shadows))
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    ckpt_paths = [resolve_repo_path(p) for p in args.ckpts]
    for p in ckpt_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    dm_cfg = build_datamodule_cfg(cfg)
    print(f"[soup] loading data for {args.name} …", flush=True)
    datamodule = DataModule(dm_cfg)
    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(infos.num_genes)
    num_classes = int(infos.num_cell_class)

    model = build_model(cfg, gene_dim=gene_dim, num_classes=num_classes).to(device)
    avg = average_shadows(ckpt_paths)
    # Apply averaged EMA shadow onto model params
    missing = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in avg:
                missing.append(name)
                continue
            param.data.copy_(avg[name].to(device=param.device, dtype=param.dtype))
    if missing:
        print(f"[soup] warning: {len(missing)} params not in avg shadow", flush=True)

    val_cme = load_soft_cme_lookup(cfg, "validation", config_path=args.config)
    # Soup eval only needs canonical metric; zero aux so soft_cme_aux is unused.
    loss_fn = get_loss(
        "combined",
        label_smoothing=0.05,
        cls_weight=1.0,
        cme_weight=1.0,
        stage1_cls_weight=0.0,
        stage1_cme_weight=0.0,
        aux_cme_weight=0.0,
    )
    eval_fn = build_canonical_eval_loss(cfg)
    val_loader = datamodule.val_dataloader()
    print(f"[soup] evaluating {args.name} on {len(ckpt_paths)} ckpts …", flush=True)
    metrics = run_epoch(
        model,
        val_loader,
        loss_fn,
        device,
        num_classes=num_classes,
        optimizer=None,
        epoch=0,
        split="val",
        soft_cme_lookup=val_cme,
        canonical_eval_fn=eval_fn,
    )
    print(
        f"[soup] {args.name}: eval={metrics['eval_loss']:.6f} "
        f"cls={metrics['cls_loss']:.6f} cme={metrics['cme_loss']:.6f} "
        f"acc={metrics['acc']:.6f}",
        flush=True,
    )
    out = {
        "name": args.name,
        "ckpts": [str(p) for p in ckpt_paths],
        "metrics": {
            "eval_loss": metrics["eval_loss"],
            "cls_loss": metrics["cls_loss"],
            "cme_loss": metrics["cme_loss"],
            "acc": metrics["acc"],
        },
    }
    out_path = resolve_repo_path("exploration/cell_types/experiments") / f"{args.name}_metrics.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[soup] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
