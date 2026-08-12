"""Grid-search non-negative mix weights for a logit ensemble on val.

Caches per-model logits once, then evaluates many weight vectors cheaply.

Usage (repo root, LUNA)::

    CUDA_VISIBLE_DEVICES=5 python exploration/cell_types/experiments/search_logit_weights.py \
      --config exploration/cell_types/experiments/configs/exp135_ls006_aux017.json \
      --ckpts .../exp135/.../best.pt .../exp157/.../best.pt .../exp159/.../best.pt \
      --step 0.1
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_train import (  # noqa: E402
    attach_soft_cme,
    build_canonical_eval_loss,
    build_datamodule_cfg,
    build_model,
    load_config,
    load_ema_weights_into_model,
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
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def weight_grid(n: int, step: float) -> List[Tuple[float, ...]]:
    """Non-negative weights summing to 1 on a simplex grid."""
    k = int(round(1.0 / step))
    out: List[Tuple[float, ...]] = []
    for comb in itertools.combinations_with_replacement(range(n), k):
        counts = [0] * n
        for i in comb:
            counts[i] += 1
        out.append(tuple(c / float(k) for c in counts))
    return sorted(set(out))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    ckpt_paths = [resolve_repo_path(p) for p in args.ckpts]
    n = len(ckpt_paths)

    dm_cfg = build_datamodule_cfg(cfg)
    print("[wsearch] loading data …", flush=True)
    datamodule = DataModule(dm_cfg)
    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(infos.num_genes)
    num_classes = int(infos.num_cell_class)

    models = []
    for p in ckpt_paths:
        m = build_model(cfg, gene_dim=gene_dim, num_classes=num_classes).to(device)
        load_ema_weights_into_model(m, p, device)
        m.eval()
        models.append(m)
        print(f"[wsearch] loaded {p.name}", flush=True)

    val_cme = load_soft_cme_lookup(cfg, "validation", config_path=args.config)
    eval_fn = build_canonical_eval_loss(cfg)
    val_loader = datamodule.val_dataloader()

    cache = []
    with torch.no_grad():
        for raw in val_loader:
            batch = to_batch(raw, device=device)
            if val_cme is not None:
                attach_soft_cme(batch, val_cme, device)
            cls_list = []
            cme_list = []
            for m in models:
                out = m(batch)
                cls_list.append(out["cls_logits"].detach().cpu())
                cme_list.append(out["cme_logits"].detach().cpu())
            cache.append(
                {
                    "cls": cls_list,
                    "cme": cme_list,
                    "node_mask": batch.node_mask.detach().cpu(),
                    "cell_class": None
                    if batch.cell_class is None
                    else batch.cell_class.detach().cpu(),
                    "soft_cme": None
                    if getattr(batch, "soft_cme", None) is None
                    else batch.soft_cme.detach().cpu(),
                }
            )
    print(
        f"[wsearch] cached {len(cache)} batches; "
        f"grid size={len(weight_grid(n, args.step))} …",
        flush=True,
    )

    results = []
    for w in weight_grid(n, args.step):
        total_eval = 0.0
        total_cls = 0.0
        total_cme = 0.0
        total_correct = 0
        total_cells = 0
        n_batches = 0
        with torch.no_grad():
            for item in cache:
                cls = sum(wi * item["cls"][i] for i, wi in enumerate(w) if wi > 0)
                cme = sum(wi * item["cme"][i] for i, wi in enumerate(w) if wi > 0)
                cls = cls.to(device)
                cme = cme.to(device)
                batch = SimpleNamespace(
                    node_mask=item["node_mask"].to(device),
                    cell_class=None
                    if item["cell_class"] is None
                    else item["cell_class"].to(device),
                    soft_cme=None
                    if item["soft_cme"] is None
                    else item["soft_cme"].to(device),
                )
                outputs = {"cls_logits": cls, "cme_logits": cme}
                _, parts = eval_fn(outputs, batch)
                total_cls += float(parts["cls"].item())
                total_cme += float(parts["cme"].item())
                total_eval += float(parts["cls"].item() + parts["cme"].item())
                acc, n_cells = masked_accuracy(cls, batch)
                total_correct += int(acc * n_cells)
                total_cells += n_cells
                n_batches += 1
        mean_eval = total_eval / max(n_batches, 1)
        mean_cls = total_cls / max(n_batches, 1)
        mean_cme = total_cme / max(n_batches, 1)
        mean_acc = total_correct / max(total_cells, 1)
        results.append((mean_eval, mean_cls, mean_cme, mean_acc, w))

    results.sort(key=lambda x: x[0])
    print(f"[wsearch] top-{args.top}:", flush=True)
    for i, (ev, cl, cm, ac, w) in enumerate(results[: args.top]):
        wstr = ",".join(f"{x:.2f}" for x in w)
        print(
            f"  #{i+1} eval={ev:.6f} cls={cl:.6f} cme={cm:.6f} "
            f"acc={ac:.6f} w=[{wstr}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
