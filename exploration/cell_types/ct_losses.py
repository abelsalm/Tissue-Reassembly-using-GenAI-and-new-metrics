"""Losses for cell-type classification.

Select via ``config[\"loss\"][\"name\"]`` (see ``ct_config.json``).

Labels are **integer class indices** ``0 … C-1`` (not one-hot). The model
emits raw logits ``(B, N, C)``; ``F.cross_entropy`` applies log-softmax + NLL.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from utils.data.dataholder import DataHolder


LossFn = Callable[[torch.Tensor, DataHolder], torch.Tensor]


def class_indices_from_batch(batch: DataHolder) -> torch.Tensor:
    """Return per-cell class indices as ``long`` tensor ``(B, N)``.

    Accepts ``cell_class`` shaped ``(B, N)`` or ``(B, N, 1)``. Rejects
    accidental one-hot ``(B, N, C)`` with ``C > 1``.
    """
    if batch.cell_class is None:
        raise ValueError("batch.cell_class is required (integer class indices)")
    targets = batch.cell_class
    if targets.dim() == 3:
        if targets.size(-1) != 1:
            raise ValueError(
                "cell_class looks one-hot or multi-dim "
                f"(shape {tuple(targets.shape)}). "
                "Use integer class indices of shape (B, N) or (B, N, 1)."
            )
        targets = targets.squeeze(-1)
    if targets.dim() != 2:
        raise ValueError(
            f"cell_class must be (B, N) or (B, N, 1), got {tuple(targets.shape)}"
        )
    return targets.long()


def cross_entropy(logits: torch.Tensor, batch: DataHolder) -> torch.Tensor:
    """Masked mean CE: logits ``(B, N, C)`` vs integer labels ``(B, N)``.

    Pads are dropped via ``node_mask`` (not via ``ignore_index``), so only
    real cells enter the mean.
    """
    if batch.node_mask is None:
        raise ValueError("batch.node_mask is required for cross_entropy loss")
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, N, C), got {tuple(logits.shape)}")

    targets = class_indices_from_batch(batch)
    mask = batch.node_mask.bool()

    logits_m = logits[mask]  # (M, C)
    targets_m = targets[mask]  # (M,)
    if logits_m.numel() == 0:
        return logits.sum() * 0.0

    num_classes = logits_m.size(-1)
    if targets_m.numel() > 0:
        tmin = int(targets_m.min().item())
        tmax = int(targets_m.max().item())
        if tmin < 0 or tmax >= num_classes:
            raise ValueError(
                f"class index out of range for C={num_classes}: "
                f"min={tmin}, max={tmax}"
            )

    # Integer-index CE (equivalent to one-hot CE, cheaper / standard).
    return F.cross_entropy(logits_m, targets_m)


LOSS_REGISTRY: Dict[str, LossFn] = {
    "cross_entropy": cross_entropy,
}


def get_loss(name: str) -> LossFn:
    try:
        return LOSS_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Known: {known}") from exc
