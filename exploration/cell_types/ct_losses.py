"""Losses for cell-type classification and soft CME prediction.

Select via ``config[\"loss\"][\"name\"]`` (see ``ct_config.json``).

* ``cross_entropy`` — hard integer labels ``0…C-1`` (cell type).
* ``soft_cross_entropy`` — soft target distributions ``(B, N, C)`` (CME);
  applies ``log_softmax`` on logits internally.
* ``combined`` — weighted sum of both.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from utils.data.dataholder import DataHolder


LossFn = Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]]


def _as_logits(
    outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], key: str
) -> torch.Tensor:
    """Accept raw logits or a model output dict keyed by ``key``."""
    if isinstance(outputs, dict):
        if key not in outputs:
            raise ValueError(f"outputs missing '{key}' (keys={list(outputs)})")
        return outputs[key]
    return outputs


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


def soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked soft CE between logits and a probability target.

    ``loss = -Σ_c target_c · log_softmax(logits)_c``, averaged over real cells.
    Softmax is applied here (logits must be raw scores).
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits/target shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(target.shape)}"
        )
    mask = node_mask.bool()
    if mask.ndim != 2:
        raise ValueError(f"node_mask must be (B, N), got {tuple(mask.shape)}")

    # Renormalize targets on the simplex (numerical safety).
    target = target.clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    log_probs = F.log_softmax(logits, dim=-1)
    per_cell = -(target * log_probs).sum(dim=-1)  # (B, N)
    per_cell_m = per_cell[mask]
    if per_cell_m.numel() == 0:
        return logits.sum() * 0.0
    return per_cell_m.mean()


def make_cross_entropy(label_smoothing: float = 0.0) -> LossFn:
    """Build masked hard CE; ``label_smoothing`` softens hard targets."""
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError(
            f"label_smoothing must be in [0, 1), got {label_smoothing}"
        )
    ls = float(label_smoothing)

    def cross_entropy(
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: DataHolder
    ) -> torch.Tensor:
        logits = _as_logits(outputs, "cls_logits")
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

        return F.cross_entropy(logits_m, targets_m, label_smoothing=ls)

    return cross_entropy


def make_soft_cross_entropy(eps: float = 1e-8) -> LossFn:
    """Soft CE expecting ``batch.soft_cme`` of shape ``(B, N, C)``."""

    def _fn(
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: DataHolder
    ) -> torch.Tensor:
        logits = _as_logits(outputs, "cme_logits")
        if batch.node_mask is None:
            raise ValueError("batch.node_mask is required")
        soft = getattr(batch, "soft_cme", None)
        if soft is None:
            raise ValueError("batch.soft_cme is required for soft_cross_entropy")
        return soft_cross_entropy(logits, soft, batch.node_mask, eps=eps)

    return _fn


def make_combined_loss(
    label_smoothing: float = 0.0,
    cls_weight: float = 1.0,
    cme_weight: float = 1.0,
    stage1_cls_weight: float = 0.0,
    stage1_cme_weight: float = 0.0,
    eps: float = 1e-8,
) -> LossFn:
    """Weighted final-head losses + optional stage-1 probe aux losses.

    ``cls_weight * CE(cls_logits) + cme_weight * soft_CE(cme_logits)``
    and, if present / weights > 0:
    ``stage1_*_weight`` on ``cls_logits_stage1`` / ``cme_logits_stage1``.
    """
    cls_fn = make_cross_entropy(label_smoothing)
    w_cls = float(cls_weight)
    w_cme = float(cme_weight)
    w_s1_cls = float(stage1_cls_weight)
    w_s1_cme = float(stage1_cme_weight)

    def combined(
        outputs: Dict[str, torch.Tensor], batch: DataHolder
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(outputs, dict) or "cls_logits" not in outputs:
            raise ValueError("combined loss expects outputs['cls_logits']")
        cls_loss = cls_fn(outputs["cls_logits"], batch)
        parts = {"cls": cls_loss}
        total = w_cls * cls_loss

        if w_cme != 0.0:
            if "cme_logits" not in outputs:
                raise ValueError(
                    "cme_weight>0 requires outputs['cme_logits'] "
                    "(enable model.predict_cme)"
                )
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError("batch.soft_cme is required when cme_weight>0")
            cme_loss = soft_cross_entropy(
                outputs["cme_logits"], soft, batch.node_mask, eps=eps
            )
            parts["cme"] = cme_loss
            total = total + w_cme * cme_loss
        else:
            parts["cme"] = cls_loss.detach() * 0.0

        if w_s1_cls != 0.0:
            if "cls_logits_stage1" not in outputs:
                raise ValueError(
                    "stage1_cls_weight>0 requires outputs['cls_logits_stage1'] "
                    "(enable model.feature_cross)"
                )
            s1_cls = cls_fn(outputs["cls_logits_stage1"], batch)
            parts["cls_stage1"] = s1_cls
            total = total + w_s1_cls * s1_cls

        if w_s1_cme != 0.0:
            if "cme_logits_stage1" not in outputs:
                raise ValueError(
                    "stage1_cme_weight>0 requires outputs['cme_logits_stage1'] "
                    "(enable model.feature_cross)"
                )
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError("batch.soft_cme is required when stage1_cme_weight>0")
            s1_cme = soft_cross_entropy(
                outputs["cme_logits_stage1"], soft, batch.node_mask, eps=eps
            )
            parts["cme_stage1"] = s1_cme
            total = total + w_s1_cme * s1_cme

        return total, parts

    return combined


# Default no-smoothing instance for direct imports / tests.
cross_entropy = make_cross_entropy(0.0)


LOSS_REGISTRY: Dict[str, Callable[..., LossFn]] = {
    "cross_entropy": make_cross_entropy,
    "soft_cross_entropy": make_soft_cross_entropy,
    "combined": make_combined_loss,
}


def get_loss(name: str, **kwargs: Any) -> LossFn:
    try:
        factory = LOSS_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Known: {known}") from exc
    # Drop Nones so callers can pass optional keys freely.
    clean = {k: v for k, v in kwargs.items() if v is not None}
    return factory(**clean)
