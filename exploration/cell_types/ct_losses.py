"""Losses for cell-type classification and soft CME prediction.

Select via ``config[\"loss\"][\"name\"]`` (see ``ct_config.json``).

* ``cross_entropy`` — hard integer labels ``0…C-1`` (cell type).
* ``soft_cross_entropy`` — soft target distributions ``(B, N, C)`` (CME);
  applies ``log_softmax`` on logits internally.
* ``combined`` — weighted sum of both (+ optional soft-Spearman CME rank term).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from metrics.train_spatial_transcriptomics import soft_spearman_distance
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


def temper_soft_target(
    target: torch.Tensor, temperature: float, eps: float = 1e-8
) -> torch.Tensor:
    """Sharpen (T<1) or soften (T>1) a soft probability target, then renormalize."""
    t = float(temperature)
    if t == 1.0:
        return target
    if t <= 0.0:
        raise ValueError(f"cme_target_temperature must be > 0 (got {temperature})")
    # Raise probs to 1/T then renorm (same effect as logits/T on a simplex).
    powered = target.clamp_min(eps).pow(1.0 / t)
    return powered / powered.sum(dim=-1, keepdim=True).clamp_min(eps)


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


def soft_cross_entropy_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked soft CE when predictions are already a simplex (gated CME)."""
    if probs.shape != target.shape:
        raise ValueError(
            f"probs/target shape mismatch: {tuple(probs.shape)} vs "
            f"{tuple(target.shape)}"
        )
    mask = node_mask.bool()
    if mask.ndim != 2:
        raise ValueError(f"node_mask must be (B, N), got {tuple(mask.shape)}")

    target = target.clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_probs = probs.clamp_min(eps).log()
    per_cell = -(target * log_probs).sum(dim=-1)
    per_cell_m = per_cell[mask]
    if per_cell_m.numel() == 0:
        return probs.sum() * 0.0
    return per_cell_m.mean()


def soft_presence_from_mass(
    cme_mass: torch.Tensor,
    presence_scale: Union[float, torch.Tensor],
    *,
    presence_saturation: float = 1.0,
) -> torch.Tensor:
    """Soft presence π*_c = 1 - exp(-m_c / (scale_c * saturation)).

    ``cme_mass`` is unnormalized Gaussian type mass ``(B, N, C)``.
    ``presence_scale`` is typically a per-type median of positive masses
    ``(C,)`` (or a scalar broadcast to all types).
    """
    sat = float(presence_saturation)
    if sat <= 0:
        raise ValueError(f"presence_saturation must be > 0 (got {sat})")
    if not torch.is_tensor(presence_scale):
        scale = torch.as_tensor(
            float(presence_scale),
            device=cme_mass.device,
            dtype=cme_mass.dtype,
        )
    else:
        scale = presence_scale.to(device=cme_mass.device, dtype=cme_mass.dtype)
    # Broadcast ``(C,)`` or scalar over ``(B, N, C)``.
    while scale.ndim < cme_mass.ndim:
        scale = scale.unsqueeze(0)
    scale = scale * sat
    if torch.any(scale <= 0):
        raise ValueError("presence_scale * presence_saturation must be > 0")
    mass = cme_mass.clamp_min(0.0)
    return 1.0 - torch.exp(-(mass / scale))


def masked_bce_presence(
    presence_logits: torch.Tensor,
    presence_target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked BCE-with-logits between predicted π and soft presence target."""
    if presence_logits.shape != presence_target.shape:
        raise ValueError(
            f"presence logits/target shape mismatch: "
            f"{tuple(presence_logits.shape)} vs {tuple(presence_target.shape)}"
        )
    per_elem = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target, reduction="none"
    )
    per_cell = per_elem.mean(dim=-1)
    mask = node_mask.bool()
    if mask.any():
        return per_cell[mask].mean()
    return per_cell.mean() * 0.0


def soft_js_divergence(
    logits: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked Jensen–Shannon divergence between softmax(logits) and target.

    ``JS(p, q) = 0.5·KL(p||m) + 0.5·KL(q||m)`` with ``m = 0.5·(p + q)``,
    averaged over real cells. Target is renormalized like ``soft_cross_entropy``.
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits/target shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(target.shape)}"
        )
    mask = node_mask.bool()
    if mask.ndim != 2:
        raise ValueError(f"node_mask must be (B, N), got {tuple(mask.shape)}")

    target = target.clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    p = F.softmax(logits, dim=-1)
    q = target
    m = 0.5 * (p + q)
    log_m = m.clamp_min(eps).log()

    kl_pm = F.kl_div(log_m, p, log_target=False, reduction="none").sum(dim=-1)
    kl_qm = F.kl_div(log_m, q, log_target=False, reduction="none").sum(dim=-1)
    per_cell = 0.5 * kl_pm + 0.5 * kl_qm

    per_cell_m = per_cell[mask]
    if per_cell_m.numel() == 0:
        return logits.sum() * 0.0
    return per_cell_m.mean()


def soft_spearman_cme_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-8,
    chunk_size: int = 64,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked soft-Spearman distance between predicted CME and soft GT.

    Soft-ranks both predicted probs (``softmax(logits)`` or provided ``probs``)
    and the soft CME target along the class axis, then returns the mean of
    ``1 - Pearson(ranks_pred, ranks_gt)`` over real cells.
    """
    ref = probs if probs is not None else logits
    if ref.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(ref.shape)} vs "
            f"{tuple(target.shape)}"
        )
    mask = node_mask.bool()
    if mask.ndim != 2:
        raise ValueError(f"node_mask must be (B, N), got {tuple(mask.shape)}")
    if float(tau) <= 0.0:
        raise ValueError(f"cme_spearman_tau must be > 0, got {tau}")

    target = target.clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred = probs if probs is not None else F.softmax(logits, dim=-1)

    # (B, N) distance; C is small so chunking is cheap.
    dist = soft_spearman_distance(
        pred, target, float(tau), eps=eps, chunk_size=int(chunk_size)
    )
    dist_m = dist[mask]
    if dist_m.numel() == 0:
        return pred.sum() * 0.0
    return dist_m.mean()


def _gene_sim_neighbor_targets(
    batch: DataHolder,
    outputs: Dict[str, torch.Tensor],
    *,
    source: str,
    temperature: float,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gene-similarity-weighted neighbor type mixtures ``(B, N, C)``.

    Returns ``(target, has_neighbor)`` where ``has_neighbor`` is ``(B, N)`` and
    marks cells with at least one valid in-bag neighbor (excludes self / pads).
    """
    if batch.node_features is None:
        raise ValueError("batch.node_features is required for gene_sim CME aux")
    if batch.node_mask is None:
        raise ValueError("batch.node_mask is required for gene_sim CME aux")
    if source not in ("label", "cls_pred"):
        raise ValueError(
            f"gene_sim_source must be 'label' or 'cls_pred', got {source!r}"
        )
    if temperature <= 0:
        raise ValueError(f"gene_sim_temperature must be > 0, got {temperature}")

    feats = batch.node_features
    mask = batch.node_mask.bool()
    if feats.dim() != 3:
        raise ValueError(
            f"node_features must be (B, N, G), got {tuple(feats.shape)}"
        )

    feats_norm = F.normalize(feats, dim=-1, eps=1e-8)
    sim = torch.bmm(feats_norm, feats_norm.transpose(1, 2))  # (B, N, N)

    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    has_neighbor = pair_mask.any(dim=-1)  # (B, N)

    sim_masked = sim.masked_fill(~pair_mask, float("-inf"))
    weights = F.softmax(sim_masked / float(temperature), dim=-1)
    weights = weights.nan_to_num(0.0) * pair_mask.float()

    if source == "label":
        indices = class_indices_from_batch(batch)
        type_dist = F.one_hot(indices, num_classes).float()
    else:
        if "cls_logits" not in outputs:
            raise ValueError(
                "gene_sim_source='cls_pred' requires outputs['cls_logits']"
            )
        type_dist = F.softmax(outputs["cls_logits"].detach(), dim=-1)

    target = torch.bmm(weights, type_dist)
    return target, has_neighbor


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
    aux_cme_weight: float = 0.0,
    cme_divergence: str = "soft_ce",
    cme_temperature: float = 1.0,
    cme_target_temperature: float = 1.0,
    cme_entropy_match_weight: float = 0.0,
    bag_cme_weight: float = 0.0,
    gene_sim_cme_weight: float = 0.0,
    gene_sim_temperature: float = 0.5,
    gene_sim_source: str = "label",
    cme_spearman_weight: float = 0.0,
    cme_spearman_tau: float = 0.1,
    cme_spearman_chunk_size: int = 64,
    presence_weight: float = 0.0,
    presence_scale: Optional[Union[float, list, tuple]] = None,
    presence_saturation: float = 1.0,
    eps: float = 1e-8,
) -> LossFn:
    """Weighted final-head losses + optional stage-1 probe aux losses.

    ``cls_weight * CE(cls_logits) + cme_weight * soft_CE(cme)``
    and, if present / weights > 0:
    ``stage1_*_weight`` on ``cls_logits_stage1`` / ``cme_logits_stage1``.

    When the model returns ``cme_probs`` (presence gate), the main / aux CME
    SoftCE terms use those probs directly instead of ``softmax(cme_logits)``.

    Optional presence loss (``presence_weight > 0``, train-only): BCE between
    ``presence_logits`` and soft presence
    ``π*_c = 1 - exp(-m_c / (scale_c * sat))`` from ``batch.cme_mass``
    (unnormalized Gaussian type mass). ``presence_scale`` is usually a
    per-type vector (median of positive ``m_c`` per class). Canonical eval
    ignores this term.

    Optional multi-sigma CME aux (``aux_cme_weight > 0``): for each target in
    ``batch.soft_cme_aux`` (sigma → tensor), adds
    ``aux_cme_weight * mean(soft_CE(cme, target))``. Training-only;
    canonical eval ignores aux targets.

    Optional gene-similarity CME aux (``gene_sim_cme_weight > 0``): within each
    bag, softmax-weighted cosine gene similarity over neighbors builds a soft
    type target from hard labels or detached cls softmax; adds
    ``gene_sim_cme_weight * soft_CE(cme, target)``. Uses only
    ``node_features`` (no spatial inputs). Disabled when weight is 0.

    Optional soft-Spearman CME aux (``cme_spearman_weight > 0``): adds
    ``cme_spearman_weight * soft_spearman_cme_loss(...)`` so
    predicted class-importance ranks match GT soft ranks (temperature
    ``cme_spearman_tau``). Training-only; canonical eval ignores it.

    ``cme_divergence`` selects the CME training objective: ``"soft_ce"`` (default)
    or ``"js"`` (Jensen–Shannon). Set via config ``loss.cme_divergence``; it flows
    through ``loss_kwargs`` in ``ct_train.py`` (all ``loss`` keys except ``name``).
    Canonical eval always uses soft CE regardless of this setting.

    ``cme_temperature`` scales CME logits as ``logits / T`` before the main and
    stage-1 CME objectives (temperature scaling toward soft targets). Eval omits
    this (``T=1``) via ``build_canonical_eval_loss``. Ignored when ``cme_probs``
    is provided by a presence-gated model.

    ``cme_target_temperature`` sharpens (T<1) or softens (T>1) soft CME targets
    during training only; canonical eval uses raw σ=96 targets.

    ``cme_entropy_match_weight`` adds a training-only penalty
    ``mean(relu(H(pred) - H(target)))`` to discourage over-dispersed CME
    predictions relative to the soft target entropy. Canonical eval ignores it.

    ``bag_cme_weight`` adds a training-only bag-level soft CE between the
    masked-mean predicted CME distribution and the masked-mean soft target
    (global composition consistency). Canonical eval ignores it.
    """
    if cme_divergence not in ("soft_ce", "js"):
        raise ValueError(
            f"cme_divergence must be 'soft_ce' or 'js', got {cme_divergence!r}"
        )
    cme_fn = soft_js_divergence if cme_divergence == "js" else soft_cross_entropy

    cls_fn = make_cross_entropy(label_smoothing)
    w_cls = float(cls_weight)
    w_cme = float(cme_weight)
    w_s1_cls = float(stage1_cls_weight)
    w_s1_cme = float(stage1_cme_weight)
    w_aux_cme = float(aux_cme_weight)
    w_ent = float(cme_entropy_match_weight)
    w_bag = float(bag_cme_weight)
    w_gene_sim = float(gene_sim_cme_weight)
    w_spearman = float(cme_spearman_weight)
    w_presence = float(presence_weight)
    spearman_tau = float(cme_spearman_tau)
    spearman_chunk = int(cme_spearman_chunk_size)
    gene_temp = float(gene_sim_temperature)
    gene_src = str(gene_sim_source)
    cme_temp = float(cme_temperature)
    tgt_temp = float(cme_target_temperature)
    presence_sat = float(presence_saturation)
    if presence_scale is None:
        presence_scale_v: Optional[torch.Tensor] = None
    elif isinstance(presence_scale, (list, tuple)):
        presence_scale_v = torch.tensor(
            [float(x) for x in presence_scale], dtype=torch.float32
        )
    else:
        presence_scale_v = torch.tensor(
            [float(presence_scale)], dtype=torch.float32
        )

    def _scaled_cme_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits / cme_temp if cme_temp != 1.0 else logits

    def _tempered_target(soft: torch.Tensor) -> torch.Tensor:
        return temper_soft_target(soft, tgt_temp, eps=eps)

    def _cme_pred_term(
        outputs: Dict[str, torch.Tensor],
        soft: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        logits_key: str = "cme_logits",
    ) -> torch.Tensor:
        """SoftCE / JS on gated ``cme_probs`` when present, else logits."""
        if logits_key == "cme_logits" and "cme_probs" in outputs:
            if cme_divergence == "js":
                # soft_js expects logits; convert probs → log-space scores
                # that softmax recovers (approx) via log(p).
                scores = outputs["cme_probs"].clamp_min(eps).log()
                return soft_js_divergence(scores, soft, node_mask, eps=eps)
            return soft_cross_entropy_probs(
                outputs["cme_probs"], soft, node_mask, eps=eps
            )
        if logits_key not in outputs:
            raise ValueError(f"missing outputs['{logits_key}']")
        return cme_fn(
            _scaled_cme_logits(outputs[logits_key]),
            soft,
            node_mask,
            eps=eps,
        )

    def combined(
        outputs: Dict[str, torch.Tensor], batch: DataHolder
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(outputs, dict) or "cls_logits" not in outputs:
            raise ValueError("combined loss expects outputs['cls_logits']")
        cls_loss = cls_fn(outputs["cls_logits"], batch)
        parts = {"cls": cls_loss}
        total = w_cls * cls_loss

        if w_cme != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError(
                    "cme_weight>0 requires outputs['cme_logits'] or "
                    "outputs['cme_probs'] (enable model.predict_cme)"
                )
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError("batch.soft_cme is required when cme_weight>0")
            soft = _tempered_target(soft)
            cme_loss = _cme_pred_term(outputs, soft, batch.node_mask)
            parts["cme"] = cme_loss
            total = total + w_cme * cme_loss
        else:
            parts["cme"] = cls_loss.detach() * 0.0

        if w_presence != 0.0:
            if "presence_logits" not in outputs:
                raise ValueError(
                    "presence_weight>0 requires outputs['presence_logits'] "
                    "(enable model.cme_presence_gate)"
                )
            mass = getattr(batch, "cme_mass", None)
            if mass is None:
                raise ValueError(
                    "batch.cme_mass is required when presence_weight>0 "
                    "(recompute soft CME npz with cme_mass)"
                )
            scale = presence_scale_v
            if scale is None:
                raise ValueError(
                    "presence_weight>0 requires loss.presence_scale "
                    "(or auto-filled from soft-CME per-type medians)"
                )
            # Broadcast scalar-length-1 or length-C onto mass's class axis.
            if scale.numel() == 1:
                scale_t = scale.to(device=mass.device, dtype=mass.dtype)
            elif scale.numel() != mass.size(-1):
                raise ValueError(
                    f"presence_scale length {scale.numel()} != "
                    f"num_classes {mass.size(-1)}"
                )
            else:
                scale_t = scale.to(device=mass.device, dtype=mass.dtype)
            pi_star = soft_presence_from_mass(
                mass, scale_t, presence_saturation=presence_sat
            )
            presence_loss = masked_bce_presence(
                outputs["presence_logits"], pi_star, batch.node_mask
            )
            parts["presence"] = presence_loss
            total = total + w_presence * presence_loss
        else:
            parts["presence"] = cls_loss.detach() * 0.0

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
            soft = _tempered_target(soft)
            s1_cme = cme_fn(
                _scaled_cme_logits(outputs["cme_logits_stage1"]),
                soft,
                batch.node_mask,
                eps=eps,
            )
            parts["cme_stage1"] = s1_cme
            total = total + w_s1_cme * s1_cme

        if w_aux_cme != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError(
                    "aux_cme_weight>0 requires outputs['cme_logits'] or "
                    "outputs['cme_probs'] (enable model.predict_cme)"
                )
            aux = getattr(batch, "soft_cme_aux", None)
            if not aux:
                raise ValueError(
                    "aux_cme_weight>0 requires batch.soft_cme_aux "
                    "(set soft_cme.aux_sigmas and load aux tables)"
                )
            aux_losses = [
                _cme_pred_term(outputs, _tempered_target(target), batch.node_mask)
                for target in aux.values()
            ]
            cme_aux = torch.stack(aux_losses).mean()
            parts["cme_aux"] = cme_aux
            total = total + w_aux_cme * cme_aux
        else:
            parts["cme_aux"] = cls_loss.detach() * 0.0

        if w_ent != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError(
                    "cme_entropy_match_weight>0 requires CME outputs"
                )
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError(
                    "batch.soft_cme is required when cme_entropy_match_weight>0"
                )
            soft = _tempered_target(soft)
            if "cme_probs" in outputs:
                p = outputs["cme_probs"].clamp_min(eps)
                log_p = p.log()
            else:
                log_p = F.log_softmax(
                    _scaled_cme_logits(outputs["cme_logits"]), dim=-1
                )
                p = log_p.exp()
            h_pred = -(p * log_p).sum(dim=-1)
            h_tgt = -(soft.clamp_min(eps) * soft.clamp_min(eps).log()).sum(dim=-1)
            # Penalize only over-dispersion vs target (under-dispersion OK).
            excess = torch.relu(h_pred - h_tgt)
            mask = batch.node_mask.bool()
            ent_loss = excess[mask].mean() if mask.any() else excess.mean() * 0.0
            parts["cme_entropy"] = ent_loss
            total = total + w_ent * ent_loss
        else:
            parts["cme_entropy"] = cls_loss.detach() * 0.0

        if w_bag != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError("bag_cme_weight>0 requires CME outputs")
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError("batch.soft_cme is required when bag_cme_weight>0")
            soft = _tempered_target(soft)
            mask_f = batch.node_mask.unsqueeze(-1).to(dtype=soft.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
            bag_tgt = (soft * mask_f).sum(dim=1) / denom  # (B, C)
            bag_tgt = bag_tgt / bag_tgt.sum(dim=-1, keepdim=True).clamp_min(eps)
            if "cme_probs" in outputs:
                bag_p = (outputs["cme_probs"] * mask_f).sum(dim=1) / denom
            else:
                log_p = F.log_softmax(
                    _scaled_cme_logits(outputs["cme_logits"]), dim=-1
                )
                bag_p = (log_p.exp() * mask_f).sum(dim=1) / denom
            bag_p = bag_p.clamp_min(eps)
            bag_p = bag_p / bag_p.sum(dim=-1, keepdim=True)
            bag_loss = -(bag_tgt * bag_p.log()).sum(dim=-1).mean()
            parts["cme_bag"] = bag_loss
            total = total + w_bag * bag_loss
        else:
            parts["cme_bag"] = cls_loss.detach() * 0.0

        if w_gene_sim != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError(
                    "gene_sim_cme_weight>0 requires CME outputs "
                    "(enable model.predict_cme)"
                )
            num_classes = (
                outputs["cme_probs"].size(-1)
                if "cme_probs" in outputs
                else outputs["cme_logits"].size(-1)
            )
            target, has_neighbor = _gene_sim_neighbor_targets(
                batch,
                outputs,
                source=gene_src,
                temperature=gene_temp,
                num_classes=num_classes,
            )
            aux_mask = batch.node_mask.bool() & has_neighbor
            if "cme_probs" in outputs:
                gene_sim_loss = soft_cross_entropy_probs(
                    outputs["cme_probs"], target, aux_mask, eps=eps
                )
            else:
                gene_sim_loss = soft_cross_entropy(
                    outputs["cme_logits"], target, aux_mask, eps=eps
                )
            parts["cme_gene_sim"] = gene_sim_loss
            total = total + w_gene_sim * gene_sim_loss
        else:
            parts["cme_gene_sim"] = cls_loss.detach() * 0.0

        if w_spearman != 0.0:
            if "cme_logits" not in outputs and "cme_probs" not in outputs:
                raise ValueError(
                    "cme_spearman_weight>0 requires CME outputs "
                    "(enable model.predict_cme)"
                )
            soft = getattr(batch, "soft_cme", None)
            if soft is None:
                raise ValueError(
                    "batch.soft_cme is required when cme_spearman_weight>0"
                )
            soft = _tempered_target(soft)
            if "cme_probs" in outputs:
                spearman_loss = soft_spearman_cme_loss(
                    outputs.get("cme_logits", outputs["cme_probs"]),
                    soft,
                    batch.node_mask,
                    tau=spearman_tau,
                    eps=eps,
                    chunk_size=spearman_chunk,
                    probs=outputs["cme_probs"],
                )
            else:
                spearman_loss = soft_spearman_cme_loss(
                    _scaled_cme_logits(outputs["cme_logits"]),
                    soft,
                    batch.node_mask,
                    tau=spearman_tau,
                    eps=eps,
                    chunk_size=spearman_chunk,
                )
            parts["cme_spearman"] = spearman_loss
            total = total + w_spearman * spearman_loss
        else:
            parts["cme_spearman"] = cls_loss.detach() * 0.0

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
