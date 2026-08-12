"""Train ``CellTypeTransformer`` for per-cell type classification.

Usage (from repo root, LUNA env)::

    python exploration/cell_types/ct_train.py
    python exploration/cell_types/ct_train.py --config exploration/cell_types/ct_config.json

Training logs (console tee, history.json, wandb local files) are written under
``log_dir`` (default ``exploration/cell_types/logs/<run_name>/<timestamp>/``).
Checkpoints stay in ``output_dir/<run_name>/``.
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
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

# Repo root on sys.path when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUTPUT_DIR = "exploration/cell_types/outputs"
DEFAULT_LOG_DIR = "exploration/cell_types/logs"

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import (  # noqa: E402
    class_indices_from_batch,
    get_loss,
    soft_cross_entropy,
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


def attach_soft_cme_aux(
    batch,
    lookups: Dict[float, SoftCMELookup],
    device: torch.device,
):
    """Set ``batch.soft_cme_aux`` mapping sigma → soft target tensor."""
    if not lookups:
        return batch
    if batch.cell_ID is None:
        raise ValueError("batch.cell_ID is required to attach aux soft CME targets")
    batch.soft_cme_aux = {
        float(sigma): lookup.gather(batch.cell_ID, device)
        for sigma, lookup in lookups.items()
    }
    return batch


def soft_cme_sigmas(cfg: Dict[str, Any]) -> List[float]:
    """Primary + auxiliary soft-CME sigmas configured for training."""
    scfg = cfg.get("soft_cme") or {}
    primary = float(scfg["sigma"])
    aux = [float(s) for s in (scfg.get("aux_sigmas") or [])]
    sigmas: List[float] = [primary]
    for s in aux:
        if s not in sigmas:
            sigmas.append(s)
    return sigmas


def soft_cme_is_needed(cfg: Dict[str, Any]) -> bool:
    scfg = cfg.get("soft_cme") or {}
    mcfg = cfg.get("model") or {}
    lcfg = cfg.get("loss") or {}
    return (
        bool(scfg.get("enabled", False))
        or bool(mcfg.get("predict_cme", False))
        or str(lcfg.get("name", "")) == "combined"
        or float(lcfg.get("cme_weight", 0.0) or 0.0) != 0.0
        or float(lcfg.get("aux_cme_weight", 0.0) or 0.0) != 0.0
        or float(lcfg.get("cme_spearman_weight", 0.0) or 0.0) != 0.0
        or bool(scfg.get("aux_sigmas"))
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
    splits = ["train", "validation", "test"]
    sigmas = soft_cme_sigmas(cfg)
    missing_by_sigma: Dict[float, List[str]] = {}
    for sigma in sigmas:
        missing = [
            s for s in splits if not soft_cme_npz_path(out_dir, s, sigma).exists()
        ]
        if missing:
            missing_by_sigma[sigma] = missing
    if not missing_by_sigma:
        return

    exclude_self = bool(scfg.get("exclude_self", True))
    cutoff = scfg.get("cutoff_radius", None)
    cutoff = float(cutoff) if cutoff is not None else None
    for sigma, missing in missing_by_sigma.items():
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
            f"[ct_train] soft CME precompute done sigma={sigma}: "
            + ", ".join(f"{k}={v.name}" for k, v in written.items()),
            flush=True,
        )


def load_soft_cme_lookup(
    cfg: Dict[str, Any],
    split: str,
    *,
    sigma: Optional[float] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[SoftCMELookup]:
    if not soft_cme_is_needed(cfg):
        return None
    scfg = cfg.get("soft_cme") or {}
    out_dir = soft_cme_out_dir(cfg)
    if sigma is None:
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


def load_soft_cme_aux_lookups(
    cfg: Dict[str, Any],
    split: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[float, SoftCMELookup]:
    """Load auxiliary soft-CME tables for ``soft_cme.aux_sigmas`` (excl. primary)."""
    scfg = cfg.get("soft_cme") or {}
    primary = float(scfg["sigma"])
    aux_sigmas = [float(s) for s in (scfg.get("aux_sigmas") or [])]
    lookups: Dict[float, SoftCMELookup] = {}
    for sigma in aux_sigmas:
        if sigma == primary:
            continue
        lookup = load_soft_cme_lookup(
            cfg, split, sigma=sigma, config_path=config_path
        )
        if lookup is not None:
            lookups[sigma] = lookup
    return lookups


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
    """Create a unique per-run directory under ``log_dir`` and tee stdout/stderr.

    Layout::

        exploration/cell_types/logs/<run_name>/<YYYYMMDD_HHMMSS>/
            train.log
            history.json
            ct_config.snapshot.json
            wandb/ …
    """
    log_root = resolve_repo_path(cfg.get("log_dir", DEFAULT_LOG_DIR))
    run_name = str(cfg.get("run_name", "ct_transformer"))
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
        gene_mlp_depth=int(mcfg.get("gene_mlp_depth", 2)),
        layer_activation=str(mcfg.get("layer_activation", "relu")),
        cls_activation=str(mcfg.get("cls_activation", "relu")),
        cme_activation=str(mcfg.get("cme_activation", mcfg.get("cls_activation", "relu"))),
        fusion_activation=mcfg.get("fusion_activation", None),
        global_skip=mcfg.get("global_skip", None),
        subgraph_summary=bool(mcfg.get("subgraph_summary", False)),
        subgraph_pool=str(mcfg.get("subgraph_pool", "mean")),
        subgraph_pool_source=str(mcfg.get("subgraph_pool_source", "mixed")),
        attention_use_node_mask=bool(mcfg.get("attention_use_node_mask", False)),
        cme_detach_type_in_fusion=bool(
            mcfg.get("cme_detach_type_in_fusion", False)
        ),
        dual_fusion=bool(mcfg.get("dual_fusion", False)),
        gene_knn_k=int(mcfg.get("gene_knn_k", 0) or 0),
        pre_norm=bool(mcfg.get("pre_norm", False)),
        cme_condition_on_cls=bool(mcfg.get("cme_condition_on_cls", False)),
        cme_entropy_temp_t0=float(mcfg.get("cme_entropy_temp_t0", 1.0)),
        cme_entropy_temp_alpha=float(mcfg.get("cme_entropy_temp_alpha", 0.0)),
        predict_cme=predict_cme,
        feature_cross=bool(mcfg.get("feature_cross", False)),
        feature_cross_include_h=bool(mcfg.get("feature_cross_include_h", True)),
    )


def build_canonical_eval_loss(cfg: Dict[str, Any]):
    """Fixed eval objective: CE(cls, ls=0.05) + softCE(cme, σ=96 targets).

    Independent of training loss weights / aux heads so experiments remain
    comparable. ``label_smoothing`` is hardcoded to 0.05 for cls CE even when
    training uses a different value (e.g. exp012 trains with ls=0). CME uses
    raw logits (``cme_temperature=1``) regardless of training temperature.
    """
    return get_loss(
        "combined",
        label_smoothing=0.05,
        cls_weight=1.0,
        cme_weight=1.0,
        stage1_cls_weight=0.0,
        stage1_cme_weight=0.0,
    )


def load_ema_weights_into_model(
    model: CellTypeTransformer, path: Path, device: torch.device
) -> None:
    """Load EMA shadow (preferred) or raw model weights from a checkpoint."""
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


def load_teacher_ensemble(
    cfg: Dict[str, Any],
    ckpt_paths: List[str],
    gene_dim: int,
    num_classes: int,
    device: torch.device,
) -> List[CellTypeTransformer]:
    """Build frozen teacher models (same arch as ``cfg``) from checkpoints."""
    teachers: List[CellTypeTransformer] = []
    for raw in ckpt_paths:
        path = resolve_repo_path(raw)
        t = build_model(cfg, gene_dim=gene_dim, num_classes=num_classes).to(device)
        load_ema_weights_into_model(t, path, device)
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        teachers.append(t)
        print(f"[ct_train] teacher loaded: {path}", flush=True)
    return teachers


def _normalize_teacher_weights(
    weights: Optional[List[float]], n: int, name: str
) -> List[float]:
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"{name} len={len(weights)} != n_teachers={n}")
    w_sum = float(sum(weights))
    if w_sum <= 0.0:
        raise ValueError(f"{name} must sum to > 0")
    return [float(w) / w_sum for w in weights]


@torch.no_grad()
def teacher_mean_logits(
    teachers: List[CellTypeTransformer],
    batch,
    weights: Optional[List[float]] = None,
    cls_weights: Optional[List[float]] = None,
    cme_weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Average teacher logits; optional separate CLS/CME mix weights."""
    n = len(teachers)
    # Shared ``weights`` is the default for both heads; per-head overrides win.
    base = _normalize_teacher_weights(weights, n, "teacher_weights")
    w_cls = (
        _normalize_teacher_weights(cls_weights, n, "cls_teacher_weights")
        if cls_weights is not None
        else base
    )
    w_cme = (
        _normalize_teacher_weights(cme_weights, n, "cme_teacher_weights")
        if cme_weights is not None
        else base
    )

    cls_sum = None
    cme_sum = None
    for t, wc, wm in zip(teachers, w_cls, w_cme):
        if wc == 0.0 and wm == 0.0:
            continue
        out = t(batch)
        if not isinstance(out, dict):
            out = {"cls_logits": out}
        if wc != 0.0:
            cls = out["cls_logits"] * wc
            cls_sum = cls if cls_sum is None else cls_sum + cls
        if wm != 0.0 and "cme_logits" in out:
            cme = out["cme_logits"] * wm
            cme_sum = cme if cme_sum is None else cme_sum + cme
    if cls_sum is None:
        raise ValueError("all CLS teacher weights were zero")
    result = {"cls_logits": cls_sum}
    if cme_sum is not None:
        result["cme_logits"] = cme_sum
    return result


def apply_cme_entropy_temperature(
    cme_logits: torch.Tensor,
    t0: float = 1.0,
    alpha: float = 0.0,
    t_min: float = 0.7,
    t_max: float = 1.4,
) -> torch.Tensor:
    """Per-cell CME logit temperature from predictive entropy: T=t0+α(H-H̄)."""
    if abs(float(alpha)) < 1e-12 and abs(float(t0) - 1.0) < 1e-12:
        return cme_logits
    logp = F.log_softmax(cme_logits, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1, keepdim=True)
    T = (float(t0) + float(alpha) * (H - H.mean())).clamp(float(t_min), float(t_max))
    return cme_logits / T


def masked_kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    node_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student) averaged over real cells."""
    t = float(temperature)
    if t <= 0.0:
        raise ValueError(f"distill temperature must be > 0 (got {temperature})")
    log_p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    per_cell = F.kl_div(log_p, q, reduction="none").sum(dim=-1) * (t * t)
    mask = node_mask.bool()
    vals = per_cell[mask]
    if vals.numel() == 0:
        return student_logits.sum() * 0.0
    return vals.mean()


def distill_loss_from_teachers(
    outputs: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    batch,
    *,
    cls_weight: float,
    cme_weight: float,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """KD terms: CLS via tempered KL; CME via softCE to teacher probs."""
    mask = batch.node_mask
    parts: Dict[str, torch.Tensor] = {}
    total = outputs["cls_logits"].sum() * 0.0
    if cls_weight > 0.0:
        kd_cls = masked_kd_kl(
            outputs["cls_logits"],
            teacher_out["cls_logits"],
            mask,
            temperature=temperature,
        )
        parts["kd_cls"] = kd_cls
        total = total + float(cls_weight) * kd_cls
    if cme_weight > 0.0 and "cme_logits" in outputs and "cme_logits" in teacher_out:
        # Match teacher predictive distribution at T=1 (canonical CME scale).
        teacher_probs = F.softmax(teacher_out["cme_logits"], dim=-1)
        kd_cme = soft_cross_entropy(outputs["cme_logits"], teacher_probs, mask)
        parts["kd_cme"] = kd_cme
        total = total + float(cme_weight) * kd_cme
    parts["kd"] = total
    return total, parts


class ModelEMA:
    """Exponential moving average of trainable model parameters."""

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
        """Temporarily replace model weights with EMA shadow copies."""
        self._backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: CellTypeTransformer) -> None:
        """Restore model weights saved by ``apply_shadow``."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self._backup = {}


def build_scheduler(optimizer: AdamW, cfg: Dict[str, Any], n_epochs: int):
    """Optional LR schedule from ``train.scheduler``."""
    tcfg = cfg.get("train") or {}
    scfg = tcfg.get("scheduler") or {}
    name = str(scfg.get("name", "none")).strip().lower()
    if name in ("", "none", "null", "off"):
        return None
    if name in ("cosine", "cosine_warmup"):
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        min_lr = float(scfg.get("min_lr", 1e-6))
        # Prefer explicit t_max so early-stopped runs still see LR decay.
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
            mode=str(scfg.get("mode", "min")),
            factor=float(scfg.get("factor", 0.5)),
            patience=int(scfg.get("patience", 8)),
            min_lr=float(scfg.get("min_lr", 1e-6)),
            threshold=float(scfg.get("threshold", 0.0003)),
        )
    raise ValueError(f"Unknown scheduler '{name}'")


def setup_wandb_run(
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    log_dir: Path,
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
    soft_cme_aux_lookups: Optional[Dict[float, SoftCMELookup]] = None,
    canonical_eval_fn=None,
    ema: Optional[ModelEMA] = None,
    teachers: Optional[List[CellTypeTransformer]] = None,
    distill_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    use_distill = bool(
        train
        and teachers
        and distill_cfg
        and distill_cfg.get("enabled", True)
    )
    hard_weight = float((distill_cfg or {}).get("hard_weight", 1.0))
    kd_cls_w = float((distill_cfg or {}).get("cls_weight", 1.0))
    kd_cme_w = float((distill_cfg or {}).get("cme_weight", 1.0))
    kd_temp = float((distill_cfg or {}).get("temperature", 2.0))
    teacher_weights = (distill_cfg or {}).get("teacher_weights")
    if teacher_weights is not None:
        teacher_weights = [float(w) for w in teacher_weights]
    cls_teacher_weights = (distill_cfg or {}).get("cls_teacher_weights")
    if cls_teacher_weights is not None:
        cls_teacher_weights = [float(w) for w in cls_teacher_weights]
    cme_teacher_weights = (distill_cfg or {}).get("cme_teacher_weights")
    if cme_teacher_weights is not None:
        cme_teacher_weights = [float(w) for w in cme_teacher_weights]

    total_loss = 0.0
    total_cls_loss = 0.0
    total_cme_loss = 0.0
    total_cme_aux_loss = 0.0
    total_cme_spearman_loss = 0.0
    total_kd_loss = 0.0
    total_canon_cls = 0.0
    total_canon_cme = 0.0
    use_canon = canonical_eval_fn is not None
    total_correct = 0
    total_cells = 0
    n_batches = 0
    correct_counts = torch.zeros(num_classes, dtype=torch.long)
    total_counts = torch.zeros(num_classes, dtype=torch.long)

    for step, raw in enumerate(loader):
        batch = to_batch(raw, device=device)
        if soft_cme_lookup is not None:
            attach_soft_cme(batch, soft_cme_lookup, device)
        if soft_cme_aux_lookups:
            attach_soft_cme_aux(batch, soft_cme_aux_lookups, device)

        outputs = model(batch)
        if not isinstance(outputs, dict):
            outputs = {"cls_logits": outputs}

        loss_out = loss_fn(outputs, batch)
        if isinstance(loss_out, tuple):
            loss, parts = loss_out
            cls_loss_val = float(parts["cls"].detach().item())
            cme_loss_val = float(parts.get("cme", parts["cls"] * 0).detach().item())
            cme_aux_loss_val = float(
                parts.get("cme_aux", parts["cls"] * 0).detach().item()
            )
            cme_spearman_loss_val = float(
                parts.get("cme_spearman", parts["cls"] * 0).detach().item()
            )
        else:
            # Hard-CE-only path still expects logits tensor historically.
            loss = loss_out
            cls_loss_val = float(loss.detach().item())
            cme_loss_val = 0.0
            cme_aux_loss_val = 0.0
            cme_spearman_loss_val = 0.0

        kd_loss_val = 0.0
        if use_distill:
            teacher_out = teacher_mean_logits(
                teachers,
                batch,
                weights=teacher_weights,
                cls_weights=cls_teacher_weights,
                cme_weights=cme_teacher_weights,
            )
            if "cme_logits" in teacher_out and (
                abs(float((distill_cfg or {}).get("entropy_temp_alpha", 0.0))) > 1e-12
                or abs(float((distill_cfg or {}).get("entropy_temp_t0", 1.0)) - 1.0) > 1e-12
            ):
                teacher_out = dict(teacher_out)
                teacher_out["cme_logits"] = apply_cme_entropy_temperature(
                    teacher_out["cme_logits"],
                    t0=float((distill_cfg or {}).get("entropy_temp_t0", 1.0)),
                    alpha=float((distill_cfg or {}).get("entropy_temp_alpha", 0.0)),
                )
            kd_term, _kd_parts = distill_loss_from_teachers(
                outputs,
                teacher_out,
                batch,
                cls_weight=kd_cls_w,
                cme_weight=kd_cme_w,
                temperature=kd_temp,
            )
            loss = hard_weight * loss + kd_term
            kd_loss_val = float(kd_term.detach().item())

        if use_canon:
            with torch.no_grad():
                canon_out = canonical_eval_fn(outputs, batch)
            if isinstance(canon_out, tuple):
                _, canon_parts = canon_out
                canon_cls_val = float(canon_parts["cls"].detach().item())
                canon_cme_val = float(
                    canon_parts.get("cme", canon_parts["cls"] * 0).detach().item()
                )
            else:
                canon_cls_val = float(canon_out.detach().item())
                canon_cme_val = 0.0
            total_canon_cls += canon_cls_val
            total_canon_cme += canon_cme_val

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        cls_logits = outputs["cls_logits"].detach()
        acc, n = masked_accuracy(cls_logits, batch)
        update_per_class_counts(cls_logits, batch, correct_counts, total_counts)
        loss_val = float(loss.detach().item())
        total_loss += loss_val
        total_cls_loss += cls_loss_val
        total_cme_loss += cme_loss_val
        total_cme_aux_loss += cme_aux_loss_val
        total_cme_spearman_loss += cme_spearman_loss_val
        total_kd_loss += kd_loss_val
        total_correct += int(acc * n)
        total_cells += n
        n_batches += 1

        if log_every and train and (step + 1) % log_every == 0:
            extra = f" kd={kd_loss_val:.4f}" if use_distill else ""
            if cme_spearman_loss_val != 0.0:
                extra += f" spr={cme_spearman_loss_val:.4f}"
            print(
                f"  [{split}] epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss_val:.4f} cls={cls_loss_val:.4f} "
                f"cme={cme_loss_val:.4f} acc={acc:.4f}{extra}",
                flush=True,
            )
            if wandb_log_steps:
                payload = {
                    f"{split}/step_loss": loss_val,
                    f"{split}/step_cls_loss": cls_loss_val,
                    f"{split}/step_cme_loss": cme_loss_val,
                    f"{split}/step_cme_spearman": cme_spearman_loss_val,
                    f"{split}/step_acc": acc,
                    "epoch": epoch,
                }
                if use_distill:
                    payload[f"{split}/step_kd_loss"] = kd_loss_val
                wandb_log(
                    payload,
                    step=global_step_offset + step + 1,
                )

    mean_loss = total_loss / max(n_batches, 1)
    mean_acc = total_correct / max(total_cells, 1)
    mean_cls = total_cls_loss / max(n_batches, 1)
    mean_cme = total_cme_loss / max(n_batches, 1)
    mean_cme_aux = total_cme_aux_loss / max(n_batches, 1)
    mean_cme_spearman = total_cme_spearman_loss / max(n_batches, 1)
    mean_kd = total_kd_loss / max(n_batches, 1)
    if use_canon and not train:
        mean_cls = total_canon_cls / max(n_batches, 1)
        mean_cme = total_canon_cme / max(n_batches, 1)
    result: Dict[str, Any] = {
        "loss": mean_loss,
        "cls_loss": mean_cls,
        "cme_loss": mean_cme,
        "cme_aux_loss": mean_cme_aux,
        "cme_spearman_loss": mean_cme_spearman,
        "kd_loss": mean_kd,
        "acc": mean_acc,
        "n_cells": float(total_cells),
        "per_class_acc": per_class_accuracy_list(correct_counts, total_counts),
        "per_class_n": [int(x) for x in total_counts.tolist()],
    }
    if use_canon:
        result["eval_loss"] = (
            total_canon_cls + total_canon_cme
        ) / max(n_batches, 1)
    return result


def save_checkpoint(
    path: Path,
    model: CellTypeTransformer,
    optimizer: AdamW,
    epoch: int,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    ema: Optional[ModelEMA] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
        "metrics": metrics,
        "gene_dim": model.gene_dim,
        "num_classes": model.num_classes,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "cuda")))

    out_dir = resolve_repo_path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    run_dir = out_dir / str(cfg.get("run_name", "ct_transformer"))
    run_dir.mkdir(parents=True, exist_ok=True)

    log_run_dir = setup_train_log_dir(cfg)
    with (log_run_dir / "ct_config.snapshot.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    # Keep a copy next to checkpoints for convenience.
    with (run_dir / "ct_config.snapshot.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[ct_train] config={args.config}")
    print(f"[ct_train] device={device}  checkpoints={run_dir}")
    print(f"[ct_train] logs={log_run_dir}")

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

    init_ckpt = cfg.get("init_checkpoint") or (cfg.get("train") or {}).get(
        "init_checkpoint"
    )
    if init_ckpt:
        init_path = resolve_repo_path(str(init_ckpt))
        load_ema_weights_into_model(model, init_path, device)
        print(f"[ct_train] initialized student from {init_path}", flush=True)

    distill_cfg = dict((cfg.get("train") or {}).get("distill") or {})
    teachers: Optional[List[CellTypeTransformer]] = None
    if distill_cfg.get("enabled"):
        ckpts = list(distill_cfg.get("ckpts") or [])
        if not ckpts:
            raise ValueError("train.distill.enabled requires train.distill.ckpts")
        teachers = load_teacher_ensemble(
            cfg, ckpts, gene_dim=gene_dim, num_classes=num_classes, device=device
        )
        print(
            f"[ct_train] distillation ON: n_teachers={len(teachers)} "
            f"hard_w={distill_cfg.get('hard_weight', 1.0)} "
            f"kd_cls={distill_cfg.get('cls_weight', 1.0)} "
            f"kd_cme={distill_cfg.get('cme_weight', 1.0)} "
            f"T={distill_cfg.get('temperature', 2.0)}",
            flush=True,
        )

    ensure_soft_cme_precomputed(cfg, args.config)
    train_cme = load_soft_cme_lookup(cfg, "train", config_path=args.config)
    val_cme = load_soft_cme_lookup(cfg, "validation", config_path=args.config)
    train_cme_aux = load_soft_cme_aux_lookups(
        cfg, "train", config_path=args.config
    )
    val_cme_aux = load_soft_cme_aux_lookups(
        cfg, "validation", config_path=args.config
    )
    lcfg = cfg.get("loss") or {}
    aux_cme_weight = float(lcfg.get("aux_cme_weight", 0.0) or 0.0)
    aux_sigmas = [float(s) for s in (cfg.get("soft_cme") or {}).get("aux_sigmas") or []]
    if aux_cme_weight > 0.0 and not aux_sigmas:
        raise ValueError("loss.aux_cme_weight>0 requires soft_cme.aux_sigmas")
    if aux_cme_weight > 0.0 and not train_cme_aux:
        raise ValueError(
            "loss.aux_cme_weight>0 but no aux soft CME tables loaded "
            "(check soft_cme.aux_sigmas / out_dir)"
        )
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
        log_dir=log_run_dir,
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
    eval_loss_fn = build_canonical_eval_loss(cfg)
    tcfg = cfg["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    n_epochs = int(tcfg["n_epochs"])
    scheduler = build_scheduler(optimizer, cfg, n_epochs)
    if scheduler is not None:
        scfg = tcfg.get("scheduler") or {}
        print(
            f"[ct_train] scheduler={scfg.get('name')} "
            f"warmup_epochs={scfg.get('warmup_epochs', 0)} "
            f"min_lr={scfg.get('min_lr', 1e-6)}",
            flush=True,
        )
    ema_cfg = tcfg.get("ema") or {}
    use_ema = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema: Optional[ModelEMA] = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)
        print(f"[ct_train] EMA enabled decay={ema_decay}", flush=True)

    train_loader = datamodule.train_dataloader()
    val_loader = None
    if getattr(datamodule, "validation_dataset", None) is not None:
        val_loader = datamodule.validation_dataloader()

    best_val_acc = -1.0
    best_val_eval_loss = float("inf")
    epochs_without_improve = 0
    history = []
    val_every = int(tcfg.get("val_every_n_epochs", 1))
    save_every = int(tcfg.get("save_every_n_epochs", 10))
    log_every = int(tcfg.get("log_every", 0))
    grad_clip = tcfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    rechunk_every = int(tcfg.get("rechunk_every_n_epochs", 0))
    es_cfg = tcfg.get("early_stopping") or {}
    early_stop = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 20))
    es_monitor = str(es_cfg.get("monitor", "val/eval_loss"))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_uses_eval_loss = es_monitor in ("val/loss", "val/eval_loss")
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
                soft_cme_aux_lookups=train_cme_aux or None,
                ema=ema,
                teachers=teachers,
                distill_cfg=distill_cfg if teachers else None,
            )
            global_step += len(train_loader)
            kd_msg = (
                f" kd={train_metrics['kd_loss']:.4f}"
                if teachers and train_metrics.get("kd_loss", 0.0) > 0
                else ""
            )
            print(
                f"[train] loss={train_metrics['loss']:.4f} "
                f"cls={train_metrics['cls_loss']:.4f} "
                f"cme={train_metrics['cme_loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} "
                f"cells={int(train_metrics['n_cells'])}{kd_msg}",
                flush=True,
            )

            val_metrics = None
            if val_loader is not None and (epoch % val_every == 0):
                if ema is not None:
                    ema.apply_shadow(model)
                try:
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
                        soft_cme_aux_lookups=val_cme_aux or None,
                        canonical_eval_fn=eval_loss_fn,
                    )
                finally:
                    if ema is not None:
                        ema.restore(model)
                print(
                    f"[val]   loss={val_metrics['loss']:.4f} "
                    f"eval={val_metrics.get('eval_loss', val_metrics['loss']):.4f} "
                    f"cls={val_metrics['cls_loss']:.4f} "
                    f"cme={val_metrics['cme_loss']:.4f} "
                    f"acc={val_metrics['acc']:.4f} "
                    f"cells={int(val_metrics['n_cells'])}",
                    flush=True,
                )
                # Strict best for checkpointing; min_delta only for ES patience.
                is_best = False
                es_improved = False
                prev_best_acc = best_val_acc
                prev_best_eval = best_val_eval_loss
                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]

                if es_uses_eval_loss:
                    eval_loss = float(val_metrics["eval_loss"])
                    if eval_loss < best_val_eval_loss:
                        best_val_eval_loss = eval_loss
                        is_best = True
                    if eval_loss < prev_best_eval - es_min_delta:
                        es_improved = True
                elif es_monitor == "val/acc":
                    if val_metrics["acc"] > prev_best_acc:
                        is_best = True
                    if val_metrics["acc"] > prev_best_acc + es_min_delta:
                        es_improved = True
                else:
                    raise ValueError(
                        f"Unknown early_stopping.monitor '{es_monitor}' "
                        "(expected val/eval_loss, val/loss, or val/acc)"
                    )

                if is_best:
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        epoch,
                        cfg,
                        {"train": train_metrics, "val": val_metrics},
                        ema=ema,
                    )
                    print(
                        f"[ct_train] saved best checkpoint "
                        f"(val_eval_loss={best_val_eval_loss:.4f} "
                        f"val_acc={best_val_acc:.4f})",
                        flush=True,
                    )
                    if use_wandb:
                        wandb_log(
                            {
                                "best_val_acc": best_val_acc,
                                "best_val_eval_loss": best_val_eval_loss,
                                "best_epoch": epoch,
                            },
                            step=global_step,
                        )

                if early_stop:
                    if es_improved:
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
                "train/cme_spearman": train_metrics["cme_spearman_loss"],
                "train/acc": train_metrics["acc"],
                "train/n_cells": train_metrics["n_cells"],
            }
            if val_metrics is not None:
                epoch_log.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/cls_loss": val_metrics["cls_loss"],
                        "val/cme_loss": val_metrics["cme_loss"],
                        "val/cme_spearman": val_metrics["cme_spearman_loss"],
                        "val/eval_loss": val_metrics["eval_loss"],
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
                "cme_spearman_loss",
                "eval_loss",
                "acc",
                "n_cells",
                "per_class_acc",
                "per_class_n",
            )
            record = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train": {
                    k: v for k, v in train_metrics.items() if k in _metric_keys
                },
                "val": None
                if val_metrics is None
                else {k: v for k, v in val_metrics.items() if k in _metric_keys},
            }
            history.append(record)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    metric = None
                    if val_metrics is not None:
                        metric = float(
                            val_metrics.get("eval_loss", val_metrics["loss"])
                        )
                    if metric is not None:
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
                    {"train": train_metrics, "val": val_metrics},
                    ema=ema,
                )

            with (log_run_dir / "history.json").open("w") as f:
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
                "best_val_eval_loss": best_val_eval_loss,
            },
            ema=ema,
        )
        if use_wandb:
            wandb_log(
                {
                    "best_val_acc": best_val_acc,
                    "best_val_eval_loss": best_val_eval_loss,
                },
                step=global_step,
            )
            wandb.summary["best_val_acc"] = best_val_acc
            wandb.summary["best_val_eval_loss"] = best_val_eval_loss
        print(
            f"\n[ct_train] done. best_val_acc={best_val_acc:.4f}  "
            f"best_val_eval_loss={best_val_eval_loss:.4f}  "
            f"checkpoints={run_dir}  logs={log_run_dir}"
        )
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

