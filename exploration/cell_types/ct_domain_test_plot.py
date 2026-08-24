"""Test the best domain checkpoint and plot accuracy by domain and cell type.

Given a run folder (config snapshot + ``best.pt``), rebuilds the model, runs
the test split, and writes:

* ``test_acc_per_domain.csv``
* ``test_acc_per_cell_type.csv``
* ``test_acc_histograms.png`` / ``.pdf`` — two super-wide bar subplots

Usage (from repo root)::

    python exploration/cell_types/ct_domain_test_plot.py \\
        exploration/cell_types/outputs/ct_transformer_abc_domains_best
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.data_module import DataModule, Infos  # noqa: E402
from exploration.cell_types.ct_losses import class_indices_from_batch  # noqa: E402
from exploration.cell_types.ct_train_domains import (  # noqa: E402
    apply_domain_labels,
    attach_aux_to_batch,
    build_cme_lookup,
    build_datamodule_cfg,
    build_model,
    build_type_lookup,
    load_config,
    resolve_device,
    resolve_repo_path,
    set_seed,
)
from exploration.cell_types.ct_transformer import CellTypeTransformer  # noqa: E402
from utils.data.misc import to_batch  # noqa: E402

CONFIG_NAME = "ct_config.snapshot.json"
BEST_CKPT_NAMES = ("best.pt", "best.ckpt")
UNK_TYPE_NAME = "UNK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best domain checkpoint on test data and plot "
            "accuracy per domain / cell type"
        )
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help=(
            "Run folder with ct_config.snapshot.json and best.pt "
            "(e.g. exploration/cell_types/outputs/ct_transformer_abc_domains_best)"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config JSON (default: <run_dir>/ct_config.snapshot.json)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override checkpoint path (default: <run_dir>/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write CSVs and figure (default: the run folder)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "validation"),
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config device (e.g. cuda, cuda:0, cpu)",
    )
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = resolve_repo_path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Run folder not found: {path}")
    return path


def find_best_checkpoint(run_dir: Path) -> Path:
    for name in BEST_CKPT_NAMES:
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No best checkpoint in {run_dir} (looked for {', '.join(BEST_CKPT_NAMES)})"
    )


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_weights(model: CellTypeTransformer, ckpt: Dict[str, Any]) -> str:
    """Load EMA shadow if present, otherwise raw ``model_state_dict``."""
    ema = ckpt.get("ema_state_dict")
    if ema and isinstance(ema, dict) and ema.get("shadow"):
        shadow = ema["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(
                        shadow[name].to(device=param.device, dtype=param.dtype)
                    )
        return "ema"
    state = ckpt.get("model_state_dict")
    if not state:
        raise KeyError("checkpoint missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return "model_state_dict"


def skip_unused_validation(cfg: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Avoid loading the val CSV twice when only the test split is needed."""
    if split != "test":
        return cfg
    cfg = copy.deepcopy(cfg)
    dataset = cfg.setdefault("dataset", {})
    dataset["validation_data_path"] = ""
    return cfg


def _as_1d_long(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.dim() == 3:
        values = values.squeeze(-1)
    return values[mask].long()


def _counts_to_rows(
    names: List[str],
    correct: torch.Tensor,
    total: torch.Tensor,
    name_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        n = int(total[i].item()) if i < total.numel() else 0
        n_ok = int(correct[i].item()) if i < correct.numel() else 0
        acc = (float(n_ok) / float(n)) if n > 0 else np.nan
        rows.append({name_col: name, "acc": acc, "n": n, "n_correct": n_ok})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_split(
    model: CellTypeTransformer,
    loader,
    device: torch.device,
    *,
    num_domains: int,
    domain_names: List[str],
    type_names: List[str],
    type_table: Optional[torch.Tensor],
    cme_table: Optional[torch.Tensor],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    model.eval()
    type_labels = [UNK_TYPE_NAME] + list(type_names)
    n_types = len(type_labels)

    domain_correct = torch.zeros(num_domains, dtype=torch.long)
    domain_total = torch.zeros(num_domains, dtype=torch.long)
    type_correct = torch.zeros(n_types, dtype=torch.long)
    type_total = torch.zeros(n_types, dtype=torch.long)

    if type_table is not None:
        type_table = type_table.to(device)
    if cme_table is not None:
        cme_table = cme_table.to(device)

    n_batches = len(loader)
    print(f"[ct_test_plot] evaluating {n_batches} batches …", flush=True)
    for step, raw in enumerate(loader):
        batch = to_batch(raw, device=device)
        attach_aux_to_batch(batch, device, type_table, cme_table)
        outputs = model(batch)
        if not isinstance(outputs, dict):
            outputs = {"cls_logits": outputs}
        logits = outputs["cls_logits"]

        mask = batch.node_mask.bool()
        targets = class_indices_from_batch(batch)
        preds = logits.argmax(dim=-1)
        targets_m = targets[mask]
        preds_m = preds[mask]
        correct = preds_m == targets_m

        t_cpu = targets_m.detach().cpu()
        ok_cpu = correct.detach().cpu()
        domain_total += torch.bincount(t_cpu, minlength=num_domains)
        if ok_cpu.any():
            domain_correct += torch.bincount(t_cpu[ok_cpu], minlength=num_domains)

        if getattr(batch, "cell_type", None) is not None:
            type_ids = _as_1d_long(batch.cell_type, mask).clamp(0, n_types - 1).cpu()
            type_total += torch.bincount(type_ids, minlength=n_types)
            if ok_cpu.any():
                type_correct += torch.bincount(
                    type_ids[ok_cpu], minlength=n_types
                )

        if (step + 1) % 10 == 0 or (step + 1) == n_batches:
            seen = int(domain_total.sum().item())
            acc_so_far = (
                float(domain_correct.sum().item()) / float(seen) if seen else 0.0
            )
            print(
                f"  [test] batch={step + 1}/{n_batches}  "
                f"cells={seen:,}  acc={acc_so_far:.4f}",
                flush=True,
            )

    domain_df = _counts_to_rows(
        domain_names, domain_correct, domain_total, "domain"
    )
    type_df = _counts_to_rows(type_labels, type_correct, type_total, "cell_type")
    # Drop UNK if the split never produced it.
    if (
        not type_df.empty
        and type_df.iloc[0]["cell_type"] == UNK_TYPE_NAME
        and int(type_df.iloc[0]["n"]) == 0
    ):
        type_df = type_df.iloc[1:].reset_index(drop=True)

    n_cells = int(domain_total.sum().item())
    n_ok = int(domain_correct.sum().item())
    scored = domain_df.loc[domain_df["n"] > 0, "acc"]
    summary = {
        "micro_acc": (float(n_ok) / float(n_cells)) if n_cells else float("nan"),
        "macro_acc": float(scored.mean()) if len(scored) else float("nan"),
        "n_cells": float(n_cells),
        "n_correct": float(n_ok),
        "n_domains": float(len(domain_df)),
        "n_cell_types": float(len(type_df)),
    }
    return domain_df, type_df, summary


def _annotate_bars(ax: plt.Axes, bars, accs: np.ndarray, ns: np.ndarray) -> None:
    y_max = ax.get_ylim()[1]
    for bar, acc, n in zip(bars, accs, ns):
        if not np.isfinite(acc):
            label = f"n/a\nn={int(n):,}"
            y = 0.02 * y_max
        else:
            label = f"{acc:.3f}\nn={int(n):,}"
            y = bar.get_height() + 0.012 * y_max
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            clip_on=False,
        )


def _draw_acc_bars(
    ax: plt.Axes,
    names: List[str],
    accs: np.ndarray,
    ns: np.ndarray,
    *,
    title: str,
    xlabel: str,
    micro_acc: Optional[float],
    cmap_name: str,
) -> None:
    x = np.arange(len(names))
    finite = np.where(np.isfinite(accs), accs, 0.0)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.clip(finite, 0.0, 1.0))
    bars = ax.bar(x, finite, color=colors, edgecolor="0.25", linewidth=0.4, width=0.82)

    ax.set_ylim(0.0, 1.12)
    ax.set_yticks(np.arange(0.0, 1.01, 0.05))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, max(len(names) - 0.3, 0.3))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if micro_acc is not None and np.isfinite(micro_acc):
        ax.axhline(
            micro_acc,
            color="0.2",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"micro acc = {micro_acc:.4f}",
        )
        ax.legend(loc="upper right", frameon=False)
    _annotate_bars(ax, bars, accs, ns)


def plot_accuracy_histograms(
    domain_df: pd.DataFrame,
    type_df: pd.DataFrame,
    summary: Dict[str, float],
    out_stem: Path,
    *,
    split: str,
    run_name: str,
) -> List[Path]:
    n_domain = max(len(domain_df), 1)
    n_type = max(len(type_df), 1)
    width = max(32.0, 0.95 * max(n_domain, n_type))
    height = 13.5
    fig, (ax_domain, ax_type) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        constrained_layout=True,
    )

    micro = summary.get("micro_acc")
    n_cells = int(summary.get("n_cells") or 0)
    macro = summary.get("macro_acc")
    macro_s = f"{macro:.4f}" if isinstance(macro, float) and np.isfinite(macro) else "n/a"

    _draw_acc_bars(
        ax_domain,
        domain_df["domain"].astype(str).tolist(),
        domain_df["acc"].to_numpy(dtype=float),
        domain_df["n"].to_numpy(dtype=float),
        title=(
            f"{run_name}  ·  {split} accuracy per domain"
            f"   (micro={micro:.4f}, macro={macro_s}, n={n_cells:,})"
        ),
        xlabel="Domain",
        micro_acc=micro,
        cmap_name="RdYlGn",
    )
    _draw_acc_bars(
        ax_type,
        type_df["cell_type"].astype(str).tolist(),
        type_df["acc"].to_numpy(dtype=float),
        type_df["n"].to_numpy(dtype=float),
        title=f"{run_name}  ·  {split} domain-prediction accuracy per cell type",
        xlabel="Cell type",
        micro_acc=micro,
        cmap_name="RdYlGn",
    )

    written: List[Path] = []
    for suffix in (".png", ".pdf"):
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    out_dir = resolve_repo_path(args.output_dir) if args.output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = (
        resolve_repo_path(args.config) if args.config else run_dir / CONFIG_NAME
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    ckpt_path = resolve_repo_path(args.ckpt) if args.ckpt else find_best_checkpoint(run_dir)

    cfg = load_config(config_path)
    cfg = skip_unused_validation(cfg, args.split)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(args.device or cfg.get("device", "cuda")))
    run_name = str(cfg.get("run_name", run_dir.name))

    print(f"[ct_test_plot] run_dir={run_dir}")
    print(f"[ct_test_plot] config={config_path}")
    print(f"[ct_test_plot] ckpt={ckpt_path}")
    print(f"[ct_test_plot] device={device}  split={args.split}")

    ckpt = load_checkpoint(ckpt_path)
    ckpt_epoch = ckpt.get("epoch")
    print(f"[ct_test_plot] loaded checkpoint epoch={ckpt_epoch}", flush=True)

    dm_cfg = build_datamodule_cfg(cfg)
    print("[ct_test_plot] Loading data …", flush=True)
    datamodule = DataModule(dm_cfg)

    acfg = cfg.get("aux_inputs") or {}
    use_type = bool(acfg.get("cell_type", False))
    use_cme = bool(acfg.get("cme", False))

    # Type lookup must happen before domain labels overwrite cell_class.
    type_tables, num_cell_types, type_names = build_type_lookup(datamodule)
    print(
        f"[ct_test_plot] cell-type vocab T={num_cell_types}  "
        f"sample={type_names[:6]}",
        flush=True,
    )

    num_classes, class_names, decoder = apply_domain_labels(datamodule, cfg)
    ckpt_names = list(ckpt.get("class_names") or [])
    if ckpt_names and ckpt_names != class_names:
        print(
            "[ct_test_plot] warning: checkpoint class_names differ from "
            "config/data vocab; using data vocab for metrics",
            flush=True,
        )

    cme_tables: Dict[str, torch.Tensor] = {}
    cme_dim = 0
    if use_cme:
        cme_tables, cme_dim = build_cme_lookup(datamodule, cfg)

    infos = Infos(datamodule, dm_cfg)
    gene_dim = int(ckpt.get("gene_dim") or infos.num_genes)
    infos.num_cell_class = num_classes
    infos.cell_class_decoder = decoder
    print(
        f"[ct_test_plot] genes={gene_dim}  domain_classes={num_classes}  "
        f"type_input={use_type}  cme_input={use_cme} (dim={cme_dim})",
        flush=True,
    )

    model = build_model(
        cfg,
        gene_dim=gene_dim,
        num_classes=num_classes,
        num_cell_types=num_cell_types if use_type else 0,
        cme_input_dim=cme_dim,
    ).to(device)
    weight_src = load_weights(model, ckpt)
    print(f"[ct_test_plot] loaded weights from {weight_src}", flush=True)

    if args.split == "validation":
        loader = datamodule.validation_dataloader()
        type_table = type_tables.get("validation")
        cme_table = cme_tables.get("validation")
    else:
        loader = datamodule.test_dataloader()
        type_table = type_tables.get("test")
        cme_table = cme_tables.get("test")
    if loader is None:
        raise RuntimeError(f"No {args.split} dataloader")
    if type_table is None:
        raise RuntimeError(
            f"No cell-type lookup for split={args.split}; cannot score per type"
        )

    domain_df, type_df, summary = evaluate_split(
        model,
        loader,
        device,
        num_domains=num_classes,
        domain_names=class_names,
        type_names=type_names,
        type_table=type_table,
        cme_table=cme_table if use_cme else None,
    )

    domain_csv = out_dir / "test_acc_per_domain.csv"
    type_csv = out_dir / "test_acc_per_cell_type.csv"
    domain_df.to_csv(domain_csv, index=False)
    type_df.to_csv(type_csv, index=False)

    fig_paths = plot_accuracy_histograms(
        domain_df,
        type_df,
        summary,
        out_dir / "test_acc_histograms",
        split=args.split,
        run_name=run_name,
    )

    print(
        f"[ct_test_plot] {args.split} micro_acc={summary['micro_acc']:.4f}  "
        f"macro_acc={summary['macro_acc']:.4f}  n_cells={int(summary['n_cells']):,}",
        flush=True,
    )
    print(f"[ct_test_plot] wrote {domain_csv}")
    print(f"[ct_test_plot] wrote {type_csv}")
    for path in fig_paths:
        print(f"[ct_test_plot] wrote {path}")


if __name__ == "__main__":
    main()
