#!/usr/bin/env python3
"""Post-prediction optimization with the *training* combined loss.

Uses ``metrics.train_loss.CombinedTrainLoss`` built entirely from
``configs/train/default.yaml`` (vanilla / transcriptome / CH / PCA /
directional / MMD weights and hyperparameters). A second JSON config
(``train_loss_opt_cfg.json``) only controls:

  * which CSV slices to run
  * Adam vs Metropolis–Hastings optimizer knobs
  * extra regularizers (box / anchor) not part of training
  * I/O / device / plotting

Gene features come from the experiment YAML (same path as
``script_test_mmd_transcriptomics_from_csv.py``).

Usage::

    conda activate LUNA
    python post_pred_optimization/training_loss_post_test.py
    python post_pred_optimization/training_loss_post_test.py \\
        --opt-cfg post_pred_optimization/train_loss_opt_cfg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.train_loss import CombinedTrainLoss  # noqa: E402
from metrics.train_mmds import COORD_HI, COORD_LO, box_penalty  # noqa: E402
from post_pred_optimization.estimate_then_optimize import (  # noqa: E402
    pick_device,
    plot_scatter_three_panel,
    resolve_slices,
)
from script_test_mmd_transcriptomics_from_csv import (  # noqa: E402
    load_gene_table,
    load_slice_holders,
)
from utils.data.dataholder import DataHolder  # noqa: E402

DEFAULT_OPT_CFG = Path(__file__).resolve().parent / "train_loss_opt_cfg.json"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def make_pred_holder(positions_n2: torch.Tensor, template: DataHolder) -> DataHolder:
    """Build a pred ``DataHolder`` with updated coords (no re-centering)."""
    return DataHolder(
        positions=positions_n2.unsqueeze(0),
        node_features=template.node_features,
        diffusion_time=None,
        cell_ID=template.cell_ID,
        cell_class=template.cell_class,
        node_mask=template.node_mask,
    )


def train_loss_extras(
    x: torch.Tensor,
    x0: torch.Tensor,
    *,
    anchor_weight: float,
    box_penalty_weight: float,
) -> torch.Tensor:
    extra = x.new_zeros(())
    if anchor_weight > 0:
        extra = extra + float(anchor_weight) * (x - x0).pow(2).mean()
    if box_penalty_weight > 0:
        extra = extra + float(box_penalty_weight) * box_penalty(x)
    return extra


def evaluate_combined(
    loss_mod: CombinedTrainLoss,
    x: torch.Tensor,
    x0: torch.Tensor,
    pred_template: DataHolder,
    true_holder: DataHolder,
    *,
    anchor_weight: float,
    box_penalty_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_h = make_pred_holder(x, pred_template)
    total, _ = loss_mod(pred_h, true_holder, train_stage=False, log=False)
    extra = train_loss_extras(
        x, x0, anchor_weight=anchor_weight, box_penalty_weight=box_penalty_weight
    )
    total = total + extra
    parts = {
        "total": float(total.detach().item()),
        "train_combined": float((total - extra).detach().item()),
        "extra": float(extra.detach().item()),
        "vanilla": float(loss_mod._last_vanilla),
        "transcriptome": float(loss_mod._last_transcriptome_multi_radius),
        "ch": float(loss_mod._last_ch),
        "pca": float(loss_mod._last_pca),
        "directional": float(loss_mod._last_directional),
        "mmd": float(loss_mod._last_mmd),
    }
    return total, parts


def _fmt_part(v: float) -> str:
    return "n/a" if v < 0 else f"{v:.6g}"


def optimize_adam(
    x0: torch.Tensor,
    pred_template: DataHolder,
    true_holder: DataHolder,
    loss_mod: CombinedTrainLoss,
    opt_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, List[float]]:
    x = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam(
        [x],
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("adam_betas", [0.9, 0.999])),
    )
    steps = int(opt_cfg["steps"])
    log_every = int(opt_cfg.get("log_every", 10))
    history: List[float] = []

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total, parts = evaluate_combined(
            loss_mod,
            x,
            x0,
            pred_template,
            true_holder,
            anchor_weight=float(opt_cfg.get("anchor_weight", 0.0)),
            box_penalty_weight=float(opt_cfg.get("box_penalty_weight", 0.0)),
        )
        total.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(COORD_LO, COORD_HI)
        history.append(parts["total"])
        if step == 1 or step == steps or step % max(1, log_every) == 0:
            print(
                f"  [adam] step {step:4d}/{steps}  total={parts['total']:.6g}  "
                f"train={parts['train_combined']:.6g}  "
                f"van={_fmt_part(parts['vanilla'])}  "
                f"tx={_fmt_part(parts['transcriptome'])}  "
                f"mmd={_fmt_part(parts['mmd'])}"
            )
    return x.detach(), history


def optimize_mh(
    x0: torch.Tensor,
    pred_template: DataHolder,
    true_holder: DataHolder,
    loss_mod: CombinedTrainLoss,
    opt_cfg: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[float]]:
    x = x0.detach().clone()
    n_points = int(x.shape[0])
    k_min = max(1, int(opt_cfg.get("mh_n_points_min", 1)))
    k_max = min(n_points, max(k_min, int(opt_cfg.get("mh_n_points_max", 16))))
    sigma = float(opt_cfg.get("mh_move_sigma", 0.01))
    steps = int(opt_cfg["steps"])
    log_every = int(opt_cfg.get("log_every", 10))
    aw = float(opt_cfg.get("anchor_weight", 0.0))
    bw = float(opt_cfg.get("box_penalty_weight", 0.0))

    with torch.no_grad():
        _, parts = evaluate_combined(
            loss_mod, x, x0, pred_template, true_holder,
            anchor_weight=aw, box_penalty_weight=bw,
        )
    cur_loss = parts["total"]
    cur_parts = parts
    history: List[float] = [cur_loss]
    n_accept = 0
    print(
        f"  [mh] init total={cur_loss:.6g}  train={parts['train_combined']:.6g}  "
        f"move k∈[{k_min},{k_max}]  σ={sigma:g}"
    )

    for step in range(1, steps + 1):
        k = int(torch.randint(k_min, k_max + 1, (1,), generator=generator).item())
        move_idx = torch.randperm(n_points, generator=generator)[:k].to(x.device)
        noise = (
            torch.randn((k, x.shape[1]), dtype=x.dtype, generator=generator).to(x.device)
            * sigma
        )
        proposal = x.clone()
        proposal[move_idx] = (proposal[move_idx] + noise).clamp(COORD_LO, COORD_HI)

        with torch.no_grad():
            _, prop_parts = evaluate_combined(
                loss_mod, proposal, x0, pred_template, true_holder,
                anchor_weight=aw, box_penalty_weight=bw,
            )
        prop_loss = prop_parts["total"]
        if prop_loss < cur_loss:
            x = proposal
            cur_loss = prop_loss
            cur_parts = prop_parts
            n_accept += 1

        history.append(cur_loss)
        if step == 1 or step == steps or step % max(1, log_every) == 0:
            print(
                f"  [mh] step {step:4d}/{steps}  total={cur_loss:.6g}  "
                f"train={cur_parts['train_combined']:.6g}  "
                f"van={_fmt_part(cur_parts['vanilla'])}  "
                f"tx={_fmt_part(cur_parts['transcriptome'])}  "
                f"accept={n_accept / step:.2%}  last_k={k}"
            )
    return x, history


def optimize_slice(
    x0: torch.Tensor,
    pred_template: DataHolder,
    true_holder: DataHolder,
    loss_mod: CombinedTrainLoss,
    opt_cfg: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[float]]:
    mode = str(opt_cfg.get("optimizer", "adam")).strip().lower()
    if mode in ("metropolis_hastings", "mh", "metropolis"):
        return optimize_mh(
            x0, pred_template, true_holder, loss_mod, opt_cfg, generator=generator
        )
    if mode != "adam":
        raise ValueError(f"Unknown optimizer '{mode}'. Use 'adam' or 'metropolis_hastings'.")
    return optimize_adam(x0, pred_template, true_holder, loss_mod, opt_cfg)


def print_active_train_terms(loss_mod: CombinedTrainLoss) -> None:
    print("Active CombinedTrainLoss terms (from train config):")
    print(f"  vanilla_weight={loss_mod.vanilla_weight}  active={loss_mod.vanilla is not None}")
    print(
        f"  transcriptome_multi_radius_weight={loss_mod.transcriptome_multi_radius_weight}  "
        f"active={loss_mod.neighborhood is not None}"
    )
    print(f"  ch_weight={loss_mod.ch_weight}  active={loss_mod.ch is not None}")
    print(f"  pca_weight={loss_mod.pca_weight}  active={loss_mod.pca is not None}")
    print(
        f"  directional_weight={loss_mod.directional_weight}  "
        f"active={loss_mod.directional is not None}"
    )
    print(f"  mmd_weight={loss_mod.mmd_weight}  active={loss_mod.mmd is not None}")
    print(
        f"  other_trigger={loss_mod.other_trigger}  "
        f"current_epoch={loss_mod._current_epoch}  "
        f"other_active={loss_mod._other_losses_active()}"
    )


def run_slice(
    stem: str,
    true_path: Path,
    pred_path: Path,
    *,
    gene_table: pd.DataFrame,
    loss_mod: CombinedTrainLoss,
    opt_cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    print(f"\n=== {stem} ===")
    pred_h, true_h, n_cells = load_slice_holders(
        pred_path, true_path, gene_table, device
    )
    # Positions after load_slice_holders are already mask-centered.
    x0 = pred_h.positions[0].detach().clone()
    true_df = pd.read_csv(true_path, index_col=0)
    pred_df = pd.read_csv(pred_path, index_col=0)
    shared = true_df.index.intersection(pred_df.index)
    true_df = true_df.loc[shared]
    pred_df = pred_df.loc[shared]

    # Align dataframes to the centered coords used for optimization (for plots).
    raw_plot_df = pred_df.copy()
    raw_plot_df["coord_X"] = x0[:, 0].cpu().numpy()
    raw_plot_df["coord_Y"] = x0[:, 1].cpu().numpy()
    gt_plot_df = true_df.copy()
    gt_xy = true_h.positions[0].detach().cpu().numpy()
    gt_plot_df["coord_X"] = gt_xy[:, 0]
    gt_plot_df["coord_Y"] = gt_xy[:, 1]

    print(f"  n_cells={n_cells}")
    with torch.no_grad():
        _, init_parts = evaluate_combined(
            loss_mod,
            x0,
            x0,
            pred_h,
            true_h,
            anchor_weight=float(opt_cfg.get("anchor_weight", 0.0)),
            box_penalty_weight=float(opt_cfg.get("box_penalty_weight", 0.0)),
        )
    print(
        f"  init total={init_parts['total']:.6g}  "
        f"train={init_parts['train_combined']:.6g}  "
        f"vanilla={_fmt_part(init_parts['vanilla'])}  "
        f"tx={_fmt_part(init_parts['transcriptome'])}  "
        f"mmd={_fmt_part(init_parts['mmd'])}"
    )

    gen = torch.Generator()
    gen.manual_seed(int(opt_cfg.get("seed", 0)))
    x_opt, history = optimize_slice(
        x0, pred_h, true_h, loss_mod, opt_cfg, generator=gen
    )

    out_dir = _resolve_path(opt_cfg["output_dir"]) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_df = pred_df.copy()
    opt_np = x_opt.cpu().numpy()
    opt_df["coord_X"] = opt_np[:, 0]
    opt_df["coord_Y"] = opt_np[:, 1]
    opt_df.to_csv(out_dir / f"{stem}_metadata_pred_optimized_train_loss.csv")
    np.save(out_dir / "loss_history.npy", np.asarray(history, dtype=np.float64))

    with torch.no_grad():
        _, final_parts = evaluate_combined(
            loss_mod,
            x_opt,
            x0,
            pred_h,
            true_h,
            anchor_weight=float(opt_cfg.get("anchor_weight", 0.0)),
            box_penalty_weight=float(opt_cfg.get("box_penalty_weight", 0.0)),
        )
    print(
        f"  final total={final_parts['total']:.6g}  "
        f"train={final_parts['train_combined']:.6g}  "
        f"vanilla={_fmt_part(final_parts['vanilla'])}  "
        f"tx={_fmt_part(final_parts['transcriptome'])}  "
        f"mmd={_fmt_part(final_parts['mmd'])}"
    )
    (out_dir / "loss_summary.json").write_text(
        json.dumps({"init": init_parts, "final": final_parts}, indent=2)
    )

    uniques = sorted(true_df["cell_class"].astype(str).unique().tolist())
    plot_scatter_three_panel(
        gt_plot_df,
        raw_plot_df,
        opt_df,
        uniques,
        out_dir / f"{stem}_class_scatter_gt_raw_opt.png",
        dpi=int(opt_cfg.get("dpi", 200)),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opt-cfg",
        type=Path,
        default=DEFAULT_OPT_CFG,
        help="Optimization / I/O JSON (default: train_loss_opt_cfg.json)",
    )
    args = parser.parse_args(argv)
    opt_cfg = load_json(args.opt_cfg)

    train_cfg_path = _resolve_path(
        opt_cfg.get("train_config", "configs/train/default.yaml")
    )
    exp_cfg_path = _resolve_path(
        opt_cfg.get(
            "experiment_config",
            "configs/experiment/MERFISH_small_transcripts.yaml",
        )
    )
    train_cfg = OmegaConf.load(train_cfg_path)
    exp_cfg = OmegaConf.load(exp_cfg_path)

    device = pick_device(str(opt_cfg.get("device", "cuda")))
    torch.manual_seed(int(opt_cfg.get("seed", 0)))
    np.random.seed(int(opt_cfg.get("seed", 0)))

    csv_dir = _resolve_path(opt_cfg["csv_dir"])
    pairs = resolve_slices(csv_dir, opt_cfg.get("slice", None))

    gene_start = int(exp_cfg.dataset.gene_columns_start)
    gene_end = int(exp_cfg.dataset.gene_columns_end)
    data_paths: List[Path] = []
    for key in ("test_data_path", "train_data_path", "validation_data_path"):
        p = Path(str(exp_cfg.dataset[key]))
        if p.is_file() and p not in data_paths:
            data_paths.append(p)
    if not data_paths:
        raise FileNotFoundError("No MERFISH gene CSVs found from experiment config.")

    print(f"Opt cfg:    {args.opt_cfg}")
    print(f"Train cfg:  {train_cfg_path}")
    print(f"Experiment: {exp_cfg_path}")
    print(f"CSV dir:    {csv_dir}")
    print(f"Device:     {device}")
    print(f"Slices:     {len(pairs)}")

    print("Loading gene feature table …")
    gene_table, gene_names = load_gene_table(data_paths, gene_start, gene_end)
    print(f"  {len(gene_table):,} cells, {len(gene_names)} genes")

    loss_mod = CombinedTrainLoss(train_cfg).to(device)
    loss_mod.set_current_epoch(int(opt_cfg.get("current_epoch", 0)))
    loss_mod.eval()
    print_active_train_terms(loss_mod)

    for stem, true_path, pred_path in pairs:
        # resolve_slices returns (stem, true, pred) in estimate_then_optimize
        # but list_slice_pairs order differs — check resolve_slices signature.
        run_slice(
            stem,
            true_path,
            pred_path,
            gene_table=gene_table,
            loss_mod=loss_mod,
            opt_cfg=opt_cfg,
            device=device,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
