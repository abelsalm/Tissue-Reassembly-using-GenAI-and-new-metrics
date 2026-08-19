#!/usr/bin/env python3
"""Compact one-line / JSONL summary of a finished (or mid) training run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def summarize(log_dir: Path) -> dict:
    hist_path = log_dir / "history.json"
    snap_path = log_dir / "ct_config.snapshot.json"
    if not hist_path.exists():
        return {"log_dir": str(log_dir), "status": "no_history"}
    hist = json.loads(hist_path.read_text())
    best = None
    last_val = None
    for h in hist:
        v = h.get("val")
        if not isinstance(v, dict) or "eval_loss" not in v:
            continue
        last_val = {
            "epoch": h["epoch"],
            "eval": float(v["eval_loss"]),
            "cme": float(v.get("cme_loss", float("nan"))),
            "cls": float(v.get("cls_loss", float("nan"))),
            "acc": float(v.get("acc", float("nan"))),
            "presence": float(v.get("presence_loss", float("nan")))
            if v.get("presence_loss") is not None
            else None,
        }
        if best is None or last_val["eval"] < best["eval"]:
            best = dict(last_val)
    loss = {}
    model = {}
    run_name = log_dir.parent.name
    if snap_path.exists():
        cfg = json.loads(snap_path.read_text())
        loss = cfg.get("loss") or {}
        model = cfg.get("model") or {}
        run_name = str(cfg.get("run_name", run_name))
    gate_keys = {
        k: loss.get(k)
        for k in (
            "presence_weight",
            "presence_target",
            "presence_saturation",
            "presence_pos_weight",
            "presence_target_power",
            "presence_target_eps",
            "presence_sparsity_weight",
            "cme_softce_source",
            "cme_gated_weight",
            "cme_divergence",
            "cme_weight",
            "cme_target_temperature",
            "cme_target_support_eps",
            "cme_class_balance",
            "cme_train_sigma",
            "bag_cme_weight",
            "aux_cme_weight",
            "cme_spearman_weight",
            "cme_entropy_match_weight",
        )
        if k in loss
    }
    return {
        "run": run_name,
        "stamp": log_dir.name,
        "epochs": len(hist),
        "best": best,
        "last_val": last_val,
        "gate": gate_keys,
        "presence_dim": (model.get("hidden_mlp_dims") or {}).get("presence"),
        "status": "ok" if best else "no_val",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=str)
    ap.add_argument("--line", action="store_true", help="one-line TSV")
    args = ap.parse_args()
    s = summarize(Path(args.log_dir))
    if args.line:
        b = s.get("best") or {}
        print(
            f"{s.get('run')}\t{s.get('stamp')}\t"
            f"cme={b.get('cme')}\teval={b.get('eval')}\t"
            f"cls={b.get('cls')}\tep={b.get('epoch')}\t"
            f"{json.dumps(s.get('gate') or {}, separators=(',', ':'))}"
        )
    else:
        json.dump(s, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
