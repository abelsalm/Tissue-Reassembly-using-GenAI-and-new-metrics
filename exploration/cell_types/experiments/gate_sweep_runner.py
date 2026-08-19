#!/usr/bin/env python3
"""Launch a gated-CME experiment from a loss/soft_cme/presence override dict.

Writes a unique config under experiments/configs/, launches ct_train.py,
appends a one-line result to experiments/GATE_SWEEP.jsonl when done.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CT = ROOT / "exploration" / "cell_types"
BASE_CFG = CT / "ct_config.json"
CFG_DIR = CT / "experiments" / "configs"
LOG_ROOT = CT / "logs"
JSONL = CT / "experiments" / "GATE_SWEEP.jsonl"
SUMMARIZE = CT / "experiments" / "summarize_run.py"


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def launch(exp_id: str, gpu: int, overrides: dict, dry_run: bool = False) -> int:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CFG.read_text())
    deep_update(cfg, overrides)
    cfg["run_name"] = exp_id
    # Compact logging (less token-heavy train.log / wandb).
    w = cfg.setdefault("wandb", {})
    w["log_every_steps"] = False
    w["log_per_class_acc"] = False
    w.setdefault("tags", [])
    if "gate_sweep" not in w["tags"]:
        w["tags"] = list(w["tags"]) + ["gate_sweep"]
    cfg_path = CFG_DIR / f"{exp_id}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    meta = {
        "exp_id": exp_id,
        "gpu": gpu,
        "config": str(cfg_path),
        "overrides": overrides,
        "t0": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (CFG_DIR / f"{exp_id}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "/home/abel/miniconda3/envs/LUNA/bin/python",
        "-u",
        str(CT / "ct_train.py"),
        "--config",
        str(cfg_path),
    ]
    print(f"[runner] {exp_id} GPU={gpu} cfg={cfg_path}", flush=True)
    if dry_run:
        print(" ".join(cmd), flush=True)
        return 0

    log_out = CFG_DIR / f"{exp_id}.launch.log"
    with log_out.open("w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    # Find newest log stamp for this run_name.
    run_logs = LOG_ROOT / exp_id
    stamp = None
    if run_logs.is_dir():
        stamps = sorted([p for p in run_logs.iterdir() if p.is_dir()], key=lambda p: p.name)
        if stamps:
            stamp = stamps[-1]

    summary = {"exp_id": exp_id, "gpu": gpu, "exit": proc.returncode, "t1": time.strftime("%Y-%m-%d %H:%M:%S")}
    if stamp is not None:
        summary["log_dir"] = str(stamp)
        try:
            out = subprocess.check_output(
                [sys.executable, str(SUMMARIZE), str(stamp)],
                text=True,
            )
            summary["summary"] = json.loads(out)
        except Exception as e:  # noqa: BLE001
            summary["summary_error"] = str(e)

    with JSONL.open("a") as f:
        f.write(json.dumps(summary, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--overrides", type=str, required=True, help="JSON object")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    overrides = json.loads(args.overrides)
    raise SystemExit(launch(args.exp_id, args.gpu, overrides, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
